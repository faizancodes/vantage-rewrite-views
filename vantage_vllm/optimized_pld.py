"""Optimized fixed-n PLD lookup over one full-prefix token sequence.

This module deliberately does not depend on vLLM.  The public API accepts the
full prefix seen by a proposer, treats the last ``match_n`` tokens as the query,
searches prior non-overlapping positions, and copies the following continuation
up to ``max_draft_len`` and optional ``cap``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence


TokenId = int
TieBreak = Literal["latest", "earliest"]
Backend = Literal["python", "numba"]


@dataclass(frozen=True)
class OptimizedPLDResult:
    """A PLD draft and full-prefix-coordinate metadata."""

    tokens: list[TokenId]
    match_n: int
    source_start: int
    source_end: int
    follow_start: int
    follow_end: int
    query_start: int
    query_end: int
    capped: bool = False
    cap: int | None = None
    backend: Backend = "python"

    @property
    def draft_len(self) -> int:
        return len(self.tokens)

    def to_trace_row(self) -> dict[str, int | str | bool | list[int] | None]:
        return {
            "proposal_token_ids": list(self.tokens),
            "proposal_tokens": len(self.tokens),
            "proposal_match_len": self.match_n,
            "proposal_source_start_token": self.source_start,
            "proposal_source_end_token": self.source_end,
            "proposal_follow_start_token": self.follow_start,
            "proposal_follow_end_token": self.follow_end,
            "proposal_query_start_token": self.query_start,
            "proposal_query_end_token": self.query_end,
            "proposal_capped": self.capped,
            "proposal_cap": self.cap,
            "proposal_backend": self.backend,
        }


def find_full_prefix_pld_proposal(
    prefix_tokens: Sequence[TokenId] | Any,
    *,
    match_n: int = 10,
    max_draft_len: int = 128,
    cap: int | None = None,
    tie_break: TieBreak = "latest",
    prefer_numba: bool = True,
) -> OptimizedPLDResult | None:
    """Return the deterministic fixed-n PLD proposal for a full prefix.

    The function is equivalent to calling the pure oracle with
    ``prompt_ids=prefix[:-match_n]`` and ``generated_ids=prefix[-match_n:]``.
    That split makes the whole pre-query prefix searchable and keeps the query
    equal to the last ``match_n`` prefix tokens.
    """

    if match_n <= 0:
        raise ValueError("match_n must be positive")
    if max_draft_len < 0:
        raise ValueError("max_draft_len must be non-negative")
    if cap is not None and cap < 0:
        raise ValueError("cap must be non-negative")
    if tie_break not in ("latest", "earliest"):
        raise ValueError("tie_break must be 'latest' or 'earliest'")

    prefix_len = _len(prefix_tokens)
    if prefix_len < match_n:
        return None

    effective_max = min(max_draft_len, cap) if cap is not None else max_draft_len
    if effective_max <= 0:
        return None

    query_end = prefix_len
    query_start = query_end - match_n
    # A source match ending exactly at the query start has no continuation to
    # propose. The pure oracle skips those matches and continues searching, so
    # the optimized search only considers matches with at least one following
    # token before the query.
    max_source_start = query_start - match_n - 1
    if max_source_start < 0:
        return None

    source_start, backend = _find_source_start(
        prefix_tokens,
        prefix_len=prefix_len,
        match_n=match_n,
        tie_break=tie_break,
        prefer_numba=prefer_numba,
    )
    if source_start < 0:
        return None

    source_end = source_start + match_n
    follow_start = source_end
    raw_follow_end = min(follow_start + max_draft_len, query_start)
    uncapped_len = raw_follow_end - follow_start
    follow_end = min(raw_follow_end, follow_start + effective_max)
    if follow_start >= follow_end:
        return None

    return OptimizedPLDResult(
        tokens=_slice_tokens(prefix_tokens, follow_start, follow_end),
        match_n=match_n,
        source_start=source_start,
        source_end=source_end,
        follow_start=follow_start,
        follow_end=follow_end,
        query_start=query_start,
        query_end=query_end,
        capped=bool(cap is not None and cap < uncapped_len),
        cap=cap,
        backend=backend,
    )


def numba_available() -> bool:
    """Return whether the optional Numba backend can be initialized."""

    return _get_numba_kernel() is not None


def _find_source_start(
    prefix_tokens: Sequence[TokenId] | Any,
    *,
    prefix_len: int,
    match_n: int,
    tie_break: TieBreak,
    prefer_numba: bool,
) -> tuple[int, Backend]:
    if prefer_numba:
        kernel = _get_numba_kernel()
        if kernel is not None:
            values = _as_numpy_int64(prefix_tokens)
            if values is not None:
                return int(kernel(values, match_n, tie_break == "latest")), "numba"
    return _find_source_start_python(prefix_tokens, prefix_len, match_n, tie_break), "python"


def _find_source_start_python(
    prefix_tokens: Sequence[TokenId] | Any,
    prefix_len: int,
    match_n: int,
    tie_break: TieBreak,
) -> int:
    query_start = prefix_len - match_n
    max_source_start = query_start - match_n - 1
    if tie_break == "latest":
        starts = range(max_source_start, -1, -1)
    else:
        starts = range(0, max_source_start + 1)

    for source_start in starts:
        matched = True
        for offset in range(match_n):
            if int(prefix_tokens[source_start + offset]) != int(
                prefix_tokens[query_start + offset]
            ):
                matched = False
                break
        if matched:
            return source_start
    return -1


_NUMBA_KERNEL: Any | None = None
_NUMBA_INITIALIZED = False


def _get_numba_kernel() -> Any | None:
    global _NUMBA_INITIALIZED, _NUMBA_KERNEL
    if _NUMBA_INITIALIZED:
        return _NUMBA_KERNEL
    _NUMBA_INITIALIZED = True
    try:
        from numba import njit
    except Exception:
        _NUMBA_KERNEL = None
        return None

    @njit
    def _kernel(values: Any, match_n: int, latest: bool) -> int:
        prefix_len = values.shape[0]
        query_start = prefix_len - match_n
        max_source_start = query_start - match_n - 1
        if max_source_start < 0:
            return -1

        if latest:
            source_start = max_source_start
            while source_start >= 0:
                matched = True
                for offset in range(match_n):
                    if values[source_start + offset] != values[query_start + offset]:
                        matched = False
                        break
                if matched:
                    return source_start
                source_start -= 1
            return -1

        source_start = 0
        while source_start <= max_source_start:
            matched = True
            for offset in range(match_n):
                if values[source_start + offset] != values[query_start + offset]:
                    matched = False
                    break
            if matched:
                return source_start
            source_start += 1
        return -1

    _NUMBA_KERNEL = _kernel
    return _NUMBA_KERNEL


def _as_numpy_int64(value: Sequence[TokenId] | Any) -> Any | None:
    try:
        import numpy as np
    except Exception:
        return None
    try:
        array = np.asarray(value, dtype=np.int64)
    except Exception:
        return None
    if array.ndim != 1:
        return None
    return array


def _len(value: Sequence[TokenId] | Any) -> int:
    try:
        return len(value)
    except TypeError:
        shape = getattr(value, "shape", None)
        if shape:
            return int(shape[0])
        raise


def _slice_tokens(value: Sequence[TokenId] | Any, start: int, end: int) -> list[int]:
    sliced = value[start:end]
    if hasattr(sliced, "tolist"):
        return [int(token) for token in sliced.tolist()]
    return [int(token) for token in sliced]
