"""Pure VANTAGE PLD proposal lookup.

The default parameters implement the frozen vLLM-equivalence rule:
match the last 10 generated tokens, copy at most 128 following tokens from the
latest prior non-overlapping occurrence, and optionally apply an explicit vLLM
proposal cap.  The function is dependency-free so it can be used as the oracle
for vLLM adapters and trace-comparison tooling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence


TokenId = int
TokenRange = tuple[int, int]
TieBreak = Literal["latest", "earliest"]


@dataclass(frozen=True)
class PLDLookupResult:
    """A deterministic PLD draft and the combined-coordinate source metadata."""

    tokens: list[TokenId]
    match_n: int
    source_start: int
    source_end: int
    follow_start: int
    follow_end: int
    query_start: int
    query_end: int
    source: Literal["prompt", "generated"]
    capped: bool = False
    cap: int | None = None

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
            "proposal_source_region": self.source,
            "proposal_capped": self.capped,
            "proposal_cap": self.cap,
        }


def find_pld_proposal(
    prompt_ids: Sequence[TokenId] | None,
    generated_ids: Sequence[TokenId] | None,
    *,
    match_n: int = 10,
    max_draft_len: int = 128,
    cap: int | None = None,
    exclude_ranges: Iterable[TokenRange] = (),
    search_prompt: bool = True,
    search_generated: bool = True,
    tie_break: TieBreak = "latest",
) -> PLDLookupResult | None:
    """Find one exact PLD proposal.

    Coordinates are in the combined ``prompt_ids + generated_ids`` token space.
    ``exclude_ranges`` are half-open spans in that same space; excluded tokens
    cannot be used for the source match or copied into the draft.  ``cap`` is an
    additional effective proposal limit, typically vLLM's
    ``num_speculative_tokens``.  When present, the effective draft length is
    ``min(max_draft_len, cap)``.
    """

    if match_n <= 0:
        raise ValueError("match_n must be positive")
    if max_draft_len < 0:
        raise ValueError("max_draft_len must be non-negative")
    if cap is not None and cap < 0:
        raise ValueError("cap must be non-negative")
    if tie_break not in ("latest", "earliest"):
        raise ValueError("tie_break must be 'latest' or 'earliest'")
    if not search_prompt and not search_generated:
        return None

    prompt = [int(token) for token in (prompt_ids or ())]
    generated = [int(token) for token in (generated_ids or ())]
    if len(generated) < match_n:
        return None

    effective_max = min(max_draft_len, cap) if cap is not None else max_draft_len
    if effective_max <= 0:
        return None

    combined = prompt + generated
    prompt_len = len(prompt)
    query_end = len(combined)
    query_start = query_end - match_n
    excluded = _normalize_ranges(exclude_ranges)
    if _overlaps_any(query_start, query_end, excluded):
        return None

    needle = combined[query_start:query_end]
    candidate_starts = range(0, query_start - match_n + 1)
    if tie_break == "latest":
        candidate_starts = range(query_start - match_n, -1, -1)

    for source_start in candidate_starts:
        source_end = source_start + match_n
        source = "prompt" if source_start < prompt_len else "generated"
        if source == "prompt" and not search_prompt:
            continue
        if source == "generated" and not search_generated:
            continue
        if _overlaps_any(source_start, source_end, excluded):
            continue
        if combined[source_start:source_end] != needle:
            continue

        follow_start = source_end
        raw_follow_end = min(follow_start + max_draft_len, query_start)
        uncapped_len = raw_follow_end - follow_start
        cap_follow_end = min(raw_follow_end, follow_start + effective_max)
        follow_end = _truncate_before_excluded(follow_start, cap_follow_end, excluded)
        if follow_start >= follow_end:
            continue

        tokens = list(combined[follow_start:follow_end])
        capped = cap is not None and cap < uncapped_len and cap_follow_end <= follow_end
        return PLDLookupResult(
            tokens=tokens,
            match_n=match_n,
            source_start=source_start,
            source_end=source_end,
            follow_start=follow_start,
            follow_end=follow_end,
            query_start=query_start,
            query_end=query_end,
            source=source,
            capped=bool(capped),
            cap=cap,
        )

    return None


def _normalize_ranges(ranges: Iterable[TokenRange]) -> tuple[TokenRange, ...]:
    normalized: list[TokenRange] = []
    for start, end in ranges:
        start_i = int(start)
        end_i = int(end)
        if start_i < 0 or end_i < 0:
            raise ValueError("exclude ranges must be non-negative")
        if end_i < start_i:
            raise ValueError("exclude range end must be >= start")
        if start_i == end_i:
            continue
        normalized.append((start_i, end_i))
    normalized.sort()

    merged: list[TokenRange] = []
    for start, end in normalized:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        merged[-1] = (prev_start, max(prev_end, end))
    return tuple(merged)


def _overlaps_any(start: int, end: int, ranges: Sequence[TokenRange]) -> bool:
    if start >= end:
        return False
    return any(start < range_end and end > range_start for range_start, range_end in ranges)


def _truncate_before_excluded(start: int, end: int, ranges: Sequence[TokenRange]) -> int:
    if start >= end:
        return end
    truncated = end
    for range_start, range_end in ranges:
        if start < range_end and end > range_start:
            truncated = min(truncated, max(start, range_start))
    return truncated
