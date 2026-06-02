"""Pure BlazEdit/VANTAGE prompt lookup proposer.

This module intentionally has no dependency on vLLM, torch, tokenizers, or GPU
runtime state.  It implements the exact token lookup primitive needed by the
vLLM integration layer: find the latest prior occurrence of the last ``n``
task-local generated tokens in searchable task context plus generated prefix,
then copy the following tokens as a deterministic draft.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence


TokenId = int
TokenRange = tuple[int, int]


@dataclass(frozen=True)
class PLDConfig:
    """Configuration for the pure PLD lookup rule.

    ``max_draft_length=128`` and ``match_length=10`` correspond to the
    BlazEdit/VANTAGE ``w128_n10`` setting.  Excluded ranges are half-open
    token spans in combined ``context_ids + generated_ids`` coordinates.
    """

    max_draft_length: int = 128
    match_length: int = 10
    min_match_length: int | None = None
    exclude_ranges: tuple[TokenRange, ...] = ()

    def __post_init__(self) -> None:
        if self.max_draft_length < 0:
            raise ValueError("max_draft_length must be non-negative")
        if self.match_length <= 0:
            raise ValueError("match_length must be positive")
        min_match = self.match_length if self.min_match_length is None else self.min_match_length
        if min_match <= 0:
            raise ValueError("min_match_length must be positive")
        if min_match > self.match_length:
            raise ValueError("min_match_length cannot exceed match_length")
        object.__setattr__(self, "min_match_length", min_match)
        object.__setattr__(self, "exclude_ranges", _normalize_ranges(self.exclude_ranges))


@dataclass(frozen=True)
class PLDProposal:
    """A deterministic PLD draft and its source coordinates."""

    tokens: list[TokenId]
    match_length: int
    source_start: int
    source_end: int
    follow_start: int
    follow_end: int
    query_start: int
    query_end: int
    source: str


@dataclass
class PLDStats:
    """Mutable counters for repeated proposer calls."""

    calls: int = 0
    hits: int = 0
    misses: int = 0
    tokens_proposed: int = 0
    prompt_hits: int = 0
    generated_hits: int = 0
    last_match_length: int | None = None
    last_source_start: int | None = None
    last_draft_length: int = 0

    def record(self, proposal: PLDProposal | None) -> None:
        self.calls += 1
        if proposal is None:
            self.misses += 1
            self.last_match_length = None
            self.last_source_start = None
            self.last_draft_length = 0
            return
        self.hits += 1
        self.tokens_proposed += len(proposal.tokens)
        self.last_match_length = proposal.match_length
        self.last_source_start = proposal.source_start
        self.last_draft_length = len(proposal.tokens)
        if proposal.source == "prompt":
            self.prompt_hits += 1
        elif proposal.source == "generated":
            self.generated_hits += 1


@dataclass
class PromptLookupProposer:
    """Dependency-free prompt lookup proposer.

    The search space is ``context_ids + generated_ids``.  Query tokens are the
    last ``match_length`` tokens of ``generated_ids`` by default.  If no exact
    ``match_length`` hit exists and ``min_match_length`` is smaller, the lookup
    tries shorter suffixes down to that minimum.
    """

    config: PLDConfig = field(default_factory=PLDConfig)
    stats: PLDStats = field(default_factory=PLDStats)

    def propose(
        self,
        *,
        context_ids: Sequence[TokenId] | None = None,
        generated_ids: Sequence[TokenId] | None = None,
        exclude_ranges: Iterable[TokenRange] = (),
    ) -> PLDProposal | None:
        context = [int(token) for token in (context_ids or ())]
        generated = [int(token) for token in (generated_ids or ())]
        proposal = self._lookup(context, generated, exclude_ranges)
        self.stats.record(proposal)
        return proposal

    def _lookup(
        self,
        context: list[TokenId],
        generated: list[TokenId],
        exclude_ranges: Iterable[TokenRange],
    ) -> PLDProposal | None:
        if self.config.max_draft_length == 0:
            return None
        combined = context + generated
        context_len = len(context)
        query_end = len(combined)
        if not generated or query_end == 0:
            return None

        excluded = _normalize_ranges((*self.config.exclude_ranges, *tuple(exclude_ranges)))
        max_match = min(self.config.match_length, len(generated))
        min_match = int(self.config.min_match_length or self.config.match_length)
        if max_match < min_match:
            return None

        for match_len in range(max_match, min_match - 1, -1):
            query_start = query_end - match_len
            if _overlaps_any(query_start, query_end, excluded):
                continue
            needle = combined[query_start:query_end]
            best_start: int | None = None
            # Latest prior, non-overlapping match wins.  Matches may come from
            # source/prompt context or already generated task-local prefix, but
            # never from the live query suffix itself.
            for start in range(0, query_start - match_len + 1):
                end = start + match_len
                if _overlaps_any(start, end, excluded):
                    continue
                if combined[start:end] == needle:
                    best_start = start
            if best_start is None:
                continue

            follow_start = best_start + match_len
            follow_end = min(follow_start + self.config.max_draft_length, query_start)
            if _overlaps_any(follow_start, follow_end, excluded):
                follow_end = _first_excluded_start(follow_start, follow_end, excluded)
            if follow_start >= follow_end:
                continue
            source = "prompt" if best_start < context_len else "generated"
            return PLDProposal(
                tokens=list(combined[follow_start:follow_end]),
                match_length=match_len,
                source_start=best_start,
                source_end=best_start + match_len,
                follow_start=follow_start,
                follow_end=follow_end,
                query_start=query_start,
                query_end=query_end,
                source=source,
            )
        return None


def lookup_w128_n10(
    context_ids: Sequence[TokenId] | None,
    generated_ids: Sequence[TokenId] | None,
    *,
    exclude_ranges: Iterable[TokenRange] = (),
) -> PLDProposal | None:
    """Convenience wrapper for the frozen ``w128_n10`` lookup rule."""

    return PromptLookupProposer(PLDConfig(max_draft_length=128, match_length=10)).propose(
        context_ids=context_ids,
        generated_ids=generated_ids,
        exclude_ranges=exclude_ranges,
    )


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
        else:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
    return tuple(merged)


def _overlaps_any(start: int, end: int, ranges: Sequence[TokenRange]) -> bool:
    if start >= end:
        return False
    return any(start < range_end and end > range_start for range_start, range_end in ranges)


def _first_excluded_start(start: int, end: int, ranges: Sequence[TokenRange]) -> int:
    first = end
    for range_start, range_end in ranges:
        if start < range_end and end > range_start:
            first = min(first, max(start, range_start))
    return first
