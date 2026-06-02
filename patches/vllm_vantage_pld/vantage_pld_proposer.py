"""Skeleton vLLM proposer for VANTAGE prompt lookup decoding.

Copy this file to ``vllm/v1/spec_decode/vantage_pld_proposer.py`` in a
patched vLLM tree. It intentionally depends only on Python/numpy-shaped inputs
so it can share the CPU proposal path used by vLLM's n-gram proposer.

The patched GPUModelRunner must pass request prefix metadata. The upstream
custom_class API on main only passes sampled_token_ids, num_tokens_no_spec,
token_ids_cpu, and slot_mappings, which is not enough for PLD equivalence.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


TokenRange = tuple[int, int]


@dataclass
class VantagePLDStats:
    calls: int = 0
    skipped_empty_sample: int = 0
    skipped_max_model_len: int = 0
    metadata_missing: int = 0
    hits: int = 0
    misses: int = 0
    tokens_proposed: int = 0
    proposal_cap: int = 0
    cap_truncations: int = 0
    prompt_hits: int = 0
    generated_hits: int = 0
    last_match_length: int | None = None
    last_source_start: int | None = None
    match_len_histogram: Counter[int] = field(default_factory=Counter)
    draft_len_histogram: Counter[int] = field(default_factory=Counter)

    def snapshot(self) -> dict[str, Any]:
        return {
            "pld_calls": self.calls,
            "pld_skipped_empty_sample": self.skipped_empty_sample,
            "pld_skipped_max_model_len": self.skipped_max_model_len,
            "pld_metadata_missing": self.metadata_missing,
            "pld_hits": self.hits,
            "pld_misses": self.misses,
            "pld_tokens_proposed": self.tokens_proposed,
            "pld_cap": self.proposal_cap,
            "pld_cap_truncations": self.cap_truncations,
            "pld_prompt_hits": self.prompt_hits,
            "pld_generated_hits": self.generated_hits,
            "pld_last_match_length": self.last_match_length,
            "pld_last_source_start": self.last_source_start,
            "pld_match_len_histogram": dict(self.match_len_histogram),
            "pld_draft_len_histogram": dict(self.draft_len_histogram),
        }


@dataclass(frozen=True)
class LookupResult:
    tokens: list[int]
    match_length: int
    source_start: int
    follow_start: int
    query_start: int
    source_region: str


class VantagePLDProposer:
    """No-model vLLM proposer for the frozen ``w128_n10`` PLD rule."""

    def __init__(self, vllm_config: Any):
        spec_config = vllm_config.speculative_config
        self.max_model_len = int(vllm_config.model_config.max_model_len)

        self.window_tokens = _int_config(
            spec_config, "pld_window_tokens", "VANTAGE_PLD_WINDOW_TOKENS", 128
        )
        self.match_tokens = _int_config(
            spec_config, "pld_match_tokens", "VANTAGE_PLD_MATCH_TOKENS", 10
        )
        self.min_match_tokens = _optional_int_config(
            spec_config, "pld_min_match_tokens", "VANTAGE_PLD_MIN_MATCH_TOKENS"
        )
        if self.min_match_tokens is None:
            self.min_match_tokens = self.match_tokens
        self.num_speculative_tokens = int(spec_config.num_speculative_tokens)
        self.proposal_cap = min(self.num_speculative_tokens, self.window_tokens)
        self.require_prompt_metadata = _bool_config(
            spec_config,
            "pld_require_prompt_metadata",
            "VANTAGE_PLD_REQUIRE_PROMPT_METADATA",
            True,
        )
        self.stats_enabled = _bool_config(
            spec_config, "pld_stats_enabled", "VANTAGE_PLD_STATS_ENABLED", True
        )
        self.trace_path = _str_config(
            spec_config, "pld_trace_path", "VANTAGE_PLD_TRACE_PATH"
        )
        self.stats = VantagePLDStats(proposal_cap=self.proposal_cap)

        if self.window_tokens <= 0:
            raise ValueError("pld_window_tokens must be positive")
        if self.match_tokens <= 0:
            raise ValueError("pld_match_tokens must be positive")
        if self.min_match_tokens <= 0:
            raise ValueError("pld_min_match_tokens must be positive")
        if self.min_match_tokens > self.match_tokens:
            raise ValueError("pld_min_match_tokens cannot exceed pld_match_tokens")

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: Any,
        token_ids_cpu: Any,
        *,
        num_prompt_tokens: Any | None = None,
        pld_context_start: Any | None = None,
        pld_context_end: Any | None = None,
        pld_exclude_ranges: Sequence[Sequence[TokenRange]] | None = None,
        slot_mappings: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> list[list[int]]:
        del slot_mappings
        drafts: list[list[int]] = []

        for i, sampled_ids in enumerate(sampled_token_ids):
            self.stats.calls += 1
            if not sampled_ids:
                self.stats.skipped_empty_sample += 1
                drafts.append([])
                continue

            num_tokens = int(num_tokens_no_spec[i])
            if num_tokens >= self.max_model_len:
                self.stats.skipped_max_model_len += 1
                drafts.append([])
                continue

            prompt_len = _optional_index(num_prompt_tokens, i)
            if prompt_len is None:
                self.stats.metadata_missing += 1
                drafts.append([])
                continue

            origin = _row_to_int_list(token_ids_cpu[i], num_tokens)
            context_start = _optional_index(pld_context_start, i)
            context_end = _optional_index(pld_context_end, i)
            if context_start is None:
                context_start = 0
            if context_end is None:
                context_end = prompt_len
            if self.require_prompt_metadata and prompt_len is None:
                self.stats.metadata_missing += 1
                drafts.append([])
                continue

            context_start = max(0, min(int(context_start), prompt_len))
            context_end = max(context_start, min(int(context_end), prompt_len))
            generated_ids = origin[prompt_len:num_tokens]
            context_ids = origin[context_start:context_end]
            exclude_ranges = _project_exclude_ranges(
                _ranges_at(pld_exclude_ranges, i),
                context_start=context_start,
                context_end=context_end,
                prompt_len=prompt_len,
                num_tokens=num_tokens,
                context_len=len(context_ids),
            )

            result = _lookup(
                context_ids=context_ids,
                generated_ids=generated_ids,
                max_draft_length=self.window_tokens,
                match_length=self.match_tokens,
                min_match_length=self.min_match_tokens,
                exclude_ranges=exclude_ranges,
            )
            if result is None:
                self.stats.misses += 1
                self.stats.last_match_length = None
                self.stats.last_source_start = None
                drafts.append([])
                continue

            draft = result.tokens[: self.proposal_cap]
            if len(result.tokens) > len(draft):
                self.stats.cap_truncations += 1
            self.stats.hits += 1
            self.stats.tokens_proposed += len(draft)
            self.stats.last_match_length = result.match_length
            self.stats.last_source_start = result.source_start
            self.stats.match_len_histogram[result.match_length] += 1
            self.stats.draft_len_histogram[len(draft)] += 1
            if result.source_region == "prompt":
                self.stats.prompt_hits += 1
            else:
                self.stats.generated_hits += 1
            drafts.append(draft)

            if self.trace_path:
                _append_trace(
                    self.trace_path,
                    {
                        "request_index": i,
                        "proposal_tokens": draft,
                        "proposal_match_len": result.match_length,
                        "proposal_source_start_token": result.source_start,
                        "proposal_follow_start_token": result.follow_start,
                        "proposal_query_start_token": result.query_start,
                        "proposal_source_region": result.source_region,
                    },
                )

        return drafts

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


def _lookup(
    *,
    context_ids: Sequence[int],
    generated_ids: Sequence[int],
    max_draft_length: int,
    match_length: int,
    min_match_length: int,
    exclude_ranges: Sequence[TokenRange],
) -> LookupResult | None:
    if max_draft_length <= 0 or not generated_ids:
        return None

    combined = [int(x) for x in context_ids] + [int(x) for x in generated_ids]
    context_len = len(context_ids)
    query_end = len(combined)
    max_match = min(match_length, len(generated_ids))
    if max_match < min_match_length:
        return None

    for match_len in range(max_match, min_match_length - 1, -1):
        query_start = query_end - match_len
        if _overlaps_any(query_start, query_end, exclude_ranges):
            continue
        needle = combined[query_start:query_end]
        best_start: int | None = None
        for start in range(0, query_start - match_len + 1):
            end = start + match_len
            if _overlaps_any(start, end, exclude_ranges):
                continue
            if combined[start:end] == needle:
                best_start = start
        if best_start is None:
            continue

        follow_start = best_start + match_len
        follow_end = min(follow_start + max_draft_length, query_start)
        if _overlaps_any(follow_start, follow_end, exclude_ranges):
            follow_end = _first_excluded_start(follow_start, follow_end, exclude_ranges)
        if follow_start >= follow_end:
            continue
        return LookupResult(
            tokens=list(combined[follow_start:follow_end]),
            match_length=match_len,
            source_start=best_start,
            follow_start=follow_start,
            query_start=query_start,
            source_region="prompt" if best_start < context_len else "generated",
        )
    return None


def _project_exclude_ranges(
    ranges: Iterable[TokenRange],
    *,
    context_start: int,
    context_end: int,
    prompt_len: int,
    num_tokens: int,
    context_len: int,
) -> tuple[TokenRange, ...]:
    projected: list[TokenRange] = []
    for start, end in _normalize_ranges(ranges):
        prompt_a = max(start, context_start)
        prompt_b = min(end, context_end)
        if prompt_a < prompt_b:
            projected.append((prompt_a - context_start, prompt_b - context_start))
        gen_a = max(start, prompt_len)
        gen_b = min(end, num_tokens)
        if gen_a < gen_b:
            projected.append(
                (context_len + gen_a - prompt_len, context_len + gen_b - prompt_len)
            )
    return _normalize_ranges(projected)


def _normalize_ranges(ranges: Iterable[TokenRange]) -> tuple[TokenRange, ...]:
    normalized: list[TokenRange] = []
    for start, end in ranges:
        start_i = int(start)
        end_i = int(end)
        if start_i < 0 or end_i < 0:
            raise ValueError("exclude ranges must be non-negative")
        if end_i < start_i:
            raise ValueError("exclude range end must be >= start")
        if start_i != end_i:
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


def _row_to_int_list(row: Any, length: int) -> list[int]:
    values = row[:length]
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(x) for x in values]


def _ranges_at(
    values: Sequence[Sequence[TokenRange]] | None, index: int
) -> Sequence[TokenRange]:
    if values is None or index >= len(values):
        return ()
    return values[index] or ()


def _optional_index(values: Any | None, index: int) -> int | None:
    if values is None:
        return None
    return int(values[index])


def _int_config(obj: Any, field_name: str, env_name: str, default: int) -> int:
    value = getattr(obj, field_name, None)
    if value is None:
        value = os.environ.get(env_name)
    return int(default if value in (None, "") else value)


def _optional_int_config(obj: Any, field_name: str, env_name: str) -> int | None:
    value = getattr(obj, field_name, None)
    if value is None:
        value = os.environ.get(env_name)
    return None if value in (None, "") else int(value)


def _bool_config(obj: Any, field_name: str, env_name: str, default: bool) -> bool:
    value = getattr(obj, field_name, None)
    if value is None:
        value = os.environ.get(env_name)
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _str_config(obj: Any, field_name: str, env_name: str) -> str | None:
    value = getattr(obj, field_name, None)
    if value is None:
        value = os.environ.get(env_name)
    return None if value in (None, "") else str(value)


def _append_trace(path: str, payload: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
