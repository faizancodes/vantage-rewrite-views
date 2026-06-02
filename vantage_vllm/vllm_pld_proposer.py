"""Lightweight vLLM adapter for the pure PLD lookup.

This class intentionally avoids importing vLLM.  It mirrors the CPU no-model
proposer shape used by vLLM's n-gram/custom proposer path closely enough to
unit-test the boundary handling with fake inputs:

``propose(sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings=None)``

Real PLD equivalence requires request-local prompt length metadata.  If that
metadata is absent, the adapter returns no draft for that sequence and records a
``metadata_missing`` counter rather than guessing boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .pld_lookup import PLDLookupResult, TokenRange, find_pld_proposal


@dataclass
class VllmPLDStats:
    calls: int = 0
    sequences: int = 0
    hits: int = 0
    misses: int = 0
    tokens_proposed: int = 0
    prompt_hits: int = 0
    generated_hits: int = 0
    cap_truncations: int = 0
    metadata_missing: int = 0
    skipped_empty_sample: int = 0
    skipped_max_model_len: int = 0
    last_match_n: int | None = None
    last_source_start: int | None = None
    last_draft_len: int = 0
    match_len_histogram: dict[int, int] = field(default_factory=dict)
    draft_len_histogram: dict[int, int] = field(default_factory=dict)

    def record_hit(self, proposal: PLDLookupResult) -> None:
        self.hits += 1
        self.tokens_proposed += len(proposal.tokens)
        self.last_match_n = proposal.match_n
        self.last_source_start = proposal.source_start
        self.last_draft_len = len(proposal.tokens)
        self.match_len_histogram[proposal.match_n] = (
            self.match_len_histogram.get(proposal.match_n, 0) + 1
        )
        self.draft_len_histogram[len(proposal.tokens)] = (
            self.draft_len_histogram.get(len(proposal.tokens), 0) + 1
        )
        if proposal.source == "prompt":
            self.prompt_hits += 1
        elif proposal.source == "generated":
            self.generated_hits += 1
        if proposal.capped:
            self.cap_truncations += 1

    def record_miss(self) -> None:
        self.misses += 1
        self.last_match_n = None
        self.last_source_start = None
        self.last_draft_len = 0


@dataclass(frozen=True)
class RequestPLDMetadata:
    prompt_len: int
    exclude_ranges: tuple[TokenRange, ...] = ()
    search_prompt: bool = True
    search_generated: bool = True


class VantageVllmPLDProposer:
    """No-model PLD proposer adapter for vLLM-style custom proposer calls."""

    def __init__(
        self,
        vllm_config: Any | None = None,
        *,
        match_n: int = 10,
        max_draft_len: int = 128,
        cap: int | None = None,
        search_prompt: bool = True,
        search_generated: bool = True,
        tie_break: str = "latest",
        require_metadata: bool = True,
    ) -> None:
        self.vllm_config = vllm_config
        self.match_n = int(_config_value(vllm_config, "vantage_match_tokens", match_n))
        self.max_draft_len = int(
            _config_value(vllm_config, "vantage_window_tokens", max_draft_len)
        )
        self.cap = _int_or_none(
            cap
            if cap is not None
            else _config_value(vllm_config, "num_speculative_tokens", None)
        )
        self.search_prompt = bool(search_prompt)
        self.search_generated = bool(search_generated)
        self.tie_break = str(tie_break)
        self.require_metadata = bool(require_metadata)
        self.max_model_len = _int_or_none(_config_value(vllm_config, "max_model_len", None))
        self.stats = VllmPLDStats()

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        return None

    def propose(
        self,
        sampled_token_ids: Any,
        num_tokens_no_spec: Any,
        token_ids_cpu: Any,
        slot_mappings: Any | None = None,
        *,
        request_metadata: Any | None = None,
        prompt_lens: Sequence[int] | None = None,
        prompt_lengths: Sequence[int] | None = None,
        prompt_token_counts: Sequence[int] | None = None,
        exclude_ranges: Sequence[Iterable[TokenRange]] | None = None,
        **kwargs: Any,
    ) -> list[list[int]]:
        del kwargs
        self.stats.calls += 1
        n_sequences = _infer_num_sequences(sampled_token_ids, num_tokens_no_spec, token_ids_cpu)
        self.stats.sequences += n_sequences

        drafts: list[list[int]] = []
        for index in range(n_sequences):
            sample = _sequence_at(sampled_token_ids, index)
            if sample is not None and len(sample) == 0:
                self.stats.skipped_empty_sample += 1
                self.stats.record_miss()
                drafts.append([])
                continue

            history_len = _int_at(num_tokens_no_spec, index)
            history = _history_row(token_ids_cpu, index, history_len)
            if self.max_model_len is not None and history_len >= self.max_model_len:
                self.stats.skipped_max_model_len += 1
                self.stats.record_miss()
                drafts.append([])
                continue

            metadata = self._metadata_for_index(
                index,
                history_len=history_len,
                slot_mappings=slot_mappings,
                request_metadata=request_metadata,
                prompt_lens=_first_present(prompt_lens, prompt_lengths, prompt_token_counts),
                exclude_ranges=exclude_ranges,
            )
            if metadata is None:
                if self.require_metadata:
                    self.stats.metadata_missing += 1
                    self.stats.record_miss()
                    drafts.append([])
                    continue
                metadata = RequestPLDMetadata(prompt_len=max(0, history_len - len(sample or [])))

            prompt_len = max(0, min(metadata.prompt_len, history_len))
            prompt_ids = history[:prompt_len]
            generated_ids = history[prompt_len:]
            proposal = find_pld_proposal(
                prompt_ids,
                generated_ids,
                match_n=self.match_n,
                max_draft_len=self.max_draft_len,
                cap=self.cap,
                exclude_ranges=metadata.exclude_ranges,
                search_prompt=metadata.search_prompt,
                search_generated=metadata.search_generated,
                tie_break=self.tie_break,  # type: ignore[arg-type]
            )
            if proposal is None:
                self.stats.record_miss()
                drafts.append([])
                continue

            self.stats.record_hit(proposal)
            drafts.append(list(proposal.tokens))

        return drafts

    def _metadata_for_index(
        self,
        index: int,
        *,
        history_len: int,
        slot_mappings: Any | None,
        request_metadata: Any | None,
        prompt_lens: Sequence[int] | None,
        exclude_ranges: Sequence[Iterable[TokenRange]] | None,
    ) -> RequestPLDMetadata | None:
        per_request = _metadata_record_at(request_metadata, index)
        slot_record = _metadata_record_at(slot_mappings, index)

        prompt_len = _first_present_int(
            _sequence_value_at(prompt_lens, index),
            _metadata_value(per_request, "prompt_len"),
            _metadata_value(per_request, "prompt_length"),
            _metadata_value(per_request, "prompt_token_count"),
            _metadata_value(per_request, "context_len"),
            _metadata_value(slot_record, "prompt_len"),
            _metadata_value(slot_record, "prompt_length"),
            _metadata_value(slot_record, "prompt_token_count"),
            _metadata_value(slot_record, "context_len"),
            _sequence_value_at(_metadata_value(request_metadata, "prompt_lens"), index),
            _sequence_value_at(_metadata_value(request_metadata, "prompt_lengths"), index),
            _sequence_value_at(_metadata_value(request_metadata, "prompt_token_counts"), index),
            _sequence_value_at(_metadata_value(slot_mappings, "prompt_lens"), index),
            _sequence_value_at(_metadata_value(slot_mappings, "prompt_lengths"), index),
        )
        if prompt_len is None:
            return None

        ranges = _first_present(
            _sequence_value_at(exclude_ranges, index),
            _metadata_value(per_request, "exclude_ranges"),
            _metadata_value(per_request, "pld_exclude_ranges"),
            _metadata_value(per_request, "gold_ranges"),
            _metadata_value(per_request, "gold_token_ranges"),
            _metadata_value(slot_record, "exclude_ranges"),
            _metadata_value(slot_record, "pld_exclude_ranges"),
        )
        search_prompt = _first_present_bool(
            _metadata_value(per_request, "search_prompt"),
            _metadata_value(slot_record, "search_prompt"),
            default=self.search_prompt,
        )
        search_generated = _first_present_bool(
            _metadata_value(per_request, "search_generated"),
            _metadata_value(slot_record, "search_generated"),
            default=self.search_generated,
        )
        return RequestPLDMetadata(
            prompt_len=max(0, min(int(prompt_len), history_len)),
            exclude_ranges=_normalize_metadata_ranges(ranges),
            search_prompt=search_prompt,
            search_generated=search_generated,
        )


def _config_value(config: Any | None, key: str, default: Any) -> Any:
    if config is None:
        return default
    for source in (
        getattr(config, "speculative_config", None),
        getattr(config, "spec_config", None),
        config,
    ):
        value = _metadata_value(source, key)
        if value is not None:
            return value
    model_config = getattr(config, "model_config", None)
    if key == "max_model_len":
        value = _metadata_value(model_config, "max_model_len")
        if value is not None:
            return value
    return default


def _metadata_record_at(source: Any, index: int) -> Any:
    if source is None:
        return None
    if isinstance(source, dict):
        for key in ("requests", "request_metadata", "metadata", "items"):
            rows = source.get(key)
            value = _sequence_value_at(rows, index)
            if value is not None:
                return value
        value = source.get(index)
        if value is not None:
            return value
        value = source.get(str(index))
        return value
    return _sequence_value_at(source, index)


def _metadata_value(source: Any, key: str) -> Any:
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _first_present_int(*values: Any) -> int | None:
    value = _first_present(*values)
    return _int_or_none(value)


def _first_present_bool(*values: Any, default: bool) -> bool:
    value = _first_present(*values)
    return bool(default if value is None else value)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _infer_num_sequences(sampled_token_ids: Any, num_tokens_no_spec: Any, token_ids_cpu: Any) -> int:
    for value in (num_tokens_no_spec, sampled_token_ids, token_ids_cpu):
        try:
            return len(value)
        except TypeError:
            pass
    shape = getattr(token_ids_cpu, "shape", None)
    if shape:
        return int(shape[0])
    return 1


def _sequence_value_at(value: Any, index: int) -> Any:
    if value is None:
        return None
    try:
        return value[index]
    except (IndexError, KeyError, TypeError):
        return None


def _sequence_at(value: Any, index: int) -> list[int] | None:
    item = _sequence_value_at(value, index)
    if item is None:
        return None
    return [int(token) for token in _tolist(item)]


def _int_at(value: Any, index: int) -> int:
    item = _sequence_value_at(value, index)
    if item is None:
        return int(value)
    return int(item)


def _history_row(token_ids_cpu: Any, index: int, length: int) -> list[int]:
    row = _sequence_value_at(token_ids_cpu, index)
    if row is None:
        row = token_ids_cpu
    return [int(token) for token in _tolist(row)[:length]]


def _tolist(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
        return [converted]
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    try:
        return list(value)
    except TypeError:
        return [value]


def _normalize_metadata_ranges(value: Any) -> tuple[TokenRange, ...]:
    if value is None:
        return ()
    if isinstance(value, dict):
        start = _metadata_value(value, "start")
        end = _metadata_value(value, "end")
        if start is not None and end is not None:
            return ((int(start), int(end)),)
        value = value.values()
    ranges: list[TokenRange] = []
    for item in value:
        if isinstance(item, dict):
            start = item.get("start")
            end = item.get("end")
        else:
            start, end = item
        ranges.append((int(start), int(end)))
    return tuple(ranges)
