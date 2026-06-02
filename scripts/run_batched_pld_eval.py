#!/usr/bin/env python3
"""Continuous-batched BlazEdit PLD evaluation harness.

This is a scheduler-level experiment: PLD lookup remains per task, while target
verification is grouped across active tasks by draft-length bucket.  The output
semantics remain ordinary greedy target verification: every drafted token is
checked by the target model, and the target correction/bonus token is emitted
exactly as in the sequential PLD loop.

The first implementation is intentionally conservative:

* each task owns its compact KV cache;
* batched forwards use padded cache rows plus attention masks;
* after each batched forward, each selected cache row is compacted back to the
  task's true cache positions, so padding never becomes task state;
* prompt prefill is done when a task enters the active pool and is counted in
  wall time.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import statistics
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from asts.blazedit_decoder import (
    BlazEditConfig,
    blazedit_speculative_ar,
    prompt_lookup_draft,
)
from asts.decoder import crop_dynamic_cache
from asts.humaneval import Problem, load_problems_from_jsonl
from asts.rejection import GreedyVerifyResult, greedy_verify
from scripts.benchmark_real_shape_forward import bucket_leq, model_dtype_arg, percentile, summarize
from scripts.run_eagle_eval import _encode_prompt_ids, _load_model


FINAL_METHOD_NAME = "continuous_batched_pld_w128_n10"
FINAL_CONFIG_NAME = "continuous_batched_pld_final_b8_pool32"
BUCKET_POLICIES = {
    "default": [8, 16, 32, 64, 128],
    "fine": [1, 2, 4, 8, 16, 32, 64, 128],
    "single": [128],
    "custom": [],
}


@dataclass
class ActiveTask:
    task_id: str
    prompt: str
    prompt_ids: list[int]
    prefix: list[int]
    prompt_len: int
    target_cache: object | None = None
    target_cache_len: int = 0
    finished: bool = False
    finish_reason: str = ""
    steps: int = 0
    generated_tokens: int = 0
    prefill_ms: float = 0.0
    latency_ms: float = 0.0
    start_ns: int = 0
    end_ns: int = 0


@dataclass
class PendingVerify:
    task: ActiveTask
    drafts: list[int]
    bucket_len: int
    match_len: int
    source_start: int
    lookup_us: float
    old_prefix_len: int


@dataclass
class BatchedRunMetrics:
    batch_size: int
    active_pool_size: int
    refill_policy: str = "continuous"
    bucket_policy: str = "custom"
    n_tasks: int = 0
    total_new_tokens: int = 0
    wall_ms: float = 0.0
    generated_tokens_per_sec: float = 0.0
    total_forward_ms: float = 0.0
    total_prefill_ms: float = 0.0
    scheduler_overhead_ms: float = 0.0
    pld_lookup_ms: float = 0.0
    decode_steps: int = 0
    verifier_forwards: int = 0
    verified_tokens_per_forward: float = 0.0
    accepted_tokens_per_forward: float = 0.0
    real_verified_tokens: int = 0
    padding_waste_tokens: int = 0
    input_padding_waste_tokens: int = 0
    cache_padding_waste_tokens: int = 0
    memory_peak_gb: float = 0.0
    output_match_count: int = 0
    output_mismatch_count: int = 0
    active_tasks_mean: float = 0.0
    active_tasks_p50: float = 0.0
    active_tasks_p90: float = 0.0
    bucket_counts: dict[str, int] = field(default_factory=dict)
    bucket_forward_ms: dict[str, dict[str, float]] = field(default_factory=dict)
    task_latency_ms: dict[str, float] = field(default_factory=dict)
    task_latency_summary_ms: dict[str, float] = field(default_factory=dict)
    output_equivalence_note: str = ""
    audit_trace_path: str = ""
    error: str = ""


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _safe_token(tokenizer, target) -> int:
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(getattr(target, "config", None), "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    return int(pad or 0)


def _parse_ints(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def resolve_bucket_sizes(bucket_policy: str, bucket_sizes: str | list[int]) -> list[int]:
    if bucket_policy not in BUCKET_POLICIES:
        raise ValueError(f"unknown bucket_policy: {bucket_policy}")
    if bucket_policy == "custom":
        if isinstance(bucket_sizes, str):
            return _parse_ints(bucket_sizes)
        return [int(x) for x in bucket_sizes]
    return list(BUCKET_POLICIES[bucket_policy])


def _hash_token_ids(tokens: list[int]) -> str:
    h = hashlib.sha1()
    for tok in tokens:
        h.update(int(tok).to_bytes(8, byteorder="little", signed=True))
    return h.hexdigest()[:16]


def _eos_truncate(tokens: list[int], eos_token_ids: list[int], budget: int) -> tuple[list[int], bool]:
    out = list(tokens)
    for i, tok in enumerate(out):
        if tok in eos_token_ids:
            out = out[: i + 1]
            break
    out = out[: max(0, budget)]
    return out, any(t in eos_token_ids for t in out)


def _to_legacy_cache(cache):
    if cache is None:
        return None
    if isinstance(cache, tuple):
        return cache
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    if hasattr(cache, "layers"):
        layers = []
        for layer in cache.layers:
            key = getattr(layer, "keys", None)
            val = getattr(layer, "values", None)
            if key is None:
                key = getattr(layer, "key_cache", None)
            if val is None:
                val = getattr(layer, "value_cache", None)
            if key is None or val is None:
                return cache
            layers.append((key, val))
        return tuple(layers)
    return cache


def _legacy_cache_for_model(legacy):
    """Build a HF cache object from legacy key/value tensors when required.

    Newer Transformers releases no longer accept a raw tuple as
    ``past_key_values`` because mask construction calls ``get_seq_length`` on
    the cache object.  ``DynamicCache`` still accepts an iterable of layer
    key/value pairs as its first argument, which is exactly the legacy layout.
    """

    try:
        from transformers.cache_utils import DynamicCache

        if hasattr(DynamicCache, "from_legacy_cache"):
            return DynamicCache.from_legacy_cache(legacy)
        try:
            return DynamicCache(ddp_cache_data=legacy)
        except Exception:
            try:
                return DynamicCache(legacy)
            except Exception:
                pass
    except Exception:
        pass
    return legacy


def _cache_layers(cache) -> tuple[list[torch.Tensor], list[torch.Tensor]] | None:
    if cache is None:
        return None
    if isinstance(cache, tuple):
        keys: list[torch.Tensor] = []
        vals: list[torch.Tensor] = []
        for layer in cache:
            if not (isinstance(layer, tuple) and len(layer) >= 2):
                return None
            keys.append(layer[0])
            vals.append(layer[1])
        return keys, vals
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return cache.key_cache, cache.value_cache
    if hasattr(cache, "layers"):
        keys: list[torch.Tensor] = []
        vals: list[torch.Tensor] = []
        for layer in cache.layers:
            key = getattr(layer, "keys", None)
            val = getattr(layer, "values", None)
            if key is None:
                key = getattr(layer, "key_cache", None)
            if val is None:
                val = getattr(layer, "value_cache", None)
            if key is None or val is None:
                return None
            keys.append(key)
            vals.append(val)
        return keys, vals
    legacy = _to_legacy_cache(cache)
    if legacy is not cache:
        return _cache_layers(legacy)
    return None


def _crop_task_cache(task: ActiveTask, target_len: int) -> None:
    target_len = max(0, int(target_len))
    cache = _to_legacy_cache(task.target_cache)
    if cache is None:
        task.target_cache = None
        task.target_cache_len = 0
        return
    if isinstance(cache, tuple):
        cropped = []
        for layer in cache:
            key, val = layer[0], layer[1]
            cropped.append((key[:, :, :target_len, :].contiguous(), val[:, :, :target_len, :].contiguous()))
        task.target_cache = tuple(cropped)
        task.target_cache_len = target_len
        return
    crop_dynamic_cache(cache, target_len)
    task.target_cache = cache
    task.target_cache_len = target_len


def _release_task_cache(task: ActiveTask) -> None:
    """Drop GPU KV state once a task has been scattered to CPU outputs.

    Completed tasks are kept for latency/step accounting.  Keeping their
    ``target_cache`` as well retains a full per-task KV cache until the end of
    the run, which is especially expensive for fp32/eager long-context timing.
    """

    task.target_cache = None
    task.target_cache_len = 0


def _cuda_memory_hygiene(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _prefill_tokens_chunked(
    *,
    target,
    tokens: list[int],
    device: torch.device,
    chunk_size: int,
) -> tuple[object | None, int, float]:
    if not tokens:
        return None, 0, 0.0
    cache = None
    total_ms = 0.0
    chunk = max(1, int(chunk_size))
    for start in range(0, len(tokens), chunk):
        ids = torch.tensor([tokens[start : start + chunk]], dtype=torch.long, device=device)
        _sync(device)
        t0 = time.perf_counter_ns()
        with torch.inference_mode():
            try:
                out = target(ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
            except TypeError:
                out = target(ids, past_key_values=cache, use_cache=True)
        _sync(device)
        total_ms += (time.perf_counter_ns() - t0) / 1_000_000.0
        cache = out.past_key_values
    return _to_legacy_cache(cache), len(tokens), total_ms


def _combine_task_caches(tasks: list[ActiveTask]) -> tuple[object | None, int]:
    caches = [t.target_cache for t in tasks]
    if any(c is None for c in caches):
        if all(c is None for c in caches):
            return None, 0
        raise RuntimeError("cannot batch mixed None/non-None caches")
    legacy_caches = [_to_legacy_cache(c) for c in caches]
    first = legacy_caches[0]
    layers = _cache_layers(first)
    if layers is None:
        raise RuntimeError(f"unsupported cache type for batching: {type(first).__name__}")
    first_keys, first_vals = layers
    max_cache_len = max(int(t.target_cache_len) for t in tasks)
    batched_layers = []
    for layer_idx in range(len(first_keys)):
        sample_k = first_keys[layer_idx]
        sample_v = first_vals[layer_idx]
        if sample_k is None or sample_v is None:
            batched_layers.append((sample_k, sample_v))
            continue
        bsz = len(tasks)
        key_out = sample_k.new_zeros((bsz, *sample_k.shape[1:2], max_cache_len, sample_k.shape[-1]))
        val_out = sample_v.new_zeros((bsz, *sample_v.shape[1:2], max_cache_len, sample_v.shape[-1]))
        for row, task in enumerate(tasks):
            k_layers, v_layers = _cache_layers(task.target_cache)  # type: ignore[arg-type]
            k = k_layers[layer_idx]
            v = v_layers[layer_idx]
            n = int(task.target_cache_len)
            if n > 0:
                key_out[row : row + 1, :, :n, :] = k[:, :, :n, :]
                val_out[row : row + 1, :, :n, :] = v[:, :, :n, :]
        batched_layers.append((key_out.contiguous(), val_out.contiguous()))
    return _legacy_cache_for_model(tuple(batched_layers)), max_cache_len


def _compact_cache_row(
    cache,
    *,
    row: int,
    real_cache_len: int,
    max_cache_len: int,
    keep_input_len: int,
) -> object | None:
    if cache is None:
        return None
    cache = _to_legacy_cache(cache)
    layers = _cache_layers(cache)
    if layers is None:
        raise RuntimeError(f"unsupported cache type for row compaction: {type(cache).__name__}")
    keys, vals = layers
    out_layers = []
    keep_len = int(real_cache_len) + int(keep_input_len)
    for layer_idx in range(len(keys)):
        key = keys[layer_idx]
        val = vals[layer_idx]
        if key is None or val is None:
            out_layers.append((key, val))
            continue
        pieces_k: list[torch.Tensor] = []
        pieces_v: list[torch.Tensor] = []
        if real_cache_len > 0:
            pieces_k.append(key[row : row + 1, :, :real_cache_len, :])
            pieces_v.append(val[row : row + 1, :, :real_cache_len, :])
        if keep_input_len > 0:
            pieces_k.append(key[row : row + 1, :, max_cache_len : max_cache_len + keep_input_len, :])
            pieces_v.append(val[row : row + 1, :, max_cache_len : max_cache_len + keep_input_len, :])
        if pieces_k:
            out_layers.append((torch.cat(pieces_k, dim=-2).contiguous(), torch.cat(pieces_v, dim=-2).contiguous()))
        else:
            out_layers.append((key[row : row + 1, :, :0, :].contiguous(), val[row : row + 1, :, :0, :].contiguous()))
    return tuple(out_layers)


def _prefill_task(
    task: ActiveTask,
    target,
    device: torch.device,
    prefill_chunk_size: int = 0,
) -> float:
    if task.prompt_len <= 1:
        task.target_cache = None
        task.target_cache_len = 0
        return 0.0
    if prefill_chunk_size and task.prompt_len - 1 > prefill_chunk_size:
        cache, cache_len, ms = _prefill_tokens_chunked(
            target=target,
            tokens=task.prompt_ids[:-1],
            device=device,
            chunk_size=prefill_chunk_size,
        )
        task.target_cache = cache
        task.target_cache_len = cache_len
        task.prefill_ms += ms
        return ms
    ids = torch.tensor([task.prompt_ids[:-1]], dtype=torch.long, device=device)
    _sync(device)
    t0 = time.perf_counter_ns()
    with torch.inference_mode():
        try:
            out = target(ids, use_cache=True, logits_to_keep=1)
        except TypeError:
            out = target(ids, use_cache=True)
    _sync(device)
    task.target_cache = _to_legacy_cache(out.past_key_values)
    task.target_cache_len = len(task.prompt_ids) - 1
    ms = (time.perf_counter_ns() - t0) / 1_000_000.0
    task.prefill_ms += ms
    return ms


def _make_verify_tensors(
    pending: list[PendingVerify],
    *,
    max_cache_len: int,
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]], list[int]]:
    row_tokens: list[list[int]] = []
    n_pre_values: list[int] = []
    for item in pending:
        task = item.task
        if task.target_cache_len >= len(task.prefix):
            task.target_cache_len = max(0, len(task.prefix) - 1)
            _crop_task_cache(task, task.target_cache_len)
        n_pre = len(task.prefix) - int(task.target_cache_len)
        toks = list(task.prefix[int(task.target_cache_len) :]) + list(item.drafts)
        row_tokens.append(toks)
        n_pre_values.append(n_pre)
    max_input_len = max(len(x) for x in row_tokens)
    batch = len(pending)
    input_ids = torch.full((batch, max_input_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch, max_cache_len + max_input_len), dtype=torch.long, device=device)
    position_ids = torch.zeros((batch, max_input_len), dtype=torch.long, device=device)
    for row, (item, toks) in enumerate(zip(pending, row_tokens, strict=True)):
        task = item.task
        cache_n = int(task.target_cache_len)
        input_ids[row, : len(toks)] = torch.tensor(toks, dtype=torch.long, device=device)
        if cache_n:
            attention_mask[row, :cache_n] = 1
        attention_mask[row, max_cache_len : max_cache_len + len(toks)] = 1
        position_ids[row, : len(toks)] = torch.arange(
            cache_n,
            cache_n + len(toks),
            dtype=torch.long,
            device=device,
        )
    return input_ids, attention_mask, position_ids, row_tokens, n_pre_values


def _run_batched_verify(
    *,
    target,
    pending: list[PendingVerify],
    pad_id: int,
    eos_token_ids: list[int],
    max_new_tokens: int,
    device: torch.device,
    global_step: int = 0,
    verifier_batch_id: int = 0,
    audit_writer=None,
) -> tuple[float, int, int, int, int, int]:
    tasks = [p.task for p in pending]
    batched_cache, max_cache_len = _combine_task_caches(tasks)
    input_ids, attention_mask, position_ids, row_tokens, n_pre_values = _make_verify_tensors(
        pending,
        max_cache_len=max_cache_len,
        pad_id=pad_id,
        device=device,
    )
    padding_waste = 0
    cache_padding_waste = 0
    input_padding_waste = 0
    for item, toks in zip(pending, row_tokens, strict=True):
        cache_pad = max(0, max_cache_len - int(item.task.target_cache_len))
        input_pad = max(0, input_ids.shape[1] - len(toks))
        cache_padding_waste += cache_pad
        input_padding_waste += input_pad
        padding_waste += cache_pad + input_pad

    _sync(device)
    t0 = time.perf_counter_ns()
    with torch.inference_mode():
        out = target(
            input_ids,
            past_key_values=batched_cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
    _sync(device)
    forward_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

    accepted_total = 0
    verified_total = 0
    for row, item in enumerate(pending):
        task = item.task
        if task.finished:
            raise RuntimeError(f"finished task re-entered verifier: {task.task_id}")
        drafts = list(item.drafts)
        n_pre = n_pre_values[row]
        row_logits = out.logits[row : row + 1, : len(row_tokens[row]), :]
        if drafts:
            result = greedy_verify(drafts=drafts, target_logits=row_logits, n_pre=n_pre)
            verifier_output_tokens = [
                int(x)
                for x in row_logits[
                    0,
                    n_pre - 1 : n_pre - 1 + len(drafts) + 1,
                    :,
                ]
                .argmax(dim=-1)
                .tolist()
            ]
        else:
            next_tok = int(row_logits[0, n_pre - 1].argmax(dim=-1).item())
            result = GreedyVerifyResult(
                accepted_tokens=[next_tok],
                n_accepted_drafts=0,
                rejected=False,
            )
            verifier_output_tokens = [next_tok]
        budget = (task.prompt_len + max_new_tokens) - len(task.prefix)
        prefix_len_before = len(task.prefix)
        cache_len_before = int(task.target_cache_len)
        cache_handle_before = int(id(task.target_cache))
        finished_before = bool(task.finished)
        accepted_capped, hit_eos = _eos_truncate(
            [int(x) for x in result.accepted_tokens],
            eos_token_ids,
            budget,
        )
        accepted_drafts = min(int(result.n_accepted_drafts), max(0, len(accepted_capped)))
        keep_input_len = n_pre + accepted_drafts
        if keep_input_len > len(row_tokens[row]):
            raise RuntimeError(
                f"cache keep length exceeds verifier input for {task.task_id}: "
                f"keep={keep_input_len} input={len(row_tokens[row])}"
            )
        task.target_cache = _compact_cache_row(
            out.past_key_values,
            row=row,
            real_cache_len=int(task.target_cache_len),
            max_cache_len=max_cache_len,
            keep_input_len=keep_input_len,
        )
        task.prefix.extend(accepted_capped)
        task.target_cache_len = max(0, len(task.prefix) - 1)
        if task.target_cache_len != max(0, len(task.prefix) - 1):
            raise RuntimeError(f"cache length invariant failed for {task.task_id}")
        task.generated_tokens = len(task.prefix) - task.prompt_len
        task.steps += 1
        accepted_total += len(accepted_capped)
        verified_total += max(1, len(drafts))
        if hit_eos:
            task.finished = True
            task.finish_reason = "eos"
        elif len(task.prefix) >= task.prompt_len + max_new_tokens:
            task.finished = True
            task.finish_reason = "max_new_tokens"
        if audit_writer is not None:
            correction_token = accepted_capped[-1] if accepted_capped else None
            audit_writer(
                {
                    "event": "verify_scatter",
                    "global_step": int(global_step),
                    "verifier_batch_id": int(verifier_batch_id),
                    "task_id": task.task_id,
                    "batch_slot": int(row),
                    "bucket_len": int(item.bucket_len),
                    "local_decode_step": int(task.steps - 1),
                    "input_ids_hash": _hash_token_ids(row_tokens[row]),
                    "kv_cache_task_id_or_cache_handle": f"{task.task_id}:{id(task.target_cache)}",
                    "cache_handle_before": cache_handle_before,
                    "cache_handle_after": int(id(task.target_cache)),
                    "cache_len_before": cache_len_before,
                    "cache_len_after": int(task.target_cache_len),
                    "prefix_len_before": int(prefix_len_before),
                    "prefix_len_after": int(len(task.prefix)),
                    "n_pre": int(n_pre),
                    "draft_tokens": [int(x) for x in drafts],
                    "verifier_output_tokens": verifier_output_tokens,
                    "accepted_tokens": [int(x) for x in result.accepted_tokens],
                    "accepted_drafts": int(result.n_accepted_drafts),
                    "correction_token": int(correction_token) if correction_token is not None else None,
                    "emitted_tokens": [int(x) for x in accepted_capped],
                    "finished_flag_before": finished_before,
                    "finished_flag_after": bool(task.finished),
                    "finish_reason": task.finish_reason,
                    "source_start": int(item.source_start),
                    "match_len": int(item.match_len),
                    "old_prefix_len": int(item.old_prefix_len),
                }
            )
    return (
        forward_ms,
        padding_waste,
        verified_total,
        accepted_total,
        cache_padding_waste,
        input_padding_waste,
    )


def run_sequential_baseline(
    *,
    problems: list[Problem],
    tokenizer,
    target,
    max_new_tokens: int,
    eos_token_ids: list[int],
    chat_template: str,
    memory_hygiene: bool = False,
    empty_cache_every: int = 0,
    prefill_chunk_size: int = 0,
) -> dict[str, Any]:
    config = BlazEditConfig(
        mode="pld",
        micro_draft_tokens=128,
        max_matching_ngram_size=10,
        target_prefill_chunk_size=prefill_chunk_size,
    )
    outputs: dict[str, list[int]] = {}
    steps = 0
    total_new = 0
    device = next(target.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if memory_hygiene:
        _cuda_memory_hygiene(device)
    t0 = time.perf_counter_ns()
    for idx, prob in enumerate(problems):
        prompt_ids = _encode_prompt_ids(tokenizer, prob.prompt, chat_template)
        result = blazedit_speculative_ar(
            prompt_ids=prompt_ids,
            target=target,
            assistant=None,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            config=config,
            method_name="blazedit_pld_w128_n10",
            tokenizer=tokenizer,
        )
        new_tokens = [int(x) for x in result.output_token_ids[len(prompt_ids) :]]
        outputs[prob.task_id] = new_tokens
        total_new += len(new_tokens)
        steps += len(result.steps)
        del result
        del prompt_ids
        if memory_hygiene and empty_cache_every > 0 and (idx + 1) % empty_cache_every == 0:
            _cuda_memory_hygiene(device)
    wall_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
    peak_gb = (
        torch.cuda.max_memory_allocated(device) / (1024.0**3)
        if device.type == "cuda"
        else 0.0
    )
    return {
        "method": "blazedit_pld_w128_n10",
        "wall_ms": wall_ms,
        "tokens": total_new,
        "tokens_per_sec": total_new / max(1e-9, wall_ms / 1000.0),
        "steps": steps,
        "outputs": outputs,
        "memory_peak_gb": peak_gb,
    }


def run_batched_scheduler(
    *,
    problems: list[Problem],
    tokenizer,
    target,
    max_new_tokens: int,
    eos_token_ids: list[int],
    chat_template: str,
    batch_size: int,
    active_pool_size: int,
    bucket_sizes: list[int],
    baseline_outputs: dict[str, list[int]] | None,
    device: torch.device,
    refill_policy: str = "continuous",
    bucket_policy: str = "custom",
    audit_trace_path: Path | None = None,
    prefill_chunk_size: int = 0,
) -> tuple[BatchedRunMetrics, dict[str, list[int]]]:
    if refill_policy not in {"continuous", "no_refill"}:
        raise ValueError(f"unsupported refill_policy: {refill_policy}")
    pad_id = _safe_token(tokenizer, target)
    pending_problems = deque(problems)
    active: list[ActiveTask] = []
    completed: list[ActiveTask] = []
    outputs: dict[str, list[int]] = {}
    bucket_forward_samples: dict[int, list[float]] = defaultdict(list)
    bucket_counts: Counter[int] = Counter()
    active_counts: list[int] = []
    lookup_ms = 0.0
    forward_ms_total = 0.0
    prefill_ms_total = 0.0
    padding_waste = 0
    cache_padding_waste = 0
    input_padding_waste = 0
    verified_total = 0
    accepted_total = 0
    verifier_forwards = 0
    scheduler_tick = 0
    verifier_batch_id = 0
    audit_fh = None

    if audit_trace_path is not None:
        audit_trace_path.parent.mkdir(parents=True, exist_ok=True)
        audit_fh = audit_trace_path.open("w", encoding="utf-8")

    def write_audit(event: dict[str, Any]) -> None:
        if audit_fh is not None:
            audit_fh.write(json.dumps(event, sort_keys=True) + "\n")

    def refill() -> None:
        nonlocal prefill_ms_total
        while len(active) < active_pool_size and pending_problems:
            prob = pending_problems.popleft()
            prompt_ids = _encode_prompt_ids(tokenizer, prob.prompt, chat_template)
            prompt_list = [int(x) for x in prompt_ids.tolist()]
            task = ActiveTask(
                task_id=prob.task_id,
                prompt=prob.prompt,
                prompt_ids=prompt_list,
                prefix=list(prompt_list),
                prompt_len=len(prompt_list),
                start_ns=time.perf_counter_ns(),
            )
            prefill_ms_total += _prefill_task(
                task,
                target,
                device,
                prefill_chunk_size=prefill_chunk_size,
            )
            active.append(task)
            write_audit(
                {
                    "event": "task_start",
                    "task_id": task.task_id,
                    "prompt_len": int(task.prompt_len),
                    "active_pool_size": int(active_pool_size),
                    "refill_policy": refill_policy,
                    "start_ns": int(task.start_ns),
                }
            )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.perf_counter_ns()
    refill()
    while active:
        scheduler_tick += 1
        active_counts.append(len(active))
        groups: dict[int, list[PendingVerify]] = defaultdict(list)
        t_lookup = time.perf_counter_ns()
        for task in active:
            old_prefix_len = len(task.prefix)
            drafts, match_len, source_start, _follow_start = prompt_lookup_draft(
                task.prefix,
                max_matching_ngram_size=10,
                max_draft_tokens=128,
            )
            remaining = (task.prompt_len + max_new_tokens) - old_prefix_len
            drafts = list(drafts[:remaining])
            bucket_len = bucket_leq(max(1, len(drafts)), bucket_sizes)
            groups[bucket_len].append(
                PendingVerify(
                    task=task,
                    drafts=drafts,
                    bucket_len=bucket_len,
                    match_len=match_len,
                    source_start=source_start,
                    lookup_us=0.0,
                    old_prefix_len=old_prefix_len,
                )
            )
        lookup_ms += (time.perf_counter_ns() - t_lookup) / 1_000_000.0

        for bucket in sorted(groups):
            items = groups[bucket]
            for i in range(0, len(items), batch_size):
                chunk = items[i : i + batch_size]
                verifier_batch_id += 1
                ms, pad_waste, verified, accepted, cache_pad, input_pad = _run_batched_verify(
                    target=target,
                    pending=chunk,
                    pad_id=pad_id,
                    eos_token_ids=eos_token_ids,
                    max_new_tokens=max_new_tokens,
                    device=device,
                    global_step=scheduler_tick,
                    verifier_batch_id=verifier_batch_id,
                    audit_writer=write_audit if audit_fh is not None else None,
                )
                bucket_counts[bucket] += len(chunk)
                bucket_forward_samples[bucket].append(ms)
                forward_ms_total += ms
                padding_waste += pad_waste
                cache_padding_waste += cache_pad
                input_padding_waste += input_pad
                verified_total += verified
                accepted_total += accepted
                verifier_forwards += 1

        still_active: list[ActiveTask] = []
        for task in active:
            if task.finished:
                task.end_ns = time.perf_counter_ns()
                task.latency_ms = (task.end_ns - task.start_ns) / 1_000_000.0
                completed.append(task)
                outputs[task.task_id] = list(task.prefix[task.prompt_len:])
                _release_task_cache(task)
                write_audit(
                    {
                        "event": "task_finish",
                        "task_id": task.task_id,
                        "finish_reason": task.finish_reason,
                        "generated_tokens": int(task.generated_tokens),
                        "steps": int(task.steps),
                        "latency_ms": float(task.latency_ms),
                    }
                )
            else:
                still_active.append(task)
        active = still_active
        if refill_policy == "continuous" or not active:
            refill()

    wall_ms = (time.perf_counter_ns() - t_start) / 1_000_000.0
    if audit_fh is not None:
        audit_fh.close()
    total_new = sum(len(v) for v in outputs.values())
    match_count = 0
    mismatch_count = 0
    if baseline_outputs is not None:
        for task_id, toks in outputs.items():
            if baseline_outputs.get(task_id) == toks:
                match_count += 1
            else:
                mismatch_count += 1
    latency = [t.latency_ms for t in completed]
    bucket_report = {
        str(k): {
            "count": int(bucket_counts[k]),
            **summarize(v),
        }
        for k, v in sorted(bucket_forward_samples.items())
    }
    metrics = BatchedRunMetrics(
        batch_size=batch_size,
        active_pool_size=active_pool_size,
        refill_policy=refill_policy,
        bucket_policy=bucket_policy,
        n_tasks=len(problems),
        total_new_tokens=total_new,
        wall_ms=wall_ms,
        generated_tokens_per_sec=total_new / max(1e-9, wall_ms / 1000.0),
        total_forward_ms=forward_ms_total,
        total_prefill_ms=prefill_ms_total,
        scheduler_overhead_ms=max(0.0, wall_ms - forward_ms_total - prefill_ms_total - lookup_ms),
        pld_lookup_ms=lookup_ms,
        decode_steps=sum(t.steps for t in completed),
        verifier_forwards=verifier_forwards,
        verified_tokens_per_forward=verified_total / max(1, verifier_forwards),
        accepted_tokens_per_forward=accepted_total / max(1, verifier_forwards),
        real_verified_tokens=verified_total,
        padding_waste_tokens=padding_waste,
        input_padding_waste_tokens=input_padding_waste,
        cache_padding_waste_tokens=cache_padding_waste,
        memory_peak_gb=(
            torch.cuda.max_memory_allocated(device) / (1024.0**3)
            if device.type == "cuda"
            else 0.0
        ),
        output_match_count=match_count,
        output_mismatch_count=mismatch_count,
        active_tasks_mean=float(statistics.fmean(active_counts)) if active_counts else 0.0,
        active_tasks_p50=float(statistics.median(active_counts)) if active_counts else 0.0,
        active_tasks_p90=percentile([float(x) for x in active_counts], 0.90) if active_counts else 0.0,
        bucket_counts={str(k): int(v) for k, v in sorted(bucket_counts.items())},
        bucket_forward_ms=bucket_report,
        task_latency_ms={t.task_id: t.latency_ms for t in completed},
        task_latency_summary_ms=summarize(latency),
        output_equivalence_note=(
            "Compared token IDs against sequential blazedit_pld_w128_n10 in the same run. "
            "bf16/SDPA can still drift on near ties; exact deterministic claims require fp32/eager."
        ),
        audit_trace_path=str(audit_trace_path or ""),
    )
    return metrics, outputs


def write_report(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    sequential: dict[str, Any],
    batched: list[BatchedRunMetrics],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args),
        "sequential": {k: v for k, v in sequential.items() if k != "outputs"},
        "batched": [asdict(x) for x in batched],
    }
    (output_dir / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines: list[str] = []
    lines.append("# Continuous-Batched PLD Smoke Eval\n")
    lines.append(
        f"target: `{args.target}`  dtype: `{args.dtype}`  attn: `{args.attn}`  n: `{args.n}`\n"
    )
    seq_tps = float(sequential["tokens_per_sec"])
    lines.append("## Throughput\n")
    lines.append("| method | batch | active pool | refill | buckets | tok/s | speedup | steps | verifier forwards | output matches | peak GB | status |")
    lines.append("|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|")
    lines.append(
        f"| blazedit_pld_w128_n10 | 1 | 1 | n/a | n/a | {seq_tps:.1f} | 1.000 | "
        f"{int(sequential['steps'])} | {int(sequential['steps'])} | {args.n}/{args.n} | - | success |"
    )
    for row in batched:
        status = "failed" if row.error else "success"
        lines.append(
        f"| {FINAL_METHOD_NAME} | {row.batch_size} | {row.active_pool_size} | "
            f"{row.refill_policy} | {row.bucket_policy} | "
            f"{row.generated_tokens_per_sec:.1f} | {row.generated_tokens_per_sec / max(1e-9, seq_tps):.3f} | "
            f"{row.decode_steps} | {row.verifier_forwards} | "
            f"{row.output_match_count}/{row.output_match_count + row.output_mismatch_count} | "
            f"{row.memory_peak_gb:.2f} | {status} |"
        )
        if row.error:
            lines.append(f"<!-- batch {row.batch_size} error: {row.error} -->")
    lines.append("\n## Batched Metrics\n")
    lines.append("| batch | forward ms | prefill ms | lookup ms | scheduler overhead ms | active mean | accepted/forward | verified/forward | padding waste |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in batched:
        lines.append(
            f"| {row.batch_size} | {row.total_forward_ms:.1f} | {row.total_prefill_ms:.1f} | "
            f"{row.pld_lookup_ms:.1f} | {row.scheduler_overhead_ms:.1f} | {row.active_tasks_mean:.1f} | "
            f"{row.accepted_tokens_per_forward:.2f} | {row.verified_tokens_per_forward:.2f} | "
            f"{row.padding_waste_tokens} |"
        )
    lines.append("\nOutput equivalence note: " + (batched[0].output_equivalence_note if batched else "n/a"))
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print((output_dir / "report.md").read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-jsonl", required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--chat-template", default="none")
    parser.add_argument("--batch-sizes", default="2,4,8")
    parser.add_argument("--active-pool-size", type=int, default=0)
    parser.add_argument("--bucket-sizes", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--bucket-policy", choices=sorted(BUCKET_POLICIES), default="custom")
    parser.add_argument("--refill-policy", choices=["continuous", "no_refill"], default="continuous")
    parser.add_argument("--audit-trace", default="")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    tokenizer, target = _load_model(
        args.target,
        dtype=model_dtype_arg(args.dtype),
        attn_impl=args.attn,
    )
    target.eval()
    eos_token_ids = []
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos_token_ids.append(int(tokenizer.eos_token_id))
    if getattr(getattr(target, "config", None), "eos_token_id", None) is not None:
        raw = target.config.eos_token_id
        if isinstance(raw, list):
            eos_token_ids.extend(int(x) for x in raw)
        else:
            eos_token_ids.append(int(raw))
    eos_token_ids = sorted(set(eos_token_ids))
    problems = load_problems_from_jsonl(args.problem_jsonl, n=args.n)
    batch_sizes = _parse_ints(args.batch_sizes)
    bucket_sizes = resolve_bucket_sizes(args.bucket_policy, args.bucket_sizes)

    print("[sequential] running blazedit_pld_w128_n10", flush=True)
    sequential = run_sequential_baseline(
        problems=problems,
        tokenizer=tokenizer,
        target=target,
        max_new_tokens=args.max_new_tokens,
        eos_token_ids=eos_token_ids,
        chat_template=args.chat_template,
    )
    batched_rows: list[BatchedRunMetrics] = []
    for batch_size in batch_sizes:
        pool = args.active_pool_size or max(8, batch_size * 4)
        print(f"[batched] batch={batch_size} active_pool={pool}", flush=True)
        try:
            row, _outputs = run_batched_scheduler(
                problems=problems,
                tokenizer=tokenizer,
                target=target,
                max_new_tokens=args.max_new_tokens,
                eos_token_ids=eos_token_ids,
                chat_template=args.chat_template,
                batch_size=batch_size,
                active_pool_size=pool,
                bucket_sizes=bucket_sizes,
                baseline_outputs=sequential["outputs"],
                device=device,
                refill_policy=args.refill_policy,
                bucket_policy=args.bucket_policy,
                audit_trace_path=(
                    Path(args.audit_trace)
                    if args.audit_trace and len(batch_sizes) == 1
                    else None
                ),
            )
        except Exception as exc:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            row = BatchedRunMetrics(
                batch_size=batch_size,
                active_pool_size=pool,
                n_tasks=len(problems),
                error=f"{type(exc).__name__}: {exc}",
            )
            print(f"[batched] batch={batch_size} failed: {row.error}", flush=True)
        batched_rows.append(row)
    write_report(
        output_dir=Path(args.output_dir),
        args=args,
        sequential=sequential,
        batched=batched_rows,
    )


if __name__ == "__main__":
    main()
