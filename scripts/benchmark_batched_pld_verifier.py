#!/usr/bin/env python3
"""Batched PLD verifier microbenchmark with real trace-derived shapes.

This benchmark isolates the target-model verifier forward that a
continuous-batched PLD scheduler would batch across active requests.  It keeps
real task prefixes and real PLD draft-length distributions from an existing
``blazedit_pld_w128_n10`` trace, but normalizes every step to the verifier
shape used after a task has a cache:

    cache:        prefix[:-1]
    verifier in: prefix[-1:] + draft

The cache-building prefill is intentionally excluded from the measured forward
time.  The goal is to answer the Stage-1 question: does batching verifier
forwards improve aggregate verified-token throughput enough to justify a full
scheduler?
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from scripts.benchmark_real_shape_forward import (
    bucket_leq,
    encode_completion_tokens,
    load_trace,
    make_drafts,
    model_dtype_arg,
    percentile,
    summarize,
)
from scripts.run_eagle_eval import _encode_prompt_ids, _load_model


@dataclass(frozen=True)
class VerifierExample:
    task_id: str
    step_id: int
    prefix_len: int
    cache_len: int
    draft_len: int
    bucket_len: int
    accepted_len: int
    emitted: int
    cache_tokens: tuple[int, ...]
    input_tokens: tuple[int, ...]


@dataclass
class ComboResult:
    batch_size: int
    bucket_len: int
    num_batches: int
    num_examples: int
    mean_forward_ms: float | None
    p50_forward_ms: float | None
    p90_forward_ms: float | None
    effective_verified_tokens_per_sec: float | None
    speedup_vs_batch1: float | None
    memory_gb: float | None
    cache_len_mean: float | None
    cache_len_p90: float | None
    padding_waste_cache_tokens_mean: float | None
    oom_or_success: str
    error: str | None = None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parse_ints(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def _gpu_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024.0**3))


def _safe_token(tokenizer, target) -> int:
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(getattr(target, "config", None), "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    return int(pad or 0)


def build_verifier_examples(
    *,
    tokenizer,
    target,
    steps_by_task: dict[str, list[dict[str, Any]]],
    completions: dict[str, dict[str, Any]],
    method: str,
    chat_template: str,
    bucket_sizes: list[int],
    max_examples: int,
    seed: int,
) -> list[VerifierExample]:
    """Build trace-derived verifier examples.

    For each PLD step we reconstruct the verified generated prefix from the
    completion artifact and use trace draft/acceptance lengths for the draft
    shape.  Drafts are padded up to their bucket so the measured forward does
    the same amount of verifier work as a bucketed scheduler.
    """

    fallback_tokens = tokenizer(
        "def _fallback_value():\n    return None\n",
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0].tolist()
    pad_id = _safe_token(tokenizer, target)
    examples: list[VerifierExample] = []

    for task_id, task_steps in steps_by_task.items():
        completion = completions.get(task_id)
        if not completion:
            continue
        prompt_ids = _encode_prompt_ids(tokenizer, completion["prompt"], chat_template)
        prompt_list = [int(x) for x in prompt_ids.tolist()]
        if not prompt_list:
            continue
        generated_ids = encode_completion_tokens(tokenizer, completion, method)
        prefix = list(prompt_list)
        generated_pos = 0
        for row in task_steps:
            if max_examples and len(examples) >= max_examples:
                break
            old_prefix_len = len(prefix)
            if old_prefix_len < 1:
                continue

            trace_draft_len = int(row.get("target_draft_tokens") or row.get("k") or 0)
            accepted_len = int(row.get("target_accepted_nonroot") or row.get("n_accepted_drafts") or 0)
            emitted = int(row.get("n_emitted") or (accepted_len + 1))
            bucket_len = bucket_leq(max(1, trace_draft_len), bucket_sizes)

            # Cache-normalized verifier shape: cache all but the final prefix
            # token, then verify that token plus a bucket-padded draft.
            cache_len = max(0, old_prefix_len - 1)
            cache_tokens = tuple(int(x) for x in prefix[:cache_len])
            drafts = make_drafts(
                generated_ids=generated_ids,
                generated_pos=generated_pos,
                accepted_len=accepted_len,
                draft_len=bucket_len,
                fallback_tokens=fallback_tokens or [pad_id],
            )
            input_tokens = tuple([int(prefix[-1])] + [int(x) for x in drafts])
            examples.append(
                VerifierExample(
                    task_id=str(task_id),
                    step_id=int(row.get("step", len(examples))),
                    prefix_len=old_prefix_len,
                    cache_len=cache_len,
                    draft_len=trace_draft_len,
                    bucket_len=bucket_len,
                    accepted_len=accepted_len,
                    emitted=emitted,
                    cache_tokens=cache_tokens,
                    input_tokens=input_tokens,
                )
            )

            append_tokens = generated_ids[generated_pos : generated_pos + emitted]
            if len(append_tokens) < emitted:
                append_tokens = append_tokens + (fallback_tokens or [pad_id])[: emitted - len(append_tokens)]
            prefix.extend(int(x) for x in append_tokens)
            generated_pos += emitted
        if max_examples and len(examples) >= max_examples:
            break

    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples


def _build_prefill_tensors(
    examples: list[VerifierExample],
    *,
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(examples)
    max_cache_len = max(max(1, ex.cache_len) for ex in examples)
    input_ids = torch.full((batch, max_cache_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch, max_cache_len), dtype=torch.long, device=device)
    position_ids = torch.zeros((batch, max_cache_len), dtype=torch.long, device=device)
    for row, ex in enumerate(examples):
        toks = list(ex.cache_tokens)
        if not toks:
            toks = [pad_id]
        n = len(toks)
        input_ids[row, :n] = torch.tensor(toks, dtype=torch.long, device=device)
        attention_mask[row, :n] = 1
        position_ids[row, :n] = torch.arange(n, dtype=torch.long, device=device)
    return input_ids, attention_mask, position_ids


def _build_verify_tensors(
    examples: list[VerifierExample],
    *,
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(examples)
    max_cache_len = max(max(1, ex.cache_len) for ex in examples)
    max_input_len = max(len(ex.input_tokens) for ex in examples)
    input_ids = torch.full((batch, max_input_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch, max_cache_len + max_input_len), dtype=torch.long, device=device)
    position_ids = torch.zeros((batch, max_input_len), dtype=torch.long, device=device)
    for row, ex in enumerate(examples):
        toks = list(ex.input_tokens)
        n = len(toks)
        cache_n = max(1, ex.cache_len)
        input_ids[row, :n] = torch.tensor(toks, dtype=torch.long, device=device)
        attention_mask[row, :cache_n] = 1
        attention_mask[row, max_cache_len : max_cache_len + max_input_len] = 1
        start_pos = int(ex.cache_len)
        position_ids[row, :max_input_len] = torch.arange(
            start_pos,
            start_pos + max_input_len,
            dtype=torch.long,
            device=device,
        )
    return input_ids, attention_mask, position_ids


def _measure_one_batch(
    *,
    target,
    examples: list[VerifierExample],
    pad_id: int,
    device: torch.device,
) -> float:
    prefill_ids, prefill_mask, prefill_pos = _build_prefill_tensors(examples, pad_id=pad_id, device=device)
    with torch.inference_mode():
        prefill = target(
            prefill_ids,
            attention_mask=prefill_mask,
            position_ids=prefill_pos,
            use_cache=True,
        )
        verify_ids, verify_mask, verify_pos = _build_verify_tensors(examples, pad_id=pad_id, device=device)
        _sync(device)
        t0 = time.perf_counter_ns()
        _ = target(
            verify_ids,
            past_key_values=prefill.past_key_values,
            attention_mask=verify_mask,
            position_ids=verify_pos,
            use_cache=True,
        )
        _sync(device)
    return (time.perf_counter_ns() - t0) / 1_000_000.0


def run_combo(
    *,
    target,
    examples: list[VerifierExample],
    batch_size: int,
    bucket_len: int,
    pad_id: int,
    device: torch.device,
    iters_per_combo: int,
    warmup_batches: int,
) -> ComboResult:
    candidates = [ex for ex in examples if ex.bucket_len == bucket_len]
    num_batches = min(iters_per_combo, len(candidates) // batch_size)
    if num_batches <= 0:
        return ComboResult(
            batch_size=batch_size,
            bucket_len=bucket_len,
            num_batches=0,
            num_examples=len(candidates),
            mean_forward_ms=None,
            p50_forward_ms=None,
            p90_forward_ms=None,
            effective_verified_tokens_per_sec=None,
            speedup_vs_batch1=None,
            memory_gb=None,
            cache_len_mean=None,
            cache_len_p90=None,
            padding_waste_cache_tokens_mean=None,
            oom_or_success="insufficient_examples",
        )

    samples: list[float] = []
    cache_lens: list[int] = []
    padding_waste: list[int] = []
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        all_batches = warmup_batches + num_batches
        for batch_idx in range(all_batches):
            start = batch_idx * batch_size
            batch = candidates[start : start + batch_size]
            if len(batch) < batch_size:
                break
            max_cache_len = max(max(1, ex.cache_len) for ex in batch)
            cache_lens.extend(int(ex.cache_len) for ex in batch)
            padding_waste.extend(max_cache_len - max(1, ex.cache_len) for ex in batch)
            ms = _measure_one_batch(
                target=target,
                examples=batch,
                pad_id=pad_id,
                device=device,
            )
            if batch_idx >= warmup_batches:
                samples.append(ms)
            del batch
        if not samples:
            return ComboResult(
                batch_size=batch_size,
                bucket_len=bucket_len,
                num_batches=0,
                num_examples=len(candidates),
                mean_forward_ms=None,
                p50_forward_ms=None,
                p90_forward_ms=None,
                effective_verified_tokens_per_sec=None,
                speedup_vs_batch1=None,
                memory_gb=_gpu_memory_gb(device),
                cache_len_mean=None,
                cache_len_p90=None,
                padding_waste_cache_tokens_mean=None,
                oom_or_success="no_samples",
            )
        total_verified_tokens = float(len(samples) * batch_size * bucket_len)
        total_seconds = sum(samples) / 1000.0
        return ComboResult(
            batch_size=batch_size,
            bucket_len=bucket_len,
            num_batches=len(samples),
            num_examples=len(candidates),
            mean_forward_ms=float(statistics.fmean(samples)),
            p50_forward_ms=float(statistics.median(samples)),
            p90_forward_ms=percentile(samples, 0.90),
            effective_verified_tokens_per_sec=total_verified_tokens / max(1e-9, total_seconds),
            speedup_vs_batch1=None,
            memory_gb=_gpu_memory_gb(device),
            cache_len_mean=float(statistics.fmean(cache_lens)) if cache_lens else None,
            cache_len_p90=percentile([float(x) for x in cache_lens], 0.90) if cache_lens else None,
            padding_waste_cache_tokens_mean=(
                float(statistics.fmean(padding_waste)) if padding_waste else None
            ),
            oom_or_success="success",
        )
    except torch.cuda.OutOfMemoryError as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return ComboResult(
            batch_size=batch_size,
            bucket_len=bucket_len,
            num_batches=len(samples),
            num_examples=len(candidates),
            mean_forward_ms=(float(statistics.fmean(samples)) if samples else None),
            p50_forward_ms=(float(statistics.median(samples)) if samples else None),
            p90_forward_ms=(percentile(samples, 0.90) if samples else None),
            effective_verified_tokens_per_sec=None,
            speedup_vs_batch1=None,
            memory_gb=_gpu_memory_gb(device),
            cache_len_mean=None,
            cache_len_p90=None,
            padding_waste_cache_tokens_mean=None,
            oom_or_success="oom",
            error=f"{type(exc).__name__}: {exc}",
        )
    except Exception as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return ComboResult(
            batch_size=batch_size,
            bucket_len=bucket_len,
            num_batches=len(samples),
            num_examples=len(candidates),
            mean_forward_ms=(float(statistics.fmean(samples)) if samples else None),
            p50_forward_ms=(float(statistics.median(samples)) if samples else None),
            p90_forward_ms=(percentile(samples, 0.90) if samples else None),
            effective_verified_tokens_per_sec=None,
            speedup_vs_batch1=None,
            memory_gb=_gpu_memory_gb(device),
            cache_len_mean=None,
            cache_len_p90=None,
            padding_waste_cache_tokens_mean=None,
            oom_or_success="error",
            error=f"{type(exc).__name__}: {exc}",
        )


def attach_speedups(results: list[ComboResult]) -> None:
    base_by_bucket: dict[int, float] = {}
    for row in results:
        if row.batch_size == 1 and row.effective_verified_tokens_per_sec:
            base_by_bucket[row.bucket_len] = row.effective_verified_tokens_per_sec
    for row in results:
        base = base_by_bucket.get(row.bucket_len)
        if base and row.effective_verified_tokens_per_sec:
            row.speedup_vs_batch1 = row.effective_verified_tokens_per_sec / base


def aggregate_decision(results: list[ComboResult]) -> dict[str, Any]:
    by_batch: dict[int, list[ComboResult]] = {}
    for row in results:
        if row.oom_or_success == "success" and row.effective_verified_tokens_per_sec:
            by_batch.setdefault(row.batch_size, []).append(row)
    weighted_tps: dict[str, float] = {}
    for batch, rows in by_batch.items():
        numer = sum((r.effective_verified_tokens_per_sec or 0.0) * max(1, r.num_batches) for r in rows)
        denom = sum(max(1, r.num_batches) for r in rows)
        weighted_tps[str(batch)] = numer / max(1, denom)
    base = weighted_tps.get("1", 0.0)
    speedups = {k: (v / base if base else 0.0) for k, v in weighted_tps.items()}
    b4 = speedups.get("4", 0.0)
    b8 = speedups.get("8", 0.0)
    if b8 >= 1.50:
        decision = "strong_pass_batch8"
    elif b4 >= 1.25:
        decision = "pass_batch4"
    elif max(speedups.values() or [0.0]) < 1.10:
        decision = "stop_batching_under_1p10"
    else:
        decision = "marginal"
    return {
        "weighted_effective_verified_tokens_per_sec_by_batch": weighted_tps,
        "weighted_speedup_vs_batch1_by_batch": speedups,
        "decision": decision,
    }


def write_report(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    examples: list[VerifierExample],
    results: list[ComboResult],
    decision: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args),
        "num_examples": len(examples),
        "example_bucket_counts": {
            str(b): sum(1 for ex in examples if ex.bucket_len == b)
            for b in _parse_ints(args.bucket_sizes)
        },
        "decision": decision,
        "results": [asdict(r) for r in results],
    }
    (output_dir / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines: list[str] = []
    lines.append("# Batched PLD Verifier Microbenchmark\n")
    lines.append(
        f"target: `{args.target}`  dtype: `{args.dtype}`  attn: `{args.attn}`  "
        f"examples: `{len(examples)}`\n"
    )
    lines.append(
        "This is a cache-normalized verifier benchmark: prefill/cache construction "
        "is excluded; the timed operation is the batched `prefix[-1] + draft` target forward.\n"
    )
    lines.append("## Decision\n")
    speedups = decision.get("weighted_speedup_vs_batch1_by_batch", {})
    for b in sorted(speedups, key=lambda x: int(x)):
        lines.append(f"- batch `{b}` weighted verifier-throughput speedup: `{speedups[b]:.3f}x`")
    lines.append(f"- gate decision: `{decision.get('decision')}`\n")

    lines.append("## Results\n")
    lines.append(
        "| batch_size | bucket_len | batches | examples | mean_forward_ms | p50 | p90 | "
        "verified_tok/s | speedup_vs_b1 | memory_gb | cache_p90 | pad_cache_mean | status |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in sorted(results, key=lambda x: (x.bucket_len, x.batch_size)):
        lines.append(
            f"| {r.batch_size} | {r.bucket_len} | {r.num_batches} | {r.num_examples} | "
            f"{(r.mean_forward_ms if r.mean_forward_ms is not None else math.nan):.3f} | "
            f"{(r.p50_forward_ms if r.p50_forward_ms is not None else math.nan):.3f} | "
            f"{(r.p90_forward_ms if r.p90_forward_ms is not None else math.nan):.3f} | "
            f"{(r.effective_verified_tokens_per_sec if r.effective_verified_tokens_per_sec is not None else math.nan):.1f} | "
            f"{(r.speedup_vs_batch1 if r.speedup_vs_batch1 is not None else math.nan):.3f} | "
            f"{(r.memory_gb if r.memory_gb is not None else math.nan):.2f} | "
            f"{(r.cache_len_p90 if r.cache_len_p90 is not None else math.nan):.0f} | "
            f"{(r.padding_waste_cache_tokens_mean if r.padding_waste_cache_tokens_mean is not None else math.nan):.1f} | "
            f"{r.oom_or_success} |"
        )
        if r.error:
            lines.append(f"<!-- {r.batch_size=} {r.bucket_len=}: {r.error} -->")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print((output_dir / "report.md").read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", required=True)
    parser.add_argument("--completions", required=True)
    parser.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--method", default="blazedit_pld_w128_n10")
    parser.add_argument("--chat-template", default="none")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16")
    parser.add_argument("--bucket-sizes", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--iters-per-combo", type=int, default=12)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    batch_sizes = _parse_ints(args.batch_sizes)
    bucket_sizes = _parse_ints(args.bucket_sizes)

    tokenizer, target = _load_model(
        args.target,
        dtype=model_dtype_arg(args.dtype),
        attn_impl=args.attn,
    )
    target.eval()
    steps_by_task, completions = load_trace(
        steps_path=args.steps,
        completions_path=args.completions,
        method=args.method,
    )
    examples = build_verifier_examples(
        tokenizer=tokenizer,
        target=target,
        steps_by_task=steps_by_task,
        completions=completions,
        method=args.method,
        chat_template=args.chat_template,
        bucket_sizes=bucket_sizes,
        max_examples=args.max_examples,
        seed=args.seed,
    )
    pad_id = _safe_token(tokenizer, target)
    results: list[ComboResult] = []
    for bucket_len in bucket_sizes:
        for batch_size in batch_sizes:
            print(f"[combo] bucket={bucket_len} batch={batch_size}", flush=True)
            results.append(
                run_combo(
                    target=target,
                    examples=examples,
                    batch_size=batch_size,
                    bucket_len=bucket_len,
                    pad_id=pad_id,
                    device=device,
                    iters_per_combo=args.iters_per_combo,
                    warmup_batches=args.warmup_batches,
                )
            )
    attach_speedups(results)
    decision = aggregate_decision(results)
    write_report(
        output_dir=Path(args.output_dir),
        args=args,
        examples=examples,
        results=results,
        decision=decision,
    )


if __name__ == "__main__":
    main()
