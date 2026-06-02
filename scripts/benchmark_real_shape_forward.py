#!/usr/bin/env python3
"""Replay target forward calls with real PLD trace shapes.

This isolates target-model forward cost from the rest of the decoder while
keeping the important real-shape variables from a PLD trace: prompt lengths,
generated-prefix growth, cache length, and draft lengths.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from asts.decoder import crop_dynamic_cache
from asts.rejection import greedy_verify
from scripts.run_eagle_eval import _encode_prompt_ids, _load_model


SYNTHETIC_FULL_VERIFY_FIT = {
    "fixed_ms": 26.067660141212997,
    "incremental_ms_per_token": 0.06981296830236014,
    "source": "pld_verify_len_l40s_bf16_sdpa_v1",
}


@dataclass
class RealShapeRecord:
    task_id: str
    step_id: int
    prefix_len: int
    draft_len: int
    accepted_len: int
    emitted: int
    cache_len: int
    n_pre: int
    input_tokens: int
    input_shape: list[int]
    attention_mask_shape: list[int] | None
    position_ids_shape: list[int] | None
    forward_ms: float
    logits_projection_ms: float | None
    cache_update_ms: float | None
    argmax_rejection_ms: float
    crop_ms: float
    trace_verify_ms: float | None
    first_step_for_task: bool
    cache_type: str


def model_dtype_arg(name: str) -> str:
    return {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "fp32": "float32",
        "float32": "float32",
    }[name]


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return float(values[idx])


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
    return {
        "mean": float(statistics.fmean(values)),
        "p50": float(statistics.median(values)),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
    }


def bucket_leq(value: int, buckets: Iterable[int]) -> int:
    ordered = sorted(int(b) for b in buckets)
    for b in ordered:
        if value <= b:
            return b
    return ordered[-1]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_trace(
    *,
    steps_path: str | Path,
    completions_path: str | Path,
    method: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    steps_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(steps_path):
        if row.get("method") == method:
            steps_by_task[str(row["task_id"])].append(row)
    for rows in steps_by_task.values():
        rows.sort(key=lambda r: int(r.get("step", 0)))
    completions = {str(row["task_id"]): row for row in load_jsonl(completions_path)}
    return dict(steps_by_task), completions


def encode_completion_tokens(tokenizer, completion_row: dict[str, Any], method: str) -> list[int]:
    output = completion_row.get("outputs", {}).get(method)
    if not output:
        return []
    text = output.get("raw_text", output.get("text", ""))
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
    n_new = output.get("n_new_tokens")
    if isinstance(n_new, int) and n_new >= 0:
        ids = ids[:n_new]
    return [int(x) for x in ids]


def make_drafts(
    *,
    generated_ids: list[int],
    generated_pos: int,
    accepted_len: int,
    draft_len: int,
    fallback_tokens: list[int],
) -> list[int]:
    if draft_len <= 0:
        return []
    accepted = generated_ids[generated_pos : generated_pos + min(accepted_len, draft_len)]
    out = [int(x) for x in accepted]
    filler_source = generated_ids[generated_pos + len(out) :] + fallback_tokens
    idx = 0
    while len(out) < draft_len:
        if idx < len(filler_source):
            out.append(int(filler_source[idx]))
        else:
            out.append(int(fallback_tokens[idx % max(1, len(fallback_tokens))]))
        idx += 1
    return out[:draft_len]


def replay_real_shape_forward(
    *,
    tokenizer,
    target,
    steps_by_task: dict[str, list[dict[str, Any]]],
    completions: dict[str, dict[str, Any]],
    method: str,
    max_steps: int,
    chat_template: str,
    device: torch.device,
    bucket_pad: bool = False,
    bucket_sizes: list[int] | None = None,
    compile_model: bool = False,
) -> list[RealShapeRecord]:
    if compile_model:
        target = torch.compile(target, mode="reduce-overhead")
    records: list[RealShapeRecord] = []
    fallback_tokens = tokenizer(
        "def _fallback_value():\n    return None\n",
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0].tolist()
    bucket_sizes = bucket_sizes or [1, 2, 4, 8, 16, 32, 64, 128]

    with torch.inference_mode():
        for task_id, task_steps in steps_by_task.items():
            completion = completions.get(task_id)
            if not completion:
                continue
            prompt_ids = _encode_prompt_ids(tokenizer, completion["prompt"], chat_template)
            prompt_list = [int(x) for x in prompt_ids.tolist()]
            generated_ids = encode_completion_tokens(tokenizer, completion, method)
            prefix = list(prompt_list)
            generated_pos = 0
            target_cache = None
            target_cache_len = 0
            for row in task_steps:
                if max_steps and len(records) >= max_steps:
                    return records
                old_prefix_len = len(prefix)
                if target_cache_len >= old_prefix_len:
                    target_cache_len = max(0, old_prefix_len - 1)
                    crop_dynamic_cache(target_cache, target_cache_len)
                n_pre = old_prefix_len - target_cache_len
                trace_draft_len = int(row.get("target_draft_tokens") or row.get("k") or 0)
                accepted_len = int(row.get("target_accepted_nonroot") or row.get("n_accepted_drafts") or 0)
                emitted = int(row.get("n_emitted") or (accepted_len + 1))
                draft_len = trace_draft_len
                if bucket_pad and draft_len > 0:
                    draft_len = bucket_leq(draft_len, bucket_sizes)
                elif bucket_pad and draft_len == 0:
                    draft_len = 1
                drafts = make_drafts(
                    generated_ids=generated_ids,
                    generated_pos=generated_pos,
                    accepted_len=accepted_len,
                    draft_len=draft_len,
                    fallback_tokens=fallback_tokens,
                )
                target_input_list = prefix[target_cache_len:] + drafts
                target_input = torch.tensor([target_input_list], device=device, dtype=torch.long)
                sync(device)
                t0 = time.perf_counter_ns()
                out = target(target_input, past_key_values=target_cache, use_cache=True)
                sync(device)
                forward_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
                target_cache = out.past_key_values
                target_cache_len = target_cache_len + len(target_input_list)

                t_argmax = time.perf_counter_ns()
                if drafts:
                    _ = greedy_verify(drafts=drafts, target_logits=out.logits, n_pre=n_pre)
                else:
                    _ = int(out.logits[0, n_pre - 1].argmax(dim=-1).item())
                sync(device)
                argmax_ms = (time.perf_counter_ns() - t_argmax) / 1_000_000.0

                append_tokens = generated_ids[generated_pos : generated_pos + emitted]
                if len(append_tokens) < emitted:
                    append_tokens = append_tokens + fallback_tokens[: emitted - len(append_tokens)]
                prefix.extend(int(x) for x in append_tokens)
                generated_pos += emitted

                t_crop = time.perf_counter_ns()
                crop_dynamic_cache(target_cache, max(0, len(prefix) - 1))
                target_cache_len = max(0, len(prefix) - 1)
                sync(device)
                crop_ms = (time.perf_counter_ns() - t_crop) / 1_000_000.0
                records.append(
                    RealShapeRecord(
                        task_id=task_id,
                        step_id=int(row.get("step", len(records))),
                        prefix_len=old_prefix_len,
                        draft_len=trace_draft_len,
                        accepted_len=accepted_len,
                        emitted=emitted,
                        cache_len=old_prefix_len - n_pre,
                        n_pre=n_pre,
                        input_tokens=len(target_input_list),
                        input_shape=list(target_input.shape),
                        attention_mask_shape=None,
                        position_ids_shape=None,
                        forward_ms=forward_ms,
                        logits_projection_ms=None,
                        cache_update_ms=None,
                        argmax_rejection_ms=argmax_ms,
                        crop_ms=crop_ms,
                        trace_verify_ms=(
                            float(row["verify_us"]) / 1000.0
                            if row.get("verify_us") is not None
                            else None
                        ),
                        first_step_for_task=(int(row.get("step", 0)) == 0),
                        cache_type=type(target_cache).__name__ if target_cache is not None else "None",
                    )
                )
    return records


def aggregate_records(records: list[RealShapeRecord]) -> dict[str, Any]:
    buckets = [0, 1, 2, 4, 8, 16, 32, 64, 128]
    prefix_buckets = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    out: dict[str, Any] = {
        "n_steps": len(records),
        "forward_ms": summarize([r.forward_ms for r in records]),
        "argmax_rejection_ms": summarize([r.argmax_rejection_ms for r in records]),
        "crop_ms": summarize([r.crop_ms for r in records]),
        "draft_len_mean": statistics.fmean([r.draft_len for r in records]) if records else 0.0,
        "prefix_len_mean": statistics.fmean([r.prefix_len for r in records]) if records else 0.0,
        "first_step_forward_ms": summarize([r.forward_ms for r in records if r.first_step_for_task]),
        "cached_step_forward_ms": summarize([r.forward_ms for r in records if not r.first_step_for_task]),
    }
    by_draft: dict[str, Any] = {}
    for b in buckets:
        vals = [
            r.forward_ms
            for r in records
            if (r.draft_len == b if b == 0 else bucket_leq(r.draft_len, buckets[1:]) == b)
        ]
        if vals:
            by_draft[str(b)] = {**summarize(vals), "n": len(vals)}
    out["forward_ms_by_draft_bucket"] = by_draft
    by_prefix: dict[str, Any] = {}
    for b in prefix_buckets:
        vals = [r.forward_ms for r in records if bucket_leq(r.prefix_len, prefix_buckets) == b]
        if vals:
            by_prefix[str(b)] = {**summarize(vals), "n": len(vals)}
    out["forward_ms_by_prefix_bucket"] = by_prefix
    synth = [
        SYNTHETIC_FULL_VERIFY_FIT["fixed_ms"]
        + SYNTHETIC_FULL_VERIFY_FIT["incremental_ms_per_token"] * r.draft_len
        for r in records
    ]
    out["synthetic_projection_ms"] = summarize(synth)
    if records:
        out["real_vs_synthetic_mean_ratio"] = out["forward_ms"]["mean"] / max(
            1e-9, out["synthetic_projection_ms"]["mean"]
        )
    else:
        out["real_vs_synthetic_mean_ratio"] = 0.0
    return out


def write_report(
    *,
    records: list[RealShapeRecord],
    aggregate: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "target": args.target,
        "dtype": args.dtype,
        "attn": args.attn,
        "steps": str(args.steps),
        "completions": str(args.completions),
        "method": args.method,
        "max_steps": args.max_steps,
        "synthetic_reference": SYNTHETIC_FULL_VERIFY_FIT,
        "aggregate": aggregate,
        "records": [asdict(r) for r in records],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with (output_dir / "records.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), sort_keys=True) + "\n")
    lines: list[str] = []
    lines.append("# Real-Shape Target Forward Benchmark\n")
    lines.append(
        f"target: `{args.target}`  dtype: `{args.dtype}`  attn: `{args.attn}`  "
        f"steps: `{aggregate['n_steps']}`\n"
    )
    fm = aggregate["forward_ms"]
    cm = aggregate["cached_step_forward_ms"]
    first = aggregate["first_step_forward_ms"]
    synth = aggregate["synthetic_projection_ms"]
    lines.append("| metric | mean ms | p50 | p90 | p99 |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, row in [
        ("real forward all", fm),
        ("real forward first step", first),
        ("real forward cached steps", cm),
        ("synthetic fit projection", synth),
        ("argmax/rejection", aggregate["argmax_rejection_ms"]),
        ("cache crop", aggregate["crop_ms"]),
    ]:
        lines.append(
            f"| {name} | {row['mean']:.3f} | {row['p50']:.3f} | {row['p90']:.3f} | {row['p99']:.3f} |"
        )
    lines.append(f"\nreal/synthetic mean ratio: `{aggregate['real_vs_synthetic_mean_ratio']:.2f}x`\n")
    lines.append("## By Draft Bucket\n")
    lines.append("| bucket | n | mean forward ms | p90 |")
    lines.append("|---:|---:|---:|---:|")
    for bucket, row in aggregate["forward_ms_by_draft_bucket"].items():
        lines.append(f"| {bucket} | {row['n']} | {row['mean']:.3f} | {row['p90']:.3f} |")
    lines.append("\n## By Prefix Length Bucket\n")
    lines.append("| bucket <= tokens | n | mean forward ms | p90 |")
    lines.append("|---:|---:|---:|---:|")
    for bucket, row in aggregate["forward_ms_by_prefix_bucket"].items():
        lines.append(f"| {bucket} | {row['n']} | {row['mean']:.3f} | {row['p90']:.3f} |")
    if fm["mean"] <= 35:
        decision = "real-shape forward-only is close to the synthetic microbench; investigate runtime-path mismatch."
    elif fm["mean"] >= 80:
        decision = "real-shape forward-only is high; the runtime profile reflects real target-forward cost."
    else:
        decision = "real-shape forward-only sits between synthetic and runtime; inspect prefix/cache buckets."
    lines.append(f"\nDecision: **{decision}**\n")
    (output_dir / "report.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", required=True)
    parser.add_argument("--completions", required=True)
    parser.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--method", default="blazedit_pld_w128_n10")
    parser.add_argument("--chat-template", default="none")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bucket-pad", action="store_true")
    parser.add_argument("--bucket-sizes", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--enable-torch-compile", action="store_true")
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
    steps_by_task, completions = load_trace(
        steps_path=args.steps,
        completions_path=args.completions,
        method=args.method,
    )
    bucket_sizes = [int(x) for x in args.bucket_sizes.split(",") if x.strip()]
    records = replay_real_shape_forward(
        tokenizer=tokenizer,
        target=target,
        steps_by_task=steps_by_task,
        completions=completions,
        method=args.method,
        max_steps=args.max_steps,
        chat_template=args.chat_template,
        device=device,
        bucket_pad=args.bucket_pad,
        bucket_sizes=bucket_sizes,
        compile_model=args.enable_torch_compile,
    )
    aggregate = aggregate_records(records)
    write_report(records=records, aggregate=aggregate, output_dir=Path(args.output_dir), args=args)
    print((Path(args.output_dir) / "report.md").read_text())


if __name__ == "__main__":
    main()
