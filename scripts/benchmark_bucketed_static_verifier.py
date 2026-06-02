#!/usr/bin/env python3
"""Verifier-only bucket/static-shape prototype for PLD target forwards."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from scripts.benchmark_real_shape_forward import (
    aggregate_records,
    load_trace,
    model_dtype_arg,
    replay_real_shape_forward,
    summarize,
)
from scripts.run_eagle_eval import _load_model


def _bool_arg(raw: str | bool) -> bool:
    if isinstance(raw, bool):
        return raw
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _make_prompt(tokenizer, target_tokens: int, device: torch.device) -> torch.Tensor:
    seed = (
        "def verify_bucket_shape(items):\n"
        "    total = 0\n"
        "    for item in items:\n"
        "        total += item.value\n"
        "    return total\n\n"
    )
    text = seed
    while len(tokenizer(text, add_special_tokens=False).input_ids) < target_tokens:
        text += seed
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[:, :target_tokens]
    return ids.to(device)


def _attempt_cuda_graph_buckets(
    *,
    tokenizer,
    target,
    bucket_sizes: list[int],
    device: torch.device,
    warmup: int = 10,
    iters: int = 50,
) -> dict[str, Any]:
    if device.type != "cuda":
        return {"enabled": False, "reason": "CUDA graphs require cuda device", "buckets": {}}
    results: dict[str, Any] = {"enabled": True, "buckets": {}}
    for bucket in bucket_sizes:
        try:
            prompt = _make_prompt(tokenizer, 1024 + bucket + 8, device)
            with torch.inference_mode():
                prefill = target(prompt[:, :1023], use_cache=True)
                cache = prefill.past_key_values
                static_input = prompt[:, 1023 : 1024 + bucket].contiguous()
                for _ in range(warmup):
                    _ = target(static_input, past_key_values=cache, use_cache=True)
                torch.cuda.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    _ = target(static_input, past_key_values=cache, use_cache=True)
                torch.cuda.synchronize(device)
                samples: list[float] = []
                for _ in range(iters):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    graph.replay()
                    end.record()
                    torch.cuda.synchronize(device)
                    samples.append(float(start.elapsed_time(end)))
            results["buckets"][str(bucket)] = {
                "captured": True,
                "mean_replay_ms": statistics.fmean(samples),
                "p50_replay_ms": statistics.median(samples),
                "p90_replay_ms": sorted(samples)[int(0.9 * (len(samples) - 1))],
            }
        except Exception as exc:  # CUDA graph support is expected to be fragile here.
            results["buckets"][str(bucket)] = {
                "captured": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", required=True)
    parser.add_argument("--completions", required=True)
    parser.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bucket-sizes", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--enable-cuda-graphs", default="false")
    parser.add_argument("--enable-torch-compile", default="false")
    parser.add_argument("--static-cache", default="false")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--method", default="blazedit_pld_w128_n10")
    parser.add_argument("--chat-template", default="none")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    bucket_sizes = [int(x) for x in args.bucket_sizes.split(",") if x.strip()]
    enable_compile = _bool_arg(args.enable_torch_compile)
    enable_graphs = _bool_arg(args.enable_cuda_graphs)
    static_cache = _bool_arg(args.static_cache)

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
    if static_cache:
        static_cache_report = {
            "requested": True,
            "supported": False,
            "reason": (
                "PLD verifier needs rollback/crop after rejected drafts. "
                "HF StaticCache has no cheap rollback path in this prototype."
            ),
        }
    else:
        static_cache_report = {"requested": False, "supported": False}

    records = replay_real_shape_forward(
        tokenizer=tokenizer,
        target=target,
        steps_by_task=steps_by_task,
        completions=completions,
        method=args.method,
        max_steps=args.max_steps,
        chat_template=args.chat_template,
        device=device,
        bucket_pad=True,
        bucket_sizes=bucket_sizes,
        compile_model=enable_compile,
    )
    aggregate = aggregate_records(records)
    graph_report = (
        _attempt_cuda_graph_buckets(
            tokenizer=tokenizer,
            target=target,
            bucket_sizes=bucket_sizes,
            device=device,
        )
        if enable_graphs
        else {"enabled": False, "reason": "disabled"}
    )
    memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0) if device.type == "cuda" else 0.0
    report = {
        "target": args.target,
        "dtype": args.dtype,
        "attn": args.attn,
        "bucket_sizes": bucket_sizes,
        "enable_cuda_graphs": enable_graphs,
        "enable_torch_compile": enable_compile,
        "static_cache": static_cache_report,
        "aggregate": aggregate,
        "cuda_graphs": graph_report,
        "gpu_max_memory_mb": memory_mb,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines: list[str] = []
    lines.append("# Bucketed Static Verifier Prototype\n")
    lines.append(
        f"target: `{args.target}`  dtype: `{args.dtype}`  attn: `{args.attn}`  "
        f"steps: `{aggregate['n_steps']}`\n"
    )
    lines.append(f"- torch compile: `{enable_compile}`")
    lines.append(f"- CUDA graphs requested: `{enable_graphs}`")
    lines.append(f"- static cache: `{static_cache_report}`")
    lines.append(f"- GPU max memory MB: `{memory_mb:.1f}`\n")
    lines.append("| metric | mean ms | p50 | p90 | p99 |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, row in [
        ("bucketed forward", aggregate["forward_ms"]),
        ("argmax/rejection", aggregate["argmax_rejection_ms"]),
        ("cache crop", aggregate["crop_ms"]),
    ]:
        lines.append(f"| {name} | {row['mean']:.3f} | {row['p50']:.3f} | {row['p90']:.3f} | {row['p99']:.3f} |")
    lines.append("\n## By Draft Bucket\n")
    lines.append("| bucket | n | mean forward ms | p90 |")
    lines.append("|---:|---:|---:|---:|")
    for bucket, row in aggregate["forward_ms_by_draft_bucket"].items():
        lines.append(f"| {bucket} | {row['n']} | {row['mean']:.3f} | {row['p90']:.3f} |")
    if enable_graphs:
        lines.append("\n## CUDA Graph Capture Probe\n")
        lines.append("| bucket | captured | mean replay ms / error |")
        lines.append("|---:|---|---|")
        for bucket, row in graph_report.get("buckets", {}).items():
            if row.get("captured"):
                msg = f"{row.get('mean_replay_ms', 0.0):.3f}"
            else:
                msg = str(row.get("error", "failed"))
            lines.append(f"| {bucket} | {row.get('captured')} | {msg} |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print((out_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
