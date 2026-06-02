#!/usr/bin/env python3
"""PyTorch-profiler breakdown for real-shape PLD target forwards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.profiler import ProfilerActivity, profile

from scripts.benchmark_real_shape_forward import (
    aggregate_records,
    load_trace,
    model_dtype_arg,
    replay_real_shape_forward,
)
from scripts.run_eagle_eval import _load_model


def _category(name: str) -> str:
    low = name.lower()
    if "attention" in low or "flash" in low or "fmha" in low or "scaled_dot_product" in low:
        return "attention"
    if "matmul" in low or "mm" in low or "addmm" in low or "linear" in low or "gemm" in low:
        return "linear_mlp_lm_head"
    if "rms" in low or "norm" in low:
        return "norm"
    if "rotary" in low or "rope" in low or "cos" in low or "sin" in low:
        return "rotary_position"
    if "cache" in low or "cat" in low or "copy" in low or "slice" in low or "index" in low:
        return "kv_cache_or_copy"
    if "argmax" in low or "max" in low:
        return "argmax_sampling"
    if "to" in low or "cast" in low or "_to_copy" in low:
        return "dtype_device_cast"
    return "other"


def _profiler_rows(prof) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evt in prof.key_averages():
        cuda_total = float(getattr(evt, "cuda_time_total", 0.0) or getattr(evt, "device_time_total", 0.0) or 0.0)
        self_cuda = float(
            getattr(evt, "self_cuda_time_total", 0.0)
            or getattr(evt, "self_device_time_total", 0.0)
            or 0.0
        )
        rows.append(
            {
                "name": evt.key,
                "cpu_total_us": float(evt.cpu_time_total),
                "cuda_total_us": cuda_total,
                "self_cpu_us": float(evt.self_cpu_time_total),
                "self_cuda_us": self_cuda,
                "count": int(evt.count),
                "category": _category(evt.key),
            }
        )
    rows.sort(key=lambda r: r["cuda_total_us"], reverse=True)
    return rows


def _category_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, dict[str, float]] = {}
    total_cuda = sum(float(r["cuda_total_us"]) for r in rows)
    total_self_cuda = sum(float(r["self_cuda_us"]) for r in rows)
    for r in rows:
        cat = str(r["category"])
        cur = out.setdefault(
            cat,
            {
                "cuda_total_us": 0.0,
                "self_cuda_us": 0.0,
                "cpu_total_us": 0.0,
                "count": 0.0,
            },
        )
        cur["cuda_total_us"] += float(r["cuda_total_us"])
        cur["self_cuda_us"] += float(r["self_cuda_us"])
        cur["cpu_total_us"] += float(r["cpu_total_us"])
        cur["count"] += float(r["count"])
    return {
        k: {
            **v,
            "cuda_total_share": v["cuda_total_us"] / total_cuda if total_cuda > 0 else 0.0,
            "self_cuda_share": v["self_cuda_us"] / total_self_cuda if total_self_cuda > 0 else 0.0,
        }
        for k, v in sorted(out.items(), key=lambda kv: kv[1]["cuda_total_us"], reverse=True)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", required=True)
    parser.add_argument("--completions", required=True)
    parser.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--method", default="blazedit_pld_w128_n10")
    parser.add_argument("--chat-template", default="none")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

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
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode(), profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        records = replay_real_shape_forward(
            tokenizer=tokenizer,
            target=target,
            steps_by_task=steps_by_task,
            completions=completions,
            method=args.method,
            max_steps=args.max_steps,
            chat_template=args.chat_template,
            device=device,
        )
        for _ in records:
            prof.step()
    trace_path = out_dir / "trace.json"
    prof.export_chrome_trace(str(trace_path))
    rows = _profiler_rows(prof)
    categories = _category_summary(rows)
    aggregate = aggregate_records(records)
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=40)
    report = {
        "target": args.target,
        "dtype": args.dtype,
        "attn": args.attn,
        "max_steps": args.max_steps,
        "aggregate": aggregate,
        "categories": categories,
        "top_ops": rows[:80],
        "trace": str(trace_path),
        "profiler_table": table,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines: list[str] = []
    lines.append("# Real-Shape Forward Internals Profile\n")
    lines.append(f"target: `{args.target}`  dtype: `{args.dtype}`  attn: `{args.attn}`  steps: `{len(records)}`\n")
    lines.append("| category | cuda total ms | cuda total share | self cuda ms | self share |")
    lines.append("|---|---:|---:|---:|---:|")
    for cat, row in categories.items():
        lines.append(
            f"| {cat} | {row['cuda_total_us']/1000.0:.3f} | {row['cuda_total_share']:.1%} | "
            f"{row['self_cuda_us']/1000.0:.3f} | {row['self_cuda_share']:.1%} |"
        )
    lines.append("\n## Top CUDA Ops\n")
    lines.append("| op | category | cuda total ms | self cuda ms | calls |")
    lines.append("|---|---|---:|---:|---:|")
    for row in rows[:25]:
        lines.append(
            f"| `{row['name'][:80]}` | {row['category']} | {row['cuda_total_us']/1000.0:.3f} | "
            f"{row['self_cuda_us']/1000.0:.3f} | {row['count']} |"
        )
    lines.append("\n## Key Averages\n")
    lines.append("```text")
    lines.append(table)
    lines.append("```")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print((out_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
