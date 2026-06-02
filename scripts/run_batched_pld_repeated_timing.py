#!/usr/bin/env python3
"""Repeated timing for the frozen Continuous Batched PLD configuration."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np

from asts.humaneval import load_problems_from_jsonl
from scripts.benchmark_real_shape_forward import model_dtype_arg
from scripts.run_batched_pld_eval import (
    FINAL_CONFIG_NAME,
    FINAL_METHOD_NAME,
    _parse_ints,
    resolve_bucket_sizes,
    run_batched_scheduler,
    run_sequential_baseline,
)
from scripts.run_eagle_eval import _load_model


def _eos_ids(tokenizer, target) -> list[int]:
    eos_token_ids: list[int] = []
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos_token_ids.append(int(tokenizer.eos_token_id))
    raw = getattr(getattr(target, "config", None), "eos_token_id", None)
    if raw is not None:
        if isinstance(raw, list):
            eos_token_ids.extend(int(x) for x in raw)
        else:
            eos_token_ids.append(int(raw))
    return sorted(set(eos_token_ids))


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "ci95": 0.0,
        }
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(std),
        "min": float(min(values)),
        "max": float(max(values)),
        "ci95": float(1.96 * std / math.sqrt(len(values))) if len(values) > 1 else 0.0,
    }


def _set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _cuda_memory_hygiene(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-jsonl", required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--chat-template", default="none")
    parser.add_argument("--batch-sizes", default="2,4,8")
    parser.add_argument("--active-pool-size", type=int, default=32)
    parser.add_argument("--bucket-policy", choices=["default", "fine", "single", "custom"], default="default")
    parser.add_argument("--bucket-sizes", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--refill-policy", choices=["continuous", "no_refill"], default="continuous")
    parser.add_argument("--audit-batch-size", type=int, default=8)
    parser.add_argument("--write-audit-trace", action="store_true")
    parser.add_argument(
        "--memory-hygiene",
        action="store_true",
        help="Run gc/empty_cache between large phases and sequential tasks. Intended for fp32/eager OOM debugging.",
    )
    parser.add_argument(
        "--empty-cache-every",
        type=int,
        default=1,
        help="When --memory-hygiene is set, empty CUDA cache every N sequential tasks.",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=0,
        help="Chunk long prompt prefill to avoid full-prefix eager attention OOM. 0 keeps the historical path.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        _set_deterministic(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    tokenizer, target = _load_model(
        args.target,
        dtype=model_dtype_arg(args.dtype),
        attn_impl=args.attn,
    )
    target.eval()
    eos_token_ids = _eos_ids(tokenizer, target)
    problems = load_problems_from_jsonl(args.problem_jsonl, n=args.n)
    batch_sizes = _parse_ints(args.batch_sizes)
    bucket_sizes = resolve_bucket_sizes(args.bucket_policy, args.bucket_sizes)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        print(f"[repeat {repeat}] sequential blazedit_pld_w128_n10", flush=True)
        sequential = run_sequential_baseline(
            problems=problems,
            tokenizer=tokenizer,
            target=target,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos_token_ids,
            chat_template=args.chat_template,
            memory_hygiene=args.memory_hygiene,
            empty_cache_every=args.empty_cache_every,
            prefill_chunk_size=args.prefill_chunk_size,
        )
        seq_row = {
            "run_id": repeat,
            "method": "blazedit_pld_w128_n10",
            "batch_size": 1,
            "active_pool_size": 1,
            "tok_s": float(sequential["tokens_per_sec"]),
            "speedup_vs_same_run_sequential": 1.0,
            "verifier_forwards": int(sequential["steps"]),
            "verifier_forward_reduction": 0.0,
            "total_generated_tokens": int(sequential["tokens"]),
            "wall_ms": float(sequential["wall_ms"]),
            "scheduler_overhead_ms": 0.0,
            "pld_lookup_ms": 0.0,
            "bucket_forward_ms": {},
            "memory_peak_gb": float(sequential.get("memory_peak_gb", 0.0)),
            "output_match_count": args.n,
            "output_mismatch_count": 0,
        }
        rows.append(seq_row)
        seq_tps = float(sequential["tokens_per_sec"])
        if args.memory_hygiene:
            _cuda_memory_hygiene(device)
        for batch_size in batch_sizes:
            print(
                f"[repeat {repeat}] {FINAL_METHOD_NAME} batch={batch_size} "
                f"pool={args.active_pool_size} bucket={args.bucket_policy} refill={args.refill_policy}",
                flush=True,
            )
            audit_trace = None
            if args.write_audit_trace and repeat == 0 and batch_size == args.audit_batch_size:
                audit_trace = output_dir / f"repeat{repeat}_batch{batch_size}_audit_trace.jsonl"
            metrics, _outputs = run_batched_scheduler(
                problems=problems,
                tokenizer=tokenizer,
                target=target,
                max_new_tokens=args.max_new_tokens,
                eos_token_ids=eos_token_ids,
                chat_template=args.chat_template,
                batch_size=batch_size,
                active_pool_size=args.active_pool_size,
                bucket_sizes=bucket_sizes,
                baseline_outputs=sequential["outputs"],
                device=device,
                refill_policy=args.refill_policy,
                bucket_policy=args.bucket_policy,
                audit_trace_path=audit_trace,
                prefill_chunk_size=args.prefill_chunk_size,
            )
            del _outputs
            row = asdict(metrics)
            row.update(
                {
                    "run_id": repeat,
                    "method": FINAL_METHOD_NAME,
                    "tok_s": float(metrics.generated_tokens_per_sec),
                    "speedup_vs_same_run_sequential": (
                        float(metrics.generated_tokens_per_sec) / max(1e-9, seq_tps)
                    ),
                    "verifier_forward_reduction": (
                        1.0 - float(metrics.verifier_forwards) / max(1, int(sequential["steps"]))
                    ),
                    "total_generated_tokens": int(metrics.total_new_tokens),
                }
            )
            rows.append(row)
            if args.memory_hygiene:
                _cuda_memory_hygiene(device)
        del sequential
        if args.memory_hygiene:
            _cuda_memory_hygiene(device)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = f"{row['method']}_b{row['batch_size']}"
        grouped.setdefault(key, []).append(row)
    summary: dict[str, dict[str, Any]] = {}
    for key, vals in grouped.items():
        summary[key] = {
            "method": vals[0]["method"],
            "batch_size": vals[0]["batch_size"],
            "active_pool_size": vals[0].get("active_pool_size", 0),
            "tok_s": _summary([float(v["tok_s"]) for v in vals]),
            "speedup": _summary([float(v["speedup_vs_same_run_sequential"]) for v in vals]),
            "verifier_forwards": _summary([float(v["verifier_forwards"]) for v in vals]),
            "verifier_forward_reduction": _summary(
                [100.0 * float(v["verifier_forward_reduction"]) for v in vals]
            ),
            "memory_peak_gb": _summary([float(v.get("memory_peak_gb", 0.0)) for v in vals]),
            "output_match_count": _summary([float(v.get("output_match_count", 0)) for v in vals]),
        }

    payload = {
        "args": vars(args),
        "config_name": FINAL_CONFIG_NAME,
        "method_name": FINAL_METHOD_NAME,
        "rows": rows,
        "summary": summary,
    }
    (output_dir / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Continuous Batched PLD Repeated Timing",
        "",
        f"config: `{FINAL_CONFIG_NAME}`  n: `{args.n}`  repeats: `{args.repeats}`",
        "",
        "| method | batch | tok/s mean ± std | speedup mean ± std | verifier forwards mean | forward reduction mean | peak GB mean | output matches mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(summary, key=lambda k: summary[k]["batch_size"]):
        s = summary[key]
        lines.append(
            f"| {s['method']} | {s['batch_size']} | "
            f"{s['tok_s']['mean']:.1f} ± {s['tok_s']['std']:.1f} | "
            f"{s['speedup']['mean']:.3f} ± {s['speedup']['std']:.3f} | "
            f"{s['verifier_forwards']['mean']:.1f} | "
            f"{s['verifier_forward_reduction']['mean']:.1f}% | "
            f"{s['memory_peak_gb']['mean']:.2f} | "
            f"{s['output_match_count']['mean']:.1f} |"
        )
    lines.append("")
    lines.append("95% CIs are available in `report.json` under each summary field's `ci95`.")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print((output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
