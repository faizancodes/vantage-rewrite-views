#!/usr/bin/env python3
"""Ablations for Continuous Batched PLD Verification."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from asts.humaneval import load_problems_from_jsonl
from scripts.benchmark_real_shape_forward import model_dtype_arg
from scripts.run_batched_pld_eval import (
    _parse_ints,
    BUCKET_POLICIES,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-jsonl", required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--chat-template", default="none")
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--active-pool-sizes", default="8,16,32")
    parser.add_argument("--bucket-policies", default="default,fine")
    parser.add_argument("--refill-policies", default="continuous,no_refill")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _padding_pct(row: dict[str, Any]) -> float:
    real = float(row.get("real_verified_tokens", 0))
    pad = float(row.get("input_padding_waste_tokens", 0))
    return 100.0 * pad / max(1.0, real + pad)


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
    eos_token_ids = _eos_ids(tokenizer, target)
    problems = load_problems_from_jsonl(args.problem_jsonl, n=args.n)
    batch_sizes = _parse_ints(args.batch_sizes)
    active_pool_sizes = _parse_ints(args.active_pool_sizes)
    bucket_policies = [x.strip() for x in args.bucket_policies.split(",") if x.strip()]
    refill_policies = [x.strip() for x in args.refill_policies.split(",") if x.strip()]

    sequential = run_sequential_baseline(
        problems=problems,
        tokenizer=tokenizer,
        target=target,
        max_new_tokens=args.max_new_tokens,
        eos_token_ids=eos_token_ids,
        chat_template=args.chat_template,
    )
    seq_tps = float(sequential["tokens_per_sec"])
    rows: list[dict[str, Any]] = []

    for batch_size in batch_sizes:
        for active_pool in active_pool_sizes:
            if active_pool < batch_size:
                continue
            for bucket_policy in bucket_policies:
                if bucket_policy not in BUCKET_POLICIES:
                    raise SystemExit(f"unknown bucket policy: {bucket_policy}")
                for refill_policy in refill_policies:
                    config_id = (
                        f"b{batch_size}_pool{active_pool}_"
                        f"{bucket_policy}_{refill_policy}"
                    )
                    print(f"[ablation] {config_id}", flush=True)
                    try:
                        metrics, _outputs = run_batched_scheduler(
                            problems=problems,
                            tokenizer=tokenizer,
                            target=target,
                            max_new_tokens=args.max_new_tokens,
                            eos_token_ids=eos_token_ids,
                            chat_template=args.chat_template,
                            batch_size=batch_size,
                            active_pool_size=active_pool,
                            bucket_sizes=BUCKET_POLICIES[bucket_policy],
                            baseline_outputs=sequential["outputs"],
                            device=device,
                            refill_policy=refill_policy,
                            bucket_policy=bucket_policy,
                        )
                        row = asdict(metrics)
                        row["config_id"] = config_id
                        row["speedup_vs_sequential"] = (
                            row["generated_tokens_per_sec"] / max(1e-9, seq_tps)
                        )
                        row["verifier_forward_reduction"] = (
                            1.0 - row["verifier_forwards"] / max(1, int(sequential["steps"]))
                        )
                        row["input_padding_waste_pct"] = _padding_pct(row)
                    except Exception as exc:
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        row = {
                            "config_id": config_id,
                            "batch_size": batch_size,
                            "active_pool_size": active_pool,
                            "bucket_policy": bucket_policy,
                            "refill_policy": refill_policy,
                            "error": f"{type(exc).__name__}: {exc}",
                            "speedup_vs_sequential": 0.0,
                        }
                        print(f"[ablation] {config_id} failed: {row['error']}", flush=True)
                    rows.append(row)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args),
        "sequential": {k: v for k, v in sequential.items() if k != "outputs"},
        "rows": rows,
    }
    (out / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Continuous Batched PLD Ablation",
        "",
        f"baseline sequential tok/s: `{seq_tps:.1f}`  steps: `{int(sequential['steps'])}`",
        "",
        "| config | tok/s | speedup | forwards | forward reduction | output matches | input pad % | peak GB | error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda r: r.get("speedup_vs_sequential", 0.0), reverse=True):
        matches = f"{row.get('output_match_count', 0)}/{row.get('output_match_count', 0) + row.get('output_mismatch_count', 0)}"
        lines.append(
            f"| {row['config_id']} | {row.get('generated_tokens_per_sec', 0.0):.1f} | "
            f"{row.get('speedup_vs_sequential', 0.0):.3f} | {row.get('verifier_forwards', 0)} | "
            f"{100.0 * row.get('verifier_forward_reduction', 0.0):.1f}% | {matches} | "
            f"{row.get('input_padding_waste_pct', 0.0):.1f}% | "
            f"{row.get('memory_peak_gb', 0.0):.2f} | {row.get('error', '')} |"
        )
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print((out / "report.md").read_text())


if __name__ == "__main__":
    main()
