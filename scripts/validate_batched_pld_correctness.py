#!/usr/bin/env python3
"""Deterministic correctness validation for continuous-batched PLD.

This harness compares sequential ``blazedit_pld_w128_n10`` against the
continuous-batched scheduler under deterministic settings.  It is intentionally
separate from the timing harness so paper-grade correctness checks can run in
fp32/eager mode without changing the fast bf16/SDPA benchmark path.
"""

from __future__ import annotations

import argparse
import json
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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from asts.humaneval import load_problems_from_jsonl
from scripts.benchmark_real_shape_forward import model_dtype_arg
from scripts.run_batched_pld_eval import (
    _parse_ints,
    resolve_bucket_sizes,
    run_batched_scheduler,
    run_sequential_baseline,
)
from scripts.run_eagle_eval import _load_model


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


def _finish_reason(tokens: list[int], eos_token_ids: list[int], max_new_tokens: int) -> str:
    if any(int(t) in eos_token_ids for t in tokens):
        return "eos"
    if len(tokens) >= max_new_tokens:
        return "max_new_tokens"
    return "stopped"


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
    vals = sorted(float(x) for x in values)
    def pct(p: float) -> float:
        if len(vals) == 1:
            return vals[0]
        idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * p))))
        return vals[idx]

    return {
        "mean": float(statistics.fmean(vals)),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
    }


def _token_context(tokenizer, tokens: list[int], index: int | None, radius: int = 16) -> dict[str, Any]:
    if index is None:
        index = 0
    start = max(0, index - radius)
    end = min(len(tokens), index + radius + 1)
    return {
        "start": start,
        "end": end,
        "token_ids": [int(x) for x in tokens[start:end]],
        "decoded": tokenizer.decode(tokens[start:end], skip_special_tokens=False),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-jsonl", required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--dtype", default="fp32")
    parser.add_argument("--attn", default="eager")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--chat-template", default="none")
    parser.add_argument("--batch-sizes", default="1,4,8")
    parser.add_argument("--active-pool-size", type=int, default=0)
    parser.add_argument("--bucket-sizes", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--bucket-policy", choices=["custom", "default", "fine", "single"], default="custom")
    parser.add_argument("--refill-policy", choices=["continuous", "no_refill"], default="continuous")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--write-audit-traces", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    sequential = run_sequential_baseline(
        problems=problems,
        tokenizer=tokenizer,
        target=target,
        max_new_tokens=args.max_new_tokens,
        eos_token_ids=eos_token_ids,
        chat_template=args.chat_template,
    )
    baseline_outputs: dict[str, list[int]] = sequential["outputs"]
    rows: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        pool = args.active_pool_size or max(8, batch_size * 4)
        audit_trace = (
            output_dir / f"batch{batch_size}_audit_trace.jsonl"
            if args.write_audit_traces
            else None
        )
        metrics, outputs = run_batched_scheduler(
            problems=problems,
            tokenizer=tokenizer,
            target=target,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos_token_ids,
            chat_template=args.chat_template,
            batch_size=batch_size,
            active_pool_size=pool,
            bucket_sizes=bucket_sizes,
            baseline_outputs=baseline_outputs,
            device=device,
            refill_policy=args.refill_policy,
            bucket_policy=args.bucket_policy,
            audit_trace_path=audit_trace,
        )
        mismatches = []
        finish_mismatch_count = 0
        decoded_mismatch_count = 0
        length_mismatch_count = 0
        length_deltas = []
        for prob in problems:
            base = baseline_outputs.get(prob.task_id, [])
            got = outputs.get(prob.task_id, [])
            length_deltas.append(len(got) - len(base))
            if len(base) != len(got):
                length_mismatch_count += 1
            base_text = tokenizer.decode(base, skip_special_tokens=False)
            got_text = tokenizer.decode(got, skip_special_tokens=False)
            if base_text != got_text:
                decoded_mismatch_count += 1
            if _finish_reason(base, eos_token_ids, args.max_new_tokens) != _finish_reason(
                got,
                eos_token_ids,
                args.max_new_tokens,
            ):
                finish_mismatch_count += 1
            if base != got and len(mismatches) < 20:
                first_diff = next((i for i, (a, b) in enumerate(zip(base, got)) if a != b), None)
                if first_diff is None and len(base) != len(got):
                    first_diff = min(len(base), len(got))
                baseline_token_id = (
                    int(base[first_diff])
                    if first_diff is not None and first_diff < len(base)
                    else None
                )
                batched_token_id = (
                    int(got[first_diff])
                    if first_diff is not None and first_diff < len(got)
                    else None
                )
                mismatches.append(
                    {
                        "task_id": prob.task_id,
                        "baseline_len": len(base),
                        "batched_len": len(got),
                        "first_diff_index": first_diff,
                        "baseline_token_id": baseline_token_id,
                        "batched_token_id": batched_token_id,
                        "baseline_finish": _finish_reason(base, eos_token_ids, args.max_new_tokens),
                        "batched_finish": _finish_reason(got, eos_token_ids, args.max_new_tokens),
                        "baseline_decoded_snippet": base_text[
                            max(0, (first_diff or 0) - 24) : (first_diff or 0) + 160
                        ],
                        "batched_decoded_snippet": got_text[
                            max(0, (first_diff or 0) - 24) : (first_diff or 0) + 160
                        ],
                        "baseline_token_context": _token_context(tokenizer, base, first_diff),
                        "batched_token_context": _token_context(tokenizer, got, first_diff),
                    }
                )
        row = {
            "batch_size": batch_size,
            "active_pool_size": pool,
            "matches": metrics.output_match_count,
            "mismatches": metrics.output_mismatch_count,
            "decoded_output_matches": len(problems) - decoded_mismatch_count,
            "decoded_output_mismatches": decoded_mismatch_count,
            "exact_match": metrics.output_mismatch_count == 0,
            "finish_mismatch_count": finish_mismatch_count,
            "generated_length_matches": len(problems) - length_mismatch_count,
            "generated_length_mismatches": length_mismatch_count,
            "length_delta_summary": _summarize([float(x) for x in length_deltas]),
            "metrics": asdict(metrics),
            "mismatch_examples": mismatches,
            "audit_trace": str(audit_trace or ""),
        }
        rows.append(row)

    payload = {
        "args": vars(args),
        "sequential": {k: v for k, v in sequential.items() if k != "outputs"},
        "rows": rows,
        "all_exact": all(r["exact_match"] for r in rows),
        "correctness_note": (
            "This is the deterministic validation path. Exact paper claims should use "
            "fp32/eager results; bf16/SDPA timing runs may drift on near ties."
        ),
    }
    (output_dir / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    seq_tps = float(sequential["tokens_per_sec"])
    lines = [
        "# Batched PLD Deterministic Correctness",
        "",
        f"target: `{args.target}`  dtype: `{args.dtype}`  attn: `{args.attn}`  n: `{args.n}`",
        "",
        "| batch | active pool | token-id matches | decoded matches | length matches | finish mismatches | tok/s | speedup | exact |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        m = row["metrics"]
        lines.append(
            f"| {row['batch_size']} | {row['active_pool_size']} | "
            f"{row['matches']}/{row['matches'] + row['mismatches']} | "
            f"{row['decoded_output_matches']}/{row['decoded_output_matches'] + row['decoded_output_mismatches']} | "
            f"{row['generated_length_matches']}/{row['generated_length_matches'] + row['generated_length_mismatches']} | "
            f"{row['finish_mismatch_count']} | {m['generated_tokens_per_sec']:.1f} | "
            f"{m['generated_tokens_per_sec'] / max(1e-9, seq_tps):.3f} | "
            f"{'yes' if row['exact_match'] else 'no'} |"
        )
    lines.append("")
    lines.append(f"Overall exact pass: `{payload['all_exact']}`.")
    lines.append("")
    lines.append(payload["correctness_note"])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print((output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
