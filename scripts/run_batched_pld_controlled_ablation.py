#!/usr/bin/env python3
"""Controlled repeated ablations for Continuous Batched PLD Verification.

This replaces the older single-run diagnostic ablation with a fixed controlled
configuration set.  Each repeat runs its own sequential PLD baseline, and every
batched row in that repeat is compared against that repeat's baseline outputs.
The script writes raw rows plus aggregate summaries; it never fills in missing
measurements with fabricated values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from asts.humaneval import load_problems_from_jsonl
from scripts.benchmark_real_shape_forward import model_dtype_arg
from scripts.run_batched_pld_eval import (
    BUCKET_POLICIES,
    FINAL_METHOD_NAME,
    run_batched_scheduler,
    run_sequential_baseline,
)
from scripts.run_eagle_eval import _load_model


SEQUENTIAL_METHOD = "blazedit_pld_w128_n10"


@dataclass(frozen=True)
class AblationConfig:
    config_id: str
    batch_size: int
    active_pool_size: int
    bucket_policy: str
    refill_policy: str


REQUIRED_CONFIGS: tuple[AblationConfig, ...] = (
    AblationConfig("b2_pool32_default_continuous", 2, 32, "default", "continuous"),
    AblationConfig("b4_pool32_default_continuous", 4, 32, "default", "continuous"),
    AblationConfig("b8_pool32_default_continuous", 8, 32, "default", "continuous"),
    AblationConfig("b8_pool8_default_continuous", 8, 8, "default", "continuous"),
    AblationConfig("b8_pool16_default_continuous", 8, 16, "default", "continuous"),
    AblationConfig("b8_pool32_default_no_refill", 8, 32, "default", "no_refill"),
    AblationConfig("b8_pool32_fine_continuous", 8, 32, "fine", "continuous"),
    AblationConfig("b8_pool32_single_continuous", 8, 32, "single", "continuous"),
)

NUMERIC_SUMMARY_FIELDS = (
    "tok_s",
    "speedup_vs_same_run_sequential",
    "speedup_vs_sequential_mean",
    "verifier_forwards",
    "verifier_forward_reduction_pct",
    "decode_steps",
    "wall_ms",
    "total_forward_ms",
    "total_prefill_ms",
    "scheduler_overhead_ms",
    "pld_lookup_ms",
    "input_padding_waste_pct",
    "memory_peak_gb",
    "output_match_count",
    "output_mismatch_count",
)


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
    finite = [float(x) for x in values if math.isfinite(float(x))]
    if not finite:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "ci95": 0.0}
    std = statistics.stdev(finite) if len(finite) > 1 else 0.0
    return {
        "n": len(finite),
        "mean": float(statistics.fmean(finite)),
        "std": float(std),
        "min": float(min(finite)),
        "max": float(max(finite)),
        "ci95": float(1.96 * std / math.sqrt(len(finite))) if len(finite) > 1 else 0.0,
    }


def _padding_pct(row: dict[str, Any]) -> float:
    real = float(row.get("real_verified_tokens", 0.0) or 0.0)
    pad = float(row.get("input_padding_waste_tokens", 0.0) or 0.0)
    return 100.0 * pad / max(1.0, real + pad)


def _sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ""


def _source_metadata(args: argparse.Namespace) -> dict[str, Any]:
    source_files = [
        "scripts/run_batched_pld_controlled_ablation.py",
        "scripts/run_batched_pld_ablation.py",
        "scripts/run_batched_pld_repeated_timing.py",
        "scripts/run_batched_pld_eval.py",
        "vantage_runtime_debian_app.py",
    ]
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "cwd": str(ROOT),
        "command": " ".join(sys.argv),
        "git_revision": _git(["rev-parse", "HEAD"]),
        "git_status_short": _git(["status", "--short"]),
        "args": vars(args),
        "source_files": {
            rel: {"path": str(ROOT / rel), "sha256": _sha256(ROOT / rel)}
            for rel in source_files
        },
        "vantage_launcher_note": (
            "No launcher edit is required for this harness. If Modal launch support is "
            "needed later, mirror run_batched_pld_repeated_timing_job with the command "
            "scripts/run_batched_pld_controlled_ablation.py and out_dir "
            "/data/batched_pld_controlled_ablation/{version}."
        ),
        "planned_config_set": [
            {
                "config_id": "seq",
                "method": SEQUENTIAL_METHOD,
                "batch_size": 1,
                "active_pool_size": 1,
                "bucket_policy": "n/a",
                "refill_policy": "n/a",
            },
            *[asdict(cfg) for cfg in REQUIRED_CONFIGS],
        ],
    }


def _make_sequential_row(repeat: int, sequential: dict[str, Any], *, n: int) -> dict[str, Any]:
    return {
        "repeat": repeat,
        "config_id": "seq",
        "method": SEQUENTIAL_METHOD,
        "batch_size": 1,
        "active_pool_size": 1,
        "bucket_policy": "n/a",
        "refill_policy": "n/a",
        "n_tasks": n,
        "status": "success",
        "error": "",
        "tok_s": float(sequential["tokens_per_sec"]),
        "speedup_vs_same_run_sequential": 1.0,
        "speedup_vs_sequential_mean": 1.0,
        "verifier_forwards": int(sequential["steps"]),
        "verifier_forward_reduction_pct": 0.0,
        "decode_steps": int(sequential["steps"]),
        "total_generated_tokens": int(sequential["tokens"]),
        "wall_ms": float(sequential["wall_ms"]),
        "total_forward_ms": 0.0,
        "total_prefill_ms": 0.0,
        "scheduler_overhead_ms": 0.0,
        "pld_lookup_ms": 0.0,
        "input_padding_waste_pct": 0.0,
        "memory_peak_gb": 0.0,
        "output_match_count": n,
        "output_mismatch_count": 0,
    }


def _make_batched_row(
    *,
    repeat: int,
    cfg: AblationConfig,
    metrics: dict[str, Any],
    seq_tps: float,
    seq_steps: int,
) -> dict[str, Any]:
    row = dict(metrics)
    row.update(
        {
            "repeat": repeat,
            "config_id": cfg.config_id,
            "method": FINAL_METHOD_NAME,
            "batch_size": cfg.batch_size,
            "active_pool_size": cfg.active_pool_size,
            "bucket_policy": cfg.bucket_policy,
            "refill_policy": cfg.refill_policy,
            "status": "success" if not row.get("error") else "failed",
            "tok_s": float(row.get("generated_tokens_per_sec", 0.0) or 0.0),
            "speedup_vs_same_run_sequential": (
                float(row.get("generated_tokens_per_sec", 0.0) or 0.0) / max(1e-9, seq_tps)
            ),
            "verifier_forward_reduction_pct": (
                100.0
                * (
                    1.0
                    - float(row.get("verifier_forwards", 0.0) or 0.0)
                    / max(1, int(seq_steps))
                )
            ),
            "total_generated_tokens": int(row.get("total_new_tokens", 0) or 0),
            "input_padding_waste_pct": _padding_pct(row),
        }
    )
    return row


def _make_error_row(
    *,
    repeat: int,
    cfg: AblationConfig,
    exc: Exception,
    n: int,
    seq_tps: float,
    seq_steps: int,
) -> dict[str, Any]:
    return {
        "repeat": repeat,
        "config_id": cfg.config_id,
        "method": FINAL_METHOD_NAME,
        "batch_size": cfg.batch_size,
        "active_pool_size": cfg.active_pool_size,
        "bucket_policy": cfg.bucket_policy,
        "refill_policy": cfg.refill_policy,
        "n_tasks": n,
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
        "tok_s": 0.0,
        "speedup_vs_same_run_sequential": 0.0,
        "verifier_forwards": 0,
        "verifier_forward_reduction_pct": 0.0,
        "decode_steps": 0,
        "total_generated_tokens": 0,
        "wall_ms": 0.0,
        "total_forward_ms": 0.0,
        "total_prefill_ms": 0.0,
        "scheduler_overhead_ms": 0.0,
        "pld_lookup_ms": 0.0,
        "input_padding_waste_pct": 0.0,
        "memory_peak_gb": 0.0,
        "output_match_count": 0,
        "output_mismatch_count": n,
        "same_run_sequential_tok_s": seq_tps,
        "same_run_sequential_steps": seq_steps,
    }


def _attach_mean_speedups(rows: list[dict[str, Any]]) -> None:
    seq_values = [float(r["tok_s"]) for r in rows if r.get("config_id") == "seq" and r.get("status") == "success"]
    seq_mean = statistics.fmean(seq_values) if seq_values else 0.0
    for row in rows:
        if seq_mean:
            row["speedup_vs_sequential_mean"] = float(row.get("tok_s", 0.0) or 0.0) / seq_mean
        else:
            row["speedup_vs_sequential_mean"] = 0.0


def _build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["config_id"]), []).append(row)

    summary: dict[str, Any] = {}
    for config_id, vals in grouped.items():
        exemplar = vals[0]
        successes = [v for v in vals if v.get("status") == "success"]
        failures = [v for v in vals if v.get("status") != "success"]
        fields = {
            field: _summary([float(v.get(field, 0.0) or 0.0) for v in successes])
            for field in NUMERIC_SUMMARY_FIELDS
        }
        summary[config_id] = {
            "config_id": config_id,
            "method": exemplar.get("method", ""),
            "batch_size": exemplar.get("batch_size", 0),
            "active_pool_size": exemplar.get("active_pool_size", 0),
            "bucket_policy": exemplar.get("bucket_policy", ""),
            "refill_policy": exemplar.get("refill_policy", ""),
            "n_repeats_requested": len(vals),
            "n_success": len(successes),
            "n_failed": len(failures),
            "fields": fields,
            "errors": [v.get("error", "") for v in failures if v.get("error")],
        }
    return summary


def _write_outputs(output_dir: Path, metadata: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _attach_mean_speedups(rows)
    summary = _build_summary(rows)
    payload = {
        "metadata": metadata,
        "rows": rows,
        "summary": summary,
    }
    (output_dir / "raw_rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Controlled Continuous Batched PLD Ablation",
        "",
        f"target: `{metadata['args']['target']}`  dtype: `{metadata['args']['dtype']}`  "
        f"attn: `{metadata['args']['attn']}`  n: `{metadata['args']['n']}`  "
        f"repeats: `{metadata['args']['repeats']}`",
        "",
        "Raw per-repeat measurements are in `raw_rows.jsonl`. Summary statistics below use successful repeats only; failures are counted separately and reported as errors.",
        "",
        "| config | method | batch | pool | buckets | refill | ok/repeats | tok/s mean +/- std | speedup vs same-run seq | speedup vs seq mean | forwards mean | forward reduction mean | output matches mean | peak GB mean |",
        "|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = ["seq", *(cfg.config_id for cfg in REQUIRED_CONFIGS)]
    for config_id in order:
        if config_id not in summary:
            continue
        item = summary[config_id]
        fields = item["fields"]
        lines.append(
            f"| {config_id} | {item['method']} | {item['batch_size']} | "
            f"{item['active_pool_size']} | {item['bucket_policy']} | {item['refill_policy']} | "
            f"{item['n_success']}/{item['n_repeats_requested']} | "
            f"{fields['tok_s']['mean']:.1f} +/- {fields['tok_s']['std']:.1f} | "
            f"{fields['speedup_vs_same_run_sequential']['mean']:.3f} +/- {fields['speedup_vs_same_run_sequential']['std']:.3f} | "
            f"{fields['speedup_vs_sequential_mean']['mean']:.3f} +/- {fields['speedup_vs_sequential_mean']['std']:.3f} | "
            f"{fields['verifier_forwards']['mean']:.1f} | "
            f"{fields['verifier_forward_reduction_pct']['mean']:.1f}% | "
            f"{fields['output_match_count']['mean']:.1f} | "
            f"{fields['memory_peak_gb']['mean']:.2f} |"
        )
        for error in item.get("errors", []):
            lines.append(f"<!-- {config_id} error: {error} -->")
    lines.extend(
        [
            "",
            "## Source Metadata",
            "",
            f"- git revision: `{metadata.get('git_revision') or 'unknown'}`",
            f"- generated at UTC: `{metadata['generated_at_utc']}`",
            f"- launcher note: {metadata['vantage_launcher_note']}",
            "",
            "95% confidence intervals are available in `summary.json` under each field's `ci95`.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((output_dir / "summary.md").read_text(encoding="utf-8"))


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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write only the planned config metadata; do not load models or run benchmarks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    for cfg in REQUIRED_CONFIGS:
        if cfg.bucket_policy not in BUCKET_POLICIES:
            raise SystemExit(f"unknown bucket policy in required config: {cfg.bucket_policy}")
        if cfg.active_pool_size < cfg.batch_size:
            raise SystemExit(f"active pool smaller than batch in required config: {cfg.config_id}")

    output_dir = Path(args.output_dir)
    metadata = _source_metadata(args)
    if args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "plan.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(metadata["planned_config_set"], indent=2, sort_keys=True))
        return

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

    rows: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        print(f"[repeat {repeat}] seq {SEQUENTIAL_METHOD}", flush=True)
        sequential = run_sequential_baseline(
            problems=problems,
            tokenizer=tokenizer,
            target=target,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos_token_ids,
            chat_template=args.chat_template,
        )
        rows.append(_make_sequential_row(repeat, sequential, n=len(problems)))
        seq_tps = float(sequential["tokens_per_sec"])
        seq_steps = int(sequential["steps"])

        for cfg in REQUIRED_CONFIGS:
            print(f"[repeat {repeat}] {cfg.config_id}", flush=True)
            try:
                metrics, _outputs = run_batched_scheduler(
                    problems=problems,
                    tokenizer=tokenizer,
                    target=target,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_ids=eos_token_ids,
                    chat_template=args.chat_template,
                    batch_size=cfg.batch_size,
                    active_pool_size=cfg.active_pool_size,
                    bucket_sizes=BUCKET_POLICIES[cfg.bucket_policy],
                    baseline_outputs=sequential["outputs"],
                    device=device,
                    refill_policy=cfg.refill_policy,
                    bucket_policy=cfg.bucket_policy,
                )
                rows.append(
                    _make_batched_row(
                        repeat=repeat,
                        cfg=cfg,
                        metrics=asdict(metrics),
                        seq_tps=seq_tps,
                        seq_steps=seq_steps,
                    )
                )
            except Exception as exc:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"[repeat {repeat}] {cfg.config_id} failed: {type(exc).__name__}: {exc}", flush=True)
                rows.append(
                    _make_error_row(
                        repeat=repeat,
                        cfg=cfg,
                        exc=exc,
                        n=len(problems),
                        seq_tps=seq_tps,
                        seq_steps=seq_steps,
                    )
                )

    _write_outputs(output_dir, metadata, rows)


if __name__ == "__main__":
    main()
