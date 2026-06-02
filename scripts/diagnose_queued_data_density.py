#!/usr/bin/env python3
"""Diagnose why queued-use residual tensor collection is sparse."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_RAW_DIR = (
    ROOT
    / "artifacts/vantage_residual/phase5_data/raw/vantage_residual_phase5_collect_train500_v1"
)
DEFAULT_PACKAGED_DIR = ROOT / "artifacts/vantage_residual/phase5_data/queued_v1"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/vantage_residual/phase6_density/train500_v1"
DEFAULT_TABLE = ROOT / "artifacts/vantage_residual/tables/phase6_density_diagnostics.md"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _resolve_raw_dir(raw_dir: Path) -> Path:
    if raw_dir.exists():
        return raw_dir
    candidates = sorted(
        (ROOT / "artifacts/vantage_residual/phase5_data/raw").glob("*train500*")
    )
    if candidates:
        return candidates[-1]
    raise SystemExit(f"raw dir does not exist: {raw_dir}")


def _weak(row: dict[str, Any], *, threshold: int, weak_field: str) -> bool:
    if weak_field == "draft_len":
        return int(row.get("proposal_tokens") or row.get("k") or 0) <= threshold
    if weak_field == "accepted_len":
        return int(row.get("n_accepted_drafts") or 0) <= threshold
    raise ValueError(f"unsupported weak_field: {weak_field!r}")


def _pct(num: int | float, den: int | float) -> float:
    return 0.0 if not den else float(num) / float(den)


def _stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"mean": 0.0, "p50": 0, "p90": 0, "max": 0}
    vals = sorted(values)
    return {
        "mean": float(statistics.mean(vals)),
        "p50": int(statistics.median(vals)),
        "p90": int(vals[int(0.90 * (len(vals) - 1))]),
        "max": int(max(vals)),
    }


def _load_queued_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    import torch

    obj = torch.load(path, map_location="cpu")
    return obj if isinstance(obj, dict) else None


def _task_ids(payload: dict[str, Any] | None) -> list[str]:
    if not payload:
        return []
    raw = payload.get("task_id") or payload.get("task_ids") or []
    return [str(x) for x in raw]


def _task_count_map(payload: dict[str, Any] | None) -> Counter[str]:
    return Counter(_task_ids(payload))


def _diagnose(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.collect_pld_mtp_training_data import _steps_by_task

    raw_dir = _resolve_raw_dir(args.raw_dir)
    steps_path = raw_dir / "pld_trace/eval/steps.jsonl"
    completions_path = raw_dir / "pld_trace/eval/completions.jsonl"
    queued_path = raw_dir / "queued_raw.pt"
    if not steps_path.exists():
        raise SystemExit(f"missing steps JSONL: {steps_path}")
    if not completions_path.exists():
        raise SystemExit(f"missing completions JSONL: {completions_path}")

    steps_by_task = _steps_by_task(steps_path, method=args.method)
    queued = _load_queued_payload(queued_path)
    packaged = {
        split: _load_queued_payload(args.packaged_dir / f"{split}.pt")
        for split in ("train", "val", "test")
    }
    raw_task_counts = _task_count_map(queued)
    valid_tensor = queued.get("valid_queued_example") if queued else None
    valid_raw = int(valid_tensor.bool().sum().item()) if hasattr(valid_tensor, "bool") else len(_task_ids(queued))

    pair_counts = Counter()
    per_task_rows: list[dict[str, Any]] = []
    simulated_valid_all_next = 0
    simulated_valid_use_weak = 0
    simulated_valid_both_weak = 0
    derived_invalidation = Counter()
    first_20: list[dict[str, Any]] = []
    for task_id in sorted(steps_by_task):
        rows = steps_by_task[task_id]
        total_output_len = sum(max(1, int(r.get("n_emitted") or 0)) for r in rows)
        task_pair_counts = Counter()
        for i, create in enumerate(rows):
            if i == len(rows) - 1:
                derived_invalidation["no_next_step"] += 1
                continue
            use = rows[i + 1]
            task_pair_counts["adjacent_pairs"] += 1
            create_weak = _weak(create, threshold=args.trigger_threshold, weak_field=args.weak_field)
            use_weak = _weak(use, threshold=args.trigger_threshold, weak_field=args.weak_field)
            if create_weak:
                pair_counts["create_weak"] += 1
                task_pair_counts["create_weak"] += 1
            else:
                derived_invalidation["phase5_create_not_weak"] += 1
            if use_weak:
                pair_counts["use_weak"] += 1
                task_pair_counts["use_weak"] += 1
            else:
                derived_invalidation["phase5_use_not_weak_dropped_strong"] += int(create_weak)
            if create_weak and use_weak:
                pair_counts["both_weak"] += 1
                task_pair_counts["both_weak"] += 1
            if (not create_weak) and use_weak:
                pair_counts["create_strong_use_weak"] += 1
                task_pair_counts["create_strong_use_weak"] += 1
            create_start = int(create.get("_generated_start") or 0)
            use_start = int(use.get("_generated_start") or 0)
            create_emitted = max(1, int(create.get("n_emitted") or 0))
            create_accepted = int(create.get("n_accepted_drafts") or 0)
            if create_emitted <= create_accepted:
                derived_invalidation["no_baseline_next"] += 1
                continue
            if create_start + create_emitted != use_start:
                derived_invalidation["position_mismatch"] += 1
                continue
            if use_start + args.k > total_output_len:
                derived_invalidation["label_missing_horizon_k"] += 1
                continue
            simulated_valid_all_next += 1
            task_pair_counts["valid_all_next"] += 1
            if use_weak:
                simulated_valid_use_weak += 1
                task_pair_counts["valid_use_weak"] += 1
            if create_weak and use_weak:
                simulated_valid_both_weak += 1
                task_pair_counts["valid_both_weak"] += 1
        per_task_rows.append(
            {
                "task_id": task_id,
                "decode_steps": len(rows),
                "adjacent_pairs": int(task_pair_counts["adjacent_pairs"]),
                "create_weak": int(task_pair_counts["create_weak"]),
                "use_weak": int(task_pair_counts["use_weak"]),
                "both_weak": int(task_pair_counts["both_weak"]),
                "create_strong_use_weak": int(task_pair_counts["create_strong_use_weak"]),
                "valid_all_next": int(task_pair_counts["valid_all_next"]),
                "valid_use_weak": int(task_pair_counts["valid_use_weak"]),
                "valid_both_weak": int(task_pair_counts["valid_both_weak"]),
                "raw_examples_written": int(raw_task_counts.get(task_id, 0)),
            }
        )
    first_20 = per_task_rows[:20]

    raw_records_written = len(_task_ids(queued))
    total_steps = sum(len(rows) for rows in steps_by_task.values())
    adjacent_pairs = sum(max(0, len(rows) - 1) for rows in steps_by_task.values())
    examples_per_task = [row["raw_examples_written"] for row in per_task_rows]
    valid_per_task = examples_per_task
    packaged_counts = {}
    for split, payload in packaged.items():
        valid = payload.get("valid_queued_example") if payload else None
        packaged_counts[split] = {
            "exists": payload is not None,
            "examples": len(_task_ids(payload)),
            "valid_examples": int(valid.bool().sum().item()) if hasattr(valid, "bool") else len(_task_ids(payload)),
            "tasks": len(set(_task_ids(payload))),
        }

    if simulated_valid_all_next >= 5000 and raw_records_written < 1000:
        decision = "collector_filter_bug_or_overly_strict_filter"
        critical_answer = (
            "No: raw output is not exactly one example per task. Some tasks produce many examples. "
            "However, Phase 5 filtering requires both creation and use steps to be weak, which drops "
            "most adjacent decode-step pairs before hidden rows are written."
        )
    elif raw_records_written >= 5000 and raw_records_written >= 0.90 * max(1, simulated_valid_all_next):
        decision = "collector_density_fixed"
        critical_answer = (
            "The density issue is fixed for this raw collection. Hidden rows now track almost all "
            "valid adjacent decode-step pairs instead of only consecutive weak-step pairs."
        )
    else:
        decision = "inherent_or_pool_size_limited"
        critical_answer = (
            "The raw collection is not obviously dropping most valid adjacent pairs, but the available "
            "pool is still too small for the requested training gate."
        )
    return {
        "raw_dir": _rel(raw_dir),
        "packaged_dir": _rel(args.packaged_dir),
        "method": args.method,
        "trigger_threshold": args.trigger_threshold,
        "weak_field": args.weak_field,
        "tasks_processed": len(steps_by_task),
        "total_decode_steps_observed": total_steps,
        "adjacent_step_pairs": adjacent_pairs,
        "raw_records_written": raw_records_written,
        "valid_queued_examples_written": valid_raw,
        "invalid_queued_examples_written": 0,
        "note_on_invalid_examples": "Phase 5 collector omitted invalid/skipped pairs instead of writing invalid records.",
        "pair_counts": dict(pair_counts),
        "derived_invalidation_reason_counts": dict(derived_invalidation),
        "simulated_valid_if_queue_after_every_step": simulated_valid_all_next,
        "simulated_valid_if_use_step_weak_only": simulated_valid_use_weak,
        "simulated_valid_if_create_and_use_weak": simulated_valid_both_weak,
        "examples_per_task": _stats(examples_per_task),
        "valid_examples_per_task": _stats(valid_per_task),
        "raw_tasks_with_examples": len(raw_task_counts),
        "packaged_counts": packaged_counts,
        "verifier_events": total_steps,
        "hidden_captures_written": raw_records_written,
        "hidden_captures_per_verifier_event": _pct(raw_records_written, total_steps),
        "valid_examples_per_verifier_event": _pct(valid_raw, total_steps),
        "valid_all_next_per_verifier_event": _pct(simulated_valid_all_next, total_steps),
        "first_20_tasks": first_20,
        "critical_question_answer": critical_answer,
        "decision": decision,
    }


def _fmt_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Phase 6 Queued Data Density Diagnostics",
        "",
        f"raw dir: `{report['raw_dir']}`",
        f"packaged dir: `{report['packaged_dir']}`",
        f"decision: **{report['decision']}**",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| tasks processed | {report['tasks_processed']} |",
        f"| total decode/verifier steps observed | {report['total_decode_steps_observed']} |",
        f"| adjacent step pairs | {report['adjacent_step_pairs']} |",
        f"| raw records written | {report['raw_records_written']} |",
        f"| valid queued examples written | {report['valid_queued_examples_written']} |",
        f"| simulated valid if queue after every step | {report['simulated_valid_if_queue_after_every_step']} |",
        f"| simulated valid if use step weak only | {report['simulated_valid_if_use_step_weak_only']} |",
        f"| simulated valid if create and use weak | {report['simulated_valid_if_create_and_use_weak']} |",
        f"| hidden captures / verifier events | {_fmt_pct(report['hidden_captures_per_verifier_event'])} |",
        f"| valid all-next pairs / verifier events | {_fmt_pct(report['valid_all_next_per_verifier_event'])} |",
        "",
        "## Pair Counts",
        "",
        "| count | value |",
        "|---|---:|",
    ]
    for key, value in sorted(report["pair_counts"].items()):
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Derived Skip / Invalidation Counts", "", "| reason | count |", "|---|---:|"])
    for key, value in sorted(report["derived_invalidation_reason_counts"].items()):
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## First 20 Tasks",
            "",
            "| task | steps | pairs | create weak | use weak | both weak | valid all-next | raw written |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["first_20_tasks"]:
        lines.append(
            f"| `{row['task_id']}` | {row['decode_steps']} | {row['adjacent_pairs']} | "
            f"{row['create_weak']} | {row['use_weak']} | {row['both_weak']} | "
            f"{row['valid_all_next']} | {row['raw_examples_written']} |"
        )
    lines.extend(["", "## Interpretation", "", report["critical_question_answer"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    ap.add_argument("--packaged-dir", type=Path, default=DEFAULT_PACKAGED_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    ap.add_argument("--method", default="blazedit_pld_w128_n10")
    ap.add_argument("--trigger-threshold", type=int, default=4)
    ap.add_argument("--weak-field", choices=["draft_len", "accepted_len"], default="draft_len")
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    report = _diagnose(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown(args.output_dir / "report.md", report)
    _write_markdown(args.table, report)
    print((args.output_dir / "report.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
