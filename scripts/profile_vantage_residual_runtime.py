"""Profile VANTAGE-Residual runtime overhead from aggregate and step logs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_RUN_DIR = (
    Path("artifacts")
    / "vantage_residual"
    / "runs"
    / "vantage_residual_smoke50_v1"
    / "eval"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _f(row: dict[str, Any], key: str) -> float:
    value = row.get(key, 0.0)
    return float(value or 0.0)


def _i(row: dict[str, Any], key: str) -> int:
    value = row.get(key, 0)
    return int(value or 0)


def _by_method_from_steps(steps: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in steps:
        method = row.get("method")
        if method is not None:
            grouped[str(method)].append(row)
    out: dict[str, dict[str, Any]] = {}
    for method, rows in grouped.items():
        out[method] = {
            "steps": len(rows),
            "verify_us": sum(_f(r, "verify_us") for r in rows),
            "wall_us": sum(_f(r, "wall_us") for r in rows),
            "proposal_us": sum(_f(r, "proposal_us") for r in rows),
            "mtp_triggers": sum(1 for r in rows if r.get("mtp_triggered") is True),
            "mtp_extra_verify_calls": sum(_i(r, "mtp_extra_verify_calls") for r in rows),
            "mtp_head_compute_us": sum(_f(r, "mtp_head_compute_us") for r in rows),
            "mtp_verify_extra_us": sum(_f(r, "mtp_verify_extra_us") for r in rows),
            "mtp_total_overhead_us": sum(_f(r, "mtp_total_overhead_us") for r in rows),
            "mtp_actual_extra_progress": sum(_i(r, "mtp_actual_extra_progress") for r in rows),
            "mtp_extra_accepted_drafts": sum(_i(r, "mtp_extra_accepted_drafts") for r in rows),
            "mtp_token0_rejects": sum(1 for r in rows if r.get("mtp_token0_rejected") is True),
        }
    return out


def _method_summary(method: str, aggregate_method: dict[str, Any], step_method: dict[str, Any]) -> dict[str, Any]:
    triggers = int(aggregate_method.get("mtp_trigger_count", step_method.get("mtp_triggers", 0)) or 0)
    extra_verify_calls = int(
        aggregate_method.get("mtp_extra_verify_calls", step_method.get("mtp_extra_verify_calls", 0)) or 0
    )
    head_us = float(
        aggregate_method.get("mtp_head_compute_us_total", step_method.get("mtp_head_compute_us", 0.0)) or 0.0
    )
    extra_verify_us = float(
        aggregate_method.get("mtp_verify_extra_us_total", step_method.get("mtp_verify_extra_us", 0.0)) or 0.0
    )
    total_overhead_us = float(
        aggregate_method.get("mtp_total_overhead_us_total", step_method.get("mtp_total_overhead_us", 0.0)) or 0.0
    )
    wall_us = float(aggregate_method.get("wall_us_total", step_method.get("wall_us", 0.0)) or 0.0)
    verify_us = float(aggregate_method.get("verify_us_total", step_method.get("verify_us", 0.0)) or 0.0)
    proposal_us = float(aggregate_method.get("proposal_us_total", step_method.get("proposal_us", 0.0)) or 0.0)
    return {
        "method": method,
        "tokens_per_sec": aggregate_method.get("tokens_per_sec"),
        "emitted_tokens": aggregate_method.get("n_new_tokens_total", aggregate_method.get("n_emitted_total")),
        "steps": aggregate_method.get("n_steps", step_method.get("steps")),
        "wall_us": wall_us,
        "verify_us": verify_us,
        "proposal_us": proposal_us,
        "mtp_triggers": triggers,
        "mtp_head_compute_us": head_us,
        "mtp_verify_extra_us": extra_verify_us,
        "mtp_total_overhead_us": total_overhead_us,
        "mtp_extra_verify_calls": extra_verify_calls,
        "mtp_token0_rejects": aggregate_method.get("mtp_token0_reject_count", step_method.get("mtp_token0_rejects", 0)),
        "mtp_actual_extra_progress": aggregate_method.get(
            "mtp_actual_extra_progress_sum",
            step_method.get("mtp_actual_extra_progress", 0),
        ),
        "mtp_extra_accepted_drafts": aggregate_method.get(
            "mtp_extra_accepted_drafts_sum",
            step_method.get("mtp_extra_accepted_drafts", 0),
        ),
        "verify_share_of_wall": verify_us / wall_us if wall_us else None,
        "mtp_overhead_share_of_wall": total_overhead_us / wall_us if wall_us else None,
        "mtp_extra_verify_share_of_wall": extra_verify_us / wall_us if wall_us else None,
        "mtp_head_compute_share_of_wall": head_us / wall_us if wall_us else None,
        "mtp_overhead_per_trigger_us": total_overhead_us / triggers if triggers else None,
        "mtp_extra_verify_per_trigger_us": extra_verify_us / triggers if triggers else None,
        "mtp_head_compute_per_trigger_us": head_us / triggers if triggers else None,
        "extra_verify_calls_per_trigger": extra_verify_calls / triggers if triggers else None,
    }


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    methods = payload["methods"]
    lines = [
        "# VANTAGE-Residual Phase 2 Overhead Breakdown",
        "",
        f"Run directory: `{payload['run_dir']}`",
        "",
        "## Method Summary",
        "",
        "| Method | tok/s | steps | wall s | verify s | MTP triggers | extra verify calls | MTP head s | MTP extra verify s | MTP overhead s | overhead / trigger ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in methods.items():
        lines.append(
            "| {method} | {tps:.1f} | {steps} | {wall:.3f} | {verify:.3f} | {trig} | {extra} | {head:.3f} | {extra_verify:.3f} | {over:.3f} | {per:.1f} |".format(
                method=method,
                tps=float(row.get("tokens_per_sec") or 0.0),
                steps=row.get("steps"),
                wall=float(row.get("wall_us") or 0.0) / 1_000_000.0,
                verify=float(row.get("verify_us") or 0.0) / 1_000_000.0,
                trig=row.get("mtp_triggers"),
                extra=row.get("mtp_extra_verify_calls"),
                head=float(row.get("mtp_head_compute_us") or 0.0) / 1_000_000.0,
                extra_verify=float(row.get("mtp_verify_extra_us") or 0.0) / 1_000_000.0,
                over=float(row.get("mtp_total_overhead_us") or 0.0) / 1_000_000.0,
                per=(float(row.get("mtp_overhead_per_trigger_us") or 0.0) / 1000.0),
            )
        )
    lines.extend(
        [
            "",
            "## Answer",
            "",
            payload["interpretation"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Directory containing aggregate.json and steps.jsonl, or its parent run directory.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts")
        / "vantage_residual"
        / "phase2_overhead"
        / "current_k4_t4",
    )
    ap.add_argument(
        "--table-out",
        type=Path,
        default=Path("artifacts")
        / "vantage_residual"
        / "tables"
        / "phase2_current_overhead_breakdown.md",
    )
    args = ap.parse_args()

    run_dir = args.run_dir
    if (run_dir / "eval").is_dir():
        run_dir = run_dir / "eval"
    aggregate_path = run_dir / "aggregate.json"
    steps_path = run_dir / "steps.jsonl"
    if not aggregate_path.exists():
        raise FileNotFoundError(f"missing aggregate.json under {run_dir}")

    aggregate = _load_json(aggregate_path)
    steps = _load_jsonl(steps_path)
    step_methods = _by_method_from_steps(steps)
    methods = {}
    for method, row in aggregate.get("by_method", {}).items():
        methods[method] = _method_summary(method, row, step_methods.get(method, {}))

    residual_method = next((m for m in methods if m.startswith("vantage_residual")), None)
    if residual_method:
        r = methods[residual_method]
        interpretation = (
            "The current residual method primarily loses because residual verification "
            "adds many extra target forwards. The residual head itself is a small part "
            f"of wall time ({float(r.get('mtp_head_compute_us') or 0.0) / 1_000_000.0:.3f}s), "
            f"while extra residual verification costs {float(r.get('mtp_verify_extra_us') or 0.0) / 1_000_000.0:.3f}s "
            f"across {r.get('mtp_extra_verify_calls')} extra verify calls. "
            "Residual quality is not zero: the run saves steps and accepts extra drafts, "
            "but the post-PLD second verifier pass is compute-negative."
        )
    else:
        interpretation = "No VANTAGE-Residual method was found in the aggregate."

    payload = {
        "run_dir": str(run_dir),
        "methods": methods,
        "output_equivalence": aggregate.get("output_equivalence", {}),
        "interpretation": interpretation,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_md(args.table_out, payload)
    print(json.dumps({"status": "ok", "output": str(args.output_dir / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
