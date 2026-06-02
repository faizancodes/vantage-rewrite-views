"""Aggregate controlled edit-drift sweep results."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _task_tps(output: dict[str, Any]) -> float:
    wall_us = float(output.get("wall_us") or 0.0)
    tokens = float(output.get("n_new_tokens") or 0.0)
    return tokens / (wall_us / 1e6) if wall_us > 0 else 0.0


def _group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    meta = row.get("metadata") or {}
    axis = str(meta.get("axis") or row.get("axis") or "unknown")
    cell = str(meta.get("requested_cell") or row.get("requested_cell") or "unknown")
    family = str(meta.get("drift_family") or row.get("drift_family") or "unknown")
    return family, axis, cell


def analyze(
    completions: list[dict[str, Any]],
    *,
    pld_method: str,
    vantage_method: str,
) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    all_rows: list[dict[str, Any]] = []
    for row in completions:
        outputs = row.get("outputs") or {}
        if pld_method not in outputs or vantage_method not in outputs:
            continue
        meta = row.get("metadata") or {}
        pld_tps = _task_tps(outputs[pld_method])
        nh_tps = _task_tps(outputs[vantage_method])
        task = {
            "task_id": row.get("task_id"),
            "family": meta.get("drift_family") or row.get("drift_family"),
            "axis": meta.get("axis") or row.get("axis"),
            "cell": str(meta.get("requested_cell") or row.get("requested_cell")),
            "pld_tps": pld_tps,
            "vantage_tps": nh_tps,
            "ratio": nh_tps / pld_tps if pld_tps else 0.0,
            "drift_intensity": float(
                meta.get("drift_intensity")
                or row.get("drift_intensity")
                or meta.get("rename_percentage_realized")
                or row.get("rename_percentage_realized")
                or 0.0
            ),
            "copy_ratio": float(
                meta.get("copied_token_percentage")
                or row.get("copied_token_percentage")
                or 0.0
            ),
            "edit_distance": float(
                meta.get("edit_distance_tokens") or row.get("edit_distance_tokens") or 0.0
            ),
            "hunks": float(meta.get("changed_hunk_count") or row.get("changed_hunk_count") or 0.0),
            "longest_span": float(
                meta.get("longest_unchanged_span_tokens")
                or row.get("longest_unchanged_span_tokens")
                or 0.0
            ),
        }
        all_rows.append(task)
        groups[_group_key(row)].append(task)

    grouped = []
    for (family, axis, cell), rows in sorted(groups.items()):
        ratios = [float(r["ratio"]) for r in rows]
        grouped.append(
            {
                "family": family,
                "axis": axis,
                "cell": cell,
                "n": len(rows),
                "ratio_mean": _mean(ratios),
                "ratio_median": _median(ratios),
                "pld_tps_mean": _mean([float(r["pld_tps"]) for r in rows]),
                "vantage_tps_mean": _mean([float(r["vantage_tps"]) for r in rows]),
                "drift_intensity_mean": _mean([float(r["drift_intensity"]) for r in rows]),
                "copy_ratio_mean": _mean([float(r["copy_ratio"]) for r in rows]),
                "edit_distance_mean": _mean([float(r["edit_distance"]) for r in rows]),
                "hunks_mean": _mean([float(r["hunks"]) for r in rows]),
                "longest_span_mean": _mean([float(r["longest_span"]) for r in rows]),
                "reading": "VANTAGE faster" if _mean(ratios) > 1.0 else "PLD faster",
            }
        )
    return {
        "schema": "asts-spec/drift-sweep/v1",
        "pld_method": pld_method,
        "vantage_method": vantage_method,
        "n_tasks": len(all_rows),
        "overall_ratio_mean": _mean([float(r["ratio"]) for r in all_rows]),
        "groups": grouped,
        "tasks": all_rows,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Drift Sweep",
        "",
        f"PLD: `{report['pld_method']}`",
        f"VANTAGE: `{report['vantage_method']}`",
        "",
        "| Family | Axis | Cell | n | VANTAGE/PLD mean | Drift | Copy | Edit dist | Hunks | Longest span | Reading |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["groups"]:
        lines.append(
            f"| {row['family']} | {row['axis']} | {row['cell']} | {row['n']} | "
            f"{row['ratio_mean']:.3f} | {row['drift_intensity_mean']:.1f} | "
            f"{100.0 * row['copy_ratio_mean']:.1f}% | {row['edit_distance_mean']:.1f} | "
            f"{row['hunks_mean']:.1f} | {row['longest_span_mean']:.1f} | {row['reading']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--completions", required=True)
    p.add_argument("--pld-method", required=True)
    p.add_argument("--vantage-method", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    args = p.parse_args()
    report = analyze(
        _load_jsonl(Path(args.completions)),
        pld_method=args.pld_method,
        vantage_method=args.vantage_method,
    )
    Path(args.output_json).write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, Path(args.output_md))


if __name__ == "__main__":
    main()
