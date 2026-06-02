#!/usr/bin/env python3
"""Build Markdown tables for VANTAGE residual/post-PLD analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import analyze_vantage_residual_results as analyzer
except ModuleNotFoundError:  # pragma: no cover - import path depends on caller
    from scripts import analyze_vantage_residual_results as analyzer  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("artifacts/vantage_residual/tables")
DEFAULT_ANALYSIS_JSON = DEFAULT_OUTPUT_DIR / "residual_analysis.json"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def stable_relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_analysis(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "not captured"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_pct_fraction(value: Any, digits: int = 1) -> str:
    if value is None:
        return "not captured"
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "not captured"
    return f"{100.0 * float(value):.{digits}f}%"


def fmt_pct_value(value: Any, digits: int = 1) -> str:
    if value is None:
        return "not captured"
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "not captured"
    return f"{float(value):.{digits}f}%"


def markdown_table(headers: list[str], rows: list[list[str]], aligns: list[str] | None = None) -> str:
    aligns = aligns or ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def runtime_notes(row: dict[str, Any]) -> str:
    notes: list[str] = []
    if row.get("matches_baseline_rate") is not None and row.get("matches_baseline_rate") != 1.0:
        notes.append("not output-equivalent to PLD baseline")
    if row.get("is_baseline_method"):
        notes.append("baseline")
    if (row.get("queue_predictions_created") or 0) > 0 or (row.get("queue_predictions_used") or 0) > 0:
        notes.append("queued MTP")
    if row.get("rerank_trigger_rate") is not None:
        notes.append("rerank residual")
    return "; ".join(notes) if notes else "measured"


def write_runtime_table(analysis: dict[str, Any], path: Path) -> None:
    rows = []
    for row in analysis.get("runtime_measured", []):
        rows.append(
            [
                "runtime measured",
                str(row.get("run_label") or ""),
                str(row.get("method") or ""),
                fmt(row.get("n_tasks"), 0),
                fmt(row.get("tokens_per_sec"), 1),
                fmt(row.get("speedup_vs_baseline"), 3),
                fmt_pct_fraction(row.get("matches_baseline_rate")),
                fmt(row.get("n_steps"), 0),
                fmt(row.get("mtp_trigger_count"), 0),
                fmt(row.get("mtp_used_count"), 0),
                fmt(row.get("mtp_actual_extra_progress_sum"), 0),
                fmt(row.get("mtp_total_overhead_us_per_trigger"), 1),
                fmt_pct_fraction(row.get("mtp_token0_reject_rate")),
                runtime_notes(row),
            ]
        )
    text = "# VANTAGE Residual Runtime-Measured Results\n\n"
    text += (
        "Rows in this table are measured runtime aggregates. They are not offline projections.\n\n"
    )
    text += markdown_table(
        [
            "Evidence",
            "Run",
            "Method",
            "Tasks",
            "tok/s",
            "Speedup vs PLD baseline",
            "Output parity vs PLD",
            "Steps",
            "Triggers",
            "MTP used",
            "Extra progress",
            "Overhead us/trigger",
            "Token0 reject",
            "Notes",
        ],
        rows,
        ["---", "---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---"],
    )
    path.write_text(text, encoding="utf-8")


def write_projection_table(analysis: dict[str, Any], path: Path) -> None:
    rows = []
    for row in analysis.get("offline_projection", []):
        rows.append(
            [
                "offline projection",
                str(row.get("run_label") or ""),
                str(row.get("trigger_policy") or ""),
                "yes" if row.get("is_best_policy") else "no",
                fmt(row.get("baseline_steps"), 0),
                fmt(row.get("projected_steps"), 0),
                fmt_pct_value(row.get("step_reduction_pct")),
                fmt(row.get("corrected_projected_speedup"), 3),
                fmt(row.get("trigger_count"), 0),
                fmt(row.get("missing_predictions"), 0),
                fmt(row.get("avg_extra_accepted_mtp_tokens_per_trigger"), 3),
                fmt_pct_value(row.get("token0_rejection_rate_pct")),
            ]
        )
    text = "# VANTAGE Residual Offline Projection Results\n\n"
    text += (
        "Rows in this table are offline projections from post-PLD MTP diagnostics. "
        "They are not runtime-measured speedups.\n\n"
    )
    text += markdown_table(
        [
            "Evidence",
            "Run",
            "Trigger policy",
            "Best",
            "Baseline steps",
            "Projected steps",
            "Step reduction",
            "Projected speedup",
            "Triggers",
            "Missing predictions",
            "Extra accepted/trigger",
            "Token0 reject",
        ],
        rows,
        ["---", "---", "---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
    )
    path.write_text(text, encoding="utf-8")


def write_combined_summary(analysis: dict[str, Any], path: Path) -> None:
    projection_rows = analysis.get("offline_projection", [])
    runtime_rows = analysis.get("runtime_measured", [])
    best_projection = [row for row in projection_rows if row.get("is_best_policy")]
    runtime_nonbaseline = [
        row for row in runtime_rows if row.get("method") != analysis.get("baseline_method")
    ]

    lines = [
        "# VANTAGE Residual Analysis Summary",
        "",
        f"Generated from `{len(runtime_rows)}` runtime-measured rows and `{len(projection_rows)}` offline projection rows.",
        "",
        "## Best Offline Projection Rows",
        "",
    ]
    if best_projection:
        lines.append(
            markdown_table(
                [
                    "Evidence",
                    "Run",
                    "Policy",
                    "Projected speedup",
                    "Step reduction",
                    "Recommendation",
                ],
                [
                    [
                        "offline projection",
                        str(row.get("run_label") or ""),
                        str(row.get("trigger_policy") or ""),
                        fmt(row.get("corrected_projected_speedup"), 3),
                        fmt_pct_value(row.get("step_reduction_pct")),
                        str(row.get("recommendation") or ""),
                    ]
                    for row in best_projection
                ],
                ["---", "---", "---", "---:", "---:", "---"],
            ).rstrip()
        )
    else:
        lines.append("No best-policy projection row was present.")

    lines.extend(["", "## Runtime-Measured Residual Rows", ""])
    if runtime_nonbaseline:
        lines.append(
            markdown_table(
                [
                    "Evidence",
                    "Run",
                    "Method",
                    "tok/s",
                    "Speedup vs PLD baseline",
                    "Output parity vs PLD",
                    "Notes",
                ],
                [
                    [
                        "runtime measured",
                        str(row.get("run_label") or ""),
                        str(row.get("method") or ""),
                        fmt(row.get("tokens_per_sec"), 1),
                        fmt(row.get("speedup_vs_baseline"), 3),
                        fmt_pct_fraction(row.get("matches_baseline_rate")),
                        runtime_notes(row),
                    ]
                    for row in runtime_nonbaseline
                ],
                ["---", "---", "---", "---:", "---:", "---:", "---"],
            ).rstrip()
        )
    else:
        lines.append("No non-baseline runtime residual rows were present.")

    lines.extend(["", "## Limitations", ""])
    for item in analysis.get("limitations", []):
        lines.append(f"- {item}")
    if analysis.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for item in analysis["warnings"]:
            lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_or_load_analysis(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    analysis_path = resolve_path(args.analysis_json)
    explicit_inputs = bool(args.run_json or args.projection_json or args.no_discover)
    if not explicit_inputs and analysis_path.exists():
        return load_analysis(analysis_path), analysis_path

    runtime_paths = [analyzer.resolve_path(path) for path in args.run_json]
    projection_paths = [analyzer.resolve_path(path) for path in args.projection_json]
    if not args.no_discover:
        runtime_paths.extend(analyzer.discover_paths(analyzer.DEFAULT_RUNTIME_GLOBS))
        projection_paths.extend(analyzer.discover_paths(analyzer.DEFAULT_PROJECTION_GLOBS))
    runtime_paths = sorted({path.resolve() for path in runtime_paths})
    projection_paths = sorted({path.resolve() for path in projection_paths})
    payload = analyzer.build_analysis(
        runtime_paths=runtime_paths,
        projection_paths=projection_paths,
        baseline=args.baseline,
    )
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload, analysis_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-json", type=Path, default=DEFAULT_ANALYSIS_JSON)
    parser.add_argument("--run-json", action="append", default=[], type=Path)
    parser.add_argument("--projection-json", action="append", default=[], type=Path)
    parser.add_argument("--no-discover", action="store_true")
    parser.add_argument("--baseline", default=analyzer.DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis, analysis_path = build_or_load_analysis(args)

    runtime_path = output_dir / "vantage_residual_runtime_measured.md"
    projection_path = output_dir / "vantage_residual_offline_projection.md"
    summary_path = output_dir / "vantage_residual_combined_summary.md"
    write_runtime_table(analysis, runtime_path)
    write_projection_table(analysis, projection_path)
    write_combined_summary(analysis, summary_path)

    print(f"analysis: {stable_relpath(analysis_path)}")
    print(f"wrote {stable_relpath(runtime_path)}")
    print(f"wrote {stable_relpath(projection_path)}")
    print(f"wrote {stable_relpath(summary_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
