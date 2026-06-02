#!/usr/bin/env python3
"""Summarize emitted-token denominators from available VANTAGE artifacts.

This script does not rerun generation. It extracts the emitted-token totals
that were actually used for throughput denominators in the final timing logs
and contrasts them with gold-target lengths from the manifest statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPEATS = ROOT / "analysis" / "continuous_batched_pld_final_repeats" / "report.json"
DEFAULT_DATASET_STATS = ROOT / "artifacts" / "dataset_stats.json"
DEFAULT_OUT = ROOT / "artifacts" / "generation_stats.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stop_reason_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("stop_reason", "finish_reason", "termination_reason", "stop")
    counts: dict[str, int] = {}
    seen_key = None
    for row in rows:
        for key in keys:
            if key in row and row.get(key) is not None:
                seen_key = key
                value = str(row.get(key))
                counts[value] = counts.get(value, 0) + 1
                break
    if counts:
        return {"available": True, "field": seen_key, "counts": counts}
    return {
        "available": False,
        "reason": "No stop_reason/finish_reason/termination_reason/stop field is present in the local timing report rows.",
    }


def _summarize_runs(repeats: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in repeats.get("rows", []):
        key = (str(row.get("method")), int(row.get("batch_size", 0)))
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (method, batch), rows in sorted(groups.items()):
        emitted = [float(r.get("total_generated_tokens", 0.0) or 0.0) for r in rows]
        total_new = [float(r.get("total_new_tokens", 0.0) or 0.0) for r in rows if r.get("total_new_tokens") is not None]
        tasks = float(rows[0].get("n_tasks", repeats.get("args", {}).get("n", 500)) or 500)
        out.append(
            {
                "method": method,
                "backend": f"{repeats.get('args', {}).get('dtype', 'unknown')}/{str(repeats.get('args', {}).get('attn', 'unknown')).upper()}",
                "batch_size": batch,
                "repeats": len(rows),
                "total_emitted_tokens_mean": mean(emitted) if emitted else 0.0,
                "total_new_tokens_mean": mean(total_new) if total_new else None,
                "mean_emitted_tokens_per_task": (mean(emitted) / tasks) if emitted and tasks else 0.0,
                "throughput_denominator": "total_generated_tokens from timing report rows",
                "throughput_denominator_definition": "count of model-emitted generated tokens used for tok/s; distinct from manifest gold target length",
                "generated_output_length_availability": "aggregate total_generated_tokens only; per-task generated text/lengths are not present in this artifact",
                "stop_reason_distribution": _stop_reason_summary(rows),
            }
        )
    return out


def _write_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Generation Denominator Statistics",
        "",
        "Throughput is computed over emitted model tokens from the timing report (`total_generated_tokens`), not over gold post-edit target tokens from the manifest.",
        "",
        "| Method | Backend | Batch | Total emitted tokens (mean) | Mean emitted tokens/task | Denominator | Stop reasons |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in report["rows"]:
        stop = row["stop_reason_distribution"]
        stop_text = json.dumps(stop.get("counts", {}), sort_keys=True) if stop.get("available") else stop.get("reason", "unavailable")
        lines.append(
            "| {method} | {backend} | {batch_size} | {total_emitted_tokens_mean:.0f} | "
            "{mean_emitted_tokens_per_task:.1f} | {throughput_denominator} | {stop_text} |".format(
                stop_text=stop_text, **row
            )
        )
    if report.get("gold_target_summary"):
        g = report["gold_target_summary"]
        lines += [
            "",
            "Gold target output lengths are dataset metadata from `deterministic_target`, not generated output lengths and not the throughput denominator:",
            f"- test split mean gold target tokens: {g.get('mean_output_tokens', 0.0):.1f}",
            f"- test split p50/p90/p99 gold target tokens: {g.get('output_tokens_p50', 0.0):.0f}/{g.get('output_tokens_p90', 0.0):.0f}/{g.get('output_tokens_p99', 0.0):.0f}",
        ]
    lines += [
        "",
        "Generated output length availability: this artifact stores aggregate emitted-token totals, not generated text or per-task generated lengths.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", default=str(DEFAULT_REPEATS))
    parser.add_argument("--dataset-stats", default=str(DEFAULT_DATASET_STATS))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    repeats = _load(Path(args.repeats))
    dataset_stats = _load(Path(args.dataset_stats)) if Path(args.dataset_stats).exists() else {"splits": []}
    gold = None
    for split in dataset_stats.get("splits", []):
        if "test500" in split.get("name", ""):
            gold = split
            break
    report = {
        "source": str(Path(args.repeats)),
        "rows": _summarize_runs(repeats),
        "gold_target_summary": gold,
        "denominator_definitions": {
            "emitted_tokens": "timing row total_generated_tokens; this is the tok/s denominator",
            "gold_target_tokens": "dataset_stats mean_output_tokens from deterministic_target tokenizer length; not generated output length",
            "generated_output_lengths": "not available per task in the local timing report",
        },
        "notes": [
            "The timing artifact does not store generated text or per-task emitted lengths.",
            "Stop reason distribution is reported only if a stop/finish reason field is present; otherwise the unavailable reason is explicit.",
            "The task-isolation audit count of 100,780 emitted tokens comes from the batch=8 audit trace for one timing repeat.",
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_md(report, out.with_suffix(".md"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
