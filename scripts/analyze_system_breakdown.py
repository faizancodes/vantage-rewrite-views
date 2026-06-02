#!/usr/bin/env python3
"""Compute measured and residual system-time breakdowns from timing logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "analysis" / "continuous_batched_pld_final_repeats" / "report.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "system_breakdown_analysis.json"
DEFAULT_TABLE_MD = ROOT / "artifacts" / "tables" / "system_breakdown_residual_accounting.md"
DEFAULT_TABLE_TEX = ROOT / "artifacts" / "tables" / "system_breakdown_residual_accounting.tex"
DEFAULT_DOC = ROOT / "docs" / "system_breakdown_residual_accounting.md"

TOP_LEVEL_TIMERS = [
    ("total_forward_ms", "Verifier target forward"),
    ("total_prefill_ms", "Prompt prefill"),
    ("pld_lookup_ms", "PLD lookup"),
    ("scheduler_overhead_ms", "Aggregate scheduler/runtime overhead"),
]

FINE_GRAINED_TIMER_PROPOSAL = [
    (
        "prompt_encode_ms",
        "Prompt tokenization/encoding during refill.",
        "scripts/run_batched_pld_eval.py:740-759, around _encode_prompt_ids() before _prefill_task().",
    ),
    (
        "cache_combine_ms",
        "Batched KV-cache scatter/gather before verifier forward.",
        "scripts/run_batched_pld_eval.py:485-487, around _combine_task_caches().",
    ),
    (
        "verify_tensor_build_ms",
        "input_ids, attention_mask, and position_ids construction.",
        "scripts/run_batched_pld_eval.py:487-492, around _make_verify_tensors().",
    ),
    (
        "cuda_sync_pre_forward_ms",
        "CUDA synchronization wait immediately before timing the target forward.",
        "scripts/run_batched_pld_eval.py:503-504, around _sync(device).",
    ),
    (
        "cuda_sync_post_forward_ms",
        "CUDA synchronization wait after the target call; currently included in total_forward_ms.",
        "scripts/run_batched_pld_eval.py:505-514, split model call and post-call _sync(device).",
    ),
    (
        "greedy_verify_ms",
        "Logit slicing, argmax, and greedy draft verification.",
        "scripts/run_batched_pld_eval.py:516-545, inside the per-row post-forward loop.",
    ),
    (
        "cache_compact_ms",
        "Per-row KV-cache compaction/crop after accepted drafts.",
        "scripts/run_batched_pld_eval.py:562-568, around _compact_cache_row().",
    ),
    (
        "token_bookkeeping_ms",
        "Prefix append, generated-token counters, EOS/max-token checks, and finish flags.",
        "scripts/run_batched_pld_eval.py:569-582, after cache compaction.",
    ),
    (
        "audit_write_ms",
        "Optional audit JSON serialization/write overhead.",
        "scripts/run_batched_pld_eval.py:583-616 and 736-739, around audit_writer/write_audit().",
    ),
    (
        "finish_refill_bookkeeping_ms",
        "Finished-task filtering, output copy, cache release, task latency, and refill calls.",
        "scripts/run_batched_pld_eval.py:831-853, around the active/still_active pass and refill().",
    ),
]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return mean(float(row.get(key, 0.0) or 0.0) for row in rows) if rows else 0.0


def _pct(part: float, total: float) -> float:
    return 100.0 * part / total if total else 0.0


def _field_status(rows: list[dict[str, Any]], field: str) -> str:
    values = [row.get(field) for row in rows]
    if any(field not in row for row in rows):
        return "missing"
    if all(value is None for value in values):
        return "null"
    if all((float(value or 0.0) == 0.0) for value in values):
        return "zero"
    return "measured"


def _padding_pct(row: dict[str, Any]) -> float:
    input_pad = float(row.get("input_padding_waste_tokens", 0.0) or 0.0)
    real_verified = float(row.get("real_verified_tokens", 0.0) or 0.0)
    return _pct(input_pad, input_pad + real_verified)


def _fine_timer_status(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "field": field,
            "status": _field_status(rows, field),
            "meaning": meaning,
            "insertion_point": insertion_point,
        }
        for field, meaning, insertion_point in FINE_GRAINED_TIMER_PROPOSAL
    ]


def summarize(report: dict[str, Any], source: Path) -> dict[str, Any]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in report.get("rows", []):
        method = str(row.get("method"))
        batch = int(row.get("batch_size", 0))
        if method != "continuous_batched_pld_w128_n10":
            continue
        groups.setdefault((method, batch), []).append(row)

    rows: list[dict[str, Any]] = []
    for (_, batch), group in sorted(groups.items()):
        wall = _mean(group, "wall_ms")
        verifier_forward = _mean(group, "total_forward_ms")
        prefill = _mean(group, "total_prefill_ms")
        pld_lookup = _mean(group, "pld_lookup_ms")
        scheduler = _mean(group, "scheduler_overhead_ms")
        legacy_residual = wall - verifier_forward - pld_lookup - scheduler
        residual = wall - verifier_forward - prefill - pld_lookup - scheduler
        residual = 0.0 if abs(residual) < 1e-6 else residual
        top_level_measured = verifier_forward + prefill + pld_lookup + scheduler
        input_padding_pct = mean(_padding_pct(row) for row in group) if group else 0.0
        component_rows = [
            {
                "component": "Verifier target forward",
                "field": "total_forward_ms",
                "ms": verifier_forward,
                "pct": _pct(verifier_forward, wall),
                "status": _field_status(group, "total_forward_ms"),
                "scope": "Measured around target verifier forward calls.",
            },
            {
                "component": "Prompt prefill",
                "field": "total_prefill_ms",
                "ms": prefill,
                "pct": _pct(prefill, wall),
                "status": _field_status(group, "total_prefill_ms"),
                "scope": "Measured prompt prefill forward time during initial fill and continuous refill.",
            },
            {
                "component": "PLD lookup",
                "field": "pld_lookup_ms",
                "ms": pld_lookup,
                "pct": _pct(pld_lookup, wall),
                "status": _field_status(group, "pld_lookup_ms"),
                "scope": "Measured prompt-lookup draft search loop.",
            },
            {
                "component": "Aggregate scheduler/runtime overhead",
                "field": "scheduler_overhead_ms",
                "ms": scheduler,
                "pct": _pct(scheduler, wall),
                "status": _field_status(group, "scheduler_overhead_ms"),
                "scope": "Computed by runner as wall - verifier forward - prompt prefill - PLD lookup.",
            },
            {
                "component": "Residual after logged top-level timers",
                "field": "derived",
                "ms": residual,
                "pct": _pct(residual, wall),
                "status": "derived",
                "scope": "Remaining wall time after the four logged top-level components.",
            },
        ]
        rows.append(
            {
                "config": f"b{batch}",
                "batch_size": batch,
                "n_repeats": len(group),
                "wall_ms": wall,
                "top_level_measured_ms": top_level_measured,
                "top_level_measured_pct": _pct(top_level_measured, wall),
                "target_forward_ms": verifier_forward,
                "verifier_forward_ms": verifier_forward,
                "verifier_forward_pct": _pct(verifier_forward, wall),
                "prefill_ms": prefill,
                "prefill_pct": _pct(prefill, wall),
                "pld_lookup_ms": pld_lookup,
                "pld_lookup_pct": _pct(pld_lookup, wall),
                "scheduler_overhead_ms": scheduler,
                "scheduler_overhead_pct": _pct(scheduler, wall),
                "legacy_residual_without_prefill_ms": legacy_residual,
                "legacy_residual_without_prefill_pct": _pct(legacy_residual, wall),
                "residual_ms": residual,
                "residual_pct": _pct(residual, wall),
                "verifier_forwards": _mean(group, "verifier_forwards"),
                "mean_active_tasks": _mean(group, "active_tasks_mean"),
                "verified_tokens_per_forward": _mean(group, "verified_tokens_per_forward"),
                "accepted_tokens_per_forward": _mean(group, "accepted_tokens_per_forward"),
                "input_padding_waste_pct": input_padding_pct,
                "memory_peak_gb": _mean(group, "memory_peak_gb"),
                "top_level_timer_status": [
                    {"field": field, "component": component, "status": _field_status(group, field)}
                    for field, component in TOP_LEVEL_TIMERS
                ],
                "components": component_rows,
                "missing_or_null_fine_grained_timers": _fine_timer_status(group),
            }
        )
    return {
        "source": str(source),
        "rows": rows,
        "notes": [
            "The prior 18-25% residual was caused by omitting total_prefill_ms from the analyzer, not by missing wall time.",
            "After including total_prefill_ms, the final repeated timing logs are fully accounted at the logged top-level timer granularity.",
            "scheduler_overhead_ms is still an aggregate runner field, computed as wall_ms - total_forward_ms - total_prefill_ms - pld_lookup_ms.",
            "Do not claim a fine-grained system breakdown until a rerun records the missing/null fine-grained timers listed in this artifact.",
        ],
        "scope_statement": (
            "Reviewer-facing claim should be scoped to top-level logged accounting: verifier forward, prompt prefill, "
            "PLD lookup, and aggregate scheduler/runtime overhead account for wall time. The current logs do not "
            "attribute the aggregate scheduler/runtime overhead into scatter/gather, tensor construction, cache "
            "compaction, synchronization, bookkeeping, or logging subcomponents."
        ),
    }


def _write_analysis_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# System Breakdown Analysis",
        "",
        report["scope_statement"],
        "",
        "| Config | Wall ms | Verifier forward ms | Prefill ms | PLD lookup ms | Scheduler/runtime ms | Residual after logged timers | Legacy residual without prefill | Peak GB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            "| {config} | {wall_ms:.0f} | {verifier_forward_ms:.0f} ({verifier_forward_pct:.1f}%) | "
            "{prefill_ms:.0f} ({prefill_pct:.1f}%) | {pld_lookup_ms:.0f} ({pld_lookup_pct:.1f}%) | "
            "{scheduler_overhead_ms:.0f} ({scheduler_overhead_pct:.1f}%) | "
            "{residual_ms:.0f} ({residual_pct:.1f}%) | "
            "{legacy_residual_without_prefill_ms:.0f} ({legacy_residual_without_prefill_pct:.1f}%) | "
            "{memory_peak_gb:.2f} |".format(**row)
        )
    lines += [
        "",
        "## Missing/null fine-grained timers",
        "",
        "| Field | Status | Meaning | Proposed insertion point |",
        "|---|---|---|---|",
    ]
    for item in report["rows"][0]["missing_or_null_fine_grained_timers"] if report["rows"] else []:
        lines.append(
            "| {field} | {status} | {meaning} | {insertion_point} |".format(**item)
        )
    lines += ["", "## Notes"]
    lines.extend(f"- {note}" for note in report["notes"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_table_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# System Breakdown Residual Accounting",
        "",
        "| Config | Wall ms | Verifier forward | Prompt prefill | PLD lookup | Aggregate scheduler/runtime | Residual after logged timers | Claim status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        status = (
            "Top-level accounted; scheduler/runtime remains aggregate"
            if abs(float(row["residual_ms"])) < 0.5
            else "Top-level residual remains"
        )
        lines.append(
            "| {config} | {wall_ms:.0f} | {verifier_forward_ms:.0f} ({verifier_forward_pct:.1f}%) | "
            "{prefill_ms:.0f} ({prefill_pct:.1f}%) | {pld_lookup_ms:.0f} ({pld_lookup_pct:.1f}%) | "
            "{scheduler_overhead_ms:.0f} ({scheduler_overhead_pct:.1f}%) | "
            "{residual_ms:.0f} ({residual_pct:.1f}%) | {status} |".format(status=status, **row)
        )
    lines.extend(
        [
            "",
            "Reviewer-facing wording: the existing logs support a top-level accounting, not a fine-grained system attribution. "
            "The previous 18-25% residual is accounted for by `total_prefill_ms`; the remaining unresolved component is the "
            "`scheduler_overhead_ms` aggregate.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tex_escape(value: str) -> str:
    return value.replace("%", "\\%").replace("_", "\\_")


def _write_table_tex(report: dict[str, Any], path: Path) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Top-level system-time accounting from final repeated timing logs. The scheduler/runtime column is aggregate.}",
        "\\label{tab:system-breakdown-residual-accounting}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Config & Wall ms & Verifier fwd & Prefill & PLD lookup & Scheduler/runtime & Residual \\\\",
        "\\midrule",
    ]
    for row in report["rows"]:
        lines.append(
            "{config} & {wall_ms:.0f} & {verifier_forward_ms:.0f} ({verifier_forward_pct:.1f}\\%) & "
            "{prefill_ms:.0f} ({prefill_pct:.1f}\\%) & {pld_lookup_ms:.0f} ({pld_lookup_pct:.1f}\\%) & "
            "{scheduler_overhead_ms:.0f} ({scheduler_overhead_pct:.1f}\\%) & {residual_ms:.0f} ({residual_pct:.1f}\\%) \\\\".format(
                **row
            )
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_tex_escape(line) if i in [] else line for i, line in enumerate(lines)) + "\n", encoding="utf-8")


def _write_doc(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# System Breakdown Residual Accounting",
        "",
        "## Current conclusion",
        "",
        "The current 18-25% residual is not missing runtime. It is prompt prefill time that was present in the timing logs as `total_prefill_ms` but omitted from the prior analyzer table.",
        "",
        "After adding `total_prefill_ms`, the repeated timing artifact accounts for wall time at top-level granularity. The remaining limitation is that `scheduler_overhead_ms` is an aggregate computed by the runner, not a fine-grained runtime decomposition.",
        "",
        "## Reviewer-facing scope",
        "",
        report["scope_statement"],
        "",
        "## Top-level accounting",
        "",
        "| Config | Wall ms | Verifier forward | Prompt prefill | PLD lookup | Aggregate scheduler/runtime | Residual after logged timers |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            "| {config} | {wall_ms:.0f} | {verifier_forward_ms:.0f} ({verifier_forward_pct:.1f}%) | "
            "{prefill_ms:.0f} ({prefill_pct:.1f}%) | {pld_lookup_ms:.0f} ({pld_lookup_pct:.1f}%) | "
            "{scheduler_overhead_ms:.0f} ({scheduler_overhead_pct:.1f}%) | "
            "{residual_ms:.0f} ({residual_pct:.1f}%) |".format(**row)
        )
    lines += [
        "",
        "## Missing fine-grained timers",
        "",
        "These fields are absent from the current timing rows, so they cannot be used to subdivide `scheduler_overhead_ms`:",
        "",
        "| Proposed field | Meaning | Exact insertion point |",
        "|---|---|---|",
    ]
    for item in report["rows"][0]["missing_or_null_fine_grained_timers"] if report["rows"] else []:
        lines.append("| {field} | {meaning} | {insertion_point} |".format(**item))
    lines += [
        "",
        "## Minimal rerun/instrumentation requirement",
        "",
        "A rerun is only required if the paper wants to claim a fine-grained system breakdown below the aggregate scheduler/runtime line. The minimal patch is to add the fields above to `BatchedRunMetrics`, accumulate them in `_run_batched_verify()` and `run_batched_continuous()`, and compute `scheduler_other_ms = wall_ms - total_forward_ms - total_prefill_ms - pld_lookup_ms - sum(fine_grained_scheduler_fields)`.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--table-md", default=str(DEFAULT_TABLE_MD))
    parser.add_argument("--table-tex", default=str(DEFAULT_TABLE_TEX))
    parser.add_argument("--doc", default=str(DEFAULT_DOC))
    args = parser.parse_args()

    input_path = Path(args.input)
    report = summarize(_load(input_path), input_path)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_analysis_md(report, out.with_suffix(".md"))
    _write_table_md(report, Path(args.table_md))
    _write_table_tex(report, Path(args.table_tex))
    _write_doc(report, Path(args.doc))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
