#!/usr/bin/env python3
"""Characterize bf16/SDPA timing-path drift from aggregate or trace artifacts.

The final timing reports currently provide aggregate output match counts.  When
per-task comparison JSONL is supplied, this script computes token-level drift
statistics.  Otherwise it emits the aggregate counts and marks trace-only
fields unavailable.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "analysis" / "continuous_batched_pld_final_repeats" / "report.json"
DEFAULT_OUTPUT_JSON = ROOT / "artifacts" / "timing_path_drift_analysis.json"
DEFAULT_OUTPUT_MD = ROOT / "artifacts" / "timing_path_drift_analysis.md"
DEFAULT_TABLE_MD = ROOT / "artifacts" / "tables" / "drift_timing_path.md"
DEFAULT_TABLE_TEX = ROOT / "artifacts" / "tables" / "drift_timing_path.tex"
DEFAULT_FINAL_TABLE_MD = (
    ROOT
    / "analysis"
    / "final_paper_artifacts"
    / "continuous_batched_pld_final"
    / "tables"
    / "timing_path_drift.md"
)
DEFAULT_FINAL_TABLE_TEX = DEFAULT_FINAL_TABLE_MD.with_suffix(".tex")
DEFAULT_DOC = ROOT / "docs" / "drift_timing_path.md"
DEFAULT_INSPECT_REPORTS = [
    DEFAULT_INPUT,
    ROOT
    / "analysis"
    / "final_paper_artifacts"
    / "continuous_batched_pld_final"
    / "reports"
    / "continuous_batched_pld_final_repeats_report.json",
    ROOT / "analysis" / "continuous_batched_pld_correctness_sharded" / "report.json",
    ROOT / "analysis" / "continuous_batched_pld_task_audit_test500" / "report.json",
]

AGGREGATE_ONLY_REASON = (
    "The available bf16/SDPA timing reports contain aggregate output_match_count/"
    "output_mismatch_count fields but no per-task baseline/batched output records."
)
TOKEN_DETAIL_REASON = (
    "Per-task baseline and batched token IDs are required for first mismatch "
    "positions and differing-token counts."
)
FINISH_LENGTH_REASON = (
    "Per-task baseline and batched token IDs plus finish reasons are required "
    "for length/finish mismatch counts."
)
QUALITY_REASON = (
    "Per-task decoded baseline/batched outputs and a deterministic target "
    "are required for target edit-distance quality deltas."
)
LOGIT_REASON = "Top-1/top-2 logits or margins are unavailable in the inspected artifacts."
NEXT_TRACE_SCHEMA = {
    "task_id": "real-commit task id",
    "run_id": 0,
    "batch_size": 8,
    "baseline_token_ids": [123, 456],
    "batched_token_ids": [123, 789],
    "baseline_finish": "eos|max_new_tokens|stopped",
    "batched_finish": "eos|max_new_tokens|stopped",
    "baseline_text": "decoded sequential output",
    "batched_text": "decoded batched output",
    "target_text": "deterministic target output",
}
NEXT_RUN_COMMAND = (
    "Rerun the bf16/SDPA timing path with per-task comparison export enabled. "
    "The copied reports show that current launchers preserve only aggregate "
    "counts plus an audit-trace path, so the rerun must also write "
    "timing_path_drift_comparisons.jsonl with one line per task and batch. "
    "Use the same timing protocol:\n"
    "modal run vantage_runtime_debian_app.py::launch_batched_pld_repeated_timing "
    "--split test --n 500 --repeats 1 --batch-sizes 2,4,8 "
    "--active-pool-size 32 --bucket-policy default --refill-policy continuous "
    "--dtype bf16 --attn sdpa "
    "--version continuous_batched_pld_bf16_sdpa_drift_trace_v1 --wait\n"
    "Then run the analyzer:\n"
    "python3 scripts/analyze_timing_path_drift.py "
    "--input analysis/continuous_batched_pld_final_repeats/<trace-run>/report.json "
    "--comparison-jsonl analysis/continuous_batched_pld_final_repeats/<trace-run>/"
    "timing_path_drift_comparisons.jsonl"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def unavailable(reason: str) -> dict[str, Any]:
    return {"available": False, "value": None, "reason": reason}


def task_count(row: dict[str, Any]) -> int | None:
    if row.get("n_tasks") is not None:
        return int(row["n_tasks"])
    match = row.get("output_match_count")
    mismatch = row.get("output_mismatch_count")
    if match is not None and mismatch is not None:
        return int(match) + int(mismatch)
    return None


def _num(value: Any, default: float = 0.0) -> float:
    if isinstance(value, dict) and value.get("mean") is not None:
        return float(value["mean"])
    if value is None:
        return default
    return float(value)


def as_task_id_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    ids: set[str] = set()
    for item in value:
        if isinstance(item, str):
            ids.add(item)
        elif isinstance(item, dict) and item.get("task_id") is not None:
            ids.add(str(item["task_id"]))
    return ids


def scan_json_for_task_evidence(obj: Any) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "has_task_ids": False,
        "has_per_task_outputs": False,
        "has_mismatch_task_ids": False,
        "has_first_mismatch": False,
        "has_differing_tokens": False,
        "has_finish_or_length": False,
        "has_quality_metrics": False,
        "has_logit_margins": False,
        "has_audit_trace_path": False,
        "mismatch_task_ids_by_key": {},
        "audit_trace_paths": [],
    }

    def visit(node: Any, context_key: str | None = None) -> None:
        if isinstance(node, dict):
            keys = set(node)
            if "task_id" in keys:
                evidence["has_task_ids"] = True
            token_keys = {
                "baseline_token_ids",
                "batched_token_ids",
                "output_token_ids",
                "completion_token_ids",
                "token_ids",
            }
            text_keys = {"baseline_text", "batched_text", "output", "text", "raw_text"}
            if "task_id" in keys and (keys & token_keys or keys & text_keys):
                evidence["has_per_task_outputs"] = True
            if keys & {"first_mismatch", "first_mismatch_position", "first_diff_index"}:
                evidence["has_first_mismatch"] = True
            if keys & {"differing_tokens", "num_differing_tokens", "token_diff_count"}:
                evidence["has_differing_tokens"] = True
            if keys & {
                "finish_reason",
                "finish_mismatch",
                "finish_mismatch_count",
                "length_mismatch",
                "generated_length",
                "generated_length_mismatches",
                "baseline_finish",
                "batched_finish",
            }:
                evidence["has_finish_or_length"] = True
            if keys & {
                "quality",
                "target_exact_match",
                "normalized_edit_distance",
                "edit_distance_to_gold",
                "target_text",
                "deterministic_target",
            }:
                evidence["has_quality_metrics"] = True
            if keys & {"logit_margin", "top1_top2_margin", "top_logit", "second_logit"}:
                evidence["has_logit_margins"] = True

            next_context = context_key
            if isinstance(node.get("key"), str):
                next_context = node["key"]
            elif isinstance(node.get("method"), str) and context_key is None:
                next_context = node["method"]

            for key, value in node.items():
                if key in {"mismatch_task_ids", "output_mismatch_task_ids", "mismatched_task_ids"}:
                    ids = as_task_id_set(value)
                    if ids:
                        evidence["has_mismatch_task_ids"] = True
                        evidence["mismatch_task_ids_by_key"][next_context or key] = sorted(ids)
                if key in {"audit_trace", "audit_trace_path", "trace"} and isinstance(value, str) and value:
                    evidence["has_audit_trace_path"] = True
                    evidence["audit_trace_paths"].append(value)
                visit(value, next_context)
        elif isinstance(node, list):
            for item in node:
                visit(item, context_key)

    visit(obj)
    evidence["audit_trace_paths"] = sorted(set(evidence["audit_trace_paths"]))
    return evidence


def inspect_report_jsons(paths: list[Path]) -> dict[str, Any]:
    inspected: list[dict[str, Any]] = []
    combined: dict[str, Any] = {
        "has_task_ids": False,
        "has_per_task_outputs": False,
        "has_mismatch_task_ids": False,
        "has_first_mismatch": False,
        "has_differing_tokens": False,
        "has_finish_or_length": False,
        "has_quality_metrics": False,
        "has_logit_margins": False,
        "has_audit_trace_path": False,
        "mismatch_task_ids_by_key": {},
        "audit_trace_paths": [],
    }
    seen: set[Path] = set()
    for path in paths:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            inspected.append({"path": rel(path), "exists": False})
            continue
        evidence = scan_json_for_task_evidence(load_json(path))
        inspected.append(
            {
                "path": rel(path),
                "exists": True,
                "evidence": {
                    k: v
                    for k, v in evidence.items()
                    if k not in {"mismatch_task_ids_by_key", "audit_trace_paths"}
                },
                "audit_trace_paths": evidence["audit_trace_paths"],
            }
        )
        for key, value in evidence.items():
            if key == "mismatch_task_ids_by_key":
                combined[key].update(value)
            elif key == "audit_trace_paths":
                combined[key].extend(value)
            else:
                combined[key] = bool(combined[key] or value)

    combined["audit_trace_paths"] = sorted(set(combined["audit_trace_paths"]))
    mismatch_sets = {key: set(ids) for key, ids in combined["mismatch_task_ids_by_key"].items()}
    overlap: dict[str, Any] = unavailable("No mismatch task IDs are present in the inspected report JSONs.")
    if len(mismatch_sets) >= 2:
        keys = sorted(mismatch_sets)
        intersection = set.intersection(*(mismatch_sets[key] for key in keys))
        union = set.union(*(mismatch_sets[key] for key in keys))
        overlap = {
            "available": True,
            "keys": keys,
            "intersection_count": len(intersection),
            "union_count": len(union),
            "jaccard": (len(intersection) / len(union)) if union else None,
        }

    return {
        "inspected_reports": inspected,
        "combined_evidence": {
            k: v
            for k, v in combined.items()
            if k not in {"mismatch_task_ids_by_key", "audit_trace_paths"}
        },
        "audit_trace_paths": combined["audit_trace_paths"],
        "mismatch_overlap": overlap,
    }


def token_edit_distance(a: list[int], b: list[int]) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if ca == cb else 1),
                )
            )
        previous = current
    return previous[-1]


def first_diff_index(a: list[int], b: list[int]) -> int | None:
    for idx, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return idx
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def normalized_edit_distance(a: str, b: str) -> float | None:
    if a == "" and b == "":
        return 0.0
    # The outputs are short generations in this experiment.  Refuse pathological
    # inputs instead of turning a report generator into an accidental quadratic job.
    if len(a) * len(b) > 30_000_000:
        return None
    left = list(a)
    right = list(b)
    dist = token_edit_distance([ord(c) for c in left], [ord(c) for c in right])
    return dist / max(1, max(len(left), len(right)))


def _nested(row: dict[str, Any], side: str, keys: Iterable[str]) -> Any:
    obj = row.get(side)
    if isinstance(obj, dict):
        for key in keys:
            if obj.get(key) is not None:
                return obj[key]
    for key in keys:
        compound = f"{side}_{key}"
        if row.get(compound) is not None:
            return row[compound]
    return None


def _tokens(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [int(x) for x in value]
    return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def read_comparison_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                task_id = str(row.get("task_id") or row.get("id") or f"{rel(path)}:{lineno}")
                batch_size = int(row.get("batch_size") or row.get("batch") or 0)
                baseline_tokens = _tokens(
                    _nested(row, "baseline", ("token_ids", "tokens", "output_token_ids"))
                    or row.get("baseline_token_ids")
                    or row.get("sequential_token_ids")
                )
                batched_tokens = _tokens(
                    _nested(row, "batched", ("token_ids", "tokens", "output_token_ids"))
                    or row.get("batched_token_ids")
                    or row.get("output_token_ids")
                )
                records.append(
                    {
                        "source": rel(path),
                        "task_id": task_id,
                        "run_id": int(row.get("run_id") or row.get("repeat") or 0),
                        "batch_size": batch_size,
                        "method": str(row.get("method") or "continuous_batched_pld_w128_n10"),
                        "baseline_token_ids": baseline_tokens,
                        "batched_token_ids": batched_tokens,
                        "baseline_finish": _text(
                            _nested(row, "baseline", ("finish_reason", "finish"))
                            or row.get("baseline_finish")
                        ),
                        "batched_finish": _text(
                            _nested(row, "batched", ("finish_reason", "finish"))
                            or row.get("batched_finish")
                        ),
                        "baseline_text": _text(
                            _nested(row, "baseline", ("text", "decoded", "output"))
                            or row.get("baseline_text")
                            or row.get("sequential_text")
                        ),
                        "batched_text": _text(
                            _nested(row, "batched", ("text", "decoded", "output"))
                            or row.get("batched_text")
                            or row.get("output_text")
                        ),
                        "target_text": _text(
                            row.get("target_text")
                            or row.get("deterministic_target")
                            or row.get("reference")
                            or row.get("gold")
                        ),
                    }
                )
    return records


def reconstruct_outputs_from_audit_trace(path: Path) -> dict[str, dict[str, Any]]:
    """Reconstruct batched emitted tokens from an audit trace.

    Audit traces are useful evidence, but by themselves they do not include the
    sequential baseline needed for drift comparison.
    """

    by_task: dict[str, dict[str, Any]] = defaultdict(lambda: {"token_ids": [], "finish_reason": None})
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = row.get("task_id")
            if not task_id:
                continue
            task = by_task[str(task_id)]
            if row.get("event") == "verify_scatter":
                task["token_ids"].extend(int(x) for x in row.get("emitted_tokens", []) or [])
            elif row.get("event") == "task_finish":
                task["finish_reason"] = row.get("finish_reason")
                task["generated_tokens"] = row.get("generated_tokens")
    return dict(by_task)


def trace_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_batch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    skipped = 0
    for row in records:
        if row["batch_size"] <= 0:
            skipped += 1
            continue
        if row["baseline_token_ids"] is None or row["batched_token_ids"] is None:
            skipped += 1
            continue
        by_batch[int(row["batch_size"])].append(row)

    rows: list[dict[str, Any]] = []
    mismatch_sets: dict[str, set[str]] = {}
    for batch in sorted(by_batch):
        group = by_batch[batch]
        mismatches: list[dict[str, Any]] = []
        first_positions: list[int] = []
        edit_counts: list[int] = []
        edit_fracs: list[float] = []
        length_mismatches = 0
        finish_mismatches = 0
        quality_deltas: list[float] = []
        baseline_quality: list[float] = []
        batched_quality: list[float] = []
        unavailable_quality = 0

        for row in group:
            base = row["baseline_token_ids"] or []
            got = row["batched_token_ids"] or []
            first = first_diff_index(base, got)
            token_dist = token_edit_distance(base, got)
            if len(base) != len(got):
                length_mismatches += 1
            if row["baseline_finish"] is not None and row["batched_finish"] is not None:
                if row["baseline_finish"] != row["batched_finish"]:
                    finish_mismatches += 1
            elif len(base) != len(got):
                # Length evidence is weaker than explicit finish reasons, so keep
                # it separate in the unavailable notes while still counting length.
                pass
            if first is not None:
                first_positions.append(first)
                edit_counts.append(token_dist)
                edit_fracs.append(token_dist / max(1, max(len(base), len(got))))
                mismatches.append(
                    {
                        "task_id": row["task_id"],
                        "first_mismatch_position": first,
                        "differing_token_count": token_dist,
                        "differing_token_fraction": token_dist / max(1, max(len(base), len(got))),
                        "baseline_len": len(base),
                        "batched_len": len(got),
                        "baseline_finish": row["baseline_finish"],
                        "batched_finish": row["batched_finish"],
                    }
                )

            if row["target_text"] and row["baseline_text"] is not None and row["batched_text"] is not None:
                base_q = normalized_edit_distance(row["baseline_text"], row["target_text"])
                got_q = normalized_edit_distance(row["batched_text"], row["target_text"])
                if base_q is None or got_q is None:
                    unavailable_quality += 1
                else:
                    baseline_quality.append(base_q)
                    batched_quality.append(got_q)
                    quality_deltas.append(got_q - base_q)
            else:
                unavailable_quality += 1

        mismatch_ids = {m["task_id"] for m in mismatches}
        mismatch_sets[f"b{batch}"] = mismatch_ids
        rows.append(
            {
                "key": f"continuous_batched_pld_w128_n10_b{batch}",
                "method": group[0]["method"],
                "batch_size": batch,
                "backend_comparison": "bf16/SDPA same-run vs sequential PLD",
                "tasks": len(group),
                "exact_task_matches_mean": len(group) - len(mismatches),
                "task_mismatches_mean": len(mismatches),
                "task_mismatch_fraction": len(mismatches) / max(1, len(group)),
                "task_mismatch_ids": {"available": True, "values": sorted(mismatch_ids)},
                "mismatch_examples": mismatches[:20],
                "first_mismatch": {
                    "available": bool(first_positions),
                    "median": median(first_positions) if first_positions else None,
                    "mean": mean(first_positions) if first_positions else None,
                    "min": min(first_positions) if first_positions else None,
                    "max": max(first_positions) if first_positions else None,
                },
                "differing_tokens": {
                    "available": bool(edit_counts),
                    "mean_count_among_mismatches": mean(edit_counts) if edit_counts else None,
                    "median_count_among_mismatches": median(edit_counts) if edit_counts else None,
                    "mean_fraction_among_mismatches": mean(edit_fracs) if edit_fracs else None,
                },
                "length_mismatch": {"available": True, "count": length_mismatches},
                "finish_mismatch": {
                    "available": all(r["baseline_finish"] is not None and r["batched_finish"] is not None for r in group),
                    "count": finish_mismatches,
                    "reason": None
                    if all(r["baseline_finish"] is not None and r["batched_finish"] is not None for r in group)
                    else "Some comparison rows do not contain both finish reasons.",
                },
                "quality_impact": {
                    "available": bool(quality_deltas),
                    "baseline_mean_normalized_edit_distance_to_target": mean(baseline_quality)
                    if baseline_quality
                    else None,
                    "batched_mean_normalized_edit_distance_to_target": mean(batched_quality)
                    if batched_quality
                    else None,
                    "mean_delta_batched_minus_baseline": mean(quality_deltas)
                    if quality_deltas
                    else None,
                    "unavailable_task_count": unavailable_quality,
                    "reason": None if quality_deltas else QUALITY_REASON,
                },
                "logit_margin": unavailable(LOGIT_REASON),
                "notes": "Computed from per-task comparison JSONL.",
            }
        )

    overlap: dict[str, Any] = unavailable("Mismatch overlap requires mismatch task IDs for at least two batches.")
    wanted = [key for key in ("b2", "b4", "b8") if key in mismatch_sets]
    if len(wanted) >= 2:
        union = set.union(*(mismatch_sets[key] for key in wanted))
        intersection = set.intersection(*(mismatch_sets[key] for key in wanted))
        overlap = {
            "available": True,
            "keys": wanted,
            "intersection_count": len(intersection),
            "union_count": len(union),
            "jaccard": len(intersection) / len(union) if union else None,
            "pairwise": {},
        }
        for i, left in enumerate(wanted):
            for right in wanted[i + 1 :]:
                u = mismatch_sets[left] | mismatch_sets[right]
                inter = mismatch_sets[left] & mismatch_sets[right]
                overlap["pairwise"][f"{left}_{right}"] = {
                    "intersection_count": len(inter),
                    "union_count": len(u),
                    "jaccard": len(inter) / len(u) if u else None,
                }

    return {
        "available": bool(rows),
        "rows": rows,
        "mismatch_overlap": overlap,
        "skipped_records": skipped,
        "comparison_record_count": len(records),
    }


def aggregate_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    source_rows = report.get("rows", [])
    by_key: dict[str, list[dict[str, Any]]] = {}
    for row in source_rows:
        if row.get("method") is None or row.get("batch_size") is None:
            continue
        key = f"{row.get('method')}_b{int(row.get('batch_size', 0))}"
        by_key.setdefault(key, []).append(row)

    output_rows: list[dict[str, Any]] = []
    for key in [
        "blazedit_pld_w128_n10_b1",
        "continuous_batched_pld_w128_n10_b2",
        "continuous_batched_pld_w128_n10_b4",
        "continuous_batched_pld_w128_n10_b8",
    ]:
        group = by_key.get(key, [])
        if not group:
            continue
        task_counts = [count for count in (task_count(r) for r in group) if count is not None]
        matches = mean(_num(r.get("output_match_count")) for r in group)
        mismatches = mean(_num(r.get("output_mismatch_count")) for r in group)
        tasks = int(mean(float(count) for count in task_counts)) if task_counts else int(matches + mismatches)
        output_rows.append(
            {
                "key": key,
                "method": group[0].get("method"),
                "batch_size": int(group[0].get("batch_size", 0)),
                "backend_comparison": "bf16/SDPA same-run vs sequential PLD",
                "tasks": tasks,
                "exact_task_matches_mean": matches,
                "task_mismatches_mean": mismatches,
                "task_mismatch_fraction": mismatches / max(1, tasks),
                "task_mismatch_ids": unavailable(AGGREGATE_ONLY_REASON),
                "first_mismatch": unavailable(TOKEN_DETAIL_REASON),
                "differing_tokens": unavailable(TOKEN_DETAIL_REASON),
                "length_mismatch": unavailable(FINISH_LENGTH_REASON),
                "finish_mismatch": unavailable(FINISH_LENGTH_REASON),
                "quality_impact": unavailable(QUALITY_REASON),
                "logit_margin": unavailable(LOGIT_REASON),
                "notes": (
                    "Only aggregate task-level output match/mismatch counts are present. "
                    "The artifact does not identify which tasks mismatched."
                ),
            }
        )
    return output_rows


def summarize(
    report: dict[str, Any],
    source: Path,
    inspect_paths: list[Path],
    comparison_jsonl: list[Path],
    audit_traces: list[Path],
) -> dict[str, Any]:
    inspection = inspect_report_jsons(inspect_paths)
    aggregate = aggregate_rows(report)
    comparison_records = read_comparison_jsonl(comparison_jsonl) if comparison_jsonl else []
    trace = trace_summary(comparison_records) if comparison_records else {
        "available": False,
        "rows": [],
        "mismatch_overlap": unavailable("No per-task comparison JSONL was supplied."),
        "skipped_records": 0,
        "comparison_record_count": 0,
    }

    audit_trace_summaries = []
    for path in audit_traces:
        if path.exists():
            reconstructed = reconstruct_outputs_from_audit_trace(path)
            audit_trace_summaries.append(
                {
                    "path": rel(path),
                    "exists": True,
                    "tasks_with_batched_tokens": len(reconstructed),
                    "tokens_reconstructed": sum(len(v.get("token_ids", [])) for v in reconstructed.values()),
                    "usable_for_drift": False,
                    "reason": "Audit traces reconstruct batched outputs only; sequential baseline token IDs are still required.",
                }
            )
        else:
            audit_trace_summaries.append({"path": rel(path), "exists": False})

    rows = trace["rows"] if trace["available"] else aggregate
    overlap = trace["mismatch_overlap"] if trace["available"] else inspection["mismatch_overlap"]
    for row in rows:
        row["mismatch_overlap"] = overlap

    missing_inputs = []
    if not trace["available"]:
        missing_inputs = [
            "task_id for every compared task",
            "batch_size for each batched run (b2, b4, b8)",
            "sequential blazedit_pld_w128_n10 output token IDs per task from the same bf16/SDPA run",
            "continuous_batched_pld_w128_n10 output token IDs per task for b2, b4, and b8",
            "baseline and batched finish reasons per task",
            "decoded baseline and batched outputs plus deterministic target text for edit-distance deltas",
        ]

    return {
        "source": str(source),
        "mode": "trace" if trace["available"] else "aggregate_only",
        "source_reports_inspected": inspection,
        "comparison_jsonl": [rel(p) for p in comparison_jsonl],
        "audit_traces": audit_trace_summaries,
        "rows": rows,
        "mismatch_overlap": overlap,
        "trace_summary": trace,
        "missing_inputs": missing_inputs,
        "required_trace_schema": NEXT_TRACE_SCHEMA,
        "next_run_needed": NEXT_RUN_COMMAND if missing_inputs else None,
        "limitations": [
            AGGREGATE_ONLY_REASON,
            "Mismatch overlap cannot be computed without mismatch task IDs for b2/b4/b8.",
            "First-mismatch positions cannot be computed without per-task token IDs.",
            "Differing-token counts/fractions cannot be computed without per-task token IDs.",
            "Length and finish mismatch counts require per-task generated lengths and finish reasons.",
            "Target edit-distance deltas require decoded outputs and deterministic targets.",
            "Top-1/top-2 logit margins require logits or margins, which are unavailable in these artifacts.",
        ]
        if missing_inputs
        else [LOGIT_REASON],
    }


def _fmt_unavailable(value: dict[str, Any] | Any, fallback: str = "unavailable") -> str:
    if isinstance(value, dict) and not value.get("available", False):
        return fallback
    return str(value)


def _quality_cell(row: dict[str, Any]) -> str:
    q = row.get("quality_impact", {})
    if not isinstance(q, dict) or not q.get("available"):
        return "unavailable"
    delta = q.get("mean_delta_batched_minus_baseline")
    if delta is None:
        return "unavailable"
    return f"{float(delta):+.4f} norm. edit"


def _first_cell(row: dict[str, Any]) -> str:
    first = row.get("first_mismatch", {})
    if not isinstance(first, dict) or not first.get("available"):
        return "unavailable"
    return f"median {float(first['median']):.1f}"


def _diff_cell(row: dict[str, Any]) -> str:
    diff = row.get("differing_tokens", {})
    if not isinstance(diff, dict) or not diff.get("available"):
        return "unavailable"
    return (
        f"{float(diff['mean_count_among_mismatches']):.1f} "
        f"({100.0 * float(diff['mean_fraction_among_mismatches']):.1f}%)"
    )


def _length_finish_cell(row: dict[str, Any]) -> str:
    length = row.get("length_mismatch", {})
    finish = row.get("finish_mismatch", {})
    if not isinstance(length, dict) or not length.get("available"):
        return "unavailable"
    finish_text = "unavailable"
    if isinstance(finish, dict) and finish.get("available"):
        finish_text = str(finish.get("count"))
    return f"length {length.get('count')}; finish {finish_text}"


def markdown_table(report: dict[str, Any], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "| Comparison | Batch | Exact task matches | Task mismatches | First mismatch | Differing tokens | Length/finish | Quality delta | Notes |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in report["rows"]:
        tasks = int(row["tasks"])
        matches = float(row["exact_task_matches_mean"])
        mismatches = float(row["task_mismatches_mean"])
        notes = "trace computed" if report["mode"] == "trace" else "aggregate counts only; trace rerun required"
        lines.append(
            "| {comparison} | {batch} | {matches:.0f}/{tasks} | {mismatches:.0f} "
            "({frac:.1f}%) | {first} | {diff} | {length_finish} | {quality} | {notes} |".format(
                comparison=row["backend_comparison"],
                batch=row["batch_size"],
                matches=matches,
                tasks=tasks,
                mismatches=mismatches,
                frac=100.0 * float(row.get("task_mismatch_fraction", 0.0)),
                first=_first_cell(row),
                diff=_diff_cell(row),
                length_finish=_length_finish_cell(row),
                quality=_quality_cell(row),
                notes=notes,
            )
        )
    return "\n".join(lines) + "\n"


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def latex_table(report: dict[str, Any]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Production/timing-path drift under bf16/SDPA.}",
        r"\label{tab:timing-path-drift}",
        r"\begin{tabular}{lllllllll}",
        r"\toprule",
        r"Comparison & Batch & Exact task matches & Task mismatches & First mismatch & Differing tokens & Length/finish & Quality delta & Notes \\",
        r"\midrule",
    ]
    for row in report["rows"]:
        tasks = int(row["tasks"])
        matches = float(row["exact_task_matches_mean"])
        mismatches = float(row["task_mismatches_mean"])
        notes = "trace computed" if report["mode"] == "trace" else "aggregate counts only; trace rerun required"
        cells = [
            row["backend_comparison"],
            str(row["batch_size"]),
            f"{matches:.0f}/{tasks}",
            f"{mismatches:.0f} ({100.0 * float(row.get('task_mismatch_fraction', 0.0)):.1f}%)",
            _first_cell(row),
            _diff_cell(row),
            _length_finish_cell(row),
            _quality_cell(row),
            notes,
        ]
        lines.append(" & ".join(latex_escape(c) for c in cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Timing Path Drift Analysis",
        "",
        f"Mode: `{report['mode']}`.",
        "Missing fields are reported as unavailable, not estimated.",
        "",
        "## Source inspection",
        "",
    ]
    inspection = report["source_reports_inspected"]
    for item in inspection["inspected_reports"]:
        if not item["exists"]:
            lines.append(f"- `{item['path']}`: not present")
            continue
        found = [name for name, present in item["evidence"].items() if present]
        detail = ", ".join(found) if found else "aggregate fields only"
        traces = item.get("audit_trace_paths") or []
        trace_note = f"; audit trace path(s): {', '.join(f'`{p}`' for p in traces)}" if traces else ""
        lines.append(f"- `{item['path']}`: {detail}{trace_note}")
    lines.extend(["", f"Mismatch overlap: {json.dumps(report['mismatch_overlap'], sort_keys=True)}", ""])
    lines.append("## Drift table")
    lines.append("")
    lines.extend(markdown_table(report, "Production/timing-path drift under bf16/SDPA.").splitlines()[2:])

    if report["missing_inputs"]:
        lines.extend(["", "## Missing inputs", ""])
        lines.extend(f"- {item}" for item in report["missing_inputs"])
        lines.extend(
            [
                "",
                "Required comparison JSONL schema:",
                "",
                "```json",
                json.dumps(NEXT_TRACE_SCHEMA, indent=2),
                "```",
                "",
                "Next analyzer command after rerun:",
                "",
                "The trace-producing rerun must use the same bf16/SDPA timing protocol and emit `timing_path_drift_comparisons.jsonl`; the copied current launch output does not contain this file.",
                "",
                "```bash",
                "modal run vantage_runtime_debian_app.py::launch_batched_pld_repeated_timing \\",
                "  --split test \\",
                "  --n 500 \\",
                "  --repeats 1 \\",
                "  --batch-sizes 2,4,8 \\",
                "  --active-pool-size 32 \\",
                "  --bucket-policy default \\",
                "  --refill-policy continuous \\",
                "  --dtype bf16 \\",
                "  --attn sdpa \\",
                "  --version continuous_batched_pld_bf16_sdpa_drift_trace_v1 \\",
                "  --wait",
                "```",
                "",
                "```bash",
                "python3 scripts/analyze_timing_path_drift.py \\",
                "  --input analysis/continuous_batched_pld_final_repeats/<trace-run>/report.json \\",
                "  --comparison-jsonl analysis/continuous_batched_pld_final_repeats/<trace-run>/timing_path_drift_comparisons.jsonl",
                "```",
            ]
        )
    if report["audit_traces"]:
        lines.extend(["", "## Audit traces", ""])
        for item in report["audit_traces"]:
            if item.get("exists"):
                lines.append(
                    f"- `{item['path']}`: reconstructed {item['tokens_reconstructed']} batched tokens "
                    f"for {item['tasks_with_batched_tokens']} tasks; {item['reason']}"
                )
            else:
                lines.append(f"- `{item['path']}`: not present")
    lines.extend(["", "## Limitations"])
    lines.extend(f"- {item}" for item in report["limitations"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_doc(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Drift Trace Requirements",
        "",
        "The current bf16/SDPA timing artifact supports aggregate drift counts only.",
        "Token-level characterization requires a per-task comparison trace.",
        "",
        "## Computed from available artifacts",
        "",
        "- Exact task match and mismatch counts by batch.",
        "- The existence of a batch-8 audit trace path in the copied report metadata.",
        "",
        "## Unavailable without trace export",
        "",
        "- First mismatch token position.",
        "- Differing token count and fraction.",
        "- Mismatch task-ID overlap across b2/b4/b8.",
        "- Length and finish-reason mismatch counts.",
        "- Target edit-distance quality delta.",
        "",
        "## Required JSONL schema",
        "",
        "Each line should contain:",
        "",
        "```json",
        json.dumps(NEXT_TRACE_SCHEMA, indent=2),
        "```",
        "",
        "The analyzer also accepts nested `baseline` and `batched` objects with `token_ids`, `text`, and `finish_reason` fields.",
        "",
        "## Next command",
        "",
        "The next run must use the same bf16/SDPA timing protocol and additionally export per-task comparison rows. The current copied reports do not include the required `timing_path_drift_comparisons.jsonl` file.",
        "",
        "```bash",
        "modal run vantage_runtime_debian_app.py::launch_batched_pld_repeated_timing \\",
        "  --split test \\",
        "  --n 500 \\",
        "  --repeats 1 \\",
        "  --batch-sizes 2,4,8 \\",
        "  --active-pool-size 32 \\",
        "  --bucket-policy default \\",
        "  --refill-policy continuous \\",
        "  --dtype bf16 \\",
        "  --attn sdpa \\",
        "  --version continuous_batched_pld_bf16_sdpa_drift_trace_v1 \\",
        "  --wait",
        "```",
        "",
        "After that trace-producing rerun has copied the report and comparison JSONL locally:",
        "",
        "```bash",
        "python3 scripts/analyze_timing_path_drift.py \\",
        "  --input analysis/continuous_batched_pld_final_repeats/<trace-run>/report.json \\",
        "  --comparison-jsonl analysis/continuous_batched_pld_final_repeats/<trace-run>/timing_path_drift_comparisons.jsonl",
        "```",
        "",
        "If only audit traces are available, pass them with `--audit-trace` for evidence accounting, but they are not sufficient for drift because they lack sequential baseline token IDs.",
        "",
    ]
    if report["source_reports_inspected"]["audit_trace_paths"]:
        lines.extend(["## Audit trace paths seen", ""])
        lines.extend(f"- `{p}`" for p in report["source_reports_inspected"]["audit_trace_paths"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--table-md", default=str(DEFAULT_TABLE_MD))
    parser.add_argument("--table-tex", default=str(DEFAULT_TABLE_TEX))
    parser.add_argument("--final-table-md", default=str(DEFAULT_FINAL_TABLE_MD))
    parser.add_argument("--final-table-tex", default=str(DEFAULT_FINAL_TABLE_TEX))
    parser.add_argument("--doc", default=str(DEFAULT_DOC))
    parser.add_argument("--inspect-report", action="append", default=[])
    parser.add_argument("--comparison-jsonl", action="append", default=[])
    parser.add_argument("--audit-trace", action="append", default=[])
    args = parser.parse_args()

    input_path = Path(args.input)
    inspect_paths = [input_path, *DEFAULT_INSPECT_REPORTS, *(Path(p) for p in args.inspect_report)]
    report = summarize(
        load_json(input_path),
        input_path,
        inspect_paths,
        [Path(p) for p in args.comparison_jsonl],
        [Path(p) for p in args.audit_trace],
    )

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    table_md = Path(args.table_md)
    table_tex = Path(args.table_tex)
    final_table_md = Path(args.final_table_md)
    final_table_tex = Path(args.final_table_tex)
    doc = Path(args.doc)
    for path in [output_json, output_md, table_md, table_tex, final_table_md, final_table_tex, doc]:
        path.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, output_md)
    table_text = markdown_table(report, "Production/timing-path drift under bf16/SDPA.")
    table_md.write_text(table_text, encoding="utf-8")
    final_table_md.write_text(table_text, encoding="utf-8")
    tex_text = latex_table(report)
    table_tex.write_text(tex_text, encoding="utf-8")
    final_table_tex.write_text(tex_text, encoding="utf-8")
    write_doc(report, doc)
    print(
        "wrote "
        + ", ".join(
            str(p)
            for p in [
                output_json,
                output_md,
                table_md,
                table_tex,
                final_table_md,
                final_table_tex,
                doc,
            ]
        )
    )


if __name__ == "__main__":
    main()
