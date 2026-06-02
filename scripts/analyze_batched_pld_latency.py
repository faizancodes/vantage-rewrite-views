#!/usr/bin/env python3
"""Latency/throughput analysis for Continuous Batched PLD reports.

The timing harness is an offline, all-at-once aggregate-throughput benchmark.
It does not model external request arrivals, and it does not instrument queue
wait or time-to-first-token.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCOPE_NOTE = (
    "Continuous batching improves offline all-at-once aggregate throughput; it "
    "is not a single-request latency result. The available task latency is "
    "active-pool residency from scheduler admission through task completion. "
    "External queue wait and time-to-first-token are not instrumented."
)


def _rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    if "batched" in report:
        return list(report["batched"])
    return list(report.get("rows", []))


def _first_number(row: dict[str, Any], keys: tuple[str, ...], default: float = 0.0) -> float:
    for key in keys:
        value = row.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return default


def _aggregate_tok_s(row: dict[str, Any]) -> float:
    return _first_number(
        row,
        (
            "generated_tokens_per_sec",
            "aggregate_tok_s_mean",
            "aggregate_tok_s",
            "tok_s",
            "tokens_per_sec",
        ),
    )


def _sequential_tok_s(report: dict[str, Any], rows: list[dict[str, Any]]) -> float:
    seq = report.get("sequential", {}) or {}
    seq_tps = _first_number(seq, ("tokens_per_sec", "tok_s"))
    if seq_tps > 0.0:
        return seq_tps
    for row in rows:
        method = str(row.get("method", ""))
        if method == "blazedit_pld_w128_n10" or method.startswith("blazedit"):
            tok_s = _aggregate_tok_s(row)
            if tok_s > 0.0:
                return tok_s
    for row in rows:
        if _first_number(row, ("speedup_mean", "speedup_vs_same_run_sequential")) == 1.0:
            tok_s = _aggregate_tok_s(row)
            if tok_s > 0.0:
                return tok_s
    return 0.0


def _speedup(row: dict[str, Any], aggregate_tok_s: float, seq_tps: float) -> float:
    direct = _first_number(row, ("speedup_mean", "speedup_vs_same_run_sequential"), default=-1.0)
    if direct >= 0.0:
        return direct
    if seq_tps > 0.0:
        return aggregate_tok_s / seq_tps
    return 0.0


def _has_local_audit_trace(report: dict[str, Any]) -> bool:
    paths: list[str] = []
    for row in _rows(report):
        raw = row.get("audit_trace_path")
        if raw:
            paths.append(str(raw))
    trace = report.get("trace")
    if trace:
        paths.append(str(trace))
    return any(Path(path).exists() for path in paths)


def _source_has_task_latencies(rows: list[dict[str, Any]]) -> bool:
    return any(bool(row.get("task_latency_ms")) for row in rows)


def _serving_decision(report: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "online_simulation_feasible_without_gpu_rerun": False,
        "decision": "do_not_run_online_arrival_simulation_from_current_artifacts",
        "reasons": [
            (
                "The scheduler harness consumes a static pending deque; all benchmark "
                "tasks are available at time zero and refill occurs whenever active "
                "slots open."
            ),
            (
                "Per-task latency in the report is measured from active-pool "
                "admission to completion, so it omits any external arrival-to-admission "
                "queue wait."
            ),
            (
                "No time-to-first-token event is recorded in report.json, and the "
                "available local artifacts do not contain a replayable per-step audit "
                "trace for serving reconstruction."
            ),
            (
                "Bucket-level forward summaries and aggregate per-task completion "
                "latencies are insufficient to replay online co-scheduling, because "
                "service time changes with bucket grouping, accepted lengths, refill "
                "timing, and concurrent tasks."
            ),
        ],
        "source_artifact_observations": {
            "has_task_latency_ms": _source_has_task_latencies(rows),
            "has_local_audit_trace": _has_local_audit_trace(report),
            "has_queue_wait_ms": any("queue_wait_ms" in row for row in rows),
            "has_ttft_ms": any(("ttft_ms" in row or "time_to_first_token_ms" in row) for row in rows),
        },
        "allowed_claim_scope": [
            "offline all-at-once aggregate generated-token throughput",
            "speedup versus the same offline sequential PLD baseline",
            "active-pool admission-to-completion latency summaries, if clearly labeled",
        ],
        "disallowed_claim_scope": [
            "external queue wait",
            "time-to-first-token",
            "online Poisson or bursty arrival serving latency",
            "single-request latency improvement",
            "production SLO or tail-latency serving claims",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = Path(args.report_json)
    report = json.loads(report_path.read_text())
    seq = report.get("sequential", {})
    rows = _rows(report)
    seq_tps = _sequential_tok_s(report, rows)
    out_rows = []
    for row in rows:
        latency = row.get("task_latency_summary_ms", {}) or {}
        aggregate_tok_s = _aggregate_tok_s(row)
        out_rows.append(
            {
                "config_id": row.get("config_id")
                or (
                    f"{row.get('method', 'unknown')}_"
                    f"b{row.get('batch_size')}_pool{row.get('active_pool_size')}"
                    f"_run{row.get('run_id')}"
                ),
                "batch_size": row.get("batch_size"),
                "active_pool_size": row.get("active_pool_size"),
                "refill_policy": row.get("refill_policy", "continuous"),
                "bucket_policy": row.get("bucket_policy", "custom"),
                "aggregate_tok_s": aggregate_tok_s,
                "speedup_vs_sequential": _speedup(row, aggregate_tok_s, seq_tps),
                "latency_mean_ms": latency.get("mean", 0.0),
                "latency_p50_ms": latency.get("p50", 0.0),
                "latency_p90_ms": latency.get("p90", 0.0),
                "latency_p99_ms": latency.get("p99", 0.0),
                "active_tasks_mean": row.get("active_tasks_mean", 0.0),
                "verifier_forwards": row.get("verifier_forwards", 0),
                "error": row.get("error", ""),
            }
        )

    payload = {
        "source_report": str(report_path),
        "sequential": seq,
        "sequential_tok_s": seq_tps,
        "rows": out_rows,
        "note": SCOPE_NOTE,
        "serving_decision": _serving_decision(report, rows),
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Batched PLD Latency/Serving Scope",
        "",
        payload["note"],
        "",
        "## Serving decision",
        "",
        "Online arrival-process simulation is not feasible from the current local "
        "artifacts without a new instrumented scheduler run.",
        "",
        "Reasons:",
    ]
    for reason in payload["serving_decision"]["reasons"]:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "Claims should be scoped to offline all-at-once aggregate throughput. "
            "Do not report queue wait or TTFT.",
            "",
            "## Offline throughput/active-pool latency",
            "",
            f"sequential aggregate tok/s: `{seq_tps:.1f}`",
            "",
            "| config | tok/s | speedup | active-pool latency mean ms | p50 | p90 | p99 | active mean | forwards |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(out_rows, key=lambda r: r["speedup_vs_sequential"], reverse=True):
        lines.append(
            f"| {row['config_id']} | {row['aggregate_tok_s']:.1f} | "
            f"{row['speedup_vs_sequential']:.3f} | {row['latency_mean_ms']:.1f} | "
            f"{row['latency_p50_ms']:.1f} | {row['latency_p90_ms']:.1f} | "
            f"{row['latency_p99_ms']:.1f} | {row['active_tasks_mean']:.1f} | "
            f"{row['verifier_forwards']} |"
        )
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print((out / "report.md").read_text())


if __name__ == "__main__":
    main()
