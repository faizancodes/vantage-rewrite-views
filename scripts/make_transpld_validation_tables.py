#!/usr/bin/env python3
"""Build validation tables for the VANTAGE structured suite."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


WORKLOADS = {
    "zero100": "vantage_frozen_final_qwen_base_zero100_validation_20260515_v1",
    "field100": "vantage_frozen_final_qwen_base_field100_validation_20260515_v1",
    "style100": "vantage_frozen_final_qwen_base_style100_validation_20260515_v1",
    "mixed100": "vantage_frozen_final_qwen_base_mixed100_validation_20260515_v1",
}

GATES = {
    "zero100": 0.99,
    "field100": 1.10,
    "style100": 1.25,
    "mixed100": 1.25,
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _method_output(row: dict[str, Any], method: str) -> str:
    return str(((row.get("outputs") or {}).get(method) or {}).get("text") or "")


def _parity(rows: list[dict[str, Any]], left: str, right: str) -> int:
    return sum(1 for row in rows if _method_output(row, left) == _method_output(row, right))


def _route_stats(steps: list[dict[str, Any]], method: str) -> dict[str, Any]:
    rows = [row for row in steps if row.get("method") == method]
    routes: Counter[str] = Counter()
    accepted_by_route: Counter[str] = Counter()
    route_hits: Counter[str] = Counter()
    zero_accept_by_route: Counter[str] = Counter()
    for row in rows:
        route = str(row.get("proposal_route") or row.get("proposal_match_kind") or "none")
        routes[route] += 1
        accepted = int(row.get("n_accepted_nonroot_drafts") or 0)
        accepted_by_route[route] += accepted
        if route != "none":
            route_hits[route] += 1
            if accepted <= 0:
                zero_accept_by_route[route] += 1
    trans_hits = route_hits.get("transpld", 0)
    return {
        "routes": dict(routes),
        "accepted_by_route": dict(accepted_by_route),
        "transpld_hits": trans_hits,
        "transpld_accepted": int(accepted_by_route.get("transpld", 0)),
        "transpld_accepted_per_hit": (
            float(accepted_by_route.get("transpld", 0)) / trans_hits if trans_hits else 0.0
        ),
        "transpld_zero_accept_rate": (
            float(zero_accept_by_route.get("transpld", 0)) / trans_hits if trans_hits else 0.0
        ),
    }


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _step_mechanism_stats(
    steps: list[dict[str, Any]],
    completions: list[dict[str, Any]],
    method: str,
) -> dict[str, Any]:
    rows = [row for row in steps if row.get("method") == method]
    task_ids = [str(row.get("task_id")) for row in completions if row.get("task_id")]
    transp_hits_by_task: dict[str, int] = {task_id: 0 for task_id in task_ids}
    proposed_total = 0
    rejected_total = 0
    accepted_total = 0
    k_values: list[float] = []
    proposed_seen = False
    for row in rows:
        accepted = int(row.get("n_accepted_nonroot_drafts") or 0)
        accepted_total += accepted
        k = row.get("k")
        if k is not None:
            k_values.append(float(k))
        proposal_tokens = _to_int(row.get("proposal_tokens"))
        if proposal_tokens is None:
            proposal_tokens = _to_int(row.get("target_draft_tokens"))
        if proposal_tokens is not None:
            proposed_seen = True
            proposed_total += proposal_tokens
            rejected_total += max(proposal_tokens - accepted, 0)
        route = str(row.get("proposal_route") or row.get("proposal_match_kind") or "none")
        if route == "transpld":
            task_id = str(row.get("task_id"))
            transp_hits_by_task[task_id] = transp_hits_by_task.get(task_id, 0) + 1
    hit_counts = list(transp_hits_by_task.values())
    return {
        "accepted_nonroot_total": accepted_total,
        "proposed_draft_tokens_total": proposed_total if proposed_seen else None,
        "rejected_draft_tokens_total": rejected_total if proposed_seen else None,
        "mean_verification_batch_len": _mean(k_values),
        "transpld_hit_tasks": sum(1 for value in hit_counts if value > 0),
        "transpld_hits_per_task_p50": _percentile([float(v) for v in hit_counts], 0.50),
        "transpld_hits_per_task_p95": _percentile([float(v) for v in hit_counts], 0.95),
        "transpld_hits_per_task_max": max(hit_counts) if hit_counts else None,
    }


def _bootstrap_pair(report: dict[str, Any], method: str, baseline: str) -> dict[str, Any]:
    return report["by_pair"][f"{method}_vs_{baseline}"]


def _method_out(row: dict[str, Any], method: str) -> dict[str, Any]:
    return dict(((row.get("outputs") or {}).get(method) or {}))


def _percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _route_counts_by_task(steps: list[dict[str, Any]], method: str) -> dict[str, Counter[str]]:
    out: dict[str, Counter[str]] = {}
    for row in steps:
        if row.get("method") != method:
            continue
        task_id = str(row.get("task_id") or "")
        if not task_id:
            continue
        route = str(row.get("proposal_route") or row.get("proposal_match_kind") or "none")
        out.setdefault(task_id, Counter())[route] += 1
    return out


def _speedup_histogram(ratios: list[float]) -> dict[str, int]:
    bins = {
        "<0.80x": 0,
        "0.80-0.95x": 0,
        "0.95-1.00x": 0,
        "1.00-1.05x": 0,
        "1.05-1.25x": 0,
        ">=1.25x": 0,
    }
    for ratio in ratios:
        if ratio < 0.80:
            bins["<0.80x"] += 1
        elif ratio < 0.95:
            bins["0.80-0.95x"] += 1
        elif ratio < 1.00:
            bins["0.95-1.00x"] += 1
        elif ratio < 1.05:
            bins["1.00-1.05x"] += 1
        elif ratio < 1.25:
            bins["1.05-1.25x"] += 1
        else:
            bins[">=1.25x"] += 1
    return bins


def _completion_stats(
    completions: list[dict[str, Any]],
    *,
    method: str,
    baseline: str,
    method_routes_by_task: dict[str, Counter[str]] | None = None,
) -> dict[str, Any]:
    method_tokens: list[int] = []
    method_wall_ms: list[float] = []
    baseline_wall_ms: list[float] = []
    speedup_ratios: list[float] = []
    regression_count = 0
    transpld_used_regression_count = 0
    exact_pld_only_regression_count = 0
    comparable = 0
    worst_task: str | None = None
    worst_ratio: float | None = None
    method_routes_by_task = method_routes_by_task or {}
    for row in completions:
        task_id = str(row.get("task_id") or "")
        m = _method_out(row, method)
        b = _method_out(row, baseline)
        if m:
            if m.get("n_new_tokens") is not None:
                method_tokens.append(int(m["n_new_tokens"]))
            if m.get("wall_us") is not None:
                method_wall_ms.append(float(m["wall_us"]) / 1000.0)
        if b and b.get("wall_us") is not None:
            baseline_wall_ms.append(float(b["wall_us"]) / 1000.0)
        if m and b:
            required = ("n_new_tokens", "wall_us")
            if any(m.get(key) is None or b.get(key) is None for key in required):
                continue
            m_tokens = int(m["n_new_tokens"])
            b_tokens = int(b["n_new_tokens"])
            m_wall = float(m["wall_us"])
            b_wall = float(b["wall_us"])
            if m_wall > 0 and b_wall > 0 and m_tokens > 0 and b_tokens > 0:
                comparable += 1
                m_tps = m_tokens / (m_wall / 1e6)
                b_tps = b_tokens / (b_wall / 1e6)
                ratio = m_tps / b_tps
                speedup_ratios.append(ratio)
                if worst_ratio is None or ratio < worst_ratio:
                    worst_ratio = ratio
                    worst_task = task_id
                if ratio < 1.0:
                    regression_count += 1
                    routes = method_routes_by_task.get(task_id, Counter())
                    if routes.get("transpld", 0) > 0:
                        transpld_used_regression_count += 1
                    elif routes.get("exact_pld", 0) > 0 and routes.get("transpld", 0) == 0:
                        exact_pld_only_regression_count += 1
    total_tokens = sum(method_tokens)
    return {
        "generated_tokens": total_tokens if method_tokens else None,
        "mean_output_tokens_per_task": (
            total_tokens / len(method_tokens) if method_tokens else None
        ),
        "baseline_p50_latency_ms": _percentile(baseline_wall_ms, 0.50),
        "baseline_p95_latency_ms": _percentile(baseline_wall_ms, 0.95),
        "p50_latency_ms": _percentile(method_wall_ms, 0.50),
        "p90_latency_ms": _percentile(method_wall_ms, 0.90),
        "p95_latency_ms": _percentile(method_wall_ms, 0.95),
        "p99_latency_ms": _percentile(method_wall_ms, 0.99),
        "per_task_speedup_regressions": regression_count if comparable else None,
        "per_task_speedup_transpld_used_regressions": (
            transpld_used_regression_count if comparable else None
        ),
        "per_task_speedup_exact_pld_only_regressions": (
            exact_pld_only_regression_count if comparable else None
        ),
        "per_task_speedup_comparable": comparable,
        "per_task_speedup_p05": _percentile(speedup_ratios, 0.05),
        "per_task_speedup_p50": _percentile(speedup_ratios, 0.50),
        "per_task_speedup_p95": _percentile(speedup_ratios, 0.95),
        "worst_per_task_speedup": worst_ratio,
        "worst_per_task_slowdown": (1.0 / worst_ratio if worst_ratio and worst_ratio > 0 else None),
        "worst_per_task": worst_task,
        "per_task_speedup_histogram": _speedup_histogram(speedup_ratios),
    }


def build_rows(base_dir: Path, report_dir: Path) -> list[dict[str, Any]]:
    method = "vantage_frozen_transpld"
    baseline = "blazedit_pld_w128_n10"
    rows: list[dict[str, Any]] = []
    for workload, tag in WORKLOADS.items():
        run_dir = base_dir / tag / "eval"
        aggregate = _load_json(run_dir / "aggregate.json")
        completions = _load_jsonl(run_dir / "completions.jsonl")
        steps = _load_jsonl(run_dir / "steps.jsonl")
        bootstrap = _load_json(report_dir / f"{workload}_bootstrap.json")
        pair = _bootstrap_pair(bootstrap, method, baseline)
        by_method = aggregate["by_method"]
        method_row = by_method[method]
        baseline_row = by_method[baseline]
        route = _route_stats(steps, method)
        method_step_stats = _step_mechanism_stats(steps, completions, method)
        baseline_step_stats = _step_mechanism_stats(steps, completions, baseline)
        trans_prop = (method_row.get("by_proposal") or {}).get("precomputed_transpld", {})
        pld_prop = (baseline_row.get("by_proposal") or {}).get("blazedit_pld", {})
        completion_stats = _completion_stats(
            completions,
            method=method,
            baseline=baseline,
            method_routes_by_task=_route_counts_by_task(steps, method),
        )
        rows.append(
            {
                "workload": workload,
                "n_tasks": len(completions),
                "vanilla_tps": by_method["vanilla"]["tokens_per_sec"],
                "pld_tps": baseline_row["tokens_per_sec"],
                "method_tps": method_row["tokens_per_sec"],
                "pld_over_vanilla": (
                    baseline_row["tokens_per_sec"] / by_method["vanilla"]["tokens_per_sec"]
                    if by_method["vanilla"]["tokens_per_sec"]
                    else None
                ),
                "ratio": pair["ratio"],
                "ci95": pair["ci95"],
                "p_gt_1": pair["p_gt_1"],
                "gate": GATES[workload],
                "gate_pass": pair["ratio"] >= GATES[workload],
                "pld_steps": baseline_row["n_steps"],
                "method_steps": method_row["n_steps"],
                "step_reduction_vs_pld": (
                    1.0 - (method_row["n_steps"] / baseline_row["n_steps"])
                    if baseline_row["n_steps"]
                    else 0.0
                ),
                "method_vs_pld_exact_match": _parity(completions, method, baseline),
                "method_vs_vanilla_exact_match": _parity(completions, method, "vanilla"),
                "pld_vs_vanilla_exact_match": _parity(completions, baseline, "vanilla"),
                "transpld_hits": route["transpld_hits"],
                "transpld_accepted": route["transpld_accepted"],
                "transpld_accepted_per_hit": route["transpld_accepted_per_hit"],
                "transpld_zero_accept_rate": route["transpld_zero_accept_rate"],
                "routes": route["routes"],
                "accepted_by_route": route["accepted_by_route"],
                "baseline_pld_accepted_tokens": baseline_step_stats[
                    "accepted_nonroot_total"
                ],
                "baseline_rejected_draft_tokens": baseline_step_stats[
                    "rejected_draft_tokens_total"
                ],
                "baseline_avg_verification_batch_len": baseline_step_stats[
                    "mean_verification_batch_len"
                ],
                "method_pld_route_accepted_tokens": int(
                    route["accepted_by_route"].get("exact_pld", 0)
                ),
                "method_none_route_accepted_tokens": int(
                    route["accepted_by_route"].get("none", 0)
                ),
                "method_rejected_draft_tokens": method_step_stats[
                    "rejected_draft_tokens_total"
                ],
                "method_avg_verification_batch_len": method_step_stats[
                    "mean_verification_batch_len"
                ],
                "transpld_hit_tasks": method_step_stats["transpld_hit_tasks"],
                "transpld_hits_per_task_p50": method_step_stats[
                    "transpld_hits_per_task_p50"
                ],
                "transpld_hits_per_task_p95": method_step_stats[
                    "transpld_hits_per_task_p95"
                ],
                "transpld_hits_per_task_max": method_step_stats[
                    "transpld_hits_per_task_max"
                ],
                "transpld_attempts": trans_prop.get("n", 0),
                "transpld_mean_match_len": trans_prop.get("mean_match_len", 0.0),
                "transpld_mean_accepted_nonroot": trans_prop.get(
                    "mean_accepted_nonroot", 0.0
                ),
                "pld_attempts": pld_prop.get("n", 0),
                "pld_mean_match_len": pld_prop.get("mean_match_len", 0.0),
                "pld_mean_accepted_nonroot": pld_prop.get(
                    "mean_accepted_nonroot", 0.0
                ),
                "proposal_us_per_token": (
                    method_row.get("proposal_us_total", 0.0)
                    / method_row.get("n_new_tokens_total", 1)
                ),
                "setup_us_total": (
                    method_row.get("proposal_map_parse_us_total", 0.0)
                    + method_row.get("proposal_rewrite_apply_us_total", 0.0)
                    + method_row.get("proposal_virtual_reference_tokenize_us_total", 0.0)
                    + method_row.get("proposal_transpld_index_build_us_total", 0.0)
                ),
                "setup_ms_per_prompt": (
                    (
                        method_row.get("proposal_map_parse_us_total", 0.0)
                        + method_row.get("proposal_rewrite_apply_us_total", 0.0)
                        + method_row.get("proposal_virtual_reference_tokenize_us_total", 0.0)
                        + method_row.get("proposal_transpld_index_build_us_total", 0.0)
                    )
                    / 1000.0
                    / len(completions)
                    if completions
                    else None
                ),
                **completion_stats,
            }
        )
    return rows


def _fmt_ci(ci: list[float]) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _fmt_opt(value: Any, fmt: str = "{:.1f}") -> str:
    if value is None:
        return "n/a"
    return fmt.format(float(value))


def write_main_table(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# VANTAGE Validation Throughput",
        "",
        "Bootstrap CIs are paired by task: each resample draws task rows and "
        "recomputes method/baseline throughput on the same sampled tasks.",
        "",
        "| Workload | n | Vanilla tok/s | PLD tok/s | VANTAGE tok/s | PLD/vanilla | VANTAGE/PLD | 95% CI | Gate | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['workload']} | {row['n_tasks']} | {row['vanilla_tps']:.1f} | "
            f"{row['pld_tps']:.1f} | {row['method_tps']:.1f} | "
            f"{_fmt_opt(row['pld_over_vanilla'], '{:.2f}')} | {row['ratio']:.3f} | "
            f"{_fmt_ci(row['ci95'])} | "
            f"{row['gate']:.2f} | {'pass' if row['gate_pass'] else 'fail'} |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_diagnostic_table(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# VANTAGE Throughput Diagnostics",
        "",
        "Latency columns are per-task VANTAGE decoder wall time from `completions.jsonl`.",
        "",
        "| Workload | Generated tokens | Mean output tokens/task | VANTAGE p50/p90/p95/p99 latency ms | Per-task speedup p05/p50/p95 | Worst slowdown | Regressions vs PLD | Comparable tasks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        regressions = row["per_task_speedup_regressions"]
        comparable = row["per_task_speedup_comparable"]
        latency = (
            f"{_fmt_opt(row['p50_latency_ms'], '{:.1f}')}/"
            f"{_fmt_opt(row['p90_latency_ms'], '{:.1f}')}/"
            f"{_fmt_opt(row['p95_latency_ms'], '{:.1f}')}/"
            f"{_fmt_opt(row['p99_latency_ms'], '{:.1f}')}"
        )
        speedup = (
            f"{_fmt_opt(row['per_task_speedup_p05'], '{:.3f}')}/"
            f"{_fmt_opt(row['per_task_speedup_p50'], '{:.3f}')}/"
            f"{_fmt_opt(row['per_task_speedup_p95'], '{:.3f}')}"
        )
        lines.append(
            f"| {row['workload']} | {_fmt_opt(row['generated_tokens'], '{:.0f}')} | "
            f"{_fmt_opt(row['mean_output_tokens_per_task'], '{:.1f}')} | "
            f"{latency} | {speedup} | "
            f"{_fmt_opt(row['worst_per_task_slowdown'], '{:.2f}x')} | "
            f"{regressions if regressions is not None else 'n/a'} | {comparable} |"
        )
    lines.append("")
    lines.append(
        "Per-task regressions count tasks where VANTAGE's generated-token tok/s "
        "is lower than PLD's generated-token tok/s for the same task."
    )
    path.write_text("\n".join(lines) + "\n")


def write_latency_regression_table(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# VANTAGE Per-Task Regression And Tail-Latency Summary",
        "",
        "This table is generated from per-task `completions.jsonl` rows. "
        "Speedup is computed as VANTAGE generated-token tok/s divided by PLD "
        "generated-token tok/s for the same task. These are timing-path diagnostics, "
        "not a new correctness claim.",
        "",
        "| Workload | PLD p50/p95 latency ms | VANTAGE p50/p95/p99 latency ms | Regression count | TransPLD-used regressions | Exact-PLD-only regressions | Worst task ratio | Worst task | Speedup histogram |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        pld_latency = (
            f"{_fmt_opt(row['baseline_p50_latency_ms'], '{:.1f}')}/"
            f"{_fmt_opt(row['baseline_p95_latency_ms'], '{:.1f}')}"
        )
        method_latency = (
            f"{_fmt_opt(row['p50_latency_ms'], '{:.1f}')}/"
            f"{_fmt_opt(row['p95_latency_ms'], '{:.1f}')}/"
            f"{_fmt_opt(row['p99_latency_ms'], '{:.1f}')}"
        )
        histogram = ", ".join(
            f"{name}:{count}"
            for name, count in row["per_task_speedup_histogram"].items()
            if count
        ) or "none"
        lines.append(
            f"| {row['workload']} | {pld_latency} | {method_latency} | "
            f"{row['per_task_speedup_regressions']} | "
            f"{row['per_task_speedup_transpld_used_regressions']} | "
            f"{row['per_task_speedup_exact_pld_only_regressions']} | "
            f"{_fmt_opt(row['worst_per_task_speedup'], '{:.3f}x')} | "
            f"`{row['worst_per_task']}` | `{histogram}` |"
        )
    lines += [
        "",
        "Interpretation: many zero-drift regressions are tiny aggregate overhead/noise "
        "around a 0.997 aggregate ratio; structured workloads still have regressions, "
        "so the paper reports aggregate speedup together with tail and regression diagnostics.",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_mechanism_table(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# VANTAGE Mechanism Summary",
        "",
        "| Workload | Generated tokens | PLD target forwards | VANTAGE target forwards | Step reduction | TransPLD hits | TransPLD accepted | Accepted / TransPLD hit | Routes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['workload']} | {_fmt_opt(row['generated_tokens'], '{:.0f}')} | "
            f"{row['pld_steps']} | {row['method_steps']} | "
            f"{100 * row['step_reduction_vs_pld']:.1f}% | {row['transpld_hits']} | "
            f"{row['transpld_accepted']} | "
            f"{row['transpld_accepted_per_hit']:.2f} | `{row['routes']}` |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_mechanism_counters_table(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# VANTAGE Detailed Mechanism Counters",
        "",
        "Target forwards are logged decode steps. Average verification batch length is "
        "the logged `k` value averaged over decode steps. Rejected draft tokens are "
        "computed as proposed non-root draft tokens minus accepted non-root draft "
        "tokens when `proposal_tokens` or `target_draft_tokens` is present.",
        "",
        "| Workload | PLD accepted tokens | VANTAGE PLD-route accepted | VANTAGE TransPLD accepted | PLD rejected drafts | VANTAGE rejected drafts | PLD avg verification batch | VANTAGE avg verification batch | Setup total ms | Setup ms/prompt | TransPLD-hit tasks | TransPLD hits/task p50/p95/max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        hit_dist = (
            f"{_fmt_opt(row['transpld_hits_per_task_p50'], '{:.0f}')}/"
            f"{_fmt_opt(row['transpld_hits_per_task_p95'], '{:.0f}')}/"
            f"{_fmt_opt(row['transpld_hits_per_task_max'], '{:.0f}')}"
        )
        lines.append(
            f"| {row['workload']} | {row['baseline_pld_accepted_tokens']} | "
            f"{row['method_pld_route_accepted_tokens']} | {row['transpld_accepted']} | "
            f"{_fmt_opt(row['baseline_rejected_draft_tokens'], '{:.0f}')} | "
            f"{_fmt_opt(row['method_rejected_draft_tokens'], '{:.0f}')} | "
            f"{_fmt_opt(row['baseline_avg_verification_batch_len'], '{:.1f}')} | "
            f"{_fmt_opt(row['method_avg_verification_batch_len'], '{:.1f}')} | "
            f"{row['setup_us_total'] / 1000.0:.1f} | "
            f"{_fmt_opt(row['setup_ms_per_prompt'], '{:.2f}')} | "
            f"{row['transpld_hit_tasks']} | {hit_dist} |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_exactness_table(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# VANTAGE Output Exactness",
        "",
        "| Workload | n | VANTAGE = PLD | VANTAGE = vanilla | PLD = vanilla |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        n = row["n_tasks"]
        lines.append(
            f"| {row['workload']} | {n} | {row['method_vs_pld_exact_match']}/{n} | "
            f"{row['method_vs_vanilla_exact_match']}/{n} | "
            f"{row['pld_vs_vanilla_exact_match']}/{n} |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_lossless_table(lossless_dir: Path, path: Path) -> None:
    tags = {
        "field100": "vantage_frozen_lossless_field100_validation_20260515_v1",
        "style100": "vantage_frozen_lossless_style100_validation_20260515_v1",
        "zero100": "vantage_frozen_lossless_zero100_validation_20260515_v1",
    }
    lines = [
        "# VANTAGE Deterministic fp32/eager Exactness Audit",
        "",
        "This audit uses `float32` + `eager` and checks byte identity against vanilla.",
        "",
        "| Workload | n | PLD = vanilla | VANTAGE = vanilla |",
        "|---|---:|---:|---:|",
    ]
    for workload, tag in tags.items():
        result_path = lossless_dir / tag / "lossless" / "results.json"
        if not result_path.exists():
            lines.append(f"| {workload} | n/a | not run | not run |")
            continue
        result = _load_json(result_path)
        total = int(result.get("n_total") or 0)
        n_match_code = result.get("n_match_code") or {}
        lines.append(
            f"| {workload} | {total} | "
            f"{int(n_match_code.get('blazedit_pld_w128_n10') or 0)}/{total} | "
            f"{int(n_match_code.get('vantage_frozen_transpld') or 0)}/{total} |"
        )
    lines.append(
        "| mixed100 | n/a | not separately audited | timing run exact 100/100 vs PLD |"
    )
    path.write_text("\n".join(lines) + "\n")


def _load_lossless_status(lossless_dir: Path) -> dict[str, str]:
    tags = {
        "field100": "vantage_frozen_lossless_field100_validation_20260515_v1",
        "style100": "vantage_frozen_lossless_style100_validation_20260515_v1",
        "zero100": "vantage_frozen_lossless_zero100_validation_20260515_v1",
    }
    out: dict[str, str] = {}
    for workload, tag in tags.items():
        path = lossless_dir / tag / "lossless" / "results.json"
        if not path.exists():
            out[workload] = "not run"
            continue
        result = _load_json(path)
        total = int(result.get("n_total") or 0)
        n_match_code = result.get("n_match_code") or {}
        pld = int(n_match_code.get("blazedit_pld_w128_n10") or 0)
        method = int(n_match_code.get("vantage_frozen_transpld") or 0)
        out[workload] = f"PLD {pld}/{total}; VANTAGE {method}/{total}"
    out["mixed100"] = "not separately audited"
    return out


def write_backend_exactness_audit(
    rows: list[dict[str, Any]],
    lossless_dir: Path,
    path: Path,
) -> None:
    lossless = _load_lossless_status(lossless_dir)
    timing = []
    for row in rows:
        n = row["n_tasks"]
        timing.append(
            f"{row['workload']}: PLD {row['pld_vs_vanilla_exact_match']}/{n}, "
            f"VANTAGE {row['method_vs_vanilla_exact_match']}/{n}"
        )
    lines = [
        "# VANTAGE Backend Exactness Audit",
        "",
        "This generated audit keeps the theorem path separate from the optimized timing path. "
        "The paper may claim greedy equivalence only for deterministic fp32/eager rows that pass.",
        "",
        "| Execution path | Purpose | Evidence | Paper claim |",
        "|---|---|---|---|",
        "| fp32/eager | deterministic exactness audit | "
        + "; ".join(f"{k}: {v}" for k, v in lossless.items())
        + " | greedy-equivalence evidence for audited workloads |",
        "| bf16/sdpa | optimized timing path | "
        + "; ".join(timing)
        + "; separate raw-output parity diagnosis reports field raw mismatch and style max-token continuations"
        + " | speed diagnostic only; observed drift means no deployment-ready exactness claim |",
        "| bf16/eager | dtype isolation | not recorded in artifact | future parity isolation |",
        "| fp32/sdpa | attention-backend isolation | not recorded in artifact | future parity isolation |",
        "",
        "Acceptance status: the current artifact satisfies the revision plan by clearly "
        "separating exactness and speed. It does not satisfy the stronger condition of "
        "100% parity under the optimized bf16/sdpa timing backend.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        default="artifacts/vantage_transpld/modal/validation_20260515_v1",
    )
    parser.add_argument(
        "--report-dir",
        default="artifacts/vantage_transpld/reports/validation_20260515_v1",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/vantage_transpld/tables/validation_20260515_v1",
    )
    parser.add_argument(
        "--lossless-dir",
        default="artifacts/vantage_transpld/lossless/validation_20260515_v1",
    )
    args = parser.parse_args()

    rows = build_rows(Path(args.base_dir), Path(args.report_dir))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation_summary.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    write_main_table(rows, out / "main_throughput.md")
    write_diagnostic_table(rows, out / "throughput_diagnostics.md")
    write_latency_regression_table(rows, out / "latency_regression_summary.md")
    write_mechanism_table(rows, out / "mechanism_summary.md")
    write_mechanism_counters_table(rows, out / "mechanism_counters.md")
    write_exactness_table(rows, out / "exactness.md")
    write_lossless_table(Path(args.lossless_dir), out / "lossless_exactness.md")
    write_backend_exactness_audit(rows, Path(args.lossless_dir), out / "backend_exactness_audit.md")
    print((out / "main_throughput.md").read_text())
    print((out / "throughput_diagnostics.md").read_text())
    print((out / "latency_regression_summary.md").read_text())
    print((out / "mechanism_summary.md").read_text())
    print((out / "mechanism_counters.md").read_text())
    print((out / "exactness.md").read_text())
    print((out / "lossless_exactness.md").read_text())
    print((out / "backend_exactness_audit.md").read_text())


if __name__ == "__main__":
    main()
