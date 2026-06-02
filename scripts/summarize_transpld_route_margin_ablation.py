#!/usr/bin/env python3
"""Summarize TransPLD prompt-only route-margin ablation runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("artifacts/vantage_transpld/modal/route_margin_20260516_v1")
DEFAULT_OUTPUT_DIR = Path("artifacts/vantage_transpld/tables/route_margin_20260516_v1")

PLD = "blazedit_pld_w128_n10"
METHODS = [
    ("vantage_compete_transpld_m4_margin0_w128_n10", "margin 0"),
    ("vantage_compete_transpld_m4_margin16_w128_n10", "margin 16"),
    ("vantage_compete_transpld_m4_margin32_w128_n10", "margin 32"),
    ("vantage_frozen_transpld", "frozen SafeRoute"),
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _workload(run_tag: str) -> str:
    lowered = run_tag.lower()
    if "field" in lowered:
        return "Field substitution"
    if "style" in lowered:
        return "Identifier-style substitution"
    if "mixed" in lowered:
        return "Mixed"
    if "zero" in lowered:
        return "Zero drift"
    return run_tag


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _fmt(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _task_speedup(row: dict[str, Any], method: str) -> float | None:
    outputs = row.get("outputs") or {}
    pld = outputs.get(PLD) or {}
    cand = outputs.get(method) or {}
    if not pld or not cand:
        return None
    pld_wall = float(pld.get("wall_us") or 0.0)
    cand_wall = float(cand.get("wall_us") or 0.0)
    pld_tokens = float(pld.get("n_new_tokens") or 0.0)
    cand_tokens = float(cand.get("n_new_tokens") or 0.0)
    if pld_wall <= 0.0 or cand_wall <= 0.0 or pld_tokens <= 0.0 or cand_tokens <= 0.0:
        return None
    return (cand_tokens / cand_wall) / (pld_tokens / pld_wall)


def _task_latency_ms(row: dict[str, Any], method: str) -> float | None:
    outputs = row.get("outputs") or {}
    cand = outputs.get(method) or {}
    if not cand:
        return None
    cand_wall = float(cand.get("wall_us") or 0.0)
    if cand_wall <= 0.0:
        return None
    return cand_wall / 1000.0


def _route_counts(steps: list[dict[str, Any]], method: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in steps:
        if row.get("method") != method:
            continue
        route = (
            row.get("proposal_route")
            or row.get("proposal_kind")
            or row.get("strategy")
            or "none"
        )
        counts[str(route)] += 1
    return counts


def _tasks_with_transpld(steps: list[dict[str, Any]], method: str) -> set[str]:
    out: set[str] = set()
    for row in steps:
        if row.get("method") != method:
            continue
        if row.get("proposal_route") == "transpld":
            out.add(str(row.get("task_id")))
    return out


def summarize_run(path: Path) -> dict[str, Any]:
    aggregate = _load_json(path)
    completions = _load_jsonl(path.parent / "completions.jsonl")
    steps = _load_jsonl(path.parent / "steps.jsonl")
    by_method = aggregate.get("by_method") or {}
    pld_tok_s = float((by_method.get(PLD) or {}).get("tokens_per_sec") or 0.0)
    rows: list[dict[str, Any]] = []
    for method, label in METHODS:
        data = by_method.get(method) or {}
        tok_s = data.get("tokens_per_sec")
        task_ratios_by_id: dict[str, float] = {}
        for row in completions:
            ratio = _task_speedup(row, method)
            if ratio is not None:
                task_ratios_by_id[str(row.get("task_id"))] = ratio
        latencies_ms = [
            latency
            for row in completions
            if (latency := _task_latency_ms(row, method)) is not None
        ]
        ratios = list(task_ratios_by_id.values())
        trans_tasks = _tasks_with_transpld(steps, method)
        trans_regressions = sum(
            1 for task_id, ratio in task_ratios_by_id.items() if task_id in trans_tasks and ratio < 1.0
        )
        rows.append(
            {
                "method": method,
                "label": label,
                "tokens_per_sec": tok_s,
                "ratio_vs_pld": float(tok_s) / pld_tok_s if tok_s and pld_tok_s else None,
                "per_task_p05": _quantile(ratios, 0.05),
                "per_task_p50": _quantile(ratios, 0.50),
                "per_task_p95": _quantile(ratios, 0.95),
                "per_task_p99": _quantile(ratios, 0.99),
                "latency_ms_p95": _quantile(latencies_ms, 0.95),
                "latency_ms_p99": _quantile(latencies_ms, 0.99),
                "worst_per_task": min(ratios) if ratios else None,
                "regression_count": sum(1 for r in ratios if r < 1.0),
                "transpld_route_regression_count": trans_regressions,
                "transpld_route_task_count": len(trans_tasks),
                "route_counts": dict(_route_counts(steps, method)),
                "n_steps": data.get("n_steps"),
                "parity_matches": (aggregate.get("output_equivalence") or {}).get(method, {}).get("matches_vanilla"),
                "parity_tasks": (aggregate.get("output_equivalence") or {}).get(method, {}).get("tasks"),
            }
        )
    return {
        "run_tag": path.parent.parent.name,
        "workload": _workload(path.parent.parent.name),
        "aggregate_path": str(path),
        "pld_tok_s": pld_tok_s,
        "rows": rows,
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# TransPLD Route-Margin Ablation",
        "",
        "This generated table evaluates prompt-only route policies that require the transformed candidate to exceed the PLD candidate by a fixed token margin. The `disable after early reject` policy is not included here because it requires post-verification feedback and is therefore a different adaptive runtime policy, not a purely prompt/candidate-only route rule.",
        "",
        "| Workload | Policy | PLD tok/s | Policy tok/s | Policy/PLD | Speedup p05 | Speedup p50 | Speedup p95 | Speedup p99 | Latency p95 ms | Latency p99 ms | Worst speedup | Regressions | rewrite-view-route regressions | rewrite-view-route tasks | Route counts | Parity |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for run in summary["runs"]:
        for row in run["rows"]:
            parity = "unavailable"
            if row.get("parity_matches") is not None and row.get("parity_tasks") is not None:
                parity = f"{row['parity_matches']}/{row['parity_tasks']}"
            lines.append(
                "| {workload} | {policy} | {pld} | {tok_s} | {ratio} | {p05} | {p50} | {p95} | {p99} | {lat_p95} | {lat_p99} | {worst} | {reg} | {trans_reg} | {trans_tasks} | `{routes}` | {parity} |".format(
                    workload=run["workload"],
                    policy=row["label"],
                    pld=_fmt(run["pld_tok_s"]),
                    tok_s=_fmt(row["tokens_per_sec"]),
                    ratio=_fmt(row["ratio_vs_pld"]),
                    p05=_fmt(row["per_task_p05"]),
                    p50=_fmt(row["per_task_p50"]),
                    p95=_fmt(row["per_task_p95"]),
                    p99=_fmt(row["per_task_p99"]),
                    lat_p95=_fmt(row["latency_ms_p95"]),
                    lat_p99=_fmt(row["latency_ms_p99"]),
                    worst=_fmt(row["worst_per_task"]),
                    reg=row["regression_count"],
                    trans_reg=row["transpld_route_regression_count"],
                    trans_tasks=row["transpld_route_task_count"],
                    routes=row["route_counts"],
                    parity=parity,
                )
            )
    lines += [
        "",
        "Interpretation rule: use this table to decide whether a stricter prompt-only margin materially reduces tail risk without erasing aggregate speedup. If no margin policy improves the tail/aggregate tradeoff, keep frozen SafeRoute and state the tail risk explicitly.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    root = Path(args.root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [summarize_run(path) for path in sorted(root.glob("*/eval/aggregate.json"))]
    runs.sort(key=lambda r: r["workload"])
    summary = {
        "schema": "vantage/transpld_route_margin_ablation/v1",
        "root": str(root),
        "runs": runs,
    }
    (out_dir / "route_margin_ablation.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "route_margin_ablation.md").write_text(markdown(summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
