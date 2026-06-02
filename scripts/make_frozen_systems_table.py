"""Assemble the frozen VANTAGE systems table from saved run artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = (len(ordered) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return float(ordered[lo] * (1 - frac) + ordered[hi] * frac)


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _setup_stats(steps: list[dict[str, Any]], method: str) -> dict[str, float]:
    by_task: dict[str, float] = {}
    for step in steps:
        if step.get("method") != method:
            continue
        task = str(step.get("task_id"))
        setup = (
            float(step.get("proposal_map_parse_us") or 0.0)
            + float(step.get("proposal_rewrite_apply_us") or 0.0)
            + float(step.get("proposal_virtual_reference_tokenize_us") or 0.0)
            + float(step.get("proposal_transpld_index_build_us") or 0.0)
        )
        by_task[task] = by_task.get(task, 0.0) + setup
    values_ms = [v / 1000.0 for v in by_task.values()]
    return {
        "setup_ms_mean": _mean(values_ms),
        "setup_ms_p50": _percentile(values_ms, 0.5),
        "setup_ms_p95": _percentile(values_ms, 0.95),
    }


def _route_stats(steps: list[dict[str, Any]], method: str) -> dict[str, Any]:
    routes: dict[str, int] = {}
    accepted: dict[str, int] = {}
    zero_trans = 0
    trans_hits = 0
    for step in steps:
        if step.get("method") != method:
            continue
        route = str(step.get("proposal_route") or step.get("proposal_match_kind") or "none")
        routes[route] = routes.get(route, 0) + 1
        n_acc = int(step.get("n_accepted_nonroot_drafts") or 0)
        accepted[route] = accepted.get(route, 0) + n_acc
        if route == "transpld":
            trans_hits += 1
            if n_acc <= 0:
                zero_trans += 1
    return {
        "routes": routes,
        "accepted_by_route": accepted,
        "transpld_zero_accept_rate": zero_trans / trans_hits if trans_hits else 0.0,
    }


def _latency_group(report: dict[str, Any], method: str) -> dict[str, Any]:
    for group in report.get("groups", []):
        if group.get("method") == method:
            return group
    raise KeyError(method)


def build_row(name: str, run_dir: Path, report_dir: Path, method: str, pld: str) -> dict[str, Any]:
    bootstrap = _load_json(report_dir / "bootstrap.json")
    latency = _load_json(report_dir / "latency.json")
    recovery = _load_json(report_dir / "recovery.json")
    steps = _load_jsonl(run_dir / "eval" / "steps.jsonl")
    method_group = _latency_group(latency, method)
    pld_group = _latency_group(latency, pld)
    pair = bootstrap["by_pair"][f"{method}_vs_{pld}"]
    rec_frac = recovery.get("overlap", {}).get("fractions", {}).get("after_unrooted_pld_miss", 0.0)
    anchor = recovery.get("anchor", {})
    row = {
        "workload": name,
        "pld_tps": bootstrap["by_method"][pld]["tokens_per_sec"],
        "method_tps": bootstrap["by_method"][method]["tokens_per_sec"],
        "ratio": pair["ratio"],
        "ci95": pair["ci95"],
        "tokens_per_verify": method_group["tokens_per_verify"]["mean"],
        "target_forward_reduction": method_group["target_forward_reduction"]["mean"],
        "proposal_us_per_token": method_group["proposal_us_per_token"]["mean"],
        "p50_wall_ms": method_group["wall_ms"]["p50"],
        "p95_wall_ms": method_group["wall_ms"]["p95"],
        "pld_p50_wall_ms": pld_group["wall_ms"]["p50"],
        "pld_p95_wall_ms": pld_group["wall_ms"]["p95"],
        "pld_miss_recovery": rec_frac,
        "transpld_zero_accept_rate": anchor.get("zero_accept_rate", 0.0),
    }
    row.update(_setup_stats(steps, method))
    row.update(_route_stats(steps, method))
    return row


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Frozen VANTAGE Mechanism and Latency Table",
        "",
        "| Workload | PLD tok/s | VANTAGE tok/s | VANTAGE/PLD | tokens/verify | target forward reduction | proposal us/token | setup ms p50/p95 | p50 wall ms | p95 wall ms | PLD-miss recovery | zero-accept rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lo, hi = row["ci95"]
        lines.append(
            f"| {row['workload']} | {row['pld_tps']:.2f} | {row['method_tps']:.2f} | "
            f"{row['ratio']:.3f} [{lo:.3f}, {hi:.3f}] | "
            f"{row['tokens_per_verify']:.2f} | {100 * row['target_forward_reduction']:.1f}% | "
            f"{row['proposal_us_per_token']:.2f} | "
            f"{row['setup_ms_p50']:.2f}/{row['setup_ms_p95']:.2f} | "
            f"{row['p50_wall_ms']:.2f} | {row['p95_wall_ms']:.2f} | "
            f"{100 * row['pld_miss_recovery']:.1f}% | "
            f"{100 * row['transpld_zero_accept_rate']:.1f}% |"
        )
    lines += [
        "",
        "Route counts and accepted tokens by route:",
        "",
    ]
    for row in rows:
        lines.append(f"- {row['workload']}: routes={row['routes']}; accepted={row['accepted_by_route']}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default="analysis/frozen_method_v1")
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    args = p.parse_args()
    base = Path(args.base_dir)
    configs = [
        ("zero", "vantage_frozen_final_qwen_base_zero100_v1"),
        ("field", "vantage_frozen_final_qwen_base_field100_v1"),
        ("style", "vantage_frozen_final_qwen_base_style100_v1"),
        ("mixed", "vantage_frozen_final_qwen_base_mixed100_v1"),
    ]
    rows = [
        build_row(
            name,
            base / "modal" / tag,
            base / "final_reports" / tag,
            "vantage_frozen_transpld",
            "blazedit_pld_w128_n10",
        )
        for name, tag in configs
    ]
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    write_markdown(rows, Path(args.output_md))


if __name__ == "__main__":
    main()
