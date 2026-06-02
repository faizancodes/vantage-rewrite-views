#!/usr/bin/env python3
"""Summarize VANTAGE ViewBank real-commit timing diagnostics.

The input root is a Modal result directory containing:
  - bootstrap.json
  - mv_grid.json
  - eval/completions.jsonl
  - eval/steps.jsonl

The summary is intentionally conservative: it reports speed and proposal
counters, plus raw-output agreement with tuned PLD. It does not convert a
bf16 timing diagnostic into an exactness claim.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "analysis/real_commits/modal/vantage_real_commit_balanced1000_mv_1000_viewbased_20260520_v2"
)
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/vantage_viewbank/tables/real_commit_mv_20260520_v2"
)
BASELINE = "blazedit_pld_w128_n10"
METHODS = [
    ("blazedit_pld_w128_n10", "PLD", "identity"),
    ("vantage_force_pld_w128_n10", "Force PLD", "identity wrapper"),
    ("vantage_frozen_transpld", "Frozen TransPLD", "rewrite view only"),
    (
        "vantage_mv_pld_s96_x1_m16_t8_w128_n10",
        "Stable MV",
        "identity+rewrite/frontier",
    ),
    (
        "vantage_mv_pld_patch_s96_x1_m16_t8_w128_n10",
        "Patch MV",
        "identity+patch/frontier",
    ),
    (
        "vantage_mv_pld_rescue_s96_x1_m16_t8_w128_n10",
        "Rescue MV",
        "identity+rescue/frontier",
    ),
    (
        "vantage_mv_pld_cursor_s96_x1_m16_t8_w128_n10",
        "Cursor MV",
        "identity+cursor/frontier",
    ),
    (
        "vantage_mv_pld_hunk_s96_x1_m16_t8_w128_n10",
        "Hunk MV",
        "identity+hunk/frontier",
    ),
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def _output_tps(output: dict[str, Any]) -> float:
    wall_us = float(output.get("wall_us") or 0.0)
    toks = int(output.get("n_new_tokens") or 0)
    return toks / (wall_us / 1e6) if wall_us > 0 else 0.0


def _norm(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines()).strip()


_FENCE_RE = re.compile(
    r"^\s*```(?:[A-Za-z0-9_+-]+)?\s*\n(?P<body>.*?)(?:\n```\s*)?$",
    re.DOTALL,
)


def _extract_code(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        body_lines = lines[1:]
        for idx, line in enumerate(body_lines):
            if line.strip().startswith("```"):
                return "\n".join(body_lines[:idx]).strip()
        return "\n".join(body_lines).strip()
    match = _FENCE_RE.match(stripped)
    if match:
        return match.group("body").strip()
    return stripped


def _code_norm(text: str) -> str:
    return _norm(_extract_code(text))


def _python_syntax_ok(text: str) -> bool:
    code = _extract_code(text)
    if not code.strip():
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _is_hidden_view_step(method: str, row: dict[str, Any]) -> bool:
    if method == "vantage_frozen_transpld":
        return (
            row.get("proposal_source_region") == "precomputed_transpld"
            or row.get("proposal_pool") == "virtual_reference"
        )
    if method.startswith("vantage_mv"):
        return row.get("proposal_kind") == "vantage_mv_pld"
    return False


def summarize(root: Path) -> dict[str, Any]:
    bootstrap = _load_json(root / "bootstrap.json")
    grid = _load_json(root / "mv_grid.json")
    grid_rows = {row["method"]: row for row in grid.get("ranked_methods", [])}
    by_method = bootstrap["by_method"]

    hidden_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "hidden_steps": 0,
            "hidden_accepted": 0,
            "route_counts": Counter(),
            "kind_counts": Counter(),
            "view_counts": Counter(),
            "zero_accept_hidden_steps": 0,
        }
    )
    with (root / "eval" / "steps.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            method = row["method"]
            if method not in by_method:
                continue
            route = row.get("proposal_route")
            if route:
                hidden_stats[method]["route_counts"][str(route)] += 1
            kind = row.get("proposal_kind")
            if kind:
                hidden_stats[method]["kind_counts"][str(kind)] += 1
            view = row.get("proposal_view_id")
            if view:
                hidden_stats[method]["view_counts"][str(view)] += 1
            if _is_hidden_view_step(method, row):
                hidden_stats[method]["hidden_steps"] += 1
                accepted = int(row.get("n_accepted_nonroot_drafts") or 0)
                hidden_stats[method]["hidden_accepted"] += accepted
                if accepted == 0:
                    hidden_stats[method]["zero_accept_hidden_steps"] += 1

    pld_raw_match: dict[str, int] = defaultdict(int)
    quality_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "target_rows": 0,
            "exact_target": 0,
            "syntax_rows": 0,
            "syntax_ok": 0,
        }
    )
    latency_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"ratios": [], "latency_ms": []}
    )
    n_tasks = 0
    with (root / "eval" / "completions.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            n_tasks += 1
            outputs = row["outputs"]
            baseline_text = outputs[BASELINE]["raw_text"]
            baseline_tps = _output_tps(outputs[BASELINE])
            target = str(row.get("deterministic_target") or "")
            for method in outputs:
                if outputs[method]["raw_text"] == baseline_text:
                    pld_raw_match[method] += 1
                if method in by_method:
                    output = outputs[method]
                    text = str(output.get("raw_text") or output.get("text") or "")
                    if target.strip():
                        quality_counts[method]["target_rows"] += 1
                        if _code_norm(text) == _code_norm(target):
                            quality_counts[method]["exact_target"] += 1
                    quality_counts[method]["syntax_rows"] += 1
                    if _python_syntax_ok(text):
                        quality_counts[method]["syntax_ok"] += 1
                    method_tps = _output_tps(output)
                    if baseline_tps > 0 and method_tps > 0:
                        latency_values[method]["ratios"].append(method_tps / baseline_tps)
                    latency_values[method]["latency_ms"].append(
                        float(output.get("wall_us") or 0.0) / 1000.0
                    )

    rows: list[dict[str, Any]] = []
    for method, label, views in METHODS:
        b = by_method[method]
        g = grid_rows.get(method, {})
        ci = b["ci95"]
        rows.append(
            {
                "method": method,
                "label": label,
                "views": views,
                "n_tasks": n_tasks,
                "tokens_per_sec": b["tokens_per_sec"],
                "speedup_vs_pld": b[f"speedup_vs_{BASELINE}"],
                "ci95": ci,
                "hidden_steps": hidden_stats[method]["hidden_steps"],
                "hidden_accepted": hidden_stats[method]["hidden_accepted"],
                "setup_s": g.get("setup_s", 0.0),
                "pld_raw_match": pld_raw_match[method],
            }
        )
    route_breakdown = []
    latency_tail = []
    parity_quality = []
    for method, label, _views in METHODS:
        stats = hidden_stats[method]
        ratios = latency_values[method]["ratios"]
        latencies = latency_values[method]["latency_ms"]
        q = quality_counts[method]
        route_breakdown.append(
            {
                "method": method,
                "label": label,
                "hidden_steps": stats["hidden_steps"],
                "hidden_accepted": stats["hidden_accepted"],
                "zero_accept_hidden_steps": stats["zero_accept_hidden_steps"],
                "route_counts": dict(stats["route_counts"].most_common()),
                "proposal_kind_counts": dict(stats["kind_counts"].most_common()),
                "top_view_ids": dict(stats["view_counts"].most_common(10)),
            }
        )
        latency_tail.append(
            {
                "method": method,
                "label": label,
                "n_task_ratios": len(ratios),
                "regressions_vs_pld": sum(1 for r in ratios if r < 1.0),
                "worst_task_ratio": min(ratios) if ratios else 0.0,
                "p05_task_ratio": _percentile(ratios, 0.05),
                "p50_task_ratio": _percentile(ratios, 0.50),
                "p95_task_ratio": _percentile(ratios, 0.95),
                "p99_latency_ms": _percentile(latencies, 0.99),
                "max_latency_ms": max(latencies) if latencies else 0.0,
            }
        )
        parity_quality.append(
            {
                "method": method,
                "label": label,
                "pld_raw_match": pld_raw_match[method],
                "n_tasks": n_tasks,
                "target_rows": q["target_rows"],
                "exact_target": q["exact_target"],
                "syntax_rows": q["syntax_rows"],
                "syntax_ok": q["syntax_ok"],
            }
        )
    return {
        "schema": "vantage/viewbank_real_commit_summary/v1",
        "source_root": str(root),
        "baseline": BASELINE,
        "n_tasks": n_tasks,
        "rows": rows,
        "route_proposal_breakdown": route_breakdown,
        "latency_tail": latency_tail,
        "parity_quality": parity_quality,
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# VANTAGE ViewBank Real-Commit Summary",
        "",
        f"Source root: `{summary['source_root']}`",
        f"Baseline: `{summary['baseline']}`",
        f"Tasks: {summary['n_tasks']}",
        "",
        "| Method | Views | Tasks | tok/s | vs PLD | 95% CI | Hidden steps | Hidden accepted | Setup s | PLD raw match |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        ci = row["ci95"]
        lines.append(
            "| {label} | {views} | {n_tasks} | {tps:.1f} | {ratio:.3f} | [{lo:.3f},{hi:.3f}] | "
            "{hidden_steps} | {hidden_accepted} | {setup:.1f} | {match}/{n_tasks} |".format(
                label=row["label"],
                views=row["views"],
                tps=row["tokens_per_sec"],
                ratio=row["speedup_vs_pld"],
                lo=ci[0],
                hi=ci[1],
                hidden_steps=row["hidden_steps"],
                hidden_accepted=row["hidden_accepted"],
                setup=row["setup_s"],
                match=row["pld_raw_match"],
                n_tasks=row["n_tasks"],
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: this is a bf16/sdpa timing diagnostic. The PLD raw-match column reports raw-output agreement with tuned PLD; it is not a vanilla exactness audit.",
            "",
        ]
    )
    return "\n".join(lines)


def route_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# VANTAGE ViewBank Route/Proposal Breakdown",
        "",
        f"Source root: `{summary['source_root']}`",
        "",
        "| Method | Hidden steps | Hidden accepted | Zero-accept hidden steps | Top routes | Top proposal kinds |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in summary["route_proposal_breakdown"]:
        routes = ", ".join(f"{k}:{v}" for k, v in list(row["route_counts"].items())[:4]) or "-"
        kinds = ", ".join(f"{k}:{v}" for k, v in list(row["proposal_kind_counts"].items())[:4]) or "-"
        lines.append(
            f"| {row['label']} | {row['hidden_steps']} | {row['hidden_accepted']} | "
            f"{row['zero_accept_hidden_steps']} | `{routes}` | `{kinds}` |"
        )
    lines.append("")
    return "\n".join(lines)


def latency_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# VANTAGE ViewBank Latency/Tail Summary",
        "",
        f"Source root: `{summary['source_root']}`",
        "",
        "| Method | Task ratios | Regressions vs PLD | Worst ratio | p05 ratio | p50 ratio | p95 ratio | p99 latency ms | Max latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["latency_tail"]:
        lines.append(
            f"| {row['label']} | {row['n_task_ratios']} | {row['regressions_vs_pld']} | "
            f"{row['worst_task_ratio']:.3f} | {row['p05_task_ratio']:.3f} | "
            f"{row['p50_task_ratio']:.3f} | {row['p95_task_ratio']:.3f} | "
            f"{row['p99_latency_ms']:.1f} | {row['max_latency_ms']:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parity_quality_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# VANTAGE ViewBank Parity/Quality Summary",
        "",
        f"Source root: `{summary['source_root']}`",
        "",
        "This is a bf16/sdpa timing diagnostic. PLD raw match is agreement with tuned PLD, not a vanilla greedy exactness audit.",
        "",
        "| Method | PLD raw match | Exact target | Python syntax OK |",
        "|---|---:|---:|---:|",
    ]
    for row in summary["parity_quality"]:
        lines.append(
            f"| {row['label']} | {row['pld_raw_match']}/{row['n_tasks']} | "
            f"{row['exact_target']}/{row['target_rows']} | "
            f"{row['syntax_ok']}/{row['syntax_rows']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    summary = summarize(args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "viewbank_real_commit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (args.output_dir / "viewbank_real_commit_summary.md").write_text(
        markdown(summary)
    )
    (args.output_dir / "viewbank_route_proposal_breakdown.json").write_text(
        json.dumps(summary["route_proposal_breakdown"], indent=2) + "\n"
    )
    (args.output_dir / "viewbank_route_proposal_breakdown.md").write_text(
        route_markdown(summary)
    )
    (args.output_dir / "viewbank_latency_tail.json").write_text(
        json.dumps(summary["latency_tail"], indent=2) + "\n"
    )
    (args.output_dir / "viewbank_latency_tail.md").write_text(
        latency_markdown(summary)
    )
    (args.output_dir / "viewbank_parity_quality.json").write_text(
        json.dumps(summary["parity_quality"], indent=2) + "\n"
    )
    (args.output_dir / "viewbank_parity_quality.md").write_text(
        parity_quality_markdown(summary)
    )


if __name__ == "__main__":
    main()
