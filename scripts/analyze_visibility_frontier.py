"""Analyze VANTAGE visibility zones and uncertainty frontiers.

This script is CPU-only and consumes the per-step JSONL emitted by
``scripts/run_eagle_eval.py``.  It adds the analyses needed by the paper's
VANTAGE framing:

  - dark/lit/mid zone acceptance statistics,
  - per-node useful frontier depth,
  - router strategy mix and per-strategy acceptance.

Example:

    python scripts/analyze_visibility_frontier.py \\
        --steps /tmp/vantage_eval/steps.jsonl \\
        --method vantage_full \\
        --output-md /tmp/vantage_frontier.md \\
        --output-json /tmp/vantage_frontier.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asts.vantage_policy import zone_for_node


@dataclass
class GroupStats:
    n: int = 0
    sum_k: int = 0
    sum_accepted: int = 0
    sum_emitted: int = 0
    sum_wall_us: float = 0.0

    def add(self, row: dict[str, Any]) -> None:
        self.n += 1
        self.sum_k += int(row.get("k", 0))
        self.sum_accepted += int(row.get("n_accepted_drafts", 0))
        self.sum_emitted += int(row.get("n_emitted", 0))
        self.sum_wall_us += float(row.get("wall_us", 0.0))

    def summary(self) -> dict[str, float | int]:
        mean_k = self.sum_k / self.n if self.n else 0.0
        mean_accepted = self.sum_accepted / self.n if self.n else 0.0
        return {
            "n": self.n,
            "mean_k": mean_k,
            "mean_accepted": mean_accepted,
            "acceptance_rate": mean_accepted / mean_k if mean_k else 0.0,
            "tokens_per_sec": self.sum_emitted / (self.sum_wall_us / 1e6)
            if self.sum_wall_us > 0
            else 0.0,
        }


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def frontier_depth(
    rows: list[dict[str, Any]],
    *,
    threshold: float = 0.50,
    max_depth: int = 8,
    min_support: int = 25,
) -> tuple[int, dict[int, float]]:
    """Largest depth whose survival probability clears ``threshold``.

    Survival at depth d is P(accepted at least d drafts | requested at least d).
    For EAGLE-1 methods the first candidate is often the target argmax, so the
    informative frontier is usually whether depths 2+ survive.
    """
    survival: dict[int, float] = {}
    frontier = 0
    for d in range(1, max_depth + 1):
        eligible = [r for r in rows if int(r.get("k", 0)) >= d]
        if len(eligible) < min_support:
            continue
        survived = sum(1 for r in eligible if int(r.get("n_accepted_drafts", 0)) >= d)
        prob = survived / len(eligible)
        survival[d] = prob
        if prob >= threshold:
            frontier = d
    return frontier, survival


def analyze(
    rows: list[dict[str, Any]],
    *,
    method: str,
    threshold: float,
    top_n: int,
    min_support: int,
) -> dict[str, Any]:
    method_rows = [r for r in rows if r.get("method") == method]
    if not method_rows:
        raise ValueError(f"no rows found for method={method!r}")

    by_zone: dict[str, GroupStats] = defaultdict(GroupStats)
    by_node_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_deepest_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_error: dict[str, GroupStats] = defaultdict(GroupStats)
    by_ancestor_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_strategy: dict[str, GroupStats] = defaultdict(GroupStats)

    for row in method_rows:
        node = row.get("node_type") or "default"
        deepest = row.get("deepest_type")
        zone = row.get("zone") or zone_for_node(node, deepest)
        by_zone[zone].add(row)
        by_node_rows[node].append(row)
        by_deepest_rows[str(deepest or "default")].append(row)
        error_key = "parser_error" if row.get("parser_in_error") else "clean_or_unknown"
        by_error[error_key].add(row)
        ancestors = row.get("ancestor_types") or []
        if isinstance(ancestors, list) and ancestors:
            by_ancestor_rows[" > ".join(str(x) for x in ancestors[:4])].append(row)
        if row.get("strategy"):
            by_strategy[str(row["strategy"])].add(row)

    zone_summary = {zone: stats.summary() for zone, stats in sorted(by_zone.items())}
    strategy_summary = {
        strategy: stats.summary()
        for strategy, stats in sorted(by_strategy.items(), key=lambda item: -item[1].n)
    }

    node_frontiers = []
    for node, node_rows in sorted(by_node_rows.items(), key=lambda item: -len(item[1]))[:top_n]:
        stats = GroupStats()
        for row in node_rows:
            stats.add(row)
        frontier, survival = frontier_depth(
            node_rows,
            threshold=threshold,
            min_support=min_support,
        )
        item = stats.summary()
        item.update(
            {
                "node_type": node,
                "zone": zone_for_node(node, node_rows[0].get("deepest_type")),
                "frontier_depth": frontier,
                "survival": survival,
            }
        )
        node_frontiers.append(item)

    deepest_frontiers = []
    for deepest, deepest_rows in sorted(by_deepest_rows.items(), key=lambda item: -len(item[1]))[:top_n]:
        stats = GroupStats()
        for row in deepest_rows:
            stats.add(row)
        frontier, survival = frontier_depth(
            deepest_rows,
            threshold=threshold,
            min_support=min_support,
        )
        item = stats.summary()
        item.update(
            {
                "deepest_type": deepest,
                "zone": zone_for_node(None, deepest),
                "frontier_depth": frontier,
                "survival": survival,
            }
        )
        deepest_frontiers.append(item)

    ancestor_frontiers = []
    for path, path_rows in sorted(by_ancestor_rows.items(), key=lambda item: -len(item[1]))[:top_n]:
        stats = GroupStats()
        for row in path_rows:
            stats.add(row)
        frontier, survival = frontier_depth(
            path_rows,
            threshold=threshold,
            min_support=min_support,
        )
        item = stats.summary()
        item.update(
            {
                "ancestor_path": path,
                "frontier_depth": frontier,
                "survival": survival,
            }
        )
        ancestor_frontiers.append(item)

    overall_frontier, overall_survival = frontier_depth(
        method_rows,
        threshold=threshold,
        min_support=min_support,
    )
    return {
        "method": method,
        "n_steps": len(method_rows),
        "frontier_threshold": threshold,
        "overall_frontier_depth": overall_frontier,
        "overall_survival": overall_survival,
        "zones": zone_summary,
        "parser_error": {key: stats.summary() for key, stats in sorted(by_error.items())},
        "strategies": strategy_summary,
        "node_frontiers": node_frontiers,
        "deepest_frontiers": deepest_frontiers,
        "ancestor_frontiers": ancestor_frontiers,
    }


def _fmt_float(x: Any, digits: int = 3) -> str:
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# VANTAGE Visibility/Frontier Analysis: `{report['method']}`")
    lines.append("")
    lines.append(
        f"Steps: {report['n_steps']:,}. Frontier threshold: "
        f"{report['frontier_threshold']:.2f}. Overall frontier depth: "
        f"{report['overall_frontier_depth']}."
    )
    lines.append("")

    lines.append("## Zone Breakdown")
    lines.append("")
    lines.append("| Zone | Steps | Mean k | Mean accepted | Accept rate | tok/s |")
    lines.append("|------|------:|-------:|--------------:|------------:|------:|")
    for zone, stats in report["zones"].items():
        lines.append(
            f"| {zone} | {stats['n']} | {stats['mean_k']:.2f} | "
            f"{stats['mean_accepted']:.2f} | {stats['acceptance_rate']:.3f} | "
            f"{stats['tokens_per_sec']:.2f} |"
        )
    lines.append("")

    if report["strategies"]:
        lines.append("## Strategy Mix")
        lines.append("")
        lines.append("| Strategy | Steps | Mean k | Mean accepted | Accept rate | tok/s |")
        lines.append("|----------|------:|-------:|--------------:|------------:|------:|")
        for strategy, stats in report["strategies"].items():
            lines.append(
                f"| `{strategy}` | {stats['n']} | {stats['mean_k']:.2f} | "
                f"{stats['mean_accepted']:.2f} | {stats['acceptance_rate']:.3f} | "
                f"{stats['tokens_per_sec']:.2f} |"
            )
        lines.append("")

    if report.get("parser_error"):
        lines.append("## Parser Error Breakdown")
        lines.append("")
        lines.append("| State | Steps | Mean k | Mean accepted | Accept rate | tok/s |")
        lines.append("|-------|------:|-------:|--------------:|------------:|------:|")
        for state, stats in report["parser_error"].items():
            lines.append(
                f"| {state} | {stats['n']} | {stats['mean_k']:.2f} | "
                f"{stats['mean_accepted']:.2f} | {stats['acceptance_rate']:.3f} | "
                f"{stats['tokens_per_sec']:.2f} |"
            )
        lines.append("")

    lines.append("## Node Frontier")
    lines.append("")
    lines.append(
        "| Node | Zone | Steps | Mean k | Mean accepted | Frontier | P(>=1) | P(>=2) | P(>=3) |"
    )
    lines.append(
        "|------|------|------:|-------:|--------------:|---------:|-------:|-------:|-------:|"
    )
    for item in report["node_frontiers"]:
        surv = item["survival"]
        lines.append(
            f"| `{item['node_type']}` | {item['zone']} | {item['n']} | "
            f"{item['mean_k']:.2f} | {item['mean_accepted']:.2f} | "
            f"{item['frontier_depth']} | {_fmt_float(surv.get(1, 'n/a'))} | "
            f"{_fmt_float(surv.get(2, 'n/a'))} | {_fmt_float(surv.get(3, 'n/a'))} |"
        )
    lines.append("")

    if report.get("deepest_frontiers"):
        lines.append("## Deepest-Type Frontier")
        lines.append("")
        lines.append(
            "| Deepest type | Zone | Steps | Mean k | Mean accepted | Frontier | P(>=1) | P(>=2) | P(>=3) |"
        )
        lines.append(
            "|--------------|------|------:|-------:|--------------:|---------:|-------:|-------:|-------:|"
        )
        for item in report["deepest_frontiers"]:
            surv = item["survival"]
            lines.append(
                f"| `{item['deepest_type']}` | {item['zone']} | {item['n']} | "
                f"{item['mean_k']:.2f} | {item['mean_accepted']:.2f} | "
                f"{item['frontier_depth']} | {_fmt_float(surv.get(1, 'n/a'))} | "
                f"{_fmt_float(surv.get(2, 'n/a'))} | {_fmt_float(surv.get(3, 'n/a'))} |"
            )
        lines.append("")

    if report.get("ancestor_frontiers"):
        lines.append("## Ancestor-Path Frontier")
        lines.append("")
        lines.append(
            "| Ancestor path | Steps | Mean k | Mean accepted | Frontier | P(>=1) | P(>=2) | P(>=3) |"
        )
        lines.append(
            "|---------------|------:|-------:|--------------:|---------:|-------:|-------:|-------:|"
        )
        for item in report["ancestor_frontiers"]:
            surv = item["survival"]
            lines.append(
                f"| `{item['ancestor_path']}` | {item['n']} | "
                f"{item['mean_k']:.2f} | {item['mean_accepted']:.2f} | "
                f"{item['frontier_depth']} | {_fmt_float(surv.get(1, 'n/a'))} | "
                f"{_fmt_float(surv.get(2, 'n/a'))} | {_fmt_float(surv.get(3, 'n/a'))} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", required=True)
    parser.add_argument("--method", default="vantage_full")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-support", type=int, default=25)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    rows = load_jsonl(args.steps)
    report = analyze(
        rows,
        method=args.method,
        threshold=args.threshold,
        top_n=args.top_n,
        min_support=args.min_support,
    )
    md = to_markdown(report)
    print(md)

    if args.output_md:
        Path(args.output_md).write_text(md)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
