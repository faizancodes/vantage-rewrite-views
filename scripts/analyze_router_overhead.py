"""Break down router overhead from run_eagle_eval.py step traces."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


COMPONENTS = (
    "parse_us",
    "target_prefill_us",
    "confidence_us",
    "retrieval_us",
    "scope_us",
    "route_us",
    "draft_us",
    "verify_us",
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class Stats:
    def __init__(self) -> None:
        self.n = 0
        self.emitted = 0
        self.wall_us = 0.0
        self.components = {k: 0.0 for k in COMPONENTS}
        self.k = 0
        self.accepted = 0

    def add(self, row: dict[str, Any]) -> None:
        self.n += 1
        self.emitted += int(row.get("n_emitted", 0))
        self.wall_us += float(row.get("wall_us", 0.0))
        self.k += int(row.get("k", 0))
        self.accepted += int(row.get("n_accepted_drafts", 0))
        for key in COMPONENTS:
            self.components[key] += float(row.get(key, 0.0) or 0.0)

    def summary(self) -> dict[str, Any]:
        out = {
            "n": self.n,
            "n_emitted": self.emitted,
            "tokens_per_sec": self.emitted / (self.wall_us / 1e6) if self.wall_us else 0.0,
            "wall_us_total": self.wall_us,
            "mean_wall_us": self.wall_us / self.n if self.n else 0.0,
            "mean_k": self.k / self.n if self.n else 0.0,
            "mean_accepted": self.accepted / self.n if self.n else 0.0,
            "us_per_token": self.wall_us / self.emitted if self.emitted else 0.0,
            "components_total": self.components,
            "components_mean": {
                k: v / self.n if self.n else 0.0 for k, v in self.components.items()
            },
            "components_share_of_wall": {
                k: v / self.wall_us if self.wall_us else 0.0 for k, v in self.components.items()
            },
        }
        return out


def analyze(rows: list[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    by_method: dict[str, Stats] = defaultdict(Stats)
    by_strategy: dict[str, Stats] = defaultdict(Stats)
    by_zone: dict[str, Stats] = defaultdict(Stats)
    selected = [r for r in rows if not methods or r.get("method") in methods]
    for row in selected:
        method = str(row.get("method"))
        by_method[method].add(row)
        if row.get("strategy"):
            by_strategy[f"{method}:{row['strategy']}"].add(row)
        if row.get("zone"):
            by_zone[f"{method}:{row['zone']}"].add(row)
    return {
        "n_steps": len(selected),
        "methods": {k: v.summary() for k, v in sorted(by_method.items())},
        "strategies": {k: v.summary() for k, v in sorted(by_strategy.items())},
        "zones": {k: v.summary() for k, v in sorted(by_zone.items())},
    }


def _component_text(item: dict[str, Any]) -> str:
    means = item["components_mean"]
    keys = ["parse_us", "target_prefill_us", "confidence_us", "retrieval_us", "scope_us", "route_us", "draft_us", "verify_us"]
    return ", ".join(f"{k}={means.get(k, 0.0):.1f}" for k in keys)


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Router Overhead Breakdown",
        "",
        f"Steps: {report['n_steps']:,}.",
        "",
        "## By Method",
        "",
        "| Method | Steps | tok/s | us/token | mean wall us | mean k | mean accepted | component means |",
        "|--------|------:|------:|---------:|-------------:|-------:|--------------:|-----------------|",
    ]
    for method, item in report["methods"].items():
        lines.append(
            f"| `{method}` | {item['n']} | {item['tokens_per_sec']:.2f} | "
            f"{item['us_per_token']:.1f} | {item['mean_wall_us']:.1f} | "
            f"{item['mean_k']:.2f} | {item['mean_accepted']:.2f} | "
            f"{_component_text(item)} |"
        )
    lines += ["", "## By Strategy", "", "| Method:strategy | Steps | tok/s | mean wall us | component means |", "|-----------------|------:|------:|-------------:|-----------------|"]
    for strategy, item in report["strategies"].items():
        lines.append(
            f"| `{strategy}` | {item['n']} | {item['tokens_per_sec']:.2f} | "
            f"{item['mean_wall_us']:.1f} | {_component_text(item)} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", required=True)
    parser.add_argument("--methods", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    report = analyze(load_jsonl(args.steps), methods)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2))
    md = to_markdown(report)
    Path(args.output_md).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
