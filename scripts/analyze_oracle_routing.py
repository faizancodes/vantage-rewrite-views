"""Compute per-task oracle routing upper bounds from completion timings."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _aggregate(rows: list[dict[str, Any]], method: str) -> dict[str, float]:
    tokens = 0
    wall_us = 0.0
    for row in rows:
        out = row.get("outputs", {}).get(method)
        if out is None:
            continue
        tokens += int(out.get("n_new_tokens", 0))
        wall_us += float(out.get("wall_us", 0.0))
    return {
        "n_new_tokens": tokens,
        "wall_us": wall_us,
        "tokens_per_sec": tokens / (wall_us / 1e6) if wall_us else 0.0,
    }


def analyze(rows: list[dict[str, Any]], methods: list[str], baseline: str) -> dict[str, Any]:
    by_method = {method: _aggregate(rows, method) for method in methods}
    baseline_tps = by_method[baseline]["tokens_per_sec"] if baseline in by_method else 0.0
    choices: Counter[str] = Counter()
    oracle_tokens = 0
    oracle_wall_us = 0.0
    task_rows = []

    for row in rows:
        candidates = []
        for method in methods:
            out = row.get("outputs", {}).get(method)
            if out is None:
                continue
            tokens = int(out.get("n_new_tokens", 0))
            wall_us = float(out.get("wall_us", 0.0))
            if wall_us <= 0:
                continue
            candidates.append((tokens / (wall_us / 1e6), method, tokens, wall_us))
        if not candidates:
            continue
        _, method, tokens, wall_us = max(candidates, key=lambda x: x[0])
        choices[method] += 1
        oracle_tokens += tokens
        oracle_wall_us += wall_us
        task_rows.append({"task_id": row.get("task_id"), "oracle_method": method})

    oracle_tps = oracle_tokens / (oracle_wall_us / 1e6) if oracle_wall_us else 0.0
    return {
        "n_tasks": len(task_rows),
        "baseline": baseline,
        "methods": methods,
        "by_method": by_method,
        "oracle": {
            "n_new_tokens": oracle_tokens,
            "wall_us": oracle_wall_us,
            "tokens_per_sec": oracle_tps,
            "speedup_vs_baseline": oracle_tps / baseline_tps if baseline_tps else 0.0,
            "choices": dict(choices),
        },
        "task_choices": task_rows,
    }


def to_markdown(report: dict[str, Any]) -> str:
    baseline = report["baseline"]
    base_tps = report["by_method"].get(baseline, {}).get("tokens_per_sec", 0.0)
    lines = [
        "# Oracle Routing Upper Bound",
        "",
        f"Tasks: {report['n_tasks']}. Baseline: `{baseline}`.",
        "",
        "## Methods",
        "",
        "| Method | tok/s | speedup vs baseline |",
        "|--------|------:|--------------------:|",
    ]
    for method, item in report["by_method"].items():
        speedup = item["tokens_per_sec"] / base_tps if base_tps else 0.0
        lines.append(f"| `{method}` | {item['tokens_per_sec']:.2f} | {speedup:.3f} |")
    oracle = report["oracle"]
    lines += [
        "",
        "## Oracle",
        "",
        f"Oracle tok/s: {oracle['tokens_per_sec']:.2f}. "
        f"Speedup vs `{baseline}`: {oracle['speedup_vs_baseline']:.3f}.",
        "",
        "| Chosen method | Tasks |",
        "|---------------|------:|",
    ]
    for method, n in sorted(oracle["choices"].items(), key=lambda item: -item[1]):
        lines.append(f"| `{method}` | {n} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completions", required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--baseline", default="vanilla")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    report = analyze(load_jsonl(args.completions), methods, args.baseline)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2))
    md = to_markdown(report)
    Path(args.output_md).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
