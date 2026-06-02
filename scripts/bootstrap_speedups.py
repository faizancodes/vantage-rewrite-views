"""Bootstrap confidence intervals for run_eagle_eval.py completions.

The evaluation harness records one completion row per task with per-method
generated-token counts and wall times.  This script bootstraps over tasks,
which matches the paper's within-run comparison convention.
"""

from __future__ import annotations

import argparse
import json
import random
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


def _tps(rows: list[dict[str, Any]], method: str) -> float:
    tokens = 0
    wall_us = 0.0
    for row in rows:
        out = row.get("outputs", {}).get(method)
        if out is None:
            continue
        tokens += int(out.get("n_new_tokens", 0))
        wall_us += float(out.get("wall_us", 0.0))
    return tokens / (wall_us / 1e6) if wall_us > 0 else 0.0


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def _ratio(rows: list[dict[str, Any]], method: str, baseline: str) -> float:
    base = _tps(rows, baseline)
    val = _tps(rows, method)
    return val / base if base > 0 else 0.0


def parse_pairs(text: str) -> list[tuple[str, str]]:
    pairs = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"pair entries must be method:baseline, got {item!r}")
        method, baseline = item.split(":", 1)
        pairs.append((method.strip(), baseline.strip()))
    return pairs


def bootstrap(
    rows: list[dict[str, Any]],
    *,
    methods: list[str],
    baseline: str,
    pairs: list[tuple[str, str]],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    n = len(rows)
    point_tps = {method: _tps(rows, method) for method in methods}
    point_speedups = {
        method: point_tps[method] / point_tps[baseline]
        if point_tps.get(baseline, 0.0) > 0
        else 0.0
        for method in methods
    }
    point_pairs = {
        f"{method}_vs_{base}": _ratio(rows, method, base) for method, base in pairs
    }

    samples_by_method: dict[str, list[float]] = {method: [] for method in methods}
    samples_by_pair: dict[str, list[float]] = {k: [] for k in point_pairs}
    for _ in range(n_boot):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        for method in methods:
            samples_by_method[method].append(_ratio(sample, method, baseline))
        for method, base in pairs:
            samples_by_pair[f"{method}_vs_{base}"].append(_ratio(sample, method, base))

    by_method = {}
    for method in methods:
        xs = samples_by_method[method]
        by_method[method] = {
            "tokens_per_sec": point_tps[method],
            f"speedup_vs_{baseline}": point_speedups[method],
            "ci95": [_percentile(xs, 0.025), _percentile(xs, 0.975)],
            "p_gt_1": sum(1 for x in xs if x > 1.0) / len(xs) if xs else 0.0,
        }

    by_pair = {}
    for key, point in point_pairs.items():
        xs = samples_by_pair[key]
        by_pair[key] = {
            "ratio": point,
            "ci95": [_percentile(xs, 0.025), _percentile(xs, 0.975)],
            "p_gt_1": sum(1 for x in xs if x > 1.0) / len(xs) if xs else 0.0,
        }
    return {
        "n_tasks": n,
        "baseline": baseline,
        "n_boot": n_boot,
        "seed": seed,
        "by_method": by_method,
        "by_pair": by_pair,
    }


def to_markdown(report: dict[str, Any]) -> str:
    baseline = report["baseline"]
    lines = [
        "# Bootstrap Speedup Confidence Intervals",
        "",
        f"Tasks: {report['n_tasks']}. Bootstrap samples: {report['n_boot']}. Baseline: `{baseline}`.",
        "",
        "## Method Speedups",
        "",
        "| Method | tok/s | Speedup | 95% CI | P(speedup > 1) |",
        "|--------|------:|--------:|-------:|---------------:|",
    ]
    for method, item in report["by_method"].items():
        lo, hi = item["ci95"]
        lines.append(
            f"| `{method}` | {item['tokens_per_sec']:.2f} | "
            f"{item[f'speedup_vs_{baseline}']:.3f} | "
            f"[{lo:.3f}, {hi:.3f}] | {item['p_gt_1']:.3f} |"
        )
    lines += ["", "## Pairwise Ratios", "", "| Pair | Ratio | 95% CI | P(ratio > 1) |", "|------|------:|-------:|-------------:|"]
    for pair, item in report["by_pair"].items():
        lo, hi = item["ci95"]
        lines.append(
            f"| `{pair}` | {item['ratio']:.3f} | [{lo:.3f}, {hi:.3f}] | {item['p_gt_1']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completions", required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--baseline", default="vanilla")
    parser.add_argument("--pairs", default="")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    pairs = parse_pairs(args.pairs)
    report = bootstrap(
        load_jsonl(args.completions),
        methods=methods,
        baseline=args.baseline,
        pairs=pairs,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2))
    md = to_markdown(report)
    Path(args.output_md).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
