"""Serving-style latency metrics from VANTAGE research-harness traces."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


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


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)) if values else 0.0,
        "p50": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
    }


def analyze(
    completions: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    *,
    methods: list[str],
) -> dict[str, Any]:
    steps_by_method_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for step in steps:
        steps_by_method_task[(str(step.get("method")), str(step.get("task_id")))].append(step)
    for vals in steps_by_method_task.values():
        vals.sort(key=lambda r: int(r.get("step") or 0))

    groups = []
    vanilla_steps = {
        task: len(vals)
        for (method, task), vals in steps_by_method_task.items()
        if method == "vanilla"
    }
    for method in methods:
        task_rows = []
        for row in completions:
            task_id = str(row.get("task_id"))
            output = (row.get("outputs") or {}).get(method)
            if not output:
                continue
            method_steps = steps_by_method_task.get((method, task_id), [])
            wall_us = float(output.get("wall_us") or 0.0)
            new_tokens = float(output.get("n_new_tokens") or 0.0)
            decode_step_us = sum(float(s.get("wall_us") or 0.0) for s in method_steps)
            proposal_us = sum(float(s.get("proposal_us") or 0.0) for s in method_steps)
            target_prefill_us = sum(float(s.get("target_prefill_us") or 0.0) for s in method_steps)
            first_step_us = float(method_steps[0].get("wall_us") or 0.0) if method_steps else 0.0
            tokens_per_verify = new_tokens / len(method_steps) if method_steps else 0.0
            vanilla_n = vanilla_steps.get(task_id, 0)
            task_rows.append(
                {
                    "task_id": task_id,
                    "wall_us": wall_us,
                    "decode_step_us": decode_step_us,
                    "first_step_us": first_step_us + target_prefill_us,
                    "proposal_us_per_token": proposal_us / new_tokens if new_tokens else 0.0,
                    "tokens_per_verify": tokens_per_verify,
                    "target_forward_reduction": (
                        1.0 - len(method_steps) / vanilla_n if vanilla_n else 0.0
                    ),
                    "tokens_per_sec": new_tokens / (wall_us / 1e6) if wall_us else 0.0,
                    "hit_max_new_tokens": any(bool(s.get("hit_max_new_tokens")) for s in method_steps),
                }
            )
        groups.append(
            {
                "method": method,
                "n": len(task_rows),
                "wall_ms": _summary([r["wall_us"] / 1000.0 for r in task_rows]),
                "decode_step_ms": _summary([r["decode_step_us"] / 1000.0 for r in task_rows]),
                "first_step_proxy_ms": _summary([r["first_step_us"] / 1000.0 for r in task_rows]),
                "proposal_us_per_token": _summary([r["proposal_us_per_token"] for r in task_rows]),
                "tokens_per_verify": _summary([r["tokens_per_verify"] for r in task_rows]),
                "target_forward_reduction": _summary([r["target_forward_reduction"] for r in task_rows]),
                "tokens_per_sec": _summary([r["tokens_per_sec"] for r in task_rows]),
                "hit_max_new_tokens_rate": (
                    sum(1 for r in task_rows if r["hit_max_new_tokens"]) / len(task_rows)
                    if task_rows
                    else 0.0
                ),
                "tasks": task_rows,
            }
        )
    return {"schema": "asts-spec/latency/v1", "methods": methods, "groups": groups}


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Serving-Style Latency Metrics",
        "",
        "| Method | n | p50 wall ms | p95 wall ms | p50 first-step proxy ms | tok/s mean | tokens/verify | target forward reduction | proposal us/token | hit max-new |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["groups"]:
        lines.append(
            f"| {row['method']} | {row['n']} | {row['wall_ms']['p50']:.2f} | "
            f"{row['wall_ms']['p95']:.2f} | {row['first_step_proxy_ms']['p50']:.2f} | "
            f"{row['tokens_per_sec']['mean']:.2f} | {row['tokens_per_verify']['mean']:.2f} | "
            f"{100 * row['target_forward_reduction']['mean']:.1f}% | "
            f"{row['proposal_us_per_token']['mean']:.2f} | "
            f"{100 * row['hit_max_new_tokens_rate']:.1f}% |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--completions", required=True)
    p.add_argument("--steps", required=True)
    p.add_argument("--methods", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    args = p.parse_args()
    report = analyze(
        _load_jsonl(Path(args.completions)),
        _load_jsonl(Path(args.steps)),
        methods=[m.strip() for m in args.methods.split(",") if m.strip()],
    )
    Path(args.output_json).write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, Path(args.output_md))


if __name__ == "__main__":
    main()
