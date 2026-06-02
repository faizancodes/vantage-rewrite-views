"""Compare original, prompt-augmented, and oracle prompt PLD rows."""

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


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _tps(output: dict[str, Any]) -> float:
    wall_us = float(output.get("wall_us") or 0.0)
    tokens = float(output.get("n_new_tokens") or 0.0)
    return tokens / (wall_us / 1e6) if wall_us > 0 else 0.0


def analyze(
    completions: list[dict[str, Any]],
    *,
    methods: list[str],
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in completions:
        meta = row.get("metadata") or {}
        mode = str(meta.get("prompt_mode") or row.get("prompt_mode") or "original")
        groups[mode].append(row)
    summaries = []
    for mode, rows in sorted(groups.items()):
        for method in methods:
            task_rows = []
            for row in rows:
                output = (row.get("outputs") or {}).get(method)
                if not output:
                    continue
                prompt_tokens = float(
                    (row.get("metadata") or {}).get("prompt_tokens")
                    or row.get("prompt_tokens")
                    or 0.0
                )
                task_rows.append(
                    {
                        "task_id": row.get("task_id"),
                        "tokens_per_sec": _tps(output),
                        "wall_us": float(output.get("wall_us") or 0.0),
                        "n_new_tokens": float(output.get("n_new_tokens") or 0.0),
                        "prompt_tokens": prompt_tokens,
                    }
                )
            if not task_rows:
                continue
            summaries.append(
                {
                    "prompt_mode": mode,
                    "method": method,
                    "n": len(task_rows),
                    "tokens_per_sec_mean": _mean([r["tokens_per_sec"] for r in task_rows]),
                    "wall_us_mean": _mean([r["wall_us"] for r in task_rows]),
                    "new_tokens_mean": _mean([r["n_new_tokens"] for r in task_rows]),
                    "prompt_tokens_mean": _mean([r["prompt_tokens"] for r in task_rows]),
                }
            )
    return {
        "schema": "asts-spec/prompt-oracle/v1",
        "methods": methods,
        "groups": summaries,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Prompt Augmentation Oracle",
        "",
        "| Prompt mode | Method | n | tok/s | mean wall ms | mean new tokens | mean prompt tokens |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["groups"]:
        lines.append(
            f"| {row['prompt_mode']} | {row['method']} | {row['n']} | "
            f"{row['tokens_per_sec_mean']:.2f} | {row['wall_us_mean'] / 1000.0:.2f} | "
            f"{row['new_tokens_mean']:.1f} | {row['prompt_tokens_mean']:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--completions", required=True)
    p.add_argument("--methods", required=True, help="Comma-separated method names")
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    args = p.parse_args()
    report = analyze(
        _load_jsonl(Path(args.completions)),
        methods=[m.strip() for m in args.methods.split(",") if m.strip()],
    )
    Path(args.output_json).write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, Path(args.output_md))


if __name__ == "__main__":
    main()
