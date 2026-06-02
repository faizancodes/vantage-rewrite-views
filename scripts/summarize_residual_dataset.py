#!/usr/bin/env python3
"""Summarize VANTAGE residual dataset artifacts into a markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "artifacts/vantage_residual/data/residual_dataset_summary.json"
DEFAULT_OUTPUT = ROOT / "artifacts/vantage_residual/tables/residual_dataset_summary.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(markdown_lines(summary)) + "\n", encoding="utf-8")
    print(f"wrote {args.output_md}")
    return 0


def markdown_lines(summary: dict[str, Any]) -> list[str]:
    lines = [
        "# Residual Dataset Summary",
        "",
        f"source kind: `{summary.get('source_kind', 'unknown')}`",
        f"accepted_len filter: `<= {summary.get('accepted_len_threshold', 'n/a')}`",
        f"allow all tokens: `{summary.get('allow_all_tokens', False)}`",
        f"PLD miss only: `{summary.get('pld_miss_only', False)}`",
        "",
        "| Split | Source rows | Eligible rows | Written rows | Hidden shape | Labels shape | PLD misses | Tasks | accepted_len histogram |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for split in ("train", "test"):
        row = (summary.get("splits") or {}).get(split, {})
        hist = ", ".join(
            f"{key}:{value}"
            for key, value in (row.get("accepted_len_histogram") or {}).items()
        )
        lines.append(
            "| {split} | {source} | {eligible} | {written} | `{hidden}` | `{labels}` | {misses} | {tasks} | {hist} |".format(
                split=split,
                source=row.get("source_examples", 0),
                eligible=row.get("eligible_examples", 0),
                written=row.get("written_examples", 0),
                hidden=row.get("hidden_shape", []),
                labels=row.get("labels_shape", []),
                misses=row.get("pld_miss_count", 0),
                tasks=row.get("task_count", 0),
                hist=hist,
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- This is a data packaging summary only; no residual model was trained.",
            "- A PLD residual dataset is valid only when threshold filtering is enabled or explicitly overridden.",
        ]
    )
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
