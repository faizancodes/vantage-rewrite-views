#!/usr/bin/env python3
"""Summarize VANTAGE PLD proposer trace JSONL files."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_jsonl(Path(args.trace))
    summary = summarize(rows, trace_path=args.trace)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output_md:
        path = Path(args.output_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(markdown(summary)) + "\n", encoding="utf-8")
    return 0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict[str, Any]], *, trace_path: str) -> dict[str, Any]:
    proposal_lens = [int(row.get("proposal_len") or row.get("proposal_tokens") or 0) for row in rows]
    prefix_lens = [int(row.get("prefix_len") or 0) for row in rows]
    elapsed = [float(row.get("elapsed_us") or 0.0) for row in rows]
    nonempty = [value for value in proposal_lens if value > 0]
    caps = sorted({row.get("num_speculative_tokens_cap") for row in rows if row.get("num_speculative_tokens_cap") is not None})
    labels = sorted({row.get("equivalence_label_candidate") for row in rows if row.get("equivalence_label_candidate")})
    return {
        "trace_path": trace_path,
        "rows": len(rows),
        "match_found_rows": sum(1 for row in rows if row.get("match_found")),
        "nonempty_proposal_rows": len(nonempty),
        "hit_rate": (len(nonempty) / len(rows)) if rows else None,
        "proposal_len": stats(proposal_lens),
        "nonempty_proposal_len": stats(nonempty),
        "prefix_len": stats(prefix_lens),
        "elapsed_us": stats(elapsed),
        "caps": caps,
        "equivalence_label_candidates": labels,
        "token_trace_rows": sum(1 for row in rows if row.get("history_token_ids") is not None and row.get("proposal_token_ids") is not None),
    }


def stats(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    sorted_values = sorted(float(value) for value in values)
    return {
        "count": len(sorted_values),
        "mean": statistics.fmean(sorted_values),
        "median": percentile(sorted_values, 50),
        "p50": percentile(sorted_values, 50),
        "p90": percentile(sorted_values, 90),
        "p95": percentile(sorted_values, 95),
        "p99": percentile(sorted_values, 99),
        "min": sorted_values[0],
        "max": sorted_values[-1],
    }


def percentile(sorted_values: list[float], pct: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def markdown(summary: dict[str, Any]) -> list[str]:
    return [
        "# PLD Proposer Trace Summary",
        "",
        f"- Trace: `{summary['trace_path']}`",
        f"- Rows: `{summary['rows']}`",
        f"- Nonempty proposal rows: `{summary['nonempty_proposal_rows']}`",
        f"- Hit rate: `{fmt(summary['hit_rate'], 4)}`",
        f"- Proposal length mean/p50/p95/max: `{fmt(summary['proposal_len']['mean'], 2)}/{fmt(summary['proposal_len']['p50'], 1)}/{fmt(summary['proposal_len']['p95'], 1)}/{fmt(summary['proposal_len']['max'], 0)}`",
        f"- Nonempty proposal length mean/p50/p95/max: `{fmt(summary['nonempty_proposal_len']['mean'], 2)}/{fmt(summary['nonempty_proposal_len']['p50'], 1)}/{fmt(summary['nonempty_proposal_len']['p95'], 1)}/{fmt(summary['nonempty_proposal_len']['max'], 0)}`",
        f"- Prefix length mean/p50/p95/max: `{fmt(summary['prefix_len']['mean'], 1)}/{fmt(summary['prefix_len']['p50'], 1)}/{fmt(summary['prefix_len']['p95'], 1)}/{fmt(summary['prefix_len']['max'], 0)}`",
        f"- Proposer elapsed us mean/p95/max: `{fmt(summary['elapsed_us']['mean'], 2)}/{fmt(summary['elapsed_us']['p95'], 2)}/{fmt(summary['elapsed_us']['max'], 2)}`",
        f"- Token trace rows: `{summary['token_trace_rows']}`",
        f"- Caps: `{summary['caps']}`",
        f"- Label candidates: `{summary['equivalence_label_candidates']}`",
    ]


def fmt(value: Any, digits: int) -> str:
    if value is None:
        return "not captured"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


if __name__ == "__main__":
    raise SystemExit(main())
