#!/usr/bin/env python3
"""Summarize PLD-adjacent decoder benchmark aggregates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("aggregate_json", type=Path)
    parser.add_argument("--baseline", default="blazedit_pld_w128_n10")
    args = parser.parse_args()

    agg = json.loads(args.aggregate_json.read_text())
    by_method = agg.get("by_method", {})
    baseline = by_method.get(args.baseline)
    if not baseline:
        raise SystemExit(f"baseline {args.baseline!r} not found in aggregate")
    baseline_tps = float(baseline.get("tokens_per_sec", 0.0) or 1.0)
    output_eq = agg.get("output_equivalence", {})

    rows = []
    for method, row in by_method.items():
        if method == "vanilla":
            continue
        tps = float(row.get("tokens_per_sec", 0.0) or 0.0)
        eq = output_eq.get(method, {})
        tasks = int(eq.get("tasks", 0) or 0)
        matches_vanilla = int(eq.get("matches_vanilla", 0) or 0)
        rows.append(
            {
                "method": method,
                "tok_s": tps,
                "ratio": tps / baseline_tps if baseline_tps else 0.0,
                "exact_hit": row.get("pld_exact_hit_rate"),
                "triggers": row.get("pld_variant_triggers_total"),
                "token01": row.get("pld_trigger_token01_rejection_rate")
                if row.get("pld_trigger_token01_rejection_rate") is not None
                else row.get("pld_token01_rejection_rate"),
                "overhead": row.get("pld_variant_overhead_us_per_step"),
                "delta_reuse": row.get("pld_delta_reuse_rate"),
                "delta_tail": row.get("pld_delta_patch_accept_tail_mean"),
                "fuzzy_hit": row.get("pld_fuzzy_hit_rate_among_exact_misses"),
                "vanilla_parity": (matches_vanilla / tasks) if tasks else None,
            }
        )

    headers = [
        "Method",
        "tok/s",
        "vs PLD",
        "exact hit",
        "triggers",
        "tok0/1 rej",
        "overhead us/step",
        "delta reuse",
        "delta tail",
        "fuzzy/miss",
        "vanilla parity",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        def maybe_pct(value):
            return "-" if value is None else _fmt_pct(float(value))

        print(
            "| "
            + " | ".join(
                [
                    row["method"],
                    f"{row['tok_s']:.1f}",
                    f"{row['ratio']:.3f}x",
                    maybe_pct(row["exact_hit"]),
                    "-" if row["triggers"] is None else str(int(row["triggers"] or 0)),
                    maybe_pct(row["token01"]),
                    "-" if row["overhead"] is None else f"{float(row['overhead']):.2f}",
                    maybe_pct(row["delta_reuse"]),
                    "-" if row["delta_tail"] is None else f"{float(row['delta_tail']):.2f}",
                    maybe_pct(row["fuzzy_hit"]),
                    maybe_pct(row["vanilla_parity"]),
                ]
            )
            + " |"
        )


if __name__ == "__main__":
    main()
