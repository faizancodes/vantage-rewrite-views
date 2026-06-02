"""Analyze baseline PLD opportunity traces.

The input is the ``steps.jsonl`` produced by ``scripts/run_eagle_eval.py`` with
``--pld-opportunity-trace`` enabled. The report focuses on whether weak PLD
steps consume enough runtime for PLD-adjacent methods to plausibly reach a
large speedup over tuned exact prompt lookup.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _pct(n: float, d: float) -> float:
    return 100.0 * n / d if d else 0.0


def _accepted_len(row: dict[str, Any]) -> int:
    val = row.get("pld_opp_accepted_len")
    if val is None:
        val = row.get("n_accepted_drafts", 0)
    return int(val or 0)


def _exact_hit(row: dict[str, Any]) -> bool:
    val = row.get("pld_opp_exact_hit")
    if val is None:
        val = row.get("pld_exact_hit")
    if val is None:
        return bool(row.get("k", 0))
    return bool(val)


def _candidate_matches(row: dict[str, Any]) -> int:
    val = row.get("pld_opp_candidate_matches")
    return int(val or 0)


def _hist_bins(values: list[int]) -> dict[str, int]:
    bins = {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5-8": 0,
        "9-16": 0,
        "17-32": 0,
        "33-64": 0,
        "65+": 0,
    }
    for v in values:
        if v <= 4:
            bins[str(v)] += 1
        elif v <= 8:
            bins["5-8"] += 1
        elif v <= 16:
            bins["9-16"] += 1
        elif v <= 32:
            bins["17-32"] += 1
        elif v <= 64:
            bins["33-64"] += 1
        else:
            bins["65+"] += 1
    return bins


def build_report(rows: list[dict[str, Any]], *, method: str) -> dict[str, Any]:
    method_rows = [r for r in rows if r.get("method") == method]
    traced = [r for r in method_rows if r.get("pld_opp_trace") is True]
    if traced:
        method_rows = traced
    n = len(method_rows)
    accepted = [_accepted_len(r) for r in method_rows]
    exact_hits = [r for r in method_rows if _exact_hit(r)]
    misses = [r for r in method_rows if not _exact_hit(r)]
    weak = [r for r in method_rows if _accepted_len(r) <= 4]
    ambiguous = [r for r in exact_hits if _candidate_matches(r) > 1]
    non_ambiguous = [r for r in exact_hits if _candidate_matches(r) <= 1]
    total_wall_us = sum(float(r.get("wall_us") or 0.0) for r in method_rows)
    weak_wall_us = sum(float(r.get("wall_us") or 0.0) for r in weak)
    weak_lookup_us = sum(float(r.get("pld_opp_lookup_us") or r.get("proposal_us") or 0.0) for r in weak)
    weak_verify_us = sum(float(r.get("pld_opp_verify_us") or r.get("verify_us") or 0.0) for r in weak)
    exact_hit_weak = [r for r in exact_hits if _accepted_len(r) <= 4]
    tasks = {str(r.get("task_id")) for r in method_rows if r.get("task_id") is not None}
    per_task_weak_wall = Counter()
    for r in weak:
        task_id = str(r.get("task_id"))
        per_task_weak_wall[task_id] += float(r.get("wall_us") or 0.0)

    return {
        "method": method,
        "n_steps": n,
        "n_tasks": len(tasks),
        "trace_present": bool(traced),
        "accepted_length_histogram": _hist_bins(accepted),
        "accepted_len_0_pct": _pct(sum(1 for v in accepted if v == 0), n),
        "accepted_len_le1_pct": _pct(sum(1 for v in accepted if v <= 1), n),
        "accepted_len_le4_pct": _pct(sum(1 for v in accepted if v <= 4), n),
        "exact_hits_accepted_le4_pct": _pct(len(exact_hit_weak), len(exact_hits)),
        "misses_pct": _pct(len(misses), n),
        "ambiguous_exact_hits_pct": _pct(len(ambiguous), len(exact_hits)),
        "mean_accepted_len": mean(accepted) if accepted else 0.0,
        "mean_accepted_len_ambiguous_hits": (
            mean(_accepted_len(r) for r in ambiguous) if ambiguous else 0.0
        ),
        "mean_accepted_len_non_ambiguous_hits": (
            mean(_accepted_len(r) for r in non_ambiguous) if non_ambiguous else 0.0
        ),
        "weak_pld_steps": len(weak),
        "weak_pld_wall_us": weak_wall_us,
        "weak_pld_lookup_us": weak_lookup_us,
        "weak_pld_verify_us": weak_verify_us,
        "weak_pld_runtime_fraction": weak_wall_us / total_wall_us if total_wall_us else 0.0,
        "total_wall_us": total_wall_us,
        "top_weak_runtime_tasks": per_task_weak_wall.most_common(20),
    }


def _print_markdown(report: dict[str, Any]) -> None:
    print(f"# PLD Opportunity Report: `{report['method']}`")
    print()
    print(f"- tasks: {report['n_tasks']}")
    print(f"- decode steps: {report['n_steps']}")
    print(f"- trace present: {report['trace_present']}")
    print(f"- mean accepted length: {report['mean_accepted_len']:.2f}")
    print(f"- accepted_len = 0: {report['accepted_len_0_pct']:.2f}%")
    print(f"- accepted_len <= 1: {report['accepted_len_le1_pct']:.2f}%")
    print(f"- accepted_len <= 4: {report['accepted_len_le4_pct']:.2f}%")
    print(f"- exact hits with accepted_len <= 4: {report['exact_hits_accepted_le4_pct']:.2f}%")
    print(f"- misses: {report['misses_pct']:.2f}%")
    print(f"- ambiguous exact hits (>1 candidate): {report['ambiguous_exact_hits_pct']:.2f}%")
    print(
        "- mean accepted length, ambiguous vs non-ambiguous hits: "
        f"{report['mean_accepted_len_ambiguous_hits']:.2f} vs "
        f"{report['mean_accepted_len_non_ambiguous_hits']:.2f}"
    )
    print(
        "- weak PLD runtime fraction (accepted_len <= 4): "
        f"{100.0 * report['weak_pld_runtime_fraction']:.2f}%"
    )
    print(
        "- weak PLD time: "
        f"wall={report['weak_pld_wall_us'] / 1e6:.3f}s, "
        f"lookup={report['weak_pld_lookup_us'] / 1e6:.3f}s, "
        f"verify={report['weak_pld_verify_us'] / 1e6:.3f}s"
    )
    print()
    print("| accepted length | steps |")
    print("|---:|---:|")
    for k, v in report["accepted_length_histogram"].items():
        print(f"| {k} | {v} |")

    frac = report["weak_pld_runtime_fraction"]
    print()
    if frac < 0.20:
        print(
            "**Interpretation:** weak PLD consumes <20% of wall time, so PLD-adjacent "
            "lookup methods are unlikely to deliver a 1.20x overall speedup on this mix."
        )
    elif frac >= 0.30:
        print(
            "**Interpretation:** weak PLD consumes >=30% of wall time, so a 1.20x "
            "PLD-adjacent path remains plausible if those steps can be rescued cheaply."
        )
    else:
        print(
            "**Interpretation:** weak PLD consumes 20-30% of wall time. There is some "
            "headroom, but a 1.20x overall speedup would require rescuing most weak steps "
            "with very low overhead."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("steps_jsonl", type=Path)
    parser.add_argument("--method", default="blazedit_pld_w128_n10")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    report = build_report(_load_jsonl(args.steps_jsonl), method=args.method)
    _print_markdown(report)
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
