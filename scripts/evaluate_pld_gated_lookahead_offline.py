#!/usr/bin/env python3
"""Offline summary for PLD-gated Lookahead runs.

This script intentionally does not run an extra decoder.  It consumes
``steps.jsonl`` from ``scripts/run_eagle_eval.py`` and reports whether the
observed weak-PLD and Lookahead steps have enough coverage to justify a larger
GPU sweep.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def _load_steps(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _summarize(rows: list[dict], method: str, weak_threshold: int) -> dict:
    method_rows = [r for r in rows if r.get("method") == method]
    total_wall = sum(float(r.get("wall_us", 0.0) or 0.0) for r in method_rows)
    total_tokens = sum(int(r.get("n_emitted", 0) or 0) for r in method_rows)
    weak_rows = [
        r
        for r in method_rows
        if int(r.get("n_accepted_drafts", 0) or 0) <= int(weak_threshold)
    ]
    lookahead_rows = [
        r for r in method_rows if r.get("lookahead_triggered") is True
    ]
    accepted = [int(r.get("lookahead_accepted_len", 0) or 0) for r in lookahead_rows]
    candidate = [int(r.get("lookahead_candidate_len", 0) or 0) for r in lookahead_rows]
    calls = sum(int(r.get("lookahead_forward_calls", 0) or 0) for r in lookahead_rows)
    lookahead_us = sum(float(r.get("lookahead_us", 0.0) or 0.0) for r in lookahead_rows)
    lookahead_forward_us = sum(
        float(r.get("lookahead_forward_us", 0.0) or 0.0) for r in lookahead_rows
    )
    lookahead_verify_us = sum(
        float(r.get("lookahead_verify_us", 0.0) or 0.0) for r in lookahead_rows
    )
    lookahead_build_us = sum(
        float(r.get("lookahead_candidate_build_us", 0.0) or 0.0)
        for r in lookahead_rows
    )
    return {
        "method": method,
        "steps": len(method_rows),
        "tokens": total_tokens,
        "tokens_per_sec": total_tokens / (total_wall / 1e6) if total_wall else 0.0,
        "weak_step_count": len(weak_rows),
        "weak_runtime_fraction": (
            sum(float(r.get("wall_us", 0.0) or 0.0) for r in weak_rows) / total_wall
            if total_wall
            else 0.0
        ),
        "lookahead_calls": len(lookahead_rows),
        "lookahead_candidate_len_mean": mean(candidate) if candidate else 0.0,
        "lookahead_accepted_len_mean": mean(accepted) if accepted else 0.0,
        "lookahead_tok0_reject_rate": (
            sum(1 for r in lookahead_rows if r.get("lookahead_tok0_reject") is True)
            / max(1, len(lookahead_rows))
        ),
        "lookahead_forward_calls_per_call": calls / max(1, len(lookahead_rows)),
        "lookahead_ms_per_call": lookahead_us / max(1, len(lookahead_rows)) / 1000.0,
        "lookahead_forward_ms_per_call": (
            lookahead_forward_us / max(1, len(lookahead_rows)) / 1000.0
        ),
        "lookahead_verify_ms_per_call": (
            lookahead_verify_us / max(1, len(lookahead_rows)) / 1000.0
        ),
        "lookahead_candidate_build_ms_mean": (
            lookahead_build_us / max(1, len(lookahead_rows)) / 1000.0
        ),
        "lookahead_accepted_per_forward": sum(accepted) / max(1, calls),
        "total_model_forward_calls": len(method_rows) + calls,
        "total_model_forward_ms": (
            sum(float(r.get("verify_us", 0.0) or 0.0) for r in method_rows)
            + lookahead_forward_us
        )
        / 1000.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True, type=Path)
    ap.add_argument("--pld-method", default="blazedit_pld_w128_n10")
    ap.add_argument("--lookahead-method", default="pld_gated_lookahead_w128_n10")
    ap.add_argument("--weak-threshold", type=int, default=4)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    rows = _load_steps(args.steps)
    report = {
        "weak_threshold": args.weak_threshold,
        "methods": {
            args.pld_method: _summarize(rows, args.pld_method, args.weak_threshold),
            args.lookahead_method: _summarize(
                rows, args.lookahead_method, args.weak_threshold
            ),
        },
    }
    pld = report["methods"][args.pld_method]
    gated = report["methods"][args.lookahead_method]
    report["speedup_vs_pld"] = (
        gated["tokens_per_sec"] / pld["tokens_per_sec"]
        if pld["tokens_per_sec"]
        else 0.0
    )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
