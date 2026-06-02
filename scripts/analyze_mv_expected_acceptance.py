#!/usr/bin/env python3
"""Train a tiny offline expected-acceptance scorer for VANTAGE-MV traces.

This is intentionally trace-only: it does not claim to be a deployed router.
It answers whether the existing hard gates leave a learnable signal in the
steps that did run transformed lookup.  We bucket transformed attempts by
pre-verification features and evaluate, on a held-out task split, whether a
bucket scorer can keep zero-accepts low while retaining a useful fraction of
accepted transformed tokens.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TRANS_KINDS = {
    "vantage_mv_pld",
    "vantage_mv_branch_tree",
    "vantage_mv_branch_common",
}


@dataclass
class Attempt:
    task_id: str
    method: str
    accepted: int
    emitted: int
    proposal_tokens: int
    match_len: int
    frontier_distance: int | None
    route_reason: str
    proposal_us: float
    verify_us: float


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _is_trans_attempt(row: dict[str, Any]) -> bool:
    # Route reasons also include skipped prechecks such as
    # ``trans_precheck_no_token_candidate``.  The scorer should train only on
    # transformed candidates that were actually verified.
    return row.get("proposal_kind") in TRANS_KINDS and int(row.get("proposal_tokens") or 0) > 0


def _attempt_from_row(row: dict[str, Any]) -> Attempt:
    return Attempt(
        task_id=str(row.get("task_id")),
        method=str(row.get("method")),
        accepted=int(row.get("n_accepted_nonroot_drafts") or 0),
        emitted=int(row.get("n_emitted") or 0),
        proposal_tokens=int(row.get("proposal_tokens") or 0),
        match_len=int(row.get("proposal_match_len") or 0),
        frontier_distance=(
            int(row["proposal_frontier_distance"])
            if row.get("proposal_frontier_distance") is not None
            else None
        ),
        route_reason=str(row.get("proposal_route_reason") or ""),
        proposal_us=float(row.get("proposal_us") or 0.0),
        verify_us=float(row.get("verify_us") or 0.0),
    )


def _task_fold(task_id: str, train_fraction: float) -> str:
    # Stable, dependency-free split.
    h = 0
    for ch in task_id:
        h = (h * 131 + ord(ch)) % 10_000
    return "fit" if h < int(train_fraction * 10_000) else "eval"


def _bin_attempt(a: Attempt) -> tuple[str, str, str, str]:
    if a.match_len >= 10:
        match = "m10+"
    elif a.match_len >= 8:
        match = "m8-9"
    elif a.match_len >= 4:
        match = "m4-7"
    else:
        match = "m0-3"
    if a.frontier_distance is None:
        frontier = "f_none"
    elif a.frontier_distance == 0:
        frontier = "f0"
    elif a.frontier_distance <= 8:
        frontier = "f1-8"
    elif a.frontier_distance <= 32:
        frontier = "f9-32"
    else:
        frontier = "f33+"
    if a.proposal_tokens >= 64:
        length = "l64+"
    elif a.proposal_tokens >= 32:
        length = "l32-63"
    elif a.proposal_tokens >= 16:
        length = "l16-31"
    else:
        length = "l0-15"
    reason = a.route_reason or "unknown"
    return match, frontier, length, reason


def _bucket_stats(attempts: list[Attempt]) -> dict[tuple[str, str, str, str], dict[str, float]]:
    acc: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(
        lambda: {"n": 0.0, "accepted": 0.0, "zero": 0.0, "proposal_us": 0.0, "verify_us": 0.0}
    )
    for a in attempts:
        key = _bin_attempt(a)
        row = acc[key]
        row["n"] += 1
        row["accepted"] += a.accepted
        row["zero"] += 1 if a.accepted == 0 else 0
        row["proposal_us"] += a.proposal_us
        row["verify_us"] += a.verify_us
    out: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for key, row in acc.items():
        n = max(1.0, row["n"])
        out[key] = {
            **row,
            "mean_accepted": row["accepted"] / n,
            "zero_rate": row["zero"] / n,
            "mean_cost_us": (row["proposal_us"] + row["verify_us"]) / n,
        }
    return out


def _evaluate(
    attempts: list[Attempt],
    stats: dict[tuple[str, str, str, str], dict[str, float]],
    *,
    min_expected_accept: float,
    max_expected_zero: float,
) -> dict[str, Any]:
    selected: list[Attempt] = []
    for a in attempts:
        s = stats.get(_bin_attempt(a))
        if not s:
            continue
        if s["mean_accepted"] >= min_expected_accept and s["zero_rate"] <= max_expected_zero:
            selected.append(a)

    total_accepted = sum(a.accepted for a in attempts)
    selected_accepted = sum(a.accepted for a in selected)
    return {
        "attempts": len(attempts),
        "selected_attempts": len(selected),
        "accepted_tokens": total_accepted,
        "selected_accepted_tokens": selected_accepted,
        "accepted_capture": selected_accepted / total_accepted if total_accepted else 0.0,
        "zero_accept_rate": (
            sum(1 for a in selected if a.accepted == 0) / len(selected)
            if selected
            else 0.0
        ),
        "mean_selected_accepted": (
            selected_accepted / len(selected) if selected else 0.0
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True)
    ap.add_argument("--methods", required=True, help="comma-separated MV methods")
    ap.add_argument("--fit-fraction", type=float, default=0.7)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    rows = _load_jsonl(Path(args.steps))
    attempts_by_method: dict[str, list[Attempt]] = {m: [] for m in methods}
    for row in rows:
        method = str(row.get("method"))
        if method in attempts_by_method and _is_trans_attempt(row):
            attempts_by_method[method].append(_attempt_from_row(row))

    thresholds = [
        (2.0, 0.08),
        (4.0, 0.08),
        (8.0, 0.08),
        (4.0, 0.10),
        (8.0, 0.10),
        (16.0, 0.10),
    ]
    report: dict[str, Any] = {"methods": {}}
    for method, attempts in attempts_by_method.items():
        fit = [a for a in attempts if _task_fold(a.task_id, args.fit_fraction) == "fit"]
        eval_ = [a for a in attempts if _task_fold(a.task_id, args.fit_fraction) == "eval"]
        stats = _bucket_stats(fit)
        eval_rows = []
        for min_acc, max_zero in thresholds:
            eval_rows.append(
                {
                    "min_expected_accept": min_acc,
                    "max_expected_zero": max_zero,
                    **_evaluate(
                        eval_,
                        stats,
                        min_expected_accept=min_acc,
                        max_expected_zero=max_zero,
                    ),
                }
            )
        report["methods"][method] = {
            "attempts": len(attempts),
            "fit_attempts": len(fit),
            "eval_attempts": len(eval_),
            "fit_buckets": len(stats),
            "eval": eval_rows,
            "top_buckets": [
                {"bucket": list(k), **v}
                for k, v in sorted(
                    stats.items(),
                    key=lambda kv: (kv[1]["mean_accepted"], -kv[1]["zero_rate"], kv[1]["n"]),
                    reverse=True,
                )[:20]
            ],
        }

    Path(args.output_json).write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = ["# MV Expected-Acceptance Scorer", ""]
    for method, payload in report["methods"].items():
        lines.append(f"## `{method}`")
        lines.append("")
        lines.append(
            f"Attempts: {payload['attempts']} (fit {payload['fit_attempts']}, eval {payload['eval_attempts']}); buckets: {payload['fit_buckets']}."
        )
        lines.append("")
        lines.append("| min E[accept] | max zero | selected | accepted capture | selected zero | mean selected accept |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for row in payload["eval"]:
            lines.append(
                "| "
                f"{row['min_expected_accept']:.1f} | "
                f"{row['max_expected_zero']:.2f} | "
                f"{row['selected_attempts']}/{row['attempts']} | "
                f"{row['accepted_capture']:.1%} | "
                f"{row['zero_accept_rate']:.1%} | "
                f"{row['mean_selected_accepted']:.2f} |"
            )
        lines.append("")
    Path(args.output_md).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
