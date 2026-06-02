"""Compute PLD-vs-method oracle headroom across run directories.

The oracle chooses the faster row per task between exact PLD and a candidate
method, then aggregates total emitted tokens over chosen wall time.  This is a
ceiling for any router that can choose perfectly between those two methods.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _task_tps(output: dict[str, Any]) -> float:
    wall_us = float(output.get("wall_us", 0.0) or 0.0)
    tokens = int(output.get("n_new_tokens", 0) or 0)
    return tokens / (wall_us / 1e6) if wall_us > 0 else 0.0


def _aggregate(rows: list[dict[str, Any]], method: str) -> dict[str, float]:
    tokens = 0
    wall_us = 0.0
    n = 0
    for row in rows:
        out = (row.get("outputs") or {}).get(method)
        if not out:
            continue
        tokens += int(out.get("n_new_tokens", 0) or 0)
        wall_us += float(out.get("wall_us", 0.0) or 0.0)
        n += 1
    return {
        "n": n,
        "tokens": tokens,
        "wall_us": wall_us,
        "tokens_per_sec": tokens / (wall_us / 1e6) if wall_us > 0 else 0.0,
    }


def _oracle(rows: list[dict[str, Any]], baseline: str, candidate: str) -> dict[str, Any]:
    base = _aggregate(rows, baseline)
    cand = _aggregate(rows, candidate)
    oracle_tokens = 0
    oracle_wall_us = 0.0
    candidate_wins = 0
    baseline_wins = 0
    tied_or_missing = 0
    per_task: list[dict[str, Any]] = []

    for row in rows:
        outputs = row.get("outputs") or {}
        b = outputs.get(baseline)
        c = outputs.get(candidate)
        if not b or not c:
            tied_or_missing += 1
            continue
        b_tps = _task_tps(b)
        c_tps = _task_tps(c)
        if c_tps > b_tps:
            chosen = candidate
            out = c
            candidate_wins += 1
        else:
            chosen = baseline
            out = b
            baseline_wins += 1
        tokens = int(out.get("n_new_tokens", 0) or 0)
        wall_us = float(out.get("wall_us", 0.0) or 0.0)
        oracle_tokens += tokens
        oracle_wall_us += wall_us
        per_task.append(
            {
                "task_id": row.get("task_id"),
                "baseline_tps": b_tps,
                "candidate_tps": c_tps,
                "candidate_over_baseline": c_tps / b_tps if b_tps else 0.0,
                "oracle_method": chosen,
            }
        )

    oracle_tps = oracle_tokens / (oracle_wall_us / 1e6) if oracle_wall_us > 0 else 0.0
    base_tps = base["tokens_per_sec"]
    return {
        "baseline": baseline,
        "candidate": candidate,
        "n_tasks": len(per_task),
        "baseline_tps": base_tps,
        "candidate_tps": cand["tokens_per_sec"],
        "candidate_over_baseline": cand["tokens_per_sec"] / base_tps if base_tps else 0.0,
        "oracle_tps": oracle_tps,
        "oracle_over_baseline": oracle_tps / base_tps if base_tps else 0.0,
        "candidate_task_wins": candidate_wins,
        "baseline_task_wins": baseline_wins,
        "missing_or_tied": tied_or_missing,
        "per_task": per_task,
    }


def _tag_label(path: Path) -> str:
    name = path.name
    for prefix in ("vantage_adopt_", "vantage_dispatch_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _to_markdown(reports: list[dict[str, Any]]) -> str:
    lines = [
        "# Oracle Headroom",
        "",
        "Oracle chooses the faster row per task between exact PLD and the candidate.",
        "",
        "| Run | Candidate | Candidate/PLD | Oracle/PLD | Candidate task wins | Tasks |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        lines.append(
            "| {run} | `{candidate}` | {cand:.3f} | {oracle:.3f} | {wins}/{tasks} | {tasks} |".format(
                run=report["run"],
                candidate=report["candidate"],
                cand=report["candidate_over_baseline"],
                oracle=report["oracle_over_baseline"],
                wins=report["candidate_task_wins"],
                tasks=report["n_tasks"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--baseline", default="blazedit_pld_w128_n10")
    parser.add_argument(
        "--candidates",
        default="vantage_routed_transpld_m4_w128_n10,vantage_adopt_simple_transpld_m4_w128_n10",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    root = Path(args.raw_root)
    candidates = [m.strip() for m in args.candidates.split(",") if m.strip()]
    reports: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        completions = run_dir / "completions.jsonl"
        if not completions.exists():
            continue
        rows = _load_jsonl(completions)
        if not rows:
            continue
        methods = set((rows[0].get("outputs") or {}).keys())
        if args.baseline not in methods:
            continue
        for candidate in candidates:
            if candidate not in methods:
                continue
            report = _oracle(rows, args.baseline, candidate)
            report["run"] = _tag_label(run_dir)
            reports.append(report)

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(reports, indent=2))
    md = _to_markdown(reports)
    Path(args.output_md).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
