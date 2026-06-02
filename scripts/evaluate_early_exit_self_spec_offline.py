#!/usr/bin/env python3
"""Offline scaffold for early-exit self-speculation on PLD-weak steps.

This script deliberately does not implement a runtime decoder.  It summarizes
existing PLD traces and builds a cost-model shell for the next diagnostic: if
someone later records shallow-layer draft accuracy for the same step ids, this
script can combine it with measured shallow/full forward costs and report a
projected speedup.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _is_weak(row: dict[str, Any], threshold: int) -> bool:
    return int(row.get("n_accepted_drafts", 0) or 0) <= threshold


def _load_accuracy(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    rows = _read_jsonl(path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("step_key") or f"{row.get('task_id')}:{row.get('step')}")
        out[key] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True, help="steps.jsonl from a PLD run")
    ap.add_argument("--output-dir", default="analysis/early_exit_self_spec_offline")
    ap.add_argument("--weak-threshold", type=int, default=4)
    ap.add_argument("--layers", default="4,8,12,16,20")
    ap.add_argument("--draft-lengths", default="1,2,4")
    ap.add_argument(
        "--accuracy-jsonl",
        default="",
        help=(
            "Optional future shallow-draft rows keyed by task_id:step with fields "
            "layer, draft_len, token0_correct, accepted_len."
        ),
    )
    ap.add_argument("--full-forward-ms", type=float, default=0.0)
    ap.add_argument(
        "--shallow-forward-ms",
        default="",
        help="Optional comma list like 4:1.2,8:2.4 for cost-model projection.",
    )
    args = ap.parse_args()

    steps_path = Path(args.steps)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(steps_path)
    weak_rows = [r for r in rows if _is_weak(r, args.weak_threshold)]
    total_wall_us = sum(float(r.get("wall_us", 0.0) or 0.0) for r in rows)
    weak_wall_us = sum(float(r.get("wall_us", 0.0) or 0.0) for r in weak_rows)
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    draft_lengths = [int(x) for x in args.draft_lengths.split(",") if x.strip()]

    shallow_cost: dict[int, float] = {}
    for item in args.shallow_forward_ms.split(","):
        item = item.strip()
        if not item:
            continue
        k, v = item.split(":", 1)
        shallow_cost[int(k)] = float(v)

    accuracy = _load_accuracy(Path(args.accuracy_jsonl) if args.accuracy_jsonl else None)
    projections: list[dict[str, Any]] = []
    if accuracy and shallow_cost and args.full_forward_ms > 0:
        by_setting: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for row in accuracy.values():
            by_setting.setdefault((int(row["layer"]), int(row["draft_len"])), []).append(row)
        for layer in layers:
            for draft_len in draft_lengths:
                acc_rows = by_setting.get((layer, draft_len), [])
                if not acc_rows:
                    continue
                accepted = [float(r.get("accepted_len", 0.0) or 0.0) for r in acc_rows]
                token0 = [1.0 if r.get("token0_correct") else 0.0 for r in acc_rows]
                shallow_ms = shallow_cost.get(layer)
                if shallow_ms is None:
                    continue
                # Conservative one-use model: a shallow draft replaces one weak
                # full forward only when it accepts at least one token, otherwise
                # it adds shallow cost before falling back to the full forward.
                accepted_mean = mean(accepted) if accepted else 0.0
                token0_mean = mean(token0) if token0 else 0.0
                extra_ms = shallow_ms * len(weak_rows)
                saved_full_calls = sum(1 for x in accepted if x >= 1.0)
                saved_ms = saved_full_calls * args.full_forward_ms
                projected_wall_ms = max(1e-9, total_wall_us / 1000.0 + extra_ms - saved_ms)
                base_wall_ms = max(1e-9, total_wall_us / 1000.0)
                projections.append(
                    {
                        "layer": layer,
                        "draft_len": draft_len,
                        "accepted_tokens_per_use": accepted_mean,
                        "token0_accuracy": token0_mean,
                        "projected_speedup": base_wall_ms / projected_wall_ms,
                    }
                )

    report = {
        "steps_path": str(steps_path),
        "n_steps": len(rows),
        "weak_threshold": args.weak_threshold,
        "weak_steps": len(weak_rows),
        "weak_step_rate": len(weak_rows) / max(1, len(rows)),
        "weak_runtime_fraction": weak_wall_us / total_wall_us if total_wall_us > 0 else 0.0,
        "layers": layers,
        "draft_lengths": draft_lengths,
        "projection_available": bool(projections),
        "projections": projections,
        "note": (
            "This is a scaffold. To produce real early-exit projections, collect "
            "shallow-layer draft rows and pass --accuracy-jsonl plus measured "
            "--full-forward-ms and --shallow-forward-ms."
        ),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    md = [
        "# Early-Exit Self-Spec Offline Diagnostic",
        "",
        f"- steps: `{steps_path}`",
        f"- total steps: {len(rows)}",
        f"- weak steps (accepted <= {args.weak_threshold}): {len(weak_rows)}",
        f"- weak step rate: {report['weak_step_rate']:.3f}",
        f"- weak runtime fraction: {report['weak_runtime_fraction']:.3f}",
        "",
    ]
    if projections:
        md.extend(
            [
                "| layer | draft len | token0 acc | accepted/use | projected speedup |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for row in projections:
            md.append(
                f"| {row['layer']} | {row['draft_len']} | "
                f"{row['token0_accuracy']:.3f} | {row['accepted_tokens_per_use']:.2f} | "
                f"{row['projected_speedup']:.3f}x |"
            )
    else:
        md.append(
            "No shallow-layer accuracy rows were supplied, so this run only reports "
            "the PLD-weak opportunity size."
        )
    (out_dir / "report.md").write_text("\n".join(md) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
