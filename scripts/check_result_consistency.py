#!/usr/bin/env python3
"""Check paper-facing archived continuous-batched PLD prototype result arithmetic against source JSON.

This script intentionally does not trust hand-written LaTeX. It verifies the
headline repeated timing report, controlled ablation arithmetic when present,
and the historically confusing older 845.0-vs-753.9 diagnostic row.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "result_consistency_report.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _close(a: float, b: float, tol: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def _check(condition: bool, failures: list[str], msg: str) -> None:
    if not condition:
        failures.append(msg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--tolerance", type=float, default=5e-3)
    args = parser.parse_args()

    failures: list[str] = []
    notes: list[str] = []

    repeats_path = ROOT / "analysis" / "continuous_batched_pld_final_repeats" / "report.json"
    ablation_path = ROOT / "analysis" / "batched_pld_ablation" / "report.json"
    controlled_ablation_path = (
        ROOT
        / "analysis"
        / "batched_pld_controlled_ablation"
        / "controlled_ablation_test500_v1"
        / "summary.json"
    )
    repeats = _load(repeats_path)
    ablation = _load(ablation_path)

    summary = repeats["summary"]
    seq = summary["blazedit_pld_w128_n10_b1"]
    b8 = summary["continuous_batched_pld_w128_n10_b8"]
    seq_tok_s = float(seq["tok_s"]["mean"])
    b8_tok_s = float(b8["tok_s"]["mean"])
    b8_speedup = float(b8["speedup"]["mean"])
    expected_b8_speedup = b8_tok_s / seq_tok_s
    _check(
        _close(b8_speedup, expected_b8_speedup, args.tolerance),
        failures,
        f"headline b8 speedup {b8_speedup:.6f} != {expected_b8_speedup:.6f}",
    )
    seq_forwards = float(seq["verifier_forwards"]["mean"])
    b8_forwards = float(b8["verifier_forwards"]["mean"])
    expected_reduction = 100.0 * (1.0 - b8_forwards / seq_forwards)
    reported_reduction = float(b8["verifier_forward_reduction"]["mean"])
    _check(
        _close(reported_reduction, expected_reduction, args.tolerance),
        failures,
        f"headline forward reduction {reported_reduction:.6f} != {expected_reduction:.6f}",
    )

    ab_seq = ablation["sequential"]
    ab_seq_tok_s = float(ab_seq["tokens_per_sec"])
    rows = {row["config_id"]: row for row in ablation["rows"]}
    ab_final = rows["b8_pool32_default_continuous"]
    ab_tok_s = float(ab_final["generated_tokens_per_sec"])
    ab_speedup = float(ab_final["speedup_vs_sequential"])
    expected_ab_speedup = ab_tok_s / ab_seq_tok_s
    _check(
        _close(ab_speedup, expected_ab_speedup, args.tolerance),
        failures,
        f"ablation speedup {ab_speedup:.6f} != {expected_ab_speedup:.6f}",
    )
    if not _close(ab_tok_s, b8_tok_s, 0.01):
        notes.append(
            "The ablation row b8_pool32_default_continuous is a separate "
            f"single-run ablation ({ab_tok_s:.1f} tok/s, same-run baseline "
            f"{ab_seq_tok_s:.1f} tok/s), not the final repeated timing row "
            f"({b8_tok_s:.1f} tok/s, repeated baseline {seq_tok_s:.1f} tok/s)."
        )
    if not _close(ab_speedup, b8_speedup, 0.01):
        notes.append(
            "The 1.737x ablation speedup uses the ablation report's own "
            "434.1 tok/s baseline. It must not be compared against the "
            "492.1 tok/s repeated-timing baseline."
        )
    controlled_summary: dict[str, Any] = {}
    if controlled_ablation_path.exists():
        controlled = _load(controlled_ablation_path)
        controlled_summary = controlled.get("summary", {})
        controlled_seq = controlled_summary.get("seq", {})
        controlled_b8 = controlled_summary.get("b8_pool32_default_continuous", {})
        if controlled_seq and controlled_b8:
            controlled_seq_tps = float(controlled_seq["fields"]["tok_s"]["mean"])
            controlled_b8_tps = float(controlled_b8["fields"]["tok_s"]["mean"])
            controlled_b8_speedup = float(
                controlled_b8["fields"]["speedup_vs_same_run_sequential"]["mean"]
            )
            expected_controlled_speedup = controlled_b8_tps / controlled_seq_tps
            _check(
                _close(controlled_b8_speedup, expected_controlled_speedup, args.tolerance),
                failures,
                "controlled ablation b8 speedup "
                f"{controlled_b8_speedup:.6f} != {expected_controlled_speedup:.6f}",
            )
            notes.append(
                "Controlled ablation b8/pool32/default/continuous is an "
                f"independent repeated run ({controlled_b8_tps:.1f} tok/s, "
                f"same-run baseline {controlled_seq_tps:.1f} tok/s), while the "
                f"locked headline timing remains {b8_tok_s:.1f} tok/s."
            )

    report = {
        "passed": not failures,
        "failures": failures,
        "notes": notes,
        "sources": {
            "repeated_timing": str(repeats_path),
            "ablation": str(ablation_path),
            "controlled_ablation": str(controlled_ablation_path)
            if controlled_ablation_path.exists()
            else "",
        },
        "headline": {
            "sequential_tok_s": seq_tok_s,
            "b8_tok_s": b8_tok_s,
            "speedup": b8_speedup,
            "computed_speedup": expected_b8_speedup,
            "sequential_forwards": seq_forwards,
            "b8_forwards": b8_forwards,
            "forward_reduction_pct": reported_reduction,
            "computed_forward_reduction_pct": expected_reduction,
        },
        "ablation_distinction": {
            "same_config_id": "b8_pool32_default_continuous",
            "ablation_tok_s": ab_tok_s,
            "ablation_baseline_tok_s": ab_seq_tok_s,
            "ablation_speedup": ab_speedup,
            "computed_ablation_speedup": expected_ab_speedup,
        },
        "controlled_ablation": {
            "source": str(controlled_ablation_path) if controlled_ablation_path.exists() else "",
            "present": controlled_ablation_path.exists(),
            "final_config": controlled_summary.get("b8_pool32_default_continuous", {}),
        },
    }
    exact_full_sharded_path = (
        ROOT
        / "analysis"
        / "continuous_batched_pld_final_repeats"
        / "continuous_batched_pld_fp32_eager_throughput_test500_sharded_chunk512_v1"
        / "report.json"
    )
    if exact_full_sharded_path.exists():
        exact_full = _load(exact_full_sharded_path)
        exact_summary = exact_full.get("summary", {})
        exact_seq = exact_summary.get("blazedit_pld_w128_n10_b1", {})
        exact_b8 = exact_summary.get("continuous_batched_pld_w128_n10_b8", {})
        if exact_seq and exact_b8:
            exact_seq_tps = float(exact_seq["tok_s"]["mean"])
            exact_b8_tps = float(exact_b8["tok_s"]["mean"])
            exact_b8_speedup = float(exact_b8["speedup"]["mean"])
            expected_exact_speedup = exact_b8_tps / exact_seq_tps
            _check(
                _close(exact_b8_speedup, expected_exact_speedup, args.tolerance),
                failures,
                "fp32/eager sharded test500 b8 speedup "
                f"{exact_b8_speedup:.6f} != {expected_exact_speedup:.6f}",
            )
            report["fp32_eager_full_sharded"] = {
                "source": str(exact_full_sharded_path),
                "n": exact_full.get("args", {}).get("n"),
                "repeats": exact_full.get("args", {}).get("repeats"),
                "sharded": bool(exact_full.get("sharded")),
                "shard_protocol": exact_full.get("shard_protocol", {}),
                "completed_shards": len(exact_full.get("completed_shards", [])),
                "failed_attempts": len(exact_full.get("failed_attempts", [])),
                "sequential_tok_s": exact_seq_tps,
                "b8_tok_s": exact_b8_tps,
                "speedup": exact_b8_speedup,
                "computed_speedup": expected_exact_speedup,
                "task_matches": float(exact_b8.get("output_match_count", {}).get("mean", 0.0)),
            }
    exact_subset_path = (
        ROOT
        / "analysis"
        / "continuous_batched_pld_final_repeats"
        / "continuous_batched_pld_fp32_eager_throughput_test100_subset_v1"
        / "report.json"
    )
    if exact_subset_path.exists():
        exact_subset = _load(exact_subset_path)
        exact_summary = exact_subset.get("summary", {})
        exact_seq = exact_summary.get("blazedit_pld_w128_n10_b1", {})
        exact_b8 = exact_summary.get("continuous_batched_pld_w128_n10_b8", {})
        if exact_seq and exact_b8:
            exact_seq_tps = float(exact_seq["tok_s"]["mean"])
            exact_b8_tps = float(exact_b8["tok_s"]["mean"])
            exact_b8_speedup = float(exact_b8["speedup"]["mean"])
            expected_exact_speedup = exact_b8_tps / exact_seq_tps
            _check(
                _close(exact_b8_speedup, expected_exact_speedup, args.tolerance),
                failures,
                f"fp32/eager subset b8 speedup {exact_b8_speedup:.6f} != {expected_exact_speedup:.6f}",
            )
            report["fp32_eager_subset"] = {
                "source": str(exact_subset_path),
                "n": exact_subset.get("args", {}).get("n"),
                "repeats": exact_subset.get("args", {}).get("repeats"),
                "sequential_tok_s": exact_seq_tps,
                "b8_tok_s": exact_b8_tps,
                "speedup": exact_b8_speedup,
                "computed_speedup": expected_exact_speedup,
                "task_matches": float(exact_b8.get("output_match_count", {}).get("mean", 0.0)),
            }
    report["passed"] = not failures
    report["failures"] = failures
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    for note in notes:
        print(f"NOTE: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
