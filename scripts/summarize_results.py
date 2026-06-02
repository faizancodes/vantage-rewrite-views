#!/usr/bin/env python3
"""Summarize archived continuous-batched PLD prototype result artifacts without inventing data.

This script is a lightweight provenance index for reproducibility. It reads
the checked JSON artifacts that already exist, reports which required evidence
is present, and marks missing experiments explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "results_summary.json"


def _load_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _exists(path: Path) -> bool:
    return path.exists()


def build_summary() -> dict[str, Any]:
    final_repeats_path = ROOT / "analysis" / "continuous_batched_pld_final_repeats" / "report.json"
    exact_path = (
        ROOT
        / "analysis"
        / "continuous_batched_pld_final_repeats"
        / "continuous_batched_pld_fp32_eager_throughput_test500_v1"
        / "report.json"
    )
    exact_subset_path = (
        ROOT
        / "analysis"
        / "continuous_batched_pld_final_repeats"
        / "continuous_batched_pld_fp32_eager_throughput_test100_subset_v1"
        / "report.json"
    )
    exact_sharded_path = (
        ROOT
        / "analysis"
        / "continuous_batched_pld_final_repeats"
        / "continuous_batched_pld_fp32_eager_throughput_test500_sharded_chunk512_v1"
        / "report.json"
    )
    controlled_ablation_path = (
        ROOT
        / "analysis"
        / "batched_pld_controlled_ablation"
        / "controlled_ablation_test500_v1"
        / "summary.json"
    )
    final_repeats = _load_optional(final_repeats_path)
    headline: dict[str, Any] = {}
    if final_repeats:
        summary = final_repeats.get("summary", {})
        seq = summary.get("blazedit_pld_w128_n10_b1", {})
        b8 = summary.get("continuous_batched_pld_w128_n10_b8", {})
        headline = {
            "source": str(final_repeats_path),
            "backend": "bf16/SDPA",
            "sequential_tok_s_mean": seq.get("tok_s", {}).get("mean"),
            "sequential_tok_s_std": seq.get("tok_s", {}).get("std"),
            "sequential_verifier_forwards": seq.get("verifier_forwards", {}).get("mean"),
            "b8_tok_s_mean": b8.get("tok_s", {}).get("mean"),
            "b8_tok_s_std": b8.get("tok_s", {}).get("std"),
            "b8_speedup_mean": b8.get("speedup", {}).get("mean"),
            "b8_verifier_forwards": b8.get("verifier_forwards", {}).get("mean"),
            "b8_task_matches": b8.get("output_match_count", {}).get("mean"),
            "byte_exact_claim": False,
        }

    required_evidence = {
        "bf16_sdpa_repeated_timing": {
            "present": _exists(final_repeats_path),
            "path": str(final_repeats_path),
        },
        "fp32_eager_exact_throughput": {
            "present": _exists(exact_path),
            "path": str(exact_path),
        },
        "fp32_eager_exact_throughput_sharded_test500": {
            "present": _exists(exact_sharded_path),
            "path": str(exact_sharded_path),
        },
        "fp32_eager_exact_throughput_subset": {
            "present": _exists(exact_subset_path),
            "path": str(exact_subset_path),
        },
        "controlled_repeated_ablation": {
            "present": _exists(controlled_ablation_path),
            "path": str(controlled_ablation_path),
        },
        "sharded_fp32_eager_correctness": {
            "present": _exists(ROOT / "analysis" / "continuous_batched_pld_correctness_sharded" / "report.json"),
            "path": str(ROOT / "analysis" / "continuous_batched_pld_correctness_sharded" / "report.json"),
        },
        "task_isolation_audit": {
            "present": _exists(ROOT / "analysis" / "continuous_batched_pld_task_audit_test500" / "report.json"),
            "path": str(ROOT / "analysis" / "continuous_batched_pld_task_audit_test500" / "report.json"),
        },
        "dataset_leakage_audit": {
            "present": _exists(ROOT / "artifacts" / "dataset_leakage_audit.json"),
            "path": str(ROOT / "artifacts" / "dataset_leakage_audit.json"),
        },
        "external_baseline_attempt_artifacts": {
            "present": _exists(ROOT / "artifacts" / "external_baselines"),
            "path": str(ROOT / "artifacts" / "external_baselines"),
        },
        "successful_vllm_or_hf_external_baselines": {
            "present": _exists(ROOT / "analysis" / "external_baselines"),
            "path": str(ROOT / "analysis" / "external_baselines"),
        },
    }
    return {
        "headline": headline,
        "required_evidence": required_evidence,
        "missing_required_evidence": [
            key for key, value in required_evidence.items() if not value["present"]
        ],
        "notes": [
            "This script summarizes artifact presence and headline arithmetic sources only.",
            "Missing evidence must be generated by running the documented benchmark commands; this script never fabricates values.",
        ],
    }


def _write_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# archived continuous-batched PLD prototype Results Summary",
        "",
        "## Headline",
        "",
    ]
    if report["headline"]:
        for key, value in report["headline"].items():
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- no final repeated timing artifact found")
    lines += [
        "",
        "## Evidence Inventory",
        "",
        "| Evidence | Present | Path |",
        "|---|---|---|",
    ]
    for key, value in report["required_evidence"].items():
        lines.append(f"| {key} | {value['present']} | `{value['path']}` |")
    lines += ["", "## Missing Required Evidence"]
    if report["missing_required_evidence"]:
        lines.extend(f"- {item}" for item in report["missing_required_evidence"])
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    report = build_summary()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_md(report, out.with_suffix(".md"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
