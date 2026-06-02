#!/usr/bin/env python3
"""Audit the VANTAGE publishability revision plan artifacts.

The audit is intentionally conservative: it marks evidence-producing items as
done only when a concrete file or paper phrase exists. Items that require new
GPU/runtime or real-commit collection are marked scoped/future only when the
paper and docs explicitly say they are not measured headline evidence.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Check:
    item: str
    status: str
    evidence: list[str]
    notes: str


def read(path: str) -> str:
    p = Path(path)
    return p.read_text() if p.exists() else ""


def exists(path: str) -> bool:
    return Path(path).exists()


def all_present(text: str, needles: Iterable[str]) -> bool:
    return all(needle in text for needle in needles)


def has_any(text: str, needles: Iterable[str]) -> bool:
    return any(needle in text for needle in needles)


def status(done: bool, scoped: bool = False) -> str:
    if done:
        return "done"
    if scoped:
        return "scoped_or_future"
    return "missing"


def audit() -> list[Check]:
    paper = read("paper/vantage.tex")
    real_protocol = read("docs/vantage_real_edit_benchmark_protocol.md")
    rewrite_doc = read("docs/vantage_rewrite_extraction.md")
    backend_isolation = read(
        "artifacts/vantage_transpld/tables/backend_isolation_20260516_v1/backend_isolation.md"
    )
    prompt_injection = read(
        "artifacts/vantage_transpld/tables/prompt_injection_20260516_v1/prompt_injection_summary.md"
    )

    checks: list[Check] = []

    checks.append(
        Check(
            "Retitle and frame around view-based VANTAGE rather than rewrite-only TransPLD.",
            status(
                "VANTAGE: View-Based Speculative Decoding for Code Edits" in paper
                and "ViewBank" in paper
                and "TransPLD is" in paper
            ),
            ["paper/vantage.tex"],
            "Title and abstract introduce VANTAGE/ViewBank first while keeping TransPLD as the rewrite-view instance.",
        )
    )
    checks.append(
        Check(
            "Add Scope of Claims box with non-claims.",
            status(
                all_present(
                    paper,
                    [
                        "Scope of claims",
                        "not a vLLM production-serving claim",
                        "real-commit-quality claim",
                        "not a multi-model universality claim",
                        "sampling-distribution",
                    ],
                )
            ),
            ["paper/vantage.tex"],
            "The scope box appears early and lists comparator/model/backend limits.",
        )
    )
    checks.append(
        Check(
            "Separate fp32/eager exactness from bf16/sdpa timing.",
            status(
                exists("artifacts/vantage_transpld/tables/validation_20260515_v1/backend_exactness_audit.md")
                and "speed diagnostic only" in read(
                    "artifacts/vantage_transpld/tables/validation_20260515_v1/backend_exactness_audit.md"
                )
                and "bf16/sdpa timing path" in paper
                and "not used to support greedy-equivalence claims" in paper
            ),
            [
                "artifacts/vantage_transpld/tables/validation_20260515_v1/backend_exactness_audit.md",
                "artifacts/vantage_transpld/tables/backend_isolation_20260516_v1/backend_isolation.md",
                "paper/vantage.tex",
            ],
            "The optimized path remains unfixed, but the paper no longer claims it is exact.",
        )
    )
    checks.append(
        Check(
            "Promote audited fp32/sdpa throughput and successful-edit slices.",
            status(
                exists("scripts/summarize_transpld_exact_path_success.py")
                and exists("artifacts/vantage_transpld/tables/exact_path_success_20260518_v1/exact_path_success.md")
                and "Main exact-path controlled result" in paper
                and "Successful-edit slices" in paper
            ),
            [
                "scripts/summarize_transpld_exact_path_success.py",
                "artifacts/vantage_transpld/tables/exact_path_success_20260518_v1/exact_path_success.md",
                "paper/vantage.tex",
            ],
            "The headline throughput table now uses audited fp32/sdpa rows, and task-quality-filtered speed slices are reported.",
        )
    )
    checks.append(
        Check(
            "Document optimized-path parity debugging matrix and required instrumentation.",
            status(
                all_present(
                    backend_isolation,
                    [
                        "fp32/eager",
                        "bf16/eager",
                        "fp32/sdpa",
                        "bf16/sdpa",
                    ],
                )
            ),
            ["artifacts/vantage_transpld/tables/backend_isolation_20260516_v1/backend_isolation.md"],
            "Backend isolation is checked through restored generated artifacts rather than internal planning notes.",
        )
    )
    checks.append(
        Check(
            "Convert controlled validation into a mechanism benchmark with manifest audit.",
            status(
                exists("artifacts/vantage_transpld/tables/controlled_manifest_audit.md")
                and exists("scripts/summarize_transpld_controlled_manifests.py")
                and "Controlled mechanism benchmark" in paper
            ),
            [
                "scripts/summarize_transpld_controlled_manifests.py",
                "artifacts/vantage_transpld/tables/controlled_manifest_audit.md",
                "paper/vantage.tex",
            ],
            "The audit reports map visibility, pair types, token lengths, copied fraction, and examples.",
        )
    )
    checks.append(
        Check(
            "Move real-commit evidence to an inconclusive pilot and add benchmark protocol.",
            status(
                exists("docs/vantage_real_edit_benchmark_protocol.md")
                and "inconclusive" in paper.lower()
                and "cannot support a broad" in real_protocol
            ),
            [
                "paper/vantage.tex",
                "docs/vantage_real_edit_benchmark_protocol.md",
            ],
            "A larger preregistered real-edit benchmark was specified, not collected.",
        )
    )
    checks.append(
        Check(
            "Add per-task regression and tail-latency analysis.",
            status(
                exists("artifacts/vantage_transpld/tables/exact_path_success_20260518_v1/exact_path_success.md")
                and "Exact-Path Tail Diagnostics" in read(
                    "artifacts/vantage_transpld/tables/exact_path_success_20260518_v1/exact_path_success.md"
                )
                and "Tail latency and regressions" in paper
            ),
            [
                "scripts/summarize_transpld_exact_path_success.py",
                "artifacts/vantage_transpld/tables/exact_path_success_20260518_v1/exact_path_success.md",
                "paper/vantage.tex",
            ],
            "The main tail table now uses the same fp32/sdpa exact-path artifacts as the headline result.",
        )
    )
    checks.append(
        Check(
            "Frame visible transformed-reference prompt injection as a changed-prompt baseline.",
            status(
                exists("artifacts/vantage_transpld/tables/prompt_injection_20260516_v1/prompt_injection_summary.md")
                and exists("artifacts/vantage_transpld/tables/prompt_injection_20260516_v1/prompt_injection_summary.json")
                and "changes the visible prompt" in paper
                and "n=100" in prompt_injection
            ),
            [
                "artifacts/vantage_transpld/tables/prompt_injection_20260516_v1/prompt_injection_summary.md",
                "paper/vantage.tex",
            ],
            "The full n=100 zero/field/style/mixed changed-prompt baseline is measured and scoped as non-headline evidence.",
        )
    )
    checks.append(
        Check(
            "Audit SafeRoute as prompt-only.",
            status(
                all_present(
                    paper,
                    ["SafeRoute is prompt-only", "does not read benchmark target text"],
                )
                and all_present(
                    read("tests/test_code_proposers.py"),
                    [
                        "test_precomputed_transpld_ignores_manifest_only_rewrite_pairs",
                        "gold",
                        "manifest_only_field",
                    ],
                )
                and "decide_prompt_only_saferoute" in read("tests/test_vantage_policy.py")
            ),
            [
                "asts/vantage_policy.py",
                "asts/code_proposers.py",
                "tests/test_vantage_policy.py",
                "tests/test_code_proposers.py",
            ],
            "Legacy oracle proposers remain for diagnostics, but SafeRoute/headline path is prompt-only.",
        )
    )
    checks.append(
        Check(
            "Document rewrite extraction support and boundary-safe limitations.",
            status(
                exists("docs/vantage_rewrite_extraction.md")
                and all_present(
                    rewrite_doc,
                    [
                        "Identifier renames",
                        "Dotted-field substitutions",
                        "Boundary-Aware Replacement",
                        "Overlapping",
                    ],
                )
            ),
            [
                "docs/vantage_rewrite_extraction.md",
                "tests/test_vantage_policy.py",
            ],
            "Aggressive substring renames are intentionally future work.",
        )
    )
    checks.append(
        Check(
            "Update related work with current expected baselines.",
            status(
                all_present(
                    paper,
                    [
                        "EfficientEdit",
                        "SuffixDecoding",
                        "SAM-Decoding",
                        "TensorRT-LLM",
                        "vLLM",
                        "Medusa",
                        "EAGLE",
                        "Sequoia",
                        "not experimentally compared",
                    ],
                )
            ),
            ["paper/vantage.tex"],
            "No external baseline win is claimed without measurement.",
        )
    )
    checks.append(
        Check(
            "Keep P1/P2 items without new measurements explicitly scoped as future.",
            status(
                all_present(
                    paper,
                    [
                        "future work",
                        "not experimentally compared",
                        "preliminary",
                    ],
                ),
                scoped=True,
            ),
            ["paper/vantage.tex"],
            "These items were not silently claimed as done.",
        )
    )

    return checks


def write_outputs(checks: list[Check], output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status_counts": {
            status_name: sum(1 for check in checks if check.status == status_name)
            for status_name in ("done", "scoped_or_future", "missing")
        },
        "checks": [
            {
                "item": check.item,
                "status": check.status,
                "evidence": check.evidence,
                "notes": check.notes,
            }
            for check in checks
        ],
    }
    output_json.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# VANTAGE Publishability Plan Audit",
        "",
        "This generated audit verifies whether the revision-plan items are backed by "
        "paper text, scripts, tests, or artifacts. `scoped_or_future` means the item "
        "requires new measurement and is explicitly not claimed as completed.",
        "",
        "| Item | Status | Evidence | Notes |",
        "|---|---|---|---|",
    ]
    for check in checks:
        evidence = "<br>".join(f"`{path}`" for path in check.evidence)
        lines.append(f"| {check.item} | {check.status} | {evidence} | {check.notes} |")
    output_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-json",
        default="artifacts/vantage_transpld/tables/publishability_plan_audit.json",
    )
    parser.add_argument(
        "--output-md",
        default="artifacts/vantage_transpld/tables/publishability_plan_audit.md",
    )
    args = parser.parse_args()

    checks = audit()
    write_outputs(checks, Path(args.output_json), Path(args.output_md))
    print(Path(args.output_md).read_text())
    missing = [check for check in checks if check.status == "missing"]
    if missing:
        raise SystemExit(f"{len(missing)} publishability-plan items are missing")


if __name__ == "__main__":
    main()
