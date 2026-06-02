#!/usr/bin/env python3
"""Prepare runnable commands for repeated VANTAGE fp32/sdpa timing.

This script does not run the model. It writes a versioned artifact directory
with a run configuration, README, and shell script that can be executed on a
GPU host with the same dependencies as the headline validation harness.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = {
    "zero": "data/manifests_frozen_audit/zero_drift100.jsonl",
    "field": "data/manifests_frozen_audit/field_rename100.jsonl",
    "identifier_style": "data/manifests_frozen_audit/style_rewrite100.jsonl",
    "mixed": "data/manifests_frozen_audit/mixed_zero_field_style100.jsonl",
}
METHOD_ORDERS = [
    "vanilla,blazedit_pld_w128_n10,vantage_frozen_transpld",
    "blazedit_pld_w128_n10,vantage_frozen_transpld,vanilla",
    "vantage_frozen_transpld,vanilla,blazedit_pld_w128_n10",
]


def _command(output_dir: Path, manifest: str, methods: str) -> str:
    return " ".join(
        [
            "python3",
            "scripts/run_eagle_eval.py",
            "--output-dir",
            str(output_dir),
            "--target",
            "Qwen/Qwen2.5-Coder-7B",
            "--eagle-checkpoint",
            "/data/eagle_v1_normfix/eagle/eagle_final.pt",
            "--n",
            "100",
            "--max-new-tokens",
            "256",
            "--methods",
            methods,
            "--dtype",
            "fp32",
            "--attn-impl",
            "sdpa",
            "--problem-jsonl",
            manifest,
            "--skip-eagle-load",
            "--code-proposer-fallback",
            "root",
            "--transpld-min-match-len",
            "4",
            "--chat-template",
            "none",
            "--log-level",
            "INFO",
        ]
    )


def main() -> int:
    default_dir = (
        ROOT
        / "artifacts"
        / "vantage_transpld"
        / "tables"
        / f"repeated_fp32_sdpa_{date.today():%Y%m%d}_v1"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.repeats < 3:
        raise SystemExit("--repeats must be at least 3 for the planned experiment")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands: list[dict[str, str | int]] = []
    for rep in range(1, args.repeats + 1):
        methods = METHOD_ORDERS[(rep - 1) % len(METHOD_ORDERS)]
        for workload, manifest in WORKLOADS.items():
            out = args.output_dir / "raw" / f"repeat_{rep:02d}" / workload / "eval"
            commands.append(
                {
                    "repeat": rep,
                    "workload": workload,
                    "manifest": manifest,
                    "methods": methods,
                    "command": _command(out, manifest, methods),
                }
            )

    run_config = {
        "schema": "vantage/repeated_fp32_sdpa_plan/v1",
        "status": "not_run_plan_only",
        "model": "Qwen/Qwen2.5-Coder-7B",
        "requested_revision": "0396a76181e127dfc13e5c5ec48a8cee09938b02",
        "revision_note": (
            "scripts/run_eagle_eval.py does not expose a revision flag; pin the "
            "HF cache/snapshot externally if exact revision enforcement is needed."
        ),
        "dtype": "fp32",
        "attn_impl": "sdpa",
        "max_new_tokens": 256,
        "eos_token_id": 151643,
        "chat_template": "none",
        "repeats": args.repeats,
        "workloads": WORKLOADS,
        "commands": commands,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd \"$(dirname \"$0\")/../../../..\"",
        "",
        "# Execute on a GPU host with the VANTAGE/ASTS environment installed.",
    ]
    for item in commands:
        lines.append("")
        lines.append(f"echo 'repeat={item['repeat']} workload={item['workload']}'")
        lines.append(str(item["command"]))
    (args.output_dir / "commands.sh").write_text("\n".join(lines) + "\n")

    readme = [
        "# Repeated fp32/sdpa Timing Plan",
        "",
        "Status: not run. This directory contains the exact local GPU commands",
        "needed to run the repeated timing validation; it contains no timing",
        "results and must not be cited as evidence.",
        "",
        "Acceptance gate: PLD and VANTAGE/SafeRoute must match vanilla greedy on",
        "100/100 tasks for every workload and repeat before any timing result is",
        "used in the paper.",
        "",
        "Run from this repository root:",
        "",
        f"```bash\nbash {args.output_dir / 'commands.sh'}\n```",
        "",
        "After execution, summarize per-repeat tok/s, parity, route counts, verifier",
        "steps, rewrite-view hits, and accepted rewrite-view tokens before adding",
        "any paper result.",
        "",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme))
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
