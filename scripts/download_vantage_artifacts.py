#!/usr/bin/env python3
"""Download VANTAGE generated artifacts from a Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "faizancodes/vantage-artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("VANTAGE_HF_DATASET", DEFAULT_REPO_ID),
        help="Hugging Face dataset id containing generated VANTAGE artifacts.",
    )
    parser.add_argument(
        "--local-dir",
        default=".",
        help="Repository root where artifact paths should be restored.",
    )
    parser.add_argument(
        "--patterns",
        action="append",
        default=None,
        help="Optional allow-pattern passed to snapshot_download. Repeatable.",
    )
    parser.add_argument("--revision", default=None, help="Optional dataset revision.")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    local_dir = Path(args.local_dir).resolve()
    patterns = args.patterns or [
        "artifacts/**",
        "analysis/**",
        "out/**",
        "data/real_commits/**",
        "data/manifests/**",
        "data/manifests_frozen_audit_raw/**",
        "data/manifests_phase2/**",
        "data/manifests_phase3/**",
        "data/manifests_prompt_injection/**",
        "data/manifests_transpld_ext/**",
        "data/routers/**",
    ]

    print(f"Repo id: {args.repo_id}")
    print(f"Local dir: {local_dir}")
    print("Patterns:")
    for pattern in patterns:
        print(f"  - {pattern}")
    if args.dry_run:
        return 0

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(local_dir),
        allow_patterns=patterns,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
