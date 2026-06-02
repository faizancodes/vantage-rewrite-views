#!/usr/bin/env python3
"""Select a balanced final manifest from verified real-commit rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--target-rows", type=int, default=1000)
    p.add_argument(
        "--family-quota",
        default="real_rename=500,real_field_migration=500",
        help="Comma-separated family=count quotas. Remaining rows are filled round-robin.",
    )
    p.add_argument("--max-per-repo", type=int, default=90)
    p.add_argument(
        "--unique-commits",
        action="store_true",
        help="Allow at most one row per repo/commit_sha.",
    )
    args = p.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.input_jsonl).read_text().splitlines()
        if line.strip()
    ]
    quotas: dict[str, int] = {}
    for item in args.family_quota.split(","):
        if not item.strip():
            continue
        family, count = item.split("=", 1)
        quotas[family.strip()] = int(count)

    by_family_repo: dict[str, dict[str, deque[dict[str, Any]]]] = defaultdict(lambda: defaultdict(deque))
    for row in rows:
        by_family_repo[str(row.get("drift_family"))][str(row.get("repo"))].append(row)

    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    selected_commits: set[tuple[str, str]] = set()
    repo_counts: Counter[str] = Counter()

    def add_row(row: dict[str, Any]) -> bool:
        key = str(row["task_id"])
        repo = str(row.get("repo"))
        commit_key = (repo, str(row.get("commit_sha")))
        if key in selected_keys or repo_counts[repo] >= args.max_per_repo:
            return False
        if args.unique_commits and commit_key in selected_commits:
            return False
        selected_keys.add(key)
        selected_commits.add(commit_key)
        repo_counts[repo] += 1
        row = dict(row)
        row["task_id"] = f"real_commit_python/{len(selected):04d}"
        selected.append(row)
        return True

    def take_family(family: str, count: int) -> None:
        repo_names = deque(sorted(by_family_repo.get(family, {})))
        while repo_names and len([r for r in selected if r.get("drift_family") == family]) < count:
            repo = repo_names.popleft()
            bucket = by_family_repo[family][repo]
            while bucket and not add_row(bucket.popleft()):
                pass
            if bucket:
                repo_names.append(repo)

    for family, count in quotas.items():
        take_family(family, count)

    # Fill any remaining target rows without a family preference.
    all_buckets: dict[tuple[str, str], deque[dict[str, Any]]] = {}
    for family, by_repo in by_family_repo.items():
        for repo, bucket in by_repo.items():
            all_buckets[(family, repo)] = bucket
    active = deque(sorted(all_buckets))
    while len(selected) < args.target_rows and active:
        key = active.popleft()
        bucket = all_buckets[key]
        while bucket and not add_row(bucket.popleft()):
            pass
        if bucket:
            active.append(key)

    if len(selected) != args.target_rows:
        raise SystemExit(f"selected {len(selected)} rows, expected {args.target_rows}")

    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for row in selected:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"wrote {len(selected)} rows to {output}")
    print("families", dict(Counter(r["drift_family"] for r in selected)))
    print("repos", dict(repo_counts.most_common()))


if __name__ == "__main__":
    main()
