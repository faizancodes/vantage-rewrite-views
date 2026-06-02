#!/usr/bin/env python3
"""Summarize TransPLD proposal quality by transformed-view match length."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


TRANSPLD_MATCH_KINDS = {
    "transpld_vref",
    "routed_transpld_vref",
    "transpld_bidir",
    "routed_transpld_bidir",
    "transpld_bidir_inferred",
}


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _bucket(match_len: int | None) -> str:
    value = int(match_len or 0)
    if value <= 0:
        return "0"
    if value <= 4:
        return str(value)
    if value <= 7:
        return "5-7"
    return "8+"


def analyze(steps: list[dict], *, methods: set[str] | None = None) -> dict:
    buckets = ["1", "2", "3", "4", "5-7", "8+"]
    counts: Counter[str] = Counter()
    zero: Counter[str] = Counter()
    accepted: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    first_token_reject: Counter[str] = Counter()
    later_reject: Counter[str] = Counter()
    full_accept: Counter[str] = Counter()
    examples: dict[str, list[dict]] = defaultdict(list)

    for row in steps:
        if methods and row.get("method") not in methods:
            continue
        if row.get("proposal_match_kind") not in TRANSPLD_MATCH_KINDS:
            continue
        bucket = _bucket(row.get("proposal_match_len"))
        if bucket == "0":
            continue
        counts[bucket] += 1
        acc = int(row.get("n_accepted_nonroot_drafts") or 0)
        accepted[bucket] += acc
        if acc == 0:
            zero[bucket] += 1
        if row.get("rejected"):
            rejected[bucket] += 1
            if row.get("proposal_target_reject_index") == 0:
                first_token_reject[bucket] += 1
            else:
                later_reject[bucket] += 1
        else:
            full_accept[bucket] += 1
        if len(examples[bucket]) < 5 and row.get("rejected"):
            examples[bucket].append(
                {
                    "task_id": row.get("task_id"),
                    "step": row.get("step"),
                    "method": row.get("method"),
                    "match_kind": row.get("proposal_match_kind"),
                    "match_len": row.get("proposal_match_len"),
                    "accepted_nonroot": acc,
                    "draft0": row.get("proposal_first_token_text"),
                    "target_reject": row.get("proposal_target_reject_token_text"),
                    "reject_index": row.get("proposal_target_reject_index"),
                    "preview": row.get("proposal_text_preview"),
                }
            )

    total = sum(counts.values())
    bucket_rows = []
    for bucket in buckets:
        n = counts[bucket]
        bucket_rows.append(
            {
                "bucket": bucket,
                "attempts": n,
                "share": n / total if total else 0.0,
                "accepted_nonroot": accepted[bucket],
                "accepted_per_attempt": accepted[bucket] / n if n else 0.0,
                "zero_accept_rate": zero[bucket] / n if n else 0.0,
                "rejection_rate": rejected[bucket] / n if n else 0.0,
                "first_token_reject_rate": first_token_reject[bucket] / n if n else 0.0,
                "full_accept_rate": full_accept[bucket] / n if n else 0.0,
            }
        )
    return {
        "total_attempts": total,
        "buckets": bucket_rows,
        "examples": examples,
    }


def write_md(report: dict, path: Path, *, label: str) -> None:
    with path.open("w") as f:
        f.write(f"# TransPLD Match Quality: {label}\n\n")
        f.write(f"Total transformed-view attempts: `{report['total_attempts']}`\n\n")
        f.write(
            "| match_len bucket | attempts | share | accepted/attempt | zero-accept | first-token reject | full accept |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in report["buckets"]:
            f.write(
                f"| {row['bucket']} | {row['attempts']} | {row['share']:.1%} | "
                f"{row['accepted_per_attempt']:.2f} | {row['zero_accept_rate']:.1%} | "
                f"{row['first_token_reject_rate']:.1%} | {row['full_accept_rate']:.1%} |\n"
            )
        f.write("\n## Rejection Examples\n\n")
        for bucket, rows in report["examples"].items():
            f.write(f"### match_len {bucket}\n\n")
            for row in rows:
                f.write(
                    f"- `{row['task_id']}` step {row['step']}: acc={row['accepted_nonroot']}, "
                    f"draft0=`{row['draft0']}`, target=`{row['target_reject']}`, "
                    f"reject_idx={row['reject_index']}, preview=`{str(row['preview'])[:80]}`\n"
                )
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", required=True)
    parser.add_argument("--methods", default="")
    parser.add_argument("--label", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    methods = {m.strip() for m in args.methods.split(",") if m.strip()} or None
    report = analyze(_load_jsonl(Path(args.steps)), methods=methods)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2))
    write_md(report, Path(args.output_md), label=args.label or Path(args.steps).parent.name)
    print(Path(args.output_md).read_text())


if __name__ == "__main__":
    main()
