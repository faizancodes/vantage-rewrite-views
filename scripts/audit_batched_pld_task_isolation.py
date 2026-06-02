#!/usr/bin/env python3
"""Audit continuous-batched PLD task-isolation traces."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _is_prefix(prefix: list[int], full: list[int]) -> bool:
    return len(prefix) <= len(full) and full[: len(prefix)] == prefix


def _emitted_tokens_are_verified(event: dict[str, Any]) -> bool:
    drafts = [int(x) for x in event.get("draft_tokens", [])]
    emitted = [int(x) for x in event.get("emitted_tokens", [])]
    verifier = [int(x) for x in event.get("verifier_output_tokens", [])]
    accepted_drafts = int(event.get("accepted_drafts", 0))
    if not emitted:
        return True
    if not _is_prefix(emitted, [int(x) for x in event.get("accepted_tokens", [])]):
        return False
    for i, tok in enumerate(emitted):
        if i < accepted_drafts:
            if i >= len(drafts) or i >= len(verifier):
                return False
            if tok != drafts[i] or tok != verifier[i]:
                return False
        else:
            if i >= len(verifier):
                return False
            if tok != verifier[i]:
                return False
    return True


def audit_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    finished: set[str] = set()
    tasks: set[str] = set()
    emitted_tokens = 0
    verify_events = 0
    by_batch: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for line_no, event in enumerate(events, start=1):
        kind = event.get("event")
        task_id = str(event.get("task_id", ""))
        if task_id:
            tasks.add(task_id)
        if kind == "task_finish":
            if task_id in finished:
                violations.append(
                    {
                        "line": line_no,
                        "type": "duplicate_task_finish",
                        "task_id": task_id,
                    }
                )
            finished.add(task_id)
            continue
        if kind != "verify_scatter":
            continue
        verify_events += 1
        emitted = [int(x) for x in event.get("emitted_tokens", [])]
        emitted_tokens += len(emitted)
        batch_id = int(event.get("verifier_batch_id", -1))
        by_batch[batch_id].append(event)

        if task_id in finished:
            violations.append(
                {
                    "line": line_no,
                    "type": "finished_task_reentered_verifier",
                    "task_id": task_id,
                }
            )
        if event.get("finished_flag_before"):
            violations.append(
                {
                    "line": line_no,
                    "type": "finished_flag_before_verify",
                    "task_id": task_id,
                }
            )
        prefix_before = int(event.get("prefix_len_before", -1))
        prefix_after = int(event.get("prefix_len_after", -1))
        if prefix_after != prefix_before + len(emitted):
            violations.append(
                {
                    "line": line_no,
                    "type": "prefix_length_mismatch",
                    "task_id": task_id,
                    "prefix_before": prefix_before,
                    "prefix_after": prefix_after,
                    "emitted_len": len(emitted),
                }
            )
        cache_after = int(event.get("cache_len_after", -1))
        if cache_after != max(0, prefix_after - 1):
            violations.append(
                {
                    "line": line_no,
                    "type": "cache_length_mismatch",
                    "task_id": task_id,
                    "cache_after": cache_after,
                    "expected": max(0, prefix_after - 1),
                }
            )
        handle = str(event.get("kv_cache_task_id_or_cache_handle", ""))
        if handle and not handle.startswith(f"{task_id}:"):
            violations.append(
                {
                    "line": line_no,
                    "type": "cache_owner_mismatch",
                    "task_id": task_id,
                    "handle": handle,
                }
            )
        if not _emitted_tokens_are_verified(event):
            violations.append(
                {
                    "line": line_no,
                    "type": "unverified_emitted_token",
                    "task_id": task_id,
                    "emitted_tokens": emitted,
                    "draft_tokens": event.get("draft_tokens", []),
                    "verifier_output_tokens": event.get("verifier_output_tokens", []),
                    "accepted_drafts": event.get("accepted_drafts"),
                }
            )

    for batch_id, rows in by_batch.items():
        slots = [int(r.get("batch_slot", -1)) for r in rows]
        task_ids = [str(r.get("task_id", "")) for r in rows]
        if len(slots) != len(set(slots)):
            violations.append(
                {
                    "type": "duplicate_batch_slot",
                    "verifier_batch_id": batch_id,
                    "slots": slots,
                }
            )
        if len(task_ids) != len(set(task_ids)):
            violations.append(
                {
                    "type": "duplicate_task_in_verifier_batch",
                    "verifier_batch_id": batch_id,
                    "task_ids": task_ids,
                }
            )

    return {
        "tasks_audited": len(tasks),
        "emitted_tokens_audited": emitted_tokens,
        "verify_events_audited": verify_events,
        "verifier_batches_audited": len(by_batch),
        "task_mixing_violations": sum(1 for v in violations if "task" in v["type"] or "slot" in v["type"]),
        "cache_ownership_violations": sum(1 for v in violations if "cache" in v["type"]),
        "finished_task_violations": sum(1 for v in violations if "finished" in v["type"]),
        "scatter_gather_mismatches": sum(1 for v in violations if "mismatch" in v["type"]),
        "unverified_token_violations": sum(1 for v in violations if v["type"] == "unverified_emitted_token"),
        "violation_count": len(violations),
        "violations": violations[:100],
        "passed": len(violations) == 0,
    }


def load_events(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = audit_events(load_events(Path(args.trace)))
    report["trace"] = args.trace
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Batched PLD Task-Isolation Audit",
        "",
        f"trace: `{args.trace}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| tasks audited | {report['tasks_audited']} |",
        f"| emitted tokens audited | {report['emitted_tokens_audited']} |",
        f"| verifier events audited | {report['verify_events_audited']} |",
        f"| verifier batches audited | {report['verifier_batches_audited']} |",
        f"| task-mixing violations | {report['task_mixing_violations']} |",
        f"| cache ownership violations | {report['cache_ownership_violations']} |",
        f"| finished-task violations | {report['finished_task_violations']} |",
        f"| scatter/gather mismatches | {report['scatter_gather_mismatches']} |",
        f"| unverified-token violations | {report['unverified_token_violations']} |",
        f"| total violations | {report['violation_count']} |",
        "",
        f"Pass: `{report['passed']}`",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print((out / "report.md").read_text())
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
