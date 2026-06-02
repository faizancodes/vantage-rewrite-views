#!/usr/bin/env python3
"""Sharded deterministic correctness validation for Continuous Batched PLD.

The full fp32/eager validation can exceed L40S memory on long-context held-out
sets when all tasks are resident in one process.  This wrapper preserves the
existing deterministic comparison harness but runs the held-out manifest in
small independent shards, then aggregates exact-match counts across all tasks.
If a shard still OOMs, it is split into smaller shards instead of silently
skipping tasks.  The reviewer-facing plan is 50 -> 25 -> 10; pathological
long-context shards continue to 5 and then 1 so every requested task is either
validated or the run fails loudly.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


VALIDATE_SCRIPT = ROOT / "scripts" / "validate_batched_pld_correctness.py"


@dataclass
class ShardAttempt:
    shard_id: str
    start: int
    end: int
    attempted_size: int
    output_dir: str
    returncode: int
    oom: bool
    command: list[str] = field(default_factory=list)
    stderr_tail: str = ""


@dataclass
class CompletedShard:
    shard_id: str
    start: int
    end: int
    shard_size: int
    output_dir: str
    task_ids: list[str]
    report: dict[str, Any]
    fallback_from: int | None = None


def read_jsonl_rows(path: str | Path, n: int | None) -> list[str]:
    rows: list[str] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(line if line.endswith("\n") else line + "\n")
            if n is not None and len(rows) >= n:
                break
    return rows


def task_id_from_row(line: str, fallback: str) -> str:
    try:
        row = json.loads(line)
    except Exception:
        return fallback
    return str(row.get("task_id") or fallback)


def split_ranges(total: int, shard_size: int) -> list[tuple[int, int]]:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    return [(start, min(total, start + shard_size)) for start in range(0, total, shard_size)]


def next_fallback_size(current_size: int) -> int | None:
    if current_size > 25:
        return 25
    if current_size > 10:
        return 10
    if current_size > 5:
        return 5
    if current_size > 1:
        return 1
    return None


def is_oom_failure(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "out of memory",
            "cuda oom",
            "cuda error: out of memory",
            "torch.outofmemoryerror",
            "cublas_status_alloc_failed",
        )
    )


def write_shard_jsonl(rows: list[str], start: int, end: int, path: Path) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(rows[start:end]), encoding="utf-8")
    return [
        task_id_from_row(line, f"row/{idx}")
        for idx, line in enumerate(rows[start:end], start=start)
    ]


def validate_task_coverage(completed: list[CompletedShard], expected_task_ids: list[str]) -> dict[str, Any]:
    seen: list[str] = []
    for shard in completed:
        seen.extend(shard.task_ids)
    seen_set = set(seen)
    expected_set = set(expected_task_ids)
    duplicates = sorted({x for x in seen if seen.count(x) > 1})
    missing = [x for x in expected_task_ids if x not in seen_set]
    unexpected = sorted(x for x in seen_set if x not in expected_set)
    return {
        "expected_count": len(expected_task_ids),
        "seen_count": len(seen),
        "unique_seen_count": len(seen_set),
        "duplicate_task_ids": duplicates,
        "missing_task_ids": missing,
        "unexpected_task_ids": unexpected,
        "covers_all_tasks_exactly_once": (
            len(seen) == len(expected_task_ids)
            and not duplicates
            and not missing
            and not unexpected
        ),
    }


def aggregate_shard_reports(
    completed: list[CompletedShard],
    *,
    batch_sizes: list[int],
    total_tasks: int,
    initial_shard_size: int,
    oom_attempts: list[ShardAttempt],
    skipped_count: int = 0,
    expected_task_ids: list[str] | None = None,
) -> dict[str, Any]:
    batch_results: dict[str, dict[str, Any]] = {
        str(batch): {
            "batch_size": batch,
            "tasks": 0,
            "exact_token_id_matches": 0,
            "token_id_mismatches": 0,
            "decoded_output_matches": 0,
            "decoded_output_mismatches": 0,
            "finish_reason_matches": 0,
            "finish_reason_mismatches": 0,
            "generated_length_matches": 0,
            "generated_length_mismatches": 0,
            "exact": False,
        }
        for batch in batch_sizes
    }
    mismatch_examples: list[dict[str, Any]] = []
    for shard in completed:
        for row in shard.report.get("rows", []):
            batch = str(row.get("batch_size"))
            if batch not in batch_results:
                continue
            total = int(row.get("matches", 0)) + int(row.get("mismatches", 0))
            out = batch_results[batch]
            out["tasks"] += total
            out["exact_token_id_matches"] += int(row.get("matches", 0))
            out["token_id_mismatches"] += int(row.get("mismatches", 0))
            out["decoded_output_matches"] += int(row.get("decoded_output_matches", 0))
            out["decoded_output_mismatches"] += int(row.get("decoded_output_mismatches", 0))
            finish_mismatches = int(row.get("finish_mismatch_count", 0))
            out["finish_reason_mismatches"] += finish_mismatches
            out["finish_reason_matches"] += max(0, total - finish_mismatches)
            out["generated_length_matches"] += int(row.get("generated_length_matches", 0))
            out["generated_length_mismatches"] += int(row.get("generated_length_mismatches", 0))
            for example in row.get("mismatch_examples", []) or []:
                if len(mismatch_examples) >= 10:
                    break
                enriched = dict(example)
                enriched["shard_id"] = shard.shard_id
                enriched["batch_size"] = int(row.get("batch_size", 0))
                mismatch_examples.append(enriched)
    for out in batch_results.values():
        out["exact"] = (
            out["tasks"] == total_tasks
            and out["exact_token_id_matches"] == total_tasks
            and out["decoded_output_matches"] == total_tasks
            and out["finish_reason_mismatches"] == 0
            and out["generated_length_matches"] == total_tasks
        )
    coverage = (
        validate_task_coverage(completed, expected_task_ids)
        if expected_task_ids is not None
        else {}
    )
    return {
        "total_tasks": total_tasks,
        "initial_shard_size": initial_shard_size,
        "initial_shard_count": len(split_ranges(total_tasks, initial_shard_size)),
        "completed_shard_count": len(completed),
        "completed_shards": [
            {
                "shard_id": s.shard_id,
                "start": s.start,
                "end": s.end,
                "shard_size": s.shard_size,
                "fallback_from": s.fallback_from,
                "output_dir": s.output_dir,
                "task_count": len(s.task_ids),
            }
            for s in completed
        ],
        "batch_results": batch_results,
        "all_exact": all(batch_results[str(batch)]["exact"] for batch in batch_sizes),
        "mismatch_count": sum(
            int(batch_results[str(batch)]["token_id_mismatches"]) for batch in batch_sizes
        ),
        "first_10_mismatch_examples": mismatch_examples,
        "oom_count": len(oom_attempts),
        "oom_attempts": [
            {
                "shard_id": a.shard_id,
                "start": a.start,
                "end": a.end,
                "attempted_size": a.attempted_size,
                "returncode": a.returncode,
                "stderr_tail": a.stderr_tail,
            }
            for a in oom_attempts
        ],
        "skipped_count": skipped_count,
        "coverage": coverage,
    }


def _run_validation_command(
    *,
    args: argparse.Namespace,
    shard_jsonl: Path,
    shard_n: int,
    shard_out_dir: Path,
) -> tuple[int, bool, list[str], str]:
    cmd = [
        sys.executable,
        str(VALIDATE_SCRIPT),
        "--problem-jsonl",
        str(shard_jsonl),
        "--n",
        str(shard_n),
        "--target",
        args.target,
        "--dtype",
        args.dtype,
        "--attn",
        args.attn,
        "--device",
        args.device,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--chat-template",
        args.chat_template,
        "--batch-sizes",
        args.batch_sizes,
        "--bucket-sizes",
        args.bucket_sizes,
        "--bucket-policy",
        args.bucket_policy,
        "--refill-policy",
        args.refill_policy,
        "--active-pool-size",
        str(args.active_pool_size),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(shard_out_dir),
    ]
    if args.write_audit_traces:
        cmd.append("--write-audit-traces")
    env = dict(os.environ)
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    shard_out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    (shard_out_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (shard_out_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    combined = proc.stdout + "\n" + proc.stderr
    return proc.returncode, is_oom_failure(combined), cmd, proc.stderr[-4000:]


def run_sharded_validation(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl_rows(args.problem_jsonl, args.n)
    if not rows:
        raise SystemExit(f"no tasks found in {args.problem_jsonl}")
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    expected_task_ids = [
        task_id_from_row(line, f"row/{idx}") for idx, line in enumerate(rows)
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_input_dir = output_dir / "_shard_inputs"
    completed: list[CompletedShard] = []
    oom_attempts: list[ShardAttempt] = []

    def run_range(start: int, end: int, attempted_size: int, shard_prefix: str, fallback_from: int | None = None) -> None:
        shard_id = f"{shard_prefix}_{start:04d}_{end:04d}"
        shard_out_dir = output_dir / f"shard_{shard_id}"
        shard_jsonl = shard_input_dir / f"shard_{shard_id}.jsonl"
        task_ids = write_shard_jsonl(rows, start, end, shard_jsonl)
        print(
            f"[shard {shard_id}] validating rows {start}:{end} "
            f"(size={end - start}, attempted_size={attempted_size})",
            flush=True,
        )
        returncode, oom, cmd, stderr_tail = _run_validation_command(
            args=args,
            shard_jsonl=shard_jsonl,
            shard_n=end - start,
            shard_out_dir=shard_out_dir,
        )
        if returncode == 0:
            report_path = shard_out_dir / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            completed.append(
                CompletedShard(
                    shard_id=shard_id,
                    start=start,
                    end=end,
                    shard_size=end - start,
                    output_dir=str(shard_out_dir),
                    task_ids=task_ids,
                    report=report,
                    fallback_from=fallback_from,
                )
            )
            return
        attempt = ShardAttempt(
            shard_id=shard_id,
            start=start,
            end=end,
            attempted_size=attempted_size,
            output_dir=str(shard_out_dir),
            returncode=returncode,
            oom=oom,
            command=cmd,
            stderr_tail=stderr_tail,
        )
        if oom:
            oom_attempts.append(attempt)
        fallback_size = next_fallback_size(attempted_size)
        if oom and fallback_size is not None:
            print(
                f"[shard {shard_id}] OOM at size {attempted_size}; retrying with size {fallback_size}",
                flush=True,
            )
            for sub_start, sub_end in split_ranges(end - start, fallback_size):
                run_range(
                    start + sub_start,
                    start + sub_end,
                    fallback_size,
                    f"{shard_prefix}r",
                    fallback_from=attempted_size,
                )
            return
        if oom:
            raise RuntimeError(
                f"Shard {shard_id} OOMed at minimum shard size {attempted_size}; "
                f"stderr tail:\n{stderr_tail}"
            )
        raise RuntimeError(
            f"Shard {shard_id} failed with return code {returncode}; stderr tail:\n{stderr_tail}"
        )

    for ordinal, (start, end) in enumerate(split_ranges(len(rows), args.shard_size)):
        run_range(start, end, args.shard_size, f"{ordinal:03d}")

    aggregate = aggregate_shard_reports(
        completed,
        batch_sizes=batch_sizes,
        total_tasks=len(rows),
        initial_shard_size=args.shard_size,
        oom_attempts=oom_attempts,
        skipped_count=0,
        expected_task_ids=expected_task_ids,
    )
    payload = {
        "args": vars(args),
        "aggregate": aggregate,
        "correctness_note": (
            "This report aggregates independent fp32/eager deterministic shards. "
            "Shard fallback is used only to avoid validation-memory OOM; no task is skipped."
        ),
    }
    write_aggregate_reports(payload, output_dir)
    return payload


def write_aggregate_reports(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    agg = payload["aggregate"]
    lines = [
        "# Sharded Batched PLD Deterministic Correctness",
        "",
        payload["correctness_note"],
        "",
        f"total tasks: `{agg['total_tasks']}`  shard size: `{agg['initial_shard_size']}`  "
        f"initial shards: `{agg['initial_shard_count']}`  completed shards: `{agg['completed_shard_count']}`",
        "",
        "| batch | token-id matches | decoded matches | finish matches | generated-length matches | exact |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for batch in sorted(agg["batch_results"], key=lambda x: int(x)):
        row = agg["batch_results"][batch]
        tasks = int(row["tasks"])
        lines.append(
            f"| {batch} | {row['exact_token_id_matches']}/{tasks} | "
            f"{row['decoded_output_matches']}/{tasks} | "
            f"{row['finish_reason_matches']}/{tasks} | "
            f"{row['generated_length_matches']}/{tasks} | "
            f"{'yes' if row['exact'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"OOM retry attempts: `{agg['oom_count']}`.",
            f"Skipped tasks: `{agg['skipped_count']}`.",
            f"Coverage exact once: `{agg.get('coverage', {}).get('covers_all_tasks_exactly_once')}`.",
            f"Overall exact pass: `{agg['all_exact']}`.",
        ]
    )
    if agg["first_10_mismatch_examples"]:
        lines.extend(["", "## First Mismatches", ""])
        for ex in agg["first_10_mismatch_examples"]:
            lines.append(
                f"- shard `{ex.get('shard_id')}`, batch `{ex.get('batch_size')}`, "
                f"task `{ex.get('task_id')}`, first diff `{ex.get('first_diff_index')}`, "
                f"seq token `{ex.get('baseline_token_id')}`, batched token `{ex.get('batched_token_id')}`"
            )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-jsonl", required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--shard-size", type=int, default=50)
    parser.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--dtype", default="fp32")
    parser.add_argument("--attn", default="eager")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--chat-template", default="none")
    parser.add_argument("--batch-sizes", default="1,4,8")
    parser.add_argument("--active-pool-size", type=int, default=32)
    parser.add_argument("--bucket-sizes", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--bucket-policy", choices=["custom", "default", "fine", "single"], default="default")
    parser.add_argument("--refill-policy", choices=["continuous", "no_refill"], default="continuous")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--write-audit-traces", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    payload = run_sharded_validation(parse_args())
    print((Path(payload["args"]["output_dir"]) / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
