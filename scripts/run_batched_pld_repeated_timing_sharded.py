#!/usr/bin/env python3
"""Shard repeated PLD timing to avoid fp32/eager long-run CUDA OOM.

The single-process fp32/eager test500 timing attempt can OOM on L40S even
though deterministic correctness succeeds in shards.  This wrapper keeps the
same decoder and verifier semantics, but runs independent manifest shards in
subprocesses and aggregates emitted-token throughput across all shards.  Model
load remains outside each shard's timed region, matching the normal timing
harness.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TIMING_SCRIPT = ROOT / "scripts" / "run_batched_pld_repeated_timing.py"


@dataclass
class ShardResult:
    repeat: int
    shard_id: str
    start: int
    end: int
    shard_size: int
    output_dir: Path
    report: dict[str, Any]
    fallback_from: int | None = None


@dataclass
class FailedAttempt:
    repeat: int
    shard_id: str
    start: int
    end: int
    attempted_size: int
    output_dir: Path
    returncode: int
    oom: bool
    stderr_tail: str
    command: list[str]


def _read_jsonl_rows(path: Path, n: int) -> list[str]:
    rows: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(line if line.endswith("\n") else line + "\n")
            if len(rows) >= n:
                break
    return rows


def _task_id(line: str, fallback: str) -> str:
    try:
        row = json.loads(line)
    except Exception:
        return fallback
    return str(row.get("task_id") or fallback)


def _write_jsonl(rows: list[str], start: int, end: int, path: Path) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(rows[start:end]), encoding="utf-8")
    return [_task_id(line, f"row/{idx}") for idx, line in enumerate(rows[start:end], start=start)]


def _split_ranges(total: int, shard_size: int) -> list[tuple[int, int]]:
    return [(s, min(total, s + shard_size)) for s in range(0, total, shard_size)]


def _next_fallback(size: int) -> int | None:
    if size > 50:
        return 50
    if size > 25:
        return 25
    if size > 10:
        return 10
    if size > 5:
        return 5
    if size > 1:
        return 1
    return None


def _is_oom(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "out of memory",
            "cuda oom",
            "torch.outofmemoryerror",
            "cuda error: out of memory",
            "cublas_status_alloc_failed",
        )
    )


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "ci95": 0.0}
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(std),
        "min": float(min(values)),
        "max": float(max(values)),
        "ci95": float(1.96 * std / math.sqrt(len(values))) if len(values) > 1 else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-jsonl", required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--shard-size", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--dtype", default="fp32")
    parser.add_argument("--attn", default="eager")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--chat-template", default="none")
    parser.add_argument("--batch-sizes", default="2,4,8")
    parser.add_argument("--active-pool-size", type=int, default=32)
    parser.add_argument("--bucket-policy", default="default")
    parser.add_argument("--refill-policy", default="continuous")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--memory-hygiene", action="store_true")
    parser.add_argument("--empty-cache-every", type=int, default=1)
    parser.add_argument("--prefill-chunk-size", type=int, default=1024)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _run_one_shard(
    *,
    args: argparse.Namespace,
    rows: list[str],
    repeat: int,
    start: int,
    end: int,
    attempted_size: int,
    fallback_from: int | None,
    failures: list[FailedAttempt],
    completed: list[ShardResult],
) -> None:
    shard_id = f"r{repeat:02d}_{start:04d}_{end:04d}_s{attempted_size}"
    shard_dir = Path(args.output_dir) / "shards" / shard_id
    shard_jsonl = shard_dir / "tasks.jsonl"
    _write_jsonl(rows, start, end, shard_jsonl)
    cmd = [
        sys.executable,
        str(TIMING_SCRIPT),
        "--problem-jsonl",
        str(shard_jsonl),
        "--n",
        str(end - start),
        "--repeats",
        "1",
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
        "--active-pool-size",
        str(args.active_pool_size),
        "--bucket-policy",
        args.bucket_policy,
        "--refill-policy",
        args.refill_policy,
        "--output-dir",
        str(shard_dir),
        "--prefill-chunk-size",
        str(args.prefill_chunk_size),
    ]
    if args.memory_hygiene:
        cmd.extend(["--memory-hygiene", "--empty-cache-every", str(args.empty_cache_every)])
    if args.deterministic:
        cmd.extend(["--deterministic", "--seed", str(args.seed + repeat)])

    if args.dry_run:
        print("$ " + " ".join(cmd), flush=True)
        return

    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if proc.returncode == 0:
        report = json.loads((shard_dir / "report.json").read_text(encoding="utf-8"))
        completed.append(
            ShardResult(
                repeat=repeat,
                shard_id=shard_id,
                start=start,
                end=end,
                shard_size=end - start,
                output_dir=shard_dir,
                report=report,
                fallback_from=fallback_from,
            )
        )
        return

    stderr_tail = (proc.stderr or proc.stdout or "")[-4000:]
    oom = _is_oom(proc.stderr + "\n" + proc.stdout)
    failures.append(
        FailedAttempt(
            repeat=repeat,
            shard_id=shard_id,
            start=start,
            end=end,
            attempted_size=attempted_size,
            output_dir=shard_dir,
            returncode=proc.returncode,
            oom=oom,
            stderr_tail=stderr_tail,
            command=cmd,
        )
    )
    fallback_size = _next_fallback(attempted_size) if oom else None
    if fallback_size is None:
        raise RuntimeError(
            f"shard {shard_id} failed with returncode={proc.returncode}; "
            f"oom={oom}; tail:\n{stderr_tail}"
        )
    print(
        f"OOM in shard {shard_id}; retrying {start}:{end} with shard_size={fallback_size}",
        flush=True,
    )
    for sub_start, sub_end in _split_ranges(end - start, fallback_size):
        _run_one_shard(
            args=args,
            rows=rows,
            repeat=repeat,
            start=start + sub_start,
            end=start + sub_end,
            attempted_size=fallback_size,
            fallback_from=attempted_size,
            failures=failures,
            completed=completed,
        )


def _aggregate_repeat(shards: list[ShardResult], repeat: int, total_tasks: int) -> list[dict[str, Any]]:
    by_key: dict[str, list[dict[str, Any]]] = {}
    for shard in shards:
        if shard.repeat != repeat:
            continue
        for row in shard.report.get("rows", []):
            key = f"{row['method']}_b{row['batch_size']}"
            by_key.setdefault(key, []).append(row)

    seq_key = "blazedit_pld_w128_n10_b1"
    seq_rows = by_key.get(seq_key, [])
    if not seq_rows:
        raise RuntimeError(f"repeat {repeat} missing sequential shard rows")
    seq_tokens = sum(float(r.get("total_generated_tokens", 0.0)) for r in seq_rows)
    seq_wall = sum(float(r.get("wall_ms", 0.0)) for r in seq_rows)
    seq_steps = sum(float(r.get("verifier_forwards", 0.0)) for r in seq_rows)
    seq_tps = seq_tokens / max(1e-9, seq_wall / 1000.0)

    out: list[dict[str, Any]] = []
    for key, rows in sorted(by_key.items()):
        tokens = sum(float(r.get("total_generated_tokens", 0.0)) for r in rows)
        wall = sum(float(r.get("wall_ms", 0.0)) for r in rows)
        forwards = sum(float(r.get("verifier_forwards", 0.0)) for r in rows)
        matches = sum(float(r.get("output_match_count", 0.0)) for r in rows)
        mismatches = sum(float(r.get("output_mismatch_count", 0.0)) for r in rows)
        peak = max(float(r.get("memory_peak_gb", 0.0) or 0.0) for r in rows)
        first = rows[0]
        tps = tokens / max(1e-9, wall / 1000.0)
        out.append(
            {
                "run_id": repeat,
                "method": first["method"],
                "batch_size": int(first["batch_size"]),
                "active_pool_size": int(first.get("active_pool_size", 0) or 0),
                "tok_s": tps,
                "speedup_vs_same_run_sequential": tps / max(1e-9, seq_tps),
                "verifier_forwards": int(forwards),
                "verifier_forward_reduction": 0.0
                if key == seq_key
                else 1.0 - forwards / max(1.0, seq_steps),
                "total_generated_tokens": int(tokens),
                "wall_ms": wall,
                "memory_peak_gb": peak,
                "output_match_count": int(matches),
                "output_mismatch_count": int(mismatches),
                "shard_count": len(rows),
            }
        )
    return out


def _aggregate(completed: list[ShardResult], failures: list[FailedAttempt], args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        rows.extend(_aggregate_repeat(completed, repeat, args.n))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(f"{row['method']}_b{row['batch_size']}", []).append(row)
    summary: dict[str, dict[str, Any]] = {}
    for key, vals in grouped.items():
        summary[key] = {
            "method": vals[0]["method"],
            "batch_size": vals[0]["batch_size"],
            "active_pool_size": vals[0].get("active_pool_size", 0),
            "tok_s": _summary([float(v["tok_s"]) for v in vals]),
            "speedup": _summary([float(v["speedup_vs_same_run_sequential"]) for v in vals]),
            "verifier_forwards": _summary([float(v["verifier_forwards"]) for v in vals]),
            "verifier_forward_reduction": _summary(
                [100.0 * float(v["verifier_forward_reduction"]) for v in vals]
            ),
            "memory_peak_gb": _summary([float(v.get("memory_peak_gb", 0.0)) for v in vals]),
            "output_match_count": _summary([float(v.get("output_match_count", 0)) for v in vals]),
        }
    return {
        "args": vars(args),
        "rows": rows,
        "summary": summary,
        "sharded": True,
        "shard_protocol": {
            "initial_shard_size": args.shard_size,
            "fallback_policy": [50, 25, 10, 5, 1],
            "timing_note": (
                "Throughput is aggregated by summing emitted tokens and generation wall_ms "
                "across independent shards. Model load remains outside each shard's timed region."
            ),
        },
        "completed_shards": [
            {
                "repeat": s.repeat,
                "shard_id": s.shard_id,
                "start": s.start,
                "end": s.end,
                "shard_size": s.shard_size,
                "fallback_from": s.fallback_from,
                "output_dir": str(s.output_dir),
            }
            for s in completed
        ],
        "failed_attempts": [
            {
                "repeat": f.repeat,
                "shard_id": f.shard_id,
                "start": f.start,
                "end": f.end,
                "attempted_size": f.attempted_size,
                "returncode": f.returncode,
                "oom": f.oom,
                "stderr_tail": f.stderr_tail,
                "command": f.command,
            }
            for f in failures
        ],
    }


def _write_reports(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Sharded Continuous Batched PLD Repeated Timing",
        "",
        f"n: `{payload['args']['n']}`  repeats: `{payload['args']['repeats']}`  "
        f"initial shard size: `{payload['args']['shard_size']}`",
        "",
        "| method | batch | tok/s mean ± std | speedup mean ± std | verifier forwards mean | forward reduction mean | peak GB mean | output matches mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(payload["summary"], key=lambda k: payload["summary"][k]["batch_size"]):
        s = payload["summary"][key]
        lines.append(
            f"| {s['method']} | {s['batch_size']} | "
            f"{s['tok_s']['mean']:.1f} ± {s['tok_s']['std']:.1f} | "
            f"{s['speedup']['mean']:.3f} ± {s['speedup']['std']:.3f} | "
            f"{s['verifier_forwards']['mean']:.1f} | "
            f"{s['verifier_forward_reduction']['mean']:.1f}% | "
            f"{s['memory_peak_gb']['mean']:.2f} | "
            f"{s['output_match_count']['mean']:.1f} |"
        )
    lines.append("")
    lines.append("This is a sharded exact-backend timing artifact; do not mix it with the unsharded bf16/SDPA headline protocol.")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = _read_jsonl_rows(Path(args.problem_jsonl), args.n)
    if len(rows) != args.n:
        raise SystemExit(f"requested n={args.n}, found {len(rows)} rows")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "n": args.n,
        "repeats": args.repeats,
        "initial_shard_size": args.shard_size,
        "ranges": _split_ranges(args.n, args.shard_size),
    }
    (out_dir / "shard_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        for repeat in range(args.repeats):
            for start, end in _split_ranges(args.n, args.shard_size):
                _run_one_shard(
                    args=args,
                    rows=rows,
                    repeat=repeat,
                    start=start,
                    end=end,
                    attempted_size=args.shard_size,
                    fallback_from=None,
                    failures=[],
                    completed=[],
                )
        return

    completed: list[ShardResult] = []
    failures: list[FailedAttempt] = []
    for repeat in range(args.repeats):
        for start, end in _split_ranges(args.n, args.shard_size):
            _run_one_shard(
                args=args,
                rows=rows,
                repeat=repeat,
                start=start,
                end=end,
                attempted_size=args.shard_size,
                fallback_from=None,
                failures=failures,
                completed=completed,
            )
    payload = _aggregate(completed, failures, args)
    _write_reports(payload, out_dir)
    print((out_dir / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
