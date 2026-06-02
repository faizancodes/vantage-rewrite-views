"""Replay the oracle ceiling for queued PLD+MTP heads.

This is an offline diagnostic for the current queued runtime shape:

* exact PLD runs first;
* an MTP queue item is created only after a weak PLD step;
* that queue item can only be used on the immediately following generation
  position, and only if PLD is weak there too;
* queued tokens are verified by the normal next verifier pass.

The oracle replaces the trained MTP prediction with the true future token
stream.  It therefore measures architectural headroom, not model-head quality.
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_pld_candidate_reranker import _load_jsonl  # noqa: E402


@dataclass(frozen=True)
class QueuedReplayStep:
    task_id: str
    step_id: int
    start: int
    emitted: int
    accepted_len: int
    draft_len: int
    pld_miss: bool


@dataclass(frozen=True)
class PendingQueue:
    position: int
    create_step_id: int


def load_pld_steps(path: Path, *, method: str) -> dict[str, list[QueuedReplayStep]]:
    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _load_jsonl(path):
        if row.get("method") == method:
            rows_by_task[str(row.get("task_id") or "")].append(row)

    out: dict[str, list[QueuedReplayStep]] = {}
    for task_id, rows in rows_by_task.items():
        pos = 0
        steps: list[QueuedReplayStep] = []
        for row in sorted(rows, key=lambda r: int(r.get("step") or 0)):
            emitted = max(1, int(row.get("n_emitted") or 0))
            accepted = int(row.get("n_accepted_drafts") or 0)
            draft_len = int(row.get("proposal_tokens") or row.get("k") or 0)
            pld_miss = not (
                row.get("pld_exact_hit") is True or row.get("proposal_kind") == "blazedit_pld"
            )
            steps.append(
                QueuedReplayStep(
                    task_id=task_id,
                    step_id=int(row.get("step") or 0),
                    start=pos,
                    emitted=emitted,
                    accepted_len=accepted,
                    draft_len=draft_len,
                    pld_miss=pld_miss,
                )
            )
            pos += emitted
        out[task_id] = steps
    return out


def is_weak_step(step: QueuedReplayStep, *, threshold: int, weak_field: str) -> bool:
    if weak_field == "draft_len":
        return step.draft_len <= threshold
    if weak_field == "accepted_len":
        return step.accepted_len <= threshold
    raise ValueError(f"unsupported weak_field: {weak_field!r}")


def replay_perfect_queued_oracle(
    steps_by_task: dict[str, list[QueuedReplayStep]],
    *,
    num_heads: int = 4,
    trigger_threshold: int = 4,
    weak_field: str = "draft_len",
) -> dict[str, Any]:
    """Return a decode-step replay using perfect queued predictions.

    Runtime K=4 post-PLD heads spend head 0 on the PLD correction/bonus token
    that has already been emitted by the creation step.  The queued draft
    therefore has at most K-1 tokens available to verify on the next step.
    """

    baseline_steps = sum(len(steps) for steps in steps_by_task.values())
    projected_steps = 0
    skipped_baseline_steps = 0

    queue_created = 0
    queue_used = 0
    queue_dropped_pld_strong = 0
    queue_dropped_position_mismatch = 0
    queue_expired = 0
    create_candidates = 0
    use_candidates_before_policy = 0
    progress_deficits = 0

    accepted_tokens_per_used: list[int] = []
    progress_per_used: list[int] = []
    extra_progress_per_used: list[int] = []

    for _task_id, steps in steps_by_task.items():
        starts = [step.start for step in steps]
        task_end = steps[-1].start + steps[-1].emitted if steps else 0
        queue: PendingQueue | None = None
        i = 0
        while i < len(steps):
            step = steps[i]
            projected_steps += 1
            selected_progress = step.emitted
            used_queue_this_step = False

            if queue is not None:
                use_candidates_before_policy += 1
                if int(queue.position) != int(step.start):
                    queue_dropped_position_mismatch += 1
                    queue = None
                elif not is_weak_step(step, threshold=trigger_threshold, weak_field=weak_field):
                    queue_dropped_pld_strong += 1
                    queue = None
                else:
                    remaining = max(0, task_end - step.start)
                    oracle_accepted = min(max(0, num_heads - 1), max(0, remaining - 1))
                    oracle_progress = min(remaining, oracle_accepted + 1) if remaining else 0
                    if oracle_progress <= 0:
                        queue_expired += 1
                        queue = None
                    else:
                        queue_used += 1
                        used_queue_this_step = True
                        accepted_tokens_per_used.append(oracle_accepted)
                        progress_per_used.append(oracle_progress)
                        extra_progress_per_used.append(max(0, oracle_progress - step.emitted))
                        if oracle_progress < step.emitted:
                            progress_deficits += 1
                        # The projection is an architectural ceiling.  It counts
                        # step reductions from queued drafts but does not credit
                        # a perfect oracle with regressions against baseline PLD.
                        selected_progress = max(step.emitted, oracle_progress)
                        queue = None

            if (
                not used_queue_this_step
                and is_weak_step(step, threshold=trigger_threshold, weak_field=weak_field)
                and step.emitted > step.accepted_len
            ):
                position = step.start + step.emitted
                if position < task_end:
                    create_candidates += 1
                    queue_created += 1
                    queue = PendingQueue(position=position, create_step_id=step.step_id)

            if selected_progress <= step.emitted:
                i += 1
                continue
            covered_until = step.start + selected_progress
            j = bisect.bisect_left(starts, covered_until, lo=i + 1)
            skipped_baseline_steps += max(0, j - (i + 1))
            i = max(j, i + 1)

    projected_steps = max(1, projected_steps)
    avg = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
    return {
        "baseline_steps": baseline_steps,
        "projected_queued_oracle_steps": projected_steps,
        "projected_speedup": baseline_steps / projected_steps if projected_steps else 0.0,
        "step_reduction_pct": (
            100.0 * (baseline_steps - projected_steps) / baseline_steps if baseline_steps else 0.0
        ),
        "skipped_baseline_steps": skipped_baseline_steps,
        "num_heads": num_heads,
        "queued_draft_tokens": max(0, num_heads - 1),
        "trigger_threshold": trigger_threshold,
        "weak_field": weak_field,
        "queue_create_candidates": create_candidates,
        "queue_created": queue_created,
        "queue_used": queue_used,
        "queue_use_candidates_before_policy": use_candidates_before_policy,
        "queue_dropped_because_pld_strong": queue_dropped_pld_strong,
        "queue_dropped_position_mismatch": queue_dropped_position_mismatch,
        "queue_expired": queue_expired,
        "oracle_token0_reject_count": 0,
        "oracle_token0_reject_rate": 0.0,
        "oracle_accepted_queued_tokens_per_used_draft": avg(accepted_tokens_per_used),
        "oracle_progress_per_used_draft": avg(progress_per_used),
        "oracle_extra_progress_per_used_draft": avg(extra_progress_per_used),
        "queue_progress_less_than_baseline_count": progress_deficits,
        "projection_note": (
            "Uses max(baseline_progress, oracle_queue_progress) for step skipping so the oracle "
            "measures non-regressing architectural ceiling rather than penalizing perfect queued "
            "drafts that are shorter than a weak PLD step near the end of a draft."
        ),
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Perfect Queued-MTP Oracle",
        "",
        f"steps: `{payload['steps']}`",
        f"method: `{payload['method']}`",
        "",
        "| baseline steps | projected steps | step reduction | speedup | queue created | queue used | dropped PLD strong | token0 reject | accepted queued/use | progress/use |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    row = payload["oracle"]
    lines.append(
        f"| {row['baseline_steps']} | {row['projected_queued_oracle_steps']} | "
        f"{row['step_reduction_pct']:.2f}% | {row['projected_speedup']:.3f}x | "
        f"{row['queue_created']} | {row['queue_used']} | "
        f"{row['queue_dropped_because_pld_strong']} | {row['oracle_token0_reject_rate']:.1%} | "
        f"{row['oracle_accepted_queued_tokens_per_used_draft']:.2f} | "
        f"{row['oracle_progress_per_used_draft']:.2f} |"
    )
    lines.extend(
        [
            "",
            f"Decision: **{payload['decision']}**",
            "",
            row["projection_note"],
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=Path, required=True)
    ap.add_argument("--method", default="blazedit_pld_w128_n10")
    ap.add_argument("--output-dir", type=Path, default=Path("/tmp/pld_mtp/queued_oracle"))
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--trigger-threshold", type=int, default=4)
    ap.add_argument("--weak-field", choices=["draft_len", "accepted_len"], default="draft_len")
    args = ap.parse_args()

    steps = load_pld_steps(args.steps, method=args.method)
    if not steps:
        raise SystemExit(f"no rows for method {args.method!r} in {args.steps}")

    oracle = replay_perfect_queued_oracle(
        steps,
        num_heads=args.num_heads,
        trigger_threshold=args.trigger_threshold,
        weak_field=args.weak_field,
    )
    decision = (
        "continue: queued architecture has >=1.20x perfect-oracle ceiling"
        if oracle["projected_speedup"] >= 1.20
        else "abandon queued architecture: perfect-oracle ceiling is below 1.20x"
    )
    payload = {
        "steps": str(args.steps),
        "method": args.method,
        "oracle": oracle,
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(args.output_dir / "report.md", payload)
    print((args.output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
