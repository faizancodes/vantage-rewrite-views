"""Replay queued MTP using a pre-verification weak-PLD router."""

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

from asts.mtp_heads import accepted_prefix_length  # noqa: E402
from scripts.evaluate_queued_mtp_confidence import PredictionRecord, load_prediction_records  # noqa: E402
from scripts.evaluate_queued_mtp_oracle import QueuedReplayStep, load_pld_steps  # noqa: E402


@dataclass(frozen=True)
class PendingRouterQueue:
    position: int
    create_task_id: str
    create_step_id: int


def load_router_probabilities(path: Path, *, router_name: str) -> dict[tuple[str, int], float]:
    payload = json.loads(path.read_text())
    task_ids = [str(x) for x in payload["task_id"]]
    step_ids = [int(x) for x in payload["step_id"]]
    routers = payload["routers"]
    if router_name not in routers:
        raise SystemExit(f"router {router_name!r} not in {path}; available={sorted(routers)}")
    probs = [float(x) for x in routers[router_name]]
    return {
        (task_id, step_id): prob
        for task_id, step_id, prob in zip(task_ids, step_ids, probs, strict=True)
    }


def _label_weak(step: QueuedReplayStep, *, accepted_len_threshold: int) -> bool:
    return int(step.accepted_len) <= int(accepted_len_threshold)


def replay_with_router(
    steps_by_task: dict[str, list[QueuedReplayStep]],
    router_probabilities: dict[tuple[str, int], float],
    *,
    threshold: float,
    accepted_len_threshold: int = 4,
    num_heads: int = 4,
    trained_predictions: dict[tuple[str, int], PredictionRecord] | None = None,
) -> dict[str, Any]:
    """Replay strict queued routing.

    If a queue exists and the router predicts weak, the queued draft replaces
    PLD in that verifier slot.  Progress is therefore the queued verifier
    progress, not ``max(PLD, queued)``.
    """

    baseline_steps = sum(len(steps) for steps in steps_by_task.values())
    projected_steps = 0
    skipped_baseline_steps = 0

    router_pred_weak = 0
    router_false_positive = 0
    router_false_negative = 0
    missing_router_probs = 0

    queue_created = 0
    queue_used = 0
    queue_dropped_router_strong = 0
    queue_dropped_position_mismatch = 0
    queue_expired = 0
    queue_use_candidates = 0
    progress_less_than_baseline = 0
    missing_trained_predictions = 0

    accepted_tokens_per_used: list[int] = []
    progress_per_used: list[int] = []
    token0_rejects = 0

    for _task_id, steps in steps_by_task.items():
        starts = [step.start for step in steps]
        task_end = steps[-1].start + steps[-1].emitted if steps else 0
        queue: PendingRouterQueue | None = None
        i = 0
        while i < len(steps):
            step = steps[i]
            projected_steps += 1
            actual_weak = _label_weak(step, accepted_len_threshold=accepted_len_threshold)
            prob = router_probabilities.get((step.task_id, step.step_id))
            if prob is None:
                missing_router_probs += 1
                prob = 0.0
            predicted_weak = prob >= threshold
            router_pred_weak += int(predicted_weak)
            router_false_positive += int(predicted_weak and not actual_weak)
            router_false_negative += int((not predicted_weak) and actual_weak)

            selected_progress = step.emitted
            used_queue_this_step = False

            if queue is not None:
                if int(queue.position) != int(step.start):
                    queue_dropped_position_mismatch += 1
                    queue = None
                elif not predicted_weak:
                    queue_dropped_router_strong += 1
                    queue = None
                else:
                    queue_use_candidates += 1
                    remaining = max(0, task_end - step.start)
                    if trained_predictions is None:
                        accepted = min(max(0, num_heads - 1), max(0, remaining - 1))
                    else:
                        rec = trained_predictions.get((queue.create_task_id, queue.create_step_id))
                        if rec is None:
                            missing_trained_predictions += 1
                            accepted = 0
                        elif not rec.baseline_token_matched:
                            token0_rejects += 1
                            accepted = 0
                        else:
                            accepted = accepted_prefix_length(
                                list(rec.queued_predictions),
                                list(rec.queued_labels),
                            )
                            token0_rejects += int(accepted == 0)
                    progress = min(remaining, accepted + 1) if remaining else 0
                    if progress <= 0:
                        queue_expired += 1
                        queue = None
                    else:
                        queue_used += 1
                        used_queue_this_step = True
                        accepted_tokens_per_used.append(accepted)
                        progress_per_used.append(progress)
                        progress_less_than_baseline += int(progress < step.emitted)
                        selected_progress = progress
                        queue = None

            if not used_queue_this_step and actual_weak and step.emitted > step.accepted_len:
                position = step.start + step.emitted
                if position < task_end:
                    queue_created += 1
                    queue = PendingRouterQueue(
                        position=position,
                        create_task_id=step.task_id,
                        create_step_id=step.step_id,
                    )

            if selected_progress <= step.emitted:
                i += 1
                continue
            covered_until = step.start + selected_progress
            j = bisect.bisect_left(starts, covered_until, lo=i + 1)
            skipped_baseline_steps += max(0, j - (i + 1))
            i = max(j, i + 1)

    projected_steps = max(1, projected_steps)
    actual_steps = max(1, baseline_steps)
    avg = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
    return {
        "threshold": threshold,
        "mode": "trained_heads" if trained_predictions is not None else "perfect_mtp",
        "baseline_steps": baseline_steps,
        "projected_steps": projected_steps,
        "projected_speedup": baseline_steps / projected_steps,
        "step_reduction_pct": 100.0 * (baseline_steps - projected_steps) / actual_steps,
        "skipped_baseline_steps": skipped_baseline_steps,
        "queue_created": queue_created,
        "queue_used": queue_used,
        "queue_dropped_router_strong": queue_dropped_router_strong,
        "queue_dropped_position_mismatch": queue_dropped_position_mismatch,
        "queue_expired": queue_expired,
        "queue_use_candidates": queue_use_candidates,
        "router_weak_prediction_rate": router_pred_weak / actual_steps,
        "router_false_positives": router_false_positive,
        "router_false_negatives": router_false_negative,
        "router_false_positive_rate": router_false_positive / max(1, baseline_steps - sum(
            1 for steps in steps_by_task.values() for step in steps if _label_weak(step, accepted_len_threshold=accepted_len_threshold)
        )),
        "router_false_negative_rate": router_false_negative / max(1, sum(
            1 for steps in steps_by_task.values() for step in steps if _label_weak(step, accepted_len_threshold=accepted_len_threshold)
        )),
        "missing_router_probabilities": missing_router_probs,
        "missing_trained_predictions": missing_trained_predictions,
        "token0_reject_count": token0_rejects,
        "token0_reject_rate": token0_rejects / max(1, queue_used),
        "avg_accepted_queued_tokens_per_used_draft": avg(accepted_tokens_per_used),
        "avg_progress_per_used_queued_draft": avg(progress_per_used),
        "queue_progress_less_than_baseline_count": progress_less_than_baseline,
        "strict_routing_note": (
            "When the router chooses queued MTP, PLD is not also credited; queued progress replaces "
            "PLD progress in that verifier slot."
        ),
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Queued-MTP Weak-Router Projection",
        "",
        f"steps: `{payload['steps']}`",
        f"router: `{payload['router_predictions']}` / `{payload['router_name']}`",
        "",
        "## Perfect MTP",
        "",
        "| threshold | projected steps | speedup | queue created | queue used | pred weak | FP | FN |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["perfect_mtp"]:
        lines.append(
            f"| {row['threshold']:.2f} | {row['projected_steps']} | {row['projected_speedup']:.3f}x | "
            f"{row['queue_created']} | {row['queue_used']} | "
            f"{row['router_weak_prediction_rate']:.3f} | {row['router_false_positives']} | "
            f"{row['router_false_negatives']} |"
        )
    if payload.get("trained_heads"):
        lines.extend(
            [
                "",
                "## Trained Heads",
                "",
                "| threshold | projected steps | speedup | queue used | token0 reject | accepted queued/use |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in payload["trained_heads"]:
            lines.append(
                f"| {row['threshold']:.2f} | {row['projected_steps']} | {row['projected_speedup']:.3f}x | "
                f"{row['queue_used']} | {100.0 * row['token0_reject_rate']:.1f}% | "
                f"{row['avg_accepted_queued_tokens_per_used_draft']:.2f} |"
            )
    lines.extend(["", f"Decision: **{payload['decision']}**"])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=Path, required=True)
    ap.add_argument("--router-predictions", type=Path, required=True)
    ap.add_argument("--router-name", default="logistic")
    ap.add_argument("--method", default="blazedit_pld_w128_n10")
    ap.add_argument("--output-dir", type=Path, default=Path("/tmp/pld_mtp/weak_router_projection"))
    ap.add_argument("--accepted-len-threshold", type=int, default=4)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--thresholds", default="0.1,0.2,0.3,0.5,0.7,0.9")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--heads", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    steps = load_pld_steps(args.steps, method=args.method)
    if not steps:
        raise SystemExit(f"no rows for method {args.method!r} in {args.steps}")
    router_probs = load_router_probabilities(args.router_predictions, router_name=args.router_name)
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    perfect_rows = [
        replay_with_router(
            steps,
            router_probs,
            threshold=threshold,
            accepted_len_threshold=args.accepted_len_threshold,
            num_heads=args.num_heads,
            trained_predictions=None,
        )
        for threshold in thresholds
    ]

    trained_rows: list[dict[str, Any]] = []
    trained_meta: dict[str, Any] | None = None
    if args.data and args.heads:
        trained_predictions, trained_meta = load_prediction_records(
            data_path=args.data,
            heads_path=args.heads,
            batch_size=args.batch_size,
            device=args.device,
            confidence_source="queued_first",
        )
        trained_rows = [
            replay_with_router(
                steps,
                router_probs,
                threshold=threshold,
                accepted_len_threshold=args.accepted_len_threshold,
                num_heads=args.num_heads,
                trained_predictions=trained_predictions,
            )
            for threshold in thresholds
        ]

    best_perfect = max(perfect_rows, key=lambda row: row["projected_speedup"], default={})
    best_trained = max(trained_rows, key=lambda row: row["projected_speedup"], default={})
    if best_perfect.get("projected_speedup", 0.0) < 1.20:
        decision = "abandon queued MTP: perfect MTP plus learned weak router is below 1.20x"
    elif not trained_rows or best_trained.get("projected_speedup", 0.0) < 1.15:
        decision = "improve/train MTP heads specifically for router-selected queued-use cases"
    elif best_trained.get("projected_speedup", 0.0) >= 1.20:
        decision = "implement runtime weak-router queued MTP"
    else:
        decision = "router has headroom, but trained heads are marginal; retrain before runtime"

    payload = {
        "steps": str(args.steps),
        "method": args.method,
        "router_predictions": str(args.router_predictions),
        "router_name": args.router_name,
        "accepted_len_threshold": args.accepted_len_threshold,
        "num_heads": args.num_heads,
        "perfect_mtp": perfect_rows,
        "trained_heads": trained_rows,
        "trained_heads_metadata": trained_meta,
        "best_perfect_speedup": best_perfect.get("projected_speedup"),
        "best_trained_speedup": best_trained.get("projected_speedup") if best_trained else None,
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(args.output_dir / "report.md", payload)
    print((args.output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
