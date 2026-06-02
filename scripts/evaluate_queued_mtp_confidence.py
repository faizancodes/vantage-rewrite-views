"""Confidence-gated offline replay for queued PLD+MTP heads."""

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

from asts.mtp_heads import MTPHeadConfig, PLDMTPHeads, accepted_prefix_length  # noqa: E402
from scripts.evaluate_queued_mtp_oracle import (  # noqa: E402
    QueuedReplayStep,
    is_weak_step,
    load_pld_steps,
)


@dataclass(frozen=True)
class PredictionRecord:
    task_id: str
    step_id: int
    queued_predictions: tuple[int, ...]
    queued_labels: tuple[int, ...]
    confidence: float
    margin: float
    baseline_token_matched: bool


@dataclass(frozen=True)
class PendingPrediction:
    position: int
    record: PredictionRecord


def _bucket(confidence: float) -> str:
    lo = max(0, min(9, int(confidence * 10))) / 10.0
    hi = min(1.0, lo + 0.1)
    return f"{lo:.1f}-{hi:.1f}"


def load_prediction_records(
    *,
    data_path: Path,
    heads_path: Path,
    batch_size: int,
    device: str,
    confidence_source: str = "first_head",
) -> tuple[dict[tuple[str, int], PredictionRecord], dict[str, Any]]:
    import torch

    data = torch.load(data_path, map_location="cpu")
    ckpt = torch.load(heads_path, map_location="cpu")
    config = MTPHeadConfig(**ckpt["config"])
    output_weight = ckpt.get("output_weight")
    if output_weight is not None:
        output_weight = output_weight.to(device)
    model = PLDMTPHeads(config, output_weight=output_weight).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.prepare_for_inference()
    model.eval()

    hidden = data["hidden"].float()
    labels = data["labels"].long()[:, : config.num_heads]
    task_ids = [str(x) for x in data["task_id"]]
    step_ids = [int(x) for x in data["step_id"]]
    metadata = data.get("metadata") or {}
    label_mode = str(metadata.get("label_mode") or "post_pld")
    if label_mode not in {"post_pld", "queued_use"}:
        # Old post-PLD datasets did not carry label_mode.
        label_mode = "post_pld"

    records: dict[tuple[str, int], PredictionRecord] = {}
    horizon_correct = [0 for _ in range(config.num_heads)]
    horizon_total = [0 for _ in range(config.num_heads)]
    baseline_matches = 0
    usable_records = 0

    with torch.inference_mode():
        for start in range(0, hidden.shape[0], batch_size):
            x = hidden[start : start + batch_size].to(device)
            logits = model.forward_logits(x)
            probs = logits.float().softmax(dim=-1)
            top2_prob, top2_idx = probs.topk(k=2, dim=-1)
            pred = top2_idx[:, :, 0].cpu()
            top1 = top2_prob[:, :, 0].cpu()
            top2 = top2_prob[:, :, 1].cpu()
            for j in range(pred.shape[0]):
                idx = start + j
                pred_tokens = [int(v) for v in pred[j].tolist()]
                label_tokens = [int(v) for v in labels[idx].tolist() if int(v) >= 0]
                if label_mode == "queued_use":
                    queued_pred = tuple(pred_tokens[: config.num_heads])
                    queued_label = tuple(label_tokens[: config.num_heads])
                    baseline_ok = True
                    conf_head = 0
                else:
                    baseline_ok = bool(label_tokens) and bool(pred_tokens) and pred_tokens[0] == label_tokens[0]
                    if baseline_ok:
                        baseline_matches += 1
                    queued_pred = tuple(pred_tokens[1: config.num_heads])
                    queued_label = tuple(label_tokens[1: config.num_heads])
                    conf_head = 1 if confidence_source == "queued_first" and config.num_heads > 1 else 0
                if confidence_source == "queued_first" and label_mode == "queued_use":
                    conf_head = 0
                conf = float(top1[j, conf_head].item()) if config.num_heads > conf_head else 0.0
                margin = (
                    float((top1[j, conf_head] - top2[j, conf_head]).item())
                    if config.num_heads > conf_head
                    else 0.0
                )
                if baseline_ok and queued_pred and queued_label:
                    usable_records += 1
                records[(task_ids[idx], step_ids[idx])] = PredictionRecord(
                    task_id=task_ids[idx],
                    step_id=step_ids[idx],
                    queued_predictions=queued_pred,
                    queued_labels=queued_label,
                    confidence=conf,
                    margin=margin,
                    baseline_token_matched=baseline_ok,
                )
                for h in range(min(config.num_heads, len(label_tokens))):
                    horizon_total[h] += 1
                    horizon_correct[h] += int(pred_tokens[h] == label_tokens[h])

    meta = {
        "mode": "confidence_gated_trained_heads",
        "data": str(data_path),
        "heads": str(heads_path),
        "label_mode": label_mode,
        "confidence_source": confidence_source,
        "num_predictions": len(records),
        "usable_records_before_policy": usable_records,
        "baseline_token_match_count": baseline_matches,
        "top1_accuracy_by_horizon_pct": [
            100.0 * c / t if t else 0.0 for c, t in zip(horizon_correct, horizon_total)
        ],
    }
    return records, meta


def replay_confidence_threshold(
    steps_by_task: dict[str, list[QueuedReplayStep]],
    predictions: dict[tuple[str, int], PredictionRecord],
    *,
    threshold: float,
    trigger_threshold: int = 4,
    weak_field: str = "draft_len",
    use_margin: bool = False,
) -> dict[str, Any]:
    baseline_steps = sum(len(steps) for steps in steps_by_task.values())
    projected_steps = 0
    skipped_baseline_steps = 0
    missing_predictions = 0

    queue_candidates = 0
    queue_created = 0
    queue_used_before_gating = 0
    queue_used_after_gating = 0
    queue_gated_drop = 0
    queue_dropped_pld_strong = 0
    queue_dropped_position_mismatch = 0
    queue_expired = 0
    create_token0_reject = 0
    progress_deficits = 0

    accepted_per_used: list[int] = []
    progress_per_used: list[int] = []
    extra_progress_per_used: list[int] = []
    prefix_hist: dict[str, int] = defaultdict(int)
    calibration: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "token0_accept": 0})

    for _task_id, steps in steps_by_task.items():
        starts = [step.start for step in steps]
        task_end = steps[-1].start + steps[-1].emitted if steps else 0
        queue: PendingPrediction | None = None
        i = 0
        while i < len(steps):
            step = steps[i]
            projected_steps += 1
            selected_progress = step.emitted
            used_queue_this_step = False

            if queue is not None:
                if int(queue.position) != int(step.start):
                    queue_dropped_position_mismatch += 1
                    queue = None
                elif not is_weak_step(step, threshold=trigger_threshold, weak_field=weak_field):
                    queue_dropped_pld_strong += 1
                    queue = None
                else:
                    queue_used_before_gating += 1
                    score = queue.record.margin if use_margin else queue.record.confidence
                    accepted = accepted_prefix_length(
                        list(queue.record.queued_predictions),
                        list(queue.record.queued_labels),
                    )
                    bucket = _bucket(queue.record.confidence)
                    calibration[bucket]["n"] += 1
                    calibration[bucket]["token0_accept"] += int(accepted > 0)
                    if score < threshold:
                        queue_gated_drop += 1
                        queue = None
                    else:
                        remaining = max(0, task_end - step.start)
                        progress = min(remaining, accepted + 1) if remaining else 0
                        if progress <= 0:
                            queue_expired += 1
                            queue = None
                        else:
                            queue_used_after_gating += 1
                            used_queue_this_step = True
                            accepted_per_used.append(accepted)
                            progress_per_used.append(progress)
                            extra_progress_per_used.append(max(0, progress - step.emitted))
                            prefix_hist[str(accepted)] += 1
                            if progress < step.emitted:
                                progress_deficits += 1
                            selected_progress = max(step.emitted, progress)
                            queue = None

            if (
                not used_queue_this_step
                and is_weak_step(step, threshold=trigger_threshold, weak_field=weak_field)
                and step.emitted > step.accepted_len
            ):
                position = step.start + step.emitted
                if position < task_end:
                    queue_candidates += 1
                    rec = predictions.get((step.task_id, step.step_id))
                    if rec is None:
                        missing_predictions += 1
                    elif not rec.baseline_token_matched:
                        create_token0_reject += 1
                    elif rec.queued_predictions and rec.queued_labels:
                        queue_created += 1
                        queue = PendingPrediction(position=position, record=rec)

            if selected_progress <= step.emitted:
                i += 1
                continue
            covered_until = step.start + selected_progress
            j = bisect.bisect_left(starts, covered_until, lo=i + 1)
            skipped_baseline_steps += max(0, j - (i + 1))
            i = max(j, i + 1)

    projected_steps = max(1, projected_steps)
    token0_rejects = sum(1 for v in accepted_per_used if v == 0)
    avg = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
    calib_out = {
        bucket: {
            "n": values["n"],
            "token0_accept": values["token0_accept"],
            "token0_accept_rate": values["token0_accept"] / max(1, values["n"]),
        }
        for bucket, values in sorted(calibration.items())
    }
    return {
        "confidence_threshold": threshold,
        "score": "margin" if use_margin else "top1_probability",
        "baseline_steps": baseline_steps,
        "projected_steps": projected_steps,
        "projected_speedup": baseline_steps / projected_steps,
        "step_reduction_pct": 100.0 * (baseline_steps - projected_steps) / baseline_steps
        if baseline_steps
        else 0.0,
        "skipped_baseline_steps": skipped_baseline_steps,
        "queue_candidates": queue_candidates,
        "queue_created": queue_created,
        "queue_used_before_gating": queue_used_before_gating,
        "queue_used_after_gating": queue_used_after_gating,
        "gated_drop_count": queue_gated_drop,
        "queue_dropped_because_pld_strong": queue_dropped_pld_strong,
        "queue_dropped_position_mismatch": queue_dropped_position_mismatch,
        "queue_expired": queue_expired,
        "missing_predictions": missing_predictions,
        "create_token0_reject_count": create_token0_reject,
        "used_token0_reject_count": token0_rejects,
        "used_token0_reject_rate": token0_rejects / max(1, len(accepted_per_used)),
        "avg_accepted_queued_tokens_per_used_draft": avg(accepted_per_used),
        "avg_progress_per_used_queued_draft": avg(progress_per_used),
        "avg_extra_progress_per_used_queued_draft": avg(extra_progress_per_used),
        "accepted_prefix_len_distribution": dict(sorted(prefix_hist.items())),
        "head_confidence_calibration": calib_out,
        "queue_progress_less_than_baseline_count": progress_deficits,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Queued-MTP Confidence Evaluation",
        "",
        f"steps: `{payload['steps']}`",
        f"data: `{payload['metadata'].get('data')}`",
        f"heads: `{payload['metadata'].get('heads')}`",
        f"label_mode: `{payload['metadata'].get('label_mode')}`",
        f"confidence_source: `{payload['metadata'].get('confidence_source')}`",
        "",
        "| threshold | projected steps | speedup | queue used | gated | used token0 reject | accepted queued/use | progress/use |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["thresholds"]:
        lines.append(
            f"| {row['confidence_threshold']:.2f} | {row['projected_steps']} | "
            f"{row['projected_speedup']:.3f}x | {row['queue_used_after_gating']} | "
            f"{row['gated_drop_count']} | {100.0 * row['used_token0_reject_rate']:.1f}% | "
            f"{row['avg_accepted_queued_tokens_per_used_draft']:.2f} | "
            f"{row['avg_progress_per_used_queued_draft']:.2f} |"
        )
    lines.extend(["", f"Decision: **{payload['decision']}**"])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--heads", type=Path, required=True)
    ap.add_argument("--method", default="blazedit_pld_w128_n10")
    ap.add_argument("--output-dir", type=Path, default=Path("/tmp/pld_mtp/queued_confidence_eval"))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trigger-threshold", type=int, default=4)
    ap.add_argument("--weak-field", choices=["draft_len", "accepted_len"], default="draft_len")
    ap.add_argument("--confidence-thresholds", default="0.0,0.1,0.2,0.3,0.5,0.7,0.9")
    ap.add_argument("--confidence-source", choices=["first_head", "queued_first"], default="first_head")
    ap.add_argument("--use-margin", action="store_true")
    args = ap.parse_args()

    steps = load_pld_steps(args.steps, method=args.method)
    if not steps:
        raise SystemExit(f"no rows for method {args.method!r} in {args.steps}")
    predictions, meta = load_prediction_records(
        data_path=args.data,
        heads_path=args.heads,
        batch_size=args.batch_size,
        device=args.device,
        confidence_source=args.confidence_source,
    )
    thresholds = [float(item) for item in args.confidence_thresholds.split(",") if item.strip()]
    rows = [
        replay_confidence_threshold(
            steps,
            predictions,
            threshold=threshold,
            trigger_threshold=args.trigger_threshold,
            weak_field=args.weak_field,
            use_margin=args.use_margin,
        )
        for threshold in thresholds
    ]
    best = max(rows, key=lambda row: row["projected_speedup"], default={})
    pass_rows = [
        row
        for row in rows
        if row["used_token0_reject_rate"] <= 0.40
        and row["avg_accepted_queued_tokens_per_used_draft"] >= 1.0
        and row["projected_speedup"] >= 1.15
    ]
    decision = (
        "offline gates pass: queued runtime smoke is justified"
        if pass_rows
        else "offline gates fail: current queued heads are not good enough for runtime"
    )
    payload = {
        "steps": str(args.steps),
        "method": args.method,
        "metadata": meta,
        "trigger_threshold": args.trigger_threshold,
        "weak_field": args.weak_field,
        "thresholds": rows,
        "best_speedup_threshold": best.get("confidence_threshold"),
        "best_speedup": best.get("projected_speedup"),
        "passing_thresholds": [row["confidence_threshold"] for row in pass_rows],
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(args.output_dir / "report.md", payload)
    print((args.output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
