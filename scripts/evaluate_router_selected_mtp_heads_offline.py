"""Offline replay for learned weak-router queued MTP with trained heads."""

from __future__ import annotations

import argparse
import bisect
import json
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asts.mtp_heads import MTPHeadConfig, PLDMTPHeads, accepted_prefix_length  # noqa: E402
from scripts.evaluate_queued_mtp_oracle import QueuedReplayStep, load_pld_steps  # noqa: E402
from scripts.train_weak_pld_router import (  # noqa: E402
    _empty_history,
    _safe_int,
    _update_history,
    extract_feature_dict,
    load_method_rows,
)


@dataclass(frozen=True)
class HeadPrediction:
    task_id: str
    create_step_id: int
    predictions: tuple[int, ...]
    labels: tuple[int, ...]
    confidence: float
    margin: float
    router_probability: float
    router_true_positive: bool


@dataclass(frozen=True)
class PendingQueue:
    position: int
    task_id: str
    create_step_id: int


def _load_router(path: Path):
    with path.open("rb") as f:
        payload = pickle.load(f)
    model = payload.get("model")
    if model is None:
        raise SystemExit(f"{path} does not contain a trained router model")
    return model


def _router_probabilities(
    steps_path: Path,
    *,
    method: str,
    router_path: Path,
    accepted_len_threshold: int,
) -> dict[tuple[str, int], float]:
    rows_by_task = load_method_rows(steps_path, method=method)
    router = _load_router(router_path)
    out: dict[tuple[str, int], float] = {}
    for task_id, rows in rows_by_task.items():
        history = _empty_history()
        features: list[dict[str, Any]] = []
        keys: list[tuple[str, int]] = []
        for step_index, row in enumerate(rows):
            features.append(
                extract_feature_dict(
                    row,
                    generated_start=_safe_int(row.get("_generated_start"), 0),
                    history=history,
                    step_index=step_index,
                )
            )
            keys.append((task_id, _safe_int(row.get("step"), 0)))
            _update_history(history, row, threshold=accepted_len_threshold)
        if features:
            probs = router.predict_proba(features)[:, 1]
            out.update({key: float(prob) for key, prob in zip(keys, probs, strict=True)})
    return out


def _load_head_predictions(
    *,
    data_path: Path,
    heads_path: Path,
    batch_size: int,
    device: str,
) -> tuple[dict[tuple[str, int], HeadPrediction], dict[str, Any]]:
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
    create_step_ids = [int(x) for x in data.get("create_step_id", data["step_id"])]
    router_prob = data.get("router_probability")
    if router_prob is None:
        router_prob = torch.ones(hidden.shape[0], dtype=torch.float32)
    router_tp = data.get("router_true_positive")
    if router_tp is None:
        router_tp = torch.ones(hidden.shape[0], dtype=torch.bool)

    records: dict[tuple[str, int], HeadPrediction] = {}
    horizon_total = [0 for _ in range(config.num_heads)]
    horizon_correct = [0 for _ in range(config.num_heads)]
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
                pred_tokens = tuple(int(v) for v in pred[j].tolist())
                label_tokens = tuple(int(v) for v in labels[idx].tolist() if int(v) >= 0)
                records[(task_ids[idx], create_step_ids[idx])] = HeadPrediction(
                    task_id=task_ids[idx],
                    create_step_id=create_step_ids[idx],
                    predictions=pred_tokens,
                    labels=label_tokens,
                    confidence=float(top1[j, 0].item()),
                    margin=float((top1[j, 0] - top2[j, 0]).item()),
                    router_probability=float(router_prob[idx].item()),
                    router_true_positive=bool(router_tp[idx].item()),
                )
                for h in range(min(config.num_heads, len(label_tokens))):
                    horizon_total[h] += 1
                    horizon_correct[h] += int(pred_tokens[h] == label_tokens[h])
    meta = {
        "data": str(data_path),
        "heads": str(heads_path),
        "num_predictions": len(records),
        "top1_accuracy_by_horizon_pct": [
            100.0 * c / t if t else 0.0 for c, t in zip(horizon_correct, horizon_total)
        ],
        "data_metadata": data.get("metadata") or {},
        "head_metadata": ckpt.get("metadata") or {},
    }
    return records, meta


def replay(
    steps_by_task: dict[str, list[QueuedReplayStep]],
    router_probs: dict[tuple[str, int], float],
    head_predictions: dict[tuple[str, int], HeadPrediction],
    *,
    router_threshold: float,
    confidence_threshold: float,
    accepted_len_threshold: int,
) -> dict[str, Any]:
    baseline_steps = sum(len(steps) for steps in steps_by_task.values())
    projected_steps = 0
    skipped_steps = 0
    queue_created = 0
    queue_selected = 0
    queue_used = 0
    queue_dropped_router = 0
    queue_dropped_conf = 0
    queue_dropped_position = 0
    missing_predictions = 0
    router_tp = 0
    router_fp = 0
    token0_reject = 0
    progress_less_than_pld = 0

    accepted_prefixes: list[int] = []
    progress_values: list[int] = []
    prefix_hist = Counter()
    calibration: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "token0_accept": 0})

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
            prob = router_probs.get((step.task_id, step.step_id), 0.0)

            if queue is not None:
                if int(queue.position) != int(step.start):
                    queue_dropped_position += 1
                    queue = None
                elif prob < router_threshold:
                    queue_dropped_router += 1
                    queue = None
                else:
                    queue_selected += 1
                    rec = head_predictions.get((queue.task_id, queue.create_step_id))
                    if rec is None:
                        missing_predictions += 1
                        queue_dropped_conf += 1
                        queue = None
                    else:
                        bucket = f"{min(9, int(rec.confidence * 10)) / 10:.1f}"
                        accepted = accepted_prefix_length(list(rec.predictions), list(rec.labels))
                        calibration[bucket]["n"] += 1
                        calibration[bucket]["token0_accept"] += int(accepted > 0)
                        if rec.confidence < confidence_threshold:
                            queue_dropped_conf += 1
                            queue = None
                        else:
                            remaining = max(0, task_end - step.start)
                            progress = min(remaining, accepted + 1) if remaining else 0
                            queue_used += 1
                            used_queue_this_step = True
                            accepted_prefixes.append(accepted)
                            progress_values.append(progress)
                            prefix_hist[str(accepted)] += 1
                            token0_reject += int(accepted == 0)
                            router_tp += int(step.accepted_len <= accepted_len_threshold)
                            router_fp += int(step.accepted_len > accepted_len_threshold)
                            progress_less_than_pld += int(progress < step.emitted)
                            selected_progress = max(1, progress)
                            queue = None

            if (
                not used_queue_this_step
                and step.accepted_len <= accepted_len_threshold
                and step.emitted > step.accepted_len
            ):
                position = step.start + step.emitted
                if position < task_end:
                    queue_created += 1
                    queue = PendingQueue(
                        position=position,
                        task_id=step.task_id,
                        create_step_id=step.step_id,
                    )

            if selected_progress <= step.emitted:
                i += 1
                continue
            covered_until = step.start + selected_progress
            j = bisect.bisect_left(starts, covered_until, lo=i + 1)
            skipped_steps += max(0, j - (i + 1))
            i = max(j, i + 1)

    projected_steps = max(1, projected_steps)
    avg = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
    return {
        "router_threshold": router_threshold,
        "confidence_threshold": confidence_threshold,
        "baseline_steps": baseline_steps,
        "projected_steps": projected_steps,
        "projected_speedup": baseline_steps / projected_steps,
        "step_reduction_pct": 100.0 * (baseline_steps - projected_steps) / max(1, baseline_steps),
        "skipped_baseline_steps": skipped_steps,
        "queue_created": queue_created,
        "queue_selected_by_router": queue_selected,
        "queue_used_after_confidence_gate": queue_used,
        "queue_dropped_by_router": queue_dropped_router,
        "queue_dropped_by_confidence_gate": queue_dropped_conf,
        "queue_dropped_position_mismatch": queue_dropped_position,
        "missing_head_predictions": missing_predictions,
        "used_token0_reject_count": token0_reject,
        "used_token0_reject_rate": token0_reject / max(1, queue_used),
        "avg_accepted_queued_tokens_per_use": avg(accepted_prefixes),
        "avg_progress_per_use": avg(progress_values),
        "accepted_prefix_distribution": {str(i): int(prefix_hist.get(str(i), 0)) for i in range(5)},
        "router_true_positives_used": router_tp,
        "router_false_positives_used": router_fp,
        "queue_progress_less_than_pld_count": progress_less_than_pld,
        "empirical_token0_acceptance_by_confidence_bucket": {
            bucket: {
                "n": values["n"],
                "token0_accept": values["token0_accept"],
                "token0_accept_rate": values["token0_accept"] / max(1, values["n"]),
            }
            for bucket, values in sorted(calibration.items())
        },
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Router-Selected MTP Offline Replay",
        "",
        f"steps: `{payload['steps']}`",
        f"data: `{payload['data']}`",
        f"heads: `{payload['heads']}`",
        "",
        "| router thr | conf gate | projected steps | speedup | queue used | token0 reject | accepted/use | progress/use |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["results"]:
        lines.append(
            f"| {row['router_threshold']:.2f} | {row['confidence_threshold']:.2f} | "
            f"{row['projected_steps']} | {row['projected_speedup']:.3f}x | "
            f"{row['queue_used_after_confidence_gate']} | "
            f"{100.0 * row['used_token0_reject_rate']:.1f}% | "
            f"{row['avg_accepted_queued_tokens_per_use']:.2f} | "
            f"{row['avg_progress_per_use']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"Best: router `{payload['best']['router_threshold']:.2f}`, confidence `{payload['best']['confidence_threshold']:.2f}`, "
            f"{payload['best']['projected_speedup']:.3f}x",
            "",
            f"Decision: **{payload['decision']}**",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=Path, required=True)
    ap.add_argument("--completions", type=Path, default=None)  # accepted for CLI symmetry; not needed
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--router", type=Path, required=True)
    ap.add_argument("--heads", type=Path, required=True)
    ap.add_argument("--method", default="blazedit_pld_w128_n10")
    ap.add_argument("--output-dir", type=Path, default=Path("/tmp/pld_mtp/router_selected_offline_eval"))
    ap.add_argument("--accepted-len-threshold", type=int, default=4)
    ap.add_argument("--router-thresholds", default="0.3,0.5,0.7")
    ap.add_argument("--confidence-thresholds", default="0.0,0.3,0.5,0.7,0.9")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    steps = load_pld_steps(args.steps, method=args.method)
    router_probs = _router_probabilities(
        args.steps,
        method=args.method,
        router_path=args.router,
        accepted_len_threshold=args.accepted_len_threshold,
    )
    preds, pred_meta = _load_head_predictions(
        data_path=args.data,
        heads_path=args.heads,
        batch_size=args.batch_size,
        device=args.device,
    )
    router_thresholds = [float(x) for x in args.router_thresholds.split(",") if x.strip()]
    confidence_thresholds = [float(x) for x in args.confidence_thresholds.split(",") if x.strip()]
    results = [
        replay(
            steps,
            router_probs,
            preds,
            router_threshold=router_threshold,
            confidence_threshold=confidence_threshold,
            accepted_len_threshold=args.accepted_len_threshold,
        )
        for router_threshold in router_thresholds
        for confidence_threshold in confidence_thresholds
    ]
    best = max(results, key=lambda row: row["projected_speedup"])
    passing = [
        row
        for row in results
        if row["projected_speedup"] >= 1.20
        and row["used_token0_reject_rate"] <= 0.40
        and row["avg_accepted_queued_tokens_per_use"] >= 1.0
    ]
    if passing:
        decision = "implement runtime weak-router queued MTP"
    elif 1.10 <= best["projected_speedup"] < 1.20:
        decision = "improve head training with more router-selected data or 2 epochs"
    else:
        decision = "MTP head quality is insufficient despite good routing"
    payload = {
        "steps": str(args.steps),
        "data": str(args.data),
        "router": str(args.router),
        "heads": str(args.heads),
        "method": args.method,
        "prediction_metadata": pred_meta,
        "results": results,
        "best": best,
        "passing_configs": passing,
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(args.output_dir / "report.md", payload)
    print((args.output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
