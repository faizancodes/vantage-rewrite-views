#!/usr/bin/env python3
"""Gate VANTAGE-Residual-Queued from existing datasets and replay artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "artifacts/vantage_residual/checkpoints/linear_k4_v1/model.pt"
DEFAULT_DATA = ROOT / "artifacts/vantage_residual/data/test500.pt"
DEFAULT_HIDDEN = ROOT / "artifacts/vantage_residual/phase3_hidden_capture/summary.json"
DEFAULT_OUTPUT = ROOT / "artifacts/vantage_residual/phase3_offline_queued"
DEFAULT_EXISTING_REPORTS = (
    ROOT / "analysis/pld_mtp/router_selected_finetune_n917_v1/router_selected_finetune_offline_eval/report.json",
    ROOT / "analysis/pld_mtp/router_selected_k4_v2/router_selected_offline_eval/report.json",
)
DEFAULT_PHASE4_OUTPUT = ROOT / "artifacts/vantage_residual/phase4_offline/queued_linear_k4_v1"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _load_torch(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    import torch

    obj = torch.load(path, map_location="cpu")
    return obj if isinstance(obj, dict) else None


def _data_schema(path: Path) -> dict[str, Any]:
    payload = _load_torch(path)
    if payload is None:
        return {"path": _rel(path), "exists": False}
    meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    source = meta.get("source_metadata") if isinstance(meta.get("source_metadata"), dict) else {}
    label_mode = meta.get("label_mode") or source.get("label_mode")
    hidden_source = meta.get("hidden_source") or source.get("hidden_source")
    mtp_position = meta.get("mtp_position") or source.get("mtp_position")
    source_kind = meta.get("source_kind")
    return {
        "path": _rel(path),
        "exists": True,
        "schema": meta.get("schema"),
        "source_kind": source_kind,
        "label_mode": label_mode,
        "hidden_source": hidden_source,
        "mtp_position": mtp_position,
        "hidden_shape": list(payload["hidden"].shape) if hasattr(payload.get("hidden"), "shape") else None,
        "labels_shape": list(payload["labels"].shape) if hasattr(payload.get("labels"), "shape") else None,
        "queued_label_compatible": label_mode in {"queued_use", "router_selected_queued_use"},
    }


def _checkpoint_schema(path: Path) -> dict[str, Any]:
    payload = _load_torch(path)
    if payload is None:
        return {"path": _rel(path), "exists": False, "has_model_state": False}
    return {
        "path": _rel(path),
        "exists": True,
        "has_model_state": "model_state" in payload,
        "config": payload.get("config") if isinstance(payload.get("config"), dict) else {},
        "metadata": payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
    }


def _predict_queued_records(
    *,
    data_path: Path,
    checkpoint_path: Path,
    batch_size: int,
    device: str,
    confidence_source: str,
) -> tuple[dict[tuple[str, int], Any], dict[str, Any]]:
    """Load trained queued predictions.

    The primary path supports ``train_pld_mtp_heads.py`` checkpoints because
    those are the scientifically relevant tied-output heads for Qwen-7B.  A
    compact ``ResidualMTPHeads`` checkpoint is intentionally not treated as a
    production-equivalent result unless it carries the same token IDs.
    """

    from scripts.evaluate_queued_mtp_confidence import load_prediction_records

    return load_prediction_records(
        data_path=data_path,
        heads_path=checkpoint_path,
        batch_size=batch_size,
        device=device,
        confidence_source=confidence_source,
    )


def _strict_replay(
    *,
    steps_path: Path,
    method: str,
    predictions: dict[tuple[str, int], Any],
    confidence_threshold: float,
    pld_draft_len_threshold: int,
    previous_accepted_len_threshold: int,
    weak_field: str,
) -> dict[str, Any]:
    from asts.mtp_heads import accepted_prefix_length
    from scripts.evaluate_queued_mtp_oracle import is_weak_step, load_pld_steps

    steps_by_task = load_pld_steps(steps_path, method=method)
    baseline_steps = sum(len(steps) for steps in steps_by_task.values())
    projected_steps = 0
    skipped_steps = 0
    queue_created = 0
    queue_available = 0
    queue_used = 0
    queue_dropped_conf = 0
    queue_dropped_pld_strong = 0
    queue_dropped_position = 0
    missing_predictions = 0
    token0_rejects = 0
    progress_less_than_pld = 0
    accepted_per_use: list[int] = []
    progress_per_use: list[int] = []

    for _task_id, steps in steps_by_task.items():
        starts = [step.start for step in steps]
        task_end = steps[-1].start + steps[-1].emitted if steps else 0
        queue: tuple[int, Any] | None = None
        i = 0
        while i < len(steps):
            step = steps[i]
            projected_steps += 1
            selected_progress = step.emitted
            used_queue = False
            if queue is not None:
                position, rec = queue
                if int(position) != int(step.start):
                    queue_dropped_position += 1
                    queue = None
                elif not is_weak_step(step, threshold=pld_draft_len_threshold, weak_field=weak_field):
                    queue_dropped_pld_strong += 1
                    queue = None
                else:
                    queue_available += 1
                    if float(rec.confidence) < float(confidence_threshold):
                        queue_dropped_conf += 1
                        queue = None
                    else:
                        accepted = accepted_prefix_length(
                            list(rec.queued_predictions),
                            list(rec.queued_labels),
                        )
                        remaining = max(0, task_end - step.start)
                        progress = min(remaining, accepted + 1) if remaining else 0
                        if progress > 0:
                            queue_used += 1
                            used_queue = True
                            token0_rejects += int(accepted == 0)
                            accepted_per_use.append(int(accepted))
                            progress_per_use.append(int(progress))
                            progress_less_than_pld += int(progress < step.emitted)
                            selected_progress = progress
                        queue = None

            if (
                not used_queue
                and int(step.accepted_len) <= int(previous_accepted_len_threshold)
                and step.emitted > step.accepted_len
            ):
                rec = predictions.get((step.task_id, step.step_id))
                if rec is None:
                    missing_predictions += 1
                elif rec.queued_predictions and rec.queued_labels:
                    position = step.start + step.emitted
                    if position < task_end:
                        queue_created += 1
                        queue = (position, rec)

            if selected_progress <= step.emitted:
                i += 1
                continue
            import bisect

            covered_until = step.start + selected_progress
            j = bisect.bisect_left(starts, covered_until, lo=i + 1)
            skipped_steps += max(0, j - (i + 1))
            i = max(j, i + 1)

    avg = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
    baseline_steps = max(1, baseline_steps)
    projected_steps = max(1, projected_steps)
    return {
        "confidence_threshold": confidence_threshold,
        "pld_draft_len_threshold": pld_draft_len_threshold,
        "previous_accepted_len_threshold": previous_accepted_len_threshold,
        "weak_field": weak_field,
        "baseline_steps": baseline_steps,
        "projected_steps": projected_steps,
        "projected_speedup": baseline_steps / projected_steps,
        "step_reduction_pct": 100.0 * (baseline_steps - projected_steps) / baseline_steps,
        "skipped_steps": skipped_steps,
        "queue_created": queue_created,
        "queue_available": queue_available,
        "queue_used": queue_used,
        "queue_dropped_confidence": queue_dropped_conf,
        "queue_dropped_pld_strong": queue_dropped_pld_strong,
        "queue_dropped_position": queue_dropped_position,
        "missing_predictions": missing_predictions,
        "token0_reject_count": token0_rejects,
        "token0_reject_rate": token0_rejects / max(1, queue_used),
        "accepted_per_use": avg(accepted_per_use),
        "progress_per_use": avg(progress_per_use),
        "progress_less_than_pld_count": progress_less_than_pld,
        "strict_replacement_note": (
            "Queued residual replaces PLD in one verifier slot; progress less "
            "than PLD is counted as a false-positive selector risk."
        ),
    }


def _direct_hidden_overhead_pct(hidden_capture: dict[str, Any] | None) -> float | None:
    if not isinstance(hidden_capture, dict):
        return None
    value = hidden_capture.get("direct_hidden_capture_overhead_pct")
    return float(value) if isinstance(value, (int, float)) else None


def _best_from_report(path: Path) -> dict[str, Any] | None:
    payload = _load_json(path)
    if payload is None:
        return None
    best = payload.get("best")
    if isinstance(best, dict):
        return {"path": _rel(path), **best}
    candidates: list[dict[str, Any]] = []
    for key in ("trained_heads", "thresholds", "policies"):
        rows = payload.get(key)
        if isinstance(rows, list):
            candidates.extend(row for row in rows if isinstance(row, dict))
    if not candidates:
        return None
    def score(row: dict[str, Any]) -> float:
        return float(row.get("projected_speedup") or row.get("corrected_projected_speedup") or 0.0)

    return {"path": _rel(path), **max(candidates, key=score)}


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# VANTAGE-Residual Queued Offline Gate",
        "",
        f"checkpoint: `{payload['checkpoint']['path']}`",
        f"data: `{payload['data']['path']}`",
        f"hidden capture: `{payload['hidden_capture_path']}`",
        "",
        "## Schema",
        "",
        f"- Data label mode: `{payload['data'].get('label_mode')}`.",
        f"- Data hidden source: `{payload['data'].get('hidden_source') or payload['data'].get('mtp_position')}`.",
        f"- Queued label compatible: `{payload['data'].get('queued_label_compatible')}`.",
        "",
        "## Existing Queued Replays",
        "",
        "| report | projected speedup | token0 reject | accepted/use | decision signal |",
        "|---|---:|---:|---:|---|",
    ]
    for row in payload["existing_replay_best_rows"]:
        lines.append(
            "| {path} | {speed:.3f}x | {tok0} | {accepted} | {signal} |".format(
                path=row.get("path"),
                speed=float(row.get("projected_speedup") or row.get("corrected_projected_speedup") or 0.0),
                tok0=_pct(row.get("used_token0_reject_rate") or row.get("token0_reject_rate")),
                accepted=_num(row.get("avg_accepted_queued_tokens_per_use") or row.get("avg_accepted_queued_tokens_per_used_draft")),
                signal=row.get("decision_signal"),
            )
        )
    if payload.get("phase4_rows"):
        lines.extend(
            [
                "",
                "## Phase 4 Queued-Head Replay",
                "",
                "| conf | PLD len <= | prev accepted <= | speedup | after hidden | queue used | token0 reject | accepted/use | PLD-regressions |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in payload["phase4_rows"]:
            lines.append(
                "| {conf:.2f} | {pld} | {prev} | {speed:.3f}x | {adj:.3f}x | {used} | {tok0} | {accepted} | {regress} |".format(
                    conf=float(row.get("confidence_threshold", 0.0)),
                    pld=row.get("pld_draft_len_threshold"),
                    prev=row.get("previous_accepted_len_threshold"),
                    speed=float(row.get("projected_speedup", 0.0)),
                    adj=float(row.get("projected_speedup_after_hidden_overhead", 0.0)),
                    used=row.get("queue_used"),
                    tok0=_pct(row.get("token0_reject_rate")),
                    accepted=_num(row.get("accepted_per_use")),
                    regress=row.get("progress_less_than_pld_count"),
                )
            )
    lines.extend(["", f"Decision: **{payload['decision']}**"])
    path.write_text("\n".join(lines) + "\n")


def _pct(value: Any) -> str:
    return "n/a" if not isinstance(value, (int, float)) else f"{100.0 * value:.1f}%"


def _num(value: Any) -> str:
    return "n/a" if not isinstance(value, (int, float)) else f"{value:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--steps", type=Path, default=None)
    ap.add_argument("--method", default="blazedit_pld_w128_n10")
    ap.add_argument("--hidden-capture-overhead-json", type=Path, default=DEFAULT_HIDDEN)
    ap.add_argument("--existing-report", action="append", type=Path, default=[])
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--confidence-thresholds", default="0.0,0.1,0.2,0.3,0.5,0.7,0.9")
    ap.add_argument("--pld-draft-len-thresholds", default="0,1,2,4")
    ap.add_argument("--previous-accepted-len-thresholds", default="0,1,2,4")
    ap.add_argument("--weak-field", choices=["draft_len", "accepted_len"], default="draft_len")
    ap.add_argument("--confidence-source", choices=["first_head", "queued_first"], default="first_head")
    args = ap.parse_args()

    data = _data_schema(args.data)
    checkpoint = _checkpoint_schema(args.checkpoint)
    hidden_capture = _load_json(args.hidden_capture_overhead_json)
    report_paths = args.existing_report or list(DEFAULT_EXISTING_REPORTS)
    best_rows = []
    for path in report_paths:
        row = _best_from_report(path)
        if row is None:
            continue
        speed = float(row.get("projected_speedup") or row.get("corrected_projected_speedup") or 0.0)
        tok0 = row.get("used_token0_reject_rate") or row.get("token0_reject_rate")
        row["decision_signal"] = (
            "passes speed gate"
            if speed >= 1.10 and (not isinstance(tok0, (int, float)) or tok0 <= 0.4)
            else "fails speed/token0 gate"
        )
        best_rows.append(row)

    best_speed = max(
        [float(row.get("projected_speedup") or row.get("corrected_projected_speedup") or 0.0) for row in best_rows],
        default=0.0,
    )
    overhead_pct = _direct_hidden_overhead_pct(hidden_capture)
    hidden_direct = overhead_pct is not None
    phase4_rows: list[dict[str, Any]] = []
    if data.get("queued_label_compatible") and checkpoint.get("has_model_state") and args.steps is not None:
        predictions, pred_meta = _predict_queued_records(
            data_path=args.data,
            checkpoint_path=args.checkpoint,
            batch_size=int(args.batch_size),
            device=args.device,
            confidence_source=args.confidence_source,
        )
        confs = [float(x) for x in args.confidence_thresholds.split(",") if x.strip()]
        pld_lens = [int(x) for x in args.pld_draft_len_thresholds.split(",") if x.strip()]
        prevs = [int(x) for x in args.previous_accepted_len_thresholds.split(",") if x.strip()]
        for conf in confs:
            for pld_len in pld_lens:
                for prev in prevs:
                    row = _strict_replay(
                        steps_path=args.steps,
                        method=args.method,
                        predictions=predictions,
                        confidence_threshold=conf,
                        pld_draft_len_threshold=pld_len,
                        previous_accepted_len_threshold=prev,
                        weak_field=args.weak_field,
                    )
                    adjusted = float(row["projected_speedup"])
                    if overhead_pct is not None:
                        adjusted = adjusted / (1.0 + max(0.0, float(overhead_pct)) / 100.0)
                    row["projected_speedup_after_hidden_overhead"] = adjusted
                    phase4_rows.append(row)
        phase4_rows.sort(
            key=lambda row: (
                float(row.get("projected_speedup_after_hidden_overhead", 0.0)),
                float(row.get("projected_speedup", 0.0)),
            ),
            reverse=True,
        )
        best_speed = max(best_speed, float(phase4_rows[0].get("projected_speedup", 0.0)) if phase4_rows else 0.0)
    elif data.get("queued_label_compatible") and args.steps is None:
        pred_meta = {"blocked": "no --steps path; cannot project verifier-step reduction"}
    else:
        pred_meta = {}

    if not data.get("queued_label_compatible"):
        decision = "stop: provided data/checkpoint are not queued-label compatible; retraining queued-use heads is required"
    elif not hidden_direct:
        decision = "stop: hidden-capture overhead is not directly measured"
    elif not phase4_rows:
        decision = "stop: no PLD steps were supplied, so queued verifier-step reduction cannot be projected"
    elif not [
        row
        for row in phase4_rows
        if float(row.get("projected_speedup_after_hidden_overhead", 0.0)) >= 1.15
        and float(row.get("token0_reject_rate", 1.0)) <= 0.45
        and float(row.get("accepted_per_use", 0.0)) >= 1.0
        and int(row.get("queue_used", 0)) > 0
    ]:
        decision = "stop: queued offline projection fails the speed/token0/accepted-use gates"
    else:
        decision = "continue: queued offline gate passed"

    payload = {
        "checkpoint": checkpoint,
        "data": data,
        "hidden_capture_path": _rel(args.hidden_capture_overhead_json),
        "hidden_capture": hidden_capture,
        "existing_replay_best_rows": best_rows,
        "best_existing_projected_speedup": best_speed,
        "prediction_metadata": pred_meta,
        "phase4_rows": phase4_rows[:20],
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_md(args.output_dir / "report.md", payload)
    print((args.output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
