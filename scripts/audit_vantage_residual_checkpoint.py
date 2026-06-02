#!/usr/bin/env python3
"""Audit whether a VANTAGE residual checkpoint can support queued decoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "artifacts/vantage_residual/checkpoints/linear_k4_v1/model.pt"
DEFAULT_TRAIN = ROOT / "artifacts/vantage_residual/data/train.pt"
DEFAULT_TEST = ROOT / "artifacts/vantage_residual/data/test500.pt"
DEFAULT_OUTPUT = ROOT / "artifacts/vantage_residual/phase3_checkpoint_audit"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_torch(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    import torch

    obj = torch.load(path, map_location="cpu")
    return obj if isinstance(obj, dict) else {"_object_type": type(obj).__name__}


def _shape(value: Any) -> list[int] | None:
    return list(value.shape) if hasattr(value, "shape") else None


def _metadata(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    raw = payload.get("metadata")
    return raw if isinstance(raw, dict) else {}


def _source_metadata(payload: dict[str, Any] | None) -> dict[str, Any]:
    meta = _metadata(payload)
    raw = meta.get("source_metadata")
    return raw if isinstance(raw, dict) else {}


def _dataset_summary(path: Path, payload: dict[str, Any] | None) -> dict[str, Any]:
    meta = _metadata(payload)
    source = _source_metadata(payload)
    return {
        "path": _rel(path),
        "exists": payload is not None,
        "schema": meta.get("schema"),
        "split": meta.get("split"),
        "source_kind": meta.get("source_kind"),
        "hidden_source": meta.get("hidden_source") or source.get("hidden_source"),
        "label_mode": meta.get("label_mode") or source.get("label_mode"),
        "mtp_position": meta.get("mtp_position") or source.get("mtp_position"),
        "hidden_shape": _shape(payload.get("hidden")) if isinstance(payload, dict) else None,
        "labels_shape": _shape(payload.get("labels")) if isinstance(payload, dict) else None,
        "num_heads": meta.get("num_heads") or source.get("num_heads"),
        "trigger_counts": source.get("trigger_counts"),
    }


def _checkpoint_summary(path: Path, payload: dict[str, Any] | None) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".json")
    sidecar_payload: dict[str, Any] = {}
    if sidecar.exists():
        try:
            sidecar_payload = json.loads(sidecar.read_text())
        except json.JSONDecodeError:
            sidecar_payload = {"decode_error": True}
    config = payload.get("config") if isinstance(payload, dict) else None
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    return {
        "path": _rel(path),
        "exists": payload is not None,
        "sidecar": _rel(sidecar),
        "sidecar_exists": sidecar.exists(),
        "has_model_state": bool(isinstance(payload, dict) and "model_state" in payload),
        "config": config if isinstance(config, dict) else {},
        "metadata": metadata if isinstance(metadata, dict) else sidecar_payload,
        "label_vocab_size": (
            int(payload["label_vocab"].numel())
            if isinstance(payload, dict) and hasattr(payload.get("label_vocab"), "numel")
            else None
        ),
    }


def _compatibility(train: dict[str, Any], test: dict[str, Any], ckpt: dict[str, Any]) -> dict[str, Any]:
    mtp_positions = {train.get("mtp_position"), test.get("mtp_position")}
    label_modes = {train.get("label_mode"), test.get("label_mode")}
    hidden_sources = {train.get("hidden_source"), test.get("hidden_source")}
    source_kinds = {train.get("source_kind"), test.get("source_kind")}

    queued_label_mode = bool(
        {"queued_use", "router_selected_queued_use"} & {str(x) for x in label_modes if x}
    )
    queue_creation_hidden = bool(
        {"post_pld_queue_creation"} & {str(x) for x in hidden_sources if x}
    )
    post_pld_current_step = (
        "post_pld" in {str(x) for x in mtp_positions if x}
        and "postpld_mtp_artifact" in {str(x) for x in source_kinds if x}
        and not queued_label_mode
    )
    has_checkpoint = bool(ckpt.get("exists") and ckpt.get("has_model_state"))
    hidden_shape = train.get("hidden_shape") or []
    ckpt_hidden = ckpt.get("config", {}).get("hidden_size")
    shape_ok = bool(hidden_shape and ckpt_hidden and int(hidden_shape[-1]) == int(ckpt_hidden))

    return {
        "has_real_checkpoint": has_checkpoint,
        "checkpoint_hidden_matches_train": shape_ok,
        "hidden_state_available_after_verifier_call": "post_pld" in {str(x) for x in mtp_positions if x}
        or queue_creation_hidden,
        "can_predict_next_step_tokens_if_queued": queued_label_mode,
        "labels_aligned_to": (
            "next_decode_step"
            if queued_label_mode
            else "current_post_pld_residual_position"
            if post_pld_current_step
            else "unknown"
        ),
        "queued_runtime_compatible": bool(has_checkpoint and shape_ok and queued_label_mode),
        "retraining_required_for_queued_use": bool(has_checkpoint and shape_ok and not queued_label_mode),
        "decision": (
            "continue: checkpoint/data are queued-use compatible"
            if has_checkpoint and shape_ok and queued_label_mode
            else "stop: current checkpoint/data are post-PLD current-step residuals; queued next-step use requires retraining"
            if has_checkpoint and shape_ok and post_pld_current_step
            else "stop: missing or shape-incompatible checkpoint/data"
        ),
    }


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    compat = payload["compatibility"]
    lines = [
        "# VANTAGE-Residual Checkpoint Audit",
        "",
        f"checkpoint: `{payload['checkpoint']['path']}`",
        f"train data: `{payload['train_data']['path']}`",
        f"test data: `{payload['test_data']['path']}`",
        "",
        "## Answer",
        "",
        f"- Hidden state source: `{payload['test_data'].get('hidden_source') or payload['test_data'].get('mtp_position')}`.",
        f"- Label alignment: `{compat['labels_aligned_to']}`.",
        f"- Hidden state available after verifier call: `{compat['hidden_state_available_after_verifier_call']}`.",
        f"- Can predict next-step queued tokens: `{compat['can_predict_next_step_tokens_if_queued']}`.",
        f"- Queued-compatible without retraining: `{compat['queued_runtime_compatible']}`.",
        "",
        f"Decision: **{compat['decision']}**",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--test-data", type=Path, default=DEFAULT_TEST)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    ckpt_payload = _load_torch(args.checkpoint)
    train_payload = _load_torch(args.train_data)
    test_payload = _load_torch(args.test_data)
    checkpoint = _checkpoint_summary(args.checkpoint, ckpt_payload)
    train = _dataset_summary(args.train_data, train_payload)
    test = _dataset_summary(args.test_data, test_payload)
    payload = {
        "checkpoint": checkpoint,
        "train_data": train,
        "test_data": test,
        "compatibility": _compatibility(train, test, checkpoint),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_md(args.output_dir / "report.md", payload)
    print((args.output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
