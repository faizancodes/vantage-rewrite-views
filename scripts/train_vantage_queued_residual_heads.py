#!/usr/bin/env python3
"""Train queued-use VANTAGE-Residual heads.

This wrapper refuses post-PLD/current-step residual tensors.  It trains the
same lightweight PLD-MTP head family used by earlier offline projections, but
only on datasets whose metadata says the labels are queued next-step labels.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "artifacts/vantage_residual/phase4_data/queued_v1/train.pt"
DEFAULT_VAL = ROOT / "artifacts/vantage_residual/phase4_data/queued_v1/val.pt"
DEFAULT_OUTPUT = ROOT / "artifacts/vantage_residual/phase4_checkpoints/queued_linear_k4_v1"
DEFAULT_TABLE = ROOT / "artifacts/vantage_residual/tables/phase4_training_gate.md"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_torch(path: Path) -> dict[str, Any]:
    import torch

    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise SystemExit(f"{path} is not a tensor dictionary")
    return obj


def _assert_queued(path: Path) -> dict[str, Any]:
    payload = _load_torch(path)
    meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    label_mode = meta.get("label_mode")
    if label_mode not in {"queued_use", "router_selected_queued_use"}:
        raise SystemExit(f"{path} is not queued-use data; metadata.label_mode={label_mode!r}")
    labels_to = meta.get("labels_aligned_to")
    if labels_to is not None and labels_to != "step_t_plus_1":
        raise SystemExit(f"{path} labels are not queued next-step labels: {labels_to!r}")
    return payload


def _evaluate_checkpoint(checkpoint: Path, data_path: Path, *, k: int, batch_size: int, device: str) -> dict[str, Any]:
    import torch

    from asts.mtp_heads import MTPHeadConfig, PLDMTPHeads

    data = _assert_queued(data_path)
    ckpt = torch.load(checkpoint, map_location="cpu")
    config = MTPHeadConfig(**ckpt["config"])
    output_weight = ckpt.get("output_weight")
    if output_weight is not None:
        output_weight = output_weight.to(device=device, dtype=torch.float32)
    model = PLDMTPHeads(config, output_weight=output_weight).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.prepare_for_inference()
    model.eval()
    trainable_params = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    total_params = sum(int(p.numel()) for p in model.parameters())
    hidden = data["hidden"].float()
    labels = data["labels"].long()[:, : min(k, config.num_heads)]
    pred_chunks = []
    top5_chunks = []
    losses = []
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    with torch.inference_mode():
        for start in range(0, int(hidden.shape[0]), batch_size):
            x = hidden[start : start + batch_size].to(device)
            y = labels[start : start + batch_size].to(device)
            logits_by_head = model(x)
            logits = torch.stack(logits_by_head[: labels.shape[1]], dim=1)
            losses.append(float(loss_fn(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).cpu().item()))
            pred_chunks.append(logits.argmax(dim=-1).cpu())
            top5_chunks.append(logits.topk(k=min(5, logits.shape[-1]), dim=-1).indices.cpu())
    pred = torch.cat(pred_chunks, dim=0) if pred_chunks else torch.empty_like(labels)
    top5 = torch.cat(top5_chunks, dim=0) if top5_chunks else torch.empty((*labels.shape, 0), dtype=torch.long)
    top1 = []
    top5_acc = []
    for h in range(labels.shape[1]):
        valid = labels[:, h] >= 0
        denom = max(1, int(valid.sum().item()))
        top1.append(float((pred[:, h][valid] == labels[:, h][valid]).float().sum().item() / denom))
        if top5.shape[-1] > 0:
            gold = labels[:, h].reshape(-1, 1, 1)
            hit = (top5[:, h : h + 1, :] == gold).any(dim=-1).reshape(-1)
            top5_acc.append(float(hit[valid].float().sum().item() / denom))
        else:
            top5_acc.append(0.0)
    return {
        "data": _rel(data_path),
        "examples": int(hidden.shape[0]),
        "loss": sum(losses) / max(1, len(losses)),
        "top1_accuracy_by_horizon": top1,
        "top5_accuracy_by_horizon": top5_acc,
        "h1_top1": top1[0] if top1 else 0.0,
        "trainable_parameter_count": trainable_params,
        "total_parameter_count_excluding_frozen_output": total_params,
        "output_weight_shape": list(output_weight.shape) if output_weight is not None else None,
    }


def _write_table(path: Path, payload: dict[str, Any]) -> None:
    val = payload.get("validation", {})
    lines = [
        "# Queued Head Training Gate",
        "",
        f"checkpoint: `{payload['checkpoint']}`",
        f"train data: `{payload['train']}`",
        f"val data: `{payload['val']}`",
        f"checkpoint size: `{payload.get('checkpoint_size_bytes', 0)}` bytes",
        f"trainable parameters: `{val.get('trainable_parameter_count', 0)}`",
        "",
        "| split | examples | loss | h1 top1 | h2 top1 | h4 top1 | h1 top5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in (("validation", val),):
        top1 = row.get("top1_accuracy_by_horizon") or []
        top5 = row.get("top5_accuracy_by_horizon") or []
        lines.append(
            f"| {name} | {row.get('examples', 0)} | {float(row.get('loss', 0.0)):.4f} | "
            f"{_pct(top1, 0)} | {_pct(top1, 1)} | {_pct(top1, 3)} | {_pct(top5, 0)} |"
        )
    lines.extend(["", f"Gate: **{payload['training_gate']}**"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _pct(values: list[float], idx: int) -> str:
    return "n/a" if idx >= len(values) else f"{100.0 * float(values[idx]):.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--val", type=Path, default=DEFAULT_VAL)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--head-type", choices=["linear", "mlp"], default="linear")
    ap.add_argument("--k", "--num-heads", dest="k", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    ap.add_argument("--output-projection", type=Path, default=None)
    ap.add_argument("--head-loss-weights", "--loss-weights", dest="loss_weights", default="")
    ap.add_argument("--hidden-dim", type=int, default=2048)
    ap.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    args = ap.parse_args()

    _assert_queued(args.train)
    _assert_queued(args.val)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "model.pt"
    cmd = [
        sys.executable,
        str(ROOT / "scripts/train_pld_mtp_heads.py"),
        "--data",
        str(args.train),
        "--output",
        str(checkpoint),
        "--target",
        args.target,
        "--num-heads",
        str(args.k),
        "--head-type",
        args.head_type,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--device",
        args.device,
        "--hidden-dim",
        str(args.hidden_dim),
    ]
    if args.output_projection is not None:
        cmd.extend(["--output-projection", str(args.output_projection)])
    if args.loss_weights:
        cmd.extend(["--loss-weights", args.loss_weights])
    config_payload = {
        "train": _rel(args.train),
        "val": _rel(args.val),
        "output_dir": _rel(args.output_dir),
        "checkpoint": _rel(checkpoint),
        "head_type": args.head_type,
        "k": int(args.k),
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "device": args.device,
        "target": args.target,
        "output_projection": _rel(args.output_projection) if args.output_projection is not None else None,
        "head_loss_weights": args.loss_weights,
        "hidden_dim": int(args.hidden_dim),
        "train_command": cmd,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config_payload, indent=2) + "\n")
    subprocess.run(cmd, cwd=ROOT, check=True)

    validation = _evaluate_checkpoint(checkpoint, args.val, k=args.k, batch_size=args.batch_size, device=args.device)
    payload = {
        "train": _rel(args.train),
        "val": _rel(args.val),
        "checkpoint": _rel(checkpoint),
        "command": cmd,
        "validation": validation,
        "checkpoint_size_bytes": checkpoint.stat().st_size if checkpoint.exists() else 0,
        "training_gate": "pass" if float(validation["h1_top1"]) >= 0.65 else "fail: h1 top-1 below 65%",
    }
    (args.output_dir / "queued_training_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_table(args.table, payload)
    print(args.table.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
