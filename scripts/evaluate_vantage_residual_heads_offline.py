"""Evaluate trained VANTAGE residual heads on held-out post-PLD tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asts.residual_mtp import (  # noqa: E402
    TRIGGER_POLICIES,
    ResidualMTPConfig,
    ResidualMTPHeads,
    accepted_prefix_lengths,
    accuracy_by_head,
    load_residual_tensor_payload,
    should_trigger_residual,
)
from asts.mtp_heads import MTPHeadConfig, PLDMTPHeads  # noqa: E402


DEFAULT_DATA = REPO_ROOT / "artifacts" / "vantage_residual" / "data" / "test500.pt"
DEFAULT_CHECKPOINT = REPO_ROOT / "artifacts" / "vantage_residual" / "models" / "residual_heads.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "vantage_residual" / "offline" / "eval"


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _tensor_bool_list(value: Any, n: int, fallback: bool = False) -> list[bool]:
    import torch

    if torch.is_tensor(value):
        return [bool(x) for x in value.reshape(-1)[:n].tolist()]
    if isinstance(value, (list, tuple)):
        return [bool(x) for x in value[:n]]
    return [fallback for _ in range(n)]


def _tensor_int_list(value: Any, n: int, fallback: int = 0) -> list[int]:
    import torch

    if torch.is_tensor(value):
        return [int(x) for x in value.reshape(-1)[:n].tolist()]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value[:n]]
    return [fallback for _ in range(n)]


def _apply_label_vocab(labels, label_vocab):
    import torch

    if not torch.is_tensor(label_vocab):
        return labels
    label_vocab = label_vocab.long()
    if int(label_vocab.numel()) == 0:
        return torch.full_like(labels.long(), -100)
    idx = torch.searchsorted(label_vocab, labels.clamp_min(0).long())
    clamped = idx.clamp_max(max(0, int(label_vocab.numel()) - 1))
    valid = (labels >= 0) & (label_vocab[clamped] == labels.long())
    out = torch.full_like(labels.long(), -100)
    out[valid] = idx[valid].long()
    return out


def _map_predictions_to_token_ids(pred, label_vocab):
    import torch

    if not torch.is_tensor(label_vocab):
        return pred
    vocab = label_vocab.long()
    out = torch.full_like(pred.long(), -1)
    valid = (pred >= 0) & (pred < int(vocab.numel()))
    out[valid] = vocab[pred[valid].long()]
    return out


def _coverage(labels, label_vocab) -> float:
    import torch

    if not torch.is_tensor(label_vocab):
        return 1.0
    valid = labels >= 0
    if int(valid.sum().item()) == 0:
        return 0.0
    return float(torch.isin(labels[valid], label_vocab.long()).float().mean().item())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-path", "--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--checkpoint", "--heads", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--baseline-steps", type=int, default=6443)
    args = ap.parse_args()

    missing = [str(path) for path in (args.data_path, args.checkpoint) if not path.exists()]
    if missing:
        _emit(
            {
                "status": "missing_input",
                "missing": missing,
                "note": "No evaluation was run. Provide the held-out tensor and residual-head checkpoint.",
            }
        )
        return

    import torch

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    raw_config = dict(ckpt["config"])
    output_weight = ckpt.get("output_weight")
    if "bottleneck" in raw_config or output_weight is not None:
        mtp_config = MTPHeadConfig(**raw_config)
        output_weight = (
            output_weight.to(device=args.device, dtype=torch.float32)
            if torch.is_tensor(output_weight)
            else None
        )
        model = PLDMTPHeads(mtp_config, output_weight=output_weight).to(args.device)
        num_heads = int(mtp_config.num_heads)
    else:
        config = ResidualMTPConfig(**raw_config)
        model = ResidualMTPHeads(config).to(args.device)
        num_heads = int(config.num_heads)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    if hasattr(model, "prepare_for_inference"):
        model.prepare_for_inference()

    payload = load_residual_tensor_payload(args.data_path, num_heads=num_heads)
    hidden = payload["hidden"].float()
    original_labels = payload["labels"].long()
    labels = _apply_label_vocab(original_labels, ckpt.get("label_vocab"))
    base_logits = payload.get("base_logits")
    predictions = []
    with torch.no_grad():
        for start in range(0, int(hidden.shape[0]), int(args.batch_size)):
            end = start + int(args.batch_size)
            base = base_logits[start:end].to(args.device) if torch.is_tensor(base_logits) else None
            if isinstance(model, PLDMTPHeads):
                logits = model.forward_logits(hidden[start:end].to(args.device))
            else:
                logits = model.forward_logits(hidden[start:end].to(args.device), base_logits=base)
            predictions.append(logits.argmax(dim=-1).cpu())
    pred = torch.cat(predictions, dim=0) if predictions else torch.empty_like(labels)
    pred_token_ids = _map_predictions_to_token_ids(pred, ckpt.get("label_vocab"))
    prefix = accepted_prefix_lengths(pred_token_ids, original_labels)
    prefix_hist = {str(i): int((prefix == i).sum().item()) for i in range(num_heads + 1)}
    accepted_len = _tensor_int_list(payload.get("accepted_len"), int(hidden.shape[0]), fallback=0)
    pld_miss = _tensor_bool_list(payload.get("pld_miss"), int(hidden.shape[0]), fallback=False)

    trigger_metrics = {}
    for policy in TRIGGER_POLICIES:
        mask_items = [
            should_trigger_residual(policy, accepted_len=a, pld_miss=m)
            for a, m in zip(accepted_len, pld_miss, strict=True)
        ]
        mask = torch.tensor(mask_items, dtype=torch.bool)
        count = int(mask.sum().item())
        selected_prefix = prefix[mask]
        extra_after_baseline = (selected_prefix - 1).clamp_min(0)
        projected_steps = int(args.baseline_steps) - int(extra_after_baseline.sum().item())
        projected_steps = max(1, projected_steps)
        trigger_metrics[policy] = {
            "count": count,
            "mean_accepted_prefix": float(prefix[mask].float().mean().item()) if count else 0.0,
            "top1_accuracy_by_horizon": accuracy_by_head(pred_token_ids[mask], original_labels[mask]) if count else [],
            "token0_reject_rate": float((selected_prefix == 0).float().mean().item()) if count else 0.0,
            "mean_extra_tokens_after_baseline": float(extra_after_baseline.float().mean().item()) if count else 0.0,
            "conservative_projected_saved_steps": int(extra_after_baseline.sum().item()),
            "conservative_projected_steps": projected_steps,
            "conservative_projected_speedup": float(args.baseline_steps) / float(projected_steps),
        }

    payload_out = {
        "status": "ok",
        "data_path": str(args.data_path),
        "checkpoint": str(args.checkpoint),
        "num_examples": int(hidden.shape[0]),
        "num_heads": int(num_heads),
        "baseline_steps": int(args.baseline_steps),
        "label_vocab_coverage": _coverage(original_labels, ckpt.get("label_vocab")),
        "top1_accuracy_by_horizon": accuracy_by_head(pred_token_ids, original_labels),
        "mean_accepted_prefix": float(prefix.float().mean().item()) if int(prefix.numel()) else 0.0,
        "accepted_prefix_histogram": prefix_hist,
        "trigger_metrics": trigger_metrics,
        "notes": [
            "Conservative projected speedup uses max(accepted_prefix-1, 0) as saved steps.",
            "This is an offline projection, not measured runtime throughput.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(payload_out, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "offline_projection.md").write_text(_markdown(payload_out), encoding="utf-8")
    _emit(payload_out)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# VANTAGE-Residual Offline Projection",
        "",
        f"checkpoint: `{payload['checkpoint']}`",
        f"data: `{payload['data_path']}`",
        f"examples: `{payload['num_examples']}`",
        f"label vocab coverage: `{payload['label_vocab_coverage']:.4f}`",
        "",
        "| Trigger | Calls | H1 top1 | Mean accepted prefix | Token0 reject | Saved steps | Projected speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy, row in payload.get("trigger_metrics", {}).items():
        h1 = (row.get("top1_accuracy_by_horizon") or [0.0])[0]
        lines.append(
            "| {p} | {c} | {h1:.3f} | {m:.3f} | {r:.3f} | {s} | {sp:.3f} |".format(
                p=policy,
                c=row.get("count", 0),
                h1=float(h1),
                m=float(row.get("mean_accepted_prefix", 0.0)),
                r=float(row.get("token0_reject_rate", 0.0)),
                s=int(row.get("conservative_projected_saved_steps", 0)),
                sp=float(row.get("conservative_projected_speedup", 1.0)),
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Offline projection only; not measured runtime throughput.",
            "- Saved steps are conservatively estimated as `max(accepted_prefix-1, 0)` per residual call.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
