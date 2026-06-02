"""Train tiny VANTAGE residual MTP heads on post-PLD tensors."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asts.residual_mtp import (  # noqa: E402
    ResidualMTPConfig,
    ResidualMTPHeads,
    accuracy_by_head,
    infer_vocab_size,
    load_residual_tensor_payload,
    residual_cross_entropy,
)


DEFAULT_TRAIN = REPO_ROOT / "artifacts" / "vantage_residual" / "data" / "train.pt"
DEFAULT_EVAL = REPO_ROOT / "artifacts" / "vantage_residual" / "data" / "test500.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "vantage_residual" / "checkpoints" / "linear_k4_v1"


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _map_predictions_to_token_ids(pred, label_vocab):
    import torch

    if not torch.is_tensor(label_vocab):
        return pred
    vocab = label_vocab.long()
    out = torch.full_like(pred.long(), -1)
    valid = (pred >= 0) & (pred < int(vocab.numel()))
    out[valid] = vocab[pred[valid].long()]
    return out


def _evaluate(
    model,
    hidden,
    train_labels,
    original_labels,
    base_logits,
    *,
    label_vocab,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    import torch

    model.eval()
    preds = []
    losses = []
    with torch.no_grad():
        for start in range(0, int(hidden.shape[0]), batch_size):
            end = start + batch_size
            x = hidden[start:end].to(device)
            y = train_labels[start:end].to(device)
            base = base_logits[start:end].to(device) if torch.is_tensor(base_logits) else None
            logits = model.forward_logits(x, base_logits=base)
            losses.append(float(residual_cross_entropy(logits, y).detach().cpu().item()))
            preds.append(logits.argmax(dim=-1).cpu())
    pred = torch.cat(preds, dim=0) if preds else torch.empty((0, train_labels.shape[1]), dtype=torch.long)
    pred_token_ids = _map_predictions_to_token_ids(pred, label_vocab)
    valid_labels = original_labels >= 0
    label_coverage = 1.0
    if torch.is_tensor(label_vocab):
        label_coverage = float(torch.isin(original_labels[valid_labels], label_vocab.long()).float().mean().item())
    return {
        "loss": sum(losses) / max(1, len(losses)),
        "top1_accuracy_by_horizon": accuracy_by_head(pred_token_ids, original_labels),
        "label_vocab_coverage": label_coverage,
    }


def _compact_labels(labels):
    import torch

    valid_tokens = labels[labels >= 0]
    if int(valid_tokens.numel()) == 0:
        return torch.full_like(labels, -100), torch.empty((0,), dtype=torch.long)
    vocab = torch.unique(valid_tokens).sort().values.long()
    idx = torch.searchsorted(vocab, labels.clamp_min(0).long())
    clamped = idx.clamp_max(max(0, int(vocab.numel()) - 1))
    valid = (labels >= 0) & (vocab[clamped] == labels.long())
    out = torch.full_like(labels.long(), -100)
    out[valid] = idx[valid].long()
    return out, vocab


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-path", "--data", "--train", dest="data_path", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--eval", "--eval-data", dest="eval_path", type=Path, default=DEFAULT_EVAL)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--num-heads", "--k", dest="num_heads", type=int, default=4)
    ap.add_argument("--head-type", choices=["linear", "mlp"], default="linear")
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--residual-scale", type=float, default=1.0)
    ap.add_argument("--vocab-size", type=int, default=0)
    ap.add_argument("--compact-label-vocab", dest="compact_label_vocab", action="store_true", default=True)
    ap.add_argument("--full-vocab", dest="compact_label_vocab", action="store_false")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--max-eval-examples", type=int, default=0)
    ap.add_argument(
        "--allow-large-vocab-head",
        action="store_true",
        help=(
            "Permit an untied hidden_size x vocab_size residual head. "
            "Without this, the script refuses configurations likely to OOM."
        ),
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or (args.output_dir / "model.pt")
    if not args.data_path.exists():
        _emit(
            {
                "status": "missing_data",
                "data_path": str(args.data_path),
                "note": "No training was run. Provide artifacts/vantage_residual/data/train.pt.",
            }
        )
        return

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(int(args.seed))
    payload = load_residual_tensor_payload(args.data_path, num_heads=args.num_heads)
    hidden = payload["hidden"].float()
    original_labels = payload["labels"].long()
    labels = original_labels.clone()
    base_logits = payload.get("base_logits")
    if args.max_examples > 0:
        n = min(int(args.max_examples), int(hidden.shape[0]))
        hidden = hidden[:n]
        labels = labels[:n]
        original_labels = original_labels[:n]
        if torch.is_tensor(base_logits):
            base_logits = base_logits[:n]
    label_vocab = None
    if args.compact_label_vocab and int(args.vocab_size) <= 0:
        labels, label_vocab = _compact_labels(labels)
        vocab_size = max(1, int(label_vocab.numel()))
    else:
        vocab_size = infer_vocab_size(labels, requested=int(args.vocab_size))
    config = ResidualMTPConfig(
        hidden_size=int(hidden.shape[-1]),
        vocab_size=vocab_size,
        num_heads=int(args.num_heads),
        head_type=args.head_type,
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        residual_scale=float(args.residual_scale),
    )
    estimated_params = int(config.hidden_size) * int(config.vocab_size) * int(config.num_heads)
    if estimated_params > 200_000_000 and not args.allow_large_vocab_head:
        _emit(
            {
                "status": "refused_large_untied_head",
                "data_path": str(args.data_path),
                "estimated_parameters": estimated_params,
                "hidden_size": int(config.hidden_size),
                "vocab_size": int(config.vocab_size),
                "num_heads": int(config.num_heads),
                "note": (
                    "This CPU/local trainer uses an untied classifier. Use the existing "
                    "PLD-MTP tied-output training path on GPU, pass a smaller --vocab-size "
                    "for smoke tests, or override with --allow-large-vocab-head."
                ),
            }
        )
        return
    model = ResidualMTPHeads(config).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    tensors = [hidden, labels]
    has_base = torch.is_tensor(base_logits)
    if has_base:
        tensors.append(base_logits.float())
    loader = DataLoader(TensorDataset(*tensors), batch_size=int(args.batch_size), shuffle=True)

    train_config = {
        "train": str(args.data_path),
        "eval": str(args.eval_path),
        "output": str(output_path),
        "num_heads": int(args.num_heads),
        "head_type": args.head_type,
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "residual_scale": float(args.residual_scale),
        "compact_label_vocab": bool(label_vocab is not None),
        "vocab_size": int(vocab_size),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "max_examples": int(args.max_examples),
        "max_eval_examples": int(args.max_eval_examples),
    }
    (args.output_dir / "config.json").write_text(json.dumps(train_config, indent=2, sort_keys=True) + "\n")

    reports = []
    for epoch in range(int(args.epochs)):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in loader:
            x = batch[0].to(args.device)
            y = batch[1].to(args.device)
            base = batch[2].to(args.device) if has_base else None
            logits = model.forward_logits(x, base_logits=base)
            loss = residual_cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu().item())
            batches += 1
        report = _evaluate(
            model,
            hidden,
            labels,
            original_labels,
            base_logits,
            label_vocab=label_vocab,
            batch_size=int(args.batch_size),
            device=args.device,
        )
        report["epoch"] = epoch + 1
        report["train_loss"] = total_loss / max(1, batches)
        reports.append(report)

    eval_report = None
    if args.eval_path.exists():
        eval_payload = load_residual_tensor_payload(args.eval_path, num_heads=args.num_heads)
        eval_hidden = eval_payload["hidden"].float()
        eval_original_labels = eval_payload["labels"].long()
        eval_labels = eval_original_labels.clone()
        if label_vocab is not None:
            eval_labels, _ = _labels_to_vocab(eval_original_labels, label_vocab)
        eval_base = eval_payload.get("base_logits")
        if args.max_eval_examples > 0:
            m = min(int(args.max_eval_examples), int(eval_hidden.shape[0]))
            eval_hidden = eval_hidden[:m]
            eval_labels = eval_labels[:m]
            eval_original_labels = eval_original_labels[:m]
            if torch.is_tensor(eval_base):
                eval_base = eval_base[:m]
        eval_report = _evaluate(
            model,
            eval_hidden,
            eval_labels,
            eval_original_labels,
            eval_base,
            label_vocab=label_vocab,
            batch_size=int(args.batch_size),
            device=args.device,
        )

    checkpoint = {
        "model_state": model.state_dict(),
        "config": asdict(config),
        "metadata": {
            "data_path": str(args.data_path),
            "eval_path": str(args.eval_path),
            "n_examples": int(hidden.shape[0]),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "has_base_logits": bool(has_base),
            "compact_label_vocab": bool(label_vocab is not None),
            "label_vocab_size": int(label_vocab.numel()) if label_vocab is not None else 0,
            "reports": reports,
            "eval_report": eval_report,
        },
    }
    if label_vocab is not None:
        checkpoint["label_vocab"] = label_vocab
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    report_path = args.output_dir / "metrics.json"
    report_path.write_text(json.dumps(checkpoint["metadata"], indent=2, sort_keys=True) + "\n")
    # Backward-compatible sidecar beside the checkpoint.
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(checkpoint["metadata"], indent=2, sort_keys=True) + "\n"
    )
    _emit({"status": "ok", "output": str(output_path), **checkpoint["metadata"]})


def _labels_to_vocab(labels, vocab):
    import torch

    vocab = vocab.long()
    if int(vocab.numel()) == 0:
        return torch.full_like(labels.long(), -100), vocab
    idx = torch.searchsorted(vocab, labels.clamp_min(0).long())
    clamped = idx.clamp_max(max(0, int(vocab.numel()) - 1))
    valid = (labels >= 0) & (vocab[clamped] == labels.long())
    out = torch.full_like(labels.long(), -100)
    out[valid] = idx[valid].long()
    return out, vocab


if __name__ == "__main__":
    main()
