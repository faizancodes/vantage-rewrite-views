"""Train offline MTP heads on frozen hidden states from PLD traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asts.mtp_heads import MTPHeadConfig, PLDMTPHeads  # noqa: E402


def _parse_head_loss_weights(raw: str, num_heads: int) -> list[float]:
    if not raw:
        return [1.0 for _ in range(num_heads)]
    weights = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(weights) != num_heads:
        raise SystemExit(
            f"--head-loss-weights must provide exactly {num_heads} values, got {len(weights)}"
        )
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise SystemExit("--head-loss-weights must be non-negative and have positive sum")
    return weights


def _accuracy(pred, labels, mask):
    if int(mask.sum().item()) == 0:
        return 0.0
    return float((pred[mask] == labels[mask]).float().mean().item())


def _accuracy_by_head(pred_by_head, labels, mask):
    out = []
    for i, pred in enumerate(pred_by_head):
        out.append(_accuracy(pred, labels[:, i], (labels[:, i] >= 0) & mask))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--head-type", choices=["linear", "mlp"], default="linear")
    ap.add_argument("--hidden-dim", type=int, default=2048)
    ap.add_argument("--bottleneck", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vocab-size", type=int, default=0)
    ap.add_argument("--output-projection", type=Path, default=None)
    ap.add_argument("--projection-dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--head-loss-weights", "--loss-weights", dest="head_loss_weights", default="")
    ap.add_argument(
        "--init-heads",
        type=Path,
        default=None,
        help="Optional MTP checkpoint to initialize head weights from before fine-tuning.",
    )
    ap.add_argument(
        "--max-train-examples",
        type=int,
        default=0,
        help="Debug/ablation option: train and report on only the first N examples.",
    )
    ap.add_argument(
        "--eval-train-subset",
        action="store_true",
        help="Record that metrics are intentionally evaluated on the training subset.",
    )
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoConfig, AutoModelForCausalLM

    data = torch.load(args.data, map_location="cpu")
    hidden = data["hidden"].float()
    labels = data["labels"].long()[:, : args.num_heads]
    if args.max_train_examples > 0:
        n = min(int(args.max_train_examples), int(hidden.shape[0]))
        hidden = hidden[:n]
        labels = labels[:n]
    accepted_len = data.get("accepted_len", torch.zeros(hidden.shape[0], dtype=torch.long)).long()
    if args.max_train_examples > 0:
        accepted_len = accepted_len[: hidden.shape[0]]
    eq0 = accepted_len == 0
    le1 = accepted_len <= 1
    le2 = accepted_len <= 2
    le4 = accepted_len <= 4
    miss = data.get("pld_miss", torch.zeros(hidden.shape[0], dtype=torch.bool)).bool()
    if args.max_train_examples > 0:
        miss = miss[: hidden.shape[0]]
    router_prob = data.get("router_probability")
    if router_prob is not None:
        router_prob = router_prob.float()[: hidden.shape[0]]
    router_tp = data.get("router_true_positive")
    if router_tp is not None:
        router_tp = router_tp.bool()[: hidden.shape[0]]
    router_fp = data.get("router_false_positive")
    if router_fp is not None:
        router_fp = router_fp.bool()[: hidden.shape[0]]

    output_weight = None
    init_payload = None
    if args.init_heads is not None:
        init_payload = torch.load(args.init_heads, map_location="cpu")
        if not isinstance(init_payload, dict) or "model_state" not in init_payload:
            raise SystemExit(f"--init-heads {args.init_heads} is not an MTP head checkpoint")
    if args.output_projection is not None:
        obj = torch.load(args.output_projection, map_location="cpu")
        output_weight = obj["output_weight"] if isinstance(obj, dict) else obj
        vocab_size = int(output_weight.shape[0])
    elif init_payload is not None and init_payload.get("output_weight") is not None:
        output_weight = init_payload["output_weight"]
        vocab_size = int(output_weight.shape[0])
    elif args.vocab_size > 0:
        vocab_size = args.vocab_size
    else:
        dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[args.projection_dtype]
        target = AutoModelForCausalLM.from_pretrained(
            args.target,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        emb = target.get_output_embeddings()
        if emb is None:
            raise SystemExit(f"{args.target} does not expose output embeddings")
        output_weight = emb.weight.detach().cpu()
        vocab_size = int(output_weight.shape[0])
        del target
    if output_weight is None and args.vocab_size <= 0:
        cfg = AutoConfig.from_pretrained(args.target, trust_remote_code=True)
        vocab_size = int(getattr(cfg, "vocab_size", int(labels[labels >= 0].max().item()) + 1))
    if output_weight is not None:
        # Training hidden states and adapter weights are float32. Keep the
        # frozen LM-head projection in the same dtype so the tied-output matmul
        # is stable and does not fail when the saved projection came from a
        # bf16/fp16 target model.
        output_weight = output_weight.to(device=args.device, dtype=torch.float32)
    model = PLDMTPHeads(
        MTPHeadConfig(
            hidden_size=int(hidden.shape[-1]),
            vocab_size=vocab_size,
            num_heads=args.num_heads,
            head_type=args.head_type,
            hidden_dim=args.hidden_dim,
            bottleneck=args.bottleneck,
            dropout=args.dropout,
        ),
        output_weight=output_weight,
    ).to(args.device)
    if init_payload is not None:
        init_config = dict(init_payload.get("config") or {})
        expected = model.config.__dict__
        comparable_keys = ("hidden_size", "vocab_size", "num_heads", "head_type")
        mismatches = {
            key: (init_config.get(key), expected.get(key))
            for key in comparable_keys
            if init_config.get(key) != expected.get(key)
        }
        if mismatches:
            raise SystemExit(f"--init-heads config mismatch: {mismatches}")
        missing, unexpected = model.load_state_dict(init_payload["model_state"], strict=False)
        if missing or unexpected:
            raise SystemExit(
                f"--init-heads state mismatch: missing={list(missing)} unexpected={list(unexpected)}"
            )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    head_loss_weights = torch.tensor(
        _parse_head_loss_weights(args.head_loss_weights, args.num_heads),
        dtype=torch.float32,
        device=args.device,
    )
    ds = TensorDataset(hidden, labels, le4.bool(), miss.bool())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    epoch_reports = []
    for epoch in range(args.epochs):
        model.train()
        loss_sums = [0.0 for _ in range(args.num_heads)]
        batches = 0
        for x, y, _, _ in loader:
            x = x.to(args.device)
            y = y.to(args.device)
            logits = model(x)
            losses = [loss_fn(logits[i], y[:, i]) for i in range(args.num_heads)]
            loss = sum(losses[i] * head_loss_weights[i] for i in range(args.num_heads)) / head_loss_weights.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            for i, item in enumerate(losses):
                loss_sums[i] += float(item.detach().cpu())
            batches += 1

        model.eval()
        acc = []
        with torch.no_grad():
            logits = []
            for start in range(0, hidden.shape[0], args.batch_size):
                logits.append(model(hidden[start : start + args.batch_size].to(args.device)))
            by_head = [
                torch.cat([chunk[i].argmax(dim=-1).cpu() for chunk in logits], dim=0)
                for i in range(args.num_heads)
            ]
            for i, pred in enumerate(by_head):
                mask = labels[:, i] >= 0
                acc.append(_accuracy(pred, labels[:, i], mask))
        acc_eq0 = _accuracy_by_head(by_head, labels, eq0)
        acc_le1 = _accuracy_by_head(by_head, labels, le1)
        acc_le2 = _accuracy_by_head(by_head, labels, le2)
        acc_le4 = _accuracy_by_head(by_head, labels, le4)
        acc_miss = _accuracy_by_head(by_head, labels, miss)
        acc_router_tp = _accuracy_by_head(by_head, labels, router_tp) if router_tp is not None else []
        acc_router_fp = _accuracy_by_head(by_head, labels, router_fp) if router_fp is not None else []
        router_bucket_acc = {}
        if router_prob is not None:
            for lo in [i / 10 for i in range(10)]:
                hi = min(1.0, lo + 0.1)
                mask = (router_prob >= lo) & ((router_prob < hi) | ((lo == 0.9) & (router_prob <= 1.0)))
                router_bucket_acc[f"{lo:.1f}-{hi:.1f}"] = {
                    "n": int(mask.sum().item()),
                    "top1_accuracy_per_horizon": _accuracy_by_head(by_head, labels, mask),
                }

        epoch_reports.append(
            {
                "epoch": epoch + 1,
                "loss_per_head": [v / max(1, batches) for v in loss_sums],
                "top1_accuracy_per_horizon": acc,
                "top1_accuracy_t_plus_1": acc[0] if len(acc) > 0 else 0.0,
                "top1_accuracy_t_plus_2": acc[1] if len(acc) > 1 else 0.0,
                "top1_accuracy_t_plus_3": acc[2] if len(acc) > 2 else 0.0,
                "top1_accuracy_t_plus_4": acc[3] if len(acc) > 3 else 0.0,
                "top1_accuracy_accepted_len_eq_0": acc_eq0,
                "top1_accuracy_accepted_len_le_1": acc_le1,
                "top1_accuracy_accepted_len_le_2": acc_le2,
                "top1_accuracy_accepted_len_le_4": acc_le4,
                "top1_accuracy_pld_miss_regions": acc_miss,
                "top1_accuracy_router_true_positive": acc_router_tp,
                "top1_accuracy_router_false_positive": acc_router_fp,
                "top1_accuracy_by_router_probability_bucket": router_bucket_acc,
            }
        )

    payload = {
        "model_state": model.state_dict(),
        "config": model.config.__dict__,
        "metadata": {
            "target": args.target,
            "data": str(args.data),
            "n_examples": int(hidden.shape[0]),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "head_type": args.head_type,
            "hidden_dim": args.hidden_dim,
            "head_loss_weights": [float(x) for x in head_loss_weights.detach().cpu().tolist()],
            "uses_tied_output_projection": output_weight is not None,
            "init_heads": str(args.init_heads) if args.init_heads is not None else None,
            "max_train_examples": int(args.max_train_examples),
            "eval_train_subset": bool(args.eval_train_subset),
            "reports": epoch_reports,
        },
    }
    if output_weight is not None:
        payload["output_weight"] = output_weight.detach().cpu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    report_path.write_text(json.dumps(payload["metadata"], indent=2) + "\n")
    print(json.dumps(payload["metadata"], indent=2))
    print(f"saved heads to {args.output}")


if __name__ == "__main__":
    main()
