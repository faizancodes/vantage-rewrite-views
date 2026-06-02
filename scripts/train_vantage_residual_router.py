"""Train a tiny VANTAGE residual-trigger router."""

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

from asts.vantage_residual_router import (  # noqa: E402
    DEFAULT_ROUTER_FEATURES,
    ResidualRouter,
    ResidualRouterConfig,
    binary_metrics,
    load_router_payload,
    normalize_features,
    router_loss,
)
from asts.residual_mtp import TRIGGER_POLICIES, normalize_trigger_policy  # noqa: E402


DEFAULT_DATA = REPO_ROOT / "artifacts" / "vantage_residual" / "data" / "train.pt"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "vantage_residual" / "models" / "residual_router.pt"


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-path", "--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--target-policy", choices=TRIGGER_POLICIES, default="accepted_len_le_4")
    ap.add_argument("--model-type", choices=["linear", "mlp"], default="linear")
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-examples", type=int, default=0)
    args = ap.parse_args()

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
    target_policy = normalize_trigger_policy(args.target_policy)
    payload = load_router_payload(
        args.data_path,
        feature_names=DEFAULT_ROUTER_FEATURES,
        target_policy=target_policy,
    )
    X = payload["features"].float()
    y = payload["labels"].float()
    if args.max_examples > 0:
        n = min(int(args.max_examples), int(X.shape[0]))
        X = X[:n]
        y = y[:n]
    X_norm, _, mean, std = normalize_features(X)
    config = ResidualRouterConfig(
        feature_dim=int(X_norm.shape[1]),
        hidden_dim=int(args.hidden_dim),
        model_type=args.model_type,
        dropout=float(args.dropout),
    )
    model = ResidualRouter(config).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loader = DataLoader(TensorDataset(X_norm, y), batch_size=int(args.batch_size), shuffle=True)

    reports = []
    for epoch in range(int(args.epochs)):
        model.train()
        loss_total = 0.0
        batches = 0
        for xb, yb in loader:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            logits = model.forward_logits(xb)
            loss = router_loss(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_total += float(loss.detach().cpu().item())
            batches += 1
        model.eval()
        with torch.no_grad():
            probs = model.forward_proba(X_norm.to(args.device)).cpu()
        report = binary_metrics(probs, y, threshold=float(args.threshold))
        report.update({"epoch": epoch + 1, "loss": loss_total / max(1, batches)})
        reports.append(report)

    checkpoint = {
        "model_state": model.state_dict(),
        "config": asdict(config),
        "feature_names": tuple(payload["feature_names"]),
        "target_policy": target_policy,
        "normalization": {"mean": mean, "std": std},
        "metadata": {
            "data_path": str(args.data_path),
            "n_examples": int(X.shape[0]),
            "positive_rate": float(y.mean().item()) if int(y.numel()) else 0.0,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "threshold": float(args.threshold),
            "reports": reports,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    report_path.write_text(json.dumps(checkpoint["metadata"], indent=2, sort_keys=True) + "\n")
    _emit({"status": "ok", "output": str(args.output), **checkpoint["metadata"]})


if __name__ == "__main__":
    main()
