"""Evaluate a trained VANTAGE residual-trigger router."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asts.vantage_residual_router import (  # noqa: E402
    ResidualRouter,
    ResidualRouterConfig,
    binary_metrics,
    load_router_payload,
)


DEFAULT_DATA = REPO_ROOT / "artifacts" / "vantage_residual" / "data" / "test500.pt"
DEFAULT_CHECKPOINT = REPO_ROOT / "artifacts" / "vantage_residual" / "models" / "residual_router.pt"


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-path", "--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--checkpoint", "--router", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    missing = [str(path) for path in (args.data_path, args.checkpoint) if not path.exists()]
    if missing:
        _emit(
            {
                "status": "missing_input",
                "missing": missing,
                "note": "No evaluation was run. Provide the held-out tensor and router checkpoint.",
            }
        )
        return

    import torch

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    feature_names = tuple(ckpt.get("feature_names") or ())
    payload = load_router_payload(
        args.data_path,
        feature_names=feature_names,
        target_policy=str(ckpt.get("target_policy") or "accepted_len_le_4"),
    )
    X = payload["features"].float()
    y = payload["labels"].float()
    config = ResidualRouterConfig(**ckpt["config"])
    if int(X.shape[1]) != int(config.feature_dim):
        raise SystemExit(f"feature_dim mismatch: data={X.shape[1]} checkpoint={config.feature_dim}")
    norm = ckpt.get("normalization") or {}
    mean = norm.get("mean")
    std = norm.get("std")
    if torch.is_tensor(mean) and torch.is_tensor(std):
        X = (X - mean.float()) / std.float().clamp_min(1e-6)
    model = ResidualRouter(config).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        probs = model.forward_proba(X.to(args.device)).cpu()
    metrics = binary_metrics(probs, y, threshold=float(args.threshold))
    feature_names = list(payload["feature_names"])
    shown_features = feature_names if len(feature_names) <= 32 else [*feature_names[:32], "..."]
    _emit(
        {
            "status": "ok",
            "data_path": str(args.data_path),
            "checkpoint": str(args.checkpoint),
            "target_policy": str(ckpt.get("target_policy") or "accepted_len_le_4"),
            "num_examples": int(X.shape[0]),
            "feature_count": len(feature_names),
            "feature_names": shown_features,
            "threshold": float(args.threshold),
            "metrics": metrics,
        }
    )


if __name__ == "__main__":
    main()
