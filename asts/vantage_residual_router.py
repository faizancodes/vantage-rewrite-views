"""Tiny residual-trigger router utilities for VANTAGE residual studies."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from asts.residual_mtp import normalize_trigger_policy, should_trigger_residual


DEFAULT_ROUTER_FEATURES = (
    "accepted_len",
    "accepted_len_eq_0",
    "accepted_len_le_1",
    "accepted_len_le_2",
    "accepted_len_le_4",
    "pld_miss",
    "proposal_len",
    "draft_len",
    "prefix_len",
    "step",
)


@dataclass(frozen=True)
class ResidualRouterConfig:
    feature_dim: int
    hidden_dim: int = 32
    model_type: str = "linear"
    dropout: float = 0.0


class ResidualRouter(nn.Module):
    """Small binary classifier deciding whether to invoke residual heads."""

    def __init__(self, config: ResidualRouterConfig) -> None:
        super().__init__()
        if config.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if config.model_type not in {"linear", "mlp"}:
            raise ValueError(f"unsupported router model_type: {config.model_type!r}")
        self.config = config
        if config.model_type == "linear":
            self.net = nn.Linear(config.feature_dim, 1)
        else:
            if config.hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive for mlp router")
            self.net = nn.Sequential(
                nn.LayerNorm(config.feature_dim),
                nn.Linear(config.feature_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, 1),
            )

    def forward_logits(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"features must be [batch, feature_dim], got {tuple(features.shape)}")
        return self.net(features.float()).squeeze(-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward_logits(features)

    def forward_proba(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(features))

    def predict(self, features: torch.Tensor, *, threshold: float = 0.5) -> torch.Tensor:
        return self.forward_proba(features) >= float(threshold)


def feature_dict_from_row(row: Mapping[str, Any]) -> dict[str, float]:
    """Extract a compact feature dictionary from tensor or JSONL-style rows."""

    accepted = int(_lookup_float(row, ("accepted_len", "n_accepted_drafts", "pld_accepted_len"), 0.0))
    proposal_len = _lookup_float(row, ("proposal_len", "proposal_tokens", "k", "draft_len"), 0.0)
    draft_len = _lookup_float(row, ("draft_len", "k", "proposal_tokens", "proposal_len"), proposal_len)
    prefix_len = _lookup_float(
        row,
        ("prefix_len", "prompt_len", "generated_start", "_generated_start", "generated_len"),
        0.0,
    )
    step = _lookup_float(row, ("step", "step_id", "index"), 0.0)
    miss_raw = _lookup(row, ("pld_miss", "is_pld_miss"), None)
    if miss_raw is None:
        pld_miss = 1.0 if accepted == 0 else 0.0
    else:
        pld_miss = 1.0 if _safe_bool(miss_raw) else 0.0
    return {
        "accepted_len": float(accepted),
        "accepted_len_eq_0": 1.0 if accepted == 0 else 0.0,
        "accepted_len_le_1": 1.0 if accepted <= 1 else 0.0,
        "accepted_len_le_2": 1.0 if accepted <= 2 else 0.0,
        "accepted_len_le_4": 1.0 if accepted <= 4 else 0.0,
        "pld_miss": pld_miss,
        "proposal_len": float(proposal_len),
        "draft_len": float(draft_len),
        "prefix_len": float(prefix_len),
        "step": float(step),
    }


def label_from_row(row: Mapping[str, Any], *, target_policy: str = "accepted_len_le_4") -> int:
    return int(should_trigger_residual(target_policy, row))


def build_feature_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_names: Sequence[str] = DEFAULT_ROUTER_FEATURES,
) -> torch.Tensor:
    matrix = []
    for row in rows:
        features = feature_dict_from_row(row)
        matrix.append([float(features.get(name, 0.0)) for name in feature_names])
    if not matrix:
        return torch.empty((0, len(tuple(feature_names))), dtype=torch.float32)
    return torch.tensor(matrix, dtype=torch.float32)


def build_labeled_feature_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_names: Sequence[str] = DEFAULT_ROUTER_FEATURES,
    target_policy: str = "accepted_len_le_4",
) -> tuple[torch.Tensor, torch.Tensor]:
    X = build_feature_matrix(rows, feature_names=feature_names)
    y = torch.tensor([label_from_row(row, target_policy=target_policy) for row in rows], dtype=torch.float32)
    return X, y


def load_router_payload(
    path: Path,
    *,
    feature_names: Sequence[str] = DEFAULT_ROUTER_FEATURES,
    target_policy: str = "accepted_len_le_4",
) -> dict[str, Any]:
    """Load row, feature-matrix, or columnar tensor router artifacts."""

    obj = torch.load(path, map_location="cpu")
    rows: list[Mapping[str, Any]] = []
    stored_feature_names: Sequence[str] = feature_names
    X: torch.Tensor | None = None
    y: torch.Tensor | None = None

    if isinstance(obj, Mapping):
        stored_feature_names = tuple(obj.get("feature_names") or feature_names)
        raw_features = _lookup(obj, ("router_features", "features", "X", "x"), None)
        raw_labels = _lookup(obj, ("router_labels", "trigger_labels", "binary_labels", "y"), None)
        if torch.is_tensor(raw_features):
            X = raw_features.float()
            if len(tuple(stored_feature_names)) != int(X.shape[1]):
                stored_feature_names = tuple(f"feature_{i}" for i in range(int(X.shape[1])))
        elif isinstance(raw_features, Sequence) and raw_features and isinstance(raw_features[0], Mapping):
            rows = list(raw_features)
        raw_rows = _lookup(obj, ("rows", "steps", "meta", "metadata_rows"), None)
        if isinstance(raw_rows, Sequence) and raw_rows and isinstance(raw_rows[0], Mapping):
            rows = list(raw_rows)
        if raw_labels is not None and torch.is_tensor(raw_labels):
            y = raw_labels.float().reshape(-1)
        elif raw_labels is not None and _is_scalar_sequence(raw_labels):
            y = torch.tensor([float(item) for item in raw_labels], dtype=torch.float32)
        if not rows and (X is None or raw_labels is None):
            rows = _columnar_rows(obj)
    elif isinstance(obj, Sequence) and obj and isinstance(obj[0], Mapping):
        rows = list(obj)
    else:
        raise ValueError(f"{path} is not a supported router artifact")

    if X is None:
        X = build_feature_matrix(rows, feature_names=stored_feature_names)
    if y is None and rows:
        y = torch.tensor([label_from_row(row, target_policy=target_policy) for row in rows], dtype=torch.float32)
    if y is None:
        raise ValueError(f"{path} does not contain router labels or accepted_len columns")
    if int(X.shape[0]) != int(y.shape[0]):
        raise ValueError(f"feature/label length mismatch: X={tuple(X.shape)} y={tuple(y.shape)}")
    return {
        "features": X.float(),
        "labels": y.float(),
        "feature_names": tuple(stored_feature_names),
        "rows": rows,
    }


def router_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits.reshape(-1), labels.float().reshape(-1))


def binary_metrics(probabilities: torch.Tensor, labels: torch.Tensor, *, threshold: float = 0.5) -> dict[str, float]:
    probs = probabilities.detach().float().reshape(-1)
    y = labels.detach().float().reshape(-1) >= 0.5
    pred = probs >= float(threshold)
    total = max(1, int(y.numel()))
    tp = int((pred & y).sum().item())
    fp = int((pred & ~y).sum().item())
    fn = int((~pred & y).sum().item())
    tn = int((~pred & ~y).sum().item())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_rate": int(y.sum().item()) / total,
        "predicted_positive_rate": int(pred.sum().item()) / total,
        "true_positive": float(tp),
        "false_positive": float(fp),
        "false_negative": float(fn),
        "true_negative": float(tn),
    }


def summarize_trigger_policy(
    rows: Sequence[Mapping[str, Any]],
    policy: str,
    *,
    weak_policy: str = "accepted_len_le_4",
) -> dict[str, float | int | str]:
    """Report trigger precision/recall against a PLD-weak definition."""

    triggered = [should_trigger_residual(policy, row) for row in rows]
    actual = [should_trigger_residual(weak_policy, row) for row in rows]
    tp = sum(1 for pred, gold in zip(triggered, actual, strict=True) if pred and gold)
    fp = sum(1 for pred, gold in zip(triggered, actual, strict=True) if pred and not gold)
    fn = sum(1 for pred, gold in zip(triggered, actual, strict=True) if (not pred) and gold)
    total = max(1, len(rows))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "policy": policy,
        "weak_policy": weak_policy,
        "rows": len(rows),
        "trigger_count": sum(1 for item in triggered if item),
        "actual_weak_count": sum(1 for item in actual if item),
        "trigger_rate": sum(1 for item in triggered if item) / total,
        "precision": precision,
        "recall": recall,
        "f1": 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall),
    }


def normalize_features(
    train: torch.Tensor,
    eval_features: torch.Tensor | None = None,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    mean = train.float().mean(dim=0)
    std = train.float().std(dim=0, unbiased=False).clamp_min(eps)
    train_norm = (train.float() - mean) / std
    eval_norm = None if eval_features is None else (eval_features.float() - mean) / std
    return train_norm, eval_norm, mean, std


def _columnar_rows(mapping: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidate_keys = (
        "accepted_len",
        "n_accepted_drafts",
        "pld_miss",
        "is_pld_miss",
        "proposal_len",
        "proposal_tokens",
        "k",
        "draft_len",
        "prefix_len",
        "prompt_len",
        "generated_start",
        "_generated_start",
        "step",
        "step_id",
    )
    length = 0
    for key in candidate_keys:
        value = mapping.get(key)
        if torch.is_tensor(value) and value.ndim >= 1:
            length = int(value.shape[0])
            break
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            length = len(value)
            break
    rows: list[Mapping[str, Any]] = []
    for i in range(length):
        row: dict[str, Any] = {}
        for key in candidate_keys:
            value = mapping.get(key)
            if torch.is_tensor(value) and value.ndim >= 1 and int(value.shape[0]) > i:
                row[key] = value[i]
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) > i:
                row[key] = value[i]
        rows.append(row)
    return rows


def _lookup(mapping: Mapping[str, Any], keys: Sequence[str], default: Any) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _lookup_float(mapping: Mapping[str, Any], keys: Sequence[str], default: float) -> float:
    return _safe_float(_lookup(mapping, keys, default), default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if torch.is_tensor(value):
        if value.numel() != 1:
            return default
        value = value.item()
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(out) or math.isinf(out) else out


def _safe_bool(value: Any) -> bool:
    if torch.is_tensor(value):
        if value.numel() != 1:
            return False
        value = value.item()
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _is_scalar_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and (
        not value or not isinstance(value[0], Mapping)
    )
