"""Residual MTP heads for VANTAGE-Residual.

These heads are scoped to post-PLD residual examples: PLD handles the copyable
prefix, then the tiny residual model proposes the next few tokens that PLD did
not cover.  Runtime use must still verify residual drafts with the target
model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .mtp_heads import MTPHeadConfig, PLDMTPHeads, accepted_prefix_length


TRIGGER_POLICIES = (
    "accepted_len_eq_0",
    "accepted_len_le_1",
    "accepted_len_le_2",
    "accepted_len_le_4",
    "pld_miss_only",
    "router_predicted_weak",
    "router",
    "always",
    "never",
)

_TRIGGER_ALIASES = {
    "eq_0": "accepted_len_eq_0",
    "eq0": "accepted_len_eq_0",
    "accepted_len_le_0": "accepted_len_eq_0",
    "le_1": "accepted_len_le_1",
    "le1": "accepted_len_le_1",
    "le_2": "accepted_len_le_2",
    "le2": "accepted_len_le_2",
    "le_4": "accepted_len_le_4",
    "le4": "accepted_len_le_4",
    "pld_miss": "pld_miss_only",
    "miss": "pld_miss_only",
    "weak": "accepted_len_le_4",
    "weak4": "accepted_len_le_4",
    "router_weak": "router",
}

RESIDUAL_REQUIRED_KEYS = {
    "hidden",
    "labels",
    "accepted_len",
    "trigger_accepted_len_eq_0",
    "trigger_accepted_len_le_1",
    "trigger_accepted_len_le_2",
    "trigger_accepted_len_le_4",
}


@dataclass(frozen=True)
class ResidualMTPConfig:
    """Runtime/training configuration for a tiny residual drafter."""

    hidden_size: int = 0
    vocab_size: int = 0
    num_heads: int = 4
    head_type: str = "linear"
    hidden_dim: int = 256
    dropout: float = 0.0
    residual_scale: float = 1.0
    trigger_policy: str = "accepted_len_le_4"
    mtp_position: str = "post_pld"
    checkpoint_path: str = "/data/pld_mtp/postpld_linear_k4_n917_v1/postpld_mtp_heads_k4_linear.pt"

    def validate(self) -> None:
        if self.num_heads < 1:
            raise ValueError("num_heads must be positive")
        if self.head_type not in {"linear", "mlp"}:
            raise ValueError(f"unsupported residual head_type: {self.head_type!r}")
        if self.mtp_position != "post_pld":
            raise ValueError("VANTAGE-Residual requires post_pld examples")
        if normalize_trigger_policy(self.trigger_policy) not in TRIGGER_POLICIES:
            raise ValueError(f"unsupported residual trigger_policy: {self.trigger_policy!r}")


@dataclass(frozen=True)
class ResidualTriggerStep:
    accepted_len: int = 0
    pld_miss: bool = False


class ResidualMTPHeads(nn.Module):
    """Tiny horizon-specific residual heads.

    If ``base_logits`` are supplied, outputs are ``base_logits + residual``.
    Without baseline logits, the residuals are used directly as token logits.
    """

    def __init__(self, config: ResidualMTPConfig) -> None:
        super().__init__()
        config.validate()
        if int(config.hidden_size) <= 0:
            raise ValueError("hidden_size must be positive")
        if int(config.vocab_size) <= 0:
            raise ValueError("vocab_size must be positive")
        self.config = config
        if config.head_type == "linear":
            self.heads = nn.ModuleList(
                [nn.Linear(config.hidden_size, config.vocab_size) for _ in range(config.num_heads)]
            )
        else:
            dim = int(config.hidden_dim)
            if dim <= 0:
                raise ValueError("hidden_dim must be positive for mlp heads")
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(config.hidden_size),
                        nn.Linear(config.hidden_size, dim),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(dim, config.vocab_size),
                    )
                    for _ in range(config.num_heads)
                ]
            )

    def prepare_for_inference(self) -> "ResidualMTPHeads":
        return self

    def forward_logits(
        self,
        hidden: torch.Tensor,
        base_logits: torch.Tensor | Sequence[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if hidden.ndim != 2:
            raise ValueError(f"hidden must be [batch, hidden], got {tuple(hidden.shape)}")
        residual = torch.stack([head(hidden) for head in self.heads], dim=1)
        if base_logits is None:
            return residual
        base = _coerce_base_logits(
            base_logits,
            num_heads=int(self.config.num_heads),
            vocab_size=int(self.config.vocab_size),
            device=residual.device,
            dtype=residual.dtype,
        )
        return base + float(self.config.residual_scale) * residual

    def forward(
        self,
        hidden: torch.Tensor,
        base_logits: torch.Tensor | Sequence[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        logits = self.forward_logits(hidden, base_logits=base_logits)
        return [logits[:, i, :] for i in range(logits.shape[1])]

    def predict_token_tensor(
        self,
        hidden: torch.Tensor,
        base_logits: torch.Tensor | Sequence[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        with torch.inference_mode():
            return self.forward_logits(hidden, base_logits=base_logits).argmax(dim=-1)


def validate_residual_dataset(payload: dict[str, Any]) -> None:
    """Validate that a tensor payload is a residual, not all-token, dataset."""

    missing = sorted(RESIDUAL_REQUIRED_KEYS - set(payload))
    if missing:
        raise ValueError(f"residual dataset missing required keys: {missing}")
    metadata = payload.get("metadata") or {}
    mtp_position = metadata.get("mtp_position")
    if mtp_position is not None and mtp_position != "post_pld":
        raise ValueError(f"expected post_pld residual data, got {mtp_position!r}")
    hidden = payload["hidden"]
    labels = payload["labels"]
    if not hasattr(hidden, "shape") or not hasattr(labels, "shape"):
        raise ValueError("hidden and labels must be tensors")
    if int(hidden.shape[0]) != int(labels.shape[0]):
        raise ValueError("hidden and labels row counts differ")


def trigger_mask(payload: dict[str, Any], policy: str) -> torch.Tensor:
    """Return the residual-example mask for a trigger policy."""

    validate_residual_dataset(payload)
    policy = normalize_trigger_policy(policy)
    accepted_len = payload["accepted_len"].long()
    if policy == "accepted_len_eq_0":
        return accepted_len == 0
    if policy == "accepted_len_le_1":
        return accepted_len <= 1
    if policy == "accepted_len_le_2":
        return accepted_len <= 2
    if policy == "accepted_len_le_4":
        return accepted_len <= 4
    if policy == "pld_miss_only":
        miss = payload.get("pld_miss", payload.get("is_pld_miss"))
        if miss is None:
            raise ValueError("pld_miss_only requires pld_miss/is_pld_miss")
        return miss.bool()
    if policy == "router_predicted_weak":
        router = payload.get("router_predicted_weak")
        if router is None:
            raise ValueError("router_predicted_weak requires router_predicted_weak")
        return router.bool()
    if policy == "router":
        return accepted_len <= 4
    if policy == "always":
        return torch.ones_like(accepted_len, dtype=torch.bool)
    if policy == "never":
        return torch.zeros_like(accepted_len, dtype=torch.bool)
    raise ValueError(f"unknown residual trigger policy: {policy!r}")


def normalize_trigger_policy(policy: str) -> str:
    normalized = _TRIGGER_ALIASES.get(str(policy).strip(), str(policy).strip())
    if normalized not in TRIGGER_POLICIES:
        valid = ", ".join(TRIGGER_POLICIES)
        raise ValueError(f"unknown residual trigger policy {policy!r}; valid={valid}")
    return normalized


def should_trigger_residual(
    policy: str,
    step: Any | None = None,
    *,
    accepted_len: int | None = None,
    pld_miss: bool | None = None,
    router_probability: float | None = None,
    router_threshold: float = 0.5,
) -> bool:
    """Evaluate a residual trigger policy from explicit fields or a step row."""

    policy = normalize_trigger_policy(policy)
    if accepted_len is None:
        accepted_len = int(
            _get_field(step, ("accepted_len", "n_accepted_drafts", "pld_accepted_len"), 0)
        )
    if pld_miss is None:
        raw_miss = _get_field(step, ("pld_miss", "is_pld_miss"), None)
        pld_miss = bool(raw_miss) if raw_miss is not None else int(accepted_len) == 0
    accepted = int(accepted_len)
    if policy == "accepted_len_eq_0":
        return accepted == 0
    if policy == "accepted_len_le_1":
        return accepted <= 1
    if policy == "accepted_len_le_2":
        return accepted <= 2
    if policy == "accepted_len_le_4":
        return accepted <= 4
    if policy == "pld_miss_only":
        return bool(pld_miss)
    if policy == "router_predicted_weak":
        if router_probability is None:
            raw_prob = _get_field(step, ("router_probability", "weak_probability"), None)
            router_probability = None if raw_prob is None else float(raw_prob)
        return accepted <= 4 if router_probability is None else float(router_probability) >= float(router_threshold)
    if policy == "router":
        return accepted <= 4
    if policy == "always":
        return True
    if policy == "never":
        return False
    raise AssertionError(f"unhandled policy {policy!r}")


def accepted_prefix_lengths(predicted: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return accepted-prefix length per row for ``[batch, K]`` tensors."""

    if predicted.ndim != 2 or labels.ndim != 2:
        raise ValueError("predicted and labels must both be [batch, num_heads]")
    n = min(int(predicted.shape[1]), int(labels.shape[1]))
    out = []
    for pred, label in zip(
        predicted[:, :n].detach().cpu().tolist(),
        labels[:, :n].detach().cpu().tolist(),
        strict=True,
    ):
        filtered = []
        for item in label:
            if int(item) < 0:
                break
            filtered.append(int(item))
        out.append(accepted_prefix_length([int(item) for item in pred], filtered))
    return torch.tensor(out, dtype=torch.long, device=predicted.device)


def residual_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy over valid residual labels in ``[batch, K, vocab]``."""

    if logits.ndim != 3:
        raise ValueError(f"logits must be [batch, num_heads, vocab], got {tuple(logits.shape)}")
    labels = labels.long()
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    if labels.ndim != 2:
        raise ValueError(f"labels must be [batch, num_heads], got {tuple(labels.shape)}")
    heads = min(int(logits.shape[1]), int(labels.shape[1]))
    logits = logits[:, :heads, :]
    labels = labels[:, :heads]
    valid = (labels >= 0) & (labels != int(ignore_index))
    if mask is not None:
        if mask.ndim == 1:
            mask = mask.unsqueeze(1).expand(-1, heads)
        valid = valid & mask[:, :heads].bool()
    if int(valid.sum().item()) == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], labels[valid])


def accuracy_by_head(predicted: torch.Tensor, labels: torch.Tensor) -> list[float]:
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    heads = min(int(predicted.shape[1]), int(labels.shape[1]))
    out: list[float] = []
    for i in range(heads):
        valid = labels[:, i] >= 0
        total = int(valid.sum().item())
        if total == 0:
            out.append(0.0)
        else:
            out.append(float((predicted[:, i][valid] == labels[:, i][valid]).float().mean().item()))
    return out


def infer_vocab_size(labels: torch.Tensor, requested: int = 0, minimum: int = 1) -> int:
    if requested > 0:
        return int(requested)
    valid = labels[labels >= 0]
    if valid.numel() == 0:
        return int(minimum)
    return max(int(minimum), int(valid.max().item()) + 1)


def load_residual_tensor_payload(path: Path, *, num_heads: int | None = None) -> dict[str, Any]:
    """Load residual-head tensors using aliases from current/offline artifacts."""

    obj = torch.load(path, map_location="cpu")
    if hasattr(obj, "tensors"):
        tensors = list(obj.tensors)
        obj = {"hidden": tensors[0], "labels": tensors[1]}
    elif isinstance(obj, (tuple, list)) and len(obj) >= 2 and torch.is_tensor(obj[0]):
        obj = {"hidden": obj[0], "labels": obj[1]}
    if not isinstance(obj, Mapping):
        raise ValueError(f"{path} is not a supported residual tensor artifact")
    hidden = _first_present_tensor(
        obj,
        ("hidden", "hidden_states", "post_pld_hidden", "residual_hidden", "x"),
    )
    labels = _first_present_tensor(
        obj,
        ("labels", "future_token_ids", "target_token_ids", "targets", "y"),
    )
    if hidden is None:
        raise ValueError(f"{path} does not contain hidden states")
    if labels is None:
        raise ValueError(f"{path} does not contain future-token labels")
    payload: dict[str, Any] = {
        "hidden": hidden.float(),
        "labels": _normalize_labels(labels, num_heads=num_heads),
        "metadata": dict(obj.get("metadata") or {}),
    }
    for out_key, aliases in {
        "base_logits": ("base_logits", "target_logits", "logits"),
        "accepted_len": ("accepted_len", "n_accepted_drafts", "pld_accepted_len"),
        "pld_miss": ("pld_miss", "is_pld_miss"),
        "task_id": ("task_id", "task_ids"),
        "step_id": ("step_id", "step_ids"),
    }.items():
        value = _first_present(obj, aliases)
        if value is not None:
            payload[out_key] = value.float() if out_key == "base_logits" and torch.is_tensor(value) else value
    return payload


def filter_residual_dataset(payload: dict[str, Any], policy: str) -> dict[str, Any]:
    """Return a copy filtered to residual examples selected by ``policy``."""

    mask = trigger_mask(payload, policy)
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if torch.is_tensor(value) and value.shape[:1] == mask.shape:
            out[key] = value[mask].clone()
        elif isinstance(value, list) and len(value) == int(mask.shape[0]):
            out[key] = [item for item, keep in zip(value, mask.tolist(), strict=True) if keep]
        else:
            out[key] = value
    metadata = dict(out.get("metadata") or {})
    metadata["residual_filter_policy"] = policy
    metadata["residual_filter_kept"] = int(mask.sum().item())
    metadata["residual_filter_total"] = int(mask.numel())
    metadata["method_family"] = "VANTAGE-Residual"
    out["metadata"] = metadata
    return out


def load_residual_heads(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    max_heads: int | None = None,
) -> nn.Module:
    """Load a trained residual MTP checkpoint.

    Historical post-PLD checkpoints used ``PLDMTPHeads`` directly. New phase
    artifacts may use ``ResidualMTPHeads`` with additive residual logits.
    """

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config_dict = dict(ckpt["config"])
    if "residual_scale" in config_dict or "mtp_position" in config_dict:
        config = ResidualMTPConfig(**config_dict)
        if max_heads is not None and int(config.num_heads) < int(max_heads):
            raise ValueError(f"checkpoint has only {config.num_heads} heads, requested {max_heads}")
        model = ResidualMTPHeads(config)
        model.load_state_dict(ckpt["model_state"])
        model.to(device=device, dtype=dtype)
        model.eval()
        model.prepare_for_inference()
        return model
    config = MTPHeadConfig(**config_dict)
    if max_heads is not None and int(config.num_heads) < int(max_heads):
        raise ValueError(f"checkpoint has only {config.num_heads} heads, requested {max_heads}")
    output_weight = ckpt.get("output_weight")
    if output_weight is not None:
        output_weight = output_weight.to(device=device, dtype=dtype)
    model = PLDMTPHeads(config, output_weight=output_weight)
    model.load_state_dict(ckpt["model_state"])
    model.to(device=device, dtype=dtype)
    model.eval()
    model.prepare_for_inference()
    return model


def predict_residual_tokens(
    model: nn.Module,
    hidden: torch.Tensor,
    *,
    max_heads: int | None = None,
) -> list[int]:
    """Predict future token IDs from one residual hidden state."""

    if hidden.ndim == 1:
        hidden = hidden.reshape(1, -1)
    tokens = model.predict_token_tensor(hidden)[0].detach().cpu().tolist()
    if max_heads is not None:
        tokens = tokens[: int(max_heads)]
    return [int(t) for t in tokens]


def _coerce_base_logits(
    base_logits: torch.Tensor | Sequence[torch.Tensor],
    *,
    num_heads: int,
    vocab_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(base_logits, Sequence) and not torch.is_tensor(base_logits):
        base = torch.stack([item for item in base_logits], dim=1)
    else:
        base = base_logits
    if not torch.is_tensor(base):
        raise TypeError("base_logits must be a tensor or sequence of tensors")
    base = base.to(device=device, dtype=dtype)
    if base.ndim == 2:
        base = base.unsqueeze(1).expand(-1, num_heads, -1)
    elif base.ndim != 3:
        raise ValueError(
            f"base_logits must be [batch, vocab] or [batch, num_heads, vocab], got {tuple(base.shape)}"
        )
    if int(base.shape[1]) < num_heads:
        raise ValueError(f"base_logits has {base.shape[1]} heads, expected at least {num_heads}")
    if int(base.shape[-1]) != vocab_size:
        raise ValueError(f"base_logits vocab {base.shape[-1]} does not match config vocab {vocab_size}")
    return base[:, :num_heads, :]


def _normalize_labels(labels: torch.Tensor, *, num_heads: int | None) -> torch.Tensor:
    labels = labels.long()
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    elif labels.ndim > 2:
        labels = labels.reshape(labels.shape[0], -1)
    if num_heads is not None:
        labels = labels[:, : int(num_heads)]
    return labels


def _first_present_tensor(mapping: Mapping[str, Any], keys: Sequence[str]) -> torch.Tensor | None:
    value = _first_present(mapping, keys)
    return value if torch.is_tensor(value) else None


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _get_field(step: Any | None, names: Sequence[str], default: Any) -> Any:
    if step is None:
        return default
    for name in names:
        if isinstance(step, Mapping) and name in step:
            value = step[name]
            return value.item() if torch.is_tensor(value) and value.numel() == 1 else value
        if hasattr(step, name):
            value = getattr(step, name)
            return value.item() if torch.is_tensor(value) and value.numel() == 1 else value
    return default


__all__ = [
    "TRIGGER_POLICIES",
    "ResidualMTPConfig",
    "ResidualMTPHeads",
    "ResidualTriggerStep",
    "accepted_prefix_length",
    "accepted_prefix_lengths",
    "accuracy_by_head",
    "filter_residual_dataset",
    "infer_vocab_size",
    "load_residual_heads",
    "load_residual_tensor_payload",
    "normalize_trigger_policy",
    "predict_residual_tokens",
    "residual_cross_entropy",
    "should_trigger_residual",
    "trigger_mask",
    "validate_residual_dataset",
]
