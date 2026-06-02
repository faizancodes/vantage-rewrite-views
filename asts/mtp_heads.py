"""Offline multi-token prediction heads for PLD-adjacent diagnostics.

These modules are intentionally not wired into runtime decoding.  They train
small heads on frozen target-model hidden states collected from existing PLD
traces, then offline evaluators can estimate whether the candidate source has
enough headroom to justify a real decoder implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MTPHeadConfig:
    hidden_size: int
    vocab_size: int
    num_heads: int = 4
    head_type: str = "linear"
    hidden_dim: int = 2048
    bottleneck: int = 256
    dropout: float = 0.0


class PLDMTPHeads(nn.Module):
    """A small bank of horizon-specific token classifiers.

    Each head predicts one future token from the same current hidden state:
    head 0 predicts t+1, head 1 predicts t+2, and so on.  The bottleneck keeps
    the diagnostic materially cheaper than a second language model while still
    allowing horizon-specific transformations.
    """

    def __init__(self, config: MTPHeadConfig, output_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.config = config
        self.uses_tied_output = output_weight is not None
        self.register_buffer("_fused_linear_weight", None, persistent=False)
        if output_weight is not None:
            if output_weight.ndim != 2:
                raise ValueError("output_weight must be [vocab, hidden]")
            self.register_buffer("output_weight", output_weight.detach().clone())
        else:
            self.output_weight = None
        if config.head_type not in {"linear", "mlp"}:
            raise ValueError(f"unsupported MTP head_type: {config.head_type!r}")
        if config.head_type == "linear" and self.uses_tied_output:
            self.heads = nn.ModuleList(
                [nn.Linear(config.hidden_size, config.hidden_size, bias=False) for _ in range(config.num_heads)]
            )
            for head in self.heads:
                nn.init.eye_(head.weight)
        elif config.head_type == "linear":
            self.heads = nn.ModuleList(
                [nn.Linear(config.hidden_size, config.vocab_size) for _ in range(config.num_heads)]
            )
        elif self.uses_tied_output:
            dim = int(config.hidden_dim or config.bottleneck)
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(config.hidden_size),
                        nn.Linear(config.hidden_size, dim),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(dim, config.hidden_size),
                    )
                    for _ in range(config.num_heads)
                ]
            )
        else:
            dim = int(config.hidden_dim or config.bottleneck)
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

    def prepare_for_inference(self) -> "PLDMTPHeads":
        """Cache fused adapter weights for the tied-linear runtime path."""

        if self.config.head_type == "linear" and self.uses_tied_output:
            self._fused_linear_weight = torch.stack(
                [head.weight.detach() for head in self.heads], dim=0
            ).contiguous()
        return self

    def forward(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        if hidden.ndim != 2:
            raise ValueError(f"hidden must be [batch, hidden], got {tuple(hidden.shape)}")
        if self.config.head_type == "linear" and self.uses_tied_output:
            fused_logits = self.forward_logits(hidden)
            return [fused_logits[:, i, :] for i in range(fused_logits.shape[1])]
        logits = []
        for head in self.heads:
            out = head(hidden)
            if self.uses_tied_output:
                out = out @ self.output_weight.T
            logits.append(out)
        return logits

    def forward_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return all horizon logits as ``[batch, num_heads, vocab]``.

        Runtime decoding needs all K heads for a single hidden vector.  The
        common trained configuration is a bank of linear hidden adapters tied
        to the frozen target LM head.  Fusing those adapters into one batched
        projection avoids four Python module calls and four separate vocab
        matmuls in the hot path while preserving the older ``forward`` API for
        training/evaluation code.
        """

        if hidden.ndim != 2:
            raise ValueError(f"hidden must be [batch, hidden], got {tuple(hidden.shape)}")
        if self.config.head_type == "linear" and self.uses_tied_output:
            weights = self._fused_linear_weight
            if weights is None:
                weights = torch.stack([head.weight for head in self.heads], dim=0)
            projected = torch.einsum("bh,koh->bko", hidden, weights)
            return torch.matmul(projected, self.output_weight.T)
        logits = self.forward(hidden)
        return torch.stack(logits, dim=1)

    def predict_token_tensor(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return argmax token ids for all heads as ``[batch, num_heads]``."""

        with torch.inference_mode():
            return self.forward_logits(hidden).argmax(dim=-1)


def accepted_prefix_length(predicted: list[int] | tuple[int, ...], labels: list[int] | tuple[int, ...]) -> int:
    """Return the longest matching prefix length for future-token predictions."""

    n = min(len(predicted), len(labels))
    out = 0
    for i in range(n):
        if int(predicted[i]) != int(labels[i]):
            break
        out += 1
    return out
