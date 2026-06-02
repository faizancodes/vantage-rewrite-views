"""EAGLE-1-style draft head for Qwen2.5-Coder-7B.

A single Qwen2 decoder block that predicts the next-position hidden state
given the current position's penultimate-layer hidden state. Combined with
the target's frozen LM head, this acts as a fast draft model.

Architecture choices (matched to Qwen2.5-Coder-7B):
- hidden_size = 3584
- num_attention_heads = 28, num_kv_heads = 4 (GQA)
- intermediate_size = 18944
- 1 transformer decoder block (vs target's 28)
- LM head shared with target (frozen, no extra params)

Total trainable parameters: ~250M (one Qwen2 decoder layer at this width).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class EagleConfig:
    hidden_size: int = 3584
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    intermediate_size: int = 18944
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    rope_scaling: dict | None = None
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    @classmethod
    def from_target_config(cls, target_config) -> "EagleConfig":
        """Construct an EagleConfig matching the target model's shape."""
        return cls(
            hidden_size=target_config.hidden_size,
            num_attention_heads=target_config.num_attention_heads,
            num_key_value_heads=getattr(
                target_config, "num_key_value_heads", target_config.num_attention_heads
            ),
            intermediate_size=target_config.intermediate_size,
            max_position_embeddings=getattr(
                target_config, "max_position_embeddings", 32768
            ),
            rms_norm_eps=getattr(target_config, "rms_norm_eps", 1e-6),
            rope_theta=getattr(target_config, "rope_theta", 1_000_000.0),
            rope_scaling=getattr(target_config, "rope_scaling", None),
        )


class EagleHead(nn.Module):
    """One Qwen2 decoder block trained to predict next-position hidden states.

    We use HF transformers' `Qwen2DecoderLayer` as the building block — this
    guarantees the attention/FFN math exactly matches Qwen-Coder-7B and avoids
    re-deriving GQA/RoPE/RMSNorm details.
    """

    def __init__(self, config: EagleConfig):
        super().__init__()
        # Lazy-import to keep the module importable without transformers
        # available (e.g., for shape-only unit tests).
        from transformers import Qwen2Config
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

        # Build a minimal Qwen2Config matching the target. We set num_hidden_layers=1
        # but the layer itself doesn't care about the parent config's layer count.
        hf_config = Qwen2Config(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=1,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            attention_dropout=config.attention_dropout,
        )
        self.config = config
        self.hf_config = hf_config
        self.layer = Qwen2DecoderLayer(hf_config, layer_idx=0)
        # NOTE: no separate final norm — caller (training/inference) MUST
        # apply the target model's `model.norm` before projecting through
        # the LM head. This matches the EAGLE-1 official convention and
        # ensures pred_logits are computed via the same path as target_logits.
        # RoPE (rotary embeddings) — Qwen2DecoderLayer expects position_embeddings
        # to be passed in. We construct them here.
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
        self.rotary_emb = Qwen2RotaryEmbedding(config=hf_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_value=None,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
    ):
        """Predict next-position hidden states from current.

        Args:
            hidden_states: [B, T, H]  — typically target's penultimate layer
            position_ids: [B, T]      — defaults to torch.arange(T)
            attention_mask: [B, T]    — 1 for tokens to attend to (training)
            past_key_value:           — DynamicCache from prior call (inference)
            use_cache: True at inference to return new past_key_value
            cache_position:           — [T] absolute positions for each input token

        Returns:
            If use_cache: (predicted_h, new_past_key_value)
            Else:        predicted_h alone
        """
        bsz, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # Default cache_position to absolute positions assumed equal to position_ids
        # (we only support non-overlapping increments).
        if position_ids is None:
            past_len = 0
            if past_key_value is not None and hasattr(past_key_value, "get_seq_length"):
                past_len = past_key_value.get_seq_length(layer_idx=0)
            position_ids = torch.arange(past_len, past_len + seq_len, device=device).unsqueeze(0).expand(bsz, -1)

        if cache_position is None:
            past_len = 0
            if past_key_value is not None and hasattr(past_key_value, "get_seq_length"):
                past_len = past_key_value.get_seq_length(layer_idx=0)
            cache_position = torch.arange(past_len, past_len + seq_len, device=device)

        # Build attention mask. With cache, attention spans (past_len + seq_len).
        # The HF Qwen2 layer expects an additive mask [B, 1, T, S] where S is
        # the total sequence length (past + current).
        past_len = int(cache_position[0].item())
        total_len = past_len + seq_len
        causal_mask_4d = _make_causal_4d_mask_with_past(
            seq_len=seq_len,
            past_len=past_len,
            dtype=hidden_states.dtype,
            device=device,
        )
        if attention_mask is not None and past_len == 0:
            pad_mask = (1.0 - attention_mask.to(hidden_states.dtype))[:, None, None, :] * torch.finfo(
                hidden_states.dtype
            ).min
            causal_mask_4d = causal_mask_4d + pad_mask

        # Generate position embeddings (cos, sin tuples)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        layer_kwargs = dict(
            attention_mask=causal_mask_4d,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            output_attentions=False,
            use_cache=use_cache,
        )
        # cache_position is required by some transformers versions for proper
        # cache slot indexing; pass it conditionally.
        if cache_position is not None:
            layer_kwargs["cache_position"] = cache_position

        try:
            layer_out = self.layer(hidden_states, **layer_kwargs)
        except TypeError:
            # Older transformers don't accept some of these kwargs
            layer_kwargs.pop("position_embeddings", None)
            layer_kwargs.pop("cache_position", None)
            layer_out = self.layer(hidden_states, **layer_kwargs)
        # Recent transformers returns just hidden_states; use_cache state is
        # mutated on the past_key_value object passed in. Older versions return
        # a tuple. Handle both.
        if isinstance(layer_out, tuple):
            h = layer_out[0]
        else:
            h = layer_out
        # Output PRE-final-norm; caller applies target.model.norm.
        if use_cache:
            return h, past_key_value
        return h


def _make_causal_4d_mask(seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Build [1, 1, T, T] causal attention mask with -inf above the diagonal."""
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask[None, None, :, :]


def _make_causal_4d_mask_with_past(
    seq_len: int, past_len: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Build [1, 1, seq_len, past_len + seq_len] causal mask for KV-cache decoding.

    All past positions are visible (mask=0); for the new positions, position i
    can see only past_len + i and earlier within the new chunk.
    """
    total_len = past_len + seq_len
    # Start with all 0 (visible)
    mask = torch.zeros((seq_len, total_len), dtype=dtype, device=device)
    # Apply causal mask within the new chunk: position i (in new) cannot see
    # positions past_len + j for j > i.
    for i in range(seq_len):
        # Mask out positions [past_len + i + 1 : past_len + seq_len]
        if past_len + i + 1 < total_len:
            mask[i, past_len + i + 1 :] = torch.finfo(dtype).min
    return mask[None, None, :, :]


# ---------------------------------------------------------------------------
# Total parameter count
# ---------------------------------------------------------------------------


def count_params(head: EagleHead) -> int:
    return sum(p.numel() for p in head.parameters() if p.requires_grad)


def estimate_trainable_params(config: EagleConfig) -> int:
    """Closed-form estimate so we can size the head before instantiating."""
    h = config.hidden_size
    n_h = config.num_attention_heads
    n_kv = config.num_key_value_heads
    head_dim = h // n_h
    intermediate = config.intermediate_size

    # Attention: q (h*h), k (h*n_kv*head_dim), v (h*n_kv*head_dim), o (h*h)
    attn = h * h + h * n_kv * head_dim + h * n_kv * head_dim + h * h
    # MLP (SwiGLU): up (h*intermediate), gate (h*intermediate), down (intermediate*h)
    mlp = 3 * h * intermediate
    # Two RMSNorms (input + post-attn) inside the decoder block
    norms = 2 * h
    return attn + mlp + norms
