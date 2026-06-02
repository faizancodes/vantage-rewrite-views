"""KV-cache primitives for PVP (Predictive Verifier Pipelining).

Single-row primitives, no batching. ``pvp_decoder.py`` combines them into the
batched B=2 cache and attention mask used by the speculative forward.

Two cache layouts are supported transparently:

* Legacy: ``cache.key_cache[i]`` / ``cache.value_cache[i]`` are the storage.
* New (transformers >= ~4.46): ``cache.layers[i].keys`` / ``.values`` are the
  canonical storage and ``key_cache`` is a derived view.

Public helpers:

* ``layer_keys_values(cache)`` — iterate per-layer (keys, values) tensors in a
  layout-agnostic way.
* ``set_layer_keys_values(cache, layer_idx, keys, values)`` — write back.
* ``clone_cache(cache)`` — deep copy that is safe for downstream mutation.
* ``cache_length(cache)`` — return seq length (the size of dim=-2 of layer 0).
* ``lift_prompt_kv(prefix_kv, prompt_kv_full, src_start, src_len)`` — append
  ``src_len`` slots lifted from ``prompt_kv_full[src_start:src_start+src_len]``
  to a clone of ``prefix_kv``. Inputs are not mutated.
"""

from __future__ import annotations

import copy
from typing import Iterator

import torch


def _has_new_layout(cache) -> bool:
    return hasattr(cache, "layers") and cache.layers is not None and len(cache.layers) > 0


def layer_keys_values(cache) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (keys, values) for every layer, layout-agnostic.

    Both tensors have shape ``[B, num_kv_heads, seq, head_dim]``.
    """
    if cache is None:
        return
    if _has_new_layout(cache):
        for layer in cache.layers:
            yield layer.keys, layer.values
        return
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for i in range(len(cache.key_cache)):
            yield cache.key_cache[i], cache.value_cache[i]
        return
    raise TypeError(f"unrecognized KV cache layout: {type(cache).__name__}")


def set_layer_keys_values(cache, layer_idx: int, keys: torch.Tensor, values: torch.Tensor) -> None:
    """Write back per-layer K/V, layout-agnostic."""
    if _has_new_layout(cache):
        cache.layers[layer_idx].keys = keys
        cache.layers[layer_idx].values = values
        return
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        cache.key_cache[layer_idx] = keys
        cache.value_cache[layer_idx] = values
        return
    raise TypeError(f"unrecognized KV cache layout: {type(cache).__name__}")


def num_layers(cache) -> int:
    if cache is None:
        return 0
    if _has_new_layout(cache):
        return len(cache.layers)
    if hasattr(cache, "key_cache"):
        return len(cache.key_cache)
    raise TypeError(f"unrecognized KV cache layout: {type(cache).__name__}")


def cache_length(cache) -> int:
    """Return seq length of the cache (size of dim=-2 of layer 0). 0 if empty."""
    if cache is None:
        return 0
    for layer_kv in layer_keys_values(cache):
        k = layer_kv[0]
        if k is None:
            return 0
        return int(k.shape[-2])
    return 0


def clone_cache(cache):
    """Deep copy a cache. Cheap-ish because tensors are copied along with metadata."""
    return copy.deepcopy(cache)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """HF-style half-rotation: concat(-x_top_half, x_bottom_half) along last dim."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope_delta(
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Rotate K by the angle encoded in ``(cos, sin)``.

    K is post-rotary; this applies a *delta* rotation that shifts the baked-in
    position by ``delta`` for whatever ``cos = cos(delta·θ)``, ``sin = sin(delta·θ)``
    the caller passed in. Composing rotations is additive in the angle, so a
    K stored with rotary at source position ``s`` becomes rotary at ``s + delta``
    after this call.

    Shapes:
        k:   [B, H, S, D]
        cos: broadcastable to k (e.g. [1, 1, 1, D] for a constant delta).
        sin: same.
    """
    return k * cos + rotate_half(k) * sin


def lift_prompt_kv(
    prefix_kv,
    prompt_kv_full,
    src_start: int,
    src_len: int,
    *,
    rotary_cos: torch.Tensor | None = None,
    rotary_sin: torch.Tensor | None = None,
):
    """Append ``src_len`` slots from ``prompt_kv_full`` to a clone of ``prefix_kv``.

    Per-layer, per-head, K and V both. The source slice spans
    ``[src_start, src_start + src_len)`` along the seq dim of ``prompt_kv_full``.

    The result is a fresh cache object whose seq length is
    ``cache_length(prefix_kv) + src_len``. Neither input is mutated.

    When both ``rotary_cos`` and ``rotary_sin`` are provided, each lifted K is
    re-rotated by the delta they encode (V is *not* re-rotated — RoPE only
    applies to Q and K). This is how PVP makes lifted KV behave as if it had
    been computed at the destination positions rather than the source
    positions; without this rotation, the rotary embeddings encode the wrong
    relative offset and row-1 attention is corrupted.

    Edge cases:
        * ``src_len <= 0`` → returns a plain clone of ``prefix_kv``.
        * ``prompt_kv_full`` may be the same object as ``prefix_kv`` (the prompt
          KV is just the cache populated by prefill); the slice is taken from
          the current state.
    """
    if prefix_kv is None or prompt_kv_full is None:
        raise ValueError("lift_prompt_kv requires non-None caches")
    if src_len < 0:
        raise ValueError(f"src_len must be >= 0, got {src_len}")
    if src_start < 0:
        raise ValueError(f"src_start must be >= 0, got {src_start}")

    out = clone_cache(prefix_kv)
    if src_len == 0:
        return out

    n = num_layers(prefix_kv)
    if num_layers(prompt_kv_full) != n:
        raise ValueError(
            f"layer-count mismatch: prefix has {n} layers, prompt has "
            f"{num_layers(prompt_kv_full)}"
        )

    prefix_layers = list(layer_keys_values(prefix_kv))
    prompt_layers = list(layer_keys_values(prompt_kv_full))

    do_rotation = rotary_cos is not None and rotary_sin is not None

    for i in range(n):
        pk, pv = prefix_layers[i]
        sk, sv = prompt_layers[i]
        if pk is None or sk is None:
            continue
        if sk.shape[-2] < src_start + src_len:
            raise ValueError(
                f"layer {i}: prompt KV has only {sk.shape[-2]} slots, "
                f"need src_start+src_len={src_start + src_len}"
            )
        lifted_k = sk[..., src_start : src_start + src_len, :].clone()
        lifted_v = sv[..., src_start : src_start + src_len, :].clone()
        if do_rotation:
            assert rotary_cos is not None and rotary_sin is not None
            cos = rotary_cos.to(lifted_k.dtype)
            sin = rotary_sin.to(lifted_k.dtype)
            lifted_k = apply_rope_delta(lifted_k, cos, sin)
        new_k = torch.cat([pk, lifted_k], dim=-2).contiguous()
        new_v = torch.cat([pv, lifted_v], dim=-2).contiguous()
        set_layer_keys_values(out, i, new_k, new_v)

    if hasattr(out, "_seen_tokens"):
        out._seen_tokens = int(out._seen_tokens or 0) + src_len
    return out


