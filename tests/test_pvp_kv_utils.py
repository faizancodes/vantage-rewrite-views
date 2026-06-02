"""CPU unit tests for asts.pvp_kv_utils.

These tests build small synthetic KV caches (no model load) and exercise the
single-row primitives. Real-model integration tests live in
``tests/test_pvp_lossless.py`` (GPU only, Step 5).
"""

from __future__ import annotations

import pytest
import torch
from transformers import DynamicCache

from asts.pvp_kv_utils import (
    apply_rope_delta,
    cache_length,
    layer_keys_values,
    lift_prompt_kv,
    num_layers,
    rotate_half,
    set_layer_keys_values,
)


def _fake_rope_rotate(k_raw: torch.Tensor, position: int, base: float = 1_000_000.0) -> torch.Tensor:
    """Apply standard half-rotation RoPE at a single absolute position.

    Matches HF's apply_rotary_pos_emb when called with position_ids = [[position]].
    k_raw shape: [..., head_dim].
    """
    head_dim = k_raw.shape[-1]
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    angles = position * inv_freq  # [head_dim // 2]
    cos_half = torch.cos(angles).float()
    sin_half = torch.sin(angles).float()
    cos = torch.cat([cos_half, cos_half], dim=-1).to(k_raw.dtype)
    sin = torch.cat([sin_half, sin_half], dim=-1).to(k_raw.dtype)
    while cos.dim() < k_raw.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return k_raw * cos + rotate_half(k_raw) * sin


def _rope_cos_sin_for_delta(
    delta: int, head_dim: int, base: float = 1_000_000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cos, sin tensors of shape [1, 1, 1, head_dim] for ``delta * theta_i``."""
    abs_delta = abs(int(delta))
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    angles = abs_delta * inv_freq
    cos_half = torch.cos(angles).float()
    sin_half = torch.sin(angles).float()
    cos = torch.cat([cos_half, cos_half], dim=-1)[None, None, None, :]
    sin = torch.cat([sin_half, sin_half], dim=-1)[None, None, None, :]
    if delta < 0:
        sin = -sin
    return cos, sin


# ---------------------------------------------------------------------------
# RoPE delta tests
# ---------------------------------------------------------------------------


def test_rope_identity_at_zero_delta():
    """delta = 0 → cos = 1, sin = 0, apply_rope_delta is identity."""
    head_dim = 64
    k = torch.randn(2, 3, 5, head_dim, dtype=torch.float32)
    cos = torch.ones(1, 1, 1, head_dim)
    sin = torch.zeros(1, 1, 1, head_dim)
    out = apply_rope_delta(k, cos, sin)
    torch.testing.assert_close(out, k)


def test_rope_inverse_roundtrip():
    """rotate(k, +δ) followed by rotate(·, -δ) returns k."""
    head_dim = 64
    delta = 37
    k = torch.randn(1, 2, 4, head_dim, dtype=torch.float32)
    cos_fwd, sin_fwd = _rope_cos_sin_for_delta(delta, head_dim)
    cos_inv, sin_inv = _rope_cos_sin_for_delta(-delta, head_dim)
    fwd = apply_rope_delta(k, cos_fwd, sin_fwd)
    back = apply_rope_delta(fwd, cos_inv, sin_inv)
    torch.testing.assert_close(back, k, atol=1e-5, rtol=1e-5)


def test_rope_composition_matches_direct():
    """K rotated at source s, then re-rotated by (d-s), equals K rotated directly at d."""
    head_dim = 64
    source = 12
    destination = 47
    k_raw = torch.randn(1, 1, 1, head_dim, dtype=torch.float32)
    k_at_s = _fake_rope_rotate(k_raw, source)
    k_at_d_direct = _fake_rope_rotate(k_raw, destination)
    cos, sin = _rope_cos_sin_for_delta(destination - source, head_dim)
    k_at_d_via_delta = apply_rope_delta(k_at_s, cos, sin)
    torch.testing.assert_close(k_at_d_via_delta, k_at_d_direct, atol=1e-5, rtol=1e-5)


def test_rope_negative_delta_composition():
    """Composition still holds when destination < source (delta is negative)."""
    head_dim = 64
    source = 73
    destination = 10
    k_raw = torch.randn(1, 1, 1, head_dim, dtype=torch.float32)
    k_at_s = _fake_rope_rotate(k_raw, source)
    k_at_d_direct = _fake_rope_rotate(k_raw, destination)
    cos, sin = _rope_cos_sin_for_delta(destination - source, head_dim)
    k_at_d_via_delta = apply_rope_delta(k_at_s, cos, sin)
    torch.testing.assert_close(k_at_d_via_delta, k_at_d_direct, atol=1e-5, rtol=1e-5)


def test_lift_with_rotation_matches_freshly_rotated():
    """End-to-end: build a fake prompt cache rotated at source positions, lift with
    re-rotation, assert lifted slots equal a freshly-built cache rotated at
    destination positions.
    """
    n_layers, B, H, head_dim = 2, 1, 2, 32
    prefix_seq = 8
    prompt_seq = 40
    src_start, src_len = 20, 5
    dst_start = prefix_seq  # lift appends after the prefix
    delta = dst_start - src_start

    # Build per-position K_raw values.
    torch.manual_seed(0)
    k_raw_per_layer = [
        torch.randn(B, H, prompt_seq, head_dim, dtype=torch.float32)
        for _ in range(n_layers)
    ]
    # Build prompt cache by rotating each slot at its absolute position.
    from transformers import DynamicCache
    prompt_cache = DynamicCache()
    for layer in range(n_layers):
        k_per_pos = [
            _fake_rope_rotate(k_raw_per_layer[layer][:, :, p : p + 1, :], position=p)
            for p in range(prompt_seq)
        ]
        k_rot = torch.cat(k_per_pos, dim=-2).contiguous()
        v = torch.randn(B, H, prompt_seq, head_dim, dtype=torch.float32)
        prompt_cache.update(k_rot, v, layer)

    # Build prefix cache (independent content, just to give the prefix a length).
    prefix_cache = DynamicCache()
    for layer in range(n_layers):
        k = torch.randn(B, H, prefix_seq, head_dim, dtype=torch.float32)
        v = torch.randn(B, H, prefix_seq, head_dim, dtype=torch.float32)
        prefix_cache.update(k, v, layer)

    cos, sin = _rope_cos_sin_for_delta(delta, head_dim)

    lifted = lift_prompt_kv(
        prefix_cache, prompt_cache, src_start, src_len,
        rotary_cos=cos, rotary_sin=sin,
    )

    # The lifted region of `lifted` should equal K_raw[..., src_start:src_start+src_len, :]
    # freshly rotated at destination positions dst_start..dst_start+src_len-1.
    lifted_layers = list(layer_keys_values(lifted))
    for layer in range(n_layers):
        out_k = lifted_layers[layer][0]
        # Compare the lifted region only.
        lifted_region = out_k[..., prefix_seq:, :]
        for i in range(src_len):
            expected = _fake_rope_rotate(
                k_raw_per_layer[layer][:, :, src_start + i : src_start + i + 1, :],
                position=dst_start + i,
            )
            torch.testing.assert_close(
                lifted_region[..., i : i + 1, :], expected,
                atol=1e-5, rtol=1e-5,
            )


def _make_marked_cache(
    *,
    n_layers: int,
    B: int,
    H: int,
    seq: int,
    D: int,
    base: float,
    layer_stride: float = 1000.0,
    pos_stride: float = 1.0,
) -> DynamicCache:
    """Make a DynamicCache whose layer/position content is uniquely identifiable.

    K[layer, batch, head, pos, d] = base + layer*layer_stride + pos*pos_stride
    V[layer, ...] = K + 0.5  (so K and V can never collide).
    """
    cache = DynamicCache()
    for layer in range(n_layers):
        # shape [B, H, seq, D]; vary along seq only
        per_pos = torch.arange(seq, dtype=torch.float32).view(1, 1, -1, 1)
        k = (base + layer * layer_stride + per_pos * pos_stride).expand(B, H, seq, D).contiguous()
        v = (k + 0.5).contiguous()
        cache.update(k, v, layer)
    return cache


def test_lift_appends_correct_slots():
    """Lifted slots match the prompt source; prefix slots are preserved."""
    n_layers, B, H, D = 3, 1, 2, 4
    L_prefix = 5
    L_prompt = 20
    src_start, src_len = 7, 6

    prefix_kv = _make_marked_cache(
        n_layers=n_layers, B=B, H=H, seq=L_prefix, D=D, base=0.0,
    )
    prompt_kv = _make_marked_cache(
        n_layers=n_layers, B=B, H=H, seq=L_prompt, D=D, base=100.0,
    )

    out = lift_prompt_kv(prefix_kv, prompt_kv, src_start=src_start, src_len=src_len)

    assert cache_length(out) == L_prefix + src_len
    assert num_layers(out) == n_layers

    out_layers = list(layer_keys_values(out))
    prefix_layers = list(layer_keys_values(prefix_kv))
    prompt_layers = list(layer_keys_values(prompt_kv))

    for layer in range(n_layers):
        ok, ov = out_layers[layer]
        pk, pv = prefix_layers[layer]
        sk, sv = prompt_layers[layer]
        assert ok.shape[-2] == L_prefix + src_len

        torch.testing.assert_close(ok[..., :L_prefix, :], pk)
        torch.testing.assert_close(ov[..., :L_prefix, :], pv)
        torch.testing.assert_close(
            ok[..., L_prefix:, :], sk[..., src_start : src_start + src_len, :]
        )
        torch.testing.assert_close(
            ov[..., L_prefix:, :], sv[..., src_start : src_start + src_len, :]
        )


def test_lift_does_not_mutate_inputs():
    n_layers, B, H, D = 2, 1, 2, 4
    L_prefix, L_prompt = 4, 10

    prefix_kv = _make_marked_cache(
        n_layers=n_layers, B=B, H=H, seq=L_prefix, D=D, base=0.0,
    )
    prompt_kv = _make_marked_cache(
        n_layers=n_layers, B=B, H=H, seq=L_prompt, D=D, base=100.0,
    )

    # Snapshot inputs.
    prefix_snap = [(k.clone(), v.clone()) for (k, v) in layer_keys_values(prefix_kv)]
    prompt_snap = [(k.clone(), v.clone()) for (k, v) in layer_keys_values(prompt_kv)]

    _ = lift_prompt_kv(prefix_kv, prompt_kv, src_start=2, src_len=3)

    for i, (k, v) in enumerate(layer_keys_values(prefix_kv)):
        torch.testing.assert_close(k, prefix_snap[i][0])
        torch.testing.assert_close(v, prefix_snap[i][1])
    for i, (k, v) in enumerate(layer_keys_values(prompt_kv)):
        torch.testing.assert_close(k, prompt_snap[i][0])
        torch.testing.assert_close(v, prompt_snap[i][1])

    assert cache_length(prefix_kv) == L_prefix
    assert cache_length(prompt_kv) == L_prompt


def test_lift_zero_len_is_clone():
    n_layers, B, H, D = 2, 1, 1, 4
    prefix_kv = _make_marked_cache(
        n_layers=n_layers, B=B, H=H, seq=3, D=D, base=0.0,
    )
    prompt_kv = _make_marked_cache(
        n_layers=n_layers, B=B, H=H, seq=10, D=D, base=100.0,
    )

    out = lift_prompt_kv(prefix_kv, prompt_kv, src_start=5, src_len=0)
    assert cache_length(out) == cache_length(prefix_kv)
    # mutate out, prefix_kv must remain unchanged
    for i, (k, v) in enumerate(layer_keys_values(out)):
        set_layer_keys_values(out, i, k * 0.0, v * 0.0)
    prefix_layers = list(layer_keys_values(prefix_kv))
    for layer in range(n_layers):
        pk, pv = prefix_layers[layer]
        assert pk.abs().sum() > 0  # not zeroed
        assert pv.abs().sum() > 0


def test_lift_layer_count_mismatch_raises():
    a = _make_marked_cache(n_layers=2, B=1, H=1, seq=3, D=4, base=0.0)
    b = _make_marked_cache(n_layers=3, B=1, H=1, seq=10, D=4, base=100.0)
    with pytest.raises(ValueError, match="layer-count mismatch"):
        lift_prompt_kv(a, b, src_start=0, src_len=1)


def test_lift_out_of_range_src_raises():
    a = _make_marked_cache(n_layers=1, B=1, H=1, seq=3, D=4, base=0.0)
    b = _make_marked_cache(n_layers=1, B=1, H=1, seq=5, D=4, base=100.0)
    with pytest.raises(ValueError, match="prompt KV has only"):
        lift_prompt_kv(a, b, src_start=3, src_len=5)


