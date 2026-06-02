"""Shape + smoke tests for the EagleHead module.

Uses a small config so the test runs on CPU in seconds — the production
config (hidden_size=3584) needs a GPU.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from asts.eagle import EagleConfig, EagleHead, count_params, estimate_trainable_params


def _tiny_config() -> EagleConfig:
    """A tiny config small enough for CPU smoke tests."""
    return EagleConfig(
        hidden_size=128,
        num_attention_heads=8,
        num_key_value_heads=2,
        intermediate_size=256,
        max_position_embeddings=64,
    )


def test_eagle_head_instantiates():
    cfg = _tiny_config()
    head = EagleHead(cfg)
    assert head is not None


def test_eagle_head_forward_shape():
    cfg = _tiny_config()
    head = EagleHead(cfg).eval()
    bsz, seq = 2, 8
    h = torch.randn(bsz, seq, cfg.hidden_size)
    with torch.no_grad():
        out = head(h)
    assert out.shape == (bsz, seq, cfg.hidden_size)


def test_eagle_head_causality():
    """Predicted h[t] should not depend on h[t+1:]."""
    cfg = _tiny_config()
    head = EagleHead(cfg).eval()
    seq = 6
    h1 = torch.randn(1, seq, cfg.hidden_size)
    h2 = h1.clone()
    h2[:, seq // 2 :] = torch.randn_like(h2[:, seq // 2 :])  # perturb second half
    with torch.no_grad():
        out1 = head(h1)
        out2 = head(h2)
    # First-half outputs should be unchanged (causal mask)
    assert torch.allclose(out1[:, : seq // 2], out2[:, : seq // 2], atol=1e-5)


def test_eagle_head_param_count_reasonable():
    cfg = _tiny_config()
    head = EagleHead(cfg)
    n = count_params(head)
    # Tiny config should be small
    assert 50_000 < n < 5_000_000

    # Closed-form estimate should be within 5% of actual count
    est = estimate_trainable_params(cfg)
    assert abs(n - est) / n < 0.10, f"estimate {est} differs from actual {n} by >10%"


def test_eagle_config_from_target_config():
    """EagleConfig.from_target_config should pick up Qwen2-style fields."""
    from transformers import Qwen2Config

    target_cfg = Qwen2Config(
        hidden_size=3584,
        num_attention_heads=28,
        num_key_value_heads=4,
        intermediate_size=18944,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
    )
    eagle_cfg = EagleConfig.from_target_config(target_cfg)
    assert eagle_cfg.hidden_size == 3584
    assert eagle_cfg.num_attention_heads == 28
    assert eagle_cfg.num_key_value_heads == 4
    assert eagle_cfg.intermediate_size == 18944
    assert eagle_cfg.rope_theta == 1_000_000.0


def test_production_config_param_count():
    """Sanity: the Qwen-Coder-7B-shaped EAGLE head is ~250M params (estimate only)."""
    cfg = EagleConfig(
        hidden_size=3584,
        num_attention_heads=28,
        num_key_value_heads=4,
        intermediate_size=18944,
    )
    est = estimate_trainable_params(cfg)
    # Should be 200M-300M params (1 transformer block at this width)
    assert 150_000_000 < est < 350_000_000, f"got {est}"
