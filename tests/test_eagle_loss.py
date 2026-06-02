"""Unit tests for the EAGLE training loss.

Pure tensor ops — runs on CPU in milliseconds.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from asts.eagle_train import eagle_loss


def test_eagle_loss_shape_smoke():
    B, T, H, V = 2, 8, 16, 32
    pred_h = torch.randn(B, T, H)
    target_h = torch.randn(B, T, H)
    pred_logits = torch.randn(B, T, V)
    target_logits = torch.randn(B, T, V)
    mask = torch.ones(B, T)

    loss, breakdown = eagle_loss(pred_h, pred_logits, target_h, target_logits, mask)
    assert loss.dim() == 0  # scalar
    assert "kl" in breakdown and "h_l1" in breakdown
    assert breakdown["loss"] >= 0


def test_eagle_loss_zero_when_pred_matches_target():
    B, T, H, V = 1, 4, 8, 16
    h = torch.randn(B, T, H)
    logits = torch.randn(B, T, V)
    mask = torch.ones(B, T)

    loss, breakdown = eagle_loss(h, logits, h, logits, mask)
    # KL(p || p) == 0 and h_l1 == 0 → total loss == 0
    assert loss.item() < 1e-5
    assert breakdown["kl"] < 1e-5
    assert breakdown["h_l1"] < 1e-5


def test_eagle_loss_padding_mask_excludes_pad():
    B, T, H, V = 1, 6, 4, 8
    pred_h = torch.zeros(B, T, H)
    target_h = torch.zeros(B, T, H)
    target_h[0, 4:] = 100.0  # huge values at padding positions
    pred_logits = torch.zeros(B, T, V)
    target_logits = torch.zeros(B, T, V)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]])  # last 2 positions padded

    loss, breakdown = eagle_loss(pred_h, pred_logits, target_h, target_logits, mask)
    # Even though target_h has huge values at masked positions, loss should
    # be ~0 because they're excluded
    assert breakdown["h_l1"] < 1.0
    assert loss.item() < 10.0


def test_eagle_loss_kl_dominates_when_mismatched():
    B, T, H, V = 1, 4, 8, 16
    pred_h = torch.zeros(B, T, H)
    target_h = torch.zeros(B, T, H)
    pred_logits = torch.zeros(B, T, V)  # uniform
    # Sharp target distribution → large KL when pred is uniform
    target_logits = torch.zeros(B, T, V)
    target_logits[..., 0] = 50.0
    mask = torch.ones(B, T)

    loss, breakdown = eagle_loss(
        pred_h, pred_logits, target_h, target_logits, mask, kl_weight=1.0, h_weight=0.0
    )
    # KL of uniform vs near-delta ≈ log(V) ≈ 2.77 for V=16
    assert breakdown["kl"] > 1.0
