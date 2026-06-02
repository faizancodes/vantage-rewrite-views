from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from asts.residual_mtp import (
    ResidualMTPConfig,
    ResidualMTPHeads,
    accepted_prefix_lengths,
    accuracy_by_head,
    filter_residual_dataset,
    infer_vocab_size,
    residual_cross_entropy,
    should_trigger_residual,
    trigger_mask,
    validate_residual_dataset,
)


def _payload():
    accepted = torch.tensor([0, 1, 2, 5], dtype=torch.long)
    labels = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.long)
    return {
        "hidden": torch.randn(4, 3),
        "labels": labels,
        "accepted_len": accepted,
        "pld_miss": accepted == 0,
        "trigger_accepted_len_eq_0": accepted == 0,
        "trigger_accepted_len_le_1": accepted <= 1,
        "trigger_accepted_len_le_2": accepted <= 2,
        "trigger_accepted_len_le_4": accepted <= 4,
        "metadata": {"mtp_position": "post_pld"},
    }


def test_validate_and_filter_residual_dataset():
    payload = _payload()
    validate_residual_dataset(payload)
    mask = trigger_mask(payload, "accepted_len_le_2")
    assert mask.tolist() == [True, True, True, False]
    filtered = filter_residual_dataset(payload, "accepted_len_le_1")
    assert filtered["hidden"].shape[0] == 2
    assert filtered["metadata"]["residual_filter_policy"] == "accepted_len_le_1"


def test_residual_trigger_policies():
    assert should_trigger_residual("accepted_len_eq_0", accepted_len=0)
    assert not should_trigger_residual("accepted_len_eq_0", accepted_len=1)
    assert should_trigger_residual("accepted_len_le_4", accepted_len=4)
    assert not should_trigger_residual("accepted_len_le_4", accepted_len=5)
    assert should_trigger_residual("pld_miss_only", accepted_len=3, pld_miss=True)
    assert should_trigger_residual("always", accepted_len=99)
    assert not should_trigger_residual("never", accepted_len=0)


def test_residual_head_shapes_and_metrics():
    labels = torch.tensor([[1, 2], [1, 3]], dtype=torch.long)
    config = ResidualMTPConfig(hidden_size=4, vocab_size=infer_vocab_size(labels), num_heads=2)
    model = ResidualMTPHeads(config)
    logits = model.forward_logits(torch.randn(2, 4))
    assert logits.shape == (2, 2, config.vocab_size)
    loss = residual_cross_entropy(logits, labels)
    assert loss.item() >= 0.0

    pred = torch.tensor([[1, 2], [0, 3]], dtype=torch.long)
    assert accuracy_by_head(pred, labels) == [0.5, 1.0]
    assert accepted_prefix_lengths(pred, labels).tolist() == [2, 0]
