"""Tests for the greedy rejection rule.

We construct synthetic logits so that target's argmax at each position is
deterministic, then verify the acceptance/correction/bonus mechanics.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from asts.rejection import greedy_verify


def _make_logits(target_argmaxes, vocab=100):
    """Build logits where argmax at each position is the given token id."""
    seq = len(target_argmaxes)
    logits = torch.zeros(seq, vocab)
    for i, t in enumerate(target_argmaxes):
        logits[i, t] = 10.0  # pump up the target token's logit
    return logits


def test_all_drafts_accepted():
    # Target predicts: [10, 20, 30] for positions 0, 1, 2 (drafts) and 99 at the bonus position
    drafts = [10, 20, 30]
    logits = _make_logits([10, 20, 30, 99])  # n_pre=1, then 3 draft positions, but bonus is at last
    # Wait: n_pre=1 means logits[0] predicts drafts[0], logits[1] predicts drafts[1], logits[2] predicts drafts[2], logits[3] predicts bonus.
    result = greedy_verify(drafts=drafts, target_logits=logits, n_pre=1)
    assert result.n_accepted_drafts == 3
    assert result.rejected is False
    assert result.accepted_tokens == [10, 20, 30, 99]  # 3 drafts + bonus


def test_first_draft_rejected():
    drafts = [10, 20, 30]
    # Target predicts 7 at the position where draft proposed 10 → reject immediately
    logits = _make_logits([7, 20, 30, 99])
    result = greedy_verify(drafts=drafts, target_logits=logits, n_pre=1)
    assert result.n_accepted_drafts == 0
    assert result.rejected is True
    assert result.accepted_tokens == [7]  # just the correction


def test_middle_draft_rejected():
    drafts = [10, 20, 30]
    # Target accepts position 0 (10) and 1 (20) but rejects 2 (predicts 50 instead of 30)
    logits = _make_logits([10, 20, 50, 99])
    result = greedy_verify(drafts=drafts, target_logits=logits, n_pre=1)
    assert result.n_accepted_drafts == 2
    assert result.rejected is True
    assert result.accepted_tokens == [10, 20, 50]  # 2 accepted drafts + 1 correction


def test_n_pre_anchoring():
    # When n_pre > 1, predictions for drafts start at logits[n_pre - 1]
    drafts = [10, 20]
    # logits has shape [4, V]; n_pre=2; so logits[1], logits[2] predict drafts; logits[3] is bonus
    logits = _make_logits([99, 10, 20, 77])  # logits[0] is "anchor" prefix prediction (irrelevant)
    result = greedy_verify(drafts=drafts, target_logits=logits, n_pre=2)
    assert result.n_accepted_drafts == 2
    assert result.rejected is False
    assert result.accepted_tokens == [10, 20, 77]


def test_n_pre_zero_raises():
    with pytest.raises(ValueError):
        greedy_verify(drafts=[1], target_logits=_make_logits([1]), n_pre=0)


def test_too_few_logits_raises():
    # 2 drafts + n_pre=1 needs at least 3 logit positions; supply 2 → error
    with pytest.raises(ValueError):
        greedy_verify(drafts=[1, 2], target_logits=_make_logits([1, 2]), n_pre=1)


def test_3d_logits_squeeze_batch():
    # Accepts [1, seq, V] tensors (typical model output)
    drafts = [10]
    logits = _make_logits([10, 99]).unsqueeze(0)  # [1, 2, V]
    result = greedy_verify(drafts=drafts, target_logits=logits, n_pre=1)
    assert result.accepted_tokens == [10, 99]
