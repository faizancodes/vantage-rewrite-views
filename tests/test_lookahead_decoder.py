import torch

from asts.blazedit_decoder import (
    BlazEditConfig,
    _LookaheadState,
    _lookahead_jacobi_draft,
    blazedit_speculative_ar,
)
from asts.decoder import vanilla_ar


class _ToyOutput:
    def __init__(self, logits):
        self.logits = logits
        self.past_key_values = None


class _IncrementModel(torch.nn.Module):
    def __init__(self, vocab_size=64):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(()))
        self.config = type("Config", (), {"pad_token_id": 0})()
        self.vocab_size = vocab_size

    def forward(self, input_ids, past_key_values=None, use_cache=True, **kwargs):
        del past_key_values, use_cache, kwargs
        batch, seq = input_ids.shape
        logits = torch.full((batch, seq, self.vocab_size), -1000.0)
        for b in range(batch):
            for i in range(seq):
                tok = int(input_ids[b, i].item())
                logits[b, i, (tok + 1) % self.vocab_size] = 1000.0
        return _ToyOutput(logits)


def test_lookahead_jacobi_candidate_returns_token_ids():
    target = _IncrementModel()
    cfg = BlazEditConfig(
        mode="lookahead",
        lookahead_window=4,
        lookahead_ngram=2,
        lookahead_iters=4,
        lookahead_max_draft=4,
    )

    stats = _lookahead_jacobi_draft(
        prefix=[1],
        target=target,
        target_cache=None,
        target_cache_len=0,
        config=cfg,
        state=_LookaheadState(),
    )

    assert stats.forward_calls == 4
    assert stats.drafts[:3] == [2, 3, 4]
    assert all(isinstance(tok, int) for tok in stats.drafts)


def test_lookahead_one_forward_uses_single_extra_forward():
    target = _IncrementModel()
    cfg = BlazEditConfig(
        mode="lookahead",
        lookahead_window=4,
        lookahead_ngram=2,
        lookahead_iters=4,
        lookahead_max_draft=4,
        lookahead_one_forward=True,
    )

    stats = _lookahead_jacobi_draft(
        prefix=[1],
        target=target,
        target_cache=None,
        target_cache_len=0,
        config=cfg,
        state=_LookaheadState(),
    )

    assert stats.forward_calls == 1
    assert stats.forward_us > 0
    assert stats.lookahead_us >= stats.forward_us
    assert len(stats.drafts) <= 2


def test_lookahead_decoder_matches_greedy_on_toy_model():
    target = _IncrementModel()
    prompt = torch.tensor([1], dtype=torch.long)
    cfg = BlazEditConfig(
        mode="lookahead",
        lookahead_window=4,
        lookahead_ngram=2,
        lookahead_iters=4,
        lookahead_max_draft=4,
    )

    greedy = vanilla_ar(prompt, target, max_new_tokens=8, eos_token_ids=[])
    lookahead = blazedit_speculative_ar(
        prompt,
        target,
        assistant=None,
        max_new_tokens=8,
        eos_token_ids=[],
        config=cfg,
        method_name="lookahead_w4_n2_i4",
    )

    assert lookahead.output_token_ids == greedy.output_token_ids
    assert any(step.lookahead_triggered for step in lookahead.steps)


def test_pld_gated_lookahead_fallback_and_verified_output_on_toy_model():
    target = _IncrementModel()
    prompt = torch.tensor([1], dtype=torch.long)
    cfg = BlazEditConfig(
        mode="pld_gated_lookahead",
        micro_draft_tokens=8,
        max_matching_ngram_size=10,
        lookahead_window=4,
        lookahead_ngram=2,
        lookahead_iters=4,
        lookahead_max_draft=4,
        pld_lookahead_router="rule",
        pld_lookahead_weak_threshold=4,
    )

    greedy = vanilla_ar(prompt, target, max_new_tokens=8, eos_token_ids=[])
    gated = blazedit_speculative_ar(
        prompt,
        target,
        assistant=None,
        max_new_tokens=8,
        eos_token_ids=[],
        config=cfg,
        method_name="pld_gated_lookahead_w128_n10",
    )

    assert gated.output_token_ids == greedy.output_token_ids
    assert any(step.pld_lookahead_predicted_weak for step in gated.steps)
    assert all(
        step.n_emitted == step.n_accepted_drafts + 1
        for step in gated.steps
        if step.lookahead_triggered and not step.hit_max_new_tokens
    )


def test_pld_gated_lookahead_one_forward_metrics_on_toy_model():
    target = _IncrementModel()
    prompt = torch.tensor([1], dtype=torch.long)
    cfg = BlazEditConfig(
        mode="pld_gated_lookahead",
        micro_draft_tokens=8,
        max_matching_ngram_size=10,
        lookahead_window=4,
        lookahead_ngram=2,
        lookahead_iters=4,
        lookahead_max_draft=4,
        lookahead_one_forward=True,
        pld_lookahead_router="rule",
        pld_lookahead_weak_threshold=4,
    )

    gated = blazedit_speculative_ar(
        prompt,
        target,
        assistant=None,
        max_new_tokens=6,
        eos_token_ids=[],
        config=cfg,
        method_name="pld_gated_lookahead_w4_n2_i1_d4",
    )

    triggered = [step for step in gated.steps if step.lookahead_triggered]
    assert triggered
    assert all(step.lookahead_forward_calls == 1 for step in triggered)
    assert all((step.lookahead_accepted_per_forward or 0.0) >= 0.0 for step in triggered)
