from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from asts.decoder import vanilla_ar
from asts.blazedit_decoder import is_blazedit_method, parse_blazedit_method
from asts.vantage_residual_decoder import (
    VantageResidualConfig,
    _QueuedResidualDraft,
    _prefix_hash,
    is_vantage_residual_method,
    vantage_residual_decode,
    parse_vantage_residual_method,
    prompt_lookup_draft,
    queued_residual_invalid_reason,
)


ROOT = Path(__file__).resolve().parents[1]


class _ToyOutput:
    def __init__(self, logits):
        self.logits = logits
        self.past_key_values = None


class _IncrementModel(torch.nn.Module):
    def __init__(self, *, delta: int = 1, vocab_size: int = 128) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(()))
        self.config = type("Config", (), {"pad_token_id": 0})()
        self.delta = int(delta)
        self.vocab_size = int(vocab_size)

    def forward(self, input_ids, past_key_values=None, use_cache=True, **kwargs):
        del past_key_values, use_cache, kwargs
        batch, seq = input_ids.shape
        logits = torch.full((batch, seq, self.vocab_size), -1000.0)
        for batch_i in range(batch):
            for pos_i in range(seq):
                token = int(input_ids[batch_i, pos_i].item())
                logits[batch_i, pos_i, (token + self.delta) % self.vocab_size] = 1000.0
        return _ToyOutput(logits)


def test_parse_vantage_residual_method_names():
    assert parse_vantage_residual_method("vantage_residual_k1").residual_k == 1
    assert parse_vantage_residual_method("vantage_residual_k2").residual_k == 2
    assert parse_vantage_residual_method("vantage_residual_k4").residual_k == 4

    routed = parse_vantage_residual_method("router_k4")
    assert routed.residual_k == 4
    assert routed.residual_trigger == "router"
    assert routed.router_min_pld_draft_len == 4
    assert is_vantage_residual_method("vantage_residual_router_k4")
    pre = parse_vantage_residual_method("vantage_residual_preverify_replace_k4")
    assert pre.residual_k == 4
    assert pre.preverify_replace is True
    assert pre.residual_trigger == "router"
    queued = parse_vantage_residual_method("vantage_residual_queued_k4")
    assert queued.residual_k == 4
    assert queued.queued_residual is True
    assert queued.residual_trigger == "queued_conservative"
    assert not is_vantage_residual_method("vantage_residual_k3")


def test_blazedit_runtime_accepts_vantage_residual_aliases():
    cfg = parse_blazedit_method("vantage_residual_k2_t1_w64_n10")
    assert cfg.mode == "pld_plus_mtp_heads"
    assert cfg.mtp_num_heads == 2
    assert cfg.mtp_trigger_accepted_len == 1
    assert cfg.micro_draft_tokens == 64
    assert is_blazedit_method("vantage_residual_router_k4_w128_n10")


def test_local_pld_lookup_matches_blazedit_semantics_for_basic_cases():
    draft, match_len, source_start, follow_start = prompt_lookup_draft(
        [1, 2, 3, 9, 8, 1, 2, 3],
        max_matching_ngram_size=3,
        max_draft_tokens=4,
    )

    assert draft == [9, 8]
    assert match_len == 3
    assert source_start == 0
    assert follow_start == 3


def test_residual_disabled_matches_pld_only_path():
    prompt = torch.tensor([1, 2, 3, 9, 8, 1, 2, 3], dtype=torch.long)
    target = _IncrementModel()
    residual = _IncrementModel()
    base_config = VantageResidualConfig(
        method_name="vantage_residual_k4",
        residual_enabled=False,
        residual_k=4,
        pld_max_matching_ngram_size=3,
        pld_max_draft_tokens=4,
    )
    enabled_config = VantageResidualConfig(
        method_name="vantage_residual_k4",
        residual_enabled=True,
        residual_k=4,
        pld_max_matching_ngram_size=3,
        pld_max_draft_tokens=4,
    )

    pld_only = vantage_residual_decode(
        prompt,
        target,
        max_new_tokens=8,
        eos_token_ids=[],
        config=base_config,
        residual_model=residual,
    )
    trigger_false = vantage_residual_decode(
        prompt,
        target,
        max_new_tokens=8,
        eos_token_ids=[],
        config=enabled_config,
        residual_model=residual,
        residual_trigger=lambda _prefix, _pld, _cfg: False,
    )

    assert trigger_false.output_token_ids == pld_only.output_token_ids
    assert [s.n_emitted for s in trigger_false.steps] == [s.n_emitted for s in pld_only.steps]
    assert all(step.pld_variant_triggered is False for step in trigger_false.steps)


def test_residual_exact_drafts_are_target_verified_and_match_greedy():
    prompt = torch.tensor([1], dtype=torch.long)
    target = _IncrementModel()
    residual = _IncrementModel()
    config = VantageResidualConfig(
        method_name="vantage_residual_k4",
        residual_enabled=True,
        residual_k=4,
        residual_trigger="always",
    )

    greedy = vanilla_ar(prompt, target, max_new_tokens=8, eos_token_ids=[])
    hybrid = vantage_residual_decode(
        prompt,
        target,
        max_new_tokens=8,
        eos_token_ids=[],
        config=config,
        residual_model=residual,
    )

    assert hybrid.output_token_ids == greedy.output_token_ids
    assert any(step.proposal_kind == "vantage_residual" for step in hybrid.steps)
    assert any((step.n_accepted_drafts or 0) > 0 for step in hybrid.steps)


def test_bad_residual_draft_is_rejected_with_target_correction():
    prompt = torch.tensor([1], dtype=torch.long)
    target = _IncrementModel(delta=1)
    bad_residual = _IncrementModel(delta=2)
    config = VantageResidualConfig(
        method_name="vantage_residual_k2",
        residual_enabled=True,
        residual_k=2,
        residual_trigger="always",
    )

    greedy = vanilla_ar(prompt, target, max_new_tokens=4, eos_token_ids=[])
    hybrid = vantage_residual_decode(
        prompt,
        target,
        max_new_tokens=4,
        eos_token_ids=[],
        config=config,
        residual_model=bad_residual,
    )

    assert hybrid.output_token_ids == greedy.output_token_ids
    assert hybrid.steps[0].proposal_kind == "vantage_residual"
    assert hybrid.steps[0].rejected is True
    assert hybrid.steps[0].n_accepted_drafts == 0
    assert hybrid.steps[0].n_emitted == 1
    assert hybrid.steps[0].pld_token01_rejection is True


def test_router_mode_triggers_when_pld_is_short_and_keeps_exact_output():
    prompt = torch.tensor([1], dtype=torch.long)
    target = _IncrementModel()
    residual = _IncrementModel()
    config = parse_vantage_residual_method("router_k4")

    greedy = vanilla_ar(prompt, target, max_new_tokens=6, eos_token_ids=[])
    routed = vantage_residual_decode(
        prompt,
        target,
        max_new_tokens=6,
        eos_token_ids=[],
        config=config,
        residual_model=residual,
    )

    assert routed.output_token_ids == greedy.output_token_ids
    assert routed.steps
    assert routed.steps[0].pld_variant_triggered is True


def test_residual_fake_model_repeatability():
    prompt = torch.tensor([1, 2, 3, 9, 8, 1, 2, 3], dtype=torch.long)
    config = VantageResidualConfig(
        method_name="vantage_residual_k2",
        residual_enabled=True,
        residual_k=2,
        residual_trigger="always",
        pld_max_matching_ngram_size=3,
        pld_max_draft_tokens=4,
    )

    first = vantage_residual_decode(
        prompt,
        _IncrementModel(),
        max_new_tokens=8,
        eos_token_ids=[],
        config=config,
        residual_model=_IncrementModel(),
    )
    second = vantage_residual_decode(
        prompt,
        _IncrementModel(),
        max_new_tokens=8,
        eos_token_ids=[],
        config=config,
        residual_model=_IncrementModel(),
    )

    def stable_steps(result):
        return [
            (
                step.proposal_kind,
                step.n_accepted_drafts,
                step.n_emitted,
                step.rejected,
                step.pld_variant_triggered,
            )
            for step in result.steps
        ]

    assert second.output_token_ids == first.output_token_ids
    assert stable_steps(second) == stable_steps(first)


def test_residual_eval_script_toy_smoke(tmp_path):
    output = tmp_path / "residual.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_vantage_residual_eval.py"),
            "--method",
            "vantage_residual_k2",
            "--max-new-tokens",
            "4",
            "--output-json",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["method"] == "vantage_residual_k2"
    assert payload["output_token_ids"] == [1, 2, 3, 4, 5]
    assert payload["steps"]


def test_preverify_replace_uses_one_target_verifier_path_and_matches_greedy():
    prompt = torch.tensor([1], dtype=torch.long)
    target = _IncrementModel()
    residual = _IncrementModel()
    config = VantageResidualConfig(
        method_name="vantage_residual_preverify_replace_k4",
        residual_enabled=True,
        residual_k=4,
        residual_trigger="always",
        preverify_replace=True,
    )

    greedy = vanilla_ar(prompt, target, max_new_tokens=8, eos_token_ids=[])
    hybrid = vantage_residual_decode(
        prompt,
        target,
        max_new_tokens=8,
        eos_token_ids=[],
        config=config,
        residual_model=residual,
    )

    assert hybrid.output_token_ids == greedy.output_token_ids
    assert any(step.proposal_kind == "vantage_residual_preverify_replace" for step in hybrid.steps)
    assert all(step.mtp_extra_verify_calls == 0 for step in hybrid.steps)
    assert all(step.mtp_verify_extra_us == 0.0 for step in hybrid.steps)
    assert all(
        step.mtp_normal_verify_reuse is True
        for step in hybrid.steps
        if step.proposal_kind == "vantage_residual_preverify_replace"
    )


def test_preverify_replace_rejects_bad_residual_without_unverified_emission():
    prompt = torch.tensor([1], dtype=torch.long)
    target = _IncrementModel(delta=1)
    bad_residual = _IncrementModel(delta=2)
    config = VantageResidualConfig(
        method_name="vantage_residual_preverify_replace_k2",
        residual_enabled=True,
        residual_k=2,
        residual_trigger="always",
        preverify_replace=True,
    )

    greedy = vanilla_ar(prompt, target, max_new_tokens=4, eos_token_ids=[])
    hybrid = vantage_residual_decode(
        prompt,
        target,
        max_new_tokens=4,
        eos_token_ids=[],
        config=config,
        residual_model=bad_residual,
    )

    assert hybrid.output_token_ids == greedy.output_token_ids
    assert hybrid.steps[0].proposal_kind == "vantage_residual_preverify_replace"
    assert hybrid.steps[0].rejected is True
    assert hybrid.steps[0].n_accepted_drafts == 0
    assert hybrid.steps[0].n_emitted == 1
    assert hybrid.steps[0].mtp_extra_verify_calls == 0


def test_preverify_replace_refuses_post_verification_trigger_policy():
    prompt = torch.tensor([1], dtype=torch.long)
    target = _IncrementModel()
    residual = _IncrementModel()
    config = VantageResidualConfig(
        method_name="vantage_residual_preverify_replace_k4",
        residual_enabled=True,
        residual_k=4,
        residual_trigger="accepted_len_le_4",
        preverify_replace=True,
    )

    with pytest.raises(ValueError, match="cannot use post-verification trigger"):
        vantage_residual_decode(
            prompt,
            target,
            max_new_tokens=4,
            eos_token_ids=[],
            config=config,
            residual_model=residual,
        )


def test_queued_residual_uses_previous_draft_once_and_matches_greedy():
    prompt = torch.tensor([1], dtype=torch.long)
    target = _IncrementModel()
    residual = _IncrementModel()
    config = VantageResidualConfig(
        method_name="vantage_residual_queued_k4",
        residual_enabled=True,
        residual_k=4,
        residual_trigger="always",
        queued_residual=True,
    )

    greedy = vanilla_ar(prompt, target, max_new_tokens=8, eos_token_ids=[])
    hybrid = vantage_residual_decode(
        prompt,
        target,
        max_new_tokens=8,
        eos_token_ids=[],
        config=config,
        residual_model=residual,
    )

    assert hybrid.output_token_ids == greedy.output_token_ids
    assert any(step.mtp_queue_prediction_created is True for step in hybrid.steps)
    assert any(step.queued_used is True for step in hybrid.steps)
    assert any(step.proposal_kind == "vantage_residual_queued" for step in hybrid.steps)
    assert all(step.verifier_calls_this_step == 1 for step in hybrid.steps)
    assert all(step.mtp_extra_verify_calls == 0 for step in hybrid.steps)
    assert all(step.mtp_verify_extra_us == 0.0 for step in hybrid.steps)


def test_queued_residual_bad_draft_rejected_without_unverified_emission():
    prompt = torch.tensor([1], dtype=torch.long)
    target = _IncrementModel(delta=1)
    bad_residual = _IncrementModel(delta=2)
    config = VantageResidualConfig(
        method_name="vantage_residual_queued_k2",
        residual_enabled=True,
        residual_k=2,
        residual_trigger="always",
        queued_residual=True,
    )

    greedy = vanilla_ar(prompt, target, max_new_tokens=5, eos_token_ids=[])
    hybrid = vantage_residual_decode(
        prompt,
        target,
        max_new_tokens=5,
        eos_token_ids=[],
        config=config,
        residual_model=bad_residual,
    )

    assert hybrid.output_token_ids == greedy.output_token_ids
    used = [step for step in hybrid.steps if step.queued_used is True]
    assert used
    assert any(step.rejected is True and step.n_accepted_drafts == 0 for step in used)
    assert all(step.n_emitted >= 1 for step in used)
    assert all(step.mtp_extra_verify_calls == 0 for step in used)


def test_queued_residual_invalid_reasons_are_strict():
    prefix = [1, 2, 3]
    queued = _QueuedResidualDraft(
        position=3,
        prefix_hash=_prefix_hash(prefix),
        draft_tokens=[4, 5],
        source_step=0,
    )

    assert queued_residual_invalid_reason(prefix, queued) is None
    assert queued_residual_invalid_reason([1, 2], queued) == "position_mismatch"
    assert queued_residual_invalid_reason([1, 2, 9], queued) == "prefix_hash_mismatch"
    assert queued_residual_invalid_reason(prefix, queued, eos_hit=True) == "eos"
    assert queued_residual_invalid_reason(prefix, queued, max_length_hit=True) == "max_length"
    assert (
        queued_residual_invalid_reason(
            prefix,
            queued,
            task_id="task_b",
            queued_task_id="task_a",
        )
        == "task_id_mismatch"
    )
