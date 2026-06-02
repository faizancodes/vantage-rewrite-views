"""Custom AR + speculative decoding loop for ASTS-Spec evaluation.

Three modes share a single outer driver, differing only in how each outer
step's `step_result` is produced:

  - `"vanilla"`     : single-token greedy AR (the lossless baseline)
  - `"fixed_spec"`  : speculative decode with a fixed draft length k
  - `"asts_spec"`   : speculative decode with k chosen per AST node type

Cache invariants
----------------

To avoid the "no anchor for predicting drafts[0]" edge case, the speculative
modes maintain `target_cache_len < len(prefix)` always (the target's cache
covers all but the last accepted token). The vanilla mode uses
`cache_len == len(prefix)` (no cross-position lookups needed).

Lossless property (greedy)
--------------------------

In greedy mode, all three modes produce **byte-identical** output for the
same prompt. This is verified by `verify_lossless.py`. Floating-point
non-determinism in CUDA can occasionally cause near-tie argmax differences;
deterministic mode is enabled in the lossless test.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import torch

from .rejection import GreedyVerifyResult, greedy_verify


# ---------------------------------------------------------------------------
# Cache utilities
# ---------------------------------------------------------------------------


def crop_dynamic_cache(cache, target_length: int) -> None:
    """Crop a HuggingFace DynamicCache (or compatible) in place.

    Tries the official `crop()` method (transformers >= 4.39); falls back to
    manual slicing of key/value tensors at the seq dim.
    """
    if cache is None:
        return
    if hasattr(cache, "crop"):
        cache.crop(target_length)
        return
    # Manual fallback. Key/value layout is [batch, heads, seq, head_dim].
    if hasattr(cache, "key_cache"):
        for i in range(len(cache.key_cache)):
            if cache.key_cache[i] is not None:
                cache.key_cache[i] = cache.key_cache[i][..., :target_length, :].contiguous()
            if cache.value_cache[i] is not None:
                cache.value_cache[i] = cache.value_cache[i][..., :target_length, :].contiguous()
    if hasattr(cache, "_seen_tokens"):
        cache._seen_tokens = target_length


# ---------------------------------------------------------------------------
# Per-step record (logged to JSONL)
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    method: str
    step: int
    k: int
    # Accepted speculative candidate positions for this outer step. For
    # vanilla draft-model speculation this is the usual accepted-draft count
    # (0..k). EAGLE-1 chain/tree-tail and retrieval variants use the target's
    # cached argmax as the first candidate, so their count is at least 1
    # unless truncated by the generation budget. EAGLE-2 records accepted
    # non-root children; see eagle2_decoder.py for that convention.
    n_accepted_drafts: int
    # Tokens actually appended to the prefix after EOS/budget truncation. In
    # the common non-truncated case this is n_accepted_drafts + 1: either a
    # correction at the first rejected candidate or a verify-forward bonus.
    n_emitted: int
    rejected: bool
    node_type: str | None
    deepest_type: str | None
    wall_us: float
    draft_us: float = 0.0
    verify_us: float = 0.0
    parse_us: float = 0.0
    strategy: str | None = None
    visibility: float | None = None
    frontier_depth: int | None = None
    zone: str | None = None
    draft_top1_prob: float | None = None
    draft_top2_margin: float | None = None
    retrieval_match_len: int | None = None
    rolling_accept_rate: float | None = None
    ancestor_types: tuple[str, ...] | None = None
    parser_in_error: bool | None = None
    confidence_us: float = 0.0
    retrieval_us: float = 0.0
    scope_us: float = 0.0
    route_us: float = 0.0
    target_prefill_us: float = 0.0
    proposal_kind: str | None = None
    proposal_match_len: int | None = None
    proposal_us: float = 0.0
    proposal_tokens: int | None = None
    n_guaranteed_drafts: int = 0
    n_accepted_nonroot_drafts: int | None = None
    hit_max_new_tokens: bool = False
    prompt_len: int | None = None
    proposal_source_start_token: int | None = None
    proposal_source_end_token: int | None = None
    proposal_follow_start_token: int | None = None
    proposal_follow_end_token: int | None = None
    proposal_query_len: int | None = None
    proposal_pool: str | None = None
    proposal_source_region: str | None = None
    proposal_root_included: bool | None = None
    proposal_match_kind: str | None = None
    proposal_canonical_match_len: int | None = None
    proposal_substitution_count: int | None = None
    proposal_scope_fill_count: int | None = None
    proposal_stopped_on_unmapped: bool | None = None
    proposal_alpha_exact_filtered: bool | None = None
    proposal_zero_nonroot_accept: bool | None = None
    proposal_tree_nodes: int | None = None
    proposal_tree_candidates: int | None = None
    proposal_tree_branch_depth: int | None = None
    proposal_tree_branch_selected: int | None = None
    assistant_model: str | None = None
    assistant_us: float = 0.0
    assistant_prefill_us: float = 0.0
    assistant_pld_us: float = 0.0
    assistant_verify_us: float = 0.0
    blazedit_micro_runs: int | None = None
    blazedit_micro_draft_tokens: int | None = None
    blazedit_max_num_run: int | None = None
    blazedit_pld_proposed: int | None = None
    blazedit_pld_accepted: int | None = None
    target_draft_tokens: int | None = None
    target_accepted_nonroot: int | None = None
    assistant_cache_catchup_tokens: int | None = None
    proposal_map_source: str | None = None
    proposal_inferred_map_count: int | None = None
    proposal_inference_confidence: float | None = None
    proposal_cursor_pos: int | None = None
    proposal_cursor_confidence: float | None = None
    proposal_cursor_resync: bool | None = None
    proposal_view_id: str | None = None
    proposal_compound_view_count: int | None = None
    proposal_active_map_count: int | None = None
    proposal_route: str | None = None
    proposal_route_reason: str | None = None
    proposal_backoff_active: bool | None = None
    proposal_rewrite_hit_count: int | None = None
    proposal_route_window_accept_rate: float | None = None
    proposal_rewrite_zero_accept_streak: int | None = None
    proposal_adoption_state: str | None = None
    proposal_adoption_transition: str | None = None
    proposal_frontier_distance: int | None = None
    proposal_frontier_probes: int | None = None
    proposal_accepted_crossed_rewrite: int | None = None
    proposal_rejected_old_form_frontiers: int | None = None
    proposal_blacklisted_rewrite_occurrences: int | None = None
    proposal_disabled_by_adoption_gate: bool | None = None
    proposal_root_old_match_count: int | None = None
    proposal_root_new_match_count: int | None = None
    proposal_map_parse_us: float = 0.0
    proposal_rewrite_apply_us: float = 0.0
    proposal_virtual_reference_tokenize_us: float = 0.0
    proposal_transpld_index_build_us: float = 0.0
    proposal_text_preview: str | None = None
    proposal_first_token: int | None = None
    proposal_first_token_text: str | None = None
    proposal_target_reject_token: int | None = None
    proposal_target_reject_token_text: str | None = None
    proposal_target_reject_index: int | None = None
    verify_staged_chunks: int | None = None
    verify_staged_draft_tokens: int | None = None
    verify_staged_saved_tokens: int | None = None
    proposal_neural_draft_tokens: int | None = None
    proposal_neural_draft_us: float = 0.0
    pld_variant: str | None = None
    pld_exact_hit: bool | None = None
    pld_variant_triggered: bool | None = None
    pld_variant_overhead_us: float = 0.0
    pld_candidate_accepted_len: int | None = None
    pld_token01_rejection: bool | None = None
    pld_delta_cache_size: int | None = None
    pld_delta_patch_count: int | None = None
    pld_delta_patch_accepted: bool | None = None
    pld_delta_patch_accept_tail: int | None = None
    pld_fuzzy_candidate_count: int | None = None
    pld_fuzzy_edit_distance: int | None = None
    pld_fuzzy_match_len: int | None = None
    pld_opp_trace: bool | None = None
    pld_opp_step_id: int | None = None
    pld_opp_exact_hit: bool | None = None
    pld_opp_candidate_matches: int | None = None
    pld_opp_source_position: int | None = None
    pld_opp_draft_len: int | None = None
    pld_opp_accepted_len: int | None = None
    pld_opp_rejected_at_position: int | None = None
    pld_opp_target_token_at_rejection: int | None = None
    pld_opp_pld_token_at_rejection: int | None = None
    pld_opp_lookup_us: float = 0.0
    pld_opp_verify_us: float = 0.0
    pld_opp_generated_suffix_16_text: str | None = None
    pld_opp_draft_prefix_32_text: str | None = None
    pld_opp_source_snippet_text: str | None = None
    pld_rerank_triggered: bool | None = None
    pld_rerank_ambiguous: bool | None = None
    pld_rerank_candidate_count: int | None = None
    pld_rerank_selected_rank: int | None = None
    pld_rerank_fallback: bool | None = None
    pld_rerank_overhead_us: float = 0.0
    pld_rerank_baseline_rank: int | None = None
    pld_rerank_selected_is_baseline: bool | None = None
    pld_rerank_selected_score: float | None = None
    pld_rerank_baseline_score: float | None = None
    pld_rerank_score_margin: float | None = None
    pld_rerank_baseline_score_missing: bool | None = None
    pld_rerank_candidate_positions: list[int] | None = None
    pld_rerank_candidate_source_kinds: list[str] | None = None
    pld_rerank_debug_features: list[dict[str, object]] | None = None
    mtp_triggered: bool | None = None
    mtp_token0_rejected: bool | None = None
    mtp_accepted_prefix_len: int | None = None
    mtp_actual_extra_progress: int = 0
    mtp_extra_accepted_drafts: int = 0
    mtp_head_compute_us: float = 0.0
    mtp_verify_extra_us: float = 0.0
    mtp_total_overhead_us: float = 0.0
    mtp_predicted_tokens: int | None = None
    mtp_verified_draft_tokens: int | None = None
    mtp_queue_prediction_created: bool | None = None
    mtp_queue_prediction_used: bool | None = None
    mtp_queue_dropped_pld_strong: bool | None = None
    mtp_queue_dropped_position_mismatch: bool | None = None
    mtp_queue_expired: bool | None = None
    mtp_used_token0_rejected: bool | None = None
    mtp_extra_verify_calls: int = 0
    mtp_normal_verify_reuse: bool | None = None
    queued_available: bool | None = None
    queued_used: bool | None = None
    queued_invalid_reason: str | None = None
    prefix_hash_created: str | None = None
    prefix_hash_used: str | None = None
    residual_confidence: float | None = None
    queued_draft_len: int | None = None
    verifier_calls_this_step: int | None = None
    hidden_capture_us: float = 0.0
    residual_head_us: float = 0.0
    pld_cap_variant: str | None = None
    pld_cap_router_probability: float | None = None
    pld_cap_router_predicted_weak: bool | None = None
    pld_cap_threshold: float | None = None
    pld_cap_value: int | None = None
    pld_cap_triggered: bool | None = None
    pld_cap_raw_draft_len: int | None = None
    pld_cap_capped_draft_len: int | None = None
    pld_cap_wasted_verified_tokens: int | None = None
    pld_cap_router_us: float = 0.0
    lookahead_triggered: bool | None = None
    lookahead_candidate_len: int | None = None
    lookahead_accepted_len: int | None = None
    lookahead_tok0_reject: bool | None = None
    lookahead_iters: int | None = None
    lookahead_forward_calls: int | None = None
    lookahead_us: float = 0.0
    lookahead_forward_us: float = 0.0
    lookahead_candidate_build_us: float = 0.0
    lookahead_verify_us: float = 0.0
    lookahead_accepted_per_forward: float | None = None
    lookahead_window: int | None = None
    lookahead_ngram: int | None = None
    lookahead_stable_prefix_len: int | None = None
    lookahead_cache_seeded: bool | None = None
    pld_lookahead_router: str | None = None
    pld_lookahead_router_prob: float | None = None
    pld_lookahead_predicted_weak: bool | None = None
    pld_lookahead_predicted_weak_reason: str | None = None
    pld_lookahead_trigger: str | None = None
    pld_would_have_draft_len: int | None = None
    pld_lookahead_mode: str | None = None
    pld_lookahead_pld_used: bool | None = None
    pld_lookahead_skipped_pld: bool | None = None
    pld_lookahead_fallback_used: bool | None = None


@dataclass
class DecodeResult:
    output_token_ids: list[int]
    """All tokens INCLUDING the prompt prefix."""

    output_text: str
    """`tokenizer.decode(output_token_ids)`"""

    n_new_tokens: int
    """How many tokens were generated past the prompt."""

    steps: list[StepRecord] = field(default_factory=list)
    """One record per outer decode step."""

    wall_us_total: float = 0.0


# ---------------------------------------------------------------------------
# Vanilla AR (greedy)
# ---------------------------------------------------------------------------


def _argmax_int(logits_row: torch.Tensor) -> int:
    return int(logits_row.argmax(dim=-1).item())


@torch.no_grad()
def vanilla_ar(
    prompt_ids: torch.Tensor,  # shape [seq]
    target,
    max_new_tokens: int,
    eos_token_ids: list[int],
    method_name: str = "vanilla",
) -> DecodeResult:
    """Single-token greedy AR using the target model only.

    Cache convention: after each step, `cache` covers exactly `len(prefix)`
    tokens. The next step feeds only the most recently appended token.
    """
    device = next(target.parameters()).device
    prefix = prompt_ids.tolist()
    prompt_len = len(prompt_ids)

    cache = None
    next_input = (
        prompt_ids.unsqueeze(0).to(device)
        if prompt_ids.dim() == 1
        else prompt_ids.to(device)
    )

    steps: list[StepRecord] = []
    t_start = time.perf_counter_ns()

    step_idx = 0
    while len(prefix) - prompt_len < max_new_tokens:
        t0 = time.perf_counter_ns()
        out = target(next_input, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        next_tok = _argmax_int(out.logits[0, -1])
        prefix.append(next_tok)
        # Subsequent step feeds only the new token
        next_input = torch.tensor([[next_tok]], device=device)
        t1 = time.perf_counter_ns()

        steps.append(StepRecord(
            method=method_name,
            step=step_idx,
            k=1,
            n_accepted_drafts=0,
            n_emitted=1,
            rejected=False,
            node_type=None,
            deepest_type=None,
            wall_us=(t1 - t0) / 1000.0,
        ))
        step_idx += 1

        if next_tok in eos_token_ids:
            break

    t_end = time.perf_counter_ns()
    return DecodeResult(
        output_token_ids=prefix,
        output_text="",  # caller decodes
        n_new_tokens=len(prefix) - prompt_len,
        steps=steps,
        wall_us_total=(t_end - t_start) / 1000.0,
    )


# ---------------------------------------------------------------------------
# Speculative decoding (fixed-k or AST-gated)
# ---------------------------------------------------------------------------


@torch.no_grad()
def speculative_ar(
    prompt_ids: torch.Tensor,
    target,
    draft,
    max_new_tokens: int,
    eos_token_ids: list[int],
    method_name: str,
    k_resolver: Callable[[list[int]], tuple[int, str | None, str | None]],
    parse_callback: Callable[[list[int]], None] | None = None,
) -> DecodeResult:
    """Speculative greedy decoding.

    Args:
        k_resolver: callable that takes the current prefix (token ids) and
            returns (k, node_type, deepest_type). For fixed-k mode, returns
            a constant k and (None, None). For ASTS-Spec, queries the AST.
        parse_callback: optional hook called with current prefix tokens
            after each outer step (used by ASTS-Spec to update tree-sitter).
    """
    device = next(target.parameters()).device
    prefix = prompt_ids.tolist()

    target_cache = None
    target_cache_len = 0
    draft_cache = None
    draft_cache_len = 0
    steps: list[StepRecord] = []
    t_start = time.perf_counter_ns()

    # Anchoring invariant for spec modes: target_cache_len < len(prefix)
    # We'll maintain this lazily — only crop target_cache after the verify pass.

    step_idx = 0
    while len(prefix) < len(prompt_ids) + max_new_tokens:
        t_step_start = time.perf_counter_ns()

        # ---- Resolve k (and optionally the AST node type) for this step ----
        t_parse_0 = time.perf_counter_ns()
        k, node_type, deepest_type = k_resolver(prefix)
        t_parse_1 = time.perf_counter_ns()
        parse_us = (t_parse_1 - t_parse_0) / 1000.0
        if k < 1:
            k = 1

        # ---- Draft phase: generate k tokens ----
        t_draft_0 = time.perf_counter_ns()
        # Bring draft cache to len(prefix), then generate k tokens AR
        if draft_cache_len < len(prefix):
            draft_feed = torch.tensor([prefix[draft_cache_len:]], device=device)
        else:
            # Cache already covers everything — feed last token to get logits
            # (shouldn't happen in normal flow)
            draft_feed = torch.tensor([[prefix[-1]]], device=device)
        drafts: list[int] = []
        for _ in range(k):
            d_out = draft(draft_feed, past_key_values=draft_cache, use_cache=True)
            draft_cache = d_out.past_key_values
            next_d = _argmax_int(d_out.logits[0, -1])
            drafts.append(next_d)
            draft_feed = torch.tensor([[next_d]], device=device)
        # draft cache now covers len(prefix) + k positions
        draft_cache_len = len(prefix) + k
        t_draft_1 = time.perf_counter_ns()
        draft_us = (t_draft_1 - t_draft_0) / 1000.0

        # ---- Verify phase: target processes prefix_remainder + drafts ----
        # Anchoring invariant: ensure n_pre >= 1 by feeding at least 1 prefix
        # token to the target. If target_cache_len == len(prefix), crop back
        # by 1 so the last prefix token gets re-fed (it'll be in the verify
        # logits at position 0 → predicts drafts[0]).
        if target_cache_len >= len(prefix):
            crop_dynamic_cache(target_cache, len(prefix) - 1)
            target_cache_len = len(prefix) - 1

        n_pre = len(prefix) - target_cache_len  # >= 1 by construction
        target_input_list = prefix[target_cache_len:] + drafts
        target_input = torch.tensor([target_input_list], device=device)

        t_verify_0 = time.perf_counter_ns()
        t_out = target(target_input, past_key_values=target_cache, use_cache=True)
        target_cache = t_out.past_key_values
        # target cache now covers target_cache_len + len(target_input_list)
        target_cache_len = target_cache_len + len(target_input_list)
        t_verify_1 = time.perf_counter_ns()
        verify_us = (t_verify_1 - t_verify_0) / 1000.0

        # Greedy rejection sampling
        result: GreedyVerifyResult = greedy_verify(
            drafts=drafts,
            target_logits=t_out.logits,
            n_pre=n_pre,
        )

        # Update prefix — respect max_new_tokens AND first-EOS so spec
        # output matches vanilla's stopping behavior. (Without EOS truncation,
        # spec can emit tokens AFTER EOS when EOS is mid-batch.)
        old_prefix_len = len(prefix)
        eos_truncated = list(result.accepted_tokens)
        for i, tk in enumerate(eos_truncated):
            if tk in eos_token_ids:
                eos_truncated = eos_truncated[: i + 1]
                break
        budget = (len(prompt_ids) + max_new_tokens) - len(prefix)
        if budget < len(eos_truncated):
            accepted_tokens_capped = eos_truncated[:budget]
        else:
            accepted_tokens_capped = eos_truncated
        prefix.extend(accepted_tokens_capped)
        n_emitted = len(accepted_tokens_capped)

        # ---- Cache rollback ----
        # Target processed all of target_input (n_pre + k positions). Its cache
        # length = old target_cache_len + n_pre + k = old_prefix_len + k.
        # We want target cache at len(prefix) - 1 (anchoring invariant).
        # Note: len(prefix) = old_prefix_len + n_emitted
        # If rejected: n_emitted = n_accepted_drafts + 1 (correction)
        # If accepted: n_emitted = k + 1 (bonus)
        crop_dynamic_cache(target_cache, len(prefix) - 1)
        target_cache_len = len(prefix) - 1

        # Draft cache covers old_prefix_len + k. We want it at
        # len(prefix) - 1 = old_prefix_len + n_emitted - 1. The draft
        # never saw the bonus/correction (target's contribution at the end).
        # So crop draft cache to old_prefix_len + n_accepted_drafts.
        # Note: when rejected, n_accepted_drafts = n_emitted - 1; when accepted, n_accepted_drafts = k = n_emitted - 1. Always equal!
        target_draft_cache_len = old_prefix_len + result.n_accepted_drafts
        crop_dynamic_cache(draft_cache, target_draft_cache_len)
        draft_cache_len = target_draft_cache_len

        # Optional: notify parser callback so it can update its tree
        if parse_callback is not None:
            t_pcb_0 = time.perf_counter_ns()
            parse_callback(prefix)
            t_pcb_1 = time.perf_counter_ns()
            parse_us += (t_pcb_1 - t_pcb_0) / 1000.0

        t_step_end = time.perf_counter_ns()
        steps.append(StepRecord(
            method=method_name,
            step=step_idx,
            k=k,
            n_accepted_drafts=result.n_accepted_drafts,
            n_emitted=n_emitted,
            rejected=result.rejected,
            node_type=node_type,
            deepest_type=deepest_type,
            wall_us=(t_step_end - t_step_start) / 1000.0,
            draft_us=draft_us,
            verify_us=verify_us,
            parse_us=parse_us,
        ))
        step_idx += 1

        # Termination: any emitted EOS
        if any(t in eos_token_ids for t in result.accepted_tokens):
            break

    t_end = time.perf_counter_ns()
    return DecodeResult(
        output_token_ids=prefix,
        output_text="",
        n_new_tokens=len(prefix) - len(prompt_ids),
        steps=steps,
        wall_us_total=(t_end - t_start) / 1000.0,
    )


# ---------------------------------------------------------------------------
# Convenience wrappers for the three modes
# ---------------------------------------------------------------------------


def fixed_spec_ar(
    prompt_ids: torch.Tensor,
    target,
    draft,
    max_new_tokens: int,
    eos_token_ids: list[int],
    k: int = 8,
) -> DecodeResult:
    """Fixed-k speculative AR (no AST gating)."""
    def _resolver(_prefix):
        return k, None, None

    return speculative_ar(
        prompt_ids=prompt_ids,
        target=target,
        draft=draft,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        method_name=f"fixed_k{k}",
        k_resolver=_resolver,
    )


def asts_spec_ar(
    prompt_ids: torch.Tensor,
    target,
    draft,
    max_new_tokens: int,
    eos_token_ids: list[int],
    tokenizer,
    ast_policy,
) -> DecodeResult:
    """ASTS-Spec: variable-length speculation gated by tree-sitter AST node type."""
    def _decode_to_bytes(prefix_ids: list[int]) -> bytes:
        return tokenizer.decode(prefix_ids, skip_special_tokens=False).encode("utf-8")

    def _resolver(prefix_ids: list[int]):
        ast_policy.update(_decode_to_bytes(prefix_ids))
        ctx = ast_policy.context_at_cursor()
        return ctx.k, ctx.node_type, ctx.deepest_type

    def _parse_callback(prefix_ids: list[int]):
        # Already updated in resolver before this step; no-op here.
        # Future optimization: cache the bytes and skip re-decode.
        pass

    return speculative_ar(
        prompt_ids=prompt_ids,
        target=target,
        draft=draft,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        method_name="asts_spec",
        k_resolver=_resolver,
        parse_callback=_parse_callback,
    )
