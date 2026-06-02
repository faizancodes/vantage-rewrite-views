"""BlazEdit-style assisted decoding baselines inside the VANTAGE harness.

The public BlazEdit system combines assisted decoding with prompt-lookup
drafting inside the assistant model.  This module implements comparable
baselines under the same greedy verifier used by the rest of this repo:

* PLD-only: exact prompt/local n-gram lookup verified by the target.
* Assisted static/dynamic: assistant-model drafts verified by the target.
* Two-layer: the assistant's own drafting is accelerated by prompt lookup
  micro-runs before one target verification call.

The assisted methods intentionally do not use VANTAGE's guaranteed target
root convention.  The assistant proposes the first target candidate and the
target may reject it, matching ordinary speculative decoding accounting.
"""

from __future__ import annotations

import itertools
import copy
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

from .decoder import DecodeResult, StepRecord, crop_dynamic_cache
from .mtp_heads import MTPHeadConfig, PLDMTPHeads
from .pld_reranker import (
    DEFAULT_WEIGHTS_PATH,
    FEATURE_NAMES,
    PLDRerankCandidate,
    PLDRerankContext,
    PLDRerankerWeights,
    apply_score_margin_gate,
    compute_left_extension,
    load_reranker_weights,
    select_candidate_by_policy,
)
from .rejection import GreedyVerifyResult, greedy_verify
from .code_proposers import (
    ProposalTreeNode,
    RewriteMapSet,
    _apply_word_map,
    _coerce_rewrite_pairs,
    _derived_field_views,
    _extract_reference_blocks,
    _rewrite_pairs,
    _token_offsets,
    build_candidate_prefix_tree,
    encode_no_special,
)


BlazEditMode = Literal[
    "pld",
    "pld_plus_mtp_heads",
    "pld_queued_mtp_heads",
    "weak_router_capped_pld",
    "assisted_static",
    "assisted_dynamic",
    "two_layer",
    "delta_cache_pld",
    "fuzzy_resync_pld",
    "rerank_exact_pld",
    "lookahead",
    "pld_gated_lookahead",
]


@dataclass(frozen=True)
class BlazEditConfig:
    mode: BlazEditMode
    assistant_model_name: str = "Qwen/Qwen2.5-Coder-0.5B"
    micro_draft_tokens: int = 40
    max_num_run: int = 1
    max_matching_ngram_size: int = 10
    min_matching_ngram_size: int = 1
    assistant_confidence_threshold: float | None = None
    use_staged_verification: bool = False
    staged_first_tokens: int = 16
    staged_second_tokens: int = 32
    delta_context_tokens: int = 4
    delta_lru_size: int = 64
    delta_max_patches: int = 1
    delta_patch_window: int = 64
    fuzzy_weak_draft_len: int = 8
    fuzzy_max_draft_tokens: int = 32
    fuzzy_require_unique: bool = True
    pld_opportunity_trace: bool = False
    pld_rerank_top_k: int = 4
    pld_rerank_weights_path: str | None = None
    pld_rerank_only_ambiguous: bool = True
    pld_rerank_fallback: str = "baseline"
    pld_rerank_debug_trace: bool = False
    pld_rerank_margin: float = 0.0
    pld_rerank_margin_gate: bool = False
    pld_rerank_always_include_baseline: bool = True
    pld_rerank_enable_left_extension: bool = False
    pld_rerank_left_extension_max: int = 128
    pld_rerank_policy: str = "learned"
    pld_rerank_fixed_rank: int = 0
    mtp_heads_checkpoint: str = "/data/pld_mtp/postpld_linear_k4_n917_v1/postpld_mtp_heads_k4_linear.pt"
    mtp_num_heads: int = 4
    mtp_trigger_accepted_len: int = 4
    mtp_position: str = "post_pld"
    mtp_disable: bool = False
    mtp_queue_enabled: bool = True
    mtp_use_queued_only_on_weak_pld: bool = True
    mtp_disable_extra_verify: bool = False
    weak_pld_router_path: str = "/tmp/pld_mtp/weak_router/router.pkl"
    weak_pld_router_threshold: float = 0.5
    weak_pld_cap_tokens: int = 4
    lookahead_window: int = 8
    lookahead_ngram: int = 4
    lookahead_iters: int = 4
    lookahead_max_draft: int = 16
    lookahead_stable_prefix: bool = True
    lookahead_trajectory_cache: bool = True
    lookahead_one_forward: bool = False
    pld_lookahead_router: str = "rule"
    pld_lookahead_router_path: str = "/tmp/pld_mtp/weak_router/router.pkl"
    pld_lookahead_router_threshold: float = 0.3
    pld_lookahead_weak_threshold: int = 4
    pld_lookahead_trigger: str = "router_weak"
    pld_lookahead_mode: str = "replace_weak_pld"
    pld_lookahead_fallback: str = "pld"
    pld_lookahead_min_candidate_len: int = 1
    target_prefill_chunk_size: int = 0


@dataclass
class _DraftStats:
    drafts: list[int]
    assistant_cache: object | None
    assistant_cache_len: int
    assistant_us: float = 0.0
    assistant_prefill_us: float = 0.0
    assistant_pld_us: float = 0.0
    assistant_verify_us: float = 0.0
    micro_runs: int = 0
    pld_proposed: int = 0
    pld_accepted: int = 0
    pld_max_match_len: int = 0
    catchup_tokens: int = 0


@dataclass
class _DeltaEntry:
    new_token: int
    support: int = 0
    conflicts: int = 0
    uses: int = 0
    accepted_uses: int = 0


@dataclass
class _QueuedMTPDraft:
    position: int
    draft_tokens: list[int]
    source_step: int


@dataclass
class _PLDVariantStepInfo:
    exact_hit: bool = False
    triggered: bool = False
    overhead_us: float = 0.0
    token01_rejection: bool = False
    delta_cache_size: int | None = None
    delta_patch_count: int | None = None
    delta_patch_accepted: bool | None = None
    delta_patch_accept_tail: int | None = None
    fuzzy_candidate_count: int | None = None
    fuzzy_edit_distance: int | None = None
    fuzzy_match_len: int | None = None
    candidate_accepted_len: int | None = None
    rerank_candidate_count: int | None = None
    rerank_selected_rank: int | None = None
    rerank_fallback: bool | None = None
    rerank_baseline_rank: int | None = None
    rerank_selected_is_baseline: bool | None = None
    rerank_selected_score: float | None = None
    rerank_baseline_score: float | None = None
    rerank_score_margin: float | None = None
    rerank_baseline_score_missing: bool = False
    rerank_candidate_positions: list[int] | None = None
    rerank_candidate_source_kinds: list[str] | None = None
    rerank_debug_features: list[dict[str, Any]] | None = None


@dataclass
class _PLDOpportunityTrace:
    exact_hit: bool
    candidate_matches: int
    source_position: int
    draft_len: int
    generated_suffix_16_text: str | None = None
    draft_prefix_32_text: str | None = None
    source_snippet_text: str | None = None


@dataclass
class _LookaheadState:
    guess: list[int] = field(default_factory=list)


@dataclass
class _LookaheadDraftStats:
    drafts: list[int]
    candidate_len: int
    stable_prefix_len: int
    forward_calls: int
    lookahead_us: float
    forward_us: float = 0.0
    candidate_build_us: float = 0.0
    cache_seeded: bool = False


@dataclass(frozen=True)
class VantageMVConfig:
    """PLD-first multi-view lookup policy.

    The decoder is intentionally BlazEdit-derived: exact PLD is queried first
    and uses the same verifier/cache path as ``blazedit_pld``. Transformed
    views are optional draft sources that can override exact PLD only when
    exact PLD is weak and the transformed candidate is both long and close to
    an explicit rewrite frontier.
    """

    max_draft_tokens: int = 128
    max_matching_ngram_size: int = 10
    transformed_min_matching_ngram_size: int = 8
    exact_strong_min_len: int = 64
    exact_strong_min_match: int = 1
    trans_len_margin: int = 16
    exact_match1_draft_cap: int = 0
    exact_match2_7_draft_cap: int = 0
    frontier_window: int = 32
    no_frontier_probe_exact_len: int = 0
    max_views: int = 8
    cold_trans_max_draft: int = 32
    medium_trans_max_draft: int = 64
    low_accept_disable_attempts: int = 2
    low_accept_disable_rate: float = 0.08
    branch_common_prefix_min: int = 8
    vocab_margin_bypass: bool = True
    allow_long_exact_frontier_probe: bool = True
    frontier_probe_min_draft_tokens: int = 16
    generated_reindex_interval: int = 0
    cursor_min_accept: int = 8
    cursor_max_draft_tokens: int = 256
    use_rewrite_fst: bool = False
    use_map_quality_gate: bool = False
    use_frontier_branch: bool = False
    use_pair_priors: bool = False
    use_stateful_cursor: bool = False
    use_hunk_alignment: bool = False
    use_pld_rejection_rescue: bool = False
    rescue_window_steps: int = 2
    use_segment_patch: bool = False
    patch_segment_tokens: int = 64
    use_staged_verification: bool = False
    staged_first_tokens: int = 16
    staged_second_tokens: int = 32
    use_packed_branch: bool = False
    use_edit_neural_drafter: bool = False
    edit_draft_tokens: int = 64
    edit_draft_min_exact_len: int = 16


@dataclass
class _MVView:
    view_id: str
    tokens: list[int]
    rewrite_map: dict[str, str]
    source_label: str
    map_source: str
    frontiers: list[int]
    index: dict[int, dict[tuple[int, ...], list[int]]]
    transducer: bool = False
    query_map: dict[str, str] = field(default_factory=dict)
    reference_text: str | None = None
    value_tokens: list[int] = field(default_factory=list)
    value_index: dict[int, dict[tuple[int, ...], list[int]]] = field(default_factory=dict)
    value_frontiers: list[int] = field(default_factory=list)
    value_pair_spans: list[tuple[int, int, str]] = field(default_factory=list)
    source_to_value_start: list[int] = field(default_factory=list)
    source_to_value_end: list[int] = field(default_factory=list)
    pair_spans: list[tuple[int, int, str]] = field(default_factory=list)
    query_replacements: list[tuple[tuple[int, ...], tuple[int, ...], str]] = field(
        default_factory=list
    )


@dataclass
class _MVViewStats:
    attempts: int = 0
    accepted: int = 0
    proposed: int = 0
    disabled: bool = False
    adopted: bool = False
    blacklist: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class _MVPairStats:
    attempts: int = 0
    accepted: int = 0
    proposed: int = 0
    disabled: bool = False
    adopted: bool = False
    zero_accepts: int = 0


@dataclass
class _MVTextPrecheckView:
    view_id: str
    rewrite_map: dict[str, str]
    source_label: str
    text: str


@dataclass
class _MVLazyPlan:
    reference: str
    rewrite_map: dict[str, str]
    map_source: str
    old_terms: list[str]
    new_terms: list[str]
    text_views: list[_MVTextPrecheckView] | None = None


@dataclass
class _MVCandidate:
    tokens: list[int]
    match_len: int
    source_start: int
    follow_start: int
    view: _MVView
    frontier_distance: int | None
    crosses_frontier: bool
    score: float
    pair_keys: tuple[str, ...] = ()
    branch_eligible: bool = False
    frontier_probe_override: bool = False
    from_cursor: bool = False


@dataclass
class _MVCursorState:
    view_id: str | None = None
    pos: int = 0
    confidence: float = 0.0
    active: bool = False

    def reset(self) -> None:
        self.view_id = None
        self.pos = 0
        self.confidence = 0.0
        self.active = False


_MTP_RUNTIME_HEADS_CACHE: dict[tuple[str, str, str, int], PLDMTPHeads] = {}
_WEAK_PLD_ROUTER_CACHE: dict[str, object] = {}


def _load_weak_pld_router_for_runtime(path: str) -> object:
    cached = _WEAK_PLD_ROUTER_CACHE.get(path)
    if cached is not None:
        return cached
    import pickle

    with open(path, "rb") as f:
        payload = pickle.load(f)
    model = payload.get("model") if isinstance(payload, dict) else payload
    if model is None:
        raise ValueError(f"weak PLD router checkpoint has no model: {path}")
    _WEAK_PLD_ROUTER_CACHE[path] = model
    return model


def _load_optional_weak_pld_router_for_runtime(path: str) -> object | None:
    if not path or not Path(path).exists():
        return None
    try:
        return _load_weak_pld_router_for_runtime(path)
    except Exception:
        return None


def _lookahead_seed_guess(
    *,
    prefix: list[int],
    state: _LookaheadState | None,
    window: int,
    use_cache: bool,
) -> tuple[list[int], bool]:
    if window <= 0:
        return [], False
    if use_cache and state is not None and state.guess:
        seed = list(state.guess[:window])
        if len(seed) < window:
            seed.extend([seed[-1] if seed else prefix[-1]] * (window - len(seed)))
        return seed, True
    dummy = prefix[-1] if prefix else 0
    return [dummy for _ in range(window)], False


@torch.no_grad()
def _lookahead_jacobi_draft(
    *,
    prefix: list[int],
    target,
    target_cache,
    target_cache_len: int,
    config: BlazEditConfig,
    state: _LookaheadState | None,
    max_tokens: int | None = None,
) -> _LookaheadDraftStats:
    """Build a conservative training-free lookahead candidate.

    The candidate is only a proposal: the normal target verifier still decides
    every emitted token.  The implementation uses Jacobi-style parallel updates
    over a fixed guess window.  To avoid corrupting the main decode cache, each
    lookahead iteration runs against a copied prefix cache and discards it.
    """
    t0 = time.perf_counter_ns()
    if not prefix:
        return _LookaheadDraftStats([], 0, 0, 0, 0.0, 0.0, 0.0, False)
    window = max(0, int(config.lookahead_window))
    max_draft = max(0, int(config.lookahead_max_draft))
    if max_tokens is not None:
        max_draft = min(max_draft, max(0, int(max_tokens)))
    window = min(window, max_draft)
    if window <= 0:
        return _LookaheadDraftStats([], 0, 0, 0, 0.0, 0.0, 0.0, False)

    device = next(target.parameters()).device
    old_prefix_len = len(prefix)
    base_cache_len = int(target_cache_len)
    if base_cache_len >= old_prefix_len:
        base_cache_len = max(0, old_prefix_len - 1)
    n_pre = old_prefix_len - base_cache_len
    prefix_feed = list(prefix[base_cache_len:])
    if not prefix_feed:
        prefix_feed = [prefix[-1]]
        base_cache_len = old_prefix_len - 1
        n_pre = 1

    guess, cache_seeded = _lookahead_seed_guess(
        prefix=prefix,
        state=state,
        window=window,
        use_cache=bool(config.lookahead_trajectory_cache),
    )
    trajectory: list[list[int]] = [list(guess)]
    forward_calls = 0
    iters = 1 if bool(config.lookahead_one_forward) else max(1, int(config.lookahead_iters))
    forward_us = 0.0

    for _ in range(iters):
        iter_cache = copy.deepcopy(target_cache)
        if iter_cache is not None:
            crop_dynamic_cache(iter_cache, base_cache_len)
        input_ids = torch.tensor(
            [prefix_feed + list(guess)],
            device=device,
            dtype=torch.long,
        )
        t_forward = time.perf_counter_ns()
        out = target(input_ids, past_key_values=iter_cache, use_cache=False)
        forward_us += (time.perf_counter_ns() - t_forward) / 1000.0
        forward_calls += 1
        logits = out.logits[0] if out.logits.dim() == 3 else out.logits
        next_guess: list[int] = []
        for j in range(window):
            logit_idx = n_pre - 1 + j
            if logit_idx < 0 or logit_idx >= logits.shape[0]:
                break
            next_guess.append(_argmax_int(logits[logit_idx]))
        if not next_guess:
            break
        if len(next_guess) < window:
            next_guess.extend([next_guess[-1]] * (window - len(next_guess)))
        guess = next_guess[:window]
        trajectory.append(list(guess))

    stable_prefix_len = 0
    if len(trajectory) >= 2:
        prev = trajectory[-2]
        curr = trajectory[-1]
        for a, b in zip(prev, curr):
            if a != b:
                break
            stable_prefix_len += 1
    final_guess = trajectory[-1] if trajectory else []
    if config.lookahead_stable_prefix and stable_prefix_len > 0:
        candidate = final_guess[:stable_prefix_len]
    else:
        candidate = final_guess[: min(len(final_guess), max(1, int(config.lookahead_ngram)))]
    candidate = candidate[:max_draft]
    if state is not None and final_guess:
        shifted = list(final_guess[1:]) + [final_guess[-1]]
        state.guess = shifted[:window]
    total_us = (time.perf_counter_ns() - t0) / 1000.0
    return _LookaheadDraftStats(
        drafts=[int(t) for t in candidate],
        candidate_len=len(candidate),
        stable_prefix_len=stable_prefix_len,
        forward_calls=forward_calls,
        lookahead_us=total_us,
        forward_us=forward_us,
        candidate_build_us=max(0.0, total_us - forward_us),
        cache_seeded=cache_seeded,
    )


def _pld_lookahead_rule_weak_reason(
    *,
    drafts: list[int],
    match_len: int,
    candidate_count: int,
    threshold: int,
) -> str | None:
    if not drafts:
        return "pld_miss"
    if len(drafts) <= threshold:
        return "short_draft"
    if match_len <= 0:
        return "nonpositive_match"
    # Ambiguous short hits were a consistent PLD failure mode in prior traces.
    if candidate_count > 1 and match_len <= 4 and len(drafts) <= max(8, threshold * 2):
        return "short_ambiguous_hit"
    return None


def _pld_lookahead_rule_predict_weak(
    *,
    drafts: list[int],
    match_len: int,
    candidate_count: int,
    threshold: int,
) -> bool:
    return (
        _pld_lookahead_rule_weak_reason(
            drafts=drafts,
            match_len=match_len,
            candidate_count=candidate_count,
            threshold=threshold,
        )
        is not None
    )


def _weak_pld_router_step_features(
    *,
    prefix_len: int,
    prompt_len: int,
    step_idx: int,
    drafts: list[int],
    match_len: int,
    source_start: int,
    follow_start: int,
    proposal_us: float,
    history: dict[str, Any],
) -> dict[str, Any]:
    from scripts.train_weak_pld_router import extract_feature_dict

    row = {
        "step": int(step_idx),
        "k": int(len(drafts)),
        "proposal_tokens": int(len(drafts)),
        "proposal_match_len": int(match_len),
        "proposal_query_len": int(match_len),
        "proposal_source_start_token": int(source_start),
        "proposal_follow_start_token": int(follow_start),
        "proposal_us": float(proposal_us),
        "prompt_len": int(prompt_len),
        "proposal_kind": "blazedit_pld",
        "proposal_pool": "local",
        "proposal_source_region": None,
        "proposal_root_included": False,
        "blazedit_micro_draft_tokens": int(len(drafts)),
        "blazedit_max_num_run": 1,
        "blazedit_pld_proposed": int(len(drafts)),
    }
    return extract_feature_dict(
        row,
        generated_start=max(0, int(prefix_len) - int(prompt_len)),
        history=history,
        step_index=int(step_idx),
    )


def _weak_pld_router_update_history(
    *,
    history: dict[str, Any],
    threshold: int,
    accepted_len: int,
    emitted: int,
    rejected: bool,
    draft_len: int,
    source_start: int,
) -> None:
    from scripts.train_weak_pld_router import _update_history

    _update_history(
        history,
        {
            "n_accepted_drafts": int(accepted_len),
            "n_emitted": int(emitted),
            "rejected": bool(rejected),
            "proposal_kind": "blazedit_pld",
            "proposal_tokens": int(draft_len),
            "k": int(draft_len),
            "proposal_source_start_token": int(source_start),
        },
        threshold=int(threshold),
    )


def _load_mtp_heads_for_runtime(
    *,
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype,
    expected_num_heads: int,
) -> PLDMTPHeads:
    key = (checkpoint_path, str(device), str(dtype), int(expected_num_heads))
    cached = _MTP_RUNTIME_HEADS_CACHE.get(key)
    if cached is not None:
        return cached
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = MTPHeadConfig(**ckpt["config"])
    if int(config.num_heads) < int(expected_num_heads):
        raise ValueError(
            f"MTP checkpoint has {config.num_heads} heads, requested {expected_num_heads}"
        )
    output_weight = ckpt.get("output_weight")
    if output_weight is not None:
        output_weight = output_weight.to(device=device, dtype=dtype)
    model = PLDMTPHeads(config, output_weight=output_weight)
    model.load_state_dict(ckpt["model_state"])
    model.to(device=device, dtype=dtype)
    model.eval()
    model.prepare_for_inference()
    _MTP_RUNTIME_HEADS_CACHE[key] = model
    return model


def parse_blazedit_method(
    method: str,
    *,
    assistant_model_name: str = "Qwen/Qwen2.5-Coder-0.5B",
    confidence_threshold: float | None = None,
    default_ngram_size: int = 10,
) -> BlazEditConfig:
    """Parse paper/eval method names into a concrete BlazEdit config."""
    staged_pld = re.fullmatch(
        r"(?:blazedit_pld_staged|vantage_staged_pld)_v(?P<v1>\d+)_(?P<v2>\d+)_w(?P<w>\d+)_n(?P<n>\d+)",
        method,
    )
    if staged_pld:
        return BlazEditConfig(
            mode="pld",
            assistant_model_name=assistant_model_name,
            micro_draft_tokens=int(staged_pld.group("w")),
            max_num_run=1,
            max_matching_ngram_size=int(staged_pld.group("n")),
            use_staged_verification=True,
            staged_first_tokens=int(staged_pld.group("v1")),
            staged_second_tokens=int(staged_pld.group("v2")),
        )

    pld_with_min = re.fullmatch(
        r"(?:blazedit_pld|vantage_force_pld)_m(?P<m>\d+)_w(?P<w>\d+)_n(?P<n>\d+)",
        method,
    )
    if pld_with_min:
        return BlazEditConfig(
            mode="pld",
            assistant_model_name=assistant_model_name,
            micro_draft_tokens=int(pld_with_min.group("w")),
            max_num_run=1,
            max_matching_ngram_size=int(pld_with_min.group("n")),
            min_matching_ngram_size=int(pld_with_min.group("m")),
        )

    pld = re.fullmatch(r"(?:blazedit_pld|vantage_force_pld)_w(\d+)_n(\d+)", method)
    if pld:
        return BlazEditConfig(
            mode="pld",
            assistant_model_name=assistant_model_name,
            micro_draft_tokens=int(pld.group(1)),
            max_num_run=1,
            max_matching_ngram_size=int(pld.group(2)),
            min_matching_ngram_size=1,
        )

    mtp = re.fullmatch(r"pld_plus_mtp_heads(?:_w(?P<w>\d+)_n(?P<n>\d+))?", method)
    if mtp:
        return BlazEditConfig(
            mode="pld_plus_mtp_heads",
            assistant_model_name=assistant_model_name,
            micro_draft_tokens=int(mtp.group("w") or 128),
            max_num_run=1,
            max_matching_ngram_size=int(mtp.group("n") or default_ngram_size),
        )

    residual = re.fullmatch(
        r"vantage_residual_k(?P<k>[124])(?:_t(?P<t>\d+))?(?:_w(?P<w>\d+)_n(?P<n>\d+))?",
        method,
    )
    if residual:
        return BlazEditConfig(
            mode="pld_plus_mtp_heads",
            assistant_model_name=assistant_model_name,
            micro_draft_tokens=int(residual.group("w") or 128),
            max_num_run=1,
            max_matching_ngram_size=int(residual.group("n") or default_ngram_size),
            mtp_num_heads=int(residual.group("k")),
            mtp_trigger_accepted_len=int(residual.group("t") or 4),
        )

    residual_router = re.fullmatch(
        r"vantage_residual_router_k(?P<k>[124])(?:_w(?P<w>\d+)_n(?P<n>\d+))?",
        method,
    )
    if residual_router:
        return BlazEditConfig(
            mode="pld_plus_mtp_heads",
            assistant_model_name=assistant_model_name,
            micro_draft_tokens=int(residual_router.group("w") or 128),
            max_num_run=1,
            max_matching_ngram_size=int(residual_router.group("n") or default_ngram_size),
            mtp_num_heads=int(residual_router.group("k")),
            mtp_trigger_accepted_len=4,
        )

    queued_mtp = re.fullmatch(r"pld_queued_mtp_heads(?:_w(?P<w>\d+)_n(?P<n>\d+))?", method)
    if queued_mtp:
        return BlazEditConfig(
            mode="pld_queued_mtp_heads",
            assistant_model_name=assistant_model_name,
            micro_draft_tokens=int(queued_mtp.group("w") or 128),
            max_num_run=1,
            max_matching_ngram_size=int(queued_mtp.group("n") or default_ngram_size),
            mtp_queue_enabled=True,
            mtp_use_queued_only_on_weak_pld=True,
            mtp_disable_extra_verify=True,
        )

    lookahead = re.fullmatch(
        r"lookahead_w(?P<w>\d+)_n(?P<n>\d+)_i(?P<i>\d+)",
        method,
    )
    if lookahead:
        window = int(lookahead.group("w"))
        return BlazEditConfig(
            mode="lookahead",
            assistant_model_name=assistant_model_name,
            micro_draft_tokens=window,
            max_num_run=1,
            max_matching_ngram_size=default_ngram_size,
            lookahead_window=window,
            lookahead_ngram=int(lookahead.group("n")),
            lookahead_iters=int(lookahead.group("i")),
            lookahead_max_draft=window,
        )

    gated_lookahead_shape = re.fullmatch(
        r"pld_gated_lookahead_w(?P<w>\d+)_n(?P<n>\d+)_i(?P<i>\d+)(?:_d(?P<d>\d+))?",
        method,
    )
    if gated_lookahead_shape:
        window = int(gated_lookahead_shape.group("w"))
        iters = int(gated_lookahead_shape.group("i"))
        return BlazEditConfig(
            mode="pld_gated_lookahead",
            assistant_model_name=assistant_model_name,
            micro_draft_tokens=128,
            max_num_run=1,
            max_matching_ngram_size=10,
            lookahead_window=window,
            lookahead_ngram=int(gated_lookahead_shape.group("n")),
            lookahead_iters=iters,
            lookahead_max_draft=int(gated_lookahead_shape.group("d") or window),
            lookahead_one_forward=(iters == 1),
        )

    gated_lookahead = re.fullmatch(
        r"pld_gated_lookahead_w(?P<w>\d+)_n(?P<n>\d+)",
        method,
    )
    if gated_lookahead:
        return BlazEditConfig(
            mode="pld_gated_lookahead",
            assistant_model_name=assistant_model_name,
            micro_draft_tokens=int(gated_lookahead.group("w")),
            max_num_run=1,
            max_matching_ngram_size=int(gated_lookahead.group("n")),
        )

    weak_cap = re.fullmatch(
        r"weak_router_capped_pld(?:_t(?P<t>\d+))?(?:_cap(?P<cap>\d+))?_w(?P<w>\d+)_n(?P<n>\d+)",
        method,
    )
    if weak_cap:
        threshold_text = weak_cap.group("t")
        cap_text = weak_cap.group("cap")
        return BlazEditConfig(
            mode="weak_router_capped_pld",
            assistant_model_name=assistant_model_name,
            micro_draft_tokens=int(weak_cap.group("w")),
            max_num_run=1,
            max_matching_ngram_size=int(weak_cap.group("n")),
            weak_pld_router_threshold=(
                float(int(threshold_text) / 100.0) if threshold_text else 0.5
            ),
            weak_pld_cap_tokens=int(cap_text) if cap_text is not None else 4,
        )

    rerank = re.fullmatch(r"rerank_exact_pld_k(?P<k>\d+)_w(?P<w>\d+)_n(?P<n>\d+)", method)
    if rerank:
        return BlazEditConfig(
            mode="rerank_exact_pld",
            assistant_model_name=assistant_model_name,
            micro_draft_tokens=int(rerank.group("w")),
            max_num_run=1,
            max_matching_ngram_size=int(rerank.group("n")),
            pld_rerank_top_k=int(rerank.group("k")),
        )

    delta = re.fullmatch(
        r"delta_cache_pld(?:_p(?P<patch>\d+))?(?:_c(?P<context>\d+))?(?:_lru(?P<lru>\d+))?(?:_pw(?P<window>\d+))?_w(?P<w>\d+)_n(?P<n>\d+)",
        method,
    )
    if delta:
        return BlazEditConfig(
            mode="delta_cache_pld",
            assistant_model_name=assistant_model_name,
            micro_draft_tokens=int(delta.group("w")),
            max_num_run=1,
            max_matching_ngram_size=int(delta.group("n")),
            delta_max_patches=int(delta.group("patch") or 1),
            delta_context_tokens=int(delta.group("context") or 4),
            delta_lru_size=int(delta.group("lru") or 64),
            delta_patch_window=int(delta.group("window") or 64),
        )

    fuzzy = re.fullmatch(
        r"fuzzy_resync_pld(?:_fd(?P<fd>\d+))?(?:_weak(?P<weak>\d+))?_w(?P<w>\d+)_n(?P<n>\d+)",
        method,
    )
    if fuzzy:
        return BlazEditConfig(
            mode="fuzzy_resync_pld",
            assistant_model_name=assistant_model_name,
            micro_draft_tokens=int(fuzzy.group("w")),
            max_num_run=1,
            max_matching_ngram_size=int(fuzzy.group("n")),
            fuzzy_max_draft_tokens=int(fuzzy.group("fd") or 32),
            fuzzy_weak_draft_len=int(fuzzy.group("weak") or 8),
        )

    assisted = re.fullmatch(r"blazedit_assisted_(static|dynamic)_w(\d+)", method)
    if assisted:
        mode: BlazEditMode = (
            "assisted_static" if assisted.group(1) == "static" else "assisted_dynamic"
        )
        return BlazEditConfig(
            mode=mode,
            assistant_model_name=assistant_model_name,
            micro_draft_tokens=int(assisted.group(2)),
            max_num_run=1,
            max_matching_ngram_size=default_ngram_size,
            assistant_confidence_threshold=(
                (0.4 if confidence_threshold is None else confidence_threshold)
                if mode == "assisted_dynamic"
                else None
            ),
        )

    two_layer = re.fullmatch(r"blazedit_two_layer_m(\d+)_r(\d+)_n(\d+)", method)
    if two_layer:
        return BlazEditConfig(
            mode="two_layer",
            assistant_model_name=assistant_model_name,
            micro_draft_tokens=int(two_layer.group(1)),
            max_num_run=int(two_layer.group(2)),
            max_matching_ngram_size=int(two_layer.group(3)),
        )

    raise ValueError(f"unknown BlazEdit method: {method}")


def parse_vantage_mv_method(method: str) -> VantageMVConfig:
    """Parse PLD-first VANTAGE-MV method names.

    Supported forms:
      * ``vantage_mv_pld_w128_n10`` (defaults: strong=64, margin=16, min=8)
      * ``vantage_mv_pld_s64_m16_w128_n10``
      * ``vantage_mv_pld_s64_m16_t8_w128_n10``
      * ``vantage_mv_pld_s96_x1_m16_t8_w128_n10`` (stable real-commit MV)
      * ``vantage_mv_pld_s64_m16_f4_t8_w128_n10``
      * ``vantage_mv_pld_s64_x8_c116_c764_m16_t8_w128_n10``
      * ``vantage_mv_pld_fst_q_pair_tree_cursor_hunk_s64_m8_f8_t8_g64_w128_n10``
        where feature flags enable rewrite-FST, map-quality gating,
        per-pair priors, frontier-local tree probing, stateful transformed
        cursors, hunk/line alignment, and generated-prefix reindexing.
      * feature flags ``rescue`` and ``patch`` enable PLD-rejection rescue
        and segment-level transformed patch chunks.
      * feature flags ``stage``, ``pbranch``, and ``edraft`` enable staged
        target verification, packed PLD-vs-transformed branch verification,
        and edit-frontier neural drafting.
    """
    prefix = "vantage_mv_pld_"
    if not method.startswith(prefix):
        raise ValueError(f"unknown VANTAGE-MV method: {method}")

    values: dict[str, int] = {
        "strong": 64,
        "exact_match": 1,
        "margin": 16,
        "cap1": 0,
        "cap7": 0,
        "frontier": 0,
        "tmin": 8,
        "gen": 0,
    }
    features = {
        "fst": False,
        "q": False,
        "branch": False,
        "tree": False,
        "pair": False,
        "cursor": False,
        "hunk": False,
        "rescue": False,
        "patch": False,
        "stage": False,
        "pbranch": False,
        "edraft": False,
    }
    w: int | None = None
    n: int | None = None
    for part in method[len(prefix) :].split("_"):
        if part == "all":
            for key in (
                "fst",
                "q",
                "branch",
                "tree",
                "pair",
                "cursor",
                "hunk",
                "rescue",
                "patch",
            ):
                features[key] = True
            continue
        if part in features:
            features[part] = True
            continue
        if re.fullmatch(r"s\d+", part):
            values["strong"] = int(part[1:])
        elif re.fullmatch(r"x\d+", part):
            values["exact_match"] = int(part[1:])
        elif re.fullmatch(r"m\d+", part):
            values["margin"] = int(part[1:])
        elif re.fullmatch(r"c1\d+", part):
            values["cap1"] = int(part[2:])
        elif re.fullmatch(r"c7\d+", part):
            values["cap7"] = int(part[2:])
        elif re.fullmatch(r"f\d+", part):
            values["frontier"] = int(part[1:])
        elif re.fullmatch(r"t\d+", part):
            values["tmin"] = int(part[1:])
        elif re.fullmatch(r"g\d+", part):
            values["gen"] = int(part[1:])
        elif re.fullmatch(r"w\d+", part):
            w = int(part[1:])
        elif re.fullmatch(r"n\d+", part):
            n = int(part[1:])
        else:
            raise ValueError(f"unknown VANTAGE-MV method segment {part!r}: {method}")
    if w is None or n is None:
        raise ValueError(f"unknown VANTAGE-MV method: {method}")
    return VantageMVConfig(
        max_draft_tokens=w,
        max_matching_ngram_size=n,
        exact_strong_min_len=values["strong"],
        exact_strong_min_match=values["exact_match"],
        trans_len_margin=values["margin"],
        exact_match1_draft_cap=values["cap1"],
        exact_match2_7_draft_cap=values["cap7"],
        transformed_min_matching_ngram_size=values["tmin"],
        no_frontier_probe_exact_len=values["frontier"],
        generated_reindex_interval=values["gen"],
        use_rewrite_fst=features["fst"],
        use_map_quality_gate=features["q"],
        use_frontier_branch=features["branch"] or features["tree"],
        use_pair_priors=features["pair"],
        use_stateful_cursor=features["cursor"],
        use_hunk_alignment=features["hunk"],
        use_pld_rejection_rescue=features["rescue"],
        use_segment_patch=features["patch"],
        use_staged_verification=features["stage"],
        use_packed_branch=features["pbranch"],
        use_edit_neural_drafter=features["edraft"],
    )


def is_vantage_mv_method(method: str) -> bool:
    try:
        parse_vantage_mv_method(method)
        return True
    except ValueError:
        return False


def is_blazedit_method(method: str) -> bool:
    return (
        re.fullmatch(r"blazedit_pld_w\d+_n\d+", method) is not None
        or re.fullmatch(r"blazedit_pld_m\d+_w\d+_n\d+", method) is not None
        or re.fullmatch(r"blazedit_pld_staged_v\d+_\d+_w\d+_n\d+", method) is not None
        or re.fullmatch(r"vantage_force_pld_w\d+_n\d+", method) is not None
        or re.fullmatch(r"vantage_force_pld_m\d+_w\d+_n\d+", method) is not None
        or re.fullmatch(r"pld_plus_mtp_heads(?:_w\d+_n\d+)?", method) is not None
        or re.fullmatch(r"vantage_residual_k[124](?:_t\d+)?(?:_w\d+_n\d+)?", method) is not None
        or re.fullmatch(r"vantage_residual_router_k[124](?:_w\d+_n\d+)?", method) is not None
        or re.fullmatch(r"pld_queued_mtp_heads(?:_w\d+_n\d+)?", method) is not None
        or re.fullmatch(r"lookahead_w\d+_n\d+_i\d+", method) is not None
        or re.fullmatch(r"pld_gated_lookahead_w\d+_n\d+_i\d+(?:_d\d+)?", method) is not None
        or re.fullmatch(r"pld_gated_lookahead_w\d+_n\d+", method) is not None
        or re.fullmatch(
            r"weak_router_capped_pld(?:_t\d+)?(?:_cap\d+)?_w\d+_n\d+",
            method,
        )
        is not None
        or re.fullmatch(r"rerank_exact_pld_k\d+_w\d+_n\d+", method) is not None
        or re.fullmatch(r"vantage_staged_pld_v\d+_\d+_w\d+_n\d+", method) is not None
        or re.fullmatch(
            r"delta_cache_pld(?:_p\d+)?(?:_c\d+)?(?:_lru\d+)?(?:_pw\d+)?_w\d+_n\d+",
            method,
        )
        is not None
        or re.fullmatch(
            r"fuzzy_resync_pld(?:_fd\d+)?(?:_weak\d+)?_w\d+_n\d+",
            method,
        )
        is not None
        or re.fullmatch(r"blazedit_assisted_(static|dynamic)_w\d+", method) is not None
        or re.fullmatch(r"blazedit_two_layer_m\d+_r\d+_n\d+", method) is not None
    )


def _argmax_int(logits_row: torch.Tensor) -> int:
    return int(logits_row.argmax(dim=-1).item())


def prompt_lookup_draft(
    tokens: list[int],
    *,
    max_matching_ngram_size: int,
    max_draft_tokens: int,
    min_matching_ngram_size: int = 1,
) -> tuple[list[int], int, int, int]:
    """Return a local prompt-lookup continuation.

    The query is the current suffix of ``tokens``.  We search previous
    non-overlapping occurrences from longest to shortest n-gram length and
    copy the following tokens up to ``max_draft_tokens``.  The continuation
    is truncated before it would overlap the current query.

    Returns ``(draft, match_len, source_start, follow_start)``.  Positions are
    token indices in ``tokens``; ``-1`` indicates no match.
    """
    if max_draft_tokens <= 0 or not tokens:
        return [], 0, -1, -1
    max_n = min(max_matching_ngram_size, len(tokens))
    for n in range(max_n, min_matching_ngram_size - 1, -1):
        current_start = len(tokens) - n
        if current_start <= 0:
            continue
        suffix = tokens[current_start:]
        best_start: int | None = None
        # Most recent non-overlapping match.
        for start in range(current_start - n, -1, -1):
            if tokens[start : start + n] == suffix:
                best_start = start
                break
        if best_start is None:
            continue
        follow_start = best_start + n
        follow_end = min(follow_start + max_draft_tokens, current_start)
        draft = tokens[follow_start:follow_end]
        if draft:
            return list(draft), n, best_start, follow_start
    return [], 0, -1, -1


def _count_prompt_lookup_matches(tokens: list[int], *, match_len: int) -> int:
    """Count prior non-overlapping exact matches for the current suffix length."""
    if match_len <= 0 or len(tokens) < match_len * 2:
        return 0
    current_start = len(tokens) - match_len
    suffix = tokens[current_start:]
    count = 0
    for start in range(current_start - match_len, -1, -1):
        if tokens[start : start + match_len] == suffix:
            count += 1
    return count


def _prompt_lookup_candidate_positions(
    tokens: list[int],
    *,
    match_len: int,
    top_k: int,
) -> list[int]:
    """Return most-recent-first exact PLD source positions for the suffix."""
    if match_len <= 0 or top_k <= 0 or len(tokens) < match_len * 2:
        return []
    current_start = len(tokens) - match_len
    suffix = tokens[current_start:]
    positions: list[int] = []
    for start in range(current_start - match_len, -1, -1):
        if tokens[start : start + match_len] == suffix:
            positions.append(start)
            if len(positions) >= top_k:
                break
    return positions


def _prompt_lookup_candidate_positions_and_count(
    tokens: list[int],
    *,
    match_len: int,
    top_k: int,
) -> tuple[list[int], int]:
    """Return top candidate positions and total exact suffix-match count."""
    if match_len <= 0 or len(tokens) < match_len * 2:
        return [], 0
    current_start = len(tokens) - match_len
    suffix = tokens[current_start:]
    positions: list[int] = []
    count = 0
    for start in range(current_start - match_len, -1, -1):
        if tokens[start : start + match_len] == suffix:
            count += 1
            if len(positions) < top_k:
                positions.append(start)
    return positions, count


def _ensure_position_in_top_k(
    positions: list[int],
    *,
    baseline_position: int,
    top_k: int,
) -> list[int]:
    """Keep the baseline PLD source in the candidate set used for reranking."""
    if baseline_position < 0 or top_k <= 0:
        return positions[: max(0, top_k)]
    limited = list(positions[:top_k])
    if baseline_position in limited:
        return limited
    if baseline_position in positions:
        if len(limited) < top_k:
            limited.append(baseline_position)
        else:
            limited[-1] = baseline_position
    return limited


def _prompt_lookup_draft_from_position(
    tokens: list[int],
    *,
    source_start: int,
    match_len: int,
    max_draft_tokens: int,
) -> tuple[list[int], int]:
    current_start = len(tokens) - match_len
    follow_start = source_start + match_len
    follow_end = min(follow_start + max_draft_tokens, current_start)
    if follow_start >= follow_end:
        return [], follow_start
    return list(tokens[follow_start:follow_end]), follow_start


def _pld_source_type(source_start: int, prompt_len: int) -> str:
    return "prompt/reference" if source_start < prompt_len else "generated"


def _decode_trace_tokens(tokenizer, token_ids: list[int]) -> str | None:
    if tokenizer is None or not token_ids:
        return None
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=False)
    except Exception:
        return None


def _build_pld_opportunity_trace(
    *,
    tokenizer,
    prefix: list[int],
    drafts: list[int],
    match_len: int,
    source_start: int,
) -> _PLDOpportunityTrace:
    candidate_matches = _count_prompt_lookup_matches(prefix, match_len=match_len)
    snippet: list[int] = []
    if source_start >= 0:
        snippet_start = max(0, source_start - 8)
        snippet_end = min(len(prefix), source_start + max(match_len, 1) + 32)
        snippet = prefix[snippet_start:snippet_end]
    return _PLDOpportunityTrace(
        exact_hit=bool(drafts),
        candidate_matches=candidate_matches,
        source_position=source_start,
        draft_len=len(drafts),
        generated_suffix_16_text=_decode_trace_tokens(tokenizer, prefix[-16:]),
        draft_prefix_32_text=_decode_trace_tokens(tokenizer, drafts[:32]),
        source_snippet_text=_decode_trace_tokens(tokenizer, snippet),
    )


def _delta_context_key(
    tokens: list[int],
    *,
    token_pos: int,
    old_token: int,
    context_tokens: int,
) -> tuple[tuple[int, ...], int]:
    start = max(0, token_pos - max(0, context_tokens))
    return tuple(tokens[start:token_pos]), int(old_token)


def _delta_cache_note_failure(
    cache: OrderedDict[tuple[tuple[int, ...], int], _DeltaEntry],
    *,
    context_key: tuple[tuple[int, ...], int],
    new_token: int,
    max_entries: int,
) -> None:
    entry = cache.get(context_key)
    if entry is None:
        cache[context_key] = _DeltaEntry(new_token=int(new_token), support=1)
    elif entry.new_token == int(new_token):
        entry.support += 1
        cache.move_to_end(context_key)
    else:
        entry.conflicts += 1
        cache.move_to_end(context_key)
    while len(cache) > max_entries:
        cache.popitem(last=False)


def _delta_cache_patch_draft(
    *,
    prefix: list[int],
    drafts: list[int],
    cache: OrderedDict[tuple[tuple[int, ...], int], _DeltaEntry],
    context_tokens: int,
    max_patches: int,
    patch_window: int,
) -> tuple[list[int], list[tuple[int, int, int, tuple[tuple[int, ...], int]]]]:
    """Patch future PLD drafts using previously verified local substitutions.

    The returned patch tuples are ``(draft_index, old_token, new_token, key)``.
    Patches are still target-verified; this only changes the draft candidate.
    """
    if not drafts or not cache or max_patches <= 0 or patch_window <= 0:
        return list(drafts), []
    patched = list(drafts)
    stream = list(prefix)
    patches: list[tuple[int, int, int, tuple[tuple[int, ...], int]]] = []
    limit = min(len(patched), int(patch_window))
    for i in range(limit):
        old_token = patched[i]
        key = _delta_context_key(
            stream,
            token_pos=len(stream),
            old_token=old_token,
            context_tokens=context_tokens,
        )
        entry = cache.get(key)
        if entry is None:
            stream.append(old_token)
            continue
        if entry.support <= entry.conflicts or entry.new_token == old_token:
            stream.append(old_token)
            continue
        patched[i] = entry.new_token
        entry.uses += 1
        cache.move_to_end(key)
        patches.append((i, old_token, entry.new_token, key))
        stream.append(entry.new_token)
        if len(patches) >= max_patches:
            stream.extend(patched[i + 1 :])
            break
    return patched, patches


def _edit_distance_leq1(a: list[int], b: list[int]) -> tuple[bool, int]:
    """Return whether two token lists are edit-distance <=1 and that distance."""
    if a == b:
        return True, 0
    if abs(len(a) - len(b)) > 1:
        return False, 2
    if len(a) == len(b):
        mismatches = sum(1 for x, y in zip(a, b) if x != y)
        return mismatches <= 1, mismatches
    if len(a) < len(b):
        a, b = b, a
    i = j = 0
    edits = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False, edits
        i += 1
    if i < len(a):
        edits += 1
    return edits <= 1, edits


def fuzzy_resync_draft(
    tokens: list[int],
    *,
    query_len: int,
    max_draft_tokens: int,
    require_unique: bool = True,
) -> tuple[list[int], int, int, int, int, int]:
    """Approximate PLD resynchronization with edit-distance-1 suffix matching.

    Returns ``(draft, match_len, source_start, follow_start, n_candidates,
    edit_distance)``.  The implementation searches recent non-overlapping
    previous spans of length ``query_len-1``, ``query_len``, and ``query_len+1``.
    Ambiguous matches are accepted only when their initial continuation is
    identical, which keeps the draft source high-confidence without changing
    verifier semantics.
    """
    if query_len <= 0 or max_draft_tokens <= 0 or len(tokens) < query_len + 1:
        return [], 0, -1, -1, 0, 0
    query = tokens[-query_len:]
    current_start = len(tokens) - query_len
    candidates: list[tuple[int, int, list[int], int]] = []
    for span_len in (query_len, query_len - 1, query_len + 1):
        if span_len <= 0 or current_start - span_len < 0:
            continue
        for start in range(current_start - span_len, -1, -1):
            span = tokens[start : start + span_len]
            ok, dist = _edit_distance_leq1(query, span)
            if not ok or dist != 1:
                continue
            follow_start = start + span_len
            follow_end = min(follow_start + max_draft_tokens, current_start)
            draft = tokens[follow_start:follow_end]
            if draft:
                candidates.append((start, span_len, list(draft), dist))
                if len(candidates) > 8:
                    break
        if len(candidates) > 8:
            break
    if not candidates:
        return [], 0, -1, -1, 0, 0

    if require_unique and len(candidates) > 1:
        head_len = min(8, max_draft_tokens)
        heads = {tuple(c[2][:head_len]) for c in candidates}
        if len(heads) != 1:
            return [], 0, -1, -1, len(candidates), 1

    start, span_len, draft, dist = candidates[0]
    return draft, span_len, start, start + span_len, len(candidates), dist


def _mv_rewrite_pairs(prompt_text: str, metadata: dict[str, Any] | None) -> tuple[dict[str, str], str]:
    del metadata
    pairs = _rewrite_pairs(prompt_text or "")
    if pairs:
        return pairs, "explicit"
    return {}, "none"


def _rewrite_term_in_text(term: str, text: str) -> bool:
    if not term or not text:
        return False
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", term):
        return re.search(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", text) is not None
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+", term):
        return re.search(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", text) is not None
    return term in text


def _rewrite_term_matches_at(text: str, term: str, pos: int) -> bool:
    if not term or not text.startswith(term, pos):
        return False
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\.]*|[0-9]+(?:\.[0-9]+)?", term):
        before = text[pos - 1] if pos > 0 else ""
        after_pos = pos + len(term)
        after = text[after_pos] if after_pos < len(text) else ""
        if before and re.match(r"[A-Za-z0-9_]", before):
            return False
        if after and re.match(r"[A-Za-z0-9_]", after):
            return False
    return True


def _rewrite_matches(
    text: str,
    mapping: dict[str, str],
) -> list[tuple[int, int, str, str, str]]:
    """Return non-overlapping rewrite matches in source order.

    This is the setup-time transducer equivalent of ``_apply_word_map``.  It
    applies longest keys first at each source position and records the source
    span so we can map source-token offsets to target-token offsets without
    re-decoding candidate spans in the decode loop.
    """

    if not text or not mapping:
        return []
    ordered = [
        (old, new)
        for old, new in sorted(mapping.items(), key=lambda item: -len(item[0]))
        if old and new and old != new
    ]
    matches: list[tuple[int, int, str, str, str]] = []
    pos = 0
    while pos < len(text):
        chosen: tuple[str, str] | None = None
        for old, new in ordered:
            if _rewrite_term_matches_at(text, old, pos):
                chosen = (old, new)
                break
        if chosen is None:
            pos += 1
            continue
        old, new = chosen
        end = pos + len(old)
        matches.append((pos, end, old, new, _rewrite_pair_key(old, new)))
        pos = end
    return matches


def _apply_word_map_with_offsets(
    text: str,
    mapping: dict[str, str],
) -> tuple[str, list[int], list[tuple[int, int, str, str, str]]]:
    """Apply a rewrite map once and track source-char to value-char offsets."""

    matches = _rewrite_matches(text, mapping)
    if not matches:
        return text, list(range(len(text) + 1)), []

    out: list[str] = []
    source_to_value = [0] * (len(text) + 1)
    cursor = 0
    value_pos = 0
    for start, end, old, new, key in matches:
        if cursor < start:
            chunk = text[cursor:start]
            out.append(chunk)
            for char_pos in range(cursor, start + 1):
                source_to_value[char_pos] = value_pos + (char_pos - cursor)
            value_pos += len(chunk)
        else:
            source_to_value[start] = value_pos

        out.append(new)
        old_len = max(1, end - start)
        new_len = len(new)
        for char_pos in range(start, end + 1):
            if char_pos == end:
                source_to_value[char_pos] = value_pos + new_len
            else:
                rel = char_pos - start
                source_to_value[char_pos] = value_pos + min(
                    new_len,
                    max(0, round(rel * new_len / old_len)),
                )
        value_pos += new_len
        cursor = end

    if cursor < len(text):
        chunk = text[cursor:]
        out.append(chunk)
        for char_pos in range(cursor, len(text) + 1):
            source_to_value[char_pos] = value_pos + (char_pos - cursor)
    else:
        source_to_value[len(text)] = value_pos

    return "".join(out), source_to_value, matches


_PY_KEYWORDS = {
    "False",
    "None",
    "True",
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "try",
    "while",
    "with",
    "yield",
}

_NOISY_REWRITE_TERMS = {
    "a",
    "an",
    "arg",
    "args",
    "cls",
    "config",
    "data",
    "file",
    "i",
    "id",
    "item",
    "items",
    "j",
    "k",
    "m",
    "model",
    "n",
    "name",
    "obj",
    "option",
    "options",
    "param",
    "params",
    "path",
    "result",
    "results",
    "self",
    "test",
    "tests",
    "tmp",
    "type",
    "value",
    "values",
    "x",
    "y",
}


def _rewrite_pair_key(old: str, new: str) -> str:
    return f"{old}\u241f{new}"


def _is_noisy_rewrite_pair(old: str, new: str) -> bool:
    old_tail = old.rsplit(".", 1)[-1].strip("_").lower()
    new_tail = new.rsplit(".", 1)[-1].strip("_").lower()
    if old in _PY_KEYWORDS or new in _PY_KEYWORDS:
        return True
    if len(old_tail) <= 1 or len(new_tail) <= 1:
        return True
    # Single common words cause many accidental frontier signals in real
    # commits.  Dotted rewrites are kept unless both tails are generic.
    if "." not in old and "." not in new and (
        old_tail in _NOISY_REWRITE_TERMS or new_tail in _NOISY_REWRITE_TERMS
    ):
        return True
    if "." in old or "." in new:
        return old_tail in _NOISY_REWRITE_TERMS and new_tail in _NOISY_REWRITE_TERMS
    return False


def _filter_rewrite_pairs_for_quality(pairs: dict[str, str]) -> dict[str, str]:
    return {
        old: new
        for old, new in pairs.items()
        if not _is_noisy_rewrite_pair(old, new)
    }


def _decode_token_slice(tokenizer, tokens: list[int]) -> str:
    if not tokens:
        return ""
    if hasattr(tokenizer, "decode"):
        try:
            return tokenizer.decode(tokens, skip_special_tokens=False)
        except TypeError:
            return tokenizer.decode(tokens)
    try:
        return "".join(chr(t) for t in tokens)
    except (TypeError, ValueError):
        return ""


def _build_mv_lazy_plan(
    tokenizer,
    *,
    prompt_text: str,
    reference: str,
    metadata: dict[str, Any] | None,
    config: VantageMVConfig | None = None,
) -> tuple[_MVLazyPlan | None, dict[str, str], str, float]:
    del tokenizer
    t_map = time.perf_counter_ns()
    pairs, map_source = _mv_rewrite_pairs(prompt_text, metadata)
    if not reference:
        refs = _extract_reference_blocks(prompt_text or "")
        reference = refs[0] if refs else ""
    effective_pairs = {
        old: new
        for old, new in pairs.items()
        if old != new and _rewrite_term_in_text(old, reference)
    }
    if config is not None and config.use_map_quality_gate:
        effective_pairs = _filter_rewrite_pairs_for_quality(effective_pairs)
    map_us = (time.perf_counter_ns() - t_map) / 1000.0
    if not reference or not effective_pairs:
        return None, effective_pairs, map_source, map_us
    plan = _MVLazyPlan(
        reference=reference,
        rewrite_map=effective_pairs,
        map_source=map_source,
        old_terms=sorted(effective_pairs.keys(), key=len, reverse=True),
        new_terms=sorted(effective_pairs.values(), key=len, reverse=True),
    )
    return plan, effective_pairs, map_source, map_us


def _mv_frontier_gate(
    tokenizer,
    *,
    prefix: list[int],
    exact_drafts: list[int],
    plan: _MVLazyPlan,
    config: VantageMVConfig,
) -> tuple[bool, str]:
    # Build the expensive transformed index only when there is concrete local
    # evidence that the next PLD region is near a rewrite boundary.  This keeps
    # ordinary exact-copy regions on the same hot path as BlazEdit PLD.
    draft_probe = _decode_token_slice(
        tokenizer,
        exact_drafts[: max(config.frontier_window, config.transformed_min_matching_ngram_size)],
    )
    if any(_rewrite_term_in_text(term, draft_probe) for term in plan.old_terms):
        return True, "exact_candidate_contains_old_term"

    suffix_probe = _decode_token_slice(
        tokenizer,
        prefix[-max(config.frontier_window * 2, config.transformed_min_matching_ngram_size) :],
    )
    if any(_rewrite_term_in_text(term, suffix_probe) for term in plan.new_terms):
        return True, "generated_suffix_contains_new_term"

    if config.use_hunk_alignment and len(exact_drafts) < config.exact_strong_min_len:
        return True, "hunk_alignment_probe"

    if (
        config.no_frontier_probe_exact_len > 0
        and len(exact_drafts) <= config.no_frontier_probe_exact_len
    ):
        return True, "exact_candidate_weak_probe"

    return False, "no_rewrite_frontier_signal"


def _make_token_index(
    tokens: list[int],
    *,
    min_match_len: int,
    max_match_len: int,
) -> dict[int, dict[tuple[int, ...], list[int]]]:
    index: dict[int, dict[tuple[int, ...], list[int]]] = {}
    max_n = min(max_match_len, len(tokens) - 1)
    for n in range(min_match_len, max_n + 1):
        by_key: dict[tuple[int, ...], list[int]] = {}
        for start in range(0, len(tokens) - n):
            by_key.setdefault(tuple(tokens[start : start + n]), []).append(start)
        index[n] = by_key
    return index


def _mv_view_specs(
    pairs: dict[str, str],
    *,
    map_source: str,
    config: VantageMVConfig,
) -> list[tuple[str, dict[str, str], str]]:
    view_specs: list[tuple[str, dict[str, str], str]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()

    def add_view(view_id: str, view_map: dict[str, str], source_label: str) -> None:
        nonlocal view_specs
        if len(view_specs) >= config.max_views:
            return
        cleaned = RewriteMapSet.from_pairs(view_map, source=map_source).as_dict()
        key = tuple(sorted(cleaned.items()))
        if not cleaned or key in seen:
            return
        seen.add(key)
        view_specs.append((view_id, cleaned, source_label))

    full = RewriteMapSet.from_pairs(pairs, source=map_source).as_dict()
    add_view("full", full, "transformed_reference")

    if not config.use_rewrite_fst or config.use_pair_priors:
        singles = RewriteMapSet.from_pairs(pairs, source=map_source).singles()
        for i, single in enumerate(singles):
            if len(view_specs) >= config.max_views:
                break
            add_view(f"single{i}", single.as_dict(), "single_map_reference")

        items = list(full.items())
        if len(items) > 2:
            subset_i = 0
            # Real commits often partially apply a compound map.  Prioritize
            # small subsets after singles so {r1, r2} views are available before
            # the max-view cap is exhausted.
            for size in range(2, len(items)):
                for subset in itertools.combinations(items, size):
                    if len(view_specs) >= config.max_views:
                        break
                    add_view(
                        f"subset{subset_i}_{size}",
                        dict(subset),
                        "compound_subset_reference",
                    )
                    subset_i += 1
                if len(view_specs) >= config.max_views:
                    break

    for i, derived in enumerate(_derived_field_views(pairs)):
        if len(view_specs) >= config.max_views:
            break
        add_view(f"field{i}", derived, "field_normalized_reference")
    return view_specs


def _materialize_mv_text_views(
    plan: _MVLazyPlan,
    *,
    config: VantageMVConfig,
) -> tuple[list[_MVTextPrecheckView], float]:
    if plan.text_views is not None:
        return plan.text_views, 0.0

    t_apply = time.perf_counter_ns()
    text_views: list[_MVTextPrecheckView] = []
    seen: set[str] = set()
    for view_id, view_map, source_label in _mv_view_specs(
        plan.rewrite_map,
        map_source=plan.map_source,
        config=config,
    ):
        text = _apply_word_map(plan.reference, view_map)
        if text == plan.reference or text in seen:
            continue
        seen.add(text)
        text_views.append(
            _MVTextPrecheckView(
                view_id=view_id,
                rewrite_map=view_map,
                source_label=source_label,
                text=text,
            )
        )
    plan.text_views = text_views
    apply_us = (time.perf_counter_ns() - t_apply) / 1000.0
    return text_views, apply_us


def _find_subsequence(haystack: list[int], needle: tuple[int, ...]) -> int:
    if not needle or len(needle) > len(haystack):
        return -1
    first = needle[0]
    limit = len(haystack) - len(needle)
    for i in range(limit + 1):
        if haystack[i] == first and tuple(haystack[i : i + len(needle)]) == needle:
            return i
    return -1


def _source_token_boundary_chars(text: str, offsets: list[tuple[int, int]], n_tokens: int) -> list[int]:
    boundaries = [0] * (n_tokens + 1)
    if n_tokens <= 0:
        return boundaries
    boundaries[0] = 0
    for token_i in range(1, n_tokens):
        if token_i < len(offsets):
            boundaries[token_i] = max(0, min(len(text), int(offsets[token_i][0])))
        elif offsets:
            boundaries[token_i] = max(0, min(len(text), int(offsets[-1][1])))
        else:
            boundaries[token_i] = 0
    boundaries[n_tokens] = len(text)
    return boundaries


def _token_index_for_char_start(offsets: list[tuple[int, int]], char_pos: int) -> int:
    for i, (_start, end) in enumerate(offsets):
        if end > char_pos:
            return i
    return len(offsets)


def _token_index_for_char_end(offsets: list[tuple[int, int]], char_pos: int) -> int:
    for i, (start, end) in enumerate(offsets):
        if start >= char_pos:
            return i
        if start < char_pos < end:
            return i + 1
    return len(offsets)


def _encode_with_offsets(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        ids = encoded.input_ids
        offsets = encoded.offset_mapping
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if offsets and isinstance(offsets[0], list):
            offsets = offsets[0]
        ids = [int(t) for t in ids]
        offsets = [(int(s), int(e)) for s, e in offsets]
        if len(ids) == len(offsets):
            return ids, offsets
    except TypeError:
        pass
    except Exception:
        pass

    ids = encode_no_special(tokenizer, text)
    return ids, _token_offsets(tokenizer, text, ids)


def _source_token_span_for_chars(
    offsets: list[tuple[int, int]],
    char_start: int,
    char_end: int,
) -> tuple[int, int]:
    return (
        _token_index_for_char_start(offsets, char_start),
        _token_index_for_char_end(offsets, char_end),
    )


def _dedupe_query_replacements(
    replacements: list[tuple[tuple[int, ...], tuple[int, ...], str]],
) -> list[tuple[tuple[int, ...], tuple[int, ...], str]]:
    seen: set[tuple[tuple[int, ...], tuple[int, ...], str]] = set()
    deduped: list[tuple[tuple[int, ...], tuple[int, ...], str]] = []
    for value_tokens, source_tokens, key in replacements:
        if not value_tokens or not source_tokens or value_tokens == source_tokens:
            continue
        item = (value_tokens, source_tokens, key)
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    # Prefer longer target-side spans so dotted names and BPE spans beat their
    # shorter identifier components during canonicalization.
    return sorted(deduped, key=lambda item: (-len(item[0]), -len(item[1]), item[2]))


def _canonicalize_tokens_for_lookup(
    tokens: list[int],
    replacements: list[tuple[tuple[int, ...], tuple[int, ...], str]],
) -> tuple[int, ...]:
    """Project target-surface tokens into FST key-space without text round trips."""

    if not tokens or not replacements:
        return tuple(tokens)
    out: list[int] = []
    pos = 0
    n_tokens = len(tokens)
    while pos < n_tokens:
        matched = False
        for value_tokens, source_tokens, _key in replacements:
            width = len(value_tokens)
            if width <= 0 or pos + width > n_tokens:
                continue
            if tuple(tokens[pos : pos + width]) == value_tokens:
                out.extend(source_tokens)
                pos += width
                matched = True
                break
        if not matched:
            out.append(tokens[pos])
            pos += 1
    return tuple(out)


def _build_transducer_token_view(
    tokenizer,
    *,
    reference: str,
    rewrite_map: dict[str, str],
    include_hunk_frontiers: bool = False,
) -> tuple[
    list[int],
    list[int],
    list[int],
    list[int],
    list[int],
    list[tuple[int, int, str]],
    list[int],
    list[tuple[int, int, str]],
    list[tuple[tuple[int, ...], tuple[int, ...], str]],
]:
    """Build token-aligned FST lookup/value streams.

    The FST key stream is the original reference token stream.  The value
    stream is the whole transformed reference tokenized once.  The two boundary
    arrays map source-token boundaries to value-token boundaries so decode-loop
    lookup can emit ``value_tokens[a:b]`` directly instead of decoding a source
    span, applying rewrites, and retokenizing it.
    """

    value_text, source_to_value_char, matches = _apply_word_map_with_offsets(
        reference,
        rewrite_map,
    )
    source_tokens, source_offsets = _encode_with_offsets(tokenizer, reference)
    value_tokens, value_offsets = _encode_with_offsets(tokenizer, value_text)

    source_boundaries = _source_token_boundary_chars(reference, source_offsets, len(source_tokens))
    source_to_value_start: list[int] = []
    source_to_value_end: list[int] = []
    for source_char in source_boundaries:
        value_char = source_to_value_char[max(0, min(len(reference), source_char))]
        source_to_value_start.append(_token_index_for_char_start(value_offsets, value_char))
        source_to_value_end.append(_token_index_for_char_end(value_offsets, value_char))

    frontiers: list[int] = []
    value_frontiers: list[int] = []
    pair_spans: list[tuple[int, int, str]] = []
    value_pair_spans: list[tuple[int, int, str]] = []
    query_replacements: list[tuple[tuple[int, ...], tuple[int, ...], str]] = []
    for char_start, char_end, _old, _new, key in matches:
        token_start, token_end = _source_token_span_for_chars(
            source_offsets,
            char_start,
            char_end,
        )
        frontiers.append(token_start)
        pair_spans.append((token_start, token_end, key))
        value_start = source_to_value_start[token_start]
        value_end = source_to_value_end[min(token_end, len(source_to_value_end) - 1)]
        value_frontiers.append(value_start)
        value_pair_spans.append((value_start, value_end, key))
        source_span = tuple(source_tokens[token_start:token_end])
        value_span = tuple(value_tokens[value_start:value_end])
        query_replacements.append((value_span, source_span, key))

    if include_hunk_frontiers:
        frontiers.extend(_line_start_token_positions(tokenizer, reference))
        value_frontiers.extend(_line_start_token_positions(tokenizer, value_text))

    return (
        source_tokens,
        value_tokens,
        source_to_value_start,
        source_to_value_end,
        sorted(set(frontiers)),
        pair_spans,
        sorted(set(value_frontiers)),
        value_pair_spans,
        _dedupe_query_replacements(query_replacements),
    )


def _inverse_rewrite_map(mapping: dict[str, str]) -> dict[str, str]:
    return {new: old for old, new in mapping.items() if old and new and old != new}


def _canonicalize_text_for_lookup(text: str, mapping: dict[str, str]) -> str:
    return _apply_word_map(text, _inverse_rewrite_map(mapping))


def _pairs_touched_by_text(text: str, mapping: dict[str, str]) -> tuple[str, ...]:
    if not text or not mapping:
        return ()
    keys: list[str] = []
    for old, new in mapping.items():
        if _rewrite_term_in_text(old, text) or _rewrite_term_in_text(new, text):
            keys.append(_rewrite_pair_key(old, new))
    return tuple(keys)


def _active_map_for_pairs(
    mapping: dict[str, str],
    pair_stats: dict[str, _MVPairStats] | None,
) -> dict[str, str]:
    if not pair_stats:
        return dict(mapping)
    active: dict[str, str] = {}
    for old, new in mapping.items():
        stats = pair_stats.get(_rewrite_pair_key(old, new))
        if stats is not None and stats.disabled and not stats.adopted:
            continue
        active[old] = new
    return active


def _pair_prior(keys: tuple[str, ...], pair_stats: dict[str, _MVPairStats] | None) -> float:
    if not keys or not pair_stats:
        return 1.0
    priors: list[float] = []
    for key in keys:
        stats = pair_stats.get(key)
        if stats is None:
            priors.append(0.45)
        elif stats.disabled and not stats.adopted:
            priors.append(0.05)
        else:
            priors.append((stats.accepted + 4.0) / max(1.0, stats.proposed + 10.0))
    return sum(priors) / max(1, len(priors))


def _tokens_contain_rewrite_value(tokenizer, tokens: list[int], rewrite_map: dict[str, str]) -> bool:
    if not tokens or not rewrite_map:
        return False
    text = _decode_token_slice(tokenizer, tokens)
    return any(_rewrite_term_in_text(new, text) for new in rewrite_map.values())


def _tokens_contain_rewrite_key(tokenizer, tokens: list[int], rewrite_map: dict[str, str]) -> bool:
    if not tokens or not rewrite_map:
        return False
    text = _decode_token_slice(tokenizer, tokens)
    return any(_rewrite_term_in_text(old, text) for old in rewrite_map)


def _mv_exact_draft_cap(match_len: int, config: VantageMVConfig) -> int | None:
    if match_len <= 0:
        return None
    if match_len == 1 and config.exact_match1_draft_cap > 0:
        return min(config.max_draft_tokens, config.exact_match1_draft_cap)
    if 2 <= match_len <= 7 and config.exact_match2_7_draft_cap > 0:
        return min(config.max_draft_tokens, config.exact_match2_7_draft_cap)
    return None


def _mv_cap_exact_drafts(
    exact_drafts: list[int],
    match_len: int,
    config: VantageMVConfig,
) -> tuple[list[int], bool, int | None]:
    cap = _mv_exact_draft_cap(match_len, config)
    if cap is None or len(exact_drafts) <= cap:
        return exact_drafts, False, cap
    return list(exact_drafts[:cap]), True, cap


def _mv_exact_is_strong(
    exact_drafts: list[int],
    exact_match_len: int,
    config: VantageMVConfig,
) -> bool:
    return (
        len(exact_drafts) >= config.exact_strong_min_len
        and exact_match_len >= config.exact_strong_min_match
    )


def _mv_effective_margin(tokenizer, candidate: _MVCandidate, config: VantageMVConfig) -> int:
    if not config.vocab_margin_bypass:
        return config.trans_len_margin
    if candidate.crosses_frontier or candidate.pair_keys:
        return 0
    if _tokens_contain_rewrite_value(tokenizer, candidate.tokens, candidate.view.rewrite_map):
        return 0
    return config.trans_len_margin


def _mv_long_exact_frontier_probe_signal(
    tokenizer,
    *,
    prefix: list[int],
    exact_drafts: list[int],
    plan: _MVLazyPlan,
    config: VantageMVConfig,
) -> bool:
    if not config.allow_long_exact_frontier_probe:
        return False
    draft_probe = _decode_token_slice(
        tokenizer,
        exact_drafts[: max(config.frontier_window, config.transformed_min_matching_ngram_size)],
    )
    if any(_rewrite_term_in_text(term, draft_probe) for term in plan.old_terms):
        return True
    suffix_probe = _decode_token_slice(
        tokenizer,
        prefix[-max(config.frontier_window * 2, config.transformed_min_matching_ngram_size) :],
    )
    return any(_rewrite_term_in_text(term, suffix_probe) for term in plan.new_terms)


def _mv_frontier_probe_required_len(
    *,
    normal_required: int,
    cold_cap: int,
    long_exact_frontier_probe: bool,
    candidate_touches_rewrite: bool,
    config: VantageMVConfig,
) -> int:
    if long_exact_frontier_probe and candidate_touches_rewrite:
        return min(
            cold_cap,
            max(config.frontier_probe_min_draft_tokens, config.transformed_min_matching_ngram_size),
        )
    return normal_required


def _mv_allows_frontier_probe_override(
    tokenizer,
    *,
    candidate: _MVCandidate,
    exact_drafts: list[int],
    config: VantageMVConfig,
) -> bool:
    if not config.allow_long_exact_frontier_probe:
        return False
    min_len = min(
        config.cold_trans_max_draft,
        max(config.frontier_probe_min_draft_tokens, config.transformed_min_matching_ngram_size),
    )
    if len(candidate.tokens) < min_len:
        return False
    if candidate.frontier_distance is None and not candidate.crosses_frontier:
        return False
    if candidate.crosses_frontier or candidate.pair_keys:
        return True
    if (
        candidate.frontier_distance is not None
        and candidate.match_len >= config.max_matching_ngram_size
    ):
        return True
    if _tokens_contain_rewrite_value(tokenizer, candidate.tokens, candidate.view.rewrite_map):
        return True
    return _tokens_contain_rewrite_key(tokenizer, exact_drafts, candidate.view.rewrite_map)


def _mv_conflict_branch_eligible(
    *,
    exact_match_len: int,
    exact_drafts: list[int],
    candidate: _MVCandidate,
    config: VantageMVConfig,
) -> bool:
    exact_weak = exact_match_len < config.transformed_min_matching_ngram_size
    exact_strong_rescue = (
        config.use_pld_rejection_rescue
        and len(exact_drafts) >= config.exact_strong_min_len
    )
    return (
        (config.use_frontier_branch or config.use_packed_branch)
        and bool(exact_drafts)
        and (exact_weak or exact_strong_rescue)
        and candidate.match_len >= config.transformed_min_matching_ngram_size
        and (candidate.crosses_frontier or candidate.frontier_distance == 0)
    )


def _mv_candidate_precheck(
    tokenizer,
    *,
    prefix: list[int],
    exact_drafts: list[int],
    plan: _MVLazyPlan,
    config: VantageMVConfig,
) -> tuple[bool, str, float, float]:
    """Cheaply rule out transformed lookup before full view tokenization.

    A frontier signal only says a rewrite is nearby; it does not mean the
    current token suffix exists in a transformed view or can beat exact PLD.
    This probe materializes transformed reference text once, then tokenizes
    only small windows around text matches for the current suffix.  Full-view
    tokenization/indexing is reserved for cases where the local token probe
    can actually produce a candidate under the same length-margin constraint
    used by the real MV lookup.
    """
    cold_cap = min(config.max_draft_tokens, config.cold_trans_max_draft)
    min_needed = len(exact_drafts)
    long_exact_frontier_probe = (
        min_needed > cold_cap
        and _mv_long_exact_frontier_probe_signal(
            tokenizer,
            prefix=prefix,
            exact_drafts=exact_drafts,
            plan=plan,
            config=config,
        )
    )
    if min_needed > cold_cap and not long_exact_frontier_probe:
        return False, "trans_precheck_margin_impossible", 0.0, 0.0
    probe_window_tokens = max(cold_cap, len(exact_drafts) + config.trans_len_margin)

    max_n = min(config.max_matching_ngram_size, len(prefix))
    if max_n < config.transformed_min_matching_ngram_size:
        return False, "trans_precheck_prefix_too_short", 0.0, 0.0

    apply_us = 0.0
    tokenize_us = 0.0
    if config.use_rewrite_fst:
        for n in range(max_n, config.transformed_min_matching_ngram_size - 1, -1):
            query_text = _decode_token_slice(tokenizer, prefix[-n:])
            if not query_text:
                continue
            canonical_query = _canonicalize_text_for_lookup(query_text, plan.rewrite_map)
            start = plan.reference.rfind(canonical_query)
            checked = 0
            while start >= 0 and checked < 12:
                local_start = max(0, start - 256)
                local_end = min(
                    len(plan.reference),
                    start + len(canonical_query) + max(512, probe_window_tokens * 12),
                )
                local_text = plan.reference[local_start:local_end]
                t_tok = time.perf_counter_ns()
                local_tokens = encode_no_special(tokenizer, local_text)
                local_query_tokens = encode_no_special(tokenizer, canonical_query)
                tokenize_us += (time.perf_counter_ns() - t_tok) / 1000.0
                pos = _find_subsequence(local_tokens, tuple(local_query_tokens))
                if pos >= 0:
                    available = len(local_tokens) - (pos + len(local_query_tokens))
                    if available > 0:
                        source_probe = _decode_token_slice(
                            tokenizer,
                            local_tokens[
                                pos
                                + len(local_query_tokens) : pos
                                + len(local_query_tokens)
                                + cold_cap
                            ],
                        )
                        draft_probe = encode_no_special(
                            tokenizer,
                            _apply_word_map(source_probe, plan.rewrite_map),
                        )
                        margin = (
                            0
                            if config.vocab_margin_bypass
                            and _tokens_contain_rewrite_value(tokenizer, draft_probe, plan.rewrite_map)
                            else config.trans_len_margin
                        )
                        required = _mv_frontier_probe_required_len(
                            normal_required=len(exact_drafts) + margin,
                            cold_cap=cold_cap,
                            long_exact_frontier_probe=long_exact_frontier_probe,
                            candidate_touches_rewrite=long_exact_frontier_probe
                            or _tokens_contain_rewrite_value(
                                tokenizer, draft_probe, plan.rewrite_map
                            ),
                            config=config,
                        )
                        if len(draft_probe) >= required:
                            return True, "trans_precheck_fst_candidate_exists", apply_us, tokenize_us
                start = plan.reference.rfind(canonical_query, 0, start)
                checked += 1
        return False, "trans_precheck_no_token_candidate", apply_us, tokenize_us

    text_views, apply_us = _materialize_mv_text_views(plan, config=config)
    if not text_views:
        return False, "trans_precheck_no_materialized_view", apply_us, tokenize_us

    for n in range(max_n, config.transformed_min_matching_ngram_size - 1, -1):
        query_tokens = tuple(prefix[-n:])
        query_text = _decode_token_slice(tokenizer, list(query_tokens))
        if not query_text:
            continue
        for view in text_views:
            start = view.text.rfind(query_text)
            checked = 0
            while start >= 0 and checked < 12:
                # Tokenize a bounded neighborhood.  This is deliberately
                # conservative: false positives only pay the old full-build
                # cost, while false negatives could lose a useful MV override.
                local_start = max(0, start - 256)
                local_end = min(
                    len(view.text),
                    start + len(query_text) + max(512, probe_window_tokens * 12),
                )
                t_tok = time.perf_counter_ns()
                local_tokens = encode_no_special(tokenizer, view.text[local_start:local_end])
                tokenize_us += (time.perf_counter_ns() - t_tok) / 1000.0
                pos = _find_subsequence(local_tokens, query_tokens)
                if pos >= 0:
                    available = len(local_tokens) - (pos + n)
                    if available <= 0:
                        start = view.text.rfind(query_text, 0, start)
                        checked += 1
                        continue
                    probe_draft = local_tokens[pos + n : pos + n + cold_cap]
                    margin = (
                        0
                        if config.vocab_margin_bypass
                        and _tokens_contain_rewrite_value(tokenizer, probe_draft, view.rewrite_map)
                        else config.trans_len_margin
                    )
                    required = _mv_frontier_probe_required_len(
                        normal_required=len(exact_drafts) + margin,
                        cold_cap=cold_cap,
                        long_exact_frontier_probe=long_exact_frontier_probe,
                        candidate_touches_rewrite=long_exact_frontier_probe
                        or _tokens_contain_rewrite_value(
                            tokenizer, probe_draft, view.rewrite_map
                        ),
                        config=config,
                    )
                    if len(probe_draft) >= required:
                        return True, "trans_precheck_candidate_exists", apply_us, tokenize_us
                start = view.text.rfind(query_text, 0, start)
                checked += 1

    return False, "trans_precheck_no_token_candidate", apply_us, tokenize_us


def _term_token_positions(tokenizer, text: str, terms: list[str]) -> list[int]:
    positions: list[int] = []
    if not text or not terms:
        return positions
    for term in sorted({t for t in terms if t}, key=len, reverse=True):
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\.]*|[0-9]+(?:\.[0-9]+)?", term):
            pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])")
            matches = pattern.finditer(text)
        else:
            matches = re.finditer(re.escape(term), text)
        for match in matches:
            # Tokenize the full prefix up to the byte/char frontier. This is
            # setup-only work; the decode loop never retokenizes spans.
            positions.append(len(encode_no_special(tokenizer, text[: match.start()])))
    return sorted(set(positions))


def _line_start_token_positions(tokenizer, text: str) -> list[int]:
    """Return token offsets for line starts and block-ish boundaries.

    These are setup-only hunk anchors.  They let hunk-aware MV rows resync to
    the transformed reference after an insertion/deletion without pretending
    every token is near a lexical rewrite occurrence.
    """

    if not text:
        return []
    starts = {0}
    for match in re.finditer(r"\n[ \t]*(?=\S)", text):
        starts.add(match.end())
    return sorted({len(encode_no_special(tokenizer, text[:pos])) for pos in starts})


def _build_mv_views(
    tokenizer,
    *,
    prompt_text: str,
    reference: str,
    metadata: dict[str, Any] | None,
    config: VantageMVConfig,
) -> tuple[list[_MVView], dict[str, str], str, float, float, float, float]:
    plan, pairs, map_source, map_us = _build_mv_lazy_plan(
        tokenizer,
        prompt_text=prompt_text,
        reference=reference,
        metadata=metadata,
        config=config,
    )
    if plan is None:
        return [], pairs, map_source, map_us, 0.0, 0.0, 0.0
    views, apply_us, tok_us, index_us = _build_mv_views_from_plan(
        tokenizer,
        plan=plan,
        config=config,
    )
    return views, pairs, map_source, map_us, apply_us, tok_us, index_us


def _build_mv_views_from_plan(
    tokenizer,
    *,
    plan: _MVLazyPlan,
    config: VantageMVConfig,
) -> tuple[list[_MVView], float, float, float]:
    if config.use_rewrite_fst:
        materialized_views: list[_MVTextPrecheckView] = []
        apply_us = 0.0
    else:
        materialized_views, apply_us = _materialize_mv_text_views(plan, config=config)

    t_tok = time.perf_counter_ns()
    fst_payloads: list[
        tuple[
            str,
            dict[str, str],
            str,
            tuple[
                list[int],
                list[int],
                list[int],
                list[int],
                list[int],
                list[tuple[int, int, str]],
                list[int],
                list[tuple[int, int, str]],
                list[tuple[tuple[int, ...], tuple[int, ...], str]],
            ],
        ]
    ] = []
    tokenized: list[tuple[str, dict[str, str], str, str, list[int], list[int]]] = []
    if config.use_rewrite_fst:
        for view_id, view_map, source_label in _mv_view_specs(
            plan.rewrite_map,
            map_source=plan.map_source,
            config=config,
        ):
            fst_payloads.append(
                (
                    view_id,
                    view_map,
                    source_label,
                    _build_transducer_token_view(
                        tokenizer,
                        reference=plan.reference,
                        rewrite_map=view_map,
                        include_hunk_frontiers=config.use_hunk_alignment,
                    ),
                )
            )
    for text_view in materialized_views:
        toks = encode_no_special(tokenizer, text_view.text)
        frontiers = _term_token_positions(
            tokenizer,
            text_view.text,
            list(text_view.rewrite_map.values()),
        )
        if config.use_hunk_alignment:
            frontiers = sorted(set([*frontiers, *_line_start_token_positions(tokenizer, text_view.text)]))
        tokenized.append(
            (
                text_view.view_id,
                text_view.rewrite_map,
                text_view.source_label,
                text_view.text,
                toks,
                frontiers,
            )
        )
    tok_us = (time.perf_counter_ns() - t_tok) / 1000.0

    t_index = time.perf_counter_ns()
    views: list[_MVView] = []
    for view_id, view_map, source_label, fst_payload in fst_payloads:
        (
            source_tokens,
            value_tokens,
            source_to_value_start,
            source_to_value_end,
            source_frontiers,
            pair_spans,
            value_frontiers,
            value_pair_spans,
            query_replacements,
        ) = fst_payload
        source_index = _make_token_index(
            source_tokens,
            min_match_len=config.transformed_min_matching_ngram_size,
            max_match_len=config.max_matching_ngram_size,
        )
        value_index = _make_token_index(
            value_tokens,
            min_match_len=config.transformed_min_matching_ngram_size,
            max_match_len=config.max_matching_ngram_size,
        )
        if value_index or source_index:
            views.append(
                _MVView(
                    view_id=f"rewrite_transducer:{view_id}",
                    tokens=source_tokens,
                    rewrite_map=dict(view_map),
                    source_label=source_label,
                    map_source=plan.map_source,
                    frontiers=source_frontiers,
                    index=source_index,
                    transducer=True,
                    query_map=_inverse_rewrite_map(view_map),
                    reference_text=plan.reference,
                    value_tokens=value_tokens,
                    value_index=value_index,
                    value_frontiers=value_frontiers,
                    value_pair_spans=value_pair_spans,
                    source_to_value_start=source_to_value_start,
                    source_to_value_end=source_to_value_end,
                    pair_spans=pair_spans,
                    query_replacements=query_replacements,
                )
            )
    for view_id, view_map, source_label, _text, toks, frontiers in tokenized:
        if not toks:
            continue
        index = _make_token_index(
            toks,
            min_match_len=config.transformed_min_matching_ngram_size,
            max_match_len=config.max_matching_ngram_size,
        )
        if not index:
            continue
        views.append(
            _MVView(
                view_id=f"{source_label}:{view_id}",
                tokens=toks,
                rewrite_map=view_map,
                source_label=source_label,
                map_source=plan.map_source,
                frontiers=frontiers,
                index=index,
                transducer=(source_label == "rewrite_transducer"),
                query_map=(
                    _inverse_rewrite_map(view_map)
                    if source_label == "rewrite_transducer"
                    else {}
                ),
                reference_text=(_text if source_label == "rewrite_transducer" else None),
            )
        )
    index_us = (time.perf_counter_ns() - t_index) / 1000.0
    return views, apply_us, tok_us, index_us


def _build_generated_reindex_views(
    tokenizer,
    *,
    prefix: list[int],
    prompt_len: int,
    plan: _MVLazyPlan,
    config: VantageMVConfig,
) -> tuple[list[_MVView], float, float, float]:
    """Fold generated target-surface rewrite neighborhoods into MV lookup.

    Exact PLD already indexes the generated prefix globally.  This generated
    view is narrower: it activates only after the model has emitted a rewritten
    term, marks those occurrences as rewrite frontiers, and is treated as an
    adopted transformed source.  That lets later repeated rewritten
    neighborhoods compete using the transformed-view priors rather than waiting
    for a static reference hit.
    """
    generated_tokens = list(prefix[prompt_len:])
    if len(generated_tokens) <= config.transformed_min_matching_ngram_size:
        return [], 0.0, 0.0, 0.0

    t_decode = time.perf_counter_ns()
    generated_text = _decode_token_slice(tokenizer, generated_tokens)
    tok_us = (time.perf_counter_ns() - t_decode) / 1000.0

    t_frontier = time.perf_counter_ns()
    adopted_map = {
        old: new
        for old, new in plan.rewrite_map.items()
        if _rewrite_term_in_text(new, generated_text)
    }
    if not adopted_map:
        apply_us = (time.perf_counter_ns() - t_frontier) / 1000.0
        return [], apply_us, tok_us, 0.0
    generated_frontiers = _term_token_positions(tokenizer, generated_text, list(adopted_map.values()))
    if not generated_frontiers:
        apply_us = (time.perf_counter_ns() - t_frontier) / 1000.0
        return [], apply_us, tok_us, 0.0
    apply_us = (time.perf_counter_ns() - t_frontier) / 1000.0

    t_tok = time.perf_counter_ns()
    # Reuse the actual generated token ids, not a retokenized string, so the
    # view is tokenizer-boundary exact for the target output already observed.
    adopted_tokens = generated_tokens
    tok_us += (time.perf_counter_ns() - t_tok) / 1000.0

    t_index = time.perf_counter_ns()
    index = _make_token_index(
        adopted_tokens,
        min_match_len=config.transformed_min_matching_ngram_size,
        max_match_len=config.max_matching_ngram_size,
    )
    index_us = (time.perf_counter_ns() - t_index) / 1000.0
    if not index:
        return [], apply_us, tok_us, index_us
    return [
        _MVView(
            view_id=f"generated_prefix_reindexed:{len(adopted_map)}",
            tokens=adopted_tokens,
            rewrite_map=adopted_map,
            source_label="generated_prefix_reindexed",
            map_source=plan.map_source,
            frontiers=generated_frontiers,
            index=index,
        )
    ], apply_us, tok_us, index_us


def _mv_view_prior(stats: _MVViewStats) -> float:
    # Conservative prior: cold views get one bounded probe near a frontier; they
    # earn longer drafts only after verifier acceptance.
    return (stats.accepted + 4.0) / max(1.0, stats.proposed + 10.0)


def _mv_draft_cap(config: VantageMVConfig, stats: _MVViewStats, match_len: int) -> int:
    if stats.disabled:
        return 0
    if stats.attempts <= 0:
        if config.use_segment_patch:
            return min(config.max_draft_tokens, config.patch_segment_tokens)
        return min(config.max_draft_tokens, config.cold_trans_max_draft)
    rate = stats.accepted / max(1, stats.proposed)
    if stats.attempts >= config.low_accept_disable_attempts and rate < config.low_accept_disable_rate:
        return 0
    if stats.adopted and rate >= 0.45:
        return config.max_draft_tokens
    if rate >= 0.25 or match_len >= config.max_matching_ngram_size:
        return min(config.max_draft_tokens, config.medium_trans_max_draft)
    return min(config.max_draft_tokens, 8)


def _frontier_relation(
    *,
    view: _MVView,
    source_start: int,
    follow_start: int,
    follow_end: int,
    window: int,
) -> tuple[int | None, bool]:
    return _frontier_relation_in_positions(
        frontiers=view.frontiers,
        source_start=source_start,
        follow_start=follow_start,
        follow_end=follow_end,
        window=window,
    )


def _frontier_relation_in_positions(
    *,
    frontiers: list[int],
    source_start: int,
    follow_start: int,
    follow_end: int,
    window: int,
) -> tuple[int | None, bool]:
    if not frontiers:
        return None, False
    best_dist: int | None = None
    crosses = False
    lo = max(0, source_start - window)
    hi = follow_end + window
    for frontier in frontiers:
        if source_start <= frontier < follow_end:
            crosses = True
            dist = 0
        elif lo <= frontier <= hi:
            dist = min(abs(frontier - follow_start), abs(frontier - source_start))
        else:
            continue
        if best_dist is None or dist < best_dist:
            best_dist = dist
    return best_dist, crosses


def _mv_lookup(
    prefix: list[int],
    view: _MVView,
    stats: _MVViewStats,
    config: VantageMVConfig,
    *,
    tokenizer=None,
    pair_stats: dict[str, _MVPairStats] | None = None,
) -> _MVCandidate | None:
    if stats.disabled or len(prefix) < config.transformed_min_matching_ngram_size:
        return None
    if view.transducer:
        if tokenizer is None:
            return None
        return _mv_transducer_lookup(
            tokenizer,
            prefix=prefix,
            view=view,
            stats=stats,
            config=config,
            pair_stats=pair_stats,
        )
    max_n = min(config.max_matching_ngram_size, len(prefix), len(view.tokens) - 1)
    for n in range(max_n, config.transformed_min_matching_ngram_size - 1, -1):
        starts = view.index.get(n, {}).get(tuple(prefix[-n:]))
        if not starts:
            continue
        cap = _mv_draft_cap(config, stats, n)
        if cap <= 0:
            return None
        for start in reversed(starts):
            if any(lo <= start <= hi for lo, hi in stats.blacklist):
                continue
            follow_start = start + n
            follow_end = min(follow_start + cap, len(view.tokens))
            if follow_start >= follow_end:
                continue
            distance, crosses = _frontier_relation(
                view=view,
                source_start=start,
                follow_start=follow_start,
                follow_end=follow_end,
                window=config.frontier_window,
            )
            if distance is None and not stats.adopted:
                continue
            draft = list(view.tokens[follow_start:follow_end])
            draft_text = ""
            pair_keys: tuple[str, ...] = ()
            if config.use_pair_priors:
                draft_text = _decode_token_slice(tokenizer, draft) if tokenizer is not None else ""
                pair_keys = _pairs_touched_by_text(draft_text, view.rewrite_map)
                if any(
                    pair_stats is not None
                    and pair_stats.get(key) is not None
                    and pair_stats[key].disabled
                    and not pair_stats[key].adopted
                    for key in pair_keys
                ):
                    continue
            prior = _mv_view_prior(stats)
            prior *= _pair_prior(pair_keys, pair_stats) if config.use_pair_priors else 1.0
            score = prior * len(draft) + n / max(1, config.max_matching_ngram_size)
            return _MVCandidate(
                tokens=draft,
                match_len=n,
                source_start=start,
                follow_start=follow_start,
                view=view,
                frontier_distance=distance,
                crosses_frontier=crosses,
                score=score,
                pair_keys=pair_keys,
            )
    return None


def _mv_transducer_lookup(
    tokenizer,
    *,
    prefix: list[int],
    view: _MVView,
    stats: _MVViewStats,
    config: VantageMVConfig,
    pair_stats: dict[str, _MVPairStats] | None,
) -> _MVCandidate | None:
    if not view.value_tokens or not view.source_to_value_start or not view.source_to_value_end:
        return None
    source_max_width = min(
        config.max_matching_ngram_size,
        len(prefix),
        max(0, len(view.tokens) - 1),
    )
    # First try the real transformed-view transducer path: canonicalize the
    # target-surface query into reference-side key tokens, match the source
    # index, and emit from the target-side value stream using the token-offset
    # map.  This is the path that covers different old/new token lengths.
    if view.index and view.query_replacements:
        for width in range(source_max_width, config.transformed_min_matching_ngram_size - 1, -1):
            canonical_query = _canonicalize_tokens_for_lookup(
                prefix[-width:],
                view.query_replacements,
            )
            q_len = len(canonical_query)
            if q_len < config.transformed_min_matching_ngram_size:
                continue
            starts = view.index.get(q_len, {}).get(canonical_query)
            if not starts:
                continue
            cap = _mv_draft_cap(config, stats, q_len)
            if cap <= 0:
                return None
            for start in reversed(starts):
                source_follow = start + q_len
                if source_follow >= len(view.source_to_value_start):
                    continue
                value_start = view.source_to_value_start[source_follow]
                if any(lo <= value_start <= hi for lo, hi in stats.blacklist):
                    continue
                value_end = min(value_start + cap, len(view.value_tokens))
                if value_start >= value_end:
                    continue
                distance, crosses = _frontier_relation_in_positions(
                    frontiers=view.value_frontiers,
                    source_start=view.source_to_value_start[min(start, len(view.source_to_value_start) - 1)],
                    follow_start=value_start,
                    follow_end=value_end,
                    window=config.frontier_window,
                )
                if distance is None and not stats.adopted:
                    continue
                pair_keys = tuple(
                    key
                    for lo, hi, key in view.value_pair_spans
                    if value_start < hi and lo < value_end
                )
                if config.use_pair_priors and any(
                    pair_stats is not None
                    and pair_stats.get(key) is not None
                    and pair_stats[key].disabled
                    and not pair_stats[key].adopted
                    for key in pair_keys
                ):
                    continue
                draft = list(view.value_tokens[value_start:value_end])
                if not draft:
                    continue
                prior = _mv_view_prior(stats)
                prior *= _pair_prior(pair_keys, pair_stats) if config.use_pair_priors else 1.0
                score = prior * len(draft) + q_len / max(1, config.max_matching_ngram_size)
                return _MVCandidate(
                    tokens=draft,
                    match_len=q_len,
                    source_start=value_start,
                    follow_start=value_start,
                    view=view,
                    frontier_distance=distance,
                    crosses_frontier=crosses,
                    score=score,
                    pair_keys=pair_keys,
                )

    direct_max_n = min(config.max_matching_ngram_size, len(prefix), len(view.value_tokens) - 1)
    for n in range(direct_max_n, config.transformed_min_matching_ngram_size - 1, -1):
        starts = view.value_index.get(n, {}).get(tuple(prefix[-n:]))
        if not starts:
            continue
        cap = _mv_draft_cap(config, stats, n)
        if cap <= 0:
            return None
        for start in reversed(starts):
            if any(lo <= start <= hi for lo, hi in stats.blacklist):
                continue
            follow_start = start + n
            follow_end = min(follow_start + cap, len(view.value_tokens))
            if follow_start >= follow_end:
                continue
            distance, crosses = _frontier_relation_in_positions(
                frontiers=view.value_frontiers,
                source_start=start,
                follow_start=follow_start,
                follow_end=follow_end,
                window=config.frontier_window,
            )
            if distance is None and not stats.adopted:
                continue
            pair_keys = tuple(
                key
                for lo, hi, key in view.value_pair_spans
                if start < hi and lo < follow_end
            )
            if config.use_pair_priors and any(
                pair_stats is not None
                and pair_stats.get(key) is not None
                and pair_stats[key].disabled
                and not pair_stats[key].adopted
                for key in pair_keys
            ):
                continue
            draft = list(view.value_tokens[follow_start:follow_end])
            if not draft:
                continue
            prior = _mv_view_prior(stats)
            prior *= _pair_prior(pair_keys, pair_stats) if config.use_pair_priors else 1.0
            score = prior * len(draft) + n / max(1, config.max_matching_ngram_size)
            return _MVCandidate(
                tokens=draft,
                match_len=n,
                source_start=start,
                follow_start=follow_start,
                view=view,
                frontier_distance=distance,
                crosses_frontier=crosses,
                score=score,
                pair_keys=pair_keys,
            )
    return None


def _mv_cursor_candidate(
    cursor: _MVCursorState,
    views: list[_MVView],
    stats_by_view: dict[str, _MVViewStats],
    config: VantageMVConfig,
) -> _MVCandidate | None:
    if not config.use_stateful_cursor or not cursor.active or cursor.view_id is None:
        return None
    view = next((v for v in views if v.view_id == cursor.view_id), None)
    if view is None:
        cursor.reset()
        return None
    stats = stats_by_view.setdefault(view.view_id, _MVViewStats())
    if stats.disabled or not stats.adopted:
        cursor.reset()
        return None
    tokens = view.value_tokens if view.transducer and view.value_tokens else view.tokens
    if cursor.pos < 0 or cursor.pos >= len(tokens):
        cursor.reset()
        return None
    cap = min(config.cursor_max_draft_tokens, config.max_draft_tokens * 2)
    draft = list(tokens[cursor.pos : min(cursor.pos + cap, len(tokens))])
    if not draft:
        cursor.reset()
        return None
    distance, crosses = _frontier_relation_in_positions(
        frontiers=view.value_frontiers if view.transducer and view.value_frontiers else view.frontiers,
        source_start=cursor.pos,
        follow_start=cursor.pos,
        follow_end=cursor.pos + len(draft),
        window=config.frontier_window,
    )
    return _MVCandidate(
        tokens=draft,
        match_len=config.max_matching_ngram_size,
        source_start=cursor.pos,
        follow_start=cursor.pos,
        view=view,
        frontier_distance=distance,
        crosses_frontier=crosses,
        score=(cursor.confidence + 1.0) * len(draft),
        from_cursor=True,
    )


def _common_prefix_len(left: list[int], right: list[int]) -> int:
    n = min(len(left), len(right))
    i = 0
    while i < n and left[i] == right[i]:
        i += 1
    return i


def _tree_ancestor_flat_indices(nodes: list[ProposalTreeNode], flat_idx: int) -> set[int]:
    ancestors = {flat_idx}
    node_idx = flat_idx - 1
    while node_idx >= 0:
        parent = nodes[node_idx].parent
        if parent < 0:
            ancestors.add(0)
            break
        flat_parent = parent + 1
        ancestors.add(flat_parent)
        node_idx = parent
    ancestors.add(0)
    return ancestors


def _select_tree_cache_path(
    cache,
    *,
    base_cache_len: int,
    root_flat_count: int,
    selected_node_flats: list[int],
):
    """Keep prefix/root cache plus the accepted tree path.

    The tree verifier lays out branch nodes as siblings in one forward.  The
    selected path is not necessarily contiguous in that layout, so ordinary
    crop() would keep sibling KV entries.  For DynamicCache-like objects we
    gather exactly the prefix/root positions and selected branch nodes.
    """
    if cache is None:
        return cache
    keep = list(range(base_cache_len + root_flat_count))
    keep.extend(base_cache_len + flat for flat in selected_node_flats)
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for i in range(len(cache.key_cache)):
            key = cache.key_cache[i]
            val = cache.value_cache[i]
            if key is None or val is None:
                continue
            idx = torch.tensor(keep, device=key.device, dtype=torch.long)
            cache.key_cache[i] = key.index_select(-2, idx).contiguous()
            cache.value_cache[i] = val.index_select(-2, idx).contiguous()
        if hasattr(cache, "_seen_tokens"):
            cache._seen_tokens = len(keep)
        return cache
    # Unknown cache type: keep only the verified original prefix/root and let
    # the next decode step catch up the accepted branch with ordinary causal
    # attention.  This preserves correctness but loses some branch speedup.
    crop_dynamic_cache(cache, base_cache_len + root_flat_count)
    return cache


def _mv_rewrite_frontier_signal_from_tokens(
    tokenizer,
    *,
    tokens: list[int],
    plan: _MVLazyPlan | None,
) -> bool:
    if plan is None or not tokens:
        return False
    text = _decode_token_slice(tokenizer, tokens)
    if not text:
        return False
    return any(_rewrite_term_in_text(term, text) for term in [*plan.old_terms, *plan.new_terms])


def _eos_truncate_and_extend(
    prefix: list[int],
    accepted_tokens: list[int],
    eos_token_ids: list[int],
    prompt_len: int,
    max_new_tokens: int,
) -> tuple[int, list[int]]:
    tokens = list(accepted_tokens)
    for i, tk in enumerate(tokens):
        if tk in eos_token_ids:
            tokens = tokens[: i + 1]
            break
    budget = (prompt_len + max_new_tokens) - len(prefix)
    capped = tokens[:budget]
    prefix.extend(capped)
    return len(capped), capped


def _target_chunked_prefill(
    *,
    target,
    tokens: list[int],
    chunk_size: int,
) -> tuple[object | None, int, float]:
    """Build a target KV cache for ``tokens`` without one full eager pass."""

    if not tokens:
        return None, 0, 0.0
    device = next(target.parameters()).device
    cache = None
    total_us = 0.0
    chunk = max(1, int(chunk_size))
    for start in range(0, len(tokens), chunk):
        feed = torch.tensor([tokens[start : start + chunk]], device=device, dtype=torch.long)
        t0 = time.perf_counter_ns()
        try:
            out = target(feed, past_key_values=cache, use_cache=True, logits_to_keep=1)
        except TypeError:
            out = target(feed, past_key_values=cache, use_cache=True)
        total_us += (time.perf_counter_ns() - t0) / 1000.0
        cache = out.past_key_values
    return cache, len(tokens), total_us


@torch.no_grad()
def _target_verify_step(
    *,
    prefix: list[int],
    prompt_len: int,
    target,
    target_cache,
    target_cache_len: int,
    drafts: list[int],
    max_new_tokens: int,
    eos_token_ids: list[int],
    prefill_chunk_size: int = 0,
) -> tuple[object | None, int, GreedyVerifyResult, int, list[int], float]:
    """Verify target drafts or emit one vanilla token when ``drafts`` is empty."""
    device = next(target.parameters()).device
    old_prefix_len = len(prefix)
    prefill_us = 0.0
    if (
        prefill_chunk_size
        and target_cache is None
        and target_cache_len == 0
        and old_prefix_len > 1
    ):
        target_cache, target_cache_len, prefill_us = _target_chunked_prefill(
            target=target,
            tokens=prefix[: old_prefix_len - 1],
            chunk_size=prefill_chunk_size,
        )
    if target_cache_len >= old_prefix_len:
        target_cache_len = max(0, old_prefix_len - 1)
        crop_dynamic_cache(target_cache, target_cache_len)

    n_pre = old_prefix_len - target_cache_len
    target_input_list = prefix[target_cache_len:] + list(drafts)
    target_input = torch.tensor([target_input_list], device=device, dtype=torch.long)

    t0 = time.perf_counter_ns()
    out = target(target_input, past_key_values=target_cache, use_cache=True)
    target_cache = out.past_key_values
    target_cache_len = target_cache_len + len(target_input_list)
    verify_us = prefill_us + (time.perf_counter_ns() - t0) / 1000.0

    if drafts:
        result = greedy_verify(drafts=drafts, target_logits=out.logits, n_pre=n_pre)
    else:
        next_tok = _argmax_int(out.logits[0, n_pre - 1])
        result = GreedyVerifyResult(
            accepted_tokens=[next_tok],
            n_accepted_drafts=0,
            rejected=False,
        )

    n_emitted, accepted_capped = _eos_truncate_and_extend(
        prefix, result.accepted_tokens, eos_token_ids, prompt_len, max_new_tokens
    )

    crop_dynamic_cache(target_cache, max(0, len(prefix) - 1))
    target_cache_len = max(0, len(prefix) - 1)
    return target_cache, target_cache_len, result, n_emitted, accepted_capped, verify_us


@torch.no_grad()
def _target_verify_step_with_mtp_hidden(
    *,
    prefix: list[int],
    prompt_len: int,
    target,
    target_cache,
    target_cache_len: int,
    drafts: list[int],
    max_new_tokens: int,
    eos_token_ids: list[int],
) -> tuple[object | None, int, GreedyVerifyResult, int, list[int], float, torch.Tensor | None]:
    """Verify PLD drafts and return the post-PLD hidden state for MTP.

    The hidden state is at ``old_prefix_len - 1 + n_accepted_drafts``: after
    the PLD-accepted draft prefix, but before the verifier correction/bonus
    token.  That matches the post-PLD offline training target.  The normal
    emitted progress remains ``n_emitted`` (usually ``accepted_len + 1``).
    """
    device = next(target.parameters()).device
    old_prefix_len = len(prefix)
    if target_cache_len >= old_prefix_len:
        target_cache_len = max(0, old_prefix_len - 1)
        crop_dynamic_cache(target_cache, target_cache_len)

    n_pre = old_prefix_len - target_cache_len
    target_input_list = prefix[target_cache_len:] + list(drafts)
    target_input = torch.tensor([target_input_list], device=device, dtype=torch.long)

    t0 = time.perf_counter_ns()
    out = target(
        target_input,
        past_key_values=target_cache,
        use_cache=True,
        output_hidden_states=True,
    )
    target_cache = out.past_key_values
    target_cache_len = target_cache_len + len(target_input_list)
    verify_us = (time.perf_counter_ns() - t0) / 1000.0

    if drafts:
        result = greedy_verify(drafts=drafts, target_logits=out.logits, n_pre=n_pre)
    else:
        next_tok = _argmax_int(out.logits[0, n_pre - 1])
        result = GreedyVerifyResult(
            accepted_tokens=[next_tok],
            n_accepted_drafts=0,
            rejected=False,
        )

    hidden_state: torch.Tensor | None = None
    hidden_abs_pos = old_prefix_len - 1 + int(result.n_accepted_drafts)
    hidden_idx = hidden_abs_pos - (target_cache_len - len(target_input_list))
    if 0 <= hidden_idx < len(target_input_list):
        hidden_state = out.hidden_states[-1][0, hidden_idx].detach()

    n_emitted, accepted_capped = _eos_truncate_and_extend(
        prefix, result.accepted_tokens, eos_token_ids, prompt_len, max_new_tokens
    )

    crop_dynamic_cache(target_cache, max(0, len(prefix) - 1))
    target_cache_len = max(0, len(prefix) - 1)
    return target_cache, target_cache_len, result, n_emitted, accepted_capped, verify_us, hidden_state


@torch.no_grad()
def _target_verify_step_controlled(
    *,
    prefix: list[int],
    prompt_len: int,
    target,
    target_cache,
    target_cache_len: int,
    drafts: list[int],
    max_new_tokens: int,
    eos_token_ids: list[int],
    append_bonus_on_full_accept: bool,
) -> tuple[object | None, int, GreedyVerifyResult, int, list[int], float]:
    """Verify one draft chunk with optional bonus suppression.

    Ordinary speculative verification appends a target "bonus" token when all
    drafts accept.  Staged verification must suppress that bonus for
    intermediate chunks so the next chunk can still be verified against the
    same full candidate rather than against an inserted target token.
    """
    if append_bonus_on_full_accept or not drafts:
        return _target_verify_step(
            prefix=prefix,
            prompt_len=prompt_len,
            target=target,
            target_cache=target_cache,
            target_cache_len=target_cache_len,
            drafts=drafts,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
        )

    device = next(target.parameters()).device
    old_prefix_len = len(prefix)
    if target_cache_len >= old_prefix_len:
        target_cache_len = max(0, old_prefix_len - 1)
        crop_dynamic_cache(target_cache, target_cache_len)

    n_pre = old_prefix_len - target_cache_len
    target_input_list = prefix[target_cache_len:] + list(drafts)
    target_input = torch.tensor([target_input_list], device=device, dtype=torch.long)

    t0 = time.perf_counter_ns()
    out = target(target_input, past_key_values=target_cache, use_cache=True)
    target_cache = out.past_key_values
    target_cache_len = target_cache_len + len(target_input_list)
    verify_us = (time.perf_counter_ns() - t0) / 1000.0

    logits = out.logits[0] if out.logits.dim() == 3 else out.logits
    pred_for_draft = logits[n_pre - 1 : n_pre - 1 + len(drafts)].argmax(dim=-1).tolist()
    accepted_tokens: list[int] = []
    rejected = False
    n_accepted = 0
    for i, tok in enumerate(drafts):
        pred = int(pred_for_draft[i])
        if tok == pred:
            accepted_tokens.append(tok)
            n_accepted += 1
        else:
            accepted_tokens.append(pred)
            rejected = True
            break

    result = GreedyVerifyResult(
        accepted_tokens=accepted_tokens,
        n_accepted_drafts=n_accepted,
        rejected=rejected,
    )
    n_emitted, accepted_capped = _eos_truncate_and_extend(
        prefix,
        result.accepted_tokens,
        eos_token_ids,
        prompt_len,
        max_new_tokens,
    )
    crop_dynamic_cache(target_cache, max(0, len(prefix) - 1))
    target_cache_len = max(0, len(prefix) - 1)
    return target_cache, target_cache_len, result, n_emitted, accepted_capped, verify_us


@torch.no_grad()
def _target_verify_step_staged(
    *,
    prefix: list[int],
    prompt_len: int,
    target,
    target_cache,
    target_cache_len: int,
    drafts: list[int],
    max_new_tokens: int,
    eos_token_ids: list[int],
    first_tokens: int = 16,
    second_tokens: int = 32,
) -> tuple[object | None, int, GreedyVerifyResult, int, list[int], float, int, int]:
    """Verify long drafts in chunks while preserving the full-candidate path."""
    if not drafts or len(drafts) <= max(1, first_tokens):
        (
            target_cache,
            target_cache_len,
            result,
            n_emitted,
            accepted_capped,
            verify_us,
        ) = _target_verify_step(
            prefix=prefix,
            prompt_len=prompt_len,
            target=target,
            target_cache=target_cache,
            target_cache_len=target_cache_len,
            drafts=drafts,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
        )
        return target_cache, target_cache_len, result, n_emitted, accepted_capped, verify_us, 1, len(drafts)

    old_prefix_len = len(prefix)
    total_verify_us = 0.0
    total_accepted_drafts = 0
    verified_draft_tokens = 0
    chunks = [max(1, first_tokens), max(1, second_tokens)]
    pos = 0
    chunk_count = 0
    while pos < len(drafts):
        chunk_len = chunks[chunk_count] if chunk_count < len(chunks) else len(drafts) - pos
        chunk = drafts[pos : min(len(drafts), pos + chunk_len)]
        final_chunk = pos + len(chunk) >= len(drafts)
        (
            target_cache,
            target_cache_len,
            result,
            n_emitted,
            accepted_capped,
            verify_us,
        ) = _target_verify_step_controlled(
            prefix=prefix,
            prompt_len=prompt_len,
            target=target,
            target_cache=target_cache,
            target_cache_len=target_cache_len,
            drafts=chunk,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            append_bonus_on_full_accept=final_chunk,
        )
        chunk_count += 1
        verified_draft_tokens += len(chunk)
        total_verify_us += verify_us
        total_accepted_drafts += int(result.n_accepted_drafts)
        if (
            result.rejected
            or final_chunk
            or len(prefix) >= prompt_len + max_new_tokens
            or any(t in eos_token_ids for t in accepted_capped)
        ):
            return (
                target_cache,
                target_cache_len,
                GreedyVerifyResult(
                    accepted_tokens=list(prefix[old_prefix_len:]),
                    n_accepted_drafts=total_accepted_drafts,
                    rejected=bool(result.rejected),
                ),
                len(prefix[old_prefix_len:]),
                list(prefix[old_prefix_len:]),
                total_verify_us,
                chunk_count,
                verified_draft_tokens,
            )
        pos += len(chunk)

    return (
        target_cache,
        target_cache_len,
        GreedyVerifyResult(
            accepted_tokens=list(prefix[old_prefix_len:]),
            n_accepted_drafts=total_accepted_drafts,
            rejected=False,
        ),
        len(prefix[old_prefix_len:]),
        list(prefix[old_prefix_len:]),
        total_verify_us,
        chunk_count,
        len(drafts),
    )


@torch.no_grad()
def _target_frontier_branch_step_linear(
    *,
    prefix: list[int],
    prompt_len: int,
    target,
    target_cache,
    target_cache_len: int,
    exact_drafts: list[int],
    transformed_drafts: list[int],
    common_len: int,
    max_new_tokens: int,
    eos_token_ids: list[int],
) -> tuple[
    object | None,
    int,
    GreedyVerifyResult,
    int,
    list[int],
    float,
    int | None,
    int,
]:
    """Verify a rewrite-frontier two-branch conflict with linear HF forwards.

    This is a conservative "true branch" prototype for the ordinary
    Transformers path.  It verifies the shared prefix once.  If the target
    bonus token selects either exact PLD or transformed PLD at the divergence,
    it immediately verifies the selected branch tail.  This is not a custom
    tree-attention kernel, but it prevents the decoder from committing to the
    wrong branch before the target has revealed the divergence token.

    ``branch_selected`` is 0 for exact PLD, 1 for transformed PLD, and ``None``
    when the target rejected the shared prefix or selected neither branch.
    """
    old_prefix_len = len(prefix)
    common = list(exact_drafts[:common_len])
    if (
        common_len < 0
        or (common_len > 0 and common != transformed_drafts[:common_len])
        or common_len >= len(exact_drafts)
        or common_len >= len(transformed_drafts)
    ):
        return _target_verify_step(
            prefix=prefix,
            prompt_len=prompt_len,
            target=target,
            target_cache=target_cache,
            target_cache_len=target_cache_len,
            drafts=transformed_drafts,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
        ) + (1, 0)

    if common_len > 0:
        (
            target_cache,
            target_cache_len,
            first_result,
            _first_emitted,
            _first_accepted,
            first_verify_us,
        ) = _target_verify_step(
            prefix=prefix,
            prompt_len=prompt_len,
            target=target,
            target_cache=target_cache,
            target_cache_len=target_cache_len,
            drafts=common,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
        )
    else:
        (
            target_cache,
            target_cache_len,
            first_result,
            _first_emitted,
            _first_accepted,
            first_verify_us,
        ) = _target_verify_step(
            prefix=prefix,
            prompt_len=prompt_len,
            target=target,
            target_cache=target_cache,
            target_cache_len=target_cache_len,
            drafts=[],
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
        )

    branch_selected: int | None = None
    branch_accepted = int(first_result.n_accepted_drafts)
    total_verify_us = first_verify_us
    if (
        first_result.rejected
        or len(prefix) >= prompt_len + max_new_tokens
        or any(t in eos_token_ids for t in prefix[old_prefix_len:])
        or len(prefix) <= old_prefix_len + common_len
    ):
        accepted = prefix[old_prefix_len:]
        return (
            target_cache,
            target_cache_len,
            GreedyVerifyResult(
                accepted_tokens=list(accepted),
                n_accepted_drafts=branch_accepted,
                rejected=first_result.rejected,
            ),
            len(accepted),
            list(accepted),
            total_verify_us,
            branch_selected,
            branch_accepted,
        )

    divergence_tok = prefix[old_prefix_len + common_len]
    exact_next = exact_drafts[common_len]
    trans_next = transformed_drafts[common_len]
    if divergence_tok == trans_next:
        branch_selected = 1
        selected = transformed_drafts
    elif divergence_tok == exact_next:
        branch_selected = 0
        selected = exact_drafts
    else:
        accepted = prefix[old_prefix_len:]
        return (
            target_cache,
            target_cache_len,
            GreedyVerifyResult(
                accepted_tokens=list(accepted),
                n_accepted_drafts=branch_accepted,
                rejected=False,
            ),
            len(accepted),
            list(accepted),
            total_verify_us,
            branch_selected,
            branch_accepted,
        )

    branch_accepted += 1
    remaining = list(selected[common_len + 1 :])
    if remaining and len(prefix) < prompt_len + max_new_tokens:
        (
            target_cache,
            target_cache_len,
            second_result,
            _second_emitted,
            _second_accepted,
            second_verify_us,
        ) = _target_verify_step(
            prefix=prefix,
            prompt_len=prompt_len,
            target=target,
            target_cache=target_cache,
            target_cache_len=target_cache_len,
            drafts=remaining,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
        )
        total_verify_us += second_verify_us
        branch_accepted += int(second_result.n_accepted_drafts)
        rejected = bool(second_result.rejected)
    else:
        rejected = False

    accepted = prefix[old_prefix_len:]
    return (
        target_cache,
        target_cache_len,
        GreedyVerifyResult(
            accepted_tokens=list(accepted),
            n_accepted_drafts=branch_accepted,
            rejected=rejected,
        ),
        len(accepted),
        list(accepted),
        total_verify_us,
        branch_selected,
        branch_accepted,
    )


def _repeat_cache_batch(cache, batch_size: int):
    """Return a batch-repeated copy of a HF cache without mutating the input."""
    if cache is None:
        return None
    repeated = copy.deepcopy(cache)
    if hasattr(repeated, "key_cache") and hasattr(repeated, "value_cache"):
        for i in range(len(repeated.key_cache)):
            if repeated.key_cache[i] is not None:
                repeated.key_cache[i] = (
                    repeated.key_cache[i]
                    .expand(batch_size, *repeated.key_cache[i].shape[1:])
                    .contiguous()
                )
            if repeated.value_cache[i] is not None:
                repeated.value_cache[i] = (
                    repeated.value_cache[i]
                    .expand(batch_size, *repeated.value_cache[i].shape[1:])
                    .contiguous()
                )
        return repeated
    if isinstance(cache, tuple):
        out = []
        for layer in cache:
            if isinstance(layer, tuple) and len(layer) >= 2:
                key = layer[0].expand(batch_size, *layer[0].shape[1:]).contiguous()
                value = layer[1].expand(batch_size, *layer[1].shape[1:]).contiguous()
                out.append((key, value, *layer[2:]))
            else:
                out.append(layer)
        return tuple(out)
    return repeated


def _select_cache_batch(cache, batch_index: int):
    """Select one row from a batched HF cache."""
    if cache is None:
        return None
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for i in range(len(cache.key_cache)):
            if cache.key_cache[i] is not None:
                cache.key_cache[i] = cache.key_cache[i][batch_index : batch_index + 1].contiguous()
            if cache.value_cache[i] is not None:
                cache.value_cache[i] = cache.value_cache[i][batch_index : batch_index + 1].contiguous()
        return cache
    if isinstance(cache, tuple):
        out = []
        for layer in cache:
            if isinstance(layer, tuple) and len(layer) >= 2:
                key = layer[0][batch_index : batch_index + 1].contiguous()
                value = layer[1][batch_index : batch_index + 1].contiguous()
                out.append((key, value, *layer[2:]))
            else:
                out.append(layer)
        return tuple(out)
    return cache


@torch.no_grad()
def _target_packed_branch_step(
    *,
    prefix: list[int],
    prompt_len: int,
    target,
    target_cache,
    target_cache_len: int,
    exact_drafts: list[int],
    transformed_drafts: list[int],
    max_new_tokens: int,
    eos_token_ids: list[int],
) -> tuple[
    object | None,
    int,
    GreedyVerifyResult,
    int,
    list[int],
    float,
    int | None,
    int,
]:
    """Verify exact and transformed branches in one batched target forward.

    This avoids custom tree-attention masks.  Both candidate continuations are
    evaluated as separate batch rows sharing a repeated prefix cache; after the
    target's greedy path selects a branch, the sibling row is discarded.
    """
    old_prefix_len = len(prefix)
    common = _common_prefix_len(exact_drafts, transformed_drafts)
    if (
        old_prefix_len <= 0
        or not exact_drafts
        or not transformed_drafts
        or common >= len(exact_drafts)
        or common >= len(transformed_drafts)
    ):
        return _target_frontier_branch_step_linear(
            prefix=prefix,
            prompt_len=prompt_len,
            target=target,
            target_cache=target_cache,
            target_cache_len=target_cache_len,
            exact_drafts=exact_drafts,
            transformed_drafts=transformed_drafts,
            common_len=common,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
        )

    try:
        device = next(target.parameters()).device
        if target_cache_len >= old_prefix_len:
            target_cache_len = max(0, old_prefix_len - 1)
            crop_dynamic_cache(target_cache, target_cache_len)
        n_pre = old_prefix_len - target_cache_len
        prefix_feed = prefix[target_cache_len:]
        rows = [prefix_feed + list(exact_drafts), prefix_feed + list(transformed_drafts)]
        row_lens = [len(r) for r in rows]
        max_len = max(row_lens)
        pad_id = int(getattr(getattr(target, "config", None), "pad_token_id", 0) or 0)
        input_ids = torch.full((2, max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((2, target_cache_len + max_len), dtype=torch.long, device=device)
        if target_cache_len:
            attention_mask[:, :target_cache_len] = 1
        position_ids = torch.zeros((2, max_len), dtype=torch.long, device=device)
        for row_idx, row in enumerate(rows):
            input_ids[row_idx, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
            attention_mask[row_idx, target_cache_len : target_cache_len + len(row)] = 1
            position_ids[row_idx, : len(row)] = torch.arange(
                target_cache_len,
                target_cache_len + len(row),
                dtype=torch.long,
                device=device,
            )

        batched_cache = _repeat_cache_batch(target_cache, 2)
        t0 = time.perf_counter_ns()
        out = target(
            input_ids,
            past_key_values=batched_cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        verify_us = (time.perf_counter_ns() - t0) / 1000.0
        logits = out.logits

        accepted_tokens: list[int] = []
        accepted_drafts = 0
        branch_selected: int | None = None
        selected_branch = 0
        selected_drafts = exact_drafts
        rejected = False

        for i in range(common):
            pred = _argmax_int(logits[0, n_pre - 1 + i])
            if pred != exact_drafts[i]:
                accepted_tokens.append(pred)
                rejected = True
                break
            accepted_tokens.append(pred)
            accepted_drafts += 1

        if not rejected:
            pred = _argmax_int(logits[0, n_pre - 1 + common])
            exact_next = exact_drafts[common]
            trans_next = transformed_drafts[common]
            if pred == trans_next:
                branch_selected = 1
                selected_branch = 1
                selected_drafts = transformed_drafts
            elif pred == exact_next:
                branch_selected = 0
                selected_branch = 0
                selected_drafts = exact_drafts
            else:
                accepted_tokens.append(pred)
                rejected = True

        if not rejected:
            for i in range(common, len(selected_drafts)):
                pred = _argmax_int(logits[selected_branch, n_pre - 1 + i])
                if pred != selected_drafts[i]:
                    accepted_tokens.append(pred)
                    rejected = True
                    break
                accepted_tokens.append(pred)
                accepted_drafts += 1
            if not rejected:
                accepted_tokens.append(
                    _argmax_int(logits[selected_branch, n_pre - 1 + len(selected_drafts)])
                )

        n_emitted, accepted_capped = _eos_truncate_and_extend(
            prefix,
            accepted_tokens,
            eos_token_ids,
            prompt_len,
            max_new_tokens,
        )
        accepted_drafts = min(accepted_drafts, max(0, n_emitted))
        target_cache = _select_cache_batch(out.past_key_values, selected_branch)
        crop_dynamic_cache(target_cache, max(0, len(prefix) - 1))
        target_cache_len = max(0, len(prefix) - 1)
        return (
            target_cache,
            target_cache_len,
            GreedyVerifyResult(
                accepted_tokens=list(accepted_capped),
                n_accepted_drafts=accepted_drafts,
                rejected=rejected,
            ),
            n_emitted,
            list(accepted_capped),
            verify_us,
            branch_selected,
            accepted_drafts,
        )
    except Exception:
        return _target_frontier_branch_step_linear(
            prefix=prefix,
            prompt_len=prompt_len,
            target=target,
            target_cache=target_cache,
            target_cache_len=target_cache_len,
            exact_drafts=exact_drafts,
            transformed_drafts=transformed_drafts,
            common_len=common,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
        )


@torch.no_grad()
def _target_frontier_branch_step(
    *,
    prefix: list[int],
    prompt_len: int,
    target,
    target_cache,
    target_cache_len: int,
    exact_drafts: list[int],
    transformed_drafts: list[int],
    common_len: int,
    max_new_tokens: int,
    eos_token_ids: list[int],
) -> tuple[
    object | None,
    int,
    GreedyVerifyResult,
    int,
    list[int],
    float,
    int | None,
    int,
]:
    """Verify exact-vs-transformed continuations with one tree-attention call.

    The input tree has the current last prefix token as flat root and the two
    root-excluded continuations as candidate branches.  After the target picks
    a path, DynamicCache entries for sibling branches are removed and the
    selected KV path is grafted into ordinary sequential order.
    """
    del common_len  # the tree builder derives the shared prefix directly.
    old_prefix_len = len(prefix)
    if (
        old_prefix_len <= 0
        or not exact_drafts
        or not transformed_drafts
        or target_cache_len != old_prefix_len - 1
    ):
        return _target_frontier_branch_step_linear(
            prefix=prefix,
            prompt_len=prompt_len,
            target=target,
            target_cache=target_cache,
            target_cache_len=target_cache_len,
            exact_drafts=exact_drafts,
            transformed_drafts=transformed_drafts,
            common_len=_common_prefix_len(exact_drafts, transformed_drafts),
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
        )

    try:
        device = next(target.parameters()).device
        nodes = build_candidate_prefix_tree(
            [exact_drafts, transformed_drafts],
            max_nodes=len(exact_drafts) + len(transformed_drafts),
        )
        if not nodes:
            return _target_frontier_branch_step_linear(
                prefix=prefix,
                prompt_len=prompt_len,
                target=target,
                target_cache=target_cache,
                target_cache_len=target_cache_len,
                exact_drafts=exact_drafts,
                transformed_drafts=transformed_drafts,
                common_len=_common_prefix_len(exact_drafts, transformed_drafts),
                max_new_tokens=max_new_tokens,
                eos_token_ids=eos_token_ids,
            )

        tree_tokens = [prefix[-1]] + [node.token for node in nodes]
        n_tree = len(tree_tokens)
        tree_input = torch.tensor([tree_tokens], device=device, dtype=torch.long)
        tree_positions = torch.empty(n_tree, dtype=torch.long, device=device)
        tree_positions[0] = old_prefix_len - 1
        for flat_idx, node in enumerate(nodes, start=1):
            tree_positions[flat_idx] = old_prefix_len + node.depth - 1
        tree_positions = tree_positions.unsqueeze(0)

        kv_len = target_cache_len + n_tree
        dtype = getattr(target, "dtype", torch.float32)
        attn_mask = torch.zeros((1, 1, n_tree, kv_len), dtype=dtype, device=device)
        neg_inf = torch.finfo(dtype).min
        for q_i in range(n_tree):
            visible_new = _tree_ancestor_flat_indices(nodes, q_i) if q_i else {0}
            for k_j in range(n_tree):
                if k_j not in visible_new:
                    attn_mask[0, 0, q_i, target_cache_len + k_j] = neg_inf

        t0 = time.perf_counter_ns()
        out = target(
            tree_input,
            past_key_values=target_cache,
            attention_mask=attn_mask,
            position_ids=tree_positions,
            use_cache=True,
        )
        target_cache = out.past_key_values
        verify_us = (time.perf_counter_ns() - t0) / 1000.0
        logits = out.logits

        children: dict[int, list[int]] = {}
        for idx, node in enumerate(nodes, start=1):
            parent_flat = 0 if node.parent < 0 else node.parent + 1
            children.setdefault(parent_flat, []).append(idx)

        accepted_tokens: list[int] = []
        selected_node_flats: list[int] = []
        current_flat = 0
        rejected = False
        while children.get(current_flat):
            target_pred = _argmax_int(logits[0, current_flat])
            match_flat: int | None = None
            for child_flat in children[current_flat]:
                if tree_tokens[child_flat] == target_pred:
                    match_flat = child_flat
                    break
            if match_flat is None:
                accepted_tokens.append(target_pred)
                rejected = True
                break
            accepted_tokens.append(target_pred)
            selected_node_flats.append(match_flat)
            current_flat = match_flat

        if not rejected:
            accepted_tokens.append(_argmax_int(logits[0, current_flat]))

        n_emitted, accepted_capped = _eos_truncate_and_extend(
            prefix,
            accepted_tokens,
            eos_token_ids,
            prompt_len,
            max_new_tokens,
        )
        accepted_drafts = min(len(selected_node_flats), n_emitted)
        selected_node_flats = selected_node_flats[:accepted_drafts]
        target_cache = _select_tree_cache_path(
            target_cache,
            base_cache_len=target_cache_len,
            root_flat_count=1,
            selected_node_flats=selected_node_flats,
        )
        target_cache_len = old_prefix_len + accepted_drafts

        common = _common_prefix_len(exact_drafts, transformed_drafts)
        branch_selected: int | None = None
        if accepted_drafts > common:
            if accepted_tokens[common] == transformed_drafts[common]:
                branch_selected = 1
            elif accepted_tokens[common] == exact_drafts[common]:
                branch_selected = 0

        return (
            target_cache,
            target_cache_len,
            GreedyVerifyResult(
                accepted_tokens=list(accepted_capped),
                n_accepted_drafts=accepted_drafts,
                rejected=rejected,
            ),
            n_emitted,
            list(accepted_capped),
            verify_us,
            branch_selected,
            accepted_drafts,
        )
    except Exception:
        return _target_frontier_branch_step_linear(
            prefix=prefix,
            prompt_len=prompt_len,
            target=target,
            target_cache=target_cache,
            target_cache_len=target_cache_len,
            exact_drafts=exact_drafts,
            transformed_drafts=transformed_drafts,
            common_len=_common_prefix_len(exact_drafts, transformed_drafts),
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
        )


@torch.no_grad()
def _assistant_catchup(
    *,
    assistant,
    prefix: list[int],
    assistant_cache,
    assistant_cache_len: int,
) -> tuple[object | None, int, torch.Tensor, float, int]:
    """Bring assistant KV cache to ``len(prefix)`` and return next-token logits."""
    device = next(assistant.parameters()).device
    if assistant_cache_len >= len(prefix):
        assistant_cache_len = max(0, len(prefix) - 1)
        crop_dynamic_cache(assistant_cache, assistant_cache_len)
    feed = prefix[assistant_cache_len:]
    if not feed:
        raise RuntimeError("assistant catchup requires at least one token to feed")
    t0 = time.perf_counter_ns()
    out = assistant(
        torch.tensor([feed], device=device, dtype=torch.long),
        past_key_values=assistant_cache,
        use_cache=True,
    )
    assistant_cache = out.past_key_values
    assistant_cache_len += len(feed)
    prefill_us = (time.perf_counter_ns() - t0) / 1000.0
    return assistant_cache, assistant_cache_len, out.logits[0, -1].clone(), prefill_us, len(feed)


@torch.no_grad()
def _append_assistant_token(
    *,
    assistant,
    token: int,
    assistant_cache,
    assistant_cache_len: int,
) -> tuple[object | None, int, torch.Tensor, float]:
    device = next(assistant.parameters()).device
    t0 = time.perf_counter_ns()
    out = assistant(
        torch.tensor([[token]], device=device, dtype=torch.long),
        past_key_values=assistant_cache,
        use_cache=True,
    )
    assistant_cache = out.past_key_values
    assistant_cache_len += 1
    forward_us = (time.perf_counter_ns() - t0) / 1000.0
    return assistant_cache, assistant_cache_len, out.logits[0, -1].clone(), forward_us


def _top1_probability(logits_row: torch.Tensor) -> float:
    logits_f = logits_row.float()
    top = logits_f.max()
    log_z = torch.logsumexp(logits_f, dim=-1)
    return float(torch.exp(top - log_z).item())


@torch.no_grad()
def _assistant_plain_draft(
    *,
    assistant,
    prefix: list[int],
    assistant_cache,
    assistant_cache_len: int,
    max_tokens: int,
    eos_token_ids: list[int],
    confidence_threshold: float | None,
) -> _DraftStats:
    t_start = time.perf_counter_ns()
    (
        assistant_cache,
        assistant_cache_len,
        last_logits,
        prefill_us,
        catchup_tokens,
    ) = _assistant_catchup(
        assistant=assistant,
        prefix=prefix,
        assistant_cache=assistant_cache,
        assistant_cache_len=assistant_cache_len,
    )
    drafts: list[int] = []
    verify_us = prefill_us
    for _ in range(max_tokens):
        if confidence_threshold is not None and _top1_probability(last_logits) < confidence_threshold:
            break
        next_tok = _argmax_int(last_logits)
        drafts.append(next_tok)
        assistant_cache, assistant_cache_len, last_logits, fwd_us = _append_assistant_token(
            assistant=assistant,
            token=next_tok,
            assistant_cache=assistant_cache,
            assistant_cache_len=assistant_cache_len,
        )
        verify_us += fwd_us
        if next_tok in eos_token_ids:
            break
    return _DraftStats(
        drafts=drafts,
        assistant_cache=assistant_cache,
        assistant_cache_len=assistant_cache_len,
        assistant_us=(time.perf_counter_ns() - t_start) / 1000.0,
        assistant_prefill_us=prefill_us,
        assistant_verify_us=verify_us,
        catchup_tokens=catchup_tokens,
    )


@torch.no_grad()
def _assistant_pld_micro_run(
    *,
    assistant,
    assistant_prefix: list[int],
    assistant_cache,
    assistant_cache_len: int,
    last_logits: torch.Tensor,
    max_matching_ngram_size: int,
    max_draft_tokens: int,
    eos_token_ids: list[int],
) -> tuple[list[int], object | None, int, torch.Tensor, int, int, int, float, float]:
    """Run one PLD micro-run verified by the assistant model."""
    t_lookup = time.perf_counter_ns()
    pld, match_len, _, _ = prompt_lookup_draft(
        assistant_prefix,
        max_matching_ngram_size=max_matching_ngram_size,
        max_draft_tokens=max_draft_tokens,
    )
    pld_us = (time.perf_counter_ns() - t_lookup) / 1000.0
    if not pld:
        next_tok = _argmax_int(last_logits)
        assistant_cache, assistant_cache_len, last_logits, fwd_us = _append_assistant_token(
            assistant=assistant,
            token=next_tok,
            assistant_cache=assistant_cache,
            assistant_cache_len=assistant_cache_len,
        )
        return [next_tok], assistant_cache, assistant_cache_len, last_logits, 0, 0, match_len, pld_us, fwd_us

    device = next(assistant.parameters()).device
    base_len = assistant_cache_len
    t_verify = time.perf_counter_ns()
    out = assistant(
        torch.tensor([pld], device=device, dtype=torch.long),
        past_key_values=assistant_cache,
        use_cache=True,
    )
    assistant_cache = out.past_key_values
    assistant_cache_len += len(pld)
    verify_us = (time.perf_counter_ns() - t_verify) / 1000.0

    preds: list[int] = [_argmax_int(last_logits)]
    if len(pld) > 1:
        preds.extend(int(x) for x in out.logits[0, : len(pld) - 1].argmax(dim=-1).tolist())

    accepted = 0
    emitted: list[int] = []
    for i, tok in enumerate(pld):
        if tok == preds[i]:
            emitted.append(tok)
            accepted += 1
            if tok in eos_token_ids:
                crop_dynamic_cache(assistant_cache, base_len + accepted)
                assistant_cache_len = base_len + accepted
                return (
                    emitted,
                    assistant_cache,
                    assistant_cache_len,
                    out.logits[0, accepted - 1].clone(),
                    len(pld),
                    accepted,
                    match_len,
                    pld_us,
                    verify_us,
                )
        else:
            correction = preds[i]
            emitted.append(correction)
            crop_dynamic_cache(assistant_cache, base_len + accepted)
            assistant_cache_len = base_len + accepted
            assistant_cache, assistant_cache_len, last_logits, fwd_us = _append_assistant_token(
                assistant=assistant,
                token=correction,
                assistant_cache=assistant_cache,
                assistant_cache_len=assistant_cache_len,
            )
            return (
                emitted,
                assistant_cache,
                assistant_cache_len,
                last_logits,
                len(pld),
                accepted,
                match_len,
                pld_us,
                verify_us + fwd_us,
            )

    last_logits = out.logits[0, -1].clone()
    if len(emitted) < max_draft_tokens and emitted[-1] not in eos_token_ids:
        bonus = _argmax_int(last_logits)
        emitted.append(bonus)
        assistant_cache, assistant_cache_len, last_logits, fwd_us = _append_assistant_token(
            assistant=assistant,
            token=bonus,
            assistant_cache=assistant_cache,
            assistant_cache_len=assistant_cache_len,
        )
        verify_us += fwd_us
    return (
        emitted,
        assistant_cache,
        assistant_cache_len,
        last_logits,
        len(pld),
        accepted,
        match_len,
        pld_us,
        verify_us,
    )


@torch.no_grad()
def _assistant_two_layer_draft(
    *,
    assistant,
    prefix: list[int],
    assistant_cache,
    assistant_cache_len: int,
    config: BlazEditConfig,
    eos_token_ids: list[int],
) -> _DraftStats:
    t_start = time.perf_counter_ns()
    (
        assistant_cache,
        assistant_cache_len,
        last_logits,
        prefill_us,
        catchup_tokens,
    ) = _assistant_catchup(
        assistant=assistant,
        prefix=prefix,
        assistant_cache=assistant_cache,
        assistant_cache_len=assistant_cache_len,
    )
    drafts: list[int] = []
    pld_us = 0.0
    verify_us = prefill_us
    pld_proposed = 0
    pld_accepted = 0
    max_match = 0
    max_total = config.micro_draft_tokens * config.max_num_run
    micro_runs = 0

    for _ in range(config.max_num_run):
        if len(drafts) >= max_total:
            break
        micro_runs += 1
        assistant_prefix = prefix + drafts
        remaining = min(config.micro_draft_tokens, max_total - len(drafts))
        (
            emitted,
            assistant_cache,
            assistant_cache_len,
            last_logits,
            proposed,
            accepted,
            match_len,
            this_pld_us,
            this_verify_us,
        ) = _assistant_pld_micro_run(
            assistant=assistant,
            assistant_prefix=assistant_prefix,
            assistant_cache=assistant_cache,
            assistant_cache_len=assistant_cache_len,
            last_logits=last_logits,
            max_matching_ngram_size=config.max_matching_ngram_size,
            max_draft_tokens=remaining,
            eos_token_ids=eos_token_ids,
        )
        drafts.extend(emitted[:remaining])
        pld_us += this_pld_us
        verify_us += this_verify_us
        pld_proposed += proposed
        pld_accepted += accepted
        max_match = max(max_match, match_len)
        if any(t in eos_token_ids for t in emitted):
            break

    return _DraftStats(
        drafts=drafts[:max_total],
        assistant_cache=assistant_cache,
        assistant_cache_len=assistant_cache_len,
        assistant_us=(time.perf_counter_ns() - t_start) / 1000.0,
        assistant_prefill_us=prefill_us,
        assistant_pld_us=pld_us,
        assistant_verify_us=verify_us,
        micro_runs=micro_runs,
        pld_proposed=pld_proposed,
        pld_accepted=pld_accepted,
        pld_max_match_len=max_match,
        catchup_tokens=catchup_tokens,
    )


@torch.no_grad()
def blazedit_speculative_ar(
    prompt_ids: torch.Tensor,
    target,
    assistant,
    max_new_tokens: int,
    eos_token_ids: list[int],
    *,
    config: BlazEditConfig,
    method_name: str,
    tokenizer=None,
) -> DecodeResult:
    """Run a BlazEdit-style baseline under the repo's greedy verifier."""
    device = next(target.parameters()).device
    prompt_ids_list = prompt_ids.tolist() if prompt_ids.dim() == 1 else prompt_ids[0].tolist()
    prefix: list[int] = list(prompt_ids_list)
    prompt_len = len(prefix)

    target_cache = None
    target_cache_len = 0
    assistant_cache = None
    assistant_cache_len = 0
    queued_mtp_draft: _QueuedMTPDraft | None = None

    steps: list[StepRecord] = []
    t_start = time.perf_counter_ns()
    step_idx = 0
    delta_cache: OrderedDict[tuple[tuple[int, ...], int], _DeltaEntry] = OrderedDict()
    previous_good_source: int | None = None
    lookahead_state = _LookaheadState()
    reranker_weights: PLDRerankerWeights | None = None
    if config.mode == "rerank_exact_pld":
        reranker_weights = load_reranker_weights(
            config.pld_rerank_weights_path or DEFAULT_WEIGHTS_PATH
        )
    mtp_heads: PLDMTPHeads | None = None
    if config.mode in {"pld_plus_mtp_heads", "pld_queued_mtp_heads"} and not config.mtp_disable:
        if config.mtp_position != "post_pld":
            raise ValueError("runtime MTP heads only support post_pld mode")
        mtp_heads = _load_mtp_heads_for_runtime(
            checkpoint_path=config.mtp_heads_checkpoint,
            device=device,
            dtype=next(target.parameters()).dtype,
            expected_num_heads=config.mtp_num_heads,
        )
    weak_pld_router = None
    weak_pld_router_history: dict[str, Any] | None = None
    if config.mode == "weak_router_capped_pld":
        from scripts.train_weak_pld_router import _empty_history

        weak_pld_router = _load_weak_pld_router_for_runtime(config.weak_pld_router_path)
        weak_pld_router_history = _empty_history()
    elif config.mode == "pld_gated_lookahead":
        from scripts.train_weak_pld_router import _empty_history

        weak_pld_router_history = _empty_history()
        if config.pld_lookahead_router == "hist_gbdt":
            weak_pld_router = _load_optional_weak_pld_router_for_runtime(
                config.pld_lookahead_router_path
            )
    pld_family_modes = {
        "pld",
        "pld_plus_mtp_heads",
        "pld_queued_mtp_heads",
        "weak_router_capped_pld",
        "delta_cache_pld",
        "fuzzy_resync_pld",
        "rerank_exact_pld",
        "pld_gated_lookahead",
    }

    while len(prefix) < prompt_len + max_new_tokens:
        t_step_start = time.perf_counter_ns()
        old_prefix_len = len(prefix)
        variant_info = _PLDVariantStepInfo()
        delta_patches: list[tuple[int, int, int, tuple[tuple[int, ...], int]]] = []
        pld_opp_trace: _PLDOpportunityTrace | None = None
        mtp_triggered = False
        mtp_token0_rejected: bool | None = None
        mtp_used_token0_rejected: bool | None = None
        mtp_accepted_prefix_len = 0
        mtp_actual_extra_progress = 0
        mtp_extra_accepted_drafts = 0
        mtp_head_compute_us = 0.0
        mtp_verify_extra_us = 0.0
        mtp_predicted_tokens: int | None = None
        mtp_verified_draft_tokens = 0
        mtp_hidden_state: torch.Tensor | None = None
        mtp_queue_prediction_created = False
        mtp_queue_prediction_used = False
        mtp_queue_dropped_pld_strong = False
        mtp_queue_dropped_position_mismatch = False
        mtp_queue_expired = False
        mtp_extra_verify_calls = 0
        mtp_normal_verify_reuse = False
        pld_cap_router_probability: float | None = None
        pld_cap_router_predicted_weak: bool | None = None
        pld_cap_triggered = False
        pld_cap_raw_draft_len: int | None = None
        pld_cap_capped_draft_len: int | None = None
        pld_cap_router_us = 0.0
        lookahead_triggered = False
        lookahead_candidate_len: int | None = None
        lookahead_accepted_len: int | None = None
        lookahead_tok0_reject: bool | None = None
        lookahead_forward_calls = 0
        lookahead_us = 0.0
        lookahead_forward_us = 0.0
        lookahead_candidate_build_us = 0.0
        lookahead_verify_us = 0.0
        lookahead_accepted_per_forward: float | None = None
        lookahead_stable_prefix_len: int | None = None
        lookahead_cache_seeded: bool | None = None
        pld_lookahead_router_prob: float | None = None
        pld_lookahead_predicted_weak: bool | None = None
        pld_lookahead_predicted_weak_reason: str | None = None
        pld_would_have_draft_len: int | None = None
        pld_lookahead_pld_used: bool | None = None
        pld_lookahead_skipped_pld: bool | None = None
        pld_lookahead_fallback_used: bool | None = None

        if config.mode == "lookahead":
            remaining_budget = (prompt_len + max_new_tokens) - old_prefix_len
            la_stats = _lookahead_jacobi_draft(
                prefix=prefix,
                target=target,
                target_cache=target_cache,
                target_cache_len=target_cache_len,
                config=config,
                state=lookahead_state,
                max_tokens=remaining_budget,
            )
            drafts = la_stats.drafts
            match_len = 0
            source_start = -1
            follow_start = -1
            proposal_us = la_stats.lookahead_us
            proposal_kind = "lookahead"
            lookahead_triggered = True
            lookahead_candidate_len = la_stats.candidate_len
            lookahead_forward_calls = la_stats.forward_calls
            lookahead_us = la_stats.lookahead_us
            lookahead_forward_us = la_stats.forward_us
            lookahead_candidate_build_us = la_stats.candidate_build_us
            lookahead_stable_prefix_len = la_stats.stable_prefix_len
            lookahead_cache_seeded = la_stats.cache_seeded
            draft_stats = _DraftStats(
                drafts=drafts,
                assistant_cache=assistant_cache,
                assistant_cache_len=assistant_cache_len,
                assistant_pld_us=proposal_us,
                pld_proposed=0,
                pld_accepted=0,
                pld_max_match_len=0,
            )
            assistant_us = 0.0
            assistant_prefill_us = 0.0
            assistant_verify_us = 0.0
            assistant_pld_us = proposal_us
        elif config.mode in pld_family_modes:
            t_lookup = time.perf_counter_ns()
            drafts, match_len, source_start, follow_start = prompt_lookup_draft(
                prefix,
                max_matching_ngram_size=config.max_matching_ngram_size,
                max_draft_tokens=config.micro_draft_tokens,
                min_matching_ngram_size=config.min_matching_ngram_size,
            )
            raw_pld_drafts = list(drafts)
            proposal_us = (time.perf_counter_ns() - t_lookup) / 1000.0
            variant_info.exact_hit = bool(drafts)
            if config.pld_opportunity_trace and config.mode == "pld":
                pld_opp_trace = _build_pld_opportunity_trace(
                    tokenizer=tokenizer,
                    prefix=prefix,
                    drafts=drafts,
                    match_len=match_len,
                    source_start=source_start,
                )
            if config.mode in {"pld", "pld_plus_mtp_heads", "pld_queued_mtp_heads"}:
                proposal_kind = "blazedit_pld"
            elif config.mode == "weak_router_capped_pld":
                proposal_kind = "weak_router_capped_pld"
            elif config.mode == "pld_gated_lookahead":
                proposal_kind = "blazedit_pld"
            elif config.mode == "delta_cache_pld":
                proposal_kind = "delta_cache_pld"
            elif config.mode == "fuzzy_resync_pld":
                proposal_kind = "fuzzy_resync_pld"
            else:
                proposal_kind = "rerank_exact_pld"

            if config.mode == "weak_router_capped_pld":
                pld_cap_raw_draft_len = len(drafts)
                if weak_pld_router is None or weak_pld_router_history is None:
                    raise RuntimeError("weak PLD router was not initialized")
                t_router = time.perf_counter_ns()
                features = _weak_pld_router_step_features(
                    prefix_len=len(prefix),
                    prompt_len=prompt_len,
                    step_idx=step_idx,
                    drafts=drafts,
                    match_len=match_len,
                    source_start=source_start,
                    follow_start=follow_start,
                    proposal_us=proposal_us,
                    history=weak_pld_router_history,
                )
                pld_cap_router_probability = float(
                    weak_pld_router.predict_proba([features])[0][1]
                )
                pld_cap_router_us = (time.perf_counter_ns() - t_router) / 1000.0
                pld_cap_router_predicted_weak = (
                    pld_cap_router_probability >= float(config.weak_pld_router_threshold)
                )
                if pld_cap_router_predicted_weak:
                    cap = max(0, int(config.weak_pld_cap_tokens))
                    if len(drafts) > cap:
                        pld_cap_triggered = True
                        drafts = list(drafts[:cap])
                pld_cap_capped_draft_len = len(drafts)
                proposal_us += pld_cap_router_us

            if config.mode == "pld_gated_lookahead":
                candidate_count = _count_prompt_lookup_matches(prefix, match_len=match_len)
                pld_would_have_draft_len = len(drafts)
                pld_lookahead_predicted_weak = False
                router_predicted_weak = False
                if config.pld_lookahead_router == "none":
                    router_predicted_weak = False
                elif config.pld_lookahead_router == "hist_gbdt" and weak_pld_router is not None and weak_pld_router_history is not None:
                    t_router = time.perf_counter_ns()
                    features = _weak_pld_router_step_features(
                        prefix_len=len(prefix),
                        prompt_len=prompt_len,
                        step_idx=step_idx,
                        drafts=drafts,
                        match_len=match_len,
                        source_start=source_start,
                        follow_start=follow_start,
                        proposal_us=proposal_us,
                        history=weak_pld_router_history,
                    )
                    pld_lookahead_router_prob = float(
                        weak_pld_router.predict_proba([features])[0][1]
                    )
                    pld_cap_router_us = (time.perf_counter_ns() - t_router) / 1000.0
                    router_predicted_weak = (
                        pld_lookahead_router_prob
                        >= float(config.pld_lookahead_router_threshold)
                    )
                    if router_predicted_weak:
                        pld_lookahead_predicted_weak_reason = "hist_gbdt"
                    proposal_us += pld_cap_router_us
                else:
                    pld_lookahead_predicted_weak_reason = _pld_lookahead_rule_weak_reason(
                        drafts=drafts,
                        match_len=match_len,
                        candidate_count=candidate_count,
                        threshold=int(config.pld_lookahead_weak_threshold),
                    )
                    router_predicted_weak = pld_lookahead_predicted_weak_reason is not None
                trigger_mode = str(config.pld_lookahead_trigger)
                if trigger_mode == "pld_miss":
                    pld_lookahead_predicted_weak = not bool(drafts)
                    pld_lookahead_predicted_weak_reason = (
                        "pld_miss" if pld_lookahead_predicted_weak else None
                    )
                elif trigger_mode == "router_high_conf_weak":
                    if pld_lookahead_router_prob is not None:
                        pld_lookahead_predicted_weak = (
                            pld_lookahead_router_prob
                            >= float(config.pld_lookahead_router_threshold)
                        )
                    else:
                        pld_lookahead_predicted_weak = bool(router_predicted_weak)
                else:
                    pld_lookahead_predicted_weak = bool(router_predicted_weak)
                if pld_lookahead_predicted_weak:
                    remaining_budget = (prompt_len + max_new_tokens) - old_prefix_len
                    la_stats = _lookahead_jacobi_draft(
                        prefix=prefix,
                        target=target,
                        target_cache=target_cache,
                        target_cache_len=target_cache_len,
                        config=config,
                        state=lookahead_state,
                        max_tokens=remaining_budget,
                    )
                    lookahead_candidate_len = la_stats.candidate_len
                    lookahead_forward_calls = la_stats.forward_calls
                    lookahead_us = la_stats.lookahead_us
                    lookahead_forward_us = la_stats.forward_us
                    lookahead_candidate_build_us = la_stats.candidate_build_us
                    lookahead_stable_prefix_len = la_stats.stable_prefix_len
                    lookahead_cache_seeded = la_stats.cache_seeded
                    proposal_us += la_stats.lookahead_us
                    if len(la_stats.drafts) >= int(config.pld_lookahead_min_candidate_len):
                        drafts = list(la_stats.drafts)
                        source_start = -1
                        follow_start = -1
                        match_len = 0
                        proposal_kind = "pld_gated_lookahead"
                        lookahead_triggered = True
                        pld_lookahead_skipped_pld = True
                        pld_lookahead_pld_used = False
                    else:
                        pld_lookahead_fallback_used = True
                        pld_lookahead_pld_used = bool(drafts)
                        pld_lookahead_skipped_pld = False
                        if config.pld_lookahead_fallback == "greedy":
                            drafts = []
                            proposal_kind = "pld_gated_lookahead"
                else:
                    pld_lookahead_pld_used = True
                    pld_lookahead_skipped_pld = False

            if config.mode == "rerank_exact_pld" and drafts:
                t_variant = time.perf_counter_ns()
                try:
                    baseline_source_start = source_start
                    positions, candidate_count = _prompt_lookup_candidate_positions_and_count(
                        prefix,
                        match_len=match_len,
                        top_k=config.pld_rerank_top_k,
                    )
                    if config.pld_rerank_always_include_baseline:
                        positions = _ensure_position_in_top_k(
                            positions,
                            baseline_position=baseline_source_start,
                            top_k=config.pld_rerank_top_k,
                        )
                    variant_info.rerank_candidate_count = candidate_count
                    ambiguous = candidate_count > 1
                    if ambiguous or not config.pld_rerank_only_ambiguous:
                        if reranker_weights is None:
                            raise RuntimeError("PLD reranker weights were not loaded")
                        candidates: list[PLDRerankCandidate] = []
                        generated_suffix_start = len(prefix) - match_len
                        for rank0, pos in enumerate(positions):
                            cand_draft, _cand_follow = _prompt_lookup_draft_from_position(
                                prefix,
                                source_start=pos,
                                match_len=match_len,
                                max_draft_tokens=config.micro_draft_tokens,
                            )
                            if not cand_draft:
                                continue
                            draft_text = ""
                            if tokenizer is not None:
                                draft_text = tokenizer.decode(
                                    cand_draft[:32],
                                    skip_special_tokens=False,
                                )
                            left_extension = (
                                compute_left_extension(
                                    prefix,
                                    generated_suffix_start=generated_suffix_start,
                                    candidate_source_suffix_start=pos,
                                    max_left=config.pld_rerank_left_extension_max,
                                )
                                if config.pld_rerank_enable_left_extension
                                else 0
                            )
                            candidates.append(
                                PLDRerankCandidate(
                                    rank0=rank0,
                                    source_position=pos,
                                    source_type=_pld_source_type(pos, prompt_len),
                                    source_distance_from_previous_good_source=(
                                        abs(pos - previous_good_source)
                                        if previous_good_source is not None
                                        else None
                                    ),
                                    draft_tokens=tuple(cand_draft),
                                    draft_prefix_text=draft_text,
                                    left_extension=left_extension,
                                )
                            )
                        context = PLDRerankContext(
                            candidate_count=candidate_count,
                            match_len=match_len,
                        )
                        selected, scores, feature_rows = select_candidate_by_policy(
                            candidates,
                            reranker_weights,
                            context=context,
                            top_k=config.pld_rerank_top_k,
                            policy=config.pld_rerank_policy,
                            fixed_rank=config.pld_rerank_fixed_rank,
                        )
                        variant_info.rerank_candidate_positions = [
                            c.source_position for c in candidates
                        ]
                        variant_info.rerank_candidate_source_kinds = [
                            c.source_type for c in candidates
                        ]
                        baseline_idx = next(
                            (
                                i
                                for i, c in enumerate(candidates)
                                if c.source_position == baseline_source_start
                            ),
                            None,
                        )
                        if baseline_idx is not None:
                            variant_info.rerank_baseline_rank = candidates[
                                baseline_idx
                            ].rank0
                            variant_info.rerank_baseline_score = (
                                scores[baseline_idx]
                                if baseline_idx < len(scores)
                                else None
                            )
                        else:
                            variant_info.rerank_baseline_score_missing = True
                        baseline_candidate = (
                            candidates[baseline_idx]
                            if baseline_idx is not None
                            else None
                        )
                        if selected is not None:
                            selected_idx = next(
                                (
                                    i
                                    for i, c in enumerate(candidates)
                                    if c.source_position == selected.source_position
                                ),
                                None,
                            )
                            variant_info.rerank_selected_score = (
                                scores[selected_idx]
                                if selected_idx is not None and selected_idx < len(scores)
                                else None
                            )
                            variant_info.rerank_selected_is_baseline = (
                                selected.source_position == baseline_source_start
                            )
                            if (
                                variant_info.rerank_selected_score is not None
                                and variant_info.rerank_baseline_score is not None
                            ):
                                variant_info.rerank_score_margin = (
                                    variant_info.rerank_selected_score
                                    - variant_info.rerank_baseline_score
                                )
                            if (
                                not variant_info.rerank_selected_is_baseline
                            ):
                                gated_selected, score_margin, gated = (
                                    apply_score_margin_gate(
                                        selected=selected,
                                        baseline=baseline_candidate,
                                        selected_score=variant_info.rerank_selected_score,
                                        baseline_score=variant_info.rerank_baseline_score,
                                        margin_gate=config.pld_rerank_margin_gate,
                                        margin=config.pld_rerank_margin,
                                    )
                                )
                                if score_margin is not None:
                                    variant_info.rerank_score_margin = score_margin
                                if gated:
                                    selected = gated_selected
                                    variant_info.rerank_selected_score = (
                                        variant_info.rerank_baseline_score
                                    )
                                    variant_info.rerank_selected_is_baseline = True
                        if config.pld_rerank_debug_trace:
                            variant_info.rerank_debug_features = []
                            for cand, score, feats in zip(
                                candidates, scores, feature_rows, strict=False
                            ):
                                variant_info.rerank_debug_features.append(
                                    {
                                        "rank": cand.rank0,
                                        "source_position": cand.source_position,
                                        "source_type": cand.source_type,
                                        "score": score,
                                        "features": dict(zip(FEATURE_NAMES, feats)),
                                    }
                                )
                        if selected is not None:
                            selected_draft, selected_follow = (
                                _prompt_lookup_draft_from_position(
                                    prefix,
                                    source_start=selected.source_position,
                                    match_len=match_len,
                                    max_draft_tokens=config.micro_draft_tokens,
                                )
                            )
                            if selected_draft:
                                drafts = selected_draft
                                source_start = selected.source_position
                                follow_start = selected_follow
                                variant_info.triggered = True
                                variant_info.rerank_selected_rank = selected.rank0
                                variant_info.rerank_selected_is_baseline = (
                                    selected.source_position == baseline_source_start
                                )
                            else:
                                variant_info.rerank_fallback = True
                        elif ambiguous:
                            variant_info.rerank_fallback = True
                except Exception:
                    if config.pld_rerank_fallback != "baseline":
                        raise
                    variant_info.rerank_fallback = True
                variant_info.overhead_us += (
                    time.perf_counter_ns() - t_variant
                ) / 1000.0
                proposal_us += variant_info.overhead_us

            if config.mode == "delta_cache_pld" and drafts:
                t_variant = time.perf_counter_ns()
                patched, delta_patches = _delta_cache_patch_draft(
                    prefix=prefix,
                    drafts=drafts,
                    cache=delta_cache,
                    context_tokens=config.delta_context_tokens,
                    max_patches=config.delta_max_patches,
                    patch_window=config.delta_patch_window,
                )
                variant_info.overhead_us += (
                    time.perf_counter_ns() - t_variant
                ) / 1000.0
                variant_info.delta_cache_size = len(delta_cache)
                variant_info.delta_patch_count = len(delta_patches)
                if delta_patches:
                    drafts = patched
                    variant_info.triggered = True
                proposal_us += variant_info.overhead_us

            elif config.mode == "fuzzy_resync_pld" and (
                not drafts or len(drafts) < config.fuzzy_weak_draft_len
            ):
                t_variant = time.perf_counter_ns()
                (
                    fuzzy_drafts,
                    fuzzy_match_len,
                    fuzzy_source_start,
                    fuzzy_follow_start,
                    fuzzy_candidate_count,
                    fuzzy_dist,
                ) = fuzzy_resync_draft(
                    prefix,
                    query_len=config.max_matching_ngram_size,
                    max_draft_tokens=min(
                        config.fuzzy_max_draft_tokens,
                        config.micro_draft_tokens,
                    ),
                    require_unique=config.fuzzy_require_unique,
                )
                variant_info.overhead_us += (
                    time.perf_counter_ns() - t_variant
                ) / 1000.0
                variant_info.fuzzy_candidate_count = fuzzy_candidate_count
                variant_info.fuzzy_edit_distance = fuzzy_dist if fuzzy_drafts else None
                variant_info.fuzzy_match_len = fuzzy_match_len if fuzzy_drafts else None
                if fuzzy_drafts:
                    drafts = fuzzy_drafts
                    match_len = fuzzy_match_len
                    source_start = fuzzy_source_start
                    follow_start = fuzzy_follow_start
                    variant_info.triggered = True
                    proposal_us += variant_info.overhead_us

            if config.mode == "pld_queued_mtp_heads":
                if queued_mtp_draft is not None:
                    if len(prefix) != int(queued_mtp_draft.position):
                        mtp_queue_dropped_position_mismatch = True
                        queued_mtp_draft = None
                    elif not queued_mtp_draft.draft_tokens:
                        mtp_queue_expired = True
                        queued_mtp_draft = None
                    elif (
                        config.mtp_use_queued_only_on_weak_pld
                        and len(drafts) > int(config.mtp_trigger_accepted_len)
                    ):
                        mtp_queue_dropped_pld_strong = True
                        queued_mtp_draft = None
                    elif config.mtp_queue_enabled:
                        drafts = list(queued_mtp_draft.draft_tokens)
                        queued_mtp_draft = None
                        mtp_queue_prediction_used = True
                        mtp_normal_verify_reuse = True
                        proposal_kind = "pld_queued_mtp_heads"
                        source_start = -1
                        follow_start = -1
                        match_len = 0
                    else:
                        mtp_queue_expired = True
                        queued_mtp_draft = None

            draft_stats = _DraftStats(
                drafts=drafts,
                assistant_cache=assistant_cache,
                assistant_cache_len=assistant_cache_len,
                assistant_pld_us=proposal_us,
                pld_proposed=len(drafts),
                pld_accepted=0,
                pld_max_match_len=match_len,
            )
            assistant_us = 0.0
            assistant_prefill_us = 0.0
            assistant_verify_us = 0.0
            assistant_pld_us = proposal_us
        else:
            if assistant is None:
                raise ValueError(f"{method_name} requires an assistant model")
            if config.mode in {"assisted_static", "assisted_dynamic"}:
                draft_stats = _assistant_plain_draft(
                    assistant=assistant,
                    prefix=prefix,
                    assistant_cache=assistant_cache,
                    assistant_cache_len=assistant_cache_len,
                    max_tokens=config.micro_draft_tokens,
                    eos_token_ids=eos_token_ids,
                    confidence_threshold=config.assistant_confidence_threshold,
                )
                proposal_kind = config.mode
            else:
                draft_stats = _assistant_two_layer_draft(
                    assistant=assistant,
                    prefix=prefix,
                    assistant_cache=assistant_cache,
                    assistant_cache_len=assistant_cache_len,
                    config=config,
                    eos_token_ids=eos_token_ids,
                )
                proposal_kind = "blazedit_two_layer"
            assistant_cache = draft_stats.assistant_cache
            assistant_cache_len = draft_stats.assistant_cache_len
            proposal_us = draft_stats.assistant_pld_us
            source_start = -1
            follow_start = -1
            assistant_us = draft_stats.assistant_us
            assistant_prefill_us = draft_stats.assistant_prefill_us
            assistant_verify_us = draft_stats.assistant_verify_us
            assistant_pld_us = draft_stats.assistant_pld_us

        remaining_budget = (prompt_len + max_new_tokens) - old_prefix_len
        drafts = draft_stats.drafts[:remaining_budget]
        staged_chunks = None
        staged_tokens = None
        staged_saved = None
        want_queued_mtp_hidden = (
            config.mode == "pld_queued_mtp_heads"
            and mtp_heads is not None
            and not mtp_queue_prediction_used
            and len(drafts) <= int(config.mtp_trigger_accepted_len)
            and remaining_budget > 1
        )
        if config.use_staged_verification:
            (
                target_cache,
                target_cache_len,
                result,
                n_emitted,
                accepted_capped,
                verify_us,
                staged_chunks,
                staged_tokens,
            ) = _target_verify_step_staged(
                prefix=prefix,
                prompt_len=prompt_len,
                target=target,
                target_cache=target_cache,
                target_cache_len=target_cache_len,
                drafts=drafts,
                max_new_tokens=max_new_tokens,
                eos_token_ids=eos_token_ids,
                first_tokens=config.staged_first_tokens,
                second_tokens=config.staged_second_tokens,
            )
            staged_saved = max(0, len(drafts) - int(staged_tokens or 0))
        elif (
            (config.mode == "pld_plus_mtp_heads" and mtp_heads is not None)
            or want_queued_mtp_hidden
        ):
            (
                target_cache,
                target_cache_len,
                result,
                n_emitted,
                accepted_capped,
                verify_us,
                mtp_hidden_state,
            ) = _target_verify_step_with_mtp_hidden(
                prefix=prefix,
                prompt_len=prompt_len,
                target=target,
                target_cache=target_cache,
                target_cache_len=target_cache_len,
                drafts=drafts,
                max_new_tokens=max_new_tokens,
                eos_token_ids=eos_token_ids,
            )
        else:
            (
                target_cache,
                target_cache_len,
                result,
                n_emitted,
                accepted_capped,
                verify_us,
            ) = _target_verify_step(
                prefix=prefix,
                prompt_len=prompt_len,
                target=target,
                target_cache=target_cache,
                target_cache_len=target_cache_len,
                drafts=drafts,
                max_new_tokens=max_new_tokens,
                eos_token_ids=eos_token_ids,
                prefill_chunk_size=config.target_prefill_chunk_size,
            )

        if (
            config.mode == "pld_plus_mtp_heads"
            and mtp_heads is not None
            and mtp_hidden_state is not None
            and int(result.n_accepted_drafts) <= int(config.mtp_trigger_accepted_len)
        ):
            # This block is deliberately post-PLD: token 0 from the MTP heads
            # must match the verifier correction/bonus token that PLD already
            # emitted. Only tokens beyond that baseline progress are verified
            # in an additional target forward.
            accepted_len = int(result.n_accepted_drafts)
            has_baseline_next = len(accepted_capped) > accepted_len
            base_hit_eos = any(t in eos_token_ids for t in accepted_capped)
            remaining_after_pld = (prompt_len + max_new_tokens) - len(prefix)
            if has_baseline_next and not base_hit_eos and remaining_after_pld > 0:
                mtp_triggered = True
                t_mtp = time.perf_counter_ns()
                hidden = mtp_hidden_state.reshape(1, -1).to(
                    device=device,
                    dtype=next(mtp_heads.parameters()).dtype,
                )
                mtp_predictions = (
                    mtp_heads.predict_token_tensor(hidden)[0].detach().cpu().tolist()
                )[: int(config.mtp_num_heads)]
                mtp_head_compute_us = (time.perf_counter_ns() - t_mtp) / 1000.0
                mtp_predicted_tokens = len(mtp_predictions)
                baseline_next = int(accepted_capped[accepted_len])
                if not mtp_predictions or mtp_predictions[0] != baseline_next:
                    mtp_token0_rejected = True
                else:
                    mtp_token0_rejected = False
                    extra_drafts = mtp_predictions[1 : 1 + max(0, remaining_after_pld)]
                    mtp_verified_draft_tokens = len(extra_drafts)
                    mtp_accepted_prefix_len = 1
                    if extra_drafts:
                        (
                            target_cache,
                            target_cache_len,
                            mtp_result,
                            mtp_n_emitted,
                            mtp_accepted_capped,
                            mtp_verify_extra_us,
                        ) = _target_verify_step(
                            prefix=prefix,
                            prompt_len=prompt_len,
                            target=target,
                            target_cache=target_cache,
                            target_cache_len=target_cache_len,
                            drafts=extra_drafts,
                            max_new_tokens=max_new_tokens,
                            eos_token_ids=eos_token_ids,
                        )
                        mtp_extra_verify_calls += 1
                        mtp_extra_accepted_drafts = int(mtp_result.n_accepted_drafts)
                        mtp_accepted_prefix_len += mtp_extra_accepted_drafts
                        mtp_actual_extra_progress = int(mtp_n_emitted)
                        n_emitted += int(mtp_n_emitted)
                        accepted_capped = [*accepted_capped, *mtp_accepted_capped]
                        verify_us += mtp_verify_extra_us

        if (
            config.mode == "pld_queued_mtp_heads"
            and mtp_heads is not None
            and mtp_hidden_state is not None
            and not mtp_queue_prediction_used
            and int(result.n_accepted_drafts) <= int(config.mtp_trigger_accepted_len)
        ):
            accepted_len = int(result.n_accepted_drafts)
            has_baseline_next = len(accepted_capped) > accepted_len
            base_hit_eos = any(t in eos_token_ids for t in accepted_capped)
            remaining_after_pld = (prompt_len + max_new_tokens) - len(prefix)
            if has_baseline_next and not base_hit_eos and remaining_after_pld > 0:
                mtp_triggered = True
                t_mtp = time.perf_counter_ns()
                hidden = mtp_hidden_state.reshape(1, -1).to(
                    device=device,
                    dtype=next(mtp_heads.parameters()).dtype,
                )
                mtp_predictions = (
                    mtp_heads.predict_token_tensor(hidden)[0].detach().cpu().tolist()
                )[: int(config.mtp_num_heads)]
                mtp_head_compute_us = (time.perf_counter_ns() - t_mtp) / 1000.0
                mtp_predicted_tokens = len(mtp_predictions)
                baseline_next = int(accepted_capped[accepted_len])
                if not mtp_predictions or int(mtp_predictions[0]) != baseline_next:
                    mtp_token0_rejected = True
                else:
                    mtp_token0_rejected = False
                    queued_tokens = [
                        int(t)
                        for t in mtp_predictions[1 : 1 + max(0, remaining_after_pld)]
                    ]
                    if queued_tokens:
                        queued_mtp_draft = _QueuedMTPDraft(
                            position=len(prefix),
                            draft_tokens=queued_tokens,
                            source_step=step_idx,
                        )
                        mtp_queue_prediction_created = True

        if config.mode == "pld_queued_mtp_heads" and mtp_queue_prediction_used:
            mtp_accepted_prefix_len = int(result.n_accepted_drafts)
            mtp_extra_accepted_drafts = int(result.n_accepted_drafts)
            mtp_actual_extra_progress = int(n_emitted)
            mtp_verified_draft_tokens = len(drafts)
            mtp_used_token0_rejected = bool(
                drafts and result.rejected and int(result.n_accepted_drafts) == 0
            )

        if assistant is not None and config.mode in {"assisted_static", "assisted_dynamic", "two_layer"}:
            keep_len = old_prefix_len + result.n_accepted_drafts
            crop_dynamic_cache(assistant_cache, keep_len)
            assistant_cache_len = keep_len

        variant_info.candidate_accepted_len = int(result.n_accepted_drafts)
        variant_info.token01_rejection = bool(
            drafts and result.rejected and int(result.n_accepted_drafts) <= 1
        )
        if lookahead_triggered:
            lookahead_accepted_len = int(result.n_accepted_drafts)
            lookahead_tok0_reject = bool(
                drafts and result.rejected and int(result.n_accepted_drafts) == 0
            )
            lookahead_verify_us = float(verify_us)
            lookahead_accepted_per_forward = (
                float(lookahead_accepted_len) / float(lookahead_forward_calls)
                if lookahead_forward_calls
                else 0.0
            )
        if int(result.n_accepted_drafts) >= 16 and source_start >= 0:
            previous_good_source = int(source_start)
        if config.mode == "weak_router_capped_pld" and weak_pld_router_history is not None:
            _weak_pld_router_update_history(
                history=weak_pld_router_history,
                threshold=4,
                accepted_len=int(result.n_accepted_drafts),
                emitted=int(n_emitted),
                rejected=bool(result.rejected),
                draft_len=len(drafts),
                source_start=source_start,
            )
        if config.mode == "pld_gated_lookahead" and weak_pld_router_history is not None:
            _weak_pld_router_update_history(
                history=weak_pld_router_history,
                threshold=int(config.pld_lookahead_weak_threshold),
                accepted_len=int(result.n_accepted_drafts),
                emitted=int(n_emitted),
                rejected=bool(result.rejected),
                draft_len=len(drafts),
                source_start=source_start,
            )
        pld_reject_pos: int | None = None
        pld_target_reject_token: int | None = None
        pld_draft_reject_token: int | None = None
        if drafts and result.rejected:
            pld_reject_pos = int(result.n_accepted_drafts)
            if pld_reject_pos < len(result.accepted_tokens):
                pld_target_reject_token = int(result.accepted_tokens[pld_reject_pos])
            if pld_reject_pos < len(drafts):
                pld_draft_reject_token = int(drafts[pld_reject_pos])
        if config.mode == "delta_cache_pld":
            accepted_drafts = int(result.n_accepted_drafts)
            if delta_patches:
                first_patch_idx = min(p[0] for p in delta_patches)
                patch_accepted = accepted_drafts > first_patch_idx
                variant_info.delta_patch_accepted = patch_accepted
                variant_info.delta_patch_accept_tail = (
                    max(0, accepted_drafts - first_patch_idx) if patch_accepted else 0
                )
                for patch_idx, _old, _new, key in delta_patches:
                    entry = delta_cache.get(key)
                    if entry is not None and accepted_drafts > patch_idx:
                        entry.accepted_uses += 1
            if result.rejected:
                rejected_idx = int(result.n_accepted_drafts)
                if (
                    "raw_pld_drafts" in locals()
                    and rejected_idx < len(raw_pld_drafts)
                    and rejected_idx < len(result.accepted_tokens)
                ):
                    old_token = int(raw_pld_drafts[rejected_idx])
                    new_token = int(result.accepted_tokens[rejected_idx])
                    if old_token != new_token:
                        context_stream = list(prefix[:old_prefix_len]) + list(
                            raw_pld_drafts[:rejected_idx]
                        )
                        key = _delta_context_key(
                            context_stream,
                            token_pos=len(context_stream),
                            old_token=old_token,
                            context_tokens=config.delta_context_tokens,
                        )
                        _delta_cache_note_failure(
                            delta_cache,
                            context_key=key,
                            new_token=new_token,
                            max_entries=config.delta_lru_size,
                        )
            variant_info.delta_cache_size = len(delta_cache)

        t_step_end = time.perf_counter_ns()
        hit_max_new_tokens = len(prefix) >= prompt_len + max_new_tokens and not any(
            t in eos_token_ids for t in accepted_capped
        )
        if config.mode == "pld_queued_mtp_heads" and mtp_queue_prediction_used:
            record_accepted_drafts = int(result.n_accepted_drafts)
            record_draft_tokens = len(drafts)
        else:
            record_accepted_drafts = int(result.n_accepted_drafts) + int(
                mtp_extra_accepted_drafts
            )
            record_draft_tokens = len(drafts) + int(mtp_verified_draft_tokens)
        mtp_total_overhead_us = mtp_head_compute_us + mtp_verify_extra_us
        mtp_mode_active = config.mode in {"pld_plus_mtp_heads", "pld_queued_mtp_heads"}
        pld_cap_wasted_verified_tokens = (
            max(0, int(pld_cap_capped_draft_len or 0) - int(result.n_accepted_drafts))
            if config.mode == "weak_router_capped_pld"
            else None
        )
        steps.append(
            StepRecord(
                method=method_name,
                step=step_idx,
                k=record_draft_tokens,
                n_accepted_drafts=record_accepted_drafts,
                n_emitted=n_emitted,
                rejected=result.rejected,
                node_type=None,
                deepest_type=None,
                wall_us=(t_step_end - t_step_start) / 1000.0,
                draft_us=assistant_us,
                verify_us=verify_us,
                proposal_kind=(
                    proposal_kind
                    if drafts
                    or config.mode
                    in {"weak_router_capped_pld", "lookahead", "pld_gated_lookahead"}
                    else None
                ),
                proposal_match_len=draft_stats.pld_max_match_len or None,
                proposal_us=proposal_us,
                proposal_tokens=record_draft_tokens,
                n_guaranteed_drafts=0,
                n_accepted_nonroot_drafts=record_accepted_drafts,
                hit_max_new_tokens=hit_max_new_tokens,
                prompt_len=prompt_len,
                proposal_source_start_token=source_start if source_start >= 0 else None,
                proposal_follow_start_token=follow_start if follow_start >= 0 else None,
                proposal_query_len=draft_stats.pld_max_match_len or None,
                proposal_pool="local",
                proposal_root_included=False,
                assistant_model=(
                    config.assistant_model_name
                    if config.mode in {"assisted_static", "assisted_dynamic", "two_layer"}
                    else None
                ),
                assistant_us=assistant_us,
                assistant_prefill_us=assistant_prefill_us,
                assistant_pld_us=assistant_pld_us,
                assistant_verify_us=assistant_verify_us,
                blazedit_micro_runs=(
                    draft_stats.micro_runs if config.mode == "two_layer" else None
                ),
                blazedit_micro_draft_tokens=config.micro_draft_tokens,
                blazedit_max_num_run=config.max_num_run,
                blazedit_pld_proposed=draft_stats.pld_proposed,
                blazedit_pld_accepted=draft_stats.pld_accepted,
                target_draft_tokens=record_draft_tokens,
                target_accepted_nonroot=record_accepted_drafts,
                assistant_cache_catchup_tokens=draft_stats.catchup_tokens,
                verify_staged_chunks=staged_chunks,
                verify_staged_draft_tokens=staged_tokens,
                verify_staged_saved_tokens=staged_saved,
                pld_variant=(
                    config.mode
                    if config.mode
                    in {"delta_cache_pld", "fuzzy_resync_pld", "rerank_exact_pld"}
                    else None
                ),
                pld_exact_hit=(
                    variant_info.exact_hit
                    if config.mode in pld_family_modes
                    else None
                ),
                pld_variant_triggered=(
                    variant_info.triggered
                    if config.mode
                    in {"delta_cache_pld", "fuzzy_resync_pld", "rerank_exact_pld"}
                    else None
                ),
                pld_variant_overhead_us=variant_info.overhead_us,
                pld_candidate_accepted_len=(
                    variant_info.candidate_accepted_len
                    if config.mode
                    in {"delta_cache_pld", "fuzzy_resync_pld", "rerank_exact_pld"}
                    else None
                ),
                pld_token01_rejection=(
                    variant_info.token01_rejection
                    if config.mode
                    in {"delta_cache_pld", "fuzzy_resync_pld", "rerank_exact_pld"}
                    else None
                ),
                pld_delta_cache_size=variant_info.delta_cache_size,
                pld_delta_patch_count=variant_info.delta_patch_count,
                pld_delta_patch_accepted=variant_info.delta_patch_accepted,
                pld_delta_patch_accept_tail=variant_info.delta_patch_accept_tail,
                pld_fuzzy_candidate_count=variant_info.fuzzy_candidate_count,
                pld_fuzzy_edit_distance=variant_info.fuzzy_edit_distance,
                pld_fuzzy_match_len=variant_info.fuzzy_match_len,
                pld_rerank_triggered=(
                    variant_info.triggered
                    if config.mode == "rerank_exact_pld"
                    else None
                ),
                pld_rerank_ambiguous=(
                    (variant_info.rerank_candidate_count or 0) > 1
                    if config.mode == "rerank_exact_pld"
                    else None
                ),
                pld_rerank_candidate_count=(
                    variant_info.rerank_candidate_count
                    if config.mode == "rerank_exact_pld"
                    else None
                ),
                pld_rerank_selected_rank=(
                    variant_info.rerank_selected_rank
                    if config.mode == "rerank_exact_pld"
                    else None
                ),
                pld_rerank_fallback=(
                    variant_info.rerank_fallback
                    if config.mode == "rerank_exact_pld"
                    else None
                ),
                pld_rerank_overhead_us=(
                    variant_info.overhead_us if config.mode == "rerank_exact_pld" else 0.0
                ),
                pld_rerank_baseline_rank=(
                    variant_info.rerank_baseline_rank
                    if config.mode == "rerank_exact_pld"
                    else None
                ),
                pld_rerank_selected_is_baseline=(
                    variant_info.rerank_selected_is_baseline
                    if config.mode == "rerank_exact_pld"
                    else None
                ),
                pld_rerank_selected_score=(
                    variant_info.rerank_selected_score
                    if config.mode == "rerank_exact_pld"
                    else None
                ),
                pld_rerank_baseline_score=(
                    variant_info.rerank_baseline_score
                    if config.mode == "rerank_exact_pld"
                    else None
                ),
                pld_rerank_score_margin=(
                    variant_info.rerank_score_margin
                    if config.mode == "rerank_exact_pld"
                    else None
                ),
                pld_rerank_baseline_score_missing=(
                    variant_info.rerank_baseline_score_missing
                    if config.mode == "rerank_exact_pld"
                    else None
                ),
                pld_rerank_candidate_positions=(
                    variant_info.rerank_candidate_positions
                    if config.mode == "rerank_exact_pld"
                    else None
                ),
                pld_rerank_candidate_source_kinds=(
                    variant_info.rerank_candidate_source_kinds
                    if config.mode == "rerank_exact_pld"
                    else None
                ),
                pld_rerank_debug_features=(
                    variant_info.rerank_debug_features
                    if config.mode == "rerank_exact_pld"
                    else None
                ),
                pld_opp_trace=bool(pld_opp_trace),
                pld_opp_step_id=step_idx if pld_opp_trace else None,
                pld_opp_exact_hit=pld_opp_trace.exact_hit if pld_opp_trace else None,
                pld_opp_candidate_matches=(
                    pld_opp_trace.candidate_matches if pld_opp_trace else None
                ),
                pld_opp_source_position=(
                    pld_opp_trace.source_position if pld_opp_trace else None
                ),
                pld_opp_draft_len=pld_opp_trace.draft_len if pld_opp_trace else None,
                pld_opp_accepted_len=(
                    int(result.n_accepted_drafts) if pld_opp_trace else None
                ),
                pld_opp_rejected_at_position=pld_reject_pos if pld_opp_trace else None,
                pld_opp_target_token_at_rejection=(
                    pld_target_reject_token if pld_opp_trace else None
                ),
                pld_opp_pld_token_at_rejection=(
                    pld_draft_reject_token if pld_opp_trace else None
                ),
                pld_opp_lookup_us=proposal_us if pld_opp_trace else 0.0,
                pld_opp_verify_us=verify_us if pld_opp_trace else 0.0,
                pld_opp_generated_suffix_16_text=(
                    pld_opp_trace.generated_suffix_16_text if pld_opp_trace else None
                ),
                pld_opp_draft_prefix_32_text=(
                    pld_opp_trace.draft_prefix_32_text if pld_opp_trace else None
                ),
                pld_opp_source_snippet_text=(
                    pld_opp_trace.source_snippet_text if pld_opp_trace else None
                ),
                mtp_triggered=mtp_triggered if mtp_mode_active else None,
                mtp_token0_rejected=(
                    mtp_token0_rejected if mtp_mode_active else None
                ),
                mtp_accepted_prefix_len=(
                    mtp_accepted_prefix_len if mtp_mode_active else None
                ),
                mtp_actual_extra_progress=mtp_actual_extra_progress,
                mtp_extra_accepted_drafts=mtp_extra_accepted_drafts,
                mtp_head_compute_us=mtp_head_compute_us,
                mtp_verify_extra_us=mtp_verify_extra_us,
                mtp_total_overhead_us=mtp_total_overhead_us,
                mtp_predicted_tokens=(
                    mtp_predicted_tokens if mtp_mode_active else None
                ),
                mtp_verified_draft_tokens=(
                    mtp_verified_draft_tokens if mtp_mode_active else None
                ),
                mtp_queue_prediction_created=(
                    mtp_queue_prediction_created if mtp_mode_active else None
                ),
                mtp_queue_prediction_used=(
                    mtp_queue_prediction_used if mtp_mode_active else None
                ),
                mtp_queue_dropped_pld_strong=(
                    mtp_queue_dropped_pld_strong if mtp_mode_active else None
                ),
                mtp_queue_dropped_position_mismatch=(
                    mtp_queue_dropped_position_mismatch if mtp_mode_active else None
                ),
                mtp_queue_expired=mtp_queue_expired if mtp_mode_active else None,
                mtp_used_token0_rejected=(
                    mtp_used_token0_rejected if mtp_mode_active else None
                ),
                mtp_extra_verify_calls=mtp_extra_verify_calls,
                mtp_normal_verify_reuse=(
                    mtp_normal_verify_reuse if mtp_mode_active else None
                ),
                pld_cap_variant=(
                    config.mode if config.mode == "weak_router_capped_pld" else None
                ),
                pld_cap_router_probability=pld_cap_router_probability,
                pld_cap_router_predicted_weak=pld_cap_router_predicted_weak,
                pld_cap_threshold=(
                    float(config.weak_pld_router_threshold)
                    if config.mode == "weak_router_capped_pld"
                    else None
                ),
                pld_cap_value=(
                    int(config.weak_pld_cap_tokens)
                    if config.mode == "weak_router_capped_pld"
                    else None
                ),
                pld_cap_triggered=(
                    pld_cap_triggered if config.mode == "weak_router_capped_pld" else None
                ),
                pld_cap_raw_draft_len=pld_cap_raw_draft_len,
                pld_cap_capped_draft_len=pld_cap_capped_draft_len,
                pld_cap_wasted_verified_tokens=pld_cap_wasted_verified_tokens,
                pld_cap_router_us=pld_cap_router_us,
                lookahead_triggered=(
                    lookahead_triggered
                    if config.mode in {"lookahead", "pld_gated_lookahead"}
                    else None
                ),
                lookahead_candidate_len=(
                    lookahead_candidate_len
                    if config.mode in {"lookahead", "pld_gated_lookahead"}
                    else None
                ),
                lookahead_accepted_len=(
                    lookahead_accepted_len
                    if config.mode in {"lookahead", "pld_gated_lookahead"}
                    else None
                ),
                lookahead_tok0_reject=(
                    lookahead_tok0_reject
                    if config.mode in {"lookahead", "pld_gated_lookahead"}
                    else None
                ),
                lookahead_iters=(
                    (1 if bool(config.lookahead_one_forward) else int(config.lookahead_iters))
                    if config.mode in {"lookahead", "pld_gated_lookahead"}
                    else None
                ),
                lookahead_forward_calls=(
                    lookahead_forward_calls
                    if config.mode in {"lookahead", "pld_gated_lookahead"}
                    else None
                ),
                lookahead_us=(
                    lookahead_us
                    if config.mode in {"lookahead", "pld_gated_lookahead"}
                    else 0.0
                ),
                lookahead_forward_us=(
                    lookahead_forward_us
                    if config.mode in {"lookahead", "pld_gated_lookahead"}
                    else 0.0
                ),
                lookahead_candidate_build_us=(
                    lookahead_candidate_build_us
                    if config.mode in {"lookahead", "pld_gated_lookahead"}
                    else 0.0
                ),
                lookahead_verify_us=(
                    lookahead_verify_us
                    if config.mode in {"lookahead", "pld_gated_lookahead"}
                    else 0.0
                ),
                lookahead_accepted_per_forward=(
                    lookahead_accepted_per_forward
                    if config.mode in {"lookahead", "pld_gated_lookahead"}
                    else None
                ),
                lookahead_window=(
                    int(config.lookahead_window)
                    if config.mode in {"lookahead", "pld_gated_lookahead"}
                    else None
                ),
                lookahead_ngram=(
                    int(config.lookahead_ngram)
                    if config.mode in {"lookahead", "pld_gated_lookahead"}
                    else None
                ),
                lookahead_stable_prefix_len=(
                    lookahead_stable_prefix_len
                    if config.mode in {"lookahead", "pld_gated_lookahead"}
                    else None
                ),
                lookahead_cache_seeded=(
                    lookahead_cache_seeded
                    if config.mode in {"lookahead", "pld_gated_lookahead"}
                    else None
                ),
                pld_lookahead_router=(
                    config.pld_lookahead_router
                    if config.mode == "pld_gated_lookahead"
                    else None
                ),
                pld_lookahead_router_prob=pld_lookahead_router_prob,
                pld_lookahead_predicted_weak=pld_lookahead_predicted_weak,
                pld_lookahead_predicted_weak_reason=pld_lookahead_predicted_weak_reason,
                pld_lookahead_trigger=(
                    config.pld_lookahead_trigger
                    if config.mode == "pld_gated_lookahead"
                    else None
                ),
                pld_would_have_draft_len=pld_would_have_draft_len,
                pld_lookahead_mode=(
                    config.pld_lookahead_mode
                    if config.mode == "pld_gated_lookahead"
                    else None
                ),
                pld_lookahead_pld_used=pld_lookahead_pld_used,
                pld_lookahead_skipped_pld=pld_lookahead_skipped_pld,
                pld_lookahead_fallback_used=pld_lookahead_fallback_used,
            )
        )
        step_idx += 1

        if any(t in eos_token_ids for t in accepted_capped):
            break

    return DecodeResult(
        output_token_ids=prefix,
        output_text="",
        n_new_tokens=len(prefix) - prompt_len,
        steps=steps,
        wall_us_total=(time.perf_counter_ns() - t_start) / 1000.0,
    )


@torch.no_grad()
def vantage_mv_pld_speculative_ar(
    prompt_ids: torch.Tensor,
    target,
    tokenizer,
    max_new_tokens: int,
    eos_token_ids: list[int],
    *,
    config: VantageMVConfig,
    method_name: str,
    prompt_text: str = "",
    reference: str = "",
    metadata: dict[str, Any] | None = None,
    assistant=None,
    assistant_model_name: str | None = None,
) -> DecodeResult:
    """Run VANTAGE-MV inside the BlazEdit PLD verifier loop.

    This decoder keeps PLD as the substrate: every step computes exact PLD
    first, immediately verifies it when it is strong, and only consults
    transformed lookup views when exact PLD is weak.  Unlike the older
    code-proposer MultiView path, the exact route never falls back to rooted
    PLD; it is the same rootless BlazEdit PLD cache/crop path used by the
    baseline.
    """
    prompt_ids_list = prompt_ids.tolist() if prompt_ids.dim() == 1 else prompt_ids[0].tolist()
    prefix: list[int] = list(prompt_ids_list)
    prompt_len = len(prefix)

    target_cache = None
    target_cache_len = 0
    assistant_cache = None
    assistant_cache_len = 0
    steps: list[StepRecord] = []
    t_start = time.perf_counter_ns()
    step_idx = 0

    views: list[_MVView] | None = None
    lazy_plan: _MVLazyPlan | None = None
    plan_checked = False
    rewrite_map: dict[str, str] = {}
    map_source = "none"
    view_stats: dict[str, _MVViewStats] = {}
    pair_stats: dict[str, _MVPairStats] = {}
    generated_reindex_views: list[_MVView] = []
    generated_reindex_len = 0
    cursor_state = _MVCursorState()
    rescue_steps_remaining = 0

    while len(prefix) < prompt_len + max_new_tokens:
        t_step_start = time.perf_counter_ns()
        old_prefix_len = len(prefix)

        t_lookup = time.perf_counter_ns()
        exact_drafts, exact_match_len, exact_source_start, exact_follow_start = prompt_lookup_draft(
            prefix,
            max_matching_ngram_size=config.max_matching_ngram_size,
            max_draft_tokens=config.max_draft_tokens,
        )
        exact_lookup_us = (time.perf_counter_ns() - t_lookup) / 1000.0
        exact_drafts, exact_capped, exact_cap = _mv_cap_exact_drafts(
            list(exact_drafts),
            exact_match_len,
            config,
        )

        drafts = list(exact_drafts)
        proposal_kind = "blazedit_pld" if drafts else None
        proposal_match_kind = "exact_pld"
        proposal_pool = "local"
        proposal_route = "exact_pld"
        proposal_route_reason = "pld_default"
        proposal_source_start = exact_source_start
        proposal_follow_start = exact_follow_start
        proposal_match_len = exact_match_len
        proposal_us = exact_lookup_us
        map_parse_us = 0.0
        rewrite_apply_us = 0.0
        tokenize_us = 0.0
        index_build_us = 0.0
        proposal_view_id: str | None = None
        frontier_distance: int | None = None
        compound_view_count: int | None = None
        active_map_count: int | None = None
        route_window_accept_rate = 0.0
        rewrite_zero_accept_streak = 0
        branch_candidates: int | None = None
        branch_depth: int | None = None
        branch_exact_drafts: list[int] | None = None
        branch_transformed_drafts: list[int] | None = None
        branch_common_len: int = 0
        branch_selected: int | None = None
        branch_accepted_tokens: int | None = None
        chosen_mv: _MVCandidate | None = None
        proposal_cursor_pos: int | None = cursor_state.pos if cursor_state.active else None
        proposal_cursor_confidence: float | None = (
            cursor_state.confidence if cursor_state.active else None
        )
        proposal_cursor_resync: bool | None = None
        proposal_adoption_transition: str | None = None
        neural_draft_tokens: int | None = None
        neural_draft_us: float = 0.0
        assistant_cache_catchup_tokens = 0

        rescue_active = bool(config.use_pld_rejection_rescue and rescue_steps_remaining > 0)
        exact_is_strong = _mv_exact_is_strong(
            exact_drafts,
            exact_match_len,
            config,
        )
        allow_mv_lookup = (not exact_is_strong) or (
            config.use_stateful_cursor and cursor_state.active
        ) or rescue_active
        if exact_is_strong:
            proposal_route_reason = (
                "pld_rejection_rescue_probe" if rescue_active else "exact_pld_strong"
            )
        if allow_mv_lookup:
            if not plan_checked:
                (
                    lazy_plan,
                    rewrite_map,
                    map_source,
                    map_parse_us,
                ) = _build_mv_lazy_plan(
                    tokenizer,
                    prompt_text=prompt_text,
                    reference=reference,
                    metadata=metadata,
                    config=config,
                )
                plan_checked = True
            if lazy_plan is None:
                proposal_route_reason = "no_effective_rewrite_map"
            elif views is None:
                if rescue_active:
                    should_build, gate_reason = True, "pld_rejection_rescue_window"
                else:
                    t_gate = time.perf_counter_ns()
                    should_build, gate_reason = _mv_frontier_gate(
                        tokenizer,
                        prefix=prefix,
                        exact_drafts=exact_drafts,
                        plan=lazy_plan,
                        config=config,
                    )
                    proposal_us += (time.perf_counter_ns() - t_gate) / 1000.0
                if should_build:
                    precheck_exact_drafts = [] if rescue_active else exact_drafts
                    (
                        precheck_passed,
                        precheck_reason,
                        precheck_apply_us,
                        precheck_tokenize_us,
                    ) = _mv_candidate_precheck(
                        tokenizer,
                        prefix=prefix,
                        exact_drafts=precheck_exact_drafts,
                        plan=lazy_plan,
                        config=config,
                    )
                    rewrite_apply_us += precheck_apply_us
                    tokenize_us += precheck_tokenize_us
                    if precheck_passed:
                        (
                            views,
                            build_apply_us,
                            build_tokenize_us,
                            index_build_us,
                        ) = _build_mv_views_from_plan(
                            tokenizer,
                            plan=lazy_plan,
                            config=config,
                        )
                        rewrite_apply_us += build_apply_us
                        tokenize_us += build_tokenize_us
                        view_stats = {view.view_id: _MVViewStats() for view in views}
                        proposal_route_reason = "trans_view_built_" + gate_reason
                    else:
                        proposal_route_reason = precheck_reason
                else:
                    proposal_route_reason = gate_reason
            if views:
                if (
                    config.generated_reindex_interval > 0
                    and lazy_plan is not None
                    and len(prefix) - generated_reindex_len >= config.generated_reindex_interval
                ):
                    (
                        generated_reindex_views,
                        gen_apply_us,
                        gen_tok_us,
                        gen_index_us,
                    ) = _build_generated_reindex_views(
                        tokenizer,
                        prefix=prefix,
                        prompt_len=prompt_len,
                        plan=lazy_plan,
                        config=config,
                    )
                    generated_reindex_len = len(prefix)
                    rewrite_apply_us += gen_apply_us
                    tokenize_us += gen_tok_us
                    index_build_us += gen_index_us
                    for view in generated_reindex_views:
                        view_stats.setdefault(view.view_id, _MVViewStats(adopted=True))
                t_trans = time.perf_counter_ns()
                candidates: list[_MVCandidate] = []
                all_views = [*views, *generated_reindex_views]
                cursor_cand = _mv_cursor_candidate(
                    cursor_state,
                    all_views,
                    view_stats,
                    config,
                )
                if cursor_cand is not None:
                    candidates.append(cursor_cand)
                for view in all_views:
                    stats = view_stats.setdefault(view.view_id, _MVViewStats())
                    cand = _mv_lookup(
                        prefix,
                        view,
                        stats,
                        config,
                        tokenizer=tokenizer,
                        pair_stats=pair_stats if config.use_pair_priors else None,
                    )
                    if cand is None:
                        continue
                    effective_margin = 0 if rescue_active else _mv_effective_margin(tokenizer, cand, config)
                    margin_exact_drafts = [] if rescue_active else exact_drafts
                    common_for_branch = _common_prefix_len(exact_drafts, cand.tokens) if exact_drafts else 0
                    branch_min = (
                        0
                        if (config.use_frontier_branch or config.use_packed_branch)
                        else config.branch_common_prefix_min
                    )
                    conflict_branch_eligible = _mv_conflict_branch_eligible(
                        exact_match_len=exact_match_len,
                        exact_drafts=exact_drafts,
                        candidate=cand,
                        config=config,
                    )
                    branch_eligible = (
                        not cand.from_cursor
                        and conflict_branch_eligible
                        and common_for_branch >= branch_min
                        and common_for_branch < len(cand.tokens)
                        and common_for_branch < len(exact_drafts)
                    )
                    frontier_probe_override = _mv_allows_frontier_probe_override(
                        tokenizer,
                        candidate=cand,
                        exact_drafts=margin_exact_drafts,
                        config=config,
                    )
                    patch_segment_eligible = (
                        config.use_segment_patch
                        and (cand.crosses_frontier or cand.frontier_distance is not None)
                        and cand.match_len >= config.transformed_min_matching_ngram_size
                    )
                    if patch_segment_eligible and len(cand.tokens) > config.patch_segment_tokens:
                        cand.tokens = cand.tokens[: config.patch_segment_tokens]
                    if (
                        len(cand.tokens) < len(margin_exact_drafts) + effective_margin
                        and not cand.from_cursor
                        and not branch_eligible
                        and not frontier_probe_override
                        and not conflict_branch_eligible
                        and not patch_segment_eligible
                    ):
                        continue
                    cand.branch_eligible = branch_eligible
                    cand.frontier_probe_override = frontier_probe_override
                    # Cold views may probe only at rewrite frontiers. Adopted
                    # views can draft elsewhere, but still need a long match.
                    if not stats.adopted and cand.frontier_distance is None:
                        continue
                    candidates.append(cand)
                proposal_us += (time.perf_counter_ns() - t_trans) / 1000.0
                if not candidates and proposal_route_reason.startswith("trans_view_built_"):
                    proposal_route_reason = "trans_view_built_no_candidate"
                if candidates:
                    chosen_mv = max(
                        candidates,
                        key=lambda c: (
                            c.score,
                            c.crosses_frontier,
                            c.match_len,
                            len(c.tokens),
                        ),
                    )
                    stats = view_stats.setdefault(chosen_mv.view.view_id, _MVViewStats())
                    common = _common_prefix_len(exact_drafts, chosen_mv.tokens) if exact_drafts else 0
                    use_common_branch = (
                        chosen_mv.branch_eligible
                        and (
                            common
                            >= (
                                0
                                if (config.use_frontier_branch or config.use_packed_branch)
                                else config.branch_common_prefix_min
                            )
                        )
                        and common < len(chosen_mv.tokens)
                        and common < len(exact_drafts)
                        and (chosen_mv.crosses_frontier or stats.adopted or chosen_mv.branch_eligible)
                    )
                    if use_common_branch:
                        drafts = chosen_mv.tokens[:common]
                        proposal_kind = (
                            "vantage_mv_branch_packed"
                            if config.use_packed_branch
                            else (
                                "vantage_mv_branch_tree"
                                if config.use_frontier_branch
                                else "vantage_mv_branch_common"
                            )
                        )
                        proposal_match_kind = (
                            "mv_packed_branch"
                            if config.use_packed_branch
                            else (
                                "mv_frontier_branch"
                                if config.use_frontier_branch
                                else "mv_branch_common_prefix"
                            )
                        )
                        proposal_route = (
                            "branch_packed"
                            if config.use_packed_branch
                            else (
                                "branch_tree"
                                if config.use_frontier_branch
                                else "branch_common_prefix"
                            )
                        )
                        proposal_route_reason = (
                            "trans_packed_branch_wins"
                            if config.use_packed_branch
                            else (
                                "trans_conflict_branch_wins"
                                if config.use_frontier_branch
                                else "exact_trans_shared_prefix"
                            )
                        )
                        branch_candidates = 2
                        branch_depth = common
                        if config.use_frontier_branch or config.use_packed_branch:
                            branch_exact_drafts = list(exact_drafts)
                            branch_transformed_drafts = list(chosen_mv.tokens)
                            branch_common_len = common
                        else:
                            chosen_mv = None
                    else:
                        drafts = list(chosen_mv.tokens)
                        proposal_kind = "vantage_mv_pld"
                        proposal_match_kind = "mv_transformed"
                        proposal_route = "transpld"
                        proposal_route_reason = (
                            "trans_cursor_wins"
                            if chosen_mv.from_cursor
                            else (
                                "trans_conflict_branch_wins"
                                if (
                                    config.use_frontier_branch
                                    and exact_match_len < config.transformed_min_matching_ngram_size
                                    and chosen_mv.match_len >= config.transformed_min_matching_ngram_size
                                    and (
                                        chosen_mv.crosses_frontier
                                        or chosen_mv.frontier_distance == 0
                                    )
                                )
                                else (
                                    "pld_rejection_rescue_wins"
                                    if rescue_active
                                    else (
                                        "patch_segment_wins"
                                        if config.use_segment_patch
                                        and (
                                            chosen_mv.crosses_frontier
                                            or chosen_mv.frontier_distance is not None
                                        )
                                        else (
                                    "trans_frontier_probe_wins"
                                    if chosen_mv.frontier_probe_override
                                    else "trans_candidate_wins"
                                        )
                                    )
                                )
                            )
                        )
                        if proposal_route_reason == "trans_conflict_branch_wins":
                            branch_candidates = 2
                            branch_depth = common
                    if proposal_kind in {"vantage_mv_pld", "vantage_mv_branch_common", "vantage_mv_branch_tree", "vantage_mv_branch_packed"}:
                        proposal_pool = chosen_mv.view.source_label if chosen_mv else "multiview"
                        if chosen_mv is not None:
                            proposal_source_start = chosen_mv.source_start
                            proposal_follow_start = chosen_mv.follow_start
                            proposal_match_len = chosen_mv.match_len
                            proposal_view_id = chosen_mv.view.view_id
                            frontier_distance = chosen_mv.frontier_distance
                            compound_view_count = len(views)
                            active_map_count = len(rewrite_map)
                            attempts = stats.attempts
                            route_window_accept_rate = (
                                stats.accepted / attempts if attempts else 0.0
                            )
                            rewrite_zero_accept_streak = sum(1 for lo, hi in stats.blacklist if hi >= lo)
                            if chosen_mv.from_cursor:
                                proposal_cursor_pos = chosen_mv.follow_start
                                proposal_cursor_confidence = cursor_state.confidence

        if (
            config.use_edit_neural_drafter
            and assistant is not None
            and chosen_mv is None
            and proposal_kind != "vantage_mv_branch_packed"
        ):
            should_neural = len(exact_drafts) < config.edit_draft_min_exact_len
            if not should_neural and lazy_plan is not None:
                should_neural = _mv_rewrite_frontier_signal_from_tokens(
                    tokenizer,
                    tokens=[*prefix[-config.frontier_window :], *exact_drafts[: config.frontier_window]],
                    plan=lazy_plan,
                )
            if should_neural:
                t_neural = time.perf_counter_ns()
                neural_stats = _assistant_plain_draft(
                    assistant=assistant,
                    prefix=prefix,
                    assistant_cache=assistant_cache,
                    assistant_cache_len=assistant_cache_len,
                    max_tokens=config.edit_draft_tokens,
                    eos_token_ids=eos_token_ids,
                    confidence_threshold=0.35,
                )
                neural_draft_us = (time.perf_counter_ns() - t_neural) / 1000.0
                assistant_cache = neural_stats.assistant_cache
                assistant_cache_len = neural_stats.assistant_cache_len
                assistant_cache_catchup_tokens = neural_stats.catchup_tokens
                if neural_stats.drafts:
                    drafts = list(neural_stats.drafts)
                    neural_draft_tokens = len(drafts)
                    proposal_kind = "vantage_edit_neural_draft"
                    proposal_match_kind = "edit_neural"
                    proposal_pool = "assistant_edit_frontier"
                    proposal_route = "edit_neural"
                    proposal_route_reason = "edit_neural_frontier_or_pld_weak"
                    proposal_match_len = None
                    proposal_source_start = None
                    proposal_follow_start = None
                    proposal_us += neural_draft_us

        remaining_budget = (prompt_len + max_new_tokens) - old_prefix_len
        drafts = drafts[:remaining_budget]
        staged_chunks = None
        staged_tokens = None
        staged_saved = None
        if (
            proposal_kind in {"vantage_mv_branch_tree", "vantage_mv_branch_packed"}
            and branch_exact_drafts is not None
            and branch_transformed_drafts is not None
            and branch_common_len >= 0
        ):
            branch_fn = (
                _target_packed_branch_step
                if proposal_kind == "vantage_mv_branch_packed"
                else _target_frontier_branch_step
            )
            kwargs = dict(
                prefix=prefix,
                prompt_len=prompt_len,
                target=target,
                target_cache=target_cache,
                target_cache_len=target_cache_len,
                exact_drafts=branch_exact_drafts,
                transformed_drafts=branch_transformed_drafts,
                max_new_tokens=max_new_tokens,
                eos_token_ids=eos_token_ids,
            )
            if branch_fn is _target_frontier_branch_step:
                kwargs["common_len"] = branch_common_len
            (
                target_cache,
                target_cache_len,
                result,
                n_emitted,
                accepted_capped,
                verify_us,
                branch_selected,
                branch_accepted_tokens,
            ) = branch_fn(**kwargs)
        else:
            if config.use_staged_verification:
                (
                    target_cache,
                    target_cache_len,
                    result,
                    n_emitted,
                    accepted_capped,
                    verify_us,
                    staged_chunks,
                    staged_tokens,
                ) = _target_verify_step_staged(
                    prefix=prefix,
                    prompt_len=prompt_len,
                    target=target,
                    target_cache=target_cache,
                    target_cache_len=target_cache_len,
                    drafts=drafts,
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=eos_token_ids,
                    first_tokens=config.staged_first_tokens,
                    second_tokens=config.staged_second_tokens,
                )
                staged_saved = max(0, len(drafts) - int(staged_tokens or 0))
            else:
                (
                    target_cache,
                    target_cache_len,
                    result,
                    n_emitted,
                    accepted_capped,
                    verify_us,
                ) = _target_verify_step(
                    prefix=prefix,
                    prompt_len=prompt_len,
                    target=target,
                    target_cache=target_cache,
                    target_cache_len=target_cache_len,
                    drafts=drafts,
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=eos_token_ids,
                )

        if assistant is not None and config.use_edit_neural_drafter:
            keep_len = old_prefix_len + result.n_accepted_drafts
            crop_dynamic_cache(assistant_cache, keep_len)
            assistant_cache_len = keep_len

        if (
            config.use_pld_rejection_rescue
            and exact_is_strong
            and chosen_mv is None
            and int(result.n_accepted_drafts) == 0
            and accepted_capped
        ):
            if not plan_checked:
                (
                    lazy_plan,
                    rewrite_map,
                    map_source,
                    extra_map_parse_us,
                ) = _build_mv_lazy_plan(
                    tokenizer,
                    prompt_text=prompt_text,
                    reference=reference,
                    metadata=metadata,
                    config=config,
                )
                map_parse_us += extra_map_parse_us
                plan_checked = True
            if _mv_rewrite_frontier_signal_from_tokens(
                tokenizer,
                tokens=[*accepted_capped[:1], *exact_drafts[: config.frontier_window]],
                plan=lazy_plan,
            ):
                rescue_steps_remaining = max(
                    rescue_steps_remaining,
                    int(config.rescue_window_steps),
                )
                proposal_adoption_transition = "pld_rejection_rescue_enabled"

        if chosen_mv is not None and (
            proposal_kind not in {"vantage_mv_branch_tree", "vantage_mv_branch_packed"}
            or branch_selected == 1
        ):
            branch_method = proposal_kind in {
                "vantage_mv_branch_tree",
                "vantage_mv_branch_packed",
            }
            stats = view_stats.setdefault(chosen_mv.view.view_id, _MVViewStats())
            accepted = max(
                0,
                int(
                    branch_accepted_tokens
                    if branch_method and branch_selected == 1 and branch_accepted_tokens is not None
                    else result.n_accepted_drafts
                ),
            )
            proposed = max(
                1,
                len(branch_transformed_drafts or [])
                if branch_method
                else len(drafts),
            )
            stats.attempts += 1
            stats.accepted += accepted
            stats.proposed += proposed
            if accepted > 0:
                stats.adopted = True
            if accepted == 0:
                lo = max(0, chosen_mv.source_start - 64)
                hi = chosen_mv.source_start + 64
                stats.blacklist.append((lo, hi))
            rate = stats.accepted / max(1, stats.proposed)
            if (
                stats.attempts >= config.low_accept_disable_attempts
                and rate < config.low_accept_disable_rate
            ):
                stats.disabled = True
            if config.use_pair_priors:
                pair_touch_tokens = branch_transformed_drafts if branch_method else drafts
                pair_keys = chosen_mv.pair_keys or _pairs_touched_by_text(
                    _decode_token_slice(tokenizer, pair_touch_tokens or []),
                    chosen_mv.view.rewrite_map,
                )
                for key in pair_keys:
                    pair_stat = pair_stats.setdefault(key, _MVPairStats())
                    pair_stat.attempts += 1
                    pair_stat.accepted += accepted
                    pair_stat.proposed += proposed
                    if accepted > 0:
                        pair_stat.adopted = True
                        pair_stat.zero_accepts = 0
                    else:
                        pair_stat.zero_accepts += 1
                    pair_rate = pair_stat.accepted / max(1, pair_stat.proposed)
                    if (
                        pair_stat.attempts >= config.low_accept_disable_attempts
                        and pair_rate < config.low_accept_disable_rate
                    ):
                        pair_stat.disabled = True
            route_window_accept_rate = stats.accepted / max(1, stats.attempts)
            rewrite_zero_accept_streak = len(stats.blacklist)

            if config.use_stateful_cursor:
                if accepted >= config.cursor_min_accept or (
                    chosen_mv.from_cursor and accepted > 0
                ):
                    cursor_state.view_id = chosen_mv.view.view_id
                    cursor_state.pos = chosen_mv.follow_start + accepted
                    cursor_state.confidence = stats.accepted / max(1, stats.proposed)
                    cursor_state.active = True
                    proposal_cursor_resync = not chosen_mv.from_cursor
                    proposal_cursor_pos = cursor_state.pos
                    proposal_cursor_confidence = cursor_state.confidence
                elif accepted == 0 and chosen_mv.from_cursor:
                    cursor_state.reset()
                    proposal_cursor_resync = False
        elif config.use_stateful_cursor and cursor_state.active and accepted_capped:
            view = next(
                (
                    v
                    for v in [*(views or []), *generated_reindex_views]
                    if v.view_id == cursor_state.view_id
                ),
                None,
            )
            stream = (
                view.value_tokens
                if view is not None and view.transducer and view.value_tokens
                else (view.tokens if view is not None else [])
            )
            emitted = list(accepted_capped)
            if stream[cursor_state.pos : cursor_state.pos + len(emitted)] == emitted:
                cursor_state.pos += len(emitted)
                cursor_state.confidence = min(1.0, cursor_state.confidence + 0.02)
                proposal_cursor_pos = cursor_state.pos
                proposal_cursor_confidence = cursor_state.confidence
            else:
                cursor_state.reset()
                proposal_cursor_resync = False
        elif config.use_stateful_cursor and proposal_kind == "vantage_mv_branch_tree":
            cursor_state.reset()

        if rescue_active and rescue_steps_remaining > 0:
            rescue_steps_remaining -= 1

        t_step_end = time.perf_counter_ns()
        hit_max_new_tokens = len(prefix) >= prompt_len + max_new_tokens and not any(
            t in eos_token_ids for t in accepted_capped
        )
        record_draft_tokens = (
            len(branch_transformed_drafts or [])
            if proposal_kind in {"vantage_mv_branch_tree", "vantage_mv_branch_packed"}
            else len(drafts)
        )
        steps.append(
            StepRecord(
                method=method_name,
                step=step_idx,
                k=record_draft_tokens,
                n_accepted_drafts=result.n_accepted_drafts,
                n_emitted=n_emitted,
                rejected=result.rejected,
                node_type=None,
                deepest_type=None,
                wall_us=(t_step_end - t_step_start) / 1000.0,
                verify_us=verify_us,
                proposal_kind=proposal_kind if (drafts or proposal_kind in {"vantage_mv_branch_tree", "vantage_mv_branch_packed"}) else None,
                proposal_match_len=proposal_match_len or None,
                proposal_us=proposal_us,
                proposal_tokens=record_draft_tokens,
                n_guaranteed_drafts=0,
                n_accepted_nonroot_drafts=result.n_accepted_drafts,
                hit_max_new_tokens=hit_max_new_tokens,
                prompt_len=prompt_len,
                proposal_source_start_token=(
                    proposal_source_start if proposal_source_start is not None and proposal_source_start >= 0 else None
                ),
                proposal_follow_start_token=(
                    proposal_follow_start if proposal_follow_start is not None and proposal_follow_start >= 0 else None
                ),
                proposal_query_len=proposal_match_len or None,
                proposal_pool=proposal_pool,
                proposal_source_region=proposal_pool,
                proposal_root_included=False,
                proposal_match_kind=proposal_match_kind,
                proposal_alpha_exact_filtered=exact_capped,
                proposal_tree_candidates=branch_candidates,
                proposal_tree_branch_depth=branch_depth,
                proposal_tree_branch_selected=branch_selected,
                blazedit_micro_draft_tokens=config.max_draft_tokens,
                blazedit_max_num_run=1,
                blazedit_pld_proposed=len(exact_drafts),
                target_draft_tokens=record_draft_tokens,
                target_accepted_nonroot=result.n_accepted_drafts,
                assistant_model=assistant_model_name if proposal_kind == "vantage_edit_neural_draft" else None,
                assistant_us=neural_draft_us,
                assistant_verify_us=neural_draft_us,
                assistant_cache_catchup_tokens=assistant_cache_catchup_tokens,
                verify_staged_chunks=staged_chunks,
                verify_staged_draft_tokens=staged_tokens,
                verify_staged_saved_tokens=staged_saved,
                proposal_neural_draft_tokens=neural_draft_tokens,
                proposal_neural_draft_us=neural_draft_us,
                proposal_map_source=map_source if rewrite_map else None,
                proposal_view_id=proposal_view_id,
                proposal_compound_view_count=compound_view_count,
                proposal_active_map_count=active_map_count,
                proposal_route=proposal_route,
                proposal_route_reason=proposal_route_reason,
                proposal_backoff_active=False,
                proposal_rewrite_hit_count=(
                    sum(s.attempts for s in view_stats.values()) if view_stats else 0
                ),
                proposal_route_window_accept_rate=route_window_accept_rate,
                proposal_rewrite_zero_accept_streak=rewrite_zero_accept_streak,
                proposal_adoption_state=(
                    "adopted"
                    if chosen_mv is not None
                    and view_stats.get(chosen_mv.view.view_id, _MVViewStats()).adopted
                    else ("cold" if rewrite_map else None)
                ),
                proposal_adoption_transition=proposal_adoption_transition,
                proposal_frontier_distance=frontier_distance,
                proposal_cursor_pos=proposal_cursor_pos,
                proposal_cursor_confidence=proposal_cursor_confidence,
                proposal_cursor_resync=proposal_cursor_resync,
                proposal_map_parse_us=map_parse_us,
                proposal_rewrite_apply_us=rewrite_apply_us,
                proposal_virtual_reference_tokenize_us=tokenize_us,
                proposal_transpld_index_build_us=index_build_us,
            )
        )
        step_idx += 1

        if any(t in eos_token_ids for t in accepted_capped):
            break

    return DecodeResult(
        output_token_ids=prefix,
        output_text="",
        n_new_tokens=len(prefix) - prompt_len,
        steps=steps,
        wall_us_total=(time.perf_counter_ns() - t_start) / 1000.0,
    )
