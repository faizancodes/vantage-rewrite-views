from asts.blazedit_decoder import (
    _build_mv_lazy_plan,
    _build_generated_reindex_views,
    _build_mv_views,
    _MVCandidate,
    _MVView,
    _mv_cap_exact_drafts,
    _mv_conflict_branch_eligible,
    _mv_exact_is_strong,
    _mv_draft_cap,
    _canonicalize_tokens_for_lookup,
    _filter_rewrite_pairs_for_quality,
    _delta_cache_note_failure,
    _delta_cache_patch_draft,
    _edit_distance_leq1,
    _mv_allows_frontier_probe_override,
    _mv_candidate_precheck,
    _mv_frontier_gate,
    _mv_lookup,
    _mv_cursor_candidate,
    _MVViewStats,
    _MVCursorState,
    is_vantage_mv_method,
    is_blazedit_method,
    parse_vantage_mv_method,
    parse_blazedit_method,
    fuzzy_resync_draft,
    prompt_lookup_draft,
)
from collections import OrderedDict
from scripts.run_eagle_eval import (
    _routed_transpld_exact_route_reason,
    _routed_transpld_pld_config,
)


class _CharTokenizer:
    def __call__(self, text, add_special_tokens=False):
        class Enc:
            input_ids = [ord(ch) for ch in text]

        return Enc()

    def decode(self, tokens, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(t) for t in tokens)


def test_prompt_lookup_chooses_longest_match():
    tokens = [1, 2, 3, 4, 1, 2, 3]

    draft, match_len, source_start, follow_start = prompt_lookup_draft(
        tokens,
        max_matching_ngram_size=3,
        max_draft_tokens=4,
    )

    assert draft == [4]
    assert match_len == 3
    assert source_start == 0
    assert follow_start == 3


def test_prompt_lookup_chooses_most_recent_match():
    tokens = [9, 1, 2, 5, 1, 2, 6, 1, 2]

    draft, match_len, source_start, follow_start = prompt_lookup_draft(
        tokens,
        max_matching_ngram_size=2,
        max_draft_tokens=4,
    )

    assert draft == [6]
    assert match_len == 2
    assert source_start == 4
    assert follow_start == 6


def test_prompt_lookup_enforces_non_overlap():
    tokens = [1, 2, 1, 2]

    draft, match_len, source_start, follow_start = prompt_lookup_draft(
        tokens,
        max_matching_ngram_size=2,
        min_matching_ngram_size=2,
        max_draft_tokens=4,
    )

    assert draft == []
    assert match_len == 0
    assert source_start == -1
    assert follow_start == -1


def test_prompt_lookup_respects_draft_cap():
    tokens = [1, 2, 3, 4, 5, 1, 2]

    draft, match_len, _, _ = prompt_lookup_draft(
        tokens,
        max_matching_ngram_size=2,
        max_draft_tokens=2,
    )

    assert draft == [3, 4]
    assert match_len == 2


def test_parse_blazedit_methods():
    pld = parse_blazedit_method("blazedit_pld_w40_n10")
    assert pld.mode == "pld"
    assert pld.micro_draft_tokens == 40
    assert pld.max_matching_ngram_size == 10
    assert pld.min_matching_ngram_size == 1

    pld_m4 = parse_blazedit_method("blazedit_pld_m4_w128_n10")
    assert pld_m4.mode == "pld"
    assert pld_m4.micro_draft_tokens == 128
    assert pld_m4.max_matching_ngram_size == 10
    assert pld_m4.min_matching_ngram_size == 4
    assert is_blazedit_method("blazedit_pld_m4_w128_n10")

    force_pld = parse_blazedit_method("vantage_force_pld_w128_n10")
    assert force_pld.mode == "pld"
    assert force_pld.micro_draft_tokens == 128
    assert force_pld.max_matching_ngram_size == 10

    staged = parse_blazedit_method("vantage_staged_pld_v16_32_w128_n10")
    assert staged.mode == "pld"
    assert staged.micro_draft_tokens == 128
    assert staged.max_matching_ngram_size == 10
    assert staged.use_staged_verification
    assert staged.staged_first_tokens == 16
    assert staged.staged_second_tokens == 32

    delta = parse_blazedit_method("delta_cache_pld_p1_c4_lru32_pw64_w128_n10")
    assert delta.mode == "delta_cache_pld"
    assert delta.micro_draft_tokens == 128
    assert delta.max_matching_ngram_size == 10
    assert delta.delta_max_patches == 1
    assert delta.delta_context_tokens == 4
    assert delta.delta_lru_size == 32
    assert delta.delta_patch_window == 64

    fuzzy = parse_blazedit_method("fuzzy_resync_pld_fd32_weak8_w128_n10")
    assert fuzzy.mode == "fuzzy_resync_pld"
    assert fuzzy.micro_draft_tokens == 128
    assert fuzzy.max_matching_ngram_size == 10
    assert fuzzy.fuzzy_max_draft_tokens == 32
    assert fuzzy.fuzzy_weak_draft_len == 8

    static = parse_blazedit_method("blazedit_assisted_static_w40")
    assert static.mode == "assisted_static"
    assert static.micro_draft_tokens == 40

    dynamic = parse_blazedit_method(
        "blazedit_assisted_dynamic_w40",
        confidence_threshold=0.45,
    )
    assert dynamic.mode == "assisted_dynamic"
    assert dynamic.assistant_confidence_threshold == 0.45

    two_layer = parse_blazedit_method("blazedit_two_layer_m20_r4_n10")
    assert two_layer.mode == "two_layer"
    assert two_layer.micro_draft_tokens == 20
    assert two_layer.max_num_run == 4
    assert two_layer.max_matching_ngram_size == 10

    mtp = parse_blazedit_method("pld_plus_mtp_heads")
    assert mtp.mode == "pld_plus_mtp_heads"
    assert mtp.micro_draft_tokens == 128
    assert mtp.max_matching_ngram_size == 10
    assert mtp.mtp_num_heads == 4
    assert mtp.mtp_trigger_accepted_len == 4
    assert mtp.mtp_position == "post_pld"

    queued_mtp = parse_blazedit_method("pld_queued_mtp_heads")
    assert queued_mtp.mode == "pld_queued_mtp_heads"
    assert queued_mtp.micro_draft_tokens == 128
    assert queued_mtp.max_matching_ngram_size == 10
    assert queued_mtp.mtp_queue_enabled
    assert queued_mtp.mtp_use_queued_only_on_weak_pld
    assert queued_mtp.mtp_disable_extra_verify

    capped = parse_blazedit_method("weak_router_capped_pld_t30_cap8_w128_n10")
    assert capped.mode == "weak_router_capped_pld"
    assert capped.micro_draft_tokens == 128
    assert capped.max_matching_ngram_size == 10
    assert capped.weak_pld_router_threshold == 0.3
    assert capped.weak_pld_cap_tokens == 8

    lookahead = parse_blazedit_method("lookahead_w8_n4_i4")
    assert lookahead.mode == "lookahead"
    assert lookahead.lookahead_window == 8
    assert lookahead.lookahead_ngram == 4
    assert lookahead.lookahead_iters == 4

    gated_lookahead = parse_blazedit_method("pld_gated_lookahead_w128_n10")
    assert gated_lookahead.mode == "pld_gated_lookahead"
    assert gated_lookahead.micro_draft_tokens == 128
    assert gated_lookahead.max_matching_ngram_size == 10

    gated_i1 = parse_blazedit_method("pld_gated_lookahead_w8_n4_i1_d4")
    assert gated_i1.mode == "pld_gated_lookahead"
    assert gated_i1.micro_draft_tokens == 128
    assert gated_i1.max_matching_ngram_size == 10
    assert gated_i1.lookahead_window == 8
    assert gated_i1.lookahead_ngram == 4
    assert gated_i1.lookahead_iters == 1
    assert gated_i1.lookahead_max_draft == 4
    assert gated_i1.lookahead_one_forward


def test_is_blazedit_method():
    assert is_blazedit_method("blazedit_pld_w40_n10")
    assert is_blazedit_method("vantage_force_pld_w128_n10")
    assert is_blazedit_method("blazedit_pld_staged_v16_32_w128_n10")
    assert is_blazedit_method("delta_cache_pld_w128_n10")
    assert is_blazedit_method("delta_cache_pld_p1_c4_lru64_pw64_w128_n10")
    assert is_blazedit_method("fuzzy_resync_pld_w128_n10")
    assert is_blazedit_method("fuzzy_resync_pld_fd16_weak4_w128_n10")
    assert is_blazedit_method("blazedit_assisted_static_w40")
    assert is_blazedit_method("blazedit_assisted_dynamic_w40")
    assert is_blazedit_method("blazedit_two_layer_m40_r10_n10")
    assert is_blazedit_method("pld_plus_mtp_heads")
    assert is_blazedit_method("pld_queued_mtp_heads")
    assert is_blazedit_method("weak_router_capped_pld_t50_cap4_w128_n10")
    assert is_blazedit_method("weak_router_capped_pld_w128_n10")
    assert is_blazedit_method("lookahead_w8_n4_i4")
    assert is_blazedit_method("pld_gated_lookahead_w128_n10")
    assert is_blazedit_method("pld_gated_lookahead_w8_n4_i1_d4")
    assert not is_blazedit_method("vantage_edit_anchor_tail")


def test_delta_cache_patches_context_keyed_future_draft():
    cache = OrderedDict()
    prefix = [10, 11, 12, 13]
    _delta_cache_note_failure(
        cache,
        context_key=(tuple(prefix[-4:]), 100),
        new_token=200,
        max_entries=4,
    )

    patched, patches = _delta_cache_patch_draft(
        prefix=prefix,
        drafts=[100, 101, 102],
        cache=cache,
        context_tokens=4,
        max_patches=1,
        patch_window=64,
    )

    assert patched == [200, 101, 102]
    assert len(patches) == 1
    assert patches[0][1:3] == (100, 200)


def test_fuzzy_resync_finds_unique_edit_distance_one_match():
    source = [1, 2, 3, 4, 99, 6, 7, 8, 9, 10, 50, 51, 52]
    filler = [70, 71, 72]
    query = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    tokens = source + filler + query

    draft, match_len, source_start, follow_start, n_candidates, dist = fuzzy_resync_draft(
        tokens,
        query_len=10,
        max_draft_tokens=3,
        require_unique=True,
    )

    assert draft == [50, 51, 52]
    assert match_len == 10
    assert source_start == 0
    assert follow_start == 10
    assert n_candidates == 1
    assert dist == 1
    assert _edit_distance_leq1(query, source[:10]) == (True, 1)


def test_parse_vantage_mv_method_defaults_and_knobs():
    default = parse_vantage_mv_method("vantage_mv_pld_w128_n10")
    assert default.max_draft_tokens == 128
    assert default.max_matching_ngram_size == 10
    assert default.exact_strong_min_len == 64
    assert default.trans_len_margin == 16
    assert default.transformed_min_matching_ngram_size == 8

    tuned = parse_vantage_mv_method("vantage_mv_pld_s96_m32_t10_w80_n8")
    assert tuned.exact_strong_min_len == 96
    assert tuned.trans_len_margin == 32
    assert tuned.transformed_min_matching_ngram_size == 10
    assert tuned.max_draft_tokens == 80
    assert tuned.max_matching_ngram_size == 8

    guard = parse_vantage_mv_method("vantage_mv_pld_s96_x8_c116_c764_m32_t10_w80_n8")
    assert guard.exact_strong_min_len == 96
    assert guard.exact_strong_min_match == 8
    assert guard.exact_match1_draft_cap == 16
    assert guard.exact_match2_7_draft_cap == 64

    stable = parse_vantage_mv_method("vantage_mv_pld_s96_x1_m16_t8_w128_n10")
    assert stable.exact_strong_min_len == 96
    assert stable.exact_strong_min_match == 1
    assert stable.trans_len_margin == 16
    assert stable.transformed_min_matching_ngram_size == 8
    assert stable.max_draft_tokens == 128

    tree_cursor = parse_vantage_mv_method(
        "vantage_mv_pld_fst_q_pair_tree_cursor_hunk_s96_x1_m16_t8_g32_w128_n10"
    )
    assert tree_cursor.use_rewrite_fst
    assert tree_cursor.use_map_quality_gate
    assert tree_cursor.use_pair_priors
    assert tree_cursor.use_frontier_branch
    assert tree_cursor.use_stateful_cursor
    assert tree_cursor.use_hunk_alignment
    assert tree_cursor.generated_reindex_interval == 32

    rescue_patch = parse_vantage_mv_method(
        "vantage_mv_pld_rescue_patch_tree_s96_x1_m16_t8_w128_n10"
    )
    assert rescue_patch.use_pld_rejection_rescue
    assert rescue_patch.use_segment_patch
    assert rescue_patch.use_frontier_branch

    frontier20 = parse_vantage_mv_method(
        "vantage_mv_pld_stage_pbranch_edraft_patch_s96_x1_m16_t8_w128_n10"
    )
    assert frontier20.use_staged_verification
    assert frontier20.use_packed_branch
    assert frontier20.use_edit_neural_drafter
    assert frontier20.use_segment_patch

    all_legacy = parse_vantage_mv_method("vantage_mv_pld_all_s64_m8_f8_t8_g64_w128_n10")
    assert all_legacy.use_rewrite_fst
    assert all_legacy.use_frontier_branch
    assert all_legacy.use_pair_priors
    assert all_legacy.use_stateful_cursor
    assert all_legacy.use_segment_patch
    assert not all_legacy.use_staged_verification
    assert not all_legacy.use_packed_branch
    assert not all_legacy.use_edit_neural_drafter

    assert is_vantage_mv_method("vantage_mv_pld_s64_m16_w128_n10")
    assert not is_vantage_mv_method("vantage_mvpld_s64_m16_w128_n10")


def test_vantage_mv_exact_strong_requires_match_quality():
    cfg = parse_vantage_mv_method("vantage_mv_pld_s64_x8_m16_t8_w128_n10")

    assert not _mv_exact_is_strong([1] * 128, 1, cfg)
    assert _mv_exact_is_strong([1] * 128, 8, cfg)


def test_vantage_mv_exact_draft_caps_by_match_quality():
    cfg = parse_vantage_mv_method("vantage_mv_pld_s64_c116_c764_m16_t8_w128_n10")

    capped, changed, cap = _mv_cap_exact_drafts(list(range(80)), 1, cfg)
    assert changed
    assert cap == 16
    assert capped == list(range(16))

    capped, changed, cap = _mv_cap_exact_drafts(list(range(80)), 4, cfg)
    assert changed
    assert cap == 64
    assert capped == list(range(64))

    capped, changed, cap = _mv_cap_exact_drafts(list(range(80)), 8, cfg)
    assert not changed
    assert cap is None
    assert capped == list(range(80))


def test_vantage_mv_conflict_branch_requires_weak_exact_and_frontier():
    cfg = parse_vantage_mv_method("vantage_mv_pld_branch_s64_m16_t8_w128_n10")
    view = _MVView(
        view_id="test",
        tokens=list(range(100)),
        rewrite_map={"user": "account"},
        source_label="transformed_reference",
        map_source="explicit",
        frontiers=[10],
        index={},
    )
    cand = _MVCandidate(
        tokens=[1] * 32,
        match_len=10,
        source_start=0,
        follow_start=10,
        view=view,
        frontier_distance=0,
        crosses_frontier=True,
        score=32.0,
    )

    assert _mv_conflict_branch_eligible(
        exact_match_len=1,
        exact_drafts=[2] * 128,
        candidate=cand,
        config=cfg,
    )
    assert not _mv_conflict_branch_eligible(
        exact_match_len=8,
        exact_drafts=[2] * 128,
        candidate=cand,
        config=cfg,
    )

    rescue_cfg = parse_vantage_mv_method(
        "vantage_mv_pld_tree_rescue_s64_m16_t8_w128_n10"
    )
    assert _mv_conflict_branch_eligible(
        exact_match_len=8,
        exact_drafts=[2] * 128,
        candidate=cand,
        config=rescue_cfg,
    )


def test_vantage_mv_builds_transformed_view_and_lookup_near_frontier():
    tok = _CharTokenizer()
    cfg = parse_vantage_mv_method("vantage_mv_pld_s64_m4_t4_w32_n8")
    reference = "name = user.name.strip()\\nreturn name\\n"
    views, pairs, source, *_ = _build_mv_views(
        tok,
        prompt_text="Explicit rewrite map: user -> account.",
        reference=reference,
        metadata={},
        config=cfg,
    )

    assert pairs == {"user": "account"}
    assert source == "explicit"
    assert views
    view = views[0]
    assert view.frontiers

    prefix = [ord(ch) for ch in "name = account.name"]
    cand = _mv_lookup(prefix, view, _MVViewStats(), cfg)

    assert cand is not None
    assert cand.match_len >= cfg.transformed_min_matching_ngram_size
    assert cand.frontier_distance is not None


def test_vantage_mv_lazy_gate_skips_unrelated_weak_exact_candidate():
    tok = _CharTokenizer()
    cfg = parse_vantage_mv_method("vantage_mv_pld_s64_m4_t4_w32_n8")
    plan, pairs, source, _ = _build_mv_lazy_plan(
        tok,
        prompt_text="Explicit rewrite map: user -> account.",
        reference="name = user.name.strip()\nreturn name\n",
        metadata={},
    )

    assert plan is not None
    assert pairs == {"user": "account"}
    assert source == "explicit"

    should_build, reason = _mv_frontier_gate(
        tok,
        prefix=[ord(ch) for ch in "return "],
        exact_drafts=[ord(ch) for ch in "name\n"],
        plan=plan,
        config=cfg,
    )

    assert not should_build
    assert reason == "no_rewrite_frontier_signal"


def test_vantage_mv_lazy_gate_builds_when_exact_candidate_copies_old_term():
    tok = _CharTokenizer()
    cfg = parse_vantage_mv_method("vantage_mv_pld_s64_m4_t4_w32_n8")
    plan, *_ = _build_mv_lazy_plan(
        tok,
        prompt_text="Explicit rewrite map: user -> account.",
        reference="name = user.name.strip()\nreturn name\n",
        metadata={},
    )

    assert plan is not None
    should_build, reason = _mv_frontier_gate(
        tok,
        prefix=[ord(ch) for ch in "name = "],
        exact_drafts=[ord(ch) for ch in "user.name.strip()\n"],
        plan=plan,
        config=cfg,
    )

    assert should_build
    assert reason == "exact_candidate_contains_old_term"


def test_vantage_mv_lazy_gate_builds_after_generated_new_term():
    tok = _CharTokenizer()
    cfg = parse_vantage_mv_method("vantage_mv_pld_s64_m4_t4_w32_n8")
    plan, *_ = _build_mv_lazy_plan(
        tok,
        prompt_text="Explicit rewrite map: user -> account.",
        reference="name = user.name.strip()\nreturn name\n",
        metadata={},
    )

    assert plan is not None
    should_build, reason = _mv_frontier_gate(
        tok,
        prefix=[ord(ch) for ch in "name = account"],
        exact_drafts=[],
        plan=plan,
        config=cfg,
    )

    assert should_build
    assert reason == "generated_suffix_contains_new_term"


def test_vantage_mv_lazy_gate_can_probe_weak_exact_without_frontier_signal():
    tok = _CharTokenizer()
    cfg = parse_vantage_mv_method("vantage_mv_pld_s64_m4_f4_t4_w32_n8")
    plan, *_ = _build_mv_lazy_plan(
        tok,
        prompt_text="Explicit rewrite map: user -> account.",
        reference="name = user.name.strip()\nreturn name\n",
        metadata={},
    )

    assert plan is not None
    should_build, reason = _mv_frontier_gate(
        tok,
        prefix=[ord(ch) for ch in "unrelated"],
        exact_drafts=[ord("x")] * 4,
        plan=plan,
        config=cfg,
    )

    assert should_build
    assert reason == "exact_candidate_weak_probe"


def test_vantage_mv_precheck_skips_when_margin_impossible():
    tok = _CharTokenizer()
    cfg = parse_vantage_mv_method("vantage_mv_pld_s64_m16_t4_w32_n8")
    plan, *_ = _build_mv_lazy_plan(
        tok,
        prompt_text="Explicit rewrite map: user -> account.",
        reference="name = user.name.strip()\nreturn name\n",
        metadata={},
    )

    assert plan is not None
    should_build, reason, apply_us, tokenize_us = _mv_candidate_precheck(
        tok,
        prefix=[ord(ch) for ch in "name = profile.name"],
        exact_drafts=[ord("x")] * 40,
        plan=plan,
        config=cfg,
    )

    assert not should_build
    assert reason == "trans_precheck_margin_impossible"
    assert apply_us == 0.0
    assert tokenize_us == 0.0


def test_vantage_mv_precheck_allows_bounded_probe_near_frontier_with_long_exact():
    tok = _CharTokenizer()
    cfg = parse_vantage_mv_method("vantage_mv_pld_s64_m16_t4_w32_n8")
    plan, *_ = _build_mv_lazy_plan(
        tok,
        prompt_text="Explicit rewrite map: user -> account.",
        reference="name = user.name.strip()\nreturn name\n",
        metadata={},
    )

    assert plan is not None
    should_build, reason, apply_us, tokenize_us = _mv_candidate_precheck(
        tok,
        prefix=[ord(ch) for ch in "name = account.name"],
        exact_drafts=[ord("x")] * 40,
        plan=plan,
        config=cfg,
    )

    assert should_build
    assert reason == "trans_precheck_candidate_exists"
    assert apply_us >= 0.0
    assert tokenize_us >= 0.0


def test_vantage_mv_frontier_probe_override_can_beat_long_exact_candidate():
    tok = _CharTokenizer()
    cfg = parse_vantage_mv_method("vantage_mv_pld_s64_m16_t4_w32_n8")
    reference = "name = user.name.strip()\nreturn name\n"
    views, *_ = _build_mv_views(
        tok,
        prompt_text="Explicit rewrite map: user -> account.",
        reference=reference,
        metadata={},
        config=cfg,
    )
    view = views[0]
    cand = _mv_lookup(
        [ord(ch) for ch in "name = account.name"],
        view,
        _MVViewStats(),
        cfg,
    )

    assert cand is not None
    assert len(cand.tokens) < 40
    assert _mv_allows_frontier_probe_override(
        tok,
        candidate=cand,
        exact_drafts=[ord("x")] * 40,
        config=cfg,
    )


def test_vantage_mv_precheck_vocab_margin_bypass_allows_rewrite_tokens():
    tok = _CharTokenizer()
    cfg = parse_vantage_mv_method("vantage_mv_pld_s64_m16_t4_w32_n8")
    plan, *_ = _build_mv_lazy_plan(
        tok,
        prompt_text="Explicit rewrite map: user -> account.",
        reference=("user " * 16) + "\n",
        metadata={},
    )

    assert plan is not None
    should_build, reason, apply_us, tokenize_us = _mv_candidate_precheck(
        tok,
        prefix=[ord(ch) for ch in "account"],
        exact_drafts=[ord("x")] * 20,
        plan=plan,
        config=cfg,
    )

    assert should_build
    assert reason == "trans_precheck_candidate_exists"
    assert apply_us >= 0.0
    assert tokenize_us >= 0.0


def test_vantage_mv_precheck_skips_when_no_token_candidate():
    tok = _CharTokenizer()
    cfg = parse_vantage_mv_method("vantage_mv_pld_s64_m4_t4_w32_n8")
    plan, *_ = _build_mv_lazy_plan(
        tok,
        prompt_text="Explicit rewrite map: user -> account.",
        reference="name = user.name.strip()\nreturn name\n",
        metadata={},
    )

    assert plan is not None
    should_build, reason, apply_us, tokenize_us = _mv_candidate_precheck(
        tok,
        prefix=[ord(ch) for ch in "totally unrelated suffix"],
        exact_drafts=[],
        plan=plan,
        config=cfg,
    )

    assert not should_build
    assert reason == "trans_precheck_no_token_candidate"
    assert apply_us >= 0.0
    assert tokenize_us == 0.0


def test_vantage_mv_precheck_builds_only_when_local_candidate_exists():
    tok = _CharTokenizer()
    cfg = parse_vantage_mv_method("vantage_mv_pld_s64_m4_t4_w32_n8")
    plan, *_ = _build_mv_lazy_plan(
        tok,
        prompt_text="Explicit rewrite map: user -> account.",
        reference="name = user.name.strip()\nreturn name\n",
        metadata={},
    )

    assert plan is not None
    should_build, reason, apply_us, tokenize_us = _mv_candidate_precheck(
        tok,
        prefix=[ord(ch) for ch in "name = account.name"],
        exact_drafts=[],
        plan=plan,
        config=cfg,
    )

    assert should_build
    assert reason == "trans_precheck_candidate_exists"
    assert apply_us >= 0.0
    assert tokenize_us >= 0.0


def test_vantage_mv_builds_compound_subset_views():
    tok = _CharTokenizer()
    cfg = parse_vantage_mv_method("vantage_mv_pld_s64_m4_f4_t4_w32_n8")
    views, pairs, _source, *_ = _build_mv_views(
        tok,
        prompt_text="Rewrite alpha -> beta, gamma -> delta, left -> right.",
        reference="alpha = gamma + left\nreturn alpha\n",
        metadata={"rewrite_pairs": {"alpha": "wrong", "gamma": "wrong", "left": "wrong"}},
        config=cfg,
    )

    assert len(pairs) == 3
    assert any(v.source_label == "compound_subset_reference" for v in views)
    assert any(v.source_label == "single_map_reference" for v in views)


def test_vantage_mv_fst_canonical_lookup_emits_target_side_tokens():
    tok = _CharTokenizer()
    cfg = parse_vantage_mv_method("vantage_mv_pld_fst_s64_m4_t4_w64_n32")
    views, pairs, _source, *_ = _build_mv_views(
        tok,
        prompt_text="Explicit rewrite map: user -> account.",
        reference="name = user.name.strip()\nreturn user.name\n",
        metadata={},
        config=cfg,
    )

    assert pairs == {"user": "account"}
    view = views[0]
    assert view.transducer
    cand = _mv_lookup(
        [ord(ch) for ch in "name = account.name"],
        view,
        _MVViewStats(),
        cfg,
        tokenizer=tok,
    )

    assert cand is not None
    assert "account" in tok.decode(cand.tokens)
    assert "user" not in tok.decode(cand.tokens)


def test_vantage_mv_cursor_candidate_follows_value_stream():
    cfg = parse_vantage_mv_method("vantage_mv_pld_cursor_s64_m4_t4_w32_n8")
    view = _MVView(
        view_id="cursor_view",
        tokens=[1, 2, 3, 4, 5, 6],
        rewrite_map={"user": "account"},
        source_label="transformed_reference",
        map_source="explicit",
        frontiers=[2],
        index={},
    )
    cursor = _MVCursorState(view_id="cursor_view", pos=2, confidence=0.75, active=True)

    cand = _mv_cursor_candidate(cursor, [view], {"cursor_view": _MVViewStats(adopted=True)}, cfg)

    assert cand is not None
    assert cand.from_cursor
    assert cand.tokens == [3, 4, 5, 6]
    assert cand.follow_start == 2


def test_vantage_mv_patch_uses_larger_cold_segment_cap():
    normal = parse_vantage_mv_method("vantage_mv_pld_s64_m4_t4_w128_n8")
    patch = parse_vantage_mv_method("vantage_mv_pld_patch_s64_m4_t4_w128_n8")

    assert _mv_draft_cap(normal, _MVViewStats(), 10) == normal.cold_trans_max_draft
    assert _mv_draft_cap(patch, _MVViewStats(), 10) == patch.patch_segment_tokens


def test_vantage_mv_parser_accepts_frontier_probe_suffix():
    cfg = parse_vantage_mv_method("vantage_mv_pld_s32_m8_f4_t8_g64_w128_n10")

    assert is_vantage_mv_method("vantage_mv_pld_s32_m8_f4_t8_g64_w128_n10")
    assert cfg.exact_strong_min_len == 32
    assert cfg.trans_len_margin == 8
    assert cfg.no_frontier_probe_exact_len == 4
    assert cfg.transformed_min_matching_ngram_size == 8
    assert cfg.generated_reindex_interval == 64


def test_vantage_mv_parser_accepts_frontier_feature_flags():
    cfg = parse_vantage_mv_method(
        "vantage_mv_pld_fst_q_pair_branch_s32_m8_f4_t8_g64_w128_n10"
    )

    assert is_vantage_mv_method(
        "vantage_mv_pld_fst_q_pair_branch_s32_m8_f4_t8_g64_w128_n10"
    )
    assert cfg.use_rewrite_fst
    assert cfg.use_map_quality_gate
    assert cfg.use_pair_priors
    assert cfg.use_frontier_branch
    assert cfg.generated_reindex_interval == 64


def test_vantage_mv_map_quality_gate_filters_common_short_pairs():
    pairs = _filter_rewrite_pairs_for_quality(
        {
            "x": "y",
            "data": "value",
            ".add_ten": ".add_ten_updated",
            "user.name": "account.display_name",
        }
    )

    assert "x" not in pairs
    assert "data" not in pairs
    assert pairs[".add_ten"] == ".add_ten_updated"
    assert pairs["user.name"] == "account.display_name"


def test_vantage_mv_fst_view_matches_new_prefix_and_emits_transformed_span():
    tok = _CharTokenizer()
    cfg = parse_vantage_mv_method("vantage_mv_pld_fst_pair_s64_m0_f4_t4_w32_n8")
    views, pairs, _source, *_ = _build_mv_views(
        tok,
        prompt_text="Rewrite alpha -> beta and gamma -> delta.",
        reference="alpha = gamma\nreturn alpha\n",
        metadata={"rewrite_pairs": {"alpha": "wrong", "gamma": "wrong"}},
        config=cfg,
    )

    assert pairs == {"alpha": "beta", "gamma": "delta"}
    assert any(v.transducer for v in views)
    assert not any(v.source_label == "compound_subset_reference" for v in views)

    fst = next(v for v in views if v.transducer and v.view_id.endswith(":full"))
    assert fst.value_tokens
    assert fst.value_index
    assert fst.value_frontiers
    assert len(fst.source_to_value_start) == len(fst.tokens) + 1
    assert len(fst.source_to_value_end) == len(fst.tokens) + 1
    assert fst.query_replacements
    assert _canonicalize_tokens_for_lookup(
        [ord(ch) for ch in "beta = "],
        fst.query_replacements,
    ) == tuple(ord(ch) for ch in "alpha = ")
    cand = _mv_lookup(
        [ord(ch) for ch in "beta = "],
        fst,
        _MVViewStats(),
        cfg,
        tokenizer=tok,
        pair_stats={},
    )

    assert cand is not None
    assert "".join(chr(t) for t in cand.tokens).startswith("delta")
    assert cand.crosses_frontier


def test_vantage_mv_generated_reindex_builds_adopted_subset_view():
    tok = _CharTokenizer()
    cfg = parse_vantage_mv_method("vantage_mv_pld_fst_g8_s64_m0_f4_t4_w32_n8")
    plan, *_ = _build_mv_lazy_plan(
        tok,
        prompt_text="Rewrite alpha -> beta and gamma -> delta.",
        reference="alpha = gamma\nreturn alpha\n",
        metadata={"rewrite_pairs": {"alpha": "wrong", "gamma": "wrong"}},
        config=cfg,
    )

    assert plan is not None
    views, apply_us, tok_us, index_us = _build_generated_reindex_views(
        tok,
        prefix=[ord(ch) for ch in "beta = gamma\n"],
        prompt_len=0,
        plan=plan,
        config=cfg,
    )

    assert views
    assert views[0].rewrite_map == {"alpha": "beta"}
    assert views[0].source_label == "generated_prefix_reindexed"
    assert apply_us >= 0.0
    assert tok_us >= 0.0
    assert index_us >= 0.0


def test_routed_transpld_exact_route_uses_blazedit_pld_config():
    class Args:
        assistant_model = "Qwen/Qwen2.5-Coder-0.5B"
        blazedit_assistant_confidence_threshold = None
        blazedit_max_matching_ngram_size = 10

    routed = _routed_transpld_pld_config("vantage_routed_transpld_w128_n10", Args())
    baseline = parse_blazedit_method("blazedit_pld_w128_n10")

    assert routed.mode == "pld"
    assert routed.micro_draft_tokens == baseline.micro_draft_tokens == 128
    assert routed.max_matching_ngram_size == baseline.max_matching_ngram_size == 10
    assert routed.max_num_run == baseline.max_num_run == 1

    dispatch = _routed_transpld_pld_config(
        "vantage_dispatch_transpld_m4_w128_n10",
        Args(),
    )
    assert dispatch.mode == "pld"
    assert dispatch.micro_draft_tokens == baseline.micro_draft_tokens
    assert dispatch.max_matching_ngram_size == baseline.max_matching_ngram_size
    assert dispatch.max_num_run == baseline.max_num_run

    frozen = _routed_transpld_pld_config("vantage_frozen_transpld", Args())
    assert frozen.mode == "pld"
    assert frozen.micro_draft_tokens == baseline.micro_draft_tokens
    assert frozen.max_matching_ngram_size == baseline.max_matching_ngram_size
    assert frozen.max_num_run == baseline.max_num_run


def test_routed_transpld_exact_route_reasons_for_no_effect_cases():
    tok = _CharTokenizer()

    class MetadataPoisonedNoMap:
        prompt = "Copy exactly.\n```python\nx = user\n```\n"
        reference = "x = user\n"
        metadata = {
            "target_is_reference": True,
            "rewrite_pairs": {"user": "account"},
            "gold": "x = account\n",
            "label": "synthetic_field",
            "manifest_only_field": "must_not_route",
        }

    assert _routed_transpld_exact_route_reason(MetadataPoisonedNoMap, tok) == "no_rewrite_map"

    class NoMap:
        prompt = "Copy exactly.\n```python\nx = user\n```\n"
        reference = "x = user\n"
        metadata = {}

    assert _routed_transpld_exact_route_reason(NoMap, tok) == "no_rewrite_map"

    class NoEffectMap:
        prompt = "Rename missing to absent.\n```python\nx = user\n```\n"
        reference = "x = user\n"
        metadata = {"rewrite_pairs": {"user": "account"}}

    assert _routed_transpld_exact_route_reason(NoEffectMap, tok) == "rewrite_map_no_effect"

    class EffectiveMap:
        prompt = "Rename user to account.\n```python\nx = user\n```\n"
        reference = "x = user\n"
        metadata = {"rewrite_pairs": {"user": "wrong"}}

    assert _routed_transpld_exact_route_reason(EffectiveMap, tok) is None
