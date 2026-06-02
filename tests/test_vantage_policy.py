from __future__ import annotations

from asts.code_proposers import apply_boundary_rewrites, extract_explicit_rewrites
from asts.vantage_policy import (
    VantageRouterConfig,
    RollingAcceptance,
    decide_prompt_only_saferoute,
    VisibilityFeatures,
    choose_strategy,
    compute_visibility,
    zone_for_node,
)


def test_zone_for_node_marks_error_and_identifier_dark():
    assert zone_for_node("ERROR") == "dark"
    assert zone_for_node("identifier") == "dark"
    assert zone_for_node("comment") == "lit"
    assert zone_for_node("module") == "mid"


def test_visibility_score_orders_lit_above_dark_with_same_confidence():
    dark = VisibilityFeatures(
        node_type="identifier",
        deepest_type="identifier",
        draft_top1_prob=0.35,
        draft_top2_margin=0.05,
    )
    lit = VisibilityFeatures(
        node_type="type_annotation",
        deepest_type="predefined_type",
        draft_top1_prob=0.35,
        draft_top2_margin=0.05,
    )
    dark_score, _ = compute_visibility(dark)
    lit_score, _ = compute_visibility(lit)
    assert lit_score > dark_score


def test_router_prefers_scope_for_identifier_match():
    features = VisibilityFeatures(
        node_type="identifier",
        deepest_type="identifier",
        draft_top1_prob=0.20,
        draft_top2_margin=0.01,
        scope_match_len=3,
    )
    decision = choose_strategy(features, scope_available=True)
    assert decision.strategy == "scope"


def test_router_prefers_retrieval_for_long_lit_match():
    features = VisibilityFeatures(
        node_type="type_annotation",
        deepest_type="predefined_type",
        draft_top1_prob=0.30,
        draft_top2_margin=0.05,
        retrieval_match_len=12,
    )
    decision = choose_strategy(features, retrieval_available=True)
    assert decision.strategy == "retrieve"


def test_router_uses_tail_at_ambiguity_frontier():
    cfg = VantageRouterConfig(low_visibility_threshold=0.10, tail_max_margin=0.08)
    features = VisibilityFeatures(
        node_type="module",
        deepest_type="identifier",
        draft_top1_prob=0.40,
        draft_top2_margin=0.02,
        rolling_accept_rate=0.90,
    )
    decision = choose_strategy(features, cfg)
    assert decision.strategy == "tail_k2w2"


def test_router_v2_defaults_to_tail_after_low_visibility_filter():
    cfg = VantageRouterConfig(
        low_visibility_threshold=0.10,
        default_to_tail=True,
        use_scope=False,
    )
    features = VisibilityFeatures(
        node_type="module",
        deepest_type="call",
        draft_top1_prob=0.40,
        draft_top2_margin=0.20,
        rolling_accept_rate=0.90,
    )
    decision = choose_strategy(features, cfg)
    assert decision.strategy == "tail_k2w2"


def test_rolling_acceptance_falls_back_to_global_then_node_specific():
    rolling = RollingAcceptance(window=4, default=0.5)
    assert rolling.rate("identifier") == 0.5
    rolling.update("module", accepted=2, k=2)
    assert rolling.rate("identifier") == 1.0
    rolling.update("identifier", accepted=1, k=2)
    assert rolling.rate("identifier") == 0.5


def test_saferoute_uses_only_prompt_reference_inputs():
    decision = decide_prompt_only_saferoute(
        reference="x = user.name\n",
        rewrite_map={"user": "account"},
        transformed_reference="x = account.name\n",
        reference_tokens=[1, 2, 3],
        transformed_tokens=[1, 4, 3],
    )
    assert decision.use_transpld
    assert decision.reason is None


def test_saferoute_routes_no_effect_and_noop_maps_to_pld():
    no_map = decide_prompt_only_saferoute(reference="x = user\n", rewrite_map={})
    assert not no_map.use_transpld
    assert no_map.reason == "no_rewrite_map"

    explicit_noop = decide_prompt_only_saferoute(
        reference="x = user\n",
        rewrite_map={"user": "user"},
    )
    assert not explicit_noop.use_transpld
    assert explicit_noop.reason == "no_rewrite_map"

    no_effect = decide_prompt_only_saferoute(
        reference="x = user\n",
        rewrite_map={"missing": "absent"},
        transformed_reference="x = user\n",
    )
    assert not no_effect.use_transpld
    assert no_effect.reason == "rewrite_map_no_effect"


def test_saferoute_rejects_tokenization_equivalent_transforms():
    decision = decide_prompt_only_saferoute(
        reference="x = user\n",
        rewrite_map={"user": "account"},
        transformed_reference="x = account\n",
        reference_tokens=[1, 2, 3],
        transformed_tokens=[1, 2, 3],
    )
    assert not decision.use_transpld
    assert decision.reason == "transformed_reference_tokens_equal_reference"


def test_extract_explicit_rewrites_supported_examples():
    assert extract_explicit_rewrites("rename user to account") == {"user": "account"}
    assert extract_explicit_rewrites("replace .name with .display_name") == {
        ".name": ".display_name"
    }
    assert extract_explicit_rewrites(
        "Explicit rewrite map: user -> account, .name -> .display_name."
    ) == {"user": "account", ".name": ".display_name"}
    assert extract_explicit_rewrites("replace 'pending' with 'complete'") == {
        "pending": "complete"
    }


def test_extract_explicit_rewrites_negative_examples():
    assert extract_explicit_rewrites("do not rename user to account") == {}
    assert extract_explicit_rewrites("Do not replace .name with .display_name.") == {}
    assert extract_explicit_rewrites("rename user to user") == {}
    assert (
        extract_explicit_rewrites(
            "No rewrite requested.\n```python\n# rename user to account\nuser = 1\n```\n"
        )
        == {}
    )
    assert (
        extract_explicit_rewrites(
            "Example only.\n```python\n# user -> account\nreturn user.name\n```\n"
        )
        == {}
    )
    assert (
        extract_explicit_rewrites(
            "```python\nreturn user.name\n```\nAfter the code, rename user to account.\n"
        )
        == {}
    )


def test_extract_explicit_rewrites_documents_supported_instruction_region():
    prompt = (
        "Rewrite map: user -> account.\n"
        "```python\n# replace account with user\nreturn user.name\n```\n"
    )
    assert extract_explicit_rewrites(prompt) == {"user": "account"}


def test_extract_explicit_rewrites_multiple_and_overlapping_pairs():
    pairs = extract_explicit_rewrites(
        "replace user.name with account.display_name and rename user to account"
    )
    assert pairs == {"user.name": "account.display_name", "user": "account"}
    assert (
        apply_boundary_rewrites("return user.name, user\n", pairs)
        == "return account.display_name, account\n"
    )


def test_apply_boundary_rewrites_preserves_identifier_boundaries():
    pairs = {"user": "account"}
    assert (
        apply_boundary_rewrites("user user_id get_user other.user\n", pairs)
        == "account user_id get_user other.account\n"
    )


def test_apply_boundary_rewrites_preserves_dotted_field_boundaries():
    pairs = {".name": ".display_name", "client.chat": "responses.create"}
    assert (
        apply_boundary_rewrites(
            "user.name user.name_extra client.chat(timeout=30) old.client.chat()\n",
            pairs,
        )
        == "user.display_name user.name_extra responses.create(timeout=30) old.responses.create()\n"
    )


def test_apply_boundary_rewrites_old_and_new_both_present():
    pairs = extract_explicit_rewrites("rename user to account")
    assert apply_boundary_rewrites("user = account\n", pairs) == "account = account\n"
