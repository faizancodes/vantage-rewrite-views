import json

from asts.pld_reranker import (
    DEFAULT_WEIGHTS_PATH,
    PLDRerankCandidate,
    PLDRerankContext,
    apply_score_margin_gate,
    compute_left_extension,
    load_reranker_weights,
    select_candidate_by_policy,
    select_best_candidate,
)
from asts.blazedit_decoder import _ensure_position_in_top_k
from scripts.train_pld_candidate_reranker import _parse_examples, _choose_linear


def test_default_reranker_weights_load():
    weights = load_reranker_weights(DEFAULT_WEIGHTS_PATH)
    assert weights.top_k == 4
    assert len(weights.weights) == len(weights.feature_names)


def test_runtime_and_offline_selection_match_on_fixture():
    weights = load_reranker_weights(DEFAULT_WEIGHTS_PATH)
    raw = {
        "task_id": "fixture/0",
        "step_id": 7,
        "baseline_candidate_position": 100,
        "baseline_accepted_len": 1,
        "candidate_count_n10": 4,
        "match_len": 10,
        "candidates": [
            {
                "rank": 1,
                "source_position": 100,
                "source_type": "prompt/reference",
                "source_distance_from_previous_good_source": None,
                "lcp_with_actual_future_output": 1,
                "candidate_draft_prefix_128": "foo bar common",
            },
            {
                "rank": 2,
                "source_position": 80,
                "source_type": "prompt/reference",
                "source_distance_from_previous_good_source": None,
                "lcp_with_actual_future_output": 2,
                "candidate_draft_prefix_128": "unique alpha beta gamma",
            },
            {
                "rank": 3,
                "source_position": 60,
                "source_type": "prompt/reference",
                "source_distance_from_previous_good_source": None,
                "lcp_with_actual_future_output": 3,
                "candidate_draft_prefix_128": "same same",
            },
            {
                "rank": 4,
                "source_position": 40,
                "source_type": "prompt/reference",
                "source_distance_from_previous_good_source": None,
                "lcp_with_actual_future_output": 4,
                "candidate_draft_prefix_128": "delta epsilon zeta eta",
            },
        ],
    }
    examples = _parse_examples([json.loads(json.dumps(raw))], k=4)
    offline_selected = _choose_linear(weights.weights)(examples[0])

    runtime_candidates = [
        PLDRerankCandidate(
            rank0=i,
            source_position=int(c["source_position"]),
            source_type=str(c["source_type"]),
            source_distance_from_previous_good_source=None,
            draft_prefix_text=str(c["candidate_draft_prefix_128"]),
        )
        for i, c in enumerate(raw["candidates"])
    ]
    runtime_selected, scores1 = select_best_candidate(
        runtime_candidates,
        weights,
        context=PLDRerankContext(candidate_count=4, match_len=10),
        top_k=4,
    )
    runtime_selected2, scores2 = select_best_candidate(
        runtime_candidates,
        weights,
        context=PLDRerankContext(candidate_count=4, match_len=10),
        top_k=4,
    )

    assert runtime_selected is not None
    assert runtime_selected2 is not None
    assert runtime_selected.rank0 == offline_selected.rank0 == 3
    assert runtime_selected2.rank0 == runtime_selected.rank0
    assert scores2 == scores1


def test_baseline_candidate_is_always_included():
    positions = [100, 80, 60, 40]
    assert _ensure_position_in_top_k(
        positions, baseline_position=80, top_k=4
    ) == positions
    assert _ensure_position_in_top_k(
        positions, baseline_position=20, top_k=4
    ) == positions
    assert _ensure_position_in_top_k(
        [100, 80, 60, 40, 20], baseline_position=20, top_k=4
    ) == [100, 80, 60, 20]


def test_margin_gate_falls_back_and_selects():
    baseline = PLDRerankCandidate(rank0=0, source_position=10, source_type="prompt")
    selected = PLDRerankCandidate(rank0=1, source_position=20, source_type="prompt")
    chosen, margin, gated = apply_score_margin_gate(
        selected=selected,
        baseline=baseline,
        selected_score=1.2,
        baseline_score=1.0,
        margin_gate=True,
        margin=0.5,
    )
    assert chosen == baseline
    assert abs(float(margin) - 0.2) < 1e-9
    assert gated is True

    chosen, margin, gated = apply_score_margin_gate(
        selected=selected,
        baseline=baseline,
        selected_score=2.0,
        baseline_score=1.0,
        margin_gate=True,
        margin=0.5,
    )
    assert chosen == selected
    assert margin == 1.0
    assert gated is False


def test_left_extension_counts_matching_left_context():
    # Match suffix starts are at 3 and 10; tokens [1, 2, 3] match to the left.
    tokens = [1, 2, 3, 9, 9, 9, 0, 1, 2, 3, 9, 9, 9]
    assert compute_left_extension(
        tokens,
        generated_suffix_start=10,
        candidate_source_suffix_start=3,
        max_left=128,
    ) == 3
    assert compute_left_extension(
        tokens,
        generated_suffix_start=10,
        candidate_source_suffix_start=3,
        max_left=2,
    ) == 2


def test_policy_selection_is_deterministic():
    weights = load_reranker_weights(DEFAULT_WEIGHTS_PATH)
    candidates = [
        PLDRerankCandidate(rank0=0, source_position=10, source_type="prompt"),
        PLDRerankCandidate(rank0=1, source_position=20, source_type="prompt"),
        PLDRerankCandidate(rank0=2, source_position=30, source_type="generated"),
    ]
    context = PLDRerankContext(candidate_count=3, match_len=10)
    selected1, scores1, feats1 = select_candidate_by_policy(
        candidates, weights, context=context, top_k=3, policy="learned"
    )
    selected2, scores2, feats2 = select_candidate_by_policy(
        candidates, weights, context=context, top_k=3, policy="learned"
    )
    assert selected1 == selected2
    assert scores1 == scores2
    assert feats1 == feats2
