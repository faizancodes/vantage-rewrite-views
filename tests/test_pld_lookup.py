from __future__ import annotations

from vantage_vllm import PLDConfig, PromptLookupProposer, lookup_w128_n10


def _propose(context, generated, *, match_length=3, max_draft_length=4, min_match_length=None, exclude_ranges=()):
    proposer = PromptLookupProposer(
        PLDConfig(
            match_length=match_length,
            min_match_length=min_match_length,
            max_draft_length=max_draft_length,
        )
    )
    return proposer.propose(
        context_ids=context,
        generated_ids=generated,
        exclude_ranges=exclude_ranges,
    )


def test_prompt_match_copies_source_continuation():
    proposal = _propose([1, 2, 3, 9, 8, 7], [1, 2, 3])

    assert proposal is not None
    assert proposal.tokens == [9, 8, 7]
    assert proposal.source == "prompt"
    assert proposal.source_start == 0
    assert proposal.match_length == 3


def test_generated_prefix_match_copies_generated_continuation():
    proposal = _propose([0, 0], [1, 2, 3, 9, 8, 1, 2, 3])

    assert proposal is not None
    assert proposal.tokens == [9, 8]
    assert proposal.source == "generated"
    assert proposal.source_start == 2


def test_no_match_returns_none_and_records_miss():
    proposer = PromptLookupProposer(PLDConfig(match_length=3, max_draft_length=4))

    assert proposer.propose(context_ids=[1, 2, 3], generated_ids=[4, 5, 6]) is None
    assert proposer.stats.calls == 1
    assert proposer.stats.misses == 1
    assert proposer.stats.hits == 0


def test_edge_match_at_start_and_follow_stops_before_query():
    proposal = _propose([3, 4, 1, 2], [3, 4], match_length=2, max_draft_length=8)

    assert proposal is not None
    assert proposal.tokens == [1, 2]
    assert proposal.source_start == 0
    assert proposal.follow_end == proposal.query_start


def test_multiple_matches_tie_break_prefers_latest_match():
    proposal = _propose([1, 2, 3, 4, 1, 2, 9, 8], [1, 2], match_length=2, max_draft_length=4)

    assert proposal is not None
    assert proposal.source_start == 4
    assert proposal.tokens == [9, 8]


def test_max_cap_limits_draft_length():
    proposal = _propose([1, 2, 3, 9, 8, 7], [1, 2, 3], max_draft_length=2)

    assert proposal is not None
    assert proposal.tokens == [9, 8]


def test_n_longer_than_generated_prefix_returns_none_by_default():
    assert _propose([1, 2, 9], [1, 2], match_length=3) is None


def test_shorter_fallback_is_configurable():
    proposal = _propose([1, 2, 9], [1, 2], match_length=3, min_match_length=2)

    assert proposal is not None
    assert proposal.tokens == [9]
    assert proposal.match_length == 2


def test_empty_input_returns_none():
    assert _propose([], [], match_length=1) is None
    assert _propose([], [1], match_length=1) is None


def test_exclude_range_blocks_source_match():
    proposal = _propose(
        [1, 2, 3, 9, 8, 1, 2, 3, 7, 6],
        [1, 2, 3],
        exclude_ranges=[(5, 8)],
    )

    assert proposal is not None
    assert proposal.source_start == 0
    assert proposal.tokens == [9, 8]


def test_deterministic_output_across_calls_and_instances():
    context = [1, 2, 3, 4, 1, 2, 9, 8]
    generated = [1, 2]
    proposer_a = PromptLookupProposer(PLDConfig(match_length=2, max_draft_length=4))
    proposer_b = PromptLookupProposer(PLDConfig(match_length=2, max_draft_length=4))

    first = proposer_a.propose(context_ids=context, generated_ids=generated)
    second = proposer_a.propose(context_ids=context, generated_ids=generated)
    third = proposer_b.propose(context_ids=context, generated_ids=generated)

    assert first == second == third
    assert proposer_a.stats.calls == 2
    assert proposer_a.stats.hits == 2
    assert proposer_a.stats.tokens_proposed == 4


def test_w128_n10_convenience_uses_frozen_defaults():
    proposal = lookup_w128_n10(
        context_ids=list(range(10)) + [99, 98, 97],
        generated_ids=list(range(10)),
    )

    assert proposal is not None
    assert proposal.tokens == [99, 98, 97]
    assert proposal.match_length == 10
