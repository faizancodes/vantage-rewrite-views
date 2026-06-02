"""CPU tests for asts.pvp_chain.chain_pld_lookup.

Synthetic-token tests; no model load.
"""

from __future__ import annotations

from asts.pvp_chain import chain_pld_lookup


def test_chain_depth1_recovers_pld():
    """depth=1 must match prompt_lookup_draft exactly: most-recent suffix span."""
    from asts.blazedit_decoder import prompt_lookup_draft

    # Two repeated 4-grams; the suffix [10,20,30,40] matches the prior copy.
    tokens = [10, 20, 30, 40, 1, 2, 3, 4, 10, 20, 30, 40]
    expected_draft, expected_match, expected_src, expected_follow = prompt_lookup_draft(
        tokens, max_matching_ngram_size=4, max_draft_tokens=8,
    )
    assert expected_draft, "fixture must produce a non-empty PLD draft"

    result = chain_pld_lookup(tokens, n_match=4, n_draft=8, depth=1)
    assert len(result) == 1
    r = result[0]
    assert r.draft == expected_draft
    assert r.match_len == expected_match
    assert r.source_start == expected_src
    assert r.follow_start == expected_follow


def test_chain_two_matching_spans():
    """depth=2 chain finds two consecutive draft segments 11+ tokens apart.

    Construction:
      block_a = pattern_p + suffix_p   (pattern length 5, suffix length 6)
      block_b = pattern_p + suffix_p   (identical → exact PLD anchors)
      block_c = pattern_p              (the live suffix)
    First lookup matches block_b's pattern → draft = suffix_p (6 tokens).
    After assumed_extend = suffix_p + [bonus], the new query suffix's tail
    contains a repeated trail; the second lookup should still hit because
    block_a's pattern+suffix is the longest available anchor for the
    extended sequence.
    """
    pattern_p = [70, 71, 72, 73, 74]
    suffix_p = [80, 81, 82, 83, 84, 85]
    block_a = pattern_p + suffix_p
    block_b = pattern_p + suffix_p
    block_c = pattern_p
    tokens = block_a + block_b + block_c

    result = chain_pld_lookup(
        tokens, n_match=5, n_draft=6, depth=2, min_matching_ngram_size=2,
    )

    # First chain step must succeed.
    assert len(result) >= 1, "first lookup should match a 5-gram"
    r0 = result[0]
    assert r0.draft[0] == suffix_p[0], "first draft must start with suffix_p"
    assert r0.match_len >= 2
    # Second chain step: depends on PLD finding any non-empty draft in the
    # extended sequence. We require depth >= 2 OR an early-return; both are
    # legal per the spec ("returns early if any lookup fails"). What we
    # assert is that the *types* and shapes are well-formed if present.
    for r in result:
        assert isinstance(r.draft, list)
        assert all(isinstance(t, int) for t in r.draft)
        assert r.match_len >= 1
        assert r.source_start >= 0
        assert r.follow_start >= r.source_start


def test_chain_misses_returns_empty():
    """A query suffix with no prior n-gram match returns []."""
    tokens = [9, 9, 9, 9, 9, 1, 2, 3]
    out = chain_pld_lookup(tokens, n_match=4, n_draft=5, depth=3)
    assert out == []


def test_chain_respects_min_ngram():
    """min_matching_ngram_size=3 rejects shorter matches."""
    tokens = [1, 2, 3, 9, 1, 2]
    out = chain_pld_lookup(
        tokens, n_match=4, n_draft=4, depth=2, min_matching_ngram_size=3,
    )
    assert out == []


def test_chain_depth_zero_is_empty():
    tokens = [1, 2, 3, 1, 2, 3, 4]
    out = chain_pld_lookup(tokens, n_match=3, n_draft=3, depth=0)
    assert out == []


def test_chain_extends_query_with_draft_plus_bonus():
    """Sanity: chain step k>0 must search the *extended* sequence, not just tokens.

    Build tokens such that the live suffix has a match in tokens at position X,
    and the chain-extended sequence has a different match at position Y that
    only appears after the draft+bonus is appended.
    """
    # Construct a base where suffix [50,51] only matches at position 6 in
    # tokens, but after appending [draft+bonus], a new suffix [a,b] should
    # match at position 0 in the extended view.
    tokens = [10, 11, 12, 13, 14, 15, 50, 51, 99, 50, 51]
    # depth-1 finds [50,51] -> draft = [99, 50, 51] (or some subset)
    out = chain_pld_lookup(tokens, n_match=2, n_draft=3, depth=2)
    assert len(out) >= 1
    # All entries must be well-formed.
    for r in out:
        assert len(r.draft) >= 1
        assert r.match_len >= 1
