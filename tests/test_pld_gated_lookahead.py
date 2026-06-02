from asts.blazedit_decoder import (
    _pld_lookahead_rule_predict_weak,
    _pld_lookahead_rule_weak_reason,
)


def test_pld_lookahead_rule_routes_miss_to_weak():
    assert _pld_lookahead_rule_predict_weak(
        drafts=[],
        match_len=0,
        candidate_count=0,
        threshold=4,
    )


def test_pld_lookahead_rule_keeps_long_exact_pld_strong():
    assert not _pld_lookahead_rule_predict_weak(
        drafts=list(range(32)),
        match_len=10,
        candidate_count=1,
        threshold=4,
    )


def test_pld_lookahead_rule_routes_short_ambiguous_hit_to_weak():
    assert _pld_lookahead_rule_predict_weak(
        drafts=list(range(6)),
        match_len=4,
        candidate_count=3,
        threshold=4,
    )


def test_pld_lookahead_rule_reason_is_stable():
    assert (
        _pld_lookahead_rule_weak_reason(
            drafts=[],
            match_len=0,
            candidate_count=0,
            threshold=4,
        )
        == "pld_miss"
    )
    assert (
        _pld_lookahead_rule_weak_reason(
            drafts=list(range(32)),
            match_len=10,
            candidate_count=1,
            threshold=4,
        )
        is None
    )
