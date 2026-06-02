from scripts.analyze_syntax_slot_opportunity import (
    IDENTIFIER,
    KEYWORD,
    LITERAL,
    OPERATOR,
    WHITESPACE,
    _build_index,
    _choose_candidate,
    accepted_prefix_length,
    build_slot_map,
    classify_token_text,
    concretize_continuation,
    SyntaxCandidate,
)


def test_syntax_token_classifier() -> None:
    assert classify_token_text("def") == KEYWORD
    assert classify_token_text("my_name") == IDENTIFIER
    assert classify_token_text("123") == LITERAL
    assert classify_token_text("(") == OPERATOR
    assert classify_token_text("\n    ") == WHITESPACE


def test_syntax_ngram_index_tracks_unique_and_collisions() -> None:
    classes = [KEYWORD, IDENTIFIER, OPERATOR, KEYWORD, IDENTIFIER, OPERATOR]
    index = _build_index(classes, 3)
    assert index[(KEYWORD, IDENTIFIER, OPERATOR)] == [0, 3]
    assert (IDENTIFIER, OPERATOR, KEYWORD) in index


def test_slot_map_fills_identifier_and_stops_before_uncertain_slot() -> None:
    source_ids = [10, 11, 12, 13]
    source_classes = [IDENTIFIER, OPERATOR, IDENTIFIER, OPERATOR]
    query_ids = [20, 11]
    query_classes = [IDENTIFIER, OPERATOR]
    slot_map = build_slot_map(source_ids[:2], source_classes[:2], query_ids, query_classes)
    assert slot_map == {10: 20}

    draft, uncertain, attempts, success = concretize_continuation(
        source_ids=source_ids,
        source_classes=source_classes,
        start=0,
        cap=4,
        slot_map=slot_map or {},
        recent_ids=set(query_ids),
    )
    assert draft == [20, 11]
    assert uncertain is True
    assert attempts == 2
    assert success == 1


def test_slot_map_rejects_inconsistent_mapping() -> None:
    assert (
        build_slot_map(
            [10, 10],
            [IDENTIFIER, IDENTIFIER],
            [20, 21],
            [IDENTIFIER, IDENTIFIER],
        )
        is None
    )


def test_choose_candidate_honors_unique_requirement_and_min_prefix() -> None:
    collision = SyntaxCandidate("t", 0, "reference", 0, 2, False, 20, 20, False, 0, 0)
    unique_short = SyntaxCandidate("t", 0, "reference", 1, 1, True, 4, 4, False, 0, 0)
    assert _choose_candidate(
        [collision, unique_short],
        require_unique=True,
        min_concrete_prefix=8,
    ) is None
    assert _choose_candidate(
        [collision],
        require_unique=False,
        min_concrete_prefix=8,
    ) == collision


def test_accepted_prefix_length() -> None:
    assert accepted_prefix_length([1, 2, 3], [1, 2, 9]) == 2
    assert accepted_prefix_length([1, 2, 3], [9, 2, 3]) == 0
