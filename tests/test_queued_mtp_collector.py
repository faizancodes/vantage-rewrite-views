from scripts.collect_queued_mtp_training_data import _eligible_pairs


def _row(step: int, *, start: int, emitted: int = 1, draft_len: int = 128, accepted: int = 0):
    return {
        "step": step,
        "_generated_start": start,
        "n_emitted": emitted,
        "proposal_tokens": draft_len,
        "k": draft_len,
        "n_accepted_drafts": accepted,
    }


def test_all_filter_collects_every_adjacent_step():
    rows = [_row(i, start=i) for i in range(5)]
    pairs, counts = _eligible_pairs(
        rows,
        threshold=4,
        weak_field="draft_len",
        include_dropped=False,
        create_filter="all",
        use_filter="all",
    )
    assert len(pairs) == 4
    assert counts["queue_created_candidates"] == 4
    assert counts["queue_used_examples"] == 4


def test_phase5_weak_filters_require_consecutive_weak_steps():
    rows = [
        _row(0, start=0, draft_len=2),
        _row(1, start=1, draft_len=128),
        _row(2, start=2, draft_len=2),
        _row(3, start=3, draft_len=2),
    ]
    pairs, counts = _eligible_pairs(
        rows,
        threshold=4,
        weak_field="draft_len",
        include_dropped=False,
        create_filter="weak",
        use_filter="weak",
    )
    assert len(pairs) == 1
    assert pairs[0][0]["step"] == 2
    assert pairs[0][1]["step"] == 3
    assert counts["dropped_create_filter"] == 1
    assert counts["dropped_pld_strong"] == 1


def test_dropped_use_filter_can_be_retained_for_analysis():
    rows = [_row(0, start=0, draft_len=2), _row(1, start=1, draft_len=128)]
    pairs, counts = _eligible_pairs(
        rows,
        threshold=4,
        weak_field="draft_len",
        include_dropped=True,
        create_filter="weak",
        use_filter="weak",
    )
    assert len(pairs) == 1
    assert pairs[0][2] is True
    assert counts["dropped_use_filter"] == 1


def test_position_mismatch_is_not_silent():
    rows = [_row(0, start=0, emitted=2, draft_len=2), _row(1, start=7, draft_len=2)]
    pairs, counts = _eligible_pairs(
        rows,
        threshold=4,
        weak_field="draft_len",
        include_dropped=False,
        create_filter="all",
        use_filter="all",
    )
    assert pairs == []
    assert counts["position_mismatch"] == 1
