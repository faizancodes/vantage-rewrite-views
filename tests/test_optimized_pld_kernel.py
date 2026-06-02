from __future__ import annotations

import random

import pytest

from vantage_vllm.optimized_pld import find_full_prefix_pld_proposal
from vantage_vllm.pld_lookup import PLDLookupResult, find_pld_proposal


def _oracle(
    prefix: list[int],
    *,
    match_n: int,
    max_draft_len: int = 128,
    cap: int | None = None,
    tie_break: str = "latest",
) -> PLDLookupResult | None:
    return find_pld_proposal(
        prefix[:-match_n],
        prefix[-match_n:],
        match_n=match_n,
        max_draft_len=max_draft_len,
        cap=cap,
        tie_break=tie_break,  # type: ignore[arg-type]
    )


def _assert_matches_oracle(
    prefix: list[int],
    *,
    match_n: int,
    max_draft_len: int = 128,
    cap: int | None = None,
    tie_break: str = "latest",
) -> None:
    expected = _oracle(
        prefix,
        match_n=match_n,
        max_draft_len=max_draft_len,
        cap=cap,
        tie_break=tie_break,
    )
    actual = find_full_prefix_pld_proposal(
        prefix,
        match_n=match_n,
        max_draft_len=max_draft_len,
        cap=cap,
        tie_break=tie_break,  # type: ignore[arg-type]
        prefer_numba=False,
    )

    if expected is None:
        assert actual is None
        return
    assert actual is not None
    assert actual.tokens == expected.tokens
    assert actual.match_n == expected.match_n
    assert actual.source_start == expected.source_start
    assert actual.source_end == expected.source_end
    assert actual.follow_start == expected.follow_start
    assert actual.follow_end == expected.follow_end
    assert actual.query_start == expected.query_start
    assert actual.query_end == expected.query_end
    assert actual.capped == expected.capped
    assert actual.cap == expected.cap


def test_no_match_returns_none() -> None:
    _assert_matches_oracle([1, 2, 3, 4, 5, 6], match_n=2)


def test_full_prefix_match_copies_continuation() -> None:
    _assert_matches_oracle([0, 0, 1, 2, 3, 9, 8, 1, 2, 3], match_n=3)

    proposal = find_full_prefix_pld_proposal(
        [0, 0, 1, 2, 3, 9, 8, 1, 2, 3],
        match_n=3,
        prefer_numba=False,
    )

    assert proposal is not None
    assert proposal.tokens == [9, 8]
    assert proposal.source_start == 2
    assert proposal.follow_start == 5


def test_multiple_matches_default_to_latest_match() -> None:
    prefix = [1, 2, 3, 4, 1, 2, 9, 8, 1, 2]

    latest = find_full_prefix_pld_proposal(prefix, match_n=2, prefer_numba=False)
    earliest = find_full_prefix_pld_proposal(
        prefix,
        match_n=2,
        tie_break="earliest",
        prefer_numba=False,
    )

    assert latest is not None
    assert latest.source_start == 4
    assert latest.tokens == [9, 8]
    assert earliest is not None
    assert earliest.source_start == 0
    assert earliest.tokens == [3, 4, 1, 2, 9, 8]
    _assert_matches_oracle(prefix, match_n=2)
    _assert_matches_oracle(prefix, match_n=2, tie_break="earliest")


def test_max_draft_len_limits_proposal() -> None:
    prefix = [1, 2, 3, 4, 5, 6, 1, 2]

    proposal = find_full_prefix_pld_proposal(
        prefix,
        match_n=2,
        max_draft_len=3,
        prefer_numba=False,
    )

    assert proposal is not None
    assert proposal.tokens == [3, 4, 5]
    assert proposal.capped is False
    _assert_matches_oracle(prefix, match_n=2, max_draft_len=3)


def test_cap_limits_proposal_and_records_metadata() -> None:
    prefix = [1, 2, 3, 4, 5, 6, 1, 2]

    proposal = find_full_prefix_pld_proposal(
        prefix,
        match_n=2,
        max_draft_len=5,
        cap=2,
        prefer_numba=False,
    )

    assert proposal is not None
    assert proposal.tokens == [3, 4]
    assert proposal.capped is True
    assert proposal.cap == 2
    _assert_matches_oracle(prefix, match_n=2, max_draft_len=5, cap=2)


def test_short_prefix_returns_none() -> None:
    assert find_full_prefix_pld_proposal([1, 2], match_n=3, prefer_numba=False) is None
    assert find_full_prefix_pld_proposal([1, 2], match_n=2, prefer_numba=False) is None


def test_end_match_without_continuation_returns_none() -> None:
    _assert_matches_oracle([1, 2, 1, 2], match_n=2)


def test_random_sequences_match_pure_oracle() -> None:
    rng = random.Random(1337)
    for _ in range(250):
        prefix_len = rng.randint(0, 96)
        prefix = [rng.randrange(0, 17) for _ in range(prefix_len)]
        match_n = rng.randint(1, 8)
        max_draft_len = rng.randint(0, 16)
        cap = rng.choice([None, 0, 1, 2, 4, 8, 16])
        tie_break = rng.choice(["latest", "earliest"])

        _assert_matches_oracle(
            prefix,
            match_n=match_n,
            max_draft_len=max_draft_len,
            cap=cap,
            tie_break=tie_break,
        )


def test_numpy_array_input_matches_list_input_when_numpy_is_available() -> None:
    np = pytest.importorskip("numpy")
    prefix = [1, 2, 3, 4, 1, 2, 9, 8, 1, 2]

    from_array = find_full_prefix_pld_proposal(
        np.asarray(prefix, dtype=np.int64),
        match_n=2,
        prefer_numba=False,
    )
    from_list = find_full_prefix_pld_proposal(prefix, match_n=2, prefer_numba=False)

    assert from_array == from_list


def test_numba_backend_matches_python_when_available() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("numba")
    prefix = [1, 2, 3, 4, 1, 2, 9, 8, 1, 2]

    from_numba = find_full_prefix_pld_proposal(
        np.asarray(prefix, dtype=np.int64),
        match_n=2,
        cap=4,
        prefer_numba=True,
    )
    from_python = find_full_prefix_pld_proposal(
        prefix,
        match_n=2,
        cap=4,
        prefer_numba=False,
    )

    assert from_numba is not None
    assert from_python is not None
    assert from_numba.tokens == from_python.tokens
    assert from_numba.source_start == from_python.source_start
    assert from_numba.backend == "numba"


def test_invalid_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="match_n"):
        find_full_prefix_pld_proposal([1, 2, 3], match_n=0)
    with pytest.raises(ValueError, match="max_draft_len"):
        find_full_prefix_pld_proposal([1, 2, 3], max_draft_len=-1)
    with pytest.raises(ValueError, match="cap"):
        find_full_prefix_pld_proposal([1, 2, 3], cap=-1)
    with pytest.raises(ValueError, match="tie_break"):
        find_full_prefix_pld_proposal([1, 2, 3], tie_break="middle")  # type: ignore[arg-type]
