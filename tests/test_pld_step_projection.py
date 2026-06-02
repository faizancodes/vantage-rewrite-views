from scripts.evaluate_pld_reranker_step_projection import (
    BaselineStep,
    _oracle_choose_k,
    _choose_rank,
    _limit_examples_like_runtime,
    _project_with_full_steps,
)
from scripts.train_pld_candidate_reranker import Candidate, Example


def _cand(rank: int, accepted: int, pos: int | None = None) -> Candidate:
    return Candidate(
        idx=rank,
        rank0=rank,
        source_position=rank if pos is None else pos,
        source_type="prompt/reference",
        source_distance_from_previous_good_source=None,
        accepted_len=accepted,
        draft_prefix="",
        left_extension=0,
        next2="",
        next4="",
    )


def _example(*, accepted: tuple[int, ...], baseline: int = 0) -> Example:
    candidates = tuple(_cand(i, val) for i, val in enumerate(accepted))
    return Example(
        task_id="task",
        step_id=0,
        baseline_position=baseline,
        baseline_accepted_len=accepted[baseline],
        candidate_count=len(candidates),
        match_len=10,
        candidates=candidates,
    )


def _steps() -> dict[str, list[BaselineStep]]:
    return {
        "task": [
            BaselineStep("task", 0, start=0, emitted=2, accepted_len=1),
            BaselineStep("task", 1, start=2, emitted=1, accepted_len=0),
            BaselineStep("task", 2, start=3, emitted=1, accepted_len=0),
        ]
    }


def test_baseline_replay_reproduces_baseline_step_count() -> None:
    result = _project_with_full_steps(
        examples=[_example(accepted=(1, 3))],
        steps_by_task=_steps(),
        choose=_choose_rank(0),
    )
    assert result.projected_steps == 3
    assert result.corrected_projected_speedup == 1.0


def test_longer_candidate_skips_future_steps() -> None:
    result = _project_with_full_steps(
        examples=[_example(accepted=(1, 3))],
        steps_by_task=_steps(),
        choose=_choose_rank(1),
    )
    assert result.projected_steps == 1
    assert result.skipped_baseline_steps == 2


def test_shorter_candidate_adds_catchup_steps() -> None:
    result = _project_with_full_steps(
        examples=[_example(accepted=(1, 0))],
        steps_by_task=_steps(),
        choose=_choose_rank(1),
    )
    assert result.projected_steps == 4
    assert result.catchup_steps == 1


def test_fixed_rank_policy_is_deterministic() -> None:
    ex = _example(accepted=(1, 4, 2, 9))
    assert _choose_rank(3)(ex).rank0 == 3
    assert _choose_rank(99)(ex).rank0 == 3


def test_oracle_k_uses_requested_candidate_set() -> None:
    ex = _example(accepted=(1, 4, 2, 9, 12, 3), baseline=0)
    assert _oracle_choose_k(4)(ex).accepted_len == 9
    assert _oracle_choose_k(8)(ex).accepted_len == 12


def test_limit_examples_inserts_baseline_for_large_k() -> None:
    ex = _example(accepted=(1, 4, 2, 9, 12), baseline=4)
    limited = _limit_examples_like_runtime([ex], top_k=4)[0]
    assert len(limited.candidates) == 4
    assert limited.candidates[-1].source_position == ex.baseline_position
