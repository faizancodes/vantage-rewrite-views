from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from vantage_vllm.pld_lookup import find_pld_proposal
from vantage_vllm.vllm_pld_proposer import VantageVllmPLDProposer
from scripts.check_pld_proposal_equivalence import main as equivalence_main


ROOT = Path(__file__).resolve().parents[1]


def test_find_pld_proposal_no_match_returns_none():
    proposal = find_pld_proposal([1, 2, 3], [4, 5], match_n=2)

    assert proposal is None


def test_find_pld_proposal_prompt_match_copies_prompt_continuation():
    proposal = find_pld_proposal([1, 2, 3, 9, 8], [1, 2, 3], match_n=3)

    assert proposal is not None
    assert proposal.tokens == [9, 8]
    assert proposal.source == "prompt"
    assert proposal.source_start == 0
    assert proposal.follow_start == 3


def test_find_pld_proposal_defaults_to_w128_n10():
    proposal = find_pld_proposal(list(range(10)) + [99, 98], list(range(10)))

    assert proposal is not None
    assert proposal.tokens == [99, 98]
    assert proposal.match_n == 10


def test_find_pld_proposal_generated_prefix_match():
    proposal = find_pld_proposal([0, 0], [1, 2, 3, 9, 8, 1, 2, 3], match_n=3)

    assert proposal is not None
    assert proposal.tokens == [9, 8]
    assert proposal.source == "generated"
    assert proposal.source_start == 2


def test_find_pld_proposal_multiple_matches_honor_tie_break():
    latest = find_pld_proposal(
        [1, 2, 3, 4, 1, 2, 9, 8],
        [1, 2],
        match_n=2,
        max_draft_len=2,
    )
    earliest = find_pld_proposal(
        [1, 2, 3, 4, 1, 2, 9, 8],
        [1, 2],
        match_n=2,
        max_draft_len=2,
        tie_break="earliest",
    )

    assert latest is not None
    assert latest.source_start == 4
    assert latest.tokens == [9, 8]
    assert earliest is not None
    assert earliest.source_start == 0
    assert earliest.tokens == [3, 4]


def test_find_pld_proposal_max_draft_and_vllm_cap_limit_output():
    max_limited = find_pld_proposal([1, 2, 3, 4, 5, 6], [1, 2], match_n=2, max_draft_len=3)
    cap_limited = find_pld_proposal(
        [1, 2, 3, 4, 5, 6],
        [1, 2],
        match_n=2,
        max_draft_len=5,
        cap=2,
    )

    assert max_limited is not None
    assert max_limited.tokens == [3, 4, 5]
    assert max_limited.capped is False
    assert cap_limited is not None
    assert cap_limited.tokens == [3, 4]
    assert cap_limited.capped is True


def test_find_pld_proposal_excludes_gold_ranges_and_falls_back_to_valid_source():
    proposal = find_pld_proposal(
        [1, 2, 3, 9, 8, 1, 2, 3, 77, 78],
        [1, 2, 3],
        match_n=3,
        max_draft_len=2,
        exclude_ranges=[(8, 10)],
    )

    assert proposal is not None
    assert proposal.source_start == 0
    assert proposal.tokens == [9, 8]


def test_find_pld_proposal_prompt_only_and_generated_only_search():
    prompt_only = find_pld_proposal(
        [1, 2, 9],
        [1, 2, 8, 1, 2],
        match_n=2,
        max_draft_len=1,
        search_generated=False,
    )
    generated_only = find_pld_proposal(
        [1, 2, 9],
        [1, 2, 8, 1, 2],
        match_n=2,
        max_draft_len=1,
        search_prompt=False,
    )

    assert prompt_only is not None
    assert prompt_only.source == "prompt"
    assert prompt_only.tokens == [9]
    assert generated_only is not None
    assert generated_only.source == "generated"
    assert generated_only.tokens == [8]


def test_find_pld_proposal_is_deterministic_across_repeated_calls():
    kwargs = {
        "prompt_ids": [1, 2, 3, 4, 1, 2, 9, 8],
        "generated_ids": [1, 2],
        "match_n": 2,
        "max_draft_len": 4,
    }

    assert find_pld_proposal(**kwargs) == find_pld_proposal(**kwargs)


def test_vllm_pld_adapter_uses_fake_boundary_metadata_and_records_stats():
    config = SimpleNamespace(
        speculative_config={"num_speculative_tokens": 2},
        model_config=SimpleNamespace(max_model_len=64),
    )
    proposer = VantageVllmPLDProposer(config, match_n=2, max_draft_len=4)
    prompt = [1, 2, 9, 8, 7]
    generated = [1, 2]

    drafts = proposer.propose(
        sampled_token_ids=[[42]],
        num_tokens_no_spec=[len(prompt) + len(generated)],
        token_ids_cpu=[prompt + generated],
        request_metadata={"prompt_lens": [len(prompt)]},
    )

    assert drafts == [[9, 8]]
    assert proposer.stats.calls == 1
    assert proposer.stats.hits == 1
    assert proposer.stats.tokens_proposed == 2
    assert proposer.stats.cap_truncations == 1
    assert proposer.stats.prompt_hits == 1


def test_vllm_pld_adapter_returns_empty_without_prompt_metadata():
    proposer = VantageVllmPLDProposer(match_n=2, max_draft_len=4)

    drafts = proposer.propose(
        sampled_token_ids=[[42]],
        num_tokens_no_spec=[6],
        token_ids_cpu=[[1, 2, 9, 8, 1, 2]],
    )

    assert drafts == [[]]
    assert proposer.stats.metadata_missing == 1
    assert proposer.stats.misses == 1


def test_vllm_pld_adapter_is_deterministic():
    proposer = VantageVllmPLDProposer(match_n=2, max_draft_len=4, cap=2)
    kwargs = {
        "sampled_token_ids": [[42]],
        "num_tokens_no_spec": [6],
        "token_ids_cpu": [[1, 2, 9, 8, 1, 2]],
        "request_metadata": {"prompt_lens": [4]},
    }

    assert proposer.propose(**kwargs) == proposer.propose(**kwargs) == [[9, 8]]


def test_check_pld_proposal_equivalence_help():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_pld_proposal_equivalence.py"), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--trace" in result.stdout


def test_check_pld_proposal_equivalence_compares_jsonl_traces(tmp_path):
    expected = tmp_path / "expected.jsonl"
    actual = tmp_path / "actual.jsonl"
    row = {
        "task_id": "task",
        "step": 0,
        "proposal_token_ids": [9, 8],
        "proposal_match_len": 2,
        "proposal_source_start_token": 0,
        "proposal_follow_start_token": 2,
    }
    expected.write_text(json.dumps(row) + "\n", encoding="utf-8")
    actual.write_text(json.dumps(row) + "\n", encoding="utf-8")

    assert equivalence_main(["--expected-trace", str(expected), "--actual-trace", str(actual)]) == 0


def test_check_pld_proposal_equivalence_replays_trace_rows(tmp_path):
    trace = tmp_path / "trace.jsonl"
    row = {
        "task_id": "task",
        "step": 0,
        "prompt_token_ids": [1, 2, 9, 8],
        "generated_token_ids": [1, 2],
        "proposal_token_ids": [9, 8],
        "proposal_match_len": 2,
        "proposal_source_start_token": 0,
        "proposal_follow_start_token": 2,
    }
    trace.write_text(json.dumps(row) + "\n", encoding="utf-8")

    assert equivalence_main(["--trace", str(trace), "--match-n", "2"]) == 0
