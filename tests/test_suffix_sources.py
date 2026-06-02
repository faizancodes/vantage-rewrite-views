from __future__ import annotations

from scripts.analyze_suffix_sources import analyze, classify_pool, classify_prompt_region


def test_classify_pool_prompt_generated_mixed():
    assert classify_pool(0, 4, 10) == "prompt"
    assert classify_pool(10, 14, 10) == "generated"
    assert classify_pool(8, 12, 10) == "mixed"


def test_classify_prompt_region_without_tokenizer_offsets():
    prompt = 'def f(x):\n    """Return x.\n    assert f(1) == 1\n    """\n'
    # Character offsets stand in for tokenizer offsets in this unit test.
    offsets = [(i, i + 1) for i in range(len(prompt))]
    assert classify_prompt_region(prompt, offsets, 0, 3) == "signature"
    assert classify_prompt_region(prompt, offsets, prompt.index("Return"), prompt.index("Return") + 1) == "docstring"
    assert classify_prompt_region(prompt, offsets, prompt.index("assert"), prompt.index("assert") + 1) == "assert/test"


def test_analyze_suffix_sources_synthetic_prompt():
    prompt = 'def f(x):\n    """Return x.\n    assert f(1) == 1\n    """\n'
    completions = [{"task_id": "T/0", "prompt": prompt}]
    steps = [
        {
            "task_id": "T/0",
            "method": "vantage_suffix",
            "proposal_kind": "local_suffix",
            "proposal_source_start_token": prompt.index("assert"),
            "proposal_source_end_token": prompt.index("assert") + 3,
            "prompt_len": len(prompt),
            "n_accepted_nonroot_drafts": 2,
            "proposal_tokens": 3,
            "proposal_match_len": 3,
            "wall_us": 10.0,
            "proposal_us": 1.0,
        }
    ]
    report = analyze(steps, completions, "", {"vantage_suffix"})
    row = report["methods"]["vantage_suffix"]["by_source"][0]
    assert row["pool"] == "prompt"
    assert row["region"] == "assert/test"
    assert row["accepted_nonroot_total"] == 2
