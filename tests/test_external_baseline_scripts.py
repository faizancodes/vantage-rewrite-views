from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_hf_prompt_lookup_baseline as hf_baseline
from scripts import run_vllm_baseline_eval as vllm_baseline


def test_vllm_missing_dependency_writes_report(tmp_path, monkeypatch):
    def fake_run(_args):
        raise ModuleNotFoundError("No module named 'vllm'")

    monkeypatch.setattr(vllm_baseline, "run_vllm", fake_run)
    rc = vllm_baseline.main(
        [
            "--problem-jsonl",
            "does-not-matter.jsonl",
            "--n",
            "1",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert rc == 2
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["failure"]["type"] == "missing_dependency"
    assert "vllm" in (tmp_path / "report.md").read_text()


def test_hf_prompt_lookup_incompatibility_writes_report(tmp_path, monkeypatch):
    monkeypatch.setattr(
        hf_baseline,
        "transformers_supports_prompt_lookup",
        lambda: (False, "GenerationConfig lacks prompt_lookup_num_tokens"),
    )

    rc = hf_baseline.main(
        [
            "--problem-jsonl",
            "does-not-matter.jsonl",
            "--n",
            "1",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert rc == 1
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["failure"]["type"] == "incompatibility"
    assert "prompt_lookup_num_tokens" in (tmp_path / "report.md").read_text()


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("greedy", None),
        ("ngram_speculation", {"method": "ngram", "prompt_lookup_max": 64}),
    ],
)
def test_vllm_engine_kwargs_records_ngram_config(backend, expected):
    args = vllm_baseline.parse_args(
        [
            "--backend",
            backend,
            "--ngram-prompt-lookup-min",
            "2",
            "--ngram-prompt-lookup-max",
            "64",
            "--num-speculative-tokens",
            "8",
            "--output-dir",
            "unused",
        ]
    )

    kwargs = vllm_baseline.build_engine_kwargs(args)
    assert kwargs["dtype"] == "bfloat16"
    if expected is None:
        assert "speculative_config" not in kwargs
    else:
        assert kwargs["speculative_config"]["method"] == expected["method"]
        assert kwargs["speculative_config"]["prompt_lookup_min"] == 2
        assert kwargs["speculative_config"]["prompt_lookup_max"] == expected["prompt_lookup_max"]
        assert kwargs["speculative_config"]["num_speculative_tokens"] == 8


def test_hf_generation_kwargs_records_prompt_lookup_knobs():
    args = hf_baseline.parse_args(
        [
            "--prompt-lookup-num-tokens",
            "16",
            "--max-matching-ngram-size",
            "4",
            "--max-new-tokens",
            "32",
            "--output-dir",
            "unused",
        ]
    )

    class FakeTokenizer:
        eos_token_id = 1
        pad_token_id = 2

    kwargs = hf_baseline.build_generation_kwargs(args, FakeTokenizer())
    assert kwargs["do_sample"] is False
    assert kwargs["prompt_lookup_num_tokens"] == 16
    assert kwargs["max_matching_ngram_size"] == 4
    assert kwargs["max_new_tokens"] == 32


def test_default_external_baseline_split_is_balanced_manifest():
    vllm_args = vllm_baseline.parse_args(["--output-dir", "unused"])
    hf_args = hf_baseline.parse_args(["--output-dir", "unused"])
    assert vllm_baseline.resolve_problem_jsonl(vllm_args).name == (
        "real_commit_manifest_balanced_1000_v2_test500.jsonl"
    )
    assert hf_baseline.resolve_problem_jsonl(hf_args).name == (
        "real_commit_manifest_balanced_1000_v2_test500.jsonl"
    )
