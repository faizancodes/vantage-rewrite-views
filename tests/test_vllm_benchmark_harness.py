from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_vllm_benchmarks as harness


def write_manifest(path: Path) -> None:
    rows = [
        {"task_id": "task/1", "prompt": "def add(a, b):\n", "language": "python"},
        {"task_id": "task/2", "prompt": "def sub(a, b):\n", "language": "python"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def install_fake_vllm(monkeypatch, seen: dict) -> None:
    fake = types.ModuleType("vllm")

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            seen["sampling_kwargs"] = kwargs

    class Candidate:
        def __init__(self, index: int):
            self.text = f"completion-{index}"
            self.token_ids = [index, index + 10]
            self.finish_reason = "length"
            self.stop_reason = None

    class RequestOutput:
        def __init__(self, index: int, prompt: str):
            self.request_id = f"req-{index}"
            self.prompt_token_ids = list(range(len(prompt.split())))
            self.outputs = [Candidate(index)]

    class LLM:
        def __init__(self, **kwargs):
            seen["engine_kwargs"] = kwargs

        def generate(self, prompts, sampling_params):
            seen["prompts"] = list(prompts)
            seen["sampling_params"] = sampling_params
            return [RequestOutput(i, prompt) for i, prompt in enumerate(prompts)]

    fake.LLM = LLM
    fake.SamplingParams = SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_greedy_run_writes_shared_schema(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest)
    seen: dict = {}
    install_fake_vllm(monkeypatch, seen)

    rc = harness.main(
        [
            "--method",
            "greedy",
            "--manifest-path",
            str(manifest),
            "--n",
            "2",
            "--max-new-tokens",
            "33",
            "--stop",
            "<END>",
            "--output-dir",
            str(tmp_path / "run"),
            "--run-id",
            "unit-greedy",
        ]
    )

    assert rc == 0
    run_dir = tmp_path / "run"
    config = load_json(run_dir / "config.json")
    summary = load_json(run_dir / "run_summary.json")
    outputs = [json.loads(line) for line in (run_dir / "outputs.jsonl").read_text().splitlines()]

    assert config["run_id"] == "unit-greedy"
    assert config["sampling_params"] == {
        "max_tokens": 33,
        "stop": ["<END>"],
        "temperature": 0.0,
        "top_p": 1.0,
    }
    assert config["speculative_config"] is None
    assert seen["engine_kwargs"]["model"] == "Qwen/Qwen2.5-Coder-7B"
    assert "speculative_config" not in seen["engine_kwargs"]
    assert seen["sampling_kwargs"]["max_tokens"] == 33
    assert summary["status"] == "success"
    assert summary["method"] == "greedy"
    assert summary["num_tasks"] == 2
    assert summary["total_emitted_tokens"] == 4
    assert outputs[0]["task_id"] == "task/1"
    assert outputs[0]["output_text"] == "completion-0"
    assert outputs[0]["output_token_ids"] == [0, 10]
    assert outputs[0]["prompt_hash"] == harness.prompt_hash("def add(a, b):\n")
    assert (run_dir / "raw_stdout.txt").exists()
    assert (run_dir / "raw_stderr.txt").exists()


def test_ngram_method_passes_vllm_speculative_config(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest)
    seen: dict = {}
    install_fake_vllm(monkeypatch, seen)

    rc = harness.main(
        [
            "--method",
            "ngram",
            "--manifest-path",
            str(manifest),
            "--n",
            "1",
            "--ngram-prompt-lookup-min",
            "3",
            "--ngram-prompt-lookup-max",
            "64",
            "--num-speculative-tokens",
            "7",
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )

    assert rc == 0
    speculative = seen["engine_kwargs"]["speculative_config"]
    assert speculative == {
        "method": "ngram",
        "prompt_lookup_min": 3,
        "prompt_lookup_max": 64,
        "num_speculative_tokens": 7,
    }
    summary = load_json(tmp_path / "run" / "run_summary.json")
    assert summary["speculative_config"]["method"] == "ngram"


def test_vantage_fallback_uses_fixed_n10_ngram_and_notes_cap(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest)
    seen: dict = {}
    install_fake_vllm(monkeypatch, seen)

    rc = harness.main(
        [
            "--method",
            "vantage_prompt_only",
            "--manifest-path",
            str(manifest),
            "--n",
            "1",
            "--vantage-match-tokens",
            "10",
            "--vantage-window-tokens",
            "128",
            "--num-speculative-tokens",
            "8",
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )

    assert rc == 0
    speculative = seen["engine_kwargs"]["speculative_config"]
    assert speculative["method"] == "ngram"
    assert speculative["prompt_lookup_min"] == 10
    assert speculative["prompt_lookup_max"] == 10
    assert speculative["num_speculative_tokens"] == 8
    assert "label" not in speculative
    notes = "\n".join(load_json(tmp_path / "run" / "run_summary.json")["notes"])
    assert "not a custom proposer" in notes
    assert "capped by vLLM num_speculative_tokens" in notes


def test_vantage_custom_unavailable_fails_cleanly(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest)
    monkeypatch.delitem(sys.modules, "vllm", raising=False)

    rc = harness.main(
        [
            "--method",
            "vantage_custom",
            "--manifest-path",
            str(manifest),
            "--n",
            "1",
            "--custom-proposer-module",
            "missing_custom_proposer_for_test",
            "--output-dir",
            str(tmp_path / "run"),
            "--run-id",
            "unit-custom-missing",
        ]
    )

    assert rc == 3
    run_dir = tmp_path / "run"
    summary = load_json(run_dir / "run_summary.json")
    assert summary["status"] == "failed"
    assert summary["failure"]["type"] == "custom_proposer_unavailable"
    assert "missing_custom_proposer_for_test" in summary["failure"]["message"]
    assert load_json(run_dir / "config.json")["method"] == "vantage_custom"
    assert (run_dir / "outputs.jsonl").read_text() == ""


def test_custom_class_model_variant_uses_documented_shape(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest)
    seen: dict = {}
    install_fake_vllm(monkeypatch, seen)

    rc = harness.main(
        [
            "--method",
            "vantage_custom",
            "--manifest-path",
            str(manifest),
            "--n",
            "1",
            "--custom-proposer-module",
            "vantage_vllm.minimal_custom_proposer",
            "--custom-proposer-class",
            "MinimalCustomProposer",
            "--custom-config-variant",
            "custom_class_model",
            "--num-speculative-tokens",
            "8",
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )

    assert rc == 0
    speculative = seen["engine_kwargs"]["speculative_config"]
    assert speculative == {
        "method": "custom_class",
        "model": "vantage_vllm.minimal_custom_proposer.MinimalCustomProposer",
        "num_speculative_tokens": 8,
    }
    summary = load_json(tmp_path / "run" / "run_summary.json")
    assert summary["custom_config_variant"] == "custom_class_model"


def test_missing_vllm_fails_cleanly_after_config(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest)
    monkeypatch.delitem(sys.modules, "vllm", raising=False)
    real_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name == "vllm":
            raise ModuleNotFoundError("No module named 'vllm'")
        return real_import_module(name)

    monkeypatch.setattr(harness.importlib, "import_module", fake_import_module)

    rc = harness.main(
        [
            "--method",
            "greedy",
            "--manifest-path",
            str(manifest),
            "--n",
            "1",
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )

    assert rc == 2
    summary = load_json(tmp_path / "run" / "run_summary.json")
    assert summary["status"] == "failed"
    assert summary["failure"]["type"] == "missing_dependency"
    assert (tmp_path / "run" / "config.json").exists()
    assert (tmp_path / "run" / "outputs.jsonl").exists()
