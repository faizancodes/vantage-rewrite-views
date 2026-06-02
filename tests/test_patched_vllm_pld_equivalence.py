from __future__ import annotations

import importlib.util
import importlib.machinery
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from scripts.check_pld_proposal_equivalence import main as equivalence_main


ROOT = Path(__file__).resolve().parents[1]


def test_patch_and_unpatch_restore_dummy_ngram_module(tmp_path):
    fake = tmp_path / "ngram_proposer.py"
    original = """
class NgramProposer:
    def __init__(self):
        self.num_speculative_tokens = 2

    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, *args, **kwargs):
        return [[101]]
"""
    fake.write_text(original, encoding="utf-8")
    backup_dir = tmp_path / "backups"

    patch = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/patch_installed_vllm_ngram_to_vantage.py"),
            "--ngram-path",
            str(fake),
            "--backup-dir",
            str(backup_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert patch.returncode == 0, patch.stderr
    assert "VANTAGE_PLD_INSTALLED_NGRAM_PATCH_V1" in fake.read_text(encoding="utf-8")
    assert (tmp_path / "ngram_proposer.py.vantage_original").exists()
    assert (backup_dir / "ngram_proposer_original.py").exists()

    unpatch = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/unpatch_installed_vllm_ngram.py"),
            "--ngram-path",
            str(fake),
            "--backup-dir",
            str(backup_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert unpatch.returncode == 0, unpatch.stderr
    assert fake.read_text(encoding="utf-8") == original


def test_env_gated_wrapper_delegates_or_uses_pld(tmp_path, monkeypatch):
    fake = tmp_path / "ngram_proposer.py"
    fake.write_text(
        """
class NgramProposer:
    def __init__(self):
        self.num_speculative_tokens = 2

    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, *args, **kwargs):
        return [[101]]
""",
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/patch_installed_vllm_ngram_to_vantage.py"),
            "--ngram-path",
            str(fake),
            "--backup-dir",
            str(tmp_path / "backups"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    module = _load_module(fake)
    proposer = module.NgramProposer()

    assert proposer.propose([[0]], [6], [[1, 2, 9, 8, 1, 2]]) == [[101]]

    trace = tmp_path / "trace.jsonl"
    monkeypatch.setenv("VANTAGE_PLD_PATCH", "1")
    monkeypatch.setenv("VANTAGE_PLD_MATCH_N", "2")
    monkeypatch.setenv("VANTAGE_PLD_TRACE_PATH", str(trace))
    monkeypatch.setenv("VANTAGE_PLD_TRACE_SAMPLE_RATE", "1")
    monkeypatch.setenv("VANTAGE_PLD_TRACE_TOKENS", "1")

    assert proposer.propose([[0]], [6], [[1, 2, 9, 8, 1, 2]]) == [[9, 8]]
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["proposal_token_ids"] == [9, 8]
    assert rows[0]["history_token_ids"] == [1, 2, 9, 8, 1, 2]
    assert rows[0]["equivalence_label_candidate"] == "capped_full_prefix_pld"


def test_equivalence_checker_rejects_hash_only_equivalence_label(tmp_path):
    trace = tmp_path / "hash_only.jsonl"
    trace.write_text(
        json.dumps(
            {
                "task_id": "task",
                "step": 0,
                "prefix_len": 6,
                "proposal_len": 2,
                "proposal_token_hash": "abc",
                "proposal_tokens": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "summary.json"

    assert equivalence_main(["--trace", str(trace), "--output-json", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    summary = payload["summaries"][0]
    assert summary["equivalence_label"] == "metadata_insufficient"
    assert summary["skipped_no_tokens"] == 1


def test_equivalence_checker_labels_token_trace_as_capped_full_prefix(tmp_path):
    trace = tmp_path / "token_trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "task_id": "task",
                "step": 0,
                "history_token_ids": [1, 2, 9, 8, 1, 2],
                "prompt_len": 0,
                "proposal_token_ids": [9],
                "proposal_match_len": 2,
                "proposal_source_start_token": 0,
                "proposal_follow_start_token": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "summary.json"

    assert equivalence_main(
        [
            "--trace",
            str(trace),
            "--match-n",
            "2",
            "--cap",
            "1",
            "--output-json",
            str(out),
        ]
    ) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    summary = payload["summaries"][0]
    assert summary["equivalence_label"] == "capped_full_prefix_pld_equivalent"
    assert summary["replayed_rows"] == 1
    assert summary["mismatches"] == []


def test_equivalence_checker_labels_hash_only_with_context_as_insufficient(tmp_path):
    trace = tmp_path / "hash_only_with_context.jsonl"
    trace.write_text(
        json.dumps(
            {
                "task_id": "task",
                "step": 0,
                "prompt_token_ids": [1, 2, 9, 8],
                "generated_token_ids": [1, 2],
                "proposal_hash": "same-hash-is-not-a-token-trace",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "summary.json"

    assert (
        equivalence_main(["--trace", str(trace), "--match-n", "2", "--output-json", str(out)])
        == 0
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    summary = payload["summaries"][0]
    assert payload["status"] == "insufficient_token_trace"
    assert payload["pld_equivalence_certified"] is False
    assert summary["evidence_label"] == "hash_only"
    assert summary["hash_only_rows"] == 1
    assert summary["pld_equivalence_certified"] is False


def test_equivalence_checker_does_not_certify_matching_hash_only_compare(tmp_path):
    expected = tmp_path / "expected.jsonl"
    actual = tmp_path / "actual.jsonl"
    row = {"task_id": "task", "step": 0, "proposal_hash": "abc123"}
    expected.write_text(json.dumps(row) + "\n", encoding="utf-8")
    actual.write_text(json.dumps(row) + "\n", encoding="utf-8")
    out = tmp_path / "summary.json"

    assert (
        equivalence_main(
            [
                "--expected-trace",
                str(expected),
                "--actual-trace",
                str(actual),
                "--output-json",
                str(out),
            ]
        )
        == 0
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    summary = payload["summaries"][0]
    assert payload["status"] == "insufficient_token_trace"
    assert summary["evidence_label"] == "hash_only"
    assert summary["pld_equivalence_certified"] is False


def test_equivalence_checker_token_trace_mismatch_fails(tmp_path):
    expected = tmp_path / "expected.jsonl"
    actual = tmp_path / "actual.jsonl"
    expected.write_text(
        json.dumps({"task_id": "task", "step": 0, "proposal_token_ids": [9, 8]}) + "\n",
        encoding="utf-8",
    )
    actual.write_text(
        json.dumps({"task_id": "task", "step": 0, "proposal_token_ids": [9, 7]}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "summary.json"

    assert (
        equivalence_main(
            [
                "--expected-trace",
                str(expected),
                "--actual-trace",
                str(actual),
                "--output-json",
                str(out),
            ]
        )
        == 1
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    summary = payload["summaries"][0]
    assert payload["status"] == "mismatch"
    assert payload["pld_equivalence_certified"] is False
    assert summary["evidence_label"] == "token_trace"
    assert summary["mismatches"]


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("patched_ngram_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    original_spec_from_file_location = importlib.util.spec_from_file_location

    def spec_from_file_location(name, location, *args, **kwargs):
        nested_spec = original_spec_from_file_location(name, location, *args, **kwargs)
        if nested_spec is not None and nested_spec.loader is not None:
            return nested_spec
        loader = importlib.machinery.SourceFileLoader(name, str(location))
        return importlib.util.spec_from_loader(name, loader, origin=str(location))

    importlib.util.spec_from_file_location = spec_from_file_location
    try:
        spec.loader.exec_module(module)
    finally:
        importlib.util.spec_from_file_location = original_spec_from_file_location
    return module


def test_patch_mode_off_overrides_legacy_patch_flag(tmp_path, monkeypatch):
    module = _patched_dummy_module(tmp_path)
    proposer = module.NgramProposer()

    monkeypatch.setenv("VANTAGE_PLD_PATCH", "1")
    monkeypatch.setenv("VANTAGE_PLD_PATCH_MODE", "off")

    assert proposer.propose([[0]], [6], [[1, 2, 9, 8, 1, 2]]) == [[101]]


def test_trace_delegate_modes_do_not_change_native_tokens(tmp_path, monkeypatch):
    module = _patched_dummy_module(tmp_path)
    proposer = module.NgramProposer()
    trace = tmp_path / "delegate_trace.jsonl"

    for mode in ("passthrough_trace", "native_fixed_n"):
        trace.unlink(missing_ok=True)
        monkeypatch.setenv("VANTAGE_PLD_PATCH_MODE", mode)
        monkeypatch.setenv("VANTAGE_PLD_TRACE_PATH", str(trace))
        monkeypatch.setenv("VANTAGE_PLD_TRACE_TOKENS", "1")

        assert proposer.propose([[0]], [6], [[1, 2, 9, 8, 1, 2]]) == [[101]]
        row = _trace_rows(trace)[0]
        assert row["mode"] == mode
        assert row["delegated_original"] is True
        assert row["request_count"] == 1
        assert row["prefix_len"] == 6
        assert row["proposal_len"] == 1
        assert row["proposal_token_ids"] == [101]
        assert row["cap"] == 2
        assert row["hit"] is True
        assert row["miss"] is False
        assert row["elapsed_us"] >= 0


def test_legacy_patch_flag_defaults_to_pld_python_mode(tmp_path, monkeypatch):
    module = _patched_dummy_module(tmp_path)
    proposer = module.NgramProposer()
    trace = tmp_path / "pld_python_trace.jsonl"

    monkeypatch.setenv("VANTAGE_PLD_PATCH", "1")
    monkeypatch.delenv("VANTAGE_PLD_PATCH_MODE", raising=False)
    monkeypatch.setenv("VANTAGE_PLD_MATCH_N", "2")
    monkeypatch.setenv("VANTAGE_PLD_TRACE_PATH", str(trace))
    monkeypatch.setenv("VANTAGE_PLD_TRACE_SAMPLE_RATE", "1")
    monkeypatch.setenv("VANTAGE_PLD_TRACE_TOKENS", "1")

    assert proposer.propose([[0]], [6], [[1, 2, 9, 8, 1, 2]]) == [[9, 8]]
    row = _trace_rows(trace)[0]
    assert row["mode"] == "pld_python"
    assert row["request_count"] == 1
    assert row["prefix_len"] == 6
    assert row["proposal_len"] == 2
    assert row["proposal_token_ids"] == [9, 8]
    assert row["cap"] == 2
    assert row["hit"] is True
    assert row["miss"] is False


def test_pld_optimized_falls_back_to_python_with_trace_flag(tmp_path, monkeypatch):
    module = _patched_dummy_module(tmp_path)
    proposer = module.NgramProposer()
    trace = tmp_path / "optimized_trace.jsonl"

    def fake_import_module(name):
        if name == "vantage_vllm.optimized_pld":
            raise ImportError("forced missing optimized path")
        return importlib.import_module(name)

    monkeypatch.setattr(module._importlib, "import_module", fake_import_module)
    module._OPTIMIZED_MODULE = None
    module._OPTIMIZED_IMPORT_ERROR = None
    monkeypatch.setenv("VANTAGE_PLD_PATCH_MODE", "pld_optimized")
    monkeypatch.setenv("VANTAGE_PLD_MATCH_N", "2")
    monkeypatch.setenv("VANTAGE_PLD_TRACE_PATH", str(trace))
    monkeypatch.setenv("VANTAGE_PLD_TRACE_SAMPLE_RATE", "1")
    monkeypatch.setenv("VANTAGE_PLD_TRACE_TOKENS", "1")

    assert proposer.propose([[0]], [6], [[1, 2, 9, 8, 1, 2]]) == [[9, 8]]
    row = _trace_rows(trace)[0]
    assert row["mode"] == "pld_optimized"
    assert row["proposal_token_ids"] == [9, 8]
    assert row["optimized_pld_used"] is False
    assert row["optimized_pld_fallback"] is True
    assert "forced missing optimized path" in row["optimized_pld_fallback_reason"]


def test_pld_optimized_uses_optimized_module_when_available(tmp_path, monkeypatch):
    module = _patched_dummy_module(tmp_path)
    proposer = module.NgramProposer()
    trace = tmp_path / "optimized_trace.jsonl"

    monkeypatch.setenv("VANTAGE_PLD_PATCH_MODE", "pld_optimized")
    monkeypatch.setenv("VANTAGE_PLD_MATCH_N", "2")
    monkeypatch.setenv("VANTAGE_PLD_TRACE_PATH", str(trace))
    monkeypatch.setenv("VANTAGE_PLD_TRACE_SAMPLE_RATE", "1")
    monkeypatch.setenv("VANTAGE_PLD_TRACE_TOKENS", "1")

    assert proposer.propose([[0]], [6], [[1, 2, 9, 8, 1, 2]]) == [[9, 8]]
    row = _trace_rows(trace)[0]
    assert row["mode"] == "pld_optimized"
    assert row["proposal_token_ids"] == [9, 8]
    assert row["optimized_pld_used"] is True
    assert row["optimized_pld_fallback"] is False
    assert row["optimized_pld_fallback_reason"] is None


def _patched_dummy_module(tmp_path: Path):
    fake = tmp_path / "ngram_proposer.py"
    fake.write_text(
        """
class NgramProposer:
    def __init__(self):
        self.num_speculative_tokens = 2

    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, *args, **kwargs):
        return [[101]]
""",
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/patch_installed_vllm_ngram_to_vantage.py"),
            "--ngram-path",
            str(fake),
            "--backup-dir",
            str(tmp_path / "backups"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return _load_module(fake)


def _trace_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
