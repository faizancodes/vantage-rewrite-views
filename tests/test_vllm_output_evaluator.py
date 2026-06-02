from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import evaluate_vllm_outputs as evaluator


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_evaluator_compares_flat_vllm_outputs_to_gold(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    vllm = tmp_path / "vllm.jsonl"
    out_dir = tmp_path / "eval"
    write_jsonl(
        manifest,
        [
            {"task_id": "a", "deterministic_target": "hello", "tokens": [1, 2], "finish_reason": "stop"},
            {"task_id": "b", "deterministic_target": "world", "tokens": [3, 4], "finish_reason": "length"},
        ],
    )
    write_jsonl(
        vllm,
        [
            {"task_id": "a", "text": "hello", "token_ids": [1, 2], "finish_reason": "stop"},
            {"task_id": "b", "text": "word", "token_ids": [3], "finish_reason": "stop"},
        ],
    )

    rc = evaluator.main(
        [
            "--manifest",
            str(manifest),
            "--output-jsonl",
            f"vllm={vllm}",
            "--output-dir",
            str(out_dir),
        ]
    )

    assert rc == 0
    summary = json.loads((out_dir / "eval_summary.json").read_text(encoding="utf-8"))
    method = summary["methods"]["vllm"]
    assert method["present_tasks"] == 2
    assert method["emitted_tokens"] == 3
    assert method["exact_matches"] == 1
    assert method["length_mismatches"] == 1
    assert method["finish_mismatches"] == 1
    rows = read_jsonl(out_dir / "eval_per_task.jsonl")
    b = next(row for row in rows if row["task_id"] == "b")
    assert b["methods"]["vllm"]["token_edit_distance"] == 1
    assert b["methods"]["vllm"]["first_mismatch_position"] == 1
    assert b["methods"]["vllm"]["normalized_char_edit_distance"] == 0.2


def test_evaluator_handles_nested_local_harness_outputs_and_pairwise(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    completions = tmp_path / "completions.jsonl"
    out_dir = tmp_path / "eval"
    write_jsonl(
        manifest,
        [
            {"task_id": "a", "target_text": "alpha"},
            {"task_id": "b", "target_text": "beta"},
        ],
    )
    write_jsonl(
        completions,
        [
            {
                "task_id": "a",
                "outputs": {
                    "vanilla": {"text": "alpha", "tokens": [10, 11], "finish_reason": "eos"},
                    "hf": {"text": "alpha", "tokens": [10, 11], "finish_reason": "eos"},
                },
            },
            {
                "task_id": "b",
                "outputs": {
                    "vanilla": {"text": "beta", "tokens": [20], "finish_reason": "eos"},
                    "hf": {"text": "bet", "tokens": [20, 21], "finish_reason": "length"},
                },
            },
        ],
    )

    result = evaluator.evaluate(
        evaluator.load_gold_manifest(manifest),
        evaluator.merge_outputs([(None, completions)]),
    )
    evaluator.write_reports(result, out_dir)

    summary = json.loads((out_dir / "eval_summary.json").read_text(encoding="utf-8"))
    assert set(summary["methods"]) == {"hf", "vanilla"}
    assert summary["methods"]["vanilla"]["exact_match_rate"] == 1.0
    pair = summary["pairwise_exact_task_matches"]["hf__vs__vanilla"]
    assert pair["compared_tasks"] == 2
    assert pair["exact_task_matches"] == 1
    assert pair["length_mismatches"] == 1
    assert pair["finish_mismatches"] == 1


def test_evaluator_reports_lightweight_quality_proxies(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    vllm = tmp_path / "vllm.jsonl"
    out_dir = tmp_path / "eval"
    table_dir = tmp_path / "tables"
    write_jsonl(
        manifest,
        [
            {
                "task_id": "a",
                "prompt": "def keep():\n    return 1\n",
                "deterministic_target": "def keep():\n    return 1\n",
                "tokens": [1, 2, 3, 4],
            },
            {
                "task_id": "b",
                "prompt": "x = 1\ny = 2\n",
                "deterministic_target": "x = 1\ny = 3\n",
                "tokens": [5, 6],
            },
            {
                "task_id": "c",
                "prompt": "alpha\n",
                "deterministic_target": "alpha\nbeta\n",
                "tokens": [7, 8, 9, 10],
            },
        ],
    )
    write_jsonl(
        vllm,
        [
            {"task_id": "a", "output_text": "def keep():\n    return 1\n", "output_token_ids": [1, 2, 3, 4], "finish_reason": "stop"},
            {"task_id": "b", "output_text": "x = 1\ny = 2\n", "output_token_ids": [5, 11], "finish_reason": "length"},
            {"task_id": "c", "output_text": "alpha\n", "output_token_ids": [7], "finish_reason": "max_tokens"},
        ],
    )

    result = evaluator.evaluate(
        evaluator.load_gold_manifest(manifest),
        evaluator.merge_outputs([("vllm", vllm)]),
    )
    evaluator.write_reports(result, out_dir)
    evaluator.write_tables(result, table_dir)

    summary = json.loads((out_dir / "eval_summary.json").read_text(encoding="utf-8"))
    method = summary["methods"]["vllm"]
    assert method["stop_tasks"] == 1
    assert method["truncated_tasks"] == 2
    assert method["truncation_rate"] == 2 / 3
    assert method["emitted_token_stats"]["p50"] == 2.0
    assert method["emitted_token_stats"]["p95"] == 3.8
    assert method["exact_line_edit_distance_tasks"] == 3
    assert method["normalized_line_distance_proxy"]["count"] == 3
    assert method["source_copy_line_overlap_proxy_tasks"] == 3
    assert method["source_copy_line_overlap_proxy"]["mean"] == 1.0
    assert method["output_gold_token_delta"]["min"] == -3.0

    rows = read_jsonl(out_dir / "eval_per_task.jsonl")
    b = next(row for row in rows if row["task_id"] == "b")
    b_metrics = b["methods"]["vllm"]
    assert b_metrics["line_edit_distance"] == 1
    assert b_metrics["normalized_line_edit_distance"] == 0.5
    assert b_metrics["source_copy_line_overlap_proxy"] == 1.0
    assert b_metrics["is_truncated"] is True
    assert b_metrics["output_gold_token_delta"] == 0

    quality_table = (table_dir / "external_baseline_quality.md").read_text(encoding="utf-8")
    assert "Line distance proxy mean" in quality_table
    assert "Source-copy line-overlap proxy mean" in quality_table
