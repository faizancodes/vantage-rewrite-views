from argparse import Namespace

import torch

from scripts.run_batched_greedy_eval import (
    GreedyRunMetrics,
    _make_greedy_tensors,
    write_report,
)
from scripts.run_batched_pld_eval import ActiveTask


def test_make_greedy_tensors_preserves_task_rows():
    tasks = [
        ActiveTask("a", "", [1, 2], [1, 2], 2, target_cache=None, target_cache_len=1),
        ActiveTask("b", "", [3, 4, 5], [3, 4, 5], 3, target_cache=None, target_cache_len=2),
    ]
    input_ids, attention_mask, position_ids, row_tokens = _make_greedy_tensors(
        tasks,
        max_cache_len=2,
        pad_id=0,
        device=torch.device("cpu"),
    )
    assert input_ids.tolist() == [[2], [5]]
    assert row_tokens == [[2], [5]]
    assert attention_mask.tolist() == [[1, 0, 1], [1, 1, 1]]
    assert position_ids.tolist() == [[1], [2]]


def test_generic_greedy_report_writes_expected_fields(tmp_path):
    args = Namespace(
        target="toy",
        dtype="fp32",
        attn="eager",
        n=2,
    )
    sequential = {
        "tokens_per_sec": 10.0,
        "model_forwards": 4,
        "tokens": 4,
        "memory_peak_gb": 0.0,
        "task_latency_summary_ms": {"p50": 1.0},
        "outputs": {"a": [1], "b": [2]},
    }
    row = GreedyRunMetrics(
        method="greedy_batched",
        batch_size=2,
        active_pool_size=2,
        n_tasks=2,
        total_new_tokens=4,
        generated_tokens_per_sec=15.0,
        model_forwards=2,
        output_match_count=2,
        output_mismatch_count=0,
        task_latency_summary_ms={"p50": 2.0},
    )
    write_report(output_dir=tmp_path, args=args, sequential=sequential, batched=[row])
    report_json = tmp_path / "report.json"
    report_md = tmp_path / "report.md"
    assert report_json.exists()
    text = report_md.read_text()
    assert "greedy_sequential" in text
    assert "greedy_batched" in text
    assert "1.500" in text
