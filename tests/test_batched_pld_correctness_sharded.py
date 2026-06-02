from scripts.validate_batched_pld_correctness_sharded import (
    CompletedShard,
    aggregate_shard_reports,
    next_fallback_size,
    split_ranges,
    validate_task_coverage,
)


def _report(batch_sizes, matches, mismatches=0):
    rows = []
    for batch in batch_sizes:
        total = matches + mismatches
        rows.append(
            {
                "batch_size": batch,
                "matches": matches,
                "mismatches": mismatches,
                "decoded_output_matches": matches,
                "decoded_output_mismatches": mismatches,
                "finish_mismatch_count": 0,
                "generated_length_matches": matches,
                "generated_length_mismatches": mismatches,
                "mismatch_examples": [
                    {
                        "task_id": "task/bad",
                        "first_diff_index": 3,
                        "baseline_token_id": 10,
                        "batched_token_id": 11,
                    }
                ]
                if mismatches
                else [],
                "metrics": {},
            }
        )
    return {"rows": rows}


def test_split_ranges_covers_all_task_indices_once():
    ranges = split_ranges(500, 50)
    assert len(ranges) == 10
    assert ranges[0] == (0, 50)
    assert ranges[-1] == (450, 500)
    covered = [idx for start, end in ranges for idx in range(start, end)]
    assert covered == list(range(500))


def test_fallback_shard_sizes_eventually_reach_single_task():
    assert next_fallback_size(50) == 25
    assert next_fallback_size(26) == 25
    assert next_fallback_size(25) == 10
    assert next_fallback_size(11) == 10
    assert next_fallback_size(10) == 5
    assert next_fallback_size(6) == 5
    assert next_fallback_size(5) == 1
    assert next_fallback_size(2) == 1
    assert next_fallback_size(1) is None


def test_task_coverage_requires_exactly_once():
    shards = [
        CompletedShard("a", 0, 2, 2, "", ["t0", "t1"], _report([1], 2)),
        CompletedShard("b", 2, 4, 2, "", ["t2", "t3"], _report([1], 2)),
    ]
    report = validate_task_coverage(shards, ["t0", "t1", "t2", "t3"])
    assert report["covers_all_tasks_exactly_once"]

    duplicate = [
        CompletedShard("a", 0, 2, 2, "", ["t0", "t1"], _report([1], 2)),
        CompletedShard("b", 2, 4, 2, "", ["t1", "t3"], _report([1], 2)),
    ]
    report = validate_task_coverage(duplicate, ["t0", "t1", "t2", "t3"])
    assert not report["covers_all_tasks_exactly_once"]
    assert report["duplicate_task_ids"] == ["t1"]
    assert report["missing_task_ids"] == ["t2"]


def test_aggregate_report_sums_all_batches_and_mismatch_metadata():
    completed = [
        CompletedShard("000_0000_0002", 0, 2, 2, "", ["t0", "t1"], _report([1, 4, 8], 2)),
        CompletedShard(
            "001_0002_0004",
            2,
            4,
            2,
            "",
            ["t2", "t3"],
            _report([1, 4, 8], 1, mismatches=1),
            fallback_from=50,
        ),
    ]
    agg = aggregate_shard_reports(
        completed,
        batch_sizes=[1, 4, 8],
        total_tasks=4,
        initial_shard_size=50,
        oom_attempts=[],
        expected_task_ids=["t0", "t1", "t2", "t3"],
    )
    assert set(agg["batch_results"]) == {"1", "4", "8"}
    for batch in ("1", "4", "8"):
        assert agg["batch_results"][batch]["tasks"] == 4
        assert agg["batch_results"][batch]["exact_token_id_matches"] == 3
        assert agg["batch_results"][batch]["token_id_mismatches"] == 1
        assert not agg["batch_results"][batch]["exact"]
    assert agg["mismatch_count"] == 3
    assert agg["first_10_mismatch_examples"][0]["shard_id"] == "001_0002_0004"
    assert agg["first_10_mismatch_examples"][0]["batch_size"] in {1, 4, 8}


def test_aggregate_exact_when_every_batch_matches_every_task():
    completed = [
        CompletedShard("000_0000_0002", 0, 2, 2, "", ["t0", "t1"], _report([1, 4, 8], 2)),
        CompletedShard("001_0002_0004", 2, 4, 2, "", ["t2", "t3"], _report([1, 4, 8], 2)),
    ]
    agg = aggregate_shard_reports(
        completed,
        batch_sizes=[1, 4, 8],
        total_tasks=4,
        initial_shard_size=50,
        oom_attempts=[],
        expected_task_ids=["t0", "t1", "t2", "t3"],
    )
    assert agg["all_exact"]
    assert agg["coverage"]["covers_all_tasks_exactly_once"]
    for batch in ("1", "4", "8"):
        assert agg["batch_results"][batch]["exact"]
