"""Test the verdict computation with synthetic measurements."""

from __future__ import annotations

from asts.analysis import compute_verdict


def _ts_meas(language: str, op: str, k: int, p50_us: float) -> dict:
    return {
        "sample_name": f"{language}-test",
        "language": language,
        "operation": op,
        "prefix_bytes": 1000,
        "k": k,
        "iterations": 50,
        "stats_us": {"n": 50, "mean": p50_us, "p50": p50_us, "p95": p50_us * 1.2, "p99": p50_us * 1.5, "min": p50_us * 0.8, "max": p50_us * 2},
    }


def _model_meas(model_id: str, op: str, k: int, prefix: int, p50_us: float) -> dict:
    return {
        "model_id": model_id,
        "operation": op,
        "prefix_tokens": prefix,
        "k": k,
        "iterations": 30,
        "stats_us": {"n": 30, "mean": p50_us, "p50": p50_us, "p95": p50_us * 1.2, "p99": p50_us * 1.5, "min": p50_us * 0.8, "max": p50_us * 2},
        "extra": {"dtype": "bfloat16", "attn_impl": "sdpa"},
    }


def test_verdict_proceed_when_parse_is_cheap():
    """Parse 100us, target 50000us, draft 5000us, verify 60000us — easy proceed."""
    ts_results = {
        "schema": "asts-spec/treesitter_bench/v1",
        "measurements": [
            _ts_meas("python", "cold", 0, 500),
            _ts_meas("python", "incremental_1tok", 1, 100),
            _ts_meas("python", "incremental_kstep", 8, 200),
            _ts_meas("typescript", "cold", 0, 800),
            _ts_meas("typescript", "incremental_1tok", 1, 150),
            _ts_meas("typescript", "incremental_kstep", 8, 300),
        ],
    }
    model_results = {
        "schema": "asts-spec/model_bench/v1",
        "target_id": "qwen-7b",
        "draft_id": "qwen-0.5b",
        "dtype": "bfloat16",
        "attn_impl": "sdpa",
        "measurements": [
            _model_meas("qwen-7b", "ar_step", 1, 2048, 50_000),
            _model_meas("qwen-7b", "verify_kstep", 8, 2048, 60_000),
            _model_meas("qwen-0.5b", "ar_step", 1, 2048, 5_000),
        ],
    }
    v = compute_verdict(ts_results, model_results, k=8)
    assert v["verdict"] == "PROCEED"
    assert v["verdict_per_language"]["python"] == "PROCEED"
    assert v["verdict_per_language"]["typescript"] == "PROCEED"


def test_verdict_kill_when_parse_dominates():
    """Parse 200000us — way more than the savings from spec decode."""
    ts_results = {
        "schema": "asts-spec/treesitter_bench/v1",
        "measurements": [
            _ts_meas("python", "cold", 0, 500_000),
            _ts_meas("python", "incremental_1tok", 1, 200_000),
            _ts_meas("python", "incremental_kstep", 8, 200_000),
            _ts_meas("typescript", "cold", 0, 500_000),
            _ts_meas("typescript", "incremental_1tok", 1, 200_000),
            _ts_meas("typescript", "incremental_kstep", 8, 200_000),
        ],
    }
    model_results = {
        "schema": "asts-spec/model_bench/v1",
        "target_id": "qwen-7b",
        "draft_id": "qwen-0.5b",
        "dtype": "bfloat16",
        "attn_impl": "sdpa",
        "measurements": [
            _model_meas("qwen-7b", "ar_step", 1, 2048, 50_000),
            _model_meas("qwen-7b", "verify_kstep", 8, 2048, 60_000),
            _model_meas("qwen-0.5b", "ar_step", 1, 2048, 5_000),
        ],
    }
    v = compute_verdict(ts_results, model_results, k=8)
    # If parse alone is 200ms vs 50ms target ar_step, spec is much slower
    # → KILL or at least not PROCEED.
    assert v["verdict"] in {"KILL", "PIVOT"}


def test_verdict_handles_missing_data():
    ts_results = {
        "schema": "asts-spec/treesitter_bench/v1",
        "measurements": [],
    }
    model_results = {
        "schema": "asts-spec/model_bench/v1",
        "target_id": "qwen-7b",
        "draft_id": "qwen-0.5b",
        "dtype": "bfloat16",
        "attn_impl": "sdpa",
        "measurements": [],
    }
    v = compute_verdict(ts_results, model_results, k=8)
    assert v["verdict"] == "ERROR"
