"""Smoke tests for the tree-sitter benchmark.

Runs with --iters small enough to finish in <2s; primarily verifies the
benchmark machinery doesn't crash and produces well-formed output.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tree_sitter_language_pack")

from asts.corpus import get_sample, get_samples
from asts.treesitter_bench import (
    bench_cold_parse,
    bench_incremental_1tok,
    bench_incremental_kstep,
    run_sweep,
    summarize,
    _byte_to_point,
)


def test_byte_to_point_basic():
    src = b"abc\ndef\nghi"
    assert _byte_to_point(src, 0) == (0, 0)
    assert _byte_to_point(src, 3) == (0, 3)
    assert _byte_to_point(src, 4) == (1, 0)  # immediately after first \n
    assert _byte_to_point(src, 7) == (1, 3)
    assert _byte_to_point(src, 8) == (2, 0)
    assert _byte_to_point(src, 11) == (2, 3)


def test_corpus_has_both_languages():
    py = get_samples("python")
    ts = get_samples("typescript")
    assert len(py) >= 3
    assert len(ts) >= 3
    # Sizes should span at least 2 orders of magnitude
    py_sizes = [s.approx_tokens for s in py]
    assert max(py_sizes) > min(py_sizes) * 50


def test_cold_parse_runs():
    s = get_sample("py-small")
    m = bench_cold_parse(s, iterations=5, warmup=1)
    assert m.operation == "cold"
    assert m.stats_us["n"] == 5
    assert m.stats_us["p50"] > 0


def test_incremental_1tok_runs():
    s = get_sample("py-medium")
    m = bench_incremental_1tok(s, iterations=10, warmup=2)
    assert m.operation == "incremental_1tok"
    assert m.k == 1
    assert m.stats_us["n"] >= 1
    assert m.stats_us["p50"] > 0


def test_incremental_kstep_runs():
    s = get_sample("ts-small")
    m = bench_incremental_kstep(s, k=4, iterations=5, warmup=1)
    assert m.operation == "incremental_kstep"
    assert m.k == 4
    assert m.stats_us["p50"] > 0


def test_run_sweep_returns_well_formed_report():
    report = run_sweep(
        iterations_cold=5,
        iterations_inc=5,
        iterations_kstep=3,
        k_values=(4,),
    )
    assert "schema" in report
    assert len(report["measurements"]) > 0

    summary = summarize(report["measurements"])
    assert "python" in summary
    assert "typescript" in summary
    assert "cold" in summary["python"]
    assert "incremental_1tok" in summary["python"]
    assert "incremental_kstep_k4" in summary["python"]
