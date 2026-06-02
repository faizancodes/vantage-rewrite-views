"""Tree-sitter parse-latency microbenchmark.

We measure three patterns that map directly to ASTS-Spec's decode-loop cost:

  1. cold_parse: full reparse of a prefix from scratch (no prior tree). This is
     the upper bound and the fallback path.
  2. incremental_1tok: append 1 byte (proxy for 1 token) to a parsed prefix,
     call tree.edit(), reparse with the old tree. This is the cost paid AFTER
     each greedy/AR step in a parser-in-the-loop decoder.
  3. incremental_kstep: append K bytes in one shot (proxy for verifying a
     K-token speculative draft subtree). This is the cost paid per spec-decode
     step.

For each, we sweep prefix sizes (~10 / ~100 / ~1000 / ~5000 tokens) and
languages (Python, TypeScript), report p50/p95/p99 in microseconds.

We also probe whether incremental parsing actually buys anything over cold
parse (it should — that's tree-sitter's whole pitch).
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass

from .corpus import Sample, get_samples


# ---------------------------------------------------------------------------
# Tree-sitter loader (defer import so this module can be imported without
# tree_sitter installed, e.g. for unit tests of the data classes)
# ---------------------------------------------------------------------------


def _get_parser(language: str):
    from tree_sitter_language_pack import get_parser

    if language == "typescript":
        # tree-sitter-language-pack exposes both `typescript` and `tsx`. Use
        # plain typescript for `.ts` content, no JSX.
        return get_parser("typescript")
    return get_parser(language)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _byte_to_point(source: bytes, byte_offset: int) -> tuple[int, int]:
    """Convert a byte offset to (row, column). tree-sitter uses byte-columns."""
    if byte_offset <= 0:
        return (0, 0)
    head = source[:byte_offset]
    row = head.count(b"\n")
    last_nl = head.rfind(b"\n")
    col = byte_offset - (last_nl + 1)
    return (row, col)


def _percentiles(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(samples)
    n = len(s)

    def pct(p: float) -> float:
        # Linear-interpolation percentile (matches numpy.percentile default).
        k = (n - 1) * p
        f = int(k)
        c = min(f + 1, n - 1)
        return s[f] + (s[c] - s[f]) * (k - f)

    return {
        "n": n,
        "mean": statistics.fmean(s),
        "p50": pct(0.50),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "min": s[0],
        "max": s[-1],
    }


# ---------------------------------------------------------------------------
# Benchmark primitives
# ---------------------------------------------------------------------------


@dataclass
class ParseMeasurement:
    sample_name: str
    language: str
    operation: str  # "cold" | "incremental_1tok" | "incremental_kstep"
    prefix_bytes: int
    k: int  # for incremental_kstep; 1 for the others
    iterations: int
    stats_us: dict  # mean/p50/p95/p99/min/max in microseconds


def bench_cold_parse(
    sample: Sample,
    iterations: int = 100,
    warmup: int = 10,
) -> ParseMeasurement:
    parser = _get_parser(sample.language)
    src = sample.code.encode("utf-8")

    times_us: list[float] = []
    for i in range(iterations + warmup):
        t0 = time.perf_counter_ns()
        parser.parse(src)
        t1 = time.perf_counter_ns()
        if i >= warmup:
            times_us.append((t1 - t0) / 1000.0)

    return ParseMeasurement(
        sample_name=sample.name,
        language=sample.language,
        operation="cold",
        prefix_bytes=len(src),
        k=0,
        iterations=iterations,
        stats_us=_percentiles(times_us),
    )


def bench_incremental_1tok(
    sample: Sample,
    iterations: int = 100,
    warmup: int = 5,
    chunk_size: int = 1,
) -> ParseMeasurement:
    """Cut the sample to length L - iterations, then append `chunk_size` bytes
    at a time. Measure each incremental reparse.
    """
    parser = _get_parser(sample.language)
    full = sample.code.encode("utf-8")
    if len(full) < (iterations + warmup) * chunk_size + 16:
        # Not enough room for the full append schedule; tighten iterations.
        iterations = max(1, (len(full) - 16) // chunk_size - warmup)

    base_len = len(full) - (iterations + warmup) * chunk_size
    base = full[:base_len]
    tree = parser.parse(base)
    cur_src = base

    times_us: list[float] = []
    for step in range(iterations + warmup):
        appended = full[base_len + step * chunk_size : base_len + (step + 1) * chunk_size]
        if not appended:
            break
        old_end_byte = len(cur_src)
        new_src = cur_src + appended
        new_end_byte = len(new_src)
        old_end_point = _byte_to_point(cur_src, old_end_byte)
        new_end_point = _byte_to_point(new_src, new_end_byte)

        # Notify the tree of the edit (insertion at the very end).
        tree.edit(
            start_byte=old_end_byte,
            old_end_byte=old_end_byte,
            new_end_byte=new_end_byte,
            start_point=old_end_point,
            old_end_point=old_end_point,
            new_end_point=new_end_point,
        )

        t0 = time.perf_counter_ns()
        tree = parser.parse(new_src, tree)
        t1 = time.perf_counter_ns()
        if step >= warmup:
            times_us.append((t1 - t0) / 1000.0)

        cur_src = new_src

    return ParseMeasurement(
        sample_name=sample.name,
        language=sample.language,
        operation="incremental_1tok",
        prefix_bytes=base_len,
        k=chunk_size,
        iterations=len(times_us),
        stats_us=_percentiles(times_us),
    )


def bench_incremental_kstep(
    sample: Sample,
    k: int = 8,
    iterations: int = 50,
    warmup: int = 5,
) -> ParseMeasurement:
    """Append k bytes in a single edit, K times. Models verifying a k-token
    speculative draft as one parse step.
    """
    m = bench_incremental_1tok(sample, iterations=iterations, warmup=warmup, chunk_size=k)
    # Re-tag: the wrapped call labels itself incremental_1tok; this is the
    # k-step variant.
    return ParseMeasurement(
        sample_name=m.sample_name,
        language=m.language,
        operation="incremental_kstep",
        prefix_bytes=m.prefix_bytes,
        k=k,
        iterations=m.iterations,
        stats_us=m.stats_us,
    )


# ---------------------------------------------------------------------------
# Driver: run the full sweep
# ---------------------------------------------------------------------------


def run_sweep(
    iterations_cold: int = 100,
    iterations_inc: int = 100,
    iterations_kstep: int = 50,
    k_values: tuple[int, ...] = (4, 8, 16),
) -> dict:
    """Run cold + incremental + k-step benchmarks across all samples.

    Returns a dict of measurements keyed by (sample, operation) suitable for
    JSON serialization and downstream analysis.
    """
    samples = get_samples()
    results: list[dict] = []

    for s in samples:
        cold = bench_cold_parse(s, iterations=iterations_cold)
        results.append(asdict(cold))

        inc1 = bench_incremental_1tok(s, iterations=iterations_inc)
        results.append(asdict(inc1))

        for k in k_values:
            kstep = bench_incremental_kstep(s, k=k, iterations=iterations_kstep)
            results.append(asdict(kstep))

    return {
        "schema": "asts-spec/treesitter_bench/v1",
        "measurements": results,
    }


def summarize(measurements: list[dict]) -> dict:
    """Compute per-language, per-operation aggregates suitable for the verdict.

    Returns: {language: {operation: {p50_us, p95_us, ...}}} averaged across
    samples (with each sample contributing its own p50/p95). For ops where
    k is meaningful (incremental_kstep), the k value is appended to the key.
    """
    by_lang_op: dict[tuple[str, str, int], list[dict]] = {}
    for m in measurements:
        key = (m["language"], m["operation"], m["k"])
        by_lang_op.setdefault(key, []).append(m["stats_us"])

    summary: dict = {}
    for (lang, op, k), stats_list in by_lang_op.items():
        # cold parse and 1-token incremental have no meaningful k variant;
        # only incremental_kstep does.
        if op == "incremental_kstep":
            agg_key = f"{op}_k{k}"
        else:
            agg_key = op
        lang_d = summary.setdefault(lang, {})
        lang_d[agg_key] = {
            "n_samples": len(stats_list),
            "mean_p50_us": statistics.fmean(s["p50"] for s in stats_list),
            "mean_p95_us": statistics.fmean(s["p95"] for s in stats_list),
            "mean_p99_us": statistics.fmean(s["p99"] for s in stats_list),
            "min_p50_us": min(s["p50"] for s in stats_list),
            "max_p99_us": max(s["p99"] for s in stats_list),
        }
    return summary
