#!/usr/bin/env python3
"""Microbenchmark pure Python versus optimized full-prefix PLD lookup.

This is a local proposer-policy benchmark only. It does not run vLLM and must
not be interpreted as end-to-end generation throughput.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import numpy as np
except Exception:  # pragma: no cover - optional local dependency
    np = None  # type: ignore[assignment]

from vantage_vllm.optimized_pld import find_full_prefix_pld_proposal, numba_available
from vantage_vllm.pld_lookup import find_pld_proposal


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-count", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--min-prefix-len", type=int, default=128)
    parser.add_argument("--max-prefix-len", type=int, default=4096)
    parser.add_argument("--match-n", type=int, default=10)
    parser.add_argument("--max-draft-len", type=int, default=128)
    parser.add_argument("--cap", type=int, default=16)
    parser.add_argument("--tie-break", choices=["latest", "earliest"], default="latest")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prefixes = build_prefixes(args)

    # Warm the optional Numba path before timing.
    if np is not None and numba_available() and prefixes:
        find_full_prefix_pld_proposal(
            np.asarray(prefixes[0], dtype=np.int64),
            match_n=args.match_n,
            max_draft_len=args.max_draft_len,
            cap=args.cap,
            tie_break=args.tie_break,
            prefer_numba=True,
        )

    validate_equivalence(prefixes, args)

    rows = [
        bench("pure_pld_lookup", prefixes, args, run_pure),
        bench("optimized_python", prefixes, args, run_optimized_python),
    ]
    if np is not None and numba_available():
        rows.append(bench("optimized_numba", prefixes, args, run_optimized_numba))

    payload = {
        "schema": "vantage_pld_kernel_microbench_v1",
        "config": vars(args),
        "numba_available": bool(numba_available()),
        "numpy_available": np is not None,
        "rows": rows,
        "notes": [
            "Synthetic proposer lookup only; not an end-to-end vLLM benchmark.",
            "Prefixes include injected latest-match hits and random no-hit cases.",
        ],
    }
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output_md:
        path = Path(args.output_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(markdown(payload)) + "\n", encoding="utf-8")
    return 0


def build_prefixes(args: argparse.Namespace) -> list[list[int]]:
    rng = random.Random(args.seed)
    prefixes: list[list[int]] = []
    for i in range(args.prefix_count):
        length = rng.randint(args.min_prefix_len, args.max_prefix_len)
        prefix = [rng.randrange(0, 50000) for _ in range(length)]
        if length >= args.match_n * 2 + 8 and i % 2 == 0:
            # Inject a latest valid hit with a nonempty continuation.
            source_start = rng.randint(0, length - args.match_n * 2 - 2)
            key = [rng.randrange(50000, 50100) for _ in range(args.match_n)]
            continuation_len = min(args.cap or args.max_draft_len, args.max_draft_len, 16)
            continuation = [rng.randrange(60000, 60100) for _ in range(max(1, continuation_len))]
            prefix[source_start : source_start + args.match_n] = key
            prefix[
                source_start + args.match_n : source_start + args.match_n + len(continuation)
            ] = continuation
            prefix[-args.match_n :] = key
        prefixes.append(prefix)
    return prefixes


def bench(
    name: str,
    prefixes: list[list[int]],
    args: argparse.Namespace,
    fn: Any,
) -> dict[str, Any]:
    elapsed_values: list[float] = []
    hits = 0
    proposal_tokens = 0
    for _ in range(args.repeats):
        start = time.perf_counter()
        for prefix in prefixes:
            result = fn(prefix, args)
            if result is not None:
                hits += 1
                proposal_tokens += len(result)
        elapsed_values.append(time.perf_counter() - start)
    calls = len(prefixes) * args.repeats
    total_elapsed = sum(elapsed_values)
    return {
        "name": name,
        "calls": calls,
        "hits": hits,
        "hit_rate": hits / calls if calls else None,
        "proposal_tokens": proposal_tokens,
        "elapsed_seconds_mean": statistics.fmean(elapsed_values),
        "elapsed_seconds_min": min(elapsed_values),
        "calls_per_second_mean": calls / total_elapsed if total_elapsed > 0 else None,
        "microseconds_per_call_mean": (total_elapsed / calls) * 1_000_000.0 if calls else None,
    }


def run_pure(prefix: list[int], args: argparse.Namespace) -> list[int] | None:
    result = find_pld_proposal(
        prefix[: -args.match_n],
        prefix[-args.match_n :],
        match_n=args.match_n,
        max_draft_len=args.max_draft_len,
        cap=args.cap,
        tie_break=args.tie_break,
    )
    return result.tokens if result is not None else None


def run_optimized_python(prefix: list[int], args: argparse.Namespace) -> list[int] | None:
    result = find_full_prefix_pld_proposal(
        prefix,
        match_n=args.match_n,
        max_draft_len=args.max_draft_len,
        cap=args.cap,
        tie_break=args.tie_break,
        prefer_numba=False,
    )
    return result.tokens if result is not None else None


def run_optimized_numba(prefix: list[int], args: argparse.Namespace) -> list[int] | None:
    if np is None:
        return run_optimized_python(prefix, args)
    result = find_full_prefix_pld_proposal(
        np.asarray(prefix, dtype=np.int64),
        match_n=args.match_n,
        max_draft_len=args.max_draft_len,
        cap=args.cap,
        tie_break=args.tie_break,
        prefer_numba=True,
    )
    return result.tokens if result is not None else None


def validate_equivalence(prefixes: list[list[int]], args: argparse.Namespace) -> None:
    for index, prefix in enumerate(prefixes):
        expected = run_pure(prefix, args)
        optimized = run_optimized_python(prefix, args)
        if expected != optimized:
            raise SystemExit(
                "optimized_python mismatch at prefix "
                f"{index}: expected={expected} actual={optimized}"
            )
        if np is not None and numba_available():
            optimized_numba = run_optimized_numba(prefix, args)
            if expected != optimized_numba:
                raise SystemExit(
                    "optimized_numba mismatch at prefix "
                    f"{index}: expected={expected} actual={optimized_numba}"
                )


def markdown(payload: dict[str, Any]) -> list[str]:
    lines = [
        "# PLD Kernel Microbenchmark",
        "",
        "Synthetic proposer lookup only; not an end-to-end vLLM benchmark.",
        "",
        "| Method | Calls | Hits | Hit rate | us/call mean | Calls/s mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {name} | {calls} | {hits} | {hit_rate:.3f} | {us:.2f} | {cps:.1f} |".format(
                name=row["name"],
                calls=row["calls"],
                hits=row["hits"],
                hit_rate=row["hit_rate"] or 0.0,
                us=row["microseconds_per_call_mean"] or 0.0,
                cps=row["calls_per_second_mean"] or 0.0,
            )
        )
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
