"""CPU-only tree-sitter parse-latency benchmark.

Runs the full sweep across embedded Python+TS samples and writes a JSON
report. Safe to run on a laptop (no GPU needed).

Usage:
    python scripts/bench_treesitter.py --output out/treesitter.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make `asts` importable when invoked via `python scripts/...` without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asts.treesitter_bench import run_sweep, summarize


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, help="Path to write JSON report")
    p.add_argument("--iters-cold", type=int, default=100)
    p.add_argument("--iters-inc", type=int, default=100)
    p.add_argument("--iters-kstep", type=int, default=50)
    p.add_argument(
        "--k-values",
        default="4,8,16",
        help="comma-separated k values for verify-step bench",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("bench_treesitter")

    k_values = tuple(int(x) for x in args.k_values.split(","))

    log.info("running tree-sitter sweep ...")
    report = run_sweep(
        iterations_cold=args.iters_cold,
        iterations_inc=args.iters_inc,
        iterations_kstep=args.iters_kstep,
        k_values=k_values,
    )
    report["summary"] = summarize(report["measurements"])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    log.info("wrote %s (%d measurements)", out, len(report["measurements"]))

    # Print a short summary
    print()
    print("=== Tree-sitter sweep summary (mean p50, microseconds) ===")
    for lang, ops in report["summary"].items():
        print(f"  {lang}:")
        for op, stats in sorted(ops.items()):
            print(
                f"    {op:<25}  p50={stats['mean_p50_us']:>9.1f} us  "
                f"p95={stats['mean_p95_us']:>9.1f} us  "
                f"p99={stats['mean_p99_us']:>9.1f} us  "
                f"(n={stats['n_samples']})"
            )


if __name__ == "__main__":
    main()
