"""GPU model forward-pass benchmark for target+draft pair.

Loads Qwen2.5-Coder-7B (target) and Qwen2.5-Coder-0.5B (draft) onto cuda
and measures: prefill, ar_step, verify_kstep at multiple k values across
two prefix lengths.

Usage:
    python scripts/bench_models.py --output out/models.json \\
        --target Qwen/Qwen2.5-Coder-7B --draft Qwen/Qwen2.5-Coder-0.5B
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asts.model_bench import run_sweep


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    p.add_argument("--draft", default="Qwen/Qwen2.5-Coder-0.5B")
    p.add_argument("--prefix-lens", default="512,2048")
    p.add_argument("--k-values", default="4,8,16")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument(
        "--attn-impl",
        default="sdpa",
        help="sdpa | flash_attention_2 | eager (default sdpa is universally available)",
    )
    p.add_argument("--ar-iters", type=int, default=50)
    p.add_argument("--verify-iters", type=int, default=30)
    p.add_argument("--ar-warmup", type=int, default=10)
    p.add_argument("--verify-warmup", type=int, default=5)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("bench_models")

    prefix_lens = tuple(int(x) for x in args.prefix_lens.split(","))
    k_values = tuple(int(x) for x in args.k_values.split(","))

    log.info(
        "running model sweep: target=%s draft=%s prefix_lens=%s k_values=%s",
        args.target, args.draft, prefix_lens, k_values,
    )

    report = run_sweep(
        target_id=args.target,
        draft_id=args.draft,
        prefix_lens=prefix_lens,
        k_values=k_values,
        dtype=args.dtype,
        attn_impl=args.attn_impl,
        ar_iters=args.ar_iters,
        verify_iters=args.verify_iters,
        ar_warmup=args.ar_warmup,
        verify_warmup=args.verify_warmup,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    log.info("wrote %s (%d measurements)", out, len(report["measurements"]))

    # Short summary
    print()
    print(f"=== Model sweep summary (p50, microseconds) ===")
    for m in report["measurements"]:
        print(
            f"  {m['model_id']:<35} {m['operation']:<14} "
            f"prefix={m['prefix_tokens']:<5} k={m['k']:<3} "
            f"p50={m['stats_us']['p50']:>9.1f} us  "
            f"p95={m['stats_us']['p95']:>9.1f} us"
        )


if __name__ == "__main__":
    main()
