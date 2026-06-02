#!/usr/bin/env python3
"""Small CLI wrapper for PLD-gated Lookahead sweeps.

The heavy lifting stays in ``scripts/run_eagle_eval.py``; this wrapper just
builds the method list and forwards the Lookahead knobs so local and Modal
commands stay readable.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--problem-jsonl", required=True)
    ap.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn-impl", default="sdpa")
    ap.add_argument("--lookahead-configs", default="lookahead_w8_n4_i4")
    ap.add_argument(
        "--methods",
        default="blazedit_pld_w128_n10,pld_gated_lookahead_w128_n10",
    )
    ap.add_argument("--lookahead-window", type=int, default=8)
    ap.add_argument("--lookahead-ngram", type=int, default=4)
    ap.add_argument("--lookahead-iters", type=int, default=4)
    ap.add_argument("--lookahead-max-draft", type=int, default=16)
    ap.add_argument("--lookahead-one-forward", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--pld-lookahead-router", default="rule")
    ap.add_argument("--pld-lookahead-router-threshold", type=float, default=0.3)
    ap.add_argument("--pld-lookahead-trigger", default="router_weak")
    args = ap.parse_args()

    methods = [m for m in args.methods.split(",") if m]
    methods.extend(m for m in args.lookahead_configs.split(",") if m)
    deduped = []
    seen = set()
    for method in methods:
        if method not in seen:
            deduped.append(method)
            seen.add(method)

    cmd = [
        sys.executable,
        "scripts/run_eagle_eval.py",
        "--output-dir",
        args.output_dir,
        "--problem-jsonl",
        args.problem_jsonl,
        "--target",
        args.target,
        "--n",
        str(args.n),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--methods",
        ",".join(deduped),
        "--dtype",
        args.dtype,
        "--attn-impl",
        args.attn_impl,
        "--skip-eagle-load",
        "--lookahead-window",
        str(args.lookahead_window),
        "--lookahead-ngram",
        str(args.lookahead_ngram),
        "--lookahead-iters",
        str(args.lookahead_iters),
        "--lookahead-max-draft",
        str(args.lookahead_max_draft),
        "--lookahead-one-forward" if args.lookahead_one_forward else "--no-lookahead-one-forward",
        "--pld-lookahead-router",
        args.pld_lookahead_router,
        "--pld-lookahead-router-threshold",
        str(args.pld_lookahead_router_threshold),
        "--pld-lookahead-trigger",
        args.pld_lookahead_trigger,
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
