"""Step-1 microbench for PVP (Predictive Verifier Pipelining).

PVP wants to do a B=2 verifier forward in place of a B=1 PLD verifier forward,
in exchange for one extra "future" PLD path that may or may not commit. The
economics only work if the B=2 forward is not much slower than B=1. This
script times that ratio at the shape grid the PVP design requires.

Kill criterion: if T(B=2) / T(B=1) > 1.40 at any tested prefix length, PVP
is dead and no decoder code should be written.

Shape grid (matches the PVP brief verbatim):
    (B, seq) in {(1,11), (2,11), (4,11), (8,11)}
    prefix_len in {512, 2048, 4096}
    100 reps each, warm 10.

The default target is the one used by `blazedit_pld_w128_n10` in
`scripts/run_eagle_eval.py` and `proto_app.py::run_eagle_eval`:
``Qwen/Qwen2.5-Coder-7B`` at bf16 + sdpa.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from asts.model_bench import _load_model, _make_input_ids


DEFAULT_SHAPES = ((1, 11), (2, 11), (4, 11), (8, 11))
DEFAULT_PREFIX_LENS = (512, 2048, 4096)
KILL_RATIO = 1.40


@dataclass
class Cell:
    B: int
    seq: int
    prefix_len: int
    n_iters: int
    warmup: int
    mean_us: float
    p50_us: float
    p95_us: float
    min_us: float
    max_us: float


def _percentiles(samples: list[float]) -> dict[str, float]:
    s = sorted(samples)
    n = len(s)

    def pct(p: float) -> float:
        if n == 0:
            return 0.0
        k = (n - 1) * p
        f = int(k)
        c = min(f + 1, n - 1)
        return s[f] + (s[c] - s[f]) * (k - f)

    return {
        "mean": statistics.fmean(s) if s else 0.0,
        "p50": pct(0.50),
        "p95": pct(0.95),
        "min": s[0] if s else 0.0,
        "max": s[-1] if s else 0.0,
    }


def _repeat_cache_batch(cache, batch_size: int):
    """Expand a HF DynamicCache along the batch dimension without mutating input.

    Handles both the legacy layout (``cache.key_cache[i]`` is the storage) and
    the newer transformers layout (``cache.layers[i].keys`` is the storage and
    ``key_cache`` is a derived view).
    """
    if cache is None or batch_size == 1:
        return cache
    repeated = copy.deepcopy(cache)

    def _expand(t):
        return t.expand(batch_size, *t.shape[1:]).contiguous()

    if hasattr(repeated, "layers") and repeated.layers:
        for layer in repeated.layers:
            if hasattr(layer, "keys") and layer.keys is not None:
                layer.keys = _expand(layer.keys)
            if hasattr(layer, "values") and layer.values is not None:
                layer.values = _expand(layer.values)
        return repeated

    if hasattr(repeated, "key_cache") and hasattr(repeated, "value_cache"):
        for i in range(len(repeated.key_cache)):
            if repeated.key_cache[i] is not None:
                repeated.key_cache[i] = _expand(repeated.key_cache[i])
            if repeated.value_cache[i] is not None:
                repeated.value_cache[i] = _expand(repeated.value_cache[i])
    return repeated


@torch.no_grad()
def _prefill_cache(model, prefix_ids: torch.Tensor):
    out = model(prefix_ids, use_cache=True)
    return out.past_key_values


@torch.no_grad()
def _bench_one_cell(
    model,
    tokenizer,
    *,
    B: int,
    seq: int,
    prefix_len: int,
    iters: int,
    warmup: int,
) -> Cell:
    """Time `iters` runs of a (B, seq) forward over a populated KV cache of len prefix_len."""
    device = next(model.parameters()).device
    vocab_size = model.config.vocab_size

    prefix_ids = _make_input_ids(tokenizer, prefix_len, device=str(device))
    cache_b1 = _prefill_cache(model, prefix_ids)
    cache = _repeat_cache_batch(cache_b1, B) if B > 1 else cache_b1

    times_us: list[float] = []
    for i in range(iters + warmup):
        new_tokens = torch.randint(0, vocab_size, (B, seq), device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        # Pass a fresh deep-copy of the cache each iter so seq doesn't grow.
        iter_cache = copy.deepcopy(cache)
        model(new_tokens, past_key_values=iter_cache, use_cache=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        if i >= warmup:
            times_us.append((t1 - t0) / 1000.0)
        del iter_cache

    stats = _percentiles(times_us)
    return Cell(
        B=B,
        seq=seq,
        prefix_len=prefix_len,
        n_iters=len(times_us),
        warmup=warmup,
        mean_us=stats["mean"],
        p50_us=stats["p50"],
        p95_us=stats["p95"],
        min_us=stats["min"],
        max_us=stats["max"],
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--output", required=True, help="Path to write JSON results")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA not available — this microbench is meaningful only on a real "
            "target GPU (L40S/H100). Run on Modal."
        )

    print(f"Loading target={args.target} dtype={args.dtype} attn={args.attn_impl}", flush=True)
    tok, model = _load_model(args.target, dtype=args.dtype, attn_impl=args.attn_impl)

    cells: list[Cell] = []
    ratios: dict[int, dict[int, float]] = {}  # prefix_len -> {B: ratio_to_B1}
    error: str | None = None
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_partial() -> None:
        out_path.write_text(
            json.dumps(
                {
                    "schema": "asts-spec/pvp_microbench/v1",
                    "target": args.target,
                    "dtype": args.dtype,
                    "attn_impl": args.attn_impl,
                    "iters": args.iters,
                    "warmup": args.warmup,
                    "shapes": list(DEFAULT_SHAPES),
                    "prefix_lens": list(DEFAULT_PREFIX_LENS),
                    "cells": [asdict(c) for c in cells],
                    "ratios_vs_B1_by_prefix": ratios,
                    "kill_ratio_threshold": KILL_RATIO,
                    "any_kill": any(by_B.get(2, float("inf")) > KILL_RATIO for by_B in ratios.values()),
                    "error": error,
                },
                indent=2,
            )
        )

    try:
        for prefix_len in DEFAULT_PREFIX_LENS:
            print(f"\n=== prefix_len = {prefix_len} ===", flush=True)
            base_mean = None
            for B, seq in DEFAULT_SHAPES:
                print(f"  bench B={B} seq={seq} ...", flush=True)
                cell = _bench_one_cell(
                    model, tok,
                    B=B, seq=seq, prefix_len=prefix_len,
                    iters=args.iters, warmup=args.warmup,
                )
                cells.append(cell)
                if B == 1:
                    base_mean = cell.mean_us
                    ratio = 1.0
                else:
                    ratio = cell.mean_us / base_mean if base_mean else float("inf")
                ratios.setdefault(prefix_len, {})[B] = ratio
                print(
                    f"    mean={cell.mean_us:8.1f} us  p50={cell.p50_us:8.1f}  "
                    f"p95={cell.p95_us:8.1f}  ratio_vs_B1={ratio:6.3f}",
                    flush=True,
                )
                _write_partial()
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        print(f"\n  ✗ microbench crashed mid-run: {error}", flush=True)
        _write_partial()
        raise

    verdict_lines = []
    for prefix_len, by_B in ratios.items():
        r2 = by_B.get(2, float("nan"))
        flag = "KILL" if r2 > KILL_RATIO else "ok"
        verdict_lines.append(f"  prefix={prefix_len}: T(B=2)/T(B=1) = {r2:.3f} → {flag}")
    print("\n=== PVP Step-1 verdict ===")
    print("\n".join(verdict_lines))
    any_kill = any(by_B.get(2, float("nan")) > KILL_RATIO for by_B in ratios.values())
    if any_kill:
        print(f"\n  ✗ STOP — at least one prefix shows T(B=2)/T(B=1) > {KILL_RATIO}.")
        print("  PVP economics are dead. Do not write decoder code.")
    else:
        print(f"\n  ✓ PROCEED — all prefixes within T(B=2)/T(B=1) ≤ {KILL_RATIO}.")

    _write_partial()
    print(f"\nresults: {out_path}")
    sys.exit(2 if any_kill else 0)


if __name__ == "__main__":
    main()
