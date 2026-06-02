#!/usr/bin/env python3
"""Benchmark cached PLD verifier cost as a function of draft length.

The benchmark matches the BlazEdit PLD verifier shape: keep a KV cache for all
but the last prefix token, then verify ``prefix[-1] + drafts`` with
``use_cache=True``.  It reports both raw target forward time and full verifier
time including greedy rejection and cache crop.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from asts.decoder import crop_dynamic_cache
from asts.rejection import greedy_verify


def _dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }[name]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return float(values[idx])


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "p99": 0.0}
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "p10": _percentile(values, 0.10),
        "p90": _percentile(values, 0.90),
        "p99": _percentile(values, 0.99),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _elapsed_ms(device: torch.device, fn) -> float:
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        return float(start.elapsed_time(end))
    t0 = time.perf_counter_ns()
    fn()
    return (time.perf_counter_ns() - t0) / 1_000_000.0


def _fit_line(xs: list[int], ys: list[float]) -> dict[str, float]:
    if len(xs) < 2:
        return {"intercept_ms": 0.0, "slope_ms_per_token": 0.0, "r2": 0.0}
    n = float(len(xs))
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x <= 0:
        return {"intercept_ms": mean_y, "slope_ms_per_token": 0.0, "r2": 0.0}
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / var_x
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {
        "intercept_ms": float(intercept),
        "slope_ms_per_token": float(slope),
        "r2": float(r2),
    }


def _make_prompt(tokenizer, target_tokens: int) -> torch.Tensor:
    seed = (
        "def compute_total(items):\n"
        "    total = 0\n"
        "    for item in items:\n"
        "        if item.enabled:\n"
        "            total += item.value\n"
        "    return total\n\n"
        "class Example:\n"
        "    def update(self, account, payload):\n"
        "        account.name = payload.get('name', account.name)\n"
        "        account.updated = True\n"
        "        return account\n\n"
    )
    text = seed
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens:
        text += seed
    ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    if len(ids) < target_tokens:
        raise RuntimeError("failed to build prompt tokens")
    return torch.tensor([ids], dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--attn-impl", default="sdpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefix-len", type=int, default=1024)
    parser.add_argument("--lengths", default="0,1,2,4,8,16,32,64,128")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    lengths = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
    if min(lengths) < 0:
        raise SystemExit("--lengths must be non-negative")
    if args.prefix_len < 2:
        raise SystemExit("--prefix-len must be at least 2")

    tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.target,
        torch_dtype=dtype,
        attn_implementation=args.attn_impl,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()

    prompt = _make_prompt(tok, args.prefix_len + max(lengths) + 8).to(device)
    prefix = prompt[:, : args.prefix_len]
    cache_input = prefix[:, :-1]
    cache_len = int(cache_input.shape[1])
    with torch.inference_mode():
        out = model(cache_input, use_cache=True)
        cache = out.past_key_values
        _sync(device)

        rows: list[dict[str, Any]] = []
        for draft_len in lengths:
            verify_input = prompt[:, cache_len : cache_len + 1 + draft_len].contiguous()
            drafts = verify_input[0, 1:].detach().clone().tolist()
            n_pre = 1

            # Warm the exact shape.
            for _ in range(max(0, args.warmup)):
                crop_dynamic_cache(cache, cache_len)
                out = model(verify_input, past_key_values=cache, use_cache=True)
                _ = greedy_verify(drafts=drafts, target_logits=out.logits, n_pre=n_pre) if drafts else torch.argmax(out.logits[0, 0])
                crop_dynamic_cache(out.past_key_values, cache_len)
                cache = out.past_key_values
            _sync(device)

            forward_ms: list[float] = []
            full_ms: list[float] = []
            for _ in range(max(1, args.iters)):
                crop_dynamic_cache(cache, cache_len)

                def forward_only():
                    nonlocal cache
                    out = model(verify_input, past_key_values=cache, use_cache=True)
                    cache = out.past_key_values

                forward_ms.append(_elapsed_ms(device, forward_only))
                crop_dynamic_cache(cache, cache_len)

                def full_verify():
                    nonlocal cache
                    out = model(verify_input, past_key_values=cache, use_cache=True)
                    if drafts:
                        _ = greedy_verify(drafts=drafts, target_logits=out.logits, n_pre=n_pre)
                    else:
                        _ = int(torch.argmax(out.logits[0, 0]).item())
                    crop_dynamic_cache(out.past_key_values, cache_len)
                    cache = out.past_key_values

                full_ms.append(_elapsed_ms(device, full_verify))

            fwd = _summarize(forward_ms)
            full = _summarize(full_ms)
            rows.append(
                {
                    "draft_len": draft_len,
                    "input_tokens": 1 + draft_len,
                    "forward_ms_mean": fwd["mean"],
                    "forward_ms_median": fwd["median"],
                    "forward_ms_p90": fwd["p90"],
                    "full_verify_ms_mean": full["mean"],
                    "full_verify_ms_median": full["median"],
                    "full_verify_ms_p90": full["p90"],
                    "forward_tokens_per_sec": (1 + draft_len) / (fwd["mean"] / 1000.0) if fwd["mean"] > 0 else 0.0,
                    "full_verify_tokens_per_sec": (1 + draft_len) / (full["mean"] / 1000.0) if full["mean"] > 0 else 0.0,
                }
            )

    fit_forward = _fit_line([r["draft_len"] for r in rows], [r["forward_ms_mean"] for r in rows])
    fit_full = _fit_line([r["draft_len"] for r in rows], [r["full_verify_ms_mean"] for r in rows])
    report = {
        "target": args.target,
        "dtype": args.dtype,
        "attn_impl": args.attn_impl,
        "device": str(device),
        "prefix_len": args.prefix_len,
        "warmup": args.warmup,
        "iters": args.iters,
        "lengths": lengths,
        "forward_fit": fit_forward,
        "full_verify_fit": fit_full,
        "rows": rows,
    }

    md: list[str] = []
    md.append("# PLD Verifier Length Microbenchmark\n")
    md.append(
        f"target: `{args.target}`  dtype: `{args.dtype}`  attn: `{args.attn_impl}`  "
        f"prefix_len: `{args.prefix_len}`  iters: `{args.iters}`\n"
    )
    md.append("| draft len | input tokens | forward ms | full verify ms | forward tok/s | full tok/s |")
    md.append("|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(
            "| {draft_len} | {input_tokens} | {forward_ms_mean:.3f} | {full_verify_ms_mean:.3f} | "
            "{forward_tokens_per_sec:.1f} | {full_verify_tokens_per_sec:.1f} |".format(**r)
        )
    md.append("\n## Linear Fit\n")
    md.append(
        "- forward: fixed `{intercept_ms:.3f} ms`, incremental `{slope_ms_per_token:.4f} ms/token`, r2 `{r2:.3f}`".format(
            **fit_forward
        )
    )
    md.append(
        "- full verify: fixed `{intercept_ms:.3f} ms`, incremental `{slope_ms_per_token:.4f} ms/token`, r2 `{r2:.3f}`".format(
            **fit_full
        )
    )
    text = "\n".join(md) + "\n"
    print(text)
    print(json.dumps(report, indent=2, sort_keys=True))

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text(text)


if __name__ == "__main__":
    main()
