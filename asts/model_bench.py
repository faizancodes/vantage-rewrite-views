"""Forward-pass latency microbenchmark for the target and draft code LMs.

We measure two patterns that the decision gate needs:

  1. ar_step: single-token forward pass with a populated KV cache. This is
     the cost of one autoregressive token with the target model — the
     baseline that speculative decoding is trying to beat.

  2. verify_kstep: K-token forward pass with a populated KV cache. This is
     the cost of verifying a k-token speculative draft with one parallel
     forward pass through the target. For typical k=4..8 this should be
     close to ar_step cost (attention is memory-bound at long context),
     which is why spec decode wins.

We also measure the draft model's `ar_step` since spec decode pays
`k * draft_ar_step` per outer step.

All times are measured with `torch.cuda.synchronize()` around perf_counter,
warmed up before recording, and reported in microseconds.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_model(
    model_id: str,
    dtype: str = "bfloat16",
    attn_impl: str | None = None,
    trust_remote_code: bool = False,
):
    """Load a HuggingFace causal LM onto cuda. Returns (tok, model)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        dtype
    ]

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs: dict = {"torch_dtype": torch_dtype, "device_map": "cuda"}
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
    except (ValueError, ImportError, RuntimeError) as e:
        # Fall back to default attention if flash-attn isn't available.
        if attn_impl and "attn_implementation" in kwargs:
            print(f"  [warn] {attn_impl} unavailable ({e}); falling back to sdpa")
            kwargs.pop("attn_implementation")
            kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        else:
            raise

    model.eval()
    return tok, model


def _make_input_ids(tok, target_len: int, device: str = "cuda"):
    """Produce a [1, target_len] input_ids tensor by repeating realistic tokens.

    We use a chunk of code text (not arbitrary token ids) so the prefix is
    in-distribution for the model — gives realistic KV-cache memory pressure
    and attention patterns. Random ids would underestimate latency.
    """
    seed_text = (
        "def fibonacci(n: int) -> int:\n"
        "    if n < 2:\n"
        "        return n\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n - 1):\n"
        "        a, b = b, a + b\n"
        "    return b\n\n"
    )
    # Tokenize once, then tile to length.
    base = tok(seed_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if base.numel() == 0:
        raise RuntimeError("seed text tokenized to 0 tokens; check tokenizer")
    repeats = (target_len + base.numel() - 1) // base.numel()
    tiled = base.repeat(repeats)[:target_len]
    return tiled.unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Measurement primitives
# ---------------------------------------------------------------------------


@dataclass
class ModelMeasurement:
    model_id: str
    operation: str  # "ar_step" | "verify_kstep" | "prefill"
    prefix_tokens: int
    k: int  # 1 for ar_step, k for verify_kstep
    iterations: int
    stats_us: dict
    extra: dict


def _percentiles(samples: list[float]) -> dict:
    if not samples:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(samples)
    n = len(s)

    def pct(p: float) -> float:
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


def bench_ar_step(
    model_id: str,
    prefix_len: int,
    iterations: int = 50,
    warmup: int = 5,
    dtype: str = "bfloat16",
    attn_impl: str | None = None,
) -> tuple[ModelMeasurement, ModelMeasurement]:
    """Benchmark single-token AR forward pass with a populated KV cache.

    Returns (prefill_measurement, ar_step_measurement). Prefill is reported
    once (cost of building the KV cache to `prefix_len`); ar_step is the
    per-token decode cost we care about.
    """
    import torch

    tok, model = _load_model(model_id, dtype=dtype, attn_impl=attn_impl)

    input_ids = _make_input_ids(tok, prefix_len)
    vocab_size = model.config.vocab_size

    # ---- Prefill (one-time cost; we time it but it's separate from per-step) --
    prefill_times_us: list[float] = []
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    cache = out.past_key_values
    with torch.no_grad():
        for _ in range(3):  # 3 measurements is enough — prefill is one-shot per request
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            out = model(input_ids, use_cache=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter_ns()
            prefill_times_us.append((t1 - t0) / 1000.0)
            cache = out.past_key_values

    # ---- AR step (per-token decode) --
    # We also keep the first 20 raw per-iteration latencies (including warmup)
    # so the analysis can detect pathological warmup curves on H100/SDPA.
    ar_times_us: list[float] = []
    trace_first_20_us: list[float] = []
    next_token = torch.randint(0, vocab_size, (1, 1), device="cuda")
    with torch.no_grad():
        for i in range(iterations + warmup):
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            out = model(next_token, past_key_values=cache, use_cache=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter_ns()
            this_us = (t1 - t0) / 1000.0
            if len(trace_first_20_us) < 20:
                trace_first_20_us.append(this_us)
            if i >= warmup:
                ar_times_us.append(this_us)
            cache = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # Free GPU memory before next call
    del model, cache, out, input_ids, next_token, tok  # noqa: F821
    torch.cuda.empty_cache()

    prefill_meas = ModelMeasurement(
        model_id=model_id,
        operation="prefill",
        prefix_tokens=prefix_len,
        k=prefix_len,
        iterations=len(prefill_times_us),
        stats_us=_percentiles(prefill_times_us),
        extra={"dtype": dtype, "attn_impl": attn_impl or "default"},
    )
    ar_meas = ModelMeasurement(
        model_id=model_id,
        operation="ar_step",
        prefix_tokens=prefix_len,
        k=1,
        iterations=len(ar_times_us),
        stats_us=_percentiles(ar_times_us),
        extra={
            "dtype": dtype,
            "attn_impl": attn_impl or "default",
            "warmup": warmup,
            "trace_first_20_us": trace_first_20_us,
        },
    )
    return prefill_meas, ar_meas


def bench_verify_kstep(
    model_id: str,
    prefix_len: int,
    k_values: tuple[int, ...] = (4, 8, 16),
    iterations: int = 30,
    warmup: int = 3,
    dtype: str = "bfloat16",
    attn_impl: str | None = None,
) -> list[ModelMeasurement]:
    """Benchmark K-token verification forward pass with populated KV cache.

    Models the spec-decode "target verifies k draft tokens in one parallel
    forward" cost. Returns one Measurement per k.
    """
    import torch

    tok, model = _load_model(model_id, dtype=dtype, attn_impl=attn_impl)
    input_ids = _make_input_ids(tok, prefix_len)
    vocab_size = model.config.vocab_size

    # Build a fresh KV cache for each k (cheap relative to k-step time).
    measurements: list[ModelMeasurement] = []

    # Initialize once so the variables are bound even if k_values is empty.
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    cache = out.past_key_values

    for k in k_values:
        with torch.no_grad():
            out = model(input_ids, use_cache=True)
        cache = out.past_key_values

        times_us: list[float] = []
        with torch.no_grad():
            for i in range(iterations + warmup):
                k_tokens = torch.randint(0, vocab_size, (1, k), device="cuda")
                torch.cuda.synchronize()
                t0 = time.perf_counter_ns()
                model(k_tokens, past_key_values=cache, use_cache=True)
                torch.cuda.synchronize()
                t1 = time.perf_counter_ns()
                if i >= warmup:
                    times_us.append((t1 - t0) / 1000.0)
                # IMPORTANT: don't accumulate the cache across iterations,
                # otherwise effective context length grows. Re-prefill instead.
                if i + 1 < iterations + warmup:
                    out = model(input_ids, use_cache=True)
                    cache = out.past_key_values

        measurements.append(
            ModelMeasurement(
                model_id=model_id,
                operation="verify_kstep",
                prefix_tokens=prefix_len,
                k=k,
                iterations=len(times_us),
                stats_us=_percentiles(times_us),
                extra={"dtype": dtype, "attn_impl": attn_impl or "default"},
            )
        )

    del model, cache, out, input_ids, tok
    torch.cuda.empty_cache()
    return measurements


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_sweep(
    target_id: str = "Qwen/Qwen2.5-Coder-7B",
    draft_id: str = "Qwen/Qwen2.5-Coder-0.5B",
    prefix_lens: tuple[int, ...] = (512, 2048),
    k_values: tuple[int, ...] = (4, 8, 16),
    dtype: str = "bfloat16",
    attn_impl: str | None = "sdpa",
    ar_iters: int = 50,
    verify_iters: int = 30,
    ar_warmup: int = 10,
    verify_warmup: int = 5,
) -> dict:
    """Run the full model benchmark sweep.

    Two prefix lengths give us the slope of how attention scales; we'd
    extrapolate or add more if the trend is interesting.
    """
    results: list[dict] = []

    for model_id in (target_id, draft_id):
        for L in prefix_lens:
            print(
                f"  benchmarking {model_id} @ prefix={L} "
                f"(ar warmup={ar_warmup}, verify warmup={verify_warmup}) ...",
                flush=True,
            )
            prefill_m, ar_m = bench_ar_step(
                model_id,
                prefix_len=L,
                iterations=ar_iters,
                warmup=ar_warmup,
                dtype=dtype,
                attn_impl=attn_impl,
            )
            results.append(asdict(prefill_m))
            results.append(asdict(ar_m))

            kstep_ms = bench_verify_kstep(
                model_id,
                prefix_len=L,
                k_values=k_values,
                iterations=verify_iters,
                warmup=verify_warmup,
                dtype=dtype,
                attn_impl=attn_impl,
            )
            for m in kstep_ms:
                results.append(asdict(m))

    return {
        "schema": "asts-spec/model_bench/v1",
        "target_id": target_id,
        "draft_id": draft_id,
        "dtype": dtype,
        "attn_impl": attn_impl or "default",
        "ar_warmup": ar_warmup,
        "verify_warmup": verify_warmup,
        "measurements": results,
    }
