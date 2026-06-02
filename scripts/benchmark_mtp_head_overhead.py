#!/usr/bin/env python3
"""Microbenchmark runtime MTP head inference.

This isolates the head projection from target-model verification so queued MTP
runs can separate model signal from head-kernel overhead.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from asts.mtp_heads import MTPHeadConfig, PLDMTPHeads


def _load_heads(path: str, *, device: torch.device, dtype: torch.dtype, num_heads: int) -> PLDMTPHeads:
    ckpt = torch.load(path, map_location="cpu")
    cfg = MTPHeadConfig(**ckpt["config"])
    if int(cfg.num_heads) != int(num_heads):
        raise SystemExit(f"checkpoint has {cfg.num_heads} heads, expected {num_heads}")
    output_weight = ckpt.get("output_weight")
    if output_weight is not None:
        output_weight = output_weight.to(device=device, dtype=dtype)
    model = PLDMTPHeads(cfg, output_weight=output_weight)
    model.load_state_dict(ckpt["model_state"])
    model.to(device=device, dtype=dtype)
    model.eval()
    model.prepare_for_inference()
    return model


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return sorted(values)[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", required=True)
    parser.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    model = _load_heads(args.heads, device=device, dtype=dtype, num_heads=args.num_heads)
    if args.compile:
        model = torch.compile(model)  # type: ignore[assignment]

    hidden_size = int(model.config.hidden_size)
    hidden = torch.randn(1, hidden_size, device=device, dtype=dtype)

    with torch.inference_mode():
        for _ in range(max(0, args.warmup)):
            _ = model.predict_token_tensor(hidden)
        if device.type == "cuda":
            torch.cuda.synchronize()

        samples_ms: list[float] = []
        for _ in range(max(1, args.iters)):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model.predict_token_tensor(hidden)
                end.record()
                torch.cuda.synchronize()
                samples_ms.append(float(start.elapsed_time(end)))
            else:
                t0 = time.perf_counter_ns()
                _ = model.predict_token_tensor(hidden)
                samples_ms.append((time.perf_counter_ns() - t0) / 1_000_000.0)

    memory_mb = 0.0
    if device.type == "cuda":
        memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    report = {
        "heads": str(Path(args.heads)),
        "target": args.target,
        "device": str(device),
        "dtype": args.dtype,
        "num_heads": args.num_heads,
        "hidden_shape": [1, hidden_size],
        "iters": args.iters,
        "avg_head_compute_ms": sum(samples_ms) / len(samples_ms),
        "p50_head_compute_ms": _percentile(samples_ms, 0.50),
        "p90_head_compute_ms": _percentile(samples_ms, 0.90),
        "p99_head_compute_ms": _percentile(samples_ms, 0.99),
        "gpu_max_memory_mb": memory_mb,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
