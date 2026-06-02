#!/usr/bin/env python3
"""Build conservative LM-head row clusters for selective verification."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from asts.selective_lm_head import (
    build_clusters_from_assignment,
    contiguous_lm_head_clusters,
    save_lm_head_clusters,
    sorted_projection_lm_head_clusters,
)
from scripts.benchmark_real_shape_forward import model_dtype_arg
from scripts.run_eagle_eval import _load_model


def _kmeans_assignments(weight: torch.Tensor, num_clusters: int, seed: int) -> torch.Tensor:
    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception as exc:  # pragma: no cover - depends on environment.
        raise SystemExit(f"sklearn is required for --method kmeans: {exc}") from exc
    x = weight.detach().float().cpu().numpy()
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        random_state=seed,
        batch_size=4096,
        n_init="auto",
        max_iter=100,
        verbose=0,
    )
    labels = kmeans.fit_predict(x)
    return torch.tensor(labels, dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--num-clusters", type=int, default=256)
    parser.add_argument(
        "--method",
        choices=["contiguous", "random_projection", "kmeans"],
        default="random_projection",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    tokenizer, model = _load_model(
        args.target,
        dtype=model_dtype_arg(args.dtype),
        attn_impl=args.attn,
    )
    del tokenizer
    weight = model.lm_head.weight.detach().float().cpu()
    bias = getattr(model.lm_head, "bias", None)
    bias_tensor = bias.detach().float().cpu() if bias is not None else None
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    if args.method == "contiguous":
        clusters = contiguous_lm_head_clusters(weight, args.num_clusters, bias=bias_tensor)
    elif args.method == "random_projection":
        clusters = sorted_projection_lm_head_clusters(
            weight,
            args.num_clusters,
            seed=args.seed,
            bias=bias_tensor,
        )
    else:
        assignments = _kmeans_assignments(weight, args.num_clusters, args.seed)
        clusters = build_clusters_from_assignment(weight, assignments, bias=bias_tensor)
    build_s = time.perf_counter() - t1

    metadata = {
        "target": args.target,
        "dtype": args.dtype,
        "attn": args.attn,
        "num_clusters": args.num_clusters,
        "method": args.method,
        "seed": args.seed,
        "vocab_size": int(weight.shape[0]),
        "hidden_size": int(weight.shape[1]),
        "load_seconds": load_s,
        "build_seconds": build_s,
        "radius_mean": float(clusters.radii.float().mean().item()),
        "radius_p90": float(torch.quantile(clusters.radii.float(), 0.9).item()),
        "max_cluster_size": int(max(ids.numel() for ids in clusters.token_ids)),
        "min_cluster_size": int(min(ids.numel() for ids in clusters.token_ids)),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_lm_head_clusters(args.output, clusters, metadata)
    report_path = Path(args.report) if args.report else Path(args.output).with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
