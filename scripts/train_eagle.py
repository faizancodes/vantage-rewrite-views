"""CLI: train an EAGLE-1 head on Qwen-Coder-7B for ASTS-Spec speculation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asts.eagle_train import TrainConfig, train


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-samples", type=int, default=10_000)
    p.add_argument("--chunk-len", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--kl-weight", type=float, default=0.7)
    p.add_argument("--h-weight", type=float, default=0.3)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    cfg = TrainConfig(
        target_model_id=args.target,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        chunk_len=args.chunk_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        save_every=args.save_every,
        kl_weight=args.kl_weight,
        h_weight=args.h_weight,
        dtype=args.dtype,
        seed=args.seed,
    )

    result = train(cfg)
    print()
    print("=== EAGLE training complete ===")
    print(f"  final checkpoint: {result['final_path']}")
    print(f"  n_params:         {result['n_params']:,} ({result['n_params']/1e6:.2f}M)")
    print(f"  total_steps:      {result['total_steps']}")
    print(f"  wall_seconds:     {result['wall_seconds']:.1f}")
    if result["final_loss"]:
        fl = result["final_loss"]
        print(f"  final loss:       total={fl['loss']:.4f}  kl={fl['kl']:.4f}  h={fl['h_l1']:.4f}")


if __name__ == "__main__":
    main()
