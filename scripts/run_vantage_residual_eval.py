#!/usr/bin/env python3
"""Smoke/evaluation entrypoint for VANTAGE residual decoding.

This is a lightweight runtime scaffold.  The default ``--toy`` path exercises
the decoder with small deterministic fake models so the integration can be
validated without downloading a checkpoint.  Full model/manifest evaluation can
be layered on this CLI once the residual draft model and router are selected.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asts.vantage_residual_decoder import (  # noqa: E402
    vantage_residual_decode,
    parse_vantage_residual_method,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        default="vantage_residual_k4",
        help=(
            "Residual method name: vantage_residual_k1/k2/k4, router_k4, "
            "vantage_residual_preverify_replace_k1/k2/k4, or "
            "vantage_residual_queued_k1/k2/k4."
        ),
    )
    parser.add_argument("--prompt-ids", default="1", help="Comma-separated toy prompt token IDs.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--eos-token-id", action="append", type=int, default=[])
    parser.add_argument("--toy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-residual", action="store_true")
    parser.add_argument("--trigger-false", action="store_true")
    parser.add_argument("--output-json", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.toy:
        raise SystemExit(
            "Only --toy runtime is implemented in this scaffold. "
            "Use the decoder module directly for custom model experiments."
        )

    prompt = torch.tensor(_parse_prompt_ids(args.prompt_ids), dtype=torch.long)
    config = parse_vantage_residual_method(args.method)
    if args.disable_residual:
        config = _replace_config(config, residual_enabled=False)

    target = _IncrementModel()
    residual = _IncrementModel()
    trigger = (lambda _prefix, _pld, _cfg: False) if args.trigger_false else None

    t0 = time.perf_counter_ns()
    result = vantage_residual_decode(
        prompt,
        target,
        max_new_tokens=args.max_new_tokens,
        eos_token_ids=list(args.eos_token_id),
        config=config,
        residual_model=residual,
        residual_trigger=trigger,
    )
    wall_us = (time.perf_counter_ns() - t0) / 1000.0
    payload: dict[str, Any] = {
        "method": args.method,
        "toy": True,
        "prompt_ids": prompt.tolist(),
        "output_token_ids": result.output_token_ids,
        "n_new_tokens": result.n_new_tokens,
        "wall_us": wall_us,
        "decoder_wall_us_total": result.wall_us_total,
        "steps": [
            {
                "step": step.step,
                "k": step.k,
                "n_accepted_drafts": step.n_accepted_drafts,
                "n_emitted": step.n_emitted,
                "rejected": step.rejected,
                "proposal_kind": step.proposal_kind,
                "pld_variant_triggered": step.pld_variant_triggered,
                "queued_available": step.queued_available,
                "queued_used": step.queued_used,
                "queued_invalid_reason": step.queued_invalid_reason,
                "verifier_calls_this_step": step.verifier_calls_this_step,
                "mtp_extra_verify_calls": step.mtp_extra_verify_calls,
            }
            for step in result.steps
        ],
        "limitations": [
            "toy scaffold only",
            "no checkpoint loading or manifest iteration in this entrypoint yet",
            "residual drafts are verified with target logits before emission",
        ],
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


def _parse_prompt_ids(value: str) -> list[int]:
    ids = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not ids:
        raise SystemExit("--prompt-ids must contain at least one token")
    return ids


def _replace_config(config, **updates):
    values = {
        "method_name": config.method_name,
        "residual_k": config.residual_k,
        "residual_enabled": config.residual_enabled,
        "residual_trigger": config.residual_trigger,
        "preverify_replace": config.preverify_replace,
        "queued_residual": config.queued_residual,
        "router_min_pld_draft_len": config.router_min_pld_draft_len,
        "pld_max_draft_tokens": config.pld_max_draft_tokens,
        "pld_max_matching_ngram_size": config.pld_max_matching_ngram_size,
        "pld_min_matching_ngram_size": config.pld_min_matching_ngram_size,
    }
    values.update(updates)
    return type(config)(**values)


class _ToyOutput:
    def __init__(self, logits):
        self.logits = logits
        self.past_key_values = None


class _IncrementModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 256) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(()))
        self.config = type("Config", (), {"pad_token_id": 0})()
        self.vocab_size = int(vocab_size)

    def forward(self, input_ids, past_key_values=None, use_cache=True, **kwargs):
        del past_key_values, use_cache, kwargs
        batch, seq = input_ids.shape
        logits = torch.full((batch, seq, self.vocab_size), -1000.0)
        for batch_i in range(batch):
            for pos_i in range(seq):
                token = int(input_ids[batch_i, pos_i].item())
                logits[batch_i, pos_i, (token + 1) % self.vocab_size] = 1000.0
        return _ToyOutput(logits)


if __name__ == "__main__":
    raise SystemExit(main())
