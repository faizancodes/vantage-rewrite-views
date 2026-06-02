"""Lossless smoke test: vanilla AR == fixed-k spec == ASTS-Spec, byte-for-byte.

Loads target + draft, runs each mode on N HumanEval prompts in greedy mode,
asserts output token IDs are identical across all three.

Run with --strict-determinism to enforce CUDA deterministic algorithms (slower
but eliminates floating-point near-tie flakiness).

Usage:
    python scripts/verify_lossless.py --n 3 --max-new-tokens 32
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from asts.ast_policy import ASTPolicy
from asts.decoder import asts_spec_ar, fixed_spec_ar, vanilla_ar
from asts.humaneval import load_problems
from asts.model_bench import _load_model


def _set_deterministic():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    p.add_argument("--draft", default="Qwen/Qwen2.5-Coder-0.5B")
    p.add_argument("--n", type=int, default=3, help="number of HumanEval prompts to test")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--k", type=int, default=4, help="draft length for fixed_spec")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--strict-determinism", action="store_true")
    p.add_argument("--output", default=None)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("verify_lossless")

    if args.strict_determinism:
        log.info("enabling deterministic CUDA algorithms")
        _set_deterministic()

    log.info("loading target=%s draft=%s", args.target, args.draft)
    target_tok, target = _load_model(args.target, dtype=args.dtype, attn_impl=args.attn_impl)
    draft_tok, draft = _load_model(args.draft, dtype=args.dtype, attn_impl=args.attn_impl)

    if target_tok.get_vocab() != draft_tok.get_vocab():
        log.warning(
            "tokenizer vocabs differ between target and draft; "
            "lossless rejection sampling requires shared vocab — results may diverge"
        )

    eos = [int(target_tok.eos_token_id)]
    if hasattr(target_tok, "additional_special_tokens_ids") and target_tok.additional_special_tokens_ids:
        # Qwen often has multiple special tokens that act as stops (e.g., <|im_end|>)
        for sid in target_tok.additional_special_tokens_ids:
            if sid is not None and sid not in eos:
                eos.append(int(sid))
    log.info("eos token ids: %s", eos)

    problems = load_problems(n=args.n)
    log.info("loaded %d problems", len(problems))

    results = []
    n_match_vf = 0  # vanilla == fixed
    n_match_va = 0  # vanilla == asts

    for i, prob in enumerate(problems):
        log.info("[%d/%d] task=%s", i + 1, len(problems), prob.task_id)
        prompt_ids = target_tok(
            prob.prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids[0]

        # Vanilla AR
        v_res = vanilla_ar(
            prompt_ids=prompt_ids,
            target=target,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos,
        )
        v_new = v_res.output_token_ids[len(prompt_ids):]

        # Fixed-k speculative
        f_res = fixed_spec_ar(
            prompt_ids=prompt_ids,
            target=target,
            draft=draft,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos,
            k=args.k,
        )
        f_new = f_res.output_token_ids[len(prompt_ids):]

        # ASTS-Spec
        ast_policy = ASTPolicy(language="python")
        a_res = asts_spec_ar(
            prompt_ids=prompt_ids,
            target=target,
            draft=draft,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos,
            tokenizer=target_tok,
            ast_policy=ast_policy,
        )
        a_new = a_res.output_token_ids[len(prompt_ids):]

        match_vf = v_new == f_new
        match_va = v_new == a_new
        if match_vf:
            n_match_vf += 1
        if match_va:
            n_match_va += 1

        # Diagnostic: where do they diverge?
        first_diff_vf = next(
            (j for j in range(min(len(v_new), len(f_new))) if v_new[j] != f_new[j]),
            min(len(v_new), len(f_new)) if len(v_new) != len(f_new) else None,
        )
        first_diff_va = next(
            (j for j in range(min(len(v_new), len(a_new))) if v_new[j] != a_new[j]),
            min(len(v_new), len(a_new)) if len(v_new) != len(a_new) else None,
        )

        log.info(
            "  vanilla=%d toks  fixed=%d toks  asts=%d toks  match_vf=%s (first_diff=%s)  match_va=%s (first_diff=%s)",
            len(v_new), len(f_new), len(a_new),
            match_vf, first_diff_vf, match_va, first_diff_va,
        )

        results.append({
            "task_id": prob.task_id,
            "vanilla_n_new": len(v_new),
            "fixed_n_new": len(f_new),
            "asts_n_new": len(a_new),
            "vanilla_wall_us": v_res.wall_us_total,
            "fixed_wall_us": f_res.wall_us_total,
            "asts_wall_us": a_res.wall_us_total,
            "match_vf": match_vf,
            "match_va": match_va,
            "first_diff_vf": first_diff_vf,
            "first_diff_va": first_diff_va,
            "asts_n_steps": len(a_res.steps),
            "asts_mean_accepted": (
                sum(s.n_accepted_drafts for s in a_res.steps) / max(1, len(a_res.steps))
            ),
        })

    # Verdict
    print()
    print("=" * 60)
    print("LOSSLESS VERIFICATION")
    print("=" * 60)
    print(f"  vanilla == fixed_k{args.k}:  {n_match_vf}/{len(problems)}")
    print(f"  vanilla == asts_spec:    {n_match_va}/{len(problems)}")
    if n_match_vf == len(problems) and n_match_va == len(problems):
        print()
        print("  ✓ ALL OUTPUTS BYTE-IDENTICAL")
        print("  Lossless invariant holds.")
    else:
        print()
        print("  ✗ DIVERGENCE DETECTED")
        print("  See per-task results for first-diff positions.")
    print("=" * 60)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "schema": "asts-spec/lossless_verify/v1",
            "config": vars(args),
            "n_match_vf": n_match_vf,
            "n_match_va": n_match_va,
            "n_total": len(problems),
            "results": results,
        }, indent=2))
        log.info("wrote %s", out_path)

    # Exit code: 0 if all match, 1 otherwise
    sys.exit(0 if (n_match_vf == len(problems) and n_match_va == len(problems)) else 1)


if __name__ == "__main__":
    main()
