"""PVP Step 5 — byte-identical check vs plain PLD on 20 held-out commits.

This file is a *script*, not a pytest unit test, because it loads
Qwen2.5-Coder-7B and is meaningful only on a real target GPU. Invoke via
``scripts/run_pvp_lossless_modal.py`` (Modal wrapper), or directly:

    python tests/test_pvp_lossless.py \\
        --n 20 --output analysis/pvp/runs/pvp_lossless_v1.json

Exits 0 if PVP K=2 == PLD byte-for-byte on every commit, else 1 with a
per-task divergence report.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from asts.blazedit_decoder import blazedit_speculative_ar, parse_blazedit_method
from asts.humaneval import load_problems_from_jsonl
from asts.model_bench import _load_model
from asts.pvp_decoder import parse_pvp_method, pvp_speculative_ar


@dataclass
class TaskResult:
    task_id: str
    pld_n_new: int
    pvp_n_new: int
    match: bool
    first_diff_index: int | None
    pld_tokens_head: list[int]
    pvp_tokens_head: list[int]


def _eos_ids(tokenizer, target) -> list[int]:
    eos: list[int] = []
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos.append(int(tokenizer.eos_token_id))
    raw = getattr(getattr(target, "config", None), "eos_token_id", None)
    if raw is not None:
        if isinstance(raw, list):
            eos.extend(int(x) for x in raw)
        else:
            eos.append(int(raw))
    return sorted(set(eos))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument(
        "--problems",
        default="data/real_commits/path_a_test500_v1.jsonl",
        help="JSONL manifest of held-out problems.",
    )
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--pld-method", default="blazedit_pld_w128_n10")
    p.add_argument("--pvp-method", default="vantage_pvp_k2_w128_n10")
    p.add_argument("--output", required=True)
    p.add_argument(
        "--strict-determinism",
        action="store_true",
        help="Set torch.use_deterministic_algorithms + matmul TF32 off.",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for lossless test (loads Qwen2.5-Coder-7B).")
    if args.strict_determinism:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    problems_path = ROOT / args.problems
    if not problems_path.exists():
        raise SystemExit(f"problems file not found: {problems_path}")

    print(f"loading target={args.target} dtype={args.dtype} attn={args.attn_impl}", flush=True)
    target_tok, target = _load_model(args.target, dtype=args.dtype, attn_impl=args.attn_impl)
    eos = _eos_ids(target_tok, target)
    print(f"eos_ids={eos}", flush=True)

    problems = load_problems_from_jsonl(str(problems_path), n=args.n)
    print(f"loaded {len(problems)} problems from {problems_path.name}", flush=True)

    pld_cfg = parse_blazedit_method(args.pld_method)
    pvp_cfg, K = parse_pvp_method(args.pvp_method)
    print(f"pld_cfg.mode={pld_cfg.mode}  pvp K={K}", flush=True)

    results: list[TaskResult] = []
    n_match = 0
    n_total = 0
    for i, prob in enumerate(problems):
        n_total += 1
        prompt_ids = target_tok(
            prob.prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids[0]

        print(f"[{i+1}/{len(problems)}] {prob.task_id} (prompt_len={len(prompt_ids)})", flush=True)

        pld_res = blazedit_speculative_ar(
            prompt_ids=prompt_ids,
            target=target,
            assistant=None,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos,
            config=pld_cfg,
            method_name=args.pld_method,
        )
        pld_new = pld_res.output_token_ids[len(prompt_ids):]

        pvp_res = pvp_speculative_ar(
            prompt_ids=prompt_ids,
            target=target,
            assistant=None,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos,
            config=pvp_cfg,
            method_name=args.pvp_method,
            K=K,
        )
        pvp_new = pvp_res.output_token_ids[len(prompt_ids):]

        match = pld_new == pvp_new
        first_diff = None
        if not match:
            for j in range(min(len(pld_new), len(pvp_new))):
                if pld_new[j] != pvp_new[j]:
                    first_diff = j
                    break
            if first_diff is None:
                first_diff = min(len(pld_new), len(pvp_new))

        result = TaskResult(
            task_id=prob.task_id,
            pld_n_new=len(pld_new),
            pvp_n_new=len(pvp_new),
            match=match,
            first_diff_index=first_diff,
            pld_tokens_head=pld_new[: (first_diff + 5) if first_diff is not None else 10],
            pvp_tokens_head=pvp_new[: (first_diff + 5) if first_diff is not None else 10],
        )
        results.append(result)
        if match:
            n_match += 1
            print(f"  ✓ match  pld_n_new={len(pld_new)}  pvp_n_new={len(pvp_new)}", flush=True)
        else:
            print(
                f"  ✗ DIVERGE  first_diff={first_diff}  "
                f"pld_n_new={len(pld_new)}  pvp_n_new={len(pvp_new)}",
                flush=True,
            )

    report: dict[str, Any] = {
        "schema": "asts-spec/pvp_lossless/v1",
        "config": vars(args),
        "n_total": n_total,
        "n_match": n_match,
        "all_match": n_match == n_total,
        "results": [asdict(r) for r in results],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print()
    print("=" * 60)
    print(f"PVP LOSSLESS VERIFICATION  ({args.pld_method} vs {args.pvp_method})")
    print("=" * 60)
    print(f"  match: {n_match}/{n_total}")
    if report["all_match"]:
        print("  ✓ ALL BYTE-IDENTICAL — proceeding to Step 6 is justified")
        print(f"  results: {out_path}")
        return 0
    print("  ✗ DIVERGENCE — see per-task results for first_diff_index")
    print(f"  results: {out_path}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
