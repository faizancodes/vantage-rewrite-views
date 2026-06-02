"""PVP Step-7 evaluator: run blazedit_pld_w128_n10 and vantage_pvp_k2_w128_n10
side-by-side on the held-out 500, log per-task wall time + per-step records,
and emit a JSON artifact that ``analyze_pvp_run.py`` (Step 8) reads.

Standalone — does not modify scripts/run_eagle_eval.py. Same model load
(``asts.model_bench._load_model``) and same problem loader
(``load_problems_from_jsonl``) so the numbers are directly comparable to
existing PLD baselines.

For every task we *also* re-run the byte-identical lossless check: PLD and
PVP outputs must match. Any divergence aborts the run loudly (and is
recorded in the artifact for forensics).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from asts.blazedit_decoder import blazedit_speculative_ar, parse_blazedit_method
from asts.humaneval import load_problems_from_jsonl
from asts.model_bench import _load_model
from asts.pvp_decoder import parse_pvp_method, pvp_speculative_ar


@dataclass
class TaskRecord:
    task_id: str
    prompt_len: int
    pld_wall_us: float
    pld_n_new: int
    pvp_wall_us: float
    pvp_n_new: int
    match: bool
    first_diff_index: int | None
    pld_steps: int
    pvp_steps: int
    # Per-step proposal_kind histogram for PVP (string -> count).
    pvp_proposal_kinds: dict[str, int] = field(default_factory=dict)
    # Aggregated PVP-step counters.
    pvp_row1_attempted: int = 0
    pvp_row1_certified: int = 0
    pvp_row1_commits: int = 0          # total tokens committed by row 1
    pvp_full_accept_steps: int = 0     # steps where row 0 fully accepted draft1
    pvp_b2_steps: int = 0              # steps where the B=2 forward fired
    # Speedup vs PLD.
    speedup_vs_pld: float = 1.0


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


def _summarize_pvp_steps(steps) -> dict:
    """Aggregate PVP step records into per-task counters."""
    kinds: dict[str, int] = {}
    row1_attempted = 0
    row1_certified = 0
    row1_commits = 0
    full_accept = 0
    b2_steps = 0
    for s in steps:
        kind = s.proposal_kind or "unknown"
        kinds[kind] = kinds.get(kind, 0) + 1
        if kind == "vantage_pvp_k2":
            b2_steps += 1
            # row 0 fully accepted iff n_accepted_drafts == proposal_tokens (n_1).
            if s.proposal_tokens is not None and s.n_accepted_drafts == s.proposal_tokens:
                full_accept += 1
                row1_attempted += 1
            if s.mtp_extra_accepted_drafts == 1:
                row1_certified += 1
            if s.proposal_neural_draft_tokens:
                row1_commits += int(s.proposal_neural_draft_tokens)
    return {
        "kinds": kinds,
        "row1_attempted": row1_attempted,
        "row1_certified": row1_certified,
        "row1_commits": row1_commits,
        "full_accept": full_accept,
        "b2_steps": b2_steps,
    }


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
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--pld-method", default="blazedit_pld_w128_n10")
    p.add_argument("--pvp-method", default="vantage_pvp_k2_w128_n10")
    p.add_argument(
        "--abort-on-divergence",
        action="store_true",
        help="If set, exit non-zero on the first PLD/PVP token mismatch.",
    )
    p.add_argument("--output", required=True)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (loads Qwen2.5-Coder-7B).")

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

    records: list[TaskRecord] = []
    n_total = 0
    n_match = 0
    pld_total_us = 0.0
    pvp_total_us = 0.0

    t_start = time.perf_counter_ns()
    for i, prob in enumerate(problems):
        n_total += 1
        prompt_ids = target_tok(
            prob.prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids[0]

        # Plain PLD baseline.
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

        # PVP.
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

        agg = _summarize_pvp_steps(pvp_res.steps)
        speedup = pld_res.wall_us_total / pvp_res.wall_us_total if pvp_res.wall_us_total else 0.0

        records.append(
            TaskRecord(
                task_id=prob.task_id,
                prompt_len=int(len(prompt_ids)),
                pld_wall_us=float(pld_res.wall_us_total),
                pld_n_new=len(pld_new),
                pvp_wall_us=float(pvp_res.wall_us_total),
                pvp_n_new=len(pvp_new),
                match=match,
                first_diff_index=first_diff,
                pld_steps=len(pld_res.steps),
                pvp_steps=len(pvp_res.steps),
                pvp_proposal_kinds=agg["kinds"],
                pvp_row1_attempted=agg["row1_attempted"],
                pvp_row1_certified=agg["row1_certified"],
                pvp_row1_commits=agg["row1_commits"],
                pvp_full_accept_steps=agg["full_accept"],
                pvp_b2_steps=agg["b2_steps"],
                speedup_vs_pld=float(speedup),
            )
        )
        if match:
            n_match += 1
        pld_total_us += pld_res.wall_us_total
        pvp_total_us += pvp_res.wall_us_total

        if i % 25 == 0 or i + 1 == len(problems):
            print(
                f"[{i+1}/{len(problems)}] {prob.task_id} prompt_len={len(prompt_ids)} "
                f"match={match} pld={pld_res.wall_us_total/1e6:.2f}s "
                f"pvp={pvp_res.wall_us_total/1e6:.2f}s "
                f"speedup={speedup:.3f}x "
                f"row1_cert/attempt={agg['row1_certified']}/{agg['row1_attempted']}",
                flush=True,
            )
        if not match and args.abort_on_divergence:
            print(
                f"\n  ✗ DIVERGENCE on {prob.task_id} at index {first_diff}; aborting.",
                flush=True,
            )
            break

    elapsed_total = (time.perf_counter_ns() - t_start) / 1e9

    speedups = [r.speedup_vs_pld for r in records if r.speedup_vs_pld > 0]
    summary = {
        "schema": "asts-spec/pvp_eval/v1",
        "n_total": n_total,
        "n_match": n_match,
        "all_match": n_match == n_total,
        "pld_total_us": pld_total_us,
        "pvp_total_us": pvp_total_us,
        "overall_speedup": (pld_total_us / pvp_total_us) if pvp_total_us else 0.0,
        "speedup_median": statistics.median(speedups) if speedups else 0.0,
        "speedup_mean": statistics.fmean(speedups) if speedups else 0.0,
        "elapsed_wall_s": elapsed_total,
    }
    print("\n" + "=" * 60)
    print(f"PVP eval — {args.pld_method} vs {args.pvp_method}")
    print("=" * 60)
    print(f"  n_match           : {summary['n_match']} / {summary['n_total']}")
    print(f"  pld total wall    : {pld_total_us/1e6:.1f}s")
    print(f"  pvp total wall    : {pvp_total_us/1e6:.1f}s")
    print(f"  overall speedup   : {summary['overall_speedup']:.4f}x")
    print(f"  speedup median    : {summary['speedup_median']:.4f}x")
    print(f"  speedup mean      : {summary['speedup_mean']:.4f}x")
    print(f"  elapsed (wall)    : {elapsed_total:.0f}s")

    report = {
        **summary,
        "config": vars(args),
        "records": [asdict(r) for r in records],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"  artifact          : {out_path}")

    if not summary["all_match"]:
        return 2  # lossless gate broken
    return 0


if __name__ == "__main__":
    sys.exit(main())
