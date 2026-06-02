"""Full HumanEval evaluation of vanilla AR vs fixed-k spec vs ASTS-Spec.

Loads target + draft, iterates over HumanEval problems, runs each method on
each problem, logs per-step records to JSONL, and writes aggregate metrics
to JSON.

Usage:
    python scripts/run_prototype.py \
        --output-dir out/proto_eval \
        --n 164 \
        --max-new-tokens 256 \
        --k 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from asts.ast_policy import ASTPolicy
from asts.decoder import asts_spec_ar, fixed_spec_ar, vanilla_ar
from asts.humaneval import load_problems, truncate_at_stop
from asts.model_bench import _load_model


METHODS = ("vanilla", "fixed_k4", "fixed_k8", "asts_spec")


def _aggregate(per_step_records: list[dict], n_new_tokens_per_method: dict) -> dict:
    """Compute aggregate metrics from per-step records.

    Returns: {method: {tokens_per_sec, mean_accepted_drafts, n_steps, ...},
              "by_node_type": {node_type: {n, accept_rate, mean_k}}}
    """
    by_method: dict[str, dict] = {}
    by_node_type: dict[str, dict] = {}

    # Use the dynamic key set so any fixed-k values beyond the legacy
    # METHODS tuple are included in aggregate output.
    method_names = list(n_new_tokens_per_method.keys()) or list(METHODS)
    for method in method_names:
        method_steps = [r for r in per_step_records if r["method"] == method]
        if not method_steps:
            continue

        total_us = sum(r["wall_us"] for r in method_steps)
        total_emitted = sum(r["n_emitted"] for r in method_steps)
        total_accepted_drafts = sum(r["n_accepted_drafts"] for r in method_steps)
        total_k_requested = sum(r["k"] for r in method_steps)

        by_method[method] = {
            "n_steps": len(method_steps),
            "n_emitted_total": total_emitted,
            "wall_us_total": total_us,
            "tokens_per_sec": total_emitted / (total_us / 1e6) if total_us > 0 else 0,
            "us_per_token": total_us / total_emitted if total_emitted > 0 else 0,
            "mean_accepted_drafts_per_step": (
                total_accepted_drafts / len(method_steps) if method_steps else 0
            ),
            "mean_k_requested": (
                total_k_requested / len(method_steps) if method_steps else 0
            ),
            "n_new_tokens_total": n_new_tokens_per_method.get(method, 0),
        }

    # Per-node-type breakdown for ASTS-Spec only
    asts_steps = [r for r in per_step_records if r["method"] == "asts_spec"]
    for r in asts_steps:
        nt = r.get("node_type") or "default"
        d = by_node_type.setdefault(nt, {
            "n": 0,
            "sum_k": 0,
            "sum_accepted": 0,
            "sum_wall_us": 0,
        })
        d["n"] += 1
        d["sum_k"] += r["k"]
        d["sum_accepted"] += r["n_accepted_drafts"]
        d["sum_wall_us"] += r["wall_us"]

    for nt, d in by_node_type.items():
        n = d["n"]
        d["mean_k"] = d["sum_k"] / n if n else 0
        d["mean_accepted"] = d["sum_accepted"] / n if n else 0
        d["acceptance_rate"] = d["mean_accepted"] / d["mean_k"] if d["mean_k"] > 0 else 0
        d["mean_wall_us"] = d["sum_wall_us"] / n if n else 0

    return {"by_method": by_method, "by_node_type": by_node_type}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    p.add_argument("--draft", default="Qwen/Qwen2.5-Coder-0.5B")
    p.add_argument("--n", type=int, default=164, help="number of HumanEval problems (164 = full)")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--k-fixed", default="4,8", help="comma-sep fixed-k spec values")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--methods", default="vanilla,fixed,asts", help="comma-sep methods to run")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("run_prototype")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    steps_path = output_dir / "steps.jsonl"
    aggregate_path = output_dir / "aggregate.json"
    completions_path = output_dir / "completions.jsonl"

    methods_to_run = set(args.methods.split(","))
    fixed_ks = [int(x) for x in args.k_fixed.split(",")] if "fixed" in methods_to_run else []

    log.info("loading target=%s draft=%s dtype=%s", args.target, args.draft, args.dtype)
    target_tok, target = _load_model(args.target, dtype=args.dtype, attn_impl=args.attn_impl)
    draft_tok, draft = _load_model(args.draft, dtype=args.dtype, attn_impl=args.attn_impl)

    eos = [int(target_tok.eos_token_id)]
    log.info("eos token ids: %s", eos)

    problems = load_problems(n=args.n)
    log.info("loaded %d problems", len(problems))

    per_step_records: list[dict] = []
    completions: list[dict] = []
    # Build the method-key set dynamically from CLI args; the legacy METHODS
    # tuple only covered (vanilla, fixed_k4, fixed_k8, asts_spec) and breaks
    # for non-default --k-fixed values.
    active_methods: list[str] = []
    if "vanilla" in methods_to_run:
        active_methods.append("vanilla")
    for k in fixed_ks:
        active_methods.append(f"fixed_k{k}")
    if "asts" in methods_to_run:
        active_methods.append("asts_spec")
    n_new_tokens_per_method: dict = {m: 0 for m in active_methods}

    t_eval_start = time.perf_counter_ns()

    for idx, prob in enumerate(problems):
        log.info("[%d/%d] %s", idx + 1, len(problems), prob.task_id)
        prompt_ids = target_tok(
            prob.prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids[0]

        # Ensure GPU is idle before starting timing for this problem
        torch.cuda.synchronize()

        method_outputs: dict[str, dict] = {}

        if "vanilla" in methods_to_run:
            v_res = vanilla_ar(
                prompt_ids=prompt_ids,
                target=target,
                max_new_tokens=args.max_new_tokens,
                eos_token_ids=eos,
                method_name="vanilla",
            )
            for s in v_res.steps:
                rec = asdict(s)
                rec["task_id"] = prob.task_id
                per_step_records.append(rec)
            n_new_tokens_per_method["vanilla"] += v_res.n_new_tokens
            method_outputs["vanilla"] = {
                "tokens": v_res.output_token_ids[len(prompt_ids):],
                "wall_us": v_res.wall_us_total,
                "n_new_tokens": v_res.n_new_tokens,
            }

        for k in fixed_ks:
            method_name = f"fixed_k{k}"
            f_res = fixed_spec_ar(
                prompt_ids=prompt_ids,
                target=target,
                draft=draft,
                max_new_tokens=args.max_new_tokens,
                eos_token_ids=eos,
                k=k,
            )
            # Override method name on each step record
            for s in f_res.steps:
                rec = asdict(s)
                rec["method"] = method_name
                rec["task_id"] = prob.task_id
                per_step_records.append(rec)
            n_new_tokens_per_method[method_name] += f_res.n_new_tokens
            method_outputs[method_name] = {
                "tokens": f_res.output_token_ids[len(prompt_ids):],
                "wall_us": f_res.wall_us_total,
                "n_new_tokens": f_res.n_new_tokens,
            }

        if "asts" in methods_to_run:
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
            for s in a_res.steps:
                rec = asdict(s)
                rec["task_id"] = prob.task_id
                per_step_records.append(rec)
            n_new_tokens_per_method["asts_spec"] += a_res.n_new_tokens
            method_outputs["asts_spec"] = {
                "tokens": a_res.output_token_ids[len(prompt_ids):],
                "wall_us": a_res.wall_us_total,
                "n_new_tokens": a_res.n_new_tokens,
            }

        # Per-task summary log line
        speed_lines = []
        for m, o in method_outputs.items():
            speed = o["n_new_tokens"] / (o["wall_us"] / 1e6) if o["wall_us"] > 0 else 0
            speed_lines.append(f"{m}={speed:.1f}t/s")
        log.info("    %s", "  ".join(speed_lines))

        completions.append({
            "task_id": prob.task_id,
            "prompt": prob.prompt,
            "outputs": {
                m: {
                    "n_new_tokens": o["n_new_tokens"],
                    "wall_us": o["wall_us"],
                    "text": truncate_at_stop(
                        target_tok.decode(o["tokens"], skip_special_tokens=True)
                    ),
                }
                for m, o in method_outputs.items()
            },
        })

    t_eval_end = time.perf_counter_ns()

    # Write per-step JSONL
    with steps_path.open("w") as f:
        for r in per_step_records:
            f.write(json.dumps(r) + "\n")
    log.info("wrote %d step records → %s", len(per_step_records), steps_path)

    # Write completions JSONL
    with completions_path.open("w") as f:
        for c in completions:
            f.write(json.dumps(c) + "\n")
    log.info("wrote %d completions → %s", len(completions), completions_path)

    # Compute + write aggregate
    agg = _aggregate(per_step_records, n_new_tokens_per_method)
    agg["meta"] = {
        "schema": "asts-spec/proto_eval/v1",
        "target": args.target,
        "draft": args.draft,
        "dtype": args.dtype,
        "attn_impl": args.attn_impl,
        "n_problems": len(problems),
        "max_new_tokens": args.max_new_tokens,
        "fixed_ks": fixed_ks,
        "wall_us_total": (t_eval_end - t_eval_start) / 1000.0,
    }
    aggregate_path.write_text(json.dumps(agg, indent=2))
    log.info("wrote aggregate → %s", aggregate_path)

    # Print summary table
    print()
    print("=" * 78)
    print("ASTS-Spec Prototype: HumanEval Eval Summary")
    print("=" * 78)
    print(f"  target:    {args.target}")
    print(f"  draft:     {args.draft}")
    print(f"  problems:  {len(problems)}  ({args.max_new_tokens} max new tokens)")
    print(f"  dtype:     {args.dtype}  attn_impl: {args.attn_impl}")
    print()
    print(f"  {'method':<14} {'tokens/sec':>12} {'us/token':>10} {'mean_acc':>10} {'mean_k':>8} {'n_steps':>9}")
    print("  " + "-" * 70)
    vanilla_tps = agg["by_method"].get("vanilla", {}).get("tokens_per_sec", 0) or 1
    for method in METHODS:
        if method not in agg["by_method"]:
            continue
        m = agg["by_method"][method]
        speedup = m["tokens_per_sec"] / vanilla_tps
        marker = "  ✓" if method != "vanilla" and speedup >= 1.5 else ("  ~" if speedup >= 1.0 else "  ✗")
        print(
            f"  {method:<14} {m['tokens_per_sec']:>11.1f}  "
            f"{m['us_per_token']:>9.0f}  "
            f"{m['mean_accepted_drafts_per_step']:>9.2f}  "
            f"{m['mean_k_requested']:>7.2f}  "
            f"{m['n_steps']:>8}  ({speedup:.2f}x{marker if method != 'vanilla' else ''})"
        )
    print()
    print("  Per-AST-node-type acceptance (ASTS-Spec only, top 15 by n):")
    print(f"    {'node_type':<25} {'n':>5} {'mean_k':>7} {'mean_acc':>9} {'accept_rate':>11}")
    print("    " + "-" * 60)
    by_nt_sorted = sorted(agg["by_node_type"].items(), key=lambda x: -x[1]["n"])[:15]
    for nt, d in by_nt_sorted:
        print(
            f"    {nt:<25} {d['n']:>5} {d['mean_k']:>6.1f} "
            f"{d['mean_accepted']:>8.2f} {d['acceptance_rate']:>10.1%}"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()
