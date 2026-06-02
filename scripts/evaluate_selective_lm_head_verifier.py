#!/usr/bin/env python3
"""Offline certification diagnostic for selective LM-head PLD verification."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from asts.decoder import crop_dynamic_cache
from asts.selective_lm_head import certify_or_fallback_argmax, load_lm_head_clusters
from scripts.benchmark_real_shape_forward import (
    bucket_leq,
    load_trace,
    make_drafts,
    model_dtype_arg,
    summarize,
)
from scripts.profile_lm_head_cost import encode_completion_tokens
from scripts.run_eagle_eval import _encode_prompt_ids, _load_model


@dataclass
class SelectiveEvalRecord:
    task_id: str
    step_id: int
    draft_len: int
    token_index: int
    draft_token: int
    baseline_argmax: int
    selected_token: int
    certified: bool
    full_fallback: bool
    risky_cluster_count: int
    risky_token_count: int
    mismatch: bool


def _last_hidden(backbone_out) -> torch.Tensor:
    if hasattr(backbone_out, "last_hidden_state"):
        return backbone_out.last_hidden_state
    return backbone_out[0]


def _past(backbone_out):
    if hasattr(backbone_out, "past_key_values"):
        return backbone_out.past_key_values
    return backbone_out[1] if len(backbone_out) > 1 else None


def _profile_share(profile_json: str | None, fallback: float) -> float:
    if not profile_json:
        return fallback
    try:
        data = json.loads(Path(profile_json).read_text())
        return float(data["aggregate"]["lm_head_share_of_full_forward"])
    except Exception:
        return fallback


def run_eval(args: argparse.Namespace) -> tuple[list[SelectiveEvalRecord], dict[str, Any]]:
    device = torch.device(args.device)
    tokenizer, target = _load_model(
        args.target,
        dtype=model_dtype_arg(args.dtype),
        attn_impl=args.attn,
    )
    target.eval()
    clusters = load_lm_head_clusters(args.clusters, device=device)
    weight = target.lm_head.weight.detach()
    args.vocab_size = int(weight.shape[0])
    bias = getattr(target.lm_head, "bias", None)
    if bias is not None:
        bias = bias.detach()
    steps_by_task, completions = load_trace(
        steps_path=args.steps,
        completions_path=args.completions,
        method=args.method,
    )
    records: list[SelectiveEvalRecord] = []
    fallback_tokens = tokenizer(
        "def _fallback_value():\n    return None\n",
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0].tolist()
    total_steps = 0
    with torch.inference_mode():
        for task_id, task_steps in steps_by_task.items():
            completion = completions.get(task_id)
            if not completion:
                continue
            prompt_ids = _encode_prompt_ids(tokenizer, completion["prompt"], args.chat_template)
            prefix = [int(x) for x in prompt_ids.tolist()]
            generated_ids = encode_completion_tokens(tokenizer, completion, args.method)
            generated_pos = 0
            cache = None
            cache_len = 0
            for row in task_steps:
                if args.max_steps and total_steps >= args.max_steps:
                    return records, summarize_eval(records, clusters.num_clusters, args)
                total_steps += 1
                old_prefix_len = len(prefix)
                if cache_len >= old_prefix_len:
                    cache_len = max(0, old_prefix_len - 1)
                    crop_dynamic_cache(cache, cache_len)
                n_pre = old_prefix_len - cache_len
                draft_len = int(row.get("target_draft_tokens") or row.get("k") or 0)
                accepted_len = int(row.get("target_accepted_nonroot") or row.get("n_accepted_drafts") or 0)
                emitted = int(row.get("n_emitted") or (accepted_len + 1))
                drafts = make_drafts(
                    generated_ids=generated_ids,
                    generated_pos=generated_pos,
                    accepted_len=accepted_len,
                    draft_len=draft_len,
                    fallback_tokens=fallback_tokens,
                )
                target_input = torch.tensor([prefix[cache_len:] + drafts], device=device, dtype=torch.long)
                out = target.model(
                    input_ids=target_input,
                    past_key_values=cache,
                    use_cache=True,
                )
                hidden = _last_hidden(out)
                logits = target.lm_head(hidden)
                cache = _past(out)
                cache_len = cache_len + int(target_input.shape[1])
                if drafts:
                    pred_positions = range(n_pre - 1, n_pre - 1 + len(drafts))
                    baseline_preds = logits[0, list(pred_positions)].argmax(dim=-1).tolist()
                    for i, (draft_token, baseline_argmax) in enumerate(zip(drafts, baseline_preds)):
                        cert = certify_or_fallback_argmax(
                            hidden[0, n_pre - 1 + i],
                            int(draft_token),
                            weight,
                            clusters,
                            bias=bias,
                            tie_eps=args.tie_eps,
                            bound_slack=args.bound_slack,
                        )
                        selected = int(cert.selected_token)
                        records.append(
                            SelectiveEvalRecord(
                                task_id=task_id,
                                step_id=int(row.get("step", total_steps - 1)),
                                draft_len=draft_len,
                                token_index=i,
                                draft_token=int(draft_token),
                                baseline_argmax=int(baseline_argmax),
                                selected_token=selected,
                                certified=cert.certified,
                                full_fallback=cert.full_fallback,
                                risky_cluster_count=cert.risky_cluster_count,
                                risky_token_count=cert.risky_token_count,
                                mismatch=(selected != int(baseline_argmax)),
                            )
                        )
                append_tokens = generated_ids[generated_pos : generated_pos + emitted]
                if len(append_tokens) < emitted:
                    append_tokens = append_tokens + fallback_tokens[: emitted - len(append_tokens)]
                prefix.extend(int(x) for x in append_tokens)
                generated_pos += emitted
                crop_dynamic_cache(cache, max(0, len(prefix) - 1))
                cache_len = max(0, len(prefix) - 1)
    return records, summarize_eval(records, clusters.num_clusters, args)


def summarize_eval(records: list[SelectiveEvalRecord], num_clusters: int, args: argparse.Namespace) -> dict[str, Any]:
    vocab_size = int(args.vocab_size or 0)
    if vocab_size <= 0 and records:
        vocab_size = max(max(r.baseline_argmax, r.draft_token, r.selected_token) for r in records) + 1
    n = len(records)
    certified = sum(1 for r in records if r.certified)
    fallback = sum(1 for r in records if r.full_fallback)
    mismatches = sum(1 for r in records if r.mismatch)
    risky_tokens = [r.risky_token_count for r in records]
    risky_clusters = [r.risky_cluster_count for r in records]
    full_vocab_logits = n * vocab_size
    selective_exact_logits = sum(r.risky_token_count for r in records) + fallback * vocab_size
    cluster_bound_logits = n * num_clusters
    work_ratio = (
        (selective_exact_logits + cluster_bound_logits) / max(1, full_vocab_logits)
        if full_vocab_logits
        else 1.0
    )
    lm_head_share = _profile_share(args.lm_head_profile_json, args.lm_head_share)
    projected_speedup = 1.0 / max(1e-9, (1.0 - lm_head_share) + lm_head_share * work_ratio)
    buckets = [1, 2, 4, 8, 16, 32, 64, 128]
    by_bucket: dict[str, Any] = {}
    for b in buckets:
        rows = [r for r in records if bucket_leq(r.draft_len, buckets) == b]
        if rows:
            by_bucket[str(b)] = {
                "n_tokens": len(rows),
                "certification_rate": sum(1 for r in rows if r.certified) / len(rows),
                "fallback_rate": sum(1 for r in rows if r.full_fallback) / len(rows),
                "risky_token_mean": statistics.fmean([r.risky_token_count for r in rows]),
            }
    return {
        "draft_tokens_evaluated": n,
        "certified_tokens": certified,
        "fallback_tokens": fallback,
        "certification_rate": certified / n if n else 0.0,
        "fallback_rate": fallback / n if n else 0.0,
        "exactness_mismatches": mismatches,
        "risky_token_count": summarize([float(x) for x in risky_tokens]),
        "risky_cluster_count": summarize([float(x) for x in risky_clusters]),
        "average_exact_logits_computed": statistics.fmean(risky_tokens) if risky_tokens else 0.0,
        "full_vocab_size": vocab_size,
        "num_clusters": num_clusters,
        "selective_exact_logits": selective_exact_logits,
        "cluster_bound_logits": cluster_bound_logits,
        "full_vocab_logits": full_vocab_logits,
        "estimated_lm_head_work_ratio": work_ratio,
        "lm_head_share_used": lm_head_share,
        "projected_end_to_end_speedup": projected_speedup,
        "by_draft_bucket": by_bucket,
    }


def write_report(args: argparse.Namespace, records: list[SelectiveEvalRecord], summary: dict[str, Any]) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "target": args.target,
        "dtype": args.dtype,
        "attn": args.attn,
        "clusters": args.clusters,
        "summary": summary,
        "records_sample": [asdict(r) for r in records[:1000]],
    }
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Selective LM-Head Offline Verifier",
        "",
        f"target: `{args.target}`  dtype: `{args.dtype}`  attn: `{args.attn}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| draft tokens evaluated | {summary['draft_tokens_evaluated']} |",
        f"| certification rate | {summary['certification_rate']:.1%} |",
        f"| fallback rate | {summary['fallback_rate']:.1%} |",
        f"| exactness mismatches | {summary['exactness_mismatches']} |",
        f"| risky tokens mean | {summary['risky_token_count']['mean']:.1f} |",
        f"| estimated LM-head work ratio | {summary['estimated_lm_head_work_ratio']:.3f} |",
        f"| LM-head share used | {summary['lm_head_share_used']:.1%} |",
        f"| projected end-to-end speedup | {summary['projected_end_to_end_speedup']:.3f}x |",
        "",
        "## By Draft Bucket",
        "",
        "| bucket | tokens | certification rate | fallback rate | risky token mean |",
        "|---:|---:|---:|---:|---:|",
    ]
    for bucket, row in summary["by_draft_bucket"].items():
        lines.append(
            f"| {bucket} | {row['n_tokens']} | {row['certification_rate']:.1%} | "
            f"{row['fallback_rate']:.1%} | {row['risky_token_mean']:.1f} |"
        )
    if summary["exactness_mismatches"] != 0:
        decision = "fail: exactness mismatches found."
    elif summary["projected_end_to_end_speedup"] >= 1.20:
        decision = "pass: runtime selective verifier is worth prototyping."
    else:
        decision = "fail: offline projected speedup is below 1.20x."
    lines.append("")
    lines.append(f"Decision: **{decision}**")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    with (out_dir / "records.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", required=True)
    parser.add_argument("--completions", required=True)
    parser.add_argument("--clusters", required=True)
    parser.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--method", default="blazedit_pld_w128_n10")
    parser.add_argument("--chat-template", default="none")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tie-eps", type=float, default=1e-6)
    parser.add_argument("--bound-slack", type=float, default=1e-3)
    parser.add_argument("--lm-head-share", type=float, default=0.15)
    parser.add_argument("--lm-head-profile-json", default=None)
    parser.add_argument("--vocab-size", type=int, default=0)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but CUDA is unavailable")
    t0 = time.perf_counter()
    records, summary = run_eval(args)
    summary["wall_seconds"] = time.perf_counter() - t0
    write_report(args, records, summary)
    print((Path(args.output_dir) / "report.md").read_text())


if __name__ == "__main__":
    main()
