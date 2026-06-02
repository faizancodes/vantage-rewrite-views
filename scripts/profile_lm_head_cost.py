#!/usr/bin/env python3
"""Isolate backbone, LM-head, and argmax cost on real PLD verifier shapes."""

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
from asts.rejection import greedy_verify
from scripts.benchmark_real_shape_forward import (
    bucket_leq,
    load_trace,
    make_drafts,
    model_dtype_arg,
    percentile,
    summarize,
)
from scripts.run_eagle_eval import _encode_prompt_ids, _load_model


@dataclass
class LMHeadCostRecord:
    task_id: str
    step_id: int
    prefix_len: int
    draft_len: int
    input_tokens: int
    first_step_for_task: bool
    full_logits_forward_ms: float
    transformer_backbone_only_ms: float
    lm_head_only_ms: float
    argmax_ms: float
    full_verify_ms: float


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def encode_completion_tokens(tokenizer, completion_row: dict[str, Any], method: str) -> list[int]:
    output = completion_row.get("outputs", {}).get(method)
    if not output:
        return []
    text = output.get("raw_text", output.get("text", ""))
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
    n_new = output.get("n_new_tokens")
    if isinstance(n_new, int) and n_new >= 0:
        ids = ids[:n_new]
    return [int(x) for x in ids]


def _last_hidden(backbone_out) -> torch.Tensor:
    if hasattr(backbone_out, "last_hidden_state"):
        return backbone_out.last_hidden_state
    return backbone_out[0]


def _past(backbone_out):
    if hasattr(backbone_out, "past_key_values"):
        return backbone_out.past_key_values
    return backbone_out[1] if len(backbone_out) > 1 else None


def _replay_records(
    *,
    tokenizer,
    target,
    steps_by_task: dict[str, list[dict[str, Any]]],
    completions: dict[str, dict[str, Any]],
    method: str,
    max_steps: int,
    chat_template: str,
    device: torch.device,
) -> list[LMHeadCostRecord]:
    if not hasattr(target, "model") or not hasattr(target, "lm_head"):
        raise SystemExit("target model must expose .model and .lm_head for split profiling")
    records: list[LMHeadCostRecord] = []
    fallback_tokens = tokenizer(
        "def _fallback_value():\n    return None\n",
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0].tolist()

    def run_mode(split: bool) -> dict[tuple[str, int], tuple[float, float, float, float, int, int, bool]]:
        result: dict[tuple[str, int], tuple[float, float, float, float, int, int, bool]] = {}
        with torch.inference_mode():
            for task_id, task_steps in steps_by_task.items():
                completion = completions.get(task_id)
                if not completion:
                    continue
                prompt_ids = _encode_prompt_ids(tokenizer, completion["prompt"], chat_template)
                prefix = [int(x) for x in prompt_ids.tolist()]
                generated_ids = encode_completion_tokens(tokenizer, completion, method)
                generated_pos = 0
                cache = None
                cache_len = 0
                for row in task_steps:
                    if max_steps and len(result) >= max_steps:
                        return result
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
                    target_input = torch.tensor(
                        [prefix[cache_len:] + drafts],
                        device=device,
                        dtype=torch.long,
                    )
                    if split:
                        sync(device)
                        t0 = time.perf_counter_ns()
                        backbone_out = target.model(
                            input_ids=target_input,
                            past_key_values=cache,
                            use_cache=True,
                        )
                        sync(device)
                        backbone_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
                        hidden = _last_hidden(backbone_out)
                        sync(device)
                        t1 = time.perf_counter_ns()
                        logits = target.lm_head(hidden)
                        sync(device)
                        lm_head_ms = (time.perf_counter_ns() - t1) / 1_000_000.0
                        cache = _past(backbone_out)
                        full_forward_ms = backbone_ms + lm_head_ms
                    else:
                        sync(device)
                        t0 = time.perf_counter_ns()
                        out = target(target_input, past_key_values=cache, use_cache=True)
                        sync(device)
                        full_forward_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
                        backbone_ms = 0.0
                        lm_head_ms = 0.0
                        logits = out.logits
                        cache = out.past_key_values
                    cache_len = cache_len + int(target_input.shape[1])
                    t_arg = time.perf_counter_ns()
                    if drafts:
                        _ = greedy_verify(drafts=drafts, target_logits=logits, n_pre=n_pre)
                    else:
                        _ = int(logits[0, n_pre - 1].argmax(dim=-1).item())
                    sync(device)
                    argmax_ms = (time.perf_counter_ns() - t_arg) / 1_000_000.0
                    append_tokens = generated_ids[generated_pos : generated_pos + emitted]
                    if len(append_tokens) < emitted:
                        append_tokens = append_tokens + fallback_tokens[: emitted - len(append_tokens)]
                    prefix.extend(int(x) for x in append_tokens)
                    generated_pos += emitted
                    crop_dynamic_cache(cache, max(0, len(prefix) - 1))
                    cache_len = max(0, len(prefix) - 1)
                    result[(task_id, int(row.get("step", len(result))))] = (
                        full_forward_ms,
                        backbone_ms,
                        lm_head_ms,
                        argmax_ms,
                        old_prefix_len,
                        draft_len,
                        int(row.get("step", 0)) == 0,
                    )
        return result

    full = run_mode(split=False)
    split = run_mode(split=True)
    for key, full_vals in full.items():
        if key not in split:
            continue
        full_forward_ms, _, _, full_argmax_ms, prefix_len, draft_len, first_step = full_vals
        _, backbone_ms, lm_head_ms, split_argmax_ms, _, _, _ = split[key]
        records.append(
            LMHeadCostRecord(
                task_id=key[0],
                step_id=key[1],
                prefix_len=prefix_len,
                draft_len=draft_len,
                input_tokens=draft_len + 1,
                first_step_for_task=first_step,
                full_logits_forward_ms=full_forward_ms,
                transformer_backbone_only_ms=backbone_ms,
                lm_head_only_ms=lm_head_ms,
                argmax_ms=split_argmax_ms or full_argmax_ms,
                full_verify_ms=full_forward_ms + full_argmax_ms,
            )
        )
    return records


def _aggregate(records: list[LMHeadCostRecord]) -> dict[str, Any]:
    out = {
        "n_steps": len(records),
        "full_logits_forward_ms": summarize([r.full_logits_forward_ms for r in records]),
        "transformer_backbone_only_ms": summarize([r.transformer_backbone_only_ms for r in records]),
        "lm_head_only_ms": summarize([r.lm_head_only_ms for r in records]),
        "argmax_ms": summarize([r.argmax_ms for r in records]),
        "full_verify_ms": summarize([r.full_verify_ms for r in records]),
    }
    full_mean = out["full_logits_forward_ms"]["mean"]
    out["lm_head_share_of_full_forward"] = (
        out["lm_head_only_ms"]["mean"] / full_mean if full_mean > 0 else 0.0
    )
    buckets = [0, 1, 2, 4, 8, 16, 32, 64, 128]
    by_bucket: dict[str, Any] = {}
    for b in buckets:
        vals = [
            r
            for r in records
            if (r.draft_len == b if b == 0 else bucket_leq(r.draft_len, buckets[1:]) == b)
        ]
        if vals:
            lm = [r.lm_head_only_ms for r in vals]
            full = [r.full_logits_forward_ms for r in vals]
            by_bucket[str(b)] = {
                "n": len(vals),
                "lm_head_ms": summarize(lm),
                "full_forward_ms": summarize(full),
                "lm_head_share": statistics.fmean(lm) / max(1e-9, statistics.fmean(full)),
            }
    out["by_draft_bucket"] = by_bucket
    cached = [r for r in records if not r.first_step_for_task]
    out["cached_lm_head_share"] = (
        statistics.fmean([r.lm_head_only_ms for r in cached])
        / max(1e-9, statistics.fmean([r.full_logits_forward_ms for r in cached]))
        if cached
        else 0.0
    )
    return out


def _write_report(path: Path, records: list[LMHeadCostRecord], aggregate: dict[str, Any], args) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "target": args.target,
        "dtype": args.dtype,
        "attn": args.attn,
        "method": args.method,
        "aggregate": aggregate,
        "records": [asdict(r) for r in records],
    }
    (path / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# LM-Head Cost Profile",
        "",
        f"target: `{args.target}`  dtype: `{args.dtype}`  attn: `{args.attn}`  steps: `{len(records)}`",
        "",
        "| metric | mean ms | p50 | p90 | p99 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in [
        "transformer_backbone_only_ms",
        "lm_head_only_ms",
        "full_logits_forward_ms",
        "argmax_ms",
        "full_verify_ms",
    ]:
        row = aggregate[name]
        lines.append(f"| {name} | {row['mean']:.3f} | {row['p50']:.3f} | {row['p90']:.3f} | {row['p99']:.3f} |")
    lines.append("")
    lines.append(f"LM-head share of full forward: `{aggregate['lm_head_share_of_full_forward']:.1%}`")
    lines.append(f"Cached-step LM-head share: `{aggregate['cached_lm_head_share']:.1%}`")
    lines.append("")
    lines.append("## By Draft Bucket")
    lines.append("")
    lines.append("| bucket | n | lm_head mean ms | full mean ms | lm_head share |")
    lines.append("|---:|---:|---:|---:|---:|")
    for bucket, row in aggregate["by_draft_bucket"].items():
        lines.append(
            f"| {bucket} | {row['n']} | {row['lm_head_ms']['mean']:.3f} | "
            f"{row['full_forward_ms']['mean']:.3f} | {row['lm_head_share']:.1%} |"
        )
    share = aggregate["lm_head_share_of_full_forward"]
    if share < 0.10:
        decision = "LM-head share is below 10%; selective LM-head is low priority."
    elif share >= 0.15:
        decision = "LM-head share is at least 15%; continue selective-certification diagnostics."
    else:
        decision = "LM-head share is 10-15%; run a cheap certification diagnostic before runtime work."
    lines.append("")
    lines.append(f"Decision: **{decision}**")
    (path / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", required=True)
    parser.add_argument("--completions", required=True)
    parser.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--method", default="blazedit_pld_w128_n10")
    parser.add_argument("--chat-template", default="none")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but CUDA is unavailable")
    device = torch.device(args.device)
    tokenizer, target = _load_model(
        args.target,
        dtype=model_dtype_arg(args.dtype),
        attn_impl=args.attn,
    )
    target.eval()
    steps_by_task, completions = load_trace(
        steps_path=args.steps,
        completions_path=args.completions,
        method=args.method,
    )
    records = _replay_records(
        tokenizer=tokenizer,
        target=target,
        steps_by_task=steps_by_task,
        completions=completions,
        method=args.method,
        max_steps=args.max_steps,
        chat_template=args.chat_template,
        device=device,
    )
    aggregate = _aggregate(records)
    _write_report(Path(args.output_dir), records, aggregate, args)
    print((Path(args.output_dir) / "report.md").read_text())


if __name__ == "__main__":
    main()
