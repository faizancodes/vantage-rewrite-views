#!/usr/bin/env python3
"""Profile the fixed overhead in the BlazEdit PLD verifier path.

This is intentionally PLD-only.  It mirrors ``blazedit_pld_w128_n10`` while
breaking each step into lookup, input preparation, target forward, greedy
rejection, prefix extension, and KV-cache crop/update.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from asts.blazedit_decoder import prompt_lookup_draft
from asts.decoder import crop_dynamic_cache
from asts.humaneval import load_problems_from_jsonl
from asts.rejection import greedy_verify
from scripts.run_eagle_eval import _encode_prompt_ids, _load_model


@dataclass
class ProfileStep:
    task_id: str
    step: int
    draft_len: int
    accepted_len: int
    emitted: int
    rejected: bool
    match_len: int
    lookup_us: float
    input_list_us: float
    tensor_create_us: float
    pre_crop_us: float
    model_forward_cpu_us: float
    model_forward_cuda_us: float
    greedy_us: float
    extend_us: float
    post_crop_us: float
    step_wall_us: float
    python_residual_us: float
    token0_reject: bool


def _dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }[name]


def _model_dtype_arg(name: str) -> str:
    return {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "fp32": "float32",
        "float32": "float32",
    }[name]


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return float(values[idx])


def _summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": _mean(values),
        "median": float(statistics.median(values)) if values else 0.0,
        "p90": _percentile(values, 0.90),
        "p99": _percentile(values, 0.99),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _eos_truncate(tokens: list[int], eos_token_ids: set[int], budget: int) -> list[int]:
    out = list(tokens)
    for i, tok in enumerate(out):
        if tok in eos_token_ids:
            out = out[: i + 1]
            break
    return out[: max(0, budget)]


def _cuda_elapsed_ms(device: torch.device, start, end) -> float:
    if device.type != "cuda" or start is None or end is None:
        return 0.0
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end))


@torch.no_grad()
def _profile_task(
    *,
    task_id: str,
    prompt_ids: torch.Tensor,
    target,
    max_new_tokens: int,
    eos_token_ids: set[int],
    max_draft_tokens: int,
    max_matching_ngram_size: int,
    device: torch.device,
    sync_after_forward: bool,
) -> tuple[list[int], list[ProfileStep]]:
    prompt_ids_list = prompt_ids.tolist() if prompt_ids.dim() == 1 else prompt_ids[0].tolist()
    prefix: list[int] = list(prompt_ids_list)
    prompt_len = len(prefix)
    target_cache = None
    target_cache_len = 0
    steps: list[ProfileStep] = []
    step_idx = 0

    while len(prefix) < prompt_len + max_new_tokens:
        step_start = time.perf_counter_ns()
        old_prefix_len = len(prefix)

        with record_function("pld_lookup"):
            t0 = time.perf_counter_ns()
            drafts, match_len, _source_start, _follow_start = prompt_lookup_draft(
                prefix,
                max_matching_ngram_size=max_matching_ngram_size,
                max_draft_tokens=max_draft_tokens,
            )
            lookup_us = (time.perf_counter_ns() - t0) / 1000.0
        drafts = drafts[: max(0, prompt_len + max_new_tokens - old_prefix_len)]

        with record_function("cache_precrop"):
            t0 = time.perf_counter_ns()
            if target_cache_len >= old_prefix_len:
                target_cache_len = max(0, old_prefix_len - 1)
                crop_dynamic_cache(target_cache, target_cache_len)
            pre_crop_us = (time.perf_counter_ns() - t0) / 1000.0

        with record_function("verify_input_list"):
            t0 = time.perf_counter_ns()
            n_pre = old_prefix_len - target_cache_len
            target_input_list = prefix[target_cache_len:] + list(drafts)
            input_list_us = (time.perf_counter_ns() - t0) / 1000.0

        with record_function("verify_tensor_create"):
            t0 = time.perf_counter_ns()
            target_input = torch.tensor([target_input_list], device=device, dtype=torch.long)
            tensor_create_us = (time.perf_counter_ns() - t0) / 1000.0

        with record_function("target_forward"):
            start_event = end_event = None
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            t0 = time.perf_counter_ns()
            out = target(target_input, past_key_values=target_cache, use_cache=True)
            if sync_after_forward:
                _sync(device)
            model_forward_cpu_us = (time.perf_counter_ns() - t0) / 1000.0
            if device.type == "cuda":
                end_event.record()
            target_cache = out.past_key_values
            target_cache_len = target_cache_len + len(target_input_list)

        with record_function("greedy_argmax_reject"):
            t0 = time.perf_counter_ns()
            if drafts:
                result = greedy_verify(drafts=drafts, target_logits=out.logits, n_pre=n_pre)
            else:
                next_tok = int(out.logits[0, n_pre - 1].argmax(dim=-1).item())
                from asts.rejection import GreedyVerifyResult

                result = GreedyVerifyResult(
                    accepted_tokens=[next_tok],
                    n_accepted_drafts=0,
                    rejected=False,
                )
            greedy_us = (time.perf_counter_ns() - t0) / 1000.0
        model_forward_cuda_us = _cuda_elapsed_ms(device, start_event, end_event)

        with record_function("prefix_extend"):
            t0 = time.perf_counter_ns()
            accepted_capped = _eos_truncate(
                result.accepted_tokens,
                eos_token_ids,
                (prompt_len + max_new_tokens) - len(prefix),
            )
            prefix.extend(accepted_capped)
            emitted = len(accepted_capped)
            extend_us = (time.perf_counter_ns() - t0) / 1000.0

        with record_function("cache_postcrop"):
            t0 = time.perf_counter_ns()
            crop_dynamic_cache(target_cache, max(0, len(prefix) - 1))
            target_cache_len = max(0, len(prefix) - 1)
            post_crop_us = (time.perf_counter_ns() - t0) / 1000.0

        step_wall_us = (time.perf_counter_ns() - step_start) / 1000.0
        accounted = (
            lookup_us
            + pre_crop_us
            + input_list_us
            + tensor_create_us
            + model_forward_cpu_us
            + greedy_us
            + extend_us
            + post_crop_us
        )
        steps.append(
            ProfileStep(
                task_id=task_id,
                step=step_idx,
                draft_len=len(drafts),
                accepted_len=int(result.n_accepted_drafts),
                emitted=emitted,
                rejected=bool(result.rejected),
                match_len=int(match_len),
                lookup_us=lookup_us,
                input_list_us=input_list_us,
                tensor_create_us=tensor_create_us,
                pre_crop_us=pre_crop_us,
                model_forward_cpu_us=model_forward_cpu_us,
                model_forward_cuda_us=model_forward_cuda_us,
                greedy_us=greedy_us,
                extend_us=extend_us,
                post_crop_us=post_crop_us,
                step_wall_us=step_wall_us,
                python_residual_us=max(0.0, step_wall_us - accounted),
                token0_reject=bool(drafts and result.rejected and int(result.n_accepted_drafts) == 0),
            )
        )
        step_idx += 1
        if any(tok in eos_token_ids for tok in accepted_capped):
            break
    return prefix, steps


def _aggregate_steps(steps: list[ProfileStep]) -> dict[str, Any]:
    fields = [
        "lookup_us",
        "input_list_us",
        "tensor_create_us",
        "pre_crop_us",
        "model_forward_cpu_us",
        "model_forward_cuda_us",
        "greedy_us",
        "extend_us",
        "post_crop_us",
        "step_wall_us",
        "python_residual_us",
    ]
    out: dict[str, Any] = {
        "steps": len(steps),
        "draft_len_mean": _mean([float(s.draft_len) for s in steps]),
        "accepted_len_mean": _mean([float(s.accepted_len) for s in steps]),
        "emitted_mean": _mean([float(s.emitted) for s in steps]),
        "token0_reject_rate": (
            sum(1 for s in steps if s.token0_reject) / max(1, sum(1 for s in steps if s.draft_len > 0))
        ),
    }
    for field in fields:
        values = [float(getattr(s, field)) for s in steps]
        summary = _summarize(values)
        for k, v in summary.items():
            out[f"{field}_{k}"] = v
    total_wall = sum(s.step_wall_us for s in steps)
    for field in fields:
        total = sum(float(getattr(s, field)) for s in steps)
        out[f"{field}_total"] = total
        out[f"{field}_share"] = total / total_wall if total_wall > 0 else 0.0
    return out


def _write_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with (output_dir / "steps.jsonl").open("w", encoding="utf-8") as f:
        for row in report["step_records"]:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    agg = report["aggregate"]
    lines: list[str] = []
    lines.append("# BlazEdit PLD Runtime Profile\n")
    lines.append(
        f"target: `{report['target']}`  dtype: `{report['dtype']}`  "
        f"attn: `{report['attn_impl']}`  tasks: `{report['n_tasks']}`\n"
    )
    lines.append("| component | mean ms/step | p90 ms | total share |")
    lines.append("|---|---:|---:|---:|")
    for field, label in [
        ("lookup_us", "PLD lookup"),
        ("input_list_us", "candidate/input list prep"),
        ("tensor_create_us", "tensor creation/copy"),
        ("pre_crop_us", "pre-crop"),
        ("model_forward_cuda_us", "model forward CUDA"),
        ("model_forward_cpu_us", "model forward CPU enqueue"),
        ("greedy_us", "argmax/rejection"),
        ("extend_us", "prefix extend"),
        ("post_crop_us", "post-crop/cache update"),
        ("python_residual_us", "Python residual"),
        ("step_wall_us", "step wall"),
    ]:
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.1%} |".format(
                label,
                agg.get(f"{field}_mean", 0.0) / 1000.0,
                agg.get(f"{field}_p90", 0.0) / 1000.0,
                agg.get(f"{field}_share", 0.0),
            )
        )
    lines.append("\n## Decode Summary\n")
    lines.append(f"- steps: `{agg['steps']}`")
    lines.append(f"- mean draft len: `{agg['draft_len_mean']:.2f}`")
    lines.append(f"- mean accepted len: `{agg['accepted_len_mean']:.2f}`")
    lines.append(f"- mean emitted tokens/step: `{agg['emitted_mean']:.2f}`")
    lines.append(f"- token0 rejection rate: `{agg['token0_reject_rate']:.1%}`")
    if report.get("torch_profiler_table"):
        lines.append("\n## PyTorch Profiler Key Averages\n")
        lines.append("```text")
        lines.append(report["torch_profiler_table"])
        lines.append("```")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--attn-impl", default="sdpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--problem-jsonl", required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--chat-template", default="none")
    parser.add_argument("--max-draft-tokens", type=int, default=128)
    parser.add_argument("--max-matching-ngram-size", type=int, default=10)
    parser.add_argument("--torch-profiler-steps", type=int, default=20)
    parser.add_argument("--sync-after-forward", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)

    tokenizer, target = _load_model(
        args.target,
        dtype=_model_dtype_arg(args.dtype),
        attn_impl=args.attn_impl,
        trust_remote_code=args.trust_remote_code,
    )
    target.eval()
    eos = {int(tokenizer.eos_token_id)}
    problems = load_problems_from_jsonl(args.problem_jsonl, n=args.n)
    all_steps: list[ProfileStep] = []
    generated_tokens = 0
    actual_tasks = 0

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    profiler_steps_remaining = max(0, args.torch_profiler_steps)
    prof_table = ""
    prof_trace_path = ""

    prof_ctx = (
        profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        if profiler_steps_remaining > 0
        else None
    )
    if prof_ctx is not None:
        prof_ctx.__enter__()
    try:
        with torch.inference_mode():
            for problem in problems:
                prompt_ids = _encode_prompt_ids(tokenizer, problem.prompt, args.chat_template)
                _sync(device)
                before = len(all_steps)
                output, steps = _profile_task(
                    task_id=problem.task_id,
                    prompt_ids=prompt_ids,
                    target=target,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_ids=eos,
                    max_draft_tokens=args.max_draft_tokens,
                    max_matching_ngram_size=args.max_matching_ngram_size,
                    device=device,
                    sync_after_forward=args.sync_after_forward,
                )
                actual_tasks += 1
                all_steps.extend(steps)
                generated_tokens += max(0, len(output) - len(prompt_ids))
                if prof_ctx is not None:
                    for _ in range(len(all_steps) - before):
                        prof_ctx.step()
                    profiler_steps_remaining -= len(all_steps) - before
                    if profiler_steps_remaining <= 0:
                        break
    finally:
        if prof_ctx is not None:
            prof_ctx.__exit__(None, None, None)
            prof_table = prof_ctx.key_averages().table(
                sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total",
                row_limit=30,
            )
            trace_path = Path(args.output_dir) / "torch_trace.json"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            prof_ctx.export_chrome_trace(str(trace_path))
            prof_trace_path = str(trace_path)

    _sync(device)
    aggregate = _aggregate_steps(all_steps)
    wall_step_ms = aggregate.get("step_wall_us_mean", 0.0) / 1000.0
    report = {
        "target": args.target,
        "dtype": args.dtype,
        "attn_impl": args.attn_impl,
        "problem_jsonl": args.problem_jsonl,
        "n_tasks": actual_tasks,
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens": generated_tokens,
        "aggregate": aggregate,
        "torch_profiler_table": prof_table,
        "torch_profiler_trace": prof_trace_path,
        "step_wall_ms_mean": wall_step_ms,
        "step_records": [asdict(s) for s in all_steps],
    }
    _write_report(report, Path(args.output_dir))
    print((Path(args.output_dir) / "report.md").read_text())


if __name__ == "__main__":
    main()
