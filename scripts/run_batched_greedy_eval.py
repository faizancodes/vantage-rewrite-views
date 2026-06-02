#!/usr/bin/env python3
"""Generic continuous-batched greedy decoding baseline.

This is a reviewer-control baseline for Continuous Batched PLD Verification. It
does not use PLD or any speculative draft.  Each active task emits exactly one
greedy token per decode step, while target-model forwards are batched across
tasks.  The point is to separate "generic continuous batching helps" from the
PLD-specific result that batched speculative verification reduces verifier
forwards.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from asts.humaneval import Problem, load_problems_from_jsonl
from scripts.benchmark_real_shape_forward import model_dtype_arg, percentile, summarize
from scripts.run_batched_pld_eval import (
    ActiveTask,
    _compact_cache_row,
    _combine_task_caches,
    _eos_truncate,
    _hash_token_ids,
    _parse_ints,
    _prefill_task,
    _safe_token,
    _sync,
)
from scripts.run_eagle_eval import _encode_prompt_ids, _load_model


GENERIC_BATCHED_GREEDY_NAME = "greedy_batched"


@dataclass
class GreedyRunMetrics:
    method: str
    batch_size: int
    active_pool_size: int
    refill_policy: str = "continuous"
    n_tasks: int = 0
    total_new_tokens: int = 0
    wall_ms: float = 0.0
    generated_tokens_per_sec: float = 0.0
    total_forward_ms: float = 0.0
    total_prefill_ms: float = 0.0
    scheduler_overhead_ms: float = 0.0
    decode_steps: int = 0
    model_forwards: int = 0
    prefill_forwards: int = 0
    tokens_per_model_forward: float = 0.0
    memory_peak_gb: float = 0.0
    output_match_count: int = 0
    output_mismatch_count: int = 0
    active_tasks_mean: float = 0.0
    active_tasks_p50: float = 0.0
    active_tasks_p90: float = 0.0
    task_latency_ms: dict[str, float] = field(default_factory=dict)
    task_latency_summary_ms: dict[str, float] = field(default_factory=dict)
    output_equivalence_note: str = ""
    error: str = ""


def _eos_ids(tokenizer, target) -> list[int]:
    eos_token_ids: list[int] = []
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos_token_ids.append(int(tokenizer.eos_token_id))
    raw = getattr(getattr(target, "config", None), "eos_token_id", None)
    if raw is not None:
        if isinstance(raw, list):
            eos_token_ids.extend(int(x) for x in raw)
        else:
            eos_token_ids.append(int(raw))
    return sorted(set(eos_token_ids))


def _new_task(prob: Problem, tokenizer, chat_template: str) -> ActiveTask:
    prompt_ids = _encode_prompt_ids(tokenizer, prob.prompt, chat_template)
    prompt_list = [int(x) for x in prompt_ids.tolist()]
    return ActiveTask(
        task_id=prob.task_id,
        prompt=prob.prompt,
        prompt_ids=prompt_list,
        prefix=list(prompt_list),
        prompt_len=len(prompt_list),
        start_ns=time.perf_counter_ns(),
    )


def _make_greedy_tensors(
    tasks: list[ActiveTask],
    *,
    max_cache_len: int,
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]]]:
    batch = len(tasks)
    input_ids = torch.full((batch, 1), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch, max_cache_len + 1), dtype=torch.long, device=device)
    position_ids = torch.zeros((batch, 1), dtype=torch.long, device=device)
    row_tokens: list[list[int]] = []
    for row, task in enumerate(tasks):
        if not task.prefix:
            raise RuntimeError(f"empty prefix for task {task.task_id}")
        cache_n = int(task.target_cache_len)
        tok = int(task.prefix[-1])
        input_ids[row, 0] = tok
        if cache_n:
            attention_mask[row, :cache_n] = 1
        attention_mask[row, max_cache_len] = 1
        position_ids[row, 0] = cache_n
        row_tokens.append([tok])
    return input_ids, attention_mask, position_ids, row_tokens


def _run_batched_greedy_forward(
    *,
    target,
    tasks: list[ActiveTask],
    pad_id: int,
    eos_token_ids: list[int],
    max_new_tokens: int,
    device: torch.device,
) -> tuple[float, int]:
    batched_cache, max_cache_len = _combine_task_caches(tasks)
    input_ids, attention_mask, position_ids, row_tokens = _make_greedy_tensors(
        tasks,
        max_cache_len=max_cache_len,
        pad_id=pad_id,
        device=device,
    )
    _sync(device)
    t0 = time.perf_counter_ns()
    with torch.inference_mode():
        out = target(
            input_ids,
            past_key_values=batched_cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
    _sync(device)
    forward_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

    emitted = 0
    next_tokens = out.logits[:, -1, :].argmax(dim=-1).detach().cpu().tolist()
    for row, task in enumerate(tasks):
        if task.finished:
            raise RuntimeError(f"finished task re-entered greedy batch: {task.task_id}")
        budget = (task.prompt_len + max_new_tokens) - len(task.prefix)
        accepted_capped, hit_eos = _eos_truncate([int(next_tokens[row])], eos_token_ids, budget)
        prefix_len_before = len(task.prefix)
        task.target_cache = _compact_cache_row(
            out.past_key_values,
            row=row,
            real_cache_len=int(task.target_cache_len),
            max_cache_len=max_cache_len,
            keep_input_len=1,
        )
        task.prefix.extend(accepted_capped)
        task.target_cache_len = max(0, len(task.prefix) - 1)
        task.generated_tokens = len(task.prefix) - task.prompt_len
        task.steps += 1
        emitted += len(accepted_capped)
        if hit_eos:
            task.finished = True
            task.finish_reason = "eos"
        elif len(task.prefix) >= task.prompt_len + max_new_tokens:
            task.finished = True
            task.finish_reason = "max_new_tokens"
        if len(task.prefix) < prefix_len_before:
            raise RuntimeError(f"prefix regressed for task {task.task_id}")
    return forward_ms, emitted


def run_sequential_greedy(
    *,
    problems: list[Problem],
    tokenizer,
    target,
    max_new_tokens: int,
    eos_token_ids: list[int],
    chat_template: str,
    device: torch.device,
    progress_every_tasks: int = 0,
) -> dict[str, Any]:
    outputs: dict[str, list[int]] = {}
    total_new = 0
    steps = 0
    prefill_ms = 0.0
    forward_ms = 0.0
    latencies: list[float] = []
    pad_id = _safe_token(tokenizer, target)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.perf_counter_ns()
    for idx, prob in enumerate(problems, start=1):
        task = _new_task(prob, tokenizer, chat_template)
        prefill_ms += _prefill_task(task, target, device)
        while len(task.prefix) - task.prompt_len < max_new_tokens and not task.finished:
            ms, emitted = _run_batched_greedy_forward(
                target=target,
                tasks=[task],
                pad_id=pad_id,
                eos_token_ids=eos_token_ids,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            forward_ms += ms
            total_new += emitted
            steps += 1
        task.end_ns = time.perf_counter_ns()
        task.latency_ms = (task.end_ns - task.start_ns) / 1_000_000.0
        latencies.append(task.latency_ms)
        outputs[task.task_id] = list(task.prefix[task.prompt_len:])
        if progress_every_tasks > 0 and (idx % progress_every_tasks == 0 or idx == len(problems)):
            elapsed_ms = (time.perf_counter_ns() - t_start) / 1_000_000.0
            print(
                "[sequential] "
                f"completed={idx}/{len(problems)} "
                f"tokens={total_new} "
                f"forwards={steps} "
                f"tok/s={total_new / max(1e-9, elapsed_ms / 1000.0):.1f}",
                flush=True,
            )
    wall_ms = (time.perf_counter_ns() - t_start) / 1_000_000.0
    return {
        "method": "greedy_sequential",
        "batch_size": 1,
        "active_pool_size": 1,
        "wall_ms": wall_ms,
        "tokens": total_new,
        "tokens_per_sec": total_new / max(1e-9, wall_ms / 1000.0),
        "steps": steps,
        "model_forwards": steps,
        "prefill_forwards": len(problems),
        "total_forward_ms": forward_ms,
        "total_prefill_ms": prefill_ms,
        "scheduler_overhead_ms": max(0.0, wall_ms - forward_ms - prefill_ms),
        "tokens_per_model_forward": total_new / max(1, steps),
        "memory_peak_gb": (
            torch.cuda.max_memory_allocated(device) / (1024.0**3)
            if device.type == "cuda"
            else 0.0
        ),
        "task_latency_summary_ms": summarize(latencies),
        "outputs": outputs,
    }


def run_batched_greedy_scheduler(
    *,
    problems: list[Problem],
    tokenizer,
    target,
    max_new_tokens: int,
    eos_token_ids: list[int],
    chat_template: str,
    batch_size: int,
    active_pool_size: int,
    baseline_outputs: dict[str, list[int]] | None,
    device: torch.device,
    refill_policy: str = "continuous",
    progress_every_tasks: int = 0,
    progress_every_scheduler_steps: int = 0,
) -> tuple[GreedyRunMetrics, dict[str, list[int]]]:
    if refill_policy not in {"continuous", "no_refill"}:
        raise ValueError(f"unsupported refill_policy: {refill_policy}")
    pad_id = _safe_token(tokenizer, target)
    pending = deque(problems)
    active: list[ActiveTask] = []
    completed: list[ActiveTask] = []
    outputs: dict[str, list[int]] = {}
    prefill_ms = 0.0
    forward_ms = 0.0
    emitted_total = 0
    model_forwards = 0
    active_counts: list[int] = []
    scheduler_ticks = 0
    last_completed_log = 0

    def refill() -> None:
        nonlocal prefill_ms
        while len(active) < active_pool_size and pending:
            task = _new_task(pending.popleft(), tokenizer, chat_template)
            prefill_ms += _prefill_task(task, target, device)
            active.append(task)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.perf_counter_ns()
    refill()
    while active:
        scheduler_ticks += 1
        active_counts.append(len(active))
        for i in range(0, len(active), batch_size):
            chunk = active[i : i + batch_size]
            ms, emitted = _run_batched_greedy_forward(
                target=target,
                tasks=chunk,
                pad_id=pad_id,
                eos_token_ids=eos_token_ids,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            forward_ms += ms
            emitted_total += emitted
            model_forwards += 1

        still_active: list[ActiveTask] = []
        for task in active:
            if task.finished:
                task.end_ns = time.perf_counter_ns()
                task.latency_ms = (task.end_ns - task.start_ns) / 1_000_000.0
                completed.append(task)
                outputs[task.task_id] = list(task.prefix[task.prompt_len:])
            else:
                still_active.append(task)
        active = still_active
        if refill_policy == "continuous" or not active:
            refill()
        if progress_every_tasks > 0 and (
            len(completed) - last_completed_log >= progress_every_tasks
            or len(completed) == len(problems)
        ):
            last_completed_log = len(completed)
            elapsed_ms = (time.perf_counter_ns() - t_start) / 1_000_000.0
            print(
                "[batched] "
                f"batch={batch_size} pool={active_pool_size} "
                f"completed={len(completed)}/{len(problems)} "
                f"active={len(active)} pending={len(pending)} "
                f"tokens={emitted_total} forwards={model_forwards} "
                f"tok/s={emitted_total / max(1e-9, elapsed_ms / 1000.0):.1f}",
                flush=True,
            )
        elif progress_every_scheduler_steps > 0 and scheduler_ticks % progress_every_scheduler_steps == 0:
            elapsed_ms = (time.perf_counter_ns() - t_start) / 1_000_000.0
            print(
                "[batched] "
                f"batch={batch_size} pool={active_pool_size} "
                f"ticks={scheduler_ticks} completed={len(completed)}/{len(problems)} "
                f"active={len(active)} pending={len(pending)} "
                f"tokens={emitted_total} forwards={model_forwards} "
                f"tok/s={emitted_total / max(1e-9, elapsed_ms / 1000.0):.1f}",
                flush=True,
            )

    wall_ms = (time.perf_counter_ns() - t_start) / 1_000_000.0
    match_count = 0
    mismatch_count = 0
    if baseline_outputs is not None:
        for task_id, toks in outputs.items():
            if baseline_outputs.get(task_id) == toks:
                match_count += 1
            else:
                mismatch_count += 1
    latencies = [t.latency_ms for t in completed]
    metrics = GreedyRunMetrics(
        method=GENERIC_BATCHED_GREEDY_NAME,
        batch_size=batch_size,
        active_pool_size=active_pool_size,
        refill_policy=refill_policy,
        n_tasks=len(problems),
        total_new_tokens=sum(len(v) for v in outputs.values()),
        wall_ms=wall_ms,
        generated_tokens_per_sec=sum(len(v) for v in outputs.values()) / max(1e-9, wall_ms / 1000.0),
        total_forward_ms=forward_ms,
        total_prefill_ms=prefill_ms,
        scheduler_overhead_ms=max(0.0, wall_ms - forward_ms - prefill_ms),
        decode_steps=sum(t.steps for t in completed),
        model_forwards=model_forwards,
        prefill_forwards=len(problems),
        tokens_per_model_forward=emitted_total / max(1, model_forwards),
        memory_peak_gb=(
            torch.cuda.max_memory_allocated(device) / (1024.0**3)
            if device.type == "cuda"
            else 0.0
        ),
        output_match_count=match_count,
        output_mismatch_count=mismatch_count,
        active_tasks_mean=float(statistics.fmean(active_counts)) if active_counts else 0.0,
        active_tasks_p50=float(statistics.median(active_counts)) if active_counts else 0.0,
        active_tasks_p90=percentile([float(x) for x in active_counts], 0.90) if active_counts else 0.0,
        task_latency_ms={t.task_id: t.latency_ms for t in completed},
        task_latency_summary_ms=summarize(latencies),
        output_equivalence_note=(
            "Compared token IDs against sequential greedy in the same run. "
            "Deterministic exactness should cite fp32/eager validation."
        ),
    )
    return metrics, outputs


def write_report(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    sequential: dict[str, Any],
    batched: list[GreedyRunMetrics],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args),
        "sequential": {k: v for k, v in sequential.items() if k != "outputs"},
        "batched": [asdict(x) for x in batched],
    }
    (output_dir / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    seq_tps = float(sequential["tokens_per_sec"])
    seq_match = f"{args.n}/{args.n}" if not getattr(args, "skip_sequential", False) else "baseline report"
    lines = [
        "# Generic Continuous-Batched Greedy Baseline",
        "",
        f"target: `{args.target}`  dtype: `{args.dtype}`  attn: `{args.attn}`  n: `{args.n}`",
        "",
        "This baseline uses ordinary greedy autoregressive decoding, without PLD or speculative drafts.",
        "",
        "| method | batch | active pool | tok/s | speedup vs greedy seq | model forwards | total tokens | output matches | latency p50 ms | peak GB | status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| greedy_sequential | 1 | 1 | {seq_tps:.1f} | 1.000 | "
            f"{int(sequential['model_forwards'])} | {int(sequential['tokens'])} | "
            f"{seq_match} | {sequential['task_latency_summary_ms'].get('p50', 0.0):.1f} | "
            f"{sequential.get('memory_peak_gb', 0.0):.2f} | success |"
        ),
    ]
    for row in batched:
        status = "failed" if row.error else "success"
        compared = row.output_match_count + row.output_mismatch_count
        match_text = (
            f"{row.output_match_count}/{compared}" if compared else "not compared"
        )
        lines.append(
            f"| greedy_batched | {row.batch_size} | {row.active_pool_size} | "
            f"{row.generated_tokens_per_sec:.1f} | "
            f"{row.generated_tokens_per_sec / max(1e-9, seq_tps):.3f} | "
            f"{row.model_forwards} | {row.total_new_tokens} | "
            f"{match_text} | "
            f"{row.task_latency_summary_ms.get('p50', 0.0):.1f} | "
            f"{row.memory_peak_gb:.2f} | {status} |"
        )
        if row.error:
            lines.append(f"<!-- batch {row.batch_size} error: {row.error} -->")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This control answers whether the final result is merely generic batching. "
            "The paper comparison should report generic batched greedy separately from "
            "Continuous Batched PLD, because PLD changes the number of target verification "
            "forwards while greedy batching only batches one-token autoregressive steps.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print((output_dir / "report.md").read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-jsonl", required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--chat-template", default="none")
    parser.add_argument("--batch-sizes", default="2,4,8")
    parser.add_argument("--active-pool-size", type=int, default=32)
    parser.add_argument("--refill-policy", choices=["continuous", "no_refill"], default="continuous")
    parser.add_argument("--skip-sequential", action="store_true")
    parser.add_argument("--baseline-report", default="")
    parser.add_argument("--progress-every-tasks", type=int, default=25)
    parser.add_argument("--progress-every-scheduler-steps", type=int, default=1000)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _load_sequential_from_report(path: str) -> dict[str, Any]:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"baseline report not found: {src}")
    payload = json.loads(src.read_text(encoding="utf-8"))
    seq = dict(payload.get("sequential") or {})
    if not seq:
        raise ValueError(f"baseline report does not contain sequential metrics: {src}")
    seq.setdefault("outputs", {})
    seq.setdefault("tokens_per_sec", seq.get("generated_tokens_per_sec", 0.0))
    seq.setdefault("tokens", seq.get("total_new_tokens", 0))
    seq.setdefault("model_forwards", seq.get("steps", 0))
    seq.setdefault("task_latency_summary_ms", {})
    seq.setdefault("memory_peak_gb", 0.0)
    return seq


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    tokenizer, target = _load_model(
        args.target,
        dtype=model_dtype_arg(args.dtype),
        attn_impl=args.attn,
    )
    target.eval()
    eos_token_ids = _eos_ids(tokenizer, target)
    problems = load_problems_from_jsonl(args.problem_jsonl, n=args.n)
    batch_sizes = _parse_ints(args.batch_sizes)

    if args.skip_sequential:
        if not args.baseline_report:
            raise SystemExit("--skip-sequential requires --baseline-report")
        sequential = _load_sequential_from_report(args.baseline_report)
        print(
            "[sequential] skipped; using baseline report "
            f"{args.baseline_report} tok/s={float(sequential.get('tokens_per_sec', 0.0)):.1f}",
            flush=True,
        )
    else:
        print("[sequential] running greedy_sequential", flush=True)
        sequential = run_sequential_greedy(
            problems=problems,
            tokenizer=tokenizer,
            target=target,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos_token_ids,
            chat_template=args.chat_template,
            device=device,
            progress_every_tasks=args.progress_every_tasks,
        )
    rows: list[GreedyRunMetrics] = []
    for batch_size in batch_sizes:
        print(f"[batched] greedy batch={batch_size} active_pool={args.active_pool_size}", flush=True)
        try:
            metrics, _outputs = run_batched_greedy_scheduler(
                problems=problems,
                tokenizer=tokenizer,
                target=target,
                max_new_tokens=args.max_new_tokens,
                eos_token_ids=eos_token_ids,
                chat_template=args.chat_template,
                batch_size=batch_size,
                active_pool_size=args.active_pool_size,
                baseline_outputs=sequential.get("outputs") or None,
                device=device,
                refill_policy=args.refill_policy,
                progress_every_tasks=args.progress_every_tasks,
                progress_every_scheduler_steps=args.progress_every_scheduler_steps,
            )
        except Exception as exc:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            metrics = GreedyRunMetrics(
                method=GENERIC_BATCHED_GREEDY_NAME,
                batch_size=batch_size,
                active_pool_size=args.active_pool_size,
                n_tasks=len(problems),
                error=f"{type(exc).__name__}: {exc}",
            )
            print(f"[batched] greedy batch={batch_size} failed: {metrics.error}", flush=True)
        rows.append(metrics)
    write_report(
        output_dir=Path(args.output_dir),
        args=args,
        sequential=sequential,
        batched=rows,
    )


if __name__ == "__main__":
    main()
