#!/usr/bin/env python3
"""Offline shadow-candidate oracle for skipped VANTAGE-MV transformed probes.

The runtime decoder deliberately avoids building transformed views on many
steps.  This script reconstructs those skipped prefixes from an existing
completion/step trace and asks a counterfactual question:

    If transformed lookup had been built here, how many next output tokens
    would its best candidate have matched?

The oracle is CPU-only.  It reuses the decoder's prompt/map parsing and view
construction, but it ignores runtime build gates and enumerates candidate
matches directly from transformed/FST indexes.  The output is diagnostic only;
it is not a lossless decode replay.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asts.blazedit_decoder import (  # noqa: E402
    VantageMVConfig,
    _build_mv_lazy_plan,
    _build_mv_views_from_plan,
    _frontier_relation,
    _frontier_relation_in_positions,
    _mv_effective_margin,
    _MVViewStats,
    _MVCandidate,
    parse_vantage_mv_method,
    prompt_lookup_draft,
)
from asts.code_proposers import encode_no_special  # noqa: E402


DEFAULT_SKIP_REASONS = {
    "trans_precheck_no_token_candidate",
    "no_rewrite_frontier_signal",
    "trans_precheck_margin_impossible",
}


@dataclass
class ShadowCandidate:
    tokens: list[int]
    view_id: str
    source_label: str
    match_len: int
    source_start: int
    follow_start: int
    frontier_distance: int | None
    crosses_frontier: bool
    pair_keys: tuple[str, ...] = ()


@dataclass
class ShadowHit:
    task_id: str
    step: int
    reason: str
    shadow_kind: str
    accepted: int
    incremental_accepted: int
    candidate_len: int
    match_len: int
    current_accepted: int
    current_emitted: int
    current_wall_us: float
    exact_len: int
    exact_match_len: int
    runtime_eligible: bool
    view_id: str
    source_label: str
    frontier_distance: int | None
    crosses_frontier: bool


@dataclass
class TaskMethodTrace:
    task_id: str
    method: str
    prompt: str
    reference: str
    metadata: dict[str, Any]
    output_text: str
    output_tokens: list[int]
    prompt_tokens: list[int]
    steps: list[dict[str, Any]]
    starts: list[int] = field(default_factory=list)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _encode_prompt_ids(tokenizer, prompt: str, chat_template: str) -> list[int]:
    mode = (chat_template or "none").strip().lower()
    if mode in {"", "none", "raw"}:
        return encode_no_special(tokenizer, prompt)
    if mode in {"user", "chat", "single_user"}:
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("--chat-template requested, but tokenizer has no chat_template")
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
        )
        if isinstance(encoded, dict):
            encoded = encoded["input_ids"]
        return list(encoded)
    raise ValueError(f"unsupported chat template mode: {chat_template!r}")


def _output_tokens(tokenizer, output: dict[str, Any]) -> list[int]:
    token_ids = output.get("tokens") or output.get("token_ids") or output.get("completion_token_ids")
    if isinstance(token_ids, list) and all(isinstance(x, int) for x in token_ids):
        return list(token_ids)
    text = str(output.get("raw_text") if output.get("raw_text") is not None else output.get("text") or "")
    return encode_no_special(tokenizer, text)


def _load_traces(
    *,
    tokenizer,
    completions_path: Path,
    steps_path: Path,
    method: str,
    chat_template: str,
    task_limit: int | None,
) -> tuple[list[TaskMethodTrace], dict[str, Any]]:
    rows = _load_jsonl(completions_path)
    if task_limit is not None:
        rows = rows[:task_limit]

    step_rows = [
        r
        for r in _load_jsonl(steps_path)
        if r.get("method") == method
    ]
    by_task_steps: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in step_rows:
        by_task_steps[str(row.get("task_id"))].append(row)
    for task_steps in by_task_steps.values():
        task_steps.sort(key=lambda r: int(r.get("step") or 0))

    traces: list[TaskMethodTrace] = []
    reconstruction = {
        "tasks_seen": len(rows),
        "tasks_with_method": 0,
        "tasks_with_steps": 0,
        "token_length_mismatches": 0,
        "max_token_length_delta": 0,
    }
    for row in rows:
        task_id = str(row.get("task_id"))
        outputs = row.get("outputs") if isinstance(row.get("outputs"), dict) else {}
        output = outputs.get(method)
        if not isinstance(output, dict):
            continue
        reconstruction["tasks_with_method"] += 1
        steps = by_task_steps.get(task_id, [])
        if not steps:
            continue
        reconstruction["tasks_with_steps"] += 1
        prompt = str(row.get("prompt") or "")
        out_tokens = _output_tokens(tokenizer, output)
        n_new = int(output.get("n_new_tokens") or len(out_tokens))
        delta = abs(n_new - len(out_tokens))
        if delta:
            reconstruction["token_length_mismatches"] += 1
            reconstruction["max_token_length_delta"] = max(
                reconstruction["max_token_length_delta"],
                delta,
            )
        trace = TaskMethodTrace(
            task_id=task_id,
            method=method,
            prompt=prompt,
            reference=str(row.get("reference") or ""),
            metadata=row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
            output_text=str(output.get("raw_text") if output.get("raw_text") is not None else output.get("text") or ""),
            output_tokens=out_tokens,
            prompt_tokens=_encode_prompt_ids(tokenizer, prompt, chat_template),
            steps=steps,
        )
        emitted = 0
        for step in steps:
            trace.starts.append(emitted)
            emitted += int(step.get("n_emitted") or 0)
        traces.append(trace)
    return traces, reconstruction


def _common_prefix_len(left: list[int], right: list[int]) -> int:
    n = min(len(left), len(right))
    i = 0
    while i < n and left[i] == right[i]:
        i += 1
    return i


def _pair_keys_for_value_span(view, start: int, follow_end: int) -> tuple[str, ...]:
    spans = view.value_pair_spans if getattr(view, "transducer", False) else getattr(view, "pair_spans", [])
    return tuple(key for lo, hi, key in spans if start < hi and lo < follow_end)


def _frontier_for_candidate(view, start: int, follow_start: int, follow_end: int, config: VantageMVConfig) -> tuple[int | None, bool]:
    if getattr(view, "transducer", False):
        return _frontier_relation_in_positions(
            frontiers=view.value_frontiers,
            source_start=start,
            follow_start=follow_start,
            follow_end=follow_end,
            window=config.frontier_window,
        )
    return _frontier_relation(
        view=view,
        source_start=start,
        follow_start=follow_start,
        follow_end=follow_end,
        window=config.frontier_window,
    )


def _enumerate_shadow_candidates(
    *,
    prefix: list[int],
    views,
    config: VantageMVConfig,
    max_draft_tokens: int,
) -> list[ShadowCandidate]:
    candidates: list[ShadowCandidate] = []
    max_n = min(config.max_matching_ngram_size, len(prefix))
    if max_n < config.transformed_min_matching_ngram_size:
        return candidates
    for view in views:
        tokens = view.value_tokens if getattr(view, "transducer", False) else view.tokens
        index = view.value_index if getattr(view, "transducer", False) else view.index
        if not tokens or not index:
            continue
        max_view_n = min(max_n, len(tokens) - 1)
        for n in range(max_view_n, config.transformed_min_matching_ngram_size - 1, -1):
            starts = index.get(n, {}).get(tuple(prefix[-n:]))
            if not starts:
                continue
            for start in reversed(starts):
                follow_start = start + n
                follow_end = min(follow_start + max_draft_tokens, len(tokens))
                if follow_start >= follow_end:
                    continue
                distance, crosses = _frontier_for_candidate(
                    view,
                    start,
                    follow_start,
                    follow_end,
                    config,
                )
                candidates.append(
                    ShadowCandidate(
                        tokens=list(tokens[follow_start:follow_end]),
                        view_id=str(view.view_id),
                        source_label=str(view.source_label),
                        match_len=n,
                        source_start=start,
                        follow_start=follow_start,
                        frontier_distance=distance,
                        crosses_frontier=crosses,
                        pair_keys=_pair_keys_for_value_span(view, start, follow_end),
                    )
                )
    return candidates


def _runtime_eligible(
    *,
    tokenizer,
    candidate: ShadowCandidate,
    view,
    exact_len: int,
    config: VantageMVConfig,
) -> bool:
    stats = _MVViewStats()
    mv_candidate = _MVCandidate(
        tokens=candidate.tokens,
        match_len=candidate.match_len,
        source_start=candidate.source_start,
        follow_start=candidate.follow_start,
        view=view,
        frontier_distance=candidate.frontier_distance,
        crosses_frontier=candidate.crosses_frontier,
        score=0.0,
        pair_keys=candidate.pair_keys,
    )
    margin = _mv_effective_margin(tokenizer, mv_candidate, config)
    if len(candidate.tokens) < exact_len + margin:
        return False
    if not stats.adopted and candidate.frontier_distance is None:
        return False
    return True


def _candidate_view_map(views) -> dict[str, Any]:
    return {str(v.view_id): v for v in views}


def _analyze_trace(
    *,
    tokenizer,
    trace: TaskMethodTrace,
    config: VantageMVConfig,
    shadow_kind: str,
    skip_reasons: set[str],
    min_hit_tokens: int,
    max_draft_mode: str,
) -> tuple[list[ShadowHit], dict[str, Any]]:
    shadow_config = VantageMVConfig(**{**config.__dict__})
    shadow_config = VantageMVConfig(
        **{
            **shadow_config.__dict__,
            "use_rewrite_fst": shadow_kind == "fst",
        }
    )
    plan, rewrite_map, map_source, map_parse_us = _build_mv_lazy_plan(
        tokenizer,
        prompt_text=trace.prompt,
        reference=trace.reference,
        metadata=trace.metadata,
        config=shadow_config,
    )
    if plan is None:
        return [], {
            "task_id": trace.task_id,
            "has_plan": False,
            "rewrite_map_size": len(rewrite_map),
            "map_source": map_source,
            "map_parse_us": map_parse_us,
        }

    t_build = time.perf_counter_ns()
    views, apply_us, tokenize_us, index_us = _build_mv_views_from_plan(
        tokenizer,
        plan=plan,
        config=shadow_config,
    )
    build_us = (time.perf_counter_ns() - t_build) / 1000.0
    by_view = _candidate_view_map(views)
    hits: list[ShadowHit] = []
    max_draft_tokens = (
        shadow_config.max_draft_tokens
        if max_draft_mode == "max"
        else min(shadow_config.max_draft_tokens, shadow_config.cold_trans_max_draft)
    )
    for idx, step in enumerate(trace.steps):
        reason = str(step.get("proposal_route_reason") or "")
        if reason not in skip_reasons:
            continue
        start = trace.starts[idx] if idx < len(trace.starts) else 0
        prefix = trace.prompt_tokens + trace.output_tokens[:start]
        next_tokens = trace.output_tokens[start:]
        if not next_tokens:
            continue
        exact_drafts, exact_match_len, _, _ = prompt_lookup_draft(
            prefix,
            max_matching_ngram_size=shadow_config.max_matching_ngram_size,
            max_draft_tokens=shadow_config.max_draft_tokens,
        )
        candidates = _enumerate_shadow_candidates(
            prefix=prefix,
            views=views,
            config=shadow_config,
            max_draft_tokens=max_draft_tokens,
        )
        if not candidates:
            continue
        scored: list[tuple[int, int, int, bool, ShadowCandidate]] = []
        for cand in candidates:
            accepted = _common_prefix_len(cand.tokens, next_tokens)
            runtime = _runtime_eligible(
                tokenizer=tokenizer,
                candidate=cand,
                view=by_view[cand.view_id],
                exact_len=len(exact_drafts),
                config=shadow_config,
            )
            scored.append((accepted, cand.match_len, len(cand.tokens), runtime, cand))
        accepted, _match_len, _cand_len, runtime, best = max(
            scored,
            key=lambda x: (x[0], x[3], x[1], x[2]),
        )
        current_accepted = int(step.get("n_accepted_nonroot_drafts") or 0)
        hits.append(
            ShadowHit(
                task_id=trace.task_id,
                step=int(step.get("step") or idx),
                reason=reason,
                shadow_kind=shadow_kind,
                accepted=accepted,
                incremental_accepted=max(0, accepted - current_accepted),
                candidate_len=len(best.tokens),
                match_len=best.match_len,
                current_accepted=current_accepted,
                current_emitted=int(step.get("n_emitted") or 0),
                current_wall_us=float(step.get("wall_us") or 0.0),
                exact_len=len(exact_drafts),
                exact_match_len=exact_match_len,
                runtime_eligible=runtime,
                view_id=best.view_id,
                source_label=best.source_label,
                frontier_distance=best.frontier_distance,
                crosses_frontier=best.crosses_frontier,
            )
        )
    return hits, {
        "task_id": trace.task_id,
        "has_plan": True,
        "rewrite_map_size": len(rewrite_map),
        "map_source": map_source,
        "n_views": len(views),
        "map_parse_us": map_parse_us,
        "build_us": build_us,
        "apply_us": apply_us,
        "tokenize_us": tokenize_us,
        "index_us": index_us,
    }


def _simulate_oracle_wall(
    *,
    traces: list[TaskMethodTrace],
    hits: list[ShadowHit],
    min_hit_tokens: int,
    build_cost_by_task: dict[tuple[str, str], float],
) -> dict[str, Any]:
    hits_by_task_kind: dict[tuple[str, str], list[ShadowHit]] = defaultdict(list)
    for hit in hits:
        if hit.accepted >= min_hit_tokens:
            hits_by_task_kind[(hit.task_id, hit.shadow_kind)].append(hit)

    current_wall_us = 0.0
    current_tokens = 0
    trace_by_task = {t.task_id: t for t in traces}
    for trace in traces:
        current_tokens += len(trace.output_tokens)
        current_wall_us += sum(float(s.get("wall_us") or 0.0) for s in trace.steps)

    by_kind: dict[str, dict[str, Any]] = {}
    kinds = sorted({h.shadow_kind for h in hits})
    for kind in kinds:
        oracle_wall = 0.0
        used_hits = 0
        skipped_steps = 0
        added_build_us = 0.0
        for trace in traces:
            task_hits = {
                h.step: h
                for h in hits_by_task_kind.get((trace.task_id, kind), [])
            }
            covered_until = -1
            built = False
            for idx, step in enumerate(trace.steps):
                start = trace.starts[idx] if idx < len(trace.starts) else 0
                if start < covered_until:
                    skipped_steps += 1
                    continue
                step_wall = float(step.get("wall_us") or 0.0)
                hit = task_hits.get(int(step.get("step") or idx))
                if hit and hit.accepted + 1 > int(step.get("n_emitted") or 0):
                    if not built:
                        cost = build_cost_by_task.get((trace.task_id, kind), 0.0)
                        oracle_wall += cost
                        added_build_us += cost
                        built = True
                    oracle_wall += step_wall
                    covered_until = start + hit.accepted + 1
                    used_hits += 1
                else:
                    oracle_wall += step_wall
        by_kind[kind] = {
            "current_wall_us": current_wall_us,
            "oracle_wall_us": oracle_wall,
            "tokens": current_tokens,
            "current_tps": current_tokens / (current_wall_us / 1e6) if current_wall_us else 0.0,
            "oracle_tps": current_tokens / (oracle_wall / 1e6) if oracle_wall else 0.0,
            "oracle_speedup_vs_current": (current_wall_us / oracle_wall) if oracle_wall else 0.0,
            "used_hits": used_hits,
            "skipped_steps": skipped_steps,
            "added_build_us": added_build_us,
        }
    return by_kind


def _summarize_hits(hits: list[ShadowHit], *, min_hit_tokens: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    by_kind_reason: dict[tuple[str, str], list[ShadowHit]] = defaultdict(list)
    for hit in hits:
        by_kind_reason[(hit.shadow_kind, hit.reason)].append(hit)

    summary_rows = []
    for (kind, reason), rows in sorted(by_kind_reason.items()):
        ge = [h for h in rows if h.accepted >= min_hit_tokens]
        runtime_ge = [h for h in ge if h.runtime_eligible]
        accepted_values = [h.accepted for h in rows]
        summary_rows.append(
            {
                "shadow_kind": kind,
                "reason": reason,
                "skipped_steps_with_candidate": len(rows),
                f"steps_ge_{min_hit_tokens}": len(ge),
                f"runtime_eligible_ge_{min_hit_tokens}": len(runtime_ge),
                f"tasks_ge_{min_hit_tokens}": len({h.task_id for h in ge}),
                "accepted_tokens_sum": sum(h.accepted for h in rows),
                "incremental_accepted_sum": sum(h.incremental_accepted for h in rows),
                "accepted_p50": statistics.median(accepted_values) if accepted_values else 0.0,
                "accepted_p90": _percentile(accepted_values, 0.90),
                "max_accepted": max(accepted_values) if accepted_values else 0,
                "runtime_eligible_steps": sum(1 for h in rows if h.runtime_eligible),
            }
        )
    out["by_kind_reason"] = summary_rows
    by_kind = defaultdict(list)
    for hit in hits:
        by_kind[hit.shadow_kind].append(hit)
    out["by_kind"] = []
    for kind, rows in sorted(by_kind.items()):
        ge = [h for h in rows if h.accepted >= min_hit_tokens]
        out["by_kind"].append(
            {
                "shadow_kind": kind,
                "steps_with_candidate": len(rows),
                f"steps_ge_{min_hit_tokens}": len(ge),
                f"tasks_ge_{min_hit_tokens}": len({h.task_id for h in ge}),
                "accepted_tokens_sum": sum(h.accepted for h in rows),
                "incremental_accepted_sum": sum(h.incremental_accepted for h in rows),
                "runtime_eligible_steps": sum(1 for h in rows if h.runtime_eligible),
            }
        )
    out["top_hits"] = [
        hit.__dict__
        for hit in sorted(hits, key=lambda h: (h.accepted, h.incremental_accepted), reverse=True)[:50]
    ]
    return out


def _percentile(xs: list[int | float], q: float) -> float:
    if not xs:
        return 0.0
    vals = sorted(float(x) for x in xs)
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    min_hit = report["min_hit_tokens"]
    lines = [
        "# Shadow-Candidate Oracle",
        "",
        f"Method: `{report['method']}`",
        f"Skipped reasons: {', '.join(report['skip_reasons'])}",
        f"Hit threshold: >= {min_hit} accepted tokens",
        "",
        "## Verdict",
        "",
    ]
    for row in report["summary"]["by_kind"]:
        steps = row[f"steps_ge_{min_hit}"]
        verdict = "VALIDATES" if steps >= 50 else "does not validate"
        lines.append(
            f"- `{row['shadow_kind']}`: {steps} skipped steps across "
            f"{row[f'tasks_ge_{min_hit}']} tasks would accept >= {min_hit} tokens; "
            f"{verdict} the missed-candidate hypothesis."
        )
    lines += [
        "",
        "## By Skip Reason",
        "",
        f"| shadow | reason | candidates | >= {min_hit} steps | >= {min_hit} tasks | runtime-eligible >= {min_hit} | accepted sum | incremental sum | p50 | p90 | max | runtime-eligible steps |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary"]["by_kind_reason"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['shadow_kind']}`",
                    f"`{row['reason']}`",
                    str(row["skipped_steps_with_candidate"]),
                    str(row[f"steps_ge_{min_hit}"]),
                    str(row[f"tasks_ge_{min_hit}"]),
                    str(row[f"runtime_eligible_ge_{min_hit}"]),
                    str(row["accepted_tokens_sum"]),
                    str(row["incremental_accepted_sum"]),
                    f"{row['accepted_p50']:.1f}",
                    f"{row['accepted_p90']:.1f}",
                    str(row["max_accepted"]),
                    str(row["runtime_eligible_steps"]),
                ]
            )
            + " |"
        )
    lines += ["", "## Latency-Adjusted Oracle Simulation", ""]
    lines.append("| shadow | current tok/s | oracle tok/s | oracle/current | used hits | skipped recorded steps | added build s |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for kind, row in sorted(report["oracle_simulation"].items()):
        lines.append(
            f"| `{kind}` | {row['current_tps']:.2f} | {row['oracle_tps']:.2f} | "
            f"{row['oracle_speedup_vs_current']:.4f} | {row['used_hits']} | "
            f"{row['skipped_steps']} | {row['added_build_us'] / 1e6:.3f} |"
        )
    lines += [
        "",
        "## Reconstruction",
        "",
        "```json",
        json.dumps(report["reconstruction"], indent=2, sort_keys=True),
        "```",
        "",
        "## Top Shadow Hits",
        "",
        "| shadow | task | step | reason | accepted | current accepted | candidate len | match len | runtime eligible | view | frontier |",
        "|---|---|---:|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for hit in report["summary"]["top_hits"][:20]:
        lines.append(
            f"| `{hit['shadow_kind']}` | `{hit['task_id']}` | {hit['step']} | "
            f"`{hit['reason']}` | {hit['accepted']} | {hit['current_accepted']} | "
            f"{hit['candidate_len']} | {hit['match_len']} | {hit['runtime_eligible']} | "
            f"`{hit['view_id']}` | {hit['frontier_distance']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completions", type=Path, required=True)
    parser.add_argument("--steps", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--target-trust-remote-code", action="store_true")
    parser.add_argument("--chat-template", default="none")
    parser.add_argument("--skip-reasons", default=",".join(sorted(DEFAULT_SKIP_REASONS)))
    parser.add_argument("--shadow-kinds", default="mv,fst", help="comma-separated: mv,fst")
    parser.add_argument("--min-hit-tokens", type=int, default=16)
    parser.add_argument("--max-draft-mode", choices=["cold", "max"], default="cold")
    parser.add_argument("--task-limit", type=int)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.target,
        trust_remote_code=args.target_trust_remote_code,
    )
    method_config = parse_vantage_mv_method(args.method)
    skip_reasons = {x.strip() for x in args.skip_reasons.split(",") if x.strip()}
    shadow_kinds = [x.strip() for x in args.shadow_kinds.split(",") if x.strip()]
    unknown = set(shadow_kinds) - {"mv", "fst"}
    if unknown:
        raise ValueError(f"unsupported shadow kind(s): {sorted(unknown)}")

    traces, reconstruction = _load_traces(
        tokenizer=tokenizer,
        completions_path=args.completions,
        steps_path=args.steps,
        method=args.method,
        chat_template=args.chat_template,
        task_limit=args.task_limit,
    )

    all_hits: list[ShadowHit] = []
    task_build_rows: list[dict[str, Any]] = []
    build_cost_by_task: dict[tuple[str, str], float] = {}
    for trace in traces:
        for kind in shadow_kinds:
            hits, build = _analyze_trace(
                tokenizer=tokenizer,
                trace=trace,
                config=method_config,
                shadow_kind=kind,
                skip_reasons=skip_reasons,
                min_hit_tokens=args.min_hit_tokens,
                max_draft_mode=args.max_draft_mode,
            )
            all_hits.extend(hits)
            build["shadow_kind"] = kind
            task_build_rows.append(build)
            build_cost_by_task[(trace.task_id, kind)] = float(build.get("build_us") or 0.0)

    report = {
        "method": args.method,
        "target": args.target,
        "chat_template": args.chat_template,
        "skip_reasons": sorted(skip_reasons),
        "shadow_kinds": shadow_kinds,
        "min_hit_tokens": args.min_hit_tokens,
        "max_draft_mode": args.max_draft_mode,
        "n_traces": len(traces),
        "reconstruction": reconstruction,
        "build": {
            "tasks_with_plan_by_kind": dict(
                Counter(
                    row["shadow_kind"]
                    for row in task_build_rows
                    if row.get("has_plan")
                )
            ),
            "mean_build_us_by_kind": {
                kind: statistics.mean(
                    [
                        float(row.get("build_us") or 0.0)
                        for row in task_build_rows
                        if row.get("shadow_kind") == kind and row.get("has_plan")
                    ]
                    or [0.0]
                )
                for kind in shadow_kinds
            },
        },
        "summary": _summarize_hits(all_hits, min_hit_tokens=args.min_hit_tokens),
        "oracle_simulation": _simulate_oracle_wall(
            traces=traces,
            hits=all_hits,
            min_hit_tokens=args.min_hit_tokens,
            build_cost_by_task=build_cost_by_task,
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, args.output_md)


if __name__ == "__main__":
    main()
