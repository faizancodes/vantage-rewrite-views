"""Best-of-K and longer-context oracle for ambiguous exact PLD hits.

This is an offline diagnostic. It reconstructs each PLD decode step from
``steps.jsonl`` and the verified PLD completion, enumerates alternative exact
prompt-lookup source positions, and scores each alternative by longest common
prefix against the already-produced greedy output suffix. It does not run the
target model and does not change decoding semantics.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asts.code_proposers import encode_no_special  # noqa: E402


@dataclass
class TaskTrace:
    task_id: str
    prompt: str
    reference: str
    prompt_tokens: list[int]
    output_tokens: list[int]
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


def _output_tokens(tokenizer, completion: dict[str, Any], method: str) -> list[int]:
    outputs = completion.get("outputs") or {}
    row = outputs.get(method)
    if row is None:
        raise KeyError(f"method {method!r} missing for task {completion.get('task_id')}")
    for key in ("tokens", "token_ids", "completion_token_ids", "output_token_ids"):
        val = row.get(key)
        if isinstance(val, list) and all(isinstance(x, int) for x in val):
            return list(val)
    text = str(row.get("raw_text") if row.get("raw_text") is not None else row.get("text") or "")
    return encode_no_special(tokenizer, text)


def _build_traces(
    *,
    tokenizer,
    steps_path: Path,
    completions_path: Path,
    method: str,
    chat_template: str,
) -> dict[str, TaskTrace]:
    completions = {str(r["task_id"]): r for r in _load_jsonl(completions_path)}
    steps_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step in _load_jsonl(steps_path):
        if step.get("method") == method:
            steps_by_task[str(step.get("task_id"))].append(step)

    traces: dict[str, TaskTrace] = {}
    for task_id, steps in steps_by_task.items():
        comp = completions.get(task_id)
        if comp is None:
            continue
        ordered_steps = sorted(steps, key=lambda r: int(r.get("step") or 0))
        starts: list[int] = []
        pos = 0
        for step in ordered_steps:
            starts.append(pos)
            pos += int(step.get("n_emitted") or 0)
        traces[task_id] = TaskTrace(
            task_id=task_id,
            prompt=str(comp.get("prompt") or ""),
            reference=str(comp.get("reference") or ""),
            prompt_tokens=_encode_prompt_ids(tokenizer, str(comp.get("prompt") or ""), chat_template),
            output_tokens=_output_tokens(tokenizer, comp, method),
            steps=ordered_steps,
            starts=starts,
        )
    return traces


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _matching_source_positions(tokens: list[int], *, n: int) -> list[int]:
    if n <= 0 or len(tokens) < n * 2:
        return []
    current_start = len(tokens) - n
    suffix = tokens[current_start:]
    positions: list[int] = []
    for start in range(current_start - n, -1, -1):
        if tokens[start : start + n] == suffix:
            positions.append(start)
    return positions


def _draft_from_source(tokens: list[int], *, source_start: int, n: int, max_draft: int) -> list[int]:
    current_start = len(tokens) - n
    follow_start = source_start + n
    follow_end = min(follow_start + max_draft, current_start)
    if follow_start >= follow_end:
        return []
    return list(tokens[follow_start:follow_end])


def _left_extension_len(
    tokens: list[int],
    *,
    source_start: int,
    current_start: int,
    max_left: int,
) -> int:
    count = 0
    while count < max_left:
        src_idx = source_start - 1 - count
        cur_idx = current_start - 1 - count
        if src_idx < 0 or cur_idx < 0:
            break
        if tokens[src_idx] != tokens[cur_idx]:
            break
        count += 1
    return count


def _source_type(source_start: int, prompt_len: int) -> str:
    return "prompt/reference" if source_start < prompt_len else "generated"


def _projected_speedup(
    *,
    total_wall_us: float,
    ambiguous_wall_us: float,
    baseline_accepted: float,
    oracle_accepted: float,
) -> float:
    if total_wall_us <= 0 or ambiguous_wall_us <= 0:
        return 1.0
    if baseline_accepted <= 0 or oracle_accepted <= baseline_accepted:
        return 1.0
    # Approximate only the ambiguous-step component as inversely proportional
    # to accepted tokens per verify; all other runtime remains unchanged.
    improved_ambiguous_wall = ambiguous_wall_us * (baseline_accepted / oracle_accepted)
    projected_wall = total_wall_us - ambiguous_wall_us + improved_ambiguous_wall
    return total_wall_us / projected_wall if projected_wall > 0 else 1.0


def _mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else 0.0


def _pct(n: float, d: float) -> float:
    return 100.0 * n / d if d else 0.0


def analyze(
    *,
    tokenizer,
    steps_path: Path,
    completions_path: Path,
    method: str,
    chat_template: str,
    max_draft_tokens: int,
    k_values: list[int],
    n_sweep: list[int],
    detail_limit: int,
    details_jsonl_out: Path | None = None,
) -> dict[str, Any]:
    traces = _build_traces(
        tokenizer=tokenizer,
        steps_path=steps_path,
        completions_path=completions_path,
        method=method,
        chat_template=chat_template,
    )
    total_wall_us = 0.0
    ambiguous_wall_us = 0.0
    ambiguous_rows: list[dict[str, Any]] = []
    best_by_k = {k: [] for k in k_values}
    details: list[dict[str, Any]] = []
    longctx_by_resolving_n: dict[int, list[int]] = {n: [] for n in n_sweep}
    longctx_attempts = 0
    full_details: list[dict[str, Any]] = []
    max_k = max(k_values) if k_values else 0

    for trace in traces.values():
        previous_good_source: int | None = None
        for idx, step in enumerate(trace.steps):
            start = trace.starts[idx]
            prefix = trace.prompt_tokens + trace.output_tokens[:start]
            future = trace.output_tokens[start:]
            wall_us = float(step.get("wall_us") or 0.0)
            total_wall_us += wall_us
            baseline_accepted = int(step.get("n_accepted_drafts") or 0)
            match_len = int(step.get("proposal_match_len") or step.get("proposal_query_len") or 0)
            previous_good_source_for_detail = previous_good_source
            step_source = step.get("proposal_source_start_token")
            if baseline_accepted >= 16 and isinstance(step_source, int) and step_source >= 0:
                previous_good_source = int(step_source)
            if not future or match_len <= 0:
                continue
            positions = _matching_source_positions(prefix, n=match_len)
            candidate_count = int(step.get("pld_opp_candidate_matches") or len(positions))
            if candidate_count <= 1:
                continue
            if len(positions) <= 1:
                continue
            ambiguous_wall_us += wall_us
            baseline_pos = int(step.get("proposal_source_start_token") or positions[0])
            row = {
                "task_id": trace.task_id,
                "step_id": int(step.get("step") or idx),
                "match_len": match_len,
                "generated_offset": start,
                "candidate_count_n10": len(positions),
                "baseline_candidate_position": baseline_pos,
                "baseline_accepted_len": baseline_accepted,
                "wall_us": wall_us,
            }
            ambiguous_rows.append(row)
            full_candidate_details: list[dict[str, Any]] = []
            if details_jsonl_out is not None:
                current_start = len(prefix) - match_len
                for rank, pos in enumerate(positions[:max_k], start=1):
                    draft = _draft_from_source(
                        prefix,
                        source_start=pos,
                        n=match_len,
                        max_draft=max_draft_tokens,
                    )
                    full_candidate_details.append(
                        {
                            "rank": rank,
                            "source_position": pos,
                            "source_type": _source_type(pos, len(trace.prompt_tokens)),
                            "source_distance_from_previous_good_source": (
                                abs(pos - previous_good_source_for_detail)
                                if previous_good_source_for_detail is not None
                                else None
                            ),
                            "left_extension_len": _left_extension_len(
                                prefix,
                                source_start=pos,
                                current_start=current_start,
                                max_left=128,
                            ),
                            "lcp_with_actual_future_output": _common_prefix_len(draft, future),
                            "candidate_draft_prefix_128": tokenizer.decode(
                                draft[:128], skip_special_tokens=False
                            ),
                        }
                    )
            for k in k_values:
                top = positions[:k]
                lcp_values: list[int] = []
                cand_details: list[dict[str, Any]] = []
                for pos in top:
                    draft = _draft_from_source(
                        prefix,
                        source_start=pos,
                        n=match_len,
                        max_draft=max_draft_tokens,
                    )
                    lcp = _common_prefix_len(draft, future)
                    lcp_values.append(lcp)
                    if len(details) < detail_limit:
                        cand_details.append(
                            {
                                "source_position": pos,
                                "source_type": _source_type(pos, len(trace.prompt_tokens)),
                                "lcp_with_actual_future_output": lcp,
                                "candidate_draft_prefix_128": tokenizer.decode(
                                    draft[:128], skip_special_tokens=False
                                ),
                            }
                        )
                best = max(lcp_values) if lcp_values else 0
                best_by_k[k].append(best)
                row[f"oracle_best_{k}_accepted_len"] = best
                row[f"oracle_best_{k}_gain_over_baseline"] = best - baseline_accepted
                if len(details) < detail_limit:
                    row[f"candidates_k{k}"] = cand_details
            if details_jsonl_out is not None:
                full_details.append(
                    {
                        "task_id": trace.task_id,
                        "step_id": int(step.get("step") or idx),
                        "generated_suffix_64": tokenizer.decode(
                            prefix[-64:], skip_special_tokens=False
                        ),
                        "baseline_candidate_position": baseline_pos,
                        "baseline_accepted_len": baseline_accepted,
                        "candidate_count_n10": len(positions),
                        "match_len": match_len,
                        "oracle_best_by_k": {
                            str(k): {
                                "accepted_len": row.get(f"oracle_best_{k}_accepted_len", 0),
                                "gain_over_baseline": row.get(
                                    f"oracle_best_{k}_gain_over_baseline", 0
                                ),
                            }
                            for k in k_values
                        },
                        "candidates": full_candidate_details,
                    }
                )

            # Longer-context uniqueness sweep. We search from longest to
            # shortest and credit the first unique exact match; bucket results
            # by that resolving suffix length.
            longctx_attempts += 1
            for n in n_sweep:
                if len(prefix) < n * 2:
                    continue
                n_positions = _matching_source_positions(prefix, n=n)
                if len(n_positions) == 1:
                    draft = _draft_from_source(
                        prefix,
                        source_start=n_positions[0],
                        n=n,
                        max_draft=max_draft_tokens,
                    )
                    longctx_by_resolving_n[n].append(_common_prefix_len(draft, future))
                    break

            if len(details) < detail_limit:
                row["generated_suffix_64"] = tokenizer.decode(
                    prefix[-64:], skip_special_tokens=False
                )
                row["baseline_draft_prefix_128"] = tokenizer.decode(
                    _draft_from_source(
                        prefix,
                        source_start=baseline_pos,
                        n=match_len,
                        max_draft=max_draft_tokens,
                    )[:128],
                    skip_special_tokens=False,
                )
                details.append(row)

    if details_jsonl_out is not None:
        details_jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        with details_jsonl_out.open("w") as f:
            for row in full_details:
                f.write(json.dumps(row) + "\n")

    baseline_values = [float(r["baseline_accepted_len"]) for r in ambiguous_rows]
    baseline_mean = _mean(baseline_values)
    best_of_k_rows = []
    for k in k_values:
        vals = [float(v) for v in best_by_k[k]]
        oracle_mean = _mean(vals)
        best_of_k_rows.append(
            {
                "K": k,
                "ambiguous_coverage_pct": _pct(len(vals), len(ambiguous_rows)),
                "baseline_ambig_accepted": baseline_mean,
                "oracle_ambig_accepted": oracle_mean,
                "oracle_gain_over_baseline": oracle_mean - baseline_mean,
                "projected_speedup": _projected_speedup(
                    total_wall_us=total_wall_us,
                    ambiguous_wall_us=ambiguous_wall_us,
                    baseline_accepted=baseline_mean,
                    oracle_accepted=oracle_mean,
                ),
            }
        )

    longctx_rows = []
    resolved_so_far = 0
    accepted_so_far: list[int] = []
    for n in n_sweep:
        vals = longctx_by_resolving_n[n]
        resolved_so_far += len(vals)
        accepted_so_far.extend(vals)
        accepted_mean = _mean([float(v) for v in vals])
        cumulative_mean = _mean([float(v) for v in accepted_so_far])
        longctx_rows.append(
            {
                "longest_exact_suffix_n": n,
                "resolved_count": len(vals),
                "resolved_unique_pct": _pct(len(vals), longctx_attempts),
                "cumulative_resolved_unique_pct": _pct(resolved_so_far, longctx_attempts),
                "accepted_len_if_chosen": accepted_mean,
                "cumulative_accepted_len_if_chosen": cumulative_mean,
                "projected_speedup_cumulative": _projected_speedup(
                    total_wall_us=total_wall_us,
                    ambiguous_wall_us=ambiguous_wall_us * (resolved_so_far / longctx_attempts)
                    if longctx_attempts
                    else 0.0,
                    baseline_accepted=baseline_mean,
                    oracle_accepted=cumulative_mean,
                ),
            }
        )

    return {
        "method": method,
        "n_tasks": len(traces),
        "total_steps": sum(len(t.steps) for t in traces.values()),
        "ambiguous_steps": len(ambiguous_rows),
        "ambiguous_runtime_fraction": ambiguous_wall_us / total_wall_us if total_wall_us else 0.0,
        "baseline_ambiguous_accepted_mean": baseline_mean,
        "best_of_k": best_of_k_rows,
        "longer_context": longctx_rows,
        "details": details,
    }


def _print_report(report: dict[str, Any]) -> None:
    print(f"# PLD Candidate Oracle: `{report['method']}`")
    print()
    print(f"- tasks: {report['n_tasks']}")
    print(f"- total steps: {report['total_steps']}")
    print(f"- ambiguous exact-hit steps: {report['ambiguous_steps']}")
    print(f"- ambiguous runtime fraction: {100 * report['ambiguous_runtime_fraction']:.2f}%")
    print(f"- baseline ambiguous accepted mean: {report['baseline_ambiguous_accepted_mean']:.2f}")
    print()
    print("## Best-of-K Oracle")
    print()
    print("| K | ambiguous coverage | baseline ambig accepted | oracle ambig accepted | projected speedup |")
    print("|---:|---:|---:|---:|---:|")
    for row in report["best_of_k"]:
        print(
            f"| {row['K']} | {row['ambiguous_coverage_pct']:.1f}% | "
            f"{row['baseline_ambig_accepted']:.2f} | "
            f"{row['oracle_ambig_accepted']:.2f} | "
            f"{row['projected_speedup']:.3f}x |"
        )
    print()
    print("## Longer-Context Exact Disambiguation")
    print()
    print("| longest exact suffix n | % ambiguous hits resolved uniquely | cumulative resolved | accepted len if chosen | projected speedup |")
    print("|---:|---:|---:|---:|---:|")
    for row in report["longer_context"]:
        print(
            f"| {row['longest_exact_suffix_n']} | "
            f"{row['resolved_unique_pct']:.1f}% | "
            f"{row['cumulative_resolved_unique_pct']:.1f}% | "
            f"{row['accepted_len_if_chosen']:.2f} | "
            f"{row['projected_speedup_cumulative']:.3f}x |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=Path, required=True)
    parser.add_argument("--completions", type=Path, required=True)
    parser.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--method", default="blazedit_pld_w128_n10")
    parser.add_argument("--chat-template", default="none")
    parser.add_argument("--max-draft-tokens", type=int, default=128)
    parser.add_argument("--k-values", default="4,8,16,32")
    parser.add_argument("--n-sweep", default="64,48,32,24,16,12,10")
    parser.add_argument("--detail-limit", type=int, default=20)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--details-jsonl-out", type=Path, default=None)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    report = analyze(
        tokenizer=tokenizer,
        steps_path=args.steps,
        completions_path=args.completions,
        method=args.method,
        chat_template=args.chat_template,
        max_draft_tokens=args.max_draft_tokens,
        k_values=[int(x) for x in args.k_values.split(",") if x.strip()],
        n_sweep=[int(x) for x in args.n_sweep.split(",") if x.strip()],
        detail_limit=args.detail_limit,
        details_jsonl_out=args.details_jsonl_out,
    )
    _print_report(report)
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
