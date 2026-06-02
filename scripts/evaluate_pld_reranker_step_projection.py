"""Replay-style projection for exact-PLD candidate reranking policies.

The older reranker reports summarized accepted length on ambiguous PLD hits.
That is a useful label, but it does not estimate decoding speed: a longer
candidate only helps when it skips later decode steps, while a shorter candidate
can create catch-up work that the mean-length metric ignores.

This script consumes the ambiguous candidate oracle JSONL and optionally the
baseline PLD steps trace.  With a full trace it replays baseline decode steps by
generated-token offset and substitutes each policy's oracle accepted length at
ambiguous hits.  Without a full trace it falls back to an ambiguous-only replay
and marks the result as conservative/incomplete.
"""

from __future__ import annotations

import argparse
import bisect
import json
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_pld_candidate_reranker import (  # noqa: E402
    Candidate,
    Example,
    _candidate_by_baseline,
    _choose_linear,
    _choose_linear_margin,
    _choose_rank,
    _load_jsonl,
    _parse_examples,
)


DEFAULT_INPUT = Path("/tmp/pld_candidate_oracle_v2/ambiguous_candidates.jsonl")
DEFAULT_OUT_DIR = Path("/tmp/pld_candidate_oracle_v2/step_projection")


@dataclass(frozen=True)
class BaselineStep:
    task_id: str
    step_id: int
    start: int
    emitted: int
    accepted_len: int


@dataclass(frozen=True)
class ProjectionResult:
    policy: str
    baseline_steps: int
    projected_steps: int
    projected_step_reduction_pct: float
    corrected_projected_speedup: float
    ambiguous_steps_seen: int
    ambiguous_steps_used: int
    ambiguous_accepted_len: float
    token01_rejection_proxy_pct: float
    oracle_gain_captured_pct: float
    selected_rank_distribution: dict[str, int]
    selected_baseline_pct: float
    skipped_baseline_steps: int
    catchup_steps: int
    trace_mode: str


def _read_report(input_path: Path) -> dict[str, Any]:
    report_path = input_path.with_name("report.json")
    if not report_path.exists():
        return {}
    try:
        return json.loads(report_path.read_text())
    except Exception:
        return {}


def _load_weights(path: Path) -> list[float]:
    data = json.loads(path.read_text())
    weights = data.get("weights")
    if not isinstance(weights, list):
        raise ValueError(f"{path} does not contain a weights list")
    return [float(x) for x in weights]


def _discover_weight_files(input_path: Path) -> list[Path]:
    parent = input_path.parent
    return sorted(parent.glob("reranker_eval*/reranker_weights.json"))


def _safe_mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else 0.0


def _oracle_choose(ex: Example) -> Candidate:
    return max(ex.candidates, key=lambda c: (c.accepted_len, -c.rank0))


def _limit_example_like_runtime(ex: Example, *, top_k: int) -> Example:
    """Mirror runtime --pld-rerank-always-include-baseline for one example."""

    raw = list(ex.candidates)
    limited = raw[:top_k]
    baseline_pos = ex.baseline_position
    baseline = None
    if baseline_pos is not None:
        for cand in raw:
            if cand.source_position == baseline_pos:
                baseline = cand
                break
    if baseline_pos is not None and all(c.source_position != baseline_pos for c in limited):
        baseline = baseline or _synthetic_baseline_candidate(
            ex, rank0=max(0, min(top_k - 1, len(limited)))
        )
        if len(limited) < top_k:
            limited.append(baseline)
        elif top_k > 0:
            limited[-1] = baseline
    reranked = tuple(_rerank_candidate(c, rank0=i) for i, c in enumerate(limited[:top_k]))
    return Example(
        task_id=ex.task_id,
        step_id=ex.step_id,
        baseline_position=ex.baseline_position,
        baseline_accepted_len=ex.baseline_accepted_len,
        candidate_count=ex.candidate_count,
        match_len=ex.match_len,
        candidates=reranked,
    )


def _oracle_choose_k(top_k: int) -> Callable[[Example], Candidate]:
    def choose(ex: Example) -> Candidate:
        return _oracle_choose(_limit_example_like_runtime(ex, top_k=top_k))

    return choose


def _rerank_candidate(
    cand: Candidate,
    *,
    rank0: int,
) -> Candidate:
    return Candidate(
        idx=rank0,
        rank0=rank0,
        source_position=cand.source_position,
        source_type=cand.source_type,
        source_distance_from_previous_good_source=cand.source_distance_from_previous_good_source,
        accepted_len=cand.accepted_len,
        draft_prefix=cand.draft_prefix,
        left_extension=cand.left_extension,
        next2=cand.next2,
        next4=cand.next4,
    )


def _synthetic_baseline_candidate(ex: Example, *, rank0: int) -> Candidate:
    return Candidate(
        idx=rank0,
        rank0=rank0,
        source_position=ex.baseline_position if ex.baseline_position is not None else -1,
        source_type="baseline_pld",
        source_distance_from_previous_good_source=None,
        accepted_len=ex.baseline_accepted_len,
        draft_prefix="",
        left_extension=0,
        next2="",
        next4="",
    )


def _limit_examples_like_runtime(examples: list[Example], *, top_k: int) -> list[Example]:
    """Mirror runtime --pld-rerank-always-include-baseline behavior.

    The oracle JSONL was historically written with the first K raw candidates,
    while the runtime reranker replaces the last slot with the baseline PLD
    source when that source is outside the top-K list.  This normalization makes
    fixed-rank projections comparable to runtime fixed-rank rows.
    """

    return [_limit_example_like_runtime(ex, top_k=top_k) for ex in examples]


def _policy_table(
    examples: list[Example],
    *,
    weights_files: list[Path],
    margin_values: list[float],
    oracle_ks: list[int],
) -> list[tuple[str, Callable[[Example], Candidate]]]:
    policies: list[tuple[str, Callable[[Example], Candidate]]] = [
        ("baseline_pld", _candidate_by_baseline),
        ("fixed_rank0", _choose_rank(0)),
        ("fixed_rank1", _choose_rank(1)),
        ("fixed_rank2", _choose_rank(2)),
        ("fixed_rank3", _choose_rank(3)),
    ]
    for oracle_k in sorted(set(oracle_ks)):
        policies.append((f"best_of_k_oracle_k{oracle_k}", _oracle_choose_k(oracle_k)))
    for weights_path in weights_files:
        try:
            weights = _load_weights(weights_path)
        except Exception as exc:
            print(f"warning: skipping weights {weights_path}: {exc}", file=sys.stderr)
            continue
        stem = weights_path.parent.name.replace("reranker_eval_", "").replace(
            "reranker_eval", "learned_v1"
        )
        policies.append((f"learned_{stem}", _choose_linear(weights)))
        for margin in margin_values:
            policies.append(
                (
                    f"learned_{stem}_margin_{margin:g}",
                    _choose_linear_margin(weights, margin),
                )
            )
    return policies


def _load_baseline_steps(path: Path, *, method: str) -> dict[str, list[BaselineStep]]:
    rows_by_task: dict[str, list[dict[str, Any]]] = {}
    for row in _load_jsonl(path):
        if row.get("method") != method:
            continue
        rows_by_task.setdefault(str(row.get("task_id") or ""), []).append(row)

    out: dict[str, list[BaselineStep]] = {}
    for task_id, rows in rows_by_task.items():
        pos = 0
        steps: list[BaselineStep] = []
        for row in sorted(rows, key=lambda r: int(r.get("step") or 0)):
            emitted = int(row.get("n_emitted") or 0)
            accepted_len = int(row.get("n_accepted_drafts") or 0)
            steps.append(
                BaselineStep(
                    task_id=task_id,
                    step_id=int(row.get("step") or 0),
                    start=pos,
                    emitted=max(1, emitted),
                    accepted_len=accepted_len,
                )
            )
            pos += max(1, emitted)
        out[task_id] = steps
    return out


def _example_map(examples: list[Example]) -> dict[tuple[str, int], Example]:
    return {(ex.task_id, ex.step_id): ex for ex in examples}


def _baseline_candidate_position(ex: Example) -> int | None:
    return _candidate_by_baseline(ex).source_position


def _summarize_selection(
    examples_used: list[Example],
    selected: list[Candidate],
) -> tuple[float, float, float, dict[str, int], float]:
    if not selected:
        return 0.0, 0.0, 0.0, {}, 0.0
    selected_mean = _safe_mean([float(c.accepted_len) for c in selected])
    token01 = 100.0 * sum(c.accepted_len <= 1 for c in selected) / len(selected)
    baseline_vals = [float(_candidate_by_baseline(ex).accepted_len) for ex in examples_used]
    oracle_vals = [float(max(c.accepted_len for c in ex.candidates)) for ex in examples_used]
    denom = _safe_mean(oracle_vals) - _safe_mean(baseline_vals)
    captured = (
        100.0 * (selected_mean - _safe_mean(baseline_vals)) / denom if denom > 0 else 0.0
    )
    ranks = Counter(str(c.rank0) for c in selected)
    baseline_same = sum(
        c.source_position == _baseline_candidate_position(ex)
        for ex, c in zip(examples_used, selected, strict=False)
    )
    selected_baseline_pct = 100.0 * baseline_same / len(selected)
    return selected_mean, token01, captured, dict(sorted(ranks.items())), selected_baseline_pct


def _project_with_full_steps(
    *,
    examples: list[Example],
    steps_by_task: dict[str, list[BaselineStep]],
    choose: Callable[[Example], Candidate],
) -> ProjectionResult:
    ex_by_key = _example_map(examples)
    baseline_steps = sum(len(v) for v in steps_by_task.values())
    projected_steps = 0
    skipped_steps = 0
    catchup_steps = 0
    used_examples: list[Example] = []
    selected_candidates: list[Candidate] = []

    for task_id, steps in steps_by_task.items():
        starts = [s.start for s in steps]
        i = 0
        while i < len(steps):
            step = steps[i]
            projected_steps += 1
            ex = ex_by_key.get((task_id, step.step_id))
            if ex is None:
                i += 1
                continue
            selected = choose(ex)
            used_examples.append(ex)
            selected_candidates.append(selected)
            baseline_emit = max(1, step.emitted)
            selected_emit = max(1, int(selected.accepted_len) + 1)
            if selected_emit <= baseline_emit:
                # The policy failed to cover tokens that baseline PLD emitted
                # in this step.  The trace has no intermediate rows inside the
                # baseline draft, so model the uncovered tail as greedy catch-up
                # work before returning to the next baseline step.
                shortfall = baseline_emit - selected_emit
                projected_steps += shortfall
                catchup_steps += shortfall
                i += 1
                continue
            covered_until = step.start + selected_emit
            j = bisect.bisect_left(starts, covered_until, lo=i + 1)
            skipped_steps += max(0, j - (i + 1))
            i = max(j, i + 1)

    selected_mean, token01, captured, ranks, baseline_pct = _summarize_selection(
        used_examples, selected_candidates
    )
    projected_steps = max(1, projected_steps)
    return ProjectionResult(
        policy="",
        baseline_steps=baseline_steps,
        projected_steps=projected_steps,
        projected_step_reduction_pct=100.0 * (baseline_steps - projected_steps) / baseline_steps
        if baseline_steps
        else 0.0,
        corrected_projected_speedup=baseline_steps / projected_steps if projected_steps else 1.0,
        ambiguous_steps_seen=len(examples),
        ambiguous_steps_used=len(used_examples),
        ambiguous_accepted_len=selected_mean,
        token01_rejection_proxy_pct=token01,
        oracle_gain_captured_pct=captured,
        selected_rank_distribution=ranks,
        selected_baseline_pct=baseline_pct,
        skipped_baseline_steps=skipped_steps,
        catchup_steps=catchup_steps,
        trace_mode="full_steps",
    )


def _project_ambiguous_only(
    *,
    examples: list[Example],
    choose: Callable[[Example], Candidate],
    baseline_steps: int,
) -> ProjectionResult:
    by_task: dict[str, list[Example]] = {}
    for ex in examples:
        by_task.setdefault(ex.task_id, []).append(ex)
    projected_ambiguous_steps = 0
    skipped_ambiguous_steps = 0
    catchup_steps = 0
    used_examples: list[Example] = []
    selected_candidates: list[Candidate] = []
    for task_examples in by_task.values():
        task_examples = sorted(task_examples, key=lambda ex: ex.step_id)
        i = 0
        while i < len(task_examples):
            ex = task_examples[i]
            projected_ambiguous_steps += 1
            selected = choose(ex)
            used_examples.append(ex)
            selected_candidates.append(selected)
            baseline_emit = max(1, ex.baseline_accepted_len + 1)
            selected_emit = max(1, selected.accepted_len + 1)
            if selected_emit <= baseline_emit:
                shortfall = baseline_emit - selected_emit
                projected_ambiguous_steps += shortfall
                catchup_steps += shortfall
                i += 1
                continue
            # Ambiguous-only rows do not contain all intervening decode steps.
            # Use baseline emitted-token budgets to skip later ambiguous rows
            # conservatively; non-ambiguous steps are left untouched.
            budget = selected_emit - baseline_emit
            j = i + 1
            while j < len(task_examples) and budget > 0:
                consumed = max(1, task_examples[j].baseline_accepted_len + 1)
                budget -= consumed
                j += 1
            skipped_ambiguous_steps += max(0, j - (i + 1))
            i = max(j, i + 1)

    non_ambiguous_steps = max(0, baseline_steps - len(examples))
    projected_steps = non_ambiguous_steps + projected_ambiguous_steps
    selected_mean, token01, captured, ranks, baseline_pct = _summarize_selection(
        used_examples, selected_candidates
    )
    projected_steps = max(1, projected_steps)
    return ProjectionResult(
        policy="",
        baseline_steps=baseline_steps,
        projected_steps=projected_steps,
        projected_step_reduction_pct=100.0 * (baseline_steps - projected_steps) / baseline_steps
        if baseline_steps
        else 0.0,
        corrected_projected_speedup=baseline_steps / projected_steps if projected_steps else 1.0,
        ambiguous_steps_seen=len(examples),
        ambiguous_steps_used=len(used_examples),
        ambiguous_accepted_len=selected_mean,
        token01_rejection_proxy_pct=token01,
        oracle_gain_captured_pct=captured,
        selected_rank_distribution=ranks,
        selected_baseline_pct=baseline_pct,
        skipped_baseline_steps=skipped_ambiguous_steps,
        catchup_steps=catchup_steps,
        trace_mode="ambiguous_only",
    )


def _result_to_dict(result: ProjectionResult) -> dict[str, Any]:
    return {
        "policy": result.policy,
        "baseline_steps": result.baseline_steps,
        "projected_steps": result.projected_steps,
        "projected_step_reduction_pct": result.projected_step_reduction_pct,
        "corrected_projected_speedup": result.corrected_projected_speedup,
        "ambiguous_steps_seen": result.ambiguous_steps_seen,
        "ambiguous_steps_used": result.ambiguous_steps_used,
        "ambiguous_accepted_len": result.ambiguous_accepted_len,
        "token0_1_rejection_proxy_pct": result.token01_rejection_proxy_pct,
        "oracle_gain_captured_pct": result.oracle_gain_captured_pct,
        "selected_rank_distribution": result.selected_rank_distribution,
        "selected_baseline_pct": result.selected_baseline_pct,
        "skipped_baseline_steps": result.skipped_baseline_steps,
        "catchup_steps": result.catchup_steps,
        "trace_mode": result.trace_mode,
    }


def _old_projection_for_policy(
    *,
    selected_mean: float,
    baseline_mean: float,
    ambiguous_runtime_fraction: float,
) -> float:
    if selected_mean <= 0 or baseline_mean <= 0:
        return 1.0
    improved = ambiguous_runtime_fraction * (baseline_mean / selected_mean)
    wall = (1.0 - ambiguous_runtime_fraction) + improved
    return 1.0 / wall if wall > 0 else 1.0


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# PLD Reranker Step-Replay Projection")
    lines.append("")
    lines.append(f"input: `{payload['input']}`")
    lines.append(f"trace mode: `{payload['trace_mode']}`")
    lines.append(f"baseline steps: `{payload['baseline_steps']}`")
    if payload.get("projection_warning"):
        lines.append("")
        lines.append(f"> {payload['projection_warning']}")
    lines.append("")
    lines.append(
        "| policy | projected steps | step reduction | corrected speedup | ambig accepted | <=1 | oracle captured | selected ranks | catch-up steps |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---:|")
    for row in payload["policies"]:
        ranks = ", ".join(f"{k}:{v}" for k, v in row["selected_rank_distribution"].items())
        lines.append(
            f"| {row['policy']} | {row['projected_steps']} | "
            f"{row['projected_step_reduction_pct']:.2f}% | "
            f"{row['corrected_projected_speedup']:.3f}x | "
            f"{row['ambiguous_accepted_len']:.2f} | "
            f"{row['token0_1_rejection_proxy_pct']:.1f}% | "
            f"{row['oracle_gain_captured_pct']:.1f}% | {ranks} | "
            f"{row['catchup_steps']} |"
        )
    if payload.get("old_vs_corrected_vs_actual"):
        lines.append("")
        lines.append("## Old Projection vs Corrected Projection vs Actual Runtime")
        lines.append("")
        lines.append("| policy | old accepted-len projection | corrected step projection | actual step ratio | actual tok/s ratio |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in payload["old_vs_corrected_vs_actual"]:
            lines.append(
                f"| {row['policy']} | {row['old_projection']:.3f}x | "
                f"{row['corrected_projection']:.3f}x | "
                f"{row.get('actual_step_ratio', 0.0):.3f}x | "
                f"{row.get('actual_tps_ratio', 0.0):.3f}x |"
            )
    path.write_text("\n".join(lines) + "\n")


def evaluate(
    *,
    input_path: Path,
    output_dir: Path,
    k: int,
    steps_path: Path | None,
    method: str,
    weights_files: list[Path],
    margin_values: list[float],
    oracle_ks: list[int],
    actual_policy_name: str | None = None,
    actual_baseline_steps: int | None = None,
    actual_policy_steps: int | None = None,
    actual_baseline_tps: float | None = None,
    actual_policy_tps: float | None = None,
) -> dict[str, Any]:
    rows = _load_jsonl(input_path)
    # Parse all candidates present in the JSONL first, then apply the runtime
    # top-K/baseline-inclusion rule.  Parsing only K raw candidates makes
    # fixed-rank policies disagree with the deployed decoder when the baseline
    # source is outside the first K raw occurrences.
    examples = _limit_examples_like_runtime(
        _parse_examples(rows, max(k, 64), enable_left_extension=True),
        top_k=k,
    )
    if not examples:
        raise SystemExit(f"no examples found in {input_path}")

    report = _read_report(input_path)
    report_steps = int(report.get("total_steps") or 0)
    ambiguous_runtime_fraction = float(report.get("ambiguous_runtime_fraction") or 0.0)
    steps_by_task: dict[str, list[BaselineStep]] | None = None
    if steps_path is not None:
        steps_by_task = _load_baseline_steps(steps_path, method=method)
        if not steps_by_task:
            raise SystemExit(f"no {method} rows found in {steps_path}")
        baseline_steps = sum(len(v) for v in steps_by_task.values())
        trace_mode = "full_steps"
        warning = ""
    else:
        baseline_steps = report_steps or len(examples)
        trace_mode = "ambiguous_only"
        warning = (
            "No full baseline steps trace was supplied; non-ambiguous decode "
            "steps are left untouched and short-candidate catch-up is modeled "
            "conservatively. Pass --steps for the accurate replay."
        )

    policies = _policy_table(
        examples,
        weights_files=weights_files,
        margin_values=margin_values,
        oracle_ks=oracle_ks,
    )
    result_rows: list[dict[str, Any]] = []
    baseline_mean = _safe_mean([float(_candidate_by_baseline(ex).accepted_len) for ex in examples])
    for name, choose in policies:
        if steps_by_task is not None:
            result = _project_with_full_steps(
                examples=examples, steps_by_task=steps_by_task, choose=choose
            )
        else:
            result = _project_ambiguous_only(
                examples=examples, choose=choose, baseline_steps=baseline_steps
            )
        row = _result_to_dict(ProjectionResult(policy=name, **{k: v for k, v in result.__dict__.items() if k != "policy"}))
        old_projection = _old_projection_for_policy(
            selected_mean=row["ambiguous_accepted_len"],
            baseline_mean=baseline_mean,
            ambiguous_runtime_fraction=ambiguous_runtime_fraction,
        )
        row["old_accepted_len_projection"] = old_projection
        result_rows.append(row)
    # Put baseline first, then highest corrected projection.
    result_rows.sort(
        key=lambda r: (
            1 if r["policy"] == "baseline_pld" else 0,
            r["corrected_projected_speedup"],
        ),
        reverse=True,
    )

    actual_rows: list[dict[str, Any]] = []
    if actual_policy_name:
        lookup = {row["policy"]: row for row in result_rows}
        # Allow users to pass the runtime label even when local policy names
        # contain the weight directory suffix.
        candidates = [
            row
            for row in result_rows
            if row["policy"] == actual_policy_name
            or actual_policy_name in row["policy"]
            or (
                actual_policy_name == "learned_v1"
                and row["policy"].startswith("learned_")
                and "_margin_" not in row["policy"]
            )
        ]
        if candidates:
            row = max(candidates, key=lambda r: r["corrected_projected_speedup"])
            actual_rows.append(
                {
                    "policy": actual_policy_name,
                    "old_projection": row["old_accepted_len_projection"],
                    "corrected_projection": row["corrected_projected_speedup"],
                    "actual_step_ratio": (
                        actual_baseline_steps / actual_policy_steps
                        if actual_baseline_steps and actual_policy_steps
                        else 0.0
                    ),
                    "actual_tps_ratio": (
                        actual_policy_tps / actual_baseline_tps
                        if actual_policy_tps and actual_baseline_tps
                        else 0.0
                    ),
                }
            )

    payload = {
        "input": str(input_path),
        "k": k,
        "method": method,
        "weights_files": [str(p) for p in weights_files],
        "oracle_ks": oracle_ks,
        "baseline_steps": baseline_steps,
        "trace_mode": trace_mode,
        "projection_warning": warning,
        "ambiguous_runtime_fraction": ambiguous_runtime_fraction,
        "policies": result_rows,
        "old_vs_corrected_vs_actual": actual_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(output_dir / "report.md", payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--steps", type=Path, default=None)
    ap.add_argument("--method", default="blazedit_pld_w128_n10")
    ap.add_argument("--weights", action="append", type=Path, default=[])
    ap.add_argument("--no-discover-weights", action="store_true")
    ap.add_argument("--margin-values", default="-2,-1,-0.5,0,0.25,0.5,1,2,4")
    ap.add_argument("--oracle-ks", default="4")
    ap.add_argument("--actual-policy-name", default="learned_v1")
    ap.add_argument("--actual-baseline-steps", type=int, default=6443)
    ap.add_argument("--actual-policy-steps", type=int, default=6148)
    ap.add_argument("--actual-baseline-tps", type=float, default=476.8)
    ap.add_argument("--actual-policy-tps", type=float, default=486.4)
    args = ap.parse_args()

    weights_files = list(args.weights)
    if not args.no_discover_weights:
        for path in _discover_weight_files(args.input):
            if path not in weights_files:
                weights_files.append(path)
    margin_values = [float(x) for x in args.margin_values.split(",") if x.strip()]
    oracle_ks = [int(x) for x in args.oracle_ks.split(",") if x.strip()]
    payload = evaluate(
        input_path=args.input,
        output_dir=args.output_dir,
        k=args.k,
        steps_path=args.steps,
        method=args.method,
        weights_files=weights_files,
        margin_values=margin_values,
        oracle_ks=oracle_ks,
        actual_policy_name=args.actual_policy_name,
        actual_baseline_steps=args.actual_baseline_steps,
        actual_policy_steps=args.actual_policy_steps,
        actual_baseline_tps=args.actual_baseline_tps,
        actual_policy_tps=args.actual_policy_tps,
    )
    print((args.output_dir / "report.md").read_text())
    best = max(
        (row for row in payload["policies"] if row["policy"] != "baseline_pld"),
        key=lambda r: r["corrected_projected_speedup"],
    )
    print(
        f"best non-baseline corrected projection: {best['policy']} "
        f"{best['corrected_projected_speedup']:.3f}x"
    )


if __name__ == "__main__":
    main()
