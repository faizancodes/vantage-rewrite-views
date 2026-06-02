"""Offline reranker evaluator for ambiguous PLD candidates.

This script consumes the ambiguous-candidate oracle rows produced by
``scripts/analyze_pld_candidate_oracle.py``.  It does not run a model and it
does not change decoding.  Candidate labels are the already-computed longest
common prefix with the verified PLD future output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asts.pld_reranker import (
    FEATURE_NAMES,
    PLDRerankCandidate,
    PLDRerankContext,
    continuation_key,
    extract_candidate_features,
    score_candidate,
)


DEFAULT_INPUT = Path("/tmp/pld_candidate_oracle_v2/ambiguous_candidates.jsonl")
DEFAULT_K = 4
DEFAULT_RANDOM_TRIALS = 2000


@dataclass(frozen=True)
class Candidate:
    idx: int
    rank0: int
    source_position: int
    source_type: str
    source_distance_from_previous_good_source: int | None
    accepted_len: int
    draft_prefix: str
    left_extension: int
    next2: str
    next4: str


@dataclass(frozen=True)
class Example:
    task_id: str
    step_id: int
    baseline_position: int | None
    baseline_accepted_len: int
    candidate_count: int
    match_len: int
    candidates: tuple[Candidate, ...]


@dataclass
class PolicyResult:
    name: str
    split: str
    n_examples: int
    mean_accepted_len: float
    projected_speedup: float
    oracle_gain_captured_pct: float
    oracle_best_selected_pct: float
    accepted_ge4_pct: float
    accepted_ge8_pct: float
    accepted_ge16_pct: float
    token01_rejection_pct: float
    selected_baseline_pct: float = 0.0
    selected_nonbaseline_pct: float = 0.0
    trigger_rate_pct: float = 0.0


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _left_extension_from_row(cand: dict[str, Any]) -> int:
    # Current oracle rows do not include source-left context.  Keep this feature
    # wired for richer future rows and make the policy a harmless no-op today.
    for key in (
        "left_extension",
        "left_extension_len",
        "source_left_extension",
        "left_match_len",
    ):
        val = cand.get(key)
        if isinstance(val, (int, float)):
            return int(val)
    return 0


def _parse_examples(
    rows: list[dict[str, Any]], k: int, *, enable_left_extension: bool = False
) -> list[Example]:
    examples: list[Example] = []
    for row in rows:
        raw_candidates = list(row.get("candidates") or [])[:k]
        candidates: list[Candidate] = []
        for idx, cand in enumerate(raw_candidates):
            draft = str(cand.get("candidate_draft_prefix_128") or "")
            candidates.append(
                Candidate(
                    idx=idx,
                    rank0=idx,
                    source_position=int(cand.get("source_position") or -1),
                    source_type=str(cand.get("source_type") or ""),
                    source_distance_from_previous_good_source=(
                        int(cand["source_distance_from_previous_good_source"])
                        if cand.get("source_distance_from_previous_good_source") is not None
                        else None
                    ),
                    accepted_len=int(cand.get("lcp_with_actual_future_output") or 0),
                    draft_prefix=draft,
                    left_extension=(
                        _left_extension_from_row(cand) if enable_left_extension else 0
                    ),
                    next2=continuation_key(draft, 2),
                    next4=continuation_key(draft, 4),
                )
            )
        if not candidates:
            continue
        examples.append(
            Example(
                task_id=str(row.get("task_id") or ""),
                step_id=int(row.get("step_id") or 0),
                baseline_position=(
                    int(row["baseline_candidate_position"])
                    if row.get("baseline_candidate_position") is not None
                    else None
                ),
                baseline_accepted_len=int(row.get("baseline_accepted_len") or 0),
                candidate_count=int(row.get("candidate_count_n10") or len(candidates)),
                match_len=int(row.get("match_len") or 0),
                candidates=tuple(candidates),
            )
        )
    return examples


def _split_by_task(examples: list[Example], seed: int) -> tuple[list[Example], list[Example]]:
    tasks = sorted({ex.task_id for ex in examples})

    def keyed(task_id: str) -> str:
        return hashlib.sha1(f"{seed}:{task_id}".encode("utf-8")).hexdigest()

    shuffled = sorted(tasks, key=keyed)
    train_tasks = set(shuffled[: len(shuffled) // 2])
    train = [ex for ex in examples if ex.task_id in train_tasks]
    valid = [ex for ex in examples if ex.task_id not in train_tasks]
    return train, valid


def _candidate_by_baseline(ex: Example) -> Candidate:
    if ex.baseline_position is not None:
        for cand in ex.candidates:
            if cand.source_position == ex.baseline_position:
                return cand
    return ex.candidates[0]


def _choose_rank(rank0: int) -> Callable[[Example], Candidate]:
    def choose(ex: Example) -> Candidate:
        return ex.candidates[min(rank0, len(ex.candidates) - 1)]

    return choose


def _choose_last_occurrence(ex: Example) -> Candidate:
    return max(ex.candidates, key=lambda c: (c.source_position, -c.rank0))


def _choose_source_continuity(ex: Example) -> Candidate:
    with_distance = [
        c for c in ex.candidates if c.source_distance_from_previous_good_source is not None
    ]
    if not with_distance:
        return _candidate_by_baseline(ex)
    return min(
        with_distance,
        key=lambda c: (
            c.source_distance_from_previous_good_source
            if c.source_distance_from_previous_good_source is not None
            else 10**12,
            c.rank0,
        ),
    )


def _choose_left_extension(ex: Example) -> Candidate:
    return max(ex.candidates, key=lambda c: (c.left_extension, -c.rank0))


def _choose_source_region_priority(ex: Example) -> Candidate:
    def score(c: Candidate) -> tuple[int, int, int]:
        typ = c.source_type.lower()
        generated = 1 if "generated" in typ else 0
        # Recent reference beats distant reference when source type is tied.
        return (generated, c.source_position, -c.rank0)

    return max(ex.candidates, key=score)


def _continuation_freqs(ex: Example) -> tuple[dict[str, int], dict[str, int]]:
    f2: dict[str, int] = {}
    f4: dict[str, int] = {}
    for c in ex.candidates:
        f2[c.next2] = f2.get(c.next2, 0) + 1
        f4[c.next4] = f4.get(c.next4, 0) + 1
    return f2, f4


def _choose_continuation_rarity(ex: Example) -> Candidate:
    f2, f4 = _continuation_freqs(ex)
    return max(
        ex.candidates,
        key=lambda c: (
            1.0 / f4.get(c.next4, 1),
            1.0 / f2.get(c.next2, 1),
            len(c.next4),
            -c.rank0,
        ),
    )


def _features(ex: Example, cand: Candidate) -> list[float]:
    shared_candidates = [
        PLDRerankCandidate(
            rank0=c.rank0,
            source_position=c.source_position,
            source_type=c.source_type,
            source_distance_from_previous_good_source=(
                c.source_distance_from_previous_good_source
            ),
            draft_prefix_text=c.draft_prefix,
            left_extension=c.left_extension,
            accepted_len=c.accepted_len,
        )
        for c in ex.candidates
    ]
    shared_cand = shared_candidates[cand.rank0]
    return extract_candidate_features(
        shared_cand,
        PLDRerankContext(candidate_count=ex.candidate_count, match_len=ex.match_len),
        all_candidates=shared_candidates,
    )


def _score(weights: list[float], ex: Example, cand: Candidate) -> float:
    return score_candidate(_features(ex, cand), weights)


def _choose_linear(weights: list[float]) -> Callable[[Example], Candidate]:
    def choose(ex: Example) -> Candidate:
        return max(ex.candidates, key=lambda c: (_score(weights, ex, c), -c.rank0))

    return choose


def _choose_linear_margin(weights: list[float], margin: float) -> Callable[[Example], Candidate]:
    def choose(ex: Example) -> Candidate:
        baseline = _candidate_by_baseline(ex)
        best = max(ex.candidates, key=lambda c: (_score(weights, ex, c), -c.rank0))
        if _score(weights, ex, best) - _score(weights, ex, baseline) < margin:
            return baseline
        return best

    return choose


def _mean_selected(examples: list[Example], choose: Callable[[Example], Candidate]) -> float:
    if not examples:
        return 0.0
    return statistics.mean(choose(ex).accepted_len for ex in examples)


def _feature_rows(examples: list[Example]) -> list[tuple[list[int], list[list[float]]]]:
    rows: list[tuple[list[int], list[list[float]]]] = []
    for ex in examples:
        rows.append(
            (
                [cand.accepted_len for cand in ex.candidates],
                [_features(ex, cand) for cand in ex.candidates],
            )
        )
    return rows


def _mean_linear_from_feature_rows(
    rows: list[tuple[list[int], list[list[float]]]], weights: list[float]
) -> float:
    if not rows:
        return 0.0
    total = 0.0
    for labels, feature_sets in rows:
        best_i = 0
        best_score = -1.0e300
        for i, feats in enumerate(feature_sets):
            score = 0.0
            for w, x in zip(weights, feats):
                score += w * x
            if score > best_score:
                best_i = i
                best_score = score
        total += labels[best_i]
    return total / len(rows)


def _train_linear_random_search(
    train: list[Example],
    *,
    trials: int,
    seed: int,
) -> tuple[list[float], float]:
    rng = random.Random(seed)
    candidates: list[list[float]] = []

    def one_hot(name: str, val: float = 1.0) -> list[float]:
        w = [0.0] * len(FEATURE_NAMES)
        w[FEATURE_NAMES.index(name)] = val
        return w

    # Seed with interpretable policies and common priors before random search.
    candidates.extend(
        [
            one_hot("rank0", 1.0),
            one_hot("rank1", 1.0),
            one_hot("rank2", 1.0),
            one_hot("rank3", 1.0),
            one_hot("source_pos_norm", 1.0),
            one_hot("is_most_recent", 1.0),
            one_hot("continuity_close", 1.0),
            one_hot("left_extension", 1.0),
            one_hot("next4_inv_freq", 1.0),
            one_hot("next2_inv_freq", 1.0),
        ]
    )
    for _ in range(trials):
        # Keep weights small and cheap-looking; this is meant to approximate a
        # runtime rule, not a high-capacity offline model.
        candidates.append([rng.gauss(0.0, 1.0) for _ in FEATURE_NAMES])

    feature_rows = _feature_rows(train)
    best_weights = candidates[0]
    best_train = -1.0
    for weights in candidates:
        mean = _mean_linear_from_feature_rows(feature_rows, weights)
        if mean > best_train:
            best_train = mean
            best_weights = weights
    return best_weights, best_train


def _projected_speedup(
    *,
    baseline_mean: float,
    policy_mean: float,
    ambiguous_runtime_fraction: float,
) -> float:
    if baseline_mean <= 0 or policy_mean <= 0:
        return 1.0
    improved_ambiguous_wall = ambiguous_runtime_fraction * (baseline_mean / policy_mean)
    projected_wall = (1.0 - ambiguous_runtime_fraction) + improved_ambiguous_wall
    return 1.0 / projected_wall if projected_wall > 0 else 1.0


def _evaluate_policy(
    *,
    name: str,
    split: str,
    examples: list[Example],
    choose: Callable[[Example], Candidate],
    baseline_mean: float,
    oracle_mean: float,
    ambiguous_runtime_fraction: float,
) -> PolicyResult:
    selected = [choose(ex) for ex in examples]
    n = len(selected)
    mean = statistics.mean(c.accepted_len for c in selected) if selected else 0.0
    oracle_best = [max(c.accepted_len for c in ex.candidates) for ex in examples]
    baseline_selected = [
        cand.source_position == _candidate_by_baseline(ex).source_position
        for ex, cand in zip(examples, selected, strict=False)
    ]
    denom = oracle_mean - baseline_mean
    captured = 100.0 * (mean - baseline_mean) / denom if denom > 0 else 0.0
    return PolicyResult(
        name=name,
        split=split,
        n_examples=n,
        mean_accepted_len=mean,
        projected_speedup=_projected_speedup(
            baseline_mean=baseline_mean,
            policy_mean=mean,
            ambiguous_runtime_fraction=ambiguous_runtime_fraction,
        ),
        oracle_gain_captured_pct=captured,
        oracle_best_selected_pct=(
            100.0
            * sum(
                cand.accepted_len == best
                for cand, best in zip(selected, oracle_best, strict=False)
            )
            / n
            if n
            else 0.0
        ),
        accepted_ge4_pct=100.0 * sum(c.accepted_len >= 4 for c in selected) / n if n else 0.0,
        accepted_ge8_pct=100.0 * sum(c.accepted_len >= 8 for c in selected) / n if n else 0.0,
        accepted_ge16_pct=100.0 * sum(c.accepted_len >= 16 for c in selected) / n if n else 0.0,
        token01_rejection_pct=(
            100.0 * sum(c.accepted_len <= 1 for c in selected) / n if n else 0.0
        ),
        selected_baseline_pct=(
            100.0 * sum(1 for x in baseline_selected if x) / n if n else 0.0
        ),
        selected_nonbaseline_pct=(
            100.0 * sum(1 for x in baseline_selected if not x) / n if n else 0.0
        ),
        trigger_rate_pct=(
            100.0 * sum(1 for x in baseline_selected if not x) / n if n else 0.0
        ),
    )


def _policy_result_dict(r: PolicyResult) -> dict[str, Any]:
    return {
        "policy": r.name,
        "split": r.split,
        "n_examples": r.n_examples,
        "heldout_ambiguous_accepted_len": r.mean_accepted_len,
        "projected_speedup_vs_pld": r.projected_speedup,
        "k4_oracle_gain_captured_pct": r.oracle_gain_captured_pct,
        "selected_oracle_best_candidate_rate_pct": r.oracle_best_selected_pct,
        "selected_accepted_len_ge4_pct": r.accepted_ge4_pct,
        "selected_accepted_len_ge8_pct": r.accepted_ge8_pct,
        "selected_accepted_len_ge16_pct": r.accepted_ge16_pct,
        "token0_1_rejection_proxy_pct": r.token01_rejection_pct,
        "selected_baseline_rate_pct": r.selected_baseline_pct,
        "selected_nonbaseline_rate_pct": r.selected_nonbaseline_pct,
        "trigger_rate_pct": r.trigger_rate_pct,
    }


def _read_ambiguous_runtime_fraction(input_path: Path) -> float:
    report = input_path.with_name("report.json")
    if not report.exists():
        return 0.4074417299646001
    try:
        data = json.loads(report.read_text())
        return float(data.get("ambiguous_runtime_fraction") or 0.4074417299646001)
    except Exception:
        return 0.4074417299646001


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# PLD Candidate Reranker Evaluation")
    lines.append("")
    lines.append(f"input: `{payload['input']}`")
    lines.append(f"K: `{payload['k']}`")
    lines.append(
        f"train examples: `{payload['train_examples']}`, valid examples: `{payload['valid_examples']}`"
    )
    lines.append(
        f"valid baseline ambiguous accepted: `{payload['valid_baseline_accepted_len']:.3f}`"
    )
    lines.append(f"valid K=4 oracle accepted: `{payload['valid_oracle_accepted_len']:.3f}`")
    lines.append("")
    lines.append("| policy | accepted len | projected speedup | oracle gain captured | oracle-best rate | >=4 | >=8 | >=16 | <=1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in payload["valid_results"]:
        lines.append(
            "| {policy} | {heldout_ambiguous_accepted_len:.3f} | {projected_speedup_vs_pld:.3f}x | "
            "{k4_oracle_gain_captured_pct:.1f}% | {selected_oracle_best_candidate_rate_pct:.1f}% | "
            "{selected_accepted_len_ge4_pct:.1f}% | {selected_accepted_len_ge8_pct:.1f}% | "
            "{selected_accepted_len_ge16_pct:.1f}% | {token0_1_rejection_proxy_pct:.1f}% |".format(
                **row
            )
        )
    lines.append("")
    lines.append(f"best valid policy: `{payload['best_valid_policy']}`")
    lines.append(f"success threshold met: `{payload['success_threshold_met']}`")
    if payload.get("margin_sweep_valid"):
        lines.append("")
        lines.append("## Learned Linear Margin Sweep")
        lines.append("")
        lines.append("| margin | trigger rate | accepted len | projected speedup | oracle gain captured | selected baseline | selected non-baseline | <=1 |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in payload["margin_sweep_valid"]:
            lines.append(
                "| {margin:g} | {trigger_rate_pct:.1f}% | {heldout_ambiguous_accepted_len:.3f} | "
                "{projected_speedup_vs_pld:.3f}x | {k4_oracle_gain_captured_pct:.1f}% | "
                "{selected_baseline_rate_pct:.1f}% | {selected_nonbaseline_rate_pct:.1f}% | "
                "{token0_1_rejection_proxy_pct:.1f}% |".format(**row)
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--random-trials", type=int, default=DEFAULT_RANDOM_TRIALS)
    ap.add_argument("--ambiguous-runtime-fraction", type=float, default=None)
    ap.add_argument("--enable-left-extension", action="store_true")
    ap.add_argument("--sweep-margin", action="store_true")
    ap.add_argument(
        "--margin-values",
        default="-2,-1,-0.5,0,0.25,0.5,1,2,4",
        help="Comma-separated score margins for --sweep-margin.",
    )
    args = ap.parse_args()

    rows = _load_jsonl(args.input)
    examples = _parse_examples(
        rows, args.k, enable_left_extension=args.enable_left_extension
    )
    if not examples:
        raise SystemExit(f"no examples found in {args.input}")
    train, valid = _split_by_task(examples, args.seed)
    ambiguous_runtime_fraction = (
        float(args.ambiguous_runtime_fraction)
        if args.ambiguous_runtime_fraction is not None
        else _read_ambiguous_runtime_fraction(args.input)
    )

    weights, train_mean = _train_linear_random_search(
        train, trials=args.random_trials, seed=args.seed
    )
    policies: list[tuple[str, Callable[[Example], Candidate]]] = [
        ("baseline_current_pld_choice", _candidate_by_baseline),
        ("candidate_rank_0", _choose_rank(0)),
        ("candidate_rank_1", _choose_rank(1)),
        ("candidate_rank_2", _choose_rank(2)),
        ("candidate_rank_3", _choose_rank(3)),
        ("last_occurrence_most_recent_source", _choose_last_occurrence),
        ("source_continuity", _choose_source_continuity),
        ("left_extension", _choose_left_extension),
        ("source_region_priority", _choose_source_region_priority),
        ("continuation_rarity", _choose_continuation_rarity),
        ("learned_linear_random_search", _choose_linear(weights)),
    ]
    margin_values = [
        float(x)
        for x in str(args.margin_values).split(",")
        if x.strip()
    ]
    if args.sweep_margin:
        for margin in margin_values:
            policies.append(
                (
                    f"learned_linear_margin_{margin:g}",
                    _choose_linear_margin(weights, margin),
                )
            )

    baseline_valid_mean = _mean_selected(valid, _candidate_by_baseline)
    oracle_valid_mean = statistics.mean(
        max(c.accepted_len for c in ex.candidates) for ex in valid
    )
    baseline_train_mean = _mean_selected(train, _candidate_by_baseline)
    oracle_train_mean = statistics.mean(
        max(c.accepted_len for c in ex.candidates) for ex in train
    )

    train_results = [
        _policy_result_dict(
            _evaluate_policy(
                name=name,
                split="train",
                examples=train,
                choose=choose,
                baseline_mean=baseline_train_mean,
                oracle_mean=oracle_train_mean,
                ambiguous_runtime_fraction=ambiguous_runtime_fraction,
            )
        )
        for name, choose in policies
    ]
    valid_results = [
        _policy_result_dict(
            _evaluate_policy(
                name=name,
                split="valid",
                examples=valid,
                choose=choose,
                baseline_mean=baseline_valid_mean,
                oracle_mean=oracle_valid_mean,
                ambiguous_runtime_fraction=ambiguous_runtime_fraction,
            )
        )
        for name, choose in policies
    ]
    valid_results.sort(key=lambda r: r["heldout_ambiguous_accepted_len"], reverse=True)
    train_results.sort(key=lambda r: r["heldout_ambiguous_accepted_len"], reverse=True)
    margin_sweep_valid = [
        dict(row, margin=float(row["policy"].rsplit("_", 1)[-1]))
        for row in valid_results
        if row["policy"].startswith("learned_linear_margin_")
    ]
    margin_sweep_valid.sort(key=lambda r: r["margin"])
    margin_sweep_train = [
        dict(row, margin=float(row["policy"].rsplit("_", 1)[-1]))
        for row in train_results
        if row["policy"].startswith("learned_linear_margin_")
    ]
    margin_sweep_train.sort(key=lambda r: r["margin"])

    best = valid_results[0]
    selected_margin = None
    if best["policy"].startswith("learned_linear_margin_"):
        selected_margin = float(best["policy"].rsplit("_", 1)[-1])
    success_threshold_met = bool(
        best["heldout_ambiguous_accepted_len"] >= 3.6
        or best["projected_speedup_vs_pld"] >= 1.20
    )
    strong_threshold_met = bool(best["heldout_ambiguous_accepted_len"] >= 4.5)
    excellent_threshold_met = bool(best["heldout_ambiguous_accepted_len"] >= 5.0)

    out_dir = args.output_dir or args.input.parent / "reranker_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "reranker_weights.json"
    payload = {
        "input": str(args.input),
        "k": args.k,
        "seed": args.seed,
        "random_trials": args.random_trials,
        "enable_left_extension": args.enable_left_extension,
        "sweep_margin": args.sweep_margin,
        "ambiguous_runtime_fraction": ambiguous_runtime_fraction,
        "n_examples": len(examples),
        "train_examples": len(train),
        "valid_examples": len(valid),
        "train_tasks": len({ex.task_id for ex in train}),
        "valid_tasks": len({ex.task_id for ex in valid}),
        "train_baseline_accepted_len": baseline_train_mean,
        "valid_baseline_accepted_len": baseline_valid_mean,
        "train_oracle_accepted_len": oracle_train_mean,
        "valid_oracle_accepted_len": oracle_valid_mean,
        "learned_linear_train_mean": train_mean,
        "feature_names": FEATURE_NAMES,
        "learned_linear_weights": {
            name: weight for name, weight in zip(FEATURE_NAMES, weights)
        },
        "train_results": train_results,
        "valid_results": valid_results,
        "margin_sweep_train": margin_sweep_train,
        "margin_sweep_valid": margin_sweep_valid,
        "best_valid_policy": best["policy"],
        "success_threshold_met": success_threshold_met,
        "strong_threshold_met": strong_threshold_met,
        "excellent_threshold_met": excellent_threshold_met,
        "recommend_runtime_decoder": (
            "rerank_exact_pld_k4_w128_n10" if success_threshold_met else None
        ),
        "reranker_weights_path": str(weights_path) if success_threshold_met else None,
    }
    (out_dir / "reranker_report.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(out_dir / "reranker_report.md", payload)
    if success_threshold_met:
        weights_payload = {
            "decoder": "rerank_exact_pld_k4_w128_n10",
            "k": args.k,
            "feature_names": FEATURE_NAMES,
            "weights": weights,
            "enable_left_extension": args.enable_left_extension,
            "margin_sweep": margin_sweep_valid,
            "selected_margin": selected_margin,
            "validation_policy": best["policy"],
            "validation_ambiguous_accepted_len": best[
                "heldout_ambiguous_accepted_len"
            ],
            "validation_projected_speedup_vs_pld": best["projected_speedup_vs_pld"],
        }
        weights_path.write_text(json.dumps(weights_payload, indent=2) + "\n")

    print((out_dir / "reranker_report.md").read_text())
    if success_threshold_met:
        print(f"Wrote {weights_path}")
        print("Recommendation: implement runtime decoder rerank_exact_pld_k4_w128_n10")
    else:
        print("No policy reached the minimum success threshold; no weights file written.")


if __name__ == "__main__":
    main()
