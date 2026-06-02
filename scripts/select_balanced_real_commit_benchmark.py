#!/usr/bin/env python3
"""Select a method-neutral real-commit benchmark from verified commit rows.

The earlier 1000-task real-commit manifest was balanced primarily by repository
and coarse family.  That is useful for external validity, but it creates a
PLD-heavy aggregate because many real refactors are tiny, high-copy edits with
long unchanged spans.  This selector instead balances only pre-decode workload
properties:

* exact-copy pressure: copied-token percentage and longest unchanged span
* transformed-reference opportunity: rewrite density and transformed-reference
  fit to the committed target
* edit difficulty: edit distance and hunk count
* surface diversity: family, map count, and repository caps

The script never reads model completions or method timings.  The resulting
manifest is intended for a fair PLD-vs-VANTAGE comparison with stratified
reporting, not as a benchmark tuned to either method.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+|\S")
_COMMON_TERMS = {
    "a",
    "b",
    "c",
    "d",
    "data",
    "dict",
    "e",
    "f",
    "file",
    "files",
    "g",
    "h",
    "i",
    "id",
    "ids",
    "item",
    "items",
    "j",
    "k",
    "key",
    "keys",
    "list",
    "m",
    "n",
    "name",
    "names",
    "new",
    "none",
    "null",
    "obj",
    "object",
    "old",
    "path",
    "random",
    "result",
    "results",
    "self",
    "set",
    "str",
    "temp",
    "tmp",
    "type",
    "val",
    "value",
    "values",
    "x",
    "y",
    "z",
}


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text or "")


def _token_common_ratio(a: str, b: str) -> float:
    left = _tokens(a)
    right = _tokens(b)
    if not right:
        return 0.0
    matcher = SequenceMatcher(a=left, b=right, autojunk=False)
    return sum(match.size for match in matcher.get_matching_blocks()) / len(right)


def _apply_pairs(reference: str, pairs: dict[str, str]) -> str:
    out = reference
    for old, new in sorted(pairs.items(), key=lambda item: -len(item[0])):
        if not old or old == new:
            continue
        if re.match(r"^[A-Za-z_][A-Za-z_0-9.]*$", old):
            pattern = re.compile(
                r"(?<![A-Za-z_0-9.])" + re.escape(old) + r"(?![A-Za-z_0-9.])"
            )
            out = pattern.sub(new, out)
        else:
            out = out.replace(old, new)
    return out


def _coerce_pairs(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {
            str(k): str(v)
            for k, v in value.items()
            if str(k) and str(v) and str(k) != str(v)
        }
    if isinstance(value, list):
        out: dict[str, str] = {}
        for item in value:
            if isinstance(item, dict):
                old = item.get("old") or item.get("from")
                new = item.get("new") or item.get("to")
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                old, new = item
            else:
                continue
            if old and new and str(old) != str(new):
                out[str(old)] = str(new)
        return out
    return {}


def _bin_copy(value: float) -> str:
    if value < 0.75:
        return "copy_low_lt75"
    if value < 0.90:
        return "copy_mid_75_90"
    if value < 0.98:
        return "copy_high_90_98"
    return "copy_very_high_ge98"


def _bin_density(value: float) -> str:
    if value < 1.0:
        return "rewrite_sparse_lt1"
    if value < 3.0:
        return "rewrite_mid_1_3"
    return "rewrite_dense_ge3"


def _bin_fit(value: float) -> str:
    if value < 0.95:
        return "transformed_dirty_lt95"
    if value < 0.995:
        return "transformed_aligned_95_995"
    return "transformed_exact_ge995"


def _bin_edit(value: int) -> str:
    if value <= 8:
        return "edit_small_le8"
    if value <= 64:
        return "edit_medium_9_64"
    return "edit_large_ge65"


def _bin_longest(value: int) -> str:
    if value < 64:
        return "span_short_lt64"
    if value < 128:
        return "span_medium_64_127"
    return "span_long_ge128"


def _bin_hunks(value: int) -> str:
    if value <= 1:
        return "hunks_1"
    if value == 2:
        return "hunks_2"
    return "hunks_ge3"


def _bin_maps(value: int) -> str:
    if value <= 1:
        return "maps_1"
    if value == 2:
        return "maps_2"
    return "maps_ge3"


@dataclass(frozen=True)
class RowMetrics:
    copy_bin: str
    density_bin: str
    fit_bin: str
    edit_bin: str
    longest_bin: str
    hunk_bin: str
    map_bin: str
    family: str
    repo: str
    commit_key: tuple[str, str]
    copied_token_percentage: float
    rewrite_density_per_100_tokens: float
    transformed_reference_fit: float
    dirty_vs_transformed_reference: float
    rewrite_occurrences_in_target: int
    rewrite_occurrences_in_reference: int
    noisy_map_count: int
    benchmark_regime: str


def _metric_row(row: dict[str, Any]) -> RowMetrics:
    pairs = _coerce_pairs(row.get("rewrite_pairs"))
    reference = str(row.get("reference") or "")
    target = str(row.get("deterministic_target") or "")
    target_tokens = _tokens(target)
    transformed = _apply_pairs(reference, pairs)
    transformed_fit = _token_common_ratio(transformed, target)
    rewrite_occ_target = sum(target.count(new) for new in pairs.values())
    rewrite_occ_ref = sum(reference.count(old) for old in pairs)
    density = rewrite_occ_target / max(1, len(target_tokens)) * 100.0
    noisy = 0
    for old, new in pairs.items():
        if (
            len(old) <= 2
            or len(new) <= 2
            or old.lower() in _COMMON_TERMS
            or new.lower() in _COMMON_TERMS
        ):
            noisy += 1
    copied = float(row.get("copied_token_percentage") or 0.0)
    edit = int(row.get("edit_distance_tokens") or 0)
    longest = int(row.get("longest_unchanged_span_tokens") or 0)
    hunks = int(row.get("changed_hunk_count") or 0)
    family = str(row.get("drift_family") or "unknown")
    repo = str(row.get("repo") or "unknown")
    if copied >= 0.98 and longest >= 128 and density < 2.0:
        regime = "pld_favored_exact_copy"
    elif transformed_fit >= 0.95 and density >= 1.0 and edit >= 4:
        regime = "rewrite_aligned_drift"
    elif transformed_fit < 0.95 or edit >= 65 or hunks >= 3:
        regime = "dirty_or_large_refactor"
    else:
        regime = "mixed_reference_edit"
    return RowMetrics(
        copy_bin=_bin_copy(copied),
        density_bin=_bin_density(density),
        fit_bin=_bin_fit(transformed_fit),
        edit_bin=_bin_edit(edit),
        longest_bin=_bin_longest(longest),
        hunk_bin=_bin_hunks(hunks),
        map_bin=_bin_maps(len(pairs)),
        family=family,
        repo=repo,
        commit_key=(repo, str(row.get("commit_sha") or "")),
        copied_token_percentage=copied,
        rewrite_density_per_100_tokens=density,
        transformed_reference_fit=transformed_fit,
        dirty_vs_transformed_reference=1.0 - transformed_fit,
        rewrite_occurrences_in_target=rewrite_occ_target,
        rewrite_occurrences_in_reference=rewrite_occ_ref,
        noisy_map_count=noisy,
        benchmark_regime=regime,
    )


def _target_leaked(row: dict[str, Any]) -> bool:
    target = str(row.get("deterministic_target") or "").strip()
    prompt = str(row.get("prompt") or "")
    return not target or target in prompt


def _parseable_python(text: str) -> bool:
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        return False


def _counter_table(selected_metrics: list[RowMetrics], attr: str) -> dict[str, int]:
    return dict(Counter(getattr(metric, attr) for metric in selected_metrics))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = (len(values) - 1) * q / 100.0
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - idx) + values[hi] * (idx - lo)


def _numeric_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values) if values else float("nan"),
        "p10": _percentile(values, 10),
        "p25": _percentile(values, 25),
        "median": _percentile(values, 50),
        "p75": _percentile(values, 75),
        "p90": _percentile(values, 90),
    }


def _format_summary(name: str, summary: dict[str, float]) -> str:
    return (
        f"| {name} | {summary['mean']:.3f} | {summary['p10']:.3f} | "
        f"{summary['p25']:.3f} | {summary['median']:.3f} | "
        f"{summary['p75']:.3f} | {summary['p90']:.3f} |"
    )


def _write_report(
    *,
    output_md: Path,
    selected: list[dict[str, Any]],
    selected_metrics: list[RowMetrics],
    pool_metrics: list[RowMetrics],
    targets: dict[str, dict[str, int]],
    source: Path,
) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Balanced Real-Commit Benchmark Audit")
    lines.append("")
    lines.append(f"Source pool: `{source}`")
    lines.append(f"Selected rows: {len(selected)}")
    lines.append("")
    lines.append("## Selection Principle")
    lines.append("")
    lines.append(
        "Rows are selected using pre-decode workload properties only. The "
        "selector does not read model outputs, PLD timings, VANTAGE timings, "
        "or any method-specific acceptance statistics."
    )
    lines.append("")
    lines.append("## Targeted Marginal Balance")
    lines.append("")
    for attr, quota in targets.items():
        lines.append(f"### {attr}")
        lines.append("")
        selected_counts = _counter_table(selected_metrics, attr)
        pool_counts = _counter_table(pool_metrics, attr)
        lines.append("| bin | target | selected | pool |")
        lines.append("|---|---:|---:|---:|")
        for key in sorted(set(quota) | set(selected_counts) | set(pool_counts)):
            lines.append(
                f"| `{key}` | {quota.get(key, 0)} | "
                f"{selected_counts.get(key, 0)} | {pool_counts.get(key, 0)} |"
            )
        lines.append("")
    lines.append("## Numeric Distribution")
    lines.append("")
    numeric: list[tuple[str, Callable[[RowMetrics], float]]] = [
        ("copied_token_percentage", lambda m: m.copied_token_percentage),
        ("rewrite_density_per_100_tokens", lambda m: m.rewrite_density_per_100_tokens),
        ("transformed_reference_fit", lambda m: m.transformed_reference_fit),
        ("dirty_vs_transformed_reference", lambda m: m.dirty_vs_transformed_reference),
        ("rewrite_occurrences_in_target", lambda m: float(m.rewrite_occurrences_in_target)),
    ]
    lines.append("| metric | mean | p10 | p25 | median | p75 | p90 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, getter in numeric:
        lines.append(_format_summary(name, _numeric_summary([getter(m) for m in selected_metrics])))
    lines.append("")
    lines.append("## Method-Neutral Regime Counts")
    lines.append("")
    lines.append("| regime | selected | pool |")
    lines.append("|---|---:|---:|")
    selected_regimes = _counter_table(selected_metrics, "benchmark_regime")
    pool_regimes = _counter_table(pool_metrics, "benchmark_regime")
    for key in sorted(set(selected_regimes) | set(pool_regimes)):
        lines.append(f"| `{key}` | {selected_regimes.get(key, 0)} | {pool_regimes.get(key, 0)} |")
    lines.append("")
    lines.append("## Validity Checks")
    lines.append("")
    lines.append(f"- unique repo/commit rows: {len({m.commit_key for m in selected_metrics})}/{len(selected)}")
    lines.append(
        f"- target leak rows: {sum(1 for row in selected if _target_leaked(row))}/{len(selected)}"
    )
    lines.append(
        "- parseable deterministic targets: "
        f"{sum(1 for row in selected if _parseable_python(str(row.get('deterministic_target') or '')))}/{len(selected)}"
    )
    lines.append(
        f"- tasks with noisy/common maps: "
        f"{sum(1 for m in selected_metrics if m.noisy_map_count > 0)}/{len(selected_metrics)}"
    )
    lines.append("")
    output_md.write_text("\n".join(lines) + "\n")


def _score_candidate(
    metric: RowMetrics,
    selected_counts: dict[str, Counter[str]],
    targets: dict[str, dict[str, int]],
    repo_counts: Counter[str],
    max_per_repo: int,
) -> float:
    score = 0.0
    attrs = {
        "copy_bin": metric.copy_bin,
        "density_bin": metric.density_bin,
        "fit_bin": metric.fit_bin,
        "edit_bin": metric.edit_bin,
        "longest_bin": metric.longest_bin,
        "hunk_bin": metric.hunk_bin,
        "map_bin": metric.map_bin,
        "family": metric.family,
        "benchmark_regime": metric.benchmark_regime,
    }
    weights = {
        "copy_bin": 4.0,
        "density_bin": 4.0,
        "fit_bin": 4.0,
        "edit_bin": 3.0,
        "longest_bin": 3.0,
        "hunk_bin": 2.0,
        "map_bin": 1.5,
        "family": 2.0,
        "benchmark_regime": 4.0,
    }
    for attr, value in attrs.items():
        target = targets[attr].get(value, 0)
        current = selected_counts[attr][value]
        if current < target:
            # Larger score for scarce underfilled bins.
            score += weights[attr] * (target - current) / max(1, target)
        else:
            score -= weights[attr] * (current - target + 1) / max(1, target)
    score -= 1.5 * repo_counts[metric.repo] / max(1, max_per_repo)
    score -= 0.15 * metric.noisy_map_count
    return score


def _default_targets(target_rows: int) -> dict[str, dict[str, int]]:
    # The quotas intentionally do not mirror the source distribution. They cap
    # exact-copy dominance while preserving enough PLD-favored examples to keep
    # the benchmark honest.
    return {
        "copy_bin": {
            "copy_low_lt75": 190,
            "copy_mid_75_90": 240,
            "copy_high_90_98": 280,
            "copy_very_high_ge98": target_rows - 190 - 240 - 280,
        },
        "density_bin": {
            "rewrite_sparse_lt1": 300,
            "rewrite_mid_1_3": 360,
            "rewrite_dense_ge3": target_rows - 300 - 360,
        },
        "fit_bin": {
            "transformed_dirty_lt95": 360,
            "transformed_aligned_95_995": 320,
            "transformed_exact_ge995": target_rows - 360 - 320,
        },
        "edit_bin": {
            "edit_small_le8": 300,
            "edit_medium_9_64": 500,
            "edit_large_ge65": target_rows - 300 - 500,
        },
        "longest_bin": {
            "span_short_lt64": 330,
            "span_medium_64_127": 280,
            "span_long_ge128": target_rows - 330 - 280,
        },
        "hunk_bin": {
            "hunks_1": 300,
            "hunks_2": 250,
            "hunks_ge3": target_rows - 300 - 250,
        },
        "map_bin": {
            "maps_1": 600,
            "maps_2": 200,
            "maps_ge3": target_rows - 600 - 200,
        },
        "family": {
            "real_rename": target_rows // 2,
            "real_field_migration": target_rows - target_rows // 2,
        },
        "benchmark_regime": {
            "pld_favored_exact_copy": 220,
            "rewrite_aligned_drift": 170,
            "mixed_reference_edit": 260,
            "dirty_or_large_refactor": target_rows - 220 - 170 - 260,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-md", required=True)
    parser.add_argument("--target-rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--max-per-repo", type=int, default=90)
    parser.add_argument("--unique-commits", action="store_true", default=True)
    args = parser.parse_args()

    source = Path(args.input_jsonl)
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    pairs = [(row, _metric_row(row)) for row in rows]
    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    targets = _default_targets(args.target_rows)
    selected: list[dict[str, Any]] = []
    selected_metrics: list[RowMetrics] = []
    selected_task_keys: set[str] = set()
    selected_commit_keys: set[tuple[str, str]] = set()
    repo_counts: Counter[str] = Counter()
    selected_counts: dict[str, Counter[str]] = defaultdict(Counter)

    def add(row: dict[str, Any], metric: RowMetrics) -> bool:
        if len(selected) >= args.target_rows:
            return False
        original_task = str(row.get("task_id") or "")
        if original_task in selected_task_keys:
            return False
        if args.unique_commits and metric.commit_key in selected_commit_keys:
            return False
        if repo_counts[metric.repo] >= args.max_per_repo:
            return False
        if _target_leaked(row):
            return False
        if not _parseable_python(str(row.get("deterministic_target") or "")):
            return False
        selected_task_keys.add(original_task)
        selected_commit_keys.add(metric.commit_key)
        repo_counts[metric.repo] += 1
        row = dict(row)
        row["source_task_id"] = original_task
        row["task_id"] = f"real_commit_balanced_python/{len(selected):04d}"
        row["benchmark_name"] = "real_commit_balanced_1000_v1"
        row["benchmark_regime"] = metric.benchmark_regime
        row["benchmark_bins"] = {
            "copy": metric.copy_bin,
            "rewrite_density": metric.density_bin,
            "transformed_fit": metric.fit_bin,
            "edit_distance": metric.edit_bin,
            "longest_span": metric.longest_bin,
            "hunks": metric.hunk_bin,
            "map_count": metric.map_bin,
        }
        row["transformed_reference_fit"] = metric.transformed_reference_fit
        row["rewrite_density_per_100_tokens"] = metric.rewrite_density_per_100_tokens
        row["dirty_vs_transformed_reference"] = metric.dirty_vs_transformed_reference
        row["rewrite_occurrences_in_target"] = metric.rewrite_occurrences_in_target
        row["rewrite_occurrences_in_reference"] = metric.rewrite_occurrences_in_reference
        row["noisy_map_count"] = metric.noisy_map_count
        selected.append(row)
        selected_metrics.append(metric)
        for attr in targets:
            selected_counts[attr][getattr(metric, attr)] += 1
        return True

    # Greedy deficit minimization. Recompute after each pick; 1299 rows is small
    # enough that the simple loop is easier to audit than an optimizer.
    remaining = pairs[:]
    while len(selected) < args.target_rows and remaining:
        best_idx = -1
        best_score = -1e18
        for idx, (row, metric) in enumerate(remaining):
            if repo_counts[metric.repo] >= args.max_per_repo:
                continue
            if args.unique_commits and metric.commit_key in selected_commit_keys:
                continue
            score = _score_candidate(metric, selected_counts, targets, repo_counts, args.max_per_repo)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx < 0:
            break
        row, metric = remaining.pop(best_idx)
        add(row, metric)

    if len(selected) != args.target_rows:
        raise SystemExit(f"selected {len(selected)} rows, expected {args.target_rows}")

    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    _write_report(
        output_md=Path(args.report_md),
        selected=selected,
        selected_metrics=selected_metrics,
        pool_metrics=[metric for _, metric in pairs],
        targets=targets,
        source=source,
    )

    print(f"wrote {len(selected)} rows to {output}")
    print(f"wrote report to {args.report_md}")
    print("families", dict(Counter(metric.family for metric in selected_metrics)))
    print("regimes", dict(Counter(metric.benchmark_regime for metric in selected_metrics)))
    print("copy", dict(Counter(metric.copy_bin for metric in selected_metrics)))
    print("density", dict(Counter(metric.density_bin for metric in selected_metrics)))
    print("fit", dict(Counter(metric.fit_bin for metric in selected_metrics)))


if __name__ == "__main__":
    main()
