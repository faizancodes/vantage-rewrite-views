#!/usr/bin/env python3
"""Summarize VANTAGE residual/post-PLD projection and runtime artifacts.

The analyzer is intentionally conservative: it reports only fields present in
the supplied JSON files and labels each row as either an offline projection or
a runtime measured result.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_GLOBS = (
    "analysis/real_commits/modal/vantage_real_commit_pld_adjacent*/eval/aggregate.json",
)
DEFAULT_PROJECTION_GLOBS = (
    "analysis/pld_mtp/postpld*/report.json",
)
DEFAULT_OUTPUT_JSON = Path("artifacts/vantage_residual/tables/residual_analysis.json")
DEFAULT_BASELINE = "blazedit_pld_w128_n10"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def stable_relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def discover_paths(patterns: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(ROOT.glob(pattern))
    return sorted({path.resolve() for path in paths if path.is_file()})


def as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def ratio(numerator: Any, denominator: Any) -> float | None:
    num = as_float(numerator)
    den = as_float(denominator)
    if num is None or den in (None, 0.0):
        return None
    return num / den


def pct_from_fraction(value: Any) -> float | None:
    number = as_float(value)
    if number is None:
        return None
    return 100.0 * number


def pld_exact_hit_rate(row: dict[str, Any]) -> float | None:
    direct = as_float(row.get("pld_exact_hit_rate"))
    if direct is not None:
        return direct
    hits = as_float(row.get("pld_exact_hits_total"))
    misses = as_float(row.get("pld_exact_misses_total"))
    if hits is None or misses is None or hits + misses <= 0:
        return None
    return hits / (hits + misses)


def output_equivalence_summary(
    output_equivalence: dict[str, Any],
    method: str,
    baseline: str,
) -> dict[str, Any]:
    eq = output_equivalence.get(method, {})
    if not isinstance(eq, dict):
        eq = {}
    tasks = as_int(eq.get("tasks"))
    baseline_key = f"matches_{baseline}"
    matches_baseline = as_int(eq.get(baseline_key))
    matches_vanilla = as_int(eq.get("matches_vanilla"))
    return {
        "output_equivalence_tasks": tasks,
        "matches_baseline": matches_baseline,
        "matches_baseline_rate": ratio(matches_baseline, tasks),
        "matches_vanilla": matches_vanilla,
        "matches_vanilla_rate": ratio(matches_vanilla, tasks),
    }


def runtime_label_from_path(path: Path) -> str:
    parts = path.parts
    if "modal" in parts:
        idx = parts.index("modal")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if len(parts) >= 3:
        return parts[-3]
    return path.stem


def summarize_runtime_aggregate(path: Path, payload: dict[str, Any], baseline: str) -> list[dict[str, Any]]:
    by_method = payload.get("by_method")
    if not isinstance(by_method, dict):
        return summarize_run_summary(path, payload)

    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    output_equivalence = payload.get("output_equivalence")
    if not isinstance(output_equivalence, dict):
        output_equivalence = {}
    baseline_row = by_method.get(baseline)
    baseline_tps = (
        as_float(baseline_row.get("tokens_per_sec"))
        if isinstance(baseline_row, dict)
        else None
    )

    rows: list[dict[str, Any]] = []
    for method, raw_row in sorted(by_method.items()):
        if not isinstance(raw_row, dict):
            continue
        tps = as_float(raw_row.get("tokens_per_sec"))
        trigger_count = as_int(raw_row.get("mtp_trigger_count")) or as_int(
            raw_row.get("pld_variant_triggers_total")
        )
        mtp_used_count = as_int(raw_row.get("mtp_used_count"))
        row: dict[str, Any] = {
            "measurement_type": "runtime measured",
            "source_path": stable_relpath(path),
            "run_label": runtime_label_from_path(path),
            "method": method,
            "baseline_method": baseline,
            "is_baseline_method": method == baseline,
            "n_tasks": as_int(meta.get("n_problems")),
            "n_steps": as_int(raw_row.get("n_steps")),
            "emitted_tokens": as_int(raw_row.get("n_emitted_total"))
            or as_int(raw_row.get("n_new_tokens_total")),
            "tokens_per_sec": tps,
            "speedup_vs_baseline": ratio(tps, baseline_tps),
            "wall_us_total": as_float(raw_row.get("wall_us_total")),
            "decode_wall_us_total": as_float(raw_row.get("decode_wall_us_total")),
            "verify_us_total": as_float(raw_row.get("verify_us_total")),
            "proposal_us_total": as_float(raw_row.get("proposal_us_total")),
            "mean_accepted_nonroot_drafts_per_step": as_float(
                raw_row.get("mean_accepted_nonroot_drafts_per_step")
            ),
            "nonroot_acceptance_rate": as_float(raw_row.get("nonroot_acceptance_rate")),
            "pld_exact_hits_total": as_int(raw_row.get("pld_exact_hits_total")),
            "pld_exact_misses_total": as_int(raw_row.get("pld_exact_misses_total")),
            "pld_exact_hit_rate": pld_exact_hit_rate(raw_row),
            "mtp_trigger_count": trigger_count,
            "mtp_trigger_rate": ratio(trigger_count, raw_row.get("n_steps")),
            "mtp_used_count": mtp_used_count,
            "mtp_used_rate": ratio(mtp_used_count, raw_row.get("n_steps")),
            "mtp_actual_extra_progress_sum": as_int(raw_row.get("mtp_actual_extra_progress_sum")),
            "mtp_extra_accepted_drafts_sum": as_int(raw_row.get("mtp_extra_accepted_drafts_sum")),
            "mtp_decode_steps_saved_estimate": as_int(raw_row.get("mtp_decode_steps_saved_estimate")),
            "mtp_avg_extra_tokens_per_trigger": as_float(
                raw_row.get("mtp_avg_extra_tokens_per_trigger")
            ),
            "mtp_avg_extra_accepted_drafts_per_trigger": as_float(
                raw_row.get("mtp_avg_extra_accepted_drafts_per_trigger")
            ),
            "mtp_head_compute_us_total": as_float(raw_row.get("mtp_head_compute_us_total")),
            "mtp_verify_extra_us_total": as_float(raw_row.get("mtp_verify_extra_us_total")),
            "mtp_total_overhead_us_total": as_float(raw_row.get("mtp_total_overhead_us_total")),
            "mtp_total_overhead_us_per_trigger": as_float(
                raw_row.get("mtp_total_overhead_us_per_trigger")
            ),
            "mtp_token0_reject_rate": as_float(raw_row.get("mtp_token0_reject_rate")),
            "mtp_used_token0_reject_rate": as_float(raw_row.get("mtp_used_token0_reject_rate")),
            "queue_predictions_created": as_int(raw_row.get("mtp_queue_predictions_created")),
            "queue_predictions_used": as_int(raw_row.get("mtp_queue_predictions_used")),
            "queue_predictions_dropped_pld_strong": as_int(
                raw_row.get("mtp_queue_predictions_dropped_pld_strong")
            ),
            "rerank_trigger_rate": as_float(raw_row.get("pld_rerank_trigger_rate")),
            "rerank_accepted_len_mean": as_float(raw_row.get("pld_rerank_accepted_len_mean")),
        }
        row.update(output_equivalence_summary(output_equivalence, method, baseline))
        rows.append(row)
    return rows


def summarize_run_summary(path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    method = str(payload.get("method") or path.parent.name)
    return [
        {
            "measurement_type": "runtime measured",
            "source_path": stable_relpath(path),
            "run_label": path.parent.parent.name if path.parent.parent != ROOT else path.parent.name,
            "method": method,
            "status": payload.get("status"),
            "n_tasks": as_int(payload.get("num_tasks")),
            "emitted_tokens": as_int(payload.get("total_emitted_tokens")),
            "tokens_per_sec": as_float(payload.get("tok_per_s_excluding_init")),
            "tokens_per_sec_including_init": as_float(payload.get("tok_per_s_including_init")),
            "wall_us_total": (
                as_float(payload.get("generation_wall_seconds")) * 1_000_000.0
                if as_float(payload.get("generation_wall_seconds")) is not None
                else None
            ),
            "speculative_config": payload.get("speculative_config"),
        }
    ]


def projection_label_from_path(path: Path) -> str:
    if len(path.parts) >= 2:
        return path.parent.name
    return path.stem


def summarize_projection_report(path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    policies = payload.get("policies")
    if not isinstance(policies, list):
        return []
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    best_policy = payload.get("best_policy")
    rows: list[dict[str, Any]] = []
    for policy in policies:
        if not isinstance(policy, dict):
            continue
        trigger_policy = str(policy.get("trigger_policy") or "unknown")
        rows.append(
            {
                "measurement_type": "offline projection",
                "source_path": stable_relpath(path),
                "run_label": projection_label_from_path(path),
                "method": str(payload.get("method") or ""),
                "mode": payload.get("mode"),
                "mtp_position": policy.get("mtp_position") or payload.get("mtp_position"),
                "trigger_policy": trigger_policy,
                "is_best_policy": trigger_policy == best_policy,
                "baseline_steps": as_int(policy.get("baseline_steps")),
                "projected_steps": as_int(policy.get("projected_steps")),
                "step_reduction_pct": as_float(policy.get("step_reduction_pct")),
                "corrected_projected_speedup": as_float(
                    policy.get("corrected_projected_speedup")
                ),
                "trigger_count": as_int(policy.get("trigger_count")),
                "missing_predictions": as_int(policy.get("missing_predictions")),
                "skipped_baseline_steps": as_int(policy.get("skipped_baseline_steps")),
                "avg_extra_accepted_mtp_tokens_per_trigger": as_float(
                    policy.get("avg_extra_accepted_mtp_tokens_per_trigger")
                ),
                "avg_mtp_accepted_prefix": as_float(policy.get("avg_mtp_accepted_prefix")),
                "token0_rejection_rate_pct": as_float(policy.get("token0_rejection_rate_pct")),
                "accepted_by_horizon_pct": policy.get("accepted_by_horizon_pct"),
                "accepted_prefix_len_distribution": policy.get(
                    "accepted_prefix_len_distribution"
                ),
                "num_predictions": as_int(metadata.get("num_predictions")),
                "mean_mtp_accepted_prefix": as_float(
                    metadata.get("mean_mtp_accepted_prefix")
                ),
                "top1_accuracy_by_horizon_pct": metadata.get(
                    "top1_accuracy_by_horizon_pct"
                ),
                "recommendation": payload.get("recommendation"),
            }
        )
    return rows


def build_analysis(
    *,
    runtime_paths: list[Path],
    projection_paths: list[Path],
    baseline: str,
) -> dict[str, Any]:
    runtime_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for path in runtime_paths:
        if not path.exists():
            warnings.append(f"runtime JSON not found: {path}")
            continue
        payload = load_json(path)
        runtime_rows.extend(summarize_runtime_aggregate(path, payload, baseline))

    for path in projection_paths:
        if not path.exists():
            warnings.append(f"projection JSON not found: {path}")
            continue
        payload = load_json(path)
        projection_rows.extend(summarize_projection_report(path, payload))

    limitations = [
        "Offline projection rows are not runtime measurements and should not be reported as measured speedups.",
        "Runtime measured rows are summarized only from fields present in aggregate/run JSON files.",
        "Output-equivalence rates are reported when aggregate JSON includes output_equivalence; otherwise they are left unset.",
        "Acceptance/proposer details are reported only when counters are present in the input artifacts.",
    ]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "baseline_method": baseline,
        "inputs": {
            "runtime_json": [stable_relpath(path) for path in runtime_paths],
            "projection_json": [stable_relpath(path) for path in projection_paths],
        },
        "runtime_measured": runtime_rows,
        "offline_projection": projection_rows,
        "warnings": warnings,
        "limitations": limitations,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-json",
        action="append",
        default=[],
        type=Path,
        help="Runtime aggregate/run JSON to summarize. Repeatable.",
    )
    parser.add_argument(
        "--projection-json",
        action="append",
        default=[],
        type=Path,
        help="Offline post-PLD projection report JSON to summarize. Repeatable.",
    )
    parser.add_argument(
        "--no-discover",
        action="store_true",
        help="Do not add default discovered post-PLD projection/runtime JSONs.",
    )
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runtime_paths = [resolve_path(path) for path in args.run_json]
    projection_paths = [resolve_path(path) for path in args.projection_json]
    if not args.no_discover:
        runtime_paths.extend(discover_paths(DEFAULT_RUNTIME_GLOBS))
        projection_paths.extend(discover_paths(DEFAULT_PROJECTION_GLOBS))
    runtime_paths = sorted({path.resolve() for path in runtime_paths})
    projection_paths = sorted({path.resolve() for path in projection_paths})

    analysis = build_analysis(
        runtime_paths=runtime_paths,
        projection_paths=projection_paths,
        baseline=args.baseline,
    )
    output_path = resolve_path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {stable_relpath(output_path)}")
    print(
        f"runtime rows: {len(analysis['runtime_measured'])}; "
        f"offline projection rows: {len(analysis['offline_projection'])}"
    )
    if analysis["warnings"]:
        print(f"warnings: {len(analysis['warnings'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
