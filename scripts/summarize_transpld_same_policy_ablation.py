#!/usr/bin/env python3
"""Summarize same-match-policy PLD/TransPLD ablation runs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("artifacts/vantage_transpld/modal/same_policy_20260516_v1")
DEFAULT_OUTPUT_DIR = Path("artifacts/vantage_transpld/tables/same_policy_20260516_v1")

METHODS = [
    ("blazedit_pld_m10_w128_n10", "PLD-n10", "identity", "exact n=10"),
    ("blazedit_pld_m4_w128_n10", "PLD-m4-n10", "identity", "min 4 / max 10"),
    ("vantage_fast_transpld_m10_w128_n10", "TransPLD-n10", "transformed", "exact n=10"),
    ("vantage_fast_transpld_m4_w128_n10", "TransPLD-m4-n10", "transformed", "min 4 / max 10"),
    (
        "vantage_compete_transpld_m4_exactm4_margin0_w128_n10",
        "SafeRoute same-policy",
        "mixed",
        "min 4 / max 10",
    ),
    ("vantage_frozen_transpld", "Frozen SafeRoute", "mixed", "current"),
]
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 123


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _workload(run_tag: str) -> str:
    lowered = run_tag.lower()
    if "field" in lowered:
        return "Field substitution"
    if "style" in lowered:
        return "Identifier-style substitution"
    if "mixed" in lowered:
        return "Mixed"
    if "zero" in lowered:
        return "Zero drift"
    return run_tag


def _method_text(row: dict[str, Any], method: str) -> str:
    return str(((row.get("outputs") or {}).get(method) or {}).get("text") or "")


def _parity(completions: list[dict[str, Any]], method: str) -> str:
    total = 0
    matches = 0
    for row in completions:
        outputs = row.get("outputs") or {}
        if "vanilla" not in outputs or method not in outputs:
            continue
        total += 1
        matches += int(_method_text(row, "vanilla") == _method_text(row, method))
    return f"{matches}/{total}" if total else "unavailable"


def _paired_speedup_ci(
    completions: list[dict[str, Any]],
    numerator: str,
    denominator: str,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[float] | None:
    pairs: list[tuple[float, float, float, float]] = []
    for row in completions:
        outputs = row.get("outputs") or {}
        num = outputs.get(numerator) or {}
        den = outputs.get(denominator) or {}
        try:
            num_tokens = float(num["n_new_tokens"])
            num_wall = float(num["wall_us"])
            den_tokens = float(den["n_new_tokens"])
            den_wall = float(den["wall_us"])
        except (KeyError, TypeError, ValueError):
            continue
        if num_tokens <= 0 or den_tokens <= 0 or num_wall <= 0 or den_wall <= 0:
            continue
        pairs.append((num_tokens, num_wall, den_tokens, den_wall))
    if not pairs:
        return None

    rng = random.Random(seed)
    n = len(pairs)
    values: list[float] = []
    for _ in range(n_resamples):
        num_tokens_total = 0.0
        num_wall_total = 0.0
        den_tokens_total = 0.0
        den_wall_total = 0.0
        for _ in range(n):
            num_tokens, num_wall, den_tokens, den_wall = pairs[rng.randrange(n)]
            num_tokens_total += num_tokens
            num_wall_total += num_wall
            den_tokens_total += den_tokens
            den_wall_total += den_wall
        if num_wall_total > 0 and den_wall_total > 0 and den_tokens_total > 0:
            values.append((num_tokens_total / num_wall_total) / (den_tokens_total / den_wall_total))
    if not values:
        return None
    values.sort()
    lo = values[int(0.025 * (len(values) - 1))]
    hi = values[int(0.975 * (len(values) - 1))]
    return [lo, hi]


def summarize_run(aggregate_path: Path) -> dict[str, Any]:
    aggregate = _load_json(aggregate_path)
    completions_path = aggregate_path.parent / "completions.jsonl"
    completions = _load_jsonl(completions_path) if completions_path.exists() else []
    by_method = aggregate.get("by_method") or {}
    tuned_pld = float((by_method.get("blazedit_pld_w128_n10") or {}).get("tokens_per_sec") or 0.0)
    pld_m4 = float((by_method.get("blazedit_pld_m4_w128_n10") or {}).get("tokens_per_sec") or 0.0)
    rows: list[dict[str, Any]] = []
    for method, label, view, match_policy in METHODS:
        data = by_method.get(method) or {}
        tok_s = data.get("tokens_per_sec")
        rows.append(
            {
                "method": method,
                "label": label,
                "view": view,
                "match_policy": match_policy,
                "tokens_per_sec": tok_s,
                "ratio_vs_tuned_pld": (float(tok_s) / tuned_pld if tok_s and tuned_pld else None),
                "ratio_vs_pld_m4": (float(tok_s) / pld_m4 if tok_s and pld_m4 else None),
                "ci95_vs_tuned_pld": _paired_speedup_ci(
                    completions, method, "blazedit_pld_w128_n10"
                ),
                "ci95_vs_pld_m4": _paired_speedup_ci(
                    completions, method, "blazedit_pld_m4_w128_n10"
                ),
                "n_steps": data.get("n_steps"),
                "n_emitted_total": data.get("n_emitted_total"),
                "parity_vs_vanilla": _parity(completions, method),
            }
        )
    return {
        "run_tag": aggregate_path.parent.parent.name,
        "workload": _workload(aggregate_path.parent.parent.name),
        "aggregate_path": str(aggregate_path),
        "meta": aggregate.get("meta") or {},
        "baseline_tuned_pld_tok_s": tuned_pld or None,
        "baseline_tuned_pld_parity": _parity(completions, "blazedit_pld_w128_n10"),
        "baseline_pld_m4_tok_s": pld_m4 or None,
        "rows": rows,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _fmt_ci(value: Any) -> str:
    if not value:
        return "unavailable"
    return f"[{float(value[0]):.3f},{float(value[1]):.3f}]"


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Same-Policy PLD / TransPLD Ablation",
        "",
        "This generated table addresses the reviewer concern that TransPLD might be benefiting from a different min/max lookup policy rather than the transformed view. All ratios below are computed within this same-policy artifact, using the tuned PLD and identity PLD-m4 denominators displayed in the same row; they are not recomputed from the headline Table 1 timing run.",
        "",
        "## Denominator-Explicit Summary",
        "",
        f"Paired task-bootstrap 95% confidence intervals use {BOOTSTRAP_RESAMPLES} resamples with seed {BOOTSTRAP_SEED}. They measure task heterogeneity within this run, not run-to-run GPU timing variance.",
        "",
        "| Workload | Tuned PLD tok/s | Identity PLD-m4 tok/s | TransPLD-m4 tok/s | SafeRoute same-policy tok/s | Frozen SafeRoute tok/s | TransPLD/PLD-m4 | TransPLD/PLD-m4 CI | SafeRoute/PLD-m4 | SafeRoute/PLD-m4 CI | SafeRoute/tuned PLD | SafeRoute/tuned CI | Parity | Reading |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for run in summary["runs"]:
        by_label = {row["label"]: row for row in run["rows"]}
        pld_m4 = by_label.get("PLD-m4-n10", {})
        trans_m4 = by_label.get("TransPLD-m4-n10", {})
        safe = by_label.get("SafeRoute same-policy", {})
        frozen = by_label.get("Frozen SafeRoute", {})
        if run["workload"] == "Zero drift":
            reading = "no-map fallback / identity-view control; not transformed-work evidence"
        elif run["workload"] == "Field substitution":
            reading = "view effect survives; same-policy route remains diagnostic"
        elif run["workload"] == "Identifier-style substitution":
            reading = "transformed view dominates same-policy identity lookup"
        else:
            reading = "mixed structured transformed-view win"
        lines.append(
            "| {workload} | {tuned} | {pld_m4} | {trans_m4} | {safe_tps} | {frozen_tps} | {trans_ratio} | {trans_ci} | {safe_m4} | {safe_m4_ci} | {safe_tuned} | {safe_tuned_ci} | {parity} | {reading} |".format(
                workload=run["workload"],
                tuned=_fmt(run.get("baseline_tuned_pld_tok_s")),
                pld_m4=_fmt(run.get("baseline_pld_m4_tok_s")),
                trans_m4=_fmt(trans_m4.get("tokens_per_sec")),
                safe_tps=_fmt(safe.get("tokens_per_sec")),
                frozen_tps=_fmt(frozen.get("tokens_per_sec")),
                trans_ratio=_fmt(trans_m4.get("ratio_vs_pld_m4")),
                trans_ci=_fmt_ci(trans_m4.get("ci95_vs_pld_m4")),
                safe_m4=_fmt(safe.get("ratio_vs_pld_m4")),
                safe_m4_ci=_fmt_ci(safe.get("ci95_vs_pld_m4")),
                safe_tuned=_fmt(safe.get("ratio_vs_tuned_pld")),
                safe_tuned_ci=_fmt_ci(safe.get("ci95_vs_tuned_pld")),
                parity=f"Tuned {run.get('baseline_tuned_pld_parity', 'unavailable')}; identity {pld_m4.get('parity_vs_vanilla', 'unavailable')}; trans {trans_m4.get('parity_vs_vanilla', 'unavailable')}; SR {safe.get('parity_vs_vanilla', 'unavailable')}",
                reading=reading,
            )
        )
    lines += [
        "",
        "## Full Method Rows",
        "",
        "| Workload | Method | View | Match policy | tok/s | vs tuned PLD | vs PLD-m4 | Steps | Parity vs vanilla | Raw aggregate |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for run in summary["runs"]:
        for row in run["rows"]:
            lines.append(
                "| {workload} | {label} | {view} | {match_policy} | {tok_s} | {tuned} | {m4} | {steps} | {parity} | `{path}` |".format(
                    workload=run["workload"],
                    label=row["label"],
                    view=row["view"],
                    match_policy=row["match_policy"],
                    tok_s=_fmt(row["tokens_per_sec"]),
                    tuned=_fmt(row["ratio_vs_tuned_pld"]),
                    m4=_fmt(row["ratio_vs_pld_m4"]),
                    steps=_fmt(row["n_steps"]),
                    parity=row["parity_vs_vanilla"],
                    path=run["aggregate_path"],
                )
            )
    lines += [
        "",
        "Acceptance rule: transformed-view rows should be interpreted as a clean transformed-view effect only if they beat the same-policy identity-view PLD rows on the structured workloads.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    root = Path(args.root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [summarize_run(path) for path in sorted(root.glob("*/eval/aggregate.json"))]
    runs.sort(key=lambda r: r["workload"])
    summary = {
        "schema": "vantage/transpld_same_policy_ablation/v1",
        "root": str(root),
        "runs": runs,
    }
    (out_dir / "same_policy_ablation.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "same_policy_ablation.md").write_text(markdown(summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
