"""Stratify PLD/TransPLD speed by realized edit regime.

The same synthetic instruction can land in different regimes depending on the
target model's greedy output.  This script buckets each task by the output of a
chosen classifier method, then reports per-bucket speed for all methods.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asts.code_proposers import _apply_word_map, _coerce_rewrite_pairs  # noqa: E402


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _nested(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    nested = metadata.get("metadata")
    return nested if isinstance(nested, dict) else {}


def _pairs(row: dict[str, Any]) -> dict[str, str]:
    metadata = row.get("metadata") or {}
    nested = _nested(metadata)
    return _coerce_rewrite_pairs(
        row.get("rewrite_pairs")
        or metadata.get("rewrite_pairs")
        or nested.get("rewrite_pairs")
    )


def _text(row: dict[str, Any], method: str) -> str:
    out = (row.get("outputs") or {}).get(method) or {}
    return str(out.get("text") or out.get("raw_text") or "")


def _ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(a=a, b=b).ratio()


def classify(row: dict[str, Any], method: str) -> str:
    reference = str(row.get("reference") or "")
    output = _text(row, method)
    if not output:
        return "missing_output"
    pairs = _pairs(row)
    transformed = _apply_word_map(reference, pairs) if reference and pairs else reference
    if not reference:
        return "no_reference"
    old_count = sum(output.count(old) for old in pairs)
    new_count = sum(output.count(new) for new in pairs.values())
    ref_ratio = _ratio(output, reference)
    trans_ratio = _ratio(output, transformed)
    if trans_ratio >= 0.92 and trans_ratio >= ref_ratio + 0.03:
        return "transformed_reference_aligned"
    if ref_ratio >= 0.92 and (not pairs or new_count == 0 or old_count >= new_count):
        return "verbatim_or_original_aligned"
    if max(ref_ratio, trans_ratio) < 0.55:
        return "low_copy_or_malformed"
    return "mixed_or_partial"


def analyze(rows: list[dict[str, Any]], *, methods: list[str], classifier_method: str, baseline: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_buckets = []
    for row in rows:
        bucket = classify(row, classifier_method)
        buckets[bucket].append(row)
        task_buckets.append({"task_id": row.get("task_id"), "bucket": bucket})

    report = {
        "schema": "asts-spec/realized-regimes/v1",
        "classifier_method": classifier_method,
        "baseline": baseline,
        "methods": methods,
        "bucket_counts": {key: len(value) for key, value in buckets.items()},
        "buckets": {},
        "task_buckets": task_buckets,
    }
    for bucket, bucket_rows in buckets.items():
        method_stats = {}
        base_tps = 0.0
        for method in methods:
            tokens = 0
            wall_us = 0.0
            for row in bucket_rows:
                out = (row.get("outputs") or {}).get(method)
                if not out:
                    continue
                tokens += int(out.get("n_new_tokens") or 0)
                wall_us += float(out.get("wall_us") or 0.0)
            tps = tokens / (wall_us / 1e6) if wall_us else 0.0
            method_stats[method] = {
                "n_new_tokens": tokens,
                "wall_us": wall_us,
                "tokens_per_sec": tps,
            }
            if method == baseline:
                base_tps = tps
        for stats in method_stats.values():
            stats["speedup_vs_baseline"] = stats["tokens_per_sec"] / base_tps if base_tps else 0.0
        report["buckets"][bucket] = {"n": len(bucket_rows), "methods": method_stats}
    return report


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Realized-Regime Stratification",
        "",
        f"Classifier method: `{report['classifier_method']}`. Baseline: `{report['baseline']}`.",
        "",
    ]
    for bucket, row in sorted(report["buckets"].items(), key=lambda kv: (-kv[1]["n"], kv[0])):
        lines += [
            f"## {bucket} (n={row['n']})",
            "",
            "| Method | tok/s | vs baseline |",
            "|---|---:|---:|",
        ]
        for method, stats in row["methods"].items():
            lines.append(
                f"| `{method}` | {stats['tokens_per_sec']:.2f} | {stats['speedup_vs_baseline']:.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completions", required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--classifier-method", default="vanilla")
    parser.add_argument("--baseline", default="blazedit_pld_w128_n10")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    report = analyze(
        _load_jsonl(Path(args.completions)),
        methods=methods,
        classifier_method=args.classifier_method,
        baseline=args.baseline,
    )
    Path(args.output_json).write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, Path(args.output_md))
    print(Path(args.output_md).read_text())


if __name__ == "__main__":
    main()
