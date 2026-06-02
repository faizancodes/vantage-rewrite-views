#!/usr/bin/env python3
"""Summarize controlled TransPLD validation manifests for paper provenance."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_MANIFESTS = {
    "zero100": Path("data/manifests_frozen_audit/zero_drift100.jsonl"),
    "field100": Path("data/manifests_frozen_audit/field_rename100.jsonl"),
    "identifier_style100": Path("data/manifests_frozen_audit/style_rewrite100.jsonl"),
    "mixed100": Path("data/manifests_frozen_audit/mixed_zero_field_style100.jsonl"),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    values = sorted(xs)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def pair_type(old: str, new: str) -> str:
    if old.startswith(".") or new.startswith(".") or "." in old or "." in new:
        return "dotted_field"
    if old.isidentifier() and new.isidentifier():
        if "_" in old or "_" in new:
            return "identifier_style"
        return "identifier"
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", old) and re.fullmatch(
        r"[0-9]+(?:\.[0-9]+)?", new
    ):
        return "numeric_literal"
    if (old.startswith(("'", '"')) and old.endswith(("'", '"'))) or (
        new.startswith(("'", '"')) and new.endswith(("'", '"'))
    ):
        return "string_literal"
    return "other"


def prompt_template(row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt") or "")
    before_code = prompt.split("```python", 1)[0].strip()
    return " ".join(before_code.split())


def summarize(name: str, path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    pair_counts: list[int] = []
    type_counts: Counter[str] = Counter()
    cells: Counter[str] = Counter()
    map_visible = 0
    no_map = 0
    transformed_differs = 0
    ref_tokens: list[float] = []
    out_tokens: list[float] = []
    edit_distance: list[float] = []
    copied_pct: list[float] = []
    longest_span: list[float] = []
    hunks: list[float] = []

    for row in rows:
        pairs = row.get("rewrite_pairs") or {}
        if not isinstance(pairs, dict):
            pairs = {}
        pair_counts.append(len(pairs))
        if not pairs:
            no_map += 1
        prompt = str(row.get("prompt") or "")
        if pairs and all(str(k) in prompt and str(v) in prompt for k, v in pairs.items()):
            map_visible += 1
        for old, new in pairs.items():
            type_counts[pair_type(str(old), str(new))] += 1
        if row.get("reference") != row.get("deterministic_target"):
            transformed_differs += 1
        cells[str(row.get("requested_cell") or row.get("drift_family") or "unknown")] += 1
        ref_tokens.append(float(row.get("reference_tokens") or 0))
        out_tokens.append(float(row.get("output_tokens") or 0))
        edit_distance.append(float(row.get("edit_distance_tokens") or 0))
        copied_pct.append(100.0 * float(row.get("copied_token_percentage") or 0.0))
        longest_span.append(float(row.get("longest_unchanged_span_tokens") or 0))
        hunks.append(float(row.get("changed_hunk_count") or 0))

    example = rows[0] if rows else {}
    return {
        "workload": name,
        "manifest": str(path),
        "n": len(rows),
        "rows_with_rewrite_pairs": sum(1 for n in pair_counts if n > 0),
        "rows_without_rewrite_pairs": no_map,
        "rows_where_target_differs_from_reference": transformed_differs,
        "prompt_visible_maps": map_visible,
        "mean_rewrite_pairs": mean(pair_counts) if pair_counts else 0.0,
        "max_rewrite_pairs": max(pair_counts) if pair_counts else 0,
        "rewrite_pair_types": dict(type_counts),
        "requested_cells": dict(cells),
        "reference_tokens_mean": mean(ref_tokens) if ref_tokens else 0.0,
        "reference_tokens_p50": percentile(ref_tokens, 0.50),
        "output_tokens_mean": mean(out_tokens) if out_tokens else 0.0,
        "output_tokens_p50": percentile(out_tokens, 0.50),
        "edit_distance_tokens_mean": mean(edit_distance) if edit_distance else 0.0,
        "edit_distance_tokens_p50": percentile(edit_distance, 0.50),
        "copied_token_percent_mean": mean(copied_pct) if copied_pct else 0.0,
        "longest_unchanged_span_tokens_mean": mean(longest_span) if longest_span else 0.0,
        "changed_hunks_mean": mean(hunks) if hunks else 0.0,
        "example_task_id": example.get("task_id"),
        "example_prompt_template": prompt_template(example),
        "example_rewrite_pairs": example.get("rewrite_pairs") or {},
    }


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Controlled TransPLD Manifest Audit",
        "",
        "This table is generated from the locked validation manifests. It describes the controlled mechanism benchmark; it is not a real-commit benchmark.",
        "",
        "| Workload | n | Rewrite rows | No-map rows | Target differs | Prompt-visible maps | Pair types | Mean pairs | Mean ref tok. | Mean out tok. | Mean edit dist. | Mean copied % | Mean longest span |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        pair_types = ", ".join(f"{k}:{v}" for k, v in sorted(row["rewrite_pair_types"].items())) or "none"
        lines.append(
            f"| {row['workload']} | {row['n']} | {row['rows_with_rewrite_pairs']} | "
            f"{row['rows_without_rewrite_pairs']} | {row['rows_where_target_differs_from_reference']} | "
            f"{row['prompt_visible_maps']} | `{pair_types}` | {row['mean_rewrite_pairs']:.2f} | "
            f"{row['reference_tokens_mean']:.1f} | {row['output_tokens_mean']:.1f} | "
            f"{row['edit_distance_tokens_mean']:.1f} | {row['copied_token_percent_mean']:.1f} | "
            f"{row['longest_unchanged_span_tokens_mean']:.1f} |"
        )
    lines += [
        "",
        "## Example Prompt Templates",
        "",
    ]
    for row in rows:
        lines += [
            f"### {row['workload']}",
            "",
            f"- Manifest: `{row['manifest']}`",
            f"- Example task: `{row['example_task_id']}`",
            f"- Example rewrite pairs: `{row['example_rewrite_pairs']}`",
            f"- Prompt header: {row['example_prompt_template']}",
            "",
        ]
    path.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", default="artifacts/vantage_transpld/tables/controlled_manifest_audit.json")
    parser.add_argument("--output-md", default="artifacts/vantage_transpld/tables/controlled_manifest_audit.md")
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="Optional name:path entry. Defaults to the four locked validation manifests.",
    )
    args = parser.parse_args()

    manifests = dict(DEFAULT_MANIFESTS)
    for item in args.manifest:
        if ":" not in item:
            raise SystemExit("--manifest entries must be name:path")
        name, raw_path = item.split(":", 1)
        manifests[name] = Path(raw_path)

    rows = [summarize(name, path) for name, path in manifests.items()]
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    write_markdown(rows, out_md)
    print(out_md.read_text())


if __name__ == "__main__":
    main()
