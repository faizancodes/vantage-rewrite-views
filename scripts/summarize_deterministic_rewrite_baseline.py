#!/usr/bin/env python3
"""Summarize a deterministic rewrite-engine sanity baseline.

This is not a decoder baseline.  It applies the prompt-visible explicit rewrite
map directly to the reference text in the locked controlled manifests and
checks the resulting text against the deterministic target.  The purpose is to
make the paper's scope explicit: TransPLD accelerates fixed-prompt model
decoding; it does not claim that lexical rewrite tasks require an LLM.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
import textwrap
import time
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asts.code_proposers import apply_boundary_rewrites, extract_explicit_rewrites


DEFAULT_OUTPUT_DIR = Path(
    "artifacts/vantage_transpld/tables/deterministic_rewrite_baseline_20260521_v1"
)
DEFAULT_MANIFESTS = {
    "Zero drift": Path("data/manifests_frozen_audit/zero_drift100.jsonl"),
    "Field substitution": Path("data/manifests_frozen_audit/field_rename100.jsonl"),
    "Identifier-style substitution": Path("data/manifests_frozen_audit/style_rewrite100.jsonl"),
    "Mixed": Path("data/manifests_frozen_audit/mixed_zero_field_style100.jsonl"),
}
_IDENT_CHARS = r"A-Za-z0-9_"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _coerce_pairs(row: dict[str, Any]) -> dict[str, str]:
    value = row.get("rewrite_pairs") or (row.get("metadata") or {}).get("rewrite_pairs")
    if isinstance(value, dict):
        return {
            str(k): str(v)
            for k, v in value.items()
            if str(k) and str(v) and str(k) != str(v)
        }
    return extract_explicit_rewrites(str(row.get("prompt") or ""))


def _boundary_pattern(term: str) -> re.Pattern[str]:
    term_s = str(term)
    escaped = re.escape(term_s)
    if term_s.startswith(".") and re.search(r"[A-Za-z0-9_]", term_s):
        return re.compile(rf"{escaped}(?![{_IDENT_CHARS}])")
    if re.search(r"[A-Za-z0-9_]", term_s):
        return re.compile(rf"(?<![{_IDENT_CHARS}]){escaped}(?![{_IDENT_CHARS}])")
    return re.compile(escaped)


def _contains_boundary(text: str, term: str) -> bool:
    return bool(_boundary_pattern(term).search(text))


def _rewrite_compliant(text: str, pairs: dict[str, str]) -> bool | None:
    if not pairs:
        return None
    return all(_contains_boundary(text, new) for new in pairs.values()) and not any(
        _contains_boundary(text, old) for old in pairs.keys()
    )


def _syntax_ok(code: str) -> bool:
    if not code.strip():
        return False
    try:
        ast.parse(textwrap.dedent(code))
        return True
    except SyntaxError:
        return False


def _summarize_manifest(name: str, path: Path) -> dict[str, Any]:
    rows = _load_jsonl(path)
    exact = 0
    syntax = 0
    compliance = 0
    compliance_total = 0
    rewrite_rows = 0
    runtimes_us: list[float] = []
    examples: list[dict[str, Any]] = []
    for row in rows:
        reference = str(row.get("reference") or "")
        target = str(row.get("deterministic_target") or "")
        pairs = _coerce_pairs(row)
        rewrite_rows += int(bool(pairs))
        start_ns = time.perf_counter_ns()
        rewritten = apply_boundary_rewrites(reference, pairs) if pairs else reference
        elapsed_us = (time.perf_counter_ns() - start_ns) / 1000.0
        runtimes_us.append(elapsed_us)
        exact += int(rewritten.strip() == target.strip())
        syntax += int(_syntax_ok(rewritten))
        ok = _rewrite_compliant(rewritten, pairs)
        if ok is not None:
            compliance_total += 1
            compliance += int(ok)
        if len(examples) < 2 and pairs:
            examples.append(
                {
                    "task_id": row.get("task_id"),
                    "pairs": pairs,
                    "exact_target": rewritten.strip() == target.strip(),
                }
            )
    return {
        "workload": name,
        "manifest": str(path),
        "tasks": len(rows),
        "rewrite_rows": rewrite_rows,
        "exact_target": exact,
        "syntax_ok": syntax,
        "rewrite_compliance": compliance,
        "rewrite_compliance_total": compliance_total,
        "mean_runtime_us": statistics.fmean(runtimes_us) if runtimes_us else 0.0,
        "p95_runtime_us": sorted(runtimes_us)[int(0.95 * (len(runtimes_us) - 1))] if runtimes_us else 0.0,
        "examples": examples,
    }


def _fmt_count(num: int, den: int) -> str:
    return f"{num}/{den}" if den else "n/a"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Deterministic Rewrite-Engine Sanity Baseline",
        "",
        "This is a task-solver sanity check, not a speculative-decoding baseline. It applies prompt-visible rewrite pairs directly to the reference text using the same conservative boundary-aware rewrite routine used to build TransPLD's hidden view.",
        "",
        "| Workload | Tasks | Rewrite rows | Exact target | Syntax | Rewrite compliance | Mean runtime us | p95 runtime us |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            "| {workload} | {tasks} | {rewrite_rows} | {exact} | {syntax} | {compliance} | {mean:.2f} | {p95:.2f} |".format(
                workload=row["workload"],
                tasks=row["tasks"],
                rewrite_rows=row["rewrite_rows"],
                exact=_fmt_count(row["exact_target"], row["tasks"]),
                syntax=_fmt_count(row["syntax_ok"], row["tasks"]),
                compliance=_fmt_count(row["rewrite_compliance"], row["rewrite_compliance_total"]),
                mean=row["mean_runtime_us"],
                p95=row["p95_runtime_us"],
            )
        )
    lines += [
        "",
        "Interpretation: deterministic rewriting is the right task-solver sanity check for explicit lexical maps. TransPLD's contribution is fixed-prompt model-output acceleration under target verification, not solving lexical rewrites better than a rewrite engine.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="Optional NAME=PATH manifest spec. Defaults to the four locked controlled manifests.",
    )
    args = parser.parse_args()
    manifest_map = dict(DEFAULT_MANIFESTS)
    for spec in args.manifest:
        if "=" not in spec:
            raise SystemExit(f"invalid --manifest {spec!r}; expected NAME=PATH")
        name, path = spec.split("=", 1)
        manifest_map[name] = Path(path)
    report = {
        "schema": "vantage/transpld_deterministic_rewrite_baseline/v1",
        "rows": [_summarize_manifest(name, path) for name, path in manifest_map.items()],
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "deterministic_rewrite_baseline.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (out_dir / "deterministic_rewrite_baseline.md").write_text(_markdown(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
