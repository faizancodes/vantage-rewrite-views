#!/usr/bin/env python3
"""Summarize TransPLD backend/dtype parity-isolation runs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("artifacts/vantage_transpld/modal/backend_isolation_20260516_v1")
DEFAULT_OUTPUT_DIR = Path("artifacts/vantage_transpld/tables/backend_isolation_20260516_v1")
METHODS = ["blazedit_pld_w128_n10", "vantage_frozen_transpld"]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _parse_run(run_tag: str) -> tuple[str, str]:
    backend = "unknown"
    workload = run_tag
    match = re.search(r"_((?:fp32|bf16|fp16)_(?:eager|sdpa))_", run_tag)
    if match:
        backend = match.group(1).replace("_", "/")
    for key, label in (
        ("zero100", "Zero drift"),
        ("field100", "Field substitution"),
        ("style100", "Identifier-style substitution"),
        ("mixed100", "Mixed"),
    ):
        if key in run_tag.lower():
            workload = label
            break
    return backend, workload


def _output_text(row: dict[str, Any], method: str) -> str:
    output = ((row.get("outputs") or {}).get(method) or {})
    return str(output.get("raw_text") if output.get("raw_text") is not None else output.get("text") or "")


def _parity(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    total = 0
    matches = 0
    mismatches: list[str] = []
    for row in rows:
        outputs = row.get("outputs") or {}
        if "vanilla" not in outputs or method not in outputs:
            continue
        total += 1
        if _output_text(row, "vanilla") == _output_text(row, method):
            matches += 1
        else:
            mismatches.append(str(row.get("task_id") or ""))
    return {
        "matches": matches,
        "tasks": total,
        "rate": (matches / total if total else None),
        "mismatch_task_ids": mismatches,
    }


def summarize_run(path: Path) -> dict[str, Any]:
    aggregate = _load_json(path)
    completions_path = path.parent / "completions.jsonl"
    completions = _load_jsonl(completions_path) if completions_path.exists() else []
    backend, workload = _parse_run(path.parent.parent.name)
    by_method = aggregate.get("by_method") or {}
    return {
        "run_tag": path.parent.parent.name,
        "backend": backend,
        "workload": workload,
        "aggregate_path": str(path),
        "meta": aggregate.get("meta") or {},
        "methods": {
            method: {
                "tokens_per_sec": (by_method.get(method) or {}).get("tokens_per_sec"),
                "n_steps": (by_method.get(method) or {}).get("n_steps"),
                "parity_vs_vanilla": _parity(completions, method),
            }
            for method in METHODS
        },
    }


def _fmt_parity(p: dict[str, Any]) -> str:
    if not p.get("tasks"):
        return "unavailable"
    return f"{p['matches']}/{p['tasks']}"


def _fmt(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# TransPLD Backend Isolation Audit",
        "",
        "This generated table separates deterministic exactness evidence from optimized timing-path evidence. Parity is computed from stored raw outputs in `completions.jsonl`.",
        "",
        "| Backend | Workload | PLD parity | VANTAGE parity | PLD tok/s | VANTAGE tok/s | Mismatch task ids | Raw aggregate |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for run in summary["runs"]:
        pld = run["methods"]["blazedit_pld_w128_n10"]
        nh = run["methods"]["vantage_frozen_transpld"]
        mismatch_ids = nh["parity_vs_vanilla"].get("mismatch_task_ids") or []
        lines.append(
            "| {backend} | {workload} | {pld_parity} | {nh_parity} | {pld_tps} | {nh_tps} | {mismatches} | `{path}` |".format(
                backend=run["backend"],
                workload=run["workload"],
                pld_parity=_fmt_parity(pld["parity_vs_vanilla"]),
                nh_parity=_fmt_parity(nh["parity_vs_vanilla"]),
                pld_tps=_fmt(pld["tokens_per_sec"]),
                nh_tps=_fmt(nh["tokens_per_sec"]),
                mismatches=", ".join(f"`{x}`" for x in mismatch_ids) or "none",
                path=run["aggregate_path"],
            )
        )
    lines += [
        "",
        "Acceptance rule: a backend can support greedy-equivalence language only if PLD and VANTAGE both match vanilla on every audited workload.",
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
    runs.sort(key=lambda r: (r["backend"], r["workload"]))
    summary = {
        "schema": "vantage/transpld_backend_isolation/v1",
        "root": str(root),
        "runs": runs,
    }
    (out_dir / "backend_isolation.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "backend_isolation.md").write_text(markdown(summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
