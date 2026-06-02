#!/usr/bin/env python3
"""Summarize VANTAGE-vLLM benchmark artifacts.

The script reads one or more run directories produced by
``scripts/run_vllm_benchmarks.py`` and writes a compact JSON/Markdown/LaTeX
summary. It never invents missing numbers: absent fields are rendered as
``not captured``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("artifacts/vllm_results")
DEFAULT_TABLE_DIR = Path("artifacts/vllm_tables")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_run_dirs(paths: list[str]) -> list[Path]:
    if paths:
        return [Path(path) for path in paths]
    return sorted(path.parent for path in DEFAULT_ROOT.glob("*/*/run_summary.json"))


def fmt(value: Any, digits: int = 1) -> str:
    if value is None or value == "":
        return "not captured"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def row_from_dir(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "run_summary.json"
    config_path = run_dir / "config.json"
    summary = load_json(summary_path)
    config = load_json(config_path) if config_path.exists() else {}
    return {
        "run_dir": str(run_dir),
        "run_id": summary.get("run_id"),
        "status": summary.get("status"),
        "method": summary.get("method"),
        "model": summary.get("model"),
        "split": summary.get("split"),
        "num_tasks": summary.get("num_tasks"),
        "tok_per_s_excluding_init": summary.get("tok_per_s_excluding_init"),
        "tok_per_s_including_init": summary.get("tok_per_s_including_init"),
        "total_emitted_tokens": summary.get("total_emitted_tokens"),
        "generation_wall_seconds": summary.get("generation_wall_seconds"),
        "init_seconds": summary.get("init_seconds"),
        "peak_memory_gb_if_available": summary.get("peak_memory_gb_if_available"),
        "vllm_version": summary.get("vllm_version"),
        "speculative_config": summary.get("speculative_config"),
        "sampling_params": summary.get("sampling_params"),
        "failure": summary.get("failure"),
        "command": (run_dir / "modal_summary.json").exists()
        and load_json(run_dir / "modal_summary.json").get("command"),
        "config": config,
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Method | Status | Tasks | Tok/s excl. init | Tok/s incl. init | Emitted tokens | vLLM | Speculative config | Notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        spec = row.get("speculative_config")
        spec_text = "`none`" if spec is None else "`" + json.dumps(spec, sort_keys=True) + "`"
        failure = row.get("failure") or {}
        if row.get("status") != "success":
            message = str(failure.get("message", "")).splitlines()[0]
            notes = f"{failure.get('type', 'failure')}: {message[:180]}"
        else:
            notes = row.get("run_dir", "")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("method", "")),
                    str(row.get("status", "")),
                    fmt(row.get("num_tasks"), 0),
                    fmt(row.get("tok_per_s_excluding_init"), 1),
                    fmt(row.get("tok_per_s_including_init"), 1),
                    fmt(row.get("total_emitted_tokens"), 0),
                    str(row.get("vllm_version") or "not captured"),
                    spec_text,
                    str(notes).replace("\n", " "),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def latex_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{VANTAGE-vLLM benchmark artifacts.}",
        "\\label{tab:vantage-vllm-results}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Method & Status & Tasks & Tok/s & Cold tok/s & Tokens \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{latex_escape(str(row.get('method', '')))} & "
            f"{latex_escape(str(row.get('status', '')))} & "
            f"{fmt(row.get('num_tasks'), 0)} & "
            f"{fmt(row.get('tok_per_s_excluding_init'), 1)} & "
            f"{fmt(row.get('tok_per_s_including_init'), 1)} & "
            f"{fmt(row.get('total_emitted_tokens'), 0)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def build_comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method = {
        str(row.get("method")): row
        for row in rows
        if row.get("status") == "success" and row.get("tok_per_s_excluding_init")
    }

    def ratio(numerator: str, denominator: str) -> float | None:
        left = by_method.get(numerator)
        right = by_method.get(denominator)
        if not left or not right:
            return None
        denom = right.get("tok_per_s_excluding_init")
        if not denom:
            return None
        return float(left["tok_per_s_excluding_init"]) / float(denom)

    return {
        "vantage_prompt_only_vs_ngram_speedup": ratio("vantage_prompt_only", "ngram"),
        "vantage_prompt_only_vs_greedy_speedup": ratio("vantage_prompt_only", "greedy"),
        "ngram_vs_greedy_speedup": ratio("ngram", "greedy"),
    }


def write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    payload = {
        "schema": "vantage_vllm_results_manifest_v1",
        "runs": rows,
        "comparisons": build_comparisons(rows),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="*", help="Run directories; defaults to artifacts/vllm_results/*/*")
    parser.add_argument("--output-dir", default=str(DEFAULT_TABLE_DIR))
    parser.add_argument("--manifest", default="artifacts/vllm_results/manifest.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dirs = discover_run_dirs(args.run_dirs)
    rows = [row_from_dir(path) for path in run_dirs if (path / "run_summary.json").exists()]
    rows.sort(key=lambda row: (str(row.get("run_id")), str(row.get("method"))))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vantage_vllm_results.md").write_text(markdown_table(rows), encoding="utf-8")
    (out_dir / "vantage_vllm_results.tex").write_text(latex_table(rows), encoding="utf-8")
    write_manifest(rows, Path(args.manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
