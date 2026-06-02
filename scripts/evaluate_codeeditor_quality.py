"""Lightweight quality parity checks for CodeEditorBench-style edit outputs.

This script intentionally separates task quality from decoder correctness.  It
does not replace CodeEditorBench's official scorers, but it gives quick
submission-table checks: exact match to deterministic target when available,
Python syntax validity, and bf16 output equality against vanilla.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _norm(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines()).strip()


_FENCE_RE = re.compile(r"^\s*```(?:[A-Za-z0-9_+-]+)?\s*\n(?P<body>.*?)(?:\n```\s*)?$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    if match:
        return match.group("body").strip()
    return stripped


def _extract_code(text: str) -> tuple[str, str]:
    stripped = text.strip()
    if not stripped:
        return "", "empty"
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        body_lines = lines[1:]
        for idx, line in enumerate(body_lines):
            if line.strip().startswith("```"):
                return "\n".join(body_lines[:idx]).strip(), "first_fenced_code"
        return "\n".join(body_lines).strip(), "first_fenced_code_unclosed"
    return _strip_code_fence(stripped), "unfenced_text"


def _code_norm(text: str) -> str:
    return _norm(_extract_code(text)[0])


def _python_syntax_result(text: str) -> tuple[bool, str | None]:
    code = _extract_code(text)[0]
    if not code.strip():
        return False, "empty code"
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as exc:
        return False, f"{exc.msg} line {exc.lineno}"


def _python_syntax_ok(text: str) -> bool:
    return _python_syntax_result(text)[0]


def _output_text(output: dict[str, Any], source: str) -> str:
    if source == "text":
        return str(output.get("text") or "")
    if source == "raw_text":
        return str(output.get("raw_text") or "")
    return str(output.get("raw_text") or output.get("text") or "")


def _failure_reason(
    *,
    text: str,
    raw_text: str,
    selected_text: str,
    n_new_tokens: int | None,
    exact_target_match: bool,
    python_syntax_ok: bool | None,
    syntax_error: str | None,
) -> str:
    postprocessed_fence_only = text.strip().startswith("```") and "\n" not in text.strip() and raw_text.strip()
    if exact_target_match and python_syntax_ok:
        return "ok_after_raw_fence_extraction" if postprocessed_fence_only else "ok"
    if postprocessed_fence_only and python_syntax_ok:
        return "postprocessed_text_fence_only_target_mismatch"
    if postprocessed_fence_only and python_syntax_ok is False:
        return "postprocessed_text_fence_only_and_extracted_code_invalid"
    if python_syntax_ok is False and n_new_tokens == 256:
        return "max_token_truncation_or_incomplete_code"
    if python_syntax_ok is False and syntax_error:
        return f"syntax_error:{syntax_error}"
    if not exact_target_match and python_syntax_ok:
        return "valid_python_but_not_exact_target"
    if not selected_text.strip():
        return "empty_output"
    return "ok"


def analyze(
    completions: list[dict[str, Any]],
    *,
    methods: list[str],
    text_source: str = "auto",
) -> dict[str, Any]:
    groups = []
    for method in methods:
        rows = []
        failure_counts: dict[str, int] = {}
        examples: dict[str, list[dict[str, Any]]] = {}
        for row in completions:
            output = (row.get("outputs") or {}).get(method)
            if not output:
                continue
            text = _output_text(output, text_source)
            code, extraction = _extract_code(text)
            syntax_ok, syntax_error = _python_syntax_result(text)
            target = str(row.get("deterministic_target") or "")
            vanilla = (row.get("outputs") or {}).get("vanilla") or {}
            vanilla_text = _output_text(vanilla, text_source)
            language = str(row.get("language") or "")
            is_python = language in {
                "python",
                "repo_edit_python",
                "repo_edit_rename_python",
                "real_commit_python",
                "codeeditor_python",
                "codeeditor_switch_python",
            }
            exact_target_match = bool(target.strip()) and _code_norm(text) == _code_norm(target)
            python_syntax_ok = syntax_ok if is_python else None
            reason = _failure_reason(
                text=str(output.get("text") or ""),
                raw_text=str(output.get("raw_text") or ""),
                selected_text=text,
                n_new_tokens=output.get("n_new_tokens"),
                exact_target_match=exact_target_match,
                python_syntax_ok=python_syntax_ok,
                syntax_error=syntax_error,
            )
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
            examples.setdefault(reason, [])
            if len(examples[reason]) < 3:
                examples[reason].append(
                    {
                        "task_id": row.get("task_id"),
                        "n_new_tokens": output.get("n_new_tokens"),
                        "extraction": extraction,
                        "syntax_error": syntax_error,
                        "text_preview": str(output.get("text") or "")[:160],
                        "raw_text_preview": str(output.get("raw_text") or "")[:300],
                        "extracted_code_preview": code[:300],
                        "target_preview": target[:300],
                    }
                )
            rows.append(
                {
                    "task_id": row.get("task_id"),
                    "has_target": bool(target.strip()),
                    "exact_target_match": exact_target_match,
                    "matches_vanilla": _code_norm(text) == _code_norm(vanilla_text) if vanilla_text else None,
                    "python_syntax_ok": python_syntax_ok,
                    "extraction": extraction,
                    "failure_reason": reason,
                }
            )
        denom_target = sum(1 for r in rows if r["has_target"])
        denom_vanilla = sum(1 for r in rows if r["matches_vanilla"] is not None)
        denom_syntax = sum(1 for r in rows if r["python_syntax_ok"] is not None)
        groups.append(
            {
                "method": method,
                "n": len(rows),
                "target_rows": denom_target,
                "exact_target_match_rate": (
                    sum(1 for r in rows if r["exact_target_match"]) / denom_target
                    if denom_target
                    else None
                ),
                "matches_vanilla_rate": (
                    sum(1 for r in rows if r["matches_vanilla"]) / denom_vanilla
                    if denom_vanilla
                    else None
                ),
                "python_syntax_ok_rate": (
                    sum(1 for r in rows if r["python_syntax_ok"]) / denom_syntax
                    if denom_syntax
                    else None
                ),
                "tasks": rows,
                "failure_counts": failure_counts,
                "examples": examples,
            }
        )
    return {
        "schema": "asts-spec/codeeditor-quality/v2",
        "methods": methods,
        "text_source": text_source,
        "groups": groups,
    }


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# CodeEditorBench Quality Parity",
        "",
        "| Method | n | target rows | exact target match | matches vanilla | Python syntax OK |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["groups"]:
        lines.append(
            f"| {row['method']} | {row['n']} | {row['target_rows']} | "
            f"{_fmt(row['exact_target_match_rate'])} | {_fmt(row['matches_vanilla_rate'])} | "
            f"{_fmt(row['python_syntax_ok_rate'])} |"
        )
    lines.extend(["", "## Failure Reasons", ""])
    for row in report["groups"]:
        lines.append(f"### {row['method']}")
        lines.append("")
        lines.append("| Reason | Count |")
        lines.append("|---|---:|")
        for reason, count in sorted(row["failure_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"| `{reason}` | {count} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--completions", required=True)
    p.add_argument("--methods", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    p.add_argument(
        "--text-source",
        choices=["auto", "raw_text", "text"],
        default="auto",
        help="Which stored output field to score. auto prefers raw_text over post-processed text.",
    )
    args = p.parse_args()
    report = analyze(
        _load_jsonl(Path(args.completions)),
        methods=[m.strip() for m in args.methods.split(",") if m.strip()],
        text_source=args.text_source,
    )
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, Path(args.output_md))


if __name__ == "__main__":
    main()
