"""Characterize copy structure in edit-style completion workloads.

The edit-anchor claim depends on a concrete workload property: the target
greedy output often contains long unchanged spans from a pre-edit reference
program in the prompt.  This script measures that property from
``completions.jsonl`` produced by ``run_eagle_eval.py``.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics
from pathlib import Path
from typing import Any

from asts.code_proposers import _apply_word_map, _extract_reference_blocks, _rewrite_pairs


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_tokenizer(name: str):
    if not name:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    except Exception:
        return None


def _fallback_tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def _tokenize(tokenizer, text: str) -> list[int] | list[str]:
    if tokenizer is None:
        return _fallback_tokens(text)
    return tokenizer(text, add_special_tokens=False).input_ids


def _best_reference(prompt: str, tokenizer) -> str:
    blocks = _extract_reference_blocks(prompt)
    if not blocks:
        return ""
    return max(blocks, key=lambda block: len(_tokenize(tokenizer, block)))


def _token_diff_stats(ref_tokens: list[Any], out_tokens: list[Any]) -> dict[str, float | int]:
    matcher = difflib.SequenceMatcher(a=ref_tokens, b=out_tokens, autojunk=False)
    copied = 0
    longest = 0
    distance = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        ref_len = i2 - i1
        out_len = j2 - j1
        if tag == "equal":
            copied += out_len
            longest = max(longest, out_len)
        elif tag == "replace":
            distance += max(ref_len, out_len)
        else:
            distance += ref_len + out_len
    out_len = len(out_tokens)
    return {
        "output_tokens": out_len,
        "reference_tokens": len(ref_tokens),
        "copied_tokens": copied,
        "copied_token_percentage": copied / out_len if out_len else 0.0,
        "edit_distance_tokens": distance,
        "longest_unchanged_span_tokens": longest,
    }


def _changed_hunks(ref: str, out: str) -> int:
    ref_lines = ref.splitlines()
    out_lines = out.splitlines()
    matcher = difflib.SequenceMatcher(a=ref_lines, b=out_lines, autojunk=False)
    return sum(1 for tag, *_ in matcher.get_opcodes() if tag != "equal")


def _deterministic_repo_edit_target(reference: str) -> str:
    lines = reference.rstrip().splitlines()
    out: list[str] = []
    inserted = False
    for line in lines:
        if not inserted and line.lstrip().startswith("return "):
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}# Return the computed result.")
            inserted = True
        out.append(line)
    if not inserted:
        out.append("# Return the computed result.")
    return "\n".join(out) + "\n"


def _deterministic_repo_edit_rename_target(prompt: str, reference: str) -> str:
    return _apply_word_map(reference, _rewrite_pairs(prompt))


def _target_text_for_row(row: dict[str, Any], method: str, reference: str, target_mode: str) -> tuple[str, str]:
    language = str(row.get("language") or "")
    if target_mode not in {"auto", "completion", "deterministic_repo_edit"}:
        raise ValueError(f"unsupported target mode: {target_mode}")
    if target_mode == "deterministic_repo_edit" or (
        target_mode == "auto" and language == "repo_edit_python"
    ):
        return _deterministic_repo_edit_target(reference), "deterministic_repo_edit"
    if target_mode == "auto" and language == "repo_edit_rename_python":
        return (
            _deterministic_repo_edit_rename_target(str(row.get("prompt") or ""), reference),
            "deterministic_repo_edit_rename",
        )
    deterministic_target = str(row.get("deterministic_target") or "")
    if target_mode == "auto" and deterministic_target:
        return deterministic_target, "manifest_deterministic_target"
    outputs = row.get("outputs") or {}
    output = outputs.get(method) or {}
    return str(output.get("raw_text") or output.get("text") or ""), "completion"


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    idx = (len(ordered) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return float(ordered[lo] * (1 - frac) + ordered[hi] * frac)


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0}
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "p25": _percentile(values, 0.25),
        "p75": _percentile(values, 0.75),
    }


def analyze(
    completions: list[dict[str, Any]],
    *,
    method: str,
    tokenizer_name: str,
    target_mode: str,
) -> dict[str, Any]:
    tokenizer = _load_tokenizer(tokenizer_name)
    rows: list[dict[str, Any]] = []
    missing_method = 0
    missing_reference = 0
    for row in completions:
        outputs = row.get("outputs") or {}
        if method not in outputs:
            missing_method += 1
            continue
        reference = str(row.get("reference") or "") or _best_reference(str(row.get("prompt") or ""), tokenizer)
        if not reference:
            missing_reference += 1
            continue
        output_text, target_source = _target_text_for_row(row, method, reference, target_mode)
        ref_tokens = _tokenize(tokenizer, reference)
        out_tokens = _tokenize(tokenizer, output_text)
        stats = _token_diff_stats(ref_tokens, out_tokens)
        stats.update(
            {
                "task_id": row.get("task_id"),
                "language": row.get("language"),
                "prompt_variant": row.get("prompt_variant"),
                "target_source": target_source,
                "changed_hunk_count": _changed_hunks(reference, output_text),
                "output_lines": len(output_text.splitlines()),
                "reference_lines": len(reference.splitlines()),
            }
        )
        rows.append(stats)

    keys = [
        "output_tokens",
        "reference_tokens",
        "copied_token_percentage",
        "edit_distance_tokens",
        "changed_hunk_count",
        "longest_unchanged_span_tokens",
        "output_lines",
    ]
    aggregate = {
        key: _summary([float(r[key]) for r in rows])
        for key in keys
    }
    return {
        "method": method,
        "target_mode": target_mode,
        "tokenizer": tokenizer_name if tokenizer is not None else "regex-fallback",
        "n_rows": len(rows),
        "missing_method": missing_method,
        "missing_reference": missing_reference,
        "aggregate": aggregate,
        "rows": rows,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    agg = report["aggregate"]
    lines = [
        "# Edit Workload Characterization",
        "",
        f"Method: `{report['method']}`",
        f"Rows analyzed: {report['n_rows']}",
        f"Tokenizer: `{report['tokenizer']}`",
        "",
        "| Metric | Mean | Median | P25 | P75 |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "output_tokens": "Output tokens",
        "reference_tokens": "Reference tokens",
        "copied_token_percentage": "Copied-token percentage",
        "edit_distance_tokens": "Token edit distance",
        "changed_hunk_count": "Changed hunk count",
        "longest_unchanged_span_tokens": "Longest unchanged span",
        "output_lines": "Output lines",
    }
    for key, label in labels.items():
        vals = agg[key]
        if key == "copied_token_percentage":
            fmt = lambda x: f"{100.0 * x:.1f}%"
        else:
            fmt = lambda x: f"{x:.2f}"
        lines.append(
            f"| {label} | {fmt(vals['mean'])} | {fmt(vals['median'])} | "
            f"{fmt(vals['p25'])} | {fmt(vals['p75'])} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--completions", required=True)
    p.add_argument("--method", default="vanilla")
    p.add_argument(
        "--target-mode",
        choices=["auto", "completion", "deterministic_repo_edit"],
        default="auto",
        help=(
            "Which target text to characterize. In auto mode, repo_edit_python "
            "uses the deterministic benchmark edit; other workloads use raw_text "
            "from completions when available."
        ),
    )
    p.add_argument("--target-tokenizer", default="Qwen/Qwen2.5-Coder-7B")
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    args = p.parse_args()

    report = analyze(
        _load_jsonl(Path(args.completions)),
        method=args.method,
        tokenizer_name=args.target_tokenizer,
        target_mode=args.target_mode,
    )
    Path(args.output_json).write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, Path(args.output_md))


if __name__ == "__main__":
    main()
