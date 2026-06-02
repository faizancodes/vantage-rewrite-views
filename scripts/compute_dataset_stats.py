#!/usr/bin/env python3
"""Compute paper-facing descriptive statistics for real-commit manifests."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "dataset_stats.json"


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _pct(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    frac = k - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _approx_token_count(text: str) -> int:
    return len(text.split())


def _load_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        return None


def _token_count(text: str, tokenizer) -> int:
    if tokenizer is None:
        return _approx_token_count(text)
    return len(tokenizer.encode(text, add_special_tokens=False))


def _field_count(rows: list[dict[str, Any]], field: str) -> int:
    return sum(1 for r in rows if r.get(field) is not None)


def _stats_for(path: Path, tokenizer) -> dict[str, Any]:
    rows = list(_iter_jsonl(path))
    prompt_tokens = [float(_token_count(str(r.get("prompt") or ""), tokenizer)) for r in rows]
    pre_edit_tokens = [float(_token_count(str(r.get("reference") or ""), tokenizer)) for r in rows]
    gold_target_tokens = [
        float(_token_count(str(r.get("deterministic_target") or ""), tokenizer)) for r in rows
    ]
    manifest_output_tokens = [float(r.get("output_tokens") or 0.0) for r in rows]
    edit_distance = [float(r.get("edit_distance_tokens") or 0.0) for r in rows]
    copy_overlap = [float(r.get("copied_token_percentage") or 0.0) for r in rows]
    repos = {str(r.get("repo") or "") for r in rows if r.get("repo")}
    commits = {str(r.get("commit_sha") or "") for r in rows if r.get("commit_sha")}
    regimes: dict[str, int] = {}
    for r in rows:
        regime = str(r.get("benchmark_regime") or "unknown")
        regimes[regime] = regimes.get(regime, 0) + 1
    return {
        "manifest": str(path),
        "tasks": len(rows),
        "unique_repos": len(repos),
        "unique_commits": len(commits),
        "split_values": sorted({str(r.get("split") or "") for r in rows}),
        "benchmark_regimes": regimes,
        "field_availability": {
            "prompt": _field_count(rows, "prompt"),
            "reference": _field_count(rows, "reference"),
            "deterministic_target": _field_count(rows, "deterministic_target"),
            "output_tokens": _field_count(rows, "output_tokens"),
            "copied_token_percentage": _field_count(rows, "copied_token_percentage"),
            "edit_distance_tokens": _field_count(rows, "edit_distance_tokens"),
        },
        "mean_input_tokens": _mean(prompt_tokens),
        "input_tokens_p50": _pct(prompt_tokens, 50),
        "input_tokens_p90": _pct(prompt_tokens, 90),
        "input_tokens_p99": _pct(prompt_tokens, 99),
        "mean_pre_edit_context_tokens": _mean(pre_edit_tokens),
        "pre_edit_context_tokens_p50": _pct(pre_edit_tokens, 50),
        "pre_edit_context_tokens_p90": _pct(pre_edit_tokens, 90),
        "pre_edit_context_tokens_p99": _pct(pre_edit_tokens, 99),
        "input_token_count_source": "Qwen/Qwen2.5-Coder-7B tokenizer"
        if tokenizer is not None
        else "whitespace fallback; tokenizer not available locally",
        "mean_output_tokens": _mean(gold_target_tokens),
        "output_tokens_p50": _pct(gold_target_tokens, 50),
        "output_tokens_p90": _pct(gold_target_tokens, 90),
        "output_tokens_p99": _pct(gold_target_tokens, 99),
        "output_length_definition": "gold target length from deterministic_target, not generated/emitted output length",
        "manifest_output_tokens_mean": _mean(manifest_output_tokens),
        "manifest_output_tokens_definition": "manifest lexical token count of deterministic_target from dataset construction",
        "copy_overlap_mean": _mean(copy_overlap),
        "copy_overlap_p50": _pct(copy_overlap, 50),
        "copy_overlap_p90": _pct(copy_overlap, 90),
        "copy_overlap_definition": "copied_token_percentage = equal lexical tokens in SequenceMatcher(pre_edit_source_context, deterministic_target) divided by lexical target tokens",
        "edit_distance_tokens_mean": _mean(edit_distance),
        "edit_distance_tokens_p50": _pct(edit_distance, 50),
        "edit_distance_tokens_p90": _pct(edit_distance, 90),
        "edit_distance_definition": "edit_distance_tokens = max(pre-edit source lexical token count, target lexical token count) - equal lexical token count",
        "emitted_token_denominator": "unavailable in manifests; see generation_stats for timing total_generated_tokens",
        "notes": "Prompt/pre-edit/gold target token counts use the Qwen2.5-Coder-7B tokenizer when available. Copy overlap and edit distance are manifest lexical edit metadata.",
    }


def _write_markdown(report: dict[str, Any], out: Path) -> None:
    rows = report["splits"]
    lines = [
        "# Dataset Statistics",
        "",
        "Token counts use the Qwen tokenizer when it is available locally. Output lengths here are gold `deterministic_target` lengths, not generated/emitted model output lengths.",
        "",
        "| split/manifest | tasks | repos | commits | mean input toks | input p50/p90/p99 | mean output toks | output p50/p90/p99 | copy overlap mean | edit distance mean |",
        "|---|---:|---:|---:|---:|---|---:|---|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {name} | {tasks} | {unique_repos} | {unique_commits} | {mean_input_tokens:.1f} | "
            "{input_tokens_p50:.0f}/{input_tokens_p90:.0f}/{input_tokens_p99:.0f} | {mean_output_tokens:.1f} | "
            "{output_tokens_p50:.0f}/{output_tokens_p90:.0f}/{output_tokens_p99:.0f} | "
            "{copy_overlap_mean:.3f} | {edit_distance_tokens_mean:.1f} |".format(**r)
        )
    lines += [
        "",
        "Definitions:",
        "- `copy_overlap`: equal lexical tokens in `SequenceMatcher(pre_edit_source_context, deterministic_target)` divided by lexical target tokens.",
        "- `edit_distance_tokens`: `max(pre-edit source lexical tokens, target lexical tokens) - equal lexical token count`.",
        "- Emitted generation tokens are unavailable in manifests and are reported separately in `generation_stats` from timing artifacts.",
        "",
        report["notes"],
    ]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="Manifest path. May be repeated. Defaults to test500 and train500.",
    )
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    tokenizer = _load_tokenizer(args.tokenizer)
    paths = [Path(p) for p in args.manifest] or [
        ROOT / "data" / "real_commits" / "real_commit_manifest_balanced_1000_v2_test500.jsonl",
        ROOT / "data" / "real_commits" / "real_commit_manifest_balanced_1000_v2_train500.jsonl",
    ]
    splits = []
    for path in paths:
        stat = _stats_for(path, tokenizer)
        stat["name"] = path.stem
        splits.append(stat)
    report = {
        "splits": splits,
        "tokenizer": args.tokenizer,
        "tokenizer_loaded": tokenizer is not None,
        "field_definitions": {
            "prompt": "model input prompt, including task text and pre-edit context",
            "reference": "pre-edit source context",
            "deterministic_target": "gold post-edit target output",
            "output_tokens": "manifest lexical token count of deterministic_target; not emitted generation length",
            "copied_token_percentage": "equal lexical token overlap with pre-edit source context divided by lexical target tokens",
            "edit_distance_tokens": "lexical edit distance proxy from pre-edit source context to gold target",
        },
        "notes": "No new benchmark was run by this script. It summarizes existing manifest metadata for the dataset appendix.",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
