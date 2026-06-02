#!/usr/bin/env python3
"""Evaluate or generate diff/hunk-only edit artifacts.

Inputs may be a simple JSONL dataset with ``source``/``completion`` fields or a
real-commit style completions JSONL with ``reference`` plus ``outputs``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asts.diff_hunk_generation import evaluate_completion


PROMPT_FORMATS = ("unified_diff", "json_replacements", "search_replace")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def source_text(row: dict[str, Any]) -> str | None:
    for key in ("source", "reference", "original", "before", "input"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return None


def expected_text(row: dict[str, Any]) -> str | None:
    for key in ("expected", "deterministic_target", "target", "after"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return None


def row_format(row: dict[str, Any], default: str | None) -> str | None:
    for key in ("patch_format", "format", "completion_format"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return default


def output_text(row: dict[str, Any], method: str | None) -> str | None:
    outputs = row.get("outputs")
    if isinstance(outputs, dict) and method:
        item = outputs.get(method)
        if isinstance(item, dict):
            for key in ("text", "raw_text", "completion", "patch", "diff"):
                value = item.get(key)
                if isinstance(value, str):
                    return value
        if isinstance(item, str):
            return item
    for key in ("completion", "patch", "diff", "edit", "output"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return None


def infer_methods(rows: list[dict[str, Any]], requested: list[str]) -> list[str | None]:
    if requested:
        return requested
    methods: set[str] = set()
    for row in rows:
        outputs = row.get("outputs")
        if isinstance(outputs, dict):
            methods.update(str(k) for k in outputs.keys())
    return sorted(methods) if methods else [None]


def evaluate_rows(
    rows: list[dict[str, Any]],
    *,
    methods: list[str | None],
    patch_format: str | None,
) -> dict[str, Any]:
    groups = []
    for method in methods:
        task_rows = []
        failures: Counter[str] = Counter()
        for idx, row in enumerate(rows):
            src = source_text(row)
            completion = output_text(row, method)
            expected = expected_text(row)
            task = {
                "index": idx,
                "task_id": row.get("task_id", idx),
                "method": method or "completion",
                "has_source": isinstance(src, str),
                "has_completion": isinstance(completion, str),
                "has_expected": isinstance(expected, str),
            }
            if src is None:
                task.update(_failure("missing_source", "row has no source/reference text"))
            elif completion is None:
                task.update(_failure("missing_completion", "row has no completion/patch text"))
            else:
                task.update(
                    evaluate_completion(
                        src,
                        completion,
                        expected=expected,
                        patch_format=row_format(row, patch_format),
                    )
                )
            if task.get("failure_code"):
                failures[str(task["failure_code"])] += 1
            task_rows.append(task)
        _attach_timing_to_task_rows(task_rows, rows)
        groups.append(summarize_group(method or "completion", task_rows, failures))
    return {"schema": "asts-spec/diff-hunk-generation-eval/v1", "groups": groups}


def build_generation_prompt(row: dict[str, Any], patch_format: str) -> str:
    src = source_text(row)
    if src is None:
        raise ValueError("row has no source/reference text")
    context = _prompt_context(row)
    language = str(row.get("language") or row.get("metadata", {}).get("language") or "python")
    if patch_format == "unified_diff":
        instruction = (
            "You are editing the given source file.\n"
            "Output only a unified diff patch.\n"
            "Do not repeat unchanged file contents.\n"
            "The patch must apply cleanly.\n"
        )
    elif patch_format == "json_replacements":
        instruction = (
            "You are editing the given source file.\n"
            "Output only valid JSON with this exact schema:\n"
            "{\n"
            '  "replacements": [\n'
            "    {\n"
            '      "start_anchor": "...",\n'
            '      "end_anchor": "...",\n'
            '      "replacement": "..."\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Use anchors that uniquely identify the text to replace. Do not repeat unchanged file contents.\n"
        )
    elif patch_format == "search_replace":
        instruction = (
            "You are editing the given source file.\n"
            "Output only changed hunks in this exact format:\n"
            "<<<<<<< SEARCH\n"
            "old text\n"
            "=======\n"
            "new text\n"
            ">>>>>>> REPLACE\n"
            "Each SEARCH block must occur exactly once in the source. Do not repeat unchanged file contents outside hunks.\n"
        )
    else:
        raise ValueError(f"unsupported prompt format: {patch_format}")
    return f"{context}{instruction}\nSource file:\n```{language}\n{src}\n```\n"


def generate_diff_hunk_outputs(
    rows: list[dict[str, Any]],
    *,
    patch_formats: list[str],
    target: str,
    dtype: str,
    attn: str,
    device: str,
    max_tasks: int,
    max_new_tokens: int,
    baseline_method: str,
) -> list[dict[str, Any]]:
    import torch

    from scripts.benchmark_real_shape_forward import model_dtype_arg
    from scripts.run_eagle_eval import _load_model

    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but CUDA is unavailable")
    tokenizer, model = _load_model(target, dtype=model_dtype_arg(dtype), attn_impl=attn)
    model.eval()
    torch_device = torch.device(device)
    selected = rows[:max_tasks] if max_tasks else rows
    generated: list[dict[str, Any]] = []
    with torch.inference_mode():
        for idx, row in enumerate(selected):
            out_row = dict(row)
            out_row.setdefault("outputs", {})
            for fmt in patch_formats:
                prompt = build_generation_prompt(row, fmt)
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(torch_device) for k, v in inputs.items()}
                if torch_device.type == "cuda":
                    torch.cuda.synchronize(torch_device)
                t0 = time.perf_counter_ns()
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                if torch_device.type == "cuda":
                    torch.cuda.synchronize(torch_device)
                wall_us = (time.perf_counter_ns() - t0) / 1000.0
                new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
                text = tokenizer.decode(new_ids, skip_special_tokens=True)
                method = f"diff_hunk_{fmt}"
                out_row["outputs"][method] = {
                    "text": text,
                    "raw_text": text,
                    "n_new_tokens": int(new_ids.numel()),
                    "wall_us": wall_us,
                    "prompt_tokens": int(inputs["input_ids"].shape[1]),
                    "patch_format": fmt,
                }
            baseline = row.get("outputs", {}).get(baseline_method)
            if isinstance(baseline, dict) and "wall_us" in baseline:
                out_row["baseline_wall_us"] = float(baseline["wall_us"])
            generated.append(out_row)
            print(f"[{idx + 1}/{len(selected)}] generated {','.join(patch_formats)}", flush=True)
    return generated


def _failure(code: str, message: str) -> dict[str, Any]:
    return {
        "parse_success": False,
        "apply_success": False,
        "failure_code": code,
        "failure_message": message,
        "patch_format": None,
        "output_length": 0,
        "edit_distance": None,
        "exact_match": None,
    }


def summarize_group(
    method: str,
    rows: list[dict[str, Any]],
    failures: Counter[str],
) -> dict[str, Any]:
    n = len(rows)
    expected_rows = [r for r in rows if r["has_expected"]]
    applied_rows = [r for r in rows if r["apply_success"]]
    exact_rows = [r for r in applied_rows if r.get("exact_match") is not None]
    edit_distances = [
        int(r["edit_distance"]) for r in applied_rows if r.get("edit_distance") is not None
    ]
    output_lengths = [int(r["output_length"]) for r in applied_rows]
    baseline_speeds = []
    token_reductions = []
    for r in rows:
        source_len = r.get("source_token_count")
        output_tokens = r.get("output_token_count")
        if source_len and output_tokens:
            token_reductions.append(float(output_tokens) / max(1.0, float(source_len)))
        if r.get("baseline_wall_ms") and r.get("generation_wall_ms"):
            baseline_speeds.append(float(r["baseline_wall_ms"]) / max(1e-9, float(r["generation_wall_ms"])))
    return {
        "method": method,
        "n": n,
        "parse_success": sum(1 for r in rows if r["parse_success"]),
        "parse_success_rate": _rate(sum(1 for r in rows if r["parse_success"]), n),
        "apply_success": len(applied_rows),
        "apply_success_rate": _rate(len(applied_rows), n),
        "expected_rows": len(expected_rows),
        "exact_match": sum(1 for r in exact_rows if r["exact_match"]),
        "exact_match_rate": _rate(sum(1 for r in exact_rows if r["exact_match"]), len(exact_rows)),
        "mean_edit_distance": _mean(edit_distances),
        "mean_output_length": _mean(output_lengths),
        "mean_speedup_vs_full_file_pld": _mean_float(baseline_speeds),
        "p50_speedup_vs_full_file_pld": _percentile(baseline_speeds, 0.50),
        "p90_speedup_vs_full_file_pld": _percentile(baseline_speeds, 0.90),
        "mean_output_token_ratio_vs_source": _mean_float(token_reductions),
        "failure_taxonomy": dict(sorted(failures.items())),
        "tasks": rows,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Diff/Hunk Generation Evaluation",
        "",
        "| Method | n | parse ok | apply ok | expected rows | exact match | "
        "mean edit distance | mean output length | mean speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in report["groups"]:
        lines.append(
            f"| {group['method']} | {group['n']} | {_fmt_rate(group['parse_success_rate'])} | "
            f"{_fmt_rate(group['apply_success_rate'])} | {group['expected_rows']} | "
            f"{_fmt_rate(group['exact_match_rate'])} | {_fmt_num(group['mean_edit_distance'])} | "
            f"{_fmt_num(group['mean_output_length'])} | "
            f"{_fmt_speedup(group.get('mean_speedup_vs_full_file_pld'))} |"
        )
    lines.append("")
    lines.append("## Failure Taxonomy")
    for group in report["groups"]:
        lines.append("")
        lines.append(f"### {group['method']}")
        failures = group["failure_taxonomy"]
        if not failures:
            lines.append("")
            lines.append("No failures.")
            continue
        lines.append("")
        lines.append("| Failure | count |")
        lines.append("|---|---:|")
        for code, count in failures.items():
            lines.append(f"| {code} | {count} |")
    path.write_text("\n".join(lines) + "\n")


def _rate(num: int, denom: int) -> float | None:
    return num / denom if denom else None


def _mean(values: list[int]) -> float | None:
    return sum(values) / len(values) if values else None


def _mean_float(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def _fmt_rate(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


def _fmt_num(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _fmt_speedup(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}x"


def parse_methods(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return out


def _prompt_context(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if not isinstance(prompt, str):
        return ""
    fence = prompt.find("```")
    context = prompt[:fence] if fence >= 0 else prompt
    # Remove the full-file output instruction when it is present, while
    # preserving commit messages and explicit rewrite maps.
    lines = []
    for line in context.splitlines():
        low = line.lower()
        if "output the complete edited" in low or "output the complete" in low:
            continue
        if "rewrite the pre-commit" in low and "function below" in low:
            continue
        lines.append(line)
    text = "\n".join(lines).strip()
    return f"{text}\n\n" if text else ""


def _enrich_generation_rows(rows: list[dict[str, Any]], methods: list[str | None], baseline_method: str) -> None:
    for row in rows:
        source = source_text(row) or ""
        source_tokens = len(source.split())
        baseline = row.get("outputs", {}).get(baseline_method)
        baseline_wall_ms = None
        if isinstance(baseline, dict) and baseline.get("wall_us") is not None:
            baseline_wall_ms = float(baseline["wall_us"]) / 1000.0
        for method in methods:
            if method is None:
                continue
            output = row.get("outputs", {}).get(method)
            if isinstance(output, dict):
                output.setdefault("source_token_count", source_tokens)
                output.setdefault("baseline_wall_ms", baseline_wall_ms)


def _attach_timing_to_task_rows(group_rows: list[dict[str, Any]], input_rows: list[dict[str, Any]]) -> None:
    by_task = {row.get("task_id", idx): row for idx, row in enumerate(input_rows)}
    for task in group_rows:
        row = by_task.get(task["task_id"])
        if not row:
            continue
        output = row.get("outputs", {}).get(task["method"])
        if isinstance(output, dict):
            if output.get("wall_us") is not None:
                task["generation_wall_ms"] = float(output["wall_us"]) / 1000.0
            if output.get("n_new_tokens") is not None:
                task["output_token_count"] = int(output["n_new_tokens"])
            if output.get("source_token_count") is not None:
                task["source_token_count"] = int(output["source_token_count"])
            if output.get("baseline_wall_ms") is not None:
                task["baseline_wall_ms"] = float(output["baseline_wall_ms"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "--jsonl", "--completions", dest="input", required=True)
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="output method name; repeat or comma-separate",
    )
    parser.add_argument(
        "--patch-format",
        default=None,
        help="unified_diff, json_replacements, or search_replace",
    )
    parser.add_argument("--generate-model", action="store_true")
    parser.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt-format", default="all")
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--baseline-method", default="blazedit_pld_w128_n10")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    if args.generate_model:
        if args.prompt_format == "all":
            patch_formats = list(PROMPT_FORMATS)
        else:
            patch_formats = parse_methods([args.prompt_format])
        unknown = [fmt for fmt in patch_formats if fmt not in PROMPT_FORMATS]
        if unknown:
            raise SystemExit(f"unsupported --prompt-format values: {unknown}")
        rows = generate_diff_hunk_outputs(
            rows,
            patch_formats=patch_formats,
            target=args.target,
            dtype=args.dtype,
            attn=args.attn,
            device=args.device,
            max_tasks=args.max_tasks,
            max_new_tokens=args.max_new_tokens,
            baseline_method=args.baseline_method,
        )
        methods = [f"diff_hunk_{fmt}" for fmt in patch_formats]
        patch_format_by_method = {
            f"diff_hunk_{fmt}": fmt for fmt in patch_formats
        }
        for row in rows:
            for method, fmt in patch_format_by_method.items():
                output = row.get("outputs", {}).get(method)
                if isinstance(output, dict):
                    output["source_token_count"] = len((source_text(row) or "").split())
                    baseline = row.get("outputs", {}).get(args.baseline_method)
                    if isinstance(baseline, dict) and baseline.get("wall_us") is not None:
                        output["baseline_wall_ms"] = float(baseline["wall_us"]) / 1000.0
    else:
        methods = infer_methods(rows, parse_methods(args.method))
    report = evaluate_rows(rows, methods=methods, patch_format=args.patch_format)
    report["generation"] = {
        "enabled": bool(args.generate_model),
        "target": args.target if args.generate_model else None,
        "dtype": args.dtype if args.generate_model else None,
        "attn": args.attn if args.generate_model else None,
        "max_tasks": args.max_tasks if args.generate_model else None,
        "max_new_tokens": args.max_new_tokens if args.generate_model else None,
        "baseline_method": args.baseline_method if args.generate_model else None,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.generate_model:
        with (output_dir / "generated_outputs.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
    write_markdown(report, output_dir / "report.md")
    print((output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
