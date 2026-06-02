#!/usr/bin/env python3
"""Evaluate generated output equivalence against manifest gold targets.

The script accepts one manifest/gold JSONL file and one or more generated
output JSONL files from vLLM, Hugging Face, or local harnesses. It writes:

  - eval_summary.json
  - eval_per_task.jsonl
"""

from __future__ import annotations

import argparse
import difflib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TEXT_KEYS = (
    "text",
    "raw_text",
    "output_text",
    "completion",
    "generated_text",
    "prediction",
)
GOLD_TEXT_KEYS = (
    "deterministic_target",
    "gold",
    "gold_text",
    "target",
    "target_text",
    "reference",
    "expected",
)
TOKEN_KEYS = (
    "tokens",
    "token_ids",
    "output_token_ids",
    "completion_token_ids",
    "generated_token_ids",
    "gold_token_ids",
    "target_token_ids",
    "deterministic_target_token_ids",
)
FINISH_KEYS = ("finish_reason", "finish", "stop_reason")
SOURCE_PROMPT_KEYS = (
    "prompt",
    "prompt_text",
    "input_prompt",
    "source_prompt",
    "source_text",
    "source",
    "input",
)
MAX_EXACT_CHAR_EDIT_CELLS = 300_000
MAX_EXACT_LINE_EDIT_CELLS = 50_000
TRUNCATION_FINISH_REASONS = {"length", "max_tokens", "max_new_tokens", "truncated"}
STOP_FINISH_REASONS = {"stop", "eos", "eos_token", "end", "finished"}
ACCEPTANCE_COUNTER_KEYS = (
    "num_drafts",
    "num_draft_tokens",
    "num_accepted_tokens",
    "num_accepted_tokens_per_pos",
    "num_rejected_tokens",
    "accepted_tokens",
    "draft_tokens",
    "rejected_tokens",
    "acceptance_rate",
    "draft_acceptance_rate",
    "accepted_length_histogram",
    "acceptance_length_histogram",
    "rejection_histogram",
    "num_fully_accepted_drafts",
    "pld_tokens_accepted",
    "pld_acceptance_rate",
    "pld_accepted_length_histogram",
    "pld_rejected_tokens",
)
PROPOSER_COUNTER_KEYS = (
    "proposer_hits",
    "proposer_misses",
    "proposer_hit_rate",
    "proposal_count",
    "proposal_tokens",
    "proposal_lengths",
    "ngram_hits",
    "ngram_misses",
    "ngram_hit_rate",
    "eligible_queries",
    "nonempty_proposals",
    "tokens_proposed",
    "pld_calls",
    "pld_skipped_empty_sample",
    "pld_skipped_max_model_len",
    "pld_metadata_missing",
    "pld_hits",
    "pld_misses",
    "pld_tokens_proposed",
    "pld_cap",
    "pld_cap_truncations",
    "pld_prompt_hits",
    "pld_generated_hits",
    "pld_last_match_length",
    "pld_last_source_start",
    "pld_match_len_histogram",
    "pld_draft_len_histogram",
)
STATS_CONTAINER_KEYS = (
    "spec_decode_stats",
    "spec_decoding_stats",
    "acceptance_proposer_stats",
    "proposer_stats",
    "pld_stats",
    "vantage_pld_stats",
    "vantage_pld",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSONL: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{lineno}: expected a JSON object")
            rows.append(row)
    return rows


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def task_id(row: dict[str, Any], fallback: str) -> str:
    for key in ("task_id", "id", "problem_id", "sample_id"):
        if row.get(key) is not None:
            return str(row[key])
    return fallback


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if row.get(key) is not None:
            return row[key]
    return None


def as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def as_token_ids(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    out: list[int] = []
    for item in value:
        if isinstance(item, bool):
            return None
        if not isinstance(item, int):
            return None
        out.append(int(item))
    return out


def token_ids_from(row: dict[str, Any]) -> list[int] | None:
    return as_token_ids(first_present(row, TOKEN_KEYS))


def finish_from(row: dict[str, Any]) -> str | None:
    value = first_present(row, FINISH_KEYS)
    if value is None:
        return None
    return str(value)


def source_prompt_from(row: dict[str, Any]) -> str:
    value = first_present(row, SOURCE_PROMPT_KEYS)
    return as_text(value)


def emitted_token_count(row: dict[str, Any], tokens: list[int] | None, text: str) -> int:
    if tokens is not None:
        return len(tokens)
    for key in ("new_tokens", "n_new_tokens", "num_tokens", "generated_tokens"):
        value = row.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
    return len(text.split())


def edit_distance(a: list[Any], b: list[Any]) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if ca == cb else 1),
                )
            )
        previous = current
    return previous[-1]


def normalized_char_edit_distance(output: str, gold: str) -> float:
    if output == "" and gold == "":
        return 0.0
    dist = edit_distance(list(output), list(gold))
    return dist / max(1, max(len(output), len(gold)))


def bounded_char_edit_distance(output: str, gold: str) -> int | None:
    """Return exact Levenshtein distance when the DP table is small enough.

    Full character-level Levenshtein over code completions can require millions
    of Python cells per task. For long strings we deliberately return ``None``
    and report a labeled SequenceMatcher distance proxy instead of silently
    spending minutes on an evaluator-side bottleneck.
    """

    if output == gold:
        return 0
    if (len(output) + 1) * (len(gold) + 1) > MAX_EXACT_CHAR_EDIT_CELLS:
        return None
    return edit_distance(list(output), list(gold))


def bounded_line_edit_distance(output: str, gold: str) -> int | None:
    output_lines = output.splitlines()
    gold_lines = gold.splitlines()
    if output_lines == gold_lines:
        return 0
    if (len(output_lines) + 1) * (len(gold_lines) + 1) > MAX_EXACT_LINE_EDIT_CELLS:
        return None
    return edit_distance(output_lines, gold_lines)


def normalized_char_distance_proxy(output: str, gold: str) -> float:
    if output == "" and gold == "":
        return 0.0
    ratio = difflib.SequenceMatcher(None, output, gold, autojunk=False).ratio()
    return 1.0 - ratio


def normalized_line_distance_proxy(output: str, gold: str) -> float:
    output_lines = output.splitlines()
    gold_lines = gold.splitlines()
    if not output_lines and not gold_lines:
        return 0.0
    ratio = difflib.SequenceMatcher(None, output_lines, gold_lines, autojunk=False).ratio()
    return 1.0 - ratio


def source_copy_line_overlap_proxy(output: str, source_prompt: str) -> float | None:
    """Share of nonblank output lines that appear exactly in the source prompt.

    This is a cheap copy-overlap proxy, not a plagiarism or acceptance metric:
    it ignores indentation-normalized variants, tokenization, and line order.
    """

    if not source_prompt:
        return None
    output_lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not output_lines:
        return 0.0
    source_lines = {line.strip() for line in source_prompt.splitlines() if line.strip()}
    if not source_lines:
        return None
    copied = sum(1 for line in output_lines if line in source_lines)
    return copied / len(output_lines)


def first_mismatch_position(a: list[int], b: list[int]) -> int | None:
    for idx, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return idx
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p50": percentile(ordered, 50),
        "p90": percentile(ordered, 90),
        "p95": percentile(ordered, 95),
        "p99": percentile(ordered, 99),
    }


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def load_gold_manifest(path: Path) -> dict[str, dict[str, Any]]:
    gold: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(load_jsonl(path)):
        tid = task_id(row, f"row-{idx}")
        text = as_text(first_present(row, GOLD_TEXT_KEYS))
        tokens = token_ids_from(row)
        finish = finish_from(row)
        gold[tid] = {
            "task_id": tid,
            "text": text,
            "tokens": tokens,
            "finish_reason": finish,
            "source_prompt": source_prompt_from(row),
            "source_row": row,
        }
    return gold


def parse_output_arg(value: str) -> tuple[str | None, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"empty method name in output argument {value!r}")
        return name, Path(path)
    return None, Path(value)


def method_name_from_path(path: Path) -> str:
    parent = path.parent.name
    if path.stem in {"completions", "outputs", "generations"} and parent:
        return parent
    return path.stem


def extract_nested_outputs(
    row: dict[str, Any],
    *,
    default_method: str,
    fallback_task_id: str,
) -> list[tuple[str, str, dict[str, Any]]]:
    tid = task_id(row, fallback_task_id)
    outputs = row.get("outputs")
    if isinstance(outputs, dict):
        extracted: list[tuple[str, str, dict[str, Any]]] = []
        for method, output in outputs.items():
            if isinstance(output, dict):
                output_row = dict(output)
            else:
                output_row = {"text": output}
            extracted.append((str(method), tid, output_row))
        return extracted
    return [(str(row.get("method") or default_method), tid, row)]


def load_outputs(path: Path, method_override: str | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    default_method = method_override or method_name_from_path(path)
    by_method: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for idx, row in enumerate(load_jsonl(path)):
        for method, tid, output_row in extract_nested_outputs(
            row,
            default_method=default_method,
            fallback_task_id=f"{path.name}:{idx}",
        ):
            text = as_text(first_present(output_row, TEXT_KEYS))
            tokens = token_ids_from(output_row)
            by_method[method][tid] = {
                "task_id": tid,
                "method": method,
                "text": text,
                "tokens": tokens,
                "finish_reason": finish_from(output_row),
                "emitted_tokens": emitted_token_count(output_row, tokens, text),
                "source_path": str(path),
                "source_row": output_row,
            }
    return {method: dict(rows) for method, rows in by_method.items()}


def merge_outputs(output_specs: list[tuple[str | None, Path]]) -> dict[str, dict[str, dict[str, Any]]]:
    merged: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for method_override, path in output_specs:
        for method, rows in load_outputs(path, method_override).items():
            merged[method].update(rows)
    return {method: dict(rows) for method, rows in merged.items()}


def parse_method_path_arg(value: str) -> tuple[str | None, Path]:
    return parse_output_arg(value)


def compact_run_metadata(path: Path, row: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "source_path": str(path),
        "status": row.get("status"),
        "method": row.get("method"),
        "run_id": row.get("run_id"),
        "timestamp": row.get("timestamp"),
        "model": row.get("model"),
        "vllm_version": row.get("vllm_version"),
        "num_tasks": row.get("num_tasks"),
        "total_emitted_tokens": row.get("total_emitted_tokens"),
        "generation_wall_seconds": row.get("generation_wall_seconds"),
        "init_seconds": row.get("init_seconds"),
        "tok_per_s_excluding_init": row.get("tok_per_s_excluding_init"),
        "tok_per_s_including_init": row.get("tok_per_s_including_init"),
        "hardware": row.get("hardware"),
        "speculative_config": row.get("speculative_config"),
        "notes": row.get("notes"),
    }
    for key in STATS_CONTAINER_KEYS:
        if key in row:
            metadata[key] = row.get(key)
    return metadata


def load_run_summaries(
    output_specs: list[tuple[str | None, Path]],
    run_summary_specs: list[str],
) -> dict[str, dict[str, Any]]:
    run_summaries: dict[str, dict[str, Any]] = {}
    for method_override, output_path in output_specs:
        summary_path = output_path.parent / "run_summary.json"
        if not summary_path.exists():
            continue
        method = method_override or method_name_from_path(output_path)
        run_summaries[method] = compact_run_metadata(summary_path, load_json(summary_path))

    for value in run_summary_specs:
        method_override, summary_path = parse_method_path_arg(value)
        row = load_json(summary_path)
        method = method_override or str(row.get("method") or summary_path.parent.name)
        run_summaries[method] = compact_run_metadata(summary_path, row)
    return run_summaries


def find_nested_values(data: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if key in keys and value is not None:
                found.setdefault(key, value)
            nested = find_nested_values(value, keys)
            for nested_key, nested_value in nested.items():
                found.setdefault(nested_key, nested_value)
    elif isinstance(data, list):
        for item in data:
            nested = find_nested_values(item, keys)
            for nested_key, nested_value in nested.items():
                found.setdefault(nested_key, nested_value)
    return found


def acceptance_proposer_stats(
    methods: dict[str, dict[str, Any]],
    run_summaries: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for method in sorted(methods):
        run = run_summaries.get(method, {})
        speculative_config = run.get("speculative_config")
        acceptance_fields = find_nested_values(run, ACCEPTANCE_COUNTER_KEYS)
        proposer_fields = find_nested_values(run, PROPOSER_COUNTER_KEYS)

        draft_tokens = first_numeric_field(
            acceptance_fields,
            ("num_draft_tokens", "draft_tokens", "pld_tokens_proposed"),
        )
        if draft_tokens is None:
            draft_tokens = first_numeric_field(
                proposer_fields,
                ("pld_tokens_proposed", "tokens_proposed", "proposal_tokens"),
            )
        accepted_tokens = first_numeric_field(
            acceptance_fields,
            ("num_accepted_tokens", "accepted_tokens", "pld_tokens_accepted"),
        )
        acceptance_rate = first_numeric_field(
            acceptance_fields,
            ("acceptance_rate", "draft_acceptance_rate", "pld_acceptance_rate"),
        )
        if acceptance_rate is None and draft_tokens not in (None, 0) and accepted_tokens is not None:
            acceptance_rate = accepted_tokens / draft_tokens

        if acceptance_fields or proposer_fields:
            status = "available"
            note = "Acceptance/proposer counters were found in run metadata."
        elif not speculative_config:
            status = "not_applicable_no_speculative_decoding"
            note = "Greedy run has no speculative proposer."
        else:
            status = "unavailable_in_artifacts"
            note = (
                "No accepted-token, draft-token, accepted-length, or proposer-hit "
                "counters were captured in these vLLM artifacts."
            )

        rows[method] = {
            "status": status,
            "note": note,
            "speculative_config": speculative_config,
            "num_draft_tokens": draft_tokens,
            "num_accepted_tokens": accepted_tokens,
            "acceptance_rate": acceptance_rate,
            "acceptance_fields": acceptance_fields,
            "proposer_fields": proposer_fields,
        }
    return rows


def first_numeric_field(fields: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = fields.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            return float(value)
    return None


def evaluate(gold: dict[str, dict[str, Any]], outputs: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    method_stats: dict[str, dict[str, Any]] = {}
    per_task: list[dict[str, Any]] = []
    pairwise_exact: dict[str, dict[str, Any]] = {}

    task_ids = sorted(gold)
    methods = sorted(outputs)

    for tid in task_ids:
        gold_row = gold[tid]
        gold_text = str(gold_row["text"])
        gold_tokens = gold_row["tokens"]
        gold_finish = gold_row["finish_reason"]
        source_prompt = str(gold_row.get("source_prompt") or "")
        task_methods: dict[str, dict[str, Any]] = {}
        for method in methods:
            out = outputs[method].get(tid)
            if out is None:
                task_methods[method] = {
                    "present": False,
                    "exact_match": False,
                    "length_mismatch": True,
                    "finish_mismatch": gold_finish is not None,
                }
                continue

            text = str(out["text"])
            tokens = out["tokens"]
            char_dist = bounded_char_edit_distance(text, gold_text)
            line_dist = bounded_line_edit_distance(text, gold_text)
            token_dist = None
            first_mismatch = None
            if tokens is not None and gold_tokens is not None:
                token_dist = edit_distance(tokens, gold_tokens)
                first_mismatch = first_mismatch_position(tokens, gold_tokens)
            gold_len = len(gold_text)
            out_len = len(text)
            token_len = len(tokens) if tokens is not None else None
            gold_token_len = len(gold_tokens) if gold_tokens is not None else None
            finish = out["finish_reason"]
            finish_norm = str(finish).lower() if finish is not None else None
            task_methods[method] = {
                "present": True,
                "text": text,
                "emitted_tokens": out["emitted_tokens"],
                "output_chars": out_len,
                "gold_chars": gold_len,
                "output_gold_char_delta": out_len - gold_len,
                "output_tokens": token_len,
                "gold_tokens": gold_token_len,
                "output_gold_token_delta": (
                    token_len - gold_token_len
                    if token_len is not None and gold_token_len is not None
                    else None
                ),
                "finish_reason": finish,
                "is_stop": finish_norm in STOP_FINISH_REASONS if finish_norm is not None else None,
                "is_truncated": (
                    finish_norm in TRUNCATION_FINISH_REASONS if finish_norm is not None else None
                ),
                "exact_match": text == gold_text,
                "char_edit_distance": char_dist,
                "normalized_char_edit_distance": (
                    char_dist / max(1, max(out_len, gold_len)) if char_dist is not None else None
                ),
                "normalized_char_distance_proxy": normalized_char_distance_proxy(text, gold_text),
                "line_edit_distance": line_dist,
                "normalized_line_edit_distance": (
                    line_dist
                    / max(1, max(len(text.splitlines()), len(gold_text.splitlines())))
                    if line_dist is not None
                    else None
                ),
                "normalized_line_distance_proxy": normalized_line_distance_proxy(text, gold_text),
                "token_edit_distance": token_dist,
                "first_mismatch_position": first_mismatch,
                "source_copy_line_overlap_proxy": source_copy_line_overlap_proxy(text, source_prompt),
                "output_gold_length_ratio": out_len / gold_len if gold_len else (1.0 if out_len == 0 else None),
                "length_mismatch": (
                    token_len != gold_token_len
                    if token_len is not None and gold_token_len is not None
                    else out_len != gold_len
                ),
                "finish_mismatch": (
                    finish != gold_finish if finish is not None and gold_finish is not None else None
                ),
            }
        per_task.append(
            {
                "task_id": tid,
                "gold_chars": len(gold_text),
                "gold_tokens": len(gold_tokens) if gold_tokens is not None else None,
                "gold_finish_reason": gold_finish,
                "methods": task_methods,
            }
        )

    for method in methods:
        rows = [task["methods"][method] for task in per_task if method in task["methods"]]
        present = [row for row in rows if row.get("present")]
        exact = sum(1 for row in present if row.get("exact_match"))
        token_edit_values = [
            float(row["token_edit_distance"])
            for row in present
            if row.get("token_edit_distance") is not None
        ]
        char_edit_values = [
            float(row["normalized_char_edit_distance"])
            for row in present
            if row.get("normalized_char_edit_distance") is not None
        ]
        line_edit_values = [
            float(row["normalized_line_edit_distance"])
            for row in present
            if row.get("normalized_line_edit_distance") is not None
        ]
        line_proxy_values = [float(row["normalized_line_distance_proxy"]) for row in present]
        copy_overlap_values = [
            float(row["source_copy_line_overlap_proxy"])
            for row in present
            if row.get("source_copy_line_overlap_proxy") is not None
        ]
        first_mismatches = [
            int(row["first_mismatch_position"])
            for row in present
            if row.get("first_mismatch_position") is not None
        ]
        finish_counts = Counter(str(row.get("finish_reason")) for row in present if row.get("finish_reason") is not None)
        method_stats[method] = {
            "tasks": len(task_ids),
            "present_tasks": len(present),
            "missing_tasks": len(task_ids) - len(present),
            "emitted_tokens": sum(int(row.get("emitted_tokens") or 0) for row in present),
            "emitted_token_stats": stats([float(row.get("emitted_tokens") or 0) for row in present]),
            "output_char_stats": stats([float(row.get("output_chars") or 0) for row in present]),
            "gold_char_stats": stats([float(row.get("gold_chars") or 0) for row in present]),
            "output_gold_char_delta": stats([float(row.get("output_gold_char_delta") or 0) for row in present]),
            "output_token_stats": stats(
                [float(row["output_tokens"]) for row in present if row.get("output_tokens") is not None]
            ),
            "gold_token_stats": stats(
                [float(row["gold_tokens"]) for row in present if row.get("gold_tokens") is not None]
            ),
            "output_gold_token_delta": stats(
                [
                    float(row["output_gold_token_delta"])
                    for row in present
                    if row.get("output_gold_token_delta") is not None
                ]
            ),
            "finish_reasons": dict(sorted(finish_counts.items())),
            "stop_tasks": sum(1 for row in present if row.get("is_stop") is True),
            "stop_rate": (
                sum(1 for row in present if row.get("is_stop") is True) / len(present) if present else 0.0
            ),
            "truncated_tasks": sum(1 for row in present if row.get("is_truncated") is True),
            "truncation_rate": (
                sum(1 for row in present if row.get("is_truncated") is True) / len(present)
                if present
                else 0.0
            ),
            "exact_matches": exact,
            "exact_match_rate": exact / len(present) if present else 0.0,
            "normalized_char_edit_distance": stats(char_edit_values),
            "exact_char_edit_distance_tasks": len(char_edit_values),
            "normalized_char_distance_proxy": stats(
                [float(row["normalized_char_distance_proxy"]) for row in present]
            ),
            "normalized_line_edit_distance": stats(line_edit_values),
            "exact_line_edit_distance_tasks": len(line_edit_values),
            "normalized_line_distance_proxy": stats(line_proxy_values),
            "token_edit_distance": stats(token_edit_values),
            "source_copy_line_overlap_proxy": stats(copy_overlap_values),
            "source_copy_line_overlap_proxy_tasks": len(copy_overlap_values),
            "output_gold_length_ratio": stats(
                [
                    float(row["output_gold_length_ratio"])
                    for row in present
                    if row.get("output_gold_length_ratio") is not None
                ]
            ),
            "first_mismatch_position": stats([float(v) for v in first_mismatches]),
            "length_mismatches": sum(1 for row in present if row.get("length_mismatch")),
            "finish_mismatches": sum(1 for row in present if row.get("finish_mismatch") is True),
        }

    for i, left in enumerate(methods):
        for right in methods[i + 1 :]:
            compared = 0
            exact = 0
            token_first_mismatches: list[int] = []
            length_mismatches = 0
            finish_mismatches = 0
            for tid in task_ids:
                left_row = outputs[left].get(tid)
                right_row = outputs[right].get(tid)
                if left_row is None or right_row is None:
                    continue
                compared += 1
                if left_row["text"] == right_row["text"]:
                    exact += 1
                left_tokens = left_row["tokens"]
                right_tokens = right_row["tokens"]
                if left_tokens is not None and right_tokens is not None:
                    mismatch = first_mismatch_position(left_tokens, right_tokens)
                    if mismatch is not None:
                        token_first_mismatches.append(mismatch)
                    if len(left_tokens) != len(right_tokens):
                        length_mismatches += 1
                elif len(str(left_row["text"])) != len(str(right_row["text"])):
                    length_mismatches += 1
                left_finish = left_row["finish_reason"]
                right_finish = right_row["finish_reason"]
                if left_finish is not None and right_finish is not None and left_finish != right_finish:
                    finish_mismatches += 1
            key = f"{left}__vs__{right}"
            pairwise_exact[key] = {
                "left": left,
                "right": right,
                "compared_tasks": compared,
                "exact_task_matches": exact,
                "exact_task_match_rate": exact / compared if compared else 0.0,
                "first_mismatch_position": stats([float(v) for v in token_first_mismatches]),
                "length_mismatches": length_mismatches,
                "finish_mismatches": finish_mismatches,
            }

    return {
        "summary": {
            "tasks": len(task_ids),
            "methods": method_stats,
            "pairwise_exact_task_matches": pairwise_exact,
        },
        "per_task": per_task,
    }


def write_reports(result: dict[str, Any], output_dir: Path, report_prefix: str = "") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{report_prefix}eval_summary.json").write_text(
        json.dumps(result["summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / f"{report_prefix}eval_per_task.jsonl").open("w", encoding="utf-8") as handle:
        for row in result["per_task"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_acceptance_json(result: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "acceptance_proposer_stats": result["summary"].get("acceptance_proposer_stats", {}),
        "documentation": {
            "availability": (
                "Acceptance/proposer counters are reported only when present in captured "
                "run artifacts. The evaluator does not infer accepted tokens from output "
                "equivalence or throughput."
            ),
            "needed_instrumentation": [
                "Capture vLLM speculative decoding counters such as num_drafts, num_draft_tokens, num_accepted_tokens, and num_accepted_tokens_per_pos.",
                "Persist accepted-length or rejection histograms per run if the report needs distributional acceptance analysis.",
                "Add proposer-side denominators for n-gram proposals: eligible queries, nonempty proposals, proposal tokens, and proposal lengths.",
                "Run with stats logging or Prometheus export enabled, or write those counters into the benchmark run_summary.json.",
            ],
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "not captured"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


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


def quality_markdown(summary: dict[str, Any]) -> list[str]:
    methods = summary["methods"]
    quality_md = [
        "| Method | Tasks | Emitted tokens p50/p95 | Exact vs gold | Line distance proxy mean | Char distance proxy mean | Truncation rate | Source-copy line-overlap proxy mean | Finish reasons |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for method in sorted(methods):
        row = methods[method]
        token_stats = row["emitted_token_stats"]
        char_proxy = row["normalized_char_distance_proxy"]["mean"]
        line_proxy = row["normalized_line_distance_proxy"]["mean"]
        copy_proxy = row["source_copy_line_overlap_proxy"]["mean"]
        quality_md.append(
            "| "
            + " | ".join(
                [
                    method,
                    str(row["present_tasks"]),
                    f"{fmt(token_stats['p50'], 1)}/{fmt(token_stats['p95'], 1)}",
                    f"{row['exact_matches']}/{row['present_tasks']}",
                    fmt(line_proxy, 4),
                    fmt(char_proxy, 4),
                    fmt(row["truncation_rate"], 3),
                    fmt(copy_proxy, 4),
                    "`" + json.dumps(row["finish_reasons"], sort_keys=True) + "`",
                ]
            )
            + " |"
        )
    return quality_md


def equivalence_markdown(summary: dict[str, Any]) -> list[str]:
    equiv_md = [
        "| Pair | Compared | Exact task matches | Exact match rate | Length mismatches | Finish mismatches | Mean first mismatch position |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair, row in sorted(summary["pairwise_exact_task_matches"].items()):
        first_mean = row["first_mismatch_position"]["mean"]
        exact = f"{row['exact_task_matches']}/{row['compared_tasks']}"
        equiv_md.append(
            "| "
            + " | ".join(
                [
                    pair,
                    str(row["compared_tasks"]),
                    exact,
                    fmt(row["exact_task_match_rate"], 3),
                    str(row["length_mismatches"]),
                    str(row["finish_mismatches"]),
                    fmt(first_mean, 1),
                ]
            )
            + " |"
        )
    return equiv_md


def throughput_value(run: dict[str, Any], key: str) -> float | None:
    value = run.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def quality_throughput_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    methods = summary["methods"]
    run_metadata = summary.get("run_metadata", {})
    rows: list[dict[str, Any]] = []
    for method in sorted(methods):
        method_stats = methods[method]
        run = run_metadata.get(method, {})
        rows.append(
            {
                "method": method,
                "tasks": method_stats["present_tasks"],
                "tok_per_s_excluding_init": throughput_value(run, "tok_per_s_excluding_init"),
                "tok_per_s_including_init": throughput_value(run, "tok_per_s_including_init"),
                "emitted_tokens": run.get("total_emitted_tokens") or method_stats["emitted_tokens"],
                "exact_match_rate": method_stats["exact_match_rate"],
                "line_proxy_mean": method_stats["normalized_line_distance_proxy"]["mean"],
                "char_proxy_mean": method_stats["normalized_char_distance_proxy"]["mean"],
                "truncation_rate": method_stats["truncation_rate"],
                "source_copy_proxy_mean": method_stats["source_copy_line_overlap_proxy"]["mean"],
                "speculative_config": run.get("speculative_config"),
            }
        )

    for row in rows:
        throughput = row["tok_per_s_excluding_init"]
        line_proxy = row["line_proxy_mean"]
        char_proxy = row["char_proxy_mean"]
        exact_rate = row["exact_match_rate"]
        dominated = False
        if throughput is not None and line_proxy is not None and char_proxy is not None:
            for other in rows:
                if other is row:
                    continue
                other_throughput = other["tok_per_s_excluding_init"]
                other_line = other["line_proxy_mean"]
                other_char = other["char_proxy_mean"]
                other_exact = other["exact_match_rate"]
                if (
                    other_throughput is not None
                    and other_line is not None
                    and other_char is not None
                    and other_throughput >= throughput
                    and other_line <= line_proxy
                    and other_char <= char_proxy
                    and other_exact >= exact_rate
                    and (
                        other_throughput > throughput
                        or other_line < line_proxy
                        or other_char < char_proxy
                        or other_exact > exact_rate
                    )
                ):
                    dominated = True
                    break
        row["frontier"] = "no" if dominated else "yes"
    return rows


def quality_throughput_frontier_markdown(summary: dict[str, Any]) -> list[str]:
    rows = quality_throughput_rows(summary)
    greedy = next((row for row in rows if row["method"] == "greedy"), None)
    greedy_tps = greedy["tok_per_s_excluding_init"] if greedy else None
    lines = [
        "| Method | Tasks | Tok/s excl. init | Speedup vs greedy | Tok/s incl. init | Emitted tokens | Line proxy mean | Char proxy mean | Truncation rate | Frontier | Speculative config |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in sorted(
        rows,
        key=lambda item: item["tok_per_s_excluding_init"] or -1.0,
        reverse=True,
    ):
        throughput = row["tok_per_s_excluding_init"]
        speedup = throughput / greedy_tps if throughput is not None and greedy_tps else None
        lines.append(
            "| "
            + " | ".join(
                [
                    row["method"],
                    str(row["tasks"]),
                    fmt(throughput, 1),
                    fmt(speedup, 3),
                    fmt(row["tok_per_s_including_init"], 1),
                    str(row["emitted_tokens"]),
                    fmt(row["line_proxy_mean"], 4),
                    fmt(row["char_proxy_mean"], 4),
                    fmt(row["truncation_rate"], 3),
                    row["frontier"],
                    "`" + json.dumps(row["speculative_config"], sort_keys=True) + "`",
                ]
            )
            + " |"
        )
    return lines


def acceptance_markdown(summary: dict[str, Any]) -> list[str]:
    rows = summary.get("acceptance_proposer_stats", {})
    lines = [
        "# Acceptance/Proposer Stats",
        "",
        "The evaluator only reports accepted-token and proposer counters when they are present in captured artifacts. It does not infer acceptance from throughput or output equivalence.",
        "",
        "| Method | Status | Draft tokens | Accepted tokens | Acceptance rate | Acceptance fields | Proposer fields | Speculative config |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for method, row in sorted(rows.items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    row["status"],
                    fmt(row.get("num_draft_tokens"), 0),
                    fmt(row.get("num_accepted_tokens"), 0),
                    fmt(row.get("acceptance_rate"), 4),
                    "`" + json.dumps(row.get("acceptance_fields") or {}, sort_keys=True) + "`",
                    "`" + json.dumps(row.get("proposer_fields") or {}, sort_keys=True) + "`",
                    "`" + json.dumps(row.get("speculative_config"), sort_keys=True) + "`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Unavailable means the saved vLLM outputs, run summaries, and logs do not contain accepted-token counters. To populate this table, persist vLLM speculative decoding counters (`num_drafts`, `num_draft_tokens`, `num_accepted_tokens`, `num_accepted_tokens_per_pos`) plus accepted-length/rejection histograms and proposer-side denominators such as eligible queries, nonempty proposals, proposal tokens, and proposal lengths.",
        ]
    )
    return lines


def legacy_latex_tables(summary: dict[str, Any]) -> tuple[list[str], list[str]]:
    methods = summary["methods"]
    quality_tex = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{vLLM output quality against gold post-edit targets. Line and character distance proxies are $1-\\mathrm{SequenceMatcherRatio}$; source-copy overlap is the share of nonblank output lines exactly present in the source prompt when captured.}",
        "\\label{tab:vllm-quality}",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Method & Tasks & Tok p50 & Tok p95 & Gold exact & Line proxy & Trunc. & Copy proxy \\\\",
        "\\midrule",
    ]
    for method in sorted(methods):
        row = methods[method]
        token_stats = row["emitted_token_stats"]
        line_proxy = row["normalized_line_distance_proxy"]["mean"]
        copy_proxy = row["source_copy_line_overlap_proxy"]["mean"]
        quality_tex.append(
            f"{latex_escape(method)} & {row['present_tasks']} & {fmt(token_stats['p50'], 1)} & "
            f"{fmt(token_stats['p95'], 1)} & {row['exact_matches']}/{row['present_tasks']} & "
            f"{fmt(line_proxy, 4)} & {fmt(row['truncation_rate'], 3)} & {fmt(copy_proxy, 4)} \\\\"
        )
    quality_tex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    equiv_tex = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Pairwise exact-output equivalence for vLLM runs.}",
        "\\label{tab:vllm-equivalence}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Pair & Compared & Exact & Length diff. & Finish diff. & First mismatch \\\\",
        "\\midrule",
    ]
    for pair, row in sorted(summary["pairwise_exact_task_matches"].items()):
        first_mean = row["first_mismatch_position"]["mean"]
        exact = f"{row['exact_task_matches']}/{row['compared_tasks']}"
        equiv_tex.append(
            f"{latex_escape(pair)} & {row['compared_tasks']} & {latex_escape(exact)} & "
            f"{row['length_mismatches']} & {row['finish_mismatches']} & {fmt(first_mean, 1)} \\\\"
        )
    equiv_tex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return quality_tex, equiv_tex


def write_tables(result: dict[str, Any], table_dir: Path, table_prefix: str = "") -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    summary = result["summary"]
    quality_md = quality_markdown(summary)
    equiv_md = equivalence_markdown(summary)

    if table_prefix:
        (table_dir / f"{table_prefix}quality_table.md").write_text(
            "\n".join(quality_md) + "\n",
            encoding="utf-8",
        )
        (table_dir / f"{table_prefix}equivalence_table.md").write_text(
            "\n".join(equiv_md) + "\n",
            encoding="utf-8",
        )
        (table_dir / f"{table_prefix}quality_throughput_frontier.md").write_text(
            "\n".join(quality_throughput_frontier_markdown(summary)) + "\n",
            encoding="utf-8",
        )
        (table_dir / f"{table_prefix}acceptance_proposer_stats.md").write_text(
            "\n".join(acceptance_markdown(summary)) + "\n",
            encoding="utf-8",
        )
        return

    quality_tex, equiv_tex = legacy_latex_tables(summary)
    (table_dir / "external_baseline_quality.md").write_text("\n".join(quality_md) + "\n", encoding="utf-8")
    (table_dir / "external_baseline_quality.tex").write_text("\n".join(quality_tex), encoding="utf-8")
    (table_dir / "vllm_equivalence.md").write_text("\n".join(equiv_md) + "\n", encoding="utf-8")
    (table_dir / "vllm_equivalence.tex").write_text("\n".join(equiv_tex), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        "--gold",
        required=True,
        help="JSONL file containing task_id and gold text",
    )
    parser.add_argument(
        "--output-jsonl",
        "--outputs",
        "--output",
        action="append",
        required=True,
        help="Output JSONL path, optionally method_name=path. Repeat for multiple methods.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--report-prefix",
        default="",
        help="Optional filename prefix for eval_summary.json and eval_per_task.jsonl.",
    )
    parser.add_argument("--table-dir", help="Optional directory for generated Markdown/LaTeX tables")
    parser.add_argument(
        "--table-prefix",
        default="",
        help="Optional filename prefix for Markdown-only phase tables.",
    )
    parser.add_argument(
        "--run-summary",
        action="append",
        default=[],
        help="Optional run_summary.json path, optionally method_name=path. Repeat for multiple methods.",
    )
    parser.add_argument(
        "--acceptance-output",
        help="Optional JSON path for acceptance/proposer availability stats.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_specs = [parse_output_arg(value) for value in args.output_jsonl]
    outputs = merge_outputs(output_specs)
    result = evaluate(load_gold_manifest(Path(args.manifest)), outputs)
    run_summaries = load_run_summaries(output_specs, args.run_summary)
    result["summary"]["run_metadata"] = run_summaries
    result["summary"]["acceptance_proposer_stats"] = acceptance_proposer_stats(
        result["summary"]["methods"],
        run_summaries,
    )
    write_reports(result, Path(args.output_dir), report_prefix=args.report_prefix)
    if args.acceptance_output:
        write_acceptance_json(result, Path(args.acceptance_output))
    if args.table_dir:
        write_tables(result, Path(args.table_dir), table_prefix=args.table_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
