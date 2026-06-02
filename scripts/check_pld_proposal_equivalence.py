#!/usr/bin/env python3
"""Check PLD proposal equivalence from JSONL traces when trace fields exist.

The tool supports two useful modes:

1. ``--trace TRACE.jsonl`` replays rows that contain prompt/generated token IDs
   (or combined token history plus prompt length) through the pure PLD oracle
   and compares any proposal fields present in the row.
2. ``--expected-trace A.jsonl --actual-trace B.jsonl`` compares proposal fields
   shared by two existing traces, keyed by ``task_id`` and ``step`` when present.

Rows without enough token/proposal metadata are counted as skipped. Hash-only
evidence is labeled explicitly and never certifies PLD equivalence. Any
concrete mismatch exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vantage_vllm.pld_lookup import PLDLookupResult, TokenRange, find_pld_proposal


PROPOSAL_FIELDS = (
    "proposal_token_ids",
    "proposal_tokens",
    "proposal_match_len",
    "proposal_source_start_token",
    "proposal_follow_start_token",
    "proposal_query_start_token",
    "proposal_source_region",
)

HASH_FIELDS = (
    "proposal_hash",
    "proposal_token_hash",
    "proposal_token_ids_hash",
    "proposal_token_ids_sha256",
    "draft_hash",
    "draft_token_hash",
    "draft_token_ids_hash",
    "draft_token_ids_sha256",
    "token_trace_hash",
    "trace_hash",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", default=[], help="JSONL trace to replay.")
    parser.add_argument("--expected-trace", default="", help="Reference JSONL trace.")
    parser.add_argument("--actual-trace", default="", help="Candidate JSONL trace to compare.")
    parser.add_argument("--match-n", type=int, default=10)
    parser.add_argument("--max-draft-len", type=int, default=128)
    parser.add_argument("--cap", type=int, default=None)
    parser.add_argument(
        "--exclude-range",
        action="append",
        default=[],
        help="Extra half-open start:end range.",
    )
    parser.add_argument("--no-search-prompt", action="store_true")
    parser.add_argument("--no-search-generated", action="store_true")
    parser.add_argument("--tie-break", choices=["latest", "earliest"], default="latest")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--output-json", default="", help="Optional path for the summary JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.trace and not (args.expected_trace and args.actual_trace):
        parse_args(["--help"])
        return 0

    extra_ranges = _parse_cli_ranges(args.exclude_range)
    summaries: list[dict[str, Any]] = []
    rc = 0

    if args.expected_trace and args.actual_trace:
        summary = compare_trace_files(
            Path(args.expected_trace),
            Path(args.actual_trace),
            max_rows=args.max_rows,
        )
        summaries.append(summary)
        if summary["mismatches"]:
            rc = 1

    for trace in args.trace:
        summary = replay_trace(
            Path(trace),
            match_n=args.match_n,
            max_draft_len=args.max_draft_len,
            cap=args.cap,
            exclude_ranges=extra_ranges,
            search_prompt=not args.no_search_prompt,
            search_generated=not args.no_search_generated,
            tie_break=args.tie_break,
            max_rows=args.max_rows,
        )
        summaries.append(summary)
        if summary["mismatches"]:
            rc = 1

    status = _overall_status(summaries, has_mismatch=bool(rc))
    output = {
        "status": status,
        "pld_equivalence_certified": status == "pld_equivalent",
        "summaries": summaries,
    }
    text = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return rc


def compare_trace_files(
    expected_path: Path,
    actual_path: Path,
    *,
    max_rows: int = 0,
) -> dict[str, Any]:
    expected_rows = _load_jsonl(expected_path, max_rows=max_rows)
    actual_rows = _load_jsonl(actual_path, max_rows=max_rows)
    expected_by_key = {_row_key(row, i): row for i, row in enumerate(expected_rows)}
    actual_by_key = {_row_key(row, i): row for i, row in enumerate(actual_rows)}
    common_keys = sorted(set(expected_by_key) & set(actual_by_key), key=str)

    mismatches: list[dict[str, Any]] = []
    compared_fields = 0
    compared_token_trace_rows = 0
    compared_metadata_only_rows = 0
    hash_only_rows = 0
    skipped_no_fields = 0
    for key in common_keys:
        expected_row = expected_by_key[key]
        actual_row = actual_by_key[key]
        expected_sig = _proposal_signature(expected_row)
        actual_sig = _proposal_signature(actual_row)
        shared = sorted(set(expected_sig) & set(actual_sig))
        hash_diffs = _shared_hash_diffs(expected_row, actual_row)
        if not shared:
            if _hash_signature(expected_row) or _hash_signature(actual_row):
                hash_only_rows += 1
                if hash_diffs:
                    mismatches.append({"key": list(key), "diffs": hash_diffs})
            else:
                skipped_no_fields += 1
            continue
        if "proposal_token_ids" in shared:
            compared_token_trace_rows += 1
        else:
            compared_metadata_only_rows += 1
        compared_fields += len(shared)
        diffs = {
            field: {"expected": expected_sig[field], "actual": actual_sig[field]}
            for field in shared
            if expected_sig[field] != actual_sig[field]
        }
        diffs.update(hash_diffs)
        if diffs:
            mismatches.append({"key": list(key), "diffs": diffs})

    missing_from_actual = sorted(set(expected_by_key) - set(actual_by_key), key=str)
    missing_from_expected = sorted(set(actual_by_key) - set(expected_by_key), key=str)
    pld_equivalent = (
        not mismatches
        and bool(common_keys)
        and not missing_from_actual
        and not missing_from_expected
        and compared_token_trace_rows == len(common_keys)
    )

    equivalence_label = _equivalence_label(
        pld_equivalence_certified=pld_equivalent,
        evidence_label=_evidence_label(
            total_rows=len(common_keys),
            token_trace_rows=compared_token_trace_rows,
            metadata_only_rows=compared_metadata_only_rows,
            hash_only_rows=hash_only_rows,
            skipped_rows=skipped_no_fields + len(missing_from_actual) + len(missing_from_expected),
        ),
        capped=False,
    )
    return {
        "mode": "compare",
        "expected_trace": str(expected_path),
        "actual_trace": str(actual_path),
        "expected_rows": len(expected_rows),
        "actual_rows": len(actual_rows),
        "common_rows": len(common_keys),
        "missing_from_actual": len(missing_from_actual),
        "missing_from_expected": len(missing_from_expected),
        "compared_fields": compared_fields,
        "compared_token_trace_rows": compared_token_trace_rows,
        "compared_metadata_only_rows": compared_metadata_only_rows,
        "hash_only_rows": hash_only_rows,
        "skipped_no_fields": skipped_no_fields,
        "evidence_label": _evidence_label(
            total_rows=len(common_keys),
            token_trace_rows=compared_token_trace_rows,
            metadata_only_rows=compared_metadata_only_rows,
            hash_only_rows=hash_only_rows,
            skipped_rows=skipped_no_fields + len(missing_from_actual) + len(missing_from_expected),
        ),
        "equivalence_label": equivalence_label,
        "pld_equivalence_certified": pld_equivalent,
        "mismatches": mismatches,
    }


def replay_trace(
    path: Path,
    *,
    match_n: int,
    max_draft_len: int,
    cap: int | None,
    exclude_ranges: Iterable[TokenRange],
    search_prompt: bool,
    search_generated: bool,
    tie_break: str,
    max_rows: int = 0,
) -> dict[str, Any]:
    rows = _load_jsonl(path, max_rows=max_rows)
    mismatches: list[dict[str, Any]] = []
    replayed = 0
    compared_token_trace_rows = 0
    compared_metadata_only_rows = 0
    hash_only_rows = 0
    skipped_no_tokens = 0
    skipped_no_fields = 0
    compared_fields = 0

    for index, row in enumerate(rows):
        extracted = _extract_prompt_generated(row)
        if extracted is None:
            skipped_no_tokens += 1
            continue
        prompt_ids, generated_ids = extracted
        row_ranges = _extract_ranges(row)
        proposal = find_pld_proposal(
            prompt_ids,
            generated_ids,
            match_n=match_n,
            max_draft_len=max_draft_len,
            cap=cap,
            exclude_ranges=(*tuple(exclude_ranges), *row_ranges),
            search_prompt=search_prompt,
            search_generated=search_generated,
            tie_break=tie_break,  # type: ignore[arg-type]
        )
        replayed += 1
        expected_sig = _proposal_signature_from_result(proposal)
        actual_sig = _proposal_signature(row)
        shared = sorted(set(expected_sig) & set(actual_sig))
        if not shared:
            if _hash_signature(row):
                hash_only_rows += 1
            else:
                skipped_no_fields += 1
            continue
        if "proposal_token_ids" in shared:
            compared_token_trace_rows += 1
        else:
            compared_metadata_only_rows += 1
        compared_fields += len(shared)
        diffs = {
            field: {"expected": expected_sig[field], "actual": actual_sig[field]}
            for field in shared
            if expected_sig[field] != actual_sig[field]
        }
        if diffs:
            mismatches.append({"key": list(_row_key(row, index)), "diffs": diffs})

    evidence_label = _evidence_label(
        total_rows=replayed,
        token_trace_rows=compared_token_trace_rows,
        metadata_only_rows=compared_metadata_only_rows,
        hash_only_rows=hash_only_rows,
        skipped_rows=skipped_no_tokens + skipped_no_fields,
    )
    pld_equivalence_certified = (
        not mismatches
        and replayed > 0
        and skipped_no_tokens == 0
        and skipped_no_fields == 0
        and compared_token_trace_rows == replayed
    )
    return {
        "mode": "replay",
        "trace": str(path),
        "rows": len(rows),
        "replayed_rows": replayed,
        "compared_fields": compared_fields,
        "compared_token_trace_rows": compared_token_trace_rows,
        "compared_metadata_only_rows": compared_metadata_only_rows,
        "hash_only_rows": hash_only_rows,
        "skipped_no_tokens": skipped_no_tokens,
        "skipped_no_fields": skipped_no_fields,
        "evidence_label": evidence_label,
        "equivalence_label": _equivalence_label(
            pld_equivalence_certified=pld_equivalence_certified,
            evidence_label=evidence_label,
            capped=cap is not None and cap < max_draft_len,
        ),
        "pld_equivalence_certified": pld_equivalence_certified,
        "mismatches": mismatches,
    }


def _load_jsonl(path: Path, *, max_rows: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if max_rows and len(rows) >= max_rows:
                    break
    return rows


def _row_key(row: dict[str, Any], index: int) -> tuple[Any, ...]:
    task_id = row.get("task_id")
    step = row.get("step")
    if task_id is not None and step is not None:
        return (task_id, step)
    if task_id is not None:
        return (task_id, index)
    return (index,)


def _proposal_signature(row: dict[str, Any]) -> dict[str, Any]:
    signature: dict[str, Any] = {}
    for field in PROPOSAL_FIELDS:
        value = row.get(field)
        if value is not None:
            signature[field] = _json_scalar(value)
    token_ids = _first_present(
        row.get("proposal_token_ids"),
        row.get("draft_token_ids"),
        row.get("proposed_token_ids"),
    )
    if token_ids is not None:
        signature["proposal_token_ids"] = [int(token) for token in token_ids]
        signature["proposal_tokens"] = len(signature["proposal_token_ids"])
    return signature


def _hash_signature(row: dict[str, Any]) -> dict[str, Any]:
    return {
        field: _json_scalar(row[field])
        for field in HASH_FIELDS
        if row.get(field) is not None
    }


def _shared_hash_diffs(
    expected_row: dict[str, Any],
    actual_row: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    expected_hashes = _hash_signature(expected_row)
    actual_hashes = _hash_signature(actual_row)
    return {
        field: {"expected": expected_hashes[field], "actual": actual_hashes[field]}
        for field in sorted(set(expected_hashes) & set(actual_hashes))
        if expected_hashes[field] != actual_hashes[field]
    }


def _proposal_signature_from_result(proposal: PLDLookupResult | None) -> dict[str, Any]:
    if proposal is None:
        return {"proposal_token_ids": [], "proposal_tokens": 0}
    return {
        "proposal_token_ids": list(proposal.tokens),
        "proposal_tokens": len(proposal.tokens),
        "proposal_match_len": proposal.match_n,
        "proposal_source_start_token": proposal.source_start,
        "proposal_follow_start_token": proposal.follow_start,
        "proposal_query_start_token": proposal.query_start,
        "proposal_source_region": proposal.source,
    }


def _extract_prompt_generated(row: dict[str, Any]) -> tuple[list[int], list[int]] | None:
    prompt_ids = _token_list_from_keys(
        row,
        "prompt_ids",
        "prompt_token_ids",
        "context_ids",
        "context_token_ids",
    )
    generated_ids = _token_list_from_keys(
        row,
        "generated_ids",
        "generated_token_ids",
        "generated_prefix_ids",
        "output_so_far_token_ids",
    )
    if prompt_ids is not None and generated_ids is not None:
        return prompt_ids, generated_ids

    history = _token_list_from_keys(row, "token_ids", "history_token_ids", "input_token_ids")
    prompt_len = _first_present_int(
        row.get("prompt_len"),
        row.get("prompt_length"),
        row.get("prompt_token_count"),
    )
    if history is None or prompt_len is None:
        return None
    prompt_len = max(0, min(prompt_len, len(history)))
    return history[:prompt_len], history[prompt_len:]


def _extract_ranges(row: dict[str, Any]) -> tuple[TokenRange, ...]:
    ranges = _first_present(
        row.get("exclude_ranges"),
        row.get("pld_exclude_ranges"),
        row.get("gold_ranges"),
        row.get("gold_token_ranges"),
    )
    if ranges is None:
        return ()
    return _normalize_ranges(ranges)


def _token_list_from_keys(row: dict[str, Any], *keys: str) -> list[int] | None:
    value = _first_present(*(row.get(key) for key in keys))
    if value is None:
        return None
    return [int(token) for token in value]


def _parse_cli_ranges(values: Iterable[str]) -> tuple[TokenRange, ...]:
    ranges: list[TokenRange] = []
    for value in values:
        left, sep, right = value.partition(":")
        if not sep:
            raise SystemExit(f"invalid --exclude-range {value!r}; expected start:end")
        ranges.append((int(left), int(right)))
    return tuple(ranges)


def _normalize_ranges(value: Any) -> tuple[TokenRange, ...]:
    if isinstance(value, dict):
        start = value.get("start")
        end = value.get("end")
        if start is not None and end is not None:
            return ((int(start), int(end)),)
        value = value.values()
    ranges: list[TokenRange] = []
    for item in value:
        if isinstance(item, dict):
            start = item.get("start")
            end = item.get("end")
        else:
            start, end = item
        ranges.append((int(start), int(end)))
    return tuple(ranges)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _first_present_int(*values: Any) -> int | None:
    value = _first_present(*values)
    if value is None:
        return None
    return int(value)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int, float)):
        return value
    return value


def _evidence_label(
    *,
    total_rows: int,
    token_trace_rows: int,
    metadata_only_rows: int,
    hash_only_rows: int,
    skipped_rows: int,
) -> str:
    if total_rows <= 0:
        return "none"
    if token_trace_rows == total_rows and skipped_rows == 0:
        return "token_trace"
    if token_trace_rows:
        return "partial_token_trace"
    if hash_only_rows and not metadata_only_rows:
        return "hash_only"
    if metadata_only_rows:
        return "metadata_only"
    return "none"


def _overall_status(summaries: list[dict[str, Any]], *, has_mismatch: bool) -> str:
    if has_mismatch:
        return "mismatch"
    if summaries and all(summary.get("pld_equivalence_certified") for summary in summaries):
        return "pld_equivalent"
    return "insufficient_token_trace"


def _equivalence_label(
    *,
    pld_equivalence_certified: bool,
    evidence_label: str,
    capped: bool,
) -> str:
    if not pld_equivalence_certified:
        if evidence_label == "hash_only":
            return "metadata_insufficient"
        if evidence_label in {"metadata_only", "partial_token_trace", "none"}:
            return "metadata_insufficient"
        return "not_equivalent"
    if capped:
        return "capped_full_prefix_pld_equivalent"
    return "true_full_prefix_pld_equivalent"


if __name__ == "__main__":
    raise SystemExit(main())
