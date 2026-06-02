"""Diagnose TransPLD failures by model/workload.

This script is meant for the reviewer-priority decision point after a timing
run.  It does not rerun the model; it reads `steps.jsonl` and
`completions.jsonl`, optionally replays PLD-overlap with the target tokenizer,
and reports:

* tokenizer-boundary samples around explicit rewrite pairs;
* accepted tokens per TransPLD hit;
* exact-PLD and TransPLD hit/acceptance rates;
* realized rewrite/compliance buckets from the model output;
* proposal-overhead fractions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asts.code_proposers import _apply_word_map, _coerce_rewrite_pairs  # noqa: E402


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _nested(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    nested = metadata.get("metadata")
    return nested if isinstance(nested, dict) else {}


def _rewrite_pairs(row: dict[str, Any]) -> dict[str, str]:
    metadata = row.get("metadata") or {}
    nested = _nested(metadata)
    return _coerce_rewrite_pairs(
        row.get("rewrite_pairs")
        or metadata.get("rewrite_pairs")
        or nested.get("rewrite_pairs")
    )


def _output_text(row: dict[str, Any], method: str) -> str:
    out = (row.get("outputs") or {}).get(method) or {}
    return str(out.get("text") or out.get("raw_text") or "")


def _ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(a=a, b=b).ratio()


def _method_steps(steps: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    return [s for s in steps if s.get("method") == method]


def _pld_stats(steps: list[dict[str, Any]], method: str) -> dict[str, Any]:
    rows = _method_steps(steps, method)
    hits = [
        s
        for s in rows
        if int(s.get("blazedit_pld_proposed") or s.get("proposal_tokens") or 0) > 0
    ]
    accepted = sum(
        int(s.get("blazedit_pld_accepted") or s.get("n_accepted_nonroot_drafts") or 0)
        for s in hits
    )
    proposal_us = sum(float(s.get("proposal_us") or 0.0) for s in rows)
    wall_us = sum(float(s.get("wall_us") or 0.0) for s in rows)
    return {
        "steps": len(rows),
        "hits": len(hits),
        "hit_rate": len(hits) / len(rows) if rows else 0.0,
        "accepted_nonroot": accepted,
        "accepted_per_hit": accepted / len(hits) if hits else 0.0,
        "accepted_per_step": accepted / len(rows) if rows else 0.0,
        "proposal_us_total": proposal_us,
        "proposal_wall_share": proposal_us / wall_us if wall_us else 0.0,
    }


def _transpld_stats(steps: list[dict[str, Any]], method: str) -> dict[str, Any]:
    rows = _method_steps(steps, method)
    hits = [
        s
        for s in rows
        if s.get("proposal_match_kind")
        in {
            "bidir",
            "transpld_bidir",
            "transpld_vref",
            "routed_transpld_bidir",
            "routed_transpld_vref",
            "transpld_bidir_inferred",
        }
        or s.get("proposal_kind")
        in {"rewrite_norm_pld", "transpld", "routed_transpld"}
    ]
    accepted = sum(int(s.get("n_accepted_nonroot_drafts") or 0) for s in hits)
    zero = sum(1 for s in hits if int(s.get("n_accepted_nonroot_drafts") or 0) <= 0)
    proposal_us = sum(float(s.get("proposal_us") or 0.0) for s in rows)
    wall_us = sum(float(s.get("wall_us") or 0.0) for s in rows)
    by_match = Counter(str(s.get("proposal_match_kind") or "none") for s in rows)
    by_route = Counter(str(s.get("proposal_route") or "none") for s in rows)
    return {
        "steps": len(rows),
        "hits": len(hits),
        "hit_rate": len(hits) / len(rows) if rows else 0.0,
        "accepted_nonroot": accepted,
        "accepted_per_hit": accepted / len(hits) if hits else 0.0,
        "accepted_per_step": accepted / len(rows) if rows else 0.0,
        "zero_accept_hits": zero,
        "zero_accept_rate": zero / len(hits) if hits else 0.0,
        "proposal_us_total": proposal_us,
        "proposal_wall_share": proposal_us / wall_us if wall_us else 0.0,
        "match_kind_counts": dict(by_match),
        "route_counts": dict(by_route),
    }


def _realized_regimes(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    details: list[dict[str, Any]] = []
    for row in rows:
        reference = str(row.get("reference") or "")
        if not reference:
            continue
        pairs = _rewrite_pairs(row)
        transformed = _apply_word_map(reference, pairs) if pairs else reference
        output = _output_text(row, method)
        if not output:
            continue
        old_count = sum(output.count(old) for old in pairs)
        new_count = sum(output.count(new) for new in pairs.values())
        ref_ratio = _ratio(output, reference)
        transformed_ratio = _ratio(output, transformed)
        if transformed_ratio >= 0.92 and transformed_ratio >= ref_ratio + 0.03:
            bucket = "transformed_reference_aligned"
        elif ref_ratio >= 0.92 and new_count == 0:
            bucket = "verbatim_or_original_aligned"
        elif transformed_ratio < 0.55 and ref_ratio < 0.55:
            bucket = "low_copy_or_malformed"
        else:
            bucket = "mixed_or_partial"
        counts[bucket] += 1
        details.append(
            {
                "task_id": row.get("task_id"),
                "bucket": bucket,
                "old_occurrences_in_output": old_count,
                "new_occurrences_in_output": new_count,
                "output_vs_reference_ratio": ref_ratio,
                "output_vs_transformed_ratio": transformed_ratio,
            }
        )
    return {"counts": dict(counts), "tasks": details}


def _tokenizer_boundary_audit(
    rows: list[dict[str, Any]],
    *,
    tokenizer_name: str,
    output_method: str,
    limit: int,
) -> list[dict[str, Any]]:
    if not tokenizer_name:
        return []
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    samples: list[dict[str, Any]] = []
    for row in rows:
        pairs = _rewrite_pairs(row)
        reference = str(row.get("reference") or "")
        if not pairs or not reference:
            continue
        for old, new in pairs.items():
            idx = reference.find(old)
            if idx < 0:
                continue
            before = reference[max(0, idx - 16) : idx]
            after = reference[idx + len(old) : idx + len(old) + 16]
            old_surface = before + old + after
            new_surface = before + new + after
            output = _output_text(row, output_method)
            output_new_piece_ids: list[int] = []
            output_old_piece_ids: list[int] = []
            out_new_idx = output.find(new)
            if out_new_idx >= 0:
                output_new_piece_ids = tok(
                    output[out_new_idx : out_new_idx + len(new)],
                    add_special_tokens=False,
                ).input_ids
            out_old_idx = output.find(old)
            if out_old_idx >= 0:
                output_old_piece_ids = tok(
                    output[out_old_idx : out_old_idx + len(old)],
                    add_special_tokens=False,
                ).input_ids
            samples.append(
                {
                    "task_id": row.get("task_id"),
                    "old": old,
                    "new": new,
                    "old_surface": old_surface,
                    "new_surface": new_surface,
                    "old_ids": tok(old_surface, add_special_tokens=False).input_ids,
                    "new_ids": tok(new_surface, add_special_tokens=False).input_ids,
                    "old_piece_ids": tok(old, add_special_tokens=False).input_ids,
                    "new_piece_ids": tok(new, add_special_tokens=False).input_ids,
                    "output_method": output_method,
                    "output_new_piece_ids": output_new_piece_ids,
                    "output_old_piece_ids": output_old_piece_ids,
                }
            )
            if len(samples) >= limit:
                return samples
    return samples


def _speed_table(rows: list[dict[str, Any]], methods: list[str], baseline: str) -> dict[str, Any]:
    out = {}
    base_tps = 0.0
    for method in methods:
        tokens = 0
        wall_us = 0.0
        for row in rows:
            output = (row.get("outputs") or {}).get(method)
            if not output:
                continue
            tokens += int(output.get("n_new_tokens") or 0)
            wall_us += float(output.get("wall_us") or 0.0)
        tps = tokens / (wall_us / 1e6) if wall_us else 0.0
        out[method] = {"tokens": tokens, "wall_us": wall_us, "tokens_per_sec": tps}
        if method == baseline:
            base_tps = tps
    for item in out.values():
        item["speedup_vs_baseline"] = item["tokens_per_sec"] / base_tps if base_tps else 0.0
    return out


def analyze(
    completions: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    *,
    pld_method: str,
    transpld_method: str,
    routed_method: str,
    tokenizer_name: str,
    boundary_method: str = "vanilla",
    pld_window: int = 128,
    pld_ngram: int = 10,
) -> dict[str, Any]:
    methods = [pld_method, transpld_method, routed_method]
    report = {
        "schema": "asts-spec/transpld-diagnostics/v1",
        "methods": methods,
        "speed": _speed_table(completions, methods, pld_method),
        "pld": _pld_stats(steps, pld_method),
        "transpld": _transpld_stats(steps, transpld_method),
        "routed": _transpld_stats(steps, routed_method),
        "realized_regimes": {
            method: _realized_regimes(completions, method) for method in methods
        },
        "tokenizer_boundary_samples": _tokenizer_boundary_audit(
            completions,
            tokenizer_name=tokenizer_name,
            output_method=boundary_method,
            limit=8,
        ),
    }
    if tokenizer_name:
        from scripts.analyze_anchor_pld_overlap import analyze as analyze_overlap

        report["pld_miss_recovery"] = {
            transpld_method: analyze_overlap(
                completions=completions,
                steps=steps,
                method=transpld_method,
                tokenizer_name=tokenizer_name,
                pld_window=pld_window,
                pld_ngram=pld_ngram,
            ).get("fractions", {}),
            routed_method: analyze_overlap(
                completions=completions,
                steps=steps,
                method=routed_method,
                tokenizer_name=tokenizer_name,
                pld_window=pld_window,
                pld_ngram=pld_ngram,
            ).get("fractions", {}),
        }
    return report


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# TransPLD Run Diagnostics",
        "",
        "## Speed",
        "",
        "| Method | tok/s | vs PLD |",
        "|---|---:|---:|",
    ]
    for method, row in report["speed"].items():
        lines.append(
            f"| `{method}` | {row['tokens_per_sec']:.2f} | {row['speedup_vs_baseline']:.3f} |"
        )
    lines += [
        "",
        "## Acceptance And Overhead",
        "",
        "| Source | steps | hit rate | accepted/hit | accepted/step | zero-hit rate | proposal wall share |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("pld", "transpld", "routed"):
        row = report[label]
        lines.append(
            f"| {label} | {row['steps']} | {100 * row['hit_rate']:.1f}% | "
            f"{row['accepted_per_hit']:.2f} | {row['accepted_per_step']:.2f} | "
            f"{100 * row.get('zero_accept_rate', 0.0):.1f}% | "
            f"{100 * row['proposal_wall_share']:.2f}% |"
        )
    lines += ["", "## Realized Regime Buckets", ""]
    for method, reg in report["realized_regimes"].items():
        lines.append(f"### `{method}`")
        for key, value in sorted(reg["counts"].items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- `{key}`: {value}")
        lines.append("")
    lines += ["## Tokenizer Boundary Samples", ""]
    for sample in report["tokenizer_boundary_samples"]:
        lines.append(
            f"- `{sample['task_id']}` `{sample['old']}` -> `{sample['new']}`: "
            f"old_piece={sample['old_piece_ids']} new_piece={sample['new_piece_ids']} "
            f"output_new_piece={sample['output_new_piece_ids']} "
            f"output_old_piece={sample['output_old_piece_ids']}"
        )
    if report.get("pld_miss_recovery"):
        lines += ["", "## PLD-Miss Recovery Fractions", ""]
        for method, fractions in report["pld_miss_recovery"].items():
            lines.append(f"### `{method}`")
            for key, value in sorted(fractions.items()):
                lines.append(f"- `{key}`: {100 * float(value):.1f}%")
            lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completions", required=True)
    parser.add_argument("--steps", required=True)
    parser.add_argument("--pld-method", default="blazedit_pld_w128_n10")
    parser.add_argument("--transpld-method", default="vantage_transpld_w128_n10")
    parser.add_argument("--routed-method", default="vantage_routed_transpld_w128_n10")
    parser.add_argument("--target-tokenizer", default="")
    parser.add_argument("--boundary-method", default="vanilla")
    parser.add_argument("--pld-window", type=int, default=128)
    parser.add_argument("--pld-ngram", type=int, default=10)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    report = analyze(
        _load_jsonl(Path(args.completions)),
        _load_jsonl(Path(args.steps)),
        pld_method=args.pld_method,
        transpld_method=args.transpld_method,
        routed_method=args.routed_method,
        tokenizer_name=args.target_tokenizer,
        boundary_method=args.boundary_method,
        pld_window=args.pld_window,
        pld_ngram=args.pld_ngram,
    )
    Path(args.output_json).write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, Path(args.output_md))
    print(Path(args.output_md).read_text())


if __name__ == "__main__":
    main()
