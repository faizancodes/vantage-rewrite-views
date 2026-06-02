"""Fit simple prompt-time routing rules from completed PLD/TransPLD runs.

This is intentionally conservative: it does not train a model.  It evaluates a
small menu of transparent prompt-time rules over existing completion artifacts
and reports aggregate throughput if each task had routed to either exact PLD or
the candidate VANTAGE row before decoding began.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from asts.code_proposers import _apply_word_map, _rewrite_pairs


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _nested_metadata(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("metadata")
    if not isinstance(meta, dict):
        return {}
    nested = meta.get("metadata")
    if isinstance(nested, dict):
        merged = dict(nested)
        merged.update(meta)
        return merged
    return meta


def _first_step_lengths(steps: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for step in steps:
        if int(step.get("step") or 0) != 0:
            continue
        task = str(step.get("task_id") or "")
        method = str(step.get("method") or "")
        if not task or not method:
            continue
        out[(task, method)] = int(step.get("proposal_tokens") or max(0, int(step.get("k") or 1) - 1))
    return out


def _rewrite_site_count(reference: str, pairs: dict[str, str]) -> int:
    total = 0
    for old in pairs:
        old_s = str(old)
        if old_s:
            total += reference.count(old_s)
    return total


def _method_time(row: dict[str, Any], method: str) -> tuple[int, float] | None:
    out = (row.get("outputs") or {}).get(method)
    if not isinstance(out, dict):
        return None
    return int(out.get("n_new_tokens") or 0), float(out.get("wall_us") or 0.0)


def _effective_map(row: dict[str, Any]) -> bool:
    pairs = _rewrite_pairs(str(row.get("prompt") or ""))
    if not pairs:
        return False
    reference = str(row.get("reference") or "")
    return bool(reference and _apply_word_map(reference, pairs) != reference)


def _rows_from_run(run_dir: Path, *, pld_method: str, trans_method: str) -> list[dict[str, Any]]:
    completions = _load_jsonl(run_dir / "completions.jsonl")
    steps = _load_jsonl(run_dir / "steps.jsonl")
    first_lens = _first_step_lengths(steps)
    out: list[dict[str, Any]] = []
    for row in completions:
        task_id = str(row.get("task_id") or "")
        pld_time = _method_time(row, pld_method)
        trans_time = _method_time(row, trans_method)
        if pld_time is None or trans_time is None:
            continue
        meta = _nested_metadata(row)
        pairs = _rewrite_pairs(str(row.get("prompt") or ""))
        reference = str(row.get("reference") or "")
        out.append(
            {
                "task_id": task_id,
                "run": str(run_dir),
                "drift_family": str(meta.get("drift_family") or meta.get("requested_cell") or ""),
                "has_effective_map": _effective_map(row),
                "rewrite_site_count": _rewrite_site_count(reference, pairs),
                "reference_len": int(meta.get("reference_tokens") or 0),
                "expected_output_len": int(meta.get("output_tokens") or 0),
                "longest_span": float(meta.get("longest_unchanged_span_tokens") or 0.0),
                "edit_distance": float(meta.get("edit_distance_tokens") or 0.0),
                "hunks": float(meta.get("changed_hunk_count") or 0.0),
                "exact_first_len": first_lens.get((task_id, pld_method), 0),
                "trans_first_len": first_lens.get((task_id, trans_method), 0),
                "pld_tokens": pld_time[0],
                "pld_wall_us": pld_time[1],
                "trans_tokens": trans_time[0],
                "trans_wall_us": trans_time[1],
            }
        )
    return out


def _aggregate(rows: list[dict[str, Any]], choose_trans) -> dict[str, Any]:
    tokens = 0
    wall = 0.0
    pld_tokens = 0
    pld_wall = 0.0
    trans_chosen = 0
    for row in rows:
        pld_tokens += int(row["pld_tokens"])
        pld_wall += float(row["pld_wall_us"])
        if choose_trans(row):
            tokens += int(row["trans_tokens"])
            wall += float(row["trans_wall_us"])
            trans_chosen += 1
        else:
            tokens += int(row["pld_tokens"])
            wall += float(row["pld_wall_us"])
    tps = tokens / (wall / 1e6) if wall > 0 else 0.0
    pld_tps = pld_tokens / (pld_wall / 1e6) if pld_wall > 0 else 0.0
    return {
        "n": len(rows),
        "trans_chosen": trans_chosen,
        "tokens_per_sec": tps,
        "ratio_vs_pld": tps / pld_tps if pld_tps > 0 else 0.0,
    }


def _rule_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rules: dict[str, Any] = {
        "always_pld": lambda r: False,
        "always_trans": lambda r: True,
        "effective_map_dispatch": lambda r: bool(r["has_effective_map"]),
        "style_or_field_dispatch": lambda r: bool(r["has_effective_map"])
        and str(r["drift_family"]) in {"style_rewrite", "field_rename"},
    }
    for margin in (0, 8, 16, 32):
        rules[f"first_len_margin_{margin}"] = (
            lambda r, margin=margin: bool(r["has_effective_map"])
            and int(r["trans_first_len"]) >= int(r["exact_first_len"]) + margin
        )
    rules["oracle_best_of_two"] = lambda r: float(r["trans_wall_us"]) < float(r["pld_wall_us"])
    return {name: _aggregate(rows, fn) for name, fn in rules.items()}


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Prompt-Time Router Fit",
        "",
        f"Rows: {report['n_rows']}. PLD: `{report['pld_method']}`. Candidate: `{report['trans_method']}`.",
        "",
        "| Rule | n | Trans chosen | tok/s | Ratio vs PLD |",
        "|------|--:|-------------:|------:|-------------:|",
    ]
    for name, item in report["rules"].items():
        lines.append(
            f"| `{name}` | {item['n']} | {item['trans_chosen']} | "
            f"{item['tokens_per_sec']:.2f} | {item['ratio_vs_pld']:.3f} |"
        )
    lines += ["", "## By Workload", ""]
    for workload, sub in report["by_workload"].items():
        lines.append(f"### {workload}")
        lines.append("")
        lines.append("| Rule | n | Trans chosen | tok/s | Ratio vs PLD |")
        lines.append("|------|--:|-------------:|------:|-------------:|")
        for name, item in sub.items():
            lines.append(
                f"| `{name}` | {item['n']} | {item['trans_chosen']} | "
                f"{item['tokens_per_sec']:.2f} | {item['ratio_vs_pld']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", required=True, help="Comma-separated eval directories.")
    parser.add_argument("--pld-method", default="blazedit_pld_w128_n10")
    parser.add_argument("--trans-method", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for item in args.run_dirs.split(","):
        item = item.strip()
        if item:
            rows.extend(
                _rows_from_run(
                    Path(item),
                    pld_method=args.pld_method,
                    trans_method=args.trans_method,
                )
            )
    by_workload: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_workload[str(row["drift_family"]) or "unknown"].append(row)
    report = {
        "n_rows": len(rows),
        "pld_method": args.pld_method,
        "trans_method": args.trans_method,
        "rules": _rule_table(rows),
        "by_workload": {name: _rule_table(group) for name, group in sorted(by_workload.items())},
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2))
    md = _markdown(report)
    Path(args.output_md).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
