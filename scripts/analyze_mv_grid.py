#!/usr/bin/env python3
"""Summarize VANTAGE-MV train-grid runs.

The script ranks methods by weighted aggregate throughput and reports the
route/proposal counters needed to choose one held-out-test candidate. It does
not inspect or require the test split.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _method_wall_tokens(completions: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(lambda: {"wall_us": 0.0, "tokens": 0.0})
    for row in completions:
        methods = (
            row.get("outputs")
            if isinstance(row.get("outputs"), dict)
            else row.get("methods")
            if isinstance(row.get("methods"), dict)
            else {}
        )
        for method, item in methods.items():
            if not isinstance(item, dict):
                continue
            out[method]["wall_us"] += float(item.get("wall_us") or 0.0)
            out[method]["tokens"] += float(item.get("n_new_tokens") or len(item.get("tokens") or []))
    return out


def _fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def analyze(eval_dir: Path, *, baseline: str) -> dict[str, Any]:
    aggregate = _read_json(eval_dir / "aggregate.json")
    completions = _read_jsonl(eval_dir / "completions.jsonl")
    steps = _read_jsonl(eval_dir / "steps.jsonl")
    wall_tokens = _method_wall_tokens(completions)

    baseline_tps = None
    methods: dict[str, dict[str, Any]] = {}
    by_method_steps: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step in steps:
        method = str(step.get("method") or "")
        if method:
            by_method_steps[method].append(step)

    for method, row in sorted((aggregate.get("by_method") or {}).items()):
        wt = wall_tokens.get(method, {})
        wall_s = float(wt.get("wall_us") or 0.0) / 1_000_000.0
        tokens = float(wt.get("tokens") or row.get("n_new_tokens") or 0.0)
        tps = tokens / wall_s if wall_s > 0 else float(row.get("tokens_per_sec") or 0.0)
        if method == baseline:
            baseline_tps = tps
        method_steps = by_method_steps.get(method, [])
        route_reasons = Counter(str(s.get("proposal_route_reason") or "none") for s in method_steps)
        proposal_kinds = Counter(str(s.get("proposal_kind") or "none") for s in method_steps)
        trans_steps = [
            s
            for s in method_steps
            if str(s.get("proposal_kind") or "")
            in {
                "vantage_mv_pld",
                "vantage_mv_branch_common",
                "vantage_mv_branch_tree",
                "vantage_mv_branch_packed",
                "vantage_edit_neural_draft",
            }
        ]
        trans_tasks = {
            str(s.get("task_id"))
            for s in trans_steps
            if s.get("task_id") is not None
        }
        zero_accept = sum(1 for s in trans_steps if int(s.get("n_accepted_drafts") or 0) == 0)
        accepted_total = sum(int(s.get("n_accepted_drafts") or 0) for s in method_steps)
        verify_s = sum(float(s.get("verify_us") or 0.0) for s in method_steps) / 1_000_000.0
        exact_capped = sum(1 for s in method_steps if bool(s.get("proposal_alpha_exact_filtered")))
        branch_accepted = sum(
            int(s.get("n_accepted_drafts") or 0)
            for s in method_steps
            if int(s.get("proposal_tree_candidates") or 0) >= 2
        )
        branch_steps = [
            s for s in method_steps if int(s.get("proposal_tree_candidates") or 0) >= 2
        ]
        transformed_branch_selected = sum(
            1 for s in branch_steps if s.get("proposal_tree_branch_selected") == 1
        )
        rescue_steps = [
            s
            for s in trans_steps
            if str(s.get("proposal_route_reason") or "").startswith("pld_rejection_rescue")
        ]
        patch_steps = [
            s for s in trans_steps if str(s.get("proposal_route_reason") or "") == "patch_segment_wins"
        ]
        methods[method] = {
            "tokens_per_sec": tps,
            "tokens": tokens,
            "wall_s": wall_s,
            "steps": len(method_steps),
            "trans_steps": len(trans_steps),
            "trans_tasks": len(trans_tasks),
            "trans_zero_accept_steps": zero_accept,
            "trans_zero_accept_rate": zero_accept / len(trans_steps) if trans_steps else 0.0,
            "trans_accepted": sum(int(s.get("n_accepted_drafts") or 0) for s in trans_steps),
            "trans_proposed": sum(int(s.get("proposal_tokens") or 0) for s in trans_steps),
            "accepted_per_verify": accepted_total / len(method_steps) if method_steps else 0.0,
            "verify_s": verify_s,
            "exact_capped_steps": exact_capped,
            "branch_accepted_tokens": branch_accepted,
            "branch_steps": len(branch_steps),
            "transformed_branch_selected": transformed_branch_selected,
            "rescue_steps": len(rescue_steps),
            "rescue_accepted": sum(int(s.get("n_accepted_drafts") or 0) for s in rescue_steps),
            "patch_steps": len(patch_steps),
            "patch_accepted": sum(int(s.get("n_accepted_drafts") or 0) for s in patch_steps),
            "staged_chunks": sum(int(s.get("verify_staged_chunks") or 0) for s in method_steps),
            "staged_saved_tokens": sum(int(s.get("verify_staged_saved_tokens") or 0) for s in method_steps),
            "neural_steps": sum(
                1 for s in method_steps if s.get("proposal_kind") == "vantage_edit_neural_draft"
            ),
            "neural_tokens": sum(int(s.get("proposal_neural_draft_tokens") or 0) for s in method_steps),
            "route_reasons": dict(route_reasons),
            "proposal_kinds": dict(proposal_kinds),
            "setup_s": sum(
                float(s.get("proposal_map_parse_us") or 0.0)
                + float(s.get("proposal_rewrite_apply_us") or 0.0)
                + float(s.get("proposal_virtual_reference_tokenize_us") or 0.0)
                + float(s.get("proposal_transpld_index_build_us") or 0.0)
                for s in method_steps
            )
            / 1_000_000.0,
        }

    if baseline_tps is None:
        raise ValueError(f"baseline {baseline!r} not found")
    for row in methods.values():
        row["vs_baseline"] = row["tokens_per_sec"] / baseline_tps if baseline_tps else 0.0

    ranked = sorted(methods.items(), key=lambda kv: kv[1]["tokens_per_sec"], reverse=True)
    return {
        "baseline": baseline,
        "baseline_tokens_per_sec": baseline_tps,
        "ranked_methods": [{"method": method, **row} for method, row in ranked],
    }


def write_markdown(report: dict[str, Any], path: Path, *, top_k: int) -> None:
    lines = [
        "# VANTAGE-MV Train Grid",
        "",
        f"Baseline: `{report['baseline']}` at {_fmt(report['baseline_tokens_per_sec'], 2)} tok/s.",
        "",
        "## Top Methods",
        "",
        "| rank | method | tok/s | vs PLD | acc/verify | verify s | trans tasks | trans steps | zero-accept | frontier wins | branch wins | branch accepted | staged saved | neural steps | neural tokens | exact capped | setup s | skipped no-frontier | skipped margin | built/no-candidate |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(report["ranked_methods"][:top_k], start=1):
        reasons = row.get("route_reasons") or {}
        zero_rate = row["trans_zero_accept_rate"] * 100.0
        lines.append(
            "| "
            + " | ".join(
                [
                    str(i),
                    f"`{row['method']}`",
                    _fmt(row["tokens_per_sec"], 2),
                    _fmt(row["vs_baseline"], 4),
                    _fmt(row["accepted_per_verify"], 2),
                    _fmt(row["verify_s"], 2),
                    str(row["trans_tasks"]),
                    str(row["trans_steps"]),
                    f"{zero_rate:.1f}%",
                    str(reasons.get("trans_frontier_probe_wins", 0)),
                    str(reasons.get("trans_conflict_branch_wins", 0)),
                    str(row.get("branch_accepted_tokens", 0)),
                    str(row.get("staged_saved_tokens", 0)),
                    str(row.get("neural_steps", 0)),
                    str(row.get("neural_tokens", 0)),
                    str(row.get("exact_capped_steps", 0)),
                    _fmt(row["setup_s"], 3),
                    str(reasons.get("no_rewrite_frontier_signal", 0)),
                    str(reasons.get("trans_precheck_margin_impossible", 0)),
                    str(reasons.get("trans_view_built_no_candidate", 0)),
                ]
            )
            + " |"
        )
        if (
            row.get("rescue_steps")
            or row.get("patch_steps")
            or row.get("branch_steps")
            or row.get("staged_chunks")
            or row.get("neural_steps")
        ):
            lines.append(
                f"<!-- {row['method']} rescue_steps={row.get('rescue_steps', 0)} "
                f"rescue_accepted={row.get('rescue_accepted', 0)} "
                f"patch_steps={row.get('patch_steps', 0)} "
                f"patch_accepted={row.get('patch_accepted', 0)} "
                f"branch_steps={row.get('branch_steps', 0)} "
                f"transformed_branch_selected={row.get('transformed_branch_selected', 0)} "
                f"staged_chunks={row.get('staged_chunks', 0)} "
                f"staged_saved_tokens={row.get('staged_saved_tokens', 0)} "
                f"neural_steps={row.get('neural_steps', 0)} "
                f"neural_tokens={row.get('neural_tokens', 0)} -->"
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--baseline", default="blazedit_pld_w128_n10")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    report = analyze(args.eval_dir, baseline=args.baseline)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_markdown(report, args.out_md, top_k=args.top_k)


if __name__ == "__main__":
    main()
