"""Summarize cheap code-proposer traces from run_eagle_eval.py."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load_steps(path: Path, methods: set[str] | None) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if methods and row.get("method") not in methods:
                continue
            out.append(row)
    return out


def _summarize_method(rows: list[dict]) -> dict:
    n = len(rows)
    total_wall = sum(r.get("wall_us", 0.0) for r in rows)
    total_emit = sum(r.get("n_emitted", 0) for r in rows)
    initial_prefill = sum(r.get("target_prefill_us", 0.0) for r in rows if r.get("step") == 0)
    proposal_rows = [r for r in rows if r.get("proposal_kind")]
    fallback_rows = [r for r in rows if not r.get("proposal_kind")]
    nonroot_values = [r.get("n_accepted_nonroot_drafts", 0) or 0 for r in rows]
    out = {
        "n_steps": n,
        "n_emitted": total_emit,
        "tokens_per_sec": total_emit / (total_wall / 1e6) if total_wall else 0.0,
        "decode_tokens_per_sec": (
            total_emit / ((total_wall - initial_prefill) / 1e6)
            if total_wall > initial_prefill
            else 0.0
        ),
        "proposal_hit_rate": len(proposal_rows) / n if n else 0.0,
        "fallback_rate": len(fallback_rows) / n if n else 0.0,
        "mean_proposal_us": (
            sum(r.get("proposal_us", 0.0) for r in rows) / n if n else 0.0
        ),
        "mean_accepted_nonroot": (
            sum(nonroot_values) / n if n else 0.0
        ),
        "p_accept_ge_1_nonroot": (
            sum(1 for v in nonroot_values if v >= 1) / n if n else 0.0
        ),
        "p_accept_ge_2_nonroot": (
            sum(1 for v in nonroot_values if v >= 2) / n if n else 0.0
        ),
        "mean_emitted_per_step": total_emit / n if n else 0.0,
        "hit_max_new_tokens_steps": sum(1 for r in rows if r.get("hit_max_new_tokens")),
    }
    by_kind: dict[str, dict] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in proposal_rows:
        grouped[str(r.get("proposal_kind"))].append(r)
    for kind, kind_rows in grouped.items():
        kn = len(kind_rows)
        by_kind[kind] = {
            "n": kn,
            "share_steps": kn / n if n else 0.0,
            "mean_match_len": sum(r.get("proposal_match_len", 0) or 0 for r in kind_rows) / kn,
            "mean_proposal_tokens": sum(r.get("proposal_tokens", 0) or 0 for r in kind_rows)
            / kn,
            "mean_tree_nodes": sum(r.get("proposal_tree_nodes", 0) or 0 for r in kind_rows)
            / kn,
            "mean_tree_candidates": sum(
                r.get("proposal_tree_candidates", 0) or 0 for r in kind_rows
            )
            / kn,
            "tree_branch_selected_rate": sum(
                1 for r in kind_rows if r.get("proposal_tree_branch_selected") is not None
            )
            / kn,
            "mean_tree_branch_depth": (
                sum(r.get("proposal_tree_branch_depth", 0) or 0 for r in kind_rows)
                / max(1, sum(1 for r in kind_rows if r.get("proposal_tree_branch_depth") is not None))
            ),
            "mean_accepted_nonroot": sum(
                r.get("n_accepted_nonroot_drafts", 0) or 0 for r in kind_rows
            )
            / kn,
            "mean_accepted_nonroot_per_step": sum(
                r.get("n_accepted_nonroot_drafts", 0) or 0 for r in kind_rows
            )
            / n if n else 0.0,
            "p_accept_ge_1_nonroot_given_hit": sum(
                1 for r in kind_rows if (r.get("n_accepted_nonroot_drafts", 0) or 0) >= 1
            )
            / kn,
            "p_accept_ge_2_nonroot_given_hit": sum(
                1 for r in kind_rows if (r.get("n_accepted_nonroot_drafts", 0) or 0) >= 2
            )
            / kn,
            "mean_proposal_us": sum(r.get("proposal_us", 0.0) for r in kind_rows) / kn,
            "mean_wall_us": sum(r.get("wall_us", 0.0) for r in kind_rows) / kn,
        }
    out["by_proposal_kind"] = by_kind

    by_context: dict[str, dict] = {}
    context_groups: dict[str, list[dict]] = defaultdict(list)
    for r in proposal_rows:
        context_groups[str(r.get("deepest_type") or r.get("node_type") or "default")].append(r)
    for ctx, ctx_rows in sorted(context_groups.items(), key=lambda item: -len(item[1]))[:30]:
        cn = len(ctx_rows)
        by_context[ctx] = {
            "n": cn,
            "share_proposal_steps": cn / max(1, len(proposal_rows)),
            "mean_accepted_nonroot": sum(
                r.get("n_accepted_nonroot_drafts", 0) or 0 for r in ctx_rows
            )
            / cn,
        }
    out["top_contexts"] = by_context
    return out


def _write_md(report: dict, path: Path) -> str:
    lines = ["# Code Proposer Analysis", ""]
    for method, summary in report["methods"].items():
        lines.append(f"## {method}")
        lines.append("")
        lines.append(
            f"- steps: {summary['n_steps']}  emitted: {summary['n_emitted']}  "
            f"tok/s: {summary['tokens_per_sec']:.2f}  "
            f"decode tok/s: {summary['decode_tokens_per_sec']:.2f}"
        )
        lines.append(
            f"- proposal hit rate: {summary['proposal_hit_rate']:.2%}  "
            f"fallback rate: {summary['fallback_rate']:.2%}  "
            f"mean proposal cost: {summary['mean_proposal_us']:.1f} us"
        )
        if "target_forward_reduction_vs_vanilla" in summary:
            lines.append(
                f"- target-forward step reduction vs vanilla: "
                f"{summary['target_forward_reduction_vs_vanilla']:.2%}"
            )
        lines.append(
            f"- P(nonroot>=1): {summary['p_accept_ge_1_nonroot']:.2%}  "
            f"P(nonroot>=2): {summary['p_accept_ge_2_nonroot']:.2%}  "
            f"mean emitted/step: {summary['mean_emitted_per_step']:.2f}"
        )
        if summary["by_proposal_kind"]:
            lines.append("")
            lines.append("| kind | n | share | match | prop tok | tree nodes | tree cands | acc/hit | acc/step | P>=1 | P>=2 | proposal us |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for kind, row in summary["by_proposal_kind"].items():
                lines.append(
                    f"| {kind} | {row['n']} | {row['share_steps']:.2%} | "
                    f"{row['mean_match_len']:.2f} | "
                    f"{row['mean_proposal_tokens']:.2f} | "
                    f"{row['mean_tree_nodes']:.2f} | "
                    f"{row['mean_tree_candidates']:.2f} | "
                    f"{row['mean_accepted_nonroot']:.2f} | "
                    f"{row['mean_accepted_nonroot_per_step']:.2f} | "
                    f"{row['p_accept_ge_1_nonroot_given_hit']:.2%} | "
                    f"{row['p_accept_ge_2_nonroot_given_hit']:.2%} | "
                    f"{row['mean_proposal_us']:.1f} |"
                )
        lines.append("")
    md = "\n".join(lines)
    path.write_text(md)
    return md


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", required=True)
    p.add_argument("--methods", default="")
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    args = p.parse_args()

    methods = {m.strip() for m in args.methods.split(",") if m.strip()} or None
    steps = _load_steps(Path(args.steps), methods)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in steps:
        grouped[row["method"]].append(row)
    report = {
        "schema": "asts-spec/code_proposer_analysis/v1",
        "steps": args.steps,
        "methods": {method: _summarize_method(rows) for method, rows in grouped.items()},
    }
    vanilla_steps = report["methods"].get("vanilla", {}).get("n_steps")
    if vanilla_steps:
        for summary in report["methods"].values():
            summary["target_forward_reduction_vs_vanilla"] = (
                (vanilla_steps - summary["n_steps"]) / vanilla_steps
            )
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    print(_write_md(report, Path(args.output_md)))


if __name__ == "__main__":
    main()
