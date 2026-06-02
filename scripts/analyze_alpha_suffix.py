"""Analyze Scoped alpha-suffix proposal traces.

The report isolates genuinely renamed canonical matches from exact suffix hits
and summarizes their acceptance/overhead by method, match kind, and AST context.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def _load_steps(path: Path, methods: set[str] | None) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if methods and row.get("method") not in methods:
                continue
            rows.append(row)
    return rows


def _avg(rows: list[dict], key: str) -> float:
    vals = [r.get(key, 0.0) or 0.0 for r in rows]
    return mean(vals) if vals else 0.0


def _summary(rows: list[dict], denom: int) -> dict:
    n = len(rows)
    nonroot = [r.get("n_accepted_nonroot_drafts", 0) or 0 for r in rows]
    return {
        "n": n,
        "share_steps": n / denom if denom else 0.0,
        "mean_canonical_match_len": _avg(rows, "proposal_canonical_match_len"),
        "mean_exact_match_len": _avg(rows, "proposal_match_len"),
        "mean_proposal_tokens": _avg(rows, "proposal_tokens"),
        "mean_accepted_nonroot_per_hit": mean(nonroot) if nonroot else 0.0,
        "mean_accepted_nonroot_per_step": sum(nonroot) / denom if denom else 0.0,
        "p_accept_ge_1_nonroot": sum(1 for v in nonroot if v >= 1) / n if n else 0.0,
        "p_accept_ge_2_nonroot": sum(1 for v in nonroot if v >= 2) / n if n else 0.0,
        "zero_nonroot_accept_rate": sum(1 for v in nonroot if v == 0) / n if n else 0.0,
        "mean_substitution_count": _avg(rows, "proposal_substitution_count"),
        "mean_scope_fill_count": _avg(rows, "proposal_scope_fill_count"),
        "stopped_on_unmapped_rate": (
            sum(1 for r in rows if r.get("proposal_stopped_on_unmapped")) / n if n else 0.0
        ),
        "mean_proposal_us": _avg(rows, "proposal_us"),
        "mean_wall_us": _avg(rows, "wall_us"),
    }


def _method_report(rows: list[dict]) -> dict:
    denom = len(rows)
    proposal_rows = [r for r in rows if r.get("proposal_kind")]
    alpha_rows = [
        r
        for r in proposal_rows
        if str(r.get("proposal_kind") or "").startswith("alpha")
        or str(r.get("proposal_match_kind") or "").startswith("alpha")
    ]
    exact_rows = [
        r
        for r in proposal_rows
        if r.get("proposal_kind") in {"local_suffix", "ngram"}
        or r.get("proposal_match_kind") == "exact"
    ]

    by_kind: dict[str, dict] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in proposal_rows:
        key = str(row.get("proposal_match_kind") or row.get("proposal_kind") or "unknown")
        grouped[key].append(row)
    for key, group in sorted(grouped.items()):
        by_kind[key] = _summary(group, denom)

    by_context: dict[str, dict] = {}
    context_groups: dict[str, list[dict]] = defaultdict(list)
    for row in alpha_rows:
        context = str(row.get("deepest_type") or row.get("node_type") or "default")
        context_groups[context].append(row)
    for key, group in sorted(context_groups.items(), key=lambda item: -len(item[1]))[:30]:
        by_context[key] = _summary(group, denom)

    return {
        "n_steps": denom,
        "n_proposal_steps": len(proposal_rows),
        "proposal_hit_rate": len(proposal_rows) / denom if denom else 0.0,
        "alpha_hit_rate": len(alpha_rows) / denom if denom else 0.0,
        "exact_hit_rate": len(exact_rows) / denom if denom else 0.0,
        "mean_accepted_nonroot_all_steps": _avg(rows, "n_accepted_nonroot_drafts"),
        "mean_proposal_us_all_steps": _avg(rows, "proposal_us"),
        "alpha": _summary(alpha_rows, denom),
        "exact": _summary(exact_rows, denom),
        "by_match_kind": by_kind,
        "top_alpha_contexts": by_context,
    }


def _write_md(report: dict, path: Path) -> str:
    lines = ["# Scoped Alpha-Suffix Analysis", ""]
    for method, summary in report["methods"].items():
        lines.append(f"## {method}")
        lines.append("")
        lines.append(
            f"- steps: {summary['n_steps']}  proposal hit: {summary['proposal_hit_rate']:.2%}  "
            f"alpha hit: {summary['alpha_hit_rate']:.2%}  exact hit: {summary['exact_hit_rate']:.2%}"
        )
        lines.append(
            f"- mean accepted non-root/step: {summary['mean_accepted_nonroot_all_steps']:.3f}  "
            f"mean proposal cost: {summary['mean_proposal_us_all_steps']:.1f} us"
        )
        lines.append("")
        lines.append("| match kind | n | share | canon match | prop tok | acc/hit | acc/step | P>=1 | zero | subst | scope | prop us |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for kind, row in summary["by_match_kind"].items():
            lines.append(
                f"| {kind} | {row['n']} | {row['share_steps']:.2%} | "
                f"{row['mean_canonical_match_len']:.2f} | "
                f"{row['mean_proposal_tokens']:.2f} | "
                f"{row['mean_accepted_nonroot_per_hit']:.2f} | "
                f"{row['mean_accepted_nonroot_per_step']:.3f} | "
                f"{row['p_accept_ge_1_nonroot']:.2%} | "
                f"{row['zero_nonroot_accept_rate']:.2%} | "
                f"{row['mean_substitution_count']:.2f} | "
                f"{row['mean_scope_fill_count']:.2f} | "
                f"{row['mean_proposal_us']:.1f} |"
            )
        if summary["top_alpha_contexts"]:
            lines.append("")
            lines.append("| alpha context | n | share | acc/hit | zero | prop us |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
            for context, row in summary["top_alpha_contexts"].items():
                lines.append(
                    f"| {context} | {row['n']} | {row['share_steps']:.2%} | "
                    f"{row['mean_accepted_nonroot_per_hit']:.2f} | "
                    f"{row['zero_nonroot_accept_rate']:.2%} | "
                    f"{row['mean_proposal_us']:.1f} |"
                )
        lines.append("")
    md = "\n".join(lines)
    path.write_text(md)
    return md


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", required=True)
    parser.add_argument("--methods", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    methods = {m.strip() for m in args.methods.split(",") if m.strip()} or None
    rows = _load_steps(Path(args.steps), methods)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("method"))].append(row)
    report = {
        "schema": "asts-spec/alpha_suffix_analysis/v1",
        "steps": args.steps,
        "methods": {method: _method_report(method_rows) for method, method_rows in grouped.items()},
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    print(_write_md(report, Path(args.output_md)))


if __name__ == "__main__":
    main()
