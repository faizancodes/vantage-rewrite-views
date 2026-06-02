"""Aggregate steps.jsonl + aggregate.json into paper-ready tables.

Reads the artifacts produced by `scripts/run_eagle_eval.py`:
  - steps.jsonl   (one record per outer decode step, schema in decoder.StepRecord)
  - aggregate.json (per-method totals: tokens/sec, mean k, mean accepted, etc.)

Produces:
  - Method comparison table (Python and TS, with speedup vs vanilla)
  - AST node-type breakdown for ASTS-EAGLE method (Python and TS)
  - Speedup-vs-mean-chain-length data points (for plotting elsewhere)

No GPU. No external deps beyond stdlib. Run with the JSONL paths as args:

    python scripts/analyze_paper_results.py \\
        --python-steps    /tmp/eagle_eval_data/python_steps.jsonl \\
        --python-agg      /tmp/eagle_eval_data/python_aggregate.json \\
        --ts-steps        /tmp/eagle_eval_data/ts_steps.jsonl \\
        --ts-agg          /tmp/eagle_eval_data/ts_aggregate.json \\
        --output-md       /tmp/eagle_eval_data/paper_tables.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Method comparison
# ---------------------------------------------------------------------------


def method_comparison_table(agg: dict, language: str) -> str:
    """One-row-per-method table. Speedup is method tok/s / vanilla tok/s."""
    by_method = agg["by_method"]
    if "vanilla" not in by_method:
        raise ValueError("vanilla method missing — can't compute speedup")
    van_tps = by_method["vanilla"]["tokens_per_sec"]

    # Order: vanilla, fixed-k (sorted by k), then asts_eagle last (the headline)
    ordered = ["vanilla"]
    fixed = sorted(
        [m for m in by_method if m.startswith("eagle_k")],
        key=lambda m: int(m.replace("eagle_k", "")),
    )
    ordered += fixed
    if "asts_eagle" in by_method:
        ordered.append("asts_eagle")

    rows = []
    rows.append(f"### Method comparison — {language} (n={agg['meta']['n_problems']})\n")
    rows.append(
        "| Method | Tokens/sec | Mean k | Mean accepted candidates | Mean emitted | Accept rate | Speedup vs vanilla |"
    )
    rows.append("|--------|-----------:|-------:|-------------------------:|-------------:|------------:|-------------------:|")
    for m in ordered:
        d = by_method[m]
        tps = d["tokens_per_sec"]
        mean_k = d["mean_k_requested"]
        mean_acc = d["mean_accepted_drafts_per_step"]
        mean_emitted = d.get("mean_emitted_per_step", 1.0 if m == "vanilla" else mean_acc + 1)
        # Accept rate is per-draft (excluding the always-emit-1 vanilla "draft").
        if m == "vanilla":
            ar_str = "—"
        else:
            ar_str = f"{(mean_acc / max(1e-6, mean_k)):.3f}"
        speedup = tps / van_tps
        bold = "**" if m == "asts_eagle" else ""
        rows.append(
            f"| {bold}{m}{bold} | {bold}{tps:.2f}{bold} | {mean_k:.2f} | {mean_acc:.2f} "
            f"| {mean_emitted:.2f} | {ar_str} | {bold}{speedup:.3f}×{bold} |"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# AST node-type breakdown (ASTS-EAGLE only)
# ---------------------------------------------------------------------------


@dataclass
class NodeStats:
    n_steps: int = 0
    sum_k: int = 0
    sum_accepted: int = 0
    sum_emitted: int = 0
    sum_wall_us: float = 0.0


def node_type_breakdown(
    steps: list[dict], language: str, top_n: int = 15
) -> str:
    """Group ASTS-EAGLE steps by node_type. Show top N by frequency,
    roll the long tail into 'other'.
    """
    asts_steps = [s for s in steps if s["method"] == "asts_eagle"]
    if not asts_steps:
        return f"### AST node-type breakdown — {language}\n\n*No asts_eagle method in steps.jsonl.*"

    total_steps = len(asts_steps)
    total_emitted = sum(s["n_emitted"] for s in asts_steps)

    by_node: dict[str, NodeStats] = defaultdict(NodeStats)
    for s in asts_steps:
        nt = s.get("node_type") or "default"
        ns = by_node[nt]
        ns.n_steps += 1
        ns.sum_k += s["k"]
        ns.sum_accepted += s["n_accepted_drafts"]
        ns.sum_emitted += s["n_emitted"]
        ns.sum_wall_us += s["wall_us"]

    sorted_nodes = sorted(by_node.items(), key=lambda kv: -kv[1].n_steps)

    rows = []
    rows.append(f"### AST node-type breakdown (ASTS-EAGLE) — {language}\n")
    rows.append(
        "| Node type | Steps | % steps | % tokens | Mean k | Mean accepted | Accept rate | Wasted drafts/step |"
    )
    rows.append(
        "|-----------|------:|--------:|---------:|-------:|--------------:|------------:|-------------------:|"
    )

    head = sorted_nodes[:top_n]
    tail = sorted_nodes[top_n:]

    def fmt_row(nt: str, ns: NodeStats) -> str:
        mean_k = ns.sum_k / ns.n_steps
        mean_acc = ns.sum_accepted / ns.n_steps
        accept_rate = mean_acc / max(1e-6, mean_k)
        wasted = mean_k - mean_acc
        pct_steps = 100 * ns.n_steps / total_steps
        pct_tokens = 100 * ns.sum_emitted / max(1, total_emitted)
        return (
            f"| `{nt}` | {ns.n_steps} | {pct_steps:.1f}% | {pct_tokens:.1f}% "
            f"| {mean_k:.2f} | {mean_acc:.2f} | {accept_rate:.3f} | {wasted:.2f} |"
        )

    for nt, ns in head:
        rows.append(fmt_row(nt, ns))

    if tail:
        rolled = NodeStats()
        for _, ns in tail:
            rolled.n_steps += ns.n_steps
            rolled.sum_k += ns.sum_k
            rolled.sum_accepted += ns.sum_accepted
            rolled.sum_emitted += ns.sum_emitted
            rolled.sum_wall_us += ns.sum_wall_us
        rolled_label = f"other ({len(tail)} types)"
        rows.append(fmt_row(rolled_label, rolled))

    rows.append("")
    rows.append(
        f"*Total: {total_steps:,} steps, {total_emitted:,} emitted tokens, "
        f"{len(by_node)} distinct node types.*"
    )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Speedup vs mean-chain-length data (for plotting)
# ---------------------------------------------------------------------------


def speedup_vs_chain_length(agg: dict, language: str) -> dict[str, Any]:
    """Emit the (mean_k, tokens_per_sec, speedup) tuples per method.
    Useful for an external plot (matplotlib/tikz)."""
    by_method = agg["by_method"]
    van_tps = by_method["vanilla"]["tokens_per_sec"]
    points = []
    for m, d in by_method.items():
        points.append({
            "method": m,
            "mean_k": d["mean_k_requested"],
            "tokens_per_sec": d["tokens_per_sec"],
            "speedup": d["tokens_per_sec"] / van_tps,
            "mean_accepted_per_step": d["mean_accepted_drafts_per_step"],
        })
    points.sort(key=lambda p: p["mean_k"])
    return {"language": language, "points": points}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def merge_aggregate(*aggs: dict) -> dict:
    """Merge multiple aggregate.json dicts. The first is the base; each
    subsequent contributes additional methods (no overwrite of existing keys).
    Meta is taken from the first.
    """
    base = {"meta": dict(aggs[0].get("meta", {})), "by_method": {}, "by_node_type": {}}
    for a in aggs:
        for m, d in a.get("by_method", {}).items():
            if m not in base["by_method"]:
                base["by_method"][m] = d
        for nt, d in a.get("by_node_type", {}).items():
            if nt not in base["by_node_type"]:
                base["by_node_type"][nt] = d
    return base


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--python-steps", required=True)
    p.add_argument("--python-agg", required=True)
    p.add_argument("--ts-steps", required=True)
    p.add_argument("--ts-agg", required=True)
    p.add_argument(
        "--python-extra-aggs",
        nargs="*",
        default=[],
        help="Additional aggregate.json files for Python (e.g. k=2,6 ablation).",
    )
    p.add_argument(
        "--ts-extra-aggs",
        nargs="*",
        default=[],
        help="Additional aggregate.json files for TypeScript.",
    )
    p.add_argument("--output-md", default="/tmp/eagle_eval_data/paper_tables.md")
    p.add_argument("--output-json", default="/tmp/eagle_eval_data/paper_data.json")
    p.add_argument("--top-n", type=int, default=15)
    args = p.parse_args()

    py_steps = load_jsonl(args.python_steps)
    py_agg_main = load_json(args.python_agg)
    py_extras = [load_json(p) for p in args.python_extra_aggs]
    py_agg = merge_aggregate(py_agg_main, *py_extras)

    ts_steps = load_jsonl(args.ts_steps)
    ts_agg_main = load_json(args.ts_agg)
    ts_extras = [load_json(p) for p in args.ts_extra_aggs]
    ts_agg = merge_aggregate(ts_agg_main, *ts_extras)

    sections: list[str] = []
    sections.append("# ASTS-Spec — Paper Result Tables\n")
    sections.append(
        "_Generated by `scripts/analyze_paper_results.py` from "
        "`steps.jsonl` + `aggregate.json` produced by `scripts/run_eagle_eval.py`._\n"
    )

    sections.append("## 1. Method comparison\n")
    sections.append(method_comparison_table(py_agg, "Python (HumanEval)"))
    sections.append("")
    sections.append(method_comparison_table(ts_agg, "TypeScript (MultiPL-E)"))
    sections.append("")

    sections.append("## 2. AST node-type breakdown (ASTS-EAGLE)\n")
    sections.append(node_type_breakdown(py_steps, "Python", args.top_n))
    sections.append("")
    sections.append(node_type_breakdown(ts_steps, "TypeScript", args.top_n))
    sections.append("")

    sections.append("## 3. Speedup-vs-chain-length data (for Pareto plot)\n")
    for lang_label, agg_obj in [("Python", py_agg), ("TypeScript", ts_agg)]:
        pts = speedup_vs_chain_length(agg_obj, lang_label)["points"]
        sections.append(f"### {lang_label}\n")
        sections.append("| Method | Mean k | Tokens/sec | Speedup |")
        sections.append("|--------|-------:|-----------:|--------:|")
        for p in pts:
            bold = "**" if p["method"] == "asts_eagle" else ""
            sections.append(
                f"| {bold}{p['method']}{bold} | {p['mean_k']:.2f} | "
                f"{p['tokens_per_sec']:.2f} | {bold}{p['speedup']:.3f}×{bold} |"
            )
        sections.append("")

    md = "\n".join(sections)

    plot_data = {
        "python": speedup_vs_chain_length(py_agg, "Python"),
        "typescript": speedup_vs_chain_length(ts_agg, "TypeScript"),
    }

    with open(args.output_md, "w") as f:
        f.write(md)
    with open(args.output_json, "w") as f:
        json.dump(plot_data, f, indent=2)

    print(md)
    print()
    print(f"[wrote] {args.output_md}")
    print(f"[wrote] {args.output_json}")


if __name__ == "__main__":
    main()
