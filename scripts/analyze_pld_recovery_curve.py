"""Summarize how much rewrite-anchor recovers after PLD misses."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyze_anchor_pld_overlap import analyze as analyze_anchor_overlap  # noqa: E402


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _step_groups(steps: list[dict[str, Any]], method: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step in steps:
        if step.get("method") == method:
            out[str(step.get("task_id"))].append(step)
    return out


def _pld_stats(steps: list[dict[str, Any]], method: str) -> dict[str, Any]:
    method_steps = [s for s in steps if s.get("method") == method]
    hits = [
        s
        for s in method_steps
        if int(s.get("blazedit_pld_proposed") or s.get("proposal_tokens") or 0) > 0
    ]
    accepted = sum(int(s.get("blazedit_pld_accepted") or s.get("n_accepted_nonroot_drafts") or 0) for s in hits)
    return {
        "steps": len(method_steps),
        "hits": len(hits),
        "hit_rate": len(hits) / len(method_steps) if method_steps else 0.0,
        "accepted_tokens": accepted,
        "accepted_per_hit": accepted / len(hits) if hits else 0.0,
        "accepted_per_step": accepted / len(method_steps) if method_steps else 0.0,
    }


def _anchor_stats(steps: list[dict[str, Any]], method: str) -> dict[str, Any]:
    method_steps = [s for s in steps if s.get("method") == method]
    hits = [
        s
        for s in method_steps
        if s.get("proposal_match_kind")
        in {
            "edit_anchor",
            "rewrite_anchor",
            "bidir",
            "vref",
            "oracle",
            "transpld_bidir",
            "transpld_vref",
            "transpld_bidir_inferred",
            "routed_transpld_bidir",
            "routed_transpld_vref",
            "cursor",
            "precomputed_transpld_compete",
        }
        or s.get("proposal_kind")
        in {
            "edit_anchor",
            "rewrite_anchor_pld",
            "edit_anchor_pld",
            "rewrite_norm_pld",
            "transpld",
            "transpld_infer",
            "transpld_compound",
            "transpld_cursor",
            "routed_transpld",
        }
    ]
    accepted = sum(int(s.get("n_accepted_nonroot_drafts") or 0) for s in hits)
    zero = sum(1 for s in hits if int(s.get("n_accepted_nonroot_drafts") or 0) <= 0)
    return {
        "steps": len(method_steps),
        "hits": len(hits),
        "hit_rate": len(hits) / len(method_steps) if method_steps else 0.0,
        "accepted_tokens": accepted,
        "accepted_per_hit": accepted / len(hits) if hits else 0.0,
        "accepted_per_step": accepted / len(method_steps) if method_steps else 0.0,
        "zero_accept_hits": zero,
        "zero_accept_rate": zero / len(hits) if hits else 0.0,
    }


def _task_speedups(completions: list[dict[str, Any]], pld_method: str, anchor_method: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in completions:
        outputs = row.get("outputs") or {}
        if pld_method not in outputs or anchor_method not in outputs:
            continue
        pld = outputs[pld_method]
        anchor = outputs[anchor_method]
        pld_tps = float(pld.get("n_new_tokens") or 0) / (float(pld.get("wall_us") or 1) / 1e6)
        anchor_tps = float(anchor.get("n_new_tokens") or 0) / (float(anchor.get("wall_us") or 1) / 1e6)
        out[str(row.get("task_id"))] = anchor_tps / pld_tps if pld_tps else 0.0
    return out


def analyze(
    completions: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    *,
    pld_method: str,
    anchor_method: str,
    tokenizer_name: str,
    pld_window: int,
    pld_ngram: int,
    rewrite_pld_method: str = "",
) -> dict[str, Any]:
    overlap = analyze_anchor_overlap(
        completions=completions,
        steps=steps,
        method=anchor_method,
        tokenizer_name=tokenizer_name,
        pld_window=pld_window,
        pld_ngram=pld_ngram,
        rewrite_pld_method=rewrite_pld_method,
    )
    speedups = _task_speedups(completions, pld_method, anchor_method)
    return {
        "schema": "asts-spec/pld-recovery/v1",
        "pld_method": pld_method,
        "anchor_method": anchor_method,
        "pld": _pld_stats(steps, pld_method),
        "anchor": _anchor_stats(steps, anchor_method),
        "overlap": overlap,
        "mean_speedup_vs_pld": (
            sum(speedups.values()) / len(speedups) if speedups else 0.0
        ),
        "task_speedups": speedups,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    pld = report["pld"]
    anchor = report["anchor"]
    overlap = report["overlap"]
    totals = overlap.get("totals") or {}
    frac = overlap.get("fractions") or {}
    lines = [
        "# PLD Miss Recovery",
        "",
        f"PLD: `{report['pld_method']}`",
        f"Anchor: `{report['anchor_method']}`",
        f"Rewrite-normalized PLD: `{overlap.get('rewrite_pld_method') or 'not computed'}`",
        f"Mean task speedup Anchor/PLD: `{report['mean_speedup_vs_pld']:.3f}`",
        "",
        "| Metric | PLD | Anchor |",
        "|---|---:|---:|",
        f"| Hit rate | {100 * pld['hit_rate']:.1f}% | {100 * anchor['hit_rate']:.1f}% |",
        f"| Accepted tokens / hit | {pld['accepted_per_hit']:.2f} | {anchor['accepted_per_hit']:.2f} |",
        f"| Accepted tokens / step | {pld['accepted_per_step']:.2f} | {anchor['accepted_per_step']:.2f} |",
        f"| Zero-accept hit rate | - | {100 * anchor['zero_accept_rate']:.1f}% |",
        "",
        "| Overlap metric | Value |",
        "|---|---:|",
        f"| Anchor accepted non-root tokens | {totals.get('anchor_accepted_nonroot', 0):.0f} |",
        f"| Accepted after unrooted PLD miss | {totals.get('after_unrooted_pld_miss', 0):.0f} |",
        f"| Accepted after rooted PLD miss | {totals.get('after_rooted_pld_miss', 0):.0f} |",
        f"| Accepted after rewrite-normalized PLD miss | {totals.get('after_rewrite_norm_pld_miss', 0):.0f} |",
        f"| Fraction after unrooted PLD miss | {100 * frac.get('after_unrooted_pld_miss', 0.0):.1f}% |",
        f"| Fraction after rooted PLD miss | {100 * frac.get('after_rooted_pld_miss', 0.0):.1f}% |",
        f"| Fraction after rewrite-normalized PLD miss | {100 * frac.get('after_rewrite_norm_pld_miss', 0.0):.1f}% |",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--completions", required=True)
    p.add_argument("--steps", required=True)
    p.add_argument("--pld-method", required=True)
    p.add_argument("--anchor-method", required=True)
    p.add_argument("--target-tokenizer", default="Qwen/Qwen2.5-Coder-7B")
    p.add_argument("--pld-window", type=int, default=128)
    p.add_argument("--pld-ngram", type=int, default=10)
    p.add_argument("--rewrite-pld-method", default="")
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    args = p.parse_args()
    report = analyze(
        _load_jsonl(Path(args.completions)),
        _load_jsonl(Path(args.steps)),
        pld_method=args.pld_method,
        anchor_method=args.anchor_method,
        tokenizer_name=args.target_tokenizer,
        pld_window=args.pld_window,
        pld_ngram=args.pld_ngram,
        rewrite_pld_method=args.rewrite_pld_method,
    )
    Path(args.output_json).write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, Path(args.output_md))


if __name__ == "__main__":
    main()
