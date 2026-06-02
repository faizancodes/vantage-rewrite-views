#!/usr/bin/env python3
"""Diagnostics for the real-commit TransPLD/PLD comparison.

The script is intentionally artifact-only: it consumes existing completion and
step traces and produces the six reviewer-requested diagnostics without any GPU
work.
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


_FENCE_RE = re.compile(r"^\s*```(?:python|py)?\s*\n(?P<body>.*?)(?:\n```)?\s*$", re.S)
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _strip_fence(text: str) -> str:
    text = text.strip()
    match = _FENCE_RE.match(text)
    if match:
        return match.group("body").strip()
    return text


def _syntax_ok(text: str) -> bool:
    try:
        ast.parse(_strip_fence(text))
        return True
    except SyntaxError:
        return False


def _tps(row: dict[str, Any], method: str) -> float:
    out = (row.get("outputs") or {}).get(method) or {}
    wall_us = float(out.get("wall_us") or 0.0)
    tokens = int(out.get("n_new_tokens") or 0)
    return tokens / (wall_us / 1e6) if wall_us > 0 else 0.0


def _aggregate_tps(rows: list[dict[str, Any]], method: str) -> float:
    tokens = 0
    wall_us = 0.0
    for row in rows:
        out = (row.get("outputs") or {}).get(method) or {}
        tokens += int(out.get("n_new_tokens") or 0)
        wall_us += float(out.get("wall_us") or 0.0)
    return tokens / (wall_us / 1e6) if wall_us > 0 else 0.0


def _method_text(row: dict[str, Any], method: str) -> str:
    out = (row.get("outputs") or {}).get(method) or {}
    return str(out.get("text") or out.get("raw_text") or "")


def _pairs(row: dict[str, Any]) -> dict[str, str]:
    meta = row.get("metadata") or {}
    pairs = row.get("rewrite_pairs") or meta.get("rewrite_pairs") or {}
    if isinstance(pairs, dict):
        return {str(k): str(v) for k, v in pairs.items() if str(k) and str(v)}
    if isinstance(pairs, list):
        out: dict[str, str] = {}
        for item in pairs:
            if isinstance(item, dict):
                old = item.get("old")
                new = item.get("new")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                old, new = item[0], item[1]
            else:
                continue
            if old and new:
                out[str(old)] = str(new)
        return out
    return {}


def _term_count(text: str, term: str) -> int:
    if not term:
        return 0
    if _IDENT_RE.match(term):
        return len(re.findall(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", text))
    if "." in term and all(_IDENT_RE.match(part) for part in term.split(".") if part):
        return len(
            re.findall(
                rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])",
                text,
            )
        )
    return text.count(term)


def _adoption(row: dict[str, Any], method: str = "vanilla") -> dict[str, Any]:
    text = _strip_fence(_method_text(row, method))
    old_hits = 0
    new_hits = 0
    adopted_pairs = 0
    pair_rows: list[dict[str, Any]] = []
    for old, new in _pairs(row).items():
        old_n = _term_count(text, old)
        new_n = _term_count(text, new)
        old_hits += old_n
        new_hits += new_n
        adopted = new_n > 0 and old_n == 0
        adopted_pairs += int(adopted)
        pair_rows.append({"old": old, "new": new, "old_hits": old_n, "new_hits": new_n, "adopted": adopted})
    denom = old_hits + new_hits
    adoption_fraction = new_hits / denom if denom else 0.0
    pairs = _pairs(row)
    full_pair_adoption = adopted_pairs / len(pairs) if pairs else 0.0
    return {
        "old_hits": old_hits,
        "new_hits": new_hits,
        "adoption_fraction": adoption_fraction,
        "full_pair_adoption": full_pair_adoption,
        "pair_rows": pair_rows,
    }


def _target_match(row: dict[str, Any], method: str) -> bool:
    return _strip_fence(_method_text(row, method)) == _strip_fence(str(row.get("deterministic_target") or ""))


def _similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(a=a, b=b).ratio()


def _bootstrap_ratio(
    rows: list[dict[str, Any]],
    method: str,
    baseline: str,
    *,
    n_boot: int = 3000,
    seed: int = 17,
) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "ratio": 0.0, "ci95": [0.0, 0.0], "p_gt_1": 0.0}
    rng = random.Random(seed)
    ratios: list[float] = []
    for _ in range(n_boot):
        sample = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
        base = _aggregate_tps(sample, baseline)
        val = _aggregate_tps(sample, method)
        ratios.append(val / base if base else 0.0)
    ratios.sort()
    base = _aggregate_tps(rows, baseline)
    val = _aggregate_tps(rows, method)
    return {
        "n": len(rows),
        "ratio": val / base if base else 0.0,
        "ci95": [ratios[int(0.025 * (len(ratios) - 1))], ratios[int(0.975 * (len(ratios) - 1))]],
        "p_gt_1": sum(1 for r in ratios if r > 1.0) / len(ratios),
    }


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def _histogram_svg(values: list[float], path: Path) -> None:
    bins = [0.0, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0]
    counts = [0] * (len(bins) - 1)
    for value in values:
        for i in range(len(bins) - 1):
            if bins[i] <= value < bins[i + 1] or (i == len(bins) - 2 and value >= bins[i + 1]):
                counts[i] += 1
                break
    width, height = 900, 480
    ml, mr, mt, mb = 70, 30, 50, 80
    pw, ph = width - ml - mr, height - mt - mb
    max_count = max(counts + [1])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{ml}" y="28" font-size="19" font-weight="700">Per-task VANTAGE / PLD ratio histogram</text>',
        f'<line x1="{ml}" y1="{mt+ph}" x2="{width-mr}" y2="{mt+ph}" stroke="#374151"/>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#374151"/>',
    ]
    bar_w = pw / len(counts)
    for i, count in enumerate(counts):
        x = ml + i * bar_w + 6
        h = (count / max_count) * ph
        y = mt + ph - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-12:.1f}" height="{h:.1f}" fill="#0f766e" opacity="0.82"/>')
        parts.append(f'<text x="{x + (bar_w-12)/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-size="11">{count}</text>')
        label = f"{bins[i]:.2g}-{bins[i+1]:.2g}" if i < len(counts)-1 else f">={bins[i]:.2g}"
        parts.append(f'<text x="{x + (bar_w-12)/2:.1f}" y="{mt+ph+20}" text-anchor="middle" font-size="11">{html.escape(label)}</text>')
    x1 = ml + (1.0 - bins[0]) / (bins[-1] - bins[0]) * pw
    parts.append(f'<line x1="{x1:.1f}" y1="{mt}" x2="{x1:.1f}" y2="{mt+ph}" stroke="#b91c1c" stroke-dasharray="4 4"/>')
    parts.append(f'<text x="{ml+pw/2}" y="{height-24}" text-anchor="middle" font-size="14">Per-task throughput ratio</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def _scatter_svg(rows: list[dict[str, Any]], path: Path) -> None:
    width, height = 900, 560
    ml, mr, mt, mb = 75, 35, 45, 72
    pw, ph = width - ml - mr, height - mt - mb
    max_y = max([1.5] + [r["ratio"] for r in rows])
    max_y = min(3.0, max_y * 1.08)

    def xs(v: float) -> float:
        return ml + max(0, min(1, v)) * pw

    def ys(v: float) -> float:
        return mt + ph - max(0, min(max_y, v)) / max_y * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{ml}" y="28" font-size="19" font-weight="700">Real commits: rewrite adoption vs VANTAGE / PLD</text>',
        f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="#d1d5db"/>',
        f'<line x1="{ml}" y1="{ys(1):.1f}" x2="{ml+pw}" y2="{ys(1):.1f}" stroke="#6b7280" stroke-dasharray="5 5"/>',
    ]
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        x = xs(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{mt+ph}" x2="{x:.1f}" y2="{mt+ph+6}" stroke="#6b7280"/>')
        parts.append(f'<text x="{x:.1f}" y="{mt+ph+24}" text-anchor="middle" font-size="12">{tick:.2g}</text>')
    for tick in [0, 0.5, 1, 1.5, 2, 2.5, 3]:
        if tick > max_y:
            continue
        y = ys(tick)
        parts.append(f'<line x1="{ml-6}" y1="{y:.1f}" x2="{ml}" y2="{y:.1f}" stroke="#6b7280"/>')
        parts.append(f'<text x="{ml-10}" y="{y+4:.1f}" text-anchor="end" font-size="12">{tick:.1f}</text>')
    xvals = [r["adoption_fraction"] for r in rows]
    yvals = [r["ratio"] for r in rows]
    if len(set(xvals)) > 1:
        mx = sum(xvals) / len(xvals)
        my = sum(yvals) / len(yvals)
        denom = sum((x - mx) ** 2 for x in xvals)
        slope = sum((x - mx) * (y - my) for x, y in zip(xvals, yvals)) / denom
        intercept = my - slope * mx
        parts.append(
            f'<line x1="{xs(0):.1f}" y1="{ys(intercept):.1f}" x2="{xs(1):.1f}" y2="{ys(intercept+slope):.1f}" '
            'stroke="#111827" stroke-width="2.2" opacity="0.75"/>'
        )
        parts.append(f'<text x="{ml+12}" y="{mt+22}" font-size="12">linear trend slope {slope:.2f}</text>')
    colors = {"real_rename": "#2563eb", "real_field_migration": "#b45309"}
    for r in rows:
        color = colors.get(r["family"], "#0f766e")
        title = html.escape(f"{r['task_id']} {r['family']} c={r['adoption_fraction']:.2f} ratio={r['ratio']:.2f}")
        parts.append(f'<circle cx="{xs(r["adoption_fraction"]):.1f}" cy="{ys(r["ratio"]):.1f}" r="3.8" fill="{color}" opacity="0.62"><title>{title}</title></circle>')
    parts.append(f'<text x="{ml+pw/2}" y="{height-25}" text-anchor="middle" font-size="14">Rewrite adoption in vanilla output: new / (old + new)</text>')
    parts.append(f'<text x="25" y="{mt+ph/2}" transform="rotate(-90 25 {mt+ph/2})" text-anchor="middle" font-size="14">VANTAGE / PLD throughput</text>')
    parts.append(f'<circle cx="{width-225}" cy="25" r="5" fill="#2563eb"/><text x="{width-214}" y="30" font-size="12">rename</text>')
    parts.append(f'<circle cx="{width-145}" cy="25" r="5" fill="#b45309"/><text x="{width-134}" y="30" font-size="12">field migration</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def _steps_stats(steps_path: Path, vantage_method: str) -> dict[str, Any]:
    task_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "steps": 0,
        "trans_attempts": 0,
        "trans_zero_accepts": 0,
        "accepted": [],
        "exact_steps": 0,
    })
    total = Counter()
    accepted_hist = Counter()
    for line in steps_path.open():
        step = json.loads(line)
        if step.get("method") != vantage_method:
            continue
        tid = str(step.get("task_id"))
        task_stats[tid]["steps"] += 1
        total["steps"] += 1
        route = step.get("proposal_route")
        match = step.get("proposal_match_kind")
        if route == "exact_pld":
            task_stats[tid]["exact_steps"] += 1
        is_trans = (
            route == "transpld"
            or match in {"precomputed_transpld", "precomputed_transpld_compete", "routed_transpld_vref", "routed_transpld_bidir"}
        )
        if not is_trans:
            continue
        accepted = int(step.get("target_accepted_nonroot") or step.get("n_accepted_nonroot_drafts") or 0)
        task_stats[tid]["trans_attempts"] += 1
        task_stats[tid]["accepted"].append(accepted)
        total["trans_attempts"] += 1
        if accepted == 0:
            task_stats[tid]["trans_zero_accepts"] += 1
            total["trans_zero_accepts"] += 1
        if accepted == 0:
            accepted_hist["0"] += 1
        elif accepted == 1:
            accepted_hist["1"] += 1
        elif accepted == 2:
            accepted_hist["2"] += 1
        elif accepted <= 4:
            accepted_hist["3-4"] += 1
        elif accepted <= 7:
            accepted_hist["5-7"] += 1
        elif accepted <= 15:
            accepted_hist["8-15"] += 1
        elif accepted <= 31:
            accepted_hist["16-31"] += 1
        else:
            accepted_hist["32+"] += 1
    total["tasks_with_trans_attempts"] = sum(1 for s in task_stats.values() if s["trans_attempts"] > 0)
    total["tasks"] = len(task_stats)
    return {"total": dict(total), "accepted_hist": dict(accepted_hist), "by_task": task_stats}


def _diagnose(row: dict[str, Any], pld_method: str, vantage_method: str) -> str:
    vanilla = _strip_fence(_method_text(row, "vanilla"))
    pld = _strip_fence(_method_text(row, pld_method))
    nh = _strip_fence(_method_text(row, vantage_method))
    target = _strip_fence(str(row.get("deterministic_target") or ""))
    reference = str(row.get("reference") or "")
    adoption = _adoption(row, "vanilla")
    if not nh:
        return "missing VANTAGE output"
    if pld == nh:
        return "same output as PLD; loss is timing/proposal overhead"
    if adoption["new_hits"] == 0 and adoption["old_hits"] > 0:
        return "target model mostly kept old form; PLD mirrors non-rewrite output"
    if adoption["new_hits"] > 0 and adoption["old_hits"] > 0:
        return "partial rewrite adoption; transformed view competes with mixed target behavior"
    if len(nh) < 0.75 * len(target) or len(pld) < 0.75 * len(target):
        return "truncation/format mismatch dominates"
    ref_sim = _similarity(vanilla, reference)
    tgt_sim = _similarity(vanilla, target)
    if ref_sim >= tgt_sim:
        return "vanilla output closer to original reference than rewritten target"
    return "PLD has longer exact-copy spans or lower proposer overhead"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completions", required=True)
    parser.add_argument("--steps", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pld-method", default="blazedit_pld_w128_n10")
    parser.add_argument("--vantage-method", default="vantage_frozen_transpld")
    parser.add_argument("--compliance-method", default="vanilla")
    args = parser.parse_args()

    completions = _load_jsonl(Path(args.completions))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for row in completions:
        pld_tps = _tps(row, args.pld_method)
        nh_tps = _tps(row, args.vantage_method)
        adoption = _adoption(row, args.compliance_method)
        family = str((row.get("metadata") or {}).get("drift_family") or row.get("drift_family") or "unknown")
        target_match = _target_match(row, args.compliance_method)
        rows.append(
            {
                "task_id": row.get("task_id"),
                "repo": (row.get("metadata") or {}).get("repo"),
                "commit_sha": (row.get("metadata") or {}).get("commit_sha"),
                "commit_url": (row.get("metadata") or {}).get("commit_url"),
                "file_path": (row.get("metadata") or {}).get("file_path"),
                "family": family,
                "ratio": nh_tps / pld_tps if pld_tps else 0.0,
                "pld_tps": pld_tps,
                "vantage_tps": nh_tps,
                "adoption_fraction": adoption["adoption_fraction"],
                "full_pair_adoption": adoption["full_pair_adoption"],
                "old_hits": adoption["old_hits"],
                "new_hits": adoption["new_hits"],
                "target_match": target_match,
                "syntax_ok": _syntax_ok(_method_text(row, args.compliance_method)),
                "rewrite_pairs": _pairs(row),
            }
        )

    ratios = [r["ratio"] for r in rows]
    adoptions = [r["adoption_fraction"] for r in rows]
    target_match_rows = [row for row, r in zip(completions, rows) if r["target_match"]]
    target_match_report = _bootstrap_ratio(target_match_rows, args.vantage_method, args.pld_method)
    adopted_rows = [row for row, r in zip(completions, rows) if r["adoption_fraction"] >= 0.99]
    adopted_report = _bootstrap_ratio(adopted_rows, args.vantage_method, args.pld_method)

    steps = _steps_stats(Path(args.steps), args.vantage_method)
    steps_by_task = steps["by_task"]
    for r in rows:
        st = steps_by_task.get(str(r["task_id"]), {})
        r["trans_attempts"] = int(st.get("trans_attempts") or 0)
        r["trans_zero_accepts"] = int(st.get("trans_zero_accepts") or 0)
        r["trans_mean_accept"] = (
            sum(st.get("accepted") or []) / len(st.get("accepted") or [])
            if st.get("accepted")
            else 0.0
        )

    trans_attempt_rows = [
        row
        for row, r in zip(completions, rows)
        if r.get("trans_attempts", 0) > 0
    ]
    no_trans_attempt_rows = [
        row
        for row, r in zip(completions, rows)
        if r.get("trans_attempts", 0) <= 0
    ]
    trans_attempt_report = _bootstrap_ratio(
        trans_attempt_rows,
        args.vantage_method,
        args.pld_method,
    )
    no_trans_attempt_report = _bootstrap_ratio(
        no_trans_attempt_rows,
        args.vantage_method,
        args.pld_method,
    )

    by_family: dict[str, dict[str, Any]] = {}
    for family in sorted(set(r["family"] for r in rows)):
        part = [r for r in rows if r["family"] == family]
        task_part = [row for row, rr in zip(completions, rows) if rr["family"] == family]
        by_family[family] = {
            "n": len(part),
            "median_ratio": statistics.median([r["ratio"] for r in part]),
            "mean_ratio": sum(r["ratio"] for r in part) / len(part),
            "share_ratio_ge_1": sum(1 for r in part if r["ratio"] >= 1.0) / len(part),
            "median_adoption": statistics.median([r["adoption_fraction"] for r in part]),
            "target_match_rate": sum(1 for r in part if r["target_match"]) / len(part),
            "aggregate_ratio": _bootstrap_ratio(task_part, args.vantage_method, args.pld_method, n_boot=1500),
        }

    corr = {
        "pearson_adoption_ratio": _pearson(adoptions, ratios),
        "spearman_adoption_ratio": _pearson(_rank(adoptions), _rank(ratios)),
    }
    summary = {
        "n": len(rows),
        "ratio": {
            "median": statistics.median(ratios),
            "mean": sum(ratios) / len(ratios),
            "p10": sorted(ratios)[int(0.10 * (len(ratios) - 1))],
            "p90": sorted(ratios)[int(0.90 * (len(ratios) - 1))],
            "share_ge_1": sum(1 for r in ratios if r >= 1.0) / len(ratios),
            "share_ge_1_25": sum(1 for r in ratios if r >= 1.25) / len(ratios),
            "share_le_0_75": sum(1 for r in ratios if r <= 0.75) / len(ratios),
        },
        "compliance": {
            "median_adoption_fraction": statistics.median(adoptions),
            "mean_adoption_fraction": sum(adoptions) / len(adoptions),
            "full_pair_adoption_rate": sum(1 for r in rows if r["full_pair_adoption"] >= 1.0) / len(rows),
            "target_match_rate": sum(1 for r in rows if r["target_match"]) / len(rows),
            "syntax_ok_rate": sum(1 for r in rows if r["syntax_ok"]) / len(rows),
        },
        "correlation": corr,
        "high_compliance_target_match_subset": target_match_report,
        "full_pair_adoption_subset": adopted_report,
        "transpld_attempt_subset": trans_attempt_report,
        "no_transpld_attempt_subset": no_trans_attempt_report,
        "families": by_family,
        "transpld_steps": steps["total"],
        "transpld_accepted_hist": steps["accepted_hist"],
    }

    worst = sorted(rows, key=lambda r: r["ratio"])[:20]
    worst_lines = [
        "# Worst 20 VANTAGE losses on real commits",
        "",
        "| # | Task | Family | Ratio | Compliance | Trans attempts | Diagnosis | Commit |",
        "|---:|---|---|---:|---:|---:|---|---|",
    ]
    row_by_id = {row.get("task_id"): row for row in completions}
    worst_details: list[dict[str, Any]] = []
    for i, r in enumerate(worst, 1):
        source = row_by_id.get(r["task_id"], {})
        diagnosis = _diagnose(source, args.pld_method, args.vantage_method)
        url = r.get("commit_url") or ""
        label = f"{r.get('repo')}@{str(r.get('commit_sha') or '')[:7]}"
        worst_lines.append(
            f"| {i} | `{r['task_id']}` | {r['family']} | {r['ratio']:.3f} | "
            f"{r['adoption_fraction']:.3f} | {r['trans_attempts']} | {diagnosis} | "
            f"[{label}]({url}) |"
        )
        worst_details.append(
            {
                **r,
                "diagnosis": diagnosis,
                "reference_excerpt": str(source.get("reference") or "")[:1200],
                "target_excerpt": str(source.get("deterministic_target") or "")[:1200],
                "vanilla_excerpt": _strip_fence(_method_text(source, "vanilla"))[:1200],
                "pld_excerpt": _strip_fence(_method_text(source, args.pld_method))[:1200],
                "vantage_excerpt": _strip_fence(_method_text(source, args.vantage_method))[:1200],
            }
        )

    lines = [
        "# Real-Commit Path A Diagnostics",
        "",
        "## 1. Per-task ratio histogram",
        "",
        f"Median VANTAGE/PLD: **{summary['ratio']['median']:.3f}**; "
        f"p10/p90: **{summary['ratio']['p10']:.3f}/{summary['ratio']['p90']:.3f}**; "
        f"share >= 1: **{summary['ratio']['share_ge_1']:.1%}**; "
        f"share >= 1.25: **{summary['ratio']['share_ge_1_25']:.1%}**; "
        f"share <= 0.75: **{summary['ratio']['share_le_0_75']:.1%}**.",
        "",
        f"![ratio histogram](ratio_histogram.svg)",
        "",
        "## 2. Per-task compliance",
        "",
        f"Median rewrite-adoption fraction: **{summary['compliance']['median_adoption_fraction']:.3f}**. "
        f"Full pair adoption: **{summary['compliance']['full_pair_adoption_rate']:.1%}**. "
        f"Exact target match: **{summary['compliance']['target_match_rate']:.1%}**.",
        "",
        "## 3. Compliance x ratio scatter",
        "",
        f"Pearson: **{corr['pearson_adoption_ratio']:.3f}**. "
        f"Spearman: **{corr['spearman_adoption_ratio']:.3f}**.",
        "",
        "![compliance scatter](compliance_ratio_scatter.svg)",
        "",
        "## 4. High-compliance subset",
        "",
        f"Exact-target subset n={target_match_report['n']}: VANTAGE/PLD "
        f"**{target_match_report['ratio']:.3f} [{target_match_report['ci95'][0]:.3f}, {target_match_report['ci95'][1]:.3f}]**, "
        f"P>1={target_match_report['p_gt_1']:.3f}.",
        "",
        f"Full-pair-adoption subset n={adopted_report['n']}: VANTAGE/PLD "
        f"**{adopted_report['ratio']:.3f} [{adopted_report['ci95'][0]:.3f}, {adopted_report['ci95'][1]:.3f}]**, "
        f"P>1={adopted_report['p_gt_1']:.3f}.",
        "",
        f"Predicted/attempted transformed subset n={trans_attempt_report['n']}: VANTAGE/PLD "
        f"**{trans_attempt_report['ratio']:.3f} [{trans_attempt_report['ci95'][0]:.3f}, {trans_attempt_report['ci95'][1]:.3f}]**, "
        f"P>1={trans_attempt_report['p_gt_1']:.3f}.",
        "",
        f"PLD-routed/no-attempt subset n={no_trans_attempt_report['n']}: VANTAGE/PLD "
        f"**{no_trans_attempt_report['ratio']:.3f} [{no_trans_attempt_report['ci95'][0]:.3f}, {no_trans_attempt_report['ci95'][1]:.3f}]**, "
        f"P>1={no_trans_attempt_report['p_gt_1']:.3f}.",
        "",
        "## 5. TransPLD firing and zero-accept distribution",
        "",
        f"TransPLD attempts: **{steps['total'].get('trans_attempts', 0)}** over "
        f"**{steps['total'].get('steps', 0)}** VANTAGE steps; tasks with TransPLD attempts: "
        f"**{steps['total'].get('tasks_with_trans_attempts', 0)}/{steps['total'].get('tasks', 0)}**. "
        f"Zero-accept attempts: **{steps['total'].get('trans_zero_accepts', 0)}**.",
        "",
        "| Accepted non-root tokens | Attempts |",
        "|---:|---:|",
    ]
    for key in ["0", "1", "2", "3-4", "5-7", "8-15", "16-31", "32+"]:
        lines.append(f"| {key} | {steps['accepted_hist'].get(key, 0)} |")
    lines += [
        "",
        "## Family breakdown",
        "",
        "| Family | n | Aggregate ratio | Median ratio | Share >= 1 | Median adoption | Exact target match |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family, item in by_family.items():
        ar = item["aggregate_ratio"]
        lines.append(
            f"| {family} | {item['n']} | {ar['ratio']:.3f} [{ar['ci95'][0]:.3f}, {ar['ci95'][1]:.3f}] | "
            f"{item['median_ratio']:.3f} | {item['share_ratio_ge_1']:.1%} | "
            f"{item['median_adoption']:.3f} | {item['target_match_rate']:.1%} |"
        )
    lines += ["", "## 6. Worst-loss qualitative read", ""]
    lines.extend(worst_lines[2:])

    out_dir.joinpath("summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    out_dir.joinpath("per_task_rows.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    )
    out_dir.joinpath("worst20_details.json").write_text(json.dumps(worst_details, indent=2) + "\n")
    out_dir.joinpath("worst20.md").write_text("\n".join(worst_lines) + "\n")
    out_dir.joinpath("diagnostics.md").write_text("\n".join(lines) + "\n")
    _histogram_svg(ratios, out_dir / "ratio_histogram.svg")
    _scatter_svg(rows, out_dir / "compliance_ratio_scatter.svg")
    print(out_dir.joinpath("diagnostics.md").read_text())


if __name__ == "__main__":
    main()
