#!/usr/bin/env python3
"""Gate TransPLD routing on per-task rewrite compliance.

This CPU-only analysis consumes existing `completions.jsonl` artifacts and
answers the reviewer-critical question: do higher-compliance tasks get better
TransPLD/PLD ratios?  It writes workload-specific scatter plots plus a
median-split report by model.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from analyze_edit_compliance import _parse_run_spec, analyze_run


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx = _mean(xs)
    my = _mean(ys)
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


def _spearman(xs: list[float], ys: list[float]) -> float:
    return _pearson(_rank(xs), _rank(ys))


def _split_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["model"]), str(row["workload"]))].append(row)
    out: list[dict[str, Any]] = []
    for (model, workload), group in sorted(groups.items()):
        median_c = _median([float(r["rewrite_compliance"]) for r in group])
        low = [r for r in group if float(r["rewrite_compliance"]) <= median_c]
        high = [r for r in group if float(r["rewrite_compliance"]) > median_c]
        if not high:
            # Degenerate tie case: split by sorted order so the report is still
            # informative for all-equal compliance workloads.
            ordered = sorted(group, key=lambda r: float(r["rewrite_compliance"]))
            mid = len(ordered) // 2
            low, high = ordered[:mid], ordered[mid:]
        xs = [float(r["rewrite_compliance"]) for r in group]
        ys = [float(r["transpld_over_pld"]) for r in group]
        for label, part in [("low", low), ("high", high)]:
            ratios = [float(r["transpld_over_pld"]) for r in part]
            compliances = [float(r["rewrite_compliance"]) for r in part]
            out.append(
                {
                    "model": model,
                    "workload": workload,
                    "split": label,
                    "n": len(part),
                    "compliance_median_cut": median_c,
                    "median_compliance": _median(compliances),
                    "mean_transpld_over_pld": _mean(ratios),
                    "median_transpld_over_pld": _median(ratios),
                    "share_ratio_ge_1": (
                        sum(1 for v in ratios if v >= 1.0) / len(ratios)
                        if ratios
                        else 0.0
                    ),
                    "pearson_all": _pearson(xs, ys),
                    "spearman_all": _spearman(xs, ys),
                }
            )
    return out


def _write_split_md(split_rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Compliance Gate Median Split",
        "",
        "Within each `(model, workload)`, tasks are split by median rewrite compliance. "
        "Higher compliance means the target output is closer to the deterministic rewritten target than to the original reference.",
        "",
        "| Model | Workload | Split | n | Median compliance | Median TransPLD/PLD | Mean TransPLD/PLD | Share >= 1 | Pearson all | Spearman all |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in split_rows:
        lines.append(
            f"| {row['model']} | {row['workload']} | {row['split']} | {row['n']} | "
            f"{row['median_compliance']:.3f} | {row['median_transpld_over_pld']:.3f} | "
            f"{row['mean_transpld_over_pld']:.3f} | {row['share_ratio_ge_1']:.1%} | "
            f"{row['pearson_all']:.3f} | {row['spearman_all']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_scatter_svg(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    width, height = 900, 560
    ml, mr, mt, mb = 72, 34, 42, 72
    pw, ph = width - ml - mr, height - mt - mb
    max_y = max([1.35] + [float(r["transpld_over_pld"]) for r in rows])
    max_y = min(3.0, max_y * 1.08)
    colors = {
        "qwen_base": "#0f766e",
        "qwen": "#0f766e",
        "deepseek_instruct": "#b45309",
        "deepseek": "#b45309",
    }

    def xscale(x: float) -> float:
        return ml + max(0.0, min(1.0, x)) * pw

    def yscale(y: float) -> float:
        return mt + ph - max(0.0, min(max_y, y)) / max_y * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{ml}" y="24" font-size="18" font-weight="700" fill="#111827">{html.escape(title)}</text>',
        f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="#d1d5db"/>',
        f'<line x1="{ml}" y1="{yscale(1):.1f}" x2="{width-mr}" y2="{yscale(1):.1f}" stroke="#6b7280" stroke-dasharray="5 5"/>',
    ]
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        x = xscale(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{mt+ph}" x2="{x:.1f}" y2="{mt+ph+6}" stroke="#6b7280"/>')
        parts.append(f'<text x="{x:.1f}" y="{mt+ph+24}" text-anchor="middle" font-size="12" fill="#374151">{tick:.2g}</text>')
    for tick in [0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        if tick > max_y:
            continue
        y = yscale(tick)
        parts.append(f'<line x1="{ml-6}" y1="{y:.1f}" x2="{ml}" y2="{y:.1f}" stroke="#6b7280"/>')
        parts.append(f'<text x="{ml-10}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="#374151">{tick:.1f}</text>')

    xs = [float(r["rewrite_compliance"]) for r in rows]
    ys = [float(r["transpld_over_pld"]) for r in rows]
    if len(xs) >= 2 and max(xs) - min(xs) > 1e-9:
        mx = _mean(xs)
        my = _mean(ys)
        denom = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom if denom else 0.0
        intercept = my - slope * mx
        x0, x1 = min(xs), max(xs)
        y0, y1 = intercept + slope * x0, intercept + slope * x1
        parts.append(
            f'<line x1="{xscale(x0):.1f}" y1="{yscale(y0):.1f}" x2="{xscale(x1):.1f}" y2="{yscale(y1):.1f}" '
            'stroke="#111827" stroke-width="2.4" stroke-opacity="0.70"/>'
        )
        parts.append(
            f'<text x="{ml+12}" y="{mt+22}" font-size="12" fill="#111827">linear trend: slope {slope:.2f}</text>'
        )
    else:
        parts.append(
            f'<text x="{ml+12}" y="{mt+22}" font-size="12" fill="#111827">compliance tied; trend not estimable</text>'
        )

    for row in rows:
        model = str(row["model"])
        color = colors.get(model.lower(), "#2563eb")
        x = xscale(float(row["rewrite_compliance"]))
        y = yscale(float(row["transpld_over_pld"]))
        title_text = html.escape(
            f"{row['model']} {row['task_id']} compliance={row['rewrite_compliance']:.3f} ratio={row['transpld_over_pld']:.3f}"
        )
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{color}" fill-opacity="0.70"><title>{title_text}</title></circle>'
        )
    parts.extend(
        [
            f'<text x="{ml + pw/2}" y="{height-24}" text-anchor="middle" font-size="15" fill="#111827">Rewrite compliance</text>',
            f'<text x="24" y="{mt + ph/2}" transform="rotate(-90 24 {mt + ph/2})" text-anchor="middle" font-size="15" fill="#111827">TransPLD / PLD throughput</text>',
            f'<circle cx="{width-245}" cy="24" r="5" fill="#0f766e"/><text x="{width-234}" y="29" font-size="13">Qwen base</text>',
            f'<circle cx="{width-145}" cy="24" r="5" fill="#b45309"/><text x="{width-134}" y="29" font-size="13">DeepSeek-Instruct</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for item in args.run:
        rows.extend(analyze_run(_parse_run_spec(item)))

    split_rows = _split_rows(rows)
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    (out_dir / "median_split.json").write_text(json.dumps(split_rows, indent=2) + "\n")
    _write_split_md(split_rows, out_dir / "median_split.md")
    for workload in sorted({str(r["workload"]) for r in rows}):
        workload_rows = [r for r in rows if str(r["workload"]) == workload]
        _write_scatter_svg(
            workload_rows,
            out_dir / f"scatter_{workload}.svg",
            f"{workload}: compliance vs. TransPLD/PLD",
        )
    print((out_dir / "median_split.md").read_text())


if __name__ == "__main__":
    main()
