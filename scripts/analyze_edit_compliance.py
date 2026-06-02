#!/usr/bin/env python3
"""Analyze whether edit outputs follow the requested rewrite.

The script is intentionally CPU-only and consumes existing `completions.jsonl`
artifacts.  It reports:

* token edit distance from each output to the original reference;
* token edit distance from each output to the deterministic rewritten target;
* a rewrite-compliance score in [0, 1], where larger means closer to the
  rewritten target than to the original reference;
* direct old/new symbol occurrence checks;
* a per-task TransPLD/PLD throughput ratio; and
* a small manual-inspection worksheet for low-compliance DeepSeek failures.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PLD_METHOD = "blazedit_pld_w128_n10"
RATIO_METHOD_PRIORITY = [
    "vantage_routed_transpld_m4_w128_n10",
    "vantage_routed_transpld_w128_n10",
    "vantage_transpld_m4_w128_n10",
    "vantage_transpld_w128_n10",
]


@dataclass(frozen=True)
class RunSpec:
    model: str
    workload: str
    path: Path
    ratio_method: str
    pld_method: str


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _nested(metadata: dict[str, Any]) -> dict[str, Any]:
    nested = metadata.get("metadata")
    return nested if isinstance(nested, dict) else {}


def _rewrite_pairs(row: dict[str, Any]) -> dict[str, str]:
    metadata = row.get("metadata") or {}
    nested = _nested(metadata)
    pairs = row.get("rewrite_pairs") or metadata.get("rewrite_pairs") or nested.get("rewrite_pairs")
    if isinstance(pairs, dict):
        return {str(k): str(v) for k, v in pairs.items() if str(k) and str(v)}
    if isinstance(pairs, list):
        out: dict[str, str] = {}
        for item in pairs:
            if isinstance(item, dict):
                old = item.get("old") or item.get("from") or item.get("source")
                new = item.get("new") or item.get("to") or item.get("target")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                old, new = item[0], item[1]
            else:
                continue
            if old and new:
                out[str(old)] = str(new)
        return out
    return {}


def _clean_output_text(text: str) -> str:
    # DeepSeek decoded artifacts in existing runs include byte-level marker
    # glyphs.  Normalize them before string checks and code tokenization.
    text = text.replace("Ċ", "\n").replace("Ġ", " ").replace("▁", " ")
    text = re.sub(r"```[a-zA-Z0-9_+-]*", "", text)
    text = text.replace("```", "")
    return text


_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*"
    r"|[0-9]+(?:\.[0-9]+)?"
    r"|==|!=|<=|>=|->|:=|\+=|-=|\*=|/=|//=|\*\*"
    r"|[^\s]"
)


def _code_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(_clean_output_text(text))


def _levenshtein(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, 1):
        current = [i]
        for j, item_b in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (item_a != item_b),
                )
            )
        previous = current
    return previous[-1]


def _tokens_per_sec(output: dict[str, Any]) -> float:
    wall_us = float(output.get("wall_us") or 0.0)
    tokens = float(output.get("n_new_tokens") or 0.0)
    return tokens / (wall_us / 1e6) if wall_us > 0 else 0.0


def _choose_method(outputs: dict[str, Any], requested: str) -> str:
    if requested and requested in outputs:
        return requested
    for method in RATIO_METHOD_PRIORITY:
        if method in outputs:
            return method
    raise KeyError(f"none of the ratio methods are present: {sorted(outputs)}")


def _symbol_counts(output_text: str, pairs: dict[str, str]) -> tuple[int, int, dict[str, dict[str, int]]]:
    cleaned = _clean_output_text(output_text)
    compact = re.sub(r"\s+", "", cleaned)
    old_total = 0
    new_total = 0
    details: dict[str, dict[str, int]] = {}
    for old, new in pairs.items():
        old_count = cleaned.count(old) + (compact.count(old) if old not in cleaned else 0)
        new_count = cleaned.count(new) + (compact.count(new) if new not in cleaned else 0)
        old_total += old_count
        new_total += new_count
        details[f"{old}->{new}"] = {"old": old_count, "new": new_count}
    return old_total, new_total, details


def _direct_class(old_count: int, new_count: int, pair_count: int) -> str:
    if pair_count <= 0:
        return "no_map"
    if new_count > 0 and old_count == 0:
        return "full"
    if new_count > 0 and old_count > 0:
        return "partial"
    if old_count > 0 and new_count == 0:
        return "none"
    return "no_evidence"


def _manual_case(old_count: int, new_count: int) -> str:
    if old_count > 0 and new_count == 0:
        return "A_wrong_edit_kept_old_identifier"
    if old_count > 0 and new_count > 0:
        return "C_mixed_edit"
    return "B_different_or_restructured_edit"


def analyze_run(spec: RunSpec) -> list[dict[str, Any]]:
    rows = _load_jsonl(spec.path)
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        outputs = row.get("outputs") or {}
        if spec.pld_method not in outputs:
            continue
        ratio_method = _choose_method(outputs, spec.ratio_method)
        ratio_output = outputs[ratio_method]
        pld_output = outputs[spec.pld_method]
        output_text = str(ratio_output.get("text") or ratio_output.get("raw_text") or "")
        reference = str(row.get("reference") or "")
        rewritten = str(row.get("deterministic_target") or "")
        pairs = _rewrite_pairs(row)
        output_tokens = _code_tokens(output_text)
        reference_tokens = _code_tokens(reference)
        rewritten_tokens = _code_tokens(rewritten)
        dist_original = _levenshtein(output_tokens, reference_tokens)
        dist_rewritten = _levenshtein(output_tokens, rewritten_tokens)
        denom = dist_original + dist_rewritten
        rewrite_compliance = dist_original / denom if denom else 0.5
        old_count, new_count, symbol_details = _symbol_counts(output_text, pairs)
        ratio_tps = _tokens_per_sec(ratio_output)
        pld_tps = _tokens_per_sec(pld_output)
        out_rows.append(
            {
                "model": spec.model,
                "workload": spec.workload,
                "task_id": row.get("task_id"),
                "ratio_method": ratio_method,
                "pld_method": spec.pld_method,
                "ratio_tps": ratio_tps,
                "pld_tps": pld_tps,
                "transpld_over_pld": ratio_tps / pld_tps if pld_tps else 0.0,
                "dist_to_original": dist_original,
                "dist_to_rewritten": dist_rewritten,
                # Higher means closer to the rewritten target.  This is the
                # mathematically consistent orientation for d(original)/sum.
                "rewrite_compliance": rewrite_compliance,
                "old_symbol_count": old_count,
                "new_symbol_count": new_count,
                "direct_compliance": _direct_class(old_count, new_count, len(pairs)),
                "manual_case": _manual_case(old_count, new_count),
                "rewrite_pairs": pairs,
                "symbol_details": symbol_details,
                "output_preview": _clean_output_text(output_text)[:700],
                "reference_preview": reference[:700],
                "rewritten_preview": rewritten[:700],
            }
        )
    return out_rows


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["workload"])].append(row)
    summaries = []
    for (model, workload), group in sorted(groups.items()):
        counts = Counter(row["direct_compliance"] for row in group)
        n = len(group)
        summaries.append(
            {
                "model": model,
                "workload": workload,
                "n": n,
                "median_rewrite_compliance": _median([row["rewrite_compliance"] for row in group]),
                "median_transpld_over_pld": _median([row["transpld_over_pld"] for row in group]),
                "median_dist_to_original": _median([row["dist_to_original"] for row in group]),
                "median_dist_to_rewritten": _median([row["dist_to_rewritten"] for row in group]),
                "full_rate": counts["full"] / n if n else 0.0,
                "partial_rate": counts["partial"] / n if n else 0.0,
                "none_rate": counts["none"] / n if n else 0.0,
                "no_evidence_rate": counts["no_evidence"] / n if n else 0.0,
                "counts": dict(counts),
            }
        )
    return summaries


def _write_summary_md(summaries: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Edit Compliance Summary",
        "",
        "Rewrite compliance is `d(output, original) / (d(output, original) + d(output, rewritten))`; larger means the output is closer to the rewritten target.",
        "",
        "| Model | Workload | n | Median compliance | Median TransPLD/PLD | Full | Partial | None | No evidence | Median d_orig | Median d_rewrite |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['model']} | {row['workload']} | {row['n']} | "
            f"{row['median_rewrite_compliance']:.3f} | {row['median_transpld_over_pld']:.3f} | "
            f"{row['full_rate']:.1%} | {row['partial_rate']:.1%} | "
            f"{row['none_rate']:.1%} | {row['no_evidence_rate']:.1%} | "
            f"{row['median_dist_to_original']:.1f} | {row['median_dist_to_rewritten']:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "model",
        "workload",
        "task_id",
        "ratio_method",
        "pld_method",
        "transpld_over_pld",
        "rewrite_compliance",
        "dist_to_original",
        "dist_to_rewritten",
        "old_symbol_count",
        "new_symbol_count",
        "direct_compliance",
        "manual_case",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _write_scatter_svg(rows: list[dict[str, Any]], path: Path) -> None:
    width, height = 900, 560
    margin_left, margin_right, margin_top, margin_bottom = 72, 34, 32, 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_y = max([1.4] + [float(row["transpld_over_pld"]) for row in rows])
    max_y = min(3.0, max_y * 1.08)
    colors = {"qwen": "#0f766e", "deepseek": "#b45309"}

    def xscale(x: float) -> float:
        return margin_left + max(0.0, min(1.0, x)) * plot_w

    def yscale(y: float) -> float:
        y = max(0.0, min(max_y, y))
        return margin_top + plot_h - (y / max_y) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin_left}" y1="{yscale(1.0):.1f}" x2="{width-margin_right}" y2="{yscale(1.0):.1f}" stroke="#6b7280" stroke-dasharray="5 5"/>',
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#d1d5db"/>',
    ]
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        x = xscale(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{margin_top+plot_h}" x2="{x:.1f}" y2="{margin_top+plot_h+6}" stroke="#6b7280"/>')
        parts.append(f'<text x="{x:.1f}" y="{margin_top+plot_h+24}" text-anchor="middle" font-size="12" fill="#374151">{tick:.2g}</text>')
    y_ticks = [0, 0.5, 1.0, 1.5, 2.0]
    if max_y > 2.3:
        y_ticks.append(2.5)
    for tick in y_ticks:
        if tick > max_y:
            continue
        y = yscale(tick)
        parts.append(f'<line x1="{margin_left-6}" y1="{y:.1f}" x2="{margin_left}" y2="{y:.1f}" stroke="#6b7280"/>')
        parts.append(f'<text x="{margin_left-10}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="#374151">{tick:.1f}</text>')
    for row in rows:
        color = colors.get(str(row["model"]).lower(), "#2563eb")
        x = xscale(float(row["rewrite_compliance"]))
        y = yscale(float(row["transpld_over_pld"]))
        title = html.escape(f"{row['model']} {row['workload']} {row['task_id']} ratio={row['transpld_over_pld']:.2f} compliance={row['rewrite_compliance']:.2f}")
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{color}" fill-opacity="0.68"><title>{title}</title></circle>')
    parts += [
        f'<text x="{margin_left + plot_w/2}" y="{height-22}" text-anchor="middle" font-size="15" fill="#111827">Rewrite compliance: 0 = original-like, 1 = rewritten-target-like</text>',
        f'<text x="22" y="{margin_top + plot_h/2}" transform="rotate(-90 22 {margin_top + plot_h/2})" text-anchor="middle" font-size="15" fill="#111827">Per-task TransPLD / PLD throughput</text>',
        f'<text x="{margin_left}" y="22" font-size="17" font-weight="700" fill="#111827">Compliance vs. VANTAGE speedup</text>',
        f'<circle cx="{width-205}" cy="23" r="5" fill="{colors["qwen"]}"/><text x="{width-194}" y="28" font-size="13">Qwen</text>',
        f'<circle cx="{width-135}" cy="23" r="5" fill="{colors["deepseek"]}"/><text x="{width-124}" y="28" font-size="13">DeepSeek</text>',
        "</svg>",
    ]
    path.write_text("\n".join(parts) + "\n")


def _write_manual_report(rows: list[dict[str, Any]], path: Path, *, model: str, workload: str, limit: int) -> None:
    candidates = [
        row
        for row in rows
        if row["model"].lower() == model.lower()
        and row["workload"] == workload
        and row["direct_compliance"] != "full"
    ]
    candidates.sort(key=lambda row: (row["rewrite_compliance"], -row["transpld_over_pld"]))
    selected = candidates[:limit]
    counts = Counter(row["manual_case"] for row in selected)
    lines = [
        f"# Manual Inspection Worksheet: {model} {workload}",
        "",
        f"Selected {len(selected)} low-compliance rows. Heuristic case counts: {dict(counts)}.",
        "",
        "Case A = wrong edit/keeps old identifier. Case B = different or restructured edit. Case C = mixed old and new.",
        "",
    ]
    for idx, row in enumerate(selected, 1):
        lines += [
            f"## {idx}. `{row['task_id']}`",
            "",
            f"- Heuristic case: `{row['manual_case']}`",
            f"- Direct compliance: `{row['direct_compliance']}`",
            f"- Rewrite compliance: {row['rewrite_compliance']:.3f}",
            f"- TransPLD/PLD: {row['transpld_over_pld']:.3f}",
            f"- Old/new counts: {row['old_symbol_count']} / {row['new_symbol_count']}",
            f"- Rewrite pairs: `{row['rewrite_pairs']}`",
            "",
            "Output preview:",
            "",
            "```text",
            row["output_preview"],
            "```",
            "",
            "Rewritten target preview:",
            "",
            "```text",
            row["rewritten_preview"][:500],
            "```",
            "",
        ]
    path.write_text("\n".join(lines))


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys) or max(xs) - min(xs) < 1e-12:
        return None
    mx = _mean(xs)
    my = _mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
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


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys) or max(xs) - min(xs) < 1e-12:
        return None
    return _pearson(_rank(xs), _rank(ys))


def _parse_bootstrap_spec(text: str) -> tuple[str, str, Path, str]:
    parts = text.split(":", 3)
    if len(parts) < 3:
        raise ValueError("--boundary-bootstrap must be model:workload:path[:method]")
    model, workload, path_s = parts[0], parts[1], parts[2]
    method = parts[3] if len(parts) >= 4 else ""
    return model, workload, Path(path_s), method


def _bootstrap_row(path: Path, method: str, workload: str) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if "rows" in data:
        aliases = {
            "field_rename": "field100",
            "style_rewrite": "style100",
            "zero_drift": "zero100",
        }
        wanted = aliases.get(workload, workload)
        row = next((item for item in data["rows"] if item.get("workload") == wanted), None)
        if row is None:
            raise ValueError(f"cannot find workload {wanted!r} in {path}")
        return {
            "method": method or "vantage_frozen_transpld",
            "n_tasks": row["n_tasks"],
            "ratio": row["ratio"],
            "ci95": row["ci95"],
            "p_gt_1": row.get("p_gt_1"),
            "tokens_per_sec": row["method_tps"],
        }
    if not method:
        candidates = [key for key in data.get("by_method", {}) if key not in {data.get("baseline"), "vanilla"}]
        if not candidates:
            raise ValueError(f"cannot infer method from {path}")
        method = candidates[-1]
    by_method = data["by_method"][method]
    return {
        "method": method,
        "n_tasks": data["n_tasks"],
        "ratio": by_method["speedup_vs_blazedit_pld_w128_n10"],
        "ci95": by_method["ci95"],
        "p_gt_1": by_method.get("p_gt_1"),
        "tokens_per_sec": by_method["tokens_per_sec"],
    }


def _boundary_summary(
    rows: list[dict[str, Any]],
    *,
    bootstrap_specs: list[str],
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["model"]), str(row["workload"]))].append(row)

    bootstraps: dict[tuple[str, str], dict[str, Any]] = {}
    for item in bootstrap_specs:
        model, workload, path, method = _parse_bootstrap_spec(item)
        boot = _bootstrap_row(path, method, workload)
        boot["path"] = str(path)
        bootstraps[(model, workload)] = boot

    boundary_rows: list[dict[str, Any]] = []
    for key in sorted(set(groups) | set(bootstraps)):
        group = groups.get(key, [])
        xs = [float(row["rewrite_compliance"]) for row in group]
        ys = [float(row["transpld_over_pld"]) for row in group]
        pearson = _pearson(xs, ys)
        spearman = _spearman(xs, ys)
        boot = bootstraps.get(key, {})
        boundary_rows.append(
            {
                "model": key[0],
                "workload": key[1],
                "n": len(group) or boot.get("n_tasks", 0),
                "median_compliance": _median(xs),
                "median_per_task_ratio": _median(ys),
                "pearson": pearson,
                "spearman": spearman,
                "correlation_status": "not estimable: constant or missing compliance" if pearson is None else "computed",
                "method": boot.get("method", group[0]["ratio_method"] if group else ""),
                "speedup": boot.get("ratio"),
                "ci95": boot.get("ci95"),
                "p_gt_1": boot.get("p_gt_1"),
                "bootstrap_path": boot.get("path"),
            }
        )

    compliant = [
        row
        for row in rows
        if float(row["rewrite_compliance"]) >= 0.95 and float(row["transpld_over_pld"]) > 1.05
    ]
    compliant.sort(key=lambda row: (-float(row["transpld_over_pld"]), str(row["model"]), str(row["task_id"])))
    non_compliant = [
        row
        for row in rows
        if row["model"] == "deepseek_instruct"
        and float(row["rewrite_compliance"]) <= 0.51
        and float(row["transpld_over_pld"]) < 1.0
    ]
    non_compliant.sort(key=lambda row: (float(row["rewrite_compliance"]), float(row["transpld_over_pld"])))

    return {
        "schema": "asts-spec/transpld-compliance-boundary/v1",
        "metric_definition": "rewrite_compliance = d(output, original_reference) / (d(output, original_reference) + d(output, deterministic_rewritten_target)); larger means closer to the rewritten target. Distances are token-level Levenshtein distances over lightweight code tokens after artifact cleanup.",
        "rows": boundary_rows,
        "examples": {
            "compliant_success": compliant[:2],
            "non_compliant_failure": non_compliant[:2],
        },
    }


def _fmt_float(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _write_boundary_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# TransPLD Instruction-Compliance Boundary",
        "",
        f"Compliance metric: `{report['metric_definition']}`",
        "",
        "| Model | Workload | n | Median compliance | Speedup vs PLD | 95% CI | Spearman r | Correlation status |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        ci = row.get("ci95")
        ci_s = "n/a" if not ci else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        lines.append(
            f"| {row['model']} | {row['workload']} | {row['n']} | "
            f"{_fmt_float(row.get('median_compliance'))} | {_fmt_float(row.get('speedup'))} | "
            f"{ci_s} | {_fmt_float(row.get('spearman'))} | {row['correlation_status']} |"
        )
    lines += ["", "## Examples", ""]
    for label, title in [
        ("compliant_success", "Compliant success"),
        ("non_compliant_failure", "Non-compliant failure"),
    ]:
        lines += [f"### {title}", ""]
        examples = report["examples"].get(label, [])
        if not examples:
            lines.append("No example found under the selection heuristic.")
            lines.append("")
            continue
        for row in examples:
            lines += [
                f"- `{row['model']}` `{row['workload']}` `{row['task_id']}`: "
                f"compliance `{row['rewrite_compliance']:.3f}`, TransPLD/PLD `{row['transpld_over_pld']:.3f}`, "
                f"pairs `{row['rewrite_pairs']}`.",
                "",
                "  Output preview:",
                "",
                "  ```text",
                "\n".join("  " + line for line in str(row["output_preview"]).splitlines()[:8]),
                "  ```",
                "",
            ]
    path.write_text("\n".join(lines) + "\n")


def _parse_run_spec(text: str) -> RunSpec:
    parts = text.split(":", 4)
    if len(parts) < 3:
        raise ValueError(
            "--run must be model:workload:path[:ratio_method[:pld_method]]"
        )
    model, workload, path_s = parts[0], parts[1], parts[2]
    ratio_method = parts[3] if len(parts) >= 4 else ""
    pld_method = parts[4] if len(parts) >= 5 else DEFAULT_PLD_METHOD
    return RunSpec(
        model=model,
        workload=workload,
        path=Path(path_s),
        ratio_method=ratio_method,
        pld_method=pld_method or DEFAULT_PLD_METHOD,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="model:workload:path[:ratio_method[:pld_method]]",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manual-model", default="deepseek")
    parser.add_argument("--manual-workload", default="field_rename")
    parser.add_argument("--manual-limit", type=int, default=20)
    parser.add_argument(
        "--boundary-bootstrap",
        action="append",
        default=[],
        help="Optional model:workload:bootstrap.json[:method] entries for a paper-facing boundary table.",
    )
    parser.add_argument("--boundary-output-md")
    parser.add_argument("--boundary-output-json")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for item in args.run:
        rows.extend(analyze_run(_parse_run_spec(item)))
    summaries = _summarize(rows)
    (output_dir / "compliance_rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    (output_dir / "compliance_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    _write_rows_csv(rows, output_dir / "compliance_rows.csv")
    _write_summary_md(summaries, output_dir / "compliance_summary.md")
    _write_scatter_svg(rows, output_dir / "compliance_scatter.svg")
    _write_manual_report(
        rows,
        output_dir / "manual_deepseek_failures.md",
        model=args.manual_model,
        workload=args.manual_workload,
        limit=args.manual_limit,
    )
    print((output_dir / "compliance_summary.md").read_text())
    print(f"Wrote {output_dir / 'compliance_scatter.svg'}")
    print(f"Wrote {output_dir / 'manual_deepseek_failures.md'}")
    if args.boundary_output_md or args.boundary_output_json:
        boundary = _boundary_summary(rows, bootstrap_specs=args.boundary_bootstrap)
        if args.boundary_output_json:
            out_json = Path(args.boundary_output_json)
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(boundary, indent=2) + "\n")
            print(f"Wrote {out_json}")
        if args.boundary_output_md:
            out_md = Path(args.boundary_output_md)
            out_md.parent.mkdir(parents=True, exist_ok=True)
            _write_boundary_md(boundary, out_md)
            print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
