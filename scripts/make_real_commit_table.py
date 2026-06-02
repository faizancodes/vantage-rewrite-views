#!/usr/bin/env python3
"""Build the paper-facing real-commit/refactor benchmark table.

Input is a `run_eagle_eval.py` completions JSONL containing `vanilla`,
`blazedit_pld_w128_n10`, and the frozen VANTAGE method. Rows are grouped by
manifest `drift_family` into the two reviewer-facing workloads:

* Real rename commits
* Real field migration commits
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
from pathlib import Path
from typing import Any


_FENCE_RE = re.compile(r"^\s*```(?:[A-Za-z0-9_+-]+)?\s*\n(?P<body>.*?)(?:\n```\s*)?$", re.DOTALL)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    return match.group("body").strip() if match else stripped


def _norm(text: str) -> str:
    return "\n".join(line.rstrip() for line in _strip_code_fence(text).strip().splitlines()).strip()


def _syntax_ok(text: str) -> bool:
    try:
        ast.parse(_strip_code_fence(text))
        return True
    except SyntaxError:
        return False


def _group_name(row: dict[str, Any]) -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    family = str(row.get("drift_family") or meta.get("drift_family") or "")
    pairs = row.get("rewrite_pairs") or meta.get("rewrite_pairs") or {}
    if "field" in family or any("." in str(k) or "." in str(v) for k, v in pairs.items()):
        return "Real field migration commits"
    return "Real rename commits"


def _tps(rows: list[dict[str, Any]], method: str) -> float:
    tokens = 0
    wall_us = 0.0
    for row in rows:
        out = (row.get("outputs") or {}).get(method) or {}
        tokens += int(out.get("n_new_tokens") or 0)
        wall_us += float(out.get("wall_us") or 0.0)
    return tokens / (wall_us / 1e6) if wall_us > 0 else 0.0


def _ratio(rows: list[dict[str, Any]], method: str, baseline: str) -> float:
    base = _tps(rows, baseline)
    val = _tps(rows, method)
    return val / base if base > 0 else 0.0


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def _bootstrap_ci(rows: list[dict[str, Any]], method: str, baseline: str, *, n_boot: int, seed: int) -> tuple[float, float]:
    if not rows:
        return 0.0, 0.0
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        samples.append(_ratio(sample, method, baseline))
    return _percentile(samples, 0.025), _percentile(samples, 0.975)


def _quality(rows: list[dict[str, Any]], method: str) -> dict[str, float]:
    exact = 0
    syntax = 0
    parity = 0
    denom = 0
    for row in rows:
        outputs = row.get("outputs") or {}
        out = outputs.get(method) or {}
        vanilla = outputs.get("vanilla") or {}
        text = str(out.get("text") or out.get("raw_text") or "")
        vanilla_text = str(vanilla.get("text") or vanilla.get("raw_text") or "")
        target = str(row.get("deterministic_target") or "")
        if not text:
            continue
        denom += 1
        exact += int(bool(target.strip()) and _norm(text) == _norm(target))
        syntax += int(_syntax_ok(text))
        parity += int(bool(vanilla_text) and _norm(text) == _norm(vanilla_text))
    return {
        "exact_target": exact / denom if denom else 0.0,
        "syntax_ok": syntax / denom if denom else 0.0,
        "vanilla_parity": parity / denom if denom else 0.0,
    }


def build_table(
    rows: list[dict[str, Any]],
    *,
    pld_method: str,
    vantage_method: str,
    n_boot: int,
    seed: int,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_group_name(row), []).append(row)
    out = []
    for name in ["Real rename commits", "Real field migration commits"]:
        group = groups.get(name, [])
        if not group:
            continue
        pld = _tps(group, pld_method)
        nh = _tps(group, vantage_method)
        ratio = nh / pld if pld > 0 else 0.0
        lo, hi = _bootstrap_ci(group, vantage_method, pld_method, n_boot=n_boot, seed=seed)
        q = _quality(group, vantage_method)
        out.append(
            {
                "workload": name,
                "n": len(group),
                "pld_tps": pld,
                "vantage_tps": nh,
                "ratio": ratio,
                "ci95": [lo, hi],
                "quality": q,
            }
        )
    return out


def _quality_string(q: dict[str, float]) -> str:
    return (
        f"target {100*q['exact_target']:.1f}\\%, "
        f"syntax {100*q['syntax_ok']:.1f}\\%, "
        f"parity {100*q['vanilla_parity']:.1f}\\%"
    )


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Real-commit/refactor benchmark",
        "",
        "| Workload | n | PLD tok/s | VANTAGE tok/s | Ratio | Quality |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lo, hi = row["ci95"]
        lines.append(
            f"| {row['workload']} | {row['n']} | {row['pld_tps']:.2f} | "
            f"{row['vantage_tps']:.2f} | {row['ratio']:.3f} [{lo:.3f},{hi:.3f}] | "
            f"{_quality_string(row['quality'])} |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_latex(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        r"\begin{table}[h]",
        r"\centering\scriptsize",
        r"\caption{Real-commit/refactor benchmark. Quality reports exact target match, Python syntax validity, and vanilla-output parity for VANTAGE.}",
        r"\label{tab:real-commit}",
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"Workload & $n$ & PLD & VANTAGE & Ratio & Quality \\",
        r"\midrule",
    ]
    for row in rows:
        lo, hi = row["ci95"]
        lines.append(
            f"{row['workload']} & {row['n']} & {row['pld_tps']:.2f} & "
            f"{row['vantage_tps']:.2f} & {row['ratio']:.3f} [{lo:.3f},{hi:.3f}] & "
            f"{_quality_string(row['quality'])} " + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--completions", required=True)
    p.add_argument("--pld-method", default="blazedit_pld_w128_n10")
    p.add_argument("--vantage-method", default="vantage_frozen_transpld")
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    p.add_argument("--output-tex", required=True)
    args = p.parse_args()

    table = build_table(
        _load_jsonl(Path(args.completions)),
        pld_method=args.pld_method,
        vantage_method=args.vantage_method,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(table, indent=2) + "\n")
    write_markdown(table, Path(args.output_md))
    write_latex(table, Path(args.output_tex))
    print(Path(args.output_md).read_text())


if __name__ == "__main__":
    main()
