#!/usr/bin/env python3
"""Summarize exact-path and success-filtered VANTAGE speedups.

This script uses existing per-task completion artifacts.  It does not create
new model outputs.  The exact-path table uses the fp32/sdpa backend-isolation
run because that path passed byte-parity checks for both PLD and VANTAGE on
the controlled headline workloads.  The success-filter table answers the
reviewer question "does the decoder still speed up successful edits?" by
recomputing PLD and VANTAGE throughput after filtering to tasks where both
compared decoders satisfy the chosen task-quality predicate.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
from pathlib import Path
from typing import Any, Callable


DEFAULT_BACKEND_ROOT = Path("artifacts/vantage_transpld/modal/backend_isolation_20260516_v1")
DEFAULT_OUTPUT_DIR = Path("artifacts/vantage_transpld/tables/exact_path_success_20260518_v1")
PLD = "blazedit_pld_w128_n10"
METHOD = "vantage_frozen_transpld"
WORKLOAD_ORDER = [
    "Zero drift",
    "Field substitution",
    "Identifier-style substitution",
    "Mixed",
]
_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(?P<code>.*?)```", re.DOTALL | re.IGNORECASE)
_IDENT_CHARS = r"A-Za-z0-9_"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _output(row: dict[str, Any], method: str) -> dict[str, Any]:
    return dict(((row.get("outputs") or {}).get(method) or {}))


def _output_text(row: dict[str, Any], method: str) -> str:
    out = _output(row, method)
    return str(out.get("raw_text") if out.get("raw_text") is not None else out.get("text") or "")


def _extract_code(text: str) -> str:
    match = _FENCE_RE.search(text)
    if match:
        return match.group("code").strip()
    return text.strip()


def _syntax_ok(text: str) -> bool:
    code = _extract_code(text)
    if not code:
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _boundary_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(str(term))
    term_s = str(term)
    if term_s.startswith(".") and re.search(r"[A-Za-z0-9_]", term_s):
        return re.compile(rf"{escaped}(?![{_IDENT_CHARS}])")
    if re.search(r"[A-Za-z0-9_]", term_s):
        return re.compile(rf"(?<![{_IDENT_CHARS}]){escaped}(?![{_IDENT_CHARS}])")
    return re.compile(escaped)


def _contains_boundary(text: str, term: str) -> bool:
    return bool(_boundary_pattern(term).search(text))


def _rewrite_pairs(row: dict[str, Any]) -> dict[str, str]:
    pairs = (row.get("metadata") or {}).get("rewrite_pairs") or row.get("rewrite_pairs") or {}
    if not isinstance(pairs, dict):
        return {}
    return {str(k): str(v) for k, v in pairs.items() if str(k) and str(v) and str(k) != str(v)}


def _rewrite_compliant(row: dict[str, Any], method: str) -> bool | None:
    pairs = _rewrite_pairs(row)
    if not pairs:
        return None
    code = _extract_code(_output_text(row, method))
    return all(_contains_boundary(code, new) for new in pairs.values()) and not any(
        _contains_boundary(code, old) for old in pairs.keys()
    )


def _exact_target(row: dict[str, Any], method: str) -> bool:
    target = str(row.get("deterministic_target") or "").strip()
    return bool(target) and _extract_code(_output_text(row, method)).strip() == target


def _tps(rows: list[dict[str, Any]], method: str) -> float:
    tokens = 0
    wall_us = 0.0
    for row in rows:
        out = _output(row, method)
        if out.get("n_new_tokens") is None or out.get("wall_us") is None:
            continue
        tokens += int(out["n_new_tokens"])
        wall_us += float(out["wall_us"])
    return tokens / (wall_us / 1e6) if wall_us > 0 else 0.0


def _ratio(rows: list[dict[str, Any]]) -> float:
    base = _tps(rows, PLD)
    val = _tps(rows, METHOD)
    return val / base if base > 0 else 0.0


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _steps_path_for_completions(path: Path) -> Path:
    return path.with_name("steps.jsonl")


def _transpld_route_tasks(steps_path: Path) -> set[str]:
    if not steps_path.exists():
        return set()
    tasks: set[str] = set()
    for row in _load_jsonl(steps_path):
        if row.get("method") != METHOD:
            continue
        if row.get("proposal_route") == "transpld":
            task_id = row.get("task_id")
            if task_id is not None:
                tasks.add(str(task_id))
    return tasks


def _per_task_speedup(row: dict[str, Any]) -> float | None:
    base = _output(row, PLD)
    val = _output(row, METHOD)
    try:
        base_tps = float(base["n_new_tokens"]) / (float(base["wall_us"]) / 1e6)
        val_tps = float(val["n_new_tokens"]) / (float(val["wall_us"]) / 1e6)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    return val_tps / base_tps if base_tps > 0 else None


def _method_latency_ms(row: dict[str, Any], method: str) -> float | None:
    out = _output(row, method)
    try:
        return float(out["wall_us"]) / 1000.0
    except (KeyError, TypeError, ValueError):
        return None


def _tail_row(workload: str, rows: list[dict[str, Any]], completions_path: Path) -> dict[str, Any]:
    speedups: list[float] = []
    nh_latencies_ms: list[float] = []
    transformed_tasks = _transpld_route_tasks(_steps_path_for_completions(completions_path))
    regression_task_ids: list[str] = []
    transformed_route_regression_task_ids: list[str] = []
    for row in rows:
        ratio = _per_task_speedup(row)
        latency_ms = _method_latency_ms(row, METHOD)
        if ratio is None or latency_ms is None:
            continue
        task_id = str(row.get("task_id"))
        speedups.append(ratio)
        nh_latencies_ms.append(latency_ms)
        if ratio < 1.0:
            regression_task_ids.append(task_id)
            if task_id in transformed_tasks:
                transformed_route_regression_task_ids.append(task_id)
    return {
        "workload": workload,
        "tasks": len(speedups),
        "regressions": len(regression_task_ids),
        "transpld_route_regressions": len(transformed_route_regression_task_ids),
        "transpld_route_tasks": len(transformed_tasks),
        "worst_speedup": min(speedups) if speedups else 0.0,
        "speedup_p05": _percentile(speedups, 0.05),
        "speedup_p50": _percentile(speedups, 0.50),
        "speedup_p95": _percentile(speedups, 0.95),
        "latency_p50_ms": _percentile(nh_latencies_ms, 0.50),
        "latency_p95_ms": _percentile(nh_latencies_ms, 0.95),
        "latency_p99_ms": _percentile(nh_latencies_ms, 0.99),
        "completions_path": str(completions_path),
        "steps_path": str(_steps_path_for_completions(completions_path)),
    }


def _bootstrap_ratio(rows: list[dict[str, Any]], n_boot: int, seed: int) -> dict[str, Any]:
    point = _ratio(rows)
    if not rows:
        return {"ratio": 0.0, "ci95": [0.0, 0.0], "p_gt_1": 0.0}
    rng = random.Random(seed)
    n = len(rows)
    samples = []
    for _ in range(n_boot):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        samples.append(_ratio(sample))
    return {
        "ratio": point,
        "ci95": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
        "p_gt_1": sum(1 for x in samples if x > 1.0) / len(samples),
    }


def _workload_from_run_tag(run_tag: str) -> str:
    low = run_tag.lower()
    if "zero100" in low:
        return "Zero drift"
    if "field100" in low:
        return "Field substitution"
    if "style100" in low:
        return "Identifier-style substitution"
    if "mixed100" in low:
        return "Mixed"
    return run_tag


def _fp32_sdpa_completion_paths(root: Path) -> list[Path]:
    paths = sorted(root.glob("*fp32_sdpa*100*/eval/completions.jsonl"))
    return sorted(paths, key=lambda p: WORKLOAD_ORDER.index(_workload_from_run_tag(p.parent.parent.name)))


def _parity(rows: list[dict[str, Any]], method: str) -> str:
    total = 0
    matches = 0
    for row in rows:
        if method not in (row.get("outputs") or {}) or "vanilla" not in (row.get("outputs") or {}):
            continue
        total += 1
        matches += int(_output_text(row, method) == _output_text(row, "vanilla"))
    return f"{matches}/{total}" if total else "unavailable"


def _quality_counts(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any], str], bool | None]) -> dict[str, int]:
    eligible = 0
    both = 0
    for row in rows:
        pld_ok = predicate(row, PLD)
        nh_ok = predicate(row, METHOD)
        if pld_ok is None or nh_ok is None:
            continue
        eligible += 1
        both += int(bool(pld_ok) and bool(nh_ok))
    return {"both": both, "eligible": eligible}


def summarize(root: Path, *, n_boot: int, seed: int) -> dict[str, Any]:
    exact_rows: list[dict[str, Any]] = []
    success_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    for path in _fp32_sdpa_completion_paths(root):
        rows = _load_jsonl(path)
        workload = _workload_from_run_tag(path.parent.parent.name)
        boot = _bootstrap_ratio(rows, n_boot, seed)
        exact_rows.append(
            {
                "workload": workload,
                "n": len(rows),
                "pld_tps": _tps(rows, PLD),
                "vantage_tps": _tps(rows, METHOD),
                "ratio": boot["ratio"],
                "ci95": boot["ci95"],
                "p_gt_1": boot["p_gt_1"],
                "pld_parity": _parity(rows, PLD),
                "vantage_parity": _parity(rows, METHOD),
                "completions_path": str(path),
            }
        )

        tail_rows.append(_tail_row(workload, rows, path))

        criteria: list[tuple[str, Callable[[dict[str, Any], str], bool | None], str]] = [
            ("exact target", _exact_target, "both PLD and VANTAGE exactly match the deterministic target after code extraction"),
            ("syntax-valid", lambda r, m: _syntax_ok(_output_text(r, m)), "both PLD and VANTAGE parse as Python after code extraction"),
            ("rewrite-compliant", _rewrite_compliant, "rewrite-map rows only; both PLD and VANTAGE contain new terms and omit old terms under boundary-aware matching"),
        ]
        for criterion, pred, definition in criteria:
            selected = [row for row in rows if pred(row, PLD) is True and pred(row, METHOD) is True]
            counts = _quality_counts(rows, pred)
            if counts["eligible"] == 0:
                continue
            boot_sel = _bootstrap_ratio(selected, n_boot, seed)
            success_rows.append(
                {
                    "workload": workload,
                    "criterion": criterion,
                    "definition": definition,
                    "selected_tasks": len(selected),
                    "eligible_tasks": counts["eligible"],
                    "pld_tps": _tps(selected, PLD),
                    "vantage_tps": _tps(selected, METHOD),
                    "ratio": boot_sel["ratio"],
                    "ci95": boot_sel["ci95"],
                    "p_gt_1": boot_sel["p_gt_1"],
                    "completions_path": str(path),
                }
            )
    return {
        "schema": "vantage/transpld_exact_path_success/v1",
        "backend": "fp32/sdpa",
        "method": METHOD,
        "baseline": PLD,
        "n_boot": n_boot,
        "seed": seed,
        "exact_path_rows": exact_rows,
        "success_filtered_rows": success_rows,
        "tail_rows": tail_rows,
    }


def _fmt(x: float | int | None, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{digits}f}"


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# TransPLD Exact-Path And Successful-Edit Speedups",
        "",
        "Generated from existing fp32/sdpa backend-isolation completion artifacts. This path passed byte-parity checks for PLD and VANTAGE on every controlled workload, so these rows are exact-path throughput evidence rather than bf16 timing diagnostics.",
        "",
        "## Exact fp32/sdpa Headline Candidate",
        "",
        "| Workload | n | PLD tok/s | VANTAGE tok/s | VANTAGE/PLD | 95% CI | PLD parity | VANTAGE parity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["exact_path_rows"]:
        lo, hi = row["ci95"]
        lines.append(
            f"| {row['workload']} | {row['n']} | {_fmt(row['pld_tps'], 1)} | "
            f"{_fmt(row['vantage_tps'], 1)} | {_fmt(row['ratio'])} | "
            f"[{_fmt(lo)}, {_fmt(hi)}] | {row['pld_parity']} | {row['vantage_parity']} |"
        )

    lines += [
        "",
        "## Successful-Edit Slices",
        "",
        "Rows filter to tasks where both compared decoders satisfy the criterion, then recompute generated-token throughput on the selected tasks. Rewrite compliance is only defined for rows with rewrite pairs, so zero drift has no rewrite-compliance row and mixed uses only its rewrite-map subset.",
        "",
        "| Workload | Success criterion | Selected / eligible tasks | PLD tok/s | VANTAGE tok/s | VANTAGE/PLD | 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["success_filtered_rows"]:
        lo, hi = row["ci95"]
        lines.append(
            f"| {row['workload']} | {row['criterion']} | {row['selected_tasks']}/{row['eligible_tasks']} | "
            f"{_fmt(row['pld_tps'], 1)} | {_fmt(row['vantage_tps'], 1)} | "
            f"{_fmt(row['ratio'])} | [{_fmt(lo)}, {_fmt(hi)}] |"
        )
    lines += [
        "",
        "## Exact-Path Tail Diagnostics",
        "",
        "Rows compute per-task generated-token speedup and VANTAGE latency from the same fp32/sdpa completion artifacts used for the headline table. rewrite-view-route regressions count tasks with at least one transformed-view route in `steps.jsonl` and per-task speedup below 1.",
        "",
        "| Workload | Regressions | rewrite-view-route regressions | Worst speedup | Speedup p05/p50/p95 | VANTAGE latency p50/p95/p99 ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["tail_rows"]:
        lines.append(
            f"| {row['workload']} | {row['regressions']}/{row['tasks']} | "
            f"{row['transpld_route_regressions']}/{row['transpld_route_tasks']} | "
            f"{_fmt(row['worst_speedup'])} | "
            f"{_fmt(row['speedup_p05'])}/{_fmt(row['speedup_p50'])}/{_fmt(row['speedup_p95'])} | "
            f"{_fmt(row['latency_p50_ms'], 1)}/{_fmt(row['latency_p95_ms'], 1)}/{_fmt(row['latency_p99_ms'], 1)} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_BACKEND_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(Path(args.root), n_boot=args.n_boot, seed=args.seed)
    (out_dir / "exact_path_success.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "exact_path_success.md").write_text(markdown(summary))
    print(markdown(summary))


if __name__ == "__main__":
    main()
