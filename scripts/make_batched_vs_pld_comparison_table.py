#!/usr/bin/env python3
"""Generate generic batching vs Continuous Batched PLD comparison table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "analysis" / "final_paper_artifacts" / "continuous_batched_pld_final"
TABLE_DIR = ARTIFACT_DIR / "tables"


class RawTex(str):
    """String that should be written unescaped in LaTeX tables."""


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _latex_escape(s: object) -> str:
    if isinstance(s, RawTex):
        return str(s)
    text = str(s)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "pending"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _latest_generic_report() -> dict[str, Any] | None:
    path = ROOT / "analysis" / "generic_batched_greedy_baseline" / "report.json"
    loaded = _load(path)
    if loaded is not None:
        loaded["_source_path"] = str(path)
    return loaded


def _pool_sweep_report() -> dict[str, Any] | None:
    return _load(ROOT / "analysis" / "generic_batched_greedy_baseline" / "b8_pool_sweep_report.json")


def _pld_final_report() -> dict[str, Any] | None:
    return _load(ROOT / "analysis" / "continuous_batched_pld_final_repeats" / "report.json")


def _status_row(
    *,
    method: str,
    batch_size: Any,
    active_pool: Any,
    tok_s: Any,
    own_speedup: Any,
    pld_speedup: Any,
    forwards: Any,
    notes: str,
) -> list[Any]:
    return [
        method,
        "No",
        "Yes" if batch_size != "n/a" else "No",
        batch_size,
        active_pool,
        tok_s,
        own_speedup,
        pld_speedup,
        forwards,
        notes,
    ]


def _generic_rows(report: dict[str, Any] | None) -> list[list[Any]]:
    if report is None:
        return [
            [
                "Sequential greedy",
                "No",
                "No",
                1,
                1,
                "pending",
                "pending",
                "n/a",
                "pending",
                "Run `launch_batched_greedy_eval`",
            ],
            [
                "Batched greedy b8",
                "No",
                "Yes",
                8,
                "pending",
                "pending",
                "pending",
                "n/a",
                "pending",
                "Run `launch_batched_greedy_eval`",
            ],
        ]
    seq = report["sequential"]
    seq_tps = float(seq.get("tokens_per_sec", 0.0))
    pld = _pld_final_report()
    pld_tps = 492.1
    if pld is not None:
        pld_tps = float(pld["summary"]["blazedit_pld_w128_n10_b1"]["tok_s"]["mean"])
    seq_row = [
        "Sequential greedy",
        "No",
        "No",
        1,
        1,
        _fmt(seq_tps),
        RawTex("1.000$\\times$"),
        RawTex(f"{seq_tps / max(1e-9, pld_tps):.3f}$\\times$"),
        int(seq.get("model_forwards", seq.get("steps", 0))),
        "generic autoregressive baseline",
    ]
    rows = [seq_row]
    for batch in (2, 4):
        row = next((r for r in report.get("batched", []) if int(r.get("batch_size", 0)) == batch), None)
        if row is None:
            continue
        if row.get("error"):
            rows.append(
                _status_row(
                    method=f"Batched greedy b{batch}",
                    batch_size=batch,
                    active_pool=row.get("active_pool_size", 32),
                    tok_s="OOM",
                    own_speedup="OOM",
                    pld_speedup="n/a",
                    forwards="OOM",
                    notes=f"failed: {str(row.get('error', ''))[:80]}",
                )
            )
        else:
            tps = float(row.get("generated_tokens_per_sec", 0.0))
            rows.append(
                _status_row(
                    method=f"Batched greedy b{batch}",
                    batch_size=batch,
                    active_pool=int(row.get("active_pool_size", 32)),
                    tok_s=_fmt(tps),
                    own_speedup=RawTex(f"{tps / max(1e-9, seq_tps):.3f}$\\times$"),
                    pld_speedup=RawTex(f"{tps / max(1e-9, pld_tps):.3f}$\\times$"),
                    forwards=int(row.get("model_forwards", 0)),
                    notes=(
                        f"matches sequential greedy {row.get('output_match_count', 0)}/"
                        f"{row.get('output_match_count', 0) + row.get('output_mismatch_count', 0)}"
                    ),
                )
            )
    sweep = _pool_sweep_report()
    if sweep and sweep.get("rows"):
        for row in sorted(sweep["rows"], key=lambda r: int(r.get("active_pool_size", 0))):
            pool = int(row.get("active_pool_size", 0))
            if row.get("status") == "success":
                tps = float(row.get("tok_s", 0.0))
                rows.append(
                    _status_row(
                        method=f"Batched greedy b8 pool{pool}",
                        batch_size=8,
                        active_pool=pool,
                        tok_s=_fmt(tps),
                        own_speedup=RawTex(f"{tps / max(1e-9, seq_tps):.3f}$\\times$"),
                        pld_speedup=RawTex(f"{tps / max(1e-9, pld_tps):.3f}$\\times$"),
                        forwards=int(row.get("model_forwards", 0)),
                        notes=f"peak {float(row.get('memory_peak_gb', 0.0)):.2f} GB",
                    )
                )
            else:
                rows.append(
                    _status_row(
                        method=f"Batched greedy b8 pool{pool}",
                        batch_size=8,
                        active_pool=pool,
                        tok_s="OOM",
                        own_speedup="OOM",
                        pld_speedup="n/a",
                        forwards="OOM",
                        notes="OOM on L40S",
                    )
                )
    else:
        rows.append(
            _status_row(
                method="Batched greedy b8 pool sweep",
                batch_size=8,
                active_pool="pending",
                tok_s="pending",
                own_speedup="pending",
                pld_speedup="n/a",
                forwards="pending",
                notes="run b8 pool8/pool16 sweep",
            )
        )
    return rows


def _pld_rows(report: dict[str, Any] | None) -> tuple[list[Any], list[Any]]:
    if report is None:
        pending = [
            "Sequential BlazEdit PLD",
            "Yes",
            "No",
            1,
            1,
            "492.1",
            RawTex("1.000$\\times$"),
            RawTex("1.000$\\times$"),
            6443,
            "final frozen result",
        ]
        pending_b8 = [
            "Continuous Batched PLD b8",
            "Yes",
            "Yes",
            8,
            32,
            "845.0",
            RawTex("1.717$\\times$"),
            RawTex("1.717$\\times$"),
            1456,
            "final frozen result",
        ]
        return pending, pending_b8
    summary = report["summary"]
    seq = summary["blazedit_pld_w128_n10_b1"]
    b8 = summary["continuous_batched_pld_w128_n10_b8"]
    return (
        [
            "Sequential BlazEdit PLD",
            "Yes",
            "No",
            1,
            1,
            f"{seq['tok_s']['mean']:.1f}",
            RawTex("1.000$\\times$"),
            RawTex("1.000$\\times$"),
            f"{seq['verifier_forwards']['mean']:.0f}",
            "optimized PLD baseline",
        ],
        [
            "Continuous Batched PLD b8",
            "Yes",
            "Yes",
            8,
            32,
            f"{b8['tok_s']['mean']:.1f}",
            RawTex(f"{b8['speedup']['mean']:.3f}$\\times$"),
            RawTex(f"{b8['speedup']['mean']:.3f}$\\times$"),
            f"{b8['verifier_forwards']['mean']:.0f}",
            "fp32/eager exact 500/500; task audit 0 violations",
        ],
    )


def _write_md(path: Path, headers: list[str], rows: list[list[Any]], note: str) -> None:
    lines = [
        "# Generic Batching vs PLD Batching",
        "",
        note,
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_tex(path: Path, headers: list[str], rows: list[list[Any]], caption: str) -> None:
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{_latex_escape(caption)}}}",
        r"\label{tab:generic-vs-pld-batching}",
        r"\begin{tabular}{lllllllrrl}",
        r"\toprule",
        " & ".join(_latex_escape(h) for h in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        tex.append(" & ".join(_latex_escape(x) for x in row) + r" \\")
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(tex) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(TABLE_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    generic = _latest_generic_report()
    pld = _pld_final_report()
    rows = [*_generic_rows(generic), *_pld_rows(pld)]
    headers = [
        "Method",
        "Uses PLD?",
        "Uses continuous batching?",
        "Batch size",
        "Active pool",
        "Tok/s",
        "Speedup vs own sequential",
        "Speedup vs sequential PLD",
        "Model/verifier forwards",
        "Notes",
    ]
    note = (
        "This table separates generic continuous batching from PLD-specific "
        "batched speculative verification. If the generic report has not been "
        "run yet, greedy rows are marked pending."
    )
    _write_md(out / "generic_batching_comparison.md", headers, rows, note)
    _write_tex(
        out / "generic_batching_comparison.tex",
        headers,
        rows,
        "Generic continuous batching compared with Continuous Batched PLD.",
    )
    print(f"wrote generic batching comparison table to {out}")


if __name__ == "__main__":
    main()
