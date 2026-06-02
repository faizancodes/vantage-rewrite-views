"""Diagnose VANTAGE-Residual output mismatches against a PLD baseline."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_RUN_DIR = (
    Path("artifacts")
    / "vantage_residual"
    / "runs"
    / "vantage_residual_smoke50_v1"
    / "eval"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _first_mismatch(a: str, b: str) -> int | None:
    upto = min(len(a), len(b))
    for idx in range(upto):
        if a[idx] != b[idx]:
            return idx
    if len(a) != len(b):
        return upto
    return None


def _context(text: str, pos: int | None, radius: int = 80) -> str:
    if pos is None:
        return ""
    lo = max(0, pos - radius)
    hi = min(len(text), pos + radius)
    return text[lo:hi]


def _steps_by_task_method(steps: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in steps:
        task_id = row.get("task_id")
        method = row.get("method")
        if task_id is None or method is None:
            continue
        grouped[(str(task_id), str(method))].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: int(r.get("step", 0) or 0))
    return grouped


def _step_summary_before_mismatch(
    rows: list[dict[str, Any]],
    *,
    mismatch_char: int | None,
) -> dict[str, Any]:
    # The current artifacts do not include per-step decoded character spans, so
    # "before mismatch" cannot be aligned exactly.  We still report aggregate
    # step behavior for the task and mark alignment as unavailable.
    del mismatch_char
    trigger_rows = [r for r in rows if r.get("mtp_triggered") is True]
    token0_reject_rows = [r for r in rows if r.get("mtp_token0_rejected") is True]
    partial_accept_rows = [
        r
        for r in rows
        if r.get("mtp_triggered") is True
        and (r.get("mtp_accepted_prefix_len") or 0) not in (0, None)
    ]
    extra_verify = sum(int(r.get("mtp_extra_verify_calls", 0) or 0) for r in rows)
    return {
        "step_alignment_to_character_mismatch": "unavailable",
        "steps": len(rows),
        "residual_triggers": len(trigger_rows),
        "residual_token0_rejects": len(token0_reject_rows),
        "residual_partial_or_full_accept_steps": len(partial_accept_rows),
        "residual_extra_verify_calls": extra_verify,
        "residual_head_compute_us": sum(float(r.get("mtp_head_compute_us", 0.0) or 0.0) for r in rows),
        "residual_verify_extra_us": sum(float(r.get("mtp_verify_extra_us", 0.0) or 0.0) for r in rows),
        "residual_total_overhead_us": sum(float(r.get("mtp_total_overhead_us", 0.0) or 0.0) for r in rows),
    }


def _write_markdown(
    *,
    path: Path,
    payload: dict[str, Any],
    pld_method: str,
    residual_method: str,
) -> None:
    lines = [
        "# VANTAGE-Residual Mismatch Diagnostics",
        "",
        f"PLD method: `{pld_method}`",
        f"Residual method: `{residual_method}`",
        "",
        "## Summary",
        "",
        f"- Tasks: `{payload['summary']['tasks']}`",
        f"- Mismatches: `{payload['summary']['mismatches']}`",
        f"- Matches: `{payload['summary']['matches']}`",
        f"- Root-cause category: `{payload['summary']['root_cause_category']}`",
        "",
        "The existing smoke artifacts include decoded outputs and per-step timing,",
        "but they do not include token IDs, correction-token traces, cache lengths,",
        "or character spans per step. Token-level and cache-level root cause cannot",
        "be proven from this run alone.",
        "",
        "## Mismatched Tasks",
        "",
        "| task | first mismatch char | PLD chars | residual chars | length delta | residual triggers | token0 rejects | extra verify calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["mismatches"]:
        step = item["residual_step_summary"]
        lines.append(
            "| {task} | {pos} | {pld_len} | {res_len} | {delta} | {trig} | {rej} | {extra} |".format(
                task=item["task_id"],
                pos=item["first_mismatch_char_pos"],
                pld_len=item["pld_text_len"],
                res_len=item["residual_text_len"],
                delta=item["length_delta_residual_minus_pld"],
                trig=step["residual_triggers"],
                rej=step["residual_token0_rejects"],
                extra=step["residual_extra_verify_calls"],
            )
        )
    lines.extend(
        [
            "",
            "## Diagnostic Limits",
            "",
            "- Token IDs: unavailable in `completions.jsonl`.",
            "- Verifier correction token at mismatch: unavailable.",
            "- Cache crop/update state at mismatch: unavailable.",
            "- EOS/max-length event before mismatch: not alignable without per-step output spans.",
            "- Classification is therefore conservative: metadata-insufficient bf16/SDPA output mismatch.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--completions-jsonl",
        "--outputs-jsonl",
        "--pld-outputs",
        "--residual-outputs",
        dest="completions_jsonl",
        type=Path,
        default=DEFAULT_RUN_DIR / "completions.jsonl",
        help="Combined completions JSONL with per-task outputs.",
    )
    ap.add_argument(
        "--run-summary",
        "--aggregate",
        dest="aggregate",
        type=Path,
        default=DEFAULT_RUN_DIR / "aggregate.json",
        help="Aggregate JSON for the run.",
    )
    ap.add_argument(
        "--steps-jsonl",
        type=Path,
        default=DEFAULT_RUN_DIR / "steps.jsonl",
        help="Optional per-step JSONL for residual event summaries.",
    )
    ap.add_argument("--pld-method", default="blazedit_pld_w128_n10")
    ap.add_argument("--residual-method", default="vantage_residual_k4_t4_w128_n10")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts")
        / "vantage_residual"
        / "phase2_mismatch_diagnostics"
        / "smoke50_v1",
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    completions = _load_jsonl(args.completions_jsonl)
    aggregate = _load_json(args.aggregate) if args.aggregate.exists() else {}
    steps = _load_jsonl(args.steps_jsonl) if args.steps_jsonl.exists() else []
    grouped_steps = _steps_by_task_method(steps)

    mismatches: list[dict[str, Any]] = []
    matches = 0
    missing = 0
    for row in completions:
        outputs = row.get("outputs", {})
        pld = outputs.get(args.pld_method)
        residual = outputs.get(args.residual_method)
        if not isinstance(pld, dict) or not isinstance(residual, dict):
            missing += 1
            continue
        pld_text = str(pld.get("text", pld.get("raw_text", "")))
        residual_text = str(residual.get("text", residual.get("raw_text", "")))
        mismatch_pos = _first_mismatch(pld_text, residual_text)
        if mismatch_pos is None:
            matches += 1
            continue
        task_id = str(row.get("task_id", "unknown"))
        residual_steps = grouped_steps.get((task_id, args.residual_method), [])
        item = {
            "task_id": task_id,
            "first_mismatch_char_pos": mismatch_pos,
            "pld_text_len": len(pld_text),
            "residual_text_len": len(residual_text),
            "length_delta_residual_minus_pld": len(residual_text) - len(pld_text),
            "pld_n_new_tokens": pld.get("n_new_tokens"),
            "residual_n_new_tokens": residual.get("n_new_tokens"),
            "pld_wall_us": pld.get("wall_us"),
            "residual_wall_us": residual.get("wall_us"),
            "pld_context": _context(pld_text, mismatch_pos),
            "residual_context": _context(residual_text, mismatch_pos),
            "finish_reason_mismatch": "unavailable",
            "token_id_mismatch": "unavailable",
            "verifier_correction_token_at_mismatch": "unavailable",
            "cache_event_at_mismatch": "unavailable",
            "eos_or_max_length_before_mismatch": "unavailable",
            "residual_step_summary": _step_summary_before_mismatch(
                residual_steps,
                mismatch_char=mismatch_pos,
            ),
        }
        mismatches.append(item)

    output_equiv = aggregate.get("output_equivalence", {}) if isinstance(aggregate, dict) else {}
    summary = {
        "tasks": len(completions),
        "matches": matches,
        "mismatches": len(mismatches),
        "missing_method_outputs": missing,
        "pld_method": args.pld_method,
        "residual_method": args.residual_method,
        "aggregate_output_equivalence": output_equiv,
        "root_cause_category": "metadata_insufficient_bf16_output_mismatch",
        "root_cause_note": (
            "Decoded-output mismatches are present, but existing artifacts do not "
            "include token IDs, correction traces, cache-length traces, or per-step "
            "decoded spans. Rerun n=10/n=20 with detailed tracing to distinguish "
            "cache/verification bugs from bf16/SDPA drift."
        ),
    }
    payload = {
        "summary": summary,
        "mismatches": mismatches,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(
        path=args.output_dir / "report.md",
        payload=payload,
        pld_method=args.pld_method,
        residual_method=args.residual_method,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
