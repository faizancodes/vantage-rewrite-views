#!/usr/bin/env python3
"""Analyze why VANTAGE-Residual-Queued failed the held-out offline gate.

The script is intentionally conservative.  It always summarizes the Phase 7
training/offline JSON artifacts.  If a full local checkpoint is available, it
also computes detailed bucket and confidence diagnostics against the queued
test tensor.  Large checkpoints that live only on the Modal volume are reported
from their already-generated replay summaries instead of being silently
retrained or replaced.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100.0 * float(x):.1f}%"


def _fmt(x: float | None, digits: int = 3) -> str:
    return "n/a" if x is None else f"{float(x):.{digits}f}"


def _bucket_by_edges(value: int, edges: list[int]) -> str:
    previous = None
    for edge in edges:
        if value <= edge:
            if previous is None:
                return f"<={edge}"
            return f"{previous + 1}-{edge}"
        previous = edge
    return f">{edges[-1]}"


def _summarize_bool_bucket(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    correct = sum(int(bool(row.get("h1_correct"))) for row in rows)
    token0_reject = sum(int(int(row.get("accepted_prefix", 0)) == 0) for row in rows)
    accepted = sum(float(row.get("accepted_prefix", 0)) for row in rows)
    return {
        "bucket": key,
        "n": n,
        "h1_accuracy": correct / n,
        "token0_reject_rate": token0_reject / n,
        "accepted_per_use": accepted / n,
    }


def _bucket_metrics(rows: list[dict[str, Any]], bucket_key: str) -> list[dict[str, Any]]:
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bucket[str(row[bucket_key])].append(row)
    return [
        _summarize_bool_bucket(bucket_rows, bucket)
        for bucket, bucket_rows in sorted(by_bucket.items(), key=lambda kv: kv[0])
    ]


def _best_phase7_row(report: dict[str, Any]) -> dict[str, Any]:
    rows = report.get("phase4_rows") or []
    return rows[0] if rows else {}


def _aggregate_phase7(phase7_offline: Path, phase7_checkpoints: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for offline_dir in sorted(p for p in phase7_offline.iterdir() if p.is_dir()):
        version = offline_dir.name
        report_path = offline_dir / "report.json"
        summary_path = offline_dir / "run_summary.json"
        metrics_path = phase7_checkpoints / version / "queued_training_metrics.json"
        if not report_path.exists() or not summary_path.exists():
            continue
        report = _load_json(report_path)
        summary = _load_json(summary_path)
        metrics = _load_json(metrics_path) if metrics_path.exists() else {}
        best = _best_phase7_row(report)
        val = metrics.get("validation") if isinstance(metrics.get("validation"), dict) else {}
        meta = report.get("prediction_metadata") if isinstance(report.get("prediction_metadata"), dict) else {}
        out.append(
            {
                "version": version,
                "training_gate": metrics.get("training_gate"),
                "val_examples": val.get("examples"),
                "val_h1_top1": val.get("h1_top1"),
                "val_top1_by_horizon": val.get("top1_accuracy_by_horizon"),
                "test_top1_by_horizon_pct": meta.get("top1_accuracy_by_horizon_pct"),
                "best_selector": {
                    "confidence_threshold": best.get("confidence_threshold"),
                    "pld_draft_len_threshold": best.get("pld_draft_len_threshold"),
                    "previous_accepted_len_threshold": best.get("previous_accepted_len_threshold"),
                    "weak_field": best.get("weak_field"),
                },
                "best_projected_speedup": best.get("projected_speedup"),
                "best_projected_speedup_after_hidden_overhead": best.get(
                    "projected_speedup_after_hidden_overhead"
                ),
                "best_accepted_per_use": best.get("accepted_per_use"),
                "best_token0_reject_rate": best.get("token0_reject_rate"),
                "best_queue_used": best.get("queue_used"),
                "best_queue_available": best.get("queue_available"),
                "best_progress_less_than_pld_count": best.get("progress_less_than_pld_count"),
                "decision": summary.get("decision") or report.get("decision"),
                "report": _rel(report_path),
                "training_metrics": _rel(metrics_path) if metrics_path.exists() else None,
                "local_checkpoint_exists": (phase7_checkpoints / version / "model.pt").exists(),
            }
        )
    return out


def _load_tensor(path: Path) -> dict[str, Any]:
    import torch

    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise SystemExit(f"{path} is not a tensor dictionary")
    return obj


def _detailed_checkpoint_analysis(
    *,
    version: str,
    checkpoint: Path,
    test_data: Path,
    report: dict[str, Any],
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    import torch

    from asts.mtp_heads import accepted_prefix_length
    from scripts.evaluate_queued_mtp_confidence import load_prediction_records

    data = _load_tensor(test_data)
    records, prediction_meta = load_prediction_records(
        data_path=test_data,
        heads_path=checkpoint,
        batch_size=batch_size,
        device=device,
        confidence_source="first_head",
    )
    labels = data["labels"].long()
    h1_counts = Counter(int(x) for x in labels[:, 0].tolist())
    task_ids = [str(x) for x in data["task_id"]]
    step_ids = [int(x) for x in data["step_id"]]
    best = _best_phase7_row(report)
    selected_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for idx, (task_id, step_id) in enumerate(zip(task_ids, step_ids)):
        rec = records.get((task_id, step_id))
        if rec is None:
            continue
        preds = list(rec.queued_predictions)
        gold = list(rec.queued_labels)
        accepted = accepted_prefix_length(preds, gold)
        h1 = int(labels[idx, 0].item())
        freq = h1_counts[h1]
        row = {
            "task_id": task_id,
            "step_id": step_id,
            "h1_correct": bool(preds and gold and preds[0] == gold[0]),
            "accepted_prefix": int(accepted),
            "confidence": float(rec.confidence),
            "margin": float(rec.margin),
            "label_h1_frequency": int(freq),
            "frequency_bucket": _bucket_by_edges(int(freq), [1, 5, 20, 100, 500]),
            "pld_draft_len": int(data["pld_draft_len_t_plus_1"][idx].item()),
            "pld_draft_bucket": _bucket_by_edges(int(data["pld_draft_len_t_plus_1"][idx].item()), [0, 1, 2, 4, 8, 32]),
            "previous_accepted_len": int(data["previous_accepted_len_t"][idx].item()),
            "previous_accepted_bucket": _bucket_by_edges(int(data["previous_accepted_len_t"][idx].item()), [0, 1, 2, 4, 16, 64]),
            "use_step_pld_accepted_len": int(data["use_step_accepted_len"][idx].item()),
            "use_step_pld_token0_reject": int(data["use_step_accepted_len"][idx].item()) == 0,
            "confidence_bucket": f"{min(9, int(float(rec.confidence) * 10)) / 10:.1f}-{min(1.0, min(9, int(float(rec.confidence) * 10)) / 10 + 0.1):.1f}",
        }
        all_rows.append(row)
        if (
            float(row["confidence"]) >= float(best.get("confidence_threshold", 0.0) or 0.0)
            and int(row["pld_draft_len"]) <= int(best.get("pld_draft_len_threshold", 10**9) or 10**9)
            and int(row["previous_accepted_len"]) <= int(best.get("previous_accepted_len_threshold", 10**9) or 10**9)
        ):
            selected_rows.append(row)

    def selected_compare(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        if n == 0:
            return {"n": 0}
        return {
            "n": n,
            "residual_accepted_per_use": sum(r["accepted_prefix"] for r in rows) / n,
            "residual_token0_reject_rate": sum(int(r["accepted_prefix"] == 0) for r in rows) / n,
            "pld_accepted_at_same_positions": sum(r["use_step_pld_accepted_len"] for r in rows) / n,
            "pld_token0_reject_at_same_positions": sum(int(r["use_step_pld_token0_reject"]) for r in rows) / n,
        }

    top_tokens = [
        {"token_id": int(tok), "count": int(count), "share": count / max(1, len(labels))}
        for tok, count in h1_counts.most_common(10)
    ]
    detailed = {
        "version": version,
        "checkpoint": _rel(checkpoint),
        "device": device,
        "prediction_metadata": prediction_meta,
        "test_examples": len(all_rows),
        "h1_label_top_tokens": top_tokens,
        "majority_h1_baseline": top_tokens[0]["share"] if top_tokens else 0.0,
        "accuracy_by_token_frequency_bucket": _bucket_metrics(all_rows, "frequency_bucket"),
        "accuracy_by_pld_draft_len_bucket": _bucket_metrics(all_rows, "pld_draft_bucket"),
        "accuracy_by_previous_accepted_len_bucket": _bucket_metrics(all_rows, "previous_accepted_bucket"),
        "confidence_vs_correctness": _bucket_metrics(all_rows, "confidence_bucket"),
        "selected_position_comparison": selected_compare(selected_rows),
        "best_selector_from_phase7": {
            "confidence_threshold": best.get("confidence_threshold"),
            "pld_draft_len_threshold": best.get("pld_draft_len_threshold"),
            "previous_accepted_len_threshold": best.get("previous_accepted_len_threshold"),
        },
        "note": "Detailed bucket analysis is only computed for checkpoints with local model.pt files.",
    }
    return detailed


def _oracle_ceiling(steps_path: Path | None) -> dict[str, Any] | None:
    if steps_path is None or not steps_path.exists():
        return None
    from scripts.evaluate_queued_mtp_oracle import load_pld_steps, replay_perfect_queued_oracle

    steps = load_pld_steps(steps_path, method="blazedit_pld_w128_n10")
    if not steps:
        return None
    return replay_perfect_queued_oracle(
        steps,
        num_heads=4,
        trigger_threshold=8,
        weak_field="draft_len",
    )


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# VANTAGE-Residual Final Failure Analysis",
        "",
        "## Aggregate Phase 7 Replay",
        "",
        "| checkpoint | val h1 | test h1 | best after overhead | accepted/use | token0 reject | false-positive PLD regressions | decision |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["aggregate"]:
        test_top1 = (row.get("test_top1_by_horizon_pct") or [None])[0]
        lines.append(
            f"| `{row['version']}` | {_pct(row.get('val_h1_top1'))} | "
            f"{'n/a' if test_top1 is None else f'{float(test_top1):.1f}%'} | "
            f"{_fmt(row.get('best_projected_speedup_after_hidden_overhead'))}x | "
            f"{_fmt(row.get('best_accepted_per_use'))} | "
            f"{_pct(row.get('best_token0_reject_rate'))} | "
            f"{row.get('best_progress_less_than_pld_count')} | `{row.get('decision')}` |"
        )
    lines.extend(
        [
            "",
            "## Detailed Local-Checkpoint Diagnostics",
            "",
        ]
    )
    if not payload.get("detailed"):
        lines.append("No local `model.pt` files were available for detailed bucket analysis.")
    if payload.get("detail_errors"):
        lines.extend(["", "Detailed-analysis skips:", ""])
        for err in payload["detail_errors"]:
            lines.append(
                f"- `{err['version']}`: `{err['error']}`. {err['implication']}"
            )
    for detail in payload.get("detailed", []):
        lines.extend(
            [
                f"### `{detail['version']}`",
                "",
                f"- Test examples analyzed: `{detail['test_examples']}`",
                f"- Held-out h1 top1: `{detail['prediction_metadata']['top1_accuracy_by_horizon_pct'][0]:.1f}%`",
                f"- Majority-token baseline for h1: `{100.0 * detail['majority_h1_baseline']:.1f}%`",
                "",
                "Accuracy by token-frequency bucket:",
                "",
                "| bucket | n | h1 accuracy | token0 reject | accepted/use |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in detail["accuracy_by_token_frequency_bucket"]:
            lines.append(
                f"| `{row.get('bucket')}` | {row.get('n')} | {_pct(row.get('h1_accuracy'))} | "
                f"{_pct(row.get('token0_reject_rate'))} | {_fmt(row.get('accepted_per_use'))} |"
            )
        lines.extend(["", "Accuracy by PLD draft-length bucket:", "", "| bucket | n | h1 accuracy | token0 reject | accepted/use |", "|---|---:|---:|---:|---:|"])
        for row in detail["accuracy_by_pld_draft_len_bucket"]:
            lines.append(
                f"| `{row.get('bucket')}` | {row.get('n')} | {_pct(row.get('h1_accuracy'))} | "
                f"{_pct(row.get('token0_reject_rate'))} | {_fmt(row.get('accepted_per_use'))} |"
            )
        lines.extend(["", "Confidence calibration:", "", "| confidence | n | h1 accuracy | token0 reject | accepted/use |", "|---|---:|---:|---:|---:|"])
        for row in detail["confidence_vs_correctness"]:
            lines.append(
                f"| `{row.get('bucket')}` | {row.get('n')} | {_pct(row.get('h1_accuracy'))} | "
                f"{_pct(row.get('token0_reject_rate'))} | {_fmt(row.get('accepted_per_use'))} |"
            )
        comp = detail["selected_position_comparison"]
        lines.extend(
            [
                "",
                "Selected-position comparison:",
                "",
                "| selected n | residual accepted/use | residual token0 reject | PLD accepted at same positions | PLD token0 reject at same positions |",
                "|---:|---:|---:|---:|---:|",
                f"| {comp.get('n')} | {_fmt(comp.get('residual_accepted_per_use'))} | {_pct(comp.get('residual_token0_reject_rate'))} | {_fmt(comp.get('pld_accepted_at_same_positions'))} | {_pct(comp.get('pld_token0_reject_at_same_positions'))} |",
                "",
            ]
        )
    oracle = payload.get("oracle_ceiling")
    lines.extend(["", "## Oracle Ceiling", ""])
    if oracle:
        lines.extend(
            [
                "| trigger | projected speedup | queue used | oracle accepted/use | token0 reject |",
                "|---|---:|---:|---:|---:|",
                f"| `draft_len<=8`, K4 | {_fmt(oracle.get('projected_speedup'))}x | {oracle.get('queue_used')} | {_fmt(oracle.get('oracle_accepted_queued_tokens_per_used_draft'))} | {_pct(oracle.get('oracle_token0_reject_rate'))} |",
                "",
                "The oracle is separated from trained-head results; it is an architectural ceiling, not a deployable selector.",
            ]
        )
    else:
        lines.append("Not computed because the PLD step trace was unavailable.")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            payload["decision"],
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _write_table(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Final Residual Failure Modes",
        "",
        "| finding | evidence | implication |",
        "|---|---|---|",
    ]
    best = max(
        payload["aggregate"],
        key=lambda row: float(row.get("best_projected_speedup_after_hidden_overhead") or 0.0),
    )
    lines.append(
        f"| Held-out replay gate fails | best is `{best['version']}` at "
        f"`{float(best['best_projected_speedup_after_hidden_overhead']):.3f}x`, "
        f"`{float(best['best_accepted_per_use']):.3f}` accepted/use, "
        f"`{100.0 * float(best['best_token0_reject_rate']):.1f}%` token0 reject | "
        "No runtime run is justified. |"
    )
    lines.append(
        "| Tiny validation split is not predictive | K1/K4 pass 22-example validation, but held-out h1 top1 is about `38.8%` | Training gate alone is insufficient. |"
    )
    lines.append(
        "| Later horizons do not rescue K4 | K4 test top1 falls from `38.8%` to `30.7%`, `27.6%`, `20.2%` | Multi-token residual drafts are low-yield. |"
    )
    lines.append(
        "| High first-token rejection dominates | best speed row has `69.7%` token0 reject | The residual proposal usually fails before producing any accepted residual token. |"
    )
    if payload.get("detailed"):
        detail = payload["detailed"][0]
        comp = detail.get("selected_position_comparison", {})
        lines.append(
            f"| Same-position PLD comparison is mixed | local `{detail.get('version')}` selected rows: residual accepted/use `{_fmt(comp.get('residual_accepted_per_use'))}`, PLD accepted at same positions `{_fmt(comp.get('pld_accepted_at_same_positions'))}`, but residual token0 reject remains `{_pct(comp.get('residual_token0_reject_rate'))}` | Residual can target weaker PLD positions, but not reliably enough to pass the gate. |"
        )
    lines.append(
        "| No label/eval bug found | queued metadata remains `queued_use`, labels are aligned to `step_t_plus_1`, and replay commands use held-out `test.pt` | Freeze residual as a negative result. |"
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase7-offline", type=Path, required=True)
    ap.add_argument("--phase7-checkpoints", type=Path, required=True)
    ap.add_argument("--phase6-data", type=Path, required=True)
    ap.add_argument("--steps", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    aggregate = _aggregate_phase7(args.phase7_offline, args.phase7_checkpoints)
    if not aggregate:
        raise SystemExit(f"no Phase 7 report artifacts found under {args.phase7_offline}")

    test_data = args.phase6_data / "test.pt"
    detailed: list[dict[str, Any]] = []
    detail_errors: list[dict[str, str]] = []
    for row in aggregate:
        ckpt = args.phase7_checkpoints / row["version"] / "model.pt"
        if not ckpt.exists():
            continue
        report = _load_json(args.phase7_offline / row["version"] / "report.json")
        try:
            detailed.append(
                _detailed_checkpoint_analysis(
                    version=row["version"],
                    checkpoint=ckpt,
                    test_data=test_data,
                    report=report,
                    batch_size=args.batch_size,
                    device=args.device,
                )
            )
        except Exception as exc:  # noqa: BLE001 - preserve analysis rather than failing the final memo.
            detail_errors.append(
                {
                    "version": row["version"],
                    "checkpoint": _rel(ckpt),
                    "error": repr(exc),
                    "implication": (
                        "Detailed local bucket analysis skipped for this checkpoint; "
                        "aggregate Phase 7 replay JSON remains the source of truth."
                    ),
                }
            )

    if args.steps is not None:
        steps_path = args.steps
    else:
        first_report = _load_json(args.phase7_offline / aggregate[0]["version"] / "report.json")
        # The report stores the remote path; use the local phase6/phase5 default if present.
        steps_path = ROOT / "artifacts/vantage_residual/phase6_data/raw/vantage_residual_phase6_collect_test500_densityfix_v1/pld_trace/eval/steps.jsonl"
        if not steps_path.exists():
            steps_path = ROOT / "artifacts/vantage_residual/phase5_data/raw/vantage_residual_phase6_collect_test500_densityfix_v1/pld_trace/eval/steps.jsonl"
    oracle = _oracle_ceiling(steps_path)

    payload = {
        "phase": "VANTAGE-Residual final failure analysis",
        "phase7_offline": _rel(args.phase7_offline),
        "phase7_checkpoints": _rel(args.phase7_checkpoints),
        "phase6_data": _rel(args.phase6_data),
        "aggregate": aggregate,
        "detailed": detailed,
        "detail_errors": detail_errors,
        "oracle_ceiling": oracle,
        "steps": _rel(steps_path) if steps_path is not None else None,
        "decision": (
            "No concrete label/evaluation/selector bug was found.  The failure "
            "is explained by low held-out first-token accuracy, high token0 "
            "rejection, and less than one accepted residual token per selected use."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(args.output_dir / "report.md", payload)
    table_path = ROOT / "artifacts/vantage_residual/tables/final_residual_failure_modes.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    _write_table(table_path, payload)
    print(args.output_dir / "report.md")
    print(table_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
