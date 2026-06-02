#!/usr/bin/env python3
"""Profile or bound final-hidden capture overhead for queued residual decoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/vantage_residual/phase3_hidden_capture"
DEFAULT_ARTIFACTS = (
    ROOT / "analysis/real_commits/modal/vantage_real_commit_pld_adjacent_test50_queued_mtp_k4_smoke_v1/eval/aggregate.json",
    ROOT / "analysis/real_commits/modal/vantage_real_commit_pld_adjacent_test500_queued_mtp_k4_test500_v1/eval/aggregate.json",
    ROOT / "artifacts/vantage_residual/runs/vantage_residual_smoke50_v1/eval/aggregate.json",
)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _method_row(payload: dict[str, Any], method: str) -> dict[str, Any]:
    rows = payload.get("by_method")
    if isinstance(rows, dict) and isinstance(rows.get(method), dict):
        return rows[method]
    return {}


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _ratio(num: Any, den: Any) -> float | None:
    n = _safe_float(num)
    d = _safe_float(den)
    if n is None or d in (None, 0.0):
        return None
    return n / d


def _artifact_rows(paths: list[Path], *, baseline: str, queued: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        payload = _load_json(path)
        base = _method_row(payload, baseline)
        queue = _method_row(payload, queued)
        if not base and not queue:
            continue
        row = {
            "path": _rel(path),
            "baseline_method": baseline,
            "queued_method": queued,
            "baseline_steps": base.get("n_steps"),
            "queued_steps": queue.get("n_steps"),
            "baseline_verify_us_total": base.get("verify_us_total"),
            "queued_verify_us_total": queue.get("verify_us_total"),
            "baseline_verify_us_per_step": _ratio(base.get("verify_us_total"), base.get("n_steps")),
            "queued_verify_us_per_step": _ratio(queue.get("verify_us_total"), queue.get("n_steps")),
            "queued_speedup_vs_baseline": _ratio(
                queue.get("tokens_per_sec"), base.get("tokens_per_sec")
            ),
            "queued_token0_reject_rate": queue.get("mtp_used_token0_reject_rate")
            or queue.get("mtp_token0_reject_rate"),
            "queued_predictions_created": queue.get("mtp_queue_predictions_created"),
            "queued_predictions_used": queue.get("mtp_queue_predictions_used"),
            "note": (
                "artifact comparison is confounded by different draft selection and is not a direct hidden-capture overhead measurement"
            ),
        }
        base_per = row["baseline_verify_us_per_step"]
        queued_per = row["queued_verify_us_per_step"]
        row["verify_us_per_step_delta_pct"] = (
            100.0 * (queued_per - base_per) / base_per
            if isinstance(base_per, (int, float)) and isinstance(queued_per, (int, float)) and base_per
            else None
        )
        out.append(row)
    return out


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# VANTAGE-Residual Hidden-Capture Profile",
        "",
        f"mode: `{payload['mode']}`",
        f"status: `{payload['status']}`",
        "",
        "| source | baseline verify us/step | queued verify us/step | delta | queued speed | note |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["artifact_rows"]:
        delta = row.get("verify_us_per_step_delta_pct")
        lines.append(
            "| {path} | {base} | {queued} | {delta} | {speed} | {note} |".format(
                path=row["path"],
                base=_fmt(row.get("baseline_verify_us_per_step")),
                queued=_fmt(row.get("queued_verify_us_per_step")),
                delta="n/a" if delta is None else f"{delta:.1f}%",
                speed=_fmt(row.get("queued_speedup_vs_baseline"), suffix="x"),
                note=row["note"],
            )
        )
    lines.extend(["", f"Decision: **{payload['decision']}**"])
    path.write_text("\n".join(lines) + "\n")


def _fmt(value: Any, *, suffix: str = "") -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:.3f}{suffix}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["local_or_trace", "artifact"], default="local_or_trace")
    ap.add_argument("--aggregate", action="append", type=Path, default=[])
    ap.add_argument("--baseline-method", default="blazedit_pld_w128_n10")
    ap.add_argument("--queued-method", default="pld_queued_mtp_heads")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    paths = args.aggregate or list(DEFAULT_ARTIFACTS)
    rows = _artifact_rows(paths, baseline=args.baseline_method, queued=args.queued_method)
    status = "artifact_only_not_directly_measured"
    decision = (
        "stop: direct hidden-capture overhead has not been measured; existing queued artifacts are confounded and also fail parity"
    )
    payload = {
        "mode": args.mode,
        "status": status,
        "baseline_method": args.baseline_method,
        "queued_method": args.queued_method,
        "artifact_rows": rows,
        "direct_hidden_capture_overhead_pct": None,
        "memory_delta_gb": None,
        "oom": None,
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_md(args.output_dir / "summary.md", payload)
    print((args.output_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
