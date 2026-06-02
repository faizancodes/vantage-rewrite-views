"""Cost diagnostic for weak-router adaptive PLD draft capping.

This script does not simulate capped decoding.  It answers the gating question:
do steps predicted weak by the pre-verification router spend enough verifier
time on rejected PLD drafts to justify a runtime cap sweep?
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_pld_candidate_reranker import _load_jsonl  # noqa: E402
from scripts.train_weak_pld_router import (  # noqa: E402
    _empty_history,
    _safe_int,
    _update_history,
    extract_feature_dict,
    load_method_rows,
)


def _load_router(path: Path):
    with path.open("rb") as f:
        payload = pickle.load(f)
    model = payload.get("model") if isinstance(payload, dict) else payload
    if model is None:
        raise SystemExit(f"{path} does not contain a router model")
    return model


def _row_features(rows_by_task: dict[str, list[dict[str, Any]]], router, threshold: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task_id, rows in sorted(rows_by_task.items()):
        history = _empty_history()
        features: list[dict[str, Any]] = []
        keys: list[dict[str, Any]] = []
        for step_index, row in enumerate(rows):
            generated_start = _safe_int(row.get("_generated_start"), 0)
            features.append(
                extract_feature_dict(
                    row,
                    generated_start=generated_start,
                    history=history,
                    step_index=step_index,
                )
            )
            keys.append({"task_id": task_id, "step": _safe_int(row.get("step"), 0), "row": row})
            _update_history(history, row, threshold=threshold)
        if not features:
            continue
        probs = router.predict_proba(features)[:, 1]
        for key, prob in zip(keys, probs, strict=True):
            row = key["row"]
            draft_len = _safe_int(row.get("proposal_tokens", row.get("k")), 0)
            accepted_len = _safe_int(row.get("n_accepted_drafts"), 0)
            verify_us = float(row.get("verify_us", 0.0) or 0.0)
            lookup_us = float(row.get("proposal_us", row.get("pld_opp_lookup_us", 0.0)) or 0.0)
            out.append(
                {
                    "task_id": key["task_id"],
                    "step": key["step"],
                    "draft_len": draft_len,
                    "accepted_len": accepted_len,
                    "verify_us": verify_us,
                    "lookup_us": lookup_us,
                    "wasted_verified_tokens": max(0, draft_len - accepted_len),
                    "token0_reject": bool(row.get("rejected")) and accepted_len == 0,
                    "token01_reject": bool(row.get("rejected")) and accepted_len <= 1,
                    "weak_label": accepted_len <= threshold,
                    "router_probability": float(prob),
                }
            )
    return out


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "verify_ms_per_step": 0.0,
            "lookup_ms_per_step": 0.0,
            "draft_len_mean": 0.0,
            "accepted_len_mean": 0.0,
            "wasted_verified_tokens_mean": 0.0,
            "token0_reject_rate": 0.0,
            "token01_reject_rate": 0.0,
            "verify_ms_total": 0.0,
            "lookup_ms_total": 0.0,
        }
    return {
        "n": len(rows),
        "verify_ms_per_step": mean([r["verify_us"] for r in rows]) / 1000.0,
        "lookup_ms_per_step": mean([r["lookup_us"] for r in rows]) / 1000.0,
        "draft_len_mean": mean([r["draft_len"] for r in rows]),
        "accepted_len_mean": mean([r["accepted_len"] for r in rows]),
        "wasted_verified_tokens_mean": mean([r["wasted_verified_tokens"] for r in rows]),
        "token0_reject_rate": sum(1 for r in rows if r["token0_reject"]) / len(rows),
        "token01_reject_rate": sum(1 for r in rows if r["token01_reject"]) / len(rows),
        "verify_ms_total": sum(r["verify_us"] for r in rows) / 1000.0,
        "lookup_ms_total": sum(r["lookup_us"] for r in rows) / 1000.0,
    }


def _bucket_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        value = int(row[field])
        if value == 0:
            key = "0"
        elif value <= 1:
            key = "1"
        elif value <= 2:
            key = "2"
        elif value <= 4:
            key = "3-4"
        elif value <= 8:
            key = "5-8"
        elif value <= 16:
            key = "9-16"
        elif value <= 64:
            key = "17-64"
        else:
            key = "65+"
        counts[key] += 1
    return dict(counts)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Weak-Router PLD Cost Diagnostic",
        "",
        f"steps: `{payload['steps']}`",
        f"router: `{payload['router']}`",
        "",
        "| group | n | verify ms/step | lookup ms/step | draft len | accepted len | wasted tokens | token0 reject | verify ms total |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in payload["groups"].items():
        lines.append(
            f"| {name} | {row['n']} | {row['verify_ms_per_step']:.2f} | "
            f"{row['lookup_ms_per_step']:.3f} | {row['draft_len_mean']:.2f} | "
            f"{row['accepted_len_mean']:.2f} | {row['wasted_verified_tokens_mean']:.2f} | "
            f"{100.0 * row['token0_reject_rate']:.1f}% | {row['verify_ms_total']:.1f} |"
        )
    lines.extend(["", "## Threshold Sweep", ""])
    lines.append("| threshold | predicted weak % | precision | recall | weak verify fraction | wasted tokens/weak step |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in payload["thresholds"]:
        lines.append(
            f"| {row['threshold']:.2f} | {100.0 * row['predicted_weak_rate']:.1f}% | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | "
            f"{100.0 * row['predicted_weak_verify_fraction']:.1f}% | "
            f"{row['predicted_weak']['wasted_verified_tokens_mean']:.2f} |"
        )
    lines.extend(["", f"Decision: **{payload['decision']}**"])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=Path, required=True)
    ap.add_argument("--router", type=Path, required=True)
    ap.add_argument("--method", default="blazedit_pld_w128_n10")
    ap.add_argument("--accepted-len-threshold", type=int, default=4)
    ap.add_argument("--thresholds", default="0.3,0.5,0.7")
    ap.add_argument("--output-dir", type=Path, default=Path("/tmp/pld_mtp/weak_router_pld_cost"))
    args = ap.parse_args()

    router = _load_router(args.router)
    rows_by_task = load_method_rows(args.steps, method=args.method)
    rows = _row_features(rows_by_task, router, args.accepted_len_threshold)
    total_verify_us = sum(r["verify_us"] for r in rows)
    weak_rows = [r for r in rows if r["weak_label"]]
    strong_rows = [r for r in rows if not r["weak_label"]]
    thresholds = []
    for threshold in [float(x) for x in args.thresholds.split(",") if x.strip()]:
        predicted = [r for r in rows if r["router_probability"] >= threshold]
        tp = sum(1 for r in predicted if r["weak_label"])
        fp = sum(1 for r in predicted if not r["weak_label"])
        fn = sum(1 for r in rows if r["weak_label"] and r["router_probability"] < threshold)
        thresholds.append(
            {
                "threshold": threshold,
                "predicted_weak_rate": len(predicted) / max(1, len(rows)),
                "precision": tp / max(1, tp + fp),
                "recall": tp / max(1, tp + fn),
                "false_positive_count": fp,
                "false_negative_count": fn,
                "predicted_weak_verify_fraction": (
                    sum(r["verify_us"] for r in predicted) / total_verify_us
                    if total_verify_us > 0
                    else 0.0
                ),
                "predicted_weak": _summarize(predicted),
            }
        )
    groups = {
        "all": _summarize(rows),
        "actual weak accepted<=4": _summarize(weak_rows),
        "actual strong accepted>4": _summarize(strong_rows),
    }
    for row in thresholds:
        groups[f"router weak >= {row['threshold']:.2f}"] = row["predicted_weak"]

    best_predicted = max(thresholds, key=lambda row: row["predicted_weak_verify_fraction"])
    exploitable = (
        best_predicted["predicted_weak_verify_fraction"] >= 0.35
        and best_predicted["predicted_weak"]["wasted_verified_tokens_mean"] >= 8.0
    )
    decision = (
        "run weak-router capped PLD cap sweep"
        if exploitable
        else "PLD verifier waste looks too small for capping"
    )
    payload = {
        "steps": str(args.steps),
        "router": str(args.router),
        "method": args.method,
        "accepted_len_threshold": args.accepted_len_threshold,
        "groups": groups,
        "thresholds": thresholds,
        "accepted_len_bucket_counts": _bucket_counts(rows, "accepted_len"),
        "draft_len_bucket_counts": _bucket_counts(rows, "draft_len"),
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(args.output_dir / "report.md", payload)
    print((args.output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
