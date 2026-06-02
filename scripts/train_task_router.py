"""Train a conservative prompt-time router for PLD vs VANTAGE.

The router uses only manifest/prompt features available before decoding.  It
labels a task positive when the chosen VANTAGE candidate is at least
``--win-margin`` faster than exact PLD on the training artifacts, then fits a
small logistic model and selects a threshold that prioritizes no-regression.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asts.task_router import FEATURE_NAMES, extract_features


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _method_output(row: dict[str, Any], method: str) -> tuple[int, float] | None:
    out = (row.get("outputs") or {}).get(method)
    if not isinstance(out, dict):
        return None
    tokens = int(out.get("n_new_tokens") or 0)
    wall_us = float(out.get("wall_us") or 0.0)
    if tokens <= 0 or wall_us <= 0:
        return None
    return tokens, wall_us


def _tps(item: tuple[int, float]) -> float:
    return item[0] / (item[1] / 1e6)


def _standardize(xs: list[list[float]]) -> tuple[list[float], list[float], list[list[float]]]:
    n = len(xs)
    d = len(xs[0]) if xs else 0
    means = [0.0] * d
    scales = [1.0] * d
    for j in range(d):
        if FEATURE_NAMES[j] == "bias":
            continue
        means[j] = sum(row[j] for row in xs) / max(1, n)
        var = sum((row[j] - means[j]) ** 2 for row in xs) / max(1, n)
        scales[j] = math.sqrt(var) or 1.0
    out: list[list[float]] = []
    for row in xs:
        out.append([
            1.0 if FEATURE_NAMES[j] == "bias" else (row[j] - means[j]) / scales[j]
            for j in range(d)
        ])
    return means, scales, out


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _fit_logistic(xs: list[list[float]], ys: list[int], *, epochs: int = 4000, lr: float = 0.05, l2: float = 0.02) -> list[float]:
    if not xs:
        return [0.0] * len(FEATURE_NAMES)
    d = len(xs[0])
    weights = [0.0] * d
    n = len(xs)
    for _ in range(epochs):
        grads = [0.0] * d
        for x, y in zip(xs, ys):
            p = _sigmoid(sum(w * v for w, v in zip(weights, x)))
            err = p - y
            for j, value in enumerate(x):
                grads[j] += err * value
        for j in range(d):
            reg = l2 * weights[j] if FEATURE_NAMES[j] != "bias" else 0.0
            weights[j] -= lr * ((grads[j] / n) + reg)
    return weights


def _predict(weights: list[float], x: list[float]) -> float:
    return _sigmoid(sum(w * v for w, v in zip(weights, x)))


def _aggregate(rows: list[dict[str, Any]], probs: list[float], threshold: float) -> dict[str, float]:
    tokens = 0
    wall = 0.0
    pld_tokens = 0
    pld_wall = 0.0
    chosen = 0
    winners_chosen = 0
    for row, prob in zip(rows, probs):
        pld_tokens += int(row["pld_tokens"])
        pld_wall += float(row["pld_wall_us"])
        if prob >= threshold:
            chosen += 1
            winners_chosen += int(row["label"])
            tokens += int(row["candidate_tokens"])
            wall += float(row["candidate_wall_us"])
        else:
            tokens += int(row["pld_tokens"])
            wall += float(row["pld_wall_us"])
    tps = tokens / (wall / 1e6) if wall > 0 else 0.0
    pld_tps = pld_tokens / (pld_wall / 1e6) if pld_wall > 0 else 0.0
    return {
        "threshold": threshold,
        "ratio_vs_pld": tps / pld_tps if pld_tps else 0.0,
        "tokens_per_sec": tps,
        "pld_tokens_per_sec": pld_tps,
        "trans_chosen": float(chosen),
        "true_winners_chosen": float(winners_chosen),
    }


def _choose_threshold(rows: list[dict[str, Any]], probs: list[float], min_ratio: float) -> dict[str, float]:
    candidates = sorted({0.0, 0.5, 1.0, *probs})
    scored = [_aggregate(rows, probs, t) for t in candidates]
    feasible = [item for item in scored if item["ratio_vs_pld"] >= min_ratio]
    if feasible:
        # Optimize the same weighted objective used in the paper tables:
        # aggregate output tokens divided by aggregate wall time.  Task win
        # count is only a tie-breaker because long losses dominate serving
        # latency even when median per-task ratio is near parity.
        return max(
            feasible,
            key=lambda item: (
                item["ratio_vs_pld"],
                item["tokens_per_sec"],
                item["true_winners_chosen"],
                -item["trans_chosen"],
            ),
        )
    return max(scored, key=lambda item: item["ratio_vs_pld"])


def _rows_from_artifacts(
    completions: list[dict[str, Any]],
    manifest_by_id: dict[str, dict[str, Any]],
    *,
    pld_method: str,
    candidate_methods: list[str],
    win_margin: float,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in completions:
        task_id = str(row.get("task_id") or "")
        pld = _method_output(row, pld_method)
        if pld is None:
            continue
        best_method = ""
        best = None
        for method in candidate_methods:
            item = _method_output(row, method)
            if item is None:
                continue
            if best is None or _tps(item) > _tps(best):
                best = item
                best_method = method
        if best is None:
            continue
        manifest = manifest_by_id.get(task_id, row)
        feats = extract_features(
            prompt=str(manifest.get("prompt") or row.get("prompt") or ""),
            reference=str(manifest.get("reference") or row.get("reference") or ""),
            metadata=manifest.get("metadata") or row.get("metadata") or manifest,
            output_budget=max_new_tokens,
        )
        rows.append(
            {
                "task_id": task_id,
                "features": feats,
                "label": int(_tps(best) >= (1.0 + win_margin) * _tps(pld)),
                "best_method": best_method,
                "pld_tokens": pld[0],
                "pld_wall_us": pld[1],
                "candidate_tokens": best[0],
                "candidate_wall_us": best[1],
                "candidate_ratio": _tps(best) / _tps(pld),
            }
        )
    return rows


def _markdown(report: dict[str, Any]) -> str:
    chosen = report["chosen_threshold"]
    lines = [
        "# Task-Level Router Training",
        "",
        f"Rows: {report['n_rows']}. Positives: {report['n_positive']} ({report['positive_rate']:.1%}).",
        f"PLD: `{report['pld_method']}`. Candidates: `{', '.join(report['candidate_methods'])}`.",
        "",
        "## Selected Threshold",
        "",
        "| Threshold | tok/s | Ratio vs PLD | Trans chosen | True winners chosen |",
        "|----------:|------:|-------------:|-------------:|--------------------:|",
        (
            f"| {chosen['threshold']:.4f} | {chosen['tokens_per_sec']:.2f} | "
            f"{chosen['ratio_vs_pld']:.3f} | {int(chosen['trans_chosen'])} | "
            f"{int(chosen['true_winners_chosen'])} |"
        ),
        "",
        "## Top Weights",
        "",
        "| Feature | Weight |",
        "|---------|-------:|",
    ]
    for name, weight in report["top_weights"]:
        lines.append(f"| `{name}` | {weight:.4f} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completions", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pld-method", default="blazedit_pld_w128_n10")
    parser.add_argument("--candidate-methods", required=True)
    parser.add_argument("--win-margin", type=float, default=0.05)
    parser.add_argument("--min-train-ratio", type=float, default=0.99)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    completions = _load_jsonl(Path(args.completions))
    manifest = _load_jsonl(Path(args.manifest))
    manifest_by_id = {str(row.get("task_id") or ""): row for row in manifest}
    candidate_methods = [m.strip() for m in args.candidate_methods.split(",") if m.strip()]
    rows = _rows_from_artifacts(
        completions,
        manifest_by_id,
        pld_method=args.pld_method,
        candidate_methods=candidate_methods,
        win_margin=args.win_margin,
        max_new_tokens=args.max_new_tokens,
    )
    xs_raw = [[float(row["features"].get(name, 0.0)) for name in FEATURE_NAMES] for row in rows]
    ys = [int(row["label"]) for row in rows]
    means, scales, xs = _standardize(xs_raw)
    weights = _fit_logistic(xs, ys)
    probs = [_predict(weights, x) for x in xs]
    chosen = _choose_threshold(rows, probs, args.min_train_ratio)
    top_weights = sorted(
        zip(FEATURE_NAMES, weights),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:12]
    router = {
        "type": "logistic_prompt_router",
        "feature_names": FEATURE_NAMES,
        "weights": weights,
        "means": dict(zip(FEATURE_NAMES, means)),
        "scales": dict(zip(FEATURE_NAMES, scales)),
        "threshold": chosen["threshold"],
        "pld_method": args.pld_method,
        "candidate_methods": candidate_methods,
        "win_margin": args.win_margin,
        "min_train_ratio": args.min_train_ratio,
        "chosen_threshold": chosen,
        "n_rows": len(rows),
        "n_positive": sum(ys),
    }
    report = {
        **router,
        "positive_rate": sum(ys) / max(1, len(ys)),
        "top_weights": top_weights,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(router, indent=2))
    Path(args.output_md).write_text(_markdown(report))
    print(_markdown(report))


if __name__ == "__main__":
    main()
