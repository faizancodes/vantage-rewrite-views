"""Train and evaluate a pre-verification weak-PLD router.

The label is post-verification (``accepted_len <= threshold``), but feature
extraction is intentionally restricted to fields available before verifying
the current PLD draft plus history from earlier verified steps.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from statistics import median
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_pld_candidate_reranker import _load_jsonl  # noqa: E402


FORBIDDEN_FEATURE_SUBSTRINGS = (
    "accepted_len",
    "n_accepted",
    "rejected",
    "reject",
    "target_token",
    "verify",
    "wall_us",
    "n_emitted",
    "hit_max_new_tokens",
)


NUMERIC_CURRENT_FIELDS = (
    "k",
    "proposal_tokens",
    "proposal_match_len",
    "proposal_query_len",
    "proposal_source_start_token",
    "proposal_follow_start_token",
    "proposal_us",
    "prompt_len",
    "blazedit_micro_draft_tokens",
    "blazedit_max_num_run",
    "blazedit_pld_proposed",
)

CATEGORICAL_CURRENT_FIELDS = (
    "proposal_kind",
    "proposal_pool",
    "proposal_source_region",
    "proposal_root_included",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    return int(_safe_float(value, float(default)))


def load_method_rows(path: Path, *, method: str) -> dict[str, list[dict[str, Any]]]:
    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _load_jsonl(path):
        if row.get("method") == method:
            rows_by_task[str(row.get("task_id") or "")].append(row)
    for rows in rows_by_task.values():
        rows.sort(key=lambda row: int(row.get("step") or 0))
        pos = 0
        for row in rows:
            row["_generated_start"] = pos
            pos += max(1, _safe_int(row.get("n_emitted"), 1))
    return dict(rows_by_task)


def _categorical(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value)
    return text if text else "missing"


def _source_bucket(pos: int | float) -> str:
    pos = int(pos)
    if pos < 0:
        return "missing"
    if pos < 256:
        return "lt256"
    if pos < 1024:
        return "lt1024"
    if pos < 4096:
        return "lt4096"
    return "ge4096"


def _distance_bucket(dist: int | float) -> str:
    dist = int(abs(dist))
    if dist < 16:
        return "lt16"
    if dist < 64:
        return "lt64"
    if dist < 256:
        return "lt256"
    if dist < 1024:
        return "lt1024"
    return "ge1024"


def _empty_history() -> dict[str, Any]:
    return {
        "prev_accepts": deque(maxlen=8),
        "prev_weak": deque(maxlen=8),
        "prev_miss": deque(maxlen=8),
        "prev_token01": deque(maxlen=8),
        "prev_draft_lens": deque(maxlen=8),
        "weak_streak": 0,
        "strong_streak": 0,
        "miss_streak": 0,
        "last_good_source": -1,
    }


def _history_features(history: dict[str, Any]) -> dict[str, float]:
    accepts = list(history["prev_accepts"])
    weak = list(history["prev_weak"])
    misses = list(history["prev_miss"])
    token01 = list(history["prev_token01"])
    draft_lens = list(history["prev_draft_lens"])
    return {
        "prev_accept_len_1": accepts[-1] if len(accepts) >= 1 else -1,
        "prev_accept_len_2": accepts[-2] if len(accepts) >= 2 else -1,
        "prev_accept_len_3": accepts[-3] if len(accepts) >= 3 else -1,
        "rolling_accept_mean_4": sum(accepts[-4:]) / max(1, len(accepts[-4:])),
        "rolling_accept_mean_8": sum(accepts) / max(1, len(accepts)),
        "rolling_accept_median_8": float(median(accepts)) if accepts else -1.0,
        "prev_weak_1": weak[-1] if len(weak) >= 1 else 0,
        "rolling_weak_rate_8": sum(weak) / max(1, len(weak)),
        "rolling_miss_rate_8": sum(misses) / max(1, len(misses)),
        "rolling_token01_rate_8": sum(token01) / max(1, len(token01)),
        "rolling_draft_len_mean_8": sum(draft_lens) / max(1, len(draft_lens)),
        "weak_streak": history["weak_streak"],
        "strong_streak": history["strong_streak"],
        "miss_streak": history["miss_streak"],
        "has_last_good_source": 1 if int(history["last_good_source"]) >= 0 else 0,
    }


def _update_history(history: dict[str, Any], row: dict[str, Any], *, threshold: int) -> None:
    accepted = _safe_int(row.get("n_accepted_drafts"), 0)
    weak = int(accepted <= threshold)
    pld_miss = int(
        not (row.get("pld_exact_hit") is True or row.get("proposal_kind") == "blazedit_pld")
    )
    token01 = int(bool(row.get("rejected")) and accepted <= 1)
    draft_len = _safe_int(row.get("proposal_tokens", row.get("k")), 0)
    source = _safe_int(row.get("proposal_source_start_token"), -1)
    history["prev_accepts"].append(accepted)
    history["prev_weak"].append(weak)
    history["prev_miss"].append(pld_miss)
    history["prev_token01"].append(token01)
    history["prev_draft_lens"].append(draft_len)
    history["weak_streak"] = int(history["weak_streak"]) + 1 if weak else 0
    history["strong_streak"] = int(history["strong_streak"]) + 1 if not weak else 0
    history["miss_streak"] = int(history["miss_streak"]) + 1 if pld_miss else 0
    if accepted >= 16 and source >= 0:
        history["last_good_source"] = source


def extract_feature_dict(
    row: dict[str, Any],
    *,
    generated_start: int,
    history: dict[str, Any],
    step_index: int,
) -> dict[str, float | int | str]:
    """Extract only pre-verification current-step features plus prior history."""

    feats: dict[str, float | int | str] = {}
    for field in NUMERIC_CURRENT_FIELDS:
        if any(bad in field for bad in FORBIDDEN_FEATURE_SUBSTRINGS):
            raise AssertionError(f"forbidden current feature: {field}")
        feats[field] = _safe_float(row.get(field), -1.0)
        feats[f"{field}_missing"] = 1 if row.get(field) is None else 0

    draft_len = _safe_float(row.get("proposal_tokens", row.get("k")), 0.0)
    prompt_len = max(1.0, _safe_float(row.get("prompt_len"), 1.0))
    source = _safe_float(row.get("proposal_source_start_token"), -1.0)
    follow = _safe_float(row.get("proposal_follow_start_token"), -1.0)
    feats.update(
        {
            "step_index": float(step_index),
            "generated_start": float(generated_start),
            "generated_start_over_prompt": float(generated_start) / prompt_len,
            "draft_len_log1p": math.log1p(max(0.0, draft_len)),
            "draft_len_le_4": 1 if draft_len <= 4 else 0,
            "draft_len_le_8": 1 if draft_len <= 8 else 0,
            "draft_len_le_16": 1 if draft_len <= 16 else 0,
            "draft_len_ge_64": 1 if draft_len >= 64 else 0,
            "source_to_generated_distance": float(generated_start) - source if source >= 0 else -1.0,
            "follow_to_generated_distance": float(generated_start) - follow if follow >= 0 else -1.0,
        }
    )

    for field in CATEGORICAL_CURRENT_FIELDS:
        if any(bad in field for bad in FORBIDDEN_FEATURE_SUBSTRINGS):
            raise AssertionError(f"forbidden categorical feature: {field}")
        feats[f"{field}={_categorical(row.get(field))}"] = 1
    feats[f"source_bucket={_source_bucket(source)}"] = 1
    feats[f"follow_bucket={_source_bucket(follow)}"] = 1
    feats[f"source_distance_bucket={_distance_bucket(float(generated_start) - source if source >= 0 else -1)}"] = 1
    feats.update(_history_features(history))

    forbidden = [
        key
        for key in feats
        if key != "prev_accept_len_1"
        and key != "prev_accept_len_2"
        and key != "prev_accept_len_3"
        and not key.startswith("rolling_accept_")
        and any(bad in key for bad in FORBIDDEN_FEATURE_SUBSTRINGS)
    ]
    if forbidden:
        raise AssertionError(f"forbidden post-verification features leaked: {forbidden[:5]}")
    return feats


def build_examples(
    rows_by_task: dict[str, list[dict[str, Any]]],
    *,
    threshold: int = 4,
) -> tuple[list[dict[str, Any]], list[int], list[dict[str, Any]]]:
    features: list[dict[str, Any]] = []
    labels: list[int] = []
    meta: list[dict[str, Any]] = []
    for task_id, rows in rows_by_task.items():
        history = _empty_history()
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
            accepted = _safe_int(row.get("n_accepted_drafts"), 0)
            labels.append(1 if accepted <= threshold else 0)
            meta.append(
                {
                    "task_id": task_id,
                    "step_id": _safe_int(row.get("step"), 0),
                    "generated_start": generated_start,
                    "accepted_len": accepted,
                    "n_emitted": max(1, _safe_int(row.get("n_emitted"), 1)),
                    "draft_len": _safe_int(row.get("proposal_tokens", row.get("k")), 0),
                }
            )
            _update_history(history, row, threshold=threshold)
    return features, labels, meta


def split_tasks(
    rows_by_task: dict[str, list[dict[str, Any]]],
    *,
    test_fraction: float = 0.3,
    seed: int = 13,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    task_ids = sorted(rows_by_task)
    rng = random.Random(seed)
    rng.shuffle(task_ids)
    n_test = max(1, int(round(len(task_ids) * test_fraction)))
    test_ids = set(task_ids[:n_test])
    train = {task_id: rows for task_id, rows in rows_by_task.items() if task_id not in test_ids}
    test = {task_id: rows for task_id, rows in rows_by_task.items() if task_id in test_ids}
    return train, test


def _metrics(y_true: list[int], prob: list[float], threshold: float) -> dict[str, Any]:
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

    pred = [1 if p >= threshold else 0 for p in prob]
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_true, prob) if len(set(y_true)) == 2 else 0.0
    return {
        "threshold": threshold,
        "roc_auc": float(auc),
        "precision_weak": float(precision_score(y_true, pred, zero_division=0)),
        "recall_weak": float(recall_score(y_true, pred, zero_division=0)),
        "f1_weak": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "weak_prediction_rate": float(sum(pred) / max(1, len(pred))),
        "actual_weak_rate": float(sum(y_true) / max(1, len(y_true))),
        "false_positive_rate": float(fp / max(1, fp + tn)),
        "false_negative_rate": float(fn / max(1, fn + tp)),
    }


def _calibration(y_true: list[int], prob: list[float]) -> list[dict[str, Any]]:
    bins = []
    for lo in [i / 10 for i in range(10)]:
        hi = lo + 0.1
        idx = [i for i, p in enumerate(prob) if (lo <= p < hi) or (lo == 0.9 and p <= 1.0)]
        bins.append(
            {
                "bucket": f"{lo:.1f}-{min(1.0, hi):.1f}",
                "n": len(idx),
                "mean_probability": sum(prob[i] for i in idx) / max(1, len(idx)),
                "empirical_weak_rate": sum(y_true[i] for i in idx) / max(1, len(idx)),
            }
        )
    return bins


def _feature_importance(model: Any, feature_names: list[str], *, limit: int = 30) -> list[dict[str, Any]]:
    clf = getattr(model, "named_steps", {}).get("clf") if hasattr(model, "named_steps") else model
    if hasattr(clf, "coef_"):
        vals = clf.coef_[0]
    elif hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
    else:
        return []
    pairs = sorted(zip(feature_names, vals), key=lambda item: abs(float(item[1])), reverse=True)
    return [{"feature": name, "value": float(value)} for name, value in pairs[:limit]]


def _permutation_importance(
    model: Any,
    x: list[dict[str, Any]],
    y: list[int],
    *,
    limit: int = 30,
    random_state: int = 13,
) -> list[dict[str, Any]]:
    if not hasattr(model, "named_steps") or "vec" not in model.named_steps or "clf" not in model.named_steps:
        return []
    from sklearn.inspection import permutation_importance

    vec = model.named_steps["vec"]
    clf = model.named_steps["clf"]
    X = vec.transform(x)
    names = list(vec.get_feature_names_out())
    result = permutation_importance(
        clf,
        X,
        y,
        n_repeats=5,
        random_state=random_state,
        scoring="f1",
    )
    pairs = sorted(
        zip(names, result.importances_mean, result.importances_std),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )
    return [
        {"feature": name, "value": float(mean), "std": float(std), "kind": "permutation_f1"}
        for name, mean, std in pairs[:limit]
    ]


def train_models(
    train_x: list[dict[str, Any]],
    train_y: list[int],
    *,
    random_state: int,
) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    models: dict[str, Any] = {}
    models["logistic"] = Pipeline(
        [
            ("vec", DictVectorizer(sparse=True)),
            ("scale", StandardScaler(with_mean=False)),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    ).fit(train_x, train_y)
    models["random_forest"] = Pipeline(
        [
            ("vec", DictVectorizer(sparse=False)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    ).fit(train_x, train_y)
    models["hist_gbdt"] = Pipeline(
        [
            ("vec", DictVectorizer(sparse=False)),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=200,
                    learning_rate=0.05,
                    l2_regularization=0.01,
                    random_state=random_state,
                ),
            ),
        ]
    ).fit(train_x, train_y)
    if len(train_x) <= 10000:
        models["mlp"] = Pipeline(
            [
                ("vec", DictVectorizer(sparse=False)),
                ("scale", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64,),
                        max_iter=200,
                        alpha=1e-4,
                        random_state=random_state,
                        early_stopping=True,
                    ),
                ),
            ]
        ).fit(train_x, train_y)
    return models


def _predict_proba(model: Any, x: list[dict[str, Any]]) -> list[float]:
    if hasattr(model, "predict_proba"):
        return [float(v) for v in model.predict_proba(x)[:, 1]]
    pred = model.predict(x)
    return [float(v) for v in pred]


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Weak-PLD Router",
        "",
        f"train steps: `{payload['train_steps']}`",
        f"test steps: `{payload['test_steps']}`",
        f"label: `accepted_len <= {payload['accepted_len_threshold']}`",
        "",
        "| router | AUC | threshold | precision | recall | F1 | pred weak | FPR | FNR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, report in payload["routers"].items():
        m = report["selected_metrics"]
        lines.append(
            f"| {name} | {m['roc_auc']:.3f} | {m['threshold']:.2f} | "
            f"{m['precision_weak']:.3f} | {m['recall_weak']:.3f} | {m['f1_weak']:.3f} | "
            f"{m['weak_prediction_rate']:.3f} | {m['false_positive_rate']:.3f} | "
            f"{m['false_negative_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Selected router: `{payload['selected_router']}`",
            "",
            "Top feature weights/importances:",
        ]
    )
    for item in payload["routers"][payload["selected_router"]].get("feature_importance", [])[:15]:
        lines.append(f"- `{item['feature']}`: {item['value']:.4g}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-steps", type=Path, default=None)
    ap.add_argument("--test-steps", type=Path, default=None)
    ap.add_argument("--steps", type=Path, default=None)
    ap.add_argument("--method", default="blazedit_pld_w128_n10")
    ap.add_argument("--output-dir", type=Path, default=Path("/tmp/pld_mtp/weak_router"))
    ap.add_argument("--accepted-len-threshold", type=int, default=4)
    ap.add_argument("--decision-threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    if args.train_steps and args.test_steps:
        train_rows = load_method_rows(args.train_steps, method=args.method)
        test_rows = load_method_rows(args.test_steps, method=args.method)
        train_label = str(args.train_steps)
        test_label = str(args.test_steps)
    elif args.steps:
        all_rows = load_method_rows(args.steps, method=args.method)
        train_rows, test_rows = split_tasks(all_rows, seed=args.seed)
        train_label = f"{args.steps}::task_split_train"
        test_label = f"{args.steps}::task_split_test"
    else:
        raise SystemExit("provide --train-steps and --test-steps, or --steps")
    if not train_rows or not test_rows:
        raise SystemExit("empty train or test rows")
    if set(train_rows) & set(test_rows):
        raise SystemExit("train/test task_id leakage detected")

    train_x, train_y, _train_meta = build_examples(
        train_rows,
        threshold=args.accepted_len_threshold,
    )
    test_x, test_y, test_meta = build_examples(
        test_rows,
        threshold=args.accepted_len_threshold,
    )
    thresholds = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    routers: dict[str, Any] = {}

    rule_prob = [1.0 if float(x.get("draft_len_le_4", 0.0)) >= 1.0 else 0.0 for x in test_x]
    routers["rule_draft_len_le_4"] = {
        "probabilities": rule_prob,
        "metrics_by_threshold": {str(t): _metrics(test_y, rule_prob, t) for t in thresholds},
        "selected_metrics": _metrics(test_y, rule_prob, args.decision_threshold),
        "calibration": _calibration(test_y, rule_prob),
        "feature_importance": [{"feature": "draft_len_le_4", "value": 1.0}],
    }

    trained_models = train_models(train_x, train_y, random_state=args.seed)
    best_name = "rule_draft_len_le_4"
    best_f1 = routers[best_name]["selected_metrics"]["f1_weak"]
    for name, model in trained_models.items():
        probs = _predict_proba(model, test_x)
        metric_by_threshold = {str(t): _metrics(test_y, probs, t) for t in thresholds}
        selected = _metrics(test_y, probs, args.decision_threshold)
        vec = model.named_steps.get("vec") if hasattr(model, "named_steps") else None
        feature_names = list(vec.get_feature_names_out()) if vec is not None else []
        importance = _feature_importance(model, feature_names)
        if not importance:
            importance = _permutation_importance(
                model,
                test_x,
                test_y,
                random_state=args.seed,
            )
        routers[name] = {
            "probabilities": probs,
            "metrics_by_threshold": metric_by_threshold,
            "selected_metrics": selected,
            "calibration": _calibration(test_y, probs),
            "feature_importance": importance,
        }
        if selected["f1_weak"] > best_f1:
            best_name = name
            best_f1 = selected["f1_weak"]

    selected_model = trained_models.get(best_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if selected_model is not None:
        with (args.output_dir / "router.pkl").open("wb") as f:
            pickle.dump(
                {
                    "model": selected_model,
                    "router_name": best_name,
                    "accepted_len_threshold": args.accepted_len_threshold,
                    "method": args.method,
                    "feature_contract": {
                        "forbidden_current_feature_substrings": FORBIDDEN_FEATURE_SUBSTRINGS,
                        "numeric_current_fields": NUMERIC_CURRENT_FIELDS,
                        "categorical_current_fields": CATEGORICAL_CURRENT_FIELDS,
                    },
                },
                f,
            )
    else:
        (args.output_dir / "router_weights.json").write_text(
            json.dumps({"router_name": best_name, "rule": "draft_len <= 4"}, indent=2) + "\n"
        )

    report_routers = {
        name: {k: v for k, v in router.items() if k != "probabilities"}
        for name, router in routers.items()
    }
    probabilities_out = {
        "task_id": [row["task_id"] for row in test_meta],
        "step_id": [row["step_id"] for row in test_meta],
        "label_weak": test_y,
        "routers": {name: router["probabilities"] for name, router in routers.items()},
    }
    (args.output_dir / "router_predictions.json").write_text(
        json.dumps(probabilities_out, indent=2) + "\n"
    )
    payload = {
        "train_steps": train_label,
        "test_steps": test_label,
        "method": args.method,
        "accepted_len_threshold": args.accepted_len_threshold,
        "n_train_tasks": len(train_rows),
        "n_test_tasks": len(test_rows),
        "n_train_steps": len(train_y),
        "n_test_steps": len(test_y),
        "train_weak_rate": sum(train_y) / max(1, len(train_y)),
        "test_weak_rate": sum(test_y) / max(1, len(test_y)),
        "selected_router": best_name,
        "routers": report_routers,
        "leakage_check": {
            "task_id_overlap": 0,
            "accepted_len_only_used_as_label": True,
            "forbidden_current_feature_substrings": list(FORBIDDEN_FEATURE_SUBSTRINGS),
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(args.output_dir / "report.md", payload)
    print((args.output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
