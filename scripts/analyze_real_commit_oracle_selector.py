#!/usr/bin/env python3
"""Oracle and selector analysis for the balanced real-commit benchmark.

This is deliberately artifact-only.  It consumes an existing
``completions.jsonl`` containing PLD, frozen TransPLD, and MV candidate rows,
then:

* computes task-level oracle ceilings;
* stratifies where VANTAGE wins/loses;
* writes inspection packets for the largest wins/losses;
* fits a conservative train-only selector that can route among PLD, stable MV,
  and frozen TransPLD using pre-decode manifest features.

The selector is intentionally a simple rule grid, not a black-box model.  It is
optimized for aggregate weighted tok/s with regressions penalized more heavily
than missed wins.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PLD = "blazedit_pld_w128_n10"
FORCE_PLD = "vantage_force_pld_w128_n10"
FROZEN = "vantage_frozen_transpld"
MV = "vantage_mv_pld_s96_x1_m16_t8_w128_n10"
TREE = "vantage_mv_pld_tree_s96_x1_m16_t8_w128_n10"
G32 = "vantage_mv_pld_s96_x1_m16_t8_g32_w128_n10"
HUNK = "vantage_mv_pld_hunk_s96_x1_m16_t8_w128_n10"

DEFAULT_CANDIDATES = [MV, TREE, G32, HUNK, FROZEN]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def source_key(row: dict[str, Any]) -> str:
    meta = row.get("metadata") or {}
    return str(
        meta.get("source_task_id")
        or row.get("source_task_id")
        or meta.get("commit_sha")
        or row.get("commit_sha")
        or row.get("task_id")
        or ""
    )


def output(row: dict[str, Any], method: str) -> dict[str, Any] | None:
    out = (row.get("outputs") or {}).get(method)
    return out if isinstance(out, dict) else None


def wall(row: dict[str, Any], method: str) -> float:
    out = output(row, method)
    return float(out.get("wall_us") or 0.0) if out else math.inf


def tokens(row: dict[str, Any], method: str) -> int:
    out = output(row, method)
    return int(out.get("n_new_tokens") or 0) if out else 0


def tps(row: dict[str, Any], method: str) -> float:
    w = wall(row, method)
    return tokens(row, method) / (w / 1e6) if w and math.isfinite(w) else 0.0


def aggregate_tps(rows: list[dict[str, Any]], chooser) -> float:
    total_tokens = 0
    total_wall = 0.0
    for row in rows:
        method = chooser(row)
        total_tokens += tokens(row, method)
        total_wall += wall(row, method)
    return total_tokens / (total_wall / 1e6) if total_wall > 0 else 0.0


def best_method(row: dict[str, Any], methods: list[str]) -> str:
    available = [m for m in methods if output(row, m)]
    return min(available, key=lambda m: wall(row, m)) if available else PLD


def ratio(row: dict[str, Any], method: str, baseline: str = PLD) -> float:
    base = tps(row, baseline)
    return tps(row, method) / base if base else 0.0


def meta(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row.get("metadata") or {})
    for k, v in row.items():
        if k not in {"outputs", "prompt", "reference", "deterministic_target", "metadata"}:
            out.setdefault(k, v)
    return out


def fnum(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    val = meta(row).get(key, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def rewrite_pairs(row: dict[str, Any]) -> dict[str, str]:
    pairs = meta(row).get("rewrite_pairs") or {}
    if isinstance(pairs, dict):
        return {str(k): str(v) for k, v in pairs.items() if str(k) and str(v)}
    if isinstance(pairs, list):
        out: dict[str, str] = {}
        for item in pairs:
            if isinstance(item, dict):
                old, new = item.get("old"), item.get("new")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                old, new = item[0], item[1]
            else:
                continue
            if old and new:
                out[str(old)] = str(new)
        return out
    return {}


def term_count(text: str, term: str) -> int:
    if not term:
        return 0
    return text.count(term)


def adoption_labels(row: dict[str, Any], method: str) -> dict[str, Any]:
    text = str((output(row, method) or {}).get("text") or "")
    old_hits = 0
    new_hits = 0
    adopted_pairs = 0
    pairs = rewrite_pairs(row)
    for old, new in pairs.items():
        old_n = term_count(text, old)
        new_n = term_count(text, new)
        old_hits += old_n
        new_hits += new_n
        adopted_pairs += int(new_n > 0 and old_n == 0)
    denom = old_hits + new_hits
    return {
        "old_hits": old_hits,
        "new_hits": new_hits,
        "adoption_fraction": new_hits / denom if denom else 0.0,
        "full_pair_adoption": adopted_pairs / len(pairs) if pairs else 0.0,
    }


def regime(row: dict[str, Any]) -> str:
    fit = fnum(row, "transformed_reference_fit")
    copy = fnum(row, "copied_token_percentage")
    edit = fnum(row, "edit_distance_tokens")
    hunks = fnum(row, "changed_hunk_count")
    density = fnum(row, "rewrite_density_per_100_tokens")
    if fit >= 0.92 and density >= 0.5:
        return "rewrite_aligned_drift"
    if fit <= 0.55 or hunks >= 5 or edit >= 96:
        return "dirty_large_refactor"
    if copy >= 0.95 and density < 1.0:
        return "pld_favored_exact_copy"
    return "mixed_reference_edit"


def family(row: dict[str, Any]) -> str:
    return str(meta(row).get("drift_family") or meta(row).get("benchmark_regime") or "unknown")


def row_features(row: dict[str, Any]) -> dict[str, float | str]:
    cached = row.get("_selector_features")
    if isinstance(cached, dict):
        return cached
    pairs = rewrite_pairs(row)
    old_tok = sum(max(1, len(k.split("."))) for k in pairs)
    new_tok = sum(max(1, len(v.split("."))) for v in pairs)
    has_dotted = any("." in k or "." in v for k, v in pairs.items())
    has_literal = any(any(c.isdigit() or c in "'\"" for c in f"{k}{v}") for k, v in pairs.items())
    features: dict[str, float | str] = {
        "family": family(row),
        "regime": regime(row),
        "copy_ratio": fnum(row, "copied_token_percentage"),
        "fit": fnum(row, "transformed_reference_fit"),
        "dirty": fnum(row, "dirty_vs_transformed_reference"),
        "rewrite_density": fnum(row, "rewrite_density_per_100_tokens"),
        "rewrite_occ": fnum(row, "rewrite_occurrences_in_reference"),
        "map_count": float(len(pairs)),
        "noisy_map_count": fnum(row, "noisy_map_count"),
        "edit_distance": fnum(row, "edit_distance_tokens"),
        "hunk_count": fnum(row, "changed_hunk_count"),
        "longest_span": fnum(row, "longest_unchanged_span_tokens"),
        "old_new_token_ratio": old_tok / max(1, new_tok),
        "has_dotted": float(has_dotted),
        "has_literal": float(has_literal),
    }
    row["_selector_features"] = features
    return features


@dataclass(frozen=True)
class Rule:
    mv_fit: float
    mv_density: float
    mv_max_noisy: int
    mv_max_dirty: float
    mv_max_copy: float
    frozen_fit: float
    frozen_density: float
    frozen_max_noisy: int
    frozen_max_dirty: float
    frozen_min_occ: int

    def choose(self, row: dict[str, Any]) -> str:
        ft = row_features(row)
        if ft["rewrite_occ"] <= 0 or ft["map_count"] <= 0:
            return PLD
        if (
            ft["fit"] >= self.frozen_fit
            and ft["rewrite_density"] >= self.frozen_density
            and ft["noisy_map_count"] <= self.frozen_max_noisy
            and ft["dirty"] <= self.frozen_max_dirty
            and ft["rewrite_occ"] >= self.frozen_min_occ
        ):
            return FROZEN
        if (
            ft["fit"] >= self.mv_fit
            and ft["rewrite_density"] >= self.mv_density
            and ft["noisy_map_count"] <= self.mv_max_noisy
            and ft["dirty"] <= self.mv_max_dirty
            and ft["copy_ratio"] <= self.mv_max_copy
        ):
            return MV
        return PLD

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "real_commit_mv_frozen_rule_v1",
            **self.__dict__,
            "pld_method": PLD,
            "mv_method": MV,
            "frozen_method": FROZEN,
        }


def choose_from_rule_json(row: dict[str, Any], selector: dict[str, Any]) -> str:
    return Rule(
        mv_fit=float(selector["mv_fit"]),
        mv_density=float(selector["mv_density"]),
        mv_max_noisy=int(selector["mv_max_noisy"]),
        mv_max_dirty=float(selector["mv_max_dirty"]),
        mv_max_copy=float(selector["mv_max_copy"]),
        frozen_fit=float(selector["frozen_fit"]),
        frozen_density=float(selector["frozen_density"]),
        frozen_max_noisy=int(selector["frozen_max_noisy"]),
        frozen_max_dirty=float(selector["frozen_max_dirty"]),
        frozen_min_occ=int(selector["frozen_min_occ"]),
    ).choose(row)


def candidate_rules() -> list[Rule]:
    rules: list[Rule] = []
    for mv_fit in (0.65, 0.75, 0.85, 0.92):
        for mv_density in (0.0, 1.0, 2.0, 4.0):
            for mv_noisy in (0, 99):
                for mv_dirty in (0.30, 0.50, 1.00):
                    for mv_copy in (0.95, 0.985, 1.01):
                        for fr_fit in (0.85, 0.92, 0.97):
                            for fr_density in (1.0, 2.0, 4.0):
                                for fr_noisy in (0,):
                                    for fr_dirty in (0.20, 0.35):
                                        for fr_occ in (1, 2):
                                            rules.append(
                                                Rule(
                                                    mv_fit,
                                                    mv_density,
                                                    mv_noisy,
                                                    mv_dirty,
                                                    mv_copy,
                                                    fr_fit,
                                                    fr_density,
                                                    fr_noisy,
                                                    fr_dirty,
                                                    fr_occ,
                                                )
                                            )
    return rules


def selector_score(rows: list[dict[str, Any]], rule: Rule) -> dict[str, Any]:
    pld_t = aggregate_tps(rows, lambda _: PLD)
    chosen_methods = [rule.choose(r) for r in rows]
    total_t = sum(tokens(r, m) for r, m in zip(rows, chosen_methods))
    total_w = sum(wall(r, m) for r, m in zip(rows, chosen_methods))
    tps_val = total_t / (total_w / 1e6) if total_w else 0.0
    task_losses = [
        ratio(r, m) for r, m in zip(rows, chosen_methods)
        if m != PLD and ratio(r, m) < 1.0
    ]
    severe = sum(1 for x in task_losses if x < 0.95)
    choices = Counter(chosen_methods)
    return {
        "tokens_per_sec": tps_val,
        "ratio_vs_pld": tps_val / pld_t if pld_t else 0.0,
        "choices": dict(choices),
        "loss_count": len(task_losses),
        "severe_loss_count": severe,
        "min_chosen_ratio": min([ratio(r, m) for r, m in zip(rows, chosen_methods)] or [1.0]),
    }


def train_selector(train_rows: list[dict[str, Any]]) -> tuple[Rule, dict[str, Any]]:
    best_rule = None
    best_score = None
    for rule in candidate_rules():
        score = selector_score(train_rows, rule)
        # Weighted serving throughput is primary, but severe regressions are
        # explicitly penalized.  This makes the selector conservative enough
        # for a systems paper: missed wins are preferable to long-tail losses.
        objective = (
            score["ratio_vs_pld"]
            - 0.004 * score["severe_loss_count"]
            - 0.0005 * score["loss_count"]
            + 0.0001 * score["choices"].get(MV, 0)
            + 0.00005 * score["choices"].get(FROZEN, 0)
        )
        if best_score is None or objective > best_score["objective"]:
            best_rule = rule
            best_score = {**score, "objective": objective}
    assert best_rule is not None and best_score is not None
    return best_rule, best_score


def markdown_table(rows: list[list[Any]]) -> str:
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    out = []
    for idx, row in enumerate(rows):
        out.append("| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)) + " |")
        if idx == 0:
            out.append("| " + " | ".join("-" * widths[i] for i in range(len(row))) + " |")
    return "\n".join(out)


def summarize_oracles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selectors = {
        "PLD only": [PLD],
        "Best of PLD + MV": [PLD, MV],
        "Best of PLD + MV + frozen": [PLD, MV, FROZEN],
        "Best of all candidates": [PLD, *DEFAULT_CANDIDATES],
    }
    pld_tps = aggregate_tps(rows, lambda _: PLD)
    out = []
    for name, methods in selectors.items():
        tps_val = aggregate_tps(rows, lambda r, ms=methods: best_method(r, ms))
        out.append({"selector": name, "tokens_per_sec": tps_val, "ratio_vs_pld": tps_val / pld_tps})
    return out


def bucket_breakdown(rows: list[dict[str, Any]], bucket_key) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(bucket_key(row))].append(row)
    out = []
    for key, group in sorted(buckets.items(), key=lambda kv: kv[0]):
        pld_t = aggregate_tps(group, lambda _: PLD)
        oracle_t = aggregate_tps(group, lambda r: best_method(r, [PLD, *DEFAULT_CANDIDATES]))
        out.append(
            {
                "bucket": key,
                "n": len(group),
                "mv_vs_pld": aggregate_tps(group, lambda _: MV) / pld_t if pld_t else 0.0,
                "frozen_vs_pld": aggregate_tps(group, lambda _: FROZEN) / pld_t if pld_t else 0.0,
                "oracle_vs_pld": oracle_t / pld_t if pld_t else 0.0,
            }
        )
    return out


def inspect_rows(rows: list[dict[str, Any]], method: str, *, largest: bool, n: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda r: ratio(r, method), reverse=largest)[:n]
    out = []
    for row in ordered:
        labels = adoption_labels(row, method)
        pld_labels = adoption_labels(row, PLD)
        ft = row_features(row)
        out.append(
            {
                "task_id": row["task_id"],
                "source_task_id": source_key(row),
                "commit_url": meta(row).get("commit_url"),
                "repo": meta(row).get("repo"),
                "file_path": meta(row).get("file_path"),
                "family": family(row),
                "regime": regime(row),
                "method": method,
                "ratio_vs_pld": ratio(row, method),
                "pld_tps": tps(row, PLD),
                "method_tps": tps(row, method),
                "features": ft,
                "rewrite_pairs": rewrite_pairs(row),
                "heuristic_labels": {
                    "model_adopted_rewrite": labels["adoption_fraction"] > pld_labels["adoption_fraction"],
                    "output_stayed_close_to_old_reference": pld_labels["old_hits"] >= pld_labels["new_hits"],
                    "transformed_view_matched_target": ratio(row, method) >= 1.05 and labels["new_hits"] > 0,
                    "pld_copied_unchanged_span": ft["copy_ratio"] >= 0.95 and ratio(row, method) <= 1.0,
                    "extracted_map_was_noisy_or_wrong": ft["noisy_map_count"] > 0,
                    "commit_was_too_dirty_multi_edit": ft["dirty"] >= 0.35 or ft["hunk_count"] >= 5,
                },
                "reference_excerpt": str(row.get("reference") or "")[:900],
                "deterministic_target_excerpt": str(row.get("deterministic_target") or "")[:900],
                "pld_output_excerpt": str((output(row, PLD) or {}).get("text") or "")[:900],
                "method_output_excerpt": str((output(row, method) or {}).get("text") or "")[:900],
            }
        )
    return out


def inspection_markdown(name: str, rows: list[dict[str, Any]]) -> str:
    lines = [f"# {name}", ""]
    for idx, row in enumerate(rows, 1):
        labels = ", ".join(k for k, v in row["heuristic_labels"].items() if v) or "none"
        lines.extend(
            [
                f"## {idx}. `{row['task_id']}` ratio={row['ratio_vs_pld']:.3f}",
                "",
                f"- repo: `{row.get('repo')}`",
                f"- file: `{row.get('file_path')}`",
                f"- commit: {row.get('commit_url')}",
                f"- family/regime: `{row['family']}` / `{row['regime']}`",
                f"- rewrite pairs: `{row['rewrite_pairs']}`",
                f"- heuristic labels: {labels}",
                "",
                "```python",
                row["method_output_excerpt"],
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--completions", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--test-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(Path(args.completions))
    manifest = {source_key(r): r for r in load_jsonl(Path(args.manifest))}
    for row in rows:
        key = source_key(row)
        if key in manifest:
            merged_meta = dict(manifest[key])
            merged_meta.update(row.get("metadata") or {})
            row["metadata"] = merged_meta

    train_keys = {source_key(r) for r in load_jsonl(Path(args.train_manifest))}
    test_keys = {source_key(r) for r in load_jsonl(Path(args.test_manifest))}
    train_rows = [r for r in rows if source_key(r) in train_keys]
    test_rows = [r for r in rows if source_key(r) in test_keys]

    for group_name, group in {"all1000": rows, "train500": train_rows, "test500": test_rows}.items():
        write_jsonl(out_dir / f"per_task_{group_name}.jsonl", [
            {
                "task_id": r["task_id"],
                "source_task_id": source_key(r),
                "family": family(r),
                "regime": regime(r),
                "features": row_features(r),
                "pld_tps": tps(r, PLD),
                **{f"{m}_ratio": ratio(r, m) for m in DEFAULT_CANDIDATES if output(r, m)},
                "best_candidate": best_method(r, DEFAULT_CANDIDATES),
                "best_candidate_ratio": ratio(r, best_method(r, DEFAULT_CANDIDATES)),
                "best_all": best_method(r, [PLD, *DEFAULT_CANDIDATES]),
                "best_all_ratio": ratio(r, best_method(r, [PLD, *DEFAULT_CANDIDATES])),
                "mv_win_5": ratio(r, MV) >= 1.05,
                "mv_win_10": ratio(r, MV) >= 1.10,
                "mv_win_20": ratio(r, MV) >= 1.20,
                "frozen_win_5": ratio(r, FROZEN) >= 1.05,
                "frozen_win_10": ratio(r, FROZEN) >= 1.10,
                "frozen_win_20": ratio(r, FROZEN) >= 1.20,
            }
            for r in group
        ])

    rule, train_score = train_selector(train_rows)
    selector_json = rule.to_json()
    selector_json["train_score"] = train_score
    selector_json["test_offline_score"] = selector_score(test_rows, rule)
    selector_json["all_offline_score"] = selector_score(rows, rule)
    (out_dir / "selector_mv_frozen_rule.json").write_text(json.dumps(selector_json, indent=2, sort_keys=True) + "\n")

    report = {
        "n_all": len(rows),
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "frozen_candidates": {
            "baseline": PLD,
            "pld_sanity": FORCE_PLD,
            "best_current_method": MV,
            "high_risk_high_reward": FROZEN,
            "backup_variants": [TREE, G32, HUNK],
        },
        "oracle": {
            "all1000": summarize_oracles(rows),
            "train500": summarize_oracles(train_rows),
            "test500": summarize_oracles(test_rows),
        },
        "win_counts_all1000": {
            "mv_5": sum(ratio(r, MV) >= 1.05 for r in rows),
            "mv_10": sum(ratio(r, MV) >= 1.10 for r in rows),
            "mv_20": sum(ratio(r, MV) >= 1.20 for r in rows),
            "frozen_5": sum(ratio(r, FROZEN) >= 1.05 for r in rows),
            "frozen_10": sum(ratio(r, FROZEN) >= 1.10 for r in rows),
            "frozen_20": sum(ratio(r, FROZEN) >= 1.20 for r in rows),
        },
        "by_regime": bucket_breakdown(rows, regime),
        "by_family": bucket_breakdown(rows, family),
        "selector": selector_json,
    }
    (out_dir / "oracle_selector_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    inspection_sets = {
        "top25_mv_wins": inspect_rows(rows, MV, largest=True, n=25),
        "top25_frozen_wins": inspect_rows(rows, FROZEN, largest=True, n=25),
        "top25_mv_losses": inspect_rows(rows, MV, largest=False, n=25),
        "top25_frozen_losses": inspect_rows(rows, FROZEN, largest=False, n=25),
    }
    for name, items in inspection_sets.items():
        (out_dir / f"{name}.json").write_text(json.dumps(items, indent=2, sort_keys=True) + "\n")
        (out_dir / f"{name}.md").write_text(inspection_markdown(name, items) + "\n")

    md_lines = [
        "# Balanced Real-Commit Oracle and Selector",
        "",
        "## Frozen Candidate Set",
        "",
        f"- Baseline: `{PLD}`",
        f"- PLD sanity: `{FORCE_PLD}`",
        f"- Best current method: `{MV}`",
        f"- High-risk/high-reward: `{FROZEN}`",
        f"- Backups: `{TREE}`, `{G32}`, `{HUNK}`",
        "",
        "## Oracle Headroom",
        "",
    ]
    for group_name, group in [("All 1000", rows), ("Train 500", train_rows), ("Test 500", test_rows)]:
        md_lines.append(f"### {group_name}")
        table = [["Selector", "tok/s", "vs PLD"]]
        for item in summarize_oracles(group):
            table.append([item["selector"], f"{item['tokens_per_sec']:.1f}", f"{item['ratio_vs_pld']:.3f}x"])
        md_lines.append(markdown_table(table))
        md_lines.append("")
    md_lines.extend(
        [
            "## Win Counts",
            "",
            markdown_table(
                [
                    ["Method", ">=5%", ">=10%", ">=20%"],
                    ["MV", report["win_counts_all1000"]["mv_5"], report["win_counts_all1000"]["mv_10"], report["win_counts_all1000"]["mv_20"]],
                    ["Frozen", report["win_counts_all1000"]["frozen_5"], report["win_counts_all1000"]["frozen_10"], report["win_counts_all1000"]["frozen_20"]],
                ]
            ),
            "",
            "## Regime Breakdown",
            "",
            markdown_table(
                [["Bucket", "n", "MV/PLD", "Frozen/PLD", "Oracle/PLD"]]
                + [
                    [b["bucket"], b["n"], f"{b['mv_vs_pld']:.3f}x", f"{b['frozen_vs_pld']:.3f}x", f"{b['oracle_vs_pld']:.3f}x"]
                    for b in report["by_regime"]
                ]
            ),
            "",
            "## Family Breakdown",
            "",
            markdown_table(
                [["Bucket", "n", "MV/PLD", "Frozen/PLD", "Oracle/PLD"]]
                + [
                    [b["bucket"], b["n"], f"{b['mv_vs_pld']:.3f}x", f"{b['frozen_vs_pld']:.3f}x", f"{b['oracle_vs_pld']:.3f}x"]
                    for b in report["by_family"]
                ]
            ),
            "",
            "## Conservative Train Selector",
            "",
            f"Train ratio: {train_score['ratio_vs_pld']:.3f}x, choices: `{train_score['choices']}`.",
            f"Offline test ratio: {selector_json['test_offline_score']['ratio_vs_pld']:.3f}x, choices: `{selector_json['test_offline_score']['choices']}`.",
            f"Offline all ratio: {selector_json['all_offline_score']['ratio_vs_pld']:.3f}x, choices: `{selector_json['all_offline_score']['choices']}`.",
            "",
            "Selector JSON: `selector_mv_frozen_rule.json`.",
            "",
            "Inspection packets: `top25_mv_wins.md`, `top25_frozen_wins.md`, `top25_mv_losses.md`, `top25_frozen_losses.md`.",
        ]
    )
    (out_dir / "oracle_selector_report.md").write_text("\n".join(md_lines) + "\n")
    print(out_dir / "oracle_selector_report.md")


if __name__ == "__main__":
    main()
