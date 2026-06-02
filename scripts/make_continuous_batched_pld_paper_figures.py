#!/usr/bin/env python3
"""Generate final paper figures for Continuous Batched PLD Verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "analysis" / "final_paper_artifacts" / "continuous_batched_pld_final"
FIG_DIR = ARTIFACT_DIR / "figures"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def speedup_by_batch(out_dir: Path) -> None:
    report = _load(ROOT / "analysis" / "continuous_batched_pld_final_repeats" / "report.json")
    summary = report["summary"]
    batches = [1, 2, 4, 8]
    keys = [
        "blazedit_pld_w128_n10_b1",
        "continuous_batched_pld_w128_n10_b2",
        "continuous_batched_pld_w128_n10_b4",
        "continuous_batched_pld_w128_n10_b8",
    ]
    speedups = [summary[k]["speedup"]["mean"] for k in keys]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(batches, speedups, marker="o", linewidth=2.2, color="#1f77b4")
    ax.set_title("Aggregate Speedup by Batch Size")
    ax.set_xlabel("Verifier batch size")
    ax.set_ylabel("Speedup vs sequential PLD")
    ax.set_xticks(batches)
    ax.set_ylim(0.95, max(speedups) * 1.12)
    ax.grid(True, axis="y", alpha=0.3)
    for x, y in zip(batches, speedups):
        ax.annotate(f"{y:.3f}x", (x, y), textcoords="offset points", xytext=(0, 8), ha="center")
    _save(fig, out_dir, "speedup_by_batch_size")


def verifier_forwards(out_dir: Path) -> None:
    labels = ["PLD", "b2", "b4", "b8"]
    forwards = [6443, 3519, 2097, 1456]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bars = ax.bar(labels, forwards, color=["#777777", "#4c78a8", "#4c78a8", "#4c78a8"])
    ax.set_title("Verifier Forwards on Held-out test500")
    ax.set_xlabel("Method")
    ax.set_ylabel("Verifier forwards")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, forwards):
        ax.annotate(str(val), (bar.get_x() + bar.get_width() / 2, val), textcoords="offset points", xytext=(0, 6), ha="center")
    _save(fig, out_dir, "verifier_forwards_reduction")


def throughput_vs_latency(out_dir: Path) -> None:
    report = _load(ROOT / "analysis" / "continuous_batched_pld_latency" / "report.json")
    rows = report["rows"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for row in rows:
        batch = row["batch_size"]
        tok_s = row["aggregate_tok_s_mean"]
        latency = row["latency_p50_ms"] or 0.0
        label = "PLD" if batch == 1 and row["method"] == "blazedit_pld_w128_n10" else f"b{batch}"
        ax.scatter(latency, tok_s, s=80, label=label)
        ax.annotate(label, (latency, tok_s), textcoords="offset points", xytext=(6, 6))
    ax.set_title("Throughput / Latency Tradeoff")
    ax.set_xlabel("Per-task p50 latency (ms; PLD latency not instrumented)")
    ax.set_ylabel("Aggregate tok/s")
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, "throughput_vs_latency")


def refill_ablation(out_dir: Path) -> None:
    controlled_path = (
        ROOT
        / "analysis"
        / "batched_pld_controlled_ablation"
        / "controlled_ablation_test500_v1"
        / "summary.json"
    )
    if controlled_path.exists():
        report = _load(controlled_path)
        rows = report.get("summary", {})
        keys = ["b8_pool32_default_continuous", "b8_pool32_default_no_refill"]
        if all(k in rows for k in keys):
            labels = ["continuous refill", "no refill"]
            tok_s = [rows[k]["fields"]["tok_s"]["mean"] for k in keys]
            fig, ax = plt.subplots(figsize=(5.4, 3.4))
            bars = ax.bar(labels, tok_s, color=["#2ca02c", "#d62728"])
            ax.set_title("Effect of Continuous Refill")
            ax.set_ylabel("Aggregate tok/s")
            ax.grid(True, axis="y", alpha=0.3)
            for bar, val in zip(bars, tok_s):
                ax.annotate(
                    f"{val:.1f}",
                    (bar.get_x() + bar.get_width() / 2, val),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                )
            _save(fig, out_dir, "refill_ablation")
            return
    report = _load(ROOT / "analysis" / "batched_pld_ablation" / "report.json")
    rows = {row["config_id"]: row for row in report["rows"]}
    keys = ["b8_pool32_default_continuous", "b8_pool32_default_no_refill"]
    if not all(k in rows for k in keys):
        (out_dir / "refill_ablation_skipped.txt").write_text(
            "Refill ablation skipped: required configs not found.\n",
            encoding="utf-8",
        )
        return
    labels = ["continuous refill", "no refill"]
    tok_s = [rows[k]["generated_tokens_per_sec"] for k in keys]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    bars = ax.bar(labels, tok_s, color=["#2ca02c", "#d62728"])
    ax.set_title("Effect of Continuous Refill")
    ax.set_ylabel("Aggregate tok/s")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, tok_s):
        ax.annotate(f"{val:.1f}", (bar.get_x() + bar.get_width() / 2, val), textcoords="offset points", xytext=(0, 6), ha="center")
    _save(fig, out_dir, "refill_ablation")


def negative_results_summary(out_dir: Path) -> None:
    labels = [
        "exact rerank\nK32 oracle",
        "queued MTP\nactual policy oracle",
        "trained router\nMTP",
        "syntax-slot",
        "weak-router\ncapping",
        "selective\nLM-head",
        "batched PLD\nb8",
    ]
    speedups = [1.122, 1.129, 1.035, 1.005, 0.968, 0.997, 1.717]
    colors = ["#999999"] * 6 + ["#1f77b4"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bars = ax.bar(labels, speedups, color=colors)
    ax.axhline(1.20, color="#d62728", linestyle="--", linewidth=1.5, label="1.20x target")
    ax.set_title("Decoder Variants vs Final Verifier Scheduling")
    ax.set_ylabel("Speedup / projected speedup")
    ax.set_ylim(0.85, 1.85)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", frameon=False)
    for bar, val in zip(bars, speedups):
        ax.annotate(f"{val:.3f}x", (bar.get_x() + bar.get_width() / 2, val), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)
    _save(fig, out_dir, "negative_results_summary")


def generic_vs_pld_batching(out_dir: Path) -> None:
    generic_path = ROOT / "analysis" / "generic_batched_greedy_baseline" / "report.json"
    if not generic_path.exists():
        (out_dir / "generic_vs_pld_batching_skipped.txt").write_text(
            "Generic-vs-PLD batching figure skipped: generic greedy report not found.\n",
            encoding="utf-8",
        )
        return
    generic = _load(generic_path)
    sweep_path = ROOT / "analysis" / "generic_batched_greedy_baseline" / "b8_pool_sweep_report.json"
    sweep = _load(sweep_path) if sweep_path.exists() else {"rows": []}
    pld = _load(ROOT / "analysis" / "continuous_batched_pld_final_repeats" / "report.json")
    greedy_seq = float(generic["sequential"].get("tokens_per_sec", 0.0))
    successful: list[dict[str, Any]] = []
    for row in generic.get("batched", []):
        if not row.get("error"):
            successful.append(
                {
                    "batch_size": int(row.get("batch_size", 0)),
                    "active_pool_size": int(row.get("active_pool_size", 0)),
                    "generated_tokens_per_sec": float(row.get("generated_tokens_per_sec", 0.0)),
                    "source": "main",
                }
            )
    for row in sweep.get("rows", []):
        if row.get("status") == "success":
            successful.append(
                {
                    "batch_size": int(row.get("batch_size", 0)),
                    "active_pool_size": int(row.get("active_pool_size", 0)),
                    "generated_tokens_per_sec": float(row.get("tok_s", 0.0)),
                    "source": "pool_sweep",
                }
            )
    greedy_best = max(successful, key=lambda r: float(r.get("generated_tokens_per_sec", 0.0))) if successful else None
    if greedy_best is None:
        (out_dir / "generic_vs_pld_batching_skipped.txt").write_text(
            "Generic-vs-PLD batching figure skipped: no successful batched greedy row found.\n",
            encoding="utf-8",
        )
        return
    greedy_label = (
        f"Batched\ngreedy b{int(greedy_best.get('batch_size'))}"
        f"/pool{int(greedy_best.get('active_pool_size'))}"
    )
    labels = ["Sequential\ngreedy", greedy_label, "Sequential\nPLD", "Batched\nPLD b8"]
    values = [
        greedy_seq,
        float(greedy_best.get("generated_tokens_per_sec", 0.0)),
        float(pld["summary"]["blazedit_pld_w128_n10_b1"]["tok_s"]["mean"]),
        float(pld["summary"]["continuous_batched_pld_w128_n10_b8"]["tok_s"]["mean"]),
    ]
    colors = ["#777777", "#9ecae1", "#777777", "#1f77b4"]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Generic Batching vs Batched PLD")
    ax.set_ylabel("Aggregate tok/s")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        ax.annotate(f"{val:.1f}", (bar.get_x() + bar.get_width() / 2, val), textcoords="offset points", xytext=(0, 6), ha="center")
    if any(row.get("status") == "oom" and int(row.get("active_pool_size", 0)) in {16, 32} for row in sweep.get("rows", [])):
        ax.text(
            0.5,
            0.94,
            "Generic greedy b8 pool16/pool32 OOM on L40S",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color="#555555",
        )
    _save(fig, out_dir, "generic_vs_pld_batching")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(FIG_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    speedup_by_batch(out_dir)
    verifier_forwards(out_dir)
    throughput_vs_latency(out_dir)
    refill_ablation(out_dir)
    negative_results_summary(out_dir)
    generic_vs_pld_batching(out_dir)
    print(f"wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
