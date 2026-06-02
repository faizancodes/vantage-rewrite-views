#!/usr/bin/env python3
"""Generate paper tables for Continuous Batched PLD Verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "analysis" / "final_paper_artifacts" / "continuous_batched_pld_final"
TABLE_DIR = ARTIFACT_DIR / "tables"
EXACT_THROUGHPUT_PATH = (
    ROOT
    / "analysis"
    / "continuous_batched_pld_final_repeats"
    / "continuous_batched_pld_fp32_eager_throughput_test500_v1"
    / "report.json"
)
EXACT_THROUGHPUT_SHARDED_PATH = (
    ROOT
    / "analysis"
    / "continuous_batched_pld_final_repeats"
    / "continuous_batched_pld_fp32_eager_throughput_test500_sharded_chunk512_v1"
    / "report.json"
)
EXACT_THROUGHPUT_SUBSET_PATH = (
    ROOT
    / "analysis"
    / "continuous_batched_pld_final_repeats"
    / "continuous_batched_pld_fp32_eager_throughput_test100_subset_v1"
    / "report.json"
)
CONTROLLED_ABLATION_PATH = (
    ROOT
    / "analysis"
    / "batched_pld_controlled_ablation"
    / "controlled_ablation_test500_v1"
    / "summary.json"
)


class RawTex(str):
    """String that should be written unescaped in LaTeX tables."""


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _load(path)


def _fmt(x: float, digits: int = 1) -> str:
    return f"{float(x):.{digits}f}"


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


def _write_table(name: str, headers: list[str], rows: list[list[object]], caption: str, label: str) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    md_lines = [
        f"# {caption}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        md_lines.append("| " + " | ".join(str(x) for x in row) + " |")
    (TABLE_DIR / f"{name}.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    colspec = "l" * len(headers)
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{_latex_escape(caption)}}}",
        rf"\label{{{_latex_escape(label)}}}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(_latex_escape(h) for h in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        tex_lines.append(" & ".join(_latex_escape(x) for x in row) + r" \\")
    tex_lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    (TABLE_DIR / f"{name}.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def main_results() -> None:
    source = ROOT / "analysis" / "continuous_batched_pld_final_repeats" / "report.json"
    report = _load(source)
    summary = report["summary"]
    order = [
        ("blazedit_pld_w128_n10_b1", "Sequential BlazEdit PLD"),
        ("continuous_batched_pld_w128_n10_b2", "Continuous Batched PLD"),
        ("continuous_batched_pld_w128_n10_b4", "Continuous Batched PLD"),
        ("continuous_batched_pld_w128_n10_b8", "Continuous Batched PLD"),
    ]
    baseline_forwards = summary["blazedit_pld_w128_n10_b1"]["verifier_forwards"]["mean"]
    rows: list[list[object]] = []
    for key, label in order:
        row = summary[key]
        batch = int(row["batch_size"])
        forwards = row["verifier_forwards"]["mean"]
        reduction = 100.0 * (1.0 - forwards / baseline_forwards)
        drift = "500/500" if batch == 1 else f"{row['output_match_count']['mean']:.0f}/500"
        rows.append(
            [
                label,
                batch,
                f"{_fmt(row['tok_s']['mean'])} ± {_fmt(row['tok_s']['std'])}",
                RawTex(f"{row['speedup']['mean']:.3f}$\\times$"),
                f"{forwards:.0f}",
                f"{reduction:.1f}%",
                drift,
            ]
        )
    _write_table(
        "main_results",
        [
            "Method",
            "Batch",
            "Tok/s mean ± std",
            "Speedup",
            "Verifier forwards",
            "Forward reduction",
            "bf16/SDPA task matches",
        ],
        rows,
        "Held-out test500 aggregate throughput for Continuous Batched PLD.",
        "tab:continuous-batched-main",
    )


def correctness() -> None:
    corr = _load(ROOT / "analysis" / "continuous_batched_pld_correctness_sharded" / "report.json")
    audit = _load(ROOT / "analysis" / "continuous_batched_pld_task_audit_test500" / "report.json")
    rows: list[list[object]] = []
    for batch in (1, 4, 8):
        r = corr["aggregate"]["batch_results"][str(batch)]
        rows.append(
            [
                "fp32/eager sharded test500",
                batch,
                f"{r['exact_token_id_matches']}/{r['tasks']}",
                f"{r['decoded_output_matches']}/{r['tasks']}",
                f"{r['finish_reason_matches']}/{r['tasks']}",
                f"{r['generated_length_matches']}/{r['tasks']}",
                "0 skipped",
            ]
        )
    _write_table(
        "correctness",
        ["Validation", "Batch/metric", "Token IDs", "Decoded", "Finish", "Length", "Notes"],
        rows,
        "Deterministic correctness validation for Continuous Batched PLD.",
        "tab:continuous-batched-correctness",
    )
    audit_rows = [
        ["task-isolation audit", "emitted tokens audited", audit.get("emitted_tokens_audited", 0)],
        ["task-isolation audit", "task-mixing violations", audit.get("task_mixing_violations", 0)],
        ["task-isolation audit", "cache ownership violations", audit.get("cache_ownership_violations", 0)],
        ["task-isolation audit", "finished-task violations", audit.get("finished_task_violations", 0)],
        ["task-isolation audit", "scatter/gather mismatches", audit.get("scatter_gather_mismatches", 0)],
        ["task-isolation audit", "unverified-token violations", audit.get("unverified_token_violations", 0)],
    ]
    _write_table(
        "task_isolation_audit",
        ["Validation", "Metric", "Value"],
        audit_rows,
        "Task isolation audit for the full batch=8 timing trace.",
        "tab:continuous-batched-audit",
    )
    repeats = _load(ROOT / "analysis" / "continuous_batched_pld_final_repeats" / "report.json")
    if EXACT_THROUGHPUT_PATH.exists():
        exact_timing_path = EXACT_THROUGHPUT_PATH
        exact_timing_kind = "unsharded_full"
    elif EXACT_THROUGHPUT_SHARDED_PATH.exists():
        exact_timing_path = EXACT_THROUGHPUT_SHARDED_PATH
        exact_timing_kind = "sharded_full"
    else:
        exact_timing_path = EXACT_THROUGHPUT_SUBSET_PATH
        exact_timing_kind = "subset"
    exact_timing = _load_optional(exact_timing_path)
    exact_timing_note = "no speed claim from available artifacts"
    exact_timing_rows: list[list[object]] = []
    if exact_timing is not None:
        exact_summary = exact_timing.get("summary", {})
        exact_order = [
            ("blazedit_pld_w128_n10_b1", "Sequential BlazEdit PLD"),
            ("continuous_batched_pld_w128_n10_b2", "archived continuous-batched PLD prototype"),
            ("continuous_batched_pld_w128_n10_b4", "archived continuous-batched PLD prototype"),
            ("continuous_batched_pld_w128_n10_b8", "archived continuous-batched PLD prototype"),
        ]
        seq_exact = exact_summary.get("blazedit_pld_w128_n10_b1", {})
        seq_forwards = float(seq_exact.get("verifier_forwards", {}).get("mean", 0.0) or 0.0)
        for key, label in exact_order:
            if key not in exact_summary:
                continue
            row = exact_summary[key]
            batch = int(row.get("batch_size", 0))
            forwards = float(row.get("verifier_forwards", {}).get("mean", 0.0) or 0.0)
            reduction = 100.0 * (1.0 - forwards / seq_forwards) if seq_forwards else 0.0
            n_tasks = int(exact_timing.get("args", {}).get("n", 500)) if exact_timing else 500
            matches = float(row.get("output_match_count", {}).get("mean", 0.0) or 0.0)
            peak_gb = float(row.get("memory_peak_gb", {}).get("mean", 0.0) or 0.0)
            exact_timing_rows.append(
                [
                    "fp32/eager",
                    label,
                    batch,
                    f"{_fmt(row['tok_s']['mean'])} ± {_fmt(row['tok_s']['std'])}",
                    RawTex(f"{row['speedup']['mean']:.3f}$\\times$"),
                    f"{forwards:.0f}",
                    f"{reduction:.1f}%",
                    f"{matches:.0f}/{n_tasks}",
                    "unavailable" if batch == 1 and peak_gb == 0.0 else f"{peak_gb:.2f}",
                ]
            )
        if exact_timing_rows:
            n_tasks = int(exact_timing.get("args", {}).get("n", 500))
            if exact_timing_kind in {"unsharded_full", "sharded_full"}:
                exact_timing_note = RawTex(
                    f"{exact_timing_rows[-1][3]}; {exact_timing_rows[-1][4]} for batch=8"
                )
            else:
                exact_timing_note = RawTex(
                    f"subset n={n_tasks}: {exact_timing_rows[-1][3]}; "
                    f"{exact_timing_rows[-1][4]} for batch=8"
                )
    if not exact_timing_rows:
        exact_timing_rows = [
            [
                "fp32/eager",
                "no throughput artifact",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                "500/500 correctness only",
                "n/a",
            ]
        ]
    if exact_timing is not None and exact_timing_kind == "unsharded_full":
        exact_caption = "Exact-backend throughput and correctness evidence on held-out test500."
    elif exact_timing is not None and exact_timing_kind == "sharded_full":
        exact_caption = (
            "Exact-backend throughput and correctness evidence on held-out test500, "
            "aggregated from independent shards with 512-token chunked prompt prefill."
        )
    else:
        exact_caption = (
            "Exact-backend throughput diagnostic on a deterministic test100 subset; "
            "full test500 fp32/eager timing is absent."
        )
    _write_table(
        "exact_backend_throughput",
        [
            "Backend",
            "Method",
            "Batch",
            "Tok/s mean ± std",
            "Speedup",
            "Verifier forwards",
            "Forward reduction",
            "Task matches",
            "Peak GB",
        ],
        exact_timing_rows,
        exact_caption,
        "tab:exact-backend-throughput",
    )
    backend_rows = [
        [
            "fp32/eager",
            "deterministic equivalence validation",
            "sequential + batch=1/4/8",
            "500/500 token-ID exact for all validated batch sizes",
            exact_timing_note,
            "byte-exact scheduler claim",
        ],
        [
            "bf16/SDPA",
            "production timing baseline",
            "sequential",
            "500/500 self-baseline",
            "492.1 ± 2.1 tok/s",
            "fast timing path",
        ],
        [
            "bf16/SDPA",
            "production timing method",
            "batch=8",
            "452/500 vs bf16/SDPA sequential",
            RawTex("845.0 ± 7.1 tok/s; 1.717$\\times$"),
            "not byte-exact; drift summarized separately",
        ],
    ]
    _write_table(
        "exactness_vs_timing_backend",
        [
            "Backend",
            "Claim path",
            "Configuration",
            "Task-match evidence",
            "Throughput evidence",
            "Notes",
        ],
        backend_rows,
        "Exact scheduler validation and production timing path are separate claims.",
        "tab:backend-claims",
    )
    drift_rows = []
    for key in [
        "blazedit_pld_w128_n10_b1",
        "continuous_batched_pld_w128_n10_b2",
        "continuous_batched_pld_w128_n10_b4",
        "continuous_batched_pld_w128_n10_b8",
    ]:
        r = repeats["summary"][key]
        batch = int(r["batch_size"])
        matches = int(r["output_match_count"]["mean"])
        tasks = 500
        drift_rows.append(
            [
                "bf16/SDPA same-run vs sequential PLD",
                batch,
                f"{matches}/{tasks}",
                tasks - matches,
                "absent from timing artifact",
                "absent from timing artifact",
                "task-level token-ID counts only; detailed drift tracing requires a rerun",
            ]
        )
    _write_table(
        "timing_path_drift",
        [
            "Comparison",
            "Batch",
            "Exact task matches",
            "Task mismatches",
            "First mismatch",
            "Quality delta",
            "Notes",
        ],
        drift_rows,
        "Production/timing-path drift under bf16/SDPA.",
        "tab:timing-path-drift",
    )


def submission_blockers() -> None:
    rows = []
    if not EXACT_THROUGHPUT_PATH.exists() and not EXACT_THROUGHPUT_SHARDED_PATH.exists():
        status = (
            "subset source artifact present; full test500 source artifact absent"
            if EXACT_THROUGHPUT_SUBSET_PATH.exists()
            else "no source artifact present"
        )
        rows.append(
            [
                "fp32/eager full-test500 throughput",
                status,
                "required before claiming byte-exact speedup",
                "modal run vantage_runtime_debian_app.py::launch_batched_pld_repeated_timing --split test --n 500 --repeats 3 --dtype fp32 --attn eager --batch-sizes 2,4,8 --active-pool-size 32 --bucket-policy default --refill-policy continuous --version continuous_batched_pld_fp32_eager_throughput_test500_v1",
            ]
        )
    if not CONTROLLED_ABLATION_PATH.exists():
        rows.append(
            [
                "controlled 3-repeat ablation grid",
                "no source artifact present",
                "required before main-text scheduler-lever claims",
                "modal run vantage_runtime_debian_app.py::launch_batched_pld_controlled_ablation --split test --n 500 --repeats 3 --dtype bf16 --attn sdpa --version controlled_ablation_test500_v1 --wait",
            ]
        )
    rows.extend([
        [
            "external vLLM/HF prompt-lookup baselines",
            "local smoke-failure artifacts present; no comparable throughput artifact",
            "limits external serving comparison",
            "restore generated external-baseline artifacts from Hugging Face before regenerating this archived table",
        ],
    ])
    _write_table(
        "submission_blockers",
        ["Missing artifact", "Current status", "Effect on claim", "Reproduction command/config"],
        rows,
        "Submission-critical experiments absent from the available artifact package.",
        "tab:submission-blockers",
    )


def ablation() -> None:
    if CONTROLLED_ABLATION_PATH.exists():
        report = _load(CONTROLLED_ABLATION_PATH)
        summary = report.get("summary", {})
        selected = [
            "seq",
            "b2_pool32_default_continuous",
            "b4_pool32_default_continuous",
            "b8_pool32_default_continuous",
            "b8_pool8_default_continuous",
            "b8_pool16_default_continuous",
            "b8_pool32_default_no_refill",
            "b8_pool32_fine_continuous",
            "b8_pool32_single_continuous",
        ]
        rows = []
        for key in selected:
            if key not in summary:
                continue
            item = summary[key]
            fields = item.get("fields", {})
            peak_gb = float(fields.get("memory_peak_gb", {}).get("mean", 0.0) or 0.0)
            rows.append(
                [
                    key,
                    item.get("active_pool_size", ""),
                    item.get("bucket_policy", ""),
                    item.get("refill_policy", ""),
                    f"{fields.get('tok_s', {}).get('mean', 0.0):.1f} ± {fields.get('tok_s', {}).get('std', 0.0):.1f}",
                    RawTex(
                        f"{fields.get('speedup_vs_same_run_sequential', {}).get('mean', 0.0):.3f}$\\times$"
                    ),
                    f"{fields.get('verifier_forwards', {}).get('mean', 0.0):.0f}",
                    f"{fields.get('verifier_forward_reduction_pct', {}).get('mean', 0.0):.1f}%",
                    f"{fields.get('input_padding_waste_pct', {}).get('mean', 0.0):.1f}%",
                    "unavailable" if key == "seq" and peak_gb == 0.0 else f"{peak_gb:.2f}",
                    f"{item.get('n_success', 0)}/{item.get('n_repeats_requested', 0)}",
                ]
            )
        _write_table(
            "ablation",
            [
                "Config",
                "Pool",
                "Buckets",
                "Refill",
                "Tok/s mean ± std",
                "Speedup",
                "Forwards",
                "Forward reduction",
                "Input padding waste",
                "Peak GB",
                "Repeats",
            ],
            rows,
            "Controlled scheduler ablations under the held-out test500 bf16/SDPA protocol.",
            "tab:scheduler-ablation",
        )
        return

    report = _load(ROOT / "analysis" / "batched_pld_ablation" / "report.json")
    same_run_baseline = float(report["sequential"]["tokens_per_sec"])
    rows_by_id = {row["config_id"]: row for row in report["rows"]}
    selected = [
        "b8_pool32_default_continuous",
        "b8_pool32_default_no_refill",
        "b8_pool8_default_continuous",
        "b8_pool16_default_continuous",
        "b8_pool32_fine_continuous",
        "b8_pool32_single_continuous",
    ]
    rows = []
    for key in selected:
        if key not in rows_by_id:
            continue
        r = rows_by_id[key]
        rows.append(
            [
                key,
                r["active_pool_size"],
                r["bucket_policy"],
                r["refill_policy"],
                f"{r['generated_tokens_per_sec']:.1f}",
                RawTex(f"{r['speedup_vs_sequential']:.3f}$\\times$"),
                f"{same_run_baseline:.1f}",
                r["verifier_forwards"],
                f"{100.0 * r['verifier_forward_reduction']:.1f}%",
                f"{r.get('input_padding_waste_pct', 0.0):.1f}%",
                "earlier one-repeat diagnostic protocol",
            ]
        )
    _write_table(
        "ablation",
        [
            "Config",
            "Pool",
            "Buckets",
            "Refill",
            "Tok/s",
            "Speedup vs same-run PLD",
            "Same-run PLD tok/s",
            "Forwards",
            "Forward reduction",
            "Input padding waste",
            "Source note",
        ],
        rows,
        "Earlier scheduler diagnostics for batch=8 Continuous Batched PLD.",
        "tab:continuous-batched-ablation",
    )


def latency() -> None:
    report = _load(ROOT / "analysis" / "continuous_batched_pld_latency" / "report.json")
    rows = []
    for r in report["rows"]:
        method = "Sequential PLD" if r["method"] == "blazedit_pld_w128_n10" else "Continuous Batched PLD"
        rows.append(
            [
                method,
                r["batch_size"],
                f"{r['aggregate_tok_s_mean']:.1f}",
                RawTex(f"{r['speedup_mean']:.3f}$\\times$"),
                "n/a" if r["latency_mean_ms"] == 0 else f"{r['latency_mean_ms']:.1f}",
                "n/a" if r["latency_p50_ms"] == 0 else f"{r['latency_p50_ms']:.1f}",
                "n/a" if r["latency_p90_ms"] == 0 else f"{r['latency_p90_ms']:.1f}",
                "n/a" if r["latency_p99_ms"] == 0 else f"{r['latency_p99_ms']:.1f}",
                "offline all-at-once trace; queue/TTFT absent",
            ]
        )
    _write_table(
        "latency",
        [
            "Method",
            "Batch",
            "Aggregate tok/s",
            "Speedup",
            "Mean latency ms",
            "p50",
            "p90",
            "p99",
            "Queue wait",
        ],
        rows,
        "Aggregate throughput and per-task latency tradeoff.",
        "tab:continuous-batched-latency",
    )


def dataset_stats() -> None:
    stats_path = ROOT / "artifacts" / "dataset_stats.json"
    if not stats_path.exists():
        return
    report = _load(stats_path)
    rows = []
    for r in report.get("splits", []):
        rows.append(
            [
                r["name"],
                r["tasks"],
                r["unique_repos"],
                r["unique_commits"],
                f"{r['mean_input_tokens']:.1f}",
                f"{r['input_tokens_p50']:.0f}/{r['input_tokens_p90']:.0f}/{r['input_tokens_p99']:.0f}",
                f"{r['mean_output_tokens']:.1f}",
                f"{r['output_tokens_p50']:.0f}/{r['output_tokens_p90']:.0f}/{r['output_tokens_p99']:.0f}",
                f"{r['copy_overlap_mean']:.3f}",
                f"{r['edit_distance_tokens_mean']:.1f}",
                "manifest metadata",
            ]
        )
    _write_table(
        "dataset_stats",
        [
            "Split",
            "Tasks",
            "Repos",
            "Commits",
            "Mean input toks",
            "Input p50/p90/p99",
            "Mean output toks",
            "Output p50/p90/p99",
            "Copy overlap mean",
            "Edit distance mean",
            "Notes",
        ],
        rows,
        "Real-commit benchmark statistics.",
        "tab:dataset-stats",
    )


def generation_stats() -> None:
    stats_path = ROOT / "artifacts" / "generation_stats.json"
    if not stats_path.exists():
        return
    report = _load(stats_path)
    rows = []
    for r in report.get("rows", []):
        if r.get("batch_size") not in (1, 8):
            continue
        method = "Sequential PLD" if r["method"] == "blazedit_pld_w128_n10" else "archived continuous-batched PLD prototype"
        rows.append(
            [
                method,
                r["backend"],
                r["batch_size"],
                f"{r['total_emitted_tokens_mean']:.0f}",
                f"{r['mean_emitted_tokens_per_task']:.1f}",
                r["throughput_denominator"],
                "absent from timing artifact"
                if r["stop_reason_distribution"] == "unavailable in current artifact"
                else r["stop_reason_distribution"],
            ]
        )
    _write_table(
        "generation_denominator",
        [
            "Method",
            "Backend",
            "Batch",
            "Total emitted toks",
            "Mean emitted/task",
            "Throughput denominator",
            "Stop reasons",
        ],
        rows,
        "Generation-token denominator used for throughput.",
        "tab:generation-denominator",
    )


def leakage_audit_table() -> None:
    audit_path = ROOT / "artifacts" / "dataset_leakage_audit.json"
    if not audit_path.exists():
        return
    report = _load(audit_path)
    rows = []
    for r in report.get("manifests", []):
        rows.append(
            [
                Path(r["manifest"]).stem,
                r["tasks"],
                r["exact_target_in_prompt"],
                r["exact_target_in_pre_edit_context"],
                r["target_equals_pre_edit_context"],
                r.get("large_target_chunk_in_prompt", 0),
                r.get("large_target_chunk_in_pre_edit_context", 0),
                r.get("patch_or_diff_marker_in_prompt", 0),
            ]
        )
    _write_table(
        "leakage_audit",
        [
            "Split",
            "Tasks",
            "Exact target in prompt",
            "Exact target in context",
            "Pre-edit equals target",
            "Large chunk in prompt",
            "Large chunk in context",
            "Diff marker in prompt",
        ],
        rows,
        "Dataset leakage audit over manifest prompt and pre-edit source context.",
        "tab:leakage-audit",
    )


def system_breakdown() -> None:
    breakdown_path = ROOT / "artifacts" / "system_breakdown_analysis.json"
    if breakdown_path.exists():
        report = _load(breakdown_path)
        rows = []
        for row in report.get("rows", []):
            rows.append(
                [
                    row.get("config", ""),
                    f"{float(row.get('top_level_measured_ms') or 0.0):.0f}",
                    f"{float(row.get('target_forward_ms') or row.get('verifier_forward_ms') or 0.0):.0f} ({float(row.get('verifier_forward_pct') or 0.0):.1f}%)",
                    f"{float(row.get('prefill_ms') or 0.0):.0f} ({float(row.get('prefill_pct') or 0.0):.1f}%)",
                    f"{float(row.get('pld_lookup_ms') or 0.0):.0f} ({float(row.get('pld_lookup_pct') or 0.0):.1f}%)",
                    f"{float(row.get('scheduler_overhead_ms') or 0.0):.0f} ({float(row.get('scheduler_overhead_pct') or 0.0):.1f}%)",
                    f"{float(row.get('residual_ms') or 0.0):.0f} ({float(row.get('residual_pct') or 0.0):.1f}%)",
                    f"{float(row.get('memory_peak_gb') or 0.0):.2f}",
                    "top-level accounted; scheduler/runtime is aggregate",
                ]
            )
        _write_table(
            "system_breakdown",
            [
                "Config",
                "Wall ms",
                "Verifier forward",
                "Prompt prefill",
                "PLD lookup",
                "Scheduler/runtime",
                "Residual",
                "Peak GB",
                "Claim status",
            ],
            rows,
            "Top-level system time accounting from final repeated timing logs.",
            "tab:system-breakdown",
        )
        return

    report = _load(ROOT / "analysis" / "continuous_batched_pld_final_repeats" / "report.json")
    rows = []
    for batch in (2, 4, 8):
        batch_rows = [
            r
            for r in report["rows"]
            if r.get("method") == "continuous_batched_pld_w128_n10"
            and int(r.get("batch_size", 0)) == batch
        ]
        if not batch_rows:
            continue
        mean = lambda key: sum(float(r.get(key) or 0.0) for r in batch_rows) / len(batch_rows)
        forwards = sum(float(r.get("verifier_forwards") or 0.0) for r in batch_rows) / len(batch_rows)
        real_verified = sum(float(r.get("real_verified_tokens") or 0.0) for r in batch_rows) / len(batch_rows)
        input_pad = sum(float(r.get("input_padding_waste_tokens") or 0.0) for r in batch_rows) / len(batch_rows)
        pad_pct = 100.0 * input_pad / max(1.0, real_verified + input_pad)
        rows.append(
            [
                f"b{batch}",
                f"{mean('wall_ms'):.0f}",
                f"{mean('total_forward_ms'):.0f}",
                f"{mean('pld_lookup_ms'):.0f}",
                f"{mean('scheduler_overhead_ms'):.0f}",
                f"{max(0.0, mean('wall_ms') - mean('total_forward_ms') - mean('pld_lookup_ms') - mean('scheduler_overhead_ms')):.0f}",
                f"{100.0 * max(0.0, mean('wall_ms') - mean('total_forward_ms') - mean('pld_lookup_ms') - mean('scheduler_overhead_ms')) / max(1.0, mean('wall_ms')):.1f}%",
                f"{forwards:.0f}",
                f"{mean('active_tasks_mean'):.1f}",
                f"{mean('verified_tokens_per_forward'):.1f}",
                f"{mean('accepted_tokens_per_forward'):.1f}",
                f"{pad_pct:.1f}%",
                f"{mean('memory_peak_gb'):.2f}",
            ]
        )
    _write_table(
        "system_breakdown",
        [
            "Config",
            "Wall ms",
            "Target forward ms",
            "PLD lookup ms",
            "Scheduler overhead ms",
            "Residual ms",
            "Residual %",
            "Verifier forwards",
            "Mean active tasks",
            "Verified toks/forward",
            "Accepted toks/forward",
            "Input padding waste",
            "Peak GB",
        ],
        rows,
        "Measured system breakdown from final repeated timing logs.",
        "tab:system-breakdown",
    )


def _external_report(path: str) -> dict[str, Any] | None:
    report_path = ROOT / path
    return _load_optional(report_path)


def _external_memory_text(value: Any) -> str:
    if isinstance(value, (int, float)) and value > 0:
        return f"{float(value):.2f}"
    return "not captured"


def external_baselines() -> None:
    specs = [
        (
            "vLLM greedy",
            "vLLM",
            "generic greedy continuous batching",
            "analysis/external_baselines/external_baselines_l40s_test500_v1/vllm_greedy/report.json",
            "generic greedy output; not PLD-equivalent",
        ),
        (
            "vLLM n-gram",
            "vLLM",
            "ngram speculation, prompt_lookup_min=2, prompt_lookup_max=128, num_speculative_tokens=8",
            "analysis/external_baselines/external_baselines_l40s_test500_v2/vllm_ngram/report.json",
            "external prompt-lookup serving baseline; not PLD-equivalent",
        ),
        (
            "HF prompt lookup",
            "Transformers",
            "generate(prompt_lookup_num_tokens=128, max_matching_ngram_size=2)",
            "analysis/external_baselines/external_baselines_l40s_test500_v1/hf_prompt_lookup/report.json",
            "sequential HF generate baseline; not a continuous-batching engine",
        ),
    ]
    rows: list[list[object]] = []
    for method, engine, policy, path, notes in specs:
        report = _external_report(path)
        if not report:
            rows.append([method, engine, policy, "missing", "", "", "", "", notes])
            continue
        config = report.get("config", {})
        result = report.get("result", {})
        env = report.get("environment", {})
        packages = env.get("packages", {}) if isinstance(env, dict) else {}
        status = report.get("status", "unknown")
        tok_s = result.get("tokens_per_sec")
        wall_ms = result.get("generation_wall_ms", result.get("wall_ms"))
        init_ms = result.get("engine_init_ms")
        rows.append(
            [
                method,
                f"{engine} {packages.get('vllm') or packages.get('transformers') or ''}".strip(),
                policy,
                f"{config.get('n', result.get('n_prompts', ''))}",
                f"{float(tok_s):.1f}" if isinstance(tok_s, (int, float)) else status,
                f"{int(result.get('total_new_tokens') or 0)}" if result else "",
                f"{float(wall_ms) / 1000.0:.1f}" if isinstance(wall_ms, (int, float)) else "",
                f"{float(init_ms) / 1000.0:.1f}" if isinstance(init_ms, (int, float)) else "n/a",
                _external_memory_text(result.get("memory_peak_gb")),
                notes,
            ]
        )
    _write_table(
        "external_baselines",
        [
            "Method",
            "Engine",
            "Policy",
            "Tasks",
            "Tok/s",
            "Emitted toks",
            "Gen wall s",
            "Init s",
            "Peak GB",
            "Notes",
        ],
        rows,
        "External generation baselines on held-out test500 L40S. These use vLLM/HF generation paths, not the custom PLD-equivalent decoder.",
        "tab:external-baselines",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(TABLE_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global TABLE_DIR
    TABLE_DIR = Path(args.output_dir)
    main_results()
    correctness()
    ablation()
    latency()
    dataset_stats()
    generation_stats()
    leakage_audit_table()
    system_breakdown()
    external_baselines()
    print(f"wrote tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
