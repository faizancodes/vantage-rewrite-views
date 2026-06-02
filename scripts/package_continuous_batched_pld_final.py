#!/usr/bin/env python3
"""Freeze the Continuous Batched PLD paper artifact bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "analysis" / "final_paper_artifacts" / "continuous_batched_pld_final"


REPORT_SOURCES = {
    "continuous_batched_pld_final_report.md": ROOT / "analysis" / "continuous_batched_pld_final_report.md",
    "continuous_batched_pld_final_repeats_report.md": ROOT / "analysis" / "continuous_batched_pld_final_repeats" / "report.md",
    "continuous_batched_pld_final_repeats_report.json": ROOT / "analysis" / "continuous_batched_pld_final_repeats" / "report.json",
    "continuous_batched_pld_correctness_sharded_report.md": ROOT / "analysis" / "continuous_batched_pld_correctness_sharded" / "report.md",
    "continuous_batched_pld_correctness_sharded_report.json": ROOT / "analysis" / "continuous_batched_pld_correctness_sharded" / "report.json",
    "continuous_batched_pld_task_audit_test500_report.md": ROOT / "analysis" / "continuous_batched_pld_task_audit_test500" / "report.md",
    "continuous_batched_pld_task_audit_test500_report.json": ROOT / "analysis" / "continuous_batched_pld_task_audit_test500" / "report.json",
    "continuous_batched_pld_latency_report.md": ROOT / "analysis" / "continuous_batched_pld_latency" / "report.md",
    "continuous_batched_pld_latency_report.json": ROOT / "analysis" / "continuous_batched_pld_latency" / "report.json",
    "generic_batched_greedy_baseline_report.md": ROOT / "analysis" / "generic_batched_greedy_baseline" / "report.md",
    "generic_batched_greedy_baseline_report.json": ROOT / "analysis" / "generic_batched_greedy_baseline" / "report.json",
    "generic_batched_greedy_b8_pool_sweep_report.md": ROOT / "analysis" / "generic_batched_greedy_baseline" / "b8_pool_sweep_report.md",
    "generic_batched_greedy_b8_pool_sweep_report.json": ROOT / "analysis" / "generic_batched_greedy_baseline" / "b8_pool_sweep_report.json",
    "robustness_report.md": ROOT / "analysis" / "continuous_batched_pld_robustness" / "report.md",
    "robustness_report.json": ROOT / "analysis" / "continuous_batched_pld_robustness" / "report.json",
    "dataset_leakage_audit.md": ROOT / "artifacts" / "dataset_leakage_audit.md",
    "dataset_leakage_audit.json": ROOT / "artifacts" / "dataset_leakage_audit.json",
    "dataset_stats.md": ROOT / "artifacts" / "dataset_stats.md",
    "dataset_stats.json": ROOT / "artifacts" / "dataset_stats.json",
    "generation_stats.md": ROOT / "artifacts" / "generation_stats.md",
    "generation_stats.json": ROOT / "artifacts" / "generation_stats.json",
    "timing_path_drift_analysis.md": ROOT / "artifacts" / "timing_path_drift_analysis.md",
    "timing_path_drift_analysis.json": ROOT / "artifacts" / "timing_path_drift_analysis.json",
    "system_breakdown_analysis.md": ROOT / "artifacts" / "system_breakdown_analysis.md",
    "system_breakdown_analysis.json": ROOT / "artifacts" / "system_breakdown_analysis.json",
    "result_consistency_report.json": ROOT / "artifacts" / "result_consistency_report.json",
    "results_summary.md": ROOT / "artifacts" / "results_summary.md",
    "results_summary.json": ROOT / "artifacts" / "results_summary.json",
    "external_baseline_attempts.md": ROOT / "docs" / "external_baseline_attempts.md",
    "external_baselines_report.md": ROOT / "analysis" / "external_baselines" / "report.md",
    "external_baselines_report.json": ROOT / "analysis" / "external_baselines" / "report.json",
    "vllm_greedy_test500_report.md": ROOT / "analysis" / "external_baselines" / "external_baselines_l40s_test500_v1" / "vllm_greedy" / "report.md",
    "vllm_greedy_test500_report.json": ROOT / "analysis" / "external_baselines" / "external_baselines_l40s_test500_v1" / "vllm_greedy" / "report.json",
    "vllm_ngram_test500_report.md": ROOT / "analysis" / "external_baselines" / "external_baselines_l40s_test500_v2" / "vllm_ngram" / "report.md",
    "vllm_ngram_test500_report.json": ROOT / "analysis" / "external_baselines" / "external_baselines_l40s_test500_v2" / "vllm_ngram" / "report.json",
    "hf_prompt_lookup_test500_report.md": ROOT / "analysis" / "external_baselines" / "external_baselines_l40s_test500_v1" / "hf_prompt_lookup" / "report.md",
    "hf_prompt_lookup_test500_report.json": ROOT / "analysis" / "external_baselines" / "external_baselines_l40s_test500_v1" / "hf_prompt_lookup" / "report.json",
    "vllm_greedy_local_smoke_report.md": ROOT / "artifacts" / "external_baselines" / "vllm_greedy_local_smoke" / "report.md",
    "vllm_greedy_local_smoke_report.json": ROOT / "artifacts" / "external_baselines" / "vllm_greedy_local_smoke" / "report.json",
    "hf_prompt_lookup_local_cuda_smoke_report.md": ROOT / "artifacts" / "external_baselines" / "hf_prompt_lookup_local_cuda_smoke" / "report.md",
    "hf_prompt_lookup_local_cuda_smoke_report.json": ROOT / "artifacts" / "external_baselines" / "hf_prompt_lookup_local_cuda_smoke" / "report.json",
    "fp32_eager_subset_throughput_report.md": ROOT / "analysis" / "continuous_batched_pld_final_repeats" / "continuous_batched_pld_fp32_eager_throughput_test100_subset_v1" / "report.md",
    "fp32_eager_subset_throughput_report.json": ROOT / "analysis" / "continuous_batched_pld_final_repeats" / "continuous_batched_pld_fp32_eager_throughput_test100_subset_v1" / "report.json",
    "fp32_eager_sharded_test500_throughput_report.md": ROOT / "analysis" / "continuous_batched_pld_final_repeats" / "continuous_batched_pld_fp32_eager_throughput_test500_sharded_chunk512_v1" / "report.md",
    "fp32_eager_sharded_test500_throughput_report.json": ROOT / "analysis" / "continuous_batched_pld_final_repeats" / "continuous_batched_pld_fp32_eager_throughput_test500_sharded_chunk512_v1" / "report.json",
    "controlled_ablation_summary.md": ROOT / "analysis" / "batched_pld_controlled_ablation" / "controlled_ablation_test500_v1" / "summary.md",
    "controlled_ablation_summary.json": ROOT / "analysis" / "batched_pld_controlled_ablation" / "controlled_ablation_test500_v1" / "summary.json",
    "batched_pld_continuous_verification_report.md": ROOT / "analysis" / "batched_pld_continuous_verification_report.md",
    "final_decoder_ceiling_report.md": ROOT / "analysis" / "final_decoder_ceiling_report.md",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _copy_reports(out_dir: Path) -> dict[str, str]:
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for name, src in REPORT_SOURCES.items():
        if not src.exists():
            copied[name] = ""
            continue
        dst = reports_dir / name
        shutil.copy2(src, dst)
        copied[name] = str(dst.relative_to(out_dir))
    return copied


def _manifest(out_dir: Path, copied: dict[str, str]) -> dict[str, Any]:
    repeats = _load_json(ROOT / "analysis" / "continuous_batched_pld_final_repeats" / "report.json")
    correctness = _load_json(ROOT / "analysis" / "continuous_batched_pld_correctness_sharded" / "report.json")
    audit = _load_json(ROOT / "analysis" / "continuous_batched_pld_task_audit_test500" / "report.json")
    generic_report = ROOT / "analysis" / "generic_batched_greedy_baseline" / "report.json"
    generic_pool_sweep_report = ROOT / "analysis" / "generic_batched_greedy_baseline" / "b8_pool_sweep_report.json"
    robustness_report = ROOT / "analysis" / "continuous_batched_pld_robustness" / "report.json"
    controlled_ablation_report = (
        ROOT
        / "analysis"
        / "batched_pld_controlled_ablation"
        / "controlled_ablation_test500_v1"
        / "summary.json"
    )
    exact_sharded_report = (
        ROOT
        / "analysis"
        / "continuous_batched_pld_final_repeats"
        / "continuous_batched_pld_fp32_eager_throughput_test500_sharded_chunk512_v1"
        / "report.json"
    )
    external_baselines_report = ROOT / "analysis" / "external_baselines" / "report.json"
    controls: dict[str, Any] = {
        "generic_batched_greedy_baseline": {
            "available": generic_report.exists(),
            "report": copied.get("generic_batched_greedy_baseline_report.json", ""),
            "b8_pool_sweep_report": copied.get("generic_batched_greedy_b8_pool_sweep_report.json", ""),
        },
        "robustness_check": {
            "available": robustness_report.exists(),
            "report": copied.get("robustness_report.json", ""),
        },
        "controlled_ablation": {
            "available": controlled_ablation_report.exists(),
            "report": copied.get("controlled_ablation_summary.json", ""),
        },
        "fp32_eager_sharded_test500_throughput": {
            "available": exact_sharded_report.exists(),
            "report": copied.get("fp32_eager_sharded_test500_throughput_report.json", ""),
        },
        "external_baselines": {
            "available": external_baselines_report.exists(),
            "report": copied.get("external_baselines_report.json", ""),
        },
    }
    if generic_report.exists():
        generic = _load_json(generic_report)
        pool_sweep = _load_json(generic_pool_sweep_report) if generic_pool_sweep_report.exists() else {"rows": []}
        successful: list[dict[str, Any]] = [
            {
                "method": f"greedy_batched_b{row.get('batch_size')}_pool{row.get('active_pool_size')}",
                "tok_s": row.get("generated_tokens_per_sec"),
                "batch_size": row.get("batch_size"),
                "active_pool_size": row.get("active_pool_size"),
            }
            for row in generic.get("batched", [])
            if not row.get("error")
        ]
        successful.extend(
            {
                "method": f"greedy_batched_b{row.get('batch_size')}_pool{row.get('active_pool_size')}",
                "tok_s": row.get("tok_s"),
                "batch_size": row.get("batch_size"),
                "active_pool_size": row.get("active_pool_size"),
            }
            for row in pool_sweep.get("rows", [])
            if row.get("status") == "success"
        )
        best = max(successful, key=lambda row: float(row.get("tok_s", 0.0) or 0.0)) if successful else {}
        pool_status = {
            str(row.get("active_pool_size")): row.get("status")
            for row in pool_sweep.get("rows", [])
            if int(row.get("batch_size", 0)) == 8
        }
        best_tps = float(best.get("tok_s", 0.0) or 0.0)
        archived_batch_tps = 845.0
        controls["generic_batched_greedy_baseline"]["summary"] = {
            "sequential_tok_s": generic.get("sequential", {}).get("tokens_per_sec"),
            "best_successful_method": best.get("method"),
            "best_successful_batch": best.get("batch_size"),
            "best_successful_active_pool": best.get("active_pool_size"),
            "best_successful_tok_s": best_tps or None,
            "archived_batch_vs_best_generic_greedy_speedup": (
                archived_batch_tps / best_tps if best_tps else None
            ),
            "batch8_tok_s": next(
                (
                    row.get("generated_tokens_per_sec")
                    for row in generic.get("batched", [])
                    if int(row.get("batch_size", 0)) == 8
                ),
                None,
            ),
            "b8_pool8_status": pool_status.get("8", "missing"),
            "b8_pool16_status": pool_status.get("16", "missing"),
            "b8_pool32_status": pool_status.get("32", "missing"),
        }
    if robustness_report.exists():
        robust = _load_json(robustness_report)
        controls["robustness_check"]["summary_keys"] = sorted(robust.get("summary", {}).keys())
    if controlled_ablation_report.exists():
        controlled = _load_json(controlled_ablation_report)
        final_row = controlled.get("summary", {}).get("b8_pool32_default_continuous", {})
        seq_row = controlled.get("summary", {}).get("seq", {})
        controls["controlled_ablation"]["final_config"] = {
            "seq_tok_s_mean": seq_row.get("fields", {}).get("tok_s", {}).get("mean"),
            "b8_tok_s_mean": final_row.get("fields", {}).get("tok_s", {}).get("mean"),
            "b8_tok_s_std": final_row.get("fields", {}).get("tok_s", {}).get("std"),
            "b8_speedup": final_row.get("fields", {})
            .get("speedup_vs_same_run_sequential", {})
            .get("mean"),
            "b8_verifier_forwards": final_row.get("fields", {})
            .get("verifier_forwards", {})
            .get("mean"),
        }
    if exact_sharded_report.exists():
        exact = _load_json(exact_sharded_report)
        exact_summary = exact.get("summary", {})
        seq = exact_summary.get("blazedit_pld_w128_n10_b1", {})
        b8 = exact_summary.get("continuous_batched_pld_w128_n10_b8", {})
        controls["fp32_eager_sharded_test500_throughput"]["summary"] = {
            "sequential_tok_s": seq.get("tok_s", {}).get("mean"),
            "b8_tok_s": b8.get("tok_s", {}).get("mean"),
            "b8_speedup": b8.get("speedup", {}).get("mean"),
            "sequential_verifier_forwards": seq.get("verifier_forwards", {}).get("mean"),
            "b8_verifier_forwards": b8.get("verifier_forwards", {}).get("mean"),
            "b8_task_matches": b8.get("output_match_count", {}).get("mean"),
            "completed_shards": len(exact.get("completed_shards", [])),
            "failed_attempts": len(exact.get("failed_attempts", [])),
            "prefill_chunk_size": exact.get("args", {}).get("prefill_chunk_size"),
        }
    if external_baselines_report.exists():
        external = _load_json(external_baselines_report)
        controls["external_baselines"]["summary"] = {
            row.get("method"): {
                "tokens_per_sec": row.get("tokens_per_sec"),
                "total_new_tokens": row.get("total_new_tokens"),
                "artifact": row.get("artifact"),
            }
            for row in external.get("rows", [])
        }
    return {
        "artifact": "continuous_batched_pld_final",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git_commit": _git_commit(),
        "method_name": "continuous_batched_pld_w128_n10",
        "method_title": "Continuous Batched PLD Verification",
        "final_config": {
            "batch_size": 8,
            "active_pool_size": 32,
            "buckets": [8, 16, 32, 64, 128],
            "bucket_policy": "default",
            "refill_policy": "continuous",
            "draft": "w128_n10",
        },
        "baseline_method": "blazedit_pld_w128_n10",
        "model": "Qwen/Qwen2.5-Coder-7B",
        "dataset_split": "held-out real-commit test500",
        "hardware": "Modal L40S",
        "timing_settings": {
            "dtype": "bf16",
            "attention": "SDPA",
            "repeats": 3,
        },
        "correctness_settings": {
            "dtype": "fp32",
            "attention": "eager",
            "sharded": True,
            "n": 500,
        },
        "headline_numbers": {
            "sequential_pld": {
                "tok_s_mean": 492.1,
                "tok_s_std": 2.1,
                "speedup": 1.0,
                "verifier_forwards": 6443,
            },
            "continuous_batched_pld_b2": {
                "tok_s_mean": 617.5,
                "tok_s_std": 1.4,
                "speedup": 1.255,
                "verifier_forwards": 3519,
            },
            "continuous_batched_pld_b4": {
                "tok_s_mean": 768.4,
                "tok_s_std": 0.6,
                "speedup": 1.561,
                "verifier_forwards": 2097,
            },
            "continuous_batched_pld_b8": {
                "tok_s_mean": 845.0,
                "tok_s_std": 7.1,
                "speedup": 1.717,
                "verifier_forwards": 1456,
            },
        },
        "correctness": {
            "batch_1_exact": correctness["aggregate"]["batch_results"]["1"]["exact_token_id_matches"],
            "batch_4_exact": correctness["aggregate"]["batch_results"]["4"]["exact_token_id_matches"],
            "batch_8_exact": correctness["aggregate"]["batch_results"]["8"]["exact_token_id_matches"],
            "total_tasks": correctness["aggregate"]["total_tasks"],
            "skipped_tasks": correctness["aggregate"]["skipped_count"],
            "coverage_exact_once": correctness["aggregate"]["coverage"]["covers_all_tasks_exactly_once"],
        },
        "task_isolation_audit": {
            "emitted_tokens_audited": audit.get("emitted_tokens_audited"),
            "task_mixing_violations": audit.get("task_mixing_violations"),
            "cache_ownership_violations": audit.get("cache_ownership_violations"),
            "finished_task_violations": audit.get("finished_task_violations"),
            "scatter_gather_mismatches": audit.get("scatter_gather_mismatches"),
            "unverified_token_violations": audit.get("unverified_token_violations"),
        },
        "reviewer_controls": controls,
        "best_successful_generic_greedy_method": controls.get("generic_batched_greedy_baseline", {})
        .get("summary", {})
        .get("best_successful_method"),
        "best_successful_generic_greedy_tok_s": controls.get("generic_batched_greedy_baseline", {})
        .get("summary", {})
        .get("best_successful_tok_s"),
        "archived_batch_vs_best_generic_greedy_speedup": controls.get("generic_batched_greedy_baseline", {})
        .get("summary", {})
        .get("archived_batch_vs_best_generic_greedy_speedup"),
        "generic_greedy_b8_pool8_status": controls.get("generic_batched_greedy_baseline", {})
        .get("summary", {})
        .get("b8_pool8_status"),
        "generic_greedy_b8_pool16_status": controls.get("generic_batched_greedy_baseline", {})
        .get("summary", {})
        .get("b8_pool16_status"),
        "generic_greedy_b8_pool32_status": controls.get("generic_batched_greedy_baseline", {})
        .get("summary", {})
        .get("b8_pool32_status"),
        "source_reports": copied,
        "derived_artifacts": {
            "tables": [
                "tables/main_results.md",
                "tables/main_results.tex",
                "tables/generic_batching_comparison.md",
                "tables/generic_batching_comparison.tex",
                "tables/correctness.md",
                "tables/correctness.tex",
                "tables/ablation.md",
                "tables/ablation.tex",
                "tables/latency.md",
                "tables/latency.tex",
                "tables/timing_path_drift.md",
                "tables/timing_path_drift.tex",
                "tables/exactness_vs_timing_backend.md",
                "tables/exactness_vs_timing_backend.tex",
                "tables/exact_backend_throughput.md",
                "tables/exact_backend_throughput.tex",
                "tables/dataset_stats.md",
                "tables/dataset_stats.tex",
                "tables/generation_denominator.md",
                "tables/generation_denominator.tex",
                "tables/leakage_audit.md",
                "tables/leakage_audit.tex",
                "tables/external_baselines.md",
                "tables/external_baselines.tex",
                "tables/system_breakdown.md",
                "tables/system_breakdown.tex",
                "tables/negative_results_appendix.md",
                "tables/negative_results_appendix.tex",
            ],
            "figures": [
                "figures/speedup_by_batch_size.png",
                "figures/speedup_by_batch_size.pdf",
                "figures/verifier_forwards_reduction.png",
                "figures/verifier_forwards_reduction.pdf",
                "figures/throughput_vs_latency.png",
                "figures/throughput_vs_latency.pdf",
                "figures/refill_ablation.png",
                "figures/refill_ablation.pdf",
                "figures/negative_results_summary.png",
                "figures/negative_results_summary.pdf",
                "figures/generic_vs_pld_batching.png",
                "figures/generic_vs_pld_batching.pdf",
            ],
            "claims_ledger": [
                "claims_ledger.md",
                "claims_ledger.json",
            ],
            "reviewer_risk_qa": "reviewer_risk_qa.md",
            "validation_report": "validation_report.md",
            "summary": "paper_summary.md",
        },
        "command_lines": {
            "repeated_timing": (
                "modal run vantage_runtime_debian_app.py::launch_batched_pld_repeated_timing "
                "--split test --n 500 --repeats 3 --batch-sizes 2,4,8 "
                "--active-pool-size 32 --bucket-policy default --refill-policy continuous "
                "--version continuous_batched_pld_final_repeats_repro --wait"
            ),
            "sharded_correctness": (
                "modal run vantage_runtime_debian_app.py::launch_batched_pld_correctness_sharded "
                "--split test --n 500 --shard-size 50 --batch-sizes 1,4,8 "
                "--dtype fp32 --attn eager --active-pool-size 32 --bucket-policy default "
                "--refill-policy continuous "
                "--version continuous_batched_pld_fp32_eager_correctness_sharded_repro --wait"
            ),
            "fp32_eager_full_test500_attempt": (
                "modal run vantage_runtime_debian_app.py::launch_batched_pld_repeated_timing "
                "--split test --n 500 --repeats 3 --batch-sizes 2,4,8 --dtype fp32 --attn eager "
                "--active-pool-size 32 --bucket-policy default --refill-policy continuous "
                "--version continuous_batched_pld_fp32_eager_throughput_test500_v1 --no-write-audit-trace --wait"
            ),
            "fp32_eager_sharded_test500": (
                "modal run vantage_runtime_debian_app.py::launch_batched_pld_repeated_timing_sharded "
                "--split test --n 500 --shard-size 50 --repeats 1 --batch-sizes 2,4,8 "
                "--dtype fp32 --attn eager --active-pool-size 32 --bucket-policy default "
                "--refill-policy continuous --prefill-chunk-size 512 "
                "--version continuous_batched_pld_fp32_eager_throughput_test500_sharded_chunk512_v1 --wait"
            ),
            "fp32_eager_test100_subset": (
                "modal run vantage_runtime_debian_app.py::launch_batched_pld_repeated_timing "
                "--split test --n 100 --repeats 1 --batch-sizes 2,4,8 --dtype fp32 --attn eager "
                "--active-pool-size 32 --bucket-policy default --refill-policy continuous "
                "--version continuous_batched_pld_fp32_eager_throughput_test100_subset_v1 --no-write-audit-trace --wait"
            ),
            "controlled_ablation": (
                "modal run vantage_runtime_debian_app.py::launch_batched_pld_controlled_ablation "
                "--split test --n 500 --repeats 3 --dtype bf16 --attn sdpa "
                "--version controlled_ablation_test500_v1 --wait"
            ),
            "task_audit": (
                "python3 scripts/audit_batched_pld_task_isolation.py "
                "--trace <BATCHED_BATCH8_TEST500_TRACE_JSONL> "
                "--output-dir analysis/continuous_batched_pld_task_audit_test500_repro"
            ),
            "generic_batched_greedy": (
                "modal run vantage_runtime_debian_app.py::launch_batched_greedy_eval "
                "--split test --n 500 --batch-sizes 2,4,8 --active-pool-size 32 "
                "--refill-policy continuous --version generic_batched_greedy_test500_repro --wait"
            ),
            "generic_batched_greedy_b8_pool8": (
                "modal run vantage_runtime_debian_app.py::launch_batched_greedy_eval "
                "--split test --n 500 --batch-sizes 8 --active-pool-size 8 "
                "--refill-policy continuous --skip-sequential "
                "--version generic_batched_greedy_b8_pool8_test500_repro --wait"
            ),
            "generic_batched_greedy_b8_pool16": (
                "modal run vantage_runtime_debian_app.py::launch_batched_greedy_eval "
                "--split test --n 500 --batch-sizes 8 --active-pool-size 16 "
                "--refill-policy continuous --skip-sequential "
                "--version generic_batched_greedy_b8_pool16_test500_repro --wait"
            ),
            "generic_batched_greedy_correctness": (
                "modal run vantage_runtime_debian_app.py::launch_batched_greedy_correctness "
                "--split test --n 100 --batch-sizes 1,8 --dtype fp32 --attn eager "
                "--version generic_batched_greedy_fp32_eager_correctness_n100_repro --wait"
            ),
            "robustness_alt_split": (
                "modal run vantage_runtime_debian_app.py::launch_continuous_batched_pld_robustness "
                "--split train --n 500 --version continuous_batched_pld_robustness_alt_split_repro --wait"
            ),
            "tables": "python3 scripts/make_continuous_batched_pld_paper_tables.py",
            "generic_comparison_table": "python3 scripts/make_batched_vs_pld_comparison_table.py",
            "figures": "python3 scripts/make_continuous_batched_pld_paper_figures.py",
            "negative_results": "python3 scripts/make_negative_results_appendix.py",
            "result_consistency": "python3 scripts/check_result_consistency.py",
            "dataset_leakage_audit": "python3 scripts/audit_dataset_leakage.py",
            "dataset_stats": "python3 scripts/compute_dataset_stats.py",
        },
        "derived_from": {
            "repeats_config": repeats.get("config_name", ""),
            "repeats_method": repeats.get("method_name", ""),
        },
    }


def _readme() -> str:
    return """# Continuous Batched PLD Final Artifact

This directory freezes the submission artifact package for
`continuous_batched_pld_w128_n10`, the final Continuous Batched PLD Verification
method.

## Contents

- `manifest.json`: method/config metadata, headline numbers, commands, and report paths.
- `reports/`: copied source reports used for the final paper tables.
- `tables/`: generated Markdown and LaTeX tables.
- `figures/`: generated PNG/PDF figures.
- `paper_summary.md`: concise paper-ready summary.
- `claims_ledger.{md,json}`: paper claims mapped to exact evidence paths.
- `reviewer_risk_qa.md`: concise answers to expected reviewer objections.
- `validation_report.md`: local compile/test/build-status report.
- `tables/generic_batching_comparison.*`: reviewer control comparing generic
  continuous batching against Continuous Batched PLD.
- `tables/external_baselines.*`: GPU-backed vLLM/Hugging Face baseline table.

## Reproduce Timing

```bash
modal run vantage_runtime_debian_app.py::launch_batched_pld_repeated_timing \\
  --split test \\
  --n 500 \\
  --repeats 3 \\
  --batch-sizes 2,4,8 \\
  --active-pool-size 32 \\
  --bucket-policy default \\
  --refill-policy continuous \\
  --version continuous_batched_pld_final_repeats_repro \\
  --wait
```

## Reproduce Correctness

```bash
modal run vantage_runtime_debian_app.py::launch_batched_pld_correctness_sharded \\
  --split test \\
  --n 500 \\
  --shard-size 50 \\
  --batch-sizes 1,4,8 \\
  --dtype fp32 \\
  --attn eager \\
  --active-pool-size 32 \\
  --bucket-policy default \\
  --refill-policy continuous \\
  --version continuous_batched_pld_fp32_eager_correctness_sharded_repro \\
  --wait
```

## Reproduce Exact-Backend Throughput

```bash
modal run vantage_runtime_debian_app.py::launch_batched_pld_repeated_timing_sharded \\
  --split test \\
  --n 500 \\
  --shard-size 50 \\
  --repeats 1 \\
  --batch-sizes 2,4,8 \\
  --dtype fp32 \\
  --attn eager \\
  --active-pool-size 32 \\
  --bucket-policy default \\
  --refill-policy continuous \\
  --prefill-chunk-size 512 \\
  --version continuous_batched_pld_fp32_eager_throughput_test500_sharded_chunk512_repro \\
  --wait
```

## Reproduce Task Audit

```bash
python3 scripts/audit_batched_pld_task_isolation.py \\
  --trace <BATCHED_BATCH8_TEST500_TRACE_JSONL> \\
  --output-dir analysis/continuous_batched_pld_task_audit_test500_repro
```

## Reproduce Generic Batching Control

```bash
modal run vantage_runtime_debian_app.py::launch_batched_greedy_eval \\
  --split test \\
  --n 500 \\
  --batch-sizes 2,4,8 \\
  --active-pool-size 32 \\
  --refill-policy continuous \\
  --version generic_batched_greedy_test500_repro \\
  --wait

modal run vantage_runtime_debian_app.py::launch_batched_greedy_eval \\
  --split test \\
  --n 500 \\
  --batch-sizes 8 \\
  --active-pool-size 8 \\
  --refill-policy continuous \\
  --skip-sequential \\
  --version generic_batched_greedy_b8_pool8_test500_repro \\
  --wait

modal run vantage_runtime_debian_app.py::launch_batched_greedy_eval \\
  --split test \\
  --n 500 \\
  --batch-sizes 8 \\
  --active-pool-size 16 \\
  --refill-policy continuous \\
  --skip-sequential \\
  --version generic_batched_greedy_b8_pool16_test500_repro \\
  --wait
```

## Reproduce Controlled Scheduler Ablation

```bash
modal run vantage_runtime_debian_app.py::launch_batched_pld_controlled_ablation \\
  --split test \\
  --n 500 \\
  --repeats 3 \\
  --dtype bf16 \\
  --attn sdpa \\
  --version controlled_ablation_test500_v1 \\
  --wait
```

## Reproduce External Baselines

```bash
modal run vantage_runtime_debian_app.py::launch_external_baseline \\
  --baseline vllm_greedy \\
  --split test \\
  --n 500 \\
  --max-new-tokens 256 \\
  --dtype bf16 \\
  --max-model-len 12288 \\
  --gpu-memory-utilization 0.9 \\
  --version external_baselines_l40s_test500_repro \\
  --wait

modal run vantage_runtime_debian_app.py::launch_external_baseline \\
  --baseline vllm_ngram \\
  --split test \\
  --n 500 \\
  --max-new-tokens 256 \\
  --dtype bf16 \\
  --max-model-len 12288 \\
  --gpu-memory-utilization 0.9 \\
  --ngram-prompt-lookup-max 128 \\
  --num-speculative-tokens 8 \\
  --version external_baselines_l40s_test500_repro \\
  --wait

modal run vantage_runtime_debian_app.py::launch_external_baseline \\
  --baseline hf_prompt_lookup \\
  --split test \\
  --n 500 \\
  --max-new-tokens 256 \\
  --dtype bf16 \\
  --prompt-lookup-num-tokens 128 \\
  --version external_baselines_l40s_test500_repro \\
  --wait
```

## Reproduce Robustness Check

```bash
modal run vantage_runtime_debian_app.py::launch_continuous_batched_pld_robustness \\
  --split train \\
  --n 500 \\
  --version continuous_batched_pld_robustness_alt_split_repro \\
  --wait
```

## Regenerate Tables And Figures

```bash
python3 scripts/make_continuous_batched_pld_paper_tables.py
python3 scripts/make_batched_vs_pld_comparison_table.py
python3 scripts/make_continuous_batched_pld_paper_figures.py
python3 scripts/make_negative_results_appendix.py
```

## Reviewer Audit Commands

```bash
python3 scripts/check_result_consistency.py
python3 scripts/audit_dataset_leakage.py
python3 scripts/compute_dataset_stats.py
```
"""


def _paper_summary() -> str:
    return """# Paper Summary: Continuous Batched PLD Verification

## Motivation

Prompt lookup decoding (PLD) is already a strong baseline for code editing
because real commits contain substantial exact-copy structure. Multiple
PLD-adjacent draft-improvement branches failed to clear a 20% speedup target,
which shifted the investigation from finding better draft tokens to scheduling
the verifier more efficiently.

## Failed Draft-Improvement Attempts

The negative-result suite included exact candidate reranking, large-K exact
reranking oracles, fuzzy/delta variants, syntax-slot PLD, MTP and queued MTP,
weak-router capped PLD, selective LM-head verification, and diff/hunk-only
generation. These branches either lacked coverage, added too much overhead, or
attacked a cost component that was too small to move end-to-end throughput.

## Diagnosis: Verifier Scheduling Bottleneck

Verifier-length microbenchmarks showed that cached target verification has a
large fixed cost and a small incremental cost per draft token. This explains why
shortening weak PLD drafts did not help: it preserved most of the verifier cost
while often increasing decode steps. The remaining opportunity was therefore to
reduce verifier launches by batching verification across concurrent tasks.

## Method: Continuous Batched PLD Verification

Continuous Batched PLD Verification keeps the PLD draft rule and target-model
verification semantics unchanged. Each active task performs its own PLD lookup,
then tasks with compatible draft buckets are verified together in batched target
forwards. The final configuration is:

- method: `continuous_batched_pld_w128_n10`
- batch size: `8`
- active pool size: `32`
- buckets: `8,16,32,64,128`
- refill policy: `continuous`
- baseline: `blazedit_pld_w128_n10`

Every emitted token remains target-verified.

## Main Results

Continuous Batched PLD Verification improves aggregate throughput by 1.717x over
optimized BlazEdit PLD on held-out real-commit code-edit tasks, reducing verifier
forwards from 6443 to 1456 in the bf16/SDPA timing path. Deterministic
fp32/eager output equivalence is validated separately on 500/500 tasks; the
bf16/SDPA timing path is not reported as byte-exact.

| method | tok/s | speedup | verifier forwards |
|---|---:|---:|---:|
| sequential BlazEdit PLD | 492.1 ± 2.1 | 1.000x | 6443 |
| continuous batched PLD b2 | 617.5 ± 1.4 | 1.255x | 3519 |
| continuous batched PLD b4 | 768.4 ± 0.6 | 1.561x | 2097 |
| continuous batched PLD b8 | 845.0 ± 7.1 | 1.717x | 1456 |

Controlled scheduler ablation rerun on the same held-out test500 bf16/SDPA
protocol produced an independent b8/pool32/default/continuous result of
859.0 ± 4.7 tok/s, 1.746x versus its same-run sequential PLD baseline of
492.0 ± 2.2 tok/s. This ablation is used for pool/refill/bucket conclusions;
the locked headline table above remains the conservative headline result.

## Reviewer Controls

We include a generic continuous-batched greedy baseline to separate batching
effects from PLD-specific speculative verification. Generic batching should be
reported honestly: if it also improves aggregate throughput, the paper claim is
that PLD plus continuous batched verification improves the custom sequential
PLD path by reducing verifier forwards, not that batching alone is novel.

We also include one robustness check on the alternate real-commit train500 split
with the same model and final batch=8 configuration. This is intentionally a
single control run, not a new sweep.

Generic batching control on held-out test500:

| method | batch | tok/s | speedup vs greedy sequential | forwards | note |
|---|---:|---:|---:|---:|---|
| greedy sequential | 1 | 39.2 | 1.000x | 100339 | no PLD |
| batched greedy | 2 | 61.6 | 1.570x | 50435 | no PLD |
| batched greedy | 4/pool32 | 97.6 | 2.487x | 25233 | no PLD |
| batched greedy | 8/pool8 | 122.4 | 3.120x | 12683 | best successful generic row |
| batched greedy | 8/pool16 | OOM | OOM | OOM | L40S memory limit |
| batched greedy | 8/pool32 | OOM | OOM | OOM | L40S memory limit |

Custom generic continuous batching helps greedy decoding relative to its own
baseline, but it remains far below optimized sequential PLD and Continuous
Batched PLD in this custom harness. The best successful custom generic greedy
control is batch=8/pool=8 at 122.4 tok/s; archived continuous-batched PLD prototype reaches 845.0 tok/s,
about 6.9x faster on this benchmark. The final method's advantage is
PLD-specific: it reduces verifier forwards while batching verifier calls.

External generation baselines on held-out test500:

| method | tok/s | emitted tokens | note |
|---|---:|---:|---|
| vLLM greedy | 1996.4 | 100100 | generic serving engine, not PLD-equivalent |
| vLLM n-gram prompt lookup | 2671.3 | 100953 | fastest external baseline in this artifact |
| Hugging Face prompt lookup | 425.6 | 106439 | sequential HF generate baseline |

These results require the paper to scope the archived continuous-batched PLD prototype claim carefully:
vLLM n-gram is faster than the custom research harness, so archived continuous-batched PLD prototype is a
PLD-equivalent scheduler contribution and integration target rather than a
claim of production-serving superiority over vLLM.

Alternate-split robustness on real-commit train500:

| method | batch | tok/s | speedup | verifier forwards |
|---|---:|---:|---:|---:|
| sequential PLD | 1 | 504.8 | 1.000x | 6255 |
| continuous batched PLD | 8 | 875.2 | 1.734x | 1385 |

## Correctness Validation

Deterministic fp32/eager sharded held-out test500 validation matched sequential
PLD exactly:

- batch=1: 500/500 exact token-id matches
- batch=4: 500/500 exact token-id matches
- batch=8: 500/500 exact token-id matches
- decoded outputs: 500/500
- finish reasons: 500/500
- generated lengths: 500/500
- skipped tasks: 0

The full bf16/SDPA batch=8 task-isolation audit checked 100,780 emitted tokens
and found zero task-mixing, cache ownership, finished-task, scatter/gather, or
unverified-token violations.

The bf16/SDPA timing path matches same-run sequential PLD on 452/500 tasks for
batch=8. Current logs contain task-level match counts only; deeper per-token
drift attribution requires a rerun with detailed tracing. A full-test500
fp32/eager sharded timing artifact with 512-token chunked prompt prefill reaches
1.256x speedup with 500/500 exact task matches.

## Latency / Throughput Tradeoff

The method improves aggregate throughput, not necessarily single-request
latency. It requires multiple concurrent code-edit tasks to realize batching
gains. Batch=8 had 845.0 aggregate tok/s and p50 task latency of 3527.4 ms in
the available offline all-at-once trace.

## Limitations

- The method is an aggregate-throughput method.
- bf16/SDPA timing runs may show numerical-path drift; deterministic fp32/eager
  validation matches exactly.
- Full-test500 fp32/eager throughput is available only as a sharded artifact
  with 512-token chunked prompt prefill; it should not be mixed with the
  unsharded bf16/SDPA headline protocol.
- External vLLM and Hugging Face prompt-lookup baselines are reported. vLLM
  n-gram is faster than the custom research harness, so the contribution is
  scoped to PLD-equivalent scheduling over the custom sequential PLD baseline.
- The method requires concurrent tasks and enough GPU memory for batched cached
  verification.
- Batch=16 is not part of the claim.

## Reproducibility

Use `scripts/reproduce_continuous_batched_pld_final.sh` to rerun timing,
correctness, packaging, tables, figures, and appendix generation.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = _copy_reports(out_dir)
    (out_dir / "manifest.json").write_text(
        json.dumps(_manifest(out_dir, copied), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(_readme(), encoding="utf-8")
    (out_dir / "paper_summary.md").write_text(_paper_summary(), encoding="utf-8")
    print(f"wrote artifact manifest and README to {out_dir}")


if __name__ == "__main__":
    main()
