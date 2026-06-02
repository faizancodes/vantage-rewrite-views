#!/usr/bin/env bash
set -euo pipefail

# Reproduce the final Continuous Batched PLD Verification artifact.
#
# Override these from the environment if needed.
DATA_DIR="${DATA_DIR:-analysis}"
OUTPUT_DIR="${OUTPUT_DIR:-analysis/final_paper_artifacts/continuous_batched_pld_final}"
RUN_TAG="${RUN_TAG:-repro}"
TRACE_PATH="${TRACE_PATH:-<BATCHED_BATCH8_TEST500_TRACE_JSONL>}"

echo "DATA_DIR=${DATA_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "RUN_TAG=${RUN_TAG}"

# 1. Repeated held-out test500 timing.
modal run vantage_runtime_debian_app.py::launch_batched_pld_repeated_timing \
  --split test \
  --n 500 \
  --repeats 3 \
  --batch-sizes 2,4,8 \
  --active-pool-size 32 \
  --bucket-policy default \
  --refill-policy continuous \
  --version "continuous_batched_pld_final_repeats_${RUN_TAG}" \
  --wait

# 2. Sharded fp32/eager deterministic correctness.
modal run vantage_runtime_debian_app.py::launch_batched_pld_correctness_sharded \
  --split test \
  --n 500 \
  --shard-size 50 \
  --batch-sizes 1,4,8 \
  --dtype fp32 \
  --attn eager \
  --active-pool-size 32 \
  --bucket-policy default \
  --refill-policy continuous \
  --version "continuous_batched_pld_fp32_eager_correctness_sharded_${RUN_TAG}" \
  --wait

# 2b. Exact-backend throughput. The historical unsharded fp32/eager command
# can OOM on L40S long-context prompts. The paper-facing full-test500 exact
# artifact uses independent shards plus 512-token chunked prompt prefill.
modal run vantage_runtime_debian_app.py::launch_batched_pld_repeated_timing_sharded \
  --split test \
  --n 500 \
  --shard-size 50 \
  --repeats 1 \
  --batch-sizes 2,4,8 \
  --dtype fp32 \
  --attn eager \
  --active-pool-size 32 \
  --bucket-policy default \
  --refill-policy continuous \
  --prefill-chunk-size 512 \
  --version "continuous_batched_pld_fp32_eager_throughput_test500_sharded_chunk512_${RUN_TAG}" \
  --wait

# Optional provenance check: unsharded exact-backend timing may still OOM.
modal run vantage_runtime_debian_app.py::launch_batched_pld_repeated_timing \
  --split test \
  --n 500 \
  --repeats 3 \
  --batch-sizes 2,4,8 \
  --dtype fp32 \
  --attn eager \
  --active-pool-size 32 \
  --bucket-policy default \
  --refill-policy continuous \
  --version "continuous_batched_pld_fp32_eager_throughput_test500_${RUN_TAG}" \
  --no-write-audit-trace \
  --wait || true

# 3. Task-isolation audit. Set TRACE_PATH to the batch=8 JSONL trace produced
# by the timing job if you want to regenerate this locally.
if [[ "${TRACE_PATH}" != "<BATCHED_BATCH8_TEST500_TRACE_JSONL>" ]]; then
  python3 scripts/audit_batched_pld_task_isolation.py \
    --trace "${TRACE_PATH}" \
    --output-dir "${DATA_DIR}/continuous_batched_pld_task_audit_test500_${RUN_TAG}"
else
  echo "Skipping task audit: set TRACE_PATH to a batch=8 audit trace JSONL to rerun it."
fi

# 4. Generic continuous-batched greedy reviewer control.
modal run vantage_runtime_debian_app.py::launch_batched_greedy_eval \
  --split test \
  --n 500 \
  --batch-sizes 2,4,8 \
  --active-pool-size 32 \
  --refill-policy continuous \
  --version "generic_batched_greedy_test500_${RUN_TAG}" \
  --wait

modal run vantage_runtime_debian_app.py::launch_batched_greedy_eval \
  --split test \
  --n 500 \
  --batch-sizes 8 \
  --active-pool-size 8 \
  --refill-policy continuous \
  --skip-sequential \
  --version "generic_batched_greedy_b8_pool8_test500_${RUN_TAG}" \
  --wait

modal run vantage_runtime_debian_app.py::launch_batched_greedy_eval \
  --split test \
  --n 500 \
  --batch-sizes 8 \
  --active-pool-size 16 \
  --refill-policy continuous \
  --skip-sequential \
  --version "generic_batched_greedy_b8_pool16_test500_${RUN_TAG}" \
  --wait

# 5. One alternate-split robustness check for the final PLD batching config.
modal run vantage_runtime_debian_app.py::launch_continuous_batched_pld_robustness \
  --split train \
  --n 500 \
  --version "continuous_batched_pld_robustness_alt_split_${RUN_TAG}" \
  --wait

# 5b. Controlled ablation rerun under the headline protocol.
modal run vantage_runtime_debian_app.py::launch_batched_pld_controlled_ablation \
  --split test \
  --n 500 \
  --repeats 3 \
  --dtype bf16 \
  --attn sdpa \
  --version "controlled_ablation_test500_${RUN_TAG}" \
  --wait

# 5c. Local external-baseline smoke attempts. These are not throughput
# baselines; they document local dependency/GPU availability.
python3 scripts/run_vllm_baseline_eval.py \
  --problem-jsonl data/real_commits/real_commit_manifest_balanced_1000_v2_test500.jsonl \
  --n 1 \
  --max-new-tokens 1 \
  --backend greedy \
  --output-dir "artifacts/external_baselines/vllm_greedy_local_smoke_${RUN_TAG}" || true

python3 scripts/run_hf_prompt_lookup_baseline.py \
  --problem-jsonl data/real_commits/real_commit_manifest_balanced_1000_v2_test500.jsonl \
  --n 1 \
  --max-new-tokens 1 \
  --device cuda \
  --output-dir "artifacts/external_baselines/hf_prompt_lookup_local_cuda_smoke_${RUN_TAG}" || true

# 6. Freeze/copy the final artifact manifest and source reports.
python3 scripts/package_continuous_batched_pld_final.py \
  --output-dir "${OUTPUT_DIR}"

# 7. Generate paper tables.
python3 scripts/make_continuous_batched_pld_paper_tables.py \
  --output-dir "${OUTPUT_DIR}/tables"
python3 scripts/make_batched_vs_pld_comparison_table.py \
  --output-dir "${OUTPUT_DIR}/tables"

# 8. Generate paper figures.
python3 scripts/make_continuous_batched_pld_paper_figures.py \
  --output-dir "${OUTPUT_DIR}/figures"

# 9. Generate the negative-results appendix.
python3 scripts/make_negative_results_appendix.py \
  --output-dir "${OUTPUT_DIR}/tables"
