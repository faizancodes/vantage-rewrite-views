#!/usr/bin/env bash
set -euo pipefail

SPLIT="${SPLIT:-train}"
MANIFEST="${MANIFEST:-data/real_commits/path_a_${SPLIT}500_v1.jsonl}"
RUN_TAG="${RUN_TAG:-vantage_real_commit_path_a_${SPLIT}500_v1}"
TARGET="${TARGET:-Qwen/Qwen2.5-Coder-7B}"
MODAL_FUNC_REF="${MODAL_FUNC_REF:-vantage_runtime_debian_app.py::run_eagle_eval_job_any}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  exit 1
fi

METHODS="${METHODS:-blazedit_pld_w128_n10,vantage_lazy_transpld_s16_m16_z1_w128_n10,vantage_lazy_transpld_s16_m32_z1_w128_n10,vantage_lazy_transpld_s16_m64_z1_w128_n10,vantage_lazy_transpld_s32_m16_z1_w128_n10,vantage_lazy_transpld_s32_m32_z1_w128_n10,vantage_lazy_transpld_s32_m64_z1_w128_n10,vantage_lazy_transpld_s64_m16_z1_w128_n10,vantage_lazy_transpld_s64_m32_z1_w128_n10,vantage_lazy_transpld_s64_m64_z1_w128_n10}"

N="$(wc -l < "${MANIFEST}" | tr -d ' ')"

echo "Launching Path A real-commit ${SPLIT} sweep"
echo "  run tag: ${RUN_TAG}"
echo "  manifest: ${MANIFEST}"
echo "  tasks: ${N}"
echo "  methods: ${METHODS}"

modal run -d "${MODAL_FUNC_REF}" \
  --run-tag "${RUN_TAG}" \
  --target "${TARGET}" \
  --n "${N}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --methods "${METHODS}" \
  --problem-jsonl "/root/asts-spec/${MANIFEST}" \
  --dtype bfloat16 \
  --attn-impl sdpa \
  --code-proposer-fallback root
