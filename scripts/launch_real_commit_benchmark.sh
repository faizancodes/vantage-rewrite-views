#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${MANIFEST:-data/real_commits/real_commit_manifest.jsonl}"
RUN_TAG="${RUN_TAG:-vantage_real_commit_qwen_base_v1}"
METHODS="${METHODS:-vanilla,blazedit_pld_w128_n10,vantage_frozen_transpld}"
TARGET="${TARGET:-Qwen/Qwen2.5-Coder-7B}"
MODAL_FUNC_REF="${MODAL_FUNC_REF:-vantage_runtime_debian_app.py::run_eagle_eval_job_any}"
N="${N:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  exit 1
fi

if [[ "${N}" == "0" ]]; then
  N="$(wc -l < "${MANIFEST}" | tr -d ' ')"
fi

echo "Launching real-commit benchmark"
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
