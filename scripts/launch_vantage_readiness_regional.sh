#!/usr/bin/env bash
set -euo pipefail

# Launch policy after the 2026-05-07 AWS us-east-1 AZ incident:
# 1. Run one tiny regional smoke first.
# 2. Only launch full jobs if the smoke completes.
# 3. Default away from AWS us-east; override MODAL_FUNC_REF if needed.

METHODS="${METHODS:-vanilla,blazedit_pld_w128_n10,vantage_transpld_w128_n10,vantage_routed_transpld_w128_n10}"
QWEN="${QWEN:-Qwen/Qwen2.5-Coder-7B}"
DEEPSEEK="${DEEPSEEK:-deepseek-ai/deepseek-coder-6.7b-base}"
MODAL_FUNC_REF="${MODAL_FUNC_REF:-vantage_runtime_app.py::run_eagle_eval_job_aws_west}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

modal_eval() {
  local tag="$1"
  local target="$2"
  local manifest="$3"
  local n="$4"
  shift 4
  modal run "${MODAL_FUNC_REF}" \
    --run-tag "${tag}" \
    --target "${target}" \
    --n "${n}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --methods "${METHODS}" \
    --problem-jsonl "${manifest}" \
    --dtype bfloat16 \
    --attn-impl sdpa \
    --code-proposer-fallback root \
    "$@"
}

echo "Using Modal function: ${MODAL_FUNC_REF}"
echo "Running regional smoke. Full launch will not proceed unless this completes."
modal_eval \
  vantage_routed_regional_smoke_v1 \
  "${QWEN}" \
  /root/asts-spec/data/manifests_transpld_ext/drift_renamepct_0.jsonl \
  1 \
  --max-new-tokens 8

if [[ "${SMOKE_ONLY:-0}" == "1" ]]; then
  echo "SMOKE_ONLY=1, stopping after smoke."
  exit 0
fi

modal_eval vantage_routed_qwen_zerodrift50_v2 "${QWEN}" /root/asts-spec/data/manifests_transpld_ext/drift_renamepct_0.jsonl 50
modal_eval vantage_routed_qwen_span150_v2 "${QWEN}" /root/asts-spec/data/manifests_phase2/drift_axis_span.jsonl 150
modal_eval vantage_routed_qwen_editdist150_v2 "${QWEN}" /root/asts-spec/data/manifests_phase2/drift_axis_editdist.jsonl 150
modal_eval vantage_routed_qwen_hunks200_v2 "${QWEN}" /root/asts-spec/data/manifests_phase2/drift_axis_hunks.jsonl 200
modal_eval vantage_routed_qwen_field50_v2 "${QWEN}" /root/asts-spec/data/manifests_phase3/drift_nonrename_field_rename.jsonl 50
modal_eval vantage_routed_qwen_style50_v2 "${QWEN}" /root/asts-spec/data/manifests_phase3/drift_nonrename_style_rewrite.jsonl 50
modal_eval vantage_routed_qwen_mixed250_v2 "${QWEN}" /root/asts-spec/data/manifests_phase2/drift_axis_renamepct.jsonl 250

modal_eval vantage_routed_deepseek_zerodrift50_v2 "${DEEPSEEK}" /root/asts-spec/data/manifests_transpld_ext/drift_renamepct_0.jsonl 50 --target-trust-remote-code
modal_eval vantage_routed_deepseek_span150_v2 "${DEEPSEEK}" /root/asts-spec/data/manifests_phase2/drift_axis_span.jsonl 150 --target-trust-remote-code
modal_eval vantage_routed_deepseek_editdist150_v2 "${DEEPSEEK}" /root/asts-spec/data/manifests_phase2/drift_axis_editdist.jsonl 150 --target-trust-remote-code
modal_eval vantage_routed_deepseek_hunks200_v2 "${DEEPSEEK}" /root/asts-spec/data/manifests_phase2/drift_axis_hunks.jsonl 200 --target-trust-remote-code
modal_eval vantage_routed_deepseek_field50_v2 "${DEEPSEEK}" /root/asts-spec/data/manifests_phase3/drift_nonrename_field_rename.jsonl 50 --target-trust-remote-code
modal_eval vantage_routed_deepseek_style50_v2 "${DEEPSEEK}" /root/asts-spec/data/manifests_phase3/drift_nonrename_style_rewrite.jsonl 50 --target-trust-remote-code
