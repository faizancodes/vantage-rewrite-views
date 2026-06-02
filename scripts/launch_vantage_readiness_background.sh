#!/usr/bin/env bash
set -euo pipefail

METHODS="${METHODS:-vanilla,blazedit_pld_w128_n10,vantage_transpld_w128_n10,vantage_routed_transpld_w128_n10}"
QWEN="${QWEN:-Qwen/Qwen2.5-Coder-7B}"
DEEPSEEK="${DEEPSEEK:-deepseek-ai/deepseek-coder-6.7b-base}"
MODAL_FUNC_REF="${MODAL_FUNC_REF:-vantage_runtime_debian_app.py::run_eagle_eval_job_any}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TAG_SUFFIX="${TAG_SUFFIX:-v3}"
SKIP_QWEN_ZERO="${SKIP_QWEN_ZERO:-0}"

mkdir -p logs/modal_launch

launch_eval() {
  local tag="$1"
  local target="$2"
  local manifest="$3"
  local n="$4"
  shift 4
  local log="logs/modal_launch/${tag}.log"
  echo "Launching ${tag}; log=${log}"
  nohup modal run -d "${MODAL_FUNC_REF}" \
    --run-tag "${tag}" \
    --target "${target}" \
    --n "${n}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --methods "${METHODS}" \
    --problem-jsonl "${manifest}" \
    --dtype bfloat16 \
    --attn-impl sdpa \
    --code-proposer-fallback root \
    "$@" >"${log}" 2>&1 &
  echo "$!" >"logs/modal_launch/${tag}.pid"
  sleep 0.5
}

echo "Using Modal function: ${MODAL_FUNC_REF}"
echo "Methods: ${METHODS}"
echo "Tag suffix: ${TAG_SUFFIX}"

if [[ "${SKIP_QWEN_ZERO}" != "1" ]]; then
  launch_eval "vantage_routed_qwen_zerodrift50_${TAG_SUFFIX}" "${QWEN}" /root/asts-spec/data/manifests_transpld_ext/drift_renamepct_0.jsonl 50
fi
launch_eval "vantage_routed_qwen_span150_${TAG_SUFFIX}" "${QWEN}" /root/asts-spec/data/manifests_phase2/drift_axis_span.jsonl 150
launch_eval "vantage_routed_qwen_editdist150_${TAG_SUFFIX}" "${QWEN}" /root/asts-spec/data/manifests_phase2/drift_axis_editdist.jsonl 150
launch_eval "vantage_routed_qwen_hunks200_${TAG_SUFFIX}" "${QWEN}" /root/asts-spec/data/manifests_phase2/drift_axis_hunks.jsonl 200
launch_eval "vantage_routed_qwen_field50_${TAG_SUFFIX}" "${QWEN}" /root/asts-spec/data/manifests_phase3/drift_nonrename_field_rename.jsonl 50
launch_eval "vantage_routed_qwen_style50_${TAG_SUFFIX}" "${QWEN}" /root/asts-spec/data/manifests_phase3/drift_nonrename_style_rewrite.jsonl 50
launch_eval "vantage_routed_qwen_mixed250_${TAG_SUFFIX}" "${QWEN}" /root/asts-spec/data/manifests_phase2/drift_axis_renamepct.jsonl 250

launch_eval "vantage_routed_deepseek_zerodrift50_${TAG_SUFFIX}" "${DEEPSEEK}" /root/asts-spec/data/manifests_transpld_ext/drift_renamepct_0.jsonl 50 --target-trust-remote-code
launch_eval "vantage_routed_deepseek_span150_${TAG_SUFFIX}" "${DEEPSEEK}" /root/asts-spec/data/manifests_phase2/drift_axis_span.jsonl 150 --target-trust-remote-code
launch_eval "vantage_routed_deepseek_editdist150_${TAG_SUFFIX}" "${DEEPSEEK}" /root/asts-spec/data/manifests_phase2/drift_axis_editdist.jsonl 150 --target-trust-remote-code
launch_eval "vantage_routed_deepseek_hunks200_${TAG_SUFFIX}" "${DEEPSEEK}" /root/asts-spec/data/manifests_phase2/drift_axis_hunks.jsonl 200 --target-trust-remote-code
launch_eval "vantage_routed_deepseek_field50_${TAG_SUFFIX}" "${DEEPSEEK}" /root/asts-spec/data/manifests_phase3/drift_nonrename_field_rename.jsonl 50 --target-trust-remote-code
launch_eval "vantage_routed_deepseek_style50_${TAG_SUFFIX}" "${DEEPSEEK}" /root/asts-spec/data/manifests_phase3/drift_nonrename_style_rewrite.jsonl 50 --target-trust-remote-code
