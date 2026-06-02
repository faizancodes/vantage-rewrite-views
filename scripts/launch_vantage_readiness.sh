#!/usr/bin/env bash
set -euo pipefail

METHODS="vanilla,blazedit_pld_w128_n10,vantage_transpld_w128_n10,vantage_routed_transpld_w128_n10"
QWEN="Qwen/Qwen2.5-Coder-7B"
DEEPSEEK="deepseek-ai/deepseek-coder-6.7b-base"
COMMON_ARGS=(
  --max-new-tokens 256
  --methods "${METHODS}"
  --skip-eagle-load
  --code-proposer-fallback root
  --dtype bfloat16
  --attn-impl sdpa
)

run_eval() {
  local tag="$1"
  local target="$2"
  local manifest="$3"
  local n="$4"
  shift 4
  modal run proto_app.py::run_eagle_eval \
    --run-tag "${tag}" \
    --target "${target}" \
    --n "${n}" \
    --problem-jsonl "${manifest}" \
    "${COMMON_ARGS[@]}" \
    "$@"
}

# Phase 1: Qwen zero-drift fix and explicit-drift preservation.
run_eval vantage_routed_qwen_zerodrift50_v1 "${QWEN}" /root/asts-spec/data/manifests_transpld_ext/drift_renamepct_0.jsonl 50
run_eval vantage_routed_qwen_span150_v1 "${QWEN}" /root/asts-spec/data/manifests_phase2/drift_axis_span.jsonl 150
run_eval vantage_routed_qwen_editdist150_v1 "${QWEN}" /root/asts-spec/data/manifests_phase2/drift_axis_editdist.jsonl 150
run_eval vantage_routed_qwen_hunks200_v1 "${QWEN}" /root/asts-spec/data/manifests_phase2/drift_axis_hunks.jsonl 200
run_eval vantage_routed_qwen_field50_v1 "${QWEN}" /root/asts-spec/data/manifests_phase3/drift_nonrename_field_rename.jsonl 50
run_eval vantage_routed_qwen_style50_v1 "${QWEN}" /root/asts-spec/data/manifests_phase3/drift_nonrename_style_rewrite.jsonl 50
run_eval vantage_routed_qwen_mixed250_v1 "${QWEN}" /root/asts-spec/data/manifests_phase2/drift_axis_renamepct.jsonl 250

# Phase 2: second-model check. If DeepSeek fails to load, rerun with
# --target bigcode/starcoder2-7b and remove --target-trust-remote-code.
run_eval vantage_routed_deepseek_zerodrift50_v1 "${DEEPSEEK}" /root/asts-spec/data/manifests_transpld_ext/drift_renamepct_0.jsonl 50 --target-trust-remote-code
run_eval vantage_routed_deepseek_span150_v1 "${DEEPSEEK}" /root/asts-spec/data/manifests_phase2/drift_axis_span.jsonl 150 --target-trust-remote-code
run_eval vantage_routed_deepseek_editdist150_v1 "${DEEPSEEK}" /root/asts-spec/data/manifests_phase2/drift_axis_editdist.jsonl 150 --target-trust-remote-code
run_eval vantage_routed_deepseek_hunks200_v1 "${DEEPSEEK}" /root/asts-spec/data/manifests_phase2/drift_axis_hunks.jsonl 200 --target-trust-remote-code
run_eval vantage_routed_deepseek_field50_v1 "${DEEPSEEK}" /root/asts-spec/data/manifests_phase3/drift_nonrename_field_rename.jsonl 50 --target-trust-remote-code
run_eval vantage_routed_deepseek_style50_v1 "${DEEPSEEK}" /root/asts-spec/data/manifests_phase3/drift_nonrename_style_rewrite.jsonl 50 --target-trust-remote-code
