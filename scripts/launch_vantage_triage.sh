#!/usr/bin/env bash
set -euo pipefail

# Launch the VANTAGE triage wave in parallel on Modal.
# Run from research/asts-spec:
#   bash scripts/launch_vantage_triage.sh

METHODS="vanilla,blazedit_pld_w80_n10,blazedit_pld_w128_n10,vantage_pld_w128_n10,vantage_rewrite_anchor_pld_g32_a128_w40_n10,vantage_rewrite_anchor_pld_g64_a160_w80_n10"
TARGET="${TARGET:-Qwen/Qwen2.5-Coder-7B}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
COMMON_FLAGS=(
  --target "$TARGET"
  --methods "$METHODS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --dtype bfloat16
  --attn-impl sdpa
  --skip-eagle-load
  --code-proposer-fallback root
)

launch_eval() {
  local tag="$1"
  local manifest="$2"
  local n="$3"
  echo "Launching $tag ($manifest, n=$n)"
  modal run --detach proto_app.py::run_eagle_eval \
    --run-tag "$tag" \
    --language python \
    --problem-jsonl "$manifest" \
    --n "$n" \
    "${COMMON_FLAGS[@]}" \
    > "logs/${tag}.log" 2>&1 &
}

mkdir -p logs

launch_eval vantage_drift_axis_renamepct_triage_v1 data/manifests/drift_axis_renamepct.jsonl 100
launch_eval vantage_drift_axis_identcount_triage_v1 data/manifests/drift_axis_identcount.jsonl 80
launch_eval vantage_drift_axis_hunks_triage_v1 data/manifests/drift_axis_hunks.jsonl 80
launch_eval vantage_drift_axis_span_triage_v1 data/manifests/drift_axis_span.jsonl 60
launch_eval vantage_drift_axis_editdist_triage_v1 data/manifests/drift_axis_editdist.jsonl 60
launch_eval vantage_drift_nonrename_triage_v1 data/manifests/drift_nonrename.jsonl 100
launch_eval vantage_codeeditor_translate80_v1 data/manifests/codeeditor_translate80.jsonl 80
launch_eval vantage_codeeditor_polish80_v1 data/manifests/codeeditor_polish80.jsonl 80
launch_eval vantage_prompt_oracle_triage_v1 data/manifests/prompt_oracle_selected.jsonl 60

wait
echo "All Modal triage launch commands returned. See logs/*.log for app ids/status."
