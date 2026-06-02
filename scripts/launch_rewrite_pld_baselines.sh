#!/usr/bin/env bash
set -euo pipefail

# Launch the rewrite-normalized PLD timing sweep in parallel on Modal.
# Run from research/asts-spec:
#   bash scripts/launch_rewrite_pld_baselines.sh

METHODS="${METHODS:-vanilla,blazedit_pld_w128_n10,vantage_pld_w128_n10,rewrite_pld_vref_w80_n10,rewrite_pld_vref_w128_n10,rewrite_pld_bidir_w128_n10,rewrite_pld_oracle_w128_n10,vantage_rewrite_anchor_pld_g32_a128_w40_n10,vantage_rewrite_anchor_pld_g64_a160_w80_n10}"
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

launch_eval vantage_rewrite_pld_renamepct_n100_v1 data/manifests_phase2/drift_axis_renamepct.jsonl 100
launch_eval vantage_rewrite_pld_span_n50_v1 data/manifests_phase2/drift_axis_span.jsonl 50
launch_eval vantage_rewrite_pld_editdist_n50_v1 data/manifests_phase2/drift_axis_editdist.jsonl 50
launch_eval vantage_rewrite_pld_hunks_n50_v1 data/manifests_phase2/drift_axis_hunks.jsonl 50
launch_eval vantage_rewrite_pld_field_n50_v1 data/manifests_phase3/drift_nonrename_field_rename.jsonl 50
launch_eval vantage_rewrite_pld_style_n50_v1 data/manifests_phase3/drift_nonrename_style_rewrite.jsonl 50

wait
echo "All Modal rewrite-PLD launch commands returned. See logs/vantage_rewrite_pld_*.log for app ids/status."
