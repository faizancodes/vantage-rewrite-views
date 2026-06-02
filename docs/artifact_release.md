# VANTAGE Artifact Release

The GitHub repository is intentionally source-focused. It contains code, tests,
paper source, documentation, and the small locked controlled manifests required
to identify the headline workloads.

Large generated outputs are released through a Hugging Face dataset repository
instead of GitHub:

<https://huggingface.co/datasets/faizancodes/vantage-artifacts>

The helper scripts default to the dataset id `faizancodes/vantage-artifacts`.
The upload completed from the full local research tree on June 2, 2026.

## GitHub Contents

Keep these in Git:

- `asts/`
- `scripts/`
- `tests/`
- `docs/`
- `paper/`
- `vantage_vllm/`
- `patches/`
- `data/manifests_frozen_audit/`

The checked-in `data/manifests_frozen_audit/` directory is small and defines
the locked n=100 controlled workloads used by the primary paper claim.

## Hugging Face Dataset Payload

Generated or bulky data is stored in the Hugging Face dataset:

- `artifacts/vantage_transpld/`
- `artifacts/vantage_viewbank/`
- `artifacts/vllm_results/`
- `artifacts/vllm_tables/`
- `artifacts/vantage_residual/`
- `analysis/`
- `out/`
- `data/real_commits/`
- `data/manifests/`
- `data/manifests_frozen_audit_raw/`
- `data/manifests_phase2/`
- `data/manifests_phase3/`
- `data/manifests_prompt_injection/`
- `data/manifests_transpld_ext/`
- `data/routers/`

The public upload uses VANTAGE-facing artifact roots. Some paths intentionally
preserve the historical `transpld` artifact tag because generated run tags and
summarization scripts still use it as an internal Rewrite-View Lookup label.

## Upload

From a machine with a full research tree containing the VANTAGE-facing artifact
roots listed above and Hugging Face write credentials, inspect the upload plan:

```bash
python3 scripts/upload_vantage_data_to_hf.py \
  --source-root /path/to/full/research/asts-spec \
  --repo-id "$VANTAGE_HF_DATASET" \
  --dry-run
```

Inspect the plan, then run without `--dry-run`:

```bash
python3 scripts/upload_vantage_data_to_hf.py \
  --source-root /path/to/full/research/asts-spec \
  --repo-id "$VANTAGE_HF_DATASET"
```

The script requires either `HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN` or an existing
`huggingface-cli login`/`hf auth login` session.

## Download

```bash
python3 scripts/download_vantage_artifacts.py \
  --repo-id faizancodes/vantage-artifacts \
  --local-dir .
```

This restores the repository-relative artifact paths expected by paper
summarization scripts.
