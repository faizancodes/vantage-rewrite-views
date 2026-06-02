---
license: other
pretty_name: VANTAGE generated artifacts
task_categories:
- text-generation
language:
- code
tags:
- speculative-decoding
- code-generation
- reproducibility
---

# VANTAGE Generated Artifacts

This dataset stores generated artifacts for:

**VANTAGE: Hidden Rewrite Views for Fixed-Prompt Speculative Code-Edit Decoding**

The companion code repository is expected to be:

```text
https://github.com/faizancodes/vantage-rewrite-views
```

## Contents

The dataset preserves repository-relative paths used by the paper and
summarization scripts. Public upload paths use VANTAGE-facing prefixes:

- `artifacts/vantage_transpld/`
- `artifacts/vantage_viewbank/`
- `artifacts/vllm_results/`
- `artifacts/vllm_tables/`
- `artifacts/vantage_residual/`
- `analysis/`
- `out/`
- selected large `data/` subdirectories

The public upload uses VANTAGE-facing artifact roots. Some paths retain
`transpld`, the old internal artifact tag for Rewrite-View Lookup, because
generated run tags and summarization scripts still use that label. The public
method name is VANTAGE and the core primitive is Rewrite-View Lookup.

## Scope

These artifacts support a narrow fixed-prompt mechanism study. They should not
be interpreted as evidence of production-serving readiness, broad real-world
code-edit acceleration, broad edit quality, multi-model universality, stochastic
sampling preservation, or exact bf16 behavior.

## Restore

From the companion repository:

```bash
python3 scripts/download_vantage_artifacts.py \
  --repo-id faizancodes/vantage-artifacts \
  --local-dir .
```

This restores the artifact paths expected by the paper table-generation
scripts.
