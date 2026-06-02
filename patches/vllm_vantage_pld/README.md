# VANTAGE PLD vLLM Patch Skeleton

Date: 2026-05-14

This directory is a minimal patch design, not a vendored vLLM fork. It targets
vLLM `v0.20.2` first because that is the Modal version recorded in the local
artifacts. The same patch points still exist in `v0.21.0`. Upstream `main`
currently has a `custom_class` proposer hook, but that hook does not pass the
request prefix/source metadata needed for PLD equivalence.

## Files

- `vantage_pld_proposer.py`: skeleton for
  `vllm/v1/spec_decode/vantage_pld_proposer.py`.
- `vllm_0_20_2_diff_fragments.md`: exact vLLM files/functions to patch and
  small diff fragments.

No broad source copy is included here. Source inspection notes are in
`third_party/vllm_vantage/README.md`.

## Required vLLM Config

Use:

```python
speculative_config = {
    "method": "vantage_pld",
    "num_speculative_tokens": 128,
    "pld_window_tokens": 128,
    "pld_match_tokens": 10,
    "pld_min_match_tokens": None,
    "pld_require_prompt_metadata": True,
    "pld_stats_enabled": True,
    "pld_trace_path": None,
}
```

Config fields win over environment variables. Environment variables are a
fallback for custom-class experiments or partially patched trees:

- `VANTAGE_PLD_WINDOW_TOKENS`, default `128`
- `VANTAGE_PLD_MATCH_TOKENS`, default `10`
- `VANTAGE_PLD_MIN_MATCH_TOKENS`, default same as match tokens
- `VANTAGE_PLD_REQUIRE_PROMPT_METADATA`, default `true`
- `VANTAGE_PLD_STATS_ENABLED`, default `true`
- `VANTAGE_PLD_TRACE_PATH`, default unset
- `VANTAGE_PLD_METADATA_KEY`, default `vantage_pld`

Literal `w128_n10` equivalence requires `num_speculative_tokens >= 128`. If
the cap is lower, label the run `w128_n10_capK`.

## Required Request Metadata

The patched runner must pass at least `num_prompt_tokens` to
`VantagePLDProposer.propose(...)` so the proposer can split:

- `context_ids = origin[context_start:context_end]`
- `generated_ids = origin[prompt_token_count:num_tokens_no_spec]`

For source-only prompts, default `context_start=0` and
`context_end=prompt_token_count` are acceptable. For prompts that contain any
gold target/post-edit text, the harness must also pass explicit source and
exclusion metadata:

```python
SamplingParams(
    temperature=0.0,
    max_tokens=max_new_tokens,
    extra_args={
        "vantage_pld": {
            "context_start": 0,
            "context_end": source_token_end,
            "exclude_ranges": [[gold_start, gold_end]],
        },
    },
)
```

Coordinates are half-open token spans in the request-local `token_ids_cpu` /
`Request.all_token_ids` coordinate system.

## Exact Blocker

Do not claim VANTAGE/PLD equivalence if the vLLM source/API cannot expose
request prefix metadata to the proposer.

Minimum non-negotiable metadata is `num_prompt_tokens` for each request slot.
If prompts contain gold target text, the minimum expands to
`context_start`, `context_end`, and `exclude_ranges`. Without this metadata,
the proposer cannot split prompt from generated prefix or prevent target
leakage; a custom-class or n-gram run is then only a non-equivalent baseline.

## Stats Contract

The proposer skeleton exports:

- `pld_calls`
- `pld_skipped_empty_sample`
- `pld_skipped_max_model_len`
- `pld_metadata_missing`
- `pld_hits`
- `pld_misses`
- `pld_tokens_proposed`
- `pld_cap`
- `pld_cap_truncations`
- `pld_prompt_hits`
- `pld_generated_hits`
- `pld_last_match_length`
- `pld_last_source_start`
- `pld_match_len_histogram`
- `pld_draft_len_histogram`

Scheduler/rejection-sampler instrumentation, if added later, should also emit:

- `pld_tokens_accepted`
- `pld_acceptance_rate`
- `pld_accepted_length_histogram`
- `pld_rejected_tokens`

Per-task trace export should include `task_id`, `step`, `proposal_tokens`,
`proposal_match_len`, `proposal_source_start_token`,
`proposal_follow_start_token`, `proposal_query_start_token`,
`proposal_source_region`, `accepted_tokens`, `rejected`, and
`first_reject_index`.

## Minimal Verification

Do not run expensive GPU jobs for the patch-design phase. Verification should
start with:

1. CPU unit tests for the pure lookup rule against local
   `vantage_vllm.proposer.PromptLookupProposer`.
2. A fake `VllmConfig` unit test for `VantagePLDProposer.propose(...)`.
3. A config-validation smoke that `method="vantage_pld"` is accepted.
4. A tiny GPU smoke only after Agent 0 approves patch execution:
   `n=1`, `max_new_tokens=1`, `num_speculative_tokens=128`.
5. Equivalence gate before throughput claims: greedy and patched PLD output
   token IDs must match exactly for every task.
