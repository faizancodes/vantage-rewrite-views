# vLLM Source Inspection Notes

Date: 2026-05-14

This directory intentionally does not contain a vendored vLLM checkout.
Inspected source trees were sparse clones under `/tmp` only.

## Inspected refs

- `vllm-project/vllm` tag `v0.20.2`
  - commit `bc150f50299199599673614f80d12a196f377655`
  - relevant files inspected:
    - `vllm/config/speculative.py`
    - `vllm/v1/worker/gpu_model_runner.py`
    - `vllm/v1/spec_decode/ngram_proposer.py`
    - `vllm/v1/worker/gpu_input_batch.py`
    - `vllm/v1/request.py`
    - `vllm/v1/core/sched/output.py`
    - `vllm/v1/core/sched/scheduler.py`
    - `vllm/sampling_params.py`
- `vllm-project/vllm` tag `v0.21.0`
  - commit `9da56fd18b95ff1b11360e8400f7c41b126d190b`
  - same relevant files inspected
- `vllm-project/vllm` branch `main`
  - commit `f8848b2f2da56418480fa4be1a8a9adbe4b960b4`
  - same relevant files plus
    `vllm/v1/spec_decode/custom_class_proposer.py`

## Findings

`v0.20.2` and `v0.21.0`:

- `SpeculativeMethod` is a closed literal set and does not include
  `custom_class`.
- `GPUModelRunner` constructs drafters through explicit branches.
- CPU n-gram proposal shape is
  `propose(sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings=None)`.
- `InputBatch` already maintains `num_prompt_tokens`, but the n-gram proposer
  call does not pass it.
- `SamplingParams.extra_args` exists and can carry request-local metadata into
  `InputBatch.add_request(...)` if the internal patch reads it.

`main`:

- `custom_class` is present in `SpeculativeMethod`.
- `create_custom_proposer(vllm_config)` imports the class path from
  `speculative_config.model` and instantiates it with `VllmConfig`.
- `GPUModelRunner.propose_draft_token_ids(...)` calls the custom proposer with
  only `sampled_token_ids`, `num_tokens_no_spec`, `token_ids_cpu`, and
  `slot_mappings`.
- Because `num_prompt_tokens`, source ranges, and exclude ranges are not
  passed, the current custom-class API is a construction smoke path, not a PLD
  equivalence path.

## Non-vendoring rule

Keep future imported vLLM content limited to small patch fragments, skeleton
files, or inspection notes. Do not copy a full `vllm/` source tree into this
repo unless Agent 0 explicitly chooses a fork path.
