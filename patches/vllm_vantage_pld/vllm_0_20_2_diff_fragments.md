# vLLM 0.20.2 `vantage_pld` Diff Fragments

These are targeted fragments for a minimal internal vLLM patch. They are not a
vendored source copy.

Source inspected:

- vLLM `v0.20.2`, tag commit `bc150f50299199599673614f80d12a196f377655`
- vLLM `v0.21.0`, tag commit `9da56fd18b95ff1b11360e8400f7c41b126d190b`
- vLLM `main`, commit `f8848b2f2da56418480fa4be1a8a9adbe4b960b4`

`v0.20.2` and `v0.21.0` do not have `custom_class`. `main` has
`custom_class`, but its `GPUModelRunner.propose_draft_token_ids` branch passes
only the n-gram-style arguments, not `num_prompt_tokens` or source/exclude
metadata.

## `vllm/config/speculative.py`

Patch locations in `v0.20.2`:

- `SpeculativeMethod` literal at lines 57-65
- `SpeculativeConfig` fields near the n-gram fields at lines 131-136
- `SpeculativeConfig.__post_init__` method inference and model assignment at
  lines 439-486
- n-gram/suffix setup branch at lines 488-526

Fragment:

```diff
@@
 SpeculativeMethod = Literal[
     "ngram",
     "medusa",
     "mlp_speculator",
     "draft_model",
     "suffix",
+    "vantage_pld",
     EagleModelTypes,
     NgramGPUTypes,
 ]
@@
     prompt_lookup_min: int | None = Field(default=None, ge=1)
     """Minimum size of ngram token window when using Ngram proposer, if
     provided. Defaults to 1."""
+
+    # VANTAGE prompt lookup decoding configuration.
+    pld_window_tokens: int = Field(default=128, ge=1)
+    pld_match_tokens: int = Field(default=10, ge=1)
+    pld_min_match_tokens: int | None = Field(default=None, ge=1)
+    pld_require_prompt_metadata: bool = True
+    pld_stats_enabled: bool = True
+    pld_trace_path: str | None = None
@@
             elif self.method == "ngram_gpu":
                 self.model = "ngram_gpu"
             elif self.method == "suffix":
                 self.model = "suffix"
+            elif self.method == "vantage_pld":
+                self.model = "vantage_pld"
             elif self.method == "extract_hidden_states":
                 self.model = "extract_hidden_states"
@@
        if self.method in ("ngram", "[ngram]"):
            self.method = "ngram"
 
         if self.method in ("ngram", "ngram_gpu"):
@@
             self.draft_model_config = self.target_model_config
             self.draft_parallel_config = self.target_parallel_config
+        elif self.method == "vantage_pld":
+            if self.pld_min_match_tokens is None:
+                self.pld_min_match_tokens = self.pld_match_tokens
+            if self.pld_min_match_tokens > self.pld_match_tokens:
+                raise ValueError(
+                    "pld_min_match_tokens must be <= pld_match_tokens"
+                )
+            if self.num_speculative_tokens < self.pld_window_tokens:
+                logger.warning(
+                    "vantage_pld is capped: num_speculative_tokens=%d < "
+                    "pld_window_tokens=%d; label this run as capped.",
+                    self.num_speculative_tokens,
+                    self.pld_window_tokens,
+                )
+            self.prompt_lookup_min = 0
+            self.prompt_lookup_max = 0
+            self.draft_model_config = self.target_model_config
+            self.draft_parallel_config = self.target_parallel_config
         elif self.method == "suffix":
```

## `vllm/v1/spec_decode/vantage_pld_proposer.py`

Add a new file. Use `patches/vllm_vantage_pld/vantage_pld_proposer.py` as
the starting skeleton.

Required public shape:

```python
class VantagePLDProposer:
    def __init__(self, vllm_config): ...

    def propose(
        self,
        sampled_token_ids,
        num_tokens_no_spec,
        token_ids_cpu,
        *,
        num_prompt_tokens,
        pld_context_start=None,
        pld_context_end=None,
        pld_exclude_ranges=None,
        slot_mappings=None,
    ) -> list[list[int]]: ...

    def load_model(self, *args, **kwargs): ...
```

The important difference from `NgramProposer` is that `vantage_pld` must use
`generated_ids = token_ids_cpu[i, prompt_len:num_tokens_no_spec[i]]` as the
query suffix source and must restrict searchable prompt tokens to the
request-local source span.

## `vllm/v1/worker/gpu_model_runner.py`

Patch locations in `v0.20.2`:

- type union and drafter setup around lines 517-570
- `GPUModelRunner.propose_draft_token_ids(...)` around lines 4506-4532

Drafter setup fragment:

```diff
@@
             if self.speculative_config.method == "ngram":
                 from vllm.v1.spec_decode.ngram_proposer import NgramProposer
 
                 self.drafter = NgramProposer(self.vllm_config)
+            elif self.speculative_config.method == "vantage_pld":
+                from vllm.v1.spec_decode.vantage_pld_proposer import (
+                    VantagePLDProposer,
+                )
+
+                self.drafter = VantagePLDProposer(self.vllm_config)
             elif self.speculative_config.uses_draft_model():
```

Proposal branch fragment:

```diff
@@
         if spec_config.method == "ngram":
             from vllm.v1.spec_decode.ngram_proposer import NgramProposer
@@
                 slot_mappings=slot_mappings,
             )
+        elif spec_config.method == "vantage_pld":
+            from vllm.v1.spec_decode.vantage_pld_proposer import (
+                VantagePLDProposer,
+            )
+
+            assert isinstance(sampled_token_ids, list)
+            assert isinstance(self.drafter, VantagePLDProposer)
+            draft_token_ids = self.drafter.propose(
+                sampled_token_ids,
+                self.input_batch.num_tokens_no_spec,
+                self.input_batch.token_ids_cpu,
+                num_prompt_tokens=self.input_batch.num_prompt_tokens,
+                pld_context_start=self.input_batch.pld_context_start,
+                pld_context_end=self.input_batch.pld_context_end,
+                pld_exclude_ranges=self.input_batch.pld_exclude_ranges,
+                slot_mappings=slot_mappings,
+            )
         elif spec_config.use_ngram_gpu():
```

## `vllm/v1/worker/gpu_input_batch.py`

Patch locations in `v0.20.2`:

- imports near file top
- `InputBatch.__init__` speculative-decoding fields near lines 220-224
- `InputBatch.add_request(...)` after `num_prompt_tokens` is known at
  lines 338-355
- `InputBatch.remove_request(...)` cleanup around lines 489-543
- `InputBatch.swap_states(...)` around lines 545-620
- `InputBatch.condense(...)` movement around lines 662-785

The minimal metadata storage can be numpy arrays plus a Python list for ranges:

```diff
@@
+import os
 from dataclasses import dataclass
 from typing import cast
@@
         self.num_accepted_tokens_cpu = self.num_accepted_tokens_cpu_tensor.numpy()
+        self.pld_context_start = np.zeros((max_num_reqs,), dtype=np.int32)
+        self.pld_context_end = np.zeros((max_num_reqs,), dtype=np.int32)
+        self.pld_exclude_ranges: list[list[tuple[int, int]]] = [
+            [] for _ in range(max_num_reqs)
+        ]
```

In `add_request`, parse `SamplingParams.extra_args["vantage_pld"]`. Defaults
are source-only prompt defaults:

```python
metadata = {}
if request.sampling_params and request.sampling_params.extra_args:
    metadata_key = os.environ.get("VANTAGE_PLD_METADATA_KEY", "vantage_pld")
    metadata = request.sampling_params.extra_args.get(metadata_key, {}) or {}
self.pld_context_start[req_index] = int(metadata.get("context_start", 0))
self.pld_context_end[req_index] = int(
    metadata.get("context_end", num_prompt_tokens)
)
self.pld_exclude_ranges[req_index] = [
    (int(start), int(end))
    for start, end in metadata.get("exclude_ranges", ())
]
```

Also move/swap/clear this state anywhere request slot state moves:

- `remove_request`: reset `pld_context_start/end[req_index]` and clear
  `pld_exclude_ranges[req_index]`
- `swap_states`: swap both arrays and list entries for `i1`, `i2`
- `condense`: copy moved row state from `last_req_index` to `empty_index`, then
  clear the old list entry

## `vllm/sampling_params.py`

No code change is required for the minimal patch. `SamplingParams.extra_args`
already exists in `v0.20.2` at lines 289-292 and is copied into
`NewRequestData`/`CachedRequestState` through the normal request path.

## `vllm/v1/request.py` and `vllm/v1/core/sched/output.py`

No code change is required for the minimal patch if metadata is parsed in
`InputBatch.add_request` from `request.sampling_params.extra_args`.

Relevant inspected path:

- `Request.__init__` keeps `sampling_params` at lines 79-83.
- `Request.from_engine_core_request(...)` copies `sampling_params` at
  lines 181-203.
- `NewRequestData.from_request(...)` copies `sampling_params` at lines 45-63.
- `GPUModelRunner` creates `CachedRequestState(..., sampling_params=...)` at
  lines 1158-1170.

## Optional Stats Patch

Existing vLLM aggregate acceptance stats live in:

- `vllm/v1/spec_decode/metrics.py::SpecDecodingStats`
- `vllm/v1/core/sched/scheduler.py::make_spec_decoding_stats`
- `vllm/v1/core/sched/scheduler.py::update_from_output` around lines 1367-1391

The proposer can expose hit/miss/proposed stats itself. Accepted token stats
require scheduler/rejection-sampler instrumentation or log scraping, because
`LLM.generate` outputs do not include draft acceptance fields.
