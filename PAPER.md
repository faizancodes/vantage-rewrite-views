# VANTAGE: Hidden Rewrite Views for Fixed-Prompt Speculative Code-Edit Decoding

The source of truth for the current paper draft is
[`paper/vantage.tex`](paper/vantage.tex).

Earlier markdown drafts used older internal names and contained stale cross-run
claims. The current LaTeX draft frames the paper around VANTAGE: a fixed-prompt
decoder whose core primitive, Rewrite-View Lookup, builds a hidden
prompt-derived rewrite view for target-verified speculative drafts. SafeRoute is
the validated router. Preliminary ViewBank variants are diagnostic only.

The controlled rewrite-view claim is conditional on Qwen2.5-Coder-7B structured
Python edit workloads with prompt-visible explicit maps. The main controlled
throughput table uses the audited fp32/sdpa exact path; bf16 timing artifacts
are diagnostic only because they show parity drift. The balanced 1000-task
real-commit ViewBank result is treated as a timing diagnostic, not a broad
real-edit-quality claim. Exact synthetic-target match, syntax, and local
rewrite compliance are reported separately, including successful-edit speed
slices. It is not a vLLM production-serving claim, a broad real-commit-quality
claim, a multi-model universality claim, or a bf16 optimized-path
deployment-readiness claim.
