# VANTAGE Real-Edit Benchmark Protocol

Date: 2026-05-16

## Purpose

The current 33-row real-commit pilot is inconclusive and cannot support a broad
real-world code-edit acceleration claim. This protocol defines what evidence
would be required to reopen that claim.

## Data Source

Use public GitHub commits or locally mined public-repository manifests. Each
candidate row must preserve:

- repository URL.
- commit hash and parent hash.
- commit URL.
- file path.
- old file/function text.
- new file/function text.
- prompt text.
- extracted rewrite map.
- task category.
- rejection status and rejection reason.

## Fixed Collection Protocol

1. Define categories and minimum counts before mining.
2. Mine candidate commits without running any decoder speed measurement.
3. Extract old/new single-file Python edits.
4. Build prompt and rewrite map from the diff, commit message, or explicit
   transformation evidence.
5. Curate rows blind to decoder speed.
6. Save a rejection log for every candidate.
7. Freeze manifests before timing runs.
8. Run vanilla, tuned PLD, and VANTAGE/SafeRoute under identical settings.

## Categories

Required categories:

- variable rename.
- attribute / field migration.
- literal replacement.
- API replacement.
- naming-style change.
- import migration.
- multi-edit structured change.
- comment/docstring rewrite.
- nonlocal semantic edit.
- negative / no-op edit.

Each row must label:

- explicit, partial, ambiguous, or inferred map.
- local or nonlocal edit.
- whether old/new parse as Python.
- whether unit tests are available.
- whether syntax/AST checks are available.
- whether the map is prompt-visible.
- whether the model output complies with the rewrite map.

## Metrics

Correctness and quality:

- decoder parity with vanilla.
- exact target text match.
- exact patch match where applicable.
- syntax validity.
- AST validity.
- unit-test pass rate where available.
- semantic-equivalence proxy.
- rewrite compliance.

Speed:

- tok/s over vanilla.
- tok/s over tuned PLD.
- per-task latency.
- p50/p90/p95/p99 latency.
- worst-case slowdown.
- per-category speedup.
- regression rate.
- route choice distribution.

Mechanism:

- target forward count.
- PLD route hits.
- rewrite-view route hits.
- accepted tokens per route.
- rejected draft tokens.
- setup time.
- transformed-view coverage.

## Minimum Credible Sizes

These are planning thresholds for future evidence:

- Workshop: 100--200 rows, at least 20 per major category, full provenance.
- Serious systems venue: 500--1000 rows, at least 50 per category, fixed
  protocol, rejection log, parity metrics, and task-quality metrics.
- Artifact-backed arXiv: 300--500 rows with commit hashes, manifests, scripts,
  and category tables.

## Current Pilot Status

Current pilot:

- 25 real rename rows.
- 8 real field migration rows.
- confidence intervals cross 1.
- exact-target, syntax, and parity rates are weak.
- rejection reasons and blind-curation status are not recorded.

Therefore the pilot remains an appendix limitation, not main evidence.
