# VANTAGE paper — Overleaf-ready bundle

This directory contains the LaTeX source for **VANTAGE: Hidden Rewrite Views
for Fixed-Prompt Speculative Code-Edit Decoding**.

## Contents

- `vantage.tex` — main and only TeX source (single-file paper)
- `README.md` — this file

The bibliography is embedded in the `.tex` file via `thebibliography`,
so there is no separate `.bib` file. Figures live in `paper/figures/`.

## How to compile on Overleaf

1. **Upload this folder as a zip** via Overleaf's "New Project → Upload Project"
2. Overleaf should auto-detect `vantage.tex` as the main file. If not,
   set the main document manually in Project Settings.
3. Choose **pdfLaTeX** as the compiler (default; this paper uses
   `inputenc + utf8` which is pdfLaTeX-friendly).
4. Compile.

The paper should build cleanly on a full TeX Live / Overleaf install and was
kept free of `enumitem`/`multirow` dependencies for minimal local TeX installs.

## How to compile locally

```sh
pdflatex vantage.tex
pdflatex vantage.tex   # second pass to resolve cross-references
```

If local `pdflatex` fails with a missing package, install the missing LaTeX
package or use Overleaf/full TeX Live.

## Structure

- §1   Introduction
- §2   Background (speculative decoding, PLD as identity-view lookup,
       reference drift)
- §3   Method: view-based speculative decoding, Rewrite-View Lookup,
       SafeRoute, and preliminary ViewBank terminology
- §4   Experimental setup
- §5   Results (controlled rewrite-view mechanism benchmark,
       same-policy lookup ablation, multi-view real-commit diagnostic,
       prompt-injection baseline, backend audit, tails, instruction-compliance
       boundary, quality/audits)
- §6   Related work
- §7   Discussion and limitations
- §8   Conclusion
- Appendix  Focused run tags, router fit, inconclusive real-commit pilot,
  real-edit benchmark protocol, full systems counters, strict target/syntax
  diagnostics

## To convert to a conference template

Replace the first line:

```latex
\documentclass[11pt,a4paper]{article}
```

with the appropriate documentclass for your venue, e.g.:
- NeurIPS:  `\documentclass{neurips_2024}`
- ICML:     `\documentclass{icml2024}`
- arXiv:    leave as `article` and submit as-is

Most conference templates also require their own bibliography style
(`\bibliographystyle{unsrtnat}` etc.); the inline `thebibliography`
should still render.
