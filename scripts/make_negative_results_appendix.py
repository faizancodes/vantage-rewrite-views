#!/usr/bin/env python3
"""Build the negative-results appendix table for the final paper package."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis" / "final_paper_artifacts" / "continuous_batched_pld_final" / "tables"


ROWS = [
    [
        "delta_cache_pld",
        "~1.00x",
        "Local delta/cache reuse did not add enough accepted spans over optimized PLD.",
        "Exact-copy PLD already captures most reusable local context.",
    ],
    [
        "fuzzy_resync_pld",
        "~1.00x",
        "Fuzzy recovery added overhead and noisy candidates without broad real-commit coverage.",
        "Approximate text matching is not enough when verifier fixed cost dominates.",
    ],
    [
        "exact candidate reranking",
        "~1.020x runtime",
        "Offline accepted-length projections overestimated real step reduction; token0/1 rejection stayed high.",
        "Candidate quality must be measured by replayed decode-step reduction, not average accepted length.",
    ],
    [
        "large-K exact reranking oracle",
        "K=32 corrected oracle 1.122x",
        "Even perfect selection from larger exact candidate sets stayed below the 1.20x target.",
        "Exact PLD ambiguity is not the main remaining bottleneck.",
    ],
    [
        "MTP / queued MTP",
        "perfect actual-policy queued oracle 1.129x; trained router-selected MTP 1.035x",
        "The MTP signal existed offline, but runtime scheduling and router-selected head quality were too weak.",
        "Better tokens alone do not help unless the use distribution and verification schedule align.",
    ],
    [
        "syntax_slot_pld",
        "~1.005x best",
        "Syntax-class matches and slot filling did not produce long reliable concrete prefixes.",
        "Structure-only recurrence is too sparse/noisy for this benchmark.",
    ],
    [
        "weak-router capped PLD",
        "best smoke 0.968x",
        "The router predicted weak PLD steps well, but shortening drafts did not remove the verifier fixed cost.",
        "Verifier time is mostly fixed with respect to draft length.",
    ],
    [
        "selective LM-head verifier",
        "projected 0.997x; LM-head share 5.7%",
        "Certification was exact on sampled tokens, but risky sets were too large and LM-head cost was too small.",
        "LM-head avoidance cannot deliver 20% when the transformer dominates.",
    ],
    [
        "diff/hunk-only generation",
        "best n=50 apply rate 4%; best speedup 0.362x",
        "The model often ignored patch-only formatting and emitted malformed or non-applicable edits.",
        "Task reparameterization needs stronger format control before it can be compared as a speed method.",
    ],
    [
        "static/CUDA graph verifier",
        "no credible 20% win in current PyTorch path",
        "Static-shape/CUDA graph prototypes did not expose enough removable overhead.",
        "The productive systems lever was batching verifier launches across tasks.",
    ],
]


def _latex_escape(s: object) -> str:
    text = str(s)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    headers = ["Approach", "Best result / ceiling", "Why it failed", "Lesson"]
    md = [
        "# Negative Results Appendix",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in ROWS:
        md.append("| " + " | ".join(row) + " |")
    (out_dir / "negative_results_appendix.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Negative results from PLD-adjacent decoder branches.}",
        r"\label{tab:negative-results}",
        r"\begin{tabular}{p{0.18\linewidth}p{0.18\linewidth}p{0.30\linewidth}p{0.24\linewidth}}",
        r"\toprule",
        " & ".join(_latex_escape(h) for h in headers) + r" \\",
        r"\midrule",
    ]
    for row in ROWS:
        tex.append(" & ".join(_latex_escape(x) for x in row) + r" \\")
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (out_dir / "negative_results_appendix.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"wrote negative-results appendix to {out_dir}")


if __name__ == "__main__":
    main()
