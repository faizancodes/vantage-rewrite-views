"""Offline scaffold for hidden-state PLD candidate reranking.

This script does not implement runtime hidden-state reranking.  It evaluates a
future trace format where each ambiguous PLD step stores the current decode
hidden state and compact hidden states for the top-K candidate source positions.
Candidate selection is then scored offline and passed through the same
step-replay projection used by ``evaluate_pld_reranker_step_projection.py``.

Expected hidden trace JSONL schema, one row per ambiguous step:

{
  "task_id": "...",
  "step_id": 12,
  "current_hidden": [float, ...],
  "candidates": [
    {
      "rank": 0,
      "source_position": 123,
      "source_hidden": [float, ...],
      "source_window_mean_hidden": [float, ...]   # optional
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_pld_reranker_step_projection import (  # noqa: E402
    DEFAULT_INPUT,
    _limit_examples_like_runtime,
    _load_baseline_steps,
    _project_ambiguous_only,
    _project_with_full_steps,
    _read_report,
    _result_to_dict,
)
from scripts.train_pld_candidate_reranker import (  # noqa: E402
    Candidate,
    Example,
    _candidate_by_baseline,
    _load_jsonl,
    _parse_examples,
)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return -1.0e30
    n = min(len(a), len(b))
    dot = 0.0
    aa = 0.0
    bb = 0.0
    for i in range(n):
        x = float(a[i])
        y = float(b[i])
        dot += x * y
        aa += x * x
        bb += y * y
    if aa <= 0.0 or bb <= 0.0:
        return -1.0e30
    return dot / (math.sqrt(aa) * math.sqrt(bb))


def _load_hidden_rows(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row in _load_jsonl(path):
        rows[(str(row.get("task_id") or ""), int(row.get("step_id") or 0))] = row
    return rows


def _choose_by_hidden(
    ex: Example,
    *,
    hidden_rows: dict[tuple[str, int], dict[str, Any]],
    use_window_mean: bool,
    mix_rank_prior: float,
) -> Candidate:
    row = hidden_rows.get((ex.task_id, ex.step_id))
    if row is None:
        return _candidate_by_baseline(ex)
    current = row.get("current_hidden")
    if not isinstance(current, list):
        return _candidate_by_baseline(ex)
    hidden_by_pos: dict[int, dict[str, Any]] = {}
    for cand in row.get("candidates") or []:
        if isinstance(cand, dict) and cand.get("source_position") is not None:
            hidden_by_pos[int(cand["source_position"])] = cand

    best = _candidate_by_baseline(ex)
    best_score = -1.0e30
    for cand in ex.candidates:
        hrow = hidden_by_pos.get(cand.source_position)
        if hrow is None:
            continue
        key = "source_window_mean_hidden" if use_window_mean else "source_hidden"
        hidden = hrow.get(key) or hrow.get("source_hidden")
        if not isinstance(hidden, list):
            continue
        score = _cosine(current, hidden) - mix_rank_prior * cand.rank0
        if score > best_score:
            best = cand
            best_score = score
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ambiguous-candidates", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--hidden-trace", type=Path, required=True)
    ap.add_argument("--steps", type=Path, default=None)
    ap.add_argument("--method", default="blazedit_pld_w128_n10")
    ap.add_argument("--output-dir", type=Path, default=Path("/tmp/pld_candidate_oracle_v2/hidden_state_reranker"))
    ap.add_argument("--pld-hidden-rerank-proj-dim", type=int, default=128)
    ap.add_argument("--pld-hidden-rerank-top-k", type=int, default=4)
    ap.add_argument("--use-window-mean", action="store_true")
    ap.add_argument("--mix-rank-prior", type=float, default=0.0)
    args = ap.parse_args()

    rows = _load_jsonl(args.ambiguous_candidates)
    examples = _limit_examples_like_runtime(
        _parse_examples(
            rows,
            max(args.pld_hidden_rerank_top_k, 64),
            enable_left_extension=True,
        ),
        top_k=args.pld_hidden_rerank_top_k,
    )
    hidden_rows = _load_hidden_rows(args.hidden_trace)
    if not hidden_rows:
        raise SystemExit(f"no hidden rows found in {args.hidden_trace}")

    def choose(ex: Example) -> Candidate:
        return _choose_by_hidden(
            ex,
            hidden_rows=hidden_rows,
            use_window_mean=args.use_window_mean,
            mix_rank_prior=args.mix_rank_prior,
        )

    if args.steps:
        result = _project_with_full_steps(
            examples=examples,
            steps_by_task=_load_baseline_steps(args.steps, method=args.method),
            choose=choose,
        )
    else:
        report = _read_report(args.ambiguous_candidates)
        baseline_steps = int(report.get("total_steps") or len(examples))
        result = _project_ambiguous_only(
            examples=examples,
            choose=choose,
            baseline_steps=baseline_steps,
        )

    payload = {
        "ambiguous_candidates": str(args.ambiguous_candidates),
        "hidden_trace": str(args.hidden_trace),
        "top_k": args.pld_hidden_rerank_top_k,
        "proj_dim": args.pld_hidden_rerank_proj_dim,
        "use_window_mean": args.use_window_mean,
        "mix_rank_prior": args.mix_rank_prior,
        "result": dict(_result_to_dict(result), policy="hidden_state_cosine"),
        "decision_rule": (
            "Proceed to runtime hidden-state reranking only if corrected "
            "projected speedup is >= 1.25x."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "hidden_state_reranker_projection.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    lines = [
        "# Hidden-State PLD Reranker Projection",
        "",
        f"hidden trace: `{args.hidden_trace}`",
        f"corrected projected speedup: `{payload['result']['corrected_projected_speedup']:.3f}x`",
        f"projected steps: `{payload['result']['projected_steps']}`",
        f"token0/1 proxy: `{payload['result']['token0_1_rejection_proxy_pct']:.1f}%`",
        "",
        payload["decision_rule"],
    ]
    (args.output_dir / "hidden_state_reranker_projection.md").write_text(
        "\n".join(lines) + "\n"
    )
    print((args.output_dir / "hidden_state_reranker_projection.md").read_text())


if __name__ == "__main__":
    main()
