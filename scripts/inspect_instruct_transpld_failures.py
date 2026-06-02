"""Inspect base-model TransPLD successes against instruct-model failures.

This is intentionally qualitative but reproducible: it selects the largest
candidate-over-PLD wins from one run and the largest losses from another, then
prints prompt/reference/target/output excerpts plus trace summaries.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _task_tps(output: dict[str, Any]) -> float:
    wall_us = float(output.get("wall_us", 0.0) or 0.0)
    tokens = int(output.get("n_new_tokens", 0) or 0)
    return tokens / (wall_us / 1e6) if wall_us > 0 else 0.0


def _ratio(row: dict[str, Any], candidate: str, baseline: str) -> float:
    outputs = row.get("outputs") or {}
    cand = outputs.get(candidate)
    base = outputs.get(baseline)
    if not cand or not base:
        return 0.0
    base_tps = _task_tps(base)
    return _task_tps(cand) / base_tps if base_tps else 0.0


def _rewrite_pairs(row: dict[str, Any]) -> dict[str, str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    pairs = metadata.get("rewrite_pairs") or {}
    if isinstance(pairs, dict):
        return {str(k): str(v) for k, v in pairs.items()}
    out: dict[str, str] = {}
    if isinstance(pairs, list):
        for item in pairs:
            if isinstance(item, dict):
                old = item.get("old") or item.get("from")
                new = item.get("new") or item.get("to")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                old, new = item[0], item[1]
            else:
                continue
            if old is not None and new is not None:
                out[str(old)] = str(new)
    return out


def _marker_summary(text: str, pairs: dict[str, str]) -> dict[str, Any]:
    stripped = text.lstrip()
    pre_fence = stripped.split("```", 1)[0] if "```" in stripped else stripped[:120]
    old_hits = sum(text.count(old) for old in pairs if old)
    new_hits = sum(text.count(new) for new in pairs.values() if new)
    return {
        "old_hits": old_hits,
        "new_hits": new_hits,
        "starts_with_fence": stripped.startswith("```"),
        "fence_count": text.count("```"),
        "pre_fence_chars": len(pre_fence.strip()),
        "has_conversational_preamble": bool(
            re.search(r"\b(here|sure|certainly|below|updated)\b", pre_fence, flags=re.I)
        ),
    }


def _step_summary(steps: list[dict[str, Any]], task_id: str, method: str) -> dict[str, Any]:
    rows = [s for s in steps if s.get("task_id") == task_id and s.get("method") == method]
    reasons: Counter[str] = Counter()
    match_kinds: Counter[str] = Counter()
    trans_steps = 0
    zero_accept = 0
    accepted = 0
    emitted = 0
    for row in rows:
        reason = row.get("proposal_route_reason")
        if reason:
            reasons[str(reason)] += 1
        match_kind = row.get("proposal_match_kind")
        if match_kind:
            match_kinds[str(match_kind)] += 1
        if row.get("proposal_route") == "transpld" or (
            match_kind and "transpld" in str(match_kind)
        ):
            trans_steps += 1
            acc = int(row.get("n_accepted_nonroot_drafts") or 0)
            accepted += acc
            if acc == 0:
                zero_accept += 1
        emitted += int(row.get("n_emitted") or 0)
    return {
        "steps": len(rows),
        "emitted": emitted,
        "transpld_steps": trans_steps,
        "transpld_zero_accept": zero_accept,
        "transpld_accepted_nonroot": accepted,
        "route_reasons": dict(reasons.most_common(5)),
        "match_kinds": dict(match_kinds.most_common(5)),
    }


def _excerpt(text: str, limit: int = 700) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n..."


def _case_block(
    *,
    title: str,
    row: dict[str, Any],
    ratio: float,
    candidate: str,
    baseline: str,
    steps: list[dict[str, Any]],
) -> str:
    outputs = row.get("outputs") or {}
    cand_out = outputs.get(candidate) or {}
    base_out = outputs.get(baseline) or {}
    cand_text = cand_out.get("raw_text") or cand_out.get("text") or ""
    base_text = base_out.get("raw_text") or base_out.get("text") or ""
    pairs = _rewrite_pairs(row)
    cand_markers = _marker_summary(cand_text, pairs)
    base_markers = _marker_summary(base_text, pairs)
    target = row.get("deterministic_target") or ""
    reference = row.get("reference") or ""
    task_id = row.get("task_id")
    step_summary = _step_summary(steps, str(task_id), candidate)
    lines = [
        f"### {title}: `{task_id}`",
        "",
        f"Candidate/PLD ratio: `{ratio:.3f}`",
        f"Rewrite pairs: `{pairs}`",
        f"Candidate markers: `{cand_markers}`",
        f"PLD markers: `{base_markers}`",
        f"Candidate trace: `{step_summary}`",
        "",
        "**Reference excerpt**",
        "```python",
        _excerpt(reference, 550),
        "```",
        "",
        "**Target excerpt**",
        "```python",
        _excerpt(target, 550),
        "```",
        "",
        f"**Candidate `{candidate}` output excerpt**",
        "```python",
        _excerpt(cand_text),
        "```",
        "",
        f"**PLD `{baseline}` output excerpt**",
        "```python",
        _excerpt(base_text),
        "```",
        "",
    ]
    return "\n".join(lines)


def _select(rows: list[dict[str, Any]], candidate: str, baseline: str, *, high: bool, n: int):
    scored = [(row, _ratio(row, candidate, baseline)) for row in rows]
    scored = [(row, r) for row, r in scored if r > 0]
    scored.sort(key=lambda item: item[1], reverse=high)
    return scored[:n]


def _aggregate_markers(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    totals = defaultdict(int)
    n = 0
    for row in rows:
        out = (row.get("outputs") or {}).get(method)
        if not out:
            continue
        n += 1
        markers = _marker_summary(out.get("raw_text") or out.get("text") or "", _rewrite_pairs(row))
        for key, value in markers.items():
            if isinstance(value, bool):
                totals[key] += int(value)
            elif isinstance(value, int):
                totals[key] += value
    return {"n": n, **dict(totals)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run-dir", required=True)
    parser.add_argument("--instruct-run-dir", required=True)
    parser.add_argument("--candidate", default="vantage_adopt_simple_transpld_m4_w128_n10")
    parser.add_argument("--baseline", default="blazedit_pld_w128_n10")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    base_dir = Path(args.base_run_dir)
    instruct_dir = Path(args.instruct_run_dir)
    base_rows = _load_jsonl(base_dir / "completions.jsonl")
    instruct_rows = _load_jsonl(instruct_dir / "completions.jsonl")
    base_steps = _load_jsonl(base_dir / "steps.jsonl")
    instruct_steps = _load_jsonl(instruct_dir / "steps.jsonl")

    base_successes = _select(base_rows, args.candidate, args.baseline, high=True, n=args.n)
    instruct_failures = _select(instruct_rows, args.candidate, args.baseline, high=False, n=args.n)

    report = {
        "base_run": base_dir.name,
        "instruct_run": instruct_dir.name,
        "candidate": args.candidate,
        "baseline": args.baseline,
        "base_marker_totals": _aggregate_markers(base_rows, args.candidate),
        "instruct_marker_totals": _aggregate_markers(instruct_rows, args.candidate),
        "base_successes": [
            {"task_id": row.get("task_id"), "ratio": ratio} for row, ratio in base_successes
        ],
        "instruct_failures": [
            {"task_id": row.get("task_id"), "ratio": ratio} for row, ratio in instruct_failures
        ],
    }

    lines = [
        "# Instruct TransPLD Trace Inspection",
        "",
        f"Candidate: `{args.candidate}`. Baseline: `{args.baseline}`.",
        "",
        "## Aggregate Output Markers",
        "",
        f"Base candidate markers: `{report['base_marker_totals']}`",
        "",
        f"Instruct candidate markers: `{report['instruct_marker_totals']}`",
        "",
        "## Base Successes",
        "",
    ]
    for i, (row, ratio) in enumerate(base_successes, 1):
        lines.append(
            _case_block(
                title=f"Base success {i}",
                row=row,
                ratio=ratio,
                candidate=args.candidate,
                baseline=args.baseline,
                steps=base_steps,
            )
        )
    lines += ["", "## Instruct Failures", ""]
    for i, (row, ratio) in enumerate(instruct_failures, 1):
        lines.append(
            _case_block(
                title=f"Instruct failure {i}",
                row=row,
                ratio=ratio,
                candidate=args.candidate,
                baseline=args.baseline,
                steps=instruct_steps,
            )
        )

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2))
    md = "\n".join(lines)
    Path(args.output_md).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
