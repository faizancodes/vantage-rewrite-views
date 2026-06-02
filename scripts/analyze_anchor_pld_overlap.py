"""Attribute rewrite-aware accepted tokens to PLD-overlap or PLD-miss buckets.

The key reviewer question after adding BlazEdit-style PLD is whether
reference alignment or transformed lookup views contribute tokens that exact
prompt lookup could not have proposed at the same decode step. This script
reconstructs method prefixes from ``completions.jsonl`` and ``steps.jsonl`` and
compares accepted non-root tokens from Anchor/TransPLD-style steps against:

* unrooted BlazEdit-style PLD over the current prefix;
* rooted VANTAGE PLD over ``prefix + target_root``.
* optional rewrite-normalized PLD over transformed reference pools.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asts.blazedit_decoder import prompt_lookup_draft  # noqa: E402
from asts.code_proposers import ProposerState, RewriteNormalizedPLDProposer  # noqa: E402


ANCHOR_KINDS = {
    "edit_anchor",
    "edit_anchor_only",
    "edit_anchor_suffix",
    "edit_anchor_tail",
    "edit_anchor_pld",
    "rewrite_anchor_pld",
    "edit_symbol_anchor",
    "rewrite_norm_pld",
    "transpld",
    "transpld_infer",
    "transpld_compound",
    "transpld_cursor",
    "routed_transpld",
}
ANCHOR_MATCH_KINDS = {
    "edit_anchor",
    "rewrite_anchor",
    "bidir",
    "vref",
    "oracle",
    "transpld_bidir",
    "transpld_vref",
    "transpld_bidir_inferred",
    "routed_transpld_bidir",
    "routed_transpld_vref",
    "cursor",
    "precomputed_transpld_compete",
}
_REWRITE_PLD_RE = re.compile(
    r"rewrite_pld_(?P<mode>vref|bidir|oracle)_w(?P<w>\d+)_n(?P<n>\d+)$"
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_tokenizer(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def _encode(tokenizer, text: str) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def _matching_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if int(x) != int(y):
            break
        n += 1
    return n


def _is_anchor_step(row: dict[str, Any]) -> bool:
    kind = row.get("proposal_kind")
    match_kind = row.get("proposal_match_kind")
    return kind in ANCHOR_KINDS or match_kind in ANCHOR_MATCH_KINDS


def analyze(
    *,
    completions: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    method: str,
    tokenizer_name: str,
    pld_window: int,
    pld_ngram: int,
    rewrite_pld_method: str = "",
) -> dict[str, Any]:
    tokenizer = _load_tokenizer(tokenizer_name)
    rewrite_pld: RewriteNormalizedPLDProposer | None = None
    if rewrite_pld_method:
        match = _REWRITE_PLD_RE.fullmatch(rewrite_pld_method)
        if match is None:
            raise ValueError(f"unsupported rewrite-normalized PLD method: {rewrite_pld_method}")
        rewrite_pld = RewriteNormalizedPLDProposer(
            tokenizer,
            mode=match.group("mode"),
            max_draft_len=int(match.group("w")),
            max_matching_ngram_size=int(match.group("n")),
            min_matching_ngram_size=1,
        )
    steps_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step in steps:
        if step.get("method") == method:
            steps_by_task[str(step.get("task_id"))].append(step)
    for values in steps_by_task.values():
        values.sort(key=lambda r: int(r.get("step") or 0))

    totals: defaultdict[str, float] = defaultdict(float)
    task_rows: list[dict[str, Any]] = []
    for row in completions:
        task_id = str(row.get("task_id"))
        output = (row.get("outputs") or {}).get(method)
        method_steps = steps_by_task.get(task_id) or []
        if not output or not method_steps:
            continue
        prompt_ids = _encode(tokenizer, str(row.get("prompt") or ""))
        prompt_text = str(row.get("prompt") or "")
        row_metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        metadata = dict(row_metadata)
        if "rewrite_pairs" in row and "rewrite_pairs" not in metadata:
            metadata["rewrite_pairs"] = row["rewrite_pairs"]
        reference = str(row.get("reference") or metadata.get("reference") or "")
        output_ids = _encode(tokenizer, str(output.get("raw_text") or output.get("text") or ""))
        pos = 0
        task: defaultdict[str, float] = defaultdict(float)
        for step in method_steps:
            n_emitted = int(step.get("n_emitted") or 0)
            if n_emitted <= 0 or pos >= len(output_ids):
                continue
            accepted_nonroot = int(step.get("n_accepted_nonroot_drafts") or 0)
            if _is_anchor_step(step):
                task["anchor_hits"] += 1
                totals["anchor_hits"] += 1
                if accepted_nonroot <= 0:
                    task["zero_accept_anchor_hits"] += 1
                    totals["zero_accept_anchor_hits"] += 1
                else:
                    root = output_ids[pos]
                    accepted = output_ids[pos + 1 : pos + 1 + accepted_nonroot]
                    prefix = prompt_ids + output_ids[:pos]

                    unrooted, unrooted_match, _, _ = prompt_lookup_draft(
                        prefix,
                        max_matching_ngram_size=pld_ngram,
                        max_draft_tokens=pld_window,
                    )
                    unrooted_after_root = unrooted[1:] if unrooted and unrooted[0] == root else []
                    unrooted_overlap = _matching_prefix_len(unrooted_after_root, accepted)

                    rooted, rooted_match, _, _ = prompt_lookup_draft(
                        prefix + [root],
                        max_matching_ngram_size=pld_ngram,
                        max_draft_tokens=pld_window,
                    )
                    rooted_overlap = _matching_prefix_len(rooted, accepted)
                    rewrite_overlap = 0
                    if rewrite_pld is not None:
                        text_before = tokenizer.decode(prefix, skip_special_tokens=False)
                        text_after = tokenizer.decode(prefix + [root], skip_special_tokens=False)
                        rewrite_prop = rewrite_pld.propose(
                            ProposerState(
                                prefix=prefix,
                                teacher_argmax=root,
                                text_before=text_before,
                                text_after=text_after,
                                ctx=None,
                                language=str(row.get("language") or "python"),
                                prompt_len=len(prompt_ids),
                                reference=reference,
                                metadata=metadata,
                            )
                        )
                        if rewrite_prop is not None:
                            task["rewrite_norm_pld_hits"] += 1
                            totals["rewrite_norm_pld_hits"] += 1
                            rewrite_overlap = _matching_prefix_len(rewrite_prop.tokens, accepted)

                    task["anchor_accepted_nonroot"] += accepted_nonroot
                    task["after_unrooted_pld_miss"] += max(0, accepted_nonroot - unrooted_overlap)
                    task["after_rooted_pld_miss"] += max(0, accepted_nonroot - rooted_overlap)
                    task["after_rewrite_norm_pld_miss"] += max(0, accepted_nonroot - rewrite_overlap)
                    task["unrooted_pld_overlap"] += unrooted_overlap
                    task["rooted_pld_overlap"] += rooted_overlap
                    task["rewrite_norm_pld_overlap"] += rewrite_overlap
                    task["unrooted_pld_match_len_sum"] += unrooted_match
                    task["rooted_pld_match_len_sum"] += rooted_match

                    totals["anchor_accepted_nonroot"] += accepted_nonroot
                    totals["after_unrooted_pld_miss"] += max(0, accepted_nonroot - unrooted_overlap)
                    totals["after_rooted_pld_miss"] += max(0, accepted_nonroot - rooted_overlap)
                    totals["after_rewrite_norm_pld_miss"] += max(0, accepted_nonroot - rewrite_overlap)
                    totals["unrooted_pld_overlap"] += unrooted_overlap
                    totals["rooted_pld_overlap"] += rooted_overlap
                    totals["rewrite_norm_pld_overlap"] += rewrite_overlap
            pos += n_emitted
        if task:
            task_rows.append({"task_id": task_id, **dict(task)})

    accepted = totals["anchor_accepted_nonroot"]
    return {
        "schema": "asts-spec/anchor-pld-overlap/v1",
        "method": method,
        "tokenizer": tokenizer_name,
        "pld_window": pld_window,
        "pld_ngram": pld_ngram,
        "rewrite_pld_method": rewrite_pld_method,
        "n_tasks": len(task_rows),
        "totals": dict(totals),
        "fractions": {
            "after_unrooted_pld_miss": totals["after_unrooted_pld_miss"] / accepted if accepted else 0.0,
            "after_rooted_pld_miss": totals["after_rooted_pld_miss"] / accepted if accepted else 0.0,
            "after_rewrite_norm_pld_miss": totals["after_rewrite_norm_pld_miss"] / accepted if accepted else 0.0,
            "zero_accept_anchor_hit_rate": totals["zero_accept_anchor_hits"] / totals["anchor_hits"]
            if totals["anchor_hits"]
            else 0.0,
        },
        "tasks": task_rows,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    totals = report["totals"]
    frac = report["fractions"]
    lines = [
        "# Anchor vs PLD Overlap",
        "",
        f"Method: `{report['method']}`",
        f"Tasks: {report['n_tasks']}",
        f"PLD config: window={report['pld_window']}, ngram={report['pld_ngram']}",
        f"Rewrite-normalized PLD: `{report.get('rewrite_pld_method') or 'not computed'}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Anchor hits | {totals.get('anchor_hits', 0):.0f} |",
        f"| Zero-accept anchor hits | {totals.get('zero_accept_anchor_hits', 0):.0f} |",
        f"| Anchor accepted non-root tokens | {totals.get('anchor_accepted_nonroot', 0):.0f} |",
        f"| Accepted tokens after unrooted PLD miss | {totals.get('after_unrooted_pld_miss', 0):.0f} |",
        f"| Accepted tokens after rooted PLD miss | {totals.get('after_rooted_pld_miss', 0):.0f} |",
        f"| Accepted tokens after rewrite-normalized PLD miss | {totals.get('after_rewrite_norm_pld_miss', 0):.0f} |",
        f"| Fraction after unrooted PLD miss | {100.0 * frac.get('after_unrooted_pld_miss', 0.0):.1f}% |",
        f"| Fraction after rooted PLD miss | {100.0 * frac.get('after_rooted_pld_miss', 0.0):.1f}% |",
        f"| Fraction after rewrite-normalized PLD miss | {100.0 * frac.get('after_rewrite_norm_pld_miss', 0.0):.1f}% |",
        f"| Zero-accept anchor hit rate | {100.0 * frac.get('zero_accept_anchor_hit_rate', 0.0):.1f}% |",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--completions", required=True)
    p.add_argument("--steps", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--target-tokenizer", default="Qwen/Qwen2.5-Coder-7B")
    p.add_argument("--pld-window", type=int, default=40)
    p.add_argument("--pld-ngram", type=int, default=10)
    p.add_argument("--rewrite-pld-method", default="")
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    args = p.parse_args()

    report = analyze(
        completions=_load_jsonl(Path(args.completions)),
        steps=_load_jsonl(Path(args.steps)),
        method=args.method,
        tokenizer_name=args.target_tokenizer,
        pld_window=args.pld_window,
        pld_ngram=args.pld_ngram,
        rewrite_pld_method=args.rewrite_pld_method,
    )
    Path(args.output_json).write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, Path(args.output_md))


if __name__ == "__main__":
    main()
