"""Attribute local suffix / n-gram proposal hits to prompt or generated source.

The decoder records token-index provenance for local reuse proposals.  This
script joins those step traces with completion prompts, classifies the source
pool and prompt region, and reports where accepted continuation tokens came
from.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_tokenizer(name: str):
    if not name:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    except Exception:
        return None


def _prompt_offsets(tokenizer, prompt: str) -> list[tuple[int, int]]:
    if tokenizer is None:
        return [(i, i + 1) for i in range(len(prompt))]
    try:
        enc = tokenizer(
            prompt,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        return [(int(a), int(b)) for a, b in enc.offset_mapping]
    except Exception:
        return []


def _line_at(text: str, char_idx: int) -> tuple[str, int]:
    char_idx = max(0, min(char_idx, len(text)))
    line_no = text.count("\n", 0, char_idx)
    line_start = text.rfind("\n", 0, char_idx) + 1
    line_end = text.find("\n", char_idx)
    if line_end < 0:
        line_end = len(text)
    return text[line_start:line_end], line_no


def _inside_triple_quote(text: str, char_idx: int) -> bool:
    before = text[:char_idx]
    return (before.count('"""') % 2 == 1) or (before.count("'''") % 2 == 1)


def classify_prompt_region(prompt: str, offsets: list[tuple[int, int]], start: int, end: int) -> str:
    if not offsets or start >= len(offsets):
        return "unknown"
    char_start = offsets[start][0]
    line, _ = _line_at(prompt, char_start)
    stripped = line.strip()
    if stripped.startswith(("from ", "import ")):
        return "import"
    if stripped.startswith(("#", "//")):
        return "comment"
    if stripped.startswith(("def ", "async def ", "class ")):
        return "signature"
    if _inside_triple_quote(prompt, char_start):
        if stripped.startswith((">>>", "...", "assert ")) or "assert " in stripped:
            return "assert/test"
        if "==" in stripped or "=>" in stripped:
            return "example"
        return "docstring"
    if stripped.startswith(("assert ", "print(")):
        return "assert/test"
    return "body"


def classify_pool(start: int | None, end: int | None, prompt_len: int | None) -> str:
    if start is None or end is None or prompt_len is None:
        return "unknown"
    if end <= prompt_len:
        return "prompt"
    if start >= prompt_len:
        return "generated"
    return "mixed"


def analyze(
    steps: list[dict[str, Any]],
    completions: list[dict[str, Any]],
    tokenizer_name: str,
    methods: set[str] | None,
) -> dict[str, Any]:
    tokenizer = _load_tokenizer(tokenizer_name)
    prompts = {row["task_id"]: row.get("prompt", "") for row in completions}
    offsets_by_task = {
        task_id: _prompt_offsets(tokenizer, prompt)
        for task_id, prompt in prompts.items()
    }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in steps:
        method = row.get("method")
        if methods and method not in methods:
            continue
        if row.get("proposal_kind") not in {"local_suffix", "ngram"}:
            continue
        grouped[str(method)].append(row)

    report: dict[str, Any] = {
        "schema": "asts-spec/suffix_source_analysis/v1",
        "tokenizer": tokenizer_name,
        "methods": {},
    }
    total_steps_by_method: dict[str, int] = defaultdict(int)
    for row in steps:
        method = str(row.get("method"))
        if methods and method not in methods:
            continue
        total_steps_by_method[method] += 1

    for method, rows in grouped.items():
        by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            start = row.get("proposal_source_start_token")
            end = row.get("proposal_source_end_token")
            prompt_len = row.get("prompt_len")
            pool = classify_pool(start, end, prompt_len)
            region = pool
            if pool == "prompt":
                prompt = prompts.get(row.get("task_id"), "")
                offsets = offsets_by_task.get(row.get("task_id"), [])
                region = classify_prompt_region(prompt, offsets, int(start), int(end))
            key = (pool, region)
            item = by_key.setdefault(
                key,
                {
                    "n_hits": 0,
                    "accepted_nonroot": 0,
                    "proposal_tokens": 0,
                    "match_len": 0,
                    "wall_us": 0.0,
                    "proposal_us": 0.0,
                },
            )
            item["n_hits"] += 1
            item["accepted_nonroot"] += row.get("n_accepted_nonroot_drafts", 0) or 0
            item["proposal_tokens"] += row.get("proposal_tokens", 0) or 0
            item["match_len"] += row.get("proposal_match_len", 0) or 0
            item["wall_us"] += row.get("wall_us", 0.0) or 0.0
            item["proposal_us"] += row.get("proposal_us", 0.0) or 0.0

        n_steps = total_steps_by_method.get(method, 0)
        n_hits = sum(item["n_hits"] for item in by_key.values())
        rows_out = []
        for (pool, region), item in sorted(
            by_key.items(), key=lambda kv: -kv[1]["accepted_nonroot"]
        ):
            hits = item["n_hits"]
            rows_out.append(
                {
                    "pool": pool,
                    "region": region,
                    "n_hits": hits,
                    "share_steps": hits / n_steps if n_steps else 0.0,
                    "share_hits": hits / n_hits if n_hits else 0.0,
                    "accepted_nonroot_total": item["accepted_nonroot"],
                    "accepted_nonroot_per_hit": item["accepted_nonroot"] / hits if hits else 0.0,
                    "accepted_nonroot_per_step": item["accepted_nonroot"] / n_steps if n_steps else 0.0,
                    "mean_match_len": item["match_len"] / hits if hits else 0.0,
                    "mean_proposal_tokens": item["proposal_tokens"] / hits if hits else 0.0,
                    "mean_wall_us": item["wall_us"] / hits if hits else 0.0,
                    "mean_proposal_us": item["proposal_us"] / hits if hits else 0.0,
                }
            )
        report["methods"][method] = {
            "n_steps": n_steps,
            "n_hits": n_hits,
            "hit_rate": n_hits / n_steps if n_steps else 0.0,
            "by_source": rows_out,
        }
    return report


def to_markdown(report: dict[str, Any]) -> str:
    lines = ["# Suffix Source Attribution", ""]
    for method, item in report["methods"].items():
        lines.append(f"## {method}")
        lines.append("")
        lines.append(f"- steps: {item['n_steps']}  hits: {item['n_hits']}  hit rate: {item['hit_rate']:.2%}")
        lines.append("")
        lines.append("| pool | region | hits | step share | hit share | acc/hit | acc/step | match | prop tok |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in item["by_source"]:
            lines.append(
                f"| {row['pool']} | {row['region']} | {row['n_hits']} | "
                f"{row['share_steps']:.2%} | {row['share_hits']:.2%} | "
                f"{row['accepted_nonroot_per_hit']:.2f} | "
                f"{row['accepted_nonroot_per_step']:.2f} | "
                f"{row['mean_match_len']:.2f} | {row['mean_proposal_tokens']:.2f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", required=True)
    p.add_argument("--completions", required=True)
    p.add_argument("--target-tokenizer", default="Qwen/Qwen2.5-Coder-7B")
    p.add_argument("--methods", default="")
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    args = p.parse_args()

    methods = {m.strip() for m in args.methods.split(",") if m.strip()} or None
    report = analyze(
        load_jsonl(Path(args.steps)),
        load_jsonl(Path(args.completions)),
        args.target_tokenizer,
        methods,
    )
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    md = to_markdown(report)
    Path(args.output_md).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
