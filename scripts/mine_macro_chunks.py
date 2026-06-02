"""Mine frequent token chunks for the macro proposer.

This is intentionally lightweight and diagnostic.  It mines from a user-chosen
external corpus and writes decoded chunks that can be passed back to
run_eagle_eval.py via --macro-chunks-json.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def _row_text(row: dict) -> str:
    return (
        row.get("content")
        or row.get("text")
        or row.get("code")
        or row.get("output")
        or row.get("response")
        or ""
    )


def _looks_macro_like(text: str) -> bool:
    if len(text.strip()) < 2:
        return False
    if re.fullmatch(r"[A-Za-z0-9_ ]+", text):
        return False
    return any(ch in text for ch in "\n:(){}[],.;=")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    p.add_argument("--corpus", default="codeparrot/codeparrot-clean")
    p.add_argument("--config", default="")
    p.add_argument("--data-dir", default="")
    p.add_argument("--language-filter", default="")
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--max-chars-per-sample", type=int, default=8000)
    p.add_argument("--max-tokens-per-sample", type=int, default=1200)
    p.add_argument("--min-tokens", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--top-k", type=int, default=256)
    p.add_argument("--output-json", required=True)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.target)
    load_kwargs: dict = {"split": "train", "streaming": True}
    if args.config:
        load_kwargs["name"] = args.config
    if args.data_dir:
        load_kwargs["data_dir"] = args.data_dir
    ds = load_dataset(args.corpus, **load_kwargs)
    if args.language_filter:
        ds = ds.filter(lambda r: r.get("language") == args.language_filter)

    counts: Counter[tuple[int, ...]] = Counter()
    n_seen = 0
    for row in ds:
        if n_seen >= args.n_samples:
            break
        text = _row_text(row)
        if not text:
            continue
        text = text[: args.max_chars_per_sample]
        ids = tok(text, add_special_tokens=False).input_ids[: args.max_tokens_per_sample]
        if len(ids) < args.min_tokens:
            continue
        for n in range(args.min_tokens, args.max_tokens + 1):
            if len(ids) < n:
                continue
            for i in range(0, len(ids) - n + 1):
                counts[tuple(int(t) for t in ids[i : i + n])] += 1
        n_seen += 1

    chunks = []
    for token_tuple, count in counts.most_common(args.top_k * 20):
        text = tok.decode(list(token_tuple), skip_special_tokens=True)
        if _looks_macro_like(text):
            chunks.append({"text": text, "count": count, "n_tokens": len(token_tuple)})
        if len(chunks) >= args.top_k:
            break

    out = {
        "schema": "asts-spec/macro_chunks/v1",
        "target": args.target,
        "corpus": args.corpus,
        "config": args.config or None,
        "data_dir": args.data_dir or None,
        "language_filter": args.language_filter or None,
        "n_samples": n_seen,
        "chunks": [c["text"] for c in chunks],
        "details": chunks,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

