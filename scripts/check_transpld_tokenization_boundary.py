#!/usr/bin/env python3
"""Show why TransPLD tokenizes the transformed reference as one string.

The failure mode is simple: after applying a rewrite, a tokenizer may merge
across the replacement boundary.  If an implementation tokenizes
``prefix + replacement + suffix`` as separate pieces and concatenates the IDs,
the resulting token stream can differ from tokenizing the transformed
reference as a whole.  TransPLD indexes the whole transformed reference to avoid
that boundary error.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B"
DEFAULT_OUTPUT = Path("artifacts/vantage_transpld/tables/tokenization_boundary.md")


@dataclass(frozen=True)
class Example:
    name: str
    original: str
    old: str
    new: str
    note: str


EXAMPLES = [
    Example(
        name="leading-space identifier rename",
        original="    return user.name.strip()\n",
        old="user",
        new="account",
        note="identifier after a space; Qwen can encode the leading-space form differently",
    ),
    Example(
        name="dotted attribute substitution",
        original="def label(user):\n    return user.name\n",
        old=".name",
        new=".display_name",
        note="replacement begins at an attribute boundary",
    ),
    Example(
        name="identifier boundary before punctuation",
        original="items = [user for user in users if user.name]\n",
        old="user",
        new="account",
        note="identifier appears before punctuation, spaces, and suffix characters",
    ),
    Example(
        name="snake-case style rewrite",
        original="def make_label(user_name):\n    return user_name.title()\n",
        old="user_name",
        new="display_name",
        note="underscore-containing identifier replacement",
    ),
    Example(
        name="quoted literal replacement",
        original='status = "user active"\nprint(status)\n',
        old="user active",
        new="account active",
        note="replacement inside a string literal with a left quote boundary",
    ),
]


def encode(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def offsets(tokenizer: Any, text: str, ids: list[int]) -> list[tuple[int, int]]:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        mapping = encoded.get("offset_mapping")
        if mapping and len(mapping) == len(ids):
            return [(int(a), int(b)) for a, b in mapping]
    except Exception:
        pass

    # Slow-tokenizer fallback.  This is only for diagnostics; Qwen uses the
    # fast path in normal environments.
    out: list[tuple[int, int]] = []
    cursor = 0
    for token_id in ids:
        piece = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        idx = text.find(piece, cursor)
        if idx < 0:
            idx = cursor
        out.append((idx, idx + len(piece)))
        cursor = idx + len(piece)
    return out


def replace_once(original: str, old: str, new: str) -> tuple[str, int, int]:
    start = original.find(old)
    if start < 0:
        raise ValueError(f"old text {old!r} not found in example")
    transformed = original[:start] + new + original[start + len(old) :]
    return transformed, start, start + len(new)


def token_window_from_offsets(
    ids: list[int],
    offs: list[tuple[int, int]],
    span_start: int,
    span_end: int,
    *,
    pad: int = 1,
) -> list[int]:
    touched = [
        i
        for i, (start, end) in enumerate(offs)
        if end > max(0, span_start - 1) and start < span_end + 1
    ]
    if not touched:
        return []
    left = max(0, min(touched) - pad)
    right = min(len(ids), max(touched) + pad + 1)
    return ids[left:right]


def token_window_from_piece_boundary(
    prefix_ids: list[int],
    replacement_ids: list[int],
    suffix_ids: list[int],
    *,
    pad: int = 1,
) -> list[int]:
    piece_ids = prefix_ids + replacement_ids + suffix_ids
    start = len(prefix_ids)
    end = start + len(replacement_ids)
    left = max(0, start - pad)
    right = min(len(piece_ids), end + pad)
    return piece_ids[left:right]


def analyze_example(tokenizer: Any, example: Example) -> dict[str, Any]:
    transformed, start, end = replace_once(example.original, example.old, example.new)
    prefix = transformed[:start]
    suffix = transformed[end:]
    whole_ids = encode(tokenizer, transformed)
    whole_offsets = offsets(tokenizer, transformed, whole_ids)
    prefix_ids = encode(tokenizer, prefix)
    replacement_ids = encode(tokenizer, example.new)
    suffix_ids = encode(tokenizer, suffix)
    piece_ids = prefix_ids + replacement_ids + suffix_ids
    whole_relevant_ids = token_window_from_offsets(whole_ids, whole_offsets, start, end)
    piece_relevant_ids = (
        whole_relevant_ids
        if whole_ids == piece_ids
        else token_window_from_piece_boundary(prefix_ids, replacement_ids, suffix_ids)
    )

    return {
        "name": example.name,
        "note": example.note,
        "old": example.old,
        "new": example.new,
        "original": example.original,
        "transformed": transformed,
        "whole_ids": whole_ids,
        "piece_ids": piece_ids,
        "whole_relevant_ids": whole_relevant_ids,
        "piece_relevant_ids": piece_relevant_ids,
        "differs": whole_ids != piece_ids,
        "replacement_span": [start, end],
    }


def md_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("\n", "<br>")
        .replace("|", "\\|")
        .replace("`", "\\`")
    )


def compact_ids(ids: list[int], *, max_len: int = 12) -> str:
    if len(ids) <= max_len:
        return "[" + ", ".join(str(x) for x in ids) + "]"
    head = ", ".join(str(x) for x in ids[: max_len // 2])
    tail = ", ".join(str(x) for x in ids[-max_len // 2 :])
    return f"[{head}, ..., {tail}]"


def write_markdown(path: Path, model: str, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    differing = sum(1 for row in rows if row["differs"])
    lines = [
        "# TransPLD Tokenization Boundary Check",
        "",
        f"- Tokenizer: `{model}`",
        f"- Examples checked: {len(rows)}",
        f"- Whole-reference and pieced tokenization differed: {differing}/{len(rows)}",
        "",
        "TransPLD tokenizes the full transformed reference before building its lookup index. "
        "This table compares that policy to a deliberately unsafe alternative that tokenizes "
        "`prefix`, `replacement`, and `suffix` separately and concatenates the IDs.",
        "",
        "| Example | Original text | Transformed text | Whole-tokenized relevant token IDs | Piece-tokenized relevant token IDs | Differ? |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    md_escape(str(row["name"])),
                    "`" + md_escape(str(row["original"])) + "`",
                    "`" + md_escape(str(row["transformed"])) + "`",
                    "`" + compact_ids(row["whole_relevant_ids"]) + "`",
                    "`" + compact_ids(row["piece_relevant_ids"]) + "`",
                    "yes" if row["differs"] else "no",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The relevant IDs include the replacement span plus one neighboring token on each side. "
            "A `yes` row means pieced tokenization would build a different transformed-view token stream "
            "from the one the target tokenizer assigns to the actual transformed text.",
            "",
            "<details><summary>Raw JSON rows</summary>",
            "",
            "```json",
            json.dumps(rows, indent=2),
            "```",
            "",
            "</details>",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare whole-reference TransPLD tokenization with unsafe pieced "
            "replacement tokenization for Qwen-style tokenizers."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face tokenizer name")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Markdown table path to write",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=4,
        help="Maximum examples to include; prioritizes rows where tokenization differs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = [analyze_example(tokenizer, example) for example in EXAMPLES]
    rows.sort(key=lambda row: (not row["differs"], row["name"]))
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    write_markdown(args.output, args.model, rows)
    differing = sum(1 for row in rows if row["differs"])
    print(f"wrote {args.output} with {differing}/{len(rows)} differing examples")
    if differing == 0:
        print("warning: no tokenization-boundary difference appeared in the checked examples")


if __name__ == "__main__":
    main()
