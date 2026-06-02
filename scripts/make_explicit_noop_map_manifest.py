#!/usr/bin/env python3
"""Build an explicit no-op rewrite-map control manifest.

The control starts from the locked zero-drift/reference-target manifest and
adds prompt-visible rewrite instructions whose ``old`` term is guaranteed not
to occur in the reference. Applying the supported boundary-aware rewrite
therefore leaves the reference and its whole-reference tokenization unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asts.code_proposers import apply_boundary_rewrites, extract_explicit_rewrites  # noqa: E402


DEFAULT_SOURCE = ROOT / "data/manifests_frozen_audit/zero_drift100.jsonl"
DEFAULT_OUTPUT = ROOT / "data/manifests_frozen_audit/explicit_noop_map_100.jsonl"
DEFAULT_TOKENIZER = "Qwen/Qwen2.5-Coder-7B"
DEFAULT_REVISION = "0396a76181e127dfc13e5c5ec48a8cee09938b02"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _inject_instruction(prompt: str, old: str, new: str) -> str:
    instruction = (
        f"Rename {old} to {new}. This explicit rewrite map is part of the "
        "control prompt but should not change the reference.\n\n"
    )
    return instruction + prompt


def _make_row(row: dict[str, Any], idx: int, tokenizer: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    reference = str(row.get("reference") or "")
    if not reference:
        raise ValueError(f"row {idx} has no reference")

    old = f"__vantage_unused_old_{idx:03d}"
    new = f"__vantage_unused_new_{idx:03d}"
    if old in reference or new in reference:
        raise ValueError(f"row {idx} unexpectedly contains generated no-op term")

    prompt = _inject_instruction(str(row.get("prompt") or ""), old, new)
    pairs = extract_explicit_rewrites(prompt)
    expected = {old: new}
    if pairs != expected:
        raise ValueError(f"row {idx} extracted {pairs!r}, expected {expected!r}")

    rewritten = apply_boundary_rewrites(reference, pairs)
    if rewritten != reference:
        raise ValueError(f"row {idx} rewrite unexpectedly changed reference")

    reference_tokens = tokenizer.encode(reference, add_special_tokens=False)
    rewritten_tokens = tokenizer.encode(rewritten, add_special_tokens=False)
    if reference_tokens != rewritten_tokens:
        raise ValueError(f"row {idx} no-op rewrite changed tokenization")

    out = dict(row)
    metadata = dict(out.get("metadata") or {})
    metadata.update(
        {
            "explicit_noop_map_control": True,
            "noop_subtype": "non_applying_explicit_map",
            "source_task_id": row.get("task_id"),
            "target_is_reference": True,
        }
    )
    out.update(
        {
            "task_id": f"explicit_noop_map/{idx:03d}/{row.get('task_id', idx)}",
            "prompt": prompt,
            "reference": reference,
            "deterministic_target": reference,
            "target_is_reference": True,
            "rewrite_pairs": expected,
            "metadata": metadata,
        }
    )

    audit = {
        "task_id": out["task_id"],
        "source_task_id": row.get("task_id"),
        "old": old,
        "new": new,
        "reference_tokens": len(reference_tokens),
        "prompt_visible_map": True,
        "rewrite_changed_reference": False,
        "tokenization_changed": False,
    }
    return out, audit


def _write_audit(output: Path, audits: list[dict[str, Any]], *, source: Path, tokenizer_name: str) -> None:
    summary = {
        "schema": "vantage_explicit_noop_map_manifest_audit_v1",
        "source_manifest": str(source),
        "output_manifest": str(output),
        "tokenizer": tokenizer_name,
        "n": len(audits),
        "prompt_visible_maps": sum(1 for row in audits if row["prompt_visible_map"]),
        "rewrite_changed_reference": sum(1 for row in audits if row["rewrite_changed_reference"]),
        "tokenization_changed": sum(1 for row in audits if row["tokenization_changed"]),
        "rows": audits,
    }
    json_path = output.with_suffix(".audit.json")
    md_path = output.with_suffix(".audit.md")
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    md_path.write_text(
        "\n".join(
            [
                "# Explicit No-Op Map Manifest Audit",
                "",
                f"- Source manifest: `{source}`",
                f"- Output manifest: `{output}`",
                f"- Tokenizer: `{tokenizer_name}`",
                f"- Rows: {summary['n']}",
                f"- Prompt-visible maps: {summary['prompt_visible_maps']}/{summary['n']}",
                f"- Rewrite changed reference: {summary['rewrite_changed_reference']}",
                f"- Tokenization changed: {summary['tokenization_changed']}",
                "",
            ]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()

    rows = _load_jsonl(args.source)
    if len(rows) < args.n:
        raise SystemExit(f"source has {len(rows)} rows, need {args.n}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)
    out_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for idx, row in enumerate(rows[: args.n]):
        out, audit = _make_row(row, idx, tokenizer)
        out_rows.append(out)
        audits.append(audit)

    _write_jsonl(args.output, out_rows)
    _write_audit(args.output, audits, source=args.source, tokenizer_name=args.tokenizer)
    print(f"wrote {args.output} rows={len(out_rows)}")
    print(f"wrote {args.output.with_suffix('.audit.json')}")
    print(f"wrote {args.output.with_suffix('.audit.md')}")


if __name__ == "__main__":
    main()
