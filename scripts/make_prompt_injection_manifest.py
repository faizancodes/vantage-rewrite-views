"""Build visible transformed-reference prompt-injection manifests.

This is a reviewer-baseline utility, not the VANTAGE method.  It modifies the
target model's visible prompt by inserting the reference after applying the
prompt-visible rewrite map, then leaves decoding to ordinary methods such as
``blazedit_pld_w128_n10``.  Because the prompt is changed, rows produced from
this manifest are not greedy-equivalence evidence for the original prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asts.code_proposers import apply_boundary_rewrites, extract_explicit_rewrites


EDITED_MARKER = "\n\nEdited function:\n"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _coerce_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for old, new in value.items():
        old_s = str(old).strip().rstrip(".,;:")
        new_s = str(new).strip().rstrip(".,;:")
        if old_s and new_s and old_s != new_s:
            out[old_s] = new_s
    return out


def _inject_transformed_reference(prompt: str, transformed_reference: str) -> str:
    block = (
        "\n\nTransformed reference after applying the requested rewrite "
        "(visible prompt-injection baseline; not used by VANTAGE):\n"
        "```python\n"
        f"{transformed_reference.rstrip()}\n"
        "```\n\n"
        "Edited function:\n"
    )
    if EDITED_MARKER in prompt:
        return prompt.rsplit(EDITED_MARKER, 1)[0] + block
    return prompt.rstrip() + block


def transform_row(row: dict[str, Any]) -> dict[str, Any]:
    prompt = str(row.get("prompt") or "")
    reference = str(row.get("reference") or "")
    rewrite_map = _coerce_map(row.get("rewrite_pairs")) or extract_explicit_rewrites(prompt)
    transformed_reference = apply_boundary_rewrites(reference, rewrite_map) if rewrite_map else reference
    applied = bool(reference and rewrite_map and transformed_reference != reference)

    out = dict(row)
    if applied:
        out["prompt"] = _inject_transformed_reference(prompt, transformed_reference)
    else:
        out["prompt"] = prompt

    out["prompt_injection_baseline"] = "visible_transformed_reference_v1"
    out["prompt_injection_applied"] = applied
    out["prompt_injection_note"] = (
        "Visible transformed-reference baseline: changes target prompt/model behavior; "
        "not a VANTAGE internal-view method."
    )
    out["prompt_injection_original_prompt_sha256"] = _sha256(prompt)
    out["prompt_injection_reference_sha256"] = _sha256(reference)
    out["prompt_injection_transformed_reference_sha256"] = _sha256(transformed_reference)
    out["prompt_injection_rewrite_map"] = rewrite_map
    out["prompt_injection_transformed_reference_equals_target"] = (
        transformed_reference == str(row.get("deterministic_target") or "")
    )
    out["visible_transformed_reference"] = transformed_reference if applied else ""
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL manifest.")
    ap.add_argument("--output", required=True, help="Output JSONL manifest.")
    ap.add_argument("--n", type=int, default=0, help="Optional maximum rows to emit.")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    applied = 0
    equals_target = 0
    with in_path.open() as f_in, out_path.open("w") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            row = transform_row(json.loads(line))
            total += 1
            applied += int(bool(row.get("prompt_injection_applied")))
            equals_target += int(bool(row.get("prompt_injection_transformed_reference_equals_target")))
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            if args.n and total >= args.n:
                break

    summary = {
        "input": str(in_path),
        "output": str(out_path),
        "rows": total,
        "prompt_injection_applied": applied,
        "transformed_reference_equals_target": equals_target,
        "schema": "vantage/prompt_injection_manifest/v1",
    }
    (out_path.with_suffix(out_path.suffix + ".summary.json")).write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
