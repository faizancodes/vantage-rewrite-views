#!/usr/bin/env python3
"""Audit real-commit manifests for target-output leakage into searchable input.

The historical manifest field named ``reference`` is treated here as pre-edit
source context. The final PLD baseline searches the encoded prompt/prefix, but
we also audit the manifest ``reference`` field because older prose used the
ambiguous term "reference context".
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFESTS = [
    ROOT / "data" / "real_commits" / "real_commit_manifest_balanced_1000_v2_test500.jsonl",
    ROOT / "data" / "real_commits" / "real_commit_manifest_balanced_1000_v2_train500.jsonl",
]
DEFAULT_OUT = ROOT / "artifacts" / "dataset_leakage_audit.json"


@dataclass
class ManifestAudit:
    manifest: str
    tasks: int = 0
    exact_target_in_prompt: int = 0
    exact_target_in_pre_edit_context: int = 0
    exact_target_in_prompt_or_pre_edit_context: int = 0
    target_equals_pre_edit_context: int = 0
    target_substring_in_prompt: int = 0
    target_substring_in_pre_edit_context: int = 0
    large_target_chunk_in_prompt: int = 0
    large_target_chunk_in_pre_edit_context: int = 0
    patch_or_diff_marker_in_prompt: int = 0
    patch_or_diff_marker_in_pre_edit_context: int = 0
    patch_or_diff_marker_in_target: int = 0
    empty_target: int = 0
    empty_prompt: int = 0
    empty_pre_edit_context: int = 0
    examples: list[dict[str, Any]] | None = None


def _norm(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines()).strip()


def _target_chunks(text: str, *, min_chars: int = 240) -> list[str]:
    lines = [line for line in _norm(text).splitlines() if line.strip()]
    chunks: list[str] = []
    for i in range(len(lines)):
        buf: list[str] = []
        total = 0
        for line in lines[i:]:
            buf.append(line)
            total += len(line) + 1
            if total >= min_chars:
                chunks.append("\n".join(buf))
                break
    return chunks


def _has_patch_or_diff_marker(text: str) -> bool:
    lowered = text.lower()
    line_markers = ("\n@@ ", "\n+++ ", "\n--- ", "\n+ ", "\n- ")
    return "diff --git" in lowered or lowered.startswith("@@ ") or any(marker in lowered for marker in line_markers)


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                yield line_no, json.loads(line)


def _audit_manifest(path: Path, max_examples: int) -> ManifestAudit:
    result = ManifestAudit(manifest=str(path), examples=[])
    for line_no, row in _iter_jsonl(path):
        result.tasks += 1
        prompt = str(row.get("prompt") or "")
        pre_edit = str(row.get("reference") or "")
        target = str(row.get("deterministic_target") or "")
        if not prompt:
            result.empty_prompt += 1
        if not pre_edit:
            result.empty_pre_edit_context += 1
        if not target:
            result.empty_target += 1
            continue
        nprompt = _norm(prompt)
        npre = _norm(pre_edit)
        ntarget = _norm(target)
        flags: list[str] = []
        if ntarget and ntarget in nprompt:
            result.exact_target_in_prompt += 1
            flags.append("exact_target_in_prompt")
        if ntarget and ntarget in npre:
            result.exact_target_in_pre_edit_context += 1
            flags.append("exact_target_in_pre_edit_context")
        if ntarget and (ntarget in nprompt or ntarget in npre):
            result.exact_target_in_prompt_or_pre_edit_context += 1
        if ntarget and npre and ntarget == npre:
            result.target_equals_pre_edit_context += 1
            flags.append("target_equals_pre_edit_context")
        if len(ntarget) >= 160 and (ntarget[:160] in nprompt or ntarget[-160:] in nprompt):
            result.target_substring_in_prompt += 1
            flags.append("target_substring_in_prompt")
        if len(ntarget) >= 160 and (ntarget[:160] in npre or ntarget[-160:] in npre):
            result.target_substring_in_pre_edit_context += 1
            flags.append("target_substring_in_pre_edit_context")
        chunks = _target_chunks(ntarget)
        if chunks and any(chunk in nprompt for chunk in chunks):
            result.large_target_chunk_in_prompt += 1
            flags.append("large_target_chunk_in_prompt")
        if chunks and any(chunk in npre for chunk in chunks):
            result.large_target_chunk_in_pre_edit_context += 1
            flags.append("large_target_chunk_in_pre_edit_context")
        if _has_patch_or_diff_marker(nprompt):
            result.patch_or_diff_marker_in_prompt += 1
            flags.append("patch_or_diff_marker_in_prompt")
        if _has_patch_or_diff_marker(npre):
            result.patch_or_diff_marker_in_pre_edit_context += 1
            flags.append("patch_or_diff_marker_in_pre_edit_context")
        if _has_patch_or_diff_marker(ntarget):
            result.patch_or_diff_marker_in_target += 1
            flags.append("patch_or_diff_marker_in_target")
        suspicious_flags = [
            flag
            for flag in flags
            if flag
            in {
                "exact_target_in_prompt",
                "exact_target_in_pre_edit_context",
                "target_equals_pre_edit_context",
            }
        ]
        if suspicious_flags and result.examples is not None and len(result.examples) < max_examples:
            result.examples.append(
                {
                    "line_no": line_no,
                    "task_id": row.get("task_id"),
                    "repo": row.get("repo"),
                    "commit_sha": row.get("commit_sha"),
                    "file_path": row.get("file_path"),
                    "flags": suspicious_flags,
                    "prompt_prefix": prompt[:300],
                    "pre_edit_prefix": pre_edit[:300],
                    "target_prefix": target[:300],
                }
            )
    return result


def _write_markdown(report: dict[str, Any], json_path: Path) -> None:
    md_path = json_path.with_suffix(".md")
    lines = [
        "# Dataset Leakage Audit",
        "",
        "The manifest field historically named `reference` is audited as pre-edit source context, not as a gold final output.",
        "The final PLD baseline searches the prompt/generated prefix; this audit additionally checks the pre-edit context field to resolve reviewer ambiguity.",
        "Large chunk inclusion means an exact normalized contiguous target slice of at least 240 characters appears in the prompt or pre-edit context.",
        "",
        f"Overall pass: `{report['passed']}`",
        "",
        "| manifest | tasks | exact target in prompt | exact target in pre-edit context | pre-edit equals target | large shared target chunks | patch/diff markers prompt/context/target | empty target |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for item in report["manifests"]:
        lines.append(
            "| {manifest} | {tasks} | {exact_target_in_prompt} | {exact_target_in_pre_edit_context} | "
            "{target_equals_pre_edit_context} | {large_target_chunk_in_prompt} prompt / "
            "{large_target_chunk_in_pre_edit_context} context | {patch_or_diff_marker_in_prompt}/"
            "{patch_or_diff_marker_in_pre_edit_context}/{patch_or_diff_marker_in_target} | {empty_target} |".format(**item)
        )
    if report["examples"]:
        lines += ["", "## Flagged Examples", ""]
        for ex in report["examples"]:
            lines.append(f"- `{ex.get('task_id')}` line {ex.get('line_no')}: {', '.join(ex.get('flags', []))}")
    lines += [
        "",
        "Large shared target chunks are reported as natural copy-overlap signals in code editing, not automatic leakage. "
        "The pass/fail criterion is exact full-target inclusion or pre-edit equality with the gold target.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", help="Manifest JSONL path. May be repeated.")
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args()

    paths = [Path(p) for p in args.manifest] if args.manifest else DEFAULT_MANIFESTS
    audits = [_audit_manifest(path, args.max_examples) for path in paths]
    examples: list[dict[str, Any]] = []
    for audit in audits:
        examples.extend(audit.examples or [])
    passed = all(
        a.exact_target_in_prompt == 0
        and a.exact_target_in_pre_edit_context == 0
        and a.target_equals_pre_edit_context == 0
        for a in audits
    )
    report = {
        "passed": passed,
        "terminology": {
            "manifest_reference_field": "pre-edit source context",
            "gold_output_field": "deterministic_target",
            "paper_term": "input/source context",
        },
        "definitions": {
            "exact_target_in_prompt": "normalized full deterministic_target string appears in normalized prompt",
            "exact_target_in_pre_edit_context": "normalized full deterministic_target string appears in normalized reference/pre-edit context",
            "target_equals_pre_edit_context": "normalized deterministic_target exactly equals normalized reference/pre-edit context",
            "large_target_chunk": "contiguous normalized target line chunk of at least 240 characters appears verbatim in the audited input field",
            "patch_or_diff_marker": "diff --git, hunk header @@, or line-prefixed +++/---/+/- marker appears in normalized text",
        },
        "manifests": [asdict(a) | {"examples": None} for a in audits],
        "examples": examples[: args.max_examples],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report, out)
    print(f"wrote {out}")
    print(f"pass={passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
