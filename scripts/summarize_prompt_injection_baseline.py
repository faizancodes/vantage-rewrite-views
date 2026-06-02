"""Summarize visible transformed-reference prompt-injection pilot runs.

This summarizes a deliberately changed-prompt baseline.  It is not a hidden
TransPLD decoder and should not be compared as greedy-equivalence evidence for
the original prompt.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("artifacts/vantage_transpld/modal/prompt_injection_20260515_v1")
DEFAULT_OUTPUT_DIR = Path("artifacts/vantage_transpld/tables/prompt_injection_20260515_v1")
DEFAULT_ORIGINAL_ROOT = Path("artifacts/vantage_transpld/modal/validation_20260515_v1")


def _round(value: Any, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _workload_from_run_tag(run_tag: str) -> str:
    lowered = run_tag.lower()
    if "field" in lowered:
        return "Field substitution"
    if "style" in lowered:
        return "Identifier-style substitution"
    if "mixed" in lowered:
        return "Mixed"
    if "zero" in lowered:
        return "Zero drift"
    return run_tag


def _workload_key(run_tag: str) -> str:
    lowered = run_tag.lower()
    if "field" in lowered:
        return "field"
    if "style" in lowered:
        return "style"
    if "mixed" in lowered:
        return "mixed"
    if "zero" in lowered:
        return "zero"
    return lowered


def _local_manifest_path(remote_path: str | None) -> Path | None:
    if not remote_path:
        return None
    prefix = "/root/asts-spec/"
    if remote_path.startswith(prefix):
        return Path(remote_path[len(prefix) :])
    return Path(remote_path)


def _manifest_summary(local_manifest: Path | None) -> dict[str, Any]:
    if local_manifest is None:
        return {}
    summary_path = local_manifest.with_suffix(local_manifest.suffix + ".summary.json")
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text())


_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(?P<code>.*?)```", re.DOTALL | re.IGNORECASE)
_IDENT_CHARS = r"A-Za-z0-9_"


def _output_text(output: dict[str, Any]) -> str:
    return str(output.get("raw_text") if output.get("raw_text") is not None else output.get("text") or "")


def _extract_code(text: str) -> str:
    match = _FENCE_RE.search(text)
    if match:
        return match.group("code").strip()
    return text.strip()


def _syntax_ok(text: str) -> bool:
    code = _extract_code(text)
    if not code:
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _boundary_pattern(term: str) -> re.Pattern[str]:
    """Match a rewrite term without treating identifier substrings as hits."""

    term_s = str(term)
    escaped = re.escape(term_s)
    if term_s.startswith(".") and re.search(r"[A-Za-z0-9_]", term_s):
        return re.compile(rf"{escaped}(?![{_IDENT_CHARS}])")
    if re.search(r"[A-Za-z0-9_]", term_s):
        return re.compile(rf"(?<![{_IDENT_CHARS}]){escaped}(?![{_IDENT_CHARS}])")
    return re.compile(escaped)


def _contains_boundary(text: str, term: str) -> bool:
    return bool(_boundary_pattern(term).search(text))


def _rewrite_compliant(text: str, pairs: dict[str, str]) -> bool | None:
    if not pairs:
        return None
    code = _extract_code(text)
    # Use boundary-aware matching so dotted-field rewrites such as
    # `.add_ten -> .add_ten_updated` are not falsely marked noncompliant merely
    # because the old field spelling is a substring of the new field spelling.
    return all(_contains_boundary(code, str(new)) for new in pairs.values()) and not any(
        _contains_boundary(code, str(old)) for old in pairs.keys()
    )


def _row_quality(rows: list[dict[str, Any]], method: str = "vanilla") -> dict[str, Any]:
    total = 0
    exact = 0
    syntax = 0
    compliance_total = 0
    compliance = 0
    for row in rows:
        output = ((row.get("outputs") or {}).get(method) or {})
        if not output:
            continue
        total += 1
        text = _output_text(output)
        target = str(row.get("deterministic_target") or "").strip()
        if _extract_code(text).strip() == target:
            exact += 1
        syntax += int(_syntax_ok(text))
        pairs = row.get("rewrite_pairs") or (row.get("metadata") or {}).get("rewrite_pairs") or {}
        if isinstance(pairs, dict) and pairs:
            ok = _rewrite_compliant(text, pairs)
            if ok is not None:
                compliance_total += 1
                compliance += int(ok)
    return {
        "tasks": total,
        "exact_target_matches": exact,
        "syntax_valid": syntax,
        "rewrite_compliance_matches": compliance,
        "rewrite_compliance_tasks": compliance_total,
        "exact_target_rate": exact / total if total else None,
        "syntax_rate": syntax / total if total else None,
        "rewrite_compliance_rate": compliance / compliance_total if compliance_total else None,
    }


def _load_completion_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _original_rows_by_workload(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for path in sorted(root.glob("*/eval/completions.jsonl")):
        key = _workload_key(path.parent.parent.name)
        rows = _load_completion_rows(path)
        out[key] = {str(row.get("task_id")): row for row in rows}
    return out


def summarize_run(
    aggregate_path: Path,
    original_by_workload: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    data = json.loads(aggregate_path.read_text())
    meta = data.get("meta", {})
    by_method = data.get("by_method", {})
    vanilla = by_method.get("vanilla", {})
    pld = by_method.get("blazedit_pld_w128_n10", {})
    output_equiv = data.get("output_equivalence", {}).get("blazedit_pld_w128_n10", {})
    run_tag = aggregate_path.parent.parent.name
    workload_key = _workload_key(run_tag)
    manifest = _local_manifest_path(meta.get("problem_jsonl"))
    manifest_summary = _manifest_summary(manifest)
    completion_rows = _load_completion_rows(aggregate_path.parent / "completions.jsonl")
    quality = _row_quality(completion_rows, method="vanilla")
    original_rows = (original_by_workload or {}).get(workload_key, {})
    changed = 0
    comparable = 0
    for row in completion_rows:
        task_id = str(row.get("task_id"))
        original = original_rows.get(task_id)
        if not original:
            continue
        injected_text = _output_text(((row.get("outputs") or {}).get("vanilla") or {}))
        original_text = _output_text(((original.get("outputs") or {}).get("vanilla") or {}))
        comparable += 1
        changed += int(injected_text != original_text)
    vanilla_tok_s = vanilla.get("tokens_per_sec")
    pld_tok_s = pld.get("tokens_per_sec")
    ratio = None
    if vanilla_tok_s and pld_tok_s:
        ratio = float(pld_tok_s) / float(vanilla_tok_s)

    tasks = output_equiv.get("tasks")
    matches = output_equiv.get("matches_vanilla")
    parity = None
    if tasks:
        parity = float(matches or 0) / float(tasks)

    return {
        "run_tag": run_tag,
        "workload": _workload_from_run_tag(run_tag),
        "workload_key": workload_key,
        "n": int(meta.get("n_problems") or tasks or 0),
        "target": meta.get("target"),
        "dtype": meta.get("dtype"),
        "attention_backend": meta.get("attn_impl"),
        "max_new_tokens": meta.get("max_new_tokens"),
        "manifest": str(manifest) if manifest else meta.get("problem_jsonl"),
        "manifest_rows": manifest_summary.get("rows"),
        "prompt_injection_applied": manifest_summary.get("prompt_injection_applied"),
        "transformed_reference_equals_target": manifest_summary.get(
            "transformed_reference_equals_target"
        ),
        "vanilla_tok_s": _round(vanilla_tok_s),
        "pld_tok_s": _round(pld_tok_s),
        "pld_over_vanilla": _round(ratio),
        "vanilla_steps": vanilla.get("n_steps"),
        "pld_steps": pld.get("n_steps"),
        "generated_tokens": pld.get("n_emitted_total") or vanilla.get("n_emitted_total"),
        "pld_mean_accepted_drafts_per_step": _round(
            pld.get("mean_accepted_drafts_per_step")
        ),
        "pld_parity_vs_vanilla": {
            "matches": matches,
            "tasks": tasks,
            "rate": _round(parity),
        },
        "vanilla_quality": quality,
        "injection_output_delta_vs_original_vanilla": {
            "changed": changed,
            "tasks": comparable,
            "rate": _round(changed / comparable if comparable else None),
        },
        "aggregate_path": str(aggregate_path),
        "completions_path": str(aggregate_path.parent / "completions.jsonl"),
        "steps_path": str(aggregate_path.parent / "steps.jsonl"),
    }


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Visible Transformed-Reference Prompt-Injection Pilot",
        "",
        "This is a changed-prompt baseline, not the VANTAGE hidden internal-view decoder. "
        "The manifest visibly inserts the transformed reference into the model prompt, so "
        "the target greedy function can change. On the measured controlled rows, the "
        "transformed reference equals the deterministic target in the manifest summaries; "
        "therefore this is a serious practical alternative only when prompt modification "
        "is allowed, not headline fixed-prompt evidence for TransPLD.",
        "",
        "| Workload | n | Manifest injected rows | Manifest equals-target rows | Vanilla tok/s | PLD tok/s | PLD/vanilla | Vanilla steps | PLD steps | PLD parity vs vanilla | Raw aggregate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        parity = row["pld_parity_vs_vanilla"]
        parity_s = "unavailable"
        if parity.get("matches") is not None and parity.get("tasks") is not None:
            parity_s = f"{parity['matches']}/{parity['tasks']}"
        lines.append(
            "| {workload} | {n} | {applied} | {equals_target} | {vanilla_tok_s} | "
            "{pld_tok_s} | {ratio} | {vanilla_steps} | {pld_steps} | {parity} | `{aggregate}` |".format(
                workload=row["workload"],
                n=row["n"],
                applied=row.get("prompt_injection_applied", "unavailable"),
                equals_target=row.get("transformed_reference_equals_target", "unavailable"),
                vanilla_tok_s=row.get("vanilla_tok_s", "unavailable"),
                pld_tok_s=row.get("pld_tok_s", "unavailable"),
                ratio=row.get("pld_over_vanilla", "unavailable"),
                vanilla_steps=row.get("vanilla_steps", "unavailable"),
                pld_steps=row.get("pld_steps", "unavailable"),
                parity=parity_s,
                aggregate=row["aggregate_path"],
            )
        )
    lines.extend(
        [
            "",
            "## Prompt-Injection Quality And Output Delta",
            "",
            "| Workload | Injection exact target | Injection syntax | Injection rewrite compliance | Injection output differs from original-prompt vanilla | Injection PLD tok/s |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        q = row.get("vanilla_quality") or {}
        delta = row.get("injection_output_delta_vs_original_vanilla") or {}
        exact = (
            f"{q.get('exact_target_matches')}/{q.get('tasks')}"
            if q.get("tasks") is not None
            else "unavailable"
        )
        syntax = (
            f"{q.get('syntax_valid')}/{q.get('tasks')}"
            if q.get("tasks") is not None
            else "unavailable"
        )
        if q.get("rewrite_compliance_tasks"):
            compliance = f"{q.get('rewrite_compliance_matches')}/{q.get('rewrite_compliance_tasks')}"
        else:
            compliance = "n/a"
        changed = (
            f"{delta.get('changed')}/{delta.get('tasks')}"
            if delta.get("tasks") is not None
            else "unavailable"
        )
        lines.append(
            f"| {row['workload']} | {exact} | {syntax} | {compliance} | {changed} | {row.get('pld_tok_s', 'unavailable')} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: visible prompt injection can make exact PLD extremely fast because "
            "it changes the prompt to include the transformed answer-like reference. It should "
            "be treated as a changed-prompt practical baseline, not dismissed as a sanity check. "
            "The measured VANTAGE claim remains against tuned PLD on the original prompt, "
            "where TransPLD uses an internal transformed lookup view without changing the "
            "target greedy function.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="Root containing run/eval artifacts.")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    ap.add_argument(
        "--original-root",
        default=str(DEFAULT_ORIGINAL_ROOT),
        help="Original-prompt validation root used to compute output deltas.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    original_by_workload = _original_rows_by_workload(Path(args.original_root))
    rows = [
        summarize_run(path, original_by_workload)
        for path in sorted(root.glob("*/eval/aggregate.json"))
    ]
    rows.sort(key=lambda row: row["workload"])

    summary = {
        "schema": "vantage/prompt_injection_baseline_summary/v1",
        "root": str(root),
        "rows": rows,
        "interpretation": (
            "Changed-prompt visible transformed-reference baseline; not a hidden "
            "TransPLD result and not greedy-equivalence evidence for the original prompt."
        ),
    }
    (output_dir / "prompt_injection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (output_dir / "prompt_injection_summary.md").write_text(_markdown(rows))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
