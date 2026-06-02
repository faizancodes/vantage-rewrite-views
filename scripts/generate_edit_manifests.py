"""Generate manifest JSONL workloads for VANTAGE edit-drift experiments.

The generated manifests are intentionally metadata-rich.  The evaluator only
needs ``task_id`` and ``prompt``; the analysis scripts use the extra fields to
aggregate by realized drift, copy ratio, hunk count, and prompt-oracle mode.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import keyword
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asts.humaneval import (  # noqa: E402
    Problem,
    _iter_dataset_rows,
    _line_offsets,
    _load_codeeditor_polish,
    _load_codeeditor_translate,
)


IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
ATTR_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")
DOTTED_CALL_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\s*\("
)
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])([0-9]+(?:\.[0-9]+)?)(?![A-Za-z0-9_])")


def _json_dump(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def _apply_token_map(text: str, mapping: dict[str, str]) -> str:
    out = text
    for old, new in sorted(mapping.items(), key=lambda item: -len(item[0])):
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\.]*|[0-9]+(?:\.[0-9]+)?", old):
            out = re.sub(
                rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])",
                new,
                out,
            )
        else:
            out = out.replace(old, new)
    return out


def _changed_hunks(reference: str, target: str) -> int:
    matcher = difflib.SequenceMatcher(
        a=reference.splitlines(),
        b=target.splitlines(),
        autojunk=False,
    )
    return sum(1 for tag, *_ in matcher.get_opcodes() if tag != "equal")


def _token_stats(reference: str, target: str) -> dict[str, Any]:
    ref_tokens = re.findall(r"\w+|[^\w\s]", reference)
    tgt_tokens = re.findall(r"\w+|[^\w\s]", target)
    matcher = difflib.SequenceMatcher(a=ref_tokens, b=tgt_tokens, autojunk=False)
    copied = 0
    longest = 0
    distance = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            span = j2 - j1
            copied += span
            longest = max(longest, span)
        elif tag == "replace":
            distance += max(i2 - i1, j2 - j1)
        else:
            distance += (i2 - i1) + (j2 - j1)
    output_len = len(tgt_tokens)
    return {
        "reference_tokens": len(ref_tokens),
        "output_tokens": output_len,
        "copied_token_percentage": copied / output_len if output_len else 0.0,
        "edit_distance_tokens": distance,
        "longest_unchanged_span_tokens": longest,
        "changed_hunk_count": _changed_hunks(reference, target),
    }


def _bucket(value: float, cuts: tuple[float, float], labels: tuple[str, str, str]) -> str:
    if value < cuts[0]:
        return labels[0]
    if value < cuts[1]:
        return labels[1]
    return labels[2]


def _function_sources(seed: int, max_rows: int) -> list[tuple[str, str, str]]:
    rng = random.Random(seed)
    out: list[tuple[str, str, str]] = []
    for source_name in ("the-stack-smol", "codeparrot"):
        try:
            rows = _iter_dataset_rows(source_name)
            for i, row in enumerate(rows):
                if i % 3 != rng.randrange(3):
                    continue
                content = row.get("content") or row.get("code") or row.get("text") or ""
                if not isinstance(content, str) or len(content) < 500:
                    continue
                for fn_idx, fn in enumerate(_extract_python_functions(content)):
                    if 80 <= len(fn) <= 6000:
                        out.append((f"{source_name}/{i}/{fn_idx}", fn, source_name))
                    if len(out) >= max_rows:
                        return out
        except Exception:
            if out:
                return out
            continue
    return out


def _extract_python_functions(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    offsets = _line_offsets(source)
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and getattr(node, "end_lineno", None)
        and node.body
    ]
    nodes.sort(key=lambda node: (node.lineno, node.name))
    out: list[str] = []
    for node in nodes:
        start = int(node.lineno)
        end = int(node.end_lineno or start)
        if end - start < 5 or end >= len(offsets):
            continue
        text = source[offsets[start - 1] : offsets[end]].rstrip() + "\n"
        if "return " in text:
            out.append(text)
    return out


def _identifier_candidates(reference: str) -> list[tuple[str, int]]:
    try:
        tree = ast.parse(reference)
    except SyntaxError:
        return []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            names.append(node.arg)
        elif isinstance(node, ast.Name):
            names.append(node.id)
    counts = Counter(
        name
        for name in names
        if name
        and name not in {"self", "cls"}
        and name not in keyword.kwlist
        and not name.startswith("__")
        and not name.endswith("__")
        and len(name) >= 2
    )
    ranked = [
        (name, count)
        for name, count in counts.items()
        if count >= 2 and not re.search(rf"\b{re.escape(name)}_updated\b", reference)
    ]
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked


def _rename_map(reference: str, *, max_names: int, suffix: str = "_updated") -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name, _ in _identifier_candidates(reference):
        if len(mapping) >= max_names:
            break
        new = f"{name}{suffix}"
        if not re.search(rf"\b{re.escape(new)}\b", reference):
            mapping[name] = new
    return mapping


def _rename_map_for_pct(reference: str, requested_pct: int) -> dict[str, str]:
    if requested_pct <= 0:
        return {}
    total_id_tokens = max(1, len(IDENT_RE.findall(reference)))
    selected: dict[str, str] = {}
    for name, count in _identifier_candidates(reference):
        selected[name] = f"{name}_updated"
        realized = 100.0 * sum(
            len(re.findall(rf"\b{re.escape(old)}\b", reference)) for old in selected
        ) / total_id_tokens
        if realized >= requested_pct:
            break
    return selected


def _rename_prompt(reference: str, mapping: dict[str, str], *, family: str) -> str:
    if mapping:
        pairs = ", ".join(f"`{old}` to `{new}`" for old, new in mapping.items())
        instruction = (
            f"Edit the Python function below. Rename {pairs} everywhere they appear. "
            "Preserve every other token exactly. Output only the full edited function."
        )
    else:
        instruction = (
            "Copy the Python function below exactly. Preserve every token exactly. "
            "Output only the full function."
        )
    return f"{instruction}\n\n```python\n{reference.rstrip()}\n```\n\nEdited function:\n"


def _manifest_row(
    *,
    task_id: str,
    prompt: str,
    reference: str,
    deterministic_target: str,
    language: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    stats = _token_stats(reference, deterministic_target)
    row = {
        "task_id": task_id,
        "prompt": prompt,
        "reference": reference,
        "deterministic_target": deterministic_target,
        "language": language,
        **metadata,
        **stats,
    }
    row["rename_percentage_realized"] = 100.0 * (
        1.0 - float(row.get("copied_token_percentage", 0.0))
    )
    row["drift_intensity"] = row["rename_percentage_realized"]
    row["edit_distance_bucket"] = _bucket(
        float(row["edit_distance_tokens"]),
        (8.0, 24.0),
        ("small", "medium", "large"),
    )
    row["span_bucket"] = _bucket(
        float(row["longest_unchanged_span_tokens"]),
        (48.0, 128.0),
        ("short", "medium", "long"),
    )
    return row


def _controlled_rows(
    functions: list[tuple[str, str, str]],
    *,
    axis: str,
    cell: Any,
    per_cell: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_id, reference, _ in functions:
        if axis == "renamepct":
            mapping = _rename_map_for_pct(reference, int(cell))
        elif axis == "identcount":
            mapping = _rename_map(reference, max_names=int(cell))
        elif axis == "hunks":
            mapping = _rename_map(reference, max_names=max(1, int(cell)))
        else:
            mapping = _rename_map(reference, max_names=2)
        if axis != "renamepct" or int(cell) > 0:
            if not mapping:
                continue
        target = _apply_token_map(reference, mapping)
        row = _manifest_row(
            task_id=f"drift_{axis}/{cell}/{source_id}",
            prompt=_rename_prompt(reference, mapping, family=axis),
            reference=reference,
            deterministic_target=target,
            language="repo_edit_rename_python",
            metadata={
                "drift_family": "rename",
                "axis": axis,
                "requested_cell": str(cell),
                "rename_count": len(mapping),
                "rewrite_pairs": mapping,
                "source_id": source_id,
                "target_is_reference": not bool(mapping),
            },
        )
        if axis == "span" and row["span_bucket"] != cell:
            continue
        if axis == "editdist" and row["edit_distance_bucket"] != cell:
            continue
        if axis == "hunks" and int(row["changed_hunk_count"]) < min(int(cell), 2):
            continue
        rows.append(row)
        if len(rows) >= per_cell:
            break
    return rows


def _first_nonrename_map(reference: str, family: str) -> dict[str, str]:
    if family == "field_rename":
        for _, attr in ATTR_RE.findall(reference):
            if attr not in {"append", "items", "keys", "values", "strip", "lower", "upper"}:
                return {f".{attr}": f".{attr}_updated"}
    if family == "api_migration":
        match = DOTTED_CALL_RE.search(reference)
        if match:
            old = match.group(1)
            base = old.split(".")[0]
            return {old: f"{base}.responses.create"}
    if family == "style_rewrite":
        for name, _ in _identifier_candidates(reference):
            if "_" in name:
                parts = name.split("_")
                return {name: parts[0] + "".join(p.title() for p in parts[1:])}
            if re.search(r"[a-z][A-Z]", name):
                snake = re.sub(r"(?<!^)([A-Z])", r"_\1", name).lower()
                return {name: snake}
    if family == "literal_config":
        match = NUMBER_RE.search(reference)
        if match:
            old = match.group(1)
            try:
                new = str(int(float(old)) + 1)
            except Exception:
                new = "1"
            if new != old:
                return {old: new}
    if family == "library_alias":
        for old, new in (("np.", "torch."), ("pd.", "pl."), ("math.", "np.")):
            if old in reference:
                return {old: new}
    if family == "namespace_migration":
        match = re.search(r"\b([A-Z][A-Za-z0-9_]*)\.", reference)
        if match:
            old = match.group(1)
            return {old: f"New{old}"}
    return {}


def _nonrename_prompt(reference: str, mapping: dict[str, str], family: str) -> str:
    pairs = ", ".join(f"`{old}` to `{new}`" for old, new in mapping.items())
    return (
        f"Edit the Python function below. Apply this {family.replace('_', ' ')}: "
        f"replace {pairs}. Preserve every other token exactly. "
        "Output only the full edited function.\n\n"
        f"```python\n{reference.rstrip()}\n```\n\nEdited function:\n"
    )


def _nonrename_rows(
    functions: list[tuple[str, str, str]],
    *,
    families: Iterable[str],
    per_family: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for source_id, reference, _ in functions:
        for family in families:
            if counts[family] >= per_family:
                continue
            mapping = _first_nonrename_map(reference, family)
            if not mapping:
                continue
            target = _apply_token_map(reference, mapping)
            if target == reference:
                continue
            rows.append(
                _manifest_row(
                    task_id=f"drift_nonrename/{family}/{source_id}",
                    prompt=_nonrename_prompt(reference, mapping, family),
                    reference=reference,
                    deterministic_target=target,
                    language="repo_edit_rename_python",
                    metadata={
                        "drift_family": family,
                        "axis": "nonrename",
                        "requested_cell": family,
                        "rename_count": 0,
                        "rewrite_pairs": mapping,
                        "source_id": source_id,
                    },
                )
            )
            counts[family] += 1
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as f:
        for row in rows:
            f.write(_json_dump(row) + "\n")
            count += 1
    return count


def _write_codeeditor_manifest(path: Path, problems: list[Problem]) -> int:
    rows = []
    for problem in problems:
        rows.append(
            {
                "task_id": problem.task_id,
                "prompt": problem.prompt,
                "reference": problem.reference,
                "deterministic_target": problem.deterministic_target,
                "language": problem.language,
                **problem.metadata,
            }
        )
    return _write_jsonl(path, rows)


def _prompt_oracle_rows(selected: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in selected[:max_rows]:
        base_id = row["task_id"].replace("/", "_")
        for mode in ("original", "augmented_transformed_reference", "oracle_transformed_reference"):
            prompt = row["prompt"]
            if mode == "augmented_transformed_reference":
                prompt = (
                    prompt
                    + "\nTransformed reference for prompt-augmentation oracle:\n"
                    + "```python\n"
                    + row["deterministic_target"].rstrip()
                    + "\n```\n\nEdited function:\n"
                )
            elif mode == "oracle_transformed_reference":
                prompt = (
                    "Copy the already transformed Python function below exactly. "
                    "Output only the full function.\n\n```python\n"
                    + row["deterministic_target"].rstrip()
                    + "\n```\n\nEdited function:\n"
                )
            new_row = dict(row)
            new_row["task_id"] = f"prompt_oracle/{mode}/{base_id}"
            new_row["prompt"] = prompt
            new_row["prompt_mode"] = mode
            new_row["oracle_includes_target"] = mode != "original"
            out.append(new_row)
    return out


def _validate_no_target_leak(rows: Iterable[dict[str, Any]]) -> None:
    for row in rows:
        if row.get("oracle_includes_target"):
            continue
        if row.get("target_is_reference"):
            continue
        target = str(row.get("deterministic_target") or "").strip()
        prompt = str(row.get("prompt") or "")
        if target and len(target) > 80 and target in prompt:
            raise ValueError(f"target leaked into prompt for {row.get('task_id')}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="research/asts-spec/data/manifests")
    p.add_argument("--triage-per-cell", type=int, default=20)
    p.add_argument("--full-per-cell", type=int, default=50)
    p.add_argument("--repo-pool-size", type=int, default=2500)
    p.add_argument("--seed", type=int, default=90210)
    p.add_argument("--codeeditor-n", type=int, default=80)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    functions = _function_sources(args.seed, args.repo_pool_size)
    if not functions:
        raise RuntimeError("no repository functions found for manifest generation")

    axes: dict[str, list[Any]] = {
        "renamepct": [0, 10, 25, 50, 100],
        "identcount": [1, 2, 4, 8],
        "hunks": [1, 2, 4, 8],
        "span": ["short", "medium", "long"],
        "editdist": ["small", "medium", "large"],
    }
    all_drift_rows: list[dict[str, Any]] = []
    for axis, cells in axes.items():
        rows: list[dict[str, Any]] = []
        for cell in cells:
            rows.extend(
                _controlled_rows(
                    functions,
                    axis=axis,
                    cell=cell,
                    per_cell=args.triage_per_cell,
                )
            )
        _validate_no_target_leak(rows)
        count = _write_jsonl(output_dir / f"drift_axis_{axis}.jsonl", rows)
        print(f"wrote drift_axis_{axis}.jsonl rows={count}")
        all_drift_rows.extend(rows)

    nonrename_rows = _nonrename_rows(
        functions,
        families=[
            "api_migration",
            "field_rename",
            "style_rewrite",
            "literal_config",
            "library_alias",
            "namespace_migration",
        ],
        per_family=args.triage_per_cell,
    )
    _validate_no_target_leak(nonrename_rows)
    print(f"wrote drift_nonrename.jsonl rows={_write_jsonl(output_dir / 'drift_nonrename.jsonl', nonrename_rows)}")

    oracle_seed = [
        row
        for row in all_drift_rows + nonrename_rows
        if row.get("drift_family") != "rename" or row.get("requested_cell") in {"25", "50", "100"}
    ]
    oracle_rows = _prompt_oracle_rows(oracle_seed, max_rows=args.triage_per_cell)
    print(f"wrote prompt_oracle_selected.jsonl rows={_write_jsonl(output_dir / 'prompt_oracle_selected.jsonl', oracle_rows)}")

    translate = _load_codeeditor_translate(
        n=args.codeeditor_n,
        source_lang="java",
        target_lang="cpp",
    )
    if not translate:
        translate = _load_codeeditor_translate(n=args.codeeditor_n)
    print(f"wrote codeeditor_translate80.jsonl rows={_write_codeeditor_manifest(output_dir / 'codeeditor_translate80.jsonl', translate)}")

    polish = _load_codeeditor_polish(
        n=args.codeeditor_n,
        language_filter="cpp",
    )
    if not polish:
        polish = _load_codeeditor_polish(n=args.codeeditor_n)
    print(f"wrote codeeditor_polish80.jsonl rows={_write_codeeditor_manifest(output_dir / 'codeeditor_polish80.jsonl', polish)}")


if __name__ == "__main__":
    main()
