"""HumanEval / MBPP prompt loader + completion-extraction helpers.

We don't compute pass@1 — the prototype is about latency and acceptance
rate, not correctness — but we do reuse public benchmark prompts since
they're short, standard, and representative of code completion.

Supported benchmarks:
- HumanEval Python  via openai/openai_humaneval (~164 problems)
- HumanEval TypeScript via nuprl/MultiPL-E humaneval-ts (~159 problems)
- MBPP-Sanitized (Python)  via google-research-datasets/mbpp (~427 problems)
- CodeEditorBench Python debugging and requirement-switching subsets
"""

from __future__ import annotations

import ast
import json
import keyword
import re
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class Problem:
    task_id: str
    prompt: str
    language: str = "python"
    reference: str = ""
    deterministic_target: str = ""
    metadata: dict = field(default_factory=dict)


def load_problems(n: int | None = None, hf_split: str = "test") -> list[Problem]:
    """Load Python HumanEval prompts. (Backwards-compatible.)"""
    return _load_python(n=n, hf_split=hf_split)


def _load_python(
    n: int | None = None,
    hf_split: str = "test",
    prompt_variant: str = "full",
) -> list[Problem]:
    from datasets import load_dataset

    ds = load_dataset("openai_humaneval", split=hf_split)
    out: list[Problem] = []
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        prompt = _transform_humaneval_prompt(row["prompt"], prompt_variant)
        out.append(Problem(task_id=row["task_id"], prompt=prompt, language="python"))
    return out


def _load_typescript(n: int | None = None) -> list[Problem]:
    """Load TypeScript HumanEval prompts via MultiPL-E."""
    from datasets import load_dataset

    # MultiPL-E config "humaneval-ts" has ~159 problems
    ds = load_dataset("nuprl/MultiPL-E", "humaneval-ts", split="test")
    out: list[Problem] = []
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        out.append(Problem(
            task_id=row.get("name", f"ts/{i}"),
            prompt=row["prompt"],
            language="typescript",
        ))
    return out


def load_problems_from_jsonl(path: str, n: int | None = None) -> list[Problem]:
    """Load manifest-style edit/code problems from JSONL.

    The manifest path is the common surface for controlled drift sweeps,
    prompt-augmentation oracle rows, cross-language edit rows, and any
    future benchmark that should not become another hard-coded selector.
    Required fields are ``task_id`` and ``prompt``.  Optional fields are kept
    in ``metadata`` and also exposed through the first-class Problem fields
    used by workload/quality analyses.
    """
    out: list[Problem] = []
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = str(row.get("task_id") or f"{path}:{line_no}")
            prompt = str(row.get("prompt") or "")
            if not prompt:
                raise ValueError(f"{path}:{line_no}: missing prompt")
            language = str(row.get("language") or row.get("benchmark") or "manifest")
            metadata = {
                k: v
                for k, v in row.items()
                if k not in {"task_id", "prompt", "language", "reference", "deterministic_target"}
            }
            out.append(
                Problem(
                    task_id=task_id,
                    prompt=prompt,
                    language=language,
                    reference=str(row.get("reference") or ""),
                    deterministic_target=str(row.get("deterministic_target") or ""),
                    metadata=metadata,
                )
            )
            if n is not None and len(out) >= n:
                break
    return out


def _mbpp_to_completion_prompt(
    description: str,
    test_list: list[str],
    *,
    prompt_variant: str = "full",
) -> str:
    """Build a HumanEval-style code-completion prompt from an MBPP entry.

    MBPP rows give a natural-language description and a few assertions
    (test cases). We pack them into a docstring; the model generates
    `def fn(...):\\n    ...` from there. The first assertion is included
    because it pins down the function name + signature shape — without
    it the model often invents arbitrary names and the prompt becomes
    underspecified.
    """
    desc = (description or "").strip()
    example = (test_list[0] if test_list else "").strip()
    if example and prompt_variant != "desc_only":
        return f'"""\n{desc}\n{example}\n"""\n'
    return f'"""\n{desc}\n"""\n'


def _load_mbpp(n: int | None = None, prompt_variant: str = "full") -> list[Problem]:
    """Load MBPP-Sanitized problems as HumanEval-style completion prompts.

    Returns up to 427 problems (the full sanitized set across train+val+test
    splits). We don't care about train/test leakage — we're benchmarking
    decoding latency, not training a model.
    """
    from datasets import load_dataset

    out: list[Problem] = []
    # The `sanitized` config is the curated subset (~427 total). It splits
    # into train/validation/test/prompt; we concatenate everything except
    # the few-shot prompt set.
    for split in ("train", "validation", "test"):
        try:
            ds = load_dataset("google-research-datasets/mbpp", "sanitized", split=split)
        except Exception:
            continue
        for row in ds:
            if n is not None and len(out) >= n:
                return out
            desc = row.get("prompt") or row.get("text") or ""
            test_list = row.get("test_list") or []
            task_id = row.get("task_id")
            out.append(Problem(
                task_id=f"MBPP/{task_id}",
                prompt=_mbpp_to_completion_prompt(
                    desc,
                    test_list,
                    prompt_variant=prompt_variant,
                ),
                language="mbpp",
            ))
    return out


def _transform_humaneval_prompt(prompt: str, variant: str) -> str:
    if variant in {"", "full"}:
        return prompt
    if variant == "no_examples":
        return _remove_docstring_examples(prompt)
    if variant == "signature_only":
        return _signature_only_prompt(prompt)
    raise ValueError(f"unsupported HumanEval prompt variant: {variant}")


def _remove_docstring_examples(prompt: str) -> str:
    lines = prompt.splitlines(keepends=True)
    out: list[str] = []
    in_doc = False
    quote = ""
    for line in lines:
        stripped = line.strip()
        if not in_doc and ('"""' in line or "'''" in line):
            quote = '"""' if '"""' in line else "'''"
            in_doc = True
            out.append(line)
            # Single-line docstrings are rare in HumanEval; preserve them.
            if line.count(quote) >= 2:
                in_doc = False
            continue
        if in_doc:
            if quote in line:
                in_doc = False
                out.append(line)
                continue
            if _looks_like_example_line(stripped):
                continue
            out.append(line)
            continue
        out.append(line)
    return "".join(out)


def _looks_like_example_line(stripped: str) -> bool:
    if not stripped:
        return False
    if stripped.startswith((">>>", "...")):
        return True
    if stripped.startswith(("assert ", "print(")):
        return True
    if re.search(r"\b(assert|==|=>)\b", stripped):
        return True
    return False


def _signature_only_prompt(prompt: str) -> str:
    lines = prompt.splitlines(keepends=True)
    out: list[str] = []
    for line in lines:
        out.append(line)
        stripped = line.lstrip()
        if stripped.startswith(("def ", "async def ")) and line.rstrip().endswith(":"):
            break
    return "".join(out)


def _line_offsets(source: str) -> list[int]:
    offsets = [0]
    total = 0
    for line in source.splitlines(keepends=True):
        total += len(line)
        offsets.append(total)
    return offsets


def _iter_dataset_rows(name: str) -> Iterable[dict]:
    from datasets import load_dataset

    if name == "the-stack-smol":
        ds = load_dataset(
            "bigcode/the-stack-smol",
            data_dir="data/python",
            split="train",
            streaming=True,
        )
        for row in ds:
            lang = str(row.get("language", "")).lower()
            if not lang or lang in {"python", "py"}:
                yield row
    elif name == "codeparrot":
        ds = load_dataset(
            "codeparrot/github-code-clean",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        for row in ds:
            yield row
    else:
        raise ValueError(name)


def _load_repo_python(n: int | None = None, tokenizer=None, seed: int = 123) -> list[Problem]:
    """Build a small prefix-only same-file completion benchmark.

    The primary source is The Stack Smol Python rows.  If that dataset is
    unavailable in the execution environment, fall back to codeparrot clean and
    mark task ids accordingly.
    """
    out: list[Problem] = []
    sources = ("the-stack-smol", "codeparrot")
    for source_name in sources:
        try:
            iterator = _iter_dataset_rows(source_name)
            for i, row in enumerate(iterator):
                if i % max(1, seed % 17) != 0:
                    continue
                content = row.get("content") or row.get("code") or row.get("text") or ""
                if not isinstance(content, str):
                    continue
                problem = _repo_problem_from_source(content, len(out), source_name, tokenizer)
                if problem is None:
                    continue
                out.append(problem)
                if n is not None and len(out) >= n:
                    return out
            if out:
                return out
        except Exception:
            if out:
                return out
            continue
    return out


def _repo_problem_from_source(
    source: str,
    idx: int,
    source_name: str,
    tokenizer=None,
) -> Problem | None:
    if len(source) < 500:
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    if tokenizer is not None:
        n_tokens = len(tokenizer(source, add_special_tokens=False).input_ids)
        if n_tokens < 200 or n_tokens > 4000:
            return None
    elif len(source) < 800 or len(source) > 16000:
        return None

    offsets = _line_offsets(source)
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and getattr(node, "end_lineno", None)
        and node.body
    ]
    candidates.sort(key=lambda node: (node.lineno, node.name))
    for node in candidates:
        body_start_line = node.body[0].lineno
        end_line = int(node.end_lineno or body_start_line)
        if end_line - body_start_line < 3:
            continue
        cursor_line = body_start_line + max(1, (end_line - body_start_line) // 4)
        if cursor_line >= len(offsets):
            continue
        cursor = offsets[cursor_line - 1]
        prompt = source[:cursor]
        if tokenizer is not None:
            prompt_tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)
            if prompt_tokens < 64:
                continue
        elif len(prompt) < 256:
            continue
        return Problem(
            task_id=f"repo_python/{source_name}/{idx}",
            prompt=prompt,
            language="repo_python",
        )
    return None


def _load_repo_edit_python(n: int | None = None, tokenizer=None, seed: int = 321) -> list[Problem]:
    out: list[Problem] = []
    for source_name in ("the-stack-smol", "codeparrot"):
        try:
            iterator = _iter_dataset_rows(source_name)
            for i, row in enumerate(iterator):
                if i % max(1, seed % 19) != 0:
                    continue
                content = row.get("content") or row.get("code") or row.get("text") or ""
                if not isinstance(content, str):
                    continue
                problem = _repo_edit_problem_from_source(content, len(out), source_name, tokenizer)
                if problem is None:
                    continue
                out.append(problem)
                if n is not None and len(out) >= n:
                    return out
            if out:
                return out
        except Exception:
            if out:
                return out
            continue
    return out


def _load_repo_edit_rename_python(n: int | None = None, tokenizer=None, seed: int = 421) -> list[Problem]:
    out: list[Problem] = []
    for source_name in ("the-stack-smol", "codeparrot"):
        try:
            iterator = _iter_dataset_rows(source_name)
            for i, row in enumerate(iterator):
                if i % max(1, seed % 23) != 0:
                    continue
                content = row.get("content") or row.get("code") or row.get("text") or ""
                if not isinstance(content, str):
                    continue
                problem = _repo_edit_rename_problem_from_source(content, len(out), source_name, tokenizer)
                if problem is None:
                    continue
                out.append(problem)
                if n is not None and len(out) >= n:
                    return out
            if out:
                return out
        except Exception:
            if out:
                return out
            continue
    return out


def _repo_edit_problem_from_source(
    source: str,
    idx: int,
    source_name: str,
    tokenizer=None,
) -> Problem | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    offsets = _line_offsets(source)
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and getattr(node, "end_lineno", None)
        and node.body
    ]
    candidates.sort(key=lambda node: (node.lineno, node.name))
    for node in candidates:
        start_line = int(node.lineno)
        end_line = int(node.end_lineno or start_line)
        if end_line - start_line < 6 or end_line - start_line > 80:
            continue
        if end_line >= len(offsets):
            continue
        function_source = source[offsets[start_line - 1] : offsets[end_line]]
        if "return " not in function_source:
            continue
        if tokenizer is not None:
            n_tokens = len(tokenizer(function_source, add_special_tokens=False).input_ids)
            if n_tokens < 80 or n_tokens > 1200:
                continue
        prompt = (
            "Edit the Python function below. Add one short comment immediately "
            "before the first `return` statement. Preserve every other line exactly. "
            "Output only the full edited function.\n\n"
            "```python\n"
            f"{function_source.rstrip()}\n"
            "```\n\n"
            "Edited function:\n"
        )
        return Problem(
            task_id=f"repo_edit_python/{source_name}/{idx}",
            prompt=prompt,
            language="repo_edit_python",
        )
    return None


def _identifier_rename_pair(function_source: str) -> tuple[str, str] | None:
    try:
        tree = ast.parse(function_source)
    except SyntaxError:
        return None
    candidates: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            candidates.append(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Load)):
            candidates.append(node.id)
    seen: set[str] = set()
    ordered: list[str] = []
    for name in candidates:
        if (
            not name
            or name in seen
            or name.startswith("__")
            or name.endswith("__")
            or len(name) < 2
            or name in {"self", "cls"}
            or name in keyword.kwlist
        ):
            continue
        seen.add(name)
        ordered.append(name)
    for name in ordered:
        count = len(re.findall(rf"\b{re.escape(name)}\b", function_source))
        if count < 3:
            continue
        new_name = f"{name}_updated"
        if re.search(rf"\b{re.escape(new_name)}\b", function_source):
            new_name = f"{name}_renamed"
        if not re.search(rf"\b{re.escape(new_name)}\b", function_source):
            return name, new_name
    return None


def _repo_edit_rename_problem_from_source(
    source: str,
    idx: int,
    source_name: str,
    tokenizer=None,
) -> Problem | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    offsets = _line_offsets(source)
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and getattr(node, "end_lineno", None)
        and node.body
    ]
    candidates.sort(key=lambda node: (node.lineno, node.name))
    for node in candidates:
        start_line = int(node.lineno)
        end_line = int(node.end_lineno or start_line)
        if end_line - start_line < 6 or end_line - start_line > 80:
            continue
        if end_line >= len(offsets):
            continue
        function_source = source[offsets[start_line - 1] : offsets[end_line]]
        rename = _identifier_rename_pair(function_source)
        if rename is None:
            continue
        old_name, new_name = rename
        if tokenizer is not None:
            n_tokens = len(tokenizer(function_source, add_special_tokens=False).input_ids)
            if n_tokens < 80 or n_tokens > 1200:
                continue
        prompt = (
            "Edit the Python function below. Rename "
            f"`{old_name}` to `{new_name}` everywhere it appears in this function. "
            "Preserve every other token exactly. Output only the full edited function.\n\n"
            "```python\n"
            f"{function_source.rstrip()}\n"
            "```\n\n"
            "Edited function:\n"
        )
        return Problem(
            task_id=f"repo_edit_rename_python/{source_name}/{idx}",
            prompt=prompt,
            language="repo_edit_rename_python",
        )
    return None


def _strip_code_block_or_language_header(code: str, language: str = "python") -> str:
    text = (code or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    first, sep, rest = text.partition("\n")
    header = first.strip().lower()
    lang = language.lower()
    if sep and header in {lang, f"{lang}3", "py", "python", "python3"}:
        text = rest.strip()
    return text


def _iter_codeeditor_jsonl(file_name: str) -> Iterable[dict]:
    """Yield CodeEditorBench rows without pyarrow schema inference.

    Some CodeEditorBench JSONL files have mixed scalar types in optional
    metadata columns.  ``datasets.load_dataset("json")`` can fail when it
    tries to infer one Arrow schema for the whole file, so we read JSONL
    records directly from the Hugging Face cache.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="m-a-p/CodeEditorBench",
        filename=file_name,
        repo_type="dataset",
    )
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _load_codeeditor_python(n: int | None = None, tokenizer=None) -> list[Problem]:
    """Load a Python CodeEditorBench debugging subset.

    The Hugging Face repository mixes several schemas, so we load the debug
    JSONL files directly instead of asking the dataset builder to merge every
    task type.  Debugging is the closest real counterpart to the synthetic edit
    benchmark: a buggy reference program is given, the target output is a full
    corrected program, and unchanged spans are common but not artificially
    inserted by our benchmark generator.
    """
    out: list[Problem] = []
    files = ("code_debug_primary.jsonl", "code_debug_plus.jsonl")
    for file_name in files:
        for row in _iter_codeeditor_jsonl(file_name):
            lang = str(row.get("code_language", "")).lower()
            if lang not in {"python", "python3", "py"}:
                continue
            problem = _codeeditor_debug_problem_from_row(row, len(out), file_name, tokenizer)
            if problem is None:
                continue
            out.append(problem)
            if n is not None and len(out) >= n:
                return out
    return out


def _codeeditor_debug_problem_from_row(
    row: dict,
    idx: int,
    source_name: str,
    tokenizer=None,
) -> Problem | None:
    buggy = _strip_code_block_or_language_header(
        str(row.get("incorrect_solutions") or ""),
        "python",
    )
    if not buggy or len(buggy) < 80:
        return None
    if tokenizer is not None:
        n_tokens = len(tokenizer(buggy, add_special_tokens=False).input_ids)
        if n_tokens < 40 or n_tokens > 1600:
            return None
    bug_type = str(row.get("type") or "bug").strip()
    public_in = str(row.get("public_tests_input") or "").strip()
    public_out = str(row.get("public_tests_output") or "").strip()
    prompt = (
        "Fix the bug in the Python code below. Preserve unchanged code when possible. "
        "Output only the full corrected Python code, with no explanation.\n\n"
        f"Bug type: {bug_type}\n"
    )
    if public_in or public_out:
        prompt += (
            "\nPublic test:\n"
            f"Input:\n{public_in}\n"
            f"Expected output:\n{public_out}\n"
        )
    prompt += (
        "\n```python\n"
        f"{buggy.rstrip()}\n"
        "```\n\n"
        "Corrected code:\n"
    )
    row_id = row.get("idx", idx)
    return Problem(
        task_id=f"codeeditor_python/{source_name}/{row_id}",
        prompt=prompt,
        language="codeeditor_python",
        reference=buggy,
        deterministic_target=_strip_code_block_or_language_header(
            str(row.get("solutions") or ""),
            "python",
        ),
        metadata={
            "source_file": source_name,
            "category": "debug",
            "bug_type": bug_type,
            "public_tests_input": row.get("public_tests_input"),
            "public_tests_output": row.get("public_tests_output"),
            "private_tests_input": row.get("private_tests_input"),
            "private_tests_output": row.get("private_tests_output"),
        },
    )


def _load_codeeditor_switch_python(n: int | None = None, tokenizer=None) -> list[Problem]:
    """Load a Python CodeEditorBench requirement-switching subset.

    The prompt includes a similar/reference program and the new requirement,
    but never the corrected target program.  This gives a second real edit-like
    category beyond debugging while preserving the no-target-leakage boundary.
    """
    out: list[Problem] = []
    files = ("code_switch_primary.jsonl", "code_switch_plus.jsonl")
    for file_name in files:
        for row in _iter_codeeditor_jsonl(file_name):
            lang = str(row.get("language", "")).lower()
            if lang not in {"python", "python3", "py"}:
                continue
            problem = _codeeditor_switch_problem_from_row(row, len(out), file_name, tokenizer)
            if problem is None:
                continue
            out.append(problem)
            if n is not None and len(out) >= n:
                return out
    return out


def _codeeditor_switch_problem_from_row(
    row: dict,
    idx: int,
    source_name: str,
    tokenizer=None,
) -> Problem | None:
    reference = _strip_code_block_or_language_header(
        str(row.get("similar_source_code") or ""),
        "python",
    )
    if not reference or len(reference) < 80:
        return None
    if tokenizer is not None:
        n_tokens = len(tokenizer(reference, add_special_tokens=False).input_ids)
        if n_tokens < 40 or n_tokens > 1600:
            return None
    similar_req = str(row.get("similar_content") or "").strip()
    target_req = str(row.get("target_content") or "").strip()
    if not target_req:
        return None
    prompt = (
        "Rewrite the reference Python solution for the new requirement. "
        "Use the reference code as a guide, preserve reusable structure when "
        "it still applies, and output only the full adapted Python code with "
        "no explanation.\n\n"
    )
    if similar_req:
        prompt += f"Reference requirement:\n{similar_req}\n\n"
    prompt += (
        "Reference code:\n"
        "```python\n"
        f"{reference.rstrip()}\n"
        "```\n\n"
        f"New requirement:\n{target_req}\n\n"
        "Adapted code:\n"
    )
    row_id = row.get("idx", idx)
    return Problem(
        task_id=f"codeeditor_switch_python/{source_name}/{row_id}",
        prompt=prompt,
        language="codeeditor_switch_python",
        reference=reference,
        deterministic_target=_strip_code_block_or_language_header(
            str(row.get("target_source_code") or ""),
            "python",
        ),
        metadata={
            "source_file": source_name,
            "category": "switch",
            "target_requirement": target_req,
            "similar_requirement": similar_req,
            "public_tests_input": row.get("public_tests_input"),
            "public_tests_output": row.get("public_tests_output"),
            "private_tests_input": row.get("private_tests_input"),
            "private_tests_output": row.get("private_tests_output"),
        },
    )


def _load_codeeditor_translate(
    n: int | None = None,
    tokenizer=None,
    *,
    source_lang: str | None = None,
    target_lang: str | None = None,
) -> list[Problem]:
    out: list[Problem] = []
    files = ("code_translate_primary.jsonl", "code_translate_plus.jsonl")
    for file_name in files:
        for row in _iter_codeeditor_jsonl(file_name):
            src_lang = str(row.get("source_lang") or row.get("source_language") or "").lower()
            tgt_lang = str(row.get("target_lang") or row.get("target_language") or "").lower()
            if source_lang and src_lang != source_lang.lower():
                continue
            if target_lang and tgt_lang != target_lang.lower():
                continue
            problem = _codeeditor_translate_problem_from_row(row, len(out), file_name, tokenizer)
            if problem is None:
                continue
            out.append(problem)
            if n is not None and len(out) >= n:
                return out
    return out


def _codeeditor_translate_problem_from_row(
    row: dict,
    idx: int,
    source_name: str,
    tokenizer=None,
) -> Problem | None:
    source_lang = str(row.get("source_lang") or row.get("source_language") or "").strip()
    target_lang = str(row.get("target_lang") or row.get("target_language") or "").strip()
    source_code = _strip_code_block_or_language_header(
        str(row.get("source_code") or ""),
        source_lang or "code",
    )
    target_code = _strip_code_block_or_language_header(
        str(row.get("target_code") or ""),
        target_lang or "code",
    )
    if not source_code or len(source_code) < 80 or not target_lang:
        return None
    if tokenizer is not None:
        n_tokens = len(tokenizer(source_code, add_special_tokens=False).input_ids)
        if n_tokens < 40 or n_tokens > 2400:
            return None
    prompt = (
        f"Translate the {source_lang or 'source'} code below to {target_lang}. "
        f"Preserve structure where possible and output only the full translated {target_lang} code.\n\n"
        f"```{source_lang or ''}\n"
        f"{source_code.rstrip()}\n"
        "```\n\n"
        f"Translated {target_lang} code:\n"
    )
    row_id = row.get("idx", idx)
    return Problem(
        task_id=f"codeeditor_translate/{source_name}/{row_id}",
        prompt=prompt,
        language="codeeditor_translate",
        reference=source_code,
        deterministic_target=target_code,
        metadata={
            "source_file": source_name,
            "category": "translate",
            "source_lang": source_lang,
            "target_lang": target_lang,
            "public_tests_input": row.get("public_tests_input"),
            "public_tests_output": row.get("public_tests_output"),
            "private_tests_input": row.get("private_tests_input"),
            "private_tests_output": row.get("private_tests_output"),
        },
    )


def _load_codeeditor_polish(
    n: int | None = None,
    tokenizer=None,
    *,
    language_filter: str | None = None,
) -> list[Problem]:
    out: list[Problem] = []
    files = ("code_polishment_primary.jsonl", "code_polishment_plus.jsonl")
    for file_name in files:
        for row in _iter_codeeditor_jsonl(file_name):
            lang = str(row.get("source_lang") or row.get("code_language") or row.get("language") or "").lower()
            if language_filter and lang != language_filter.lower():
                continue
            problem = _codeeditor_polish_problem_from_row(row, len(out), file_name, tokenizer)
            if problem is None:
                continue
            out.append(problem)
            if n is not None and len(out) >= n:
                return out
    return out


def _codeeditor_polish_problem_from_row(
    row: dict,
    idx: int,
    source_name: str,
    tokenizer=None,
) -> Problem | None:
    language = str(row.get("source_lang") or row.get("code_language") or row.get("language") or "code").strip()
    source_code = _strip_code_block_or_language_header(
        str(row.get("source_code") or row.get("code") or ""),
        language,
    )
    target_code = _strip_code_block_or_language_header(
        str(row.get("target_code") or row.get("polished_code") or row.get("solution") or ""),
        language,
    )
    if not source_code or len(source_code) < 80:
        return None
    if tokenizer is not None:
        n_tokens = len(tokenizer(source_code, add_special_tokens=False).input_ids)
        if n_tokens < 40 or n_tokens > 2400:
            return None
    prompt = (
        f"Polish and optimize the {language} code below without changing its behavior. "
        "Preserve reusable structure and output only the full polished code.\n\n"
        f"```{language}\n"
        f"{source_code.rstrip()}\n"
        "```\n\n"
        "Polished code:\n"
    )
    row_id = row.get("idx", idx)
    return Problem(
        task_id=f"codeeditor_polish/{source_name}/{row_id}",
        prompt=prompt,
        language="codeeditor_polish",
        reference=source_code,
        deterministic_target=target_code,
        metadata={
            "source_file": source_name,
            "category": "polish",
            "source_lang": language,
            "public_tests_input": row.get("public_tests_input"),
            "public_tests_output": row.get("public_tests_output"),
            "private_tests_input": row.get("private_tests_input"),
            "private_tests_output": row.get("private_tests_output"),
            "average_runtime": row.get("average_runtime"),
            "average_memory": row.get("average_memory"),
        },
    )


def load_problems_for_language(
    language: str = "python",
    n: int | None = None,
    *,
    prompt_variant: str = "full",
    tokenizer=None,
) -> list[Problem]:
    """Load benchmark problems. Note: `language` is overloaded — it's
    really a benchmark selector.
    """
    if language == "python":
        return _load_python(n=n, prompt_variant=prompt_variant)
    elif language in ("ts", "typescript"):
        return _load_typescript(n=n)
    elif language == "mbpp":
        return _load_mbpp(n=n, prompt_variant=prompt_variant)
    elif language == "repo_python":
        return _load_repo_python(n=n, tokenizer=tokenizer)
    elif language == "repo_edit_python":
        return _load_repo_edit_python(n=n, tokenizer=tokenizer)
    elif language == "repo_edit_rename_python":
        return _load_repo_edit_rename_python(n=n, tokenizer=tokenizer)
    elif language == "codeeditor_python":
        return _load_codeeditor_python(n=n, tokenizer=tokenizer)
    elif language == "codeeditor_switch_python":
        return _load_codeeditor_switch_python(n=n, tokenizer=tokenizer)
    elif language == "codeeditor_translate":
        return _load_codeeditor_translate(n=n, tokenizer=tokenizer)
    elif language == "codeeditor_translate_javacpp":
        return _load_codeeditor_translate(
            n=n,
            tokenizer=tokenizer,
            source_lang="java",
            target_lang="cpp",
        )
    elif language == "codeeditor_polish":
        return _load_codeeditor_polish(n=n, tokenizer=tokenizer)
    elif language == "codeeditor_polish_cpp":
        return _load_codeeditor_polish(n=n, tokenizer=tokenizer, language_filter="cpp")
    else:
        raise ValueError(f"unsupported language/benchmark: {language}")


# ---------------------------------------------------------------------------
# Stop sequences for code completion
# ---------------------------------------------------------------------------

# Python: stop on next function/class boundary, top-level statements that
# typically follow a function body.
PYTHON_STOP_TEXTS = (
    "\nclass ",
    "\ndef ",
    "\nif __name__",
    "\n#",
    "\nprint(",
    "\nassert ",
)

# TypeScript: stop on next top-level declaration. MultiPL-E TS prompts
# typically include a JSDoc comment + function signature, so we stop when
# the model starts writing the next function/class/export.
TYPESCRIPT_STOP_TEXTS = (
    "\nfunction ",
    "\nclass ",
    "\nexport ",
    "\nconst ",
    "\nlet ",
    "\nvar ",
    "\ninterface ",
    "\ntype ",
    "\nenum ",
    "\n//",
    "\n/*",
    "\nconsole.",
)

# Backwards-compat alias used by older code paths.
DEFAULT_STOP_TEXTS = PYTHON_STOP_TEXTS


def stop_texts_for_language(language: str = "python") -> tuple[str, ...]:
    if language in {
        "python",
        "mbpp",
        "repo_python",
        "repo_edit_python",
        "repo_edit_rename_python",
        "codeeditor_python",
        "codeeditor_switch_python",
    }:
        return PYTHON_STOP_TEXTS
    elif language in {
        "manifest",
        "codeeditor_translate",
        "codeeditor_translate_javacpp",
        "codeeditor_polish",
        "codeeditor_polish_cpp",
    }:
        return ()
    elif language in ("ts", "typescript"):
        return TYPESCRIPT_STOP_TEXTS
    else:
        raise ValueError(f"unsupported language: {language}")


def truncate_at_stop(
    text: str, stop_texts: tuple[str, ...] = DEFAULT_STOP_TEXTS
) -> str:
    """Cut `text` at the earliest stop-sequence occurrence (if any)."""
    earliest = len(text)
    for s in stop_texts:
        idx = text.find(s)
        if idx != -1 and idx < earliest:
            earliest = idx
    return text[:earliest]
