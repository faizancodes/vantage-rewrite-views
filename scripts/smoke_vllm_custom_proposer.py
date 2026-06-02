#!/usr/bin/env python3
"""Inspect vLLM speculative proposer APIs in the current environment.

This is intentionally read-only: it imports vLLM when available, reports
versions, signatures, selected class sources, and the speculative method
dispatch branches that decide which proposer classes can be constructed.
It can also inspect an unpacked vLLM source tree with --source-root when the
package cannot be imported in the local environment.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata
import inspect
import json
import platform
import sys
from pathlib import Path
from typing import Any


DEFAULT_MODULES = [
    "vllm",
    "vllm.config.speculative",
    "vllm.engine.arg_utils",
    "vllm.v1.spec_decode.ngram_proposer",
    "vllm.v1.spec_decode.suffix_decoding",
    "vllm.v1.spec_decode.medusa",
    "vllm.v1.spec_decode.extract_hidden_states",
    "vllm.v1.spec_decode.llm_base_proposer",
    "vllm.v1.worker.gpu_model_runner",
]

DEFAULT_OBJECTS = [
    "vllm.LLM",
    "vllm.SamplingParams",
    "vllm.config.speculative.SpeculativeConfig",
    "vllm.v1.spec_decode.ngram_proposer.NgramProposer",
    "vllm.v1.spec_decode.suffix_decoding.SuffixDecodingProposer",
    "vllm.v1.spec_decode.medusa.MedusaProposer",
    "vllm.v1.spec_decode.extract_hidden_states.ExtractHiddenStatesProposer",
    "vllm.v1.spec_decode.llm_base_proposer.SpecDecodeBaseProposer",
    "vllm.v1.worker.gpu_model_runner.GPUModelRunner",
]


SOURCE_FILES = [
    "vllm/config/speculative.py",
    "vllm/engine/arg_utils.py",
    "vllm/v1/spec_decode/ngram_proposer.py",
    "vllm/v1/spec_decode/suffix_decoding.py",
    "vllm/v1/spec_decode/medusa.py",
    "vllm/v1/spec_decode/extract_hidden_states.py",
    "vllm/v1/spec_decode/llm_base_proposer.py",
    "vllm/v1/worker/gpu_model_runner.py",
]


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def safe_signature(obj: Any) -> str:
    try:
        return str(inspect.signature(obj))
    except Exception as exc:
        return f"<signature unavailable: {type(exc).__name__}: {exc}>"


def safe_source_head(obj: Any, max_lines: int) -> list[str]:
    try:
        return inspect.getsource(obj).splitlines()[:max_lines]
    except Exception as exc:
        return [f"<source unavailable: {type(exc).__name__}: {exc}>"]


def resolve_object(dotted: str) -> Any:
    module_name, _, attr_path = dotted.partition(".")
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in attr_path.split("."):
        if part:
            obj = getattr(obj, part)
    return obj


def inspect_imported(max_source_lines: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "modules": {},
        "objects": {},
    }
    for module_name in DEFAULT_MODULES:
        try:
            module = importlib.import_module(module_name)
            out["modules"][module_name] = {
                "ok": True,
                "file": getattr(module, "__file__", None),
            }
        except Exception as exc:
            out["modules"][module_name] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    for dotted in DEFAULT_OBJECTS:
        try:
            obj = resolve_object(dotted)
            item: dict[str, Any] = {
                "ok": True,
                "type": type(obj).__name__,
                "module": getattr(obj, "__module__", None),
                "signature": safe_signature(obj),
            }
            for method_name in ("__init__", "propose", "load_model"):
                method = getattr(obj, method_name, None)
                if method is not None:
                    item[f"{method_name}_signature"] = safe_signature(method)
            if max_source_lines:
                item["source_head"] = safe_source_head(obj, max_source_lines)
            out["objects"][dotted] = item
        except Exception as exc:
            out["objects"][dotted] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return out


def extract_ast_summary(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    classes: dict[str, Any] = {}
    functions: dict[str, Any] = {}
    assignments: dict[str, str] = {}
    comparisons: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = {}
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods[child.name] = {
                        "lineno": child.lineno,
                        "signature": ast.unparse(child.args),
                    }
            classes[node.name] = {"lineno": node.lineno, "methods": methods}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = {
                "lineno": node.lineno,
                "signature": ast.unparse(node.args),
            }
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "SpeculativeMethod",
                    "MTPModelTypes",
                    "NgramGPUTypes",
                    "EagleModelTypes",
                }:
                    assignments[target.id] = ast.unparse(node.value)
        elif isinstance(node, ast.Compare):
            rendered = ast.unparse(node)
            if "method" in rendered or "spec_config" in rendered:
                comparisons.append(rendered)

    return {
        "path": str(path),
        "classes": classes,
        "functions": functions,
        "assignments": assignments,
        "method_comparisons": sorted(set(comparisons))[:200],
    }


def inspect_source_tree(source_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"source_root": str(source_root), "files": {}}
    root = source_root
    if not (root / "vllm").exists():
        nested = sorted(root.glob("*/vllm"))
        if nested:
            root = nested[0].parent
            out["resolved_source_root"] = str(root)
    for rel in SOURCE_FILES:
        path = root / rel
        if not path.exists():
            out["files"][rel] = {"ok": False, "error": "missing"}
            continue
        try:
            out["files"][rel] = {"ok": True, **extract_ast_summary(path)}
        except Exception as exc:
            out["files"][rel] = {
                "ok": False,
                "path": str(path),
                "error": f"{type(exc).__name__}: {exc}",
            }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="", help="Optional unpacked vLLM source root.")
    parser.add_argument("--write-json", default="", help="Optional output JSON path.")
    parser.add_argument("--max-source-lines", type=int, default=80)
    args = parser.parse_args(argv)

    payload: dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "packages": {
            "vllm": package_version("vllm"),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "tokenizers": package_version("tokenizers"),
        },
        "imported": None,
        "source_tree": None,
    }

    rc = 0
    try:
        importlib.import_module("vllm")
        payload["imported"] = inspect_imported(args.max_source_lines)
    except Exception as exc:
        payload["import_error"] = f"{type(exc).__name__}: {exc}"
        rc = 2

    if args.source_root:
        payload["source_tree"] = inspect_source_tree(Path(args.source_root))
        rc = 0 if payload["source_tree"] else rc

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.write_json:
        Path(args.write_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write_json).write_text(text, encoding="utf-8")
    print(text, end="")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
