"""Evaluate pass@1 from run_eagle_eval.py completions.

The eval harness writes generated suffixes, not full programs.  This script
reconstructs each benchmark program from the saved prompt + generated text and
runs the public tests in a subprocess with a timeout.

Supported benchmarks:
  - python: openai_humaneval
  - mbpp: google-research-datasets/mbpp sanitized
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


PRELUDE = """
from typing import *
import math
import re
import itertools
import collections
import functools
import heapq
import bisect
import string
"""


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def humaneval_tests() -> dict[str, dict[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("openai_humaneval", split="test")
    out = {}
    for row in ds:
        out[str(row["task_id"])] = {
            "test": str(row["test"]),
            "entry_point": str(row["entry_point"]),
        }
    return out


def mbpp_tests() -> dict[str, list[str]]:
    from datasets import load_dataset

    out: dict[str, list[str]] = {}
    for split in ("train", "validation", "test"):
        try:
            ds = load_dataset("google-research-datasets/mbpp", "sanitized", split=split)
        except Exception:
            continue
        for row in ds:
            task_id = f"MBPP/{row.get('task_id')}"
            tests = row.get("test_list") or []
            out[task_id] = [str(t) for t in tests]
    return out


def make_program(
    benchmark: str,
    task_id: str,
    prompt: str,
    completion: str,
    test_data: dict[str, Any],
) -> str:
    if benchmark == "python":
        item = test_data[task_id]
        return "\n".join(
            [
                PRELUDE,
                prompt,
                completion,
                item["test"],
                f"check({item['entry_point']})",
                "",
            ]
        )
    if benchmark == "mbpp":
        tests = test_data[task_id]
        return "\n".join([PRELUDE, prompt, completion, *tests, ""])
    raise ValueError(f"unsupported benchmark for pass@1: {benchmark}")


def run_program(program: str, timeout_s: float) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(program)
        try:
            res = subprocess.run(
                [sys.executable, str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if res.returncode == 0:
        return True, ""
    err = (res.stderr or res.stdout or "").strip()
    return False, err[-1000:]


def evaluate(
    completions: list[dict[str, Any]],
    benchmark: str,
    methods: list[str] | None,
    timeout_s: float,
) -> dict[str, Any]:
    test_data: dict[str, Any]
    if benchmark == "python":
        test_data = humaneval_tests()
    elif benchmark == "mbpp":
        test_data = mbpp_tests()
    else:
        raise ValueError("pass@1 script supports benchmark=python or mbpp")

    counts = defaultdict(lambda: {"n": 0, "passed": 0})
    failures: list[dict[str, Any]] = []

    for row in completions:
        task_id = str(row["task_id"])
        if task_id not in test_data:
            continue
        prompt = str(row["prompt"])
        outputs = row.get("outputs", {})
        method_names = methods if methods is not None else list(outputs.keys())
        for method in method_names:
            if method not in outputs:
                continue
            completion = str(outputs[method].get("text", ""))
            program = make_program(benchmark, task_id, prompt, completion, test_data)
            passed, error = run_program(program, timeout_s)
            counts[method]["n"] += 1
            counts[method]["passed"] += int(passed)
            if not passed:
                failures.append({"task_id": task_id, "method": method, "error": error})

    by_method = {}
    for method, item in sorted(counts.items()):
        n = item["n"]
        passed = item["passed"]
        by_method[method] = {
            "n": n,
            "passed": passed,
            "pass_at_1": passed / n if n else 0.0,
        }
    return {"benchmark": benchmark, "by_method": by_method, "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completions", required=True)
    parser.add_argument("--benchmark", required=True, choices=["python", "mbpp"])
    parser.add_argument("--methods", default="")
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()] or None
    report = evaluate(load_jsonl(args.completions), args.benchmark, methods, args.timeout_s)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(json.dumps({"benchmark": report["benchmark"], "by_method": report["by_method"]}, indent=2))


if __name__ == "__main__":
    main()
