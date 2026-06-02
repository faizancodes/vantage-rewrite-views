#!/usr/bin/env python3
"""External vLLM baseline harness for archived continuous-batched PLD prototype.

This script intentionally does not implement a new decoder. It invokes vLLM's
serving/generation path and records either measured results or the exact reason
the baseline could not be run in the local environment.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asts.humaneval import load_problems_from_jsonl


DEFAULT_SPLIT_PATHS = {
    "train": "data/real_commits/real_commit_manifest_balanced_1000_v2_train500.jsonl",
    "test": "data/real_commits/real_commit_manifest_balanced_1000_v2_test500.jsonl",
}

VLLM_NGRAM_DOCS_URL = "https://docs.vllm.ai/usage/speculative_decoding/"


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def collect_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "packages": {
            "vllm": package_version("vllm"),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "tokenizers": package_version("tokenizers"),
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    try:
        import torch

        env["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["gpu_count"] = int(torch.cuda.device_count())
            env["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            env["cuda_version"] = getattr(torch.version, "cuda", None)
    except Exception as exc:  # pragma: no cover - depends on local torch install
        env["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
    return env


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_problem_jsonl(args: argparse.Namespace) -> Path:
    if args.problem_jsonl:
        path = Path(args.problem_jsonl)
    else:
        path = ROOT / DEFAULT_SPLIT_PATHS[args.split]
    if not path.is_absolute():
        path = ROOT / path
    return path


def classify_failure(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "out of memory" in text or "cuda oom" in text:
        return "oom"
    if "not supported" in text or "unsupported" in text or "unexpected keyword" in text:
        return "incompatibility"
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return "missing_dependency"
    return "runtime_error"


def vllm_dtype(name: str) -> str:
    return {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "fp32": "float32",
        "float32": "float32",
    }[name]


def write_reports(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n"
    )

    config = payload["config"]
    result = payload.get("result", {})
    status = payload.get("status", "unknown")
    lines = [
        "# vLLM External Baseline Attempt",
        "",
        f"status: `{status}`",
        f"backend: `{config['backend']}`",
        f"target: `{config['target']}`",
        f"dtype: `{config['dtype']}`",
        f"n: `{config['n']}`",
        f"max_new_tokens: `{config['max_new_tokens']}`",
        "",
        "## Command",
        "",
        "```bash",
        " ".join(payload["command"]),
        "```",
        "",
        "## Environment",
        "",
    ]
    for name, version in payload["environment"].get("packages", {}).items():
        lines.append(f"- {name}: `{version}`")
    if payload.get("failure"):
        failure = payload["failure"]
        lines.extend(
            [
                "",
                "## Failure",
                "",
                f"type: `{failure['type']}`",
                f"message: `{failure['message']}`",
            ]
        )
    if result:
        memory_peak = result.get("memory_peak_gb")
        memory_text = f"{memory_peak:.2f}" if isinstance(memory_peak, (int, float)) else "not captured"
        lines.extend(
            [
                "",
                "## Result",
                "",
                f"- prompts: `{result.get('n_prompts', 0)}`",
                f"- emitted tokens: `{result.get('total_new_tokens', 0)}`",
                f"- engine init ms: `{result.get('engine_init_ms', 0.0):.1f}`",
                f"- generation wall ms: `{result.get('generation_wall_ms', result.get('wall_ms', 0.0)):.1f}`",
                f"- cold wall ms: `{result.get('cold_wall_ms', 0.0):.1f}`",
                f"- generation tokens/sec: `{result.get('tokens_per_sec', 0.0):.2f}`",
                f"- cold tokens/sec: `{result.get('cold_tokens_per_sec', 0.0):.2f}`",
                f"- peak memory GB: `{memory_text}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Comparability Notes",
            "",
        ]
    )
    for note in payload.get("comparability_notes", []):
        lines.append(f"- {note}")
    lines.extend(
        [
            "- A failed or incompatible run is not a negative result for vLLM; it records that this environment did not produce a comparable artifact.",
            "",
            "## External References",
            "",
            f"- vLLM speculative decoding docs: {payload.get('external_docs', {}).get('vllm_speculative_decoding', '')}",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def build_engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.target,
        "dtype": vllm_dtype(args.dtype),
        "trust_remote_code": args.trust_remote_code,
        "max_model_len": args.max_model_len,
    }
    if args.tensor_parallel_size:
        kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    if args.gpu_memory_utilization:
        kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    if args.backend == "ngram_speculation":
        kwargs["speculative_config"] = {
            "method": "ngram",
            "prompt_lookup_min": args.ngram_prompt_lookup_min,
            "prompt_lookup_max": args.ngram_prompt_lookup_max,
            "num_speculative_tokens": args.num_speculative_tokens,
        }
    return kwargs


def _json_safe(value: Any) -> Any:
    """Convert external-library config objects to JSON-safe report values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def build_comparability_notes(args: argparse.Namespace) -> list[str]:
    notes = [
        "External vLLM baseline using vLLM's LLM.generate path; this script does not add VANTAGE/PLD decoding logic.",
        "Comparable only when run on the same held-out split, model, dtype, max_new_tokens, and stop policy as the paper run.",
        "Throughput is generated tokens divided by generation-only wall-clock time; engine construction/model-load time is recorded separately as engine_init_ms.",
    ]
    if args.backend == "ngram_speculation":
        notes.append(
            "vLLM n-gram speculation is configured through speculative_config with method=ngram, prompt_lookup_min/max, and num_speculative_tokens as documented by vLLM."
        )
    return notes


def run_vllm(args: argparse.Namespace) -> dict[str, Any]:
    vllm = importlib.import_module("vllm")
    llm_cls = getattr(vllm, "LLM")
    sampling_params_cls = getattr(vllm, "SamplingParams")

    problem_jsonl = resolve_problem_jsonl(args)
    problems = load_problems_from_jsonl(str(problem_jsonl), n=args.n)
    prompts = [p.prompt for p in problems]
    engine_kwargs = build_engine_kwargs(args)
    sampling_kwargs = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": args.max_new_tokens,
    }
    if args.stop:
        sampling_kwargs["stop"] = args.stop
    sampling_params = sampling_params_cls(**sampling_kwargs)

    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    init_t0 = time.perf_counter_ns()
    llm = llm_cls(**engine_kwargs)
    engine_init_ms = (time.perf_counter_ns() - init_t0) / 1_000_000.0
    t0 = time.perf_counter_ns()
    outputs = llm.generate(prompts, sampling_params)
    wall_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
    cold_wall_ms = engine_init_ms + wall_ms

    rows = []
    total_new_tokens = 0
    for problem, output in zip(problems, outputs, strict=False):
        candidates = getattr(output, "outputs", []) or []
        text = candidates[0].text if candidates else ""
        token_ids = list(getattr(candidates[0], "token_ids", []) or []) if candidates else []
        total_new_tokens += len(token_ids)
        rows.append(
            {
                "task_id": problem.task_id,
                "new_tokens": len(token_ids),
                "text": text,
                "finish_reason": getattr(candidates[0], "finish_reason", "") if candidates else "",
            }
        )

    out_path = Path(args.output_dir) / "completions.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    memory_peak_gb = None
    if torch.cuda.is_available():
        allocated = torch.cuda.max_memory_allocated() / (1024.0**3)
        memory_peak_gb = allocated if allocated > 0.0 else None

    return {
        "problem_jsonl": str(problem_jsonl),
        "engine_kwargs": engine_kwargs,
        "sampling_kwargs": sampling_kwargs,
        "n_prompts": len(prompts),
        "total_new_tokens": total_new_tokens,
        "engine_init_ms": engine_init_ms,
        "wall_ms": wall_ms,
        "generation_wall_ms": wall_ms,
        "cold_wall_ms": cold_wall_ms,
        "tokens_per_sec": total_new_tokens / max(1e-9, wall_ms / 1000.0),
        "cold_tokens_per_sec": total_new_tokens / max(1e-9, cold_wall_ms / 1000.0),
        "memory_peak_gb": memory_peak_gb,
        "completions_jsonl": str(out_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-jsonl", default="")
    parser.add_argument("--split", choices=sorted(DEFAULT_SPLIT_PATHS), default="test")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--backend", choices=["greedy", "ngram_speculation"], default="greedy")
    parser.add_argument("--ngram-prompt-lookup-min", type=int, default=2)
    parser.add_argument("--ngram-prompt-lookup-max", type=int, default=128)
    parser.add_argument("--num-speculative-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--stop", action="append", default=[])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    problem_jsonl = resolve_problem_jsonl(args)
    payload: dict[str, Any] = {
        "status": "started",
        "started_at_utc": iso_now(),
        "command": [sys.executable, *sys.argv] if argv is None else [sys.executable, __file__, *argv],
        "config": vars(args),
        "problem_jsonl_resolved": str(problem_jsonl),
        "comparability_notes": build_comparability_notes(args),
        "external_docs": {
            "vllm_speculative_decoding": VLLM_NGRAM_DOCS_URL,
        },
        "environment": collect_environment(),
    }
    try:
        payload["result"] = run_vllm(args)
        payload["status"] = "success"
        rc = 0
    except (ImportError, ModuleNotFoundError) as exc:
        payload["status"] = "failed"
        payload["failure"] = {
            "type": "missing_dependency",
            "message": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        rc = 2
    except BaseException as exc:
        payload["status"] = "failed"
        payload["failure"] = {
            "type": classify_failure(exc),
            "message": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        rc = 1
    payload["finished_at_utc"] = iso_now()
    write_reports(output_dir, payload)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
