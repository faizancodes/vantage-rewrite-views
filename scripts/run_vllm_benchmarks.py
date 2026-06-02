#!/usr/bin/env python3
"""vLLM benchmark harness for real-commit VANTAGE comparisons.

The harness intentionally keeps one artifact schema across runnable and blocked
methods so downstream agents can summarize successes, missing dependencies, and
custom-proposer API gaps without special cases.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asts.humaneval import Problem, load_problems_from_jsonl


DEFAULT_SPLIT_PATHS = {
    "train": "data/real_commits/real_commit_manifest_balanced_1000_v2_train500.jsonl",
    "test": "data/real_commits/real_commit_manifest_balanced_1000_v2_test500.jsonl",
}
DEFAULT_OUTPUT_ROOT = "artifacts/vllm_results"
PROMPT_TEMPLATE_VERSION = "real_commit_manifest_prompt_v1"
STOP_POLICY = "vllm_sampling_stop_list_or_max_tokens"


class CleanBenchmarkFailure(RuntimeError):
    def __init__(self, failure_type: str, message: str):
        super().__init__(message)
        self.failure_type = failure_type


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


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
        env["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            env["gpu_count"] = int(torch.cuda.device_count())
            env["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as exc:  # pragma: no cover - host dependent
        env["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
    return env


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def vllm_dtype(name: str) -> str:
    return {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "fp32": "float32",
        "float32": "float32",
    }[name]


def resolve_manifest_path(args: argparse.Namespace) -> Path:
    path = Path(args.manifest_path or DEFAULT_SPLIT_PATHS[args.split])
    if not path.is_absolute():
        path = ROOT / path
    return path


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        path = Path(args.output_dir)
        return path if path.is_absolute() else ROOT / path
    root = Path(args.output_root)
    if not root.is_absolute():
        root = ROOT / root
    return root / args.run_id


def default_run_id(args: argparse.Namespace) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{args.method}_{args.split}_n{args.n}_{stamp}"


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def build_sampling_params(args: argparse.Namespace) -> dict[str, Any]:
    sampling: dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_new_tokens,
    }
    if args.stop:
        sampling["stop"] = list(args.stop)
    return sampling


def build_speculative_config(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.method == "greedy":
        return None
    if args.method == "ngram":
        return {
            "method": "ngram",
            "prompt_lookup_min": args.ngram_prompt_lookup_min,
            "prompt_lookup_max": args.ngram_prompt_lookup_max,
            "num_speculative_tokens": args.num_speculative_tokens,
        }
    if args.method == "vantage_patched_ngram":
        if args.vantage_pld_patch_mode == "passthrough_trace":
            return {
                "method": "ngram",
                "prompt_lookup_min": args.ngram_prompt_lookup_min,
                "prompt_lookup_max": args.ngram_prompt_lookup_max,
                "num_speculative_tokens": args.num_speculative_tokens,
            }
        return {
            "method": "ngram",
            "prompt_lookup_min": args.vantage_match_tokens,
            "prompt_lookup_max": args.vantage_match_tokens,
            "num_speculative_tokens": args.num_speculative_tokens,
        }
    if args.method == "vantage_prompt_only":
        return {
            "method": "ngram",
            "prompt_lookup_min": args.vantage_match_tokens,
            "prompt_lookup_max": args.vantage_match_tokens,
            "num_speculative_tokens": args.num_speculative_tokens,
        }
    if args.method == "vantage_custom":
        return build_custom_speculative_config(args)
    raise ValueError(f"unsupported method: {args.method}")


def build_custom_speculative_config(args: argparse.Namespace) -> dict[str, Any]:
    proposer_path = f"{args.custom_proposer_module}.{args.custom_proposer_class}"
    common = {"num_speculative_tokens": args.num_speculative_tokens}
    if args.custom_config_variant == "legacy_custom":
        return {
            "method": "custom",
            "proposer_module": args.custom_proposer_module,
            "proposer_class": args.custom_proposer_class,
            "window_tokens": args.vantage_window_tokens,
            "match_tokens": args.vantage_match_tokens,
            "num_speculative_tokens": args.num_speculative_tokens,
            "label": "vantage_custom_w128_n10",
        }
    if args.custom_config_variant == "custom_class_model":
        return {"method": "custom_class", "model": proposer_path, **common}
    if args.custom_config_variant == "custom_class_field":
        return {"method": "custom_class", "custom_class": proposer_path, **common}
    if args.custom_config_variant == "model_only":
        return {"model": proposer_path, **common}
    raise ValueError(f"unsupported custom config variant: {args.custom_config_variant}")


def validate_custom_proposer(args: argparse.Namespace) -> None:
    try:
        module = importlib.import_module(args.custom_proposer_module)
    except (ImportError, ModuleNotFoundError) as exc:
        raise CleanBenchmarkFailure(
            "custom_proposer_unavailable",
            f"cannot import custom proposer module {args.custom_proposer_module!r}: {exc}",
        ) from exc
    if not hasattr(module, args.custom_proposer_class):
        raise CleanBenchmarkFailure(
            "custom_proposer_unavailable",
            f"custom proposer class {args.custom_proposer_class!r} not found in {args.custom_proposer_module!r}",
        )


def build_engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": vllm_dtype(args.dtype),
        "trust_remote_code": args.trust_remote_code,
        "max_model_len": args.max_model_len,
    }
    if args.tokenizer:
        kwargs["tokenizer"] = args.tokenizer
    if args.model_revision:
        kwargs["revision"] = args.model_revision
    if args.tokenizer_revision:
        kwargs["tokenizer_revision"] = args.tokenizer_revision
    if args.tensor_parallel_size:
        kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    if args.gpu_memory_utilization:
        kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    speculative_config = build_speculative_config(args)
    if speculative_config:
        kwargs["speculative_config"] = speculative_config
    return kwargs


def base_config(args: argparse.Namespace, manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    sampling_params = build_sampling_params(args)
    speculative_config = build_speculative_config(args)
    return {
        "run_id": args.run_id,
        "timestamp": args.timestamp,
        "git_commit": git_commit(),
        "agent": "Agent C",
        "engine": "vllm",
        "method": args.method,
        "custom_config_variant": args.custom_config_variant if args.method == "vantage_custom" else None,
        "model": args.model,
        "model_revision": args.model_revision,
        "tokenizer": args.tokenizer or args.model,
        "tokenizer_revision": args.tokenizer_revision,
        "manifest_path": str(manifest_path),
        "split": args.split,
        "requested_num_tasks": args.n,
        "sampling_params": sampling_params,
        "speculative_config": speculative_config,
        "max_new_tokens": args.max_new_tokens,
        "stop_policy": STOP_POLICY,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "output_dir": str(output_dir),
        "output_path": str(output_dir / "outputs.jsonl"),
        "config_path": str(output_dir / "config.json"),
        "raw_stdout_path": str(output_dir / "raw_stdout.txt"),
        "raw_stderr_path": str(output_dir / "raw_stderr.txt"),
        "minimal_proposer_log_path": str(output_dir / "minimal_proposer_events.jsonl")
        if args.method == "vantage_custom"
        else None,
        "vantage_pld_trace_path": str(output_dir / "proposer_trace.jsonl")
        if args.method == "vantage_patched_ngram"
        else None,
        "vantage_pld_patch_report_path": str(output_dir / "patch_report.json")
        if args.method == "vantage_patched_ngram"
        else None,
        "vantage_pld_patch_mode": args.vantage_pld_patch_mode
        if args.method == "vantage_patched_ngram"
        else None,
        "environment": collect_environment(),
        "engine_kwargs": build_engine_kwargs(args),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")


def write_outputs(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")


def prompt_token_count(output: Any) -> int | None:
    token_ids = getattr(output, "prompt_token_ids", None)
    if token_ids is None:
        return None
    try:
        return len(token_ids)
    except TypeError:
        return None


def first_candidate(output: Any) -> Any | None:
    candidates = getattr(output, "outputs", None) or []
    return candidates[0] if candidates else None


def output_row(problem: Problem, output: Any) -> tuple[dict[str, Any], int]:
    candidate = first_candidate(output)
    output_token_ids = list(getattr(candidate, "token_ids", []) or []) if candidate else []
    text = getattr(candidate, "text", "") if candidate else ""
    row = {
        "task_id": problem.task_id,
        "prompt_hash": prompt_hash(problem.prompt),
        "prompt_token_count": prompt_token_count(output),
        "output_text": text,
        "output_token_ids": output_token_ids,
        "output_token_count": len(output_token_ids),
        "finish_reason": getattr(candidate, "finish_reason", "") if candidate else "",
        "stop_reason": getattr(candidate, "stop_reason", None) if candidate else None,
        "generation_time_seconds_if_available": getattr(output, "metrics", None)
        and getattr(getattr(output, "metrics"), "finished_time", None),
        "engine_request_id": getattr(output, "request_id", None),
    }
    return row, len(output_token_ids)


def peak_memory_gb_if_available() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            allocated = torch.cuda.max_memory_allocated() / (1024.0**3)
            return allocated if allocated > 0.0 else None
    except Exception:  # pragma: no cover - host dependent
        return None
    return None


def prepare_vantage_ngram_patch(args: argparse.Namespace) -> None:
    """Patch installed vLLM before importing it, then set shim env vars."""

    output_dir = resolve_output_dir(args)
    trace_path = output_dir / "proposer_trace.jsonl"
    report_path = output_dir / "patch_report.json"
    backup_dir = output_dir / "patch_backup"
    env_updates = {
        "VANTAGE_PLD_PATCH": "1",
        "VANTAGE_PLD_MATCH_N": str(args.vantage_match_tokens),
        "VANTAGE_PLD_MAX_DRAFT_LEN": str(args.vantage_window_tokens),
        "VANTAGE_PLD_NUM_SPECULATIVE_TOKENS": str(args.num_speculative_tokens),
        "VANTAGE_PLD_TRACE_PATH": str(trace_path),
        "VANTAGE_PLD_TRACE_SAMPLE_RATE": str(args.vantage_pld_trace_sample_rate),
        "VANTAGE_PLD_TRACE_TOKENS": "1" if args.vantage_pld_trace_tokens else "0",
        "VANTAGE_PLD_PATCH_STRICT": "1" if args.vantage_pld_patch_strict else "0",
        "VANTAGE_PLD_PATCH_MODE": args.vantage_pld_patch_mode,
        "VANTAGE_PLD_NUMBA": "1" if args.vantage_pld_numba else "0",
        "VANTAGE_PLD_RUN_ID": args.run_id,
    }
    os.environ.update(env_updates)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "patch_installed_vllm_ngram_to_vantage.py"),
        "--backup-dir",
        str(backup_dir),
        "--report-path",
        str(report_path),
    ]
    subprocess.run(cmd, check=True)


def run_vllm(args: argparse.Namespace, problems: list[Problem]) -> dict[str, Any]:
    if args.method == "vantage_custom":
        validate_custom_proposer(args)
    if args.method == "vantage_patched_ngram":
        prepare_vantage_ngram_patch(args)

    try:
        vllm = importlib.import_module("vllm")
    except (ImportError, ModuleNotFoundError) as exc:
        raise CleanBenchmarkFailure("missing_dependency", f"{type(exc).__name__}: {exc}") from exc

    llm_cls = getattr(vllm, "LLM")
    sampling_params_cls = getattr(vllm, "SamplingParams")
    engine_kwargs = build_engine_kwargs(args)
    sampling_kwargs = build_sampling_params(args)
    sampling_params = sampling_params_cls(**sampling_kwargs)

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # pragma: no cover - host dependent
        pass

    init_t0 = time.perf_counter()
    llm = llm_cls(**engine_kwargs)
    init_seconds = time.perf_counter() - init_t0

    prompts = [problem.prompt for problem in problems]
    gen_t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    generation_wall_seconds = time.perf_counter() - gen_t0

    rows: list[dict[str, Any]] = []
    total_emitted_tokens = 0
    for problem, output in zip(problems, outputs, strict=False):
        row, emitted = output_row(problem, output)
        rows.append(row)
        total_emitted_tokens += emitted

    return {
        "rows": rows,
        "total_emitted_tokens": total_emitted_tokens,
        "generation_wall_seconds": generation_wall_seconds,
        "init_seconds": init_seconds,
        "peak_memory_gb_if_available": peak_memory_gb_if_available(),
    }


def classify_failure(exc: BaseException) -> str:
    if isinstance(exc, CleanBenchmarkFailure):
        return exc.failure_type
    text = f"{type(exc).__name__}: {exc}".lower()
    if "out of memory" in text or "cuda oom" in text:
        return "oom"
    if (
        "not supported" in text
        or "unsupported" in text
        or "unexpected keyword" in text
        or "literal_error" in text
        or "validationerror" in text
    ):
        return "incompatibility"
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return "missing_dependency"
    return "runtime_error"


def build_summary(
    config: dict[str, Any],
    *,
    status: str,
    num_tasks: int,
    result: dict[str, Any] | None = None,
    failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = result or {}
    total_emitted_tokens = int(result.get("total_emitted_tokens") or 0)
    generation_wall_seconds = float(result.get("generation_wall_seconds") or 0.0)
    init_seconds = float(result.get("init_seconds") or 0.0)
    including_init = generation_wall_seconds + init_seconds
    notes = [
        "Shared run-summary schema for VANTAGE vLLM diagnostics.",
        "Greedy sampling uses temperature=0 by default and no vLLM speculative_config.",
    ]
    if config["method"] == "vantage_prompt_only":
        notes.append(
            "VANTAGE fallback uses vLLM's built-in ngram method with fixed prompt_lookup_min=max=match_tokens. "
            "It is not a custom proposer and does not expose source/gold boundary metadata."
        )
        notes.append(
            "The PLD max draft length is capped by vLLM num_speculative_tokens in this fallback."
        )
    if failure:
        notes.append("Run failed cleanly; artifacts record the blocker and are not a negative result.")
    if config["method"] == "vantage_custom":
        notes.append(
            f"Custom proposer API probe variant: {config.get('custom_config_variant', 'unknown')}."
        )
    if config["method"] == "vantage_patched_ngram":
        notes.append(
            "Uses vLLM method='ngram' after installing an env-gated no-build shim over "
            "vLLM's native NgramProposer. Label as capped full-prefix PLD unless token "
            "proposal traces certify a stronger equivalence claim."
        )
        notes.append(f"VANTAGE_PLD_PATCH_MODE={config.get('vantage_pld_patch_mode')}.")
    summary = {
        "run_id": config["run_id"],
        "timestamp": config["timestamp"],
        "git_commit": config["git_commit"],
        "agent": config["agent"],
        "engine": config["engine"],
        "method": config["method"],
        "custom_config_variant": config.get("custom_config_variant"),
        "model": config["model"],
        "model_revision": config["model_revision"],
        "tokenizer": config["tokenizer"],
        "tokenizer_revision": config["tokenizer_revision"],
        "vllm_version": config["environment"]["packages"]["vllm"],
        "transformers_version": config["environment"]["packages"]["transformers"],
        "torch_version": config["environment"]["packages"]["torch"],
        "cuda_version": config["environment"].get("cuda_version"),
        "hardware": {
            "platform": config["environment"].get("platform"),
            "cuda_visible_devices": config["environment"].get("cuda_visible_devices"),
            "gpus": config["environment"].get("gpus", []),
        },
        "manifest_path": config["manifest_path"],
        "split": config["split"],
        "num_tasks": num_tasks,
        "sampling_params": config["sampling_params"],
        "speculative_config": config["speculative_config"],
        "max_new_tokens": config["max_new_tokens"],
        "stop_policy": config["stop_policy"],
        "prompt_template_version": config["prompt_template_version"],
        "total_emitted_tokens": total_emitted_tokens,
        "generation_wall_seconds": generation_wall_seconds,
        "init_seconds": init_seconds,
        "tok_per_s_excluding_init": total_emitted_tokens / max(1e-9, generation_wall_seconds),
        "tok_per_s_including_init": total_emitted_tokens / max(1e-9, including_init),
        "peak_memory_gb_if_available": result.get("peak_memory_gb_if_available"),
        "output_path": config["output_path"],
        "config_path": config["config_path"],
        "notes": notes,
        "status": status,
    }
    if failure:
        summary["failure"] = failure
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=[
            "greedy",
            "ngram",
            "vantage_custom",
            "vantage_patched_ngram",
            "vantage_prompt_only",
        ],
        default="greedy",
    )
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--split", choices=sorted(DEFAULT_SPLIT_PATHS), default="test")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--tokenizer-revision", default="")
    parser.add_argument("--dtype", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"], default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--stop", action="append", default=[])
    parser.add_argument("--ngram-prompt-lookup-min", type=int, default=2)
    parser.add_argument("--ngram-prompt-lookup-max", type=int, default=128)
    parser.add_argument("--vantage-window-tokens", type=int, default=128)
    parser.add_argument("--vantage-match-tokens", type=int, default=10)
    parser.add_argument("--num-speculative-tokens", type=int, default=8)
    parser.add_argument("--custom-proposer-module", default="vantage_vllm.minimal_custom_proposer")
    parser.add_argument("--custom-proposer-class", default="MinimalCustomProposer")
    parser.add_argument(
        "--vantage-pld-trace-sample-rate",
        type=float,
        default=1.0,
        help="Trace sample rate for vantage_patched_ngram proposal rows.",
    )
    parser.add_argument(
        "--vantage-pld-trace-tokens",
        action="store_true",
        help="Write token prefixes/proposals in proposer_trace.jsonl for equivalence checking.",
    )
    parser.add_argument(
        "--vantage-pld-patch-strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the patched shim to infer or receive the vLLM speculation cap.",
    )
    parser.add_argument(
        "--vantage-pld-patch-mode",
        choices=["off", "passthrough_trace", "native_fixed_n", "pld_python", "pld_optimized"],
        default="pld_python",
        help="Env-gated mode for vantage_patched_ngram.",
    )
    parser.add_argument(
        "--vantage-pld-numba",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow the optimized PLD shim to use the optional Numba backend.",
    )
    parser.add_argument(
        "--custom-config-variant",
        choices=["legacy_custom", "custom_class_model", "custom_class_field", "model_only"],
        default="legacy_custom",
        help="Speculative config shape for vantage_custom API probes.",
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--run-id", default="")
    args = parser.parse_args(argv)
    args.timestamp = iso_now()
    if not args.run_id:
        args.run_id = default_run_id(args)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = resolve_manifest_path(args)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.method == "vantage_custom":
        os.environ.setdefault(
            "VANTAGE_VLLM_MINIMAL_PROPOSER_LOG",
            str(output_dir / "minimal_proposer_events.jsonl"),
        )

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    rc = 0
    rows: list[dict[str, Any]] = []
    result: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None
    num_tasks = 0

    try:
        config = base_config(args, manifest_path, output_dir)
        write_json(output_dir / "config.json", config)
        problems = load_problems_from_jsonl(str(manifest_path), n=args.n)
        num_tasks = len(problems)
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            result = run_vllm(args, problems)
        rows = result.pop("rows")
        status = "success"
    except BaseException as exc:
        status = "failed"
        failure = {
            "type": classify_failure(exc),
            "message": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        rc = 2 if failure["type"] == "missing_dependency" else 3 if failure["type"] == "custom_proposer_unavailable" else 1
        if "config" not in locals():
            config = base_config(args, manifest_path, output_dir)
            write_json(output_dir / "config.json", config)
    finally:
        (output_dir / "raw_stdout.txt").write_text(stdout_buffer.getvalue())
        (output_dir / "raw_stderr.txt").write_text(stderr_buffer.getvalue())
        write_outputs(output_dir / "outputs.jsonl", rows)
        summary = build_summary(
            config,
            status=status,
            num_tasks=num_tasks,
            result=result,
            failure=failure,
        )
        write_json(output_dir / "run_summary.json", summary)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
