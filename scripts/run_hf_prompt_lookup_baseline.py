#!/usr/bin/env python3
"""Hugging Face prompt-lookup generation baseline for archived continuous-batched PLD prototype.

This script only uses Transformers' built-in ``generate`` support for
``prompt_lookup_num_tokens``. If the installed Transformers release does not
expose that option, it writes an incompatibility report and exits non-zero.
"""

from __future__ import annotations

import argparse
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
from scripts.run_eagle_eval import _encode_prompt_ids


DEFAULT_SPLIT_PATHS = {
    "train": "data/real_commits/real_commit_manifest_balanced_1000_v2_train500.jsonl",
    "test": "data/real_commits/real_commit_manifest_balanced_1000_v2_test500.jsonl",
}

HF_PROMPT_LOOKUP_DOCS_URL = "https://huggingface.co/docs/transformers/assisted_decoding#prompt-lookup-decoding"


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
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "accelerate": package_version("accelerate"),
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


def transformers_supports_prompt_lookup() -> tuple[bool, str]:
    try:
        from transformers import GenerationConfig
    except Exception as exc:
        return False, f"could not import transformers.GenerationConfig: {type(exc).__name__}: {exc}"
    config = GenerationConfig()
    if hasattr(config, "prompt_lookup_num_tokens"):
        return True, "GenerationConfig exposes prompt_lookup_num_tokens"
    return False, "GenerationConfig lacks prompt_lookup_num_tokens"


def resolve_problem_jsonl(args: argparse.Namespace) -> Path:
    if args.problem_jsonl:
        path = Path(args.problem_jsonl)
    else:
        path = ROOT / DEFAULT_SPLIT_PATHS[args.split]
    if not path.is_absolute():
        path = ROOT / path
    return path


def torch_dtype(name: str):
    import torch

    return {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }[name]


def build_model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    model_kwargs = {
        "torch_dtype": torch_dtype(args.dtype),
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn and args.attn != "default":
        model_kwargs["attn_implementation"] = args.attn
    if args.device == "cuda":
        model_kwargs["device_map"] = "cuda"
    return model_kwargs


def build_generation_kwargs(args: argparse.Namespace, tokenizer: Any) -> dict[str, Any]:
    generation_kwargs = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": args.max_new_tokens,
        "prompt_lookup_num_tokens": args.prompt_lookup_num_tokens,
        "max_matching_ngram_size": args.max_matching_ngram_size,
        "return_dict_in_generate": True,
    }
    if tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
    return generation_kwargs


def build_comparability_notes(args: argparse.Namespace) -> list[str]:
    return [
        "External Hugging Face baseline using Transformers' built-in generate(prompt_lookup_num_tokens=...); this script does not add VANTAGE/PLD decoding logic.",
        "Comparable only when run on the same held-out split, model, dtype, max_new_tokens, and stop policy as the paper run.",
        "This harness evaluates one prompt at a time because Transformers prompt-lookup generation is an assisted-decoding path rather than a continuous-batching serving engine.",
        f"Attention implementation is requested as {args.attn!r}; the model load fails loudly if that setting is unsupported by the installed Transformers/model combination.",
    ]


def classify_failure(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "out of memory" in text or "cuda oom" in text:
        return "oom"
    if "prompt_lookup_num_tokens" in text or "unused model_kwargs" in text:
        return "incompatibility"
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return "missing_dependency"
    return "runtime_error"


def write_reports(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    config = payload["config"]
    result = payload.get("result", {})
    lines = [
        "# Hugging Face Prompt-Lookup Baseline Attempt",
        "",
        f"status: `{payload.get('status', 'unknown')}`",
        f"target: `{config['target']}`",
        f"dtype: `{config['dtype']}`",
        f"prompt_lookup_num_tokens: `{config['prompt_lookup_num_tokens']}`",
        f"max_matching_ngram_size: `{config['max_matching_ngram_size']}`",
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
    if payload.get("compatibility"):
        compat = payload["compatibility"]
        lines.extend(
            [
                "",
                "## Compatibility Preflight",
                "",
                f"- Transformers prompt lookup supported: `{compat.get('transformers_prompt_lookup_supported')}`",
                f"- reason: `{compat.get('reason')}`",
            ]
        )
    if result:
        lines.extend(
            [
                "",
                "## Result",
                "",
                f"- prompts: `{result.get('n_prompts', 0)}`",
                f"- emitted tokens: `{result.get('total_new_tokens', 0)}`",
                f"- wall ms: `{result.get('wall_ms', 0.0):.1f}`",
                f"- tokens/sec: `{result.get('tokens_per_sec', 0.0):.2f}`",
                f"- peak memory GB: `{result.get('memory_peak_gb', 0.0):.2f}`",
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
            "",
            "## External References",
            "",
            f"- Transformers prompt-lookup decoding docs: {payload.get('external_docs', {}).get('transformers_prompt_lookup_decoding', '')}",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def run_hf_prompt_lookup(args: argparse.Namespace) -> dict[str, Any]:
    supported, reason = transformers_supports_prompt_lookup()
    if not supported:
        raise RuntimeError(f"Transformers prompt lookup unavailable: {reason}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    problem_jsonl = resolve_problem_jsonl(args)
    problems = load_problems_from_jsonl(str(problem_jsonl), n=args.n)
    tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = build_model_kwargs(args)
    model = AutoModelForCausalLM.from_pretrained(args.target, **model_kwargs)
    if args.device != "cuda":
        model.to(args.device)
    model.eval()

    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    rows = []
    total_new_tokens = 0
    generation_kwargs = build_generation_kwargs(args, tokenizer)

    t0 = time.perf_counter_ns()
    for problem in problems:
        input_ids = _encode_prompt_ids(tokenizer, problem.prompt, args.chat_template).to(device)
        input_ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
        prompt_len = int(input_ids.shape[-1])
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
        sequence = output.sequences[0].detach().cpu().tolist()
        new_ids = sequence[prompt_len:]
        total_new_tokens += len(new_ids)
        rows.append(
            {
                "task_id": problem.task_id,
                "new_tokens": len(new_ids),
                "text": tokenizer.decode(new_ids, skip_special_tokens=False),
            }
        )
    wall_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

    out_path = Path(args.output_dir) / "completions.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    return {
        "problem_jsonl": str(problem_jsonl),
        "model_kwargs": {
            key: str(value) if key == "torch_dtype" else value for key, value in model_kwargs.items()
        },
        "generation_kwargs": generation_kwargs,
        "n_prompts": len(problems),
        "total_new_tokens": total_new_tokens,
        "wall_ms": wall_ms,
        "tokens_per_sec": total_new_tokens / max(1e-9, wall_ms / 1000.0),
        "memory_peak_gb": (
            torch.cuda.max_memory_allocated(device) / (1024.0**3) if device.type == "cuda" else 0.0
        ),
        "completions_jsonl": str(out_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-jsonl", default="")
    parser.add_argument("--split", choices=sorted(DEFAULT_SPLIT_PATHS), default="test")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn", default="sdpa", help="Passed as attn_implementation unless set to 'default'.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--prompt-lookup-num-tokens", type=int, default=128)
    parser.add_argument("--max-matching-ngram-size", type=int, default=2)
    parser.add_argument("--chat-template", default="none")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    problem_jsonl = resolve_problem_jsonl(args)
    supported, support_reason = transformers_supports_prompt_lookup()
    payload: dict[str, Any] = {
        "status": "started",
        "started_at_utc": iso_now(),
        "command": [sys.executable, *sys.argv] if argv is None else [sys.executable, __file__, *argv],
        "config": vars(args),
        "problem_jsonl_resolved": str(problem_jsonl),
        "compatibility": {
            "transformers_prompt_lookup_supported": supported,
            "reason": support_reason,
        },
        "comparability_notes": build_comparability_notes(args),
        "external_docs": {
            "transformers_prompt_lookup_decoding": HF_PROMPT_LOOKUP_DOCS_URL,
        },
        "environment": collect_environment(),
    }
    try:
        payload["result"] = run_hf_prompt_lookup(args)
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
