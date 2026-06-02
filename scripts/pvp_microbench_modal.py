"""Modal wrapper for PVP Step-1 microbench.

Usage:
    modal run scripts/pvp_microbench_modal.py::run \\
        --gpu L40S --iters 100 --warmup 10

Writes results to the asts-spec-data volume at /data/pvp/microbench_<gpu>.json
and downloads them back into analysis/pvp/runs/ if invoked through `modal run`.
"""

from __future__ import annotations

from pathlib import Path

import modal


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "build-essential", "clang")
    .pip_install(
        "tree-sitter>=0.23.0",
        "tree-sitter-language-pack>=0.4.0",
        "numpy>=1.26",
        "torch>=2.4",
        "transformers>=4.46",
        "accelerate>=1.0",
        "huggingface-hub>=0.26",
        "ninja",
        "packaging",
        "wheel",
    )
    .env({"MAX_JOBS": "2", "CUDA_HOME": "/usr/local/cuda"})
    .pip_install(
        "flash-attn>=2.7,<3.0",
        extra_options="--no-build-isolation",
    )
    .add_local_dir(
        str(_PROJECT_ROOT),
        "/root/asts-spec",
        copy=True,
        ignore=[
            ".venv", "out", "__pycache__", "*.egg-info",
            ".pytest_cache", "*.pyc", ".git", "node_modules",
            ".vscode", ".idea", ".DS_Store",
        ],
    )
    .run_commands("cd /root/asts-spec && pip install -e . --quiet")
    .env({
        "HF_HOME": "/cache/huggingface",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
)

data_volume = modal.Volume.from_name("asts-spec-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("asts-spec-hf-cache", create_if_missing=True)

app = modal.App("pvp-microbench", image=image)


def _run_microbench(
    *,
    gpu: str,
    target: str,
    dtype: str,
    attn_impl: str,
    iters: int,
    warmup: int,
) -> dict:
    import json
    import os
    import subprocess

    os.chdir("/root/asts-spec")
    output_path = f"/data/pvp/microbench_{gpu.lower()}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "scripts/microbench_batch_forward.py",
        "--target", target,
        "--dtype", dtype,
        "--attn-impl", attn_impl,
        "--iters", str(iters),
        "--warmup", str(warmup),
        "--output", output_path,
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, check=False)
    data_volume.commit()
    hf_cache.commit()

    if os.path.exists(output_path):
        with open(output_path) as f:
            report = json.load(f)
    else:
        report = {
            "schema": "asts-spec/pvp_microbench/v1",
            "error": "microbench wrote no output (likely crashed before any cell completed)",
            "ratios_vs_B1_by_prefix": {},
            "cells": [],
            "any_kill": None,
        }
    return {
        "exit_code": res.returncode,
        "output_path": output_path,
        "ratios": report.get("ratios_vs_B1_by_prefix", {}),
        "any_kill": report.get("any_kill"),
        "report": report,
    }


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=3600,
    cpu=4,
)
def microbench_l40s(
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bfloat16",
    attn_impl: str = "sdpa",
    iters: int = 100,
    warmup: int = 10,
) -> dict:
    return _run_microbench(
        gpu="L40S",
        target=target,
        dtype=dtype,
        attn_impl=attn_impl,
        iters=iters,
        warmup=warmup,
    )


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="H100",
    timeout=3600,
    cpu=4,
)
def microbench_h100(
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bfloat16",
    attn_impl: str = "sdpa",
    iters: int = 100,
    warmup: int = 10,
) -> dict:
    return _run_microbench(
        gpu="H100",
        target=target,
        dtype=dtype,
        attn_impl=attn_impl,
        iters=iters,
        warmup=warmup,
    )


@app.local_entrypoint()
def run(
    gpu: str = "L40S",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bfloat16",
    attn_impl: str = "sdpa",
    iters: int = 100,
    warmup: int = 10,
) -> None:
    import json

    gpu_norm = gpu.upper()
    if gpu_norm == "L40S":
        result = microbench_l40s.remote(
            target=target, dtype=dtype, attn_impl=attn_impl,
            iters=iters, warmup=warmup,
        )
    elif gpu_norm == "H100":
        result = microbench_h100.remote(
            target=target, dtype=dtype, attn_impl=attn_impl,
            iters=iters, warmup=warmup,
        )
    else:
        raise SystemExit(f"--gpu must be L40S or H100, got {gpu!r}")

    local_out = _PROJECT_ROOT / "analysis" / "pvp" / "runs" / f"microbench_{gpu_norm.lower()}.json"
    local_out.parent.mkdir(parents=True, exist_ok=True)
    local_out.write_text(json.dumps(result["report"], indent=2))

    print("\n=== PVP Step-1 microbench ===")
    print(f"  gpu              : {gpu_norm}")
    print(f"  any_kill         : {result['any_kill']}")
    for prefix_len, by_B in result["ratios"].items():
        ratio2 = by_B.get("2", by_B.get(2))
        print(f"  prefix={prefix_len}: T(B=2)/T(B=1) = {ratio2:.3f}")
    print(f"  local artifact   : {local_out}")
    print(f"  remote artifact  : {result['output_path']}")
    print(f"  exit_code        : {result['exit_code']}")
