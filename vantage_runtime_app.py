"""Runtime-mounted Modal entrypoint for VANTAGE timing jobs.

This avoids baking the repository into a Modal image. The source tree is mounted
at runtime and Python dependencies are installed inside the container before the
evaluation command runs. It is intentionally a little less elegant than the
normal app, but it keeps image startup minimal when Modal image builds are the
launch bottleneck.
"""

from __future__ import annotations

from pathlib import Path

import modal


_PROJECT_ROOT = Path(__file__).resolve().parent

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_DATASETS_CACHE": "/cache/huggingface/datasets",
            "PYTHONPATH": "/root/asts-spec",
        }
    )
    .add_local_dir(
        str(_PROJECT_ROOT),
        "/root/asts-spec",
        copy=False,
        ignore=[
            ".venv",
            "out",
            "out/**",
            "logs",
            "logs/**",
            "paper",
            "paper/**",
            "tests",
            "tests/**",
            ".cache",
            ".cache/**",
            "__pycache__",
            "**/__pycache__/**",
            "*.egg-info",
            "*.egg-info/**",
            ".pytest_cache",
            ".pytest_cache/**",
            "*.pyc",
            ".git",
            ".git/**",
            "node_modules",
            "node_modules/**",
            ".vscode",
            ".idea",
            ".DS_Store",
            "*.zip",
        ],
    )
)

data_volume = modal.Volume.from_name("asts-spec-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("asts-spec-hf-cache", create_if_missing=True)

app = modal.App("vantage-runtime", image=image)


def _run_eagle_eval_impl(
    run_tag: str,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    target_trust_remote_code: bool = False,
    n: int = 50,
    max_new_tokens: int = 256,
    methods: str = "vanilla,blazedit_pld_w128_n10,vantage_transpld_w128_n10,vantage_routed_transpld_w128_n10",
    problem_jsonl: str = "",
    dtype: str = "bfloat16",
    attn_impl: str = "sdpa",
    code_proposer_fallback: str = "root",
) -> dict:
    import json
    import os
    import subprocess
    import sys

    os.chdir("/root/asts-spec")
    install_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "tree-sitter>=0.23.0",
        "tree-sitter-language-pack>=0.4.0",
        "numpy>=1.26",
        "transformers>=4.46",
        "accelerate>=1.0",
        "huggingface-hub>=0.26",
        "datasets>=3.0",
        "pydivsufsort>=0.0.18",
        "editdistance>=0.8.1",
    ]
    print(f"$ {' '.join(install_cmd)}", flush=True)
    subprocess.run(install_cmd, check=True)

    output_dir = f"/data/{run_tag}/eval"
    cmd = [
        sys.executable,
        "scripts/run_eagle_eval.py",
        "--output-dir",
        output_dir,
        "--target",
        target,
        "--n",
        str(n),
        "--max-new-tokens",
        str(max_new_tokens),
        "--methods",
        methods,
        "--dtype",
        dtype,
        "--attn-impl",
        attn_impl,
        "--problem-jsonl",
        problem_jsonl,
        "--skip-eagle-load",
        "--code-proposer-fallback",
        code_proposer_fallback,
        "--log-level",
        "INFO",
    ]
    if target_trust_remote_code:
        cmd.append("--target-trust-remote-code")
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    hf_cache.commit()
    with open(f"{output_dir}/aggregate.json") as f:
        aggregate = json.load(f)
    return {
        "run_tag": run_tag,
        "output_dir": output_dir,
        "by_method": aggregate.get("by_method", {}),
        "meta": aggregate.get("meta", {}),
    }


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=7200,
    startup_timeout=1800,
    cpu=4,
)
def run_eagle_eval_job(
    run_tag: str,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    target_trust_remote_code: bool = False,
    n: int = 50,
    max_new_tokens: int = 256,
    methods: str = "vanilla,blazedit_pld_w128_n10,vantage_transpld_w128_n10,vantage_routed_transpld_w128_n10",
    problem_jsonl: str = "",
    dtype: str = "bfloat16",
    attn_impl: str = "sdpa",
    code_proposer_fallback: str = "root",
) -> dict:
    return _run_eagle_eval_impl(
        run_tag=run_tag,
        target=target,
        target_trust_remote_code=target_trust_remote_code,
        n=n,
        max_new_tokens=max_new_tokens,
        methods=methods,
        problem_jsonl=problem_jsonl,
        dtype=dtype,
        attn_impl=attn_impl,
        code_proposer_fallback=code_proposer_fallback,
    )


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=7200,
    startup_timeout=1800,
    cpu=4,
    cloud="aws",
    region="us-west",
)
def run_eagle_eval_job_aws_west(
    run_tag: str,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    target_trust_remote_code: bool = False,
    n: int = 50,
    max_new_tokens: int = 256,
    methods: str = "vanilla,blazedit_pld_w128_n10,vantage_transpld_w128_n10,vantage_routed_transpld_w128_n10",
    problem_jsonl: str = "",
    dtype: str = "bfloat16",
    attn_impl: str = "sdpa",
    code_proposer_fallback: str = "root",
) -> dict:
    return _run_eagle_eval_impl(
        run_tag=run_tag,
        target=target,
        target_trust_remote_code=target_trust_remote_code,
        n=n,
        max_new_tokens=max_new_tokens,
        methods=methods,
        problem_jsonl=problem_jsonl,
        dtype=dtype,
        attn_impl=attn_impl,
        code_proposer_fallback=code_proposer_fallback,
    )


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=7200,
    startup_timeout=1800,
    cpu=4,
    cloud="gcp",
)
def run_eagle_eval_job_gcp(
    run_tag: str,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    target_trust_remote_code: bool = False,
    n: int = 50,
    max_new_tokens: int = 256,
    methods: str = "vanilla,blazedit_pld_w128_n10,vantage_transpld_w128_n10,vantage_routed_transpld_w128_n10",
    problem_jsonl: str = "",
    dtype: str = "bfloat16",
    attn_impl: str = "sdpa",
    code_proposer_fallback: str = "root",
) -> dict:
    return _run_eagle_eval_impl(
        run_tag=run_tag,
        target=target,
        target_trust_remote_code=target_trust_remote_code,
        n=n,
        max_new_tokens=max_new_tokens,
        methods=methods,
        problem_jsonl=problem_jsonl,
        dtype=dtype,
        attn_impl=attn_impl,
        code_proposer_fallback=code_proposer_fallback,
    )
