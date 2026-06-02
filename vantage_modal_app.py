"""Lightweight Modal entrypoints for VANTAGE timing runs.

This avoids the large multi-function ``proto_app.py`` precreate/mount path.
"""

from __future__ import annotations

from pathlib import Path

import modal


_PROJECT_ROOT = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "tree-sitter>=0.23.0",
        "tree-sitter-language-pack>=0.4.0",
        "numpy>=1.26",
        "torch>=2.4",
        "transformers>=4.46",
        "accelerate>=1.0",
        "huggingface-hub>=0.26",
        "datasets>=3.0",
        "pydivsufsort>=0.0.18",
    )
    .add_local_dir(
        str(_PROJECT_ROOT),
        "/root/asts-spec",
        copy=True,
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
        ],
    )
    .run_commands("cd /root/asts-spec && pip install -e . --quiet")
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_DATASETS_CACHE": "/cache/huggingface/datasets",
        }
    )
)

data_volume = modal.Volume.from_name("asts-spec-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("asts-spec-hf-cache", create_if_missing=True)

app = modal.App("vantage-modal", image=image)


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=7200,
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
    import json
    import os
    import subprocess

    os.chdir("/root/asts-spec")
    output_dir = f"/data/{run_tag}/eval"
    cmd = [
        "python",
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
