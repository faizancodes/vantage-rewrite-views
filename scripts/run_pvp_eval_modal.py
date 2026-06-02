"""Modal wrapper for PVP Step-7 eval.

Usage:
    modal run scripts/run_pvp_eval_modal.py::run --gpu L40S --n 500 --max-new-tokens 256
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
    .pip_install("flash-attn>=2.7,<3.0", extra_options="--no-build-isolation")
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

app = modal.App("pvp-eval", image=image)


def _run(
    *,
    gpu: str,
    run_tag: str,
    n: int,
    max_new_tokens: int,
    target: str,
    dtype: str,
    attn_impl: str,
    pld_method: str,
    pvp_method: str,
) -> dict:
    import json
    import os
    import subprocess

    os.chdir("/root/asts-spec")
    output_path = f"/data/pvp/{run_tag}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "scripts/run_pvp_eval.py",
        "--target", target,
        "--dtype", dtype,
        "--attn-impl", attn_impl,
        "--n", str(n),
        "--max-new-tokens", str(max_new_tokens),
        "--pld-method", pld_method,
        "--pvp-method", pvp_method,
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
            "schema": "asts-spec/pvp_eval/v1",
            "n_total": 0,
            "n_match": 0,
            "all_match": False,
            "overall_speedup": 0.0,
            "speedup_median": 0.0,
            "speedup_mean": 0.0,
            "records": [],
            "error": "no output written",
        }
    return {
        "exit_code": res.returncode,
        "output_path": output_path,
        "summary": {
            k: report.get(k)
            for k in (
                "n_total", "n_match", "all_match",
                "overall_speedup", "speedup_median", "speedup_mean",
                "pld_total_us", "pvp_total_us",
            )
        },
        "report": report,
    }


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=21600,
    cpu=4,
)
def eval_l40s(
    run_tag: str = "pvp_k2_v1",
    n: int = 500,
    max_new_tokens: int = 256,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bfloat16",
    attn_impl: str = "sdpa",
    pld_method: str = "blazedit_pld_w128_n10",
    pvp_method: str = "vantage_pvp_k2_w128_n10",
) -> dict:
    return _run(
        gpu="L40S", run_tag=run_tag,
        n=n, max_new_tokens=max_new_tokens,
        target=target, dtype=dtype, attn_impl=attn_impl,
        pld_method=pld_method, pvp_method=pvp_method,
    )


@app.local_entrypoint()
def run(
    gpu: str = "L40S",
    run_tag: str = "pvp_k2_v1",
    n: int = 500,
    max_new_tokens: int = 256,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bfloat16",
    attn_impl: str = "sdpa",
    pld_method: str = "blazedit_pld_w128_n10",
    pvp_method: str = "vantage_pvp_k2_w128_n10",
) -> None:
    import json

    if gpu.upper() != "L40S":
        raise SystemExit(f"only L40S supported, got {gpu!r}")

    result = eval_l40s.remote(
        run_tag=run_tag, n=n, max_new_tokens=max_new_tokens,
        target=target, dtype=dtype, attn_impl=attn_impl,
        pld_method=pld_method, pvp_method=pvp_method,
    )

    local_out = _PROJECT_ROOT / "analysis" / "pvp" / "runs" / f"{run_tag}.json"
    local_out.parent.mkdir(parents=True, exist_ok=True)
    local_out.write_text(json.dumps(result["report"], indent=2))

    print("\n=== PVP eval ===")
    for k, v in result["summary"].items():
        print(f"  {k:18s}: {v}")
    print(f"  artifact          : {local_out}")
    print(f"  remote artifact   : {result['output_path']}")
    print(f"  exit_code         : {result['exit_code']}")
