"""Modal app for the ASTS-Spec decision-gate microbenchmark.

Three functions, one shared volume:

    bench_treesitter   (CPU,  ~2 min, ~$0.01)
        runs scripts/bench_treesitter.py, writes /data/<run_tag>/treesitter.json

    bench_models       (L40S, ~10-15 min, ~$0.40 incl. one-time ~$0.10 download)
        loads Qwen2.5-Coder-7B + 0.5B, runs scripts/bench_models.py,
        writes /data/<run_tag>/models.json

    verdict            (CPU,  <1 min, free)
        runs scripts/verdict.py against the two JSONs above, prints a clear
        PROCEED/CAUTIOUS/PIVOT/KILL verdict, writes verdict.json

Usage:

    pip install -e .[modal]
    modal token new

    # full sweep (default)
    modal run modal_app.py

    # just the cheap CPU pieces (handy for iterating on the analysis)
    modal run modal_app.py --stage treesitter

    # just the GPU piece (assumes treesitter.json already on the volume)
    modal run modal_app.py --stage models

    # different target/draft
    modal run modal_app.py --target Qwen/Qwen2.5-Coder-7B --draft Qwen/Qwen2.5-Coder-1.5B

    # download artifacts
    modal volume get asts-spec-data v0/ ./out/
"""

from __future__ import annotations

from pathlib import Path

import modal


_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_TAG = "v0"


# ---------------------------------------------------------------------------
# Image: bake the package + deps so subprocess `python scripts/...` works.
# ---------------------------------------------------------------------------

image = (
    # Use CUDA devel base so nvcc + headers are available for flash-attn's
    # source build. Pip-installed nvidia-cuda-runtime alone has no compiler.
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    # `clang` (Ubuntu 22.04 → clang 14) is required by CUDA 13's nvcc; the
    # nvidia base image's clang++ stub reports version 0.0.0 which fails
    # CUDA's "clang++ >= 7.0" check during flash-attn's nvcc build.
    .apt_install("git", "build-essential", "clang")
    # Step 1: install everything except flash-attn so torch + transformers
    # are available before flash-attn's setup.py runs.
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
    # Step 2: install flash-attn with --no-build-isolation so it can pick up
    # the torch we just installed. nvcc is now available from the devel base
    # so source build will succeed. MAX_JOBS=2 avoids OOM during build.
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
            ".venv",
            "out",
            "__pycache__",
            "*.egg-info",
            ".pytest_cache",
            "*.pyc",
            ".git",
            "node_modules",
            ".vscode",
            ".idea",
            ".DS_Store",
        ],
    )
    .run_commands("cd /root/asts-spec && pip install -e . --quiet")
    .env({
        "HF_HOME": "/cache/huggingface",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
)


# ---------------------------------------------------------------------------
# Volumes:
#   - data_volume: benchmark output JSONs
#   - hf_cache:    Hugging Face model cache; survives across runs
# ---------------------------------------------------------------------------

data_volume = modal.Volume.from_name("asts-spec-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("asts-spec-hf-cache", create_if_missing=True)


app = modal.App("asts-spec", image=image)


# ---------------------------------------------------------------------------
# Stage 1: tree-sitter on CPU
# ---------------------------------------------------------------------------


@app.function(
    volumes={"/data": data_volume},
    timeout=600,
    cpu=2,
)
def bench_treesitter(
    run_tag: str = DEFAULT_RUN_TAG,
    iters_cold: int = 100,
    iters_inc: int = 100,
    iters_kstep: int = 50,
    k_values: str = "4,8,16",
) -> dict:
    """Run scripts/bench_treesitter.py inside the container."""
    import json
    import os
    import subprocess

    os.chdir("/root/asts-spec")
    output_path = f"/data/{run_tag}/treesitter.json"

    cmd = [
        "python", "scripts/bench_treesitter.py",
        "--output", output_path,
        "--iters-cold", str(iters_cold),
        "--iters-inc", str(iters_inc),
        "--iters-kstep", str(iters_kstep),
        "--k-values", k_values,
        "--log-level", "INFO",
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()

    with open(output_path) as f:
        report = json.load(f)
    return {
        "n_measurements": len(report["measurements"]),
        "summary": report["summary"],
        "output_path": output_path,
    }


# ---------------------------------------------------------------------------
# Stage 2: model benchmark on L40S
# ---------------------------------------------------------------------------


def _run_models_bench(
    run_tag: str,
    target: str,
    draft: str,
    prefix_lens: str,
    k_values: str,
    dtype: str,
    attn_impl: str,
    ar_iters: int,
    verify_iters: int,
    ar_warmup: int,
    verify_warmup: int,
) -> dict:
    """Inner driver shared by every GPU variant of bench_models_*."""
    import json
    import os
    import subprocess

    os.chdir("/root/asts-spec")
    output_path = f"/data/{run_tag}/models.json"

    cmd = [
        "python", "scripts/bench_models.py",
        "--output", output_path,
        "--target", target,
        "--draft", draft,
        "--prefix-lens", prefix_lens,
        "--k-values", k_values,
        "--dtype", dtype,
        "--attn-impl", attn_impl,
        "--ar-iters", str(ar_iters),
        "--verify-iters", str(verify_iters),
        "--ar-warmup", str(ar_warmup),
        "--verify-warmup", str(verify_warmup),
        "--log-level", "INFO",
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    hf_cache.commit()

    with open(output_path) as f:
        report = json.load(f)
    return {
        "n_measurements": len(report["measurements"]),
        "target_id": report["target_id"],
        "draft_id": report["draft_id"],
        "output_path": output_path,
    }


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=2400,
    cpu=4,
)
def bench_models_l40s(
    run_tag: str = DEFAULT_RUN_TAG,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    draft: str = "Qwen/Qwen2.5-Coder-0.5B",
    prefix_lens: str = "512,2048",
    k_values: str = "4,8,16",
    dtype: str = "bfloat16",
    attn_impl: str = "sdpa",
    ar_iters: int = 50,
    verify_iters: int = 30,
    ar_warmup: int = 10,
    verify_warmup: int = 5,
) -> dict:
    """Run scripts/bench_models.py inside an L40S 48GB container."""
    return _run_models_bench(
        run_tag, target, draft, prefix_lens, k_values, dtype, attn_impl,
        ar_iters, verify_iters, ar_warmup, verify_warmup,
    )


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="H100",
    timeout=2400,
    cpu=4,
)
def bench_models_h100(
    run_tag: str = DEFAULT_RUN_TAG,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    draft: str = "Qwen/Qwen2.5-Coder-0.5B",
    prefix_lens: str = "512,2048",
    k_values: str = "4,8,16",
    dtype: str = "bfloat16",
    attn_impl: str = "sdpa",
    ar_iters: int = 50,
    verify_iters: int = 30,
    ar_warmup: int = 10,
    verify_warmup: int = 5,
) -> dict:
    """Run scripts/bench_models.py inside an H100 SXM container.

    H100 has ~3.9× the memory bandwidth of L40S (3.35 TB/s vs 864 GB/s) so
    single-token AR latency should drop substantially. The interesting
    question is whether the draft/target ratio shifts in our favor.
    """
    return _run_models_bench(
        run_tag, target, draft, prefix_lens, k_values, dtype, attn_impl,
        ar_iters, verify_iters, ar_warmup, verify_warmup,
    )


# ---------------------------------------------------------------------------
# Stage 3: verdict on CPU
# ---------------------------------------------------------------------------


@app.function(
    volumes={"/data": data_volume},
    timeout=120,
    cpu=1,
)
def verdict(
    run_tag: str = DEFAULT_RUN_TAG,
    k: int = 8,
    a_values: str = "1,2,4,6,8",
) -> dict:
    """Run scripts/verdict.py to compute the projected speedup verdict."""
    import json
    import os
    import subprocess

    os.chdir("/root/asts-spec")
    ts_path = f"/data/{run_tag}/treesitter.json"
    models_path = f"/data/{run_tag}/models.json"
    output_path = f"/data/{run_tag}/verdict.json"

    if not os.path.exists(ts_path):
        raise RuntimeError(f"missing {ts_path}; run --stage treesitter first")
    if not os.path.exists(models_path):
        raise RuntimeError(f"missing {models_path}; run --stage models first")

    cmd = [
        "python", "scripts/verdict.py",
        "--treesitter", ts_path,
        "--models", models_path,
        "--output", output_path,
        "--k", str(k),
        "--a-values", a_values,
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()

    with open(output_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    stage: str = "all",
    run_tag: str = DEFAULT_RUN_TAG,
    gpu: str = "L40S",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    draft: str = "Qwen/Qwen2.5-Coder-0.5B",
    prefix_lens: str = "512,2048",
    k_values: str = "4,8,16",
    dtype: str = "bfloat16",
    attn_impl: str = "sdpa",
    ar_iters: int = 50,
    verify_iters: int = 30,
    ar_warmup: int = 10,
    verify_warmup: int = 5,
    iters_cold: int = 100,
    iters_inc: int = 100,
    iters_kstep: int = 50,
    verdict_k: int = 8,
    a_values: str = "1,2,4,6,8",
) -> None:
    """Drive the decision-gate microbenchmark.

    Stages: treesitter | models | verdict | all (default).
    GPU:    L40S (default) | H100  — picks which bench_models_* fn to invoke.
    """
    if stage not in {"treesitter", "models", "verdict", "all"}:
        raise SystemExit(f"--stage must be one of treesitter/models/verdict/all, got {stage!r}")

    gpu_norm = gpu.upper()
    if gpu_norm not in {"L40S", "H100"}:
        raise SystemExit(f"--gpu must be one of L40S/H100, got {gpu!r}")

    print()
    print("=== ASTS-Spec decision-gate microbenchmark ===")
    print(f"  stage:              {stage}")
    print(f"  run_tag:            {run_tag}")
    print(f"  gpu:                {gpu_norm}")
    print(f"  target:             {target}")
    print(f"  draft:              {draft}")
    print(f"  prefix_lens:        {prefix_lens}")
    print(f"  k_values:           {k_values}")
    print()

    if stage in ("treesitter", "all"):
        print(">>> Stage 1: tree-sitter parse-latency benchmark (CPU)")
        ts = bench_treesitter.remote(
            run_tag=run_tag,
            iters_cold=iters_cold,
            iters_inc=iters_inc,
            iters_kstep=iters_kstep,
            k_values=k_values,
        )
        print(f"  wrote {ts['n_measurements']} measurements → {ts['output_path']}")
        for lang, ops in ts["summary"].items():
            print(f"  {lang}:")
            for op, stats in sorted(ops.items()):
                print(
                    f"    {op:<25}  p50={stats['mean_p50_us']:>9.1f} us  "
                    f"p95={stats['mean_p95_us']:>9.1f} us"
                )
        print()

    if stage in ("models", "all"):
        print(f">>> Stage 2: model forward-pass benchmark ({gpu_norm})")
        bench_fn = bench_models_l40s if gpu_norm == "L40S" else bench_models_h100
        m = bench_fn.remote(
            run_tag=run_tag,
            target=target,
            draft=draft,
            prefix_lens=prefix_lens,
            k_values=k_values,
            dtype=dtype,
            attn_impl=attn_impl,
            ar_iters=ar_iters,
            verify_iters=verify_iters,
            ar_warmup=ar_warmup,
            verify_warmup=verify_warmup,
        )
        print(
            f"  wrote {m['n_measurements']} measurements → {m['output_path']}"
        )
        print()

    if stage in ("verdict", "all"):
        print(">>> Stage 3: decision-gate verdict (CPU)")
        v = verdict.remote(run_tag=run_tag, k=verdict_k, a_values=a_values)

        # Re-print the verdict from the local side using the same printer
        from asts.analysis import print_verdict
        print_verdict(v)

        decision = v["verdict"]
        print()
        print("=== Next step ===")
        if decision == "PROCEED":
            print("  Build the full ASTS-Spec prototype:")
            print("    1. Implement tree-sitter-driven variable-length speculation in vLLM/HF.")
            print("    2. Run end-to-end on HumanEval + RepoEval against EAGLE-2 / REST.")
        elif decision == "CAUTIOUS_PROCEED":
            print("  Proceed but engineer carefully:")
            print("    1. Amortize parse calls (e.g. one parse per draft, not per token).")
            print("    2. Profile the parser to confirm there's no easy speedup.")
        elif decision == "PIVOT":
            print("  Re-frame before building. Options:")
            print("    1. Drop tree-sitter for a brace-balance heuristic.")
            print("    2. Reframe contribution as quality (constrained gen), not speed.")
            print("    3. Add the per-AST-node-type policy as the headline contribution.")
        elif decision == "KILL":
            print("  Tree-sitter overhead is fatal. Pivot to:")
            print("    1. Pre-compiled grammar mask (XGrammar-style, no live parser).")
            print("    2. Or training-time AST signals (Verilog-paper-style [FRAG] tokens).")
        print()
