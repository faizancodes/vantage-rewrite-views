"""Debian runtime-install Modal entrypoint for emergency launches.

This is the lowest image-build path: the Modal image is plain Debian/Python and
all ML dependencies install when the container starts. It is slower per job than
the PyTorch image path, but it avoids registry/image-build stalls.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


_PROJECT_ROOT = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
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
            "analysis",
            "analysis/**",
            "artifacts",
            "artifacts/**",
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

app = modal.App("vantage-runtime-debian", image=image)

REAL_COMMIT_MV_STABLE_METHOD = "vantage_mv_pld_s96_x1_m16_t8_w128_n10"
REAL_COMMIT_MV_PREVIOUS_METHOD = "vantage_mv_pld_s64_m16_t8_w128_n10"


def _run_eagle_eval_impl(
    run_tag: str,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    target_trust_remote_code: bool = False,
    eagle_checkpoint: str = "/data/eagle_v1_normfix/eagle/eagle_final.pt",
    n: int = 50,
    max_new_tokens: int = 256,
    methods: str = "vanilla,blazedit_pld_w128_n10,vantage_transpld_w128_n10,vantage_routed_transpld_w128_n10",
    problem_jsonl: str = "",
    dtype: str = "bfloat16",
    attn_impl: str = "sdpa",
    code_proposer_fallback: str = "root",
    transpld_min_match_len: int = 4,
    chat_template: str = "none",
    task_id_file: str = "",
    task_router_json: str = "",
    task_router_exact_strong_threshold: int = 32,
    task_router_trans_margin: int = 0,
    pld_opportunity_trace: bool = False,
    pld_rerank_top_k: int = 4,
    pld_rerank_weights: str = "",
    pld_rerank_only_ambiguous: str = "true",
    pld_rerank_fallback: str = "baseline",
    pld_rerank_debug_trace: bool = False,
    pld_rerank_margin: float = 0.0,
    pld_rerank_margin_gate: str = "false",
    pld_rerank_always_include_baseline: str = "true",
    pld_rerank_enable_left_extension: str = "false",
    pld_rerank_left_extension_max: int = 128,
    pld_rerank_policy: str = "learned",
    pld_rerank_fixed_rank: int = 0,
    mtp_heads_checkpoint: str = "/data/pld_mtp/postpld_linear_k4_n917_v1/postpld_mtp_heads_k4_linear.pt",
    mtp_num_heads: int = 4,
    mtp_trigger_accepted_len: int = 4,
    mtp_position: str = "post_pld",
    mtp_disable: bool = False,
    mtp_queue_enabled: bool = True,
    mtp_use_queued_only_on_weak_pld: bool = True,
    mtp_disable_extra_verify: bool = False,
    weak_pld_router_path: str = "/data/pld_mtp/router_selected_k4_v2/weak_router/router.pkl",
    weak_pld_router_threshold: float | None = None,
    weak_pld_cap_tokens: int | None = None,
    lookahead_window: int = 8,
    lookahead_ngram: int = 4,
    lookahead_iters: int = 4,
    lookahead_max_draft: int = 16,
    lookahead_one_forward: bool = False,
    lookahead_stable_prefix: bool = True,
    lookahead_trajectory_cache: bool = True,
    pld_lookahead_router: str = "rule",
    pld_lookahead_router_path: str = "/data/pld_mtp/router_selected_k4_v2/weak_router/router.pkl",
    pld_lookahead_router_threshold: float = 0.3,
    pld_lookahead_weak_threshold: int = 4,
    pld_lookahead_trigger: str = "router_weak",
    pld_lookahead_mode: str = "replace_weak_pld",
    pld_lookahead_fallback: str = "pld",
    pld_lookahead_min_candidate_len: int = 1,
) -> dict:
    import json
    import os
    import subprocess
    import sys

    os.chdir("/root/asts-spec")
    install_cmds = [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
            "torch==2.5.1",
        ],
        [
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
            "scikit-learn>=1.4",
            "pydivsufsort>=0.0.18",
            "editdistance>=0.8.1",
        ],
    ]
    for install_cmd in install_cmds:
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
        "--eagle-checkpoint",
        eagle_checkpoint,
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
        "--transpld-min-match-len",
        str(transpld_min_match_len),
        "--chat-template",
        chat_template,
        "--pld-rerank-top-k",
        str(pld_rerank_top_k),
        "--pld-rerank-only-ambiguous",
        pld_rerank_only_ambiguous,
        "--pld-rerank-fallback",
        pld_rerank_fallback,
        "--pld-rerank-margin",
        str(pld_rerank_margin),
        "--pld-rerank-margin-gate",
        pld_rerank_margin_gate,
        "--pld-rerank-always-include-baseline",
        pld_rerank_always_include_baseline,
        "--pld-rerank-enable-left-extension",
        pld_rerank_enable_left_extension,
        "--pld-rerank-left-extension-max",
        str(pld_rerank_left_extension_max),
        "--pld-rerank-policy",
        pld_rerank_policy,
        "--pld-rerank-fixed-rank",
        str(pld_rerank_fixed_rank),
        "--mtp-heads-checkpoint",
        mtp_heads_checkpoint,
        "--mtp-num-heads",
        str(mtp_num_heads),
        "--mtp-trigger-accepted-len",
        str(mtp_trigger_accepted_len),
        "--mtp-position",
        mtp_position,
        "--mtp-queue-enabled" if mtp_queue_enabled else "--no-mtp-queue-enabled",
        "--mtp-use-queued-only-on-weak-pld"
        if mtp_use_queued_only_on_weak_pld
        else "--no-mtp-use-queued-only-on-weak-pld",
        "--mtp-disable-extra-verify"
        if mtp_disable_extra_verify
        else "--no-mtp-disable-extra-verify",
        "--log-level",
        "INFO",
    ]
    cmd += ["--weak-pld-router-path", weak_pld_router_path]
    cmd += [
        "--lookahead-window",
        str(lookahead_window),
        "--lookahead-ngram",
        str(lookahead_ngram),
        "--lookahead-iters",
        str(lookahead_iters),
        "--lookahead-max-draft",
        str(lookahead_max_draft),
        "--lookahead-one-forward" if lookahead_one_forward else "--no-lookahead-one-forward",
        "--lookahead-stable-prefix" if lookahead_stable_prefix else "--no-lookahead-stable-prefix",
        "--lookahead-trajectory-cache"
        if lookahead_trajectory_cache
        else "--no-lookahead-trajectory-cache",
        "--pld-lookahead-router",
        pld_lookahead_router,
        "--pld-lookahead-router-path",
        pld_lookahead_router_path,
        "--pld-lookahead-router-threshold",
        str(pld_lookahead_router_threshold),
        "--pld-lookahead-weak-threshold",
        str(pld_lookahead_weak_threshold),
        "--pld-lookahead-trigger",
        pld_lookahead_trigger,
        "--pld-lookahead-mode",
        pld_lookahead_mode,
        "--pld-lookahead-fallback",
        pld_lookahead_fallback,
        "--pld-lookahead-min-candidate-len",
        str(pld_lookahead_min_candidate_len),
    ]
    if weak_pld_router_threshold is not None:
        cmd += ["--weak-pld-router-threshold", str(weak_pld_router_threshold)]
    if weak_pld_cap_tokens is not None:
        cmd += ["--weak-pld-cap-tokens", str(weak_pld_cap_tokens)]
    if mtp_disable:
        cmd.append("--mtp-disable")
    if pld_rerank_weights:
        cmd += ["--pld-rerank-weights", pld_rerank_weights]
    if pld_rerank_debug_trace:
        cmd.append("--pld-rerank-debug-trace")
    if task_id_file:
        cmd += ["--task-id-file", task_id_file]
    if task_router_json:
        cmd += [
            "--task-router-json",
            task_router_json,
            "--task-router-exact-strong-threshold",
            str(task_router_exact_strong_threshold),
            "--task-router-trans-margin",
            str(task_router_trans_margin),
        ]
    if target_trust_remote_code:
        cmd.append("--target-trust-remote-code")
    if pld_opportunity_trace:
        cmd.append("--pld-opportunity-trace")
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
    timeout=10800,
    startup_timeout=3600,
    cpu=4,
    cloud="aws",
    region="us-west",
)
def run_eagle_eval_job(
    run_tag: str,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    target_trust_remote_code: bool = False,
    eagle_checkpoint: str = "/data/eagle_v1_normfix/eagle/eagle_final.pt",
    n: int = 50,
    max_new_tokens: int = 256,
    methods: str = "vanilla,blazedit_pld_w128_n10,vantage_transpld_w128_n10,vantage_routed_transpld_w128_n10",
    problem_jsonl: str = "",
    dtype: str = "bfloat16",
    attn_impl: str = "sdpa",
    code_proposer_fallback: str = "root",
    transpld_min_match_len: int = 4,
    chat_template: str = "none",
    task_id_file: str = "",
    task_router_json: str = "",
    task_router_exact_strong_threshold: int = 32,
    task_router_trans_margin: int = 0,
    pld_rerank_top_k: int = 4,
    pld_rerank_weights: str = "",
    pld_rerank_only_ambiguous: str = "true",
    pld_rerank_fallback: str = "baseline",
    pld_rerank_debug_trace: bool = False,
    pld_rerank_margin: float = 0.0,
    pld_rerank_margin_gate: str = "false",
    pld_rerank_always_include_baseline: str = "true",
    pld_rerank_enable_left_extension: str = "false",
    pld_rerank_left_extension_max: int = 128,
    pld_rerank_policy: str = "learned",
    pld_rerank_fixed_rank: int = 0,
    mtp_heads_checkpoint: str = "/data/pld_mtp/postpld_linear_k4_n917_v1/postpld_mtp_heads_k4_linear.pt",
    mtp_num_heads: int = 4,
    mtp_trigger_accepted_len: int = 4,
    mtp_position: str = "post_pld",
    mtp_disable: bool = False,
    mtp_queue_enabled: bool = True,
    mtp_use_queued_only_on_weak_pld: bool = True,
    mtp_disable_extra_verify: bool = False,
    lookahead_window: int = 8,
    lookahead_ngram: int = 4,
    lookahead_iters: int = 4,
    lookahead_max_draft: int = 16,
    lookahead_one_forward: bool = False,
    pld_lookahead_router: str = "rule",
    pld_lookahead_router_threshold: float = 0.3,
    pld_lookahead_trigger: str = "router_weak",
) -> dict:
    return _run_eagle_eval_impl(
        run_tag=run_tag,
        target=target,
        target_trust_remote_code=target_trust_remote_code,
        eagle_checkpoint=eagle_checkpoint,
        n=n,
        max_new_tokens=max_new_tokens,
        methods=methods,
        problem_jsonl=problem_jsonl,
        dtype=dtype,
        attn_impl=attn_impl,
        code_proposer_fallback=code_proposer_fallback,
        transpld_min_match_len=transpld_min_match_len,
        chat_template=chat_template,
        task_id_file=task_id_file,
        task_router_json=task_router_json,
        task_router_exact_strong_threshold=task_router_exact_strong_threshold,
        task_router_trans_margin=task_router_trans_margin,
        pld_rerank_top_k=pld_rerank_top_k,
        pld_rerank_weights=pld_rerank_weights,
        pld_rerank_only_ambiguous=pld_rerank_only_ambiguous,
        pld_rerank_fallback=pld_rerank_fallback,
        pld_rerank_debug_trace=pld_rerank_debug_trace,
        pld_rerank_margin=pld_rerank_margin,
        pld_rerank_margin_gate=pld_rerank_margin_gate,
        pld_rerank_always_include_baseline=pld_rerank_always_include_baseline,
        pld_rerank_enable_left_extension=pld_rerank_enable_left_extension,
        pld_rerank_left_extension_max=pld_rerank_left_extension_max,
        pld_rerank_policy=pld_rerank_policy,
        pld_rerank_fixed_rank=pld_rerank_fixed_rank,
        mtp_heads_checkpoint=mtp_heads_checkpoint,
        mtp_num_heads=mtp_num_heads,
        mtp_trigger_accepted_len=mtp_trigger_accepted_len,
        mtp_position=mtp_position,
        mtp_disable=mtp_disable,
        mtp_queue_enabled=mtp_queue_enabled,
        mtp_use_queued_only_on_weak_pld=mtp_use_queued_only_on_weak_pld,
        mtp_disable_extra_verify=mtp_disable_extra_verify,
        lookahead_window=lookahead_window,
        lookahead_ngram=lookahead_ngram,
        lookahead_iters=lookahead_iters,
        lookahead_max_draft=lookahead_max_draft,
        lookahead_one_forward=lookahead_one_forward,
        pld_lookahead_router=pld_lookahead_router,
        pld_lookahead_router_threshold=pld_lookahead_router_threshold,
        pld_lookahead_trigger=pld_lookahead_trigger,
    )


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=10800,
    startup_timeout=3600,
    cpu=4,
)
def run_eagle_eval_job_any(
    run_tag: str,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    target_trust_remote_code: bool = False,
    eagle_checkpoint: str = "/data/eagle_v1_normfix/eagle/eagle_final.pt",
    n: int = 50,
    max_new_tokens: int = 256,
    methods: str = "vanilla,blazedit_pld_w128_n10,vantage_transpld_w128_n10,vantage_routed_transpld_w128_n10",
    problem_jsonl: str = "",
    dtype: str = "bfloat16",
    attn_impl: str = "sdpa",
    code_proposer_fallback: str = "root",
    transpld_min_match_len: int = 4,
    chat_template: str = "none",
    task_id_file: str = "",
    task_router_json: str = "",
    task_router_exact_strong_threshold: int = 32,
    task_router_trans_margin: int = 0,
    pld_opportunity_trace: bool = False,
    pld_rerank_top_k: int = 4,
    pld_rerank_weights: str = "",
    pld_rerank_only_ambiguous: str = "true",
    pld_rerank_fallback: str = "baseline",
    pld_rerank_debug_trace: bool = False,
    pld_rerank_margin: float = 0.0,
    pld_rerank_margin_gate: str = "false",
    pld_rerank_always_include_baseline: str = "true",
    pld_rerank_enable_left_extension: str = "false",
    pld_rerank_left_extension_max: int = 128,
    pld_rerank_policy: str = "learned",
    pld_rerank_fixed_rank: int = 0,
    mtp_heads_checkpoint: str = "/data/pld_mtp/postpld_linear_k4_n917_v1/postpld_mtp_heads_k4_linear.pt",
    mtp_num_heads: int = 4,
    mtp_trigger_accepted_len: int = 4,
    mtp_position: str = "post_pld",
    mtp_disable: bool = False,
    mtp_queue_enabled: bool = True,
    mtp_use_queued_only_on_weak_pld: bool = True,
    mtp_disable_extra_verify: bool = False,
    lookahead_window: int = 8,
    lookahead_ngram: int = 4,
    lookahead_iters: int = 4,
    lookahead_max_draft: int = 16,
    lookahead_one_forward: bool = False,
    pld_lookahead_router: str = "rule",
    pld_lookahead_router_threshold: float = 0.3,
    pld_lookahead_trigger: str = "router_weak",
) -> dict:
    return _run_eagle_eval_impl(
        run_tag=run_tag,
        target=target,
        target_trust_remote_code=target_trust_remote_code,
        eagle_checkpoint=eagle_checkpoint,
        n=n,
        max_new_tokens=max_new_tokens,
        methods=methods,
        problem_jsonl=problem_jsonl,
        dtype=dtype,
        attn_impl=attn_impl,
        code_proposer_fallback=code_proposer_fallback,
        transpld_min_match_len=transpld_min_match_len,
        chat_template=chat_template,
        task_id_file=task_id_file,
        task_router_json=task_router_json,
        task_router_exact_strong_threshold=task_router_exact_strong_threshold,
        task_router_trans_margin=task_router_trans_margin,
        pld_opportunity_trace=pld_opportunity_trace,
        pld_rerank_top_k=pld_rerank_top_k,
        pld_rerank_weights=pld_rerank_weights,
        pld_rerank_only_ambiguous=pld_rerank_only_ambiguous,
        pld_rerank_fallback=pld_rerank_fallback,
        pld_rerank_debug_trace=pld_rerank_debug_trace,
        pld_rerank_margin=pld_rerank_margin,
        pld_rerank_margin_gate=pld_rerank_margin_gate,
        pld_rerank_always_include_baseline=pld_rerank_always_include_baseline,
        pld_rerank_enable_left_extension=pld_rerank_enable_left_extension,
        pld_rerank_left_extension_max=pld_rerank_left_extension_max,
        pld_rerank_policy=pld_rerank_policy,
        pld_rerank_fixed_rank=pld_rerank_fixed_rank,
        mtp_heads_checkpoint=mtp_heads_checkpoint,
        mtp_num_heads=mtp_num_heads,
        mtp_trigger_accepted_len=mtp_trigger_accepted_len,
        mtp_position=mtp_position,
        mtp_disable=mtp_disable,
        mtp_queue_enabled=mtp_queue_enabled,
        mtp_use_queued_only_on_weak_pld=mtp_use_queued_only_on_weak_pld,
        mtp_disable_extra_verify=mtp_disable_extra_verify,
        lookahead_window=lookahead_window,
        lookahead_ngram=lookahead_ngram,
        lookahead_iters=lookahead_iters,
        lookahead_max_draft=lookahead_max_draft,
        lookahead_one_forward=lookahead_one_forward,
        pld_lookahead_router=pld_lookahead_router,
        pld_lookahead_router_threshold=pld_lookahead_router_threshold,
        pld_lookahead_trigger=pld_lookahead_trigger,
    )


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=7200,
    startup_timeout=1800,
    cpu=4,
)
def run_mtp_head_benchmark_job(
    heads: str = "/data/pld_mtp/postpld_linear_k4_n917_v1/postpld_mtp_heads_k4_linear.pt",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    num_heads: int = 4,
    iters: int = 1000,
    dtype: str = "bf16",
) -> dict:
    import json
    import os
    import subprocess
    import sys

    os.chdir("/root/asts-spec")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
            "torch==2.5.1",
        ],
        check=True,
    )
    out_json = "/data/pld_mtp/head_overhead_benchmark.json"
    cmd = [
        sys.executable,
        "scripts/benchmark_mtp_head_overhead.py",
        "--heads",
        heads,
        "--target",
        target,
        "--num-heads",
        str(num_heads),
        "--device",
        "cuda",
        "--dtype",
        dtype,
        "--iters",
        str(iters),
        "--output-json",
        out_json,
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    with open(out_json, "r", encoding="utf-8") as f:
        return json.load(f)


@app.local_entrypoint()
def launch_mtp_head_benchmark(
    wait: bool = True,
    heads: str = "/data/pld_mtp/postpld_linear_k4_n917_v1/postpld_mtp_heads_k4_linear.pt",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    num_heads: int = 4,
    iters: int = 1000,
    dtype: str = "bf16",
) -> None:
    call = run_mtp_head_benchmark_job.spawn(
        heads=heads,
        target=target,
        num_heads=num_heads,
        iters=iters,
        dtype=dtype,
    )
    print(f"mtp_head_benchmark\t{call.object_id}", flush=True)
    if wait:
        report = call.get()
        print(
            "DONE mtp_head_benchmark: "
            f"avg={report.get('avg_head_compute_ms', 0.0):.3f}ms "
            f"p50={report.get('p50_head_compute_ms', 0.0):.3f}ms "
            f"p90={report.get('p90_head_compute_ms', 0.0):.3f}ms "
            f"p99={report.get('p99_head_compute_ms', 0.0):.3f}ms",
            flush=True,
        )


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=7200,
    startup_timeout=1800,
    cpu=4,
)
def run_pld_verify_length_benchmark_job(
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn_impl: str = "sdpa",
    prefix_len: int = 1024,
    lengths: str = "0,1,2,4,8,16,32,64,128",
    iters: int = 100,
    warmup: int = 20,
    version: str = "v1",
) -> dict:
    import json
    import os
    import subprocess
    import sys

    os.chdir("/root/asts-spec")
    install_cmds = [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
            "torch==2.5.1",
        ],
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "transformers>=4.46",
            "accelerate>=1.0",
            "huggingface-hub>=0.26",
        ],
    ]
    for install_cmd in install_cmds:
        print(f"$ {' '.join(install_cmd)}", flush=True)
        subprocess.run(install_cmd, check=True)

    out_dir = f"/data/pld_verify_length_benchmark/{version}"
    out_json = f"{out_dir}/report.json"
    out_md = f"{out_dir}/report.md"
    cmd = [
        sys.executable,
        "scripts/benchmark_pld_verify_lengths.py",
        "--target",
        target,
        "--dtype",
        dtype,
        "--attn-impl",
        attn_impl,
        "--device",
        "cuda",
        "--prefix-len",
        str(prefix_len),
        "--lengths",
        lengths,
        "--iters",
        str(iters),
        "--warmup",
        str(warmup),
        "--output-json",
        out_json,
        "--output-md",
        out_md,
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    with open(out_json, "r", encoding="utf-8") as f:
        return json.load(f)


@app.local_entrypoint()
def launch_pld_verify_length_benchmark(
    wait: bool = True,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn_impl: str = "sdpa",
    prefix_len: int = 1024,
    lengths: str = "0,1,2,4,8,16,32,64,128",
    iters: int = 100,
    warmup: int = 20,
    version: str = "v1",
) -> None:
    call = run_pld_verify_length_benchmark_job.spawn(
        target=target,
        dtype=dtype,
        attn_impl=attn_impl,
        prefix_len=prefix_len,
        lengths=lengths,
        iters=iters,
        warmup=warmup,
        version=version,
    )
    print(f"pld_verify_length_benchmark\t{call.object_id}", flush=True)
    if wait:
        report = call.get()
        fwd = report.get("forward_fit", {})
        full = report.get("full_verify_fit", {})
        print(
            "DONE pld_verify_length_benchmark: "
            f"forward_fixed={fwd.get('intercept_ms', 0.0):.3f}ms "
            f"forward_inc={fwd.get('slope_ms_per_token', 0.0):.4f}ms/token "
            f"full_fixed={full.get('intercept_ms', 0.0):.3f}ms "
            f"full_inc={full.get('slope_ms_per_token', 0.0):.4f}ms/token",
            flush=True,
        )


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=7200,
    startup_timeout=1800,
    cpu=4,
)
def run_blazedit_pld_profile_job(
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn_impl: str = "sdpa",
    split: str = "test",
    n: int = 5,
    max_new_tokens: int = 256,
    torch_profiler_steps: int = 40,
    sync_after_forward: bool = False,
    version: str = "v1",
) -> dict:
    import json
    import os
    import subprocess
    import sys

    os.chdir("/root/asts-spec")
    install_cmds = [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
            "torch==2.5.1",
        ],
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "transformers>=4.46",
            "accelerate>=1.0",
            "huggingface-hub>=0.26",
            "datasets>=3.0",
            "numpy>=1.26",
        ],
    ]
    for install_cmd in install_cmds:
        print(f"$ {' '.join(install_cmd)}", flush=True)
        subprocess.run(install_cmd, check=True)

    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    problem_jsonl = f"/root/asts-spec/data/real_commits/path_a_{split}500_v1.jsonl"
    out_dir = f"/data/pld_runtime_profile/{version}"
    cmd = [
        sys.executable,
        "scripts/profile_blazedit_pld_runtime.py",
        "--target",
        target,
        "--dtype",
        dtype,
        "--attn-impl",
        attn_impl,
        "--device",
        "cuda",
        "--problem-jsonl",
        problem_jsonl,
        "--n",
        str(n),
        "--max-new-tokens",
        str(max_new_tokens),
        "--torch-profiler-steps",
        str(torch_profiler_steps),
    ]
    if sync_after_forward:
        cmd.append("--sync-after-forward")
    cmd.extend(["--output-dir", out_dir])
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    audit_reports = {}
    if write_audit_traces:
        for raw_batch in batch_sizes.split(","):
            raw_batch = raw_batch.strip()
            if not raw_batch:
                continue
            trace_path = f"{out_dir}/batch{raw_batch}_audit_trace.jsonl"
            audit_dir = f"/data/batched_pld_task_audit/{version}/batch{raw_batch}"
            audit_cmd = [
                sys.executable,
                "scripts/audit_batched_pld_task_isolation.py",
                "--trace",
                trace_path,
                "--output-dir",
                audit_dir,
            ]
            print(f"$ {' '.join(audit_cmd)}", flush=True)
            subprocess.run(audit_cmd, check=True)
            with open(f"{audit_dir}/report.json", "r", encoding="utf-8") as f:
                audit_reports[f"batch{raw_batch}"] = json.load(f)
    data_volume.commit()
    with open(f"{out_dir}/report.json", "r", encoding="utf-8") as f:
        report = json.load(f)
    report["audit_reports"] = audit_reports
    return report


@app.local_entrypoint()
def launch_blazedit_pld_profile(
    wait: bool = True,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn_impl: str = "sdpa",
    split: str = "test",
    n: int = 5,
    max_new_tokens: int = 256,
    torch_profiler_steps: int = 40,
    sync_after_forward: bool = False,
    version: str = "v1",
) -> None:
    call = run_blazedit_pld_profile_job.spawn(
        target=target,
        dtype=dtype,
        attn_impl=attn_impl,
        split=split,
        n=n,
        max_new_tokens=max_new_tokens,
        torch_profiler_steps=torch_profiler_steps,
        sync_after_forward=sync_after_forward,
        version=version,
    )
    print(f"blazedit_pld_profile\t{call.object_id}", flush=True)
    if wait:
        report = call.get()
        agg = report.get("aggregate", {})
        print(
            "DONE blazedit_pld_profile: "
            f"steps={agg.get('steps', 0)} "
            f"wall={agg.get('step_wall_us_mean', 0.0) / 1000.0:.3f}ms "
            f"lookup={agg.get('lookup_us_mean', 0.0) / 1000.0:.3f}ms "
            f"tensor={agg.get('tensor_create_us_mean', 0.0) / 1000.0:.3f}ms "
            f"forward_cuda={agg.get('model_forward_cuda_us_mean', 0.0) / 1000.0:.3f}ms "
            f"greedy={agg.get('greedy_us_mean', 0.0) / 1000.0:.3f}ms "
            f"crop={agg.get('post_crop_us_mean', 0.0) / 1000.0:.3f}ms",
            flush=True,
        )


def _install_forward_bench_deps() -> None:
    import subprocess
    import sys

    install_cmds = [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
            "torch==2.5.1",
        ],
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "transformers>=4.46",
            "accelerate>=1.0",
            "huggingface-hub>=0.26",
            "datasets>=3.0",
            "numpy>=1.26",
        ],
    ]
    for install_cmd in install_cmds:
        print(f"$ {' '.join(install_cmd)}", flush=True)
        subprocess.run(install_cmd, check=True)


def _install_external_baseline_deps(kind: str, vllm_package: str = "vllm") -> None:
    import subprocess
    import sys

    if kind == "vllm":
        install_cmds = [
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                vllm_package,
                "transformers>=4.46",
                "huggingface-hub>=0.26",
                "datasets>=3.0",
                "numpy>=1.26",
            ]
        ]
    elif kind == "hf":
        install_cmds = [
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--index-url",
                "https://download.pytorch.org/whl/cu124",
                "torch==2.5.1",
            ],
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "transformers>=4.46",
                "accelerate>=1.0",
                "huggingface-hub>=0.26",
                "datasets>=3.0",
                "numpy>=1.26",
            ],
        ]
    else:
        raise ValueError(f"unknown external baseline dependency kind: {kind}")
    for install_cmd in install_cmds:
        print(f"$ {' '.join(install_cmd)}", flush=True)
        subprocess.run(install_cmd, check=True)


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=24 * 60 * 60,
    startup_timeout=3600,
    cpu=8,
)
def run_external_baseline_job(
    version: str = "external_baselines_gpu_attempt_v1",
    baseline: str = "vllm_greedy",
    split: str = "test",
    n: int = 500,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    max_new_tokens: int = 256,
    problem_jsonl: str = "",
    prompt_lookup_num_tokens: int = 128,
    ngram_prompt_lookup_max: int = 128,
    num_speculative_tokens: int = 8,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.90,
    enforce_eager: bool = False,
) -> dict:
    import json
    import os
    import subprocess
    import sys
    import traceback

    os.chdir("/root/asts-spec")
    if not problem_jsonl:
        problem_jsonl = (
            f"/root/asts-spec/data/real_commits/real_commit_manifest_balanced_1000_v2_{split}500.jsonl"
        )
    out_dir = f"/data/external_baselines/{version}/{baseline}"
    if baseline in {"vllm_greedy", "vllm_ngram"}:
        _install_external_baseline_deps("vllm")
        cmd = [
            sys.executable,
            "scripts/run_vllm_baseline_eval.py",
            "--problem-jsonl",
            problem_jsonl,
            "--n",
            str(n),
            "--target",
            target,
            "--dtype",
            dtype,
            "--max-new-tokens",
            str(max_new_tokens),
            "--backend",
            "ngram_speculation" if baseline == "vllm_ngram" else "greedy",
            "--ngram-prompt-lookup-max",
            str(ngram_prompt_lookup_max),
            "--num-speculative-tokens",
            str(num_speculative_tokens),
            "--max-model-len",
            str(max_model_len),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--output-dir",
            out_dir,
        ]
        if enforce_eager:
            cmd.append("--enforce-eager")
    elif baseline == "hf_prompt_lookup":
        _install_external_baseline_deps("hf")
        cmd = [
            sys.executable,
            "scripts/run_hf_prompt_lookup_baseline.py",
            "--problem-jsonl",
            problem_jsonl,
            "--n",
            str(n),
            "--target",
            target,
            "--dtype",
            dtype,
            "--device",
            "cuda",
            "--max-new-tokens",
            str(max_new_tokens),
            "--prompt-lookup-num-tokens",
            str(prompt_lookup_num_tokens),
            "--output-dir",
            out_dir,
        ]
    else:
        raise ValueError(f"unknown baseline: {baseline}")

    print(f"$ {' '.join(cmd)}", flush=True)
    rc = 1
    launch_error = ""
    try:
        proc = subprocess.run(cmd, text=True)
        rc = int(proc.returncode)
    except BaseException as exc:
        launch_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    data_volume.commit()
    report_path = f"{out_dir}/report.json"
    markdown_path = f"{out_dir}/report.md"
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    else:
        report = {
            "status": "failed",
            "failure": {
                "type": "launcher_error",
                "message": launch_error or f"baseline exited {rc} without report.json",
            },
            "config": {
                "version": version,
                "baseline": baseline,
                "split": split,
                "n": n,
                "target": target,
                "dtype": dtype,
                "max_new_tokens": max_new_tokens,
            },
            "command": cmd,
        }
    report["modal_returncode"] = rc
    report["modal_output_dir"] = out_dir
    if os.path.exists(markdown_path):
        report["report_markdown"] = open(markdown_path, "r", encoding="utf-8").read()
    return report


@app.local_entrypoint()
def launch_external_baseline(
    baseline: str = "vllm_greedy",
    split: str = "test",
    n: int = 500,
    wait: bool = True,
    version: str = "external_baselines_gpu_attempt_v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    max_new_tokens: int = 256,
    problem_jsonl: str = "",
    prompt_lookup_num_tokens: int = 128,
    ngram_prompt_lookup_max: int = 128,
    num_speculative_tokens: int = 8,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.90,
    enforce_eager: bool = False,
) -> None:
    call = run_external_baseline_job.spawn(
        version=version,
        baseline=baseline,
        split=split,
        n=n,
        target=target,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        problem_jsonl=problem_jsonl,
        prompt_lookup_num_tokens=prompt_lookup_num_tokens,
        ngram_prompt_lookup_max=ngram_prompt_lookup_max,
        num_speculative_tokens=num_speculative_tokens,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
    )
    print(f"external_baseline\t{baseline}\t{call.object_id}", flush=True)
    if wait:
        report = call.get()
        local_dir = (
            _PROJECT_ROOT
            / "analysis"
            / "external_baselines"
            / version
            / baseline
        )
        local_dir.mkdir(parents=True, exist_ok=True)
        markdown = report.pop("report_markdown", "")
        (local_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if markdown:
            (local_dir / "report.md").write_text(markdown)
        status = report.get("status", "unknown")
        result = report.get("result", {})
        print(
            "DONE external_baseline: "
            f"baseline={baseline} status={status} "
            f"tok_s={result.get('tokens_per_sec', 'n/a')}",
            flush=True,
        )


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=24 * 60 * 60,
    startup_timeout=3600,
    cpu=8,
)
def run_vllm_benchmark_job(
    version: str = "vantage_vllm_smoke_v1",
    method: str = "greedy",
    split: str = "test",
    n: int = 20,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    max_new_tokens: int = 256,
    problem_jsonl: str = "",
    ngram_prompt_lookup_min: int = 2,
    ngram_prompt_lookup_max: int = 128,
    vantage_match_tokens: int = 10,
    vantage_window_tokens: int = 128,
    num_speculative_tokens: int = 8,
    max_model_len: int = 12288,
    gpu_memory_utilization: float = 0.90,
    enforce_eager: bool = False,
    custom_proposer_module: str = "vantage_vllm.minimal_custom_proposer",
    custom_proposer_class: str = "MinimalCustomProposer",
    custom_config_variant: str = "legacy_custom",
    vllm_package: str = "vllm",
    vantage_pld_trace_sample_rate: float = 1.0,
    vantage_pld_trace_tokens: bool = False,
    vantage_pld_patch_strict: bool = True,
    vantage_pld_patch_mode: str = "pld_python",
    vantage_pld_numba: bool = True,
) -> dict:
    import json
    import os
    import subprocess
    import sys
    import traceback

    os.chdir("/root/asts-spec")
    _install_external_baseline_deps("vllm", vllm_package=vllm_package)
    if not problem_jsonl:
        problem_jsonl = (
            f"/root/asts-spec/data/real_commits/real_commit_manifest_balanced_1000_v2_{split}500.jsonl"
        )
    out_dir = f"/data/vllm_results/{version}/{method}"
    cmd = [
        sys.executable,
        "scripts/run_vllm_benchmarks.py",
        "--manifest-path",
        problem_jsonl,
        "--split",
        split,
        "--n",
        str(n),
        "--model",
        target,
        "--dtype",
        dtype,
        "--max-new-tokens",
        str(max_new_tokens),
        "--method",
        method,
        "--ngram-prompt-lookup-min",
        str(ngram_prompt_lookup_min),
        "--ngram-prompt-lookup-max",
        str(ngram_prompt_lookup_max),
        "--vantage-match-tokens",
        str(vantage_match_tokens),
        "--vantage-window-tokens",
        str(vantage_window_tokens),
        "--num-speculative-tokens",
        str(num_speculative_tokens),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--custom-proposer-module",
        custom_proposer_module,
        "--custom-proposer-class",
        custom_proposer_class,
        "--custom-config-variant",
        custom_config_variant,
        "--vantage-pld-trace-sample-rate",
        str(vantage_pld_trace_sample_rate),
        "--vantage-pld-patch-mode",
        vantage_pld_patch_mode,
        "--run-id",
        f"{version}_{method}",
        "--output-dir",
        out_dir,
    ]
    if vantage_pld_trace_tokens:
        cmd.append("--vantage-pld-trace-tokens")
    if not vantage_pld_patch_strict:
        cmd.append("--no-vantage-pld-patch-strict")
    if not vantage_pld_numba:
        cmd.append("--no-vantage-pld-numba")
    if enforce_eager:
        cmd.append("--enforce-eager")

    print(f"$ {' '.join(cmd)}", flush=True)
    rc = 1
    launch_error = ""
    try:
        proc = subprocess.run(cmd, text=True)
        rc = int(proc.returncode)
    except BaseException as exc:
        launch_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    data_volume.commit()

    files = {}
    for name in (
        "config.json",
        "run_summary.json",
        "outputs.jsonl",
        "raw_stdout.txt",
        "raw_stderr.txt",
        "minimal_proposer_events.jsonl",
        "patch_report.json",
        "proposer_trace.jsonl",
    ):
        path = f"{out_dir}/{name}"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                files[name] = f.read()

    if "run_summary.json" in files:
        summary = json.loads(files["run_summary.json"])
    else:
        summary = {
            "status": "failed",
            "failure": {
                "type": "launcher_error",
                "message": launch_error or f"benchmark exited {rc} without run_summary.json",
            },
            "method": method,
            "run_id": f"{version}_{method}",
        }
    summary["modal_returncode"] = rc
    summary["modal_output_dir"] = out_dir
    summary["command"] = cmd
    return {"summary": summary, "files": files}


@app.local_entrypoint()
def launch_vllm_benchmark(
    method: str = "greedy",
    split: str = "test",
    n: int = 20,
    wait: bool = True,
    version: str = "vantage_vllm_smoke_v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    max_new_tokens: int = 256,
    problem_jsonl: str = "",
    ngram_prompt_lookup_min: int = 2,
    ngram_prompt_lookup_max: int = 128,
    vantage_match_tokens: int = 10,
    vantage_window_tokens: int = 128,
    num_speculative_tokens: int = 8,
    max_model_len: int = 12288,
    gpu_memory_utilization: float = 0.90,
    enforce_eager: bool = False,
    custom_proposer_module: str = "vantage_vllm.minimal_custom_proposer",
    custom_proposer_class: str = "MinimalCustomProposer",
    custom_config_variant: str = "legacy_custom",
    vllm_package: str = "vllm",
    vantage_pld_trace_sample_rate: float = 1.0,
    vantage_pld_trace_tokens: bool = False,
    vantage_pld_patch_strict: bool = True,
    vantage_pld_patch_mode: str = "pld_python",
    vantage_pld_numba: bool = True,
) -> None:
    call = run_vllm_benchmark_job.spawn(
        version=version,
        method=method,
        split=split,
        n=n,
        target=target,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        problem_jsonl=problem_jsonl,
        ngram_prompt_lookup_min=ngram_prompt_lookup_min,
        ngram_prompt_lookup_max=ngram_prompt_lookup_max,
        vantage_match_tokens=vantage_match_tokens,
        vantage_window_tokens=vantage_window_tokens,
        num_speculative_tokens=num_speculative_tokens,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        custom_proposer_module=custom_proposer_module,
        custom_proposer_class=custom_proposer_class,
        custom_config_variant=custom_config_variant,
        vllm_package=vllm_package,
        vantage_pld_trace_sample_rate=vantage_pld_trace_sample_rate,
        vantage_pld_trace_tokens=vantage_pld_trace_tokens,
        vantage_pld_patch_strict=vantage_pld_patch_strict,
        vantage_pld_patch_mode=vantage_pld_patch_mode,
        vantage_pld_numba=vantage_pld_numba,
    )
    print(f"vllm_benchmark\t{method}\t{call.object_id}", flush=True)
    if wait:
        payload = call.get()
        summary = payload.get("summary", {})
        local_dir = _PROJECT_ROOT / "artifacts" / "vllm_results" / version / method
        local_dir.mkdir(parents=True, exist_ok=True)
        for name, text in payload.get("files", {}).items():
            (local_dir / name).write_text(text)
        (local_dir / "modal_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        print(
            "DONE vllm_benchmark: "
            f"method={method} status={summary.get('status', 'unknown')} "
            f"tok_s={summary.get('tok_per_s_excluding_init', 'n/a')}",
            flush=True,
        )


@app.function(
    volumes={"/data": data_volume},
    timeout=3600,
    startup_timeout=900,
    cpu=4,
)
def probe_vllm_package_job(
    version: str = "phase3_vllm_package_probe_v1",
    vllm_package: str = "vllm",
) -> dict:
    import json
    import os
    import pathlib
    import subprocess
    import sys
    import traceback

    os.chdir("/root/asts-spec")
    out_dir = pathlib.Path(f"/data/vllm_results/{version}/package_probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {
        "version": version,
        "vllm_package": vllm_package,
        "out_dir": str(out_dir),
        "status": "failed",
        "install_command": [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            vllm_package,
        ],
    }

    try:
        print(f"$ {' '.join(result['install_command'])}", flush=True)
        with (out_dir / "raw_stdout.txt").open("w", encoding="utf-8") as stdout, (
            out_dir / "raw_stderr.txt"
        ).open("w", encoding="utf-8") as stderr:
            proc = subprocess.run(
                result["install_command"],
                text=True,
                stdout=stdout,
                stderr=stderr,
            )
        result["install_returncode"] = int(proc.returncode)
        if proc.returncode != 0:
            result["failure_type"] = "install_failed"
            return result

        probe_code = r"""
import inspect
import json
import pathlib

payload = {}
try:
    import vllm
    payload["vllm_version"] = getattr(vllm, "__version__", None)
    payload["vllm_file"] = getattr(vllm, "__file__", None)
    from vllm.config import SpeculativeConfig
    signature_text = str(inspect.signature(SpeculativeConfig))
    payload["SpeculativeConfig_signature"] = signature_text
    payload["SpeculativeConfig_signature_has_custom_class"] = "custom_class" in signature_text
    fields = getattr(SpeculativeConfig, "model_fields", None) or getattr(SpeculativeConfig, "__dataclass_fields__", None)
    payload["SpeculativeConfig_fields"] = sorted(list(fields.keys())) if hasattr(fields, "keys") else str(fields)
    try:
        from vllm.config import SpeculativeMethod
        payload["SpeculativeMethod"] = str(SpeculativeMethod)
        try:
            payload["SpeculativeMethod_members"] = [str(x) for x in list(SpeculativeMethod)]
        except Exception as exc:
            payload["SpeculativeMethod_members_error"] = repr(exc)
    except Exception as exc:
        payload["SpeculativeMethod_import_error"] = repr(exc)
    root = pathlib.Path(vllm.__file__).resolve().parent
    payload["source_has_custom_class"] = False
    payload["source_has_speculative_custom_class"] = False
    hits = []
    speculative_hits = []
    for path in root.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "custom_class" in text:
            payload["source_has_custom_class"] = True
            if len(hits) < 20:
                hits.append(str(path.relative_to(root)))
        rel = str(path.relative_to(root))
        if "custom_class" in text and (
            "spec" in rel.lower()
            or "gpu_model_runner" in rel
            or "config/speculative" in rel
        ):
            payload["source_has_speculative_custom_class"] = True
            if len(speculative_hits) < 20:
                speculative_hits.append(rel)
    payload["custom_class_source_hits"] = hits
    payload["custom_class_speculative_source_hits"] = speculative_hits
    payload["status"] = "success"
except Exception as exc:
    payload["status"] = "failed"
    payload["failure_type"] = type(exc).__name__
    payload["failure_message"] = str(exc)
print(json.dumps(payload, indent=2, sort_keys=True))
"""
        probe = subprocess.run(
            [sys.executable, "-c", probe_code],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        result["probe_returncode"] = int(probe.returncode)
        (out_dir / "probe_stdout.json").write_text(probe.stdout, encoding="utf-8")
        (out_dir / "probe_stderr.txt").write_text(probe.stderr, encoding="utf-8")
        try:
            payload = json.loads(probe.stdout)
            result.update(payload)
        except Exception:
            result["failure_type"] = "probe_parse_failed"
            result["failure_message"] = probe.stdout[-1000:]
        result["status"] = "success" if result.get("status") == "success" else "failed"
    except BaseException as exc:
        result["failure_type"] = type(exc).__name__
        result["failure_message"] = f"{exc}\n{traceback.format_exc()}"
    finally:
        (out_dir / "run_summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        files = {}
        for name in (
            "run_summary.json",
            "probe_stdout.json",
            "probe_stderr.txt",
            "raw_stdout.txt",
            "raw_stderr.txt",
        ):
            path = out_dir / name
            if path.exists():
                files[name] = path.read_text(encoding="utf-8")
        result["files"] = files
        data_volume.commit()
    return result


@app.local_entrypoint()
def launch_vllm_package_probe(
    version: str = "phase3_vllm_package_probe_v1",
    vllm_package: str = "vllm",
    wait: bool = True,
) -> None:
    call = probe_vllm_package_job.spawn(version=version, vllm_package=vllm_package)
    print(f"vllm_package_probe\t{version}\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        local_dir = _PROJECT_ROOT / "artifacts" / "vllm_results" / version / "package_probe"
        local_dir.mkdir(parents=True, exist_ok=True)
        for name, text in (result.get("files") or {}).items():
            (local_dir / name).write_text(text, encoding="utf-8")
        if "run_summary.json" not in (result.get("files") or {}):
            (local_dir / "run_summary.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(
            "DONE vllm_package_probe: "
            f"status={result.get('status')} "
            f"vllm={result.get('vllm_version', 'n/a')} "
            f"custom_class={result.get('source_has_custom_class', 'n/a')}",
            flush=True,
        )


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=7200,
    startup_timeout=1800,
    cpu=4,
)
def run_real_shape_forward_job(
    version: str = "v1",
    source_run_tag: str = "vantage_real_commit_pld_adjacent_test500_rerank_fixed_rank3_test500_v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn: str = "sdpa",
    max_steps: int = 500,
) -> dict:
    import json
    import os
    import subprocess
    import sys

    os.chdir("/root/asts-spec")
    _install_forward_bench_deps()
    eval_dir = f"/data/{source_run_tag}/eval"
    out_dir = f"/data/pld_forward_real_shape/{version}"
    cmd = [
        sys.executable,
        "scripts/benchmark_real_shape_forward.py",
        "--steps",
        f"{eval_dir}/steps.jsonl",
        "--completions",
        f"{eval_dir}/completions.jsonl",
        "--target",
        target,
        "--dtype",
        dtype,
        "--attn",
        attn,
        "--device",
        "cuda",
        "--max-steps",
        str(max_steps),
        "--output-dir",
        out_dir,
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    with open(f"{out_dir}/report.json", "r", encoding="utf-8") as f:
        report = json.load(f)
    with open(f"{out_dir}/report.md", "r", encoding="utf-8") as f:
        report["report_markdown"] = f.read()
    return report


@app.local_entrypoint()
def launch_real_shape_forward_benchmark(
    wait: bool = True,
    version: str = "v1",
    source_run_tag: str = "vantage_real_commit_pld_adjacent_test500_rerank_fixed_rank3_test500_v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn: str = "sdpa",
    max_steps: int = 500,
) -> None:
    call = run_real_shape_forward_job.spawn(
        version=version,
        source_run_tag=source_run_tag,
        target=target,
        dtype=dtype,
        attn=attn,
        max_steps=max_steps,
    )
    print(f"real_shape_forward\t{call.object_id}", flush=True)
    if wait:
        report = call.get()
        agg = report.get("aggregate", {})
        print(
            "DONE real_shape_forward: "
            f"mean={agg.get('forward_ms', {}).get('mean', 0.0):.3f}ms "
            f"p90={agg.get('forward_ms', {}).get('p90', 0.0):.3f}ms "
            f"cached={agg.get('cached_step_forward_ms', {}).get('mean', 0.0):.3f}ms "
            f"ratio={agg.get('real_vs_synthetic_mean_ratio', 0.0):.2f}x",
            flush=True,
        )


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=7200,
    startup_timeout=1800,
    cpu=4,
)
def run_batched_pld_verifier_microbench_job(
    version: str = "v1",
    source_run_tag: str = "vantage_real_commit_pld_adjacent_test500_rerank_fixed_rank3_test500_v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn: str = "sdpa",
    batch_sizes: str = "1,2,4,8,16",
    bucket_sizes: str = "1,2,4,8,16,32,64,128",
    max_examples: int = 0,
    iters_per_combo: int = 12,
    warmup_batches: int = 1,
) -> dict:
    import json
    import os
    import subprocess
    import sys

    os.chdir("/root/asts-spec")
    _install_forward_bench_deps()
    eval_dir = f"/data/{source_run_tag}/eval"
    out_dir = f"/data/batched_pld_verifier_microbench/{version}"
    cmd = [
        sys.executable,
        "scripts/benchmark_batched_pld_verifier.py",
        "--steps",
        f"{eval_dir}/steps.jsonl",
        "--completions",
        f"{eval_dir}/completions.jsonl",
        "--target",
        target,
        "--dtype",
        dtype,
        "--attn",
        attn,
        "--device",
        "cuda",
        "--batch-sizes",
        batch_sizes,
        "--bucket-sizes",
        bucket_sizes,
        "--max-examples",
        str(max_examples),
        "--iters-per-combo",
        str(iters_per_combo),
        "--warmup-batches",
        str(warmup_batches),
        "--output-dir",
        out_dir,
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    with open(f"{out_dir}/report.json", "r", encoding="utf-8") as f:
        report = json.load(f)
    with open(f"{out_dir}/report.md", "r", encoding="utf-8") as f:
        report["report_markdown"] = f.read()
    return report


@app.local_entrypoint()
def launch_batched_pld_verifier_microbench(
    split: str = "test",
    n: int = 500,
    wait: bool = True,
    version: str = "v1",
    source_run_tag: str = "",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn: str = "sdpa",
    batch_sizes: str = "1,2,4,8,16",
    bucket_sizes: str = "1,2,4,8,16,32,64,128",
    max_examples: int = 0,
    iters_per_combo: int = 12,
    warmup_batches: int = 1,
) -> None:
    if not source_run_tag:
        # Existing held-out PLD trace used throughout the PLD-adjacent studies.
        source_run_tag = "vantage_real_commit_pld_adjacent_test500_rerank_fixed_rank3_test500_v1"
    call = run_batched_pld_verifier_microbench_job.spawn(
        version=version,
        source_run_tag=source_run_tag,
        target=target,
        dtype=dtype,
        attn=attn,
        batch_sizes=batch_sizes,
        bucket_sizes=bucket_sizes,
        max_examples=max_examples,
        iters_per_combo=iters_per_combo,
        warmup_batches=warmup_batches,
    )
    print(f"batched_pld_verifier_microbench\t{call.object_id}", flush=True)
    if wait:
        report = call.get()
        local_dir = _PROJECT_ROOT / "analysis" / "batched_pld_verifier_microbench" / version
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        decision = report.get("decision", {})
        speedups = decision.get("weighted_speedup_vs_batch1_by_batch", {})
        parts = [f"b{k}={v:.3f}x" for k, v in sorted(speedups.items(), key=lambda kv: int(kv[0]))]
        print(
            "DONE batched_pld_verifier_microbench: "
            + " ".join(parts)
            + f" decision={decision.get('decision')}",
            flush=True,
        )


def _residual_phase5_install_cmds() -> list[list[str]]:
    import sys

    return [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
            "torch==2.5.1",
        ],
        [
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
            "scikit-learn>=1.4",
            "pydivsufsort>=0.0.18",
            "editdistance>=0.8.1",
        ],
    ]


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=14400,
    startup_timeout=3600,
    cpu=4,
    memory=65536,
)
def run_collect_queued_residual_data_job(
    *,
    split: str = "train",
    n: int = 20,
    version: str = "vantage_residual_phase5_collect_train20_smoke_v1",
    k: int = 4,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    eagle_checkpoint: str = "/data/eagle_v1_normfix/eagle/eagle_final.pt",
    method: str = "blazedit_pld_w128_n10",
    dtype: str = "bf16",
    attn: str = "sdpa",
    max_new_tokens: int = 256,
    collect_batch_size: int = 1,
    trigger_threshold: int = 4,
    weak_field: str = "draft_len",
    problem_jsonl: str = "",
) -> dict:
    """Run PLD then collect true queued-use hidden-state tensors.

    This job intentionally writes one raw queued tensor to the Modal volume. The
    local machine must pull it with ``modal volume get`` before training.
    """

    import json
    import os
    import subprocess
    import sys
    import time

    os.chdir("/root/asts-spec")
    for install_cmd in _residual_phase5_install_cmds():
        print(f"$ {' '.join(install_cmd)}", flush=True)
        subprocess.run(install_cmd, check=True)

    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    if not problem_jsonl:
        problem_jsonl = f"/root/asts-spec/data/real_commits/path_a_{split}500_v1.jsonl"
    eval_dtype = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(dtype, dtype)
    root = f"/data/vantage_residual/phase5_data/raw/{version}"
    trace_dir = f"{root}/pld_trace/eval"
    raw_pt = f"{root}/queued_raw.pt"
    projection_pt = f"{root}/qwen_output_projection.pt"
    os.makedirs(root, exist_ok=True)
    config = {
        "split": split,
        "n": n,
        "version": version,
        "k": k,
        "target": target,
        "eagle_checkpoint": eagle_checkpoint,
        "method": method,
        "dtype": dtype,
        "attn": attn,
        "max_new_tokens": max_new_tokens,
        "collect_batch_size": collect_batch_size,
        "trigger_threshold": trigger_threshold,
        "weak_field": weak_field,
        "problem_jsonl": problem_jsonl,
        "root": root,
        "trace_dir": trace_dir,
        "raw_pt": raw_pt,
    }
    with open(f"{root}/config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    def run_logged(cmd: list[str], name: str) -> None:
        print(f"$ {' '.join(cmd)}", flush=True)
        started = time.time()
        proc = subprocess.run(cmd, text=True, capture_output=True)
        with open(f"{root}/{name}_stdout.txt", "w", encoding="utf-8") as f:
            f.write(proc.stdout)
        with open(f"{root}/{name}_stderr.txt", "w", encoding="utf-8") as f:
            f.write(proc.stderr)
        print(proc.stdout, end="", flush=True)
        print(proc.stderr, end="", file=sys.stderr, flush=True)
        if proc.returncode != 0:
            with open(f"{root}/failure.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "failed_stage": name,
                        "command": cmd,
                        "returncode": proc.returncode,
                        "elapsed_seconds": time.time() - started,
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )
            data_volume.commit()
            hf_cache.commit()
            raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)

    eval_cmd = [
        sys.executable,
        "scripts/run_eagle_eval.py",
        "--output-dir",
        trace_dir,
        "--target",
        target,
        "--eagle-checkpoint",
        eagle_checkpoint,
        "--n",
        str(n),
        "--max-new-tokens",
        str(max_new_tokens),
        "--methods",
        method,
        "--dtype",
        eval_dtype,
        "--attn-impl",
        attn,
        "--problem-jsonl",
        problem_jsonl,
        "--skip-eagle-load",
        "--code-proposer-fallback",
        "root",
        "--chat-template",
        "none",
        "--log-level",
        "INFO",
    ]
    run_logged(eval_cmd, "pld_decode")

    collect_cmd = [
        sys.executable,
        "scripts/collect_queued_mtp_training_data.py",
        "--target",
        target,
        "--steps",
        f"{trace_dir}/steps.jsonl",
        "--completions",
        f"{trace_dir}/completions.jsonl",
        "--output",
        raw_pt,
        "--method",
        method,
        "--num-heads",
        str(k),
        "--trigger-threshold",
        str(trigger_threshold),
        "--weak-field",
        weak_field,
        "--include-dropped",
        "false",
        "--dtype",
        dtype,
        "--device",
        "cuda",
        "--batch-size",
        str(collect_batch_size),
        "--output-projection",
        projection_pt,
    ]
    run_logged(collect_cmd, "queued_collect")

    import torch

    payload = torch.load(raw_pt, map_location="cpu")
    hidden = payload.get("hidden")
    labels = payload.get("labels")
    valid = payload.get("valid_queued_example")
    nonzero_hidden = bool(hidden is not None and float(hidden.abs().sum().item()) > 0.0)
    valid_count = int(valid.bool().sum().item()) if hasattr(valid, "bool") else 0
    summary = {
        "status": "ok",
        "raw_pt": raw_pt,
        "summary_json": f"{raw_pt}.summary.json",
        "hidden_shape": list(hidden.shape) if hasattr(hidden, "shape") else None,
        "labels_shape": list(labels.shape) if hasattr(labels, "shape") else None,
        "nonzero_hidden": nonzero_hidden,
        "valid_queued_examples": valid_count,
        "metadata": payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
        "config": config,
    }
    if not nonzero_hidden or valid_count <= 0:
        summary["status"] = "failed_validation"
    with open(f"{root}/run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    data_volume.commit()
    hf_cache.commit()
    if summary["status"] != "ok":
        raise RuntimeError(f"queued collection failed validation: {summary}")
    return summary


@app.local_entrypoint()
def launch_collect_queued_residual_data(
    split: str = "train",
    n: int = 20,
    version: str = "vantage_residual_phase5_collect_train20_smoke_v1",
    k: int = 4,
    dtype: str = "bf16",
    attn: str = "sdpa",
    wait: bool = True,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    eagle_checkpoint: str = "/data/eagle_v1_normfix/eagle/eagle_final.pt",
    max_new_tokens: int = 256,
    collect_batch_size: int = 1,
    trigger_threshold: int = 4,
    weak_field: str = "draft_len",
    problem_jsonl: str = "",
) -> None:
    call = run_collect_queued_residual_data_job.spawn(
        split=split,
        n=n,
        version=version,
        k=k,
        target=target,
        eagle_checkpoint=eagle_checkpoint,
        dtype=dtype,
        attn=attn,
        max_new_tokens=max_new_tokens,
        collect_batch_size=collect_batch_size,
        trigger_threshold=trigger_threshold,
        weak_field=weak_field,
        problem_jsonl=problem_jsonl,
    )
    print(f"collect_queued_residual_data\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        print(
            "DONE collect_queued_residual_data: "
            f"raw_pt={result.get('raw_pt')} "
            f"hidden={result.get('hidden_shape')} "
            f"valid={result.get('valid_queued_examples')}",
            flush=True,
        )


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=7200,
    startup_timeout=3600,
    cpu=4,
    memory=65536,
)
def run_residual_hidden_capture_profile_job(
    *,
    split: str = "train",
    n: int = 20,
    version: str = "vantage_residual_phase5_hidden_capture_v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn: str = "sdpa",
    max_length: int = 2048,
) -> dict:
    import json
    import os
    import time

    os.chdir("/root/asts-spec")
    import subprocess

    for install_cmd in _residual_phase5_install_cmds():
        print(f"$ {' '.join(install_cmd)}", flush=True)
        subprocess.run(install_cmd, check=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    problem_jsonl = f"/root/asts-spec/data/real_commits/path_a_{split}500_v1.jsonl"
    root = f"/data/vantage_residual/phase5_hidden_capture/{version}"
    os.makedirs(root, exist_ok=True)
    dtype_obj = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(target, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        target,
        torch_dtype=dtype_obj,
        trust_remote_code=True,
        attn_implementation=attn,
    ).to("cuda")
    model.eval()
    prompts: list[str] = []
    with open(problem_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if len(prompts) >= n:
                break
            row = json.loads(line)
            prompts.append(str(row.get("prompt") or row.get("input") or row.get("source") or ""))
    if not prompts:
        raise RuntimeError(f"no prompts loaded from {problem_jsonl}")
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to("cuda")

    def timed_forward(*, capture: bool) -> tuple[float, int, list[int]]:
        captured: list[torch.Tensor] = []
        handle = None
        if capture:
            module = getattr(getattr(model, "model", None), "norm", None)
            if module is None:
                raise RuntimeError("could not locate final norm module for hidden capture hook")

            def hook(_module, _inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                captured.append(tensor[:, -1, :].detach())

            handle = module.register_forward_hook(hook)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            out = model(**enc, use_cache=False)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if handle is not None:
            handle.remove()
        peak = int(torch.cuda.max_memory_allocated())
        argmax = out.logits[:, -1, :].argmax(dim=-1).detach().cpu().tolist()
        if capture and not captured:
            raise RuntimeError("hidden capture hook did not fire")
        return elapsed_ms, peak, [int(x) for x in argmax]

    # Warmup.
    timed_forward(capture=False)
    baseline_ms, baseline_peak, baseline_argmax = timed_forward(capture=False)
    capture_ms, capture_peak, capture_argmax = timed_forward(capture=True)
    overhead_pct = 100.0 * (capture_ms - baseline_ms) / max(1e-9, baseline_ms)
    summary = {
        "split": split,
        "n": n,
        "version": version,
        "target": target,
        "dtype": dtype,
        "attn": attn,
        "baseline_ms": baseline_ms,
        "capture_ms": capture_ms,
        "direct_hidden_capture_overhead_pct": overhead_pct,
        "baseline_peak_memory_gb": baseline_peak / (1024**3),
        "capture_peak_memory_gb": capture_peak / (1024**3),
        "peak_memory_delta_gb": (capture_peak - baseline_peak) / (1024**3),
        "output_argmax_parity": baseline_argmax == capture_argmax,
        "capture_method": "final_norm_forward_hook",
        "all_hidden_states_materialized": False,
        "gate_pass": overhead_pct <= 8.0 and baseline_argmax == capture_argmax,
    }
    with open(f"{root}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    data_volume.commit()
    hf_cache.commit()
    return summary


@app.local_entrypoint()
def launch_residual_hidden_capture_profile(
    split: str = "train",
    n: int = 20,
    version: str = "vantage_residual_phase5_hidden_capture_v1",
    dtype: str = "bf16",
    attn: str = "sdpa",
    wait: bool = True,
    target: str = "Qwen/Qwen2.5-Coder-7B",
) -> None:
    call = run_residual_hidden_capture_profile_job.spawn(
        split=split,
        n=n,
        version=version,
        target=target,
        dtype=dtype,
        attn=attn,
    )
    print(f"residual_hidden_capture_profile\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        print(
            "DONE residual_hidden_capture_profile: "
            f"overhead={result.get('direct_hidden_capture_overhead_pct', 0.0):.2f}% "
            f"parity={result.get('output_argmax_parity')} "
            f"gate={result.get('gate_pass')}",
            flush=True,
        )


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=7200,
    startup_timeout=3600,
    cpu=4,
    memory=65536,
)
def run_train_queued_residual_heads_job(
    *,
    version: str = "queued_linear_k1_v1",
    train: str = "/data/vantage_residual/phase6_data/queued_v1/train.pt",
    val: str = "/data/vantage_residual/phase6_data/queued_v1/val.pt",
    output_dir: str = "/data/vantage_residual/phase7_checkpoints/queued_linear_k1_v1",
    output_projection: str = "/data/vantage_residual/phase6_data/raw/vantage_residual_phase6_collect_train500_densityfix_v1/qwen_output_projection.pt",
    head_type: str = "linear",
    k: int = 1,
    epochs: int = 30,
    seed: int = 0,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 2048,
    target: str = "Qwen/Qwen2.5-Coder-7B",
) -> dict:
    import json
    import os
    import subprocess
    import sys
    import time

    os.chdir("/root/asts-spec")
    for install_cmd in _residual_phase5_install_cmds():
        print(f"$ {' '.join(install_cmd)}", flush=True)
        subprocess.run(install_cmd, check=True)

    os.makedirs(output_dir, exist_ok=True)
    config = {
        "version": version,
        "train": train,
        "val": val,
        "output_dir": output_dir,
        "output_projection": output_projection,
        "head_type": head_type,
        "k": k,
        "epochs": epochs,
        "seed": seed,
        "batch_size": batch_size,
        "lr": lr,
        "hidden_dim": hidden_dim,
        "target": target,
        "device": "cuda",
    }
    with open(f"{output_dir}/modal_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    cmd = [
        sys.executable,
        "scripts/train_vantage_queued_residual_heads.py",
        "--train",
        train,
        "--val",
        val,
        "--output-dir",
        output_dir,
        "--head-type",
        head_type,
        "--k",
        str(k),
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--device",
        "cuda",
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--hidden-dim",
        str(hidden_dim),
        "--target",
        target,
        "--output-projection",
        output_projection,
        "--table",
        f"{output_dir}/training_gate.md",
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    started = time.time()
    proc = subprocess.run(cmd, text=True, capture_output=True)
    with open(f"{output_dir}/raw_stdout.txt", "w", encoding="utf-8") as f:
        f.write(proc.stdout)
    with open(f"{output_dir}/raw_stderr.txt", "w", encoding="utf-8") as f:
        f.write(proc.stderr)
    print(proc.stdout, end="", flush=True)
    print(proc.stderr, end="", file=sys.stderr, flush=True)
    summary = {
        "version": version,
        "command": cmd,
        "returncode": proc.returncode,
        "elapsed_seconds": time.time() - started,
        "output_dir": output_dir,
        "checkpoint": f"{output_dir}/model.pt",
        "metrics": f"{output_dir}/queued_training_metrics.json",
        "config": config,
    }
    if proc.returncode != 0:
        summary["status"] = "failed"
        with open(f"{output_dir}/failure.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        data_volume.commit()
        hf_cache.commit()
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)
    metrics_path = f"{output_dir}/queued_training_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        summary["training_gate"] = metrics.get("training_gate")
        validation = metrics.get("validation") if isinstance(metrics.get("validation"), dict) else {}
        summary["h1_top1"] = validation.get("h1_top1")
    summary["status"] = "ok"
    with open(f"{output_dir}/run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    data_volume.commit()
    hf_cache.commit()
    return summary


@app.local_entrypoint()
def launch_train_queued_residual_heads(
    version: str = "queued_linear_k1_v1",
    train: str = "/data/vantage_residual/phase6_data/queued_v1/train.pt",
    val: str = "/data/vantage_residual/phase6_data/queued_v1/val.pt",
    output_dir: str = "",
    output_projection: str = "/data/vantage_residual/phase6_data/raw/vantage_residual_phase6_collect_train500_densityfix_v1/qwen_output_projection.pt",
    head_type: str = "linear",
    k: int = 1,
    epochs: int = 30,
    seed: int = 0,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 2048,
    wait: bool = True,
) -> None:
    if not output_dir:
        output_dir = f"/data/vantage_residual/phase7_checkpoints/{version}"
    call = run_train_queued_residual_heads_job.spawn(
        version=version,
        train=train,
        val=val,
        output_dir=output_dir,
        output_projection=output_projection,
        head_type=head_type,
        k=k,
        epochs=epochs,
        seed=seed,
        batch_size=batch_size,
        lr=lr,
        hidden_dim=hidden_dim,
    )
    print(f"train_queued_residual_heads\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        print(
            "DONE train_queued_residual_heads: "
            f"version={result.get('version')} "
            f"gate={result.get('training_gate')} "
            f"h1={result.get('h1_top1')}",
            flush=True,
        )


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=7200,
    startup_timeout=3600,
    cpu=4,
    memory=65536,
)
def run_evaluate_queued_residual_offline_job(
    *,
    version: str = "queued_linear_k1_v1",
    checkpoint: str = "/data/vantage_residual/phase7_checkpoints/queued_linear_k1_v1/model.pt",
    data: str = "/data/vantage_residual/phase6_data/queued_v1/test.pt",
    steps: str = "/data/vantage_residual/phase5_data/raw/vantage_residual_phase6_collect_test500_densityfix_v1/pld_trace/eval/steps.jsonl",
    hidden_capture: str = "/data/vantage_residual/phase5_hidden_capture/vantage_residual_phase5_hidden_capture_v1/summary.json",
    output_dir: str = "",
    batch_size: int = 256,
    confidence_thresholds: str = "0.0,0.1,0.2,0.3,0.5,0.7,0.9",
    pld_draft_len_thresholds: str = "0,1,2,4,8",
    previous_accepted_len_thresholds: str = "0,1,2,4",
) -> dict:
    import json
    import os
    import subprocess
    import sys
    import time

    os.chdir("/root/asts-spec")
    for install_cmd in _residual_phase5_install_cmds():
        print(f"$ {' '.join(install_cmd)}", flush=True)
        subprocess.run(install_cmd, check=True)
    if not output_dir:
        output_dir = f"/data/vantage_residual/phase7_offline/{version}"
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/evaluate_vantage_residual_queued_offline.py",
        "--checkpoint",
        checkpoint,
        "--data",
        data,
        "--steps",
        steps,
        "--hidden-capture-overhead-json",
        hidden_capture,
        "--output-dir",
        output_dir,
        "--batch-size",
        str(batch_size),
        "--device",
        "cuda",
        "--confidence-thresholds",
        confidence_thresholds,
        "--pld-draft-len-thresholds",
        pld_draft_len_thresholds,
        "--previous-accepted-len-thresholds",
        previous_accepted_len_thresholds,
    ]
    config = {
        "version": version,
        "checkpoint": checkpoint,
        "data": data,
        "steps": steps,
        "hidden_capture": hidden_capture,
        "output_dir": output_dir,
        "batch_size": batch_size,
        "confidence_thresholds": confidence_thresholds,
        "pld_draft_len_thresholds": pld_draft_len_thresholds,
        "previous_accepted_len_thresholds": previous_accepted_len_thresholds,
        "command": cmd,
    }
    with open(f"{output_dir}/modal_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    print(f"$ {' '.join(cmd)}", flush=True)
    started = time.time()
    proc = subprocess.run(cmd, text=True, capture_output=True)
    with open(f"{output_dir}/raw_stdout.txt", "w", encoding="utf-8") as f:
        f.write(proc.stdout)
    with open(f"{output_dir}/raw_stderr.txt", "w", encoding="utf-8") as f:
        f.write(proc.stderr)
    print(proc.stdout, end="", flush=True)
    print(proc.stderr, end="", file=sys.stderr, flush=True)
    summary = {
        "version": version,
        "command": cmd,
        "returncode": proc.returncode,
        "elapsed_seconds": time.time() - started,
        "output_dir": output_dir,
        "report_json": f"{output_dir}/report.json",
        "report_md": f"{output_dir}/report.md",
    }
    if proc.returncode != 0:
        summary["status"] = "failed"
        with open(f"{output_dir}/failure.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        data_volume.commit()
        hf_cache.commit()
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)
    if os.path.exists(f"{output_dir}/report.json"):
        with open(f"{output_dir}/report.json", "r", encoding="utf-8") as f:
            report = json.load(f)
        summary["decision"] = report.get("decision")
        rows = report.get("phase4_rows") if isinstance(report.get("phase4_rows"), list) else []
        if rows:
            best = rows[0]
            summary["best_projected_speedup_after_hidden_overhead"] = best.get(
                "projected_speedup_after_hidden_overhead"
            )
            summary["best_token0_reject_rate"] = best.get("token0_reject_rate")
            summary["best_accepted_per_use"] = best.get("accepted_per_use")
            summary["best_queue_used"] = best.get("queue_used")
    summary["status"] = "ok"
    with open(f"{output_dir}/run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    data_volume.commit()
    hf_cache.commit()
    return summary


@app.local_entrypoint()
def launch_evaluate_queued_residual_offline(
    version: str = "queued_linear_k1_v1",
    checkpoint: str = "",
    data: str = "/data/vantage_residual/phase6_data/queued_v1/test.pt",
    steps: str = "/data/vantage_residual/phase5_data/raw/vantage_residual_phase6_collect_test500_densityfix_v1/pld_trace/eval/steps.jsonl",
    hidden_capture: str = "/data/vantage_residual/phase5_hidden_capture/vantage_residual_phase5_hidden_capture_v1/summary.json",
    output_dir: str = "",
    batch_size: int = 256,
    wait: bool = True,
) -> None:
    if not checkpoint:
        checkpoint = f"/data/vantage_residual/phase7_checkpoints/{version}/model.pt"
    if not output_dir:
        output_dir = f"/data/vantage_residual/phase7_offline/{version}"
    call = run_evaluate_queued_residual_offline_job.spawn(
        version=version,
        checkpoint=checkpoint,
        data=data,
        steps=steps,
        hidden_capture=hidden_capture,
        output_dir=output_dir,
        batch_size=batch_size,
    )
    print(f"evaluate_queued_residual_offline\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        print(
            "DONE evaluate_queued_residual_offline: "
            f"version={result.get('version')} "
            f"decision={result.get('decision')} "
            f"best_after_hidden={result.get('best_projected_speedup_after_hidden_overhead')}",
            flush=True,
        )


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=14400,
    startup_timeout=1800,
    cpu=4,
)
def run_batched_pld_eval_job(
    version: str = "v1",
    split: str = "test",
    n: int = 50,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn: str = "sdpa",
    batch_sizes: str = "2,4,8",
    bucket_sizes: str = "1,2,4,8,16,32,64,128",
    bucket_policy: str = "custom",
    refill_policy: str = "continuous",
    active_pool_size: int = 0,
    problem_jsonl: str = "",
    write_audit_trace: bool = False,
) -> dict:
    import json
    import os
    import subprocess
    import sys

    os.chdir("/root/asts-spec")
    _install_forward_bench_deps()
    if not problem_jsonl:
        problem_jsonl = (
            f"/root/asts-spec/data/real_commits/real_commit_manifest_balanced_1000_v2_{split}500.jsonl"
        )
    out_dir = f"/data/batched_pld_eval/{version}"
    cmd = [
        sys.executable,
        "scripts/run_batched_pld_eval.py",
        "--problem-jsonl",
        problem_jsonl,
        "--n",
        str(n),
        "--target",
        target,
        "--dtype",
        dtype,
        "--attn",
        attn,
        "--device",
        "cuda",
        "--max-new-tokens",
        "256",
        "--batch-sizes",
        batch_sizes,
        "--bucket-sizes",
        bucket_sizes,
        "--bucket-policy",
        bucket_policy,
        "--refill-policy",
        refill_policy,
        "--active-pool-size",
        str(active_pool_size),
        "--output-dir",
        out_dir,
    ]
    if write_audit_trace:
        cmd += ["--audit-trace", f"{out_dir}/batch_audit_trace.jsonl"]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    audit_report = {}
    if write_audit_trace:
        audit_dir = f"/data/batched_pld_task_audit/{version}/batch_trace"
        audit_cmd = [
            sys.executable,
            "scripts/audit_batched_pld_task_isolation.py",
            "--trace",
            f"{out_dir}/batch_audit_trace.jsonl",
            "--output-dir",
            audit_dir,
        ]
        print(f"$ {' '.join(audit_cmd)}", flush=True)
        subprocess.run(audit_cmd, check=True)
        with open(f"{audit_dir}/report.json", "r", encoding="utf-8") as f:
            audit_report = json.load(f)
    data_volume.commit()
    with open(f"{out_dir}/report.json", "r", encoding="utf-8") as f:
        report = json.load(f)
    if audit_report:
        report["audit_report"] = audit_report
    return report


@app.local_entrypoint()
def launch_batched_pld_eval(
    split: str = "test",
    n: int = 50,
    wait: bool = True,
    version: str = "v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn: str = "sdpa",
    batch_sizes: str = "2,4,8",
    bucket_sizes: str = "1,2,4,8,16,32,64,128",
    bucket_policy: str = "custom",
    refill_policy: str = "continuous",
    active_pool_size: int = 0,
    problem_jsonl: str = "",
    write_audit_trace: bool = False,
) -> None:
    call = run_batched_pld_eval_job.spawn(
        version=version,
        split=split,
        n=n,
        target=target,
        dtype=dtype,
        attn=attn,
        batch_sizes=batch_sizes,
        bucket_sizes=bucket_sizes,
        bucket_policy=bucket_policy,
        refill_policy=refill_policy,
        active_pool_size=active_pool_size,
        problem_jsonl=problem_jsonl,
        write_audit_trace=write_audit_trace,
    )
    print(f"batched_pld_eval\t{call.object_id}", flush=True)
    if wait:
        report = call.get()
        local_dir = _PROJECT_ROOT / "analysis" / "batched_pld_eval" / version
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if report.get("audit_report"):
            audit_dir = _PROJECT_ROOT / "analysis" / "batched_pld_task_audit" / version
            audit_dir.mkdir(parents=True, exist_ok=True)
            (audit_dir / "report.json").write_text(
                json.dumps(report["audit_report"], indent=2, sort_keys=True) + "\n"
            )
        seq = report.get("sequential", {})
        seq_tps = seq.get("tokens_per_sec", 0.0) or 1.0
        parts = []
        for row in report.get("batched", []):
            tps = row.get("generated_tokens_per_sec", 0.0)
            parts.append(f"b{row.get('batch_size')}={tps:.1f}t/s({tps / seq_tps:.3f}x)")
        print("DONE batched_pld_eval: " + " ".join(parts), flush=True)


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=28800,
    startup_timeout=1800,
    cpu=4,
)
def run_batched_pld_repeated_timing_job(
    version: str = "continuous_batched_pld_final_repeats_v1",
    split: str = "test",
    n: int = 500,
    repeats: int = 3,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn: str = "sdpa",
    batch_sizes: str = "2,4,8",
    active_pool_size: int = 32,
    bucket_policy: str = "default",
    refill_policy: str = "continuous",
    problem_jsonl: str = "",
    write_audit_trace: bool = True,
) -> dict:
    import json
    import os
    import subprocess
    import sys

    os.chdir("/root/asts-spec")
    _install_forward_bench_deps()
    if not problem_jsonl:
        problem_jsonl = (
            f"/root/asts-spec/data/real_commits/real_commit_manifest_balanced_1000_v2_{split}500.jsonl"
        )
    out_dir = f"/data/continuous_batched_pld_final_repeats/{version}"
    cmd = [
        sys.executable,
        "scripts/run_batched_pld_repeated_timing.py",
        "--problem-jsonl",
        problem_jsonl,
        "--n",
        str(n),
        "--repeats",
        str(repeats),
        "--target",
        target,
        "--dtype",
        dtype,
        "--attn",
        attn,
        "--device",
        "cuda",
        "--max-new-tokens",
        "256",
        "--batch-sizes",
        batch_sizes,
        "--active-pool-size",
        str(active_pool_size),
        "--bucket-policy",
        bucket_policy,
        "--refill-policy",
        refill_policy,
        "--output-dir",
        out_dir,
    ]
    if write_audit_trace:
        cmd.append("--write-audit-trace")
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    audit_report = {}
    if write_audit_trace:
        audit_trace = f"{out_dir}/repeat0_batch8_audit_trace.jsonl"
        audit_dir = f"/data/continuous_batched_pld_task_audit_test500/{version}"
        audit_cmd = [
            sys.executable,
            "scripts/audit_batched_pld_task_isolation.py",
            "--trace",
            audit_trace,
            "--output-dir",
            audit_dir,
        ]
        print(f"$ {' '.join(audit_cmd)}", flush=True)
        subprocess.run(audit_cmd, check=True)
        with open(f"{audit_dir}/report.json", "r", encoding="utf-8") as f:
            audit_report = json.load(f)
    data_volume.commit()
    with open(f"{out_dir}/report.json", "r", encoding="utf-8") as f:
        report = json.load(f)
    if audit_report:
        report["task_isolation_audit"] = audit_report
    return report


@app.local_entrypoint()
def launch_batched_pld_repeated_timing(
    split: str = "test",
    n: int = 500,
    repeats: int = 3,
    wait: bool = True,
    version: str = "continuous_batched_pld_final_repeats_v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn: str = "sdpa",
    batch_sizes: str = "2,4,8",
    active_pool_size: int = 32,
    bucket_policy: str = "default",
    refill_policy: str = "continuous",
    problem_jsonl: str = "",
    write_audit_trace: bool = True,
) -> None:
    call = run_batched_pld_repeated_timing_job.spawn(
        version=version,
        split=split,
        n=n,
        repeats=repeats,
        target=target,
        dtype=dtype,
        attn=attn,
        batch_sizes=batch_sizes,
        active_pool_size=active_pool_size,
        bucket_policy=bucket_policy,
        refill_policy=refill_policy,
        problem_jsonl=problem_jsonl,
        write_audit_trace=write_audit_trace,
    )
    print(f"batched_pld_repeated_timing\t{call.object_id}", flush=True)
    if wait:
        report = call.get()
        base_local_dir = _PROJECT_ROOT / "analysis" / "continuous_batched_pld_final_repeats"
        # Keep the locked bf16/SDPA final artifact at the historical top-level
        # path, but write all alternate protocols (for example fp32/eager
        # exact-backend throughput) to a versioned subdirectory so they cannot
        # silently overwrite the headline evidence.
        if version == "continuous_batched_pld_final_repeats_v1":
            local_dir = base_local_dir
        else:
            local_dir = base_local_dir / version
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if report.get("task_isolation_audit"):
            audit_base_dir = _PROJECT_ROOT / "analysis" / "continuous_batched_pld_task_audit_test500"
            audit_dir = audit_base_dir if version == "continuous_batched_pld_final_repeats_v1" else audit_base_dir / version
            audit_dir.mkdir(parents=True, exist_ok=True)
            (audit_dir / "report.json").write_text(
                json.dumps(report["task_isolation_audit"], indent=2, sort_keys=True) + "\n"
            )
        parts = []
        for key, row in sorted(report.get("summary", {}).items()):
            parts.append(
                f"{key}={row.get('tok_s', {}).get('mean', 0.0):.1f}t/s "
                f"{row.get('speedup', {}).get('mean', 0.0):.3f}x"
            )
        print("DONE batched_pld_repeated_timing: " + " | ".join(parts), flush=True)


@app.function(
    image=image,
    gpu="L40S",
    timeout=24 * 60 * 60,
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
)
def run_batched_pld_repeated_timing_sharded_job(
    version: str = "continuous_batched_pld_fp32_eager_throughput_test500_sharded_v1",
    split: str = "test",
    n: int = 500,
    shard_size: int = 50,
    repeats: int = 1,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "fp32",
    attn: str = "eager",
    batch_sizes: str = "2,4,8",
    active_pool_size: int = 32,
    bucket_policy: str = "default",
    refill_policy: str = "continuous",
    problem_jsonl: str = "",
    memory_hygiene: bool = True,
    empty_cache_every: int = 1,
    prefill_chunk_size: int = 1024,
    deterministic: bool = True,
) -> dict:
    import json
    import os
    import subprocess
    import sys

    os.chdir("/root/asts-spec")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _install_forward_bench_deps()
    if not problem_jsonl:
        problem_jsonl = (
            f"/root/asts-spec/data/real_commits/real_commit_manifest_balanced_1000_v2_{split}500.jsonl"
        )
    out_dir = f"/data/continuous_batched_pld_final_repeats/{version}"
    cmd = [
        sys.executable,
        "scripts/run_batched_pld_repeated_timing_sharded.py",
        "--problem-jsonl",
        problem_jsonl,
        "--n",
        str(n),
        "--shard-size",
        str(shard_size),
        "--repeats",
        str(repeats),
        "--target",
        target,
        "--dtype",
        dtype,
        "--attn",
        attn,
        "--device",
        "cuda",
        "--max-new-tokens",
        "256",
        "--batch-sizes",
        batch_sizes,
        "--active-pool-size",
        str(active_pool_size),
        "--bucket-policy",
        bucket_policy,
        "--refill-policy",
        refill_policy,
        "--output-dir",
        out_dir,
        "--prefill-chunk-size",
        str(prefill_chunk_size),
    ]
    if memory_hygiene:
        cmd.extend(["--memory-hygiene", "--empty-cache-every", str(empty_cache_every)])
    if deterministic:
        cmd.append("--deterministic")
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    with open(f"{out_dir}/report.json", "r", encoding="utf-8") as f:
        report = json.load(f)
    report["report_markdown"] = open(f"{out_dir}/report.md", "r", encoding="utf-8").read()
    return report


@app.local_entrypoint()
def launch_batched_pld_repeated_timing_sharded(
    split: str = "test",
    n: int = 500,
    shard_size: int = 50,
    repeats: int = 1,
    wait: bool = True,
    version: str = "continuous_batched_pld_fp32_eager_throughput_test500_sharded_v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "fp32",
    attn: str = "eager",
    batch_sizes: str = "2,4,8",
    active_pool_size: int = 32,
    bucket_policy: str = "default",
    refill_policy: str = "continuous",
    problem_jsonl: str = "",
    memory_hygiene: bool = True,
    empty_cache_every: int = 1,
    prefill_chunk_size: int = 1024,
    deterministic: bool = True,
) -> None:
    call = run_batched_pld_repeated_timing_sharded_job.spawn(
        version=version,
        split=split,
        n=n,
        shard_size=shard_size,
        repeats=repeats,
        target=target,
        dtype=dtype,
        attn=attn,
        batch_sizes=batch_sizes,
        active_pool_size=active_pool_size,
        bucket_policy=bucket_policy,
        refill_policy=refill_policy,
        problem_jsonl=problem_jsonl,
        memory_hygiene=memory_hygiene,
        empty_cache_every=empty_cache_every,
        prefill_chunk_size=prefill_chunk_size,
        deterministic=deterministic,
    )
    print(f"batched_pld_repeated_timing_sharded\t{call.object_id}", flush=True)
    if wait:
        report = call.get()
        local_dir = _PROJECT_ROOT / "analysis" / "continuous_batched_pld_final_repeats" / version
        local_dir.mkdir(parents=True, exist_ok=True)
        report_markdown = report.pop("report_markdown", "")
        (local_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if report_markdown:
            (local_dir / "report.md").write_text(report_markdown)
        parts = []
        for key, row in sorted(report.get("summary", {}).items()):
            parts.append(
                f"{key}={row.get('tok_s', {}).get('mean', 0.0):.1f}t/s "
                f"{row.get('speedup', {}).get('mean', 0.0):.3f}x"
            )
        print("DONE batched_pld_repeated_timing_sharded: " + " | ".join(parts), flush=True)


@app.local_entrypoint()
def launch_continuous_batched_pld_final(
    split: str = "test",
    n: int = 500,
    repeats: int = 3,
    wait: bool = True,
    version: str = "continuous_batched_pld_final_repeats_v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn: str = "sdpa",
    batch_sizes: str = "2,4,8",
    active_pool_size: int = 32,
    bucket_policy: str = "default",
    refill_policy: str = "continuous",
    problem_jsonl: str = "",
) -> None:
    """Frozen final preset for Continuous Batched PLD Verification.

    Reports the sequential PLD baseline plus batch=2/4/8 under the final
    scheduler config:
      batch size: 8 for the headline row
      active pool: 32
      buckets: default (8,16,32,64,128)
      refill: continuous
      draft: w128_n10
    """

    launch_batched_pld_repeated_timing(
        split=split,
        n=n,
        repeats=repeats,
        wait=wait,
        version=version,
        target=target,
        dtype=dtype,
        attn=attn,
        batch_sizes=batch_sizes,
        active_pool_size=active_pool_size,
        bucket_policy=bucket_policy,
        refill_policy=refill_policy,
        problem_jsonl=problem_jsonl,
        write_audit_trace=True,
    )


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=28800,
    startup_timeout=1800,
    cpu=4,
)
def run_batched_greedy_eval_job(
    version: str = "generic_batched_greedy_test500_v1",
    split: str = "test",
    n: int = 500,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn: str = "sdpa",
    batch_sizes: str = "2,4,8",
    active_pool_size: int = 32,
    refill_policy: str = "continuous",
    problem_jsonl: str = "",
    skip_sequential: bool = False,
    baseline_report: str = "",
) -> dict:
    import json
    import os
    import subprocess
    import sys

    os.chdir("/root/asts-spec")
    _install_forward_bench_deps()
    if not problem_jsonl:
        problem_jsonl = (
            f"/root/asts-spec/data/real_commits/real_commit_manifest_balanced_1000_v2_{split}500.jsonl"
        )
    out_dir = f"/data/generic_batched_greedy_baseline/{version}"
    if skip_sequential and not baseline_report:
        baseline_report = "/data/generic_batched_greedy_baseline/generic_batched_greedy_test500_v1/report.json"
    cmd = [
        sys.executable,
        "scripts/run_batched_greedy_eval.py",
        "--problem-jsonl",
        problem_jsonl,
        "--n",
        str(n),
        "--target",
        target,
        "--dtype",
        dtype,
        "--attn",
        attn,
        "--device",
        "cuda",
        "--max-new-tokens",
        "256",
        "--batch-sizes",
        batch_sizes,
        "--active-pool-size",
        str(active_pool_size),
        "--refill-policy",
        refill_policy,
        "--output-dir",
        out_dir,
        "--progress-every-tasks",
        "25",
        "--progress-every-scheduler-steps",
        "1000",
    ]
    if skip_sequential:
        cmd.append("--skip-sequential")
        cmd.extend(["--baseline-report", baseline_report])
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    with open(f"{out_dir}/report.json", "r", encoding="utf-8") as f:
        report = json.load(f)
    with open(f"{out_dir}/report.md", "r", encoding="utf-8") as f:
        report["report_markdown"] = f.read()
    return report


def _write_batched_greedy_pool_sweep_report(local_dir: Path) -> None:
    rows = []
    base_path = local_dir / "report.json"
    if base_path.exists():
        base = json.loads(base_path.read_text())
        seq_tps = float(base.get("sequential", {}).get("tokens_per_sec", 0.0) or 0.0)
        for row in base.get("batched", []):
            if int(row.get("batch_size", 0)) == 8:
                rows.append(
                    {
                        "source": "generic_batched_greedy_test500_v1",
                        "batch_size": 8,
                        "active_pool_size": int(row.get("active_pool_size", 32)),
                        "status": "oom" if row.get("error") else "success",
                        "error": row.get("error", ""),
                        "tok_s": float(row.get("generated_tokens_per_sec", 0.0) or 0.0),
                        "speedup_vs_sequential_greedy": (
                            float(row.get("generated_tokens_per_sec", 0.0) or 0.0) / seq_tps
                            if seq_tps
                            else 0.0
                        ),
                        "model_forwards": int(row.get("model_forwards", 0) or 0),
                        "memory_peak_gb": float(row.get("memory_peak_gb", 0.0) or 0.0),
                        "latency_p50_ms": float(
                            (row.get("task_latency_summary_ms") or {}).get("p50", 0.0) or 0.0
                        ),
                    }
                )
    for report_path in sorted(local_dir.glob("generic_batched_greedy_b8_pool*_test500_v*/report.json")):
        report = json.loads(report_path.read_text())
        seq_tps = float(report.get("sequential", {}).get("tokens_per_sec", 0.0) or 0.0)
        for row in report.get("batched", []):
            if int(row.get("batch_size", 0)) != 8:
                continue
            rows.append(
                {
                    "source": report_path.parent.name,
                    "batch_size": 8,
                    "active_pool_size": int(row.get("active_pool_size", 0) or 0),
                    "status": "oom" if row.get("error") else "success",
                    "error": row.get("error", ""),
                    "tok_s": float(row.get("generated_tokens_per_sec", 0.0) or 0.0),
                    "speedup_vs_sequential_greedy": (
                        float(row.get("generated_tokens_per_sec", 0.0) or 0.0) / seq_tps
                        if seq_tps
                        else 0.0
                    ),
                    "model_forwards": int(row.get("model_forwards", 0) or 0),
                    "memory_peak_gb": float(row.get("memory_peak_gb", 0.0) or 0.0),
                    "latency_p50_ms": float(
                        (row.get("task_latency_summary_ms") or {}).get("p50", 0.0) or 0.0
                    ),
                }
            )
    rows = sorted(rows, key=lambda r: (r["active_pool_size"], r["source"]))
    payload = {
        "description": "Generic greedy b8 active-pool sweep on held-out test500.",
        "rows": rows,
    }
    (local_dir / "b8_pool_sweep_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Generic Greedy b8 Active-Pool Sweep",
        "",
        "| batch | active pool | status | tok/s | speedup vs seq greedy | model forwards | peak GB | p50 latency ms | notes |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        notes = row["error"] or row["source"]
        tok_s = "OOM" if row["status"] == "oom" else f"{row['tok_s']:.1f}"
        speedup = "OOM" if row["status"] == "oom" else f"{row['speedup_vs_sequential_greedy']:.3f}x"
        forwards = "OOM" if row["status"] == "oom" else str(row["model_forwards"])
        peak = "OOM" if row["status"] == "oom" else f"{row['memory_peak_gb']:.2f}"
        latency = "OOM" if row["status"] == "oom" else f"{row['latency_p50_ms']:.1f}"
        lines.append(
            f"| {row['batch_size']} | {row['active_pool_size']} | {row['status']} | "
            f"{tok_s} | {speedup} | {forwards} | {peak} | {latency} | {notes} |"
        )
    (local_dir / "b8_pool_sweep_report.md").write_text("\n".join(lines) + "\n")


@app.local_entrypoint()
def launch_batched_greedy_eval(
    split: str = "test",
    n: int = 500,
    wait: bool = True,
    version: str = "generic_batched_greedy_test500_v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn: str = "sdpa",
    batch_sizes: str = "2,4,8",
    active_pool_size: int = 32,
    refill_policy: str = "continuous",
    problem_jsonl: str = "",
    skip_sequential: bool = False,
    baseline_report: str = "",
) -> None:
    call = run_batched_greedy_eval_job.spawn(
        version=version,
        split=split,
        n=n,
        target=target,
        dtype=dtype,
        attn=attn,
        batch_sizes=batch_sizes,
        active_pool_size=active_pool_size,
        refill_policy=refill_policy,
        problem_jsonl=problem_jsonl,
        skip_sequential=skip_sequential,
        baseline_report=baseline_report,
    )
    print(f"batched_greedy_eval\t{call.object_id}", flush=True)
    if wait:
        report = call.get()
        local_dir = _PROJECT_ROOT / "analysis" / "generic_batched_greedy_baseline"
        local_dir.mkdir(parents=True, exist_ok=True)
        is_pool_sweep = version.startswith("generic_batched_greedy_b8_pool")
        target_dir = local_dir / version if is_pool_sweep else local_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if report.get("report_markdown"):
            (target_dir / "report.md").write_text(report["report_markdown"])
        if is_pool_sweep:
            _write_batched_greedy_pool_sweep_report(local_dir)
        seq = report.get("sequential", {})
        seq_tps = seq.get("tokens_per_sec", 0.0) or 1.0
        parts = []
        for row in report.get("batched", []):
            tps = row.get("generated_tokens_per_sec", 0.0)
            parts.append(f"b{row.get('batch_size')}={tps:.1f}t/s({tps / seq_tps:.3f}x)")
        print("DONE batched_greedy_eval: " + " ".join(parts), flush=True)


@app.local_entrypoint()
def launch_batched_greedy_correctness(
    split: str = "test",
    n: int = 100,
    wait: bool = True,
    version: str = "generic_batched_greedy_fp32_eager_correctness_n100_v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "fp32",
    attn: str = "eager",
    batch_sizes: str = "1,8",
    active_pool_size: int = 32,
    refill_policy: str = "continuous",
    problem_jsonl: str = "",
) -> None:
    launch_batched_greedy_eval(
        split=split,
        n=n,
        wait=wait,
        version=version,
        target=target,
        dtype=dtype,
        attn=attn,
        batch_sizes=batch_sizes,
        active_pool_size=active_pool_size,
        refill_policy=refill_policy,
        problem_jsonl=problem_jsonl,
    )


@app.local_entrypoint()
def launch_continuous_batched_pld_robustness(
    split: str = "train",
    n: int = 500,
    wait: bool = True,
    version: str = "continuous_batched_pld_robustness_alt_split_v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn: str = "sdpa",
    problem_jsonl: str = "",
) -> None:
    """One reviewer-facing robustness run on an alternate split.

    The default uses the available train500 real-commit split with the same
    model and final batch=8 scheduler config.  It intentionally runs one timing
    repeat rather than opening a new sweep.
    """

    call = run_batched_pld_repeated_timing_job.spawn(
        version=version,
        split=split,
        n=n,
        repeats=1,
        target=target,
        dtype=dtype,
        attn=attn,
        batch_sizes="8",
        active_pool_size=32,
        bucket_policy="default",
        refill_policy="continuous",
        problem_jsonl=problem_jsonl,
        write_audit_trace=False,
    )
    print(f"continuous_batched_pld_robustness\t{call.object_id}", flush=True)
    if wait:
        report = call.get()
        local_dir = _PROJECT_ROOT / "analysis" / "continuous_batched_pld_robustness"
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        lines = [
            "# Continuous Batched PLD Robustness Check",
            "",
            f"split: `{split}`  n: `{n}`  target: `{target}`  dtype: `{dtype}`  attn: `{attn}`",
            "",
            "| method | batch | tok/s | speedup | verifier forwards |",
            "|---|---:|---:|---:|---:|",
        ]
        for key, row in sorted(report.get("summary", {}).items()):
            lines.append(
                f"| {row.get('method')} | {row.get('batch_size')} | "
                f"{row.get('tok_s', {}).get('mean', 0.0):.1f} | "
                f"{row.get('speedup', {}).get('mean', 0.0):.3f}x | "
                f"{row.get('verifier_forwards', {}).get('mean', 0.0):.0f} |"
            )
        (local_dir / "report.md").write_text("\n".join(lines) + "\n")
        parts = []
        for key, row in sorted(report.get("summary", {}).items()):
            parts.append(
                f"{key}={row.get('tok_s', {}).get('mean', 0.0):.1f}t/s "
                f"{row.get('speedup', {}).get('mean', 0.0):.3f}x"
            )
        print("DONE continuous_batched_pld_robustness: " + " | ".join(parts), flush=True)


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=14400,
    startup_timeout=1800,
    cpu=4,
)
def run_batched_pld_correctness_job(
    version: str = "v1",
    split: str = "test",
    n: int = 50,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "fp32",
    attn: str = "eager",
    batch_sizes: str = "1,4,8",
    bucket_sizes: str = "1,2,4,8,16,32,64,128",
    bucket_policy: str = "custom",
    refill_policy: str = "continuous",
    active_pool_size: int = 0,
    problem_jsonl: str = "",
    write_audit_traces: bool = True,
) -> dict:
    import json
    import os
    import subprocess
    import sys

    os.chdir("/root/asts-spec")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _install_forward_bench_deps()
    if not problem_jsonl:
        problem_jsonl = (
            f"/root/asts-spec/data/real_commits/real_commit_manifest_balanced_1000_v2_{split}500.jsonl"
        )
    out_dir = f"/data/batched_pld_correctness/{version}"
    cmd = [
        sys.executable,
        "scripts/validate_batched_pld_correctness.py",
        "--problem-jsonl",
        problem_jsonl,
        "--n",
        str(n),
        "--target",
        target,
        "--dtype",
        dtype,
        "--attn",
        attn,
        "--device",
        "cuda",
        "--max-new-tokens",
        "256",
        "--batch-sizes",
        batch_sizes,
        "--bucket-sizes",
        bucket_sizes,
        "--bucket-policy",
        bucket_policy,
        "--refill-policy",
        refill_policy,
        "--active-pool-size",
        str(active_pool_size),
        "--output-dir",
        out_dir,
    ]
    if write_audit_traces:
        cmd.append("--write-audit-traces")
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    audit_reports = {}
    if write_audit_traces:
        for raw_batch in batch_sizes.split(","):
            raw_batch = raw_batch.strip()
            if not raw_batch:
                continue
            trace_path = f"{out_dir}/batch{raw_batch}_audit_trace.jsonl"
            audit_dir = f"/data/batched_pld_task_audit/{version}/batch{raw_batch}"
            audit_cmd = [
                sys.executable,
                "scripts/audit_batched_pld_task_isolation.py",
                "--trace",
                trace_path,
                "--output-dir",
                audit_dir,
            ]
            print(f"$ {' '.join(audit_cmd)}", flush=True)
            subprocess.run(audit_cmd, check=True)
            with open(f"{audit_dir}/report.json", "r", encoding="utf-8") as f:
                audit_reports[f"batch{raw_batch}"] = json.load(f)
    data_volume.commit()
    with open(f"{out_dir}/report.json", "r", encoding="utf-8") as f:
        report = json.load(f)
    report["audit_reports"] = audit_reports
    return report


@app.local_entrypoint()
def launch_batched_pld_correctness(
    split: str = "test",
    n: int = 50,
    wait: bool = True,
    version: str = "v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "fp32",
    attn: str = "eager",
    batch_sizes: str = "1,4,8",
    bucket_sizes: str = "1,2,4,8,16,32,64,128",
    bucket_policy: str = "custom",
    refill_policy: str = "continuous",
    active_pool_size: int = 0,
    problem_jsonl: str = "",
    write_audit_traces: bool = True,
) -> None:
    call = run_batched_pld_correctness_job.spawn(
        version=version,
        split=split,
        n=n,
        target=target,
        dtype=dtype,
        attn=attn,
        batch_sizes=batch_sizes,
        bucket_sizes=bucket_sizes,
        bucket_policy=bucket_policy,
        refill_policy=refill_policy,
        active_pool_size=active_pool_size,
        problem_jsonl=problem_jsonl,
        write_audit_traces=write_audit_traces,
    )
    print(f"batched_pld_correctness\t{call.object_id}", flush=True)
    if wait:
        report = call.get()
        local_dir = _PROJECT_ROOT / "analysis" / "batched_pld_correctness" / version
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        audit_dir = _PROJECT_ROOT / "analysis" / "batched_pld_task_audit" / version
        audit_dir.mkdir(parents=True, exist_ok=True)
        (audit_dir / "report.json").write_text(
            json.dumps(report.get("audit_reports", {}), indent=2, sort_keys=True) + "\n"
        )
        parts = []
        for row in report.get("rows", []):
            parts.append(
                f"b{row.get('batch_size')}={row.get('matches')}/"
                f"{row.get('matches', 0) + row.get('mismatches', 0)}"
            )
        print(
            "DONE batched_pld_correctness: "
            + " ".join(parts)
            + f" all_exact={report.get('all_exact')}",
            flush=True,
        )


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=86400,
    startup_timeout=1800,
    cpu=4,
)
def run_batched_pld_correctness_sharded_job(
    version: str = "continuous_batched_pld_fp32_eager_correctness_sharded_v1",
    split: str = "test",
    n: int = 500,
    shard_size: int = 50,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "fp32",
    attn: str = "eager",
    batch_sizes: str = "1,4,8",
    bucket_sizes: str = "1,2,4,8,16,32,64,128",
    bucket_policy: str = "default",
    refill_policy: str = "continuous",
    active_pool_size: int = 32,
    problem_jsonl: str = "",
    write_audit_traces: bool = False,
) -> dict:
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    os.chdir("/root/asts-spec")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _install_forward_bench_deps()
    if not problem_jsonl:
        problem_jsonl = (
            f"/root/asts-spec/data/real_commits/real_commit_manifest_balanced_1000_v2_{split}500.jsonl"
        )
    out_dir = f"/data/continuous_batched_pld_correctness_sharded/{version}"
    cmd = [
        sys.executable,
        "scripts/validate_batched_pld_correctness_sharded.py",
        "--problem-jsonl",
        problem_jsonl,
        "--n",
        str(n),
        "--shard-size",
        str(shard_size),
        "--target",
        target,
        "--dtype",
        dtype,
        "--attn",
        attn,
        "--device",
        "cuda",
        "--max-new-tokens",
        "256",
        "--batch-sizes",
        batch_sizes,
        "--bucket-sizes",
        bucket_sizes,
        "--bucket-policy",
        bucket_policy,
        "--refill-policy",
        refill_policy,
        "--active-pool-size",
        str(active_pool_size),
        "--output-dir",
        out_dir,
    ]
    if write_audit_traces:
        cmd.append("--write-audit-traces")
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    out_path = Path(out_dir)
    with open(out_path / "report.json", "r", encoding="utf-8") as f:
        report = json.load(f)
    report["report_markdown"] = (out_path / "report.md").read_text(encoding="utf-8")
    shard_reports = []
    for shard_dir in sorted(out_path.glob("shard_*")):
        report_json = shard_dir / "report.json"
        report_md = shard_dir / "report.md"
        if not report_json.exists():
            continue
        shard_reports.append(
            {
                "name": shard_dir.name,
                "report_json": json.loads(report_json.read_text(encoding="utf-8")),
                "report_markdown": report_md.read_text(encoding="utf-8")
                if report_md.exists()
                else "",
            }
        )
    report["shard_reports"] = shard_reports
    return report


@app.local_entrypoint()
def launch_batched_pld_correctness_sharded(
    split: str = "test",
    n: int = 500,
    shard_size: int = 50,
    wait: bool = True,
    version: str = "continuous_batched_pld_fp32_eager_correctness_sharded_v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "fp32",
    attn: str = "eager",
    batch_sizes: str = "1,4,8",
    bucket_sizes: str = "1,2,4,8,16,32,64,128",
    bucket_policy: str = "default",
    refill_policy: str = "continuous",
    active_pool_size: int = 32,
    problem_jsonl: str = "",
    write_audit_traces: bool = False,
) -> None:
    call = run_batched_pld_correctness_sharded_job.spawn(
        version=version,
        split=split,
        n=n,
        shard_size=shard_size,
        target=target,
        dtype=dtype,
        attn=attn,
        batch_sizes=batch_sizes,
        bucket_sizes=bucket_sizes,
        bucket_policy=bucket_policy,
        refill_policy=refill_policy,
        active_pool_size=active_pool_size,
        problem_jsonl=problem_jsonl,
        write_audit_traces=write_audit_traces,
    )
    print(f"batched_pld_correctness_sharded\t{call.object_id}", flush=True)
    if wait:
        report = call.get()
        local_dir = _PROJECT_ROOT / "analysis" / "continuous_batched_pld_correctness_sharded"
        local_dir.mkdir(parents=True, exist_ok=True)
        report_markdown = report.pop("report_markdown", "")
        shard_reports = report.pop("shard_reports", [])
        (local_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if report_markdown:
            (local_dir / "report.md").write_text(report_markdown)
        for shard in shard_reports:
            shard_dir = local_dir / shard["name"]
            shard_dir.mkdir(parents=True, exist_ok=True)
            (shard_dir / "report.json").write_text(
                json.dumps(shard["report_json"], indent=2, sort_keys=True) + "\n"
            )
            if shard.get("report_markdown"):
                (shard_dir / "report.md").write_text(shard["report_markdown"])
        agg = report.get("aggregate", {})
        parts = []
        for batch, row in sorted(agg.get("batch_results", {}).items(), key=lambda item: int(item[0])):
            parts.append(
                f"b{batch}={row.get('exact_token_id_matches')}/{row.get('tasks')}"
            )
        print(
            "DONE batched_pld_correctness_sharded: "
            + " ".join(parts)
            + f" all_exact={agg.get('all_exact')}",
            flush=True,
        )


@app.function(volumes={"/data": data_volume}, timeout=1800, startup_timeout=600, cpu=1)
def run_batched_pld_task_audit_job(
    correctness_version: str,
    audit_version: str = "",
    batch_sizes: str = "1,4,8",
) -> dict:
    import json
    import os
    import subprocess
    import sys

    os.chdir("/root/asts-spec")
    if not audit_version:
        audit_version = correctness_version
    reports = {}
    for raw_batch in batch_sizes.split(","):
        raw_batch = raw_batch.strip()
        if not raw_batch:
            continue
        trace_path = f"/data/batched_pld_correctness/{correctness_version}/batch{raw_batch}_audit_trace.jsonl"
        audit_dir = f"/data/batched_pld_task_audit/{audit_version}/batch{raw_batch}"
        cmd = [
            sys.executable,
            "scripts/audit_batched_pld_task_isolation.py",
            "--trace",
            trace_path,
            "--output-dir",
            audit_dir,
        ]
        print(f"$ {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
        with open(f"{audit_dir}/report.json", "r", encoding="utf-8") as f:
            reports[f"batch{raw_batch}"] = json.load(f)
    data_volume.commit()
    return reports


@app.local_entrypoint()
def launch_batched_pld_task_audit(
    correctness_version: str,
    wait: bool = True,
    audit_version: str = "",
    batch_sizes: str = "1,4,8",
) -> None:
    call = run_batched_pld_task_audit_job.spawn(
        correctness_version=correctness_version,
        audit_version=audit_version,
        batch_sizes=batch_sizes,
    )
    print(f"batched_pld_task_audit\t{call.object_id}", flush=True)
    if wait:
        report = call.get()
        local_dir = _PROJECT_ROOT / "analysis" / "batched_pld_task_audit" / (audit_version or correctness_version)
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        parts = [
            f"{k}:pass={v.get('passed')} violations={v.get('violation_count')}"
            for k, v in sorted(report.items())
        ]
        print("DONE batched_pld_task_audit: " + " ".join(parts), flush=True)


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=21600,
    startup_timeout=1800,
    cpu=4,
)
def run_batched_pld_ablation_job(
    version: str = "v1",
    split: str = "test",
    n: int = 500,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn: str = "sdpa",
    batch_sizes: str = "1,2,4,8",
    active_pool_sizes: str = "8,16,32",
    bucket_policies: str = "default,fine",
    refill_policies: str = "continuous,no_refill",
    problem_jsonl: str = "",
) -> dict:
    import json
    import os
    import subprocess
    import sys

    os.chdir("/root/asts-spec")
    _install_forward_bench_deps()
    if not problem_jsonl:
        problem_jsonl = (
            f"/root/asts-spec/data/real_commits/real_commit_manifest_balanced_1000_v2_{split}500.jsonl"
        )
    out_dir = f"/data/batched_pld_ablation/{version}"
    cmd = [
        sys.executable,
        "scripts/run_batched_pld_ablation.py",
        "--problem-jsonl",
        problem_jsonl,
        "--n",
        str(n),
        "--target",
        target,
        "--dtype",
        dtype,
        "--attn",
        attn,
        "--device",
        "cuda",
        "--max-new-tokens",
        "256",
        "--batch-sizes",
        batch_sizes,
        "--active-pool-sizes",
        active_pool_sizes,
        "--bucket-policies",
        bucket_policies,
        "--refill-policies",
        refill_policies,
        "--output-dir",
        out_dir,
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    with open(f"{out_dir}/report.json", "r", encoding="utf-8") as f:
        return json.load(f)


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=43200,
    startup_timeout=1800,
    cpu=4,
)
def run_batched_pld_controlled_ablation_job(
    version: str = "controlled_ablation_test500_v1",
    split: str = "test",
    n: int = 500,
    repeats: int = 3,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn: str = "sdpa",
    problem_jsonl: str = "",
) -> dict:
    import json
    import os
    import subprocess
    import sys

    os.chdir("/root/asts-spec")
    _install_forward_bench_deps()
    if not problem_jsonl:
        problem_jsonl = (
            f"/root/asts-spec/data/real_commits/real_commit_manifest_balanced_1000_v2_{split}500.jsonl"
        )
    out_dir = f"/data/batched_pld_controlled_ablation/{version}"
    cmd = [
        sys.executable,
        "scripts/run_batched_pld_controlled_ablation.py",
        "--problem-jsonl",
        problem_jsonl,
        "--n",
        str(n),
        "--repeats",
        str(repeats),
        "--target",
        target,
        "--dtype",
        dtype,
        "--attn",
        attn,
        "--device",
        "cuda",
        "--max-new-tokens",
        "256",
        "--output-dir",
        out_dir,
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    with open(f"{out_dir}/summary.json", "r", encoding="utf-8") as f:
        return json.load(f)


@app.local_entrypoint()
def launch_batched_pld_controlled_ablation(
    split: str = "test",
    n: int = 500,
    repeats: int = 3,
    wait: bool = True,
    version: str = "controlled_ablation_test500_v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn: str = "sdpa",
    problem_jsonl: str = "",
) -> None:
    call = run_batched_pld_controlled_ablation_job.spawn(
        version=version,
        split=split,
        n=n,
        repeats=repeats,
        target=target,
        dtype=dtype,
        attn=attn,
        problem_jsonl=problem_jsonl,
    )
    print(f"batched_pld_controlled_ablation\t{call.object_id}", flush=True)
    if wait:
        report = call.get()
        local_dir = _PROJECT_ROOT / "analysis" / "batched_pld_controlled_ablation" / version
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        source_md = report.get("metadata", {}).get("args", {})
        summary = report.get("summary", {})
        parts = []
        for key in ("seq", "b2_pool32_default_continuous", "b4_pool32_default_continuous", "b8_pool32_default_continuous"):
            item = summary.get(key, {})
            tok_s = item.get("fields", {}).get("tok_s", {}).get("mean", 0.0)
            parts.append(f"{key}={tok_s:.1f}t/s")
        print(
            "DONE batched_pld_controlled_ablation: "
            + " | ".join(parts)
            + f" args={source_md}",
            flush=True,
        )


@app.local_entrypoint()
def launch_batched_pld_ablation(
    split: str = "test",
    n: int = 500,
    wait: bool = True,
    version: str = "v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn: str = "sdpa",
    batch_sizes: str = "1,2,4,8",
    active_pool_sizes: str = "8,16,32",
    bucket_policies: str = "default,fine",
    refill_policies: str = "continuous,no_refill",
    problem_jsonl: str = "",
) -> None:
    call = run_batched_pld_ablation_job.spawn(
        version=version,
        split=split,
        n=n,
        target=target,
        dtype=dtype,
        attn=attn,
        batch_sizes=batch_sizes,
        active_pool_sizes=active_pool_sizes,
        bucket_policies=bucket_policies,
        refill_policies=refill_policies,
        problem_jsonl=problem_jsonl,
    )
    print(f"batched_pld_ablation\t{call.object_id}", flush=True)
    if wait:
        report = call.get()
        local_dir = _PROJECT_ROOT / "analysis" / "batched_pld_ablation" / version
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        best = max(report.get("rows", []), key=lambda r: r.get("speedup_vs_sequential", 0.0), default={})
        print(
            "DONE batched_pld_ablation: "
            f"best={best.get('config_id')} "
            f"speedup={best.get('speedup_vs_sequential', 0.0):.3f}x",
            flush=True,
        )


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=7200,
    startup_timeout=1800,
    cpu=4,
)
def run_real_shape_forward_profile_job(
    version: str = "v1",
    source_run_tag: str = "vantage_real_commit_pld_adjacent_test500_rerank_fixed_rank3_test500_v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn: str = "sdpa",
    max_steps: int = 100,
) -> dict:
    import json
    import os
    import subprocess
    import sys

    os.chdir("/root/asts-spec")
    _install_forward_bench_deps()
    eval_dir = f"/data/{source_run_tag}/eval"
    out_dir = f"/data/pld_forward_profile/{version}"
    cmd = [
        sys.executable,
        "scripts/profile_real_shape_forward_internals.py",
        "--steps",
        f"{eval_dir}/steps.jsonl",
        "--completions",
        f"{eval_dir}/completions.jsonl",
        "--target",
        target,
        "--dtype",
        dtype,
        "--attn",
        attn,
        "--device",
        "cuda",
        "--max-steps",
        str(max_steps),
        "--output-dir",
        out_dir,
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    with open(f"{out_dir}/report.json", "r", encoding="utf-8") as f:
        return json.load(f)


@app.local_entrypoint()
def launch_real_shape_forward_profile(
    wait: bool = True,
    version: str = "v1",
    source_run_tag: str = "vantage_real_commit_pld_adjacent_test500_rerank_fixed_rank3_test500_v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn: str = "sdpa",
    max_steps: int = 100,
) -> None:
    call = run_real_shape_forward_profile_job.spawn(
        version=version,
        source_run_tag=source_run_tag,
        target=target,
        dtype=dtype,
        attn=attn,
        max_steps=max_steps,
    )
    print(f"real_shape_forward_profile\t{call.object_id}", flush=True)
    if wait:
        report = call.get()
        agg = report.get("aggregate", {})
        print(
            "DONE real_shape_forward_profile: "
            f"mean={agg.get('forward_ms', {}).get('mean', 0.0):.3f}ms "
            f"p90={agg.get('forward_ms', {}).get('p90', 0.0):.3f}ms",
            flush=True,
        )


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=7200,
    startup_timeout=1800,
    cpu=4,
)
def run_bucketed_static_verifier_job(
    version: str = "v1",
    source_run_tag: str = "vantage_real_commit_pld_adjacent_test500_rerank_fixed_rank3_test500_v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn: str = "sdpa",
    max_steps: int = 500,
    bucket_sizes: str = "1,2,4,8,16,32,64,128",
    enable_cuda_graphs: bool = False,
    enable_torch_compile: bool = False,
    static_cache: bool = False,
) -> dict:
    import json
    import os
    import subprocess
    import sys

    os.chdir("/root/asts-spec")
    _install_forward_bench_deps()
    eval_dir = f"/data/{source_run_tag}/eval"
    out_dir = f"/data/pld_bucketed_static_verifier/{version}"
    cmd = [
        sys.executable,
        "scripts/benchmark_bucketed_static_verifier.py",
        "--steps",
        f"{eval_dir}/steps.jsonl",
        "--completions",
        f"{eval_dir}/completions.jsonl",
        "--target",
        target,
        "--dtype",
        dtype,
        "--attn",
        attn,
        "--device",
        "cuda",
        "--bucket-sizes",
        bucket_sizes,
        "--enable-cuda-graphs",
        str(enable_cuda_graphs).lower(),
        "--enable-torch-compile",
        str(enable_torch_compile).lower(),
        "--static-cache",
        str(static_cache).lower(),
        "--max-steps",
        str(max_steps),
        "--output-dir",
        out_dir,
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    with open(f"{out_dir}/report.json", "r", encoding="utf-8") as f:
        return json.load(f)


@app.local_entrypoint()
def launch_bucketed_static_verifier(
    wait: bool = True,
    version: str = "v1",
    source_run_tag: str = "vantage_real_commit_pld_adjacent_test500_rerank_fixed_rank3_test500_v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn: str = "sdpa",
    max_steps: int = 500,
    bucket_sizes: str = "1,2,4,8,16,32,64,128",
    enable_cuda_graphs: bool = False,
    enable_torch_compile: bool = False,
    static_cache: bool = False,
) -> None:
    call = run_bucketed_static_verifier_job.spawn(
        version=version,
        source_run_tag=source_run_tag,
        target=target,
        dtype=dtype,
        attn=attn,
        max_steps=max_steps,
        bucket_sizes=bucket_sizes,
        enable_cuda_graphs=enable_cuda_graphs,
        enable_torch_compile=enable_torch_compile,
        static_cache=static_cache,
    )
    print(f"bucketed_static_verifier\t{call.object_id}", flush=True)
    if wait:
        report = call.get()
        agg = report.get("aggregate", {})
        print(
            "DONE bucketed_static_verifier: "
            f"mean={agg.get('forward_ms', {}).get('mean', 0.0):.3f}ms "
            f"p90={agg.get('forward_ms', {}).get('p90', 0.0):.3f}ms "
            f"compile={report.get('enable_torch_compile')} "
            f"graphs={report.get('enable_cuda_graphs')}",
            flush=True,
        )


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=7200,
    startup_timeout=1800,
    cpu=4,
)
def run_lm_head_cost_profile_job(
    version: str = "v1",
    source_run_tag: str = "vantage_real_commit_pld_adjacent_test500_rerank_fixed_rank3_test500_v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn: str = "sdpa",
    max_steps: int = 500,
) -> dict:
    import json
    import os
    import subprocess
    import sys

    os.chdir("/root/asts-spec")
    _install_forward_bench_deps()
    eval_dir = f"/data/{source_run_tag}/eval"
    out_dir = f"/data/selective_lm_head/profile/{version}"
    cmd = [
        sys.executable,
        "scripts/profile_lm_head_cost.py",
        "--steps",
        f"{eval_dir}/steps.jsonl",
        "--completions",
        f"{eval_dir}/completions.jsonl",
        "--target",
        target,
        "--dtype",
        dtype,
        "--attn",
        attn,
        "--device",
        "cuda",
        "--max-steps",
        str(max_steps),
        "--output-dir",
        out_dir,
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    with open(f"{out_dir}/report.json", "r", encoding="utf-8") as f:
        return json.load(f)


@app.local_entrypoint()
def launch_lm_head_cost_profile(
    wait: bool = True,
    version: str = "v1",
    source_run_tag: str = "vantage_real_commit_pld_adjacent_test500_rerank_fixed_rank3_test500_v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn: str = "sdpa",
    max_steps: int = 500,
) -> None:
    call = run_lm_head_cost_profile_job.spawn(
        version=version,
        source_run_tag=source_run_tag,
        target=target,
        dtype=dtype,
        attn=attn,
        max_steps=max_steps,
    )
    print(f"lm_head_cost_profile\t{call.object_id}", flush=True)
    if wait:
        report = call.get()
        agg = report.get("aggregate", {})
        print(
            "DONE lm_head_cost_profile: "
            f"lm_head={agg.get('lm_head_only_ms', {}).get('mean', 0.0):.3f}ms "
            f"full={agg.get('full_logits_forward_ms', {}).get('mean', 0.0):.3f}ms "
            f"share={agg.get('lm_head_share_of_full_forward', 0.0):.1%}",
            flush=True,
        )


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=14400,
    startup_timeout=1800,
    cpu=8,
)
def run_selective_lm_head_eval_job(
    version: str = "v1",
    source_run_tag: str = "vantage_real_commit_pld_adjacent_test500_rerank_fixed_rank3_test500_v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn: str = "sdpa",
    max_steps: int = 200,
    num_clusters: int = 256,
    cluster_method: str = "random_projection",
    lm_head_profile_version: str = "lm_head_cost_test500_v1",
) -> dict:
    import json
    import os
    import subprocess
    import sys

    os.chdir("/root/asts-spec")
    _install_forward_bench_deps()
    eval_dir = f"/data/{source_run_tag}/eval"
    out_dir = f"/data/selective_lm_head/offline_eval/{version}"
    cluster_path = f"{out_dir}/lm_head_clusters_k{num_clusters}.pt"
    profile_json = f"/data/selective_lm_head/profile/{lm_head_profile_version}/report.json"
    os.makedirs(out_dir, exist_ok=True)
    build_cmd = [
        sys.executable,
        "scripts/build_lm_head_clusters.py",
        "--target",
        target,
        "--dtype",
        dtype,
        "--attn",
        attn,
        "--num-clusters",
        str(num_clusters),
        "--method",
        cluster_method,
        "--output",
        cluster_path,
        "--report",
        f"{out_dir}/clusters_report.json",
    ]
    print(f"$ {' '.join(build_cmd)}", flush=True)
    subprocess.run(build_cmd, check=True)
    eval_cmd = [
        sys.executable,
        "scripts/evaluate_selective_lm_head_verifier.py",
        "--steps",
        f"{eval_dir}/steps.jsonl",
        "--completions",
        f"{eval_dir}/completions.jsonl",
        "--clusters",
        cluster_path,
        "--target",
        target,
        "--dtype",
        dtype,
        "--attn",
        attn,
        "--device",
        "cuda",
        "--max-steps",
        str(max_steps),
        "--lm-head-profile-json",
        profile_json,
        "--output-dir",
        out_dir,
    ]
    print(f"$ {' '.join(eval_cmd)}", flush=True)
    subprocess.run(eval_cmd, check=True)
    data_volume.commit()
    with open(f"{out_dir}/report.json", "r", encoding="utf-8") as f:
        return json.load(f)


@app.local_entrypoint()
def launch_selective_lm_head_eval(
    wait: bool = True,
    version: str = "v1",
    source_run_tag: str = "vantage_real_commit_pld_adjacent_test500_rerank_fixed_rank3_test500_v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn: str = "sdpa",
    max_steps: int = 200,
    num_clusters: int = 256,
    cluster_method: str = "random_projection",
    lm_head_profile_version: str = "lm_head_cost_test500_v1",
) -> None:
    call = run_selective_lm_head_eval_job.spawn(
        version=version,
        source_run_tag=source_run_tag,
        target=target,
        dtype=dtype,
        attn=attn,
        max_steps=max_steps,
        num_clusters=num_clusters,
        cluster_method=cluster_method,
        lm_head_profile_version=lm_head_profile_version,
    )
    print(f"selective_lm_head_eval\t{call.object_id}", flush=True)
    if wait:
        report = call.get()
        summary = report.get("summary", {})
        print(
            "DONE selective_lm_head_eval: "
            f"cert={summary.get('certification_rate', 0.0):.1%} "
            f"mismatch={summary.get('exactness_mismatches', -1)} "
            f"speedup={summary.get('projected_end_to_end_speedup', 0.0):.3f}x",
            flush=True,
        )


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=7200,
    startup_timeout=1800,
    cpu=4,
)
def run_diff_hunk_generation_eval_job(
    version: str = "v1",
    source_run_tag: str = "vantage_real_commit_pld_adjacent_test500_rerank_fixed_rank3_test500_v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn: str = "sdpa",
    max_tasks: int = 50,
    prompt_format: str = "all",
    max_new_tokens: int = 256,
    baseline_method: str = "blazedit_pld_w128_n10",
) -> dict:
    import json
    import os
    import subprocess
    import sys

    os.chdir("/root/asts-spec")
    _install_forward_bench_deps()
    input_path = f"/data/{source_run_tag}/eval/completions.jsonl"
    out_dir = f"/data/diff_hunk_generation/{version}"
    cmd = [
        sys.executable,
        "scripts/run_diff_hunk_generation_eval.py",
        "--input",
        input_path,
        "--generate-model",
        "--target",
        target,
        "--dtype",
        dtype,
        "--attn",
        attn,
        "--device",
        "cuda",
        "--prompt-format",
        prompt_format,
        "--max-tasks",
        str(max_tasks),
        "--max-new-tokens",
        str(max_new_tokens),
        "--baseline-method",
        baseline_method,
        "--output-dir",
        out_dir,
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    with open(f"{out_dir}/report.json", "r", encoding="utf-8") as f:
        return json.load(f)


@app.local_entrypoint()
def launch_diff_hunk_generation_eval(
    wait: bool = True,
    version: str = "v1",
    source_run_tag: str = "vantage_real_commit_pld_adjacent_test500_rerank_fixed_rank3_test500_v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dtype: str = "bf16",
    attn: str = "sdpa",
    max_tasks: int = 50,
    prompt_format: str = "all",
    max_new_tokens: int = 256,
    baseline_method: str = "blazedit_pld_w128_n10",
) -> None:
    call = run_diff_hunk_generation_eval_job.spawn(
        version=version,
        source_run_tag=source_run_tag,
        target=target,
        dtype=dtype,
        attn=attn,
        max_tasks=max_tasks,
        prompt_format=prompt_format,
        max_new_tokens=max_new_tokens,
        baseline_method=baseline_method,
    )
    print(f"diff_hunk_generation_eval\t{call.object_id}", flush=True)
    if wait:
        report = call.get()
        parts = []
        for group in report.get("groups", []):
            exact_rate = group.get("exact_match_rate")
            exact_text = f"{exact_rate:.1%}" if exact_rate is not None else "-"
            parts.append(
                f"{group.get('method')}: apply={group.get('apply_success_rate', 0.0):.1%} "
                f"exact={exact_text} "
                f"speed={group.get('mean_speedup_vs_full_file_pld') or 0.0:.3f}x"
            )
        print("DONE diff_hunk_generation_eval: " + " | ".join(parts), flush=True)


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    timeout=7200,
    startup_timeout=1800,
    cpu=4,
)
def run_pld_candidate_oracle_job(
    source_run_tag: str,
    version: str = "v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    method: str = "blazedit_pld_w128_n10",
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
        "transformers>=4.46",
        "huggingface-hub>=0.26",
        "tree-sitter>=0.23.0",
        "tree-sitter-language-pack>=0.4.0",
        "numpy>=1.26",
    ]
    print(f"$ {' '.join(install_cmd)}", flush=True)
    subprocess.run(install_cmd, check=True)

    eval_dir = f"/data/{source_run_tag}/eval"
    out_dir = f"/data/{source_run_tag}/pld_candidate_oracle_{version}"
    os.makedirs(out_dir, exist_ok=True)
    json_out = f"{out_dir}/report.json"
    md_out = f"{out_dir}/report.md"
    details_out = f"{out_dir}/ambiguous_candidates.jsonl"
    cmd = [
        sys.executable,
        "scripts/analyze_pld_candidate_oracle.py",
        "--steps",
        f"{eval_dir}/steps.jsonl",
        "--completions",
        f"{eval_dir}/completions.jsonl",
        "--target",
        target,
        "--method",
        method,
        "--json-out",
        json_out,
        "--details-jsonl-out",
        details_out,
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    with open(md_out, "w") as md:
        subprocess.run(cmd, check=True, stdout=md, stderr=subprocess.STDOUT)
    data_volume.commit()
    hf_cache.commit()
    with open(json_out) as f:
        report = json.load(f)
    report["source_run_tag"] = source_run_tag
    report["report_dir"] = out_dir
    return report


def _run_lossless_impl(
    run_tag: str,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    target_trust_remote_code: bool = False,
    eagle_checkpoint: str = "/data/eagle_v1_normfix/eagle/eagle_final.pt",
    n: int = 100,
    max_new_tokens: int = 256,
    code_methods: str = "blazedit_pld_w128_n10,vantage_frozen_transpld",
    problem_jsonl: str = "",
    dtype: str = "float32",
    attn_impl: str = "eager",
) -> dict:
    import json
    import os
    import subprocess
    import sys

    os.chdir("/root/asts-spec")
    install_cmds = [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
            "torch==2.5.1",
        ],
        [
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
        ],
    ]
    for install_cmd in install_cmds:
        print(f"$ {' '.join(install_cmd)}", flush=True)
        subprocess.run(install_cmd, check=True)

    output_path = f"/data/{run_tag}/lossless/results.json"
    cmd = [
        sys.executable,
        "scripts/verify_eagle_lossless.py",
        "--target",
        target,
        "--eagle-checkpoint",
        eagle_checkpoint,
        "--n",
        str(n),
        "--max-new-tokens",
        str(max_new_tokens),
        "--dtype",
        dtype,
        "--attn-impl",
        attn_impl,
        "--problem-jsonl",
        problem_jsonl,
        "--strict-determinism",
        "--skip-fixed-eagle",
        "--skip-tree",
        "--skip-asts",
        "--skip-eagle2",
        "--skip-eagle-load",
        "--include-code-proposers",
        "--code-methods",
        code_methods,
        "--output",
        output_path,
        "--log-level",
        "INFO",
    ]
    if target_trust_remote_code:
        cmd.append("--target-trust-remote-code")
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    hf_cache.commit()
    with open(output_path) as f:
        payload = json.load(f)
    return {
        "run_tag": run_tag,
        "output_path": output_path,
        "all_match": payload.get("all_match"),
        "n_tasks": payload.get("n_tasks"),
        "n_match_code": payload.get("n_match_code", {}),
    }


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=21600,
    startup_timeout=3600,
    cpu=4,
)
def run_lossless_job(
    run_tag: str,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    target_trust_remote_code: bool = False,
    eagle_checkpoint: str = "/data/eagle_v1_normfix/eagle/eagle_final.pt",
    n: int = 100,
    max_new_tokens: int = 256,
    code_methods: str = "blazedit_pld_w128_n10,vantage_frozen_transpld",
    problem_jsonl: str = "",
    dtype: str = "float32",
    attn_impl: str = "eager",
) -> dict:
    return _run_lossless_impl(
        run_tag=run_tag,
        target=target,
        target_trust_remote_code=target_trust_remote_code,
        eagle_checkpoint=eagle_checkpoint,
        n=n,
        max_new_tokens=max_new_tokens,
        code_methods=code_methods,
        problem_jsonl=problem_jsonl,
        dtype=dtype,
        attn_impl=attn_impl,
    )


@app.local_entrypoint()
def launch_path_b_matrix(version: str = "v4", wait: bool = False) -> None:
    """Spawn the focused adoption-routing validation matrix.

    This local entrypoint is intentionally used instead of shell backgrounding:
    the Modal CLI can terminate background clients in non-interactive shells,
    while `Function.spawn()` schedules independent remote calls.
    """

    methods = (
        "vanilla,blazedit_pld_w128_n10,"
        "vantage_routed_transpld_m4_w128_n10,"
        "vantage_adopt_simple_transpld_m4_w128_n10"
    )

    calls = []

    def spawn(
        name: str,
        *,
        target: str,
        chat_template: str,
        problem_jsonl: str,
        n: int = 50,
        trust_remote_code: bool = False,
    ) -> None:
        tag = f"vantage_adopt_{name}_{version}"
        call = run_eagle_eval_job_any.spawn(
            run_tag=tag,
            target=target,
            target_trust_remote_code=trust_remote_code,
            n=n,
            max_new_tokens=256,
            methods=methods,
            problem_jsonl=problem_jsonl,
            dtype="bfloat16",
            attn_impl="sdpa",
            code_proposer_fallback="root",
            transpld_min_match_len=4,
            chat_template=chat_template,
        )
        print(f"{tag}\t{call.object_id}", flush=True)
        calls.append((tag, call))

    field = "/root/asts-spec/data/manifests_phase3/drift_nonrename_field_rename.jsonl"
    style = "/root/asts-spec/data/manifests_phase3/drift_nonrename_style_rewrite.jsonl"
    zero = "/root/asts-spec/data/manifests_transpld_ext/drift_renamepct_0.jsonl"
    span = "/root/asts-spec/data/manifests_phase2/drift_axis_span.jsonl"
    editdist = "/root/asts-spec/data/manifests_phase2/drift_axis_editdist.jsonl"

    spawn("qwen_base_field50", target="Qwen/Qwen2.5-Coder-7B", chat_template="none", problem_jsonl=field)
    spawn("qwen_base_style50", target="Qwen/Qwen2.5-Coder-7B", chat_template="none", problem_jsonl=style)
    spawn("qwen_base_zero50", target="Qwen/Qwen2.5-Coder-7B", chat_template="none", problem_jsonl=zero)

    spawn(
        "deepseek_instruct_field50",
        target="deepseek-ai/deepseek-coder-6.7b-instruct",
        chat_template="user",
        problem_jsonl=field,
        trust_remote_code=True,
    )
    spawn(
        "deepseek_instruct_style50",
        target="deepseek-ai/deepseek-coder-6.7b-instruct",
        chat_template="user",
        problem_jsonl=style,
        trust_remote_code=True,
    )
    spawn(
        "deepseek_instruct_zero50",
        target="deepseek-ai/deepseek-coder-6.7b-instruct",
        chat_template="user",
        problem_jsonl=zero,
        trust_remote_code=True,
    )

    spawn(
        "qwen_instruct_field50",
        target="Qwen/Qwen2.5-Coder-7B-Instruct",
        chat_template="user",
        problem_jsonl=field,
    )
    spawn(
        "qwen_instruct_style50",
        target="Qwen/Qwen2.5-Coder-7B-Instruct",
        chat_template="user",
        problem_jsonl=style,
    )
    spawn(
        "qwen_instruct_zero50",
        target="Qwen/Qwen2.5-Coder-7B-Instruct",
        chat_template="user",
        problem_jsonl=zero,
    )
    spawn(
        "qwen_instruct_span50",
        target="Qwen/Qwen2.5-Coder-7B-Instruct",
        chat_template="user",
        problem_jsonl=span,
    )
    spawn(
        "qwen_instruct_editdist50",
        target="Qwen/Qwen2.5-Coder-7B-Instruct",
        chat_template="user",
        problem_jsonl=editdist,
    )

    if wait:
        print("Waiting for spawned calls...", flush=True)
        for tag, call in calls:
            try:
                result = call.get()
                by_method = result.get("by_method", {})
                parts = []
                for method, row in by_method.items():
                    tok_s = row.get("tokens_per_sec", 0.0)
                    parts.append(f"{method}={tok_s:.1f}t/s")
                print(f"DONE {tag}: " + "  ".join(parts), flush=True)
            except Exception as exc:
                print(f"FAILED {tag}: {exc}", flush=True)
                raise


@app.local_entrypoint()
def launch_dispatch_regression(version: str = "v1", wait: bool = False) -> None:
    """Run the reviewer-requested prompt-time dispatch regression check."""

    methods = (
        "vanilla,blazedit_pld_w128_n10,"
        "rewrite_pld_bidir_m4_w128_n10,"
        "vantage_dispatch_transpld_m4_w128_n10"
    )
    calls = []

    def spawn(name: str, problem_jsonl: str) -> None:
        tag = f"vantage_dispatch_{name}_{version}"
        call = run_eagle_eval_job_any.spawn(
            run_tag=tag,
            target="Qwen/Qwen2.5-Coder-7B",
            target_trust_remote_code=False,
            n=50,
            max_new_tokens=256,
            methods=methods,
            problem_jsonl=problem_jsonl,
            dtype="bfloat16",
            attn_impl="sdpa",
            code_proposer_fallback="root",
            transpld_min_match_len=4,
            chat_template="none",
        )
        print(f"{tag}\t{call.object_id}", flush=True)
        calls.append((tag, call))

    field = "/root/asts-spec/data/manifests_phase3/drift_nonrename_field_rename.jsonl"
    style = "/root/asts-spec/data/manifests_phase3/drift_nonrename_style_rewrite.jsonl"
    zero = "/root/asts-spec/data/manifests_transpld_ext/drift_renamepct_0.jsonl"

    spawn("qwen_base_field50", field)
    spawn("qwen_base_style50", style)
    spawn("qwen_base_zero50", zero)

    if wait:
        print("Waiting for spawned calls...", flush=True)
        for tag, call in calls:
            try:
                result = call.get()
                by_method = result.get("by_method", {})
                parts = []
                for method, row in by_method.items():
                    tok_s = row.get("tokens_per_sec", 0.0)
                    parts.append(f"{method}={tok_s:.1f}t/s")
                print(f"DONE {tag}: " + "  ".join(parts), flush=True)
            except Exception as exc:
                print(f"FAILED {tag}: {exc}", flush=True)
                raise


@app.local_entrypoint()
def launch_phase1_reproduce(version: str = "v1", wait: bool = False) -> None:
    """Run the one-task reviewer gate against old positive task IDs."""

    methods = (
        "vanilla,blazedit_pld_w128_n10,"
        "vantage_dispatch_transpld_m4_w128_n10,"
        "rewrite_pld_bidir_w128_n10,"
        "vantage_transpld_w128_n10,"
        "vantage_fast_transpld_m4_w128_n10,"
        "vantage_compete_transpld_m4_margin0_w128_n10"
    )
    calls = []

    def spawn(name: str, problem_jsonl: str, task_id_file: str) -> None:
        tag = f"vantage_phase1_{name}_{version}"
        call = run_eagle_eval_job_any.spawn(
            run_tag=tag,
            target="Qwen/Qwen2.5-Coder-7B",
            target_trust_remote_code=False,
            n=50,
            max_new_tokens=256,
            methods=methods,
            problem_jsonl=problem_jsonl,
            dtype="bfloat16",
            attn_impl="sdpa",
            code_proposer_fallback="root",
            # Old no-m method aliases should reproduce the historical
            # permissive bidir configuration; m4 methods encode their own gate.
            transpld_min_match_len=1,
            chat_template="none",
            task_id_file=task_id_file,
        )
        print(f"{tag}\t{call.object_id}", flush=True)
        calls.append((tag, call))

    spawn(
        "qwen_field_single",
        "/root/asts-spec/data/manifests_phase3/drift_nonrename_field_rename.jsonl",
        "/root/asts-spec/data/task_ids/phase1_field_single.txt",
    )
    spawn(
        "qwen_style_single",
        "/root/asts-spec/data/manifests_phase3/drift_nonrename_style_rewrite.jsonl",
        "/root/asts-spec/data/task_ids/phase1_style_single.txt",
    )

    if wait:
        print("Waiting for spawned calls...", flush=True)
        for tag, call in calls:
            result = call.get()
            by_method = result.get("by_method", {})
            parts = [
                f"{method}={row.get('tokens_per_sec', 0.0):.1f}t/s"
                for method, row in by_method.items()
            ]
            print(f"DONE {tag}: " + "  ".join(parts), flush=True)


@app.local_entrypoint()
def launch_phase3_margin(version: str = "v1", wait: bool = False) -> None:
    """Sweep exact-vs-transformed candidate margin on Qwen-base style n=20."""

    methods = (
        "vanilla,blazedit_pld_w128_n10,"
        "vantage_compete_transpld_m4_margin0_w128_n10,"
        "vantage_compete_transpld_m4_margin8_w128_n10,"
        "vantage_compete_transpld_m4_margin16_w128_n10,"
        "vantage_compete_transpld_m4_margin32_w128_n10"
    )
    tag = f"vantage_phase3_qwen_style20_{version}"
    call = run_eagle_eval_job_any.spawn(
        run_tag=tag,
        target="Qwen/Qwen2.5-Coder-7B",
        target_trust_remote_code=False,
        n=20,
        max_new_tokens=256,
        methods=methods,
        problem_jsonl="/root/asts-spec/data/manifests_phase3/drift_nonrename_style_rewrite.jsonl",
        dtype="bfloat16",
        attn_impl="sdpa",
        code_proposer_fallback="root",
        transpld_min_match_len=4,
        chat_template="none",
    )
    print(f"{tag}\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        by_method = result.get("by_method", {})
        parts = [
            f"{method}={row.get('tokens_per_sec', 0.0):.1f}t/s"
            for method, row in by_method.items()
        ]
        print(f"DONE {tag}: " + "  ".join(parts), flush=True)


@app.local_entrypoint()
def launch_phase4_style(version: str = "v1", margin: int = 0, wait: bool = False) -> None:
    """Run Qwen-base style n=50 with the selected competition margin."""

    methods = (
        "vanilla,blazedit_pld_w128_n10,"
        f"vantage_compete_transpld_m4_margin{margin}_w128_n10"
    )
    tag = f"vantage_phase4_qwen_style50_margin{margin}_{version}"
    call = run_eagle_eval_job_any.spawn(
        run_tag=tag,
        target="Qwen/Qwen2.5-Coder-7B",
        target_trust_remote_code=False,
        n=50,
        max_new_tokens=256,
        methods=methods,
        problem_jsonl="/root/asts-spec/data/manifests_phase3/drift_nonrename_style_rewrite.jsonl",
        dtype="bfloat16",
        attn_impl="sdpa",
        code_proposer_fallback="root",
        transpld_min_match_len=4,
        chat_template="none",
    )
    print(f"{tag}\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        by_method = result.get("by_method", {})
        parts = [
            f"{method}={row.get('tokens_per_sec', 0.0):.1f}t/s"
            for method, row in by_method.items()
        ]
        print(f"DONE {tag}: " + "  ".join(parts), flush=True)


@app.local_entrypoint()
def launch_phase5_router_validation(version: str = "v1", margin: int = 0, wait: bool = False) -> None:
    """Run zero/field validation rows for the prompt-time router checks."""

    methods = (
        "vanilla,blazedit_pld_w128_n10,"
        f"vantage_compete_transpld_m4_margin{margin}_w128_n10"
    )
    calls = []

    def spawn(name: str, problem_jsonl: str, n: int = 50) -> None:
        tag = f"vantage_phase5_{name}_margin{margin}_{version}"
        call = run_eagle_eval_job_any.spawn(
            run_tag=tag,
            target="Qwen/Qwen2.5-Coder-7B",
            target_trust_remote_code=False,
            n=n,
            max_new_tokens=256,
            methods=methods,
            problem_jsonl=problem_jsonl,
            dtype="bfloat16",
            attn_impl="sdpa",
            code_proposer_fallback="root",
            transpld_min_match_len=4,
            chat_template="none",
        )
        print(f"{tag}\t{call.object_id}", flush=True)
        calls.append((tag, call))

    spawn(
        "qwen_field50",
        "/root/asts-spec/data/manifests_phase3/drift_nonrename_field_rename.jsonl",
    )
    spawn(
        "qwen_zero50",
        "/root/asts-spec/data/manifests_transpld_ext/drift_renamepct_0.jsonl",
    )

    if wait:
        print("Waiting for spawned calls...", flush=True)
        for tag, call in calls:
            result = call.get()
            by_method = result.get("by_method", {})
            parts = [
                f"{method}={row.get('tokens_per_sec', 0.0):.1f}t/s"
                for method, row in by_method.items()
            ]
            print(f"DONE {tag}: " + "  ".join(parts), flush=True)


@app.local_entrypoint()
def launch_frozen_lossless(version: str = "v1", wait: bool = False) -> None:
    """Run the blocking fp32/eager byte-identity audit for the frozen method."""

    methods = "blazedit_pld_w128_n10,vantage_frozen_transpld"
    calls = []

    def spawn(name: str, problem_jsonl: str) -> None:
        tag = f"vantage_frozen_lossless_{name}_{version}"
        call = run_lossless_job.spawn(
            run_tag=tag,
            target="Qwen/Qwen2.5-Coder-7B",
            target_trust_remote_code=False,
            n=100,
            max_new_tokens=256,
            code_methods=methods,
            problem_jsonl=problem_jsonl,
            dtype="float32",
            attn_impl="eager",
        )
        print(f"{tag}\t{call.object_id}", flush=True)
        calls.append((tag, call))

    spawn("field100", "/root/asts-spec/data/manifests_frozen_audit/field_rename100.jsonl")
    spawn("style100", "/root/asts-spec/data/manifests_frozen_audit/style_rewrite100.jsonl")
    spawn("zero100", "/root/asts-spec/data/manifests_frozen_audit/zero_drift100.jsonl")

    if wait:
        print("Waiting for lossless jobs...", flush=True)
        for tag, call in calls:
            result = call.get()
            print(
                f"DONE {tag}: all_match={result.get('all_match')} "
                f"n_tasks={result.get('n_tasks')} "
                f"n_match_code={result.get('n_match_code')}",
                flush=True,
            )


@app.local_entrypoint()
def launch_frozen_instruct_retest(version: str = "v1", wait: bool = False) -> None:
    """Retest Instruct models with the fixed fast/compete VANTAGE paths."""

    print(
        "Predictions before running: if decode-loop overhead was the main issue, "
        "Instruct field/style rows should move from roughly 0.85x to >=1.05x vs PLD; "
        "zero drift should remain close to PLD because exact recurrence dominates.",
        flush=True,
    )
    methods = (
        "vanilla,blazedit_pld_w128_n10,"
        "vantage_fast_transpld_m4_w128_n10,"
        "vantage_compete_transpld_m4_margin0_w128_n10"
    )
    calls = []

    def spawn(
        name: str,
        *,
        target: str,
        problem_jsonl: str,
        chat_template: str = "user",
        trust_remote_code: bool = False,
    ) -> None:
        tag = f"vantage_frozen_instruct_{name}_{version}"
        call = run_eagle_eval_job_any.spawn(
            run_tag=tag,
            target=target,
            target_trust_remote_code=trust_remote_code,
            n=50,
            max_new_tokens=256,
            methods=methods,
            problem_jsonl=problem_jsonl,
            dtype="bfloat16",
            attn_impl="sdpa",
            code_proposer_fallback="root",
            transpld_min_match_len=4,
            chat_template=chat_template,
        )
        print(f"{tag}\t{call.object_id}", flush=True)
        calls.append((tag, call))

    field = "/root/asts-spec/data/manifests_phase3/drift_nonrename_field_rename.jsonl"
    style = "/root/asts-spec/data/manifests_phase3/drift_nonrename_style_rewrite.jsonl"
    zero = "/root/asts-spec/data/manifests_transpld_ext/drift_renamepct_0.jsonl"

    for workload, manifest in (("field50", field), ("style50", style), ("zero50", zero)):
        spawn(
            f"qwen_instruct_{workload}",
            target="Qwen/Qwen2.5-Coder-7B-Instruct",
            problem_jsonl=manifest,
        )
        spawn(
            f"deepseek_instruct_{workload}",
            target="deepseek-ai/deepseek-coder-6.7b-instruct",
            problem_jsonl=manifest,
            trust_remote_code=True,
        )

    if wait:
        print("Waiting for Instruct retest jobs...", flush=True)
        for tag, call in calls:
            result = call.get()
            by_method = result.get("by_method", {})
            parts = [
                f"{method}={row.get('tokens_per_sec', 0.0):.1f}t/s"
                for method, row in by_method.items()
            ]
            print(f"DONE {tag}: " + "  ".join(parts), flush=True)


@app.local_entrypoint()
def launch_frozen_final_validation(
    version: str = "v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    chat_template: str = "none",
    trust_remote_code: bool = False,
    model_label: str = "qwen_base",
    wait: bool = False,
) -> None:
    """Run n=100 final validation rows after the lossless/Instruct gates pass."""

    methods = "vanilla,blazedit_pld_w128_n10,vantage_frozen_transpld"
    calls = []

    def spawn(name: str, problem_jsonl: str) -> None:
        tag = f"vantage_frozen_final_{model_label}_{name}_{version}"
        call = run_eagle_eval_job_any.spawn(
            run_tag=tag,
            target=target,
            target_trust_remote_code=trust_remote_code,
            n=100,
            max_new_tokens=256,
            methods=methods,
            problem_jsonl=problem_jsonl,
            dtype="bfloat16",
            attn_impl="sdpa",
            code_proposer_fallback="root",
            transpld_min_match_len=4,
            chat_template=chat_template,
        )
        print(f"{tag}\t{call.object_id}", flush=True)
        calls.append((tag, call))

    spawn("zero100", "/root/asts-spec/data/manifests_frozen_audit/zero_drift100.jsonl")
    spawn("field100", "/root/asts-spec/data/manifests_frozen_audit/field_rename100.jsonl")
    spawn("style100", "/root/asts-spec/data/manifests_frozen_audit/style_rewrite100.jsonl")
    spawn("mixed100", "/root/asts-spec/data/manifests_frozen_audit/mixed_zero_field_style100.jsonl")

    if wait:
        print("Waiting for final validation jobs...", flush=True)
        for tag, call in calls:
            result = call.get()
            by_method = result.get("by_method", {})
            parts = [
                f"{method}={row.get('tokens_per_sec', 0.0):.1f}t/s"
                for method, row in by_method.items()
            ]
            print(f"DONE {tag}: " + "  ".join(parts), flush=True)


@app.local_entrypoint()
def launch_transpld_same_policy_ablation(
    version: str = "v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    chat_template: str = "none",
    trust_remote_code: bool = False,
    model_label: str = "qwen_base",
    n: int = 100,
    dtype: str = "bfloat16",
    attn_impl: str = "sdpa",
    wait: bool = False,
) -> None:
    """Run reviewer-requested same-match-policy PLD/rewrite-view ablations."""

    methods = (
        "vanilla,"
        "blazedit_pld_w128_n10,"
        "blazedit_pld_m4_w128_n10,"
        "blazedit_pld_m10_w128_n10,"
        "vantage_fast_transpld_m4_w128_n10,"
        "vantage_fast_transpld_m10_w128_n10,"
        "vantage_compete_transpld_m4_exactm4_margin0_w128_n10,"
        "vantage_frozen_transpld"
    )
    calls = []

    def spawn(name: str, problem_jsonl: str) -> None:
        tag = f"vantage_same_policy_{model_label}_{name}_{version}"
        call = run_eagle_eval_job_any.spawn(
            run_tag=tag,
            target=target,
            target_trust_remote_code=trust_remote_code,
            n=n,
            max_new_tokens=256,
            methods=methods,
            problem_jsonl=problem_jsonl,
            dtype=dtype,
            attn_impl=attn_impl,
            code_proposer_fallback="root",
            transpld_min_match_len=4,
            chat_template=chat_template,
        )
        print(f"{tag}\t{call.object_id}", flush=True)
        calls.append((tag, call))

    spawn("zero100", "/root/asts-spec/data/manifests_frozen_audit/zero_drift100.jsonl")
    spawn("field100", "/root/asts-spec/data/manifests_frozen_audit/field_rename100.jsonl")
    spawn("style100", "/root/asts-spec/data/manifests_frozen_audit/style_rewrite100.jsonl")
    spawn("mixed100", "/root/asts-spec/data/manifests_frozen_audit/mixed_zero_field_style100.jsonl")

    if wait:
        print("Waiting for same-policy ablation jobs...", flush=True)
        for tag, call in calls:
            result = call.get()
            by_method = result.get("by_method", {})
            pld = by_method.get("blazedit_pld_m4_w128_n10", {}).get("tokens_per_sec", 0.0) or 1.0
            parts = [
                f"{method}={row.get('tokens_per_sec', 0.0):.1f}t/s({row.get('tokens_per_sec', 0.0) / pld:.3f}x/m4)"
                for method, row in by_method.items()
            ]
            print(f"DONE {tag}: " + "  ".join(parts), flush=True)


@app.local_entrypoint()
def launch_transpld_route_margin_ablation(
    version: str = "v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    chat_template: str = "none",
    trust_remote_code: bool = False,
    model_label: str = "qwen_base",
    n: int = 100,
    wait: bool = False,
) -> None:
    """Run reviewer-requested prompt-only VANTAGE route-margin ablations."""

    methods = (
        "vanilla,"
        "blazedit_pld_w128_n10,"
        "vantage_compete_transpld_m4_margin0_w128_n10,"
        "vantage_compete_transpld_m4_margin16_w128_n10,"
        "vantage_compete_transpld_m4_margin32_w128_n10,"
        "vantage_frozen_transpld"
    )
    calls = []

    def spawn(name: str, problem_jsonl: str) -> None:
        tag = f"vantage_route_margin_{model_label}_{name}_{version}"
        call = run_eagle_eval_job_any.spawn(
            run_tag=tag,
            target=target,
            target_trust_remote_code=trust_remote_code,
            n=n,
            max_new_tokens=256,
            methods=methods,
            problem_jsonl=problem_jsonl,
            dtype="bfloat16",
            attn_impl="sdpa",
            code_proposer_fallback="root",
            transpld_min_match_len=4,
            chat_template=chat_template,
        )
        print(f"{tag}\t{call.object_id}", flush=True)
        calls.append((tag, call))

    spawn("zero100", "/root/asts-spec/data/manifests_frozen_audit/zero_drift100.jsonl")
    spawn("field100", "/root/asts-spec/data/manifests_frozen_audit/field_rename100.jsonl")
    spawn("style100", "/root/asts-spec/data/manifests_frozen_audit/style_rewrite100.jsonl")
    spawn("mixed100", "/root/asts-spec/data/manifests_frozen_audit/mixed_zero_field_style100.jsonl")

    if wait:
        print("Waiting for route-margin ablation jobs...", flush=True)
        for tag, call in calls:
            result = call.get()
            by_method = result.get("by_method", {})
            pld = by_method.get("blazedit_pld_w128_n10", {}).get("tokens_per_sec", 0.0) or 1.0
            parts = [
                f"{method}={row.get('tokens_per_sec', 0.0):.1f}t/s({row.get('tokens_per_sec', 0.0) / pld:.3f}x/pld)"
                for method, row in by_method.items()
            ]
            print(f"DONE {tag}: " + "  ".join(parts), flush=True)


@app.local_entrypoint()
def launch_transpld_backend_isolation_validation(
    version: str = "v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    chat_template: str = "none",
    trust_remote_code: bool = False,
    model_label: str = "qwen_base",
    n: int = 100,
    backends: str = "fp32/eager,bf16/eager,fp32/sdpa,bf16/sdpa",
    wait: bool = False,
) -> None:
    """Run backend/dtype parity-isolation rows for VANTAGE validation."""

    methods = "vanilla,blazedit_pld_w128_n10,vantage_frozen_transpld"
    calls = []
    dtype_map = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}
    workloads = [
        ("zero100", "/root/asts-spec/data/manifests_frozen_audit/zero_drift100.jsonl"),
        ("field100", "/root/asts-spec/data/manifests_frozen_audit/field_rename100.jsonl"),
        ("style100", "/root/asts-spec/data/manifests_frozen_audit/style_rewrite100.jsonl"),
        ("mixed100", "/root/asts-spec/data/manifests_frozen_audit/mixed_zero_field_style100.jsonl"),
    ]

    for backend in [b.strip() for b in backends.split(",") if b.strip()]:
        if "/" not in backend:
            raise ValueError("backend entries must look like fp32/eager or bf16/sdpa")
        dtype_key, attn_impl = backend.split("/", 1)
        eval_dtype = dtype_map.get(dtype_key, dtype_key)
        safe_backend = backend.replace("/", "_")
        for name, problem_jsonl in workloads:
            tag = f"vantage_backend_iso_{model_label}_{safe_backend}_{name}_{version}"
            call = run_eagle_eval_job_any.spawn(
                run_tag=tag,
                target=target,
                target_trust_remote_code=trust_remote_code,
                n=n,
                max_new_tokens=256,
                methods=methods,
                problem_jsonl=problem_jsonl,
                dtype=eval_dtype,
                attn_impl=attn_impl,
                code_proposer_fallback="root",
                transpld_min_match_len=4,
                chat_template=chat_template,
            )
            print(f"{tag}\t{call.object_id}", flush=True)
            calls.append((tag, call))

    if wait:
        print("Waiting for backend-isolation jobs...", flush=True)
        for tag, call in calls:
            result = call.get()
            equivalence = result.get("output_equivalence", {})
            parts = []
            for method, row in equivalence.items():
                parts.append(f"{method}={row.get('matches_vanilla')}/{row.get('tasks')}")
            print(f"DONE {tag}: " + "  ".join(parts), flush=True)


@app.local_entrypoint()
def launch_prompt_injection_validation(
    version: str = "v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    chat_template: str = "none",
    trust_remote_code: bool = False,
    model_label: str = "qwen_base",
    n: int = 100,
    wait: bool = False,
) -> None:
    """Run the changed-prompt visible transformed-reference baseline.

    This is not VANTAGE/SafeRoute. It inserts the transformed reference into
    the visible prompt and then evaluates vanilla plus exact PLD so reviewers
    can see the practical prompt-modification alternative.
    """

    methods = "vanilla,blazedit_pld_w128_n10"
    calls = []

    def spawn(name: str, problem_jsonl: str) -> None:
        tag = f"vantage_prompt_inject_{model_label}_{name}_{version}"
        call = run_eagle_eval_job_any.spawn(
            run_tag=tag,
            target=target,
            target_trust_remote_code=trust_remote_code,
            n=n,
            max_new_tokens=256,
            methods=methods,
            problem_jsonl=problem_jsonl,
            dtype="bfloat16",
            attn_impl="sdpa",
            code_proposer_fallback="root",
            transpld_min_match_len=4,
            chat_template=chat_template,
        )
        print(f"{tag}\t{call.object_id}", flush=True)
        calls.append((tag, call))

    spawn(
        "zero100",
        "/root/asts-spec/data/manifests_prompt_injection/zero_drift100_visible_transformed_reference.jsonl",
    )
    spawn(
        "field100",
        "/root/asts-spec/data/manifests_prompt_injection/field_rename100_visible_transformed_reference.jsonl",
    )
    spawn(
        "style100",
        "/root/asts-spec/data/manifests_prompt_injection/style_rewrite100_visible_transformed_reference.jsonl",
    )
    spawn(
        "mixed100",
        "/root/asts-spec/data/manifests_prompt_injection/mixed100_visible_transformed_reference.jsonl",
    )

    if wait:
        print("Waiting for prompt-injection validation jobs...", flush=True)
        for tag, call in calls:
            result = call.get()
            by_method = result.get("by_method", {})
            parts = [
                f"{method}={row.get('tokens_per_sec', 0.0):.1f}t/s"
                for method, row in by_method.items()
            ]
            print(f"DONE {tag}: " + "  ".join(parts), flush=True)


@app.local_entrypoint()
def launch_path_a_real_commit_sweep(
    split: str = "train",
    version: str = "v1",
    wait: bool = False,
    target: str = "Qwen/Qwen2.5-Coder-7B",
) -> None:
    """Launch the Path-A lazy rewrite-view real-commit train/test sweep."""

    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    manifest = f"/root/asts-spec/data/real_commits/path_a_{split}500_v1.jsonl"
    default_methods = (
        "blazedit_pld_w128_n10,"
        "vantage_lazy_transpld_s16_m16_z1_w128_n10,"
        "vantage_lazy_transpld_s16_m32_z1_w128_n10,"
        "vantage_lazy_transpld_s16_m64_z1_w128_n10,"
        "vantage_lazy_transpld_s32_m16_z1_w128_n10,"
        "vantage_lazy_transpld_s32_m32_z1_w128_n10,"
        "vantage_lazy_transpld_s32_m64_z1_w128_n10,"
        "vantage_lazy_transpld_s64_m16_z1_w128_n10,"
        "vantage_lazy_transpld_s64_m32_z1_w128_n10,"
        "vantage_lazy_transpld_s64_m64_z1_w128_n10"
    )
    tag = f"vantage_real_commit_path_a_{split}500_{version}"
    call = run_eagle_eval_job_any.spawn(
        run_tag=tag,
        target=target,
        n=500,
        max_new_tokens=256,
        methods=default_methods,
        problem_jsonl=manifest,
        dtype="bfloat16",
        attn_impl="sdpa",
        code_proposer_fallback="root",
        transpld_min_match_len=4,
    )
    print(f"{tag}\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        by_method = result.get("by_method", {})
        parts = [
            f"{method}={row.get('tokens_per_sec', 0.0):.1f}t/s"
            for method, row in by_method.items()
        ]
        print(f"DONE {tag}: " + "  ".join(parts), flush=True)


@app.local_entrypoint()
def launch_path_a_real_commit_final(
    method: str,
    version: str = "v1",
    wait: bool = False,
    target: str = "Qwen/Qwen2.5-Coder-7B",
) -> None:
    """Launch held-out Path-A evaluation with PLD and one selected method."""

    tag = f"vantage_real_commit_path_a_test500_{version}"
    methods = f"blazedit_pld_w128_n10,{method}"
    call = run_eagle_eval_job_any.spawn(
        run_tag=tag,
        target=target,
        n=500,
        max_new_tokens=256,
        methods=methods,
        problem_jsonl="/root/asts-spec/data/real_commits/path_a_test500_v1.jsonl",
        dtype="bfloat16",
        attn_impl="sdpa",
        code_proposer_fallback="root",
        transpld_min_match_len=4,
    )
    print(f"{tag}\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        by_method = result.get("by_method", {})
        parts = [
            f"{name}={row.get('tokens_per_sec', 0.0):.1f}t/s"
            for name, row in by_method.items()
        ]
        print(f"DONE {tag}: " + "  ".join(parts), flush=True)


@app.local_entrypoint()
def launch_multiview_real_commit_test(
    version: str = "v1",
    wait: bool = False,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    task_router_json: str = "",
    task_router_exact_strong_threshold: int = 32,
    task_router_trans_margin: int = 0,
) -> None:
    """Launch held-out real-commit MultiView and task-router rows."""

    tag = f"vantage_real_commit_multiview_test500_{version}"
    methods = (
        "blazedit_pld_w128_n10,"
        "vantage_mvpld_s32_m0_w128_n10,"
        "vantage_mvtree_s32_m0_w128_n10"
    )
    if task_router_json:
        methods += ",vantage_task_router_mvpld_w128_n10,vantage_task_router_mvtree_w128_n10"
    call = run_eagle_eval_job_any.spawn(
        run_tag=tag,
        target=target,
        n=500,
        max_new_tokens=256,
        methods=methods,
        problem_jsonl="/root/asts-spec/data/real_commits/path_a_test500_v1.jsonl",
        dtype="bfloat16",
        attn_impl="sdpa",
        code_proposer_fallback="root",
        transpld_min_match_len=4,
        task_router_json=task_router_json,
        task_router_exact_strong_threshold=task_router_exact_strong_threshold,
        task_router_trans_margin=task_router_trans_margin,
    )
    print(f"{tag}\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        by_method = result.get("by_method", {})
        parts = [
            f"{name}={row.get('tokens_per_sec', 0.0):.1f}t/s"
            for name, row in by_method.items()
        ]
        print(f"DONE {tag}: " + "  ".join(parts), flush=True)


@app.local_entrypoint()
def launch_real_commit_force_pld_test(
    version: str = "v1",
    wait: bool = False,
    target: str = "Qwen/Qwen2.5-Coder-7B",
) -> None:
    """Launch the blocking forced-PLD passthrough equivalence test."""

    tag = f"vantage_real_commit_force_pld_test500_{version}"
    methods = "blazedit_pld_w128_n10,vantage_force_pld_w128_n10"
    call = run_eagle_eval_job_any.spawn(
        run_tag=tag,
        target=target,
        n=500,
        max_new_tokens=256,
        methods=methods,
        problem_jsonl="/root/asts-spec/data/real_commits/path_a_test500_v1.jsonl",
        dtype="bfloat16",
        attn_impl="sdpa",
        code_proposer_fallback="root",
        transpld_min_match_len=4,
    )
    print(f"{tag}\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        by_method = result.get("by_method", {})
        parts = [
            f"{name}={row.get('tokens_per_sec', 0.0):.1f}t/s"
            for name, row in by_method.items()
        ]
        print(f"DONE {tag}: " + "  ".join(parts), flush=True)


@app.local_entrypoint()
def launch_real_commit_vantage_mv_decoder_test(
    version: str = "v1",
    wait: bool = False,
    target: str = "Qwen/Qwen2.5-Coder-7B",
) -> None:
    """Launch the BlazEdit-derived VANTAGE-MV decoder held-out test."""

    tag = f"vantage_real_commit_mv_decoder_test500_{version}"
    methods = ",".join(
        [
            "blazedit_pld_w128_n10",
            "vantage_force_pld_w128_n10",
            REAL_COMMIT_MV_STABLE_METHOD,
        ]
    )
    call = run_eagle_eval_job_any.spawn(
        run_tag=tag,
        target=target,
        n=500,
        max_new_tokens=256,
        methods=methods,
        problem_jsonl="/root/asts-spec/data/real_commits/path_a_test500_v1.jsonl",
        dtype="bfloat16",
        attn_impl="sdpa",
        code_proposer_fallback="root",
        transpld_min_match_len=4,
    )
    print(f"{tag}\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        by_method = result.get("by_method", {})
        parts = [
            f"{name}={row.get('tokens_per_sec', 0.0):.1f}t/s"
            for name, row in by_method.items()
        ]
        print(f"DONE {tag}: " + "  ".join(parts), flush=True)


@app.local_entrypoint()
def launch_real_commit_mv_decoder_train_relaxed_grid(
    version: str = "v1",
    wait: bool = False,
    target: str = "Qwen/Qwen2.5-Coder-7B",
) -> None:
    """Train-only relaxed MV gate sweep.

    This intentionally uses the train500 split only.  The held-out test split
    is reserved until one configuration is selected by weighted train tok/s
    and route/proposal diagnostics.
    """

    methods = ["blazedit_pld_w128_n10", "vantage_force_pld_w128_n10"]
    for strong in (16, 32, 64):
        for margin in (0, 4, 8, 16):
            for frontier in (2, 4, 8):
                methods.append(
                    f"vantage_mv_pld_s{strong}_m{margin}_f{frontier}_t8_w128_n10"
                )
    tag = f"vantage_real_commit_mv_decoder_train_relaxed_grid500_{version}"
    call = run_eagle_eval_job_any.spawn(
        run_tag=tag,
        target=target,
        n=500,
        max_new_tokens=256,
        methods=",".join(methods),
        problem_jsonl="/root/asts-spec/data/real_commits/path_a_train500_v1.jsonl",
        dtype="bfloat16",
        attn_impl="sdpa",
        code_proposer_fallback="root",
        transpld_min_match_len=4,
    )
    print(f"{tag}\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        by_method = result.get("by_method", {})
        parts = [
            f"{name}={row.get('tokens_per_sec', 0.0):.1f}t/s"
            for name, row in by_method.items()
        ]
        print(f"DONE {tag}: " + "  ".join(parts), flush=True)


@app.local_entrypoint()
def launch_real_commit_mv_decoder_train_guard_cap_branch_grid(
    version: str = "v1",
    wait: bool = False,
    target: str = "Qwen/Qwen2.5-Coder-7B",
) -> None:
    """Train-only grid for exact-match guards, adaptive exact caps, and branch probes."""

    methods = [
        "blazedit_pld_w128_n10",
        "vantage_force_pld_w128_n10",
        "vantage_mv_pld_s64_m16_t8_w128_n10",
    ]

    # Guard sweep: length-only strong PLD is not enough; vary the exact PLD
    # match-quality requirement before transformed views are skipped.
    for strong in (64, 96):
        for exact_match in (1, 4, 8, 10):
            methods.append(
                f"vantage_mv_pld_s{strong}_x{exact_match}_m16_t8_w128_n10"
            )

    # Adaptive exact-PLD draft caps for weak matches.  These reduce the cost of
    # long drafts from short exact matches and should expose more transformed
    # frontier opportunities without changing the PLD substrate.
    for cap1 in (8, 16, 32):
        for cap7 in (32, 64):
            methods.append(
                f"vantage_mv_pld_s64_x1_c1{cap1}_c7{cap7}_m16_t8_w128_n10"
            )

    # Conflict-only branch prototypes.  Branch logic is enabled only when exact
    # PLD has a weak match and the transformed candidate has a strong frontier
    # match; these rows test whether branch-like arbitration captures the
    # exact-strong shadow-oracle bucket without broad regressions.
    for strong in (64, 96):
        methods.append(
            f"vantage_mv_pld_branch_s{strong}_x8_m16_t8_w128_n10"
        )
        methods.append(
            f"vantage_mv_pld_branch_s{strong}_x8_c116_c764_m16_t8_w128_n10"
        )

    # Deduplicate while preserving order because x1 rows overlap defaults.
    methods = list(dict.fromkeys(methods))

    tag = f"vantage_real_commit_mv_decoder_train_guard_cap_branch_grid500_{version}"
    call = run_eagle_eval_job_any.spawn(
        run_tag=tag,
        target=target,
        n=500,
        max_new_tokens=256,
        methods=",".join(methods),
        problem_jsonl="/root/asts-spec/data/real_commits/path_a_train500_v1.jsonl",
        dtype="bfloat16",
        attn_impl="sdpa",
        code_proposer_fallback="root",
        transpld_min_match_len=4,
    )
    print(f"{tag}\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        by_method = result.get("by_method", {})
        parts = [
            f"{name}={row.get('tokens_per_sec', 0.0):.1f}t/s"
            for name, row in by_method.items()
        ]
        print(f"DONE {tag}: " + "  ".join(parts), flush=True)


@app.local_entrypoint()
def launch_real_commit_mv_decoder_heldout_safety_check(
    version: str = "v1",
    wait: bool = False,
    target: str = "Qwen/Qwen2.5-Coder-7B",
) -> None:
    """Held-out safety check for the robust train-selected MV guard row."""

    tag = f"vantage_real_commit_mv_decoder_heldout_safety500_{version}"
    methods = ",".join(
        [
            "blazedit_pld_w128_n10",
            "vantage_force_pld_w128_n10",
            REAL_COMMIT_MV_PREVIOUS_METHOD,
            REAL_COMMIT_MV_STABLE_METHOD,
        ]
    )
    call = run_eagle_eval_job_any.spawn(
        run_tag=tag,
        target=target,
        n=500,
        max_new_tokens=256,
        methods=methods,
        problem_jsonl="/root/asts-spec/data/real_commits/path_a_test500_v1.jsonl",
        dtype="bfloat16",
        attn_impl="sdpa",
        code_proposer_fallback="root",
        transpld_min_match_len=4,
    )
    print(f"{tag}\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        by_method = result.get("by_method", {})
        parts = [
            f"{name}={row.get('tokens_per_sec', 0.0):.1f}t/s"
            for name, row in by_method.items()
        ]
        print(f"DONE {tag}: " + "  ".join(parts), flush=True)


@app.local_entrypoint()
def launch_real_commit_mv_decoder_branch_proto_train(
    version: str = "v1",
    wait: bool = False,
    target: str = "Qwen/Qwen2.5-Coder-7B",
) -> None:
    """Focused train run for the true frontier-branch verifier prototype."""

    tag = f"vantage_real_commit_mv_decoder_branch_proto_train500_{version}"
    methods = (
        "blazedit_pld_w128_n10,"
        "vantage_mv_pld_s64_m16_t8_w128_n10,"
        "vantage_mv_pld_s96_x1_m16_t8_w128_n10,"
        "vantage_mv_pld_branch_s96_x8_m16_t8_w128_n10"
    )
    call = run_eagle_eval_job_any.spawn(
        run_tag=tag,
        target=target,
        n=500,
        max_new_tokens=256,
        methods=methods,
        problem_jsonl="/root/asts-spec/data/real_commits/path_a_train500_v1.jsonl",
        dtype="bfloat16",
        attn_impl="sdpa",
        code_proposer_fallback="root",
        transpld_min_match_len=4,
    )
    print(f"{tag}\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        by_method = result.get("by_method", {})
        parts = [
            f"{name}={row.get('tokens_per_sec', 0.0):.1f}t/s"
            for name, row in by_method.items()
        ]
        print(f"DONE {tag}: " + "  ".join(parts), flush=True)


@app.local_entrypoint()
def launch_real_commit_mv_treecursor_test(
    split: str = "test",
    version: str = "v1",
    n: int = 500,
    wait: bool = False,
    target: str = "Qwen/Qwen2.5-Coder-7B",
) -> None:
    """Run TreeCursor frontier mechanisms against PLD and stable MV.

    Rows isolate true tree verification, transformed-reference cursor,
    generated-prefix reindexing, and hunk-aware line alignment before testing
    the combined FST/pair-prior TreeCursor stack.
    """

    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    tag = f"vantage_real_commit_mv_treecursor_{split}{n}_{version}"
    methods = ",".join(
        [
            "blazedit_pld_w128_n10",
            "vantage_force_pld_w128_n10",
            REAL_COMMIT_MV_STABLE_METHOD,
            "vantage_mv_pld_rescue_s96_x1_m16_t8_w128_n10",
            "vantage_mv_pld_tree_rescue_s96_x1_m16_t8_w128_n10",
            "vantage_mv_pld_patch_s96_x1_m16_t8_w128_n10",
            "vantage_mv_pld_rescue_patch_tree_s96_x1_m16_t8_w128_n10",
            "vantage_mv_pld_tree_s96_x1_m16_t8_w128_n10",
            "vantage_mv_pld_cursor_s96_x1_m16_t8_w128_n10",
            "vantage_mv_pld_s96_x1_m16_t8_g32_w128_n10",
            "vantage_mv_pld_hunk_s96_x1_m16_t8_w128_n10",
            "vantage_mv_pld_fst_pair_tree_cursor_hunk_s96_x1_m16_t8_g32_w128_n10",
        ]
    )
    call = run_eagle_eval_job_any.spawn(
        run_tag=tag,
        target=target,
        n=n,
        max_new_tokens=256,
        methods=methods,
        problem_jsonl=f"/root/asts-spec/data/real_commits/path_a_{split}500_v1.jsonl",
        dtype="bfloat16",
        attn_impl="sdpa",
        code_proposer_fallback="root",
        transpld_min_match_len=4,
    )
    print(f"{tag}\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        by_method = result.get("by_method", {})
        parts = [
            f"{name}={row.get('tokens_per_sec', 0.0):.1f}t/s"
            for name, row in by_method.items()
        ]
        print(f"DONE {tag}: " + "  ".join(parts), flush=True)


@app.local_entrypoint()
def launch_real_commit_balanced1000_mv_test(
    version: str = "v1",
    n: int = 1000,
    wait: bool = False,
    target: str = "Qwen/Qwen2.5-Coder-7B",
) -> None:
    """Run frozen VANTAGE and frontier MV variants on the balanced 1000 commits."""

    tag = f"vantage_real_commit_balanced1000_mv_{n}_{version}"
    methods = ",".join(
        [
            "blazedit_pld_w128_n10",
            "vantage_force_pld_w128_n10",
            "vantage_frozen_transpld",
            REAL_COMMIT_MV_STABLE_METHOD,
            "vantage_mv_pld_rescue_s96_x1_m16_t8_w128_n10",
            "vantage_mv_pld_patch_s96_x1_m16_t8_w128_n10",
            "vantage_mv_pld_cursor_s96_x1_m16_t8_w128_n10",
            "vantage_mv_pld_s96_x1_m16_t8_g32_w128_n10",
            "vantage_mv_pld_hunk_s96_x1_m16_t8_w128_n10",
        ]
    )
    call = run_eagle_eval_job_any.spawn(
        run_tag=tag,
        target=target,
        n=n,
        max_new_tokens=256,
        methods=methods,
        problem_jsonl="/root/asts-spec/data/real_commits/real_commit_manifest_balanced_1000_v2.jsonl",
        dtype="bfloat16",
        attn_impl="sdpa",
        code_proposer_fallback="root",
        transpld_min_match_len=4,
    )
    print(f"{tag}\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        by_method = result.get("by_method", {})
        parts = [
            f"{name}={row.get('tokens_per_sec', 0.0):.1f}t/s"
            for name, row in by_method.items()
        ]
        print(f"DONE {tag}: " + "  ".join(parts), flush=True)


@app.local_entrypoint()
def launch_real_commit_frontier20_smoke(
    version: str = "v1",
    n: int = 50,
    wait: bool = False,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    split: str = "test",
) -> None:
    """Smoke frontier verifier/drafter variants on the balanced real commits.

    This is intentionally small because edit-aware neural drafting loads the
    assistant model.  Promote only the rows that pass this smoke to a full
    held-out run.
    """

    if split not in {"train", "test", "balanced"}:
        raise ValueError("split must be 'train', 'test', or 'balanced'")
    if split == "balanced":
        problem_jsonl = "/root/asts-spec/data/real_commits/real_commit_manifest_balanced_1000_v2.jsonl"
    else:
        problem_jsonl = f"/root/asts-spec/data/real_commits/path_a_{split}500_v1.jsonl"

    tag = f"vantage_real_commit_frontier20_{split}{n}_{version}"
    methods = ",".join(
        [
            "blazedit_pld_w128_n10",
            "vantage_force_pld_w128_n10",
            "vantage_staged_pld_v16_32_w128_n10",
            REAL_COMMIT_MV_STABLE_METHOD,
            "vantage_mv_pld_stage_patch_s96_x1_m16_t8_w128_n10",
            "vantage_mv_pld_pbranch_patch_s96_x1_m16_t8_w128_n10",
            "vantage_mv_pld_stage_pbranch_patch_s96_x1_m16_t8_w128_n10",
            "vantage_mv_pld_edraft_patch_s96_x1_m16_t8_w128_n10",
            "vantage_mv_pld_stage_pbranch_edraft_patch_s96_x1_m16_t8_w128_n10",
        ]
    )
    call = run_eagle_eval_job_any.spawn(
        run_tag=tag,
        target=target,
        n=n,
        max_new_tokens=256,
        methods=methods,
        problem_jsonl=problem_jsonl,
        dtype="bfloat16",
        attn_impl="sdpa",
        code_proposer_fallback="root",
        transpld_min_match_len=4,
    )
    print(f"{tag}\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        by_method = result.get("by_method", {})
        parts = [
            f"{name}={row.get('tokens_per_sec', 0.0):.1f}t/s"
            for name, row in by_method.items()
        ]
        print(f"DONE {tag}: " + "  ".join(parts), flush=True)


@app.local_entrypoint()
def launch_real_commit_balanced_selector_test(
    version: str = "v1",
    n: int = 500,
    wait: bool = False,
    target: str = "Qwen/Qwen2.5-Coder-7B",
) -> None:
    """Held-out balanced-test validation for the train-fitted MV/frozen selector."""

    tag = f"vantage_real_commit_balanced_selector_test{n}_{version}"
    methods = ",".join(
        [
            "blazedit_pld_w128_n10",
            "vantage_force_pld_w128_n10",
            "vantage_selected_mv_frozen_w128_n10",
        ]
    )
    call = run_eagle_eval_job_any.spawn(
        run_tag=tag,
        target=target,
        n=n,
        max_new_tokens=256,
        methods=methods,
        problem_jsonl="/root/asts-spec/data/real_commits/real_commit_manifest_balanced_1000_v2_test500.jsonl",
        dtype="bfloat16",
        attn_impl="sdpa",
        code_proposer_fallback="root",
        transpld_min_match_len=4,
        task_router_json="/root/asts-spec/data/routers/balanced1000_mv_frozen_selector_v1.json",
    )
    print(f"{tag}\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        by_method = result.get("by_method", {})
        parts = [
            f"{name}={row.get('tokens_per_sec', 0.0):.1f}t/s"
            for name, row in by_method.items()
        ]
        print(f"DONE {tag}: " + "  ".join(parts), flush=True)


@app.local_entrypoint()
def launch_real_commit_mv_decoder_selected_test(
    version: str = "v1",
    wait: bool = False,
    target: str = "Qwen/Qwen2.5-Coder-7B",
) -> None:
    """Held-out test for the promoted stable real-commit MV config."""

    tag = f"vantage_real_commit_mv_decoder_selected_test500_{version}"
    methods = ",".join(
        [
            "blazedit_pld_w128_n10",
            "vantage_force_pld_w128_n10",
            REAL_COMMIT_MV_STABLE_METHOD,
        ]
    )
    call = run_eagle_eval_job_any.spawn(
        run_tag=tag,
        target=target,
        n=500,
        max_new_tokens=256,
        methods=methods,
        problem_jsonl="/root/asts-spec/data/real_commits/path_a_test500_v1.jsonl",
        dtype="bfloat16",
        attn_impl="sdpa",
        code_proposer_fallback="root",
        transpld_min_match_len=4,
    )
    print(f"{tag}\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        by_method = result.get("by_method", {})
        parts = [
            f"{name}={row.get('tokens_per_sec', 0.0):.1f}t/s"
            for name, row in by_method.items()
        ]
        print(f"DONE {tag}: " + "  ".join(parts), flush=True)


@app.local_entrypoint()
def launch_real_commit_mv_frontier_test(
    split: str = "test",
    version: str = "v1",
    n: int = 500,
    wait: bool = False,
    target: str = "Qwen/Qwen2.5-Coder-7B",
) -> None:
    """Focused PLD comparison for frontier VANTAGE-MV variants."""

    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    tag = f"vantage_real_commit_mv_frontier_{split}{n}_{version}"
    methods = (
        "blazedit_pld_w128_n10,"
        "vantage_force_pld_w128_n10,"
        f"{REAL_COMMIT_MV_STABLE_METHOD},"
        "vantage_mv_pld_q_s64_m8_f8_t8_w128_n10,"
        "vantage_mv_pld_pair_s64_m8_f8_t8_w128_n10,"
        "vantage_mv_pld_branch_s64_m8_f8_t8_w128_n10,"
        "vantage_mv_pld_s64_m8_f8_t8_g64_w128_n10,"
        "vantage_mv_pld_fst_s64_m8_f8_t8_w128_n10,"
        "vantage_mv_pld_fst_pair_s64_m8_f8_t8_w128_n10,"
        "vantage_mv_pld_fst_q_pair_s64_m8_f8_t8_w128_n10,"
        "vantage_mv_pld_fst_q_pair_branch_s64_m8_f8_t8_w128_n10,"
        "vantage_mv_pld_all_s64_m8_f8_t8_g64_w128_n10"
    )
    call = run_eagle_eval_job_any.spawn(
        run_tag=tag,
        target=target,
        n=n,
        max_new_tokens=256,
        methods=methods,
        problem_jsonl=f"/root/asts-spec/data/real_commits/path_a_{split}500_v1.jsonl",
        dtype="bfloat16",
        attn_impl="sdpa",
        code_proposer_fallback="root",
        transpld_min_match_len=4,
    )
    print(f"{tag}\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        by_method = result.get("by_method", {})
        parts = [
            f"{name}={row.get('tokens_per_sec', 0.0):.1f}t/s"
            for name, row in by_method.items()
        ]
        print(f"DONE {tag}: " + "  ".join(parts), flush=True)


@app.local_entrypoint()
def launch_real_commit_mv_frontier_selected_test(
    version: str = "v1",
    n: int = 500,
    wait: bool = False,
    target: str = "Qwen/Qwen2.5-Coder-7B",
) -> None:
    """Held-out test for the promoted stable real-commit MV config."""

    tag = f"vantage_real_commit_mv_frontier_selected_test{n}_{version}"
    methods = ",".join(
        [
            "blazedit_pld_w128_n10",
            "vantage_force_pld_w128_n10",
            REAL_COMMIT_MV_STABLE_METHOD,
        ]
    )
    call = run_eagle_eval_job_any.spawn(
        run_tag=tag,
        target=target,
        n=n,
        max_new_tokens=256,
        methods=methods,
        problem_jsonl="/root/asts-spec/data/real_commits/path_a_test500_v1.jsonl",
        dtype="bfloat16",
        attn_impl="sdpa",
        code_proposer_fallback="root",
        transpld_min_match_len=4,
    )
    print(f"{tag}\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        by_method = result.get("by_method", {})
        parts = [
            f"{name}={row.get('tokens_per_sec', 0.0):.1f}t/s"
            for name, row in by_method.items()
        ]
        print(f"DONE {tag}: " + "  ".join(parts), flush=True)


@app.local_entrypoint()
def launch_real_commit_pld_adjacent_test(
    split: str = "test",
    version: str = "v1",
    n: int = 500,
    wait: bool = False,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    methods: str = "",
    pld_rerank_weights: str = "",
    pld_rerank_margin: float = 0.0,
    pld_rerank_margin_gate: str = "false",
    pld_rerank_always_include_baseline: str = "true",
    pld_rerank_enable_left_extension: str = "false",
    pld_rerank_left_extension_max: int = 128,
    pld_rerank_policy: str = "learned",
    pld_rerank_fixed_rank: int = 0,
    pld_rerank_debug_trace: bool = False,
    problem_jsonl: str = "",
    mtp_heads_checkpoint: str = "/data/pld_mtp/postpld_linear_k4_n917_v1/postpld_mtp_heads_k4_linear.pt",
    mtp_num_heads: int = 4,
    mtp_trigger_accepted_len: int = 4,
    mtp_position: str = "post_pld",
    mtp_disable: bool = False,
    mtp_queue_enabled: bool = True,
    mtp_use_queued_only_on_weak_pld: bool = True,
    mtp_disable_extra_verify: bool = False,
) -> None:
    """Held-out PLD-adjacent decoder test, including exact-PLD K=4 reranking."""

    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    tag = f"vantage_real_commit_pld_adjacent_{split}{n}_{version}"
    if not methods:
        methods = ",".join(
            [
                "vanilla",
                "blazedit_pld_w128_n10",
                "rerank_exact_pld_k4_w128_n10",
                "delta_cache_pld_w128_n10",
                "fuzzy_resync_pld_w128_n10",
            ]
        )
    if not problem_jsonl:
        problem_jsonl = (
            f"/root/asts-spec/data/real_commits/real_commit_manifest_balanced_1000_v2_{split}500.jsonl"
        )
    call = run_eagle_eval_job_any.spawn(
        run_tag=tag,
        target=target,
        n=n,
        max_new_tokens=256,
        methods=methods,
        problem_jsonl=problem_jsonl,
        dtype="bfloat16",
        attn_impl="sdpa",
        code_proposer_fallback="root",
        transpld_min_match_len=4,
        pld_rerank_top_k=4,
        pld_rerank_weights=pld_rerank_weights,
        pld_rerank_margin=pld_rerank_margin,
        pld_rerank_margin_gate=pld_rerank_margin_gate,
        pld_rerank_always_include_baseline=pld_rerank_always_include_baseline,
        pld_rerank_enable_left_extension=pld_rerank_enable_left_extension,
        pld_rerank_left_extension_max=pld_rerank_left_extension_max,
        pld_rerank_policy=pld_rerank_policy,
        pld_rerank_fixed_rank=pld_rerank_fixed_rank,
        pld_rerank_debug_trace=pld_rerank_debug_trace,
        mtp_heads_checkpoint=mtp_heads_checkpoint,
        mtp_num_heads=mtp_num_heads,
        mtp_trigger_accepted_len=mtp_trigger_accepted_len,
        mtp_position=mtp_position,
        mtp_disable=mtp_disable,
        mtp_queue_enabled=mtp_queue_enabled,
        mtp_use_queued_only_on_weak_pld=mtp_use_queued_only_on_weak_pld,
        mtp_disable_extra_verify=mtp_disable_extra_verify,
    )
    print(f"{tag}\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        by_method = result.get("by_method", {})
        pld_tps = by_method.get("blazedit_pld_w128_n10", {}).get("tokens_per_sec", 0.0) or 1.0
        parts = []
        for name, row in by_method.items():
            tps = row.get("tokens_per_sec", 0.0)
            parts.append(f"{name}={tps:.1f}t/s({tps / pld_tps:.3f}x)")
        print(f"DONE {tag}: " + "  ".join(parts), flush=True)


@app.local_entrypoint()
def launch_real_commit_pld_opportunity_trace(
    split: str = "test",
    version: str = "v1",
    n: int = 500,
    wait: bool = False,
    target: str = "Qwen/Qwen2.5-Coder-7B",
) -> None:
    """Trace baseline PLD opportunities on the balanced real-commit split."""

    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    tag = f"vantage_real_commit_pld_opportunity_{split}{n}_{version}"
    call = run_eagle_eval_job_any.spawn(
        run_tag=tag,
        target=target,
        n=n,
        max_new_tokens=256,
        methods="blazedit_pld_w128_n10",
        problem_jsonl=f"/root/asts-spec/data/real_commits/real_commit_manifest_balanced_1000_v2_{split}500.jsonl",
        dtype="bfloat16",
        attn_impl="sdpa",
        code_proposer_fallback="root",
        transpld_min_match_len=4,
        pld_opportunity_trace=True,
    )
    print(f"{tag}\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        by_method = result.get("by_method", {})
        row = by_method.get("blazedit_pld_w128_n10", {})
        print(
            f"DONE {tag}: "
            f"pld={row.get('tokens_per_sec', 0.0):.1f}t/s "
            f"weak_runtime={100.0 * (row.get('pld_opp_weak_runtime_fraction', 0.0) or 0.0):.2f}% "
            f"weak_steps={row.get('pld_opp_weak_steps_total', 0)}",
            flush=True,
        )


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=21600,
    startup_timeout=3600,
    cpu=4,
    memory=65536,
)
def run_pld_mtp_heads_offline_job(
    *,
    version: str,
    train_run_tag: str,
    test_run_tag: str,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    method: str = "blazedit_pld_w128_n10",
    max_examples: int = 200000,
    collect_batch_size: int = 1,
    mtp_position: str = "pre_pld",
    head_loss_weights: str = "",
) -> dict:
    import json
    import os
    import subprocess
    import sys

    os.chdir("/root/asts-spec")
    install_cmds = [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
            "torch==2.5.1",
        ],
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "transformers>=4.46",
            "accelerate>=1.0",
            "huggingface-hub>=0.26",
            "numpy>=1.26",
        ],
    ]
    for install_cmd in install_cmds:
        print(f"$ {' '.join(install_cmd)}", flush=True)
        subprocess.run(install_cmd, check=True)

    root = f"/data/pld_mtp/{version}"
    os.makedirs(root, exist_ok=True)
    prefix = "postpld_" if mtp_position == "post_pld" else ""
    train_pt = f"{root}/{prefix}train.pt"
    test_pt = f"{root}/{prefix}test500.pt"
    output_projection = f"{root}/qwen_output_projection.pt"
    linear_ckpt = f"{root}/{prefix}mtp_heads_k4_linear.pt"
    linear_eval = f"{root}/{prefix}offline_eval_linear_k4"
    mlp_ckpt = f"{root}/{prefix}mtp_heads_k4_mlp.pt"
    mlp_eval = f"{root}/{prefix}offline_eval_mlp_k4"

    train_steps = f"/data/{train_run_tag}/eval/steps.jsonl"
    train_completions = f"/data/{train_run_tag}/eval/completions.jsonl"
    test_steps = f"/data/{test_run_tag}/eval/steps.jsonl"
    test_completions = f"/data/{test_run_tag}/eval/completions.jsonl"
    for path in (train_steps, train_completions, test_steps, test_completions):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    subprocess.run(
        [
            sys.executable,
            "scripts/collect_pld_mtp_training_data.py",
            "--target",
            target,
            "--steps",
            train_steps,
            "--completions",
            train_completions,
            "--output",
            train_pt,
            "--method",
            method,
            "--weak-only",
            "true",
            "--include-accepted-len-threshold",
            "4",
            "--max-examples",
            str(max_examples),
            "--dtype",
            "bf16",
            "--device",
            "cuda",
            "--batch-size",
            str(collect_batch_size),
            "--output-projection",
            output_projection,
            "--mtp-position",
            mtp_position,
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/collect_pld_mtp_training_data.py",
            "--target",
            target,
            "--steps",
            test_steps,
            "--completions",
            test_completions,
            "--output",
            test_pt,
            "--method",
            method,
            "--weak-only",
            "false",
            "--include-accepted-len-threshold",
            "4",
            "--dtype",
            "bf16",
            "--device",
            "cuda",
            "--batch-size",
            str(collect_batch_size),
            "--mtp-position",
            mtp_position,
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/train_pld_mtp_heads.py",
            "--data",
            train_pt,
            "--output",
            linear_ckpt,
            "--target",
            target,
            "--num-heads",
            "4",
            "--head-type",
            "linear",
            "--epochs",
            "1",
            "--batch-size",
            "512",
            "--lr",
            "1e-3",
            "--device",
            "cuda",
            "--output-projection",
            output_projection,
            "--head-loss-weights",
            head_loss_weights,
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_pld_mtp_heads_offline.py",
            "--data",
            test_pt,
            "--heads",
            linear_ckpt,
            "--output-dir",
            linear_eval,
            "--steps",
            test_steps,
            "--device",
            "cuda",
            "--mtp-position",
            mtp_position,
        ],
        check=True,
    )
    with open(f"{linear_eval}/report.json") as f:
        linear_report = json.load(f)
    linear_best = max(
        linear_report.get("policies", []),
        key=lambda row: row.get("corrected_projected_speedup", 0.0),
        default={},
    )

    mlp_report = None
    linear_speedup = float(linear_best.get("corrected_projected_speedup", 0.0) or 0.0)
    if 1.10 <= linear_speedup < 1.20:
        subprocess.run(
            [
                sys.executable,
                "scripts/train_pld_mtp_heads.py",
                "--data",
                train_pt,
                "--output",
                mlp_ckpt,
                "--target",
                target,
                "--num-heads",
                "4",
                "--head-type",
                "mlp",
                "--hidden-dim",
                "2048",
                "--epochs",
                "1",
                "--batch-size",
                "512",
                "--lr",
                "5e-4",
                "--device",
                "cuda",
                "--output-projection",
                output_projection,
                "--head-loss-weights",
                head_loss_weights,
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/evaluate_pld_mtp_heads_offline.py",
                "--data",
                test_pt,
                "--heads",
                mlp_ckpt,
                "--output-dir",
                mlp_eval,
                "--steps",
                test_steps,
                "--device",
                "cuda",
                "--mtp-position",
                mtp_position,
            ],
            check=True,
        )
        with open(f"{mlp_eval}/report.json") as f:
            mlp_report = json.load(f)

    data_volume.commit()
    hf_cache.commit()
    return {
        "version": version,
        "root": root,
        "train_pt": train_pt,
        "test_pt": test_pt,
        "linear_ckpt": linear_ckpt,
        "linear_report": linear_report,
        "linear_best": linear_best,
        "mlp_report": mlp_report,
    }


@app.local_entrypoint()
def launch_pld_mtp_heads_offline(
    version: str = "linear_k4_v1",
    wait: bool = True,
    train_run_tag: str = "vantage_real_commit_pld_adjacent_train500_mtp_train_trace_v1",
    test_run_tag: str = "vantage_real_commit_pld_adjacent_test500_rerank_fixed_rank3_test500_v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    max_examples: int = 200000,
    collect_batch_size: int = 1,
    mtp_position: str = "pre_pld",
    head_loss_weights: str = "",
) -> None:
    """Collect train/test MTP tensors, train K=4 heads, and run offline replay."""

    call = run_pld_mtp_heads_offline_job.spawn(
        version=version,
        train_run_tag=train_run_tag,
        test_run_tag=test_run_tag,
        target=target,
        max_examples=max_examples,
        collect_batch_size=collect_batch_size,
        mtp_position=mtp_position,
        head_loss_weights=head_loss_weights,
    )
    print(f"pld_mtp/{version}\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        best = result.get("linear_best", {})
        print(
            f"DONE pld_mtp/{version}: "
            f"linear_best={best.get('trigger_policy')} "
            f"{best.get('corrected_projected_speedup', 0.0):.3f}x "
            f"avg_extra={best.get('avg_extra_accepted_mtp_tokens_per_trigger', 0.0):.2f} "
            f"root={result.get('root')}",
            flush=True,
        )


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=7200,
    startup_timeout=3600,
    cpu=4,
    memory=65536,
)
def run_pld_mtp_eval_only_job(
    *,
    version: str,
    test_run_tag: str,
    method: str = "blazedit_pld_w128_n10",
    mtp_position: str = "pre_pld",
) -> dict:
    import json
    import os
    import subprocess
    import sys

    os.chdir("/root/asts-spec")
    install_cmds = [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
            "torch==2.5.1",
        ],
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "transformers>=4.46",
            "accelerate>=1.0",
            "huggingface-hub>=0.26",
            "numpy>=1.26",
        ],
    ]
    for install_cmd in install_cmds:
        print(f"$ {' '.join(install_cmd)}", flush=True)
        subprocess.run(install_cmd, check=True)

    root = f"/data/pld_mtp/{version}"
    test_steps = f"/data/{test_run_tag}/eval/steps.jsonl"
    prefix = "postpld_" if mtp_position == "post_pld" else ""
    test_pt = f"{root}/{prefix}test500.pt"
    reports = {}
    for name in ("linear", "mlp"):
        ckpt = f"{root}/{prefix}mtp_heads_k4_{name}.pt"
        if not os.path.exists(ckpt):
            continue
        out = f"{root}/{prefix}offline_eval_{name}_k4"
        subprocess.run(
            [
                sys.executable,
                "scripts/evaluate_pld_mtp_heads_offline.py",
                "--data",
                test_pt,
                "--heads",
                ckpt,
                "--steps",
                test_steps,
                "--method",
                method,
                "--output-dir",
                out,
                "--device",
                "cuda",
                "--mtp-position",
                mtp_position,
            ],
            check=True,
        )
        with open(f"{out}/report.json") as f:
            reports[name] = json.load(f)
    data_volume.commit()
    return {"version": version, "root": root, "reports": reports}


@app.local_entrypoint()
def launch_pld_mtp_eval_only(
    version: str = "linear_k4_v1",
    wait: bool = True,
    test_run_tag: str = "vantage_real_commit_pld_adjacent_test500_rerank_fixed_rank3_test500_v1",
    mtp_position: str = "pre_pld",
) -> None:
    """Re-run MTP offline replay against the full held-out PLD step trace."""

    call = run_pld_mtp_eval_only_job.spawn(
        version=version,
        test_run_tag=test_run_tag,
        mtp_position=mtp_position,
    )
    print(f"pld_mtp_eval/{version}\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        for name, report in result.get("reports", {}).items():
            best = max(
                report.get("policies", []),
                key=lambda row: row.get("corrected_projected_speedup", 0.0),
                default={},
            )
            print(
                f"DONE pld_mtp_eval/{version}/{name}: "
                f"{best.get('trigger_policy')} "
                f"{best.get('corrected_projected_speedup', 0.0):.3f}x "
                f"baseline_steps={best.get('baseline_steps')}",
                flush=True,
            )


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=21600,
    startup_timeout=3600,
    cpu=4,
    memory=65536,
)
def run_router_selected_mtp_heads_job(
    *,
    version: str,
    train_run_tag: str,
    test_run_tag: str,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    method: str = "blazedit_pld_w128_n10",
    router_threshold: float = 0.5,
    test_collection_threshold: float = 0.3,
    max_examples: int = 200000,
    collect_batch_size: int = 1,
    head_loss_weights: str = "4,2,1,1",
    run_two_epoch_if_close: bool = True,
) -> dict:
    import json
    import os
    import subprocess
    import sys

    os.chdir("/root/asts-spec")
    install_cmds = [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
            "torch==2.5.1",
        ],
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "transformers>=4.46",
            "accelerate>=1.0",
            "huggingface-hub>=0.26",
            "numpy>=1.26",
            "scikit-learn>=1.4",
        ],
    ]
    for install_cmd in install_cmds:
        print(f"$ {' '.join(install_cmd)}", flush=True)
        subprocess.run(install_cmd, check=True)

    root = f"/data/pld_mtp/{version}"
    os.makedirs(root, exist_ok=True)
    train_steps = f"/data/{train_run_tag}/eval/steps.jsonl"
    train_completions = f"/data/{train_run_tag}/eval/completions.jsonl"
    test_steps = f"/data/{test_run_tag}/eval/steps.jsonl"
    test_completions = f"/data/{test_run_tag}/eval/completions.jsonl"
    for path in (train_steps, train_completions, test_steps, test_completions):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    router_dir = f"{root}/weak_router"
    router_pkl = f"{router_dir}/router.pkl"
    train_pt = f"{root}/router_selected_train.pt"
    test_pt = f"{root}/router_selected_test500.pt"
    output_projection = f"{root}/qwen_output_projection.pt"
    linear_ckpt = f"{root}/mtp_heads_k4_router_selected_linear.pt"
    linear_eval = f"{root}/router_selected_offline_eval"
    two_epoch_ckpt = f"{root}/mtp_heads_k4_router_selected_linear_2epoch.pt"
    two_epoch_eval = f"{root}/router_selected_offline_eval_2epoch"

    subprocess.run(
        [
            sys.executable,
            "scripts/train_weak_pld_router.py",
            "--train-steps",
            train_steps,
            "--test-steps",
            test_steps,
            "--method",
            method,
            "--output-dir",
            router_dir,
            "--accepted-len-threshold",
            "4",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/collect_router_selected_mtp_training_data.py",
            "--target",
            target,
            "--steps",
            train_steps,
            "--completions",
            train_completions,
            "--router",
            router_pkl,
            "--router-threshold",
            str(router_threshold),
            "--collection-threshold",
            str(router_threshold),
            "--output",
            train_pt,
            "--method",
            method,
            "--max-examples",
            str(max_examples),
            "--dtype",
            "bf16",
            "--device",
            "cuda",
            "--batch-size",
            str(collect_batch_size),
            "--output-projection",
            output_projection,
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/collect_router_selected_mtp_training_data.py",
            "--target",
            target,
            "--steps",
            test_steps,
            "--completions",
            test_completions,
            "--router",
            router_pkl,
            "--router-threshold",
            str(router_threshold),
            "--collection-threshold",
            str(test_collection_threshold),
            "--output",
            test_pt,
            "--method",
            method,
            "--dtype",
            "bf16",
            "--device",
            "cuda",
            "--batch-size",
            str(collect_batch_size),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/train_pld_mtp_heads.py",
            "--data",
            train_pt,
            "--output",
            linear_ckpt,
            "--target",
            target,
            "--num-heads",
            "4",
            "--head-type",
            "linear",
            "--epochs",
            "1",
            "--batch-size",
            "512",
            "--lr",
            "1e-3",
            "--device",
            "cuda",
            "--output-projection",
            output_projection,
            "--loss-weights",
            head_loss_weights,
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_router_selected_mtp_heads_offline.py",
            "--steps",
            test_steps,
            "--completions",
            test_completions,
            "--data",
            test_pt,
            "--router",
            router_pkl,
            "--heads",
            linear_ckpt,
            "--method",
            method,
            "--output-dir",
            linear_eval,
            "--router-thresholds",
            "0.3,0.5,0.7",
            "--confidence-thresholds",
            "0.0,0.3,0.5,0.7,0.9",
            "--device",
            "cuda",
        ],
        check=True,
    )
    with open(f"{linear_eval}/report.json") as f:
        linear_report = json.load(f)

    two_epoch_report = None
    best = linear_report.get("best", {})
    if (
        run_two_epoch_if_close
        and 1.10 <= float(best.get("projected_speedup", 0.0) or 0.0) < 1.20
    ):
        subprocess.run(
            [
                sys.executable,
                "scripts/train_pld_mtp_heads.py",
                "--data",
                train_pt,
                "--output",
                two_epoch_ckpt,
                "--target",
                target,
                "--num-heads",
                "4",
                "--head-type",
                "linear",
                "--epochs",
                "2",
                "--batch-size",
                "512",
                "--lr",
                "1e-3",
                "--device",
                "cuda",
                "--output-projection",
                output_projection,
                "--loss-weights",
                head_loss_weights,
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/evaluate_router_selected_mtp_heads_offline.py",
                "--steps",
                test_steps,
                "--completions",
                test_completions,
                "--data",
                test_pt,
                "--router",
                router_pkl,
                "--heads",
                two_epoch_ckpt,
                "--method",
                method,
                "--output-dir",
                two_epoch_eval,
                "--router-thresholds",
                "0.3,0.5,0.7",
                "--confidence-thresholds",
                "0.0,0.3,0.5,0.7,0.9",
                "--device",
                "cuda",
            ],
            check=True,
        )
        with open(f"{two_epoch_eval}/report.json") as f:
            two_epoch_report = json.load(f)

    def _read_json(path: str) -> dict:
        with open(path) as f:
            return json.load(f)

    result = {
        "version": version,
        "root": root,
        "router_report": _read_json(f"{router_dir}/report.json"),
        "train_summary": _read_json(f"{train_pt}.summary.json"),
        "test_summary": _read_json(f"{test_pt}.summary.json"),
        "linear_training_report": _read_json(f"{linear_ckpt}.json"),
        "linear_eval_report": linear_report,
        "two_epoch_eval_report": two_epoch_report,
    }
    data_volume.commit()
    hf_cache.commit()
    return result


@app.local_entrypoint()
def launch_router_selected_mtp_heads(
    version: str = "router_selected_k4_v1",
    wait: bool = True,
    train_run_tag: str = "vantage_real_commit_mv_decoder_train_guard_cap_branch_grid500_v1",
    test_run_tag: str = "vantage_real_commit_pld_adjacent_test500_rerank_fixed_rank3_test500_v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    router_threshold: float = 0.5,
    test_collection_threshold: float = 0.3,
    max_examples: int = 200000,
    collect_batch_size: int = 1,
) -> None:
    """Collect router-selected MTP data, train K=4 heads, and run offline replay."""

    call = run_router_selected_mtp_heads_job.spawn(
        version=version,
        train_run_tag=train_run_tag,
        test_run_tag=test_run_tag,
        target=target,
        router_threshold=router_threshold,
        test_collection_threshold=test_collection_threshold,
        max_examples=max_examples,
        collect_batch_size=collect_batch_size,
    )
    print(f"pld_mtp_router_selected/{version}\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        best = result.get("linear_eval_report", {}).get("best", {})
        print(
            f"DONE pld_mtp_router_selected/{version}: "
            f"best={best.get('projected_speedup', 0.0):.3f}x "
            f"router_thr={best.get('router_threshold')} conf={best.get('confidence_threshold')} "
            f"train_examples={result.get('train_summary', {}).get('n_examples')} "
            f"test_examples={result.get('test_summary', {}).get('n_examples')} "
            f"root={result.get('root')}",
            flush=True,
        )


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=28800,
    startup_timeout=3600,
    cpu=4,
    memory=65536,
)
def run_router_selected_mtp_finetune_job(
    *,
    version: str,
    train_run_tag: str,
    test_run_tag: str,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    method: str = "blazedit_pld_w128_n10",
    source_router_selected_root: str = "/data/pld_mtp/router_selected_k4_v2",
    generic_init_heads: str = "/data/pld_mtp/linear_k4_v1/mtp_heads_k4_linear.pt",
    train_collection_threshold: float = 0.3,
    router_threshold: float = 0.5,
    max_examples: int = 100000,
    collect_batch_size: int = 1,
) -> dict:
    import json
    import os
    import subprocess
    import sys

    os.chdir("/root/asts-spec")
    install_cmds = [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
            "torch==2.5.1",
        ],
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "transformers>=4.46",
            "accelerate>=1.0",
            "huggingface-hub>=0.26",
            "numpy>=1.26",
            "scikit-learn>=1.4",
        ],
    ]
    for install_cmd in install_cmds:
        print(f"$ {' '.join(install_cmd)}", flush=True)
        subprocess.run(install_cmd, check=True)

    root = f"/data/pld_mtp/{version}"
    os.makedirs(root, exist_ok=True)
    train_steps = f"/data/{train_run_tag}/eval/steps.jsonl"
    train_completions = f"/data/{train_run_tag}/eval/completions.jsonl"
    test_steps = f"/data/{test_run_tag}/eval/steps.jsonl"
    test_completions = f"/data/{test_run_tag}/eval/completions.jsonl"
    for path in (train_steps, train_completions, test_steps, test_completions):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    source_train_pt = f"{source_router_selected_root}/router_selected_train.pt"
    source_output_projection = f"{source_router_selected_root}/qwen_output_projection.pt"
    router_pkl = f"{source_router_selected_root}/weak_router/router.pkl"
    for path in (source_train_pt, source_output_projection, router_pkl, generic_init_heads):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    audit_dir = f"{root}/router_selected_audit"
    overfit_ckpt = f"{root}/debug_overfit_router_selected_256.pt"
    train_pt = f"{root}/router_selected_train_large.pt"
    test_pt = f"{root}/router_selected_test500_thr03.pt"
    output_projection = f"{root}/qwen_output_projection.pt"
    finetune_ckpt = f"{root}/mtp_heads_k4_router_selected_finetune.pt"
    finetune_eval = f"{root}/router_selected_finetune_offline_eval"
    three_epoch_ckpt = f"{root}/mtp_heads_k4_router_selected_finetune_3epoch_lr1e-4.pt"
    three_epoch_eval = f"{root}/router_selected_finetune_3epoch_lr1e-4_offline_eval"

    subprocess.run(
        [
            sys.executable,
            "scripts/audit_router_selected_mtp_data.py",
            "--data",
            source_train_pt,
            "--steps",
            f"/data/vantage_real_commit_mv_decoder_train_guard_cap_branch_grid500_v1/eval/steps.jsonl",
            "--completions",
            f"/data/vantage_real_commit_mv_decoder_train_guard_cap_branch_grid500_v1/eval/completions.jsonl",
            "--forbid-task-ids-jsonl",
            test_completions,
            "--output-dir",
            audit_dir,
            "--method",
            method,
            "--target",
            target,
        ],
        check=True,
    )
    with open(f"{audit_dir}/report.json") as f:
        audit_report = json.load(f)
    if float(audit_report.get("alignment_pass_rate", 0.0) or 0.0) < 0.999:
        result = {
            "version": version,
            "root": root,
            "audit_report": audit_report,
            "decision": "fix data alignment",
        }
        data_volume.commit()
        hf_cache.commit()
        return result

    subprocess.run(
        [
            sys.executable,
            "scripts/train_pld_mtp_heads.py",
            "--data",
            source_train_pt,
            "--output",
            overfit_ckpt,
            "--target",
            target,
            "--num-heads",
            "4",
            "--head-type",
            "linear",
            "--epochs",
            "50",
            "--batch-size",
            "64",
            "--lr",
            "1e-3",
            "--loss-weights",
            "8,4,1,1",
            "--max-train-examples",
            "256",
            "--eval-train-subset",
            "--device",
            "cuda",
            "--output-projection",
            source_output_projection,
        ],
        check=True,
    )
    with open(f"{overfit_ckpt}.json") as f:
        overfit_report = json.load(f)
    overfit_token0 = float(
        (overfit_report.get("reports") or [{}])[-1].get("top1_accuracy_t_plus_1", 0.0) or 0.0
    )
    if overfit_token0 < 0.95:
        result = {
            "version": version,
            "root": root,
            "audit_report": audit_report,
            "overfit_report": overfit_report,
            "overfit_token0_accuracy": overfit_token0,
            "decision": "fix label alignment or head wiring",
        }
        data_volume.commit()
        hf_cache.commit()
        return result

    subprocess.run(
        [
            sys.executable,
            "scripts/collect_router_selected_mtp_training_data.py",
            "--target",
            target,
            "--steps",
            train_steps,
            "--completions",
            train_completions,
            "--router",
            router_pkl,
            "--router-threshold",
            str(router_threshold),
            "--collection-threshold",
            str(train_collection_threshold),
            "--output",
            train_pt,
            "--method",
            method,
            "--max-examples",
            str(max_examples),
            "--dtype",
            "bf16",
            "--device",
            "cuda",
            "--batch-size",
            str(collect_batch_size),
            "--output-projection",
            output_projection,
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/collect_router_selected_mtp_training_data.py",
            "--target",
            target,
            "--steps",
            test_steps,
            "--completions",
            test_completions,
            "--router",
            router_pkl,
            "--router-threshold",
            str(router_threshold),
            "--collection-threshold",
            str(train_collection_threshold),
            "--output",
            test_pt,
            "--method",
            method,
            "--dtype",
            "bf16",
            "--device",
            "cuda",
            "--batch-size",
            str(collect_batch_size),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/train_pld_mtp_heads.py",
            "--data",
            train_pt,
            "--init-heads",
            generic_init_heads,
            "--output",
            finetune_ckpt,
            "--target",
            target,
            "--num-heads",
            "4",
            "--head-type",
            "linear",
            "--epochs",
            "2",
            "--batch-size",
            "512",
            "--lr",
            "5e-4",
            "--loss-weights",
            "8,4,1,1",
            "--device",
            "cuda",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_router_selected_mtp_heads_offline.py",
            "--steps",
            test_steps,
            "--completions",
            test_completions,
            "--data",
            test_pt,
            "--router",
            router_pkl,
            "--heads",
            finetune_ckpt,
            "--method",
            method,
            "--output-dir",
            finetune_eval,
            "--router-thresholds",
            "0.3,0.5,0.7",
            "--confidence-thresholds",
            "0.0,0.3,0.5,0.7,0.9",
            "--device",
            "cuda",
        ],
        check=True,
    )

    def _read_json(path: str) -> dict:
        with open(path) as f:
            return json.load(f)

    finetune_report = _read_json(f"{finetune_eval}/report.json")
    best = finetune_report.get("best", {})
    three_epoch_report = None
    if 1.10 <= float(best.get("projected_speedup", 0.0) or 0.0) < 1.20:
        subprocess.run(
            [
                sys.executable,
                "scripts/train_pld_mtp_heads.py",
                "--data",
                train_pt,
                "--init-heads",
                generic_init_heads,
                "--output",
                three_epoch_ckpt,
                "--target",
                target,
                "--num-heads",
                "4",
                "--head-type",
                "linear",
                "--epochs",
                "3",
                "--batch-size",
                "512",
                "--lr",
                "1e-4",
                "--loss-weights",
                "8,4,1,1",
                "--device",
                "cuda",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/evaluate_router_selected_mtp_heads_offline.py",
                "--steps",
                test_steps,
                "--completions",
                test_completions,
                "--data",
                test_pt,
                "--router",
                router_pkl,
                "--heads",
                three_epoch_ckpt,
                "--method",
                method,
                "--output-dir",
                three_epoch_eval,
                "--router-thresholds",
                "0.3,0.5,0.7",
                "--confidence-thresholds",
                "0.0,0.3,0.5,0.7,0.9",
                "--device",
                "cuda",
            ],
            check=True,
        )
        three_epoch_report = _read_json(f"{three_epoch_eval}/report.json")

    final_report = three_epoch_report or finetune_report
    final_best = final_report.get("best", {})
    if (
        float(final_best.get("projected_speedup", 0.0) or 0.0) >= 1.20
        and float(final_best.get("used_token0_reject_rate", 1.0) or 1.0) <= 0.40
        and float(final_best.get("avg_accepted_queued_tokens_per_use", 0.0) or 0.0) >= 1.0
    ):
        decision = "recommend runtime weak-router queued MTP implementation"
    elif float(final_best.get("projected_speedup", 0.0) or 0.0) >= 1.10:
        decision = "collect more router-selected data or continue offline training; no runtime yet"
    else:
        decision = "abandon queued MTP for this setup"

    result = {
        "version": version,
        "root": root,
        "train_run_tag": train_run_tag,
        "test_run_tag": test_run_tag,
        "source_router_selected_root": source_router_selected_root,
        "generic_init_heads": generic_init_heads,
        "audit_report": audit_report,
        "overfit_report": overfit_report,
        "overfit_token0_accuracy": overfit_token0,
        "large_train_summary": _read_json(f"{train_pt}.summary.json"),
        "large_test_summary": _read_json(f"{test_pt}.summary.json"),
        "finetune_training_report": _read_json(f"{finetune_ckpt}.json"),
        "finetune_eval_report": finetune_report,
        "three_epoch_eval_report": three_epoch_report,
        "decision": decision,
    }
    data_volume.commit()
    hf_cache.commit()
    return result


@app.local_entrypoint()
def launch_router_selected_mtp_finetune(
    version: str = "router_selected_finetune_n917_v1",
    wait: bool = True,
    train_run_tag: str = "vantage_real_commit_pld_adjacent_train917_mtp_train_trace_postpld_n917_v1",
    test_run_tag: str = "vantage_real_commit_pld_adjacent_test500_rerank_fixed_rank3_test500_v1",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    train_collection_threshold: float = 0.3,
    max_examples: int = 100000,
    collect_batch_size: int = 1,
) -> None:
    """Audit, overfit-check, collect larger router-selected data, fine-tune, and replay offline."""

    call = run_router_selected_mtp_finetune_job.spawn(
        version=version,
        train_run_tag=train_run_tag,
        test_run_tag=test_run_tag,
        target=target,
        train_collection_threshold=train_collection_threshold,
        max_examples=max_examples,
        collect_batch_size=collect_batch_size,
    )
    print(f"pld_mtp_router_selected_finetune/{version}\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        eval_report = result.get("three_epoch_eval_report") or result.get("finetune_eval_report") or {}
        best = eval_report.get("best", {})
        print(
            f"DONE pld_mtp_router_selected_finetune/{version}: "
            f"decision={result.get('decision')} "
            f"audit_pass={100.0 * float(result.get('audit_report', {}).get('alignment_pass_rate', 0.0) or 0.0):.2f}% "
            f"overfit_token0={100.0 * float(result.get('overfit_token0_accuracy', 0.0) or 0.0):.1f}% "
            f"train_examples={result.get('large_train_summary', {}).get('n_examples')} "
            f"best={float(best.get('projected_speedup', 0.0) or 0.0):.3f}x "
            f"token0_reject={100.0 * float(best.get('used_token0_reject_rate', 0.0) or 0.0):.1f}% "
            f"accepted/use={float(best.get('avg_accepted_queued_tokens_per_use', 0.0) or 0.0):.2f} "
            f"root={result.get('root')}",
            flush=True,
        )


@app.local_entrypoint()
def launch_pld_gated_lookahead_eval(
    split: str = "test",
    version: str = "v1",
    n: int = 50,
    wait: bool = False,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    methods: str = "blazedit_pld_w128_n10,lookahead_w8_n4_i4,pld_gated_lookahead_w128_n10",
    lookahead_window: int = 8,
    lookahead_ngram: int = 4,
    lookahead_iters: int = 4,
    lookahead_max_draft: int = 16,
    lookahead_one_forward: bool = False,
    pld_lookahead_router: str = "rule",
    pld_lookahead_router_threshold: float = 0.3,
    pld_lookahead_trigger: str = "router_weak",
    problem_jsonl: str = "",
) -> None:
    """Run the PLD-gated Lookahead smoke/full real-commit benchmark."""

    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    tag = f"vantage_pld_gated_lookahead_{split}{n}_{version}"
    if not problem_jsonl:
        problem_jsonl = (
            f"/root/asts-spec/data/real_commits/real_commit_manifest_balanced_1000_v2_{split}500.jsonl"
        )
    call = run_eagle_eval_job_any.spawn(
        run_tag=tag,
        target=target,
        n=n,
        max_new_tokens=256,
        methods=methods,
        problem_jsonl=problem_jsonl,
        dtype="bfloat16",
        attn_impl="sdpa",
        code_proposer_fallback="root",
        transpld_min_match_len=4,
        lookahead_window=lookahead_window,
        lookahead_ngram=lookahead_ngram,
        lookahead_iters=lookahead_iters,
        lookahead_max_draft=lookahead_max_draft,
        lookahead_one_forward=lookahead_one_forward,
        pld_lookahead_router=pld_lookahead_router,
        pld_lookahead_router_threshold=pld_lookahead_router_threshold,
        pld_lookahead_trigger=pld_lookahead_trigger,
    )
    print(f"{tag}\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        by_method = result.get("by_method", {})
        pld_tps = (
            by_method.get("blazedit_pld_w128_n10", {}).get("tokens_per_sec", 0.0)
            or 1.0
        )
        parts = []
        for name, row in by_method.items():
            tps = row.get("tokens_per_sec", 0.0)
            parts.append(
                f"{name}={tps:.1f}t/s/{(tps / pld_tps):.3f}x "
                f"la_calls={row.get('lookahead_calls', 0)} "
                f"la_acc={row.get('lookahead_accepted_len_mean', 0.0):.2f} "
                f"la_fwd={row.get('lookahead_forward_calls_total', 0)} "
                f"la_ms={row.get('lookahead_ms_per_call_mean', row.get('lookahead_ms_per_call', 0.0)):.2f} "
                f"la_tok0={row.get('lookahead_tok0_reject_rate', 0.0):.2f}"
            )
        print(f"DONE {tag}: " + "  ".join(parts), flush=True)


@app.local_entrypoint()
def launch_pld_candidate_oracle(
    source_run_tag: str = "vantage_real_commit_pld_opportunity_test500_v1",
    version: str = "v1",
    wait: bool = True,
    target: str = "Qwen/Qwen2.5-Coder-7B",
) -> None:
    """Run best-of-K and longer-context PLD candidate oracle on trace artifacts."""

    call = run_pld_candidate_oracle_job.spawn(
        source_run_tag=source_run_tag,
        version=version,
        target=target,
        method="blazedit_pld_w128_n10",
    )
    print(f"{source_run_tag}/pld_candidate_oracle_{version}\t{call.object_id}", flush=True)
    if wait:
        report = call.get()
        best = report.get("best_of_k", [])
        longctx = report.get("longer_context", [])
        best_bits = [
            f"K{row.get('K')}={row.get('oracle_ambig_accepted', 0.0):.2f}/"
            f"{row.get('projected_speedup', 1.0):.3f}x"
            for row in best
        ]
        resolved = max(
            (row.get("cumulative_resolved_unique_pct", 0.0) for row in longctx),
            default=0.0,
        )
        print(
            f"DONE oracle: ambiguous={report.get('ambiguous_steps', 0)} "
            f"runtime={100.0 * report.get('ambiguous_runtime_fraction', 0.0):.2f}% "
            f"{' '.join(best_bits)} longctx_resolved={resolved:.1f}%",
            flush=True,
        )


@app.local_entrypoint()
def launch_multiview_real_commit_grid(
    split: str = "train",
    version: str = "v1",
    wait: bool = False,
    target: str = "Qwen/Qwen2.5-Coder-7B",
) -> None:
    """Launch the strong-threshold/margin grid after PLD passthrough passes."""

    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    methods = ["blazedit_pld_w128_n10"]
    for strong in (16, 32, 64, 96):
        for margin in (0, 16, 32, 64):
            methods.append(f"vantage_mvpld_s{strong}_m{margin}_w128_n10")
    tag = f"vantage_real_commit_multiview_grid_{split}500_{version}"
    call = run_eagle_eval_job_any.spawn(
        run_tag=tag,
        target=target,
        n=500,
        max_new_tokens=256,
        methods=",".join(methods),
        problem_jsonl=f"/root/asts-spec/data/real_commits/path_a_{split}500_v1.jsonl",
        dtype="bfloat16",
        attn_impl="sdpa",
        code_proposer_fallback="root",
        transpld_min_match_len=4,
    )
    print(f"{tag}\t{call.object_id}", flush=True)
    if wait:
        result = call.get()
        by_method = result.get("by_method", {})
        parts = [
            f"{name}={row.get('tokens_per_sec', 0.0):.1f}t/s"
            for name, row in by_method.items()
        ]
        print(f"DONE {tag}: " + "  ".join(parts), flush=True)
