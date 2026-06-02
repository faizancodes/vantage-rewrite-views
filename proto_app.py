"""Modal app for the ASTS-Spec prototype evaluation.

Two functions, sharing the same data + hf_cache volumes:

    verify_lossless    (L40S, ~5-10 min, ~$0.10)
        runs scripts/verify_lossless.py on N HumanEval prompts; asserts that
        vanilla AR == fixed-k spec == ASTS-Spec byte-for-byte. Exits 1 on
        any divergence.

    run_eval           (L40S, ~30-60 min, ~$0.40-0.80)
        runs scripts/run_prototype.py on full HumanEval (164 problems); writes
        per-step JSONL log + aggregate metrics.

The image reuses the v0/v1 microbench image (no flash-attn needed since
that's a separate deps tree maintained in modal_app.py). Keeping proto_app
deliberately small and decoupled.

Usage:
    pip install -e .[modal]
    modal run proto_app.py::verify_lossless
    modal run proto_app.py::verify_lossless --n 5
    modal run proto_app.py::run_eval --n 164
"""

from __future__ import annotations

from pathlib import Path

import modal


_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_TAG = "proto_v0"


# ---------------------------------------------------------------------------
# Image: same shape as modal_app.py minus flash-attn. Adds `datasets` for
# HumanEval loading.
# ---------------------------------------------------------------------------

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
        "pydivsufsort>=0.0.18",  # Fast O(N) suffix-array construction for retrieval drafting
    )
    # Lookahead decoding (Hao et al. 2024). lade was last updated for
    # transformers ~4.41 and breaks importing under our 4.46+. Specifically
    # it imports `GreedySearchOutput` and `SampleOutput`, which were renamed
    # to `GenerateDecoderOnlyOutput` etc. in modern transformers.
    # We patch lade's source after install to use the new names so the
    # rest of the eval pipeline can run.
    .run_commands(
        "pip install git+https://github.com/hao-ai-lab/LookaheadDecoding.git@main",
        "pip install --upgrade 'transformers>=4.46'",
        # Patch lade for transformers >= 4.46:
        # - Remove the import of GreedySearchOutput/SampleOutput (renamed)
        # - Patch type annotations that still reference them by name
        "LADE_DECODING=$(find /usr/local/lib/python*/site-packages/lade -name 'decoding.py' | head -1) && "
        "echo \"patching $LADE_DECODING\" && "
        "sed -i 's/, GreedySearchOutput, SampleOutput//' \"$LADE_DECODING\" && "
        "sed -i 's/SampleOutput/GenerateDecoderOnlyOutput/g' \"$LADE_DECODING\" && "
        "sed -i 's/GreedySearchOutput/GenerateDecoderOnlyOutput/g' \"$LADE_DECODING\" && "
        "( head -10 \"$LADE_DECODING\" | grep -q GenerateDecoderOnlyOutput || "
        "  sed -i '1i from transformers.generation.utils import GenerateDecoderOnlyOutput' \"$LADE_DECODING\" ) && "
        "echo 'patched lade decoding.py'",
        # Don't fail the build if lade still won't import — we don't need it for
        # the EAGLE / EAGLE-2 paths. lookahead_decoder.init_lookahead() catches
        # the failure at runtime and skips the method.
        "python -c 'import lade; print(\"lade ok at:\", lade.__file__)' || echo 'lade import FAILED — continuing build (lookahead method will be unavailable)'",
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
    .env({
        "HF_HOME": "/cache/huggingface",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_DATASETS_CACHE": "/cache/huggingface/datasets",
    })
)


# Reuse the same volumes as the microbench app
data_volume = modal.Volume.from_name("asts-spec-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("asts-spec-hf-cache", create_if_missing=True)


app = modal.App("asts-spec-proto", image=image)


# ---------------------------------------------------------------------------
# verify_lossless: smoke test that all 3 modes produce byte-identical output
# ---------------------------------------------------------------------------


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=10800,
    cpu=4,
)
def verify_lossless(
    target: str = "Qwen/Qwen2.5-Coder-7B",
    draft: str = "Qwen/Qwen2.5-Coder-0.5B",
    n: int = 3,
    max_new_tokens: int = 32,
    k: int = 4,
    dtype: str = "bfloat16",
    attn_impl: str = "sdpa",
    strict_determinism: bool = True,
    run_tag: str = DEFAULT_RUN_TAG,
) -> dict:
    """Run the lossless smoke test on Modal."""
    import json
    import os
    import subprocess

    os.chdir("/root/asts-spec")
    output_path = f"/data/{run_tag}/lossless_verify.json"

    cmd = [
        "python", "scripts/verify_lossless.py",
        "--target", target,
        "--draft", draft,
        "--n", str(n),
        "--max-new-tokens", str(max_new_tokens),
        "--k", str(k),
        "--dtype", dtype,
        "--attn-impl", attn_impl,
        "--output", output_path,
        "--log-level", "INFO",
    ]
    if strict_determinism:
        cmd.append("--strict-determinism")

    print(f"$ {' '.join(cmd)}", flush=True)
    # check=False so we can return diagnostic info on lossless divergence
    result = subprocess.run(cmd, check=False)

    data_volume.commit()
    hf_cache.commit()

    with open(output_path) as f:
        report = json.load(f)
    return {
        "exit_code": result.returncode,
        "n_match_vf": report["n_match_vf"],
        "n_match_va": report["n_match_va"],
        "n_total": report["n_total"],
        "lossless_passed": result.returncode == 0,
        "output_path": output_path,
    }


# ---------------------------------------------------------------------------
# run_eval: full HumanEval evaluation with per-method metrics + per-node-type
# acceptance histogram
# ---------------------------------------------------------------------------


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="H100",
    timeout=14400,  # up to 4hr for 32B-class targets
    cpu=4,
    memory=49152,  # extra headroom for 32B-class weights + KV cache
)
def run_eval_large_target(
    target: str = "Qwen/Qwen2.5-Coder-32B",
    draft: str = "Qwen/Qwen2.5-Coder-0.5B",
    n: int = 164,
    max_new_tokens: int = 256,
    k_fixed: str = "2,4,8",
    methods: str = "vanilla,fixed",
    dtype: str = "bfloat16",
    attn_impl: str = "sdpa",
    run_tag: str = "proto_32b_v0",
) -> dict:
    """Run vanilla-draft speculative-decoding eval against a large target
    (e.g. Qwen2.5-Coder-32B). Uses an H100 to fit 32B in bf16 (~64GB) plus
    KV cache and the 0.5B draft.

    Why a separate function? The default `run_eval` pins L40S, which has only
    48GB and cannot host 32B in bf16. This sibling pins H100 + raises the
    memory cap so a single Modal call can load both target + draft cleanly.
    """
    import json
    import os
    import subprocess

    os.chdir("/root/asts-spec")
    output_dir = f"/data/{run_tag}/eval"

    cmd = [
        "python", "scripts/run_prototype.py",
        "--output-dir", output_dir,
        "--target", target,
        "--draft", draft,
        "--n", str(n),
        "--max-new-tokens", str(max_new_tokens),
        "--k-fixed", k_fixed,
        "--methods", methods,
        "--dtype", dtype,
        "--attn-impl", attn_impl,
        "--log-level", "INFO",
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    hf_cache.commit()

    with open(f"{output_dir}/aggregate.json") as f:
        agg = json.load(f)
    return {
        "by_method": agg["by_method"],
        "meta": agg["meta"],
        "output_dir": output_dir,
    }


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=7200,  # up to 2hr for full 164-problem eval w/ all methods
    cpu=4,
)
def run_eval(
    target: str = "Qwen/Qwen2.5-Coder-7B",
    draft: str = "Qwen/Qwen2.5-Coder-0.5B",
    n: int = 164,
    max_new_tokens: int = 256,
    k_fixed: str = "4,8",
    methods: str = "vanilla,fixed,asts",
    dtype: str = "bfloat16",
    attn_impl: str = "sdpa",
    run_tag: str = "proto_v0",
) -> dict:
    """Run the full HumanEval eval. Writes steps.jsonl + aggregate.json."""
    import json
    import os
    import subprocess

    os.chdir("/root/asts-spec")
    output_dir = f"/data/{run_tag}/eval"

    cmd = [
        "python", "scripts/run_prototype.py",
        "--output-dir", output_dir,
        "--target", target,
        "--draft", draft,
        "--n", str(n),
        "--max-new-tokens", str(max_new_tokens),
        "--k-fixed", k_fixed,
        "--methods", methods,
        "--dtype", dtype,
        "--attn-impl", attn_impl,
        "--log-level", "INFO",
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    hf_cache.commit()

    with open(f"{output_dir}/aggregate.json") as f:
        agg = json.load(f)
    return {
        "by_method": agg["by_method"],
        "by_node_type_top10": dict(
            sorted(agg["by_node_type"].items(), key=lambda x: -x[1]["n"])[:10]
        ),
        "meta": agg["meta"],
        "output_dir": output_dir,
    }


# ---------------------------------------------------------------------------
# train_eagle: train an EAGLE-1 draft head on Qwen-Coder-7B
# ---------------------------------------------------------------------------


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=14400,  # 4 hours — generous for first training run
    cpu=4,
)
def train_eagle(
    target: str = "Qwen/Qwen2.5-Coder-7B",
    n_samples: int = 10_000,
    chunk_len: int = 1024,
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 1e-4,
    warmup_steps: int = 200,
    save_every: int = 1000,
    log_every: int = 20,
    kl_weight: float = 0.7,
    h_weight: float = 0.3,
    dtype: str = "bfloat16",
    seed: int = 42,
    run_tag: str = "eagle_v0",
) -> dict:
    """Train an EAGLE-1 head on Modal L40S. Saves checkpoints to /data/<tag>/eagle/."""
    import os
    import subprocess

    os.chdir("/root/asts-spec")
    output_dir = f"/data/{run_tag}/eagle"

    cmd = [
        "python", "scripts/train_eagle.py",
        "--target", target,
        "--output-dir", output_dir,
        "--n-samples", str(n_samples),
        "--chunk-len", str(chunk_len),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--warmup-steps", str(warmup_steps),
        "--save-every", str(save_every),
        "--log-every", str(log_every),
        "--kl-weight", str(kl_weight),
        "--h-weight", str(h_weight),
        "--dtype", dtype,
        "--seed", str(seed),
        "--log-level", "INFO",
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    hf_cache.commit()

    return {
        "output_dir": output_dir,
        "final_checkpoint": f"{output_dir}/eagle_final.pt",
    }


# ---------------------------------------------------------------------------
# verify_eagle_lossless: byte-identical check for EAGLE-based spec decode
# ---------------------------------------------------------------------------


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=10800,
    cpu=4,
)
def verify_eagle_lossless(
    target: str = "Qwen/Qwen2.5-Coder-7B",
    target_trust_remote_code: bool = False,
    eagle_checkpoint: str = "/data/eagle_v0/eagle/eagle_final.pt",
    n: int = 3,
    max_new_tokens: int = 32,
    k: int = 4,
    tree_k: int = 2,
    tree_w: int = 2,
    dtype: str = "float32",
    attn_impl: str = "eager",
    strict_determinism: bool = True,
    language: str = "python",
    prompt_variant: str = "full",
    run_tag: str = "eagle_v0",
    include_vantage: bool = False,
    include_code_proposers: bool = False,
    code_methods: str = "",
    assistant_model: str = "Qwen/Qwen2.5-Coder-0.5B",
    blazedit_max_matching_ngram_size: int = 10,
    blazedit_assistant_confidence_threshold: float | None = None,
    problem_jsonl: str = "",
    skip_eagle_load: bool = False,
    skip_fixed_eagle: bool = False,
    skip_tree: bool = False,
    skip_asts: bool = False,
    skip_eagle2: bool = False,
    retrieval_index: str = "",
    retrieval_draft_len: int = 10,
) -> dict:
    """Lossless verification with the trained EAGLE head."""
    import json
    import os
    import subprocess

    os.chdir("/root/asts-spec")
    output_path = f"/data/{run_tag}/eagle_lossless.json"

    cmd = [
        "python", "scripts/verify_eagle_lossless.py",
        "--target", target,
        "--eagle-checkpoint", eagle_checkpoint,
        "--n", str(n),
        "--max-new-tokens", str(max_new_tokens),
        "--k", str(k),
        "--tree-k", str(tree_k),
        "--tree-w", str(tree_w),
        "--dtype", dtype,
        "--attn-impl", attn_impl,
        "--language", language,
        "--prompt-variant", prompt_variant,
        "--assistant-model", assistant_model,
        "--blazedit-max-matching-ngram-size", str(blazedit_max_matching_ngram_size),
        "--output", output_path,
        "--log-level", "INFO",
    ]
    if blazedit_assistant_confidence_threshold is not None:
        cmd += [
            "--blazedit-assistant-confidence-threshold",
            str(blazedit_assistant_confidence_threshold),
        ]
    if target_trust_remote_code:
        cmd.append("--target-trust-remote-code")
    if strict_determinism:
        cmd.append("--strict-determinism")
    if include_vantage:
        cmd.append("--include-vantage")
    if include_code_proposers:
        cmd.append("--include-code-proposers")
    if code_methods:
        cmd += ["--code-methods", code_methods]
    if problem_jsonl:
        cmd += ["--problem-jsonl", problem_jsonl]
    if skip_eagle_load:
        cmd.append("--skip-eagle-load")
    if skip_fixed_eagle:
        cmd.append("--skip-fixed-eagle")
    if skip_tree:
        cmd.append("--skip-tree")
    if skip_asts:
        cmd.append("--skip-asts")
    if skip_eagle2:
        cmd.append("--skip-eagle2")
    if retrieval_index:
        cmd += ["--retrieval-index", retrieval_index]
    cmd += ["--retrieval-draft-len", str(retrieval_draft_len)]
    print(f"$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=False)
    data_volume.commit()
    hf_cache.commit()

    with open(output_path) as f:
        report = json.load(f)
    return {
        "exit_code": result.returncode,
        "n_match_vf": report["n_match_vf"],
        "n_match_va": report["n_match_va"],
        "n_match_vnh": report.get("n_match_vnh"),
        "n_match_code": report.get("n_match_code"),
        "n_total": report["n_total"],
        "lossless_passed": result.returncode == 0,
        "results": report["results"],
        "output_path": output_path,
    }


# ---------------------------------------------------------------------------
# run_eagle_eval: full HumanEval eval with a trained EAGLE head
# ---------------------------------------------------------------------------


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=7200,
    cpu=4,
)
def run_eagle_eval(
    target: str = "Qwen/Qwen2.5-Coder-7B",
    target_trust_remote_code: bool = False,
    eagle_checkpoint: str = "/data/eagle_v0/eagle/eagle_final.pt",
    n: int = 164,
    max_new_tokens: int = 256,
    k_fixed: str = "4,8",
    methods: str = "vanilla,eagle,asts",
    dtype: str = "bfloat16",
    attn_impl: str = "sdpa",
    language: str = "python",
    prompt_variant: str = "full",
    run_tag: str = "eagle_eval_v0",
    policy: str = "default",
    tree_shapes: str = "2,2",
    retrieval_index: str = "",
    retrieval_draft_len: int = 10,
    router_retrieval_min_match: int = 8,
    router_retrieval_high_match: int = 12,
    router_low_visibility: float = 0.35,
    router_high_visibility: float = 0.72,
    router_tail_margin: float = 0.08,
    router_enable_long_chain: bool = False,
    router_disable_ast_zone: bool = False,
    router_disable_retrieval: bool = False,
    router_disable_scope: bool = False,
    router_disable_rolling: bool = False,
    identifier_max_draft_len: int = 6,
    literal_max_draft_len: int = 8,
    local_suffix_min_match: int = 4,
    local_suffix_max_query_len: int = 16,
    local_suffix_max_draft_len: int = 8,
    alpha_min_match_len: int = 6,
    alpha_max_query_len: int = 24,
    alpha_max_draft_len: int = 8,
    alpha_top_matches: int = 1,
    alpha_enable_roles: bool = True,
    alpha_stop_on_unmapped: bool = True,
    alpha_filter_exact: bool = True,
    alpha_scope_fill: bool = True,
    multisuffix_key_lengths: str = "3,4,5,6,8,12,16",
    multisuffix_top_k: int = 4,
    multisuffix_max_tree_nodes: int = 12,
    multisuffix_max_draft_len: int = 16,
    multisuffix_pool: str = "local",
    codespine_key_lengths: str = "4,5,6,8,12,16",
    codespine_min_match_len: int = 4,
    codespine_max_spine_len: int = 32,
    codespine_max_tree_nodes: int = 12,
    codespine_branch_budget: int = 2,
    codespine_pool: str = "local",
    codespine_allow_short_match: bool = False,
    codespine_enable_identifier_branches: bool = True,
    codespine_enable_delimiter_branches: bool = True,
    edit_anchor_max_draft_len: int = 32,
    edit_anchor_min_chars: int = 12,
    edit_anchor_require_signal: bool = True,
    symbol_tree_branch_budget: int = 4,
    symbol_tree_max_tree_nodes: int = 12,
    symbol_tree_max_symbol_tokens: int = 8,
    symbol_tree_min_prefix_chars: int = 1,
    assistant_model: str = "Qwen/Qwen2.5-Coder-0.5B",
    blazedit_micro_draft_tokens: int = 40,
    blazedit_max_num_run: int = 4,
    blazedit_max_matching_ngram_size: int = 10,
    blazedit_assistant_confidence_threshold: float | None = None,
    problem_jsonl: str = "",
    skip_eagle_load: bool = False,
    macro_chunks_json: str = "",
    code_proposer_fallback: str = "eagle_k2",
    context_tail_widths: str = "default=2,identifier=3,literal=3,margin=3",
) -> dict:
    """Run the full HumanEval eval with a trained EAGLE head."""
    import json
    import os
    import subprocess

    os.chdir("/root/asts-spec")
    output_dir = f"/data/{run_tag}/eval"

    cmd = [
        "python", "scripts/run_eagle_eval.py",
        "--output-dir", output_dir,
        "--target", target,
        "--eagle-checkpoint", eagle_checkpoint,
        "--n", str(n),
        "--max-new-tokens", str(max_new_tokens),
        "--k-fixed", k_fixed,
        "--methods", methods,
        "--dtype", dtype,
        "--attn-impl", attn_impl,
        "--language", language,
        "--prompt-variant", prompt_variant,
        "--policy", policy,
        "--tree-shapes", tree_shapes,
        "--retrieval-draft-len", str(retrieval_draft_len),
        "--router-retrieval-min-match", str(router_retrieval_min_match),
        "--router-retrieval-high-match", str(router_retrieval_high_match),
        "--router-low-visibility", str(router_low_visibility),
        "--router-high-visibility", str(router_high_visibility),
        "--router-tail-margin", str(router_tail_margin),
        "--identifier-max-draft-len", str(identifier_max_draft_len),
        "--literal-max-draft-len", str(literal_max_draft_len),
        "--local-suffix-min-match", str(local_suffix_min_match),
        "--local-suffix-max-query-len", str(local_suffix_max_query_len),
        "--local-suffix-max-draft-len", str(local_suffix_max_draft_len),
        "--alpha-min-match-len", str(alpha_min_match_len),
        "--alpha-max-query-len", str(alpha_max_query_len),
        "--alpha-max-draft-len", str(alpha_max_draft_len),
        "--alpha-top-matches", str(alpha_top_matches),
        "--multisuffix-key-lengths", multisuffix_key_lengths,
        "--multisuffix-top-k", str(multisuffix_top_k),
        "--multisuffix-max-tree-nodes", str(multisuffix_max_tree_nodes),
        "--multisuffix-max-draft-len", str(multisuffix_max_draft_len),
        "--multisuffix-pool", multisuffix_pool,
        "--codespine-key-lengths", codespine_key_lengths,
        "--codespine-min-match-len", str(codespine_min_match_len),
        "--codespine-max-spine-len", str(codespine_max_spine_len),
        "--codespine-max-tree-nodes", str(codespine_max_tree_nodes),
        "--codespine-branch-budget", str(codespine_branch_budget),
        "--codespine-pool", codespine_pool,
        "--edit-anchor-max-draft-len", str(edit_anchor_max_draft_len),
        "--edit-anchor-min-chars", str(edit_anchor_min_chars),
        "--symbol-tree-branch-budget", str(symbol_tree_branch_budget),
        "--symbol-tree-max-tree-nodes", str(symbol_tree_max_tree_nodes),
        "--symbol-tree-max-symbol-tokens", str(symbol_tree_max_symbol_tokens),
        "--symbol-tree-min-prefix-chars", str(symbol_tree_min_prefix_chars),
        "--assistant-model", assistant_model,
        "--blazedit-micro-draft-tokens", str(blazedit_micro_draft_tokens),
        "--blazedit-max-num-run", str(blazedit_max_num_run),
        "--blazedit-max-matching-ngram-size", str(blazedit_max_matching_ngram_size),
        "--code-proposer-fallback", code_proposer_fallback,
        "--context-tail-widths", context_tail_widths,
        "--log-level", "INFO",
    ]
    if problem_jsonl:
        cmd += ["--problem-jsonl", problem_jsonl]
    if skip_eagle_load:
        cmd.append("--skip-eagle-load")
    if target_trust_remote_code:
        cmd.append("--target-trust-remote-code")
    cmd.append("--alpha-enable-roles" if alpha_enable_roles else "--no-alpha-enable-roles")
    cmd.append("--alpha-stop-on-unmapped" if alpha_stop_on_unmapped else "--no-alpha-stop-on-unmapped")
    cmd.append("--alpha-filter-exact" if alpha_filter_exact else "--no-alpha-filter-exact")
    cmd.append("--alpha-scope-fill" if alpha_scope_fill else "--no-alpha-scope-fill")
    cmd.append("--codespine-allow-short-match" if codespine_allow_short_match else "--no-codespine-allow-short-match")
    cmd.append("--codespine-enable-identifier-branches" if codespine_enable_identifier_branches else "--no-codespine-enable-identifier-branches")
    cmd.append("--codespine-enable-delimiter-branches" if codespine_enable_delimiter_branches else "--no-codespine-enable-delimiter-branches")
    cmd.append("--edit-anchor-require-signal" if edit_anchor_require_signal else "--no-edit-anchor-require-signal")
    if blazedit_assistant_confidence_threshold is not None:
        cmd += [
            "--blazedit-assistant-confidence-threshold",
            str(blazedit_assistant_confidence_threshold),
        ]
    if retrieval_index:
        cmd += ["--retrieval-index", retrieval_index]
    if macro_chunks_json:
        cmd += ["--macro-chunks-json", macro_chunks_json]
    if router_enable_long_chain:
        cmd.append("--router-enable-long-chain")
    if router_disable_ast_zone:
        cmd.append("--router-disable-ast-zone")
    if router_disable_retrieval:
        cmd.append("--router-disable-retrieval")
    if router_disable_scope:
        cmd.append("--router-disable-scope")
    if router_disable_rolling:
        cmd.append("--router-disable-rolling")
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    hf_cache.commit()

    with open(f"{output_dir}/aggregate.json") as f:
        agg = json.load(f)
    return {
        "by_method": agg["by_method"],
        "by_node_type_top10": dict(
            sorted(agg["by_node_type"].items(), key=lambda x: -x[1]["n"])[:10]
        ),
        "meta": agg["meta"],
        "output_dir": output_dir,
    }


# ---------------------------------------------------------------------------
# analyze_visibility_frontier: post-process VANTAGE per-step traces
# ---------------------------------------------------------------------------


@app.function(
    volumes={"/data": data_volume},
    cpu=2,
    timeout=1800,
)
def analyze_visibility_frontier(
    run_tag: str,
    method: str = "vantage_full",
    threshold: float = 0.5,
    top_n: int = 20,
    min_support: int = 25,
) -> dict:
    """Build frontier/visibility reports from a run_eagle_eval steps.jsonl."""
    import json
    import os
    import subprocess

    os.chdir("/root/asts-spec")
    steps_path = f"/data/{run_tag}/eval/steps.jsonl"
    output_md = f"/data/{run_tag}/frontier_{method}.md"
    output_json = f"/data/{run_tag}/frontier_{method}.json"

    cmd = [
        "python", "scripts/analyze_visibility_frontier.py",
        "--steps", steps_path,
        "--method", method,
        "--threshold", str(threshold),
        "--top-n", str(top_n),
        "--min-support", str(min_support),
        "--output-md", output_md,
        "--output-json", output_json,
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()

    with open(output_json) as f:
        report = json.load(f)
    return {
        "run_tag": run_tag,
        "method": method,
        "summary": report.get("summary", {}),
        "output_md": output_md,
        "output_json": output_json,
    }


# ---------------------------------------------------------------------------
# evaluate_pass_at_one: post-process saved completions on Python benchmarks
# ---------------------------------------------------------------------------


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    cpu=4,
    timeout=3600,
)
def evaluate_pass_at_one(
    run_tag: str,
    benchmark: str,
    methods: str = "",
    timeout_s: float = 3.0,
) -> dict:
    """Execute Python/MBPP tests against completions from run_eagle_eval."""
    import json
    import os
    import subprocess

    os.chdir("/root/asts-spec")
    completions_path = f"/data/{run_tag}/eval/completions.jsonl"
    output_path = f"/data/{run_tag}/pass_at_one_{benchmark}.json"

    cmd = [
        "python", "scripts/evaluate_pass_at_one.py",
        "--completions", completions_path,
        "--benchmark", benchmark,
        "--timeout-s", str(timeout_s),
        "--output", output_path,
    ]
    if methods:
        cmd += ["--methods", methods]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    hf_cache.commit()

    with open(output_path) as f:
        report = json.load(f)
    return {
        "run_tag": run_tag,
        "benchmark": benchmark,
        "by_method": report.get("by_method", {}),
        "output_path": output_path,
    }


# ---------------------------------------------------------------------------
# bootstrap_speedups: task-bootstrap confidence intervals
# ---------------------------------------------------------------------------


@app.function(
    volumes={"/data": data_volume},
    cpu=2,
    timeout=1800,
)
def bootstrap_speedups(
    run_tag: str,
    methods: str,
    baseline: str = "vanilla",
    pairs: str = "",
    n_boot: int = 5000,
    seed: int = 123,
) -> dict:
    """Bootstrap within-run speedup confidence intervals from completions."""
    import json
    import os
    import subprocess

    os.chdir("/root/asts-spec")
    completions_path = f"/data/{run_tag}/eval/completions.jsonl"
    output_json = f"/data/{run_tag}/bootstrap_speedups.json"
    output_md = f"/data/{run_tag}/bootstrap_speedups.md"

    cmd = [
        "python", "scripts/bootstrap_speedups.py",
        "--completions", completions_path,
        "--methods", methods,
        "--baseline", baseline,
        "--pairs", pairs,
        "--n-boot", str(n_boot),
        "--seed", str(seed),
        "--output-json", output_json,
        "--output-md", output_md,
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()

    with open(output_json) as f:
        report = json.load(f)
    return {
        "run_tag": run_tag,
        "baseline": baseline,
        "by_method": report.get("by_method", {}),
        "by_pair": report.get("by_pair", {}),
        "output_json": output_json,
        "output_md": output_md,
    }


# ---------------------------------------------------------------------------
# analyze_router_overhead: timing component breakdown
# ---------------------------------------------------------------------------


@app.function(
    volumes={"/data": data_volume},
    cpu=2,
    timeout=1800,
)
def analyze_router_overhead(
    run_tag: str,
    methods: str = "",
) -> dict:
    """Break down parse/draft/verify/retrieval/router timing from steps."""
    import json
    import os
    import subprocess

    os.chdir("/root/asts-spec")
    steps_path = f"/data/{run_tag}/eval/steps.jsonl"
    output_json = f"/data/{run_tag}/router_overhead.json"
    output_md = f"/data/{run_tag}/router_overhead.md"

    cmd = [
        "python", "scripts/analyze_router_overhead.py",
        "--steps", steps_path,
        "--methods", methods,
        "--output-json", output_json,
        "--output-md", output_md,
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()

    with open(output_json) as f:
        report = json.load(f)
    return {
        "run_tag": run_tag,
        "methods": report.get("methods", {}),
        "output_json": output_json,
        "output_md": output_md,
    }


# ---------------------------------------------------------------------------
# replay_multisuffix: offline proposer replay against captured completions
# ---------------------------------------------------------------------------


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    cpu=4,
    timeout=3600,
)
def replay_multisuffix(
    run_tag: str,
    completions_path: str = "",
    tokenizer: str = "Qwen/Qwen2.5-Coder-7B",
    methods: str = "suffix,ngram_m4d5,adaptive_suffix,multisuffix_k2,multisuffix_k4,multisuffix_k8",
    max_new_tokens: int = 0,
    max_rows: int = 0,
) -> dict:
    """Replay suffix proposer policies against existing completions without GPU verification."""
    import json
    import os
    import subprocess

    os.chdir("/root/asts-spec")
    completions = completions_path or f"/data/{run_tag}/eval/completions.jsonl"
    output_json = f"/data/{run_tag}/multisuffix_replay.json"
    cmd = [
        "python", "scripts/replay_multisuffix.py",
        "--completions", completions,
        "--tokenizer", tokenizer,
        "--methods", methods,
        "--output-json", output_json,
    ]
    if max_new_tokens > 0:
        cmd += ["--max-new-tokens", str(max_new_tokens)]
    if max_rows > 0:
        cmd += ["--max-rows", str(max_rows)]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    hf_cache.commit()

    with open(output_json) as f:
        report = json.load(f)
    return {
        "run_tag": run_tag,
        "methods": report.get("methods", {}),
        "output_json": output_json,
        "n_rows": report.get("n_rows", 0),
    }


# ---------------------------------------------------------------------------
# analyze_code_proposers: hit-rate and accepted-continuation diagnostics
# ---------------------------------------------------------------------------


@app.function(
    volumes={"/data": data_volume},
    cpu=2,
    timeout=1800,
)
def analyze_code_proposers(
    run_tag: str,
    methods: str = "",
) -> dict:
    """Break down cheap code-proposer hit rate, acceptance, and overhead."""
    import json
    import os
    import subprocess

    os.chdir("/root/asts-spec")
    steps_path = f"/data/{run_tag}/eval/steps.jsonl"
    output_json = f"/data/{run_tag}/code_proposers.json"
    output_md = f"/data/{run_tag}/code_proposers.md"

    cmd = [
        "python", "scripts/analyze_code_proposers.py",
        "--steps", steps_path,
        "--methods", methods,
        "--output-json", output_json,
        "--output-md", output_md,
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()

    with open(output_json) as f:
        report = json.load(f)
    return {
        "run_tag": run_tag,
        "methods": report.get("methods", {}),
        "output_json": output_json,
        "output_md": output_md,
    }


# ---------------------------------------------------------------------------
# analyze_alpha_suffix: renamed code-shape reuse diagnostics
# ---------------------------------------------------------------------------


@app.function(
    volumes={"/data": data_volume},
    cpu=2,
    timeout=1800,
)
def analyze_alpha_suffix(
    run_tag: str,
    methods: str = "",
) -> dict:
    """Break down exact vs alpha-renamed local reuse proposal hits."""
    import json
    import os
    import subprocess

    os.chdir("/root/asts-spec")
    steps_path = f"/data/{run_tag}/eval/steps.jsonl"
    output_json = f"/data/{run_tag}/alpha_suffix.json"
    output_md = f"/data/{run_tag}/alpha_suffix.md"

    cmd = [
        "python", "scripts/analyze_alpha_suffix.py",
        "--steps", steps_path,
        "--methods", methods,
        "--output-json", output_json,
        "--output-md", output_md,
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()

    with open(output_json) as f:
        report = json.load(f)
    return {
        "run_tag": run_tag,
        "methods": report.get("methods", {}),
        "output_json": output_json,
        "output_md": output_md,
    }


# ---------------------------------------------------------------------------
# analyze_suffix_sources: prompt/generated source attribution for local reuse
# ---------------------------------------------------------------------------


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    cpu=2,
    timeout=1800,
)
def analyze_suffix_sources(
    run_tag: str,
    methods: str = "",
    target_tokenizer: str = "Qwen/Qwen2.5-Coder-7B",
) -> dict:
    """Attribute local suffix / n-gram proposal hits to source regions."""
    import json
    import os
    import subprocess

    os.chdir("/root/asts-spec")
    steps_path = f"/data/{run_tag}/eval/steps.jsonl"
    completions_path = f"/data/{run_tag}/eval/completions.jsonl"
    output_json = f"/data/{run_tag}/suffix_sources.json"
    output_md = f"/data/{run_tag}/suffix_sources.md"

    cmd = [
        "python", "scripts/analyze_suffix_sources.py",
        "--steps", steps_path,
        "--completions", completions_path,
        "--target-tokenizer", target_tokenizer,
        "--methods", methods,
        "--output-json", output_json,
        "--output-md", output_md,
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    hf_cache.commit()

    with open(output_json) as f:
        report = json.load(f)
    return {
        "run_tag": run_tag,
        "methods": report.get("methods", {}),
        "output_json": output_json,
        "output_md": output_md,
    }


# ---------------------------------------------------------------------------
# analyze_anchor_pld_overlap: EditAnchor tokens unique beyond PLD
# ---------------------------------------------------------------------------


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    cpu=2,
    timeout=1800,
)
def analyze_anchor_pld_overlap(
    run_tag: str,
    method: str,
    target_tokenizer: str = "Qwen/Qwen2.5-Coder-7B",
    pld_window: int = 40,
    pld_ngram: int = 10,
) -> dict:
    """Measure whether EditAnchor accepted tokens were also available to PLD."""
    import json
    import os
    import subprocess

    os.chdir("/root/asts-spec")
    steps_path = f"/data/{run_tag}/eval/steps.jsonl"
    completions_path = f"/data/{run_tag}/eval/completions.jsonl"
    output_json = f"/data/{run_tag}/anchor_pld_overlap_{method}.json"
    output_md = f"/data/{run_tag}/anchor_pld_overlap_{method}.md"

    cmd = [
        "python", "scripts/analyze_anchor_pld_overlap.py",
        "--steps", steps_path,
        "--completions", completions_path,
        "--method", method,
        "--target-tokenizer", target_tokenizer,
        "--pld-window", str(pld_window),
        "--pld-ngram", str(pld_ngram),
        "--output-json", output_json,
        "--output-md", output_md,
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    hf_cache.commit()

    with open(output_json) as f:
        report = json.load(f)
    return {
        "run_tag": run_tag,
        "method": method,
        "totals": report.get("totals", {}),
        "fractions": report.get("fractions", {}),
        "output_json": output_json,
        "output_md": output_md,
    }


# ---------------------------------------------------------------------------
# analyze_edit_workload: copy-ratio and edit-distance characterization
# ---------------------------------------------------------------------------


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    cpu=2,
    timeout=1800,
)
def analyze_edit_workload(
    run_tag: str,
    method: str = "vanilla",
    target_tokenizer: str = "Qwen/Qwen2.5-Coder-7B",
    target_mode: str = "auto",
) -> dict:
    """Measure copied-token ratio, edit distance, hunks, and span length."""
    import json
    import os
    import subprocess

    os.chdir("/root/asts-spec")
    completions_path = f"/data/{run_tag}/eval/completions.jsonl"
    safe_method = method.replace("/", "_").replace(":", "_")
    output_json = f"/data/{run_tag}/edit_workload_{safe_method}.json"
    output_md = f"/data/{run_tag}/edit_workload_{safe_method}.md"

    cmd = [
        "python", "scripts/analyze_edit_workload.py",
        "--completions", completions_path,
        "--method", method,
        "--target-mode", target_mode,
        "--target-tokenizer", target_tokenizer,
        "--output-json", output_json,
        "--output-md", output_md,
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    hf_cache.commit()

    with open(output_json) as f:
        report = json.load(f)
    return {
        "run_tag": run_tag,
        "method": method,
        "target_mode": target_mode,
        "n_rows": report.get("n_rows"),
        "aggregate": report.get("aggregate", {}),
        "output_json": output_json,
        "output_md": output_md,
    }


# ---------------------------------------------------------------------------
# New VANTAGE edit-drift postprocessors
# ---------------------------------------------------------------------------


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    cpu=2,
    timeout=1800,
)
def analyze_drift_sweep(
    run_tag: str,
    pld_method: str,
    vantage_method: str,
) -> dict:
    import json
    import os
    import subprocess

    os.chdir("/root/asts-spec")
    completions_path = f"/data/{run_tag}/eval/completions.jsonl"
    output_json = f"/data/{run_tag}/drift_sweep.json"
    output_md = f"/data/{run_tag}/drift_sweep.md"
    cmd = [
        "python", "scripts/analyze_drift_sweep.py",
        "--completions", completions_path,
        "--pld-method", pld_method,
        "--vantage-method", vantage_method,
        "--output-json", output_json,
        "--output-md", output_md,
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    with open(output_json) as f:
        report = json.load(f)
    return {
        "run_tag": run_tag,
        "n_tasks": report.get("n_tasks"),
        "overall_ratio_mean": report.get("overall_ratio_mean"),
        "output_json": output_json,
        "output_md": output_md,
    }


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    cpu=2,
    timeout=1800,
)
def analyze_pld_recovery_curve(
    run_tag: str,
    pld_method: str,
    anchor_method: str,
    target_tokenizer: str = "Qwen/Qwen2.5-Coder-7B",
    pld_window: int = 128,
    pld_ngram: int = 10,
    rewrite_pld_method: str = "",
) -> dict:
    import json
    import os
    import subprocess

    os.chdir("/root/asts-spec")
    completions_path = f"/data/{run_tag}/eval/completions.jsonl"
    steps_path = f"/data/{run_tag}/eval/steps.jsonl"
    output_json = f"/data/{run_tag}/pld_recovery.json"
    output_md = f"/data/{run_tag}/pld_recovery.md"
    cmd = [
        "python", "scripts/analyze_pld_recovery_curve.py",
        "--completions", completions_path,
        "--steps", steps_path,
        "--pld-method", pld_method,
        "--anchor-method", anchor_method,
        "--target-tokenizer", target_tokenizer,
        "--pld-window", str(pld_window),
        "--pld-ngram", str(pld_ngram),
        "--output-json", output_json,
        "--output-md", output_md,
    ]
    if rewrite_pld_method:
        cmd += ["--rewrite-pld-method", rewrite_pld_method]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    hf_cache.commit()
    with open(output_json) as f:
        report = json.load(f)
    return {
        "run_tag": run_tag,
        "mean_speedup_vs_pld": report.get("mean_speedup_vs_pld"),
        "pld": report.get("pld", {}),
        "anchor": report.get("anchor", {}),
        "output_json": output_json,
        "output_md": output_md,
    }


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    cpu=2,
    timeout=1800,
)
def analyze_prompt_oracle(
    run_tag: str,
    methods: str,
) -> dict:
    import json
    import os
    import subprocess

    os.chdir("/root/asts-spec")
    completions_path = f"/data/{run_tag}/eval/completions.jsonl"
    output_json = f"/data/{run_tag}/prompt_oracle.json"
    output_md = f"/data/{run_tag}/prompt_oracle.md"
    cmd = [
        "python", "scripts/analyze_prompt_oracle.py",
        "--completions", completions_path,
        "--methods", methods,
        "--output-json", output_json,
        "--output-md", output_md,
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    with open(output_json) as f:
        report = json.load(f)
    return {"run_tag": run_tag, "groups": report.get("groups", []), "output_json": output_json, "output_md": output_md}


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    cpu=2,
    timeout=1800,
)
def analyze_latency(
    run_tag: str,
    methods: str,
) -> dict:
    import json
    import os
    import subprocess

    os.chdir("/root/asts-spec")
    completions_path = f"/data/{run_tag}/eval/completions.jsonl"
    steps_path = f"/data/{run_tag}/eval/steps.jsonl"
    output_json = f"/data/{run_tag}/latency.json"
    output_md = f"/data/{run_tag}/latency.md"
    cmd = [
        "python", "scripts/analyze_latency.py",
        "--completions", completions_path,
        "--steps", steps_path,
        "--methods", methods,
        "--output-json", output_json,
        "--output-md", output_md,
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    with open(output_json) as f:
        report = json.load(f)
    return {"run_tag": run_tag, "groups": report.get("groups", []), "output_json": output_json, "output_md": output_md}


@app.function(
    volumes={"/data": data_volume},
    cpu=2,
    timeout=1800,
)
def analyze_zero_drift_overhead(
    run_tag: str,
    methods: str,
    vanilla_method: str = "vanilla",
) -> dict:
    import json
    import os
    import subprocess

    os.chdir("/root/asts-spec")
    completions_path = f"/data/{run_tag}/eval/completions.jsonl"
    steps_path = f"/data/{run_tag}/eval/steps.jsonl"
    output_json = f"/data/{run_tag}/zero_drift_overhead.json"
    output_md = f"/data/{run_tag}/zero_drift_overhead.md"
    cmd = [
        "python", "scripts/analyze_zero_drift_overhead.py",
        "--completions", completions_path,
        "--steps", steps_path,
        "--methods", methods,
        "--vanilla-method", vanilla_method,
        "--output-json", output_json,
        "--output-md", output_md,
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    with open(output_json) as f:
        report = json.load(f)
    return {
        "run_tag": run_tag,
        "methods": report.get("methods", {}),
        "output_json": output_json,
        "output_md": output_md,
    }


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    cpu=2,
    timeout=1800,
)
def evaluate_codeeditor_quality(
    run_tag: str,
    methods: str,
) -> dict:
    import json
    import os
    import subprocess

    os.chdir("/root/asts-spec")
    completions_path = f"/data/{run_tag}/eval/completions.jsonl"
    output_json = f"/data/{run_tag}/codeeditor_quality.json"
    output_md = f"/data/{run_tag}/codeeditor_quality.md"
    cmd = [
        "python", "scripts/evaluate_codeeditor_quality.py",
        "--completions", completions_path,
        "--methods", methods,
        "--output-json", output_json,
        "--output-md", output_md,
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    with open(output_json) as f:
        report = json.load(f)
    return {"run_tag": run_tag, "groups": report.get("groups", []), "output_json": output_json, "output_md": output_md}


# ---------------------------------------------------------------------------
# mine_macro_chunks: optional external-corpus macro chunks
# ---------------------------------------------------------------------------


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    cpu=4,
    memory=16384,
    timeout=7200,
)
def mine_macro_chunks(
    output_tag: str = "macro_chunks_v0",
    target: str = "Qwen/Qwen2.5-Coder-7B",
    corpus: str = "codeparrot/codeparrot-clean",
    config: str = "",
    data_dir: str = "",
    language_filter: str = "",
    n_samples: int = 1000,
    top_k: int = 256,
) -> dict:
    """Mine frequent token chunks from a disjoint external code corpus."""
    import json
    import os
    import subprocess

    os.chdir("/root/asts-spec")
    output_json = f"/data/{output_tag}/macro_chunks.json"
    cmd = [
        "python", "scripts/mine_macro_chunks.py",
        "--target", target,
        "--corpus", corpus,
        "--n-samples", str(n_samples),
        "--top-k", str(top_k),
        "--output-json", output_json,
    ]
    if config:
        cmd += ["--config", config]
    if data_dir:
        cmd += ["--data-dir", data_dir]
    if language_filter:
        cmd += ["--language-filter", language_filter]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()
    hf_cache.commit()

    with open(output_json) as f:
        report = json.load(f)
    return {
        "output_json": output_json,
        "n_samples": report.get("n_samples"),
        "n_chunks": len(report.get("chunks", [])),
        "top_chunks": report.get("chunks", [])[:20],
    }


# ---------------------------------------------------------------------------
# analyze_oracle_routing: per-task best-method upper bound
# ---------------------------------------------------------------------------


@app.function(
    volumes={"/data": data_volume},
    cpu=2,
    timeout=1800,
)
def analyze_oracle_routing(
    run_tag: str,
    methods: str,
    baseline: str = "vanilla",
) -> dict:
    """Compute an oracle that chooses the fastest method per task."""
    import json
    import os
    import subprocess

    os.chdir("/root/asts-spec")
    completions_path = f"/data/{run_tag}/eval/completions.jsonl"
    output_json = f"/data/{run_tag}/oracle_routing.json"
    output_md = f"/data/{run_tag}/oracle_routing.md"

    cmd = [
        "python", "scripts/analyze_oracle_routing.py",
        "--completions", completions_path,
        "--methods", methods,
        "--baseline", baseline,
        "--output-json", output_json,
        "--output-md", output_md,
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    data_volume.commit()

    with open(output_json) as f:
        report = json.load(f)
    return {
        "run_tag": run_tag,
        "oracle": report.get("oracle", {}),
        "by_method": report.get("by_method", {}),
        "output_json": output_json,
        "output_md": output_md,
    }


# ---------------------------------------------------------------------------
# build_retrieval_index: tokenize a Python code corpus with the target's
# tokenizer and build a suffix array for retrieval-based drafting.
# ---------------------------------------------------------------------------


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    cpu=8,
    memory=32768,  # 32GB; suffix array build needs headroom
    timeout=7200,  # 2h
)
def build_retrieval_index(
    target: str = "Qwen/Qwen2.5-Coder-7B",
    corpus: str = "codeparrot/codeparrot-clean",
    data_dir: str = "",
    config: str = "",
    language_filter: str = "",
    n_samples: int = 100000,
    max_chars_per_sample: int = 8000,
    output_tag: str = "retrieval_v0",
) -> dict:
    """Tokenize a Python corpus with the target tokenizer, then build a
    suffix array over the concatenated token IDs.

    Saves three artifacts to /data/{output_tag}/:
      - tokens.npy        — flat int32 array of all token IDs (with
                            EOS-token separators between samples)
      - suffix_array.npy  — int32 array of suffix start positions, sorted
                            lexicographically over `tokens`
      - meta.json         — corpus name, n_samples, n_tokens, target, etc.

    Build cost: ~10-30 min depending on corpus size. n_samples=100K Python
    files ≈ 20M tokens.
    """
    import json as _json
    import os
    import time

    import numpy as np
    from datasets import load_dataset
    from transformers import AutoTokenizer

    output_dir = f"/data/{output_tag}"
    os.makedirs(output_dir, exist_ok=True)

    # Authenticate with HF if a token is in the env (used for gated corpora).
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        from huggingface_hub import login as hf_login
        hf_login(token=hf_token)
        print(f"[auth] hf logged in (token len={len(hf_token)})", flush=True)
    else:
        print("[auth] no HF_TOKEN in env (gated datasets will fail)", flush=True)

    print(f"[load] target tokenizer = {target}", flush=True)
    tok = AutoTokenizer.from_pretrained(target)
    sep = int(tok.eos_token_id) if tok.eos_token_id is not None else 0

    print(
        f"[load] corpus = {corpus} "
        f"(data_dir={data_dir!r}, config={config!r}, "
        f"language_filter={language_filter!r}, "
        f"streaming, sampling {n_samples} entries)",
        flush=True,
    )
    load_kwargs: dict = {"split": "train", "streaming": True}
    if data_dir:
        load_kwargs["data_dir"] = data_dir
    if config:
        load_kwargs["name"] = config
    ds = load_dataset(corpus, **load_kwargs)
    if language_filter:
        ds = ds.filter(lambda r: r.get("language") == language_filter)

    # Tokenize streaming so we don't OOM on huge corpora.
    t0 = time.perf_counter_ns()
    chunks: list[np.ndarray] = []
    n_done = 0
    n_tokens_total = 0
    for row in ds:
        if n_done >= n_samples:
            break
        # Most code datasets store the file under "content" (codeparrot uses this),
        # The Stack uses "content" too. Instruction datasets often use "output".
        # Fall back across common keys.
        text = (
            row.get("content")
            or row.get("text")
            or row.get("code")
            or row.get("output")
            or row.get("response")
            or ""
        )
        if not text:
            continue
        if len(text) > max_chars_per_sample:
            text = text[:max_chars_per_sample]
        ids = tok(text, add_special_tokens=False).input_ids
        if not ids:
            continue
        # Append separator so that retrieval doesn't span sample boundaries.
        chunks.append(np.asarray(ids + [sep], dtype=np.int32))
        n_tokens_total += len(ids) + 1
        n_done += 1
        if n_done % 5000 == 0:
            elapsed = (time.perf_counter_ns() - t0) / 1e9
            print(
                f"[tok] {n_done}/{n_samples} samples, "
                f"{n_tokens_total} tokens, {n_tokens_total / elapsed:.0f} tok/s",
                flush=True,
            )
    print(
        f"[tok] done: {n_done} samples, {n_tokens_total} tokens",
        flush=True,
    )

    tokens = np.concatenate(chunks)
    chunks = []  # free memory
    print(f"[tok] concatenated → tokens.shape = {tokens.shape}", flush=True)
    np.save(f"{output_dir}/tokens.npy", tokens)
    print(f"[save] {output_dir}/tokens.npy ({tokens.nbytes / 1e6:.1f} MB)", flush=True)

    # Suffix array build via pydivsufsort (O(N), C-extension, fast).
    print("[sa] building suffix array (pydivsufsort)...", flush=True)
    t_sa = time.perf_counter_ns()
    import pydivsufsort
    # pydivsufsort.divsufsort accepts uint8 array natively; for int32 we use
    # divsufsort_int (handles arbitrary integer alphabets).
    if hasattr(pydivsufsort, "divsufsort_int"):
        sa = pydivsufsort.divsufsort_int(tokens)
    else:
        # Fallback: cast to int32 and use the generic interface
        sa = pydivsufsort.divsufsort(tokens.astype(np.int32))
    sa = np.asarray(sa, dtype=np.int32)
    print(f"[sa] built suffix array in {(time.perf_counter_ns() - t_sa) / 1e9:.1f}s", flush=True)
    np.save(f"{output_dir}/suffix_array.npy", sa)
    print(f"[save] {output_dir}/suffix_array.npy ({sa.nbytes / 1e6:.1f} MB)", flush=True)

    meta = {
        "schema": "asts-spec/retrieval_index/v1",
        "target": target,
        "corpus": corpus,
        "data_dir": data_dir or None,
        "config": config or None,
        "language_filter": language_filter or None,
        "n_samples": n_done,
        "n_tokens": int(tokens.shape[0]),
        "sep_token_id": sep,
        "max_chars_per_sample": max_chars_per_sample,
        "output_tag": output_tag,
        "build_wall_s": (time.perf_counter_ns() - t0) / 1e9,
    }
    with open(f"{output_dir}/meta.json", "w") as f:
        _json.dump(meta, f, indent=2)
    print(f"[done] meta = {meta}", flush=True)

    data_volume.commit()
    return meta


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    n: int = 3,
    max_new_tokens: int = 32,
    k: int = 4,
    target: str = "Qwen/Qwen2.5-Coder-7B",
    draft: str = "Qwen/Qwen2.5-Coder-0.5B",
    dtype: str = "bfloat16",
    attn_impl: str = "sdpa",
    strict_determinism: bool = True,
    run_tag: str = DEFAULT_RUN_TAG,
) -> None:
    """Run the lossless smoke test by default. Use modal run proto_app.py::run_eval for full eval."""
    print()
    print("=== ASTS-Spec lossless smoke test ===")
    print(f"  target:          {target}")
    print(f"  draft:           {draft}")
    print(f"  n_problems:      {n}")
    print(f"  max_new_tokens:  {max_new_tokens}")
    print(f"  fixed_spec_k:    {k}")
    print(f"  attn_impl:       {attn_impl}")
    print(f"  determinism:     {'strict' if strict_determinism else 'default'}")
    print()

    res = verify_lossless.remote(
        target=target,
        draft=draft,
        n=n,
        max_new_tokens=max_new_tokens,
        k=k,
        dtype=dtype,
        attn_impl=attn_impl,
        strict_determinism=strict_determinism,
        run_tag=run_tag,
    )

    print()
    print("=== Result ===")
    print(f"  vanilla == fixed_k{k}:  {res['n_match_vf']}/{res['n_total']}")
    print(f"  vanilla == asts_spec:    {res['n_match_va']}/{res['n_total']}")
    print(f"  output:                  {res['output_path']}")
    if res["lossless_passed"]:
        print("  ✓ LOSSLESS HOLDS — safe to proceed to full eval")
    else:
        print("  ✗ DIVERGENCE — debug before running full eval")
    print()
