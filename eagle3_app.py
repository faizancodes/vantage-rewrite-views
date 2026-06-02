"""Modal app for EAGLE-3 training via SpecForge.

Separate from proto_app.py because SpecForge requires:
  - Python 3.11
  - torch==2.9.1 (vs our 2.x in proto_app)
  - transformers==4.57.1 (vs our 4.46+ in proto_app)
  - sglang==0.5.9 (heavy; we use --target-model-backend hf to avoid runtime use)
  - SpecForge cloned + pip installed

Workflow:
  1. `modal run eagle3_app.py::verify_install` — import-check the image
  2. `modal run eagle3_app.py::smoke_train` — small training run (~1K samples)
  3. `modal run eagle3_app.py::production_train` — full training (~50K samples)
  4. `modal volume get asts-spec-data eagle3_v0/...` — pull checkpoint locally
"""

from __future__ import annotations

from pathlib import Path

import modal


_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_TAG = "eagle3_v0"


# ---------------------------------------------------------------------------
# Image: SpecForge stack. We deliberately use the CUDA devel base we already
# verified for flash-attn (and we'll need it again because torch 2.9.1 +
# sglang pull CUDA-bound wheels that may need nvcc for fallback builds).
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "git",
        "build-essential",
        "clang",
        # libnuma1 is required at runtime by sglang's C extensions;
        # without it `import specforge` fails with libnuma.so.1 not found.
        "libnuma1",
        "libnuma-dev",
    )
    .env({"CUDA_HOME": "/usr/local/cuda", "MAX_JOBS": "2", "PYTHONUNBUFFERED": "1"})
    # SpecForge prefers source install. Clone + install at image build time so
    # the heavy deps land in a layer we can cache.
    .run_commands(
        "git clone --depth 1 https://github.com/sgl-project/SpecForge.git /opt/SpecForge",
    )
    # Install SpecForge with sglang/flash-attn made optional. The "fa" extra
    # is opt-in and we skip it (flex_attention is the training default).
    # SpecForge docs use `uv pip --prerelease=allow`; standard pip uses `--pre`.
    .run_commands(
        "cd /opt/SpecForge && pip install -v --pre .",
    )
    # Our own deps for tree-sitter + dataset access. Keep these AFTER the heavy
    # SpecForge install so version conflicts surface immediately.
    .pip_install(
        "tree-sitter>=0.23.0",
        "tree-sitter-language-pack>=0.4.0",
        "datasets>=3.0",
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
        "TOKENIZERS_PARALLELISM": "false",
        "HF_DATASETS_CACHE": "/cache/huggingface/datasets",
    })
)


# Reuse the same volumes as proto_app.
data_volume = modal.Volume.from_name("asts-spec-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("asts-spec-hf-cache", create_if_missing=True)


app = modal.App("asts-spec-eagle3", image=image)


# ---------------------------------------------------------------------------
# Step 1: import sanity check (no GPU, fast)
# ---------------------------------------------------------------------------


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",  # SpecForge initializes CUDA at import time
    cpu=2,
    timeout=300,
)
def verify_install() -> dict:
    """Check that the image's imports work. Runs on a GPU because SpecForge
    initializes CUDA at import time (driver lookup fails on CPU-only)."""
    import importlib
    import sys

    info = {"python": sys.version}

    for pkg in ("specforge", "transformers", "torch", "datasets", "tree_sitter"):
        try:
            mod = importlib.import_module(pkg)
            info[pkg] = getattr(mod, "__version__", "unknown")
        except Exception as e:
            info[pkg] = f"FAIL: {e}"

    # Try to load the AutoEagle3DraftModel symbol (a key entrypoint for our
    # eventual inference adapter)
    try:
        from specforge import AutoEagle3DraftModel  # noqa: F401
        info["AutoEagle3DraftModel"] = "importable"
    except Exception as e:
        info["AutoEagle3DraftModel"] = f"FAIL: {e}"

    # Check that the configs/qwen2.5-7b-eagle3.json file ships in the install
    import os
    config_paths = [
        "/opt/SpecForge/configs/qwen2.5-7b-eagle3.json",
        "/opt/SpecForge/configs/qwen2.5_7b_eagle3.json",
    ]
    for p in config_paths:
        if os.path.exists(p):
            info["qwen_config_path"] = p
            break
    else:
        info["qwen_config_path"] = "MISSING"

    print()
    print("=== SpecForge install check ===")
    for k, v in info.items():
        print(f"  {k:<28}  {v}")
    print()
    return info


# ---------------------------------------------------------------------------
# Diagnostic: dump SpecForge script help text to discover exact arg names
# ---------------------------------------------------------------------------


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",  # train_eagle3.py / prepare_hidden_states.py need CUDA at import
    cpu=2,
    timeout=300,
)
def dump_help() -> dict:
    """Print --help output for SpecForge scripts so we know the real arg names."""
    import os
    import subprocess

    os.chdir("/opt/SpecForge")
    out = {}
    for script in ("prepare_data.py", "train_eagle3.py", "prepare_hidden_states.py"):
        path = f"scripts/{script}"
        if not os.path.isfile(path):
            out[script] = "MISSING"
            continue
        try:
            res = subprocess.run(
                ["python", path, "--help"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            out[script] = (res.stdout or "")[:6000]
            if res.returncode != 0 and not out[script]:
                out[script] = f"FAIL: {res.stderr[:1000]}"
        except Exception as e:
            out[script] = f"FAIL: {e}"
    for k, v in out.items():
        print(f"\n========== {k} ==========\n{v}\n", flush=True)
    return out


# ---------------------------------------------------------------------------
# Step 2: tiny smoke training to validate pipeline end-to-end
# ---------------------------------------------------------------------------


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="H100",
    timeout=21600,  # 6h — production runs at 20K steps need ~3h; smoke uses tiny n_samples
    cpu=4,
)
def smoke_train(
    target: str = "Qwen/Qwen2.5-Coder-7B",
    dataset: str = "codealpaca-20k",
    n_samples: int = 200,
    epochs: int = 1,
    max_length: int = 512,
    ttt_length: int = 4,
    lr: float = 1e-4,
    run_tag: str = "eagle3_smoke",
    attention_backend: str = "flex_attention",
    save_interval: int = 100,
    resume_from: str = "",
) -> dict:
    """Tiny SpecForge training to validate the pipeline end-to-end.

    Uses codealpaca-20k by default (small + code-themed). prepare_data.py
    has no --num-samples flag; we truncate the produced JSONL afterwards.
    Uses --target-model-backend hf to avoid sglang server runtime.

    GPU is H100 because L40S's 101KB shared memory is below the
    ~106KB that flex_attention's compiled triton kernel needs.
    H100 has ~228KB shared mem per SM and trains comfortably.
    """
    import os
    import subprocess

    os.chdir("/opt/SpecForge")
    output_dir = f"/data/{run_tag}/eagle3"
    os.makedirs(output_dir, exist_ok=True)

    output_path_arg = f"./cache/dataset/{dataset}"
    smoke_jsonl = f"./cache/dataset/{dataset}_smoke{n_samples}.jsonl"
    os.makedirs("./cache/dataset", exist_ok=True)

    # Step 1: prepare data — SpecForge treats --output-path as a directory
    # and writes one or more jsonl files inside.
    prepare_cmd = [
        "python", "scripts/prepare_data.py",
        "--dataset", dataset,
        "--output-path", output_path_arg,
    ]
    print(f"$ {' '.join(prepare_cmd)}", flush=True)
    subprocess.run(prepare_cmd, check=True)

    # Find the produced jsonl. SpecForge writes either a directory with
    # multiple files or a flat .jsonl — handle both.
    candidates = []
    if os.path.isfile(output_path_arg):
        candidates.append(output_path_arg)
    elif os.path.isdir(output_path_arg):
        for root, _dirs, files in os.walk(output_path_arg):
            for f in files:
                if f.endswith(".jsonl") or f.endswith(".json"):
                    candidates.append(os.path.join(root, f))
    print(f"[ok] candidate data files:", flush=True)
    for c in candidates:
        sz = os.path.getsize(c) if os.path.isfile(c) else "dir"
        print(f"    {c}  ({sz} bytes)", flush=True)
    if not candidates:
        raise RuntimeError(f"prepare_data did not produce any *.jsonl in {output_path_arg}")

    # Step 2: truncate to N samples for smoke. Concatenate all candidates.
    print(f"[ok] truncating to {n_samples} samples → {smoke_jsonl}", flush=True)
    written = 0
    with open(smoke_jsonl, "w") as fout:
        for src in candidates:
            if not os.path.isfile(src) or not src.endswith(".jsonl"):
                continue
            with open(src) as fin:
                for line in fin:
                    if written >= n_samples:
                        break
                    fout.write(line)
                    written += 1
            if written >= n_samples:
                break
    print(f"[ok] wrote {written} samples to {smoke_jsonl}", flush=True)

    # Step 3: train
    config_path = "/opt/SpecForge/configs/qwen2.5-7b-eagle3.json"
    if not os.path.isfile(config_path):
        raise RuntimeError(f"missing draft config: {config_path}")

    train_cmd = [
        "torchrun", "--standalone", "--nproc_per_node", "1",
        "scripts/train_eagle3.py",
        "--target-model-path", target,
        "--draft-model-config", config_path,
        "--train-data-path", smoke_jsonl,
        "--output-dir", output_dir,
        "--num-epochs", str(epochs),
        "--batch-size", "1",
        "--learning-rate", str(lr),
        "--max-length", str(max_length),
        "--chat-template", "qwen",
        "--tp-size", "1",
        "--target-model-backend", "hf",
        "--ttt-length", str(ttt_length),
        "--attention-backend", attention_backend,
        "--embedding-key", "model.embed_tokens.weight",
        "--save-interval", str(save_interval),
    ]
    if resume_from:
        if not os.path.isdir(resume_from):
            raise RuntimeError(f"resume_from directory not found: {resume_from}")
        train_cmd += ["--resume", "--ckpt-dir", resume_from]
        print(f"[resume] resuming from {resume_from}", flush=True)
    print(f"$ {' '.join(train_cmd)}", flush=True)
    subprocess.run(train_cmd, check=True)

    data_volume.commit()
    hf_cache.commit()

    # Find saved checkpoints
    ckpts = []
    for root, _dirs, files in os.walk(output_dir):
        if "model.safetensors" in files or any(
            f.startswith("model-") and f.endswith(".safetensors") for f in files
        ):
            ckpts.append(root)

    return {
        "output_dir": output_dir,
        "train_data_path": smoke_jsonl,
        "n_samples": n_samples,
        "checkpoints": ckpts,
    }


# ---------------------------------------------------------------------------
# Step 3a: Introspect a saved checkpoint so we can write the inference adapter
# without having to guess the SpecForge API.
# ---------------------------------------------------------------------------


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    cpu=2,
    timeout=600,
)
def inspect_draft(
    ckpt_path: str = "/data/eagle3_smoke/eagle3/epoch_0_step_200",
) -> dict:
    """Load a saved EAGLE-3 draft and print enough structure to write an
    inference adapter against it.

    What we want to know:
      - config.json contents (target_hidden_size, draft_vocab_size, num_layers,
        which target layers are fused, etc.)
      - which submodules exist on the loaded module (fc, midlayer, embed_tokens,
        norm, lm_head?, t2d, d2t)
      - parameter shapes for fc and lm_head/d2t (so we know vocab mapping size)
      - forward signature(s) we are expected to call
    """
    import json
    import inspect as ipy
    import os

    info: dict = {"ckpt_path": ckpt_path}

    config_file = os.path.join(ckpt_path, "config.json")
    if os.path.isfile(config_file):
        with open(config_file) as fh:
            info["config.json"] = json.load(fh)
    else:
        info["config.json"] = "MISSING"

    files = []
    if os.path.isdir(ckpt_path):
        for f in sorted(os.listdir(ckpt_path)):
            p = os.path.join(ckpt_path, f)
            sz = os.path.getsize(p) if os.path.isfile(p) else "dir"
            files.append((f, sz))
    info["files"] = files

    # Now load via SpecForge's AutoEagle3DraftModel (the entrypoint we'll use
    # for inference) and dump structure.
    try:
        from specforge import AutoEagle3DraftModel  # type: ignore
    except Exception as e:
        info["AutoEagle3DraftModel"] = f"FAIL import: {e}"
        return info

    try:
        draft = AutoEagle3DraftModel.from_pretrained(ckpt_path)
        info["draft_class"] = type(draft).__name__
        info["draft_module_path"] = type(draft).__module__
    except Exception as e:
        info["from_pretrained"] = f"FAIL: {e}"
        return info

    # Top-level attributes / submodules
    submods = []
    for name, mod in draft.named_children():
        submods.append((name, type(mod).__name__))
    info["named_children"] = submods

    # Parameter shapes — particularly fc (multi-layer fusion) and any d2t/t2d
    # buffers (vocab mapping).
    param_shapes = {}
    for name, p in draft.named_parameters():
        param_shapes[name] = list(p.shape)
    buf_shapes = {}
    for name, b in draft.named_buffers():
        buf_shapes[name] = list(b.shape)
    info["param_shapes"] = param_shapes
    info["buf_shapes"] = buf_shapes

    # Forward signature
    try:
        sig = ipy.signature(draft.forward)
        info["forward_signature"] = str(sig)
    except Exception as e:
        info["forward_signature"] = f"FAIL: {e}"

    # If the draft has any documented "compute_logits" or "fc" or "backbone"
    # convenience method, capture their signatures too.
    for meth_name in ("backbone", "compute_logits", "fc", "embed_tokens", "midlayer"):
        attr = getattr(draft, meth_name, None)
        if attr is None:
            continue
        if callable(attr):
            try:
                info[f"{meth_name}_signature"] = str(ipy.signature(attr))
            except Exception:
                info[f"{meth_name}_signature"] = "[no signature]"
        else:
            # Likely a submodule
            info[f"{meth_name}_type"] = type(attr).__name__

    # Pretty-print essentials.
    print("\n=== EAGLE-3 draft introspection ===\n")
    print(f"ckpt: {ckpt_path}")
    print(f"class: {info.get('draft_class')}  ({info.get('draft_module_path')})")
    print("\nfiles:")
    for f, sz in files:
        print(f"  {f}  ({sz})")
    print("\nconfig.json:")
    print(json.dumps(info["config.json"], indent=2))
    print("\nnamed_children:")
    for name, t in submods:
        print(f"  {name}  →  {t}")
    print("\nparam shapes (first 30):")
    for name, shape in list(param_shapes.items())[:30]:
        print(f"  {name:<60}  {shape}")
    if len(param_shapes) > 30:
        print(f"  ... and {len(param_shapes) - 30} more params")
    print("\nbuffer shapes:")
    for name, shape in buf_shapes.items():
        print(f"  {name:<60}  {shape}")
    print(f"\nforward signature: {info.get('forward_signature')}")
    for k, v in info.items():
        if k.endswith("_signature") and k != "forward_signature":
            print(f"{k}: {v}")
    return info


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    cpu=2,
    timeout=600,
)
def verify_d2t_mapping(
    ckpt_path: str = "/data/eagle3_v0_prod/eagle3/epoch_0_step_7000",
    target: str = "Qwen/Qwen2.5-Coder-7B",
) -> dict:
    """Verify the d2t formula. We need: target_id = f(draft_idx, d2t[draft_idx]).
    Test if it's additive offset or absolute mapping by round-tripping via t2d.

    Specifically: if t2d[target_id] = draft_idx for some tokens, then
    formula(draft_idx, d2t[draft_idx]) MUST equal target_id.
    """
    import torch
    from transformers import AutoTokenizer
    from specforge import AutoEagle3DraftModel  # type: ignore

    tok = AutoTokenizer.from_pretrained(target)
    draft = AutoEagle3DraftModel.from_pretrained(ckpt_path)

    t2d = draft.t2d  # [vocab_size=152064]
    d2t = draft.d2t  # [draft_vocab_size=16000]

    print(f"t2d shape: {t2d.shape}, dtype: {t2d.dtype}")
    print(f"d2t shape: {d2t.shape}, dtype: {d2t.dtype}")
    print(f"t2d unique values (first 20): {torch.unique(t2d)[:20]}")
    print(f"t2d max: {t2d.max()}, min: {t2d.min()}")
    print(f"d2t max: {d2t.max()}, min: {d2t.min()}")
    print(f"d2t first 20 values: {d2t[:20].tolist()}")
    print(f"d2t last 20 values: {d2t[-20:].tolist()}")

    # t2d is bool? or int?
    if t2d.dtype == torch.bool:
        # t2d[target_id] = True means token is in draft vocab
        in_vocab_count = int(t2d.sum().item())
        print(f"t2d is BOOL: {in_vocab_count} target tokens are in draft vocab")
        print("This means d2t alone determines the mapping (not via t2d index).")
        print("Formula is likely: target_id = draft_idx + d2t[draft_idx]")
        # Test additive offset hypothesis
        # draft_idx 0 → target_id = 0 + d2t[0] = ?
        sample_pairs = []
        for di in [0, 100, 1000, 5000, 15999]:
            full = di + int(d2t[di].item())
            tok_str = tok.decode([full]) if 0 <= full < tok.vocab_size else "[OOV]"
            sample_pairs.append((di, int(d2t[di].item()), full, tok_str))
        print("\nSample mappings (draft_idx, d2t[idx], full_id=sum, decoded):")
        for di, dv, full, ts in sample_pairs:
            print(f"  draft={di:5d}  d2t={dv:6d}  full={full:6d}  '{ts}'")
    else:
        # If t2d is int, assume t2d[target_id] = draft_idx
        # Then we can find a target_id where t2d[target_id] == 0 and verify
        print(f"t2d is INT — treating as target_id → draft_idx mapping")
        for target_id in range(min(200, t2d.shape[0])):
            di = int(t2d[target_id].item())
            if 0 <= di < d2t.shape[0]:
                inv = di + int(d2t[di].item())
                match = (inv == target_id)
                if not match and target_id < 50:
                    print(f"  target={target_id:5d}  draft={di:5d}  d2t={int(d2t[di].item())}  inv={inv}  match={match}")

    # Also test: are draft indices that come out of argmax giving sensible tokens?
    # Decode the most common 10 draft indices.
    print("\nDecoded targets for draft_idx 0..9:")
    for di in range(10):
        full = di + int(d2t[di].item())
        if 0 <= full < tok.vocab_size:
            try:
                ts = tok.decode([full])
                print(f"  draft={di}  full_id={full}  '{ts}'")
            except Exception:
                print(f"  draft={di}  full_id={full}  [decode error]")
        else:
            print(f"  draft={di}  full_id={full}  [OOV]")

    return {"d2t_first_5": d2t[:5].tolist(), "t2d_dtype": str(t2d.dtype)}


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    cpu=2,
    timeout=300,
)
def dump_specforge_internals() -> dict:
    """Print the relevant SpecForge source files so we can know:
      - which target layers are fused (low/mid/high indices)
      - exact backbone() / forward() flow we have to mimic at inference
      - how target hidden states are collected (hooks vs explicit forward)
      - how d2t / t2d are used at sampling time
    """
    import os
    out: dict = {}

    files_to_dump = [
        ("specforge/modeling/draft/llama3_eagle.py", 100000),
        ("specforge/modeling/draft/base.py", 30000),
    ]
    base = "/opt/SpecForge"
    for rel, max_chars in files_to_dump:
        p = os.path.join(base, rel)
        if os.path.isfile(p):
            with open(p) as fh:
                txt = fh.read()
            out[rel] = txt[:max_chars]
            if len(txt) > max_chars:
                out[rel] += f"\n... [truncated; total {len(txt)} chars]"
        else:
            out[rel] = "MISSING"

    for k, v in out.items():
        print(f"\n{'='*80}\n{k}\n{'='*80}\n{v}\n", flush=True)
    return out


# ---------------------------------------------------------------------------
# Step 4: Lossless verification. Tokens produced by eagle3_speculative_ar must
# be byte-identical to vanilla AR — even with the undertrained smoke checkpoint,
# because greedy rejection sampling is what guarantees losslessness, not
# acceptance rate.
# ---------------------------------------------------------------------------


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",  # default attention_backend="sdpa" — no flex_attention OOM
    cpu=4,
    timeout=900,
)
def verify_eagle3_lossless(
    target: str = "Qwen/Qwen2.5-Coder-7B",
    ckpt_path: str = "/data/eagle3_smoke/eagle3/epoch_0_step_200",
    k: int = 4,
    max_new_tokens: int = 32,
) -> dict:
    """Compare eagle3_speculative_ar vs vanilla AR. Tokens MUST match exactly."""
    import sys
    sys.path.insert(0, "/root/asts-spec")  # noqa: E402

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from specforge import AutoEagle3DraftModel  # type: ignore
    from asts.decoder import vanilla_ar  # type: ignore
    from asts.eagle3_decoder import fixed_eagle3_spec  # type: ignore

    print(f"[load] target = {target}", flush=True)
    tok = AutoTokenizer.from_pretrained(target)
    target_model = AutoModelForCausalLM.from_pretrained(
        target,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    target_model.eval()

    print(f"[load] draft  = {ckpt_path}", flush=True)
    draft = AutoEagle3DraftModel.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16)
    draft = draft.to("cuda")
    draft.eval()

    eos_token_ids = [tok.eos_token_id]
    if tok.pad_token_id is not None and tok.pad_token_id != tok.eos_token_id:
        eos_token_ids.append(tok.pad_token_id)

    prompts = [
        "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
        "def is_prime(n):\n    \"\"\"Return True if n is prime.\"\"\"\n",
        "def reverse_string(s):\n    \"\"\"Reverse a string.\"\"\"\n    return ",
    ]

    results = []
    for i, prompt in enumerate(prompts):
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")[0]
        print(f"\n[prompt {i}] {prompt!r}", flush=True)
        print(f"  input_ids: {ids.tolist()[:20]}{'...' if len(ids) > 20 else ''}", flush=True)

        # Vanilla AR
        van = vanilla_ar(
            prompt_ids=ids,
            target=target_model,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
        )
        van_tokens = van.output_token_ids[len(ids):]

        # EAGLE-3 spec
        spec = fixed_eagle3_spec(
            prompt_ids=ids,
            target=target_model,
            draft=draft,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            k=k,
        )
        spec_tokens = spec.output_token_ids[len(ids):]

        match = van_tokens == spec_tokens
        n_acc_drafts = sum(s.n_accepted_drafts for s in spec.steps)
        n_total_drafts = sum(s.k for s in spec.steps)
        acc_rate = n_acc_drafts / max(1, n_total_drafts)

        print(f"  vanilla: {van_tokens}", flush=True)
        print(f"  spec   : {spec_tokens}", flush=True)
        print(f"  match  : {match}  acceptance: {acc_rate:.2%} ({n_acc_drafts}/{n_total_drafts})", flush=True)

        results.append({
            "prompt_idx": i,
            "match": match,
            "n_vanilla_tokens": len(van_tokens),
            "n_spec_tokens": len(spec_tokens),
            "vanilla_text": tok.decode(van_tokens, skip_special_tokens=False),
            "spec_text": tok.decode(spec_tokens, skip_special_tokens=False),
            "acceptance": acc_rate,
            "n_steps": len(spec.steps),
        })

    all_match = all(r["match"] for r in results)
    print(f"\n=== Result ===", flush=True)
    print(f"  lossless: {'✓ PASS' if all_match else '✗ FAIL'}", flush=True)
    print(f"  per-prompt acceptance: {[round(r['acceptance'], 3) for r in results]}", flush=True)
    return {"all_match": all_match, "results": results}


# ---------------------------------------------------------------------------
# Step 6: Full HumanEval eval — vanilla AR vs fixed-k EAGLE-3 vs ASTS-EAGLE-3.
# Records per-method walltime, tokens, acceptance rate, and lossless-equality
# vs vanilla. Mirrors the structure of proto_app.run_eagle_eval.
# ---------------------------------------------------------------------------


@app.function(
    volumes={"/data": data_volume, "/cache/huggingface": hf_cache},
    gpu="L40S",
    timeout=14400,
    cpu=4,
)
def run_eagle3_eval(
    target: str = "Qwen/Qwen2.5-Coder-7B",
    ckpt_path: str = "/data/eagle3_v0_prod/eagle3",
    n: int = 164,
    max_new_tokens: int = 256,
    k_fixed: str = "4,8",
    methods: str = "vanilla,eagle3,asts_eagle3",
    language: str = "python",
    run_tag: str = "eagle3_eval_v0",
) -> dict:
    """Full HumanEval eval for an EAGLE-3 checkpoint.

    `ckpt_path` should point to the directory containing model.safetensors +
    config.json (i.e. epoch_X_step_Y); if it's a parent dir, we pick the
    latest step automatically.
    """
    import json
    import os
    import sys

    sys.path.insert(0, "/root/asts-spec")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from specforge import AutoEagle3DraftModel  # type: ignore

    from asts.decoder import vanilla_ar  # type: ignore
    from asts.eagle3_decoder import (  # type: ignore
        asts_eagle3_spec,
        fixed_eagle3_spec,
    )
    from asts.humaneval import load_problems_for_language  # type: ignore

    # If ckpt_path is a parent dir like /data/eagle3_v0_prod/eagle3, pick the
    # latest epoch_X_step_Y subfolder.
    if not os.path.isfile(os.path.join(ckpt_path, "model.safetensors")):
        candidates = []
        for entry in os.listdir(ckpt_path):
            full = os.path.join(ckpt_path, entry)
            if os.path.isfile(os.path.join(full, "model.safetensors")):
                candidates.append(full)
        if not candidates:
            raise RuntimeError(f"no checkpoint subfolder under {ckpt_path}")
        # Sort by epoch / step (parsed from name "epoch_E_step_S").
        def _key(p: str) -> tuple[int, int]:
            name = os.path.basename(p)
            parts = name.replace("epoch_", "").replace("step_", "").split("_")
            try:
                return int(parts[0]), int(parts[1])
            except Exception:
                return (0, 0)
        candidates.sort(key=_key)
        ckpt_path = candidates[-1]
        print(f"[ckpt] resolved to latest: {ckpt_path}", flush=True)

    print(f"[load] target = {target}", flush=True)
    tok = AutoTokenizer.from_pretrained(target)
    target_model = AutoModelForCausalLM.from_pretrained(
        target, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    target_model.eval()

    print(f"[load] draft  = {ckpt_path}", flush=True)
    draft = AutoEagle3DraftModel.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16)
    draft = draft.to("cuda")
    draft.eval()

    eos_token_ids = [tok.eos_token_id]
    if tok.pad_token_id is not None and tok.pad_token_id != tok.eos_token_id:
        eos_token_ids.append(tok.pad_token_id)

    problems = load_problems_for_language(language=language, n=n)
    print(f"[data] loaded {len(problems)} {language} problems", flush=True)

    # Build the AST policy lazily — only needed for ASTS method.
    ast_policy = None
    method_set = set(methods.split(","))
    if "asts_eagle3" in method_set:
        from asts.ast_policy import ASTPolicy  # type: ignore
        ast_policy = ASTPolicy(language=language)

    fixed_ks = [int(x) for x in k_fixed.split(",") if x.strip()]

    output_dir = f"/data/{run_tag}/eval"
    os.makedirs(output_dir, exist_ok=True)
    per_problem_path = os.path.join(output_dir, f"per_problem_{language}.jsonl")
    aggregate_path = os.path.join(output_dir, f"aggregate_{language}.json")

    # Aggregates: method → {n_problems, total_walltime, total_new_tokens,
    # total_drafts, total_accepted, lossless_count}
    aggregates: dict = {}

    def _accumulate(method_name: str, res, vanilla_tokens: list[int]):
        a = aggregates.setdefault(method_name, {
            "n_problems": 0,
            "wall_us": 0.0,
            "n_new_tokens": 0,
            "n_drafts": 0,
            "n_accepted": 0,
            "lossless_match": 0,
        })
        a["n_problems"] += 1
        a["wall_us"] += res.wall_us_total
        new_tokens = res.output_token_ids[len(prompt_ids):]
        a["n_new_tokens"] += len(new_tokens)
        for s in res.steps:
            a["n_drafts"] += s.k
            a["n_accepted"] += s.n_accepted_drafts
        if vanilla_tokens is not None and new_tokens == vanilla_tokens:
            a["lossless_match"] += 1

    fout = open(per_problem_path, "w")
    try:
        for idx, prob in enumerate(problems):
            prompt_ids = tok(prob.prompt, return_tensors="pt").input_ids.to("cuda")[0]
            row: dict = {"task_id": prob.task_id, "language": language}

            # --- Vanilla baseline (always run; needed for lossless comparison) ---
            van = vanilla_ar(
                prompt_ids=prompt_ids,
                target=target_model,
                max_new_tokens=max_new_tokens,
                eos_token_ids=eos_token_ids,
            )
            van_tokens = van.output_token_ids[len(prompt_ids):]
            row["vanilla"] = {
                "wall_us": van.wall_us_total,
                "n_new_tokens": len(van_tokens),
                "tokens": van_tokens,
            }
            if "vanilla" in method_set:
                _accumulate("vanilla", van, van_tokens)

            # --- Fixed-k EAGLE-3 ---
            if "eagle3" in method_set:
                for k in fixed_ks:
                    spec = fixed_eagle3_spec(
                        prompt_ids=prompt_ids,
                        target=target_model,
                        draft=draft,
                        max_new_tokens=max_new_tokens,
                        eos_token_ids=eos_token_ids,
                        k=k,
                    )
                    spec_tokens = spec.output_token_ids[len(prompt_ids):]
                    n_drafts = sum(s.k for s in spec.steps)
                    n_acc = sum(s.n_accepted_drafts for s in spec.steps)
                    row[f"eagle3_k{k}"] = {
                        "wall_us": spec.wall_us_total,
                        "n_new_tokens": len(spec_tokens),
                        "n_drafts": n_drafts,
                        "n_accepted": n_acc,
                        "match_vanilla": spec_tokens == van_tokens,
                    }
                    _accumulate(f"eagle3_k{k}", spec, van_tokens)

            # --- ASTS-EAGLE-3 (variable-length, AST-gated) ---
            if "asts_eagle3" in method_set and ast_policy is not None:
                spec = asts_eagle3_spec(
                    prompt_ids=prompt_ids,
                    target=target_model,
                    draft=draft,
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=eos_token_ids,
                    tokenizer=tok,
                    ast_policy=ast_policy,
                )
                spec_tokens = spec.output_token_ids[len(prompt_ids):]
                n_drafts = sum(s.k for s in spec.steps)
                n_acc = sum(s.n_accepted_drafts for s in spec.steps)
                row["asts_eagle3"] = {
                    "wall_us": spec.wall_us_total,
                    "n_new_tokens": len(spec_tokens),
                    "n_drafts": n_drafts,
                    "n_accepted": n_acc,
                    "match_vanilla": spec_tokens == van_tokens,
                }
                _accumulate("asts_eagle3", spec, van_tokens)

            fout.write(json.dumps(row) + "\n")
            fout.flush()

            if (idx + 1) % 5 == 0 or idx == len(problems) - 1:
                # Live progress + acceptance.
                summary = {}
                for m, a in aggregates.items():
                    if a["n_drafts"] > 0:
                        summary[m] = {
                            "tok_per_sec": round(
                                a["n_new_tokens"] / max(1e-6, a["wall_us"] / 1e6), 1
                            ),
                            "accept": round(a["n_accepted"] / a["n_drafts"], 3),
                            "lossless": f"{a['lossless_match']}/{a['n_problems']}",
                        }
                    else:
                        summary[m] = {
                            "tok_per_sec": round(
                                a["n_new_tokens"] / max(1e-6, a["wall_us"] / 1e6), 1
                            ),
                        }
                print(f"[progress] {idx+1}/{len(problems)}: {summary}", flush=True)
    finally:
        fout.close()

    # Compute final aggregate metrics.
    final = {}
    for m, a in aggregates.items():
        wall_s = a["wall_us"] / 1e6
        tok_per_sec = a["n_new_tokens"] / max(1e-6, wall_s)
        accept = a["n_accepted"] / max(1, a["n_drafts"])
        final[m] = {
            "n_problems": a["n_problems"],
            "wall_s": round(wall_s, 2),
            "n_new_tokens": a["n_new_tokens"],
            "tok_per_sec": round(tok_per_sec, 2),
            "accept_rate": round(accept, 4),
            "n_accepted_drafts": a["n_accepted"],
            "n_total_drafts": a["n_drafts"],
            "lossless_match": a["lossless_match"],
        }

    # Speedups vs vanilla.
    if "vanilla" in final:
        van_tps = final["vanilla"]["tok_per_sec"]
        for m, agg in final.items():
            if m != "vanilla":
                agg["speedup_vs_vanilla"] = round(agg["tok_per_sec"] / max(1e-6, van_tps), 3)

    aggregate = {
        "by_method": final,
        "meta": {
            "target": target,
            "ckpt_path": ckpt_path,
            "language": language,
            "n_problems": len(problems),
            "max_new_tokens": max_new_tokens,
            "k_fixed": fixed_ks,
            "methods": list(method_set),
        },
        "per_problem_path": per_problem_path,
    }
    with open(aggregate_path, "w") as fh:
        json.dump(aggregate, fh, indent=2)
    data_volume.commit()
    hf_cache.commit()

    print("\n=== EAGLE-3 eval results ===", flush=True)
    print(json.dumps(final, indent=2), flush=True)
    print(f"\nper-problem: {per_problem_path}", flush=True)
    print(f"aggregate:   {aggregate_path}", flush=True)
    return aggregate


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main() -> None:
    """Run install verification by default. Other functions invoked via :: syntax."""
    res = verify_install.remote()
    print()
    print("=== Result ===")
    has_failure = any(str(v).startswith("FAIL") or v == "MISSING" for v in res.values())
    if has_failure:
        print("  ✗ Some imports/files failed — see above")
    else:
        print("  ✓ All imports/files OK — ready for smoke training")
    print()
