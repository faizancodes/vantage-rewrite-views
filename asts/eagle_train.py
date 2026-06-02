"""Training loop for the simplified EAGLE-1 head on Qwen-Coder-7B.

Pipeline (per training step):
  1. Target model (frozen, bf16) forward on a batch of code → produces
     hidden_states[-2] (penultimate layer) and logits.
  2. EagleHead (trainable, bf16 forward / fp32 master weights via AdamW)
     predicts hidden states for positions [1..T-1] given input [0..T-2].
  3. Project predicted hidden states through target's frozen LM head → logits.
  4. Loss = 0.7 * KL(target_logits || pred_logits) + 0.3 * smooth_L1(pred_h, target_h).
  5. Backprop only through EagleHead (target frozen).

Training corpus: bigcode/the-stack-smol, Python subset (~10K samples).
Tokenization: target model's tokenizer (Qwen-Coder); chunked to 1024-token
contexts.

Hardware: tested on L40S 48GB. Memory budget ~30GB:
  - Qwen-Coder-7B bf16: ~14GB
  - EAGLE head + grads + AdamW state (fp32): ~3GB
  - Activations (target forward): ~10GB at batch=4 seq=1024
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .eagle import EagleConfig, EagleHead, count_params


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CodeChunkDataset(Dataset):
    """Pre-tokenized fixed-length chunks from the Stack v2 smol Python subset."""

    def __init__(self, input_ids_chunks: list[list[int]]):
        self.chunks = input_ids_chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.chunks[idx], dtype=torch.long)


def build_chunks(
    tokenizer,
    n_samples: int = 10_000,
    chunk_len: int = 1024,
    min_chunk_len: int = 256,
    seed: int = 42,
) -> list[list[int]]:
    """Load Stack v2 smol Python, tokenize, split into chunks of `chunk_len`.

    Drops chunks shorter than `min_chunk_len` (typically the trailing chunk
    of a file).
    """
    from datasets import Dataset, load_dataset

    # Try several public Python-code datasets in order of preference. Each
    # exposes a `.content` (or text-equivalent) field after normalization.
    ds = None
    text_field = "content"
    candidates = [
        # (repo, kwargs, text_field)
        ("bigcode/the-stack-smol", {"data_dir": "data/python", "split": "train"}, "content"),
        ("codeparrot/codeparrot-clean-valid", {"split": "train"}, "content"),
        ("codeparrot/github-code-clean", {"name": "Python-all", "split": "train", "streaming": False, "trust_remote_code": True}, "code"),
    ]
    last_err = None
    for repo, kwargs, field in candidates:
        try:
            log.info("loading %s with %s ...", repo, kwargs)
            ds = load_dataset(repo, **kwargs)
            text_field = field
            log.info("loaded %s (text field: %s)", repo, field)
            break
        except Exception as e:
            log.warning("failed to load %s: %s", repo, e)
            last_err = e
            continue
    if ds is None:
        raise RuntimeError(f"all training-corpus candidates failed; last error: {last_err}")
    # split="train" always returns a single Dataset, not a DatasetDict — narrow the type
    assert isinstance(ds, Dataset), f"expected Dataset, got {type(ds).__name__}"
    n_total = len(ds)
    log.info("dataset has %d samples", n_total)

    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=rng).tolist()
    chunks: list[list[int]] = []

    for i in perm:
        if len(chunks) >= n_samples:
            break
        text = ds[i][text_field]
        if not text or len(text) < 100:
            continue
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) < min_chunk_len:
            continue
        # Split into chunks
        for start in range(0, len(ids), chunk_len):
            chunk = ids[start : start + chunk_len]
            if len(chunk) >= min_chunk_len:
                chunks.append(chunk)
                if len(chunks) >= n_samples:
                    break
    log.info("built %d chunks of up to %d tokens", len(chunks), chunk_len)
    return chunks


def collate_chunks(batch: list[torch.Tensor]) -> dict:
    """Pad-right to longest in batch."""
    max_len = max(b.numel() for b in batch)
    padded = torch.full((len(batch), max_len), 0, dtype=torch.long)
    attn = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        padded[i, : b.numel()] = b
        attn[i, : b.numel()] = 1
    return {"input_ids": padded, "attention_mask": attn}


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def eagle_loss(
    pred_h: torch.Tensor,           # [B, T-1, H]
    pred_logits: torch.Tensor,      # [B, T-1, V]
    target_h: torch.Tensor,         # [B, T-1, H]
    target_logits: torch.Tensor,    # [B, T-1, V]
    attention_mask: torch.Tensor,   # [B, T-1] — 1 for real tokens
    kl_weight: float = 0.7,
    h_weight: float = 0.3,
) -> tuple[torch.Tensor, dict]:
    """EAGLE-1 loss: weighted sum of next-position KL and hidden-state regression."""
    # Mask out padding positions
    mask = attention_mask.bool()  # [B, T-1]

    # Hidden-state smooth-L1 — masked mean
    h_diff = (pred_h - target_h).abs()
    h_loss_per_pos = torch.where(
        h_diff < 1.0, 0.5 * h_diff.pow(2), h_diff - 0.5
    ).mean(dim=-1)  # [B, T-1]
    h_loss = h_loss_per_pos[mask].mean()

    # KL — mask, then mean over tokens
    log_pred = F.log_softmax(pred_logits, dim=-1)
    target_probs = F.softmax(target_logits, dim=-1)
    # KL(target || pred) = sum target * (log target - log pred)
    log_target = torch.log(target_probs.clamp_min(1e-10))
    kl_per_pos = (target_probs * (log_target - log_pred)).sum(dim=-1)  # [B, T-1]
    kl_loss = kl_per_pos[mask].mean()

    total = kl_weight * kl_loss + h_weight * h_loss
    return total, {
        "loss": total.item(),
        "kl": kl_loss.item(),
        "h_l1": h_loss.item(),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    target_model_id: str = "Qwen/Qwen2.5-Coder-7B"
    output_dir: str = "out/eagle_v0"
    n_samples: int = 10_000
    chunk_len: int = 1024
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    grad_clip: float = 1.0
    log_every: int = 20
    save_every: int = 1000
    kl_weight: float = 0.7
    h_weight: float = 0.3
    dtype: str = "bfloat16"
    seed: int = 42


def train(cfg: TrainConfig) -> dict:
    """Run EAGLE training. Returns a summary dict."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        cfg.dtype
    ]

    log.info("loading target %s", cfg.target_model_id)
    tokenizer = AutoTokenizer.from_pretrained(cfg.target_model_id)
    target = AutoModelForCausalLM.from_pretrained(
        cfg.target_model_id, torch_dtype=torch_dtype, device_map="cuda"
    )
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)

    # Resolve target config → EAGLE config
    eagle_cfg = EagleConfig.from_target_config(target.config)
    log.info("eagle config: %s", asdict(eagle_cfg))

    # Build EAGLE head, move to cuda + bf16
    eagle = EagleHead(eagle_cfg).to("cuda").to(torch_dtype)
    n_params = count_params(eagle)
    log.info("eagle head: %d trainable params (%.2fM)", n_params, n_params / 1e6)

    # Build dataset
    log.info("building training corpus (%d samples)...", cfg.n_samples)
    t0 = time.perf_counter()
    chunks = build_chunks(
        tokenizer, n_samples=cfg.n_samples, chunk_len=cfg.chunk_len, seed=cfg.seed
    )
    log.info("built %d chunks in %.1fs", len(chunks), time.perf_counter() - t0)

    dataset = CodeChunkDataset(chunks)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_chunks,
        drop_last=True,
        num_workers=0,
    )

    # Optimizer + scheduler — fp32 master weights via AdamW (default)
    optimizer = torch.optim.AdamW(
        eagle.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    total_steps = cfg.epochs * len(loader)
    log.info("total_steps=%d, warmup=%d", total_steps, cfg.warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Get references to target's LM head + final norm (both frozen). EAGLE
    # outputs pre-norm hidden states; we apply target's final norm before the
    # LM head so pred_logits go through the same path as target_logits.
    target_lm_head = target.lm_head
    target_norm = target.model.norm

    train_log: list[dict] = []
    step = 0
    t_train_start = time.perf_counter()

    for epoch in range(cfg.epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to("cuda")
            attn_mask = batch["attention_mask"].to("cuda")

            # Teacher forward (frozen)
            with torch.no_grad():
                target_out = target(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                # hidden_states[-1] is post-final-norm; we want PRE-final hidden,
                # i.e. hidden_states[-2] (output of penultimate layer).
                teacher_h_full = target_out.hidden_states[-2]   # [B, T, H]
                teacher_logits_full = target_out.logits          # [B, T, V]

            # Inputs: positions 0..T-2; targets: positions 1..T-1
            input_h = teacher_h_full[:, :-1].contiguous()
            target_h = teacher_h_full[:, 1:].contiguous()
            target_logits = teacher_logits_full[:, 1:].contiguous()
            target_attn = attn_mask[:, 1:].contiguous()

            # Eagle forward (pre-norm output)
            pred_h = eagle(input_h, attention_mask=attn_mask[:, :-1])
            # Apply target's final norm, then project through (frozen) lm_head.
            # This matches target's logit-computation path exactly.
            pred_h_post_norm = target_norm(pred_h)
            pred_logits = target_lm_head(pred_h_post_norm)

            loss, loss_breakdown = eagle_loss(
                pred_h=pred_h,
                pred_logits=pred_logits,
                target_h=target_h,
                target_logits=target_logits,
                attention_mask=target_attn,
                kl_weight=cfg.kl_weight,
                h_weight=cfg.h_weight,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(eagle.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            if step % cfg.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.perf_counter() - t_train_start
                log.info(
                    "step %d/%d  lr=%.2e  loss=%.4f  kl=%.4f  h=%.4f  elapsed=%.0fs",
                    step, total_steps, lr_now,
                    loss_breakdown["loss"], loss_breakdown["kl"], loss_breakdown["h_l1"],
                    elapsed,
                )
                train_log.append({
                    "step": step,
                    "epoch": epoch,
                    "lr": lr_now,
                    **loss_breakdown,
                })

            if step > 0 and step % cfg.save_every == 0:
                save_path = output_dir / f"eagle_step{step}.pt"
                torch.save({
                    "step": step,
                    "config": asdict(cfg),
                    "eagle_config": asdict(eagle_cfg),
                    "state_dict": {k: v.cpu() for k, v in eagle.state_dict().items()},
                }, save_path)
                log.info("saved checkpoint → %s", save_path)

            step += 1

    # Final save
    final_path = output_dir / "eagle_final.pt"
    torch.save({
        "step": step,
        "config": asdict(cfg),
        "eagle_config": asdict(eagle_cfg),
        "state_dict": {k: v.cpu() for k, v in eagle.state_dict().items()},
    }, final_path)
    log.info("saved final → %s", final_path)

    # Save training log
    log_path = output_dir / "train_log.json"
    log_path.write_text(json.dumps({"config": asdict(cfg), "log": train_log}, indent=2))

    return {
        "final_path": str(final_path),
        "n_params": n_params,
        "total_steps": step,
        "wall_seconds": time.perf_counter() - t_train_start,
        "final_loss": train_log[-1] if train_log else None,
    }
