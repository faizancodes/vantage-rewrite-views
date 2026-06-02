"""Lookahead Decoding (Hao et al. 2024) wrapper for our eval harness.

Lookahead Decoding is a *draft-free* lossless speedup. Instead of using a
separate draft model, it runs a Jacobi-iteration sliding window over the
target itself: at each step it predicts N parallel guesses at future
positions, verifies them via greedy match, and accepts the longest
contiguous prefix.

We use the reference implementation from `hao-ai-lab/LookaheadDecoding`
(pip-installed as `lade`). The library monkey-patches HF transformers so
that `model.generate(...)` becomes lookahead-augmented.

Output is provably greedy-equivalent to vanilla AR — lossless under fp32,
near-lossless under bf16 (same numerical drift as our other methods).

Usage:
    >>> from asts.lookahead_decoder import init_lookahead, lookahead_decode
    >>> init_lookahead()  # call ONCE before loading the target
    >>> # ... load target ...
    >>> result = lookahead_decode(prompt_ids, target, max_new_tokens=256, ...)

The init step monkey-patches generate() globally, so every model loaded
afterwards inherits lookahead behavior. To turn it off, restart the
process.
"""

from __future__ import annotations

import time
import warnings

import torch

from .decoder import DecodeResult, StepRecord


_INITIALIZED = False


def init_lookahead(level: int = 5, window_size: int = 7, guess_set_size: int = 7) -> bool:
    """Patch HF transformers with lookahead decoding. Returns True on success.

    Args:
        level:           Jacobi iteration depth (paper recommends 5)
        window_size:     Lookahead window W (paper recommends 7)
        guess_set_size:  N-gram cache size G (paper recommends 7)
    """
    global _INITIALIZED
    if _INITIALIZED:
        return True
    try:
        import lade  # type: ignore
    except ImportError:
        warnings.warn(
            "lade is not installed. Install with: "
            "pip install git+https://github.com/hao-ai-lab/LookaheadDecoding.git"
        )
        return False
    try:
        lade.augment_all()
        lade.config_lade(
            LEVEL=level,
            WINDOW_SIZE=window_size,
            GUESS_SET_SIZE=guess_set_size,
            DEBUG=0,
        )
    except Exception as e:
        warnings.warn(f"lade.augment_all/config_lade failed: {e}")
        return False
    _INITIALIZED = True
    return True


@torch.no_grad()
def lookahead_decode(
    prompt_ids: torch.Tensor,
    target,
    max_new_tokens: int,
    eos_token_ids: list[int],
    method_name: str = "lookahead",
) -> DecodeResult:
    """Generate `max_new_tokens` from `target` using lookahead decoding.

    Returns a single-step DecodeResult (lookahead doesn't have meaningful
    per-step granularity from our eval harness's perspective — it emits a
    variable number of tokens per outer Jacobi iteration, but those aren't
    exposed by `lade.generate`).
    """
    if not _INITIALIZED:
        raise RuntimeError(
            "init_lookahead() must be called BEFORE loading the target. "
            "Call it at process start and reload the target if you've "
            "already loaded one."
        )
    device = next(target.parameters()).device
    if prompt_ids.dim() == 1:
        prompt_ids_b = prompt_ids.unsqueeze(0).to(device)
        prompt_len = prompt_ids.shape[0]
    else:
        prompt_ids_b = prompt_ids.to(device)
        prompt_len = prompt_ids.shape[1]

    # eos_token_id only takes a single int in older transformers; pass the
    # primary stop. lade respects HF's stopping-criteria.
    eos_id = eos_token_ids[0] if eos_token_ids else None
    pad_id = (
        getattr(target.config, "pad_token_id", None)
        or eos_id
        or 0
    )

    t0 = time.perf_counter_ns()
    output = target.generate(
        prompt_ids_b,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
    )
    t1 = time.perf_counter_ns()

    output_ids = output[0].tolist()
    new_tokens = output_ids[prompt_len:]

    # Synthesize a one-step DecodeResult so the eval-harness aggregator
    # treats lookahead like other methods. The "step" emits all new tokens.
    step = StepRecord(
        method=method_name,
        step=0,
        k=len(new_tokens),
        n_accepted_drafts=0,  # lade doesn't expose internal accept counts
        n_emitted=len(new_tokens),
        rejected=False,
        node_type=None,
        deepest_type=None,
        wall_us=(t1 - t0) / 1000.0,
        draft_us=0.0,
        verify_us=(t1 - t0) / 1000.0,
        parse_us=0.0,
    )
    return DecodeResult(
        output_token_ids=output_ids,
        output_text="",
        n_new_tokens=len(new_tokens),
        steps=[step],
        wall_us_total=(t1 - t0) / 1000.0,
    )


def lookahead_spec(
    prompt_ids: torch.Tensor,
    target,
    max_new_tokens: int,
    eos_token_ids: list[int],
) -> DecodeResult:
    """Public wrapper matching the signature of other decode methods."""
    return lookahead_decode(
        prompt_ids=prompt_ids,
        target=target,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        method_name="lookahead",
    )
