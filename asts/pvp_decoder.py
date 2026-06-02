"""PVP (Predictive Verifier Pipelining) — single-request batched-K speculative decode.

PVP wraps PLD: each step builds a depth-K chain of PLD drafts and runs them as
a single B=K batched verifier forward. Row 0 is the standard PLD path (always
byte-identical to plain PLD). Row k > 0 is *certified* lossless via a cheap
single-token re-verification against the true KV path before any of its
tokens are committed.

This module does not modify ``asts/blazedit_decoder.py``. It imports the PLD
helpers (``prompt_lookup_draft``, ``_target_verify_step``, etc.) and reuses
them on the plain-PLD code path.

Per-step diagnostic counters are stored on ``StepRecord.proposal_*`` and
``pld_*`` fields so existing eval analyses can read PVP runs without changes.
"""

from __future__ import annotations

import time
from typing import Optional

import torch
from transformers import DynamicCache

from .blazedit_decoder import (
    BlazEditConfig,
    _eos_truncate_and_extend,
    _target_verify_step,
)
from .decoder import DecodeResult, StepRecord, crop_dynamic_cache
from .pvp_chain import ChainLookupResult, chain_pld_lookup
from .pvp_kv_utils import (
    cache_length,
    clone_cache,
    layer_keys_values,
    lift_prompt_kv,
    num_layers,
    set_layer_keys_values,
)


def _stack_caches_along_batch(caches: list) -> DynamicCache:
    """Stack a list of B=1 caches along the batch dim. All must share seq length."""
    if not caches:
        raise ValueError("empty cache list")
    if len(caches) == 1:
        return clone_cache(caches[0])

    out = DynamicCache()
    n_layers = num_layers(caches[0])
    layers_per_cache = [list(layer_keys_values(c)) for c in caches]
    seq_lens = {int(layers_per_cache[0][0][0].shape[-2])} | {
        int(lp[0][0].shape[-2]) for lp in layers_per_cache
    }
    if len(seq_lens) != 1:
        raise ValueError(f"stack requires uniform seq dim, got {seq_lens}")

    for layer_idx in range(n_layers):
        keys = torch.cat([lp[layer_idx][0] for lp in layers_per_cache], dim=0).contiguous()
        values = torch.cat([lp[layer_idx][1] for lp in layers_per_cache], dim=0).contiguous()
        out.update(keys, values, layer_idx)
    return out


def _pad_cache_to_length(cache, target_seq_len: int):
    """Extend a B=1 cache to ``target_seq_len`` by appending zero-padded slots.

    The padded slots are *junk* — they must be masked out in attention. Used to
    make row 0's cache the same seq length as row 1's lifted cache before
    stacking along the batch dim.
    """
    current = cache_length(cache)
    if target_seq_len < current:
        raise ValueError(f"target_seq_len={target_seq_len} < current={current}")
    if target_seq_len == current:
        return clone_cache(cache)

    out = clone_cache(cache)
    pad = target_seq_len - current
    for i, (k, v) in enumerate(layer_keys_values(out)):
        if k is None:
            continue
        pad_k = torch.zeros(*k.shape[:-2], pad, k.shape[-1], dtype=k.dtype, device=k.device)
        pad_v = torch.zeros(*v.shape[:-2], pad, v.shape[-1], dtype=v.dtype, device=v.device)
        new_k = torch.cat([k, pad_k], dim=-2).contiguous()
        new_v = torch.cat([v, pad_v], dim=-2).contiguous()
        set_layer_keys_values(out, i, new_k, new_v)

    if hasattr(out, "_seen_tokens"):
        out._seen_tokens = int(out._seen_tokens or 0) + pad
    return out


def _find_rotary_emb(target):
    """Locate the rotary embedding module on a HF causal LM."""
    for cand in (
        getattr(target, "rotary_emb", None),
        getattr(getattr(target, "model", None), "rotary_emb", None),
        getattr(getattr(target, "base_model", None), "rotary_emb", None),
    ):
        if cand is not None:
            return cand
    raise RuntimeError(
        "could not locate rotary_emb on target (looked at target.rotary_emb, "
        "target.model.rotary_emb, target.base_model.rotary_emb)"
    )


@torch.no_grad()
def _make_rerope_tensors(
    target,
    delta: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (cos, sin) of shape ``[1, 1, 1, head_dim]`` for the given delta.

    delta = destination_position - source_position. The cache stored K's with
    rotary baked in at the source positions; applying this rotation shifts
    them to the destination positions.

    For negative deltas, ``sin`` is negated (sin is odd; cos is even — the
    rotary_emb module only accepts non-negative position_ids).
    """
    abs_delta = abs(int(delta))
    position_ids = torch.tensor([[abs_delta]], dtype=torch.long, device=device)
    dummy_x = torch.zeros(1, 1, head_dim, device=device, dtype=dtype)
    rotary_emb = _find_rotary_emb(target)
    cos, sin = rotary_emb(dummy_x, position_ids)
    # rotary_emb returns shape [1, 1, head_dim]; add num_kv_heads broadcast dim.
    cos = cos.unsqueeze(1).contiguous()
    sin = sin.unsqueeze(1).contiguous()
    if int(delta) < 0:
        sin = -sin
    return cos, sin


def _select_row_and_slots(batched_cache, row_idx: int, slot_indices: list[int]):
    """Extract a B=1 cache from a B>=2 cache, selecting given slot indices."""
    if not slot_indices:
        raise ValueError("slot_indices must be non-empty")
    device = None
    for kv in layer_keys_values(batched_cache):
        if kv[0] is not None:
            device = kv[0].device
            break
    if device is None:
        raise RuntimeError("batched_cache has no populated layers")

    idx = torch.tensor(slot_indices, dtype=torch.long, device=device)
    out = DynamicCache()
    for layer_idx, (k, v) in enumerate(layer_keys_values(batched_cache)):
        if k is None:
            continue
        k_row = k[row_idx : row_idx + 1].index_select(-2, idx).contiguous()
        v_row = v[row_idx : row_idx + 1].index_select(-2, idx).contiguous()
        out.update(k_row, v_row, layer_idx)
    if hasattr(out, "_seen_tokens"):
        out._seen_tokens = len(slot_indices)
    return out


@torch.no_grad()
def _pvp_step_batched(
    *,
    prefix: list[int],
    prompt_len: int,
    target,
    target_cache,
    target_cache_len: int,
    chain: list[ChainLookupResult],
    max_new_tokens: int,
    eos_token_ids: list[int],
) -> dict:
    """Single PVP step at chain depth >= 2. Mutates ``prefix`` on commit.

    Returns a dict with diagnostic fields and the new (target_cache, target_cache_len).
    """
    assert len(chain) >= 2
    device = next(target.parameters()).device
    L = len(prefix)

    # Canonical PLD cache convention.
    if target_cache_len >= L:
        target_cache_len = max(0, L - 1)
        crop_dynamic_cache(target_cache, target_cache_len)
    L_minus_1 = target_cache_len  # cache covers [0, L_minus_1)

    chain0 = chain[0]
    chain1 = chain[1]
    draft1 = chain0.draft
    draft2 = chain1.draft
    n_1 = len(draft1)
    n_2 = len(draft2)
    follow1 = chain0.follow_start

    # Bonus placeholder (matches what chain_pld_lookup used to extend the query).
    if 0 <= follow1 + n_1 < L:
        bonus_placeholder = prefix[follow1 + n_1]
    else:
        bonus_placeholder = prefix[-1]

    # Sanity gate: we lift from prompt slots [follow1-1, follow1+n_1) into
    # destination slots [L_minus_1, L+n_1). Requires follow1 >= 1 AND the
    # source range to be inside target_cache (length L_minus_1).
    if follow1 < 1 or follow1 + n_1 > L_minus_1:
        return {"fall_back_to_pld": True}

    # ---- Compute RoPE re-rotation tensors ----
    # Source positions: [follow1-1, follow1+n_1).
    # Destination positions: [L_minus_1, L+n_1) = [L-1, L+n_1).
    # Delta per slot is uniform: (L-1) - (follow1-1) = L - follow1.
    delta = (L_minus_1 + 1) - follow1  # = L - follow1
    # Find head_dim from any K tensor in target_cache.
    first_k = None
    for kv in layer_keys_values(target_cache):
        if kv[0] is not None:
            first_k = kv[0]
            break
    if first_k is None:
        return {"fall_back_to_pld": True}
    head_dim = int(first_k.shape[-1])
    cache_dtype = first_k.dtype
    rerope_cos, rerope_sin = _make_rerope_tensors(
        target, delta, head_dim, device, cache_dtype,
    )

    # ---- Build row 1's lifted cache (B=1) of length L_minus_1 + n_1 + 1 ----
    # The lift starts at follow1-1 (not follow1) because PLD's suffix match
    # guarantees prompt[follow1-1] == prefix[L_minus_1] (the token immediately
    # before draft1 in the source is the last accepted-context token in row 1's
    # logical state). After RoPE re-rotation:
    #
    #   destination slot L_minus_1 + i ← prompt slot follow1-1 + i
    #     i=0:    K/V for prefix[L_minus_1] at position L_minus_1.
    #     i=1:    K/V for draft1[0]         at position L_minus_1+1 = L.
    #     ...
    #     i=n_1:  K/V for draft1[n_1-1]     at position L+n_1-1.
    #
    # Together these cover row 1's logical positions [0, L+n_1) without any
    # missing slot or rotary mismatch.
    row1_lifted = lift_prompt_kv(
        target_cache, target_cache, follow1 - 1, n_1 + 1,
        rotary_cos=rerope_cos, rotary_sin=rerope_sin,
    )
    batched_cache_len = cache_length(row1_lifted)  # = L_minus_1 + n_1 + 1
    assert batched_cache_len == L_minus_1 + n_1 + 1, (
        f"row1 cache length {batched_cache_len} != L_minus_1+n_1+1={L_minus_1 + n_1 + 1}"
    )

    # Pad row 0's cache to the same seq length with junk (will be masked).
    row0_padded = _pad_cache_to_length(target_cache, batched_cache_len)

    batched_cache = _stack_caches_along_batch([row0_padded, row1_lifted])

    # ---- Build batched inputs / mask / position_ids ----
    row0_input = [prefix[L_minus_1]] + list(draft1)        # length n_1 + 1
    row1_input = [bonus_placeholder] + list(draft2)        # length n_2 + 1
    input_len = max(len(row0_input), len(row1_input))

    pad_id = int(getattr(getattr(target, "config", None), "pad_token_id", 0) or 0)
    input_ids = torch.full((2, input_len), pad_id, dtype=torch.long, device=device)
    input_ids[0, : len(row0_input)] = torch.tensor(row0_input, dtype=torch.long, device=device)
    input_ids[1, : len(row1_input)] = torch.tensor(row1_input, dtype=torch.long, device=device)

    total_seq = batched_cache_len + input_len
    attention_mask = torch.zeros((2, total_seq), dtype=torch.long, device=device)
    position_ids = torch.zeros((2, input_len), dtype=torch.long, device=device)

    # Row 0: attends to real prefix [0, L_minus_1) only; mask out junk [L_minus_1, batched_cache_len).
    attention_mask[0, :L_minus_1] = 1
    attention_mask[0, batched_cache_len : batched_cache_len + len(row0_input)] = 1
    position_ids[0, : len(row0_input)] = torch.arange(
        L_minus_1, L_minus_1 + len(row0_input), dtype=torch.long, device=device,
    )
    # Pad positions kept at 0 (masked).

    # Row 1: attends to all cache slots [0, batched_cache_len) — the lifted slots
    # are kept "live" intentionally; the row-1 commit guard handles correctness.
    attention_mask[1, :batched_cache_len] = 1
    attention_mask[1, batched_cache_len : batched_cache_len + len(row1_input)] = 1
    # Row 1's first input position is L_minus_1 + n_1 + 1 (the position right after
    # the assumed bonus; the bonus itself was placed into the cache).
    position_ids[1, : len(row1_input)] = torch.arange(
        L_minus_1 + n_1 + 1,
        L_minus_1 + n_1 + 1 + len(row1_input),
        dtype=torch.long, device=device,
    )

    # ---- B=2 forward ----
    t_verify = time.perf_counter_ns()
    out = target(
        input_ids,
        past_key_values=batched_cache,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
    )
    verify_us = (time.perf_counter_ns() - t_verify) / 1000.0
    batched_cache = out.past_key_values
    logits = out.logits  # [2, input_len, V]

    # ---- Row 0 commit (always lossless) ----
    row0_logits = logits[0]  # [input_len, V]
    # Logit at index i predicts position L_minus_1 + i + 1.
    # First n_1+1 positions (0..n_1) are real; later positions are padding.
    n_pre = 1  # the anchor token is row0_input[0] = prefix[L_minus_1]
    row0_preds = row0_logits[n_pre - 1 : n_pre - 1 + n_1 + 1].argmax(dim=-1).tolist()
    accepted: list[int] = []
    rejected = False
    for i, tok in enumerate(draft1):
        pred = int(row0_preds[i])
        if tok == pred:
            accepted.append(tok)
        else:
            accepted.append(pred)
            rejected = True
            break
    if not rejected:
        accepted.append(int(row0_preds[n_1]))  # bonus

    n_accepted_row0 = len(accepted) - (0 if rejected else 1)
    bonus_pred = int(row0_preds[n_1])

    n_emitted_row0, _ = _eos_truncate_and_extend(
        prefix, accepted, eos_token_ids, prompt_len, max_new_tokens,
    )

    # ---- Build clean B=1 target_cache covering the row-0 commit ----
    # Real prefix slots [0, L_minus_1) + row-0 new K/V slots
    # [batched_cache_len, batched_cache_len + n_emitted_row0).
    selected = list(range(L_minus_1)) + list(
        range(batched_cache_len, batched_cache_len + n_emitted_row0)
    )
    new_target_cache = _select_row_and_slots(batched_cache, row_idx=0, slot_indices=selected)
    new_target_cache_len = len(selected)

    # ---- Row-1 commit attempt ----
    # Only proceed when row 0 fully accepted *and* row-0's committed prefix
    # contains every token from `accepted` (i.e., no EOS truncation, no budget
    # cap). When row 0 hit EOS partway through draft1, the assumed extension
    # row 1 is built on top of is wrong, so row 1 must not commit.
    row1_attempted = False
    row1_certified = False
    n_emitted_row1 = 0
    reverify_us = 0.0
    row0_hit_eos = n_emitted_row0 < len(accepted)
    if (not rejected) and n_accepted_row0 == n_1 and not row0_hit_eos:
        row1_attempted = True
        # Reverify: run a single-token B=1 forward at position L_minus_1 + n_1 + 1
        # using TRUE KV from row 0's outputs.
        # New cache covers slots [0, L_minus_1 + n_1 + 1) = real prefix + row-0 new K/V.
        true_slots = list(range(L_minus_1)) + list(
            range(batched_cache_len, batched_cache_len + n_1 + 1)
        )
        true_kv_after_row0 = _select_row_and_slots(
            batched_cache, row_idx=0, slot_indices=true_slots,
        )

        # Feed [bonus_pred] (the TRUE bonus emitted by row 0) at position
        # L_minus_1 + n_1 + 1. Compare to row 1's first logit (at row 1 input idx 0).
        verify_input = torch.tensor([[bonus_pred]], dtype=torch.long, device=device)
        verify_pos = torch.tensor(
            [[L_minus_1 + n_1 + 1]], dtype=torch.long, device=device,
        )
        verify_attn = torch.ones(
            (1, cache_length(true_kv_after_row0) + 1), dtype=torch.long, device=device,
        )
        t_re = time.perf_counter_ns()
        re_out = target(
            verify_input,
            past_key_values=true_kv_after_row0,
            attention_mask=verify_attn,
            position_ids=verify_pos,
            use_cache=True,
        )
        reverify_us = (time.perf_counter_ns() - t_re) / 1000.0
        true_argmax = int(re_out.logits[0, 0].argmax(dim=-1).item())
        row1_first_argmax = int(logits[1, 0].argmax(dim=-1).item())

        # Bonus placeholder mismatch check: row 1 was fed bonus_placeholder but
        # the true bonus is bonus_pred. If they differ, row 1's first logit is
        # not even comparable, so we must skip.
        if bonus_placeholder != bonus_pred:
            row1_certified = False
        elif true_argmax != row1_first_argmax:
            row1_certified = False
        else:
            row1_certified = True

        if row1_certified:
            row1_logits = logits[1]
            row1_preds = row1_logits[0 : n_2 + 1].argmax(dim=-1).tolist()
            row1_accepted: list[int] = []
            row1_rejected = False
            for i, tok in enumerate(draft2):
                pred = int(row1_preds[i])
                if tok == pred:
                    row1_accepted.append(tok)
                else:
                    row1_accepted.append(pred)
                    row1_rejected = True
                    break
            if not row1_rejected:
                row1_accepted.append(int(row1_preds[n_2]))

            # Commit row 1 tokens — prefix grows. Budget may zero this out if
            # row 0's commits already filled max_new_tokens.
            n_emitted_row1, _ = _eos_truncate_and_extend(
                prefix, row1_accepted, eos_token_ids, prompt_len, max_new_tokens,
            )
            if n_emitted_row1 == 0:
                # Budget exhausted; no row-1 KV to append.
                new_target_cache_len = L_minus_1 + n_emitted_row0
                # Crop / no-op below handles the rest. Skip the splice.
                row1_certified = False  # nothing was actually committed from row 1
                row1_extra = None
            else:
                # Rebuild target_cache to include row-1's tokens. We use row 1's
                # K/V from the B=2 forward; these depend on the rotary-mismatched
                # lifted slots and are the brief's accepted compromise. Step 5's
                # lossless gate will surface any divergence.
                extra_slots = list(
                    range(batched_cache_len, batched_cache_len + n_emitted_row1)
                )
                row1_extra = _select_row_and_slots(
                    batched_cache, row_idx=1, slot_indices=extra_slots,
                )
            if row1_extra is not None:
                # Final cache = [0, L_minus_1 + n_emitted_row0) ∪ row 1's commit slots.
                # Row 1's accepted-token slots live at batched_cache_len + i in the
                # unified cache for i in [0, n_emitted_row1). They carry the brief's
                # accepted approximation (computed from rotary-mismatched lifted slots).
                new_target_cache_len = L_minus_1 + n_emitted_row0 + n_emitted_row1
                for layer_idx, (k0, v0) in enumerate(layer_keys_values(new_target_cache)):
                    k1, v1 = list(layer_keys_values(row1_extra))[layer_idx]
                    set_layer_keys_values(
                        new_target_cache, layer_idx,
                        torch.cat([k0, k1], dim=-2).contiguous(),
                        torch.cat([v0, v1], dim=-2).contiguous(),
                    )
                if hasattr(new_target_cache, "_seen_tokens"):
                    new_target_cache._seen_tokens = new_target_cache_len

    # Final crop to prefix-1 (PLD cache convention).
    new_prefix_len = len(prefix)
    target_seq_len_after_commit = max(0, new_prefix_len - 1)
    if new_target_cache_len > target_seq_len_after_commit:
        crop_dynamic_cache(new_target_cache, target_seq_len_after_commit)
        new_target_cache_len = target_seq_len_after_commit

    return {
        "fall_back_to_pld": False,
        "target_cache": new_target_cache,
        "target_cache_len": new_target_cache_len,
        "n_emitted_row0": n_emitted_row0,
        "n_emitted_row1": n_emitted_row1,
        "n_accepted_row0": n_accepted_row0,
        "row1_attempted": row1_attempted,
        "row1_certified": row1_certified,
        "rejected": rejected,
        "verify_us": verify_us,
        "reverify_us": reverify_us,
        "n_1": n_1,
        "n_2": n_2,
        "follow1": follow1,
    }


@torch.no_grad()
def pvp_speculative_ar(
    prompt_ids: torch.Tensor,
    target,
    assistant: Optional[object],
    max_new_tokens: int,
    eos_token_ids: list[int],
    *,
    config: BlazEditConfig,
    method_name: str,
    tokenizer=None,
    K: int = 2,
) -> DecodeResult:
    """PVP decode loop. Signature matches ``blazedit_speculative_ar`` (assistant is ignored).

    K is the chain depth; the brief covers K=2 with a generalization sketch for K>=3.
    The current implementation runs K=2; K>=3 falls back to plain PLD for now (the
    higher-K logic is straightforward but adds complexity not yet exercised by Step 7).
    """
    del assistant, method_name, tokenizer  # unused
    prompt_ids_list = (
        prompt_ids.tolist() if prompt_ids.dim() == 1 else prompt_ids[0].tolist()
    )
    prefix: list[int] = list(prompt_ids_list)
    prompt_len = len(prefix)

    target_cache = None
    target_cache_len = 0
    steps: list[StepRecord] = []
    t_start = time.perf_counter_ns()
    step_idx = 0

    pvp_K = max(1, int(K))

    while len(prefix) < prompt_len + max_new_tokens:
        t_step = time.perf_counter_ns()

        # Chain PLD lookup. depth=K but we currently only batch K=2; higher K
        # falls back to plain PLD with depth=K=1 for now (future work).
        t_lookup = time.perf_counter_ns()
        chain = chain_pld_lookup(
            prefix,
            n_match=config.max_matching_ngram_size,
            n_draft=config.micro_draft_tokens,
            depth=pvp_K,
        )
        chain_depth = len(chain)
        proposal_us = (time.perf_counter_ns() - t_lookup) / 1000.0

        do_pvp = pvp_K >= 2 and chain_depth >= 2

        if do_pvp:
            res = _pvp_step_batched(
                prefix=prefix,
                prompt_len=prompt_len,
                target=target,
                target_cache=target_cache,
                target_cache_len=target_cache_len,
                chain=chain,
                max_new_tokens=max_new_tokens,
                eos_token_ids=eos_token_ids,
            )
            if res.get("fall_back_to_pld"):
                do_pvp = False
            else:
                target_cache = res["target_cache"]
                target_cache_len = res["target_cache_len"]
                t_step_end = time.perf_counter_ns()
                steps.append(
                    StepRecord(
                        method="vantage_pvp",
                        step=step_idx,
                        k=res["n_1"],
                        n_accepted_drafts=res["n_accepted_row0"],
                        n_emitted=res["n_emitted_row0"] + res["n_emitted_row1"],
                        rejected=res["rejected"],
                        node_type=None,
                        deepest_type=None,
                        wall_us=(t_step_end - t_step) / 1000.0,
                        verify_us=res["verify_us"] + res["reverify_us"],
                        proposal_kind="vantage_pvp_k2",
                        proposal_match_len=chain[0].match_len,
                        proposal_us=proposal_us,
                        proposal_tokens=res["n_1"],
                        proposal_source_start_token=res["follow1"] - chain[0].match_len,
                        proposal_follow_start_token=res["follow1"],
                        pld_exact_hit=True,
                        pld_variant_triggered=True,
                        pld_candidate_accepted_len=res["n_accepted_row0"],
                        pld_variant="pvp_k2",
                        # Extra PVP-specific diagnostics piggy-back on existing fields:
                        proposal_neural_draft_tokens=res["n_emitted_row1"],
                        mtp_extra_accepted_drafts=int(res["row1_certified"]),
                        mtp_predicted_tokens=res["n_2"],
                    )
                )
                step_idx += 1
                # EOS / no-progress break — _target_verify_step bakes this in
                # via _eos_truncate_and_extend, but the PVP path needs its own
                # check since we exited the step via `continue`.
                if (res["n_emitted_row0"] + res["n_emitted_row1"]) == 0:
                    break
                if prefix[-1] in eos_token_ids:
                    break
                continue

        # Plain PLD step (chain_depth < 2, K=1, or PVP fell back).
        drafts = chain[0].draft if chain else []
        (
            target_cache,
            target_cache_len,
            result,
            n_emitted,
            _accepted_capped,
            verify_us,
        ) = _target_verify_step(
            prefix=prefix,
            prompt_len=prompt_len,
            target=target,
            target_cache=target_cache,
            target_cache_len=target_cache_len,
            drafts=drafts,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
        )
        t_step_end = time.perf_counter_ns()

        steps.append(
            StepRecord(
                method="vantage_pvp",
                step=step_idx,
                k=len(drafts),
                n_accepted_drafts=int(result.n_accepted_drafts),
                n_emitted=int(n_emitted),
                rejected=bool(result.rejected),
                node_type=None,
                deepest_type=None,
                wall_us=(t_step_end - t_step) / 1000.0,
                verify_us=float(verify_us),
                proposal_kind="blazedit_pld",
                proposal_match_len=chain[0].match_len if chain else 0,
                proposal_us=proposal_us,
                proposal_tokens=len(drafts),
                proposal_source_start_token=(chain[0].source_start if chain else -1),
                proposal_follow_start_token=(chain[0].follow_start if chain else -1),
                pld_exact_hit=bool(drafts),
                pld_variant_triggered=False,
                pld_candidate_accepted_len=int(result.n_accepted_drafts),
                pld_variant="pvp_fallback_pld",
            )
        )
        step_idx += 1

        if n_emitted == 0:
            break
        last_tok = prefix[-1]
        if last_tok in eos_token_ids:
            break

    t_end = time.perf_counter_ns()
    return DecodeResult(
        output_token_ids=prefix,
        output_text="",
        n_new_tokens=len(prefix) - prompt_len,
        steps=steps,
        wall_us_total=(t_end - t_start) / 1000.0,
    )


# ---------------------------------------------------------------------------
# Method-name dispatch helpers (so eval scripts can pick up "vantage_pvp_*")
# ---------------------------------------------------------------------------


import re

_PVP_METHOD_RE = re.compile(r"vantage_pvp_k(?P<k>\d+)_w(?P<w>\d+)_n(?P<n>\d+)$")


def is_pvp_method(method: str) -> bool:
    return bool(_PVP_METHOD_RE.fullmatch(method))


def parse_pvp_method(method: str, *, default_ngram_size: int = 10) -> tuple[BlazEditConfig, int]:
    """Parse a PVP method name into a BlazEditConfig (PLD-shaped) and chain depth K."""
    m = _PVP_METHOD_RE.fullmatch(method)
    if not m:
        raise ValueError(f"not a PVP method: {method!r}")
    return (
        BlazEditConfig(
            mode="pld",
            micro_draft_tokens=int(m.group("w")),
            max_num_run=1,
            max_matching_ngram_size=int(m.group("n")),
        ),
        int(m.group("k")),
    )
