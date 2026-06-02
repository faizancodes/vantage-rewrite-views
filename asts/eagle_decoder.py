"""EAGLE-1 inference path: speculative AR using a trained EagleHead as draft.

Differs from `decoder.speculative_ar` in the draft-phase shape:
  - Target produces hidden states (penultimate layer) for the prefix
  - EagleHead, conditioned on the last hidden state, autoregressively predicts
    next-position hidden states for k future positions
  - Each predicted hidden state is projected through target's LM head to get
    next-token logits → argmax → draft token

The verify phase is unchanged (target runs forward on prefix_remainder + drafts;
greedy rejection accepts up to first mismatch, takes target's prediction as
correction or bonus).

Cache management: target maintains its own KV cache; EAGLE has no cache (it's
a single block run autoregressively on hidden states).

This module deliberately does NOT modify decoder.py — fewer chances of
regression on the lossless invariant for the existing fixed-k path.
"""

from __future__ import annotations

import time

import torch
from transformers import DynamicCache

from .decoder import (
    DecodeResult,
    StepRecord,
    crop_dynamic_cache,
)


def _argmax_int(logits_row: torch.Tensor) -> int:
    return int(logits_row.argmax(dim=-1).item())


@torch.no_grad()
def eagle_speculative_ar(
    prompt_ids: torch.Tensor,
    target,
    eagle_head,
    max_new_tokens: int,
    eos_token_ids: list[int],
    method_name: str,
    k_resolver,
    parse_callback=None,
) -> DecodeResult:
    """Speculative AR with EAGLE-1 head as draft.

    Args:
        prompt_ids: shape [seq] or [1, seq]
        target: HF target model loaded with output_hidden_states=True capable
        eagle_head: trained EagleHead instance
        k_resolver: callable(prefix_ids: list[int]) → (k, node_type, deepest_type)
            Same signature as decoder.speculative_ar. For fixed-k EAGLE, returns
            constant k. For ASTS-EAGLE, queries the AST.
        parse_callback: optional, called with prefix tokens after each step
    """
    device = next(target.parameters()).device
    prefix = prompt_ids.tolist()
    target_norm = target.model.norm

    # Persistent state across outer steps:
    target_cache = None              # target's KV cache (always covers full prefix at step start)
    prefix_h_buffer: torch.Tensor | None = None  # [1, P, H] teacher hidden states (pre-norm)
    last_logits: torch.Tensor | None = None      # target's logits for the next position

    steps: list[StepRecord] = []
    t_start = time.perf_counter_ns()

    step_idx = 0
    while len(prefix) < len(prompt_ids) + max_new_tokens:
        t_step_start = time.perf_counter_ns()
        P = len(prefix)

        # ---- Resolve k ----
        t_parse_0 = time.perf_counter_ns()
        k, node_type, deepest_type = k_resolver(prefix)
        t_parse_1 = time.perf_counter_ns()
        parse_us = (t_parse_1 - t_parse_0) / 1000.0
        if k < 1:
            k = 1

        # ---- Bring target up to date ----
        target_prefill_us = 0.0
        if target_cache is None:
            # First step: full prefill on the prompt.
            t0 = time.perf_counter_ns()
            full_in = torch.tensor([prefix], device=device)
            t_out = target(full_in, output_hidden_states=True, use_cache=True)
            target_cache = t_out.past_key_values
            prefix_h_buffer = t_out.hidden_states[-2]  # [1, P, H]
            last_logits = t_out.logits[0, -1].clone()
            target_prefill_us = (time.perf_counter_ns() - t0) / 1000.0
        else:
            # Subsequent steps: target_cache currently covers prefix[:-1]
            # (anchored from prior step's verify rollback). The new last token
            # is the bonus/correction we appended at the end of prior step.
            # Single-token forward extends the cache to cover full prefix and
            # gives us:
            #   - hidden state for prefix[-1] (append to buffer)
            #   - logits for next position (= first draft of this step)
            t0 = time.perf_counter_ns()
            last_tok = torch.tensor([[prefix[-1]]], device=device)
            t_out = target(
                last_tok,
                past_key_values=target_cache,
                output_hidden_states=True,
                use_cache=True,
            )
            target_cache = t_out.past_key_values  # now covers full prefix
            new_h = t_out.hidden_states[-2]  # [1, 1, H]
            assert prefix_h_buffer is not None
            prefix_h_buffer = torch.cat([prefix_h_buffer, new_h], dim=1)
            last_logits = t_out.logits[0, -1].clone()
            target_prefill_us = (time.perf_counter_ns() - t0) / 1000.0

        # Now: target_cache covers full prefix (length P), prefix_h_buffer has length P,
        # last_logits is the distribution for position P (= first draft).

        # ---- Draft phase ----
        # KV cache wasn't reliably integrating with HF's Qwen2 layer at our
        # transformers version, so we use the FULL-CONTEXT approach: feed the
        # entire eagle_input sequence each time. O(P+i) per chain step but
        # correct. This matches the v1 working baseline at 44.3 t/s.
        t_draft_0 = time.perf_counter_ns()
        drafts: list[int] = []
        # Draft 1: argmax of teacher's existing logits.
        assert last_logits is not None
        drafts.append(_argmax_int(last_logits))

        # eagle_input grows by 1 each chain iter; positions are 0..len-1.
        assert prefix_h_buffer is not None
        eagle_input = prefix_h_buffer  # [1, P, H]
        for i in range(1, k):
            seq_len_now = eagle_input.shape[1]
            pos_ids = torch.arange(seq_len_now, device=device, dtype=torch.long).unsqueeze(0)
            eagle_pred_full = eagle_head(eagle_input, position_ids=pos_ids)  # [1, seq, H]
            new_h = eagle_pred_full[:, -1:, :]
            new_h_post = target_norm(new_h)
            logits = target.lm_head(new_h_post)
            drafts.append(_argmax_int(logits[0, -1]))
            eagle_input = torch.cat([eagle_input, new_h], dim=1)
        t_draft_1 = time.perf_counter_ns()
        draft_us = (t_draft_1 - t_draft_0) / 1000.0

        # ---- Verify phase ----
        # target_cache covers full prefix (length P). Feed drafts only (no anchor)
        # since cache already has state for prefix[-1].
        target_input = torch.tensor([drafts], device=device)
        t_verify_0 = time.perf_counter_ns()
        t_out_v = target(
            target_input,
            past_key_values=target_cache,
            output_hidden_states=True,
            use_cache=True,
        )
        target_cache = t_out_v.past_key_values  # now covers P + k positions
        t_verify_1 = time.perf_counter_ns()
        verify_us = (t_verify_1 - t_verify_0) / 1000.0

        # Greedy rejection — but with our setup, drafts[0] always equals
        # argmax(last_logits) which equals target's prediction at position P-1.
        # In verify, target's logits[0, p] = distribution at position P + p + 1.
        # So target's prediction for position P (= drafts[0]) is NOT in verify
        # output; it's in last_logits. drafts[0] is GUARANTEED to match.
        # For drafts[1..k-1], compare against verify_logits[0, p-1].
        v_logits = t_out_v.logits  # [1, k, V]
        accepted_tokens: list[int] = [drafts[0]]
        n_accepted_drafts = 1
        rejected = False
        for i in range(1, k):
            target_pred = int(v_logits[0, i - 1].argmax(dim=-1).item())
            if drafts[i] == target_pred:
                accepted_tokens.append(drafts[i])
                n_accepted_drafts += 1
            else:
                accepted_tokens.append(target_pred)
                rejected = True
                break
        if not rejected:
            # All k drafts accepted. Bonus = argmax of verify_logits[0, k-1]
            # (target's distribution AFTER drafts[k-1] at position P + k).
            bonus = int(v_logits[0, k - 1].argmax(dim=-1).item())
            accepted_tokens.append(bonus)

        # ---- Update prefix (with budget cap + EOS truncation) ----
        # First truncate at first EOS so we match vanilla's stopping behavior
        # (vanilla stops at EOS; without this, spec can emit tokens AFTER EOS).
        eos_truncated = list(accepted_tokens)
        for i, tk in enumerate(eos_truncated):
            if tk in eos_token_ids:
                eos_truncated = eos_truncated[: i + 1]
                break
        budget = (len(prompt_ids) + max_new_tokens) - len(prefix)
        if budget < len(eos_truncated):
            accepted_capped = eos_truncated[:budget]
        else:
            accepted_capped = eos_truncated
        prefix.extend(accepted_capped)
        n_emitted = len(accepted_capped)

        # ---- Update prefix_h_buffer ----
        # Verify gave us target hidden states for positions P, P+1, ..., P+k-1
        # (from drafts[0..k-1]). For ACCEPTED drafts (positions P..P+n_accepted_drafts-1),
        # these hidden states are valid for the new prefix. Append.
        # For rejected/correction position and beyond, hidden states are for
        # the WRONG token; don't append. Bonus's hidden state isn't computed
        # at all. Both will be filled in next step's single-token target forward.
        if n_accepted_drafts > 0:
            accepted_h = t_out_v.hidden_states[-2][:, :n_accepted_drafts, :].contiguous()
            assert prefix_h_buffer is not None
            prefix_h_buffer = torch.cat([prefix_h_buffer, accepted_h], dim=1)

        # ---- Update target_cache to anchored state ----
        # After verify, cache covers P + k positions. New prefix has
        # P + n_accepted_drafts + 1 tokens (n_accepted_drafts accepted drafts + bonus/correction).
        # Anchored cache should cover len(new_prefix) - 1 positions.
        crop_dynamic_cache(target_cache, len(prefix) - 1)

        # last_logits is invalidated; will be recomputed at start of next step
        # via the target single-token forward on the bonus/correction.
        last_logits = None

        if parse_callback is not None:
            t_pcb_0 = time.perf_counter_ns()
            parse_callback(prefix)
            t_pcb_1 = time.perf_counter_ns()
            parse_us += (t_pcb_1 - t_pcb_0) / 1000.0

        t_step_end = time.perf_counter_ns()
        steps.append(StepRecord(
            method=method_name,
            step=step_idx,
            k=k,
            n_accepted_drafts=n_accepted_drafts,
            n_emitted=n_emitted,
            rejected=rejected,
            node_type=node_type,
            deepest_type=deepest_type,
            wall_us=(t_step_end - t_step_start) / 1000.0,
            draft_us=draft_us,
            verify_us=verify_us,
            parse_us=parse_us + target_prefill_us,
        ))
        step_idx += 1

        if any(t in eos_token_ids for t in accepted_capped):
            break

    t_end = time.perf_counter_ns()
    return DecodeResult(
        output_token_ids=prefix,
        output_text="",
        n_new_tokens=len(prefix) - len(prompt_ids),
        steps=steps,
        wall_us_total=(t_end - t_start) / 1000.0,
    )


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def fixed_eagle_spec(
    prompt_ids: torch.Tensor,
    target,
    eagle_head,
    max_new_tokens: int,
    eos_token_ids: list[int],
    k: int = 8,
) -> DecodeResult:
    """Fixed-k EAGLE speculation."""
    def _resolver(_prefix):
        return k, None, None
    return eagle_speculative_ar(
        prompt_ids=prompt_ids,
        target=target,
        eagle_head=eagle_head,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        method_name=f"eagle_k{k}",
        k_resolver=_resolver,
    )


def asts_eagle_spec(
    prompt_ids: torch.Tensor,
    target,
    eagle_head,
    max_new_tokens: int,
    eos_token_ids: list[int],
    tokenizer,
    ast_policy,
) -> DecodeResult:
    """ASTS-Spec with EAGLE draft: variable-length speculation gated by AST node type."""
    def _decode_to_bytes(prefix_ids: list[int]) -> bytes:
        return tokenizer.decode(prefix_ids, skip_special_tokens=False).encode("utf-8")

    def _resolver(prefix_ids: list[int]):
        ast_policy.update(_decode_to_bytes(prefix_ids))
        ctx = ast_policy.context_at_cursor()
        return ctx.k, ctx.node_type, ctx.deepest_type

    return eagle_speculative_ar(
        prompt_ids=prompt_ids,
        target=target,
        eagle_head=eagle_head,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        method_name="asts_eagle",
        k_resolver=_resolver,
    )


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_eagle_checkpoint(checkpoint_path: str, device: str = "cuda", dtype: str = "bfloat16"):
    """Load a trained EagleHead from a checkpoint produced by eagle_train.train()."""
    from .eagle import EagleConfig, EagleHead

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        dtype
    ]
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    eagle_cfg_dict = ckpt["eagle_config"]
    eagle_cfg = EagleConfig(**eagle_cfg_dict)
    head = EagleHead(eagle_cfg)
    head.load_state_dict(ckpt["state_dict"])
    head = head.to(device).to(torch_dtype)
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head, eagle_cfg, ckpt
