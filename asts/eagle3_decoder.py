"""EAGLE-3 inference path: speculative AR using a SpecForge-trained draft.

Differs from `eagle_decoder.py` (EAGLE-1) in three places:
  - Multi-layer feature fusion: target's hidden_states[low, mid, last] are
    concatenated and projected through draft.fc (instead of taking only the
    penultimate layer).
  - Draft hidden dim: same H as target (3584 for Qwen2.5-Coder-7B), but the
    draft has its own 16K-token vocab subset; predictions are mapped back
    to the full 152K target vocab via the d2t buffer.
  - Backbone interface: draft.backbone(input_embeds, hidden_states, ...)
    takes BOTH the embedded token AND the (predicted-or-fused) hidden state
    per position, then concats them inside.

Lossless guarantee is preserved by greedy rejection on argmax — the same
mechanism `eagle_decoder.eagle_speculative_ar` uses.

For now the draft chain is full-context (cache_hidden=None, past_key_values=None);
that mirrors the simpler EAGLE-1 implementation. EAGLE-3's incremental
cache_hidden mechanism is a future optimization.
"""

from __future__ import annotations

import time

import torch

from .decoder import (
    DecodeResult,
    StepRecord,
    crop_dynamic_cache,
)


# Index into HF's `output_hidden_states` tuple, which has shape
# [embedding_output, layer_0_out, layer_1_out, ...]. SpecForge defines:
#   offset = 1
#   low_aux_layer = 1 + offset
#   mid_aux_layer = num_layers // 2 - 1 + offset
#   last_aux_layer = num_layers - 4 + offset
def eagle3_aux_indices(num_layers: int) -> tuple[int, int, int]:
    offset = 1
    low = 1 + offset
    mid = num_layers // 2 - 1 + offset
    last = num_layers - 4 + offset
    return low, mid, last


def _argmax_int(logits_row: torch.Tensor) -> int:
    return int(logits_row.argmax(dim=-1).item())


def _run_draft_backbone(
    draft,
    input_embeds: torch.Tensor,  # [1, T, H]
    hidden_states: torch.Tensor,  # [1, T, H]
) -> torch.Tensor:
    """Single full-context backbone call. Returns [1, T, H]."""
    device = hidden_states.device
    seq_len = hidden_states.shape[1]
    pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    attn_mask = draft.prepare_decoder_attention_mask(
        attention_mask=torch.ones((1, seq_len), dtype=torch.bool, device=device),
        hidden_states=hidden_states,
        batch_size=1,
        seq_length=seq_len,
        past_key_values_length=0,
    )
    return draft.backbone(
        input_embeds=input_embeds.to(hidden_states.dtype),
        hidden_states=hidden_states,
        cache_hidden=None,
        attention_mask=attn_mask,
        position_ids=pos_ids,
        past_key_values=None,
        use_cache=False,
    )


@torch.no_grad()
def eagle3_speculative_ar(
    prompt_ids: torch.Tensor,
    target,
    draft,
    max_new_tokens: int,
    eos_token_ids: list[int],
    method_name: str,
    k_resolver,
    parse_callback=None,
) -> DecodeResult:
    """Speculative AR with EAGLE-3 head as draft.

    Args:
        prompt_ids: shape [seq] or [1, seq]
        target: HF model loaded with output_hidden_states=True capable
        draft: specforge.AutoEagle3DraftModel-loaded draft (LlamaForCausalLMEagle3)
        k_resolver: callable(prefix_ids: list[int]) → (k, node_type, deepest_type)
            Same signature as decoder.speculative_ar.
        parse_callback: optional, called with prefix tokens after each step
    """
    device = next(target.parameters()).device
    prefix = prompt_ids.tolist()

    num_target_layers = target.config.num_hidden_layers
    low_idx, mid_idx, last_idx = eagle3_aux_indices(num_target_layers)
    d2t: torch.Tensor = draft.d2t  # [16000], int64; full_id = draft_idx + d2t[draft_idx]

    # Persistent state across outer steps:
    target_cache = None                          # target's KV cache (covers full prefix at step start)
    fused_buffer: torch.Tensor | None = None     # [1, P, H] post-fc target features
    last_logits: torch.Tensor | None = None      # target's logits for the next position

    steps: list[StepRecord] = []
    t_start = time.perf_counter_ns()

    step_idx = 0
    while len(prefix) < len(prompt_ids) + max_new_tokens:
        t_step_start = time.perf_counter_ns()

        t_parse_0 = time.perf_counter_ns()
        k, node_type, deepest_type = k_resolver(prefix)
        t_parse_1 = time.perf_counter_ns()
        parse_us = (t_parse_1 - t_parse_0) / 1000.0
        if k < 1:
            k = 1

        # ---- Bring target up to date and collect aux hidden states ----
        target_prefill_us = 0.0
        if target_cache is None:
            t0 = time.perf_counter_ns()
            full_in = torch.tensor([prefix], device=device)
            t_out = target(full_in, output_hidden_states=True, use_cache=True)
            target_cache = t_out.past_key_values
            h_low = t_out.hidden_states[low_idx]    # [1, P, H]
            h_mid = t_out.hidden_states[mid_idx]
            h_last = t_out.hidden_states[last_idx]
            three = torch.cat([h_low, h_mid, h_last], dim=-1)  # [1, P, 3H]
            fused_buffer = draft.fc(three.to(draft.fc.weight.dtype))  # [1, P, H]
            last_logits = t_out.logits[0, -1].clone()
            target_prefill_us = (time.perf_counter_ns() - t0) / 1000.0
        else:
            # target_cache currently covers prefix[:-1]. Extend by one token
            # and pull aux hidden states for the new position only.
            t0 = time.perf_counter_ns()
            last_tok = torch.tensor([[prefix[-1]]], device=device)
            t_out = target(
                last_tok,
                past_key_values=target_cache,
                output_hidden_states=True,
                use_cache=True,
            )
            target_cache = t_out.past_key_values  # now covers full prefix
            h_low = t_out.hidden_states[low_idx]    # [1, 1, H]
            h_mid = t_out.hidden_states[mid_idx]
            h_last = t_out.hidden_states[last_idx]
            three = torch.cat([h_low, h_mid, h_last], dim=-1)  # [1, 1, 3H]
            new_fused = draft.fc(three.to(draft.fc.weight.dtype))  # [1, 1, H]
            assert fused_buffer is not None
            fused_buffer = torch.cat([fused_buffer, new_fused], dim=1)
            last_logits = t_out.logits[0, -1].clone()
            target_prefill_us = (time.perf_counter_ns() - t0) / 1000.0

        # Now: target_cache covers full prefix (P tokens), fused_buffer has length P,
        # last_logits is the distribution for position P (= first draft via teacher's argmax).

        # ---- Draft phase ----
        t_draft_0 = time.perf_counter_ns()
        drafts: list[int] = []
        # Draft 1: argmax of teacher's existing logits — same as EAGLE-1.
        # (This is provably correct because target.logits[0, -1] IS the
        # distribution at position P, which is the first draft we'd like to
        # propose. Using this guarantees first-draft acceptance and matches
        # vanilla AR for that token.)
        assert last_logits is not None
        drafts.append(_argmax_int(last_logits))

        # For drafts 2..k, run the EAGLE-3 chain. We seed the chain by running
        # backbone once on the prefix-only context; the last-position output
        # is the predicted h at position P (the slot where draft_0 lives).
        # Then for i = 1..k-1, we append draft_{i-1}'s embedding and the prior
        # predicted h, run backbone again, and sample draft_i from the new
        # last-position output.
        if k > 1:
            assert fused_buffer is not None
            chain_input_embeds = draft.embed_tokens(
                torch.tensor([prefix], device=device, dtype=torch.long)
            )  # [1, P, H]
            chain_h = fused_buffer  # [1, P, H]

            # Seed: predicted h at position P (where draft_0 sits).
            seed_out = _run_draft_backbone(draft, chain_input_embeds, chain_h)
            predicted_h = seed_out[:, -1:, :]  # [1, 1, H]

            for i in range(1, k):
                # Extend with draft_{i-1}'s embedding and its (just-predicted) h.
                chain_input_embeds = torch.cat(
                    [
                        chain_input_embeds,
                        draft.embed_tokens(
                            torch.tensor([[drafts[i - 1]]], device=device, dtype=torch.long)
                        ),
                    ],
                    dim=1,
                )
                chain_h = torch.cat([chain_h, predicted_h], dim=1)

                step_out = _run_draft_backbone(draft, chain_input_embeds, chain_h)
                predicted_h = step_out[:, -1:, :]
                logits = draft.compute_logits(predicted_h)  # [1, 1, draft_vocab=16000]
                draft_idx = int(logits[0, -1].argmax(dim=-1).item())
                full_id = draft_idx + int(d2t[draft_idx].item())
                drafts.append(full_id)
        t_draft_1 = time.perf_counter_ns()
        draft_us = (t_draft_1 - t_draft_0) / 1000.0

        # ---- Verify phase ----
        # target_cache covers full prefix (length P). Feed drafts only.
        target_input = torch.tensor([drafts], device=device)
        t_verify_0 = time.perf_counter_ns()
        t_out_v = target(
            target_input,
            past_key_values=target_cache,
            output_hidden_states=True,
            use_cache=True,
        )
        target_cache = t_out_v.past_key_values  # covers P + k positions
        t_verify_1 = time.perf_counter_ns()
        verify_us = (t_verify_1 - t_verify_0) / 1000.0

        # Greedy rejection. Same logic as EAGLE-1: drafts[0] is teacher-derived
        # (always matches), drafts[1..] compared against target's argmax.
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
            bonus = int(v_logits[0, k - 1].argmax(dim=-1).item())
            accepted_tokens.append(bonus)

        # ---- Update prefix (with budget cap + EOS truncation) ----
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

        # ---- Update fused_buffer for accepted positions ----
        # Verify gave us target's hidden states at positions P..P+k-1 for the
        # tokens we drafted. For ACCEPTED drafts (positions P..P+n_accepted-1),
        # those features are valid for the new prefix; project them and append.
        # For rejected/correction position and beyond, hidden states are for
        # the WRONG token; don't append. Bonus's hidden state isn't computed at
        # all. Both will be filled in next step's single-token target forward.
        if n_accepted_drafts > 0:
            h_low_v = t_out_v.hidden_states[low_idx][:, :n_accepted_drafts, :]
            h_mid_v = t_out_v.hidden_states[mid_idx][:, :n_accepted_drafts, :]
            h_last_v = t_out_v.hidden_states[last_idx][:, :n_accepted_drafts, :]
            three_v = torch.cat([h_low_v, h_mid_v, h_last_v], dim=-1).contiguous()
            new_fused = draft.fc(three_v.to(draft.fc.weight.dtype))
            assert fused_buffer is not None
            fused_buffer = torch.cat([fused_buffer, new_fused], dim=1)

        # ---- Anchor target_cache to len(prefix) - 1 ----
        crop_dynamic_cache(target_cache, len(prefix) - 1)
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


def fixed_eagle3_spec(
    prompt_ids: torch.Tensor,
    target,
    draft,
    max_new_tokens: int,
    eos_token_ids: list[int],
    k: int = 8,
) -> DecodeResult:
    """Fixed-k EAGLE-3 speculation."""
    def _resolver(_prefix):
        return k, None, None
    return eagle3_speculative_ar(
        prompt_ids=prompt_ids,
        target=target,
        draft=draft,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        method_name=f"eagle3_k{k}",
        k_resolver=_resolver,
    )


def asts_eagle3_spec(
    prompt_ids: torch.Tensor,
    target,
    draft,
    max_new_tokens: int,
    eos_token_ids: list[int],
    tokenizer,
    ast_policy,
) -> DecodeResult:
    """ASTS-Spec with EAGLE-3 draft: variable-length speculation gated by AST."""
    def _decode_to_bytes(prefix_ids: list[int]) -> bytes:
        return tokenizer.decode(prefix_ids, skip_special_tokens=False).encode("utf-8")

    def _resolver(prefix_ids: list[int]):
        ast_policy.update(_decode_to_bytes(prefix_ids))
        ctx = ast_policy.context_at_cursor()
        return ctx.k, ctx.node_type, ctx.deepest_type

    return eagle3_speculative_ar(
        prompt_ids=prompt_ids,
        target=target,
        draft=draft,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        method_name="asts_eagle3",
        k_resolver=_resolver,
    )
