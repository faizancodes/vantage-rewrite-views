"""Tree-tail EAGLE-1 spec decoder.

Variant of `eagle_decoder.eagle_speculative_ar` where the EAGLE chain is
linear for depths 1..k-1 and branches into top-W candidates at depth k.

Why chain-with-tail-branching (instead of a full tree)?
  EAGLE-1's predicted hidden state at depth d depends only on hidden states
  along the chain — not on the embedding of the chosen token at depth d-1.
  So full tree branches degenerate: depth-2 candidates would be identical
  for every depth-1 sibling. Branching only at the *last* chain position
  preserves chain semantics while letting target select from W competing
  next-tokens at the leaf — capturing the easy "top-1 vs top-2" acceptance
  jump without inventing fictitious branching at intermediate depths.

Tree shape for (k, W):
  depth 0:        1 root node — teacher's argmax (always accepted by target)
  depths 1..k-2:  1 node each — top-1 from EAGLE chain
  depth k-1:      W nodes      — top-W from EAGLE's last chain prediction

Total tree nodes: (k - 1) + W.

Lossless guarantee preserved: same greedy rejection rule applied along the
linear chain, then to the leaf branch. We accept the longest valid path.

Cache management:
  After verify, the target's KV cache extends past the prefix into the tree
  positions [P..P+(k-1+W)-1]. The K/V at P+k-1 corresponds to leaf_tokens[0],
  at P+k to leaf_tokens[1], etc. — all share position_id P+k-1.

  - If chain rejected:       crop to P+n_chain_accepted (only chain part is correct).
  - If leaf_chosen_idx == 0: crop to P+k (chain + leaf[0]'s K/V slot is correct).
  - If leaf_chosen_idx > 0:  crop to P+k-1 (drop all leaves, recompute on next step).

The next outer step's "bring target up to date" forwards however many
tokens are missing between the cache and len(prefix), so multi-token
catchup is handled transparently.
"""

from __future__ import annotations

import time

import torch

from .decoder import (
    DecodeResult,
    StepRecord,
    crop_dynamic_cache,
)


def _argmax_int(logits_row: torch.Tensor) -> int:
    return int(logits_row.argmax(dim=-1).item())


@torch.no_grad()
def tree_eagle_speculative_ar(
    prompt_ids: torch.Tensor,
    target,
    eagle_head,
    max_new_tokens: int,
    eos_token_ids: list[int],
    method_name: str,
    k: int = 2,
    width: int = 2,
    parse_callback=None,
) -> DecodeResult:
    """Linear EAGLE chain of length k with top-`width` branching at the last
    chain position.

    Args:
        prompt_ids: shape [seq] or [1, seq]
        target: HF target model loaded with output_hidden_states=True capable
        eagle_head: trained EagleHead instance
        k: chain length (number of speculative positions, including the
            top-W branch position). Must be >= 2.
        width: branching factor at the LAST chain position.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2 for tree spec; got k={k}")
    if width < 1:
        raise ValueError(f"width must be >= 1; got width={width}")

    device = next(target.parameters()).device
    prompt_ids_list = prompt_ids.tolist() if prompt_ids.dim() == 1 else prompt_ids[0].tolist()
    prefix: list[int] = list(prompt_ids_list)
    target_norm = target.model.norm

    target_cache = None
    prefix_h_buffer: torch.Tensor | None = None
    last_logits: torch.Tensor | None = None

    steps: list[StepRecord] = []
    t_start = time.perf_counter_ns()

    step_idx = 0
    while len(prefix) < len(prompt_ids_list) + max_new_tokens:
        t_step_start = time.perf_counter_ns()
        P = len(prefix)

        # ---- Bring target up to date (multi-token catchup supported) ----
        target_prefill_us = 0.0
        if target_cache is None:
            t0 = time.perf_counter_ns()
            full_in = torch.tensor([prefix], device=device)
            t_out = target(full_in, output_hidden_states=True, use_cache=True)
            target_cache = t_out.past_key_values
            prefix_h_buffer = t_out.hidden_states[-2]
            last_logits = t_out.logits[0, -1].clone()
            target_prefill_us = (time.perf_counter_ns() - t0) / 1000.0
        else:
            current_cache_len = int(target_cache.get_seq_length())
            n_to_forward = len(prefix) - current_cache_len
            if n_to_forward < 1:
                raise RuntimeError(
                    f"cache_len {current_cache_len} >= prefix_len {len(prefix)}; "
                    "cache was not cropped correctly at end of previous step"
                )
            t0 = time.perf_counter_ns()
            catchup_in = torch.tensor(
                [prefix[current_cache_len:]], device=device, dtype=torch.long
            )
            pos_ids = torch.arange(
                current_cache_len, len(prefix), device=device, dtype=torch.long
            ).unsqueeze(0)
            t_out = target(
                catchup_in,
                past_key_values=target_cache,
                position_ids=pos_ids,
                output_hidden_states=True,
                use_cache=True,
            )
            target_cache = t_out.past_key_values  # now covers len(prefix)
            new_h = t_out.hidden_states[-2]  # [1, n_to_forward, H]
            assert prefix_h_buffer is not None
            prefix_h_buffer = torch.cat([prefix_h_buffer, new_h], dim=1)
            last_logits = t_out.logits[0, -1].clone()
            target_prefill_us = (time.perf_counter_ns() - t0) / 1000.0

        # Now: target_cache covers full prefix (P tokens), prefix_h_buffer length = P,
        # last_logits = distribution at position P (= the slot we will fill next).

        # ---- Build the chain-with-tail-branch ----
        t_draft_0 = time.perf_counter_ns()
        chain_tokens: list[int] = []
        # Depth 0: teacher's argmax (always accepted)
        assert last_logits is not None
        chain_tokens.append(_argmax_int(last_logits))

        # Depths 1..k-2: linear EAGLE chain (top-1 each step)
        assert prefix_h_buffer is not None
        eagle_input = prefix_h_buffer  # [1, P, H]
        for i in range(1, k - 1):
            seq_len_now = eagle_input.shape[1]
            pos_ids = torch.arange(
                seq_len_now, device=device, dtype=torch.long
            ).unsqueeze(0)
            eagle_pred_full = eagle_head(eagle_input, position_ids=pos_ids)
            new_h = eagle_pred_full[:, -1:, :]
            new_h_post = target_norm(new_h)
            logits = target.lm_head(new_h_post)
            chain_tokens.append(_argmax_int(logits[0, -1]))
            eagle_input = torch.cat([eagle_input, new_h], dim=1)

        # Depth k-1 (leaf): top-`width` candidates from a single EAGLE forward
        seq_len_now = eagle_input.shape[1]
        pos_ids = torch.arange(
            seq_len_now, device=device, dtype=torch.long
        ).unsqueeze(0)
        eagle_pred_full = eagle_head(eagle_input, position_ids=pos_ids)
        leaf_h = eagle_pred_full[:, -1:, :]
        leaf_h_post = target_norm(leaf_h)
        leaf_logits = target.lm_head(leaf_h_post)[0, -1]  # [V]
        topW = leaf_logits.topk(width)
        leaf_tokens: list[int] = topW.indices.tolist()
        t_draft_1 = time.perf_counter_ns()
        draft_us = (t_draft_1 - t_draft_0) / 1000.0

        # ---- Tree verify with custom attention mask ----
        # Tree node order:
        #   indices 0..k-2:           chain_tokens[0..k-2] at positions P..P+k-2
        #   indices k-1..k-1+width-1: leaf_tokens[0..width-1] all at position P+k-1
        tree_tokens = chain_tokens + leaf_tokens
        n_tree = len(tree_tokens)
        tree_input = torch.tensor([tree_tokens], device=device, dtype=torch.long)

        tree_positions = torch.empty(n_tree, dtype=torch.long, device=device)
        for i in range(k - 1):
            tree_positions[i] = P + i
        for j in range(width):
            tree_positions[k - 1 + j] = P + k - 1
        tree_positions = tree_positions.unsqueeze(0)  # [1, n_tree]

        # Attention mask: q × kv, additive, -inf for masked.
        # KV side has length P + n_tree (prefix + new tokens).
        # All tree nodes attend fully to the prefix (cache positions 0..P-1).
        # For new positions (cache indices P..P+n_tree-1):
        #   Chain node at index i (i in 0..k-2): visible up to and including i (causal within chain)
        #   Leaf node at index k-1+j: visible to chain (k_j < k-1) + itself (k_j == k-1+j)
        kv_len = P + n_tree
        attn_mask = torch.zeros(
            (1, 1, n_tree, kv_len), dtype=target.dtype, device=device
        )
        neg_inf = torch.finfo(target.dtype).min
        for q_i in range(n_tree):
            for k_j in range(n_tree):
                if q_i < k - 1:
                    visible = k_j <= q_i
                else:
                    j_q = q_i - (k - 1)
                    visible = (k_j < k - 1) or (k_j == k - 1 + j_q)
                if not visible:
                    attn_mask[0, 0, q_i, P + k_j] = neg_inf

        t_verify_0 = time.perf_counter_ns()
        t_out_v = target(
            tree_input,
            past_key_values=target_cache,
            attention_mask=attn_mask,
            position_ids=tree_positions,
            output_hidden_states=True,
            use_cache=True,
        )
        t_verify_1 = time.perf_counter_ns()
        verify_us = (t_verify_1 - t_verify_0) / 1000.0
        target_cache = t_out_v.past_key_values  # cache now covers P + n_tree

        v_logits = t_out_v.logits  # [1, n_tree, V]

        # ---- Greedy verify along the linear chain first ----
        # chain_tokens[0] is teacher's argmax — always accepted by target by construction.
        # For chain_tokens[i] (i >= 1): target's argmax at position P+i comes from
        #   v_logits[0, i-1] (= distribution after consuming chain_tokens[0..i-1]).
        accepted_tokens: list[int] = [chain_tokens[0]]
        n_accepted_drafts = 1
        rejected_in_chain = False
        for i in range(1, k - 1):
            target_pred = int(v_logits[0, i - 1].argmax(dim=-1).item())
            if chain_tokens[i] == target_pred:
                accepted_tokens.append(chain_tokens[i])
                n_accepted_drafts += 1
            else:
                accepted_tokens.append(target_pred)
                rejected_in_chain = True
                break

        leaf_chosen_idx: int | None = None
        if not rejected_in_chain:
            # Linear chain succeeded. Check leaves.
            # Target's argmax at position P+k-1 = argmax(v_logits[0, k-2]).
            leaf_target_pred = int(v_logits[0, k - 2].argmax(dim=-1).item())
            for j, t in enumerate(leaf_tokens):
                if t == leaf_target_pred:
                    leaf_chosen_idx = j
                    break
            if leaf_chosen_idx is not None:
                accepted_tokens.append(leaf_tokens[leaf_chosen_idx])
                n_accepted_drafts += 1
                # Bonus = target's argmax AFTER consuming the chosen leaf
                # = v_logits[0, k - 1 + leaf_chosen_idx].
                bonus_row = k - 1 + leaf_chosen_idx
                bonus = int(v_logits[0, bonus_row].argmax(dim=-1).item())
                accepted_tokens.append(bonus)
            else:
                # All leaves miss; emit target's correction at position P+k-1.
                accepted_tokens.append(leaf_target_pred)

        rejected_overall = rejected_in_chain or (
            (not rejected_in_chain) and leaf_chosen_idx is None
        )

        # ---- EOS truncation + budget cap ----
        eos_truncated = list(accepted_tokens)
        for i_tk, tk in enumerate(eos_truncated):
            if tk in eos_token_ids:
                eos_truncated = eos_truncated[: i_tk + 1]
                break
        budget = (len(prompt_ids_list) + max_new_tokens) - len(prefix)
        if budget < len(eos_truncated):
            accepted_capped = eos_truncated[:budget]
        else:
            accepted_capped = eos_truncated
        prefix.extend(accepted_capped)
        n_emitted = len(accepted_capped)

        # ---- Update prefix_h_buffer + crop target_cache ----
        # Cases:
        #   (a) chain rejected at depth i (i in 1..k-2): n_chain_accepted = i,
        #       crop cache to P + i, append i hidden states from verify.
        #   (b) chain succeeded, leaf_chosen_idx == 0: crop to P + k (keep leaf[0]'s KV),
        #       append k hidden states (chain k-1 + leaf at index k-1).
        #   (c) chain succeeded, leaf_chosen_idx > 0: crop to P + k - 1 (drop all leaves),
        #       append k-1 hidden states (chain only); leaf KV will be recomputed at start
        #       of next step via multi-token catchup.
        #   (d) chain succeeded, no leaf matched: crop to P + k - 1, append k-1 hidden states.
        if rejected_in_chain:
            n_chain_accepted = n_accepted_drafts  # by construction in the loop
            crop_dynamic_cache(target_cache, P + n_chain_accepted)
            if n_chain_accepted > 0:
                accepted_h = t_out_v.hidden_states[-2][:, :n_chain_accepted, :].contiguous()
                assert prefix_h_buffer is not None
                prefix_h_buffer = torch.cat([prefix_h_buffer, accepted_h], dim=1)
        elif leaf_chosen_idx == 0:
            crop_dynamic_cache(target_cache, P + k)
            # k - 1 chain positions + 1 leaf position (at tree index k-1)
            chain_h = t_out_v.hidden_states[-2][:, : k - 1, :]
            leaf_h_at_k_minus_1 = t_out_v.hidden_states[-2][:, k - 1 : k, :]
            accepted_h = torch.cat([chain_h, leaf_h_at_k_minus_1], dim=1).contiguous()
            assert prefix_h_buffer is not None
            prefix_h_buffer = torch.cat([prefix_h_buffer, accepted_h], dim=1)
        else:  # leaf_chosen_idx > 0 OR all leaves missed
            crop_dynamic_cache(target_cache, P + k - 1)
            if k > 1:
                accepted_h = t_out_v.hidden_states[-2][:, : k - 1, :].contiguous()
                assert prefix_h_buffer is not None
                prefix_h_buffer = torch.cat([prefix_h_buffer, accepted_h], dim=1)

        last_logits = None  # forces multi-token catchup at start of next step

        if parse_callback is not None:
            t_pcb_0 = time.perf_counter_ns()
            parse_callback(prefix)
            t_pcb_1 = time.perf_counter_ns()
            target_prefill_us += (t_pcb_1 - t_pcb_0) / 1000.0

        t_step_end = time.perf_counter_ns()
        steps.append(StepRecord(
            method=method_name,
            step=step_idx,
            k=k - 1 + width,  # report effective tree size (number of speculative tokens)
            n_accepted_drafts=n_accepted_drafts,
            n_emitted=n_emitted,
            rejected=rejected_overall,
            node_type=None,
            deepest_type=None,
            wall_us=(t_step_end - t_step_start) / 1000.0,
            draft_us=draft_us,
            verify_us=verify_us,
            parse_us=target_prefill_us,
        ))
        step_idx += 1

        if any(t in eos_token_ids for t in accepted_capped):
            break

    t_end = time.perf_counter_ns()
    return DecodeResult(
        output_token_ids=prefix,
        output_text="",
        n_new_tokens=len(prefix) - len(prompt_ids_list),
        steps=steps,
        wall_us_total=(t_end - t_start) / 1000.0,
    )


def tree_tail_eagle_spec(
    prompt_ids: torch.Tensor,
    target,
    eagle_head,
    max_new_tokens: int,
    eos_token_ids: list[int],
    k: int = 2,
    width: int = 2,
) -> DecodeResult:
    """Convenience wrapper for fixed (k, width) tree-tail spec."""
    return tree_eagle_speculative_ar(
        prompt_ids=prompt_ids,
        target=target,
        eagle_head=eagle_head,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        method_name=f"tree_eagle_k{k}w{width}",
        k=k,
        width=width,
    )
