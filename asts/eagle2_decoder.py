"""EAGLE-2-style tree speculative decoder built on top of our EAGLE-1 head.

What this implements
--------------------

The EAGLE-2 algorithm (Li et al., 2024) generalises EAGLE-1's chain draft
into a token tree, verified in one batched target forward with a tree-attention
mask, accepting the longest greedy-equal root-to-node path.

What our EAGLE-1 head can and cannot do
---------------------------------------

EAGLE-2 in the published paper assumes a draft head that conditions on BOTH
the previous hidden state AND the previously-chosen token's embedding, so that
sibling branches at intermediate depths produce genuinely different children.

Our trained EAGLE-1 head (asts/eagle.py) conditions on hidden states ONLY.
A literal beam-search through this head therefore produces degenerate
intermediate-depth siblings (children only depend on the path of EAGLE-predicted
hidden states, not on which sibling token was chosen at the previous depth).

We therefore implement a simplified tree-builder that exploits the degeneracy:
at each depth d we expand the highest-scoring depth-(d-1) node, take the EAGLE
head's top-K logits, and attach all K children to that single best parent. The
resulting tree is "best-path chain with K branches at each depth", bounded by
total_tokens.

This is the EAGLE-2 verify/accept algorithm running over a tree shape that is
the best our hidden-state-only head can produce. A fully token-conditioned
EAGLE-2 head trained for Qwen2.5-Coder-7B is left to follow-up work.

Cache management
----------------

After verify, the target's KV cache contains slots for every tree position in
flat order; the accepted path's slots are at scattered flat indices and their
position_ids do not align with absolute token positions. Rather than reshuffle
cache K/V tensors (a correctness landmine), we crop the cache to P and rely on
a multi-token catchup forward at the start of the next outer step. This matches
the "leaf > 0" path in tree_eagle_decoder.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch

from .decoder import (
    DecodeResult,
    StepRecord,
    crop_dynamic_cache,
)


@dataclass
class TreeNode:
    parent_idx: int  # -1 for root
    depth: int
    token_id: int
    score: float  # cumulative log-prob from root


def _argmax_int(logits_row: torch.Tensor) -> int:
    return int(logits_row.argmax(dim=-1).item())


# ---------------------------------------------------------------------------
# Tree builder
# ---------------------------------------------------------------------------


@torch.no_grad()
def build_eagle2_tree(
    prefix_h: torch.Tensor,
    last_logits: torch.Tensor,
    eagle_head,
    target_norm,
    target_lm_head,
    total_tokens: int,
    topk_per_node: int,
    max_depth: int,
    device: torch.device,
) -> list[TreeNode]:
    """Build a draft tree using the EAGLE-1 head.

    Args:
        prefix_h: [1, P, H] target's penultimate hidden states for the prefix.
        last_logits: [V] target's logits at position P-1 (predicting position P).
        eagle_head: trained EagleHead instance.
        target_norm, target_lm_head: target's `model.norm` and `lm_head`.
        total_tokens: max nodes in the tree (root included).
        topk_per_node: branching factor at each depth.
        max_depth: max depth (root = depth 0).

    Returns:
        list[TreeNode] in BFS order: tree[0] is the root; tree[i] for i > 0
        has parent_idx pointing to an earlier index.
    """
    if total_tokens < 1 or max_depth < 0:
        raise ValueError(f"total_tokens={total_tokens}, max_depth={max_depth}")

    tree: list[TreeNode] = []
    root_token = _argmax_int(last_logits)
    tree.append(TreeNode(parent_idx=-1, depth=0, token_id=root_token, score=0.0))

    if total_tokens == 1 or max_depth == 0:
        return tree

    eagle_input = prefix_h  # [1, P, H]
    best_parent_at_depth: dict[int, int] = {0: 0}

    for d in range(1, max_depth + 1):
        if len(tree) >= total_tokens:
            break

        seq_len_now = eagle_input.shape[1]
        pos_ids = torch.arange(seq_len_now, device=device, dtype=torch.long).unsqueeze(0)
        eagle_pred = eagle_head(eagle_input, position_ids=pos_ids)
        new_h = eagle_pred[:, -1:, :]  # [1, 1, H]
        new_h_post = target_norm(new_h)
        logits = target_lm_head(new_h_post)[0, -1]  # [V]
        log_probs = torch.log_softmax(logits, dim=-1)

        K = min(topk_per_node, total_tokens - len(tree))
        topK = log_probs.topk(K)

        parent_idx = best_parent_at_depth[d - 1]
        parent_score = tree[parent_idx].score

        best_child_idx = None
        best_child_score = float("-inf")

        for j in range(K):
            child_token = int(topK.indices[j].item())
            child_log_prob = float(topK.values[j].item())
            child_score = parent_score + child_log_prob
            tree.append(
                TreeNode(
                    parent_idx=parent_idx,
                    depth=d,
                    token_id=child_token,
                    score=child_score,
                )
            )
            child_idx_in_tree = len(tree) - 1
            if child_score > best_child_score:
                best_child_score = child_score
                best_child_idx = child_idx_in_tree
            if len(tree) >= total_tokens:
                break

        best_parent_at_depth[d] = best_child_idx if best_child_idx is not None else parent_idx
        eagle_input = torch.cat([eagle_input, new_h], dim=1)

    return tree


# ---------------------------------------------------------------------------
# Tree mask + position ids
# ---------------------------------------------------------------------------


def _ancestors_including_self(tree: list[TreeNode], node_idx: int) -> list[int]:
    out: list[int] = []
    cur = node_idx
    while cur >= 0:
        out.append(cur)
        cur = tree[cur].parent_idx
    return out


def build_tree_mask(
    tree: list[TreeNode],
    P: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Construct a 4D additive attention mask [1, 1, N, P+N].

    Each tree node q attends to:
      - all prefix positions 0..P-1 (mask = 0)
      - tree positions corresponding to ancestors of q (including q itself)
      - everything else: -inf

    The KV side ordering for tree positions is the flat-tree order, so the
    cache slot for tree node i sits at absolute index P + i.
    """
    N = len(tree)
    kv_len = P + N
    neg_inf = torch.finfo(dtype).min
    mask = torch.full((1, 1, N, kv_len), neg_inf, dtype=dtype, device=device)
    mask[:, :, :, :P] = 0
    for q in range(N):
        for a in _ancestors_including_self(tree, q):
            mask[:, :, q, P + a] = 0
    return mask


def build_tree_positions(
    tree: list[TreeNode],
    P: int,
    device: torch.device,
) -> torch.Tensor:
    """[1, N] absolute position ids for the tree (P + depth per node)."""
    return torch.tensor([[P + n.depth for n in tree]], device=device, dtype=torch.long)


# ---------------------------------------------------------------------------
# Longest accepted path
# ---------------------------------------------------------------------------


def find_longest_accepted_path(
    tree: list[TreeNode],
    v_logits: torch.Tensor,
) -> tuple[list[int], int, int]:
    """Find the longest root-to-node path that is greedy-accepted by target.

    Args:
        tree: list[TreeNode] in BFS order with tree[0] = root.
        v_logits: [1, N, V] target logits from the verify forward.

    Returns:
        (path_indices, accepted_depth, bonus_token):
          - path_indices: flat-tree indices from root to deepest accepted node.
          - accepted_depth: depth of deepest accepted node (0 = only root accepted).
          - bonus_token: target's argmax at the position past the deepest accepted
            node, i.e., argmax(v_logits[0, deepest_accepted_idx]).
    """
    N = len(tree)
    accepted = [False] * N
    accepted[0] = True  # root is target's argmax at P by construction

    # Process in BFS order (which is the array order by construction)
    for i in range(1, N):
        node = tree[i]
        p_idx = node.parent_idx
        if p_idx < 0 or not accepted[p_idx]:
            continue
        target_pred = _argmax_int(v_logits[0, p_idx])
        if node.token_id == target_pred:
            accepted[i] = True

    # Pick the accepted node with greatest depth (ties broken by index = highest score)
    best_idx = 0
    best_depth = 0
    for i in range(N):
        if accepted[i] and tree[i].depth >= best_depth:
            best_depth = tree[i].depth
            best_idx = i

    # Reconstruct path from best_idx to root
    path_rev: list[int] = []
    cur = best_idx
    while cur >= 0:
        path_rev.append(cur)
        cur = tree[cur].parent_idx
    path = list(reversed(path_rev))

    bonus_token = _argmax_int(v_logits[0, best_idx])
    return path, best_depth, bonus_token


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def eagle2_speculative_ar(
    prompt_ids: torch.Tensor,
    target,
    eagle_head,
    max_new_tokens: int,
    eos_token_ids: list[int],
    method_name: str = "eagle2",
    total_tokens: int = 26,
    topk_per_node: int = 10,
    max_depth: int = 6,
    parse_callback: Callable[[list[int]], None] | None = None,
) -> DecodeResult:
    """EAGLE-2-style tree spec decode with our EAGLE-1 head.

    See module docstring for the architectural caveat about hidden-state-only
    conditioning. The greedy-rejection invariant is preserved: under fp32 +
    eager attention, output is bit-identical to vanilla AR.
    """
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

        # ---- Catch up target cache + prefix_h_buffer to current prefix ----
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
            target_cache = t_out.past_key_values
            new_h = t_out.hidden_states[-2]
            assert prefix_h_buffer is not None
            prefix_h_buffer = torch.cat([prefix_h_buffer, new_h], dim=1)
            last_logits = t_out.logits[0, -1].clone()
            target_prefill_us = (time.perf_counter_ns() - t0) / 1000.0

        assert prefix_h_buffer is not None
        assert last_logits is not None

        # ---- Build draft tree ----
        t_draft_0 = time.perf_counter_ns()
        tree = build_eagle2_tree(
            prefix_h=prefix_h_buffer,
            last_logits=last_logits,
            eagle_head=eagle_head,
            target_norm=target_norm,
            target_lm_head=target.lm_head,
            total_tokens=total_tokens,
            topk_per_node=topk_per_node,
            max_depth=max_depth,
            device=device,
        )
        N = len(tree)
        t_draft_1 = time.perf_counter_ns()
        draft_us = (t_draft_1 - t_draft_0) / 1000.0

        # ---- Tree verify ----
        tree_token_ids = [n.token_id for n in tree]
        tree_input = torch.tensor([tree_token_ids], device=device, dtype=torch.long)
        tree_positions = build_tree_positions(tree, P, device)
        attn_mask = build_tree_mask(tree, P, target.dtype, device)

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
        target_cache = t_out_v.past_key_values  # extended by N
        v_logits = t_out_v.logits  # [1, N, V]

        # ---- Find longest accepted path ----
        path, accepted_depth, bonus_token = find_longest_accepted_path(tree, v_logits)
        accepted_tokens: list[int] = [tree[i].token_id for i in path] + [bonus_token]
        # EAGLE-2 diagnostics record accepted non-root children. The root is
        # the target's cached argmax and is accepted by construction; excluding
        # it keeps the tree-path depth metric explicit. This differs from the
        # EAGLE-1 chain/tree-tail aggregate convention, which includes the
        # guaranteed first candidate in n_accepted_drafts.
        n_accepted_drafts = max(0, len(path) - 1)

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

        # ---- Cache management: crop to P, force multi-token catchup next step ----
        # Rationale: tree positions in cache are in flat order with non-monotonic
        # position_ids; using them at the next step's prefix length P+L would
        # require reshuffling, which is a correctness landmine. Catchup is one
        # extra forward over L<=max_depth+1 tokens — same cost pattern as
        # tree_eagle_decoder's "leaf > 0" path.
        crop_dynamic_cache(target_cache, P)
        # prefix_h_buffer is still at length P; next step's catchup will extend it.
        last_logits = None  # forces multi-token catchup at next step

        # rejected = path didn't reach the deepest leaf in the tree
        max_depth_in_tree = max((n.depth for n in tree), default=0)
        rejected = accepted_depth < max_depth_in_tree

        if parse_callback is not None:
            t_pcb_0 = time.perf_counter_ns()
            parse_callback(prefix)
            t_pcb_1 = time.perf_counter_ns()
            target_prefill_us += (t_pcb_1 - t_pcb_0) / 1000.0

        t_step_end = time.perf_counter_ns()
        steps.append(
            StepRecord(
                method=method_name,
                step=step_idx,
                k=N,  # report total tree size as the "k"
                n_accepted_drafts=n_accepted_drafts,
                n_emitted=n_emitted,
                rejected=rejected,
                node_type=None,
                deepest_type=None,
                wall_us=(t_step_end - t_step_start) / 1000.0,
                draft_us=draft_us,
                verify_us=verify_us,
                parse_us=target_prefill_us,
            )
        )
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


def eagle2_default(
    prompt_ids: torch.Tensor,
    target,
    eagle_head,
    max_new_tokens: int,
    eos_token_ids: list[int],
) -> DecodeResult:
    """Convenience wrapper using the EAGLE-2 paper's defaults for code:
    total_tokens=26, topk_per_node=10, max_depth=6."""
    return eagle2_speculative_ar(
        prompt_ids=prompt_ids,
        target=target,
        eagle_head=eagle_head,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        method_name="eagle2",
        total_tokens=26,
        topk_per_node=10,
        max_depth=6,
    )
