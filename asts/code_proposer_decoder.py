"""Verified speculative decoding with cheap code-specific proposers.

The decoder tries CPU-only code proposers before paying for an EAGLE draft.
Every candidate is still verified by the target model with the greedy
rejection rule, so the fp32/eager lossless invariant is unchanged.
"""

from __future__ import annotations

import time

import torch

from .ast_policy import ASTPolicy, CursorContext
from .code_proposers import (
    CheapProposerStack,
    Proposal,
    ProposalFeedback,
    ProposalTree,
    ProposerState,
    build_candidate_prefix_tree,
)
from .decoder import DecodeResult, StepRecord, crop_dynamic_cache


def _argmax_int(logits_row: torch.Tensor) -> int:
    return int(logits_row.argmax(dim=-1).item())


def _draft_confidence(logits: torch.Tensor) -> tuple[float, float]:
    logits_f = logits.float()
    top2 = logits_f.topk(2)
    log_z = torch.logsumexp(logits_f, dim=-1)
    p1 = float(torch.exp(top2.values[0] - log_z).item())
    p2 = float(torch.exp(top2.values[1] - log_z).item())
    return p1, p1 - p2


def _extend_with_budget(
    prefix: list[int],
    accepted_tokens: list[int],
    eos_token_ids: list[int],
    prompt_len: int,
    max_new_tokens: int,
) -> tuple[int, list[int]]:
    eos_truncated = list(accepted_tokens)
    for i, tk in enumerate(eos_truncated):
        if tk in eos_token_ids:
            eos_truncated = eos_truncated[: i + 1]
            break
    budget = (prompt_len + max_new_tokens) - len(prefix)
    accepted_capped = eos_truncated[:budget]
    prefix.extend(accepted_capped)
    return len(accepted_capped), accepted_capped


def _context_tail_width(
    ctx: CursorContext,
    second_logits: torch.Tensor,
    widths: dict[str, int],
    default_width: int,
) -> tuple[int, float | None, float | None]:
    width = int(widths.get("default", default_width))
    all_types = {ctx.node_type, ctx.deepest_type, *ctx.ancestor_types}
    if all_types & {
        "identifier",
        "property_identifier",
        "shorthand_property_identifier",
        "attribute",
        "member_expression",
    }:
        width = max(width, int(widths.get("identifier", width)))
    if all_types & {
        "string",
        "string_content",
        "integer",
        "float",
        "number",
        "list",
        "tuple",
        "dictionary",
        "pair",
        "argument_list",
        "arguments",
        "assert_statement",
    }:
        width = max(width, int(widths.get("literal", width)))

    p1, margin = _draft_confidence(second_logits)
    margin_threshold = float(widths.get("margin_threshold", 0.08))
    if margin <= margin_threshold:
        width = max(width, int(widths.get("margin", width)))
    return max(1, width), p1, margin


def _ancestor_flat_indices(nodes, flat_idx: int) -> set[int]:
    ancestors = {flat_idx}
    node_idx = flat_idx - 1
    while node_idx >= 0:
        parent = nodes[node_idx].parent
        if parent < 0:
            ancestors.add(0)
            break
        flat_parent = parent + 1
        ancestors.add(flat_parent)
        node_idx = parent
    ancestors.add(0)
    return ancestors


@torch.no_grad()
def code_proposer_speculative_ar(
    prompt_ids: torch.Tensor,
    target,
    eagle_head,
    tokenizer,
    ast_policy: ASTPolicy,
    max_new_tokens: int,
    eos_token_ids: list[int],
    *,
    proposer_stack: CheapProposerStack | None,
    fallback: str = "eagle_k2",
    tail_width: int = 2,
    context_tail_widths: dict[str, int] | None = None,
    language: str = "python",
    method_name: str = "vantage_code_stack",
    reference: str = "",
    metadata: dict | None = None,
    prompt_text: str = "",
) -> DecodeResult:
    """Run a cheap-proposer decode with EAGLE or tail fallback.

    ``fallback`` is ``"root"``, ``"eagle_k2"``, ``"tail"``, or
    ``"tail_context"``.  Root fallback emits only the cached target argmax
    when no cheap proposal fires; this isolates proposer-only throughput
    without paying for a neural fallback.
    The first target argmax is always the root draft and is accepted by
    construction; cheap proposers only supply continuation tokens after it.
    """
    if fallback not in {"root", "eagle_k2", "tail", "tail_context"}:
        raise ValueError(f"unsupported code proposer fallback: {fallback}")
    if tail_width < 1:
        raise ValueError(f"tail_width must be >= 1; got {tail_width}")

    device = next(target.parameters()).device
    prompt_ids_list = prompt_ids.tolist() if prompt_ids.dim() == 1 else prompt_ids[0].tolist()
    prefix: list[int] = list(prompt_ids_list)
    prompt_len = len(prompt_ids_list)
    target_norm = target.model.norm

    target_cache = None
    prefix_h_buffer: torch.Tensor | None = None
    last_logits: torch.Tensor | None = None

    steps: list[StepRecord] = []
    t_start = time.perf_counter_ns()
    step_idx = 0
    if proposer_stack is not None and hasattr(proposer_stack, "reset"):
        proposer_stack.reset()
    token_only_proposers = (
        proposer_stack is not None
        and hasattr(proposer_stack, "requires_text_context")
        and not proposer_stack.requires_text_context()
    )
    prepare_us_pending = 0.0
    if proposer_stack is not None and hasattr(proposer_stack, "prepare"):
        t_prepare_0 = time.perf_counter_ns()
        proposer_stack.prepare(
            prompt_ids=prompt_ids_list,
            prompt_text=prompt_text,
            reference=reference,
            metadata=metadata,
            prompt_len=prompt_len,
            language=language,
        )
        prepare_us_pending = (time.perf_counter_ns() - t_prepare_0) / 1000.0

    while len(prefix) < prompt_len + max_new_tokens:
        t_step_start = time.perf_counter_ns()
        P = len(prefix)

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
                    f"cache_len {current_cache_len} >= prefix_len {len(prefix)}"
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

        assert last_logits is not None
        assert prefix_h_buffer is not None
        teacher_argmax = _argmax_int(last_logits)

        if token_only_proposers:
            text_before = ""
            text_after = ""
            ctx = CursorContext(
                node_type="token",
                deepest_type="token",
                k=0,
                ancestor_types=("token",),
                parser_in_error=False,
            )
            parse_us = 0.0
        else:
            t_parse_0 = time.perf_counter_ns()
            text_before = tokenizer.decode(prefix, skip_special_tokens=False)
            text_after = tokenizer.decode(prefix + [teacher_argmax], skip_special_tokens=False)
            ast_policy.update(text_after.encode("utf-8"))
            ctx = ast_policy.context_at_cursor()
            t_parse_1 = time.perf_counter_ns()
            parse_us = (t_parse_1 - t_parse_0) / 1000.0

        t_prop_0 = time.perf_counter_ns()
        proposal: Proposal | ProposalTree | None = None
        if proposer_stack is not None:
            proposal = proposer_stack.propose(
                ProposerState(
                    prefix=prefix,
                    teacher_argmax=teacher_argmax,
                    text_before=text_before,
                    text_after=text_after,
                    ctx=ctx,
                    language=language,
                    prompt_len=prompt_len,
                    reference=reference,
                    metadata=metadata,
                    prompt_text=prompt_text,
                )
            )
        proposal_us = (time.perf_counter_ns() - t_prop_0) / 1000.0

        draft_us = 0.0
        verify_us = 0.0
        draft_top1_prob: float | None = None
        draft_top2_margin: float | None = None
        proposal_tree_nodes: int | None = None
        proposal_tree_candidates: int | None = None
        proposal_tree_branch_depth: int | None = None
        proposal_tree_branch_selected: int | None = None
        proposal_first_token: int | None = None
        proposal_first_token_text: str | None = None
        proposal_target_reject_token: int | None = None
        proposal_target_reject_token_text: str | None = None
        proposal_target_reject_index: int | None = None
        strategy: str

        if isinstance(proposal, ProposalTree) and proposal.candidates:
            strategy = f"proposal_tree_{proposal.kind}"
            nodes = build_candidate_prefix_tree(proposal.candidates, proposal.max_nodes)
            if not nodes:
                proposal = None
            else:
                tree_tokens = [teacher_argmax] + [node.token for node in nodes]
                n_tree = len(tree_tokens)
                proposal_tree_nodes = len(nodes)
                proposal_tree_candidates = len(proposal.candidates)
                tree_input = torch.tensor([tree_tokens], device=device, dtype=torch.long)
                tree_positions = torch.empty(n_tree, dtype=torch.long, device=device)
                tree_positions[0] = P
                for flat_idx in range(1, n_tree):
                    tree_positions[flat_idx] = P + nodes[flat_idx - 1].depth
                tree_positions = tree_positions.unsqueeze(0)

                kv_len = P + n_tree
                attn_mask = torch.zeros(
                    (1, 1, n_tree, kv_len), dtype=target.dtype, device=device
                )
                neg_inf = torch.finfo(target.dtype).min
                for q_i in range(n_tree):
                    visible_new = {0} if q_i == 0 else _ancestor_flat_indices(nodes, q_i)
                    for k_j in range(n_tree):
                        if k_j not in visible_new:
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
                target_cache = t_out_v.past_key_values
                verify_us = (time.perf_counter_ns() - t_verify_0) / 1000.0
                v_logits = t_out_v.logits

                children: dict[int, list[int]] = {}
                for idx, node in enumerate(nodes, start=1):
                    parent_flat = 0 if node.parent < 0 else node.parent + 1
                    children.setdefault(parent_flat, []).append(idx)

                accepted_tokens = [teacher_argmax]
                n_accepted_drafts = 1
                current_flat = 0
                rejected = False
                while children.get(current_flat):
                    target_pred = _argmax_int(v_logits[0, current_flat])
                    match_flat: int | None = None
                    for child_flat in children[current_flat]:
                        if tree_tokens[child_flat] == target_pred:
                            match_flat = child_flat
                            break
                    if match_flat is None:
                        accepted_tokens.append(target_pred)
                        rejected = True
                        break
                    accepted_tokens.append(target_pred)
                    n_accepted_drafts += 1
                    current_flat = match_flat
                    proposal_tree_branch_depth = nodes[current_flat - 1].depth
                    proposal_tree_branch_selected = current_flat
                if not rejected:
                    accepted_tokens.append(_argmax_int(v_logits[0, current_flat]))

                n_emitted, accepted_capped = _extend_with_budget(
                    prefix, accepted_tokens, eos_token_ids, prompt_len, max_new_tokens
                )
                # We only keep the guaranteed root in the KV cache for
                # proposal trees. If a deeper branch was accepted, the next
                # loop's catchup recomputes that accepted spine with ordinary
                # causal attention. This is slower than spine-specific cache
                # grafting, but keeps branch correctness simple.
                accepted_in_prefix = min(1, n_emitted)
                crop_dynamic_cache(target_cache, P + accepted_in_prefix)
                if accepted_in_prefix > 0:
                    root_h = t_out_v.hidden_states[-2][:, :1, :].contiguous()
                    prefix_h_buffer = torch.cat([prefix_h_buffer, root_h], dim=1)
                k_report = n_tree

        elif isinstance(proposal, Proposal) and proposal.tokens:
            strategy = f"proposal_{proposal.kind}"
            drafts = [teacher_argmax] + proposal.tokens
            proposal_first_token = int(proposal.tokens[0])
            if not token_only_proposers:
                proposal_first_token_text = tokenizer.decode(
                    [proposal_first_token], skip_special_tokens=False
                )
            t_verify_0 = time.perf_counter_ns()
            t_out_v = target(
                torch.tensor([drafts], device=device, dtype=torch.long),
                past_key_values=target_cache,
                output_hidden_states=True,
                use_cache=True,
            )
            target_cache = t_out_v.past_key_values
            verify_us = (time.perf_counter_ns() - t_verify_0) / 1000.0
            v_logits = t_out_v.logits

            accepted_tokens = [drafts[0]]
            n_accepted_drafts = 1
            rejected = False
            for i in range(1, len(drafts)):
                target_pred = _argmax_int(v_logits[0, i - 1])
                if drafts[i] == target_pred:
                    accepted_tokens.append(drafts[i])
                    n_accepted_drafts += 1
                else:
                    proposal_target_reject_token = int(target_pred)
                    if not token_only_proposers:
                        proposal_target_reject_token_text = tokenizer.decode(
                            [proposal_target_reject_token], skip_special_tokens=False
                        )
                    proposal_target_reject_index = i - 1
                    accepted_tokens.append(target_pred)
                    rejected = True
                    break
            if not rejected:
                accepted_tokens.append(_argmax_int(v_logits[0, len(drafts) - 1]))

            n_emitted, accepted_capped = _extend_with_budget(
                prefix, accepted_tokens, eos_token_ids, prompt_len, max_new_tokens
            )
            accepted_in_prefix = min(n_accepted_drafts, n_emitted)
            crop_dynamic_cache(target_cache, P + accepted_in_prefix)
            if accepted_in_prefix > 0:
                accepted_h = t_out_v.hidden_states[-2][:, :accepted_in_prefix, :].contiguous()
                prefix_h_buffer = torch.cat([prefix_h_buffer, accepted_h], dim=1)
            k_report = len(drafts)

        else:
            proposal = None
            if fallback == "root":
                strategy = "fallback_root"
                accepted_tokens = [teacher_argmax]
                n_accepted_drafts = 1
                rejected = False
                n_emitted, accepted_capped = _extend_with_budget(
                    prefix, accepted_tokens, eos_token_ids, prompt_len, max_new_tokens
                )
                k_report = 1
            else:
                t_draft_0 = time.perf_counter_ns()
                seq_len_now = prefix_h_buffer.shape[1]
                pos_ids = torch.arange(seq_len_now, device=device, dtype=torch.long).unsqueeze(0)
                eagle_pred_full = eagle_head(prefix_h_buffer, position_ids=pos_ids)
                second_h = eagle_pred_full[:, -1:, :]
                second_logits = target.lm_head(target_norm(second_h))[0, -1]

            if fallback == "eagle_k2":
                strategy = "fallback_eagle_k2"
                drafts = [teacher_argmax, _argmax_int(second_logits)]
                draft_us = (time.perf_counter_ns() - t_draft_0) / 1000.0

                t_verify_0 = time.perf_counter_ns()
                t_out_v = target(
                    torch.tensor([drafts], device=device, dtype=torch.long),
                    past_key_values=target_cache,
                    output_hidden_states=True,
                    use_cache=True,
                )
                target_cache = t_out_v.past_key_values
                verify_us = (time.perf_counter_ns() - t_verify_0) / 1000.0
                v_logits = t_out_v.logits

                accepted_tokens = [drafts[0]]
                n_accepted_drafts = 1
                rejected = False
                target_pred = _argmax_int(v_logits[0, 0])
                if drafts[1] == target_pred:
                    accepted_tokens.append(drafts[1])
                    n_accepted_drafts += 1
                    accepted_tokens.append(_argmax_int(v_logits[0, 1]))
                else:
                    accepted_tokens.append(target_pred)
                    rejected = True

                n_emitted, accepted_capped = _extend_with_budget(
                    prefix, accepted_tokens, eos_token_ids, prompt_len, max_new_tokens
                )
                accepted_in_prefix = min(n_accepted_drafts, n_emitted)
                crop_dynamic_cache(target_cache, P + accepted_in_prefix)
                if accepted_in_prefix > 0:
                    accepted_h = t_out_v.hidden_states[-2][:, :accepted_in_prefix, :].contiguous()
                    prefix_h_buffer = torch.cat([prefix_h_buffer, accepted_h], dim=1)
                k_report = 2

            elif fallback != "root":
                if fallback == "tail_context":
                    width, draft_top1_prob, draft_top2_margin = _context_tail_width(
                        ctx,
                        second_logits,
                        context_tail_widths or {},
                        tail_width,
                    )
                    strategy = f"fallback_tail_w{width}_context"
                else:
                    width = tail_width
                    strategy = f"fallback_tail_w{width}"
                top_w = second_logits.topk(width)
                leaf_tokens = [int(t) for t in top_w.indices.tolist()]
                draft_us = (time.perf_counter_ns() - t_draft_0) / 1000.0

                tree_tokens = [teacher_argmax] + leaf_tokens
                n_tree = len(tree_tokens)
                tree_input = torch.tensor([tree_tokens], device=device, dtype=torch.long)
                tree_positions = torch.empty(n_tree, dtype=torch.long, device=device)
                tree_positions[0] = P
                for j in range(width):
                    tree_positions[1 + j] = P + 1
                tree_positions = tree_positions.unsqueeze(0)

                kv_len = P + n_tree
                attn_mask = torch.zeros(
                    (1, 1, n_tree, kv_len), dtype=target.dtype, device=device
                )
                neg_inf = torch.finfo(target.dtype).min
                for q_i in range(n_tree):
                    for k_j in range(n_tree):
                        if q_i == 0:
                            visible = k_j == 0
                        else:
                            visible = k_j == 0 or k_j == q_i
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
                target_cache = t_out_v.past_key_values
                verify_us = (time.perf_counter_ns() - t_verify_0) / 1000.0
                v_logits = t_out_v.logits

                accepted_tokens = [teacher_argmax]
                n_accepted_drafts = 1
                rejected = False
                leaf_target_pred = _argmax_int(v_logits[0, 0])
                leaf_chosen_idx: int | None = None
                for j, token in enumerate(leaf_tokens):
                    if token == leaf_target_pred:
                        leaf_chosen_idx = j
                        break
                if leaf_chosen_idx is not None:
                    accepted_tokens.append(leaf_tokens[leaf_chosen_idx])
                    n_accepted_drafts += 1
                    accepted_tokens.append(_argmax_int(v_logits[0, 1 + leaf_chosen_idx]))
                else:
                    accepted_tokens.append(leaf_target_pred)
                    rejected = True

                n_emitted, accepted_capped = _extend_with_budget(
                    prefix, accepted_tokens, eos_token_ids, prompt_len, max_new_tokens
                )
                if leaf_chosen_idx == 0 and n_emitted >= 2:
                    crop_dynamic_cache(target_cache, P + 2)
                    root_h = t_out_v.hidden_states[-2][:, :1, :]
                    leaf_h = t_out_v.hidden_states[-2][:, 1:2, :]
                    prefix_h_buffer = torch.cat(
                        [prefix_h_buffer, torch.cat([root_h, leaf_h], dim=1).contiguous()],
                        dim=1,
                    )
                else:
                    crop_dynamic_cache(target_cache, P + min(1, n_emitted))
                    if n_emitted >= 1:
                        root_h = t_out_v.hidden_states[-2][:, :1, :].contiguous()
                        prefix_h_buffer = torch.cat([prefix_h_buffer, root_h], dim=1)
                k_report = 1 + width

        last_logits = None
        t_step_end = time.perf_counter_ns()
        n_accepted_nonroot = max(0, n_accepted_drafts - 1)
        hit_max_new_tokens = len(prefix) >= prompt_len + max_new_tokens and not any(
            t in eos_token_ids for t in accepted_capped
        )
        step_wall_us = (t_step_end - t_step_start) / 1000.0 + prepare_us_pending
        prepare_us_pending = 0.0
        steps.append(
            StepRecord(
                method=method_name,
                step=step_idx,
                k=k_report,
                n_accepted_drafts=n_accepted_drafts,
                n_emitted=n_emitted,
                rejected=rejected,
                node_type=ctx.node_type,
                deepest_type=ctx.deepest_type,
                wall_us=step_wall_us,
                draft_us=draft_us,
                verify_us=verify_us,
                parse_us=parse_us,
                strategy=strategy,
                draft_top1_prob=draft_top1_prob,
                draft_top2_margin=draft_top2_margin,
                ancestor_types=ctx.ancestor_types,
                parser_in_error=ctx.parser_in_error,
                target_prefill_us=target_prefill_us,
                proposal_kind=proposal.kind if proposal else None,
                proposal_match_len=proposal.match_len if proposal else None,
                proposal_us=proposal_us,
                proposal_tokens=(
                    proposal_tree_nodes
                    if isinstance(proposal, ProposalTree)
                    else len(proposal.tokens) if proposal else 0
                ),
                n_guaranteed_drafts=1,
                n_accepted_nonroot_drafts=n_accepted_nonroot,
                hit_max_new_tokens=hit_max_new_tokens,
                prompt_len=prompt_len,
                proposal_source_start_token=proposal.source_start_token if proposal else None,
                proposal_source_end_token=proposal.source_end_token if proposal else None,
                proposal_follow_start_token=proposal.follow_start_token if proposal else None,
                proposal_follow_end_token=proposal.follow_end_token if proposal else None,
                proposal_query_len=proposal.query_len if proposal else None,
                proposal_pool=proposal.pool if proposal else None,
                proposal_source_region=proposal.source_region if proposal else None,
                proposal_root_included=proposal.root_included if proposal else None,
                proposal_match_kind=proposal.match_kind if proposal else None,
                proposal_canonical_match_len=proposal.canonical_match_len if proposal else None,
                proposal_substitution_count=proposal.substitution_count if proposal else None,
                proposal_scope_fill_count=proposal.scope_fill_count if proposal else None,
                proposal_stopped_on_unmapped=proposal.stopped_on_unmapped if proposal else None,
                proposal_alpha_exact_filtered=proposal.alpha_exact_filtered if proposal else None,
                proposal_zero_nonroot_accept=(
                    n_accepted_nonroot == 0
                    if proposal and proposal.kind.startswith("alpha")
                    else proposal.zero_nonroot_accept if proposal else None
                ),
                proposal_tree_nodes=proposal_tree_nodes,
                proposal_tree_candidates=proposal_tree_candidates,
                proposal_tree_branch_depth=proposal_tree_branch_depth,
                proposal_tree_branch_selected=proposal_tree_branch_selected,
                proposal_map_source=proposal.map_source if proposal else None,
                proposal_inferred_map_count=proposal.inferred_map_count if proposal else None,
                proposal_inference_confidence=proposal.inference_confidence if proposal else None,
                proposal_cursor_pos=proposal.cursor_pos if proposal else None,
                proposal_cursor_confidence=proposal.cursor_confidence if proposal else None,
                proposal_cursor_resync=proposal.cursor_resync if proposal else None,
                proposal_view_id=proposal.view_id if proposal else None,
                proposal_compound_view_count=proposal.compound_view_count if proposal else None,
                proposal_active_map_count=proposal.active_map_count if proposal else None,
                proposal_route=proposal.route if proposal else None,
                proposal_route_reason=proposal.route_reason if proposal else None,
                proposal_backoff_active=proposal.backoff_active if proposal else None,
                proposal_rewrite_hit_count=proposal.rewrite_hit_count if proposal else None,
                proposal_route_window_accept_rate=(
                    proposal.route_window_accept_rate if proposal else None
                ),
                proposal_rewrite_zero_accept_streak=(
                    proposal.rewrite_zero_accept_streak if proposal else None
                ),
                proposal_adoption_state=proposal.adoption_state if proposal else None,
                proposal_adoption_transition=(
                    proposal.adoption_transition if proposal else None
                ),
                proposal_frontier_distance=proposal.frontier_distance if proposal else None,
                proposal_frontier_probes=proposal.frontier_probes if proposal else None,
                proposal_accepted_crossed_rewrite=(
                    proposal.accepted_crossed_rewrite if proposal else None
                ),
                proposal_rejected_old_form_frontiers=(
                    proposal.rejected_old_form_frontiers if proposal else None
                ),
                proposal_blacklisted_rewrite_occurrences=(
                    proposal.blacklisted_rewrite_occurrences if proposal else None
                ),
                proposal_disabled_by_adoption_gate=(
                    proposal.disabled_by_adoption_gate if proposal else None
                ),
                proposal_root_old_match_count=proposal.root_old_match_count if proposal else None,
                proposal_root_new_match_count=proposal.root_new_match_count if proposal else None,
                proposal_map_parse_us=proposal.map_parse_us if proposal else 0.0,
                proposal_rewrite_apply_us=proposal.rewrite_apply_us if proposal else 0.0,
                proposal_virtual_reference_tokenize_us=(
                    proposal.virtual_reference_tokenize_us if proposal else 0.0
                ),
                proposal_transpld_index_build_us=(
                    proposal.transpld_index_build_us if proposal else 0.0
                ),
                proposal_text_preview=(
                    proposal.text_preview if isinstance(proposal, Proposal) else None
                ),
                proposal_first_token=proposal_first_token,
                proposal_first_token_text=proposal_first_token_text,
                proposal_target_reject_token=proposal_target_reject_token,
                proposal_target_reject_token_text=proposal_target_reject_token_text,
                proposal_target_reject_index=proposal_target_reject_index,
            )
        )
        if proposer_stack is not None and hasattr(proposer_stack, "observe"):
            proposal_tokens = proposal.tokens if proposal else []
            proposer_stack.observe(
                ProposalFeedback(
                    prefix_start=P,
                    prefix_end=len(prefix),
                    proposed_tokens=list(proposal_tokens),
                    emitted_tokens=list(accepted_capped),
                    accepted_nonroot=n_accepted_nonroot,
                    rejected=rejected,
                    proposal_kind=proposal.kind if proposal else None,
                    proposal_match_kind=proposal.match_kind if proposal else None,
                    source_start_token=proposal.source_start_token if proposal else None,
                    source_end_token=proposal.source_end_token if proposal else None,
                    follow_start_token=proposal.follow_start_token if proposal else None,
                    follow_end_token=proposal.follow_end_token if proposal else None,
                )
            )
        step_idx += 1

        if any(t in eos_token_ids for t in accepted_capped):
            break

    t_end = time.perf_counter_ns()
    return DecodeResult(
        output_token_ids=prefix,
        output_text="",
        n_new_tokens=len(prefix) - prompt_len,
        steps=steps,
        wall_us_total=(t_end - t_start) / 1000.0,
    )


def code_proposer_spec(
    prompt_ids: torch.Tensor,
    target,
    eagle_head,
    tokenizer,
    ast_policy: ASTPolicy,
    max_new_tokens: int,
    eos_token_ids: list[int],
    proposer_stack: CheapProposerStack | None,
    fallback: str = "eagle_k2",
    tail_width: int = 2,
    context_tail_widths: dict[str, int] | None = None,
    language: str = "python",
    method_name: str = "vantage_code_stack",
    reference: str = "",
    metadata: dict | None = None,
    prompt_text: str = "",
) -> DecodeResult:
    return code_proposer_speculative_ar(
        prompt_ids=prompt_ids,
        target=target,
        eagle_head=eagle_head,
        tokenizer=tokenizer,
        ast_policy=ast_policy,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        proposer_stack=proposer_stack,
        fallback=fallback,
        tail_width=tail_width,
        context_tail_widths=context_tail_widths,
        language=language,
        method_name=method_name,
        reference=reference,
        metadata=metadata,
        prompt_text=prompt_text,
    )
