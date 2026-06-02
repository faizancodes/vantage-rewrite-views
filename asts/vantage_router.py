"""VANTAGE-Full: live visibility routing for code speculative decoding.

The router shares the same greedy verification invariant as the existing
EAGLE and retrieval decoders.  The only adaptive choice is which candidate
set to verify at the current partial-code state:

  - ``chain_k1``: target argmax + one verify bonus/correction.
  - ``chain_k2`` / ``chain_k3``: short EAGLE chain.
  - ``tail_k2w2``: teacher argmax, then top-2 EAGLE leaf branch.
  - ``retrieve``: target argmax followed by suffix-array continuation.
  - ``scope``: target argmax followed by a local identifier completion.

The implementation is deliberately conservative and optimized for clean
measurement rather than maximal speed.  It logs the visibility score,
estimated frontier depth, zone, route, and confidence signals per step so
offline analyses can test whether routing decisions improve throughput.
"""

from __future__ import annotations

import keyword
import re
import time

import torch

from .ast_policy import ASTPolicy
from .decoder import DecodeResult, StepRecord, crop_dynamic_cache
from .vantage_policy import (
    VantageRouterConfig,
    RollingAcceptance,
    VisibilityFeatures,
    choose_strategy,
)
from .retrieval_draft import RetrievalIndex, retrieve_draft


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PY_KEYWORDS = set(keyword.kwlist)
_TS_KEYWORDS = {
    "abstract",
    "any",
    "as",
    "async",
    "await",
    "boolean",
    "break",
    "case",
    "catch",
    "class",
    "const",
    "constructor",
    "continue",
    "debugger",
    "declare",
    "default",
    "delete",
    "do",
    "else",
    "enum",
    "export",
    "extends",
    "false",
    "finally",
    "for",
    "from",
    "function",
    "get",
    "if",
    "implements",
    "import",
    "in",
    "infer",
    "instanceof",
    "interface",
    "is",
    "keyof",
    "let",
    "module",
    "namespace",
    "never",
    "new",
    "null",
    "number",
    "object",
    "of",
    "private",
    "protected",
    "public",
    "readonly",
    "return",
    "set",
    "static",
    "string",
    "super",
    "switch",
    "symbol",
    "this",
    "throw",
    "true",
    "try",
    "type",
    "typeof",
    "undefined",
    "unknown",
    "var",
    "void",
    "while",
    "with",
    "yield",
}
_ALL_KEYWORDS = _PY_KEYWORDS | _TS_KEYWORDS


def _argmax_int(logits_row: torch.Tensor) -> int:
    return int(logits_row.argmax(dim=-1).item())


def _draft_confidence(logits: torch.Tensor) -> tuple[float, float]:
    logits_f = logits.float()
    top2 = logits_f.topk(2)
    log_z = torch.logsumexp(logits_f, dim=-1)
    p1 = float(torch.exp(top2.values[0] - log_z).item())
    p2 = float(torch.exp(top2.values[1] - log_z).item())
    return p1, p1 - p2


def _tokenize_no_special(tokenizer, text: str) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(t) for t in ids]


def _local_identifier_draft(
    tokenizer,
    prefix: list[int],
    teacher_argmax: int,
    max_draft_len: int,
) -> tuple[list[int], int]:
    """Draft the remaining subtokens of a recent in-scope-looking identifier.

    This is a lightweight lexical scope proxy.  It does not build a semantic
    symbol table; it looks for identifiers already present in the current
    prefix and, when the cursor is part-way through one, proposes the shortest
    token continuation that completes the most recent matching identifier.
    """
    text_after = tokenizer.decode(prefix + [teacher_argmax], skip_special_tokens=False)
    cur_match = re.search(r"[A-Za-z_][A-Za-z0-9_]*$", text_after)
    if cur_match is None:
        return [], 0
    current = cur_match.group(0)
    if not current:
        return [], 0

    history = text_after[: cur_match.start()]
    seen: set[str] = set()
    candidates: list[str] = []
    for match in _IDENT_RE.finditer(history):
        ident = match.group(0)
        if ident in _ALL_KEYWORDS or ident in seen:
            continue
        seen.add(ident)
        if ident.startswith(current) and len(ident) > len(current):
            candidates.append(ident)

    for ident in reversed(candidates):
        continuation = ident[len(current) :]
        cont_ids = _tokenize_no_special(tokenizer, continuation)
        if not cont_ids:
            continue
        return cont_ids[:max_draft_len], len(cont_ids)
    return [], 0


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


@torch.no_grad()
def vantage_router_speculative_ar(
    prompt_ids: torch.Tensor,
    target,
    eagle_head,
    tokenizer,
    ast_policy: ASTPolicy,
    max_new_tokens: int,
    eos_token_ids: list[int],
    *,
    retrieval_index: RetrievalIndex | None = None,
    config: VantageRouterConfig | None = None,
    max_retrieval_draft_len: int = 10,
    max_scope_draft_len: int = 6,
    max_query_len: int = 16,
    min_match_len: int = 3,
    method_name: str = "vantage_full",
) -> DecodeResult:
    cfg = config or VantageRouterConfig()
    device = next(target.parameters()).device
    prompt_ids_list = prompt_ids.tolist() if prompt_ids.dim() == 1 else prompt_ids[0].tolist()
    prefix: list[int] = list(prompt_ids_list)
    prompt_len = len(prompt_ids_list)
    target_norm = target.model.norm

    target_cache = None
    prefix_h_buffer: torch.Tensor | None = None
    last_logits: torch.Tensor | None = None
    rolling = RollingAcceptance(window=cfg.rolling_window, default=cfg.rolling_default_acceptance)

    steps: list[StepRecord] = []
    t_start = time.perf_counter_ns()
    step_idx = 0

    while len(prefix) < prompt_len + max_new_tokens:
        t_step_start = time.perf_counter_ns()
        P = len(prefix)

        # ---- Live parse state ----
        t_parse_0 = time.perf_counter_ns()
        ast_policy.update(tokenizer.decode(prefix, skip_special_tokens=False).encode("utf-8"))
        ctx = ast_policy.context_at_cursor()
        t_parse_1 = time.perf_counter_ns()
        parse_us = (t_parse_1 - t_parse_0) / 1000.0

        # ---- Bring target up to date ----
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

        # ---- Scout draft/confidence/retrieval signals ----
        t_draft_0 = time.perf_counter_ns()
        assert last_logits is not None
        assert prefix_h_buffer is not None
        teacher_argmax = _argmax_int(last_logits)

        t_conf_0 = time.perf_counter_ns()
        seq_len_now = prefix_h_buffer.shape[1]
        pos_ids = torch.arange(seq_len_now, device=device, dtype=torch.long).unsqueeze(0)
        eagle_pred_full = eagle_head(prefix_h_buffer, position_ids=pos_ids)
        second_h = eagle_pred_full[:, -1:, :]
        second_logits = target.lm_head(target_norm(second_h))[0, -1]
        draft_top1_prob, draft_top2_margin = _draft_confidence(second_logits)
        confidence_us = (time.perf_counter_ns() - t_conf_0) / 1000.0

        retrieved: list[int] = []
        matched_len = 0
        retrieval_us = 0.0
        if cfg.use_retrieval and retrieval_index is not None:
            t_retrieval_0 = time.perf_counter_ns()
            retrieved, matched_len = retrieve_draft(
                retrieval_index,
                prefix + [teacher_argmax],
                max_query_len=max_query_len,
                max_draft_len=max_retrieval_draft_len,
                min_match_len=min_match_len,
            )
            retrieval_us = (time.perf_counter_ns() - t_retrieval_0) / 1000.0

        scope_us = 0.0
        if cfg.use_scope:
            t_scope_0 = time.perf_counter_ns()
            scope_draft, scope_len = _local_identifier_draft(
                tokenizer,
                prefix,
                teacher_argmax,
                max_scope_draft_len,
            )
            scope_us = (time.perf_counter_ns() - t_scope_0) / 1000.0
        else:
            scope_draft, scope_len = [], 0

        rolling_rate = rolling.rate(ctx.node_type)
        t_route_0 = time.perf_counter_ns()
        features = VisibilityFeatures(
            node_type=ctx.node_type,
            deepest_type=ctx.deepest_type,
            draft_top1_prob=draft_top1_prob,
            draft_top2_margin=draft_top2_margin,
            retrieval_match_len=matched_len,
            rolling_accept_rate=rolling_rate,
            scope_match_len=scope_len,
            parser_in_error=ctx.parser_in_error,
        )
        decision = choose_strategy(
            features,
            cfg,
            retrieval_available=cfg.use_retrieval and bool(retrieved),
            scope_available=cfg.use_scope and bool(scope_draft),
        )
        route_us = (time.perf_counter_ns() - t_route_0) / 1000.0

        accepted_tokens: list[int]
        n_accepted_drafts: int
        rejected: bool
        k_report: int

        # ---- Execute the selected candidate shape ----
        if decision.strategy in {"retrieve", "scope", "chain_k1", "chain_k2", "chain_k3"}:
            if decision.strategy == "retrieve":
                drafts = [teacher_argmax] + retrieved
            elif decision.strategy == "scope":
                drafts = [teacher_argmax] + scope_draft
            else:
                if decision.strategy == "chain_k1":
                    k_chain = 1
                elif decision.strategy == "chain_k3":
                    k_chain = 3
                else:
                    k_chain = 2
                drafts = [teacher_argmax]
                eagle_input = torch.cat([prefix_h_buffer, second_h], dim=1)
                if k_chain >= 2:
                    drafts.append(_argmax_int(second_logits))
                for _ in range(2, k_chain):
                    seq_len_now = eagle_input.shape[1]
                    pos_ids = torch.arange(
                        seq_len_now, device=device, dtype=torch.long
                    ).unsqueeze(0)
                    eagle_pred_full = eagle_head(eagle_input, position_ids=pos_ids)
                    new_h = eagle_pred_full[:, -1:, :]
                    logits = target.lm_head(target_norm(new_h))
                    drafts.append(_argmax_int(logits[0, -1]))
                    eagle_input = torch.cat([eagle_input, new_h], dim=1)

            t_draft_1 = time.perf_counter_ns()
            draft_us = (t_draft_1 - t_draft_0) / 1000.0
            k_report = len(drafts)

            verify_in = torch.tensor([drafts], device=device, dtype=torch.long)
            t_verify_0 = time.perf_counter_ns()
            t_out_v = target(
                verify_in,
                past_key_values=target_cache,
                output_hidden_states=True,
                use_cache=True,
            )
            target_cache = t_out_v.past_key_values
            t_verify_1 = time.perf_counter_ns()
            verify_us = (t_verify_1 - t_verify_0) / 1000.0
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

        elif decision.strategy == "tail_k2w2":
            width = 2
            top_w = second_logits.topk(width)
            chain_tokens = [teacher_argmax]
            leaf_tokens = [int(t) for t in top_w.indices.tolist()]
            t_draft_1 = time.perf_counter_ns()
            draft_us = (t_draft_1 - t_draft_0) / 1000.0

            tree_tokens = chain_tokens + leaf_tokens
            n_tree = len(tree_tokens)
            k_report = n_tree
            tree_input = torch.tensor([tree_tokens], device=device, dtype=torch.long)
            tree_positions = torch.tensor([[P, P + 1, P + 1]], device=device, dtype=torch.long)

            kv_len = P + n_tree
            attn_mask = torch.zeros(
                (1, 1, n_tree, kv_len), dtype=target.dtype, device=device
            )
            neg_inf = torch.finfo(target.dtype).min
            # chain root sees itself; each leaf sees root + itself, not sibling.
            attn_mask[0, 0, 0, P + 1] = neg_inf
            attn_mask[0, 0, 0, P + 2] = neg_inf
            attn_mask[0, 0, 1, P + 2] = neg_inf
            attn_mask[0, 0, 2, P + 1] = neg_inf

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
            t_verify_1 = time.perf_counter_ns()
            verify_us = (t_verify_1 - t_verify_0) / 1000.0
            v_logits = t_out_v.logits

            accepted_tokens = [teacher_argmax]
            n_accepted_drafts = 1
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
                rejected = False
            else:
                accepted_tokens.append(leaf_target_pred)
                rejected = True

            n_emitted, accepted_capped = _extend_with_budget(
                prefix, accepted_tokens, eos_token_ids, prompt_len, max_new_tokens
            )
            if leaf_chosen_idx == 0 and n_emitted >= 2:
                crop_dynamic_cache(target_cache, P + 2)
                chain_h = t_out_v.hidden_states[-2][:, :1, :]
                leaf_h = t_out_v.hidden_states[-2][:, 1:2, :]
                prefix_h_buffer = torch.cat(
                    [prefix_h_buffer, torch.cat([chain_h, leaf_h], dim=1).contiguous()],
                    dim=1,
                )
            else:
                crop_dynamic_cache(target_cache, P + min(1, n_emitted))
                if n_emitted >= 1:
                    chain_h = t_out_v.hidden_states[-2][:, :1, :].contiguous()
                    prefix_h_buffer = torch.cat([prefix_h_buffer, chain_h], dim=1)
        else:
            raise RuntimeError(f"unknown VANTAGE strategy: {decision.strategy}")

        last_logits = None
        rolling.update(ctx.node_type, n_accepted_drafts, k_report)
        t_step_end = time.perf_counter_ns()

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
                wall_us=(t_step_end - t_step_start) / 1000.0,
                draft_us=draft_us,
                verify_us=verify_us,
                parse_us=parse_us,
                strategy=decision.strategy,
                visibility=decision.score,
                frontier_depth=decision.frontier_depth,
                zone=decision.zone,
                draft_top1_prob=draft_top1_prob,
                draft_top2_margin=draft_top2_margin,
                retrieval_match_len=matched_len,
                rolling_accept_rate=rolling_rate,
                ancestor_types=ctx.ancestor_types,
                parser_in_error=ctx.parser_in_error,
                confidence_us=confidence_us,
                retrieval_us=retrieval_us,
                scope_us=scope_us,
                route_us=route_us,
                target_prefill_us=target_prefill_us,
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


def vantage_router_spec(
    prompt_ids: torch.Tensor,
    target,
    eagle_head,
    tokenizer,
    ast_policy: ASTPolicy,
    max_new_tokens: int,
    eos_token_ids: list[int],
    retrieval_index: RetrievalIndex | None = None,
    config: VantageRouterConfig | None = None,
    max_retrieval_draft_len: int = 10,
    max_scope_draft_len: int = 6,
    method_name: str = "vantage_full",
) -> DecodeResult:
    return vantage_router_speculative_ar(
        prompt_ids=prompt_ids,
        target=target,
        eagle_head=eagle_head,
        tokenizer=tokenizer,
        ast_policy=ast_policy,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        retrieval_index=retrieval_index,
        config=config,
        max_retrieval_draft_len=max_retrieval_draft_len,
        max_scope_draft_len=max_scope_draft_len,
        method_name=method_name,
    )
