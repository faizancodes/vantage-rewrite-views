"""Offline replay oracle for exact local suffix proposers.

This script replays already captured greedy completions and asks how many
target forwards a CPU-side proposer would have saved if the target verifier
accepted exactly the tokens in the captured vanilla output. It is not a timing
substitute for GPU runs, but it cheaply kills proposer settings with poor hit
rate or low accepted continuation length.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asts.code_proposers import (  # noqa: E402
    CheapProposerStack,
    CodeSpineProposer,
    EditAnchorProposer,
    LocalSuffixProposer,
    MultiViewPLDProposer,
    MultiViewTreePLDProposer,
    MultiSuffixProposer,
    NGramPromptLookupProposer,
    PrecomputedTransPLDProposer,
    LazyCompeteTransPLDProposer,
    Proposal,
    ProposalFeedback,
    ProposalTree,
    ProposerState,
    DispatchTransPLDProposer,
    RewriteNormalizedPLDProposer,
    RoutedTransPLDProposer,
    SimpleAdoptionTransPLDProposer,
    RootedPLDProposer,
    SymbolTreeProposer,
    TransPLDProposer,
    TransPLDInferenceProposer,
    CompoundTransPLDProposer,
    CursorTrackedTransPLDProposer,
    build_candidate_prefix_tree,
)

_ROOTED_PLD_RE = re.compile(r"vantage_pld_w(?P<w>\d+)_n(?P<n>\d+)$")
_ANCHOR_PLD_RE = re.compile(
    r"vantage_anchor_pld_(?P<tail>tail_)?(?:(?:g(?P<gate>\d+)_))?a(?P<a>\d+)_w(?P<w>\d+)_n(?P<n>\d+)$"
)
_REWRITE_ANCHOR_PLD_RE = re.compile(
    r"vantage_rewrite_anchor_pld_g(?P<gate>\d+)_a(?P<a>\d+)_w(?P<w>\d+)_n(?P<n>\d+)$"
)
_REWRITE_PLD_RE = re.compile(
    r"rewrite_pld_(?P<mode>vref|bidir|oracle)_(?:m(?P<m>\d+)_)?w(?P<w>\d+)_n(?P<n>\d+)$"
)
_TRANSPLD_RE = re.compile(r"vantage_transpld_(?:m(?P<m>\d+)_)?w(?P<w>\d+)_n(?P<n>\d+)$")
_DISPATCH_TRANSPLD_RE = re.compile(
    r"vantage_dispatch_transpld_(?:m(?P<m>\d+)_)?w(?P<w>\d+)_n(?P<n>\d+)$"
)
_PRECOMPUTED_TRANSPLD_RE = re.compile(
    r"vantage_fast_transpld_(?:m(?P<m>\d+)_)?w(?P<w>\d+)_n(?P<n>\d+)$"
)
_COMPETE_TRANSPLD_RE = re.compile(
    r"vantage_compete_transpld_(?:m(?P<m>\d+)_)?(?:exactm(?P<exact_m>\d+)_)?margin(?P<margin>\d+)_w(?P<w>\d+)_n(?P<n>\d+)$"
)
_LAZY_TRANSPLD_RE = re.compile(
    r"vantage_lazy_transpld_s(?P<strong>\d+)_m(?P<margin>\d+)_z(?P<z>\d+)_w(?P<w>\d+)_n(?P<n>\d+)$"
)
_MULTIVIEW_PLD_RE = re.compile(
    r"vantage_mvpld_s(?P<strong>\d+)_m(?P<margin>\d+)_w(?P<w>\d+)_n(?P<n>\d+)$"
)
_MULTIVIEW_TREE_RE = re.compile(
    r"vantage_mvtree_s(?P<strong>\d+)_m(?P<margin>\d+)_w(?P<w>\d+)_n(?P<n>\d+)$"
)
_FROZEN_TRANSPLD_RE = re.compile(r"vantage_frozen_transpld$")
_ROUTED_TRANSPLD_RE = re.compile(
    r"vantage_routed_transpld_(?:m(?P<m>\d+)_)?w(?P<w>\d+)_n(?P<n>\d+)$"
)
_ADOPT_SIMPLE_TRANSPLD_RE = re.compile(
    r"vantage_adopt_simple_transpld_(?:m(?P<m>\d+)_)?w(?P<w>\d+)_n(?P<n>\d+)$"
)
_TRANSPLD_CURSOR_RE = re.compile(
    r"vantage_transpld_cursor_c(?P<c>\d+)_w(?P<w>\d+)_n(?P<n>\d+)$"
)
_TRANSPLD_INFER_RE = re.compile(
    r"vantage_transpld_(?P<mode>infer|inferonly)_w(?P<w>\d+)_n(?P<n>\d+)$"
)
_TRANSPLD_COMPOUND_RE = re.compile(r"vantage_transpld_compound_w(?P<w>\d+)_n(?P<n>\d+)$")
_TRANSPLD_FULL_RE = re.compile(
    r"vantage_transpld_full_c(?P<c>\d+)_w(?P<w>\d+)_n(?P<n>\d+)$"
)


class _CharTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(ch) for ch in text])

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(int(t)) for t in token_ids)


def _load_tokenizer(name: str):
    if name == "char":
        return _CharTokenizer()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def _encode(tokenizer, text: str) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(t) for t in ids]


def _row_text(row: dict, method: str) -> tuple[str, str] | None:
    prompt = row.get("prompt")
    if not prompt:
        return None
    outputs = row.get("outputs") or {}
    if method in outputs and isinstance(outputs[method], dict):
        text = outputs[method].get("text") or outputs[method].get("completion")
        if text is not None:
            return prompt, str(text)
    if "vanilla" in outputs and isinstance(outputs["vanilla"], dict):
        text = outputs["vanilla"].get("text") or outputs["vanilla"].get("completion")
        if text is not None:
            return prompt, str(text)
    text = row.get(method) or row.get("completion") or row.get("vanilla")
    if text is not None:
        return prompt, str(text)
    return None


def _proposer_for_method(method: str, tokenizer):
    def _transpld_min(match) -> int:
        groups = match.groupdict()
        return int(groups.get("m") or 4)

    def _exact_pld_min(match) -> int:
        groups = match.groupdict()
        return int(groups.get("exact_m") or 1)

    rooted_pld = _ROOTED_PLD_RE.fullmatch(method)
    if rooted_pld:
        return RootedPLDProposer(
            max_draft_len=int(rooted_pld.group("w")),
            max_matching_ngram_size=int(rooted_pld.group("n")),
            min_matching_ngram_size=1,
        )
    rewrite_pld = _REWRITE_PLD_RE.fullmatch(method)
    if rewrite_pld:
        return RewriteNormalizedPLDProposer(
            tokenizer,
            mode=rewrite_pld.group("mode"),
            max_draft_len=int(rewrite_pld.group("w")),
            max_matching_ngram_size=int(rewrite_pld.group("n")),
            min_matching_ngram_size=_transpld_min(rewrite_pld),
        )
    transpld = _TRANSPLD_RE.fullmatch(method)
    if transpld:
        return TransPLDProposer(
            tokenizer,
            max_draft_len=int(transpld.group("w")),
            max_matching_ngram_size=int(transpld.group("n")),
            min_matching_ngram_size=1,
            transformed_min_matching_ngram_size=_transpld_min(transpld),
        )
    dispatch_transpld = _DISPATCH_TRANSPLD_RE.fullmatch(method)
    if dispatch_transpld:
        return DispatchTransPLDProposer(
            tokenizer,
            max_draft_len=int(dispatch_transpld.group("w")),
            max_matching_ngram_size=int(dispatch_transpld.group("n")),
            transformed_min_matching_ngram_size=_transpld_min(dispatch_transpld),
        )
    precomputed_transpld = _PRECOMPUTED_TRANSPLD_RE.fullmatch(method)
    if precomputed_transpld:
        return PrecomputedTransPLDProposer(
            tokenizer,
            max_draft_len=int(precomputed_transpld.group("w")),
            max_matching_ngram_size=int(precomputed_transpld.group("n")),
            transformed_min_matching_ngram_size=_transpld_min(precomputed_transpld),
            compete_exact=False,
        )
    compete_transpld = _COMPETE_TRANSPLD_RE.fullmatch(method)
    if compete_transpld:
        return PrecomputedTransPLDProposer(
            tokenizer,
            max_draft_len=int(compete_transpld.group("w")),
            max_matching_ngram_size=int(compete_transpld.group("n")),
            transformed_min_matching_ngram_size=_transpld_min(compete_transpld),
            exact_min_matching_ngram_size=_exact_pld_min(compete_transpld),
            compete_exact=True,
            margin=int(compete_transpld.group("margin")),
        )
    lazy_transpld = _LAZY_TRANSPLD_RE.fullmatch(method)
    if lazy_transpld:
        return LazyCompeteTransPLDProposer(
            tokenizer,
            max_draft_len=int(lazy_transpld.group("w")),
            max_matching_ngram_size=int(lazy_transpld.group("n")),
            transformed_min_matching_ngram_size=_transpld_min(lazy_transpld),
            exact_strong_min_len=int(lazy_transpld.group("strong")),
            trans_len_margin=int(lazy_transpld.group("margin")),
            zero_accept_tripwire_limit=int(lazy_transpld.group("z")),
        )
    multiview_pld = _MULTIVIEW_PLD_RE.fullmatch(method)
    if multiview_pld:
        return MultiViewPLDProposer(
            tokenizer,
            max_draft_len=int(multiview_pld.group("w")),
            max_matching_ngram_size=int(multiview_pld.group("n")),
            transformed_min_matching_ngram_size=_transpld_min(multiview_pld),
            exact_strong_min_len=int(multiview_pld.group("strong")),
            trans_len_margin=int(multiview_pld.group("margin")),
        )
    multiview_tree = _MULTIVIEW_TREE_RE.fullmatch(method)
    if multiview_tree:
        return MultiViewTreePLDProposer(
            tokenizer,
            max_draft_len=int(multiview_tree.group("w")),
            max_matching_ngram_size=int(multiview_tree.group("n")),
            transformed_min_matching_ngram_size=_transpld_min(multiview_tree),
            exact_strong_min_len=int(multiview_tree.group("strong")),
            trans_len_margin=int(multiview_tree.group("margin")),
        )
    if _FROZEN_TRANSPLD_RE.fullmatch(method):
        return PrecomputedTransPLDProposer(
            tokenizer,
            max_draft_len=128,
            max_matching_ngram_size=10,
            transformed_min_matching_ngram_size=4,
            compete_exact=True,
            margin=0,
        )
    routed_transpld = _ROUTED_TRANSPLD_RE.fullmatch(method)
    if routed_transpld:
        return RoutedTransPLDProposer(
            tokenizer,
            max_draft_len=int(routed_transpld.group("w")),
            max_matching_ngram_size=int(routed_transpld.group("n")),
            min_matching_ngram_size=1,
            transformed_min_matching_ngram_size=_transpld_min(routed_transpld),
        )
    adopt_simple_transpld = _ADOPT_SIMPLE_TRANSPLD_RE.fullmatch(method)
    if adopt_simple_transpld:
        return SimpleAdoptionTransPLDProposer(
            tokenizer,
            max_draft_len=int(adopt_simple_transpld.group("w")),
            max_matching_ngram_size=int(adopt_simple_transpld.group("n")),
            transformed_min_matching_ngram_size=_transpld_min(adopt_simple_transpld),
        )
    transpld_cursor = _TRANSPLD_CURSOR_RE.fullmatch(method)
    if transpld_cursor:
        return CursorTrackedTransPLDProposer(
            tokenizer,
            max_cursor_draft_len=int(transpld_cursor.group("c")),
            max_draft_len=int(transpld_cursor.group("w")),
            max_matching_ngram_size=int(transpld_cursor.group("n")),
        )
    transpld_infer = _TRANSPLD_INFER_RE.fullmatch(method)
    if transpld_infer:
        return TransPLDInferenceProposer(
            tokenizer,
            infer_only=transpld_infer.group("mode") == "inferonly",
            max_draft_len=int(transpld_infer.group("w")),
            max_matching_ngram_size=int(transpld_infer.group("n")),
        )
    transpld_compound = _TRANSPLD_COMPOUND_RE.fullmatch(method)
    if transpld_compound:
        return CompoundTransPLDProposer(
            tokenizer,
            max_draft_len=int(transpld_compound.group("w")),
            max_matching_ngram_size=int(transpld_compound.group("n")),
        )
    transpld_full = _TRANSPLD_FULL_RE.fullmatch(method)
    if transpld_full:
        return CursorTrackedTransPLDProposer(
            tokenizer,
            max_cursor_draft_len=int(transpld_full.group("c")),
            max_draft_len=int(transpld_full.group("w")),
            max_matching_ngram_size=int(transpld_full.group("n")),
            infer=True,
            compound=True,
        )
    anchor_pld = _ANCHOR_PLD_RE.fullmatch(method)
    if anchor_pld:
        return CheapProposerStack(
            [
                EditAnchorProposer(
                    tokenizer,
                    kind="edit_anchor_pld",
                    max_draft_len=int(anchor_pld.group("a")),
                    min_draft_tokens=int(anchor_pld.group("gate") or 1),
                ),
                RootedPLDProposer(
                    max_draft_len=int(anchor_pld.group("w")),
                    max_matching_ngram_size=int(anchor_pld.group("n")),
                    min_matching_ngram_size=1,
                ),
            ]
        )
    rewrite_anchor_pld = _REWRITE_ANCHOR_PLD_RE.fullmatch(method)
    if rewrite_anchor_pld:
        return CheapProposerStack(
            [
                EditAnchorProposer(
                    tokenizer,
                    kind="rewrite_anchor_pld",
                    max_draft_len=int(rewrite_anchor_pld.group("a")),
                    min_draft_tokens=int(rewrite_anchor_pld.group("gate")),
                    rewrite_enabled=True,
                ),
                RootedPLDProposer(
                    max_draft_len=int(rewrite_anchor_pld.group("w")),
                    max_matching_ngram_size=int(rewrite_anchor_pld.group("n")),
                    min_matching_ngram_size=1,
                ),
            ]
        )
    if method == "suffix":
        return LocalSuffixProposer(min_match_len=4, max_query_len=16, max_draft_len=8)
    if method == "ngram_m4d3":
        return NGramPromptLookupProposer(4, 3, pool="local")
    if method == "ngram_m4d5":
        return NGramPromptLookupProposer(4, 5, pool="local")
    if method == "ngram_m4d8":
        return NGramPromptLookupProposer(4, 8, pool="local")
    if method == "adaptive_suffix":
        return MultiSuffixProposer(tree=False, top_k=1, max_draft_len=16)
    if method == "multisuffix_k2":
        return MultiSuffixProposer(tree=True, top_k=2, max_tree_nodes=12)
    if method == "multisuffix_k4":
        return MultiSuffixProposer(tree=True, top_k=4, max_tree_nodes=12)
    if method == "multisuffix_k8":
        return MultiSuffixProposer(tree=True, top_k=8, max_tree_nodes=16)
    if method == "multisuffix_cap8":
        return MultiSuffixProposer(tree=True, top_k=4, max_tree_nodes=8)
    if method == "multisuffix_cap16":
        return MultiSuffixProposer(tree=True, top_k=4, max_tree_nodes=16)
    if method == "codespine":
        return CodeSpineProposer(tokenizer, kind="code_spine")
    if method == "codespine_cap8":
        return CodeSpineProposer(tokenizer, kind="code_spine", max_tree_nodes=8)
    if method == "codespine_cap16":
        return CodeSpineProposer(tokenizer, kind="code_spine", max_tree_nodes=16)
    if method == "editspine":
        return CodeSpineProposer(tokenizer, kind="edit_spine", edit_mode=True)
    if method == "edit_anchor":
        return EditAnchorProposer(tokenizer)
    if method == "edit_anchor_tail":
        return CheapProposerStack(
            [
                EditAnchorProposer(tokenizer, kind="edit_anchor_tail"),
                LocalSuffixProposer(),
            ]
        )
    if method == "symbol_tree":
        return CheapProposerStack([LocalSuffixProposer(), SymbolTreeProposer(tokenizer)])
    if method == "edit_symbol_tail":
        return CheapProposerStack(
            [
                EditAnchorProposer(tokenizer),
                LocalSuffixProposer(),
                SymbolTreeProposer(tokenizer),
            ]
        )
    raise ValueError(f"unknown replay method: {method}")


def _replay_chain(proposal: Proposal, future: list[int]) -> int:
    accepted = 0
    for proposed, actual in zip(proposal.tokens, future):
        if int(proposed) != int(actual):
            break
        accepted += 1
    return accepted


def _replay_tree(proposal: ProposalTree, future: list[int]) -> tuple[int, int | None]:
    nodes = build_candidate_prefix_tree(proposal.candidates, proposal.max_nodes)
    if not nodes:
        return 0, None
    children: dict[int, list[int]] = defaultdict(list)
    tokens = [None] + [n.token for n in nodes]
    for idx, node in enumerate(nodes, start=1):
        parent = 0 if node.parent < 0 else node.parent + 1
        children[parent].append(idx)
    accepted = 0
    current = 0
    selected: int | None = None
    for actual in future:
        match = None
        for child in children.get(current, []):
            if tokens[child] == actual:
                match = child
                break
        if match is None:
            break
        accepted += 1
        selected = match
        current = match
    return accepted, selected


def _candidate_rank_for_accept(proposal: ProposalTree, accepted: int, future: list[int]) -> int | None:
    if accepted <= 0:
        return None
    accepts = [_replay_chain(Proposal(proposal.kind, c, proposal.match_len, 0.0), future) for c in proposal.candidates]
    for i, value in enumerate(accepts):
        if value == accepted:
            return i
    return None


def _match_bucket(match_len: int) -> str:
    if match_len >= 12:
        return "12+"
    if match_len >= 8:
        return "8-11"
    if match_len >= 4:
        return "4-7"
    return "3"


def _acc_bucket(accepted: int) -> str:
    if accepted == 0:
        return "0"
    if accepted == 1:
        return "1"
    if accepted == 2:
        return "2"
    if accepted <= 4:
        return "3-4"
    if accepted <= 8:
        return "5-8"
    return "9+"


def _merge_stats(dst: defaultdict, src: dict) -> None:
    for key, value in src.items():
        if isinstance(value, dict):
            current = dst.setdefault(key, defaultdict(float))
            for k2, v2 in value.items():
                current[k2] += v2
        else:
            dst[key] += value


def replay_one(
    *,
    prompt_ids: list[int],
    prompt_text: str,
    completion_ids: list[int],
    tokenizer,
    proposer,
    max_new_tokens: int | None,
    reference: str = "",
    metadata: dict | None = None,
) -> dict:
    full = prompt_ids + completion_ids
    end = len(full) if max_new_tokens is None else min(len(full), len(prompt_ids) + max_new_tokens)
    pos = len(prompt_ids)
    steps = 0
    hits = 0
    zero_hits = 0
    accepted_nonroot = 0
    proposed_tokens = 0
    tree_nodes = 0
    tree_hits = 0
    by_match: dict[str, float] = defaultdict(float)
    accepted_by_match: dict[str, float] = defaultdict(float)
    by_accept: dict[str, float] = defaultdict(float)
    by_pool: dict[str, float] = defaultdict(float)
    by_kind: dict[str, float] = defaultdict(float)
    accepted_by_kind: dict[str, float] = defaultdict(float)
    candidate_count_hist: dict[str, float] = defaultdict(float)
    selected_rank_hist: dict[str, float] = defaultdict(float)
    first_candidate_accept = 0
    best_candidate_accept = 0
    tree_extra_accept = 0
    tree_beats_first = 0
    tree_matches_first = 0
    tree_loses_first = 0
    reset = getattr(proposer, "reset", None)
    if callable(reset):
        reset()
    prepare = getattr(proposer, "prepare", None)
    if callable(prepare):
        prepare(
            prompt_ids=prompt_ids,
            prompt_text=prompt_text,
            reference=reference,
            metadata=metadata,
            prompt_len=len(prompt_ids),
            language="python",
        )
    while pos < end:
        step_start = pos
        root = full[pos]
        prefix = full[:pos]
        text_after = tokenizer.decode(prefix + [root], skip_special_tokens=False)
        proposal = proposer.propose(
            ProposerState(
                prefix=prefix,
                teacher_argmax=root,
                text_before=tokenizer.decode(prefix, skip_special_tokens=False),
                text_after=text_after,
                ctx=None,
                language="python",
                prompt_len=len(prompt_ids),
                reference=reference,
                metadata=metadata,
                prompt_text=prompt_text,
            )
        )
        steps += 1
        future = full[pos + 1 : end]
        accepted = 0
        if isinstance(proposal, ProposalTree):
            hits += 1
            accepted, selected = _replay_tree(proposal, future)
            nodes = build_candidate_prefix_tree(proposal.candidates, proposal.max_nodes)
            tree_nodes += len(nodes)
            tree_hits += 1 if selected is not None else 0
            proposed_tokens += sum(len(c) for c in proposal.candidates)
            candidate_accepts = [
                _replay_chain(Proposal(proposal.kind, c, proposal.match_len, 0.0), future)
                for c in proposal.candidates
            ]
            first_accept = candidate_accepts[0] if candidate_accepts else 0
            best_accept = max(candidate_accepts) if candidate_accepts else 0
            first_candidate_accept += first_accept
            best_candidate_accept += best_accept
            tree_extra_accept += max(0, accepted - first_accept)
            if accepted > first_accept:
                tree_beats_first += 1
            elif accepted == first_accept:
                tree_matches_first += 1
            else:
                tree_loses_first += 1
            rank = _candidate_rank_for_accept(proposal, accepted, future)
            if rank is not None:
                selected_rank_hist[str(rank + 1)] += 1
            candidate_count_hist[str(len(proposal.candidates))] += 1
        elif isinstance(proposal, Proposal):
            hits += 1
            accepted = _replay_chain(proposal, future)
            proposed_tokens += len(proposal.tokens)
        if proposal is not None:
            bucket = _match_bucket(proposal.match_len)
            by_match[bucket] += 1
            accepted_by_match[bucket] += accepted
            by_accept[_acc_bucket(accepted)] += 1
            by_pool[str(proposal.pool or "unknown")] += 1
            by_kind[str(proposal.kind or "unknown")] += 1
            accepted_by_kind[str(proposal.kind or "unknown")] += accepted
            if accepted == 0:
                zero_hits += 1
            accepted_nonroot += accepted
        # Root is guaranteed. A verifier also emits either one correction after
        # the accepted prefix or one bonus token if all proposal tokens match.
        pos += 1 + accepted
        if pos < end:
            pos += 1
        observe = getattr(proposer, "observe", None)
        if callable(observe):
            observe(
                ProposalFeedback(
                    prefix_start=step_start,
                    prefix_end=pos,
                    proposed_tokens=list(proposal.tokens) if proposal is not None else [],
                    emitted_tokens=list(full[step_start:pos]),
                    accepted_nonroot=accepted,
                    rejected=bool(
                        proposal is not None
                        and isinstance(proposal, Proposal)
                        and accepted < len(proposal.tokens)
                    ),
                    proposal_kind=proposal.kind if proposal is not None else None,
                    proposal_match_kind=proposal.match_kind if proposal is not None else None,
                    source_start_token=proposal.source_start_token if proposal is not None else None,
                    source_end_token=proposal.source_end_token if proposal is not None else None,
                    follow_start_token=proposal.follow_start_token if proposal is not None else None,
                    follow_end_token=proposal.follow_end_token if proposal is not None else None,
                )
            )
    return {
        "steps": steps,
        "tokens": max(0, end - len(prompt_ids)),
        "hits": hits,
        "zero_hits": zero_hits,
        "accepted_nonroot": accepted_nonroot,
        "proposed_tokens": proposed_tokens,
        "tree_nodes": tree_nodes,
        "tree_hits": tree_hits,
        "by_match": dict(by_match),
        "accepted_by_match": dict(accepted_by_match),
        "by_accept": dict(by_accept),
        "by_pool": dict(by_pool),
        "by_kind": dict(by_kind),
        "accepted_by_kind": dict(accepted_by_kind),
        "candidate_count_hist": dict(candidate_count_hist),
        "selected_rank_hist": dict(selected_rank_hist),
        "first_candidate_accept": first_candidate_accept,
        "best_candidate_accept": best_candidate_accept,
        "tree_extra_accept": tree_extra_accept,
        "tree_beats_first": tree_beats_first,
        "tree_matches_first": tree_matches_first,
        "tree_loses_first": tree_loses_first,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--completions", required=True)
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-Coder-7B")
    p.add_argument(
        "--methods",
        default=(
            "suffix,ngram_m4d5,adaptive_suffix,multisuffix_k4,codespine,editspine,"
            "edit_anchor,edit_anchor_tail,symbol_tree,edit_symbol_tail,"
            "vantage_pld_w40_n10,vantage_anchor_pld_a128_w40_n10,"
            "vantage_anchor_pld_g64_a128_w40_n10,"
            "vantage_rewrite_anchor_pld_g32_a128_w40_n10,"
            "rewrite_pld_vref_w128_n10,rewrite_pld_bidir_w128_n10,"
            "rewrite_pld_oracle_w128_n10,vantage_transpld_w128_n10,"
            "vantage_transpld_cursor_c256_w128_n10,"
            "vantage_transpld_inferonly_w128_n10,"
            "vantage_transpld_compound_w128_n10,"
            "vantage_transpld_full_c256_w128_n10,"
            "vantage_mvpld_s32_m0_w128_n10,"
            "vantage_mvtree_s32_m0_w128_n10"
        ),
    )
    p.add_argument("--source-method", default="vanilla")
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--output-json", default="")
    args = p.parse_args()

    tokenizer = _load_tokenizer(args.tokenizer)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    totals = {m: defaultdict(float) for m in methods}
    n_rows = 0

    with open(args.completions) as f:
        for line in f:
            if args.max_rows is not None and n_rows >= args.max_rows:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            pair = _row_text(row, args.source_method)
            if pair is None:
                continue
            prompt, completion = pair
            prompt_ids = _encode(tokenizer, prompt)
            completion_ids = _encode(tokenizer, completion)
            if not completion_ids:
                continue
            row_metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            metadata = dict(row_metadata)
            if "rewrite_pairs" in row and "rewrite_pairs" not in metadata:
                metadata["rewrite_pairs"] = row["rewrite_pairs"]
            reference = str(row.get("reference") or metadata.get("reference") or "")
            for method in methods:
                stats = replay_one(
                    prompt_ids=prompt_ids,
                    prompt_text=prompt,
                    completion_ids=completion_ids,
                    tokenizer=tokenizer,
                    proposer=_proposer_for_method(method, tokenizer),
                    max_new_tokens=args.max_new_tokens,
                    reference=reference,
                    metadata=metadata,
                )
                _merge_stats(totals[method], stats)
            n_rows += 1

    report = {"n_rows": n_rows, "methods": {}}
    for method in methods:
        t = totals[method]
        steps = max(1.0, t["steps"])
        hits = max(1.0, t["hits"])
        tokens = max(1.0, t["tokens"])
        report["methods"][method] = {
            "tokens": int(t["tokens"]),
            "steps": int(t["steps"]),
            "tokens_per_verify": t["tokens"] / steps,
            "hit_rate": t["hits"] / steps,
            "zero_hit_rate": t["zero_hits"] / hits,
            "accepted_nonroot_per_hit": t["accepted_nonroot"] / hits,
            "accepted_nonroot_per_step": t["accepted_nonroot"] / steps,
            "accepted_nonroot_per_token": t["accepted_nonroot"] / tokens,
            "mean_proposed_tokens_per_hit": t["proposed_tokens"] / hits,
            "mean_tree_nodes_per_tree_hit": t["tree_nodes"] / max(1.0, t["tree_hits"]),
            "mean_first_candidate_accept_per_hit": t["first_candidate_accept"] / hits,
            "mean_best_candidate_accept_per_hit": t["best_candidate_accept"] / hits,
            "mean_tree_extra_accept_per_hit": t["tree_extra_accept"] / hits,
            "tree_beats_first_rate": t["tree_beats_first"] / hits,
            "tree_matches_first_rate": t["tree_matches_first"] / hits,
            "tree_loses_first_rate": t["tree_loses_first"] / hits,
            "accepted_per_tree_node": (
                t["accepted_nonroot"] / t["tree_nodes"] if t["tree_nodes"] else 0.0
            ),
            "by_match": {
                k: {
                    "hits": int(v),
                    "share_hits": v / hits,
                    "accepted_per_hit": (t["accepted_by_match"].get(k, 0.0) / v if v else 0.0),
                }
                for k, v in sorted((t.get("by_match") or {}).items())
            },
            "by_accept": {
                k: int(v)
                for k, v in sorted((t.get("by_accept") or {}).items())
            },
            "by_pool": {
                k: {
                    "hits": int(v),
                    "share_hits": v / hits,
                }
                for k, v in sorted((t.get("by_pool") or {}).items())
            },
            "by_kind": {
                k: {
                    "hits": int(v),
                    "share_hits": v / hits,
                    "accepted_per_hit": (t["accepted_by_kind"].get(k, 0.0) / v if v else 0.0),
                }
                for k, v in sorted((t.get("by_kind") or {}).items())
            },
            "candidate_count_hist": {
                k: int(v)
                for k, v in sorted((t.get("candidate_count_hist") or {}).items())
            },
            "selected_rank_hist": {
                k: int(v)
                for k, v in sorted((t.get("selected_rank_hist") or {}).items(), key=lambda item: int(item[0]))
            },
        }
    print(json.dumps(report, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
