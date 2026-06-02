"""Lossless smoke test for EAGLE decoder.

Verifies that vanilla AR == fixed_eagle_spec == asts_eagle_spec, byte-identical,
on N HumanEval prompts in greedy mode. Run with --strict-determinism + fp32 to
eliminate bf16 numerical noise.

Usage:
    python scripts/verify_eagle_lossless.py \\
        --eagle-checkpoint /data/eagle_v0/eagle/eagle_final.pt \\
        --n 3 --max-new-tokens 32 --strict-determinism --dtype float32
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from asts.ast_policy import ASTPolicy
from asts.blazedit_decoder import (
    blazedit_speculative_ar,
    is_blazedit_method,
    is_vantage_mv_method,
    vantage_mv_pld_speculative_ar,
    parse_blazedit_method,
    parse_vantage_mv_method,
)
from asts.code_proposer_decoder import code_proposer_spec
from asts.code_proposers import (
    AlphaSuffixProposer,
    CheapProposerStack,
    CodeSpineProposer,
    EditAnchorProposer,
    IdentifierTrieProposer,
    LiteralCopyProposer,
    LocalSuffixProposer,
    MacroChunkProposer,
    MultiViewPLDProposer,
    MultiViewTreePLDProposer,
    MultiSuffixProposer,
    NGramPromptLookupProposer,
    PrecomputedTransPLDProposer,
    DispatchTransPLDProposer,
    RewriteNormalizedPLDProposer,
    RoutedTransPLDProposer,
    SimpleAdoptionTransPLDProposer,
    LazyCompeteTransPLDProposer,
    RootedPLDProposer,
    SymbolTreeProposer,
    TransPLDProposer,
    TransPLDInferenceProposer,
    CompoundTransPLDProposer,
    CursorTrackedTransPLDProposer,
    _apply_word_map,
    _extract_reference_blocks,
    _rewrite_pairs,
    encode_no_special,
    static_macro_chunks,
)
from asts.decoder import vanilla_ar
from asts.eagle2_decoder import eagle2_speculative_ar
from asts.eagle_decoder import asts_eagle_spec, fixed_eagle_spec, load_eagle_checkpoint
from asts.humaneval import load_problems_from_jsonl, load_problems_for_language
from asts.model_bench import _load_model
from asts.vantage_policy import VantageRouterConfig, decide_prompt_only_saferoute
from asts.vantage_router import vantage_router_spec
from asts.retrieval_draft import RetrievalIndex
from asts.tree_eagle_decoder import tree_tail_eagle_spec


CODE_PROPOSER_METHODS = (
    "vantage_id",
    "vantage_literal",
    "vantage_suffix",
    "vantage_suffix_prompt",
    "vantage_suffix_generated",
    "vantage_macro_static",
    "vantage_code_stack",
    "vantage_code_tail",
    "vantage_code_tail_w3",
    "vantage_code_tail_w4",
    "vantage_code_tail_context",
    "ngram_prompt_m4d5",
    "ngram_local_m4d3",
    "ngram_local_m4d5",
    "ngram_local_m4d8",
    "vantage_alpha_v0",
    "vantage_alpha",
    "vantage_alpha_tail_w4",
    "vantage_alpha_only",
    "alpha_idnorm",
    "alpha_idlitnorm",
    "alpha_idlit_subst",
    "alpha_role_no_subst",
    "alpha_role_subst",
    "vantage_multisuffix_chain",
    "vantage_multisuffix_tree",
    "vantage_multisuffix_tail_w4",
    "vantage_repoedit_suffix",
    "vantage_codespine",
    "vantage_editspine",
    "vantage_spinetail",
    "vantage_edit_anchor",
    "vantage_edit_anchor_only",
    "vantage_edit_anchor_suffix",
    "vantage_suffix_tail_w4",
    "vantage_edit_anchor_tail",
    "vantage_symbol_tree",
    "vantage_edit_symbol_tail",
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


def _is_dynamic_code_proposer_method(method: str) -> bool:
    return (
        _ROOTED_PLD_RE.fullmatch(method) is not None
        or _ANCHOR_PLD_RE.fullmatch(method) is not None
        or _REWRITE_ANCHOR_PLD_RE.fullmatch(method) is not None
        or _REWRITE_PLD_RE.fullmatch(method) is not None
        or _TRANSPLD_RE.fullmatch(method) is not None
        or _DISPATCH_TRANSPLD_RE.fullmatch(method) is not None
        or _PRECOMPUTED_TRANSPLD_RE.fullmatch(method) is not None
        or _COMPETE_TRANSPLD_RE.fullmatch(method) is not None
        or _LAZY_TRANSPLD_RE.fullmatch(method) is not None
        or _MULTIVIEW_PLD_RE.fullmatch(method) is not None
        or _MULTIVIEW_TREE_RE.fullmatch(method) is not None
        or _FROZEN_TRANSPLD_RE.fullmatch(method) is not None
        or _ROUTED_TRANSPLD_RE.fullmatch(method) is not None
        or _ADOPT_SIMPLE_TRANSPLD_RE.fullmatch(method) is not None
        or _TRANSPLD_CURSOR_RE.fullmatch(method) is not None
        or _TRANSPLD_INFER_RE.fullmatch(method) is not None
        or _TRANSPLD_COMPOUND_RE.fullmatch(method) is not None
        or _TRANSPLD_FULL_RE.fullmatch(method) is not None
    )


def _set_deterministic():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _problem_reference(prob) -> str:
    if getattr(prob, "reference", ""):
        return str(prob.reference)
    refs = _extract_reference_blocks(str(prob.prompt))
    return refs[0] if refs else ""


def _problem_rewrite_map(prob) -> dict[str, str]:
    return _rewrite_pairs(str(prob.prompt))


def _routed_transpld_exact_route_reason(prob, tokenizer) -> str | None:
    reference = _problem_reference(prob)
    rewrite_map = _problem_rewrite_map(prob)
    transformed_reference = _apply_word_map(reference, rewrite_map) if rewrite_map else reference
    decision = decide_prompt_only_saferoute(
        reference=reference,
        rewrite_map=rewrite_map,
        transformed_reference=transformed_reference,
        reference_tokens=encode_no_special(tokenizer, reference) if reference else [],
        transformed_tokens=(
            encode_no_special(tokenizer, transformed_reference)
            if transformed_reference
            else []
        ),
    )
    return None if decision.use_transpld else decision.reason


def _routed_transpld_pld_config(method: str, args: argparse.Namespace):
    match = (
        _ROUTED_TRANSPLD_RE.fullmatch(method)
        or _ADOPT_SIMPLE_TRANSPLD_RE.fullmatch(method)
        or _DISPATCH_TRANSPLD_RE.fullmatch(method)
        or _PRECOMPUTED_TRANSPLD_RE.fullmatch(method)
        or _COMPETE_TRANSPLD_RE.fullmatch(method)
        or _LAZY_TRANSPLD_RE.fullmatch(method)
        or _MULTIVIEW_PLD_RE.fullmatch(method)
        or _MULTIVIEW_TREE_RE.fullmatch(method)
        or _FROZEN_TRANSPLD_RE.fullmatch(method)
    )
    if not match:
        raise ValueError(f"not a dispatchable TransPLD method: {method}")
    if _FROZEN_TRANSPLD_RE.fullmatch(method):
        return parse_blazedit_method(
            "blazedit_pld_w128_n10",
            assistant_model_name=args.assistant_model,
            confidence_threshold=args.blazedit_assistant_confidence_threshold,
            default_ngram_size=args.blazedit_max_matching_ngram_size,
        )
    return parse_blazedit_method(
        f"blazedit_pld_w{match.group('w')}_n{match.group('n')}",
        assistant_model_name=args.assistant_model,
        confidence_threshold=args.blazedit_assistant_confidence_threshold,
        default_ngram_size=args.blazedit_max_matching_ngram_size,
    )


def _stack_for_method(method: str, tokenizer, benchmark: str) -> CheapProposerStack:
    proposers = []
    def _transpld_min(match) -> int:
        groups = match.groupdict()
        return int(groups.get("m") or 4)

    def _exact_pld_min(match) -> int:
        groups = match.groupdict()
        return int(groups.get("exact_m") or 1)

    rooted_pld = _ROOTED_PLD_RE.fullmatch(method)
    if rooted_pld:
        proposers.append(
            RootedPLDProposer(
                max_draft_len=int(rooted_pld.group("w")),
                max_matching_ngram_size=int(rooted_pld.group("n")),
                min_matching_ngram_size=1,
            )
        )
    rewrite_pld = _REWRITE_PLD_RE.fullmatch(method)
    if rewrite_pld:
        proposers.append(
            RewriteNormalizedPLDProposer(
                tokenizer,
                mode=rewrite_pld.group("mode"),
                max_draft_len=int(rewrite_pld.group("w")),
                max_matching_ngram_size=int(rewrite_pld.group("n")),
                min_matching_ngram_size=_transpld_min(rewrite_pld),
            )
        )
    transpld = _TRANSPLD_RE.fullmatch(method)
    if transpld:
        proposers.append(
            TransPLDProposer(
                tokenizer,
                max_draft_len=int(transpld.group("w")),
                max_matching_ngram_size=int(transpld.group("n")),
                min_matching_ngram_size=1,
                transformed_min_matching_ngram_size=_transpld_min(transpld),
            )
        )
    dispatch_transpld = _DISPATCH_TRANSPLD_RE.fullmatch(method)
    if dispatch_transpld:
        proposers.append(
            DispatchTransPLDProposer(
                tokenizer,
                max_draft_len=int(dispatch_transpld.group("w")),
                max_matching_ngram_size=int(dispatch_transpld.group("n")),
                transformed_min_matching_ngram_size=_transpld_min(dispatch_transpld),
            )
        )
    precomputed_transpld = _PRECOMPUTED_TRANSPLD_RE.fullmatch(method)
    if precomputed_transpld:
        proposers.append(
            PrecomputedTransPLDProposer(
                tokenizer,
                max_draft_len=int(precomputed_transpld.group("w")),
                max_matching_ngram_size=int(precomputed_transpld.group("n")),
                transformed_min_matching_ngram_size=_transpld_min(precomputed_transpld),
                compete_exact=False,
            )
        )
    compete_transpld = _COMPETE_TRANSPLD_RE.fullmatch(method)
    if compete_transpld:
        proposers.append(
            PrecomputedTransPLDProposer(
                tokenizer,
                max_draft_len=int(compete_transpld.group("w")),
                max_matching_ngram_size=int(compete_transpld.group("n")),
                transformed_min_matching_ngram_size=_transpld_min(compete_transpld),
                exact_min_matching_ngram_size=_exact_pld_min(compete_transpld),
                compete_exact=True,
                margin=int(compete_transpld.group("margin")),
            )
        )
    lazy_transpld = _LAZY_TRANSPLD_RE.fullmatch(method)
    if lazy_transpld:
        proposers.append(
            LazyCompeteTransPLDProposer(
                tokenizer,
                max_draft_len=int(lazy_transpld.group("w")),
                max_matching_ngram_size=int(lazy_transpld.group("n")),
                transformed_min_matching_ngram_size=_transpld_min(lazy_transpld),
                exact_strong_min_len=int(lazy_transpld.group("strong")),
                trans_len_margin=int(lazy_transpld.group("margin")),
                zero_accept_tripwire_limit=int(lazy_transpld.group("z")),
            )
        )
    multiview_pld = _MULTIVIEW_PLD_RE.fullmatch(method)
    if multiview_pld:
        proposers.append(
            MultiViewPLDProposer(
                tokenizer,
                max_draft_len=int(multiview_pld.group("w")),
                max_matching_ngram_size=int(multiview_pld.group("n")),
                transformed_min_matching_ngram_size=_transpld_min(multiview_pld),
                exact_strong_min_len=int(multiview_pld.group("strong")),
                trans_len_margin=int(multiview_pld.group("margin")),
            )
        )
    multiview_tree = _MULTIVIEW_TREE_RE.fullmatch(method)
    if multiview_tree:
        proposers.append(
            MultiViewTreePLDProposer(
                tokenizer,
                max_draft_len=int(multiview_tree.group("w")),
                max_matching_ngram_size=int(multiview_tree.group("n")),
                transformed_min_matching_ngram_size=_transpld_min(multiview_tree),
                exact_strong_min_len=int(multiview_tree.group("strong")),
                trans_len_margin=int(multiview_tree.group("margin")),
            )
        )
    if _FROZEN_TRANSPLD_RE.fullmatch(method):
        proposers.append(
            PrecomputedTransPLDProposer(
                tokenizer,
                max_draft_len=128,
                max_matching_ngram_size=10,
                transformed_min_matching_ngram_size=4,
                compete_exact=True,
                margin=0,
            )
        )
    routed_transpld = _ROUTED_TRANSPLD_RE.fullmatch(method)
    if routed_transpld:
        proposers.append(
            RoutedTransPLDProposer(
                tokenizer,
                max_draft_len=int(routed_transpld.group("w")),
                max_matching_ngram_size=int(routed_transpld.group("n")),
                min_matching_ngram_size=1,
                transformed_min_matching_ngram_size=_transpld_min(routed_transpld),
            )
        )
    adopt_simple_transpld = _ADOPT_SIMPLE_TRANSPLD_RE.fullmatch(method)
    if adopt_simple_transpld:
        proposers.append(
            SimpleAdoptionTransPLDProposer(
                tokenizer,
                max_draft_len=int(adopt_simple_transpld.group("w")),
                max_matching_ngram_size=int(adopt_simple_transpld.group("n")),
                min_matching_ngram_size=1,
                transformed_min_matching_ngram_size=_transpld_min(adopt_simple_transpld),
            )
        )
    transpld_cursor = _TRANSPLD_CURSOR_RE.fullmatch(method)
    if transpld_cursor:
        proposers.append(
            CursorTrackedTransPLDProposer(
                tokenizer,
                max_cursor_draft_len=int(transpld_cursor.group("c")),
                max_draft_len=int(transpld_cursor.group("w")),
                max_matching_ngram_size=int(transpld_cursor.group("n")),
            )
        )
    transpld_infer = _TRANSPLD_INFER_RE.fullmatch(method)
    if transpld_infer:
        proposers.append(
            TransPLDInferenceProposer(
                tokenizer,
                infer_only=transpld_infer.group("mode") == "inferonly",
                max_draft_len=int(transpld_infer.group("w")),
                max_matching_ngram_size=int(transpld_infer.group("n")),
            )
        )
    transpld_compound = _TRANSPLD_COMPOUND_RE.fullmatch(method)
    if transpld_compound:
        proposers.append(
            CompoundTransPLDProposer(
                tokenizer,
                max_draft_len=int(transpld_compound.group("w")),
                max_matching_ngram_size=int(transpld_compound.group("n")),
            )
        )
    transpld_full = _TRANSPLD_FULL_RE.fullmatch(method)
    if transpld_full:
        proposers.append(
            CursorTrackedTransPLDProposer(
                tokenizer,
                max_cursor_draft_len=int(transpld_full.group("c")),
                max_draft_len=int(transpld_full.group("w")),
                max_matching_ngram_size=int(transpld_full.group("n")),
                infer=True,
                compound=True,
            )
        )
    anchor_pld = _ANCHOR_PLD_RE.fullmatch(method)
    if anchor_pld:
        proposers.append(
            EditAnchorProposer(
                tokenizer,
                kind="edit_anchor_pld",
                max_draft_len=int(anchor_pld.group("a")),
                min_draft_tokens=int(anchor_pld.group("gate") or 1),
            )
        )
        proposers.append(
            RootedPLDProposer(
                max_draft_len=int(anchor_pld.group("w")),
                max_matching_ngram_size=int(anchor_pld.group("n")),
                min_matching_ngram_size=1,
            )
        )
    rewrite_anchor_pld = _REWRITE_ANCHOR_PLD_RE.fullmatch(method)
    if rewrite_anchor_pld:
        proposers.append(
            EditAnchorProposer(
                tokenizer,
                kind="rewrite_anchor_pld",
                max_draft_len=int(rewrite_anchor_pld.group("a")),
                min_draft_tokens=int(rewrite_anchor_pld.group("gate")),
                rewrite_enabled=True,
            )
        )
        proposers.append(
            RootedPLDProposer(
                max_draft_len=int(rewrite_anchor_pld.group("w")),
                max_matching_ngram_size=int(rewrite_anchor_pld.group("n")),
                min_matching_ngram_size=1,
            )
        )
    if method in {
        "vantage_id",
        "vantage_code_stack",
        "vantage_code_tail",
        "vantage_code_tail_w3",
        "vantage_code_tail_w4",
        "vantage_code_tail_context",
    }:
        proposers.append(IdentifierTrieProposer(tokenizer))
    if method in {
        "vantage_literal",
        "vantage_code_stack",
        "vantage_code_tail",
        "vantage_code_tail_w3",
        "vantage_code_tail_w4",
        "vantage_code_tail_context",
    }:
        proposers.append(LiteralCopyProposer(tokenizer))
    if method in {
        "vantage_suffix",
        "vantage_code_stack",
        "vantage_code_tail",
        "vantage_code_tail_w3",
        "vantage_code_tail_w4",
        "vantage_code_tail_context",
    }:
        proposers.append(LocalSuffixProposer())
    if method == "vantage_suffix_prompt":
        proposers.append(LocalSuffixProposer(pool="prompt"))
    if method == "vantage_suffix_generated":
        proposers.append(LocalSuffixProposer(pool="generated"))
    if method == "ngram_prompt_m4d5":
        proposers.append(NGramPromptLookupProposer(4, 5, pool="prompt"))
    if method == "ngram_local_m4d3":
        proposers.append(NGramPromptLookupProposer(4, 3, pool="local"))
    if method == "ngram_local_m4d5":
        proposers.append(NGramPromptLookupProposer(4, 5, pool="local"))
    if method == "ngram_local_m4d8":
        proposers.append(NGramPromptLookupProposer(4, 8, pool="local"))
    if method in {"vantage_multisuffix_chain", "vantage_repoedit_suffix"}:
        proposers.append(
            MultiSuffixProposer(
                kind="multisuffix_chain" if method == "vantage_multisuffix_chain" else "repoedit_suffix",
                top_k=1,
                tree=False,
            )
        )
    if method in {"vantage_multisuffix_tree", "vantage_multisuffix_tail_w4"}:
        proposers.append(
            MultiSuffixProposer(
                kind="multisuffix",
                top_k=4,
                max_tree_nodes=12,
                tree=True,
            )
        )
    if method in {"vantage_codespine", "vantage_editspine", "vantage_spinetail"}:
        proposers.append(
            CodeSpineProposer(
                tokenizer,
                kind=(
                    "edit_spine"
                    if method == "vantage_editspine"
                    else "spine_tail" if method == "vantage_spinetail" else "code_spine"
                ),
                edit_mode=method == "vantage_editspine",
            )
        )
    if method in {
        "vantage_edit_anchor",
        "vantage_edit_anchor_only",
        "vantage_edit_anchor_suffix",
        "vantage_edit_anchor_tail",
        "vantage_edit_symbol_tail",
    }:
        proposers.append(
            EditAnchorProposer(
                tokenizer,
                kind=(
                    "edit_anchor_only"
                    if method == "vantage_edit_anchor_only"
                    else "edit_anchor_suffix"
                    if method == "vantage_edit_anchor_suffix"
                    else "edit_anchor"
                    if method == "vantage_edit_anchor"
                    else "edit_anchor_tail" if method == "vantage_edit_anchor_tail" else "edit_symbol_anchor"
                ),
            )
        )
    if method in {
        "vantage_edit_anchor_suffix",
        "vantage_suffix_tail_w4",
        "vantage_edit_anchor_tail",
        "vantage_symbol_tree",
        "vantage_edit_symbol_tail",
    }:
        proposers.append(LocalSuffixProposer())
    if method in {"vantage_symbol_tree", "vantage_edit_symbol_tail"}:
        proposers.append(SymbolTreeProposer(tokenizer))
    if method == "vantage_alpha_tail_w4":
        proposers.append(LocalSuffixProposer())
    if method in {"vantage_alpha_v0", "alpha_idnorm"}:
        proposers.append(
            AlphaSuffixProposer(
                tokenizer,
                kind="alpha_id",
                enable_roles=False,
                normalize_literals=False,
                enable_substitution=method != "alpha_idnorm",
                scope_fill=False,
                filter_exact=True,
                stop_on_unmapped=True,
            )
        )
    if method in {"alpha_idlitnorm", "alpha_idlit_subst"}:
        proposers.append(
            AlphaSuffixProposer(
                tokenizer,
                kind="alpha_idlit",
                enable_roles=False,
                normalize_literals=True,
                enable_substitution=method == "alpha_idlit_subst",
                scope_fill=False,
                filter_exact=True,
                stop_on_unmapped=True,
            )
        )
    if method in {
        "vantage_alpha",
        "vantage_alpha_only",
        "vantage_alpha_tail_w4",
        "alpha_role_no_subst",
        "alpha_role_subst",
    }:
        proposers.append(
            AlphaSuffixProposer(
                tokenizer,
                kind="alpha_role",
                enable_roles=True,
                normalize_literals=True,
                enable_substitution=method != "alpha_role_no_subst",
                scope_fill=method != "alpha_role_no_subst",
                filter_exact=True,
                stop_on_unmapped=True,
            )
        )
    if method in {
        "vantage_macro_static",
        "vantage_code_stack",
        "vantage_code_tail",
        "vantage_code_tail_w3",
        "vantage_code_tail_w4",
        "vantage_code_tail_context",
    }:
        proposers.append(MacroChunkProposer(tokenizer, static_macro_chunks(benchmark)))
    return CheapProposerStack(proposers)


def _fallback_for_method(method: str) -> tuple[str, int]:
    if method == "vantage_code_tail":
        return "tail", 2
    if method == "vantage_code_tail_w3":
        return "tail", 3
    if method == "vantage_code_tail_w4":
        return "tail", 4
    if method == "vantage_code_tail_context":
        return "tail_context", 2
    if method == "vantage_alpha_tail_w4":
        return "tail", 4
    if method == "vantage_multisuffix_tail_w4":
        return "tail", 4
    if method == "vantage_spinetail":
        return "tail", 4
    if method in {"vantage_edit_anchor_only", "vantage_edit_anchor_suffix"}:
        return "root", 1
    if (
        _ROOTED_PLD_RE.fullmatch(method)
        or _ANCHOR_PLD_RE.fullmatch(method)
        or _REWRITE_ANCHOR_PLD_RE.fullmatch(method)
        or _REWRITE_PLD_RE.fullmatch(method)
        or _TRANSPLD_RE.fullmatch(method)
        or _DISPATCH_TRANSPLD_RE.fullmatch(method)
        or _PRECOMPUTED_TRANSPLD_RE.fullmatch(method)
        or _COMPETE_TRANSPLD_RE.fullmatch(method)
        or _FROZEN_TRANSPLD_RE.fullmatch(method)
        or _ROUTED_TRANSPLD_RE.fullmatch(method)
        or _ADOPT_SIMPLE_TRANSPLD_RE.fullmatch(method)
        or _TRANSPLD_CURSOR_RE.fullmatch(method)
        or _TRANSPLD_INFER_RE.fullmatch(method)
        or _TRANSPLD_COMPOUND_RE.fullmatch(method)
        or _TRANSPLD_FULL_RE.fullmatch(method)
    ):
        if method.startswith("vantage_anchor_pld_tail_"):
            return "tail", 4
        return "root", 1
    if method == "vantage_suffix_tail_w4":
        return "tail", 4
    if method in {"vantage_edit_anchor_tail", "vantage_symbol_tree", "vantage_edit_symbol_tail"}:
        return "tail", 4
    return "eagle_k2", 2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    p.add_argument("--eagle-checkpoint", required=True)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--dtype", default="float32")
    p.add_argument("--attn-impl", default="eager")
    p.add_argument(
        "--target-trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the target tokenizer/model.",
    )
    p.add_argument("--strict-determinism", action="store_true")
    p.add_argument("--output", default=None)
    p.add_argument(
        "--language",
        default="python",
        choices=[
            "python",
            "ts",
            "typescript",
            "mbpp",
            "repo_python",
            "repo_edit_python",
            "repo_edit_rename_python",
            "real_commit_python",
            "codeeditor_python",
            "codeeditor_switch_python",
            "codeeditor_translate",
            "codeeditor_translate_javacpp",
            "codeeditor_polish",
            "codeeditor_polish_cpp",
        ],
    )
    p.add_argument("--prompt-variant", default="full")
    p.add_argument("--tree-k", type=int, default=2, help="tree-tail spec chain length")
    p.add_argument("--tree-w", type=int, default=2, help="tree-tail spec leaf width")
    p.add_argument("--eagle2-total-tokens", type=int, default=26)
    p.add_argument("--eagle2-topk", type=int, default=10)
    p.add_argument("--eagle2-depth", type=int, default=6)
    p.add_argument("--skip-fixed-eagle", action="store_true")
    p.add_argument("--skip-tree", action="store_true")
    p.add_argument("--skip-asts", action="store_true")
    p.add_argument("--skip-eagle2", action="store_true")
    p.add_argument("--include-vantage", action="store_true")
    p.add_argument("--include-code-proposers", action="store_true")
    p.add_argument(
        "--code-methods",
        default="",
        help="Optional comma-separated subset of code proposer or BlazEdit methods to verify.",
    )
    p.add_argument("--assistant-model", default="Qwen/Qwen2.5-Coder-0.5B")
    p.add_argument("--blazedit-max-matching-ngram-size", type=int, default=10)
    p.add_argument("--blazedit-assistant-confidence-threshold", type=float, default=None)
    p.add_argument("--problem-jsonl", default="")
    p.add_argument("--skip-eagle-load", action="store_true")
    p.add_argument("--retrieval-index", default="")
    p.add_argument("--retrieval-draft-len", type=int, default=10)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("verify_eagle_lossless")

    if args.strict_determinism:
        _set_deterministic()

    log.info("loading target=%s dtype=%s attn=%s", args.target, args.dtype, args.attn_impl)
    target_tok, target = _load_model(
        args.target,
        dtype=args.dtype,
        attn_impl=args.attn_impl,
        trust_remote_code=args.target_trust_remote_code,
    )

    requested_methods = tuple(m.strip() for m in args.code_methods.split(",") if m.strip())
    code_only_root = bool(requested_methods) and all(
        _is_dynamic_code_proposer_method(m) or is_blazedit_method(m)
        for m in requested_methods
    )
    need_eagle = not (
        args.skip_eagle_load
        or (
            args.skip_fixed_eagle
            and args.skip_tree
            and args.skip_asts
            and args.skip_eagle2
            and not args.include_vantage
            and code_only_root
        )
    )
    eagle_head = None
    ckpt = {"step": -1}
    if need_eagle:
        log.info("loading EAGLE checkpoint: %s", args.eagle_checkpoint)
        eagle_head, eagle_cfg, ckpt = load_eagle_checkpoint(args.eagle_checkpoint, dtype=args.dtype)
        log.info("eagle: ckpt step=%d", ckpt.get("step", -1))
    else:
        log.info("skipping EAGLE checkpoint load")

    eos = [int(target_tok.eos_token_id)]
    if args.problem_jsonl:
        benchmark = "manifest"
        ast_lang = "python"
    elif args.language in ("ts", "typescript"):
        benchmark = "typescript"
        ast_lang = "typescript"
    elif args.language == "mbpp":
        benchmark = "mbpp"
        ast_lang = "python"
    elif args.language in {
        "repo_python",
        "repo_edit_python",
        "repo_edit_rename_python",
        "codeeditor_python",
        "codeeditor_switch_python",
        "codeeditor_translate",
        "codeeditor_translate_javacpp",
        "codeeditor_polish",
        "codeeditor_polish_cpp",
    }:
        benchmark = args.language
        ast_lang = "python"
    else:
        benchmark = "python"
        ast_lang = "python"
    log.info("benchmark: %s  ast_lang: %s", benchmark, ast_lang)
    if args.problem_jsonl:
        problems = load_problems_from_jsonl(args.problem_jsonl, n=args.n)
    else:
        problems = load_problems_for_language(
            language=benchmark,
            n=args.n,
            prompt_variant=args.prompt_variant,
            tokenizer=target_tok,
        )
    log.info("loaded %d problems", len(problems))

    results = []
    n_match_vf = 0
    n_match_va = 0
    n_match_vt = 0
    n_match_ve2 = 0
    n_match_vnh = 0
    code_methods = CODE_PROPOSER_METHODS
    if args.code_methods:
        requested = tuple(m.strip() for m in args.code_methods.split(",") if m.strip())
        unknown = sorted(
            m
            for m in requested
            if m not in CODE_PROPOSER_METHODS
            and not _is_dynamic_code_proposer_method(m)
            and not is_blazedit_method(m)
            and not is_vantage_mv_method(m)
        )
        if unknown:
            raise ValueError(f"unknown --code-methods: {unknown}")
        code_methods = requested
    n_match_code = {method: 0 for method in code_methods}

    blazedit_methods = [method for method in code_methods if is_blazedit_method(method)]
    vantage_mv_methods = [method for method in code_methods if is_vantage_mv_method(method)]
    assistant = None
    blazedit_configs = {
        method: parse_blazedit_method(
            method,
            assistant_model_name=args.assistant_model,
            confidence_threshold=args.blazedit_assistant_confidence_threshold,
            default_ngram_size=args.blazedit_max_matching_ngram_size,
        )
        for method in blazedit_methods
    }
    vantage_mv_configs = {
        method: parse_vantage_mv_method(method)
        for method in vantage_mv_methods
    }
    if any(cfg.mode != "pld" for cfg in blazedit_configs.values()):
        log.info(
            "loading BlazEdit assistant=%s dtype=%s attn=%s",
            args.assistant_model,
            args.dtype,
            args.attn_impl,
        )
        _, assistant = _load_model(
            args.assistant_model,
            dtype=args.dtype,
            attn_impl=args.attn_impl,
        )

    retrieval_index = None
    if args.include_vantage and args.retrieval_index:
        log.info("loading retrieval index: %s", args.retrieval_index)
        retrieval_index = RetrievalIndex.load(args.retrieval_index)

    for i, prob in enumerate(problems):
        log.info("[%d/%d] %s", i + 1, len(problems), prob.task_id)
        prompt_ids = target_tok(
            prob.prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids[0]

        v_res = vanilla_ar(
            prompt_ids=prompt_ids, target=target,
            max_new_tokens=args.max_new_tokens, eos_token_ids=eos,
        )
        v_new = v_res.output_token_ids[len(prompt_ids):]

        f_res = None
        f_new = []
        if not args.skip_fixed_eagle:
            f_res = fixed_eagle_spec(
                prompt_ids=prompt_ids, target=target, eagle_head=eagle_head,
                max_new_tokens=args.max_new_tokens, eos_token_ids=eos, k=args.k,
            )
            f_new = f_res.output_token_ids[len(prompt_ids):]

        a_res = None
        a_new = []
        if not args.skip_asts:
            ast_policy = ASTPolicy(language=ast_lang)
            a_res = asts_eagle_spec(
                prompt_ids=prompt_ids, target=target, eagle_head=eagle_head,
                max_new_tokens=args.max_new_tokens, eos_token_ids=eos,
                tokenizer=target_tok, ast_policy=ast_policy,
            )
            a_new = a_res.output_token_ids[len(prompt_ids):]

        t_res = None
        t_new = []
        if not args.skip_tree:
            t_res = tree_tail_eagle_spec(
                prompt_ids=prompt_ids, target=target, eagle_head=eagle_head,
                max_new_tokens=args.max_new_tokens, eos_token_ids=eos,
                k=args.tree_k, width=args.tree_w,
            )
            t_new = t_res.output_token_ids[len(prompt_ids):]

        e2_res = None
        e2_new = []
        if not args.skip_eagle2:
            e2_res = eagle2_speculative_ar(
                prompt_ids=prompt_ids, target=target, eagle_head=eagle_head,
                max_new_tokens=args.max_new_tokens, eos_token_ids=eos,
                method_name="eagle2",
                total_tokens=args.eagle2_total_tokens,
                topk_per_node=args.eagle2_topk,
                max_depth=args.eagle2_depth,
            )
            e2_new = e2_res.output_token_ids[len(prompt_ids):]

        nh_new = []
        if args.include_vantage:
            nh_policy = ASTPolicy(language=ast_lang)
            nh_res = vantage_router_spec(
                prompt_ids=prompt_ids,
                target=target,
                eagle_head=eagle_head,
                tokenizer=target_tok,
                ast_policy=nh_policy,
                max_new_tokens=args.max_new_tokens,
                eos_token_ids=eos,
                retrieval_index=retrieval_index,
                config=VantageRouterConfig(),
                max_retrieval_draft_len=args.retrieval_draft_len,
            )
            nh_new = nh_res.output_token_ids[len(prompt_ids):]

        code_new_by_method = {}
        if args.include_code_proposers:
            for code_method in code_methods:
                if is_blazedit_method(code_method):
                    cp_res = blazedit_speculative_ar(
                        prompt_ids=prompt_ids,
                        target=target,
                        assistant=assistant,
                        max_new_tokens=args.max_new_tokens,
                        eos_token_ids=eos,
                        config=blazedit_configs[code_method],
                        method_name=code_method,
                    )
                    code_new_by_method[code_method] = cp_res.output_token_ids[len(prompt_ids):]
                elif is_vantage_mv_method(code_method):
                    cp_res = vantage_mv_pld_speculative_ar(
                        prompt_ids=prompt_ids,
                        target=target,
                        tokenizer=target_tok,
                        max_new_tokens=args.max_new_tokens,
                        eos_token_ids=eos,
                        config=vantage_mv_configs[code_method],
                        method_name=code_method,
                        prompt_text=str(prob.prompt),
                        reference=str(getattr(prob, "reference", "") or ""),
                        metadata=getattr(prob, "metadata", None),
                    )
                    code_new_by_method[code_method] = cp_res.output_token_ids[len(prompt_ids):]
                elif (
                    (
                        _ROUTED_TRANSPLD_RE.fullmatch(code_method)
                        or _ADOPT_SIMPLE_TRANSPLD_RE.fullmatch(code_method)
                        or _DISPATCH_TRANSPLD_RE.fullmatch(code_method)
                        or _PRECOMPUTED_TRANSPLD_RE.fullmatch(code_method)
                        or _COMPETE_TRANSPLD_RE.fullmatch(code_method)
                        or _LAZY_TRANSPLD_RE.fullmatch(code_method)
                        or _MULTIVIEW_PLD_RE.fullmatch(code_method)
                        or _MULTIVIEW_TREE_RE.fullmatch(code_method)
                        or _FROZEN_TRANSPLD_RE.fullmatch(code_method)
                    )
                    and (
                        routed_exact_reason := _routed_transpld_exact_route_reason(
                            prob,
                            target_tok,
                        )
                    )
                    is not None
                ):
                    log.debug(
                        "routed TransPLD exact-PLD route for %s: %s",
                        prob.task_id,
                        routed_exact_reason,
                    )
                    cp_res = blazedit_speculative_ar(
                        prompt_ids=prompt_ids,
                        target=target,
                        assistant=None,
                        max_new_tokens=args.max_new_tokens,
                        eos_token_ids=eos,
                        config=_routed_transpld_pld_config(code_method, args),
                        method_name=code_method,
                    )
                    code_new_by_method[code_method] = cp_res.output_token_ids[len(prompt_ids):]
                else:
                    fallback, tail_width = _fallback_for_method(code_method)
                    cp_res = code_proposer_spec(
                        prompt_ids=prompt_ids,
                        target=target,
                        eagle_head=eagle_head,
                        tokenizer=target_tok,
                        ast_policy=ASTPolicy(language=ast_lang),
                        max_new_tokens=args.max_new_tokens,
                        eos_token_ids=eos,
                        proposer_stack=_stack_for_method(code_method, target_tok, benchmark),
                        fallback=fallback,
                        tail_width=tail_width,
                        context_tail_widths={"default": 2, "identifier": 3, "literal": 3, "margin": 3},
                        language=benchmark,
                        method_name=code_method,
                        reference=prob.reference,
                        metadata=prob.metadata,
                        prompt_text=prob.prompt,
                    )
                    code_new_by_method[code_method] = cp_res.output_token_ids[len(prompt_ids):]

        match_vt = args.skip_tree or (v_new == t_new)
        match_vf = args.skip_fixed_eagle or (v_new == f_new)
        match_va = args.skip_asts or (v_new == a_new)
        match_ve2 = v_new == e2_new
        match_ve2 = args.skip_eagle2 or (v_new == e2_new)
        match_vnh = (not args.include_vantage) or (v_new == nh_new)
        match_code = {
            method: (v_new == out) for method, out in code_new_by_method.items()
        }
        if match_vf:
            n_match_vf += 1
        if match_va:
            n_match_va += 1
        if match_vt:
            n_match_vt += 1
        if match_ve2:
            n_match_ve2 += 1
        if match_vnh:
            n_match_vnh += 1
        for method, ok in match_code.items():
            if ok:
                n_match_code[method] += 1

        first_diff_vf = None
        if not args.skip_fixed_eagle:
            first_diff_vf = next(
                (j for j in range(min(len(v_new), len(f_new))) if v_new[j] != f_new[j]),
                min(len(v_new), len(f_new)) if len(v_new) != len(f_new) else None,
            )
        first_diff_va = None
        if not args.skip_asts:
            first_diff_va = next(
                (j for j in range(min(len(v_new), len(a_new))) if v_new[j] != a_new[j]),
                min(len(v_new), len(a_new)) if len(v_new) != len(a_new) else None,
            )
        first_diff_vt = None
        if not args.skip_tree:
            first_diff_vt = next(
                (j for j in range(min(len(v_new), len(t_new))) if v_new[j] != t_new[j]),
                min(len(v_new), len(t_new)) if len(v_new) != len(t_new) else None,
            )
        first_diff_ve2 = None
        if not args.skip_eagle2:
            first_diff_ve2 = next(
                (j for j in range(min(len(v_new), len(e2_new))) if v_new[j] != e2_new[j]),
                min(len(v_new), len(e2_new)) if len(v_new) != len(e2_new) else None,
            )
        first_diff_vnh = None
        if args.include_vantage:
            first_diff_vnh = next(
                (j for j in range(min(len(v_new), len(nh_new))) if v_new[j] != nh_new[j]),
                min(len(v_new), len(nh_new)) if len(v_new) != len(nh_new) else None,
            )

        log.info(
            "  vanilla=%d  fixed_eagle_k%d=%d  asts_eagle=%d  tree_k%dw%d=%d  eagle2=%d  "
            "match_vf=%s match_va=%s match_vt=%s match_ve2=%s match_vnh=%s "
            "(first_diffs vf/va/vt/ve2/vnh=%s/%s/%s/%s/%s)",
            len(v_new), args.k, len(f_new) if not args.skip_fixed_eagle else -1,
            len(a_new) if not args.skip_asts else -1,
            args.tree_k, args.tree_w, len(t_new) if not args.skip_tree else -1,
            len(e2_new) if not args.skip_eagle2 else -1,
            match_vf, match_va, match_vt, match_ve2, match_vnh,
            first_diff_vf, first_diff_va, first_diff_vt, first_diff_ve2, first_diff_vnh,
        )

        # Also report acceptance rate of EAGLE
        f_acc = 0.0
        if f_res is not None:
            f_acc = sum(s.n_accepted_drafts for s in f_res.steps) / max(1, len(f_res.steps))
        a_acc = 0.0
        a_mean_k = 0.0
        if a_res is not None:
            a_acc = sum(s.n_accepted_drafts for s in a_res.steps) / max(1, len(a_res.steps))
            a_mean_k = sum(s.k for s in a_res.steps) / max(1, len(a_res.steps))
        e2_acc = 0.0
        e2_mean_k = 0.0
        if e2_res is not None:
            e2_acc = sum(s.n_accepted_drafts for s in e2_res.steps) / max(1, len(e2_res.steps))
            e2_mean_k = sum(s.k for s in e2_res.steps) / max(1, len(e2_res.steps))
        nh_acc = None
        nh_mean_k = None
        if args.include_vantage:
            nh_acc = sum(s.n_accepted_drafts for s in nh_res.steps) / max(1, len(nh_res.steps))
            nh_mean_k = sum(s.k for s in nh_res.steps) / max(1, len(nh_res.steps))
        log.info(
            "  fixed_eagle mean_acc=%.2f/%d  asts_eagle mean_acc=%.2f/%.1f  "
            "eagle2 mean_acc=%.2f/%.1f  vantage mean_acc=%s/%s",
            f_acc, args.k, a_acc, a_mean_k, e2_acc, e2_mean_k,
            f"{nh_acc:.2f}" if nh_acc is not None else "n/a",
            f"{nh_mean_k:.1f}" if nh_mean_k is not None else "n/a",
        )

        results.append({
            "task_id": prob.task_id,
            "vanilla_n_new": len(v_new),
            "fixed_n_new": len(f_new),
            "asts_n_new": len(a_new),
            "tree_n_new": len(t_new),
            "eagle2_n_new": len(e2_new),
            "vantage_n_new": len(nh_new) if args.include_vantage else None,
            "match_vf": match_vf,
            "match_va": match_va,
            "match_vt": match_vt,
            "match_ve2": match_ve2,
            "match_vnh": match_vnh,
            "match_code": match_code,
            "first_diff_vf": first_diff_vf,
            "first_diff_va": first_diff_va,
            "first_diff_vt": first_diff_vt,
            "first_diff_ve2": first_diff_ve2,
            "first_diff_vnh": first_diff_vnh,
            "fixed_mean_accepted": f_acc,
            "asts_mean_accepted": a_acc,
            "asts_mean_k": a_mean_k,
            "eagle2_mean_accepted": e2_acc,
            "eagle2_mean_k": e2_mean_k,
            "vantage_mean_accepted": nh_acc,
            "vantage_mean_k": nh_mean_k,
        })

    print()
    print("=" * 60)
    print("EAGLE LOSSLESS VERIFICATION")
    print("=" * 60)
    if args.skip_fixed_eagle:
        print(f"  vanilla == fixed_eagle_k{args.k}:        skipped")
    else:
        print(f"  vanilla == fixed_eagle_k{args.k}:        {n_match_vf}/{len(problems)}")
    print(f"  vanilla == asts_eagle:                {n_match_va}/{len(problems)}")
    if args.skip_tree:
        print(f"  vanilla == tree_eagle_k{args.tree_k}w{args.tree_w}:        skipped")
    else:
        print(f"  vanilla == tree_eagle_k{args.tree_k}w{args.tree_w}:        {n_match_vt}/{len(problems)}")
    print(f"  vanilla == eagle2 (t{args.eagle2_total_tokens}k{args.eagle2_topk}d{args.eagle2_depth}):  {n_match_ve2}/{len(problems)}")
    if args.include_vantage:
        print(f"  vanilla == vantage_full:            {n_match_vnh}/{len(problems)}")
    if args.include_code_proposers:
        for method in code_methods:
            print(f"  vanilla == {method:<28} {n_match_code[method]}/{len(problems)}")
    all_match = (
        n_match_vf == len(problems)
        and n_match_va == len(problems)
        and n_match_vt == len(problems)
        and n_match_ve2 == len(problems)
        and n_match_vnh == len(problems)
        and (
            not args.include_code_proposers
            or all(n == len(problems) for n in n_match_code.values())
        )
    )
    if all_match:
        print()
        print("  ✓ ALL OUTPUTS BYTE-IDENTICAL — lossless invariant holds")
    else:
        print()
        print("  ✗ DIVERGENCE — see per-task results")
    print("=" * 60)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "schema": "asts-spec/eagle_lossless/v1",
            "config": vars(args),
            "n_match_vf": n_match_vf,
            "n_match_va": n_match_va,
            "n_match_vt": n_match_vt,
            "n_match_ve2": n_match_ve2,
            "n_match_vnh": n_match_vnh,
            "n_match_code": n_match_code,
            "n_total": len(problems),
            "results": results,
        }, indent=2))

    sys.exit(0 if all_match else 1)


if __name__ == "__main__":
    main()
