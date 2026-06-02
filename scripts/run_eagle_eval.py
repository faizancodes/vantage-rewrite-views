"""HumanEval/MBPP evaluation with a trained EAGLE head as the draft.

Can compare:
  - vanilla AR (target only, baseline)
  - fixed_eagle_k* (EAGLE draft, fixed k)
  - asts_eagle (EAGLE draft, AST-gated variable k)
  - tree_eagle_k*w* (chain with top-W leaf branch)
  - retrieval_d* (suffix-array retrieval baseline)
  - vantage_full (live visibility router over chain/tail/retrieval/scope)
  - eagle2_t*k*d* (EAGLE-2 tree algorithm with the EAGLE-1 head)

Loads a checkpoint produced by `scripts/train_eagle.py` and runs the same
HumanEval harness as `run_prototype.py` but with the EAGLE draft path.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from asts.ast_policy import ASTPolicy, DATA_DERIVED_POLICY, DEFAULT_POLICY
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
    DispatchTransPLDProposer,
    LazyCompeteTransPLDProposer,
    PrecomputedTransPLDProposer,
    RewriteNormalizedPLDProposer,
    RoutedTransPLDProposer,
    SimpleAdoptionTransPLDProposer,
    RootedPLDProposer,
    SymbolTreeProposer,
    TransPLDProposer,
    TransPLDInferenceProposer,
    CompoundTransPLDProposer,
    CursorTrackedTransPLDProposer,
    _apply_word_map,
    _coerce_rewrite_pairs,
    _extract_reference_blocks,
    _rewrite_pairs,
    encode_no_special,
    static_macro_chunks,
)
from asts.decoder import vanilla_ar
from asts.eagle_decoder import asts_eagle_spec, fixed_eagle_spec, load_eagle_checkpoint
from asts.vantage_policy import decide_prompt_only_saferoute
from asts.tree_eagle_decoder import tree_tail_eagle_spec
from asts.eagle2_decoder import eagle2_speculative_ar
from asts.vantage_policy import VantageRouterConfig
from asts.vantage_router import vantage_router_spec
from asts.retrieval_draft import RetrievalIndex, retrieval_spec
from asts.lookahead_decoder import init_lookahead, lookahead_spec
from asts.humaneval import (
    load_problems_from_jsonl,
    load_problems_for_language,
    stop_texts_for_language,
    truncate_at_stop,
)
from asts.model_bench import _load_model
from asts.task_router import (
    extract_features as extract_task_router_features,
    load_router as load_task_router,
    should_use_transpld as task_router_should_use_transpld,
)


METHODS = ("vanilla", "eagle_k4", "eagle_k8", "asts_eagle")

ROUTER_METHOD_ALIASES = {
    "vantage",
    "router",
    "vantage_full",
    "vantage_conf",
    "vantage_ast_conf",
    "vantage_ast_conf_retrieval",
    "vantage_no_scope",
    "vantage_no_retrieval",
    "vantage_no_rolling",
    "vantage_v2",
    "vantage_v2_no_retrieval",
}

CODE_PROPOSER_METHODS = {
    "vantage_id",
    "vantage_literal",
    "vantage_suffix",
    "vantage_suffix_prompt",
    "vantage_suffix_generated",
    "vantage_macro_static",
    "vantage_macro_mined",
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
}

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
_TASK_ROUTER_RE = re.compile(
    r"vantage_task_router_(?P<mode>transpld|mvpld|mvtree)_w(?P<w>\d+)_n(?P<n>\d+)$"
)
_SELECTED_MV_FROZEN_RE = re.compile(r"vantage_selected_mv_frozen_w128_n10$")
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
        or _TASK_ROUTER_RE.fullmatch(method) is not None
        or _FROZEN_TRANSPLD_RE.fullmatch(method) is not None
        or _ROUTED_TRANSPLD_RE.fullmatch(method) is not None
        or _ADOPT_SIMPLE_TRANSPLD_RE.fullmatch(method) is not None
        or _TRANSPLD_CURSOR_RE.fullmatch(method) is not None
        or _TRANSPLD_INFER_RE.fullmatch(method) is not None
        or _TRANSPLD_COMPOUND_RE.fullmatch(method) is not None
        or _TRANSPLD_FULL_RE.fullmatch(method) is not None
    )


def _is_router_method(method: str) -> bool:
    return (
        method in ROUTER_METHOD_ALIASES
        or re.fullmatch(r"vantage(?:_v2)?_m\d+", method) is not None
    )


def _router_methods(methods_requested: list[str]) -> list[str]:
    out: list[str] = []
    for method in methods_requested:
        if method in {"vantage", "router"}:
            out.append("vantage_full")
        elif _is_router_method(method):
            out.append(method)
    seen: set[str] = set()
    deduped: list[str] = []
    for method in out:
        if method not in seen:
            seen.add(method)
            deduped.append(method)
    return deduped


def _code_proposer_methods(methods_requested: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for method in methods_requested:
        if (method in CODE_PROPOSER_METHODS or _is_dynamic_code_proposer_method(method)) and method not in seen:
            seen.add(method)
            out.append(method)
    return out


def _blazedit_methods(methods_requested: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for method in methods_requested:
        if is_blazedit_method(method) and method not in seen:
            seen.add(method)
            out.append(method)
    return out


def _vantage_mv_methods(methods_requested: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for method in methods_requested:
        if is_vantage_mv_method(method) and method not in seen:
            seen.add(method)
            out.append(method)
    return out


def _selected_mv_frozen_methods(methods_requested: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for method in methods_requested:
        if _SELECTED_MV_FROZEN_RE.fullmatch(method) and method not in seen:
            seen.add(method)
            out.append(method)
    return out


def _router_config_for_method(
    method: str,
    args: argparse.Namespace,
) -> VantageRouterConfig:
    use_ast_zone = not args.router_disable_ast_zone
    use_retrieval = not args.router_disable_retrieval
    use_scope = not args.router_disable_scope
    use_rolling = not args.router_disable_rolling
    default_to_tail = False
    retrieval_min_match = args.router_retrieval_min_match
    retrieval_high_match = args.router_retrieval_high_match

    if method == "vantage_conf":
        use_ast_zone = False
        use_retrieval = False
        use_scope = False
        use_rolling = False
    elif method == "vantage_ast_conf":
        use_ast_zone = True
        use_retrieval = False
        use_scope = False
        use_rolling = False
    elif method == "vantage_ast_conf_retrieval":
        use_ast_zone = True
        use_retrieval = True
        use_scope = False
        use_rolling = False
    elif method == "vantage_no_scope":
        use_scope = False
    elif method == "vantage_no_retrieval":
        use_retrieval = False
    elif method == "vantage_no_rolling":
        use_rolling = False
    elif method == "vantage_v2":
        default_to_tail = True
        use_scope = False
    elif method == "vantage_v2_no_retrieval":
        default_to_tail = True
        use_scope = False
        use_retrieval = False

    threshold_match = re.fullmatch(r"vantage(_v2)?_m(\d+)", method)
    if threshold_match:
        default_to_tail = bool(threshold_match.group(1))
        use_scope = False if default_to_tail else use_scope
        retrieval_min_match = int(threshold_match.group(2))
        retrieval_high_match = max(retrieval_min_match + 4, args.router_retrieval_high_match)

    return VantageRouterConfig(
        low_visibility_threshold=args.router_low_visibility,
        high_visibility_threshold=args.router_high_visibility,
        tail_max_margin=args.router_tail_margin,
        retrieval_min_match=retrieval_min_match,
        retrieval_high_match=retrieval_high_match,
        enable_long_chain=args.router_enable_long_chain,
        default_to_tail=default_to_tail,
        use_ast_zone=use_ast_zone,
        use_retrieval=use_retrieval,
        use_scope=use_scope,
        use_rolling=use_rolling,
    )


def _load_task_ids(path: str) -> set[str]:
    """Load task ids from a JSON list/dict or newline-delimited text file."""
    text = Path(path).read_text().strip()
    if not text:
        return set()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {line.strip() for line in text.splitlines() if line.strip()}
    if isinstance(data, list):
        return {str(x) for x in data}
    if isinstance(data, dict):
        ids = data.get("task_ids") or data.get("ids") or data.get("test") or data.get("train")
        if isinstance(ids, list):
            return {str(x) for x in ids}
    raise ValueError(f"unsupported task-id file format: {path}")


def _load_policy_table(policy_name: str, policy_json: str | None) -> dict[str, int]:
    if policy_json:
        policy = json.loads(Path(policy_json).read_text())
        if not isinstance(policy, dict) or "default" not in policy:
            raise ValueError("--policy-json must be a JSON object containing a 'default' key")
        return {str(k): int(v) for k, v in policy.items()}
    if policy_name in {"data-derived", "optimal"}:
        return DATA_DERIVED_POLICY
    return DEFAULT_POLICY


def _load_macro_chunks(path: str | None, benchmark: str) -> list[str]:
    if not path:
        return []
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict):
        for key in (benchmark, "typescript" if benchmark == "typescript" else "python", "chunks"):
            value = data.get(key)
            if isinstance(value, list):
                return [str(x) for x in value]
    raise ValueError(f"unsupported --macro-chunks-json format: {path}")


def _parse_context_tail_widths(raw: str) -> dict[str, int]:
    if not raw:
        return {"default": 2, "identifier": 3, "literal": 3, "margin": 3}
    out: dict[str, int] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if key == "margin_threshold":
            # Stored as int-like only in the decoder signature would be wrong;
            # keep the parser permissive by scaling thousandths if needed.
            out[key] = float(value)  # type: ignore[assignment]
        else:
            out[key] = int(value)
    return out


def _encode_prompt_ids(tokenizer, prompt: str, chat_template: str) -> torch.Tensor:
    """Encode a prompt, optionally wrapping it with the tokenizer chat template.

    The manifest prompt remains the raw user instruction for reference/map
    extraction and output JSON.  This function only controls the target-model
    input IDs.  Use `--chat-template user` for instruction-tuned chat models
    such as DeepSeek-Coder-Instruct.
    """
    mode = (chat_template or "none").strip().lower()
    if mode in {"", "none", "raw"}:
        return tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0]
    if mode in {"user", "chat", "single_user"}:
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError(
                "--chat-template user requested, but tokenizer has no chat_template"
            )
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        elif isinstance(encoded, dict):
            encoded = encoded["input_ids"]
        if torch.is_tensor(encoded):
            return encoded[0] if encoded.dim() == 2 else encoded
        return torch.tensor(encoded, dtype=torch.long)
    raise ValueError(f"unsupported --chat-template value: {chat_template!r}")


def _build_code_proposer_stack(
    method: str,
    tokenizer,
    benchmark: str,
    args: argparse.Namespace,
    mined_chunks: list[str],
) -> CheapProposerStack | None:
    proposers = []
    def _transpld_min(match) -> int:
        groups = match.groupdict()
        return int(groups.get("m") or args.transpld_min_match_len)

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
                transformed_min_matching_ngram_size=args.transpld_min_match_len,
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
                transformed_min_matching_ngram_size=args.transpld_min_match_len,
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
                transformed_min_matching_ngram_size=args.transpld_min_match_len,
                exact_strong_min_len=int(multiview_tree.group("strong")),
                trans_len_margin=int(multiview_tree.group("margin")),
            )
        )
    task_router = _TASK_ROUTER_RE.fullmatch(method)
    if task_router:
        mode = task_router.group("mode")
        w = int(task_router.group("w"))
        n = int(task_router.group("n"))
        if mode == "transpld":
            proposers.append(
                PrecomputedTransPLDProposer(
                    tokenizer,
                    max_draft_len=w,
                    max_matching_ngram_size=n,
                    transformed_min_matching_ngram_size=args.transpld_min_match_len,
                    compete_exact=True,
                    margin=0,
                )
            )
        elif mode == "mvpld":
            proposers.append(
                MultiViewPLDProposer(
                    tokenizer,
                    max_draft_len=w,
                    max_matching_ngram_size=n,
                    transformed_min_matching_ngram_size=args.transpld_min_match_len,
                    exact_strong_min_len=args.task_router_exact_strong_threshold,
                    trans_len_margin=args.task_router_trans_margin,
                )
            )
        elif mode == "mvtree":
            proposers.append(
                MultiViewTreePLDProposer(
                    tokenizer,
                    max_draft_len=w,
                    max_matching_ngram_size=n,
                    transformed_min_matching_ngram_size=args.transpld_min_match_len,
                    exact_strong_min_len=args.task_router_exact_strong_threshold,
                    trans_len_margin=args.task_router_trans_margin,
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
        gate = int(anchor_pld.group("gate") or 1)
        proposers.append(
            EditAnchorProposer(
                tokenizer,
                kind="edit_anchor_pld",
                max_draft_len=int(anchor_pld.group("a")),
                min_draft_tokens=gate,
                min_anchor_chars=args.edit_anchor_min_chars,
                require_edit_signal=args.edit_anchor_require_signal,
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
                min_anchor_chars=args.edit_anchor_min_chars,
                require_edit_signal=args.edit_anchor_require_signal,
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
    key_lengths = [
        int(x.strip())
        for x in str(getattr(args, "multisuffix_key_lengths", "3,4,5,6,8,12,16")).split(",")
        if x.strip()
    ]
    codespine_key_lengths = [
        int(x.strip())
        for x in str(getattr(args, "codespine_key_lengths", "4,5,6,8,12,16")).split(",")
        if x.strip()
    ]
    if method in {"vantage_id", "vantage_code_stack", "vantage_code_tail", "vantage_code_tail_w3", "vantage_code_tail_w4", "vantage_code_tail_context"}:
        proposers.append(
            IdentifierTrieProposer(
                tokenizer,
                max_draft_len=args.identifier_max_draft_len,
            )
        )
    if method in {"vantage_literal", "vantage_code_stack", "vantage_code_tail", "vantage_code_tail_w3", "vantage_code_tail_w4", "vantage_code_tail_context"}:
        proposers.append(
            LiteralCopyProposer(
                tokenizer,
                max_draft_len=args.literal_max_draft_len,
            )
        )
    if method in {"vantage_suffix", "vantage_code_stack", "vantage_code_tail", "vantage_code_tail_w3", "vantage_code_tail_w4", "vantage_code_tail_context"}:
        proposers.append(
            LocalSuffixProposer(
                min_match_len=args.local_suffix_min_match,
                max_query_len=args.local_suffix_max_query_len,
                max_draft_len=args.local_suffix_max_draft_len,
                pool="local",
            )
        )
    if method == "vantage_suffix_prompt":
        proposers.append(
            LocalSuffixProposer(
                min_match_len=args.local_suffix_min_match,
                max_query_len=args.local_suffix_max_query_len,
                max_draft_len=args.local_suffix_max_draft_len,
                pool="prompt",
            )
        )
    if method == "vantage_suffix_generated":
        proposers.append(
            LocalSuffixProposer(
                min_match_len=args.local_suffix_min_match,
                max_query_len=args.local_suffix_max_query_len,
                max_draft_len=args.local_suffix_max_draft_len,
                pool="generated",
            )
        )
    if method == "ngram_prompt_m4d5":
        proposers.append(
            NGramPromptLookupProposer(
                max_matching_ngram_size=4,
                max_draft_len=5,
                pool="prompt",
            )
        )
    if method == "ngram_local_m4d3":
        proposers.append(
            NGramPromptLookupProposer(
                max_matching_ngram_size=4,
                max_draft_len=3,
                pool="local",
            )
        )
    if method == "ngram_local_m4d5":
        proposers.append(
            NGramPromptLookupProposer(
                max_matching_ngram_size=4,
                max_draft_len=5,
                pool="local",
            )
        )
    if method == "ngram_local_m4d8":
        proposers.append(
            NGramPromptLookupProposer(
                max_matching_ngram_size=4,
                max_draft_len=8,
                pool="local",
            )
        )
    if method in {"vantage_multisuffix_chain", "vantage_repoedit_suffix"}:
        proposers.append(
            MultiSuffixProposer(
                kind="multisuffix_chain" if method == "vantage_multisuffix_chain" else "repoedit_suffix",
                key_lengths=key_lengths,
                top_k=1,
                max_draft_len=args.multisuffix_max_draft_len,
                max_tree_nodes=args.multisuffix_max_tree_nodes,
                tree=False,
                pool=args.multisuffix_pool,
            )
        )
    if method in {"vantage_multisuffix_tree", "vantage_multisuffix_tail_w4"}:
        proposers.append(
            MultiSuffixProposer(
                kind="multisuffix",
                key_lengths=key_lengths,
                top_k=args.multisuffix_top_k,
                max_draft_len=args.multisuffix_max_draft_len,
                max_tree_nodes=args.multisuffix_max_tree_nodes,
                tree=True,
                pool=args.multisuffix_pool,
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
                key_lengths=codespine_key_lengths,
                min_match_len=args.codespine_min_match_len,
                max_spine_len=args.codespine_max_spine_len,
                max_tree_nodes=args.codespine_max_tree_nodes,
                branch_budget=args.codespine_branch_budget,
                pool=args.codespine_pool,
                edit_mode=method == "vantage_editspine",
                allow_short_match=args.codespine_allow_short_match,
                enable_identifier_branches=args.codespine_enable_identifier_branches,
                enable_delimiter_branches=args.codespine_enable_delimiter_branches,
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
                max_draft_len=args.edit_anchor_max_draft_len,
                min_anchor_chars=args.edit_anchor_min_chars,
                require_edit_signal=args.edit_anchor_require_signal,
            )
        )
    if method in {
        "vantage_edit_anchor_suffix",
        "vantage_suffix_tail_w4",
        "vantage_edit_anchor_tail",
        "vantage_symbol_tree",
        "vantage_edit_symbol_tail",
    }:
        proposers.append(
            LocalSuffixProposer(
                min_match_len=args.local_suffix_min_match,
                max_query_len=args.local_suffix_max_query_len,
                max_draft_len=args.local_suffix_max_draft_len,
                pool="local",
            )
        )
    if method in {"vantage_symbol_tree", "vantage_edit_symbol_tail"}:
        proposers.append(
            SymbolTreeProposer(
                tokenizer,
                branch_budget=args.symbol_tree_branch_budget,
                max_tree_nodes=args.symbol_tree_max_tree_nodes,
                max_symbol_tokens=args.symbol_tree_max_symbol_tokens,
                min_prefix_chars=args.symbol_tree_min_prefix_chars,
            )
        )
    if method == "vantage_alpha_tail_w4":
        proposers.append(
            LocalSuffixProposer(
                min_match_len=args.local_suffix_min_match,
                max_query_len=args.local_suffix_max_query_len,
                max_draft_len=args.local_suffix_max_draft_len,
                pool="local",
            )
        )
    if method in {"vantage_alpha_v0", "alpha_idnorm"}:
        proposers.append(
            AlphaSuffixProposer(
                tokenizer,
                kind="alpha_id",
                min_match_len=args.alpha_min_match_len,
                max_query_len=args.alpha_max_query_len,
                max_draft_len=args.alpha_max_draft_len,
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
                min_match_len=args.alpha_min_match_len,
                max_query_len=args.alpha_max_query_len,
                max_draft_len=args.alpha_max_draft_len,
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
                min_match_len=args.alpha_min_match_len,
                max_query_len=args.alpha_max_query_len,
                max_draft_len=args.alpha_max_draft_len,
                enable_roles=args.alpha_enable_roles,
                normalize_literals=True,
                enable_substitution=method != "alpha_role_no_subst",
                scope_fill=args.alpha_scope_fill and method not in {"alpha_role_no_subst"},
                filter_exact=args.alpha_filter_exact or method == "vantage_alpha_only",
                stop_on_unmapped=args.alpha_stop_on_unmapped,
            )
        )
    if method in {"vantage_macro_static", "vantage_code_stack", "vantage_code_tail", "vantage_code_tail_w3", "vantage_code_tail_w4", "vantage_code_tail_context"}:
        proposers.append(
            MacroChunkProposer(
                tokenizer,
                static_macro_chunks(benchmark),
                kind="macro_static",
                max_draft_len=args.local_suffix_max_draft_len,
            )
        )
    if method == "vantage_macro_mined" or (
        mined_chunks
        and method in {
            "vantage_code_stack",
            "vantage_code_tail",
            "vantage_code_tail_w3",
            "vantage_code_tail_w4",
            "vantage_code_tail_context",
        }
    ):
        proposers.append(
            MacroChunkProposer(
                tokenizer,
                mined_chunks,
                kind="macro_mined",
                max_draft_len=args.local_suffix_max_draft_len,
            )
        )
    if not proposers:
        return None
    return CheapProposerStack(proposers)


def _nested_metadata(metadata: dict | None) -> dict:
    if not isinstance(metadata, dict):
        return {}
    nested = metadata.get("metadata")
    return nested if isinstance(nested, dict) else {}


def _problem_reference(prob) -> str:
    if getattr(prob, "reference", ""):
        return str(prob.reference)
    refs = _extract_reference_blocks(str(prob.prompt))
    return refs[0] if refs else ""


def _problem_rewrite_map(prob) -> dict[str, str]:
    return _rewrite_pairs(str(prob.prompt))


def _routed_transpld_exact_route_reason(prob, tokenizer) -> str | None:
    """Return a prompt-only reason to use the exact PLD decoder.

    SafeRoute reads only the prompt-visible rewrite map and the reference text.
    It ignores target text, gold output, benchmark labels, manifest-only fields,
    and arbitrary problem metadata.  When it returns a reason, run_eagle_eval
    calls the same BlazEdit PLD implementation used by the `blazedit_pld_*`
    baseline row, rather than the rooted code-proposer PLD.
    """
    reference = _problem_reference(prob)
    rewrite_map = _problem_rewrite_map(prob)
    transformed_reference = _apply_word_map(reference, rewrite_map) if rewrite_map else reference
    ref_tokens = encode_no_special(tokenizer, reference) if reference else []
    transformed_tokens = (
        encode_no_special(tokenizer, transformed_reference)
        if transformed_reference
        else []
    )
    decision = decide_prompt_only_saferoute(
        reference=reference,
        rewrite_map=rewrite_map,
        transformed_reference=transformed_reference,
        reference_tokens=ref_tokens,
        transformed_tokens=transformed_tokens,
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
        or _TASK_ROUTER_RE.fullmatch(method)
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


def _task_router_route_reason(prob, router: dict[str, Any] | None, args: argparse.Namespace) -> str | None:
    if router is None:
        return None
    features = extract_task_router_features(
        prompt=str(prob.prompt),
        reference=str(getattr(prob, "reference", "") or ""),
        metadata=getattr(prob, "metadata", None),
        output_budget=int(args.max_new_tokens),
    )
    return None if task_router_should_use_transpld(features, router) else "task_router_predicted_pld"


def _selector_float(meta: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(meta.get(key, default))
    except (TypeError, ValueError):
        return default


def _selector_rewrite_pairs(meta: dict[str, Any]) -> dict[str, str]:
    pairs = meta.get("rewrite_pairs") or {}
    return _coerce_rewrite_pairs(pairs)


def _selected_mv_frozen_choice(prob, selector: dict[str, Any] | None) -> str:
    """Choose PLD, stable MV, or frozen TransPLD from a train-only rule.

    The rule is intentionally prompt/metadata-only.  It mirrors
    ``scripts/analyze_real_commit_oracle_selector.py`` so the held-out GPU row
    uses exactly the selector fitted on train500, not an oracle over test
    timings.
    """
    if selector is None:
        raise ValueError("vantage_selected_mv_frozen_w128_n10 requires --task-router-json selector")
    raw_meta = getattr(prob, "metadata", None) or {}
    meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
    nested_meta = _nested_metadata(raw_meta)
    if nested_meta:
        meta.update(nested_meta)
    pairs = _selector_rewrite_pairs(meta)
    if not pairs:
        return "blazedit_pld_w128_n10"
    rewrite_occ = _selector_float(meta, "rewrite_occurrences_in_reference")
    if rewrite_occ <= 0:
        return "blazedit_pld_w128_n10"

    fit = _selector_float(meta, "transformed_reference_fit")
    dirty = _selector_float(meta, "dirty_vs_transformed_reference")
    density = _selector_float(meta, "rewrite_density_per_100_tokens")
    noisy = _selector_float(meta, "noisy_map_count")
    copy_ratio = _selector_float(meta, "copied_token_percentage")

    if (
        fit >= float(selector.get("frozen_fit", 1.01))
        and density >= float(selector.get("frozen_density", 999.0))
        and noisy <= float(selector.get("frozen_max_noisy", -1.0))
        and dirty <= float(selector.get("frozen_max_dirty", -1.0))
        and rewrite_occ >= float(selector.get("frozen_min_occ", 999.0))
    ):
        return str(selector.get("frozen_method", "vantage_frozen_transpld"))

    if (
        fit >= float(selector.get("mv_fit", 1.01))
        and density >= float(selector.get("mv_density", 999.0))
        and noisy <= float(selector.get("mv_max_noisy", -1.0))
        and dirty <= float(selector.get("mv_max_dirty", -1.0))
        and copy_ratio <= float(selector.get("mv_max_copy", -1.0))
    ):
        return str(selector.get("mv_method", "vantage_mv_pld_s96_x1_m16_t8_w128_n10"))

    return "blazedit_pld_w128_n10"


def _stamp_routed_exact_pld_steps(result, *, reason: str) -> None:
    for step in result.steps:
        step.proposal_route = "exact_pld"
        step.proposal_route_reason = reason
        step.proposal_backoff_active = False
        step.proposal_rewrite_hit_count = 0
        step.proposal_route_window_accept_rate = 0.0
        step.proposal_rewrite_zero_accept_streak = 0
        if step.proposal_kind == "blazedit_pld":
            step.proposal_match_kind = "routed_exact_pld"


def _code_proposer_fallback_and_width(
    method: str,
    default_fallback: str,
) -> tuple[str, int]:
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
    return default_fallback, 2


def _aggregate(per_step_records, n_new_tokens_per_method):
    by_method = {}
    by_node_type = {}
    # Use the dynamic key set so k values beyond the legacy METHODS tuple are included.
    method_names = list(n_new_tokens_per_method.keys()) or list(METHODS)
    for method in method_names:
        method_steps = [r for r in per_step_records if r["method"] == method]
        if not method_steps:
            continue
        def guaranteed_for_step(r):
            explicit = r.get("n_guaranteed_drafts")
            if explicit is not None:
                return explicit
            method_name = r.get("method", "")
            if (
                method_name.startswith("eagle_k")
                or method_name == "asts_eagle"
                or method_name.startswith("tree_eagle")
                or method_name.startswith("retrieval_")
                or method_name.startswith("vantage")
                or method_name.startswith("rewrite_pld_")
            ):
                return 1
            return 0

        def accepted_nonroot_for_step(r):
            explicit = r.get("n_accepted_nonroot_drafts")
            if explicit is not None:
                return explicit
            return max(0, r.get("n_accepted_drafts", 0) - guaranteed_for_step(r))

        total_us = sum(r["wall_us"] for r in method_steps)
        total_emitted = sum(r["n_emitted"] for r in method_steps)
        total_accepted = sum(r["n_accepted_drafts"] for r in method_steps)
        total_k = sum(r["k"] for r in method_steps)
        total_initial_prefill_us = sum(
            r.get("target_prefill_us", 0.0) for r in method_steps if r.get("step") == 0
        )
        decode_us = max(0.0, total_us - total_initial_prefill_us)
        total_nonroot_accepted = sum(accepted_nonroot_for_step(r) for r in method_steps)
        total_guaranteed = sum(guaranteed_for_step(r) for r in method_steps)
        total_nonroot_k = max(0, total_k - total_guaranteed)
        by_method[method] = {
            "n_steps": len(method_steps),
            "n_emitted_total": total_emitted,
            "wall_us_total": total_us,
            "tokens_per_sec": total_emitted / (total_us / 1e6) if total_us > 0 else 0,
            "decode_wall_us_total": decode_us,
            "decode_tokens_per_sec": total_emitted / (decode_us / 1e6) if decode_us > 0 else 0,
            "us_per_token": total_us / total_emitted if total_emitted > 0 else 0,
            "mean_accepted_drafts_per_step": total_accepted / len(method_steps),
            "mean_accepted_nonroot_drafts_per_step": (
                total_nonroot_accepted / len(method_steps)
            ),
            "nonroot_acceptance_rate": (
                total_nonroot_accepted / total_nonroot_k if total_nonroot_k else 0
            ),
            "mean_emitted_per_step": total_emitted / len(method_steps),
            "mean_k_requested": total_k / len(method_steps),
            "n_new_tokens_total": n_new_tokens_per_method.get(method, 0),
            "initial_prefill_us_total": total_initial_prefill_us,
            "target_prefill_us_total": sum(r.get("target_prefill_us", 0.0) for r in method_steps),
            "draft_us_total": sum(r.get("draft_us", 0.0) for r in method_steps),
            "verify_us_total": sum(r.get("verify_us", 0.0) for r in method_steps),
            "parse_us_total": sum(r.get("parse_us", 0.0) for r in method_steps),
            "proposal_us_total": sum(r.get("proposal_us", 0.0) for r in method_steps),
            "proposal_map_parse_us_total": sum(
                r.get("proposal_map_parse_us", 0.0) for r in method_steps
            ),
            "proposal_rewrite_apply_us_total": sum(
                r.get("proposal_rewrite_apply_us", 0.0) for r in method_steps
            ),
            "proposal_virtual_reference_tokenize_us_total": sum(
                r.get("proposal_virtual_reference_tokenize_us", 0.0)
                for r in method_steps
            ),
            "proposal_transpld_index_build_us_total": sum(
                r.get("proposal_transpld_index_build_us", 0.0) for r in method_steps
            ),
            "assistant_us_total": sum(r.get("assistant_us", 0.0) for r in method_steps),
            "assistant_prefill_us_total": sum(
                r.get("assistant_prefill_us", 0.0) for r in method_steps
            ),
            "assistant_pld_us_total": sum(
                r.get("assistant_pld_us", 0.0) for r in method_steps
            ),
            "assistant_verify_us_total": sum(
                r.get("assistant_verify_us", 0.0) for r in method_steps
            ),
            "blazedit_pld_proposed_total": sum(
                r.get("blazedit_pld_proposed", 0) or 0 for r in method_steps
            ),
            "blazedit_pld_accepted_total": sum(
                r.get("blazedit_pld_accepted", 0) or 0 for r in method_steps
            ),
            "target_draft_tokens_total": sum(
                r.get("target_draft_tokens", 0) or 0 for r in method_steps
            ),
            "target_accepted_nonroot_total": sum(
                r.get("target_accepted_nonroot", 0) or 0 for r in method_steps
            ),
            "assistant_cache_catchup_tokens_total": sum(
                r.get("assistant_cache_catchup_tokens", 0) or 0 for r in method_steps
            ),
            "verify_staged_chunks_total": sum(
                r.get("verify_staged_chunks", 0) or 0 for r in method_steps
            ),
            "verify_staged_draft_tokens_total": sum(
                r.get("verify_staged_draft_tokens", 0) or 0 for r in method_steps
            ),
            "verify_staged_saved_tokens_total": sum(
                r.get("verify_staged_saved_tokens", 0) or 0 for r in method_steps
            ),
            "proposal_neural_draft_tokens_total": sum(
                r.get("proposal_neural_draft_tokens", 0) or 0 for r in method_steps
            ),
            "proposal_neural_draft_us_total": sum(
                r.get("proposal_neural_draft_us", 0.0) or 0.0 for r in method_steps
            ),
            "mtp_trigger_count": sum(1 for r in method_steps if r.get("mtp_triggered") is True),
            "mtp_steps": sum(1 for r in method_steps if r.get("mtp_triggered") is not None),
            "pld_steps": len(method_steps),
            "mtp_head_compute_us_total": sum(
                r.get("mtp_head_compute_us", 0.0) or 0.0 for r in method_steps
            ),
            "mtp_verify_extra_us_total": sum(
                r.get("mtp_verify_extra_us", 0.0) or 0.0 for r in method_steps
            ),
            "mtp_total_overhead_us_total": sum(
                r.get("mtp_total_overhead_us", 0.0) or 0.0 for r in method_steps
            ),
            "mtp_token0_reject_count": sum(
                1 for r in method_steps if r.get("mtp_token0_rejected") is True
            ),
            "mtp_actual_extra_progress_sum": sum(
                r.get("mtp_actual_extra_progress", 0) or 0 for r in method_steps
            ),
            "mtp_extra_accepted_drafts_sum": sum(
                r.get("mtp_extra_accepted_drafts", 0) or 0 for r in method_steps
            ),
            "mtp_decode_steps_saved_estimate": sum(
                r.get("mtp_actual_extra_progress", 0) or 0 for r in method_steps
            ),
            "mtp_queue_predictions_created": sum(
                1 for r in method_steps if r.get("mtp_queue_prediction_created") is True
            ),
            "mtp_queue_predictions_used": sum(
                1 for r in method_steps if r.get("mtp_queue_prediction_used") is True
            ),
            "mtp_queue_predictions_dropped_pld_strong": sum(
                1 for r in method_steps if r.get("mtp_queue_dropped_pld_strong") is True
            ),
            "mtp_queue_predictions_dropped_position_mismatch": sum(
                1
                for r in method_steps
                if r.get("mtp_queue_dropped_position_mismatch") is True
            ),
            "mtp_queue_predictions_expired": sum(
                1 for r in method_steps if r.get("mtp_queue_expired") is True
            ),
            "mtp_used_count": sum(
                1 for r in method_steps if r.get("mtp_queue_prediction_used") is True
            ),
            "mtp_used_token0_reject_count": sum(
                1 for r in method_steps if r.get("mtp_used_token0_rejected") is True
            ),
            "mtp_extra_verify_calls": sum(
                r.get("mtp_extra_verify_calls", 0) or 0 for r in method_steps
            ),
            "mtp_normal_verify_reuse_count": sum(
                1 for r in method_steps if r.get("mtp_normal_verify_reuse") is True
            ),
            "pld_cap_steps": sum(
                1 for r in method_steps if r.get("pld_cap_variant") is not None
            ),
            "pld_cap_trigger_count": sum(
                1 for r in method_steps if r.get("pld_cap_triggered") is True
            ),
            "pld_cap_router_predicted_weak_count": sum(
                1 for r in method_steps if r.get("pld_cap_router_predicted_weak") is True
            ),
            "pld_cap_raw_draft_len_total": sum(
                r.get("pld_cap_raw_draft_len", 0) or 0 for r in method_steps
            ),
            "pld_cap_capped_draft_len_total": sum(
                r.get("pld_cap_capped_draft_len", 0) or 0 for r in method_steps
            ),
            "pld_cap_wasted_verified_tokens_total": sum(
                r.get("pld_cap_wasted_verified_tokens", 0) or 0 for r in method_steps
            ),
            "pld_cap_router_us_total": sum(
                r.get("pld_cap_router_us", 0.0) or 0.0 for r in method_steps
            ),
            "pld_used_count": sum(
                1
                for r in method_steps
                if r.get("pld_lookahead_pld_used") is True
                or r.get("proposal_kind") == "blazedit_pld"
            ),
            "pld_skipped_due_to_weak_router": sum(
                1 for r in method_steps if r.get("pld_lookahead_skipped_pld") is True
            ),
            "pld_tok0_reject_count": sum(
                1
                for r in method_steps
                if r.get("proposal_kind") == "blazedit_pld"
                and r.get("rejected") is True
                and (r.get("n_accepted_drafts", 0) or 0) == 0
            ),
            "router_calls": sum(
                1 for r in method_steps if r.get("pld_lookahead_router") is not None
            ),
            "router_predicted_weak": sum(
                1
                for r in method_steps
                if r.get("pld_lookahead_predicted_weak") is True
            ),
            "router_predicted_strong": sum(
                1
                for r in method_steps
                if r.get("pld_lookahead_predicted_weak") is False
            ),
            "router_prob_sum": sum(
                r.get("pld_lookahead_router_prob", 0.0) or 0.0 for r in method_steps
            ),
            "lookahead_calls": sum(
                1 for r in method_steps if r.get("lookahead_triggered") is True
            ),
            "lookahead_candidate_len_total": sum(
                r.get("lookahead_candidate_len", 0) or 0 for r in method_steps
            ),
            "lookahead_accepted_len_total": sum(
                r.get("lookahead_accepted_len", 0) or 0 for r in method_steps
            ),
            "lookahead_tok0_reject_count": sum(
                1 for r in method_steps if r.get("lookahead_tok0_reject") is True
            ),
            "lookahead_forward_calls_total": sum(
                r.get("lookahead_forward_calls", 0) or 0 for r in method_steps
            ),
            "lookahead_us_total": sum(
                r.get("lookahead_us", 0.0) or 0.0 for r in method_steps
            ),
            "lookahead_forward_us_total": sum(
                r.get("lookahead_forward_us", 0.0) or 0.0 for r in method_steps
            ),
            "lookahead_candidate_build_us_total": sum(
                r.get("lookahead_candidate_build_us", 0.0) or 0.0 for r in method_steps
            ),
            "lookahead_verify_us_total": sum(
                r.get("lookahead_verify_us", 0.0) or 0.0 for r in method_steps
            ),
            "pld_would_have_draft_len_total": sum(
                r.get("pld_would_have_draft_len", 0) or 0 for r in method_steps
            ),
            "mtp_accepted_prefix_0_count": sum(
                1 for r in method_steps if r.get("mtp_accepted_prefix_len") == 0
            ),
            "mtp_accepted_prefix_1_count": sum(
                1 for r in method_steps if r.get("mtp_accepted_prefix_len") == 1
            ),
            "mtp_accepted_prefix_2_count": sum(
                1 for r in method_steps if r.get("mtp_accepted_prefix_len") == 2
            ),
            "mtp_accepted_prefix_3_count": sum(
                1 for r in method_steps if r.get("mtp_accepted_prefix_len") == 3
            ),
            "mtp_accepted_prefix_4_count": sum(
                1 for r in method_steps if r.get("mtp_accepted_prefix_len") == 4
            ),
            "pld_exact_hits_total": sum(
                1 for r in method_steps if r.get("pld_exact_hit") is True
            ),
            "pld_exact_misses_total": sum(
                1 for r in method_steps if r.get("pld_exact_hit") is False
            ),
            "pld_variant_triggers_total": sum(
                1 for r in method_steps if r.get("pld_variant_triggered") is True
            ),
            "pld_variant_overhead_us_total": sum(
                r.get("pld_variant_overhead_us", 0.0) or 0.0 for r in method_steps
            ),
            "pld_candidate_accepted_len_total": sum(
                r.get("pld_candidate_accepted_len", 0) or 0 for r in method_steps
            ),
            "pld_token01_rejections_total": sum(
                1 for r in method_steps if r.get("pld_token01_rejection") is True
            ),
            "pld_trigger_token01_rejections_total": sum(
                1
                for r in method_steps
                if r.get("pld_variant_triggered") is True
                and r.get("pld_token01_rejection") is True
            ),
            "pld_delta_patch_count_total": sum(
                r.get("pld_delta_patch_count", 0) or 0 for r in method_steps
            ),
            "pld_delta_patch_accepted_total": sum(
                1 for r in method_steps if r.get("pld_delta_patch_accepted") is True
            ),
            "pld_delta_patch_accept_tail_total": sum(
                r.get("pld_delta_patch_accept_tail", 0) or 0 for r in method_steps
            ),
            "pld_fuzzy_candidate_count_total": sum(
                r.get("pld_fuzzy_candidate_count", 0) or 0 for r in method_steps
            ),
            "pld_rerank_trigger_count": sum(
                1 for r in method_steps if r.get("pld_rerank_triggered") is True
            ),
            "pld_rerank_ambiguous_steps": sum(
                1 for r in method_steps if r.get("pld_rerank_ambiguous") is True
            ),
            "pld_rerank_selected_rank_0": sum(
                1 for r in method_steps if r.get("pld_rerank_selected_rank") == 0
            ),
            "pld_rerank_selected_rank_1": sum(
                1 for r in method_steps if r.get("pld_rerank_selected_rank") == 1
            ),
            "pld_rerank_selected_rank_2": sum(
                1 for r in method_steps if r.get("pld_rerank_selected_rank") == 2
            ),
            "pld_rerank_selected_rank_3": sum(
                1 for r in method_steps if r.get("pld_rerank_selected_rank") == 3
            ),
            "pld_rerank_accepted_len_sum": sum(
                r.get("pld_candidate_accepted_len", 0) or 0
                for r in method_steps
                if r.get("pld_rerank_triggered") is True
            ),
            "pld_rerank_tok0_1_reject_count": sum(
                1
                for r in method_steps
                if r.get("pld_rerank_triggered") is True
                and r.get("pld_token01_rejection") is True
            ),
            "pld_rerank_overhead_us_total": sum(
                r.get("pld_rerank_overhead_us", 0.0) or 0.0 for r in method_steps
            ),
            "pld_rerank_fallback_count": sum(
                1 for r in method_steps if r.get("pld_rerank_fallback") is True
            ),
            "pld_opp_traced_steps_total": sum(
                1 for r in method_steps if r.get("pld_opp_trace") is True
            ),
            "pld_opp_weak_steps_total": sum(
                1
                for r in method_steps
                if r.get("pld_opp_trace") is True
                and (r.get("pld_opp_accepted_len", r.get("n_accepted_drafts", 0)) or 0) <= 4
            ),
            "pld_opp_weak_wall_us_total": sum(
                r.get("wall_us", 0.0) or 0.0
                for r in method_steps
                if r.get("pld_opp_trace") is True
                and (r.get("pld_opp_accepted_len", r.get("n_accepted_drafts", 0)) or 0) <= 4
            ),
            "hit_max_new_tokens_steps": sum(1 for r in method_steps if r.get("hit_max_new_tokens")),
        }
        if by_method[method]["pld_opp_traced_steps_total"]:
            traced = by_method[method]["pld_opp_traced_steps_total"]
            by_method[method]["pld_opp_weak_step_rate"] = (
                by_method[method]["pld_opp_weak_steps_total"] / traced
            )
            by_method[method]["pld_opp_weak_runtime_fraction"] = (
                by_method[method]["pld_opp_weak_wall_us_total"] / total_us
                if total_us > 0
                else 0.0
            )
        if any(r.get("mtp_triggered") is not None for r in method_steps):
            triggers = by_method[method]["mtp_trigger_count"]
            by_method[method]["mtp_head_compute_us_per_trigger"] = (
                by_method[method]["mtp_head_compute_us_total"] / max(1, triggers)
            )
            by_method[method]["mtp_verify_extra_us_per_trigger"] = (
                by_method[method]["mtp_verify_extra_us_total"] / max(1, triggers)
            )
            by_method[method]["mtp_total_overhead_us_per_trigger"] = (
                by_method[method]["mtp_total_overhead_us_total"] / max(1, triggers)
            )
            by_method[method]["runtime_overhead_per_trigger"] = by_method[method][
                "mtp_total_overhead_us_per_trigger"
            ]
            by_method[method]["mtp_token0_reject_rate"] = (
                by_method[method]["mtp_token0_reject_count"] / max(1, triggers)
            )
            by_method[method]["mtp_avg_extra_tokens_per_trigger"] = (
                by_method[method]["mtp_actual_extra_progress_sum"] / max(1, triggers)
            )
            by_method[method]["mtp_avg_extra_accepted_drafts_per_trigger"] = (
                by_method[method]["mtp_extra_accepted_drafts_sum"] / max(1, triggers)
            )
            used = by_method[method]["mtp_used_count"]
            by_method[method]["mtp_avg_extra_progress_per_used"] = (
                by_method[method]["mtp_actual_extra_progress_sum"] / max(1, used)
            )
            by_method[method]["mtp_avg_extra_accepted_tokens_per_used"] = (
                by_method[method]["mtp_extra_accepted_drafts_sum"] / max(1, used)
            )
            by_method[method]["mtp_used_token0_reject_rate"] = (
                by_method[method]["mtp_used_token0_reject_count"] / max(1, used)
            )
            by_method[method]["mtp_extra_verify_ms_per_trigger"] = (
                by_method[method]["mtp_verify_extra_us_per_trigger"] / 1000.0
            )
            by_method[method]["mtp_head_compute_ms_per_trigger"] = (
                by_method[method]["mtp_head_compute_us_per_trigger"] / 1000.0
            )
            by_method[method]["mtp_total_overhead_ms_per_trigger"] = (
                by_method[method]["mtp_total_overhead_us_per_trigger"] / 1000.0
            )
        if any(r.get("pld_cap_variant") is not None for r in method_steps):
            cap_steps = max(1, by_method[method]["pld_cap_steps"])
            cap_triggers = max(1, by_method[method]["pld_cap_trigger_count"])
            by_method[method]["pld_cap_trigger_rate"] = (
                by_method[method]["pld_cap_trigger_count"] / cap_steps
            )
            by_method[method]["pld_cap_router_predicted_weak_rate"] = (
                by_method[method]["pld_cap_router_predicted_weak_count"] / cap_steps
            )
            by_method[method]["pld_cap_raw_draft_len_mean"] = (
                by_method[method]["pld_cap_raw_draft_len_total"] / cap_steps
            )
            by_method[method]["pld_cap_capped_draft_len_mean"] = (
                by_method[method]["pld_cap_capped_draft_len_total"] / cap_steps
            )
            by_method[method]["pld_cap_wasted_verified_tokens_mean"] = (
                by_method[method]["pld_cap_wasted_verified_tokens_total"] / cap_steps
            )
            by_method[method]["pld_cap_router_us_per_step"] = (
                by_method[method]["pld_cap_router_us_total"] / cap_steps
            )
            by_method[method]["pld_cap_router_us_per_trigger"] = (
                by_method[method]["pld_cap_router_us_total"] / cap_triggers
            )
        if any(r.get("lookahead_triggered") is not None for r in method_steps):
            calls = max(1, by_method[method]["lookahead_calls"])
            router_calls = max(1, by_method[method]["router_calls"])
            by_method[method]["lookahead_candidate_len_mean"] = (
                by_method[method]["lookahead_candidate_len_total"] / calls
            )
            by_method[method]["lookahead_accepted_len_mean"] = (
                by_method[method]["lookahead_accepted_len_total"] / calls
            )
            by_method[method]["lookahead_tok0_reject_rate"] = (
                by_method[method]["lookahead_tok0_reject_count"] / calls
            )
            by_method[method]["lookahead_forward_calls_per_call"] = (
                by_method[method]["lookahead_forward_calls_total"] / calls
            )
            by_method[method]["lookahead_ms_per_call"] = (
                by_method[method]["lookahead_us_total"] / calls / 1000.0
            )
            by_method[method]["lookahead_ms_per_call_mean"] = by_method[method][
                "lookahead_ms_per_call"
            ]
            by_method[method]["lookahead_forward_ms_per_call_mean"] = (
                by_method[method]["lookahead_forward_us_total"] / calls / 1000.0
            )
            by_method[method]["lookahead_verify_ms_per_call_mean"] = (
                by_method[method]["lookahead_verify_us_total"] / calls / 1000.0
            )
            by_method[method]["lookahead_candidate_build_ms_mean"] = (
                by_method[method]["lookahead_candidate_build_us_total"] / calls / 1000.0
            )
            by_method[method]["lookahead_ms_per_accepted_token"] = (
                by_method[method]["lookahead_us_total"]
                / max(1, by_method[method]["lookahead_accepted_len_total"])
                / 1000.0
            )
            by_method[method]["lookahead_accepted_per_forward"] = (
                by_method[method]["lookahead_accepted_len_total"]
                / max(1, by_method[method]["lookahead_forward_calls_total"])
            )
            by_method[method]["total_model_forward_calls"] = (
                by_method[method]["n_steps"]
                + by_method[method]["lookahead_forward_calls_total"]
            )
            by_method[method]["total_model_forward_ms"] = (
                by_method[method]["verify_us_total"]
                + by_method[method]["lookahead_forward_us_total"]
            ) / 1000.0
            by_method[method]["pld_would_have_draft_len_mean"] = (
                by_method[method]["pld_would_have_draft_len_total"] / router_calls
            )
            by_method[method]["router_predicted_weak_rate"] = (
                by_method[method]["router_predicted_weak"] / router_calls
            )
            by_method[method]["router_prob_mean"] = (
                by_method[method]["router_prob_sum"] / router_calls
            )
        if any(r.get("pld_variant") for r in method_steps):
            triggers = by_method[method]["pld_variant_triggers_total"]
            patch_count = by_method[method]["pld_delta_patch_count_total"]
            by_method[method]["pld_exact_hit_rate"] = (
                by_method[method]["pld_exact_hits_total"] / len(method_steps)
            )
            by_method[method]["pld_variant_trigger_rate"] = triggers / len(method_steps)
            by_method[method]["pld_token01_rejection_rate"] = (
                by_method[method]["pld_token01_rejections_total"] / len(method_steps)
            )
            by_method[method]["pld_trigger_token01_rejection_rate"] = (
                by_method[method]["pld_trigger_token01_rejections_total"] / max(1, triggers)
            )
            by_method[method]["pld_variant_overhead_us_per_step"] = (
                by_method[method]["pld_variant_overhead_us_total"] / len(method_steps)
            )
            by_method[method]["pld_candidate_accepted_len_mean"] = (
                by_method[method]["pld_candidate_accepted_len_total"] / len(method_steps)
            )
            by_method[method]["pld_delta_reuse_rate"] = (
                by_method[method]["pld_delta_patch_accepted_total"] / max(1, patch_count)
            )
            by_method[method]["pld_delta_patch_accept_tail_mean"] = (
                by_method[method]["pld_delta_patch_accept_tail_total"] / max(1, patch_count)
            )
            by_method[method]["pld_fuzzy_hit_rate_among_exact_misses"] = (
                triggers / max(1, by_method[method]["pld_exact_misses_total"])
            )
        if any(r.get("pld_rerank_ambiguous") is not None for r in method_steps):
            rerank_triggers = by_method[method]["pld_rerank_trigger_count"]
            score_margins = [
                float(r.get("pld_rerank_score_margin"))
                for r in method_steps
                if r.get("pld_rerank_score_margin") is not None
            ]
            score_margins_sorted = sorted(score_margins)

            def _quantile(xs, q):
                if not xs:
                    return 0.0
                idx = min(len(xs) - 1, max(0, int(round((len(xs) - 1) * q))))
                return xs[idx]

            by_method[method]["pld_rerank_accepted_len_mean"] = (
                by_method[method]["pld_rerank_accepted_len_sum"] / max(1, rerank_triggers)
            )
            by_method[method]["pld_rerank_overhead_us_per_step"] = (
                by_method[method]["pld_rerank_overhead_us_total"] / len(method_steps)
            )
            by_method[method]["pld_rerank_tok0_1_reject_rate"] = (
                by_method[method]["pld_rerank_tok0_1_reject_count"]
                / max(1, rerank_triggers)
            )
            by_method[method]["pld_rerank_trigger_rate"] = (
                rerank_triggers / len(method_steps)
            )
            by_method[method]["pld_rerank_ambiguous_rate"] = (
                by_method[method]["pld_rerank_ambiguous_steps"] / len(method_steps)
            )
            by_method[method]["pld_rerank_score_margin_mean"] = (
                sum(score_margins) / len(score_margins) if score_margins else 0.0
            )
            by_method[method]["pld_rerank_score_margin_p50"] = _quantile(
                score_margins_sorted, 0.50
            )
            by_method[method]["pld_rerank_score_margin_p90"] = _quantile(
                score_margins_sorted, 0.90
            )
            by_method[method]["pld_rerank_selected_is_baseline_count"] = sum(
                1
                for r in method_steps
                if r.get("pld_rerank_triggered") is True
                and r.get("pld_rerank_selected_is_baseline") is True
            )
            by_method[method]["pld_rerank_selected_nonbaseline_count"] = sum(
                1
                for r in method_steps
                if r.get("pld_rerank_triggered") is True
                and r.get("pld_rerank_selected_is_baseline") is False
            )
            by_method[method]["pld_rerank_baseline_score_missing_count"] = sum(
                1
                for r in method_steps
                if r.get("pld_rerank_baseline_score_missing") is True
            )
            by_method[method]["pld_rerank_debug_trace_rows"] = sum(
                1 for r in method_steps if r.get("pld_rerank_debug_features") is not None
            )
        micro_runs = [
            r.get("blazedit_micro_runs")
            for r in method_steps
            if r.get("blazedit_micro_runs") is not None
        ]
        if micro_runs:
            by_method[method]["mean_blazedit_micro_runs"] = sum(micro_runs) / len(micro_runs)
        visibility_values = [
            r.get("visibility") for r in method_steps if r.get("visibility") is not None
        ]
        if visibility_values:
            by_method[method]["mean_visibility"] = sum(visibility_values) / len(visibility_values)
        frontier_values = [
            r.get("frontier_depth") for r in method_steps if r.get("frontier_depth") is not None
        ]
        if frontier_values:
            by_method[method]["mean_frontier_depth"] = sum(frontier_values) / len(frontier_values)
        strategy_steps = [r for r in method_steps if r.get("strategy")]
        if strategy_steps:
            by_strategy = {}
            for r in strategy_steps:
                strategy = r["strategy"]
                d = by_strategy.setdefault(
                    strategy,
                    {"n": 0, "sum_k": 0, "sum_accepted": 0, "sum_wall_us": 0.0},
                )
                d["n"] += 1
                d["sum_k"] += r["k"]
                d["sum_accepted"] += r["n_accepted_drafts"]
                d["sum_wall_us"] += r["wall_us"]
            for d in by_strategy.values():
                n = d["n"]
                d["mean_k"] = d["sum_k"] / n
                d["mean_accepted"] = d["sum_accepted"] / n
                d["acceptance_rate"] = d["mean_accepted"] / d["mean_k"] if d["mean_k"] else 0
                d["mean_wall_us"] = d["sum_wall_us"] / n
            by_method[method]["by_strategy"] = by_strategy
        proposal_steps = [r for r in method_steps if r.get("proposal_kind")]
        if proposal_steps:
            by_proposal = {}
            for r in proposal_steps:
                kind = r["proposal_kind"]
                d = by_proposal.setdefault(
                    kind,
                    {
                        "n": 0,
                        "sum_k": 0,
                        "sum_proposal_tokens": 0,
                        "sum_match_len": 0,
                        "sum_accepted_nonroot": 0,
                        "sum_wall_us": 0.0,
                        "sum_proposal_us": 0.0,
                        "sum_map_parse_us": 0.0,
                        "sum_rewrite_apply_us": 0.0,
                        "sum_virtual_reference_tokenize_us": 0.0,
                        "sum_transpld_index_build_us": 0.0,
                    },
                )
                d["n"] += 1
                d["sum_k"] += r.get("k", 0)
                d["sum_proposal_tokens"] += r.get("proposal_tokens", 0) or 0
                d["sum_match_len"] += r.get("proposal_match_len", 0) or 0
                d["sum_accepted_nonroot"] += r.get("n_accepted_nonroot_drafts", 0) or 0
                d["sum_wall_us"] += r.get("wall_us", 0.0)
                d["sum_proposal_us"] += r.get("proposal_us", 0.0)
                d["sum_map_parse_us"] += r.get("proposal_map_parse_us", 0.0)
                d["sum_rewrite_apply_us"] += r.get("proposal_rewrite_apply_us", 0.0)
                d["sum_virtual_reference_tokenize_us"] += r.get(
                    "proposal_virtual_reference_tokenize_us", 0.0
                )
                d["sum_transpld_index_build_us"] += r.get(
                    "proposal_transpld_index_build_us", 0.0
                )
            for d in by_proposal.values():
                n = d["n"]
                d["share_steps"] = n / len(method_steps)
                d["mean_k"] = d["sum_k"] / n
                d["mean_proposal_tokens"] = d["sum_proposal_tokens"] / n
                d["mean_match_len"] = d["sum_match_len"] / n
                d["mean_accepted_nonroot"] = d["sum_accepted_nonroot"] / n
                d["mean_wall_us"] = d["sum_wall_us"] / n
                d["mean_proposal_us"] = d["sum_proposal_us"] / n
                d["mean_map_parse_us"] = d["sum_map_parse_us"] / n
                d["mean_rewrite_apply_us"] = d["sum_rewrite_apply_us"] / n
                d["mean_virtual_reference_tokenize_us"] = (
                    d["sum_virtual_reference_tokenize_us"] / n
                )
                d["mean_transpld_index_build_us"] = d["sum_transpld_index_build_us"] / n
            by_method[method]["by_proposal"] = by_proposal
        if method.startswith("retrieval_"):
            # Retrieval prepends the target's cached argmax to any retrieved
            # suffix continuation. This reports only the continuation tokens
            # accepted past that guaranteed first candidate.
            total_retrieved_accepted = sum(
                max(0, r["n_accepted_drafts"] - 1) for r in method_steps
            )
            by_method[method]["mean_accepted_retrieved_drafts_per_step"] = (
                total_retrieved_accepted / len(method_steps)
            )

    asts_steps = [r for r in per_step_records if r["method"] == "asts_eagle"]
    for r in asts_steps:
        nt = r.get("node_type") or "default"
        d = by_node_type.setdefault(nt, {"n": 0, "sum_k": 0, "sum_accepted": 0, "sum_wall_us": 0})
        d["n"] += 1
        d["sum_k"] += r["k"]
        d["sum_accepted"] += r["n_accepted_drafts"]
        d["sum_wall_us"] += r["wall_us"]
    for nt, d in by_node_type.items():
        n = d["n"]
        d["mean_k"] = d["sum_k"] / n
        d["mean_accepted"] = d["sum_accepted"] / n
        d["acceptance_rate"] = d["mean_accepted"] / d["mean_k"] if d["mean_k"] > 0 else 0
        d["mean_wall_us"] = d["sum_wall_us"] / n

    return {"by_method": by_method, "by_node_type": by_node_type}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    p.add_argument("--eagle-checkpoint", required=True)
    p.add_argument("--n", type=int, default=164)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--k-fixed", default="4,8")
    p.add_argument(
        "--tree-shapes",
        default="2,2",
        help="Comma-separated list of `k,W` pairs for tree-tail spec, "
        "e.g. '2,2;3,2'. k = chain length (>=2), W = leaf width.",
    )
    p.add_argument(
        "--retrieval-index",
        default=None,
        help="Path to a retrieval index dir built by proto_app.build_retrieval_index. "
        "If set and 'retrieval' is in --methods, runs retrieval-based drafting.",
    )
    p.add_argument(
        "--retrieval-draft-len",
        type=int,
        default=10,
        help="Max number of tokens to draft per retrieval lookup.",
    )
    p.add_argument(
        "--router-retrieval-min-match",
        type=int,
        default=8,
        help="VANTAGE-Full retrieval route requires at least this suffix match length.",
    )
    p.add_argument(
        "--router-retrieval-high-match",
        type=int,
        default=12,
        help="Suffix match length treated as high visibility by VANTAGE-Full.",
    )
    p.add_argument(
        "--router-low-visibility",
        type=float,
        default=0.35,
        help="Visibility score below which VANTAGE-Full routes to chain_k1.",
    )
    p.add_argument(
        "--router-high-visibility",
        type=float,
        default=0.72,
        help="Visibility score above which VANTAGE-Full may use long-chain mode.",
    )
    p.add_argument(
        "--router-tail-margin",
        type=float,
        default=0.08,
        help="Top-1/top-2 probability margin below which VANTAGE-Full uses tail branching.",
    )
    p.add_argument(
        "--router-enable-long-chain",
        action="store_true",
        help="Allow VANTAGE-Full to route very high-visibility steps to chain_k3.",
    )
    p.add_argument(
        "--router-disable-ast-zone",
        action="store_true",
        help="Ablation: make VANTAGE-Full visibility ignore lit/mid/dark AST priors.",
    )
    p.add_argument(
        "--router-disable-retrieval",
        action="store_true",
        help="Ablation: disable retrieval routes even when --retrieval-index is provided.",
    )
    p.add_argument(
        "--router-disable-scope",
        action="store_true",
        help="Ablation: disable local identifier/scope-copy routes.",
    )
    p.add_argument(
        "--router-disable-rolling",
        action="store_true",
        help="Ablation: ignore rolling acceptance history in the visibility score.",
    )
    p.add_argument("--identifier-max-draft-len", type=int, default=6)
    p.add_argument("--literal-max-draft-len", type=int, default=8)
    p.add_argument("--local-suffix-min-match", type=int, default=4)
    p.add_argument("--local-suffix-max-query-len", type=int, default=16)
    p.add_argument("--local-suffix-max-draft-len", type=int, default=8)
    p.add_argument("--alpha-min-match-len", type=int, default=6)
    p.add_argument("--alpha-max-query-len", type=int, default=24)
    p.add_argument("--alpha-max-draft-len", type=int, default=8)
    p.add_argument("--alpha-top-matches", type=int, default=1)
    p.add_argument("--alpha-enable-roles", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--alpha-stop-on-unmapped", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--alpha-filter-exact", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--alpha-scope-fill", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--multisuffix-key-lengths", default="3,4,5,6,8,12,16")
    p.add_argument("--multisuffix-top-k", type=int, default=4)
    p.add_argument("--multisuffix-max-tree-nodes", type=int, default=12)
    p.add_argument("--multisuffix-max-draft-len", type=int, default=16)
    p.add_argument(
        "--multisuffix-pool",
        choices=["local", "prompt", "generated"],
        default="local",
    )
    p.add_argument("--codespine-key-lengths", default="4,5,6,8,12,16")
    p.add_argument("--codespine-min-match-len", type=int, default=4)
    p.add_argument("--codespine-max-spine-len", type=int, default=32)
    p.add_argument("--codespine-max-tree-nodes", type=int, default=12)
    p.add_argument("--codespine-branch-budget", type=int, default=2)
    p.add_argument(
        "--codespine-pool",
        choices=["local", "prompt", "generated"],
        default="local",
    )
    p.add_argument("--codespine-allow-short-match", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--codespine-enable-identifier-branches", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--codespine-enable-delimiter-branches", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--edit-anchor-max-draft-len", type=int, default=32)
    p.add_argument("--edit-anchor-min-chars", type=int, default=12)
    p.add_argument("--edit-anchor-require-signal", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--symbol-tree-branch-budget", type=int, default=4)
    p.add_argument("--symbol-tree-max-tree-nodes", type=int, default=12)
    p.add_argument("--symbol-tree-max-symbol-tokens", type=int, default=8)
    p.add_argument("--symbol-tree-min-prefix-chars", type=int, default=1)
    p.add_argument(
        "--assistant-model",
        default="Qwen/Qwen2.5-Coder-0.5B",
        help="Assistant model for BlazEdit-style assisted decoding baselines.",
    )
    p.add_argument(
        "--blazedit-micro-draft-tokens",
        type=int,
        default=40,
        help="Fallback/default micro draft window for custom BlazEdit rows.",
    )
    p.add_argument(
        "--blazedit-max-num-run",
        type=int,
        default=4,
        help="Fallback/default number of assistant PLD micro-runs.",
    )
    p.add_argument(
        "--blazedit-max-matching-ngram-size",
        type=int,
        default=10,
        help="Default max n-gram size for BlazEdit prompt lookup rows.",
    )
    p.add_argument(
        "--blazedit-assistant-confidence-threshold",
        type=float,
        default=None,
        help="Optional top-1 probability threshold for dynamic assisted rows.",
    )
    p.add_argument(
        "--pld-opportunity-trace",
        action="store_true",
        help=(
            "Opt-in verbose tracing for baseline PLD opportunity analysis. "
            "Adds decoded suffix/draft/source snippets and exact-match ambiguity counts."
        ),
    )
    p.add_argument(
        "--pld-rerank-top-k",
        type=int,
        default=4,
        help="Top-K ambiguous exact PLD candidates considered by rerank_exact_pld.",
    )
    p.add_argument(
        "--pld-rerank-weights",
        default="",
        help=(
            "Weights JSON for rerank_exact_pld. Empty uses the checked-in "
            "data/routers/pld_reranker_k4_v1.json default."
        ),
    )
    p.add_argument(
        "--pld-rerank-only-ambiguous",
        default="true",
        choices=["true", "false"],
        help="Only rerank exact PLD hits with more than one source candidate.",
    )
    p.add_argument(
        "--pld-rerank-fallback",
        default="baseline",
        choices=["baseline", "error"],
        help="Fallback behavior when runtime reranking cannot score candidates.",
    )
    p.add_argument(
        "--pld-rerank-debug-trace",
        action="store_true",
        help="Reserved debug switch for verbose reranker traces.",
    )
    p.add_argument("--pld-rerank-margin", type=float, default=0.0)
    p.add_argument(
        "--pld-rerank-margin-gate",
        default="false",
        choices=["true", "false"],
        help="Fall back to baseline PLD unless best-score minus baseline-score clears margin.",
    )
    p.add_argument(
        "--pld-rerank-always-include-baseline",
        default="true",
        choices=["true", "false"],
        help="Force the baseline PLD source position into the scored top-K set.",
    )
    p.add_argument(
        "--pld-rerank-enable-left-extension",
        default="false",
        choices=["true", "false"],
    )
    p.add_argument("--pld-rerank-left-extension-max", type=int, default=128)
    p.add_argument(
        "--pld-rerank-policy",
        default="learned",
        choices=[
            "learned",
            "fixed_rank",
            "source_continuity",
            "left_extension",
            "learned_leftctx_margin",
        ],
    )
    p.add_argument("--pld-rerank-fixed-rank", type=int, default=0)
    p.add_argument(
        "--mtp-heads-checkpoint",
        default="/data/pld_mtp/postpld_linear_k4_n917_v1/postpld_mtp_heads_k4_linear.pt",
    )
    p.add_argument("--mtp-num-heads", type=int, default=4)
    p.add_argument("--mtp-trigger-accepted-len", type=int, default=4)
    p.add_argument("--mtp-trigger-threshold", type=int, default=None)
    p.add_argument("--mtp-position", choices=["post_pld"], default="post_pld")
    p.add_argument("--mtp-disable", action="store_true")
    p.add_argument("--mtp-queue-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--mtp-use-queued-only-on-weak-pld",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--mtp-disable-extra-verify",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--weak-pld-router-path",
        default="/tmp/pld_mtp/weak_router/router.pkl",
        help="Pickled weak-PLD router for weak_router_capped_pld rows.",
    )
    p.add_argument(
        "--weak-pld-router-threshold",
        type=float,
        default=None,
        help="Override weak-router probability threshold. Encoded method names take precedence when unset.",
    )
    p.add_argument(
        "--weak-pld-cap-tokens",
        type=int,
        default=None,
        help="Override PLD draft cap for weak-router predicted weak steps.",
    )
    p.add_argument("--lookahead-window", type=int, default=8)
    p.add_argument("--lookahead-ngram", type=int, default=4)
    p.add_argument("--lookahead-iters", type=int, default=4)
    p.add_argument("--lookahead-max-draft", type=int, default=16)
    p.add_argument(
        "--lookahead-one-forward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Constrain Lookahead candidate generation to one extra target forward.",
    )
    p.add_argument(
        "--lookahead-stable-prefix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer the longest stable prefix across final Jacobi iterations.",
    )
    p.add_argument(
        "--lookahead-trajectory-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Seed the next lookahead window from the previous trajectory.",
    )
    p.add_argument(
        "--pld-lookahead-router",
        choices=["rule", "hist_gbdt", "none"],
        default="rule",
    )
    p.add_argument(
        "--pld-lookahead-router-path",
        default="/tmp/pld_mtp/weak_router/router.pkl",
    )
    p.add_argument("--pld-lookahead-router-threshold", type=float, default=0.3)
    p.add_argument("--pld-lookahead-weak-threshold", type=int, default=4)
    p.add_argument(
        "--pld-lookahead-trigger",
        choices=["router_weak", "router_high_conf_weak", "pld_miss"],
        default="router_weak",
    )
    p.add_argument(
        "--pld-lookahead-mode",
        choices=["replace_weak_pld", "rescue_after_pld"],
        default="replace_weak_pld",
    )
    p.add_argument(
        "--pld-lookahead-fallback",
        choices=["pld", "greedy"],
        default="pld",
    )
    p.add_argument("--pld-lookahead-min-candidate-len", type=int, default=1)
    p.add_argument(
        "--macro-chunks-json",
        default="",
        help="Optional JSON list or language-keyed dict of mined macro chunks.",
    )
    p.add_argument(
        "--code-proposer-fallback",
        choices=["root", "eagle_k2", "tail"],
        default="eagle_k2",
        help="Fallback when a code proposer has no candidate.",
    )
    p.add_argument(
        "--transpld-min-match-len",
        type=int,
        default=4,
        help=(
            "Minimum transformed-view n-gram match length for TransPLD rows. "
            "Method aliases can override this with vantage_transpld_m{N}_..."
        ),
    )
    p.add_argument(
        "--task-router-json",
        default="",
        help="Prompt-time logistic router JSON for vantage_task_router_* methods.",
    )
    p.add_argument(
        "--task-router-exact-strong-threshold",
        type=int,
        default=32,
        help="Exact-PLD fast-path length used inside task-router positive methods.",
    )
    p.add_argument(
        "--task-router-trans-margin",
        type=int,
        default=0,
        help="Minimum transformed-candidate length margin used inside task-router positive methods.",
    )
    p.add_argument(
        "--context-tail-widths",
        default="default=2,identifier=3,literal=3,margin=3",
        help="Comma-separated widths for vantage_code_tail_context.",
    )
    p.add_argument("--methods", default="vanilla,eagle,asts")
    p.add_argument(
        "--problem-jsonl",
        default="",
        help=(
            "Optional manifest JSONL with task_id/prompt/language/reference/"
            "deterministic_target fields. Overrides --language problem loading."
        ),
    )
    p.add_argument(
        "--skip-eagle-load",
        action="store_true",
        help="Skip EAGLE checkpoint loading for vanilla/PLD/root-fallback proposer runs.",
    )
    p.add_argument(
        "--eagle2-config",
        default="total_tokens=26,topk=10,depth=6",
        help="EAGLE-2 tree config: 'total_tokens=N,topk=K,depth=D'.",
    )
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument(
        "--target-trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the target tokenizer/model.",
    )
    p.add_argument(
        "--chat-template",
        default="none",
        choices=["none", "user"],
        help=(
            "Prompt formatting for chat/instruct models. `user` wraps the raw "
            "prompt as one user message with add_generation_prompt=True; the raw "
            "prompt is still used for rewrite/reference extraction."
        ),
    )
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
        help="Benchmark selector: 'python' = HumanEval Python, 'ts'/'typescript' = "
        "MultiPL-E HumanEval-TS, 'mbpp' = MBPP-Sanitized completion-style, "
        "'repo_python' = prefix-only same-file repo completion, "
        "'repo_edit_python' = synthetic repo edit/rewrite prompts, "
        "'repo_edit_rename_python' = synthetic rename/reference-drift edit prompts, "
        "'codeeditor_python' = CodeEditorBench Python debugging prompts, "
        "'codeeditor_switch_python' = CodeEditorBench Python requirement-switch prompts.",
    )
    p.add_argument(
        "--prompt-variant",
        default="full",
        choices=["full", "no_examples", "signature_only", "desc_only"],
        help="Prompt sensitivity variant. HumanEval supports full/no_examples/signature_only; "
        "MBPP supports full/desc_only.",
    )
    p.add_argument(
        "--policy",
        default="default",
        choices=["default", "data-derived", "optimal"],
        help="'optimal' is a backwards-compatible alias for 'data-derived'.",
    )
    p.add_argument(
        "--policy-json",
        default=None,
        help="Optional JSON policy table with node_type -> k entries. Used for "
        "held-out policy experiments without editing asts/ast_policy.py.",
    )
    p.add_argument(
        "--task-id-file",
        default=None,
        help="Optional JSON list/dict or newline-delimited file of task ids to evaluate. "
        "Useful for train/test split policy checks.",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("run_eagle_eval")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    steps_path = output_dir / "steps.jsonl"
    aggregate_path = output_dir / "aggregate.json"
    completions_path = output_dir / "completions.jsonl"

    methods_requested = [m.strip() for m in args.methods.split(",") if m.strip()]
    methods_to_run = set(methods_requested)
    if "tree_eagle" in methods_to_run:
        methods_to_run.add("tree")
    router_method_names = _router_methods(methods_requested)
    code_proposer_method_names = _code_proposer_methods(methods_requested)
    blazedit_method_names = _blazedit_methods(methods_requested)
    vantage_mv_method_names = _vantage_mv_methods(methods_requested)
    selected_mv_frozen_method_names = _selected_mv_frozen_methods(methods_requested)
    blazedit_configs = {
        method: replace(
            parse_blazedit_method(
                method,
                assistant_model_name=args.assistant_model,
                confidence_threshold=args.blazedit_assistant_confidence_threshold,
                default_ngram_size=args.blazedit_max_matching_ngram_size,
            ),
            pld_opportunity_trace=(
                args.pld_opportunity_trace and method == "blazedit_pld_w128_n10"
            ),
            pld_rerank_top_k=args.pld_rerank_top_k,
            pld_rerank_weights_path=args.pld_rerank_weights or None,
            pld_rerank_only_ambiguous=(args.pld_rerank_only_ambiguous == "true"),
            pld_rerank_fallback=args.pld_rerank_fallback,
            pld_rerank_debug_trace=args.pld_rerank_debug_trace,
            pld_rerank_margin=args.pld_rerank_margin,
            pld_rerank_margin_gate=(args.pld_rerank_margin_gate == "true"),
            pld_rerank_always_include_baseline=(
                args.pld_rerank_always_include_baseline == "true"
            ),
            pld_rerank_enable_left_extension=(
                args.pld_rerank_enable_left_extension == "true"
            ),
            pld_rerank_left_extension_max=args.pld_rerank_left_extension_max,
            pld_rerank_policy=args.pld_rerank_policy,
            pld_rerank_fixed_rank=args.pld_rerank_fixed_rank,
            mtp_heads_checkpoint=args.mtp_heads_checkpoint,
            mtp_num_heads=args.mtp_num_heads,
            mtp_trigger_accepted_len=(
                args.mtp_trigger_threshold
                if args.mtp_trigger_threshold is not None
                else args.mtp_trigger_accepted_len
            ),
            mtp_position=args.mtp_position,
            mtp_disable=args.mtp_disable,
            mtp_queue_enabled=args.mtp_queue_enabled,
            mtp_use_queued_only_on_weak_pld=args.mtp_use_queued_only_on_weak_pld,
            mtp_disable_extra_verify=args.mtp_disable_extra_verify,
            weak_pld_router_path=args.weak_pld_router_path,
            weak_pld_router_threshold=(
                args.weak_pld_router_threshold
                if args.weak_pld_router_threshold is not None
                else parse_blazedit_method(
                    method,
                    assistant_model_name=args.assistant_model,
                    confidence_threshold=args.blazedit_assistant_confidence_threshold,
                    default_ngram_size=args.blazedit_max_matching_ngram_size,
                ).weak_pld_router_threshold
            ),
            weak_pld_cap_tokens=(
                args.weak_pld_cap_tokens
                if args.weak_pld_cap_tokens is not None
                else parse_blazedit_method(
                    method,
                    assistant_model_name=args.assistant_model,
                    confidence_threshold=args.blazedit_assistant_confidence_threshold,
                    default_ngram_size=args.blazedit_max_matching_ngram_size,
                ).weak_pld_cap_tokens
            ),
            lookahead_window=args.lookahead_window,
            lookahead_ngram=args.lookahead_ngram,
            lookahead_iters=args.lookahead_iters,
            lookahead_max_draft=args.lookahead_max_draft,
            lookahead_one_forward=args.lookahead_one_forward,
            lookahead_stable_prefix=args.lookahead_stable_prefix,
            lookahead_trajectory_cache=args.lookahead_trajectory_cache,
            pld_lookahead_router=args.pld_lookahead_router,
            pld_lookahead_router_path=args.pld_lookahead_router_path,
            pld_lookahead_router_threshold=args.pld_lookahead_router_threshold,
            pld_lookahead_weak_threshold=args.pld_lookahead_weak_threshold,
            pld_lookahead_trigger=args.pld_lookahead_trigger,
            pld_lookahead_mode=args.pld_lookahead_mode,
            pld_lookahead_fallback=args.pld_lookahead_fallback,
            pld_lookahead_min_candidate_len=args.pld_lookahead_min_candidate_len,
        )
        for method in blazedit_method_names
    }
    # Method names such as lookahead_w4_n2_i2 and
    # pld_gated_lookahead_w8_n4_i1_d4 encode their own lookahead shape.
    # The CLI lookahead flags remain the default for pld_gated_lookahead_w128_n10.
    for method, cfg in list(blazedit_configs.items()):
        if re.fullmatch(r"lookahead_w\d+_n\d+_i\d+", method) or re.fullmatch(
            r"pld_gated_lookahead_w\d+_n\d+_i\d+(?:_d\d+)?", method
        ):
            base_cfg = parse_blazedit_method(
                method,
                assistant_model_name=args.assistant_model,
                confidence_threshold=args.blazedit_assistant_confidence_threshold,
                default_ngram_size=args.blazedit_max_matching_ngram_size,
            )
            blazedit_configs[method] = replace(
                cfg,
                lookahead_window=base_cfg.lookahead_window,
                lookahead_ngram=base_cfg.lookahead_ngram,
                lookahead_iters=base_cfg.lookahead_iters,
                lookahead_max_draft=base_cfg.lookahead_max_draft,
                lookahead_one_forward=base_cfg.lookahead_one_forward,
            )
    vantage_mv_configs = {
        method: parse_vantage_mv_method(method)
        for method in vantage_mv_method_names
    }
    selected_mv_config = parse_vantage_mv_method(
        "vantage_mv_pld_s96_x1_m16_t8_w128_n10"
    )
    fixed_ks = [int(x) for x in args.k_fixed.split(",")] if "eagle" in methods_to_run else []

    tree_shapes: list[tuple[int, int]] = []
    if "tree" in methods_to_run and args.tree_shapes.strip():
        for shape_str in args.tree_shapes.split(";"):
            parts = shape_str.split(",")
            if len(parts) != 2:
                raise ValueError(f"--tree-shapes entries must be 'k,W'; got '{shape_str}'")
            tree_shapes.append((int(parts[0]), int(parts[1])))

    # Lookahead Decoding monkey-patches HF transformers — must happen BEFORE
    # _load_model. We initialize lade only if 'lookahead' is in --methods so
    # other runs are unaffected.
    lookahead_ok = False
    if "lookahead" in methods_to_run:
        lookahead_ok = init_lookahead()
        log.info("lookahead init: %s", "OK" if lookahead_ok else "FAILED (will skip method)")

    requires_eagle = bool(
        fixed_ks
        or "asts" in methods_to_run
        or tree_shapes
        or router_method_names
        or "eagle2" in methods_to_run
    )
    for code_method in code_proposer_method_names:
        fallback, _ = _code_proposer_fallback_and_width(
            code_method,
            args.code_proposer_fallback,
        )
        if fallback != "root":
            requires_eagle = True
            break
    if args.skip_eagle_load and requires_eagle:
        raise ValueError("--skip-eagle-load requested, but at least one method needs EAGLE")

    log.info("loading target=%s dtype=%s", args.target, args.dtype)
    target_tok, target = _load_model(
        args.target,
        dtype=args.dtype,
        attn_impl=args.attn_impl,
        trust_remote_code=args.target_trust_remote_code,
    )

    assistant = None
    assistant_blazedit_modes = {"assisted_static", "assisted_dynamic", "two_layer"}
    needs_assistant = any(cfg.mode in assistant_blazedit_modes for cfg in blazedit_configs.values()) or any(
        cfg.use_edit_neural_drafter for cfg in vantage_mv_configs.values()
    )
    if needs_assistant:
        log.info(
            "loading assistant=%s dtype=%s",
            args.assistant_model,
            args.dtype,
        )
        _, assistant = _load_model(
            args.assistant_model,
            dtype=args.dtype,
            attn_impl=args.attn_impl,
        )

    eagle_head = None
    ckpt = {"step": -1}
    if requires_eagle and not args.skip_eagle_load:
        log.info("loading EAGLE checkpoint: %s", args.eagle_checkpoint)
        eagle_head, eagle_cfg, ckpt = load_eagle_checkpoint(args.eagle_checkpoint, dtype=args.dtype)
        log.info("eagle: %d params, ckpt step=%d", sum(p.numel() for p in eagle_head.parameters()), ckpt.get("step", -1))
    else:
        log.info("skipping EAGLE checkpoint load")

    eos = [int(target_tok.eos_token_id)]
    log.info("eos: %s", eos)

    # Normalize language. `benchmark` selects which problem set to load.
    # `ast_lang` is what the tree-sitter parser uses (mbpp problems are
    # Python, even though the benchmark string is "mbpp").
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
    if args.task_id_file:
        task_ids = _load_task_ids(args.task_id_file)
        before = len(problems)
        problems = [p for p in problems if p.task_id in task_ids]
        log.info("task-id filter: kept %d/%d problems from %s", len(problems), before, args.task_id_file)
    log.info("loaded %d problems", len(problems))
    task_router = load_task_router(args.task_router_json) if args.task_router_json else None
    if any(_TASK_ROUTER_RE.fullmatch(m) for m in code_proposer_method_names) and task_router is None:
        raise ValueError("vantage_task_router_* methods require --task-router-json")
    if selected_mv_frozen_method_names and task_router is None:
        raise ValueError("vantage_selected_mv_frozen_w128_n10 requires --task-router-json")
    stop_texts = stop_texts_for_language(benchmark)
    mined_macro_chunks = _load_macro_chunks(args.macro_chunks_json, benchmark)
    context_tail_widths = _parse_context_tail_widths(args.context_tail_widths)
    code_proposer_stacks = {
        method: _build_code_proposer_stack(
            method,
            target_tok,
            benchmark,
            args,
            mined_macro_chunks,
        )
        for method in code_proposer_method_names
    }
    selected_frozen_stack = (
        _build_code_proposer_stack(
            "vantage_frozen_transpld",
            target_tok,
            benchmark,
            args,
            mined_macro_chunks,
        )
        if selected_mv_frozen_method_names
        else None
    )
    per_step_records = []
    completions = []
    # Build the key set from what we'll actually run (METHODS is just legacy default)
    active_methods: list[str] = []
    if "vanilla" in methods_to_run:
        active_methods.append("vanilla")
    for k in fixed_ks:
        active_methods.append(f"eagle_k{k}")
    if "asts" in methods_to_run:
        active_methods.append("asts_eagle")
    for tk, tw in tree_shapes:
        active_methods.append(f"tree_eagle_k{tk}w{tw}")
    active_methods.extend(router_method_names)
    active_methods.extend(code_proposer_method_names)
    active_methods.extend(blazedit_method_names)
    active_methods.extend(vantage_mv_method_names)
    active_methods.extend(selected_mv_frozen_method_names)

    eagle2_total_tokens = 26
    eagle2_topk = 10
    eagle2_depth = 6
    if "eagle2" in methods_to_run:
        for kv in args.eagle2_config.split(","):
            k_, v_ = kv.split("=")
            if k_ == "total_tokens":
                eagle2_total_tokens = int(v_)
            elif k_ == "topk":
                eagle2_topk = int(v_)
            elif k_ == "depth":
                eagle2_depth = int(v_)
        active_methods.append(
            f"eagle2_t{eagle2_total_tokens}k{eagle2_topk}d{eagle2_depth}"
        )

    retrieval_index: RetrievalIndex | None = None
    router_requested = bool(router_method_names)
    retrieval_requested = "retrieval" in methods_to_run
    if retrieval_requested or (router_requested and args.retrieval_index):
        if retrieval_requested and not args.retrieval_index:
            raise ValueError("--retrieval-index is required when 'retrieval' is in --methods")
        log.info("loading retrieval index from %s", args.retrieval_index)
        retrieval_index = RetrievalIndex.load(args.retrieval_index)
        log.info(
            "retrieval index: %d tokens, sep=%d, corpus=%s",
            retrieval_index.tokens.shape[0],
            retrieval_index.sep_token_id,
            retrieval_index.meta.get("corpus", "?"),
        )
    if retrieval_requested:
        active_methods.append(f"retrieval_d{args.retrieval_draft_len}")

    if "lookahead" in methods_to_run and lookahead_ok:
        active_methods.append("lookahead")

    n_new_tokens_per_method = {m: 0 for m in active_methods}
    output_equivalence_counts = {
        m: {
            "tasks": 0,
            "matches_vanilla": 0,
            "matches_blazedit_pld_w128_n10": 0,
        }
        for m in active_methods
    }
    t_start = time.perf_counter_ns()

    for idx, prob in enumerate(problems):
        log.info("[%d/%d] %s", idx + 1, len(problems), prob.task_id)
        prompt_ids = _encode_prompt_ids(target_tok, prob.prompt, args.chat_template)
        torch.cuda.synchronize()

        method_outputs = {}

        if "vanilla" in methods_to_run:
            v_res = vanilla_ar(
                prompt_ids=prompt_ids, target=target,
                max_new_tokens=args.max_new_tokens, eos_token_ids=eos,
                method_name="vanilla",
            )
            for s in v_res.steps:
                rec = asdict(s); rec["task_id"] = prob.task_id
                per_step_records.append(rec)
            n_new_tokens_per_method["vanilla"] += v_res.n_new_tokens
            method_outputs["vanilla"] = {
                "tokens": v_res.output_token_ids[len(prompt_ids):],
                "wall_us": v_res.wall_us_total,
                "n_new_tokens": v_res.n_new_tokens,
            }

        for k in fixed_ks:
            method_name = f"eagle_k{k}"
            f_res = fixed_eagle_spec(
                prompt_ids=prompt_ids, target=target, eagle_head=eagle_head,
                max_new_tokens=args.max_new_tokens, eos_token_ids=eos, k=k,
            )
            for s in f_res.steps:
                rec = asdict(s); rec["method"] = method_name; rec["task_id"] = prob.task_id
                per_step_records.append(rec)
            n_new_tokens_per_method[method_name] += f_res.n_new_tokens
            method_outputs[method_name] = {
                "tokens": f_res.output_token_ids[len(prompt_ids):],
                "wall_us": f_res.wall_us_total,
                "n_new_tokens": f_res.n_new_tokens,
            }

        if "asts" in methods_to_run:
            policy_table = _load_policy_table(args.policy, args.policy_json)
            ast_policy = ASTPolicy(language=ast_lang, policy=policy_table)
            a_res = asts_eagle_spec(
                prompt_ids=prompt_ids, target=target, eagle_head=eagle_head,
                max_new_tokens=args.max_new_tokens, eos_token_ids=eos,
                tokenizer=target_tok, ast_policy=ast_policy,
            )
            for s in a_res.steps:
                rec = asdict(s); rec["task_id"] = prob.task_id
                per_step_records.append(rec)
            n_new_tokens_per_method["asts_eagle"] += a_res.n_new_tokens
            method_outputs["asts_eagle"] = {
                "tokens": a_res.output_token_ids[len(prompt_ids):],
                "wall_us": a_res.wall_us_total,
                "n_new_tokens": a_res.n_new_tokens,
            }

        for tk, tw in tree_shapes:
            method_name = f"tree_eagle_k{tk}w{tw}"
            t_res = tree_tail_eagle_spec(
                prompt_ids=prompt_ids, target=target, eagle_head=eagle_head,
                max_new_tokens=args.max_new_tokens, eos_token_ids=eos,
                k=tk, width=tw,
            )
            for s in t_res.steps:
                rec = asdict(s); rec["method"] = method_name; rec["task_id"] = prob.task_id
                per_step_records.append(rec)
            n_new_tokens_per_method[method_name] += t_res.n_new_tokens
            method_outputs[method_name] = {
                "tokens": t_res.output_token_ids[len(prompt_ids):],
                "wall_us": t_res.wall_us_total,
                "n_new_tokens": t_res.n_new_tokens,
            }

        for router_method in router_method_names:
            router_policy = ASTPolicy(
                language=ast_lang,
                policy=_load_policy_table(args.policy, args.policy_json),
            )
            router_cfg = _router_config_for_method(router_method, args)
            nh_res = vantage_router_spec(
                prompt_ids=prompt_ids,
                target=target,
                eagle_head=eagle_head,
                tokenizer=target_tok,
                ast_policy=router_policy,
                max_new_tokens=args.max_new_tokens,
                eos_token_ids=eos,
                retrieval_index=retrieval_index,
                config=router_cfg,
                max_retrieval_draft_len=args.retrieval_draft_len,
                method_name=router_method,
            )
            for s in nh_res.steps:
                rec = asdict(s); rec["task_id"] = prob.task_id
                per_step_records.append(rec)
            n_new_tokens_per_method[router_method] += nh_res.n_new_tokens
            method_outputs[router_method] = {
                "tokens": nh_res.output_token_ids[len(prompt_ids):],
                "wall_us": nh_res.wall_us_total,
                "n_new_tokens": nh_res.n_new_tokens,
            }

        for code_method in code_proposer_method_names:
            problem_ast_lang = (
                "typescript"
                if prob.language in {"typescript", "ts"}
                else "python"
            )
            task_router_exact_reason = (
                _task_router_route_reason(prob, task_router, args)
                if _TASK_ROUTER_RE.fullmatch(code_method)
                else None
            )
            routed_exact_reason = (
                _routed_transpld_exact_route_reason(prob, target_tok)
                if (
                    _ROUTED_TRANSPLD_RE.fullmatch(code_method)
                    or _ADOPT_SIMPLE_TRANSPLD_RE.fullmatch(code_method)
                    or _DISPATCH_TRANSPLD_RE.fullmatch(code_method)
                    or _PRECOMPUTED_TRANSPLD_RE.fullmatch(code_method)
                    or _COMPETE_TRANSPLD_RE.fullmatch(code_method)
                    or _LAZY_TRANSPLD_RE.fullmatch(code_method)
                    or _MULTIVIEW_PLD_RE.fullmatch(code_method)
                    or _MULTIVIEW_TREE_RE.fullmatch(code_method)
                    or _TASK_ROUTER_RE.fullmatch(code_method)
                    or _FROZEN_TRANSPLD_RE.fullmatch(code_method)
                )
                else None
            )
            routed_exact_reason = task_router_exact_reason or routed_exact_reason
            if routed_exact_reason is not None:
                # The zero-drift/no-map route must be the same decoder and
                # configuration as the BlazEdit PLD baseline row.  Do this at
                # decode-selection time; the rooted code-proposer verifier has
                # different root/cache accounting and is not an equivalent PLD
                # fallback for paper comparisons.
                cp_res = blazedit_speculative_ar(
                    prompt_ids=prompt_ids,
                    target=target,
                    assistant=None,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_ids=eos,
                    config=_routed_transpld_pld_config(code_method, args),
                    method_name=code_method,
                )
                _stamp_routed_exact_pld_steps(cp_res, reason=routed_exact_reason)
                for s in cp_res.steps:
                    rec = asdict(s); rec["task_id"] = prob.task_id
                    per_step_records.append(rec)
                n_new_tokens_per_method[code_method] += cp_res.n_new_tokens
                method_outputs[code_method] = {
                    "tokens": cp_res.output_token_ids[len(prompt_ids):],
                    "wall_us": cp_res.wall_us_total,
                    "n_new_tokens": cp_res.n_new_tokens,
                }
                continue
            code_policy = ASTPolicy(
                language=problem_ast_lang if benchmark == "manifest" else ast_lang,
                policy=_load_policy_table(args.policy, args.policy_json),
            )
            fallback, tail_width = _code_proposer_fallback_and_width(
                code_method,
                args.code_proposer_fallback,
            )
            cp_res = code_proposer_spec(
                prompt_ids=prompt_ids,
                target=target,
                eagle_head=eagle_head,
                tokenizer=target_tok,
                ast_policy=code_policy,
                max_new_tokens=args.max_new_tokens,
                eos_token_ids=eos,
                proposer_stack=code_proposer_stacks[code_method],
                fallback=fallback,
                tail_width=tail_width,
                context_tail_widths=context_tail_widths,
                language=prob.language if benchmark == "manifest" else benchmark,
                method_name=code_method,
                reference=prob.reference,
                metadata=prob.metadata,
                prompt_text=prob.prompt,
            )
            for s in cp_res.steps:
                rec = asdict(s); rec["task_id"] = prob.task_id
                per_step_records.append(rec)
            n_new_tokens_per_method[code_method] += cp_res.n_new_tokens
            method_outputs[code_method] = {
                "tokens": cp_res.output_token_ids[len(prompt_ids):],
                "wall_us": cp_res.wall_us_total,
                "n_new_tokens": cp_res.n_new_tokens,
            }

        for blazedit_method in blazedit_method_names:
            bz_res = blazedit_speculative_ar(
                prompt_ids=prompt_ids,
                target=target,
                assistant=assistant,
                max_new_tokens=args.max_new_tokens,
                eos_token_ids=eos,
                config=blazedit_configs[blazedit_method],
                method_name=blazedit_method,
                tokenizer=target_tok,
            )
            for s in bz_res.steps:
                rec = asdict(s); rec["task_id"] = prob.task_id
                per_step_records.append(rec)
            n_new_tokens_per_method[blazedit_method] += bz_res.n_new_tokens
            method_outputs[blazedit_method] = {
                "tokens": bz_res.output_token_ids[len(prompt_ids):],
                "wall_us": bz_res.wall_us_total,
                "n_new_tokens": bz_res.n_new_tokens,
            }

        for mv_method in vantage_mv_method_names:
            mv_res = vantage_mv_pld_speculative_ar(
                prompt_ids=prompt_ids,
                target=target,
                tokenizer=target_tok,
                max_new_tokens=args.max_new_tokens,
                eos_token_ids=eos,
                config=vantage_mv_configs[mv_method],
                method_name=mv_method,
                prompt_text=str(prob.prompt),
                reference=str(getattr(prob, "reference", "") or ""),
                metadata=getattr(prob, "metadata", None),
                assistant=assistant,
                assistant_model_name=args.assistant_model,
            )
            for s in mv_res.steps:
                rec = asdict(s); rec["task_id"] = prob.task_id
                per_step_records.append(rec)
            n_new_tokens_per_method[mv_method] += mv_res.n_new_tokens
            method_outputs[mv_method] = {
                "tokens": mv_res.output_token_ids[len(prompt_ids):],
                "wall_us": mv_res.wall_us_total,
                "n_new_tokens": mv_res.n_new_tokens,
            }

        for selected_method in selected_mv_frozen_method_names:
            selected_choice = _selected_mv_frozen_choice(prob, task_router)
            if selected_choice == "blazedit_pld_w128_n10":
                sel_res = blazedit_speculative_ar(
                    prompt_ids=prompt_ids,
                    target=target,
                    assistant=None,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_ids=eos,
                    config=parse_blazedit_method(
                        "blazedit_pld_w128_n10",
                        assistant_model_name=args.assistant_model,
                        confidence_threshold=args.blazedit_assistant_confidence_threshold,
                        default_ngram_size=args.blazedit_max_matching_ngram_size,
                    ),
                    method_name=selected_method,
                )
            elif selected_choice == "vantage_mv_pld_s96_x1_m16_t8_w128_n10":
                sel_res = vantage_mv_pld_speculative_ar(
                    prompt_ids=prompt_ids,
                    target=target,
                    tokenizer=target_tok,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_ids=eos,
                    config=selected_mv_config,
                    method_name=selected_method,
                    prompt_text=str(prob.prompt),
                    reference=str(getattr(prob, "reference", "") or ""),
                    metadata=getattr(prob, "metadata", None),
                )
            elif selected_choice == "vantage_frozen_transpld":
                if selected_frozen_stack is None:
                    raise ValueError("selected frozen stack was not initialized")
                problem_ast_lang = (
                    "typescript"
                    if prob.language in {"typescript", "ts"}
                    else "python"
                )
                code_policy = ASTPolicy(
                    language=problem_ast_lang if benchmark == "manifest" else ast_lang,
                    policy=_load_policy_table(args.policy, args.policy_json),
                )
                sel_res = code_proposer_spec(
                    prompt_ids=prompt_ids,
                    target=target,
                    eagle_head=eagle_head,
                    tokenizer=target_tok,
                    ast_policy=code_policy,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_ids=eos,
                    proposer_stack=selected_frozen_stack,
                    fallback="root",
                    tail_width=1,
                    context_tail_widths=context_tail_widths,
                    language=prob.language if benchmark == "manifest" else benchmark,
                    method_name=selected_method,
                    reference=prob.reference,
                    metadata=prob.metadata,
                    prompt_text=prob.prompt,
                )
            else:
                raise ValueError(f"selected router produced unknown method: {selected_choice}")
            for s in sel_res.steps:
                rec = asdict(s)
                rec["task_id"] = prob.task_id
                rec["selected_route"] = selected_choice
                per_step_records.append(rec)
            n_new_tokens_per_method[selected_method] += sel_res.n_new_tokens
            method_outputs[selected_method] = {
                "tokens": sel_res.output_token_ids[len(prompt_ids):],
                "wall_us": sel_res.wall_us_total,
                "n_new_tokens": sel_res.n_new_tokens,
            }

        if "eagle2" in methods_to_run:
            method_name = f"eagle2_t{eagle2_total_tokens}k{eagle2_topk}d{eagle2_depth}"
            e2_res = eagle2_speculative_ar(
                prompt_ids=prompt_ids, target=target, eagle_head=eagle_head,
                max_new_tokens=args.max_new_tokens, eos_token_ids=eos,
                method_name=method_name,
                total_tokens=eagle2_total_tokens,
                topk_per_node=eagle2_topk,
                max_depth=eagle2_depth,
            )
            for s in e2_res.steps:
                rec = asdict(s); rec["task_id"] = prob.task_id
                per_step_records.append(rec)
            n_new_tokens_per_method[method_name] += e2_res.n_new_tokens
            method_outputs[method_name] = {
                "tokens": e2_res.output_token_ids[len(prompt_ids):],
                "wall_us": e2_res.wall_us_total,
                "n_new_tokens": e2_res.n_new_tokens,
            }

        if retrieval_requested and retrieval_index is not None:
            method_name = f"retrieval_d{args.retrieval_draft_len}"
            r_res = retrieval_spec(
                prompt_ids=prompt_ids, target=target, index=retrieval_index,
                max_new_tokens=args.max_new_tokens, eos_token_ids=eos,
                max_draft_len=args.retrieval_draft_len,
            )
            for s in r_res.steps:
                rec = asdict(s); rec["method"] = method_name; rec["task_id"] = prob.task_id
                per_step_records.append(rec)
            n_new_tokens_per_method[method_name] += r_res.n_new_tokens
            method_outputs[method_name] = {
                "tokens": r_res.output_token_ids[len(prompt_ids):],
                "wall_us": r_res.wall_us_total,
                "n_new_tokens": r_res.n_new_tokens,
            }

        if "lookahead" in methods_to_run and lookahead_ok:
            method_name = "lookahead"
            la_res = lookahead_spec(
                prompt_ids=prompt_ids, target=target,
                max_new_tokens=args.max_new_tokens, eos_token_ids=eos,
            )
            for s in la_res.steps:
                rec = asdict(s); rec["method"] = method_name; rec["task_id"] = prob.task_id
                per_step_records.append(rec)
            n_new_tokens_per_method[method_name] += la_res.n_new_tokens
            method_outputs[method_name] = {
                "tokens": la_res.output_token_ids[len(prompt_ids):],
                "wall_us": la_res.wall_us_total,
                "n_new_tokens": la_res.n_new_tokens,
            }

        speed = []
        for m, o in method_outputs.items():
            tps = o["n_new_tokens"] / (o["wall_us"] / 1e6) if o["wall_us"] > 0 else 0
            speed.append(f"{m}={tps:.1f}t/s")
        log.info("    %s", "  ".join(speed))

        try:
            row_stop_texts = stop_texts_for_language(
                prob.language if benchmark == "manifest" else benchmark
            )
        except ValueError:
            row_stop_texts = stop_texts
        vanilla_tokens = (
            method_outputs.get("vanilla", {}).get("tokens") if "vanilla" in method_outputs else None
        )
        pld_tokens = (
            method_outputs.get("blazedit_pld_w128_n10", {}).get("tokens")
            if "blazedit_pld_w128_n10" in method_outputs
            else None
        )
        output_equivalence = {}
        for m, o in method_outputs.items():
            matches_vanilla = vanilla_tokens is not None and o["tokens"] == vanilla_tokens
            matches_pld = pld_tokens is not None and o["tokens"] == pld_tokens
            output_equivalence[m] = {
                "matches_vanilla": matches_vanilla if vanilla_tokens is not None else None,
                "matches_blazedit_pld_w128_n10": matches_pld if pld_tokens is not None else None,
            }
            counts = output_equivalence_counts.setdefault(
                m,
                {
                    "tasks": 0,
                    "matches_vanilla": 0,
                    "matches_blazedit_pld_w128_n10": 0,
                },
            )
            counts["tasks"] += 1
            if matches_vanilla:
                counts["matches_vanilla"] += 1
            if matches_pld:
                counts["matches_blazedit_pld_w128_n10"] += 1
        completions.append({
            "task_id": prob.task_id,
            "prompt": prob.prompt,
            "reference": prob.reference,
            "deterministic_target": prob.deterministic_target,
            "metadata": prob.metadata,
            "prompt_variant": args.prompt_variant,
            "language": prob.language,
            "outputs": {
                m: {
                    "n_new_tokens": o["n_new_tokens"],
                    "wall_us": o["wall_us"],
                    "matches_vanilla": output_equivalence.get(m, {}).get("matches_vanilla"),
                    "matches_blazedit_pld_w128_n10": output_equivalence.get(m, {}).get(
                        "matches_blazedit_pld_w128_n10"
                    ),
                    "raw_text": target_tok.decode(o["tokens"], skip_special_tokens=True),
                    "text": truncate_at_stop(
                        target_tok.decode(o["tokens"], skip_special_tokens=True),
                        stop_texts=row_stop_texts,
                    ),
                }
                for m, o in method_outputs.items()
            },
        })

    t_end = time.perf_counter_ns()

    with steps_path.open("w") as f:
        for r in per_step_records:
            f.write(json.dumps(r) + "\n")
    with completions_path.open("w") as f:
        for c in completions:
            f.write(json.dumps(c) + "\n")

    agg = _aggregate(per_step_records, n_new_tokens_per_method)
    agg["output_equivalence"] = output_equivalence_counts
    agg["meta"] = {
        "schema": "asts-spec/eagle_eval/v1",
        "target": args.target,
        "eagle_checkpoint": args.eagle_checkpoint,
        "eagle_train_step": ckpt.get("step", -1),
        "dtype": args.dtype,
        "attn_impl": args.attn_impl,
        "benchmark": benchmark,
        "ast_lang": ast_lang,
        "prompt_variant": args.prompt_variant,
        "chat_template": args.chat_template,
        "n_problems": len(problems),
        "max_new_tokens": args.max_new_tokens,
        "fixed_ks": fixed_ks,
        "methods": active_methods,
        "policy": "data-derived" if args.policy == "optimal" else args.policy,
        "policy_arg": args.policy,
        "policy_json": args.policy_json,
        "task_id_file": args.task_id_file,
        "problem_jsonl": args.problem_jsonl,
        "skip_eagle_load": args.skip_eagle_load,
        "router_config": {
            "low_visibility_threshold": args.router_low_visibility,
            "high_visibility_threshold": args.router_high_visibility,
            "tail_max_margin": args.router_tail_margin,
            "retrieval_min_match": args.router_retrieval_min_match,
            "retrieval_high_match": args.router_retrieval_high_match,
            "enable_long_chain": args.router_enable_long_chain,
            "use_ast_zone": not args.router_disable_ast_zone,
            "use_retrieval": not args.router_disable_retrieval,
            "use_scope": not args.router_disable_scope,
            "use_rolling": not args.router_disable_rolling,
        },
        "code_proposer_config": {
            "identifier_max_draft_len": args.identifier_max_draft_len,
            "literal_max_draft_len": args.literal_max_draft_len,
            "local_suffix_min_match": args.local_suffix_min_match,
            "local_suffix_max_query_len": args.local_suffix_max_query_len,
            "local_suffix_max_draft_len": args.local_suffix_max_draft_len,
            "alpha_min_match_len": args.alpha_min_match_len,
            "alpha_max_query_len": args.alpha_max_query_len,
            "alpha_max_draft_len": args.alpha_max_draft_len,
            "alpha_top_matches": args.alpha_top_matches,
            "alpha_enable_roles": args.alpha_enable_roles,
            "alpha_stop_on_unmapped": args.alpha_stop_on_unmapped,
            "alpha_filter_exact": args.alpha_filter_exact,
            "alpha_scope_fill": args.alpha_scope_fill,
            "multisuffix_key_lengths": args.multisuffix_key_lengths,
            "multisuffix_top_k": args.multisuffix_top_k,
            "multisuffix_max_tree_nodes": args.multisuffix_max_tree_nodes,
            "multisuffix_max_draft_len": args.multisuffix_max_draft_len,
            "multisuffix_pool": args.multisuffix_pool,
            "codespine_key_lengths": args.codespine_key_lengths,
            "codespine_min_match_len": args.codespine_min_match_len,
            "codespine_max_spine_len": args.codespine_max_spine_len,
            "codespine_max_tree_nodes": args.codespine_max_tree_nodes,
            "codespine_branch_budget": args.codespine_branch_budget,
            "codespine_pool": args.codespine_pool,
            "codespine_allow_short_match": args.codespine_allow_short_match,
            "codespine_enable_identifier_branches": args.codespine_enable_identifier_branches,
            "codespine_enable_delimiter_branches": args.codespine_enable_delimiter_branches,
            "transpld_min_match_len": args.transpld_min_match_len,
            "edit_anchor_max_draft_len": args.edit_anchor_max_draft_len,
            "edit_anchor_min_chars": args.edit_anchor_min_chars,
            "edit_anchor_require_signal": args.edit_anchor_require_signal,
            "symbol_tree_branch_budget": args.symbol_tree_branch_budget,
            "symbol_tree_max_tree_nodes": args.symbol_tree_max_tree_nodes,
            "symbol_tree_max_symbol_tokens": args.symbol_tree_max_symbol_tokens,
            "symbol_tree_min_prefix_chars": args.symbol_tree_min_prefix_chars,
            "macro_chunks_json": args.macro_chunks_json,
            "n_mined_macro_chunks": len(mined_macro_chunks),
            "code_proposer_fallback": args.code_proposer_fallback,
            "context_tail_widths": args.context_tail_widths,
        },
        "blazedit_config": {
            "assistant_model": args.assistant_model,
            "methods": {
                method: {
                    "mode": cfg.mode,
                    "micro_draft_tokens": cfg.micro_draft_tokens,
                    "max_num_run": cfg.max_num_run,
                    "max_matching_ngram_size": cfg.max_matching_ngram_size,
                    "assistant_confidence_threshold": cfg.assistant_confidence_threshold,
                    "delta_context_tokens": cfg.delta_context_tokens,
                    "delta_lru_size": cfg.delta_lru_size,
                    "delta_max_patches": cfg.delta_max_patches,
                    "delta_patch_window": cfg.delta_patch_window,
                    "fuzzy_weak_draft_len": cfg.fuzzy_weak_draft_len,
                    "fuzzy_max_draft_tokens": cfg.fuzzy_max_draft_tokens,
                    "fuzzy_require_unique": cfg.fuzzy_require_unique,
                    "mtp_heads_checkpoint": cfg.mtp_heads_checkpoint,
                    "mtp_num_heads": cfg.mtp_num_heads,
                    "mtp_trigger_accepted_len": cfg.mtp_trigger_accepted_len,
                    "mtp_position": cfg.mtp_position,
                    "mtp_disable": cfg.mtp_disable,
                    "mtp_queue_enabled": cfg.mtp_queue_enabled,
                    "mtp_use_queued_only_on_weak_pld": cfg.mtp_use_queued_only_on_weak_pld,
                    "mtp_disable_extra_verify": cfg.mtp_disable_extra_verify,
                }
                for method, cfg in blazedit_configs.items()
            },
        },
        "timing_scope": (
            "generated-token tok/s includes first target prefill, speculative verify, "
            "cache crop/catchup, retrieval lookup, and AST parse work; excludes prompt "
            "tokenization, text decoding/stop-string truncation, and test execution"
        ),
        "wall_us_total": (t_end - t_start) / 1000.0,
    }
    aggregate_path.write_text(json.dumps(agg, indent=2))

    print()
    print("=" * 78)
    print("ASTS-Spec + EAGLE Prototype: HumanEval Eval Summary")
    print("=" * 78)
    print(f"  target:    {args.target}")
    print(f"  eagle:     {args.eagle_checkpoint}")
    print(f"  problems:  {len(problems)}  ({args.max_new_tokens} max new tokens)")
    print(f"  dtype:     {args.dtype}")
    print()
    print(f"  {'method':<14} {'tokens/sec':>12} {'us/token':>10} {'mean_acc':>10} {'mean_k':>8} {'n_steps':>9}")
    print("  " + "-" * 70)
    vanilla_tps = agg["by_method"].get("vanilla", {}).get("tokens_per_sec", 0) or 1
    for method in active_methods:
        if method not in agg["by_method"]:
            continue
        m = agg["by_method"][method]
        speedup = m["tokens_per_sec"] / vanilla_tps
        marker = "  ✓" if method != "vanilla" and speedup >= 1.5 else ("  ~" if speedup >= 1.0 else "  ✗")
        print(
            f"  {method:<14} {m['tokens_per_sec']:>11.1f}  "
            f"{m['us_per_token']:>9.0f}  "
            f"{m['mean_accepted_drafts_per_step']:>9.2f}  "
            f"{m['mean_k_requested']:>7.2f}  "
            f"{m['n_steps']:>8}  ({speedup:.2f}x{marker if method != 'vanilla' else ''})"
        )
    print()
    if agg["by_node_type"]:
        print("  Per-AST-node-type acceptance (asts_eagle, top 15):")
        print(f"    {'node_type':<25} {'n':>5} {'mean_k':>7} {'mean_acc':>9} {'accept_rate':>11}")
        print("    " + "-" * 60)
        for nt, d in sorted(agg["by_node_type"].items(), key=lambda x: -x[1]["n"])[:15]:
            print(
                f"    {nt:<25} {d['n']:>5} {d['mean_k']:>6.1f} "
                f"{d['mean_accepted']:>8.2f} {d['acceptance_rate']:>10.1%}"
            )
    print("=" * 78)


if __name__ == "__main__":
    main()
