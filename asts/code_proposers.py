"""Cheap code-specific proposers for verified speculative decoding.

These proposers are deliberately non-neural.  They inspect the live decoded
prefix and return a short continuation that the target verifier can accept or
reject with the same greedy lossless rule used by EAGLE and tree-tail.
"""

from __future__ import annotations

import keyword
import re
import time
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Iterable, Literal, Protocol

from .ast_policy import CursorContext
from .vantage_policy import decide_prompt_only_saferoute


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER_RE = re.compile(r"(?<![A-Za-z_])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
_STRING_RE = re.compile(r"(?P<quote>['\"])(?:\\.|(?!\1).)*\1")
_PARTIAL_STRING_RE = re.compile(r"(?P<quote>['\"])(?:\\.|(?!\1).)*$")
_CONST_RE = re.compile(r"\b(?:True|False|None|true|false|null|undefined)\b")

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

IDENTIFIER_NODE_TYPES = {
    "identifier",
    "property_identifier",
    "shorthand_property_identifier",
    "attribute",
    "member_expression",
}

LITERAL_NODE_TYPES = {
    "string",
    "string_content",
    "integer",
    "float",
    "number",
    "true",
    "false",
    "none",
    "null",
    "list",
    "tuple",
    "dictionary",
    "pair",
    "object",
    "array",
    "arguments",
    "argument_list",
    "assert_statement",
}

PYTHON_STATIC_MACROS = (
    "):\n    ",
    ":\n    return ",
    ", ",
    "]\n",
    "return True",
    "return False",
    "for i in range(",
    'if __name__ == "__main__":',
)

TYPESCRIPT_STATIC_MACROS = (
    "): boolean {",
    ": number",
    ": string[]",
    ", ",
    "return true;",
    "return false;",
    "} else {",
)

_EDIT_SIGNAL_RE = re.compile(
    r"\b(?:edit|modify|change|rename|replace|fix|debug|rewrite|refactor|translate|polish|update|patch|requirement|switch)\b",
    re.IGNORECASE,
)
_FENCED_CODE_RE = re.compile(
    r"```(?:python|py|typescript|ts|javascript|js|java|cpp|c\+\+|go|rust|[A-Za-z0-9_+-]+)?\s*\n(.*?)```",
    re.DOTALL,
)
_REWRITE_BARE_TERM = r"(?:\.?[A-Za-z_][A-Za-z0-9_\.]{0,80}|[0-9]+(?:\.[0-9]+)?)"
_QUOTED_TERM_RE = re.compile(
    r"`([^`\n]{1,80})`|['\"](\.?[A-Za-z_][A-Za-z0-9_\.]{0,80})['\"]"
)
_REWRITE_PAIR_RE = re.compile(
    r"\b(?:rename|replace|change)\s+"
    rf"(?:`([^`\n]{{1,80}})`|'([^'\n]{{1,80}})'|\"([^\"\n]{{1,80}})\"|({_REWRITE_BARE_TERM}))"
    r"\s+(?:with|to)\s+"
    rf"(?:`([^`\n]{{1,80}})`|'([^'\n]{{1,80}})'|\"([^\"\n]{{1,80}})\"|({_REWRITE_BARE_TERM}))",
    re.IGNORECASE,
)
_REWRITE_ARROW_RE = re.compile(
    rf"(?:`([^`\n]{{1,80}})`|'([^'\n]{{1,80}})'|\"([^\"\n]{{1,80}})\"|({_REWRITE_BARE_TERM}))"
    r"\s*(?:->|=>|→)\s*"
    rf"(?:`([^`\n]{{1,80}})`|'([^'\n]{{1,80}})'|\"([^\"\n]{{1,80}})\"|({_REWRITE_BARE_TERM}))",
    re.IGNORECASE,
)
_NEGATED_REWRITE_PREFIX_RE = re.compile(
    r"(?:\bdo\s+not\b|\bdon't\b|\bdont\b|\bnever\b|\bavoid\b)"
    r"(?:\s+\w+){0,3}\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Proposal:
    kind: str
    tokens: list[int]
    match_len: int
    score: float
    text_preview: str = ""
    source_start_token: int | None = None
    source_end_token: int | None = None
    follow_start_token: int | None = None
    follow_end_token: int | None = None
    query_len: int | None = None
    pool: str | None = None
    source_region: str | None = None
    root_included: bool = False
    match_kind: str | None = None
    canonical_match_len: int | None = None
    substitution_count: int | None = None
    scope_fill_count: int | None = None
    stopped_on_unmapped: bool | None = None
    alpha_exact_filtered: bool | None = None
    zero_nonroot_accept: bool | None = None
    map_source: str | None = None
    inferred_map_count: int | None = None
    inference_confidence: float | None = None
    cursor_pos: int | None = None
    cursor_confidence: float | None = None
    cursor_resync: bool | None = None
    view_id: str | None = None
    compound_view_count: int | None = None
    active_map_count: int | None = None
    route: str | None = None
    route_reason: str | None = None
    backoff_active: bool | None = None
    rewrite_hit_count: int | None = None
    route_window_accept_rate: float | None = None
    rewrite_zero_accept_streak: int | None = None
    adoption_state: str | None = None
    adoption_transition: str | None = None
    frontier_distance: int | None = None
    frontier_probes: int | None = None
    accepted_crossed_rewrite: int | None = None
    rejected_old_form_frontiers: int | None = None
    blacklisted_rewrite_occurrences: int | None = None
    disabled_by_adoption_gate: bool | None = None
    root_old_match_count: int | None = None
    root_new_match_count: int | None = None
    map_parse_us: float = 0.0
    rewrite_apply_us: float = 0.0
    virtual_reference_tokenize_us: float = 0.0
    transpld_index_build_us: float = 0.0


@dataclass(frozen=True)
class ProposalTreeNode:
    token: int
    parent: int
    depth: int


@dataclass(frozen=True)
class ProposalTree:
    kind: str
    candidates: list[list[int]]
    scores: list[float]
    sources: list[str]
    match_lens: list[int]
    max_nodes: int
    score: float = 0.0
    text_preview: str = ""
    source_start_token: int | None = None
    source_end_token: int | None = None
    follow_start_token: int | None = None
    follow_end_token: int | None = None
    query_len: int | None = None
    pool: str | None = None
    source_region: str | None = None
    root_included: bool = False
    match_kind: str | None = None
    canonical_match_len: int | None = None
    substitution_count: int | None = None
    scope_fill_count: int | None = None
    stopped_on_unmapped: bool | None = None
    alpha_exact_filtered: bool | None = None
    zero_nonroot_accept: bool | None = None
    map_source: str | None = None
    inferred_map_count: int | None = None
    inference_confidence: float | None = None
    cursor_pos: int | None = None
    cursor_confidence: float | None = None
    cursor_resync: bool | None = None
    view_id: str | None = None
    compound_view_count: int | None = None
    active_map_count: int | None = None
    route: str | None = None
    route_reason: str | None = None
    backoff_active: bool | None = None
    rewrite_hit_count: int | None = None
    route_window_accept_rate: float | None = None
    rewrite_zero_accept_streak: int | None = None
    adoption_state: str | None = None
    adoption_transition: str | None = None
    frontier_distance: int | None = None
    frontier_probes: int | None = None
    accepted_crossed_rewrite: int | None = None
    rejected_old_form_frontiers: int | None = None
    blacklisted_rewrite_occurrences: int | None = None
    disabled_by_adoption_gate: bool | None = None
    root_old_match_count: int | None = None
    root_new_match_count: int | None = None
    map_parse_us: float = 0.0
    rewrite_apply_us: float = 0.0
    virtual_reference_tokenize_us: float = 0.0
    transpld_index_build_us: float = 0.0

    @property
    def tokens(self) -> list[int]:
        return self.candidates[0] if self.candidates else []

    @property
    def match_len(self) -> int:
        return max(self.match_lens) if self.match_lens else 0


@dataclass(frozen=True)
class ProposerState:
    prefix: list[int]
    teacher_argmax: int
    text_before: str
    text_after: str
    ctx: CursorContext | None = None
    language: str = "python"
    prompt_len: int = 0
    reference: str = ""
    metadata: dict[str, Any] | None = None
    prompt_text: str = ""

    @property
    def tokens_after_teacher(self) -> list[int]:
        return self.prefix + [self.teacher_argmax]


@dataclass(frozen=True)
class ProposalFeedback:
    prefix_start: int
    prefix_end: int
    proposed_tokens: list[int]
    emitted_tokens: list[int]
    accepted_nonroot: int
    rejected: bool
    proposal_kind: str | None = None
    proposal_match_kind: str | None = None
    source_start_token: int | None = None
    source_end_token: int | None = None
    follow_start_token: int | None = None
    follow_end_token: int | None = None


class CodeProposer(Protocol):
    kind: str

    def propose(self, state: ProposerState) -> Proposal | ProposalTree | None:
        ...


def encode_no_special(tokenizer, text: str) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(t) for t in ids]


def decode_tokens(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def _common_prefix_chars(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def encode_continuation_with_token_healing(
    tokenizer,
    continuation: str,
    *,
    prefix_tokens: list[int] | None = None,
    prefix_text: str | None = None,
    max_len: int | None = None,
) -> list[int]:
    """Encode a continuation with tokenizer-boundary healing.

    Rewrite-normalized PLD constructs target-side continuation text from a
    transformed reference span.  Tokenizing that continuation in isolation can
    be wrong for Llama/SentencePiece-style tokenizers whose first token depends
    on the preceding token's byte/space boundary.  We therefore retokenize the
    continuation with a short decoded anchor from the already generated prefix,
    then choose the suffix whose decoded prefix best matches
    ``prefix_text + continuation`` when appended to the existing prefix tokens.
    """

    if not continuation:
        return []
    prefix_tokens = [int(t) for t in (prefix_tokens or [])]
    if max_len is not None:
        max_len = int(max_len)
        if max_len <= 0:
            return []

    if prefix_text is None:
        prefix_text = decode_tokens(tokenizer, prefix_tokens) if prefix_tokens else ""
    desired = prefix_text + continuation

    candidates: list[tuple[int, list[int]]] = [(1, encode_no_special(tokenizer, continuation))]
    max_anchor_tokens = min(4, len(prefix_tokens))
    for anchor_n in range(1, max_anchor_tokens + 1):
        anchor_tokens = prefix_tokens[-anchor_n:]
        anchor_text = decode_tokens(tokenizer, anchor_tokens)
        if not anchor_text:
            continue
        combined_ids = encode_no_special(tokenizer, anchor_text + continuation)
        anchor_ids = encode_no_special(tokenizer, anchor_text)
        if anchor_ids and combined_ids[: len(anchor_ids)] == anchor_ids:
            candidates.append((4 + anchor_n, combined_ids[len(anchor_ids) :]))
        # If the anchor and continuation merge into a different first token,
        # the exact-prefix drop above is invalid.  Try every suffix of the
        # combined tokenization and let decoded text alignment choose the best
        # suffix that can follow the already-fixed prefix tokens.
        for cut in range(len(combined_ids) + 1):
            candidates.append((2 + anchor_n, combined_ids[cut:]))

    best_tokens: list[int] = []
    best_score: tuple[int, int, int, int] | None = None
    seen: set[tuple[int, ...]] = set()
    for priority, candidate in candidates:
        tokens = [int(t) for t in candidate]
        if max_len is not None:
            tokens = tokens[:max_len]
        key = tuple(tokens)
        if not tokens or key in seen:
            continue
        seen.add(key)
        decoded = decode_tokens(tokenizer, prefix_tokens + tokens)
        common = _common_prefix_chars(decoded, desired)
        exact_prefix = 1 if common == len(decoded) else 0
        starts_after_prefix = 1 if decoded.startswith(prefix_text) else 0
        score = (exact_prefix, starts_after_prefix, common, priority)
        if best_score is None or score > best_score:
            best_score = score
            best_tokens = tokens
    return best_tokens


def _rank_seen_strings(history: str, values: Iterable[tuple[str, int]]) -> list[str]:
    counter: Counter[str] = Counter()
    last_pos: dict[str, int] = {}
    for value, pos in values:
        counter[value] += 1
        last_pos[value] = pos
    return sorted(
        counter,
        key=lambda v: (
            -1 if v else 0,
            -counter[v],
            -last_pos[v],
            len(v),
        ),
    )


def _context_has_type(ctx: CursorContext | None, types: set[str]) -> bool:
    if ctx is None:
        return False
    return bool({ctx.node_type, ctx.deepest_type, *ctx.ancestor_types} & types)


class IdentifierTrieProposer:
    kind = "identifier"

    def __init__(self, tokenizer, max_draft_len: int = 6, min_prefix_chars: int = 1):
        self.tokenizer = tokenizer
        self.max_draft_len = max_draft_len
        self.min_prefix_chars = min_prefix_chars

    def propose(self, state: ProposerState) -> Proposal | None:
        current_match = re.search(r"[A-Za-z_][A-Za-z0-9_]*$", state.text_after)
        if current_match is None:
            return None
        current = current_match.group(0)
        if len(current) < self.min_prefix_chars:
            return None
        if not _context_has_type(state.ctx, IDENTIFIER_NODE_TYPES):
            # Regex fallback still allows identifier continuations when the
            # partial text is clearly in an identifier.
            preceding = state.text_after[: current_match.start()]
            if preceding and (preceding[-1].isalnum() or preceding[-1] == "_"):
                return None

        history = state.text_after[: current_match.start()]
        local_start = max(
            history.rfind("\ndef "),
            history.rfind("\nclass "),
            history.rfind("\nfunction "),
            history.rfind("{"),
        )
        local_history = history[local_start + 1 :] if local_start >= 0 else history

        seen_global: list[tuple[str, int]] = []
        seen_local: set[str] = set()
        for match in _IDENT_RE.finditer(history):
            ident = match.group(0)
            if ident in _ALL_KEYWORDS:
                continue
            seen_global.append((ident, match.start()))
        for match in _IDENT_RE.finditer(local_history):
            ident = match.group(0)
            if ident not in _ALL_KEYWORDS:
                seen_local.add(ident)

        candidates = []
        for ident in _rank_seen_strings(history, seen_global):
            if ident.startswith(current) and len(ident) > len(current):
                local_bonus = 1.0 if ident in seen_local else 0.0
                candidates.append((local_bonus, ident))
        candidates.sort(key=lambda item: (-item[0], len(item[1]), item[1]))

        for local_bonus, ident in candidates:
            continuation = ident[len(current) :]
            tokens = encode_no_special(self.tokenizer, continuation)
            if not tokens:
                continue
            capped = tokens[: self.max_draft_len]
            return Proposal(
                kind=self.kind,
                tokens=capped,
                match_len=len(current),
                score=2.0 + local_bonus + min(1.0, len(capped) / self.max_draft_len),
                text_preview=continuation[:40],
            )
        return None


class LiteralCopyProposer:
    kind = "literal"

    def __init__(self, tokenizer, max_draft_len: int = 8):
        self.tokenizer = tokenizer
        self.max_draft_len = max_draft_len

    def propose(self, state: ProposerState) -> Proposal | None:
        if not (
            state.ctx is None
            or state.ctx.parser_in_error
            or _context_has_type(state.ctx, LITERAL_NODE_TYPES)
        ):
            # Keep a text fallback for obvious partial literals.
            if _PARTIAL_STRING_RE.search(state.text_after) is None and _NUMBER_RE.search(
                state.text_after[-32:]
            ) is None:
                return None

        partial, candidates = self._partial_and_candidates(state.text_after)
        if not partial:
            return None
        history = state.text_after[: -len(partial)]
        ranked = _rank_seen_strings(
            history,
            [(c, history.rfind(c)) for c in candidates if history.rfind(c) >= 0],
        )
        for literal in ranked:
            if literal.startswith(partial) and len(literal) > len(partial):
                continuation = literal[len(partial) :]
                tokens = encode_no_special(self.tokenizer, continuation)
                if not tokens:
                    continue
                capped = tokens[: self.max_draft_len]
                return Proposal(
                    kind=self.kind,
                    tokens=capped,
                    match_len=len(partial),
                    score=1.8 + min(1.0, len(capped) / self.max_draft_len),
                    text_preview=continuation[:40],
                )
        return None

    def _partial_and_candidates(self, text: str) -> tuple[str, list[str]]:
        string_match = _PARTIAL_STRING_RE.search(text)
        if string_match is not None:
            partial = string_match.group(0)
            candidates = [m.group(0) for m in _STRING_RE.finditer(text[: string_match.start()])]
            return partial, candidates

        number_match = re.search(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?$", text)
        if number_match is not None:
            partial = number_match.group(0)
            candidates = [m.group(0) for m in _NUMBER_RE.finditer(text[: number_match.start()])]
            return partial, candidates

        const_match = re.search(r"(?:True|False|None|true|false|null|undefined)$", text)
        if const_match is not None:
            partial = const_match.group(0)
            candidates = [m.group(0) for m in _CONST_RE.finditer(text[: const_match.start()])]
            return partial, candidates

        return "", []


class LocalSuffixProposer:
    kind = "local_suffix"

    def __init__(
        self,
        max_query_len: int = 16,
        min_match_len: int = 4,
        max_draft_len: int = 8,
        pool: str = "local",
        kind: str | None = None,
    ):
        self.max_query_len = max_query_len
        self.min_match_len = min_match_len
        self.max_draft_len = max_draft_len
        if kind is not None:
            self.kind = kind
        if pool not in {"local", "prompt", "generated"}:
            raise ValueError(f"unsupported suffix pool: {pool}")
        self.pool = pool

    def propose(self, state: ProposerState) -> Proposal | None:
        seq = state.tokens_after_teacher
        if len(seq) <= self.min_match_len:
            return None
        max_len = min(self.max_query_len, len(seq) - 1)
        for match_len in range(max_len, self.min_match_len - 1, -1):
            current_start = len(seq) - match_len
            needle = seq[-match_len:]
            best_start: int | None = None
            # Only copy from a prior, non-overlapping occurrence.  Overlapping
            # suffixes can "copy" tokens that are part of the current query
            # suffix, which is not a valid local retrieval continuation.
            for start in range(0, current_start - match_len + 1):
                if not self._start_allowed(start, match_len, state.prompt_len):
                    continue
                if seq[start : start + match_len] == needle:
                    best_start = start
            if best_start is None:
                continue
            follow_start = best_start + match_len
            follow_end = min(follow_start + self.max_draft_len, current_start)
            if follow_start >= follow_end:
                continue
            tokens = list(seq[follow_start:follow_end])
            return Proposal(
                kind=self.kind,
                tokens=tokens,
                match_len=match_len,
                score=1.5 + match_len / max(1, self.max_query_len),
                text_preview="",
                source_start_token=best_start,
                source_end_token=best_start + match_len,
                follow_start_token=follow_start,
                follow_end_token=follow_end,
                query_len=match_len,
                pool=self.pool,
                root_included=False,
            )
        return None

    def _start_allowed(self, start: int, match_len: int, prompt_len: int) -> bool:
        if self.pool == "local":
            return True
        end = start + match_len
        if self.pool == "prompt":
            return end <= prompt_len
        if self.pool == "generated":
            return start >= prompt_len
        return True


class NGramPromptLookupProposer(LocalSuffixProposer):
    """Fixed-window prompt/local n-gram baseline.

    This intentionally mirrors common prompt-lookup / NGram settings: a fixed
    maximum matching n-gram size, a fixed maximum draft length, and a selectable
    prompt-only or prompt+generated pool.  It uses the same root-excluded,
    target-verified decoder as VANTAGE-Suffix.
    """

    kind = "ngram"

    def __init__(
        self,
        max_matching_ngram_size: int = 4,
        max_draft_len: int = 5,
        pool: str = "local",
    ):
        super().__init__(
            max_query_len=max_matching_ngram_size,
            min_match_len=max_matching_ngram_size,
            max_draft_len=max_draft_len,
            pool=pool,
        )


class RootedPLDProposer(LocalSuffixProposer):
    """BlazEdit-style prompt lookup after VANTAGE's guaranteed target root.

    This proposer uses the code-proposer decoder convention: the target argmax
    root is already appended to the query and accepted by construction, while
    PLD contributes only root-excluded continuation tokens.
    """

    kind = "rooted_pld"
    requires_text_context = False

    def __init__(
        self,
        *,
        max_matching_ngram_size: int = 10,
        max_draft_len: int = 40,
        min_matching_ngram_size: int = 1,
        pool: str = "local",
    ):
        super().__init__(
            max_query_len=max_matching_ngram_size,
            min_match_len=min_matching_ngram_size,
            max_draft_len=max_draft_len,
            pool=pool,
            kind="rooted_pld",
        )

    def propose(self, state: ProposerState) -> Proposal | None:
        proposal = super().propose(state)
        if proposal is None:
            return None
        return Proposal(
            kind=self.kind,
            tokens=proposal.tokens,
            match_len=proposal.match_len,
            score=6.0 + min(2.0, len(proposal.tokens) / 20.0) + proposal.match_len / max(1, self.max_query_len),
            text_preview=proposal.text_preview,
            source_start_token=proposal.source_start_token,
            source_end_token=proposal.source_end_token,
            follow_start_token=proposal.follow_start_token,
            follow_end_token=proposal.follow_end_token,
            query_len=proposal.query_len,
            pool=proposal.pool,
            source_region="rooted_pld",
            root_included=False,
            match_kind="rooted_pld",
        )


RewritePLDMode = Literal["vref", "bidir", "oracle"]


class _StaticTokenIndex:
    def __init__(
        self,
        tokens: list[int],
        *,
        min_match_len: int,
        max_match_len: int,
        max_draft_len: int,
        pool: str,
    ):
        self.tokens = list(tokens)
        self.min_match_len = int(min_match_len)
        self.max_match_len = int(max_match_len)
        self.max_draft_len = int(max_draft_len)
        self.pool = pool
        self.index: dict[int, dict[tuple[int, ...], list[int]]] = {}
        max_n = min(self.max_match_len, len(self.tokens))
        for n in range(self.min_match_len, max_n + 1):
            by_key: dict[tuple[int, ...], list[int]] = {}
            # Need at least one following token to draft.
            for start in range(0, len(self.tokens) - n):
                key = tuple(self.tokens[start : start + n])
                by_key.setdefault(key, []).append(start)
            self.index[n] = by_key

    def candidate(
        self,
        query_tokens: list[int],
        *,
        kind: str,
        match_kind: str,
        substitution_count: int,
        map_source: str | None,
        view_id: str | None,
        max_draft_len: int | None = None,
    ) -> Proposal | None:
        if len(query_tokens) < self.min_match_len:
            return None
        draft_cap = self.max_draft_len if max_draft_len is None else int(max_draft_len)
        if draft_cap <= 0:
            return None
        max_n = min(self.max_match_len, len(query_tokens), len(self.tokens) - 1)
        for n in range(max_n, self.min_match_len - 1, -1):
            starts = self.index.get(n, {}).get(tuple(query_tokens[-n:]))
            if not starts:
                continue
            best: Proposal | None = None
            for start in reversed(starts):
                follow_start = start + n
                follow_end = min(follow_start + draft_cap, len(self.tokens))
                if follow_start >= follow_end:
                    continue
                draft = list(self.tokens[follow_start:follow_end])
                proposal = Proposal(
                    kind=kind,
                    tokens=draft,
                    match_len=n,
                    score=7.0 + n / max(1, self.max_match_len) + min(2.0, len(draft) / 32.0),
                    source_start_token=start,
                    source_end_token=start + n,
                    follow_start_token=follow_start,
                    follow_end_token=follow_end,
                    query_len=n,
                    pool=self.pool,
                    source_region=self.pool,
                    root_included=False,
                    match_kind=match_kind,
                    substitution_count=substitution_count or None,
                    map_source=map_source,
                    view_id=view_id,
                    active_map_count=substitution_count,
                )
                if best is None or (len(proposal.tokens), proposal.source_start_token or -1) > (
                    len(best.tokens),
                    best.source_start_token or -1,
                ):
                    best = proposal
            if best is not None:
                return best
        return None


@dataclass(frozen=True)
class RewriteMapSet:
    pairs: dict[str, str]
    source: str = "explicit"

    @classmethod
    def from_pairs(cls, pairs: dict[str, str] | None, source: str = "explicit") -> "RewriteMapSet":
        out: dict[str, str] = {}
        reverse_seen: dict[str, str] = {}
        for old, new in sorted((pairs or {}).items(), key=lambda item: -len(str(item[0]))):
            old_s = _clean_rewrite_term(str(old))
            new_s = _clean_rewrite_term(str(new))
            if not old_s or not new_s or old_s == new_s:
                continue
            if old_s in out and out[old_s] != new_s:
                continue
            if new_s in reverse_seen and reverse_seen[new_s] != old_s:
                continue
            out[old_s] = new_s
            reverse_seen[new_s] = old_s
        return cls(out, source=source)

    def as_dict(self) -> dict[str, str]:
        return dict(self.pairs)

    def singles(self) -> list["RewriteMapSet"]:
        return [RewriteMapSet({old: new}, source=self.source) for old, new in self.pairs.items()]

    @property
    def active_count(self) -> int:
        return len(self.pairs)


@dataclass(frozen=True)
class RewriteView:
    view_id: str
    key_tokens: list[int]
    value_tokens: list[int]
    rewrite_map: dict[str, str]
    map_source: str
    source_label: str


class RewriteNormalizedPLDProposer:
    """Prompt lookup over rewrite-normalized reference/code streams.

    This baseline keeps the target prompt unchanged.  The transformed
    reference is only a CPU-side draft source, and emitted tokens are still
    target-verified by the code-proposer decoder.
    """

    kind = "rewrite_norm_pld"

    def __init__(
        self,
        tokenizer,
        *,
        mode: RewritePLDMode,
        max_matching_ngram_size: int = 10,
        max_draft_len: int = 128,
        min_matching_ngram_size: int = 1,
    ):
        if mode not in {"vref", "bidir", "oracle"}:
            raise ValueError(f"unsupported rewrite-normalized PLD mode: {mode}")
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_matching_ngram_size = int(max_matching_ngram_size)
        self.max_draft_len = int(max_draft_len)
        self.min_matching_ngram_size = int(min_matching_ngram_size)
        self.reset()

    def reset(self) -> None:
        self._vref_cache_key: tuple[str, tuple[tuple[str, str], ...]] | None = None
        self._vref_cache_tokens: list[int] | None = None

    def _virtual_reference_tokens(
        self,
        reference: str,
        rewrite_map: dict[str, str],
    ) -> tuple[list[int], float, float]:
        cache_key = (reference, tuple(sorted(rewrite_map.items())))
        if self._vref_cache_key == cache_key and self._vref_cache_tokens is not None:
            return list(self._vref_cache_tokens), 0.0, 0.0
        t_apply_0 = time.perf_counter_ns()
        virtual_reference = _apply_word_map(reference, rewrite_map)
        rewrite_apply_us = (time.perf_counter_ns() - t_apply_0) / 1000.0
        t_tok_0 = time.perf_counter_ns()
        tokens = encode_no_special(self.tokenizer, virtual_reference)
        tokenize_us = (time.perf_counter_ns() - t_tok_0) / 1000.0
        self._vref_cache_key = cache_key
        self._vref_cache_tokens = list(tokens)
        return tokens, rewrite_apply_us, tokenize_us

    def propose(self, state: ProposerState) -> Proposal | None:
        prompt_text, _ = _split_prompt_generated(self.tokenizer, state)
        refs = _extract_reference_blocks(prompt_text)
        reference = state.reference or (refs[0] if refs else "")
        if not reference:
            return None
        t_map_0 = time.perf_counter_ns()
        rewrite_map, map_source = self._rewrite_map(prompt_text, state)
        map_parse_us = (time.perf_counter_ns() - t_map_0) / 1000.0
        if not rewrite_map:
            return None
        if self.mode == "bidir":
            proposal = self._propose_bidir(state, reference, rewrite_map, map_source=map_source)
            return (
                replace(proposal, map_parse_us=map_parse_us)
                if proposal is not None
                else None
            )
        external_tokens, rewrite_apply_us, tokenize_us = self._virtual_reference_tokens(
            reference,
            rewrite_map,
        )
        return self._propose_exact(
            query_tokens=state.tokens_after_teacher,
            local_tokens=state.tokens_after_teacher,
            external_tokens=external_tokens,
            external_pool="oracle_reference" if self.mode == "oracle" else "virtual_reference",
            match_kind=self.mode,
            substitution_count=len(rewrite_map),
            map_source=map_source,
            view_id=f"{self.mode}:{_stable_view_key(rewrite_map)}",
            active_map_count=len(rewrite_map),
            prefix_tokens=state.tokens_after_teacher,
            prefix_text=state.text_after,
            map_parse_us=map_parse_us,
            rewrite_apply_us=rewrite_apply_us,
            virtual_reference_tokenize_us=tokenize_us,
        )

    def _rewrite_map(self, prompt_text: str, state: ProposerState) -> tuple[dict[str, str], str]:
        if self.mode == "oracle":
            metadata = state.metadata or {}
            nested = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else {}
            pairs = metadata.get("rewrite_pairs") or nested.get("rewrite_pairs")
            return _coerce_rewrite_pairs(pairs), "oracle"
        pairs = _rewrite_pairs(prompt_text)
        if pairs:
            return pairs, "explicit"
        return {}, "explicit"

    def _propose_bidir(
        self,
        state: ProposerState,
        reference: str,
        rewrite_map: dict[str, str],
        *,
        map_source: str = "explicit",
    ) -> Proposal | None:
        inverse = {new: old for old, new in rewrite_map.items()}
        normalized_text_after = _apply_word_map(state.text_after, inverse)
        normalized_reference = _apply_word_map(reference, inverse)
        normalized_query_tokens = encode_no_special(self.tokenizer, normalized_text_after)
        local_norm_tokens = normalized_query_tokens
        reference_norm_tokens = encode_no_special(self.tokenizer, normalized_reference)
        proposal = self._propose_exact(
            query_tokens=normalized_query_tokens,
            local_tokens=local_norm_tokens,
            external_tokens=reference_norm_tokens,
            external_pool="bidir_normalized",
            match_kind="bidir",
            substitution_count=len(rewrite_map),
            instantiate_map=rewrite_map,
            map_source=map_source,
            view_id=f"bidir:{_stable_view_key(rewrite_map)}",
            active_map_count=len(rewrite_map),
            prefix_tokens=state.tokens_after_teacher,
            prefix_text=state.text_after,
        )
        return proposal

    def _propose_vref(
        self,
        state: ProposerState,
        reference: str,
        rewrite_map: dict[str, str],
        *,
        map_source: str = "explicit",
    ) -> Proposal | None:
        external_tokens, rewrite_apply_us, tokenize_us = self._virtual_reference_tokens(
            reference,
            rewrite_map,
        )
        return self._propose_exact(
            query_tokens=state.tokens_after_teacher,
            local_tokens=state.tokens_after_teacher,
            external_tokens=external_tokens,
            external_pool="virtual_reference",
            match_kind="vref",
            substitution_count=len(rewrite_map),
            map_source=map_source,
            view_id=f"vref:{_stable_view_key(rewrite_map)}",
            active_map_count=len(rewrite_map),
            prefix_tokens=state.tokens_after_teacher,
            prefix_text=state.text_after,
            rewrite_apply_us=rewrite_apply_us,
            virtual_reference_tokenize_us=tokenize_us,
        )

    def _propose_exact(
        self,
        *,
        query_tokens: list[int],
        local_tokens: list[int],
        external_tokens: list[int],
        external_pool: str,
        match_kind: str,
        substitution_count: int,
        instantiate_map: dict[str, str] | None = None,
        map_source: str | None = None,
        view_id: str | None = None,
        compound_view_count: int | None = None,
        active_map_count: int | None = None,
        prefix_tokens: list[int] | None = None,
        prefix_text: str | None = None,
        map_parse_us: float = 0.0,
        rewrite_apply_us: float = 0.0,
        virtual_reference_tokenize_us: float = 0.0,
    ) -> Proposal | None:
        if len(query_tokens) <= self.min_matching_ngram_size:
            return None
        max_len = min(self.max_matching_ngram_size, len(query_tokens) - 1)
        best: Proposal | None = None
        t_lookup_0 = time.perf_counter_ns()
        for match_len in range(max_len, self.min_matching_ngram_size - 1, -1):
            needle = query_tokens[-match_len:]
            candidates: list[Proposal] = []
            local_current_start = len(local_tokens) - match_len
            if local_current_start > 0:
                candidates.extend(
                    self._source_candidates(
                        source_tokens=local_tokens,
                        needle=needle,
                        match_len=match_len,
                        current_start=local_current_start,
                        pool="local",
                        match_kind=match_kind,
                        substitution_count=substitution_count,
                        instantiate_map=instantiate_map,
                        map_source=map_source,
                        view_id=view_id,
                        compound_view_count=compound_view_count,
                        active_map_count=active_map_count,
                        prefix_tokens=prefix_tokens,
                        prefix_text=prefix_text,
                    )
                )
            candidates.extend(
                self._source_candidates(
                    source_tokens=external_tokens,
                    needle=needle,
                    match_len=match_len,
                    current_start=None,
                    pool=external_pool,
                    match_kind=match_kind,
                    substitution_count=substitution_count,
                    instantiate_map=instantiate_map,
                    map_source=map_source,
                    view_id=view_id,
                    compound_view_count=compound_view_count,
                    active_map_count=active_map_count,
                    prefix_tokens=prefix_tokens,
                    prefix_text=prefix_text,
                )
            )
            if candidates:
                best = max(
                    candidates,
                    key=lambda p: (
                        p.match_len,
                        1 if p.pool != "local" else 0,
                        len(p.tokens),
                        p.source_start_token or -1,
                    ),
                )
                break
        lookup_us = (time.perf_counter_ns() - t_lookup_0) / 1000.0
        if best is None:
            return None
        return replace(
            best,
            map_parse_us=map_parse_us,
            rewrite_apply_us=rewrite_apply_us,
            virtual_reference_tokenize_us=virtual_reference_tokenize_us,
            transpld_index_build_us=lookup_us,
        )

    def _source_candidates(
        self,
        *,
        source_tokens: list[int],
        needle: list[int],
        match_len: int,
        current_start: int | None,
        pool: str,
        match_kind: str,
        substitution_count: int,
        instantiate_map: dict[str, str] | None,
        map_source: str | None,
        view_id: str | None,
        compound_view_count: int | None,
        active_map_count: int | None,
        prefix_tokens: list[int] | None,
        prefix_text: str | None,
    ) -> list[Proposal]:
        if not source_tokens or len(source_tokens) < match_len:
            return []
        max_start = len(source_tokens) - match_len
        if current_start is not None:
            max_start = min(max_start, current_start - match_len)
        out: list[Proposal] = []
        for start in range(max_start, -1, -1):
            if source_tokens[start : start + match_len] != needle:
                continue
            follow_start = start + match_len
            follow_limit = current_start if current_start is not None else len(source_tokens)
            follow_end = min(follow_start + self.max_draft_len, follow_limit)
            if follow_start >= follow_end:
                continue
            raw_tokens = list(source_tokens[follow_start:follow_end])
            tokens = raw_tokens
            preview = ""
            if instantiate_map:
                continuation = self.tokenizer.decode(raw_tokens, skip_special_tokens=False)
                instantiated = _apply_word_map(continuation, instantiate_map)
                tokens = encode_continuation_with_token_healing(
                    self.tokenizer,
                    instantiated,
                    prefix_tokens=prefix_tokens,
                    prefix_text=prefix_text,
                    max_len=self.max_draft_len,
                )
                preview = instantiated[:40]
            if not tokens:
                continue
            out.append(
                Proposal(
                    kind=self.kind,
                    tokens=tokens[: self.max_draft_len],
                    match_len=match_len,
                    score=7.0 + match_len / max(1, self.max_matching_ngram_size) + min(2.0, len(tokens) / 32.0),
                    text_preview=preview,
                    source_start_token=start,
                    source_end_token=start + match_len,
                    follow_start_token=follow_start,
                    follow_end_token=follow_end,
                    query_len=match_len,
                    pool=pool,
                    source_region=pool,
                    root_included=False,
                    match_kind=match_kind,
                    substitution_count=substitution_count or None,
                    map_source=map_source,
                    view_id=view_id,
                    compound_view_count=compound_view_count,
                    active_map_count=active_map_count,
                )
            )
        return out


def _clone_proposal(proposal: Proposal, **updates: Any) -> Proposal:
    return replace(proposal, **updates)


def _prompt_reference_and_map(
    tokenizer,
    state: ProposerState,
    *,
    oracle: bool = False,
) -> tuple[str, str, dict[str, str], str]:
    prompt_text, _ = _split_prompt_generated(tokenizer, state)
    refs = _extract_reference_blocks(prompt_text)
    reference = state.reference or (refs[0] if refs else "")
    if oracle:
        metadata = state.metadata or {}
        nested = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else {}
        pairs = metadata.get("rewrite_pairs") or nested.get("rewrite_pairs")
        return prompt_text, reference, _coerce_rewrite_pairs(pairs), "oracle"
    pairs = _rewrite_pairs(prompt_text)
    if pairs:
        return prompt_text, reference, pairs, "explicit"
    return prompt_text, reference, {}, "explicit"


class TransPLDProposer:
    """Prompt-defined transformed-reference lookup with exact PLD fallback."""

    kind = "transpld"

    def __init__(
        self,
        tokenizer,
        *,
        max_matching_ngram_size: int = 10,
        max_draft_len: int = 128,
        min_matching_ngram_size: int = 1,
        transformed_min_matching_ngram_size: int = 4,
    ):
        self.tokenizer = tokenizer
        self.max_matching_ngram_size = int(max_matching_ngram_size)
        self.max_draft_len = int(max_draft_len)
        self.min_matching_ngram_size = int(min_matching_ngram_size)
        self.transformed_min_matching_ngram_size = int(transformed_min_matching_ngram_size)
        self._rewrite = RewriteNormalizedPLDProposer(
            tokenizer,
            mode="vref",
            max_matching_ngram_size=self.max_matching_ngram_size,
            max_draft_len=self.max_draft_len,
            min_matching_ngram_size=self.transformed_min_matching_ngram_size,
        )
        self._exact = RootedPLDProposer(
            max_matching_ngram_size=self.max_matching_ngram_size,
            max_draft_len=self.max_draft_len,
            min_matching_ngram_size=self.min_matching_ngram_size,
        )

    def reset(self) -> None:
        self._rewrite.reset()

    def observe(self, feedback: ProposalFeedback) -> None:
        return None

    def propose(self, state: ProposerState) -> Proposal | None:
        proposal = self._rewrite.propose(state)
        if proposal is not None:
            if proposal.match_len < self.transformed_min_matching_ngram_size:
                raise AssertionError(
                    "TransPLD emitted a transformed-view proposal below "
                    f"the minimum match length: {proposal.match_len} < "
                    f"{self.transformed_min_matching_ngram_size}"
                )
            return _clone_proposal(
                proposal,
                kind=self.kind,
                match_kind="transpld_vref",
                source_region="transpld",
                score=proposal.score + 0.25,
            )
        exact = self._exact.propose(state)
        if exact is None:
            return None
        return _clone_proposal(
            exact,
            kind=self.kind,
            match_kind="transpld_exact_fallback",
            source_region="transpld_exact_fallback",
            map_source=None,
            view_id="identity",
            active_map_count=0,
            score=exact.score - 0.05,
        )


class DispatchTransPLDProposer:
    """Pure prompt-time dispatch TransPLD.

    Decode-level routing handles no-map/no-effect prompts by invoking the same
    exact PLD implementation used by the standalone PLD row.  When this
    proposer is constructed, the prompt has an effective rewrite map, so it
    runs only the bidirectional transformed-view lookup.  There is no online
    adoption detector or mid-generation switching.
    """

    kind = "dispatch_transpld"

    def __init__(
        self,
        tokenizer,
        *,
        max_matching_ngram_size: int = 10,
        max_draft_len: int = 128,
        transformed_min_matching_ngram_size: int = 4,
    ):
        self.tokenizer = tokenizer
        self.max_matching_ngram_size = int(max_matching_ngram_size)
        self.max_draft_len = int(max_draft_len)
        self.transformed_min_matching_ngram_size = int(transformed_min_matching_ngram_size)
        self._rewrite = RewriteNormalizedPLDProposer(
            tokenizer,
            mode="bidir",
            max_matching_ngram_size=self.max_matching_ngram_size,
            max_draft_len=self.max_draft_len,
            min_matching_ngram_size=self.transformed_min_matching_ngram_size,
        )

    def reset(self) -> None:
        self._rewrite.reset()

    def observe(self, feedback: ProposalFeedback) -> None:
        return None

    def propose(self, state: ProposerState) -> Proposal | None:
        proposal = self._rewrite.propose(state)
        if proposal is None:
            return None
        if proposal.match_len < self.transformed_min_matching_ngram_size:
            raise AssertionError(
                "Dispatch TransPLD emitted a transformed-view proposal below "
                f"the minimum match length: {proposal.match_len} < "
                f"{self.transformed_min_matching_ngram_size}"
            )
        return _clone_proposal(
            proposal,
            kind=self.kind,
            match_kind="dispatch_transpld_bidir",
            source_region="dispatch_transpld",
            route="transpld",
            route_reason="prompt_time_rewrite_map",
            backoff_active=False,
            rewrite_hit_count=1,
            route_window_accept_rate=0.0,
            score=proposal.score + 0.25,
        )


class PrecomputedTransPLDProposer:
    """Prebuilt transformed-reference PLD with optional exact-PLD competition.

    The reference rewrite, tokenization, and n-gram index are built once in
    ``prepare`` before the decode loop.  Per step, the proposer only performs
    token suffix lookup over prebuilt dictionaries and, optionally, compares
    against rooted exact PLD.
    """

    kind = "precomputed_transpld"
    requires_text_context = False

    def __init__(
        self,
        tokenizer,
        *,
        max_matching_ngram_size: int = 10,
        max_draft_len: int = 128,
        transformed_min_matching_ngram_size: int = 4,
        compete_exact: bool = False,
        margin: int = 0,
        include_fenced_view: bool = True,
        exact_min_matching_ngram_size: int = 1,
    ):
        self.tokenizer = tokenizer
        self.max_matching_ngram_size = int(max_matching_ngram_size)
        self.max_draft_len = int(max_draft_len)
        self.transformed_min_matching_ngram_size = int(transformed_min_matching_ngram_size)
        self.compete_exact = bool(compete_exact)
        self.margin = int(margin)
        self.include_fenced_view = bool(include_fenced_view)
        self.exact_min_matching_ngram_size = int(exact_min_matching_ngram_size)
        self._exact = RootedPLDProposer(
            max_matching_ngram_size=self.max_matching_ngram_size,
            max_draft_len=self.max_draft_len,
            min_matching_ngram_size=self.exact_min_matching_ngram_size,
        )
        self.reset()

    def reset(self) -> None:
        self._indexes: list[_StaticTokenIndex] = []
        self._rewrite_map: dict[str, str] = {}
        self._map_source = "explicit"
        self._view_id = "none"
        self._prepared = False
        self._pending_map_parse_us = 0.0
        self._pending_rewrite_apply_us = 0.0
        self._pending_tokenize_us = 0.0
        self._pending_index_build_us = 0.0

    def prepare(
        self,
        *,
        prompt_ids: list[int],
        prompt_text: str,
        reference: str,
        metadata: dict[str, Any] | None,
        prompt_len: int,
        language: str,
    ) -> None:
        del prompt_ids, metadata, prompt_len, language
        self._prepared = True
        t_map = time.perf_counter_ns()
        pairs = _rewrite_pairs(prompt_text)
        map_source = "explicit"
        self._rewrite_map = pairs
        self._map_source = map_source
        self._pending_map_parse_us = (time.perf_counter_ns() - t_map) / 1000.0
        if not reference or not pairs:
            return

        t_apply = time.perf_counter_ns()
        virtual_reference = _apply_word_map(reference, pairs)
        variants = [("virtual_reference", virtual_reference)]
        if self.include_fenced_view:
            variants.append(("virtual_reference_fenced", f"```python\n{virtual_reference}```"))
        self._pending_rewrite_apply_us = (time.perf_counter_ns() - t_apply) / 1000.0

        tokenized: list[tuple[str, list[int]]] = []
        t_tok = time.perf_counter_ns()
        for label, text in variants:
            toks = encode_no_special(self.tokenizer, text)
            if toks:
                tokenized.append((label, toks))
        self._pending_tokenize_us = (time.perf_counter_ns() - t_tok) / 1000.0

        t_index = time.perf_counter_ns()
        self._indexes = [
            _StaticTokenIndex(
                toks,
                min_match_len=self.transformed_min_matching_ngram_size,
                max_match_len=self.max_matching_ngram_size,
                max_draft_len=self.max_draft_len,
                pool=label,
            )
            for label, toks in tokenized
        ]
        self._pending_index_build_us = (time.perf_counter_ns() - t_index) / 1000.0
        self._view_id = f"precomputed:{_stable_view_key(pairs)}"

    def observe(self, feedback: ProposalFeedback) -> None:
        return None

    def _take_pending_timings(self) -> tuple[float, float, float, float]:
        timings = (
            self._pending_map_parse_us,
            self._pending_rewrite_apply_us,
            self._pending_tokenize_us,
            self._pending_index_build_us,
        )
        self._pending_map_parse_us = 0.0
        self._pending_rewrite_apply_us = 0.0
        self._pending_tokenize_us = 0.0
        self._pending_index_build_us = 0.0
        return timings

    def _trans_candidate(self, state: ProposerState) -> Proposal | None:
        best: Proposal | None = None
        query = state.tokens_after_teacher
        for idx in self._indexes:
            cand = idx.candidate(
                query,
                kind=self.kind,
                match_kind="precomputed_transpld",
                substitution_count=len(self._rewrite_map),
                map_source=self._map_source,
                view_id=self._view_id,
            )
            if cand is None:
                continue
            if best is None or (
                cand.match_len,
                len(cand.tokens),
                1 if cand.pool == "virtual_reference" else 0,
            ) > (
                best.match_len,
                len(best.tokens),
                1 if best.pool == "virtual_reference" else 0,
            ):
                best = cand
        return best

    def propose(self, state: ProposerState) -> Proposal | None:
        if not self._prepared:
            self.prepare(
                prompt_ids=state.prefix[: state.prompt_len],
                prompt_text=state.prompt_text,
                reference=state.reference,
                metadata=state.metadata,
                prompt_len=state.prompt_len,
                language=state.language,
            )
        trans = self._trans_candidate(state)
        exact = self._exact.propose(state) if self.compete_exact else None
        chosen = trans
        route = "transpld"
        reason = "precomputed_transpld"
        if exact is not None and (
            trans is None or len(trans.tokens) < len(exact.tokens) + self.margin
        ):
            chosen = exact
            route = "exact_pld"
            reason = "exact_candidate_competition"
        if chosen is None:
            return None
        map_us, apply_us, tok_us, index_us = self._take_pending_timings()
        if chosen is exact:
            return _clone_proposal(
                exact,
                kind=self.kind,
                match_kind="precomputed_exact_competition",
                source_region="precomputed_exact_competition",
                route=route,
                route_reason=reason,
                map_parse_us=map_us,
                rewrite_apply_us=apply_us,
                virtual_reference_tokenize_us=tok_us,
                transpld_index_build_us=index_us,
                score=exact.score + 0.1,
            )
        return _clone_proposal(
            chosen,
            kind=self.kind,
            match_kind=(
                "precomputed_transpld_compete"
                if self.compete_exact
                else "precomputed_transpld"
            ),
            source_region="precomputed_transpld",
            route=route,
            route_reason=reason,
            backoff_active=False,
            rewrite_hit_count=1,
            map_parse_us=map_us,
            rewrite_apply_us=apply_us,
            virtual_reference_tokenize_us=tok_us,
            transpld_index_build_us=index_us,
            score=chosen.score + 0.25,
        )


class LazyCompeteTransPLDProposer:
    """Path-A conservative rewrite-view router for mixed real commits.

    The proposer first asks exact rooted PLD for a candidate.  If exact PLD
    already has a strong candidate, it returns exact PLD without constructing
    any transformed view.  Otherwise, it lazily builds the transformed-reference
    index, compares candidate lengths, and disables TransPLD for the remainder
    of the generation after a small number of consecutive zero-accept probes.
    """

    kind = "lazy_compete_transpld"
    requires_text_context = False

    def __init__(
        self,
        tokenizer,
        *,
        max_matching_ngram_size: int = 10,
        max_draft_len: int = 128,
        transformed_min_matching_ngram_size: int = 4,
        exact_strong_min_len: int = 32,
        trans_len_margin: int = 32,
        zero_accept_tripwire_limit: int = 2,
    ):
        self.tokenizer = tokenizer
        self.max_matching_ngram_size = int(max_matching_ngram_size)
        self.max_draft_len = int(max_draft_len)
        self.transformed_min_matching_ngram_size = int(transformed_min_matching_ngram_size)
        self.exact_strong_min_len = int(exact_strong_min_len)
        self.trans_len_margin = int(trans_len_margin)
        self.zero_accept_tripwire_limit = int(zero_accept_tripwire_limit)
        self._exact = RootedPLDProposer(
            max_matching_ngram_size=self.max_matching_ngram_size,
            max_draft_len=self.max_draft_len,
            min_matching_ngram_size=1,
        )
        self._trans = PrecomputedTransPLDProposer(
            tokenizer,
            max_matching_ngram_size=self.max_matching_ngram_size,
            max_draft_len=self.max_draft_len,
            transformed_min_matching_ngram_size=self.transformed_min_matching_ngram_size,
            compete_exact=False,
        )
        self.reset()

    def reset(self) -> None:
        self._trans.reset()
        self._disabled = False
        self._zero_accept_streak = 0
        self._trans_attempts = 0
        self._trans_accepted_nonroot = 0

    def _with_route(
        self,
        proposal: Proposal,
        *,
        route: str,
        reason: str,
        match_kind: str,
        source_region: str,
        score_delta: float = 0.0,
        map_parse_us: float = 0.0,
        rewrite_apply_us: float = 0.0,
        virtual_reference_tokenize_us: float = 0.0,
        transpld_index_build_us: float = 0.0,
    ) -> Proposal:
        return _clone_proposal(
            proposal,
            kind=self.kind,
            match_kind=match_kind,
            source_region=source_region,
            route=route,
            route_reason=reason,
            backoff_active=self._disabled,
            rewrite_zero_accept_streak=self._zero_accept_streak,
            route_window_accept_rate=(
                self._trans_accepted_nonroot / self._trans_attempts
                if self._trans_attempts
                else 0.0
            ),
            map_parse_us=proposal.map_parse_us + map_parse_us,
            rewrite_apply_us=proposal.rewrite_apply_us + rewrite_apply_us,
            virtual_reference_tokenize_us=(
                proposal.virtual_reference_tokenize_us + virtual_reference_tokenize_us
            ),
            transpld_index_build_us=proposal.transpld_index_build_us + transpld_index_build_us,
            score=proposal.score + score_delta,
        )

    def _take_trans_timings(self) -> tuple[float, float, float, float]:
        if hasattr(self._trans, "_take_pending_timings"):
            return self._trans._take_pending_timings()  # type: ignore[attr-defined]
        return 0.0, 0.0, 0.0, 0.0

    def propose(self, state: ProposerState) -> Proposal | None:
        exact = self._exact.propose(state)
        if self._disabled:
            return exact
        if exact is not None and len(exact.tokens) >= self.exact_strong_min_len:
            # Literal no-regression fast path: do not build transformed views,
            # do not clone the proposal, and do not attach VANTAGE route
            # metadata. This keeps the skipped path as close as possible to the
            # standalone PLD proposer inside the code-proposer verifier.
            return exact

        trans = self._trans.propose(state)
        if trans is None:
            map_us, apply_us, tok_us, index_us = self._take_trans_timings()
            if exact is None:
                return None
            return self._with_route(
                exact,
                route="exact_pld",
                reason="transpld_miss",
                match_kind="lazy_transpld_miss_exact",
                source_region="lazy_transpld_miss_exact",
                map_parse_us=map_us,
                rewrite_apply_us=apply_us,
                virtual_reference_tokenize_us=tok_us,
                transpld_index_build_us=index_us,
                score_delta=-0.02,
            )

        if exact is not None and len(trans.tokens) < len(exact.tokens) + self.trans_len_margin:
            return self._with_route(
                exact,
                route="exact_pld",
                reason="exact_candidate_competition",
                match_kind="lazy_exact_competition",
                source_region="lazy_exact_competition",
                map_parse_us=trans.map_parse_us,
                rewrite_apply_us=trans.rewrite_apply_us,
                virtual_reference_tokenize_us=trans.virtual_reference_tokenize_us,
                transpld_index_build_us=trans.transpld_index_build_us,
                score_delta=0.01,
            )

        if trans.match_len < self.transformed_min_matching_ngram_size:
            raise AssertionError(
                "Lazy TransPLD emitted a transformed-view proposal below "
                f"the minimum match length: {trans.match_len} < "
                f"{self.transformed_min_matching_ngram_size}"
            )
        return self._with_route(
            trans,
            route="transpld",
            reason="trans_candidate_wins",
            match_kind="lazy_transpld",
            source_region="lazy_transpld",
            score_delta=0.25,
        )

    def observe(self, feedback: ProposalFeedback) -> None:
        if feedback.proposal_kind != self.kind or feedback.proposal_match_kind != "lazy_transpld":
            return
        self._trans_attempts += 1
        accepted = max(0, int(feedback.accepted_nonroot))
        self._trans_accepted_nonroot += accepted
        if accepted == 0:
            self._zero_accept_streak += 1
        else:
            self._zero_accept_streak = 0
        if (
            self.zero_accept_tripwire_limit > 0
            and self._zero_accept_streak >= self.zero_accept_tripwire_limit
        ):
            self._disabled = True


class MultiViewPLDProposer:
    """View-conditioned prompt lookup for code edits.

    Exact PLD remains the default.  Only when exact PLD is weak does this
    proposer consult precomputed transformed/reference views and choose the
    candidate with the best empirical expected acceptance.  All transformed
    views are built once in ``prepare``; the decode loop performs token-index
    lookups only.
    """

    kind = "multiview_pld"
    requires_text_context = False

    def __init__(
        self,
        tokenizer,
        *,
        max_matching_ngram_size: int = 10,
        max_draft_len: int = 128,
        min_matching_ngram_size: int = 1,
        transformed_min_matching_ngram_size: int = 4,
        exact_strong_min_len: int = 32,
        trans_len_margin: int = 0,
        max_views: int = 8,
    ):
        self.tokenizer = tokenizer
        self.max_matching_ngram_size = int(max_matching_ngram_size)
        self.max_draft_len = int(max_draft_len)
        self.min_matching_ngram_size = int(min_matching_ngram_size)
        self.transformed_min_matching_ngram_size = int(transformed_min_matching_ngram_size)
        self.exact_strong_min_len = int(exact_strong_min_len)
        self.trans_len_margin = int(trans_len_margin)
        self.max_views = int(max_views)
        self._exact = RootedPLDProposer(
            max_matching_ngram_size=self.max_matching_ngram_size,
            max_draft_len=self.max_draft_len,
            min_matching_ngram_size=self.min_matching_ngram_size,
        )
        self.reset()

    def reset(self) -> None:
        self._prepared = False
        self._rewrite_map: dict[str, str] = {}
        self._map_source = "explicit"
        self._indexes: list[_StaticTokenIndex] = []
        self._view_priors: dict[str, tuple[int, int, int]] = {}
        self._last_view_id: str | None = None
        self._disabled = False
        self._zero_accept_streak = 0
        self._pending_map_parse_us = 0.0
        self._pending_rewrite_apply_us = 0.0
        self._pending_tokenize_us = 0.0
        self._pending_index_build_us = 0.0

    def prepare(
        self,
        *,
        prompt_ids: list[int],
        prompt_text: str,
        reference: str,
        metadata: dict[str, Any] | None,
        prompt_len: int,
        language: str,
    ) -> None:
        del prompt_ids, metadata, prompt_len, language
        self._prepared = True
        t_map = time.perf_counter_ns()
        pairs = _rewrite_pairs(prompt_text)
        map_source = "explicit"
        self._rewrite_map = pairs
        self._map_source = map_source
        self._pending_map_parse_us = (time.perf_counter_ns() - t_map) / 1000.0
        if not reference or not pairs:
            return

        views: list[tuple[str, dict[str, str], str]] = []
        full = RewriteMapSet.from_pairs(pairs, source=map_source).as_dict()
        if full:
            views.append(("full", full, "transformed_reference"))
        for i, single in enumerate(RewriteMapSet.from_pairs(pairs, source=map_source).singles()):
            if len(views) >= self.max_views:
                break
            views.append((f"single{i}", single.as_dict(), "single_map_reference"))
        for i, derived in enumerate(_derived_field_views(pairs)):
            if len(views) >= self.max_views:
                break
            views.append((f"field{i}", derived, "field_normalized_reference"))

        t_apply = time.perf_counter_ns()
        materialized: list[tuple[str, str, str]] = []
        seen_texts: set[str] = set()
        for view_name, view_map, source_label in views:
            text = _apply_word_map(reference, view_map)
            if text == reference or text in seen_texts:
                continue
            seen_texts.add(text)
            materialized.append((view_name, text, source_label))
        self._pending_rewrite_apply_us = (time.perf_counter_ns() - t_apply) / 1000.0

        t_tok = time.perf_counter_ns()
        tokenized = [
            (view_name, encode_no_special(self.tokenizer, text), source_label)
            for view_name, text, source_label in materialized
        ]
        self._pending_tokenize_us = (time.perf_counter_ns() - t_tok) / 1000.0

        t_index = time.perf_counter_ns()
        self._indexes = []
        for view_name, toks, source_label in tokenized:
            if not toks:
                continue
            index = _StaticTokenIndex(
                toks,
                min_match_len=self.transformed_min_matching_ngram_size,
                max_match_len=self.max_matching_ngram_size,
                max_draft_len=self.max_draft_len,
                pool=f"{source_label}:{view_name}",
            )
            self._indexes.append(index)
        self._pending_index_build_us = (time.perf_counter_ns() - t_index) / 1000.0

    def _take_pending_timings(self) -> tuple[float, float, float, float]:
        timings = (
            self._pending_map_parse_us,
            self._pending_rewrite_apply_us,
            self._pending_tokenize_us,
            self._pending_index_build_us,
        )
        self._pending_map_parse_us = 0.0
        self._pending_rewrite_apply_us = 0.0
        self._pending_tokenize_us = 0.0
        self._pending_index_build_us = 0.0
        return timings

    def _view_acceptance_prior(self, view_id: str | None) -> float:
        if not view_id:
            return 0.5
        attempts, accepted, proposed = self._view_priors.get(view_id, (0, 0, 0))
        # Beta prior centered on moderate acceptance. This keeps new views from
        # being permanently suppressed before they have evidence.
        return (accepted + 4.0) / max(1.0, proposed + 8.0)

    def _adaptive_draft_len(self, view_id: str | None, match_len: int = 0) -> int:
        """Bound transformed-view drafts by empirical acceptance history.

        Exact PLD is already strong on real commits, so transformed views start
        with a modest budget and earn longer spans only after verified accepts.
        This mirrors suffix-decoding style adaptive speculation without adding
        any model-dependent probability estimates.
        """
        if not view_id:
            return min(self.max_draft_len, 8)
        attempts, accepted, proposed = self._view_priors.get(view_id, (0, 0, 0))
        if attempts <= 0:
            # A long key match is less likely to be a coincidental transformed
            # hit, but keep the first probe bounded.
            cold_cap = 32 if match_len >= self.max_matching_ngram_size else 16
            return min(self.max_draft_len, cold_cap)
        rate = accepted / max(1, proposed)
        if attempts >= 2 and rate < 0.08:
            return 0
        if rate >= 0.55:
            return self.max_draft_len
        if rate >= 0.25:
            return min(self.max_draft_len, 32)
        return min(self.max_draft_len, 8)

    def _trans_candidates(self, state: ProposerState) -> list[Proposal]:
        out: list[Proposal] = []
        for idx in self._indexes:
            draft_cap = self._adaptive_draft_len(idx.pool)
            if draft_cap <= 0:
                continue
            cand = idx.candidate(
                state.tokens_after_teacher,
                kind=self.kind,
                match_kind="multiview_transformed",
                substitution_count=len(self._rewrite_map),
                map_source=self._map_source,
                view_id=idx.pool,
                max_draft_len=draft_cap,
            )
            if cand is None:
                continue
            refined_cap = self._adaptive_draft_len(cand.view_id, cand.match_len)
            if refined_cap <= 0:
                continue
            if refined_cap < len(cand.tokens):
                cand = _clone_proposal(
                    cand,
                    tokens=cand.tokens[:refined_cap],
                    follow_end_token=(
                        cand.follow_start_token + refined_cap
                        if cand.follow_start_token is not None
                        else cand.follow_end_token
                    ),
                    text_preview=self.tokenizer.decode(cand.tokens[:refined_cap]),
                )
            prior = self._view_acceptance_prior(cand.view_id)
            expected = prior * len(cand.tokens)
            out.append(
                _clone_proposal(
                    cand,
                    kind=self.kind,
                    source_region="multiview_transformed",
                    route="transpld",
                    route_reason="multiview_candidate",
                    score=expected + cand.match_len / max(1, self.max_matching_ngram_size),
                )
            )
        return out

    def propose(self, state: ProposerState) -> Proposal | None:
        exact = self._exact.propose(state)
        if self._disabled:
            return exact
        if exact is not None and len(exact.tokens) >= self.exact_strong_min_len:
            return exact
        if not self._prepared:
            self.prepare(
                prompt_ids=state.prefix[: state.prompt_len],
                prompt_text=state.prompt_text,
                reference=state.reference,
                metadata=state.metadata,
                prompt_len=state.prompt_len,
                language=state.language,
            )
        candidates = self._trans_candidates(state)
        if not candidates:
            self._take_pending_timings()
            return exact
        best = max(
            candidates,
            key=lambda p: (
                p.score,
                p.match_len,
                len(p.tokens),
                p.follow_start_token or -1,
            ),
        )
        exact_is_meaningful = (
            exact is not None and exact.match_len >= self.transformed_min_matching_ngram_size
        )
        if exact_is_meaningful and len(best.tokens) < len(exact.tokens) + self.trans_len_margin:
            self._take_pending_timings()
            return exact
        map_us, apply_us, tok_us, index_us = self._take_pending_timings()
        self._last_view_id = best.view_id
        return _clone_proposal(
            best,
            map_parse_us=best.map_parse_us + map_us,
            rewrite_apply_us=best.rewrite_apply_us + apply_us,
            virtual_reference_tokenize_us=best.virtual_reference_tokenize_us + tok_us,
            transpld_index_build_us=best.transpld_index_build_us + index_us,
            rewrite_zero_accept_streak=self._zero_accept_streak,
            compound_view_count=len(self._indexes),
            active_map_count=len(self._rewrite_map),
            score=best.score + 0.25,
        )

    def observe(self, feedback: ProposalFeedback) -> None:
        if feedback.proposal_kind != self.kind:
            return
        view_id = self._last_view_id or feedback.proposal_match_kind or "unknown"
        attempts, accepted, proposed = self._view_priors.get(view_id, (0, 0, 0))
        accepted_nonroot = max(0, int(feedback.accepted_nonroot))
        proposed_nonroot = max(0, len(feedback.proposed_tokens))
        self._view_priors[view_id] = (
            attempts + 1,
            accepted + accepted_nonroot,
            proposed + proposed_nonroot,
        )
        if accepted_nonroot == 0:
            self._zero_accept_streak += 1
        else:
            self._zero_accept_streak = 0


class MultiViewTreePLDProposer(MultiViewPLDProposer):
    """Tree-verify competing exact-PLD and transformed-view candidates."""

    kind = "multiview_tree_pld"

    def __init__(self, tokenizer, *, max_tree_nodes: int = 192, **kwargs: Any):
        super().__init__(tokenizer, **kwargs)
        self.max_tree_nodes = int(max_tree_nodes)

    def propose(self, state: ProposerState) -> Proposal | ProposalTree | None:
        exact = self._exact.propose(state)
        if self._disabled:
            return exact
        if exact is not None and len(exact.tokens) >= self.exact_strong_min_len:
            return exact
        if not self._prepared:
            self.prepare(
                prompt_ids=state.prefix[: state.prompt_len],
                prompt_text=state.prompt_text,
                reference=state.reference,
                metadata=state.metadata,
                prompt_len=state.prompt_len,
                language=state.language,
            )
        trans = self._trans_candidates(state)
        if not trans:
            self._take_pending_timings()
            return exact
        best_trans = max(trans, key=lambda p: (p.score, p.match_len, len(p.tokens)))
        map_us, apply_us, tok_us, index_us = self._take_pending_timings()
        if exact is None:
            self._last_view_id = best_trans.view_id
            return _clone_proposal(
                best_trans,
                map_parse_us=best_trans.map_parse_us + map_us,
                rewrite_apply_us=best_trans.rewrite_apply_us + apply_us,
                virtual_reference_tokenize_us=best_trans.virtual_reference_tokenize_us + tok_us,
                transpld_index_build_us=best_trans.transpld_index_build_us + index_us,
                compound_view_count=len(self._indexes),
                active_map_count=len(self._rewrite_map),
            )
        candidates = [exact.tokens, best_trans.tokens]
        if tuple(candidates[0]) == tuple(candidates[1]):
            return exact
        nodes = build_candidate_prefix_tree(candidates, self.max_tree_nodes)
        if len(nodes) <= 0:
            return exact
        if len(nodes) >= self.max_tree_nodes and len(best_trans.tokens) <= len(exact.tokens):
            return exact
        self._last_view_id = best_trans.view_id
        return ProposalTree(
            kind=self.kind,
            candidates=candidates,
            scores=[exact.score, best_trans.score],
            sources=["exact_pld", best_trans.view_id or "transformed"],
            match_lens=[exact.match_len, best_trans.match_len],
            max_nodes=self.max_tree_nodes,
            score=max(exact.score, best_trans.score) + 0.15,
            source_start_token=best_trans.source_start_token,
            source_end_token=best_trans.source_end_token,
            follow_start_token=best_trans.follow_start_token,
            follow_end_token=best_trans.follow_end_token,
            query_len=max(exact.query_len or 0, best_trans.query_len or 0),
            pool="multiview_tree",
            source_region="multiview_tree",
            match_kind="multiview_tree",
            substitution_count=len(self._rewrite_map) or None,
            map_source=self._map_source,
            view_id=best_trans.view_id,
            compound_view_count=len(self._indexes),
            active_map_count=len(self._rewrite_map),
            route="tree",
            route_reason="exact_transpld_branch_competition",
            map_parse_us=map_us,
            rewrite_apply_us=apply_us,
            virtual_reference_tokenize_us=tok_us,
            transpld_index_build_us=index_us,
        )

    def observe(self, feedback: ProposalFeedback) -> None:
        if feedback.proposal_kind != self.kind:
            return
        view_id = self._last_view_id or "tree"
        attempts, accepted, proposed = self._view_priors.get(view_id, (0, 0, 0))
        accepted_nonroot = max(0, int(feedback.accepted_nonroot))
        proposed_nonroot = max(0, len(feedback.proposed_tokens))
        self._view_priors[view_id] = (
            attempts + 1,
            accepted + accepted_nonroot,
            proposed + proposed_nonroot,
        )
        if accepted_nonroot == 0:
            self._zero_accept_streak += 1
        else:
            self._zero_accept_streak = 0


def _derived_field_views(pairs: dict[str, str]) -> list[dict[str, str]]:
    views: list[dict[str, str]] = []
    for old, new in pairs.items():
        if "." not in old and "." not in new:
            continue
        old_parts = [p for p in old.split(".") if p]
        new_parts = [p for p in new.split(".") if p]
        if len(old_parts) >= 2 and len(new_parts) >= 2:
            views.append({old_parts[0]: new_parts[0], old_parts[-1]: new_parts[-1]})
        if old_parts and new_parts:
            views.append({old_parts[0]: new_parts[0]})
    dedup: list[dict[str, str]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for view in views:
        cleaned = RewriteMapSet.from_pairs(view).as_dict()
        key = tuple(sorted(cleaned.items()))
        if cleaned and key not in seen:
            seen.add(key)
            dedup.append(cleaned)
    return dedup


@dataclass(frozen=True)
class _RewriteFrontier:
    old: str
    new: str
    token_start: int
    token_end: int


class RoutedTransPLDProposer:
    """Mixed-regime deployment policy for transformed-view PLD.

    Exact PLD is the correct route for no-map/verbatim prompts.  TransPLD is
    used only when the prompt text exposes an effective rewrite map, and online
    back-off switches permanently to exact PLD when early transformed-view
    hits do not pay for their lookup cost.
    """

    kind = "routed_transpld"

    def __init__(
        self,
        tokenizer,
        *,
        max_matching_ngram_size: int = 10,
        max_draft_len: int = 128,
        min_matching_ngram_size: int = 1,
        transformed_min_matching_ngram_size: int = 4,
        backoff_after_steps: int = 8,
        min_rewrite_hits: int = 2,
        min_accept_per_rewrite_hit: float = 4.0,
        probe_backoff_after_attempts: int = 3,
        min_accept_per_rewrite_attempt: float = 2.0,
        low_accept_streak_limit: int = 3,
        low_accept_streak_threshold: int = 2,
        zero_accept_tripwire_limit: int = 3,
        exact_first_min_match_len: int = 16,
        rewrite_frontier_window: int = 32,
        unknown_probe_slack: int = 8,
        unknown_probe_max_draft_len: int = 32,
        trans_len_margin: int = 8,
    ):
        self.tokenizer = tokenizer
        self.max_matching_ngram_size = int(max_matching_ngram_size)
        self.max_draft_len = int(max_draft_len)
        self.min_matching_ngram_size = int(min_matching_ngram_size)
        self.transformed_min_matching_ngram_size = int(transformed_min_matching_ngram_size)
        self.backoff_after_steps = int(backoff_after_steps)
        self.min_rewrite_hits = int(min_rewrite_hits)
        self.min_accept_per_rewrite_hit = float(min_accept_per_rewrite_hit)
        self.probe_backoff_after_attempts = int(probe_backoff_after_attempts)
        self.min_accept_per_rewrite_attempt = float(min_accept_per_rewrite_attempt)
        self.low_accept_streak_limit = int(low_accept_streak_limit)
        self.low_accept_streak_threshold = int(low_accept_streak_threshold)
        self.zero_accept_tripwire_limit = int(zero_accept_tripwire_limit)
        self.exact_first_min_match_len = int(exact_first_min_match_len)
        self.rewrite_frontier_window = int(rewrite_frontier_window)
        self.unknown_probe_slack = int(unknown_probe_slack)
        self.unknown_probe_max_draft_len = int(unknown_probe_max_draft_len)
        self.trans_len_margin = int(trans_len_margin)
        self._rewrite = RewriteNormalizedPLDProposer(
            tokenizer,
            mode="vref",
            max_matching_ngram_size=self.max_matching_ngram_size,
            max_draft_len=self.max_draft_len,
            min_matching_ngram_size=self.transformed_min_matching_ngram_size,
        )
        self._exact = RootedPLDProposer(
            max_matching_ngram_size=self.max_matching_ngram_size,
            max_draft_len=self.max_draft_len,
            min_matching_ngram_size=self.min_matching_ngram_size,
        )
        self.reset()

    def reset(self) -> None:
        self._initialized = False
        self._route = "exact_pld"
        self._route_reason = "uninitialized"
        self._backoff_active = False
        self._reference = ""
        self._rewrite_map: dict[str, str] = {}
        self._map_source = "explicit"
        self._frontiers: list[_RewriteFrontier] = []
        self._blacklisted_frontiers: set[int] = set()
        self._adoption_state = "unknown"
        self._last_transition: str | None = None
        self._frontier_probes = 0
        self._adopted_frontiers = 0
        self._rejected_old_form_frontiers = 0
        self._disabled_by_adoption_gate = False
        self._root_old_match_count = 0
        self._root_new_match_count = 0
        self._steps_seen = 0
        self._rewrite_attempts = 0
        self._rewrite_hits = 0
        self._rewrite_accepted_nonroot = 0
        self._rewrite_accepted_nonroot_by_attempt = 0
        self._low_accept_streak = 0
        self._rewrite_zero_accepts = 0
        self._rewrite_consecutive_zero_accepts = 0
        self._pending_map_parse_us = 0.0
        self._rewrite.reset()

    @property
    def _accept_per_rewrite_hit(self) -> float:
        if self._rewrite_hits <= 0:
            return 0.0
        return self._rewrite_accepted_nonroot / max(1, self._rewrite_hits)

    @property
    def _accept_per_rewrite_attempt(self) -> float:
        if self._rewrite_attempts <= 0:
            return 0.0
        return self._rewrite_accepted_nonroot_by_attempt / max(1, self._rewrite_attempts)

    def _initialize_route(self, state: ProposerState) -> None:
        if self._initialized:
            return
        self._initialized = True
        t_map_0 = time.perf_counter_ns()
        _, reference, rewrite_map, map_source = _prompt_reference_and_map(self.tokenizer, state)
        self._pending_map_parse_us = (time.perf_counter_ns() - t_map_0) / 1000.0
        self._reference = reference
        self._rewrite_map = rewrite_map
        self._map_source = map_source
        transformed_reference = _apply_word_map(reference, rewrite_map) if rewrite_map else reference
        reference_tokens = encode_no_special(self.tokenizer, reference) if reference else []
        transformed_tokens = (
            encode_no_special(self.tokenizer, transformed_reference)
            if transformed_reference
            else []
        )
        decision = decide_prompt_only_saferoute(
            reference=reference,
            rewrite_map=rewrite_map,
            transformed_reference=transformed_reference,
            reference_tokens=reference_tokens,
            transformed_tokens=transformed_tokens,
        )
        if not decision.use_transpld:
            self._route = "exact_pld"
            self._route_reason = decision.reason or "no_rewrite_map"
        else:
            self._route = "transpld"
            self._route_reason = "prompt_rewrite_map"
            self._frontiers = self._build_frontiers(reference, rewrite_map)

    def _build_frontiers(self, reference: str, rewrite_map: dict[str, str]) -> list[_RewriteFrontier]:
        virtual_reference = _apply_word_map(reference, rewrite_map)
        out: list[_RewriteFrontier] = []
        for old, new in sorted(rewrite_map.items(), key=lambda item: -len(item[0])):
            if not old or not new:
                continue
            start = 0
            while True:
                idx = virtual_reference.find(new, start)
                if idx < 0:
                    break
                token_start = len(encode_no_special(self.tokenizer, virtual_reference[:idx]))
                token_end = len(
                    encode_no_special(self.tokenizer, virtual_reference[: idx + len(new)])
                )
                out.append(
                    _RewriteFrontier(
                        old=old,
                        new=new,
                        token_start=token_start,
                        token_end=max(token_start + 1, token_end),
                    )
                )
                start = idx + max(1, len(new))
        out.sort(key=lambda frontier: (frontier.token_start, frontier.token_end))
        return out

    def _next_frontier(self, proposal: Proposal) -> tuple[int | None, _RewriteFrontier | None]:
        if proposal.follow_start_token is None:
            return None, None
        follow_start = int(proposal.follow_start_token)
        best_i: int | None = None
        best_frontier: _RewriteFrontier | None = None
        best_dist: int | None = None
        for i, frontier in enumerate(self._frontiers):
            if i in self._blacklisted_frontiers:
                continue
            if frontier.token_end < follow_start:
                continue
            dist = max(0, frontier.token_start - follow_start)
            if best_dist is None or dist < best_dist:
                best_i = i
                best_frontier = frontier
                best_dist = dist
        return best_i, best_frontier

    def _frontier_distance(self, proposal: Proposal) -> int | None:
        _, frontier = self._next_frontier(proposal)
        if frontier is None or proposal.follow_start_token is None:
            return None
        return max(0, frontier.token_start - int(proposal.follow_start_token))

    def _proposal_crosses_frontier(
        self,
        proposal: Proposal | ProposalFeedback,
        *,
        accepted_len: int | None = None,
    ) -> tuple[int | None, _RewriteFrontier | None]:
        if proposal.follow_start_token is None:
            return None, None
        follow_start = int(proposal.follow_start_token)
        if accepted_len is None:
            accepted_len = len(proposal.tokens) if isinstance(proposal, Proposal) else proposal.accepted_nonroot
        follow_end = follow_start + max(0, int(accepted_len or 0))
        for i, frontier in enumerate(self._frontiers):
            if i in self._blacklisted_frontiers:
                continue
            if frontier.token_start < follow_end and frontier.token_end >= follow_start:
                return i, frontier
        return None, None

    def _generated_suffix_has(self, state: ProposerState, term: str) -> bool:
        if not term:
            return False
        _, generated = _split_prompt_generated(self.tokenizer, state)
        compact = re.sub(r"\s+", "", generated)
        return generated.endswith(term) or compact.endswith(term)

    def _set_adoption_state(self, state: str, reason: str) -> None:
        if state != self._adoption_state:
            self._last_transition = f"{self._adoption_state}->{state}:{reason}"
            self._adoption_state = state

    def _route_fields(self, *, frontier_distance: int | None = None) -> dict[str, Any]:
        return {
            "adoption_state": self._adoption_state,
            "adoption_transition": self._last_transition,
            "frontier_distance": frontier_distance,
            "frontier_probes": self._frontier_probes,
            "accepted_crossed_rewrite": self._adopted_frontiers,
            "rejected_old_form_frontiers": self._rejected_old_form_frontiers,
            "blacklisted_rewrite_occurrences": len(self._blacklisted_frontiers),
            "disabled_by_adoption_gate": self._disabled_by_adoption_gate,
            "root_old_match_count": self._root_old_match_count,
            "root_new_match_count": self._root_new_match_count,
        }

    def _with_route(
        self,
        proposal: Proposal,
        *,
        route: str,
        reason: str,
        match_kind: str,
        source_region: str,
        score_delta: float = 0.0,
        frontier_distance: int | None = None,
    ) -> Proposal:
        map_parse_us = self._pending_map_parse_us
        self._pending_map_parse_us = 0.0
        return _clone_proposal(
            proposal,
            kind=self.kind,
            match_kind=match_kind,
            source_region=source_region,
            route=route,
            route_reason=reason,
            backoff_active=self._backoff_active,
            rewrite_hit_count=self._rewrite_hits,
            route_window_accept_rate=self._accept_per_rewrite_attempt,
            rewrite_zero_accept_streak=self._rewrite_consecutive_zero_accepts,
            map_parse_us=proposal.map_parse_us + map_parse_us,
            score=proposal.score + score_delta,
            **self._route_fields(frontier_distance=frontier_distance),
        )

    def _cap_probe(self, proposal: Proposal, cap: int) -> Proposal:
        cap = max(0, int(cap))
        if cap <= 0:
            return replace(proposal, tokens=[])
        if len(proposal.tokens) <= cap:
            return proposal
        tokens = list(proposal.tokens[:cap])
        return replace(
            proposal,
            tokens=tokens,
            follow_end_token=(
                proposal.follow_start_token + cap
                if proposal.follow_start_token is not None
                else proposal.follow_end_token
            ),
            text_preview=decode_tokens(self.tokenizer, tokens)[:40],
        )

    def _exact_proposal(self, state: ProposerState, *, reason: str, match_kind: str) -> Proposal | None:
        exact = self._exact.propose(state)
        if exact is None:
            return None
        return self._with_route(
            exact,
            route="exact_pld",
            reason=reason,
            match_kind=match_kind,
            source_region=match_kind,
            score_delta=-0.02,
        )

    def propose(self, state: ProposerState) -> Proposal | None:
        self._initialize_route(state)
        if self._route != "transpld" or self._backoff_active or self._adoption_state == "rejected":
            reason = self._route_reason if not self._backoff_active else "online_backoff_low_acceptance"
            if self._adoption_state == "rejected" and not self._backoff_active:
                reason = "adoption_rejected_exact_pld"
            return self._exact_proposal(
                state,
                reason=reason,
                match_kind="routed_exact_pld" if not self._backoff_active else "routed_backoff_exact_pld",
            )
        exact = self._exact.propose(state)
        if exact is not None and exact.match_len >= self.exact_first_min_match_len:
            return self._with_route(
                exact,
                route="exact_pld",
                reason="exact_pld_strong",
                match_kind="routed_exact_pld_strong",
                source_region="routed_exact_pld_strong",
                score_delta=0.03,
            )
        proposal = self._rewrite._propose_vref(
            state,
            self._reference,
            self._rewrite_map,
            map_source=self._map_source,
        )
        if proposal is not None:
            if proposal.match_len < self.transformed_min_matching_ngram_size:
                raise AssertionError(
                    "Routed TransPLD emitted a transformed-view proposal below "
                    f"the minimum match length: {proposal.match_len} < "
                    f"{self.transformed_min_matching_ngram_size}"
                )
            frontier_i, frontier = self._next_frontier(proposal)
            frontier_dist = self._frontier_distance(proposal)
            if self._adoption_state == "unknown":
                if frontier is None or frontier_dist is None or frontier_dist > self.rewrite_frontier_window:
                    if exact is not None:
                        return self._with_route(
                            exact,
                            route="exact_pld",
                            reason="before_rewrite_frontier",
                            match_kind="routed_exact_before_frontier",
                            source_region="routed_exact_before_frontier",
                            frontier_distance=frontier_dist,
                            score_delta=0.02,
                        )
                    return None
                self._frontier_probes += 1
                if self._generated_suffix_has(state, frontier.old):
                    if frontier_i is not None:
                        self._blacklisted_frontiers.add(frontier_i)
                    self._root_old_match_count += 1
                    self._rejected_old_form_frontiers += 1
                    if self._rejected_old_form_frontiers >= 2 and self._adopted_frontiers == 0:
                        self._disabled_by_adoption_gate = True
                        self._backoff_active = True
                        self._route_reason = "adoption_gate_old_form"
                        self._set_adoption_state("rejected", "root_old_form")
                    if exact is not None:
                        return self._with_route(
                            exact,
                            route="exact_pld",
                            reason="root_old_form_at_frontier",
                            match_kind="routed_exact_old_frontier",
                            source_region="routed_exact_old_frontier",
                            frontier_distance=frontier_dist,
                            score_delta=0.02,
                        )
                    return None
                if self._generated_suffix_has(state, frontier.new):
                    self._root_new_match_count += 1
                    self._adopted_frontiers += 1
                    self._set_adoption_state("adopted", "root_new_form")
                else:
                    cap = min(
                        self.unknown_probe_max_draft_len,
                        frontier_dist + self.unknown_probe_slack,
                    )
                    proposal = self._cap_probe(proposal, cap)
                    if not proposal.tokens:
                        return self._exact_proposal(
                            state,
                            reason="empty_frontier_probe",
                            match_kind="routed_empty_probe_exact_pld",
                        )
                    if (
                        exact is not None
                        and len(proposal.tokens) < len(exact.tokens) + self.trans_len_margin
                    ):
                        return self._with_route(
                            exact,
                            route="exact_pld",
                            reason="unknown_exact_len_margin",
                            match_kind="routed_unknown_exact_margin",
                            source_region="routed_unknown_exact_margin",
                            frontier_distance=frontier_dist,
                            score_delta=0.01,
                        )
            elif (
                self._adoption_state == "adopted"
                and exact is not None
                and len(exact.tokens) >= len(proposal.tokens)
            ):
                return self._with_route(
                    exact,
                    route="exact_pld",
                    reason="adopted_exact_best_of",
                    match_kind="routed_adopted_exact_best",
                    source_region="routed_adopted_exact_best",
                    frontier_distance=frontier_dist,
                    score_delta=0.02,
                )
            return self._with_route(
                proposal,
                route="transpld",
                reason=self._route_reason if self._adoption_state != "unknown" else "unknown_frontier_probe",
                match_kind="routed_transpld_vref",
                source_region="routed_transpld",
                frontier_distance=frontier_dist,
                score_delta=0.25,
            )
        if exact is not None:
            return self._with_route(
                exact,
                route="exact_pld",
                reason="transpld_miss",
                match_kind="routed_transpld_miss_exact_pld",
                source_region="routed_transpld_miss_exact_pld",
                score_delta=-0.02,
            )
        return None

    def observe(self, feedback: ProposalFeedback) -> None:
        if not self._initialized:
            return
        self._steps_seen += 1
        attempted_transpld = (
            feedback.proposal_kind == self.kind
            and feedback.proposal_match_kind
            in {"routed_transpld_vref", "routed_transpld_bidir", "routed_transpld_miss_exact_pld"}
        )
        attempted_rewrite_probe = (
            feedback.proposal_kind == self.kind
            and feedback.proposal_match_kind in {"routed_transpld_vref", "routed_transpld_bidir"}
        )
        accepted_nonroot = max(0, int(feedback.accepted_nonroot))
        if attempted_transpld:
            self._rewrite_attempts += 1
            self._rewrite_accepted_nonroot_by_attempt += (
                accepted_nonroot
                if attempted_rewrite_probe
                else 0
            )
            if attempted_rewrite_probe:
                if accepted_nonroot == 0:
                    self._rewrite_zero_accepts += 1
                    self._rewrite_consecutive_zero_accepts += 1
                else:
                    self._rewrite_consecutive_zero_accepts = 0
            if accepted_nonroot <= self.low_accept_streak_threshold:
                self._low_accept_streak += 1
            else:
                self._low_accept_streak = 0
        if (
            feedback.proposal_kind == self.kind
            and feedback.proposal_match_kind in {"routed_transpld_vref", "routed_transpld_bidir"}
        ):
            self._rewrite_hits += 1
            self._rewrite_accepted_nonroot += accepted_nonroot
            crossed_i, crossed_frontier = self._proposal_crosses_frontier(
                feedback,
                accepted_len=accepted_nonroot,
            )
            if crossed_frontier is not None and accepted_nonroot > 0:
                self._adopted_frontiers += 1
                self._set_adoption_state("adopted", "accepted_crossed_rewrite")
            if feedback.rejected:
                reject_i, reject_frontier = self._proposal_crosses_frontier(
                    feedback,
                    accepted_len=accepted_nonroot + 1,
                )
                emitted_text = decode_tokens(self.tokenizer, feedback.emitted_tokens)
                if reject_frontier is not None and reject_frontier.old in emitted_text:
                    if reject_i is not None:
                        self._blacklisted_frontiers.add(reject_i)
                    self._rejected_old_form_frontiers += 1
                elif reject_frontier is not None and reject_frontier.new in emitted_text:
                    self._adopted_frontiers += 1
                    self._set_adoption_state("adopted", "target_emitted_new_form")
            if self._rejected_old_form_frontiers >= 2 and self._adopted_frontiers == 0:
                self._disabled_by_adoption_gate = True
                self._backoff_active = True
                self._route_reason = "adoption_gate_old_form"
                self._set_adoption_state("rejected", "old_form_rejections")
        if (
            self._route == "transpld"
            and not self._backoff_active
            and self._rewrite_attempts > 0
            and (
                (
                    self._rewrite_attempts >= self.probe_backoff_after_attempts
                    and (
                        self._accept_per_rewrite_attempt < self.min_accept_per_rewrite_attempt
                        or (
                            self._rewrite_zero_accepts >= 2
                            and self._accept_per_rewrite_attempt < self.min_accept_per_rewrite_hit
                        )
                    )
                )
                or self._low_accept_streak >= self.low_accept_streak_limit
                or (
                    self.zero_accept_tripwire_limit > 0
                    and self._rewrite_consecutive_zero_accepts >= self.zero_accept_tripwire_limit
                )
                or (
                    self._steps_seen >= self.backoff_after_steps
                    and (
                        self._rewrite_hits < self.min_rewrite_hits
                        or self._accept_per_rewrite_hit < self.min_accept_per_rewrite_hit
                    )
                )
            )
        ):
            self._backoff_active = True
            self._route_reason = "online_backoff_low_acceptance"
            self._disabled_by_adoption_gate = True
            self._set_adoption_state("rejected", "low_acceptance")


class SimpleAdoptionTransPLDProposer(RoutedTransPLDProposer):
    """Cheap adoption-gated rewrite-view router.

    This is the reviewer-suggested first-cut policy: exact PLD is the default,
    TransPLD is enabled only when the generated prefix shows evidence that the
    target has adopted the rewrite map, or when the transformed proposal is
    clearly longer than exact PLD during the unknown state.  It is intentionally
    simpler than the frontier-state router above so the two policies can be
    evaluated independently.
    """

    kind = "adopt_simple_transpld"

    def __init__(
        self,
        tokenizer,
        *,
        adoption_threshold: float = 0.6,
        reject_threshold: float = 0.25,
        min_agreement_observations: int = 1,
        **kwargs: Any,
    ):
        super().__init__(tokenizer, **kwargs)
        self.adoption_threshold = float(adoption_threshold)
        self.reject_threshold = float(reject_threshold)
        self.min_agreement_observations = int(min_agreement_observations)

    def _agreement_counts(self, state: ProposerState) -> tuple[int, int]:
        if not self._rewrite_map:
            return 0, 0
        _, generated = _split_prompt_generated(self.tokenizer, state)
        old_hits = 0
        new_hits = 0
        for old, new in self._rewrite_map.items():
            if old:
                old_hits += generated.count(old)
            if new:
                new_hits += generated.count(new)
        return old_hits, new_hits

    def _update_adoption_from_prefix(self, state: ProposerState) -> None:
        if self._adoption_state != "unknown":
            return
        old_hits, new_hits = self._agreement_counts(state)
        total = old_hits + new_hits
        if total < self.min_agreement_observations:
            return
        score = new_hits / max(1, total)
        if score >= self.adoption_threshold:
            self._root_new_match_count = max(self._root_new_match_count, new_hits)
            self._adopted_frontiers += 1
            self._set_adoption_state("adopted", "prefix_rewrite_agreement")
        elif score <= self.reject_threshold and old_hits >= self.min_agreement_observations:
            self._root_old_match_count = max(self._root_old_match_count, old_hits)
            self._rejected_old_form_frontiers += 1
            self._disabled_by_adoption_gate = True
            self._backoff_active = True
            self._route_reason = "adoption_gate_prefix_old_form"
            self._set_adoption_state("rejected", "prefix_old_form_agreement")

    def propose(self, state: ProposerState) -> Proposal | None:
        self._initialize_route(state)
        if self._route != "transpld":
            return self._exact_proposal(
                state,
                reason=self._route_reason,
                match_kind="routed_exact_pld",
            )
        self._update_adoption_from_prefix(state)
        if self._backoff_active or self._adoption_state == "rejected":
            return self._exact_proposal(
                state,
                reason=self._route_reason if self._route_reason else "adoption_rejected_exact_pld",
                match_kind="routed_backoff_exact_pld",
            )

        exact = self._exact.propose(state)
        if exact is not None and exact.match_len >= self.exact_first_min_match_len and self._adoption_state != "adopted":
            return self._with_route(
                exact,
                route="exact_pld",
                reason="exact_pld_strong",
                match_kind="routed_exact_pld_strong",
                source_region="routed_exact_pld_strong",
                score_delta=0.03,
            )

        proposal = self._rewrite._propose_vref(
            state,
            self._reference,
            self._rewrite_map,
            map_source=self._map_source,
        )
        if proposal is None:
            if exact is not None:
                return self._with_route(
                    exact,
                    route="exact_pld",
                    reason="transpld_miss",
                    match_kind="routed_transpld_miss_exact_pld",
                    source_region="routed_transpld_miss_exact_pld",
                    score_delta=-0.02,
                )
            return None
        if proposal.match_len < self.transformed_min_matching_ngram_size:
            raise AssertionError(
                "Simple adoption TransPLD emitted a transformed-view proposal below "
                f"the minimum match length: {proposal.match_len} < "
                f"{self.transformed_min_matching_ngram_size}"
            )

        if self._adoption_state == "unknown":
            proposal = self._cap_probe(proposal, self.unknown_probe_max_draft_len)
            if (
                exact is not None
                and len(proposal.tokens) < len(exact.tokens) + self.trans_len_margin
            ):
                return self._with_route(
                    exact,
                    route="exact_pld",
                    reason="unknown_exact_len_margin",
                    match_kind="routed_unknown_exact_margin",
                    source_region="routed_unknown_exact_margin",
                    score_delta=0.01,
                )
        elif self._adoption_state == "adopted" and exact is not None and len(exact.tokens) >= len(proposal.tokens):
            return self._with_route(
                exact,
                route="exact_pld",
                reason="adopted_exact_best_of",
                match_kind="routed_adopted_exact_best",
                source_region="routed_adopted_exact_best",
                score_delta=0.02,
            )

        return self._with_route(
            proposal,
            route="transpld",
            reason="adopted_transpld" if self._adoption_state == "adopted" else "unknown_length_probe",
            match_kind="routed_transpld_vref",
            source_region="routed_transpld",
            score_delta=0.25,
        )


def _term_occurrences(text: str) -> list[tuple[str, int]]:
    pattern = re.compile(
        r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*|"
        r"[0-9]+(?:\.[0-9]+)?|"
        r"(?P<quote>['\"])(?:\\.|(?!\1).)*\1"
    )
    return [(m.group(0), m.start()) for m in pattern.finditer(text)]


def _is_rewrite_term(value: str) -> bool:
    if not value or value in _ALL_KEYWORDS:
        return False
    return bool(
        re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", value)
        or re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value)
        or _STRING_RE.fullmatch(value)
    )


def _context_signature(text: str, start: int, term: str, radius: int = 24) -> str:
    lo = max(0, start - radius)
    hi = min(len(text), start + len(term) + radius)
    snippet = text[lo:hi]
    return snippet.replace(term, "$X")


class OnlineRewriteMapInferer:
    """Conservative online map miner from repeated generated/reference mismatches."""

    def __init__(self, tokenizer, *, min_support: int = 2):
        self.tokenizer = tokenizer
        self.min_support = int(min_support)
        self.reset()

    def reset(self) -> None:
        self.support: Counter[tuple[str, str]] = Counter()
        self.active: dict[str, str] = {}
        self.confidence: float = 0.0

    def observe(self, feedback: ProposalFeedback) -> None:
        return None

    def update(self, state: ProposerState, reference: str, explicit_map: dict[str, str] | None = None) -> None:
        if not reference:
            return
        _, generated = _split_prompt_generated(self.tokenizer, state)
        if len(generated) < 8:
            return
        explicit_map = explicit_map or {}
        reference_terms = [
            (term, pos)
            for term, pos in _term_occurrences(reference)
            if _is_rewrite_term(term) and term not in explicit_map
        ]
        generated_terms = [
            (term, pos)
            for term, pos in _term_occurrences(generated)
            if _is_rewrite_term(term) and term not in explicit_map.values()
        ]
        if not reference_terms or not generated_terms:
            return
        generated_vocab = {term for term, _ in generated_terms}
        reference_vocab = {term for term, _ in reference_terms}
        candidate_refs = [(term, pos) for term, pos in reference_terms if term not in generated_vocab]
        candidate_gens = [(term, pos) for term, pos in generated_terms if term not in reference_vocab]
        for old, old_pos in candidate_refs:
            old_sig = _context_signature(reference, old_pos, old)
            for new, new_pos in candidate_gens:
                if old == new:
                    continue
                new_sig = _context_signature(generated, new_pos, new)
                if old_sig == new_sig or _shape_signature(old_sig) == _shape_signature(new_sig):
                    self.support[(old, new)] += 1
        ref_counts = Counter(term for term, _ in candidate_refs)
        gen_counts = Counter(term for term, _ in candidate_gens)
        for old, old_count in ref_counts.items():
            old_suffix = old.split(".", 1)[1] if "." in old else ""
            for new, new_count in gen_counts.items():
                if old == new:
                    continue
                new_suffix = new.split(".", 1)[1] if "." in new else ""
                same_shape = (
                    old_suffix
                    and old_suffix == new_suffix
                    and old.split(".", 1)[0] != new.split(".", 1)[0]
                )
                if same_shape or _shape_signature(old) == _shape_signature(new):
                    self.support[(old, new)] += min(old_count, new_count)
                    if same_shape:
                        old_base = old.split(".", 1)[0]
                        new_base = new.split(".", 1)[0]
                        if old_base and new_base and old_base != new_base:
                            self.support[(old_base, new_base)] += min(old_count, new_count)
        self._activate()

    def _activate(self) -> None:
        for (old, new), count in self.support.most_common():
            if count < self.min_support:
                continue
            if old in self.active and self.active[old] != new:
                continue
            if new in self.active.values() and self.active.get(old) != new:
                continue
            self.active[old] = new
        if self.active:
            self.confidence = min(1.0, max(self.support.values()) / max(1, self.min_support + 2))


def _shape_signature(text: str) -> str:
    text = re.sub(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", "ID", text)
    text = re.sub(r"[0-9]+(?:\.[0-9]+)?", "NUM", text)
    text = _STRING_RE.sub("STR", text)
    return text


class TransPLDInferenceProposer:
    kind = "transpld_infer"

    def __init__(
        self,
        tokenizer,
        *,
        max_matching_ngram_size: int = 10,
        max_draft_len: int = 128,
        min_matching_ngram_size: int = 1,
        infer_only: bool = False,
        compound: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_matching_ngram_size = int(max_matching_ngram_size)
        self.max_draft_len = int(max_draft_len)
        self.min_matching_ngram_size = int(min_matching_ngram_size)
        self.infer_only = bool(infer_only)
        self.compound = bool(compound)
        self._inferer = OnlineRewriteMapInferer(tokenizer)
        self._rewrite = RewriteNormalizedPLDProposer(
            tokenizer,
            mode="bidir",
            max_matching_ngram_size=self.max_matching_ngram_size,
            max_draft_len=self.max_draft_len,
            min_matching_ngram_size=self.min_matching_ngram_size,
        )
        self._exact = RootedPLDProposer(
            max_matching_ngram_size=self.max_matching_ngram_size,
            max_draft_len=self.max_draft_len,
            min_matching_ngram_size=self.min_matching_ngram_size,
        )

    def reset(self) -> None:
        self._inferer.reset()

    def observe(self, feedback: ProposalFeedback) -> None:
        self._inferer.observe(feedback)

    def propose(self, state: ProposerState) -> Proposal | None:
        prompt_text, reference, explicit, _ = _prompt_reference_and_map(self.tokenizer, state)
        explicit = {} if self.infer_only else explicit
        self._inferer.update(state, reference, explicit)
        inferred = dict(self._inferer.active)
        maps = dict(explicit)
        maps.update({k: v for k, v in inferred.items() if k not in maps})
        map_source = (
            "mixed"
            if explicit and inferred
            else "explicit"
            if explicit
            else "inferred"
            if inferred
            else None
        )
        proposal = self._proposal_for_maps(state, reference, maps, map_source)
        if proposal is not None:
            return proposal
        exact = self._exact.propose(state)
        if exact is None:
            return None
        return _clone_proposal(
            exact,
            kind=self.kind,
            match_kind="transpld_exact_fallback",
            source_region="transpld_exact_fallback",
            map_source=map_source,
            inferred_map_count=len(inferred) or None,
            inference_confidence=self._inferer.confidence if inferred else None,
            view_id="identity",
            active_map_count=len(maps),
            score=exact.score - 0.05,
        )

    def _proposal_for_maps(
        self,
        state: ProposerState,
        reference: str,
        maps: dict[str, str],
        map_source: str | None,
    ) -> Proposal | None:
        if not reference or not maps:
            return None
        candidates: list[Proposal] = []
        map_sets = [RewriteMapSet.from_pairs(maps, source=map_source or "unknown")]
        if self.compound:
            map_sets.extend(map_sets[0].singles()[:7])
        for rewrite_set in map_sets[:8]:
            proposal = self._rewrite._propose_bidir(
                state,
                reference,
                rewrite_set.as_dict(),
                map_source=rewrite_set.source,
            )
            if proposal is None:
                continue
            candidates.append(
                _clone_proposal(
                    proposal,
                    kind=self.kind,
                    match_kind="transpld_bidir_inferred" if map_source == "inferred" else "transpld_bidir",
                    source_region="transpld",
                    map_source=map_source,
                    inferred_map_count=len(self._inferer.active) or None,
                    inference_confidence=self._inferer.confidence if self._inferer.active else None,
                    compound_view_count=len(map_sets) if self.compound else None,
                    active_map_count=rewrite_set.active_count,
                    score=proposal.score + (0.35 if rewrite_set.active_count > 1 else 0.25),
                )
            )
        if not candidates:
            return None
        return max(candidates, key=lambda p: (p.score, p.match_len, len(p.tokens), p.active_map_count or 0))


class CompoundTransPLDProposer(TransPLDInferenceProposer):
    kind = "transpld_compound"

    def __init__(self, tokenizer, **kwargs: Any):
        super().__init__(tokenizer, compound=True, infer_only=False, **kwargs)


class CursorTrackedTransPLDProposer:
    kind = "transpld_cursor"

    def __init__(
        self,
        tokenizer,
        *,
        max_matching_ngram_size: int = 10,
        max_draft_len: int = 128,
        max_cursor_draft_len: int = 256,
        activation_accept: int = 8,
        infer: bool = False,
        compound: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_cursor_draft_len = int(max_cursor_draft_len)
        self.activation_accept = int(activation_accept)
        self._base = TransPLDInferenceProposer(
            tokenizer,
            max_matching_ngram_size=max_matching_ngram_size,
            max_draft_len=max_draft_len,
            infer_only=False,
            compound=compound,
        ) if infer or compound else TransPLDProposer(
            tokenizer,
            max_matching_ngram_size=max_matching_ngram_size,
            max_draft_len=max_draft_len,
        )
        self.reset()

    def reset(self) -> None:
        reset = getattr(self._base, "reset", None)
        if callable(reset):
            reset()
        self._cursor_tokens: list[int] = []
        self._cursor_pos: int | None = None
        self._cursor_confidence: float = 0.0
        self._cursor_view_id: str | None = None
        self._last_state: ProposerState | None = None
        self._last_proposal: Proposal | None = None

    def propose(self, state: ProposerState) -> Proposal | None:
        self._last_state = state
        if self._cursor_pos is not None and self._cursor_pos < len(self._cursor_tokens):
            tokens = self._cursor_tokens[self._cursor_pos : self._cursor_pos + self.max_cursor_draft_len]
            if tokens:
                proposal = Proposal(
                    kind=self.kind,
                    tokens=tokens,
                    match_len=0,
                    score=9.0 + min(2.0, len(tokens) / 64.0),
                    text_preview=decode_tokens(self.tokenizer, tokens[:32]),
                    follow_start_token=self._cursor_pos,
                    follow_end_token=self._cursor_pos + len(tokens),
                    pool="cursor_value",
                    source_region="cursor",
                    root_included=False,
                    match_kind="cursor",
                    cursor_pos=self._cursor_pos,
                    cursor_confidence=self._cursor_confidence,
                    cursor_resync=False,
                    view_id=self._cursor_view_id,
                )
                self._last_proposal = proposal
                return proposal
        proposal = self._base.propose(state)
        if proposal is None:
            self._last_proposal = None
            return None
        proposal = _clone_proposal(
            proposal,
            kind=self.kind,
            cursor_resync=proposal.match_kind not in {"transpld_exact_fallback", "rooted_pld"},
            cursor_confidence=self._cursor_confidence or None,
        )
        self._last_proposal = proposal
        return proposal

    def observe(self, feedback: ProposalFeedback) -> None:
        observe = getattr(self._base, "observe", None)
        if callable(observe):
            observe(feedback)
        proposal = self._last_proposal
        if proposal is None:
            self._deactivate()
            return
        if feedback.proposal_match_kind == "cursor":
            matched = _matching_prefix_len(proposal.tokens, feedback.emitted_tokens[1:])
            if matched <= 0:
                self._deactivate()
                return
            self._cursor_pos = (self._cursor_pos or 0) + matched
            self._cursor_confidence = min(1.0, self._cursor_confidence + 0.05)
            if feedback.rejected and matched < len(proposal.tokens):
                self._deactivate()
            return
        if feedback.accepted_nonroot < self.activation_accept:
            if feedback.rejected:
                self._deactivate()
            return
        self._resync_cursor(feedback)

    def _resync_cursor(self, feedback: ProposalFeedback) -> None:
        if self._last_state is None:
            return
        prompt_text, reference, rewrite_map, map_source = _prompt_reference_and_map(self.tokenizer, self._last_state)
        if not reference or not rewrite_map:
            self._deactivate()
            return
        value_text = _apply_word_map(reference, rewrite_map)
        self._cursor_tokens = encode_no_special(self.tokenizer, value_text)
        _, generated_before = _split_prompt_generated(self.tokenizer, self._last_state)
        generated_after = generated_before + decode_tokens(self.tokenizer, feedback.emitted_tokens)
        suffix = _longest_suffix_in_text(generated_after, value_text)
        if suffix is None:
            self._deactivate()
            return
        idx, suffix_text = suffix
        self._cursor_pos = len(encode_no_special(self.tokenizer, value_text[: idx + len(suffix_text)]))
        self._cursor_confidence = 0.7
        self._cursor_view_id = f"cursor:{map_source}:{_stable_view_key(rewrite_map)}"

    def _deactivate(self) -> None:
        self._cursor_pos = None
        self._cursor_confidence = 0.0
        self._cursor_view_id = None


def _matching_prefix_len(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if int(a) != int(b):
            break
        count += 1
    return count


def _longest_suffix_in_text(generated: str, reference: str, *, max_chars: int = 256) -> tuple[int, str] | None:
    if not generated or not reference:
        return None
    tail = generated[-max_chars:]
    best: tuple[int, str] | None = None
    for size in range(min(len(tail), max_chars), 3, -1):
        suffix = tail[-size:]
        idx = reference.rfind(suffix)
        if idx >= 0:
            best = (idx, suffix)
            break
    return best


@dataclass(frozen=True)
class SuffixCandidate:
    tokens: list[int]
    score: float
    match_len: int
    source_start: int
    source_end: int
    follow_start: int
    follow_end: int
    pool: str
    frequency: int


def adaptive_suffix_draft_len(match_len: int, max_draft_len: int) -> int:
    if match_len >= 12:
        return min(16, max_draft_len)
    if match_len >= 8:
        return min(8, max_draft_len)
    if match_len >= 4:
        return min(5, max_draft_len)
    return min(3, max_draft_len)


def adaptive_spine_draft_len(match_len: int, max_draft_len: int) -> int:
    if match_len >= 12:
        return min(32, max_draft_len)
    if match_len >= 8:
        return min(16, max_draft_len)
    if match_len >= 4:
        return min(8, max_draft_len)
    return min(3, max_draft_len)


def build_candidate_prefix_tree(
    candidates: list[list[int]],
    max_nodes: int,
) -> list[ProposalTreeNode]:
    """Build a parent-before-child prefix tree over root-excluded candidates."""
    nodes: list[ProposalTreeNode] = []
    children: dict[tuple[int, int], int] = {}
    if max_nodes <= 0:
        return nodes
    for cand in candidates:
        parent = -1
        depth = 0
        for token in cand:
            key = (parent, int(token))
            node_idx = children.get(key)
            if node_idx is None:
                if len(nodes) >= max_nodes:
                    break
                node_idx = len(nodes)
                depth = 1 if parent < 0 else nodes[parent].depth + 1
                nodes.append(ProposalTreeNode(token=int(token), parent=parent, depth=depth))
                children[key] = node_idx
            parent = node_idx
    return nodes


def _tree_node_count(candidates: list[list[int]], max_nodes: int) -> int:
    return len(build_candidate_prefix_tree(candidates, max_nodes))


class MultiSourceSuffixIndex:
    """Exact suffix index over prompt/generated token pools.

    The implementation rebuilds per prefix version.  That is still cheap for
    the benchmark sequence lengths here and gives us the same API as a rolling
    index, which can be optimized later without changing the decoder.
    """

    def __init__(self, key_lengths: Iterable[int] = (3, 4, 5, 6, 8, 12, 16)):
        self.key_lengths = tuple(sorted({int(k) for k in key_lengths if int(k) > 0}))
        self._seq_key: tuple[int, ...] | None = None
        self._index: dict[int, dict[tuple[int, ...], list[int]]] = {}

    def reset(self) -> None:
        self._seq_key = None
        self._index = {}

    def ensure(self, seq: list[int]) -> None:
        key = tuple(int(t) for t in seq)
        if key == self._seq_key:
            return
        self._seq_key = key
        self._index = {k: {} for k in self.key_lengths}
        for k in self.key_lengths:
            if len(seq) < k:
                continue
            table = self._index[k]
            for start in range(0, len(seq) - k + 1):
                table.setdefault(tuple(seq[start : start + k]), []).append(start)

    def query(
        self,
        seq: list[int],
        *,
        prompt_len: int,
        top_k: int = 4,
        max_draft_len: int = 16,
        pool: str = "local",
    ) -> list[SuffixCandidate]:
        if pool not in {"local", "prompt", "generated"}:
            raise ValueError(f"unsupported multisuffix pool: {pool}")
        if top_k < 1:
            return []
        self.ensure(seq)
        dedup: dict[tuple[int, ...], SuffixCandidate] = {}
        current_end = len(seq)
        for match_len in sorted(self.key_lengths, reverse=True):
            if match_len >= current_end:
                continue
            current_start = current_end - match_len
            needle = tuple(seq[current_start:current_end])
            starts = self._index.get(match_len, {}).get(needle, [])
            if not starts:
                continue
            allowed = [
                start
                for start in starts
                if start + match_len <= current_start
                and self._start_allowed(start, match_len, prompt_len, pool)
            ]
            if not allowed:
                continue
            frequency = len(allowed)
            for start in allowed:
                follow_start = start + match_len
                draft_cap = adaptive_suffix_draft_len(match_len, max_draft_len)
                follow_end = min(follow_start + draft_cap, current_start)
                if follow_start >= follow_end:
                    continue
                tokens = list(seq[follow_start:follow_end])
                key = tuple(tokens)
                candidate_pool = self._pool_for(start, follow_end, prompt_len)
                score = self._score(
                    match_len=match_len,
                    pool=candidate_pool,
                    start=start,
                    current_start=current_start,
                    frequency=frequency,
                )
                existing = dedup.get(key)
                if existing is None or score > existing.score:
                    dedup[key] = SuffixCandidate(
                        tokens=tokens,
                        score=score,
                        match_len=match_len,
                        source_start=start,
                        source_end=start + match_len,
                        follow_start=follow_start,
                        follow_end=follow_end,
                        pool=candidate_pool,
                        frequency=frequency,
                    )
        return sorted(
            dedup.values(),
            key=lambda c: (-c.score, -c.match_len, -len(c.tokens), -c.source_start),
        )[:top_k]

    def _start_allowed(self, start: int, match_len: int, prompt_len: int, pool: str) -> bool:
        if pool == "local":
            return True
        end = start + match_len
        if pool == "prompt":
            return end <= prompt_len
        if pool == "generated":
            return start >= prompt_len
        return True

    def _pool_for(self, start: int, follow_end: int, prompt_len: int) -> str:
        if follow_end <= prompt_len:
            return "prompt"
        if start >= prompt_len:
            return "generated"
        return "current_file"

    def _score(
        self,
        *,
        match_len: int,
        pool: str,
        start: int,
        current_start: int,
        frequency: int,
    ) -> float:
        pool_bonus = {
            "generated": 4.0,
            "current_file": 3.0,
            "edit_history": 3.0,
            "prompt": 2.0,
            "repo_sibling": 1.0,
        }.get(pool, 1.0)
        recency = 0.0 if current_start <= 0 else 2.0 * (start / current_start)
        frequency_bonus = min(2.0, frequency / 4.0)
        return match_len * 4.0 + pool_bonus + recency + frequency_bonus


class MultiSuffixProposer:
    """Adaptive exact suffix reuse with optional multi-candidate tree output."""

    def __init__(
        self,
        *,
        kind: str = "multi_suffix",
        key_lengths: Iterable[int] = (3, 4, 5, 6, 8, 12, 16),
        top_k: int = 4,
        max_draft_len: int = 16,
        max_tree_nodes: int = 12,
        tree: bool = True,
        pool: str = "local",
    ):
        self.kind = kind
        self.key_lengths = tuple(sorted({int(k) for k in key_lengths if int(k) > 0}))
        self.top_k = max(1, int(top_k))
        self.max_draft_len = int(max_draft_len)
        self.max_tree_nodes = int(max_tree_nodes)
        self.tree = bool(tree)
        self.pool = pool
        self.index = MultiSourceSuffixIndex(self.key_lengths)

    def propose(self, state: ProposerState) -> Proposal | ProposalTree | None:
        seq = state.tokens_after_teacher
        top_k = self.top_k if self.tree else 1
        candidates = self.index.query(
            seq,
            prompt_len=state.prompt_len,
            top_k=top_k,
            max_draft_len=self.max_draft_len,
            pool=self.pool,
        )
        if not candidates:
            return None
        if not self.tree:
            cand = candidates[0]
            return Proposal(
                kind=self.kind,
                tokens=cand.tokens,
                match_len=cand.match_len,
                score=cand.score,
                source_start_token=cand.source_start,
                source_end_token=cand.source_end,
                follow_start_token=cand.follow_start,
                follow_end_token=cand.follow_end,
                query_len=cand.match_len,
                pool=cand.pool,
                root_included=False,
                match_kind="multi_chain",
            )

        selected: list[SuffixCandidate] = []
        for cand in candidates:
            trial = [c.tokens for c in selected] + [cand.tokens]
            if _tree_node_count(trial, self.max_tree_nodes) <= self.max_tree_nodes:
                selected.append(cand)
                continue
            if not selected:
                truncated = SuffixCandidate(
                    tokens=cand.tokens[: self.max_tree_nodes],
                    score=cand.score,
                    match_len=cand.match_len,
                    source_start=cand.source_start,
                    source_end=cand.source_end,
                    follow_start=cand.follow_start,
                    follow_end=min(cand.follow_start + self.max_tree_nodes, cand.follow_end),
                    pool=cand.pool,
                    frequency=cand.frequency,
                )
                selected.append(truncated)
        if not selected:
            return None
        nodes = build_candidate_prefix_tree([c.tokens for c in selected], self.max_tree_nodes)
        if not nodes:
            return None
        best = selected[0]
        return ProposalTree(
            kind=self.kind,
            candidates=[c.tokens for c in selected],
            scores=[c.score for c in selected],
            sources=[c.pool for c in selected],
            match_lens=[c.match_len for c in selected],
            max_nodes=self.max_tree_nodes,
            score=max(c.score for c in selected),
            source_start_token=best.source_start,
            source_end_token=best.source_end,
            follow_start_token=best.follow_start,
            follow_end_token=best.follow_end,
            query_len=best.match_len,
            pool=best.pool,
            root_included=False,
            match_kind="multi_tree",
        )


class CodeSpineProposer:
    """Anisotropic exact-reuse tree: one deep spine plus sparse code branches."""

    def __init__(
        self,
        tokenizer,
        *,
        kind: str = "code_spine",
        key_lengths: Iterable[int] = (4, 5, 6, 8, 12, 16),
        min_match_len: int = 4,
        max_spine_len: int = 32,
        max_tree_nodes: int = 12,
        branch_budget: int = 2,
        pool: str = "local",
        edit_mode: bool = False,
        allow_short_match: bool = False,
        enable_identifier_branches: bool = True,
        enable_delimiter_branches: bool = True,
    ):
        self.tokenizer = tokenizer
        self.kind = kind
        self.key_lengths = tuple(sorted({int(k) for k in key_lengths if int(k) > 0}))
        self.min_match_len = int(min_match_len)
        self.max_spine_len = int(max_spine_len)
        self.max_tree_nodes = int(max_tree_nodes)
        self.branch_budget = max(0, int(branch_budget))
        self.pool = pool
        self.edit_mode = bool(edit_mode)
        self.allow_short_match = bool(allow_short_match)
        self.enable_identifier_branches = bool(enable_identifier_branches)
        self.enable_delimiter_branches = bool(enable_delimiter_branches)
        self.index = MultiSourceSuffixIndex(self.key_lengths)

    def propose(self, state: ProposerState) -> ProposalTree | Proposal | None:
        seq = state.tokens_after_teacher
        top_k = max(8, self.branch_budget + 1)
        raw = self.index.query(
            seq,
            prompt_len=state.prompt_len,
            top_k=top_k,
            max_draft_len=self.max_spine_len,
            pool=self.pool,
        )
        raw = [c for c in raw if self._candidate_allowed(c)]
        if not raw:
            return None
        candidates = sorted(raw, key=self._adjusted_score, reverse=True)
        best = candidates[0]
        spine = self._extend_spine(seq, best)
        if not spine:
            return None

        tree_candidates = [spine]
        branch_sources: list[str] = [best.pool]
        branch_scores: list[float] = [self._adjusted_score(best)]
        branch_match_lens: list[int] = [best.match_len]

        for cand in candidates[1:]:
            if len(tree_candidates) > self.branch_budget:
                break
            branch = cand.tokens[: max(1, min(4, self.max_tree_nodes))]
            if branch and tuple(branch) != tuple(spine[: len(branch)]):
                self._append_branch(
                    tree_candidates,
                    branch_sources,
                    branch_scores,
                    branch_match_lens,
                    branch,
                    cand.pool,
                    self._adjusted_score(cand),
                    cand.match_len,
                )

        if self.enable_identifier_branches:
            for branch in self._identifier_branches(state):
                if len(tree_candidates) > self.branch_budget:
                    break
                self._append_branch(
                    tree_candidates,
                    branch_sources,
                    branch_scores,
                    branch_match_lens,
                    branch,
                    "scope",
                    best.score - 0.5,
                    best.match_len,
                )

        if self.enable_delimiter_branches:
            for branch in self._delimiter_branches():
                if len(tree_candidates) > self.branch_budget:
                    break
                self._append_branch(
                    tree_candidates,
                    branch_sources,
                    branch_scores,
                    branch_match_lens,
                    branch,
                    "syntax",
                    best.score - 1.0,
                    best.match_len,
                )

        trimmed = self._trim_to_node_budget(tree_candidates)
        if not trimmed:
            return None
        if len(trimmed) == 1:
            return Proposal(
                kind=self.kind,
                tokens=trimmed[0],
                match_len=best.match_len,
                score=self._adjusted_score(best),
                source_start_token=best.source_start,
                source_end_token=best.source_end,
                follow_start_token=best.follow_start,
                follow_end_token=best.follow_start + len(trimmed[0]),
                query_len=best.match_len,
                pool=best.pool,
                source_region=self._source_region(best),
                root_included=False,
                match_kind="code_spine_chain",
            )
        return ProposalTree(
            kind=self.kind,
            candidates=trimmed,
            scores=branch_scores[: len(trimmed)],
            sources=branch_sources[: len(trimmed)],
            match_lens=branch_match_lens[: len(trimmed)],
            max_nodes=self.max_tree_nodes,
            score=max(branch_scores[: len(trimmed)]),
            source_start_token=best.source_start,
            source_end_token=best.source_end,
            follow_start_token=best.follow_start,
            follow_end_token=best.follow_start + len(trimmed[0]),
            query_len=best.match_len,
            pool=best.pool,
            source_region=self._source_region(best),
            root_included=False,
            match_kind="code_spine_tree",
        )

    def _candidate_allowed(self, cand: SuffixCandidate) -> bool:
        if cand.match_len >= self.min_match_len:
            return True
        if cand.match_len == 3 and self.allow_short_match:
            return cand.pool in {"generated", "current_file", "prompt"} and cand.frequency >= 2
        return False

    def _adjusted_score(self, cand: SuffixCandidate) -> float:
        score = cand.score
        if self.edit_mode and cand.pool in {"prompt", "current_file"}:
            score += 6.0
        if cand.match_len >= 12:
            score += 3.0
        elif cand.match_len == 3:
            score -= 4.0
        return score

    def _extend_spine(self, seq: list[int], cand: SuffixCandidate) -> list[int]:
        cap = adaptive_spine_draft_len(cand.match_len, self.max_spine_len)
        current_start = len(seq) - cand.match_len
        follow_end = min(cand.follow_start + cap, current_start)
        if cand.follow_start >= follow_end:
            return []
        return list(seq[cand.follow_start:follow_end])

    def _append_branch(
        self,
        candidates: list[list[int]],
        sources: list[str],
        scores: list[float],
        match_lens: list[int],
        branch: list[int],
        source: str,
        score: float,
        match_len: int,
    ) -> None:
        if not branch:
            return
        if any(tuple(existing) == tuple(branch) for existing in candidates):
            return
        trial = candidates + [branch]
        if _tree_node_count(trial, self.max_tree_nodes) > self.max_tree_nodes:
            return
        candidates.append(branch)
        sources.append(source)
        scores.append(score)
        match_lens.append(match_len)

    def _identifier_branches(self, state: ProposerState) -> list[list[int]]:
        if not _context_has_type(state.ctx, IDENTIFIER_NODE_TYPES):
            current_match = re.search(r"[A-Za-z_][A-Za-z0-9_]*$", state.text_after)
            if current_match is None:
                return []
        else:
            current_match = re.search(r"[A-Za-z_][A-Za-z0-9_]*$", state.text_after)
        current = current_match.group(0) if current_match else ""
        history = state.text_after[: -len(current)] if current else state.text_after
        seen = [
            (m.group(0), m.start())
            for m in _IDENT_RE.finditer(history)
            if m.group(0) not in _ALL_KEYWORDS
        ]
        out: list[list[int]] = []
        for ident in _rank_seen_strings(history, seen)[: self.branch_budget * 3 + 3]:
            if current and (not ident.startswith(current) or len(ident) <= len(current)):
                continue
            continuation = ident[len(current) :] if current else ident
            tokens = encode_no_special(self.tokenizer, continuation)[:4]
            if tokens:
                out.append(tokens)
        return out

    def _delimiter_branches(self) -> list[list[int]]:
        chunks = [")", "]", ":", ",", "\n"]
        out: list[list[int]] = []
        for chunk in chunks:
            tokens = encode_no_special(self.tokenizer, chunk)
            if tokens:
                out.append(tokens[:2])
        return out

    def _trim_to_node_budget(self, candidates: list[list[int]]) -> list[list[int]]:
        selected: list[list[int]] = []
        for cand in candidates:
            if not selected:
                if len(cand) > self.max_tree_nodes:
                    selected.append(cand[: self.max_tree_nodes])
                else:
                    selected.append(cand)
                continue
            if _tree_node_count(selected + [cand], self.max_tree_nodes) <= self.max_tree_nodes:
                selected.append(cand)
        return selected

    def _source_region(self, cand: SuffixCandidate) -> str:
        if self.edit_mode and cand.pool in {"prompt", "current_file"}:
            return "edit_reference"
        return cand.pool


def _split_prompt_generated(tokenizer, state: ProposerState) -> tuple[str, str]:
    if state.prompt_text:
        if any(marker in state.prompt_text for marker in ("Ġ", "Ċ", "▁")):
            raise AssertionError("rewrite extraction received tokenizer-decoded prompt text")
        generated = ""
        if state.text_after.startswith(state.prompt_text):
            generated = state.text_after[len(state.prompt_text) :]
        return state.prompt_text, generated
    prompt_ids = state.tokens_after_teacher[: state.prompt_len]
    if prompt_ids:
        prompt_text = decode_tokens(tokenizer, prompt_ids)
        if state.text_after.startswith(prompt_text):
            return prompt_text, state.text_after[len(prompt_text) :]
    return state.text_after, ""


def _extract_reference_blocks(prompt: str) -> list[str]:
    blocks = [m.group(1) for m in _FENCED_CODE_RE.finditer(prompt) if m.group(1).strip()]
    if blocks:
        return blocks

    lines = prompt.splitlines(keepends=True)
    spans: list[str] = []
    current: list[str] = []
    for line in lines:
        stripped = line.strip()
        code_like = (
            stripped.startswith(
                (
                    "def ",
                    "async def ",
                    "class ",
                    "import ",
                    "from ",
                    "for ",
                    "if ",
                    "while ",
                    "return ",
                    "try:",
                    "except ",
                    "@",
                )
            )
            or line.startswith(("    ", "\t"))
            or bool(re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*[=\(\{\[]", line))
        )
        if code_like or (current and not stripped):
            current.append(line)
        else:
            if sum(len(x) for x in current) >= 80:
                spans.append("".join(current))
            current = []
    if sum(len(x) for x in current) >= 80:
        spans.append("".join(current))
    return spans


def _has_edit_instruction(prompt: str) -> bool:
    first_fence = prompt.find("```")
    instruction = prompt[:first_fence] if first_fence >= 0 else prompt[:512]
    stripped = instruction.lstrip()
    if first_fence < 0 and stripped.startswith(
        ("def ", "async def ", "class ", "import ", "from ", "@", "#!")
    ):
        return False
    return _EDIT_SIGNAL_RE.search(instruction) is not None


def _edit_terms(prompt: str) -> set[str]:
    terms: set[str] = set()
    for match in _QUOTED_TERM_RE.finditer(prompt):
        value = match.group(1) or match.group(2)
        if value and len(value) >= 2:
            terms.add(value)
    for match in re.finditer(
        rf"\b(?:replace|rename|change)\s+({_REWRITE_BARE_TERM})\s+(?:with|to)\s+({_REWRITE_BARE_TERM})",
        prompt,
        flags=re.IGNORECASE,
    ):
            terms.add(match.group(1))
    return terms


def _rewrite_instruction_region(prompt: str) -> str:
    first_fence = prompt.find("```")
    return prompt[:first_fence] if first_fence >= 0 else prompt


def _is_negated_rewrite_match(text: str, start: int) -> bool:
    sentence_start = max(
        text.rfind(".", 0, start),
        text.rfind("!", 0, start),
        text.rfind("?", 0, start),
        text.rfind("\n", 0, start),
        text.rfind(";", 0, start),
    )
    prefix = text[sentence_start + 1 : start]
    return _NEGATED_REWRITE_PREFIX_RE.search(prefix) is not None


def extract_explicit_rewrites(prompt: str) -> dict[str, str]:
    """Extract the explicit prompt-visible rewrite map supported by TransPLD.

    Supported forms are ``rename OLD to NEW``, ``replace OLD with NEW``,
    ``change OLD to NEW``, and arrow maps such as ``OLD -> NEW``.  Terms may be
    identifiers, dotted fields, numbers, or quoted/backticked literals up to 80
    characters.  Extraction is limited to the instruction text before the first
    fenced code block so examples inside the reference do not become route
    inputs.
    """
    prompt = _rewrite_instruction_region(prompt)
    pairs: dict[str, str] = {}
    for match in _REWRITE_PAIR_RE.finditer(prompt):
        if _is_negated_rewrite_match(prompt, match.start()):
            continue
        old = _clean_rewrite_term(next((g for g in match.groups()[:4] if g), ""))
        new = _clean_rewrite_term(next((g for g in match.groups()[4:] if g), ""))
        if old and new and old != new:
            pairs[old] = new
    for match in _REWRITE_ARROW_RE.finditer(prompt):
        if _is_negated_rewrite_match(prompt, match.start()):
            continue
        old = _clean_rewrite_term(next((g for g in match.groups()[:4] if g), ""))
        new = _clean_rewrite_term(next((g for g in match.groups()[4:] if g), ""))
        if old and new and old != new:
            pairs[old] = new
    return pairs


def _rewrite_pairs(prompt: str) -> dict[str, str]:
    return extract_explicit_rewrites(prompt)


def _clean_rewrite_term(value: str) -> str:
    return value.strip().rstrip(".,;:")


def _coerce_rewrite_pairs(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {
            _clean_rewrite_term(str(k)): _clean_rewrite_term(str(v))
            for k, v in value.items()
            if _clean_rewrite_term(str(k)) != _clean_rewrite_term(str(v))
        }
    if isinstance(value, list):
        out: dict[str, str] = {}
        for item in value:
            if isinstance(item, dict):
                old = item.get("old") or item.get("from") or item.get("source")
                new = item.get("new") or item.get("to") or item.get("target")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                old, new = item[0], item[1]
            else:
                continue
            if old is not None and new is not None:
                old_s = _clean_rewrite_term(str(old))
                new_s = _clean_rewrite_term(str(new))
                if old_s != new_s:
                    out[old_s] = new_s
        return out
    return {}


def _stable_view_key(mapping: dict[str, str]) -> str:
    if not mapping:
        return "identity"
    parts = [f"{old}->{new}" for old, new in sorted(mapping.items())]
    text = "|".join(parts)
    return re.sub(r"[^A-Za-z0-9_.>:-]+", "_", text)[:80]


def apply_boundary_rewrites(text: str, mapping: dict[str, str]) -> str:
    if not mapping:
        return text
    out = text
    for old, new in sorted(mapping.items(), key=lambda item: -len(item[0])):
        if re.fullmatch(r"\.[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", old):
            out = re.sub(rf"{re.escape(old)}(?![A-Za-z0-9_])", new, out)
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\.]*|[0-9]+(?:\.[0-9]+)?", old):
            out = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])", new, out)
        else:
            out = out.replace(old, new)
    return out


def _apply_word_map(text: str, mapping: dict[str, str]) -> str:
    return apply_boundary_rewrites(text, mapping)


def _truncate_before_edit_terms(text: str, terms: set[str]) -> str:
    if not terms:
        return text
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        if any(re.search(rf"\b{re.escape(term)}\b", line) for term in terms):
            break
        out.append(line)
    return "".join(out)


class EditAnchorProposer:
    """Draft unchanged spans by aligning generated output to a prompt reference block."""

    def __init__(
        self,
        tokenizer,
        *,
        kind: str = "edit_anchor",
        max_draft_len: int = 32,
        min_draft_tokens: int = 1,
        min_anchor_chars: int = 12,
        max_anchor_chars: int = 384,
        require_edit_signal: bool = True,
        rewrite_enabled: bool = False,
    ):
        self.tokenizer = tokenizer
        self.kind = kind
        self.max_draft_len = int(max_draft_len)
        self.min_draft_tokens = max(1, int(min_draft_tokens))
        self.min_anchor_chars = int(min_anchor_chars)
        self.max_anchor_chars = int(max_anchor_chars)
        self.require_edit_signal = bool(require_edit_signal)
        self.rewrite_enabled = bool(rewrite_enabled)

    def propose(self, state: ProposerState) -> Proposal | None:
        prompt_text, generated = _split_prompt_generated(self.tokenizer, state)
        if self.require_edit_signal and not _has_edit_instruction(prompt_text):
            return None
        if len(generated) < self.min_anchor_chars:
            return None

        blocks = _extract_reference_blocks(prompt_text)
        if not blocks:
            return None
        terms = _edit_terms(prompt_text)
        rewrite_map = _rewrite_pairs(prompt_text) if self.rewrite_enabled else {}
        best: Proposal | None = None
        for block in blocks:
            proposal = self._proposal_from_block(block, generated, terms, rewrite_map)
            if proposal is not None and (best is None or proposal.score > best.score):
                best = proposal
        return best

    def _proposal_from_block(
        self,
        block: str,
        generated: str,
        terms: set[str],
        rewrite_map: dict[str, str],
    ) -> Proposal | None:
        generated_for_alignment = (
            _apply_word_map(generated, {new: old for old, new in rewrite_map.items()})
            if rewrite_map
            else generated
        )
        max_ctx = min(self.max_anchor_chars, len(generated_for_alignment), len(block))
        for ctx_len in range(max_ctx, self.min_anchor_chars - 1, -1):
            needle = generated_for_alignment[-ctx_len:]
            if not needle.strip():
                continue
            pos = block.rfind(needle)
            if pos < 0:
                continue
            follow_start = pos + ctx_len
            if follow_start >= len(block):
                continue
            continuation = block[follow_start:]
            if rewrite_map:
                continuation = _apply_word_map(continuation, rewrite_map)
            else:
                continuation = _truncate_before_edit_terms(continuation, terms)
            if not continuation:
                continue
            tokens = encode_no_special(self.tokenizer, continuation)[: self.max_draft_len]
            if len(tokens) < self.min_draft_tokens:
                continue
            match_tokens = encode_no_special(self.tokenizer, needle)
            return Proposal(
                kind=self.kind,
                tokens=tokens,
                match_len=len(match_tokens),
                score=10.0 + min(4.0, len(tokens) / 16.0) + min(2.0, ctx_len / 128.0),
                text_preview=continuation[:40],
                query_len=len(match_tokens),
                pool="edit_reference",
                source_region="edit_reference",
                root_included=False,
                match_kind="rewrite_anchor" if rewrite_map else "edit_anchor",
                substitution_count=len(rewrite_map) or None,
            )
        return None


@dataclass(frozen=True)
class _SymbolInfo:
    name: str
    role: str
    pos: int
    frequency: int
    base_score: float


def _collect_symbol_infos(text: str, language: str) -> list[_SymbolInfo]:
    counts: Counter[str] = Counter()
    scores: dict[str, float] = {}
    roles: dict[str, str] = {}
    positions: dict[str, int] = {}

    def add(name: str, role: str, pos: int, score: float) -> None:
        if not name or name in _ALL_KEYWORDS:
            return
        counts[name] += 1
        scores[name] = max(scores.get(name, 0.0), score)
        roles[name] = role
        positions[name] = max(positions.get(name, -1), pos)

    for match in re.finditer(r"\b(?:def|function)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)", text):
        add(match.group(1), "ID_CALL", match.start(1), 3.0)
        for param in _IDENT_RE.finditer(match.group(2)):
            add(param.group(0), "ID_PARAM", match.start(2) + param.start(), 4.0)
    for match in re.finditer(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)", text):
        add(match.group(1), "ID_TYPE", match.start(1), 2.5)
    for match in re.finditer(r"\b(?:from\s+[A-Za-z_][A-Za-z0-9_\.]*\s+import|import)\s+([A-Za-z_][A-Za-z0-9_]*)", text):
        add(match.group(1), "ID_USE", match.start(1), 2.0)
    for match in re.finditer(r"\b(?:self\.)?([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|\+=|-=|\*=|/=|:=)", text):
        add(match.group(1), "ID_DEF", match.start(1), 4.0)
    for match in re.finditer(r"\.([A-Za-z_][A-Za-z0-9_]*)", text):
        add(match.group(1), "ID_ATTR", match.start(1), 3.5)
    for match in _IDENT_RE.finditer(text):
        add(match.group(0), _identifier_role(text, match.start(), match.end(), language), match.start(), 1.0)

    return [
        _SymbolInfo(
            name=name,
            role=roles.get(name, "ID_USE"),
            pos=positions.get(name, -1),
            frequency=counts[name],
            base_score=scores.get(name, 1.0),
        )
        for name in counts
    ]


class SymbolTreeProposer:
    """Sparse identifier/member tree built from prompt/current-file symbols."""

    def __init__(
        self,
        tokenizer,
        *,
        kind: str = "symbol_tree",
        branch_budget: int = 4,
        max_tree_nodes: int = 12,
        max_symbol_tokens: int = 8,
        min_prefix_chars: int = 1,
    ):
        self.tokenizer = tokenizer
        self.kind = kind
        self.branch_budget = max(1, int(branch_budget))
        self.max_tree_nodes = int(max_tree_nodes)
        self.max_symbol_tokens = int(max_symbol_tokens)
        self.min_prefix_chars = int(min_prefix_chars)

    def propose(self, state: ProposerState) -> Proposal | ProposalTree | None:
        current_match = re.search(r"[A-Za-z_][A-Za-z0-9_]*$", state.text_after)
        current = current_match.group(0) if current_match else ""
        before_current = state.text_after[: -len(current)] if current else state.text_after
        member_access = before_current.rstrip().endswith(".")
        ctx_ok = _context_has_type(state.ctx, IDENTIFIER_NODE_TYPES)
        if not (ctx_ok or member_access or len(current) >= self.min_prefix_chars):
            return None
        if not member_access and len(current) < self.min_prefix_chars:
            return None

        symbols = self._rank_symbols(state, current, member_access)
        candidates: list[list[int]] = []
        scores: list[float] = []
        sources: list[str] = []
        match_lens: list[int] = []
        for symbol, score in symbols:
            continuation = symbol.name[len(current) :] if current else symbol.name
            if not continuation or continuation == symbol.name and current and not symbol.name.startswith(current):
                continue
            tokens = encode_no_special(self.tokenizer, continuation)[: self.max_symbol_tokens]
            if not tokens:
                continue
            trial = candidates + [tokens]
            if _tree_node_count(trial, self.max_tree_nodes) > self.max_tree_nodes:
                continue
            candidates.append(tokens)
            # Keep symbol candidates ordered by the richer internal score, but
            # keep the proposal score below exact suffix reuse.  The hybrid
            # priority is EditAnchor -> Suffix -> SymbolTree -> neural tail.
            scores.append(min(1.45, 1.05 + 0.05 * min(4, len(current)) + 0.02 * score))
            sources.append(symbol.role)
            match_lens.append(len(encode_no_special(self.tokenizer, current)) if current else 0)
            if len(candidates) >= self.branch_budget:
                break

        if not candidates:
            return None
        if len(candidates) == 1:
            return Proposal(
                kind=self.kind,
                tokens=candidates[0],
                match_len=match_lens[0],
                score=scores[0],
                pool="scope",
                source_region="symbol_table",
                root_included=False,
                match_kind="symbol_tree",
            )
        return ProposalTree(
            kind=self.kind,
            candidates=candidates,
            scores=scores,
            sources=sources,
            match_lens=match_lens,
            max_nodes=self.max_tree_nodes,
            score=max(scores),
            pool="scope",
            source_region="symbol_table",
            root_included=False,
            match_kind="symbol_tree",
        )

    def _rank_symbols(
        self,
        state: ProposerState,
        current: str,
        member_access: bool,
    ) -> list[tuple[_SymbolInfo, float]]:
        symbols = _collect_symbol_infos(state.text_after, state.language)
        desired_roles = {"ID_ATTR"} if member_access else {"ID_PARAM", "ID_DEF", "ID_USE", "ID_CALL"}
        ranked: list[tuple[_SymbolInfo, float]] = []
        for symbol in symbols:
            if current and (not symbol.name.startswith(current) or len(symbol.name) <= len(current)):
                continue
            if not current and not member_access:
                continue
            role_bonus = 2.0 if symbol.role in desired_roles else 0.0
            recency = min(2.0, max(0.0, symbol.pos / max(1, len(state.text_after))) * 2.0)
            freq = min(1.5, symbol.frequency / 3.0)
            prefix_bonus = min(2.0, len(current) / 4.0) if current else 0.5
            score = 3.0 + symbol.base_score + role_bonus + recency + freq + prefix_bonus
            ranked.append((symbol, score))
        ranked.sort(key=lambda item: (-item[1], -item[0].pos, len(item[0].name), item[0].name))
        return ranked


@dataclass(frozen=True)
class _CanonAtom:
    canon: str
    kind: str
    role: str | None
    text: str
    token_start: int
    token_end: int
    char_start: int
    char_end: int


@dataclass(frozen=True)
class _LexSpan:
    kind: str
    text: str
    char_start: int
    char_end: int
    role: str | None = None


def _token_offsets(tokenizer, text: str, token_ids: list[int]) -> list[tuple[int, int]]:
    """Best-effort token offsets for a decoded prefix.

    Fast HuggingFace tokenizers can return exact offsets.  The fallback keeps
    tests and byte-like tokenizers working; production Qwen tokenization uses
    the fast-tokenizer path.
    """

    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        ids = encoded.input_ids
        offsets = encoded.offset_mapping
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if offsets and isinstance(offsets[0], list):
            offsets = offsets[0]
        ids = [int(t) for t in ids]
        offsets = [(int(s), int(e)) for s, e in offsets]
        if len(ids) == len(token_ids) and ids == [int(t) for t in token_ids]:
            return offsets
    except TypeError:
        pass
    except Exception:
        pass

    if len(text) == len(token_ids):
        return [(i, i + 1) for i in range(len(token_ids))]

    # Conservative fallback for slow tokenizers: align decoded single tokens
    # greedily.  It is less precise around byte-fallback tokens, but still
    # avoids splitting identifier/literal spans when the decoded text is
    # token-concatenative.
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token_id in token_ids:
        piece = decode_tokens(tokenizer, [int(token_id)])
        if piece:
            idx = text.find(piece, cursor)
            if idx >= 0:
                offsets.append((idx, idx + len(piece)))
                cursor = idx + len(piece)
                continue
        offsets.append((cursor, cursor))
    return offsets


def _overlaps(span: tuple[int, int], taken: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < b and a < end for a, b in taken)


def _prev_non_ws(text: str, pos: int) -> str:
    i = pos - 1
    while i >= 0 and text[i].isspace():
        i -= 1
    return text[i] if i >= 0 else ""


def _next_non_ws(text: str, pos: int) -> str:
    i = pos
    while i < len(text) and text[i].isspace():
        i += 1
    return text[i] if i < len(text) else ""


def _identifier_role(text: str, start: int, end: int, language: str) -> str:
    before = text[:start]
    after = text[end:]
    prev_ch = _prev_non_ws(text, start)
    next_ch = _next_non_ws(text, end)
    line_start = text.rfind("\n", 0, start) + 1
    line_prefix = text[line_start:start]

    if prev_ch == ".":
        return "ID_ATTR"
    if next_ch == "(":
        return "ID_CALL"
    if re.search(r"\bdef\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*$", before[line_start:]):
        return "ID_PARAM"
    if re.search(r"\bfunction\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*$", before[line_start:]):
        return "ID_PARAM"
    if re.search(r"(?:^|\s)(?:for|async\s+for)\s+$", line_prefix):
        return "ID_DEF"
    if re.search(r"(?:^|[,(\[])\s*$", line_prefix) and "def " in before[line_start:]:
        return "ID_PARAM"
    if re.search(r"(?:^|[,(\[])\s*$", line_prefix) and "function " in before[line_start:]:
        return "ID_PARAM"
    if after.lstrip().startswith(("=", "+=", "-=", "*=", "/=", "%=", ":=")):
        return "ID_DEF"
    if language in {"ts", "typescript"} and prev_ch == ":":
        return "ID_TYPE"
    if prev_ch == ":" and re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*:\s*$", before):
        return "ID_TYPE"
    if prev_ch == "[" or next_ch == "]":
        return "ID_INDEX"
    return "ID_USE"


def _lex_spans(text: str, language: str, *, enable_roles: bool) -> list[_LexSpan]:
    spans: list[_LexSpan] = []
    taken: list[tuple[int, int]] = []

    for match in _STRING_RE.finditer(text):
        span = (match.start(), match.end())
        spans.append(_LexSpan("STR", match.group(0), span[0], span[1]))
        taken.append(span)
    partial = _PARTIAL_STRING_RE.search(text)
    if partial is not None and partial.start() < len(text) and not _overlaps(
        (partial.start(), partial.end()), taken
    ):
        span = (partial.start(), partial.end())
        spans.append(_LexSpan("STR", partial.group(0), span[0], span[1]))
        taken.append(span)

    for regex, kind in ((_CONST_RE, "BOOL"), (_NUMBER_RE, "NUM")):
        for match in regex.finditer(text):
            span = (match.start(), match.end())
            if _overlaps(span, taken):
                continue
            spans.append(_LexSpan(kind, match.group(0), span[0], span[1]))
            taken.append(span)

    for match in _IDENT_RE.finditer(text):
        ident = match.group(0)
        span = (match.start(), match.end())
        if ident in _ALL_KEYWORDS or _overlaps(span, taken):
            continue
        role = _identifier_role(text, span[0], span[1], language) if enable_roles else None
        spans.append(_LexSpan("ID", ident, span[0], span[1], role))
        taken.append(span)

    return sorted(spans, key=lambda s: (s.char_start, s.char_end))


def _atoms_from_state(
    tokenizer,
    text: str,
    token_ids: list[int],
    language: str,
    *,
    enable_roles: bool,
    normalize_literals: bool,
) -> list[_CanonAtom]:
    offsets = _token_offsets(tokenizer, text, token_ids)
    spans = _lex_spans(text, language, enable_roles=enable_roles)
    atoms: list[_CanonAtom] = []
    token_i = 0

    for span in spans:
        if span.kind in {"NUM", "STR", "BOOL"} and not normalize_literals:
            continue
        covered = [
            i
            for i, (start, end) in enumerate(offsets)
            if end > span.char_start and start < span.char_end
        ]
        if not covered:
            continue
        start_i = min(covered)
        end_i = max(covered) + 1
        while token_i < start_i:
            start, end = offsets[token_i]
            atoms.append(
                _CanonAtom(
                    canon=f"TOK:{int(token_ids[token_i])}",
                    kind="TOK",
                    role=None,
                    text=text[start:end] if end > start else decode_tokens(tokenizer, [token_ids[token_i]]),
                    token_start=token_i,
                    token_end=token_i + 1,
                    char_start=start,
                    char_end=end,
                )
            )
            token_i += 1

        role = span.role if enable_roles and span.kind == "ID" else None
        canon = role if role else span.kind
        atoms.append(
            _CanonAtom(
                canon=canon,
                kind=span.kind,
                role=role,
                text=span.text,
                token_start=start_i,
                token_end=end_i,
                char_start=span.char_start,
                char_end=span.char_end,
            )
        )
        token_i = max(token_i, end_i)

    while token_i < len(token_ids):
        start, end = offsets[token_i]
        atoms.append(
            _CanonAtom(
                canon=f"TOK:{int(token_ids[token_i])}",
                kind="TOK",
                role=None,
                text=text[start:end] if end > start else decode_tokens(tokenizer, [token_ids[token_i]]),
                token_start=token_i,
                token_end=token_i + 1,
                char_start=start,
                char_end=end,
            )
        )
        token_i += 1
    return atoms


def _scope_names(text: str, language: str) -> list[tuple[str, str, int, int]]:
    names: list[tuple[str, str, int, int]] = []
    counts: Counter[str] = Counter()
    for match in _IDENT_RE.finditer(text):
        name = match.group(0)
        if name in _ALL_KEYWORDS:
            continue
        counts[name] += 1
        role = _identifier_role(text, match.start(), match.end(), language)
        names.append((name, role, match.start(), counts[name]))
    return names


class AlphaSuffixProposer:
    """Code-shape suffix reuse under conservative identifier/literal renaming."""

    def __init__(
        self,
        tokenizer,
        *,
        kind: str = "alpha_role",
        min_match_len: int = 6,
        max_query_len: int = 24,
        max_draft_len: int = 8,
        enable_roles: bool = True,
        normalize_literals: bool = True,
        enable_substitution: bool = True,
        scope_fill: bool = True,
        filter_exact: bool = True,
        stop_on_unmapped: bool = True,
        pool: str = "local",
    ):
        if pool not in {"local", "prompt", "generated"}:
            raise ValueError(f"unsupported alpha pool: {pool}")
        self.tokenizer = tokenizer
        self.kind = kind
        self.min_match_len = min_match_len
        self.max_query_len = max_query_len
        self.max_draft_len = max_draft_len
        self.enable_roles = enable_roles
        self.normalize_literals = normalize_literals
        self.enable_substitution = enable_substitution
        self.scope_fill = scope_fill
        self.filter_exact = filter_exact
        self.stop_on_unmapped = stop_on_unmapped
        self.pool = pool

    def propose(self, state: ProposerState) -> Proposal | None:
        seq = state.tokens_after_teacher
        atoms = _atoms_from_state(
            self.tokenizer,
            state.text_after,
            seq,
            state.language,
            enable_roles=self.enable_roles,
            normalize_literals=self.normalize_literals,
        )
        if len(atoms) <= self.min_match_len:
            return None

        max_len = min(self.max_query_len, len(atoms) - 1)
        canon = [a.canon for a in atoms]
        for match_len in range(max_len, self.min_match_len - 1, -1):
            current_start = len(atoms) - match_len
            needle = canon[current_start:]
            for source_start in range(current_start - match_len, -1, -1):
                source_end = source_start + match_len
                if source_end > current_start:
                    continue
                if canon[source_start:source_end] != needle:
                    continue
                if not self._source_allowed(atoms[source_start], atoms[source_end - 1], state.prompt_len):
                    continue
                proposal = self._proposal_from_match(
                    state,
                    seq,
                    atoms,
                    source_start,
                    source_end,
                    current_start,
                    match_len,
                )
                if proposal is not None:
                    return proposal
        return None

    def _source_allowed(self, first: _CanonAtom, last: _CanonAtom, prompt_len: int) -> bool:
        if self.pool == "local":
            return True
        if self.pool == "prompt":
            return last.token_end <= prompt_len
        if self.pool == "generated":
            return first.token_start >= prompt_len
        return True

    def _proposal_from_match(
        self,
        state: ProposerState,
        seq: list[int],
        atoms: list[_CanonAtom],
        source_start: int,
        source_end: int,
        current_start: int,
        match_len: int,
    ) -> Proposal | None:
        source_atoms = atoms[source_start:source_end]
        query_atoms = atoms[current_start:]
        source_token_start = source_atoms[0].token_start
        source_token_end = source_atoms[-1].token_end
        query_token_start = query_atoms[0].token_start
        query_token_end = query_atoms[-1].token_end
        exact_tokens = seq[source_token_start:source_token_end] == seq[query_token_start:query_token_end]

        substitutions: dict[tuple[str, str | None, str], str] = {}
        changed_kinds: set[str] = set()
        for old, new in zip(source_atoms, query_atoms):
            if old.kind not in {"ID", "NUM", "STR", "BOOL"}:
                continue
            if old.kind != new.kind:
                return None
            key = (old.kind, old.role, old.text)
            prev = substitutions.get(key)
            if prev is not None and prev != new.text:
                return None
            substitutions[key] = new.text
            if old.text != new.text:
                changed_kinds.add(old.kind)

        if self.filter_exact and exact_tokens and not changed_kinds:
            return None

        scope = _scope_names(state.text_after, state.language)
        continuation_parts: list[str] = []
        stopped_on_unmapped = False
        scope_fills = 0
        follow_atom_end = source_end
        for atom in atoms[source_end:current_start]:
            text = atom.text
            if atom.kind in {"ID", "NUM", "STR", "BOOL"}:
                key = (atom.kind, atom.role, atom.text)
                if self.enable_substitution and key in substitutions:
                    text = substitutions[key]
                    if text != atom.text:
                        changed_kinds.add(atom.kind)
                elif atom.kind == "ID" and self.scope_fill:
                    fill = self._scope_fill(atom, scope, substitutions)
                    if fill is None:
                        stopped_on_unmapped = True
                        if self.stop_on_unmapped:
                            break
                    else:
                        text = fill
                        scope_fills += 1
                        if text != atom.text:
                            changed_kinds.add("ID")
                else:
                    stopped_on_unmapped = True
                    if self.stop_on_unmapped:
                        break
            continuation_parts.append(text)
            follow_atom_end += 1
            tokens = encode_no_special(self.tokenizer, "".join(continuation_parts))
            if len(tokens) >= self.max_draft_len:
                break

        continuation = "".join(continuation_parts)
        if not continuation:
            return None
        tokens = encode_no_special(self.tokenizer, continuation)[: self.max_draft_len]
        if not tokens:
            return None
        if self.filter_exact and not changed_kinds:
            return None

        match_kind = self._match_kind(changed_kinds)
        score = 1.0 + (match_len / max(1, self.max_query_len * 2.0))
        score += min(0.25, len(tokens) / max(1, self.max_draft_len * 8.0))
        return Proposal(
            kind=self.kind,
            tokens=tokens,
            match_len=match_len,
            score=score,
            text_preview=continuation[:40],
            source_start_token=source_token_start,
            source_end_token=source_token_end,
            follow_start_token=atoms[source_end].token_start if source_end < len(atoms) else None,
            follow_end_token=atoms[follow_atom_end - 1].token_end if follow_atom_end > source_end else None,
            query_len=match_len,
            pool=self.pool,
            source_region=None,
            root_included=False,
            match_kind=match_kind,
            canonical_match_len=match_len,
            substitution_count=sum(1 for old, new in substitutions.items() if old[2] != new),
            scope_fill_count=scope_fills,
            stopped_on_unmapped=stopped_on_unmapped,
            alpha_exact_filtered=self.filter_exact,
        )

    def _scope_fill(
        self,
        atom: _CanonAtom,
        scope: list[tuple[str, str, int, int]],
        substitutions: dict[tuple[str, str | None, str], str],
    ) -> str | None:
        existing = set(substitutions.values())
        candidates = [
            (name, role, pos, freq)
            for name, role, pos, freq in scope
            if name not in _ALL_KEYWORDS and name not in existing
        ]
        if not candidates:
            return None
        role = atom.role or "ID_USE"
        candidates.sort(
            key=lambda item: (
                0 if item[1] == role else 1,
                0 if role == "ID_USE" and item[1] in {"ID_PARAM", "ID_DEF", "ID_USE"} else 1,
                -item[2],
                -item[3],
                abs(len(item[0]) - len(atom.text)),
            )
        )
        best = candidates[0]
        if best[1] != role and role not in {"ID_USE", "ID_INDEX"}:
            return None
        return best[0]

    def _match_kind(self, changed_kinds: set[str]) -> str:
        if self.enable_roles:
            prefix = "alpha_role"
        elif self.normalize_literals:
            prefix = "alpha_idlit"
        else:
            prefix = "alpha_id"
        if {"ID", "NUM", "STR", "BOOL"} & changed_kinds:
            details = []
            if "ID" in changed_kinds:
                details.append("id")
            if {"NUM", "STR", "BOOL"} & changed_kinds:
                details.append("lit")
            return f"{prefix}_{'_'.join(details)}"
        return prefix


class MacroChunkProposer:
    def __init__(
        self,
        tokenizer,
        chunks: Iterable[str],
        *,
        kind: str = "macro_static",
        max_draft_len: int = 8,
    ):
        self.kind = kind
        self.tokenizer = tokenizer
        self.max_draft_len = max_draft_len
        self._chunks: list[tuple[str, list[int]]] = []
        for chunk in chunks:
            ids = encode_no_special(tokenizer, chunk)
            if len(ids) >= 2:
                self._chunks.append((chunk, ids))

    def propose(self, state: ProposerState) -> Proposal | None:
        seq = state.tokens_after_teacher
        best: Proposal | None = None
        for chunk_text, chunk_ids in self._chunks:
            max_l = min(len(chunk_ids) - 1, len(seq))
            for match_len in range(max_l, 0, -1):
                if seq[-match_len:] != chunk_ids[:match_len]:
                    continue
                continuation = chunk_ids[match_len : match_len + self.max_draft_len]
                if not continuation:
                    break
                proposal = Proposal(
                    kind=self.kind,
                    tokens=list(continuation),
                    match_len=match_len,
                    score=1.0 + match_len / len(chunk_ids),
                    text_preview=chunk_text[:40],
                )
                if best is None or proposal.score > best.score:
                    best = proposal
                break
        return best


class CheapProposerStack:
    def __init__(self, proposers: Iterable[CodeProposer]):
        self.proposers = list(proposers)

    def reset(self) -> None:
        for proposer in self.proposers:
            reset = getattr(proposer, "reset", None)
            if callable(reset):
                reset()

    def prepare(self, **kwargs: Any) -> None:
        for proposer in self.proposers:
            prepare = getattr(proposer, "prepare", None)
            if callable(prepare):
                prepare(**kwargs)

    def requires_text_context(self) -> bool:
        for proposer in self.proposers:
            if getattr(proposer, "requires_text_context", True):
                return True
        return False

    def observe(self, feedback: ProposalFeedback) -> None:
        for proposer in self.proposers:
            observe = getattr(proposer, "observe", None)
            if callable(observe):
                observe(feedback)

    def propose(self, state: ProposerState) -> Proposal | ProposalTree | None:
        proposals = [p.propose(state) for p in self.proposers]
        valid = [p for p in proposals if p is not None and p.tokens]
        if not valid:
            return None
        return max(valid, key=lambda p: (p.score, p.match_len, len(p.tokens)))


def static_macro_chunks(language: str) -> tuple[str, ...]:
    if language in {"ts", "typescript"}:
        return TYPESCRIPT_STATIC_MACROS
    return PYTHON_STATIC_MACROS
