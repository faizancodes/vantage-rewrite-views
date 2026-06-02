"""Visibility and strategy routing for VANTAGE.

This module is intentionally CPU-only.  It takes the live code-state signals
available at a decode step (tree-sitter node, draft confidence, retrieval
match length, and recent acceptance history) and maps them to a small set of
speculation strategies.  The GPU decoder in ``vantage_router.py`` executes the
selected strategy.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Mapping, Sequence


DARK_NODE_TYPES = {
    "ERROR",
    "identifier",
    "property_identifier",
    "shorthand_property_identifier",
    "attribute",
    "member_expression",
    "subscript",
    "subscript_expression",
    "call",
    "call_expression",
}

LIT_NODE_TYPES = {
    "comment",
    "string",
    "string_content",
    "predefined_type",
    "type_annotation",
    "type_arguments",
    "type_parameters",
    "import_statement",
    "import_from_statement",
    "import_clause",
    "function_signature",
}

RETRIEVAL_FRIENDLY_NODE_TYPES = {
    "comment",
    "string",
    "string_content",
    "import_statement",
    "import_from_statement",
    "import_clause",
    "parameters",
    "formal_parameters",
    "type_annotation",
    "type_arguments",
    "type_parameters",
    "function_signature",
    "interface_declaration",
    "type_alias_declaration",
}

IDENTIFIER_NODE_TYPES = {
    "identifier",
    "property_identifier",
    "shorthand_property_identifier",
}


@dataclass(frozen=True)
class VantageRouterConfig:
    """Thresholds for the live router.

    The defaults are conservative.  They prefer the measured strong baseline
    (short EAGLE chains and tail branching) unless retrieval or scope-copy
    signals are unusually strong.
    """

    low_visibility_threshold: float = 0.35
    high_visibility_threshold: float = 0.72
    tail_min_prob: float = 0.25
    tail_max_margin: float = 0.08
    retrieval_min_match: int = 8
    retrieval_high_match: int = 12
    scope_min_identifier_prefix: int = 1
    rolling_window: int = 16
    rolling_default_acceptance: float = 0.75
    enable_long_chain: bool = False
    default_to_tail: bool = False
    use_ast_zone: bool = True
    use_retrieval: bool = True
    use_scope: bool = True
    use_rolling: bool = True


@dataclass(frozen=True)
class VisibilityFeatures:
    node_type: str | None
    deepest_type: str | None
    draft_top1_prob: float
    draft_top2_margin: float
    retrieval_match_len: int = 0
    rolling_accept_rate: float = 0.75
    scope_match_len: int = 0
    parser_in_error: bool = False


@dataclass(frozen=True)
class VisibilityDecision:
    strategy: str
    score: float
    frontier_depth: int
    zone: str
    reason: str


@dataclass(frozen=True)
class SafeRouteDecision:
    """Prompt-only routing decision for VANTAGE/SafeRoute.

    SafeRoute is deliberately restricted to information visible in the prompt
    or reference text.  It must not inspect benchmark target text, gold output,
    synthetic labels, manifest-only fields, or arbitrary metadata.
    """

    use_transpld: bool
    reason: str | None


def _clean_rewrite_map(rewrite_map: Mapping[str, str] | None) -> dict[str, str]:
    if not rewrite_map:
        return {}
    out: dict[str, str] = {}
    for old, new in rewrite_map.items():
        old_s = str(old)
        new_s = str(new)
        if old_s and new_s and old_s != new_s:
            out[old_s] = new_s
    return out


def decide_prompt_only_saferoute(
    *,
    reference: str,
    rewrite_map: Mapping[str, str] | None,
    transformed_reference: str | None = None,
    reference_tokens: Sequence[int] | None = None,
    transformed_tokens: Sequence[int] | None = None,
) -> SafeRouteDecision:
    """Return whether Rewrite-View Lookup may run from prompt/reference-visible data only.

    The caller is responsible for extracting ``rewrite_map`` from the prompt and
    building ``transformed_reference`` from the reference text.  Token sequences
    are optional and only used to guard against tokenization-equivalent rewrites.
    """

    if not reference:
        return SafeRouteDecision(False, "no_reference")

    clean_map = _clean_rewrite_map(rewrite_map)
    if not clean_map:
        return SafeRouteDecision(False, "no_rewrite_map")

    if transformed_reference is None:
        transformed_reference = reference
        for old, new in sorted(clean_map.items(), key=lambda item: -len(item[0])):
            transformed_reference = transformed_reference.replace(old, new)

    if transformed_reference == reference:
        return SafeRouteDecision(False, "rewrite_map_no_effect")

    if reference_tokens is not None and transformed_tokens is not None:
        if list(reference_tokens) == list(transformed_tokens):
            return SafeRouteDecision(False, "transformed_reference_tokens_equal_reference")

    return SafeRouteDecision(True, None)


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def zone_for_node(node_type: str | None, deepest_type: str | None = None) -> str:
    """Classify a tree-sitter cursor context as a lit/mid/dark zone."""
    types = {t for t in (node_type, deepest_type) if t}
    if types & DARK_NODE_TYPES:
        return "dark"
    if types & LIT_NODE_TYPES:
        return "lit"
    return "mid"


def compute_visibility(
    features: VisibilityFeatures,
    config: VantageRouterConfig | None = None,
) -> tuple[float, str]:
    """Return ``(visibility_score, zone)`` in [0, 1].

    The score is not a learned probability.  It is a transparent online
    heuristic used to route experiments and to log a common diagnostic signal:
    how far the useful speculation frontier appears to extend at this step.
    """
    cfg = config or VantageRouterConfig()
    zone = zone_for_node(features.node_type, features.deepest_type)
    if not cfg.use_ast_zone:
        zone_prior = 0.50
    elif zone == "lit":
        zone_prior = 0.78
    elif zone == "dark":
        zone_prior = 0.22
    else:
        zone_prior = 0.50

    # Top-1 probability and top-1/top-2 margin come from the EAGLE leaf logits.
    # The normalizers are deliberately loose because draft probabilities are
    # usually much flatter than target probabilities in code.
    prob_signal = _clamp01((features.draft_top1_prob - 0.15) / 0.55)
    margin_signal = _clamp01(features.draft_top2_margin / 0.25)
    retrieval_signal = 0.0
    if cfg.use_retrieval:
        retrieval_signal = _clamp01(
            (features.retrieval_match_len - cfg.retrieval_min_match)
            / max(1, cfg.retrieval_high_match - cfg.retrieval_min_match)
        )
    scope_signal = _clamp01(features.scope_match_len / 4.0) if cfg.use_scope else 0.0
    rolling_signal = (
        _clamp01(features.rolling_accept_rate)
        if cfg.use_rolling
        else cfg.rolling_default_acceptance
    )

    score = (
        0.34 * prob_signal
        + 0.16 * margin_signal
        + 0.18 * zone_prior
        + 0.16 * max(retrieval_signal, scope_signal)
        + 0.16 * rolling_signal
    )
    if features.parser_in_error and features.draft_top1_prob < 0.30:
        score -= 0.08
    return _clamp01(score), zone


def estimate_frontier_depth(
    score: float,
    features: VisibilityFeatures,
    config: VantageRouterConfig | None = None,
) -> int:
    """Map visibility to a conservative useful-frontier depth estimate."""
    cfg = config or VantageRouterConfig()
    if features.retrieval_match_len >= cfg.retrieval_high_match:
        return min(4, features.retrieval_match_len)
    if features.scope_match_len >= 2:
        return 3
    if score < cfg.low_visibility_threshold:
        return 1
    if score >= cfg.high_visibility_threshold and cfg.enable_long_chain:
        return 3
    return 2


def choose_strategy(
    features: VisibilityFeatures,
    config: VantageRouterConfig | None = None,
    *,
    retrieval_available: bool = False,
    scope_available: bool = False,
) -> VisibilityDecision:
    """Route a decode step to one of VANTAGE's strategies.

    Strategies are named to match the method tags emitted by the decoder:
    ``chain_k1``, ``chain_k2``, ``tail_k2w2``, ``retrieve``, ``scope``.
    """
    cfg = config or VantageRouterConfig()
    score, zone = compute_visibility(features, cfg)
    frontier = estimate_frontier_depth(score, features, cfg)
    node_type = features.node_type or ""

    if (
        cfg.use_scope
        and scope_available
        and node_type in IDENTIFIER_NODE_TYPES
        and features.scope_match_len > 0
    ):
        return VisibilityDecision(
            strategy="scope",
            score=score,
            frontier_depth=max(frontier, 2),
            zone=zone,
            reason="identifier_scope_match",
        )

    retrieval_friendly = node_type in RETRIEVAL_FRIENDLY_NODE_TYPES or zone == "lit"
    if (
        cfg.use_retrieval
        and retrieval_available
        and features.retrieval_match_len >= cfg.retrieval_min_match
        and (retrieval_friendly or score >= cfg.high_visibility_threshold)
    ):
        return VisibilityDecision(
            strategy="retrieve",
            score=score,
            frontier_depth=max(frontier, 3),
            zone=zone,
            reason="long_retrieval_landmark",
        )

    if score < cfg.low_visibility_threshold:
        return VisibilityDecision(
            strategy="chain_k1",
            score=score,
            frontier_depth=1,
            zone=zone,
            reason="low_visibility",
        )

    if cfg.default_to_tail and features.draft_top1_prob >= cfg.tail_min_prob:
        return VisibilityDecision(
            strategy="tail_k2w2",
            score=score,
            frontier_depth=2,
            zone=zone,
            reason="frontier_tail_default",
        )

    if (
        features.draft_top1_prob >= cfg.tail_min_prob
        and features.draft_top2_margin <= cfg.tail_max_margin
    ):
        return VisibilityDecision(
            strategy="tail_k2w2",
            score=score,
            frontier_depth=2,
            zone=zone,
            reason="tail_ambiguity_frontier",
        )

    if score >= cfg.high_visibility_threshold and cfg.enable_long_chain:
        return VisibilityDecision(
            strategy="chain_k3",
            score=score,
            frontier_depth=3,
            zone=zone,
            reason="high_visibility",
        )

    return VisibilityDecision(
        strategy="chain_k2",
        score=score,
        frontier_depth=2,
        zone=zone,
        reason="default_short_frontier",
    )


class RollingAcceptance:
    """Rolling accepted/k history, globally and per AST node type."""

    def __init__(self, window: int = 16, default: float = 0.75):
        self.window = window
        self.default = default
        self._global: Deque[float] = deque(maxlen=window)
        self._by_node: dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window))

    def rate(self, node_type: str | None) -> float:
        key = node_type or "default"
        vals = self._by_node.get(key)
        if vals:
            return sum(vals) / len(vals)
        if self._global:
            return sum(self._global) / len(self._global)
        return self.default

    def update(self, node_type: str | None, accepted: int, k: int) -> None:
        if k <= 0:
            return
        rate = _clamp01(accepted / k)
        key = node_type or "default"
        self._global.append(rate)
        self._by_node[key].append(rate)
