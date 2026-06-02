"""Cheap candidate reranking for ambiguous exact PLD hits.

The runtime decoder and the offline evaluator both import this module so the
feature order and score computation stay identical.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS_PATH = REPO_ROOT / "data" / "routers" / "pld_reranker_k4_v1.json"
DEFAULT_LEFTCTX_WEIGHTS_PATH = (
    REPO_ROOT / "data" / "routers" / "pld_reranker_k4_leftctx_margin_v2.json"
)

FEATURE_NAMES = [
    "bias",
    "neg_rank",
    "rank0",
    "rank1",
    "rank2",
    "rank3",
    "source_pos_log",
    "source_pos_norm",
    "is_most_recent",
    "is_generated",
    "is_prompt_reference",
    "has_continuity",
    "continuity_close",
    "neg_continuity_log",
    "left_extension",
    "left_extension_len",
    "left_extension_len_capped_16",
    "left_extension_len_capped_32",
    "left_extension_len_capped_64",
    "left_extension_log1p",
    "next2_unique",
    "next4_unique",
    "next2_inv_freq",
    "next4_inv_freq",
    "candidate_count_log",
    "match_len",
]

_WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class PLDRerankerWeights:
    weights: list[float]
    feature_names: list[str]
    top_k: int = 4
    margin: float | None = None


@dataclass(frozen=True)
class PLDRerankCandidate:
    rank0: int
    source_position: int
    source_type: str
    source_distance_from_previous_good_source: int | None = None
    draft_tokens: tuple[int, ...] = ()
    draft_prefix_text: str = ""
    left_extension: int = 0
    accepted_len: int | None = None


@dataclass(frozen=True)
class PLDRerankContext:
    candidate_count: int
    match_len: int


def continuation_key(text: str, n: int) -> str:
    toks = _WORD_RE.findall(text or "")
    return "\u241f".join(toks[:n])


def load_reranker_weights(path: str | Path | None = None) -> PLDRerankerWeights:
    """Load a reranker weights JSON file.

    If ``path`` is omitted, this uses the checked-in default trained on the
    K=4 ambiguous-candidate oracle.  A missing default fails loudly so benchmark
    runs do not silently fall back to baseline PLD.
    """

    weights_path = Path(path) if path else DEFAULT_WEIGHTS_PATH
    if not weights_path.exists():
        raise FileNotFoundError(
            f"PLD reranker weights not found: {weights_path}. "
            "Pass --pld-rerank-weights or create the default weights file."
        )
    data = json.loads(weights_path.read_text())
    feature_names = list(data.get("feature_names") or FEATURE_NAMES)
    weights = list(data.get("weights") or [])
    unknown = [name for name in feature_names if name not in FEATURE_NAMES]
    if unknown:
        raise ValueError(f"PLD reranker weights contain unknown features: {unknown}")
    if len(weights) != len(FEATURE_NAMES):
        if len(weights) != len(feature_names):
            raise ValueError(
                f"PLD reranker weight length mismatch: got {len(weights)} weights "
                f"for {len(feature_names)} feature names"
            )
    top_k = int(data.get("k") or data.get("top_k") or 4)
    return PLDRerankerWeights(
        weights=[float(x) for x in weights],
        feature_names=feature_names,
        top_k=top_k,
        margin=(
            float(data["selected_margin"])
            if data.get("selected_margin") is not None
            else (
                float(data["margin"])
                if data.get("margin") is not None
                else None
            )
        ),
    )


def compute_left_extension(
    tokens: list[int] | tuple[int, ...],
    *,
    generated_suffix_start: int,
    candidate_source_suffix_start: int,
    max_left: int = 128,
) -> int:
    """Count exact tokens matching to the left of an n-gram match.

    ``candidate_source_suffix_start`` and ``generated_suffix_start`` are the
    starts of two already-matching suffixes in the same token stream.  This
    feature distinguishes repeated 10-token snippets that become unique with a
    little more left context, without changing the decoder into an adaptive
    longer-context PLD variant.
    """

    count = 0
    limit = max(0, int(max_left))
    while count < limit:
        gen_idx = generated_suffix_start - 1 - count
        src_idx = candidate_source_suffix_start - 1 - count
        if gen_idx < 0 or src_idx < 0:
            break
        if tokens[gen_idx] != tokens[src_idx]:
            break
        count += 1
    return count


def extract_candidate_features(
    candidate: PLDRerankCandidate,
    context: PLDRerankContext,
    *,
    all_candidates: list[PLDRerankCandidate],
) -> list[float]:
    max_pos = max((c.source_position for c in all_candidates), default=0)
    next2_freq: dict[str, int] = {}
    next4_freq: dict[str, int] = {}
    next2_by_rank: dict[int, str] = {}
    next4_by_rank: dict[int, str] = {}
    for cand in all_candidates:
        next2 = continuation_key(cand.draft_prefix_text, 2)
        next4 = continuation_key(cand.draft_prefix_text, 4)
        next2_by_rank[cand.rank0] = next2
        next4_by_rank[cand.rank0] = next4
        next2_freq[next2] = next2_freq.get(next2, 0) + 1
        next4_freq[next4] = next4_freq.get(next4, 0) + 1

    typ = candidate.source_type.lower()
    dist = (
        float(candidate.source_distance_from_previous_good_source)
        if candidate.source_distance_from_previous_good_source is not None
        else 0.0
    )
    next2 = next2_by_rank.get(candidate.rank0, "")
    next4 = next4_by_rank.get(candidate.rank0, "")
    left = max(0, int(candidate.left_extension))
    return [
        1.0,
        -float(candidate.rank0),
        1.0 if candidate.rank0 == 0 else 0.0,
        1.0 if candidate.rank0 == 1 else 0.0,
        1.0 if candidate.rank0 == 2 else 0.0,
        1.0 if candidate.rank0 == 3 else 0.0,
        math.log1p(max(0, candidate.source_position)),
        (candidate.source_position / max_pos) if max_pos > 0 else 0.0,
        1.0 if candidate.source_position == max_pos else 0.0,
        1.0 if "generated" in typ else 0.0,
        1.0 if "prompt" in typ or "reference" in typ else 0.0,
        1.0 if candidate.source_distance_from_previous_good_source is not None else 0.0,
        (
            1.0 / (1.0 + dist)
            if candidate.source_distance_from_previous_good_source is not None
            else 0.0
        ),
        (
            -math.log1p(dist)
            if candidate.source_distance_from_previous_good_source is not None
            else 0.0
        ),
        float(left),
        float(left),
        float(min(left, 16)),
        float(min(left, 32)),
        float(min(left, 64)),
        math.log1p(left),
        1.0 if next2_freq.get(next2, 0) == 1 else 0.0,
        1.0 if next4_freq.get(next4, 0) == 1 else 0.0,
        1.0 / max(1, next2_freq.get(next2, 1)),
        1.0 / max(1, next4_freq.get(next4, 1)),
        math.log1p(context.candidate_count),
        float(context.match_len),
    ]


def score_candidate(features: list[float], weights: PLDRerankerWeights | list[float]) -> float:
    if isinstance(weights, PLDRerankerWeights):
        if weights.feature_names == FEATURE_NAMES:
            return sum(float(w) * float(x) for w, x in zip(weights.weights, features))
        feature_by_name = dict(zip(FEATURE_NAMES, features))
        return sum(
            float(w) * float(feature_by_name.get(name, 0.0))
            for name, w in zip(weights.feature_names, weights.weights)
        )
    return sum(float(w) * float(x) for w, x in zip(weights, features))


def score_candidate_set(
    candidates: list[PLDRerankCandidate],
    weights: PLDRerankerWeights,
    *,
    context: PLDRerankContext,
    top_k: int | None = None,
) -> tuple[list[PLDRerankCandidate], list[float], list[list[float]]]:
    limited = candidates[: max(1, int(top_k or weights.top_k or 4))]
    scores: list[float] = []
    feature_rows: list[list[float]] = []
    for cand in limited:
        feats = extract_candidate_features(cand, context, all_candidates=limited)
        feature_rows.append(feats)
        scores.append(score_candidate(feats, weights))
    return limited, scores, feature_rows


def select_candidate_by_policy(
    candidates: list[PLDRerankCandidate],
    weights: PLDRerankerWeights,
    *,
    context: PLDRerankContext,
    top_k: int | None = None,
    policy: str = "learned",
    fixed_rank: int = 0,
) -> tuple[PLDRerankCandidate | None, list[float], list[list[float]]]:
    if not candidates:
        return None, [], []
    limited, scores, feature_rows = score_candidate_set(
        candidates, weights, context=context, top_k=top_k
    )
    policy = (policy or "learned").strip().lower()
    if policy in {"learned", "learned_leftctx_margin"}:
        best_i = max(range(len(limited)), key=lambda i: (scores[i], -limited[i].rank0))
        return limited[best_i], scores, feature_rows
    if policy == "fixed_rank":
        rank = max(0, min(int(fixed_rank), len(limited) - 1))
        return limited[rank], scores, feature_rows
    if policy == "source_continuity":
        with_distance = [
            (i, c)
            for i, c in enumerate(limited)
            if c.source_distance_from_previous_good_source is not None
        ]
        if with_distance:
            best_i, best = min(
                with_distance,
                key=lambda item: (
                    item[1].source_distance_from_previous_good_source
                    if item[1].source_distance_from_previous_good_source is not None
                    else 10**12,
                    item[1].rank0,
                ),
            )
            return best, scores, feature_rows
        return limited[0], scores, feature_rows
    if policy == "left_extension":
        best_i = max(
            range(len(limited)),
            key=lambda i: (limited[i].left_extension, -limited[i].rank0),
        )
        return limited[best_i], scores, feature_rows
    raise ValueError(f"unsupported PLD rerank policy: {policy}")


def apply_score_margin_gate(
    *,
    selected: PLDRerankCandidate | None,
    baseline: PLDRerankCandidate | None,
    selected_score: float | None,
    baseline_score: float | None,
    margin_gate: bool,
    margin: float,
) -> tuple[PLDRerankCandidate | None, float | None, bool]:
    """Apply the runtime safety gate against the baseline PLD candidate.

    Returns ``(candidate, score_margin, fell_back_to_baseline)``.
    """

    if selected is None:
        return None, None, False
    if baseline is None:
        return selected, None, False
    if selected.source_position == baseline.source_position:
        return selected, 0.0, False
    if selected_score is None or baseline_score is None:
        return (baseline, None, True) if margin_gate else (selected, None, False)
    delta = float(selected_score) - float(baseline_score)
    if margin_gate and delta < float(margin):
        return baseline, delta, True
    return selected, delta, False


def select_best_candidate(
    candidates: list[PLDRerankCandidate],
    weights: PLDRerankerWeights,
    *,
    context: PLDRerankContext,
    top_k: int | None = None,
) -> tuple[PLDRerankCandidate | None, list[float]]:
    if not candidates:
        return None, []
    limited = candidates[: max(1, int(top_k or weights.top_k or 4))]
    best, scores, _features = select_candidate_by_policy(
        candidates,
        weights,
        context=context,
        top_k=top_k,
        policy="learned",
    )
    return best, scores


def candidate_from_oracle_row(raw: dict[str, Any], idx: int) -> PLDRerankCandidate:
    draft = str(raw.get("candidate_draft_prefix_128") or "")
    return PLDRerankCandidate(
        rank0=idx,
        source_position=int(raw.get("source_position") or -1),
        source_type=str(raw.get("source_type") or ""),
        source_distance_from_previous_good_source=(
            int(raw["source_distance_from_previous_good_source"])
            if raw.get("source_distance_from_previous_good_source") is not None
            else None
        ),
        draft_prefix_text=draft,
        left_extension=_left_extension_from_oracle_row(raw),
        accepted_len=int(raw.get("lcp_with_actual_future_output") or 0),
    )


def _left_extension_from_oracle_row(raw: dict[str, Any]) -> int:
    for key in (
        "left_extension",
        "left_extension_len",
        "source_left_extension",
        "left_match_len",
    ):
        val = raw.get(key)
        if isinstance(val, (int, float)):
            return int(val)
    return 0
