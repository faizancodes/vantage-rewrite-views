"""Prompt-time routing features for PLD/TransPLD selection.

The router is deliberately restricted to prompt/reference-visible information
available before decode.
It is used for real-commit experiments where transformed-view lookup only helps
on a minority of tasks and must not tax ordinary PLD-heavy generations.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .code_proposers import _apply_word_map, _rewrite_pairs

_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*|"
    r"\d+(?:\.\d+)?|"
    r"==|!=|<=|>=|->|=>|[^\s]"
)

FEATURE_NAMES = [
    "bias",
    "reference_chars",
    "reference_lines",
    "reference_tokens",
    "output_budget",
    "rewrite_count",
    "rewrite_occurrences",
    "mean_rewrite_gap",
    "longest_exact_span",
    "old_new_char_ratio",
    "old_new_token_ratio",
    "pld_match_density",
    "transformed_match_density",
    "num_files_touched",
    "num_functions_touched",
    "has_identifier_map",
    "has_dotted_map",
    "has_literal_map",
]


def load_router(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def nested_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    nested = metadata.get("metadata")
    if isinstance(nested, dict):
        merged = dict(nested)
        merged.update(metadata)
        return merged
    return dict(metadata)


def coerce_rewrite_pairs(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {
            str(k).strip().rstrip(".,;:"): str(v).strip().rstrip(".,;:")
            for k, v in value.items()
            if str(k).strip().rstrip(".,;:") != str(v).strip().rstrip(".,;:")
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
            if old is None or new is None:
                continue
            old_s = str(old).strip().rstrip(".,;:")
            new_s = str(new).strip().rstrip(".,;:")
            if old_s and new_s and old_s != new_s:
                out[old_s] = new_s
        return out
    return {}


def tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text or "")


def apply_rewrite(text: str, pairs: dict[str, str]) -> str:
    out = text or ""
    for old, new in sorted(pairs.items(), key=lambda item: -len(item[0])):
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\.]*|\d+(?:\.\d+)?", old):
            out = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])", new, out)
        else:
            out = out.replace(old, new)
    return out


def ngram_density(seq: list[str], n: int = 4) -> float:
    if len(seq) < n:
        return 0.0
    counts: dict[tuple[str, ...], int] = {}
    for i in range(0, len(seq) - n + 1):
        key = tuple(seq[i : i + n])
        counts[key] = counts.get(key, 0) + 1
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / max(1, len(seq) - n + 1)


def overlap_density(left: list[str], right: list[str], n: int = 4) -> float:
    if len(left) < n or len(right) < n:
        return 0.0
    left_grams = {tuple(left[i : i + n]) for i in range(0, len(left) - n + 1)}
    if not left_grams:
        return 0.0
    hits = 0
    total = 0
    for i in range(0, len(right) - n + 1):
        total += 1
        if tuple(right[i : i + n]) in left_grams:
            hits += 1
    return hits / max(1, total)


def rewrite_occurrence_positions(reference: str, pairs: dict[str, str]) -> list[int]:
    positions: list[int] = []
    for old in pairs:
        if not old:
            continue
        start = 0
        while True:
            idx = reference.find(old, start)
            if idx < 0:
                break
            positions.append(idx)
            start = idx + max(1, len(old))
    return sorted(positions)


def longest_span_between_positions(reference_len: int, positions: list[int]) -> int:
    if not positions:
        return reference_len
    points = [0] + positions + [reference_len]
    return max(max(0, b - a) for a, b in zip(points, points[1:]))


def map_type_flags(pairs: dict[str, str]) -> tuple[float, float, float]:
    has_identifier = 0.0
    has_dotted = 0.0
    has_literal = 0.0
    for old, new in pairs.items():
        joined = f"{old} {new}"
        if "." in old or "." in new:
            has_dotted = 1.0
        if re.search(r"['\"]|\d", joined):
            has_literal = 1.0
        if re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\b", joined):
            has_identifier = 1.0
    return has_identifier, has_dotted, has_literal


def extract_features(
    *,
    prompt: str,
    reference: str,
    metadata: dict[str, Any] | None = None,
    output_budget: int = 0,
) -> dict[str, float]:
    del metadata
    pairs = _rewrite_pairs(prompt or "")
    ref = reference or ""
    ref_tokens = tokens(ref)
    virtual = _apply_word_map(ref, pairs)
    virtual_tokens = tokens(virtual)
    positions = rewrite_occurrence_positions(ref, pairs)
    gaps = [b - a for a, b in zip(positions, positions[1:])]
    old_tok = 0
    new_tok = 0
    old_chars = 0
    new_chars = 0
    for old, new in pairs.items():
        old_tok += max(1, len(tokens(old)))
        new_tok += max(1, len(tokens(new)))
        old_chars += max(1, len(old))
        new_chars += max(1, len(new))
    has_identifier, has_dotted, has_literal = map_type_flags(pairs)
    prompt_tokens = tokens(prompt or "")
    return {
        "bias": 1.0,
        "reference_chars": float(len(ref)),
        "reference_lines": float(ref.count("\n") + 1 if ref else 0),
        "reference_tokens": float(len(ref_tokens)),
        "output_budget": float(output_budget or 0),
        "rewrite_count": float(len(pairs)),
        "rewrite_occurrences": float(len(positions)),
        "mean_rewrite_gap": float(sum(gaps) / len(gaps) if gaps else len(ref)),
        "longest_exact_span": float(longest_span_between_positions(len(ref), positions)),
        "old_new_char_ratio": float(old_chars / max(1, new_chars)),
        "old_new_token_ratio": float(old_tok / max(1, new_tok)),
        "pld_match_density": float(ngram_density(ref_tokens, 4)),
        "transformed_match_density": float(overlap_density(prompt_tokens, virtual_tokens, 4)),
        "num_files_touched": 1.0,
        "num_functions_touched": 1.0,
        "has_identifier_map": has_identifier,
        "has_dotted_map": has_dotted,
        "has_literal_map": has_literal,
    }


def vectorize(features: dict[str, float], router: dict[str, Any]) -> list[float]:
    names = router.get("feature_names") or FEATURE_NAMES
    means = router.get("means") or {name: 0.0 for name in names}
    scales = router.get("scales") or {name: 1.0 for name in names}
    vec: list[float] = []
    for name in names:
        value = float(features.get(name, 0.0))
        if name == "bias":
            vec.append(1.0)
        else:
            vec.append((value - float(means.get(name, 0.0))) / max(1e-9, float(scales.get(name, 1.0))))
    return vec


def predict_win_probability(features: dict[str, float], router: dict[str, Any]) -> float:
    weights = [float(x) for x in router.get("weights", [])]
    vec = vectorize(features, router)
    if not weights or len(weights) != len(vec):
        return 0.0
    score = sum(w * x for w, x in zip(weights, vec))
    if score >= 0:
        z = math.exp(-score)
        return 1.0 / (1.0 + z)
    z = math.exp(score)
    return z / (1.0 + z)


def should_use_transpld(features: dict[str, float], router: dict[str, Any]) -> bool:
    if float(features.get("rewrite_occurrences", 0.0)) <= 0:
        return False
    threshold = float(router.get("threshold", 0.5))
    return predict_win_probability(features, router) >= threshold
