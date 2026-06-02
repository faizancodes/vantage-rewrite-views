"""Chained PLD lookup for PVP (Predictive Verifier Pipelining).

The chain assumes each prior draft is fully accepted plus one "bonus" token
that the verifier would have emitted at the end of that draft. The bonus is
approximated as the token immediately following the lifted span in the source
corpus (the next token after the matched prompt region).

This bonus approximation is the only lossy thing PVP relies on; it is
explicitly re-verified by the decoder before any row-k output is committed.
The decoder's correctness does not depend on the bonus being right — only on
the chain producing a *non-empty* draft to evaluate in parallel.
"""

from __future__ import annotations

from dataclasses import dataclass

from .blazedit_decoder import prompt_lookup_draft


@dataclass(frozen=True)
class ChainLookupResult:
    draft: list[int]
    match_len: int
    source_start: int
    follow_start: int


def chain_pld_lookup(
    tokens: list[int],
    *,
    n_match: int,
    n_draft: int,
    depth: int = 2,
    min_matching_ngram_size: int = 1,
) -> list[ChainLookupResult]:
    """Chain ``depth`` PLD lookups, assuming each prior chain is fully accepted.

    For step k > 0, the lookup query is the suffix of:

        tokens + draft_0 + [bonus_0] + draft_1 + [bonus_1] + ...

    where ``bonus_i`` is approximated as
    ``tokens[follow_start_i + len(draft_i)]`` (the token immediately following
    the lifted draft in the source corpus). If that index is out of range we
    fall back to the last token of the current chain — this keeps the chain
    alive at the cost of an almost-certainly-wrong bonus token, which the
    decoder will discover and discard.

    The chain stops early at the first lookup miss; the returned list length
    is therefore ``<= depth``.

    Args:
        tokens: current sequence (prompt + accepted-so-far).
        n_match: max matching n-gram size (PLD's ``max_matching_ngram_size``).
        n_draft: max draft tokens per chain step (PLD's ``max_draft_tokens``).
        depth: how many chain steps to attempt.
        min_matching_ngram_size: minimum n-gram length to count as a match.
    """
    out: list[ChainLookupResult] = []
    if depth <= 0:
        return out

    sequence: list[int] = list(tokens)
    for _ in range(depth):
        draft, match_len, source_start, follow_start = prompt_lookup_draft(
            sequence,
            max_matching_ngram_size=n_match,
            max_draft_tokens=n_draft,
            min_matching_ngram_size=min_matching_ngram_size,
        )
        if not draft:
            break
        out.append(
            ChainLookupResult(
                draft=list(draft),
                match_len=int(match_len),
                source_start=int(source_start),
                follow_start=int(follow_start),
            )
        )
        bonus_idx = follow_start + len(draft)
        if 0 <= bonus_idx < len(sequence):
            bonus = sequence[bonus_idx]
        elif sequence:
            bonus = sequence[-1]
        else:
            break
        sequence = sequence + list(draft) + [bonus]
    return out
