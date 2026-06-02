"""Greedy rejection rule for speculative decoding.

In greedy (temperature 0) mode, the lossless rejection rule simplifies to:
accept draft token t_i iff t_i == argmax(target_logits[i]); on first
mismatch, take target's prediction at that position as a "correction"
token; if all k drafts accept, append target's bonus prediction.

This module is deliberately model-agnostic — it operates on token IDs and
logits tensors only, no tokenizer or model dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class GreedyVerifyResult:
    accepted_tokens: list[int]
    """Tokens to append to the prefix (k+1 if all accepted, otherwise i+1)."""

    n_accepted_drafts: int
    """How many of the k draft tokens were accepted (0..k)."""

    rejected: bool
    """True if at least one draft was rejected before all k were accepted."""


def greedy_verify(
    drafts: Sequence[int],
    target_logits: torch.Tensor,
    n_pre: int,
) -> GreedyVerifyResult:
    """Greedy rejection-sample drafts against a single target forward pass.

    Args:
        drafts: the k draft tokens proposed by the draft model.
        target_logits: shape [seq_len, vocab]. Comes from one target forward
            pass over (prefix_remainder + drafts), where prefix_remainder
            has length n_pre. logits[p] = predicted next-token distribution
            AFTER target consumed input position p.
        n_pre: length of prefix_remainder. Must be >= 1 (anchoring invariant).

    Returns:
        GreedyVerifyResult with the accepted+correction/bonus tokens.

    Indexing rationale:
        - logits[n_pre - 1] predicts the token at draft position 0
        - logits[n_pre - 1 + i] predicts the token at draft position i
        - logits[-1] = logits[n_pre - 1 + k] would predict post-drafts[k-1]
          which is the "bonus" token (only used if all k accepted).

    Lossless property (greedy):
        The returned accepted_tokens equals exactly what vanilla AR would
        emit for n_accepted_drafts + 1 sequential argmax steps from the
        same prefix.
    """
    if n_pre < 1:
        raise ValueError(
            f"n_pre must be >= 1 to anchor predictions for drafts[0]; got {n_pre}"
        )
    k = len(drafts)
    if target_logits.dim() == 3:
        # Squeeze batch dim if present
        target_logits = target_logits[0]
    if target_logits.shape[0] < n_pre + k:
        raise ValueError(
            f"target_logits has {target_logits.shape[0]} positions but expected "
            f">= {n_pre + k} (n_pre={n_pre} + k={k})"
        )

    # argmax over vocab dim → predictions for each draft position + bonus
    pred_for_draft = target_logits[n_pre - 1 : n_pre - 1 + k].argmax(dim=-1).tolist()
    bonus = int(target_logits[n_pre - 1 + k].argmax(dim=-1).item())

    accepted: list[int] = []
    rejected = False
    for i in range(k):
        if drafts[i] == pred_for_draft[i]:
            accepted.append(drafts[i])
        else:
            # First mismatch: take target's prediction as correction; stop.
            accepted.append(int(pred_for_draft[i]))
            rejected = True
            break

    n_accepted_drafts = len(accepted) - (1 if rejected else 0)
    if not rejected:
        # All k accepted; append bonus (target's prediction after all drafts)
        accepted.append(bonus)

    return GreedyVerifyResult(
        accepted_tokens=accepted,
        n_accepted_drafts=n_accepted_drafts,
        rejected=rejected,
    )
