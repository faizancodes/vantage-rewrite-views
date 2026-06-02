"""Retrieval-based draft for speculative decoding.

Looks up the longest matching suffix of the prefix in a precomputed
suffix array over a Python code corpus, returns the next M tokens
after the longest match as the draft.

Compared to a neural draft (EAGLE), retrieval has near-zero per-step
latency (a single binary search) and can occasionally draft long
runs of boilerplate at high acceptance. It misses on novel code,
where it falls back to vanilla AR for that step (or to an EAGLE
chain in the hybrid path).

Lossless guarantee is unchanged: same greedy rejection rule.

Index format (built by `proto_app.build_retrieval_index`):
  - tokens.npy        — int32, shape (N,), corpus token IDs concatenated
                        with a separator token (eos_token_id) between samples
  - suffix_array.npy  — int32, shape (N,), suffix start positions sorted
                        lexicographically over `tokens`
  - meta.json         — corpus name, target tokenizer, n_tokens, sep_token_id
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .decoder import (
    DecodeResult,
    StepRecord,
    crop_dynamic_cache,
)


# ---------------------------------------------------------------------------
# Index loading + lookup
# ---------------------------------------------------------------------------


@dataclass
class RetrievalIndex:
    tokens: np.ndarray        # int32 (N,)
    suffix_array: np.ndarray  # int32 (N,), positions into `tokens`
    sep_token_id: int
    meta: dict

    @classmethod
    def load(cls, index_dir: str | Path) -> "RetrievalIndex":
        d = Path(index_dir)
        tokens = np.load(d / "tokens.npy", mmap_mode="r")
        suffix_array = np.load(d / "suffix_array.npy", mmap_mode="r")
        with open(d / "meta.json") as f:
            meta = json.load(f)
        return cls(
            tokens=tokens,
            suffix_array=suffix_array,
            sep_token_id=int(meta["sep_token_id"]),
            meta=meta,
        )

    def find_longest_match(
        self,
        query: np.ndarray,
        min_match_len: int = 3,
    ) -> tuple[int, int] | None:
        """Find the longest suffix of `query` that appears in the corpus.

        Returns (match_position, match_len) where match_position is an index
        into self.tokens and match_len is the matched suffix length.
        Returns None if no suffix of length >= min_match_len matches.

        Implementation: starting from the longest suffix of `query` (length
        L = len(query)), bisect into the suffix array to find any match.
        If no match, decrement L and retry. Stop at L < min_match_len.
        """
        sa = self.suffix_array
        toks = self.tokens
        N = toks.shape[0]
        L = len(query)

        for try_len in range(L, min_match_len - 1, -1):
            needle = query[-try_len:]
            # Binary search for the leftmost SA position whose suffix is >= needle
            lo, hi = 0, N
            while lo < hi:
                mid = (lo + hi) // 2
                pos = sa[mid]
                # Compare toks[pos : pos+try_len] vs needle
                cmp_len = min(try_len, N - pos)
                # Lexicographic compare. Convert mmap slice to numpy view.
                a = toks[pos : pos + cmp_len]
                # Compare element-by-element; np.lexsort or argmin is overkill
                # for short keys. Use a simple loop / first-difference.
                less = False
                equal = True
                for i in range(cmp_len):
                    if a[i] < needle[i]:
                        less = True
                        equal = False
                        break
                    elif a[i] > needle[i]:
                        equal = False
                        break
                if less or (equal and cmp_len < try_len):
                    # toks[pos:] starts with prefix < needle; go right
                    lo = mid + 1
                else:
                    hi = mid

            if lo < N:
                pos = sa[lo]
                # Check exact match of needle prefix
                cmp_len = min(try_len, N - pos)
                if cmp_len >= try_len and np.array_equal(
                    toks[pos : pos + try_len], needle
                ):
                    return int(pos), try_len

        return None


# ---------------------------------------------------------------------------
# Retrieval draft step (no EAGLE)
# ---------------------------------------------------------------------------


def retrieve_draft(
    index: RetrievalIndex,
    prefix_token_ids: list[int],
    max_query_len: int = 16,
    max_draft_len: int = 10,
    min_match_len: int = 3,
) -> tuple[list[int], int]:
    """Return (draft_tokens, matched_len). draft_tokens has length up to
    max_draft_len. matched_len is the suffix length that matched (0 if no
    match found)."""
    if len(prefix_token_ids) < min_match_len:
        return [], 0
    query_len = min(max_query_len, len(prefix_token_ids))
    query = np.asarray(prefix_token_ids[-query_len:], dtype=np.int32)
    res = index.find_longest_match(query, min_match_len=min_match_len)
    if res is None:
        return [], 0
    pos, match_len = res
    # Draft = the next max_draft_len tokens after the match in the corpus,
    # but cut off at the separator (we don't want to draft across samples).
    start = pos + match_len
    end = min(start + max_draft_len, index.tokens.shape[0])
    draft_slice = np.asarray(index.tokens[start:end])
    # Truncate at separator token
    sep_positions = np.where(draft_slice == index.sep_token_id)[0]
    if sep_positions.size > 0:
        draft_slice = draft_slice[: sep_positions[0]]
    return [int(t) for t in draft_slice], match_len


# ---------------------------------------------------------------------------
# Speculative AR with retrieval drafts (no EAGLE)
# ---------------------------------------------------------------------------


def _argmax_int(logits_row: torch.Tensor) -> int:
    return int(logits_row.argmax(dim=-1).item())


@torch.no_grad()
def retrieval_speculative_ar(
    prompt_ids: torch.Tensor,
    target,
    index: RetrievalIndex,
    max_new_tokens: int,
    eos_token_ids: list[int],
    method_name: str,
    max_draft_len: int = 10,
    max_query_len: int = 16,
    min_match_len: int = 3,
    parse_callback=None,
) -> DecodeResult:
    """Speculative AR with retrieval-only drafts.

    For each outer step:
      1. Bring target up to date (multi-token catchup if needed).
      2. Build draft via retrieval over the prefix's last `max_query_len`
         tokens.
      3. Verify with the target via a single forward pass.
      4. Greedy reject: accept up to first mismatch, take target's
         correction at the rejection point.
    """
    device = next(target.parameters()).device
    prompt_ids_list = prompt_ids.tolist() if prompt_ids.dim() == 1 else prompt_ids[0].tolist()
    prefix: list[int] = list(prompt_ids_list)

    target_cache = None
    last_logits: torch.Tensor | None = None

    steps: list[StepRecord] = []
    t_start = time.perf_counter_ns()

    step_idx = 0
    while len(prefix) < len(prompt_ids_list) + max_new_tokens:
        t_step_start = time.perf_counter_ns()

        # ---- Bring target up to date (multi-token catchup) ----
        target_prefill_us = 0.0
        if target_cache is None:
            t0 = time.perf_counter_ns()
            full_in = torch.tensor([prefix], device=device)
            t_out = target(full_in, use_cache=True)
            target_cache = t_out.past_key_values
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
                use_cache=True,
            )
            target_cache = t_out.past_key_values
            last_logits = t_out.logits[0, -1].clone()
            target_prefill_us = (time.perf_counter_ns() - t0) / 1000.0

        # ---- Draft: teacher's argmax + retrieved tokens ----
        # Metrics convention: n_accepted_drafts counts accepted speculative
        # candidate positions. The first candidate is target's cached argmax,
        # so it is accepted by construction; accepted retrieved-continuation
        # tokens are max(0, n_accepted_drafts - 1).
        t_draft_0 = time.perf_counter_ns()
        assert last_logits is not None
        teacher_argmax = _argmax_int(last_logits)
        retrieved, matched_len = retrieve_draft(
            index,
            prefix + [teacher_argmax],  # query includes the teacher's first draft
            max_query_len=max_query_len,
            max_draft_len=max_draft_len,
            min_match_len=min_match_len,
        )
        drafts = [teacher_argmax] + retrieved
        t_draft_1 = time.perf_counter_ns()
        draft_us = (t_draft_1 - t_draft_0) / 1000.0
        k_total = len(drafts)

        if k_total == 1:
            # No retrieval match; just emit teacher's argmax + one bonus.
            # Equivalent to taking target's argmax at this position and the next.
            verify_in = torch.tensor([drafts], device=device)
            t_verify_0 = time.perf_counter_ns()
            t_out_v = target(
                verify_in,
                past_key_values=target_cache,
                use_cache=True,
            )
            target_cache = t_out_v.past_key_values
            t_verify_1 = time.perf_counter_ns()
            verify_us = (t_verify_1 - t_verify_0) / 1000.0
            bonus = int(t_out_v.logits[0, -1].argmax(dim=-1).item())
            accepted_tokens = [drafts[0], bonus]
            n_accepted_drafts = 1  # only teacher draft
            rejected = False
        else:
            # Tree-free chain verify: feed all drafts to target in one forward.
            verify_in = torch.tensor([drafts], device=device)
            t_verify_0 = time.perf_counter_ns()
            t_out_v = target(
                verify_in,
                past_key_values=target_cache,
                use_cache=True,
            )
            target_cache = t_out_v.past_key_values
            t_verify_1 = time.perf_counter_ns()
            verify_us = (t_verify_1 - t_verify_0) / 1000.0
            v_logits = t_out_v.logits  # [1, k_total, V]

            # Greedy verify
            accepted_tokens = [drafts[0]]
            n_accepted_drafts = 1
            rejected = False
            for i in range(1, k_total):
                target_pred = int(v_logits[0, i - 1].argmax(dim=-1).item())
                if drafts[i] == target_pred:
                    accepted_tokens.append(drafts[i])
                    n_accepted_drafts += 1
                else:
                    accepted_tokens.append(target_pred)
                    rejected = True
                    break
            if not rejected:
                bonus = int(v_logits[0, k_total - 1].argmax(dim=-1).item())
                accepted_tokens.append(bonus)

        # ---- EOS truncation + budget cap ----
        eos_truncated = list(accepted_tokens)
        for i_tk, tk in enumerate(eos_truncated):
            if tk in eos_token_ids:
                eos_truncated = eos_truncated[: i_tk + 1]
                break
        budget = (len(prompt_ids_list) + max_new_tokens) - len(prefix)
        if budget < len(eos_truncated):
            accepted_capped = eos_truncated[:budget]
        else:
            accepted_capped = eos_truncated
        prefix.extend(accepted_capped)
        n_emitted = len(accepted_capped)

        # ---- Crop cache to len(prefix) - 1; force multi-token catchup next step ----
        # Cache currently has prefix_old + k_total draft slots = len(prefix_old) + k_total
        # We want to keep only the linearly-accepted prefix portion = len(prefix_old) + n_accepted_drafts
        # (drafts[0..n_accepted_drafts-1] are the accepted draft tokens, in order.)
        # The bonus / correction position is NOT in cache yet — next step's catchup handles it.
        keep_len = (len(prefix) - n_emitted) + n_accepted_drafts
        crop_dynamic_cache(target_cache, keep_len)
        last_logits = None

        if parse_callback is not None:
            t_pcb_0 = time.perf_counter_ns()
            parse_callback(prefix)
            t_pcb_1 = time.perf_counter_ns()
            target_prefill_us += (t_pcb_1 - t_pcb_0) / 1000.0

        t_step_end = time.perf_counter_ns()
        steps.append(StepRecord(
            method=method_name,
            step=step_idx,
            k=k_total,
            n_accepted_drafts=n_accepted_drafts,
            n_emitted=n_emitted,
            rejected=rejected,
            node_type=None,
            deepest_type=f"matched_len={matched_len}",
            wall_us=(t_step_end - t_step_start) / 1000.0,
            draft_us=draft_us,
            verify_us=verify_us,
            parse_us=target_prefill_us,
        ))
        step_idx += 1

        if any(t in eos_token_ids for t in accepted_capped):
            break

    t_end = time.perf_counter_ns()
    return DecodeResult(
        output_token_ids=prefix,
        output_text="",
        n_new_tokens=len(prefix) - len(prompt_ids_list),
        steps=steps,
        wall_us_total=(t_end - t_start) / 1000.0,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def retrieval_spec(
    prompt_ids: torch.Tensor,
    target,
    index: RetrievalIndex,
    max_new_tokens: int,
    eos_token_ids: list[int],
    max_draft_len: int = 10,
) -> DecodeResult:
    return retrieval_speculative_ar(
        prompt_ids=prompt_ids,
        target=target,
        index=index,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        method_name=f"retrieval_d{max_draft_len}",
        max_draft_len=max_draft_len,
    )
