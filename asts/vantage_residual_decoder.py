"""Verified VANTAGE residual decoding scaffold.

The full GPU runtime can reuse the BlazEdit PLD verification path.  This
module keeps testable, model-agnostic semantics for residual drafts: residual
tokens are never emitted unless the target verifier accepts them.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import torch

from .decoder import DecodeResult, StepRecord, crop_dynamic_cache
from .rejection import GreedyVerifyResult, greedy_verify


@dataclass(frozen=True)
class ResidualTriggerDecision:
    triggered: bool
    reason: str = ""


def should_trigger_residual(
    *,
    policy: str,
    accepted_len: int,
    pld_miss: bool = False,
) -> ResidualTriggerDecision:
    """Small local trigger policy used by the runtime scaffold.

    The richer router module is intentionally not imported here: this decoder
    is the unit-testable, model-agnostic path, and it must remain usable while
    router training/evaluation code is still in flux.
    """

    normalized = policy.lower().strip()
    if normalized in {"never", "false", "off", "disabled"}:
        return ResidualTriggerDecision(False, normalized)
    if normalized in {"always", "true", "on"}:
        return ResidualTriggerDecision(True, normalized)
    if normalized == "accepted_len_eq_0":
        return ResidualTriggerDecision(accepted_len == 0, normalized)
    if normalized == "pld_miss_only":
        return ResidualTriggerDecision(bool(pld_miss), normalized)
    if normalized in {"router", "router_weak", "router_predicted_weak"}:
        return ResidualTriggerDecision(bool(pld_miss) or accepted_len <= 4, normalized)

    match = re.fullmatch(r"accepted_len_le_(\d+)", normalized)
    if match:
        threshold = int(match.group(1))
        return ResidualTriggerDecision(accepted_len <= threshold, normalized)

    raise ValueError(f"unknown residual trigger policy: {policy!r}")


@dataclass(frozen=True)
class VantageResidualConfig:
    method_name: str = "vantage_residual_k4"
    residual_k: int = 4
    residual_enabled: bool = True
    residual_trigger: str = "accepted_len_le_4"
    preverify_replace: bool = False
    queued_residual: bool = False
    router_min_pld_draft_len: int = 4
    pld_max_draft_tokens: int = 128
    pld_max_matching_ngram_size: int = 10
    pld_min_matching_ngram_size: int = 1


class TokenVerifier(Protocol):
    def verify(self, draft_tokens: list[int]) -> tuple[list[int], int, bool]:
        """Return emitted tokens, accepted draft count, and rejected flag."""


@dataclass
class ResidualStepResult:
    emitted_tokens: list[int]
    pld_emitted_tokens: list[int]
    residual_triggered: bool
    residual_draft_tokens: list[int] = field(default_factory=list)
    residual_accepted_drafts: int = 0
    residual_rejected: bool = False


def verified_residual_step(
    *,
    pld_emitted_tokens: list[int],
    pld_accepted_len: int,
    residual_draft_tokens: list[int],
    verifier: TokenVerifier,
    trigger_policy: str = "accepted_len_le_4",
    pld_miss: bool = False,
    max_residual_tokens: int | None = None,
) -> ResidualStepResult:
    """Append verified residual progress after a PLD step.

    ``pld_emitted_tokens`` is already target-verified PLD progress for this
    step.  The residual drafter predicts tokens after that progress.  A false
    trigger returns the PLD output unchanged.
    """

    decision = should_trigger_residual(
        policy=trigger_policy,
        accepted_len=int(pld_accepted_len),
        pld_miss=bool(pld_miss),
    )
    if not decision.triggered:
        return ResidualStepResult(
            emitted_tokens=list(pld_emitted_tokens),
            pld_emitted_tokens=list(pld_emitted_tokens),
            residual_triggered=False,
        )
    draft = list(residual_draft_tokens)
    if max_residual_tokens is not None:
        draft = draft[: int(max_residual_tokens)]
    if not draft:
        return ResidualStepResult(
            emitted_tokens=list(pld_emitted_tokens),
            pld_emitted_tokens=list(pld_emitted_tokens),
            residual_triggered=True,
        )
    residual_emitted, accepted, rejected = verifier.verify(draft)
    return ResidualStepResult(
        emitted_tokens=[*pld_emitted_tokens, *residual_emitted],
        pld_emitted_tokens=list(pld_emitted_tokens),
        residual_triggered=True,
        residual_draft_tokens=draft,
        residual_accepted_drafts=int(accepted),
        residual_rejected=bool(rejected),
    )


def residual_method_to_blazedit_method(method: str) -> str:
    """Map VANTAGE-Residual runtime aliases to the existing verified path."""

    if method.startswith("vantage_residual_"):
        return method
    return method


def parse_vantage_residual_method(method: str) -> VantageResidualConfig:
    """Parse lightweight residual runtime mode names.

    Supported names are ``vantage_residual_k1``, ``vantage_residual_k2``,
    ``vantage_residual_k4``, ``router_k4``, and
    ``vantage_residual_router_k4``.  The PLD side intentionally defaults to
    the requested pure-PLD equivalence settings: match_n=10 and max_draft=128.
    """

    normalized = method.strip()
    match = re.fullmatch(r"vantage_residual_k([124])", normalized)
    if match:
        return VantageResidualConfig(
            method_name=normalized,
            residual_k=int(match.group(1)),
            residual_trigger="accepted_len_le_4",
        )

    if re.fullmatch(r"(?:vantage_residual_)?router_k4", normalized):
        return VantageResidualConfig(
            method_name=normalized,
            residual_k=4,
            residual_trigger="router",
            router_min_pld_draft_len=4,
        )

    preverify = re.fullmatch(r"vantage_residual_preverify_replace_k([124])", normalized)
    if preverify:
        return VantageResidualConfig(
            method_name=normalized,
            residual_k=int(preverify.group(1)),
            residual_trigger="router",
            preverify_replace=True,
            router_min_pld_draft_len=4,
        )

    queued = re.fullmatch(r"vantage_residual_queued_k([124])", normalized)
    if queued:
        return VantageResidualConfig(
            method_name=normalized,
            residual_k=int(queued.group(1)),
            residual_trigger="queued_conservative",
            queued_residual=True,
            router_min_pld_draft_len=4,
        )

    raise ValueError(f"unknown VANTAGE residual method: {method!r}")


def is_vantage_residual_method(method: str) -> bool:
    try:
        parse_vantage_residual_method(method)
    except ValueError:
        return False
    return True


def prompt_lookup_draft(
    tokens: list[int],
    *,
    max_matching_ngram_size: int,
    max_draft_tokens: int,
    min_matching_ngram_size: int = 1,
) -> tuple[list[int], int, int, int]:
    """Return a BlazEdit-compatible local prompt-lookup continuation."""

    if max_draft_tokens <= 0 or not tokens:
        return [], 0, -1, -1
    max_n = min(max_matching_ngram_size, len(tokens))
    for n in range(max_n, min_matching_ngram_size - 1, -1):
        current_start = len(tokens) - n
        if current_start <= 0:
            continue
        suffix = tokens[current_start:]
        best_start: int | None = None
        for start in range(current_start - n, -1, -1):
            if tokens[start : start + n] == suffix:
                best_start = start
                break
        if best_start is None:
            continue
        follow_start = best_start + n
        follow_end = min(follow_start + max_draft_tokens, current_start)
        draft = tokens[follow_start:follow_end]
        if draft:
            return list(draft), n, best_start, follow_start
    return [], 0, -1, -1


@dataclass
class _VerifyRun:
    result: GreedyVerifyResult
    cache: object | None
    cache_len: int
    verify_us: float


@dataclass
class _DraftRun:
    drafts: list[int]
    cache: object | None
    cache_len: int
    draft_us: float


@dataclass
class _QueuedResidualDraft:
    position: int
    prefix_hash: str
    draft_tokens: list[int]
    source_step: int
    confidence: float = 1.0


def _prefix_hash(prefix: Sequence[int]) -> str:
    payload = ",".join(str(int(token)) for token in prefix).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def queued_residual_invalid_reason(
    prefix: Sequence[int],
    queued: _QueuedResidualDraft | None,
    *,
    eos_hit: bool = False,
    max_length_hit: bool = False,
    task_id: str | None = None,
    queued_task_id: str | None = None,
) -> str | None:
    """Return why a queued residual draft cannot be used at this prefix."""

    if queued is None:
        return "missing"
    if task_id is not None and queued_task_id is not None and task_id != queued_task_id:
        return "task_id_mismatch"
    if eos_hit:
        return "eos"
    if max_length_hit:
        return "max_length"
    if int(queued.position) != len(prefix):
        return "position_mismatch"
    if str(queued.prefix_hash) != _prefix_hash(prefix):
        return "prefix_hash_mismatch"
    if not queued.draft_tokens:
        return "empty"
    return None


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _argmax_int(logits_row: torch.Tensor) -> int:
    return int(logits_row.argmax(dim=-1).item())


def _tensor_row(tokens: Sequence[int], *, device: torch.device) -> torch.Tensor:
    return torch.tensor([list(tokens)], dtype=torch.long, device=device)


def _verify_drafts(
    *,
    prefix: list[int],
    target,
    target_cache: object | None,
    target_cache_len: int,
    drafts: list[int],
    device: torch.device,
) -> _VerifyRun:
    if not drafts:
        raise ValueError("_verify_drafts requires at least one draft token")
    if target_cache is None:
        target_cache_len = 0
    if target_cache is not None and target_cache_len >= len(prefix):
        crop_dynamic_cache(target_cache, len(prefix) - 1)
        target_cache_len = len(prefix) - 1

    n_pre = len(prefix) - target_cache_len
    target_input_list = prefix[target_cache_len:] + drafts
    target_input = _tensor_row(target_input_list, device=device)

    t0 = time.perf_counter_ns()
    target_out = target(target_input, past_key_values=target_cache, use_cache=True)
    t1 = time.perf_counter_ns()

    result = greedy_verify(
        drafts=drafts,
        target_logits=target_out.logits,
        n_pre=n_pre,
    )
    return _VerifyRun(
        result=result,
        cache=target_out.past_key_values,
        cache_len=(
            target_cache_len + len(target_input_list)
            if target_out.past_key_values is not None
            else 0
        ),
        verify_us=(t1 - t0) / 1000.0,
    )


def _emit_one_target_token(
    *,
    prefix: list[int],
    target,
    target_cache: object | None,
    target_cache_len: int,
    device: torch.device,
) -> tuple[int, object | None, int, float]:
    if target_cache is None:
        target_cache_len = 0
    if target_cache is not None and target_cache_len >= len(prefix):
        crop_dynamic_cache(target_cache, len(prefix) - 1)
        target_cache_len = len(prefix) - 1
    target_input = _tensor_row(prefix[target_cache_len:], device=device)

    t0 = time.perf_counter_ns()
    target_out = target(target_input, past_key_values=target_cache, use_cache=True)
    t1 = time.perf_counter_ns()

    return (
        _argmax_int(target_out.logits[0, -1]),
        target_out.past_key_values,
        (
            target_cache_len + target_input.shape[1]
            if target_out.past_key_values is not None
            else 0
        ),
        (t1 - t0) / 1000.0,
    )


def _draft_residual_tokens(
    *,
    prefix: list[int],
    residual_model,
    residual_cache: object | None,
    residual_cache_len: int,
    k: int,
) -> _DraftRun:
    if residual_model is None or k <= 0:
        return _DraftRun([], residual_cache, residual_cache_len, 0.0)

    device = _model_device(residual_model)
    if residual_cache is None:
        residual_cache_len = 0
        feed = _tensor_row(prefix, device=device)
    elif residual_cache_len < len(prefix):
        feed = _tensor_row(prefix[residual_cache_len:], device=device)
    else:
        feed = _tensor_row([prefix[-1]], device=device)

    drafts: list[int] = []
    t0 = time.perf_counter_ns()
    for _ in range(k):
        out = residual_model(feed, past_key_values=residual_cache, use_cache=True)
        residual_cache = out.past_key_values
        next_token = _argmax_int(out.logits[0, -1])
        drafts.append(next_token)
        if residual_cache is None:
            residual_cache_len = 0
            feed = _tensor_row([*prefix, *drafts], device=device)
        else:
            residual_cache_len = len(prefix) + len(drafts)
            feed = _tensor_row([next_token], device=device)
    t1 = time.perf_counter_ns()

    return _DraftRun(
        drafts=drafts,
        cache=residual_cache,
        cache_len=residual_cache_len,
        draft_us=(t1 - t0) / 1000.0,
    )


def _append_with_limits(
    *,
    prefix: list[int],
    tokens: Sequence[int],
    prompt_len: int,
    max_new_tokens: int,
    eos_token_ids: set[int],
) -> tuple[list[int], bool]:
    budget = prompt_len + max_new_tokens - len(prefix)
    emitted: list[int] = []
    eos_hit = False
    for token in tokens:
        if budget <= 0:
            break
        token_i = int(token)
        emitted.append(token_i)
        budget -= 1
        if token_i in eos_token_ids:
            eos_hit = True
            break
    prefix.extend(emitted)
    return emitted, eos_hit


def _residual_triggered(
    *,
    prefix: list[int],
    pld_info: dict[str, object],
    config: VantageResidualConfig,
    residual_trigger: Callable[
        [list[int], dict[str, object], VantageResidualConfig], bool
    ]
    | None,
) -> bool:
    if not config.residual_enabled:
        return False
    if residual_trigger is not None:
        return bool(residual_trigger(list(prefix), dict(pld_info), config))

    accepted_len = int(pld_info.get("accepted_len", 0) or 0)
    pld_draft_len = int(pld_info.get("draft_len", 0) or 0)
    pld_miss = pld_draft_len == 0
    if config.residual_trigger == "router":
        return pld_miss or pld_draft_len < config.router_min_pld_draft_len
    decision = should_trigger_residual(
        policy=config.residual_trigger,
        accepted_len=accepted_len,
        pld_miss=pld_miss,
    )
    return decision.triggered


def _preverify_residual_triggered(
    *,
    prefix: list[int],
    pld_info: dict[str, object],
    config: VantageResidualConfig,
    residual_trigger: Callable[
        [list[int], dict[str, object], VantageResidualConfig], bool
    ]
    | None,
) -> bool:
    """Trigger using only information available before target verification."""

    if not config.residual_enabled:
        return False
    info = dict(pld_info)
    info["accepted_len"] = None
    info["accepted_len_available"] = False
    if residual_trigger is not None:
        return bool(residual_trigger(list(prefix), info, config))

    pld_draft_len = int(info.get("draft_len", 0) or 0)
    pld_miss = pld_draft_len == 0
    normalized = config.residual_trigger.lower().strip()
    if normalized in {"never", "false", "off", "disabled"}:
        return False
    if normalized in {"always", "true", "on"}:
        return True
    if normalized in {"router", "router_weak", "router_predicted_weak"}:
        return pld_miss or pld_draft_len < int(config.router_min_pld_draft_len)
    if normalized == "pld_miss_only":
        return pld_miss
    match = re.fullmatch(r"draft_len_le_(\d+)", normalized)
    if match:
        return pld_draft_len <= int(match.group(1))
    raise ValueError(
        "preverify replacement cannot use post-verification trigger "
        f"{config.residual_trigger!r}; use router, pld_miss_only, always, or draft_len_le_N"
    )


def _queued_residual_triggered(
    *,
    prefix: list[int],
    pld_info: dict[str, object],
    queued: _QueuedResidualDraft,
    config: VantageResidualConfig,
    residual_trigger: Callable[
        [list[int], dict[str, object], VantageResidualConfig], bool
    ]
    | None,
) -> bool:
    """Select queued drafts using only information available before verification."""

    if not config.residual_enabled:
        return False
    info = dict(pld_info)
    info.update(
        {
            "queued_available": True,
            "queued_draft_len": len(queued.draft_tokens),
            "residual_confidence": float(queued.confidence),
            "accepted_len": None,
            "accepted_len_available": False,
        }
    )
    if residual_trigger is not None:
        return bool(residual_trigger(list(prefix), info, config))

    pld_draft_len = int(info.get("draft_len", 0) or 0)
    normalized = config.residual_trigger.lower().strip()
    if normalized in {"never", "false", "off", "disabled"}:
        return False
    if normalized in {"always", "true", "on"}:
        return True
    if normalized in {"queued_conservative", "router", "router_weak", "router_predicted_weak"}:
        return (
            pld_draft_len <= int(config.router_min_pld_draft_len)
            and float(queued.confidence) >= 0.0
        )
    if normalized == "pld_miss_only":
        return pld_draft_len == 0
    match = re.fullmatch(r"draft_len_le_(\d+)", normalized)
    if match:
        return pld_draft_len <= int(match.group(1))
    raise ValueError(
        "queued residual selection cannot use post-verification trigger "
        f"{config.residual_trigger!r}; use queued_conservative, router, "
        "pld_miss_only, always, or draft_len_le_N"
    )


@torch.no_grad()
def vantage_residual_decode(
    prompt_ids: torch.Tensor,
    target,
    *,
    max_new_tokens: int,
    eos_token_ids: list[int],
    config: VantageResidualConfig | None = None,
    residual_model=None,
    residual_trigger: Callable[
        [list[int], dict[str, object], VantageResidualConfig], bool
    ]
    | None = None,
) -> DecodeResult:
    """PLD-first residual scaffold with exact target verification.

    Every PLD and residual draft is accepted only through ``greedy_verify`` on
    target logits.  When residual is disabled or not triggered, the behavior is
    the PLD/target path: prompt-lookup drafts when available, otherwise one
    greedy target token.
    """

    if config is None:
        config = VantageResidualConfig()

    device = _model_device(target)
    prefix = prompt_ids.tolist() if prompt_ids.dim() == 1 else prompt_ids[0].tolist()
    prompt_len = len(prefix)
    eos_set = set(int(token) for token in eos_token_ids)

    target_cache = None
    target_cache_len = 0
    residual_cache = None
    residual_cache_len = 0
    queued_residual_draft: _QueuedResidualDraft | None = None
    steps: list[StepRecord] = []
    step_idx = 0
    t_decode_start = time.perf_counter_ns()

    while len(prefix) - prompt_len < max_new_tokens:
        t_step_start = time.perf_counter_ns()
        step_prefix_len = len(prefix)
        draft_us = 0.0
        verify_us = 0.0
        residual_draft_us = 0.0
        total_accepted = 0
        any_rejected = False
        pld_token01_rejection = False
        residual_was_triggered = False
        residual_drafts: list[int] = []
        pld_emitted: list[int] = []
        residual_emitted: list[int] = []

        t_pld_0 = time.perf_counter_ns()
        pld_drafts, match_len, source_start, follow_start = prompt_lookup_draft(
            prefix,
            max_matching_ngram_size=config.pld_max_matching_ngram_size,
            max_draft_tokens=config.pld_max_draft_tokens,
            min_matching_ngram_size=config.pld_min_matching_ngram_size,
        )
        t_pld_1 = time.perf_counter_ns()
        proposal_us = (t_pld_1 - t_pld_0) / 1000.0

        if config.queued_residual:
            pld_info_pre: dict[str, object] = {
                "draft_len": len(pld_drafts),
                "match_len": match_len,
                "source_start": source_start,
                "follow_start": follow_start,
                "accepted_len_available": False,
            }
            invalid_reason = queued_residual_invalid_reason(prefix, queued_residual_draft)
            queued_available = invalid_reason is None
            queued_used = False
            selected_drafts = list(pld_drafts)
            selected_kind = "pld"
            selected_queue_hash: str | None = None
            queued_draft_len = (
                len(queued_residual_draft.draft_tokens)
                if queued_residual_draft is not None
                else 0
            )
            if queued_available and queued_residual_draft is not None:
                residual_was_triggered = _queued_residual_triggered(
                    prefix=prefix,
                    pld_info=pld_info_pre,
                    queued=queued_residual_draft,
                    config=config,
                    residual_trigger=residual_trigger,
                )
                if residual_was_triggered:
                    selected_drafts = list(queued_residual_draft.draft_tokens)
                    selected_kind = "vantage_residual_queued"
                    selected_queue_hash = queued_residual_draft.prefix_hash
                    queued_used = True
                else:
                    invalid_reason = "router_not_triggered"
            queued_residual_draft = None

            eos_hit = False
            if selected_drafts:
                selected_verify = _verify_drafts(
                    prefix=prefix,
                    target=target,
                    target_cache=target_cache,
                    target_cache_len=target_cache_len,
                    drafts=selected_drafts,
                    device=device,
                )
                target_cache = selected_verify.cache
                target_cache_len = selected_verify.cache_len
                verify_us += selected_verify.verify_us
                total_accepted = int(selected_verify.result.n_accepted_drafts)
                any_rejected = bool(selected_verify.result.rejected)
                emitted, eos_hit = _append_with_limits(
                    prefix=prefix,
                    tokens=selected_verify.result.accepted_tokens,
                    prompt_len=prompt_len,
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=eos_set,
                )
                if selected_kind == "pld":
                    pld_emitted = emitted
                else:
                    residual_emitted = emitted
                    pld_token01_rejection = bool(
                        selected_verify.result.rejected
                        and int(selected_verify.result.n_accepted_drafts) == 0
                    )
                crop_dynamic_cache(target_cache, len(prefix) - 1)
                target_cache_len = len(prefix) - 1 if target_cache is not None else 0
            else:
                target_token, target_cache, target_cache_len, target_us = _emit_one_target_token(
                    prefix=prefix,
                    target=target,
                    target_cache=target_cache,
                    target_cache_len=target_cache_len,
                    device=device,
                )
                verify_us += target_us
                emitted, eos_hit = _append_with_limits(
                    prefix=prefix,
                    tokens=[target_token],
                    prompt_len=prompt_len,
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=eos_set,
                )

            hit_budget = len(prefix) - prompt_len >= max_new_tokens
            created_queue_hash: str | None = None
            created_queue = False
            if (
                config.residual_enabled
                and residual_model is not None
                and not eos_hit
                and not hit_budget
                and len(prefix) > step_prefix_len
            ):
                remaining_budget = prompt_len + max_new_tokens - len(prefix)
                queue_run = _draft_residual_tokens(
                    prefix=prefix,
                    residual_model=residual_model,
                    residual_cache=None,
                    residual_cache_len=0,
                    k=min(config.residual_k, max(0, remaining_budget)),
                )
                residual_drafts = queue_run.drafts
                residual_draft_us += queue_run.draft_us
                draft_us += queue_run.draft_us
                if residual_drafts:
                    created_queue_hash = _prefix_hash(prefix)
                    queued_residual_draft = _QueuedResidualDraft(
                        position=len(prefix),
                        prefix_hash=created_queue_hash,
                        draft_tokens=list(residual_drafts),
                        source_step=step_idx,
                        confidence=1.0,
                    )
                    created_queue = True

            n_emitted = len(prefix) - step_prefix_len
            t_step_end = time.perf_counter_ns()
            steps.append(
                StepRecord(
                    method=config.method_name,
                    step=step_idx,
                    k=len(selected_drafts),
                    n_accepted_drafts=total_accepted,
                    n_emitted=n_emitted,
                    rejected=any_rejected,
                    node_type=None,
                    deepest_type=None,
                    wall_us=(t_step_end - t_step_start) / 1000.0,
                    draft_us=draft_us,
                    verify_us=verify_us,
                    proposal_kind=selected_kind if selected_drafts else "target",
                    proposal_match_len=match_len,
                    proposal_us=proposal_us,
                    proposal_tokens=len(pld_drafts),
                    proposal_source_start_token=source_start if source_start >= 0 else None,
                    proposal_source_end_token=(
                        source_start + match_len if source_start >= 0 else None
                    ),
                    proposal_follow_start_token=follow_start if follow_start >= 0 else None,
                    proposal_follow_end_token=(
                        follow_start + len(pld_drafts) if follow_start >= 0 else None
                    ),
                    proposal_query_len=match_len if match_len > 0 else None,
                    proposal_pool="prompt" if pld_drafts else None,
                    pld_variant="vantage_residual_queued",
                    pld_exact_hit=bool(pld_drafts),
                    pld_variant_triggered=queued_used,
                    pld_candidate_accepted_len=None,
                    pld_token01_rejection=pld_token01_rejection,
                    mtp_triggered=created_queue or queued_used,
                    mtp_predicted_tokens=len(residual_drafts) if residual_drafts else None,
                    mtp_verified_draft_tokens=len(selected_drafts) if queued_used else None,
                    mtp_extra_accepted_drafts=0,
                    mtp_head_compute_us=residual_draft_us,
                    mtp_verify_extra_us=0.0,
                    mtp_total_overhead_us=residual_draft_us,
                    mtp_queue_prediction_created=created_queue,
                    mtp_queue_prediction_used=queued_used,
                    mtp_queue_dropped_pld_strong=invalid_reason == "router_not_triggered",
                    mtp_queue_dropped_position_mismatch=invalid_reason
                    in {"position_mismatch", "prefix_hash_mismatch"},
                    mtp_queue_expired=invalid_reason in {"empty", "eos", "max_length"},
                    mtp_used_token0_rejected=(
                        bool(any_rejected and total_accepted == 0) if queued_used else None
                    ),
                    mtp_extra_verify_calls=0,
                    mtp_normal_verify_reuse=True if queued_used else None,
                    queued_available=queued_available,
                    queued_used=queued_used,
                    queued_invalid_reason=invalid_reason if invalid_reason else None,
                    prefix_hash_created=created_queue_hash,
                    prefix_hash_used=selected_queue_hash,
                    residual_confidence=1.0 if queued_available else None,
                    queued_draft_len=queued_draft_len if queued_draft_len else None,
                    verifier_calls_this_step=1,
                    hidden_capture_us=0.0,
                    residual_head_us=residual_draft_us,
                    hit_max_new_tokens=hit_budget,
                )
            )
            step_idx += 1
            if eos_hit or hit_budget or n_emitted == 0:
                break
            continue

        if config.preverify_replace:
            pld_info_pre: dict[str, object] = {
                "draft_len": len(pld_drafts),
                "match_len": match_len,
                "source_start": source_start,
                "follow_start": follow_start,
                "accepted_len_available": False,
            }
            residual_was_triggered = _preverify_residual_triggered(
                prefix=prefix,
                pld_info=pld_info_pre,
                config=config,
                residual_trigger=residual_trigger,
            )
            selected_drafts = list(pld_drafts)
            selected_kind = "pld"
            if residual_was_triggered and residual_model is not None:
                remaining_budget = prompt_len + max_new_tokens - len(prefix)
                draft_run = _draft_residual_tokens(
                    prefix=prefix,
                    residual_model=residual_model,
                    residual_cache=residual_cache,
                    residual_cache_len=residual_cache_len,
                    k=min(config.residual_k, max(0, remaining_budget)),
                )
                residual_cache = draft_run.cache
                residual_cache_len = draft_run.cache_len
                residual_drafts = draft_run.drafts
                residual_draft_us += draft_run.draft_us
                draft_us += draft_run.draft_us
                if residual_drafts:
                    selected_drafts = list(residual_drafts)
                    selected_kind = "vantage_residual_preverify_replace"

            eos_hit = False
            if selected_drafts:
                selected_verify = _verify_drafts(
                    prefix=prefix,
                    target=target,
                    target_cache=target_cache,
                    target_cache_len=target_cache_len,
                    drafts=selected_drafts,
                    device=device,
                )
                target_cache = selected_verify.cache
                target_cache_len = selected_verify.cache_len
                verify_us += selected_verify.verify_us
                total_accepted = int(selected_verify.result.n_accepted_drafts)
                any_rejected = bool(selected_verify.result.rejected)
                emitted, eos_hit = _append_with_limits(
                    prefix=prefix,
                    tokens=selected_verify.result.accepted_tokens,
                    prompt_len=prompt_len,
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=eos_set,
                )
                if selected_kind == "pld":
                    pld_emitted = emitted
                else:
                    residual_emitted = emitted
                    pld_token01_rejection = bool(
                        selected_verify.result.rejected
                        and int(selected_verify.result.n_accepted_drafts) == 0
                    )
                crop_dynamic_cache(target_cache, len(prefix) - 1)
                target_cache_len = len(prefix) - 1 if target_cache is not None else 0
            else:
                target_token, target_cache, target_cache_len, target_us = _emit_one_target_token(
                    prefix=prefix,
                    target=target,
                    target_cache=target_cache,
                    target_cache_len=target_cache_len,
                    device=device,
                )
                verify_us += target_us
                emitted, eos_hit = _append_with_limits(
                    prefix=prefix,
                    tokens=[target_token],
                    prompt_len=prompt_len,
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=eos_set,
                )

            n_emitted = len(prefix) - step_prefix_len
            t_step_end = time.perf_counter_ns()
            hit_budget = len(prefix) - prompt_len >= max_new_tokens
            steps.append(
                StepRecord(
                    method=config.method_name,
                    step=step_idx,
                    k=len(selected_drafts),
                    n_accepted_drafts=total_accepted,
                    n_emitted=n_emitted,
                    rejected=any_rejected,
                    node_type=None,
                    deepest_type=None,
                    wall_us=(t_step_end - t_step_start) / 1000.0,
                    draft_us=draft_us,
                    verify_us=verify_us,
                    proposal_kind=selected_kind,
                    proposal_match_len=match_len,
                    proposal_us=proposal_us,
                    proposal_tokens=len(pld_drafts),
                    proposal_source_start_token=source_start if source_start >= 0 else None,
                    proposal_source_end_token=(
                        source_start + match_len if source_start >= 0 else None
                    ),
                    proposal_follow_start_token=follow_start if follow_start >= 0 else None,
                    proposal_follow_end_token=(
                        follow_start + len(pld_drafts) if follow_start >= 0 else None
                    ),
                    proposal_query_len=match_len if match_len > 0 else None,
                    proposal_pool="prompt" if pld_drafts else None,
                    pld_variant="vantage_residual_preverify_replace",
                    pld_exact_hit=bool(pld_drafts),
                    pld_variant_triggered=residual_was_triggered,
                    pld_candidate_accepted_len=None,
                    pld_token01_rejection=pld_token01_rejection,
                    mtp_triggered=residual_was_triggered,
                    mtp_predicted_tokens=len(residual_drafts) if residual_drafts else None,
                    mtp_verified_draft_tokens=len(residual_drafts) if residual_drafts else None,
                    mtp_extra_accepted_drafts=0,
                    mtp_head_compute_us=residual_draft_us,
                    mtp_verify_extra_us=0.0,
                    mtp_total_overhead_us=residual_draft_us,
                    mtp_extra_verify_calls=0,
                    mtp_normal_verify_reuse=True if selected_kind != "pld" else None,
                    hit_max_new_tokens=hit_budget,
                )
            )
            step_idx += 1
            if eos_hit or hit_budget or n_emitted == 0:
                break
            continue

        pld_accepted_len = 0
        eos_hit = False
        if pld_drafts:
            pld_verify = _verify_drafts(
                prefix=prefix,
                target=target,
                target_cache=target_cache,
                target_cache_len=target_cache_len,
                drafts=pld_drafts,
                device=device,
            )
            target_cache = pld_verify.cache
            target_cache_len = pld_verify.cache_len
            verify_us += pld_verify.verify_us
            pld_accepted_len = pld_verify.result.n_accepted_drafts
            total_accepted += pld_verify.result.n_accepted_drafts
            any_rejected = any_rejected or pld_verify.result.rejected
            pld_emitted, eos_hit = _append_with_limits(
                prefix=prefix,
                tokens=pld_verify.result.accepted_tokens,
                prompt_len=prompt_len,
                max_new_tokens=max_new_tokens,
                eos_token_ids=eos_set,
            )
            crop_dynamic_cache(target_cache, len(prefix) - 1)
            target_cache_len = len(prefix) - 1 if target_cache is not None else 0

        pld_info: dict[str, object] = {
            "draft_len": len(pld_drafts),
            "accepted_len": pld_accepted_len,
            "match_len": match_len,
            "source_start": source_start,
            "follow_start": follow_start,
            "emitted_len": len(pld_emitted),
            "rejected": any_rejected,
        }
        remaining_budget = prompt_len + max_new_tokens - len(prefix)
        if not eos_hit and remaining_budget > 0:
            residual_was_triggered = _residual_triggered(
                prefix=prefix,
                pld_info=pld_info,
                config=config,
                residual_trigger=residual_trigger,
            )

        if residual_was_triggered and residual_model is not None and remaining_budget > 0:
            residual_k = min(config.residual_k, remaining_budget)
            draft_run = _draft_residual_tokens(
                prefix=prefix,
                residual_model=residual_model,
                residual_cache=residual_cache,
                residual_cache_len=residual_cache_len,
                k=residual_k,
            )
            residual_cache = draft_run.cache
            residual_cache_len = draft_run.cache_len
            residual_drafts = draft_run.drafts
            residual_draft_us += draft_run.draft_us
            draft_us += draft_run.draft_us

            if residual_drafts:
                residual_prefix_len = len(prefix)
                residual_verify = _verify_drafts(
                    prefix=prefix,
                    target=target,
                    target_cache=target_cache,
                    target_cache_len=target_cache_len,
                    drafts=residual_drafts,
                    device=device,
                )
                target_cache = residual_verify.cache
                target_cache_len = residual_verify.cache_len
                verify_us += residual_verify.verify_us
                total_accepted += residual_verify.result.n_accepted_drafts
                any_rejected = any_rejected or residual_verify.result.rejected
                pld_token01_rejection = residual_verify.result.rejected
                residual_emitted, eos_hit = _append_with_limits(
                    prefix=prefix,
                    tokens=residual_verify.result.accepted_tokens,
                    prompt_len=prompt_len,
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=eos_set,
                )
                crop_dynamic_cache(target_cache, len(prefix) - 1)
                target_cache_len = len(prefix) - 1 if target_cache is not None else 0

                target_residual_cache_len = (
                    residual_prefix_len + residual_verify.result.n_accepted_drafts
                )
                crop_dynamic_cache(residual_cache, target_residual_cache_len)
                residual_cache_len = (
                    target_residual_cache_len if residual_cache is not None else 0
                )

        if not pld_emitted and not residual_emitted and not eos_hit:
            target_token, target_cache, target_cache_len, target_us = _emit_one_target_token(
                prefix=prefix,
                target=target,
                target_cache=target_cache,
                target_cache_len=target_cache_len,
                device=device,
            )
            verify_us += target_us
            emitted, eos_hit = _append_with_limits(
                prefix=prefix,
                tokens=[target_token],
                prompt_len=prompt_len,
                max_new_tokens=max_new_tokens,
                eos_token_ids=eos_set,
            )
            if emitted:
                target_cache_len = len(prefix) - 1 if target_cache is not None else 0

        n_emitted = len(prefix) - step_prefix_len
        t_step_end = time.perf_counter_ns()
        hit_budget = len(prefix) - prompt_len >= max_new_tokens
        proposal_kind = (
            "vantage_residual"
            if residual_was_triggered and residual_drafts
            else ("pld" if pld_drafts else "target")
        )
        steps.append(
            StepRecord(
                method=config.method_name,
                step=step_idx,
                k=len(pld_drafts) + len(residual_drafts),
                n_accepted_drafts=total_accepted,
                n_emitted=n_emitted,
                rejected=any_rejected,
                node_type=None,
                deepest_type=None,
                wall_us=(t_step_end - t_step_start) / 1000.0,
                draft_us=draft_us,
                verify_us=verify_us,
                proposal_kind=proposal_kind,
                proposal_match_len=match_len,
                proposal_us=proposal_us,
                proposal_tokens=len(pld_drafts),
                proposal_source_start_token=source_start if source_start >= 0 else None,
                proposal_source_end_token=(
                    source_start + match_len if source_start >= 0 else None
                ),
                proposal_follow_start_token=follow_start if follow_start >= 0 else None,
                proposal_follow_end_token=(
                    follow_start + len(pld_drafts) if follow_start >= 0 else None
                ),
                proposal_query_len=match_len if match_len > 0 else None,
                proposal_pool="prompt" if pld_drafts else None,
                pld_variant="vantage_residual",
                pld_exact_hit=bool(pld_drafts),
                pld_variant_triggered=residual_was_triggered,
                pld_candidate_accepted_len=pld_accepted_len if pld_drafts else None,
                pld_token01_rejection=pld_token01_rejection,
                mtp_triggered=residual_was_triggered,
                mtp_predicted_tokens=len(residual_drafts) if residual_drafts else None,
                mtp_verified_draft_tokens=len(residual_drafts) if residual_drafts else None,
                mtp_extra_accepted_drafts=max(0, total_accepted - pld_accepted_len),
                mtp_head_compute_us=residual_draft_us,
                mtp_verify_extra_us=verify_us,
                hit_max_new_tokens=hit_budget,
            )
        )
        step_idx += 1

        if eos_hit or hit_budget or n_emitted == 0:
            break

    t_decode_end = time.perf_counter_ns()
    return DecodeResult(
        output_token_ids=prefix,
        output_text="",
        n_new_tokens=len(prefix) - prompt_len,
        steps=steps,
        wall_us_total=(t_decode_end - t_decode_start) / 1000.0,
    )


__all__ = [
    "VantageResidualConfig",
    "ResidualStepResult",
    "ResidualTriggerDecision",
    "TokenVerifier",
    "is_vantage_residual_method",
    "vantage_residual_decode",
    "parse_vantage_residual_method",
    "prompt_lookup_draft",
    "queued_residual_invalid_reason",
    "residual_method_to_blazedit_method",
    "should_trigger_residual",
    "verified_residual_step",
]
