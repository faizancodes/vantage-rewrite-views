"""Offline opportunity analysis for syntax-slot PLD.

This is intentionally a diagnostic, not a runtime decoder.  It estimates
whether a syntax-class index can recover long concrete drafts on PLD-weak
steps before we pay the engineering cost of a verifier-integrated path.
"""

from __future__ import annotations

import argparse
import bisect
import io
import json
import keyword
import re
import sys
import time
import tokenize
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asts.code_proposers import encode_no_special  # noqa: E402
from scripts.collect_pld_mtp_training_data import _output_tokens, _steps_by_task  # noqa: E402
from scripts.train_pld_candidate_reranker import _load_jsonl  # noqa: E402


KEYWORD = "KEYWORD"
IDENTIFIER = "IDENTIFIER"
LITERAL = "LITERAL"
OPERATOR = "OPERATOR"
WHITESPACE = "WHITESPACE"
OTHER = "OTHER"
SLOT_CLASSES = {IDENTIFIER, LITERAL}

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NUMBER_RE = re.compile(r"^(?:0[xX][0-9A-Fa-f]+|\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?[jJ]?$")
_PUNCT_RE = re.compile(r"^[\[\]\{\}\(\)\.,:;@=+\-*/%&|^~<>!]+$")


@dataclass(frozen=True)
class TokenizedText:
    ids: list[int]
    classes: list[str]


@dataclass(frozen=True)
class SyntaxStep:
    task_id: str
    step_id: int
    start: int
    emitted: int
    accepted_len: int


@dataclass(frozen=True)
class SyntaxCandidate:
    task_id: str
    step_id: int
    source_name: str
    source_position: int
    match_count: int
    is_unique: bool
    draft_len: int
    accepted_len: int
    uncertain_slot_stop: bool
    slot_fill_attempts: int
    slot_fill_success: int


def classify_token_text(text: str) -> str:
    """Classify a tokenizer piece without tokenizer-offset context."""

    if not text:
        return OTHER
    if text.isspace():
        return WHITESPACE
    stripped = text.strip()
    if not stripped:
        return WHITESPACE
    if keyword.iskeyword(stripped):
        return KEYWORD
    if (
        (stripped.startswith(("'", '"')) and stripped.endswith(("'", '"')))
        or _NUMBER_RE.fullmatch(stripped)
    ):
        return LITERAL
    if _IDENT_RE.fullmatch(stripped):
        return IDENTIFIER
    if _PUNCT_RE.fullmatch(stripped):
        return OPERATOR
    return OTHER


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    running = 0
    for line in text.splitlines(keepends=True):
        running += len(line)
        offsets.append(running)
    return offsets


def _abs_pos(line_offsets: list[int], pos: tuple[int, int]) -> int:
    line, col = pos
    if line <= 0:
        return 0
    if line - 1 >= len(line_offsets):
        return line_offsets[-1]
    return line_offsets[line - 1] + col


def _char_classes_from_python_tokens(text: str) -> list[str]:
    classes = [WHITESPACE if ch.isspace() else OTHER for ch in text]
    line_offsets = _line_offsets(text)
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        for tok in tokens:
            if tok.type in {
                tokenize.ENCODING,
                tokenize.ENDMARKER,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
            }:
                continue
            start = _abs_pos(line_offsets, tok.start)
            end = _abs_pos(line_offsets, tok.end)
            if end <= start:
                continue
            if tok.type == tokenize.NAME:
                cls = KEYWORD if keyword.iskeyword(tok.string) else IDENTIFIER
            elif tok.type in {tokenize.NUMBER, tokenize.STRING}:
                cls = LITERAL
            elif tok.type == tokenize.OP:
                cls = OPERATOR
            else:
                cls = OTHER
            for i in range(max(0, start), min(len(classes), end)):
                classes[i] = cls
    except (IndentationError, SyntaxError, tokenize.TokenError):
        # Real completions can be partial.  The fallback keeps whitespace and
        # punctuation useful without pretending we parsed a valid Python unit.
        for i, ch in enumerate(text):
            if ch.isspace():
                classes[i] = WHITESPACE
            elif ch in "()[]{}.,:;@=+-*/%&|^~<>!":
                classes[i] = OPERATOR
    return classes


def token_classes_for_text(tokenizer: Any, text: str) -> TokenizedText:
    """Return tokenizer ids and coarse syntax classes for each tokenizer piece."""

    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded.input_ids)
    offsets = getattr(encoded, "offset_mapping", None)
    if offsets is None:
        offsets = encoded.get("offset_mapping") if isinstance(encoded, dict) else None
    if offsets is None:
        return TokenizedText(
            ids=ids,
            classes=[classify_token_text(tokenizer.decode([token_id])) for token_id in ids],
        )

    char_classes = _char_classes_from_python_tokens(text)
    classes: list[str] = []
    for token_id, item in zip(ids, offsets, strict=True):
        start, end = int(item[0]), int(item[1])
        span = char_classes[max(0, start) : max(0, min(len(char_classes), end))]
        non_ws = [cls for cls in span if cls != WHITESPACE]
        if non_ws:
            classes.append(Counter(non_ws).most_common(1)[0][0])
        elif span:
            classes.append(WHITESPACE)
        else:
            classes.append(classify_token_text(tokenizer.decode([token_id])))
    return TokenizedText(ids=ids, classes=classes)


def _build_index(classes: list[str], n: int) -> dict[tuple[str, ...], list[int]]:
    index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    if n <= 0:
        return index
    for pos in range(0, max(0, len(classes) - n + 1)):
        index[tuple(classes[pos : pos + n])].append(pos)
    return index


def _find_matches(
    *,
    query: tuple[str, ...],
    sources: list[tuple[str, list[int], list[str]]],
    n: int,
    generated_index_limit: int,
) -> list[tuple[str, list[int], list[str], int]]:
    matches: list[tuple[str, list[int], list[str], int]] = []
    for name, ids, classes in sources:
        max_start = len(classes) - n
        if max_start <= 0:
            continue
        if name == "generated":
            max_start = min(max_start, generated_index_limit - n)
        if max_start <= 0:
            continue
        for pos in range(max_start):
            if tuple(classes[pos : pos + n]) == query:
                matches.append((name, ids, classes, pos))
    return matches


def build_slot_map(
    source_ids: list[int],
    source_classes: list[str],
    query_ids: list[int],
    query_classes: list[str],
) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    for src_id, src_cls, qry_id, qry_cls in zip(
        source_ids, source_classes, query_ids, query_classes, strict=True
    ):
        if src_cls not in SLOT_CLASSES or qry_cls != src_cls:
            continue
        previous = mapping.get(src_id)
        if previous is not None and previous != qry_id:
            return None
        mapping[src_id] = qry_id
    return mapping


def concretize_continuation(
    *,
    source_ids: list[int],
    source_classes: list[str],
    start: int,
    cap: int,
    slot_map: dict[int, int],
    recent_ids: set[int],
) -> tuple[list[int], bool, int, int]:
    draft: list[int] = []
    uncertain_stop = False
    slot_attempts = 0
    slot_success = 0
    for idx in range(start, min(len(source_ids), start + cap)):
        token_id = int(source_ids[idx])
        token_class = source_classes[idx]
        if token_class in SLOT_CLASSES:
            slot_attempts += 1
            if token_id in slot_map:
                draft.append(int(slot_map[token_id]))
                slot_success += 1
                continue
            if token_id in recent_ids:
                draft.append(token_id)
                slot_success += 1
                continue
            uncertain_stop = True
            break
        draft.append(token_id)
    return draft, uncertain_stop, slot_attempts, slot_success


def accepted_prefix_length(draft: list[int], target: list[int]) -> int:
    accepted = 0
    for a, b in zip(draft, target):
        if int(a) != int(b):
            break
        accepted += 1
    return accepted


def _load_steps(path: Path, *, method: str) -> dict[str, list[SyntaxStep]]:
    rows_by_task = _steps_by_task(path, method=method)
    out: dict[str, list[SyntaxStep]] = {}
    for task_id, rows in rows_by_task.items():
        out[task_id] = [
            SyntaxStep(
                task_id=task_id,
                step_id=int(row.get("step") or 0),
                start=int(row.get("_generated_start") or 0),
                emitted=max(1, int(row.get("n_emitted") or 0)),
                accepted_len=int(row.get("n_accepted_drafts") or 0),
            )
            for row in rows
        ]
    return out


def _choose_candidate(
    candidates: list[SyntaxCandidate],
    *,
    require_unique: bool,
    min_concrete_prefix: int,
) -> SyntaxCandidate | None:
    usable = [
        cand
        for cand in candidates
        if cand.draft_len >= min_concrete_prefix and (cand.is_unique or not require_unique)
    ]
    if not usable:
        return None
    return max(
        usable,
        key=lambda cand: (
            cand.accepted_len,
            cand.draft_len,
            int(cand.is_unique),
            -cand.match_count,
            cand.source_name == "reference",
        ),
    )


def _project_steps(
    steps_by_task: dict[str, list[SyntaxStep]],
    candidates: dict[tuple[str, int], SyntaxCandidate],
) -> dict[str, Any]:
    baseline_steps = sum(len(steps) for steps in steps_by_task.values())
    projected_steps = 0
    skipped_steps = 0
    candidate_used = 0
    progress_values: list[int] = []
    accepted_values: list[int] = []
    for _task_id, steps in steps_by_task.items():
        starts = [step.start for step in steps]
        i = 0
        while i < len(steps):
            step = steps[i]
            projected_steps += 1
            selected_progress = step.emitted
            cand = candidates.get((step.task_id, step.step_id))
            if cand is not None:
                syntax_progress = cand.accepted_len + 1
                if syntax_progress > selected_progress:
                    candidate_used += 1
                    selected_progress = syntax_progress
                    progress_values.append(syntax_progress)
                    accepted_values.append(cand.accepted_len)
            if selected_progress <= step.emitted:
                i += 1
                continue
            covered_until = step.start + selected_progress
            j = bisect.bisect_left(starts, covered_until, lo=i + 1)
            skipped_steps += max(0, j - (i + 1))
            i = max(j, i + 1)
    projected_steps = max(1, projected_steps)
    return {
        "baseline_steps": baseline_steps,
        "projected_steps": projected_steps,
        "projected_speedup": baseline_steps / projected_steps,
        "step_reduction_pct": 100.0 * (baseline_steps - projected_steps) / max(1, baseline_steps),
        "skipped_baseline_steps": skipped_steps,
        "syntax_candidates_used_for_projection": candidate_used,
        "projected_avg_syntax_progress": sum(progress_values) / len(progress_values)
        if progress_values
        else 0.0,
        "projected_avg_syntax_accepted_len": sum(accepted_values) / len(accepted_values)
        if accepted_values
        else 0.0,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    m = payload["metrics"]
    p = payload["projection"]
    lines = [
        "# Syntax-Slot PLD Opportunity",
        "",
        f"steps: `{payload['steps']}`",
        f"completions: `{payload['completions']}`",
        f"method: `{payload['method']}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| PLD weak/miss attempts | {m['syntax_attempts']} |",
        f"| syntax pattern hits | {m['syntax_hits']} |",
        f"| unique syntax hits | {m['syntax_unique_hits']} |",
        f"| syntax collisions | {m['syntax_collision_count']} |",
        f"| drafts verified offline | {m['syntax_drafts_verified']} |",
        f"| mean accepted draft len | {m['syntax_accepted_len_mean']:.2f} |",
        f"| token0/1 reject rate | {100.0 * m['syntax_tok0_1_reject_rate']:.1f}% |",
        f"| slot fill success rate | {100.0 * m['syntax_slot_fill_success_rate']:.1f}% |",
        f"| overhead us/attempt | {m['syntax_overhead_us_per_step']:.1f} |",
        "",
        "| projection | value |",
        "|---|---:|",
        f"| baseline steps | {p['baseline_steps']} |",
        f"| projected steps | {p['projected_steps']} |",
        f"| projected speedup | {p['projected_speedup']:.3f}x |",
        f"| step reduction | {p['step_reduction_pct']:.1f}% |",
        f"| skipped baseline steps | {p['skipped_baseline_steps']} |",
        "",
        f"Decision: **{payload['decision']}**",
    ]
    path.write_text("\n".join(lines) + "\n")


def _bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() not in {"0", "false", "no", "off"}


def _first_n(items: Iterable[Any], n: int) -> list[Any]:
    out: list[Any] = []
    for item in items:
        out.append(item)
        if len(out) >= n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    ap.add_argument("--steps", type=Path, required=True)
    ap.add_argument("--completions", type=Path, required=True)
    ap.add_argument("--method", default="blazedit_pld_w128_n10")
    ap.add_argument("--output-dir", type=Path, default=Path("/tmp/pld_mtp/syntax_slot_opportunity"))
    ap.add_argument("--syntax-n", type=int, default=12)
    ap.add_argument("--syntax-draft-cap", type=int, default=48)
    ap.add_argument("--syntax-require-unique", type=_bool_arg, default=True)
    ap.add_argument("--syntax-min-concrete-prefix", type=int, default=8)
    ap.add_argument("--syntax-pld-weak-threshold", type=int, default=4)
    ap.add_argument("--syntax-include-prompt", type=_bool_arg, default=True)
    ap.add_argument("--syntax-include-generated", type=_bool_arg, default=True)
    ap.add_argument("--max-tasks", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    completions = {str(row["task_id"]): row for row in _load_jsonl(args.completions)}
    steps_by_task = _load_steps(args.steps, method=args.method)
    task_ids = [task_id for task_id in sorted(steps_by_task) if task_id in completions]
    if args.max_tasks > 0:
        task_ids = task_ids[: args.max_tasks]
        steps_by_task = {task_id: steps_by_task[task_id] for task_id in task_ids}
    else:
        steps_by_task = {task_id: steps_by_task[task_id] for task_id in task_ids}

    metrics = Counter()
    accepted_values: list[int] = []
    draft_lens: list[int] = []
    concrete_prefix_values: list[int] = []
    per_step_candidates: dict[tuple[str, int], SyntaxCandidate] = {}
    examples: list[dict[str, Any]] = []
    total_attempt_us = 0.0

    for task_id in task_ids:
        comp = completions[task_id]
        output_ids = _output_tokens(tokenizer, comp, args.method)
        if not output_ids:
            continue
        output_text = str(
            ((comp.get("outputs") or {}).get(args.method) or {}).get("text")
            or ((comp.get("outputs") or {}).get(args.method) or {}).get("raw_text")
            or ""
        )
        if not output_text:
            output_text = tokenizer.decode(output_ids)
        output_tok = token_classes_for_text(tokenizer, output_text)
        if len(output_tok.ids) != len(output_ids):
            # Keep positions aligned with the exact tokens used by the step trace.
            output_tok = TokenizedText(
                ids=output_ids,
                classes=[classify_token_text(tokenizer.decode([x])) for x in output_ids],
            )
        reference_tok = token_classes_for_text(tokenizer, str(comp.get("reference") or ""))
        prompt_tok = token_classes_for_text(tokenizer, str(comp.get("prompt") or ""))

        static_sources: list[tuple[str, list[int], list[str]]] = [
            ("reference", reference_tok.ids, reference_tok.classes)
        ]
        if args.syntax_include_prompt:
            static_sources.append(("prompt", prompt_tok.ids, prompt_tok.classes))

        for step in steps_by_task[task_id]:
            if step.accepted_len > args.syntax_pld_weak_threshold:
                continue
            if step.start < args.syntax_n or step.start >= len(output_ids):
                continue
            metrics["syntax_attempts"] += 1
            query_ids = output_ids[step.start - args.syntax_n : step.start]
            query_classes = output_tok.classes[step.start - args.syntax_n : step.start]
            if len(query_classes) != args.syntax_n:
                continue
            sources = list(static_sources)
            if args.syntax_include_generated:
                sources.append(("generated", output_ids, output_tok.classes))
            t0 = time.perf_counter()
            matches = _find_matches(
                query=tuple(query_classes),
                sources=sources,
                n=args.syntax_n,
                generated_index_limit=max(0, step.start - args.syntax_n),
            )
            total_attempt_us += (time.perf_counter() - t0) * 1_000_000.0
            if not matches:
                continue
            metrics["syntax_hits"] += 1
            metrics["syntax_unique_hits"] += int(len(matches) == 1)
            metrics["syntax_collision_count"] += max(0, len(matches) - 1)
            step_candidates: list[SyntaxCandidate] = []
            recent_ids = set(query_ids)
            for source_name, source_ids, source_classes, pos in matches:
                slot_map = build_slot_map(
                    source_ids[pos : pos + args.syntax_n],
                    source_classes[pos : pos + args.syntax_n],
                    query_ids,
                    query_classes,
                )
                if slot_map is None:
                    metrics["syntax_inconsistent_slot_map"] += 1
                    continue
                draft, uncertain_stop, slot_attempts, slot_success = concretize_continuation(
                    source_ids=source_ids,
                    source_classes=source_classes,
                    start=pos + args.syntax_n,
                    cap=args.syntax_draft_cap,
                    slot_map=slot_map,
                    recent_ids=recent_ids,
                )
                metrics["syntax_slot_fill_attempts"] += slot_attempts
                metrics["syntax_slot_fill_success"] += slot_success
                metrics["syntax_uncertain_slot_stops"] += int(uncertain_stop)
                if not draft:
                    continue
                accepted = accepted_prefix_length(draft, output_ids[step.start :])
                cand = SyntaxCandidate(
                    task_id=task_id,
                    step_id=step.step_id,
                    source_name=source_name,
                    source_position=pos,
                    match_count=len(matches),
                    is_unique=len(matches) == 1,
                    draft_len=len(draft),
                    accepted_len=accepted,
                    uncertain_slot_stop=uncertain_stop,
                    slot_fill_attempts=slot_attempts,
                    slot_fill_success=slot_success,
                )
                step_candidates.append(cand)
            chosen = _choose_candidate(
                step_candidates,
                require_unique=args.syntax_require_unique,
                min_concrete_prefix=args.syntax_min_concrete_prefix,
            )
            if chosen is None:
                continue
            per_step_candidates[(task_id, step.step_id)] = chosen
            metrics["syntax_drafts_verified"] += 1
            metrics["syntax_tok0_1_reject_count"] += int(chosen.accepted_len <= 1)
            accepted_values.append(chosen.accepted_len)
            draft_lens.append(chosen.draft_len)
            concrete_prefix_values.append(chosen.draft_len)
            if len(examples) < 20:
                examples.append(
                    {
                        "task_id": task_id,
                        "step_id": step.step_id,
                        "start": step.start,
                        "baseline_accepted_len": step.accepted_len,
                        "baseline_emitted": step.emitted,
                        "source": chosen.source_name,
                        "source_position": chosen.source_position,
                        "match_count": chosen.match_count,
                        "draft_len": chosen.draft_len,
                        "accepted_len": chosen.accepted_len,
                        "query_classes": query_classes,
                        "future_text": tokenizer.decode(output_ids[step.start : step.start + 32]),
                    }
                )

    projection = _project_steps(steps_by_task, per_step_candidates)
    attempts = max(1, int(metrics["syntax_attempts"]))
    hit_rate = metrics["syntax_hits"] / attempts
    unique_rate = metrics["syntax_unique_hits"] / attempts
    slot_attempts = max(1, int(metrics["syntax_slot_fill_attempts"]))
    metrics_payload = {
        "syntax_attempts": int(metrics["syntax_attempts"]),
        "syntax_hits": int(metrics["syntax_hits"]),
        "syntax_hit_rate": hit_rate,
        "syntax_unique_hits": int(metrics["syntax_unique_hits"]),
        "syntax_unique_hit_rate": unique_rate,
        "syntax_collision_count": int(metrics["syntax_collision_count"]),
        "syntax_slot_fill_attempts": int(metrics["syntax_slot_fill_attempts"]),
        "syntax_slot_fill_success": int(metrics["syntax_slot_fill_success"]),
        "syntax_slot_fill_success_rate": metrics["syntax_slot_fill_success"] / slot_attempts,
        "syntax_uncertain_slot_stops": int(metrics["syntax_uncertain_slot_stops"]),
        "syntax_inconsistent_slot_map": int(metrics["syntax_inconsistent_slot_map"]),
        "syntax_drafts_verified": int(metrics["syntax_drafts_verified"]),
        "syntax_accepted_len_sum": int(sum(accepted_values)),
        "syntax_accepted_len_mean": sum(accepted_values) / len(accepted_values)
        if accepted_values
        else 0.0,
        "syntax_draft_len_mean": sum(draft_lens) / len(draft_lens) if draft_lens else 0.0,
        "syntax_concrete_prefix_len_mean": sum(concrete_prefix_values) / len(concrete_prefix_values)
        if concrete_prefix_values
        else 0.0,
        "syntax_tok0_1_reject_count": int(metrics["syntax_tok0_1_reject_count"]),
        "syntax_tok0_1_reject_rate": metrics["syntax_tok0_1_reject_count"]
        / max(1, int(metrics["syntax_drafts_verified"])),
        "syntax_overhead_us_per_step": total_attempt_us / attempts,
    }
    passes = (
        metrics_payload["syntax_hit_rate"] >= 0.20
        and metrics_payload["syntax_slot_fill_success_rate"] >= 0.60
        and metrics_payload["syntax_accepted_len_mean"] >= 8.0
        and projection["projected_speedup"] >= 1.20
    )
    if passes:
        decision = "continue syntax-slot with runtime prototype"
    elif metrics_payload["syntax_hit_rate"] >= 0.20 and metrics_payload["syntax_accepted_len_mean"] < 8.0:
        decision = "tune slot filling; syntax matches exist but concrete prefix is too short"
    else:
        decision = "abandon this syntax-slot configuration"
    payload = {
        "steps": str(args.steps),
        "completions": str(args.completions),
        "method": args.method,
        "config": {
            "syntax_n": args.syntax_n,
            "syntax_draft_cap": args.syntax_draft_cap,
            "syntax_require_unique": bool(args.syntax_require_unique),
            "syntax_min_concrete_prefix": args.syntax_min_concrete_prefix,
            "syntax_pld_weak_threshold": args.syntax_pld_weak_threshold,
            "syntax_include_prompt": bool(args.syntax_include_prompt),
            "syntax_include_generated": bool(args.syntax_include_generated),
        },
        "metrics": metrics_payload,
        "projection": projection,
        "examples": examples,
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(args.output_dir / "report.md", payload)
    print((args.output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
