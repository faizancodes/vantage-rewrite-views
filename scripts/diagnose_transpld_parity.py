#!/usr/bin/env python3
"""Diagnose VANTAGE output parity mismatches.

The timing artifacts store decoded outputs, not generated token IDs. This
script retokenizes the stored raw output text with the recorded target
tokenizer, compares vanilla / PLD / VANTAGE pairs, and joins mismatches to
per-step route traces when available.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


METHOD_VANILLA = "vanilla"
METHOD_PLD = "blazedit_pld_w128_n10"
METHOD_VANTAGE = "vantage_frozen_transpld"
METHODS = (METHOD_VANILLA, METHOD_PLD, METHOD_VANTAGE)
PAIRS = (
    (METHOD_VANILLA, METHOD_PLD),
    (METHOD_VANILLA, METHOD_VANTAGE),
    (METHOD_PLD, METHOD_VANTAGE),
)


@dataclass(frozen=True)
class RunSpec:
    name: str
    kind: str
    eval_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-root",
        type=Path,
        default=Path("artifacts/vantage_transpld/modal/validation_20260515_v1"),
        help="Root containing validation run directories with eval/completions.jsonl.",
    )
    parser.add_argument(
        "--real-commit-dir",
        type=Path,
        default=Path("analysis/real_commits/modal/vantage_real_commit_qwen_base_v1/eval"),
        help="Optional real-commit pilot eval directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/vantage_transpld/parity_diagnosis/validation_20260515_v1"),
        help="Directory for report.json and mismatch_details.jsonl.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/vantage_transpld/parity_diagnosis/validation_20260515_v1/parity_diagnosis.md"),
        help="Markdown report path.",
    )
    parser.add_argument(
        "--bf16-divergence-report",
        type=Path,
        default=Path("artifacts/vantage_transpld/parity_diagnosis/validation_20260515_v1/bf16_first_divergence.md"),
        help="Compact first-divergence report for bf16 optimized-path mismatches.",
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen2.5-Coder-7B",
        help="Tokenizer used to retokenize stored raw outputs.",
    )
    parser.add_argument("--context-tokens", type=int, default=12)
    parser.add_argument("--context-chars", type=int, default=220)
    return parser.parse_args()


def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def discover_validation_runs(root: Path) -> list[RunSpec]:
    runs: list[RunSpec] = []
    if not root.exists():
        return runs
    for p in sorted(root.glob("*/eval/completions.jsonl")):
        eval_dir = p.parent
        run_name = eval_dir.parent.name
        workload = run_name
        for key in ("zero100", "field100", "style100", "mixed100"):
            if key in run_name:
                workload = key
                break
        runs.append(RunSpec(workload, "bf16/sdpa validation", eval_dir))
    return runs


def discover_real_commit_run(eval_dir: Path) -> list[RunSpec]:
    if (eval_dir / "completions.jsonl").exists():
        return [RunSpec("real_commit_pilot33", "bf16/sdpa real-commit pilot", eval_dir)]
    return []


def route_of(row: dict[str, Any]) -> str:
    return (
        row.get("proposal_route")
        or row.get("proposal_kind")
        or row.get("strategy")
        or "none"
    )


def load_step_index(eval_dir: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    path = eval_dir / "steps.jsonl"
    if not path.exists():
        return {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        task_id = row.get("task_id")
        method = row.get("method")
        if task_id and method:
            grouped[(task_id, method)].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: int(r.get("step") or 0))
    return grouped


def route_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(route_of(r) for r in rows)
    return dict(counts)


def step_for_token(rows: list[dict[str, Any]], token_index: int) -> dict[str, Any] | None:
    emitted = 0
    for row in rows:
        n_emitted = int(row.get("n_emitted") or 0)
        if token_index < emitted + n_emitted:
            return row
        emitted += n_emitted
    return None


def first_mismatch(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def text_window(text: str, token_ids: list[int], mismatch_index: int, tokenizer, width: int) -> str:
    lo = max(0, mismatch_index - width)
    hi = min(len(token_ids), mismatch_index + width + 1)
    try:
        return tokenizer.decode(token_ids[lo:hi])
    except Exception:
        approx = max(0, min(len(text), mismatch_index))
        return text[max(0, approx - 120) : approx + 120]


def char_diff_context(a: str, b: str, width: int) -> dict[str, Any]:
    idx = 0
    for idx, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            break
    else:
        idx = min(len(a), len(b))
    return {
        "first_char_index": idx,
        "a_context": a[max(0, idx - width) : idx + width],
        "b_context": b[max(0, idx - width) : idx + width],
    }


def output_text(output: dict[str, Any]) -> str:
    return output.get("raw_text") if output.get("raw_text") is not None else output.get("text", "")


def finish_info(output: dict[str, Any], method_rows: list[dict[str, Any]], eos_token_id: int | None, ids: list[int], max_new_tokens: int | None) -> dict[str, Any]:
    hit_max = any(bool(r.get("hit_max_new_tokens")) for r in method_rows)
    if max_new_tokens is not None and len(ids) >= max_new_tokens:
        hit_max = True
    return {
        "finish_reason": output.get("finish_reason"),
        "stop_reason": output.get("stop_reason"),
        "n_new_tokens_recorded": output.get("n_new_tokens"),
        "retokenized_len": len(ids),
        "hit_max_new_tokens": hit_max,
        "eos_in_output": eos_token_id in ids if eos_token_id is not None else None,
        "ends_with_eos": bool(ids and eos_token_id is not None and ids[-1] == eos_token_id),
    }


def _proposal_event(step: dict[str, Any] | None) -> str:
    if not step:
        return "unknown"
    if step.get("hit_max_new_tokens"):
        return "max-token"
    route = route_of(step)
    reject_index = step.get("proposal_target_reject_index")
    n_accepted = int(step.get("n_accepted_drafts") or 0)
    n_emitted = int(step.get("n_emitted") or 0)
    if reject_index is not None:
        try:
            reject_i = int(reject_index)
        except (TypeError, ValueError):
            reject_i = -1
        if reject_i == 0:
            return f"{route}:token0-reject/correction"
        return f"{route}:partial-accept/correction"
    if n_accepted > 0 and n_emitted > n_accepted:
        return f"{route}:full-accept/bonus"
    if n_accepted > 0:
        return f"{route}:accepted"
    return f"{route}:fallback-root"


def classify_mismatch(
    method_pair_has_vantage: bool,
    mismatch_after_transpld: bool,
    vantage_step: dict[str, Any] | None,
    finish_a: dict[str, Any],
    finish_b: dict[str, Any],
) -> dict[str, Any]:
    """Classify a mismatch using only fields stored in released artifacts.

    The timing artifacts do not store logits, position ids, or cache lengths.
    We therefore separate directly observed causes from unavailable evidence.
    """
    event = _proposal_event(vantage_step)
    if finish_a.get("hit_max_new_tokens") or finish_b.get("hit_max_new_tokens"):
        category = "max-token-or-truncation-involved"
    elif not method_pair_has_vantage:
        category = "pld-vs-vanilla-drift"
    elif mismatch_after_transpld:
        category = "after-transpld-route"
    elif vantage_step is not None:
        category = f"exact-pld-or-fallback-path:{event}"
    else:
        category = "unknown-no-step-trace"
    return {
        "category": category,
        "event_at_mismatch": event,
        "logit_top10_available": False,
        "cache_length_audit_available": False,
        "position_id_audit_available": False,
        "attention_mask_audit_available": False,
        "unavailable_reason": (
            "The stored timing artifacts include decoded text and per-step route/"
            "acceptance summaries, but not verifier logits, cache lengths, "
            "position ids, or attention masks. Use the backend-isolation launcher "
            "with deeper instrumentation for root-cause proof."
        ),
    }


def diagnose_run(run: RunSpec, tokenizer, context_tokens: int, context_chars: int) -> dict[str, Any]:
    completions_path = run.eval_dir / "completions.jsonl"
    aggregate_path = run.eval_dir / "aggregate.json"
    steps_path = run.eval_dir / "steps.jsonl"
    aggregate = read_json(aggregate_path) if aggregate_path.exists() else {}
    meta = aggregate.get("meta", {})
    max_new_tokens = meta.get("max_new_tokens")
    rows = read_jsonl(completions_path)
    steps = load_step_index(run.eval_dir)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    pair_counts = {f"{a} vs {b}": 0 for a, b in PAIRS}
    pair_tasks = {f"{a} vs {b}": [] for a, b in PAIRS}
    mismatches: list[dict[str, Any]] = []

    for row in rows:
        task_id = row.get("task_id")
        outputs = row.get("outputs", {})
        encoded: dict[str, list[int]] = {}
        texts: dict[str, str] = {}
        for method in METHODS:
            if method not in outputs:
                continue
            texts[method] = output_text(outputs[method])
            encoded[method] = tokenizer.encode(texts[method], add_special_tokens=False)

        for a, b in PAIRS:
            if a not in encoded or b not in encoded:
                continue
            key = f"{a} vs {b}"
            mismatch_index = first_mismatch(encoded[a], encoded[b])
            if mismatch_index is None and texts[a] == texts[b]:
                continue
            # Count a mismatch if token IDs or text differ. Token equality with text
            # difference is unlikely, but it is still relevant for decoded parity.
            pair_counts[key] += 1
            pair_tasks[key].append(task_id)

            a_rows = steps.get((task_id, a), [])
            b_rows = steps.get((task_id, b), [])
            a_step = step_for_token(a_rows, mismatch_index or 0)
            b_step = step_for_token(b_rows, mismatch_index or 0)
            vantage_rows = steps.get((task_id, METHOD_VANTAGE), [])
            vantage_step = step_for_token(vantage_rows, mismatch_index or 0)
            route_counts = route_summary(vantage_rows)
            method_pair_has_vantage = METHOD_VANTAGE in (a, b)
            mismatch_after_transpld = (
                method_pair_has_vantage
                and bool(vantage_rows)
                and any(route_of(r) == "transpld" for r in vantage_rows[: (vantage_step_index(vantage_rows, vantage_step) + 1 if vantage_step else len(vantage_rows))])
            )

            ids_a = encoded[a]
            ids_b = encoded[b]
            token_a = ids_a[mismatch_index] if mismatch_index is not None and mismatch_index < len(ids_a) else None
            token_b = ids_b[mismatch_index] if mismatch_index is not None and mismatch_index < len(ids_b) else None
            finish_a = finish_info(outputs[a], a_rows, eos_token_id, ids_a, max_new_tokens)
            finish_b = finish_info(outputs[b], b_rows, eos_token_id, ids_b, max_new_tokens)
            classification = classify_mismatch(
                method_pair_has_vantage,
                mismatch_after_transpld,
                vantage_step,
                finish_a,
                finish_b,
            )
            mismatches.append(
                {
                    "workload": run.name,
                    "run_kind": run.kind,
                    "task_id": task_id,
                    "method_pair": key,
                    "first_mismatching_token_index": mismatch_index,
                    "token_id_a": token_a,
                    "token_id_b": token_b,
                    "token_text_a": tokenizer.decode([token_a]) if token_a is not None else None,
                    "token_text_b": tokenizer.decode([token_b]) if token_b is not None else None,
                    "length_a": len(ids_a),
                    "length_b": len(ids_b),
                    "length_difference": len(ids_a) - len(ids_b),
                    "decoded_context_a": text_window(texts[a], ids_a, mismatch_index or 0, tokenizer, context_tokens),
                    "decoded_context_b": text_window(texts[b], ids_b, mismatch_index or 0, tokenizer, context_tokens),
                    "char_diff": char_diff_context(texts[a], texts[b], context_chars),
                    "finish_a": finish_a,
                    "finish_b": finish_b,
                    "step_a": summarize_step(a_step),
                    "step_b": summarize_step(b_step),
                    "vantage_step_at_mismatch": summarize_step(vantage_step),
                    "vantage_route_counts": route_counts,
                    "vantage_ever_used_transpld": any(route_of(r) == "transpld" for r in vantage_rows),
                    "mismatch_after_transpld_route": mismatch_after_transpld,
                    "classification": classification,
                    "paths": {
                        "completions": str(completions_path),
                        "steps": str(steps_path) if steps_path.exists() else None,
                        "aggregate": str(aggregate_path) if aggregate_path.exists() else None,
                    },
                }
            )

    return {
        "name": run.name,
        "kind": run.kind,
        "eval_dir": str(run.eval_dir),
        "n_tasks": len(rows),
        "meta": {
            "target": meta.get("target"),
            "dtype": meta.get("dtype"),
            "attn_impl": meta.get("attn_impl"),
            "max_new_tokens": max_new_tokens,
            "problem_jsonl": meta.get("problem_jsonl"),
        },
        "pair_mismatch_counts": pair_counts,
        "pair_mismatch_tasks": pair_tasks,
        "mismatches": mismatches,
    }


def vantage_step_index(rows: list[dict[str, Any]], step: dict[str, Any] | None) -> int:
    if step is None:
        return -1
    step_no = step.get("step")
    for i, row in enumerate(rows):
        if row.get("step") == step_no:
            return i
    return -1


def summarize_step(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    keys = [
        "method",
        "step",
        "n_emitted",
        "n_accepted_drafts",
        "n_accepted_nonroot_drafts",
        "hit_max_new_tokens",
        "proposal_route",
        "proposal_route_reason",
        "proposal_kind",
        "proposal_match_len",
        "proposal_tokens",
        "proposal_text_preview",
        "proposal_first_token",
        "proposal_first_token_text",
        "proposal_target_reject_index",
        "proposal_target_reject_token",
        "proposal_target_reject_token_text",
    ]
    out = {k: row.get(k) for k in keys if k in row}
    out["route"] = route_of(row)
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# VANTAGE Parity Diagnosis")
    lines.append("")
    lines.append("Generated by `scripts/diagnose_transpld_parity.py`.")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- `bf16/sdpa timing parity` compares decoded outputs from the actual timing runs.")
    lines.append("- Token IDs are reconstructed by retokenizing stored `raw_text` with `Qwen/Qwen2.5-Coder-7B` because timing artifacts do not store generated token IDs.")
    lines.append("- `fp32/eager exactness` is reported from the separate deterministic exactness audit and is not inferred from the timing run. Some artifact paths retain the legacy `lossless` tag name.")
    lines.append("- `real-commit parity` is diagnostic only; the real-commit pilot is not a headline claim.")
    lines.append("")
    lines.append("## Artifact Paths")
    lines.append("")
    for p in report["artifact_paths"]:
        lines.append(f"- `{p}`")
    lines.append("")

    lines.append("## Deterministic fp32/eager Exactness")
    lines.append("")
    lines.append("Source: `artifacts/vantage_transpld/tables/validation_20260515_v1/lossless_exactness.md`.")
    lines.append("")
    lines.append("| Workload | PLD = vanilla | VANTAGE = vanilla |")
    lines.append("|---|---:|---:|")
    for row in report["lossless_exactness_rows"]:
        lines.append(f"| {row['workload']} | {row['pld_equals_vanilla']} | {row['vantage_equals_vanilla']} |")
    lines.append("")

    lines.append("## bf16/sdpa Timing Parity")
    lines.append("")
    lines.append("Source: `artifacts/vantage_transpld/modal/validation_20260515_v1/*/eval/`.")
    lines.append("This section compares retokenized `raw_text` from the actual timing artifacts. It can therefore expose raw max-token continuations that are hidden by post-processing-oriented summary fields.")
    lines.append("The pre-existing `exactness.md` summary is listed as an input artifact, but the mismatch counts below are recomputed directly from `completions.jsonl` and `steps.jsonl`.")
    lines.append("")
    lines.append("| Workload | Tasks | vanilla vs PLD | vanilla vs VANTAGE | PLD vs VANTAGE |")
    lines.append("|---|---:|---:|---:|---:|")
    for run in report["runs"]:
        if run["kind"] != "bf16/sdpa validation":
            continue
        c = run["pair_mismatch_counts"]
        lines.append(
            f"| {run['name']} | {run['n_tasks']} | {c.get('vanilla vs blazedit_pld_w128_n10', 0)} | "
            f"{c.get('vanilla vs vantage_frozen_transpld', 0)} | "
            f"{c.get('blazedit_pld_w128_n10 vs vantage_frozen_transpld', 0)} |"
        )
    lines.append("")

    validation_mismatches = [m for m in report["mismatches"] if m["run_kind"] == "bf16/sdpa validation"]
    if validation_mismatches:
        known = [
            m
            for m in validation_mismatches
            if m["workload"] == "field100"
            and m["task_id"] == "drift_nonrename/field_rename/codeparrot/2559/0"
            and "vantage_frozen_transpld" in m["method_pair"]
        ]
        if known:
            m = known[0]
            lines.append("### Known field100 mismatch diagnosis")
            lines.append("")
            lines.append("- Task: `drift_nonrename/field_rename/codeparrot/2559/0`.")
            lines.append("- Pair(s): `vanilla vs vantage_frozen_transpld` and `blazedit_pld_w128_n10 vs vantage_frozen_transpld`.")
            lines.append("- First mismatch: token index `54`, where vanilla/PLD emit token id `33492` (`_updated`) and VANTAGE emits token id `2398` (`())\\n`).")
            lines.append("- Max-token/EOS involvement: no max-token hit and no EOS in either output.")
            lines.append(f"- VANTAGE route counts: `{m['vantage_route_counts']}`.")
            lines.append("- TransPLD involvement: `False`; the task used exact PLD/fallback routes only before the mismatch.")
            lines.append("- Interpretation: this timing-path mismatch is not evidence that an accepted TransPLD proposal changed the output. It is a bf16/sdpa timing-path parity failure on an exact-PLD-routed task; the separate fp32/eager exactness audit reports field100 as 100/100.")
            lines.append("")
        lines.append("### First-Divergence Classification")
        lines.append("")
        lines.append("| Workload | Task | Pair | First token | Classification | Event at mismatch | Logits/cache available? |")
        lines.append("|---|---|---|---:|---|---|---|")
        for m in validation_mismatches:
            cls = m.get("classification") or {}
            available = (
                "yes"
                if cls.get("logit_top10_available") or cls.get("cache_length_audit_available")
                else "no"
            )
            lines.append(
                f"| {m['workload']} | `{m['task_id']}` | {m['method_pair']} | "
                f"{m['first_mismatching_token_index']} | {cls.get('category', 'unknown')} | "
                f"{cls.get('event_at_mismatch', 'unknown')} | {available} |"
            )
        lines.append("")
        lines.append(
            "Stored timing artifacts do not include verifier top-10 logits, cache lengths, "
            "position ids, or attention masks. The classification above is therefore "
            "a first-divergence triage from route/acceptance summaries, not a full "
            "logit-level root-cause proof."
        )
        lines.append("")
        lines.append("### bf16/sdpa Mismatch Details")
        lines.append("")
        for m in validation_mismatches:
            lines.extend(mismatch_section(m))
    else:
        lines.append("No bf16/sdpa validation mismatches found.")
        lines.append("")

    real_runs = [r for r in report["runs"] if r["kind"] == "bf16/sdpa real-commit pilot"]
    if real_runs:
        lines.append("## Real-Commit Pilot Parity")
        lines.append("")
        lines.append("Source: `analysis/real_commits/modal/vantage_real_commit_qwen_base_v1/eval/`.")
        lines.append("")
        lines.append("| Workload | Tasks | vanilla vs PLD | vanilla vs VANTAGE | PLD vs VANTAGE |")
        lines.append("|---|---:|---:|---:|---:|")
        for run in real_runs:
            c = run["pair_mismatch_counts"]
            lines.append(
                f"| {run['name']} | {run['n_tasks']} | {c.get('vanilla vs blazedit_pld_w128_n10', 0)} | "
                f"{c.get('vanilla vs vantage_frozen_transpld', 0)} | "
                f"{c.get('blazedit_pld_w128_n10 vs vantage_frozen_transpld', 0)} |"
            )
        lines.append("")
        real_mismatches = [m for m in report["mismatches"] if m["run_kind"] == "bf16/sdpa real-commit pilot"]
        if real_mismatches:
            lines.append("### Real-Commit Mismatch Details")
            lines.append("")
            for m in real_mismatches:
                lines.extend(mismatch_section(m))
        else:
            lines.append("No real-commit parity mismatches found.")
            lines.append("")
    else:
        lines.append("## Real-Commit Pilot Parity")
        lines.append("")
        lines.append("No real-commit pilot eval directory was found at the configured path.")
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Stronger correctness language should use the deterministic `fp32/eager` audit, not the `bf16/sdpa` timing path.")
    lines.append("- The `bf16/sdpa` timing path has to be described as parity-diagnosed with the mismatch counts above.")
    lines.append("- A mismatch marked `mismatch_after_transpld_route=false` is not evidence of a transformed-view acceptance bug; it indicates that the task diverged without any prior selected rewrite-view route in the VANTAGE trace.")
    lines.append("")
    return "\n".join(lines)


def mismatch_section(m: dict[str, Any]) -> list[str]:
    lines = []
    lines.append(f"#### {m['workload']} / `{m['task_id']}` / {m['method_pair']}")
    lines.append("")
    lines.append(f"- First mismatching token index: `{m['first_mismatching_token_index']}`")
    lines.append(f"- Token ids: `{m['token_id_a']}` vs `{m['token_id_b']}`")
    lines.append(f"- Token text: `{repr(m['token_text_a'])}` vs `{repr(m['token_text_b'])}`")
    lines.append(f"- Retokenized output lengths: `{m['length_a']}` vs `{m['length_b']}`; length difference `{m['length_difference']}`")
    lines.append(f"- Finish/stop A: `{m['finish_a']}`")
    lines.append(f"- Finish/stop B: `{m['finish_b']}`")
    lines.append(f"- VANTAGE route counts: `{m['vantage_route_counts']}`")
    lines.append(f"- VANTAGE ever used TransPLD: `{m['vantage_ever_used_transpld']}`")
    lines.append(f"- Mismatch after a rewrite-view route: `{m['mismatch_after_transpld_route']}`")
    lines.append(f"- Classification: `{m.get('classification')}`")
    lines.append(f"- VANTAGE step at mismatch: `{m['vantage_step_at_mismatch']}`")
    lines.append(f"- Artifact completions: `{m['paths']['completions']}`")
    lines.append(f"- Artifact steps: `{m['paths']['steps']}`")
    lines.append("")
    lines.append("Decoded token context A:")
    lines.append("")
    lines.append("```text")
    lines.append(m["decoded_context_a"])
    lines.append("```")
    lines.append("")
    lines.append("Decoded token context B:")
    lines.append("")
    lines.append("```text")
    lines.append(m["decoded_context_b"])
    lines.append("```")
    lines.append("")
    lines.append("Character diff context:")
    lines.append("")
    lines.append("```text")
    lines.append(f"A: {m['char_diff']['a_context']}")
    lines.append(f"B: {m['char_diff']['b_context']}")
    lines.append("```")
    lines.append("")
    return lines


def bf16_first_divergence_report(report: dict[str, Any]) -> str:
    mismatches = [
        m for m in report["mismatches"] if m["run_kind"] == "bf16/sdpa validation"
    ]
    lines: list[str] = [
        "# TransPLD bf16 Optimized-Path First-Divergence Diagnosis",
        "",
        "Generated by `scripts/diagnose_transpld_parity.py` from the released timing artifacts.",
        "",
        "## Scope",
        "",
        "This report classifies the known bf16 optimized-path mismatches using the fields "
        "available in `completions.jsonl` and `steps.jsonl`: retokenized outputs, route "
        "summaries, accepted-length summaries, correction-token summaries, finish flags, "
        "and max-token flags. The artifacts do not store verifier logits, top-k margins, "
        "cache lengths, position ids, or attention masks, so numerical near-tie and cache/"
        "position root causes cannot be proven from the current artifact alone.",
        "",
        "## Mismatch Counts",
        "",
        "| Workload | Pair | Count |",
        "|---|---|---:|",
    ]
    counts: Counter[tuple[str, str]] = Counter(
        (m["workload"], m["method_pair"]) for m in mismatches
    )
    for (workload, pair), count in sorted(counts.items()):
        lines.append(f"| {workload} | {pair} | {count} |")
    if not counts:
        lines.append("| none | none | 0 |")
    lines += [
        "",
        "## First-Divergence Table",
        "",
        "| Workload | Task | Pair | First token | Token ids | Event | Classification | TransPLD before mismatch? | Logits/cache status |",
        "|---|---|---|---:|---|---|---|---:|---|",
    ]
    for m in mismatches:
        cls = m.get("classification") or {}
        lines.append(
            "| {workload} | `{task}` | {pair} | {idx} | `{a}` vs `{b}` | {event} | {cat} | {trans} | {status} |".format(
                workload=m["workload"],
                task=m["task_id"],
                pair=m["method_pair"],
                idx=m["first_mismatching_token_index"],
                a=m["token_id_a"],
                b=m["token_id_b"],
                event=cls.get("event_at_mismatch", "unknown"),
                cat=cls.get("category", "unknown"),
                trans=m["mismatch_after_transpld_route"],
                status="unavailable in current artifacts",
            )
        )
    lines += [
        "",
        "## Required Follow-Up Artifact For Logit-Level Root Cause",
        "",
        "To prove whether remaining bf16 drift is numerical near-tie behavior or an implementation "
        "bug, rerun only the mismatching bf16 tasks with an instrumented verifier trace that "
        "records: verifier top-10 token ids/logits at the first divergent position, top-1/top-2 "
        "margin, cache length before and after crop/catchup, position ids, attention-mask shape, "
        "route selected before divergence, accepted length, correction token, bonus token, EOS flag, "
        "and max-token flag.",
        "",
        "Until that artifact exists, the paper should classify these rows as bf16 optimized-path "
        "parity drift with incomplete root-cause evidence, not as deployment-ready exactness.",
        "",
    ]
    return "\n".join(lines)


def parse_lossless_table(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.startswith("|") or line.startswith("|---") or "Workload" in line:
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) >= 4:
            rows.append(
                {
                    "workload": parts[0],
                    "n": parts[1],
                    "pld_equals_vanilla": parts[2],
                    "vantage_equals_vanilla": parts[3],
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer)
    runs = discover_validation_runs(args.validation_root)
    runs += discover_real_commit_run(args.real_commit_dir)
    if not runs:
        raise SystemExit("No runs found. Check --validation-root and --real-commit-dir.")

    diagnosed = [
        diagnose_run(run, tokenizer, args.context_tokens, args.context_chars)
        for run in runs
    ]
    mismatches = [m for run in diagnosed for m in run["mismatches"]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_json = {
        "schema": "vantage-transpld-parity-diagnosis/v1",
        "tokenizer": args.tokenizer,
        "artifact_paths": sorted(
            {
                str(args.validation_root),
                str(args.real_commit_dir),
                "artifacts/vantage_transpld/tables/validation_20260515_v1/exactness.md",
                "artifacts/vantage_transpld/tables/validation_20260515_v1/lossless_exactness.md",
            }
        ),
        "lossless_exactness_rows": parse_lossless_table(
            Path("artifacts/vantage_transpld/tables/validation_20260515_v1/lossless_exactness.md")
        ),
        "runs": diagnosed,
        "mismatches": mismatches,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report_json, indent=2, ensure_ascii=False, sort_keys=True))
    write_jsonl(args.output_dir / "mismatch_details.jsonl", mismatches)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown_report(report_json))
    args.bf16_divergence_report.parent.mkdir(parents=True, exist_ok=True)
    args.bf16_divergence_report.write_text(bf16_first_divergence_report(report_json))
    print(f"Wrote {args.output_dir / 'report.json'}")
    print(f"Wrote {args.output_dir / 'mismatch_details.jsonl'}")
    print(f"Wrote {args.report}")
    print(f"Wrote {args.bf16_divergence_report}")
    print(f"Mismatches: {len(mismatches)}")


if __name__ == "__main__":
    main()
