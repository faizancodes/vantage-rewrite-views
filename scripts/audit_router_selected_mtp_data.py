"""Audit router-selected queued-MTP training rows against PLD traces.

The router-selected dataset is easy to get subtly wrong because the hidden
state is captured at queue creation (step N), while labels are consumed at the
next routed use step (step N+1).  This script checks the tensor rows against
the original steps/completions and writes a small decoded sample report.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asts.code_proposers import encode_no_special  # noqa: E402
from scripts.collect_pld_mtp_training_data import _output_tokens  # noqa: E402
from scripts.train_pld_candidate_reranker import _load_jsonl  # noqa: E402
from scripts.train_weak_pld_router import _safe_int, load_method_rows  # noqa: E402


def _load_forbidden_task_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {str(row.get("task_id") or "") for row in _load_jsonl(path)}


def _decode_window(tokenizer, ids: list[int], center: int, *, before: int = 24, after: int = 12) -> str:
    lo = max(0, int(center) - before)
    hi = min(len(ids), int(center) + after)
    if lo >= hi:
        return ""
    return tokenizer.decode(ids[lo:hi])


def _tensor_list(data: dict[str, Any], key: str, n: int, default: Any = None) -> list[Any]:
    value = data.get(key)
    if value is None:
        return [default for _ in range(n)]
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--steps", type=Path, required=True)
    ap.add_argument("--completions", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("/tmp/pld_mtp/router_selected_audit"))
    ap.add_argument("--method", default="blazedit_pld_w128_n10")
    ap.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--sample-seed", type=int, default=17)
    ap.add_argument(
        "--forbid-task-ids-jsonl",
        type=Path,
        default=None,
        help="Optional held-out completions/steps jsonl. Any overlapping task_id is reported as leakage.",
    )
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    data = torch.load(args.data, map_location="cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    completions = {str(row.get("task_id") or ""): row for row in _load_jsonl(args.completions)}
    rows_by_task = load_method_rows(args.steps, method=args.method)
    forbidden_task_ids = _load_forbidden_task_ids(args.forbid_task_ids_jsonl)

    hidden = data.get("hidden")
    labels = data["labels"].long()
    n = int(labels.shape[0])
    task_ids = [str(x) for x in data["task_id"]]
    create_step_ids = [int(x) for x in _tensor_list(data, "create_step_id", n, 0)]
    use_step_ids = [int(x) for x in _tensor_list(data, "use_step_id", n, 0)]
    generated_starts = [int(x) for x in _tensor_list(data, "generated_start", n, -1)]
    use_generated_starts = [int(x) for x in _tensor_list(data, "use_generated_start", n, -1)]
    mtp_hidden_positions = [int(x) for x in _tensor_list(data, "mtp_hidden_pos", n, -999999)]
    create_accepted_lens = [int(x) for x in _tensor_list(data, "create_step_accepted_len", n, 0)]
    use_accepted_lens = [int(x) for x in _tensor_list(data, "use_step_accepted_len", n, 0)]
    router_probs = [float(x) for x in _tensor_list(data, "router_probability", n, 0.0)]

    encoded_cache: dict[str, tuple[list[int], list[int]]] = {}

    counts = {
        "total_examples": n,
        "checked_examples": 0,
        "aligned_examples": 0,
        "label_mismatch_count": 0,
        "task_mismatch_count": 0,
        "step_mismatch_count": 0,
        "position_mismatch_count": 0,
        "hidden_position_mismatch_count": 0,
        "missing_future_tokens": 0,
        "heldout_task_overlap_count": 0,
    }
    mismatch_examples: list[dict[str, Any]] = []

    sample_indices = list(range(n))
    random.Random(args.sample_seed).shuffle(sample_indices)
    sample_indices = sample_indices[: max(0, int(args.num_samples))]
    decoded_samples: list[dict[str, Any]] = []

    for idx in range(n):
        task_id = task_ids[idx]
        row_errors: list[str] = []
        counts["checked_examples"] += 1
        if task_id in forbidden_task_ids:
            counts["heldout_task_overlap_count"] += 1
            row_errors.append("heldout_task_overlap")
        if task_id not in rows_by_task or task_id not in completions:
            counts["task_mismatch_count"] += 1
            row_errors.append("missing_task")
        else:
            rows = rows_by_task[task_id]
            by_step = {_safe_int(row.get("step"), 0): row for row in rows}
            create = by_step.get(create_step_ids[idx])
            use = by_step.get(use_step_ids[idx])
            if create is None or use is None:
                counts["step_mismatch_count"] += 1
                row_errors.append("missing_step")
            else:
                create_pos = rows.index(create)
                use_pos = rows.index(use)
                if use_pos != create_pos + 1:
                    counts["step_mismatch_count"] += 1
                    row_errors.append("use_step_not_next_decode_step")
                create_start = _safe_int(create.get("_generated_start"), 0)
                use_start = _safe_int(use.get("_generated_start"), 0)
                create_emitted = max(1, _safe_int(create.get("n_emitted"), 1))
                expected_use_start = create_start + create_emitted
                if generated_starts[idx] != create_start or use_generated_starts[idx] != use_start:
                    counts["position_mismatch_count"] += 1
                    row_errors.append("stored_start_mismatch")
                if expected_use_start != use_start:
                    counts["position_mismatch_count"] += 1
                    row_errors.append("use_start_not_after_create_emitted")
                expected_hidden = create_start - 1 + max(0, create_accepted_lens[idx])
                if mtp_hidden_positions[idx] != expected_hidden:
                    counts["hidden_position_mismatch_count"] += 1
                    row_errors.append("mtp_hidden_pos_mismatch")

                if task_id not in encoded_cache:
                    comp = completions[task_id]
                    prompt_ids = encode_no_special(tokenizer, str(comp.get("prompt") or ""))
                    output_ids = _output_tokens(tokenizer, comp, args.method)
                    encoded_cache[task_id] = (prompt_ids, output_ids)
                _, output_ids = encoded_cache[task_id]
                if use_start + labels.shape[1] > len(output_ids):
                    counts["missing_future_tokens"] += 1
                    row_errors.append("missing_future_tokens")
                    expected_labels: list[int] = []
                else:
                    expected_labels = output_ids[use_start : use_start + labels.shape[1]]
                    observed_labels = [int(x) for x in labels[idx].tolist()]
                    if observed_labels != expected_labels:
                        counts["label_mismatch_count"] += 1
                        row_errors.append("labels_do_not_match_use_future")

                if idx in sample_indices:
                    decoded_samples.append(
                        {
                            "index": idx,
                            "task_id": task_id,
                            "create_step_id": create_step_ids[idx],
                            "use_step_id": use_step_ids[idx],
                            "router_probability": router_probs[idx],
                            "create_accepted_len": create_accepted_lens[idx],
                            "use_accepted_len": use_accepted_lens[idx],
                            "create_context": _decode_window(
                                tokenizer, output_ids, mtp_hidden_positions[idx]
                            )
                            if "missing_task" not in row_errors
                            else "",
                            "use_context": _decode_window(tokenizer, output_ids, use_generated_starts[idx])
                            if "missing_task" not in row_errors
                            else "",
                            "labels_decoded": tokenizer.decode([int(x) for x in labels[idx].tolist()]),
                            "expected_future_decoded": tokenizer.decode(expected_labels)
                            if expected_labels
                            else "",
                            "errors": row_errors,
                        }
                    )

        if row_errors:
            if len(mismatch_examples) < 50:
                mismatch_examples.append(
                    {
                        "index": idx,
                        "task_id": task_id,
                        "create_step_id": create_step_ids[idx],
                        "use_step_id": use_step_ids[idx],
                        "errors": row_errors,
                    }
                )
        else:
            counts["aligned_examples"] += 1

    pass_rate = counts["aligned_examples"] / max(1, counts["checked_examples"])
    payload = {
        "data": str(args.data),
        "steps": str(args.steps),
        "completions": str(args.completions),
        "method": args.method,
        "target": args.target,
        "hidden_shape": list(hidden.shape) if hidden is not None and hasattr(hidden, "shape") else None,
        "labels_shape": list(labels.shape),
        "counts": counts,
        "alignment_pass_rate": pass_rate,
        "mismatch_examples": mismatch_examples,
        "decoded_samples": decoded_samples,
        "metadata": data.get("metadata") or {},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Router-Selected MTP Data Audit",
        "",
        f"data: `{args.data}`",
        f"steps: `{args.steps}`",
        f"examples checked: `{counts['checked_examples']}`",
        f"alignment pass rate: `{100.0 * pass_rate:.2f}%`",
        "",
        "| check | count |",
        "|---|---:|",
    ]
    for key, value in counts.items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Decoded Samples", ""])
    for sample in decoded_samples:
        lines.extend(
            [
                f"### Example {sample['index']}",
                "",
                f"- task: `{sample['task_id']}`",
                f"- create/use step: `{sample['create_step_id']}` -> `{sample['use_step_id']}`",
                f"- router probability: `{sample['router_probability']:.4f}`",
                f"- accepted len create/use: `{sample['create_accepted_len']}` / `{sample['use_accepted_len']}`",
                f"- labels: `{sample['labels_decoded']}`",
                f"- expected: `{sample['expected_future_decoded']}`",
                f"- errors: `{sample['errors']}`",
                "",
                "create context:",
                "```text",
                sample["create_context"],
                "```",
                "use context:",
                "```text",
                sample["use_context"],
                "```",
                "",
            ]
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print((args.output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
