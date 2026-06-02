"""Collect hidden states for the actual queued-MTP use distribution.

Unlike ``collect_pld_mtp_training_data.py``, labels here are the future tokens
at the *use* step.  The hidden state still comes from the queue-creation step,
matching the current runtime architecture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asts.code_proposers import encode_no_special  # noqa: E402
from scripts.collect_pld_mtp_training_data import (  # noqa: E402
    _bool_arg,
    _hidden_and_label_positions,
    _output_tokens,
    _pld_hit_or_miss,
    _steps_by_task,
)
from scripts.train_pld_candidate_reranker import _load_jsonl  # noqa: E402


def _prefix_hash(token_ids: list[int]) -> str:
    h = hashlib.blake2b(digest_size=16)
    for token_id in token_ids:
        h.update(int(token_id).to_bytes(8, byteorder="little", signed=True))
    return h.hexdigest()


def _weak(row: dict[str, Any], *, threshold: int, weak_field: str) -> bool:
    if weak_field == "draft_len":
        return int(row.get("proposal_tokens") or row.get("k") or 0) <= threshold
    if weak_field == "accepted_len":
        return int(row.get("n_accepted_drafts") or 0) <= threshold
    raise ValueError(f"unsupported weak_field: {weak_field!r}")


def _passes_filter(row: dict[str, Any], *, mode: str, threshold: int, weak_field: str) -> bool:
    if mode == "all":
        return True
    if mode == "weak":
        return _weak(row, threshold=threshold, weak_field=weak_field)
    raise ValueError(f"unsupported queue filter mode: {mode!r}")


def _eligible_pairs(
    rows: list[dict[str, Any]],
    *,
    threshold: int,
    weak_field: str,
    include_dropped: bool,
    create_filter: str = "all",
    use_filter: str = "all",
) -> tuple[list[tuple[dict[str, Any], dict[str, Any], bool]], dict[str, int]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any], bool]] = []
    counts = Counter()
    for i in range(len(rows) - 1):
        create = rows[i]
        use = rows[i + 1]
        if not _passes_filter(create, mode=create_filter, threshold=threshold, weak_field=weak_field):
            counts["dropped_create_filter"] += 1
            continue
        counts["queue_created_candidates"] += 1
        create_start = int(create.get("_generated_start") or 0)
        create_emitted = max(1, int(create.get("n_emitted") or 0))
        create_accepted = int(create.get("n_accepted_drafts") or 0)
        if create_emitted <= create_accepted:
            counts["no_baseline_next"] += 1
            continue
        expected_use_start = create_start + create_emitted
        actual_use_start = int(use.get("_generated_start") or -1)
        if expected_use_start != actual_use_start:
            counts["position_mismatch"] += 1
            continue
        dropped = not _passes_filter(use, mode=use_filter, threshold=threshold, weak_field=weak_field)
        if dropped:
            counts["dropped_use_filter"] += 1
            if use_filter == "weak":
                counts["dropped_pld_strong"] += 1
            if not include_dropped:
                continue
        else:
            counts["queue_used_examples"] += 1
        pairs.append((create, use, dropped))
    return pairs, dict(counts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    ap.add_argument("--steps", type=Path, required=True)
    ap.add_argument("--completions", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--method", default="blazedit_pld_w128_n10")
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--trigger-threshold", type=int, default=4)
    ap.add_argument("--weak-field", choices=["draft_len", "accepted_len"], default="draft_len")
    ap.add_argument("--include-dropped", type=_bool_arg, default=False)
    ap.add_argument("--queue-create-filter", choices=["all", "weak"], default="all")
    ap.add_argument("--queue-use-filter", choices=["all", "weak"], default="all")
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--output-projection", type=Path, default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.target,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()
    if args.output_projection is not None:
        emb = model.get_output_embeddings()
        if emb is None:
            raise SystemExit(f"{args.target} does not expose output embeddings")
        args.output_projection.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"output_weight": emb.weight.detach().cpu()}, args.output_projection)
        print(f"wrote frozen output projection to {args.output_projection}")

    completions = {str(r["task_id"]): r for r in _load_jsonl(args.completions)}
    steps_by_task = _steps_by_task(args.steps, method=args.method)
    task_ids = [task_id for task_id in sorted(steps_by_task) if task_id in completions]
    if args.max_tasks > 0:
        task_ids = task_ids[: args.max_tasks]

    all_pairs: dict[str, list[tuple[dict[str, Any], dict[str, Any], bool]]] = {}
    aggregate_pair_counts = Counter()
    create_acc_dist = Counter()
    use_acc_dist = Counter()
    create_miss = 0
    use_miss = 0
    for task_id in task_ids:
        pairs, counts = _eligible_pairs(
            steps_by_task[task_id],
            threshold=args.trigger_threshold,
            weak_field=args.weak_field,
            include_dropped=args.include_dropped,
            create_filter=args.queue_create_filter,
            use_filter=args.queue_use_filter,
        )
        if pairs:
            all_pairs[task_id] = pairs
        aggregate_pair_counts.update(counts)
        for create, use, _dropped in pairs:
            create_acc_dist[int(create.get("n_accepted_drafts") or 0)] += 1
            use_acc_dist[int(use.get("n_accepted_drafts") or 0)] += 1
            create_miss += int(_pld_hit_or_miss(create) == "miss")
            use_miss += int(_pld_hit_or_miss(use) == "miss")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)

    encoded_tasks: list[tuple[str, list[int], list[int], list[int]]] = []
    for task_id in task_ids:
        if task_id not in all_pairs:
            continue
        comp = completions[task_id]
        prompt_ids = encode_no_special(tokenizer, str(comp.get("prompt") or ""))
        output_ids = _output_tokens(tokenizer, comp, args.method)
        if prompt_ids and output_ids:
            encoded_tasks.append((task_id, prompt_ids, output_ids, prompt_ids + output_ids))

    hidden_rows: list[Any] = []
    labels_rows: list[Any] = []
    task_out: list[str] = []
    step_out: list[int] = []
    create_step_out: list[int] = []
    use_step_out: list[int] = []
    create_accepted_lens: list[int] = []
    use_accepted_lens: list[int] = []
    generated_starts: list[int] = []
    use_generated_starts: list[int] = []
    mtp_hidden_positions: list[int] = []
    prefix_hashes_after_create: list[str] = []
    emitted_prefix_lens_after_create: list[int] = []
    next_pld_draft_lens: list[int] = []
    next_pld_match_lens: list[int] = []
    next_use_weak_flags: list[int] = []
    create_weak_flags: list[int] = []
    invalidation_reasons: list[str] = []
    create_misses: list[int] = []
    use_misses: list[int] = []
    dropped_flags: list[int] = []

    batch_size = max(1, int(args.batch_size))
    stop = False
    for batch_start in range(0, len(encoded_tasks), batch_size):
        batch = encoded_tasks[batch_start : batch_start + batch_size]
        max_len = max(len(item[3]) for item in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=args.device)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long, device=args.device)
        for i, (_, _, _, ids) in enumerate(batch):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=args.device)
            attention_mask[i, : len(ids)] = 1
        with torch.no_grad():
            hidden_batch = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            ).hidden_states[-1]

        for item_idx, (task_id, prompt_ids, output_ids, ids) in enumerate(batch):
            hidden = hidden_batch[item_idx, : len(ids)]
            for create, use, dropped in all_pairs[task_id]:
                create_start = int(create.get("_generated_start") or 0)
                create_emitted = max(1, int(create.get("n_emitted") or 0))
                create_accepted = int(create.get("n_accepted_drafts") or 0)
                use_start = int(use.get("_generated_start") or 0)
                mtp_hidden_pos, _ = _hidden_and_label_positions(
                    generated_start=create_start,
                    accepted_len=create_accepted,
                    mtp_position="post_pld",
                )
                if use_start + args.num_heads > len(output_ids):
                    continue
                current_idx = len(prompt_ids) + mtp_hidden_pos
                if current_idx < 0 or current_idx >= hidden.shape[0]:
                    continue
                labels = output_ids[use_start : use_start + args.num_heads]
                if len(labels) != args.num_heads:
                    continue
                emitted_after_create = create_start + create_emitted
                prefix_after_create = prompt_ids + output_ids[:emitted_after_create]
                hidden_rows.append(hidden[current_idx].detach().float().cpu())
                labels_rows.append(torch.tensor(labels, dtype=torch.long))
                task_out.append(task_id)
                step_id = int(create.get("step") or 0)
                step_out.append(step_id)
                create_step_out.append(step_id)
                use_step_out.append(int(use.get("step") or 0))
                create_accepted_lens.append(create_accepted)
                use_accepted_lens.append(int(use.get("n_accepted_drafts") or 0))
                generated_starts.append(create_start)
                use_generated_starts.append(use_start)
                mtp_hidden_positions.append(mtp_hidden_pos)
                prefix_hashes_after_create.append(_prefix_hash(prefix_after_create))
                emitted_prefix_lens_after_create.append(int(emitted_after_create))
                next_pld_draft_lens.append(int(use.get("proposal_tokens") or use.get("k") or 0))
                next_pld_match_lens.append(int(use.get("match_len") or use.get("matched_ngram_size") or -1))
                next_use_weak_flags.append(
                    1 if _weak(use, threshold=args.trigger_threshold, weak_field=args.weak_field) else 0
                )
                create_weak_flags.append(
                    1 if _weak(create, threshold=args.trigger_threshold, weak_field=args.weak_field) else 0
                )
                invalidation_reasons.append("valid")
                create_misses.append(1 if _pld_hit_or_miss(create) == "miss" else 0)
                use_misses.append(1 if _pld_hit_or_miss(use) == "miss" else 0)
                dropped_flags.append(1 if dropped else 0)
                if args.max_examples > 0 and len(hidden_rows) >= args.max_examples:
                    stop = True
                    break
            if stop:
                break
        if stop:
            break

    if not hidden_rows:
        raise SystemExit("no queued-use MTP examples collected")

    labels_tensor = torch.stack(labels_rows)
    create_acc_tensor = torch.tensor(create_accepted_lens, dtype=torch.long)
    use_acc_tensor = torch.tensor(use_accepted_lens, dtype=torch.long)
    create_miss_tensor = torch.tensor(create_misses, dtype=torch.bool)
    use_miss_tensor = torch.tensor(use_misses, dtype=torch.bool)
    dropped_tensor = torch.tensor(dropped_flags, dtype=torch.bool)
    payload = {
        "hidden": torch.stack(hidden_rows),
        "hidden_state_at_queue_creation": torch.stack(hidden_rows),
        "labels": labels_tensor,
        "future_tokens_at_use_step": labels_tensor,
        "target_tokens_step_t_plus_1_h1_to_h4": labels_tensor,
        "accepted_len": create_acc_tensor,
        "create_accepted_len": create_acc_tensor,
        "previous_accepted_len_t": create_acc_tensor,
        "previous_token0_reject_t": create_acc_tensor == 0,
        "use_step_accepted_len": use_acc_tensor,
        "generated_start": torch.tensor(generated_starts, dtype=torch.long),
        "use_generated_start": torch.tensor(use_generated_starts, dtype=torch.long),
        "mtp_hidden_pos": torch.tensor(mtp_hidden_positions, dtype=torch.long),
        "prefix_hash_after_step_t": prefix_hashes_after_create,
        "emitted_prefix_len_after_step_t": torch.tensor(emitted_prefix_lens_after_create, dtype=torch.long),
        "pld_draft_len_t_plus_1": torch.tensor(next_pld_draft_lens, dtype=torch.long),
        "pld_match_len_t_plus_1": torch.tensor(next_pld_match_lens, dtype=torch.long),
        "use_step_is_weak_t_plus_1": torch.tensor(next_use_weak_flags, dtype=torch.bool),
        "create_step_is_weak_t": torch.tensor(create_weak_flags, dtype=torch.bool),
        "eos_or_finished": torch.zeros(len(hidden_rows), dtype=torch.bool),
        "valid_queued_example": torch.ones(len(hidden_rows), dtype=torch.bool),
        "invalidation_reason": invalidation_reasons,
        "pld_miss": create_miss_tensor,
        "is_pld_miss": create_miss_tensor,
        "create_is_pld_miss": create_miss_tensor,
        "use_is_pld_miss": use_miss_tensor,
        "would_drop_pld_strong": dropped_tensor,
        "task_id": task_out,
        "step_id": step_out,
        "create_step_id": create_step_out,
        "use_step_id": use_step_out,
        "trigger_accepted_len_eq_0": create_acc_tensor == 0,
        "trigger_accepted_len_le_1": create_acc_tensor <= 1,
        "trigger_accepted_len_le_2": create_acc_tensor <= 2,
        "trigger_accepted_len_le_4": create_acc_tensor <= 4,
        "trigger_pld_miss": create_miss_tensor,
        "metadata": {
            "target": args.target,
            "method": args.method,
            "num_heads": args.num_heads,
            "trigger_threshold": args.trigger_threshold,
            "weak_field": args.weak_field,
            "include_dropped": args.include_dropped,
            "queue_create_filter": args.queue_create_filter,
            "queue_use_filter": args.queue_use_filter,
            "label_mode": "queued_use",
            "hidden_source": "post_pld_queue_creation",
            "hidden_available_after_step_t_verifier": True,
            "labels_aligned_to": "step_t_plus_1",
            "current_step_t_plus_1_verifier_fields_used_as_input": False,
            "n_examples": len(hidden_rows),
            "hidden_size": int(hidden_rows[0].numel()),
            "queue_pair_counts": dict(aggregate_pair_counts),
            "accepted_len_distribution_at_create": dict(sorted(create_acc_dist.items())),
            "accepted_len_distribution_at_use": dict(sorted(use_acc_dist.items())),
            "create_pld_miss_count": int(create_miss),
            "use_pld_miss_count": int(use_miss),
        },
    }
    for i in range(args.num_heads):
        payload[f"label_use_t_plus_{i + 1}"] = labels_tensor[:, i]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    summary = payload["metadata"].copy()
    summary["hidden_shape"] = list(payload["hidden"].shape)
    summary["labels_shape"] = list(payload["labels"].shape)
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"wrote {len(hidden_rows)} queued-use examples to {args.output} "
        f"hidden={tuple(payload['hidden'].shape)} labels={tuple(payload['labels'].shape)}"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
