"""Collect MTP examples for learned-router selected queued use cases.

This collector matches the proposed weak-router queued-MTP architecture:

* queue creation uses post-PLD information from step N (`accepted_len <= 4`);
* queue use at step N+1 is selected by a learned pre-verification router;
* hidden state comes from the post-PLD queue creation point;
* labels are the verified future tokens at the use step.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asts.code_proposers import encode_no_special  # noqa: E402
from scripts.collect_pld_mtp_training_data import (  # noqa: E402
    _hidden_and_label_positions,
    _output_tokens,
    _pld_hit_or_miss,
)
from scripts.train_pld_candidate_reranker import _load_jsonl  # noqa: E402
from scripts.train_weak_pld_router import (  # noqa: E402
    _empty_history,
    _safe_int,
    _update_history,
    extract_feature_dict,
    load_method_rows,
)


def _bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() not in {"0", "false", "no", "off"}


def _load_router(path: Path):
    with path.open("rb") as f:
        payload = pickle.load(f)
    model = payload.get("model")
    if model is None:
        raise SystemExit(f"{path} does not contain a trained router model")
    return model, payload


def _predict_router_probs(
    rows_by_task: dict[str, list[dict[str, Any]]],
    router,
    *,
    accepted_len_threshold: int,
) -> dict[tuple[str, int], float]:
    probs: dict[tuple[str, int], float] = {}
    for task_id, rows in rows_by_task.items():
        history = _empty_history()
        features: list[dict[str, Any]] = []
        keys: list[tuple[str, int]] = []
        for step_index, row in enumerate(rows):
            features.append(
                extract_feature_dict(
                    row,
                    generated_start=_safe_int(row.get("_generated_start"), 0),
                    history=history,
                    step_index=step_index,
                )
            )
            keys.append((task_id, _safe_int(row.get("step"), 0)))
            _update_history(history, row, threshold=accepted_len_threshold)
        if not features:
            continue
        pred = router.predict_proba(features)[:, 1]
        probs.update({key: float(prob) for key, prob in zip(keys, pred, strict=True)})
    return probs


def _eligible_pairs(
    rows_by_task: dict[str, list[dict[str, Any]]],
    router_probs: dict[tuple[str, int], float],
    *,
    router_threshold: float,
    collection_threshold: float,
    accepted_len_threshold: int,
) -> tuple[dict[str, list[tuple[dict[str, Any], dict[str, Any], float, bool, bool]]], dict[str, Any]]:
    pairs_by_task: dict[str, list[tuple[dict[str, Any], dict[str, Any], float, bool, bool]]] = {}
    counts = Counter()
    create_acc = Counter()
    use_acc = Counter()
    prob_hist = Counter()
    for task_id, rows in rows_by_task.items():
        pairs: list[tuple[dict[str, Any], dict[str, Any], float, bool, bool]] = []
        for idx in range(len(rows) - 1):
            create = rows[idx]
            use = rows[idx + 1]
            create_accepted = _safe_int(create.get("n_accepted_drafts"), 0)
            if create_accepted > accepted_len_threshold:
                continue
            counts["queue_created_candidates"] += 1
            create_start = _safe_int(create.get("_generated_start"), 0)
            create_emitted = max(1, _safe_int(create.get("n_emitted"), 1))
            if create_emitted <= create_accepted:
                counts["no_baseline_next"] += 1
                continue
            expected_use_start = create_start + create_emitted
            if expected_use_start != _safe_int(use.get("_generated_start"), -1):
                counts["position_mismatch"] += 1
                continue
            use_step_id = _safe_int(use.get("step"), 0)
            prob = router_probs.get((task_id, use_step_id), 0.0)
            router_selected_default = prob >= router_threshold
            router_selected_for_collection = prob >= collection_threshold
            if not router_selected_for_collection:
                counts["not_router_selected_for_collection"] += 1
                continue
            use_accepted = _safe_int(use.get("n_accepted_drafts"), 0)
            is_true_positive = use_accepted <= accepted_len_threshold
            counts["router_selected_examples"] += 1
            counts["router_true_positive"] += int(is_true_positive)
            counts["router_false_positive"] += int(not is_true_positive)
            counts["router_selected_at_default_threshold"] += int(router_selected_default)
            create_acc[create_accepted] += 1
            use_acc[use_accepted] += 1
            prob_hist[f"{min(9, int(prob * 10)) / 10:.1f}"] += 1
            pairs.append((create, use, prob, router_selected_default, is_true_positive))
        if pairs:
            pairs_by_task[task_id] = pairs
    summary = {
        "counts": dict(counts),
        "accepted_len_distribution_at_create": dict(sorted(create_acc.items())),
        "accepted_len_distribution_at_use": dict(sorted(use_acc.items())),
        "router_probability_bucket_counts": dict(sorted(prob_hist.items())),
    }
    return pairs_by_task, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    ap.add_argument("--steps", type=Path, required=True)
    ap.add_argument("--completions", type=Path, required=True)
    ap.add_argument("--router", type=Path, required=True)
    ap.add_argument("--router-threshold", type=float, default=0.5)
    ap.add_argument(
        "--collection-threshold",
        type=float,
        default=None,
        help="Collect examples with router prob above this value. Defaults to router-threshold.",
    )
    ap.add_argument("--accepted-len-threshold", type=int, default=4)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--method", default="blazedit_pld_w128_n10")
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--output-projection", type=Path, default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    collection_threshold = (
        float(args.router_threshold)
        if args.collection_threshold is None
        else float(args.collection_threshold)
    )
    router, router_payload = _load_router(args.router)
    completions = {str(row["task_id"]): row for row in _load_jsonl(args.completions)}
    rows_by_task = load_method_rows(args.steps, method=args.method)
    task_ids = [task_id for task_id in sorted(rows_by_task) if task_id in completions]
    if args.max_tasks > 0:
        task_ids = task_ids[: args.max_tasks]
        rows_by_task = {task_id: rows_by_task[task_id] for task_id in task_ids}
    else:
        rows_by_task = {task_id: rows_by_task[task_id] for task_id in task_ids}
    router_probs = _predict_router_probs(
        rows_by_task,
        router,
        accepted_len_threshold=args.accepted_len_threshold,
    )
    pairs_by_task, pair_summary = _eligible_pairs(
        rows_by_task,
        router_probs,
        router_threshold=args.router_threshold,
        collection_threshold=collection_threshold,
        accepted_len_threshold=args.accepted_len_threshold,
    )
    if not pairs_by_task:
        raise SystemExit("no router-selected queued-use examples found")

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

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    encoded_tasks: list[tuple[str, list[int], list[int], list[int]]] = []
    for task_id in sorted(pairs_by_task):
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
    create_acc: list[int] = []
    use_acc: list[int] = []
    generated_starts: list[int] = []
    use_generated_starts: list[int] = []
    mtp_hidden_positions: list[int] = []
    create_miss: list[int] = []
    use_miss: list[int] = []
    router_prob_rows: list[float] = []
    router_default_selected: list[int] = []
    router_tp: list[int] = []

    stop = False
    batch_size = max(1, int(args.batch_size))
    for batch_start in range(0, len(encoded_tasks), batch_size):
        batch = encoded_tasks[batch_start : batch_start + batch_size]
        max_len = max(len(item[3]) for item in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=args.device)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long, device=args.device)
        for idx, (_, _, _, ids) in enumerate(batch):
            input_ids[idx, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=args.device)
            attention_mask[idx, : len(ids)] = 1
        with torch.no_grad():
            hidden_batch = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            ).hidden_states[-1]

        for item_idx, (task_id, prompt_ids, output_ids, ids) in enumerate(batch):
            hidden = hidden_batch[item_idx, : len(ids)]
            for create, use, prob, selected_default, is_tp in pairs_by_task[task_id]:
                create_start = _safe_int(create.get("_generated_start"), 0)
                create_accepted = _safe_int(create.get("n_accepted_drafts"), 0)
                use_start = _safe_int(use.get("_generated_start"), 0)
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
                hidden_rows.append(hidden[current_idx].detach().float().cpu())
                labels_rows.append(torch.tensor(labels, dtype=torch.long))
                task_out.append(task_id)
                create_step = _safe_int(create.get("step"), 0)
                use_step = _safe_int(use.get("step"), 0)
                step_out.append(create_step)
                create_step_out.append(create_step)
                use_step_out.append(use_step)
                create_acc.append(create_accepted)
                use_acc.append(_safe_int(use.get("n_accepted_drafts"), 0))
                generated_starts.append(create_start)
                use_generated_starts.append(use_start)
                mtp_hidden_positions.append(mtp_hidden_pos)
                create_miss.append(1 if _pld_hit_or_miss(create) == "miss" else 0)
                use_miss.append(1 if _pld_hit_or_miss(use) == "miss" else 0)
                router_prob_rows.append(float(prob))
                router_default_selected.append(1 if selected_default else 0)
                router_tp.append(1 if is_tp else 0)
                if args.max_examples > 0 and len(hidden_rows) >= args.max_examples:
                    stop = True
                    break
            if stop:
                break
        if stop:
            break

    if not hidden_rows:
        raise SystemExit("no valid router-selected hidden-state examples collected")

    labels_tensor = torch.stack(labels_rows)
    create_acc_tensor = torch.tensor(create_acc, dtype=torch.long)
    use_acc_tensor = torch.tensor(use_acc, dtype=torch.long)
    create_miss_tensor = torch.tensor(create_miss, dtype=torch.bool)
    use_miss_tensor = torch.tensor(use_miss, dtype=torch.bool)
    router_prob_tensor = torch.tensor(router_prob_rows, dtype=torch.float32)
    router_tp_tensor = torch.tensor(router_tp, dtype=torch.bool)
    router_default_tensor = torch.tensor(router_default_selected, dtype=torch.bool)
    payload = {
        "hidden": torch.stack(hidden_rows),
        "hidden_state_at_queue_creation": torch.stack(hidden_rows),
        "labels": labels_tensor,
        "future_tokens_at_use_step": labels_tensor,
        "accepted_len": create_acc_tensor,
        "create_step_accepted_len": create_acc_tensor,
        "use_step_accepted_len": use_acc_tensor,
        "generated_start": torch.tensor(generated_starts, dtype=torch.long),
        "use_generated_start": torch.tensor(use_generated_starts, dtype=torch.long),
        "mtp_hidden_pos": torch.tensor(mtp_hidden_positions, dtype=torch.long),
        "pld_miss": create_miss_tensor,
        "is_pld_miss": create_miss_tensor,
        "create_step_is_pld_miss": create_miss_tensor,
        "use_step_is_pld_miss": use_miss_tensor,
        "router_probability": router_prob_tensor,
        "router_predicted_weak": router_default_tensor,
        "router_true_positive": router_tp_tensor,
        "router_false_positive": ~router_tp_tensor,
        "task_id": task_out,
        "step_id": step_out,
        "create_step_id": create_step_out,
        "use_step_id": use_step_out,
        "trigger_bucket": ["router_selected_queued_use" for _ in task_out],
        "trigger_accepted_len_eq_0": create_acc_tensor == 0,
        "trigger_accepted_len_le_1": create_acc_tensor <= 1,
        "trigger_accepted_len_le_2": create_acc_tensor <= 2,
        "trigger_accepted_len_le_4": create_acc_tensor <= 4,
        "trigger_pld_miss": create_miss_tensor,
        "metadata": {
            "target": args.target,
            "method": args.method,
            "num_heads": args.num_heads,
            "router": str(args.router),
            "router_name": router_payload.get("router_name"),
            "router_threshold": args.router_threshold,
            "collection_threshold": collection_threshold,
            "accepted_len_threshold": args.accepted_len_threshold,
            "label_mode": "router_selected_queued_use",
            "hidden_source": "post_pld_queue_creation",
            "n_examples": len(hidden_rows),
            "hidden_size": int(hidden_rows[0].numel()),
            "pair_summary": pair_summary,
            "collected_true_positive_count": int(router_tp_tensor.sum().item()),
            "collected_false_positive_count": int((~router_tp_tensor).sum().item()),
            "router_probability_min": float(router_prob_tensor.min().item()),
            "router_probability_mean": float(router_prob_tensor.mean().item()),
            "router_probability_max": float(router_prob_tensor.max().item()),
            "hidden_shape": list(torch.stack(hidden_rows).shape),
            "labels_shape": list(labels_tensor.shape),
        },
    }
    for i in range(args.num_heads):
        payload[f"label_t_plus_{i + 1}"] = labels_tensor[:, i]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(payload["metadata"], indent=2) + "\n")
    print(
        f"wrote {len(hidden_rows)} router-selected examples to {args.output} "
        f"hidden={tuple(payload['hidden'].shape)} labels={tuple(payload['labels'].shape)}"
    )
    print(json.dumps(payload["metadata"], indent=2))


if __name__ == "__main__":
    main()
