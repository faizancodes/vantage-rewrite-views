"""Collect frozen-backbone hidden states for offline PLD+MTP diagnostics.

This is an offline data collector, not a runtime decoder.  It replays existing
PLD completions, runs the target model over ``prompt + verified PLD output``,
and stores hidden states at weak PLD decode positions with labels for the next
K target tokens.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asts.code_proposers import encode_no_special  # noqa: E402
from scripts.train_pld_candidate_reranker import _load_jsonl  # noqa: E402


def _output_tokens(tokenizer, completion: dict[str, Any], method: str) -> list[int]:
    row = (completion.get("outputs") or {}).get(method) or {}
    for key in ("tokens", "token_ids", "completion_token_ids", "output_token_ids"):
        val = row.get(key)
        if isinstance(val, list) and all(isinstance(x, int) for x in val):
            return list(val)
    text = str(row.get("raw_text") if row.get("raw_text") is not None else row.get("text") or "")
    return encode_no_special(tokenizer, text)


def _steps_by_task(path: Path, *, method: str) -> dict[str, list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _load_jsonl(path):
        if row.get("method") == method:
            by_task[str(row.get("task_id") or "")].append(row)
    for task_id, rows in by_task.items():
        rows.sort(key=lambda r: int(r.get("step") or 0))
        pos = 0
        for row in rows:
            row["_generated_start"] = pos
            pos += max(1, int(row.get("n_emitted") or 0))
    return by_task


def _is_eligible(row: dict[str, Any], *, weak_only: bool, threshold: int) -> bool:
    if not weak_only:
        return True
    return int(row.get("n_accepted_drafts") or 0) <= threshold


def _bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() not in {"0", "false", "no", "off"}


def _pld_hit_or_miss(row: dict[str, Any]) -> str:
    if row.get("pld_exact_hit") is True or row.get("proposal_kind") == "blazedit_pld":
        return "hit"
    return "miss"


def _hidden_and_label_positions(
    *,
    generated_start: int,
    accepted_len: int,
    mtp_position: str,
) -> tuple[int, int]:
    """Return output-relative hidden position and first label position.

    ``generated_start`` is the output-token index of the first token emitted by
    the baseline PLD step.  Hidden position ``p`` predicts output token
    ``p + 1``.  Therefore pre-PLD collection uses ``generated_start - 1``.
    Post-PLD collection advances by the accepted PLD draft prefix before
    asking the MTP head to predict the next token.
    """

    step_start_pos = generated_start - 1
    if mtp_position == "pre_pld":
        mtp_hidden_pos = step_start_pos
    elif mtp_position == "post_pld":
        mtp_hidden_pos = step_start_pos + max(0, accepted_len)
    else:
        raise ValueError(f"unsupported mtp_position: {mtp_position!r}")
    return mtp_hidden_pos, mtp_hidden_pos + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    ap.add_argument("--steps", type=Path, required=True)
    ap.add_argument("--completions", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--method", default="blazedit_pld_w128_n10")
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--weak-only", type=_bool_arg, default=True)
    ap.add_argument("--include-accepted-len-threshold", type=int, default=4)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--output-projection", type=Path, default=None)
    ap.add_argument("--mtp-position", choices=["pre_pld", "post_pld"], default="pre_pld")
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
    task_ids = [t for t in sorted(steps_by_task) if t in completions]
    if args.max_tasks > 0:
        task_ids = task_ids[: args.max_tasks]

    hidden_rows: list[Any] = []
    labels_rows: list[Any] = []
    accepted_lens: list[int] = []
    n_emitted: list[int] = []
    generated_starts: list[int] = []
    step_start_positions: list[int] = []
    mtp_hidden_positions: list[int] = []
    pld_misses: list[int] = []
    exact_hit_or_miss: list[str] = []
    task_out: list[str] = []
    step_out: list[int] = []

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)

    encoded_tasks: list[tuple[str, list[int], list[int], list[int]]] = []
    for task_id in task_ids:
        comp = completions[task_id]
        prompt_ids = encode_no_special(tokenizer, str(comp.get("prompt") or ""))
        output_ids = _output_tokens(tokenizer, comp, args.method)
        if prompt_ids and output_ids:
            encoded_tasks.append((task_id, prompt_ids, output_ids, prompt_ids + output_ids))

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
            for row in steps_by_task[task_id]:
                if not _is_eligible(
                    row,
                    weak_only=args.weak_only,
                    threshold=args.include_accepted_len_threshold,
                ):
                    continue
                start = int(row.get("_generated_start") or 0)
                accepted = int(row.get("n_accepted_drafts") or 0)
                if start >= len(output_ids):
                    continue
                mtp_hidden_pos, label_start = _hidden_and_label_positions(
                    generated_start=start,
                    accepted_len=accepted,
                    mtp_position=args.mtp_position,
                )
                if args.mtp_position == "post_pld" and mtp_hidden_pos != start - 1 + accepted:
                    raise AssertionError("post-PLD hidden position accounting failed")
                if label_start != mtp_hidden_pos + 1:
                    raise AssertionError("labels must start after the MTP hidden position")
                if label_start < 0 or label_start + args.num_heads > len(output_ids):
                    continue
                current_idx = len(prompt_ids) + mtp_hidden_pos
                if current_idx < 0 or current_idx >= hidden.shape[0]:
                    continue
                labels = output_ids[label_start : label_start + args.num_heads]
                if len(labels) != args.num_heads:
                    continue
                hidden_rows.append(hidden[current_idx].detach().float().cpu())
                labels_rows.append(torch.tensor(labels, dtype=torch.long))
                emitted = max(1, int(row.get("n_emitted") or 0))
                hit_or_miss = _pld_hit_or_miss(row)
                accepted_lens.append(accepted)
                n_emitted.append(emitted)
                generated_starts.append(start)
                step_start_positions.append(start - 1)
                mtp_hidden_positions.append(mtp_hidden_pos)
                pld_misses.append(1 if hit_or_miss == "miss" else 0)
                exact_hit_or_miss.append(hit_or_miss)
                task_out.append(task_id)
                step_out.append(int(row.get("step") or 0))
                if args.max_examples > 0 and len(hidden_rows) >= args.max_examples:
                    stop = True
                    break
            if stop:
                break
        if stop:
            break

    if not hidden_rows:
        raise SystemExit("no eligible MTP examples collected")

    accepted_tensor = torch.tensor(accepted_lens, dtype=torch.long)
    labels_tensor = torch.stack(labels_rows)
    pld_miss_tensor = torch.tensor(pld_misses, dtype=torch.bool)
    payload = {
        "hidden": torch.stack(hidden_rows),
        "hidden_state_at_current_position": torch.stack(hidden_rows),
        "labels": labels_tensor,
        "future_tokens_1_to_4": labels_tensor,
        "accepted_len": accepted_tensor,
        "n_emitted": torch.tensor(n_emitted, dtype=torch.long),
        "generated_start": torch.tensor(generated_starts, dtype=torch.long),
        "step_start_pos": torch.tensor(step_start_positions, dtype=torch.long),
        "mtp_hidden_pos": torch.tensor(mtp_hidden_positions, dtype=torch.long),
        "pld_miss": pld_miss_tensor,
        "is_pld_miss": pld_miss_tensor,
        "exact_pld_hit_or_miss": exact_hit_or_miss,
        "trigger_accepted_len_eq_0": accepted_tensor == 0,
        "trigger_accepted_len_le_1": accepted_tensor <= 1,
        "trigger_accepted_len_le_2": accepted_tensor <= 2,
        "trigger_accepted_len_le_4": accepted_tensor <= 4,
        "trigger_pld_miss": pld_miss_tensor,
        "task_id": task_out,
        "step_id": step_out,
        "metadata": {
            "target": args.target,
            "method": args.method,
            "num_heads": args.num_heads,
            "weak_only": args.weak_only,
            "include_accepted_len_threshold": args.include_accepted_len_threshold,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "mtp_position": args.mtp_position,
            "n_examples": len(hidden_rows),
            "hidden_size": int(hidden_rows[0].numel()),
            "trigger_counts": {
                "accepted_len_eq_0": int((accepted_tensor == 0).sum().item()),
                "accepted_len_le_1": int((accepted_tensor <= 1).sum().item()),
                "accepted_len_le_2": int((accepted_tensor <= 2).sum().item()),
                "accepted_len_le_4": int((accepted_tensor <= 4).sum().item()),
                "pld_miss": int(pld_miss_tensor.sum().item()),
            },
        },
    }
    for i in range(args.num_heads):
        payload[f"label_t_plus_{i + 1}"] = labels_tensor[:, i]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        f"wrote {len(hidden_rows)} examples to {args.output} "
        f"hidden={tuple(payload['hidden'].shape)} labels={tuple(payload['labels'].shape)}"
    )


if __name__ == "__main__":
    main()
