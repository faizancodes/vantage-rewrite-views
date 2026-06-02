"""Collect compact hidden-state traces for PLD candidate reranking.

This is an offline diagnostic collector, not a runtime decoder.  It replays an
existing PLD completion, runs the target model over ``prompt + PLD output``, and
stores projected hidden vectors for ambiguous exact-PLD candidate source
positions.  The resulting JSONL is consumed by
``scripts/evaluate_pld_hidden_state_reranker.py``.
"""

from __future__ import annotations

import argparse
import json
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


def _step_starts(steps_path: Path, *, method: str) -> dict[str, dict[int, int]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _load_jsonl(steps_path):
        if row.get("method") == method:
            by_task[str(row.get("task_id") or "")].append(row)
    starts_by_task: dict[str, dict[int, int]] = {}
    for task_id, rows in by_task.items():
        pos = 0
        starts: dict[int, int] = {}
        for row in sorted(rows, key=lambda r: int(r.get("step") or 0)):
            step_id = int(row.get("step") or 0)
            starts[step_id] = pos
            pos += max(1, int(row.get("n_emitted") or 0))
        starts_by_task[task_id] = starts
    return starts_by_task


def _project(v, matrix):
    import torch

    out = torch.matmul(v.float(), matrix)
    norm = torch.linalg.vector_norm(out)
    if float(norm) > 0:
        out = out / norm
    return [round(float(x), 6) for x in out.cpu().tolist()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ambiguous-candidates", type=Path, required=True)
    ap.add_argument("--steps", type=Path, required=True)
    ap.add_argument("--completions", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    ap.add_argument("--method", default="blazedit_pld_w128_n10")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--proj-dim", type=int, default=128)
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.target,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()

    completions = {str(r["task_id"]): r for r in _load_jsonl(args.completions)}
    starts_by_task = _step_starts(args.steps, method=args.method)
    ambig_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _load_jsonl(args.ambiguous_candidates):
        ambig_by_task[str(row.get("task_id") or "")].append(row)

    tasks = [t for t in sorted(ambig_by_task) if t in completions and t in starts_by_task]
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    wrote = 0
    with args.output.open("w") as f:
        for task_id in tasks:
            comp = completions[task_id]
            prompt_ids = encode_no_special(tokenizer, str(comp.get("prompt") or ""))
            output_ids = _output_tokens(tokenizer, comp, args.method)
            ids = prompt_ids + output_ids
            if not ids:
                continue
            input_ids = torch.tensor([ids], device=args.device)
            with torch.no_grad():
                hidden = model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    use_cache=False,
                ).hidden_states[-1][0]
            gen = torch.Generator(device=hidden.device)
            gen.manual_seed(17)
            proj = torch.randn(
                hidden.shape[-1],
                args.proj_dim,
                generator=gen,
                device=hidden.device,
                dtype=torch.float32,
            ) / (hidden.shape[-1] ** 0.5)
            starts = starts_by_task[task_id]
            for row in ambig_by_task[task_id]:
                step_id = int(row.get("step_id") or 0)
                if step_id not in starts:
                    continue
                prefix_len = len(prompt_ids) + starts[step_id]
                current_idx = max(0, min(prefix_len - 1, hidden.shape[0] - 1))
                match_len = max(1, int(row.get("match_len") or 1))
                crows = []
                for cand in list(row.get("candidates") or [])[: args.top_k]:
                    source_pos = int(cand.get("source_position") or 0)
                    source_idx = max(0, min(source_pos + match_len - 1, hidden.shape[0] - 1))
                    lo = max(0, source_idx - 4)
                    hi = min(hidden.shape[0], source_idx + 5)
                    crows.append(
                        {
                            "rank": int(cand.get("rank") or len(crows) + 1) - 1,
                            "source_position": source_pos,
                            "source_hidden": _project(hidden[source_idx], proj),
                            "source_window_mean_hidden": _project(hidden[lo:hi].mean(dim=0), proj),
                        }
                    )
                f.write(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "step_id": step_id,
                            "current_hidden": _project(hidden[current_idx], proj),
                            "candidates": crows,
                        }
                    )
                    + "\n"
                )
                wrote += 1
    print(f"wrote {wrote} hidden-state rows to {args.output}")


if __name__ == "__main__":
    main()
