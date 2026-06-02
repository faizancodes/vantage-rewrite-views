"""Offline step-replay projection for PLD plus MTP heads.

The evaluator can either consume trained head predictions from a collected
hidden-state dataset, or run an oracle upper-bound projection that assumes the
K heads predict the next K verified tokens perfectly.  The oracle mode is useful
before spending GPU budget on hidden-state collection and training: if perfect
K-token heads cannot reach the speedup target, runtime MTP is not worth
implementing.
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asts.mtp_heads import MTPHeadConfig, PLDMTPHeads, accepted_prefix_length  # noqa: E402
from scripts.train_pld_candidate_reranker import _load_jsonl  # noqa: E402


@dataclass(frozen=True)
class MTPReplayStep:
    task_id: str
    step_id: int
    start: int
    emitted: int
    accepted_len: int
    pld_miss: bool


def _load_steps(path: Path, *, method: str) -> dict[str, list[MTPReplayStep]]:
    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _load_jsonl(path):
        if row.get("method") == method:
            rows_by_task[str(row.get("task_id") or "")].append(row)
    out: dict[str, list[MTPReplayStep]] = {}
    for task_id, rows in rows_by_task.items():
        pos = 0
        steps = []
        for row in sorted(rows, key=lambda r: int(r.get("step") or 0)):
            accepted = int(row.get("n_accepted_drafts") or 0)
            pld_miss = not (
                row.get("pld_exact_hit") is True or row.get("proposal_kind") == "blazedit_pld"
            )
            steps.append(
                MTPReplayStep(
                    task_id=task_id,
                    step_id=int(row.get("step") or 0),
                    start=pos,
                    emitted=max(1, int(row.get("n_emitted") or 0)),
                    accepted_len=accepted,
                    pld_miss=pld_miss,
                )
            )
            pos += max(1, int(row.get("n_emitted") or 0))
        out[task_id] = steps
    return out


def _load_steps_from_mtp_data(path: Path) -> dict[str, list[MTPReplayStep]]:
    import torch

    data = torch.load(path, map_location="cpu")
    task_ids = list(data["task_id"])
    step_ids = [int(x) for x in data["step_id"]]
    starts = [int(x) for x in data["generated_start"].tolist()]
    emitted = [max(1, int(x)) for x in data["n_emitted"].tolist()]
    accepted = [int(x) for x in data["accepted_len"].tolist()]
    miss_tensor = data.get("is_pld_miss", data.get("pld_miss"))
    if miss_tensor is None:
        misses = [a == 0 for a in accepted]
    else:
        misses = [bool(x) for x in miss_tensor.tolist()]
    by_task: dict[str, list[MTPReplayStep]] = defaultdict(list)
    for task_id, step_id, start, emit, acc, miss in zip(
        task_ids, step_ids, starts, emitted, accepted, misses, strict=True
    ):
        by_task[str(task_id)].append(
            MTPReplayStep(
                task_id=str(task_id),
                step_id=step_id,
                start=start,
                emitted=emit,
                accepted_len=acc,
                pld_miss=miss,
            )
        )
    for rows in by_task.values():
        rows.sort(key=lambda s: s.step_id)
    return dict(by_task)


def _trigger(policy: str, step: MTPReplayStep) -> bool:
    if policy == "accepted_len_eq_0":
        return step.accepted_len == 0
    if policy == "accepted_len_le_1":
        return step.accepted_len <= 1
    if policy == "accepted_len_le_2":
        return step.accepted_len <= 2
    if policy == "accepted_len_le_4":
        return step.accepted_len <= 4
    if policy == "pld_miss_only":
        return step.pld_miss
    raise ValueError(f"unknown trigger policy {policy!r}")


def _load_predictions_from_heads(
    *,
    data_path: Path,
    heads_path: Path,
    batch_size: int,
    device: str,
) -> tuple[dict[tuple[str, int], int], dict[str, Any]]:
    import torch

    data = torch.load(data_path, map_location="cpu")
    ckpt = torch.load(heads_path, map_location="cpu")
    config = MTPHeadConfig(**ckpt["config"])
    output_weight = ckpt.get("output_weight")
    if output_weight is not None:
        # Hidden states and trained adapter weights are evaluated in fp32.
        # Some saved output projections are bf16/fp16 from the target model;
        # cast them to fp32 here to avoid dtype-mismatch failures during the
        # tied-output matmul.
        output_weight = output_weight.to(device=device, dtype=torch.float32)
    model = PLDMTPHeads(config, output_weight=output_weight).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    hidden = data["hidden"].float()
    labels = data["labels"].long()[:, : config.num_heads]
    task_ids = list(data["task_id"])
    step_ids = list(data["step_id"])
    predictions: dict[tuple[str, int], int] = {}
    accepted_lengths = []
    horizon_correct = [0 for _ in range(config.num_heads)]
    horizon_total = [0 for _ in range(config.num_heads)]
    with torch.no_grad():
        for start in range(0, hidden.shape[0], batch_size):
            x = hidden[start : start + batch_size].to(device)
            logits = model(x)
            pred_by_head = [item.argmax(dim=-1).cpu().tolist() for item in logits]
            for j in range(x.shape[0]):
                idx = start + j
                pred = [pred_by_head[h][j] for h in range(config.num_heads)]
                label = [int(v) for v in labels[idx].tolist() if int(v) >= 0]
                accepted = accepted_prefix_length(pred, label)
                predictions[(str(task_ids[idx]), int(step_ids[idx]))] = accepted
                accepted_lengths.append(accepted)
                for h in range(config.num_heads):
                    if h < len(label):
                        horizon_total[h] += 1
                        horizon_correct[h] += int(int(pred[h]) == int(label[h]))
    meta = {
        "mode": "trained_heads",
        "data": str(data_path),
        "heads": str(heads_path),
        "data_mtp_position": (data.get("metadata") or {}).get("mtp_position"),
        "num_predictions": len(predictions),
        "mean_mtp_accepted_prefix": (
            sum(accepted_lengths) / len(accepted_lengths) if accepted_lengths else 0.0
        ),
        "top1_accuracy_by_horizon_pct": [
            100.0 * c / t if t else 0.0 for c, t in zip(horizon_correct, horizon_total)
        ],
    }
    return predictions, meta


def _project_policy(
    *,
    steps_by_task: dict[str, list[MTPReplayStep]],
    policy: str,
    num_heads: int,
    predictions: dict[tuple[str, int], int] | None,
    oracle_upper_bound: bool,
    mtp_position: str = "pre_pld",
) -> dict[str, Any]:
    if mtp_position not in {"pre_pld", "post_pld"}:
        raise ValueError(f"unsupported mtp_position: {mtp_position!r}")
    baseline_steps = sum(len(v) for v in steps_by_task.values())
    projected_steps = 0
    skipped_steps = 0
    triggers = 0
    missing_predictions = 0
    mtp_accepts: list[int] = []
    extras: list[int] = []
    selected_emits: list[int] = []
    horizon_hits = [0 for _ in range(num_heads)]
    prefix_hist = {str(i): 0 for i in range(num_heads + 1)}

    for task_id, steps in steps_by_task.items():
        starts = [s.start for s in steps]
        task_end = steps[-1].start + steps[-1].emitted if steps else 0
        i = 0
        while i < len(steps):
            step = steps[i]
            projected_steps += 1
            selected_emit = step.emitted
            if _trigger(policy, step):
                triggers += 1
                if oracle_upper_bound:
                    label_start = step.start if mtp_position == "pre_pld" else step.start + step.accepted_len
                    mtp_accept = min(num_heads, max(0, task_end - label_start))
                else:
                    if predictions is None or (task_id, step.step_id) not in predictions:
                        missing_predictions += 1
                        mtp_accept = 0
                    else:
                        mtp_accept = predictions[(task_id, step.step_id)]
                mtp_accepts.append(mtp_accept)
                prefix_hist[str(min(num_heads, max(0, mtp_accept)))] += 1
                for h in range(num_heads):
                    if mtp_accept >= h + 1:
                        horizon_hits[h] += 1
                mtp_emit = mtp_accept
                if mtp_position == "post_pld":
                    mtp_emit += max(0, step.accepted_len)
                selected_emit = max(selected_emit, mtp_emit)
                extras.append(max(0, selected_emit - step.emitted))
                selected_emits.append(selected_emit)
            if selected_emit <= step.emitted:
                i += 1
                continue
            covered_until = step.start + selected_emit
            j = bisect.bisect_left(starts, covered_until, lo=i + 1)
            skipped_steps += max(0, j - (i + 1))
            i = max(j, i + 1)

    projected_steps = max(1, projected_steps)
    token0_rejects = sum(1 for v in mtp_accepts if v == 0)
    return {
        "trigger_policy": policy,
        "mtp_position": mtp_position,
        "baseline_steps": baseline_steps,
        "projected_steps": projected_steps,
        "step_reduction_pct": 100.0 * (baseline_steps - projected_steps) / baseline_steps
        if baseline_steps
        else 0.0,
        "corrected_projected_speedup": baseline_steps / projected_steps,
        "trigger_count": triggers,
        "missing_predictions": missing_predictions,
        "avg_extra_accepted_mtp_tokens_per_trigger": (
            sum(extras) / len(extras) if extras else 0.0
        ),
        "avg_selected_emit_per_trigger": (
            sum(selected_emits) / len(selected_emits) if selected_emits else 0.0
        ),
        "baseline_progress_semantics": "step.emitted (normally accepted_len + 1)",
        "post_pld_progress_rule": (
            "combined_progress=max(step.emitted, accepted_len + mtp_accepted_prefix)"
        ),
        "avg_mtp_accepted_prefix": (
            sum(mtp_accepts) / len(mtp_accepts) if mtp_accepts else 0.0
        ),
        "token0_rejection_rate_pct": 100.0 * token0_rejects / len(mtp_accepts)
        if mtp_accepts
        else 0.0,
        "accepted_by_horizon_pct": [
            100.0 * count / len(mtp_accepts) if mtp_accepts else 0.0
            for count in horizon_hits
        ],
        "accepted_prefix_len_distribution": prefix_hist,
        "accepted_prefix_len_distribution_pct": {
            k: 100.0 * v / len(mtp_accepts) if mtp_accepts else 0.0
            for k, v in prefix_hist.items()
        },
        "skipped_baseline_steps": skipped_steps,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# PLD + MTP Heads Offline Projection",
        "",
        f"mode: `{payload['mode']}`",
        f"steps: `{payload['steps']}`",
        f"method: `{payload['method']}`",
        f"num_heads: `{payload['num_heads']}`",
        f"mtp_position: `{payload.get('mtp_position', 'pre_pld')}`",
        "",
        "| trigger policy | baseline steps | triggers | projected steps | step reduction | speedup | avg extra | token0 reject | prefix dist | horizon hit % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in payload["policies"]:
        horizons = ", ".join(f"h{i + 1}:{v:.1f}" for i, v in enumerate(row["accepted_by_horizon_pct"]))
        dist = ", ".join(
            f"{k}:{v}" for k, v in row["accepted_prefix_len_distribution"].items()
        )
        lines.append(
            f"| {row['trigger_policy']} | {row['baseline_steps']} | {row['trigger_count']} | {row['projected_steps']} | "
            f"{row['step_reduction_pct']:.2f}% | {row['corrected_projected_speedup']:.3f}x | "
            f"{row['avg_extra_accepted_mtp_tokens_per_trigger']:.2f} | "
            f"{row['token0_rejection_rate_pct']:.1f}% | {dist} | {horizons} |"
        )
    if payload["metadata"].get("top1_accuracy_by_horizon_pct"):
        lines.append("")
        acc = ", ".join(
            f"t+{i + 1}: {v:.2f}%"
            for i, v in enumerate(payload["metadata"]["top1_accuracy_by_horizon_pct"])
        )
        lines.append(f"Top-1 accuracy by horizon: {acc}")
    lines.append("")
    lines.append(payload["recommendation"])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=Path, default=None)
    ap.add_argument("--method", default="blazedit_pld_w128_n10")
    ap.add_argument("--output-dir", type=Path, default=Path("/tmp/pld_mtp/offline_eval"))
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--heads", type=Path, default=None)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--oracle-upper-bound", action="store_true")
    ap.add_argument("--mtp-position", choices=["pre_pld", "post_pld"], default="pre_pld")
    args = ap.parse_args()

    if args.steps is not None:
        steps_by_task = _load_steps(args.steps, method=args.method)
        steps_label = str(args.steps)
    elif args.data is not None:
        steps_by_task = _load_steps_from_mtp_data(args.data)
        steps_label = f"{args.data}::collected_steps"
    else:
        raise SystemExit("--steps is required unless --data contains collected all-step metadata")
    if not steps_by_task:
        raise SystemExit(f"no {args.method} rows found")

    predictions = None
    meta: dict[str, Any]
    if args.oracle_upper_bound:
        meta = {"mode": "oracle_upper_bound"}
    else:
        if not args.data or not args.heads:
            raise SystemExit("--data and --heads are required unless --oracle-upper-bound is set")
        predictions, meta = _load_predictions_from_heads(
            data_path=args.data,
            heads_path=args.heads,
            batch_size=args.batch_size,
            device=args.device,
        )

    policies = [
        _project_policy(
            steps_by_task=steps_by_task,
            policy=policy,
            num_heads=args.num_heads,
            predictions=predictions,
            oracle_upper_bound=args.oracle_upper_bound,
            mtp_position=args.mtp_position,
        )
        for policy in (
            "accepted_len_eq_0",
            "accepted_len_le_1",
            "accepted_len_le_2",
            "accepted_len_le_4",
            "pld_miss_only",
        )
    ]
    best = max(policies, key=lambda r: r["corrected_projected_speedup"])
    passes = (
        best["avg_extra_accepted_mtp_tokens_per_trigger"] >= 1.2
        and best["corrected_projected_speedup"] >= 1.20
    )
    recommendation = (
        "MTP offline diagnostic passes the runtime-implementation threshold."
        if passes
        else "MTP offline diagnostic does not pass the runtime-implementation threshold."
    )
    payload = {
        "mode": meta["mode"],
        "metadata": meta,
        "steps": steps_label,
        "method": args.method,
        "num_heads": args.num_heads,
        "mtp_position": args.mtp_position,
        "policies": policies,
        "best_policy": best["trigger_policy"],
        "recommendation": recommendation,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(args.output_dir / "report.md", payload)
    print((args.output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
