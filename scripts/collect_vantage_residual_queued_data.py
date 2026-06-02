#!/usr/bin/env python3
"""Collect and split true queued-use VANTAGE-Residual tensors.

This is a thin, provenance-preserving wrapper around
``scripts/collect_queued_mtp_training_data.py``.  It exists so Phase 4 has a
single command that produces the required ``train.pt``, ``val.pt``, ``test.pt``
and summary artifacts while refusing to fabricate hidden states when no tensor
input or trace/model collection command is available.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/vantage_residual/phase4_data/queued_v1"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _torch_load(path: Path) -> dict[str, Any]:
    import torch

    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise SystemExit(f"{path} is not a tensor dictionary")
    return obj


def _resolve_raw_input(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "queued_raw.pt"
        if candidate.exists():
            return candidate
    return path


def _task_ids(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("task_id") or payload.get("task_ids")
    if raw is None:
        raise SystemExit("queued dataset is missing task_id/task_ids")
    return [str(x) for x in raw]


def _subset(payload: dict[str, Any], indices: list[int], *, split: str) -> dict[str, Any]:
    import torch

    idx = torch.tensor(indices, dtype=torch.long)
    out: dict[str, Any] = {}
    n = len(_task_ids(payload))
    for key, value in payload.items():
        if torch.is_tensor(value) and value.shape[:1] == (n,):
            out[key] = value[idx].clone()
        elif isinstance(value, list) and len(value) == n:
            out[key] = [value[i] for i in indices]
        else:
            out[key] = value
    meta = dict(payload.get("metadata") or {})
    meta["split"] = split
    meta["n_examples"] = len(indices)
    meta["task_count"] = len(set(_task_ids(out))) if indices else 0
    meta["label_mode"] = meta.get("label_mode") or "queued_use"
    meta["phase4_split_by_task"] = True
    out["metadata"] = meta
    return out


def _summarize_payload(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    import torch

    task_ids = _task_ids(payload)
    accepted = payload.get("create_accepted_len", payload.get("accepted_len"))
    accepted_hist: dict[str, int] = {}
    if torch.is_tensor(accepted):
        accepted_hist = {str(int(k)): int(v) for k, v in Counter(accepted.long().tolist()).items()}
    valid = payload.get("valid_queued_example")
    valid_count = int(valid.bool().sum().item()) if torch.is_tensor(valid) else len(task_ids)
    labels = payload.get("labels")
    hidden = payload.get("hidden")
    meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return {
        "path": _rel(path),
        "exists": path.exists(),
        "examples": len(task_ids),
        "valid_queued_examples": valid_count,
        "task_count": len(set(task_ids)),
        "hidden_shape": list(hidden.shape) if hasattr(hidden, "shape") else None,
        "labels_shape": list(labels.shape) if hasattr(labels, "shape") else None,
        "label_mode": meta.get("label_mode"),
        "hidden_source": meta.get("hidden_source"),
        "accepted_len_histogram": dict(sorted(accepted_hist.items(), key=lambda kv: int(kv[0]))),
    }


def _split_indices(
    task_ids: list[str],
    *,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, list[int]]:
    tasks = sorted(set(task_ids))
    rng = random.Random(seed)
    rng.shuffle(tasks)
    n_test = int(round(len(tasks) * test_fraction))
    n_val = int(round(len(tasks) * val_fraction))
    test_tasks = set(tasks[:n_test])
    val_tasks = set(tasks[n_test : n_test + n_val])
    train_tasks = set(tasks[n_test + n_val :])
    splits = {"train": [], "val": [], "test": []}
    for i, task_id in enumerate(task_ids):
        if task_id in test_tasks:
            splits["test"].append(i)
        elif task_id in val_tasks:
            splits["val"].append(i)
        elif task_id in train_tasks:
            splits["train"].append(i)
    return splits


def _run_collection(args: argparse.Namespace, output: Path) -> list[str]:
    if args.source_traces is None or args.completions is None:
        raise SystemExit(
            "No queued tensor input exists. Provide --input-pt, or provide both "
            "--source-traces and --completions so hidden states can be collected."
        )
    cmd = [
        sys.executable,
        str(ROOT / "scripts/collect_queued_mtp_training_data.py"),
        "--target",
        args.target,
        "--steps",
        str(args.source_traces),
        "--completions",
        str(args.completions),
        "--output",
        str(output),
        "--method",
        args.method,
        "--num-heads",
        str(args.k),
        "--trigger-threshold",
        str(args.trigger_threshold),
        "--weak-field",
        args.weak_field,
        "--include-dropped",
        str(args.include_dropped).lower(),
        "--max-examples",
        str(args.max_examples),
        "--max-tasks",
        str(args.max_tasks),
        "--dtype",
        args.dtype,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
    ]
    if args.output_projection is not None:
        cmd.extend(["--output-projection", str(args.output_projection)])
    subprocess.run(cmd, cwd=ROOT, check=True)
    return cmd


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# VANTAGE-Residual Queued Dataset Summary",
        "",
        f"source tensor: `{summary['source_tensor']}`",
        f"label semantics: `{summary['queued_semantics']['label']}`",
        f"input semantics: `{summary['queued_semantics']['input']}`",
        "",
        "| split | examples | valid | tasks | hidden | labels | label mode | hidden source |",
        "|---|---:|---:|---:|---|---|---|---|",
    ]
    for split in ("train", "val", "test"):
        row = summary["splits"][split]
        lines.append(
            f"| {split} | {row['examples']} | {row['valid_queued_examples']} | "
            f"{row['task_count']} | `{row['hidden_shape']}` | `{row['labels_shape']}` | "
            f"`{row.get('label_mode')}` | `{row.get('hidden_source')}` |"
        )
    lines.extend(
        [
            "",
            f"Task leakage: `{summary['task_leakage']}`",
            f"Dataset gate: **{summary['dataset_gate']}**",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-traces", "--steps", type=Path, default=None)
    ap.add_argument("--completions", type=Path, default=None)
    ap.add_argument("--input-pt", type=Path, default=None)
    ap.add_argument("--train-raw", type=Path, default=None)
    ap.add_argument("--test-raw", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--target", default="Qwen/Qwen2.5-Coder-7B")
    ap.add_argument("--method", default="blazedit_pld_w128_n10")
    ap.add_argument("--k", "--num-heads", dest="k", type=int, default=4)
    ap.add_argument("--trigger-threshold", type=int, default=4)
    ap.add_argument("--weak-field", choices=["draft_len", "accepted_len"], default="draft_len")
    ap.add_argument("--include-dropped", action="store_true")
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--output-projection", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--test-fraction", type=float, default=0.10)
    args = ap.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.train_raw is not None or args.test_raw is not None:
        if args.train_raw is None or args.test_raw is None:
            raise SystemExit("--train-raw and --test-raw must be provided together")
        import torch

        train_payload = _torch_load(_resolve_raw_input(args.train_raw))
        test_payload = _torch_load(_resolve_raw_input(args.test_raw))
        for name, payload in (("train", train_payload), ("test", test_payload)):
            meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            if meta.get("label_mode") not in {"queued_use", "router_selected_queued_use"}:
                raise SystemExit(f"{name} raw tensor is not queued-use data")
        train_task_ids = _task_ids(train_payload)
        train_splits = _split_indices(
            train_task_ids,
            seed=int(args.seed),
            val_fraction=float(args.val_fraction),
            test_fraction=0.0,
        )
        split_paths: dict[str, Path] = {}
        train_out = _subset(train_payload, train_splits["train"], split="train")
        val_out = _subset(train_payload, train_splits["val"], split="val")
        test_out = _subset(test_payload, list(range(len(_task_ids(test_payload)))), split="test")
        for split, payload in (("train", train_out), ("val", val_out), ("test", test_out)):
            path = args.output_dir / f"{split}.pt"
            torch.save(payload, path)
            split_paths[split] = path
        train_tasks = set(_task_ids(train_out))
        val_tasks = set(_task_ids(val_out))
        test_tasks = set(_task_ids(test_out))
        leakage = {
            "train_val_overlap": sorted(train_tasks & val_tasks),
            "train_test_overlap": sorted(train_tasks & test_tasks),
            "val_test_overlap": sorted(val_tasks & test_tasks),
        }
        leak_count = sum(len(v) for v in leakage.values())
        summaries = {
            split: _summarize_payload(path, _torch_load(path))
            for split, path in split_paths.items()
        }
        dataset_gate = (
            "pass"
            if summaries["train"]["valid_queued_examples"] >= 5000
            and summaries["test"]["valid_queued_examples"] >= 1000
            and leak_count == 0
            else "fail: fewer than 5000 train or 1000 test valid queued examples, or task leakage detected"
        )
        summary = {
            "source_tensor": None,
            "train_raw": _rel(_resolve_raw_input(args.train_raw)),
            "test_raw": _rel(_resolve_raw_input(args.test_raw)),
            "collection_command": None,
            "queued_semantics": {
                "input": "hidden state from already-computed verifier call at step t",
                "label": "target token(s) for step t+1 queued use",
                "forbidden_runtime_inputs": [
                    "current-step t+1 accepted_len",
                    "current-step t+1 verifier result",
                    "oracle trigger labels",
                ],
            },
            "splits": summaries,
            "task_leakage": leakage,
            "dataset_gate": dataset_gate,
        }
        (args.output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        _write_markdown(args.output_dir / "dataset_summary.md", summary)
        print((args.output_dir / "dataset_summary.md").read_text())
        return 0

    all_pt = args.output_dir / "all.pt"
    collection_command: list[str] | None = None
    if args.input_pt is not None:
        all_pt = args.input_pt
    elif not all_pt.exists():
        collection_command = _run_collection(args, all_pt)

    payload = _torch_load(all_pt)
    meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    label_mode = meta.get("label_mode")
    if label_mode not in {"queued_use", "router_selected_queued_use"}:
        raise SystemExit(f"{all_pt} is not queued-use data; metadata.label_mode={label_mode!r}")
    task_ids = _task_ids(payload)
    splits = _split_indices(
        task_ids,
        seed=int(args.seed),
        val_fraction=float(args.val_fraction),
        test_fraction=float(args.test_fraction),
    )
    import torch

    split_paths: dict[str, Path] = {}
    for split, indices in splits.items():
        out = _subset(payload, indices, split=split)
        path = args.output_dir / f"{split}.pt"
        torch.save(out, path)
        split_paths[split] = path

    train_tasks = set(_task_ids(_torch_load(split_paths["train"])))
    val_tasks = set(_task_ids(_torch_load(split_paths["val"])))
    test_tasks = set(_task_ids(_torch_load(split_paths["test"])))
    leakage = {
        "train_val_overlap": sorted(train_tasks & val_tasks),
        "train_test_overlap": sorted(train_tasks & test_tasks),
        "val_test_overlap": sorted(val_tasks & test_tasks),
    }
    leak_count = sum(len(v) for v in leakage.values())
    valid_examples = sum(_summarize_payload(split_paths[s], _torch_load(split_paths[s]))["valid_queued_examples"] for s in splits)
    dataset_gate = (
        "pass"
        if valid_examples >= 5000 and leak_count == 0
        else "fail: fewer than 5000 valid queued examples or task leakage detected"
    )
    summary = {
        "source_tensor": _rel(all_pt),
        "collection_command": collection_command,
        "queued_semantics": {
            "input": "hidden state from already-computed verifier call at step t",
            "label": "target token(s) for step t+1 queued use",
            "forbidden_runtime_inputs": [
                "current-step t+1 accepted_len",
                "current-step t+1 verifier result",
                "oracle trigger labels",
            ],
        },
        "splits": {
            split: _summarize_payload(path, _torch_load(path))
            for split, path in split_paths.items()
        },
        "task_leakage": leakage,
        "dataset_gate": dataset_gate,
    }
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(args.output_dir / "dataset_summary.md", summary)
    print((args.output_dir / "dataset_summary.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
