#!/usr/bin/env python3
"""Build lightweight residual-training tensors from post-PLD MTP artifacts.

This collector does not run a model and has no GPU dependency.  It reuses
existing post-PLD hidden-state artifacts when available, or creates a small
deterministic synthetic fixture for local pipeline tests.

By default it filters to weak/residual PLD positions with
``accepted_len <= 4``.  It never collects all rows unless the caller explicitly
sets ``--allow-all-tokens``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = ROOT / "analysis/pld_mtp/postpld_linear_k4_n917_v1"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/vantage_residual/data"
DEFAULT_TABLE = ROOT / "artifacts/vantage_residual/tables/residual_dataset_summary.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-source",
        type=Path,
        default=DEFAULT_SOURCE_ROOT / "postpld_train.pt",
        help="Existing post-PLD train tensor artifact.",
    )
    parser.add_argument(
        "--test-source",
        type=Path,
        default=DEFAULT_SOURCE_ROOT / "postpld_test500.pt",
        help="Existing post-PLD test tensor artifact.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--accepted-len-threshold", type=int, default=4)
    parser.add_argument("--min-accepted-len", type=int, default=0)
    parser.add_argument("--allow-all-tokens", action="store_true")
    parser.add_argument("--pld-miss-only", action="store_true")
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument(
        "--max-examples-per-split",
        type=int,
        default=0,
        help="Deterministic first-N cap after filtering. Zero means no cap.",
    )
    parser.add_argument(
        "--synthetic-fixture",
        action="store_true",
        help="Ignore source artifacts and write a deterministic tiny fixture.",
    )
    parser.add_argument(
        "--synthetic-if-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the deterministic fixture if source artifacts are absent.",
    )
    parser.add_argument("--synthetic-hidden-size", type=int, default=16)
    parser.add_argument("--synthetic-train-examples", type=int, default=12)
    parser.add_argument("--synthetic-test-examples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.accepted_len_threshold < 0 and not args.allow_all_tokens:
        raise SystemExit("--accepted-len-threshold must be non-negative")
    if args.min_accepted_len < 0:
        raise SystemExit("--min-accepted-len must be non-negative")
    if args.num_heads <= 0:
        raise SystemExit("--num-heads must be positive")

    import torch

    source_missing = not args.train_source.exists() or not args.test_source.exists()
    use_synthetic = bool(args.synthetic_fixture or (source_missing and args.synthetic_if_missing))
    if source_missing and not use_synthetic:
        missing = [str(path) for path in (args.train_source, args.test_source) if not path.exists()]
        raise SystemExit(f"missing source artifacts: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.summary_json is None:
        args.summary_json = args.output_dir / "residual_dataset_summary.json"

    if use_synthetic:
        sources = {
            "train": _synthetic_payload(
                torch,
                n_examples=args.synthetic_train_examples,
                hidden_size=args.synthetic_hidden_size,
                num_heads=args.num_heads,
                seed=args.seed,
                split="train",
            ),
            "test": _synthetic_payload(
                torch,
                n_examples=args.synthetic_test_examples,
                hidden_size=args.synthetic_hidden_size,
                num_heads=args.num_heads,
                seed=args.seed + 1,
                split="test",
            ),
        }
        source_kind = "synthetic_fixture"
    else:
        sources = {
            "train": torch.load(args.train_source, map_location="cpu"),
            "test": torch.load(args.test_source, map_location="cpu"),
        }
        source_kind = "postpld_mtp_artifact"

    split_summaries: dict[str, dict[str, Any]] = {}
    output_paths: dict[str, str] = {}
    split_filenames = {"train": "train.pt", "test": "test500.pt"}
    for split, payload in sources.items():
        output_path = args.output_dir / split_filenames[split]
        residual_payload, split_summary = _build_split(torch, payload, split, args)
        residual_payload["metadata"]["source_kind"] = source_kind
        torch.save(residual_payload, output_path)
        output_paths[split] = str(output_path)
        split_summaries[split] = split_summary | {"output": str(output_path)}

    summary = {
        "schema": "vantage_residual_dataset_summary_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_kind": source_kind,
        "train_source": str(args.train_source),
        "test_source": str(args.test_source),
        "allow_all_tokens": bool(args.allow_all_tokens),
        "accepted_len_threshold": args.accepted_len_threshold,
        "min_accepted_len": args.min_accepted_len,
        "pld_miss_only": bool(args.pld_miss_only),
        "num_heads": args.num_heads,
        "max_examples_per_split": args.max_examples_per_split,
        "outputs": output_paths,
        "splits": split_summaries,
        "notes": [
            "No model forward pass was run; tensors come from existing CPU-loaded artifacts or a deterministic fixture.",
            "Default collection is threshold-filtered and does not train on all token positions.",
        ],
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text("\n".join(_markdown_lines(summary)) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _build_split(torch: Any, payload: dict[str, Any], split: str, args: argparse.Namespace):
    hidden = _required_tensor(payload, "hidden")
    labels = _required_tensor(payload, "labels")
    accepted_len = _required_tensor(payload, "accepted_len").long()
    pld_miss = _optional_bool_tensor(torch, payload, "pld_miss", len(accepted_len))
    if labels.ndim != 2:
        raise SystemExit(f"{split}: labels must be rank-2, got {tuple(labels.shape)}")
    num_heads = min(int(args.num_heads), int(labels.shape[1]))

    if args.allow_all_tokens:
        mask = torch.ones_like(accepted_len, dtype=torch.bool)
    else:
        mask = accepted_len <= int(args.accepted_len_threshold)
    mask &= accepted_len >= int(args.min_accepted_len)
    if args.pld_miss_only:
        mask &= pld_miss

    selected = torch.nonzero(mask, as_tuple=False).flatten()
    eligible_examples = int(selected.numel())
    if args.max_examples_per_split > 0:
        selected = selected[: int(args.max_examples_per_split)]
    if selected.numel() == 0:
        raise SystemExit(f"{split}: no rows survived residual filtering")

    task_id = list(payload.get("task_id") or [f"{split}/{i:04d}" for i in range(len(accepted_len))])
    step_id = list(payload.get("step_id") or range(len(accepted_len)))
    selected_list = [int(i) for i in selected.tolist()]
    selected_task_id = [str(task_id[i]) for i in selected_list]
    selected_step_id = [int(step_id[i]) for i in selected_list]

    out_labels = labels.index_select(0, selected)[:, :num_heads].contiguous()
    out_hidden = hidden.index_select(0, selected).float().contiguous()
    out_accepted = accepted_len.index_select(0, selected).contiguous()
    out_miss = pld_miss.index_select(0, selected).contiguous()
    source_meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    residual_payload = {
        "hidden": out_hidden,
        "features": out_hidden,
        "labels": out_labels,
        "future_tokens_1_to_k": out_labels,
        "accepted_len": out_accepted,
        "pld_miss": out_miss,
        "is_pld_miss": out_miss,
        "trigger_accepted_len_eq_0": out_accepted == 0,
        "trigger_accepted_len_le_1": out_accepted <= 1,
        "trigger_accepted_len_le_2": out_accepted <= 2,
        "trigger_accepted_len_le_4": out_accepted <= 4,
        "trigger_pld_miss_only": out_miss,
        "task_id": selected_task_id,
        "step_id": selected_step_id,
        "source_index": selected,
        "residual_mask": torch.ones_like(out_accepted, dtype=torch.bool),
        "metadata": {
            "schema": "vantage_residual_training_split_v1",
            "split": split,
            "source_metadata": source_meta,
            "source_examples": int(len(accepted_len)),
            "eligible_examples": eligible_examples,
            "written_examples": int(selected.numel()),
            "hidden_size": int(out_hidden.shape[1]),
            "num_heads": num_heads,
            "accepted_len_threshold": args.accepted_len_threshold,
            "min_accepted_len": args.min_accepted_len,
            "allow_all_tokens": bool(args.allow_all_tokens),
            "pld_miss_only": bool(args.pld_miss_only),
            "selection": "first_n_after_filter",
        },
    }
    for i in range(num_heads):
        residual_payload[f"label_t_plus_{i + 1}"] = out_labels[:, i]

    summary = {
        "source_examples": int(len(accepted_len)),
        "eligible_examples": eligible_examples,
        "written_examples": int(selected.numel()),
        "hidden_shape": list(out_hidden.shape),
        "labels_shape": list(out_labels.shape),
        "accepted_len_histogram": _histogram(out_accepted.tolist()),
        "pld_miss_count": int(out_miss.sum().item()),
        "task_count": len(set(selected_task_id)),
        "allow_all_tokens": bool(args.allow_all_tokens),
    }
    return residual_payload, summary


def _required_tensor(payload: dict[str, Any], key: str) -> Any:
    value = payload.get(key)
    if value is None or not hasattr(value, "shape"):
        raise SystemExit(f"source artifact missing tensor key {key!r}")
    return value.cpu()


def _optional_bool_tensor(torch: Any, payload: dict[str, Any], key: str, length: int) -> Any:
    value = payload.get(key)
    if value is None or not hasattr(value, "shape"):
        return torch.zeros(length, dtype=torch.bool)
    return value.cpu().bool()


def _synthetic_payload(
    torch: Any,
    *,
    n_examples: int,
    hidden_size: int,
    num_heads: int,
    seed: int,
    split: str,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(int(seed))
    hidden = torch.randn(n_examples, hidden_size, generator=generator)
    labels = torch.randint(100, 1000, (n_examples, num_heads), generator=generator)
    accepted = torch.tensor([(i * 3) % 7 for i in range(n_examples)], dtype=torch.long)
    pld_miss = accepted == 0
    return {
        "hidden": hidden,
        "labels": labels,
        "accepted_len": accepted,
        "pld_miss": pld_miss,
        "task_id": [f"synthetic_{split}/{i // 2:04d}" for i in range(n_examples)],
        "step_id": list(range(n_examples)),
        "metadata": {
            "source": "deterministic_synthetic_fixture",
            "seed": seed,
            "split": split,
            "n_examples": n_examples,
            "hidden_size": hidden_size,
            "num_heads": num_heads,
        },
    }


def _histogram(values: list[int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(int(v) for v in values).items())}


def _markdown_lines(summary: dict[str, Any]) -> list[str]:
    lines = [
        "# Residual Dataset Summary",
        "",
        f"source kind: `{summary['source_kind']}`",
        f"accepted_len filter: `<= {summary['accepted_len_threshold']}`",
        f"allow all tokens: `{summary['allow_all_tokens']}`",
        f"PLD miss only: `{summary['pld_miss_only']}`",
        "",
        "| Split | Source rows | Eligible rows | Written rows | Hidden shape | Labels shape | PLD misses | Tasks | accepted_len histogram |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for split in ("train", "test"):
        row = summary["splits"].get(split, {})
        hist = ", ".join(f"{k}:{v}" for k, v in row.get("accepted_len_histogram", {}).items())
        lines.append(
            "| {split} | {source} | {eligible} | {written} | `{hidden}` | `{labels}` | {misses} | {tasks} | {hist} |".format(
                split=split,
                source=row.get("source_examples", 0),
                eligible=row.get("eligible_examples", 0),
                written=row.get("written_examples", 0),
                hidden=row.get("hidden_shape", []),
                labels=row.get("labels_shape", []),
                misses=row.get("pld_miss_count", 0),
                tasks=row.get("task_count", 0),
                hist=hist,
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- This is a data packaging step only; no residual model was trained.",
            "- Default collection is threshold-filtered and does not collect all token positions.",
        ]
    )
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
