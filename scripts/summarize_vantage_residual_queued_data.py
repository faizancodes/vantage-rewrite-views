#!/usr/bin/env python3
"""Summarize queued-use VANTAGE-Residual tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "artifacts/vantage_residual/phase4_data/queued_v1"
DEFAULT_OUTPUT = ROOT / "artifacts/vantage_residual/tables/queued_dataset_summary.md"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    import torch

    obj = torch.load(path, map_location="cpu")
    return obj if isinstance(obj, dict) else None


def _task_ids(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("task_id") or payload.get("task_ids") or []
    return [str(x) for x in raw]


def _split_summary(path: Path) -> dict[str, Any]:
    import torch

    payload = _load_payload(path)
    if payload is None:
        return {"path": _rel(path), "exists": False}
    meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    hidden = payload.get("hidden")
    labels = payload.get("labels")
    task_ids = _task_ids(payload)
    valid = payload.get("valid_queued_example")
    accepted = payload.get("create_accepted_len", payload.get("accepted_len"))
    use_accepted = payload.get("use_step_accepted_len")
    return {
        "path": _rel(path),
        "exists": True,
        "examples": len(task_ids),
        "valid_examples": int(valid.bool().sum().item()) if torch.is_tensor(valid) else len(task_ids),
        "task_count": len(set(task_ids)),
        "hidden_shape": list(hidden.shape) if hasattr(hidden, "shape") else None,
        "labels_shape": list(labels.shape) if hasattr(labels, "shape") else None,
        "label_mode": meta.get("label_mode"),
        "hidden_source": meta.get("hidden_source"),
        "labels_aligned_to": meta.get("labels_aligned_to"),
        "accepted_len_le_4_rate": _rate_le(accepted, 4),
        "use_token0_reject_rate": _rate_eq(use_accepted, 0),
    }


def _rate_le(value: Any, threshold: int) -> float | None:
    import torch

    if not torch.is_tensor(value) or int(value.numel()) == 0:
        return None
    return float((value.long() <= int(threshold)).float().mean().item())


def _rate_eq(value: Any, target: int) -> float | None:
    import torch

    if not torch.is_tensor(value) or int(value.numel()) == 0:
        return None
    return float((value.long() == int(target)).float().mean().item())


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Queued Dataset Audit",
        "",
        f"data dir: `{summary['data_dir']}`",
        "",
        "| split | exists | examples | valid | tasks | hidden | labels | label mode | hidden source | t+1 labels | use token0 reject |",
        "|---|---|---:|---:|---:|---|---|---|---|---|---:|",
    ]
    for split in ("train", "val", "test"):
        row = summary["splits"][split]
        lines.append(
            f"| {split} | `{row.get('exists')}` | {row.get('examples', 0)} | "
            f"{row.get('valid_examples', 0)} | {row.get('task_count', 0)} | "
            f"`{row.get('hidden_shape')}` | `{row.get('labels_shape')}` | "
            f"`{row.get('label_mode')}` | `{row.get('hidden_source')}` | "
            f"`{row.get('labels_aligned_to')}` | {_fmt_pct(row.get('use_token0_reject_rate'))} |"
        )
    lines.extend(["", f"Gate: **{summary['gate']}**"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    splits = {split: _split_summary(args.data_dir / f"{split}.pt") for split in ("train", "val", "test")}
    train_valid = int(splits["train"].get("valid_examples", 0))
    test_valid = int(splits["test"].get("valid_examples", 0))
    total_valid = sum(int(row.get("valid_examples", 0)) for row in splits.values())
    queued_compatible = all(
        row.get("label_mode") in {"queued_use", "router_selected_queued_use"}
        for row in splits.values()
        if row.get("exists")
    )
    gate = (
        "pass"
        if train_valid >= 5000 and test_valid >= 1000 and queued_compatible
        else "fail: fewer than 5000 train or 1000 test valid queued examples, or not queued-label-compatible"
    )
    summary = {
        "data_dir": _rel(args.data_dir),
        "splits": splits,
        "train_valid": train_valid,
        "test_valid": test_valid,
        "total_valid": total_valid,
        "gate": gate,
    }
    args.output_md.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(args.output_md, summary)
    print(args.output_md.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
