"""Profile proposal overhead and draft fragmentation on zero-drift rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def _method_rows(steps: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    return [row for row in steps if row.get("method") == method]


def _tokens_by_task(completions: list[dict[str, Any]], method: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in completions:
        output = (row.get("outputs") or {}).get(method)
        if not output:
            continue
        out[str(row.get("task_id"))] = int(output.get("n_new_tokens") or 0)
    return out


def _wall_by_task(completions: list[dict[str, Any]], method: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in completions:
        output = (row.get("outputs") or {}).get(method)
        if not output:
            continue
        out[str(row.get("task_id"))] = float(output.get("wall_us") or 0.0)
    return out


def summarize(
    steps: list[dict[str, Any]],
    completions: list[dict[str, Any]],
    *,
    methods: list[str],
    vanilla_method: str = "vanilla",
) -> dict[str, Any]:
    vanilla_steps_by_task = Counter(
        str(row.get("task_id")) for row in steps if row.get("method") == vanilla_method
    )
    out: dict[str, Any] = {
        "schema": "asts-spec/zero-drift-overhead/v1",
        "vanilla_method": vanilla_method,
        "methods": {},
    }
    for method in methods:
        rows = _method_rows(steps, method)
        proposal_us = [float(row.get("proposal_us") or 0.0) for row in rows]
        wall_us = [float(row.get("wall_us") or 0.0) for row in rows]
        accepted_nonroot = [int(row.get("n_accepted_nonroot_drafts") or 0) for row in rows]
        proposal_tokens = [int(row.get("proposal_tokens") or 0) for row in rows]
        match_counts = Counter(str(row.get("proposal_match_kind") or "none") for row in rows)
        kind_counts = Counter(str(row.get("proposal_kind") or "none") for row in rows)
        route_counts = Counter(str(row.get("proposal_route") or "none") for row in rows)
        reason_counts = Counter(str(row.get("proposal_route_reason") or "none") for row in rows)
        fallback_count = sum(
            count
            for key, count in match_counts.items()
            if "fallback" in key or "exact_pld" in key or key == "none"
        )
        rewrite_count = sum(
            count
            for key, count in match_counts.items()
            if "bidir" in key or key in {"vref", "oracle"}
        )
        tokens_by_task = _tokens_by_task(completions, method)
        wall_by_task = _wall_by_task(completions, method)
        steps_by_task = Counter(str(row.get("task_id")) for row in rows)
        target_forward_reduction_values: list[float] = []
        for task_id, method_steps in steps_by_task.items():
            vanilla_steps = vanilla_steps_by_task.get(task_id)
            if vanilla_steps:
                target_forward_reduction_values.append(1.0 - method_steps / vanilla_steps)
        out["methods"][method] = {
            "steps": len(rows),
            "tasks": len(steps_by_task),
            "tokens": sum(tokens_by_task.values()),
            "wall_us_total": sum(wall_by_task.values()),
            "proposal_us_total": sum(proposal_us),
            "proposal_us_per_step_mean": _mean(proposal_us),
            "proposal_us_per_step_p50": _quantile(proposal_us, 0.50),
            "proposal_us_per_step_p95": _quantile(proposal_us, 0.95),
            "proposal_wall_share": sum(proposal_us) / sum(wall_us) if sum(wall_us) else 0.0,
            "accepted_nonroot_per_step": _mean([float(x) for x in accepted_nonroot]),
            "accepted_nonroot_per_hit": (
                sum(accepted_nonroot)
                / max(1, sum(1 for row in rows if int(row.get("proposal_tokens") or 0) > 0))
            ),
            "proposal_tokens_per_step": _mean([float(x) for x in proposal_tokens]),
            "proposal_tokens_per_hit": (
                sum(proposal_tokens)
                / max(1, sum(1 for row in rows if int(row.get("proposal_tokens") or 0) > 0))
            ),
            "target_forward_reduction_mean": _mean(target_forward_reduction_values),
            "match_kind_counts": dict(match_counts),
            "proposal_kind_counts": dict(kind_counts),
            "route_counts": dict(route_counts),
            "route_reason_counts": dict(reason_counts),
            "rewrite_lookup_steps": rewrite_count,
            "exact_or_fallback_steps": fallback_count,
        }
    return out


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Zero-Drift Overhead",
        "",
        "| Method | Steps | Proposal us/step p50 | Proposal us/step p95 | Proposal wall share | Accepted non-root/step | Target forward reduction | Rewrite lookup steps | Exact/fallback steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in report["methods"].items():
        lines.append(
            "| "
            f"{method} | "
            f"{row['steps']} | "
            f"{row['proposal_us_per_step_p50']:.1f} | "
            f"{row['proposal_us_per_step_p95']:.1f} | "
            f"{100 * row['proposal_wall_share']:.2f}% | "
            f"{row['accepted_nonroot_per_step']:.2f} | "
            f"{100 * row['target_forward_reduction_mean']:.1f}% | "
            f"{row['rewrite_lookup_steps']} | "
            f"{row['exact_or_fallback_steps']} |"
        )
    lines.append("")
    lines.append("## Route Counts")
    for method, row in report["methods"].items():
        lines.append("")
        lines.append(f"### {method}")
        lines.append("")
        lines.append("Match kinds:")
        for key, value in sorted(row["match_kind_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- `{key}`: {value}")
        if any(k != "none" for k in row["route_counts"]):
            lines.append("")
            lines.append("Routes:")
            for key, value in sorted(row["route_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"- `{key}`: {value}")
            lines.append("")
            lines.append("Route reasons:")
            for key, value in sorted(row["route_reason_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"- `{key}`: {value}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", required=True)
    p.add_argument("--completions", required=True)
    p.add_argument("--methods", required=True, help="Comma-separated methods to summarize.")
    p.add_argument("--vanilla-method", default="vanilla")
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    args = p.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    report = summarize(
        _load_jsonl(Path(args.steps)),
        _load_jsonl(Path(args.completions)),
        methods=methods,
        vanilla_method=args.vanilla_method,
    )
    Path(args.output_json).write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, Path(args.output_md))


if __name__ == "__main__":
    main()
