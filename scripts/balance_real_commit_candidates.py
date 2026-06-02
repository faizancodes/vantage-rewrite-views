#!/usr/bin/env python3
"""Interleave mined real-commit candidates before manifest verification.

The local miner emits candidates in repository order.  For large manifests this
can make the downstream verifier stop after the first few repositories.  This
script keeps the candidate set unchanged but reorders it round-robin by
repository and coarse drift family so early accepted rows are more diverse.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("repo") or ""), str(row.get("drift_family") or "")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--max-candidates", type=int, default=0)
    args = p.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.input_jsonl).read_text().splitlines()
        if line.strip()
    ]
    buckets: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
    for row in rows:
        buckets[_key(row)].append(row)

    # Larger buckets stay in the rotation, but every nonempty repo/family bucket
    # gets a chance before a dominant repository contributes a second row.
    active = deque(sorted(buckets))
    out: list[dict[str, Any]] = []
    while active:
        key = active.popleft()
        bucket = buckets[key]
        if not bucket:
            continue
        out.append(bucket.popleft())
        if bucket:
            active.append(key)
        if args.max_candidates and len(out) >= args.max_candidates:
            break

    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for row in out:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {len(out)} balanced candidates to {output}")


if __name__ == "__main__":
    main()
