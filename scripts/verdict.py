"""Decision-gate verdict: combine tree-sitter and model benchmark JSONs.

Usage:
    python scripts/verdict.py --treesitter out/treesitter.json --models out/models.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asts.analysis import compute_verdict, print_verdict


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--treesitter", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--k", type=int, default=8, help="draft length used in verdict")
    p.add_argument(
        "--a-values",
        default="1,2,4,6,8",
        help="comma-separated accepted-token-length values to sweep",
    )
    args = p.parse_args()

    ts_report = json.loads(Path(args.treesitter).read_text())
    model_report = json.loads(Path(args.models).read_text())
    a_values = tuple(int(x) for x in args.a_values.split(","))

    verdict = compute_verdict(ts_report, model_report, a_values=a_values, k=args.k)
    print_verdict(verdict)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(verdict, indent=2))
        print(f"wrote verdict report to {out}")


if __name__ == "__main__":
    main()
