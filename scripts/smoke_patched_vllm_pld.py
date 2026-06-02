#!/usr/bin/env python3
"""Smoke-test the no-build VANTAGE PLD n-gram shim.

By default this script patches a temporary dummy ``ngram_proposer.py`` and
checks delegation, env-gated PLD proposals, trace emission, and unpatch
restoration.  With ``--ngram-path`` it performs the same reversible smoke on an
explicit installed vLLM proposer path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ngram-path", default="", help="Explicit ngram_proposer.py path.")
    parser.add_argument("--keep-patched", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.ngram_path:
        return smoke_path(Path(args.ngram_path).resolve(), keep_patched=args.keep_patched)
    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "ngram_proposer.py"
        fake.write_text(
            """
class NgramProposer:
    def __init__(self):
        self.num_speculative_tokens = 2

    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, *args, **kwargs):
        return [[101]]
""",
            encoding="utf-8",
        )
        return smoke_path(fake, keep_patched=False)


def smoke_path(path: Path, *, keep_patched: bool) -> int:
    backup_dir = path.parent / "vantage_smoke_backup"
    trace = path.parent / "vantage_smoke_trace.jsonl"
    patch_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "patch_installed_vllm_ngram_to_vantage.py"),
        "--ngram-path",
        str(path),
        "--backup-dir",
        str(backup_dir),
    ]
    subprocess.run(patch_cmd, check=True)
    try:
        module = load_module(path)
        proposer = module.NgramProposer()
        clear_patch_env()
        delegated = proposer.propose([[0]], [6], [[1, 2, 9, 8, 1, 2]])
        if delegated != [[101]]:
            raise RuntimeError(f"delegation failed with patch disabled: {delegated!r}")

        os.environ.update(
            {
                "VANTAGE_PLD_PATCH": "1",
                "VANTAGE_PLD_PATCH_MODE": "off",
            }
        )
        mode_off = proposer.propose([[0]], [6], [[1, 2, 9, 8, 1, 2]])
        if mode_off != [[101]]:
            raise RuntimeError(f"mode=off failed to delegate: {mode_off!r}")

        trace.unlink(missing_ok=True)
        os.environ.update(
            {
                "VANTAGE_PLD_PATCH_MODE": "passthrough_trace",
                "VANTAGE_PLD_TRACE_PATH": str(trace),
                "VANTAGE_PLD_TRACE_TOKENS": "1",
            }
        )
        passthrough = proposer.propose([[0]], [6], [[1, 2, 9, 8, 1, 2]])
        if passthrough != [[101]]:
            raise RuntimeError(f"passthrough_trace changed tokens: {passthrough!r}")
        passthrough_row = read_last_trace_row(trace)
        require_trace_fields(passthrough_row, mode="passthrough_trace", proposal=[101])

        trace.unlink(missing_ok=True)
        os.environ.update(
            {
                "VANTAGE_PLD_PATCH_MODE": "native_fixed_n",
                "VANTAGE_PLD_TRACE_PATH": str(trace),
                "VANTAGE_PLD_TRACE_TOKENS": "1",
            }
        )
        native = proposer.propose([[0]], [6], [[1, 2, 9, 8, 1, 2]])
        if native != [[101]]:
            raise RuntimeError(f"native_fixed_n changed native behavior: {native!r}")
        native_row = read_last_trace_row(trace)
        require_trace_fields(native_row, mode="native_fixed_n", proposal=[101])

        trace.unlink(missing_ok=True)
        clear_patch_env()
        os.environ.update(
            {
                "VANTAGE_PLD_PATCH": "1",
                "VANTAGE_PLD_MATCH_N": "2",
                "VANTAGE_PLD_NUM_SPECULATIVE_TOKENS": "2",
                "VANTAGE_PLD_TRACE_PATH": str(trace),
                "VANTAGE_PLD_TRACE_SAMPLE_RATE": "1",
                "VANTAGE_PLD_TRACE_TOKENS": "1",
            }
        )
        patched = proposer.propose([[0]], [6], [[1, 2, 9, 8, 1, 2]])
        if patched != [[9, 8]]:
            raise RuntimeError(f"patched PLD smoke failed: {patched!r}")
        row = read_last_trace_row(trace)
        require_trace_fields(row, mode="pld_python", proposal=[9, 8])
        if row.get("equivalence_label_candidate") != "capped_full_prefix_pld":
            raise RuntimeError("trace did not contain expected proposal_token_ids")

        trace.unlink(missing_ok=True)
        os.environ.update(
            {
                "VANTAGE_PLD_PATCH": "0",
                "VANTAGE_PLD_PATCH_MODE": "pld_optimized",
                "VANTAGE_PLD_MATCH_N": "2",
                "VANTAGE_PLD_NUM_SPECULATIVE_TOKENS": "2",
                "VANTAGE_PLD_TRACE_PATH": str(trace),
                "VANTAGE_PLD_TRACE_SAMPLE_RATE": "1",
                "VANTAGE_PLD_TRACE_TOKENS": "1",
            }
        )
        optimized = proposer.propose([[0]], [6], [[1, 2, 9, 8, 1, 2]])
        if optimized != [[9, 8]]:
            raise RuntimeError(f"pld_optimized fallback smoke failed: {optimized!r}")
        optimized_row = read_last_trace_row(trace)
        require_trace_fields(optimized_row, mode="pld_optimized", proposal=[9, 8])
        if optimized_row.get("optimized_pld_used") is not True:
            raise RuntimeError(f"pld_optimized did not use optimized path: {optimized_row!r}")
        if optimized_row.get("optimized_pld_fallback") is not False:
            raise RuntimeError(f"pld_optimized unexpectedly fell back: {optimized_row!r}")
    finally:
        clear_patch_env()
        if not keep_patched:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "unpatch_installed_vllm_ngram.py"),
                    "--ngram-path",
                    str(path),
                    "--backup-dir",
                    str(backup_dir),
                ],
                check=True,
            )
    print(json.dumps({"status": "ok", "ngram_path": str(path), "trace_path": str(trace)}))
    return 0


def clear_patch_env() -> None:
    for key in (
        "VANTAGE_PLD_PATCH",
        "VANTAGE_PLD_PATCH_MODE",
        "VANTAGE_PLD_MATCH_N",
        "VANTAGE_PLD_NUM_SPECULATIVE_TOKENS",
        "VANTAGE_PLD_TRACE_PATH",
        "VANTAGE_PLD_TRACE_SAMPLE_RATE",
        "VANTAGE_PLD_TRACE_TOKENS",
    ):
        os.environ.pop(key, None)


def read_last_trace_row(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise RuntimeError(f"trace file is empty: {path}")
    return rows[-1]


def require_trace_fields(row: dict, *, mode: str, proposal: list[int]) -> None:
    if row.get("mode") != mode:
        raise RuntimeError(f"expected trace mode {mode!r}, got {row.get('mode')!r}")
    if row.get("request_count") != 1:
        raise RuntimeError(f"expected request_count=1, got {row.get('request_count')!r}")
    if row.get("prefix_len") != 6:
        raise RuntimeError(f"expected prefix_len=6, got {row.get('prefix_len')!r}")
    if row.get("proposal_len") != len(proposal):
        raise RuntimeError(f"expected proposal_len={len(proposal)}, got {row.get('proposal_len')!r}")
    if row.get("proposal_token_ids") != proposal:
        raise RuntimeError(f"expected proposal_token_ids={proposal!r}, got {row.get('proposal_token_ids')!r}")
    if "elapsed_us" not in row:
        raise RuntimeError("trace row missing elapsed_us")
    if "hit" not in row or "miss" not in row:
        raise RuntimeError("trace row missing hit/miss fields")


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("vantage_smoke_ngram", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import patched module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())
