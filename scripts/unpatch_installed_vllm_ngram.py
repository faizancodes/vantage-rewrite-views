#!/usr/bin/env python3
"""Restore vLLM's n-gram proposer after the VANTAGE no-build shim."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MARKER = "# VANTAGE_PLD_INSTALLED_NGRAM_PATCH_V1"
BACKUP_SUFFIX = ".vantage_original"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ngram-path",
        default="",
        help="Explicit installed vLLM ngram_proposer.py path. Defaults to import-based discovery.",
    )
    parser.add_argument(
        "--backup-dir",
        default=str(ROOT / "artifacts" / "vllm_patch_backups"),
        help="Directory containing original_sha256.txt, when available.",
    )
    parser.add_argument("--report-path", default="", help="Optional JSON report path.")
    parser.add_argument(
        "--keep-adjacent-backup",
        action="store_true",
        help="Restore the file but leave *.vantage_original beside it.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Locate and report without writing.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ngram_path = Path(args.ngram_path).expanduser() if args.ngram_path else locate_ngram_proposer()
    ngram_path = ngram_path.resolve()
    backup_path = ngram_path.with_name(ngram_path.name + BACKUP_SUFFIX)
    artifact_sha = Path(args.backup_dir).expanduser() / "original_sha256.txt"
    report: dict[str, Any] = {
        "status": "failed",
        "ngram_path": str(ngram_path),
        "adjacent_backup_path": str(backup_path),
        "artifact_sha_path": str(artifact_sha),
        "dry_run": bool(args.dry_run),
    }

    if not ngram_path.exists():
        raise SystemExit(f"ngram proposer path does not exist: {ngram_path}")
    current_text = ngram_path.read_text(encoding="utf-8")
    report["current_sha256"] = sha256_text(current_text)
    if MARKER not in current_text:
        report["status"] = "not_patched"
        write_report(args.report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if not backup_path.exists():
        raise SystemExit(f"adjacent backup not found: {backup_path}")

    backup_text = backup_path.read_text(encoding="utf-8")
    backup_sha = sha256_text(backup_text)
    report["backup_sha256"] = backup_sha
    if artifact_sha.exists():
        expected_sha = artifact_sha.read_text(encoding="utf-8").strip()
        report["expected_original_sha256"] = expected_sha
        if expected_sha and expected_sha != backup_sha:
            raise SystemExit(
                f"backup SHA mismatch: expected {expected_sha}, got {backup_sha}; refusing restore"
            )

    if args.dry_run:
        report["status"] = "dry_run_ok"
        write_report(args.report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    ngram_path.write_text(backup_text, encoding="utf-8")
    if not args.keep_adjacent_backup:
        backup_path.unlink()
    report["restored_sha256"] = backup_sha
    report["status"] = "restored"
    write_report(args.report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def locate_ngram_proposer() -> Path:
    try:
        import vllm  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on host env
        raise SystemExit(f"cannot import vllm to locate ngram proposer: {exc}") from exc

    root = Path(vllm.__file__).resolve().parent
    direct = root / "v1" / "spec_decode" / "ngram_proposer.py"
    if direct.exists():
        return direct
    matches = sorted(root.rglob("ngram_proposer.py"))
    if not matches:
        matches = sorted(root.rglob("ngram*proposer*.py"))
    if not matches:
        raise SystemExit(f"could not find ngram proposer under {root}")
    return matches[0]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_report(path: str, payload: dict[str, Any]) -> None:
    if not path:
        return
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
