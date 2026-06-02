#!/usr/bin/env python3
"""Upload bulky VANTAGE generated data/artifacts to a Hugging Face dataset.

Public releases use VANTAGE-facing artifact paths. This helper stages the
selected folders before upload so maintainers can keep GitHub source-focused
and place bulky generated outputs in the companion dataset.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_REPO_ID = "faizancodes/vantage-artifacts"

UPLOAD_PATHS = [
    ("artifacts/vantage_transpld", "artifacts/vantage_transpld"),
    ("artifacts/vantage_viewbank", "artifacts/vantage_viewbank"),
    ("artifacts/vllm_results", "artifacts/vllm_results"),
    ("artifacts/vllm_tables", "artifacts/vllm_tables"),
    ("artifacts/vantage_residual", "artifacts/vantage_residual"),
    ("analysis", "analysis"),
    ("out", "out"),
    ("data/real_commits", "data/real_commits"),
    ("data/manifests", "data/manifests"),
    ("data/manifests_frozen_audit_raw", "data/manifests_frozen_audit_raw"),
    ("data/manifests_phase2", "data/manifests_phase2"),
    ("data/manifests_phase3", "data/manifests_phase3"),
    ("data/manifests_prompt_injection", "data/manifests_prompt_injection"),
    ("data/manifests_transpld_ext", "data/manifests_transpld_ext"),
    ("data/routers", "data/routers"),
]

IGNORE_PATTERNS = [
    "**/.DS_Store",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.pytest_cache/**",
    "**/*.pid",
]

TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".html",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".out",
    ".py",
    ".sh",
    ".stderr",
    ".stdout",
    ".svg",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

TEXT_REPLACEMENTS: tuple[tuple[str, str], ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        required=True,
        help="Path to the full original research/asts-spec tree containing generated data.",
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("VANTAGE_HF_DATASET", DEFAULT_REPO_ID),
        help="Destination Hugging Face dataset id.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset as private if it does not already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the upload plan without creating or uploading.",
    )
    parser.add_argument(
        "--staging-dir",
        default=None,
        help="Optional directory for the normalized upload staging tree.",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Keep the normalized staging tree after upload.",
    )
    parser.add_argument(
        "--no-normalize-paths",
        action="store_true",
        help="Keep source path components unchanged inside the staging tree.",
    )
    return parser.parse_args()


def _normalize_name(name: str) -> str:
    out = name
    for old, new in TEXT_REPLACEMENTS:
        out = out.replace(old, new)
    return out


def _is_ignored(path: Path) -> bool:
    parts = set(path.parts)
    if ".DS_Store" in parts or "__pycache__" in parts or ".pytest_cache" in parts:
        return True
    if path.suffix == ".pyc":
        return True
    if path.suffix == ".pid":
        return True
    return False


def _looks_textual(path: Path) -> bool:
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    return path.name in {"README", "LICENSE", "NOTICE"}


def _normalize_text(data: str) -> str:
    out = data
    for old, new in TEXT_REPLACEMENTS:
        out = out.replace(old, new)
    return out


def iter_existing_paths(source_root: Path, *, normalize: bool) -> list[tuple[str, str, Path]]:
    existing: list[tuple[str, str, Path]] = []
    seen_dest: set[str] = set()
    for source_rel, dest_rel in UPLOAD_PATHS:
        path = source_root / source_rel
        if path.exists():
            final_dest = dest_rel if normalize else source_rel
            if final_dest in seen_dest:
                # Prefer already-normalized source paths over legacy aliases when both exist.
                continue
            seen_dest.add(final_dest)
            existing.append((source_rel, final_dest, path))
    return existing


def stage_upload_tree(
    existing: list[tuple[str, str, Path]],
    *,
    staging_root: Path,
    source_root: Path,
    normalize: bool,
) -> dict[str, int]:
    stats = {
        "files": 0,
        "hardlinked_or_copied": 0,
        "text_rewritten": 0,
        "bytes": 0,
    }
    for source_rel, dest_rel, source_path in existing:
        for src in source_path.rglob("*"):
            if not src.is_file():
                continue
            rel_to_source = src.relative_to(source_path)
            if _is_ignored(rel_to_source):
                continue
            if normalize:
                dest_parts = [_normalize_name(part) for part in Path(dest_rel, rel_to_source).parts]
                rel_dest = Path(*dest_parts)
            else:
                rel_dest = Path(source_rel, rel_to_source)
            dst = staging_root / rel_dest
            dst.parent.mkdir(parents=True, exist_ok=True)
            size = src.stat().st_size
            stats["files"] += 1
            stats["bytes"] += size
            if normalize and _looks_textual(src):
                try:
                    text = src.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    text = src.read_text(encoding="utf-8", errors="replace")
                normalized = _normalize_text(text)
                dst.write_text(normalized, encoding="utf-8")
                stats["text_rewritten"] += int(normalized != text)
                continue
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            stats["hardlinked_or_copied"] += 1
    return stats


def write_dataset_card(staging_root: Path, repo_root: Path) -> None:
    template = repo_root / "docs" / "hf_dataset_card_template.md"
    if template.exists():
        shutil.copy2(template, staging_root / "README.md")


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    normalize = not args.no_normalize_paths
    if not source_root.exists():
        raise SystemExit(f"Missing source root: {source_root}")

    existing = iter_existing_paths(source_root, normalize=normalize)
    print(f"Source root: {source_root}")
    print(f"Destination dataset: {args.repo_id}")
    print(f"Normalize staged path components: {normalize}")
    print("Will upload:")
    for source_rel, dest_rel, path in existing:
        print(f"  - {source_rel} -> {dest_rel} ({path})")
    configured_sources = {source for source, _ in UPLOAD_PATHS}
    present_sources = {source_rel for source_rel, _, _ in existing}
    missing = sorted(configured_sources - present_sources)
    if missing:
        print("Missing/skipped:")
        for rel in missing:
            print(f"  - {rel}")

    if args.dry_run:
        print("Dry run only; no staging tree was created and no files were uploaded.")
        return 0

    if args.staging_dir:
        staging_root = Path(args.staging_dir).resolve()
        if staging_root.exists():
            shutil.rmtree(staging_root)
        staging_root.mkdir(parents=True)
        cleanup = False
    else:
        staging_root = Path(tempfile.mkdtemp(prefix="vantage_hf_upload_"))
        cleanup = not args.keep_staging
    print(f"Staging root: {staging_root}")

    stats = stage_upload_tree(
        existing,
        staging_root=staging_root,
        source_root=source_root,
        normalize=normalize,
    )
    write_dataset_card(staging_root, repo_root)
    print(
        "Staged "
        f"{stats['files']} files, {stats['bytes']} bytes, "
        f"{stats['text_rewritten']} text files rewritten."
    )

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)

    for _, dest_rel, _ in existing:
        folder_path = staging_root / dest_rel
        print(f"Uploading {dest_rel} ...")
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=str(folder_path),
            path_in_repo=dest_rel,
            ignore_patterns=IGNORE_PATTERNS,
            commit_message=f"Upload {dest_rel}",
        )
    card = staging_root / "README.md"
    if card.exists():
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="dataset",
            path_or_fileobj=str(card),
            path_in_repo="README.md",
            commit_message="Update dataset card",
        )
    print(f"Uploaded VANTAGE artifacts to https://huggingface.co/datasets/{args.repo_id}")
    if cleanup:
        shutil.rmtree(staging_root)
    else:
        print(f"Kept staging tree: {staging_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
