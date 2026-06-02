#!/usr/bin/env python3
"""Mine refactor candidates from local git history without GitHub Search API.

This is the rate-limit-safe miner for the real-commit benchmark.  It clones or
fetches a fixed list of public repositories, searches `git log` locally for
rename/migrate/refactor messages, and emits one candidate per changed Python
file.  The downstream manifest builder remains the verifier: it accepts only
bounded function edits with parseable post-commit code, no target leakage, and
an evidenced rewrite map.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from build_real_commit_manifest import (
    _changed_python_files,
    _commit_message,
    _ensure_repo,
    _parent_sha,
    _repo_cache_name,
    _rewrite_pairs_from_text,
    _run,
)


DEFAULT_REPOS = [
    "django/django",
    "pandas-dev/pandas",
    "numpy/numpy",
    "scikit-learn/scikit-learn",
    "matplotlib/matplotlib",
    "pytest-dev/pytest",
    "sqlalchemy/sqlalchemy",
    "pallets/flask",
    "pallets/click",
    "pydantic/pydantic",
    "fastapi/fastapi",
    "psf/requests",
    "scrapy/scrapy",
    "celery/celery",
    "python/mypy",
    "psf/black",
    "apache/airflow",
]

DEFAULT_GREP_PATTERNS = [
    "rename",
    "renamed",
    "replace",
    "replaced",
    "migrate",
    "migrated",
    "refactor",
    "snake_case",
    "camelCase",
    "attribute",
    "field",
]

NOISE_RE = re.compile(
    r"\b(?:doc|docs|documentation|readme|typo|comment|comments|changelog|"
    r"release notes?|translation|locale|vendor|submodule)\b",
    re.IGNORECASE,
)


def _git_log_candidates(repo_dir: Path, *, max_commits: int) -> list[tuple[str, str]]:
    grep_args: list[str] = []
    for pattern in DEFAULT_GREP_PATTERNS:
        grep_args.extend(["--grep", pattern])
    cmd = [
        "git",
        "log",
        "--all",
        "--no-merges",
        "--regexp-ignore-case",
        *grep_args,
        "--pretty=format:%H%x00%s",
    ]
    raw = _run(cmd, cwd=repo_dir)
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        if "\x00" not in line:
            continue
        sha, subject = line.split("\x00", 1)
        if sha in seen:
            continue
        seen.add(sha)
        if NOISE_RE.search(subject):
            continue
        out.append((sha, subject.strip()))
        if max_commits and len(out) >= max_commits:
            break
    return out


def _classify_from_message(message: str, pairs: dict[str, str]) -> str:
    lower = message.lower()
    if "field" in lower or "attribute" in lower or any("." in k or "." in v for k, v in pairs.items()):
        return "real_field_migration"
    return "real_rename"


def _commit_url(repo: str, sha: str) -> str:
    return f"https://github.com/{repo}/commit/{sha}"


def mine_repo(
    repo: str,
    *,
    cache_dir: Path,
    max_commits_per_repo: int,
    max_files_per_commit: int,
) -> list[dict[str, Any]]:
    repo_dir = _ensure_repo({"repo": repo}, cache_dir)
    rows: list[dict[str, Any]] = []
    for sha, subject in _git_log_candidates(repo_dir, max_commits=max_commits_per_repo):
        try:
            parent = _parent_sha(repo_dir, sha)
            message = _commit_message(repo_dir, sha)
            files = _changed_python_files(repo_dir, parent, sha)
        except Exception:
            continue
        if not files or len(files) > max_files_per_commit:
            continue
        pairs = _rewrite_pairs_from_text(message)
        for file_path in files:
            rows.append(
                {
                    "repo": repo,
                    "commit_sha": sha,
                    "parent_sha": parent,
                    "file_path": file_path,
                    "commit_message": message,
                    "commit_url": _commit_url(repo, sha),
                    "rewrite_pairs": pairs,
                    "drift_family": _classify_from_message(message, pairs),
                    "curation_source": "local_git_log",
                    "curation_note": subject[:220],
                }
            )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repos", default=",".join(DEFAULT_REPOS))
    p.add_argument("--repo-cache-dir", default=".cache/real_commits")
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--max-commits-per-repo", type=int, default=1500)
    p.add_argument("--max-files-per-commit", type=int, default=8)
    p.add_argument(
        "--max-candidates-per-repo",
        type=int,
        default=0,
        help="Optional cap after file expansion. This keeps large repos from dominating.",
    )
    p.add_argument("--max-candidates", type=int, default=0)
    args = p.parse_args()

    repos = [r.strip() for r in args.repos.split(",") if r.strip()]
    cache_dir = Path(args.repo_cache_dir)
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for repo in repos:
        print(f"[mine] {repo}", flush=True)
        repo_count = 0
        for row in mine_repo(
            repo,
            cache_dir=cache_dir,
            max_commits_per_repo=args.max_commits_per_repo,
            max_files_per_commit=args.max_files_per_commit,
        ):
            key = (str(row["repo"]), str(row["commit_sha"]), str(row["file_path"]))
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
            repo_count += 1
            if args.max_candidates_per_repo and repo_count >= args.max_candidates_per_repo:
                break

    if args.max_candidates and len(out) > args.max_candidates:
        out = out[: args.max_candidates]

    path = Path(args.output_jsonl)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in out:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {len(out)} candidates to {path}", flush=True)


if __name__ == "__main__":
    main()
