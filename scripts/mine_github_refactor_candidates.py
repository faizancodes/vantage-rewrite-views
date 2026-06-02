"""Mine a small public GitHub commit-candidate list for real-edit manifests.

This is intentionally only a candidate miner.  The manifest builder performs
the actual safety checks: changed Python function extraction, target-leak
checks, rewrite-map extraction, syntax validation, and workload statistics.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any


DEFAULT_QUERIES = [
    '"rename" "to" "python"',
    '"replace" "with" "python"',
    '"rename variable" "python"',
    '"rename field" "python"',
    '"field rename" "python"',
    '"api migration" "python"',
    '"migrate" "to" "python"',
    '"snake_case" "python"',
]

DEFAULT_REPOS = [
    "django/django",
    "pandas-dev/pandas",
    "numpy/numpy",
    "scikit-learn/scikit-learn",
    "matplotlib/matplotlib",
    "pallets/flask",
    "pallets/click",
    "pytest-dev/pytest",
    "sqlalchemy/sqlalchemy",
    "ansible/ansible",
]

DEFAULT_REPO_QUERY_TEMPLATES = [
    'repo:{repo} "rename" "to"',
    'repo:{repo} "rename" "field"',
    'repo:{repo} "rename" "variable"',
    'repo:{repo} "replace" "with"',
    'repo:{repo} "migrate" "to"',
    'repo:{repo} "refactor" "rename"',
    'repo:{repo} "snake_case"',
    'repo:{repo} "camelCase"',
]


def _load_env_value(path: Path, key: str) -> str:
    if not path.exists():
        return ""
    for line in path.read_text(errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        if name == key:
            return value.strip().strip("'\"")
    return ""


def _expand_queries(raw_queries: str, raw_repos: str) -> list[str]:
    queries = [q.strip() for q in raw_queries.split(";;") if q.strip()]
    repos = [r.strip() for r in raw_repos.split(",") if r.strip()]
    if repos:
        for repo in repos:
            for template in DEFAULT_REPO_QUERY_TEMPLATES:
                queries.append(template.format(repo=repo))
    return queries or DEFAULT_QUERIES


def _request_json(url: str, *, token: str = "") -> dict[str, Any]:
    headers = {
        "Accept": "application/vnd.github.cloak-preview+json",
        "User-Agent": "vantage-real-commit-miner",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API request failed: HTTP {exc.code}: {detail[:500]}") from exc


def _search_commits(query: str, *, page: int, per_page: int, token: str = "") -> dict[str, Any]:
    encoded = urllib.parse.urlencode({"q": query, "page": page, "per_page": per_page})
    return _request_json(f"https://api.github.com/search/commits?{encoded}", token=token)


def mine(
    *,
    queries: list[str],
    pages: int,
    per_page: int,
    sleep_s: float,
    token: str,
    max_candidates: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for query in queries:
        for page in range(1, pages + 1):
            try:
                data = _search_commits(query, page=page, per_page=per_page, token=token)
            except RuntimeError as exc:
                print(f"[skip query] {query!r} page {page}: {exc}")
                break
            for item in data.get("items", []):
                repo = (item.get("repository") or {}).get("full_name")
                sha = item.get("sha")
                message = ((item.get("commit") or {}).get("message") or "").strip()
                if not repo or not sha or not message:
                    continue
                key = (str(repo), str(sha))
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    {
                        "repo": repo,
                        "commit_sha": sha,
                        "commit_message": message,
                        "commit_url": item.get("html_url") or f"https://github.com/{repo}/commit/{sha}",
                        "query": query,
                    }
                )
                if max_candidates and len(out) >= max_candidates:
                    return out
            if sleep_s > 0:
                time.sleep(sleep_s)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--queries", default="")
    p.add_argument(
        "--repos",
        default=",".join(DEFAULT_REPOS),
        help=(
            "Comma-separated repo list to search with repo-scoped refactor "
            "queries. Use --repos '' to disable repo templates."
        ),
    )
    p.add_argument("--pages", type=int, default=1)
    p.add_argument("--per-page", type=int, default=50)
    p.add_argument("--sleep-s", type=float, default=6.5)
    p.add_argument("--github-token", default="")
    p.add_argument("--github-token-env-file", default="")
    p.add_argument("--github-token-env-key", default="GITHUB_TOKEN")
    p.add_argument("--max-candidates", type=int, default=200)
    args = p.parse_args()

    token = args.github_token
    if not token and args.github_token_env_file:
        token = _load_env_value(Path(args.github_token_env_file), args.github_token_env_key)
    queries = _expand_queries(args.queries, args.repos)
    rows = mine(
        queries=queries,
        pages=args.pages,
        per_page=args.per_page,
        sleep_s=args.sleep_s,
        token=token,
        max_candidates=args.max_candidates,
    )
    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} candidates to {out}")


if __name__ == "__main__":
    main()
