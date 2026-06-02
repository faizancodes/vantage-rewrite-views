"""Build a manifest of bounded Python code-edit tasks from public commits.

The script intentionally starts from a curated candidate list instead of
scraping GitHub broadly.  Each output row contains only the commit message /
instruction plus the pre-commit reference function in the prompt; the
post-commit function is stored as ``deterministic_target`` for quality and
workload-characterization analyses.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import warnings
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=SyntaxWarning)


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*|"
    r"\d+(?:\.\d+)?|"
    r"==|!=|<=|>=|->|=>|\S"
)
_REWRITE_RE = re.compile(
    r"\b(?:rename|replace|change|migrate|swap)\s+"
    r"(?:`([^`\n]{1,120})`|'([^'\n]{1,120})'|\"([^\"\n]{1,120})\"|([A-Za-z_][A-Za-z0-9_\.]{1,120}))"
    r"\s+(?:with|to)\s+"
    r"(?:`([^`\n]{1,120})`|'([^'\n]{1,120})'|\"([^\"\n]{1,120})\"|([A-Za-z_][A-Za-z0-9_\.]{1,120}))",
    re.IGNORECASE,
)
_ARROW_RE = re.compile(
    r"(?:`([^`\n]{1,120})`|'([^'\n]{1,120})'|\"([^\"\n]{1,120})\"|([A-Za-z_][A-Za-z0-9_\.]{1,120}))"
    r"\s*(?:->|=>|→)\s*"
    r"(?:`([^`\n]{1,120})`|'([^'\n]{1,120})'|\"([^\"\n]{1,120})\"|([A-Za-z_][A-Za-z0-9_\.]{1,120}))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FunctionBlock:
    name: str
    start_line: int
    end_line: int
    text: str


def _run(cmd: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{result.stderr}")
    return result.stdout


def _repo_cache_name(repo: str) -> str:
    cleaned = repo.rstrip("/").replace("https://github.com/", "").replace("/", "__")
    return cleaned.replace(".git", "")


def _ensure_repo(row: dict[str, Any], cache_dir: Path) -> Path:
    if row.get("repo_path"):
        path = Path(str(row["repo_path"])).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    repo = str(row.get("repo") or "")
    if not repo:
        raise ValueError("candidate row needs repo or repo_path")
    url = repo if repo.startswith("http") else f"https://github.com/{repo}.git"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / _repo_cache_name(repo)
    if dst.exists():
        _run(["git", "fetch", "--all", "--quiet"], cwd=dst)
    else:
        _run(["git", "clone", "--quiet", url, str(dst)])
    return dst


def _commit_message(repo_dir: Path, sha: str) -> str:
    return _run(["git", "log", "-1", "--pretty=%B", sha], cwd=repo_dir).strip()


def _parent_sha(repo_dir: Path, sha: str) -> str:
    return _run(["git", "rev-parse", f"{sha}^"], cwd=repo_dir).strip()


def _changed_python_files(repo_dir: Path, parent: str, sha: str) -> list[str]:
    raw = _run(["git", "diff", "--name-only", parent, sha, "--", "*.py"], cwd=repo_dir)
    return [line.strip() for line in raw.splitlines() if line.strip()]


def _show_file(repo_dir: Path, rev: str, file_path: str) -> str:
    return _run(["git", "show", f"{rev}:{file_path}"], cwd=repo_dir)


def _changed_new_lines(repo_dir: Path, parent: str, sha: str, file_path: str) -> set[int]:
    raw = _run(["git", "diff", "--unified=0", parent, sha, "--", file_path], cwd=repo_dir)
    out: set[int] = set()
    for line in raw.splitlines():
        if not line.startswith("@@"):
            continue
        m = re.search(r"\+(\d+)(?:,(\d+))?", line)
        if not m:
            continue
        start = int(m.group(1))
        count = int(m.group(2) or "1")
        out.update(range(start, start + max(1, count)))
    return out


def _functions(source: str) -> list[FunctionBlock]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines()
    out: list[FunctionBlock] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = getattr(node, "end_lineno", None)
        if end is None:
            continue
        text = "\n".join(lines[node.lineno - 1 : end]).rstrip() + "\n"
        out.append(FunctionBlock(node.name, int(node.lineno), int(end), text))
    return sorted(out, key=lambda f: (f.start_line, f.end_line))


def _overlaps(lines: set[int], block: FunctionBlock) -> bool:
    return any(block.start_line <= line <= block.end_line for line in lines)


def _select_function_pair(
    parent_source: str,
    child_source: str,
    changed_lines: set[int],
    *,
    max_function_bytes: int = 80_000,
) -> tuple[FunctionBlock, FunctionBlock] | None:
    parent_blocks = [f for f in _functions(parent_source) if len(f.text.encode()) <= max_function_bytes]
    child_blocks = [
        f
        for f in _functions(child_source)
        if len(f.text.encode()) <= max_function_bytes and _overlaps(changed_lines, f)
    ]
    if len(child_blocks) != 1:
        return None
    child = child_blocks[0]
    same_name = [f for f in parent_blocks if f.name == child.name]
    if len(same_name) == 1:
        return same_name[0], child
    if not parent_blocks:
        return None
    # Avoid quadratic comparisons against obviously unrelated or very different
    # blocks in large files.  This path is only needed when a function was
    # renamed; same-name edits return above.
    plausible = [
        f
        for f in parent_blocks
        if 0.35 <= (len(f.text) / max(1, len(child.text))) <= 2.85
    ]
    if not plausible:
        return None
    parent = max(plausible, key=lambda f: SequenceMatcher(None, f.text, child.text).quick_ratio())
    if SequenceMatcher(None, parent.text, child.text).ratio() < 0.50:
        return None
    return parent, child


def _clean_term(value: str) -> str:
    return value.strip().strip("`'\"")


def _pair_from_match(match: re.Match[str]) -> tuple[str, str] | None:
    groups = [_clean_term(g) for g in match.groups() if g]
    if len(groups) != 2:
        return None
    old, new = groups
    if not old or not new or old == new:
        return None
    return old, new


def _rewrite_pairs_from_text(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for pattern in (_REWRITE_RE, _ARROW_RE):
        for match in pattern.finditer(text):
            pair = _pair_from_match(match)
            if pair is None:
                continue
            old, new = pair
            if old not in out:
                out[old] = new
    return out


def _coerce_pairs(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items() if str(k) and str(k) != str(v)}
    if isinstance(value, list):
        out: dict[str, str] = {}
        for item in value:
            if isinstance(item, dict):
                old = item.get("old") or item.get("from")
                new = item.get("new") or item.get("to")
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                old, new = item
            else:
                continue
            if old and new and str(old) != str(new):
                out[str(old)] = str(new)
        return out
    return {}


def _infer_diff_pair(parent_text: str, child_text: str, *, max_pairs: int = 4) -> dict[str, str]:
    old_terms = Counter(_IDENT_RE.findall(parent_text))
    new_terms = Counter(_IDENT_RE.findall(child_text))
    removed = [term for term, count in old_terms.items() if count > new_terms.get(term, 0)]
    added = [term for term, count in new_terms.items() if count > old_terms.get(term, 0)]
    candidates: list[tuple[float, str, str]] = []
    for old in removed:
        for new in added:
            if old == new:
                continue
            ratio = SequenceMatcher(None, old, new).ratio()
            common_suffix = len(old) > 2 and len(new) > 2 and old[-3:] == new[-3:]
            common_prefix = len(old) > 2 and len(new) > 2 and old[:3] == new[:3]
            if ratio >= 0.45 or common_prefix or common_suffix:
                support = min(old_terms[old] - new_terms.get(old, 0), new_terms[new] - old_terms.get(new, 0))
                candidates.append((support + ratio, old, new))
    out: dict[str, str] = {}
    used_new: set[str] = set()
    for _, old, new in sorted(candidates, reverse=True):
        if old in out or new in used_new:
            continue
        out[old] = new
        used_new.add(new)
        if len(out) >= max_pairs:
            break
    return out


def _map_has_evidence(before: str, after: str, pairs: dict[str, str]) -> bool:
    for old, new in pairs.items():
        if old in before and new in after and old != new:
            return True
    return False


def _family_from_pairs(pairs: dict[str, str], fallback: str = "real_commit") -> str:
    if any("." in old or "." in new for old, new in pairs.items()):
        return "real_field_migration"
    if any(old.lower() != new.lower() and (old.isidentifier() or new.isidentifier()) for old, new in pairs.items()):
        return "real_rename"
    return fallback


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _longest_common_run(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    best = 0
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def _changed_hunks(a: list[str], b: list[str]) -> int:
    hunks = 0
    in_hunk = False
    for tag, _, _, _, _ in SequenceMatcher(None, a, b).get_opcodes():
        changed = tag != "equal"
        if changed and not in_hunk:
            hunks += 1
        in_hunk = changed
    return hunks


def _stats(reference: str, target: str) -> dict[str, Any]:
    ref = _tokens(reference)
    tgt = _tokens(target)
    matcher = SequenceMatcher(None, ref, tgt)
    copied = sum(i2 - i1 for tag, i1, i2, _, _ in matcher.get_opcodes() if tag == "equal")
    edit_distance = max(len(ref), len(tgt)) - copied
    return {
        "copied_token_percentage": copied / len(tgt) if tgt else 0.0,
        "edit_distance_tokens": edit_distance,
        "longest_unchanged_span_tokens": _longest_common_run(ref, tgt),
        "changed_hunk_count": _changed_hunks(ref, tgt),
        "output_tokens": len(tgt),
    }


def _prompt(message: str, reference: str, pairs: dict[str, str]) -> str:
    pair_text = ", ".join(f"{old} -> {new}" for old, new in pairs.items())
    instruction = message.strip() or "Apply the commit edit to the function."
    if pair_text:
        instruction += f"\nExplicit rewrite map: {pair_text}."
    return (
        f"{instruction}\n\n"
        "Rewrite the pre-commit Python function below and output the complete edited function.\n"
        "```python\n"
        f"{reference.rstrip()}\n"
        "```\n\n"
        "Edited function:\n"
    )


def _target_leaked(prompt: str, target: str) -> bool:
    stripped_target = target.strip()
    if not stripped_target:
        return True
    return stripped_target in prompt


def build_rows(
    candidates: list[dict[str, Any]],
    *,
    cache_dir: Path,
    max_rows: int | None = None,
    audit_rows: list[dict[str, Any]] | None = None,
    require_map_evidence: bool = True,
    max_source_bytes: int = 500_000,
    max_function_bytes: int = 80_000,
    progress_every: int = 250,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    repo_cache: dict[str, Path] = {}
    for idx, cand in enumerate(candidates):
        if progress_every and idx and idx % progress_every == 0:
            print(f"[build] processed={idx} accepted={len(rows)}", flush=True)
        if max_rows is not None and len(rows) >= max_rows:
            break
        try:
            repo_key = str(cand.get("repo_path") or cand.get("repo") or "")
            if repo_key not in repo_cache:
                repo_cache[repo_key] = _ensure_repo(cand, cache_dir)
            repo_dir = repo_cache[repo_key]
            sha = str(cand["commit_sha"])
            parent = str(cand.get("parent_sha") or _parent_sha(repo_dir, sha))
            message = str(cand.get("instruction") or cand.get("commit_message") or _commit_message(repo_dir, sha))
            files = [str(cand["file_path"])] if cand.get("file_path") else _changed_python_files(repo_dir, parent, sha)
            accepted = False
            skip_reason = "no_bounded_python_function_with_rewrite"
            for file_path in files:
                try:
                    parent_source = _show_file(repo_dir, parent, file_path)
                    child_source = _show_file(repo_dir, sha, file_path)
                except RuntimeError:
                    continue
                if (
                    len(parent_source.encode()) > max_source_bytes
                    or len(child_source.encode()) > max_source_bytes
                ):
                    skip_reason = "source_too_large"
                    continue
                pair = _select_function_pair(
                    parent_source,
                    child_source,
                    _changed_new_lines(repo_dir, parent, sha, file_path),
                    max_function_bytes=max_function_bytes,
                )
                if pair is None:
                    continue
                before, after = pair
                if before.text == after.text:
                    continue
                try:
                    ast.parse(after.text)
                except SyntaxError:
                    continue
                rewrite_pairs = (
                    _coerce_pairs(cand.get("rewrite_pairs"))
                    or _rewrite_pairs_from_text(message)
                    or _infer_diff_pair(before.text, after.text)
                )
                if not rewrite_pairs:
                    skip_reason = "no_rewrite_pairs"
                    continue
                if require_map_evidence and not _map_has_evidence(before.text, after.text, rewrite_pairs):
                    skip_reason = "rewrite_pairs_not_evidenced_in_function"
                    continue
                prompt = _prompt(message, before.text, rewrite_pairs)
                if _target_leaked(prompt, after.text):
                    skip_reason = "target_leaked"
                    continue
                drift_family = _family_from_pairs(
                    rewrite_pairs,
                    fallback=str(cand.get("drift_family") or "real_commit"),
                )
                row = {
                    "task_id": f"real_commit_python/{len(rows):04d}",
                    "prompt": prompt,
                    "reference": before.text,
                    "deterministic_target": after.text,
                    "language": "real_commit_python",
                    "repo": cand.get("repo") or str(repo_dir),
                    "commit_sha": sha,
                    "parent_sha": parent,
                    "file_path": file_path,
                    "commit_url": cand.get("commit_url")
                    or (f"https://github.com/{cand.get('repo')}/commit/{sha}" if cand.get("repo") else ""),
                    "rewrite_pairs": rewrite_pairs,
                    "drift_family": drift_family,
                    "function_name_before": before.name,
                    "function_name_after": after.name,
                    "curation_source": cand.get("curation_source") or "github_commit_search",
                    "curation_note": cand.get("curation_note") or message.splitlines()[0][:200],
                }
                row.update(_stats(before.text, after.text))
                rows.append(row)
                accepted = True
                if audit_rows is not None:
                    audit_rows.append(
                        {
                            "accepted": True,
                            "candidate_index": idx,
                            "repo": cand.get("repo") or str(repo_dir),
                            "commit_sha": sha,
                            "file_path": file_path,
                            "rewrite_pairs": rewrite_pairs,
                            "drift_family": drift_family,
                            "task_id": row["task_id"],
                        }
                    )
                break
            if not accepted and audit_rows is not None:
                audit_rows.append(
                    {
                        "accepted": False,
                        "candidate_index": idx,
                        "repo": cand.get("repo") or str(repo_dir),
                        "commit_sha": sha,
                        "skip_reason": skip_reason,
                        "changed_python_files": files,
                        "commit_message": message.splitlines()[0][:240],
                    }
                )
        except Exception as exc:
            print(f"[skip] candidate {idx}: {exc}")
            if audit_rows is not None:
                audit_rows.append(
                    {
                        "accepted": False,
                        "candidate_index": idx,
                        "repo": cand.get("repo"),
                        "commit_sha": cand.get("commit_sha"),
                        "skip_reason": "exception",
                        "error": str(exc),
                    }
                )
            continue
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--repo-cache-dir", default=".cache/real_commits")
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--audit-jsonl", default="")
    p.add_argument(
        "--allow-unevidenced-map",
        action="store_true",
        help="Keep rows even if the selected function does not contain old->new evidence.",
    )
    p.add_argument("--max-source-bytes", type=int, default=500_000)
    p.add_argument("--max-function-bytes", type=int, default=80_000)
    p.add_argument("--progress-every", type=int, default=250)
    args = p.parse_args()
    candidates = [
        json.loads(line)
        for line in Path(args.candidates_jsonl).read_text().splitlines()
        if line.strip()
    ]
    audit_rows: list[dict[str, Any]] = []
    rows = build_rows(
        candidates,
        cache_dir=Path(args.repo_cache_dir),
        max_rows=args.max_rows or None,
        audit_rows=audit_rows,
        require_map_evidence=not args.allow_unevidenced_map,
        max_source_bytes=args.max_source_bytes,
        max_function_bytes=args.max_function_bytes,
        progress_every=args.progress_every,
    )
    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    if args.audit_jsonl:
        audit = Path(args.audit_jsonl)
        audit.parent.mkdir(parents=True, exist_ok=True)
        with audit.open("w") as f:
            for row in audit_rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
