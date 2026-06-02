"""Diff/hunk-only edit generation applicators and metrics.

The applicators in this module are intentionally deterministic.  They either
produce one edited string or raise ``PatchError`` with a stable failure code.
They do not attempt model-side recovery or decoder-side PLD behavior.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable, Literal


PatchKind = Literal["unified_diff", "json_replacements", "search_replace"]


class PatchError(ValueError):
    """Patch parsing or application failed with a machine-readable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class UnifiedHunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class JsonReplacement:
    start_anchor: str
    end_anchor: str
    replacement: str


@dataclass(frozen=True)
class SearchReplaceHunk:
    search: str
    replace: str


@dataclass(frozen=True)
class ParsedPatch:
    kind: PatchKind
    hunks: tuple[UnifiedHunk | JsonReplacement | SearchReplaceHunk, ...]


_HUNK_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)
_FENCE_RE = re.compile(r"^\s*```(?:[A-Za-z0-9_+-]+)?\s*\n(?P<body>.*?)(?:\n```\s*)?$", re.DOTALL)
_SEARCH_START = "<<<<<<< SEARCH"
_SEARCH_MID = "======="
_SEARCH_END = ">>>>>>> REPLACE"


def strip_code_fence(text: str) -> str:
    match = _FENCE_RE.match(text.strip())
    return match.group("body") if match else text


def normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in strip_code_fence(text).strip().splitlines()).strip()


def detect_patch_format(text: str) -> PatchKind:
    body = strip_code_fence(text).lstrip()
    if body.startswith("{") or body.startswith("["):
        return "json_replacements"
    if _SEARCH_START in body or _SEARCH_END in body:
        return "search_replace"
    if (
        body.startswith("--- ")
        or body.startswith("diff --git")
        or "\n@@ " in body
        or body.startswith("@@ ")
    ):
        return "unified_diff"
    raise PatchError("unknown_format", "completion is not a supported diff/hunk format")


def parse_patch(text: str, *, patch_format: str | None = None) -> ParsedPatch:
    kind = _coerce_format(patch_format) if patch_format else detect_patch_format(text)
    body = strip_code_fence(text)
    if kind == "unified_diff":
        return ParsedPatch(kind, tuple(_parse_unified_diff(body)))
    if kind == "json_replacements":
        return ParsedPatch(kind, tuple(_parse_json_replacements(body)))
    if kind == "search_replace":
        return ParsedPatch(kind, tuple(_parse_search_replace(body)))
    raise PatchError("unknown_format", f"unsupported patch format: {patch_format}")


def apply_patch_text(source: str, text: str, *, patch_format: str | None = None) -> str:
    return apply_parsed_patch(source, parse_patch(text, patch_format=patch_format))


def apply_parsed_patch(source: str, patch: ParsedPatch) -> str:
    if patch.kind == "unified_diff":
        return _apply_unified_diff(source, _typed_hunks(patch.hunks, UnifiedHunk))
    if patch.kind == "json_replacements":
        return _apply_json_replacements(source, _typed_hunks(patch.hunks, JsonReplacement))
    if patch.kind == "search_replace":
        return _apply_search_replace(source, _typed_hunks(patch.hunks, SearchReplaceHunk))
    raise PatchError("unknown_format", f"unsupported patch kind: {patch.kind}")


def evaluate_completion(
    source: str,
    completion: str,
    *,
    expected: str | None = None,
    patch_format: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "parse_success": False,
        "apply_success": False,
        "failure_code": None,
        "failure_message": None,
        "patch_format": patch_format,
        "output_length": 0,
        "edit_distance": None,
        "exact_match": None,
    }
    try:
        parsed = parse_patch(completion, patch_format=patch_format)
        row["parse_success"] = True
        row["patch_format"] = parsed.kind
        output = apply_parsed_patch(source, parsed)
        row["apply_success"] = True
        row["output_length"] = len(output)
        if expected is not None:
            row["exact_match"] = normalize_text(output) == normalize_text(expected)
            row["edit_distance"] = edit_distance(output, expected)
    except PatchError as exc:
        row["failure_code"] = exc.code
        row["failure_message"] = str(exc)
    return row


def edit_distance(a: str, b: str) -> int:
    """Return a deterministic character edit distance proxy from diff opcodes."""
    distance = 0
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            distance += max(i2 - i1, j2 - j1)
        else:
            distance += (i2 - i1) + (j2 - j1)
    return distance


def _coerce_format(value: str | None) -> PatchKind:
    aliases = {
        "diff": "unified_diff",
        "unified": "unified_diff",
        "unified_diff": "unified_diff",
        "json": "json_replacements",
        "json_replacements": "json_replacements",
        "search_replace": "search_replace",
        "search/replace": "search_replace",
        "search-replace": "search_replace",
    }
    key = str(value or "").strip().lower()
    if key not in aliases:
        raise PatchError("unknown_format", f"unsupported patch format: {value}")
    return aliases[key]  # type: ignore[return-value]


def _parse_unified_diff(text: str) -> Iterable[UnifiedHunk]:
    lines = text.splitlines(keepends=True)
    hunks: list[UnifiedHunk] = []
    i = 0
    while i < len(lines):
        header = lines[i].rstrip("\n")
        match = _HUNK_RE.match(header)
        if not match:
            i += 1
            continue
        old_start = int(match.group("old_start"))
        old_count = int(match.group("old_count") or "1")
        new_start = int(match.group("new_start"))
        new_count = int(match.group("new_count") or "1")
        i += 1
        hunk_lines: list[tuple[str, str]] = []
        while i < len(lines):
            line = lines[i]
            if _HUNK_RE.match(line.rstrip("\n")):
                break
            if line.startswith(("--- ", "+++ ", "diff --git")):
                break
            if line.startswith("\\ No newline at end of file"):
                i += 1
                continue
            if not line:
                raise PatchError(
                    "malformed_unified_diff",
                    "empty diff line is missing an operation prefix",
                )
            op = line[0]
            if op not in {" ", "+", "-"}:
                raise PatchError("malformed_unified_diff", f"invalid diff line prefix: {op!r}")
            hunk_lines.append((op, line[1:]))
            i += 1
        old_seen = sum(1 for op, _ in hunk_lines if op in {" ", "-"})
        new_seen = sum(1 for op, _ in hunk_lines if op in {" ", "+"})
        if old_seen != old_count or new_seen != new_count:
            raise PatchError(
                "hunk_count_mismatch",
                "unified diff hunk line counts do not match header",
            )
        hunks.append(UnifiedHunk(old_start, old_count, new_start, new_count, tuple(hunk_lines)))
    if not hunks:
        raise PatchError("malformed_unified_diff", "unified diff contains no hunks")
    return hunks


def _parse_json_replacements(text: str) -> Iterable[JsonReplacement]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PatchError("malformed_json", f"invalid JSON patch: {exc}") from exc
    if isinstance(payload, dict) and "replacements" in payload:
        payload = payload["replacements"]
    elif isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not payload:
        raise PatchError("malformed_json", "JSON patch must contain one or more replacements")
    replacements: list[JsonReplacement] = []
    for i, item in enumerate(payload):
        if not isinstance(item, dict):
            raise PatchError("malformed_json", f"replacement {i} is not an object")
        missing = [k for k in ("start_anchor", "end_anchor", "replacement") if k not in item]
        if missing:
            raise PatchError("malformed_json", f"replacement {i} is missing {', '.join(missing)}")
        start_anchor = item["start_anchor"]
        end_anchor = item["end_anchor"]
        replacement = item["replacement"]
        if not all(isinstance(v, str) for v in (start_anchor, end_anchor, replacement)):
            raise PatchError("malformed_json", f"replacement {i} fields must be strings")
        if not start_anchor or not end_anchor:
            raise PatchError("malformed_json", f"replacement {i} anchors must be non-empty")
        replacements.append(JsonReplacement(start_anchor, end_anchor, replacement))
    return replacements


def _parse_search_replace(text: str) -> Iterable[SearchReplaceHunk]:
    lines = text.splitlines(keepends=True)
    hunks: list[SearchReplaceHunk] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() != _SEARCH_START:
            if lines[i].strip():
                raise PatchError(
                    "malformed_search_replace",
                    "unexpected text outside SEARCH/REPLACE hunk",
                )
            i += 1
            continue
        i += 1
        search: list[str] = []
        while i < len(lines) and lines[i].strip() != _SEARCH_MID:
            search.append(lines[i])
            i += 1
        if i >= len(lines):
            raise PatchError("malformed_search_replace", "SEARCH hunk is missing separator")
        i += 1
        replace: list[str] = []
        while i < len(lines) and lines[i].strip() != _SEARCH_END:
            replace.append(lines[i])
            i += 1
        if i >= len(lines):
            raise PatchError(
                "malformed_search_replace",
                "SEARCH hunk is missing REPLACE terminator",
            )
        i += 1
        search_text = "".join(search)
        if not search_text:
            raise PatchError("malformed_search_replace", "SEARCH body must be non-empty")
        hunks.append(SearchReplaceHunk(search_text, "".join(replace)))
    if not hunks:
        raise PatchError("malformed_search_replace", "patch contains no SEARCH/REPLACE hunks")
    return hunks


def _apply_unified_diff(source: str, hunks: tuple[UnifiedHunk, ...]) -> str:
    lines = source.splitlines(keepends=True)
    out: list[str] = []
    cursor = 0
    for hunk in hunks:
        start = hunk.old_start if hunk.old_count == 0 else hunk.old_start - 1
        if start < cursor or start > len(lines):
            raise PatchError("hunk_location_mismatch", "unified diff hunk location is invalid")
        expected = [content for op, content in hunk.lines if op in {" ", "-"}]
        actual = lines[start : start + len(expected)]
        if actual != expected:
            raise PatchError(
                "hunk_context_mismatch",
                "unified diff hunk context does not match source",
            )
        out.extend(lines[cursor:start])
        out.extend(content for op, content in hunk.lines if op in {" ", "+"})
        cursor = start + len(expected)
    out.extend(lines[cursor:])
    return "".join(out)


def _apply_json_replacements(source: str, replacements: tuple[JsonReplacement, ...]) -> str:
    text = source
    cursor = 0
    out: list[str] = []
    for repl in replacements:
        start = _find_unique(text, repl.start_anchor, cursor, "start_anchor")
        content_start = start + len(repl.start_anchor)
        end = _find_unique(text, repl.end_anchor, content_start, "end_anchor")
        out.append(text[cursor:content_start])
        out.append(repl.replacement)
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def _apply_search_replace(source: str, hunks: tuple[SearchReplaceHunk, ...]) -> str:
    text = source
    for hunk in hunks:
        count = text.count(hunk.search)
        if count == 0:
            raise PatchError("search_not_found", "SEARCH body was not found in source")
        if count > 1:
            raise PatchError("ambiguous_search", "SEARCH body matched more than once")
        text = text.replace(hunk.search, hunk.replace, 1)
    return text


def _find_unique(text: str, needle: str, start: int, label: str) -> int:
    pos = text.find(needle, start)
    if pos < 0:
        raise PatchError(f"{label}_not_found", f"{label} was not found")
    next_pos = text.find(needle, pos + len(needle))
    if next_pos >= 0:
        raise PatchError(f"ambiguous_{label}", f"{label} matched more than once")
    return pos


def _typed_hunks(hunks: tuple[Any, ...], cls: type[Any]) -> tuple[Any, ...]:
    if not all(isinstance(h, cls) for h in hunks):
        raise PatchError("internal_error", "parsed patch contained unexpected hunk type")
    return hunks
