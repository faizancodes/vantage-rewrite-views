#!/usr/bin/env python3
"""Install an env-gated VANTAGE PLD shim over vLLM's n-gram proposer.

The patch is intentionally no-build and reversible.  It replaces the installed
``vllm/v1/spec_decode/ngram_proposer.py`` with a small wrapper that imports the
original implementation from a sibling backup file.  With
``VANTAGE_PLD_PATCH`` unset, the wrapper delegates to the original
``NgramProposer``.  With ``VANTAGE_PLD_PATCH=1`` and no explicit
``VANTAGE_PLD_PATCH_MODE``, it uses a full-prefix ``w128_n10`` prompt-lookup
rule while still going through vLLM's native ``method="ngram"`` route.

``VANTAGE_PLD_PATCH_MODE`` can be set to:

- ``off``: delegate to the original implementation.
- ``passthrough_trace``: delegate to the original implementation and trace
  timing/proposal rows.
- ``native_fixed_n``: delegate to native n-gram behavior and trace
  timing/proposal rows.
- ``pld_python``: use the Python full-prefix PLD shim.
- ``pld_optimized``: try ``vantage_vllm.optimized_pld`` and fall back to
  ``pld_python`` with a trace flag when unavailable.

This is not source-boundary-aware PLD: vLLM 0.20.2 does not pass source/gold
boundary metadata through the n-gram proposer path.  Runs from this shim must
therefore be labeled ``capped_full_prefix_pld`` or
``capped_full_prefix_pld_token_traced`` when token traces prove the proposal
rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from textwrap import dedent
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
        help="Directory for an immutable artifact backup and SHA files.",
    )
    parser.add_argument("--report-path", default="", help="Optional JSON report path.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing adjacent backup.")
    parser.add_argument("--dry-run", action="store_true", help="Locate and report without writing.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ngram_path = Path(args.ngram_path).expanduser() if args.ngram_path else locate_ngram_proposer()
    ngram_path = ngram_path.resolve()
    report: dict[str, Any] = {
        "status": "failed",
        "ngram_path": str(ngram_path),
        "backup_suffix": BACKUP_SUFFIX,
        "marker": MARKER,
        "dry_run": bool(args.dry_run),
    }

    if not ngram_path.exists():
        raise SystemExit(f"ngram proposer path does not exist: {ngram_path}")
    original_text = ngram_path.read_text(encoding="utf-8")
    if MARKER in original_text:
        report["status"] = "already_patched"
        report["current_sha256"] = sha256_text(original_text)
        write_report(args.report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    original_sha = sha256_text(original_text)
    backup_path = ngram_path.with_name(ngram_path.name + BACKUP_SUFFIX)
    artifact_dir = Path(args.backup_dir).expanduser()
    artifact_backup = artifact_dir / "ngram_proposer_original.py"
    artifact_sha = artifact_dir / "original_sha256.txt"
    report.update(
        {
            "original_sha256": original_sha,
            "adjacent_backup_path": str(backup_path),
            "artifact_backup_path": str(artifact_backup),
            "artifact_sha_path": str(artifact_sha),
        }
    )
    if args.dry_run:
        report["status"] = "dry_run_ok"
        write_report(args.report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if backup_path.exists() and not args.force:
        raise SystemExit(
            f"adjacent backup already exists: {backup_path}; pass --force only if it is safe to overwrite"
        )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    backup_path.write_text(original_text, encoding="utf-8")
    artifact_backup.write_text(original_text, encoding="utf-8")
    artifact_sha.write_text(original_sha + "\n", encoding="utf-8")

    wrapper = build_wrapper_source(backup_path.name, original_sha)
    ngram_path.write_text(wrapper, encoding="utf-8")
    report["patched_sha256"] = sha256_text(wrapper)
    report["status"] = "patched"
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


def build_wrapper_source(backup_name: str, original_sha: str) -> str:
    return dedent(
        f'''\
        {MARKER}
        # Original SHA256: {original_sha}
        from __future__ import annotations

        import hashlib as _hashlib
        import importlib as _importlib
        import importlib.machinery as _importlib_machinery
        import importlib.util as _importlib_util
        import json as _json
        import os as _os
        import pathlib as _pathlib
        import random as _random
        import time as _time

        _ORIGINAL_PATH = _pathlib.Path(__file__).with_name({backup_name!r})
        _LOADER = _importlib_machinery.SourceFileLoader(
            "vllm.v1.spec_decode.ngram_proposer_vantage_original",
            str(_ORIGINAL_PATH),
        )
        _SPEC = _importlib_util.spec_from_loader(_LOADER.name, _LOADER)
        if _SPEC is None or _SPEC.loader is None:
            raise ImportError(f"cannot load original vLLM ngram proposer from {{_ORIGINAL_PATH}}")
        _ORIGINAL_MODULE = _importlib_util.module_from_spec(_SPEC)
        _SPEC.loader.exec_module(_ORIGINAL_MODULE)

        for _name in dir(_ORIGINAL_MODULE):
            if _name.startswith("__"):
                continue
            globals()[_name] = getattr(_ORIGINAL_MODULE, _name)

        _OriginalNgramProposer = _ORIGINAL_MODULE.NgramProposer
        _VALID_MODES = ("off", "passthrough_trace", "native_fixed_n", "pld_python", "pld_optimized")
        _OPTIMIZED_MODULE = None
        _OPTIMIZED_IMPORT_ERROR = None
        _OPTIMIZED_WARMED = False
        _OPTIMIZED_WARMUP_ERROR = None


        class NgramProposer(_OriginalNgramProposer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if _patch_mode() == "pld_optimized":
                    _warm_optimized_pld(self)

            def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, *args, **kwargs):
                mode = _patch_mode()
                if mode == "off":
                    return _delegate_original(
                        self,
                        sampled_token_ids,
                        num_tokens_no_spec,
                        token_ids_cpu,
                        args,
                        kwargs,
                    )
                if mode in ("passthrough_trace", "native_fixed_n"):
                    return _delegate_original_with_trace(
                        self,
                        mode,
                        sampled_token_ids,
                        num_tokens_no_spec,
                        token_ids_cpu,
                        args,
                        kwargs,
                    )
                if mode == "pld_optimized":
                    return _vantage_pld_optimized_or_python(
                        self,
                        sampled_token_ids,
                        num_tokens_no_spec,
                        token_ids_cpu,
                        args,
                        kwargs,
                    )
                return _vantage_pld_propose(
                    self,
                    mode,
                    sampled_token_ids,
                    num_tokens_no_spec,
                    token_ids_cpu,
                    args,
                    kwargs,
                    optimized_fallback=None,
                )


        def _patch_mode():
            mode = _os.environ.get("VANTAGE_PLD_PATCH_MODE")
            if mode not in (None, ""):
                normalized = str(mode).strip().lower()
                if normalized not in _VALID_MODES:
                    raise ValueError(
                        "VANTAGE_PLD_PATCH_MODE must be one of " + ",".join(_VALID_MODES)
                    )
                return normalized
            if _env_bool("VANTAGE_PLD_PATCH", False):
                return "pld_python"
            return "off"


        def _delegate_original(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, args, kwargs):
            return _OriginalNgramProposer.propose(
                self,
                sampled_token_ids,
                num_tokens_no_spec,
                token_ids_cpu,
                *args,
                **kwargs,
            )


        def _delegate_original_with_trace(
            self,
            mode,
            sampled_token_ids,
            num_tokens_no_spec,
            token_ids_cpu,
            args,
            kwargs,
        ):
            t0 = _time.perf_counter()
            drafts = _delegate_original(
                self,
                sampled_token_ids,
                num_tokens_no_spec,
                token_ids_cpu,
                args,
                kwargs,
            )
            elapsed_us = (_time.perf_counter() - t0) * 1_000_000.0
            _trace_existing_drafts(
                self,
                mode,
                sampled_token_ids,
                num_tokens_no_spec,
                token_ids_cpu,
                drafts,
                args,
                kwargs,
                elapsed_us,
            )
            return drafts


        def _vantage_pld_optimized_or_python(
            self,
            sampled_token_ids,
            num_tokens_no_spec,
            token_ids_cpu,
            args,
            kwargs,
        ):
            optimized_module, import_error = _optimized_module()
            fallback_reason = None
            if optimized_module is None:
                fallback_reason = f"import_failed: {{import_error}}"
            return _vantage_pld_propose(
                self,
                "pld_optimized",
                sampled_token_ids,
                num_tokens_no_spec,
                token_ids_cpu,
                args,
                kwargs,
                optimized_module=optimized_module,
                optimized_fallback=fallback_reason,
            )


        def _vantage_pld_propose(
            self,
            mode,
            sampled_token_ids,
            num_tokens_no_spec,
            token_ids_cpu,
            args,
            kwargs,
            optimized_module=None,
            optimized_fallback=None,
        ):
            match_n = _env_int("VANTAGE_PLD_MATCH_N", 10)
            max_draft_len = _env_int("VANTAGE_PLD_MAX_DRAFT_LEN", 128)
            tie_break = _os.environ.get("VANTAGE_PLD_TIE_BREAK", "latest")
            strict = _env_bool("VANTAGE_PLD_PATCH_STRICT", True)
            cap = _infer_cap(self)
            if cap is None:
                cap = _env_optional_int("VANTAGE_PLD_NUM_SPECULATIVE_TOKENS")
            if cap is None:
                cap = _env_optional_int("VANTAGE_PLD_CAP")
            if cap is None and strict:
                raise RuntimeError(
                    "VANTAGE_PLD_PATCH_STRICT=1 but num_speculative_tokens could not be inferred; "
                    "set VANTAGE_PLD_NUM_SPECULATIVE_TOKENS"
                )

            drafts = []
            trace_path, trace_tokens, sample_rate = _trace_config(mode)
            valid_rows = _valid_rows_from_args_kwargs(args, kwargs)
            row_count = len(sampled_token_ids) if sampled_token_ids is not None else len(num_tokens_no_spec)

            for request_i in range(row_count):
                t0 = _time.perf_counter()
                row_i = int(valid_rows[request_i]) if valid_rows is not None and request_i < len(valid_rows) else request_i
                prefix_len = int(_index_value(num_tokens_no_spec, row_i))
                prefix = _row_prefix(token_ids_cpu, row_i, prefix_len)
                result = None
                fallback_reason = optimized_fallback
                if optimized_module is not None:
                    result, fallback_reason, optimized_lookup_succeeded = _find_optimized_pld(
                        optimized_module,
                        prefix,
                        match_n,
                        max_draft_len,
                        cap,
                        tie_break,
                    )
                else:
                    optimized_lookup_succeeded = False
                if result is None and (optimized_module is None or fallback_reason is not None):
                    result = _find_full_prefix_pld(prefix, match_n, max_draft_len, cap, tie_break)
                proposal = result["tokens"] if result is not None else []
                drafts.append(proposal)
                elapsed_us = (_time.perf_counter() - t0) * 1_000_000.0
                if trace_path and (sample_rate >= 1.0 or (sample_rate > 0.0 and _random.random() < sample_rate)):
                    _append_trace(
                        trace_path,
                        _trace_row(
                            mode=mode,
                            request_i=request_i,
                            row_i=row_i,
                            request_count=row_count,
                            prefix_len=prefix_len,
                            match_n=match_n,
                            max_draft_len=max_draft_len,
                            cap=cap,
                            result=result,
                            proposal=proposal,
                            elapsed_us=elapsed_us,
                            trace_tokens=trace_tokens,
                            prefix=prefix,
                            optimized_fallback=fallback_reason,
                        ),
                    )
            return drafts


        def _optimized_module():
            global _OPTIMIZED_MODULE, _OPTIMIZED_IMPORT_ERROR
            if _OPTIMIZED_MODULE is not None or _OPTIMIZED_IMPORT_ERROR is not None:
                return _OPTIMIZED_MODULE, _OPTIMIZED_IMPORT_ERROR
            try:
                _OPTIMIZED_MODULE = _importlib.import_module("vantage_vllm.optimized_pld")
            except Exception as exc:
                _OPTIMIZED_IMPORT_ERROR = f"{{type(exc).__name__}}: {{exc}}"
            return _OPTIMIZED_MODULE, _OPTIMIZED_IMPORT_ERROR


        def _warm_optimized_pld(self):
            global _OPTIMIZED_WARMED, _OPTIMIZED_WARMUP_ERROR
            if _OPTIMIZED_WARMED or _OPTIMIZED_WARMUP_ERROR is not None:
                return
            module, import_error = _optimized_module()
            if module is None:
                _OPTIMIZED_WARMUP_ERROR = f"import_failed: {{import_error}}"
                return
            try:
                match_n = _env_int("VANTAGE_PLD_MATCH_N", 10)
                max_draft_len = _env_int("VANTAGE_PLD_MAX_DRAFT_LEN", 128)
                tie_break = _os.environ.get("VANTAGE_PLD_TIE_BREAK", "latest")
                cap = _infer_cap(self)
                if cap is None:
                    cap = _env_optional_int("VANTAGE_PLD_NUM_SPECULATIVE_TOKENS")
                if cap is None:
                    cap = _env_optional_int("VANTAGE_PLD_CAP")
                if cap is None:
                    cap = 16
                key = list(range(1, match_n + 1))
                middle = list(range(1000, 1000 + max(match_n + int(cap), match_n + 1)))
                prefix = key + middle + key
                _find_optimized_pld(module, prefix, match_n, max_draft_len, cap, tie_break)
                _OPTIMIZED_WARMED = True
            except Exception as exc:
                _OPTIMIZED_WARMUP_ERROR = f"{{type(exc).__name__}}: {{exc}}"


        def _find_optimized_pld(module, prefix, match_n, max_draft_len, cap, tie_break):
            for name in (
                "find_full_prefix_pld_proposal",
                "find_full_prefix_pld",
                "find_pld_proposal",
                "propose_full_prefix",
                "propose",
            ):
                fn = getattr(module, name, None)
                if not callable(fn):
                    continue
                try:
                    raw = fn(
                        prefix,
                        match_n=match_n,
                        max_draft_len=max_draft_len,
                        cap=cap,
                        tie_break=tie_break,
                        prefer_numba=_env_bool("VANTAGE_PLD_NUMBA", True),
                    )
                except TypeError:
                    try:
                        raw = fn(prefix, match_n, max_draft_len, cap, tie_break)
                    except Exception as exc:
                        return None, f"call_failed: {{type(exc).__name__}}: {{exc}}", False
                except Exception as exc:
                    return None, f"call_failed: {{type(exc).__name__}}: {{exc}}", False
                normalized = _normalize_pld_result(raw, match_n, cap)
                if normalized is not None:
                    normalized["optimized_pld_used"] = True
                    return normalized, None, True
                return None, None, True
            return None, "unsupported_optimized_api", False


        def _normalize_pld_result(raw, match_n, cap):
            if raw is None:
                return None
            if isinstance(raw, dict):
                tokens = raw.get("tokens")
                if tokens is None:
                    tokens = raw.get("proposal_token_ids")
                if tokens is None:
                    return None
                return {{
                    "tokens": [int(token) for token in tokens],
                    "match_pos": raw.get("match_pos", raw.get("source_start")),
                    "source_start": raw.get("source_start", raw.get("proposal_source_start_token")),
                    "source_end": raw.get("source_end", raw.get("proposal_source_end_token")),
                    "follow_start": raw.get("follow_start", raw.get("proposal_follow_start_token")),
                    "follow_end": raw.get("follow_end", raw.get("proposal_follow_end_token")),
                    "query_start": raw.get("query_start", raw.get("proposal_query_start_token")),
                    "query_end": raw.get("query_end", raw.get("proposal_query_end_token")),
                    "uncapped_len": raw.get("uncapped_len", len(tokens)),
                    "proposal_match_len": raw.get("proposal_match_len", match_n),
                    "proposal_cap": raw.get("proposal_cap", cap),
                }}
            tokens = getattr(raw, "tokens", None)
            if tokens is None and isinstance(raw, (list, tuple)):
                tokens = raw
            if tokens is None:
                return None
            return {{
                "tokens": [int(token) for token in tokens],
                "match_pos": getattr(raw, "source_start", None),
                "source_start": getattr(raw, "source_start", None),
                "source_end": getattr(raw, "source_end", None),
                "follow_start": getattr(raw, "follow_start", None),
                "follow_end": getattr(raw, "follow_end", None),
                "query_start": getattr(raw, "query_start", None),
                "query_end": getattr(raw, "query_end", None),
                "uncapped_len": len(tokens),
                "proposal_match_len": getattr(raw, "match_n", match_n),
                "proposal_cap": getattr(raw, "cap", cap),
            }}


        def _find_full_prefix_pld(prefix, match_n, max_draft_len, cap, tie_break):
            if match_n <= 0:
                raise ValueError("VANTAGE_PLD_MATCH_N must be positive")
            if max_draft_len < 0:
                raise ValueError("VANTAGE_PLD_MAX_DRAFT_LEN must be non-negative")
            if cap is not None and cap < 0:
                raise ValueError("proposal cap must be non-negative")
            if tie_break not in ("latest", "earliest"):
                raise ValueError("VANTAGE_PLD_TIE_BREAK must be latest or earliest")
            prefix = [int(token) for token in prefix]
            prefix_len = len(prefix)
            if prefix_len < match_n or max_draft_len <= 0 or cap == 0:
                return None
            query_start = prefix_len - match_n
            query_end = prefix_len
            needle = prefix[query_start:query_end]
            starts = range(0, query_start - match_n + 1)
            if tie_break == "latest":
                starts = range(query_start - match_n, -1, -1)
            effective_max = min(max_draft_len, cap) if cap is not None else max_draft_len
            for source_start in starts:
                source_end = source_start + match_n
                if prefix[source_start:source_end] != needle:
                    continue
                follow_start = source_end
                raw_follow_end = min(follow_start + max_draft_len, query_start)
                follow_end = min(raw_follow_end, follow_start + effective_max)
                if follow_start >= follow_end:
                    continue
                tokens = prefix[follow_start:follow_end]
                return {{
                    "tokens": tokens,
                    "match_pos": source_start,
                    "source_start": source_start,
                    "source_end": source_end,
                    "follow_start": follow_start,
                    "follow_end": follow_end,
                    "query_start": query_start,
                    "query_end": query_end,
                    "uncapped_len": raw_follow_end - follow_start,
                }}
            return None


        def _trace_row(
            *,
            mode,
            request_i,
            row_i,
            request_count,
            prefix_len,
            match_n,
            max_draft_len,
            cap,
            result,
            proposal,
            elapsed_us,
            trace_tokens,
            prefix,
            optimized_fallback=None,
        ):
            token_blob = ",".join(str(int(token)) for token in proposal).encode("utf-8")
            hit = result is not None and len(proposal) > 0
            row = {{
                "mode": mode,
                "run_id": _os.environ.get("VANTAGE_PLD_RUN_ID"),
                "request_count": request_count,
                "request_index": request_i,
                "row_index": row_i,
                "prefix_len": prefix_len,
                "match_n": match_n,
                "max_draft_len": max_draft_len,
                "cap": cap,
                "proposal_cap": cap,
                "num_speculative_tokens_cap": cap,
                "match_found": hit,
                "hit": hit,
                "miss": not hit,
                "match_pos": None if result is None else result["match_pos"],
                "proposal_len": len(proposal),
                "proposal_tokens": len(proposal),
                "proposal_token_hash": _hashlib.sha256(token_blob).hexdigest(),
                "tie_break": _os.environ.get("VANTAGE_PLD_TIE_BREAK", "latest"),
                "elapsed_us": elapsed_us,
                "equivalence_label_candidate": (
                    "full_prefix_pld" if cap is None or cap >= max_draft_len else "capped_full_prefix_pld"
                ),
                "proposal_match_len": (
                    None if result is None else result.get("proposal_match_len", match_n)
                ),
                "proposal_source_start_token": None if result is None else result["source_start"],
                "proposal_source_end_token": None if result is None else result["source_end"],
                "proposal_follow_start_token": None if result is None else result["follow_start"],
                "proposal_follow_end_token": None if result is None else result["follow_end"],
                "proposal_query_start_token": None if result is None else result["query_start"],
                "proposal_query_end_token": None if result is None else result["query_end"],
                "proposal_capped": result is not None and cap is not None and len(proposal) < result["uncapped_len"],
            }}
            if mode == "pld_optimized":
                row["optimized_pld_used"] = bool(result and result.get("optimized_pld_used"))
                row["optimized_pld_fallback"] = optimized_fallback is not None
                row["optimized_pld_fallback_reason"] = optimized_fallback
                row["optimized_pld_warmed"] = _OPTIMIZED_WARMED
                row["optimized_pld_warmup_error"] = _OPTIMIZED_WARMUP_ERROR
            if trace_tokens:
                row["history_token_ids"] = [int(token) for token in prefix]
                row["prompt_len"] = 0
                row["proposal_token_ids"] = [int(token) for token in proposal]
            return row


        def _trace_existing_drafts(
            self,
            mode,
            sampled_token_ids,
            num_tokens_no_spec,
            token_ids_cpu,
            drafts,
            args,
            kwargs,
            elapsed_us,
        ):
            trace_path, trace_tokens, sample_rate = _trace_config(mode)
            if not trace_path:
                return
            valid_rows = _valid_rows_from_args_kwargs(args, kwargs)
            row_count = len(drafts) if drafts is not None else (
                len(sampled_token_ids) if sampled_token_ids is not None else len(num_tokens_no_spec)
            )
            cap = _infer_cap(self)
            for request_i in range(row_count):
                if not (sample_rate >= 1.0 or (sample_rate > 0.0 and _random.random() < sample_rate)):
                    continue
                row_i = int(valid_rows[request_i]) if valid_rows is not None and request_i < len(valid_rows) else request_i
                prefix_len = int(_index_value(num_tokens_no_spec, row_i))
                prefix = _row_prefix(token_ids_cpu, row_i, prefix_len)
                proposal = _proposal_at(drafts, request_i)
                token_blob = ",".join(str(int(token)) for token in proposal).encode("utf-8")
                hit = len(proposal) > 0
                row = {{
                    "mode": mode,
                    "run_id": _os.environ.get("VANTAGE_PLD_RUN_ID"),
                    "request_count": row_count,
                    "request_index": request_i,
                    "row_index": row_i,
                    "prefix_len": prefix_len,
                    "cap": cap,
                    "proposal_cap": cap,
                    "num_speculative_tokens_cap": cap,
                    "match_found": hit,
                    "hit": hit,
                    "miss": not hit,
                    "proposal_len": len(proposal),
                    "proposal_tokens": len(proposal),
                    "proposal_token_hash": _hashlib.sha256(token_blob).hexdigest(),
                    "elapsed_us": elapsed_us,
                    "equivalence_label_candidate": mode,
                    "delegated_original": True,
                }}
                if trace_tokens:
                    row["history_token_ids"] = [int(token) for token in prefix]
                    row["prompt_len"] = 0
                    row["proposal_token_ids"] = [int(token) for token in proposal]
                _append_trace(trace_path, row)


        def _append_trace(path, row):
            path_obj = _pathlib.Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            with path_obj.open("a", encoding="utf-8") as handle:
                handle.write(_json.dumps(row, sort_keys=True) + "\\n")


        def _row_prefix(token_ids_cpu, row_i, prefix_len):
            row = token_ids_cpu[row_i]
            values = row[:prefix_len]
            if hasattr(values, "tolist"):
                values = values.tolist()
            return [int(token) for token in values]


        def _proposal_at(drafts, index):
            if drafts is None:
                return []
            try:
                proposal = drafts[index]
            except Exception:
                return []
            if proposal is None:
                return []
            if hasattr(proposal, "tolist"):
                proposal = proposal.tolist()
            return [int(token) for token in proposal]


        def _index_value(values, index):
            value = values[index]
            if hasattr(value, "item"):
                value = value.item()
            return value


        def _valid_rows_from_args_kwargs(args, kwargs):
            if args:
                first = args[0]
                if first is not None and not isinstance(first, (int, float, str, bytes)):
                    if hasattr(first, "tolist"):
                        first = first.tolist()
                    try:
                        return [int(item) for item in first]
                    except Exception:
                        pass
            for key in ("valid_ngram_requests", "valid_requests", "request_indices"):
                value = kwargs.get(key)
                if value is not None:
                    if hasattr(value, "tolist"):
                        value = value.tolist()
                    return [int(item) for item in value]
            return None


        def _infer_cap(self):
            for attr in ("num_speculative_tokens", "_num_speculative_tokens", "k", "num_draft_tokens"):
                value = getattr(self, attr, None)
                if value is not None:
                    try:
                        return int(value)
                    except Exception:
                        pass
            for owner_attr in ("speculative_config", "spec_config", "vllm_config"):
                owner = getattr(self, owner_attr, None)
                if owner is None:
                    continue
                value = getattr(owner, "num_speculative_tokens", None)
                if value is not None:
                    try:
                        return int(value)
                    except Exception:
                        pass
                spec = getattr(owner, "speculative_config", None)
                value = getattr(spec, "num_speculative_tokens", None)
                if value is not None:
                    try:
                        return int(value)
                    except Exception:
                        pass
            return None


        def _env_int(name, default):
            value = _os.environ.get(name)
            return int(value) if value not in (None, "") else int(default)


        def _env_optional_int(name):
            value = _os.environ.get(name)
            return int(value) if value not in (None, "") else None


        def _env_float(name, default):
            value = _os.environ.get(name)
            return float(value) if value not in (None, "") else float(default)


        def _env_bool(name, default):
            value = _os.environ.get(name)
            if value in (None, ""):
                return bool(default)
            return str(value).lower() in ("1", "true", "yes", "on")


        def _trace_config(mode):
            trace_path = _os.environ.get("VANTAGE_PLD_TRACE_PATH", "")
            trace_tokens = _env_bool("VANTAGE_PLD_TRACE_TOKENS", False)
            default_sample = 1.0 if mode in ("passthrough_trace", "native_fixed_n") else 0.0
            sample_rate = _env_float("VANTAGE_PLD_TRACE_SAMPLE_RATE", default_sample)
            return trace_path, trace_tokens, sample_rate
        '''
    )


if __name__ == "__main__":
    raise SystemExit(main())
