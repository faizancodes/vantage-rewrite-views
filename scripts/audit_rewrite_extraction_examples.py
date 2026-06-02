#!/usr/bin/env python3
"""Audit the rewrite-extraction examples used in the VANTAGE appendix.

This is a non-GPU check. It exercises the public prompt forms documented in the
paper against the actual prompt-only extractor and boundary-aware rewrite
application used by Rewrite-View Lookup.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asts.code_proposers import apply_boundary_rewrites, extract_explicit_rewrites  # noqa: E402


@dataclass(frozen=True)
class AuditCase:
    name: str
    prompt: str
    reference: str
    expected_map: dict[str, str]
    expected_transformed: str
    expected_fallback: bool
    note: str


@dataclass(frozen=True)
class AuditResult:
    name: str
    prompt: str
    reference: str
    extracted_map: dict[str, str]
    transformed_reference: str
    expected_map: dict[str, str]
    expected_transformed: str
    fallback_condition: bool
    expected_fallback: bool
    pass_: bool
    note: str


def _cases() -> list[AuditCase]:
    return [
        AuditCase(
            name="identifier_boundary",
            prompt="rename user to account",
            reference=(
                "def get_user_name(user):\n"
                "    user_id = user.id\n"
                "    username = user.name\n"
                "    return get_user(user_id), username, user.name\n"
            ),
            expected_map={"user": "account"},
            expected_transformed=(
                "def get_user_name(account):\n"
                "    user_id = account.id\n"
                "    username = account.name\n"
                "    return get_user(user_id), username, account.name\n"
            ),
            expected_fallback=False,
            note="identifier boundaries preserve get_user, user_id, and username",
        ),
        AuditCase(
            name="dotted_field",
            prompt="replace user.name with account.display_name",
            reference=(
                "def render(user):\n"
                "    return user.name, user.email, other.user.name, username\n"
            ),
            expected_map={"user.name": "account.display_name"},
            expected_transformed=(
                "def render(user):\n"
                "    return account.display_name, user.email, other.account.display_name, username\n"
            ),
            expected_fallback=False,
            note="full dotted-field occurrences only",
        ),
        AuditCase(
            name="leading_attribute",
            prompt="change .name to .display_name",
            reference=(
                "def render(obj, username):\n"
                "    return obj.name, profile.name, username\n"
            ),
            expected_map={".name": ".display_name"},
            expected_transformed=(
                "def render(obj, username):\n"
                "    return obj.display_name, profile.display_name, username\n"
            ),
            expected_fallback=False,
            note="leading attribute form rewrites attribute suffixes, not username",
        ),
        AuditCase(
            name="arrow_identifier",
            prompt="OLD -> NEW",
            reference="OLD = 1\nOLDER = OLD + 1\n",
            expected_map={"OLD": "NEW"},
            expected_transformed="NEW = 1\nOLDER = NEW + 1\n",
            expected_fallback=False,
            note="arrow form with identifier boundaries",
        ),
        AuditCase(
            name="quoted_literal",
            prompt='replace "old" with "new"',
            reference='status = "old"\nold_value = "old"\n',
            expected_map={"old": "new"},
            expected_transformed='status = "new"\nold_value = "new"\n',
            expected_fallback=False,
            note="quoted literal form is cleaned to the literal body",
        ),
        AuditCase(
            name="identity_self_map_dropped",
            prompt="rename user to user",
            reference="def f(user):\n    return user.name\n",
            expected_map={},
            expected_transformed="def f(user):\n    return user.name\n",
            expected_fallback=True,
            note="identity self-maps are dropped by extraction",
        ),
        AuditCase(
            name="absent_old_fallback",
            prompt="rename missing_identifier to new_identifier",
            reference="def f(user):\n    return user.name\n",
            expected_map={"missing_identifier": "new_identifier"},
            expected_transformed="def f(user):\n    return user.name\n",
            expected_fallback=True,
            note="explicit absent-old map is extracted but produces no replacement",
        ),
        AuditCase(
            name="negated_instruction",
            prompt="do not rename user to account",
            reference="def f(user):\n    return user.name\n",
            expected_map={},
            expected_transformed="def f(user):\n    return user.name\n",
            expected_fallback=True,
            note="negated rewrite instructions are ignored",
        ),
    ]


def run_audit() -> list[AuditResult]:
    results: list[AuditResult] = []
    for case in _cases():
        extracted = extract_explicit_rewrites(case.prompt)
        transformed = apply_boundary_rewrites(case.reference, extracted)
        fallback = (not extracted) or (transformed == case.reference)
        passed = (
            extracted == case.expected_map
            and transformed == case.expected_transformed
            and fallback == case.expected_fallback
        )
        results.append(
            AuditResult(
                name=case.name,
                prompt=case.prompt,
                reference=case.reference,
                extracted_map=extracted,
                transformed_reference=transformed,
                expected_map=case.expected_map,
                expected_transformed=case.expected_transformed,
                fallback_condition=fallback,
                expected_fallback=case.expected_fallback,
                pass_=passed,
                note=case.note,
            )
        )
    return results


def _write_markdown(results: list[AuditResult], path: Path) -> None:
    lines = [
        "# Rewrite Extraction Audit",
        "",
        "This non-GPU audit exercises the appendix rewrite-extraction examples against",
        "`extract_explicit_rewrites` and `apply_boundary_rewrites`.",
        "",
        "| Case | Prompt | Extracted map | Fallback | Pass | Note |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        extracted = json.dumps(r.extracted_map, sort_keys=True)
        lines.append(
            f"| `{r.name}` | `{r.prompt}` | `{extracted}` | "
            f"{str(r.fallback_condition).lower()} | {str(r.pass_).lower()} | {r.note} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def _write_latex(results: list[AuditResult], path: Path) -> None:
    lines = [
        "% Generated by scripts/audit_rewrite_extraction_examples.py",
        "\\begin{tabular}{llll}",
        "\\toprule",
        "Case & Extracted map & Fallback & Pass \\\\",
        "\\midrule",
    ]
    for r in results:
        extracted = json.dumps(r.extracted_map, sort_keys=True).replace("_", "\\_")
        lines.append(
            f"{r.name.replace('_', '\\_')} & \\texttt{{{extracted}}} & "
            f"{str(r.fallback_condition).lower()} & {str(r.pass_).lower()} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines))


def main() -> int:
    default_dir = (
        ROOT
        / "artifacts"
        / "vantage_transpld"
        / "tables"
        / f"rewrite_extraction_audit_{date.today():%Y%m%d}_v1"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    args = parser.parse_args()

    results = run_audit()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "vantage/rewrite_extraction_audit/v1",
        "n_cases": len(results),
        "n_passed": sum(r.pass_ for r in results),
        "results": [asdict(r) for r in results],
    }
    (args.output_dir / "rewrite_extraction_audit.json").write_text(
        json.dumps(payload, indent=2)
    )
    _write_markdown(results, args.output_dir / "rewrite_extraction_audit.md")
    _write_latex(results, args.output_dir / "rewrite_extraction_audit.tex")

    if payload["n_passed"] != payload["n_cases"]:
        print(
            f"FAIL: {payload['n_passed']}/{payload['n_cases']} cases passed. "
            f"See {args.output_dir}"
        )
        return 1
    print(f"PASS: {payload['n_passed']}/{payload['n_cases']} cases passed.")
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
