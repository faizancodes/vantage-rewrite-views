from __future__ import annotations

import ast

from scripts.build_real_commit_manifest import (
    _family_from_pairs,
    _map_has_evidence,
    _prompt,
    _rewrite_pairs_from_text,
    _stats,
    _target_leaked,
)


def test_real_commit_rewrite_pairs_from_message():
    pairs = _rewrite_pairs_from_text("Rename user to account and replace user.name with account.display_name")
    assert pairs["user"] == "account"
    assert pairs["user.name"] == "account.display_name"


def test_real_commit_prompt_does_not_include_target():
    reference = "def f(user):\n    return user.name\n"
    target = "def f(account):\n    return account.name\n"
    prompt = _prompt("Rename user to account.", reference, {"user": "account"})
    assert reference.strip() in prompt
    assert not _target_leaked(prompt, target)
    ast.parse(target)


def test_real_commit_stats_characterize_copy_heavy_edit():
    reference = "def f(user):\n    value = user.name\n    return value\n"
    target = "def f(account):\n    value = account.name\n    return value\n"
    stats = _stats(reference, target)
    assert 0.0 < stats["copied_token_percentage"] < 1.0
    assert stats["edit_distance_tokens"] > 0
    assert stats["longest_unchanged_span_tokens"] > 0
    assert stats["changed_hunk_count"] >= 1


def test_real_commit_family_and_map_evidence():
    assert _family_from_pairs({"user": "account"}) == "real_rename"
    assert _family_from_pairs({"user.name": "account.display_name"}) == "real_field_migration"
    assert _map_has_evidence(
        "def f(user):\n    return user.name\n",
        "def f(account):\n    return account.display_name\n",
        {"user.name": "account.display_name"},
    )
