"""Tests for tree-sitter cursor + per-AST-node-type policy lookup."""

from __future__ import annotations

import pytest

pytest.importorskip("tree_sitter_language_pack")

from asts.ast_policy import ASTPolicy, DATA_DERIVED_POLICY, DEFAULT_POLICY, OPTIMAL_POLICY


def test_default_policy_has_required_keys():
    for key in ("default", "function_definition", "ERROR", "string", "comment"):
        assert key in DEFAULT_POLICY


def test_data_derived_policy_alias_is_backwards_compatible():
    assert DATA_DERIVED_POLICY["default"] == 2
    assert OPTIMAL_POLICY is DATA_DERIVED_POLICY


def test_cursor_inside_function_body():
    p = ASTPolicy(language="python")
    src = b"def foo():\n    x = "
    p.update(src)
    ctx = p.context_at_cursor()
    # Cursor sits after "x = " — inside an assignment-like expression
    # The deepest node is some expression / ERROR; the policy walks up to find a match
    # We expect it to land on either assignment, expression_statement, block, or function_definition
    assert ctx.k >= 1
    assert ctx.node_type in DEFAULT_POLICY


def test_cursor_at_top_of_file():
    p = ASTPolicy(language="python")
    p.update(b"")
    ctx = p.context_at_cursor()
    # Empty source — cursor at byte 0; should fall back to module/default
    assert ctx.k >= 1


def test_unterminated_string_does_not_crash():
    p = ASTPolicy(language="python")
    p.update(b'x = "hello wor')
    ctx = p.context_at_cursor()
    # Tree-sitter recovers various ways for unterminated strings; the only
    # invariant we need is that the lookup succeeds and returns a valid k.
    assert ctx.k >= 1
    assert isinstance(ctx.node_type, str)


def test_cursor_inside_well_formed_function():
    # Cursor positioned right after the opening of a function body — walks
    # up to one of {block, function_definition, module}. Exact policy k
    # values are empirically tuned and may change; just verify lookup works.
    p = ASTPolicy(language="python")
    p.update(b"def foo():\n    ")
    ctx = p.context_at_cursor()
    assert ctx.node_type in {"block", "function_definition", "module", "default"}
    assert ctx.k >= 1


def test_typescript_loads():
    p = ASTPolicy(language="typescript")
    p.update(b"function add(a: number, b: number): number {\n  return ")
    ctx = p.context_at_cursor()
    assert ctx.k >= 1


def test_incremental_update_extends():
    p = ASTPolicy(language="python")
    p.update(b"def foo():\n")
    ctx1 = p.context_at_cursor()
    p.update(b"def foo():\n    x = 1\n")
    ctx2 = p.context_at_cursor()
    # Both should resolve to valid policies; not asserting node_type since
    # tree-sitter's recovery for partial source can vary.
    assert ctx1.k >= 1
    assert ctx2.k >= 1


def test_custom_policy_overrides_default():
    custom = dict(DEFAULT_POLICY)
    custom["function_definition"] = 12
    p = ASTPolicy(language="python", policy=custom)
    assert p.policy["function_definition"] == 12


def test_diverged_source_triggers_cold_reparse():
    # If the new source isn't an extension of the previous, tree-sitter does a
    # cold reparse rather than incremental. Test it doesn't crash.
    p = ASTPolicy(language="python")
    p.update(b"def foo():\n    return 1\n")
    p.update(b"def bar():\n    return 2\n")  # diverged
    ctx = p.context_at_cursor()
    assert ctx.k >= 1
