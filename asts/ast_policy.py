"""Tree-sitter cursor + per-AST-node-type draft-length policy.

The cursor lives at byte position `len(prefix_bytes)` (i.e., immediately
after the last byte of the partial source). We ask tree-sitter for the
deepest node containing that position, walk up to the first ancestor whose
type has a policy entry, and use that entry's k value.
"""

from __future__ import annotations

from dataclasses import dataclass


# Empirically-tuned policy from proto_smoke_v2 run on HumanEval (Qwen-Coder-7B
# target + Qwen-Coder-0.5B draft). Each k targets ~3-5 expected accepted
# tokens given the measured per-node-type acceptance rate, balancing draft
# cost vs. accepted yield.
DEFAULT_POLICY: dict[str, int] = {
    # Highly-predictable boilerplate — measured 95% acceptance at k=4
    "parameters": 6,
    # Block-level — measured 22-38% acceptance, dropped from naive k=6
    "block": 3,
    "module": 3,
    "function_definition": 6,
    "class_definition": 5,
    "if_statement": 4,
    "elif_clause": 4,
    "else_clause": 4,
    "for_statement": 4,
    "while_statement": 4,
    "with_statement": 4,
    "try_statement": 4,
    "except_clause": 4,
    # Statement-level — measured 44-67% acceptance
    "assignment": 4,
    "augmented_assignment": 3,
    "expression_statement": 3,
    "return_statement": 3,
    # Containers
    "argument_list": 3,
    "list": 3,
    "tuple": 3,
    "dictionary": 3,
    # Expressions — low individual predictability
    "binary_operator": 2,
    "call": 2,
    "subscript": 2,
    "attribute": 2,
    "identifier": 2,
    # Literals / unstructured (don't speculate)
    "string": 1,
    "string_content": 1,
    "comment": 1,
    # Tree-sitter recovery flag — keep conservative, but the 59% acceptance
    # at k=1 suggests the model is genuinely uncertain at these positions
    "ERROR": 2,
    # ----- TypeScript-specific node types -----
    # Tree-sitter-typescript uses different names than tree-sitter-python.
    # We mirror Python defaults for similar concepts; values may need re-tuning
    # once we have a TS acceptance histogram.
    "program": 3,                  # like python "module"
    "function_declaration": 6,     # like function_definition
    "function_signature": 4,
    "arrow_function": 6,
    "method_definition": 5,
    "class_declaration": 5,
    "interface_declaration": 4,
    "type_alias_declaration": 4,
    "enum_declaration": 5,
    "module_declaration": 5,
    # TS statement / block analogues
    "statement_block": 3,          # like python "block"
    "lexical_declaration": 4,      # let/const, like assignment
    "variable_declaration": 4,     # var
    "variable_declarator": 3,
    # TS expression analogues
    "binary_expression": 2,        # like binary_operator
    "call_expression": 2,          # like call
    "member_expression": 2,        # like attribute
    "subscript_expression": 2,     # like subscript
    "conditional_expression": 2,
    "ternary_expression": 2,
    # TS argument lists / parameters
    "formal_parameters": 6,        # like parameters (high acceptance from py)
    "arguments": 3,                # like argument_list
    # TS types (declarative — usually predictable)
    "type_annotation": 2,
    "type_arguments": 2,
    "type_parameters": 3,
    "generic_type": 2,
    "predefined_type": 1,          # primitives (string/number/boolean)
    # Fallback
    "default": 3,
}


# Data-derived policy (May 2026 ablation).
#
# Methodology: from chain-k ablation on EAGLE-1 + Qwen2.5-Coder-7B,
# per-position acceptance is α_2≈0.67, α_3≈0.20-0.25, α_4+≈0.15. Past
# position 2 the draft is mostly guessing in the dominant measured contexts,
# so the extra draft compute costs more than the rare additional accepted
# tokens. The selected k is therefore 2 nearly everywhere, with k=1 reserved
# for trivial nodes
# where the next byte is highly determined (one-character literals
# inside strings, end-of-comment tokens, single-keyword type
# annotations) — saving one draft forward on those steps.
#
# In our data, switching from the hand-crafted DEFAULT_POLICY to this
# reduces mean-k from 2.20 → ~1.95 for Python while preserving
# per-step acceptance. Relative to fixed k=2, the mean-k reduction is only
# ~2.5%; the larger ~11% reduction is relative to DEFAULT_POLICY.
DATA_DERIVED_POLICY: dict[str, int] = {
    # Trivial / unstructured — next token mostly determined by tokenizer
    "comment": 1,
    "string": 1,
    "string_content": 1,
    "predefined_type": 1,  # TS primitives: string/number/boolean
    # Everything else uses k=2 via the default fallback
    "default": 2,
}

# Backwards-compatible name used by older scripts and aggregate metadata. The
# paper refers to this as DATA_DERIVED_POLICY because it is fit from traces,
# not provably optimal.
OPTIMAL_POLICY = DATA_DERIVED_POLICY


@dataclass(frozen=True)
class CursorContext:
    node_type: str
    """The AST node type used for the policy lookup (may be from an ancestor
    if the cursor's deepest node has no policy entry)."""

    deepest_type: str
    """The actual deepest node at the cursor (for instrumentation)."""

    k: int
    """Resolved draft length for this context."""

    ancestor_types: tuple[str, ...] = ()
    """Deepest-to-root tree-sitter node types at the cursor."""

    parser_in_error: bool = False
    """Whether the cursor path contains a tree-sitter ERROR recovery node."""


class ASTPolicy:
    """Maintains an incremental tree-sitter parse and answers `node_at_cursor`.

    Supports Python and TypeScript via tree-sitter-language-pack.
    """

    def __init__(self, language: str = "python", policy: dict[str, int] | None = None):
        from tree_sitter_language_pack import get_parser

        if language == "typescript":
            self._parser = get_parser("typescript")
        else:
            self._parser = get_parser(language)
        self._language = language
        self._policy = dict(policy) if policy is not None else dict(DEFAULT_POLICY)
        self._tree = None
        self._source: bytes = b""

    @property
    def policy(self) -> dict[str, int]:
        return dict(self._policy)

    def update(self, new_source: bytes) -> None:
        """Update the parsed tree to reflect `new_source`.

        Uses incremental parsing if a previous tree exists. Caller is
        responsible for ensuring `new_source` is a valid extension of the
        previously-parsed source (or fresh content).
        """
        if self._tree is None:
            self._tree = self._parser.parse(new_source)
            self._source = new_source
            return

        old_len = len(self._source)
        new_len = len(new_source)
        # Common case during decoding: append-only at the end
        if new_len >= old_len and new_source[:old_len] == self._source:
            old_end_byte = old_len
            new_end_byte = new_len
            old_end_point = _byte_to_point(self._source, old_end_byte)
            new_end_point = _byte_to_point(new_source, new_end_byte)
            self._tree.edit(
                start_byte=old_end_byte,
                old_end_byte=old_end_byte,
                new_end_byte=new_end_byte,
                start_point=old_end_point,
                old_end_point=old_end_point,
                new_end_point=new_end_point,
            )
            self._tree = self._parser.parse(new_source, self._tree)
        else:
            # Source diverged (rejection rolled back tokens): cold reparse
            self._tree = self._parser.parse(new_source)
        self._source = new_source

    def context_at_cursor(self) -> CursorContext:
        """Return the policy-relevant node + resolved k at the cursor.

        IMPORTANT: tree-sitter node ranges use *exclusive* end_byte, so
        querying at `len(source)` falls outside every descendant and returns
        the root. We query at `len(source) - 1` (the last byte INSIDE the
        source) to find the container the *next-emitted* token will logically
        belong to.
        """
        if self._tree is None or len(self._source) == 0:
            return CursorContext(
                node_type="default",
                deepest_type="default",
                k=self._policy["default"],
            )

        cursor = len(self._source) - 1
        node = self._tree.root_node.descendant_for_byte_range(cursor, cursor)
        if node is None:
            return CursorContext(
                node_type="default",
                deepest_type="default",
                k=self._policy["default"],
            )

        deepest_type = node.type
        # Walk up to find the first ancestor whose type has a policy entry
        cur = node
        ancestor_types: list[str] = []
        resolved_type: str | None = None
        resolved_k: int | None = None
        while cur is not None:
            ancestor_types.append(cur.type)
            if resolved_type is None and cur.type in self._policy:
                resolved_type = cur.type
                resolved_k = self._policy[cur.type]
            cur = cur.parent
        if resolved_type is not None and resolved_k is not None:
            return CursorContext(
                node_type=resolved_type,
                deepest_type=deepest_type,
                k=resolved_k,
                ancestor_types=tuple(ancestor_types),
                parser_in_error="ERROR" in ancestor_types,
            )
        # No ancestor matched (extremely rare — root_node should match "module")
        return CursorContext(
            node_type="default",
            deepest_type=deepest_type,
            k=self._policy["default"],
            ancestor_types=tuple(ancestor_types),
            parser_in_error="ERROR" in ancestor_types,
        )


def _byte_to_point(source: bytes, byte_offset: int) -> tuple[int, int]:
    if byte_offset <= 0:
        return (0, 0)
    head = source[:byte_offset]
    row = head.count(b"\n")
    last_nl = head.rfind(b"\n")
    col = byte_offset - (last_nl + 1)
    return (row, col)
