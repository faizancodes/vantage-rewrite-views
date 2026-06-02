from __future__ import annotations

from types import SimpleNamespace

from asts.ast_policy import CursorContext
from asts.code_proposers import (
    AlphaSuffixProposer,
    CheapProposerStack,
    CodeSpineProposer,
    EditAnchorProposer,
    IdentifierTrieProposer,
    LiteralCopyProposer,
    LocalSuffixProposer,
    MacroChunkProposer,
    MultiViewPLDProposer,
    MultiViewTreePLDProposer,
    MultiSourceSuffixIndex,
    MultiSuffixProposer,
    NGramPromptLookupProposer,
    Proposal,
    ProposalFeedback,
    ProposalTree,
    ProposerState,
    DispatchTransPLDProposer,
    LazyCompeteTransPLDProposer,
    PrecomputedTransPLDProposer,
    RewriteNormalizedPLDProposer,
    RoutedTransPLDProposer,
    SimpleAdoptionTransPLDProposer,
    RootedPLDProposer,
    SymbolTreeProposer,
    TransPLDProposer,
    TransPLDInferenceProposer,
    CompoundTransPLDProposer,
    CursorTrackedTransPLDProposer,
    adaptive_spine_draft_len,
    adaptive_suffix_draft_len,
    build_candidate_prefix_tree,
    encode_continuation_with_token_healing,
)
from asts.humaneval import (
    _codeeditor_switch_problem_from_row,
    _mbpp_to_completion_prompt,
    _repo_edit_rename_problem_from_source,
    _transform_humaneval_prompt,
)


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(ch) for ch in text])

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(int(t)) for t in token_ids)


class BoundarySensitiveTokenizer:
    def __call__(self, text, add_special_tokens=False):
        table = {
            "A": [1],
            "B": [99],
            "AB": [1, 2],
            "BC": [99, 3],
            "ABC": [1, 2, 3],
        }
        return SimpleNamespace(input_ids=table.get(text, [ord(ch) for ch in text]))

    def decode(self, token_ids, skip_special_tokens=False):
        table = {1: "A", 2: "B", 3: "C", 99: "<bad-boundary-B>"}
        return "".join(table.get(int(t), chr(int(t))) for t in token_ids)


def test_token_healing_uses_prefix_boundary_when_retokenizing_continuation():
    tok = BoundarySensitiveTokenizer()
    assert tok("B", add_special_tokens=False).input_ids == [99]
    healed = encode_continuation_with_token_healing(
        tok,
        "B",
        prefix_tokens=[1],
        prefix_text="A",
    )
    assert healed == [2]


def _ctx(node="identifier", deepest="identifier", parser_in_error=False):
    return CursorContext(
        node_type=node,
        deepest_type=deepest,
        k=2,
        ancestor_types=(deepest, node, "module"),
        parser_in_error=parser_in_error,
    )


def _state(text_before: str, teacher: str, ctx=None):
    tok = FakeTokenizer()
    prefix = tok(text_before).input_ids
    teacher_id = ord(teacher)
    return ProposerState(
        prefix=prefix,
        teacher_argmax=teacher_id,
        text_before=text_before,
        text_after=text_before + teacher,
        ctx=ctx,
        language="python",
    )


def test_identifier_proposer_completes_recent_identifier():
    tok = FakeTokenizer()
    state = _state("foo_bar = 1\nfo", "o", _ctx())
    proposal = IdentifierTrieProposer(tok).propose(state)
    assert proposal is not None
    assert proposal.kind == "identifier"
    assert tok.decode(proposal.tokens) == "_bar"


def test_literal_proposer_completes_seen_string_literal():
    tok = FakeTokenizer()
    state = _state('expected = "alpha"\nvalue = "al', "p", _ctx("string", "string"))
    proposal = LiteralCopyProposer(tok).propose(state)
    assert proposal is not None
    assert proposal.kind == "literal"
    assert tok.decode(proposal.tokens) == 'ha"'


def test_local_suffix_proposer_uses_exact_prior_token_suffix():
    state = ProposerState(
        prefix=[1, 2, 3, 4, 9, 8, 7, 1, 2, 3],
        teacher_argmax=4,
        text_before="",
        text_after="",
        ctx=None,
    )
    proposal = LocalSuffixProposer(min_match_len=4, max_query_len=4, max_draft_len=3).propose(
        state
    )
    assert proposal is not None
    assert proposal.kind == "local_suffix"
    assert proposal.match_len == 4
    assert proposal.tokens == [9, 8, 7]
    assert proposal.source_start_token == 0
    assert proposal.source_end_token == 4
    assert proposal.follow_start_token == 4
    assert proposal.follow_end_token == 7
    assert proposal.root_included is False


def test_local_suffix_proposer_rejects_overlapping_suffix_match():
    state = ProposerState(
        prefix=[1, 2, 3, 4, 1, 2, 3],
        teacher_argmax=4,
        text_before="",
        text_after="",
        ctx=None,
    )
    proposal = LocalSuffixProposer(min_match_len=4, max_query_len=4, max_draft_len=3).propose(
        state
    )
    assert proposal is None


def test_local_suffix_prompt_pool_ignores_generated_match():
    # The only prior match starts in generated tokens (prompt_len=2), so a
    # prompt-only pool must not use it.
    state = ProposerState(
        prefix=[5, 6, 1, 2, 3, 4, 9, 8, 1, 2, 3],
        teacher_argmax=4,
        text_before="",
        text_after="",
        ctx=None,
        prompt_len=2,
    )
    assert LocalSuffixProposer(min_match_len=4, max_query_len=4, pool="prompt").propose(state) is None
    proposal = LocalSuffixProposer(min_match_len=4, max_query_len=4, pool="generated").propose(state)
    assert proposal is not None
    assert proposal.tokens == [9, 8]
    assert proposal.pool == "generated"


def test_ngram_baseline_uses_fixed_match_size_and_draft_len():
    state = ProposerState(
        prefix=[1, 2, 3, 4, 9, 8, 1, 2, 3],
        teacher_argmax=4,
        text_before="",
        text_after="",
        ctx=None,
        prompt_len=7,
    )
    proposal = NGramPromptLookupProposer(4, 1, pool="local").propose(state)
    assert proposal is not None
    assert proposal.kind == "ngram"
    assert proposal.match_len == 4
    assert proposal.tokens == [9]


def test_rooted_pld_uses_target_root_and_long_window():
    state = ProposerState(
        prefix=[1, 2, 3, 4, 9, 8, 7, 6, 1, 2, 3],
        teacher_argmax=4,
        text_before="",
        text_after="",
        ctx=None,
        prompt_len=8,
    )
    proposal = RootedPLDProposer(max_matching_ngram_size=4, max_draft_len=4).propose(state)
    assert proposal is not None
    assert proposal.kind == "rooted_pld"
    assert proposal.match_len == 4
    assert proposal.tokens == [9, 8, 7, 6]
    assert proposal.root_included is False
    assert proposal.match_kind == "rooted_pld"


def test_multisource_suffix_matches_local_suffix_best_candidate():
    state = ProposerState(
        prefix=[1, 2, 3, 4, 9, 8, 7, 1, 2, 3],
        teacher_argmax=4,
        text_before="",
        text_after="",
        ctx=None,
    )
    local = LocalSuffixProposer(min_match_len=4, max_query_len=4, max_draft_len=3).propose(
        state
    )
    multi = MultiSuffixProposer(
        key_lengths=(4,),
        top_k=1,
        max_draft_len=3,
        tree=False,
    ).propose(state)
    assert local is not None
    assert multi is not None
    assert multi.tokens == local.tokens
    assert multi.match_len == local.match_len
    assert multi.source_start_token == local.source_start_token


def test_multisource_suffix_index_respects_prompt_and_generated_pools():
    seq = [5, 6, 1, 2, 3, 4, 9, 8, 1, 2, 3, 4]
    index = MultiSourceSuffixIndex((4,))
    assert index.query(seq, prompt_len=2, pool="prompt") == []
    generated = index.query(seq, prompt_len=2, pool="generated", top_k=1, max_draft_len=4)
    assert generated
    assert generated[0].tokens == [9, 8]
    assert generated[0].pool == "generated"


def test_multisuffix_adaptive_draft_len_table():
    assert adaptive_suffix_draft_len(3, 16) == 3
    assert adaptive_suffix_draft_len(4, 16) == 5
    assert adaptive_suffix_draft_len(8, 16) == 8
    assert adaptive_suffix_draft_len(12, 16) == 16
    assert adaptive_suffix_draft_len(12, 6) == 6


def test_codespine_adaptive_draft_len_table():
    assert adaptive_spine_draft_len(3, 32) == 3
    assert adaptive_spine_draft_len(4, 32) == 8
    assert adaptive_spine_draft_len(8, 32) == 16
    assert adaptive_spine_draft_len(12, 32) == 32
    assert adaptive_spine_draft_len(12, 12) == 12


def test_candidate_prefix_tree_merges_shared_prefix_and_caps_nodes():
    nodes = build_candidate_prefix_tree([[1, 2, 3], [1, 2, 4], [5]], max_nodes=4)
    assert [n.token for n in nodes] == [1, 2, 3, 4]
    assert [n.parent for n in nodes] == [-1, 0, 1, 1]
    capped = build_candidate_prefix_tree([[1, 2, 3, 4]], max_nodes=2)
    assert [n.token for n in capped] == [1, 2]


def test_multisuffix_tree_deduplicates_candidates_and_obeys_node_cap():
    state = ProposerState(
        prefix=[1, 2, 3, 4, 9, 8, 7, 1, 2, 3, 4, 9, 8, 6, 1, 2, 3],
        teacher_argmax=4,
        text_before="",
        text_after="",
        ctx=None,
    )
    proposal = MultiSuffixProposer(
        key_lengths=(4,),
        top_k=4,
        max_draft_len=4,
        max_tree_nodes=4,
        tree=True,
    ).propose(state)
    assert isinstance(proposal, ProposalTree)
    assert len(build_candidate_prefix_tree(proposal.candidates, proposal.max_nodes)) <= 4
    assert len({tuple(c) for c in proposal.candidates}) == len(proposal.candidates)
    assert proposal.root_included is False


def test_proposer_stack_can_return_tree_proposal():
    class Tree:
        kind = "tree"

        def propose(self, state):
            return ProposalTree(
                kind="tree",
                candidates=[[1, 2], [1, 3]],
                scores=[2.0, 1.5],
                sources=["generated", "prompt"],
                match_lens=[4, 4],
                max_nodes=4,
                score=2.0,
            )

    proposal = CheapProposerStack([Tree()]).propose(ProposerState([], 0, "", "", None))
    assert isinstance(proposal, ProposalTree)
    assert proposal.tokens == [1, 2]


def test_codespine_extends_strong_exact_match_deeper_than_suffix():
    tok = FakeTokenizer()
    middle = "0123456789ABCDEFGHIJKLMNO"
    text = "abcdefghijklmnop" + middle + "abcdefghijklmnop"
    state = _state(text[:-1], text[-1], _ctx("block", "block"))
    proposal = CodeSpineProposer(
        tok,
        key_lengths=(16,),
        max_spine_len=32,
        max_tree_nodes=40,
        branch_budget=0,
    ).propose(state)
    assert proposal is not None
    assert tok.decode(proposal.tokens).startswith(middle)
    assert len(proposal.tokens) > adaptive_suffix_draft_len(16, 32)


def test_codespine_does_not_use_match_len_three_by_default():
    tok = FakeTokenizer()
    state = ProposerState(
        prefix=[1, 2, 3, 9, 8, 1, 2, 3, 7, 6, 1, 2],
        teacher_argmax=3,
        text_before="",
        text_after="",
        ctx=None,
    )
    assert CodeSpineProposer(tok, key_lengths=(3,), min_match_len=4).propose(state) is None
    proposal = CodeSpineProposer(
        tok,
        key_lengths=(3,),
        min_match_len=4,
        allow_short_match=True,
        branch_budget=0,
    ).propose(state)
    assert proposal is not None
    assert proposal.match_len == 3


def test_codespine_sparse_branches_obey_node_budget():
    tok = FakeTokenizer()
    state = ProposerState(
        prefix=[1, 2, 3, 4, 9, 8, 7, 1, 2, 3, 4, 9, 8, 6, 1, 2, 3],
        teacher_argmax=4,
        text_before="",
        text_after="",
        ctx=None,
    )
    proposal = CodeSpineProposer(
        tok,
        key_lengths=(4,),
        max_spine_len=8,
        max_tree_nodes=6,
        branch_budget=2,
        enable_identifier_branches=False,
        enable_delimiter_branches=False,
    ).propose(state)
    assert isinstance(proposal, ProposalTree)
    assert len(build_candidate_prefix_tree(proposal.candidates, proposal.max_nodes)) <= 6
    assert proposal.candidates[0][:2] == [9, 8]
    assert len(proposal.candidates) <= 3


def test_editspine_labels_prompt_source_as_edit_reference():
    tok = FakeTokenizer()
    state = ProposerState(
        prefix=[1, 2, 3, 4, 9, 8, 1, 2, 3],
        teacher_argmax=4,
        text_before="",
        text_after="",
        ctx=None,
        prompt_len=6,
    )
    proposal = CodeSpineProposer(
        tok,
        key_lengths=(4,),
        edit_mode=True,
        branch_budget=0,
        enable_delimiter_branches=False,
    ).propose(state)
    assert proposal is not None
    assert proposal.pool == "prompt"
    assert proposal.source_region == "edit_reference"


def test_edit_anchor_copies_aligned_reference_continuation():
    tok = FakeTokenizer()
    prompt = (
        "Please edit this function without changing behavior.\n"
        "```python\n"
        "def foo():\n"
        "    value = 1\n"
        "    return value\n"
        "```\n"
    )
    generated = "def foo():\n    value "
    state = ProposerState(
        prefix=tok(prompt + generated[:-1]).input_ids,
        teacher_argmax=ord(generated[-1]),
        text_before=prompt + generated[:-1],
        text_after=prompt + generated,
        ctx=_ctx("block", "block"),
        language="python",
        prompt_len=len(tok(prompt).input_ids),
    )
    proposal = EditAnchorProposer(tok, min_anchor_chars=8).propose(state)
    assert proposal is not None
    assert proposal.kind == "edit_anchor"
    assert proposal.pool == "edit_reference"
    assert tok.decode(proposal.tokens).startswith("= 1")


def test_edit_anchor_stops_before_instruction_named_edit_term():
    tok = FakeTokenizer()
    prompt = (
        "Please edit this function and change `return` behavior.\n"
        "```python\n"
        "def foo():\n"
        "    value = 1\n"
        "    return value\n"
        "```\n"
    )
    generated = "def foo():\n    value "
    state = ProposerState(
        prefix=tok(prompt + generated[:-1]).input_ids,
        teacher_argmax=ord(generated[-1]),
        text_before=prompt + generated[:-1],
        text_after=prompt + generated,
        ctx=_ctx("block", "block"),
        language="python",
        prompt_len=len(tok(prompt).input_ids),
    )
    proposal = EditAnchorProposer(tok, min_anchor_chars=8).propose(state)
    assert proposal is not None
    assert "return" not in tok.decode(proposal.tokens)


def test_edit_anchor_min_draft_gate_skips_short_spans():
    tok = FakeTokenizer()
    prompt = (
        "Please edit this function without changing behavior.\n"
        "```python\n"
        "def foo():\n"
        "    value = 1\n"
        "    return value\n"
        "```\n"
    )
    generated = "def foo():\n    value "
    state = ProposerState(
        prefix=tok(prompt + generated[:-1]).input_ids,
        teacher_argmax=ord(generated[-1]),
        text_before=prompt + generated[:-1],
        text_after=prompt + generated,
        ctx=_ctx("block", "block"),
        language="python",
        prompt_len=len(tok(prompt).input_ids),
    )
    assert EditAnchorProposer(tok, min_anchor_chars=8, min_draft_tokens=80).propose(state) is None


def test_rewrite_anchor_maps_new_generated_text_to_old_reference():
    tok = FakeTokenizer()
    prompt = (
        "Rename `user_name` to `account_name` and preserve every other token.\n"
        "```python\n"
        "def f(user_name):\n"
        "    if not user_name:\n"
        "        return 'unknown'\n"
        "    return user_name.lower()\n"
        "```\n"
    )
    generated = "def f(account_name):\n    if not account_name:\n"
    state = ProposerState(
        prefix=tok(prompt + generated[:-1]).input_ids,
        teacher_argmax=ord(generated[-1]),
        text_before=prompt + generated[:-1],
        text_after=prompt + generated,
        ctx=_ctx("block", "block"),
        language="python",
        prompt_len=len(tok(prompt).input_ids),
    )
    exact = EditAnchorProposer(tok, min_anchor_chars=12, rewrite_enabled=False).propose(state)
    rewrite = EditAnchorProposer(
        tok,
        kind="rewrite_anchor_pld",
        max_draft_len=96,
        min_anchor_chars=12,
        rewrite_enabled=True,
    ).propose(state)
    assert exact is None
    assert rewrite is not None
    decoded = tok.decode(rewrite.tokens)
    assert "account_name" in decoded
    assert "user_name" not in decoded
    assert rewrite.match_kind == "rewrite_anchor"


def test_rewrite_pld_vref_uses_transformed_reference_pool():
    tok = FakeTokenizer()
    reference = "return user.name\n"
    prompt = "Rename user -> account.\n```python\n" + reference + "```\n"
    generated = "return account"
    state = ProposerState(
        prefix=tok(prompt + generated[:-1]).input_ids,
        teacher_argmax=ord(generated[-1]),
        text_before=prompt + generated[:-1],
        text_after=prompt + generated,
        ctx=_ctx("block", "block"),
        language="python",
        prompt_len=len(tok(prompt).input_ids),
        reference=reference,
    )
    proposal = RewriteNormalizedPLDProposer(
        tok,
        mode="vref",
        max_matching_ngram_size=20,
        max_draft_len=8,
    ).propose(state)
    assert proposal is not None
    assert proposal.kind == "rewrite_norm_pld"
    assert proposal.match_kind == "vref"
    assert proposal.pool == "virtual_reference"
    assert proposal.root_included is False
    assert tok.decode(proposal.tokens).startswith(".name")


def test_rewrite_pld_oracle_uses_manifest_pairs_without_prompt_map():
    tok = FakeTokenizer()
    reference = "return user.name\n"
    prompt = "Rename user to account.\n```python\n" + reference + "```\n"
    generated = "return account"
    state = ProposerState(
        prefix=tok(prompt + generated[:-1]).input_ids,
        teacher_argmax=ord(generated[-1]),
        text_before=prompt + generated[:-1],
        text_after=prompt + generated,
        ctx=_ctx("block", "block"),
        language="python",
        prompt_len=len(tok(prompt).input_ids),
        reference=reference,
        metadata={"rewrite_pairs": {"user": "account"}},
    )
    proposal = RewriteNormalizedPLDProposer(
        tok,
        mode="oracle",
        max_matching_ngram_size=20,
        max_draft_len=8,
    ).propose(state)
    assert proposal is not None
    assert proposal.match_kind == "oracle"
    assert proposal.pool == "oracle_reference"
    assert tok.decode(proposal.tokens).startswith(".name")


def test_rewrite_pld_bidir_matches_new_query_against_old_reference():
    tok = FakeTokenizer()
    reference = "\nreturn user.name\n"
    prompt = "Rename user to account.\n```python\n" + reference + "```\n"
    generated = "return account"
    state = ProposerState(
        prefix=tok(prompt + generated[:-1]).input_ids,
        teacher_argmax=ord(generated[-1]),
        text_before=prompt + generated[:-1],
        text_after=prompt + generated,
        ctx=_ctx("block", "block"),
        language="python",
        prompt_len=len(tok(prompt).input_ids),
        reference=reference,
    )
    proposal = RewriteNormalizedPLDProposer(
        tok,
        mode="bidir",
        max_matching_ngram_size=20,
        max_draft_len=8,
    ).propose(state)
    assert proposal is not None
    assert proposal.match_kind == "bidir"
    assert proposal.pool == "bidir_normalized"
    assert tok.decode(proposal.tokens).startswith(".name")


def test_dispatch_transpld_uses_bidir_view_with_min_match_gate():
    tok = FakeTokenizer()
    reference = "\nreturn user.name\n"
    prompt = "Rename user to account.\n```python\n" + reference + "```\n"
    generated = "return account"
    state = ProposerState(
        prefix=tok(prompt + generated[:-1]).input_ids,
        teacher_argmax=ord(generated[-1]),
        text_before=prompt + generated[:-1],
        text_after=prompt + generated,
        ctx=_ctx("block", "block"),
        language="python",
        prompt_len=len(tok(prompt).input_ids),
        reference=reference,
    )
    proposal = DispatchTransPLDProposer(
        tok,
        max_matching_ngram_size=20,
        max_draft_len=8,
        transformed_min_matching_ngram_size=4,
    ).propose(state)
    assert proposal is not None
    assert proposal.kind == "dispatch_transpld"
    assert proposal.match_kind == "dispatch_transpld_bidir"
    assert proposal.match_len >= 4
    assert tok.decode(proposal.tokens).startswith(".name")


def test_dispatch_transpld_respects_min_match_gate():
    tok = FakeTokenizer()
    reference = "\nreturn user.name\n"
    prompt = "Rename user to account.\n```python\n" + reference + "```\n"
    generated = "return account"
    state = ProposerState(
        prefix=tok(prompt + generated[:-1]).input_ids,
        teacher_argmax=ord(generated[-1]),
        text_before=prompt + generated[:-1],
        text_after=prompt + generated,
        ctx=_ctx("block", "block"),
        language="python",
        prompt_len=len(tok(prompt).input_ids),
        reference=reference,
    )
    proposal = DispatchTransPLDProposer(
        tok,
        max_matching_ngram_size=20,
        max_draft_len=8,
        transformed_min_matching_ngram_size=32,
    ).propose(state)
    assert proposal is None


def test_precomputed_transpld_builds_view_once_and_drafts_from_tokens():
    tok = FakeTokenizer()
    reference = "\nreturn user.name\n"
    prompt = "Rename user to account.\n```python\n" + reference + "```\n"
    generated = "return account"
    proposer = PrecomputedTransPLDProposer(
        tok,
        max_matching_ngram_size=20,
        max_draft_len=8,
        transformed_min_matching_ngram_size=4,
    )
    proposer.prepare(
        prompt_ids=tok(prompt).input_ids,
        prompt_text=prompt,
        reference=reference,
        metadata={},
        prompt_len=len(tok(prompt).input_ids),
        language="python",
    )
    state = ProposerState(
        prefix=tok(prompt + generated[:-1]).input_ids,
        teacher_argmax=ord(generated[-1]),
        text_before="",
        text_after="",
        ctx=None,
        language="python",
        prompt_len=len(tok(prompt).input_ids),
        reference=reference,
        metadata={"rewrite_pairs": {"user": "account"}},
        prompt_text=prompt,
    )
    proposal = proposer.propose(state)
    assert proposal is not None
    assert proposal.kind == "precomputed_transpld"
    assert proposal.match_kind == "precomputed_transpld"
    assert proposal.match_len >= 4
    assert proposal.map_parse_us > 0
    assert proposal.transpld_index_build_us > 0
    assert tok.decode(proposal.tokens).startswith(".name")


def test_precomputed_transpld_ignores_manifest_only_rewrite_pairs():
    tok = FakeTokenizer()
    reference = "return user.name\n"
    prompt = "Apply requested edit.\n```python\n" + reference + "```\n"
    generated = "return account"
    proposer = PrecomputedTransPLDProposer(
        tok,
        max_matching_ngram_size=20,
        max_draft_len=8,
        transformed_min_matching_ngram_size=4,
    )
    proposer.prepare(
        prompt_ids=tok(prompt).input_ids,
        prompt_text=prompt,
        reference=reference,
        metadata={
            "rewrite_pairs": {"user": "account"},
            "gold": "return account.name\n",
            "label": "synthetic_field",
            "manifest_only_field": "must_not_route",
        },
        prompt_len=len(tok(prompt).input_ids),
        language="python",
    )
    assert proposer._rewrite_map == {}
    state = ProposerState(
        prefix=tok(prompt + generated[:-1]).input_ids,
        teacher_argmax=ord(generated[-1]),
        text_before="",
        text_after="",
        ctx=None,
        language="python",
        prompt_len=len(tok(prompt).input_ids),
        reference=reference,
        prompt_text=prompt,
    )
    assert proposer.propose(state) is None


def test_compete_transpld_uses_exact_candidate_when_transformed_view_is_absent():
    tok = FakeTokenizer()
    prompt = "abcdef\n"
    generated = "abc"
    proposer = PrecomputedTransPLDProposer(
        tok,
        max_matching_ngram_size=4,
        max_draft_len=4,
        transformed_min_matching_ngram_size=4,
        compete_exact=True,
        margin=0,
    )
    proposer.prepare(
        prompt_ids=tok(prompt).input_ids,
        prompt_text=prompt,
        reference="",
        metadata={},
        prompt_len=len(tok(prompt).input_ids),
        language="python",
    )
    state = ProposerState(
        prefix=tok(prompt + generated[:-1]).input_ids,
        teacher_argmax=ord(generated[-1]),
        text_before="",
        text_after="",
        ctx=None,
        language="python",
        prompt_len=len(tok(prompt).input_ids),
        reference="",
        prompt_text=prompt,
    )
    proposal = proposer.propose(state)
    assert proposal is not None
    assert proposal.match_kind == "precomputed_exact_competition"
    assert proposal.route == "exact_pld"
    assert tok.decode(proposal.tokens).startswith("def")


def test_compete_transpld_can_gate_exact_candidate_min_match():
    tok = FakeTokenizer()
    prompt = "abcdef\n"
    generated = "abc"
    proposer = PrecomputedTransPLDProposer(
        tok,
        max_matching_ngram_size=4,
        max_draft_len=4,
        transformed_min_matching_ngram_size=4,
        compete_exact=True,
        margin=0,
        exact_min_matching_ngram_size=4,
    )
    proposer.prepare(
        prompt_ids=tok(prompt).input_ids,
        prompt_text=prompt,
        reference="",
        metadata={},
        prompt_len=len(tok(prompt).input_ids),
        language="python",
    )
    state = ProposerState(
        prefix=tok(prompt + generated[:-1]).input_ids,
        teacher_argmax=ord(generated[-1]),
        text_before="",
        text_after="",
        ctx=None,
        language="python",
        prompt_len=len(tok(prompt).input_ids),
        reference="",
        prompt_text=prompt,
    )
    assert proposer.propose(state) is None


def test_lazy_transpld_skips_view_build_when_exact_pld_is_strong():
    tok = FakeTokenizer()
    proposer = LazyCompeteTransPLDProposer(
        tok,
        max_matching_ngram_size=4,
        max_draft_len=4,
        exact_strong_min_len=3,
        trans_len_margin=0,
    )
    state = ProposerState(
        prefix=[1, 2, 3, 4, 9, 8, 7, 1, 2, 3],
        teacher_argmax=4,
        text_before="",
        text_after="",
        ctx=None,
        prompt_len=7,
        reference="return user.name\n",
        prompt_text="Rename user to account.\n```python\nreturn user.name\n```\n",
    )
    proposal = proposer.propose(state)
    assert proposal is not None
    assert proposal.kind == "rooted_pld"
    assert proposal.match_kind == "rooted_pld"
    assert proposal.route is None
    assert proposal.transpld_index_build_us == 0
    assert proposer._trans._prepared is False


def test_lazy_transpld_builds_view_only_when_exact_pld_is_weak():
    tok = FakeTokenizer()
    reference = "\nreturn user.name\n"
    prompt = "Rename user to account.\n```python\n" + reference + "```\n"
    generated = "return account"
    proposer = LazyCompeteTransPLDProposer(
        tok,
        max_matching_ngram_size=20,
        max_draft_len=8,
        transformed_min_matching_ngram_size=4,
        exact_strong_min_len=32,
        trans_len_margin=-8,
    )
    state = ProposerState(
        prefix=tok(prompt + generated[:-1]).input_ids,
        teacher_argmax=ord(generated[-1]),
        text_before="",
        text_after="",
        ctx=None,
        language="python",
        prompt_len=len(tok(prompt).input_ids),
        reference=reference,
        prompt_text=prompt,
    )
    proposal = proposer.propose(state)
    assert proposal is not None
    assert proposal.kind == "lazy_compete_transpld"
    assert proposal.match_kind == "lazy_transpld"
    assert proposal.route == "transpld"
    assert proposal.transpld_index_build_us > 0
    assert tok.decode(proposal.tokens).startswith(".name")


def test_multiview_pld_fast_path_returns_exact_without_view_build():
    tok = FakeTokenizer()
    proposer = MultiViewPLDProposer(
        tok,
        max_matching_ngram_size=4,
        max_draft_len=4,
        exact_strong_min_len=3,
        trans_len_margin=0,
    )
    state = ProposerState(
        prefix=[1, 2, 3, 4, 9, 8, 7, 1, 2, 3],
        teacher_argmax=4,
        text_before="",
        text_after="",
        ctx=None,
        prompt_len=7,
        reference="return user.name\n",
        prompt_text="Rename user to account.\n```python\nreturn user.name\n```\n",
    )
    proposal = proposer.propose(state)
    assert proposal is not None
    assert proposal.kind == "rooted_pld"
    assert proposer._prepared is False
    assert proposal.transpld_index_build_us == 0


def test_multiview_pld_uses_transformed_reference_when_exact_is_weak():
    tok = FakeTokenizer()
    reference = "\nreturn user.name\n"
    prompt = "Rename user to account.\n```python\n" + reference + "```\n"
    generated = "return account"
    proposer = MultiViewPLDProposer(
        tok,
        max_matching_ngram_size=20,
        max_draft_len=8,
        transformed_min_matching_ngram_size=4,
        exact_strong_min_len=32,
        trans_len_margin=-8,
    )
    state = ProposerState(
        prefix=tok(prompt + generated[:-1]).input_ids,
        teacher_argmax=ord(generated[-1]),
        text_before="",
        text_after="",
        ctx=None,
        language="python",
        prompt_len=len(tok(prompt).input_ids),
        reference=reference,
        metadata={"rewrite_pairs": {"user": "account"}},
        prompt_text=prompt,
    )
    proposal = proposer.propose(state)
    assert proposal is not None
    assert proposal.kind == "multiview_pld"
    assert proposal.match_kind == "multiview_transformed"
    assert tok.decode(proposal.tokens).startswith(".name")
    assert proposal.compound_view_count and proposal.compound_view_count >= 1


def test_multiview_tree_verifies_exact_and_transformed_branches():
    tok = FakeTokenizer()
    reference = "\nreturn user.name\n"
    prompt = (
        "Rename user to account.\n"
        "Prior text: return account.old\n"
        "```python\n"
        + reference
        + "```\n"
    )
    generated = "return account"
    proposer = MultiViewTreePLDProposer(
        tok,
        max_matching_ngram_size=20,
        max_draft_len=8,
        transformed_min_matching_ngram_size=4,
        exact_strong_min_len=32,
        trans_len_margin=0,
        max_tree_nodes=24,
    )
    state = ProposerState(
        prefix=tok(prompt + generated[:-1]).input_ids,
        teacher_argmax=ord(generated[-1]),
        text_before="",
        text_after="",
        ctx=None,
        language="python",
        prompt_len=len(tok(prompt).input_ids),
        reference=reference,
        metadata={"rewrite_pairs": {"user": "account"}},
        prompt_text=prompt,
    )
    proposal = proposer.propose(state)
    assert isinstance(proposal, ProposalTree)
    assert proposal.kind == "multiview_tree_pld"
    decoded = [tok.decode(c) for c in proposal.candidates]
    assert any(c.startswith(".old") for c in decoded)
    assert any(c.startswith(".name") for c in decoded)


def test_rewrite_pld_bidir_parses_leading_dot_field_rewrite():
    tok = FakeTokenizer()
    reference = "mapping = map(self.add_ten, seq)\n"
    prompt = (
        "Apply this field rename: replace .add_ten to .add_ten_updated.\n"
        "```python\n"
        + reference
        + "```\n"
    )
    generated = "mapping = map(self.add_ten_updated"
    state = ProposerState(
        prefix=tok(prompt + generated[:-1]).input_ids,
        teacher_argmax=ord(generated[-1]),
        text_before=prompt + generated[:-1],
        text_after=prompt + generated,
        ctx=_ctx("block", "block"),
        language="python",
        prompt_len=len(tok(prompt).input_ids),
        reference=reference,
    )
    proposal = RewriteNormalizedPLDProposer(
        tok,
        mode="bidir",
        max_matching_ngram_size=40,
        max_draft_len=8,
    ).propose(state)
    assert proposal is not None
    assert proposal.match_kind == "bidir"
    assert proposal.substitution_count == 1
    assert tok.decode(proposal.tokens).startswith(", seq)")


def test_rewrite_pld_requires_reference_and_rewrite_map():
    tok = FakeTokenizer()
    state = _state("return accoun", "t", _ctx("block", "block"))
    assert RewriteNormalizedPLDProposer(tok, mode="vref").propose(state) is None

    reference = "return user.name\n"
    prompt = "Rewrite this function.\n```python\n" + reference + "```\n"
    generated = "return account"
    state = ProposerState(
        prefix=tok(prompt + generated[:-1]).input_ids,
        teacher_argmax=ord(generated[-1]),
        text_before=prompt + generated[:-1],
        text_after=prompt + generated,
        ctx=_ctx("block", "block"),
        language="python",
        prompt_len=len(tok(prompt).input_ids),
        reference=reference,
    )
    assert RewriteNormalizedPLDProposer(tok, mode="vref").propose(state) is None


def test_transpld_alias_routes_to_transformed_reference_with_exact_fallback():
    tok = FakeTokenizer()
    reference = "return user.name\n"
    prompt = "Rename user to account.\n```python\n" + reference + "```\n"
    generated = "return account"
    state = ProposerState(
        prefix=tok(prompt + generated[:-1]).input_ids,
        teacher_argmax=ord(generated[-1]),
        text_before=prompt + generated[:-1],
        text_after=prompt + generated,
        ctx=_ctx("block", "block"),
        language="python",
        prompt_len=len(tok(prompt).input_ids),
        reference=reference,
    )
    proposal = TransPLDProposer(tok, max_matching_ngram_size=20, max_draft_len=8).propose(state)
    assert proposal is not None
    assert proposal.kind == "transpld"
    assert proposal.match_kind == "transpld_vref"
    assert proposal.map_source == "explicit"
    assert proposal.active_map_count == 1
    assert tok.decode(proposal.tokens).startswith(".name")

    no_map_state = ProposerState(
        prefix=[1, 2, 3, 4, 9, 8, 1, 2, 3],
        teacher_argmax=4,
        text_before="",
        text_after="",
        ctx=None,
        prompt_len=7,
    )
    fallback = TransPLDProposer(tok, max_matching_ngram_size=4, max_draft_len=2).propose(no_map_state)
    assert fallback is not None
    assert fallback.match_kind == "transpld_exact_fallback"
    assert fallback.view_id == "identity"
    assert fallback.tokens == [9, 8]


def test_transpld_uses_prompt_map_when_decoded_prompt_loses_surface_form():
    tok = FakeTokenizer()
    reference = "mapping = map(self.add_ten, seq)\n"
    prompt = "Rename .add_ten to .add_ten_updated.\n```python\n" + reference + "```\n"
    generated = "mapping = map(self.add_ten_updated"
    state = ProposerState(
        prefix=tok(prompt + generated[:-1]).input_ids,
        teacher_argmax=ord(generated[-1]),
        text_before=prompt + generated[:-1],
        text_after=prompt + generated,
        ctx=_ctx("block", "block"),
        language="python",
        prompt_len=len(tok(prompt).input_ids),
        reference=reference,
        metadata={"rewrite_pairs": {".add_ten": ".wrong_target"}},
    )
    proposal = TransPLDProposer(tok, max_matching_ngram_size=40, max_draft_len=8).propose(state)
    assert proposal is not None
    assert proposal.match_kind == "transpld_vref"
    assert proposal.map_source == "explicit"
    assert tok.decode(proposal.tokens).startswith(", seq)")


def test_transpld_does_not_use_hidden_manifest_map():
    tok = FakeTokenizer()
    reference = "mapping = map(self.add_ten, seq)\n"
    prompt = "Apply the requested refactor.\n```python\n" + reference + "```\n"
    generated = "mapping = map(self.add_ten_updated"
    state = ProposerState(
        prefix=tok(prompt + generated[:-1]).input_ids,
        teacher_argmax=ord(generated[-1]),
        text_before=prompt + generated[:-1],
        text_after=prompt + generated,
        ctx=_ctx("block", "block"),
        language="python",
        prompt_len=len(tok(prompt).input_ids),
        reference=reference,
        metadata={
            "rewrite_pairs": {".add_ten": ".add_ten_updated"},
            "map_visibility": "hidden",
        },
    )
    proposal = TransPLDProposer(tok, max_matching_ngram_size=40, max_draft_len=8).propose(state)
    assert proposal is None or proposal.match_kind != "transpld_vref"


def test_transpld_refuses_weak_transformed_view_matches():
    tok = FakeTokenizer()
    reference = "a user.name\n"
    prompt = "Rename user to account.\n```python\n" + reference + "```\n"
    generated = "a"
    state = ProposerState(
        prefix=tok(prompt).input_ids,
        teacher_argmax=ord(generated),
        text_before=prompt,
        text_after=prompt + generated,
        ctx=_ctx("block", "block"),
        language="python",
        prompt_len=len(tok(prompt).input_ids),
        reference=reference,
        prompt_text=prompt,
    )
    proposal = TransPLDProposer(
        tok,
        max_matching_ngram_size=20,
        max_draft_len=8,
        transformed_min_matching_ngram_size=3,
    ).propose(state)
    assert proposal is None or proposal.match_kind != "transpld_vref"


def test_routed_transpld_uses_exact_pld_for_prompt_only_safe_routes():
    tok = FakeTokenizer()
    state = ProposerState(
        prefix=[1, 2, 3, 4, 9, 8, 1, 2, 3],
        teacher_argmax=4,
        text_before="",
        text_after="",
        ctx=None,
        prompt_len=7,
    )
    proposer = RoutedTransPLDProposer(tok, max_matching_ngram_size=4, max_draft_len=2)
    proposal = proposer.propose(state)
    assert proposal is not None
    assert proposal.kind == "routed_transpld"
    assert proposal.match_kind == "routed_exact_pld"
    assert proposal.route == "exact_pld"
    assert proposal.route_reason == "no_reference"
    assert proposal.tokens == [9, 8]

    proposer.reset()
    poisoned_metadata_state = ProposerState(
        prefix=[1, 2, 3, 4, 9, 8, 1, 2, 3],
        teacher_argmax=4,
        text_before="",
        text_after="",
        ctx=None,
        prompt_len=7,
        reference="x = user\n",
        metadata={
            "target_is_reference": True,
            "rewrite_pairs": {"user": "account"},
            "gold": "x = account\n",
            "label": "synthetic_field",
            "manifest_only_field": "must_not_route",
        },
    )
    proposal = proposer.propose(poisoned_metadata_state)
    assert proposal is not None
    assert proposal.match_kind == "routed_exact_pld"
    assert proposal.route_reason == "no_rewrite_map"

    proposer.reset()
    no_effect_prompt = "Rename missing to absent.\n```python\nx = user\n```\n"
    no_effect_state = ProposerState(
        prefix=[1, 2, 3, 4, 9, 8, 1, 2, 3],
        teacher_argmax=4,
        text_before=no_effect_prompt,
        text_after=no_effect_prompt,
        ctx=None,
        prompt_len=len(tok(no_effect_prompt).input_ids),
        reference="x = user\n",
        metadata={"rewrite_pairs": {"user": "account"}},
    )
    proposal = proposer.propose(no_effect_state)
    assert proposal is not None
    assert proposal.match_kind == "routed_exact_pld"
    assert proposal.route_reason == "rewrite_map_no_effect"


def test_routed_transpld_uses_rewrite_map_and_backs_off():
    tok = FakeTokenizer()
    reference = "return user.name\n"
    prompt = "Rename user to account.\n```python\n" + reference + "```\n"
    generated = "return account"
    state = ProposerState(
        prefix=tok(prompt + generated[:-1]).input_ids,
        teacher_argmax=ord(generated[-1]),
        text_before=prompt + generated[:-1],
        text_after=prompt + generated,
        ctx=_ctx("block", "block"),
        language="python",
        prompt_len=len(tok(prompt).input_ids),
        reference=reference,
    )
    proposer = RoutedTransPLDProposer(
        tok,
        max_matching_ngram_size=20,
        max_draft_len=8,
        backoff_after_steps=2,
        min_rewrite_hits=2,
        min_accept_per_rewrite_hit=4.0,
    )
    proposal = proposer.propose(state)
    assert proposal is not None
    assert proposal.kind == "routed_transpld"
    assert proposal.match_kind == "routed_transpld_vref"
    assert proposal.route == "transpld"
    assert proposal.route_reason == "prompt_rewrite_map"
    assert tok.decode(proposal.tokens).startswith(".name")

    for _ in range(2):
        proposer.observe(
            ProposalFeedback(
                prefix_start=0,
                prefix_end=0,
                proposed_tokens=[],
                emitted_tokens=[],
                accepted_nonroot=0,
                rejected=False,
                proposal_kind="routed_transpld",
                proposal_match_kind="routed_transpld_miss_exact_pld",
            )
        )
    fallback_state = ProposerState(
        prefix=[1, 2, 3, 4, 9, 8, 1, 2, 3],
        teacher_argmax=4,
        text_before=prompt,
        text_after=prompt,
        ctx=None,
        prompt_len=len(tok(prompt).input_ids),
        reference=reference,
    )
    fallback = proposer.propose(fallback_state)
    assert fallback is not None
    assert fallback.match_kind == "routed_backoff_exact_pld"
    assert fallback.backoff_active is True
    assert fallback.route_reason == "online_backoff_low_acceptance"


def test_routed_transpld_backs_off_after_low_accept_probe_streak():
    tok = FakeTokenizer()
    reference = "return user.name\n"
    prompt = "Rename user to account.\n```python\n" + reference + "```\n"
    proposer = RoutedTransPLDProposer(
        tok,
        max_matching_ngram_size=20,
        max_draft_len=8,
        backoff_after_steps=99,
        probe_backoff_after_attempts=3,
        min_accept_per_rewrite_attempt=3.0,
        low_accept_streak_limit=3,
        low_accept_streak_threshold=2,
    )
    state = ProposerState(
        prefix=tok(prompt + "return accoun").input_ids,
        teacher_argmax=ord("t"),
        text_before=prompt + "return accoun",
        text_after=prompt + "return account",
        ctx=_ctx("block", "block"),
        language="python",
        prompt_len=len(tok(prompt).input_ids),
        reference=reference,
    )
    first = proposer.propose(state)
    assert first is not None
    assert first.route == "transpld"

    for _ in range(3):
        proposer.observe(
            ProposalFeedback(
                prefix_start=0,
                prefix_end=0,
                proposed_tokens=[1],
                emitted_tokens=[1],
                accepted_nonroot=1,
                rejected=False,
                proposal_kind="routed_transpld",
                proposal_match_kind="routed_transpld_vref",
            )
        )

    fallback = proposer.propose(
        ProposerState(
            prefix=[1, 2, 3, 4, 9, 8, 1, 2, 3],
            teacher_argmax=4,
            text_before=prompt,
            text_after=prompt,
            ctx=None,
            prompt_len=len(tok(prompt).input_ids),
            reference=reference,
        )
    )
    assert fallback is not None
    assert fallback.route == "exact_pld"
    assert fallback.backoff_active is True
    assert fallback.match_kind == "routed_backoff_exact_pld"


def test_routed_transpld_zero_accept_tripwire_disables_transpld():
    tok = FakeTokenizer()
    reference = "return user.name\n"
    prompt = "Rename user to account.\n```python\n" + reference + "```\n"
    proposer = RoutedTransPLDProposer(
        tok,
        max_matching_ngram_size=20,
        max_draft_len=8,
        backoff_after_steps=99,
        probe_backoff_after_attempts=99,
        low_accept_streak_limit=99,
        zero_accept_tripwire_limit=3,
    )
    state = ProposerState(
        prefix=tok(prompt + "return accoun").input_ids,
        teacher_argmax=ord("t"),
        text_before=prompt + "return accoun",
        text_after=prompt + "return account",
        ctx=_ctx("block", "block"),
        language="python",
        prompt_len=len(tok(prompt).input_ids),
        reference=reference,
    )
    first = proposer.propose(state)
    assert first is not None
    assert first.match_kind == "routed_transpld_vref"

    for _ in range(3):
        proposer.observe(
            ProposalFeedback(
                prefix_start=0,
                prefix_end=0,
                proposed_tokens=[1],
                emitted_tokens=[],
                accepted_nonroot=0,
                rejected=True,
                proposal_kind="routed_transpld",
                proposal_match_kind="routed_transpld_vref",
            )
        )

    fallback = proposer.propose(
        ProposerState(
            prefix=[1, 2, 3, 4, 9, 8, 1, 2, 3],
            teacher_argmax=4,
            text_before=prompt,
            text_after=prompt,
            ctx=None,
            prompt_len=len(tok(prompt).input_ids),
            reference=reference,
        )
    )
    assert fallback is not None
    assert fallback.route == "exact_pld"
    assert fallback.backoff_active is True
    assert fallback.match_kind == "routed_backoff_exact_pld"
    assert fallback.rewrite_zero_accept_streak == 3


def test_simple_adoption_transpld_routes_by_prefix_agreement():
    tok = FakeTokenizer()
    reference = "return user.name\n"
    prompt = "Rename user to account.\n```python\n" + reference + "```\n"
    proposer = SimpleAdoptionTransPLDProposer(
        tok,
        max_matching_ngram_size=20,
        max_draft_len=8,
        min_agreement_observations=1,
    )

    adopted_state = ProposerState(
        prefix=tok(prompt + "return accoun").input_ids,
        teacher_argmax=ord("t"),
        text_before=prompt + "return accoun",
        text_after=prompt + "return account",
        ctx=_ctx("block", "block"),
        language="python",
        prompt_len=len(tok(prompt).input_ids),
        reference=reference,
    )
    adopted = proposer.propose(adopted_state)
    assert adopted is not None
    assert adopted.kind == "adopt_simple_transpld"
    assert adopted.adoption_state == "adopted"
    assert adopted.route in {"transpld", "exact_pld"}

    proposer.reset()
    old_state = ProposerState(
        prefix=tok(prompt + "return use").input_ids,
        teacher_argmax=ord("r"),
        text_before=prompt + "return use",
        text_after=prompt + "return user",
        ctx=_ctx("block", "block"),
        language="python",
        prompt_len=len(tok(prompt).input_ids),
        reference=reference,
    )
    rejected = proposer.propose(old_state)
    assert rejected is not None
    assert rejected.route == "exact_pld"
    assert rejected.adoption_state == "rejected"
    assert rejected.disabled_by_adoption_gate is True


def test_cursor_tracked_transpld_resyncs_then_drafts_from_cursor():
    tok = FakeTokenizer()
    reference = "return user.name\nreturn user.email\n"
    prompt = "Rename user to account.\n```python\n" + reference + "```\n"
    generated = "return account"
    state = ProposerState(
        prefix=tok(prompt + generated[:-1]).input_ids,
        teacher_argmax=ord(generated[-1]),
        text_before=prompt + generated[:-1],
        text_after=prompt + generated,
        ctx=_ctx("block", "block"),
        language="python",
        prompt_len=len(tok(prompt).input_ids),
        reference=reference,
    )
    proposer = CursorTrackedTransPLDProposer(
        tok,
        max_matching_ngram_size=20,
        max_draft_len=16,
        max_cursor_draft_len=16,
        activation_accept=4,
    )
    first = proposer.propose(state)
    assert first is not None
    accepted_text = "t.name\n"
    proposer.observe(
        ProposalFeedback(
            prefix_start=len(state.prefix),
            prefix_end=len(state.prefix) + 1 + len(accepted_text),
            proposed_tokens=first.tokens,
            emitted_tokens=[state.teacher_argmax] + tok(accepted_text).input_ids,
            accepted_nonroot=len(accepted_text),
            rejected=False,
            proposal_kind=first.kind,
            proposal_match_kind=first.match_kind,
        )
    )
    next_state = ProposerState(
        prefix=tok(prompt + "return account.name\nreturn accoun").input_ids,
        teacher_argmax=ord("t"),
        text_before=prompt + "return account.name\nreturn accoun",
        text_after=prompt + "return account.name\nreturn account",
        ctx=_ctx("block", "block"),
        language="python",
        prompt_len=len(tok(prompt).input_ids),
        reference=reference,
    )
    second = proposer.propose(next_state)
    assert second is not None
    assert second.match_kind == "cursor"
    assert second.cursor_pos is not None


def test_transpld_inferonly_mines_hidden_identifier_map():
    tok = FakeTokenizer()
    reference = "return user.name\nprint(user.name)\nreturn user.email\n"
    prompt = "Apply the requested edit.\n```python\n" + reference + "```\n"
    generated = "return account.name\nprint(account.name)\nreturn account"
    state = ProposerState(
        prefix=tok(prompt + generated[:-1]).input_ids,
        teacher_argmax=ord(generated[-1]),
        text_before=prompt + generated[:-1],
        text_after=prompt + generated,
        ctx=_ctx("block", "block"),
        language="python",
        prompt_len=len(tok(prompt).input_ids),
        reference=reference,
    )
    proposal = TransPLDInferenceProposer(
        tok,
        infer_only=True,
        max_matching_ngram_size=20,
        max_draft_len=16,
    ).propose(state)
    assert proposal is not None
    assert proposal.map_source == "inferred"
    assert proposal.inferred_map_count >= 1
    assert tok.decode(proposal.tokens).startswith(".email")


def test_compound_transpld_records_multiple_active_maps():
    tok = FakeTokenizer()
    reference = "return user.name or timeout\n"
    prompt = (
        "Rename user to account and replace timeout with deadline.\n"
        "```python\n" + reference + "```\n"
    )
    generated = "return account.name or deadlin"
    state = ProposerState(
        prefix=tok(prompt + generated[:-1]).input_ids,
        teacher_argmax=ord(generated[-1]),
        text_before=prompt + generated[:-1],
        text_after=prompt + generated,
        ctx=_ctx("block", "block"),
        language="python",
        prompt_len=len(tok(prompt).input_ids),
        reference=reference,
    )
    proposal = CompoundTransPLDProposer(tok, max_matching_ngram_size=40, max_draft_len=16).propose(state)
    assert proposal is not None
    assert proposal.kind == "transpld_compound"
    assert proposal.compound_view_count is not None
    assert proposal.active_map_count is not None
    assert proposal.active_map_count >= 1


def test_symbol_tree_proposes_scope_identifier_branches():
    tok = FakeTokenizer()
    text = "def f(alpha_value, beta_count):\n    result_total = alpha_value\n    res"
    state = _state(text, "u", _ctx())
    proposal = SymbolTreeProposer(tok, branch_budget=3, max_tree_nodes=12).propose(state)
    assert proposal is not None
    assert proposal.pool == "scope"
    assert proposal.source_region == "symbol_table"
    assert tok.decode(proposal.tokens).startswith("lt_total")


def test_symbol_tree_member_access_prefers_attribute_symbols():
    tok = FakeTokenizer()
    text = "self.output_path = path\nself.output_count = 0\nself."
    state = _state(text, "o", _ctx("member_expression", "attribute"))
    proposal = SymbolTreeProposer(tok, branch_budget=3, max_tree_nodes=12).propose(state)
    assert proposal is not None
    assert tok.decode(proposal.tokens).startswith("utput_")


def test_edit_symbol_stack_prioritizes_edit_anchor_over_suffix_and_symbol():
    tok = FakeTokenizer()
    prompt = (
        "Please rewrite this code.\n"
        "```python\n"
        "def foo():\n"
        "    value = 1\n"
        "    return value\n"
        "```\n"
    )
    generated = "def foo():\n    value "
    state = ProposerState(
        prefix=tok(prompt + generated[:-1]).input_ids,
        teacher_argmax=ord(generated[-1]),
        text_before=prompt + generated[:-1],
        text_after=prompt + generated,
        ctx=_ctx("identifier", "identifier"),
        language="python",
        prompt_len=len(tok(prompt).input_ids),
    )
    proposal = CheapProposerStack(
        [EditAnchorProposer(tok, min_anchor_chars=8), LocalSuffixProposer(), SymbolTreeProposer(tok)]
    ).propose(state)
    assert proposal is not None
    assert proposal.kind == "edit_anchor"


def test_alpha_suffix_instantiates_aligned_identifier_substitutions():
    tok = FakeTokenizer()
    prefix = "for i in nums:\n    nums[i]\nfor j in vals"
    state = _state(prefix, ":", _ctx("block", "block"))
    proposal = AlphaSuffixProposer(
        tok,
        min_match_len=6,
        max_query_len=24,
        max_draft_len=32,
        enable_roles=False,
        normalize_literals=True,
        enable_substitution=True,
        scope_fill=False,
        filter_exact=True,
    ).propose(state)
    assert proposal is not None
    assert proposal.kind == "alpha_role"
    assert proposal.root_included is False
    assert proposal.match_kind in {"alpha_idlit_id", "alpha_id_id", "alpha_role_id"}
    assert proposal.substitution_count == 2
    assert proposal.follow_start_token is not None
    assert proposal.follow_end_token is not None
    assert proposal.follow_end_token >= proposal.follow_start_token
    assert tok.decode(proposal.tokens).startswith("\n    vals[j]")


def test_alpha_suffix_stops_before_unmapped_identifier_without_scope_fill():
    tok = FakeTokenizer()
    prefix = "for i in nums:\n    total += nums[i]\nfor j in vals"
    state = _state(prefix, ":", _ctx("block", "block"))
    proposal = AlphaSuffixProposer(
        tok,
        min_match_len=6,
        max_query_len=24,
        max_draft_len=32,
        enable_roles=False,
        normalize_literals=True,
        enable_substitution=True,
        scope_fill=False,
        filter_exact=True,
        stop_on_unmapped=True,
    ).propose(state)
    assert proposal is not None
    assert tok.decode(proposal.tokens) == "\n    "
    assert proposal.stopped_on_unmapped is True


def test_alpha_suffix_filters_exact_token_matches_for_alpha_only():
    tok = FakeTokenizer()
    prefix = "for i in nums:\n    nums[i]\nfor i in nums"
    state = _state(prefix, ":", _ctx("block", "block"))
    proposal = AlphaSuffixProposer(
        tok,
        min_match_len=6,
        max_query_len=24,
        max_draft_len=32,
        enable_roles=False,
        normalize_literals=True,
        enable_substitution=True,
        scope_fill=False,
        filter_exact=True,
    ).propose(state)
    assert proposal is None


def test_macro_chunk_proposer_matches_chunk_prefix_in_token_space():
    tok = FakeTokenizer()
    state = _state("if ok", ":", _ctx("block", "block"))
    proposal = MacroChunkProposer(tok, [":\n    return "], max_draft_len=32).propose(state)
    assert proposal is not None
    assert proposal.kind == "macro_static"
    assert tok.decode(proposal.tokens) == "\n    return "


def test_proposer_stack_prefers_highest_scored_candidate():
    class Low:
        kind = "low"

        def propose(self, state):
            return Proposal("low", [1], 1, 0.5)

    class High:
        kind = "high"

        def propose(self, state):
            return Proposal("high", [2], 1, 2.0)

    proposal = CheapProposerStack([Low(), High()]).propose(
        ProposerState([], 0, "", "", None)
    )
    assert proposal is not None
    assert proposal.kind == "high"


def test_identifier_proposer_returns_none_outside_identifier():
    tok = FakeTokenizer()
    state = _state("total = 1 + ", "2", _ctx("binary_operator", "integer"))
    assert IdentifierTrieProposer(tok).propose(state) is None


def test_humaneval_prompt_variants_remove_examples():
    prompt = 'from typing import List\n\ndef f(x):\n    """Return x.\n    >>> f(1)\n    1\n    assert f(2) == 2\n    """\n'
    no_examples = _transform_humaneval_prompt(prompt, "no_examples")
    assert ">>> f" not in no_examples
    assert "assert f" not in no_examples
    assert "Return x" in no_examples
    signature = _transform_humaneval_prompt(prompt, "signature_only")
    assert signature.endswith("def f(x):\n")
    assert "Return x" not in signature


def test_mbpp_desc_only_removes_assertion():
    full = _mbpp_to_completion_prompt("Find x.", ["assert foo(1) == 2"], prompt_variant="full")
    desc_only = _mbpp_to_completion_prompt("Find x.", ["assert foo(1) == 2"], prompt_variant="desc_only")
    assert "assert foo" in full
    assert "assert foo" not in desc_only


def test_codeeditor_switch_prompt_uses_reference_not_target_solution():
    row = {
        "idx": 7,
        "language": "python",
        "similar_content": "Return the sum.",
        "target_content": "Return the product.",
        "similar_source_code": (
            "def solve(a, b):\n"
            "    value = a + b\n"
            "    if value < 0:\n"
            "        value = -value\n"
            "    return value\n"
        ),
        "target_source_code": "def solve(a, b):\n    return a * b\n",
    }
    problem = _codeeditor_switch_problem_from_row(row, 0, "fixture")
    assert problem is not None
    assert "Return the product." in problem.prompt
    assert "value = a + b" in problem.prompt
    assert "return a * b" not in problem.prompt
    assert problem.language == "codeeditor_switch_python"


def test_repo_edit_rename_prompt_never_includes_corrected_target():
    source = (
        "def solve(value, items):\n"
        "    total = value\n"
        "    for item in items:\n"
        "        total += value + item\n"
        "    if total > value:\n"
        "        total -= value\n"
        "    return total\n"
    )
    problem = _repo_edit_rename_problem_from_source(source, 0, "fixture")
    assert problem is not None
    assert "Rename `value` to `value_updated`" in problem.prompt
    assert "value_updated" in problem.prompt
    assert "total = value" in problem.prompt
    assert "total = value_updated" not in problem.prompt
    assert problem.language == "repo_edit_rename_python"
