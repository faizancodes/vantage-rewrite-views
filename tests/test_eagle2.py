"""Unit tests for the EAGLE-2 tree builder, mask, and accept logic.

These tests don't require GPU or trained EAGLE weights — they verify the
combinatorial correctness of tree shape, mask, and acceptance.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from asts.eagle2_decoder import (
    TreeNode,
    build_tree_mask,
    build_tree_positions,
    find_longest_accepted_path,
    _ancestors_including_self,
)


def _make_tree(specs: list[tuple[int, int, int, float]]) -> list[TreeNode]:
    """specs: list of (parent_idx, depth, token_id, score)."""
    return [TreeNode(parent_idx=p, depth=d, token_id=t, score=s) for (p, d, t, s) in specs]


# ---------------------------------------------------------------------------
# Ancestor tracing
# ---------------------------------------------------------------------------


def test_ancestors_root_only():
    tree = _make_tree([(-1, 0, 100, 0.0)])
    assert _ancestors_including_self(tree, 0) == [0]


def test_ancestors_chain():
    # root -> A -> B -> C
    tree = _make_tree([
        (-1, 0, 100, 0.0),
        (0, 1, 200, -1.0),
        (1, 2, 300, -2.0),
        (2, 3, 400, -3.0),
    ])
    assert _ancestors_including_self(tree, 3) == [3, 2, 1, 0]
    assert _ancestors_including_self(tree, 1) == [1, 0]


def test_ancestors_branching():
    # root -> {A, B}; A -> C; B -> D
    tree = _make_tree([
        (-1, 0, 100, 0.0),
        (0, 1, 200, -1.0),  # A
        (0, 1, 201, -1.5),  # B
        (1, 2, 300, -2.0),  # C (parent A)
        (2, 2, 301, -2.5),  # D (parent B)
    ])
    assert _ancestors_including_self(tree, 3) == [3, 1, 0]  # C through A
    assert _ancestors_including_self(tree, 4) == [4, 2, 0]  # D through B


# ---------------------------------------------------------------------------
# Tree mask correctness
# ---------------------------------------------------------------------------


def _is_visible(mask: torch.Tensor, q: int, k: int) -> bool:
    """mask is [1, 1, N, P+N] additive; visible means value == 0 (not -inf)."""
    return float(mask[0, 0, q, k].item()) == 0.0


def test_mask_root_only():
    """Tree of just the root: root attends to all prefix + itself."""
    P = 5
    tree = _make_tree([(-1, 0, 100, 0.0)])
    mask = build_tree_mask(tree, P=P, dtype=torch.float32, device=torch.device("cpu"))
    assert mask.shape == (1, 1, 1, P + 1)
    # Prefix positions all visible
    for k in range(P):
        assert _is_visible(mask, 0, k)
    # Root sees itself
    assert _is_visible(mask, 0, P + 0)


def test_mask_branch_isolation():
    """In a branching tree, each branch is invisible to its siblings."""
    # Tree: root(0) -> A(1), B(2); A(1) -> C(3); B(2) -> D(4)
    # Flat indices match tree-array order.
    P = 3
    tree = _make_tree([
        (-1, 0, 100, 0.0),  # 0 root
        (0, 1, 200, -1.0),  # 1 A
        (0, 1, 201, -1.5),  # 2 B
        (1, 2, 300, -2.0),  # 3 C
        (2, 2, 301, -2.5),  # 4 D
    ])
    mask = build_tree_mask(tree, P=P, dtype=torch.float32, device=torch.device("cpu"))
    assert mask.shape == (1, 1, 5, P + 5)

    # All nodes attend to all prefix positions
    for q in range(5):
        for k in range(P):
            assert _is_visible(mask, q, k), f"q={q} should see prefix k={k}"

    # Root attends only to itself among tree nodes
    assert _is_visible(mask, 0, P + 0)
    for k in range(1, 5):
        assert not _is_visible(mask, 0, P + k), f"root should NOT see tree[{k}]"

    # A attends to {root, A}; not to {B, C, D}
    assert _is_visible(mask, 1, P + 0)
    assert _is_visible(mask, 1, P + 1)
    assert not _is_visible(mask, 1, P + 2)
    assert not _is_visible(mask, 1, P + 3)
    assert not _is_visible(mask, 1, P + 4)

    # B attends to {root, B}; not to A, C, D
    assert _is_visible(mask, 2, P + 0)
    assert not _is_visible(mask, 2, P + 1)
    assert _is_visible(mask, 2, P + 2)
    assert not _is_visible(mask, 2, P + 3)
    assert not _is_visible(mask, 2, P + 4)

    # C attends to {root, A, C}; not to {B, D}
    assert _is_visible(mask, 3, P + 0)
    assert _is_visible(mask, 3, P + 1)
    assert not _is_visible(mask, 3, P + 2)
    assert _is_visible(mask, 3, P + 3)
    assert not _is_visible(mask, 3, P + 4)

    # D attends to {root, B, D}; not to {A, C}
    assert _is_visible(mask, 4, P + 0)
    assert not _is_visible(mask, 4, P + 1)
    assert _is_visible(mask, 4, P + 2)
    assert not _is_visible(mask, 4, P + 3)
    assert _is_visible(mask, 4, P + 4)


def test_mask_dtype_neg_inf():
    """Masked positions should be the dtype's most-negative value."""
    P = 2
    tree = _make_tree([(-1, 0, 100, 0.0), (0, 1, 200, -1.0)])
    mask = build_tree_mask(tree, P=P, dtype=torch.float32, device=torch.device("cpu"))
    # Root should NOT see depth-1 child
    val = float(mask[0, 0, 0, P + 1].item())
    assert val == torch.finfo(torch.float32).min, f"got {val}"


# ---------------------------------------------------------------------------
# Position ids
# ---------------------------------------------------------------------------


def test_position_ids_match_depth():
    P = 7
    tree = _make_tree([
        (-1, 0, 100, 0.0),
        (0, 1, 200, -1.0),
        (0, 1, 201, -1.5),
        (1, 2, 300, -2.0),
    ])
    pos = build_tree_positions(tree, P=P, device=torch.device("cpu"))
    assert pos.shape == (1, 4)
    assert pos[0, 0].item() == P + 0
    assert pos[0, 1].item() == P + 1
    assert pos[0, 2].item() == P + 1  # sibling shares position
    assert pos[0, 3].item() == P + 2


# ---------------------------------------------------------------------------
# Longest accepted path
# ---------------------------------------------------------------------------


def _stub_logits(N: int, vocab: int, target_argmaxes: dict[int, int]) -> torch.Tensor:
    """Build [1, N, V] logits where target's argmax at flat position i is
    target_argmaxes[i]; positions not in the dict get token 0 as argmax."""
    logits = torch.zeros(1, N, vocab)
    for i in range(N):
        argmax = target_argmaxes.get(i, 0)
        logits[0, i, argmax] = 10.0
    return logits


def test_accept_only_root():
    """Tree: root -> A. If target predicts something other than A, only root
    is accepted; bonus = target's argmax at position past root."""
    tree = _make_tree([
        (-1, 0, 100, 0.0),
        (0, 1, 200, -1.0),  # A
    ])
    # Target's argmax at root's flat-idx 0 = 999 (not 200), so A is rejected
    # Bonus = target's argmax at flat-idx 0 = 999
    v_logits = _stub_logits(N=2, vocab=1000, target_argmaxes={0: 999, 1: 5})
    path, depth, bonus = find_longest_accepted_path(tree, v_logits)
    assert path == [0], path
    assert depth == 0
    assert bonus == 999


def test_accept_full_chain():
    """Tree: root -> A -> B -> C. If target's argmaxes match the chain,
    the full chain is accepted and bonus is at the leaf's flat index."""
    tree = _make_tree([
        (-1, 0, 100, 0.0),
        (0, 1, 200, -1.0),
        (1, 2, 300, -2.0),
        (2, 3, 400, -3.0),
    ])
    # Argmax at flat idx 0 = 200 (= A), so A accepted
    # Argmax at flat idx 1 = 300 (= B), so B accepted
    # Argmax at flat idx 2 = 400 (= C), so C accepted
    # Bonus = argmax at flat idx 3 = 555
    v_logits = _stub_logits(N=4, vocab=1000, target_argmaxes={0: 200, 1: 300, 2: 400, 3: 555})
    path, depth, bonus = find_longest_accepted_path(tree, v_logits)
    assert path == [0, 1, 2, 3]
    assert depth == 3
    assert bonus == 555


def test_accept_branch_picks_winning_branch():
    """Tree: root -> A, B; only B has accepted child. The longest accepted path
    should run through B, not A."""
    tree = _make_tree([
        (-1, 0, 100, 0.0),  # 0 root
        (0, 1, 200, -0.5),  # 1 A — score higher
        (0, 1, 201, -1.0),  # 2 B
        (2, 2, 300, -2.0),  # 3 C (child of B)
    ])
    # Target's argmax at root flat-idx 0 = 201 (= B), so A is rejected, B is accepted
    # Target's argmax at flat-idx 2 (B's position) = 300 (= C), so C is accepted
    # Bonus = argmax at flat-idx 3 = 999
    v_logits = _stub_logits(
        N=4, vocab=1000, target_argmaxes={0: 201, 1: 0, 2: 300, 3: 999}
    )
    path, depth, bonus = find_longest_accepted_path(tree, v_logits)
    assert path == [0, 2, 3]
    assert depth == 2
    assert bonus == 999


def test_accept_partial_chain_with_better_branch():
    """Tree: root -> A -> B (depth 2), root -> C (depth 1). If A is rejected
    but C is accepted, we should pick the C path (longer accepted than nothing).

    This verifies we find the longest accepted path, not just the first."""
    tree = _make_tree([
        (-1, 0, 100, 0.0),  # 0 root
        (0, 1, 200, -0.5),  # 1 A (rejected)
        (1, 2, 300, -1.5),  # 2 B (under rejected A, so unreachable)
        (0, 1, 201, -1.0),  # 3 C (accepted)
    ])
    # Argmax at root = 201 (= C, not A)
    # Argmax at 1 (A) = 0 (B is unreachable since A rejected)
    # Argmax at 3 (C) = 555 (bonus)
    v_logits = _stub_logits(
        N=4, vocab=1000, target_argmaxes={0: 201, 1: 0, 2: 0, 3: 555}
    )
    path, depth, bonus = find_longest_accepted_path(tree, v_logits)
    assert path == [0, 3]
    assert depth == 1
    assert bonus == 555


def test_accept_root_only_when_no_branches():
    """Tree of just the root. Path is [0], depth 0, bonus from flat-idx 0."""
    tree = _make_tree([(-1, 0, 100, 0.0)])
    v_logits = _stub_logits(N=1, vocab=1000, target_argmaxes={0: 777})
    path, depth, bonus = find_longest_accepted_path(tree, v_logits)
    assert path == [0]
    assert depth == 0
    assert bonus == 777
