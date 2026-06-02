import torch

from asts.selective_lm_head import (
    certify_or_fallback_argmax,
    contiguous_lm_head_clusters,
)


def test_cluster_upper_bound_is_conservative():
    weight = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [-1.0, 0.0],
            [-0.8, -0.2],
        ]
    )
    clusters = contiguous_lm_head_clusters(weight, 2)
    h = torch.tensor([0.5, 2.0])
    for cluster_id, ids in enumerate(clusters.token_ids):
        upper = torch.dot(h, clusters.centroids[cluster_id]) + torch.linalg.vector_norm(h) * clusters.radii[cluster_id]
        exact = weight.index_select(0, ids).matmul(h).max()
        assert upper + 1e-6 >= exact


def test_certification_accepts_only_safe_argmax():
    weight = torch.tensor(
        [
            [3.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    clusters = contiguous_lm_head_clusters(weight, 4)
    res = certify_or_fallback_argmax(torch.tensor([1.0, 0.1]), 0, weight, clusters)
    assert res.certified
    assert res.selected_token == 0
    assert not res.full_fallback


def test_fallback_preserves_baseline_argmax_when_draft_wrong():
    weight = torch.tensor(
        [
            [0.0, 1.0],
            [2.0, 0.0],
            [1.0, 0.0],
        ]
    )
    clusters = contiguous_lm_head_clusters(weight, 3)
    h = torch.tensor([1.0, 0.0])
    res = certify_or_fallback_argmax(h, 0, weight, clusters)
    assert not res.certified
    assert res.full_fallback
    assert res.selected_token == int(weight.matmul(h).argmax().item())


def test_near_tie_falls_back():
    weight = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    clusters = contiguous_lm_head_clusters(weight, 3)
    res = certify_or_fallback_argmax(torch.tensor([1.0, 0.0]), 1, weight, clusters)
    assert not res.certified
    assert res.full_fallback
    assert res.selected_token == 0
