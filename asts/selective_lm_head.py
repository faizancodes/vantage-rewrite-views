"""Certified selective LM-head verification utilities.

The verifier here is deliberately conservative.  It may fall back to a full
LM-head argmax often, but it must never certify a drafted token unless the
cluster upper bounds prove no skipped vocabulary row can beat it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class LMHeadClusters:
    centroids: torch.Tensor
    radii: torch.Tensor
    token_ids: list[torch.Tensor]
    token_to_cluster: torch.Tensor
    bias_max: torch.Tensor | None = None

    @property
    def num_clusters(self) -> int:
        return int(self.centroids.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.centroids.shape[1])


@dataclass(frozen=True)
class CertificationResult:
    certified: bool
    selected_token: int
    full_fallback: bool
    risky_cluster_count: int
    risky_token_count: int
    certified_margin: float | None = None


def _as_1d_hidden(hidden: torch.Tensor) -> torch.Tensor:
    if hidden.dim() == 1:
        return hidden
    if hidden.dim() == 2 and hidden.shape[0] == 1:
        return hidden[0]
    raise ValueError(f"expected hidden vector shape [D] or [1,D], got {tuple(hidden.shape)}")


def build_clusters_from_assignment(
    weight: torch.Tensor,
    assignments: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
) -> LMHeadClusters:
    """Build conservative cluster centroids and L2 radii for LM-head rows.

    Args:
        weight: LM-head matrix [vocab, hidden].
        assignments: cluster id per vocab row, shape [vocab].
        bias: optional bias vector [vocab].
    """
    if weight.dim() != 2:
        raise ValueError(f"weight must be [vocab, hidden], got {tuple(weight.shape)}")
    if assignments.dim() != 1 or assignments.shape[0] != weight.shape[0]:
        raise ValueError("assignments must be one cluster id per vocab row")
    work_weight = weight.detach().float().cpu()
    assignments = assignments.detach().long().cpu()
    n_clusters = int(assignments.max().item()) + 1 if assignments.numel() else 0
    if n_clusters <= 0:
        raise ValueError("no clusters")
    centroids: list[torch.Tensor] = []
    radii: list[float] = []
    token_ids: list[torch.Tensor] = []
    bias_max: list[float] | None = [] if bias is not None else None
    bias_cpu = bias.detach().float().cpu() if bias is not None else None
    for cluster_id in range(n_clusters):
        ids = torch.nonzero(assignments == cluster_id, as_tuple=False).flatten().long()
        if ids.numel() == 0:
            raise ValueError(f"empty cluster {cluster_id}")
        rows = work_weight.index_select(0, ids)
        centroid = rows.mean(dim=0)
        radius = torch.linalg.vector_norm(rows - centroid, dim=1).max().item()
        centroids.append(centroid)
        radii.append(float(radius))
        token_ids.append(ids)
        if bias_max is not None and bias_cpu is not None:
            bias_max.append(float(bias_cpu.index_select(0, ids).max().item()))
    return LMHeadClusters(
        centroids=torch.stack(centroids, dim=0),
        radii=torch.tensor(radii, dtype=torch.float32),
        token_ids=token_ids,
        token_to_cluster=assignments,
        bias_max=torch.tensor(bias_max, dtype=torch.float32) if bias_max is not None else None,
    )


def contiguous_lm_head_clusters(
    weight: torch.Tensor,
    num_clusters: int,
    *,
    bias: torch.Tensor | None = None,
) -> LMHeadClusters:
    vocab = int(weight.shape[0])
    if num_clusters <= 0 or num_clusters > vocab:
        raise ValueError(f"invalid num_clusters={num_clusters} for vocab={vocab}")
    assignments = torch.div(
        torch.arange(vocab, dtype=torch.long) * num_clusters,
        vocab,
        rounding_mode="floor",
    )
    return build_clusters_from_assignment(weight, assignments, bias=bias)


def sorted_projection_lm_head_clusters(
    weight: torch.Tensor,
    num_clusters: int,
    *,
    seed: int = 0,
    bias: torch.Tensor | None = None,
) -> LMHeadClusters:
    """Cluster rows by sorting a deterministic random projection.

    This is cheaper than k-means and often tighter than raw contiguous vocab
    blocks.  The certification remains exact because radii are computed from
    the final assigned rows.
    """
    vocab, hidden = weight.shape
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    direction = torch.randn(hidden, generator=gen, dtype=torch.float32)
    direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-12)
    scores = weight.detach().float().cpu().matmul(direction)
    order = torch.argsort(scores)
    assignments = torch.empty(vocab, dtype=torch.long)
    sorted_assignments = torch.div(
        torch.arange(vocab, dtype=torch.long) * num_clusters,
        vocab,
        rounding_mode="floor",
    )
    assignments[order] = sorted_assignments
    return build_clusters_from_assignment(weight, assignments, bias=bias)


def save_lm_head_clusters(path: str, clusters: LMHeadClusters, metadata: dict | None = None) -> None:
    payload = {
        "centroids": clusters.centroids.cpu(),
        "radii": clusters.radii.cpu(),
        "token_ids": [ids.cpu() for ids in clusters.token_ids],
        "token_to_cluster": clusters.token_to_cluster.cpu(),
        "bias_max": clusters.bias_max.cpu() if clusters.bias_max is not None else None,
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_lm_head_clusters(path: str, *, device: torch.device | str = "cpu") -> LMHeadClusters:
    payload = torch.load(path, map_location="cpu")
    return LMHeadClusters(
        centroids=payload["centroids"].to(device),
        radii=payload["radii"].to(device),
        token_ids=[ids.to(device) for ids in payload["token_ids"]],
        token_to_cluster=payload["token_to_cluster"].to(device),
        bias_max=payload["bias_max"].to(device) if payload.get("bias_max") is not None else None,
    )


def certify_or_fallback_argmax(
    hidden: torch.Tensor,
    draft_token: int,
    lm_head_weight: torch.Tensor,
    clusters: LMHeadClusters,
    *,
    bias: torch.Tensor | None = None,
    tie_eps: float = 1e-6,
    bound_slack: float = 1e-3,
) -> CertificationResult:
    """Certify `draft_token` is exact argmax, otherwise compute full fallback.

    The returned `selected_token` always matches full greedy argmax unless the
    caller supplies non-conservative clusters.  Certification is strict around
    ties; near ties fall back to full vocab.
    """
    h = _as_1d_hidden(hidden).to(lm_head_weight.device)
    weight = lm_head_weight
    draft = int(draft_token)
    if draft < 0 or draft >= int(weight.shape[0]):
        raise ValueError(f"draft token {draft} outside vocab size {weight.shape[0]}")

    h_work = h.to(dtype=weight.dtype)
    draft_logit_t = torch.dot(h_work, weight[draft])
    if bias is not None:
        draft_logit_t = draft_logit_t + bias[draft].to(draft_logit_t.dtype)
    draft_logit = float(draft_logit_t.detach().float().item())

    centroids = clusters.centroids.to(device=weight.device, dtype=h_work.dtype)
    radii = clusters.radii.to(device=weight.device, dtype=torch.float32)
    h_norm = torch.linalg.vector_norm(h_work.float())
    upper = centroids.matmul(h_work).float() + h_norm * radii
    if clusters.bias_max is not None:
        upper = upper + clusters.bias_max.to(upper.device)
    draft_cluster = int(clusters.token_to_cluster[draft].item())
    risky_mask = upper >= (draft_logit - tie_eps - bound_slack)
    risky_mask[draft_cluster] = True
    risky_clusters = torch.nonzero(risky_mask, as_tuple=False).flatten().tolist()
    risky_ids = torch.cat([clusters.token_ids[int(c)].to(weight.device) for c in risky_clusters], dim=0)
    risky_ids = torch.unique(risky_ids, sorted=True)
    risky_weight = weight.index_select(0, risky_ids)
    risky_logits = risky_weight.matmul(h_work).float()
    if bias is not None:
        risky_logits = risky_logits + bias.index_select(0, risky_ids).float()

    draft_positions = torch.nonzero(risky_ids == draft, as_tuple=False).flatten()
    if draft_positions.numel() != 1:
        raise RuntimeError("draft token missing from risky set")
    draft_pos = int(draft_positions[0].item())
    non_draft_logits = torch.cat([risky_logits[:draft_pos], risky_logits[draft_pos + 1 :]])
    max_non_draft = (
        float(non_draft_logits.max().item()) if non_draft_logits.numel() else float("-inf")
    )
    certified_margin = draft_logit - max_non_draft
    # Strictly greater handles PyTorch argmax tie-order conservatively.
    if certified_margin > tie_eps:
        return CertificationResult(
            certified=True,
            selected_token=draft,
            full_fallback=False,
            risky_cluster_count=len(risky_clusters),
            risky_token_count=int(risky_ids.numel()),
            certified_margin=float(certified_margin),
        )

    full_logits = weight.matmul(h_work).float()
    if bias is not None:
        full_logits = full_logits + bias.float()
    return CertificationResult(
        certified=False,
        selected_token=int(full_logits.argmax(dim=-1).item()),
        full_fallback=True,
        risky_cluster_count=len(risky_clusters),
        risky_token_count=int(risky_ids.numel()),
        certified_margin=float(certified_margin),
    )


def accepted_prefix_len(drafts: Sequence[int], predictions: Sequence[int]) -> int:
    accepted = 0
    for draft, pred in zip(drafts, predictions):
        if int(draft) != int(pred):
            break
        accepted += 1
    return accepted
