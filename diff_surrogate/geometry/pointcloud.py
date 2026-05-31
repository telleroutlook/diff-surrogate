"""Multi-scale neighborhood attention encoder for point clouds and irregular meshes.

Provides per-point embeddings via K-nearest-neighbor attention at multiple scales,
and a global geometry encoder that pools point features into a single vector.

Designed to share ``embed_dim`` with the existing SDF trunk so both geometry
representations can feed into the same downstream surrogate head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["IrregularMeshEncoder", "PointCloudGeometry"]

# Default K values for 3 neighborhood scales
_SCALES: tuple[int, ...] = (8, 16, 32)


def _knn_indices(points: Tensor, k: int) -> tuple[Tensor, int]:
    """Return indices of k nearest neighbors for each point.

    Uses ``torch.cdist`` for pairwise distances and ``torch.topk`` for selection,
    both fully differentiable when used inside the attention aggregation.

    When the number of points N is smaller than k, clamps to N (every point
    attends over all others including itself).

    Args:
        points: (B, N, 3) or (N, 3) point coordinates.
        k: Desired number of neighbors.

    Returns:
        indices: (B, N, k_eff) or (N, k_eff) long tensor of neighbor indices.
        k_eff: Effective number of neighbors (may be < k if N < k).
    """
    squeeze = False
    if points.ndim == 2:
        points = points.unsqueeze(0)
        squeeze = True

    N = points.shape[1]
    k_eff = min(k, N)

    # (B, N, N)
    dists = torch.cdist(points, points)
    # topk on negative distance = smallest distances
    _, indices = torch.topk(dists, k_eff, dim=-1, largest=False)

    if squeeze:
        indices = indices.squeeze(0)
    return indices, k_eff


def _gather_neighbors(features: Tensor, indices: Tensor) -> Tensor:
    """Gather neighbor features using index tensor.

    Args:
        features: (B, N, d) point features.
        indices: (B, N, k) neighbor indices.

    Returns:
        neighbor_feats: (B, N, k, d)
    """
    B, N, d = features.shape
    k = indices.shape[-1]
    # Expand indices for gathering along the feature dim
    idx_expanded = indices.unsqueeze(-1).expand(B, N, k, d)
    return features.unsqueeze(2).expand(B, N, k, d).gather(1, idx_expanded)


class _ScaleAttention(nn.Module):
    """Dot-product attention over K nearest neighbors at a single scale.

    For each center point, attends over its K neighbors to produce an updated
    feature vector.

    Args:
        feat_dim: Dimension of per-point input features.
        out_dim: Dimension of output features.
        k: Number of neighbors.
    """

    def __init__(self, feat_dim: int, out_dim: int, k: int):
        super().__init__()
        self.k = k
        self.q_proj = nn.Linear(feat_dim, out_dim)
        self.k_proj = nn.Linear(feat_dim, out_dim)
        self.v_proj = nn.Linear(feat_dim, out_dim)
        self.scale = out_dim**0.5

    def forward(self, points: Tensor, features: Tensor) -> Tensor:
        """Compute neighborhood attention.

        Args:
            points: (B, N, 3) coordinates.
            features: (B, N, d) input features.

        Returns:
            (B, N, out_dim) attention-aggregated features.
        """
        _B, _N, _ = features.shape

        # (B, N, k_eff) neighbor indices; k_eff may be < self.k when N is small
        indices, _k_eff = _knn_indices(points, self.k)

        # (B, N, k, d)
        neighbor_feats = _gather_neighbors(features, indices)

        # Project queries (center) and keys/values (neighbors)
        q = self.q_proj(features).unsqueeze(2)  # (B, N, 1, out_dim)
        k = self.k_proj(neighbor_feats)  # (B, N, k, out_dim)
        v = self.v_proj(neighbor_feats)  # (B, N, k, out_dim)

        # Dot-product attention: (B, N, 1, k)
        attn = (q * k).sum(dim=-1) / self.scale
        attn = torch.softmax(attn, dim=-1)

        # Weighted sum: (B, N, 1, out_dim) -> (B, N, out_dim)
        out = (attn.unsqueeze(-1) * v).sum(dim=2)
        return out


class PointCloudGeometry(nn.Module):
    """Multi-scale neighborhood attention encoder for point clouds.

    At each of 3 scales (K=8, 16, 32), finds K nearest neighbors and computes
    attention-weighted feature aggregation. The multi-scale features are
    concatenated and projected to ``embed_dim``.

    Args:
        in_dim: Dimension of optional per-point input features (0 = coords only).
        embed_dim: Output embedding dimension per point. Must match SDF trunk
            hidden_dim for compatibility.
        scales: Tuple of K values for each neighborhood scale.
    """

    def __init__(
        self,
        in_dim: int = 0,
        embed_dim: int = 128,
        scales: tuple[int, ...] = _SCALES,
    ):
        super().__init__()
        self.scales = scales
        self.embed_dim = embed_dim

        # Input projection: coordinates (3) + optional features (in_dim)
        feat_dim = 3 + in_dim
        self.input_proj = nn.Linear(feat_dim, embed_dim)

        # One attention block per scale
        self.scale_attns = nn.ModuleList([_ScaleAttention(embed_dim, embed_dim, k) for k in scales])

        # Project concatenated multi-scale features to output dim
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim * len(scales), embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, points: Tensor, features: Tensor | None = None) -> Tensor:
        """Encode point cloud into per-point embeddings.

        Args:
            points: (N, 3) or (B, N, 3) point coordinates.
            features: (N, d) or (B, N, d) optional per-point features.

        Returns:
            (N, embed_dim) or (B, N, embed_dim) per-point embeddings.
        """
        squeeze = False
        if points.ndim == 2:
            points = points.unsqueeze(0)
            squeeze = True

        _B, _N, _ = points.shape

        # Build input features: coordinates + optional features
        if features is None:
            inp = points  # (B, N, 3)
        else:
            if features.ndim == 2:
                features = features.unsqueeze(0)
            inp = torch.cat([points, features], dim=-1)

        # Project to embed_dim
        x = torch.relu(self.input_proj(inp))

        # Multi-scale attention: each produces (B, N, embed_dim)
        scale_outs = [attn(points, x) for attn in self.scale_attns]

        # Concatenate and project
        multi_scale = torch.cat(scale_outs, dim=-1)  # (B, N, embed_dim * n_scales)
        out = self.out_proj(multi_scale)  # (B, N, embed_dim)

        if squeeze:
            out = out.squeeze(0)
        return out


class IrregularMeshEncoder(nn.Module):
    """Global geometry encoder for point clouds and irregular meshes.

    Uses :class:`PointCloudGeometry` as a backbone to produce per-point
    embeddings, then pools them into a single global geometry vector via
    attention-weighted pooling.

    Args:
        in_dim: Dimension of optional per-point input features.
        embed_dim: Output embedding dimension. Share with SDF trunk hidden_dim.
        scales: Tuple of K values for each neighborhood scale.
    """

    def __init__(
        self,
        in_dim: int = 0,
        embed_dim: int = 128,
        scales: tuple[int, ...] = _SCALES,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.pc_encoder = PointCloudGeometry(in_dim=in_dim, embed_dim=embed_dim, scales=scales)

        # Learnable pooling query for global aggregation
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pool_key = nn.Linear(embed_dim, embed_dim)
        self.pool_value = nn.Linear(embed_dim, embed_dim)

    def forward(self, points: Tensor, features: Tensor | None = None) -> Tensor:
        """Encode point cloud / mesh into a global geometry embedding.

        Args:
            points: (N, 3) or (B, N, 3) point coordinates.
            features: (N, d) or (B, N, d) optional per-point features.

        Returns:
            (embed_dim,) or (B, embed_dim) global geometry embedding.
        """
        squeeze = False
        if points.ndim == 2:
            points = points.unsqueeze(0)
            squeeze = True
            if features is not None and features.ndim == 2:
                features = features.unsqueeze(0)

        B = points.shape[0]

        # Per-point embeddings: (B, N, embed_dim)
        point_feats = self.pc_encoder(points, features)

        # Attention-weighted pooling: query attends over all points
        query = self.pool_query.expand(B, -1, -1)  # (B, 1, embed_dim)
        keys = self.pool_key(point_feats)  # (B, N, embed_dim)
        values = self.pool_value(point_feats)  # (B, N, embed_dim)

        scale = self.embed_dim**0.5
        attn = (query * keys).sum(dim=-1, keepdim=True) / scale  # (B, N, 1)
        attn = torch.softmax(attn, dim=1)  # (B, N, 1)

        global_feat = (attn * values).sum(dim=1)  # (B, embed_dim)

        if squeeze:
            global_feat = global_feat.squeeze(0)
        return global_feat
