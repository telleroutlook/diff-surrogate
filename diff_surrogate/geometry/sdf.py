"""Differentiable signed distance field from curve points.

Computes the minimum distance from a grid to a set of curve points with a
differentiable winding number for inside/outside sign determination.

Convention: negative inside, positive outside.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["differentiable_winding_number", "sdf_from_curve"]


def differentiable_winding_number(
    grid_x: Tensor,
    grid_y: Tensor,
    curve_points: Tensor,
) -> Tensor:
    """Compute a differentiable winding number field.

    Parameters
    ----------
    grid_x : Tensor, shape ``(H, W)``
        X coordinates of the evaluation grid.
    grid_y : Tensor, shape ``(H, W)``
        Y coordinates of the evaluation grid.
    curve_points : Tensor, shape ``(K, 2)``
        Ordered curve points forming a closed polygon / spline.

    Returns
    -------
    winding : Tensor, shape ``(H, W)``
        Winding number field.  Integer values near solid bodies; smoothly
        differentiable via ``atan2``.
    """
    # (H, W, K)
    dx = curve_points[:, 0].unsqueeze(0).unsqueeze(0) - grid_x.unsqueeze(-1)
    dy = curve_points[:, 1].unsqueeze(0).unsqueeze(0) - grid_y.unsqueeze(-1)

    angles = torch.atan2(dy, dx)
    angle_diff = angles[..., 1:] - angles[..., :-1]
    angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))

    winding_sum = angle_diff.sum(dim=-1)
    return winding_sum / (2 * math.pi)


def sdf_from_curve(
    grid_x: Tensor,
    grid_y: Tensor,
    curve_points: Tensor,
    *,
    softmin_temp: float = 10.0,
    winding_sharpness: float = 20.0,
) -> Tensor:
    """Compute a differentiable SDF from curve points.

    Uses soft-min aggregation over distances for differentiable minimum
    selection, and the winding number for inside/outside determination.

    Parameters
    ----------
    grid_x : Tensor, shape ``(H, W)``
        X coordinates of the evaluation grid.
    grid_y : Tensor, shape ``(H, W)``
        Y coordinates of the evaluation grid.
    curve_points : Tensor, shape ``(K, 2)``
        Ordered curve points forming a closed curve.
    softmin_temp : float
        Temperature for soft-min aggregation.  Higher → sharper min.
    winding_sharpness : float
        Sharpness of the sigmoid that converts winding number to inside mask.

    Returns
    -------
    sdf : Tensor, shape ``(H, W)``
        Signed distance field.  Negative inside, positive outside.

    Notes
    -----
    Convention: ``sdf < 0`` inside the closed curve, ``sdf > 0`` outside.
    This matches the Brinkman penalisation convention used in DiffCFD.
    """
    # Distance computation
    gx = grid_x.unsqueeze(-1)  # (H, W, 1)
    gy = grid_y.unsqueeze(-1)
    cx = curve_points[:, 0].reshape(1, 1, -1)  # (1, 1, K)
    cy = curve_points[:, 1].reshape(1, 1, -1)

    dist_sq = (gx - cx) ** 2 + (gy - cy) ** 2
    dists = torch.sqrt(dist_sq + 1e-12)

    # Soft-min for differentiable aggregation
    weights = torch.softmax(-softmin_temp * dists, dim=-1)
    min_dist = (weights * dists).sum(dim=-1)

    # Winding number for inside/outside
    winding = differentiable_winding_number(grid_x, grid_y, curve_points)
    inside = torch.sigmoid(winding_sharpness * (winding.abs() - 0.5))

    return min_dist * (1 - 2 * inside)
