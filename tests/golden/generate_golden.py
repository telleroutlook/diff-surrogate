"""Generate golden numerical contracts for geometry operators.

Creates ``tests/golden/sdf_circle.npz`` and ``tests/golden/sdf_bspline.npz``
containing input control points and expected SDF + gradient values, used by
downstream repos (DiffCFD, DiffNano) to verify their implementations align.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from diff_surrogate.geometry import (
    eval_closed_cubic_bspline,
    sdf_from_curve,
    sigmoid_projection,
)


def _circle_points(n, radius=1.0):
    angles = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    return torch.stack(
        [
            radius * torch.cos(angles),
            radius * torch.sin(angles),
        ],
        dim=-1,
    )


def generate_circle_golden():
    """Golden contract for a circle SDF."""
    cp = _circle_points(16, radius=1.0)
    H, W = 32, 32
    grid_x = torch.linspace(-2, 2, W).unsqueeze(0).expand(H, W)
    grid_y = torch.linspace(-2, 2, H).unsqueeze(1).expand(H, W)

    t = torch.linspace(0, 1, 80)
    curve = eval_closed_cubic_bspline(cp, t)
    sdf = sdf_from_curve(grid_x, grid_y, curve)

    # Compute gradients w.r.t. control points
    cp_grad = cp.detach().clone().requires_grad_(True)
    curve_g = eval_closed_cubic_bspline(cp_grad, t)
    sdf_g = sdf_from_curve(grid_x, grid_y, curve_g)
    loss = sdf_g.sum()
    loss.backward()

    np.savez(
        Path(__file__).parent / "sdf_circle.npz",
        control_points=cp.numpy(),
        grid_x=grid_x.numpy(),
        grid_y=grid_y.numpy(),
        curve_points=curve.detach().numpy(),
        sdf=sdf.detach().numpy(),
        grad_control_points=cp_grad.grad.numpy(),
    )


def generate_bspline_golden():
    """Golden contract for a non-trivial B-spline shape."""
    cp = torch.tensor(
        [
            [1.0, 0.0],
            [0.3, 0.9],
            [-0.6, 0.7],
            [-1.0, 0.0],
            [-0.4, -0.8],
            [0.5, -0.7],
        ],
        dtype=torch.float64,
    )
    H, W = 24, 24
    grid_x = torch.linspace(-2, 2, W, dtype=torch.float64).unsqueeze(0).expand(H, W)
    grid_y = torch.linspace(-2, 2, H, dtype=torch.float64).unsqueeze(1).expand(H, W)

    t = torch.linspace(0, 1, 60, dtype=torch.float64)
    curve = eval_closed_cubic_bspline(cp, t)
    sdf = sdf_from_curve(grid_x, grid_y, curve)
    density = sigmoid_projection(sdf, beta=10.0)

    # Compute gradients
    cp_grad = cp.detach().clone().requires_grad_(True)
    curve_g = eval_closed_cubic_bspline(cp_grad, t)
    sdf_g = sdf_from_curve(grid_x, grid_y, curve_g)
    density_g = sigmoid_projection(sdf_g, beta=10.0)
    loss = density_g.sum()
    loss.backward()

    np.savez(
        Path(__file__).parent / "sdf_bspline.npz",
        control_points=cp.numpy(),
        grid_x=grid_x.numpy(),
        grid_y=grid_y.numpy(),
        curve_points=curve.detach().numpy(),
        sdf=sdf.detach().numpy(),
        density=density.detach().numpy(),
        grad_control_points=cp_grad.grad.numpy(),
    )


if __name__ == "__main__":
    generate_circle_golden()
    generate_bspline_golden()
    print("Golden contracts generated in tests/golden/")
