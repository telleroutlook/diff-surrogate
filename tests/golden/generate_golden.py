"""Generate golden numerical contracts for geometry operators.

Creates ``tests/golden/*.npz`` files containing input geometry, expected SDF,
projection, and gradient values.  Used by downstream repos (DiffCFD, DiffNano)
to verify their implementations align, and by ``test_golden_contract.py`` for
CI regression detection.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from diff_surrogate.geometry import (
    eval_closed_cubic_bspline,
    heaviside_projection,
    sdf_from_curve,
    sigmoid_projection,
)

GOLDEN_DIR = Path(__file__).parent


def _circle_points(n, radius=1.0, dtype=torch.float64):
    angles = torch.linspace(0, 2 * math.pi, n + 1, dtype=dtype)[:-1]
    return torch.stack(
        [radius * torch.cos(angles), radius * torch.sin(angles)],
        dim=-1,
    )


def generate_circle_golden():
    """Golden contract for a circle SDF (float32, legacy)."""
    cp = _circle_points(16, radius=1.0, dtype=torch.float32)
    H, W = 32, 32
    grid_x = torch.linspace(-2, 2, W).unsqueeze(0).expand(H, W)
    grid_y = torch.linspace(-2, 2, H).unsqueeze(1).expand(H, W)

    t = torch.linspace(0, 1, 80)
    curve = eval_closed_cubic_bspline(cp, t)
    sdf = sdf_from_curve(grid_x, grid_y, curve)

    cp_grad = cp.detach().clone().requires_grad_(True)
    curve_g = eval_closed_cubic_bspline(cp_grad, t)
    sdf_g = sdf_from_curve(grid_x, grid_y, curve_g)
    loss = sdf_g.sum()
    loss.backward()

    np.savez(
        GOLDEN_DIR / "sdf_circle.npz",
        control_points=cp.numpy(),
        grid_x=grid_x.numpy(),
        grid_y=grid_y.numpy(),
        curve_points=curve.detach().numpy(),
        sdf=sdf.detach().numpy(),
        grad_control_points=cp_grad.grad.numpy(),
    )


def generate_bspline_golden():
    """Golden contract for a non-trivial B-spline shape (float64)."""
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

    cp_grad = cp.detach().clone().requires_grad_(True)
    curve_g = eval_closed_cubic_bspline(cp_grad, t)
    sdf_g = sdf_from_curve(grid_x, grid_y, curve_g)
    density_g = sigmoid_projection(sdf_g, beta=10.0)
    loss = density_g.sum()
    loss.backward()

    np.savez(
        GOLDEN_DIR / "sdf_bspline.npz",
        control_points=cp.numpy(),
        grid_x=grid_x.numpy(),
        grid_y=grid_y.numpy(),
        curve_points=curve.detach().numpy(),
        sdf=sdf.detach().numpy(),
        density=density.detach().numpy(),
        grad_control_points=cp_grad.grad.numpy(),
    )


def generate_polygon_golden():
    """Golden contract for polygon-vertex SDF (Rust cross-validation).

    Uses raw polygon vertices (NOT B-spline evaluated points) so the Rust
    ``bspline_sdf`` polygon-edge-distance + ray-casting winding number can be
    directly validated.  High softmin_temp and winding_sharpness approximate
    the exact polygon SDF.
    """
    verts = _circle_points(16, radius=1.0, dtype=torch.float64)
    H, W = 32, 32
    grid_x = torch.linspace(-2, 2, W, dtype=torch.float64).unsqueeze(0).expand(H, W)
    grid_y = torch.linspace(-2, 2, H, dtype=torch.float64).unsqueeze(1).expand(H, W)

    sdf = sdf_from_curve(
        grid_x,
        grid_y,
        verts,
        softmin_temp=500.0,
        winding_sharpness=100.0,
    )

    verts_grad = verts.detach().clone().requires_grad_(True)
    sdf_g = sdf_from_curve(
        grid_x,
        grid_y,
        verts_grad,
        softmin_temp=500.0,
        winding_sharpness=100.0,
    )
    sdf_g.sum().backward()

    vx = verts[:, 0].numpy()
    vy = verts[:, 1].numpy()

    np.savez(
        GOLDEN_DIR / "polygon_sdf.npz",
        polygon_vx=vx,
        polygon_vy=vy,
        grid_x=grid_x.numpy(),
        grid_y=grid_y.numpy(),
        sdf=sdf.detach().numpy(),
        grad_polygon_vx=verts_grad.grad[:, 0].numpy(),
        grad_polygon_vy=verts_grad.grad[:, 1].numpy(),
    )


def generate_pipeline_golden():
    """Golden contract for full B-spline -> SDF -> projection pipeline.

    Tests multiple projection types (sigmoid, heaviside) and beta values
    (5.0, 10.0, 40.0) with gradients, all in float64.
    """
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

    betas = [5.0, 10.0, 40.0]
    projections = {
        "sigmoid": sigmoid_projection,
        "heaviside": heaviside_projection,
    }

    density_results = {}
    grad_results = {}

    for proj_name, proj_fn in projections.items():
        for beta in betas:
            density = proj_fn(sdf, beta=beta)
            density_results[f"{proj_name}_beta{int(beta)}"] = density.detach().numpy()

            cp_grad = cp.detach().clone().requires_grad_(True)
            curve_g = eval_closed_cubic_bspline(cp_grad, t)
            sdf_g = sdf_from_curve(grid_x, grid_y, curve_g)
            d_g = proj_fn(sdf_g, beta=beta)
            d_g.sum().backward()
            grad_results[f"grad_{proj_name}_beta{int(beta)}"] = cp_grad.grad.numpy()

    np.savez(
        GOLDEN_DIR / "pipeline.npz",
        control_points=cp.numpy(),
        grid_x=grid_x.numpy(),
        grid_y=grid_y.numpy(),
        curve_points=curve.detach().numpy(),
        sdf=sdf.detach().numpy(),
        **density_results,
        **grad_results,
    )


if __name__ == "__main__":
    generate_circle_golden()
    generate_bspline_golden()
    generate_polygon_golden()
    generate_pipeline_golden()
    print("Golden contracts generated in tests/golden/")
