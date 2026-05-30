"""Cross-repo golden contract validation tests.

Reloads each golden .npz file, re-runs the identical computation in
diff_surrogate, and verifies bit-exact (or near-exact) agreement.  Any
geometry operator change that perturbs the output is caught by CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from diff_surrogate.geometry import (
    eval_closed_cubic_bspline,
    heaviside_projection,
    sdf_from_curve,
    sigmoid_projection,
)

GOLDEN_DIR = Path(__file__).parent / "golden"
RTOL = 1e-6


def _assert_close(actual: np.ndarray, expected: np.ndarray, label: str):
    denom = np.abs(expected).max() + 1e-30
    rel_err = np.abs(actual - expected) / denom
    max_rel = rel_err.max()
    assert max_rel < RTOL, f"{label}: max relative error {max_rel:.2e} >= {RTOL:.1e}"


def test_circle_golden():
    g = dict(np.load(GOLDEN_DIR / "sdf_circle.npz"))
    cp = torch.from_numpy(g["control_points"])
    grid_x = torch.from_numpy(g["grid_x"])
    grid_y = torch.from_numpy(g["grid_y"])

    t = torch.linspace(0, 1, 80)
    curve = eval_closed_cubic_bspline(cp, t)
    sdf = sdf_from_curve(grid_x, grid_y, curve)

    _assert_close(sdf.detach().numpy(), g["sdf"], "sdf_circle/sdf")
    _assert_close(curve.detach().numpy(), g["curve_points"], "sdf_circle/curve")

    cp_grad = cp.detach().clone().requires_grad_(True)
    curve_g = eval_closed_cubic_bspline(cp_grad, t)
    sdf_g = sdf_from_curve(grid_x, grid_y, curve_g)
    sdf_g.sum().backward()
    _assert_close(cp_grad.grad.numpy(), g["grad_control_points"], "sdf_circle/grad_cp")


def test_bspline_golden():
    g = dict(np.load(GOLDEN_DIR / "sdf_bspline.npz"))
    cp = torch.from_numpy(g["control_points"])
    grid_x = torch.from_numpy(g["grid_x"])
    grid_y = torch.from_numpy(g["grid_y"])

    t = torch.linspace(0, 1, 60, dtype=torch.float64)
    curve = eval_closed_cubic_bspline(cp, t)
    sdf = sdf_from_curve(grid_x, grid_y, curve)
    density = sigmoid_projection(sdf, beta=10.0)

    _assert_close(sdf.detach().numpy(), g["sdf"], "sdf_bspline/sdf")
    _assert_close(density.detach().numpy(), g["density"], "sdf_bspline/density")

    cp_grad = cp.detach().clone().requires_grad_(True)
    curve_g = eval_closed_cubic_bspline(cp_grad, t)
    sdf_g = sdf_from_curve(grid_x, grid_y, curve_g)
    density_g = sigmoid_projection(sdf_g, beta=10.0)
    density_g.sum().backward()
    _assert_close(cp_grad.grad.numpy(), g["grad_control_points"], "sdf_bspline/grad_cp")


def test_polygon_golden():
    g = dict(np.load(GOLDEN_DIR / "polygon_sdf.npz"))
    vx = torch.from_numpy(g["polygon_vx"])
    vy = torch.from_numpy(g["polygon_vy"])
    verts = torch.stack([vx, vy], dim=-1)
    grid_x = torch.from_numpy(g["grid_x"])
    grid_y = torch.from_numpy(g["grid_y"])

    sdf = sdf_from_curve(
        grid_x,
        grid_y,
        verts,
        softmin_temp=500.0,
        winding_sharpness=100.0,
    )
    _assert_close(sdf.detach().numpy(), g["sdf"], "polygon_sdf/sdf")

    verts_grad = verts.detach().clone().requires_grad_(True)
    sdf_g = sdf_from_curve(
        grid_x,
        grid_y,
        verts_grad,
        softmin_temp=500.0,
        winding_sharpness=100.0,
    )
    sdf_g.sum().backward()
    _assert_close(verts_grad.grad[:, 0].numpy(), g["grad_polygon_vx"], "polygon_sdf/grad_vx")
    _assert_close(verts_grad.grad[:, 1].numpy(), g["grad_polygon_vy"], "polygon_sdf/grad_vy")


def test_pipeline_golden():
    g = dict(np.load(GOLDEN_DIR / "pipeline.npz"))
    cp = torch.from_numpy(g["control_points"])
    grid_x = torch.from_numpy(g["grid_x"])
    grid_y = torch.from_numpy(g["grid_y"])

    t = torch.linspace(0, 1, 60, dtype=torch.float64)
    curve = eval_closed_cubic_bspline(cp, t)
    sdf = sdf_from_curve(grid_x, grid_y, curve)

    _assert_close(sdf.detach().numpy(), g["sdf"], "pipeline/sdf")

    betas = [5.0, 10.0, 40.0]
    projections = {
        "sigmoid": sigmoid_projection,
        "heaviside": heaviside_projection,
    }

    for proj_name, proj_fn in projections.items():
        for beta in betas:
            key = f"{proj_name}_beta{int(beta)}"
            density = proj_fn(sdf, beta=beta)
            _assert_close(density.detach().numpy(), g[key], f"pipeline/{key}")

            cp_grad = cp.detach().clone().requires_grad_(True)
            curve_g = eval_closed_cubic_bspline(cp_grad, t)
            sdf_g = sdf_from_curve(grid_x, grid_y, curve_g)
            d_g = proj_fn(sdf_g, beta=beta)
            d_g.sum().backward()
            _assert_close(
                cp_grad.grad.numpy(),
                g[f"grad_{key}"],
                f"pipeline/grad_{key}",
            )
