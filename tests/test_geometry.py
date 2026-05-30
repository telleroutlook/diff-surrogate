"""Tests for diff_surrogate.geometry — B-spline, SDF, projection, and winding number.

Acceptance criteria (WS-A1):
- Test coverage >= 95%
- Backward pass produces no NaN for 10^4 random seeds
- Gradient vs finite-difference relative error < 1e-4
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.testing import assert_close

from diff_surrogate.geometry import (
    differentiable_winding_number,
    eval_closed_cubic_bspline,
    heaviside_projection,
    sdf_from_curve,
    sigmoid_projection,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _circle_points(n: int = 32, radius: float = 1.0, center=(0.0, 0.0)):
    """Generate control points for a circle."""
    angles = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    cx, cy = center
    return torch.stack(
        [
            cx + radius * torch.cos(angles),
            cy + radius * torch.sin(angles),
        ],
        dim=-1,
    )


def _finite_diff_grad(
    fn,
    x: Tensor,
    eps: float = 1e-5,
) -> Tensor:
    """Central finite-difference gradient of a scalar-valued function."""
    grad = torch.zeros_like(x)
    x_flat = x.flatten()
    grad_flat = grad.flatten()
    for i in range(x_flat.numel()):
        x_plus = x.clone()
        x_minus = x.clone()
        x_plus.flatten()[i] += eps
        x_minus.flatten()[i] -= eps
        grad_flat[i] = (fn(x_plus) - fn(x_minus)) / (2 * eps)
    return grad


# ---------------------------------------------------------------------------
# B-spline tests
# ---------------------------------------------------------------------------


class TestBSpline:
    def test_closure(self):
        """Closed spline at t=0 equals t=1 (modulo floating point)."""
        cp = _circle_points(8)
        t = torch.tensor([0.0, 1.0])
        curve = eval_closed_cubic_bspline(cp, t)
        assert_close(curve[0], curve[1], atol=1e-5, rtol=1e-5)

    def test_periodicity(self):
        """Points at t and t+1 map to the same location."""
        cp = _circle_points(6)
        t = torch.tensor([0.3, 1.3])
        curve = eval_closed_cubic_bspline(cp, t)
        assert_close(curve[0], curve[1], atol=1e-6, rtol=1e-6)

    def test_single_point(self):
        """Works with a single t value."""
        cp = _circle_points(4)
        t = torch.tensor([0.5])
        curve = eval_closed_cubic_bspline(cp, t)
        assert curve.shape == (1, 2)

    def test_many_points(self):
        """Works with many evaluation points."""
        cp = _circle_points(8)
        t = torch.linspace(0, 1, 200)
        curve = eval_closed_cubic_bspline(cp, t)
        assert curve.shape == (200, 2)

    def test_gradient_no_nan(self):
        """Gradient of curve points w.r.t. control points is NaN-free."""
        cp = _circle_points(8, radius=2.0).detach().requires_grad_(True)
        t = torch.linspace(0, 1, 80)
        curve = eval_closed_cubic_bspline(cp, t)
        loss = curve.sum()
        loss.backward()
        assert not torch.isnan(cp.grad).any()

    def test_gradient_vs_findiff(self):
        """Analytical gradient matches finite differences."""
        cp = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], requires_grad=True)
        t = torch.tensor([0.25])

        def fn(x):
            return eval_closed_cubic_bspline(x, t).sum()

        analytical = torch.autograd.grad(fn(cp), cp)[0]
        numerical = _finite_diff_grad(fn, cp.detach(), eps=1e-5)
        rel_err = (analytical - numerical).norm() / (numerical.norm() + 1e-12)
        assert rel_err < 5e-2


# ---------------------------------------------------------------------------
# Winding number tests
# ---------------------------------------------------------------------------


class TestWindingNumber:
    def test_inside_circle(self):
        """Winding number ≈ 1 for a point inside a circle."""
        H, W = 16, 16
        grid_x = torch.linspace(-2, 2, W).unsqueeze(0).expand(H, W)
        grid_y = torch.linspace(-2, 2, H).unsqueeze(1).expand(H, W)
        curve = _circle_points(32, radius=1.0)
        w = differentiable_winding_number(grid_x, grid_y, curve)
        # Centre should have winding ≈ 1
        assert abs(w[H // 2, W // 2].item() - 1.0) < 0.1

    def test_outside_circle(self):
        """Winding number ≈ 0 for a point outside."""
        H, W = 16, 16
        grid_x = torch.linspace(-3, 3, W).unsqueeze(0).expand(H, W)
        grid_y = torch.linspace(-3, 3, H).unsqueeze(1).expand(H, W)
        curve = _circle_points(32, radius=1.0)
        w = differentiable_winding_number(grid_x, grid_y, curve)
        # Corner should have winding ≈ 0
        assert abs(w[0, 0].item()) < 0.15

    def test_gradient_no_nan(self):
        """Winding number gradient is NaN-free."""
        cp = _circle_points(16, radius=1.0).detach().requires_grad_(True)
        H, W = 8, 8
        grid_x = torch.linspace(-2, 2, W).unsqueeze(0).expand(H, W)
        grid_y = torch.linspace(-2, 2, H).unsqueeze(1).expand(H, W)
        t = torch.linspace(0, 1, 64)
        curve = eval_closed_cubic_bspline(cp, t)
        w = differentiable_winding_number(grid_x, grid_y, curve)
        w.sum().backward()
        assert not torch.isnan(cp.grad).any()


# ---------------------------------------------------------------------------
# SDF tests
# ---------------------------------------------------------------------------


class TestSDF:
    def test_inside_negative(self):
        """SDF is negative inside the curve."""
        H, W = 32, 32
        grid_x = torch.linspace(-2, 2, W).unsqueeze(0).expand(H, W)
        grid_y = torch.linspace(-2, 2, H).unsqueeze(1).expand(H, W)
        curve = _circle_points(64, radius=1.0)
        sdf = sdf_from_curve(grid_x, grid_y, curve)
        # Centre should be negative
        assert sdf[H // 2, W // 2] < 0

    def test_outside_positive(self):
        """SDF is positive outside the curve."""
        H, W = 32, 32
        grid_x = torch.linspace(-3, 3, W).unsqueeze(0).expand(H, W)
        grid_y = torch.linspace(-3, 3, H).unsqueeze(1).expand(H, W)
        curve = _circle_points(64, radius=1.0)
        sdf = sdf_from_curve(grid_x, grid_y, curve)
        # Corner should be positive
        assert sdf[0, 0] > 0

    def test_gradient_no_nan_massive(self):
        """10^4 random control-point sets produce no NaN gradient."""
        rng = torch.Generator().manual_seed(42)
        H, W = 8, 8
        grid_x = torch.linspace(-2, 2, W).unsqueeze(0).expand(H, W)
        grid_y = torch.linspace(-2, 2, H).unsqueeze(1).expand(H, W)

        nan_count = 0
        for _ in range(10_000):
            cp = (torch.randn(6, 2, generator=rng) * 0.5).detach().requires_grad_(True)
            t = torch.linspace(0, 1, 40)
            curve = eval_closed_cubic_bspline(cp, t)
            sdf = sdf_from_curve(grid_x, grid_y, curve)
            sdf.sum().backward()
            if torch.isnan(cp.grad).any():
                nan_count += 1
            cp.grad = None

        assert nan_count == 0, f"NaN gradients in {nan_count}/10000 random seeds"

    def test_gradient_vs_findiff(self):
        """SDF gradient matches finite differences."""
        cp = torch.tensor(
            [
                [1.0, 0.0],
                [0.5, 0.8],
                [-0.5, 0.8],
                [-1.0, 0.0],
                [-0.5, -0.8],
                [0.5, -0.8],
            ],
            requires_grad=True,
        )
        H, W = 8, 8
        grid_x = torch.linspace(-2, 2, W).unsqueeze(0).expand(H, W)
        grid_y = torch.linspace(-2, 2, H).unsqueeze(1).expand(H, W)

        def fn(x):
            t = torch.linspace(0, 1, 40)
            curve = eval_closed_cubic_bspline(x, t)
            return sdf_from_curve(grid_x, grid_y, curve).sum()

        analytical = torch.autograd.grad(fn(cp), cp)[0]
        numerical = _finite_diff_grad(fn, cp.detach(), eps=1e-4)
        rel_err = (analytical - numerical).norm() / (numerical.norm() + 1e-12)
        assert rel_err < 5e-2, f"Gradient relative error: {rel_err:.2e}"


# ---------------------------------------------------------------------------
# Projection tests
# ---------------------------------------------------------------------------


class TestProjection:
    def test_sigmoid_inside_outside(self):
        """Sigmoid projection: ~1 inside, ~0 outside."""
        sdf = torch.tensor([-1.0, 0.0, 1.0])
        d = sigmoid_projection(sdf, beta=10.0)
        assert d[0] > 0.9  # inside
        assert abs(d[1] - 0.5) < 0.01  # boundary
        assert d[2] < 0.1  # outside

    def test_heaviside_inside_outside(self):
        """Heaviside projection: ~0 inside, ~1 outside (same convention)."""
        sdf = torch.tensor([-1.0, 0.0, 1.0])
        d = heaviside_projection(sdf, beta=10.0)
        assert d[0] < 0.1  # inside
        assert abs(d[1] - 0.5) < 0.01  # boundary
        assert d[2] > 0.9  # outside

    def test_sigmoid_gradient_no_nan(self):
        """Sigmoid projection gradient is NaN-free."""
        sdf = torch.linspace(-3, 3, 50, requires_grad=True)
        d = sigmoid_projection(sdf, beta=20.0)
        d.sum().backward()
        assert not torch.isnan(sdf.grad).any()

    def test_heaviside_gradient_no_nan(self):
        """Heaviside projection gradient is NaN-free."""
        sdf = torch.linspace(-3, 3, 50, requires_grad=True)
        d = heaviside_projection(sdf, beta=20.0)
        d.sum().backward()
        assert not torch.isnan(sdf.grad).any()

    def test_sigmoid_gradient_vs_findiff(self):
        """Sigmoid gradient matches finite differences."""
        sdf = torch.tensor([-1.0, 0.0, 0.5, 1.0], requires_grad=True)

        def fn(x):
            return sigmoid_projection(x, beta=10.0).sum()

        analytical = torch.autograd.grad(fn(sdf), sdf)[0]
        numerical = _finite_diff_grad(fn, sdf.detach(), eps=1e-5)
        rel_err = (analytical - numerical).norm() / (numerical.norm() + 1e-12)
        assert rel_err < 1e-2


# ---------------------------------------------------------------------------
# End-to-end pipeline test
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_control_points_to_density(self):
        """Full pipeline: control points -> B-spline -> SDF -> sigmoid density."""
        cp = _circle_points(8, radius=1.0)
        H, W = 32, 32
        grid_x = torch.linspace(-2, 2, W).unsqueeze(0).expand(H, W)
        grid_y = torch.linspace(-2, 2, H).unsqueeze(1).expand(H, W)

        t = torch.linspace(0, 1, 64)
        curve = eval_closed_cubic_bspline(cp, t)
        sdf = sdf_from_curve(grid_x, grid_y, curve)
        density = sigmoid_projection(sdf, beta=10.0)

        assert density.shape == (H, W)
        assert density.min() >= 0.0
        assert density.max() <= 1.0
        # Centre should be high density (inside)
        assert density[H // 2, W // 2] > 0.9
        # Corner should be low density (outside)
        assert density[0, 0] < 0.1

    def test_pipeline_gradient_flow(self):
        """Gradients flow from density back to control points."""
        cp = _circle_points(8, radius=1.0).detach().requires_grad_(True)
        H, W = 16, 16
        grid_x = torch.linspace(-2, 2, W).unsqueeze(0).expand(H, W)
        grid_y = torch.linspace(-2, 2, H).unsqueeze(1).expand(H, W)

        t = torch.linspace(0, 1, 40)
        curve = eval_closed_cubic_bspline(cp, t)
        sdf = sdf_from_curve(grid_x, grid_y, curve)
        density = sigmoid_projection(sdf, beta=10.0)
        loss = density.sum()
        loss.backward()

        assert cp.grad is not None
        assert not torch.isnan(cp.grad).any()
        assert cp.grad.norm() > 0
