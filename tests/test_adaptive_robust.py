"""Tests for diff_surrogate.adaptive_robust -- canonical adaptive robust optimizer."""

from __future__ import annotations

import pytest
import torch

from diff_surrogate.adaptive_robust import (
    AdaptiveRobustOptimizer,
    FabricableSubspaceProjection,
    axial_samples,
    correlated_perturbation,
)
from diff_surrogate.robust_design import CornerSpec


# =====================================================================
# axial_samples
# =====================================================================


class TestAxialSamples:
    def test_shape_2n_plus_1(self):
        """2N+1 samples for N dims."""
        for n in (1, 3, 5, 10):
            s = axial_samples(n, sigma=1.0)
            assert s.shape == (2 * n + 1, n)

    def test_nominal_is_zero(self):
        s = axial_samples(3, sigma=5.0)
        assert torch.allclose(s[0], torch.zeros(3, dtype=torch.float64))

    def test_axial_structure(self):
        n = 4
        sigma = 2.5
        s = axial_samples(n, sigma)
        for i in range(n):
            pos = s[1 + 2 * i]
            neg = s[2 + 2 * i]
            assert torch.allclose(pos, -neg)
            assert pos.abs().sum().item() == pytest.approx(sigma)

    def test_dtype_and_device(self):
        s = axial_samples(2, 1.0, dtype=torch.float32)
        assert s.dtype == torch.float32


# =====================================================================
# correlated_perturbation
# =====================================================================


class TestCorrelatedPerturbation:
    def test_output_shape(self):
        params = torch.randn(10, dtype=torch.float64)
        chol = torch.eye(3, dtype=torch.float64)
        deltas = correlated_perturbation(params, chol, n_samples=16)
        assert deltas.shape == (16, 3)

    def test_covariance_structure(self):
        """Sample covariance should approximate the input covariance for large N."""
        torch.manual_seed(0)
        # Build a known covariance via Cholesky
        L = torch.tensor([[1.0, 0.0], [0.5, 0.5]], dtype=torch.float64)
        cov_target = L @ L.T
        params = torch.zeros(2, dtype=torch.float64)
        deltas = correlated_perturbation(params, L, n_samples=5000)
        cov_empirical = torch.cov(deltas.T)
        assert torch.allclose(cov_empirical, cov_target, atol=0.15)

    def test_identity_cholesky_isotropic(self):
        params = torch.zeros(3, dtype=torch.float64)
        chol = torch.eye(3, dtype=torch.float64)
        deltas = correlated_perturbation(params, chol, n_samples=1000)
        # Each column should have std ~1.0
        stds = deltas.std(dim=0)
        assert torch.allclose(stds, torch.ones(3, dtype=torch.float64), atol=0.15)


# =====================================================================
# FabricableSubspaceProjection
# =====================================================================


class TestFabricableSubspaceProjection:
    def test_output_shape(self):
        proj = FabricableSubspaceProjection(n_levels=4)
        density = torch.rand(10, 10, dtype=torch.float64)
        result = proj.project(density)
        assert result.shape == density.shape

    def test_values_near_discrete_levels(self):
        """With low temperature, output should be close to one of the levels."""
        proj = FabricableSubspaceProjection(n_levels=4, temperature=0.05, min_cd_pixels=0)
        density = torch.rand(20, 20, dtype=torch.float64)
        projected = proj.project(density)
        levels = torch.linspace(0, 1, 4, dtype=torch.float64)
        # Each value should be close to its nearest level
        distances = torch.abs(projected.unsqueeze(-1) - levels)
        min_dist = distances.min(dim=-1).values
        assert min_dist.max().item() < 0.2

    def test_gradient_flows(self):
        proj = FabricableSubspaceProjection(n_levels=4, temperature=0.5, min_cd_pixels=0)
        density = torch.rand(8, 8, dtype=torch.float64, requires_grad=True)
        result = proj.project(density)
        result.sum().backward()
        assert density.grad is not None
        assert not torch.isnan(density.grad).any()

    def test_projection_loss(self):
        proj = FabricableSubspaceProjection(n_levels=4)
        density = torch.rand(10, 10, dtype=torch.float64, requires_grad=True)
        loss = proj.projection_loss(density)
        assert loss.numel() == 1
        assert loss.item() >= 0
        loss.backward()
        assert density.grad is not None


# =====================================================================
# AdaptiveRobustOptimizer -- axial / curriculum path
# =====================================================================


class TestAdaptiveRobustOptimizerAxial:
    def test_basic_optimization_converges(self):
        """Minimize (params**2).sum() -- should drive params toward zero."""
        torch.manual_seed(42)
        params = torch.randn(10, dtype=torch.float64)

        def forward_fn(p, delta):
            return (p ** 2).sum()

        def perturb_fn(p, delta):
            return p + delta.sum() * 0.01

        opt = AdaptiveRobustOptimizer(n_variation_dims=2, sigma=1.0)
        result, history = opt.optimize(
            params, forward_fn, perturb_fn, n_steps=50, lr=0.05, verbose=False,
        )
        assert result.shape == params.shape
        assert len(history) == 50
        # Should have decreased loss significantly
        assert history[-1] < history[0]

    def test_curriculum_increases_samples(self):
        """With curriculum_frac=1.0 we should have more samples than axial alone."""
        opt = AdaptiveRobustOptimizer(n_variation_dims=3, sigma=1.0, n_random_budget=16)
        params = torch.randn(5, dtype=torch.float64, requires_grad=True)

        loss_axial = opt.compute_robust_loss(
            params,
            forward_fn=lambda p, d: (p ** 2).sum(),
            perturbation_fn=lambda p, d: p,
            curriculum_frac=0.0,
        )
        loss_full = opt.compute_robust_loss(
            params,
            forward_fn=lambda p, d: (p ** 2).sum(),
            perturbation_fn=lambda p, d: p,
            curriculum_frac=1.0,
        )
        assert loss_axial.numel() == 1
        assert loss_full.numel() == 1

    def test_gradient_flows(self):
        opt = AdaptiveRobustOptimizer(n_variation_dims=2, sigma=1.0)
        params = torch.randn(5, dtype=torch.float64, requires_grad=True)

        loss = opt.compute_robust_loss(
            params,
            forward_fn=lambda p, d: (p ** 2).sum(),
            perturbation_fn=lambda p, d: p,
            curriculum_frac=0.5,
        )
        loss.backward()
        assert params.grad is not None

    def test_covariance_matrix(self):
        """Custom covariance should be accepted without error."""
        cov = torch.tensor([[1.0, 0.3], [0.3, 1.0]], dtype=torch.float64)
        opt = AdaptiveRobustOptimizer(n_variation_dims=2, sigma=1.0, cov_matrix=cov)
        params = torch.randn(5, dtype=torch.float64, requires_grad=True)
        loss = opt.compute_robust_loss(
            params,
            forward_fn=lambda p, d: (p ** 2).sum(),
            perturbation_fn=lambda p, d: p,
        )
        loss.backward()
        assert params.grad is not None


# =====================================================================
# AdaptiveRobustOptimizer -- corner path (with / without ensemble)
# =====================================================================


class TestAdaptiveRobustOptimizerCorners:
    def _make_corners(self):
        return [
            CornerSpec(label="nominal", weight=1.0),
            CornerSpec(label="upper", weight=0.5),
        ]

    def test_corners_without_ensemble_static_weights(self):
        """Without ensemble, weights should be static (normalized)."""
        corners = self._make_corners()
        opt = AdaptiveRobustOptimizer(
            n_variation_dims=2, sigma=1.0, corners=corners, ensemble=None,
        )
        assert opt.corner_evaluator is not None

        params = torch.randn(4, dtype=torch.float64, requires_grad=True)
        loss, info = opt.compute_robust_loss_with_corners(
            params,
            forward_fn=lambda d: (d ** 2).sum(),
            loss_fn=lambda o: o.mean(),
        )
        assert loss.numel() == 1
        assert len(info["weights"]) == 2
        # Static weights should sum to 1
        assert sum(info["weights"]) == pytest.approx(1.0)

    def test_no_corners_fallback(self):
        """Without corners, should fall back to single-point evaluation."""
        opt = AdaptiveRobustOptimizer(n_variation_dims=2, sigma=1.0)
        params = torch.randn(4, dtype=torch.float64, requires_grad=True)
        loss, info = opt.compute_robust_loss_with_corners(
            params,
            forward_fn=lambda d: (d ** 2).sum(),
            loss_fn=lambda o: o.mean(),
        )
        assert loss.numel() == 1
        assert info["weights"] == [1.0]
        assert info["uncertainties"] == [0.0]

    def test_with_ensemble_adaptive_weights(self):
        """With ensemble, weights should differ from static (uncertainty-driven)."""
        corners = self._make_corners()

        class FakeEnsemble:
            def predict_with_uncertainty(self, x):
                means = {"output": torch.zeros_like(x)}
                stds = {"output": torch.ones_like(x) * 0.5}
                return means, stds

        opt = AdaptiveRobustOptimizer(
            n_variation_dims=2,
            sigma=1.0,
            corners=corners,
            ensemble=FakeEnsemble(),
            uncertainty_weight=0.8,
        )
        params = torch.randn(4, dtype=torch.float64, requires_grad=True)
        loss, info = opt.compute_robust_loss_with_corners(
            params,
            forward_fn=lambda d: (d ** 2).sum(),
            loss_fn=lambda o: o.mean(),
        )
        assert loss.numel() == 1
        assert len(info["weights"]) == 2
        assert sum(info["weights"]) == pytest.approx(1.0)

    def test_uncertainty_weight_zero_degenerates_to_static(self):
        """uncertainty_weight=0 should produce identical weights to no ensemble."""
        corners = self._make_corners()

        class FakeEnsemble:
            def predict_with_uncertainty(self, x):
                return {"output": torch.zeros_like(x)}, {"output": torch.ones_like(x) * 10.0}

        opt_dynamic = AdaptiveRobustOptimizer(
            n_variation_dims=2,
            sigma=1.0,
            corners=corners,
            ensemble=FakeEnsemble(),
            uncertainty_weight=0.0,
        )
        opt_static = AdaptiveRobustOptimizer(
            n_variation_dims=2,
            sigma=1.0,
            corners=corners,
            ensemble=None,
        )

        params = torch.randn(4, dtype=torch.float64)

        w_dynamic = opt_dynamic.corner_evaluator.adaptive_weights(params)
        w_static = opt_static.corner_evaluator.adaptive_weights(params)

        for wd, ws in zip(w_dynamic, w_static):
            assert wd == pytest.approx(ws, abs=1e-6)

    def test_corner_gradient_flows(self):
        """Gradients must flow through corner-based loss."""
        corners = self._make_corners()
        opt = AdaptiveRobustOptimizer(
            n_variation_dims=2, sigma=1.0, corners=corners,
        )
        params = torch.randn(4, dtype=torch.float64, requires_grad=True)
        loss, _ = opt.compute_robust_loss_with_corners(
            params,
            forward_fn=lambda d: (d ** 2).sum(),
            loss_fn=lambda o: o.mean(),
        )
        loss.backward()
        assert params.grad is not None


# =====================================================================
# Integration: corner optimization loop
# =====================================================================


class TestAdaptiveRobustOptimizerIntegration:
    def test_corner_optimization_converges(self):
        """Full optimization with corners should converge on a simple problem."""
        torch.manual_seed(0)
        corners = [
            CornerSpec(label="nominal", weight=1.0),
            CornerSpec(label="stress", weight=0.5),
        ]
        opt = AdaptiveRobustOptimizer(
            n_variation_dims=2, sigma=1.0, corners=corners,
        )

        params = torch.randn(4, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([params], lr=0.05)
        losses = []

        for step in range(30):
            loss, _ = opt.compute_robust_loss_with_corners(
                params,
                forward_fn=lambda d: (d ** 2).sum(),
                loss_fn=lambda o: o.mean(),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0]
