"""Tests for ManifoldProjection (S11.1 — PMFM hard physics constraints)."""

from __future__ import annotations

import pytest
import torch

from diff_surrogate.flow_operator import (
    FlowOperator,
    ManifoldProjection,
)

# ---------------------------------------------------------------------------
# ManifoldProjection unit tests
# ---------------------------------------------------------------------------


class TestManifoldProjectionInit:
    def test_default_constraints(self):
        mp = ManifoldProjection(spatial_dim=32)
        assert mp.constraint_types == {"mass_conservation"}

    def test_custom_constraints(self):
        mp = ManifoldProjection(
            spatial_dim=16,
            constraint_types={"divergence_free", "dirichlet"},
        )
        assert "divergence_free" in mp.constraint_types
        assert "dirichlet" in mp.constraint_types

    def test_unsupported_constraint_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            ManifoldProjection(spatial_dim=8, constraint_types={"bogus"})

    @pytest.mark.parametrize("ctype", [
        "divergence_free", "mass_conservation", "flux_conservation",
        "dirichlet", "neumann",
    ])
    def test_each_constraint_individual(self, ctype: str):
        mp = ManifoldProjection(spatial_dim=16, constraint_types={ctype})
        x = torch.randn(2, 3, 16)
        out = mp(x)
        assert out.shape == x.shape


class TestDivergenceFreeProjection:
    def test_reduces_divergence(self):
        mp = ManifoldProjection(
            spatial_dim=32,
            constraint_types={"divergence_free"},
            n_projection_steps=10,
            projection_lr=0.8,
        )
        torch.manual_seed(0)
        x = torch.randn(4, 2, 32)
        projected = mp(x)

        res_before = mp._divergence_residual(x)
        res_after = mp._divergence_residual(projected)
        assert res_after <= res_before

    def test_residual_is_positive(self):
        mp = ManifoldProjection(spatial_dim=16, constraint_types={"divergence_free"})
        x = torch.randn(2, 2, 16)
        res = mp._divergence_residual(x)
        assert res >= 0


class TestMassConservationProjection:
    def test_preserves_total_mass(self):
        mp = ManifoldProjection(
            spatial_dim=32,
            constraint_types={"mass_conservation"},
        )
        torch.manual_seed(1)
        x = torch.randn(4, 3, 32)
        reference = torch.randn(4, 3, 32)
        projected = mp(x, reference=reference)

        ref_mass = reference.sum(dim=-1)
        proj_mass = projected.sum(dim=-1)
        torch.testing.assert_close(proj_mass, ref_mass, atol=1e-4, rtol=1e-4)

    def test_self_reference_is_noop(self):
        mp = ManifoldProjection(spatial_dim=16, constraint_types={"mass_conservation"})
        x = torch.randn(2, 2, 16)
        projected = mp(x, reference=x)
        torch.testing.assert_close(projected, x, atol=1e-5, rtol=1e-5)


class TestFluxConservationProjection:
    def test_reduces_flux_mismatch(self):
        mp = ManifoldProjection(
            spatial_dim=16,
            constraint_types={"flux_conservation"},
            projection_lr=1.0,
            n_projection_steps=10,
        )
        x = torch.randn(2, 2, 16)
        ref = torch.randn(2, 2, 16)
        projected = mp(x, reference=ref)

        # After enough steps, boundary flux should match reference
        torch.testing.assert_close(
            projected[:, :, 0], ref[:, :, 0], atol=1e-4, rtol=1e-4,
        )
        torch.testing.assert_close(
            projected[:, :, -1], ref[:, :, -1], atol=1e-4, rtol=1e-4,
        )


class TestDirichletProjection:
    def test_sets_boundary_to_value(self):
        mp = ManifoldProjection(
            spatial_dim=16,
            constraint_types={"dirichlet"},
            boundary_value=0.0,
        )
        x = torch.randn(4, 3, 16)
        projected = mp(x)
        assert torch.allclose(projected[:, :, 0], torch.zeros(4, 3))
        assert torch.allclose(projected[:, :, -1], torch.zeros(4, 3))

    def test_custom_boundary_value(self):
        val = 5.0
        mp = ManifoldProjection(
            spatial_dim=8,
            constraint_types={"dirichlet"},
            boundary_value=val,
        )
        x = torch.randn(2, 2, 8)
        projected = mp(x)
        assert torch.allclose(projected[:, :, 0], torch.full((2, 2), val))
        assert torch.allclose(projected[:, :, -1], torch.full((2, 2), val))


class TestNeumannProjection:
    def test_zero_gradient_at_boundary(self):
        mp = ManifoldProjection(
            spatial_dim=16,
            constraint_types={"neumann"},
        )
        x = torch.randn(4, 3, 16)
        projected = mp(x)
        torch.testing.assert_close(projected[:, :, 0], projected[:, :, 1], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(projected[:, :, -1], projected[:, :, -2], atol=1e-6, rtol=1e-6)


class TestComputeResiduals:
    def test_returns_all_enabled(self):
        mp = ManifoldProjection(
            spatial_dim=16,
            constraint_types={"divergence_free", "mass_conservation", "dirichlet"},
        )
        x = torch.randn(2, 3, 16)
        residuals = mp.compute_residuals(x)
        assert "divergence" in residuals
        assert "mass" in residuals
        assert "dirichlet" in residuals
        assert "flux" not in residuals
        assert "neumann" not in residuals

    def test_all_residuals_positive(self):
        mp = ManifoldProjection(
            spatial_dim=16,
            constraint_types=ManifoldProjection.SUPPORTED_CONSTRAINTS,
        )
        x = torch.randn(2, 3, 16)
        residuals = mp.compute_residuals(x)
        for name, val in residuals.items():
            assert val >= 0, f"Residual '{name}' is negative"


class TestCombinedProjection:
    def test_combined_mass_and_dirichlet(self):
        mp = ManifoldProjection(
            spatial_dim=16,
            constraint_types={"mass_conservation", "dirichlet"},
            boundary_value=0.0,
            n_projection_steps=5,
        )
        ref = torch.randn(2, 3, 16)
        x = torch.randn(2, 3, 16) * 3
        projected = mp(x, reference=ref)

        assert torch.allclose(projected[:, :, 0], torch.zeros(2, 3), atol=1e-4)
        assert torch.allclose(projected[:, :, -1], torch.zeros(2, 3), atol=1e-4)


# ---------------------------------------------------------------------------
# Integration with FlowOperator
# ---------------------------------------------------------------------------


class TestFlowOperatorWithConstraints:
    @pytest.fixture
    def flow_op(self):
        return FlowOperator(
            spatial_dim=8,
            embed_dim=16,
            n_fields=2,
            latent_dim=8,
            cond_dim=4,
            n_backbone_layers=1,
            n_backbone_heads=2,
            n_flow_layers=1,
            n_flow_heads=2,
            decoder_hidden=16,
        )

    def test_sample_with_constraints(self, flow_op: FlowOperator):
        mp = ManifoldProjection(
            spatial_dim=8,
            n_fields=2,
            constraint_types={"mass_conservation"},
        )
        cond = torch.randn(4)
        flow_op.eval()
        result = flow_op.sample(cond, n_steps=5, constraints=mp)
        assert result.shape == (2, 8)

    def test_generate_ensemble_with_constraints(self, flow_op: FlowOperator):
        mp = ManifoldProjection(
            spatial_dim=8,
            n_fields=2,
            constraint_types={"divergence_free"},
        )
        cond = torch.randn(4)
        flow_op.eval()
        result = flow_op.generate_ensemble(cond, n_samples=4, n_steps=5, constraints=mp)
        assert result.shape == (4, 2, 8)

    def test_constraints_reduce_residuals(self, flow_op: FlowOperator):
        mp = ManifoldProjection(
            spatial_dim=8,
            n_fields=2,
            constraint_types={"dirichlet"},
            boundary_value=0.0,
        )
        cond = torch.randn(4)
        flow_op.eval()

        fields_unconstrained = flow_op.generate_ensemble(cond, n_samples=4, n_steps=5)
        fields_constrained = flow_op.generate_ensemble(cond, n_samples=4, n_steps=5, constraints=mp)

        res_uncon = mp._boundary_residual(fields_unconstrained)
        res_con = mp._boundary_residual(fields_constrained)
        assert res_con < res_uncon

    def test_sample_with_guidance_fn(self, flow_op: FlowOperator):
        def simple_guidance(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return -0.1 * x

        cond = torch.randn(4)
        flow_op.eval()
        result = flow_op.sample(cond, n_steps=5, guidance_fn=simple_guidance, guidance_scale=1.0)
        assert result.shape == (2, 8)

    def test_generate_ensemble_with_guidance(self, flow_op: FlowOperator):
        def zero_guidance(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

        cond = torch.randn(4)
        flow_op.eval()
        result = flow_op.generate_ensemble(
            cond, n_samples=4, n_steps=5,
            guidance_fn=zero_guidance, guidance_scale=2.0,
        )
        assert result.shape == (4, 2, 8)

    def test_combined_constraints_and_guidance(self, flow_op: FlowOperator):
        mp = ManifoldProjection(
            spatial_dim=8,
            n_fields=2,
            constraint_types={"mass_conservation"},
        )

        def guidance(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return -0.05 * x

        cond = torch.randn(4)
        flow_op.eval()
        result = flow_op.generate_ensemble(
            cond, n_samples=4, n_steps=5,
            constraints=mp, guidance_fn=guidance, guidance_scale=1.0,
        )
        assert result.shape == (4, 2, 8)
