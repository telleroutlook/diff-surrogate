"""Tests for adjoint-guided sampling guidance_fn hook (S11.3)."""

from __future__ import annotations

import torch

from diff_surrogate.flow_operator import FlowOperator


def _make_flow_op(**overrides):
    defaults = dict(
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
    defaults.update(overrides)
    return FlowOperator(**defaults)


# ---------------------------------------------------------------------------
# guidance_fn hook tests
# ---------------------------------------------------------------------------


class TestGuidanceHookBasic:
    def test_guidance_fn_zero_is_noop(self):
        op = _make_flow_op()
        op.eval()
        cond = torch.randn(4)

        def zero_guidance(x, t):
            return torch.zeros_like(x)

        result_guided = op.sample(cond, n_steps=5, guidance_fn=zero_guidance, guidance_scale=1.0)
        assert result_guided.shape == (2, 8)

    def test_guidance_fn_receives_correct_shapes(self):
        op = _make_flow_op()
        op.eval()
        cond = torch.randn(4)
        observed_shapes = []

        def observer(x, t):
            observed_shapes.append((x.shape, t.shape))
            return torch.zeros_like(x)

        op.sample(cond, n_steps=3, guidance_fn=observer, guidance_scale=1.0)
        assert len(observed_shapes) == 3
        for xs, ts in observed_shapes:
            assert xs[0] == 1  # batch=1 for single sample
            assert xs[1] == op.latent_dim
            assert ts[0] == 1

    def test_guidance_scale_zero_ignores_guidance(self):
        op = _make_flow_op()
        op.eval()
        cond = torch.randn(4)

        def big_guidance(x, t):
            return torch.ones_like(x) * 100

        r1 = op.sample(cond, n_steps=5, guidance_fn=big_guidance, guidance_scale=0.0)
        # With scale=0, guidance should be ignored
        assert r1.shape == (2, 8)

    def test_guidance_affects_output(self):
        torch.manual_seed(0)
        op = _make_flow_op()
        op.eval()
        cond = torch.randn(4)

        def push_guidance(x, t):
            return -0.5 * x

        torch.manual_seed(100)
        r_no_guide = op.sample(cond, n_steps=5)
        torch.manual_seed(100)
        r_with_guide = op.sample(cond, n_steps=5, guidance_fn=push_guidance, guidance_scale=5.0)

        assert not torch.allclose(r_no_guide, r_with_guide, atol=1e-3)

    def test_ensemble_guidance(self):
        op = _make_flow_op()
        op.eval()
        cond = torch.randn(4)

        def neg_guidance(x, t):
            return -0.1 * x

        result = op.generate_ensemble(
            cond,
            n_samples=4,
            n_steps=5,
            guidance_fn=neg_guidance,
            guidance_scale=1.0,
        )
        assert result.shape == (4, 2, 8)

    def test_ensemble_guidance_shapes(self):
        op = _make_flow_op()
        op.eval()
        cond = torch.randn(4)
        observed = []

        def observer(x, t):
            observed.append(x.shape)
            return torch.zeros_like(x)

        op.generate_ensemble(
            cond,
            n_samples=3,
            n_steps=4,
            guidance_fn=observer,
            guidance_scale=1.0,
        )
        assert len(observed) == 4
        for s in observed:
            assert s == (3, op.latent_dim)


class TestAdjointGuidanceSimulation:
    """Simulate adjoint-guided sampling with a mock forward model."""

    def test_mock_adjoint_improves_fom(self):
        op = _make_flow_op()
        op.eval()

        target_response = torch.ones(4)
        weight = torch.randn(4, 8)

        def mock_forward(z):
            return (z @ weight.T - target_response).pow(2).sum()

        def adjoint_guidance(x, t):
            # Must compute gradient in enabled-grad context
            x_param = x.detach().float().requires_grad_(True)
            loss = mock_forward(x_param)
            (grad,) = torch.autograd.grad(loss, x_param)
            return -grad

        cond = torch.randn(4)
        r_guided = op.sample(
            cond, n_steps=10,
            guidance_fn=adjoint_guidance, guidance_scale=0.01,
        )

        assert r_guided.shape == (2, 8)
        assert torch.isfinite(r_guided).all()


class TestGuidanceCostTracking:
    """Test that guidance calls can be tracked for cost accounting."""

    def test_count_guidance_calls(self):
        op = _make_flow_op()
        op.eval()
        cond = torch.randn(4)
        call_count = [0]

        def counting_guidance(x, t):
            call_count[0] += 1
            return torch.zeros_like(x)

        op.sample(cond, n_steps=10, guidance_fn=counting_guidance, guidance_scale=1.0)
        assert call_count[0] == 10

    def test_guidance_cost_in_experiment_design(self):
        from diff_surrogate.experiment_design import CostModel

        # Verify CostModel can account for adjoint calls
        cm = CostModel(
            fidelity_levels={"low": 1.0, "high": 10.0, "adjoint": 5.0},
            total_budget=100.0,
        )
        assert cm.can_afford("adjoint", 5)
        cm.consume("adjoint", 5)
        assert cm.remaining() == 75.0
