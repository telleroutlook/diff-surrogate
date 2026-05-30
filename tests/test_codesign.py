"""Tests for diff_surrogate.codesign — co-design API."""

from __future__ import annotations

import torch
import pytest

from diff_surrogate.codesign import CoDesignWorkflow, CoupledLoss


# ---------------------------------------------------------------------------
# Helpers — toy forward functions
# ---------------------------------------------------------------------------


def _em_forward(params: torch.Tensor) -> dict[str, torch.Tensor]:
    """Toy EM solver: returns squared error from target."""
    target = torch.ones_like(params) * 0.5
    return {"em_field": (params - target).pow(2).mean()}


def _litho_forward(params: torch.Tensor) -> dict[str, torch.Tensor]:
    """Toy lithography model: returns EPE-like metric."""
    return {"epe": params.abs().mean()}


def _coupling_fn(merged: dict) -> dict:
    """Coupling: litho contour feeds into EM as a penalty."""
    # In a real scenario, litho output would modify EM input.
    # Here we just add a coupling term.
    epe = merged.get("litho", {}).get("epe", torch.tensor(0.0))
    merged["coupling_penalty"] = epe * 0.5
    return merged


def _coupled_loss(**kwargs) -> torch.Tensor:
    """Combined loss from merged forward outputs."""
    em_loss = kwargs.get("em", {}).get("em_field", torch.tensor(0.0))
    epe = kwargs.get("litho", {}).get("epe", torch.tensor(0.0))
    coupling = kwargs.get("coupling_penalty", torch.tensor(0.0))
    return em_loss + 0.1 * epe + coupling


# ---------------------------------------------------------------------------
# CoupledLoss tests
# ---------------------------------------------------------------------------


class TestCoupledLoss:
    def test_basic_weighted_sum(self):
        loss = CoupledLoss(
            components={
                "a": lambda: torch.tensor(2.0),
                "b": lambda: torch.tensor(3.0),
            },
            weights={"a": 0.5, "b": 1.0},
        )
        total, breakdown = loss()
        assert abs(total.item() - 4.0) < 1e-6  # 0.5*2 + 1.0*3
        assert abs(breakdown["a"] - 2.0) < 1e-6
        assert abs(breakdown["b"] - 3.0) < 1e-6
        assert abs(breakdown["total"] - 4.0) < 1e-6

    def test_default_weight_is_one(self):
        loss = CoupledLoss(
            components={"x": lambda: torch.tensor(5.0)},
        )
        total, bd = loss()
        assert abs(total.item() - 5.0) < 1e-6

    def test_multi_element_tensor_averaged(self):
        loss = CoupledLoss(
            components={
                "v": lambda: torch.tensor([1.0, 2.0, 3.0]),
            },
            weights={"v": 1.0},
        )
        total, bd = loss()
        assert abs(bd["v"] - 2.0) < 1e-6  # mean of [1,2,3]

    def test_kwargs_forwarded(self):
        def component(scale=1.0):
            return torch.tensor(scale * 3.0)

        loss = CoupledLoss(components={"s": component}, weights={"s": 2.0})
        total, bd = loss(scale=2.0)
        assert abs(bd["s"] - 6.0) < 1e-6  # 2.0 * 3.0
        assert abs(total.item() - 12.0) < 1e-6  # 2.0 * 6.0


# ---------------------------------------------------------------------------
# CoDesignWorkflow tests
# ---------------------------------------------------------------------------


class TestCoDesignWorkflow:
    def test_step_reduces_loss(self):
        params = torch.rand(8, 8)
        wf = CoDesignWorkflow(
            design_params=params,
            forward_fns={"em": _em_forward, "litho": _litho_forward},
            loss_fn=_coupled_loss,
            coupling_fn=_coupling_fn,
            lr=0.01,
        )
        losses = []
        for _ in range(10):
            val, _ = wf.step()
            losses.append(val)
        # Loss should decrease over 10 steps on this simple problem
        assert losses[-1] < losses[0]

    def test_run_returns_correct_shapes(self):
        params = torch.rand(4, 4)
        wf = CoDesignWorkflow(
            design_params=params,
            forward_fns={"em": _em_forward},
            loss_fn=_coupled_loss,
            coupling_fn=None,
            lr=0.01,
        )
        final, history = wf.run(n_steps=20, verbose=False)
        assert final.shape == (4, 4)
        assert len(history) == 20

    def test_params_clamped_to_bounds(self):
        params = torch.rand(4, 4) * 10 - 5  # [-5, 5]
        wf = CoDesignWorkflow(
            design_params=params,
            forward_fns={"em": _em_forward},
            loss_fn=_coupled_loss,
            lr=0.1,
            param_bounds=(0.0, 1.0),
        )
        wf.run(n_steps=5, verbose=False)
        assert wf.params.min().item() >= -1e-6
        assert wf.params.max().item() <= 1.0 + 1e-6

    def test_no_param_bounds(self):
        params = torch.rand(4, 4)
        wf = CoDesignWorkflow(
            design_params=params,
            forward_fns={"em": _em_forward},
            loss_fn=_coupled_loss,
            lr=0.01,
            param_bounds=None,
        )
        wf.run(n_steps=5, verbose=False)
        # Should still work without clamping
        assert wf.params.shape == (4, 4)

    def test_compare_baseline(self):
        params = torch.rand(8, 8)
        wf = CoDesignWorkflow(
            design_params=params,
            forward_fns={"em": _em_forward, "litho": _litho_forward},
            loss_fn=_coupled_loss,
            coupling_fn=_coupling_fn,
            lr=0.01,
        )
        _, coupled_history = wf.run(n_steps=10, verbose=False)
        _, baseline_history = wf.compare_baseline(n_steps=10, verbose=False)
        assert len(coupled_history) == 10
        assert len(baseline_history) == 10

    def test_report(self):
        params = torch.rand(8, 8)
        wf = CoDesignWorkflow(
            design_params=params,
            forward_fns={"em": _em_forward},
            loss_fn=_coupled_loss,
            lr=0.01,
        )
        wf.run(n_steps=5, verbose=False)
        wf.compare_baseline(n_steps=5, verbose=False)
        r = wf.report()
        assert "coupled_final_loss" in r
        assert "baseline_final_loss" in r
        assert "improvement_pct" in r
        assert "coupled_history" in r
        assert "baseline_history" in r

    def test_report_before_run(self):
        wf = CoDesignWorkflow(
            design_params=torch.rand(4, 4),
            forward_fns={"em": _em_forward},
            loss_fn=_coupled_loss,
        )
        r = wf.report()
        assert r == {}

    def test_with_coupled_loss_instance(self):
        def em_component(**kw):
            return kw.get("em", {}).get("em_field", torch.tensor(0.0))

        def litho_component(**kw):
            return kw.get("litho", {}).get("epe", torch.tensor(0.0))

        coupled = CoupledLoss(
            components={"em": em_component, "litho": litho_component},
            weights={"em": 1.0, "litho": 0.1},
        )
        params = torch.rand(4, 4)
        wf = CoDesignWorkflow(
            design_params=params,
            forward_fns={"em": _em_forward, "litho": _litho_forward},
            loss_fn=coupled,
            lr=0.01,
        )
        final, history = wf.run(n_steps=10, verbose=False)
        assert len(history) == 10
        # Verify breakdown is populated
        _, breakdown = wf.step()
        assert "em" in breakdown
        assert "litho" in breakdown

    def test_original_params_not_mutated(self):
        original = torch.rand(8, 8)
        original_clone = original.clone()
        wf = CoDesignWorkflow(
            design_params=original,
            forward_fns={"em": _em_forward},
            loss_fn=_coupled_loss,
            lr=0.01,
        )
        wf.run(n_steps=5, verbose=False)
        assert torch.allclose(original, original_clone)

    def test_nan_stops_run(self):
        def nan_forward(params):
            return {"val": torch.tensor(float("nan"))}

        def nan_loss(**kw):
            return kw["nan"]["val"]

        wf = CoDesignWorkflow(
            design_params=torch.rand(4, 4),
            forward_fns={"nan": nan_forward},
            loss_fn=nan_loss,
            lr=0.01,
        )
        _, history = wf.run(n_steps=10, verbose=False)
        assert len(history) == 1
        assert history[0] != history[0]  # NaN


# ---------------------------------------------------------------------------
# Integration: domain-agnostic end-to-end
# ---------------------------------------------------------------------------


class TestDomainAgnostic:
    """Verify the API works with completely arbitrary forward functions."""

    def test_single_domain(self):
        def simple_fwd(p):
            return {"mse": (p - 0.5).pow(2).mean()}

        def simple_loss(**kw):
            return kw["a"]["mse"]

        wf = CoDesignWorkflow(
            design_params=torch.rand(16, 16),
            forward_fns={"a": simple_fwd},
            loss_fn=simple_loss,
            lr=0.05,
        )
        final, history = wf.run(n_steps=30, verbose=False)
        assert history[-1] < history[0]

    def test_three_domains(self):
        def domain_a(p):
            return {"loss_a": p.pow(2).mean()}

        def domain_b(p):
            return {"loss_b": (1 - p).pow(2).mean()}

        def domain_c(p):
            return {"loss_c": (p - 0.7).abs().mean()}

        def combined_loss(**kw):
            return (
                kw["a"]["loss_a"]
                + kw["b"]["loss_b"]
                + 0.5 * kw["c"]["loss_c"]
            )

        wf = CoDesignWorkflow(
            design_params=torch.rand(8, 8),
            forward_fns={"a": domain_a, "b": domain_b, "c": domain_c},
            loss_fn=combined_loss,
            lr=0.02,
        )
        final, history = wf.run(n_steps=50, verbose=False)
        assert final.shape == (8, 8)
        assert history[-1] < history[0]
