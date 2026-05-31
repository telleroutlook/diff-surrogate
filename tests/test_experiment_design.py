"""Tests for Bayesian experiment design + active sampling closed loop."""

from __future__ import annotations

import torch

from diff_surrogate.experiment_design import (
    AcquisitionFunction,
    CostModel,
    DesignBenchmark,
    ExperimentDesignLoop,
    ExperimentDesignResult,
)


def _quadratic_truth(x: torch.Tensor) -> torch.Tensor:
    return x[:, 0] ** 2 + 2.0 * torch.sin(3.0 * x[:, 0])


def _cheap_surrogate(x: torch.Tensor) -> torch.Tensor:
    return x[:, 0] ** 2


def _default_cost_model(budget: float = 1000.0) -> CostModel:
    return CostModel(
        fidelity_levels={"rcwa": 1.0, "fdtd": 50.0},
        total_budget=budget,
    )


def _default_bounds() -> torch.Tensor:
    return torch.tensor([[-2.0, 2.0]])


def test_cost_model_basic():
    cm = _default_cost_model()
    assert cm.remaining() == 1000.0
    assert cm.can_afford("rcwa", 10)
    assert cm.can_afford("fdtd", 20)
    assert not cm.can_afford("fdtd", 21)
    assert not cm.can_afford("nonexistent", 1)


def test_cost_model_budget_tracking():
    cm = _default_cost_model(budget=100.0)
    assert cm.can_afford("fdtd", 2)
    cm.consume("fdtd", 2)
    assert cm.remaining() == 0.0
    assert not cm.can_afford("rcwa", 1)

    cm2 = _default_cost_model(budget=200.0)
    cm2.consume("rcwa", 50)
    assert cm2.remaining() == 150.0
    assert cm2.budget_consumed == 50.0


def test_acquisition_uncertainty_selects_high_bandwidth():
    cm = _default_cost_model()
    loop = ExperimentDesignLoop(
        cost_model=cm,
        acquisition_fn=AcquisitionFunction.UNCERTAINTY,
        input_bounds=_default_bounds(),
        seed=42,
    )

    M = 100
    candidates = torch.randn(M, 1)
    bandwidths = torch.zeros(M)
    bandwidths[77] = 10.0
    bandwidths[42] = 8.0

    indices = loop._acquire_uncertainty(candidates, bandwidths, 2)
    assert 77 in indices.tolist()
    assert 42 in indices.tolist()


def test_acquisition_diversity_spreads_points():
    cm = _default_cost_model()
    loop = ExperimentDesignLoop(
        cost_model=cm,
        acquisition_fn=AcquisitionFunction.DIVERSITY,
        input_bounds=_default_bounds(),
        seed=0,
    )

    existing = torch.tensor([[-1.5], [-0.5]])
    candidates = torch.linspace(-2.0, 2.0, 50).unsqueeze(-1)

    indices = loop._acquire_diversity(candidates, existing, 3)
    assert indices.shape[0] == 3

    selected = candidates[indices]
    spread = selected[:, 0].max() - selected[:, 0].min()
    assert spread > 1.0, f"Diversity-selected points are too clustered: spread={spread:.4f}"


def test_experiment_design_loop_runs():
    cm = _default_cost_model(budget=500.0)
    loop = ExperimentDesignLoop(
        cost_model=cm,
        acquisition_fn=AcquisitionFunction.UNCERTAINTY,
        input_bounds=_default_bounds(),
        n_candidates=200,
        seed=0,
    )

    init_x = torch.linspace(-2.0, -1.0, 10).unsqueeze(-1)
    init_y = _quadratic_truth(init_x)

    result = loop.run_loop(
        surrogate_fn=_cheap_surrogate,
        truth_fn=_quadratic_truth,
        initial_inputs=init_x,
        initial_targets=init_y,
        n_iterations=5,
        samples_per_iter=3,
    )

    assert isinstance(result, ExperimentDesignResult)
    assert result.total_cost > 0
    assert len(result.convergence_history) == 5
    assert len(result.fidelity_history) == 5
    assert result.points_selected.shape[0] > init_x.shape[0]


def test_experiment_design_vs_random():
    cm = _default_cost_model(budget=2000.0)
    loop = ExperimentDesignLoop(
        cost_model=cm,
        acquisition_fn=AcquisitionFunction.HYBRID,
        input_bounds=_default_bounds(),
        n_candidates=300,
        seed=0,
    )

    init_x = torch.linspace(-2.0, -0.5, 8).unsqueeze(-1)
    init_y = _quadratic_truth(init_x)

    result = loop.run_loop(
        surrogate_fn=_cheap_surrogate,
        truth_fn=_quadratic_truth,
        initial_inputs=init_x,
        initial_targets=init_y,
        n_iterations=8,
        samples_per_iter=4,
    )

    assert result.total_hf_evals > 0
    assert result.total_cost > 0
    assert result.final_uncertainty >= 0


def test_multi_fidelity_cost_routing():
    cm = CostModel(
        fidelity_levels={"rcwa": 1.0, "fdtd": 100.0},
        total_budget=150.0,
    )
    loop = ExperimentDesignLoop(
        cost_model=cm,
        acquisition_fn=AcquisitionFunction.UNCERTAINTY,
        input_bounds=_default_bounds(),
        seed=0,
    )

    assert loop._choose_fidelity(1) == "fdtd"
    cm.consume("fdtd", 1)
    assert cm.remaining() == 50.0
    assert loop._choose_fidelity(1) == "rcwa"


def test_benchmark_runs():
    cm = _default_cost_model(budget=500.0)
    benchmark = DesignBenchmark(
        cost_model=cm,
        input_bounds=_default_bounds(),
        acquisition_fn=AcquisitionFunction.UNCERTAINTY,
        n_candidates=200,
    )

    init_x = torch.linspace(-2.0, -0.5, 8).unsqueeze(-1)
    init_y = _quadratic_truth(init_x)

    results = benchmark.run(
        surrogate_fn=_cheap_surrogate,
        truth_fn=_quadratic_truth,
        initial_inputs=init_x,
        initial_targets=init_y,
        n_seeds=2,
        n_iterations=4,
        samples_per_iter=3,
    )

    assert "active_final_bandwidth" in results
    assert "random_final_bandwidth" in results
    assert "uniform_final_bandwidth" in results
    assert results["n_seeds"] == 2
    assert len(results["active_convergence_histories"]) == 2


def test_convergence_with_budget():
    cm = CostModel(
        fidelity_levels={"lf": 1.0, "hf": 10.0},
        total_budget=100.0,
    )
    loop = ExperimentDesignLoop(
        cost_model=cm,
        acquisition_fn=AcquisitionFunction.BAYESIAN,
        input_bounds=_default_bounds(),
        n_candidates=200,
        seed=7,
    )

    init_x = torch.linspace(-2.0, -1.0, 10).unsqueeze(-1)
    init_y = _quadratic_truth(init_x)

    result = loop.run_loop(
        surrogate_fn=_cheap_surrogate,
        truth_fn=_quadratic_truth,
        initial_inputs=init_x,
        initial_targets=init_y,
        n_iterations=6,
        samples_per_iter=3,
    )

    assert len(result.convergence_history) == 6
    assert cm.budget_consumed > 0
    assert cm.budget_consumed <= cm.total_budget


def test_suggest_next_returns_valid_points():
    cm = _default_cost_model(budget=500.0)
    loop = ExperimentDesignLoop(
        cost_model=cm,
        acquisition_fn=AcquisitionFunction.HYBRID,
        input_bounds=_default_bounds(),
        n_candidates=300,
        seed=0,
    )

    M = 50
    preds = torch.randn(M)
    bws = torch.rand(M) + 0.1
    existing = torch.tensor([[-1.0], [0.0], [1.0]])

    points, fidelity, reason = loop.suggest_next(preds, bws, existing, n_samples=5)

    assert points.shape[0] == 5
    assert points.shape[1] == 1
    assert fidelity in cm.fidelity_levels
    assert isinstance(reason, str)
    assert len(reason) > 0

    assert points[:, 0].min() >= -2.0
    assert points[:, 0].max() <= 2.0
