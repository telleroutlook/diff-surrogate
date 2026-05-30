"""Tests for uncertainty-triggered active learning."""

from __future__ import annotations

import torch

from diff_surrogate.active_sampling import (
    MultiFidelityActiveLearner,
    UncertaintyTriggeredSampler,
)
from diff_surrogate.ensemble import EnsembleSurrogate
from diff_surrogate.mlp import MLPSurrogate


def _quadratic_high_fidelity(x: torch.Tensor) -> torch.Tensor:
    """Test function: quadratic with a bump in the right half."""
    return x[:, 0] ** 2 + 10.0 * torch.sin(4.0 * x[:, 0])


def _quadratic_low_fidelity(x: torch.Tensor) -> torch.Tensor:
    """Cheap approximation: misses the sinusoidal bump."""
    return x[:, 0] ** 2


def _make_ensemble(
    n_inputs: int = 1,
    n_members: int = 5,
    seed: int = 42,
) -> EnsembleSurrogate:
    def factory() -> MLPSurrogate:
        return MLPSurrogate(
            n_inputs=n_inputs,
            properties=["value"],
            hidden=32,
            n_layers=3,
        )

    return EnsembleSurrogate(
        base_factory=factory,
        n_members=n_members,
        seed=seed,
    )


def _train_on_data(
    ensemble: EnsembleSurrogate,
    inputs: torch.Tensor,
    targets: dict[str, torch.Tensor],
    n_epochs: int = 100,
    lr: float = 1e-3,
) -> None:
    for member in ensemble._members:
        member._data_generator = lambda _n, _i=inputs, _t=targets: (_i, _t)  # type: ignore[assignment]
    ensemble.train_surrogate(
        n_samples=inputs.shape[0],
        n_epochs=n_epochs,
        lr=lr,
    )


def test_sampler_uncertainty():
    """Uncertainty should be higher where training data is sparse."""
    ensemble = _make_ensemble()
    bounds = torch.tensor([[-2.0, 2.0]])

    # Train only on left half [-2, 0]
    train_x = torch.linspace(-2.0, 0.0, 40).unsqueeze(-1)
    train_y = {"value": _quadratic_high_fidelity(train_x)}
    _train_on_data(ensemble, train_x, train_y, n_epochs=150, lr=1e-3)

    sampler = UncertaintyTriggeredSampler(
        ensemble=ensemble,
        input_bounds=bounds,
        n_candidates=500,
    )

    # Evaluate uncertainty in sparse (right) vs dense (left) region
    sparse_x = torch.linspace(0.5, 2.0, 50).unsqueeze(-1)
    dense_x = torch.linspace(-2.0, -0.5, 50).unsqueeze(-1)

    unc_sparse = sampler.compute_uncertainty(sparse_x)
    unc_dense = sampler.compute_uncertainty(dense_x)

    assert unc_sparse.mean() > unc_dense.mean(), (
        "Uncertainty should be higher in the sparse region"
    )


def test_sampler_suggests_diverse():
    """Suggested samples should not all be clustered at one point."""
    ensemble = _make_ensemble()
    bounds = torch.tensor([[-3.0, 3.0]])

    train_x = torch.linspace(-3.0, -1.0, 30).unsqueeze(-1)
    train_y = {"value": _quadratic_high_fidelity(train_x)}
    _train_on_data(ensemble, train_x, train_y, n_epochs=100, lr=1e-3)

    sampler = UncertaintyTriggeredSampler(
        ensemble=ensemble,
        input_bounds=bounds,
        n_candidates=1000,
        exploration_fraction=0.2,
    )

    samples = sampler.suggest_samples(n_samples=20)

    # Samples should span more than a tiny interval
    spread = samples[:, 0].max() - samples[:, 0].min()
    assert spread > 0.5, f"Samples are too clustered, spread={spread:.4f}"

    # All samples must be within bounds
    assert samples[:, 0].min() >= -3.0
    assert samples[:, 0].max() <= 3.0


def test_sampler_step():
    """One active learning step should reduce uncertainty on held-out region."""
    ensemble = _make_ensemble()
    bounds = torch.tensor([[-2.0, 2.0]])

    train_x = torch.linspace(-2.0, -0.5, 20).unsqueeze(-1)
    train_y = {"value": _quadratic_high_fidelity(train_x)}
    _train_on_data(ensemble, train_x, train_y, n_epochs=200, lr=1e-3)

    sampler = UncertaintyTriggeredSampler(
        ensemble=ensemble,
        input_bounds=bounds,
        n_candidates=500,
    )

    # Baseline uncertainty on the unseen right half
    probe_x = torch.linspace(0.0, 2.0, 100).unsqueeze(-1)
    unc_before = sampler.compute_uncertainty(probe_x).mean().item()

    result = sampler.step(
        high_fidelity_fn=_quadratic_high_fidelity,
        train_inputs=train_x,
        train_targets=train_y,
        n_samples=15,
        n_epochs=200,
        lr=1e-3,
    )

    assert result["n_new_samples"] == 15
    assert result["train_inputs"].shape[0] == 35  # 20 + 15

    unc_after = sampler.compute_uncertainty(probe_x).mean().item()
    assert unc_after < unc_before, (
        f"Uncertainty should decrease after active step: {unc_before:.6f} -> {unc_after:.6f}"
    )


def test_multifidelity_active_learner():
    """MF active learner should reach target accuracy with fewer HF evals than random."""
    bounds = torch.tensor([[-2.0, 2.0]])
    n_epochs = 200

    # --- Active learner ---
    ens_active = _make_ensemble(n_members=5, seed=0)
    initial_x = torch.linspace(-2.0, -1.0, 10).unsqueeze(-1)
    initial_y = {"value": _quadratic_high_fidelity(initial_x)}

    active_learner = MultiFidelityActiveLearner(
        low_fidelity_fn=_quadratic_low_fidelity,
        high_fidelity_fn=_quadratic_high_fidelity,
        ensemble=ens_active,
        input_bounds=bounds,
        n_candidates=500,
        exploration_fraction=0.15,
    )
    active_result = active_learner.fit_active(
        initial_inputs=initial_x,
        initial_targets=initial_y,
        n_iterations=5,
        budget_per_iter=4,
        n_epochs_per_iter=n_epochs,
        lr=1e-3,
    )

    # --- Random baseline: same number of HF evals, random placement ---
    ens_random = _make_ensemble(n_members=5, seed=0)
    rng = torch.Generator().manual_seed(999)
    random_x = torch.rand((active_result["total_hf_evals"], 1), generator=rng) * 4.0 - 2.0
    random_y = {"value": _quadratic_high_fidelity(random_x)}
    combined_x = torch.cat([initial_x, random_x], dim=0)
    combined_y = {"value": torch.cat([initial_y["value"], random_y["value"]], dim=0)}
    _train_on_data(ens_random, combined_x, combined_y, n_epochs=n_epochs, lr=1e-3)

    # Evaluate both on a uniform grid
    eval_x = torch.linspace(-2.0, 2.0, 200).unsqueeze(-1)
    eval_y_true = _quadratic_high_fidelity(eval_x)

    active_pred = ens_active.predict(eval_x)
    active_y = active_pred["value"]

    random_pred = ens_random.predict(eval_x)
    random_y_pred = random_pred["value"]

    active_mse = torch.mean((active_y - eval_y_true) ** 2).item()
    random_mse = torch.mean((random_y_pred - eval_y_true) ** 2).item()

    assert active_mse < random_mse, (
        f"Active learning MSE ({active_mse:.6f}) should be less than "
        f"random MSE ({random_mse:.6f}) with same HF budget"
    )
    assert active_result["total_hf_evals"] == 20  # 5 iters * 4 per iter


def test_uncertainty_calibration():
    """Uncertainty should correlate with actual prediction error."""
    ensemble = _make_ensemble()
    bounds = torch.tensor([[-2.0, 2.0]])

    # Train on a limited region
    train_x = torch.linspace(-2.0, 0.0, 30).unsqueeze(-1)
    train_y = {"value": _quadratic_high_fidelity(train_x)}
    _train_on_data(ensemble, train_x, train_y, n_epochs=150, lr=1e-3)

    sampler = UncertaintyTriggeredSampler(
        ensemble=ensemble,
        input_bounds=bounds,
        n_candidates=500,
    )

    eval_x = torch.linspace(-2.0, 2.0, 100).unsqueeze(-1)
    uncertainty = sampler.compute_uncertainty(eval_x)

    with torch.no_grad():
        pred = ensemble.predict(eval_x)
    error = (pred["value"] - _quadratic_high_fidelity(eval_x)).abs()

    # Pearson correlation between uncertainty and error
    unc_centered = uncertainty - uncertainty.mean()
    err_centered = error - error.mean()
    correlation = (unc_centered * err_centered).sum() / (
        unc_centered.norm() * err_centered.norm() + 1e-12
    )

    assert correlation > 0.3, (
        f"Uncertainty should positively correlate with error, got r={correlation:.4f}"
    )
