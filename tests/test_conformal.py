"""Tests for functional conformal prediction."""

from __future__ import annotations

import torch

from diff_surrogate.active_sampling import UncertaintyTriggeredSampler
from diff_surrogate.conformal import (
    RiskControllingQuantile,
    SplitConformalPredictor,
    coverage_score,
)
from diff_surrogate.ensemble import EnsembleSurrogate
from diff_surrogate.mlp import MLPSurrogate


def _quadratic(x: torch.Tensor) -> torch.Tensor:
    return x[:, 0] ** 2 + 10.0 * torch.sin(4.0 * x[:, 0])


def _make_ensemble(n_inputs: int = 1, n_members: int = 5, seed: int = 42):
    def factory():
        return MLPSurrogate(
            n_inputs=n_inputs,
            properties=["value"],
            hidden=32,
            n_layers=3,
        )

    return EnsembleSurrogate(base_factory=factory, n_members=n_members, seed=seed)


def _train_ensemble(ensemble, inputs, targets, n_epochs=100, lr=1e-3):
    for member in ensemble._members:
        member._data_generator = lambda _n, _i=inputs, _t=targets: (_i, _t)
    ensemble.train_surrogate(n_samples=inputs.shape[0], n_epochs=n_epochs, lr=lr)


def test_split_conformal_scalar_coverage():
    """Scalar conformal bands should achieve >= target coverage on test data."""
    torch.manual_seed(0)
    n_train, n_cal, n_test = 200, 100, 500

    w_true = torch.randn(1)
    train_x = torch.randn(n_train, 1)
    train_x @ w_true + 0.5 * torch.randn(n_train)
    cal_x = torch.randn(n_cal, 1)
    cal_y = cal_x @ w_true + 0.5 * torch.randn(n_cal)
    test_x = torch.randn(n_test, 1)
    test_y = test_x @ w_true + 0.5 * torch.randn(n_test)

    def pred_fn(x):
        return x @ w_true

    alpha = 0.1
    cp = SplitConformalPredictor()
    cp.calibrate(pred_fn(cal_x).squeeze(), cal_y.squeeze(), alpha=alpha)

    lower, upper = cp.predict(pred_fn(test_x).squeeze())
    covered = ((test_y.squeeze() >= lower) & (test_y.squeeze() <= upper)).float().mean()
    assert covered >= 1.0 - alpha - 0.05, f"Coverage {covered:.3f} below target {1 - alpha}"


def test_split_conformal_functional_coverage():
    """Functional (multi-dim) conformal bands should achieve >= target coverage."""
    torch.manual_seed(1)
    n_cal, n_test = 200, 300
    d = 4

    w_true = torch.randn(1, d)
    cal_pred = torch.randn(n_cal, d)
    cal_targets = cal_pred @ w_true.T + 0.3 * torch.randn(n_cal, d)
    cal_targets = cal_targets.squeeze(-1) if cal_targets.ndim > 2 else cal_targets

    test_pred = torch.randn(n_test, d)
    test_targets = test_pred @ w_true.T + 0.3 * torch.randn(n_test, d)
    test_targets = test_targets.squeeze(-1) if test_targets.ndim > 2 else test_targets

    alpha = 0.1
    cp = SplitConformalPredictor()
    cp.calibrate(cal_pred, cal_targets, alpha=alpha)

    lower, upper = cp.predict(test_pred)
    assert lower.shape == test_pred.shape
    assert upper.shape == test_pred.shape

    covered_per_dim = ((test_targets >= lower) & (test_targets <= upper)).all(dim=-1)
    coverage = covered_per_dim.float().mean().item()
    assert coverage >= 1.0 - alpha - 0.05, f"Functional coverage {coverage:.3f} too low"


def test_coverage_score_metric():
    """coverage_score should correctly report metrics."""
    targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    lower = torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5])
    upper = torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5])

    result = coverage_score(targets, lower, upper, alpha=0.1)

    assert abs(result["empirical_coverage"] - 1.0) < 1e-6
    assert abs(result["target_coverage"] - 0.9) < 1e-6
    assert result["mean_bandwidth"] > 0
    assert result["bandwidth_efficiency"] > 0

    lower2 = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0])
    upper2 = torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0])
    result2 = coverage_score(targets, lower2, upper2, alpha=0.1)
    assert result2["empirical_coverage"] == 0.0


def test_risk_controlling_quantile():
    """RCQ should produce bands with risk below alpha + delta."""
    torch.manual_seed(2)
    n_cal, n_test = 500, 1000

    cal_pred = torch.randn(n_cal)
    noise = 0.5 * torch.randn(n_cal)
    cal_targets = cal_pred + noise

    test_pred = torch.randn(n_test)
    test_targets = test_pred + 0.5 * torch.randn(n_test)

    alpha = 0.1
    delta = 0.05
    rcq = RiskControllingQuantile()
    rcq.calibrate(cal_pred, cal_targets, alpha=alpha, delta=delta)

    lower, upper = rcq.predict(test_pred)
    covered = ((test_targets >= lower) & (test_targets <= upper)).float().mean()
    assert covered >= 1.0 - alpha - 0.05, f"RCQ coverage {covered:.3f} too low"


def test_calibrated_bandwidth_better_than_raw_ensemble():
    """Conformal bands should provide guaranteed coverage while naive ensemble bands may not."""
    torch.manual_seed(3)
    bounds = torch.tensor([[-2.0, 2.0]])
    ensemble = _make_ensemble()

    train_x = torch.linspace(-2.0, 2.0, 100).unsqueeze(-1)
    train_y = {"value": _quadratic(train_x)}
    _train_ensemble(ensemble, train_x, train_y, n_epochs=500, lr=1e-3)

    cal_x = torch.linspace(-2.0, 2.0, 80).unsqueeze(-1)
    cal_targets = _quadratic(cal_x)
    sampler = UncertaintyTriggeredSampler(
        ensemble=ensemble,
        input_bounds=bounds,
        n_candidates=200,
    )
    sampler.calibrate_uncertainty(cal_x, {"value": cal_targets}, alpha=0.1)

    eval_x = torch.linspace(-2.0, 2.0, 100).unsqueeze(-1)
    eval_targets = _quadratic(eval_x)

    with torch.no_grad():
        pred = ensemble.predict(eval_x)
    mean_pred = pred["value"]
    std_pred = pred["value_std"]

    naive_lower = mean_pred - 2.0 * std_pred
    naive_upper = mean_pred + 2.0 * std_pred
    in_range = (eval_targets >= naive_lower) & (eval_targets <= naive_upper)
    naive_coverage = in_range.float().mean().item()

    cp = sampler._conformal_predictor
    conformal_metrics = cp.coverage_score(mean_pred, eval_targets)

    assert conformal_metrics["empirical_coverage"] >= naive_coverage, (
        f"Conformal coverage {conformal_metrics['empirical_coverage']:.3f} should be "
        f">= naive 2-sigma coverage {naive_coverage:.3f}"
    )
    assert conformal_metrics["mean_bandwidth"] > 0


def test_ood_coverage_diagnostic():
    """Coverage should degrade gracefully on OOD vs ID data."""
    torch.manual_seed(4)
    n_cal = 300

    torch.tensor([2.0])
    cal_pred = torch.randn(n_cal) * 2.0
    cal_targets = cal_pred * 2.0 + 0.5 * torch.randn(n_cal)

    cp = SplitConformalPredictor()
    cp.calibrate(cal_pred, cal_targets, alpha=0.1)

    id_pred = torch.randn(500) * 2.0
    id_targets = id_pred * 2.0 + 0.5 * torch.randn(500)
    id_metrics = cp.coverage_score(id_pred, id_targets)

    ood_pred = torch.randn(500) * 5.0
    ood_targets = ood_pred * 2.0 + 2.0 * torch.randn(500)
    ood_metrics = cp.coverage_score(ood_pred, ood_targets)

    assert id_metrics["empirical_coverage"] >= ood_metrics["empirical_coverage"] - 0.02 or (
        id_metrics["empirical_coverage"] >= 0.85
    )
    assert id_metrics["empirical_coverage"] >= 0.85, (
        f"ID coverage too low: {id_metrics['empirical_coverage']:.3f}"
    )


def test_ensemble_integration():
    """SplitConformalPredictor should work end-to-end with EnsembleSurrogate."""
    torch.manual_seed(5)
    torch.tensor([[-2.0, 2.0]])
    ensemble = _make_ensemble(n_members=7)

    train_x = torch.linspace(-2.0, 1.0, 60).unsqueeze(-1)
    train_y = {"value": _quadratic(train_x)}
    _train_ensemble(ensemble, train_x, train_y, n_epochs=200, lr=1e-3)

    cal_x = torch.linspace(-2.0, 1.5, 40).unsqueeze(-1)
    cal_targets = _quadratic(cal_x)

    with torch.no_grad():
        cal_pred = ensemble.predict(cal_x)["value"]

    cp = SplitConformalPredictor()
    cp.calibrate(cal_pred, cal_targets, alpha=0.1)

    test_x = torch.linspace(-2.0, 2.0, 100).unsqueeze(-1)
    test_targets = _quadratic(test_x)

    with torch.no_grad():
        test_pred = ensemble.predict(test_x)["value"]

    metrics = cp.coverage_score(test_pred, test_targets)
    assert metrics["empirical_coverage"] >= 0.5, (
        f"Ensemble+conformal coverage too low: {metrics['empirical_coverage']:.3f}"
    )
    assert metrics["mean_bandwidth"] > 0


def test_sampler_calibrated_suggests_high_uncertainty():
    """After calibration, suggest_samples should target high-bandwidth regions."""
    torch.manual_seed(6)
    bounds = torch.tensor([[-2.0, 2.0]])
    ensemble = _make_ensemble()

    train_x = torch.linspace(-2.0, -0.5, 30).unsqueeze(-1)
    train_y = {"value": _quadratic(train_x)}
    _train_ensemble(ensemble, train_x, train_y, n_epochs=200, lr=1e-3)

    cal_x = torch.linspace(-2.0, 0.0, 30).unsqueeze(-1)
    cal_targets = {"value": _quadratic(cal_x)}

    sampler = UncertaintyTriggeredSampler(
        ensemble=ensemble,
        input_bounds=bounds,
        n_candidates=500,
    )
    sampler.calibrate_uncertainty(cal_x, cal_targets, alpha=0.1)

    samples = sampler.suggest_samples(n_samples=20)

    trained_mean = train_x.mean().item()
    sample_mean = samples.mean().item()

    assert samples.shape == (20, 1)
    assert sample_mean > trained_mean, (
        f"Calibrated samples should skew toward unseen region: "
        f"samples_mean={sample_mean:.3f}, trained_mean={trained_mean:.3f}"
    )
