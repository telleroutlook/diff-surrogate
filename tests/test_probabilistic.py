"""Tests for probabilistic neural operator with proper scoring rule training."""

from __future__ import annotations

import torch
import torch.nn as nn

from diff_surrogate.probabilistic import (
    CRPSLoss,
    DistributionHead,
    EnergyScoreLoss,
    PNOBenchmark,
    PNOConformalPipeline,
    ProbabilisticSurrogate,
)


def _make_surrogate(in_features: int = 2, out_features: int = 1) -> ProbabilisticSurrogate:
    backbone = nn.Sequential(
        nn.Linear(in_features, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
    )
    return ProbabilisticSurrogate(backbone, in_features=16, out_features=out_features)


def test_distribution_head_shapes():
    torch.manual_seed(0)
    head = DistributionHead(in_features=16, out_features=3)
    features = torch.randn(8, 16)
    mean, scale = head(features)
    assert mean.shape == (8, 3)
    assert scale.shape == (8, 3)
    assert (scale > 0).all()


def test_energy_score_differentiable_and_positive():
    torch.manual_seed(1)
    loss_fn = EnergyScoreLoss()
    mean = torch.randn(4, 2, requires_grad=True)
    log_scale = torch.randn(4, 2, requires_grad=True)
    scale = torch.nn.functional.softplus(log_scale)
    target = torch.randn(4, 2)

    loss = loss_fn(mean, scale, target, n_samples=16)
    assert loss.item() > 0
    loss.backward()
    assert mean.grad is not None
    assert log_scale.grad is not None


def test_crps_differentiable_and_positive():
    torch.manual_seed(2)
    loss_fn = CRPSLoss()
    mean = torch.randn(8, requires_grad=True)
    log_scale = torch.randn(8, requires_grad=True)
    scale = torch.nn.functional.softplus(log_scale)
    target = torch.randn(8)

    loss = loss_fn(mean, scale, target)
    assert loss.item() > 0
    loss.backward()
    assert mean.grad is not None
    assert log_scale.grad is not None


def test_probabilistic_surrogate_valid_params():
    torch.manual_seed(3)
    surr = _make_surrogate(in_features=2, out_features=1)
    x = torch.randn(10, 2)
    mean, scale = surr(x)
    assert mean.shape == (10, 1)
    assert scale.shape == (10, 1)
    assert (scale > 0).all(), "Scale must be positive"


def test_probabilistic_surrogate_sampling():
    torch.manual_seed(4)
    surr = _make_surrogate(in_features=2, out_features=3)
    x = torch.randn(5, 2)
    samples = surr.sample(x, n_samples=32)
    assert samples.shape == (32, 5, 3)


def test_probabilistic_surrogate_predict_interval():
    torch.manual_seed(5)
    surr = _make_surrogate(in_features=2, out_features=1)
    x = torch.randn(20, 2)
    lower, upper = surr.predict_interval(x, alpha=0.1)
    assert lower.shape == upper.shape
    assert (lower <= upper).all(), "Lower must be <= upper"


def test_pno_conformal_pipeline_trains_and_calibrates():
    torch.manual_seed(6)
    surr = _make_surrogate(in_features=1, out_features=1)

    n_train, n_cal = 80, 40
    train_x = torch.randn(n_train, 1)
    train_y = train_x[:, 0] ** 2 + 0.1 * torch.randn(n_train)
    cal_x = torch.randn(n_cal, 1)
    cal_y = cal_x[:, 0] ** 2 + 0.1 * torch.randn(n_cal)

    ds = torch.utils.data.TensorDataset(train_x, train_y)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)

    pipeline = PNOConformalPipeline(surr, lr=1e-3)
    history = pipeline.train_pno(loader, n_epochs=10, scoring_rule="crps")
    assert len(history) == 10
    assert all(h > 0 or h == 0 for h in history)

    with torch.no_grad():
        cal_mean, _ = surr(cal_x)
        if cal_mean.ndim > 1 and cal_mean.shape[-1] == 1:
            cal_mean = cal_mean.squeeze(-1)
    pipeline.calibrate_conformal(cal_mean, cal_y, alpha=0.1)

    result = pipeline.predict(torch.randn(10, 1), alpha=0.1)
    assert "conformal_lower" in result
    assert "conformal_upper" in result


def test_pno_conformal_pipeline_predict_keys():
    torch.manual_seed(7)
    surr = _make_surrogate(in_features=1, out_features=1)

    n = 30
    x = torch.randn(n, 1)
    y = x[:, 0] * 2.0 + 0.1 * torch.randn(n)

    ds = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)

    pipeline = PNOConformalPipeline(surr, lr=1e-3)
    pipeline.train_pno(loader, n_epochs=5, scoring_rule="crps")

    with torch.no_grad():
        cal_mean, _ = surr(x)
        if cal_mean.ndim > 1 and cal_mean.shape[-1] == 1:
            cal_mean = cal_mean.squeeze(-1)
    pipeline.calibrate_conformal(cal_mean, y, alpha=0.1)

    result = pipeline.predict(torch.randn(5, 1))
    expected_keys = {
        "mean", "scale", "pno_lower", "pno_upper",
        "conformal_lower", "conformal_upper",
    }
    assert expected_keys.issubset(result.keys())


def test_pno_benchmark_toy_data():
    torch.manual_seed(8)
    backbone = nn.Sequential(
        nn.Linear(1, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
    )
    surr = ProbabilisticSurrogate(backbone, in_features=8, out_features=1)

    def _make_data(n: int, noise_scale: float = 0.1):
        x = torch.randn(n, 1)
        y = x[:, 0] ** 2 + noise_scale * torch.randn(n)
        return x, y

    train_x, train_y = _make_data(120)
    test_x, test_y = _make_data(60)
    ood_x, ood_y = _make_data(60, noise_scale=1.0)

    results = PNOBenchmark.run(
        surr, train_x, train_y, test_x, test_y, ood_x, ood_y,
        n_epochs=5, n_seeds=2, scoring_rule="crps",
    )

    assert "pno_conformal" in results
    assert "ensemble_conformal" in results
    assert "point_conformal" in results
    for method in results:
        assert "id_coverage" in results[method]
        assert "id_bandwidth" in results[method]
        assert "ood_coverage" in results[method]


def test_deterministic_with_seed():
    torch.manual_seed(42)
    surr1 = _make_surrogate(in_features=2, out_features=1)
    x = torch.randn(5, 2)
    mean1, scale1 = surr1(x)

    torch.manual_seed(42)
    surr2 = _make_surrogate(in_features=2, out_features=1)
    mean2, scale2 = surr2(x)

    assert torch.allclose(mean1, mean2)
    assert torch.allclose(scale1, scale2)


def test_energy_score_crps_both_scoring_rules():
    torch.manual_seed(9)
    surr = _make_surrogate(in_features=2, out_features=1)
    x = torch.randn(16, 2)
    y = x[:, 0] ** 2 + 0.1 * torch.randn(16)

    loss_e = surr.loss(x, y, scoring_rule="energy", n_samples=8)
    loss_c = surr.loss(x, y, scoring_rule="crps")

    assert loss_e.item() > 0
    assert loss_c.item() > 0
