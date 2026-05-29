"""Smoke tests for diff-surrogate -- verify all public APIs work."""

import torch

from diff_surrogate import (
    AdaptiveCorrectionPolicy,
    CNNSurrogate,
    ConvergenceAction,
    ConvergenceConfig,
    ConvergenceMonitor,
    CorrectionPolicy,
    EnsembleSurrogate,
    MLPSurrogate,
    SurrogateTrainer,
    TrainingBudget,
    hybrid_z_score,
)
from diff_surrogate.robust_design import robust_design_step


def test_correction_policy():
    p = CorrectionPolicy(correction_interval=10, warmup_steps=5)
    assert not p.should_correct(0)
    assert not p.should_correct(4)
    assert p.should_correct(10)
    assert p.should_correct(20)


def test_adaptive_correction_policy():
    p = AdaptiveCorrectionPolicy(initial_interval=10)
    assert p.should_correct(10)
    for _ in range(5):
        p.update_error(0.1)
    assert p.current_interval >= 2


def test_mlp_surrogate_predict():
    s = MLPSurrogate(n_inputs=2, properties=["density", "cp"])
    x = torch.randn(5, 2)
    out = s.predict(x)
    assert "density" in out
    assert "cp" in out
    assert out["density"].shape == (5,)


def test_cnn_surrogate_predict():
    s = CNNSurrogate(in_channels=1, out_channels=3, grid_size=16)
    x = torch.randn(2, 1, 16, 16)
    out = s.predict(x)
    assert out.shape == (2, 3, 16, 16)


def test_ensemble_mlp():
    def _mlp_factory():
        return MLPSurrogate(n_inputs=2, properties=["val"])

    ens = EnsembleSurrogate(base_factory=_mlp_factory, n_members=3)
    x = torch.randn(4, 2)
    means, stds = ens.predict_with_uncertainty(x)
    assert "val" in means
    assert "val" in stds


def test_ensemble_cnn():
    def _cnn_factory():
        return CNNSurrogate(in_channels=1, out_channels=2, grid_size=16)

    ens = EnsembleSurrogate(base_factory=_cnn_factory, n_members=3)
    x = torch.randn(2, 1, 16, 16)
    means, stds = ens.predict_with_uncertainty(x)
    assert "output" in means
    assert "output" in stds


def test_convergence_monitor():
    m = ConvergenceMonitor(ConvergenceConfig(window=10, min_steps=5))
    for i in range(20):
        action = m.update(1.0 / (i + 1), i)
    assert action in (
        ConvergenceAction.CONTINUE,
        ConvergenceAction.REDUCE_LR,
        ConvergenceAction.EARLY_STOP,
    )


def test_convergence_monitor_insufficient_data():
    m = ConvergenceMonitor(ConvergenceConfig(window=50, min_steps=3))
    action = m.update(0.5, 0)
    assert action == ConvergenceAction.CONTINUE  # not enough data, should not early-stop


def test_training_budget():
    b = TrainingBudget(total_solver_calls=100, n_regions=2, accuracy_target=0.01)
    assert b.allocate(0) > 0
    b.record_calls(0, 50)
    b.record_accuracy(0, 0.001)
    assert b.pressure == 0.5
    assert not b.is_exhausted


def test_checkpoint_roundtrip(tmp_path):
    s = MLPSurrogate(n_inputs=2, properties=["val"])
    x = torch.randn(5, 2)
    s.predict(x)  # advance step
    path = str(tmp_path / "ckpt.pt")
    s.save_checkpoint(path)
    s2 = MLPSurrogate(n_inputs=2, properties=["val"])
    s2.load_checkpoint(path)
    assert s2._step == s._step


def test_trainer_smoke():
    s = MLPSurrogate(
        n_inputs=2,
        properties=["val"],
        data_generator=lambda n: (torch.randn(n, 2), {"val": torch.randn(n)}),
    )
    trainer = SurrogateTrainer(s, lr=1e-3)
    losses = trainer.train(n_epochs=2, n_samples=16)
    assert len(losses) == 2
    assert all(isinstance(loss, float) for loss in losses)


def test_hybrid_z_score():
    score = hybrid_z_score([1.0, 2.0, 3.0, 4.0, 5.0])
    assert isinstance(score, float)


def test_robust_design_step():
    design = torch.randn(1, 4, requires_grad=True)
    loss, _action = robust_design_step(
        design,
        forward_fn=lambda d: (d**2).sum(),
        loss_fn=lambda o: o.mean(),
    )
    assert loss.requires_grad


def test_correction_action_in_all():
    from diff_surrogate import __all__

    assert "CorrectionAction" in __all__
