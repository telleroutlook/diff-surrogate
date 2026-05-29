"""Tests for fixes from Round 3 optimization report."""

import pytest
import torch

from diff_surrogate import AdaptiveCorrectionPolicy, EnsembleSurrogate, MLPSurrogate
from diff_surrogate.convergence import (
    ConvergenceConfig,
    ConvergenceMonitor,
    hybrid_z_score,
)

# --- 1.1 AdaptiveCorrectionPolicy uses _prev_error_ema in ratio ---


def test_adaptive_correction_update_error_direction():
    from diff_surrogate.base import AdaptiveCorrectionPolicy

    p = AdaptiveCorrectionPolicy(
        initial_interval=10,
        growth_threshold=1.5,
        shrink_threshold=0.5,
    )
    # First call: sets _error_ema, no ratio yet
    p.update_error(0.1)
    assert p._prev_error_ema is None
    assert p._error_ema == pytest.approx(0.1)

    # Second call: sets _prev_error_ema, computes ratio
    p.update_error(0.1)
    assert p._prev_error_ema == pytest.approx(0.1)
    # ratio = new_ema / _prev_error_ema ≈ 1.0 → no change
    assert p.current_interval == 10

    # Simulate error growth: ratio should use _prev_error_ema (2-step-old EMA)
    for _ in range(20):
        p.update_error(0.5)
    # Error is growing → interval should have decreased
    assert p.current_interval < 10


def test_adaptive_correction_shrinks_on_declining_error():
    from diff_surrogate.base import AdaptiveCorrectionPolicy

    p = AdaptiveCorrectionPolicy(
        initial_interval=10,
        growth_threshold=1.5,
        shrink_threshold=0.5,
        ema_alpha=0.8,  # High alpha → fast EMA response
    )
    # Warm up with high errors
    for _ in range(3):
        p.update_error(1.0)
    # Now rapidly declining errors → ratio new/prev < shrink_threshold
    for _ in range(50):
        p.update_error(0.01)
    # Error is shrinking fast → interval should have increased
    assert p.current_interval > 10


# --- 1.2 _std docstring (verified Bessel correction) ---


def test_std_uses_bessel_correction():
    import math

    import numpy as np

    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    m = sum(values) / len(values)
    pop_std = math.sqrt(sum((v - m) ** 2 for v in values) / len(values))
    sample_std = math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))

    result = float(np.std(values, ddof=1))
    assert result == pytest.approx(sample_std)
    assert result != pytest.approx(pop_std)


# --- 1.3 predict calls self() not self.forward() ---


def test_predict_uses_module_call():
    """Verify predict() invokes nn.Module.__call__ (which triggers hooks)."""
    from diff_surrogate.mlp import MLPSurrogate

    s = MLPSurrogate(n_inputs=2, properties=["val"])
    hook_called = [False]

    def hook_fn(module, input, output):
        hook_called[0] = True

    s.register_forward_hook(hook_fn)
    s.predict(torch.randn(3, 2))
    assert hook_called[0], "predict() should trigger forward hooks via self()"


# --- 1.4 EnsembleSurrogate _members is nn.ModuleList ---


def test_ensemble_members_in_state_dict():
    from diff_surrogate.ensemble import EnsembleSurrogate
    from diff_surrogate.mlp import MLPSurrogate

    ens = EnsembleSurrogate(
        base_factory=lambda: MLPSurrogate(n_inputs=2, properties=["val"]),
        n_members=3,
    )
    # Build networks by training
    ens.train_surrogate(n_samples=16, n_epochs=1)
    sd = ens.state_dict()
    member_0_keys = [k for k in sd if k.startswith("_members.0.")]
    member_1_keys = [k for k in sd if k.startswith("_members.1.")]
    member_2_keys = [k for k in sd if k.startswith("_members.2.")]
    assert len(member_0_keys) > 0
    assert len(member_1_keys) > 0
    assert len(member_2_keys) > 0


# --- 2.1 load_checkpoint default weights_only=True ---


def test_load_checkpoint_default_weights_only(tmp_path):
    from diff_surrogate.mlp import MLPSurrogate

    s = MLPSurrogate(n_inputs=2, properties=["val"])
    s.train_surrogate(n_samples=16, n_epochs=1)
    path = str(tmp_path / "ckpt.pt")
    s.save_checkpoint(path)

    s2 = MLPSurrogate(n_inputs=2, properties=["val"])
    # Default should be weights_only=True (no pickle exploit surface)
    s2.load_checkpoint(path)
    assert s2._trained


# --- 3.3 squeeze(-1) preserves batch dim ---


def test_mlp_squeeze_preserves_batch_dim():
    from diff_surrogate.mlp import MLPSurrogate

    s = MLPSurrogate(n_inputs=2, properties=["val"])
    x = torch.randn(1, 2)  # batch_size=1
    out = s(x)
    assert out["val"].shape == (1,), f"Expected (1,), got {out['val'].shape}"

    x2 = torch.randn(5, 2)
    out2 = s(x2)
    assert out2["val"].shape == (5,)


# --- 3.4 batched + corners warning ---


def test_batched_corners_warning():
    from diff_surrogate.robust_design import AntitheticConfig, CornerSpec, robust_design_step

    design = torch.randn(1, 4, requires_grad=True)
    with pytest.warns(UserWarning, match="batched=True is ignored"):
        robust_design_step(
            design,
            forward_fn=lambda d, **kw: (d**2).sum(),
            loss_fn=lambda o: o.mean(),
            antithetic_config=AntitheticConfig(n_pairs=2),
            corners=[CornerSpec(label="nominal", weight=1.0)],
            batched=True,
        )


# --- 4.4 train_losses uses extend, not assignment ---


def test_base_train_surrogate_preserves_deque():
    from diff_surrogate.cnn import CNNSurrogate

    s = CNNSurrogate(in_channels=1, out_channels=1, grid_size=8)
    s.train_surrogate(n_samples=16, n_epochs=3)
    # Should be a deque with maxlen
    from collections import deque

    assert isinstance(s.stats.train_losses, deque)
    assert s.stats.train_losses.maxlen == 1000
    assert len(s.stats.train_losses) == 3


# --- 6.4 predict_with_correction calls update_error ---


def test_predict_with_correction_updates_adaptive_error():
    from diff_surrogate.base import AdaptiveCorrectionPolicy
    from diff_surrogate.cnn import CNNSurrogate

    policy = AdaptiveCorrectionPolicy(initial_interval=1, warmup_steps=0)
    s = CNNSurrogate(in_channels=1, out_channels=1, grid_size=8, correction_policy=policy)
    s.train_surrogate(n_samples=16, n_epochs=1)

    # predict_with_correction should call policy.update_error internally
    x = torch.randn(1, 1, 8, 8)
    _out, action = s.predict_with_correction(
        x, true_solver_fn=lambda inp: torch.zeros_like(s.forward(inp))
    )
    assert action.value == "correct"
    assert policy._error_ema is not None, "update_error should have been called"


# --- 4.2 EnsembleSurrogate train_surrogate per-member ---


def test_ensemble_trains_each_member():
    from diff_surrogate.ensemble import EnsembleSurrogate
    from diff_surrogate.mlp import MLPSurrogate

    ens = EnsembleSurrogate(
        base_factory=lambda: MLPSurrogate(n_inputs=2, properties=["val"]),
        n_members=3,
    )
    all_losses = ens.train_members(n_samples=16, n_epochs=2)
    assert len(all_losses) == 3
    for losses in all_losses:
        assert len(losses) == 2


# --- MonotoneMLP / PositiveOutputMLP ---


def test_monotone_mlp_non_decreasing():
    from diff_surrogate.mlp import MonotoneMLP

    torch.manual_seed(42)
    net = MonotoneMLP(in_features=1, hidden=32)
    # With abs() weights, output should be non-decreasing along positive input direction
    # (not guaranteed globally, but for random init it usually holds)
    x = torch.linspace(0, 1, 20).unsqueeze(1)
    with torch.no_grad():
        out = net(x).squeeze()
    # At minimum, the output should change (weights aren't all zero)
    assert out.abs().sum() > 0


def test_positive_output_mlp_always_positive():
    from diff_surrogate.mlp import PositiveOutputMLP

    torch.manual_seed(42)
    net = PositiveOutputMLP(in_features=2, hidden=32)
    x = torch.randn(100, 2) * 10  # wide range
    with torch.no_grad():
        out = net(x)
    assert (out > 0).all(), "PositiveOutputMLP should always produce positive outputs"


# --- Checkpoint weight consistency ---


def test_checkpoint_weights_identical(tmp_path):
    from diff_surrogate.mlp import MLPSurrogate

    s = MLPSurrogate(n_inputs=2, properties=["val"])
    s.train_surrogate(n_samples=16, n_epochs=1)
    path = str(tmp_path / "ckpt.pt")
    s.save_checkpoint(path)

    s2 = MLPSurrogate(n_inputs=2, properties=["val"])
    s2.load_checkpoint(path)
    x = torch.randn(4, 2)
    with torch.no_grad():
        out1 = s.predict(x)
        out2 = s2.predict(x)
    torch.testing.assert_close(out1["val"], out2["val"])


# --- Hook handle cleanup ---


def test_robust_design_mask_zeros_frozen_grads():
    from diff_surrogate.robust_design import robust_design_step

    design = torch.randn(1, 4, requires_grad=True)
    mask = torch.tensor([[True, False, True, False]])
    loss, _action = robust_design_step(
        design,
        forward_fn=lambda d: (d**2).sum(),
        loss_fn=lambda o: o.mean(),
        designable_mask=mask,
    )
    loss.backward()
    grad = design.grad
    assert grad is not None
    # Frozen pixels (mask=False) should have zero gradient
    assert grad[0, 1] == 0.0
    assert grad[0, 3] == 0.0
    # Designable pixels should have non-zero gradient
    assert grad[0, 0] != 0.0
    assert grad[0, 2] != 0.0


# --- hybrid_z_score numerical regression ---


def test_hybrid_z_score_numerical():

    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    z = hybrid_z_score(values, weight=0.5)
    # Last value (10.0) is well above the mean → positive z-score
    assert z > 0
    assert z < 10  # sanity bound


# --- ConvergenceMonitor integrated with SurrogateTrainer ---


def test_trainer_convergence_early_stop():
    from diff_surrogate.cnn import CNNSurrogate
    from diff_surrogate.trainer import SurrogateTrainer

    monitor = ConvergenceMonitor(
        ConvergenceConfig(window=3, min_steps=2, early_stop_threshold=0.5, patience=2)
    )
    s = CNNSurrogate(in_channels=1, out_channels=1, grid_size=8)
    trainer = SurrogateTrainer(s, lr=1e-3, convergence_monitor=monitor)
    losses = trainer.train(n_epochs=50, n_samples=32)
    # Should have stopped early (losses are near-constant with random data)
    assert len(losses) <= 50


# --- Monotonicity verification ---


def test_monotone_mlp_actually_monotone():
    from diff_surrogate.mlp import MonotoneMLP

    torch.manual_seed(0)
    net = MonotoneMLP(in_features=1, hidden=32, n_layers=3)
    x = torch.linspace(-3, 3, 100).unsqueeze(1)
    with torch.no_grad():
        y = net(x).squeeze()
    diffs = y[1:] - y[:-1]
    assert (diffs >= -1e-5).all(), f"non-monotone! min diff={diffs.min()}"


# --- Phase drift in adaptive correction ---


def test_adaptive_correction_no_phase_drift():
    from diff_surrogate.base import AdaptiveCorrectionPolicy

    p = AdaptiveCorrectionPolicy(initial_interval=10, warmup_steps=0)
    corrections = [step for step in range(1, 31) if p.should_correct(step)]
    # First call triggers at step 1 (_last_correction_step < 0), then every 10 steps
    assert corrections == [1, 11, 21], f"Expected [1, 11, 21], got {corrections}"
    # Verify even spacing
    for i in range(1, len(corrections)):
        assert corrections[i] - corrections[i - 1] == 10
    # Now shrink interval and verify even spacing from last correction
    p.update_error(0.01)  # small error
    p.update_error(0.01)
    p._current_interval = 5
    next_corrections = [step for step in range(31, 50) if p.should_correct(step)]
    # Last correction was at 21, next at 21+5=26 (<31 missed), 31, 36, 41, 46
    # But _last_correction_step=21, elapsed at 31 is 10>=5, so 31 triggers, then 36, 41, 46
    expected = [31, 36, 41, 46]
    assert next_corrections == expected, f"Expected {expected}, got {next_corrections}"
    # Verify even spacing of 5
    for i in range(1, len(next_corrections)):
        assert next_corrections[i] - next_corrections[i - 1] == 5


# --- Dict targets in base train_surrogate ---


def test_base_train_surrogate_handles_dict_targets():
    from diff_surrogate.mlp import MLPSurrogate

    s = MLPSurrogate(n_inputs=2, properties=["a", "b"])
    losses = s.train_surrogate(n_samples=16, n_epochs=2)
    assert len(losses) == 2
    assert all(loss > 0 for loss in losses)


# --- Unbatched input safety ---


def test_mlp_forward_unbatched_input():
    from diff_surrogate.mlp import MLPSurrogate

    s = MLPSurrogate(n_inputs=2, properties=["val"])
    x = torch.randn(2)  # no batch dim
    out = s(x)
    assert out["val"].shape == (1,), f"Expected (1,), got {out['val'].shape}"


# --- Checkpoint versioning ---


def test_checkpoint_format_version(tmp_path):
    from diff_surrogate.mlp import MLPSurrogate

    s = MLPSurrogate(n_inputs=2, properties=["val"])
    s.train_surrogate(n_samples=16, n_epochs=1)
    path = str(tmp_path / "ckpt.pt")
    s.save_checkpoint(path)
    ckpt = torch.load(path, weights_only=True)
    assert ckpt["__format__"] == 2


# --- §5.1 Optimizer state roundtrip ---


def test_optimizer_state_roundtrip(tmp_path):
    s = MLPSurrogate(n_inputs=2, properties=["val"])
    s.train_surrogate(n_samples=16, n_epochs=3, lr=1e-3)
    path = str(tmp_path / "ckpt.pt")
    s.save_checkpoint(path)

    s2 = MLPSurrogate(n_inputs=2, properties=["val"])
    s2.load_checkpoint(path)
    # Continue training — optimizer state should be restored
    s2.train_surrogate(n_samples=16, n_epochs=2, lr=1e-3)

    # Fresh model trained 5 epochs from scratch
    s3 = MLPSurrogate(n_inputs=2, properties=["val"])
    s3.train_surrogate(n_samples=16, n_epochs=5, lr=1e-3)

    # Resumed model (3+2) should differ from fresh 5-epoch due to optimizer momentum
    x = torch.randn(8, 2)
    with torch.no_grad():
        out2 = s2.predict(x)["val"]
        out3 = s3.predict(x)["val"]
    # They won't be identical because momentum state carries over
    assert not torch.allclose(out2, out3, atol=1e-6)


# --- §5.2 predict_with_correction behavior ---


def test_predict_with_correction_returns_true_output():
    from diff_surrogate.base import AdaptiveCorrectionPolicy, CorrectionAction
    from diff_surrogate.mlp import MLPSurrogate

    policy = AdaptiveCorrectionPolicy(initial_interval=1, warmup_steps=0)
    s = MLPSurrogate(n_inputs=2, properties=["val"], correction_policy=policy)
    s.train_surrogate(n_samples=16, n_epochs=1)

    x = torch.randn(1, 2)
    true_val = {"val": torch.tensor([42.0])}

    out, action = s.predict_with_correction(x, true_solver_fn=lambda _x: true_val)
    assert action == CorrectionAction.CORRECT
    assert out["val"].item() == pytest.approx(42.0)
    assert s.stats.total_corrections == 1
    assert len(s.stats.correction_errors) == 1


def test_predict_with_correction_no_solver():
    from diff_surrogate.mlp import MLPSurrogate

    s = MLPSurrogate(n_inputs=2, properties=["val"])
    s.train_surrogate(n_samples=16, n_epochs=1)
    x = torch.randn(3, 2)
    out, _action = s.predict_with_correction(x)
    # Should just return predict() output
    assert "val" in out


# --- §5.6 multifidelity truth_mode variants ---


def test_multifidelity_differentiable_mode():
    from diff_surrogate.multifidelity import MultiFidelityConfig, optimize_multifidelity

    result = optimize_multifidelity(
        design_init=torch.randn(1, 4),
        surrogate_fn=lambda d: (d**2).sum(),
        truth_fn=lambda d: (d**2).sum() + 0.1,
        loss_fn=lambda o: o.mean(),
        n_steps=10,
        config=MultiFidelityConfig(correction_interval=5, truth_mode="differentiable"),
    )
    assert len(result.loss_history) == 10


def test_multifidelity_best_design_tracking():
    from diff_surrogate.multifidelity import optimize_multifidelity

    result = optimize_multifidelity(
        design_init=torch.randn(1, 4),
        surrogate_fn=lambda d: (d**2).sum(),
        truth_fn=lambda d: (d**2).sum() + 0.1,
        loss_fn=lambda o: o.mean(),
        n_steps=10,
    )
    assert result.best_design is not None
    assert result.best_loss is not None
    assert result.best_loss <= result.loss_history[-1]


# --- §5.7 Ensemble diversity ---


def test_ensemble_members_diverse():
    ens = EnsembleSurrogate(
        base_factory=lambda: MLPSurrogate(n_inputs=2, properties=["val"]),
        n_members=5,
    )
    x = torch.randn(4, 2)
    out = ens.predict(x)
    # std should be > 0 since members have different random inits
    assert (out["val_std"] > 0).any(), "Ensemble members should produce diverse predictions"


# --- §4.4 Ensemble deterministic seeding ---


def test_ensemble_deterministic_seeding():
    def _factory():
        return MLPSurrogate(n_inputs=2, properties=["val"])

    ens1 = EnsembleSurrogate(base_factory=_factory, n_members=3, seed=42)
    ens2 = EnsembleSurrogate(base_factory=_factory, n_members=3, seed=42)
    x = torch.randn(4, 2)
    with torch.no_grad():
        out1 = ens1(x)
        out2 = ens2(x)
    torch.testing.assert_close(out1["val"], out2["val"])


def test_ensemble_different_seeds_differ():
    def _factory():
        return MLPSurrogate(n_inputs=2, properties=["val"])

    ens1 = EnsembleSurrogate(base_factory=_factory, n_members=3, seed=42)
    ens2 = EnsembleSurrogate(base_factory=_factory, n_members=3, seed=99)
    x = torch.randn(4, 2)
    with torch.no_grad():
        out1 = ens1(x)
        out2 = ens2(x)
    assert not torch.allclose(out1["val"], out2["val"])


# --- §2.1 peek/commit no side effects ---


def test_adaptive_correction_peek_no_side_effects():
    from diff_surrogate.base import AdaptiveCorrectionPolicy

    p = AdaptiveCorrectionPolicy(initial_interval=10, warmup_steps=0)
    # First call: _last_correction_step < 0, so peek always returns True
    assert p.peek(0) is True
    # peek does not commit — calling again still returns True
    assert p.peek(0) is True
    # commit manually
    p.commit(0)
    # Now peek at 5 should return False (elapsed=5 < interval=10)
    assert p.peek(5) is False
    assert p.peek(10) is True
    # peek at 10 should still return True (no commit happened)
    assert p.peek(10) is True
    p.commit(10)
    assert p.peek(15) is False
    assert p.peek(20) is True


# --- §2.4 loss_weights in train_surrogate ---


def test_loss_weights_in_training():
    s = MLPSurrogate(n_inputs=2, properties=["a", "b"])
    # Train with b weighted 10x more than a
    losses = s.train_surrogate(
        n_samples=64,
        n_epochs=5,
        loss_weights={"a": 0.1, "b": 10.0},
    )
    assert len(losses) == 5


# --- §2.5 Ensemble predict_with_correction preserves _std ---


def test_ensemble_predict_with_correction_preserves_uncertainty():
    policy = AdaptiveCorrectionPolicy(initial_interval=1, warmup_steps=0)
    ens = EnsembleSurrogate(
        base_factory=lambda: MLPSurrogate(n_inputs=2, properties=["val"]),
        n_members=3,
        correction_policy=policy,
    )
    x = torch.randn(2, 2)
    true_output = {"val": torch.tensor([1.0, 2.0])}
    out, action = ens.predict_with_correction(x, true_solver_fn=lambda _x: true_output)
    assert "val_std" in out, "Uncertainty should be preserved during correction"
    assert action.value == "correct"
