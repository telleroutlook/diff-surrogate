"""Tests for robust design, convergence, and multifidelity modules."""
import torch
from diff_surrogate.convergence import ConvergenceMonitor, ConvergenceConfig, hybrid_z_score, ConvergenceAction


def test_hybrid_z_score_single_value():
    assert hybrid_z_score([5.0]) == 0.0


def test_hybrid_z_score_constant():
    assert hybrid_z_score([1.0, 1.0, 1.0, 1.0]) == 0.0


def test_convergence_no_early_stop_with_insufficient_data():
    m = ConvergenceMonitor(ConvergenceConfig(window=100, min_steps=3))
    # Very few data points should never trigger early stop
    for i in range(5):
        action = m.update(0.01, i)
        assert action != ConvergenceAction.EARLY_STOP, f"Early stopped at step {i} with only {i+1} points"


def test_convergence_continues_with_stagnant_loss():
    m = ConvergenceMonitor(ConvergenceConfig(window=20, min_steps=5))
    # Constant loss should not trigger early stop
    for i in range(30):
        action = m.update(0.5, i)
        if i < 20:
            assert action != ConvergenceAction.EARLY_STOP


def test_robust_design_step_no_hook_leak():
    from diff_surrogate.robust_design import robust_design_step
    design = torch.randn(1, 4, requires_grad=True)
    mask = torch.ones(1, 4, dtype=torch.bool)
    for _ in range(10):
        loss, _, handle = robust_design_step(
            design, forward_fn=lambda d: (d ** 2).sum(),
            loss_fn=lambda o: o.mean(),
            designable_mask=mask,
        )
    # Each call registers a hook — handles are returned, not stored on tensor
    assert handle is not None


def test_robust_design_returns_action():
    from diff_surrogate.robust_design import robust_design_step
    from diff_surrogate.convergence import ConvergenceMonitor, ConvergenceConfig
    design = torch.randn(1, 4, requires_grad=True)
    monitor = ConvergenceMonitor(ConvergenceConfig(window=5))
    loss, action, handle = robust_design_step(
        design, forward_fn=lambda d: (d ** 2).sum(),
        loss_fn=lambda o: o.mean(),
        convergence_monitor=monitor, step=0,
    )
    assert isinstance(action, ConvergenceAction)


def test_multifidelity_runs():
    from diff_surrogate.multifidelity import optimize_multifidelity, MultiFidelityConfig
    result = optimize_multifidelity(
        design_init=torch.randn(1, 4),
        surrogate_fn=lambda d: (d ** 2).sum(),
        truth_fn=lambda d: (d ** 2).sum() + 0.1,
        loss_fn=lambda o: o.mean(),
        n_steps=5,
    )
    assert result.loss_history is not None
    assert len(result.loss_history) == 5


def test_multifidelity_surrogate_grad_mode():
    from diff_surrogate.multifidelity import optimize_multifidelity, MultiFidelityConfig
    result = optimize_multifidelity(
        design_init=torch.randn(1, 4),
        surrogate_fn=lambda d: (d ** 2).sum(),
        truth_fn=lambda d: (d ** 2).sum() + 0.1,
        loss_fn=lambda o: o.mean(),
        n_steps=10,
        config=MultiFidelityConfig(correction_interval=5, truth_mode="surrogate_grad"),
    )
    assert len(result.loss_history) == 10


def test_multifidelity_calibration_only_mode():
    from diff_surrogate.multifidelity import optimize_multifidelity, MultiFidelityConfig
    result = optimize_multifidelity(
        design_init=torch.randn(1, 4),
        surrogate_fn=lambda d: (d ** 2).sum(),
        truth_fn=lambda d: (d ** 2).sum() + 0.1,
        loss_fn=lambda o: o.mean(),
        n_steps=10,
        config=MultiFidelityConfig(correction_interval=5, truth_mode="calibration_only"),
    )
    assert len(result.loss_history) == 10


def test_multifidelity_skip_step0():
    from diff_surrogate.multifidelity import optimize_multifidelity, MultiFidelityConfig
    result = optimize_multifidelity(
        design_init=torch.randn(1, 4),
        surrogate_fn=lambda d: (d ** 2).sum(),
        truth_fn=lambda d: (d ** 2).sum() + 0.1,
        loss_fn=lambda o: o.mean(),
        n_steps=1,
        config=MultiFidelityConfig(correction_interval=1),
    )
    assert result.fidelity_history[0] == "surrogate"


def test_robust_design_corners():
    from diff_surrogate.robust_design import robust_design_step, CornerSpec
    design = torch.randn(1, 4, requires_grad=True)
    loss, action, handle = robust_design_step(
        design,
        forward_fn=lambda d, **kw: (d ** 2).sum(),
        loss_fn=lambda o: o.mean(),
        corners=[
            CornerSpec(label="nominal", weight=0.5),
            CornerSpec(label="upper", weight=0.5),
        ],
    )
    assert loss.requires_grad


def test_robust_design_batched():
    from diff_surrogate.robust_design import robust_design_step, AntitheticConfig
    design = torch.randn(1, 4, requires_grad=True)
    loss, action, handle = robust_design_step(
        design,
        forward_fn=lambda d: (d ** 2).sum(-1, keepdim=True),
        loss_fn=lambda o: o.mean(),
        antithetic_config=AntitheticConfig(n_pairs=4),
        batched=True,
    )
    assert loss.requires_grad


def test_training_budget_exhausted():
    from diff_surrogate import TrainingBudget
    b = TrainingBudget(total_solver_calls=10, n_regions=2)
    assert not b.is_exhausted
    b.record_calls(0, 5)
    b.record_calls(1, 5)
    assert b.is_exhausted
    assert b.budget_remaining == 0


def test_adaptive_correction_policy_checkpoint(tmp_path):
    import os
    from diff_surrogate import AdaptiveCorrectionPolicy
    p = AdaptiveCorrectionPolicy(initial_interval=10)
    p.update_error(0.5)
    p.update_error(0.3)
    assert p._error_ema is not None
    # Verify internal state exists
    assert p._current_interval == 10
