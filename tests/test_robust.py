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
    for _ in range(10):
        loss, _ = robust_design_step(
            design, forward_fn=lambda d: (d ** 2).sum(),
            loss_fn=lambda o: o.mean(),
        )
    # Count hooks -- should be exactly 1, not 10
    hook_count = len(design._backward_hooks) if design._backward_hooks else 0
    assert hook_count <= 1, f"Expected at most 1 hook, got {hook_count}"


def test_robust_design_returns_action():
    from diff_surrogate.robust_design import robust_design_step
    from diff_surrogate.convergence import ConvergenceMonitor, ConvergenceConfig
    design = torch.randn(1, 4, requires_grad=True)
    monitor = ConvergenceMonitor(ConvergenceConfig(window=5))
    loss, action = robust_design_step(
        design, forward_fn=lambda d: (d ** 2).sum(),
        loss_fn=lambda o: o.mean(),
        convergence_monitor=monitor, step=0,
    )
    assert isinstance(action, ConvergenceAction)


def test_multifidelity_runs():
    from diff_surrogate.multifidelity import optimize_multifidelity
    result = optimize_multifidelity(
        design_init=torch.randn(1, 4),
        surrogate_fn=lambda d: (d ** 2).sum(),
        truth_fn=lambda d: (d ** 2).sum() + 0.1,
        loss_fn=lambda o: o.mean(),
        n_steps=5,
    )
    assert result.loss_history is not None
    assert len(result.loss_history) == 5
