"""Tests for decision-grade UQ gates."""

from __future__ import annotations

import torch

from diff_surrogate.decision import (
    AcceptRejectGate,
    CVaRRiskBudget,
    CoverageTriggeredEarlyStop,
    DecisionGate,
    DecisionVerdict,
    MultiCandidateDecision,
)


def test_accept_reject_gate_accepts_within_tolerance():
    gate = AcceptRejectGate(max_bandwidth=1.0)
    pred = torch.tensor([5.0, 6.0, 7.0])
    lower = torch.tensor([4.6, 5.6, 6.6])
    upper = torch.tensor([5.4, 6.4, 7.4])
    verdict, reasons = gate.evaluate(pred, lower, upper)
    assert verdict == DecisionVerdict.ACCEPT
    assert reasons["mean_bandwidth"] < 1.0


def test_accept_reject_gate_rejects_large_bandwidth():
    gate = AcceptRejectGate(max_bandwidth=0.5)
    pred = torch.tensor([5.0, 6.0, 7.0])
    lower = torch.tensor([3.0, 4.0, 5.0])
    upper = torch.tensor([7.0, 8.0, 9.0])
    verdict, reasons = gate.evaluate(pred, lower, upper)
    assert verdict == DecisionVerdict.REJECT
    assert reasons.get("bandwidth_exceeded") is True


def test_accept_reject_gate_rejects_below_threshold():
    gate = AcceptRejectGate(min_threshold=3.0)
    pred = torch.tensor([5.0, 2.5, 7.0])
    lower = torch.tensor([4.8, 2.3, 6.8])
    upper = torch.tensor([5.2, 2.7, 7.2])
    verdict, reasons = gate.evaluate(pred, lower, upper)
    assert verdict == DecisionVerdict.REJECT
    assert reasons.get("below_min_threshold") is True


def test_accept_reject_gate_uncertain_when_coverage_low():
    gate = AcceptRejectGate(min_coverage=0.9)
    pred = torch.tensor([5.0, 6.0])
    lower = torch.tensor([4.8, 5.8])
    upper = torch.tensor([5.2, 6.2])
    verdict, reasons = gate.evaluate(pred, lower, upper, coverage=0.7)
    assert verdict == DecisionVerdict.UNCERTAIN
    assert reasons.get("coverage_insufficient") is True


def test_cvar_risk_budget_allocates():
    budget = CVaRRiskBudget()
    alloc = budget.allocate(n_decisions=5, total_risk_budget=10.0, confidence_level=0.95)
    assert alloc.shape == (5,)
    assert (alloc > 0).all()
    assert alloc.sum().item() <= 10.0 + 1e-6


def test_cvar_risk_budget_tracks_remaining():
    budget = CVaRRiskBudget()
    budget.allocate(n_decisions=4, total_risk_budget=10.0, confidence_level=0.95)
    budget.consume(0, 0.3)
    budget.consume(1, 0.2)
    remaining = budget.remaining_budget()
    assert abs(remaining - (10.0 - 0.5)) < 1e-6


def test_cvar_risk_budget_can_afford():
    budget = CVaRRiskBudget()
    budget.allocate(n_decisions=4, total_risk_budget=1.0, confidence_level=0.95)
    assert budget.can_afford(0.01)
    assert not budget.can_afford(100.0)


def test_cvar_risk_budget_cannot_afford_when_exhausted():
    budget = CVaRRiskBudget()
    budget.allocate(n_decisions=2, total_risk_budget=1.0, confidence_level=0.95)
    budget.consume(0, 0.4)
    budget.consume(1, 0.4)
    assert not budget.can_afford(0.01)


def test_early_stop_triggers_when_stable():
    stopper = CoverageTriggeredEarlyStop(
        target_coverage=0.95, patience=3, min_bandwidth_change=1e-4
    )
    coverage = [0.92, 0.94, 0.95, 0.96, 0.96, 0.96, 0.96, 0.96]
    bandwidth = [1.0, 0.8, 0.6, 0.5, 0.50001, 0.50002, 0.50003, 0.50004]
    should, reason = stopper.should_stop(7, coverage, bandwidth)
    assert should
    assert "stable" in reason


def test_early_stop_continues_when_not_converged():
    stopper = CoverageTriggeredEarlyStop(
        target_coverage=0.95, patience=3, min_bandwidth_change=1e-4
    )
    coverage = [0.90, 0.92, 0.93]
    bandwidth = [1.0, 0.8, 0.6]
    should, reason = stopper.should_stop(2, coverage, bandwidth)
    assert not should


def test_decision_gate_combines_all():
    gate = DecisionGate(
        accept_reject=AcceptRejectGate(max_bandwidth=1.0),
        risk_budget=CVaRRiskBudget(),
        early_stop=CoverageTriggeredEarlyStop(target_coverage=0.95, patience=3),
    )
    gate.risk_budget.allocate(n_decisions=10, total_risk_budget=100.0)

    pred = torch.tensor([5.0])
    lower = torch.tensor([4.7])
    upper = torch.tensor([5.3])
    verdict, metrics = gate.evaluate_candidate(pred, lower, upper, iteration=1)
    assert verdict == DecisionVerdict.ACCEPT
    assert metrics["iteration"] == 1


def test_decision_gate_rejects_on_risk():
    gate = DecisionGate(
        accept_reject=AcceptRejectGate(max_bandwidth=100.0),
        risk_budget=CVaRRiskBudget(),
    )
    gate.risk_budget.allocate(n_decisions=2, total_risk_budget=1.0)
    gate.risk_budget.consume(0, 0.4)
    gate.risk_budget.consume(1, 0.4)

    pred = torch.tensor([5.0])
    lower = torch.tensor([4.9])
    upper = torch.tensor([5.1])
    verdict, metrics = gate.evaluate_candidate(pred, lower, upper, iteration=1)
    assert verdict == DecisionVerdict.REJECT


def test_decision_gate_should_continue():
    gate = DecisionGate(
        early_stop=CoverageTriggeredEarlyStop(target_coverage=0.95, patience=2),
    )
    history = {"coverage": [0.90, 0.92], "bandwidth": [1.0, 0.8]}
    assert gate.should_continue(1, history)

    history2 = {
        "coverage": [0.90, 0.95, 0.96, 0.96, 0.96],
        "bandwidth": [1.0, 0.6, 0.50001, 0.50002, 0.50003],
    }
    assert not gate.should_continue(4, history2)


def test_decision_gate_get_state():
    gate = DecisionGate()
    gate.risk_budget.allocate(n_decisions=3, total_risk_budget=10.0)
    gate.risk_budget.consume(0, 1.0)
    state = gate.get_state()
    assert abs(state["risk_remaining"] - 9.0) < 1e-6
    assert state["n_decisions"] == 3
    assert len(state["risk_consumed"]) == 1


def test_multi_candidate_selects_worst_case_winner():
    torch.manual_seed(42)
    selector = MultiCandidateDecision()
    candidates = torch.randn(5, 3)
    predictions = torch.tensor([10.0, 8.0, 12.0, 9.0, 11.0])
    lowers = torch.tensor([9.0, 6.0, 10.0, 7.0, 9.5])
    uppers = torch.tensor([11.0, 10.0, 14.0, 11.0, 12.5])

    best_idx, scores, verdicts = selector.select(
        candidates, predictions, lowers, uppers, maximize=True
    )
    assert best_idx == 2
    assert scores[2] == lowers[2].item()
    assert len(verdicts) == 5


def test_multi_candidate_minimize():
    selector = MultiCandidateDecision()
    candidates = torch.randn(4, 2)
    predictions = torch.tensor([3.0, 5.0, 2.0, 4.0])
    lowers = torch.tensor([2.0, 4.0, 1.0, 3.0])
    uppers = torch.tensor([4.0, 6.0, 3.0, 5.0])

    best_idx, scores, verdicts = selector.select(
        candidates, predictions, lowers, uppers, maximize=False
    )
    assert best_idx == 2
    assert scores[2] == uppers[2].item()


def test_deterministic_with_seed():
    torch.manual_seed(123)
    selector = MultiCandidateDecision()
    cands = torch.randn(10, 2)
    preds = torch.randn(10)
    lowers = preds - 0.5
    uppers = preds + 0.5

    torch.manual_seed(123)
    best2, scores2, _ = selector.select(cands.clone(), preds.clone(), lowers.clone(), uppers.clone(), maximize=True)
    best1, scores1, _ = selector.select(cands, preds, lowers, uppers, maximize=True)

    assert best1 == best2
    assert torch.allclose(scores1, scores2)
