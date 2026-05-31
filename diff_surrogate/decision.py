"""Decision-grade UQ gates: accept/reject, early stopping, risk budget.

Turns calibrated uncertainty (conformal/PNO bandwidth) into explicit
decision rules consumed by downstream inversion/optimization loops.

References:
    - Decision-theoretic conformal prediction, arXiv:2402.01960
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch
from torch import Tensor


class DecisionVerdict(Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    UNCERTAIN = "uncertain"


@dataclass
class AcceptRejectGate:
    """Rejects candidates whose conformal bands exceed tolerances."""

    min_coverage: float = 0.9
    max_bandwidth: Optional[float] = None
    min_threshold: Optional[float] = None

    def evaluate(
        self,
        predictions: Tensor,
        lower: Tensor,
        upper: Tensor,
        coverage: Optional[float] = None,
    ) -> tuple[DecisionVerdict, dict[str, str | float | bool]]:
        bandwidth = (upper - lower).abs()
        if bandwidth.ndim > 1:
            bandwidth = bandwidth.mean(dim=-1)
        mean_bw = bandwidth.mean().item()

        reasons: dict[str, str | float | bool] = {
            "mean_bandwidth": mean_bw,
            "mean_prediction": predictions.mean().item(),
        }
        verdict = DecisionVerdict.ACCEPT

        if coverage is not None and coverage < self.min_coverage:
            verdict = DecisionVerdict.UNCERTAIN
            reasons["coverage_insufficient"] = True
            reasons["coverage"] = coverage
            return verdict, reasons

        if self.max_bandwidth is not None and mean_bw > self.max_bandwidth:
            verdict = DecisionVerdict.REJECT
            reasons["bandwidth_exceeded"] = True

        if self.min_threshold is not None:
            if lower.ndim > 1:
                lb = lower.min(dim=-1).values
            else:
                lb = lower
            if (lb < self.min_threshold).any():
                verdict = DecisionVerdict.REJECT
                reasons["below_min_threshold"] = True

        return verdict, reasons


class CVaRRiskBudget:
    """Allocates a total risk budget across sequential decisions via CVaR."""

    def __init__(self) -> None:
        self._total_budget: float = 0.0
        self._per_decision: float = 0.0
        self._confidence: float = 0.95
        self._consumed: list[float] = []
        self._n_decisions: int = 0
        self._worst_losses: list[float] = []

    def allocate(
        self,
        n_decisions: int,
        total_risk_budget: float,
        confidence_level: float = 0.95,
    ) -> Tensor:
        self._n_decisions = n_decisions
        self._total_budget = total_risk_budget
        self._confidence = confidence_level
        self._consumed = []
        self._worst_losses = []

        tail_fraction = 1.0 - confidence_level
        if tail_fraction > 0 and n_decisions > 0:
            cvar_multiplier = 1.0 / tail_fraction
            per_raw = total_risk_budget / (n_decisions * cvar_multiplier)
        else:
            per_raw = total_risk_budget / max(n_decisions, 1)
        self._per_decision = per_raw

        return torch.full((n_decisions,), per_raw)

    def consume(self, decision_idx: int, realized_loss: float) -> None:
        self._consumed.append(realized_loss)
        if realized_loss > self._per_decision:
            self._worst_losses.append(realized_loss)

    def remaining_budget(self) -> float:
        spent = sum(self._consumed)
        return self._total_budget - spent

    def can_afford(self, proposed_loss: float) -> bool:
        if self._n_decisions == 0:
            return False
        remaining = self.remaining_budget()
        remaining_decisions = self._n_decisions - len(self._consumed)
        if remaining_decisions <= 0:
            return False
        budget_after = remaining - proposed_loss
        min_reserve = remaining_decisions * self._per_decision * 0.1
        return budget_after >= min_reserve


@dataclass
class CoverageTriggeredEarlyStop:
    """Stops when target coverage is achieved and bandwidth stabilizes."""

    target_coverage: float = 0.95
    patience: int = 5
    min_bandwidth_change: float = 1e-4

    def should_stop(
        self,
        iteration: int,
        coverage_history: list[float],
        bandwidth_history: list[float],
    ) -> tuple[bool, str]:
        if len(coverage_history) == 0:
            return False, "no_history"

        current_coverage = coverage_history[-1]
        if current_coverage < self.target_coverage:
            return False, "coverage_below_target"

        if len(bandwidth_history) < 2:
            return False, "insufficient_bandwidth_history"

        n_stable = 0
        for i in range(len(bandwidth_history) - 1, 0, -1):
            change = abs(bandwidth_history[i] - bandwidth_history[i - 1])
            if change < self.min_bandwidth_change:
                n_stable += 1
            else:
                break

        if n_stable >= self.patience:
            return True, "coverage_achieved_bandwidth_stable"

        return False, "bandwidth_not_stable"


@dataclass
class DecisionGate:
    """Unified gate combining accept/reject, risk budget, and early stopping."""

    accept_reject: AcceptRejectGate = field(default_factory=AcceptRejectGate)
    risk_budget: CVaRRiskBudget = field(default_factory=CVaRRiskBudget)
    early_stop: CoverageTriggeredEarlyStop = field(
        default_factory=CoverageTriggeredEarlyStop
    )

    def evaluate_candidate(
        self,
        predictions: Tensor,
        lower: Tensor,
        upper: Tensor,
        iteration: int,
        coverage: Optional[float] = None,
    ) -> tuple[DecisionVerdict, dict]:
        verdict, reasons = self.accept_reject.evaluate(
            predictions, lower, upper, coverage=coverage
        )

        metrics: dict = {
            "verdict": verdict,
            "iteration": iteration,
            "accept_reject_reasons": reasons,
        }

        if verdict == DecisionVerdict.ACCEPT and self.risk_budget._n_decisions > 0:
            mean_bw = reasons.get("mean_bandwidth", 0.0)
            if isinstance(mean_bw, float):
                can = self.risk_budget.can_afford(mean_bw)
                metrics["risk_affordable"] = can
                if not can:
                    verdict = DecisionVerdict.REJECT
                    metrics["verdict"] = verdict

        return verdict, metrics

    def should_continue(
        self,
        iteration: int,
        history: dict[str, list[float]],
    ) -> bool:
        stop, _ = self.early_stop.should_stop(
            iteration,
            history.get("coverage", []),
            history.get("bandwidth", []),
        )
        return not stop

    def get_state(self) -> dict:
        return {
            "risk_remaining": self.risk_budget.remaining_budget(),
            "risk_consumed": list(self.risk_budget._consumed),
            "risk_total": self.risk_budget._total_budget,
            "n_decisions": self.risk_budget._n_decisions,
        }


@dataclass
class MultiCandidateDecision:
    """Selects the best candidate using worst-case quantile FoM."""

    min_coverage: float = 0.9

    def select(
        self,
        candidates: Tensor,
        predictions: Tensor,
        lowers: Tensor,
        uppers: Tensor,
        maximize: bool = True,
    ) -> tuple[int, Tensor, list[DecisionVerdict]]:
        n = candidates.shape[0]
        scores = torch.zeros(n)
        verdicts: list[DecisionVerdict] = []

        for i in range(n):
            bw = (uppers[i] - lowers[i]).abs().mean().item()
            if maximize:
                scores[i] = lowers[i].mean().item()
            else:
                scores[i] = uppers[i].mean().item()
            verdicts.append(DecisionVerdict.ACCEPT)

        if maximize:
            best_idx = scores.argmax().item()
        else:
            best_idx = scores.argmin().item()

        return int(best_idx), scores, verdicts
