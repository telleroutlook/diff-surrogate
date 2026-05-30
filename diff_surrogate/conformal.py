"""Functional conformal prediction for surrogate uncertainty quantification.

Provides distribution-free, finite-sample coverage guarantees via split
conformal prediction. Works with both scalar and multi-dimensional
(functional) outputs.
"""

from __future__ import annotations

import torch
from torch import Tensor


class SplitConformalPredictor:
    """Split conformal predictor with finite-sample coverage guarantee.

    After calibration on a held-out set, produces prediction bands that
    cover unseen targets with probability at least 1 - alpha.
    """

    def __init__(self) -> None:
        self._quantile: Tensor | None = None
        self._alpha: float = 0.1

    def calibrate(
        self,
        cal_predictions: Tensor,
        cal_targets: Tensor,
        alpha: float = 0.1,
    ) -> None:
        """Calibrate on a held-out set.

        Args:
            cal_predictions: Model predictions, shape ``(N,)`` or ``(N, d)``.
            cal_targets: Ground truth, same shape as *cal_predictions*.
            alpha: Target miscoverage rate (0.1 = 90% coverage).
        """
        if cal_predictions.shape != cal_targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {cal_predictions.shape} "
                f"vs targets {cal_targets.shape}"
            )
        n = cal_predictions.shape[0]
        if n == 0:
            raise ValueError("Calibration set must be non-empty")

        scores = (cal_predictions - cal_targets).abs()
        if scores.ndim > 1:
            scores = scores.max(dim=-1).values

        q_level = min(torch.ceil(torch.tensor((1.0 - alpha) * (n + 1))).item() / n, 1.0)
        self._quantile = torch.quantile(scores, q_level)
        self._alpha = alpha

    def predict(self, predictions: Tensor) -> tuple[Tensor, Tensor]:
        """Return (lower, upper) prediction bands.

        Args:
            predictions: Model predictions, shape ``(N,)`` or ``(N, d)``.

        Returns:
            Tuple of (lower, upper) tensors with the same shape as input.
        """
        if self._quantile is None:
            raise RuntimeError("Must call calibrate() before predict()")
        q = self._quantile.to(predictions.device)
        return predictions - q, predictions + q

    def coverage_score(
        self,
        test_predictions: Tensor,
        test_targets: Tensor,
    ) -> dict[str, float]:
        """Evaluate empirical coverage and bandwidth on a test set.

        Returns:
            Dict with ``empirical_coverage``, ``target_coverage``,
            ``mean_bandwidth``, ``bandwidth_efficiency``.
        """
        lower, upper = self.predict(test_predictions)
        return coverage_score(test_targets, lower, upper, self._alpha)


class RiskControllingQuantile:
    """Calibrated quantile bands that control risk at level alpha.

    Uses a holdout-based approach to find the smallest quantile width
    such that the expected loss (fraction of points outside the band)
    is at most alpha with high probability (1 - delta).
    """

    def __init__(self) -> None:
        self._multiplier: Tensor | None = None
        self._alpha: float = 0.1

    def calibrate(
        self,
        cal_predictions: Tensor,
        cal_targets: Tensor,
        alpha: float = 0.1,
        delta: float = 0.05,
    ) -> None:
        """Calibrate the quantile multiplier.

        Args:
            cal_predictions: Model predictions ``(N,)`` or ``(N, d)``.
            cal_targets: Ground truth, same shape.
            alpha: Target risk level.
            delta: Tolerance for risk violation.
        """
        if cal_predictions.shape != cal_targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {cal_predictions.shape} "
                f"vs targets {cal_targets.shape}"
            )
        n = cal_predictions.shape[0]
        if n == 0:
            raise ValueError("Calibration set must be non-empty")

        residuals = (cal_predictions - cal_targets).abs()
        if residuals.ndim > 1:
            residuals = residuals.max(dim=-1).values

        sorted_res = torch.sort(residuals).values

        log_term = torch.log(torch.tensor(2.0 / delta, device=sorted_res.device))
        penalty = torch.sqrt((log_term) / (2.0 * n))

        target_risk = alpha - penalty
        if target_risk <= 0:
            self._multiplier = sorted_res[-1]
        else:
            quantile_level = 1.0 - target_risk
            quantile_level = torch.clamp(quantile_level, max=1.0)
            self._multiplier = torch.quantile(sorted_res, quantile_level)

        self._alpha = alpha

    def predict(self, predictions: Tensor) -> tuple[Tensor, Tensor]:
        """Return (lower, upper) risk-controlling bands."""
        if self._multiplier is None:
            raise RuntimeError("Must call calibrate() before predict()")
        m = self._multiplier.to(predictions.device)
        return predictions - m, predictions + m


def coverage_score(
    targets: Tensor,
    lower: Tensor,
    upper: Tensor,
    alpha: float = 0.1,
) -> dict[str, float]:
    """Compute coverage metrics for prediction bands.

    Args:
        targets: True values, shape ``(N,)`` or ``(N, d)``.
        lower: Lower band bounds, same shape.
        upper: Upper band bounds, same shape.
        alpha: Target miscoverage rate (for reporting target coverage).

    Returns:
        Dict with ``empirical_coverage``, ``target_coverage``,
        ``mean_bandwidth``, ``bandwidth_efficiency``.
    """
    covered = (targets >= lower) & (targets <= upper)
    if covered.ndim > 1:
        covered = covered.all(dim=-1)
    emp_cov = covered.float().mean().item()

    bandwidth = (upper - lower).abs()
    if bandwidth.ndim > 1:
        mean_bw = bandwidth.sum(dim=-1).mean().item()
    else:
        mean_bw = bandwidth.mean().item()

    if mean_bw > 0:
        targets_range = (targets.max() - targets.min()).abs().item()
        bw_efficiency = targets_range / mean_bw if targets_range > 0 else 1.0
    else:
        bw_efficiency = 0.0

    return {
        "empirical_coverage": emp_cov,
        "target_coverage": 1.0 - alpha,
        "mean_bandwidth": mean_bw,
        "bandwidth_efficiency": bw_efficiency,
    }
