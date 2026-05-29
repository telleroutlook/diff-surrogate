"""Hybrid Z-score convergence monitoring for optimization loops.

Borrowed from quant-stat-1's ``computeHybridZScore()`` (event-driven-utils.ts)
which blends a standard z-score with a robust MAD-based z-score to detect
convergence while being resistant to outliers.

The hybrid formulation:
    Z_hybrid = Z_standard * (1 - w) + Z_robust * w
where:
    Z_standard = (x - mean) / std
    Z_robust   = (x - median) / (1.4826 * MAD)

The weight ``w`` controls robustness: w=0 is pure standard z-score,
w=1 is pure robust z-score. Default w=0.5 balances sensitivity and
outlier resistance.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np


class ConvergenceAction(Enum):
    """Recommended action from the convergence monitor."""

    CONTINUE = "continue"
    EARLY_STOP = "early_stop"
    REDUCE_LR = "reduce_lr"


@dataclass
class ConvergenceConfig:
    """Configuration for the ConvergenceMonitor.

    Args:
        window: Number of recent loss values to consider.
        hybrid_weight: Blending weight w for Z_hybrid = Z_std*(1-w) + Z_robust*w.
        early_stop_threshold: Early-stop when |Z_hybrid| < this value.
        reduce_lr_threshold: Reduce learning rate when |Z_hybrid| < this value.
        min_steps: Minimum steps before any convergence action is taken.
        patience: Number of consecutive reduce-lr signals before early stop.
    """

    window: int = 20
    hybrid_weight: float = 0.5
    early_stop_threshold: float = 0.05
    reduce_lr_threshold: float = 0.2
    min_steps: int = 10
    patience: int = 5


def _median(values: list[float]) -> float:
    """Compute median of a list of floats."""
    if not values:
        return 0.0
    return float(np.median(values))


def _mean(values: list[float]) -> float:
    """Compute mean of a list of floats."""
    if not values:
        return 0.0
    return float(np.mean(values))


def _std(values: list[float]) -> float:
    """Compute sample standard deviation (Bessel-corrected, ddof=1) of a list of floats."""
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def _mad(values: list[float]) -> float:
    """Compute median absolute deviation of a list of floats."""
    if not values:
        return 0.0
    return float(np.median(np.abs(np.array(values) - np.median(values))))


def hybrid_z_score(values: Sequence[float], weight: float = 0.5) -> float:
    """Compute hybrid z-score blending standard and robust (MAD-based) z-scores.

    Borrowed from quant-stat-1's ``computeHybridZScore()``:
        Z_hybrid = Z_standard * (1 - w) + Z_robust * w

    Z_robust uses MAD (median absolute deviation) scaled by 1.4826 to be
    consistent with standard deviation for normally distributed data.

    Args:
        values: Sequence of recent scalar values (e.g. loss values).
            The last element is the current value; the rest are history.
        weight: Blending weight w in [0, 1]. 0 = pure standard, 1 = pure robust.

    Returns:
        Hybrid z-score scalar. Returns 0.0 if insufficient data.
    """
    # Early return for very small windows — O(n log n) from sorting is wasteful here.
    if len(values) < 2:
        return 0.0

    finite = [v for v in values if math.isfinite(v)]
    if len(finite) < 2:
        return 0.0

    # Constant sequence cannot produce a meaningful z-score.
    if min(values) == max(values):
        return 0.0

    current = values[-1]
    history = list(values[:-1])

    std_dev = _std(history)
    mean_val = _mean(history)
    z_standard = (current - mean_val) / std_dev if std_dev > 0 else 0.0

    mad_val = _mad(history)
    robust_scale = 1.4826 * mad_val
    med_val = _median(history)
    z_robust = (current - med_val) / robust_scale if robust_scale > 0 else z_standard

    return z_standard * (1.0 - weight) + z_robust * weight


class ConvergenceMonitor:
    """Tracks loss history and computes hybrid z-score for convergence decisions.

    Usage::

        monitor = ConvergenceMonitor()
        for step in range(n_steps):
            loss = compute_loss(...)
            action = monitor.update(loss.item(), step)
            if action == ConvergenceAction.EARLY_STOP:
                break
            elif action == ConvergenceAction.REDUCE_LR:
                for pg in optimizer.param_groups:
                    pg['lr'] *= 0.5
    """

    def __init__(self, config: ConvergenceConfig | None = None) -> None:
        self._config = config or ConvergenceConfig()
        self._history: list[float] = []
        self._reduce_lr_count: int = 0

    @property
    def config(self) -> ConvergenceConfig:
        """Read-only access to configuration."""
        return self._config

    @property
    def history(self) -> list[float]:
        """Read-only access to full loss history."""
        return list(self._history)

    def update(self, loss: float, step: int) -> ConvergenceAction:
        """Record a new loss value and return a recommended action.

        Args:
            loss: Scalar loss value from the current optimization step.
            step: Current step index (0-based).

        Returns:
            Recommended action: CONTINUE, REDUCE_LR, or EARLY_STOP.
        """
        self._history.append(loss)

        if step < self._config.min_steps:
            return ConvergenceAction.CONTINUE

        # Not enough data to judge convergence — must accumulate at least a full window.
        if len(self._history) < self._config.window:
            return ConvergenceAction.CONTINUE

        window = self._history[-self._config.window :]
        z = hybrid_z_score(window, weight=self._config.hybrid_weight)
        abs_z = abs(z)

        # Stagnant loss (std + mad == 0) — cannot distinguish true convergence from stagnation.
        window_std = _std(window)
        window_mad = _mad(window)
        if window_std + window_mad == 0:
            return ConvergenceAction.CONTINUE

        if abs_z < self._config.early_stop_threshold:
            self._reduce_lr_count = 0
            return ConvergenceAction.EARLY_STOP

        if abs_z < self._config.reduce_lr_threshold:
            self._reduce_lr_count += 1
            if self._reduce_lr_count >= self._config.patience:
                self._reduce_lr_count = 0
                return ConvergenceAction.EARLY_STOP
            return ConvergenceAction.REDUCE_LR

        # Loss is still changing meaningfully — reset patience counter
        self._reduce_lr_count = 0
        return ConvergenceAction.CONTINUE
