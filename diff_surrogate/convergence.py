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
from collections import deque
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
        cooldown_steps: Steps to wait after a REDUCE_LR before acting again.
    """

    window: int = 20
    hybrid_weight: float = 0.5
    early_stop_threshold: float = 0.05
    reduce_lr_threshold: float = 0.2
    min_steps: int = 10
    patience: int = 5
    cooldown_steps: int = 3

    def __post_init__(self):
        if self.window < 2:
            raise ValueError(f"window must be >= 2, got {self.window}")


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
    if len(values) < 2:
        return 0.0

    finite = [v for v in values if math.isfinite(v)]
    if len(finite) < 2:
        return 0.0

    # Constant sequence cannot produce a meaningful z-score.
    if min(finite) == max(finite):
        return 0.0

    current = finite[-1]
    arr = np.asarray(finite[:-1], dtype=np.float64)
    if len(arr) < 2:
        return 0.0

    mean_val = float(np.mean(arr))
    std_dev = float(np.std(arr, ddof=1))
    if not math.isfinite(std_dev) or std_dev <= 0:
        z_standard = 0.0
    else:
        z_standard = (current - mean_val) / std_dev

    med = float(np.median(arr))
    mad_val = float(np.median(np.abs(arr - med)))
    robust_scale = 1.4826 * mad_val
    z_robust = (current - med) / robust_scale if robust_scale > 0 else z_standard

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
        self._history: deque[float] = deque(maxlen=self._config.window * 2)
        self._reduce_lr_count: int = 0
        self._cooldown_remaining: int = 0

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

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return ConvergenceAction.CONTINUE

        if step < self._config.min_steps:
            return ConvergenceAction.CONTINUE

        # Not enough data to judge convergence — must accumulate at least a full window.
        if len(self._history) < self._config.window:
            return ConvergenceAction.CONTINUE

        window = list(self._history)[-self._config.window :]
        z = hybrid_z_score(window, weight=self._config.hybrid_weight)
        abs_z = abs(z)

        # Stagnant loss (std + mad == 0) — loss is constant, stop early.
        arr = np.asarray(window, dtype=np.float64)
        window_std = float(np.std(arr, ddof=1)) if len(arr) >= 2 else 0.0
        window_mad = float(np.median(np.abs(arr - np.median(arr))))
        if window_std + window_mad == 0:
            return ConvergenceAction.EARLY_STOP

        if abs_z < self._config.early_stop_threshold:
            self._reduce_lr_count = 0
            return ConvergenceAction.EARLY_STOP

        if abs_z < self._config.reduce_lr_threshold:
            self._reduce_lr_count += 1
            if self._reduce_lr_count >= self._config.patience:
                self._reduce_lr_count = 0
                return ConvergenceAction.EARLY_STOP
            self._cooldown_remaining = self._config.cooldown_steps
            return ConvergenceAction.REDUCE_LR

        # Loss is still changing meaningfully — reset patience counter
        self._reduce_lr_count = 0
        return ConvergenceAction.CONTINUE
