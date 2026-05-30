"""Multi-fidelity optimization alternating between a cheap surrogate and an
expensive truth model.

The loop spends most steps evaluating the fast ``surrogate_fn`` and periodically
calls the expensive ``truth_fn`` to correct the optimization trajectory. An
optional ``calibration_fn`` can adjust the surrogate based on truth evaluations.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor

from .convergence import ConvergenceAction, ConvergenceMonitor

logger = logging.getLogger(__name__)


@dataclass
class MultiFidelityConfig:
    """Configuration for multi-fidelity optimization.

    Args:
        correction_interval: Call ``truth_fn`` every this many steps.
        calibration_fn: Optional callable ``calibration_fn(surrogate_fn, design, truth_output)``
            that adjusts the surrogate after a truth evaluation.
        log_interval: Print status every this many steps (0 = silent).
    """
    correction_interval: int = 20
    calibration_fn: Callable | None = None
    log_interval: int = 25
    truth_mode: str = "differentiable"  # "differentiable" | "surrogate_grad" | "calibration_only"


@dataclass
class MultiFidelityResult:
    """Result of a multi-fidelity optimization run."""
    design: Tensor
    loss_history: list[float]
    fidelity_history: list[str]       # "surrogate" or "truth" per step
    truth_steps: list[int]            # indices where truth was evaluated
    converged: bool
    final_step: int


def optimize_multifidelity(
    design_init: Tensor,
    surrogate_fn: Callable,
    truth_fn: Callable,
    loss_fn: Callable,
    n_steps: int = 300,
    lr: float = 1e-3,
    config: MultiFidelityConfig | None = None,
    convergence_monitor: ConvergenceMonitor | None = None,
    grad_clip: float | None = None,
) -> MultiFidelityResult:
    """Multi-fidelity optimization alternating between surrogate and truth.

    Loop structure (per step):
        1. If ``step % correction_interval == 0``, evaluate with ``truth_fn``
           and optionally calibrate the surrogate via ``calibration_fn``.
           Otherwise evaluate with ``surrogate_fn``.
        2. Compute loss via ``loss_fn``.
        3. Backpropagate and update ``design`` with gradient descent.
        4. If ``convergence_monitor`` is provided, check for early stopping.
        5. Record loss and fidelity tag.

    Args:
        design_init: Initial design tensor.
        surrogate_fn: Fast forward callable ``surrogate_fn(design) -> Tensor``.
        truth_fn: Expensive forward callable ``truth_fn(design) -> Tensor``.
        loss_fn: Loss callable ``loss_fn(output) -> Tensor`` (scalar).
        n_steps: Maximum number of optimization steps.
        lr: Learning rate for gradient descent.
        config: Multi-fidelity configuration. Uses defaults if None.
        convergence_monitor: Optional convergence monitor for early stopping.

    Returns:
        MultiFidelityResult with final design, histories, and convergence status.
    """
    cfg = config or MultiFidelityConfig()

    design = design_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([design], lr=lr)

    loss_history: list[float] = []
    fidelity_history: list[str] = []
    truth_steps: list[int] = []
    converged = False
    final_step = n_steps - 1

    for step in range(n_steps):
        optimizer.zero_grad()

        use_truth = (step > 0 and step % cfg.correction_interval == 0)
        fidelity = "truth" if use_truth else "surrogate"
        fidelity_history.append(fidelity)

        if use_truth:
            truth_steps.append(step)

            if cfg.truth_mode == "surrogate_grad":
                # Straight-through estimator: use truth output detached, but pass
                # gradients through the surrogate so the optimizer still gets a signal.
                with torch.no_grad():
                    truth_output = truth_fn(design)
                surrogate_output = surrogate_fn(design)
                if cfg.calibration_fn is not None:
                    cfg.calibration_fn(surrogate_fn, design, truth_output)
                output = truth_output + (surrogate_output - surrogate_output.detach())
            elif cfg.truth_mode == "calibration_only":
                # Truth output is only used for calibration — loss uses surrogate.
                with torch.no_grad():
                    truth_output = truth_fn(design)
                if cfg.calibration_fn is not None:
                    cfg.calibration_fn(surrogate_fn, design, truth_output)
                output = surrogate_fn(design)
            else:
                # "differentiable": current behaviour, loss.backward() flows through truth_fn.
                output = truth_fn(design)
                if cfg.calibration_fn is not None:
                    cfg.calibration_fn(surrogate_fn, design, output)
        else:
            output = surrogate_fn(design)

        loss = loss_fn(output)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_([design], grad_clip)
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if cfg.log_interval > 0 and (step + 1) % cfg.log_interval == 0:
            logger.info("[step %4d] loss=%.6f  fidelity=%s", step + 1, loss_val, fidelity)

        if convergence_monitor is not None:
            action = convergence_monitor.update(loss_val, step)
            if action == ConvergenceAction.EARLY_STOP:
                converged = True
                final_step = step
                break
            if action == ConvergenceAction.REDUCE_LR:
                for pg in optimizer.param_groups:
                    pg["lr"] *= 0.5

    return MultiFidelityResult(
        design=design.detach(),
        loss_history=loss_history,
        fidelity_history=fidelity_history,
        truth_steps=truth_steps,
        converged=converged,
        final_step=final_step,
    )
