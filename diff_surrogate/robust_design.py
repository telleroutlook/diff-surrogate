"""Composable robust optimization step with designable masks, antithetic sampling,
and multi-corner evaluation.

Three composable techniques (from R2) combined in a single step function:
1. DesignableMask -- only compute gradients for designable pixels
2. Antithetic sampling -- paired perturbations for variance reduction
3. Multi-corner evaluation -- weighted losses at multiple operating points
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from .convergence import ConvergenceAction, ConvergenceMonitor


@dataclass
class AntitheticConfig:
    """Configuration for antithetic variate sampling.

    Args:
        n_pairs: Number of (delta, -delta) perturbation pairs to evaluate.
        perturbation_fn: Optional custom perturbation generator. Must accept
            (design, n_pairs) and return a Tensor of shape (n_pairs, *design.shape).
            If None, isotropic Gaussian noise with std=0.01 is used.
    """

    n_pairs: int = 4
    perturbation_fn: Callable | None = None

    def __post_init__(self):
        if self.n_pairs < 1:
            raise ValueError(f"n_pairs must be >= 1, got {self.n_pairs}")


@dataclass
class CornerSpec:
    """A single operating corner for multi-corner evaluation.

    Args:
        label: Human-readable name for this corner.
        weight: Relative weight for this corner's loss in the combined objective.
        params: Keyword arguments forwarded to ``forward_fn`` for this corner.
    """

    label: str
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)


def _default_perturbation(
    design: Tensor,
    n_pairs: int,
    std: float = 0.01,
    relative: bool = False,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Generate isotropic Gaussian perturbation pairs.

    Returns:
        Tensor of shape (n_pairs, *design.shape) where row i and row i+1
        are antithetic (negated) pairs when n_pairs is even. For odd n_pairs
        the last row is unpaired.
    """
    eff_std = std * (design.detach().abs().mean() + 1e-12) if relative else std
    half = n_pairs // 2
    if half == 0:
        return (
            torch.randn(
                1, *design.shape, device=design.device, dtype=design.dtype, generator=generator
            )
            * eff_std
        )
    deltas = (
        torch.randn(
            half, *design.shape, device=design.device, dtype=design.dtype, generator=generator
        )
        * eff_std
    )
    paired = torch.stack([deltas, -deltas], dim=1)
    paired = paired.reshape(2 * half, *design.shape)
    if n_pairs % 2 == 1:
        extra = (
            torch.randn(
                1, *design.shape, device=design.device, dtype=design.dtype, generator=generator
            )
            * eff_std
        )
        paired = torch.cat([paired, extra], dim=0)
    return paired[:n_pairs]


def robust_design_step(
    design: Tensor,
    forward_fn: Callable,
    loss_fn: Callable,
    *,
    designable_mask: Tensor | None = None,
    antithetic_config: AntitheticConfig | None = None,
    corners: Sequence[CornerSpec] | None = None,
    convergence_monitor: ConvergenceMonitor | None = None,
    step: int = 0,
    batched: bool = False,
) -> tuple[Tensor, ConvergenceAction]:
    """Compute robust loss with optional mask, antithetic sampling, and multi-corner evaluation.

    The evaluation proceeds in composable layers:

    1. **Designable mask**: If ``designable_mask`` is provided, frozen pixels (mask=False)
       are detached from the computation graph, so gradients only flow through designable pixels.

    2. **Multi-corner**: If ``corners`` is provided, evaluate ``forward_fn`` at each
       corner's parameters and compute a weighted sum of per-corner losses.

    3. **Antithetic sampling**: If ``antithetic_config`` is provided, also evaluate
       at ``n_pairs`` perturbed designs (paired +/- deltas) and average their losses
       with the nominal loss.

    4. **Convergence**: If ``convergence_monitor`` is provided, ``monitor.update()``
       is called with the scalar loss value.

    Args:
        design: Current design tensor (must have ``requires_grad=True`` for gradient use).
        forward_fn: Callable ``forward_fn(design, **corner_params) -> Tensor``.
        loss_fn: Callable ``loss_fn(output) -> Tensor`` (scalar).
        designable_mask: Boolean tensor matching design shape. True = designable.
        antithetic_config: If set, enables antithetic variate sampling.
        corners: If set, enables multi-corner weighted evaluation.
        convergence_monitor: If set, records loss for convergence tracking.
        step: Current step index (passed to convergence monitor).
        batched: If True, stack all perturbed designs into one batch for a single
            forward pass.  Falls back to sequential when corners are present.

    Returns:
        Tuple of (combined loss tensor with grad graph attached, convergence action).
    """
    if batched and corners:
        warnings.warn(
            "batched=True is ignored when corners are provided; "
            "falling back to sequential evaluation.",
            UserWarning,
            stacklevel=2,
        )

    # --- 1. Designable mask: detach frozen pixels, no hooks needed ---
    if designable_mask is not None:
        mask = designable_mask.to(design.device)
        eff_design = torch.where(mask, design, design.detach())
    else:
        eff_design = design

    # --- 2. Compute nominal losses (multi-corner or single) ---
    if corners:
        corner_losses: list[Tensor] = []
        for corner in corners:
            output = forward_fn(eff_design, **corner.params)
            corner_losses.append(loss_fn(output) * corner.weight)
        nominal_loss = torch.stack(corner_losses).sum()
    else:
        output = forward_fn(eff_design)
        nominal_loss = loss_fn(output)

    # --- 3. Antithetic sampling ---
    if antithetic_config is not None:
        gen_fn = antithetic_config.perturbation_fn or _default_perturbation
        deltas = gen_fn(eff_design, antithetic_config.n_pairs)

        # Batched path: single forward call with stacked perturbed designs (no corners).
        if batched and not corners:
            perturbed_batch = eff_design.unsqueeze(0) + deltas  # (N, *design.shape)
            n_pairs = perturbed_batch.shape[0]
            flat_batch = perturbed_batch.reshape(n_pairs, *eff_design.shape)
            flat_output = forward_fn(flat_batch)
            batch_losses = torch.stack(
                [loss_fn(flat_output[i]) for i in range(n_pairs)]
            )
            avg_antithetic = batch_losses.mean()
        else:
            antithetic_losses: list[Tensor] = []
            for i in range(antithetic_config.n_pairs):
                perturbed = eff_design + deltas[i]
                if corners:
                    pert_corner_losses: list[Tensor] = []
                    for corner in corners:
                        out = forward_fn(perturbed, **corner.params)
                        pert_corner_losses.append(loss_fn(out) * corner.weight)
                    antithetic_losses.append(torch.stack(pert_corner_losses).sum())
                else:
                    out = forward_fn(perturbed)
                    antithetic_losses.append(loss_fn(out))

            avg_antithetic = torch.stack(antithetic_losses).mean()

        total_loss = (nominal_loss + avg_antithetic) / 2.0
    else:
        total_loss = nominal_loss

    # --- 4. Convergence monitoring ---
    action = ConvergenceAction.CONTINUE
    if convergence_monitor is not None:
        action = convergence_monitor.update(total_loss.item(), step)

    return total_loss, action
