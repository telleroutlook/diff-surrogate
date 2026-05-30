"""Adaptive robust optimization with axial sampling and uncertainty-driven weighting.

Canonical implementation promoted from DiffNano.  Provides:

- **Axial sampling**: O(2N+1) corner samples instead of exhaustive O(3^N)
- **Correlated perturbation**: Cholesky-based multi-axis perturbation sampling
- **Fabricable subspace projection**: differentiable projection to discrete levels
- **AdaptiveRobustOptimizer**: combines axial sampling, curriculum, worst-case
  refinement, and uncertainty-driven multi-corner weighting via
  :class:`AdaptiveMultiCornerEvaluator` / :class:`EnsembleSurrogate`.

When an :class:`EnsembleSurrogate` is available, the optimizer queries its
prediction uncertainty to weight corners adaptively (up-weighting
under-explored regions).  Without an ensemble it falls back to static
worst-case weighting from axial samples.

References
----------
- Ma et al. (2024), BOSON-1: arXiv:2411.08210 (adaptive sampling, fabricable subspace)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from .adaptive_corner import AdaptiveMultiCornerEvaluator
from .robust_design import CornerSpec

__all__ = [
    "AdaptiveRobustOptimizer",
    "FabricableSubspaceProjection",
    "axial_samples",
    "correlated_perturbation",
]


def axial_samples(
    n_dims: int,
    sigma: float | torch.Tensor,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Generate 2N+1 axial samples for N variation sources.

    Returns the nominal (origin) plus 2N axial points at +/-sigma along
    each axis.

    Parameters
    ----------
    n_dims:
        Number of variation dimensions (N).
    sigma:
        Perturbation magnitude per axis.
    device, dtype:
        Tensor construction options.

    Returns
    -------
    samples : Tensor, shape ``(2N+1, N)``
    """
    nominal = torch.zeros(1, n_dims, device=device, dtype=dtype)
    axial = []
    for i in range(n_dims):
        pos = torch.zeros(1, n_dims, device=device, dtype=dtype)
        neg = torch.zeros(1, n_dims, device=device, dtype=dtype)
        pos[0, i] = sigma
        neg[0, i] = -sigma
        axial.append(pos)
        axial.append(neg)
    return torch.cat([nominal] + axial, dim=0)


def correlated_perturbation(
    params: torch.Tensor,
    cov_cholesky: torch.Tensor,
    n_samples: int = 8,
) -> torch.Tensor:
    """Sample correlated multi-axis perturbations via Cholesky decomposition.

    Generates ``delta = L @ epsilon`` where ``epsilon ~ N(0, I)`` and ``L``
    is the lower-triangular Cholesky factor of the covariance matrix.

    Parameters
    ----------
    params:
        Design parameters (used for shape/device/dtype inference).
    cov_cholesky:
        Lower-triangular Cholesky factor, shape ``(N, N)``.
    n_samples:
        Number of perturbation samples.

    Returns
    -------
    deltas : Tensor, shape ``(n_samples, N)``
    """
    N = cov_cholesky.shape[0]
    eps = torch.randn(n_samples, N, device=params.device, dtype=params.dtype)
    return eps @ cov_cholesky.T


class FabricableSubspaceProjection:
    """Project continuous density fields to nearest discrete fabricable geometry.

    Uses Gumbel-softmax relaxation for differentiable projection to discrete
    height levels, followed by minimum critical-dimension enforcement via
    morphological opening (erosion + dilation).

    Parameters
    ----------
    n_levels:
        Number of discretized height levels.
    min_cd_pixels:
        Minimum critical dimension in pixels.
    temperature:
        Gumbel-softmax temperature (lower = harder projection).
    """

    def __init__(
        self,
        n_levels: int = 4,
        min_cd_pixels: int = 2,
        temperature: float = 1.0,
    ):
        self.n_levels = n_levels
        self.min_cd_pixels = min_cd_pixels
        self.temperature = temperature

    def project(self, density: torch.Tensor) -> torch.Tensor:
        """Project density to fabricable subspace (differentiable).

        Parameters
        ----------
        density:
            Continuous density field in ``[0, 1]``, shape ``(H, W)``.

        Returns
        -------
        projected:
            Projected density (approximately discrete), same shape.
        """
        levels = torch.linspace(0, 1, self.n_levels, device=density.device, dtype=density.dtype)

        distances = torch.abs(density.unsqueeze(-1) - levels.unsqueeze(0).unsqueeze(0))
        weights = torch.softmax(-distances / max(self.temperature, 0.01), dim=-1)
        projected = (weights * levels).sum(dim=-1)

        if self.min_cd_pixels > 0:
            projected = _morphological_opening(projected, self.min_cd_pixels)

        return projected

    def projection_loss(self, density: torch.Tensor) -> torch.Tensor:
        """Penalty encouraging density to stay near discrete levels."""
        projected = self.project(density)
        return ((density - projected) ** 2).mean()


def _morphological_opening(density: torch.Tensor, radius: int) -> torch.Tensor:
    """Differentiable approximation of morphological opening (erosion + dilation)."""
    if radius <= 0:
        return density

    kernel_size = 2 * radius + 1

    padded = torch.nn.functional.pad(
        density.unsqueeze(0).unsqueeze(0),
        [radius] * 4,
        mode="constant",
        value=1.0,
    )

    eroded = -torch.nn.functional.max_pool2d(
        -padded,
        kernel_size,
        stride=1,
        padding=0,
    )

    eroded_padded = torch.nn.functional.pad(
        eroded,
        [radius] * 4,
        mode="constant",
        value=0.0,
    )

    dilated = torch.nn.functional.max_pool2d(
        eroded_padded,
        kernel_size,
        stride=1,
        padding=0,
    )

    return dilated.squeeze(0).squeeze(0)


class AdaptiveRobustOptimizer:
    """Adaptive robust optimizer with axial sampling, curriculum, and uncertainty weighting.

    Combines O(2N+1) axial sampling with adaptive worst-case refinement and
    progressive random sampling for capturing interaction effects.

    When configured with an ensemble surrogate (via *corners* + *ensemble*),
    per-corner prediction uncertainty drives the corner weights adaptively.
    Without an ensemble the optimizer uses static worst-case weighting from
    axial samples.

    Parameters
    ----------
    n_variation_dims:
        Number of variation sources (N).
    sigma:
        Perturbation magnitude (standard deviation per axis).
    cov_matrix:
        Covariance matrix for correlated perturbations, shape ``(N, N)``.
        Identity scaled by *sigma* if ``None``.
    n_random_budget:
        Additional random samples to add during curriculum phase.
    refinement_top_k:
        Number of worst-case samples to refine around.
    device:
        Torch device.
    corners:
        Optional list of :class:`CornerSpec` for multi-corner evaluation.
        When provided, :meth:`compute_robust_loss_with_corners` is available.
    ensemble:
        Optional :class:`EnsembleSurrogate` for uncertainty-driven corner
        weighting.  Only used when *corners* is also provided.
    uncertainty_weight:
        Blend factor ``alpha`` in ``[0, 1]`` for uncertainty-based corner
        weighting.  ``0`` = purely static weights, ``1`` = purely
        uncertainty-driven.
    """

    def __init__(
        self,
        n_variation_dims: int = 3,
        sigma: float = 5.0,
        cov_matrix: torch.Tensor | None = None,
        n_random_budget: int = 16,
        refinement_top_k: int = 3,
        device: str | torch.device = "cpu",
        corners: list[CornerSpec] | None = None,
        ensemble: Any | None = None,
        uncertainty_weight: float = 0.5,
    ):
        self.n_dims = n_variation_dims
        self.sigma = sigma
        self.n_random_budget = n_random_budget
        self.refinement_top_k = refinement_top_k
        self._device = torch.device(device)

        # Covariance setup
        if cov_matrix is not None:
            self.cov_chol = torch.linalg.cholesky(cov_matrix)
        else:
            self.cov_chol = (
                torch.eye(
                    n_variation_dims,
                    device=self._device,
                    dtype=torch.float64,
                )
                * sigma
            )

        # Corner evaluator: build internally when corners are provided
        self._corners = corners
        self._ensemble = ensemble
        self._uncertainty_weight = uncertainty_weight
        self._corner_evaluator: AdaptiveMultiCornerEvaluator | None = None

        if corners is not None:
            self._corner_evaluator = AdaptiveMultiCornerEvaluator(
                corners=corners,
                ensemble=ensemble,
                uncertainty_weight=uncertainty_weight,
            )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def corner_evaluator(self) -> AdaptiveMultiCornerEvaluator | None:
        """Access the internal corner evaluator (if configured)."""
        return self._corner_evaluator

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_robust_loss(
        self,
        params: torch.Tensor,
        forward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        perturbation_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        curriculum_frac: float = 0.0,
    ) -> torch.Tensor:
        """Compute adaptive robust loss estimate.

        Uses axial sampling (2N+1 points) plus optional curriculum-based
        random samples.  Loss is a weighted combination of the uniform mean
        and the worst-case top-k mean.

        Parameters
        ----------
        params:
            Design parameters.
        forward_fn:
            ``forward_fn(perturbed_params, perturbation_delta) -> loss``.
        perturbation_fn:
            ``perturbation_fn(params, delta) -> perturbed_params``.
        curriculum_frac:
            Fraction of random samples to add (0 = axial only, 1 = full).

        Returns
        -------
        robust_loss : Tensor, scalar
        """
        # Phase 1: axial samples (2N+1)
        axial = axial_samples(
            self.n_dims,
            self.sigma,
            device=self._device,
            dtype=params.dtype,
        )

        # Phase 2: curriculum-based random samples
        n_random = int(self.n_random_budget * curriculum_frac)
        if n_random > 0:
            eps = torch.randn(n_random, self.n_dims, device=self._device, dtype=params.dtype)
            random_samples = eps @ self.cov_chol.T
            all_samples = torch.cat([axial, random_samples], dim=0)
        else:
            all_samples = axial

        # Evaluate loss at every sample point
        losses = []
        for i in range(all_samples.shape[0]):
            delta = all_samples[i]
            perturbed = perturbation_fn(params, delta)
            losses.append(forward_fn(perturbed, delta))

        loss_stack = torch.stack(losses)

        # Worst-case refinement: weight top-k highest losses more heavily
        with torch.no_grad():
            sorted_indices = torch.argsort(loss_stack, descending=True)
            top_k = min(self.refinement_top_k, loss_stack.shape[0])

        uniform_loss = loss_stack.mean()
        worst_loss = loss_stack[sorted_indices[:top_k]].mean()

        return 0.7 * uniform_loss + 0.3 * worst_loss

    def compute_robust_loss_with_corners(
        self,
        params: torch.Tensor,
        forward_fn: Callable[[torch.Tensor], torch.Tensor],
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, dict]:
        """Compute robust loss using uncertainty-weighted multi-corner evaluation.

        When a corner evaluator is configured (i.e., *corners* was provided
        at construction), delegates to :class:`AdaptiveMultiCornerEvaluator`
        which blends static corner weights with ensemble-uncertainty-derived
        weights.

        Without a corner evaluator, falls back to single-corner evaluation.

        Parameters
        ----------
        params:
            Design parameters.
        forward_fn:
            ``forward_fn(design) -> Tensor``.
        loss_fn:
            ``loss_fn(output) -> Tensor`` (scalar).

        Returns
        -------
        loss : Tensor
        info : dict
        """
        if self._corner_evaluator is not None:
            return self._corner_evaluator.evaluate(params, forward_fn, loss_fn)

        # No corners configured: simple single-point evaluation
        output = forward_fn(params)
        return loss_fn(output), {
            "per_corner_loss": [loss_fn(output).item()],
            "weights": [1.0],
            "uncertainties": [0.0],
            "skipped": [],
        }

    # ------------------------------------------------------------------
    # Full optimization loop
    # ------------------------------------------------------------------

    def optimize(
        self,
        params: torch.Tensor,
        forward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        perturbation_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        n_steps: int = 200,
        lr: float = 0.01,
        verbose: bool = True,
    ) -> tuple[torch.Tensor, list[float]]:
        """Run adaptive robust optimization.

        Parameters
        ----------
        params:
            Initial design parameters.
        forward_fn:
            Loss callable ``(perturbed_params, delta) -> loss``.
        perturbation_fn:
            ``(params, delta) -> perturbed_params``.
        n_steps:
            Number of optimization steps.
        lr:
            Learning rate for Adam.
        verbose:
            Print progress every 20 steps.

        Returns
        -------
        params : Tensor
            Optimized parameters (detached).
        loss_history : list of float
        """
        params = params.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([params], lr=lr)
        loss_history: list[float] = []

        for step in range(n_steps):
            curriculum_frac = min(1.0, step / n_steps)

            loss = self.compute_robust_loss(
                params,
                forward_fn,
                perturbation_fn,
                curriculum_frac=curriculum_frac,
            )

            opt.zero_grad()
            loss.backward()

            if params.grad is not None and torch.isnan(params.grad).any():
                if verbose:
                    print(f"Step {step}: NaN gradient, stopping.")
                break

            opt.step()
            loss_history.append(loss.item())

            if verbose and step % 20 == 0:
                print(f"Step {step:4d}: loss={loss.item():.6f}")

        return params.detach(), loss_history
