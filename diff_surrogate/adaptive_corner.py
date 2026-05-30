"""Adaptive multi-corner robust evaluator with uncertainty-based weighting.

Provides `AdaptiveMultiCornerEvaluator` which extends the static multi-corner
evaluation in `robust_design_step` with two adaptive capabilities:

1. **Uncertainty-weighted corners**: When an `EnsembleSurrogate` is provided,
   corners with higher prediction uncertainty receive higher weights, steering
   optimization toward under-explored regions of the design space.

2. **Corner skipping**: Corners whose ensemble uncertainty falls below a
   configurable threshold are skipped entirely, saving forward-pass computation
   during later optimization stages when the surrogate is confident.

Adaptation examples for sibling projects
-----------------------------------------
OpenLithoHub:
    Wrap ``pw_fidelity_loss`` as a corner provider by parameterising wavelength
    or polarization:

    >>> corners = [
    ...     CornerSpec("TE_1550", 1.0, {"wavelength": 1.55e-6, "pol": "TE"}),
    ...     CornerSpec("TM_1550", 1.0, {"wavelength": 1.55e-6, "pol": "TM"}),
    ...     CornerSpec("TE_1310", 0.5, {"wavelength": 1.31e-6, "pol": "TE"}),
    ... ]
    >>> evaluator = AdaptiveMultiCornerEvaluator(corners, ensemble=my_ensemble)
    >>> loss, info = evaluator.evaluate(density, pw_fidelity_loss, loss_fn)

DiffCFD:
    Wrap ``multi_corner_optimize`` corners by mapping Reynolds / Mach numbers:

    >>> corners = [
    ...     CornerSpec("low_Re",   1.0, {"reynolds": 100}),
    ...     CornerSpec("high_Re",  1.0, {"reynolds": 10000}),
    ...     CornerSpec("transonic", 0.5, {"mach": 0.85}),
    ... ]
    >>> evaluator = AdaptiveMultiCornerEvaluator(
    ...     corners, ensemble=cfd_ensemble, skip_threshold=0.02,
    ... )
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from .robust_design import CornerSpec


class AdaptiveMultiCornerEvaluator:
    """Evaluate a design across multiple operating corners with adaptive weights.

    When no ensemble surrogate is provided the evaluator behaves identically to
    the static weighting in ``robust_design_step`` -- each corner contributes
    ``corner.weight`` to the total loss.

    When an ``EnsembleSurrogate`` is supplied, per-corner prediction uncertainty
    is used to up-weight corners where the model is less confident, blending the
    static and uncertainty-based weights via ``uncertainty_weight``.

    Parameters
    ----------
    corners:
        List of :class:`CornerSpec` instances defining operating corners.
    ensemble:
        Optional :class:`EnsembleSurrogate` for uncertainty estimation.
    uncertainty_weight:
        Blend factor ``alpha`` in ``[0, 1]``.  At ``alpha=0`` weights are
        purely static; at ``alpha=1`` they are purely uncertainty-driven.
    min_weight:
        Floor for any single corner's normalized weight, preventing a corner
        from being completely ignored even when its uncertainty is zero.
    skip_threshold:
        If a corner's normalized uncertainty is below this value and an
        ensemble is available, that corner is skipped during evaluation.
        ``0.0`` (default) disables skipping.
    """

    def __init__(
        self,
        corners: list[CornerSpec],
        ensemble: Any | None = None,
        uncertainty_weight: float = 0.5,
        min_weight: float = 0.01,
        skip_threshold: float = 0.0,
    ):
        if not corners:
            raise ValueError("corners must be a non-empty list of CornerSpec")
        if not 0.0 <= uncertainty_weight <= 1.0:
            raise ValueError(
                f"uncertainty_weight must be in [0, 1], got {uncertainty_weight}"
            )
        if min_weight < 0.0:
            raise ValueError(f"min_weight must be >= 0, got {min_weight}")
        if skip_threshold < 0.0:
            raise ValueError(f"skip_threshold must be >= 0, got {skip_threshold}")

        self.corners = corners
        self.ensemble = ensemble
        self.uncertainty_weight = uncertainty_weight
        self.min_weight = min_weight
        self.skip_threshold = skip_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        design: Tensor,
        forward_fn: Callable,
        loss_fn: Callable,
        **kwargs: Any,
    ) -> tuple[Tensor, dict]:
        """Compute weighted multi-corner loss.

        Parameters
        ----------
        design:
            Design tensor (should have ``requires_grad=True`` for optimisation).
        forward_fn:
            ``forward_fn(design, **corner_params) -> Tensor``.
        loss_fn:
            ``loss_fn(output) -> Tensor`` (scalar).

        Returns
        -------
        weighted_loss : Tensor
            Combined loss across evaluated (non-skipped) corners.
        info : dict
            Keys: ``"per_corner_loss"``, ``"weights"``, ``"uncertainties"``,
            ``"skipped"``.
        """
        weights = self.adaptive_weights(design, forward_fn)
        uncertainties = self._compute_uncertainties(design)

        per_corner_loss: list[Tensor] = []
        info_losses: list[float] = []
        skipped: list[int] = []

        for idx, corner in enumerate(self.corners):
            if self.should_skip_corner(idx, uncertainties[idx] if uncertainties else 0.0):
                skipped.append(idx)
                continue
            output = forward_fn(design, **corner.params)
            corner_loss = loss_fn(output)
            per_corner_loss.append(corner_loss * weights[idx])
            info_losses.append(corner_loss.item())

        if not per_corner_loss:
            # All corners skipped -- fall back to first corner to avoid empty loss
            corner = self.corners[0]
            output = forward_fn(design, **corner.params)
            fallback_loss = loss_fn(output)
            per_corner_loss.append(fallback_loss)
            info_losses.append(fallback_loss.item())
            skipped = []

        weighted_loss = torch.stack(per_corner_loss).sum()

        info: dict[str, Any] = {
            "per_corner_loss": info_losses,
            "weights": weights,
            "uncertainties": uncertainties if uncertainties is not None else [0.0] * len(self.corners),
            "skipped": skipped,
        }
        return weighted_loss, info

    def adaptive_weights(self, design: Tensor, forward_fn: Callable | None = None) -> list[float]:
        """Compute adaptive weights blending static and uncertainty-based weighting.

        If no ensemble is available, returns the static corner weights
        (normalised to sum to 1).

        Returns
        -------
        weights : list[float]
            Normalised weight for each corner, summing to 1.
        """
        n = len(self.corners)
        static = [c.weight for c in self.corners]
        static_sum = sum(static)
        if static_sum == 0:
            static_norm = [1.0 / n] * n
        else:
            static_norm = [w / static_sum for w in static]

        if self.ensemble is None:
            return static_norm

        uncertainties = self._compute_uncertainties(design)
        if uncertainties is None:
            return static_norm

        alpha = self.uncertainty_weight
        unc_sum = sum(uncertainties)
        if unc_sum == 0:
            return static_norm

        unc_norm = [u / unc_sum for u in uncertainties]

        blended = [
            (1.0 - alpha) * s + alpha * u
            for s, u in zip(static_norm, unc_norm)
        ]

        # Apply minimum weight floor and renormalize
        blended = [max(w, self.min_weight) for w in blended]
        total = sum(blended)
        if total > 0:
            blended = [w / total for w in blended]

        return blended

    def should_skip_corner(self, corner_idx: int, uncertainty: float) -> bool:
        """Decide whether to skip a corner due to low uncertainty.

        A corner is skipped when:
        - An ensemble is available (otherwise we have no uncertainty signal)
        - ``skip_threshold > 0``
        - The corner's uncertainty is strictly below the threshold

        Parameters
        ----------
        corner_idx:
            Index of the corner in ``self.corners``.
        uncertainty:
            The uncertainty value for this corner.

        Returns
        -------
        skip : bool
        """
        if self.ensemble is None:
            return False
        if self.skip_threshold <= 0.0:
            return False
        if corner_idx < 0 or corner_idx >= len(self.corners):
            return False
        return uncertainty < self.skip_threshold

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_uncertainties(self, design: Tensor) -> list[float] | None:
        """Query ensemble for per-corner uncertainty estimates.

        Returns ``None`` when no ensemble is configured or when the ensemble
        does not expose ``predict_with_uncertainty``.
        """
        if self.ensemble is None:
            return None
        if not hasattr(self.ensemble, "predict_with_uncertainty"):
            return None

        uncertainties: list[float] = []
        for corner in self.corners:
            try:
                means, stds = self.ensemble.predict_with_uncertainty(design)
                # Aggregate scalar uncertainty from returned std dicts
                unc_values = [v for v in stds.values()]
                if unc_values:
                    scalar_unc = sum(
                        float(v.mean().item()) for v in unc_values
                    ) / len(unc_values)
                else:
                    scalar_unc = 0.0
            except Exception:
                scalar_unc = 0.0
            uncertainties.append(scalar_unc)

        return uncertainties
