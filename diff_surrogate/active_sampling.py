"""Uncertainty-triggered active learning for multi-fidelity surrogate models.

Provides :class:`UncertaintyTriggeredSampler` that uses ensemble prediction
variance to concentrate new high-fidelity samples in regions where the
surrogate is least confident, and :class:`MultiFidelityActiveLearner` that
orchestrates the full active-learning loop.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
from torch import Tensor

from .conformal import SplitConformalPredictor
from .ensemble import EnsembleSurrogate

logger = logging.getLogger(__name__)


def _latin_hypercube_samples(
    n_samples: int,
    n_dims: int,
    bounds: Tensor,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Generate a Latin Hypercube sample within *bounds*.

    Args:
        n_samples: Number of points.
        n_dims: Dimensionality.
        bounds: ``(n_dims, 2)`` tensor of (min, max) per dimension.
        generator: Optional RNG for reproducibility.
    """
    cuts = torch.arange(n_samples, device=bounds.device, dtype=bounds.dtype) + 0.5
    cuts = cuts / n_samples
    samples = []
    for d in range(n_dims):
        perm = torch.randperm(n_samples, generator=generator, device=bounds.device)
        jitter = torch.rand(n_samples, generator=generator, device=bounds.device) * 0.5 / n_samples
        coord = cuts[perm] + jitter
        low, high = bounds[d, 0], bounds[d, 1]
        samples.append(low + coord * (high - low))
    return torch.stack(samples, dim=-1)


class UncertaintyTriggeredSampler:
    """Active sampler that uses ensemble uncertainty to guide sampling.

    When ensemble prediction variance exceeds a threshold, sample new
    high-fidelity data points in the high-uncertainty region.
    """

    def __init__(
        self,
        ensemble: EnsembleSurrogate,
        input_bounds: Tensor,
        n_candidates: int = 1000,
        uncertainty_threshold: float = 0.1,
        exploration_fraction: float = 0.1,
        generator: torch.Generator | None = None,
    ):
        self.ensemble = ensemble
        self.input_bounds = input_bounds
        self.n_candidates = n_candidates
        self.uncertainty_threshold = uncertainty_threshold
        self.exploration_fraction = exploration_fraction
        self.generator = generator
        self._device = ensemble.device
        self._n_dims = input_bounds.shape[0]
        self._conformal_predictor: SplitConformalPredictor | None = None

    def compute_uncertainty(self, x: Tensor) -> Tensor:
        """Compute prediction variance over ensemble members.

        Returns a 1-D tensor of length *N* (one uncertainty value per input
        row), averaged over output properties when the ensemble produces dict
        output.
        """
        self.ensemble.eval()
        with torch.no_grad():
            member_outputs: list[Tensor | dict[str, Tensor]] = [
                m(x.to(self._device)) for m in self.ensemble._members
            ]
        if isinstance(member_outputs[0], dict):
            all_vars: list[Tensor] = []
            keys = [k for k in member_outputs[0] if not k.endswith("_std")]
            for key in keys:
                stacked = torch.stack(
                    [out[key] for out in member_outputs],  # type: ignore[index]
                    dim=0,
                )
                var = stacked.var(dim=0)  # shape: (N,) or (N, ...)
                if var.ndim > 1:
                    var = var.mean(dim=-1)
                all_vars.append(var)
            return torch.stack(all_vars, dim=0).mean(dim=0)
        stacked = torch.stack(member_outputs, dim=0)  # type: ignore[arg-type]
        var = stacked.var(dim=0)
        if var.ndim > 1:
            var = var.mean(dim=-1)
        return var

    def calibrate_uncertainty(
        self,
        cal_inputs: Tensor,
        cal_targets: Tensor | dict[str, Tensor],
        alpha: float = 0.1,
    ) -> None:
        """Calibrate conformal bands using a held-out calibration set.

        After calibration, :meth:`suggest_samples` uses calibrated bandwidth
        instead of raw ensemble variance to rank candidate locations.

        Args:
            cal_inputs: Calibration inputs ``(N, d)``.
            cal_targets: Calibration targets (tensor or dict).
            alpha: Target miscoverage rate.
        """
        self.ensemble.eval()
        with torch.no_grad():
            pred = self.ensemble.predict(cal_inputs.to(self._device))

        if isinstance(cal_targets, dict):
            first_key = next(iter(cal_targets))
            targets = cal_targets[first_key].to(self._device)
        else:
            targets = cal_targets.to(self._device)

        main_key = next(k for k in pred if not k.endswith("_std"))
        predictions = pred[main_key]

        if predictions.ndim > 1 and predictions.shape[-1] > 1:
            if targets.ndim == 1:
                targets = targets.unsqueeze(-1)
        elif predictions.ndim == 1 and targets.ndim == 2:
            predictions = predictions.unsqueeze(-1)

        self._conformal_predictor = SplitConformalPredictor()
        self._conformal_predictor.calibrate(predictions, targets, alpha=alpha)

    def suggest_samples(self, n_samples: int = 10) -> Tensor:
        """Suggest new sample locations in high-uncertainty regions.

        When conformal calibration has been performed (via
        :meth:`calibrate_uncertainty`), uses calibrated bandwidth to rank
        candidates.  Otherwise falls back to raw ensemble variance.

        Strategy:
            1. Generate a large candidate pool (Latin hypercube).
            2. Evaluate uncertainty at each candidate.
            3. Select top-k by uncertainty (greedy acquisition).
            4. Add *exploration_fraction* random samples for diversity.
        """
        n_greedy = max(1, int(n_samples * (1.0 - self.exploration_fraction)))
        n_explore = n_samples - n_greedy

        candidates = _latin_hypercube_samples(
            self.n_candidates,
            self._n_dims,
            self.input_bounds.to(self._device),
            generator=self.generator,
        )

        if self._conformal_predictor is not None:
            uncertainty = self._calibrated_bandwidth(candidates)
        else:
            uncertainty = self.compute_uncertainty(candidates)

        _, top_idx = uncertainty.topk(min(n_greedy, self.n_candidates))
        selected = candidates[top_idx]

        if n_explore > 0:
            explore_pts = _latin_hypercube_samples(
                n_explore,
                self._n_dims,
                self.input_bounds.to(self._device),
                generator=self.generator,
            )
            selected = torch.cat([selected, explore_pts], dim=0)

        return selected

    def _calibrated_bandwidth(self, x: Tensor) -> Tensor:
        """Compute calibrated prediction bandwidth at each input point."""
        self.ensemble.eval()
        with torch.no_grad():
            pred = self.ensemble.predict(x.to(self._device))
        main_key = next(k for k in pred if not k.endswith("_std"))
        predictions = pred[main_key]
        lower, upper = self._conformal_predictor.predict(predictions)
        bw = (upper - lower).abs()
        if bw.ndim > 1:
            bw = bw.mean(dim=-1)
        return bw

    def step(
        self,
        high_fidelity_fn: Callable[[Tensor], Tensor],
        train_inputs: Tensor,
        train_targets: Tensor | dict[str, Tensor],
        n_samples: int = 10,
        n_epochs: int = 20,
        lr: float = 1e-3,
    ) -> dict:
        """One active learning step: suggest -> evaluate -> augment training.

        Args:
            high_fidelity_fn: Ground-truth callable.
            train_inputs: Existing training inputs ``(N, d)``.
            train_targets: Existing training targets.
            n_samples: Number of new samples to acquire.
            n_epochs: Epochs for re-training after augmentation.
            lr: Learning rate for re-training.

        Returns:
            Dict with ``n_new_samples``, ``max_uncertainty``,
            ``mean_uncertainty``, ``train_inputs``, ``train_targets``.
        """
        new_x = self.suggest_samples(n_samples)
        with torch.no_grad():
            new_y = high_fidelity_fn(new_x.to(self._device))

        augmented_inputs = torch.cat([train_inputs, new_x.cpu()], dim=0)

        if isinstance(train_targets, dict):
            if isinstance(new_y, dict):
                augmented_targets = {
                    k: torch.cat([train_targets[k], new_y[k].cpu()], dim=0) for k in train_targets
                }
            else:
                first_key = next(iter(train_targets))
                augmented_targets = {
                    k: torch.cat([train_targets[k], new_y.cpu()], dim=0)
                    if k == first_key
                    else torch.cat(
                        [train_targets[k], torch.zeros(new_y.shape[0])],
                        dim=0,
                    )
                    for k in train_targets
                }
        else:
            augmented_targets = torch.cat(
                [train_targets, new_y.cpu() if isinstance(new_y, Tensor) else new_y],
                dim=0,
            )

        for member in self.ensemble._members:
            member._data_generator = (  # type: ignore[assignment]
                lambda _n, _i=augmented_inputs, _t=augmented_targets: (_i, _t)
            )
        self.ensemble.train_surrogate(
            n_samples=augmented_inputs.shape[0],
            n_epochs=n_epochs,
            lr=lr,
        )

        uncertainty_after = self.compute_uncertainty(
            _latin_hypercube_samples(
                500,
                self._n_dims,
                self.input_bounds.to(self._device),
                generator=self.generator,
            ),
        )

        return {
            "n_new_samples": n_samples,
            "max_uncertainty": uncertainty_after.max().item(),
            "mean_uncertainty": uncertainty_after.mean().item(),
            "train_inputs": augmented_inputs,
            "train_targets": augmented_targets,
        }


class MultiFidelityActiveLearner:
    """Coordinates multi-fidelity surrogate with active sampling.

    Uses low-fidelity model everywhere, high-fidelity model only where
    uncertainty is high, reducing total high-fidelity evaluations.
    """

    def __init__(
        self,
        low_fidelity_fn: Callable[[Tensor], Tensor],
        high_fidelity_fn: Callable[[Tensor], Tensor],
        ensemble: EnsembleSurrogate,
        input_bounds: Tensor,
        n_candidates: int = 1000,
        uncertainty_threshold: float = 0.1,
        exploration_fraction: float = 0.1,
        generator: torch.Generator | None = None,
    ):
        self.low_fidelity_fn = low_fidelity_fn
        self.high_fidelity_fn = high_fidelity_fn
        self.ensemble = ensemble
        self.input_bounds = input_bounds
        self.sampler = UncertaintyTriggeredSampler(
            ensemble=ensemble,
            input_bounds=input_bounds,
            n_candidates=n_candidates,
            uncertainty_threshold=uncertainty_threshold,
            exploration_fraction=exploration_fraction,
            generator=generator,
        )

    def fit_active(
        self,
        initial_inputs: Tensor,
        initial_targets: Tensor | dict[str, Tensor],
        n_iterations: int = 10,
        budget_per_iter: int = 5,
        n_epochs_per_iter: int = 20,
        lr: float = 1e-3,
    ) -> dict:
        """Run active learning loop.

        Each iteration:
            1. Train ensemble on current data.
            2. Compute uncertainty on candidate pool.
            3. Select top-k high-uncertainty points.
            4. Evaluate with high-fidelity model.
            5. Add to training set.

        Args:
            initial_inputs: Seed training inputs ``(N, d)``.
            initial_targets: Seed training targets.
            n_iterations: Number of active-learning iterations.
            budget_per_iter: High-fidelity evaluations per iteration.
            n_epochs_per_iter: Training epochs after each augmentation.
            lr: Learning rate.

        Returns:
            Dict with ``total_hf_evals``, ``final_uncertainty``,
            ``convergence_history``, ``train_inputs``, ``train_targets``.
        """
        train_inputs = initial_inputs.clone()
        if isinstance(initial_targets, dict):
            train_targets = {k: v.clone() for k, v in initial_targets.items()}
        else:
            train_targets = initial_targets.clone()

        convergence_history: list[float] = []
        total_hf_evals = 0

        for iteration in range(n_iterations):
            for member in self.ensemble._members:
                member._data_generator = (  # type: ignore[assignment]
                    lambda _n, _i=train_inputs, _t=train_targets: (_i, _t)
                )
            self.ensemble.train_surrogate(
                n_samples=train_inputs.shape[0],
                n_epochs=n_epochs_per_iter,
                lr=lr,
            )

            result = self.sampler.step(
                high_fidelity_fn=self.high_fidelity_fn,
                train_inputs=train_inputs,
                train_targets=train_targets,
                n_samples=budget_per_iter,
                n_epochs=n_epochs_per_iter,
                lr=lr,
            )

            train_inputs = result["train_inputs"]
            train_targets = result["train_targets"]
            convergence_history.append(result["mean_uncertainty"])
            total_hf_evals += result["n_new_samples"]

            logger.info(
                "[active iter %d/%d] uncertainty=%.6f  hf_evals=%d",
                iteration + 1,
                n_iterations,
                result["mean_uncertainty"],
                total_hf_evals,
            )

        return {
            "total_hf_evals": total_hf_evals,
            "final_uncertainty": convergence_history[-1] if convergence_history else float("inf"),
            "convergence_history": convergence_history,
            "train_inputs": train_inputs,
            "train_targets": train_targets,
        }
