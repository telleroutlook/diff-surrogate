"""Cost-aware Bayesian experimental design with multi-fidelity routing.

Chains multi-fidelity cost models with conformal/generative ensemble
bandwidth into an active sampling loop that maximises information gain
per unit cost.

References:
    - Cost-aware multi-fidelity Bayesian optimisation, arXiv:2003.02645
    - Conformal prediction bands for uncertainty-driven sampling, arXiv:2107.07511
    - Batch Bayesian optimisation via determinantal point processes, arXiv:1706.02045
"""

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import torch
from torch import Tensor

from .active_sampling import _latin_hypercube_samples
from .conformal import SplitConformalPredictor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CostModel
# ---------------------------------------------------------------------------


@dataclass
class CostModel:
    """Tracks fidelity levels, per-evaluation cost, and budget consumption."""

    fidelity_levels: dict[str, float]
    total_budget: float
    budget_consumed: float = 0.0

    def __post_init__(self) -> None:
        if not self.fidelity_levels:
            raise ValueError("fidelity_levels must be non-empty")
        if self.total_budget <= 0:
            raise ValueError("total_budget must be positive")
        for name, cost in self.fidelity_levels.items():
            if cost <= 0:
                raise ValueError(f"Cost for fidelity '{name}' must be positive, got {cost}")

    def can_afford(self, fidelity: str, n: int) -> bool:
        if fidelity not in self.fidelity_levels:
            return False
        return self.fidelity_levels[fidelity] * n <= self.remaining()

    def consume(self, fidelity: str, n: int) -> None:
        cost = self.fidelity_levels[fidelity] * n
        if cost > self.remaining():
            raise ValueError(
                f"Cannot afford {n}x {fidelity}: need {cost:.2f}, have {self.remaining():.2f}"
            )
        self.budget_consumed += cost

    def remaining(self) -> float:
        return self.total_budget - self.budget_consumed


# ---------------------------------------------------------------------------
# AcquisitionFunction
# ---------------------------------------------------------------------------


class AcquisitionFunction(Enum):
    UNCERTAINTY = "uncertainty"
    BAYESIAN = "bayesian"
    DIVERSITY = "diversity"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# ExperimentDesignResult
# ---------------------------------------------------------------------------


@dataclass
class ExperimentDesignResult:
    """Result of a full experiment-design loop run."""

    total_hf_evals: int
    total_cost: float
    convergence_history: list[float]
    fidelity_history: list[str]
    final_uncertainty: float
    points_selected: Tensor
    improvement_vs_random: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pairwise_distances(a: Tensor, b: Tensor) -> Tensor:
    """Squared Euclidean distance matrix between rows of *a* and *b*."""
    return (a.unsqueeze(1) - b.unsqueeze(0)).pow(2).sum(-1)


def _max_min_distance_index(candidates: Tensor, existing: Tensor) -> int:
    """Index of the candidate farthest from all existing points."""
    if existing.shape[0] == 0:
        return 0
    dists = _pairwise_distances(candidates, existing)
    min_per_candidate = dists.min(dim=1).values
    return int(min_per_candidate.argmax().item())


# ---------------------------------------------------------------------------
# ExperimentDesignLoop
# ---------------------------------------------------------------------------


class ExperimentDesignLoop:
    """Cost-aware Bayesian experiment design orchestrator.

    Decides *where* to sample next via an acquisition function and *at which
    fidelity* via the cost model, then runs the full suggest-evaluate-augment
    loop.
    """

    def __init__(
        self,
        cost_model: CostModel,
        acquisition_fn: AcquisitionFunction = AcquisitionFunction.HYBRID,
        input_bounds: Tensor | None = None,
        exploration_fraction: float = 0.1,
        n_candidates: int = 1000,
        seed: int = 0,
        hybrid_weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
    ) -> None:
        self.cost_model = cost_model
        self.acquisition_fn = acquisition_fn
        self.input_bounds = input_bounds
        self.exploration_fraction = exploration_fraction
        self.n_candidates = n_candidates
        self.seed = seed
        self.hybrid_weights = hybrid_weights

        self._sorted_fidelities = sorted(
            cost_model.fidelity_levels.items(), key=lambda kv: kv[1]
        )
        self._conformal: SplitConformalPredictor | None = None

    # ------------------------------------------------------------------
    # Public: suggest next batch
    # ------------------------------------------------------------------

    def suggest_next(
        self,
        ensemble_predictions: Tensor,
        ensemble_bandwidths: Tensor,
        existing_points: Tensor,
        n_samples: int = 5,
    ) -> tuple[Tensor, str, str]:
        """Suggest next sample locations, fidelity level, and human-readable reason.

        Args:
            ensemble_predictions: Mean predictions at candidate locations ``(M, ...)``.
            ensemble_bandwidths: Bandwidth (uncertainty) at each candidate ``(M,)``.
            existing_points: Points already evaluated ``(K, d)``.
            n_samples: How many new points to select.

        Returns:
            (selected_points, fidelity_level, reason_string).
        """
        if self.input_bounds is None:
            raise ValueError("input_bounds is required for suggest_next")

        n_dims = self.input_bounds.shape[0]
        device = self.input_bounds.device

        generator = torch.Generator(device=device).manual_seed(self.seed)
        self.seed += 1

        candidates = _latin_hypercube_samples(
            self.n_candidates, n_dims, self.input_bounds, generator=generator
        )

        n_greedy = max(1, int(n_samples * (1.0 - self.exploration_fraction)))
        n_explore = n_samples - n_greedy

        if self.acquisition_fn == AcquisitionFunction.UNCERTAINTY:
            indices = self._acquire_uncertainty(candidates, ensemble_bandwidths, n_greedy)
            reason = "uncertainty"
        elif self.acquisition_fn == AcquisitionFunction.BAYESIAN:
            indices = self._acquire_bayesian(
                candidates, ensemble_predictions, ensemble_bandwidths, n_greedy
            )
            reason = "bayesian_eig"
        elif self.acquisition_fn == AcquisitionFunction.DIVERSITY:
            indices = self._acquire_diversity(candidates, existing_points, n_greedy)
            reason = "diversity"
        else:
            indices = self._acquire_hybrid(
                candidates, ensemble_predictions, ensemble_bandwidths, existing_points, n_greedy
            )
            reason = "hybrid"

        selected = candidates[indices]

        if n_explore > 0:
            explore_pts = _latin_hypercube_samples(
                n_explore, n_dims, self.input_bounds, generator=generator
            )
            selected = torch.cat([selected, explore_pts], dim=0)

        fidelity = self._choose_fidelity(n_samples)
        reason += f"|fidelity={fidelity}"

        return selected, fidelity, reason

    # ------------------------------------------------------------------
    # Public: full loop
    # ------------------------------------------------------------------

    def run_loop(
        self,
        surrogate_fn: Callable[[Tensor], Tensor],
        truth_fn: Callable[[Tensor], Tensor],
        initial_inputs: Tensor,
        initial_targets: Tensor,
        n_iterations: int = 10,
        samples_per_iter: int = 5,
        retrain_fn: Callable[[Tensor, Tensor], None] | None = None,
        calibrate_fn: Callable[[Tensor, Tensor], None] | None = None,
    ) -> ExperimentDesignResult:
        """Full suggest -> evaluate -> augment -> retrain loop.

        Args:
            surrogate_fn: Fast surrogate ``(Tensor) -> Tensor``.
            truth_fn: Expensive ground-truth ``(Tensor) -> Tensor``.
            initial_inputs: Seed inputs ``(N, d)``.
            initial_targets: Seed targets ``(N,)`` or ``(N, p)``.
            n_iterations: Number of design iterations.
            samples_per_iter: Samples acquired per iteration.
            retrain_fn: Optional ``(inputs, targets) -> None`` to retrain surrogate.
            calibrate_fn: Optional ``(inputs, targets) -> None`` for conformal calibration.

        Returns:
            ExperimentDesignResult with full history.
        """
        if self.input_bounds is None:
            raise ValueError("input_bounds is required for run_loop")

        all_points: list[Tensor] = [initial_inputs.clone()]
        train_x = initial_inputs.clone()
        train_y = initial_targets.clone()

        convergence_history: list[float] = []
        fidelity_history: list[str] = []
        total_hf_evals = 0

        sorted_fids = [name for name, _ in self._sorted_fidelities]
        cheapest_fid = sorted_fids[0]
        most_expensive_fid = sorted_fids[-1]

        for iteration in range(n_iterations):
            with torch.no_grad():
                pred = surrogate_fn(train_x)
                if isinstance(pred, dict):
                    main_key = next(k for k in pred if not k.endswith("_std"))
                    pred_tensor = pred[main_key]
                else:
                    pred_tensor = pred

            if calibrate_fn is not None:
                calibrate_fn(train_x, train_y)

            candidate_gen = torch.Generator(device=self.input_bounds.device).manual_seed(
                self.seed + iteration
            )
            candidates = _latin_hypercube_samples(
                self.n_candidates,
                self.input_bounds.shape[0],
                self.input_bounds,
                generator=candidate_gen,
            )

            with torch.no_grad():
                cand_pred = surrogate_fn(candidates)
                if isinstance(cand_pred, dict):
                    main_key = next(k for k in cand_pred if not k.endswith("_std"))
                    cand_mean = cand_pred[main_key]
                    std_key = main_key + "_std"
                    cand_std = cand_pred.get(std_key, torch.zeros_like(cand_mean))
                else:
                    cand_mean = cand_pred
                    cand_std = torch.zeros_like(cand_mean)

            bandwidths = cand_std * 2.0
            if bandwidths.ndim > 1:
                bandwidths = bandwidths.mean(dim=-1)
            if isinstance(bandwidths, Tensor) and bandwidths.numel() == 0:
                bandwidths = torch.ones(candidates.shape[0], device=candidates.device)

            if cand_mean.ndim > 1:
                cand_mean_flat = cand_mean.mean(dim=-1)
            else:
                cand_mean_flat = cand_mean

            n_greedy = max(1, int(samples_per_iter * (1.0 - self.exploration_fraction)))

            if self.acquisition_fn == AcquisitionFunction.UNCERTAINTY:
                indices = self._acquire_uncertainty(candidates, bandwidths, n_greedy)
            elif self.acquisition_fn == AcquisitionFunction.BAYESIAN:
                indices = self._acquire_bayesian(candidates, cand_mean_flat, bandwidths, n_greedy)
            elif self.acquisition_fn == AcquisitionFunction.DIVERSITY:
                indices = self._acquire_diversity(candidates, train_x, n_greedy)
            else:
                indices = self._acquire_hybrid(
                    candidates, cand_mean_flat, bandwidths, train_x, n_greedy
                )

            selected = candidates[indices]

            n_explore = samples_per_iter - n_greedy
            if n_explore > 0:
                explore_gen = torch.Generator(device=self.input_bounds.device).manual_seed(
                    self.seed + iteration + 10000
                )
                explore_pts = _latin_hypercube_samples(
                    n_explore,
                    self.input_bounds.shape[0],
                    self.input_bounds,
                    generator=explore_gen,
                )
                selected = torch.cat([selected, explore_pts], dim=0)

            fidelity = self._choose_fidelity(samples_per_iter)
            fidelity_history.append(fidelity)

            if fidelity == most_expensive_fid:
                with torch.no_grad():
                    new_targets = truth_fn(selected)
                total_hf_evals += samples_per_iter
            else:
                with torch.no_grad():
                    new_targets = surrogate_fn(selected)
                    if isinstance(new_targets, dict):
                        main_key = next(k for k in new_targets if not k.endswith("_std"))
                        new_targets = new_targets[main_key]

            if self.cost_model.can_afford(fidelity, samples_per_iter):
                self.cost_model.consume(fidelity, samples_per_iter)

            train_x = torch.cat([train_x, selected], dim=0)
            if new_targets.ndim == 1 and train_y.ndim == 2:
                new_targets = new_targets.unsqueeze(-1)
            elif new_targets.ndim == 2 and train_y.ndim == 1:
                train_y = train_y.unsqueeze(-1)
            train_y = torch.cat([train_y, new_targets], dim=0)
            all_points.append(selected)

            if retrain_fn is not None:
                retrain_fn(train_x, train_y)

            max_bw = bandwidths.max().item() if bandwidths.numel() > 0 else float("inf")
            convergence_history.append(max_bw)

            logger.info(
                "[design iter %d/%d] max_bw=%.6f  fidelity=%s  budget_remaining=%.2f",
                iteration + 1,
                n_iterations,
                max_bw,
                fidelity,
                self.cost_model.remaining(),
            )

        all_points_tensor = torch.cat(all_points, dim=0)
        final_uncertainty = convergence_history[-1] if convergence_history else float("inf")

        random_improvement = self._compute_improvement_vs_random(
            surrogate_fn, truth_fn, initial_inputs, initial_targets,
            all_points_tensor.shape[0] - initial_inputs.shape[0],
            n_iterations, samples_per_iter,
        )

        return ExperimentDesignResult(
            total_hf_evals=total_hf_evals,
            total_cost=self.cost_model.budget_consumed,
            convergence_history=convergence_history,
            fidelity_history=fidelity_history,
            final_uncertainty=final_uncertainty,
            points_selected=all_points_tensor,
            improvement_vs_random=random_improvement,
        )

    # ------------------------------------------------------------------
    # Acquisition helpers
    # ------------------------------------------------------------------

    def _acquire_uncertainty(
        self, candidates: Tensor, bandwidths: Tensor, n: int
    ) -> Tensor:
        bw = bandwidths
        if bw.shape[0] != candidates.shape[0]:
            if bw.numel() >= candidates.shape[0]:
                bw = bw[:candidates.shape[0]]
            else:
                bw = bw.repeat_interleave(candidates.shape[0] // bw.numel() + 1)[
                    :candidates.shape[0]
                ]
        k = min(n, candidates.shape[0])
        _, indices = bw.topk(k)
        return indices

    def _acquire_bayesian(
        self,
        candidates: Tensor,
        ensemble_predictions: Tensor,
        bandwidths: Tensor,
        n: int,
    ) -> Tensor:
        pred = ensemble_predictions
        bw = bandwidths

        if pred.shape[0] != candidates.shape[0]:
            if pred.numel() >= candidates.shape[0]:
                pred = pred[:candidates.shape[0]]
            else:
                pred = pred.repeat_interleave(candidates.shape[0] // pred.numel() + 1)[
                    :candidates.shape[0]
                ]
        if bw.shape[0] != candidates.shape[0]:
            if bw.numel() >= candidates.shape[0]:
                bw = bw[:candidates.shape[0]]
            else:
                bw = bw.repeat_interleave(candidates.shape[0] // bw.numel() + 1)[
                    :candidates.shape[0]
                ]

        mean_abs = pred.abs().clamp(min=1e-8)
        eig = bw * (1.0 + torch.log1p(mean_abs))
        k = min(n, candidates.shape[0])
        _, indices = eig.topk(k)
        return indices

    def _acquire_diversity(
        self, candidates: Tensor, existing_points: Tensor, n: int
    ) -> Tensor:
        k = min(n, candidates.shape[0])
        if existing_points.shape[0] == 0:
            gen = torch.Generator(device=candidates.device).manual_seed(self.seed)
            return torch.randperm(candidates.shape[0], generator=gen)[:k]

        selected_indices: list[int] = []
        remaining = torch.arange(candidates.shape[0], device=candidates.device)

        for _ in range(k):
            current_pool = candidates[remaining]
            anchor = torch.cat(
                [existing_points, candidates[torch.tensor(selected_indices, device=candidates.device)]],
                dim=0,
            ) if selected_indices else existing_points

            dists = _pairwise_distances(current_pool, anchor)
            min_dist = dists.min(dim=1).values
            best_local = int(min_dist.argmax().item())
            selected_indices.append(int(remaining[best_local].item()))
            mask = torch.ones(remaining.shape[0], dtype=torch.bool, device=candidates.device)
            mask[best_local] = False
            remaining = remaining[mask]

        return torch.tensor(selected_indices, device=candidates.device)

    def _acquire_hybrid(
        self,
        candidates: Tensor,
        ensemble_predictions: Tensor,
        bandwidths: Tensor,
        existing_points: Tensor,
        n: int,
    ) -> Tensor:
        w_u, w_b, w_d = self.hybrid_weights
        M = candidates.shape[0]

        bw = bandwidths
        if bw.shape[0] != M:
            bw = bw.repeat_interleave(M // bw.shape[0] + 1)[:M]

        unc_scores = bw / (bw.max() + 1e-12)

        pred = ensemble_predictions
        if pred.shape[0] != M:
            pred = pred.repeat_interleave(M // pred.shape[0] + 1)[:M]
        mean_abs = pred.abs().clamp(min=1e-8)
        eig = bw * (1.0 + torch.log1p(mean_abs))
        bay_scores = eig / (eig.max() + 1e-12)

        if existing_points.shape[0] > 0:
            dists = _pairwise_distances(candidates, existing_points)
            min_dist = dists.min(dim=1).values
            div_scores = min_dist / (min_dist.max() + 1e-12)
        else:
            div_scores = torch.ones(M, device=candidates.device)

        combined = w_u * unc_scores + w_b * bay_scores + w_d * div_scores
        k = min(n, M)
        _, indices = combined.topk(k)
        return indices

    # ------------------------------------------------------------------
    # Fidelity routing
    # ------------------------------------------------------------------

    def _choose_fidelity(self, n_samples: int) -> str:
        for name, _ in reversed(self._sorted_fidelities):
            if self.cost_model.can_afford(name, n_samples):
                return name
        return self._sorted_fidelities[0][0]

    # ------------------------------------------------------------------
    # Baseline comparison
    # ------------------------------------------------------------------

    def _compute_improvement_vs_random(
        self,
        surrogate_fn: Callable[[Tensor], Tensor],
        truth_fn: Callable[[Tensor], Tensor],
        initial_inputs: Tensor,
        initial_targets: Tensor,
        n_new: int,
        n_iterations: int,
        samples_per_iter: int,
    ) -> float:
        if self.input_bounds is None or n_new <= 0:
            return 0.0

        rng = torch.Generator(device=self.input_bounds.device).manual_seed(self.seed + 9999)
        n_dims = self.input_bounds.shape[0]
        random_x = _latin_hypercube_samples(n_new, n_dims, self.input_bounds, generator=rng)

        with torch.no_grad():
            random_y = truth_fn(random_x)

        rand_train_x = torch.cat([initial_inputs, random_x], dim=0)
        if random_y.ndim == 1 and initial_targets.ndim == 2:
            random_y = random_y.unsqueeze(-1)
        elif random_y.ndim == 2 and initial_targets.ndim == 1:
            init_t = initial_targets.unsqueeze(-1)
        else:
            init_t = initial_targets
        rand_train_y = torch.cat([init_t, random_y], dim=0)

        eval_x = _latin_hypercube_samples(
            200, n_dims, self.input_bounds,
            generator=torch.Generator(device=self.input_bounds.device).manual_seed(42),
        )
        with torch.no_grad():
            eval_y = truth_fn(eval_x)

        with torch.no_grad():
            rand_pred = surrogate_fn(eval_x)
            if isinstance(rand_pred, dict):
                main_key = next(k for k in rand_pred if not k.endswith("_std"))
                rand_pred = rand_pred[main_key]

        random_mse = torch.mean((rand_pred - eval_y) ** 2).item()

        with torch.no_grad():
            active_pred = surrogate_fn(eval_x)
            if isinstance(active_pred, dict):
                main_key = next(k for k in active_pred if not k.endswith("_std"))
                active_pred = active_pred[main_key]

        active_mse = torch.mean((active_pred - eval_y) ** 2).item()

        if random_mse < 1e-12:
            return 0.0
        return (random_mse - active_mse) / random_mse * 100.0


# ---------------------------------------------------------------------------
# DesignBenchmark
# ---------------------------------------------------------------------------


class DesignBenchmark:
    """Compare active experiment design vs random and uniform-grid baselines."""

    def __init__(
        self,
        cost_model: CostModel,
        input_bounds: Tensor,
        acquisition_fn: AcquisitionFunction = AcquisitionFunction.HYBRID,
        exploration_fraction: float = 0.1,
        n_candidates: int = 500,
    ) -> None:
        self.cost_model = cost_model
        self.input_bounds = input_bounds
        self.acquisition_fn = acquisition_fn
        self.exploration_fraction = exploration_fraction
        self.n_candidates = n_candidates

    def run(
        self,
        surrogate_fn: Callable[[Tensor], Tensor],
        truth_fn: Callable[[Tensor], Tensor],
        initial_inputs: Tensor,
        initial_targets: Tensor,
        n_seeds: int = 3,
        n_iterations: int = 10,
        samples_per_iter: int = 5,
    ) -> dict:
        n_dims = self.input_bounds.shape[0]
        active_results: list[ExperimentDesignResult] = []
        random_bw_histories: list[list[float]] = []
        uniform_bw_histories: list[list[float]] = []

        for seed in range(n_seeds):
            cm = CostModel(
                fidelity_levels=dict(self.cost_model.fidelity_levels),
                total_budget=self.cost_model.total_budget,
            )
            loop = ExperimentDesignLoop(
                cost_model=cm,
                acquisition_fn=self.acquisition_fn,
                input_bounds=self.input_bounds.clone(),
                exploration_fraction=self.exploration_fraction,
                n_candidates=self.n_candidates,
                seed=seed * 100,
            )
            result = loop.run_loop(
                surrogate_fn=surrogate_fn,
                truth_fn=truth_fn,
                initial_inputs=initial_inputs.clone(),
                initial_targets=initial_targets.clone(),
                n_iterations=n_iterations,
                samples_per_iter=samples_per_iter,
            )
            active_results.append(result)

            rng = torch.Generator(device=self.input_bounds.device).manual_seed(seed * 77)
            total_new = n_iterations * samples_per_iter
            rand_x = _latin_hypercube_samples(
                total_new, n_dims, self.input_bounds, generator=rng
            )
            with torch.no_grad():
                rand_y = truth_fn(rand_x)

            rand_bw_history = self._compute_random_bandwidth_history(
                surrogate_fn, truth_fn, initial_inputs, initial_targets,
                rand_x, rand_y, n_iterations, samples_per_iter,
            )
            random_bw_histories.append(rand_bw_history)

            n_per_dim = max(2, int(total_new ** (1.0 / max(n_dims, 1))))
            grids = [
                torch.linspace(self.input_bounds[d, 0], self.input_bounds[d, 1], n_per_dim)
                for d in range(n_dims)
            ]
            mesh = torch.meshgrid(*grids, indexing="ij")
            uniform_x = torch.stack([m.reshape(-1) for m in mesh], dim=-1)
            if uniform_x.shape[0] > total_new:
                idx = torch.randperm(uniform_x.shape[0], generator=rng)[:total_new]
                uniform_x = uniform_x[idx]

            with torch.no_grad():
                uniform_y = truth_fn(uniform_x)

            uniform_bw_history = self._compute_random_bandwidth_history(
                surrogate_fn, truth_fn, initial_inputs, initial_targets,
                uniform_x, uniform_y, n_iterations, samples_per_iter,
            )
            uniform_bw_histories.append(uniform_bw_history)

        active_final_bws = [r.final_uncertainty for r in active_results]
        active_total_costs = [r.total_cost for r in active_results]
        active_improvements = [r.improvement_vs_random for r in active_results]

        def _avg_final(histories: list[list[float]]) -> float:
            finals = [h[-1] for h in histories if h]
            return sum(finals) / len(finals) if finals else float("inf")

        return {
            "active_final_bandwidth": sum(active_final_bws) / len(active_final_bws),
            "active_total_cost": sum(active_total_costs) / len(active_total_costs),
            "active_improvement_vs_random_pct": sum(active_improvements) / len(active_improvements),
            "random_final_bandwidth": _avg_final(random_bw_histories),
            "uniform_final_bandwidth": _avg_final(uniform_bw_histories),
            "active_convergence_histories": [r.convergence_history for r in active_results],
            "random_convergence_histories": random_bw_histories,
            "uniform_convergence_histories": uniform_bw_histories,
            "n_seeds": n_seeds,
        }

    @staticmethod
    def _compute_random_bandwidth_history(
        surrogate_fn: Callable[[Tensor], Tensor],
        truth_fn: Callable[[Tensor], Tensor],
        initial_inputs: Tensor,
        initial_targets: Tensor,
        new_points: Tensor,
        new_targets: Tensor,
        n_iterations: int,
        samples_per_iter: int,
    ) -> list[float]:
        n_dims = initial_inputs.shape[1]
        bounds_device = initial_inputs.device
        history: list[float] = []

        n_new = new_points.shape[0]
        for i in range(min(n_iterations, n_new // max(samples_per_iter, 1))):
            end = min((i + 1) * samples_per_iter, n_new)
            batch_x = new_points[: end]
            batch_y = new_targets[: end]

            combined_x = torch.cat([initial_inputs, batch_x], dim=0)
            if batch_y.ndim == 1 and initial_targets.ndim == 2:
                batch_y = batch_y.unsqueeze(-1)
            elif batch_y.ndim == 2 and initial_targets.ndim == 1:
                init_t = initial_targets.unsqueeze(-1)
            else:
                init_t = initial_targets
            combined_y = torch.cat([init_t, batch_y], dim=0)

            eval_x = _latin_hypercube_samples(
                200, n_dims,
                torch.stack([combined_x.min(dim=0).values, combined_x.max(dim=0).values], dim=-1).to(bounds_device),
                generator=torch.Generator(device=bounds_device).manual_seed(i * 13),
            )
            with torch.no_grad():
                pred = surrogate_fn(eval_x)
                if isinstance(pred, dict):
                    main_key = next(k for k in pred if not k.endswith("_std"))
                    mean_val = pred[main_key]
                    std_key = main_key + "_std"
                    std_val = pred.get(std_key, torch.zeros_like(mean_val))
                else:
                    std_val = torch.zeros_like(pred)

            bw = std_val * 2.0
            if bw.ndim > 1:
                bw = bw.mean(dim=-1)
            history.append(bw.max().item() if bw.numel() > 0 else float("inf"))

        return history
