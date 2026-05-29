"""Ensemble surrogate with uncertainty quantification."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from .base import AdaptiveCorrectionPolicy, CorrectionAction, CorrectionPolicy, SurrogateBase


class EnsembleSurrogate(SurrogateBase):
    """Ensemble of K surrogates providing mean + uncertainty predictions.

    Wraps K independent surrogate instances. Predictions return both
    mean and standard deviation across ensemble members.
    """

    def __init__(
        self,
        base_factory: Callable,
        n_members: int = 5,
        correction_policy: CorrectionPolicy | None = None,
        device: str = "cpu",
        seed: int | None = None,
    ):
        super().__init__(correction_policy=correction_policy, device=device)
        self.base_factory = base_factory
        self.n_members = n_members
        self.seed = seed
        if n_members < 1:
            raise ValueError(f"n_members must be >= 1, got {n_members}")
        members = []
        for i in range(n_members):
            if seed is not None:
                torch.manual_seed(seed + i)
            member = base_factory()
            # Force network build while seed is active so init is deterministic
            member.get_network()
            members.append(member)
        self._members = nn.ModuleList(members)
        self.to(self.device)

    def _build_network(self) -> nn.ModuleList:
        # Network is tracked via _members (nn.ModuleList), so this
        # returns the same objects. Kept for SurrogateBase API compat.
        return self._members

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        member_outputs = [m(x) for m in self._members]
        if isinstance(member_outputs[0], dict):
            keys = member_outputs[0].keys()
            result = {}
            for key in keys:
                stacked = torch.stack([out[key] for out in member_outputs], dim=0)
                result[key] = stacked.mean(dim=0)
                result[f"{key}_std"] = stacked.std(dim=0)
            return result
        else:
            stacked = torch.stack(member_outputs, dim=0)
            mean = stacked.mean(dim=0)
            std = stacked.std(dim=0)
            return {"output": mean, "output_std": std}

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.stats.total_predictions += 1
        was_training = self.training
        self.eval()
        with torch.no_grad():
            result = self(x.to(self.device))
        if was_training:
            self.train()
        if isinstance(self.correction_policy, AdaptiveCorrectionPolicy):
            std_vals = [v for k, v in result.items() if k.endswith("_std")]
            if std_vals:
                mean_std = torch.stack([s.mean() for s in std_vals]).mean().item()
                self.correction_policy.update_uncertainty(mean_std)
        return result

    def train_members(
        self,
        n_samples: int = 256,
        n_epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 32,
        bootstrap: bool = False,
    ) -> list[list[float]]:
        """Train each ensemble member on shared data for diversity via initialization.

        Args:
            bootstrap: If True, each member gets a bootstrap resample of the data.
                If False (default), all members train on the same shared dataset.
        """
        shared_inputs, shared_targets = self.generate_training_data(n_samples)
        all_losses: list[list[float]] = []
        for m in self._members:
            member: SurrogateBase = m  # type: ignore[assignment]
            inputs: torch.Tensor
            targets: torch.Tensor | dict[str, torch.Tensor]
            if bootstrap:
                n = shared_inputs.shape[0]
                idx = torch.randint(0, n, (n,), device=shared_inputs.device)
                inputs = shared_inputs[idx]
                if isinstance(shared_targets, dict):
                    targets = {k: v[idx] for k, v in shared_targets.items()}
                else:
                    targets = shared_targets[idx]
            else:
                inputs = shared_inputs
                targets = shared_targets
            # Override data generator to use shared data, then train
            original_gen = member._data_generator  # type: ignore[attr-defined]
            member._data_generator = lambda _n, _i=inputs, _t=targets: (_i, _t)  # type: ignore[assignment,operator]
            losses = member.train_surrogate(n_samples=n_samples, n_epochs=n_epochs, lr=lr)  # type: ignore[union-attr]
            member._data_generator = original_gen  # type: ignore[assignment]
            all_losses.append(losses)
        self._trained = True
        return all_losses

    def train_surrogate(
        self,
        n_samples: int = 256,
        n_epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 32,
        loss_weights: dict[str, float] | None = None,
        **kwargs: bool,
    ) -> list[float]:
        """Train ensemble members. Returns flattened loss list for base-class compat."""
        all_losses = self.train_members(
            n_samples=n_samples,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            **kwargs,
        )
        return [loss for member_losses in all_losses for loss in member_losses]

    def predict_with_uncertainty(
        self, x: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Return (mean_predictions, uncertainty_estimates)."""
        out = self.predict(x)
        means = {}
        stds = {}
        for key in list(out.keys()):
            if key.endswith("_std"):
                prop = key[:-4]
                stds[prop] = out[key]
            else:
                means[key] = out[key]
        return means, stds

    def generate_training_data(
        self, n_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor]]:
        return self._members[0].generate_training_data(n_samples)  # type: ignore[operator]

    def predict_with_correction(
        self,
        x: torch.Tensor,
        true_solver_fn: Callable | None = None,
        step: int | None = None,
    ) -> tuple[dict[str, torch.Tensor], CorrectionAction]:
        """Predict with correction, preserving uncertainty keys for ensemble output.

        When the correction policy triggers, the true solver output is compared against
        the ensemble mean (excluding _std keys). The returned dict still contains
        the ensemble uncertainty estimates alongside the true values.
        """
        if step is not None:
            self._step = step
        else:
            self._step += 1
        action = CorrectionAction.CONTINUE
        if true_solver_fn is not None and self.correction_policy.should_correct(self._step):
            action = CorrectionAction.CORRECT
            was_training = self.training
            self.eval()
            with torch.no_grad():
                true_output = true_solver_fn(x)
                surrogate_output = self(x.to(self.device))
            # Compute error against mean predictions only (skip _std keys)
            if isinstance(surrogate_output, dict) and isinstance(true_output, dict):
                error = sum(
                    torch.mean((true_output[k].to(v.device) - v) ** 2).item()  # type: ignore[union-attr]
                    for k, v in surrogate_output.items()
                    if not k.endswith("_std") and k in true_output
                ) / max(
                    1,
                    sum(1 for k in surrogate_output if not k.endswith("_std") and k in true_output),
                )
            elif isinstance(surrogate_output, dict) and isinstance(true_output, torch.Tensor):
                main_key = "output" if "output" in surrogate_output else next(
                    k for k in surrogate_output if not k.endswith("_std")
                )
                error = torch.mean(
                    (true_output.to(surrogate_output[main_key].device)
                     - surrogate_output[main_key]) ** 2
                ).item()
            elif not isinstance(surrogate_output, dict):
                error = torch.mean(
                    (true_output.to(surrogate_output.device) - surrogate_output) ** 2
                ).item()
            else:
                error = 0.0
            self.stats.correction_errors.append(error)
            if isinstance(self.correction_policy, AdaptiveCorrectionPolicy):
                self.correction_policy.update_error(error)
                std_vals = [
                    v
                    for k, v in surrogate_output.items()
                    if isinstance(v, torch.Tensor) and k.endswith("_std")
                ]
                if std_vals:
                    mean_std = torch.stack([s.mean() for s in std_vals]).mean().item()
                    self.correction_policy.update_uncertainty(mean_std)
            if was_training:
                self.train()
            self.stats.total_corrections += 1
            # Merge true output with uncertainty from ensemble
            if isinstance(surrogate_output, dict):
                merged = dict(surrogate_output)
                if isinstance(true_output, dict):
                    for k, v in true_output.items():
                        merged[k] = v.to(self.device)
                else:
                    if "output" in merged:
                        merged["output"] = true_output.to(self.device)
                return merged, action
            return surrogate_output, action

        result = self.predict(x)
        # predict() already calls update_uncertainty, but if it didn't trigger
        # (non-adaptive policy), we still want to update for adaptive ones.
        return result, action

    def get_members(self) -> nn.ModuleList:
        return self._members
