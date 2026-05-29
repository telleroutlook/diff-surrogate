"""Ensemble surrogate with uncertainty quantification."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from .base import AdaptiveCorrectionPolicy, CorrectionPolicy, SurrogateBase


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
    ):
        self.base_factory = base_factory
        self.n_members = n_members
        super().__init__(correction_policy=correction_policy, device=device)
        self._members = nn.ModuleList([base_factory() for _ in range(n_members)])

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
        with torch.no_grad():
            result = self(x.to(self.device))
        if isinstance(self.correction_policy, AdaptiveCorrectionPolicy):
            std_vals = [v for k, v in result.items() if k.endswith("_std")]
            if std_vals:
                mean_std = torch.stack([s.mean() for s in std_vals]).mean().item()
                self.correction_policy.update_uncertainty(mean_std)
        return result

    def train_surrogate(
        self,
        n_samples: int = 256,
        n_epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Train each ensemble member independently for diversity."""
        all_losses = []
        for member in self._members:
            losses = member.train_surrogate(n_samples, n_epochs, lr, batch_size)
            all_losses.append(losses)
        self._trained = True
        return all_losses

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

    def generate_training_data(self, n_samples: int) -> tuple:
        return self._members[0].generate_training_data(n_samples)

    def get_members(self) -> nn.ModuleList:
        return self._members
