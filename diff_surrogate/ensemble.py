"""Ensemble surrogate with uncertainty quantification."""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Callable
from .base import SurrogateBase, CorrectionPolicy, AdaptiveCorrectionPolicy, SurrogateStats
from .cnn import CNNSurrogate
from .mlp import MLPSurrogate


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
        self._members: list[SurrogateBase] = [base_factory() for _ in range(n_members)]
        super().__init__(correction_policy=correction_policy, device=device)

    def _build_network(self) -> nn.ModuleList:
        return nn.ModuleList([m.get_network() for m in self._members])

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        member_outputs = [m.forward(x) for m in self._members]
        if isinstance(member_outputs[0], dict):
            keys = member_outputs[0].keys()
            result = {}
            for key in keys:
                stacked = torch.stack([out[key] for out in member_outputs], dim=0)  # (K, ...)
                result[key] = stacked.mean(dim=0)
                result[f"{key}_std"] = stacked.std(dim=0)
            return result
        else:
            stacked = torch.stack(member_outputs, dim=0)  # (K, ...)
            mean = stacked.mean(dim=0)
            std = stacked.std(dim=0)
            return {"output": mean, "output_std": std}

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._step += 1
        self.stats.total_predictions += 1
        with torch.no_grad():
            result = self.forward(x.to(self.device))
        if isinstance(self.correction_policy, AdaptiveCorrectionPolicy):
            std_vals = [v for k, v in result.items() if k.endswith("_std")]
            if std_vals:
                mean_std = torch.stack([s.mean() for s in std_vals]).mean().item()
                self.correction_policy.update_uncertainty(mean_std)
        return result

    def predict_with_uncertainty(self, x: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
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

    def get_members(self) -> list[SurrogateBase]:
        return self._members
