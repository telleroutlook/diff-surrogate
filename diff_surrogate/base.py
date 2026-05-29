"""Abstract base for differentiable physics surrogates with correction lifecycle."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
import torch.nn as nn


class CorrectionAction(Enum):
    CONTINUE = "continue"
    CORRECT = "correct"


@dataclass
class CorrectionPolicy:
    correction_interval: int = 10
    warmup_steps: int = 0

    def should_correct(self, step: int) -> bool:
        if step < self.warmup_steps:
            return False
        if self.correction_interval <= 0:
            return False
        return step % self.correction_interval == 0


@dataclass
class AdaptiveCorrectionPolicy:
    """Correction policy that adjusts frequency based on surrogate accuracy trends.

    If correction errors are growing, correct more often (decrease interval).
    If errors are stable and small, correct less often (increase interval).
    """
    min_interval: int = 2
    max_interval: int = 50
    initial_interval: int = 10
    warmup_steps: int = 5
    growth_threshold: float = 1.5
    shrink_threshold: float = 0.5
    ema_alpha: float = 0.2

    _current_interval: int = 0
    _error_ema: float | None = None
    _prev_error_ema: float | None = None
    _uncertainty_baseline: float | None = None
    _uncertainty_suppress_steps: int = 0

    def __post_init__(self):
        self._current_interval = self.initial_interval
        self._step = 0

    def should_correct(self, step: int) -> bool:
        self._step = step
        return step >= self.warmup_steps and step % self._current_interval == 0

    def update_error(self, error: float):
        """Call after each correction with the measured error."""
        new_ema = error if self._error_ema is None else self.ema_alpha * error + (1 - self.ema_alpha) * self._error_ema

        if self._prev_error_ema is not None and self._error_ema is not None and self._error_ema > 0:
            ratio = new_ema / self._error_ema
            if ratio > self.growth_threshold:
                self._current_interval = max(self.min_interval, self._current_interval - 2)
            elif ratio < self.shrink_threshold:
                self._current_interval = min(self.max_interval, self._current_interval + 2)

        self._prev_error_ema = self._error_ema
        self._error_ema = new_ema

    @property
    def current_interval(self) -> int:
        return self._current_interval

    def update_uncertainty(self, avg_uncertainty: float):
        """Adjust correction interval based on ensemble uncertainty.

        If uncertainty exceeds ``growth_threshold`` times the baseline,
        temporarily halve the correction interval.  Otherwise, gradually
        restore the interval by incrementing it back up.
        """
        if self._uncertainty_baseline is None:
            self._uncertainty_baseline = avg_uncertainty

        if avg_uncertainty > self.growth_threshold * self._uncertainty_baseline:
            self._current_interval = max(self.min_interval, self._current_interval // 2)
            self._uncertainty_suppress_steps = 5
        else:
            if self._uncertainty_suppress_steps > 0:
                self._uncertainty_suppress_steps -= 1
            elif self._current_interval < self.initial_interval:
                self._current_interval = min(self.max_interval, self._current_interval + 1)


@dataclass
class SurrogateStats:
    train_losses: list = field(default_factory=list)
    correction_errors: list = field(default_factory=list)
    total_predictions: int = 0
    total_corrections: int = 0
    per_property_accuracy: dict[str, float] = field(default_factory=dict)
    uncertainty_calibration: float = 0.0

    def update_accuracy(self, property_name: str, error: float):
        self.per_property_accuracy[property_name] = error


class SurrogateBase(ABC, nn.Module):
    """Base class for differentiable physics surrogates.

    Lifecycle: generate training data -> train -> predict with periodic correction.

    Subclasses must implement:
        - _build_network() -> nn.Module
        - forward(x) -> Any
        - generate_training_data(n_samples) -> tuple[Tensor, Tensor]
    """

    def __init__(
        self,
        correction_policy: CorrectionPolicy | None = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.correction_policy = correction_policy or CorrectionPolicy()
        self.device = torch.device(device)
        self.stats = SurrogateStats()
        self._network: nn.Module | None = None
        self._trained = False
        self._step = 0

    @property
    def trained(self) -> bool:
        return self._trained

    @abstractmethod
    def _build_network(self) -> nn.Module:
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def generate_training_data(self, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def get_network(self) -> nn.Module:
        if self._network is None:
            self._network = self._build_network()
        return self._network

    def train_surrogate(
        self,
        n_samples: int = 500,
        n_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 32,
    ) -> list[float]:
        net = self.get_network().to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        criterion = nn.MSELoss()

        inputs, targets = self.generate_training_data(n_samples)
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        dataset = torch.utils.data.TensorDataset(inputs, targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        losses = []
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = net(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)

        self._trained = True
        self.stats.train_losses = losses
        return losses

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self._step += 1
        self.stats.total_predictions += 1
        with torch.no_grad():
            return self.get_network()(x.to(self.device))

    def predict_with_correction(
        self,
        x: torch.Tensor,
        true_solver_fn: callable | None = None,
    ) -> tuple[torch.Tensor, CorrectionAction]:
        action = CorrectionAction.CONTINUE
        if true_solver_fn is not None and self.correction_policy.should_correct(self._step):
            action = CorrectionAction.CORRECT
            with torch.no_grad():
                true_output = true_solver_fn(x)
                surrogate_output = self.get_network()(x.to(self.device))
                error = torch.mean((true_output - surrogate_output) ** 2).item()
                self.stats.correction_errors.append(error)
            self.stats.total_corrections += 1
            return true_output, action

        return self.predict(x), action

    def accuracy(
        self,
        n_samples: int = 100,
        true_solver_fn: callable | None = None,
    ) -> dict[str, float]:
        if true_solver_fn is None:
            true_solver_fn = self.generate_training_data

        inputs, targets = self.generate_training_data(n_samples)
        with torch.no_grad():
            preds = self.get_network()(inputs.to(self.device))
        mse = torch.mean((preds - targets.to(self.device)) ** 2).item()
        return {"mse": mse, "rmse": mse**0.5}

    def save_checkpoint(self, path: str):
        """Save network state, stats, and step count to a file."""
        checkpoint = {
            "network_state_dict": self.get_network().state_dict(),
            "stats": {
                "train_losses": self.stats.train_losses,
                "correction_errors": self.stats.correction_errors,
                "total_predictions": self.stats.total_predictions,
                "total_corrections": self.stats.total_corrections,
                "per_property_accuracy": self.stats.per_property_accuracy,
                "uncertainty_calibration": self.stats.uncertainty_calibration,
            },
            "step": self._step,
            "trained": self._trained,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Restore network state, stats, and step count from a file."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.get_network().load_state_dict(checkpoint["network_state_dict"])
        self.get_network().to(self.device)
        stats = checkpoint["stats"]
        self.stats.train_losses = stats["train_losses"]
        self.stats.correction_errors = stats["correction_errors"]
        self.stats.total_predictions = stats["total_predictions"]
        self.stats.total_corrections = stats["total_corrections"]
        self.stats.per_property_accuracy = stats.get("per_property_accuracy", {})
        self.stats.uncertainty_calibration = stats.get("uncertainty_calibration", 0.0)
        self._step = checkpoint["step"]
        self._trained = checkpoint["trained"]
