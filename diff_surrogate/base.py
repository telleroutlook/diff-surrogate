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
class SurrogateStats:
    train_losses: list = field(default_factory=list)
    correction_errors: list = field(default_factory=list)
    total_predictions: int = 0
    total_corrections: int = 0


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
