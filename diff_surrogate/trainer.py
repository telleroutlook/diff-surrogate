"""Training utilities for surrogates: data generation helpers and training loops."""

import torch
import torch.nn as nn


class SurrogateTrainer:
    """Configurable trainer for SurrogateBase instances."""

    def __init__(
        self,
        surrogate,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        scheduler: str | None = None,
        scheduler_kwargs: dict | None = None,
    ):
        self.surrogate = surrogate
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.history: list[float] = []

    def train(
        self,
        n_samples: int = 500,
        n_epochs: int = 100,
        batch_size: int = 32,
        loss_fn: nn.Module | None = None,
    ) -> list[float]:
        return self.surrogate.train_surrogate(
            n_samples=n_samples,
            n_epochs=n_epochs,
            lr=self.lr,
            batch_size=batch_size,
        )

    @staticmethod
    def random_field_data(
        n_samples: int,
        in_channels: int,
        out_channels: int,
        grid_size: int,
        device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.rand(n_samples, in_channels, grid_size, grid_size, device=device)
        targets = torch.randn(n_samples, out_channels, grid_size, grid_size, device=device)
        return inputs, targets
