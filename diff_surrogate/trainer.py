"""Training utilities for surrogates: data generation helpers and training loops."""
from __future__ import annotations

import logging
from typing import Callable

import torch

from .base import SurrogateBase

logger = logging.getLogger(__name__)


class SurrogateTrainer:
    """Configurable trainer for SurrogateBase instances."""

    def __init__(
        self,
        surrogate: SurrogateBase,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        loss_fn: Callable | None = None,
        scheduler: str | None = None,
        scheduler_kwargs: dict | None = None,
    ):
        self.surrogate = surrogate
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = loss_fn or torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.surrogate.get_network().parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        if scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, **(scheduler_kwargs or {"T_max": 100})
            )
        elif scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, **(scheduler_kwargs or {"step_size": 30, "gamma": 0.1})
            )
        else:
            self.scheduler = None

    def train(self, n_epochs: int = 10, n_samples: int = 256) -> list[float]:
        losses = []
        for epoch in range(n_epochs):
            inputs, targets = self.surrogate.generate_training_data(n_samples)
            inputs = inputs.to(self.surrogate.device)
            if isinstance(targets, dict):
                targets = {k: v.to(self.surrogate.device) for k, v in targets.items()}
            else:
                targets = targets.to(self.surrogate.device)

            self.optimizer.zero_grad()
            output = self.surrogate.forward(inputs)

            if isinstance(targets, dict) and isinstance(output, dict):
                loss = sum(self.loss_fn(output[k], targets[k]) for k in targets)
            else:
                loss = self.loss_fn(output, targets)

            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            losses.append(loss.item())
            self.surrogate.stats.train_losses.append(loss.item())
        return losses

    def state_dict(self) -> dict:
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
        }

    def load_state_dict(self, state: dict):
        self.optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") and self.scheduler:
            self.scheduler.load_state_dict(state["scheduler"])
