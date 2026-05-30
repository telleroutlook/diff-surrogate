"""Training utilities for surrogates: data generation helpers and training loops."""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch

from .base import SurrogateBase, _build_dataloader
from .convergence import ConvergenceAction, ConvergenceMonitor

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
        convergence_monitor: ConvergenceMonitor | None = None,
        optimizer_factory: Callable | None = None,
    ):
        self.surrogate = surrogate
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = loss_fn or torch.nn.MSELoss()
        if optimizer_factory is not None:
            self.optimizer = optimizer_factory(self.surrogate.get_network().parameters())
        elif hasattr(surrogate, "_optimizer") and surrogate._optimizer is not None:
            self.optimizer = surrogate._optimizer
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        else:
            self.optimizer = torch.optim.Adam(
                self.surrogate.get_network().parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None
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
        self.convergence_monitor = convergence_monitor

    def train(
        self,
        n_epochs: int = 10,
        n_samples: int = 256,
        batch_size: int = 32,
        grad_clip: float | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        loss_weights: dict[str, float] | None = None,
    ) -> list[float]:
        inputs, targets = self.surrogate.generate_training_data(n_samples)

        loader, target_keys = _build_dataloader(
            inputs,
            targets,
            batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        losses = []
        for epoch in range(n_epochs):
            epoch_loss = torch.zeros((), device=self.surrogate.device)
            for batch in loader:
                batch_x = batch[0].to(self.surrogate.device)
                self.optimizer.zero_grad()
                output = self.surrogate(batch_x)

                if target_keys is not None:
                    loss = sum(
                        (loss_weights.get(k, 1.0) if loss_weights else 1.0)
                        * self.loss_fn(output[k], batch[i + 1].to(self.surrogate.device))
                        for i, k in enumerate(target_keys)
                    )
                else:
                    loss = self.loss_fn(output, batch[1].to(self.surrogate.device))

                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.surrogate.get_network().parameters(), grad_clip
                    )
                self.optimizer.step()
                epoch_loss = epoch_loss + loss.detach()
            avg = (epoch_loss / len(loader)).item()
            losses.append(avg)
            self.surrogate.stats.train_losses.append(avg)
            if self.scheduler:
                self.scheduler.step()
            if self.convergence_monitor is not None:
                action = self.convergence_monitor.update(avg, epoch)
                if action == ConvergenceAction.EARLY_STOP:
                    break
                if action == ConvergenceAction.REDUCE_LR:
                    for pg in self.optimizer.param_groups:
                        pg["lr"] *= 0.5
                    if self.scheduler is not None and hasattr(self.scheduler, "base_lrs"):
                        self.scheduler.base_lrs = [lr * 0.5 for lr in self.scheduler.base_lrs]
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
