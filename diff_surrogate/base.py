"""Abstract base for differentiable physics surrogates with correction lifecycle."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
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
        if step < 1:
            return False
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

    def __post_init__(self):
        self._current_interval: int = self.initial_interval
        self._last_correction_step: int = -1
        self._error_ema: float | None = None
        self._prev_error_ema: float | None = None
        self._uncertainty_baseline: float | None = None
        self._uncertainty_suppress_steps: int = 0

    def peek(self, step: int) -> bool:
        """Check whether correction is due without committing state."""
        if step < self.warmup_steps:
            return False
        if self._last_correction_step < 0:
            elapsed = self._current_interval
        else:
            elapsed = step - self._last_correction_step
        return elapsed >= self._current_interval

    def commit(self, step: int) -> None:
        """Record that a correction was performed at *step*."""
        self._last_correction_step = step

    def should_correct(self, step: int) -> bool:
        """Peek + commit: returns True and records step if correction is due."""
        if self.peek(step):
            self.commit(step)
            return True
        return False

    def update_error(self, error: float):
        """Call after each correction with the measured error."""
        new_ema = (
            error
            if self._error_ema is None
            else self.ema_alpha * error + (1 - self.ema_alpha) * self._error_ema
        )

        if self._prev_error_ema is not None and self._prev_error_ema > 0:
            ratio = new_ema / self._prev_error_ema
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

        The baseline is updated via EMA so it adapts to distribution shifts.
        """
        if self._uncertainty_baseline is None:
            self._uncertainty_baseline = avg_uncertainty
        else:
            self._uncertainty_baseline = 0.9 * self._uncertainty_baseline + 0.1 * avg_uncertainty

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
    train_losses: deque = field(default_factory=lambda: deque(maxlen=1000))
    correction_errors: deque = field(default_factory=lambda: deque(maxlen=1000))
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

    def __repr__(self) -> str:
        cls = type(self).__name__
        trained = "trained" if self._trained else "untrained"
        preds = self.stats.total_predictions
        corrs = self.stats.total_corrections
        return f"{cls}({trained}, predictions={preds}, corrections={corrs})"

    def __init__(
        self,
        correction_policy: CorrectionPolicy | None = None,
        device: str | torch.device | int = "cpu",
    ):
        super().__init__()
        self.correction_policy = correction_policy or CorrectionPolicy()
        self.device = torch.device(device)
        self.stats = SurrogateStats()
        self._network: nn.Module | None = None
        self._trained = False
        self._step = 0
        self._optimizer: torch.optim.Optimizer | None = None

    @property
    def trained(self) -> bool:
        return self._trained

    @abstractmethod
    def _build_network(self) -> nn.Module: ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]: ...

    @abstractmethod
    def generate_training_data(
        self, n_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor]]: ...

    def get_network(self) -> nn.Module:
        if self._network is None:
            self._network = self._build_network().to(self.device)
        return self._network

    def _apply(self, fn):
        dummy = torch.empty(0, device=self.device)
        self.device = fn(dummy).device
        result = super()._apply(fn)
        return result

    def train_surrogate(
        self,
        n_samples: int = 500,
        n_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 32,
        loss_weights: dict[str, float] | None = None,
    ) -> list[float]:
        from .trainer import SurrogateTrainer

        self._ensure_optimizer(lr)
        trainer = SurrogateTrainer(self, lr=lr)
        losses = trainer.train(
            n_epochs=n_epochs,
            n_samples=n_samples,
            batch_size=batch_size,
            loss_weights=loss_weights,
        )
        self._trained = True
        return losses

    def _ensure_optimizer(self, lr: float, weight_decay: float = 0.0) -> torch.optim.Optimizer:
        if not hasattr(self, "_optimizer") or self._optimizer is None:
            self._optimizer = torch.optim.Adam(
                self.get_network().parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            for pg in self._optimizer.param_groups:
                pg["lr"] = lr
        return self._optimizer

    def predict(self, x: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        self.stats.total_predictions += 1
        was_training = self.training
        self.eval()
        with torch.no_grad():
            result = self(x.to(self.device))
        if was_training:
            self.train()
        return result

    def predict_with_correction(
        self,
        x: torch.Tensor,
        true_solver_fn: Callable | None = None,
        step: int | None = None,
    ) -> tuple[Any, CorrectionAction]:
        """Predict with periodic ground-truth correction.

        When the correction policy triggers, the true solver output replaces
        the surrogate prediction (i.e. the returned value IS the correction)
        and the error is recorded for adaptive policy adjustment.

        Args:
            x: Input tensor.
            true_solver_fn: Optional ground-truth callable.
            step: External step counter. If provided, overrides internal counter.
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
                # Handle both Tensor and dict[str, Tensor] outputs
                if isinstance(surrogate_output, dict):
                    matched_keys = [k for k in surrogate_output if k in true_output]
                    error = sum(
                        torch.mean(
                            (true_output[k].to(surrogate_output[k].device)
                             - surrogate_output[k]) ** 2
                        ).item()
                        for k in matched_keys
                    ) / max(1, len(matched_keys))
                else:
                    error = torch.mean(
                        (true_output.to(surrogate_output.device) - surrogate_output) ** 2
                    ).item()
                self.stats.correction_errors.append(error)
                if isinstance(self.correction_policy, AdaptiveCorrectionPolicy):
                    self.correction_policy.update_error(error)
            if was_training:
                self.train()
            self.stats.total_corrections += 1
            return true_output, action

        return self.predict(x), action

    def accuracy(
        self,
        n_samples: int = 100,
        true_solver_fn: Callable | None = None,
    ) -> dict[str, float]:
        inputs, targets = self.generate_training_data(n_samples)
        inputs = inputs.to(self.device)
        if true_solver_fn is not None:
            with torch.no_grad():
                targets = true_solver_fn(inputs)
        if isinstance(targets, dict):
            targets = {k: v.to(self.device) for k, v in targets.items()}
        else:
            targets = targets.to(self.device)
        self.eval()
        with torch.no_grad():
            preds = self(inputs)
        if isinstance(preds, dict):
            total_mse = 0.0
            n_props = 0
            for key, pred_val in preds.items():
                if isinstance(targets, dict) and key in targets:
                    total_mse += torch.mean((pred_val - targets[key]) ** 2).item()
                    n_props += 1
            mse = total_mse / max(1, n_props)
        else:
            mse = torch.mean((preds - targets) ** 2).item()
        return {"mse": mse, "rmse": mse**0.5}

    def save_checkpoint(self, path: str, optimizer: Any | None = None):
        """Save state: network, stats, step, optimizer, correction policy, RNG."""
        checkpoint: dict[str, Any] = {
            "__format__": 2,
            "network_state_dict": self.get_network().state_dict(),
            "stats": {
                "train_losses": list(self.stats.train_losses),
                "correction_errors": list(self.stats.correction_errors),
                "total_predictions": self.stats.total_predictions,
                "total_corrections": self.stats.total_corrections,
                "per_property_accuracy": self.stats.per_property_accuracy,
                "uncertainty_calibration": self.stats.uncertainty_calibration,
            },
            "step": self._step,
            "trained": self._trained,
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
            checkpoint["optimizer_class"] = type(optimizer).__name__
            checkpoint["optimizer_lr"] = optimizer.defaults.get("lr", 1e-3)
            checkpoint["optimizer_weight_decay"] = optimizer.defaults.get("weight_decay", 0.0)
        elif hasattr(self, "_optimizer") and self._optimizer is not None:
            checkpoint["optimizer_state_dict"] = self._optimizer.state_dict()
            checkpoint["optimizer_class"] = type(self._optimizer).__name__
            checkpoint["optimizer_lr"] = self._optimizer.defaults.get("lr", 1e-3)
            checkpoint["optimizer_weight_decay"] = self._optimizer.defaults.get("weight_decay", 0.0)
        if isinstance(self.correction_policy, AdaptiveCorrectionPolicy):
            checkpoint["correction_policy"] = {
                "_current_interval": self.correction_policy._current_interval,
                "_error_ema": self.correction_policy._error_ema,
                "_prev_error_ema": self.correction_policy._prev_error_ema,
                "_uncertainty_baseline": self.correction_policy._uncertainty_baseline,
                "_uncertainty_suppress_steps": self.correction_policy._uncertainty_suppress_steps,
            }
        convergence_monitor = getattr(self, "_convergence_monitor", None)
        if convergence_monitor is not None:
            checkpoint["convergence_history"] = convergence_monitor.history
        checkpoint["rng_state"] = {
            "torch": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            checkpoint["rng_state"]["cuda"] = torch.cuda.get_rng_state_all()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, weights_only: bool = True):
        """Restore network state, stats, step count, and all saved state from a file."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=weights_only)
        fmt = checkpoint.get("__format__", 0)
        if fmt > 2:
            raise ValueError(
                f"Unsupported checkpoint format {fmt}; expected <= 2. "
                "You may need to update diff-surrogate."
            )
        if fmt < 1:
            raise ValueError(
                f"Unsupported checkpoint format {fmt}. "
                "You may need to re-train or convert the checkpoint."
            )
        self.get_network().load_state_dict(checkpoint["network_state_dict"])
        self.get_network().to(self.device)
        stats = checkpoint["stats"]
        self.stats.train_losses = deque(stats["train_losses"], maxlen=1000)
        self.stats.correction_errors = deque(stats["correction_errors"], maxlen=1000)
        self.stats.total_predictions = stats["total_predictions"]
        self.stats.total_corrections = stats["total_corrections"]
        self.stats.per_property_accuracy = stats.get("per_property_accuracy", {})
        self.stats.uncertainty_calibration = stats.get("uncertainty_calibration", 0.0)
        self._step = checkpoint["step"]
        self._trained = checkpoint["trained"]
        if "correction_policy" in checkpoint and isinstance(
            self.correction_policy, AdaptiveCorrectionPolicy
        ):
            cp = checkpoint["correction_policy"]
            self.correction_policy._current_interval = cp["_current_interval"]
            self.correction_policy._error_ema = cp["_error_ema"]
            self.correction_policy._prev_error_ema = cp["_prev_error_ema"]
            self.correction_policy._uncertainty_baseline = cp["_uncertainty_baseline"]
            self.correction_policy._uncertainty_suppress_steps = cp["_uncertainty_suppress_steps"]
        if "convergence_history" in checkpoint:
            monitor = getattr(self, "_convergence_monitor", None)
            if monitor is not None:
                monitor._history = checkpoint["convergence_history"]
        if "rng_state" in checkpoint:
            rng = checkpoint["rng_state"]
            torch.random.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and "cuda" in rng:
                torch.cuda.set_rng_state_all(rng["cuda"])
        if "optimizer_state_dict" in checkpoint:
            opt_cls_name = checkpoint.get("optimizer_class", "Adam")
            opt_lr = checkpoint.get("optimizer_lr", 1e-3)
            opt_wd = checkpoint.get("optimizer_weight_decay", 0.0)
            opt_cls = getattr(torch.optim, opt_cls_name, torch.optim.Adam)
            self._optimizer = opt_cls(
                self.get_network().parameters(), lr=opt_lr, weight_decay=opt_wd
            )
            self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            self._optimizer = None


def _build_dataloader(
    inputs: torch.Tensor,
    targets: torch.Tensor | dict[str, torch.Tensor],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[torch.utils.data.DataLoader, list[str] | None]:
    """Build a DataLoader handling dict or tensor targets."""
    if isinstance(targets, dict):
        dataset = torch.utils.data.TensorDataset(inputs, *targets.values())
        return (
            torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
            ),
            list(targets.keys()),
        )
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    return (
        torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        None,
    )
