"""Training utilities for surrogates: data generation helpers and training loops."""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
import torch.nn.functional as F

from .base import SurrogateBase, _build_dataloader
from .convergence import ConvergenceAction, ConvergenceMonitor

logger = logging.getLogger(__name__)


class SobolevLoss(torch.nn.Module):
    """MSE + optional derivative-matching (Sobolev) loss.

    When ``derivative_weight`` > 0 the loss also penalises discrepancies
    between surrogate and target *gradients* w.r.t. the inputs.

    Args:
        derivative_weight: Relative weight of the gradient MSE term.
            0.0 means plain MSE (no derivative matching).
        base_loss: Underlying value-level loss.  Defaults to MSELoss.
    """

    def __init__(
        self,
        derivative_weight: float = 1.0,
        base_loss: torch.nn.Module | None = None,
    ):
        super().__init__()
        self.derivative_weight = derivative_weight
        self.base_loss = base_loss or torch.nn.MSELoss()

    def forward(
        self,
        surrogate_output: torch.Tensor,
        target: torch.Tensor,
        inputs: torch.Tensor,
        surrogate_grad: torch.Tensor | None = None,
        target_grad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute combined value + gradient MSE.

        Pre-computed gradients can be passed directly; otherwise they are
        computed internally via ``torch.autograd.grad``.
        """
        value_loss = self.base_loss(surrogate_output, target)
        if self.derivative_weight <= 0.0:
            return value_loss

        if surrogate_grad is None:
            surrogate_grad = _compute_grad(surrogate_output, inputs)
        if target_grad is None:
            target_grad = _compute_grad(target, inputs)

        grad_loss = F.mse_loss(surrogate_grad, target_grad)
        return value_loss + self.derivative_weight * grad_loss


def _compute_grad(output: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Compute ∂output/∂inputs, summed over the output dimension."""
    grad = torch.autograd.grad(
        outputs=output,
        inputs=inputs,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
    )[0]
    return grad


def _finite_diff_grad(
    surrogate: SurrogateBase,
    x: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Approximate ∂surrogate/∂x via central finite differences."""
    x = x.detach()
    was_training = surrogate.training
    surrogate.eval()
    with torch.no_grad():
        output_plus = surrogate(x + eps)
        output_minus = surrogate(x - eps)
        if isinstance(output_plus, dict):
            output_plus = next(iter(output_plus.values()))
        if isinstance(output_minus, dict):
            output_minus = next(iter(output_minus.values()))
        grad = (output_plus - output_minus) / (2.0 * eps)
    if was_training:
        surrogate.train()
    return grad


def gradient_fidelity_score(
    surrogate: SurrogateBase,
    inputs: torch.Tensor,
    true_grad_fn: Callable[[torch.Tensor], torch.Tensor],
) -> dict[str, float]:
    """Evaluate how well the surrogate's gradients match ground truth.

    Args:
        surrogate: A trained (or untrained) surrogate.
        inputs: Batch of input tensors (``requires_grad`` will be enabled).
        true_grad_fn: Callable returning the true ∂y/∂x for a given input.

    Returns:
        Dict with ``cosine_similarity``, ``relative_error``, and
        ``max_absolute_error``.
    """
    surrogate.eval()
    # Determine network dtype from first parameter, default float32
    net = surrogate.get_network()
    try:
        net_dtype = next(net.parameters()).dtype
    except StopIteration:
        net_dtype = torch.float32

    x = inputs.detach().to(device=surrogate.device, dtype=net_dtype).requires_grad_(True)

    output = surrogate(x)
    if isinstance(output, dict):
        output = next(iter(output.values()))

    surrogate_grad = _compute_grad(output, x).to(torch.float64)
    true_grad = true_grad_fn(inputs.detach().to(surrogate.device)).to(torch.float64)

    flat_sg = surrogate_grad.flatten()
    flat_tg = true_grad.flatten()

    cosine_sim = F.cosine_similarity(flat_sg.unsqueeze(0), flat_tg.unsqueeze(0)).item()

    tg_norm = flat_tg.norm().item()
    relative_error = (
        (flat_sg - flat_tg).norm().item() / (tg_norm + 1e-30)
        if tg_norm > 0
        else 0.0
    )

    max_abs_error = (flat_sg - flat_tg).abs().max().item()

    return {
        "cosine_similarity": cosine_sim,
        "relative_error": relative_error,
        "max_absolute_error": max_abs_error,
    }


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
        derivative_weight: float = 0.0,
        target_grad_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> list[float]:
        inputs, targets = self.surrogate.generate_training_data(n_samples)

        loader, target_keys = _build_dataloader(
            inputs,
            targets,
            batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        use_sobolev = derivative_weight > 0.0

        losses = []
        for epoch in range(n_epochs):
            epoch_loss = torch.zeros((), device=self.surrogate.device)
            for batch in loader:
                batch_x = batch[0].to(self.surrogate.device)
                self.optimizer.zero_grad()

                if use_sobolev:
                    batch_x = batch_x.detach().requires_grad_(True)

                output = self.surrogate(batch_x)

                if use_sobolev and target_keys is None:
                    target_tensor = batch[1].to(self.surrogate.device).detach()
                    value_loss = torch.nn.functional.mse_loss(output, target_tensor)
                    surrogate_grad = _compute_grad(output, batch_x)

                    if target_grad_fn is not None:
                        true_grad = target_grad_fn(batch_x)
                    else:
                        true_grad = _finite_diff_grad(
                            self.surrogate, batch_x.detach()
                        )

                    grad_loss = torch.nn.functional.mse_loss(surrogate_grad, true_grad)
                    loss = value_loss + derivative_weight * grad_loss
                elif target_keys is not None:
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
