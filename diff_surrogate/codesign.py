"""Domain-agnostic co-design workflow API.  :stable:

Provides :class:`CoDesignWorkflow` for coupled multi-physics optimization
and :class:`CoupledLoss` for assembling weighted loss terms from arbitrary
domain-specific forward functions.

API contract
------------
* ``forward_fns`` — ``dict[str, Callable]`` mapping domain names to callables
  that accept ``(design_params, **kwargs)`` and return a dict of outputs.
* ``loss_fn`` — ``Callable[..., torch.Tensor]`` that consumes the merged
  output dict from all forward passes and returns a scalar loss.
* ``coupling_fn`` — optional ``Callable[[dict], dict]`` that post-processes
  the merged output dict (e.g. lithography contour feeds into EM solver).
* ``step()`` — performs one optimizer step.
* ``run(n_steps)`` — full optimisation loop returning ``(params, history)``.
* ``compare_baseline(n_steps)`` — runs each domain independently and returns
  results for A/B comparison.
* ``report()`` — returns a comparison metrics dict between the last coupled
  run and the last baseline run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = [
    "CoDesignWorkflow",
    "CoupledLoss",
]

# ---------------------------------------------------------------------------
# CoupledLoss — weighted sum of named loss components
# ---------------------------------------------------------------------------


class CoupledLoss:
    """Weighted sum of named scalar loss terms.  :stable:

    Parameters
    ----------
    components : dict[str, Callable[..., torch.Tensor]]
        Mapping of component name → callable that returns a scalar tensor.
    weights : dict[str, float]
        Mapping of component name → weight.  Missing keys default to ``1.0``.

    Example
    -------
    >>> loss = CoupledLoss(
    ...     components={"optical": optical_loss, "litho": litho_loss},
    ...     weights={"optical": 1.0, "litho": 0.1},
    ... )
    >>> total, breakdown = loss()
    """

    def __init__(
        self,
        components: dict[str, Callable[..., torch.Tensor]],
        weights: dict[str, float] | None = None,
    ) -> None:
        self.components = components
        self.weights = (
            {name: weights.get(name, 1.0) for name in components}
            if weights
            else {n: 1.0 for n in components}
        )

    def __call__(self, **kwargs: Any) -> tuple[torch.Tensor, dict[str, float]]:
        """Evaluate all components and return ``(total, breakdown)``.  :stable:

        *kwargs* are forwarded to every component callable.
        """
        breakdown: dict[str, float] = {}
        parts: list[torch.Tensor] = []
        for name, fn in self.components.items():
            val = fn(**kwargs)
            if val.numel() != 1:
                val = val.mean()
            w = self.weights[name]
            parts.append(w * val)
            breakdown[name] = val.detach().item()
        total = sum(parts)  # type: ignore[arg-type]
        breakdown["total"] = total.detach().item()
        return total, breakdown


# ---------------------------------------------------------------------------
# CoDesignWorkflow — domain-agnostic coupled optimisation
# ---------------------------------------------------------------------------


@dataclass
class _RunResult:
    """Internal container for a single run's outputs."""

    params: torch.Tensor
    loss_history: list[float] = field(default_factory=list)
    breakdown_history: list[dict[str, float]] = field(default_factory=list)


class CoDesignWorkflow:
    """Domain-agnostic co-design optimisation loop.  :stable:

    Couples multiple physics domains through a shared design parameter tensor
    and an optional coupling function that mediates information flow between
    domain outputs.

    Parameters
    ----------
    design_params : torch.Tensor
        Design parameter tensor (will be cloned and ``requires_grad=True`` set).
    forward_fns : dict[str, Callable]
        ``{domain_name: forward_fn(design_params) -> dict}``.
    loss_fn : Callable[..., torch.Tensor] or CoupledLoss
        Accepts merged forward outputs, returns scalar loss.
    coupling_fn : Callable[[dict], dict] or None
        Optional post-processing of the merged output dict before loss
        evaluation.  Use this to pass outputs of one domain as inputs to
        another (e.g. lithography contour → EM solver).
    lr : float
        Learning rate for Adam optimiser.
    max_grad_norm : float or None
        Gradient clipping threshold.  ``None`` disables clipping.
    param_bounds : tuple[float, float] or None
        Optional clamp range applied after each step.

    Example
    -------
    >>> wf = CoDesignWorkflow(
    ...     design_params=torch.rand(32, 32),
    ...     forward_fns={"em": em_forward, "litho": litho_forward},
    ...     loss_fn=my_coupled_loss,
    ...     coupling_fn=litho_to_em_coupling,
    ... )
    >>> params, history = wf.run(n_steps=200)
    """

    def __init__(
        self,
        design_params: torch.Tensor,
        forward_fns: dict[str, Callable],
        loss_fn: Callable[..., torch.Tensor] | CoupledLoss,
        coupling_fn: Callable[[dict], Any] | None = None,
        lr: float = 1e-2,
        max_grad_norm: float | None = 1.0,
        param_bounds: tuple[float, float] | None = (0.0, 1.0),
    ) -> None:
        self.params = design_params.clone().detach().requires_grad_(True)
        self.forward_fns = forward_fns
        self._loss_fn = loss_fn
        self.coupling_fn = coupling_fn
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.param_bounds = param_bounds

        self._optimizer = torch.optim.Adam([self.params], lr=lr)
        self._step_count = 0
        self._coupled_result: _RunResult | None = None
        self._baseline_result: _RunResult | None = None

    # -- forward merge -------------------------------------------------------

    def _forward(self, params: torch.Tensor) -> dict[str, Any]:
        """Run all domain forward functions and merge outputs."""
        merged: dict[str, Any] = {}
        for name, fn in self.forward_fns.items():
            merged[name] = fn(params)
        if self.coupling_fn is not None:
            merged = self.coupling_fn(merged)
        return merged

    # -- single step ---------------------------------------------------------

    def step(self) -> tuple[float, dict[str, float]]:
        """Perform one optimisation step.  :stable:

        Returns ``(loss_value, breakdown_dict)``.
        """
        merged = self._forward(self.params)

        if isinstance(self._loss_fn, CoupledLoss):
            total, breakdown = self._loss_fn(**merged)
        else:
            total = self._loss_fn(**merged)
            breakdown = {"total": total.detach().item()}

        if torch.isnan(total):
            return float("nan"), breakdown

        self._optimizer.zero_grad()
        total.backward()

        if self.max_grad_norm is not None and self.params.grad is not None:
            if torch.isnan(self.params.grad).any():
                return float("nan"), breakdown
            torch.nn.utils.clip_grad_norm_([self.params], self.max_grad_norm)

        self._optimizer.step()

        if self.param_bounds is not None:
            with torch.no_grad():
                self.params.clamp_(*self.param_bounds)

        self._step_count += 1
        return total.item(), breakdown

    # -- full loop -----------------------------------------------------------

    def run(
        self,
        n_steps: int = 200,
        verbose: bool = True,
        log_every: int = 50,
    ) -> tuple[torch.Tensor, list[float]]:
        """Run the coupled optimisation for *n_steps*.  :stable:

        Returns ``(final_params, loss_history)``.
        """
        self.params = self.params.detach().clone().requires_grad_(True)
        self._optimizer = torch.optim.Adam([self.params], lr=self.lr)
        self._step_count = 0

        loss_history: list[float] = []
        breakdown_history: list[dict[str, float]] = []

        for i in range(n_steps):
            loss_val, breakdown = self.step()
            loss_history.append(loss_val)
            breakdown_history.append(breakdown)

            if verbose and i % log_every == 0:
                parts = " ".join(f"{k}={v:.6f}" for k, v in breakdown.items() if k != "total")
                print(f"[co-design] step {i:4d}  total={loss_val:.6f} {parts}")

            if loss_val != loss_val:  # NaN check
                if verbose:
                    print(f"[co-design] NaN at step {i}, stopping.")
                break

        self._coupled_result = _RunResult(
            params=self.params.detach().clone(),
            loss_history=loss_history,
            breakdown_history=breakdown_history,
        )
        return self.params.detach().clone(), loss_history

    # -- decoupled baseline --------------------------------------------------

    def compare_baseline(
        self,
        n_steps: int = 200,
        verbose: bool = True,
    ) -> tuple[torch.Tensor, list[float]]:
        """Run each domain independently (no coupling) as a baseline.  :stable:

        Each domain gets its own parameter copy and Adam optimiser.
        Returns ``(final_params, loss_history)``.
        """
        base_params = self.params.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([base_params], lr=self.lr)

        loss_history: list[float] = []

        for i in range(n_steps):
            # Run forward with no coupling
            merged: dict[str, Any] = {}
            for name, fn in self.forward_fns.items():
                merged[name] = fn(base_params)

            if isinstance(self._loss_fn, CoupledLoss):
                total, _ = self._loss_fn(**merged)
            else:
                total = self._loss_fn(**merged)

            if torch.isnan(total):
                if verbose:
                    print(f"[baseline] NaN at step {i}, stopping.")
                break

            optimizer.zero_grad()
            total.backward()

            if self.max_grad_norm is not None and base_params.grad is not None:
                torch.nn.utils.clip_grad_norm_([base_params], self.max_grad_norm)

            optimizer.step()

            if self.param_bounds is not None:
                with torch.no_grad():
                    base_params.clamp_(*self.param_bounds)

            loss_history.append(total.item())

            if verbose and i % 50 == 0:
                print(f"[baseline] step {i:4d}  total={total.item():.6f}")

        self._baseline_result = _RunResult(
            params=base_params.detach().clone(),
            loss_history=loss_history,
        )
        return base_params.detach().clone(), loss_history

    # -- comparison report ---------------------------------------------------

    def report(self) -> dict[str, Any]:
        """Generate comparison metrics between coupled and baseline runs.  :stable:

        Requires both ``run()`` and ``compare_baseline()`` to have been called.
        Returns a dict with final losses, relative improvement, and histories.
        """
        result: dict[str, Any] = {}
        if self._coupled_result is not None:
            result["coupled_final_loss"] = self._coupled_result.loss_history[-1]
            result["coupled_steps"] = len(self._coupled_result.loss_history)
            result["coupled_history"] = self._coupled_result.loss_history
            if self._coupled_result.breakdown_history:
                result["coupled_breakdown_last"] = self._coupled_result.breakdown_history[-1]
        if self._baseline_result is not None:
            result["baseline_final_loss"] = self._baseline_result.loss_history[-1]
            result["baseline_steps"] = len(self._baseline_result.loss_history)
            result["baseline_history"] = self._baseline_result.loss_history
        if self._coupled_result is not None and self._baseline_result is not None:
            c = self._coupled_result.loss_history[-1]
            b = self._baseline_result.loss_history[-1]
            result["improvement_pct"] = (b - c) / max(abs(b), 1e-12) * 100
        return result
