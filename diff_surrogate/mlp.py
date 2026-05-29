"""MLP-based surrogate for scalar property prediction with physics constraints."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from enum import Enum

import torch
import torch.nn as nn

from .base import CorrectionPolicy, SurrogateBase


class Constraint(Enum):
    """Physics constraint for an MLP output head."""

    NONE = "none"
    MONOTONE = "monotone"
    POSITIVE = "positive"


class MonotoneLinear(nn.Module):
    """Linear layer with monotonicity constraint via softplus-parameterized positive weights.

    Uses ``softplus(raw_weight)`` instead of ``abs(raw_weight)`` to ensure
    differentiability at 0 and stable gradient flow during training.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.raw_weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        temp = torch.empty_like(self.raw_weight)
        nn.init.kaiming_uniform_(temp, nonlinearity="relu")
        temp.abs_()
        with torch.no_grad():
            clamped = torch.clamp(temp, min=1e-6)
            # y + log(1 - exp(-y)) — numerically stable inverse softplus
            self.raw_weight.copy_(clamped + torch.log(-torch.expm1(-clamped)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.nn.functional.softplus(self.raw_weight)
        out = x @ w.T
        if self.bias is not None:
            out = out + self.bias
        return out


class MonotoneMLP(nn.Module):
    """MLP with approximate monotonicity constraint via positive weights.

    Uses ``MonotoneLinear`` (softplus-parameterized weights) to guarantee
    non-negative weight matrices. Combined with ReLU activations, this
    provides approximate monotonicity: non-decreasing along each positive-signed
    input dimension. Not strictly guaranteed due to ReLU's non-smoothness.

    Args:
        in_features: Number of input features.
        hidden: Hidden layer width.
        n_layers: Total number of linear layers.
        monotone_signs: Per-input-dimension sign constraint.
            +1.0 forces non-decreasing, -1.0 forces non-increasing for that
            input dimension.  If None, all inputs are constrained to be
            non-decreasing (all +1.0).
    """

    def __init__(
        self,
        in_features: int,
        hidden: int = 64,
        n_layers: int = 3,
        monotone_signs: list[float] | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        if monotone_signs is not None:
            assert len(monotone_signs) == in_features
            self.register_buffer("_signs", torch.tensor(monotone_signs).reshape(1, -1))
        else:
            self.register_buffer("_signs", torch.ones(1, in_features))
        layers = [MonotoneLinear(in_features, hidden), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers.extend([MonotoneLinear(hidden, hidden), nn.ReLU()])
        layers.append(MonotoneLinear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        signed = x * self._signs.to(x.device)  # type: ignore[operator]
        return self.net(signed)


class PositiveOutputMLP(nn.Module):
    """MLP that guarantees positive output via softplus activation."""

    def __init__(self, in_features: int, hidden: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(in_features, hidden), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        self.positive = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.positive(self.net(x))


class MLPSurrogate(SurrogateBase):
    """MLP surrogate for scalar property prediction (T,P -> rho,h,s,cp,...).

    Each output property gets its own MLP with optional physics constraints.
    """

    def __init__(
        self,
        n_inputs: int = 2,
        properties: list[str] | None = None,
        hidden: int = 64,
        n_layers: int = 3,
        constrained: dict[str, Constraint] | None = None,
        correction_policy: CorrectionPolicy | None = None,
        device: str | torch.device | int = "cpu",
        data_generator: Callable | None = None,
        shared_trunk: bool = False,
    ):
        self.n_inputs = n_inputs
        self.properties = properties or ["value"]
        self.hidden = hidden
        self.n_layers = n_layers
        self.constrained = constrained or {}
        self._data_generator = data_generator
        self.shared_trunk = shared_trunk
        if shared_trunk and self.constrained:
            warnings.warn(
                "shared_trunk=True ignores constrained settings; "
                "per-property constraints require independent heads.",
                UserWarning,
                stacklevel=2,
            )
            self.constrained = {}
        if n_inputs < 1:
            raise ValueError(f"n_inputs must be >= 1, got {n_inputs}")
        if n_layers < 2:
            raise ValueError(f"n_layers must be >= 2, got {n_layers}")
        super().__init__(correction_policy=correction_policy, device=device)

    def _build_network(self) -> nn.ModuleDict:
        nets: dict[str, nn.Module] = {}

        if self.shared_trunk:
            # Shared trunk: first n_layers-1 layers shared, final layer per-property
            trunk_depth = max(1, self.n_layers - 1)
            trunk_layers = [nn.Linear(self.n_inputs, self.hidden), nn.ReLU()]
            for _ in range(trunk_depth - 1):
                trunk_layers.extend([nn.Linear(self.hidden, self.hidden), nn.ReLU()])
            nets["trunk"] = nn.Sequential(*trunk_layers)
            for prop in self.properties:
                nets[f"head_{prop}"] = nn.Linear(self.hidden, 1)
            return nn.ModuleDict(nets)

        for prop in self.properties:
            constraint = self.constrained.get(prop, Constraint.NONE)
            if constraint == Constraint.MONOTONE:
                nets[prop] = MonotoneMLP(self.n_inputs, self.hidden, self.n_layers)
            elif constraint == Constraint.POSITIVE:
                nets[prop] = PositiveOutputMLP(self.n_inputs, self.hidden, self.n_layers)
            else:
                layers = [nn.Linear(self.n_inputs, self.hidden), nn.ReLU()]
                for _ in range(self.n_layers - 2):
                    layers.extend([nn.Linear(self.hidden, self.hidden), nn.ReLU()])
                layers.append(nn.Linear(self.hidden, 1))
                nets[prop] = nn.Sequential(*layers)
        return nn.ModuleDict(nets)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        net = self.get_network()
        assert isinstance(net, nn.ModuleDict)
        if self.shared_trunk:
            trunk = net["trunk"]
            features = trunk(x)
            return {prop: net[f"head_{prop}"](features)[..., 0] for prop in self.properties}
        return {prop: module(x)[..., 0] for prop, module in net.items()}

    def generate_training_data(
        self, n_samples: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self._data_generator is not None:
            return self._data_generator(n_samples)
        inputs = torch.randn(n_samples, self.n_inputs)
        targets = {prop: torch.randn(n_samples) for prop in self.properties}
        return inputs, targets
