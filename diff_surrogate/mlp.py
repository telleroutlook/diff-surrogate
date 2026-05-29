"""MLP-based surrogate for scalar property prediction with physics constraints."""

from __future__ import annotations

import torch
import torch.nn as nn

from typing import Callable

from .base import SurrogateBase, CorrectionPolicy


class MonotoneMLP(nn.Module):
    """MLP with monotonicity constraint via positive weights."""

    def __init__(self, in_features: int, hidden: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(in_features, hidden), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            if isinstance(module, nn.Linear):
                x = x @ torch.abs(module.weight).T + module.bias
            else:
                x = module(x)
        return x


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
        constrained: dict[str, str] | None = None,
        correction_policy: CorrectionPolicy | None = None,
        device: str = "cpu",
        data_generator: Callable | None = None,
    ):
        self.n_inputs = n_inputs
        self.properties = properties or ["value"]
        self.hidden = hidden
        self.n_layers = n_layers
        self.constrained = constrained or {}
        self._data_generator = data_generator
        super().__init__(correction_policy=correction_policy, device=device)

    def _build_network(self) -> nn.ModuleDict:
        nets = {}
        for prop in self.properties:
            constraint = self.constrained.get(prop, "none")
            if constraint == "monotone":
                nets[prop] = MonotoneMLP(self.n_inputs, self.hidden, self.n_layers)
            elif constraint == "positive":
                nets[prop] = PositiveOutputMLP(self.n_inputs, self.hidden, self.n_layers)
            else:
                nets[prop] = nn.Sequential(
                    nn.Linear(self.n_inputs, self.hidden), nn.ReLU(),
                    nn.Linear(self.hidden, self.hidden), nn.ReLU(),
                    nn.Linear(self.hidden, 1),
                )
        return nn.ModuleDict(nets)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        net = self.get_network()
        return {prop: module(x).squeeze(-1) for prop, module in net.items()}

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.stats.total_predictions += 1
        with torch.no_grad():
            return self.forward(x.to(self.device))

    def generate_training_data(self, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._data_generator is not None:
            return self._data_generator(n_samples)
        inputs = torch.randn(n_samples, self.n_inputs)
        targets = {prop: torch.randn(n_samples) for prop in self.properties}
        return inputs, targets

    def train_surrogate(
        self,
        n_samples: int = 256,
        n_epochs: int = 10,
        lr: float = 1e-3,
        device: str | None = None,
    ) -> list[float]:
        dev = device or self.device
        inputs, targets = self.generate_training_data(n_samples)
        inputs = inputs.to(dev)
        if isinstance(targets, dict):
            targets = {k: v.to(dev) for k, v in targets.items()}
        else:
            targets = targets.to(dev)
        net = self.get_network()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        losses = []
        for _ in range(n_epochs):
            optimizer.zero_grad()
            output = self.forward(inputs)
            loss = sum(
                torch.nn.functional.mse_loss(output[k], targets[k]) for k in targets
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            self.stats.train_losses.append(loss.item())
        self._trained = True
        return losses
