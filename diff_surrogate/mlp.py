"""MLP-based surrogate for scalar property prediction with physics constraints."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from .base import CorrectionPolicy, SurrogateBase, _build_dataloader


class MonotoneLinear(nn.Linear):
    """Linear layer with monotonicity constraint via absolute weights."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.abs().T + self.bias


class MonotoneMLP(nn.Module):
    """MLP with monotonicity constraint via positive weights.

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
        signed = x * self._signs
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
        constrained: dict[str, str] | None = None,
        correction_policy: CorrectionPolicy | None = None,
        device: str = "cpu",
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
            constraint = self.constrained.get(prop, "none")
            if constraint == "monotone":
                nets[prop] = MonotoneMLP(self.n_inputs, self.hidden, self.n_layers)
            elif constraint == "positive":
                nets[prop] = PositiveOutputMLP(self.n_inputs, self.hidden, self.n_layers)
            else:
                layers = [nn.Linear(self.n_inputs, self.hidden), nn.ReLU()]
                for _ in range(self.n_layers - 2):
                    layers.extend([nn.Linear(self.hidden, self.hidden), nn.ReLU()])
                layers.append(nn.Linear(self.hidden, 1))
                nets[prop] = nn.Sequential(*layers)
        return nn.ModuleDict(nets)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        net = self.get_network()
        if self.shared_trunk:
            features = net["trunk"](x)
            return {prop: net[f"head_{prop}"](features).squeeze(-1) for prop in self.properties}
        return {prop: module(x).squeeze(-1) for prop, module in net.items()}

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.stats.total_predictions += 1
        was_training = self.training
        self.eval()
        with torch.no_grad():
            result = self(x.to(self.device))
        if was_training:
            self.train()
        return result

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
        batch_size: int = 32,
        device: str | None = None,
    ) -> list[float]:
        dev = device or self.device
        inputs, targets = self.generate_training_data(n_samples)
        inputs = inputs.to(dev)
        if isinstance(targets, dict):
            targets = {k: v.to(dev) for k, v in targets.items()}

        net = self.get_network()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        loader, target_keys = _build_dataloader(inputs, targets, batch_size)

        losses = []
        for _ in range(n_epochs):
            epoch_loss = 0.0
            for batch in loader:
                batch_x = batch[0]
                optimizer.zero_grad()
                output = self(batch_x)
                if target_keys is not None:
                    loss = sum(
                        torch.nn.functional.mse_loss(output[k], batch[i + 1])
                        for i, k in enumerate(target_keys)
                    )
                else:
                    loss = torch.nn.functional.mse_loss(output, batch[1])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg = epoch_loss / len(loader)
            losses.append(avg)
            self.stats.train_losses.append(avg)
        self._trained = True
        return losses
