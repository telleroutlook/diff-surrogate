"""CNN-based surrogate for 2D field prediction (velocity, pressure, aerial image, etc.)."""

from __future__ import annotations

import torch
import torch.nn as nn

from typing import Callable

from .base import SurrogateBase, CorrectionPolicy


class _CNNFieldNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden: int = 32, n_layers: int = 4):
        super().__init__()
        layers = [nn.Conv2d(in_channels, hidden, 3, padding=1), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU()])
        layers.append(nn.Conv2d(hidden, out_channels, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNSurrogate(SurrogateBase):
    """CNN surrogate for 2D field-to-field prediction.

    Args:
        in_channels: Number of input channels (e.g., 1 for density field).
        out_channels: Number of output channels (e.g., 3 for ux, uy, p).
        hidden: Hidden channel width.
        n_layers: Total convolutional layers.
        grid_size: Spatial resolution (H=W).
        correction_policy: When to call true solver.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        hidden: int = 32,
        n_layers: int = 4,
        grid_size: int = 64,
        correction_policy: CorrectionPolicy | None = None,
        device: str = "cpu",
        data_generator: Callable | None = None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden = hidden
        self.n_layers = n_layers
        self.grid_size = grid_size
        self._data_generator = data_generator
        super().__init__(correction_policy=correction_policy, device=device)

    def _build_network(self) -> nn.Module:
        return _CNNFieldNet(self.in_channels, self.out_channels, self.hidden, self.n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_network()(x)

    def generate_training_data(self, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._data_generator is not None:
            return self._data_generator(n_samples)
        inputs = torch.rand(n_samples, self.in_channels, self.grid_size, self.grid_size)
        targets = torch.randn(n_samples, self.out_channels, self.grid_size, self.grid_size)
        return inputs, targets
