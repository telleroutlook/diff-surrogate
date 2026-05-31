"""Geometry-aware neural operator using SDF-encoded trunk network.

GINOT-style architecture (Geometry-Informed Neural Operator with Trunk):
  - Trunk network: encodes geometry via SDF values at query points -> basis functions
  - Branch network: encodes physics parameters -> coefficients
  - Output: trunk_basis @ branch_coefficients

This decouples geometry representation from physics, enabling generalization
across non-rectangular domains without FFT constraints.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor

from .base import CorrectionPolicy, SurrogateBase


class _TrunkNet(nn.Module):
    """Encode SDF geometry into basis functions.

    Takes SDF values at spatial points and outputs a set of basis functions,
    one per n_basis dimension. Each basis function captures a spatial mode
    conditioned on the local geometry.

    Args:
        sdf_dim: Number of SDF-related input features per point (typically 1).
        hidden_dim: Hidden layer width.
        n_basis: Number of output basis functions.
        n_layers: Number of hidden layers.
    """

    def __init__(
        self, sdf_dim: int = 1, hidden_dim: int = 128,
        n_basis: int = 64, n_layers: int = 4,
    ):
        super().__init__()
        layers = [nn.Linear(sdf_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, n_basis))
        self.net = nn.Sequential(*layers)

    def forward(self, sdf: Tensor) -> Tensor:
        """Compute basis functions from SDF field.

        Args:
            sdf: (B, H, W) or (B, N_points) SDF values.

        Returns:
            (B, H, W, n_basis) or (B, N_points, n_basis) basis functions.
        """
        if sdf.ndim == 4:
            sdf = sdf.permute(0, 2, 3, 1)  # (B,1,H,W) -> (B,H,W,1)
        elif sdf.ndim == 3:
            sdf = sdf.unsqueeze(-1)  # (B,H,W) -> (B,H,W,1)
        elif sdf.ndim == 2:
            sdf = sdf.unsqueeze(-1)  # (B,N) -> (B,N,1)
        return self.net(sdf)


class _BranchNet(nn.Module):
    """Encode physics parameters into coefficient vectors.

    Takes a vector of physics parameters (Re, BC values, etc.) and outputs
    one coefficient per basis function.

    Args:
        param_dim: Dimensionality of physics parameter vector.
        hidden_dim: Hidden layer width.
        n_basis: Number of output coefficients (must match trunk n_basis).
        n_layers: Number of hidden layers.
    """

    def __init__(self, param_dim: int, hidden_dim: int = 128, n_basis: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(param_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, n_basis))
        self.net = nn.Sequential(*layers)

    def forward(self, params: Tensor) -> Tensor:
        """Compute coefficients from physics parameters.

        Args:
            params: (B, param_dim) physics parameters.

        Returns:
            (B, n_basis) coefficient vectors.
        """
        return self.net(params)


class SDFTrunkSurrogate(SurrogateBase):
    """Geometry-aware neural operator using SDF encoding.

    Combines a trunk network (SDF -> basis functions) with a branch network
    (physics params -> coefficients) in a DeepONet-like structure. The output
    is the inner product of basis functions and coefficients at each spatial
    point, allowing the model to generalize across different geometries without
    requiring a regular grid.

    Args:
        param_dim: Dimension of physics parameter vector (e.g., Re, inlet_velocity).
        n_outputs: Number of output fields (e.g., 3 for ux, uy, p).
        hidden_dim: Hidden layer width for both trunk and branch networks.
        n_basis: Number of basis functions / latent dimension.
        sdf_channels: Number of SDF-derived input features per point.
        correction_policy: When to call true solver for correction.
        device: Compute device.
        data_generator: Optional callable for synthetic training data.
    """

    def __init__(
        self,
        param_dim: int = 4,
        n_outputs: int = 3,
        hidden_dim: int = 128,
        n_basis: int = 64,
        sdf_channels: int = 1,
        correction_policy: CorrectionPolicy | None = None,
        device: str | torch.device | int = "cpu",
        data_generator: Callable | None = None,
    ):
        self.param_dim = param_dim
        self.n_outputs = n_outputs
        self.hidden_dim = hidden_dim
        self.n_basis = n_basis
        self.sdf_channels = sdf_channels
        self._data_generator = data_generator
        super().__init__(correction_policy=correction_policy, device=device)

    def _build_network(self) -> nn.ModuleDict:
        trunk = _TrunkNet(
            sdf_dim=self.sdf_channels,
            hidden_dim=self.hidden_dim,
            n_basis=self.n_basis * self.n_outputs,
        )
        branch = _BranchNet(
            param_dim=self.param_dim,
            hidden_dim=self.hidden_dim,
            n_basis=self.n_basis * self.n_outputs,
        )
        return nn.ModuleDict({"trunk": trunk, "branch": branch})

    def forward(self, x: tuple[Tensor, Tensor] | Tensor) -> Tensor:
        """Predict output fields from SDF geometry and physics parameters.

        Accepts either a tuple of (sdf_field, physics_params) or a single
        tensor (interpreted as SDF field with zero physics params).

        Args:
            x: Either a tuple (sdf_field, physics_params) or just sdf_field.

        Returns:
            (B, n_outputs, H, W) predicted fields.
        """
        if isinstance(x, tuple):
            sdf_field, physics_params = x
        else:
            sdf_field = x
            physics_params = torch.zeros(
                sdf_field.shape[0], self.param_dim, device=sdf_field.device
            )

        net = self.get_network()
        trunk_net: _TrunkNet = net["trunk"]  # type: ignore[assignment]
        branch_net: _BranchNet = net["branch"]  # type: ignore[assignment]

        # basis: (B, H, W, n_basis*n_outputs) or (B, N, n_basis*n_outputs)
        basis = trunk_net(sdf_field)
        # coeffs: (B, n_basis*n_outputs)
        coeffs = branch_net(physics_params)

        # Reshape for per-output decomposition
        spatial_shape = basis.shape[:-1]
        basis = basis.reshape(*spatial_shape, self.n_outputs, self.n_basis)
        coeffs = coeffs.reshape(-1, self.n_outputs, self.n_basis)

        # Inner product: sum over n_basis dimension
        # basis: (B, S..., n_outputs, n_basis), coeffs: (B, n_outputs, n_basis)
        # output[b, s..., o] = sum_k basis[b, s..., o, k] * coeffs[b, o, k]
        output = (basis * coeffs.unsqueeze(1).unsqueeze(1)).sum(dim=-1)

        # Reshape to (B, n_outputs, H, W) or (B, n_outputs, N)
        output = (
            output.permute(0, -1, 1, 2)
            if sdf_field.ndim >= 3
            else output.permute(0, 2, 1)
        )

        return output

    def predict(self, x: tuple[Tensor, Tensor] | Tensor) -> Tensor:
        self.stats.total_predictions += 1
        was_training = self.training
        self.eval()
        with torch.no_grad():
            if isinstance(x, tuple):
                result = self(
                    (x[0].to(self.device), x[1].to(self.device))
                )
            else:
                result = self(x.to(self.device))
        if was_training:
            self.train()
        return result

    def generate_training_data(self, n_samples: int) -> tuple[tuple[Tensor, Tensor], Tensor]:
        if self._data_generator is not None:
            return self._data_generator(n_samples)
        sdf = torch.randn(n_samples, 32, 32)
        params = torch.randn(n_samples, self.param_dim)
        targets = torch.randn(n_samples, self.n_outputs, 32, 32)
        return (sdf, params), targets
