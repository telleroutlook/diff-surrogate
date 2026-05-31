"""Cross-attention geometry operator for geometry-aware surrogate models.

GINOT-inspired architecture where geometry features (from SDF encoding) serve
as keys/values and physics-parameter features serve as queries. This enables
the model to selectively attend to geometrically relevant regions when
predicting each output field, rather than the simple inner product used by
SDFTrunkSurrogate.

Architecture:
  - Geometry encoder: SDF → spatial feature maps (keys, values)
  - Physics encoder: parameters → query vectors
  - Cross-attention: physics queries attend to geometry key/value pairs
  - Output head: attended features → per-field predictions

The cross-attention mechanism allows the surrogate to learn which parts of the
geometry matter most for each output quantity (e.g., wake region for drag,
near-wall region for heat transfer).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor

from .base import CorrectionPolicy, SurrogateBase


class _GeometryEncoder(nn.Module):
    """Encode SDF field into spatial feature maps for cross-attention.

    Produces keys and values from the geometry (SDF) field. Each spatial
    location gets a feature vector that captures the local geometric context.

    Args:
        sdf_dim: Number of SDF-derived input channels.
        hidden_dim: Feature dimension for keys and values.
        n_layers: Number of convolutional layers.
    """

    def __init__(self, sdf_dim: int = 1, hidden_dim: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [nn.Conv2d(sdf_dim, hidden_dim, 3, padding=1), nn.GELU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.GELU()])
        self.net = nn.Sequential(*layers)
        self.key_proj = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.value_proj = nn.Conv2d(hidden_dim, hidden_dim, 1)

    def forward(self, sdf: Tensor) -> tuple[Tensor, Tensor]:
        """Compute keys and values from SDF field.

        Args:
            sdf: (B, H, W) or (B, 1, H, W) SDF values.

        Returns:
            keys: (B, hidden_dim, H, W)
            values: (B, hidden_dim, H, W)
        """
        if sdf.ndim == 3:
            sdf = sdf.unsqueeze(1)  # (B,1,H,W)
        feat = self.net(sdf)
        return self.key_proj(feat), self.value_proj(feat)


class _PhysicsEncoder(nn.Module):
    """Encode physics parameters into query vectors for cross-attention.

    Produces query vectors that attend to geometry features. Multiple output
    fields get separate queries so each can focus on different geometric regions.

    Args:
        param_dim: Dimensionality of physics parameter vector.
        hidden_dim: Must match geometry encoder hidden_dim.
        n_outputs: Number of output fields (one query per output).
    """

    def __init__(self, param_dim: int, hidden_dim: int = 64, n_outputs: int = 3):
        super().__init__()
        self.n_outputs = n_outputs
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(param_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_outputs * hidden_dim),
        )

    def forward(self, params: Tensor) -> Tensor:
        """Compute query vectors from physics parameters.

        Args:
            params: (B, param_dim) physics parameters.

        Returns:
            queries: (B, n_outputs, hidden_dim)
        """
        return self.net(params).reshape(-1, self.n_outputs, self.hidden_dim)


class _CrossAttention(nn.Module):
    """Multi-head cross-attention between physics queries and geometry KV pairs.

    For each output field, the query attends over all spatial locations of the
    geometry feature map, producing a geometry-aware feature vector.

    Args:
        hidden_dim: Dimension of keys, values, and queries.
        n_heads: Number of attention heads.
    """

    def __init__(self, hidden_dim: int = 64, n_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
        """Compute cross-attention.

        Args:
            queries: (B, n_outputs, hidden_dim)
            keys: (B, hidden_dim, H, W)
            values: (B, hidden_dim, H, W)

        Returns:
            (B, n_outputs, hidden_dim) attended features.
        """
        B, n_out, d = queries.shape
        _, _, H, W = keys.shape

        # Reshape keys/values: (B, n_heads, head_dim, H*W)
        k = keys.reshape(B, self.n_heads, self.head_dim, H * W)
        v = values.reshape(B, self.n_heads, self.head_dim, H * W)
        # Reshape queries: (B, n_heads, n_out, head_dim)
        q = queries.reshape(B, self.n_heads, n_out, self.head_dim)

        # Scaled dot-product: (B, n_heads, n_out, H*W)
        scale = self.head_dim**0.5
        attn = torch.matmul(q, k) / scale
        attn = torch.softmax(attn, dim=-1)

        # Weighted sum: (B, n_heads, n_out, head_dim)
        out = torch.matmul(attn, v.transpose(-2, -1))
        out = out.reshape(B, n_out, d)

        out = self.out_proj(out)
        out = self.norm(out + queries)
        return out


class CrossAttnSurrogate(SurrogateBase):
    """Geometry-aware surrogate using cross-attention between physics and geometry.

    Unlike SDFTrunkSurrogate (which uses an inner product of trunk/branch
    features), this model uses multi-head cross-attention so each output field
    can selectively attend to the geometric regions most relevant to it.

    Args:
        param_dim: Dimension of physics parameter vector.
        n_outputs: Number of output fields.
        hidden_dim: Feature dimension for encoders and attention.
        n_heads: Number of attention heads.
        sdf_channels: Number of SDF-derived input channels.
        n_geom_layers: Depth of geometry encoder CNN.
        correction_policy: When to call true solver for correction.
        device: Compute device.
        data_generator: Optional callable for synthetic training data.
    """

    def __init__(
        self,
        param_dim: int = 4,
        n_outputs: int = 3,
        hidden_dim: int = 64,
        n_heads: int = 4,
        sdf_channels: int = 1,
        n_geom_layers: int = 3,
        correction_policy: CorrectionPolicy | None = None,
        device: str | torch.device | int = "cpu",
        data_generator: Callable | None = None,
    ):
        self.param_dim = param_dim
        self.n_outputs = n_outputs
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.sdf_channels = sdf_channels
        self.n_geom_layers = n_geom_layers
        self._data_generator = data_generator
        super().__init__(correction_policy=correction_policy, device=device)

    def _build_network(self) -> nn.ModuleDict:
        geom_enc = _GeometryEncoder(
            sdf_dim=self.sdf_channels,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_geom_layers,
        )
        phys_enc = _PhysicsEncoder(
            param_dim=self.param_dim,
            hidden_dim=self.hidden_dim,
            n_outputs=self.n_outputs,
        )
        cross_attn = _CrossAttention(
            hidden_dim=self.hidden_dim,
            n_heads=self.n_heads,
        )
        # Output head: per-output scalar field from attended features + spatial basis
        spatial_head = nn.Conv2d(self.hidden_dim * 2, self.n_outputs, 1)
        return nn.ModuleDict(
            {
                "geom_enc": geom_enc,
                "phys_enc": phys_enc,
                "cross_attn": cross_attn,
                "spatial_head": spatial_head,
            }
        )

    def forward(self, x: tuple[Tensor, Tensor] | Tensor) -> Tensor:
        """Predict output fields from SDF geometry and physics parameters.

        Args:
            x: Either (sdf_field, physics_params) tuple or just sdf_field.

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
        geom_enc: _GeometryEncoder = net["geom_enc"]  # type: ignore[assignment]
        phys_enc: _PhysicsEncoder = net["phys_enc"]  # type: ignore[assignment]
        cross_attn: _CrossAttention = net["cross_attn"]  # type: ignore[assignment]
        spatial_head: nn.Conv2d = net["spatial_head"]  # type: ignore[assignment]

        # Geometry → keys, values: (B, hidden_dim, H, W)
        keys, values = geom_enc(sdf_field)
        # Physics → queries: (B, n_outputs, hidden_dim)
        queries = phys_enc(physics_params)

        # Cross-attention: (B, n_outputs, hidden_dim)
        attended = cross_attn(queries, keys, values)

        # Broadcast attended features to spatial map and combine with geometry
        # attended: (B, n_outputs, hidden_dim) → (B, n_outputs*hidden_dim, 1, 1)
        B = sdf_field.shape[0]
        H, W = sdf_field.shape[-2], sdf_field.shape[-1]
        attended_spatial = attended.reshape(B, -1, 1, 1).expand(-1, -1, H, W)

        # Combine with geometry features: cat along channel dim
        combined = torch.cat([keys, attended_spatial[:, : self.hidden_dim]], dim=1)
        output = spatial_head(combined)
        return output

    def predict(self, x: tuple[Tensor, Tensor] | Tensor) -> Tensor:
        self.stats.total_predictions += 1
        was_training = self.training
        self.eval()
        with torch.no_grad():
            if isinstance(x, tuple):
                result = self((x[0].to(self.device), x[1].to(self.device)))
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
