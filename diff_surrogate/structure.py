"""Structure-preserving operators for physics-informed surrogate models.

Provides discrete differential operators, divergence-conserving projections,
and flux-conserving linear layers that embed conservation-law inductive biases
into neural network architectures.

The core idea: given a neural network that outputs a velocity/flux field,
we can project its output onto the manifold of divergence-free fields
(analogous to the Chorin pressure-correction step in CFD), ensuring that
the learned surrogate respects incompressibility or other flux constraints
by construction.

**Stencil convention.**  We use *forward* differences (replication-padded on
the far side) for the gradient and *backward* differences (zero-padded on the
near side) for the divergence.  This guarantees the discrete adjoint
relationship ``discrete_divergence(discrete_gradient(f)) == _laplacian_2d(f)``
exactly, which is essential for the projection step to work.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "ConservationLoss",
    "DivergenceConservingProjection",
    "FluxConservingLinear",
    "StructurePreservingEncoder",
    "discrete_divergence",
    "discrete_gradient",
]


# ---------------------------------------------------------------------------
# Low-level finite-difference stencils
# ---------------------------------------------------------------------------


def _fwd_dy(x: Tensor, h: float = 1.0) -> Tensor:
    """Forward difference along y (dim -2): ``(f[i+1] - f[i]) / h``.

    Replication-pads the far (bottom) boundary so the output has the same
    shape as the input.
    """
    p = F.pad(x, (0, 0, 0, 1), mode="replicate")  # (..., H+1, W)
    return (p[..., 1:, :] - p[..., :-1, :]) / h


def _fwd_dx(x: Tensor, h: float = 1.0) -> Tensor:
    """Forward difference along x (dim -1): ``(f[j+1] - f[j]) / h``.

    Replication-pads the far (right) boundary.
    """
    p = F.pad(x, (0, 1), mode="replicate")  # (..., H, W+1)
    return (p[..., 1:] - p[..., :-1]) / h


def _bwd_dy(x: Tensor, h: float = 1.0) -> Tensor:
    """Backward difference along y: ``(f[i] - f[i-1]) / h``.

    Zero-pads the near (top) boundary.
    """
    p = F.pad(x, (0, 0, 1, 0), mode="constant", value=0)  # (..., H+1, W)
    return (p[..., 1:, :] - p[..., :-1, :]) / h


def _bwd_dx(x: Tensor, h: float = 1.0) -> Tensor:
    """Backward difference along x: ``(f[j] - f[j-1]) / h``.

    Zero-pads the near (left) boundary.
    """
    p = F.pad(x, (1, 0), mode="constant", value=0)  # (..., H, W+1)
    return (p[..., 1:] - p[..., :-1]) / h


# ---------------------------------------------------------------------------
# Public discrete differential operators
# ---------------------------------------------------------------------------


def discrete_gradient(scalar_field: Tensor, grid_spacing: float | Tensor = 1.0) -> Tensor:
    """Discrete gradient of a scalar field on a regular 2-D grid.

    Uses forward differences with replication padding at the far boundary.

    Args:
        scalar_field: ``(..., H, W)`` scalar field.
        grid_spacing: Physical spacing (scalar or ``(h_y, h_x)``).

    Returns:
        ``(..., H, W, 2)`` gradient vector ``(du/dy, du/dx)``.
    """
    if scalar_field.ndim < 2:
        raise ValueError(f"scalar_field must have >= 2 dims (got {scalar_field.ndim})")

    if isinstance(grid_spacing, (int, float)):
        hy = hx = float(grid_spacing)
    else:
        hs = torch.as_tensor(grid_spacing, dtype=scalar_field.dtype, device=scalar_field.device)
        hy, hx = float(hs[0]), float(hs[1])

    du_dy = _fwd_dy(scalar_field, hy)
    du_dx = _fwd_dx(scalar_field, hx)
    return torch.stack([du_dy, du_dx], dim=-1)


def discrete_divergence(field: Tensor, grid_spacing: float | Tensor = 1.0) -> Tensor:
    """Discrete divergence of a 2-D vector field on a regular grid.

    Uses backward differences with zero padding at the near boundary.
    This is the exact discrete adjoint of :func:`discrete_gradient`, so
    ``discrete_divergence(discrete_gradient(f)) == _laplacian_2d(f)`` holds
    exactly.

    Args:
        field: ``(..., H, W, 2)`` vector field ``(v_y, v_x)``.
        grid_spacing: Physical spacing (scalar or ``(h_y, h_x)``).

    Returns:
        ``(..., H, W)`` scalar divergence.
    """
    if field.shape[-1] != 2:
        raise ValueError(f"Last dim must be 2 (got {field.shape[-1]})")

    if isinstance(grid_spacing, (int, float)):
        hy = hx = float(grid_spacing)
    else:
        hs = torch.as_tensor(grid_spacing, dtype=field.dtype, device=field.device)
        hy, hx = float(hs[0]), float(hs[1])

    dvy_dy = _bwd_dy(field[..., 0], hy)
    dvx_dx = _bwd_dx(field[..., 1], hx)
    return dvy_dy + dvx_dx


def _laplacian_2d(x: Tensor) -> Tensor:
    """5-point discrete Laplacian with Neumann BCs (replication padding).

    Equivalent to ``discrete_divergence(discrete_gradient(x))`` by construction.

    Args:
        x: ``(B, H, W)`` scalar field.

    Returns:
        ``(B, H, W)`` Laplacian.
    """
    p = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return (
        p[:, 1:-1, 2:]  # right
        + p[:, 1:-1, :-2]  # left
        + p[:, 2:, 1:-1]  # down
        + p[:, :-2, 1:-1]  # up
        - 4.0 * x
    )


# ---------------------------------------------------------------------------
# Divergence-conserving projection
# ---------------------------------------------------------------------------


class DivergenceConservingProjection(nn.Module):
    """Project a vector field onto a divergence-free (or prescribed divergence) space.

    Implements the Chorin-style pressure-correction projection:

    1. Compute discrete divergence of the input field.
    2. Solve the Poisson equation ``Laplacian(phi) = div(field) - div_target``
       for a scalar potential phi via Jacobi iteration.
    3. Apply correction: ``field_corrected = field - grad(phi)``

    The result satisfies ``div(field_corrected) ≈ div_target`` to the
    precision determined by the number of Jacobi iterations.

    Args:
        method: ``'direct'`` uses more Jacobi iterations for a tighter solve.
            ``'iterative'`` uses fewer iterations (faster, less accurate).
        max_iter: Maximum Jacobi iterations.
        tol: Unused (kept for API compatibility).
    """

    def __init__(
        self,
        method: str = "direct",
        max_iter: int = 100,
        tol: float = 1e-8,
    ):
        super().__init__()
        if method not in ("direct", "iterative"):
            raise ValueError(f"method must be 'direct' or 'iterative', got '{method}'")
        self.method = method
        self.max_iter = max_iter if method == "iterative" else max(max_iter * 3, 300)
        self.tol = tol

    def forward(self, field: Tensor, div_target: Tensor | None = None) -> Tensor:
        """Project field onto divergence-free (or prescribed divergence) space.

        Args:
            field: ``(B, C, H, W)`` or ``(B, H, W, C)`` grid field (C=2),
                or ``(B, N, C)`` unstructured field.
            div_target: Optional target divergence.  ``None`` means zero.

        Returns:
            Corrected field with same shape as input.
        """
        if field.ndim == 3:
            return self._project_unstructured(field, div_target)
        if field.ndim == 4:
            return self._project_grid(field, div_target)
        raise ValueError(f"field must be 3-D or 4-D, got {field.ndim}-D")

    @staticmethod
    def _is_chw(f: Tensor) -> bool:
        d1, d2, d3 = f.shape[1], f.shape[2], f.shape[3]
        return d1 <= 4 and d1 < d2 and d1 < d3

    def _project_grid(self, field: Tensor, div_target: Tensor | None) -> Tensor:
        is_chw = self._is_chw(field)
        field_hwc = field.permute(0, 2, 3, 1) if is_chw else field

        if field_hwc.shape[-1] != 2:
            raise ValueError(
                f"Need 2 vector components for 2-D grid projection, got {field_hwc.shape[-1]}"
            )

        div_current = discrete_divergence(field_hwc)

        if div_target is None:
            rhs = div_current
        else:
            dt = div_target.unsqueeze(0) if div_target.ndim == 2 else div_target
            rhs = div_current - dt

        phi = self._solve_poisson_jacobi(rhs)
        grad_phi = discrete_gradient(phi)
        corrected = field_hwc - grad_phi

        return corrected.permute(0, 3, 1, 2) if is_chw else corrected

    def _solve_poisson_jacobi(self, rhs: Tensor) -> Tensor:
        """Solve ``Laplacian(phi) = rhs`` via Jacobi iteration.

        The update rule is:
            phi_new[i,j] = (phi[i-1,j] + phi[i+1,j] + phi[i,j-1] + phi[i,j+1]
                            - rhs[i,j]) / 4
        with Neumann BCs (replication padding).
        """
        phi = torch.zeros_like(rhs)

        for _ in range(self.max_iter):
            p = F.pad(phi, (1, 1, 1, 1), mode="replicate")
            phi = (p[:, 1:-1, 2:] + p[:, 1:-1, :-2] + p[:, 2:, 1:-1] + p[:, :-2, 1:-1] - rhs) / 4.0

        return phi

    def _project_unstructured(self, field: Tensor, div_target: Tensor | None) -> Tensor:
        """Project ``(B, N, C)`` unstructured field by mean-subtraction."""
        mean_per_comp = field.mean(dim=1, keepdim=True)

        if div_target is None:
            return field - mean_per_comp
        else:
            dt = div_target.unsqueeze(0).unsqueeze(0) if div_target.ndim == 1 else div_target
            return field - mean_per_comp + dt.expand_as(mean_per_comp)


# ---------------------------------------------------------------------------
# Flux-conserving linear layer
# ---------------------------------------------------------------------------


class FluxConservingLinear(nn.Module):
    """Linear transformation that preserves total flux (sum of outputs).

    After a standard linear transformation, the output is rescaled so that
    the total (volume-weighted) flux matches the input flux.

    Args:
        in_features: Number of input features per node.
        out_features: Number of output features per node.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: Tensor, node_volumes: Tensor | None = None) -> Tensor:
        squeeze = False
        if x.ndim == 2:
            x = x.unsqueeze(0)
            squeeze = True

        _B, N, _ = x.shape
        y = self.linear(x)

        if node_volumes is None:
            w = x.new_ones(N)
        elif node_volumes.ndim == 1:
            w = node_volumes
        else:
            w = node_volumes

        w3d = w.view(1, N, 1) if w.ndim == 1 else w.unsqueeze(-1)

        input_flux = (x * w3d).sum(dim=1)
        output_flux = (y * w3d).sum(dim=1)

        input_total = input_flux.sum(dim=-1, keepdim=True)
        output_total = output_flux.sum(dim=-1, keepdim=True)

        sign = output_total.sign()
        sign[sign == 0] = 1.0
        scale = input_total / (output_total.abs() + 1e-12) * sign

        y_corrected = y * scale.unsqueeze(1)
        if squeeze:
            y_corrected = y_corrected.squeeze(0)
        return y_corrected


# ---------------------------------------------------------------------------
# Conservation loss
# ---------------------------------------------------------------------------


class ConservationLoss(nn.Module):
    """Loss measuring conservation-law violation (mean squared divergence)."""

    def forward(
        self,
        field: Tensor,
        grid_spacing: float | Tensor = 1.0,
    ) -> Tensor:
        """Compute conservation-law violation loss.

        Args:
            field: ``(B, C, H, W)`` or ``(B, H, W, C)`` grid field, or
                ``(B, N, C)`` unstructured field.
            grid_spacing: Physical spacing between grid points.

        Returns:
            Scalar loss: mean squared divergence.
        """
        if field.ndim == 4:
            d1, d2, d3 = field.shape[1], field.shape[2], field.shape[3]
            field_hwc = field.permute(0, 2, 3, 1) if d1 <= 4 and d1 < d2 and d1 < d3 else field
            div = discrete_divergence(field_hwc, grid_spacing=grid_spacing)
            return (div**2).mean()
        elif field.ndim == 3:
            mean_val = field.mean(dim=1)
            return (mean_val**2).mean()
        else:
            raise ValueError(f"field must be 3-D or 4-D, got {field.ndim}-D")


# ---------------------------------------------------------------------------
# Structure-preserving encoder
# ---------------------------------------------------------------------------


class StructurePreservingEncoder(nn.Module):
    """IrregularMeshEncoder wrapped with a divergence-conserving projection.

    Args:
        in_dim: Dimension of optional per-point input features.
        embed_dim: Output embedding dimension.
        scales: Tuple of K values for each neighborhood scale.
        projection_method: Solver method for the projection step.
    """

    def __init__(
        self,
        in_dim: int = 0,
        embed_dim: int = 128,
        scales: tuple[int, ...] = (8, 16, 32),
        projection_method: str = "direct",
    ):
        super().__init__()
        from .geometry.pointcloud import IrregularMeshEncoder

        self.encoder = IrregularMeshEncoder(in_dim=in_dim, embed_dim=embed_dim, scales=scales)
        self.projection = DivergenceConservingProjection(method=projection_method)
        self.embed_dim = embed_dim

    def forward(self, points: Tensor, features: Tensor | None = None) -> Tensor:
        output = self.encoder(points, features)

        if output.ndim == 1:
            output = output.unsqueeze(0).unsqueeze(0)
            corrected = self.projection(output)
            return corrected.squeeze(0).squeeze(0)
        else:
            output = output.unsqueeze(1)
            corrected = self.projection(output)
            return corrected.squeeze(1)
