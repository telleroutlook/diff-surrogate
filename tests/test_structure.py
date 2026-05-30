"""Tests for structure-preserving operators (divergence projection, flux conservation)."""

from __future__ import annotations

import torch
import torch.nn as nn

from diff_surrogate.structure import (
    ConservationLoss,
    DivergenceConservingProjection,
    FluxConservingLinear,
    StructurePreservingEncoder,
    _laplacian_2d,
    discrete_divergence,
    discrete_gradient,
)


# ---------------------------------------------------------------------------
# Discrete differential operator tests
# ---------------------------------------------------------------------------


def test_discrete_divergence_zero_for_constant_field():
    """A constant vector field has zero divergence in the interior.

    The backward-difference divergence uses zero-padding at the near boundary,
    so the first row/column may show non-zero values.  Interior nodes are
    exactly zero for a constant field.
    """
    B, H, W = 2, 16, 16
    field = torch.zeros(B, H, W, 2)
    field[..., 0] = 3.0
    field[..., 1] = -1.5

    div = discrete_divergence(field)
    assert div.shape == (B, H, W)
    # Interior should be exactly zero
    interior = div[:, 1:, 1:]
    assert torch.allclose(interior, torch.zeros_like(interior), atol=1e-6), (
        f"Constant field interior divergence should be ~0, got max={interior.abs().max().item()}"
    )


def test_discrete_divergence_analytical():
    """Compare discrete divergence of a known field against the analytical value.

    Field: u(x,y) = (y^2, x^2)  =>  div(u) = 2y + 2x
    The backward-difference divergence is first-order accurate (O(h)),
    so with h=0.1 we expect ~0.1 error on interior nodes.
    """
    H, W = 20, 20
    h = 0.1
    xs = torch.arange(W, dtype=torch.float64) * h
    ys = torch.arange(H, dtype=torch.float64) * h
    Y, X = torch.meshgrid(ys, xs, indexing="ij")

    # vy = y^2, vx = x^2
    field = torch.zeros(1, H, W, 2, dtype=torch.float64)
    field[0, ..., 0] = Y**2
    field[0, ..., 1] = X**2

    div = discrete_divergence(field, grid_spacing=h)
    analytical = 2.0 * Y + 2.0 * X

    # Interior: backward diff gives O(h) error for quadratic fields
    interior = (slice(None), slice(2, -2), slice(2, -2))
    diff = (div[interior] - analytical[None, 2:-2, 2:-2]).abs()
    assert diff.max() < 2 * h + 1e-10, (
        f"Divergence mismatch: max error = {diff.max().item():.6f}, expected ~{2*h}"
    )


def test_adjoint_consistency():
    """discrete_divergence(discrete_gradient(f)) == _laplacian_2d(f) exactly."""
    f = torch.randn(2, 8, 8)
    g = discrete_gradient(f)
    div_g = discrete_divergence(g)
    lap = _laplacian_2d(f)
    assert torch.allclose(div_g, lap, atol=1e-5), (
        f"Adjoint consistency violated: max diff = {(div_g - lap).abs().max().item():.6e}"
    )


# ---------------------------------------------------------------------------
# Projection tests
# ---------------------------------------------------------------------------


def test_projection_reduces_divergence():
    """After projection, divergence should be significantly smaller."""
    B, C, H, W = 2, 2, 12, 12
    torch.manual_seed(123)

    field = torch.randn(B, C, H, W)

    proj = DivergenceConservingProjection(method="direct", max_iter=200)
    corrected = proj(field)

    loss_fn = ConservationLoss()
    div_before = loss_fn(field).item()
    div_after = loss_fn(corrected).item()

    assert div_after < div_before * 0.1, (
        f"Projection should reduce divergence: before={div_before:.6f}, after={div_after:.6f}"
    )


def test_projection_reduces_divergence_iterative():
    """Iterative (fewer iterations) projection also reduces divergence."""
    B, C, H, W = 2, 2, 10, 10
    torch.manual_seed(456)

    field = torch.randn(B, C, H, W)
    proj = DivergenceConservingProjection(method="iterative", max_iter=100)
    corrected = proj(field)

    loss_fn = ConservationLoss()
    div_before = loss_fn(field).item()
    div_after = loss_fn(corrected).item()

    assert div_after < div_before * 0.5, (
        f"Iterative projection should reduce divergence: before={div_before:.6f}, after={div_after:.6f}"
    )


def test_conservation_loss_decreases_with_projection():
    """ConservationLoss should report lower values after projection."""
    torch.manual_seed(789)
    B, C, H, W = 1, 2, 10, 10
    field = torch.randn(B, C, H, W)

    loss_fn = ConservationLoss()
    proj = DivergenceConservingProjection(method="direct", max_iter=200)

    loss_before = loss_fn(field)
    corrected = proj(field)
    loss_after = loss_fn(corrected)

    assert loss_after < loss_before, (
        f"Conservation loss should decrease: before={loss_before.item():.6f}, "
        f"after={loss_after.item():.6f}"
    )


# ---------------------------------------------------------------------------
# Flux-conserving linear tests
# ---------------------------------------------------------------------------


def test_flux_conserving_linear_preserves_flux():
    """FluxConservingLinear should preserve total flux (weighted sum)."""
    torch.manual_seed(42)
    B, N, in_dim, out_dim = 3, 20, 4, 6

    layer = FluxConservingLinear(in_dim, out_dim)
    x = torch.randn(B, N, in_dim)

    y = layer(x)

    input_flux = x.sum(dim=1).sum(dim=-1)
    output_flux = y.sum(dim=1).sum(dim=-1)

    assert torch.allclose(input_flux, output_flux, atol=1e-4), (
        f"Flux not preserved: input={input_flux.tolist()}, output={output_flux.tolist()}"
    )


def test_flux_conserving_linear_with_volumes():
    """Flux conservation works with volume weighting."""
    torch.manual_seed(42)
    B, N, in_dim, out_dim = 2, 15, 3, 5

    layer = FluxConservingLinear(in_dim, out_dim)
    x = torch.randn(B, N, in_dim)
    volumes = torch.rand(N) + 0.1

    y = layer(x, node_volumes=volumes)

    w = volumes.unsqueeze(0).unsqueeze(-1)
    input_flux = (x * w).sum(dim=1).sum(dim=-1)
    output_flux = (y * w).sum(dim=1).sum(dim=-1)

    assert torch.allclose(input_flux, output_flux, atol=1e-4), (
        f"Volume-weighted flux not preserved: input={input_flux.tolist()}, output={output_flux.tolist()}"
    )


# ---------------------------------------------------------------------------
# Structure-preserving encoder tests
# ---------------------------------------------------------------------------


def test_structure_preserving_encoder_output_shape():
    """StructurePreservingEncoder produces correct output shapes."""
    embed_dim = 32
    enc = StructurePreservingEncoder(in_dim=0, embed_dim=embed_dim, scales=(4, 8))

    # Unbatched
    N = 30
    points = torch.randn(N, 3)
    out = enc(points)
    assert out.shape == (embed_dim,), f"Expected ({embed_dim},), got {out.shape}"

    # Batched
    B = 2
    pts_b = torch.randn(B, N, 3)
    out_b = enc(pts_b)
    assert out_b.shape == (B, embed_dim), f"Expected ({B}, {embed_dim}), got {out_b.shape}"


def test_structure_preserving_encoder_differentiable():
    """Gradients flow through the full encoder + projection pipeline."""
    enc = StructurePreservingEncoder(in_dim=0, embed_dim=32, scales=(4, 8))
    points = torch.randn(25, 3, requires_grad=True)
    out = enc(points)
    out.sum().backward()
    assert points.grad is not None
    assert torch.isfinite(points.grad).all()


# ---------------------------------------------------------------------------
# OOD generalization test
# ---------------------------------------------------------------------------


def test_ood_geometry_improvement():
    """Training on square-domain data, the projection helps on circular-domain data.

    With the divergence projection, a model trained on one geometry should have
    lower conservation-law violation on a different geometry compared to the
    same model without projection.
    """
    torch.manual_seed(0)
    H, W = 10, 10
    n_epochs = 80

    N_train = 40
    train_fields = torch.randn(N_train, 2, H, W)

    class SimpleModel(nn.Module):
        def __init__(self, use_projection: bool):
            super().__init__()
            self.use_projection = use_projection
            self.conv = nn.Sequential(
                nn.Conv2d(2, 16, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(16, 2, 3, padding=1),
            )
            if use_projection:
                self.proj = DivergenceConservingProjection(method="direct", max_iter=100)

        def forward(self, x):
            out = self.conv(x) + x
            if self.use_projection:
                out = self.proj(out)
            return out

    model_proj = SimpleModel(use_projection=True)
    model_noproj = SimpleModel(use_projection=False)

    for model in [model_proj, model_noproj]:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(n_epochs):
            opt.zero_grad()
            pred = model(train_fields)
            loss = ((pred - train_fields) ** 2).mean()
            loss.backward()
            opt.step()

    # OOD test: circular mask
    N_test = 15
    test_fields = torch.randn(N_test, 2, H, W)
    cy, cx = H // 2, W // 2
    r = min(H, W) // 2 - 1
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r**2
    test_fields_masked = test_fields * mask.float().unsqueeze(0).unsqueeze(0)

    loss_fn = ConservationLoss()

    with torch.no_grad():
        pred_proj = model_proj(test_fields_masked)
        pred_noproj = model_noproj(test_fields_masked)

        loss_proj = loss_fn(pred_proj).item()
        loss_noproj = loss_fn(pred_noproj).item()

    assert loss_proj < loss_noproj, (
        f"Projection should help OOD: proj={loss_proj:.6f}, noproj={loss_noproj:.6f}"
    )


# ---------------------------------------------------------------------------
# Gradient flow tests
# ---------------------------------------------------------------------------


def test_gradient_flows_through_projection():
    """Gradients flow correctly through the full projection pipeline."""
    B, C, H, W = 1, 2, 8, 8
    proj = DivergenceConservingProjection(method="direct", max_iter=100)

    field = torch.randn(B, C, H, W, requires_grad=True)
    corrected = proj(field)

    loss = corrected.sum()
    loss.backward()

    assert field.grad is not None, "Gradient is None"
    assert torch.isfinite(field.grad).all(), "Gradient contains non-finite values"
    assert (field.grad.abs() > 0).any(), "Gradient is all zeros"


def test_gradient_flows_through_projection_iterative():
    """Gradients also flow through the iterative projection."""
    B, C, H, W = 1, 2, 8, 8
    proj = DivergenceConservingProjection(method="iterative", max_iter=50)

    field = torch.randn(B, C, H, W, requires_grad=True)
    corrected = proj(field)

    loss = corrected.sum()
    loss.backward()

    assert field.grad is not None
    assert torch.isfinite(field.grad).all()
