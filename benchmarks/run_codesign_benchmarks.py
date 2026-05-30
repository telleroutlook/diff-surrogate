#!/usr/bin/env python
"""Reproducible benchmarks comparing co-design vs isolated domain optimization.

Benchmark problems:
  1. Simple 2-domain coupling: toy quadratic functions with cross-domain coupling
  2. Geometry co-design: B-spline control points affecting two different SDF objectives

Optimization strategies per problem:
  - Coupled (joint gradients through shared parameters)
  - Decoupled-alternating (alternate between domain objectives each step)
  - Decoupled-sequential (optimize domain A first, then domain B)
  - Random baseline

Metrics: convergence speed, final loss, gradient norm history.

Usage:
    python benchmarks/run_codesign_benchmarks.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from torch import Tensor

# Ensure the package is importable when running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diff_surrogate.geometry import eval_closed_cubic_bspline, sdf_from_curve

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SEED = 42
N_STEPS = 200
LR = 1e-2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grad_norm(params: list[Tensor]) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().norm().item() ** 2
    return total**0.5


def _run_optimization(
    step_fn,
    params: list[Tensor],
    n_steps: int,
    lr: float,
    label: str,
) -> dict:
    """Generic optimization loop. ``step_fn(params)`` must return a scalar loss."""
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_history: list[float] = []
    grad_norm_history: list[float] = []
    t0 = time.perf_counter()

    for step in range(n_steps):
        optimizer.zero_grad()
        loss = step_fn(params)
        loss.backward()
        grad_norm_history.append(_grad_norm(params))
        optimizer.step()
        loss_history.append(loss.item())

    elapsed = time.perf_counter() - t0
    return {
        "label": label,
        "loss_history": loss_history,
        "grad_norm_history": grad_norm_history,
        "final_loss": loss_history[-1],
        "best_loss": min(loss_history),
        "best_step": int(min(range(len(loss_history)), key=lambda i: loss_history[i])),
        "convergence_speed": _convergence_step(loss_history),
        "wall_time_s": round(elapsed, 4),
    }


def _convergence_step(losses: list[float], threshold: float = 0.01) -> int | None:
    """Step at which loss first drops below threshold * initial_loss. None if never."""
    if len(losses) < 2:
        return None
    target = threshold * losses[0]
    for i, l in enumerate(losses):
        if l <= target:
            return i
    return None


# ---------------------------------------------------------------------------
# Benchmark 1: Simple 2-domain coupling (quadratic)
# ---------------------------------------------------------------------------


def _quadratic_coupled(params: list[Tensor]) -> Tensor:
    """Coupled loss: two domains sharing a design variable z with conflicting optima.

    domain_a(x, z) = (x - 1)^2 + (z - 1.5)^2      -- z wants to be 1.5
    domain_b(y, z) = (y + 1)^2 + (z + 1.5)^2      -- z wants to be -1.5
    coupling       = 0.5 * (x - y - z)^2           -- shared constraint

    Joint optimum for z is near 0 (compromise). Decoupled methods overfit z to
    one domain and the coupling penalty forces expensive correction later.
    """
    x, y, z = params[0], params[1], params[2]
    loss_a = (x - 1.0) ** 2 + (z - 1.5) ** 2
    loss_b = (y + 1.0) ** 2 + (z + 1.5) ** 2
    coupling = 0.5 * (x - y - z) ** 2
    return loss_a + loss_b + coupling


def _quadratic_domain_a(params: list[Tensor]) -> Tensor:
    """Domain A in isolation: z optimal at 1.5."""
    x, z = params[0], params[2]
    return (x - 1.0) ** 2 + (z - 1.5) ** 2


def _quadratic_domain_b(params: list[Tensor]) -> Tensor:
    """Domain B in isolation: z optimal at -1.5."""
    y, z = params[1], params[2]
    return (y + 1.0) ** 2 + (z + 1.5) ** 2


def _quadratic_coupling_only(params: list[Tensor]) -> Tensor:
    x, y, z = params[0], params[1], params[2]
    return 0.5 * (x - y - z) ** 2


def _benchmark_quadratic() -> dict:
    results = {"problem": "quadratic_coupling", "strategies": []}

    # -- Coupled --
    torch.manual_seed(SEED)
    params = [
        torch.tensor([3.0], requires_grad=True),
        torch.tensor([-3.0], requires_grad=True),
        torch.tensor([2.0], requires_grad=True),
    ]
    results["strategies"].append(
        _run_optimization(_quadratic_coupled, params, N_STEPS, LR, "coupled")
    )

    # -- Decoupled-alternating --
    torch.manual_seed(SEED)
    params = [
        torch.tensor([3.0], requires_grad=True),
        torch.tensor([-3.0], requires_grad=True),
        torch.tensor([2.0], requires_grad=True),
    ]
    optimizer = torch.optim.Adam(params, lr=LR)
    loss_history = []
    grad_norm_history = []
    t0 = time.perf_counter()
    for step in range(N_STEPS):
        optimizer.zero_grad()
        if step % 2 == 0:
            loss = _quadratic_domain_a(params)
        else:
            loss = _quadratic_domain_b(params)
        loss.backward()
        grad_norm_history.append(_grad_norm(params))
        optimizer.step()
        loss_history.append(loss.item())
    elapsed = time.perf_counter() - t0
    results["strategies"].append(
        {
            "label": "decoupled_alternating",
            "loss_history": loss_history,
            "grad_norm_history": grad_norm_history,
            "final_loss": loss_history[-1],
            "best_loss": min(loss_history),
            "best_step": int(min(range(len(loss_history)), key=lambda i: loss_history[i])),
            "convergence_speed": _convergence_step(loss_history),
            "wall_time_s": round(elapsed, 4),
        }
    )

    # -- Decoupled-sequential (A first, then B) --
    torch.manual_seed(SEED)
    params = [
        torch.tensor([3.0], requires_grad=True),
        torch.tensor([-3.0], requires_grad=True),
        torch.tensor([2.0], requires_grad=True),
    ]
    optimizer = torch.optim.Adam(params, lr=LR)
    loss_history = []
    grad_norm_history = []
    half = N_STEPS // 2
    t0 = time.perf_counter()
    for step in range(N_STEPS):
        optimizer.zero_grad()
        if step < half:
            loss = _quadratic_domain_a(params)
        else:
            loss = _quadratic_domain_b(params)
        loss.backward()
        grad_norm_history.append(_grad_norm(params))
        optimizer.step()
        loss_history.append(loss.item())
    elapsed = time.perf_counter() - t0
    results["strategies"].append(
        {
            "label": "decoupled_sequential",
            "loss_history": loss_history,
            "grad_norm_history": grad_norm_history,
            "final_loss": loss_history[-1],
            "best_loss": min(loss_history),
            "best_step": int(min(range(len(loss_history)), key=lambda i: loss_history[i])),
            "convergence_speed": _convergence_step(loss_history),
            "wall_time_s": round(elapsed, 4),
        }
    )

    # -- Random baseline --
    torch.manual_seed(SEED)
    params = [
        torch.tensor([3.0], requires_grad=True),
        torch.tensor([-3.0], requires_grad=True),
        torch.tensor([2.0], requires_grad=True),
    ]
    rng = torch.Generator().manual_seed(SEED + 1)
    loss_history = []
    grad_norm_history = []
    t0 = time.perf_counter()
    for step in range(N_STEPS):
        perturbation = torch.randn(1, generator=rng) * 0.1
        with torch.no_grad():
            for p in params:
                p.add_(perturbation)
        loss = _quadratic_coupled(params)
        # Compute gradient for logging only
        if loss.requires_grad:
            loss_val = loss.item()
        else:
            loss_val = loss.item() if isinstance(loss, Tensor) else float(loss)
        loss_history.append(loss_val)
        grad_norm_history.append(0.0)
    elapsed = time.perf_counter() - t0
    results["strategies"].append(
        {
            "label": "random_baseline",
            "loss_history": loss_history,
            "grad_norm_history": grad_norm_history,
            "final_loss": loss_history[-1],
            "best_loss": min(loss_history),
            "best_step": int(min(range(len(loss_history)), key=lambda i: loss_history[i])),
            "convergence_speed": _convergence_step(loss_history),
            "wall_time_s": round(elapsed, 4),
        }
    )

    return results


# ---------------------------------------------------------------------------
# Benchmark 2: Geometry co-design (B-spline control points)
# ---------------------------------------------------------------------------


def _make_grid(resolution: int = 32, extent: float = 2.0):
    """Create evaluation grid for SDF computation."""
    lin = torch.linspace(-extent, extent, resolution)
    grid_x = lin.unsqueeze(0).expand(resolution, -1)
    grid_y = lin.unsqueeze(-1).expand(-1, resolution)
    return grid_x, grid_y


def _geometry_coupled(params: list[Tensor], grid_x: Tensor, grid_y: Tensor) -> Tensor:
    """Coupled geometry loss: one B-spline shape satisfies two domain objectives.

    Domain A (nanophotonics): match a circular target SDF (radius 0.8).
    Domain B (fluid dynamics): match an elliptical target SDF (wider, for low drag).

    Shared parameters: the B-spline control points. The conflict is that a circle
    (optimal for A) is not an ellipse (optimal for B), and vice versa. Joint
    optimization finds the Pareto-optimal compromise shape.
    """
    control_points = params[0]
    t = torch.linspace(0, 1, 64)
    curve = eval_closed_cubic_bspline(control_points, t)
    sdf = sdf_from_curve(grid_x, grid_y, curve)

    # Domain A: match circular target (radius 0.8)
    r = (grid_x**2 + grid_y**2).sqrt()
    target_circle = r - 0.8
    loss_a = ((sdf - target_circle) ** 2).mean()

    # Domain B: match elliptical target (semi-axes 1.0 x 0.6)
    r_ellipse = ((grid_x / 1.0) ** 2 + (grid_y / 0.6) ** 2).sqrt()
    target_ellipse = r_ellipse - 0.8
    loss_b = ((sdf - target_ellipse) ** 2).mean()

    return loss_a + loss_b


def _geometry_domain_a(params: list[Tensor], grid_x: Tensor, grid_y: Tensor) -> Tensor:
    control_points = params[0]
    t = torch.linspace(0, 1, 64)
    curve = eval_closed_cubic_bspline(control_points, t)
    sdf = sdf_from_curve(grid_x, grid_y, curve)
    r = (grid_x**2 + grid_y**2).sqrt()
    target_circle = r - 0.8
    return ((sdf - target_circle) ** 2).mean()


def _geometry_domain_b(params: list[Tensor], grid_x: Tensor, grid_y: Tensor) -> Tensor:
    control_points = params[0]
    t = torch.linspace(0, 1, 64)
    curve = eval_closed_cubic_bspline(control_points, t)
    sdf = sdf_from_curve(grid_x, grid_y, curve)
    r_ellipse = ((grid_x / 1.0) ** 2 + (grid_y / 0.6) ** 2).sqrt()
    target_ellipse = r_ellipse - 0.8
    return ((sdf - target_ellipse) ** 2).mean()


def _benchmark_geometry() -> dict:
    results = {"problem": "geometry_codesign", "strategies": []}
    grid_x, grid_y = _make_grid(resolution=24, extent=2.0)

    n_control = 8
    init_cp = torch.tensor(
        [
            [0.5 + 0.1 * i, 0.5 * (-1) ** i]
            for i in range(n_control)
        ],
        dtype=torch.float32,
    )

    # -- Coupled --
    torch.manual_seed(SEED)
    cp = init_cp.clone().detach().requires_grad_(True)
    results["strategies"].append(
        _run_optimization(
            lambda params: _geometry_coupled(params, grid_x, grid_y),
            [cp],
            N_STEPS,
            LR,
            "coupled",
        )
    )

    # -- Decoupled-alternating --
    torch.manual_seed(SEED)
    cp = init_cp.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([cp], lr=LR)
    loss_history = []
    grad_norm_history = []
    t0 = time.perf_counter()
    for step in range(N_STEPS):
        optimizer.zero_grad()
        if step % 2 == 0:
            loss = _geometry_domain_a([cp], grid_x, grid_y)
        else:
            loss = _geometry_domain_b([cp], grid_x, grid_y)
        loss.backward()
        grad_norm_history.append(_grad_norm([cp]))
        optimizer.step()
        loss_history.append(loss.item())
    elapsed = time.perf_counter() - t0
    results["strategies"].append(
        {
            "label": "decoupled_alternating",
            "loss_history": loss_history,
            "grad_norm_history": grad_norm_history,
            "final_loss": loss_history[-1],
            "best_loss": min(loss_history),
            "best_step": int(min(range(len(loss_history)), key=lambda i: loss_history[i])),
            "convergence_speed": _convergence_step(loss_history),
            "wall_time_s": round(elapsed, 4),
        }
    )

    # -- Decoupled-sequential --
    torch.manual_seed(SEED)
    cp = init_cp.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([cp], lr=LR)
    loss_history = []
    grad_norm_history = []
    half = N_STEPS // 2
    t0 = time.perf_counter()
    for step in range(N_STEPS):
        optimizer.zero_grad()
        if step < half:
            loss = _geometry_domain_a([cp], grid_x, grid_y)
        else:
            loss = _geometry_domain_b([cp], grid_x, grid_y)
        loss.backward()
        grad_norm_history.append(_grad_norm([cp]))
        optimizer.step()
        loss_history.append(loss.item())
    elapsed = time.perf_counter() - t0
    results["strategies"].append(
        {
            "label": "decoupled_sequential",
            "loss_history": loss_history,
            "grad_norm_history": grad_norm_history,
            "final_loss": loss_history[-1],
            "best_loss": min(loss_history),
            "best_step": int(min(range(len(loss_history)), key=lambda i: loss_history[i])),
            "convergence_speed": _convergence_step(loss_history),
            "wall_time_s": round(elapsed, 4),
        }
    )

    # -- Random baseline --
    torch.manual_seed(SEED)
    cp = init_cp.clone().detach()
    rng = torch.Generator().manual_seed(SEED + 1)
    loss_history = []
    grad_norm_history = []
    t0 = time.perf_counter()
    for step in range(N_STEPS):
        with torch.no_grad():
            cp.add_(torch.randn_like(cp, generator=rng) * 0.02)
        loss = _geometry_coupled([cp], grid_x, grid_y)
        loss_history.append(loss.item())
        grad_norm_history.append(0.0)
    elapsed = time.perf_counter() - t0
    results["strategies"].append(
        {
            "label": "random_baseline",
            "loss_history": loss_history,
            "grad_norm_history": grad_norm_history,
            "final_loss": loss_history[-1],
            "best_loss": min(loss_history),
            "best_step": int(min(range(len(loss_history)), key=lambda i: loss_history[i])),
            "convergence_speed": _convergence_step(loss_history),
            "wall_time_s": round(elapsed, 4),
        }
    )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 64)
    print("  Co-Design Benchmarks: Coupled vs Decoupled Optimization")
    print("=" * 64)
    print(f"  Seed: {SEED}  |  Steps: {N_STEPS}  |  LR: {LR}")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {
        "config": {
            "seed": SEED,
            "n_steps": N_STEPS,
            "lr": LR,
        },
        "benchmarks": [],
    }

    # Benchmark 1: Quadratic coupling
    print("[1/2] Quadratic coupling benchmark...")
    quad = _benchmark_quadratic()
    all_results["benchmarks"].append(quad)
    for s in quad["strategies"]:
        print(f"  {s['label']:>24s}  final_loss={s['final_loss']:.6f}  "
              f"best_loss={s['best_loss']:.6f}  conv@{s['convergence_speed']}")
    print()

    # Benchmark 2: Geometry co-design
    print("[2/2] Geometry co-design benchmark...")
    geom = _benchmark_geometry()
    all_results["benchmarks"].append(geom)
    for s in geom["strategies"]:
        print(f"  {s['label']:>24s}  final_loss={s['final_loss']:.6f}  "
              f"best_loss={s['best_loss']:.6f}  conv@{s['convergence_speed']}")
    print()

    # Save results
    out_path = RESULTS_DIR / "codesign_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
