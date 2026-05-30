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
Multi-seed: runs with ≥10 seeds, reports mean±std and Wilcoxon significance.

Usage:
    python benchmarks/run_codesign_benchmarks.py                # 10 seeds
    python benchmarks/run_codesign_benchmarks.py --seeds 20     # 20 seeds
    python benchmarks/run_codesign_benchmarks.py --seed-start 0 # start from seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diff_surrogate.geometry import eval_closed_cubic_bspline, sdf_from_curve

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_N_SEEDS = 10
DEFAULT_SEED_START = 42
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
    if len(losses) < 2:
        return None
    target = threshold * losses[0]
    for i, l in enumerate(losses):
        if l <= target:
            return i
    return None


def _aggregate_multi_seed(
    per_seed_results: list[list[dict]],
) -> list[dict]:
    """Aggregate results across seeds: mean±std + Wilcoxon significance."""
    if not per_seed_results:
        return []

    n_strategies = len(per_seed_results[0])
    strategies: list[dict] = []

    for si in range(n_strategies):
        label = per_seed_results[0][si]["label"]
        final_losses = [r[si]["final_loss"] for r in per_seed_results]
        best_losses = [r[si]["best_loss"] for r in per_seed_results]
        conv_speeds = [r[si]["convergence_speed"] for r in per_seed_results]
        wall_times = [r[si]["wall_time_s"] for r in per_seed_results]

        import numpy as np

        agg = {
            "label": label,
            "n_seeds": len(per_seed_results),
            "final_loss_mean": float(np.mean(final_losses)),
            "final_loss_std": float(np.std(final_losses, ddof=1)) if len(final_losses) > 1 else 0.0,
            "final_loss_values": [float(v) for v in final_losses],
            "best_loss_mean": float(np.mean(best_losses)),
            "best_loss_std": float(np.std(best_losses, ddof=1)) if len(best_losses) > 1 else 0.0,
            "best_loss_values": [float(v) for v in best_losses],
            "convergence_speed_mean": float(np.mean([s for s in conv_speeds if s is not None])) if any(s is not None for s in conv_speeds) else None,
            "wall_time_mean": float(np.mean(wall_times)),
            "wall_time_std": float(np.std(wall_times, ddof=1)) if len(wall_times) > 1 else 0.0,
        }
        strategies.append(agg)

    if n_strategies >= 2:
        coupled_losses = [r[0]["final_loss"] for r in per_seed_results]
        for si in range(1, n_strategies):
            other_losses = [r[si]["final_loss"] for r in per_seed_results]
            try:
                from scipy.stats import wilcoxon

                stat, p_value = wilcoxon(coupled_losses, other_losses)
                strategies[si]["wilcoxon_vs_coupled"] = {
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "significant_005": bool(p_value < 0.05),
                }
            except Exception:
                pass

    return strategies


# ---------------------------------------------------------------------------
# Benchmark 1: Simple 2-domain coupling (quadratic)
# ---------------------------------------------------------------------------


def _quadratic_coupled(params: list[Tensor]) -> Tensor:
    x, y, z = params[0], params[1], params[2]
    loss_a = (x - 1.0) ** 2 + (z - 1.5) ** 2
    loss_b = (y + 1.0) ** 2 + (z + 1.5) ** 2
    coupling = 0.5 * (x - y - z) ** 2
    return loss_a + loss_b + coupling


def _quadratic_domain_a(params: list[Tensor]) -> Tensor:
    x, z = params[0], params[2]
    return (x - 1.0) ** 2 + (z - 1.5) ** 2


def _quadratic_domain_b(params: list[Tensor]) -> Tensor:
    y, z = params[1], params[2]
    return (y + 1.0) ** 2 + (z + 1.5) ** 2


def _init_params_quadratic(seed: int) -> list[Tensor]:
    torch.manual_seed(seed)
    return [
        torch.tensor([3.0 + 0.5 * torch.randn(1).item()], requires_grad=True),
        torch.tensor([-3.0 + 0.5 * torch.randn(1).item()], requires_grad=True),
        torch.tensor([2.0 + 0.5 * torch.randn(1).item()], requires_grad=True),
    ]


def _benchmark_quadratic(seed: int) -> list[dict]:
    results = []

    # Coupled
    params = _init_params_quadratic(seed)
    results.append(_run_optimization(_quadratic_coupled, params, N_STEPS, LR, "coupled"))

    # Decoupled-alternating
    params = _init_params_quadratic(seed)
    optimizer = torch.optim.Adam(params, lr=LR)
    loss_history = []
    grad_norm_history = []
    t0 = time.perf_counter()
    for step in range(N_STEPS):
        optimizer.zero_grad()
        loss = _quadratic_domain_a(params) if step % 2 == 0 else _quadratic_domain_b(params)
        loss.backward()
        grad_norm_history.append(_grad_norm(params))
        optimizer.step()
        loss_history.append(loss.item())
    elapsed = time.perf_counter() - t0
    results.append({
        "label": "decoupled_alternating",
        "loss_history": loss_history,
        "grad_norm_history": grad_norm_history,
        "final_loss": loss_history[-1],
        "best_loss": min(loss_history),
        "best_step": int(min(range(len(loss_history)), key=lambda i: loss_history[i])),
        "convergence_speed": _convergence_step(loss_history),
        "wall_time_s": round(elapsed, 4),
    })

    # Decoupled-sequential
    params = _init_params_quadratic(seed)
    optimizer = torch.optim.Adam(params, lr=LR)
    loss_history = []
    grad_norm_history = []
    half = N_STEPS // 2
    t0 = time.perf_counter()
    for step in range(N_STEPS):
        optimizer.zero_grad()
        loss = _quadratic_domain_a(params) if step < half else _quadratic_domain_b(params)
        loss.backward()
        grad_norm_history.append(_grad_norm(params))
        optimizer.step()
        loss_history.append(loss.item())
    elapsed = time.perf_counter() - t0
    results.append({
        "label": "decoupled_sequential",
        "loss_history": loss_history,
        "grad_norm_history": grad_norm_history,
        "final_loss": loss_history[-1],
        "best_loss": min(loss_history),
        "best_step": int(min(range(len(loss_history)), key=lambda i: loss_history[i])),
        "convergence_speed": _convergence_step(loss_history),
        "wall_time_s": round(elapsed, 4),
    })

    # Random baseline
    params = _init_params_quadratic(seed)
    rng = torch.Generator().manual_seed(seed + 1)
    loss_history = []
    t0 = time.perf_counter()
    for step in range(N_STEPS):
        perturbation = torch.randn(1, generator=rng) * 0.1
        with torch.no_grad():
            for p in params:
                p.add_(perturbation)
        loss = _quadratic_coupled(params)
        loss_history.append(loss.item())
    elapsed = time.perf_counter() - t0
    results.append({
        "label": "random_baseline",
        "loss_history": loss_history,
        "grad_norm_history": [0.0] * N_STEPS,
        "final_loss": loss_history[-1],
        "best_loss": min(loss_history),
        "best_step": int(min(range(len(loss_history)), key=lambda i: loss_history[i])),
        "convergence_speed": _convergence_step(loss_history),
        "wall_time_s": round(elapsed, 4),
    })

    return results


# ---------------------------------------------------------------------------
# Benchmark 2: Geometry co-design (B-spline control points)
# ---------------------------------------------------------------------------


def _make_grid(resolution: int = 32, extent: float = 2.0):
    lin = torch.linspace(-extent, extent, resolution)
    grid_x = lin.unsqueeze(0).expand(resolution, -1)
    grid_y = lin.unsqueeze(-1).expand(-1, resolution)
    return grid_x, grid_y


def _geometry_coupled(params: list[Tensor], grid_x: Tensor, grid_y: Tensor) -> Tensor:
    control_points = params[0]
    t = torch.linspace(0, 1, 64)
    curve = eval_closed_cubic_bspline(control_points, t)
    sdf = sdf_from_curve(grid_x, grid_y, curve)
    r = (grid_x**2 + grid_y**2).sqrt()
    target_circle = r - 0.8
    loss_a = ((sdf - target_circle) ** 2).mean()
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


def _init_cp_geometry(seed: int) -> Tensor:
    n_control = 8
    init_cp = torch.tensor(
        [[0.5 + 0.1 * i, 0.5 * (-1) ** i] for i in range(n_control)],
        dtype=torch.float32,
    )
    torch.manual_seed(seed)
    return init_cp + 0.1 * torch.randn_like(init_cp)


def _benchmark_geometry(seed: int) -> list[dict]:
    grid_x, grid_y = _make_grid(resolution=24, extent=2.0)
    results = []

    # Coupled
    cp = _init_cp_geometry(seed).detach().requires_grad_(True)
    results.append(
        _run_optimization(lambda params: _geometry_coupled(params, grid_x, grid_y), [cp], N_STEPS, LR, "coupled")
    )

    # Decoupled-alternating
    cp = _init_cp_geometry(seed).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([cp], lr=LR)
    loss_history = []
    grad_norm_history = []
    t0 = time.perf_counter()
    for step in range(N_STEPS):
        optimizer.zero_grad()
        loss = _geometry_domain_a([cp], grid_x, grid_y) if step % 2 == 0 else _geometry_domain_b([cp], grid_x, grid_y)
        loss.backward()
        grad_norm_history.append(_grad_norm([cp]))
        optimizer.step()
        loss_history.append(loss.item())
    elapsed = time.perf_counter() - t0
    results.append({
        "label": "decoupled_alternating",
        "loss_history": loss_history,
        "grad_norm_history": grad_norm_history,
        "final_loss": loss_history[-1],
        "best_loss": min(loss_history),
        "best_step": int(min(range(len(loss_history)), key=lambda i: loss_history[i])),
        "convergence_speed": _convergence_step(loss_history),
        "wall_time_s": round(elapsed, 4),
    })

    # Decoupled-sequential
    cp = _init_cp_geometry(seed).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([cp], lr=LR)
    loss_history = []
    grad_norm_history = []
    half = N_STEPS // 2
    t0 = time.perf_counter()
    for step in range(N_STEPS):
        optimizer.zero_grad()
        loss = _geometry_domain_a([cp], grid_x, grid_y) if step < half else _geometry_domain_b([cp], grid_x, grid_y)
        loss.backward()
        grad_norm_history.append(_grad_norm([cp]))
        optimizer.step()
        loss_history.append(loss.item())
    elapsed = time.perf_counter() - t0
    results.append({
        "label": "decoupled_sequential",
        "loss_history": loss_history,
        "grad_norm_history": grad_norm_history,
        "final_loss": loss_history[-1],
        "best_loss": min(loss_history),
        "best_step": int(min(range(len(loss_history)), key=lambda i: loss_history[i])),
        "convergence_speed": _convergence_step(loss_history),
        "wall_time_s": round(elapsed, 4),
    })

    # Random baseline
    cp = _init_cp_geometry(seed).detach()
    rng = torch.Generator().manual_seed(seed + 1)
    loss_history = []
    t0 = time.perf_counter()
    for step in range(N_STEPS):
        with torch.no_grad():
            cp.add_(torch.randn_like(cp, generator=rng) * 0.02)
        loss = _geometry_coupled([cp], grid_x, grid_y)
        loss_history.append(loss.item())
    elapsed = time.perf_counter() - t0
    results.append({
        "label": "random_baseline",
        "loss_history": loss_history,
        "grad_norm_history": [0.0] * N_STEPS,
        "final_loss": loss_history[-1],
        "best_loss": min(loss_history),
        "best_step": int(min(range(len(loss_history)), key=lambda i: loss_history[i])),
        "convergence_speed": _convergence_step(loss_history),
        "wall_time_s": round(elapsed, 4),
    })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Co-design benchmarks with multi-seed support")
    parser.add_argument("--seeds", type=int, default=DEFAULT_N_SEEDS, help="Number of seeds to run")
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START, help="Starting seed value")
    args = parser.parse_args()

    print("=" * 64)
    print("  Co-Design Benchmarks: Coupled vs Decoupled Optimization")
    print("=" * 64)
    print(f"  Seeds: {args.seeds} (start={args.seed_start})  |  Steps: {N_STEPS}  |  LR: {LR}")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {
        "config": {
            "n_seeds": args.seeds,
            "seed_start": args.seed_start,
            "n_steps": N_STEPS,
            "lr": LR,
        },
        "benchmarks": [],
    }

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    # Benchmark 1: Quadratic coupling
    print("[1/2] Quadratic coupling benchmark...")
    quad_per_seed = []
    for i, seed in enumerate(seeds):
        quad_per_seed.append(_benchmark_quadratic(seed))
        print(f"  seed={seed} ({i + 1}/{len(seeds)})", end="\r")
    print()

    quad_strategies = _aggregate_multi_seed(quad_per_seed)
    all_results["benchmarks"].append({
        "problem": "quadratic_coupling",
        "strategies": quad_strategies,
    })
    for s in quad_strategies:
        sig = ""
        if "wilcoxon_vs_coupled" in s:
            w = s["wilcoxon_vs_coupled"]
            sig = f"  p={w['p_value']:.4f}{'*' if w['significant_005'] else ''}"
        print(f"  {s['label']:>24s}  final={s['final_loss_mean']:.6f}±{s['final_loss_std']:.6f}  "
              f"best={s['best_loss_mean']:.6f}±{s['best_loss_std']:.6f}{sig}")
    print()

    # Benchmark 2: Geometry co-design
    print("[2/2] Geometry co-design benchmark...")
    geom_per_seed = []
    for i, seed in enumerate(seeds):
        geom_per_seed.append(_benchmark_geometry(seed))
        print(f"  seed={seed} ({i + 1}/{len(seeds)})", end="\r")
    print()

    geom_strategies = _aggregate_multi_seed(geom_per_seed)
    all_results["benchmarks"].append({
        "problem": "geometry_codesign",
        "strategies": geom_strategies,
    })
    for s in geom_strategies:
        sig = ""
        if "wilcoxon_vs_coupled" in s:
            w = s["wilcoxon_vs_coupled"]
            sig = f"  p={w['p_value']:.4f}{'*' if w['significant_005'] else ''}"
        print(f"  {s['label']:>24s}  final={s['final_loss_mean']:.6f}±{s['final_loss_std']:.6f}  "
              f"best={s['best_loss_mean']:.6f}±{s['best_loss_std']:.6f}{sig}")
    print()

    out_path = RESULTS_DIR / "codesign_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
