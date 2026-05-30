#!/usr/bin/env python
"""Benchmark harness comparing FNO vs SDF-trunk vs Geo-FNO surrogates.

Compares three neural operator architectures on geometry-aware flow prediction:
  1. FNO (baseline): Fourier Neural Operator on regular grids
  2. SDF-Trunk: Geometry-aware DeepONet-style operator with SDF encoding
  3. Geo-FNO: FNO with learned domain deformation for irregular geometries

Benchmark problems:
  1. Cylinder flow: non-rectangular domain with circular obstacle SDF
  2. Heat exchanger: multiple rectangular obstacles in a channel

Metrics:
  - L2 relative error against ground truth
  - Training sample efficiency (error vs number of training samples)
  - Inference latency (forward pass time)

Multi-seed with Wilcoxon signed-rank significance testing.

Usage:
    python benchmarks/benchmark_surrogates.py
    python benchmarks/benchmark_surrogates.py --seeds 20
    python benchmarks/benchmark_surrogates.py --quick   # fast smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diff_surrogate import CrossAttnSurrogate, SDFTrunkSurrogate

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_N_SEEDS = 10
DEFAULT_SEED_START = 42


# ---------------------------------------------------------------------------
# Synthetic problem generators
# ---------------------------------------------------------------------------


def cylinder_sdf_field(
    grid_h: int = 32,
    grid_w: int = 64,
    cx: float = 0.25,
    cy: float = 0.5,
    radius: float = 0.1,
) -> Tensor:
    """Generate SDF for a cylinder in a channel.

    Convention: negative inside solid, positive in fluid.
    """
    y = torch.linspace(0, 1, grid_h)
    x = torch.linspace(0, 2, grid_w)
    gx, gy = torch.meshgrid(x, y, indexing="xy")
    dist = torch.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
    return dist - radius


def heat_exchanger_sdf_field(
    grid_h: int = 32,
    grid_w: int = 64,
    n_columns: int = 3,
    n_rows: int = 2,
) -> Tensor:
    """Generate SDF for a heat exchanger with rectangular obstacles."""
    y = torch.linspace(0, 1, grid_h)
    x = torch.linspace(0, 2, grid_w)
    gx, gy = torch.meshgrid(x, y, indexing="xy")

    sdf = torch.ones(grid_h, grid_w) * 10.0  # all fluid initially

    for col in range(n_columns):
        for row in range(n_rows):
            rx = 0.4 + col * 0.5
            ry = 0.25 + row * 0.35
            half_w, half_h = 0.08, 0.06
            dx = torch.maximum(rx - half_w - gx, gx - (rx + half_w))
            dy = torch.maximum(ry - half_h - gy, gy - (ry + half_h))
            outside = torch.sqrt(
                torch.clamp(dx, min=0) ** 2 + torch.clamp(dy, min=0) ** 2
            )
            inside = torch.clamp(torch.maximum(dx, dy), max=0)
            rect_sdf = outside + inside
            sdf = torch.minimum(sdf, rect_sdf)

    return sdf


def synthetic_flow_field(
    sdf: Tensor,
    inlet_velocity: float = 1.0,
    re: float = 100.0,
) -> Tensor:
    """Generate synthetic flow field from SDF for benchmarking.

    Creates a simplified velocity/pressure field that respects the SDF boundary.
    This is NOT a physical solve -- it's for benchmarking surrogate architectures.
    """
    H, W = sdf.shape
    fluid_mask = (sdf > 0).float()

    # Parabolic velocity profile modulated by SDF near boundaries
    y = torch.linspace(0, 1, H).unsqueeze(1).expand(H, W)
    parabolic = 4.0 * y * (1.0 - y)

    # SDF-based damping near walls
    boundary_factor = torch.sigmoid(10.0 * sdf)

    ux = inlet_velocity * parabolic * boundary_factor * fluid_mask
    uy = torch.zeros(H, W)
    p = -0.5 * inlet_velocity**2 * (1.0 - boundary_factor) * fluid_mask

    return torch.stack([ux, uy, p], dim=0)  # (3, H, W)


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------


def make_fno(modes: int = 8, width: int = 32, depth: int = 3) -> torch.nn.Module:
    """Create FNO2D model."""
    from diffcfd.surrogates.fno import FNO2D

    return FNO2D(modes=modes, width=width, depth=depth, in_channels=4, out_channels=3)


def make_sdf_trunk(
    param_dim: int = 2,
    n_outputs: int = 3,
    hidden_dim: int = 64,
    n_basis: int = 32,
) -> SDFTrunkSurrogate:
    return SDFTrunkSurrogate(
        param_dim=param_dim,
        n_outputs=n_outputs,
        hidden_dim=hidden_dim,
        n_basis=n_basis,
    )


def make_geo_fno(modes: int = 8, width: int = 32, depth: int = 3) -> torch.nn.Module:
    """Create GeoFNO model wrapping FNO2D."""
    from diffcfd.surrogates.fno import FNO2D
    from diffcfd.surrogates.geo_fno import GeoFNO

    fno = FNO2D(modes=modes, width=width, depth=depth, in_channels=4, out_channels=3)
    return GeoFNO(fno, deform_hidden=16)


def make_cross_attn(
    param_dim: int = 2,
    n_outputs: int = 3,
    hidden_dim: int = 64,
    n_heads: int = 4,
) -> CrossAttnSurrogate:
    return CrossAttnSurrogate(
        param_dim=param_dim,
        n_outputs=n_outputs,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
    )


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------


def _train_model(
    model: torch.nn.Module,
    inputs: Tensor | tuple[Tensor, Tensor],
    targets: Tensor,
    n_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 16,
) -> list[float]:
    """Train a model and return loss history."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    n = targets.shape[0]

    for _ in range(n_epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            if isinstance(inputs, tuple):
                sdf_in, param_in = inputs
                xb_sdf = sdf_in[idx]
                xb_param = param_in[idx]
            else:
                xb = inputs[idx]

            yb = targets[idx]
            opt.zero_grad()

            if isinstance(model, (SDFTrunkSurrogate, CrossAttnSurrogate)):
                pred = model((xb_sdf, xb_param))
            elif hasattr(model, "forward") and "sdf" in model.forward.__code__.co_varnames:
                # GeoFNO
                if isinstance(inputs, tuple):
                    pred = model(xb, sdf=xb_sdf.unsqueeze(1))
                else:
                    pred = model(xb)
            else:
                pred = model(xb)

            loss = ((pred - yb) ** 2).mean()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        losses.append(epoch_loss / max(1, n_batches))

    return losses


def _evaluate_l2(model: torch.nn.Module, inputs, targets: Tensor) -> float:
    """Compute L2 relative error."""
    with torch.no_grad():
        if isinstance(model, (SDFTrunkSurrogate, CrossAttnSurrogate)):
            sdf_in, param_in = inputs
            pred = model((sdf_in, param_in))
        elif hasattr(model, "forward") and "sdf" in model.forward.__code__.co_varnames:
            if isinstance(inputs, tuple):
                sdf_in, _ = inputs
                pred = model(inputs[0] if not isinstance(inputs, tuple) else inputs[0], sdf=sdf_in.unsqueeze(1))
            else:
                pred = model(inputs)
        else:
            pred = model(inputs)

        l2_err = ((pred - targets) ** 2).sum().sqrt().item()
        l2_ref = (targets**2).sum().sqrt().item()
        return l2_err / max(l2_ref, 1e-8)


def _measure_latency(
    model: torch.nn.Module,
    inputs,
    n_warmup: int = 5,
    n_runs: int = 20,
) -> float:
    """Measure mean forward pass latency in milliseconds."""
    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            if isinstance(model, (SDFTrunkSurrogate, CrossAttnSurrogate)):
                model(inputs)
            else:
                model(inputs if not isinstance(inputs, tuple) else inputs[0])

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            if isinstance(model, (SDFTrunkSurrogate, CrossAttnSurrogate)):
                model(inputs)
            else:
                model(inputs if not isinstance(inputs, tuple) else inputs[0])
        times.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(times))


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def _generate_cylinder_data(
    n_samples: int,
    grid_h: int = 32,
    grid_w: int = 64,
    seed: int = 0,
):
    """Generate cylinder flow training data."""
    torch.manual_seed(seed)
    sdf_base = cylinder_sdf_field(grid_h, grid_w)

    # FNO input: (B, 4, H, W) = [sdf, mask, u_inlet_field, Re_field]
    # SDF-trunk input: (sdf, params)
    fno_inputs = []
    sdf_inputs = []
    param_inputs = []
    targets = []

    for i in range(n_samples):
        u_inlet = 0.5 + 1.5 * (i / max(n_samples - 1, 1))
        re = 50.0 + 200.0 * torch.rand(1).item()

        # Add slight perturbation to cylinder position
        cx = 0.25 + 0.05 * torch.randn(1).item()
        sdf = cylinder_sdf_field(grid_h, grid_w, cx=cx)
        mask = torch.sigmoid(-10.0 * sdf)

        flow = synthetic_flow_field(sdf, u_inlet, re)

        fno_inputs.append(torch.stack([sdf, mask,
                                        torch.full((grid_h, grid_w), u_inlet),
                                        torch.full((grid_h, grid_w), re)], dim=0))
        sdf_inputs.append(sdf)
        param_inputs.append(torch.tensor([u_inlet, re]))
        targets.append(flow)

    fno_x = torch.stack(fno_inputs)
    sdf_x = torch.stack(sdf_inputs)
    param_x = torch.stack(param_inputs)
    y = torch.stack(targets)

    return fno_x, (sdf_x, param_x), y


def _generate_hex_data(
    n_samples: int,
    grid_h: int = 32,
    grid_w: int = 64,
    seed: int = 0,
):
    """Generate heat exchanger training data."""
    torch.manual_seed(seed)
    sdf_base = heat_exchanger_sdf_field(grid_h, grid_w)

    fno_inputs = []
    sdf_inputs = []
    param_inputs = []
    targets = []

    for i in range(n_samples):
        u_inlet = 0.3 + 1.2 * (i / max(n_samples - 1, 1))
        re = 20.0 + 100.0 * torch.rand(1).item()
        sdf = sdf_base.clone()
        mask = torch.sigmoid(-10.0 * sdf)

        flow = synthetic_flow_field(sdf, u_inlet, re)

        fno_inputs.append(torch.stack([sdf, mask,
                                        torch.full((grid_h, grid_w), u_inlet),
                                        torch.full((grid_h, grid_w), re)], dim=0))
        sdf_inputs.append(sdf)
        param_inputs.append(torch.tensor([u_inlet, re]))
        targets.append(flow)

    fno_x = torch.stack(fno_inputs)
    sdf_x = torch.stack(sdf_inputs)
    param_x = torch.stack(param_inputs)
    y = torch.stack(targets)

    return fno_x, (sdf_x, param_x), y


def _run_single_benchmark(
    problem: str,
    n_train: int,
    n_test: int,
    n_epochs: int,
    seed: int,
) -> list[dict]:
    """Run all three models on one problem/seed and return results."""
    gen_fn = _generate_cylinder_data if problem == "cylinder" else _generate_hex_data
    fno_train, (sdf_train, param_train), y_train = gen_fn(n_train, seed=seed)
    fno_test, (sdf_test, param_test), y_test = gen_fn(n_test, seed=seed + 1000)

    results = []

    # --- FNO ---
    torch.manual_seed(seed)
    fno_model = make_fno()
    _train_model(fno_model, fno_train, y_train, n_epochs=n_epochs)
    l2_fno = _evaluate_l2(fno_model, fno_test, y_test)
    lat_fno = _measure_latency(fno_model, fno_test[:1])
    results.append({
        "model": "FNO",
        "l2_error": l2_fno,
        "latency_ms": lat_fno,
    })

    # --- SDF-Trunk ---
    torch.manual_seed(seed)
    sdf_model = make_sdf_trunk()
    sdf_model.get_network()  # eagerly build network so parameters() is non-empty
    _train_model(sdf_model, (sdf_train, param_train), y_train, n_epochs=n_epochs)
    l2_sdf = _evaluate_l2(sdf_model, (sdf_test, param_test), y_test)
    lat_sdf = _measure_latency(sdf_model, (sdf_test[:1], param_test[:1]))
    results.append({
        "model": "SDFTrunk",
        "l2_error": l2_sdf,
        "latency_ms": lat_sdf,
    })

    # --- Geo-FNO ---
    torch.manual_seed(seed)
    geo_model = make_geo_fno()
    # GeoFNO uses FNO-style input + SDF for deformation
    _train_model_geo(geo_model, fno_train, sdf_train, y_train, n_epochs=n_epochs)
    l2_geo = _evaluate_l2_geo(geo_model, fno_test, sdf_test, y_test)
    lat_geo = _measure_latency_geo(geo_model, fno_test[:1], sdf_test[:1])
    results.append({
        "model": "GeoFNO",
        "l2_error": l2_geo,
        "latency_ms": lat_geo,
    })

    # --- CrossAttn ---
    torch.manual_seed(seed)
    ca_model = make_cross_attn()
    ca_model.get_network()
    _train_model(ca_model, (sdf_train, param_train), y_train, n_epochs=n_epochs)
    l2_ca = _evaluate_l2(ca_model, (sdf_test, param_test), y_test)
    lat_ca = _measure_latency(ca_model, (sdf_test[:1], param_test[:1]))
    results.append({
        "model": "CrossAttn",
        "l2_error": l2_ca,
        "latency_ms": lat_ca,
    })

    return results


def _train_model_geo(model, fno_inputs, sdf, targets, n_epochs=100, lr=1e-3, batch_size=16):
    """Train GeoFNO model."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = targets.shape[0]
    for _ in range(n_epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            pred = model(fno_inputs[idx], sdf=sdf[idx].unsqueeze(1))
            loss = ((pred - targets[idx]) ** 2).mean()
            loss.backward()
            opt.step()


def _evaluate_l2_geo(model, fno_inputs, sdf, targets):
    with torch.no_grad():
        pred = model(fno_inputs, sdf=sdf.unsqueeze(1))
        l2_err = ((pred - targets) ** 2).sum().sqrt().item()
        l2_ref = (targets**2).sum().sqrt().item()
        return l2_err / max(l2_ref, 1e-8)


def _measure_latency_geo(model, fno_inputs, sdf, n_warmup=5, n_runs=20):
    for _ in range(n_warmup):
        with torch.no_grad():
            model(fno_inputs, sdf=sdf.unsqueeze(1))
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(fno_inputs, sdf=sdf.unsqueeze(1))
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def _aggregate_results(
    per_seed_results: list[list[list[dict]]],
) -> list[dict]:
    """Aggregate across seeds with mean, std, and Wilcoxon significance."""
    n_problems = len(per_seed_results[0])
    n_models = len(per_seed_results[0][0])
    aggregated = []

    for pi in range(n_problems):
        for mi in range(n_models):
            model_name = per_seed_results[0][pi][mi]["model"]
            l2_errors = [r[pi][mi]["l2_error"] for r in per_seed_results]
            latencies = [r[pi][mi]["latency_ms"] for r in per_seed_results]

            agg = {
                "model": model_name,
                "problem_idx": pi,
                "n_seeds": len(per_seed_results),
                "l2_error_mean": float(np.mean(l2_errors)),
                "l2_error_std": float(np.std(l2_errors, ddof=1)) if len(l2_errors) > 1 else 0.0,
                "latency_ms_mean": float(np.mean(latencies)),
                "latency_ms_std": float(np.std(latencies, ddof=1)) if len(latencies) > 1 else 0.0,
                "l2_error_values": [float(v) for v in l2_errors],
            }

            # Wilcoxon test: SDFTrunk/GeoFNO vs FNO baseline
            if model_name != "FNO" and n_models >= 2:
                fno_errors = [r[pi][0]["l2_error"] for r in per_seed_results]
                try:
                    from scipy.stats import wilcoxon
                    stat, p_value = wilcoxon(fno_errors, l2_errors)
                    agg["wilcoxon_vs_fno"] = {
                        "statistic": float(stat),
                        "p_value": float(p_value),
                        "significant_005": bool(p_value < 0.05),
                    }
                except Exception:
                    pass

            aggregated.append(agg)

    return aggregated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Surrogate architecture benchmarks")
    parser.add_argument("--seeds", type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--quick", action="store_true", help="Fast smoke test (2 seeds, 10 samples)")
    args = parser.parse_args()

    n_seeds = 2 if args.quick else args.seeds
    n_train = 20 if args.quick else 80
    n_test = 5 if args.quick else 20
    n_epochs = 30 if args.quick else 100

    print("=" * 64)
    print("  Surrogate Architecture Benchmarks: FNO vs SDFTrunk vs GeoFNO")
    print("=" * 64)
    print(f"  Seeds: {n_seeds} (start={args.seed_start})  |  Train: {n_train}  |  Epochs: {n_epochs}")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_start + n_seeds))
    problems = ["cylinder", "heat_exchanger"]
    all_per_seed = []

    for i, seed in enumerate(seeds):
        print(f"  seed={seed} ({i + 1}/{n_seeds})")
        seed_results = []
        for problem in problems:
            results = _run_single_benchmark(problem, n_train, n_test, n_epochs, seed)
            seed_results.append(results)
            for r in results:
                print(f"    {problem:>16s} | {r['model']:>8s} | L2={r['l2_error']:.4f} | lat={r['latency_ms']:.1f}ms")
        all_per_seed.append(seed_results)

    print()
    print("-" * 64)
    print("  Aggregated Results (mean +/- std across seeds)")
    print("-" * 64)

    aggregated = _aggregate_results(all_per_seed)
    for problem_name in problems:
        pi = problems.index(problem_name)
        print(f"\n  [{problem_name}]")
        for agg in aggregated:
            if agg["problem_idx"] != pi:
                continue
            sig = ""
            if "wilcoxon_vs_fno" in agg:
                w = agg["wilcoxon_vs_fno"]
                sig = f"  p={w['p_value']:.4f}{'*' if w['significant_005'] else ''}"
            print(f"    {agg['model']:>8s}  L2={agg['l2_error_mean']:.4f}+/-{agg['l2_error_std']:.4f}  "
                  f"lat={agg['latency_ms_mean']:.1f}+/-{agg['latency_ms_std']:.1f}ms{sig}")

    output = {
        "config": {
            "n_seeds": n_seeds,
            "seed_start": args.seed_start,
            "n_train": n_train,
            "n_test": n_test,
            "n_epochs": n_epochs,
        },
        "problems": problems,
        "results": aggregated,
    }
    out_path = RESULTS_DIR / "surrogate_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
