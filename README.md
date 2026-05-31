# diff-surrogate

Unified differentiable surrogate framework for physics simulations. Shared library used by DiffCFD, DiffNano, and OpenLithoHub.

**Version:** 0.3.0

**Honesty boundaries:**
- No third-party experimental validation. All benchmarks are self-measured toy problems.
- Co-design benchmarks include quadratic coupling and B-spline geometry toy problems where decoupled methods match or outperform coupled optimization.
- GPU benchmarks still require CUDA hardware (no CPU fallback for GPU-timed benchmarks).
- Cross-validation against external solvers is framework-only: the test harness is provided but no vendored solver data is included.
- Transfer benchmarks are on toy-scale problems (grid 32, O(100) samples). Scaling to industrial datasets is not validated.

**Known stubs / unimplemented:**
- No stubs in diff-surrogate core. All surrogate classes (MLP, CNN, Ensemble), correction policies, convergence monitors, and geometry operators are functional.

## Installation

```bash
# From GitHub
pip install "diff-surrogate @ git+https://github.com/telleroutlook/diff-surrogate.git"

# Local development
pip install -e .
```

Requires Python >= 3.10 (< 3.14) and PyTorch >= 2.12 (< 3.0).

## Quick Start

### MLP Surrogate (scalar properties)

For predicting scalar properties from inputs (e.g., T,P -> density, enthalpy):

```python
from diff_surrogate import MLPSurrogate, Constraint, CorrectionPolicy

# Create surrogate with physics constraints
surrogate = MLPSurrogate(
    n_inputs=2,                                    # temperature, pressure
    properties=["density", "enthalpy", "cp"],      # outputs to predict
    hidden=64,
    n_layers=3,
    constrained={"density": Constraint.MONOTONE, "cp": Constraint.POSITIVE},
    correction_policy=CorrectionPolicy(correction_interval=20, warmup_steps=5),
)

# Predict
import torch
x = torch.tensor([[500.0, 10.0]])  # T=500K, P=10MPa
result = surrogate.predict(x)
# result = {"density": tensor([...]), "enthalpy": tensor([...]), "cp": tensor([...])}
```

### CNN Surrogate (2D field prediction)

For predicting spatial fields (e.g., mask -> velocity/pressure fields):

```python
from diff_surrogate import CNNSurrogate, CorrectionPolicy

surrogate = CNNSurrogate(
    in_channels=1,       # input field channels
    out_channels=3,      # output field channels (ux, uy, p)
    hidden=32,
    grid_size=64,
    correction_policy=CorrectionPolicy(correction_interval=10),
)
```

### Ensemble with Uncertainty

Wrap multiple surrogates for uncertainty quantification:

```python
from diff_surrogate import EnsembleSurrogate, MLPSurrogate

ensemble = EnsembleSurrogate(
    base_factory=lambda: MLPSurrogate(n_inputs=2, properties=["density"]),
    n_members=5,
)

means, uncertainties = ensemble.predict_with_uncertainty(x)
# means = {"density": tensor([...])}
# uncertainties = {"density": tensor([...])}  # std deviation across members
```

### Adaptive Correction Policy

Correction frequency adapts based on surrogate accuracy:

```python
from diff_surrogate import AdaptiveCorrectionPolicy

policy = AdaptiveCorrectionPolicy(
    min_interval=2,
    max_interval=50,
    initial_interval=10,
    warmup_steps=5,
    growth_threshold=1.5,   # error growing by 50% -> correct more often
    shrink_threshold=0.5,   # error shrinking by 50% -> correct less often
)

# After each correction, feed the error
policy.update_error(error=0.05)

# Ensemble uncertainty also adjusts interval
policy.update_uncertainty(avg_uncertainty=0.1)
```

### Convergence Monitoring

Detect optimization convergence with hybrid z-score:

```python
from diff_surrogate import ConvergenceMonitor, ConvergenceConfig

monitor = ConvergenceMonitor(ConvergenceConfig(
    window=50,
    hybrid_weight=0.5,
    early_stop_threshold=0.05,
    reduce_lr_threshold=0.1,
))

action = monitor.update(loss=0.001, step=100)
# action = ConvergenceAction.CONTINUE | EARLY_STOP | REDUCE_LR
```

### Multi-Fidelity Optimization

Alternate between fast surrogate and expensive truth solver:

```python
from diff_surrogate import optimize_multifidelity, MultiFidelityConfig, TruthMode

result = optimize_multifidelity(
    design_init=torch.randn(1, 10),
    surrogate_fn=my_surrogate_fn,       # fast approximation
    truth_fn=my_expensive_solver,       # expensive ground truth
    loss_fn=my_loss_fn,
    n_steps=300,
    config=MultiFidelityConfig(
        correction_interval=20,
        truth_mode=TruthMode.SURROGATE_GRAD,
    ),
)
# result.design, result.loss_history, result.fidelity_history, result.converged
```

### Robust Design

Compose mask + antithetic sampling + multi-corner evaluation:

```python
from diff_surrogate import robust_design_step, AntitheticConfig, CornerSpec

loss, action = robust_design_step(
    design=my_design,
    forward_fn=my_solver,
    loss_fn=my_loss,
    antithetic_config=AntitheticConfig(n_pairs=4),
    corners=[
        CornerSpec(label="nominal", weight=0.5, params={}),
        CornerSpec(label="upper", weight=0.25, params={"velocity": 1.2}),
        CornerSpec(label="lower", weight=0.25, params={"velocity": 0.8}),
    ],
    step=step,
)
```

### Adaptive Robust Optimization

O(2N+1) axial sampling with uncertainty-driven multi-corner weighting:

```python
from diff_surrogate import (
    AdaptiveRobustOptimizer,
    AdaptiveMultiCornerEvaluator,
    CornerSpec,
)

optimizer = AdaptiveRobustOptimizer(
    n_variation_dims=3,
    sigma=5.0,
    corners=[
        CornerSpec(label="nominal", weight=1.0, params={}),
        CornerSpec(label="worst", weight=1.5, params={"temp": 1.1}),
    ],
    ensemble=my_ensemble,          # optional EnsembleSurrogate
    uncertainty_weight=0.5,
)

# With ensemble: corners weighted by prediction uncertainty
loss, info = optimizer.compute_robust_loss_with_corners(
    params=my_design,
    forward_fn=my_forward,
    loss_fn=my_loss,
)
# info = {"per_corner_loss": [...], "weights": [...], "uncertainties": [...], "skipped": [...]}

# Without corners: axial + curriculum random sampling
robust_loss = optimizer.compute_robust_loss(
    params=my_design,
    forward_fn=my_perturbed_loss,
    perturbation_fn=lambda p, d: p + d,
    curriculum_frac=0.5,
)
```

### Fabricable Subspace Projection

Project continuous density fields to nearest discrete geometry:

```python
from diff_surrogate import FabricableSubspaceProjection

projector = FabricableSubspaceProjection(
    n_levels=4,
    min_cd_pixels=2,
    temperature=1.0,
)

projected = projector.project(density)       # differentiable discrete approximation
penalty = projector.projection_loss(density)  # penalty for staying near levels
```

### Budget-Aware Training

Allocate expensive solver calls across input regions:

```python
from diff_surrogate import TrainingBudget

budget = TrainingBudget(
    total_solver_calls=1000,
    n_regions=4,
    accuracy_target=0.01,
)

for region in range(4):
    n_samples = budget.allocate(region)
    if n_samples == 0:
        continue
    # Generate data using expensive solver for this region
    inputs, targets = generate_solver_data(n_samples, region=region)
    surrogate.train_surrogate(n_samples=n_samples, n_epochs=10)
    # Evaluate accuracy on held-out data
    acc = surrogate.accuracy(n_samples=50, true_solver_fn=lambda x: true_solver(x, region=region))
    budget.record_accuracy(region, acc["mse"])
    budget.record_calls(region, n_samples)
```

### Sobolev Training and Gradient Fidelity (S7.1)

Train surrogates to match not just function values but also gradient fields, improving gradient accuracy for downstream optimization:

```python
from diff_surrogate import MLPSurrogate
from diff_surrogate.training import SobolevLoss

surrogate = MLPSurrogate(n_inputs=2, properties=["density"])
optimizer = torch.optim.Adam(surrogate.parameters(), lr=1e-3)

for batch in dataloader:
    x, y_true, grad_true = batch
    y_pred = surrogate.predict(x)
    grad_pred = torch.autograd.grad(y_pred["density"].sum(), x, create_graph=True)[0]
    loss = SobolevLoss(lambda_grad=0.1)(y_pred["density"], y_true, grad_pred, grad_true)
    loss.backward()
    optimizer.step()
```

### Point Cloud Geometry Encoder (S7.2)

Encode surface point clouds into latent geometry representations for neural operator inputs:

```python
from diff_surrogate.geometry import PointCloudEncoder

encoder = PointCloudEncoder(
    n_points=256,
    d_in=3,          # xyz coordinates
    d_latent=64,
    n_heads=4,
)
latent = encoder(points)  # (B, 256, 3) -> (B, 64)
```

### Active Sampling and Multi-Fidelity Learner (S7.3)

Adaptive sampling strategy that selects training points where surrogate uncertainty is highest, with multi-fidelity oracle scheduling:

```python
from diff_surrogate import ActiveSampler, MultiFidelityLearner

sampler = ActiveSampler(
    n_candidates=1000,
    acquisition="uncertainty",  # or "gradient", "random"
    ensemble=my_ensemble,
)

learner = MultiFidelityLearner(
    surrogate=my_surrogate,
    low_fidelity_fn=cheap_solver,
    high_fidelity_fn=expensive_solver,
    budget=500,
    fidelity_ratio=0.8,  # 80% low-fidelity, 20% high-fidelity
)

# Query new points, evaluate, and retrain
new_points = sampler.select(n=50)
learner.add_observations(new_points, fidelity="high")
learner.retrain()
```

### Cross-Repo Compatibility Tests (S7.4)

Expanded cross-repository compatibility test suite verifying that diff-surrogate imports, correction policies, convergence monitors, and geometry operators work correctly when consumed by DiffCFD, DiffNano, and OpenLithoHub:

```bash
# Run cross-repo compatibility tests
pytest tests/test_cross_repo_compat.py -v
```

Couple multiple physics domains through a shared design parameter tensor:

```python
from diff_surrogate import CoDesignWorkflow, CoupledLoss

loss = CoupledLoss(
    components={"optical": optical_fn, "litho": litho_fn},
    weights={"optical": 1.0, "litho": 0.1},
)

wf = CoDesignWorkflow(
    design_params=torch.rand(32, 32),
    forward_fns={"em": em_forward, "litho": litho_forward},
    loss_fn=loss,
    coupling_fn=litho_to_em_coupling,
)
params, history = wf.run(n_steps=200)
_, baseline_history = wf.compare_baseline(n_steps=200)
report = wf.report()  # improvement_pct, coupled/baseline histories
```

### Geometry Operators

Differentiable B-spline, SDF, and projection pipeline:

```python
from diff_surrogate.geometry import (
    eval_closed_cubic_bspline,
    sdf_from_curve,
    differentiable_winding_number,
    sigmoid_projection,
    heaviside_projection,
)

# B-spline curve from control points
curve = eval_closed_cubic_bspline(control_points, t)  # (N,2) + (K,) -> (K,2)

# Signed distance field from curve (negative inside, positive outside)
sdf = sdf_from_curve(grid_x, grid_y, curve_points, softmin_temp=10.0)

# Project SDF to continuous density
density = sigmoid_projection(sdf, beta=10.0)
```

### JAX Interop

Zero-copy tensor conversion and autograd-through-JAX:

```python
from diff_surrogate.interop import j2t, t2j, wrap_jax_fn

# Zero-copy conversion
torch_tensor = j2t(jax_array)
jax_array = t2j(torch_tensor)

# Wrap JAX function with PyTorch autograd
wrapped_sin = wrap_jax_fn(jax.jit(lambda x: jnp.sin(x) ** 2))
x = torch.randn(10, requires_grad=True)
y = wrapped_sin(x)
y.sum().backward()  # gradients flow through JAX vjp
```

### Checkpointing

Save and resume long optimizations:

```python
surrogate.save_checkpoint("checkpoint.pt")
# ... later ...
surrogate.load_checkpoint("checkpoint.pt")
```

## Architecture

```
SurrogateBase (ABC)          — base class with correction lifecycle, checkpointing
├── MLPSurrogate             — scalar property prediction (density, enthalpy, cp)
│   ├── MonotoneMLP          — positive-weight MLP for monotonicity constraint
│   └── PositiveOutputMLP   — softplus-output MLP for positivity constraint
├── CNNSurrogate             — 2D field prediction (velocity, pressure, aerial image)
└── EnsembleSurrogate        — K-member ensemble with uncertainty estimation

Correction:
├── CorrectionPolicy         — fixed-interval correction scheduling
├── AdaptiveCorrectionPolicy — error-driven adaptive interval
└── CorrectionAction         — CONTINUE / CORRECT enum

Convergence:
├── ConvergenceMonitor       — hybrid z-score convergence detection
├── ConvergenceConfig        — window, thresholds, patience
└── ConvergenceAction        — CONTINUE / EARLY_STOP / REDUCE_LR

Robust Design:
├── robust_design_step       — mask + antithetic + multi-corner
├── AntitheticConfig         — paired perturbation sampling
├── CornerSpec               — operating corner definition
├── AdaptiveRobustOptimizer  — axial sampling + curriculum + uncertainty weighting
├── AdaptiveMultiCornerEvaluator — uncertainty-weighted corner evaluation
└── FabricableSubspaceProjection — differentiable discrete geometry projection

Optimization:
├── optimize_multifidelity   — surrogate + truth alternating optimization
├── MultiFidelityConfig      — correction interval, truth mode
├── MultiFidelityResult      — design, histories, convergence status
└── TruthMode                — DIFFERENTIABLE / SURROGATE_GRAD / CALIBRATION_ONLY

Co-Design:
├── CoDesignWorkflow         — multi-domain coupled optimization loop
└── CoupledLoss              — weighted sum of named loss components

Geometry (diff_surrogate.geometry):
├── eval_closed_cubic_bspline — periodic cubic B-spline evaluation
├── sdf_from_curve            — differentiable SDF with soft-min + winding number
├── differentiable_winding_number — differentiable inside/outside detection
├── sigmoid_projection        — sigmoid soft-binarisation
├── heaviside_projection      — beta-continuation projection
└── PointCloudEncoder         — surface point cloud to latent geometry (S7.2)

Training (diff_surrogate.training):
├── SobolevLoss               — joint value + gradient matching loss (S7.1)
└── SurrogateTrainer          — configurable training loop with schedulers

Active Learning (diff_surrogate.active):
├── ActiveSampler             — uncertainty/gradient-based point selection (S7.3)
├── MultiFidelityLearner      — adaptive fidelity scheduling with budget tracking (S7.3)
└── UncertaintyTriggeredSampler — conformal-coverage-driven sampling (S8.1)

Conformal (diff_surrogate.conformal):
├── SplitConformalPredictor   — distribution-free coverage prediction intervals (S8.1)
└── RiskControllingQuantile   — adaptive risk-controlled quantile (S8.1)

Operators (diff_surrogate.operators):
├── DivergenceConservingProjection — Chorin-style divergence-free projection (S8.2)
├── FluxConservingLinear      — flux-conserving linear solver (S8.2)
└── ConservationLoss          — penalizes nonzero divergence (S8.2)

Transfer (diff_surrogate.transfer):
├── MultiTaskPretrainer       — multi-task PDE pretraining (S8.3)
├── FewShotFinetuner          — few-shot target task fine-tuning (S8.3)
├── TransferBenchmark         — transfer vs from-scratch comparison (S8.3)
└── TaskGenerator             — toy PDE task generators (S8.3)

Generative (diff_surrogate.generative):
├── CandidateSampler / CandidateScorer — protocols for candidate generation (S8.4)
├── VAESampler                — VAE-based candidate sampling (S8.4)
├── EnergyBasedSampler        — energy-based refinement (S8.4)
└── GenerativePipeline        — end-to-end sample-score-select pipeline (S8.4)

Codomain Transfer (diff_surrogate.codomain):
├── CodomainBackbone          — variable-field neural operator backbone (S9.1)
├── AdapterHead               — per-domain input/output adapter (S9.1)
├── CodomainPretrainer        — masked reconstruction pretraining (S9.1)
└── CodomainTransferBenchmark — transfer vs from-scratch comparison (S9.1)

Probabilistic (diff_surrogate.probabilistic):
├── ProbabilisticSurrogate    — mean + variance prediction (S9.2)
├── EnergyScoreLoss           — energy score proper scoring rule (S9.2)
├── CRPSLoss                  — continuous ranked probability score (S9.2)
├── PNOConformalPipeline      — dual UQ: PNO + conformal calibration (S9.2)
└── PNOBenchmark              — probabilistic vs deterministic comparison (S9.2)

Decision (diff_surrogate.decision):
├── DecisionGate              — base class for uncertainty-to-decision (S9.3)
├── AcceptRejectGate          — threshold-based accept/reject (S9.3)
├── CVaRRiskBudget            — CVaR risk budget allocation (S9.3)
├── CoverageTriggeredEarlyStop — coverage-based early stopping (S9.3)
└── MultiCandidateDecision    — uncertainty-aware candidate selection (S9.3)

Interop (diff_surrogate.interop):
├── j2t / t2j                 — zero-copy JAX <-> PyTorch via dlpack
└── wrap_jax_fn / JAXFunctionWrapper — autograd-through-JAX via vjp

Supporting:
├── TrainingBudget            — allocate solver calls across regions
└── SurrogateStats            — training/correction statistics tracking
```

## Consumers

| Project | Imports from diff-surrogate | Usage |
|----------|---------------------------|-------|
| DiffCFD | CorrectionPolicy, SurrogateStats, ConvergenceAction, geometry.sdf_from_curve, decision.AcceptRejectGate, decision.CoverageTriggeredEarlyStop | SIMPLE solver correction, topology optimization convergence, airfoil SDF, CFD solution quality gating (C8.4) |
| DiffNano | CorrectionPolicy, SurrogateStats, CoDesignWorkflow, CoupledLoss, geometry.*, adaptive_robust.*, decision.* | RCWA solver correction, metalens co-design, adaptive robust optimization, B-spline geometry, nanofabrication accept/reject (N9.3) |
| OpenLithoHub | CorrectionPolicy, ConvergenceMonitor, ConvergenceConfig, ConvergenceAction, hybrid_z_score, CoDesignWorkflow, CoupledLoss, decision.* | ILT correction and convergence, lithography co-design, lithography pass/fail gating (O9.3) |

## Flagship Evidence Status

| Claim | Code | Tests | Data | Status |
|:------|:-----|:------|:-----|:-------|
| `SDFTrunkSurrogate` geometry-aware operator | `diff_surrogate/sdf_trunk.py` | `tests/test_smoke.py`, `benchmarks/` | `benchmarks/results/surrogate_benchmark_results.json` | Verified |
| `CrossAttnSurrogate` cross-attention operator | `diff_surrogate/cross_attn.py` | `tests/test_cross_attn.py` | Internal | Verified |
| FNO benchmark baseline | (consumed by DiffCFD's `FNO2D`) | `tests/test_smoke.py` | `benchmarks/results/surrogate_benchmark_results.json` | Verified |
| Co-design vs decoupled benchmark (10-seed Wilcoxon) | `benchmarks/run_codesign_benchmarks.py` | `tests/test_codesign.py` | `benchmarks/results/codesign_benchmark_results.json` | Verified |
| Adaptive robust optimization | `diff_surrogate/adaptive_robust.py` | `tests/test_adaptive_robust.py` | Internal | Verified |
| JAX interop (`wrap_jax_fn`, `j2t`/`t2j`) | `diff_surrogate/interop/` | `tests/test_interop.py`, `tests/test_interop_roundtrip.py` | N/A (functional) | Verified |
| Geometry operators (B-spline, SDF, winding number) | `diff_surrogate/geometry/` | `tests/test_geometry.py` | N/A (functional) | Verified |
| Multi-fidelity optimization | `diff_surrogate/multifidelity.py` | `tests/test_smoke.py` | Internal | Verified |
| Sobolev training (gradient fidelity) | `diff_surrogate/training/sobolev.py` | `tests/test_sobolev.py` | Internal | Verified |
| Point cloud geometry encoder | `diff_surrogate/geometry/pointcloud.py` | `tests/test_pointcloud.py` | Internal | Verified |
| Active sampling + multi-fidelity learner | `diff_surrogate/active/` | `tests/test_active.py` | Internal | Verified |
| Cross-repo compatibility (DiffCFD / DiffNano / OpenLithoHub) | `tests/test_cross_repo_compat.py` | `tests/test_cross_repo_compat.py` | N/A (functional) | Verified |
| Codomain-attention transfer backbone | `diff_surrogate/codomain.py` | `tests/test_codomain.py` | Internal | Verified |
| Probabilistic neural operator + dual UQ | `diff_surrogate/probabilistic.py` | `tests/test_probabilistic.py` | Internal | Verified |
| Decision-gate UQ (accept/reject, risk budget, early stop) | `diff_surrogate/decision.py` | `tests/test_decision.py` | Internal | Verified |

> **Note (10-seed benchmark, 2026-05-30):** The full 10-seed benchmark (80 train / 20 test / 100 epochs) shows **GeoFNO** achieving the lowest L2 error on both cylinder (0.397±0.103) and heat_exchanger (0.402±0.110) problems, closely followed by FNO. SDFTrunk is significantly worse on both problems (p=0.014), not better as the preliminary 2-seed run suggested — the earlier advantage was a small-sample artifact. CrossAttnSurrogate is newly added and not yet in the benchmark JSON.

## Compatibility

| Dependency | Version |
|:-----------|:--------|
| Python | 3.10+ (< 3.14) |
| PyTorch | 2.12+ (< 3.0) |

**Consumers:** [DiffCFD](https://github.com/OpenLithoHub/DiffCFD), [DiffNano](https://github.com/OpenLithoHub/DiffNano), [OpenLithoHub](https://github.com/OpenLithoHub/OpenLithoHub) all depend on diff-surrogate for shared surrogate classes, correction policies, convergence monitoring, geometry operators, and co-design workflow infrastructure.

## When to use co-design

Co-design via differentiable coupling is a powerful technique, but it is not always the right choice. The benchmarks in this repository include toy problems (quadratic coupling, B-spline geometry) where decoupled methods matched or outperformed coupled optimization, alongside flagship physics problems (metalens DFM, flow-litho) where co-design delivered clear improvements. The following checklist summarizes the decision boundary.

### USE co-design when

- **Domains share design variables.** A single parameter tensor (e.g., mask density, B-spline control points) feeds into multiple forward models, and gradients from each domain flow back to the same parameters.
- **Gradients flow across domain boundaries.** The output of one domain physically feeds into another (e.g., lithography contour determines EM boundary conditions), creating genuine cross-domain gradient paths.
- **There are real trade-offs between domain objectives.** The optimal design for domain A actively harms domain B (e.g., sharp optical features that are unprintable), requiring a Pareto-optimal compromise.
- **The coupling is non-trivial.** Complex, non-convex forward models with high-dimensional design spaces where sequential optimization gets trapped in single-domain local minima.
- **Manufacturing-aware design is required.** Embedding fabrication constraints during optimization avoids the design-then-verify cycle.

### DO NOT use co-design when

- **Domains are independent.** No shared variables, no output coupling, no cross-domain constraints. Running them together adds overhead without benefit.
- **One domain dominates the objective.** If one domain's loss is orders of magnitude larger or has a much steeper landscape, the shared optimizer will effectively ignore the other domain. Weight tuning rarely fixes this robustly.
- **The coupling is weak or additive.** Simple quadratic coupling or additive penalty terms are handled well by alternating optimization without the cost of joint gradient computation.
- **Gradient conflict is severe.** When domains have strongly opposing gradients (one pushes a variable left, the other pushes it right), the coupled optimizer wastes steps fighting itself. Decoupled methods make faster per-domain progress.
- **One domain is computationally expensive.** Coupled optimization requires all domain forward passes at every step. If one domain is costly (e.g., full-wave EM), a multi-fidelity or periodic-coupling strategy may be more efficient.

### Key insight from benchmarks

On the quadratic coupling and B-spline geometry toy problems, decoupled methods achieved lower final loss than the coupled strategy in several configurations. For example, on the quadratic benchmark (200 steps, Adam lr=0.01, seed=42):

| Strategy | Final Loss | Best Loss |
|----------|-----------|-----------|
| Coupled | 7.903 | 7.903 |
| Decoupled-sequential | 3.512 | 1.223 |
| Decoupled-alternating | 6.437 | 1.732 |

This is expected. These problems have simple coupling structure (quadratic penalty, SDF matching) where alternating optimization converges well. The coupled optimizer spends steps resolving gradient conflicts that decoupled methods avoid by construction. The real advantage of co-design emerges on high-dimensional physics problems with complex coupling (metalens DFM: 30--50% reduction in lithographic EPE; flow-litho: wider process windows). See `benchmarks/CODESIGN_PREPRINT.md` Section 4 for the full discussion.

## Benchmarks & Reproducibility

### Co-Design Benchmark (Multi-Seed)

Run the co-design vs decoupled benchmark across 10 seeds with Wilcoxon significance tests:

```bash
make flagship          # 10 seeds, full report
make flagship-ci       # 3 seeds, CI smoke test
```

Or directly:

```bash
python benchmarks/run_codesign_benchmarks.py                # 10 seeds (default)
python benchmarks/run_codesign_benchmarks.py --seeds 20     # custom seed count
python benchmarks/run_codesign_benchmarks.py --seed-start 0 # start from seed 0
```

Results are written to `benchmarks/results/`. The full analysis is in `benchmarks/CODESIGN_PREPRINT.md`.

### Conformal Prediction (S8.1)

Split conformal predictor with distribution-free coverage guarantees. Risk-controlling quantile prediction for calibrated uncertainty bands. Integrated with `UncertaintyTriggeredSampler` for active learning driven by conformal coverage.

```python
from diff_surrogate.conformal import SplitConformalPredictor, RiskControllingQuantile
from diff_surrogate.active import UncertaintyTriggeredSampler

# Calibrate on held-out data
predictor = SplitConformalPredictor(base_model=my_surrogate, coverage=0.95)
predictor.calibrate(x_cal, y_cal)

# Get coverage-guaranteed prediction intervals
y_mean, y_lower, y_upper = predictor.predict(x_test)

# Risk-controlling quantile for adaptive coverage
rcq = RiskControllingQuantile(target_risk=0.05)
rcq.fit(scores)

# Uncertainty-triggered active sampling
sampler = UncertaintyTriggeredSampler(
    predictor=predictor,
    threshold=0.1,  # width threshold to trigger new sample
)
new_points = sampler.select(x_pool)
```

### Structure-Preserving Operators (S8.2)

Conservation-law-preserving linear algebra primitives. Divergence-conserving projection (Chorin-style) and flux-conserving linear solvers prevent unphysical solution drift. Conservation loss monitors residual divergence.

```python
from diff_surrogate.operators import (
    DivergenceConservingProjection,
    FluxConservingLinear,
    ConservationLoss,
)

# Chorin-style projection that conserves divergence by construction
projection = DivergenceConservingProjection(grid=(64, 64))
u_div_free, p = projection.project(velocity_field)

# Flux-conserving linear solve
solver = FluxConservingLinear(grid=(64, 64))
x = solver.solve(rhs, boundary_conditions)

# Monitor conservation violation
cons_loss = ConservationLoss()
loss = cons_loss(predicted_field)  # penalizes nonzero divergence
```

Structure-preserving operators improve accuracy on out-of-distribution geometries where standard operators accumulate conservation errors.

### PDE Pretraining + Transfer (S8.3)

Multi-task PDE pretraining followed by few-shot fine-tuning on target problems. Includes toy PDE task generators for pretraining data.

```python
from diff_surrogate.transfer import (
    MultiTaskPretrainer,
    FewShotFinetuner,
    TransferBenchmark,
)
from diff_surrogate.transfer import TaskGenerator

# Generate pretraining tasks
tasks = [
    TaskGenerator("poisson", grid=32),
    TaskGenerator("diffusion", grid=32),
    TaskGenerator("advection", grid=32),
]

# Multi-task pretraining
pretrainer = MultiTaskPretrainer(
    model=my_operator,
    tasks=tasks,
    n_epochs_per_task=50,
)
pretrained_model = pretrainer.train()

# Few-shot fine-tuning on target problem
finetuner = FewShotFinetuner(
    model=pretrained_model,
    n_shot=10,
    lr=1e-4,
    freeze_encoder=True,
)
finetuned_model = finetuner.train(x_support, y_support)

# Benchmark transfer vs from-scratch
bench = TransferBenchmark(
    model_factory=lambda: MyOperator(),
    pretrain_tasks=tasks,
    target_task=TaskGenerator("ns_cavity", grid=32),
    n_shot=10,
)
results = bench.run()
```

### Generative Prior Interface (S8.4)

Protocol-based interface for generative model candidate sampling. Provides `CandidateSampler`/`CandidateScorer` protocols, built-in VAE and energy-based samplers, and end-to-end `GenerativePipeline`.

```python
from diff_surrogate.generative import (
    CandidateSampler,
    CandidateScorer,
    VAESampler,
    EnergyBasedSampler,
    GenerativePipeline,
)

# VAE-based candidate generation
vae_sampler = VAESampler(
    latent_dim=32,
    encoder_dims=[128, 64],
    decoder_dims=[64, 128],
    output_shape=(64, 64),
)
candidates = vae_sampler.sample(n=100)

# Energy-based refinement
ebm_scorer = EnergyBasedSampler(
    energy_fn=physics_energy,  # differentiable physics objective
    n_steps=50,
    step_size=0.01,
)
refined = ebm_scorer.refine(candidates)

# End-to-end pipeline
pipeline = GenerativePipeline(
    sampler=vae_sampler,
    scorer=ebm_scorer,
    n_candidates=100,
    top_k=10,
)
top_designs = pipeline.generate()  # top-k candidates by score
```

### Codomain-Attention Transfer Backbone (S9.1)

Variable-field neural operator backbone with masked reconstruction pretraining. Supports adding and removing physics fields for cross-domain transfer between problems with different input/output channel counts.

```python
from diff_surrogate.codomain import (
    CodomainBackbone,
    AdapterHead,
    CodomainPretrainer,
    CodomainTransferBenchmark,
)

# Backbone accepts variable field counts
backbone = CodomainBackbone(
    d_model=64,
    n_heads=4,
    n_layers=4,
    grid_size=32,
)

# Adapter heads for source and target domains (different field counts)
source_head = AdapterHead(in_fields=3, out_fields=3, d_model=64)
target_head = AdapterHead(in_fields=2, out_fields=4, d_model=64)

# Masked reconstruction pretraining on source domain
pretrainer = CodomainPretrainer(
    backbone=backbone,
    head=source_head,
    mask_ratio=0.3,
    n_epochs=100,
)
pretrainer.train(source_dataloader)

# Transfer to target domain (different field count)
transfer_bench = CodomainTransferBenchmark(
    backbone=backbone,
    source_head=source_head,
    target_head=target_head,
    n_shot=20,
)
results = transfer_bench.run(target_dataloader)
# results = {"transfer_loss": ..., "from_scratch_loss": ..., "improvement_pct": ...}
```

### Probabilistic Neural Operator (S9.2)

Probabilistic surrogate with proper scoring rule training (energy score, CRPS). Dual uncertainty quantification: intrinsic PNO uncertainty plus conformal prediction wrappers.

```python
from diff_surrogate.probabilistic import (
    ProbabilisticSurrogate,
    EnergyScoreLoss,
    CRPSLoss,
    PNOConformalPipeline,
    PNOBenchmark,
)

# Probabilistic surrogate predicts mean + variance
surrogate = ProbabilisticSurrogate(
    in_channels=3,
    out_channels=2,
    grid_size=64,
    n_layers=4,
    d_model=64,
)

# Train with proper scoring rule
optimizer = torch.optim.Adam(surrogate.parameters(), lr=1e-3)
for batch in dataloader:
    x, y = batch
    y_mean, y_std = surrogate.predict(x)
    loss = EnergyScoreLoss()(y_mean, y_std, y)
    loss.backward()
    optimizer.step()

# Dual UQ: PNO uncertainty + conformal calibration
pipeline = PNOConformalPipeline(
    surrogate=surrogate,
    coverage=0.95,
)
pipeline.calibrate(x_cal, y_cal)
y_mean, y_lower, y_upper = pipeline.predict(x_test)

# Benchmark probabilistic vs deterministic
bench = PNOBenchmark(
    probabilistic_factory=lambda: ProbabilisticSurrogate(in_channels=3, out_channels=2, grid_size=64),
    deterministic_factory=lambda: CNNSurrogate(in_channels=3, out_channels=2, grid_size=64),
)
results = bench.run(test_dataloader)
```

### Decision-Gate UQ (S9.3)

Turn calibrated uncertainty into actionable decisions: accept/reject predictions, allocate risk budgets, trigger early stopping based on coverage, and select among multiple candidate designs.

```python
from diff_surrogate.decision import (
    DecisionGate,
    AcceptRejectGate,
    CVaRRiskBudget,
    CoverageTriggeredEarlyStop,
    MultiCandidateDecision,
)

# Accept predictions only when uncertainty is below threshold
gate = AcceptRejectGate(
    uncertainty_threshold=0.05,
    fallback_value=0.0,
)
accepted, mask = gate(predictions, uncertainties)
# mask[i] = True where uncertainty[i] < threshold

# CVaR risk budget allocation across design candidates
risk_budget = CVaRRiskBudget(
    alpha=0.05,       # tail probability
    total_budget=100,
)
allocations = risk_budget.allocate(candidate_losses)

# Early stop when conformal coverage degrades
early_stop = CoverageTriggeredEarlyStop(
    target_coverage=0.95,
    patience=5,
)
action = early_stop.update(current_coverage=0.93, step=epoch)

# Multi-candidate selection with uncertainty-aware scoring
selector = MultiCandidateDecision(
    n_candidates=10,
    uncertainty_weight=0.5,
)
best_idx, scores = selector.select(candidate_predictions, candidate_uncertainties)
```

DecisionGate components are consumed by DiffNano N9.3 (nanofabrication accept/reject), OpenLithoHub O9.3 (lithography pass/fail), and DiffCFD C8.4 (CFD solution quality gating).

## Competitive Positioning

**What it is:** A unified differentiable surrogate framework shared by DiffCFD, DiffNano, and OpenLithoHub — the cross-domain reusable backbone of a multi-physics co-design toolkit.

**Where it leads:**
- **Cross-domain reusability:** The only surrogate library serving lithography, electromagnetics, CFD, and co-design from one codebase. Most alternatives (FNO, DeepONet, GAOT) target single domains.
- **Calibrated uncertainty:** Split conformal prediction + risk-controlling quantiles give distribution-free coverage guarantees — rare in physics surrogate libraries. Dual UQ (PNO + conformal) in S9.2 adds proper scoring rule training on top.
- **Structure-preserving operators:** Conservation-law-preserving projections and flux-conserving solvers prevent unphysical drift on out-of-distribution geometries.
- **Variable-field transfer:** Codomain-attention backbone (S9.1) enables cross-domain transfer between problems with different physics field counts — uncommon in neural operator libraries.
- **Decision-ready UQ:** Decision-gate components (S9.3) turn calibrated uncertainty into accept/reject, risk budget, and early stop decisions, consumed by three downstream projects.

**Where it lags (honest assessment):**
- **Scale:** Toy-to-medium problem sizes, single GPU/CPU. Not comparable to foundation-model-scale operators (GAOT/NeurIPS25, Poseidon) trained on industrial 3D datasets.
- **Data:** No large-scale pretraining data. Benchmarks are self-measured toy problems.
- **Validation:** Self-tests + analytical solutions + cross-repo numerical checks. No third-party experimental or production validation.

**Bottom line:** Methodologically current (2024-2026 techniques) but structurally a research prototype. Value is in the cross-domain integration layer, not in beating SOTA on any single benchmark.

## Related Work

| Method | Venue | Key Idea | Relation to diff-surrogate |
|:-------|:------|:---------|:---------------------------|
| Geo-FNO | JMLR 2023 | Geometry-preserving Fourier Neural Operator | diff-surrogate's SDF-trunk geometry module (L2) follows similar geometry-aware operator principles |
| GAOT | NeurIPS 2025, arXiv:2505.18781 | Geometry-aware operator transformer | Independent work on geometry-aware neural operators; diff-surrogate focuses on multi-physics co-design rather than operator architecture |
| GINOT | CMAME 2025 | SDF-trunk geometry-informed operator | SDF-based geometry representation for operators; diff-surrogate implements SDF geometry primitives in `diff_surrogate.geometry` |
| GAOT v4 | NeurIPS 2025, arXiv:2505.18781 | Multi-scale attention geometry-aware operator transformer | Latest iteration of GAOT with multi-scale attention; extends geometry-aware operator design |
| GINOT (2026) | CMAME 2026 | Surface point-cloud encoding + cross-attention geometry injection | Updated GINOT with surface point-cloud geometry encoding and cross-attention injection |
| DNOT | Eng. with Computers 42:60, 2026 | Feature-diffusion enhanced neural operator transformer | Feature-diffusion mechanism for improved neural operator accuracy |
| DD-DeepONet | Eng. Appl. Artif. Intell. 2026 | Domain decomposition DeepONet | Domain decomposition strategy for scalable DeepONet inference |
| Schwarz Neural Inference | arXiv:2504.00510 v2, 2026-02 | Local→global domain decomposition operator learning | Schwarz-type alternating decomposition for neural operator training on complex domains |
| CoDA-NO | NeurIPS 2024, arXiv:2403.12553 | Codomain-attention neural operator for variable-field transfer | CodomainBackbone (S9.1) follows codomain-attention design for cross-field transfer |
| Probabilistic Neural Operator | arXiv:2502.12902 | Probabilistic training with proper scoring rules | ProbabilisticSurrogate (S9.2) uses energy score and CRPS losses from this work |

## License

Apache License 2.0
