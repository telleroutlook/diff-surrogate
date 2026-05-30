# diff-surrogate

Unified differentiable surrogate framework for physics simulations. Shared library used by DiffCFD, DiffNano, and OpenLithoHub.

**Honesty boundaries:**
- No third-party experimental validation. All benchmarks are self-measured toy problems.
- Co-design benchmarks include quadratic coupling and B-spline geometry toy problems where decoupled methods match or outperform coupled optimization.

## Installation

```bash
# From GitHub
pip install "diff-surrogate @ git+https://github.com/telleroutlook/diff-surrogate.git"

# Local development
pip install -e .
```

Requires Python >= 3.10 (< 3.14) and PyTorch >= 2.0 (< 3.0).

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

### Co-Design API

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
└── heaviside_projection      — beta-continuation projection

Interop (diff_surrogate.interop):
├── j2t / t2j                 — zero-copy JAX <-> PyTorch via dlpack
└── wrap_jax_fn / JAXFunctionWrapper — autograd-through-JAX via vjp

Supporting:
├── TrainingBudget            — allocate solver calls across regions
├── SurrogateTrainer          — configurable training loop with schedulers
└── SurrogateStats            — training/correction statistics tracking
```

## Consumers

| Project | Imports from diff-surrogate | Usage |
|----------|---------------------------|-------|
| DiffCFD | CorrectionPolicy, SurrogateStats, ConvergenceAction, geometry.sdf_from_curve | SIMPLE solver correction, topology optimization convergence, airfoil SDF |
| DiffNano | CorrectionPolicy, SurrogateStats, CoDesignWorkflow, CoupledLoss, geometry.*, adaptive_robust.* | RCWA solver correction, metalens co-design, adaptive robust optimization, B-spline geometry |
| OpenLithoHub | CorrectionPolicy, ConvergenceMonitor, ConvergenceConfig, ConvergenceAction, hybrid_z_score, CoDesignWorkflow, CoupledLoss | ILT correction and convergence, lithography co-design |

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

## Related Work

| Method | Venue | Key Idea | Relation to diff-surrogate |
|:-------|:------|:---------|:---------------------------|
| Geo-FNO | JMLR 2023 | Geometry-preserving Fourier Neural Operator | diff-surrogate's SDF-trunk geometry module (L2) follows similar geometry-aware operator principles |
| GAOT | NeurIPS 2025, arXiv:2505.18781 | Geometry-aware operator transformer | Independent work on geometry-aware neural operators; diff-surrogate focuses on multi-physics co-design rather than operator architecture |
| GINOT | CMAME 2025 | SDF-trunk geometry-informed operator | SDF-based geometry representation for operators; diff-surrogate implements SDF geometry primitives in `diff_surrogate.geometry` |

## License

Apache License 2.0
