# diff-surrogate

Unified differentiable surrogate framework for physics simulations. Shared library used by DiffCFD, DiffNano, and OpenLithoHub.

## Installation

```bash
# From GitHub
pip install "diff-surrogate @ git+https://github.com/telleroutlook/diff-surrogate.git"

# Local development
pip install -e .
```

Requires Python >= 3.10 and PyTorch >= 2.0.

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
    growth_threshold=1.5,   # error growing by 50% -> correct more often
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
from diff_surrogate import optimize_multifidelity, MultiFidelityConfig

result = optimize_multifidelity(
    design_init=torch.randn(1, 10),
    surrogate_fn=my_surrogate_fn,       # fast approximation
    truth_fn=my_expensive_solver,       # expensive ground truth
    loss_fn=my_loss_fn,
    n_steps=300,
    config=MultiFidelityConfig(correction_interval=20),
)
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

### Checkpointing

Save and resume long optimizations:

```python
surrogate.save_checkpoint("checkpoint.pt")
# ... later ...
surrogate.load_checkpoint("checkpoint.pt")
```

## Architecture

```
SurrogateBase (ABC)
├── CNNSurrogate       — 2D field prediction (velocity, pressure, aerial image)
├── MLPSurrogate       — scalar property prediction (density, enthalpy, cp)
└── EnsembleSurrogate  — K-member ensemble with uncertainty estimation

Supporting:
├── CorrectionPolicy / AdaptiveCorrectionPolicy — when to call truth solver
├── ConvergenceMonitor — hybrid z-score convergence detection
├── TrainingBudget — allocate solver calls across regions
├── robust_design_step — mask + antithetic + multi-corner
└── optimize_multifidelity — surrogate + truth alternating optimization
```

## Consumers

| Project | Surrogate Type | Usage |
|----------|---------------|-------|
| DiffCFD | CorrectionPolicy, ConvergenceMonitor | SIMPLE solver correction and convergence |
| DiffNano | CorrectionPolicy, SurrogateStats | RCWA solver correction |
| OpenLithoHub | CorrectionPolicy, ConvergenceMonitor | ILT correction and convergence |

## License

Apache License 2.0
