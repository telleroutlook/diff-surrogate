__version__ = "0.2.0"

from . import geometry, interop
from .adaptive_corner import AdaptiveMultiCornerEvaluator
from .adaptive_robust import (
    AdaptiveRobustOptimizer,
    FabricableSubspaceProjection,
    axial_samples,
    correlated_perturbation,
)
from .base import (
    AdaptiveCorrectionPolicy,
    CorrectionAction,
    CorrectionPolicy,
    SurrogateBase,
    SurrogateStats,
)
from .budget import TrainingBudget
from .cnn import CNNSurrogate
from .codesign import CoDesignWorkflow, CoupledLoss
from .convergence import (
    ConvergenceAction,
    ConvergenceConfig,
    ConvergenceMonitor,
    hybrid_z_score,
)
from .conformal import (
    RiskControllingQuantile,
    SplitConformalPredictor,
    coverage_score,
)
from .cross_attn import CrossAttnSurrogate
from .ensemble import EnsembleSurrogate
from .mlp import Constraint, MLPSurrogate, MonotoneMLP, PositiveOutputMLP
from .pretraining import (
    FewShotFinetuner,
    MultiTaskPretrainer,
    PDENet,
    TransferBenchmark,
    task_advection_1d,
    task_diffusion_2d,
    task_poisson_1d,
    task_reaction_diffusion_1d,
)
from .multifidelity import (
    MultiFidelityConfig,
    MultiFidelityResult,
    TruthMode,
    optimize_multifidelity,
)
from .robust_design import (
    AntitheticConfig,
    CornerSpec,
    robust_design_step,
)
from .sdf_trunk import SDFTrunkSurrogate
from .structure import (
    ConservationLoss,
    DivergenceConservingProjection,
    FluxConservingLinear,
    StructurePreservingEncoder,
    discrete_divergence,
    discrete_gradient,
)
from .trainer import SobolevLoss, SurrogateTrainer, gradient_fidelity_score

__all__ = [
    "AdaptiveCorrectionPolicy",
    "AdaptiveMultiCornerEvaluator",
    "AdaptiveRobustOptimizer",
    "AntitheticConfig",
    "CNNSurrogate",
    "CoDesignWorkflow",
    "Constraint",
    "ConvergenceAction",
    "ConvergenceConfig",
    "ConvergenceMonitor",
    "CornerSpec",
    "CorrectionAction",
    "RiskControllingQuantile",
    "SplitConformalPredictor",
    "CorrectionPolicy",
    "CoupledLoss",
    "CrossAttnSurrogate",
    "EnsembleSurrogate",
    "FabricableSubspaceProjection",
    "MLPSurrogate",
    "MonotoneMLP",
    "MultiTaskPretrainer",
    "PDENet",
    "FewShotFinetuner",
    "TransferBenchmark",
    "task_advection_1d",
    "task_diffusion_2d",
    "task_poisson_1d",
    "task_reaction_diffusion_1d",
    "MultiFidelityConfig",
    "MultiFidelityResult",
    "PositiveOutputMLP",
    "SDFTrunkSurrogate",
    "SobolevLoss",
    "ConservationLoss",
    "DivergenceConservingProjection",
    "FluxConservingLinear",
    "StructurePreservingEncoder",
    "discrete_divergence",
    "discrete_gradient",
    "SurrogateBase",
    "SurrogateStats",
    "SurrogateTrainer",
    "TrainingBudget",
    "TruthMode",
    "axial_samples",
    "correlated_perturbation",
    "coverage_score",
    "geometry",
    "gradient_fidelity_score",
    "hybrid_z_score",
    "interop",
    "optimize_multifidelity",
    "robust_design_step",
]
