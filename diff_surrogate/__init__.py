__version__ = "0.3.0"

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
from .codomain import (
    AdapterHead,
    CodomainBackbone,
    CodomainPretrainer,
    CodomainTransferBenchmark,
)
from .conformal import (
    RiskControllingQuantile,
    SplitConformalPredictor,
    coverage_score,
)
from .convergence import (
    ConvergenceAction,
    ConvergenceConfig,
    ConvergenceMonitor,
    hybrid_z_score,
)
from .cross_attn import CrossAttnSurrogate
from .decision import (
    AcceptRejectGate,
    CoverageTriggeredEarlyStop,
    CVaRRiskBudget,
    DecisionGate,
    DecisionVerdict,
    MultiCandidateDecision,
)
from .ensemble import EnsembleSurrogate
from .generative import (
    CandidateSampler,
    CandidateScorer,
    CompositeScorer,
    EnergyBasedSampler,
    GenerativePipeline,
    SurrogateScorer,
    VAESampler,
)
from .mlp import Constraint, MLPSurrogate, MonotoneMLP, PositiveOutputMLP
from .multifidelity import (
    MultiFidelityConfig,
    MultiFidelityResult,
    TruthMode,
    optimize_multifidelity,
)
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
from .probabilistic import (
    CRPSLoss,
    DistributionHead,
    EnergyScoreLoss,
    PNOBenchmark,
    PNOConformalPipeline,
    ProbabilisticSurrogate,
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
    "AcceptRejectGate",
    "AdapterHead",
    "AdaptiveCorrectionPolicy",
    "AdaptiveMultiCornerEvaluator",
    "AdaptiveRobustOptimizer",
    "AntitheticConfig",
    "CNNSurrogate",
    "CRPSLoss",
    "CVaRRiskBudget",
    "CandidateSampler",
    "CandidateScorer",
    "CoDesignWorkflow",
    "CodomainBackbone",
    "CodomainPretrainer",
    "CodomainTransferBenchmark",
    "CompositeScorer",
    "ConservationLoss",
    "Constraint",
    "ConvergenceAction",
    "ConvergenceConfig",
    "ConvergenceMonitor",
    "CornerSpec",
    "CorrectionAction",
    "CorrectionPolicy",
    "CoupledLoss",
    "CoverageTriggeredEarlyStop",
    "CrossAttnSurrogate",
    "DecisionGate",
    "DecisionVerdict",
    "DistributionHead",
    "DivergenceConservingProjection",
    "EnergyBasedSampler",
    "EnergyScoreLoss",
    "EnsembleSurrogate",
    "FabricableSubspaceProjection",
    "FewShotFinetuner",
    "FluxConservingLinear",
    "GenerativePipeline",
    "MLPSurrogate",
    "MonotoneMLP",
    "MultiCandidateDecision",
    "MultiFidelityConfig",
    "MultiFidelityResult",
    "MultiTaskPretrainer",
    "PDENet",
    "PNOBenchmark",
    "PNOConformalPipeline",
    "PositiveOutputMLP",
    "ProbabilisticSurrogate",
    "RiskControllingQuantile",
    "SDFTrunkSurrogate",
    "SobolevLoss",
    "SplitConformalPredictor",
    "StructurePreservingEncoder",
    "SurrogateBase",
    "SurrogateScorer",
    "SurrogateStats",
    "SurrogateTrainer",
    "TrainingBudget",
    "TransferBenchmark",
    "TruthMode",
    "VAESampler",
    "axial_samples",
    "correlated_perturbation",
    "coverage_score",
    "discrete_divergence",
    "discrete_gradient",
    "geometry",
    "gradient_fidelity_score",
    "hybrid_z_score",
    "interop",
    "optimize_multifidelity",
    "robust_design_step",
    "task_advection_1d",
    "task_diffusion_2d",
    "task_poisson_1d",
    "task_reaction_diffusion_1d",
]
