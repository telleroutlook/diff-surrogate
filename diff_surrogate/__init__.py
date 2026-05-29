__version__ = "0.1.0"

from .base import SurrogateBase, CorrectionPolicy, AdaptiveCorrectionPolicy, SurrogateStats, CorrectionAction
from .budget import TrainingBudget
from .cnn import CNNSurrogate
from .convergence import (
    ConvergenceAction,
    ConvergenceConfig,
    ConvergenceMonitor,
    hybrid_z_score,
)
from .ensemble import EnsembleSurrogate
from .mlp import MLPSurrogate, MonotoneMLP, PositiveOutputMLP
from .multifidelity import (
    MultiFidelityConfig,
    MultiFidelityResult,
    optimize_multifidelity,
)
from .robust_design import (
    AntitheticConfig,
    CornerSpec,
    robust_design_step,
)
from .trainer import SurrogateTrainer

__all__ = [
    "SurrogateBase",
    "CorrectionPolicy",
    "AdaptiveCorrectionPolicy",
    "SurrogateStats",
    "CorrectionAction",
    "TrainingBudget",
    "CNNSurrogate",
    "ConvergenceAction",
    "ConvergenceConfig",
    "ConvergenceMonitor",
    "hybrid_z_score",
    "EnsembleSurrogate",
    "MLPSurrogate",
    "MonotoneMLP",
    "PositiveOutputMLP",
    "MultiFidelityConfig",
    "MultiFidelityResult",
    "AntitheticConfig",
    "CornerSpec",
    "optimize_multifidelity",
    "robust_design_step",
    "SurrogateTrainer",
]
