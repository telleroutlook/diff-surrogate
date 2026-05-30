"""Differentiable geometry operators: B-spline, SDF, and projection.

Provides the unified ``control points -> B-spline -> differentiable SDF ->
sigmoid projection`` pipeline used across the differentiable physics ecosystem
(DiffCFD, DiffNano, OpenLithoHub).

All operators are pure PyTorch with no domain-specific coupling.
"""

from .bspline import eval_closed_cubic_bspline
from .projection import heaviside_projection, sigmoid_projection
from .sdf import differentiable_winding_number, sdf_from_curve

__all__ = [
    "differentiable_winding_number",
    "eval_closed_cubic_bspline",
    "heaviside_projection",
    "sdf_from_curve",
    "sigmoid_projection",
]
