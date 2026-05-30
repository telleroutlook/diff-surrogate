"""Differentiable geometry operators: B-spline, SDF, projection, and point cloud.

Provides the unified ``control points -> B-spline -> differentiable SDF ->
sigmoid projection`` pipeline used across the differentiable physics ecosystem
(DiffCFD, DiffNano, OpenLithoHub), plus multi-scale neighborhood attention
encoders for point clouds and irregular meshes.

All operators are pure PyTorch with no domain-specific coupling.
"""

from .bspline import eval_closed_cubic_bspline
from .pointcloud import IrregularMeshEncoder, PointCloudGeometry
from .projection import heaviside_projection, sigmoid_projection
from .sdf import differentiable_winding_number, sdf_from_curve

__all__ = [
    "IrregularMeshEncoder",
    "PointCloudGeometry",
    "differentiable_winding_number",
    "eval_closed_cubic_bspline",
    "heaviside_projection",
    "sdf_from_curve",
    "sigmoid_projection",
]
