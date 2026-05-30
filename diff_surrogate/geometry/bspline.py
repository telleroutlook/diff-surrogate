"""Closed uniform cubic B-spline evaluation.

Evaluates a periodic cubic B-spline defined by control points at arbitrary
parameter values.  Vectorised over ``t`` for efficient curve sampling.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["eval_closed_cubic_bspline"]


def eval_closed_cubic_bspline(
    control_points: Tensor,
    t: Tensor,
) -> Tensor:
    """Evaluate a closed (periodic) uniform cubic B-spline.

    Parameters
    ----------
    control_points : Tensor, shape ``(N, 2)``
        Control-point coordinates.
    t : Tensor, shape ``(K,)``
        Parameter values in ``[0, 1)``.  The curve wraps at ``t = 0 == 1``.

    Returns
    -------
    curve : Tensor, shape ``(K, 2)``
        Evaluated curve points.

    Notes
    -----
    Standard uniform cubic basis functions on knot span ``i`` blending points
    ``i, i+1, i+2, i+3``:

    .. math::

        B_0 = (1-f)^3 / 6, \\quad
        B_1 = (3f^3 - 6f^2 + 4) / 6, \\quad
        B_2 = (-3f^3 + 3f^2 + 3f + 1) / 6, \\quad
        B_3 = f^3 / 6
    """
    N = control_points.shape[0]

    t_scaled = t * N
    segments = t_scaled.floor().long() % N
    fracs = t_scaled - t_scaled.floor()

    idx0 = segments % N
    idx1 = (segments + 1) % N
    idx2 = (segments + 2) % N
    idx3 = (segments + 3) % N

    p0 = control_points[idx0]
    p1 = control_points[idx1]
    p2 = control_points[idx2]
    p3 = control_points[idx3]

    f = fracs.unsqueeze(-1)
    curve = (
        (1 - f) ** 3 / 6 * p0
        + (3 * f**3 - 6 * f**2 + 4) / 6 * p1
        + (-3 * f**3 + 3 * f**2 + 3 * f + 1) / 6 * p2
        + f**3 / 6 * p3
    )
    return curve
