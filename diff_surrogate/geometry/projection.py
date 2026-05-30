"""Sigmoid and Heaviside projection operators for density/SDF binarisation.

Provides the standard sigmoid and smoothed Heaviside (beta-continuation)
projections used in topology optimisation and mask synthesis.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["heaviside_projection", "sigmoid_projection"]


def sigmoid_projection(
    sdf: Tensor,
    beta: float = 10.0,
) -> Tensor:
    """Sigmoid soft-binarisation of a signed distance field.

    Parameters
    ----------
    sdf : Tensor
        Signed distance field (negative inside, positive outside).
    beta : float
        Sharpness parameter.  Higher → sharper transition.

    Returns
    -------
    density : Tensor
        Continuous density in ``(0, 1)``.  ``~1`` inside, ``~0`` outside.
    """
    return torch.sigmoid(-beta * sdf)


def heaviside_projection(
    sdf: Tensor,
    beta: float = 10.0,
) -> Tensor:
    """Smoothed Heaviside (beta-continuation) projection.

    Uses the standard topology-optimisation smoothed Heaviside:

    .. math::

        H_\\beta(\\phi) = \\frac{\\tanh(\\beta \\cdot \\phi) + 1}{2}

    Unlike :func:`sigmoid_projection`, this formulation allows continuation
    (gradually increasing ``beta``) for improved convergence.

    Parameters
    ----------
    sdf : Tensor
        Signed distance field (negative inside, positive outside).
    beta : float
        Sharpness / continuation parameter.

    Returns
    -------
    density : Tensor
        Continuous density in ``(0, 1)``.
    """
    return (torch.tanh(beta * sdf) + 1.0) / 2.0
