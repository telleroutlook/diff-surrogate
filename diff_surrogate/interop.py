"""JAX <-> PyTorch zero-copy interop via dlpack.

Provides:
- ``j2t(jax_array)`` -- JAX array to PyTorch tensor (zero-copy via dlpack)
- ``t2j(torch_tensor)`` -- PyTorch tensor to JAX array (zero-copy via dlpack)
- ``JAXFunctionWrapper(jax_fn)`` -- Wrap a JAX function as a PyTorch-compatible
  callable with full autograd integration (forward + backward via JAX vjp).

All JAX imports are lazy. The module is importable without JAX installed;
calling any function raises ``ImportError`` with a clear message.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def _require_jax():
    """Lazy import JAX; raise with a helpful message if not installed."""
    try:
        import jax
        import jax.numpy as jnp

        return jax, jnp
    except ImportError as e:
        raise ImportError(
            "JAX is required for diff_surrogate.interop but is not installed. "
            "Install it with: pip install jax jaxlib"
        ) from e


def j2t(x) -> torch.Tensor:
    """Convert a JAX array to a PyTorch tensor via dlpack (zero-copy).

    Uses the ``__dlpack__`` protocol supported by JAX >= 0.4 and
    PyTorch >= 1.10.

    Args:
        x: A JAX array (``jax.Array``).

    Returns:
        A PyTorch tensor sharing the same underlying memory.
    """
    _require_jax()
    return torch.from_dlpack(x)


def t2j(x: torch.Tensor):
    """Convert a PyTorch tensor to a JAX array via dlpack (zero-copy).

    Ensures contiguous memory layout and detaches gradients before
    conversion since JAX's dlpack import only supports compact
    (non-broadcast) strides and PyTorch cannot export grad-tracking
    tensors.

    Args:
        x: A PyTorch tensor. Must be on CPU or GPU with supported dtype.

    Returns:
        A JAX array sharing the same underlying memory.
    """
    jax_mod, _ = _require_jax()
    # Ensure contiguous layout -- JAX dlpack rejects broadcast strides.
    # Detach gradients -- PyTorch cannot export grad-tracking tensors.
    x = x.detach().contiguous()
    return jax_mod.dlpack.from_dlpack(x)


class JAXFunctionWrapper(torch.autograd.Function):
    """Wraps a JAX function so it can be called from PyTorch with autograd.

    Forward:  torch tensor -> dlpack -> jax -> jax_fn -> dlpack -> torch tensor
    Backward: torch grad -> dlpack -> jax -> vjp -> dlpack -> torch grad

    Usage::

        jax_fn = jax.jit(lambda x: jnp.sin(x) ** 2)
        wrapped = JAXFunctionWrapper.apply
        x = torch.randn(10, requires_grad=True)
        y = wrapped(x)
        y.sum().backward()
        print(x.grad)  # correct gradients through JAX
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, jax_fn: Callable) -> torch.Tensor:
        _require_jax()

        jax_x = t2j(x)
        ctx._jax_x = jax_x
        ctx._jax_fn = jax_fn

        jax_y = jax_fn(jax_x)
        ctx._jax_y = jax_y
        return j2t(jax_y)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        jax_mod, _ = _require_jax()
        jax_fn = ctx._jax_fn
        jax_x = ctx._jax_x

        _, vjp_fn = jax_mod.vjp(jax_fn, jax_x)
        jax_grad = vjp_fn(t2j(grad_output))[0]
        grad_input = j2t(jax_grad)
        return grad_input, None


def wrap_jax_fn(jax_fn: Callable) -> Callable:
    """Create a convenience wrapper around :class:`JAXFunctionWrapper`.

    Returns a callable ``f(torch_tensor) -> torch_tensor`` that supports
    autograd via the JAX vjp chain.

    Args:
        jax_fn: A JAX function ``f(x) -> y`` where x and y are JAX arrays.

    Returns:
        A PyTorch-compatible function.
    """

    def wrapped(x: torch.Tensor) -> torch.Tensor:
        return JAXFunctionWrapper.apply(x, jax_fn)

    wrapped.__doc__ = f"PyTorch-autograd wrapper for {jax_fn}"
    return wrapped
