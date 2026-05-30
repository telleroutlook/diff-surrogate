"""Tests for dlpack round-trip fidelity and gradient numerical correctness."""

from __future__ import annotations

import pytest
import torch

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from diff_surrogate.interop import JAXFunctionWrapper, j2t, t2j, wrap_jax_fn  # noqa: E402


class TestRoundTripValues:
    """j2t / t2j preserve exact values through round-trips."""

    def test_j2t_t2j_roundtrip_1d(self) -> None:
        original = jnp.array([1.0, 2.0, 3.0])
        result = t2j(j2t(original))
        assert jnp.allclose(result, original)

    def test_t2j_j2t_roundtrip_1d(self) -> None:
        original = torch.tensor([1.0, 2.0, 3.0])
        result = j2t(t2j(original))
        assert torch.allclose(result, original)

    def test_roundtrip_2d(self) -> None:
        original = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = t2j(j2t(original))
        assert jnp.allclose(result, original)

    def test_roundtrip_preserves_dtype(self) -> None:
        original = jnp.array([1.0, 2.0], dtype=jnp.float32)
        torch_t = j2t(original)
        assert torch_t.dtype == torch.float32
        result = t2j(torch_t)
        assert result.dtype == jnp.float32

    def test_roundtrip_negative_values(self) -> None:
        original = jnp.array([-1.5, -0.0, 3.14, -100.0])
        result = t2j(j2t(original))
        assert jnp.allclose(result, original)


class TestGradientRoundTrip:
    """Verify autograd gradients through JAXFunctionWrapper match finite differences."""

    @staticmethod
    def _fd_gradient(wrapped_fn, x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """Compute finite-difference gradient of sum(wrapped_fn(x))."""
        grad = torch.zeros_like(x)
        x_flat = x.detach().clone().flatten()
        for i in range(x_flat.numel()):
            x_plus = x_flat.clone()
            x_plus[i] += eps
            x_minus = x_flat.clone()
            x_minus[i] -= eps
            f_plus = wrapped_fn(x_plus.reshape(x.shape)).sum().item()
            f_minus = wrapped_fn(x_minus.reshape(x.shape)).sum().item()
            grad.flatten()[i] = (f_plus - f_minus) / (2 * eps)
        return grad

    def test_sin_gradient(self) -> None:
        wrapped = wrap_jax_fn(lambda x: jnp.sin(x))
        x = torch.tensor([0.5, 1.0, 1.5, 2.0], requires_grad=True)
        y = wrapped(x)
        y.sum().backward()
        autograd = x.grad.clone()
        fd = self._fd_gradient(wrapped, x)
        assert torch.allclose(autograd, fd, atol=1e-2)

    def test_quadratic_gradient(self) -> None:
        wrapped = wrap_jax_fn(lambda x: x**2)
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = wrapped(x)
        y.sum().backward()
        autograd = x.grad.clone()
        fd = self._fd_gradient(wrapped, x)
        assert torch.allclose(autograd, fd, atol=1e-2)

    def test_composed_gradient(self) -> None:
        wrapped = wrap_jax_fn(lambda x: jnp.sin(x**2))
        x = torch.tensor([0.3, 0.7, 1.2], requires_grad=True)
        y = wrapped(x)
        y.sum().backward()
        autograd = x.grad.clone()
        fd = self._fd_gradient(wrapped, x)
        assert torch.allclose(autograd, fd, atol=1e-2)

    def test_2d_gradient(self) -> None:
        wrapped = wrap_jax_fn(lambda x: jnp.sum(x**2, axis=-1, keepdims=True))
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = wrapped(x)
        y.sum().backward()
        autograd = x.grad.clone()
        fd = self._fd_gradient(wrapped, x)
        assert torch.allclose(autograd, fd, atol=1e-2)

    def test_physics_fn_gradient(self) -> None:
        """Damped harmonic oscillator: exp(-0.3*x)*sin(2*x)."""

        def physics_fn(x):
            return jnp.exp(-0.3 * x) * jnp.sin(2.0 * x)

        wrapped = wrap_jax_fn(physics_fn)
        x = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5], requires_grad=True)
        y = wrapped(x)
        y.sum().backward()
        autograd = x.grad.clone()
        fd = self._fd_gradient(wrapped, x)
        assert torch.allclose(autograd, fd, atol=1e-2)


class TestPipelineRoundTrip:
    """Gradient flow through PyTorch -> JAX -> PyTorch chains."""

    def test_torch_linear_then_jax(self) -> None:
        """Gradients propagate through a PyTorch Linear layer then a JAX function."""
        wrapped = wrap_jax_fn(lambda x: jnp.sin(x))
        linear = torch.nn.Linear(3, 3)
        torch.manual_seed(0)
        x = torch.randn(2, 3, requires_grad=True)

        h = linear(x)
        y = wrapped(h)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert linear.weight.grad is not None

    def test_jax_then_torch_loss(self) -> None:
        """JAX function output fed into a PyTorch MSE loss."""
        wrapped = wrap_jax_fn(lambda x: x**2)
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = wrapped(x)
        target = torch.tensor([0.0, 0.0, 0.0])
        loss = torch.nn.functional.mse_loss(y, target)
        loss.backward()

        # d/dx_i MSE(x^2, 0) = (4/N) * x_i^3
        expected = (4.0 / 3.0) * x.detach() ** 3
        assert torch.allclose(x.grad, expected, atol=1e-5)
