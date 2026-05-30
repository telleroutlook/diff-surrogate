"""Tests for JAX <-> PyTorch dlpack interop."""

from __future__ import annotations

import pytest
import torch

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from diff_surrogate.interop import JAXFunctionWrapper, j2t, t2j, wrap_jax_fn


class TestBasicConversion:
    def test_j2t_round_trip_preserves_values(self) -> None:
        jax_arr = jnp.array([1.0, 2.0, 3.0])
        torch_t = j2t(jax_arr)
        assert isinstance(torch_t, torch.Tensor)
        assert torch.allclose(torch_t, torch.tensor([1.0, 2.0, 3.0]))

    def test_t2j_round_trip_preserves_values(self) -> None:
        torch_t = torch.tensor([4.0, 5.0, 6.0])
        jax_arr = t2j(torch_t)
        assert jnp.allclose(jax_arr, jnp.array([4.0, 5.0, 6.0]))

    def test_j2t_t2j_round_trip(self) -> None:
        original = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = t2j(j2t(original))
        assert jnp.allclose(result, original)

    def test_t2j_j2t_round_trip(self) -> None:
        original = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = j2t(t2j(original))
        assert torch.allclose(result, original)

    def test_2d_array(self) -> None:
        jax_arr = jnp.ones((5, 5))
        torch_t = j2t(jax_arr)
        assert torch_t.shape == (5, 5)
        assert torch.allclose(torch_t, torch.ones(5, 5))


class TestJAXFunctionWrapper:
    def test_forward_correct(self) -> None:
        jax_fn = lambda x: jnp.sin(x)
        x = torch.tensor([0.0, 1.0, 2.0])
        y = JAXFunctionWrapper.apply(x, jax_fn)
        expected = torch.sin(x)
        assert torch.allclose(y, expected, atol=1e-6)

    def test_backward_correct(self) -> None:
        jax_fn = lambda x: jnp.sin(x)
        x = torch.tensor([0.5, 1.0, 1.5], requires_grad=True)
        y = JAXFunctionWrapper.apply(x, jax_fn)
        y.sum().backward()
        expected_grad = torch.cos(x.detach())
        assert torch.allclose(x.grad, expected_grad, atol=1e-6)

    def test_quadratic_function(self) -> None:
        jax_fn = lambda x: x ** 2
        x = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
        y = JAXFunctionWrapper.apply(x, jax_fn)
        assert torch.allclose(y, torch.tensor([4.0, 9.0, 16.0]))
        y.sum().backward()
        assert torch.allclose(x.grad, 2 * x.detach())

    def test_composed_function(self) -> None:
        jax_fn = lambda x: jnp.sin(x ** 2)
        x = torch.tensor([0.5, 1.0], requires_grad=True)
        y = JAXFunctionWrapper.apply(x, jax_fn)
        y.sum().backward()
        expected_grad = 2 * x.detach() * torch.cos(x.detach() ** 2)
        assert torch.allclose(x.grad, expected_grad, atol=1e-5)

    def test_2d_input(self) -> None:
        jax_fn = lambda x: jnp.sum(x ** 2, axis=-1, keepdims=True)
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = JAXFunctionWrapper.apply(x, jax_fn)
        assert y.shape == (2, 1)
        y.sum().backward()
        assert x.grad is not None
        expected = 2 * x.detach()
        assert torch.allclose(x.grad, expected, atol=1e-5)


class TestWrapJaxFn:
    def test_convenience_wrapper_forward(self) -> None:
        jax_fn = lambda x: jnp.cos(x)
        wrapped = wrap_jax_fn(jax_fn)
        x = torch.tensor([0.0, 1.0])
        y = wrapped(x)
        assert torch.allclose(y, torch.cos(x), atol=1e-6)

    def test_convenience_wrapper_backward(self) -> None:
        jax_fn = lambda x: jnp.cos(x)
        wrapped = wrap_jax_fn(jax_fn)
        x = torch.tensor([0.5, 1.0], requires_grad=True)
        y = wrapped(x)
        y.sum().backward()
        expected_grad = -torch.sin(x.detach())
        assert torch.allclose(x.grad, expected_grad, atol=1e-6)


class TestGeometryPipelineInterop:
    def test_wrapper_with_geometry_output(self) -> None:
        """Verify interop works with the geometry pipeline's output shapes."""
        from diff_surrogate.geometry import sigmoid_projection

        params = torch.tensor([0.3, 0.7], requires_grad=True)
        field = params.unsqueeze(0).expand(10, 2)

        projected = sigmoid_projection(field, beta=10.0)
        assert projected.shape == (10, 2)
        assert projected.requires_grad

    def test_jax_fn_on_geometry_like_data(self) -> None:
        """Simulate a JAX processing step on geometry pipeline output."""
        jax_fn = lambda x: jnp.exp(-x ** 2)
        wrapped = wrap_jax_fn(jax_fn)

        x = torch.linspace(-2, 2, 20, requires_grad=True)
        y = wrapped(x)
        y.sum().backward()
        expected_grad = -2 * x.detach() * torch.exp(-(x.detach() ** 2))
        assert torch.allclose(x.grad, expected_grad, atol=1e-5)
