"""DLPack interop demo: JAX <-> PyTorch with gradient round-trip verification.

Demonstrates:
1. Zero-copy data transfer via dlpack (j2t / t2j)
2. A JAX physics function inserted into a PyTorch autograd graph
3. Full forward/backward pass through PyTorch -> JAX -> PyTorch
4. Numerical gradient verification via finite differences

Run:
    python examples/dlpack_interop_demo.py
"""

from __future__ import annotations

import sys

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    print(
        "JAX is not installed. Install it with:\n"
        "  pip install jax jaxlib\n"
        "Then re-run this example."
    )
    sys.exit(1)

import torch

from diff_surrogate.interop import JAXFunctionWrapper, j2t, t2j, wrap_jax_fn


# ---------------------------------------------------------------------------
# 1. Zero-copy data transfer
# ---------------------------------------------------------------------------
def demo_zero_copy() -> None:
    print("=" * 60)
    print("1. Zero-copy data transfer via dlpack")
    print("=" * 60)

    # JAX -> PyTorch
    jax_arr = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    torch_t = j2t(jax_arr)
    print(f"  JAX array:  {jax_arr}")
    print(f"  -> PyTorch: {torch_t}")
    assert torch.allclose(torch_t, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))

    # PyTorch -> JAX
    torch_t2 = torch.tensor([10.0, 20.0, 30.0])
    jax_arr2 = t2j(torch_t2)
    print(f"  PyTorch tensor: {torch_t2}")
    print(f"  -> JAX:         {jax_arr2}")
    assert jnp.allclose(jax_arr2, jnp.array([10.0, 20.0, 30.0]))

    # Round-trip
    original = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    roundtrip = t2j(j2t(original))
    print(f"  Round-trip error: {jnp.max(jnp.abs(roundtrip - original)):.2e}")
    print()


# ---------------------------------------------------------------------------
# 2. JAX physics function wrapped for PyTorch autograd
# ---------------------------------------------------------------------------
def make_physics_fn():
    """A small differentiable physics simulation: damped harmonic oscillator response.

    f(x) = exp(-damping * x) * sin(frequency * x)

    The JAX function is wrapped and callable from PyTorch with full autograd.
    """

    def physics_fn(x):
        damping = 0.3
        frequency = 2.0
        return jnp.exp(-damping * x) * jnp.sin(frequency * x)

    return physics_fn


def demo_forward_backward() -> None:
    print("=" * 60)
    print("2. Forward & backward through JAX function (PyTorch autograd)")
    print("=" * 60)

    jax_fn = make_physics_fn()
    wrapped = wrap_jax_fn(jax_fn)

    x = torch.linspace(0, 2 * 3.14159, 8, requires_grad=True)
    print(f"  Input x:  {x.detach().numpy().round(4)}")

    # Forward: PyTorch -> dlpack -> JAX -> dlpack -> PyTorch
    y = wrapped(x)
    print(f"  Output y: {y.detach().numpy().round(4)}")

    # Loss and backward
    loss = y.sum()
    loss.backward()

    print(f"  Loss:     {loss.item():.6f}")
    print(f"  grad(x):  {x.grad.detach().numpy().round(4)}")
    print()


# ---------------------------------------------------------------------------
# 3. Full pipeline: PyTorch layers -> JAX function -> PyTorch loss
# ---------------------------------------------------------------------------
def demo_full_pipeline() -> None:
    print("=" * 60)
    print("3. Full pipeline: PyTorch -> JAX -> PyTorch -> loss -> backward")
    print("=" * 60)

    jax_fn = make_physics_fn()
    wrapped = wrap_jax_fn(jax_fn)

    # PyTorch preprocessing layer
    linear = torch.nn.Linear(4, 4)
    torch.manual_seed(42)
    with torch.no_grad():
        linear.weight.copy_(torch.eye(4) * 0.5)
        linear.bias.zero_()

    x = torch.randn(2, 4, requires_grad=True)
    print(f"  Input shape: {x.shape}")

    # PyTorch linear -> JAX physics -> PyTorch MSE loss
    h = linear(x)
    y = wrapped(h)
    target = torch.zeros_like(y)
    loss = torch.nn.functional.mse_loss(y, target)

    loss.backward()

    print(f"  Loss: {loss.item():.6f}")
    print(f"  grad(x) norm: {x.grad.norm().item():.6f}")
    print(f"  grad(weight) norm: {linear.weight.grad.norm().item():.6f}")
    print()


# ---------------------------------------------------------------------------
# 4. Gradient round-trip verification via finite differences
# ---------------------------------------------------------------------------
def demo_gradient_verification() -> None:
    print("=" * 60)
    print("4. Gradient verification: autograd vs finite differences")
    print("=" * 60)

    jax_fn = make_physics_fn()
    wrapped = wrap_jax_fn(jax_fn)

    x0 = torch.tensor([0.5, 1.0, 1.5, 2.0], requires_grad=True)
    eps = 1e-3  # Larger eps needed due to dlpack round-trip numerical noise

    # Autograd gradient
    y = wrapped(x0)
    loss = y.sum()
    loss.backward()
    autograd_grad = x0.grad.clone()

    # Finite-difference gradient
    fd_grad = torch.zeros_like(x0)
    x_detached = x0.detach().clone()
    for i in range(x0.numel()):
        x_plus = x_detached.clone()
        x_plus[i] += eps
        x_minus = x_detached.clone()
        x_minus[i] -= eps
        f_plus = wrapped(x_plus).sum().item()
        f_minus = wrapped(x_minus).sum().item()
        fd_grad[i] = (f_plus - f_minus) / (2 * eps)

    max_err = (autograd_grad - fd_grad).abs().max().item()
    rel_err = max_err / (fd_grad.abs().max().item() + 1e-12)

    print(f"  Autograd grad:  {autograd_grad.detach().numpy().round(6)}")
    print(f"  Finite-diff grad: {fd_grad.detach().numpy().round(6)}")
    print(f"  Max absolute error: {max_err:.2e}")
    print(f"  Max relative error: {rel_err:.2e}")

    if rel_err < 1e-2:
        print("  PASS: gradients match finite differences")
    else:
        print("  FAIL: gradient error exceeds threshold")
    print()


# ---------------------------------------------------------------------------
# 5. Verify zero-copy property (shared memory)
# ---------------------------------------------------------------------------
def demo_shared_memory() -> None:
    print("=" * 60)
    print("5. Zero-copy verification (shared memory)")
    print("=" * 60)

    # JAX -> PyTorch: modifying the tensor data pointer confirms shared storage
    jax_arr = jnp.ones((4,))
    torch_t = j2t(jax_arr)
    # They should have the same data pointer address
    jax_ptr = jax_arr.unsafe_buffer_pointer()
    torch_ptr = torch_t.data_ptr()
    print(f"  JAX buffer pointer:   {jax_ptr}")
    print(f"  PyTorch data pointer: {torch_ptr}")
    print(f"  Pointers match: {jax_ptr == torch_ptr}")
    print()


if __name__ == "__main__":
    demo_zero_copy()
    demo_forward_backward()
    demo_full_pipeline()
    demo_gradient_verification()
    demo_shared_memory()
    print("All demos completed successfully.")
