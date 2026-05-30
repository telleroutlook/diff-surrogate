"""Tests for SobolevLoss and gradient_fidelity_score."""

from __future__ import annotations

import torch
import torch.nn as nn

from diff_surrogate.base import SurrogateBase
from diff_surrogate.trainer import SobolevLoss, SurrogateTrainer, gradient_fidelity_score


class _QuadraticSurrogate(SurrogateBase):
    """Minimal surrogate: y = a * x^2 + b * x + c (scalar in/out)."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def _build_network(self) -> nn.Module:
        return nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_network()(x)

    def generate_training_data(
        self, n_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.linspace(-2.0, 2.0, n_samples).unsqueeze(-1)
        y = x**2
        return x, y


class _SinSurrogate(SurrogateBase):
    """Minimal surrogate for learning sin(x)."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self._hidden = hidden

    def _build_network(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(1, self._hidden),
            nn.Tanh(),
            nn.Linear(self._hidden, self._hidden),
            nn.Tanh(),
            nn.Linear(self._hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_network()(x)

    def generate_training_data(
        self, n_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.linspace(-3.14, 3.14, n_samples).unsqueeze(-1)
        y = torch.sin(x)
        return x, y


# ---------------------------------------------------------------------------
# Test 1: SobolevLoss forward correctness
# ---------------------------------------------------------------------------


def test_sobolev_loss_forward():
    """SobolevLoss = MSE(value) + weight * MSE(grad)."""
    loss_fn = SobolevLoss(derivative_weight=1.0)

    # Known function: y = x^2, dy/dx = 2x
    x = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64, requires_grad=True)
    y_true = x**2
    y_pred = x**2 + torch.tensor([[0.1], [-0.1], [0.0]], dtype=torch.float64)

    loss = loss_fn(y_pred, y_true, x)
    assert loss.requires_grad

    # MSE on values: mean of [0.01, 0.01, 0.0] = 0.02/3
    expected_value_mse = torch.tensor(0.02 / 3, dtype=torch.float64)

    # Gradients: pred grad = 2x, true grad = 2x => same, so grad MSE = 0
    loss_val = loss.item()
    assert abs(loss_val - expected_value_mse.item()) < 1e-6, (
        f"Expected ~{expected_value_mse.item():.6f}, got {loss_val:.6f}"
    )


def test_sobolev_loss_with_derivative_weight_zero():
    """With derivative_weight=0 the loss is plain MSE."""
    loss_fn = SobolevLoss(derivative_weight=0.0)
    x = torch.tensor([[1.0], [2.0]], dtype=torch.float64, requires_grad=True)
    y_true = x**2
    y_pred = x**2 + 1.0

    loss = loss_fn(y_pred, y_true, x)
    expected = torch.nn.MSELoss()(y_pred, y_true)
    assert abs(loss.item() - expected.item()) < 1e-10


# ---------------------------------------------------------------------------
# Test 2: gradient_fidelity_score correctness
# ---------------------------------------------------------------------------


def test_gradient_fidelity_metric():
    """For y = x^2 the true gradient is 2x; metric should be near-perfect."""
    surr = _QuadraticSurrogate()
    # Replace the network with one that computes x^2 exactly
    class _ExactSquare(nn.Module):
        def forward(self, x):
            return x**2

    surr._network = _ExactSquare()

    x = torch.linspace(-1.0, 1.0, 20, dtype=torch.float64).unsqueeze(-1)
    true_grad_fn = lambda inp: 2.0 * inp  # noqa: E731

    result = gradient_fidelity_score(surr, x, true_grad_fn)
    assert result["cosine_similarity"] > 0.9999, (
        f"cosine_similarity too low: {result['cosine_similarity']}"
    )
    assert result["relative_error"] < 1e-4, (
        f"relative_error too high: {result['relative_error']}"
    )


# ---------------------------------------------------------------------------
# Test 3: Sobolev training improves gradient fidelity
# ---------------------------------------------------------------------------


def test_sobolev_training_improves_gradients():
    """Sobolev-trained surrogate should have better gradient fidelity than MSE-only."""
    torch.manual_seed(42)

    n_samples = 128
    n_epochs = 200
    lr = 1e-3

    # --- Train with plain MSE ---
    surr_mse = _SinSurrogate()
    trainer_mse = SurrogateTrainer(surr_mse, lr=lr)
    trainer_mse.train(
        n_epochs=n_epochs,
        n_samples=n_samples,
        batch_size=32,
        derivative_weight=0.0,
    )

    # --- Train with Sobolev loss ---
    torch.manual_seed(42)
    surr_sob = _SinSurrogate()
    trainer_sob = SurrogateTrainer(surr_sob, lr=lr)
    trainer_sob.train(
        n_epochs=n_epochs,
        n_samples=n_samples,
        batch_size=32,
        derivative_weight=1.0,
        target_grad_fn=lambda inp: torch.cos(inp),
    )

    # Evaluate gradient fidelity on sin(x): dy/dx = cos(x)
    x_eval = torch.linspace(-3.0, 3.0, 50, dtype=torch.float64).unsqueeze(-1)
    true_grad_fn = lambda inp: torch.cos(inp)  # noqa: E731

    score_mse = gradient_fidelity_score(surr_mse, x_eval, true_grad_fn)
    score_sob = gradient_fidelity_score(surr_sob, x_eval, true_grad_fn)

    # The Sobolev-trained model should have higher cosine similarity
    # (closer to 1.0) than the plain MSE model.
    assert score_sob["cosine_similarity"] > score_mse["cosine_similarity"], (
        f"Sobolev cosine_sim ({score_sob['cosine_similarity']:.4f}) "
        f"not better than MSE ({score_mse['cosine_similarity']:.4f})"
    )
    assert score_sob["relative_error"] < score_mse["relative_error"], (
        f"Sobolev relative_error ({score_sob['relative_error']:.4f}) "
        f"not better than MSE ({score_mse['relative_error']:.4f})"
    )
