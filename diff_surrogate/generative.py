"""Generative prior interface for multi-candidate generation and scoring.

Provides protocols and concrete implementations for the
generate -> score -> refine pipeline used by DiffNano (N8.3) and
OpenLithoHub (O8.2).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class CandidateSampler(Protocol):
    """Protocol for conditional candidate generation."""

    def sample(self, condition: torch.Tensor, n_candidates: int) -> torch.Tensor:
        """Generate n_candidates given a condition tensor.

        Returns:
            Tensor of shape (n_candidates, *candidate_shape).
        """
        ...


@runtime_checkable
class CandidateScorer(Protocol):
    """Protocol for scoring candidates."""

    def score(self, candidate: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Score a single candidate.

        Returns:
            Scalar tensor (lower is better).
        """
        ...

    def rank(self, candidates: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Return indices that sort candidates by score (best first)."""
        ...


# ---------------------------------------------------------------------------
# VAESampler
# ---------------------------------------------------------------------------


class _Encoder(nn.Module):
    """Maps condition -> (mean, log_var) in latent space."""

    def __init__(self, condition_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head_mu = nn.Linear(hidden_dim, latent_dim)
        self.head_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(condition)
        return self.head_mu(h), self.head_logvar(h)


class _Decoder(nn.Module):
    """Maps (latent, condition) -> candidate."""

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        candidate_size: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, candidate_size),
        )

    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, condition], dim=-1))


class VAESampler(nn.Module):
    """Conditional VAE sampler for multi-candidate generation.

    Encoder maps condition -> latent, decoder maps latent -> candidate.
    Sampling draws multiple latent vectors and decodes them.
    """

    def __init__(
        self,
        condition_dim: int,
        latent_dim: int,
        candidate_shape: tuple[int, ...],
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.candidate_shape = candidate_shape
        self.candidate_size = 1
        for s in candidate_shape:
            self.candidate_size *= s

        self.encoder = _Encoder(condition_dim, latent_dim, hidden_dim)
        self.decoder = _Decoder(latent_dim, condition_dim, self.candidate_size, hidden_dim)

    def encode(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, log_var) in latent space."""
        return self.encoder(condition)

    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Decode latent + condition -> candidate."""
        flat = self.decoder(z, condition)
        return flat.view(-1, *self.candidate_shape)

    def sample(
        self,
        condition: torch.Tensor,
        n_candidates: int,
    ) -> torch.Tensor:
        """Sample n_candidates from the learned posterior.

        Args:
            condition: Shape (batch, condition_dim) or (condition_dim,).
            n_candidates: Number of candidates to generate.

        Returns:
            Tensor of shape (n_candidates * batch, *candidate_shape).
        """
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)

        condition.shape[0]
        # Expand condition for multiple candidates
        cond_expanded = condition.repeat_interleave(n_candidates, dim=0)  # (batch*n, cond_dim)
        mu, logvar = self.encode(cond_expanded)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        candidates = self.decode(z, cond_expanded)
        return candidates

    def loss(
        self,
        condition: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """ELBO loss for training.

        Args:
            condition: Shape (batch, condition_dim).
            target: Shape (batch, *candidate_shape).

        Returns:
            Scalar loss tensor.
        """
        mu, logvar = self.encode(condition)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        recon = self.decode(z, condition)

        target_flat = target.reshape(target.shape[0], -1)
        recon_flat = recon.reshape(recon.shape[0], -1)

        recon_loss = F.mse_loss(recon_flat, target_flat, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss


# ---------------------------------------------------------------------------
# EnergyBasedSampler
# ---------------------------------------------------------------------------


class EnergyBasedSampler(nn.Module):
    """Energy-based model sampler using Langevin dynamics."""

    def __init__(
        self,
        candidate_shape: tuple[int, ...],
        condition_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.candidate_shape = candidate_shape
        self.candidate_size = 1
        for s in candidate_shape:
            self.candidate_size *= s

        self.energy_net = nn.Sequential(
            nn.Linear(self.candidate_size + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def energy(self, candidate: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Compute energy (lower = more plausible)."""
        c_flat = candidate.reshape(candidate.shape[0], -1)
        inp = torch.cat([c_flat, condition], dim=-1)
        return self.energy_net(inp).squeeze(-1)

    def sample(
        self,
        condition: torch.Tensor,
        n_candidates: int,
        n_steps: int = 10,
        step_size: float = 0.1,
    ) -> torch.Tensor:
        """Langevin dynamics sampling.

        Args:
            condition: Shape (batch, condition_dim) or (condition_dim,).
            n_candidates: Number of candidates per condition.
            n_steps: Number of Langevin steps.
            step_size: Step size for Langevin dynamics.

        Returns:
            Tensor of shape (n_candidates * batch, *candidate_shape).
        """
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)

        batch = condition.shape[0]
        total = batch * n_candidates
        cond_expanded = condition.repeat_interleave(n_candidates, dim=0)

        # Initialize from noise
        x = torch.randn(total, self.candidate_size, device=condition.device)
        x.requires_grad_(True)

        for _ in range(n_steps):
            e = self.energy(x.view(total, *self.candidate_shape), cond_expanded)
            grad = torch.autograd.grad(e.sum(), x)[0]
            x = x - step_size * grad + step_size * torch.randn_like(x) * 0.1

        return x.detach().view(total, *self.candidate_shape)


# ---------------------------------------------------------------------------
# CompositeScorer
# ---------------------------------------------------------------------------


class CompositeScorer:
    """Combine multiple CandidateScorer instances with weights."""

    def __init__(self, scorers: list[CandidateScorer], weights: list[float]):
        if len(scorers) != len(weights):
            raise ValueError("scorers and weights must have the same length")
        if not scorers:
            raise ValueError("need at least one scorer")
        self.scorers = scorers
        self.weights = weights

    def score(self, candidate: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Weighted sum of individual scores."""
        total = torch.tensor(0.0, device=candidate.device)
        for w, s in zip(self.weights, self.scorers, strict=False):
            total = total + w * s.score(candidate, condition)
        return total

    def rank(self, candidates: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Return indices that sort candidates by combined score (best first)."""
        n = candidates.shape[0]
        scores = torch.stack([self.score(candidates[i], condition) for i in range(n)])
        return scores.argsort()


# ---------------------------------------------------------------------------
# SurrogateScorer — convenience scorer wrapping a SurrogateBase
# ---------------------------------------------------------------------------


class SurrogateScorer:
    """Score candidates by surrogate prediction error against a target.

    Lower error -> better candidate.
    """

    def __init__(self, surrogate, target: torch.Tensor):
        self.surrogate = surrogate
        self.target = target

    def score(self, candidate: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        pred = self.surrogate(candidate.unsqueeze(0))
        if isinstance(pred, dict):
            return sum(
                torch.mean((v - self.target[k].to(v.device)) ** 2)
                for k, v in pred.items()
                if k in self.target
            )
        return torch.mean((pred - self.target.to(pred.device)) ** 2)

    def rank(self, candidates: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        n = candidates.shape[0]
        scores = torch.stack([self.score(candidates[i], condition) for i in range(n)])
        return scores.argsort()


# ---------------------------------------------------------------------------
# GenerativePipeline
# ---------------------------------------------------------------------------


class GenerativePipeline:
    """Generate candidates -> score -> select best.

    Provides the unified interface for DiffNano (N8.3) and OpenLithoHub (O8.2).
    """

    def __init__(
        self,
        sampler: CandidateSampler,
        scorer: CandidateScorer,
        refiner: Callable | None = None,
    ):
        self.sampler = sampler
        self.scorer = scorer
        self.refiner = refiner

    def generate(
        self,
        condition: torch.Tensor,
        n_candidates: int = 10,
        top_k: int = 3,
    ) -> dict[str, torch.Tensor]:
        """Generate candidates, score, optionally refine, return best.

        Args:
            condition: Conditioning tensor.
            n_candidates: Number of candidates to generate.
            top_k: Number of top candidates to return.

        Returns:
            Dict with:
            - ``best``: best candidate (top-1)
            - ``candidates``: all generated candidates
            - ``scores``: score for each candidate
            - ``top_indices``: indices of top-k candidates
            - ``refined``: refined top-k candidates (if refiner provided)
        """
        candidates = self.sampler.sample(condition, n_candidates)

        n = candidates.shape[0]
        scores = torch.stack(
            [self.scorer.score(candidates[i], condition) for i in range(n)]
        )

        ranking = self.scorer.rank(candidates, condition)
        top_indices = ranking[:top_k]

        best = candidates[ranking[0]]

        result: dict[str, torch.Tensor] = {
            "best": best,
            "candidates": candidates,
            "scores": scores,
            "top_indices": top_indices,
        }

        if self.refiner is not None:
            refined = torch.stack(
                [self.refiner(candidates[idx], condition) for idx in top_indices]
            )
            result["refined"] = refined

        return result
