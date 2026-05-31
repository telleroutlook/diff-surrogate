"""Tests for diff_surrogate.generative — protocols, samplers, scorers, pipeline."""

import torch

from diff_surrogate.generative import (
    CandidateSampler,
    CandidateScorer,
    CompositeScorer,
    EnergyBasedSampler,
    GenerativePipeline,
    SurrogateScorer,
    VAESampler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyScorer:
    """Scorer that treats L2 norm of candidate as the score."""

    def score(self, candidate: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return (candidate**2).sum()

    def rank(self, candidates: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        n = candidates.shape[0]
        norms = torch.stack([(candidates[i] ** 2).sum() for i in range(n)])
        return norms.argsort()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVAESampler:
    def test_vae_sampler_output_shape(self):
        """VAESampler.sample returns correct shape."""
        vae = VAESampler(condition_dim=4, latent_dim=8, candidate_shape=(3,))
        condition = torch.randn(2, 4)
        out = vae.sample(condition, n_candidates=5)
        # batch=2, n_candidates=5 => total 10
        assert out.shape == (10, 3)

    def test_vae_sampler_different_candidates(self):
        """Multiple samples from the same condition produce different outputs."""
        vae = VAESampler(condition_dim=4, latent_dim=16, candidate_shape=(6,))
        condition = torch.randn(4)
        out = vae.sample(condition, n_candidates=20)
        # At least two candidates should differ (stochastic sampling)
        assert not torch.allclose(out[0], out[1], atol=1e-6)

    def test_vae_loss_is_scalar(self):
        """VAE ELBO loss returns a scalar tensor."""
        vae = VAESampler(condition_dim=4, latent_dim=8, candidate_shape=(3,))
        cond = torch.randn(8, 4)
        target = torch.randn(8, 3)
        loss = vae.loss(cond, target)
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_vae_encode_decode_roundtrip_shape(self):
        """Encode then decode preserves candidate shape."""
        vae = VAESampler(condition_dim=4, latent_dim=8, candidate_shape=(2, 3))
        cond = torch.randn(5, 4)
        mu, logvar = vae.encode(cond)
        assert mu.shape == (5, 8)
        assert logvar.shape == (5, 8)
        z = mu  # use mean directly
        decoded = vae.decode(z, cond)
        assert decoded.shape == (5, 2, 3)


class TestEnergyBasedSampler:
    def test_energy_based_sampler_output_shape(self):
        """EnergyBasedSampler.sample returns correct shape."""
        ebm = EnergyBasedSampler(candidate_shape=(4,), condition_dim=3, hidden_dim=64)
        condition = torch.randn(2, 3)
        out = ebm.sample(condition, n_candidates=3, n_steps=5, step_size=0.05)
        assert out.shape == (6, 4)

    def test_energy_returns_scalar_per_sample(self):
        """energy() returns one scalar per candidate."""
        ebm = EnergyBasedSampler(candidate_shape=(4,), condition_dim=3)
        candidate = torch.randn(5, 4)
        condition = torch.randn(5, 3)
        e = ebm.energy(candidate, condition)
        assert e.shape == (5,)


class TestCompositeScorer:
    def test_composite_scorer_weights(self):
        """CompositeScorer produces weighted sum of individual scores."""
        s1 = _DummyScorer()
        s2 = _DummyScorer()
        cs = CompositeScorer(scorers=[s1, s2], weights=[1.0, 2.0])
        candidate = torch.tensor([1.0, 2.0, 3.0])
        condition = torch.zeros(2)
        score = cs.score(candidate, condition)
        # s1.score = 1+4+9 = 14, s2.score = 14, weighted = 14 + 2*14 = 42
        expected = torch.tensor(42.0)
        assert torch.isclose(score, expected)

    def test_composite_scorer_rank(self):
        """CompositeScorer.rank returns indices ordered by score."""
        s1 = _DummyScorer()
        cs = CompositeScorer(scorers=[s1], weights=[1.0])
        candidates = torch.tensor([[1.0], [0.5], [2.0]])  # norms: 1, 0.5, 4
        condition = torch.zeros(2)
        ranking = cs.rank(candidates, condition)
        assert ranking[0].item() == 1  # smallest norm first
        assert ranking[-1].item() == 2  # largest norm last

    def test_composite_scorer_mismatch_raises(self):
        """Mismatched scorers/weights lengths raises ValueError."""
        s1 = _DummyScorer()
        import pytest

        with pytest.raises(ValueError):
            CompositeScorer(scorers=[s1], weights=[1.0, 2.0])

    def test_composite_scorer_empty_raises(self):
        """Empty scorer list raises ValueError."""
        import pytest

        with pytest.raises(ValueError):
            CompositeScorer(scorers=[], weights=[])


class TestGenerativePipeline:
    def test_generative_pipeline_returns_best(self):
        """Pipeline generates, scores, and returns the best candidate."""
        vae = VAESampler(condition_dim=4, latent_dim=8, candidate_shape=(3,))
        scorer = _DummyScorer()
        pipeline = GenerativePipeline(sampler=vae, scorer=scorer)

        condition = torch.randn(4)
        result = pipeline.generate(condition, n_candidates=10, top_k=3)

        assert "best" in result
        assert "candidates" in result
        assert "scores" in result
        assert "top_indices" in result
        assert result["candidates"].shape[0] == 10
        assert result["scores"].shape[0] == 10
        assert result["best"].shape == (3,)
        assert result["top_indices"].shape[0] == 3

    def test_pipeline_with_refiner(self):
        """Pipeline applies refiner to top-k candidates."""
        vae = VAESampler(condition_dim=4, latent_dim=8, candidate_shape=(3,))
        scorer = _DummyScorer()

        def refiner(candidate: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
            return candidate * 0.5  # simple shrink refiner

        pipeline = GenerativePipeline(sampler=vae, scorer=scorer, refiner=refiner)
        condition = torch.randn(4)
        result = pipeline.generate(condition, n_candidates=5, top_k=2)

        assert "refined" in result
        assert result["refined"].shape == (2, 3)

    def test_pipeline_best_is_lowest_score(self):
        """The best candidate should correspond to the lowest score."""
        vae = VAESampler(condition_dim=4, latent_dim=8, candidate_shape=(2,))
        scorer = _DummyScorer()
        pipeline = GenerativePipeline(sampler=vae, scorer=scorer)

        condition = torch.randn(4)
        result = pipeline.generate(condition, n_candidates=8, top_k=3)

        # The best candidate should have the minimum score
        best_idx = result["scores"].argmin().item()
        assert torch.allclose(result["best"], result["candidates"][best_idx])


class TestProtocols:
    def test_candidate_protocols_satisfied(self):
        """Concrete classes satisfy the CandidateSampler / CandidateScorer protocols."""
        vae = VAESampler(condition_dim=4, latent_dim=8, candidate_shape=(3,))
        assert isinstance(vae, CandidateSampler)

        ebm = EnergyBasedSampler(candidate_shape=(3,), condition_dim=4)
        assert isinstance(ebm, CandidateSampler)

        scorer = _DummyScorer()
        assert isinstance(scorer, CandidateScorer)

        composite = CompositeScorer(scorers=[scorer], weights=[1.0])
        assert isinstance(composite, CandidateScorer)

    def test_vae_is_nn_module(self):
        """VAESampler is a proper nn.Module."""
        vae = VAESampler(condition_dim=4, latent_dim=8, candidate_shape=(3,))
        assert isinstance(vae, torch.nn.Module)
        # Should have trainable parameters
        assert sum(p.numel() for p in vae.parameters()) > 0

    def test_ebm_is_nn_module(self):
        """EnergyBasedSampler is a proper nn.Module."""
        ebm = EnergyBasedSampler(candidate_shape=(3,), condition_dim=4)
        assert isinstance(ebm, torch.nn.Module)
        assert sum(p.numel() for p in ebm.parameters()) > 0
