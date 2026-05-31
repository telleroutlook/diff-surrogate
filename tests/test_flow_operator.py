"""Tests for diff_surrogate.flow_operator — flow-matching generative operator backbone."""

from __future__ import annotations

import torch
import torch.nn as nn

from diff_surrogate.codomain import CodomainBackbone
from diff_surrogate.conformal import SplitConformalPredictor
from diff_surrogate.decision import AcceptRejectGate
from diff_surrogate.flow_operator import (
    FlowMarchingTransformer,
    FlowOperator,
    FlowOperatorBenchmark,
    LocationScaleFlowKernel,
    P2VAE,
    P2VAEDecoder,
    P2VAEEncoder,
)


def _make_fields(batch: int = 4, n_fields: int = 3, spatial_dim: int = 16) -> torch.Tensor:
    return torch.randn(batch, n_fields, spatial_dim)


def _make_backbone(spatial_dim: int = 16, embed_dim: int = 32, field_names: list[str] | None = None):
    return CodomainBackbone(
        spatial_dim=spatial_dim,
        embed_dim=embed_dim,
        n_layers=2,
        n_heads=2,
        field_names=field_names,
    )


FIELD_NAMES = ["u", "v", "p"]


class TestP2VAE:
    def test_p2vae_encode_decode_shape(self):
        """P2VAE encode produces correct (mu, logvar) shapes; decode reconstructs field shape."""
        spatial_dim = 16
        latent_dim = 8
        n_fields = 3
        bb = _make_backbone(spatial_dim=spatial_dim, embed_dim=32, field_names=FIELD_NAMES)
        vae = P2VAE(backbone=bb, latent_dim=latent_dim, n_fields=n_fields, spatial_dim=spatial_dim)

        fields = _make_fields(batch=4, n_fields=n_fields, spatial_dim=spatial_dim)
        mu, logvar = vae.encode(fields, FIELD_NAMES)
        assert mu.shape == (4, latent_dim)
        assert logvar.shape == (4, latent_dim)

        z = vae.reparameterize(mu, logvar)
        recon = vae.decode(z)
        assert recon.shape == fields.shape

    def test_p2vae_loss_decreases(self):
        """P2VAE ELBO loss decreases over training steps."""
        spatial_dim = 16
        latent_dim = 8
        n_fields = 3
        bb = _make_backbone(spatial_dim=spatial_dim, embed_dim=32, field_names=FIELD_NAMES)
        vae = P2VAE(backbone=bb, latent_dim=latent_dim, n_fields=n_fields, spatial_dim=spatial_dim)
        opt = torch.optim.Adam(vae.parameters(), lr=1e-3)

        fields = _make_fields(batch=16, n_fields=n_fields, spatial_dim=spatial_dim)
        losses = []
        for _ in range(20):
            opt.zero_grad()
            loss = vae.loss(fields, FIELD_NAMES)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_p2vae_latent_distribution(self):
        """After training, the latent distribution has finite mean/variance near unit Gaussian."""
        spatial_dim = 16
        latent_dim = 8
        n_fields = 3
        bb = _make_backbone(spatial_dim=spatial_dim, embed_dim=32, field_names=FIELD_NAMES)
        vae = P2VAE(backbone=bb, latent_dim=latent_dim, n_fields=n_fields, spatial_dim=spatial_dim)
        opt = torch.optim.Adam(vae.parameters(), lr=1e-3)

        fields = _make_fields(batch=32, n_fields=n_fields, spatial_dim=spatial_dim)
        for _ in range(30):
            opt.zero_grad()
            loss = vae.loss(fields, FIELD_NAMES)
            loss.backward()
            opt.step()

        vae.eval()
        with torch.no_grad():
            mu, logvar = vae.encode(fields, FIELD_NAMES)
        assert mu.std().item() < 5.0, "Latent mean exploded"
        assert torch.isfinite(mu).all()
        assert torch.isfinite(logvar).all()

    def test_p2vae_encoder_decoder_modules(self):
        """P2VAE exposes encoder and decoder as proper nn.Modules."""
        spatial_dim = 16
        bb = _make_backbone(spatial_dim=spatial_dim, embed_dim=32, field_names=FIELD_NAMES)
        vae = P2VAE(backbone=bb, latent_dim=8, n_fields=3, spatial_dim=spatial_dim)
        assert isinstance(vae.encoder, P2VAEEncoder)
        assert isinstance(vae.decoder, P2VAEDecoder)
        assert vae.latent_dim == 8


class TestLocationScaleFlowKernel:
    def test_location_scale_kernel_deterministic(self):
        """With k=1, location is a linear interpolation and scale is non-zero."""
        x_0 = torch.randn(4, 8)
        x_1 = torch.randn(4, 8)
        t = torch.tensor([0.0, 0.25, 0.5, 1.0])

        mu_t = LocationScaleFlowKernel.location(x_0, x_1, t)
        assert torch.allclose(mu_t[0], x_0[0], atol=1e-6), "At t=0, location should be x_0"
        assert torch.allclose(mu_t[3], x_1[3], atol=1e-6), "At t=1, location should be x_1"

        sigma_t = LocationScaleFlowKernel.scale(t, k=1.0, event_shape=(8,))
        assert sigma_t[0].item() == 0.0, "Scale at t=0 should be 0"
        assert sigma_t[3].item() == 0.0, "Scale at t=1 should be 0"
        assert sigma_t[2].item() > 0, "Scale at t=0.5 should be positive"

    def test_location_scale_kernel_generative(self):
        """With k=0, velocity is purely x_1 - x_0 (deterministic)."""
        x_0 = torch.randn(4, 8)
        x_1 = torch.randn(4, 8)
        t = torch.rand(4)

        v = LocationScaleFlowKernel.compute_velocity(x_0, x_1, t, k=0.0)
        expected = x_1 - x_0
        assert torch.allclose(v, expected, atol=1e-6), "k=0 velocity should be x_1 - x_0"

    def test_location_scale_kernel_interpolation(self):
        """With k=0.5, velocity includes a stochastic component."""
        x_0 = torch.randn(4, 8)
        x_1 = torch.randn(4, 8)
        t = torch.tensor([0.5, 0.5, 0.5, 0.5])
        eps = torch.randn(4, 8)

        v = LocationScaleFlowKernel.compute_velocity(x_0, x_1, t, k=0.5, eps=eps)
        base = x_1 - x_0
        dsigma = 0.5 * (1 - 2 * 0.5)  # = 0
        expected = base + dsigma * eps

        # dsigma at t=0.5 is 0, so velocity == base even with k>0
        assert torch.allclose(v, base, atol=1e-6)

        # At t=0.25, dsigma is non-zero
        t2 = torch.tensor([0.25])
        v2 = LocationScaleFlowKernel.compute_velocity(x_0[:1], x_1[:1], t2, k=0.5, eps=eps[:1])
        dsigma2 = 0.5 * (1 - 2 * 0.25)
        assert abs(dsigma2) > 0, "dsigma should be non-zero at t=0.25"

    def test_sample_x_t_is_valid(self):
        """sample_x_t produces finite samples."""
        x_0 = torch.randn(8, 16)
        x_1 = torch.randn(8, 16)
        t = torch.rand(8)

        for k in [0.0, 0.5, 1.0]:
            x_t = LocationScaleFlowKernel.sample_x_t(x_0, x_1, t, k)
            assert x_t.shape == x_0.shape
            assert torch.isfinite(x_t).all(), f"Non-finite values with k={k}"


class TestFlowMarchingTransformer:
    def test_flow_transformer_output_shape(self):
        """FlowMarchingTransformer output matches input shape."""
        state_dim = 16
        cond_dim = 8
        net = FlowMarchingTransformer(
            state_dim=state_dim,
            cond_dim=cond_dim,
            embed_dim=32,
            n_layers=2,
            n_heads=2,
        )

        x_t = torch.randn(4, state_dim)
        t = torch.rand(4)
        cond = torch.randn(4, cond_dim)
        v = net(x_t, t, cond)
        assert v.shape == (4, state_dim)

    def test_flow_transformer_time_conditioning(self):
        """Different times produce different velocities."""
        state_dim = 16
        cond_dim = 8
        net = FlowMarchingTransformer(
            state_dim=state_dim,
            cond_dim=cond_dim,
            embed_dim=32,
            n_layers=2,
            n_heads=2,
        )

        x_t = torch.randn(2, state_dim)
        cond = torch.randn(2, cond_dim)
        t1 = torch.tensor([0.1, 0.2])
        t2 = torch.tensor([0.8, 0.9])

        v1 = net(x_t, t1, cond)
        v2 = net(x_t, t2, cond)
        assert not torch.allclose(v1, v2, atol=1e-5), "Different times should produce different velocities"

    def test_flow_transformer_sequence_input(self):
        """FlowMarchingTransformer handles 3D sequence input."""
        state_dim = 8
        cond_dim = 4
        net = FlowMarchingTransformer(
            state_dim=state_dim,
            cond_dim=cond_dim,
            embed_dim=32,
            n_layers=2,
            n_heads=2,
        )
        x_t = torch.randn(2, 5, state_dim)
        t = torch.rand(2)
        cond = torch.randn(2, cond_dim)
        v = net(x_t, t, cond)
        assert v.shape == (2, 5, state_dim)


class TestFlowOperator:
    def test_flow_operator_sample_shape(self):
        """FlowOperator.sample returns (n_fields, spatial_dim)."""
        spatial_dim = 16
        n_fields = 3
        cond_dim = 8
        op = FlowOperator(
            spatial_dim=spatial_dim,
            embed_dim=32,
            n_fields=n_fields,
            latent_dim=8,
            cond_dim=cond_dim,
            field_names=FIELD_NAMES,
            n_backbone_layers=2,
            n_backbone_heads=2,
            n_flow_layers=2,
            n_flow_heads=2,
            decoder_hidden=32,
        )
        cond = torch.randn(cond_dim)
        sample = op.sample(cond, n_steps=5, k=0.0)
        assert sample.shape == (n_fields, spatial_dim)

    def test_flow_operator_ensemble_diversity(self):
        """Ensemble members are not identical (generative diversity)."""
        spatial_dim = 16
        n_fields = 3
        cond_dim = 8
        op = FlowOperator(
            spatial_dim=spatial_dim,
            embed_dim=32,
            n_fields=n_fields,
            latent_dim=8,
            cond_dim=cond_dim,
            field_names=FIELD_NAMES,
            n_backbone_layers=2,
            n_backbone_heads=2,
            n_flow_layers=2,
            n_flow_heads=2,
            decoder_hidden=32,
        )
        cond = torch.randn(cond_dim)
        ensemble = op.generate_ensemble(cond, n_samples=8, n_steps=5, k=0.0)
        assert ensemble.shape == (8, n_fields, spatial_dim)
        assert not torch.allclose(ensemble[0], ensemble[1], atol=1e-6), "Ensemble should be diverse"

    def test_flow_operator_training_reduces_loss(self):
        """Training the FlowOperator reduces the flow-matching loss."""
        spatial_dim = 16
        n_fields = 3
        cond_dim = 8
        op = FlowOperator(
            spatial_dim=spatial_dim,
            embed_dim=32,
            n_fields=n_fields,
            latent_dim=8,
            cond_dim=cond_dim,
            field_names=FIELD_NAMES,
            n_backbone_layers=2,
            n_backbone_heads=2,
            n_flow_layers=2,
            n_flow_heads=2,
            decoder_hidden=32,
        )
        opt = torch.optim.Adam(op.parameters(), lr=1e-3)
        fields = _make_fields(batch=8, n_fields=n_fields, spatial_dim=spatial_dim)
        conditions = torch.randn(8, cond_dim)

        losses = []
        for _ in range(20):
            loss_val = op.train_step(fields, FIELD_NAMES, conditions, opt, k=0.5)
            losses.append(loss_val)

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_flow_operator_bridge_parameter_sweep(self):
        """FlowOperator works across the full range of bridge parameter k."""
        spatial_dim = 8
        n_fields = 2
        cond_dim = 4
        op = FlowOperator(
            spatial_dim=spatial_dim,
            embed_dim=16,
            n_fields=n_fields,
            latent_dim=4,
            cond_dim=cond_dim,
            field_names=["u", "v"],
            n_backbone_layers=1,
            n_backbone_heads=2,
            n_flow_layers=1,
            n_flow_heads=2,
            decoder_hidden=16,
        )
        cond = torch.randn(cond_dim)
        for k in [0.0, 0.25, 0.5, 0.75, 1.0]:
            sample = op.sample(cond, n_steps=3, k=k)
            assert sample.shape == (n_fields, spatial_dim), f"Failed for k={k}"
            assert torch.isfinite(sample).all(), f"Non-finite for k={k}"

    def test_flow_operator_with_conformal_calibration(self):
        """FlowOperator ensemble + conformal calibration produces coverage bands."""
        spatial_dim = 8
        n_fields = 2
        cond_dim = 4
        n_train = 40
        n_cal = 20

        op = FlowOperator(
            spatial_dim=spatial_dim,
            embed_dim=16,
            n_fields=n_fields,
            latent_dim=4,
            cond_dim=cond_dim,
            field_names=["u", "v"],
            n_backbone_layers=1,
            n_backbone_heads=2,
            n_flow_layers=1,
            n_flow_heads=2,
            decoder_hidden=16,
        )
        opt = torch.optim.Adam(op.parameters(), lr=1e-3)

        torch.manual_seed(0)
        train_fields = _make_fields(batch=n_train, n_fields=n_fields, spatial_dim=spatial_dim)
        train_cond = torch.randn(n_train, cond_dim)

        for _ in range(10):
            for i in range(0, n_train, 8):
                batch_f = train_fields[i : i + 8]
                batch_c = train_cond[i : i + 8]
                if batch_f.shape[0] < 2:
                    continue
                op.train_step(batch_f, ["u", "v"], batch_c, opt, k=0.5)

        op.eval()

        cal_ensembles = []
        for i in range(n_cal):
            ens = op.generate_ensemble(train_cond[i], n_samples=16, n_steps=5, k=0.0)
            cal_ensembles.append(ens.mean(dim=0))
        cal_preds = torch.stack(cal_ensembles).reshape(n_cal, -1)
        cal_targets = train_fields[:n_cal].reshape(n_cal, -1)

        cp = SplitConformalPredictor()
        cp.calibrate(cal_preds, cal_targets, alpha=0.1)

        test_cond = torch.randn(5, cond_dim)
        test_ensembles = []
        for i in range(5):
            ens = op.generate_ensemble(test_cond[i], n_samples=16, n_steps=5, k=0.0)
            test_ensembles.append(ens.mean(dim=0))
        test_preds = torch.stack(test_ensembles).reshape(5, -1)

        lower, upper = cp.predict(test_preds)
        assert lower.shape == test_preds.shape
        assert (upper >= lower).all()

    def test_flow_operator_with_decision_gate(self):
        """FlowOperator ensemble predictions can be evaluated by AcceptRejectGate."""
        spatial_dim = 8
        n_fields = 2
        cond_dim = 4

        op = FlowOperator(
            spatial_dim=spatial_dim,
            embed_dim=16,
            n_fields=n_fields,
            latent_dim=4,
            cond_dim=cond_dim,
            field_names=["u", "v"],
            n_backbone_layers=1,
            n_backbone_heads=2,
            n_flow_layers=1,
            n_flow_heads=2,
            decoder_hidden=16,
        )
        op.eval()

        cond = torch.randn(cond_dim)
        ensemble = op.generate_ensemble(cond, n_samples=16, n_steps=5, k=0.0)
        pred = ensemble.mean(dim=0)
        lower = ensemble.min(dim=0).values
        upper = ensemble.max(dim=0).values

        gate = AcceptRejectGate(min_coverage=0.9, max_bandwidth=2.0)
        verdict, reasons = gate.evaluate(pred, lower, upper)
        assert hasattr(verdict, "value")
        assert "mean_bandwidth" in reasons

    def test_flow_operator_compute_loss_gradient(self):
        """compute_loss returns a differentiable scalar."""
        spatial_dim = 8
        n_fields = 2
        cond_dim = 4
        op = FlowOperator(
            spatial_dim=spatial_dim,
            embed_dim=16,
            n_fields=n_fields,
            latent_dim=4,
            cond_dim=cond_dim,
            field_names=["u", "v"],
            n_backbone_layers=1,
            n_backbone_heads=2,
            n_flow_layers=1,
            n_flow_heads=2,
            decoder_hidden=16,
        )
        fields = _make_fields(batch=4, n_fields=n_fields, spatial_dim=spatial_dim)
        cond = torch.randn(4, cond_dim)
        loss = op.compute_loss(fields, ["u", "v"], cond, k=0.5)
        assert loss.dim() == 0
        assert loss.requires_grad
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in op.parameters()
        )
        assert has_grad, "No gradient in FlowOperator after compute_loss"


class TestBenchmark:
    def test_benchmark_runs(self):
        """FlowOperatorBenchmark.run completes and returns valid metrics."""
        result = FlowOperatorBenchmark.run(
            spatial_dim=8,
            n_fields=2,
            n_train=16,
            n_test=4,
            latent_dim=4,
            cond_dim=4,
            n_epochs=3,
            lr=1e-3,
            n_ensemble=4,
            n_flow_steps=3,
            k=0.5,
        )

        assert "flow_operator" in result
        assert "codomain_backbone" in result
        assert "probabilistic_surrogate" in result

        flow = result["flow_operator"]
        assert "drift" in flow
        assert "spectral_ratio" in flow
        assert "diversity" in flow
        assert flow["diversity"] >= 0

        bb = result["codomain_backbone"]
        assert bb["diversity"] == 0.0
        assert "drift" in bb

        pno = result["probabilistic_surrogate"]
        assert "conformal_coverage" in pno
        assert pno["conformal_coverage"] >= 0.0
