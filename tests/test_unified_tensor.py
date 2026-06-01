"""Tests for unified tensor interface and OOD extrapolation gate (S11.2)."""

from __future__ import annotations

import pytest
import torch

from diff_surrogate.unified_tensor import (
    OODExtrapolationGate,
    OODThreshold,
    UnifiedTensor,
    UnifiedTensorBenchmark,
    cfd_spec,
    electromagnetic_spec,
    lithography_spec,
    resist_spec,
)

# ---------------------------------------------------------------------------
# TensorSpec presets
# ---------------------------------------------------------------------------


class TestPresets:
    def test_cfd_spec(self):
        spec = cfd_spec(grid_size=32)
        assert len(spec.fields) == 3
        assert spec.fields[0].name == "velocity_x"
        assert spec.grid_size == 32

    def test_electromagnetic_spec(self):
        spec = electromagnetic_spec()
        assert len(spec.fields) == 3
        assert spec.fields[0].name == "E_x"

    def test_lithography_spec(self):
        spec = lithography_spec()
        assert len(spec.fields) == 2

    def test_resist_spec(self):
        spec = resist_spec()
        assert spec.spatial_dims == 3


# ---------------------------------------------------------------------------
# UnifiedTensor
# ---------------------------------------------------------------------------


class TestUnifiedTensor:
    def test_create_2d(self):
        spec = cfd_spec(grid_size=16)
        data = torch.randn(2, 3, 16, 16)
        ut = UnifiedTensor(data, spec)
        assert ut.batch_size == 2
        assert ut.n_fields == 3

    def test_create_3d(self):
        spec = resist_spec(grid_size=8)
        data = torch.randn(2, 1, 8, 8, 8)
        ut = UnifiedTensor(data, spec)
        assert ut.n_fields == 1

    def test_wrong_n_fields_raises(self):
        spec = cfd_spec(grid_size=8)
        data = torch.randn(2, 2, 8, 8)
        with pytest.raises(ValueError, match="n_fields"):
            UnifiedTensor(data, spec)

    def test_too_few_dims_raises(self):
        spec = cfd_spec(grid_size=8)
        data = torch.randn(2, 3)
        with pytest.raises(ValueError, match="at least 3 dims"):
            UnifiedTensor(data, spec)

    def test_field_access(self):
        spec = cfd_spec(grid_size=8)
        data = torch.randn(2, 3, 8, 8)
        ut = UnifiedTensor(data, spec)
        vx = ut.field("velocity_x")
        assert vx.shape == (2, 8, 8)

    def test_field_not_found_raises(self):
        spec = cfd_spec(grid_size=8)
        data = torch.randn(2, 3, 8, 8)
        ut = UnifiedTensor(data, spec)
        with pytest.raises(KeyError, match="not found"):
            ut.field("nonexistent")

    def test_to_latent_flat(self):
        spec = cfd_spec(grid_size=8)
        data = torch.randn(2, 3, 8, 8)
        ut = UnifiedTensor(data, spec)
        flat = ut.to_latent_flat()
        assert flat.shape == (2, 3 * 8 * 8)

    def test_from_latent_flat_roundtrip(self):
        spec = cfd_spec(grid_size=8)
        data = torch.randn(2, 3, 8, 8)
        ut = UnifiedTensor(data, spec)
        flat = ut.to_latent_flat()
        recovered = UnifiedTensor.from_latent_flat(flat, spec)
        torch.testing.assert_close(ut.data, recovered.data)

    def test_resample_2d(self):
        spec = cfd_spec(grid_size=16)
        data = torch.randn(2, 3, 16, 16)
        ut = UnifiedTensor(data, spec)
        resampled = ut.resample_to(8)
        assert resampled.data.shape == (2, 3, 8, 8)
        assert resampled.spec.grid_size == 8

    def test_resample_identity(self):
        spec = cfd_spec(grid_size=16)
        data = torch.randn(2, 3, 16, 16)
        ut = UnifiedTensor(data, spec)
        same = ut.resample_to(16)
        assert same is ut


# ---------------------------------------------------------------------------
# OOD Extrapolation Gate
# ---------------------------------------------------------------------------


class TestOODExtrapolationGate:
    def test_is_ood_detection(self):
        gate = OODExtrapolationGate([
            OODThreshold("reynolds", 100, 1000),
        ])
        is_ood, violated = gate.is_ood({"reynolds": 2000})
        assert is_ood
        assert "reynolds" in violated

    def test_is_in_distribution(self):
        gate = OODExtrapolationGate([
            OODThreshold("reynolds", 100, 1000),
        ])
        is_ood, violated = gate.is_ood({"reynolds": 500})
        assert not is_ood
        assert len(violated) == 0

    def test_below_threshold(self):
        gate = OODExtrapolationGate([
            OODThreshold("reynolds", 100, 1000),
        ])
        is_ood, _violated = gate.is_ood({"reynolds": 50})
        assert is_ood

    def test_add_threshold(self):
        gate = OODExtrapolationGate()
        gate.add_threshold("wavelength", 400, 700)
        is_ood, _ = gate.is_ood({"wavelength": 800})
        assert is_ood

    def test_evaluate_extrapolation(self):
        gate = OODExtrapolationGate()
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.1, 2.1, 3.1])
        baseline = torch.tensor([0.5, 1.5, 2.5])
        result = gate.evaluate_extrapolation(preds, targets, baseline, is_ood=True)
        assert result["mse_in_context"] < result["mse_baseline"]
        assert result["improvement_ratio"] > 0
        assert result["is_ood"]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


class TestBenchmark:
    def test_benchmark_runs(self):
        result = UnifiedTensorBenchmark.run(n_seeds=2)
        assert "cfd" in result
        assert "electromagnetic" in result
        assert "lithography" in result
        assert result["n_seeds"] == 2

    def test_benchmark_reconstruction_is_zero(self):
        result = UnifiedTensorBenchmark.run(n_seeds=2)
        for domain in ("cfd", "electromagnetic", "lithography"):
            assert result[domain]["mean_reconstruction_err"] < 1e-5
