"""Unified spatiotemporal tensor interface for cross-domain generative modeling (S11.2).

Provides a common tensor layout and variable registry so that fields from
disparate physics domains (CFD velocity fields, FDTD electromagnetic fields,
lithography mask/aerial images, resist profiles) can be processed by the
same generative backbone and in-context transfer operator.

The canonical layout is:
    (batch, n_fields, [temporal], height, width[, depth])

References:
    - UniFluids: arXiv:2603.22309, 2026-03
    - MetaAI: Nature Machine Intelligence, 2026-01
    - Flow Marching: arXiv:2509.18611

Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Field registry
# ---------------------------------------------------------------------------


class PhysicsDomain(Enum):
    CFD = "cfd"
    ELECTROMAGNETIC = "electromagnetic"
    LITHOGRAPHY = "lithography"
    RESIST = "resist"
    GENERIC = "generic"


@dataclass
class FieldSpec:
    """Specification for a single physics field.

    Attributes:
        name: Human-readable field name.
        units: Physical units (e.g. "m/s", "V/m", "nm").
        bounds: Optional (min, max) value range.
    """

    name: str
    units: str = ""
    bounds: tuple[float, float] | None = None


@dataclass
class TensorSpec:
    """Complete specification for a unified tensor batch.

    Attributes:
        domain: Physics domain of the data.
        spatial_dims: Number of spatial dimensions (2 or 3).
        has_temporal: Whether a temporal dimension is present.
        fields: Ordered list of field specifications.
        grid_size: Spatial grid size (assumed square/cubic).
    """

    domain: PhysicsDomain
    spatial_dims: int = 2
    has_temporal: bool = False
    fields: list[FieldSpec] = field(default_factory=list)
    grid_size: int = 32


# ---------------------------------------------------------------------------
# Domain presets
# ---------------------------------------------------------------------------


def cfd_spec(grid_size: int = 64, has_temporal: bool = False) -> TensorSpec:
    return TensorSpec(
        domain=PhysicsDomain.CFD,
        spatial_dims=2,
        has_temporal=has_temporal,
        fields=[
            FieldSpec("velocity_x", "m/s"),
            FieldSpec("velocity_y", "m/s"),
            FieldSpec("pressure", "Pa"),
        ],
        grid_size=grid_size,
    )


def electromagnetic_spec(grid_size: int = 32, has_temporal: bool = False) -> TensorSpec:
    return TensorSpec(
        domain=PhysicsDomain.ELECTROMAGNETIC,
        spatial_dims=2,
        has_temporal=has_temporal,
        fields=[
            FieldSpec("E_x", "V/m"),
            FieldSpec("E_y", "V/m"),
            FieldSpec("H_z", "A/m"),
        ],
        grid_size=grid_size,
    )


def lithography_spec(grid_size: int = 64) -> TensorSpec:
    return TensorSpec(
        domain=PhysicsDomain.LITHOGRAPHY,
        spatial_dims=2,
        has_temporal=False,
        fields=[
            FieldSpec("mask", ""),
            FieldSpec("aerial_image", ""),
        ],
        grid_size=grid_size,
    )


def resist_spec(grid_size: int = 32, n_slices: int = 10) -> TensorSpec:
    return TensorSpec(
        domain=PhysicsDomain.RESIST,
        spatial_dims=3,
        has_temporal=False,
        fields=[
            FieldSpec("resist_profile", ""),
        ],
        grid_size=grid_size,
    )


# ---------------------------------------------------------------------------
# UnifiedTensor — canonical container
# ---------------------------------------------------------------------------


class UnifiedTensor:
    """Canonical container for physics field data across domains.

    Wraps a raw tensor and its TensorSpec, providing:
      - Validation of shape against spec
      - Padding / cropping to target grid size
      - Field slicing by name or index
      - Conversion to flat latent-compatible representation

    The underlying tensor follows the layout:
        (batch, n_fields, [T], H, W[, D])
    """

    def __init__(self, data: Tensor, spec: TensorSpec) -> None:
        self._validate(data, spec)
        self.data = data
        self.spec = spec

    @property
    def batch_size(self) -> int:
        return self.data.shape[0]

    @property
    def n_fields(self) -> int:
        return len(self.spec.fields)

    def _validate(self, data: Tensor, spec: TensorSpec) -> None:
        if data.ndim < 3:
            raise ValueError(
                f"UnifiedTensor requires at least 3 dims (B, F, ...), got {data.ndim}"
            )
        if data.shape[1] != len(spec.fields):
            raise ValueError(
                f"n_fields mismatch: tensor has {data.shape[1]}, "
                f"spec has {len(spec.fields)}"
            )

    def field(self, name: str) -> Tensor:
        """Get a single field by name."""
        for i, fs in enumerate(self.spec.fields):
            if fs.name == name:
                return self.data[:, i]
        raise KeyError(f"Field '{name}' not found in spec")

    def to_latent_flat(self) -> Tensor:
        """Flatten spatial dims for latent encoding.

        Returns:
            (batch, n_fields * spatial_volume) tensor.
        """
        return torch.flatten(self.data, start_dim=2).reshape(self.batch_size, -1)

    @classmethod
    def from_latent_flat(
        cls,
        flat: Tensor,
        spec: TensorSpec,
    ) -> UnifiedTensor:
        """Reconstruct from flat latent representation."""
        nf = len(spec.fields)
        gs = spec.grid_size
        if spec.spatial_dims == 2:
            spatial = flat.reshape(flat.shape[0], nf, gs, gs)
        else:
            spatial = flat.reshape(flat.shape[0], nf, gs, gs, gs)
        return cls(spatial, spec)

    def resample_to(
        self,
        target_grid_size: int,
        mode: str = "bilinear",
    ) -> UnifiedTensor:
        """Resample spatial dimensions to a target grid size.

        Args:
            target_grid_size: Target spatial size.
            mode: Interpolation mode.

        Returns:
            New UnifiedTensor at the target resolution.
        """
        if target_grid_size == self.spec.grid_size:
            return self

        new_spec = TensorSpec(
            domain=self.spec.domain,
            spatial_dims=self.spec.spatial_dims,
            has_temporal=self.spec.has_temporal,
            fields=list(self.spec.fields),
            grid_size=target_grid_size,
        )

        if self.spec.spatial_dims == 2:
            reshaped = self.data.reshape(
                self.batch_size, self.n_fields, self.spec.grid_size, self.spec.grid_size
            )
            resampled = torch.nn.functional.interpolate(
                reshaped,
                size=(target_grid_size, target_grid_size),
                mode=mode,
                align_corners=False,
            )
        else:
            reshaped = self.data.reshape(
                self.batch_size,
                self.n_fields,
                self.spec.grid_size,
                self.spec.grid_size,
                self.spec.grid_size,
            )
            resampled = torch.nn.functional.interpolate(
                reshaped,
                size=(target_grid_size, target_grid_size, target_grid_size),
                mode="trilinear",
                align_corners=False,
            )
        return UnifiedTensor(resampled, new_spec)


# ---------------------------------------------------------------------------
# OOD extrapolation gate
# ---------------------------------------------------------------------------


@dataclass
class OODThreshold:
    """Threshold for out-of-distribution detection.

    Attributes:
        param_name: Name of the physics parameter.
        train_min: Minimum value seen during training.
        train_max: Maximum value seen during training.
    """

    param_name: str
    train_min: float
    train_max: float


class OODExtrapolationGate:
    """Evaluates whether a condition falls outside the training distribution.

    Uses simple interval bounds on physics parameters to flag OOD inputs,
    then measures whether in-context extrapolation outperforms interpolation
    baseline on those OOD inputs.
    """

    def __init__(self, thresholds: list[OODThreshold] | None = None) -> None:
        self.thresholds = thresholds or []

    def add_threshold(self, name: str, train_min: float, train_max: float) -> None:
        self.thresholds.append(OODThreshold(name, train_min, train_max))

    def is_ood(self, params: dict[str, float]) -> tuple[bool, list[str]]:
        """Check if parameters fall outside training distribution.

        Returns:
            (is_ood, list_of_violated_param_names)
        """
        violated = []
        for t in self.thresholds:
            val = params.get(t.param_name)
            if val is not None and (val < t.train_min or val > t.train_max):
                violated.append(t.param_name)
        return len(violated) > 0, violated

    def evaluate_extrapolation(
        self,
        predictions: Tensor,
        targets: Tensor,
        baseline_predictions: Tensor,
        is_ood: bool,
    ) -> dict[str, float]:
        """Compare extrapolation vs interpolation on OOD samples.

        Args:
            predictions: In-context model predictions.
            targets: Ground truth.
            baseline_predictions: Interpolation baseline predictions.
            is_ood: Whether this is an OOD sample.

        Returns:
            Dict with MSE metrics.
        """
        mse_ic = (predictions - targets).pow(2).mean().item()
        mse_baseline = (baseline_predictions - targets).pow(2).mean().item()
        return {
            "mse_in_context": mse_ic,
            "mse_baseline": mse_baseline,
            "improvement_ratio": (
                (mse_baseline - mse_ic) / max(mse_baseline, 1e-12)
            ),
            "is_ood": is_ood,
        }


# ---------------------------------------------------------------------------
# Unified benchmark
# ---------------------------------------------------------------------------


class UnifiedTensorBenchmark:
    """Cross-domain benchmark for unified tensor interface + OOD gate."""

    @staticmethod
    def run(n_seeds: int = 3) -> dict:
        """Run cross-domain validation benchmark."""
        results: dict[str, list] = {
            "cfd": [],
            "electromagnetic": [],
            "lithography": [],
            "resist": [],
        }

        specs = {
            "cfd": cfd_spec(grid_size=16),
            "electromagnetic": electromagnetic_spec(grid_size=16),
            "lithography": lithography_spec(grid_size=16),
            "resist": resist_spec(grid_size=8),
        }

        for seed in range(n_seeds):
            torch.manual_seed(seed * 100 + 7)
            for domain_name, spec in specs.items():
                nf = len(spec.fields)
                gs = spec.grid_size
                if spec.spatial_dims == 2:
                    data = torch.randn(4, nf, gs, gs)
                else:
                    data = torch.randn(4, nf, gs, gs, gs)

                ut = UnifiedTensor(data, spec)

                flat = ut.to_latent_flat()
                recovered = UnifiedTensor.from_latent_flat(flat, spec)

                reconstruction_err = (ut.data - recovered.data).abs().max().item()
                results[domain_name].append(reconstruction_err)

        aggregated = {}
        for domain_name, errors in results.items():
            aggregated[domain_name] = {
                "mean_reconstruction_err": sum(errors) / len(errors),
                "max_reconstruction_err": max(errors),
            }

        aggregated["n_seeds"] = n_seeds
        return aggregated
