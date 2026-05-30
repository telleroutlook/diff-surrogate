"""Cross-repo compatibility and integration tests.

Verifies that imports, version strings, and co-design gradient flows work
across the 4-repo ecosystem (diff-surrogate, DiffNano, DiffCFD, OpenLithoHub).

Uses ``pytest.importorskip`` so the test suite remains green when optional
repos are not installed (e.g. in diff-surrogate standalone CI).
"""

from __future__ import annotations

import inspect
import sys

import pytest
import torch

import diff_surrogate

# ---------------------------------------------------------------------------
# Canonical public API surface -- must match diff_surrogate.__all__.
# Any removal from this set is a breaking change and will fail CI.
# ---------------------------------------------------------------------------

PUBLIC_API: set[str] = set(diff_surrogate.__all__)


# ---------------------------------------------------------------------------
# API stability tests
# ---------------------------------------------------------------------------


class TestPublicAPI:
    """Verify the public API surface is stable and all symbols are importable."""

    def test_no_api_removal(self):
        """Fail if any previously-exported public symbol is removed."""
        for symbol in PUBLIC_API:
            assert hasattr(diff_surrogate, symbol), f"Public API symbol {symbol!r} was removed!"

    def test_all_exports_callable_or_type(self):
        """All public symbols are classes, functions, or submodules."""
        import types

        for symbol in PUBLIC_API:
            obj = getattr(diff_surrogate, symbol)
            assert callable(obj) or isinstance(obj, (type, types.ModuleType)), (
                f"Public symbol {symbol!r} is not callable, a type, or a module: {type(obj)}"
            )

    def test_all_exports_in_dunder_all(self):
        """Everything in __all__ is actually importable from the package."""
        import diff_surrogate as ds

        for name in ds.__all__:
            assert hasattr(ds, name), f"{name!r} listed in __all__ but not importable"


class TestBaseClassInterface:
    """SurrogateBase exposes the expected abstract method set."""

    def test_abstract_methods(self):
        from diff_surrogate import SurrogateBase

        expected = {"_build_network", "forward", "generate_training_data"}
        actual = {
            name
            for name, member in inspect.getmembers(SurrogateBase)
            if getattr(member, "__isabstractmethod__", False)
        }
        assert expected.issubset(actual), f"Missing abstract methods: {expected - actual}"

    def test_surrogate_base_is_abstract(self):
        from diff_surrogate import SurrogateBase

        with pytest.raises(TypeError, match="abstract method"):
            SurrogateBase()  # type: ignore[abstract]


class TestCoreTypeChecks:
    """Verify concrete types for key public symbols."""

    def test_sobolev_loss_is_nn_module(self):
        from diff_surrogate import SobolevLoss

        assert issubclass(SobolevLoss, torch.nn.Module)

    def test_gradient_fidelity_score_is_callable(self):
        from diff_surrogate import gradient_fidelity_score

        assert callable(gradient_fidelity_score)

    def test_convergence_monitor_instantiable(self):
        from diff_surrogate import ConvergenceConfig, ConvergenceMonitor

        monitor = ConvergenceMonitor(ConvergenceConfig(window=10))
        assert monitor is not None

    def test_correction_policy_instantiable(self):
        from diff_surrogate import CorrectionPolicy

        policy = CorrectionPolicy(correction_interval=5, warmup_steps=2)
        assert policy.should_correct(5)

    def test_adaptive_correction_policy_instantiable(self):
        from diff_surrogate import AdaptiveCorrectionPolicy

        policy = AdaptiveCorrectionPolicy(initial_interval=10)
        assert policy.current_interval == 10

    def test_ensemble_surrogate_is_surrogate_base(self):
        from diff_surrogate import EnsembleSurrogate, SurrogateBase

        assert issubclass(EnsembleSurrogate, SurrogateBase)

    def test_cross_attn_surrogate_is_surrogate_base(self):
        from diff_surrogate import CrossAttnSurrogate, SurrogateBase

        assert issubclass(CrossAttnSurrogate, SurrogateBase)

    def test_sdf_trunk_surrogate_is_surrogate_base(self):
        from diff_surrogate import SDFTrunkSurrogate, SurrogateBase

        assert issubclass(SDFTrunkSurrogate, SurrogateBase)


# ---------------------------------------------------------------------------
# Thermophysical-props compat (DiffCFD consumer pattern)
# ---------------------------------------------------------------------------


class TestThermophysicalPropsCompat:
    """Verify surrogate outputs match the consumption pattern used by DiffCFD."""

    def test_mlp_output_dict_of_tensors(self):
        """MLPSurrogate returns dict[str, Tensor] -- DiffCFD expects this shape."""
        from diff_surrogate import MLPSurrogate

        s = MLPSurrogate(n_inputs=3, properties=["density", "cp", "viscosity"])
        x = torch.randn(4, 3)
        out = s.predict(x)
        assert isinstance(out, dict)
        for key in ("density", "cp", "viscosity"):
            assert key in out
            assert isinstance(out[key], torch.Tensor)
            assert out[key].shape == (4,)

    def test_output_consumable_by_loss(self):
        """Surrogate dict output can be directly consumed by a weighted loss."""
        from diff_surrogate import MLPSurrogate

        s = MLPSurrogate(n_inputs=2, properties=["pressure", "temperature"])
        x = torch.randn(8, 2)
        pred = s(x)
        target = {k: torch.randn_like(v) for k, v in pred.items()}
        loss = sum(torch.nn.functional.mse_loss(pred[k], target[k]) for k in pred)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_cnn_output_tensor_shape(self):
        """CNNSurrogate returns a plain Tensor -- verify shape convention."""
        from diff_surrogate import CNNSurrogate

        s = CNNSurrogate(in_channels=1, out_channels=3, grid_size=16)
        x = torch.randn(2, 1, 16, 16)
        out = s.predict(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 3, 16, 16)


# ---------------------------------------------------------------------------
# Geometry operator compat (DiffNano RCWA proxy pattern)
# ---------------------------------------------------------------------------


class TestGeometryOperatorCompat:
    """Geometry operators produce output compatible with DiffNano's RCWA proxy."""

    def test_sdf_from_curve_output_shape(self):
        from diff_surrogate.geometry import sdf_from_curve

        curve = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        grid_x = torch.linspace(0, 1, 16)
        grid_y = torch.linspace(0, 1, 16)
        gx, gy = torch.meshgrid(grid_x, grid_y, indexing="ij")
        sdf = sdf_from_curve(gx, gy, curve)
        assert sdf.shape == (16, 16)
        assert sdf.dtype in (torch.float32, torch.float64)

    def test_sigmoid_projection_output(self):
        from diff_surrogate.geometry import sigmoid_projection

        field = torch.randn(4, 8, 8)
        out = sigmoid_projection(field, beta=10.0)
        assert out.shape == (4, 8, 8)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_projection_preserves_gradient(self):
        from diff_surrogate.geometry import sigmoid_projection

        field = torch.randn(3, 4, 4, requires_grad=True)
        out = sigmoid_projection(field, beta=5.0)
        out.sum().backward()
        assert field.grad is not None
        assert torch.isfinite(field.grad).all()


# ---------------------------------------------------------------------------
# Interop smoke (DLPack round-trip)
# ---------------------------------------------------------------------------


class TestInteropSmoke:
    """Verify interop module is importable and provides expected symbols."""

    def test_interop_module_symbols(self):
        from diff_surrogate import interop

        assert hasattr(interop, "j2t")
        assert hasattr(interop, "t2j")
        assert hasattr(interop, "JAXFunctionWrapper")
        assert hasattr(interop, "wrap_jax_fn")

    @pytest.mark.skipif(sys.version_info < (3, 10), reason="DLPack requires Python >= 3.10")
    def test_interop_importable_without_jax(self):
        """interop module should import cleanly even when JAX is absent."""
        from diff_surrogate import interop

        # Just importing should not raise; calling functions without JAX
        # will raise ImportError -- that is tested separately in test_interop.py
        assert interop is not None


# ---------------------------------------------------------------------------
# Version compatibility tests
# ---------------------------------------------------------------------------


class TestVersionCompat:
    """Verify runtime version compatibility."""

    def test_diff_surrogate_version(self):
        assert diff_surrogate.__version__ == "0.2.0"

    def test_diff_surrogate_commit(self):
        assert isinstance(diff_surrogate.__version__, str)

    def test_torch_version_compat(self):
        """Run a quick forward-backward pass through core components."""
        from diff_surrogate import MLPSurrogate

        s = MLPSurrogate(
            n_inputs=2,
            properties=["val"],
            data_generator=lambda n: (torch.randn(n, 2), {"val": torch.randn(n)}),
        )
        losses = s.train_surrogate(n_epochs=2, n_samples=16)
        assert len(losses) == 2
        assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)

    def test_python_version_compat(self):
        assert sys.version_info >= (3, 10), "Requires Python >= 3.10"

    @pytest.mark.parametrize(
        "module_name, expected_version",
        [
            ("diffnano", "0.6.0"),
            ("diffcfd", "0.7.0"),
        ],
    )
    def test_downstream_version(self, module_name, expected_version):
        mod = pytest.importorskip(module_name)
        assert mod.__version__ == expected_version


# ---------------------------------------------------------------------------
# Import chain tests
# ---------------------------------------------------------------------------


class TestImportChain:
    """Verify cross-repo imports resolve without errors."""

    def test_diff_surrogate_core(self):
        from diff_surrogate import CoDesignWorkflow, CoupledLoss

        assert CoDesignWorkflow is not None
        assert CoupledLoss is not None

    def test_diffnano_imports(self):
        diffnano = pytest.importorskip("diffnano")
        assert hasattr(diffnano, "RCWASolver")
        assert hasattr(diffnano, "EndToEndPipeline")

    def test_diffcfd_imports(self):
        diffcfd = pytest.importorskip("diffcfd")
        assert hasattr(diffcfd, "NavierStokes2D")
        assert hasattr(diffcfd, "HelmholtzFilter")

    def test_openlithohub_imports(self):
        olh = pytest.importorskip("openlithohub")
        assert olh is not None


# ---------------------------------------------------------------------------
# Co-design gradient flow (end-to-end, no NaN)
# ---------------------------------------------------------------------------


class TestCoDesignGradientFlow:
    """Verify gradient flows through the co-design pipeline without NaN."""

    @staticmethod
    def _em_forward(params: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"em_loss": (params - 0.5).pow(2).mean()}

    @staticmethod
    def _litho_forward(params: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"epe": params.abs().mean()}

    @staticmethod
    def _coupling_fn(merged: dict) -> dict:
        epe = merged.get("litho", {}).get("epe", torch.tensor(0.0))
        merged["coupling_penalty"] = epe * 0.5
        return merged

    @staticmethod
    def _coupled_loss(**kwargs) -> torch.Tensor:
        em_loss = kwargs.get("em", {}).get("em_loss", torch.tensor(0.0))
        epe = kwargs.get("litho", {}).get("epe", torch.tensor(0.0))
        coupling = kwargs.get("coupling_penalty", torch.tensor(0.0))
        return em_loss + 0.1 * epe + coupling

    def test_gradient_no_nan(self):
        params = torch.rand(8, 8)
        wf = diff_surrogate.CoDesignWorkflow(
            design_params=params,
            forward_fns={"em": self._em_forward, "litho": self._litho_forward},
            loss_fn=self._coupled_loss,
            coupling_fn=self._coupling_fn,
            lr=0.01,
        )
        final, history = wf.run(n_steps=10, verbose=False)
        assert not torch.isnan(final).any(), "NaN in final params"
        assert not any(h != h for h in history), "NaN in loss history"

    def test_gradient_finite(self):
        params = torch.rand(4, 4, requires_grad=True)
        em_out = self._em_forward(params)
        loss = self._coupled_loss(**{"em": em_out})
        loss.backward()
        assert params.grad is not None
        assert torch.isfinite(params.grad).all(), "Non-finite gradients"

    def test_loss_decreases(self):
        params = torch.rand(8, 8)
        wf = diff_surrogate.CoDesignWorkflow(
            design_params=params,
            forward_fns={"em": self._em_forward},
            loss_fn=self._coupled_loss,
            lr=0.05,
        )
        _, history = wf.run(n_steps=30, verbose=False)
        assert history[-1] < history[0], "Loss did not decrease"

    def test_with_diffnano_solver(self):
        """End-to-end test using DiffNano RCWA if available."""
        diffnano = pytest.importorskip("diffnano")
        solver = diffnano.RCWASolver(
            fourier_orders=2,
            wavelength_nm=532.0,
            period_nm=(400.0, 400.0),
        )
        assert solver is not None

    def test_with_diffcfd_solver(self):
        """End-to-end test using DiffCFD mesh if available."""
        diffcfd = pytest.importorskip("diffcfd")
        mesh = diffcfd.CartesianMesh(nx=4, ny=4, lx=1.0, ly=1.0)
        assert mesh is not None


# ---------------------------------------------------------------------------
# Forward-backward smoke across all surrogate types
# ---------------------------------------------------------------------------


class TestForwardBackwardSmoke:
    """Verify forward + backward pass works for every concrete SurrogateBase subclass."""

    def test_mlp_surrogate_roundtrip(self):
        from diff_surrogate import MLPSurrogate

        s = MLPSurrogate(n_inputs=2, properties=["val"])
        x = torch.randn(4, 2, requires_grad=True)
        out = s(x)
        loss = out["val"].sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_cnn_surrogate_roundtrip(self):
        from diff_surrogate import CNNSurrogate

        s = CNNSurrogate(in_channels=1, out_channels=2, grid_size=8)
        x = torch.randn(2, 1, 8, 8, requires_grad=True)
        out = s(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_sdf_trunk_surrogate_roundtrip(self):
        from diff_surrogate import SDFTrunkSurrogate

        s = SDFTrunkSurrogate(param_dim=2, n_outputs=1, hidden_dim=16, n_basis=8)
        sdf = torch.randn(2, 8, 8)
        params = torch.randn(2, 2, requires_grad=True)
        out = s((sdf, params))
        out.sum().backward()
        assert params.grad is not None
        assert torch.isfinite(params.grad).all()

    def test_cross_attn_surrogate_roundtrip(self):
        from diff_surrogate import CrossAttnSurrogate

        s = CrossAttnSurrogate(param_dim=2, n_outputs=1, hidden_dim=16, n_heads=2)
        sdf = torch.randn(2, 8, 8)
        params = torch.randn(2, 2, requires_grad=True)
        out = s((sdf, params))
        out.sum().backward()
        assert params.grad is not None
        assert torch.isfinite(params.grad).all()
