"""Probabilistic neural operator with proper scoring rule training.

References:
    - Probabilistic Neural Operators, arXiv:2502.12902, 2025
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .conformal import SplitConformalPredictor


class DistributionHead(nn.Module):
    """Maps backbone features to (mean, log_scale) distribution parameters."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.mean_head = nn.Linear(in_features, out_features)
        self.log_scale_head = nn.Linear(in_features, out_features)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        mean = self.mean_head(features)
        scale = F.softplus(self.log_scale_head(features))
        return mean, scale


class EnergyScoreLoss(nn.Module):
    """Energy score proper scoring rule for multivariate outputs.

    E(y, F) = E[||Y - y||] - 0.5 * E[||Y - Y'||]  where Y, Y' ~ F.
    """

    def forward(
        self,
        predictions_mean: Tensor,
        predictions_scale: Tensor,
        targets: Tensor,
        n_samples: int = 16,
    ) -> Tensor:
        eps = torch.randn(n_samples, *targets.shape, device=targets.device, dtype=targets.dtype)
        samples = predictions_mean.unsqueeze(0) + predictions_scale.unsqueeze(0) * eps

        diff = samples - targets.unsqueeze(0)
        term1 = diff.norm(dim=-1).mean(dim=0)

        diff_prime = samples.unsqueeze(0) - samples.unsqueeze(1)
        term2 = 0.5 * diff_prime.norm(dim=-1).mean(dim=0).mean(dim=0)

        per_point = term1 - term2
        return per_point.mean()


class CRPSLoss(nn.Module):
    """Continuous Ranked Probability Score for univariate Gaussian predictions.

    CRPS = scale * (z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi))
    where z = (y - mu) / scale.
    """

    _SQRT_PI_INV: float = 1.0 / (torch.pi ** 0.5)

    def forward(self, mean: Tensor, scale: Tensor, target: Tensor) -> Tensor:
        z = (target - mean) / scale
        phi = torch.exp(-0.5 * z**2) * (2.0 * torch.pi) ** -0.5
        Phi = 0.5 * (1.0 + torch.erf(z / (2.0**0.5)))
        per_point = scale * (z * (2.0 * Phi - 1.0) + 2.0 * phi - self._SQRT_PI_INV)
        return per_point.mean()


class ProbabilisticSurrogate(nn.Module):
    """Backbone + DistributionHead producing a Gaussian predictive distribution."""

    def __init__(
        self,
        backbone: nn.Module,
        in_features: int,
        out_features: int = 1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.dist_head = DistributionHead(in_features, out_features)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.backbone(x)
        if features.ndim > 2:
            features = features.reshape(features.shape[0], -1)
        return self.dist_head(features)

    def sample(self, x: Tensor, n_samples: int = 16) -> Tensor:
        mean, scale = self.forward(x)
        eps = torch.randn(n_samples, *mean.shape, device=mean.device, dtype=mean.dtype)
        return mean.unsqueeze(0) + scale.unsqueeze(0) * eps

    def predict_interval(self, x: Tensor, alpha: float = 0.1) -> tuple[Tensor, Tensor]:
        samples = self.sample(x, n_samples=512)
        lower_q = alpha / 2.0
        upper_q = 1.0 - alpha / 2.0
        lower = torch.quantile(samples, lower_q, dim=0)
        upper = torch.quantile(samples, upper_q, dim=0)
        return lower, upper

    def loss(
        self,
        x: Tensor,
        target: Tensor,
        scoring_rule: str = "energy",
        n_samples: int = 16,
    ) -> Tensor:
        mean, scale = self.forward(x)
        if mean.ndim > 1 and mean.shape[-1] == 1:
            mean = mean.squeeze(-1)
        if scale.ndim > 1 and scale.shape[-1] == 1:
            scale = scale.squeeze(-1)
        if target.ndim > 1 and target.shape[-1] == 1:
            target = target.squeeze(-1)

        if scoring_rule == "energy":
            if target.ndim == 1:
                target = target.unsqueeze(-1)
                mean = mean.unsqueeze(-1)
                scale = scale.unsqueeze(-1)
            return EnergyScoreLoss()(mean, scale, target, n_samples=n_samples)
        elif scoring_rule == "crps":
            return CRPSLoss()(mean, scale, target)
        else:
            raise ValueError(f"Unknown scoring_rule: {scoring_rule}")


class PNOConformalPipeline:
    """Chains ProbabilisticSurrogate with SplitConformalPredictor for dual UQ."""

    def __init__(
        self,
        surrogate: ProbabilisticSurrogate,
        lr: float = 1e-3,
    ) -> None:
        self.surrogate = surrogate
        self.lr = lr
        self.conformal = SplitConformalPredictor()
        self._pno_trained = False
        self._conformal_calibrated = False

    def train_pno(
        self,
        train_loader: torch.utils.data.DataLoader,
        n_epochs: int = 50,
        scoring_rule: str = "energy",
        n_samples: int = 16,
    ) -> list[float]:
        optimizer = torch.optim.Adam(self.surrogate.parameters(), lr=self.lr)
        history: list[float] = []

        self.surrogate.train()
        for _ in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                x, target = batch[0], batch[1]
                optimizer.zero_grad()
                loss = self.surrogate.loss(x, target, scoring_rule=scoring_rule, n_samples=n_samples)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            history.append(epoch_loss / max(1, n_batches))

        self._pno_trained = True
        return history

    def calibrate_conformal(
        self,
        cal_predictions: Tensor,
        cal_targets: Tensor,
        alpha: float = 0.1,
    ) -> None:
        if not self._pno_trained:
            raise RuntimeError("Must call train_pno() before calibrate_conformal()")
        self.conformal.calibrate(cal_predictions, cal_targets, alpha=alpha)
        self._conformal_calibrated = True

    def predict(self, x: Tensor, alpha: float = 0.1) -> dict[str, Tensor]:
        self.surrogate.eval()
        with torch.no_grad():
            mean, scale = self.surrogate(x)
            if mean.ndim > 1 and mean.shape[-1] == 1:
                mean = mean.squeeze(-1)
            if scale.ndim > 1 and scale.shape[-1] == 1:
                scale = scale.squeeze(-1)

            pno_lower, pno_upper = self.surrogate.predict_interval(x, alpha=alpha)
            if pno_lower.ndim > 1 and pno_lower.shape[-1] == 1:
                pno_lower = pno_lower.squeeze(-1)
            if pno_upper.ndim > 1 and pno_upper.shape[-1] == 1:
                pno_upper = pno_upper.squeeze(-1)

        result: dict[str, Tensor] = {
            "mean": mean,
            "scale": scale,
            "pno_lower": pno_lower,
            "pno_upper": pno_upper,
        }

        if self._conformal_calibrated:
            conf_lower, conf_upper = self.conformal.predict(mean)
            result["conformal_lower"] = conf_lower
            result["conformal_upper"] = conf_upper

        return result


class PNOBenchmark:
    """Compare PNO+conformal, ensemble+conformal, and pure conformal."""

    @staticmethod
    def run(
        surrogate: ProbabilisticSurrogate,
        train_x: Tensor,
        train_y: Tensor,
        test_x: Tensor,
        test_y: Tensor,
        ood_x: Tensor,
        ood_y: Tensor,
        n_epochs: int = 30,
        lr: float = 1e-3,
        alpha: float = 0.1,
        n_seeds: int = 3,
        scoring_rule: str = "energy",
    ) -> dict[str, dict]:
        from .ensemble import EnsembleSurrogate
        from .mlp import MLPSurrogate

        n_cal = train_x.shape[0] // 2
        perm = torch.randperm(train_x.shape[0])
        cal_x, cal_y = train_x[perm[:n_cal]], train_y[perm[:n_cal]]
        tr_x, tr_y = train_x[perm[n_cal:]], train_y[perm[n_cal:]]

        train_ds = torch.utils.data.TensorDataset(tr_x, tr_y)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)

        results: dict[str, dict] = {}

        # --- PNO + conformal ---
        pno_results: dict[str, list[float]] = {"id_coverage": [], "id_bandwidth": [], "ood_coverage": []}
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            pno = copy.deepcopy(surrogate)
            pipeline = PNOConformalPipeline(pno, lr=lr)
            pipeline.train_pno(train_loader, n_epochs=n_epochs, scoring_rule=scoring_rule)

            with torch.no_grad():
                cal_mean, _ = pno(cal_x)
                if cal_mean.ndim > 1 and cal_mean.shape[-1] == 1:
                    cal_mean = cal_mean.squeeze(-1)
            pipeline.calibrate_conformal(cal_mean, cal_y, alpha=alpha)

            id_pred = pipeline.predict(test_x, alpha=alpha)
            ood_pred = pipeline.predict(ood_x, alpha=alpha)

            id_cov = ((test_y >= id_pred["conformal_lower"]) & (test_y <= id_pred["conformal_upper"])).float().mean().item()
            id_bw = (id_pred["conformal_upper"] - id_pred["conformal_lower"]).mean().item()
            ood_cov = ((ood_y >= ood_pred["conformal_lower"]) & (ood_y <= ood_pred["conformal_upper"])).float().mean().item()

            pno_results["id_coverage"].append(id_cov)
            pno_results["id_bandwidth"].append(id_bw)
            pno_results["ood_coverage"].append(ood_cov)

        results["pno_conformal"] = {k: sum(v) / len(v) for k, v in pno_results.items()}

        # --- Ensemble + conformal ---
        ens_results: dict[str, list[float]] = {"id_coverage": [], "id_bandwidth": [], "ood_coverage": []}
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            in_dim = train_x.shape[-1]

            def _factory(_d=in_dim):
                return MLPSurrogate(n_inputs=_d, properties=["value"], hidden=32, n_layers=3)

            ensemble = EnsembleSurrogate(base_factory=_factory, n_members=5, seed=seed)
            train_y_dict = {"value": tr_y}
            for member in ensemble._members:
                member._data_generator = lambda _n, _i=tr_x, _t=train_y_dict: (_i, _t)
            ensemble.train_surrogate(n_samples=tr_x.shape[0], n_epochs=n_epochs, lr=lr)

            with torch.no_grad():
                cal_pred = ensemble.predict(cal_x)["value"]
                test_pred = ensemble.predict(test_x)["value"]
                ood_pred_e = ensemble.predict(ood_x)["value"]

            cp = SplitConformalPredictor()
            cp.calibrate(cal_pred, cal_y, alpha=alpha)

            id_lower, id_upper = cp.predict(test_pred)
            ood_lower, ood_upper = cp.predict(ood_pred_e)

            id_cov = ((test_y >= id_lower) & (test_y <= id_upper)).float().mean().item()
            id_bw = (id_upper - id_lower).mean().item()
            ood_cov = ((ood_y >= ood_lower) & (ood_y <= ood_upper)).float().mean().item()

            ens_results["id_coverage"].append(id_cov)
            ens_results["id_bandwidth"].append(id_bw)
            ens_results["ood_coverage"].append(ood_cov)

        results["ensemble_conformal"] = {k: sum(v) / len(v) for k, v in ens_results.items()}

        # --- Pure conformal on point predictions ---
        point_results: dict[str, list[float]] = {"id_coverage": [], "id_bandwidth": [], "ood_coverage": []}
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            point_net = nn.Sequential(
                nn.Linear(train_x.shape[-1], 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
            opt = torch.optim.Adam(point_net.parameters(), lr=lr)
            for _ in range(n_epochs):
                opt.zero_grad()
                pred = point_net(tr_x).squeeze(-1)
                loss = F.mse_loss(pred, tr_y)
                loss.backward()
                opt.step()

            with torch.no_grad():
                cal_pred = point_net(cal_x).squeeze(-1)
                test_pred = point_net(test_x).squeeze(-1)
                ood_pred_p = point_net(ood_x).squeeze(-1)

            cp = SplitConformalPredictor()
            cp.calibrate(cal_pred, cal_y, alpha=alpha)

            id_lower, id_upper = cp.predict(test_pred)
            ood_lower, ood_upper = cp.predict(ood_pred_p)

            id_cov = ((test_y >= id_lower) & (test_y <= id_upper)).float().mean().item()
            id_bw = (id_upper - id_lower).mean().item()
            ood_cov = ((ood_y >= ood_lower) & (ood_y <= ood_upper)).float().mean().item()

            point_results["id_coverage"].append(id_cov)
            point_results["id_bandwidth"].append(id_bw)
            point_results["ood_coverage"].append(ood_cov)

        results["point_conformal"] = {k: sum(v) / len(v) for k, v in point_results.items()}

        return results
