"""PDE pretraining + transfer learning for few-shot PDE solving.

Provides:
  - MultiTaskPretrainer: pre-train a shared encoder on multiple PDE tasks
  - FewShotFinetuner: fine-tune a pretrained encoder on a new task with few samples
  - TransferBenchmark: benchmark transfer learning vs training from scratch
  - Toy PDE task generators for benchmarking
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Toy PDE task generators
# ---------------------------------------------------------------------------


def task_poisson_1d(n_samples: int, n_grid: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """1-D Poisson equation: -u''(x) = f(x) with random source f.

    Solves on [0, 1] with Dirichlet BC u(0) = u(1) = 0 using a spectral
    approach (sin series).  Source is a random sum of sinusoids.

    Args:
        n_samples: Number of (source, solution) pairs.
        n_grid: Number of spatial grid points.

    Returns:
        params: (n_samples, n_grid) source term f(x).
        solutions: (n_samples, n_grid) solution u(x).
    """
    x = torch.linspace(0, 1, n_grid)
    max_modes = min(8, n_grid // 2)
    n_modes = torch.randint(1, max_modes + 1, (n_samples,))
    params = torch.zeros(n_samples, n_grid)
    solutions = torch.zeros(n_samples, n_grid)
    for i in range(n_samples):
        nm = n_modes[i].item()
        modes = torch.randint(1, 15, (nm,))
        amps = torch.randn(nm) * 0.5
        for k in range(nm):
            m = modes[k].item()
            params[i] += amps[k] * torch.sin(m * torch.pi * x)
            solutions[i] += amps[k] / (m * torch.pi) ** 2 * torch.sin(m * torch.pi * x)
    return params, solutions


def task_diffusion_2d(
    n_samples: int,
    n_grid: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """2-D diffusion with random initial condition evolved to t=0.1.

    Uses spectral method on [0,1]^2 with periodic BC.

    Args:
        n_samples: Number of samples.
        n_grid: Grid resolution per side.

    Returns:
        params: (n_samples, n_grid) flattened initial condition.
        solutions: (n_samples, n_grid*n_grid) evolved field.
    """
    kx = torch.fft.fftfreq(n_grid, d=1.0 / n_grid)
    ky = torch.fft.fftfreq(n_grid, d=1.0 / n_grid)
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    K2 = KX**2 + KY**2

    # Random low-frequency initial condition
    ic = torch.randn(n_samples, n_grid, n_grid)
    ic_f = torch.fft.fft2(ic)
    low_pass = ((n_grid // 4) ** 2 > K2).float()
    ic_f = ic_f * low_pass.unsqueeze(0)
    ic = torch.fft.ifft2(ic_f).real

    # Evolve with diffusion coeff=0.01 for t=0.1
    diffusivity = 0.01
    t_final = 0.1
    decay = torch.exp(-diffusivity * 4 * torch.pi**2 * K2 * t_final)
    evolved_f = torch.fft.fft2(ic) * decay.unsqueeze(0)
    evolved = torch.fft.ifft2(evolved_f).real

    return ic.reshape(n_samples, -1), evolved.reshape(n_samples, -1)


def task_advection_1d(n_samples: int, n_grid: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """1-D advection u_t + c * u_x = 0 with random velocity c and initial profile.

    Exact solution: u(x, t) = u0(x - c*t).

    Args:
        n_samples: Number of samples.
        n_grid: Number of spatial grid points.

    Returns:
        params: (n_samples, n_grid+1) initial condition + velocity.
        solutions: (n_samples, n_grid) advected field at t=0.5.
    """
    x = torch.linspace(0, 1, n_grid)
    ic = torch.zeros(n_samples, n_grid)
    for i in range(n_samples):
        center = torch.rand(1).item() * 0.6 + 0.2
        width = torch.rand(1).item() * 0.05 + 0.03
        ic[i] = torch.exp(-0.5 * ((x - center) / width) ** 2)

    velocity = (torch.randn(n_samples) * 0.3 + 0.5).unsqueeze(1)
    t = 0.5
    shifted_x = x.unsqueeze(0) - velocity * t
    shifted_x = shifted_x % 1.0  # periodic BC

    # Reconstruct shifted profile by interpolation
    solutions = torch.zeros(n_samples, n_grid)
    for i in range(n_samples):
        for j in range(n_grid):
            idx = shifted_x[i, j] * (n_grid - 1)
            lo = int(idx.floor().item()) % n_grid
            hi = (lo + 1) % n_grid
            frac = idx - int(idx.floor().item())
            solutions[i, j] = ic[i, lo] * (1 - frac) + ic[i, hi] * frac

    params = torch.cat([ic, velocity], dim=1)
    return params, solutions


def task_reaction_diffusion_1d(
    n_samples: int,
    n_grid: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """1-D reaction-diffusion: u_t = D*u_xx + k*u*(1-u) (Fisher-KPP).

    Solved with explicit Euler for a few steps from random IC.

    Args:
        n_samples: Number of samples.
        n_grid: Number of spatial grid points.

    Returns:
        params: (n_samples, n_grid) initial condition.
        solutions: (n_samples, n_grid) field after short time.
    """
    dx = 1.0 / (n_grid - 1)
    D = 0.005
    k = 5.0
    dt = 0.4 * dx**2 / (2 * D)  # CFL-safe
    n_steps = 50

    x = torch.linspace(0, 1, n_grid)
    ic = torch.zeros(n_samples, n_grid)
    for i in range(n_samples):
        center = torch.rand(1).item()
        width = torch.rand(1).item() * 0.05 + 0.03
        ic[i] = 0.5 + 0.5 * torch.tanh((x - center) / width)

    u = ic.clone()
    for _ in range(n_steps):
        laplacian = torch.zeros_like(u)
        laplacian[:, 1:-1] = (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / dx**2
        reaction = k * u * (1 - u)
        u = u + dt * (D * laplacian + reaction)
        u = u.clamp(0, 1)

    return ic, u


# ---------------------------------------------------------------------------
# Simple encoder for pretraining experiments
# ---------------------------------------------------------------------------


class PDENet(nn.Module):
    """Simple MLP encoder-decoder for PDE surrogate pretraining.

    Architecture: input -> trunk (shared) -> head (task-specific).

    Args:
        input_dim: Flattened input dimension.
        hidden_dim: Hidden layer width.
        output_dim: Output dimension.
        n_layers: Number of hidden layers in trunk.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        output_dim: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


# ---------------------------------------------------------------------------
# MultiTaskPretrainer
# ---------------------------------------------------------------------------


class MultiTaskPretrainer:
    """Pre-train a shared encoder on multiple PDE tasks.

    The shared encoder learns general PDE features across tasks.  Downstream
    tasks then fine-tune with few samples.

    Args:
        encoder_factory: Callable returning a fresh encoder module (e.g. PDENet).
        n_tasks: Number of pretraining tasks.
        task_dim: Embedding dimension for task-specific tokens.
        device: Torch device.
    """

    def __init__(
        self,
        encoder_factory: Callable[[], nn.Module],
        n_tasks: int = 4,
        task_dim: int = 16,
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        self.encoder_factory = encoder_factory
        self.shared_encoder = encoder_factory().to(self.device)

        # Infer hidden_dim from the encoder
        with torch.no_grad():
            in_dim = getattr(self.shared_encoder, "input_dim", 64)
            dummy = torch.zeros(1, in_dim)
            dummy = dummy.to(self.device)
            hidden = self.shared_encoder.encode(dummy)
            hidden_dim = hidden.shape[-1]

        self.task_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_tasks)]).to(
            self.device
        )
        self.task_embeddings = nn.Embedding(n_tasks, task_dim).to(self.device)
        self.task_proj = nn.Linear(task_dim, hidden_dim).to(self.device)
        self.n_tasks = n_tasks
        self.hidden_dim = hidden_dim

    def pretrain(
        self,
        task_datasets: list[tuple[torch.Tensor, torch.Tensor]],
        n_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 32,
    ) -> list[float]:
        """Pre-train shared encoder on all tasks simultaneously.

        Args:
            task_datasets: List of (inputs, targets) tuples, one per task.
                Each inputs is (N_i, input_dim), targets is (N_i,) or (N_i, output_dim).
            n_epochs: Training epochs.
            lr: Learning rate.
            batch_size: Mini-batch size.

        Returns:
            List of average loss per epoch.
        """
        assert len(task_datasets) == self.n_tasks, (
            f"Expected {self.n_tasks} task datasets, got {len(task_datasets)}"
        )

        params = list(self.shared_encoder.parameters())
        for head in self.task_heads:
            params.extend(head.parameters())
        params.extend(self.task_embeddings.parameters())
        params.extend(self.task_proj.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        loss_fn = nn.MSELoss()

        history: list[float] = []
        for _epoch in range(n_epochs):
            total_loss = 0.0
            n_batches = 0
            for task_idx in range(self.n_tasks):
                inputs, targets = task_datasets[task_idx]
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                if targets.ndim == 1:
                    targets = targets.unsqueeze(1)

                n = inputs.shape[0]
                perm = torch.randperm(n)
                for start in range(0, n, batch_size):
                    idx = perm[start : start + batch_size]
                    batch_x = inputs[idx]
                    batch_y = targets[idx]

                    optimizer.zero_grad()
                    features = self.shared_encoder.encode(batch_x)
                    task_id = torch.tensor([task_idx], device=self.device)
                    task_emb = self.task_proj(self.task_embeddings(task_id))
                    # task_emb is (1, hidden_dim); broadcast to (batch, hidden_dim)
                    combined = features + task_emb
                    pred = self.task_heads[task_idx](combined)
                    loss = loss_fn(pred, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    n_batches += 1

            avg_loss = total_loss / max(1, n_batches)
            history.append(avg_loss)

        return history

    def get_pretrained_encoder(self) -> nn.Module:
        """Return the pretrained shared encoder for downstream use."""
        return self.shared_encoder


# ---------------------------------------------------------------------------
# FewShotFinetuner
# ---------------------------------------------------------------------------


class FewShotFinetuner:
    """Fine-tune a pretrained encoder on a new task with few samples.

    Args:
        pretrained_encoder: Encoder module (must have ``encode`` method).
        output_dim: Dimension of the prediction target.
        device: Torch device.
    """

    def __init__(
        self,
        pretrained_encoder: nn.Module,
        output_dim: int = 1,
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        self.encoder = copy.deepcopy(pretrained_encoder).to(self.device)

        # Infer hidden dim
        with torch.no_grad():
            in_dim = getattr(self.encoder, "input_dim", 64)
            dummy = torch.zeros(1, in_dim)
            dummy = dummy.to(self.device)
            hidden = self.encoder.encode(dummy)
            hidden_dim = hidden.shape[-1]

        self.head = nn.Linear(hidden_dim, output_dim).to(self.device)

    def finetune(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        n_epochs: int = 50,
        lr: float = 1e-3,
    ) -> list[float]:
        """Fine-tune on the new task.

        Args:
            inputs: (N, input_dim) input tensor.
            targets: (N,) or (N, output_dim) target tensor.
            n_epochs: Number of fine-tuning epochs.
            lr: Learning rate.

        Returns:
            Per-epoch loss history.
        """
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        if targets.ndim == 1:
            targets = targets.unsqueeze(1)

        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.head.parameters()),
            lr=lr,
        )
        loss_fn = nn.MSELoss()

        history: list[float] = []
        for _ in range(n_epochs):
            optimizer.zero_grad()
            features = self.encoder.encode(inputs)
            pred = self.head(features)
            loss = loss_fn(pred, targets)
            loss.backward()
            optimizer.step()
            history.append(loss.item())

        return history

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict on new inputs.

        Args:
            x: (N, input_dim) input tensor.

        Returns:
            (N, output_dim) predictions.
        """
        self.encoder.eval()
        with torch.no_grad():
            x = x.to(self.device)
            features = self.encoder.encode(x)
            return self.head(features)


# ---------------------------------------------------------------------------
# TransferBenchmark
# ---------------------------------------------------------------------------


class TransferBenchmark:
    """Benchmark transfer learning vs training from scratch."""

    @staticmethod
    def compare(
        encoder_factory: Callable[[], nn.Module],
        task_datasets: list[tuple[torch.Tensor, torch.Tensor]],
        pretrain_tasks: list[int],
        target_task_idx: int,
        few_shot_sizes: list[int] | None = None,
        n_seeds: int = 5,
        pretrain_epochs: int = 80,
        finetune_epochs: int = 50,
        lr: float = 1e-3,
    ) -> dict:
        """Compare pretrained vs scratch performance on a target task.

        Args:
            encoder_factory: Callable producing fresh encoders.
            task_datasets: List of (inputs, targets) per task.
            pretrain_tasks: Indices of tasks to pretrain on.
            target_task_idx: Index of the held-out target task.
            few_shot_sizes: List of few-shot sample counts.
            n_seeds: Number of random seeds for variance estimation.
            pretrain_epochs: Epochs for multi-task pretraining.
            finetune_epochs: Epochs for fine-tuning / scratch training.
            lr: Learning rate.

        Returns:
            Dict with:
                'pretrained': {size: (mean_loss, std_loss)}
                'scratch': {size: (mean_loss, std_loss)}
                'data_efficiency_ratio': pretrained/scratch loss ratio
        """
        if few_shot_sizes is None:
            few_shot_sizes = [10, 20, 50, 100]

        target_inputs, target_targets = task_datasets[target_task_idx]
        if target_targets.ndim == 1:
            target_targets = target_targets.unsqueeze(1)

        pretrained_results: dict[int, list[float]] = {s: [] for s in few_shot_sizes}
        scratch_results: dict[int, list[float]] = {s: [] for s in few_shot_sizes}

        for seed in range(n_seeds):
            torch.manual_seed(seed * 1000 + 42)

            # --- Pretrained path ---
            pretrain_data = [task_datasets[i] for i in pretrain_tasks]
            pretrainer = MultiTaskPretrainer(
                encoder_factory,
                n_tasks=len(pretrain_tasks),
                device="cpu",
            )
            pretrainer.pretrain(pretrain_data, n_epochs=pretrain_epochs, lr=lr)
            base_encoder = pretrainer.get_pretrained_encoder()

            # --- Scratch path: fresh encoder ---
            scratch_encoder = encoder_factory()

            for size in few_shot_sizes:
                # Subsample target task
                perm = torch.randperm(target_inputs.shape[0])[:size]
                fs_inputs = target_inputs[perm]
                fs_targets = target_targets[perm]

                # Pretrained finetune
                out_dim = target_targets.shape[-1]
                ft = FewShotFinetuner(base_encoder, output_dim=out_dim, device="cpu")
                ft.finetune(fs_inputs, fs_targets, n_epochs=finetune_epochs, lr=lr)
                with torch.no_grad():
                    pred = ft.predict(target_inputs)
                    pretrain_loss = F.mse_loss(pred, target_targets).item()
                pretrained_results[size].append(pretrain_loss)

                # Scratch train
                out_dim = target_targets.shape[-1]
                scratch_ft = FewShotFinetuner(scratch_encoder, output_dim=out_dim, device="cpu")
                scratch_ft.finetune(fs_inputs, fs_targets, n_epochs=finetune_epochs, lr=lr)
                with torch.no_grad():
                    pred_s = scratch_ft.predict(target_inputs)
                    scratch_loss = F.mse_loss(pred_s, target_targets).item()
                scratch_results[size].append(scratch_loss)

        # Aggregate
        pretrained_agg = {}
        scratch_agg = {}
        for size in few_shot_sizes:
            pvals = pretrained_results[size]
            svals = scratch_results[size]
            pretrained_agg[size] = (
                sum(pvals) / len(pvals),
                (sum((v - sum(pvals) / len(pvals)) ** 2 for v in pvals) / len(pvals)) ** 0.5,
            )
            scratch_agg[size] = (
                sum(svals) / len(svals),
                (sum((v - sum(svals) / len(svals)) ** 2 for v in svals) / len(svals)) ** 0.5,
            )

        # Data efficiency ratio: how much better pretrained is
        # ratio < 1 means pretrained wins
        efficiency = {}
        for size in few_shot_sizes:
            s_mean = scratch_agg[size][0]
            p_mean = pretrained_agg[size][0]
            efficiency[size] = p_mean / s_mean if s_mean > 0 else float("inf")

        return {
            "pretrained": pretrained_agg,
            "scratch": scratch_agg,
            "data_efficiency_ratio": efficiency,
        }
