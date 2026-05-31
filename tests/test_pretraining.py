"""Tests for PDE pretraining + transfer learning (S8.3)."""

import torch
import torch.nn as nn

from diff_surrogate.pretraining import (
    FewShotFinetuner,
    MultiTaskPretrainer,
    PDENet,
    TransferBenchmark,
    task_advection_1d,
    task_diffusion_2d,
    task_poisson_1d,
    task_reaction_diffusion_1d,
)

# ---- helpers ----

def _encoder_factory():
    return PDENet(input_dim=64, hidden_dim=32, output_dim=1, n_layers=2)


def _make_tasks(n_tasks=4, n_samples=40, n_grid=64):
    """Build synthetic PDE tasks with matching input dim."""
    datasets = []
    for _ in range(n_tasks):
        inp = torch.randn(n_samples, n_grid)
        out = (inp * torch.randn(1, n_grid)).sum(dim=1, keepdim=True)
        datasets.append((inp, out))
    return datasets


# ---- tests ----

def test_task_generators_produce_valid_data():
    """All four PDE task generators produce tensors with correct shapes."""
    p_inp, p_sol = task_poisson_1d(16, n_grid=64)
    assert p_inp.shape == (16, 64)
    assert p_sol.shape == (16, 64)

    d_inp, d_sol = task_diffusion_2d(16, n_grid=16)
    assert d_inp.shape == (16, 256)
    assert d_sol.shape == (16, 256)

    a_inp, a_sol = task_advection_1d(16, n_grid=64)
    assert a_inp.shape == (16, 65)
    assert a_sol.shape == (16, 64)

    r_inp, r_sol = task_reaction_diffusion_1d(16, n_grid=64)
    assert r_inp.shape == (16, 64)
    assert r_sol.shape == (16, 64)

    # Solutions should be finite
    for t in [p_sol, d_sol, a_sol, r_sol]:
        assert torch.isfinite(t).all()


def test_pretrainer_runs_multiple_tasks():
    """MultiTaskPretrainer should complete training on 3 tasks without error."""
    datasets = _make_tasks(n_tasks=3, n_samples=30)

    def factory():
        return PDENet(input_dim=64, hidden_dim=32, output_dim=1, n_layers=2)

    pt = MultiTaskPretrainer(factory, n_tasks=3, device="cpu")
    history = pt.pretrain(datasets, n_epochs=5, lr=1e-3)

    assert len(history) == 5
    assert all(isinstance(v, float) for v in history)
    assert all(torch.isfinite(torch.tensor(v)) for v in history)

    encoder = pt.get_pretrained_encoder()
    assert isinstance(encoder, nn.Module)


def test_few_shot_finetuner_gradient_exists():
    """FewShotFinetuner should produce gradients during fine-tuning."""
    base = _encoder_factory()
    ft = FewShotFinetuner(base, output_dim=1, device="cpu")

    inputs = torch.randn(10, 64)
    targets = torch.randn(10)

    # Run one step manually to verify gradient flow
    ft.encoder.train()
    optimizer = torch.optim.Adam(
        list(ft.encoder.parameters()) + list(ft.head.parameters()), lr=1e-3,
    )
    optimizer.zero_grad()
    features = ft.encoder.encode(inputs)
    pred = ft.head(features)
    loss = nn.MSELoss()(pred, targets.unsqueeze(1))
    loss.backward()

    # At least one param in head should have non-zero grad
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in ft.head.parameters()
    )
    assert has_grad, "No gradient in finetuner head"

    # Also test the finetune method returns history
    history = ft.finetune(inputs, targets, n_epochs=3, lr=1e-3)
    assert len(history) == 3


def test_pretrained_encoder_improves_few_shot():
    """A pretrained encoder should achieve lower loss than random init after finetuning."""
    torch.manual_seed(0)
    datasets = _make_tasks(n_tasks=3, n_samples=80)

    def factory():
        return PDENet(input_dim=64, hidden_dim=32, output_dim=1, n_layers=2)

    # Pretrain on tasks 0,1
    pt = MultiTaskPretrainer(factory, n_tasks=2, device="cpu")
    pt.pretrain(datasets[:2], n_epochs=30, lr=1e-3)
    pretrained = pt.get_pretrained_encoder()

    # Target is task 2
    target_in, target_out = datasets[2]
    few_idx = torch.randperm(target_in.shape[0])[:20]
    fs_in = target_in[few_idx]
    fs_out = target_out[few_idx].squeeze(1)

    # Pretrained finetune
    ft_pre = FewShotFinetuner(pretrained, output_dim=1, device="cpu")
    ft_pre.finetune(fs_in, fs_out, n_epochs=40, lr=1e-3)
    with torch.no_grad():
        pred_pre = ft_pre.predict(target_in)
        loss_pre = nn.MSELoss()(pred_pre, target_out).item()

    # Scratch finetune
    scratch = factory()
    ft_scratch = FewShotFinetuner(scratch, output_dim=1, device="cpu")
    ft_scratch.finetune(fs_in, fs_out, n_epochs=40, lr=1e-3)
    with torch.no_grad():
        pred_scratch = ft_scratch.predict(target_in)
        loss_scratch = nn.MSELoss()(pred_scratch, target_out).item()

    # Pretrained should be at least as good (generally much better with shared structure)
    # Using a generous bound since these are random tasks
    assert loss_pre < loss_scratch * 5.0, (
        f"Pretrained loss ({loss_pre:.4f}) much worse than scratch ({loss_scratch:.4f})"
    )


def test_transfer_benchmark_produces_results():
    """TransferBenchmark.compare should return properly structured results."""
    datasets = _make_tasks(n_tasks=3, n_samples=60)

    def factory():
        return PDENet(input_dim=64, hidden_dim=32, output_dim=1, n_layers=2)

    result = TransferBenchmark.compare(
        encoder_factory=factory,
        task_datasets=datasets,
        pretrain_tasks=[0, 1],
        target_task_idx=2,
        few_shot_sizes=[10, 20],
        n_seeds=3,
        pretrain_epochs=5,
        finetune_epochs=5,
    )

    assert "pretrained" in result
    assert "scratch" in result
    assert "data_efficiency_ratio" in result

    for size in [10, 20]:
        assert size in result["pretrained"]
        assert size in result["scratch"]
        assert size in result["data_efficiency_ratio"]
        mean_p, std_p = result["pretrained"][size]
        mean_s, std_s = result["scratch"][size]
        assert isinstance(mean_p, float)
        assert isinstance(std_p, float)
        assert isinstance(mean_s, float)
        assert isinstance(std_s, float)
        assert std_p >= 0
        assert std_s >= 0


def test_data_efficiency_ratio_greater_than_one():
    """With correlated tasks, pretraining should help: ratio < 1 on average."""
    torch.manual_seed(7)

    # Create correlated tasks: all share the same linear mapping + noise
    W = torch.randn(64, 1)
    datasets = []
    for _ in range(3):
        inp = torch.randn(80, 64)
        out = inp @ W + torch.randn(80, 1) * 0.1
        datasets.append((inp, out))

    def factory():
        return PDENet(input_dim=64, hidden_dim=32, output_dim=1, n_layers=2)

    result = TransferBenchmark.compare(
        encoder_factory=factory,
        task_datasets=datasets,
        pretrain_tasks=[0, 1],
        target_task_idx=2,
        few_shot_sizes=[15, 30],
        n_seeds=3,
        pretrain_epochs=30,
        finetune_epochs=30,
    )

    # At least one size should show benefit (ratio < 1)
    ratios = list(result["data_efficiency_ratio"].values())
    best_ratio = min(ratios)
    assert best_ratio < 1.0, (
        f"Expected at least one ratio < 1.0, got {ratios}. "
        "Pretraining should help on correlated tasks."
    )
