"""Tests for cross-attention geometry operator surrogate."""

import torch

from diff_surrogate.cross_attn import CrossAttnSurrogate


def test_cross_attn_forward_shape():
    model = CrossAttnSurrogate(param_dim=2, n_outputs=3, hidden_dim=32, n_heads=4)
    model.get_network()

    sdf = torch.randn(4, 32, 32)
    params = torch.randn(4, 2)
    out = model((sdf, params))
    assert out.shape == (4, 3, 32, 32)


def test_cross_attn_sdf_only():
    model = CrossAttnSurrogate(param_dim=2, n_outputs=3, hidden_dim=32, n_heads=4)
    model.get_network()

    sdf = torch.randn(2, 16, 16)
    out = model(sdf)
    assert out.shape == (2, 3, 16, 16)


def test_cross_attn_gradient_flows():
    model = CrossAttnSurrogate(param_dim=2, n_outputs=3, hidden_dim=32, n_heads=4)
    model.get_network()

    sdf = torch.randn(2, 16, 16, requires_grad=True)
    params = torch.randn(2, 2, requires_grad=True)
    out = model((sdf, params))
    loss = out.sum()
    loss.backward()
    assert sdf.grad is not None
    assert params.grad is not None
    assert torch.isfinite(sdf.grad).all()
    assert torch.isfinite(params.grad).all()


def test_cross_attn_predict():
    model = CrossAttnSurrogate(param_dim=2, n_outputs=3, hidden_dim=32, n_heads=4)
    model.get_network()

    sdf = torch.randn(2, 16, 16)
    params = torch.randn(2, 2)
    out = model.predict((sdf, params))
    assert out.shape == (2, 3, 16, 16)
    assert torch.isfinite(out).all()


def test_cross_attn_train_converges():
    model = CrossAttnSurrogate(param_dim=2, n_outputs=3, hidden_dim=32, n_heads=4)
    model.get_network()

    torch.manual_seed(0)
    sdf = torch.randn(8, 16, 16)
    params = torch.randn(8, 2)
    targets = torch.randn(8, 3, 16, 16)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    initial_loss = None
    final_loss = None
    for epoch in range(20):
        opt.zero_grad()
        pred = model((sdf, params))
        loss = ((pred - targets) ** 2).mean()
        loss.backward()
        opt.step()
        if epoch == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert final_loss < initial_loss, "CrossAttn model should converge during training"
