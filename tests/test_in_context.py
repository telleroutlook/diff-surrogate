"""Tests for in-context / test-time adaptive operator (S10.2)."""

import torch
import torch.nn as nn

from diff_surrogate.conformal import SplitConformalPredictor
from diff_surrogate.in_context import (
    ContextPairEncoder,
    InContextAttention,
    InContextBenchmark,
    InContextOperator,
)
from diff_surrogate.pretraining import PDENet


def _make_operator(
    input_dim: int = 32,
    output_dim: int = 32,
    embed_dim: int = 32,
    n_context_heads: int = 2,
):
    backbone = PDENet(
        input_dim=input_dim,
        hidden_dim=embed_dim,
        output_dim=output_dim,
        n_layers=2,
    )
    ctx_enc = ContextPairEncoder(
        input_dim=input_dim,
        output_dim=output_dim,
        embed_dim=embed_dim,
    )
    return InContextOperator(
        backbone=backbone,
        context_encoder=ctx_enc,
        output_dim=output_dim,
        n_context_heads=n_context_heads,
    )


def test_context_pair_encoder_shape():
    """ContextPairEncoder should produce (n_pairs, embed_dim) output."""
    enc = ContextPairEncoder(input_dim=16, output_dim=8, embed_dim=32)
    inputs = torch.randn(5, 16)
    outputs = torch.randn(5, 8)
    tokens = enc(inputs, outputs)
    assert tokens.shape == (5, 32)


def test_context_pair_encoder_different_pairs():
    """Different input pairs should produce different embeddings."""
    enc = ContextPairEncoder(input_dim=8, output_dim=8, embed_dim=16)
    x1 = torch.randn(3, 8)
    y1 = torch.randn(3, 8)
    x2 = x1 + 10.0
    y2 = y1 + 10.0

    t1 = enc(x1, y1)
    t2 = enc(x2, y2)
    assert not torch.allclose(t1, t2, atol=1e-6)


def test_in_context_attention_shape():
    """InContextAttention should preserve query shape."""
    attn = InContextAttention(embed_dim=32, n_heads=4)
    query = torch.randn(4, 32)
    context = torch.randn(7, 32)
    out = attn(query, context)
    assert out.shape == (4, 32)


def test_in_context_operator_adapt_changes_output():
    """Prediction should change after adapt() is called with new context."""
    op = _make_operator()
    x = torch.randn(3, 32)

    pred_before = op.predict(x).clone()

    ctx_x = torch.randn(5, 32)
    ctx_y = torch.randn(5, 32)
    op.adapt(ctx_x, ctx_y)

    pred_after = op.predict(x).clone()
    assert not torch.allclose(pred_before, pred_after, atol=1e-6)


def test_in_context_operator_no_gradient_during_adapt():
    """adapt() should not create or require any gradients."""
    op = _make_operator()
    op.train()

    ctx_x = torch.randn(5, 32, requires_grad=True)
    ctx_y = torch.randn(5, 32, requires_grad=True)

    op.adapt(ctx_x.detach(), ctx_y.detach())

    for name, param in op.named_parameters():
        assert param.grad is None, f"Parameter {name} has gradient after adapt()"


def test_in_context_operator_confidence():
    """predict_with_confidence should return mean and std of matching shape."""
    op = _make_operator()
    ctx_x = torch.randn(5, 32)
    ctx_y = torch.randn(5, 32)
    op.adapt(ctx_x, ctx_y)

    x = torch.randn(3, 32)
    mean_pred, std_pred = op.predict_with_confidence(x, n_forward_passes=4)
    assert mean_pred.shape == (3, 32)
    assert std_pred.shape == (3, 32)
    assert (std_pred >= 0).all()


def test_in_context_vs_adapter_benchmark():
    """InContextBenchmark.run should return results for all methods and sizes."""
    result = InContextBenchmark.run(
        n_seeds=2,
        few_shot_sizes=[5, 10],
        n_grid=16,
        embed_dim=16,
    )

    expected_methods = {"in_context", "adapter_10", "adapter_30", "from_scratch"}
    assert set(result.keys()) == expected_methods

    for method in expected_methods:
        for size in [5, 10]:
            assert size in result[method], f"Missing size {size} for {method}"
            mean_err, std_err, adapt_time = result[method][size]
            assert isinstance(mean_err, float)
            assert isinstance(std_err, float)
            assert isinstance(adapt_time, float)
            assert std_err >= 0


def test_in_context_with_conformal():
    """In-context predictions should work with SplitConformalPredictor."""
    op = _make_operator(input_dim=16, output_dim=16, embed_dim=16)

    cal_inputs = torch.randn(30, 16)
    cal_targets = torch.randn(30, 16)
    op.adapt(cal_inputs, cal_targets)

    cal_pred = op.predict(cal_inputs)

    cp = SplitConformalPredictor()
    cp.calibrate(cal_pred, cal_targets, alpha=0.1)

    test_inputs = torch.randn(10, 16)
    test_pred = op.predict(test_inputs)
    lower, upper = cp.predict(test_pred)

    assert lower.shape == test_pred.shape
    assert upper.shape == test_pred.shape
    assert (upper >= lower).all()


def test_reset_context():
    """reset_context() should clear stored context, reverting to no-context behavior."""
    op = _make_operator()
    x = torch.randn(2, 32)

    pred_empty = op.predict(x).clone()

    op.adapt(torch.randn(5, 32), torch.randn(5, 32))
    pred_with_ctx = op.predict(x).clone()

    op.reset_context()
    pred_after_reset = op.predict(x).clone()

    assert not torch.allclose(pred_empty, pred_with_ctx, atol=1e-6)
    assert torch.allclose(pred_empty, pred_after_reset, atol=1e-6)


def test_multiple_adapt_calls_accumulate():
    """Multiple adapt() calls should accumulate context, not replace it."""
    op = _make_operator()
    x = torch.randn(2, 32)

    op.adapt(torch.randn(3, 32), torch.randn(3, 32))
    pred_first = op.predict(x).clone()

    op.adapt(torch.randn(4, 32), torch.randn(4, 32))
    pred_second = op.predict(x).clone()

    assert op._context_tokens is not None
    assert op._context_tokens.shape[0] == 7
    assert not torch.allclose(pred_first, pred_second, atol=1e-6)
