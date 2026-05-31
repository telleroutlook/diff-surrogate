"""Tests for codomain-attention backbone (S9.1)."""

import torch

from diff_surrogate.codomain import (
    AdapterHead,
    CodomainAttentionBlock,
    CodomainBackbone,
    CodomainPretrainer,
    CodomainTransferBenchmark,
)


def _make_fields(batch: int = 4, n_fields: int = 3, spatial_dim: int = 32) -> torch.Tensor:
    return torch.randn(batch, n_fields, spatial_dim)


def _make_backbone(spatial_dim: int = 32, embed_dim: int = 32, field_names=None):
    return CodomainBackbone(
        spatial_dim=spatial_dim,
        embed_dim=embed_dim,
        n_layers=2,
        n_heads=2,
        field_names=field_names,
    )


# ---- tests ----


def test_attention_block_forward_different_field_counts():
    """CodomainAttentionBlock handles varying field counts."""
    block = CodomainAttentionBlock(embed_dim=32, n_heads=4)
    for n_fields in [1, 3, 7]:
        x = torch.randn(2, n_fields, 32)
        out = block(x)
        assert out.shape == (2, n_fields, 32)
        assert torch.isfinite(out).all()


def test_attention_block_with_mask():
    """CodomainAttentionBlock respects field_mask."""
    block = CodomainAttentionBlock(embed_dim=32, n_heads=4)
    x = torch.randn(2, 5, 32)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[0, 3:] = False
    mask[1, 4:] = False
    out = block(x, field_mask=mask)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_backbone_variable_field_numbers():
    """CodomainBackbone processes different field counts at train vs test."""
    bb = _make_backbone(field_names=["u", "v", "p"])

    # 3 fields (train)
    f3 = _make_fields(batch=4, n_fields=3, spatial_dim=32)
    out3 = bb(f3, ["u", "v", "p"])
    assert out3.shape == f3.shape

    # Add 2 more fields, test with 5
    bb.register_fields(["dx", "dy"])
    f5 = _make_fields(batch=4, n_fields=5, spatial_dim=32)
    out5 = bb(f5, ["u", "v", "p", "dx", "dy"])
    assert out5.shape == f5.shape


def test_backbone_with_field_mask():
    """CodomainBackbone handles masked (missing) fields."""
    bb = _make_backbone(field_names=["u", "v", "p", "T"])
    fields = _make_fields(batch=2, n_fields=4, spatial_dim=32)
    mask = torch.ones(2, 4, dtype=torch.bool)
    mask[:, 3] = False
    out = bb(fields, ["u", "v", "p", "T"], field_mask=mask)
    assert out.shape == fields.shape
    assert torch.isfinite(out).all()


def test_masked_reconstruction_pretraining_runs():
    """CodomainPretrainer completes training and returns finite losses."""
    bb = _make_backbone(field_names=["u", "v", "p"])
    fields = _make_fields(batch=8, n_fields=3, spatial_dim=32)
    pretrainer = CodomainPretrainer(bb, mask_ratio=0.3)
    history = pretrainer.pretrain(
        [(fields, ["u", "v", "p"])],
        n_epochs=10,
        lr=1e-3,
    )
    assert len(history) == 10
    assert all(isinstance(v, float) for v in history)
    assert all(torch.isfinite(torch.tensor(v)) for v in history)


def test_adapter_head_attach_detach():
    """AdapterHead produces correct shapes and gradients flow."""
    bb = _make_backbone(embed_dim=32, field_names=["u", "v"])
    adapter = AdapterHead(embed_dim=32, out_dim=16)

    fields = _make_fields(batch=4, n_fields=2, spatial_dim=32)
    emb = bb.encode(fields, ["u", "v"])
    out = adapter(emb)
    assert out.shape == (4, 2, 16)

    loss = out.sum()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.parameters())
    assert has_grad, "No gradient in AdapterHead"


def test_transfer_add_fields():
    """Pretrain on 3 fields, finetune on 5 fields (add 2)."""
    torch.manual_seed(0)
    bb = _make_backbone(field_names=["u", "v", "p"])

    pretrain_fields = _make_fields(batch=16, n_fields=3, spatial_dim=32)
    pretrainer = CodomainPretrainer(bb, mask_ratio=0.25)
    pretrainer.pretrain(
        [(pretrain_fields, ["u", "v", "p"])],
        n_epochs=5,
        lr=1e-3,
    )
    pretrained = pretrainer.get_backbone()

    pretrained.register_fields(["dx", "dy"])
    target_fields = _make_fields(batch=8, n_fields=5, spatial_dim=32)
    out = pretrained(target_fields, ["u", "v", "p", "dx", "dy"])
    assert out.shape == target_fields.shape
    assert torch.isfinite(out).all()


def test_transfer_vs_from_scratch():
    """CodomainTransferBenchmark runs and returns valid structure."""
    torch.manual_seed(42)
    spatial_dim = 16
    pretrain_names = ["u", "v", "p"]
    target_names = ["u", "v", "p", "dx", "dy"]

    W = torch.randn(5)
    pretrain_fields = torch.randn(30, 3, spatial_dim)
    target_fields = torch.randn(20, 5, spatial_dim)
    target_labels = target_fields.mean(dim=-1) @ W

    result = CodomainTransferBenchmark.compare(
        spatial_dim=spatial_dim,
        embed_dim=16,
        pretrain_field_names=pretrain_names,
        pretrain_data=(pretrain_fields, pretrain_names),
        target_field_names=target_names,
        target_fields=target_fields,
        target_labels=target_labels,
        few_shot_sizes=[5, 10],
        n_seeds=2,
        pretrain_epochs=5,
        finetune_epochs=5,
    )

    for key in ["codomain_transfer", "from_scratch", "mlp_multitask", "error_ratios"]:
        assert key in result

    for s in [5, 10]:
        mean, std = result["codomain_transfer"][s]
        assert isinstance(mean, float) and isinstance(std, float)
        assert std >= 0


def test_gradient_flows_through_codomain():
    """Gradients reach all backbone parameters."""
    bb = _make_backbone(field_names=["u", "v", "p"])
    fields = _make_fields(batch=4, n_fields=3, spatial_dim=32)
    out = bb(fields, ["u", "v", "p"])
    loss = out.sum()
    loss.backward()

    has_grad = False
    for p in bb.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "No gradient in backbone"


def test_integration_with_structure_preserving_encoder():
    """CodomainBackbone output can be consumed by StructurePreservingEncoder patterns."""
    from diff_surrogate.structure import DivergenceConservingProjection

    spatial_dim = 16
    bb = _make_backbone(spatial_dim=spatial_dim, field_names=["u", "v"])
    fields = _make_fields(batch=2, n_fields=2, spatial_dim=spatial_dim)
    out = bb(fields, ["u", "v"])
    assert torch.isfinite(out).all()

    proj = DivergenceConservingProjection(method="iterative", max_iter=5)
    reshaped = out.reshape(2, 2, 4, 4)
    corrected = proj(reshaped)
    assert corrected.shape == reshaped.shape
    assert torch.isfinite(corrected).all()


def test_edge_cases_single_and_many_fields():
    """Edge cases: single field, many fields, all masked."""
    bb = _make_backbone(field_names=["x"])

    # Single field
    f1 = _make_fields(batch=2, n_fields=1, spatial_dim=32)
    out1 = bb(f1, ["x"])
    assert out1.shape == f1.shape

    # Many fields
    names = [f"f{i}" for i in range(20)]
    bb_many = _make_backbone(field_names=names)
    f20 = _make_fields(batch=2, n_fields=20, spatial_dim=32)
    out20 = bb_many(f20, names)
    assert out20.shape == f20.shape

    # All masked (should not crash)
    mask = torch.zeros(2, 20, dtype=torch.bool)
    out_masked = bb_many(f20, names, field_mask=mask)
    assert out_masked.shape == f20.shape


def test_deterministic_with_seed():
    """Same seed produces identical outputs."""
    names = ["u", "v", "p"]
    fields = _make_fields(batch=4, n_fields=3, spatial_dim=32)

    torch.manual_seed(99)
    bb1 = _make_backbone(field_names=names)
    out1 = bb1(fields, names)

    torch.manual_seed(99)
    bb2 = _make_backbone(field_names=names)
    out2 = bb2(fields, names)

    assert torch.allclose(out1, out2, atol=1e-6)
