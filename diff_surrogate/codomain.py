"""Codomain-attention neural operator backbone for multiphysics transfer learning.

References:
    - CoDA-NO, NeurIPS 2024, arXiv:2403.12553
    - Towards Universal Neural Operators, arXiv:2511.10829, 2025
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CodomainAttentionBlock(nn.Module):
    """Multi-head attention over the field (codomain) dimension.

    Args:
        embed_dim: Per-field embedding dimension.
        n_heads: Number of attention heads.
    """

    def __init__(self, embed_dim: int = 64, n_heads: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        assert embed_dim % n_heads == 0

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: Tensor,
        field_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: (batch, n_fields, embed_dim) field embeddings.
            field_mask: (batch, n_fields) bool, True for valid fields.

        Returns:
            (batch, n_fields, embed_dim) updated field embeddings.
        """
        B, N, D = x.shape
        residual = x

        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, N, N)

        if field_mask is not None:
            mask = field_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)

        if field_mask is not None:
            attn = attn.masked_fill(torch.isnan(attn), 0.0)

        out = torch.matmul(attn, v)  # (B, H, N, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)

        return self.norm(residual + out)


class CodomainBackbone(nn.Module):
    """Stack of CodomainAttentionBlock layers with a field-type embedding registry.

    Args:
        spatial_dim: Flattened spatial dimension per field.
        embed_dim: Internal embedding dimension.
        n_layers: Number of codomain attention blocks.
        n_heads: Attention heads per block.
        field_names: Initial set of field names to register.
    """

    def __init__(
        self,
        spatial_dim: int = 64,
        embed_dim: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        field_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.spatial_dim = spatial_dim
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        self.field_in_proj = nn.Linear(spatial_dim, embed_dim)
        self.field_out_proj = nn.Linear(embed_dim, spatial_dim)

        self.blocks = nn.ModuleList(
            [CodomainAttentionBlock(embed_dim, n_heads) for _ in range(n_layers)]
        )

        self._field_registry: dict[str, int] = {}
        self._field_embeddings: nn.Embedding | None = None
        self._n_registered = 0

        if field_names:
            for name in field_names:
                self.register_field(name)

    def register_field(self, name: str) -> int:
        """Register a new field name and return its index."""
        if name in self._field_registry:
            return self._field_registry[name]
        idx = self._n_registered
        self._field_registry[name] = idx
        self._n_registered += 1

        old_emb = self._field_embeddings
        new_emb = nn.Embedding(self._n_registered, self.embed_dim)
        nn.init.normal_(new_emb.weight, std=0.02)
        if old_emb is not None:
            with torch.no_grad():
                new_emb.weight[: old_emb.num_embeddings] = old_emb.weight
        self._field_embeddings = new_emb
        return idx

    def register_fields(self, names: list[str]) -> list[int]:
        """Register multiple field names."""
        return [self.register_field(n) for n in names]

    def _get_field_indices(self, names: list[str]) -> Tensor:
        """Convert field names to index tensor."""
        indices = []
        for n in names:
            if n not in self._field_registry:
                raise KeyError(f"Field '{n}' not registered. Known: {list(self._field_registry)}")
            indices.append(self._field_registry[n])
        return torch.tensor(indices, dtype=torch.long)

    def _get_field_embeddings(self, indices: Tensor) -> Tensor:
        """(n_fields,) indices -> (1, n_fields, embed_dim) embeddings."""
        if self._field_embeddings is None:
            raise RuntimeError("No fields registered")
        return self._field_embeddings(indices).unsqueeze(0)

    def forward(
        self,
        fields: Tensor,
        field_names: list[str],
        field_mask: Tensor | None = None,
    ) -> Tensor:
        """Process a batch of multi-field data.

        Args:
            fields: (batch, n_fields, spatial_dim) input fields.
            field_names: Names for each field dimension.
            field_mask: (batch, n_fields) bool, True for valid.

        Returns:
            (batch, n_fields, spatial_dim) reconstructed fields.
        """
        indices = self._get_field_indices(field_names).to(fields.device)
        type_emb = self._get_field_embeddings(indices).to(fields.device)

        x = self.field_in_proj(fields) + type_emb

        for block in self.blocks:
            x = block(x, field_mask=field_mask)

        return self.field_out_proj(x)

    def encode(
        self,
        fields: Tensor,
        field_names: list[str],
        field_mask: Tensor | None = None,
    ) -> Tensor:
        """Return embeddings before the output projection (for transfer)."""
        indices = self._get_field_indices(field_names).to(fields.device)
        type_emb = self._get_field_embeddings(indices).to(fields.device)

        x = self.field_in_proj(fields) + type_emb
        for block in self.blocks:
            x = block(x, field_mask=field_mask)
        return x


class AdapterHead(nn.Module):
    """Lightweight task-specific head for downstream physics.

    Args:
        embed_dim: Must match backbone embed_dim.
        out_dim: Spatial output dimension per field.
        hidden_dim: Adapter hidden width.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        out_dim: int = 64,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.embed_dim = embed_dim
        self.out_dim = out_dim

    def forward(self, x: Tensor) -> Tensor:
        """(batch, n_fields, embed_dim) -> (batch, n_fields, out_dim)."""
        return self.net(x)


class CodomainPretrainer:
    """Pretraining loop with masked field reconstruction.

    Args:
        backbone: CodomainBackbone instance.
        mask_ratio: Fraction of fields to mask during pretraining.
        device: Torch device.
    """

    def __init__(
        self,
        backbone: CodomainBackbone,
        mask_ratio: float = 0.25,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.backbone = backbone.to(self.device)
        self.mask_ratio = mask_ratio

    def pretrain(
        self,
        batches: list[tuple[Tensor, list[str]]],
        n_epochs: int = 100,
        lr: float = 1e-3,
    ) -> list[float]:
        """Run masked field reconstruction pretraining.

        Args:
            batches: List of (fields_tensor, field_names) tuples.
                fields_tensor shape: (N, n_fields, spatial_dim).
            n_epochs: Training epochs.
            lr: Learning rate.

        Returns:
            Per-epoch loss history.
        """
        optimizer = torch.optim.Adam(self.backbone.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        history: list[float] = []

        for _ in range(n_epochs):
            total_loss = 0.0
            n_steps = 0
            for fields, field_names in batches:
                fields = fields.to(self.device)
                B, N, _S = fields.shape

                n_mask = max(1, int(N * self.mask_ratio))
                mask = torch.ones(B, N, dtype=torch.bool, device=self.device)
                for b in range(B):
                    mask_idx = torch.randperm(N)[:n_mask]
                    mask[b, mask_idx] = False

                masked_fields = fields * mask.unsqueeze(-1).float()

                optimizer.zero_grad()
                recon = self.backbone(masked_fields, field_names, field_mask=mask)
                loss = loss_fn(recon, fields)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_steps += 1

            history.append(total_loss / max(1, n_steps))

        return history

    def get_backbone(self) -> CodomainBackbone:
        return self.backbone


class CodomainTransferBenchmark:
    """Compare from-scratch, MLP multi-task, and codomain transfer."""

    @staticmethod
    def compare(
        spatial_dim: int,
        embed_dim: int,
        pretrain_field_names: list[str],
        pretrain_data: tuple[Tensor, list[str]],
        target_field_names: list[str],
        target_fields: Tensor,
        target_labels: Tensor,
        few_shot_sizes: list[int] | None = None,
        n_seeds: int = 3,
        pretrain_epochs: int = 50,
        finetune_epochs: int = 30,
        lr: float = 1e-3,
    ) -> dict:
        """Run three-way transfer comparison.

        Args:
            spatial_dim: Spatial dimension per field.
            embed_dim: Backbone embedding dimension.
            pretrain_field_names: Field names used during pretraining.
            pretrain_data: (fields, field_names) for pretraining.
            target_field_names: Field names for target task (may be superset).
            target_fields: (N, n_fields, spatial_dim) target inputs.
            target_labels: (N,) or (N, output_dim) targets.
            few_shot_sizes: Few-shot sample counts.
            n_seeds: Random seeds.
            pretrain_epochs: Epochs for pretraining phase.
            finetune_epochs: Epochs for finetuning.
            lr: Learning rate.

        Returns:
            Dict with 'codomain_transfer', 'from_scratch', 'mlp_multitask'
            results and 'error_ratios'.
        """
        if few_shot_sizes is None:
            few_shot_sizes = [10, 20, 50]

        all_field_names = list(set(pretrain_field_names) | set(target_field_names))

        if target_labels.ndim == 1:
            target_labels = target_labels.unsqueeze(1)
        out_dim = target_labels.shape[-1]

        codomain_results: dict[int, list[float]] = {s: [] for s in few_shot_sizes}
        scratch_results: dict[int, list[float]] = {s: [] for s in few_shot_sizes}
        mlp_results: dict[int, list[float]] = {s: [] for s in few_shot_sizes}

        pretrain_fields, pretrain_fn = pretrain_data

        for seed in range(n_seeds):
            torch.manual_seed(seed * 1000 + 42)

            # --- Codomain transfer ---
            bb = CodomainBackbone(
                spatial_dim=spatial_dim,
                embed_dim=embed_dim,
                n_layers=2,
                n_heads=2,
                field_names=all_field_names,
            )
            pt = CodomainPretrainer(bb, mask_ratio=0.25, device="cpu")
            pt.pretrain(
                [(pretrain_fields, pretrain_fn)],
                n_epochs=pretrain_epochs,
                lr=lr,
            )
            pretrained_bb = pt.get_backbone()

            # --- From-scratch backbone ---
            scratch_bb = CodomainBackbone(
                spatial_dim=spatial_dim,
                embed_dim=embed_dim,
                n_layers=2,
                n_heads=2,
                field_names=all_field_names,
            )

            # --- MLP baseline ---
            mlp_flat_dim = len(target_field_names) * spatial_dim
            mlp_net = nn.Sequential(
                nn.Linear(mlp_flat_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, out_dim),
            )

            for size in few_shot_sizes:
                perm = torch.randperm(target_fields.shape[0])[:size]
                fs_fields = target_fields[perm]
                fs_labels = target_labels[perm]

                # Codomain transfer finetune
                ft_bb = copy.deepcopy(pretrained_bb)
                adapter = AdapterHead(embed_dim, out_dim)
                opt_cd = torch.optim.Adam(
                    list(ft_bb.parameters()) + list(adapter.parameters()), lr=lr
                )
                for _ in range(finetune_epochs):
                    opt_cd.zero_grad()
                    emb = ft_bb.encode(fs_fields, target_field_names)
                    pooled = emb.mean(dim=1)
                    pred = adapter(pooled.unsqueeze(1)).squeeze(1)
                    loss = F.mse_loss(pred, fs_labels)
                    loss.backward()
                    opt_cd.step()
                with torch.no_grad():
                    emb = ft_bb.encode(target_fields, target_field_names)
                    pooled = emb.mean(dim=1)
                    pred = adapter(pooled.unsqueeze(1)).squeeze(1)
                    cd_loss = F.mse_loss(pred, target_labels).item()
                codomain_results[size].append(cd_loss)

                # Scratch finetune
                sc_bb = copy.deepcopy(scratch_bb)
                sc_adapter = AdapterHead(embed_dim, out_dim)
                opt_sc = torch.optim.Adam(
                    list(sc_bb.parameters()) + list(sc_adapter.parameters()), lr=lr
                )
                for _ in range(finetune_epochs):
                    opt_sc.zero_grad()
                    emb = sc_bb.encode(fs_fields, target_field_names)
                    pooled = emb.mean(dim=1)
                    pred = sc_adapter(pooled.unsqueeze(1)).squeeze(1)
                    loss = F.mse_loss(pred, fs_labels)
                    loss.backward()
                    opt_sc.step()
                with torch.no_grad():
                    emb = sc_bb.encode(target_fields, target_field_names)
                    pooled = emb.mean(dim=1)
                    pred = sc_adapter(pooled.unsqueeze(1)).squeeze(1)
                    sc_loss = F.mse_loss(pred, target_labels).item()
                scratch_results[size].append(sc_loss)

                # MLP baseline
                mlp_copy = copy.deepcopy(mlp_net)
                opt_mlp = torch.optim.Adam(mlp_copy.parameters(), lr=lr)
                for _ in range(finetune_epochs):
                    opt_mlp.zero_grad()
                    flat = fs_fields.reshape(size, -1)
                    pred = mlp_copy(flat)
                    loss = F.mse_loss(pred, fs_labels)
                    loss.backward()
                    opt_mlp.step()
                with torch.no_grad():
                    flat = target_fields.reshape(target_fields.shape[0], -1)
                    pred = mlp_copy(flat)
                    mlp_loss = F.mse_loss(pred, target_labels).item()
                mlp_results[size].append(mlp_loss)

        def _aggregate(d: dict[int, list[float]]) -> dict[int, tuple[float, float]]:
            out = {}
            for s, vals in d.items():
                m = sum(vals) / len(vals)
                std = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
                out[s] = (m, std)
            return out

        cd_agg = _aggregate(codomain_results)
        sc_agg = _aggregate(scratch_results)
        mlp_agg = _aggregate(mlp_results)

        error_ratios = {}
        for s in few_shot_sizes:
            sc_m = sc_agg[s][0] if sc_agg[s][0] > 0 else 1e-12
            error_ratios[s] = cd_agg[s][0] / sc_m

        return {
            "codomain_transfer": cd_agg,
            "from_scratch": sc_agg,
            "mlp_multitask": mlp_agg,
            "error_ratios": error_ratios,
        }
