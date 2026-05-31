"""Gradient-free few-shot transfer via in-context learning on neural operator backbones.

Provides test-time adaptation without any gradient updates: example (input, output)
pairs are encoded into context tokens, and cross-attention routes the relevant context
to new queries.  Works on top of both CodomainBackbone and FlowOperator.

References:
    - PDE-FM / The Well: Foundation Models for PDEs, AAAI 2026
    - ICON: In-Context Learning for Neural Operators
    - CoDA-NO: arXiv:2403.12553

Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ContextPairEncoder(nn.Module):
    """Encode (input, output) example pairs into context token embeddings.

    Each pair (x_i, y_i) is concatenated and projected through a small MLP
    into the backbone's embedding space, yielding one context token per pair.

    Args:
        input_dim: Dimension of each input x_i (flattened).
        output_dim: Dimension of each output y_i (flattened).
        embed_dim: Target embedding dimension (must match backbone).
        hidden_dim: MLP hidden width.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embed_dim: int = 64,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 2 * embed_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim + output_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self,
        context_inputs: Tensor,
        context_outputs: Tensor,
    ) -> Tensor:
        """Encode context pairs into token embeddings.

        Args:
            context_inputs: (n_pairs, input_dim) input examples.
            context_outputs: (n_pairs, output_dim) output examples.

        Returns:
            (n_pairs, embed_dim) context token embeddings.
        """
        if context_inputs.dim() == 1:
            context_inputs = context_inputs.unsqueeze(0)
        if context_outputs.dim() == 1:
            context_outputs = context_outputs.unsqueeze(0)
        pairs = torch.cat([context_inputs, context_outputs], dim=-1)
        return self.net(pairs)


class InContextAttention(nn.Module):
    """Cross-attention between a query and context token embeddings.

    The query embedding (from the backbone) attends over context key/value
    tokens produced by the ContextPairEncoder.  Multi-head attention lets the
    model attend to different aspects of the context simultaneously.

    Args:
        embed_dim: Shared embedding dimension for query, keys, values.
        n_heads: Number of attention heads.
    """

    def __init__(self, embed_dim: int = 64, n_heads: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        assert embed_dim % n_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})"
        )

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: Tensor,
        context_tokens: Tensor,
    ) -> Tensor:
        """Cross-attend query over context tokens.

        Args:
            query: (batch, embed_dim) or (batch, 1, embed_dim) query embedding.
            context_tokens: (n_pairs, embed_dim) or (batch, n_pairs, embed_dim)
                context key/value tokens.

        Returns:
            (batch, embed_dim) or (batch, 1, embed_dim) context-informed query.
        """
        if query.dim() == 2:
            query = query.unsqueeze(1)  # (B, 1, D)
        if context_tokens.dim() == 2:
            context_tokens = context_tokens.unsqueeze(0).expand(query.shape[0], -1, -1)

        B, Q, D = query.shape
        _B, N, _D = context_tokens.shape

        q = self.q_proj(query).view(B, Q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context_tokens).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context_tokens).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, Q, D)
        out = self.out_proj(out)

        result = self.norm(query + out)
        if result.shape[1] == 1:
            result = result.squeeze(1)
        return result


class InContextOperator(nn.Module):
    """Gradient-free adaptation via in-context learning.

    Stores context (input, output) pairs at test time without any gradient
    updates.  Predictions on new inputs use cross-attention over stored context
    tokens to blend backbone features with relevant example information.

    Works with any backbone that has an ``encode`` method (CodomainBackbone,
    FlowOperator's VAE encoder, or plain nn.Module).

    Args:
        backbone: Neural operator backbone with ``encode`` method.
        context_encoder: ContextPairEncoder instance for encoding pairs.
        output_dim: Dimension of the prediction output (flattened).
        n_context_heads: Number of heads for in-context cross-attention.
        device: Torch device.
    """

    def __init__(
        self,
        backbone: nn.Module,
        context_encoder: ContextPairEncoder,
        output_dim: int,
        n_context_heads: int = 4,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.backbone = backbone.to(self.device)
        self.context_encoder = context_encoder.to(self.device)

        self.embed_dim = context_encoder.embed_dim

        self.context_attention = InContextAttention(
            embed_dim=self.embed_dim,
            n_heads=n_context_heads,
        ).to(self.device)

        self.output_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, output_dim),
        ).to(self.device)

        self._context_tokens: Tensor | None = None
        self.output_dim = output_dim

    @torch.no_grad()
    def adapt(
        self,
        context_inputs: Tensor,
        context_outputs: Tensor,
    ) -> None:
        """Store context pairs for future predictions (no gradient updates).

        Args:
            context_inputs: (n_pairs, input_dim) input examples.
            context_outputs: (n_pairs, output_dim) output examples.
        """
        context_inputs = context_inputs.to(self.device)
        context_outputs = context_outputs.to(self.device)
        new_tokens = self.context_encoder(context_inputs, context_outputs)

        if self._context_tokens is None:
            self._context_tokens = new_tokens
        else:
            self._context_tokens = torch.cat([self._context_tokens, new_tokens], dim=0)

    def reset_context(self) -> None:
        """Clear all stored context tokens."""
        self._context_tokens = None

    @torch.no_grad()
    def predict(self, new_input: Tensor) -> Tensor:
        """Predict using in-context attention (no fine-tuning).

        The backbone encodes the new input into an embedding, then
        cross-attention retrieves relevant context, and an output head
        projects to the prediction space.

        Args:
            new_input: (batch, ...) input tensor compatible with backbone.encode().

        Returns:
            (batch, output_dim) predictions.
        """
        new_input = new_input.to(self.device)
        if new_input.dim() == 1:
            new_input = new_input.unsqueeze(0)

        query_emb = self._encode_query(new_input)

        if self._context_tokens is not None:
            query_emb = self.context_attention(query_emb, self._context_tokens)

        return self.output_head(query_emb)

    def _encode_query(self, new_input: Tensor) -> Tensor:
        """Encode new input via the backbone, pooling to (batch, embed_dim)."""
        if hasattr(self.backbone, "encode"):
            raw = self.backbone.encode(new_input)
        else:
            raw = self.backbone(new_input)

        if raw.dim() == 3:
            raw = raw.mean(dim=1)
        elif raw.dim() == 1:
            raw = raw.unsqueeze(0)

        if raw.shape[-1] != self.embed_dim:
            proj = nn.Linear(raw.shape[-1], self.embed_dim, device=self.device)
            raw = proj(raw)

        return raw

    @torch.no_grad()
    def predict_with_confidence(
        self,
        new_input: Tensor,
        n_forward_passes: int = 8,
    ) -> tuple[Tensor, Tensor]:
        """Predict with uncertainty via dropout-based MC forward passes.

        Enables dropout (if present) and runs multiple forward passes to
        estimate prediction uncertainty as the standard deviation.

        Args:
            new_input: (batch, ...) input tensor.
            n_forward_passes: Number of stochastic forward passes.

        Returns:
            (mean_prediction, std_prediction) each (batch, output_dim).
        """
        self.train()
        predictions = []
        for _ in range(n_forward_passes):
            pred = self.predict(new_input)
            predictions.append(pred)
        self.eval()

        stacked = torch.stack(predictions, dim=0)
        mean_pred = stacked.mean(dim=0)
        std_pred = stacked.std(dim=0)
        return mean_pred, std_pred

    def forward(self, new_input: Tensor) -> Tensor:
        """Alias for predict."""
        return self.predict(new_input)


class InContextBenchmark:
    """Compare in-context operator vs adapter few-shot transfer.

    Benchmarks four methods across multiple few-shot sizes:
      - In-context operator (this module, zero gradient steps)
      - FewShotFinetuner adapter (10 gradient steps)
      - FewShotFinetuner adapter (30 gradient steps)
      - From-scratch baseline
    """

    @staticmethod
    def run(
        n_seeds: int = 3,
        few_shot_sizes: list[int] | None = None,
        n_grid: int = 32,
        embed_dim: int = 32,
        n_forward_passes: int = 4,
    ) -> dict[str, dict[int, tuple[float, float, float]]]:
        """Run the four-way comparison.

        Args:
            n_seeds: Number of random seeds.
            few_shot_sizes: List of few-shot sample counts.
            n_grid: Spatial grid dimension.
            embed_dim: Embedding dimension for all models.
            n_forward_passes: Forward passes for in-context confidence.

        Returns:
            Dict mapping method name -> {few_shot_size: (mean_error, std_error, adaptation_time)}.
        """
        from .pretraining import FewShotFinetuner, PDENet

        if few_shot_sizes is None:
            few_shot_sizes = [5, 10, 20, 50]

        input_dim = n_grid
        output_dim = n_grid

        results: dict[str, dict[int, list[tuple[float, float]]]] = {
            "in_context": {s: [] for s in few_shot_sizes},
            "adapter_10": {s: [] for s in few_shot_sizes},
            "adapter_30": {s: [] for s in few_shot_sizes},
            "from_scratch": {s: [] for s in few_shot_sizes},
        }
        timing: dict[str, dict[int, list[float]]] = {
            "in_context": {s: [] for s in few_shot_sizes},
            "adapter_10": {s: [] for s in few_shot_sizes},
            "adapter_30": {s: [] for s in few_shot_sizes},
            "from_scratch": {s: [] for s in few_shot_sizes},
        }

        for seed in range(n_seeds):
            torch.manual_seed(seed * 100 + 7)

            W = torch.randn(input_dim, output_dim) * 0.5
            full_inputs = torch.randn(200, input_dim)
            full_targets = full_inputs @ W + torch.randn(200, output_dim) * 0.01

            test_inputs = full_inputs[:50]
            test_targets = full_targets[:50]
            train_pool_inputs = full_inputs[50:]
            train_pool_targets = full_targets[50:]

            for size in few_shot_sizes:
                perm = torch.randperm(train_pool_inputs.shape[0])[:size]
                fs_inputs = train_pool_inputs[perm]
                fs_targets = train_pool_targets[perm]

                # --- In-context operator ---
                backbone_ic = PDENet(
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
                ic_op = InContextOperator(
                    backbone=backbone_ic,
                    context_encoder=ctx_enc,
                    output_dim=output_dim,
                    n_context_heads=2,
                )

                t0 = time.perf_counter()
                ic_op.adapt(fs_inputs, fs_targets)
                ic_pred = ic_op.predict(test_inputs)
                adapt_time_ic = time.perf_counter() - t0

                ic_err = F.mse_loss(ic_pred, test_targets).item()
                results["in_context"][size].append((ic_err, adapt_time_ic))
                timing["in_context"][size].append(adapt_time_ic)

                # --- Adapter 10 gradient steps ---
                backbone_a10 = PDENet(
                    input_dim=input_dim,
                    hidden_dim=embed_dim,
                    output_dim=output_dim,
                    n_layers=2,
                )
                ft10 = FewShotFinetuner(backbone_a10, output_dim=output_dim)

                t0 = time.perf_counter()
                ft10.finetune(fs_inputs, fs_targets, n_epochs=10, lr=1e-3)
                a10_pred = ft10.predict(test_inputs)
                adapt_time_a10 = time.perf_counter() - t0

                a10_err = F.mse_loss(a10_pred, test_targets).item()
                results["adapter_10"][size].append((a10_err, adapt_time_a10))

                # --- Adapter 30 gradient steps ---
                backbone_a30 = PDENet(
                    input_dim=input_dim,
                    hidden_dim=embed_dim,
                    output_dim=output_dim,
                    n_layers=2,
                )
                ft30 = FewShotFinetuner(backbone_a30, output_dim=output_dim)

                t0 = time.perf_counter()
                ft30.finetune(fs_inputs, fs_targets, n_epochs=30, lr=1e-3)
                a30_pred = ft30.predict(test_inputs)
                adapt_time_a30 = time.perf_counter() - t0

                a30_err = F.mse_loss(a30_pred, test_targets).item()
                results["adapter_30"][size].append((a30_err, adapt_time_a30))

                # --- From-scratch baseline (10 epochs, no pretraining) ---
                scratch_net = PDENet(
                    input_dim=input_dim,
                    hidden_dim=embed_dim,
                    output_dim=output_dim,
                    n_layers=2,
                )
                scratch_ft = FewShotFinetuner(scratch_net, output_dim=output_dim)

                t0 = time.perf_counter()
                scratch_ft.finetune(fs_inputs, fs_targets, n_epochs=10, lr=1e-3)
                scratch_pred = scratch_ft.predict(test_inputs)
                adapt_time_scratch = time.perf_counter() - t0

                scratch_err = F.mse_loss(scratch_pred, test_targets).item()
                results["from_scratch"][size].append((scratch_err, adapt_time_scratch))

        aggregated: dict[str, dict[int, tuple[float, float, float]]] = {}
        for method in results:
            aggregated[method] = {}
            for size in few_shot_sizes:
                errs = [v[0] for v in results[method][size]]
                times = [v[1] for v in results[method][size]]
                mean_err = sum(errs) / len(errs)
                std_err = (sum((e - mean_err) ** 2 for e in errs) / len(errs)) ** 0.5
                mean_time = sum(times) / len(times)
                aggregated[method][size] = (mean_err, std_err, mean_time)

        return aggregated
