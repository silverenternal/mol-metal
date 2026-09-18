"""Cross-attention fusion for D-MPNN (2D) + EGNN (3D) streams.

Alternative to the simple concat+MLP fusion in :mod:`molmetal.models.fusion`.
The cross-attention design lets the 2D D-MPNN representation query the 3D
EGNN representation so the residual stream can condition on 3D geometry at
each atom position.

Design (TODO/04 architecture doc, C2 ablation B):
    q = W_q h_dmpnn        (B, N, H)
    k = W_k h_egnn         (B, N, H)
    v = W_v h_egnn         (B, N, H)
    attn(q, k, v)          -> (B, N, H)
    out   = h_dmpnn + W_o attn     # residual connection

D-MPNN queries, EGNN keys/values. This preserves the per-atom shape (B, N, H)
so the fusion output can drop into the existing dual-head pipeline unchanged.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion with D-MPNN queries and EGNN keys/values.

    Args:
        dmpnn_dim: dimension of D-MPNN per-atom features (default 128).
        egnn_dim:  dimension of EGNN per-atom features (default 128).
        hidden:    internal attention width (default 64).
        n_heads:   number of attention heads (default 4).
        dropout:   attention dropout (default 0.0).
    """

    def __init__(
        self,
        dmpnn_dim: int = 128,
        egnn_dim: int = 128,
        hidden: int = 64,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden % n_heads != 0:
            raise ValueError(
                f"hidden ({hidden}) must be divisible by n_heads ({n_heads})"
            )

        self.dmpnn_dim = dmpnn_dim
        self.egnn_dim = egnn_dim
        self.hidden = hidden
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.dropout = dropout

        # Projections (manual so we can return attention weights if needed)
        self.q_dmpnn = nn.Linear(dmpnn_dim, hidden)
        self.k_egnn = nn.Linear(egnn_dim, hidden)
        self.v_egnn = nn.Linear(egnn_dim, hidden)

        # Output projection back to dmpnn_dim with residual
        self.out = nn.Linear(hidden, dmpnn_dim)

        # LayerNorm on the residual stream to stabilise training
        self.norm = nn.LayerNorm(dmpnn_dim)

        # Cache for last attention weights (set during forward if collect_attn=True)
        self._last_attn_weights: Optional[torch.Tensor] = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, N, hidden) -> (B, n_heads, N, head_dim)."""
        B, N, _ = x.shape
        return x.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, n_heads, N, head_dim) -> (B, N, hidden)."""
        B, H, N, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, H * D)

    def forward(
        self,
        h_dmpnn: torch.Tensor,
        h_egnn: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        collect_attn: bool = False,
    ) -> torch.Tensor:
        """Fuse D-MPNN (queries) with EGNN (keys/values).

        Args:
            h_dmpnn: (B, N, dmpnn_dim) D-MPNN per-atom features.
            h_egnn:  (B, N, egnn_dim)  EGNN per-atom features.
            mask:    (B, N) optional bool tensor. True = real atom, False = padding.
                     When provided, padded keys/values are masked out of attention.
            collect_attn: if True, stash the averaged attention weights in
                          ``self._last_attn_weights`` for inspection.

        Returns:
            (B, N, dmpnn_dim) fused per-atom features with a residual connection
            from the D-MPNN input.
        """
        if h_dmpnn.shape[:2] != h_egnn.shape[:2]:
            raise ValueError(
                f"Batch/seq dims must match: h_dmpnn={tuple(h_dmpnn.shape)} "
                f"h_egnn={tuple(h_egnn.shape)}"
            )

        q = self.q_dmpnn(h_dmpnn)  # (B, N, hidden)
        k = self.k_egnn(h_egnn)    # (B, N, hidden)
        v = self.v_egnn(h_egnn)    # (B, N, hidden)

        q_h = self._split_heads(q)  # (B, n_heads, N, head_dim)
        k_h = self._split_heads(k)
        v_h = self._split_heads(v)

        # Scaled dot-product
        # scores: (B, n_heads, N_q, N_k)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # mask: (B, N) -> (B, 1, 1, N_k)
            attn_mask = (~mask.bool()).unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attn_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        if self.dropout > 0.0 and self.training:
            attn = F.dropout(attn, p=self.dropout)

        if collect_attn:
            # Average over heads -> (B, N_q, N_k)
            self._last_attn_weights = attn.mean(dim=1)

        ctx = torch.matmul(attn, v_h)          # (B, n_heads, N, head_dim)
        ctx = self._merge_heads(ctx)           # (B, N, hidden)
        out = self.out(ctx)                    # (B, N, dmpnn_dim)

        # Residual + LayerNorm
        fused = self.norm(h_dmpnn + out)
        return fused

    def get_last_attention_weights(self) -> Optional[torch.Tensor]:
        """Return the most recently computed averaged attention weights (B, N, N).

        Returns ``None`` unless ``collect_attn=True`` was passed on the last
        ``forward`` call.
        """
        return self._last_attn_weights


__all__ = ["CrossAttentionFusion"]