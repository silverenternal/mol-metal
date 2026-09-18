"""Fusion module for D-MPNN (2D) + EGNN (3D) streams.

Simple concat + MLP fusion as specified in the design doc.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FusionMLP(nn.Module):
    """Simple concatenation fusion: MLP([h_2d || h_3d]).

    Args:
        h_dim:  per-stream feature dimension (both must match, default 128)
        output_dim: output fused dimension (default 128)
    """

    def __init__(
        self,
        h_dim: int = 128,
        output_dim: int = 128,
    ):
        super().__init__()
        self.h_dim = h_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, output_dim),
        )

    def forward(self, h_2d: torch.Tensor, h_3d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_2d: (B, D) or (N, D) D-MPNN features
            h_3d: (B, D) or (N, D) EGNN features

        Returns:
            (B, output_dim) or (N, output_dim) fused features
        """
        if h_2d.shape != h_3d.shape:
            raise ValueError(
                f"Shape mismatch: h_2d={h_2d.shape}, h_3d={h_3d.shape}"
            )
        return self.net(torch.cat([h_2d, h_3d], dim=-1))
