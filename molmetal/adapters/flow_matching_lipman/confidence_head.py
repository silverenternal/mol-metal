# -*- coding: utf-8 -*-
"""
Mol-Metal ConfidenceHead — mirror of OmegaFold's pLDDT-style confidence head.

Mirrors `omegafold/confidence.py:39+96+124` (HeliXon 2022, Apache-2.0):
- `get_all_confidence(lddt_per_residue)` aggregates per-residue confidence
- `_compute_confidence(logits)` produces pLDDT [0, 1] via softmax × bin centers
- `ConfidenceHead` MLP: node_repr -> hidden -> num_bins logits

Motivation: replace binary `decode_ratio in {0.0, 1.0}` with continuous pLDDT
in [0, 1]. PaCE-style continuous confidence for Mol-Metal refiner outputs.
"""

import argparse
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class ConfidenceHeadOutput:
    """Container for per-sample + aggregate confidence."""

    plddt_per_residue: torch.Tensor  # [num_res]
    pae_per_pair: Optional[torch.Tensor] = None  # [num_res, num_res]
    ptm_score: Optional[float] = None
    overall_confidence: float = 0.0

    def to_dict(self) -> dict:
        d = {
            "plddt_mean": float(self.plddt_per_residue.mean().item()),
            "plddt_min": float(self.plddt_per_residue.min().item()),
            "plddt_max": float(self.plddt_per_residue.max().item()),
            "plddt_std": float(self.plddt_per_residue.std().item())
            if self.plddt_per_residue.numel() > 1
            else 0.0,
            "overall_confidence": float(self.overall_confidence),
        }
        if self.ptm_score is not None:
            d["ptm_score"] = float(self.ptm_score)
        if self.pae_per_pair is not None:
            d["pae_mean"] = float(self.pae_per_pair.mean().item())
        return d


def _compute_confidence(logits: torch.Tensor) -> torch.Tensor:
    """Mirrors omegafold/confidence.py:96 — softmax × bin centers → pLDDT.

    Args:
        logits: shape [..., num_bins]

    Returns:
        plddt: shape [...] in [0, 1]
    """
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bin_centers = torch.arange(
        start=0.5 * bin_width,
        end=1.0,
        step=bin_width,
        device=logits.device,
        dtype=logits.dtype,
    )
    probs = torch.softmax(logits, dim=-1)
    return (probs * bin_centers).sum(dim=-1)


def get_all_confidence(
    lddt_per_residue: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    cutoff: float = 0.7,
) -> float:
    """Aggregate per-residue pLDDT into overall confidence.

    Mirrors omegafold/confidence.py:39 `get_all_confidence` but adapted for
    Mol-Metal's flattened pIC50-style confidence (no 3D coords needed).

    Args:
        lddt_per_residue: shape [num_res], each value in [0, 1]
        mask: optional [num_res] boolean mask of valid residues
        cutoff: threshold above which a residue counts as "high confidence"

    Returns:
        overall: float in [0, 1]
    """
    if mask is None:
        mask = torch.ones_like(lddt_per_residue, dtype=torch.bool)
    valid = lddt_per_residue[mask]
    if valid.numel() == 0:
        return 0.0
    # Mask-aware mean (mirrors AlphaFold LDDT-Cα weighting)
    return float((valid * (valid >= cutoff).float()).sum().item() / valid.numel())


class ConfidenceHead(nn.Module):
    """Mirrors omegafold/confidence.py:124 ConfidenceHead — pLDDT-style MLP.

    Replaces Mol-Metal's binary decode_ratio (0.0/1.0) with continuous pLDDT.
    Default config: node_dim=128 → hidden_dim=64 → num_bins=50.
    """

    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        num_bins: int = 50,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_bins = num_bins
        self.network = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_bins),
        )

    def forward(self, node_repr: torch.Tensor) -> torch.Tensor:
        """Forward pass returning pLDDT [0, 1] per residue.

        Args:
            node_repr: shape [num_res, node_dim] or [batch, num_res, node_dim]

        Returns:
            plddt: shape [num_res] or [batch, num_res] in [0, 1]
        """
        logits = self.network(node_repr)
        return _compute_confidence(logits)

    def get_all_confidence(
        self,
        node_repr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> ConfidenceHeadOutput:
        """Convenience: forward + aggregate + return ConfidenceHeadOutput."""
        plddt = self.forward(node_repr)
        overall = get_all_confidence(plddt, mask)
        return ConfidenceHeadOutput(
            plddt_per_residue=plddt,
            overall_confidence=overall,
        )


# Convenience factory
def make_confidence_head(cfg: argparse.Namespace) -> ConfidenceHead:
    """Build ConfidenceHead from argparse cfg (mirrors OmegaFold's cfg pattern)."""
    return ConfidenceHead(
        node_dim=getattr(cfg, "node_dim", 128),
        hidden_dim=getattr(cfg, "hidden_dim", 64),
        num_bins=getattr(cfg, "num_bins", 50),
    )
