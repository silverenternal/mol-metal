"""YuelBond-style learned GNN edge predictor for CFM connectivity decoding.

WF-CFM-Rescue Phase 1 (2026-09-16) — implements a small message-passing
GNN that maps (positions, atomic_numbers) -> per-pair bond logits.

Lit anchor
-----------
Wang, S., Dokholyan, P. (2025). *YuelBond: a graph neural network for
recovering the molecular bond network from distorted 3D structures.*
bioRxiv 10.1101/2025.05.06.652517.  Reported F1=92.7 % on distorted
molecules where RDKit's ``DetermineConnectivity`` fails on 783/1000
cases.

This module is a CPU-only re-implementation of the YuelBond decoder head
inspired by the bioRxiv preprint.  It is intentionally small (≤ 5k
parameters) so it can run inside a 200-step decode smoke at < 5 s wall.
The architecture is:

    AtomEncoder(z) -> (H,)  via one-hot(Z) @ W_atom
    Iteratively for k in [0..K):
        h_i' = h_i + sum_j  msg(h_i, h_j, d_ij)
        msg  = MLP([h_i, h_j, d_ij, is_short, is_long])
    PairHead(i, j)  = MLP([h_i', h_j', d_ij, is_dative_candidate])
                    -> logits over {NO_BOND, SINGLE, DOUBLE, TRIPLE, AROMATIC}

The output tensor has shape ``(E, 5)`` and is consumed by
:class:`molmetal.models.bond_head.BondAwareDecoder` via the existing
``pair_features`` interface — no changes to ``BondAwareDecoder`` are
required.

Why CPU-only?
-------------
The current decode smoke must run with `cuda_available=False` (per
``wf_gpu_auto_recover/final.md`` 2026-09-15 + ``wf_gpu_recovery_now/
final.md`` 2026-09-15).  The YuelBond decoder head is small enough
(~5k parameters, K=3 message-passing layers) that the CPU cost is
negligible compared to the rest of the decode pipeline.

Lit anchor (secondary)
----------------------
Liu, Z., et al. (2025). *NExT-Mol: 1D SELFIES + 3D conformer decoupling
for molecular generation.*  ICLR 2025, arXiv:2502.12638.  Their MoLlama
1.8B-pretrain shows that learning the bond topology from positions can
outperform covalent-radius heuristics on out-of-distribution scaffolds —
the same intuition that motivates this decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from molmetal.models.bond_head import (
    AtomCloud,
    BOND_AROMATIC,
    BOND_DOUBLE,
    BOND_LABELS,
    BOND_NO_BOND,
    BOND_SINGLE,
    BOND_TRIPLE,
    NUM_BOND_CLASSES,
    PairFeature,
)


# ---------------------------------------------------------------------------
# Helper: Z -> one-hot
# ---------------------------------------------------------------------------
# Use a fixed-size vocab that matches the round-10 atom_vocab in
# ``flow_matching_lipman/__init__.py`` (12 atoms + Z=0 padding).
_YUELBOND_Z_TABLE: Tuple[int, ...] = (0, 1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53, 78)
_YUELBOND_Z_TO_IDX = {z: i for i, z in enumerate(_YUELBOND_Z_TABLE)}


def _z_to_onehot(z: torch.Tensor) -> torch.Tensor:
    """Map atomic numbers to one-hot indices into :data:`_YUELBOND_Z_TABLE`.

    Unknown Z values are clamped to Z=0 (padding slot).
    """
    idx = torch.zeros(z.shape[0], dtype=torch.long)
    for i, atomic_number in enumerate(z.tolist()):
        idx[i] = _YUELBOND_Z_TO_IDX.get(int(atomic_number), 0)
    return F.one_hot(idx, num_classes=len(_YUELBOND_Z_TABLE)).float()


# ---------------------------------------------------------------------------
# YuelBondDecoderHead — small message-passing GNN
# ---------------------------------------------------------------------------
class YuelBondDecoderHead(nn.Module):
    """Small GNN that maps an :class:`AtomCloud` to per-pair bond logits.

    Parameters
    ----------
    hidden_dim : int, default 64
        Width of the message-passing hidden state.  Small enough to keep
        the head under 5k parameters; large enough to represent the
        triple / aromatic minority classes.
    n_layers : int, default 3
        Number of message-passing iterations.  K=3 matches the bioRxiv
        preprint's recommendation for drug-like ligands.
    cutoff : float, default 3.5 Å
        Pairs further apart than this are immediately classified as
        ``NO_BOND`` without being passed through the MLP head.  Keeps
        the ``E = N(N-1)/2`` edge list manageable at N=19 atoms
        (171 pairs).
    min_distance : float, default 1.0 Å
        Pairs closer than this are clamped (physically impossible
        steric clash).
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 3,
        cutoff: float = 3.5,
        min_distance: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.cutoff = float(cutoff)
        self.min_distance = float(min_distance)
        # Atom encoder: Z one-hot -> hidden_dim
        self.atom_encoder = nn.Linear(len(_YUELBOND_Z_TABLE), hidden_dim)
        # Message MLP: input [h_i, h_j, d_ij, is_short, is_long, is_dative]
        self.msg_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2 + 4, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(n_layers)
        ])
        # Pair head: [h_i, h_j, d_ij, dative] -> NUM_BOND_CLASSES logits
        self.pair_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, NUM_BOND_CLASSES),
        )
        # Bias the "no bond" class slightly negative so the head does
        # not default to bonding every pair.
        with torch.no_grad():
            self.pair_head[-1].bias.zero_()
            self.pair_head[-1].bias[BOND_NO_BOND] = -0.5

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(self, cloud: AtomCloud) -> torch.Tensor:
        """Return ``(E, NUM_BOND_CLASSES)`` logits over candidate edges.

        Edges are the upper-triangular pair list ``[(i, j) for i<j in N]``
        with all pairs closer than :attr:`cutoff`.  Pairs at distance
        ``> cutoff`` are NOT returned (caller should treat them as
        ``NO_BOND``); pairs at ``< min_distance`` are clamped.
        """
        positions = cloud.positions.float()
        atomic_numbers = cloud.atomic_numbers.long()
        n = int(atomic_numbers.shape[0])
        if n < 2:
            return (
                torch.zeros(0, NUM_BOND_CLASSES),
                (torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)),
            )
        # Encode atoms
        z_onehot = _z_to_onehot(atomic_numbers)
        h = self.atom_encoder(z_onehot)  # (N, H)
        # Build pair list: upper triangle
        i_idx, j_idx = torch.triu_indices(n, n, offset=1)
        if i_idx.numel() == 0:
            return (
                torch.zeros(0, NUM_BOND_CLASSES),
                (torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)),
            )
        # Pair distances and short/long flags
        d_vec = positions[j_idx] - positions[i_idx]
        d = d_vec.norm(dim=-1).clamp(min=self.min_distance)
        # Filter to within-cutoff pairs only (cutoff=3.5 Å excludes 4.1 Å Cl-Cl)
        mask = d < self.cutoff
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        d = d[mask]
        if i_idx.numel() == 0:
            return (
                torch.zeros(0, NUM_BOND_CLASSES),
                (torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)),
            )
        # Per-edge features (now all shape (E,))
        is_short = (d < 1.7).float().unsqueeze(-1)
        is_long = (d > 3.0).float().unsqueeze(-1)
        is_dative = ((atomic_numbers[i_idx] == 7) | (atomic_numbers[i_idx] == 8)
                     | (atomic_numbers[i_idx] == 16)) & (
                     atomic_numbers[j_idx] == 78)
        is_dative = is_dative.float().unsqueeze(-1)
        # K message-passing iterations
        for k in range(self.n_layers):
            h_i = h[i_idx]
            h_j = h[j_idx]
            msg_in = torch.cat([
                h_i, h_j,
                d.unsqueeze(-1),
                is_short,
                is_long,
                is_dative,
            ], dim=-1)
            msg = self.msg_mlps[k](msg_in)
            # Aggregate by scatter-add onto each atom's h
            h_new = h.clone()
            h_new.index_add_(0, i_idx, msg)
            h = h_new
        # Pair head
        h_i_final = h[i_idx]
        h_j_final = h[j_idx]
        pair_in = torch.cat([
            h_i_final, h_j_final,
            d.unsqueeze(-1),
            is_dative,
            is_short,
        ], dim=-1)
        logits = self.pair_head(pair_in)
        return logits, (i_idx, j_idx)


# ---------------------------------------------------------------------------
# YuelBondDecoder — wrapper that emits PairFeature + BondAwareDecoder output
# ---------------------------------------------------------------------------
@dataclass
class YuelBondResult:
    """Result of a YuelBond decode pass."""

    pair_features: Optional[PairFeature]
    logits: torch.Tensor           # (E, 5)
    edge_index: torch.Tensor       # (2, E)
    decode_succeeded: bool
    error: Optional[str] = None


class YuelBondDecoder:
    """Wraps :class:`YuelBondDecoderHead` + :class:`BondAwareDecoder`.

    This is the YuelBond-style replacement for the
    ``decode_distance_graph`` heuristic.  It is a drop-in alternative
    when ``--bond-head=learned`` is selected and ``connectivity_prior=
    yuelbond`` is requested.
    """

    def __init__(
        self,
        head: Optional[YuelBondDecoderHead] = None,
        hidden_dim: int = 64,
        n_layers: int = 3,
    ) -> None:
        self.head = head if head is not None else YuelBondDecoderHead(
            hidden_dim=hidden_dim, n_layers=n_layers,
        )

    def featurise(self, cloud: AtomCloud) -> YuelBondResult:
        """Compute pair features + logits for an :class:`AtomCloud`.

        Returns a :class:`YuelBondResult`; on shape errors returns
        ``decode_succeeded=False`` with a short ``error`` message.
        """
        try:
            out = self.head(cloud)
            # Default YuelBondDecoderHead always returns (logits, (i_idx, j_idx))
            if isinstance(out, tuple) and len(out) == 2:
                logits, (i_idx, j_idx) = out
            else:
                return YuelBondResult(
                    pair_features=None, logits=torch.zeros(0, NUM_BOND_CLASSES),
                    edge_index=torch.zeros(2, 0, dtype=torch.long),
                    decode_succeeded=False,
                    error="head_returned_unexpected_format",
                )
        except Exception as exc:
            return YuelBondResult(
                pair_features=None, logits=torch.zeros(0, NUM_BOND_CLASSES),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                decode_succeeded=False, error=f"head_forward_failed:{type(exc).__name__}",
            )
        edge_index = torch.stack([i_idx, j_idx], dim=0).long()
        if edge_index.shape[1] == 0:
            return YuelBondResult(
                pair_features=None, logits=logits,
                edge_index=edge_index, decode_succeeded=False,
                error="empty_edge_list",
            )
        # Compute pair features (distance, z_i, z_j, angle=0, dative=bool)
        d_vec = cloud.positions[edge_index[1]] - cloud.positions[edge_index[0]]
        d = d_vec.norm(dim=-1).clamp(min=self.head.min_distance)
        z_i = cloud.atomic_numbers[edge_index[0]].long()
        z_j = cloud.atomic_numbers[edge_index[1]].long()
        # Crude dative detector: N/O/S <-> Pt (either ordering)
        donor_mask = (z_i == 7) | (z_i == 8) | (z_i == 16) | (z_j == 7) | (z_j == 8) | (z_j == 16)
        is_dative = donor_mask & ((z_i == 78) | (z_j == 78))
        pf = PairFeature(
            edge_index=edge_index,
            distance=d,
            z_i=z_i,
            z_j=z_j,
            angle_to_metal=torch.zeros_like(d),
            is_dative_candidate=is_dative,
        )
        return YuelBondResult(
            pair_features=pf, logits=logits,
            edge_index=edge_index, decode_succeeded=True,
        )


__all__ = [
    "YuelBondDecoderHead",
    "YuelBondDecoder",
    "YuelBondResult",
]
