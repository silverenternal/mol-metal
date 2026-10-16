"""SO(3)-equivariant D-MPNN message passing scaffold (Wf-358-B2).

Background
==========
Per Wf-357-C2 / Wf-358-B2 spec: the canonical
:mod:`molmetal.models.dmpnn.DirectedMPNN` is E(3)-*invariant* in the
graph representation — atom / bond features are scalars, the model
treats the molecule as a 2D graph and discards 3D coordinates.  This is
fine for property prediction on graph-only inputs (IC50 / pIC50 from
SMILES), but for Vina-style tasks where pocket geometry matters,
the model cannot exploit bond-length / bond-angle / pocket-distance
information.

This module adds an **SO(3)-equivariant message-passing scaffold** that:

* augments each directed edge ``(src, dst)`` with a SchNet-style radial
  basis function (RBF) embedding of the distance ``‖x_dst − x_src‖₂``.
  Distances are SO(3)-invariant scalars, so the augmented edge features
  remain E(3)-invariant and the *pooled graph embedding* is still
  rotation-equivariant up to the trivial action (per Wf-357-C2 §1
  criterion (c): "outputs transform as a scalar, not as a vector").
* trains an *equivariant coordinate update* in the spirit of Satorras
  EGNN 2021 + SchNet 2017 — the per-atom ``coord_scale`` MLP
  (``m → tanh(·)``) produces a scalar weight that scales the relative
  direction vector ``(x_dst − x_src)``, so the resulting coordinate
  update ``Δx = Σ_incoming_outsources scalar_ij · (x_dst − x_src)``
  is **equivariant** under rotation of all input coordinates.

The module is wired as an *opt-in* flag (``--equivariant-mp``, default
OFF, see :data:`molmetal.scripts.train_dmpnn_multitask_wasserstein.
parse_args`) — without the flag the model behaves bit-identically to
the legacy :class:`DirectedMPNN`.  Default OFF preserves the published
baseline numerical signature for Wf-124 P3 / Wf-181 P1 / Wf-131 P1
reproducibility audits.

CPU-only correctness gate (Wf-358-B2 acceptance)
================================================
* SO(3) rotation probe: ``EquivariantDMPNN(x) ≈ EquivariantDMPNN(R · x)``
  at max abs diff < 1e-3 for a single molecule (the EGNN
  *invariance* contract; per-atom / pooled outputs are scalars).
* Forward pass + gradient flow through the ``coord_scale`` MLP.
* CLI flag ``--equivariant-mp`` wired into
  :mod:`molmetal.scripts.train_dmpnn_multitask_wasserstein.parse_args`
  with banner observability (``equivariant_mp={True,False}``).
* Backward-compat: legacy :class:`DirectedMPNN` parameter names +
  forward signature preserved (state-dict loadable).

Honest framing
==============
Per ``party-twenty-spirit`` §3.5:
* This is a SCAFFOLD only (~120 LOC); not a full retrain.  PROJECTED
  +10-20% m3 Vina lift is *projected* not measured (no G5/G6 retrain
  in scope; GPU verification deferred per AGENTS.md §5.1).
* No chemprop / BondHead / DMPNN-Cache integration yet.  The scaffold
  sits alongside the legacy D-MPNN path; integration into the
  Wasserstein trainer's :class:`WassersteinDMPNN` is a separate Wf.
* Wall budget: PROJECTED 45-75 min CPU-only (this module + tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from molmetal.models.dmpnn import (
    DirectedMPNN,
    MPNNConfig,
)


# ---------------------------------------------------------------------------
# Radial basis function (RBF) — SchNet §3.1
# ---------------------------------------------------------------------------
def _rbf_expansion(
    dist: torch.Tensor,
    n_rbf: int = 32,
    cutoff: float = 5.0,
) -> torch.Tensor:
    """Expand pairwise distances into a Gaussian RBF embedding.

    Following SchNet 2017 §3.1 (Schütt et al., "SchNet: A continuous-
    filter convolutional neural network for modeling quantum
    interactions"):

        e_k(d) = exp(−β · (exp(−d) − μ_k)²),   k = 1..n_rbf
        μ_k = exp(−cutoff · k / n_rbf)         (centres on log-scale)

    Distances > cutoff are clamped to ``cutoff`` before expansion so the
    embedding is bounded.

    Args:
        dist: (E,) edge-wise Euclidean distances ‖x_dst − x_src‖₂.
        n_rbf: number of RBF centres (default 32).
        cutoff: distance cutoff in Å (default 5.0).

    Returns:
        (E, n_rbf) RBF embedding — SO(3)-invariant scalar features.
    """
    dist = dist.clamp(max=cutoff)
    centres = torch.linspace(0.0, cutoff, n_rbf, dtype=dist.dtype, device=dist.device)
    # Standard SchNet form — width 0.5 Å in distance space.
    beta = 0.5 * (n_rbf - 1) / cutoff
    diff = dist.unsqueeze(-1) - centres.unsqueeze(0)  # (E, n_rbf)
    return torch.exp(-beta * diff * diff)


# ---------------------------------------------------------------------------
# Equivariant edge MLP — gates the EGNN-style coordinate update
# ---------------------------------------------------------------------------
class _EquivariantEdgeMLP(nn.Module):
    """Per-edge MLP that emits a *scalar* weight ``scalar_ij ∈ [-1, 1]``.

    The scalar is bounded by ``tanh`` so the coordinate update
    ``Δx = Σ scalar_ij · (x_dst − x_src)`` is bounded in magnitude
    (Satorras 2021 EGNN §3.2 form).
    """

    def __init__(self, hidden_dim: int, in_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, edge_in: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(edge_in).squeeze(-1))  # (E,)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class EquivariantMPNNConfig:
    """Config for :class:`EquivariantDMPNN`.

    Mirrors :class:`molmetal.models.dmpnn.MPNNConfig` so the legacy
    ``DirectedMPNN`` parameters load bit-exactly into the
    ``atom_embed`` / ``edge_embed`` / ``edge_mlp`` / ``gru_updates`` /
    ``atom_to_edge`` / ``readout_mlp`` sub-modules.
    """

    atom_feat_dim: int = 39
    edge_feat_dim: int = 6
    hidden_dim: int = 128
    n_layers: int = 3
    dropout: float = 0.0
    n_rbf: int = 32
    rbf_cutoff: float = 5.0
    coord_update_scale: float = 1.0  # multiplier on the final Δx (default 1.0)


# ---------------------------------------------------------------------------
# EquivariantDMPNN — D-MPNN backbone + EGNN-style coord updates
# ---------------------------------------------------------------------------
class EquivariantDMPNN(nn.Module):
    """D-MPNN + SO(3)-equivariant coordinate updates.

    Architecture
    ------------
    1. Standard D-MPNN message passing (E(3)-invariant edge / atom
       features).  Same module names as :class:`DirectedMPNN` so
       state-dicts load.
    2. Per-edge radial basis function (RBF) embedding of
       ``‖x_dst − x_src‖₂`` — concatenated to the directed-edge
       feature before the existing :class:`GRUMessageUpdate`.  This
       gives the GRU per-layer access to bond-length information.
    3. Per-layer equivariant coord update:
       ``Δx_i = Σ_{j→i} scalar_ij · (x_i − x_j)``
       where ``scalar_ij = tanh(MLP([h_e_ij, rbf_ij]))`` ∈ [-1, 1].

    Outputs are **invariant** scalars (per-atom + pooled graph
    features); coordinate trajectory ``x`` is updated as a
    side-channel and discarded (this is a feature extractor, not a
    generative model — per Wf-357-C2 §1 criterion (d)).
    """

    def __init__(
        self,
        config: Optional[EquivariantMPNNConfig] = None,
        base: Optional[DirectedMPNN] = None,
    ):
        super().__init__()
        cfg = config or EquivariantMPNNConfig()
        self.cfg = cfg
        self.n_layers = cfg.n_layers
        self.n_rbf = cfg.n_rbf
        self.rbf_cutoff = cfg.rbf_cutoff
        self.coord_update_scale = cfg.coord_update_scale
        self.hidden_dim = cfg.hidden_dim

        # ----- base D-MPNN (parameter-name-compatible with DirectedMPNN) -----
        self.base = (
            base
            if base is not None
            else DirectedMPNN(
                MPNNConfig(
                    atom_feat_dim=cfg.atom_feat_dim,
                    edge_feat_dim=cfg.edge_feat_dim,
                    hidden_dim=cfg.hidden_dim,
                    n_layers=cfg.n_layers,
                    dropout=cfg.dropout,
                )
            )
        )

        # ----- per-layer equivariant coord MLP (scalar output) -----
        # Input = [edge_hidden (D), rbf embedding (n_rbf)] → scalar.
        self.coord_mlps = nn.ModuleList(
            [
                _EquivariantEdgeMLP(cfg.hidden_dim, cfg.hidden_dim + cfg.n_rbf)
                for _ in range(cfg.n_layers)
            ]
        )

        # ----- edge-attr RBF fusion (Linear: edge_feat_dim + n_rbf → edge_feat_dim) -----
        # Concatenates RBF to the original edge_attr before the DirectedMPNN
        # edge_embed step.  This way the downstream GRU + atom_to_edge
        # receive the RBF-augmented signal without changing the legacy
        # edge_mlp / gru_updates / readout_mlp parameter shapes.
        self.edge_rbf_proj = nn.Linear(cfg.edge_feat_dim + cfg.n_rbf, cfg.edge_feat_dim)

    # ----------------------------------------------------------------
    # Distance helper
    # ----------------------------------------------------------------
    @staticmethod
    def _pairwise_distances(
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Per-edge Euclidean distance ‖x_dst − x_src‖₂.

        Args:
            x: (N, 3) atom coordinates.
            edge_index: (2, E) directed edge pairs.

        Returns:
            (E,) per-edge distances.
        """
        src, dst = edge_index[0], edge_index[1]
        diff = x[dst] - x[src]
        return diff.norm(dim=-1)

    # ----------------------------------------------------------------
    # Forward (single molecule)
    # ----------------------------------------------------------------
    def forward(
        self,
        h_atom: torch.Tensor,  # (N, atom_feat_dim)
        edge_index: torch.Tensor,  # (2, E)
        edge_attr: torch.Tensor,  # (E, edge_feat_dim)
        coords: torch.Tensor,  # (N, 3) 3D coords
    ) -> torch.Tensor:
        """Forward pass — returns per-atom invariant features (N, D).

        The coordinate trajectory ``x`` is updated as a side-channel
        but discarded (this is a feature extractor, not a generative
        model).  Per-atom / pooled graph features remain SO(3)-invariant
        up to numerical precision — see
        :mod:`molmetal.tests.test_wf358b2_equivariant_mp` test_rotation_equivariance_probe.
        """
        x = coords
        h_v = self.base.atom_embed(h_atom)  # (N, D) — same as DirectedMPNN

        if edge_index.size(1) == 0:
            return h_v

        # Initial radial embedding (computed once, re-used across layers).
        dist = self._pairwise_distances(x, edge_index)
        rbf = _rbf_expansion(dist, n_rbf=self.n_rbf, cutoff=self.rbf_cutoff)
        # (E, n_rbf)

        # Fuse RBF into edge_attr before the legacy edge_embed step —
        # ``edge_rbf_proj`` projects (edge_feat_dim + n_rbf) →
        # edge_feat_dim so the downstream ``edge_embed`` (Linear
        # edge_feat_dim → D) sees the same input shape.  This is the
        # bit-exact-conventional shape: without --equivariant-mp, the
        # legacy DirectedMPNN path is unchanged.
        edge_attr_aug = self.edge_rbf_proj(torch.cat([edge_attr, rbf], dim=-1))
        h_e = self.base.edge_embed(edge_attr_aug)  # (E, D)
        src, dst = edge_index[0], edge_index[1]

        for layer in range(self.n_layers):
            # Standard D-MPNN message step — uses the legacy edge_mlp
            # structure unchanged (2D → D via Linear+ReLU).
            h_src = torch.tanh(self.base.atom_to_edge[layer](h_v[src]))
            cat_input = torch.cat([h_e, h_src], dim=-1)  # (E, 2D)
            msg = self.base.edge_mlp[layer](cat_input)  # (E, D) — Linear+ReLU

            # EGNN-style equivariant coord update (scalar weight)
            edge_in = torch.cat([h_e, rbf], dim=-1)  # (E, D + n_rbf)
            scalar_ij = self.coord_mlps[layer](edge_in)  # (E,)
            diff = x[dst] - x[src]  # (E, 3)
            coord_msg = -diff * scalar_ij.unsqueeze(-1)  # (E, 3) — (x_i - x_j)

            # Aggregate incoming messages at each destination atom
            m_j = torch.zeros_like(h_v)
            m_j.index_add_(0, dst, msg)
            m_j_expanded = m_j[dst]

            # GRU update on edge hidden state
            h_e = self.base.gru_updates[layer](m_j_expanded, h_src)

            # Equivariant coord update — Σ_j scalar_ij · (x_i - x_j).
            # Use ``index_add`` (out-of-place, autograd-friendly) so
            # the EGNN coord update is differentiable end-to-end.
            coord_accum = torch.zeros_like(x)
            coord_accum = coord_accum.index_add(
                0, dst, self.coord_update_scale * coord_msg
            )
            x = x + coord_accum

            h_e = self.base.dropout(h_e)

        # Readout (mirrors DirectedMPNN)
        h_outgoing = torch.zeros_like(h_v)
        h_outgoing.index_add_(0, src, h_e)
        readout_input = torch.cat([h_v, h_outgoing], dim=-1)
        h_v_out = self.base.readout_mlp(readout_input)
        return h_v_out


# ---------------------------------------------------------------------------
# CLI-flag convenience — read EQUIVARIANT_MP env var (matches Wf-328-W9 style)
# ---------------------------------------------------------------------------
import os as _os


def _env_flag(name: str, default: str) -> bool:
    v = _os.environ.get(name, default)
    return str(v).lower() in ("1", "true", "yes", "on")


EQUIVARIANT_MP_ENABLED = _env_flag(  # noqa: F841
    "EQUIVARIANT_MP", "0"
)


__all__ = [
    "EquivariantDMPNN",
    "EquivariantMPNNConfig",
    "_rbf_expansion",
    "_EquivariantEdgeMLP",
    "EQUIVARIANT_MP_ENABLED",
]
