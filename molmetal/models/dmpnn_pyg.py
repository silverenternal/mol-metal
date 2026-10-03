"""PyG-style batched MessagePassing for the D-MPNN backbone.

Background
==========
Per ``molmetal/reports/wf326a_gpu_profile/profile.json`` the per-molecule
Python loop in ``molmetal.models.dmpnn.DirectedMPNN.forward`` is the
**CONFIRMED MAJOR** bottleneck (~50% of forward time).  Concretely::

    for b in range(B):
        h_mol = h_atom[b]
        ei_mol = edge_index[b]
        ...
        h_per_atom = self.forward_per_atom(h_mol, ei_local, ea_local)
        pooled = h_per_atom.sum(dim=0)
        pooled_list.append(pooled)

``B = 256`` launches ``B`` Python-level invocations of
``forward_per_atom``; each one spins up its own scatter-sum / GRU /
readout, paying the per-launch overhead ``~256`` times in the
``encode_pool``/``forward`` hot path.

PyG-style coalesced MessagePassing collapses the ``B`` separate small
graphs into a **single disjoint union graph** (``N_total`` atoms,
``E_total`` directed edges) and runs the message-passing loop **once**
over the union.  Per-molecule boundaries are preserved by the
``batch_idx`` vector (which atom belongs to which molecule) — atoms in
different molecules never share edges, so the message passing is
mathematically identical to running the loop independently per
molecule; only the loop overhead disappears.

API
===
- :class:`MolGraphBatch` — flat-tensor batch representation (mirrors
  ``BatchMolGraph`` from Chemprop v2 but built from a padded batch).
- :class:`DirectedMPNNPyG` — same parameter names as ``DirectedMPNN``
  for checkpoint back-compat; ``forward_batched`` operates on a
  :class:`MolGraphBatch`.
- :func:`coalesce_padded_batch` — converts the padded
  ``(B, N_max, A)`` / ``(B, 2, E_max)`` / ``(B, E_max, B_dim)`` layout
  used by :class:`molmetal.models.dmpnn.DirectedMPNN` into the flat
  ``MolGraphBatch``.
- :func:`forward_dmpnn_pyg` — convenience wrapper that takes the
  padded layout and returns ``(B, hidden_dim)`` pooled embeddings,
  reusing an existing :class:`DirectedMPNN`'s parameters via
  ``load_state_dict_into_pyg``.

CPU-correctness guarantee (Wf-328-W7 acceptance gate):
- ``forward_batched`` agrees with the per-mol loop forward within
  ``atol=1e-3`` over 100 random graphs (test
  ``test_pyg_matches_loop_forward``).
- Backward gradients agree within the same tolerance (test
  ``test_pyg_backward_matches_loop``).

Speedup (PROJECTED, not MEASURED here):
- For a batch of 256 small mols (mean V=20, mean E=40) on RTX 5090 the
  per-launch overhead is the dominant cost; collapsing the 256
  Python loops into 1 batched loop projects a **5-10x speedup** on
  ``forward()`` wall time per Wf-326G §1.1.  Wf-328-W7 actual GPU
  bench is deferred to the next workflow per AGENTS.md §5.1 (no
  per-launch GPU OK in this CPU-only workflow).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from molmetal.models._scatter import scatter_sum_direct
from molmetal.models.dmpnn import (
    DirectedMPNN,
    GRUMessageUpdate,
    MPNNConfig,
    _maybe_fused_dropout_only,
)


# ---------------------------------------------------------------------------
# Flat batch representation
# ---------------------------------------------------------------------------
@dataclass
class MolGraphBatch:
    """Disjoint-union graph batch — mirrors PyG ``Batch`` semantics.

    All tensors live in the same device; per-molecule atom offsets are
    stored in ``atom_offsets`` (cumulative atom counts; ``atom_offsets[b]``
    is the start atom of molecule ``b``).  ``batch_idx`` is the per-atom
    molecule id (shape ``(N_total,)``) — used to pool atom features back
    to ``(B, hidden_dim)`` at readout time.

    Attributes
    ----------
    atom_feats : (N_total, atom_feat_dim)
        Concatenated per-molecule atom features (no padding).
    edge_index : (2, E_total) int64
        Coalesced directed edges; ``src/dst`` in ``[0, N_total)``.
    edge_attr : (E_total, edge_feat_dim)
        Concatenated per-molecule edge features.
    batch_idx : (N_total,) int64
        For each atom, the molecule id it belongs to.
    n_atoms_per_mol : (B,) int64
        Number of atoms per molecule (for diagnostics / length checks).
    n_edges_per_mol : (B,) int64
        Number of directed edges per molecule.
    """

    atom_feats: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    batch_idx: torch.Tensor
    n_atoms_per_mol: torch.Tensor
    n_edges_per_mol: torch.Tensor

    @property
    def n_atoms(self) -> int:
        return int(self.atom_feats.size(0))

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.size(1))

    @property
    def batch_size(self) -> int:
        return int(self.n_atoms_per_mol.size(0))


def coalesce_padded_batch(
    h_atom: torch.Tensor,         # (B, N_max, A)
    edge_index: torch.Tensor,     # (B, 2, E_max)
    edge_attr: torch.Tensor,      # (B, E_max, B_dim)
    batch_idx_padded: torch.Tensor,  # (B, N_max)
) -> MolGraphBatch:
    """Convert a padded-batch layout into a flat :class:`MolGraphBatch`.

    The input layout matches :meth:`molmetal.models.dmpnn.DirectedMPNN.forward`
    (padded ``(B, N_max, ...)`` / ``(B, E_max, ...)`` with ``-1`` edge
    sentinels).  Padding is detected by:
    - atom slot ``n`` of molecule ``b`` is real iff ``batch_idx_padded[b, n] == b``
    - edge slot ``e`` of molecule ``b`` is real iff ``edge_index[b, :, e]`` both
      in ``[0, N_max)`` AND ``batch_idx_padded[b, edge_index[b, :, e]] == b``
      (the latter filters sentinels = ``-1``)
    - on per-mol subgraphs we rebase the edge indices by
      ``+ atom_offset[b]`` so the coalesced graph is a true disjoint union.

    The result has ``N_total = sum_b n_atoms[b]`` atoms and
    ``E_total = sum_b n_edges[b]`` directed edges — no padding waste.
    """
    B, N_max, _ = h_atom.shape
    _, _, E_max = edge_index.shape
    device = h_atom.device

    # Per-mol atom count: real atoms are those whose batch_idx matches b.
    is_real_atom = (batch_idx_padded == torch.arange(B, device=device).unsqueeze(1))
    n_atoms_per_mol = is_real_atom.sum(dim=1).to(torch.int64)

    # Cumulative atom offsets: atom_offsets[b] is the global index of the
    # first atom of molecule b in the flat tensor.
    atom_offsets = torch.zeros(B, dtype=torch.int64, device=device)
    atom_offsets[1:] = torch.cumsum(n_atoms_per_mol[:-1], dim=0)

    # Build the flat atom tensor (no copy when each molecule has the
    # same number of atoms — we still loop over molecules here because
    # the input is padded and a single boolean mask gather would be
    # O(B * N_max * A) which can be larger than the loop on small mols).
    atom_feats_list = []
    for b in range(B):
        n_b = int(n_atoms_per_mol[b].item())
        if n_b > 0:
            atom_feats_list.append(h_atom[b, :n_b])
    atom_feats = torch.cat(atom_feats_list, dim=0) if atom_feats_list else torch.zeros(
        0, h_atom.size(-1), device=device, dtype=h_atom.dtype
    )

    # Per-mol edge count: real edges are those whose src AND dst are
    # both real atom slots of molecule b.  Edges with src/dst == -1
    # (padding sentinel) are automatically filtered because they fail
    # the < n_atoms_per_mol[b] check below.
    src_padded = edge_index[:, 0]  # (B, E_max)
    dst_padded = edge_index[:, 1]  # (B, E_max)
    # n_atoms_per_mol broadcast to (B, 1) for vectorised comparison.
    natoms_col = n_atoms_per_mol.unsqueeze(1)  # (B, 1)
    is_real_edge = (src_padded >= 0) & (dst_padded >= 0) & (src_padded < natoms_col) & (dst_padded < natoms_col)
    n_edges_per_mol = is_real_edge.sum(dim=1).to(torch.int64)

    # Build the flat edge_index + edge_attr tensors.
    edge_src_list = []
    edge_dst_list = []
    edge_attr_list = []
    for b in range(B):
        e_b = int(n_edges_per_mol[b].item())
        if e_b > 0:
            # Filter this row using the boolean mask.
            mask_b = is_real_edge[b]
            src_local = src_padded[b][mask_b] + int(atom_offsets[b].item())
            dst_local = dst_padded[b][mask_b] + int(atom_offsets[b].item())
            edge_src_list.append(src_local)
            edge_dst_list.append(dst_local)
            edge_attr_list.append(edge_attr[b][mask_b])

    if edge_src_list:
        edge_src_flat = torch.cat(edge_src_list, dim=0)
        edge_dst_flat = torch.cat(edge_dst_list, dim=0)
        edge_attr_flat = torch.cat(edge_attr_list, dim=0)
    else:
        edge_src_flat = torch.zeros(0, dtype=torch.int64, device=device)
        edge_dst_flat = torch.zeros(0, dtype=torch.int64, device=device)
        edge_attr_flat = torch.zeros(0, edge_attr.size(-1), device=device, dtype=edge_attr.dtype)

    edge_index_flat = torch.stack([edge_src_flat, edge_dst_flat], dim=0)

    # Flat batch_idx: atom i belongs to molecule batch_idx[i].
    batch_idx_flat = torch.repeat_interleave(
        torch.arange(B, device=device, dtype=torch.int64), n_atoms_per_mol
    )

    return MolGraphBatch(
        atom_feats=atom_feats,
        edge_index=edge_index_flat,
        edge_attr=edge_attr_flat,
        batch_idx=batch_idx_flat,
        n_atoms_per_mol=n_atoms_per_mol,
        n_edges_per_mol=n_edges_per_mol,
    )


# ---------------------------------------------------------------------------
# PyG-style batched DirectedMPNN
# ---------------------------------------------------------------------------
class DirectedMPNNPyG(nn.Module):
    """Directed-MPNN operating on a flat :class:`MolGraphBatch`.

    Mirrors :class:`molmetal.models.dmpnn.DirectedMPNN` parameter names
    so an existing checkpoint can be loaded verbatim via
    ``load_state_dict(strict=True)``.  The math is identical; the only
    difference is that ``forward_batched`` runs the message-passing loop
    once over the union graph instead of once per molecule.

    The forward returns a ``(B, hidden_dim)`` tensor via per-molecule
    sum pooling over the per-atom output — equivalent to the loop
    implementation's stacked ``pooled_list``.
    """

    def __init__(self, config: Optional[MPNNConfig] = None):
        super().__init__()
        cfg = config or MPNNConfig()
        # Re-use the exact same submodules as DirectedMPNN so the
        # state_dict keys line up.  This is the lightest possible
        # refactor: the per-layer Linears / GRUs are identical; only
        # the surrounding ``for b in range(B)`` loop is replaced.
        self.atom_feat_dim = cfg.atom_feat_dim
        self.edge_feat_dim = cfg.edge_feat_dim
        self.hidden_dim = cfg.hidden_dim
        self.n_layers = cfg.n_layers
        self.dropout = nn.Dropout(cfg.dropout)

        self.atom_embed = nn.Sequential(
            nn.Linear(self.atom_feat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.edge_embed = nn.Sequential(
            nn.Linear(self.edge_feat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.edge_mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                    nn.ReLU(),
                )
                for _ in range(self.n_layers)
            ]
        )
        self.gru_updates = nn.ModuleList(
            [
                GRUMessageUpdate(self.hidden_dim, self.hidden_dim, self.hidden_dim)
                for _ in range(self.n_layers)
            ]
        )
        self.atom_to_edge = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.n_layers)]
        )
        self.readout_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    # -- helpers shared with DirectedMPNN.forward_per_atom --------------
    def _scatter_sum(
        self, msg: torch.Tensor, index: torch.Tensor, dim_size: int
    ) -> torch.Tensor:
        # Use ``scatter_sum_direct`` (autograd-friendly, no dummy src
        # allocation) — strictly cheaper than ``scatter_sum_legacy`` on
        # the flat path.
        return scatter_sum_direct(msg, index, dim_size=dim_size)

    def forward_batched(self, batch: MolGraphBatch) -> torch.Tensor:
        """Run message passing once over the union graph.

        Returns:
            (B, hidden_dim) pooled molecular embeddings.
        """
        N_total = batch.n_atoms
        E_total = batch.n_edges
        device = batch.atom_feats.device
        if N_total == 0:
            # Degenerate batch (all empty molecules); return zero
            # embeddings to match the loop's default-logit fallback.
            return torch.zeros(batch.batch_size, self.hidden_dim, device=device)

        h_v = self.atom_embed(batch.atom_feats)  # (N_total, D)

        if E_total == 0:
            # No edges anywhere — pool embedded atoms per molecule.
            pooled = torch.zeros(batch.batch_size, self.hidden_dim, device=device)
            pooled.index_add_(0, batch.batch_idx, h_v)
            return pooled

        h_e = self.edge_embed(batch.edge_attr)  # (E_total, D)
        src = batch.edge_index[0]
        dst = batch.edge_index[1]

        for layer in range(self.n_layers):
            h_src = torch.tanh(self.atom_to_edge[layer](h_v[src]))  # (E_total, D)
            msg = self.edge_mlp[layer](torch.cat([h_e, h_src], dim=-1))  # (E_total, D)

            # Incoming aggregation: scatter-sum messages to destination atoms.
            m_j = self._scatter_sum(msg, dst, dim_size=N_total)  # (N_total, D)

            # Expand back to per-edge (the loop implementation does
            # ``m_j_expanded = m_j[dst]`` — equivalent here).
            m_j_expanded = m_j[dst]  # (E_total, D)

            h_e = self.gru_updates[layer](m_j_expanded, h_src)  # (E_total, D)
            h_e = _maybe_fused_dropout_only(
                h_e, p=self.dropout.p, training=self.training
            )

        # Outgoing aggregation: scatter-sum edges whose source is atom v.
        h_outgoing = self._scatter_sum(h_e, src, dim_size=N_total)  # (N_total, D)

        # Per-atom readout: concat [h_v, h_outgoing], pass through MLP.
        h_v_out = self.readout_mlp(torch.cat([h_v, h_outgoing], dim=-1))  # (N_total, D)

        # Per-molecule sum pool: scatter-add per-atom features into (B, D).
        pooled = torch.zeros(batch.batch_size, self.hidden_dim, device=device)
        pooled.index_add_(0, batch.batch_idx, h_v_out)
        return pooled

    def forward_from_padded(
        self,
        h_atom: torch.Tensor,         # (B, N_max, A)
        edge_index: torch.Tensor,     # (B, 2, E_max)
        edge_attr: torch.Tensor,      # (B, E_max, B_dim)
        batch_idx_padded: torch.Tensor,  # (B, N_max)
    ) -> torch.Tensor:
        """End-to-end: convert padded batch → flat → batched forward.

        This is the drop-in replacement for
        :meth:`molmetal.models.dmpnn.DirectedMPNN.forward` that callers
        should reach for when ``backend='pyg'``.
        """
        batch = coalesce_padded_batch(h_atom, edge_index, edge_attr, batch_idx_padded)
        return self.forward_batched(batch)


# ---------------------------------------------------------------------------
# Back-compat: load a DirectedMPNN state-dict into a DirectedMPNNPyG
# ---------------------------------------------------------------------------
def load_state_dict_into_pyg(
    pyg_model: DirectedMPNNPyG,
    legacy_model: DirectedMPNN,
    strict: bool = True,
) -> "tuple[list[str], list[str]]":
    """Copy parameters from a legacy :class:`DirectedMPNN` into a
    :class:`DirectedMPNNPyG` (or any module with the same submodule
    names).  Returns the standard ``(missing, unexpected)`` lists.
    """
    return pyg_model.load_state_dict(legacy_model.state_dict(), strict=strict)


# ---------------------------------------------------------------------------
# Convenience: instantiate a PyG model pre-loaded from a legacy one
# ---------------------------------------------------------------------------
def from_legacy(legacy_model: DirectedMPNN) -> DirectedMPNNPyG:
    """Construct a :class:`DirectedMPNNPyG` with identical weights to
    ``legacy_model`` — used in tests + by the ``--dmpnn-backend pyg``
    CLI path in :mod:`molmetal.scripts.train_dmpnn_multitask`.
    """
    cfg = MPNNConfig(
        atom_feat_dim=legacy_model.atom_feat_dim,
        edge_feat_dim=legacy_model.edge_feat_dim,
        hidden_dim=legacy_model.hidden_dim,
        n_layers=legacy_model.n_layers,
        dropout=legacy_model.dropout.p,
    )
    pyg = DirectedMPNNPyG(cfg)
    load_state_dict_into_pyg(pyg, legacy_model, strict=True)
    return pyg


__all__ = [
    "MolGraphBatch",
    "coalesce_padded_batch",
    "DirectedMPNNPyG",
    "load_state_dict_into_pyg",
    "from_legacy",
]
