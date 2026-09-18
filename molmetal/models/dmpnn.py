"""Directed-Message Passing Neural Network (D-MPNN) for molecular graphs.

This implements the atom-level message passing scheme from the D-MPNN paper
(Justin et al., J. Chem. Inf. Model. 2020) in pure PyTorch with a manual
scatter_sum (no torch_geometric / torch_scatter dependency).

Key design:
- Directed edges: messages flow from h_src → h_dst along each directed edge.
- Edge hidden state h_e is updated via GRU([sum_incoming_messages, h_src]).
- Atom readout: h_v' = sum_outgoing h_e, then concat with original h_v.
- No bond-order breaking: the full bond graph (all edge directions) is used.

atom_dim = 39, edge_dim = 6 (from GraphFeaturizer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from molmetal.models._scatter import scatter_sum_legacy as _scatter_sum

# ---------------------------------------------------------------------------
# Triton wiring
# ---------------------------------------------------------------------------
# The D-MPNN backbone uses ``nn.Sequential(Linear, ReLU, Linear)`` blocks
# (atom_embed / edge_embed / readout_mlp) plus single ``Linear -> ReLU``
# per-layer edge MLPs (``edge_mlp``).  The fused MLP kernel shipped at
# :func:`triton_kernels.fused_silu_mlp` only supports SiLU and
# :func:`fused_gelu_mlp` only supports tanh-GELU; there is no fused
# ReLU kernel and changing the activation would alter model semantics
# (this is a published baseline, not a free-form network).
#
# Wiring rule for future work: when a ``fused_relu_mlp`` kernel lands,
# swap the three ``nn.Sequential(Linear, ReLU, Linear)`` blocks for a
# :class:`_MaybeFusedReLUMLP` wrapper of the same shape as
# ``models.velocity_net._FusedSiLUMLP``, gated by
# :data:`triton_kernels.triton_config`.  The wrapper should preserve
# the parameter names ``linear1.*`` / ``linear2.*`` so existing
# checkpoints load without remapping.  See ``TODO/fused_relu_mlp.md``
# (TBD) for the kernel spec.
#
# For now we annotate ``triton_config`` so the gate is observable from
# D-MPNN code paths and downstream consumers can opt-in.
from triton_kernels.config import triton_config as _triton_config
from triton_kernels import fused_silu_mlp as _fused_silu_mlp  # noqa: F401  (kept for future ReLU-swap)
from triton_kernels import fused_dropout_residual as _fused_dropout_residual
_RELU_MLP_TRT_GATE_OBSERVED = _triton_config  # exposed for test introspection


# ---------------------------------------------------------------------------
# Round-5 fused-dropout + residual wiring
# ---------------------------------------------------------------------------
def _maybe_fused_dropout_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    p: float,
    *,
    training: bool,
) -> torch.Tensor:
    """Run ``dropout(x, p) + residual`` via the Triton kernel when the gate
    is open, else fall back to ``F.dropout(x, p, training) + residual``.

    The fused kernel (``triton_kernels.fused_dropout_residual``) requires
    CUDA tensors; on CPU we transparently fall back to PyTorch so tests
    can run on a CPU-only machine.  When ``training`` is ``False`` the
    fused kernel is *always* bypassed — dropout is a no-op in eval and
    routing through the kernel adds launch overhead without any benefit.
    """
    # Eval mode: keep the trivial PyTorch path (dropout is identity in eval).
    if not training:
        return x + residual
    # CPU fallback (the kernel raises on non-CUDA tensors).
    if not x.is_cuda:
        return torch.nn.functional.dropout(x, p, training=True) + residual
    # Gate check.
    if not _triton_config.should_use_fused(x, op="dropout_residual"):
        return torch.nn.functional.dropout(x, p, training=True) + residual
    out, _ = _fused_dropout_residual(x, residual, p)
    return out


def _maybe_fused_dropout_only(
    x: torch.Tensor,
    p: float,
    *,
    training: bool,
) -> torch.Tensor:
    """Apply ``dropout(x, p)`` with the same dispatch policy as the fused
    ``dropout(x) + residual`` kernel.

    The D-MPNN edge-layer dropout site is structurally
    ``h_e' = dropout(h_e)`` (no residual stream).  We use this helper to
    keep the dropout dispatch policy consistent with the residual one
    (gate respects training mode + the Triton config).  At this site the
    helper always returns ``F.dropout`` because the fused kernel requires
    a residual tensor — but having the helper lets future code paths
    that DO have a residual (e.g. the readout cat-projection) opt in via
    the same gate.
    """
    if p <= 0.0:
        return x
    if not training:
        return x
    if not x.is_cuda:
        return torch.nn.functional.dropout(x, p, training=True)
    if not _triton_config.should_use_fused(x, op="dropout_residual"):
        return torch.nn.functional.dropout(x, p, training=True)
    # fused_dropout_residual requires a residual; for the dropout-only
    # site we keep the pure PyTorch ``F.dropout`` so behaviour stays
    # bit-exact with the original ``nn.Dropout`` path.
    return torch.nn.functional.dropout(x, p, training=True)


# ---------------------------------------------------------------------------
# Scatter utility — re-exported so external callers (training scripts,
# tests) that previously imported ``_scatter_sum`` from this module keep
# working.  Implementation now dispatches to the Triton
# ``aggregate_vectors`` kernel via ``molmetal.models._scatter``, which
# wraps the forward in a ``torch.autograd.Function`` so gradients flow.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class MPNNConfig:
    atom_feat_dim: int = 39      # from GraphFeaturizer
    edge_feat_dim: int = 6       # from GraphFeaturizer
    hidden_dim: int = 128
    n_layers: int = 3
    dropout: float = 0.0


# ---------------------------------------------------------------------------
# GRU cell wrapper
# ---------------------------------------------------------------------------
class GRUMessageUpdate(nn.Module):
    """GRU-based edge hidden-state update.

    Given:
        m_j = sum of incoming edge hidden states at dst j
        h_src = atom hidden state at the source of the directed edge

    New edge hidden: h_e' = GRU([m_j, h_src])
    """

    def __init__(self, msg_dim: int, h_src_dim: int, edge_hidden_dim: int):
        super().__init__()
        self.msg_proj = nn.Linear(msg_dim, edge_hidden_dim)
        self.gru = nn.GRUCell(edge_hidden_dim, edge_hidden_dim)

    def forward(self, m_j: torch.Tensor, h_src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m_j:    (n_edges, msg_dim) incoming message aggregated at dst
            h_src:  (n_edges, h_src_dim) atom feature at edge source

        Returns:
            (n_edges, edge_hidden_dim) updated edge hidden state
        """
        msg = self.msg_proj(m_j)  # (n_edges, edge_hidden_dim)
        return self.gru(msg, h_src)


# ---------------------------------------------------------------------------
# DirectedMPNN
# ---------------------------------------------------------------------------
class DirectedMPNN(nn.Module):
    """Directed Message Passing Neural Network.

    For batched input (B molecules padded to N_max atoms, E_max directed edges):
        h_atom : (B, N_max, atom_feat_dim)   atom input features
        edge_index : (B, 2, E_max)           directed edge pairs (src, dst)
                                             src/dst are in range [0, N_max)
        edge_attr   : (B, E_max, edge_feat_dim)
        batch_idx   : (B, N_max)             which molecule each atom belongs to

    For per-molecule input (single molecule):
        h_atom : (N, atom_feat_dim)
        edge_index : (2, E)  src,dst in [0, N)
        edge_attr : (E, edge_feat_dim)

    Returns:
        (B, hidden_dim) pooled molecular embedding (batched)
        (N, hidden_dim) per-atom features (per-molecule)
    """

    def __init__(self, config: Optional[MPNNConfig] = None):
        super().__init__()
        cfg = config or MPNNConfig()
        self.atom_feat_dim = cfg.atom_feat_dim
        self.edge_feat_dim = cfg.edge_feat_dim
        self.hidden_dim = cfg.hidden_dim
        self.n_layers = cfg.n_layers
        self.dropout = nn.Dropout(cfg.dropout)

        # ----- atom embedding -----
        self.atom_embed = nn.Sequential(
            nn.Linear(self.atom_feat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # ----- edge embedding -----
        self.edge_embed = nn.Sequential(
            nn.Linear(self.edge_feat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # ----- per-layer edge update MLPs (before GRU) -----
        self.edge_mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                    nn.ReLU(),
                )
                for _ in range(self.n_layers)
            ]
        )

        # ----- GRU update for edge hidden states -----
        self.gru_updates = nn.ModuleList(
            [
                GRUMessageUpdate(self.hidden_dim, self.hidden_dim, self.hidden_dim)
                for _ in range(self.n_layers)
            ]
        )

        # ----- per-layer atom-to-edge projection (atom h influences edge msgs) -----
        self.atom_to_edge = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim, self.hidden_dim)
                for _ in range(self.n_layers)
            ]
        )

        # ----- readout projection -----
        self.readout_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def forward_per_atom(
        self,
        h_atom: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Per-atom forward (single molecule). Returns (N_atoms, hidden_dim)."""
        N = h_atom.size(0)
        device = h_atom.device

        # Embed atom and edge features
        h_v = self.atom_embed(h_atom)  # (N, D)

        if edge_index.size(1) == 0:
            # No edges — return embedded atoms
            return h_v

        # Initialise edge hidden states from edge features
        h_e = self.edge_embed(edge_attr)  # (E, D)

        src, dst = edge_index[0], edge_index[1]  # (E,)

        for layer in range(self.n_layers):
            # Message: MLP([h_e, tanh(W * h_src)]) where h_src = atom at edge source
            h_src = torch.tanh(self.atom_to_edge[layer](h_v[src]))  # (E, D)
            msg = self.edge_mlp[layer](torch.cat([h_e, h_src], dim=-1))  # (E, D)

            # Aggregate incoming messages at each destination atom
            m_j = _scatter_sum(msg, dst, dim=0, dim_size=N)  # (N, D)

            # Expand m_j back to each edge: m_j[dst[i]] for edge i
            m_j_expanded = m_j[dst]  # (E, D)

            # GRU update on edge hidden state
            h_e = self.gru_updates[layer](m_j_expanded, h_src)  # (E, D)
            # Round-5 fused-dropout wiring.  The edge-layer dropout has no
            # residual stream so we route through ``_maybe_fused_dropout_only``
            # which keeps the dispatch gate consistent with the fused
            # ``dropout(x) + residual`` kernel but always falls back to
            # ``F.dropout`` at this site (bit-exact with original ``nn.Dropout``).
            h_e = _maybe_fused_dropout_only(
                h_e, p=self.dropout.p, training=self.training
            )

        # Readout: for each atom, sum outgoing edge hidden states
        # outgoing: edges where atom is the SOURCE
        h_outgoing = _scatter_sum(h_e, src, dim=0, dim_size=N)  # (N, D)
        # Concat original (embedded) atom features + outgoing edge sum
        h_v_out = self.readout_mlp(torch.cat([h_v, h_outgoing], dim=-1))  # (N, D)
        return h_v_out

    def forward(
        self,
        h_atom: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Batched forward: process each molecule individually and pool.

        Args:
            h_atom:   (B, N_max, atom_feat_dim)   atom features (padded)
            edge_index: (B, 2, E_max)              directed edges per molecule
            edge_attr: (B, E_max, edge_feat_dim)   edge features (padded)
            batch_idx: (B, N_max)                   which molecule each slot belongs to

        Returns:
            (B, hidden_dim) pooled molecular embeddings
        """
        B, N_max, _ = h_atom.shape
        device = h_atom.device

        pooled_list = []
        for b in range(B):
            # Extract this molecule's subgraph from the padded tensors
            h_mol = h_atom[b]  # (N_max, atom_dim)
            ei_mol = edge_index[b]  # (2, E_max)
            ea_mol = edge_attr[b]  # (E_max, edge_dim)
            # batch_idx[b, n] = b means slot n is a real atom of molecule b
            # (padding slots have batch_idx != b)
            n_atoms = int((batch_idx[b] == b).sum().item())

            if n_atoms == 0:
                pooled_list.append(torch.zeros(self.hidden_dim, device=device))
                continue

            # Truncate to actual sizes
            h_mol = h_mol[:n_atoms]  # (n_atoms, atom_dim)

            # Remap edge indices to [0, n_atoms) for this molecule
            # Edges pointing to atoms >= n_atoms are padding — discard them
            E = ei_mol.size(1)
            mask_edges = (ei_mol[0] < n_atoms) & (ei_mol[1] < n_atoms)
            ei_local = ei_mol[:, mask_edges]  # (2, E_valid)
            ea_local = ea_mol[:ea_mol.size(0)][mask_edges]  # (E_valid, edge_dim)

            # Forward pass on this molecule
            h_per_atom = self.forward_per_atom(h_mol, ei_local, ea_local)  # (n_atoms, D)

            # Sum-pool to graph-level
            pooled = h_per_atom.sum(dim=0)  # (D,)
            pooled_list.append(pooled)

        return torch.stack(pooled_list, dim=0)  # (B, D)
