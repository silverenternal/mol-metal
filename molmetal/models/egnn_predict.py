"""EGNN predictor stack wrapping EquivariantGraphConv from egcn_rocm.

Provides a 3-layer EGNN that takes per-atom scalar features h and
3D coordinates x, performs SE(3)-equivariant message passing, and
returns per-atom 128-dim output features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from molmetal.adapters.egnn_rocm import EquivariantGraphConv


@dataclass
class EGNNPredictorConfig:
    in_node_dim: int = 128   # input atom feature dim (from D-MPNN output)
    hidden_dim: int = 128
    n_layers: int = 3


class EGNNPredictor(nn.Module):
    """Stack of EquivariantGraphConv layers with input/output projection.

    Forward signature (batch mode):

        h : (B, N_max, in_node_dim)   atom scalar features
        x : (B, N_max, 3)             atom 3D coordinates
        edge_index : (B, 2, n_edges)  directed edge connectivity
        batch_idx : (B, N_max)         molecule index per slot (-1=padding)
        atom_mask : (B, N_max)          True for real atoms, False=padding

    Returns:
        (B, N_max, hidden_dim) per-atom updated features
    """

    def __init__(self, config: Optional[EGNNPredictorConfig] = None):
        super().__init__()
        cfg = config or EGNNPredictorConfig()
        self.in_node_dim = cfg.in_node_dim
        self.hidden_dim = cfg.hidden_dim
        self.n_layers = cfg.n_layers

        # Input projection: atom features -> hidden_dim
        self.input_proj = nn.Linear(self.in_node_dim, self.hidden_dim)

        # Stack of EGNN layers
        self.layers = nn.ModuleList(
            [
                EquivariantGraphConv(self.hidden_dim, self.hidden_dim)
                for _ in range(self.n_layers)
            ]
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch_idx: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h:          (B, N_max, in_node_dim)
            x:          (B, N_max, 3)
            edge_index: (B, 2, n_edges)
            batch_idx:  (B, N_max)  molecule index per slot (-1 for padding)
            atom_mask:  (B, N_max)  True = real atom, False = padding

        Returns:
            (B, N_max, hidden_dim) per-atom updated features
        """
        B, N_max, _ = h.shape
        device = h.device

        h_out_list = []
        for b in range(B):
            n_atoms = atom_mask[b].sum().item()  # number of real atoms

            if n_atoms == 0:
                h_out_list.append(torch.zeros(N_max, self.hidden_dim, device=device))
                continue

            h_b = h[b, :n_atoms]           # (n_atoms, D)
            x_b = x[b, :n_atoms]            # (n_atoms, 3) — coords for real atoms only
            ei_b = edge_index[b]            # (2, E_max)

            # Keep only edges within the valid atom range
            if ei_b.size(1) > 0:
                valid_edge = (ei_b[0] < n_atoms) & (ei_b[1] < n_atoms)
                ei_local = ei_b[:, valid_edge]   # (2, E_valid)
            else:
                ei_local = torch.zeros((2, 0), dtype=torch.long, device=device)

            # Project input
            h_proj = self.input_proj(h_b)  # (n_atoms, hidden_dim)

            # EGNN passes
            for layer in self.layers:
                h_proj, x_b = layer(h_proj, x_b, ei_local)

            h_final = self.output_proj(h_proj)  # (n_atoms, hidden_dim)

            # Place back into padded tensor
            h_padded = torch.zeros(N_max, self.hidden_dim, device=device)
            h_padded[:n_atoms] = h_final
            h_out_list.append(h_padded)

        return torch.stack(h_out_list, dim=0)  # (B, N_max, hidden_dim)

    def forward_per_atom(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Per-atom (non-batched) version.

        Args:
            h: (N_atoms, in_node_dim) — must be consecutive atoms 0..N-1
            x: (N_atoms, 3)
            edge_index: (2, n_edges) — src/dst in [0, N_atoms)
        """
        N = h.size(0)
        h_proj = self.input_proj(h)  # (N, hidden_dim)
        for layer in self.layers:
            h_proj, x = layer(h_proj, x, edge_index, n_nodes=N)
        return self.output_proj(h_proj)

    def pool(self, h: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        """Sum-pool per-atom features over molecules.

        Args:
            h: (B, N_max, D)
            atom_mask: (B, N_max)  True for real atoms
        Returns:
            (B, D) pooled features
        """
        B, N_max, D = h.shape
        pooled_list = []
        for b in range(B):
            mask = atom_mask[b]  # (N_max,)
            mol_h = h[b][mask]   # (n_atoms, D)
            pooled_list.append(mol_h.sum(dim=0))  # (D,)
        return torch.stack(pooled_list, dim=0)  # (B, D)
