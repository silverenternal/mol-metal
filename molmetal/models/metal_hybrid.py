"""D-MPNN + EGNN hybrid model for MetalCytoToxDB.

Combines:
  1. DirectedMPNN  — 2D bond-graph stream (atom_dim=39, edge_dim=6)
  2. EGNNPredictor — 3D coordinate stream
  3. FusionMLP     — concat [h_2d || h_3d]
  4. Dual head     — pIC50 regression + activity classification

Forward signature:
    (smiles_list, coords_tensor, atom_types) -> (pic50_pred, active_logit)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from molmetal.data.featurize import GraphFeaturizer
from molmetal.models.dmpnn import DirectedMPNN, MPNNConfig
from molmetal.models.egnn_predict import EGNNPredictor, EGNNPredictorConfig
from molmetal.models.fusion import FusionMLP


@dataclass
class MetalHybridConfig:
    """Configuration for the hybrid model."""

    atom_feat_dim: int = 39    # from GraphFeaturizer
    edge_feat_dim: int = 6    # from GraphFeaturizer
    hidden_dim: int = 128
    n_dmpnn_layers: int = 3
    n_egnn_layers: int = 3
    dropout: float = 0.1
    # pIC50 normalization bounds
    pic50_min: float = 4.0
    pic50_max: float = 9.0
    # Number of activity classes (active=1, inactive=0)
    n_activity_classes: int = 2


@dataclass
class Pic50RegressionOutput:
    """Output container for the hybrid model.

    Attributes:
        pic50:         (B,)   predicted pIC50 regression values, clamped to [pic50_min, pic50_max]
        active_logits: (B, 2) binary classification logits over (inactive, active)

    Supports both attribute access (``out.pic50`` / ``out.active_logits``)
    and tuple-unpacking (``pic50_pred, active_logits = model(...)``).
    """

    pic50: torch.Tensor
    active_logits: torch.Tensor

    def __iter__(self):
        yield self.pic50
        yield self.active_logits


class MetalHybridModel(nn.Module):
    """D-MPNN + EGNN hybrid for cytotoxicity prediction.

    Args:
        config: model hyperparameters
        metal_embedding_dim: dimension for metal one-hot embedding (default 32)
    """

    def __init__(
        self,
        config: Optional[MetalHybridConfig] = None,
        metal_embedding_dim: int = 32,
    ):
        super().__init__()
        cfg = config or MetalHybridConfig()
        self.cfg = cfg
        self.hidden_dim = cfg.hidden_dim
        self.pic50_min = cfg.pic50_min
        self.pic50_max = cfg.pic50_max

        # Featurizer (stateless, shared)
        self.featurizer = GraphFeaturizer()

        # D-MPNN (2D stream)
        mpnn_cfg = MPNNConfig(
            atom_feat_dim=cfg.atom_feat_dim,
            edge_feat_dim=cfg.edge_feat_dim,
            hidden_dim=cfg.hidden_dim,
            n_layers=cfg.n_dmpnn_layers,
            dropout=cfg.dropout,
        )
        self.dmpnn = DirectedMPNN(mpnn_cfg)

        # EGNN (3D stream) — input dim matches D-MPNN output
        egnn_cfg = EGNNPredictorConfig(
            in_node_dim=cfg.hidden_dim,
            hidden_dim=cfg.hidden_dim,
            n_layers=cfg.n_egnn_layers,
        )
        self.egnn = EGNNPredictor(egnn_cfg)

        # Fusion: concat D-MPNN + EGNN per-atom features
        self.fusion = FusionMLP(h_dim=cfg.hidden_dim, output_dim=cfg.hidden_dim)

        # Metal centre embedding (one-hot metal type)
        self.metal_embed = nn.Embedding(10, metal_embedding_dim)  # Ru,Ir,Rh,Os,Re,Pt,Pd,Au,Ag,Cu

        # Readout: pooled fused features + metal embedding -> dual head
        self.head_mlp = nn.Sequential(
            nn.Linear(cfg.hidden_dim + metal_embedding_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.ReLU(),
        )

        # pIC50 regression head
        self.pic50_head = nn.Linear(cfg.hidden_dim // 2, 1)

        # Activity classification head (n_activity_classes logits over fused pooled features)
        self.active_head = nn.Linear(cfg.hidden_dim // 2, cfg.n_activity_classes)

    def _featurize_batch(self, smiles_list: List[str], mol_objects: List = None):
        """Featurize a batch of SMILES into padded graph tensors.

        Args:
            smiles_list: list of SMILES strings
            mol_objects: optional list of pre-embedded RDKit Mol objects (same len as smiles_list).
                         If provided, uses these for featurization to ensure consistent atom counts.

        Returns:
            h_atom:     (B, N_max, atom_dim)   atom features
            edge_index: (B, 2, E_max)          directed edge pairs (src, dst)
            edge_attr:  (B, E_max, edge_dim)    edge features
            batch_idx:  (B, N_max)              molecule index per slot (-1=padding)
            atom_mask:   (B, N_max)              True for real atoms, False=padding
        """
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")

        B = len(smiles_list)
        device = next(self.parameters()).device

        atom_feats = []
        edge_indices = []
        edge_attrs = []
        n_atoms_list = []

        for i, smi in enumerate(smiles_list):
            try:
                if mol_objects is not None and mol_objects[i] is not None:
                    mol = mol_objects[i]
                else:
                    mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    raise ValueError(f"RDKit parse failed for {smi!r}")
                feat_dict = self.featurizer(mol)  # pass mol directly
                atom_feats.append(feat_dict["x"])
                edge_indices.append(feat_dict["edge_index"])
                edge_attrs.append(feat_dict["edge_attr"])
                n_atoms_list.append(feat_dict["n_atoms"])
            except Exception:
                atom_feats.append(np.zeros((1, self.cfg.atom_feat_dim), dtype=np.float32))
                edge_indices.append(np.zeros((2, 0), dtype=np.int64))
                edge_attrs.append(np.zeros((0, self.cfg.edge_feat_dim), dtype=np.float32))
                n_atoms_list.append(1)

        N_max = max(max(n_atoms_list), 1)
        E_max = max(max(ei.shape[1] for ei in edge_indices), 1)

        h_atom = torch.zeros(B, N_max, self.cfg.atom_feat_dim, device=device)
        edge_index = torch.zeros(B, 2, E_max, dtype=torch.long, device=device)
        edge_attr = torch.zeros(B, E_max, self.cfg.edge_feat_dim, device=device)
        # Use -1 for padding so batch_idx[b] == b correctly selects only real atoms
        batch_idx = torch.full((B, N_max), -1, dtype=torch.long, device=device)
        atom_mask = torch.zeros(B, N_max, dtype=torch.bool, device=device)

        for b in range(B):
            af = torch.from_numpy(atom_feats[b]).float()
            n_a = min(af.size(0), N_max)
            h_atom[b, :n_a] = af[:n_a]
            atom_mask[b, :n_a] = True

            ei = torch.from_numpy(edge_indices[b]).long()
            n_e = min(ei.size(1), E_max)
            edge_index[b, :, :n_e] = ei[:, :n_e]

            ea = torch.from_numpy(edge_attrs[b]).float()
            n_ea = min(ea.size(0), E_max)
            edge_attr[b, :n_ea] = ea[:n_ea]

            # batch_idx: real atoms are labeled with molecule index b; padding = -1
            batch_idx[b, :n_a] = b

        return h_atom, edge_index, edge_attr, batch_idx, atom_mask

    def forward(
        self,
        smiles_list: List[str],
        coords: torch.Tensor,
        metal_types: torch.Tensor,
        mol_objects: List = None,
    ) -> Pic50RegressionOutput:
        """Forward pass.

        Args:
            smiles_list: list of B ligand SMILES strings
            coords:      (B, N_max, 3) atom 3D coordinates (padded with zeros)
            metal_types: (B,) integer metal type indices (0-9)
            mol_objects: optional list of pre-embedded RDKit Mol objects

        Returns:
            Pic50RegressionOutput(
                pic50         = (B,)   predicted pIC50 values (clamped to [pic50_min, pic50_max])
                active_logits = (B, 2) predicted activity logits over (inactive, active)
            )
        """
        B = len(smiles_list)
        device = coords.device

        # 1. Featurize SMILES -> atom/edge graph tensors
        h_atom, edge_index, edge_attr, batch_idx, atom_mask = self._featurize_batch(
            smiles_list, mol_objects=mol_objects
        )
        N_max = h_atom.size(1)

        # 2. D-MPNN per-atom features (process each molecule individually)
        dmpnn_per_atom = torch.zeros(B, N_max, self.hidden_dim, device=device)
        for b in range(B):
            n_a = atom_mask[b].sum().item()
            if n_a == 0:
                continue
            h_mol = h_atom[b, :n_a]  # (n_a, atom_dim)
            ei_mol = edge_index[b]    # (2, E_max)
            ea_mol = edge_attr[b]     # (E_max, edge_dim)

            # Keep only edges where both src and dst < n_a (drop padded self-loops).
            if ei_mol.size(1) > 0:
                src_all, dst_all = ei_mol[0], ei_mol[1]
                valid_mask = (
                    (src_all < n_a) & (dst_all < n_a) & (src_all != dst_all)
                )
                ei_local = ei_mol[:, valid_mask]
                ea_local = ea_mol[valid_mask]
            else:
                ei_local = torch.zeros((2, 0), dtype=torch.long, device=device)
                ea_local = torch.zeros((0, self.cfg.edge_feat_dim), device=device)

            dmpnn_out = self.dmpnn.forward_per_atom(h_mol, ei_local, ea_local)
            dmpnn_per_atom[b, :n_a] = dmpnn_out

        # 3. EGNN 3D stream — process each molecule individually with remapped edge indices
        egnn_per_atom = torch.zeros(B, N_max, self.hidden_dim, device=device)
        for b in range(B):
            n_a = atom_mask[b].sum().item()
            if n_a == 0:
                continue
            # coords are already padded with zeros beyond n_a
            x_mol = coords[b]       # (N_max, 3) — real atoms have valid coords, padding is 0
            ei_b = edge_index[b]     # (2, E_max) — global batch indices
            h_mol = dmpnn_per_atom[b] # (N_max, D) — DMPNN features (0 for padding)

            # Keep only edges within the real atoms of this molecule.
            # Filter on src<dst (drop zero-padded self-loops AND duplicate reverse edges
            # when the directed edge count exceeds the molecule's actual edges).
            if ei_b.size(1) > 0:
                src_all, dst_all = ei_b[0], ei_b[1]
                # Drop padding (0,0) self-loops and out-of-range edges.
                # Then drop self-loops where src==dst (real or padded).
                valid_edge = (
                    (src_all < n_a) & (dst_all < n_a)
                    & (src_all != dst_all)
                )
                ei_local = ei_b[:, valid_edge]   # (2, E_valid)
            else:
                ei_local = torch.zeros((2, 0), dtype=torch.long, device=device)

            # Use only the first n_a rows of dmpnn features and coords
            h_mol_local = dmpnn_per_atom[b, :n_a]  # (n_a, D)
            x_mol_local = x_mol[:n_a]               # (n_a, 3)

            egnn_out = self.egnn.forward_per_atom(h_mol_local, x_mol_local, ei_local)
            egnn_per_atom[b, :n_a] = egnn_out

        # 4. Fusion: concat [D-MPNN || EGNN] per atom
        fused_per_atom = self.fusion(dmpnn_per_atom, egnn_per_atom)

        # 5. Pool: sum over real atoms per molecule
        pooled_list = []
        for b in range(B):
            mask = atom_mask[b]
            mol_fused = fused_per_atom[b][mask]
            pooled = mol_fused.sum(dim=0)
            pooled_list.append(pooled)
        pooled = torch.stack(pooled_list, dim=0)

        # 6. Add metal embedding
        metal_emb = self.metal_embed(metal_types.long())
        head_input = torch.cat([pooled, metal_emb], dim=-1)

        # 7. Dual head
        h_head = self.head_mlp(head_input)
        pic50_pred = self.pic50_head(h_head).squeeze(-1)
        active_logits = self.active_head(h_head)  # (B, n_activity_classes)

        pic50_pred = pic50_pred.clamp(self.pic50_min, self.pic50_max)

        return Pic50RegressionOutput(
            pic50=pic50_pred,
            active_logits=active_logits,
        )
