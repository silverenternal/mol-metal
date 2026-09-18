"""MetalHybridV2 — D-MPNN + EGNN hybrid with cross-attention fusion.

This is the V2 variant used in TODO/04 C2 ablation B. Compared to
:mod:`molmetal.models.metal_hybrid` (V1, concat+MLP fusion), V2 replaces the
simple fusion with :class:`CrossAttentionFusion` so that the D-MPNN
representation can query the EGNN representation at every atom position
before pooling.

The rest of the architecture (D-MPNN, EGNN, dual head, metal embedding) is
identical to V1 so the comparison isolates the effect of fusion.
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
from molmetal.models.metal_hybrid import (
    MetalHybridConfig,
    MetalHybridModel,
    Pic50RegressionOutput,
)


@dataclass
class MetalHybridV2Config:
    """Configuration for the V2 hybrid model.

    Keeps the V1 backbone config intact while adding cross-attention fusion
    hyper-parameters.
    """

    base: MetalHybridConfig = None
    fusion_hidden: int = 64
    fusion_n_heads: int = 4
    fusion_dropout: float = 0.0


class MetalHybridV2Model(nn.Module):
    """V2 hybrid: D-MPNN + EGNN with cross-attention fusion.

    Forward signature matches V1:
        (smiles_list, coords, metal_types) -> Pic50RegressionOutput
    """

    def __init__(
        self,
        config: Optional[MetalHybridV2Config] = None,
        metal_embedding_dim: int = 32,
    ):
        super().__init__()
        # Lazily import here to avoid circular import
        from molmetal.models.cross_attention_fusion import CrossAttentionFusion

        if config is None:
            config = MetalHybridV2Config(base=MetalHybridConfig())
        elif config.base is None:
            config.base = MetalHybridConfig()

        self.cfg = config
        self.base_cfg = config.base
        self.hidden_dim = self.base_cfg.hidden_dim
        self.pic50_min = self.base_cfg.pic50_min
        self.pic50_max = self.base_cfg.pic50_max

        # Featurizer (stateless, shared with V1)
        self.featurizer = GraphFeaturizer()

        # D-MPNN (2D stream)
        mpnn_cfg = MPNNConfig(
            atom_feat_dim=self.base_cfg.atom_feat_dim,
            edge_feat_dim=self.base_cfg.edge_feat_dim,
            hidden_dim=self.base_cfg.hidden_dim,
            n_layers=self.base_cfg.n_dmpnn_layers,
            dropout=self.base_cfg.dropout,
        )
        self.dmpnn = DirectedMPNN(mpnn_cfg)

        # EGNN (3D stream)
        egnn_cfg = EGNNPredictorConfig(
            in_node_dim=self.base_cfg.hidden_dim,
            hidden_dim=self.base_cfg.hidden_dim,
            n_layers=self.base_cfg.n_egnn_layers,
        )
        self.egnn = EGNNPredictor(egnn_cfg)

        # Cross-attention fusion (D-MPNN queries, EGNN keys/values)
        self.fusion = CrossAttentionFusion(
            dmpnn_dim=self.base_cfg.hidden_dim,
            egnn_dim=self.base_cfg.hidden_dim,
            hidden=config.fusion_hidden,
            n_heads=config.fusion_n_heads,
            dropout=config.fusion_dropout,
        )

        # Metal centre embedding
        self.metal_embed = nn.Embedding(10, metal_embedding_dim)

        # Readout + dual head (same shape as V1)
        self.head_mlp = nn.Sequential(
            nn.Linear(self.base_cfg.hidden_dim + metal_embedding_dim, self.base_cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.base_cfg.dropout),
            nn.Linear(self.base_cfg.hidden_dim, self.base_cfg.hidden_dim // 2),
            nn.ReLU(),
        )
        self.pic50_head = nn.Linear(self.base_cfg.hidden_dim // 2, 1)
        self.active_head = nn.Linear(
            self.base_cfg.hidden_dim // 2, self.base_cfg.n_activity_classes
        )

    def _featurize_batch(self, smiles_list: List[str], mol_objects: List = None):
        """Featurize a batch of SMILES into padded graph tensors (V1-compatible)."""
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
                feat_dict = self.featurizer(mol)
                atom_feats.append(feat_dict["x"])
                edge_indices.append(feat_dict["edge_index"])
                edge_attrs.append(feat_dict["edge_attr"])
                n_atoms_list.append(feat_dict["n_atoms"])
            except Exception:
                atom_feats.append(
                    np.zeros((1, self.base_cfg.atom_feat_dim), dtype=np.float32)
                )
                edge_indices.append(np.zeros((2, 0), dtype=np.int64))
                edge_attrs.append(
                    np.zeros((0, self.base_cfg.edge_feat_dim), dtype=np.float32)
                )
                n_atoms_list.append(1)

        N_max = max(max(n_atoms_list), 1)
        E_max = max(max(ei.shape[1] for ei in edge_indices), 1)

        h_atom = torch.zeros(B, N_max, self.base_cfg.atom_feat_dim, device=device)
        edge_index = torch.zeros(B, 2, E_max, dtype=torch.long, device=device)
        edge_attr = torch.zeros(B, E_max, self.base_cfg.edge_feat_dim, device=device)
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

            batch_idx[b, :n_a] = b

        return h_atom, edge_index, edge_attr, batch_idx, atom_mask

    def forward(
        self,
        smiles_list: List[str],
        coords: torch.Tensor,
        metal_types: torch.Tensor,
        mol_objects: List = None,
    ) -> Pic50RegressionOutput:
        B = len(smiles_list)
        device = coords.device

        # 1. Featurize
        h_atom, edge_index, edge_attr, batch_idx, atom_mask = self._featurize_batch(
            smiles_list, mol_objects=mol_objects
        )
        N_max = h_atom.size(1)

        # 2. D-MPNN per-atom
        dmpnn_per_atom = torch.zeros(B, N_max, self.hidden_dim, device=device)
        for b in range(B):
            n_a = atom_mask[b].sum().item()
            if n_a == 0:
                continue
            h_mol = h_atom[b, :n_a]
            ei_mol = edge_index[b]
            ea_mol = edge_attr[b]

            if ei_mol.size(1) > 0:
                valid_mask = (ei_mol[0] < n_a) & (ei_mol[1] < n_a)
                ei_local = ei_mol[:, valid_mask]
                ea_local = ea_mol[valid_mask]
            else:
                ei_local = torch.zeros((2, 0), dtype=torch.long, device=device)
                ea_local = torch.zeros((0, self.base_cfg.edge_feat_dim), device=device)

            dmpnn_out = self.dmpnn.forward_per_atom(h_mol, ei_local, ea_local)
            dmpnn_per_atom[b, :n_a] = dmpnn_out

        # 3. EGNN per-atom
        egnn_per_atom = self.egnn(
            dmpnn_per_atom, coords, edge_index, batch_idx, atom_mask
        )

        # 4. Cross-attention fusion (D-MPNN queries EGNN keys/values, residual)
        fused_per_atom = self.fusion(dmpnn_per_atom, egnn_per_atom, mask=atom_mask)

        # 5. Sum-pool over real atoms per molecule
        pooled_list = []
        for b in range(B):
            mask = atom_mask[b]
            mol_fused = fused_per_atom[b][mask]
            pooled_list.append(mol_fused.sum(dim=0))
        pooled = torch.stack(pooled_list, dim=0)

        # 6. Metal embedding
        metal_emb = self.metal_embed(metal_types.long())
        head_input = torch.cat([pooled, metal_emb], dim=-1)

        # 7. Dual head
        h_head = self.head_mlp(head_input)
        pic50_pred = self.pic50_head(h_head).squeeze(-1)
        active_logits = self.active_head(h_head)
        pic50_pred = pic50_pred.clamp(self.pic50_min, self.pic50_max)

        return Pic50RegressionOutput(pic50=pic50_pred, active_logits=active_logits)


__all__ = ["MetalHybridV2Config", "MetalHybridV2Model"]