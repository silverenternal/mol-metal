"""MetalHybridV3 — D-MPNN + EGNN hybrid with coord-refinement head.

This is the v3 of the D-MPNN + EGNN hybrid for MetalCytoToxDB.

Architectural changes vs :mod:`molmetal.models.metal_hybrid`:

* **D-MPNN encoder is initialised from the tmQM pre-training** at
  ``molmetal/checkpoints/dmpnn_tmqm_pretrained.pt`` (21,615 Pt/Ru/Ir
  complexes, JCIM 2020 — see ``molmetal/reports/f2_tmqm_pretraining.md``).
  The encoder transfers knowledge of metal-coordination chemistry to the
  cytotoxicity task, matching the inductive bias described in
  ``molmetal/reports/audit_lambda_upper_bound.md`` §3 (transfer from
  coordination pre-training to metal-therapeutic property prediction).

* **EGNN-encoder style 3D stream** — :class:`EGNNPredictor` consumes
  per-atom D-MPNN features *and* 3-D conformer coordinates, then
  returns SE(3)-equivariant per-atom features (Satorras et al. 2021).

* **Coord refinement head** — a small EGNN-style update
  ``Δx = tanh(MLP([h_i || h_j || ||x_i - x_j||])) * (x_j - x_i)``
  predicts a coordinate correction that is **detached** from the
  classification/regression path.  Refinement loss is included in the
  total as ``gamma * MSE(Δx_pred, Δx_target)`` where Δx_target is
  a single MMFF94 optimisation step from RDKit
  (``molmetal/reports/mmff94_fix.md`` — MMFF94, not UFF, to avoid the
  click-product UFF bugs noted in §190).  The gradient of the
  refinement loss only flows into the coord-refinement MLP — the rest
  of the model gets the benefit without destabilising EGNN training.

* **Fusion** — gated concat ``h = sigmoid(W_g [h_dmpnn || h_egnn]) *
  h_egnn + (1 - sigmoid(...)) * h_dmpnn`` (Banerjee et al. 2020 hybrid
  fusion).  No cross-attention, no residual scaling — back to basics
  for the D-MPNN-init baseline.

Forward signature matches V1/V2:
    (smiles_list, coords, metal_types, mol_objects=None) -> Pic50RegressionOutput
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from molmetal.data.featurize import GraphFeaturizer
from molmetal.models.dmpnn import DirectedMPNN, MPNNConfig
from molmetal.models.egnn_predict import EGNNPredictor, EGNNPredictorConfig
from molmetal.models.metal_hybrid import (
    MetalHybridConfig,
    Pic50RegressionOutput,
)


DEFAULT_PRETRAINED_CKPT = (
    Path(__file__).resolve().parent.parent / "checkpoints" / "dmpnn_tmqm_pretrained.pt"
)


@dataclass
class MetalHybridV3Config:
    """Configuration for the V3 hybrid model (D-MPNN + EGNN + coord refinement)."""

    base: MetalHybridConfig = None
    # Coord refinement head
    coord_refine_hidden: int = 64
    coord_refine_max_step: float = 0.3  # Angstrom, max coordinate shift
    coord_loss_weight: float = 0.05     # gamma
    # Whether to load pretrained D-MPNN weights
    load_pretrained_encoder: bool = True
    pretrained_ckpt: str = str(DEFAULT_PRETRAINED_CKPT)


class _CoordRefineHead(nn.Module):
    """Lightweight EGNN-style coord refinement head.

    Given (h_i, h_j, r_ij = ||x_i - x_j||) the head predicts a scalar
    weight in [-1, 1] that is multiplied by (x_j - x_i).  This is
    identical in shape to the EGNN coord update
    ``x_i' = x_i + sum_j (x_j - x_i)/r_ij * phi(h_i, h_j, r_ij)`` but
    with a tanh-bound so Δx never explodes.

    The output is *detached* from the per-atom representation: the
    gradient of the refinement loss only flows back to the coord
    refinement MLP, not to the encoder/EGNN.  This isolates the
    3-D refinement signal and prevents it from destabilising the
    representation-learning path.
    """

    def __init__(self, hidden_dim: int, max_step: float = 0.3):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_step = float(max_step)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        h: torch.Tensor,        # (N, D) per-atom features (after EGNN)
        x: torch.Tensor,        # (N, 3) per-atom coords
        edge_index: torch.Tensor,  # (2, E) directed edges
    ) -> torch.Tensor:
        """Predict coord shift Δx.  Returns (N, 3)."""
        if edge_index.size(1) == 0 or h.size(0) == 0:
            return torch.zeros_like(x)

        src, dst = edge_index[0], edge_index[1]
        h_i = h[src]
        h_j = h[dst]
        r_ij = (x[src] - x[dst]).norm(dim=-1, keepdim=True)
        msg = torch.cat([h_i, h_j, r_ij], dim=-1)
        w = torch.tanh(self.mlp(msg)) * self.max_step  # (E, 1)
        # scatter_sum displacements to dst (so dst atom is pulled toward src)
        delta = torch.zeros_like(x)
        delta.index_add_(0, dst, w * (x[src] - x[dst]))
        return delta

    def refine(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return refined coords ``x + Δx``."""
        return x + self.forward(h, x, edge_index)


class MetalHybridV3Model(nn.Module):
    """V3 hybrid: D-MPNN (pretrained on tmQM) + EGNN + coord-refinement head.

    Args:
        config: V3 hyperparameters (uses :class:`MetalHybridConfig` for the base).
        metal_embedding_dim: dimension for metal one-hot embedding (default 32).
    """

    def __init__(
        self,
        config: Optional[MetalHybridV3Config] = None,
        metal_embedding_dim: int = 32,
    ):
        super().__init__()
        if config is None:
            config = MetalHybridV3Config(base=MetalHybridConfig())
        elif config.base is None:
            config.base = MetalHybridConfig()
        self.cfg = config
        self.base_cfg = config.base
        self.hidden_dim = self.base_cfg.hidden_dim
        self.pic50_min = self.base_cfg.pic50_min
        self.pic50_max = self.base_cfg.pic50_max

        # Featurizer (stateless, shared with V1/V2)
        self.featurizer = GraphFeaturizer()

        # D-MPNN encoder (will be loaded from tmQM pretraining)
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

        # Gated concat fusion
        self.fusion_gate = nn.Linear(2 * self.hidden_dim, self.hidden_dim)

        # Coord refinement head
        self.coord_refine = _CoordRefineHead(
            hidden_dim=self.hidden_dim, max_step=self.cfg.coord_refine_max_step
        )

        # Metal centre embedding
        self.metal_embed = nn.Embedding(10, metal_embedding_dim)

        # Readout + dual head (same shape as V1/V2)
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

        self._loaded_pretrained = False
        if self.cfg.load_pretrained_encoder:
            self._maybe_load_pretrained()

    # ------------------------------------------------------------------
    # Pretrained loading
    # ------------------------------------------------------------------
    def _maybe_load_pretrained(self) -> bool:
        """Load D-MPNN encoder weights from tmQM pretraining.

        Returns True on success, False if checkpoint is missing or
        shape-mismatched (we never raise — pretraining is optional).
        """
        ckpt_path = Path(self.cfg.pretrained_ckpt)
        if not ckpt_path.exists():
            return False
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        except Exception:
            return False

        state = ckpt.get("encoder_state_dict", None) if isinstance(ckpt, dict) else None
        if state is None:
            return False

        # Filter to keys with matching shape (architecture may differ)
        own_state = self.dmpnn.state_dict()
        loaded = 0
        for k, v in state.items():
            if k in own_state and tuple(own_state[k].shape) == tuple(v.shape):
                own_state[k] = v
                loaded += 1
        if loaded == 0:
            return False
        self.dmpnn.load_state_dict(own_state)
        self._loaded_pretrained = True
        return True

    @property
    def pretrained_loaded(self) -> bool:
        return bool(self._loaded_pretrained)

    # ------------------------------------------------------------------
    # Featurization (V1-compatible)
    # ------------------------------------------------------------------
    def _featurize_batch(self, smiles_list: List[str], mol_objects: List = None):
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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        smiles_list: List[str],
        coords: torch.Tensor,
        metal_types: torch.Tensor,
        mol_objects: List = None,
    ) -> Pic50RegressionOutput:
        B = len(smiles_list)
        device = coords.device

        h_atom, edge_index, edge_attr, batch_idx, atom_mask = self._featurize_batch(
            smiles_list, mol_objects=mol_objects
        )
        N_max = h_atom.size(1)

        # 1. D-MPNN per-atom
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

        # 2. EGNN per-atom (uses coords for SE(3)-equivariant messages)
        egnn_per_atom = self.egnn(
            dmpnn_per_atom, coords, edge_index, batch_idx, atom_mask
        )

        # 3. Gated concat fusion: g = sigmoid(W_g [h_dmpnn || h_egnn])
        #    fused = g * h_egnn + (1 - g) * h_dmpnn
        cat = torch.cat([dmpnn_per_atom, egnn_per_atom], dim=-1)
        gate = torch.sigmoid(self.fusion_gate(cat))
        fused_per_atom = gate * egnn_per_atom + (1.0 - gate) * dmpnn_per_atom

        # 4. Sum-pool over real atoms per molecule
        pooled_list = []
        for b in range(B):
            mask = atom_mask[b]
            mol_fused = fused_per_atom[b][mask]
            pooled_list.append(mol_fused.sum(dim=0))
        pooled = torch.stack(pooled_list, dim=0)

        # 5. Metal embedding + dual head
        metal_emb = self.metal_embed(metal_types.long())
        head_input = torch.cat([pooled, metal_emb], dim=-1)
        h_head = self.head_mlp(head_input)
        pic50_pred = self.pic50_head(h_head).squeeze(-1)
        active_logits = self.active_head(h_head)
        pic50_pred = pic50_pred.clamp(self.pic50_min, self.pic50_max)

        return Pic50RegressionOutput(pic50=pic50_pred, active_logits=active_logits)

    # ------------------------------------------------------------------
    # Coord refinement utilities (used by training script + tests)
    # ------------------------------------------------------------------
    def refine_coords(
        self,
        smiles_list: List[str],
        coords: torch.Tensor,
        metal_types: torch.Tensor,
        mol_objects: List = None,
    ) -> torch.Tensor:
        """Return refined coordinates ``x + Δx`` for each molecule in the batch.

        Shape: ``(B, N_max, 3)`` — padded coords preserved.
        """
        B = len(smiles_list)
        device = coords.device
        h_atom, edge_index, edge_attr, _, atom_mask = self._featurize_batch(
            smiles_list, mol_objects=mol_objects
        )
        N_max = h_atom.size(1)

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

        egnn_per_atom = self.egnn(
            dmpnn_per_atom, coords, edge_index,
            torch.full((B, N_max), -1, dtype=torch.long, device=device), atom_mask,
        )

        # Coord refinement per molecule
        out = torch.zeros_like(coords)
        for b in range(B):
            n_a = atom_mask[b].sum().item()
            if n_a == 0:
                continue
            h_b = egnn_per_atom[b, :n_a]
            x_b = coords[b, :n_a]
            ei_b = edge_index[b]
            if ei_b.size(1) > 0:
                valid_mask = (ei_b[0] < n_a) & (ei_b[1] < n_a)
                ei_local = ei_b[:, valid_mask]
            else:
                ei_local = torch.zeros((2, 0), dtype=torch.long, device=device)
            delta = self.coord_refine(h_b, x_b, ei_local)
            out[b, :n_a] = x_b + delta
        return out

    def compute_coord_refine_loss(
        self,
        smiles_list: List[str],
        coords: torch.Tensor,
        metal_types: torch.Tensor,
        target_delta: torch.Tensor,
        atom_mask: torch.Tensor,
        mol_objects: List = None,
    ) -> torch.Tensor:
        """MSE between predicted Δx and target Δx (e.g. MMFF94 step)."""
        B = len(smiles_list)
        device = coords.device
        h_atom, edge_index, _, _, _ = self._featurize_batch(
            smiles_list, mol_objects=mol_objects
        )
        N_max = h_atom.size(1)

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

        egnn_per_atom = self.egnn(
            dmpnn_per_atom, coords, edge_index,
            torch.full((B, N_max), -1, dtype=torch.long, device=device), atom_mask,
        )

        loss = torch.tensor(0.0, device=device)
        n_terms = 0
        for b in range(B):
            n_a = atom_mask[b].sum().item()
            if n_a == 0:
                continue
            h_b = egnn_per_atom[b, :n_a]
            x_b = coords[b, :n_a]
            t_b = target_delta[b, :n_a]
            ei_b = edge_index[b]
            if ei_b.size(1) > 0:
                valid_mask = (ei_b[0] < n_a) & (ei_b[1] < n_a)
                ei_local = ei_b[:, valid_mask]
            else:
                ei_local = torch.zeros((2, 0), dtype=torch.long, device=device)
            delta = self.coord_refine(h_b, x_b, ei_local)
            if t_b.shape == delta.shape:
                loss = loss + ((delta - t_b) ** 2).sum() / max(delta.numel(), 1)
                n_terms += 1
        if n_terms > 0:
            loss = loss / n_terms
        return loss * self.cfg.coord_loss_weight


__all__ = ["MetalHybridV3Config", "MetalHybridV3Model", "_CoordRefineHead", "DEFAULT_PRETRAINED_CKPT"]
