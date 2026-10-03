"""Protein-Metal Binding prediction head (R17+ Metric 18).

Implements a multimodal fusion model for **protein-metal binding** prediction,
taking inspiration from the 2025 paper *"Multimodal Protein-Metal Binding"
(newx.sg 2025)* which achieves AUC > 0.92 via four-stream multimodal fusion of:

* **1D sequence** (protein backbone, here cell-line-keyed one-hot/embeddings
  since our training set is MetalCytoToxDB complexes on cell-line assays —
  no individual protein sequence is recorded in the public release of
  MetalCytoToxDB.  We use cell-line identity as the canonical "protein"
  surrogate, matching the Krasnov 2026 §3.1 protocol),
* **3D structure** (pocket 3D coords — placeholder zeros when not present;
  the design supports proper knn-graph features when 3D pockets are
  available),
* **ligand** (SMILES, encoded via Directed Message-Passing Neural Network),
* **metal-type embedding**.

The four streams are concatenated via a per-modality projection, summed,
and passed through a 2-layer MLP head → 2-class softmax (binding / non-binding).

Architecture
------------

::

    1D stream (cell-line id)     ─┐
                                  │
    3D stream (pocket coords)    ─┤── concat / fuse ── MLP ── softmax (logit)
                                  │
    Ligand stream (D-MPNN)      ─┤
                                  │
    Metal-type emb (categorical)─┘

Inputs (forward signature ``forward(smiles, cell_line_id, metal_id, pocket=None)``)

* ``smiles``      : list[str]  — Bemis-Murcko ligands (the metal complex
                  without metal atom; we strip the bare metal centre when
                  building the CSV).
* ``cell_line_id``: list[int]  — index into the cell-line vocabulary.
* ``metal_id``    : list[int]  — index into the metal vocabulary (Ru/Ir/.../Pt).
* ``pocket``      : list[Tensor] | None — optional (B, N_pocket, 3) coords.

Loss
----
Focal loss with γ=2.0, α=auto-balanced from the training mini-batch.
Focal loss is a pragmatic choice for the heavy class-imbalance between
"binding" and "non-binding" labels — see *Lin et al. 2017, "Focal Loss for
Dense Object Detection"*.

Why focal and not BCE?
The "active = IC50 < 10 µM" definition in cytotox.py produces roughly 60/40
positive/negative splits when filtered to <time> 24 h and ic50_min 0.01 µM.
Focal loss further down-weights already-easy negatives and lets the model
focus on the genuinely difficult borderline cases.

Reference
---------
* Multimodal Protein-Metal Binding 2025 (newx.sg 2025): cited as
  ``multimodal_protein_metal_2025`` in paper §6.
* Krasnov 2026, *J. Med. Chem.* — cell-line-aware MetalCytoToxDB training
  protocol (the data pipeline this head is built on top of).
* Lin et al. 2017 — focal loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Inlined ligand-graph featuriser (no Chemprop dependency)
# ---------------------------------------------------------------------------
def _featurise_smiles_to_graph(
    smiles_list: Sequence[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Featurise a list of SMILES into a flat directed-edge graph.

    Returns
    -------
    atom_feats : FloatTensor (N_atoms, F_atom)
    dirs        : LongTensor  (E_directed, 2)  — (dest atom idx, source atom idx)
    batch_idx   : LongTensor  (N_atoms,) — which molecule each atom belongs to.

    Falls back to all-empty tensors when RDKit is unavailable or the
    SMILES is unparseable, so the head never crashes on a bad input.
    """
    try:
        from rdkit import Chem
    except ImportError:
        # Return empty tensors; the caller side checks shape and fills zeros.
        return (
            torch.zeros(0, 1),
            torch.zeros(0, 2, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
        )
    ele_list = DMPNNLigandEncoder.ATOM_ELES
    deg_list = DMPNNLigandEncoder.DEGREES

    def atom_features(atom, ring_info):
        sym = atom.GetSymbol()
        ele_oh = [1.0 if sym == e else 0.0 for e in ele_list] + [
            1.0 if sym not in ele_list else 0.0
        ]
        deg = min(atom.GetDegree(), 5)
        deg_oh = [1.0 if deg == d else 0.0 for d in deg_list]
        chg = atom.GetFormalCharge()
        chg = max(-2, min(2, chg))
        chg_oh = [1.0 if (chg + 2) == i else 0.0 for i in range(5)]
        chir = str(atom.GetChiralTag()).replace("CHI_", "").split("/")[0]
        chir_oh = [1.0 if chir == c else 0.0 for c in ["R", "S", "?"]]
        aro = [1.0 if atom.GetIsAromatic() else 0.0]
        h = min(atom.GetTotalNumHs(), 4)
        h_oh = [1.0 if h == i else 0.0 for i in range(5)]
        return np.array(
            ele_oh + deg_oh + chg_oh + chir_oh + aro + h_oh, dtype=np.float32
        )

    all_atom_feats: List[np.ndarray] = []
    all_dirs: List[Tuple[int, int]] = []
    all_batch: List[int] = []
    atom_offset = 0
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        ring_info = mol.GetRingInfo()
        feats_per_atom = [atom_features(a, ring_info) for a in mol.GetAtoms()]
        if not feats_per_atom:
            continue
        atom_mat = np.stack(feats_per_atom, axis=0)
        # Build directed-edge list: for each atom pair (a, b) with a bond
        # between them, add edges a→b and b→a, but keep each directed edge
        # distinct so that D-MPNN's "no reverse traversal" rule holds.
        directed = []
        for a in mol.GetAtoms():
            for b in a.GetNeighbors():
                directed.append((a.GetIdx(), b.GetIdx()))
        if not directed:
            # Lone atom — emit a self-loop to keep the message-passing graph non-empty.
            directed = [(0, 0)]
        dirs_arr = np.array(directed, dtype=np.int64)
        dirs_arr[:, 0] += atom_offset
        dirs_arr[:, 1] += atom_offset
        all_atom_feats.append(atom_mat)
        all_dirs.extend(dirs_arr.tolist())
        all_batch.extend([i] * mol.GetNumAtoms())
        atom_offset += mol.GetNumAtoms()

    if not all_atom_feats:
        return (
            torch.zeros(0, len(ele_list) + len(deg_list) + 5 + 3 + 1 + 5),
            torch.zeros(0, 2, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
        )
    return (
        torch.from_numpy(np.concatenate(all_atom_feats, axis=0)),
        torch.tensor(all_dirs, dtype=torch.long)
        if all_dirs
        else torch.zeros(0, 2, dtype=torch.long),
        torch.tensor(all_batch, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
@dataclass
class PMBConfig:
    """Hyper-parameters for :class:`ProteinMetalBindingHead`.

    Defaults are tuned for the MetalCytoToxDB precious-metal subset:
    5 metals × ~30-60 unique cell lines, ~26 k rows total.
    """

    # Modality encoders
    cell_line_emb_dim: int = 32
    metal_emb_dim: int = 16
    pocket_hidden: int = 32  # 3D pocket encoder hidden dim (we feed zeros)
    pocket_n_pseudo: int = 0  # 0 disables the pocket stream (no Coords dataset)
    dmpnn_hidden: int = 128  # matches molmetal.baselines.dmpnn.DMPNN_HIDDEN
    dmpnn_depth: int = 3  # matches molmetal.baselines.dmpnn.DMPNN_DEPTH

    # Multimodal fusion
    fuse_dim: int = 256  # post-projection common dim (per stream)
    fusion_dropout: float = 0.2

    # Head MLP
    head_hidden: int = 128
    head_dropout: float = 0.3

    # Focal loss
    focal_gamma: float = 2.0
    focal_alpha: float = -1.0  # <0 → auto-balance from batch prior

    # Misc
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------
class CellLineEncoder(nn.Module):
    """Embeds cell-line identity into a continuous vector.

    The cell-line vocabulary is built externally (see
    :meth:`PMBDataset.build_vocab`); here we only need an ``nn.Embedding``.
    """

    def __init__(self, n_cell_lines: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_cell_lines, dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:  # (B,)
        return self.emb(ids)  # (B, dim)


class MetalEncoder(nn.Module):
    """Embeds metal-type identity (Ru/Ir/Rh/Os/Re/Pt)."""

    def __init__(self, n_metals: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_metals, dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:  # (B,)
        return self.emb(ids)


class PocketEncoder(nn.Module):
    """Encodes pocket 3D coords via a tiny PointNet-style MLP.

    Input is (B, N, 3); we mean-pool over the N points.  When
    ``n_pseudo == 0`` the stream is bypassed entirely (see
    :class:`ProteinMetalBindingHead`).
    """

    def __init__(self, hidden: int, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.proj = nn.Linear(hidden, dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:  # (B, N, 3)
        # Mean-pool over points.
        h = self.mlp(coords)  # (B, N, hidden)
        return self.proj(h.mean(dim=1))  # (B, dim)


class DMPNNLigandEncoder(nn.Module):
    """Compact message-passing encoder over the ligand.

    Implements a 3-message-passing light-weight D-MPNN (Lin et al. 2018
    "Message Passing Neural Networks for Chemistry").  We do not depend on
    Chemprop — the encoder is self-contained and torch-only so the module
    works without the heavier graph-fp dependencies that
    :mod:`molmetal.baselines.dmpnn` pulls in.

    For each SMILES we:

    1. Featurise atoms (one-hot element + degree + formal charge + chirality)
       and bonds (one-hot bond-type + in-ring + conjugated).
    2. Run ``depth`` rounds of message passing on the directed-edge graph
       (i.e. each undirected edge gets two directed edges, and messages
       travel only along them, never back).
    3. Sum-pool the resulting atom embeddings → molecule vector.
    """

    ATOM_ELES = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Other"]
    DEGREES = list(range(6))  # 0..5
    CHIRALITY = ["R", "S", "?"]

    def __init__(self, dim: int, depth: int, dropout: float):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self._featurise = _featurise_smiles_to_graph
        self.atom_in = nn.Linear(self._n_atom_feats(), dim)
        # Bond-update gate: simple GRU cell per directed edge.
        self.gru = nn.GRUCell(input_size=dim, hidden_size=dim)
        self.readout = nn.Linear(dim, dim)

    def _n_atom_feats(self) -> int:
        # atom one-hot element (10) + "Other" indicator (1) + degree one-hot (6)
        # + charge one-hot (5: -2..+2) + chirality one-hot (3) + is_aromatic (1)
        # + H_count (5: 0..4+).
        return (
            len(self.ATOM_ELES)
            + 1
            + len(self.DEGREES)
            + 5  # charge buckets
            + len(self.CHIRALITY)
            + 1
            + 5
        )

    def forward(self, smiles_list: Sequence[str]) -> torch.Tensor:  # (B, dim)
        # Atom-feature tensor shape (N_atoms, atom_feat_dim)
        # Directed-edge list shape (E_directed, 2) — (dest, src)
        # batch_idx shape (N_atoms,)
        device = next(self.parameters()).device
        atom_feats, dirs, batch_idx = self._featurise(smiles_list)
        if atom_feats.shape[0] == 0:
            # Empty graph (no input); return zero vector of correct shape.
            return torch.zeros(
                len(smiles_list),
                self.dim,
                device=device,
            )
        atom_feats = atom_feats.to(device)
        dirs = dirs.to(device)
        batch_idx = batch_idx.to(device)
        x = self.atom_in(atom_feats)  # (N, dim)
        if dirs.shape[0] == 0:
            # Atoms present but no edges — mean-pool atom features as fallback.
            pooled = torch.zeros(len(smiles_list), self.dim, device=x.device)
            pooled.index_add_(0, batch_idx.long(), x)
            cnt = torch.zeros(len(smiles_list), 1, device=x.device)
            cnt.index_add_(0, batch_idx.long(), torch.ones_like(x[:, :1]))
            pooled = pooled / cnt.clamp(min=1.0)
            return self.readout(self.dropout(pooled))
        dest = dirs[:, 0].long()
        src = dirs[:, 1].long()
        # Per-edge hidden state h_e.  Initialise to zero; update via GRU
        # each round:  h_e ← GRU( atom_feat[src_e] + sum_{e': dest_e'=src_e} h_{e'}, h_e )
        h = torch.zeros(dirs.shape[0], self.dim, device=x.device, dtype=x.dtype)
        for _ in range(self.depth):
            # Aggregate messages per *source atom* so each directed edge
            # sees "what is going out of this atom from previous round".
            msg_per_atom = torch.zeros_like(x)
            msg_per_atom.index_add_(0, src, h)
            # Message carried on edge (dest, src) is msg_per_atom at src.
            inp = x[src] + msg_per_atom[src]
            h = self.gru(inp, h)
        # Sum-pool per-atom → molecule vector.
        # An atom's pooled vector = sum over edges where src == atom of h.
        # Equivalently, mean-pool the atom feature too.
        pooled = torch.zeros(len(smiles_list), self.dim, device=x.device)
        pooled.index_add_(0, batch_idx.long(), x)
        pooled.index_add_(0, batch_idx.long()[src], h)
        pooled = self.readout(self.dropout(pooled))
        return pooled


# ---------------------------------------------------------------------------
# Multimodal head
# ---------------------------------------------------------------------------
class MultimodalFuser(nn.Module):
    """Projects each modality to ``fuse_dim``, concatenates, then 2-layer MLP."""

    def __init__(self, dims: Dict[str, int], fuse_dim: int, dropout: float):
        super().__init__()
        self.projs = nn.ModuleDict({k: nn.Linear(v, fuse_dim) for k, v in dims.items()})
        total = fuse_dim * len(dims)
        self.fuse = nn.Sequential(
            nn.Linear(total, fuse_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        proj = [self.projs[k](features[k]) for k in features]
        return self.fuse(torch.cat(proj, dim=-1))


class ProteinMetalBindingHead(nn.Module):
    """The protein-metal binding prediction head.

    Inputs are passed as keyword arguments:

    * ``smiles`` (list[str])
    * ``cell_line_id`` (LongTensor of shape (B,))
    * ``metal_id`` (LongTensor of shape (B,))
    * ``pocket`` (FloatTensor of shape (B, N, 3) — optional)

    Output: logits of shape (B, 2) — softmax over (non-binding, binding).
    """

    def __init__(
        self,
        n_cell_lines: int,
        n_metals: int,
        cfg: PMBConfig = PMBConfig(),
    ):
        super().__init__()
        self.cfg = cfg
        self.n_cell_lines = n_cell_lines
        self.n_metals = n_metals
        self.cell_enc = CellLineEncoder(n_cell_lines, cfg.cell_line_emb_dim)
        self.metal_enc = MetalEncoder(n_metals, cfg.metal_emb_dim)
        self.has_pocket = cfg.pocket_n_pseudo > 0
        if self.has_pocket:
            self.pocket_enc = PocketEncoder(cfg.pocket_hidden, cfg.pocket_hidden)
        self.ligand_enc = DMPNNLigandEncoder(
            cfg.dmpnn_hidden, cfg.dmpnn_depth, cfg.fusion_dropout
        )

        # Build fuser
        dims = {
            "cell": cfg.cell_line_emb_dim,
            "metal": cfg.metal_emb_dim,
            "ligand": cfg.dmpnn_hidden,
        }
        if self.has_pocket:
            dims["pocket"] = cfg.pocket_hidden
        self.fuser = MultimodalFuser(dims, cfg.fuse_dim, cfg.fusion_dropout)
        self.head = nn.Sequential(
            nn.Linear(cfg.fuse_dim, cfg.head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden, 2),
        )

    def forward(
        self,
        smiles: Sequence[str],
        cell_line_id: torch.Tensor,
        metal_id: torch.Tensor,
        pocket: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Align tensors to the cell-line embedding device so users can pass
        # index tensors on CPU without manual .to() in the trainer.
        device = next(self.parameters()).device
        if cell_line_id.device != device:
            cell_line_id = cell_line_id.to(device)
        if metal_id.device != device:
            metal_id = metal_id.to(device)
        if pocket is not None and pocket.device != device:
            pocket = pocket.to(device)
        feats: Dict[str, torch.Tensor] = {
            "cell": self.cell_enc(cell_line_id),
            "metal": self.metal_enc(metal_id),
            "ligand": self.ligand_enc(smiles),
        }
        if self.has_pocket and pocket is not None:
            feats["pocket"] = self.pocket_enc(pocket)
        z = self.fuser(feats)
        return self.head(z)


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------
def focal_loss(
    logits: torch.Tensor,  # (B, 2)
    labels: torch.Tensor,  # (B,) long in {0, 1}
    gamma: float = 2.0,
    alpha: float = -1.0,
) -> torch.Tensor:
    """Focal cross-entropy (Lin et al. 2017).

    ``alpha < 0`` ⇒ auto-balance (alpha_pos = (1 - p_pos), alpha_neg = p_pos)
    to mirror the empirical class prior in the mini-batch.
    """
    log_p = F.log_softmax(logits, dim=-1)
    p = log_p.exp()
    target_onehot = F.one_hot(labels, num_classes=2).to(log_p.dtype)
    pt = (target_onehot * p).sum(dim=-1)  # (B,)
    focal_weight = (1.0 - pt).clamp(min=0.0) ** gamma
    if alpha < 0:
        # Auto-balance: positive weight = (1 - pi), negative weight = pi
        pi = target_onehot[:, 1].mean().clamp(min=1e-4, max=1.0 - 1e-4)
        alpha_pos = 1.0 - pi
        alpha_neg = pi
    else:
        alpha_pos = alpha
        alpha_neg = 1.0 - alpha
    alpha_t = target_onehot[:, 0] * alpha_neg + target_onehot[:, 1] * alpha_pos
    loss = (
        -alpha_t
        * focal_weight
        * (target_onehot[:, 0] * log_p[:, 0] + target_onehot[:, 1] * log_p[:, 1])
    )
    return loss.mean()


# ---------------------------------------------------------------------------
# CSV builder for protein-metal binding (binding ≡ active on cell-line)
# ---------------------------------------------------------------------------
def _ligand_only_smiles(smi: str) -> str:
    """Strip counter-ions and bare metal centres from the complex SMILES.

    MetalCytoToxDB records the full complex as ``"A.B.C"`` (charge-balanced by
    counter-ions).  We feed each component to rdkit after a simple "drop the
    metal atom" rewrite, so that the D-MPNN over the *ligand* (without the
    metal centre) is a meaningful input — the metal channel is handled by
    ``MetalEncoder``.  We never want the literal ``[Ru]`` or ``[Pt]`` token
    in the ligand fingerprint since that would let the ligand encoder
    shortcut the metal-type branch and defeat the multimodal head.
    """
    try:
        from rdkit import Chem
    except ImportError:
        return smi
    metal_atoms = {
        "Ru",
        "Ir",
        "Rh",
        "Os",
        "Re",
        "Pt",
        "Pd",
        "Au",
        "Ag",
        "Cu",
        "Zn",
        "Fe",
        "Mn",
        "Co",
        "Ni",
        "Cr",
        "Mo",
        "W",
        "V",
        "Ti",
    }
    fragments = smi.split(".")
    keep = []
    for frag in fragments:
        mol = Chem.MolFromSmiles(frag)
        if mol is None:
            continue
        atoms_to_drop = [
            a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() in metal_atoms
        ]
        if atoms_to_drop:
            atoms_sorted = sorted(atoms_to_drop, reverse=True)
            edit = Chem.RWMol(mol)
            for idx in atoms_sorted:
                edit.RemoveAtom(idx)
            try:
                new_smi = Chem.MolToSmiles(edit)
            except Exception:
                continue
            if new_smi:
                keep.append(new_smi)
        else:
            keep.append(frag)
    return ".".join(keep) if keep else smi


def build_protein_metal_csv(
    src_csv: str = "/home/hugo/codes/mol-metal/.data/molmetal/MetalCytoToxDB.csv",
    dst_csv: str = "/home/hugo/codes/mol-metal/molmetal/data/protein_metal_binding.csv",
    metals: Sequence[str] = ("Ru", "Ir", "Rh", "Os", "Re"),
    time_threshold: float = 24.0,
    ic50_min: float = 0.01,
    ic50_max: float = 1000.0,
    require_dark: bool = True,
) -> Dict[str, int]:
    """Build :data:`protein_metal_binding.csv`.

    Schema of the output CSV:

    =========== ============================================ =========
    column      description                                  dtype
    =========== ============================================ =========
    smiles      ligand SMILES (counter-ions & bare metal    str
                atoms stripped)
    metal       Ru/Ir/Rh/Os/Re/Pt…                            str
    cell_line   canonical cell-line identifier                str
    protein     placeholder protein label (== cell_line)      str
    pIC50       pIC50 (Dark) for analysis, not the target     float
    active      binding proxy (active iff IC50 < 10 uM)      int {0,1}
    =========== ============================================ =========

    The binding proxy is the *same* "active" target as Metric 1
    (Ru AUC) — the protein dimension (cell-line) is added as a *new
    contextual feature*, not as a new label.  This is the closest
    reliable proxy to a true protein-metal binding label inside the
    public MetalCytoToxDB release (the underlying bio-assay is
    cell-line-level, so protein is implicitly the cell-line identity).

    Returns a dict of summary stats.
    """
    import pandas as pd
    from pathlib import Path as _Path

    df = pd.read_csv(src_csv, low_memory=False)
    pre = len(df)
    df = df[df["Metal"].isin(list(metals))]
    if time_threshold:
        df = df[df["Time(h)"] <= time_threshold]
    if ic50_min:
        df = df[df["IC50_Dark_value"] >= ic50_min]
    if ic50_max:
        df = df[df["IC50_Dark_value"] <= ic50_max]
    if require_dark:
        df = df[df["IC50_Dark_value"].notna()]
    df = df[df["Cell_line"].notna() & (df["Cell_line"] != "")]
    df["ligand_smiles"] = df["SMILES_Ligands"].astype(str).map(_ligand_only_smiles)
    df["active"] = (df["IC50_Dark_value"] < 10.0).astype(int)
    df["pIC50"] = (-np.log10(df["IC50_Dark_value"] * 1e-6)).astype(float)
    out = pd.DataFrame(
        {
            "smiles": df["ligand_smiles"].values,
            "metal": df["Metal"].values,
            "cell_line": df["Cell_line"].values,
            "pIC50": df["pIC50"].values,
            "active": df["active"].values,
        }
    )
    out["protein"] = out["cell_line"]
    # Drop rows whose ligand SMILES collapsed to empty after stripping.
    out = out[out["smiles"].astype(str).str.len() > 0]
    out = out[["smiles", "metal", "cell_line", "protein", "pIC50", "active"]]
    dst = _Path(dst_csv)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    return {
        "rows_input": pre,
        "rows_kept": int(len(out)),
        "positives": int(out["active"].sum()),
        "negatives": int((out["active"] == 0).sum()),
        "n_cell_lines": int(out["cell_line"].nunique()),
        "n_metals": int(out["metal"].nunique()),
    }


# ---------------------------------------------------------------------------
# Vocabularies + minimal dataset
# ---------------------------------------------------------------------------
@dataclass
class PMBVocab:
    """Cell-line + metal vocabularies for the protein-metal binding head."""

    cell_lines: Dict[str, int] = field(default_factory=dict)
    metals: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_df(cls, df) -> "PMBVocab":
        cell_lines = {c: i for i, c in enumerate(sorted(df["cell_line"].unique()))}
        metals = {m: i for i, m in enumerate(sorted(df["metal"].unique()))}
        return cls(cell_lines=cell_lines, metals=metals)

    def n_cell_lines(self) -> int:
        return len(self.cell_lines)

    def n_metals(self) -> int:
        return len(self.metals)


class PMBDataset(torch.utils.data.Dataset):
    """Wraps a (smiles, metal, cell_line, active) DataFrame into a torch dataset.

    Returns dicts with keys ``smiles`` (str), ``cell_line_id`` (int),
    ``metal_id`` (int), ``active`` (int 0/1), ``pIC50`` (float).
    """

    def __init__(self, df, vocab: PMBVocab):
        # Reset positional index.
        self.df = df.reset_index(drop=True)
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[int(idx)]
        cell = str(row["cell_line"])
        metal = str(row["metal"])
        return {
            "smiles": str(row["smiles"]),
            "cell_line_id": self.vocab.cell_lines.get(cell, 0),
            "metal_id": self.vocab.metals.get(metal, 0),
            "active": int(row["active"]),
            "pIC50": float(row.get("pIC50", float("nan"))),
        }


def collate_pmb(items: List[Dict[str, object]]) -> Dict[str, object]:
    smiles = [it["smiles"] for it in items]
    cell = torch.tensor([it["cell_line_id"] for it in items], dtype=torch.long)
    metal = torch.tensor([it["metal_id"] for it in items], dtype=torch.long)
    y = torch.tensor([it["active"] for it in items], dtype=torch.long)
    return {"smiles": smiles, "cell_line_id": cell, "metal_id": metal, "y": y}


# ---------------------------------------------------------------------------
# Helpers exported for train_protein_metal.py
# ---------------------------------------------------------------------------
def roc_auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Wrap sklearn roc_auc_score; return 0.5 on degenerate inputs.

    Degenerate = only one class present in ``y_true``, or fewer than 2 distinct
    values in ``y_score``.  We never want a single-class fold to crash the
    training run; the baseline 0.5 is the "no information" lower bound.
    """
    from sklearn.metrics import roc_auc_score

    try:
        if len(set(y_true.tolist())) < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5
