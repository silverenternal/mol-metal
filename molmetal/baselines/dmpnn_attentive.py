"""Attentive D-MPNN (Directed Message Passing Neural Network) baseline.

Minimal PyTorch implementation — no torch_geometric / torch_scatter required.
Architecture follows the original D-MPNN paper (Gilmer et al. 2017, ICML) with
attentive readout (Yang et al. 2019 "Analyzing Learned Molecular Representations" style).

Key design differences from vanilla D-MPNN:
- Directed edge graph: each bond becomes two directed edges (u→v and v→u).
- Per-edge hidden state h_e updated by GRU over D rounds.
- Atom readout: h_v' = sum_outgoing h_e, concatenated with initial atom embedding.
- Attentive pooling: a_v = softmax(MLP(h_v')) over atoms; output = sum_v a_v * h_v'.

Hyperparameters: hidden=128, depth=3, lr=1e-3, dropout=0.1.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger

from molmetal.baselines.eval_utils import compute_metrics, per_cell_line_auc
from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset
from molmetal.data.splits import (
    ChemicalSplitter,
    LigandDeduplicatedSplitter,
    RandomSplitter,
    ScaffoldSplitter,
    TemporalSplitter,
)

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
DMPNN_HIDDEN: int = 128
DMPNN_DEPTH: int = 3
DMPNN_DROPOUT: float = 0.1
DMPNN_LR: float = 1e-3
DMPNN_EPOCHS: int = 30
DMPNN_BATCH_SIZE: int = 64

# ---------------------------------------------------------------------------
# RDKit featurisation (same as vanilla D-MPNN)
# ---------------------------------------------------------------------------
ATOM_FEATURE_DIM = 39
BOND_FEATURE_DIM = 10


def _atom_features(atom: Chem.Atom) -> np.ndarray:
    """Extract ~39-dim atom feature vector from RDKit atom."""
    feats = []
    Z = atom.GetAtomicNum()
    z_counts = [6, 7, 8, 9, 15, 16, 17, 26, 29, 45, 46, 77, 78]
    z_onehot = [1 if Z == z else 0 for z in z_counts]
    z_other = 1 if Z not in z_counts else 0
    feats.extend(z_onehot + [z_other])

    fc = atom.GetFormalCharge()
    feats.extend([fc == -2, fc == -1, fc == 0, fc == 1, fc == 2])

    hybrid_types = [
        Chem.HybridizationType.SP,
        Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3,
        Chem.HybridizationType.SP3D,
        Chem.HybridizationType.SP3D2,
    ]
    hybrid = atom.GetHybridization()
    for ht in hybrid_types:
        feats.append(1 if hybrid == ht else 0)
    feats.append(1 if hybrid not in hybrid_types else 0)

    feats.append(1 if atom.IsInRing() else 0)
    feats.append(1 if atom.GetIsAromatic() else 0)
    feats.append(atom.GetDegree())
    feats.append(atom.GetTotalNumHs())
    feats.append(atom.GetChiralTag())

    while len(feats) < ATOM_FEATURE_DIM:
        feats.append(0)
    return np.array(feats[:ATOM_FEATURE_DIM], dtype=np.float32)


def _bond_features(bond: Chem.Bond) -> np.ndarray:
    """Extract ~10-dim bond feature vector from RDKit bond."""
    feats = []
    bt = bond.GetBondType()
    feats.append(1 if bt == Chem.BondType.SINGLE else 0)
    feats.append(1 if bt == Chem.BondType.DOUBLE else 0)
    feats.append(1 if bt == Chem.BondType.TRIPLE else 0)
    feats.append(1 if bt == Chem.BondType.AROMATIC else 0)
    feats.append(1 if bond.GetIsConjugated() else 0)
    feats.append(1 if bond.IsInRing() else 0)
    stereo = bond.GetStereo()
    feats.append(1 if stereo == Chem.BondStereo.STEREOANY else 0)
    feats.append(1 if bond.GetBondDir() == Chem.BondDir.ENDUPRIGHT else 0)
    feats.append(1 if bond.GetBondDir() == Chem.BondDir.ENDDOWNRIGHT else 0)

    while len(feats) < BOND_FEATURE_DIM:
        feats.append(0)
    return np.array(feats[:BOND_FEATURE_DIM], dtype=np.float32)


def mol_to_graph(mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert RDKit mol to directed-edge graph representation.

    Returns:
        atom_feats: (V, ATOM_FEATURE_DIM) float32
        bond_feats: (2E, BOND_FEATURE_DIM) float32
        edge_src:   (2E,) int64
        edge_dst:   (2E,) int64
    """
    V = mol.GetNumAtoms()
    if V == 0:
        return (
            np.zeros((0, ATOM_FEATURE_DIM), dtype=np.float32),
            np.zeros((0, BOND_FEATURE_DIM), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )

    atom_feats = np.stack([_atom_features(a) for a in mol.GetAtoms()], axis=0)

    edge_src: List[int] = []
    edge_dst: List[int] = []
    bond_feats_list: List[np.ndarray] = []

    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        bf = _bond_features(bond)
        edge_src.append(u)
        edge_dst.append(v)
        bond_feats_list.append(bf)
        edge_src.append(v)
        edge_dst.append(u)
        bond_feats_list.append(bf)

    if not edge_src:
        bond_feats = np.zeros((0, BOND_FEATURE_DIM), dtype=np.float32)
    else:
        bond_feats = np.stack(bond_feats_list, axis=0)

    return (
        atom_feats,
        bond_feats,
        np.array(edge_src, dtype=np.int64),
        np.array(edge_dst, dtype=np.int64),
    )


def featurize_smiles_list(smiles_list: List[str]) -> List[Optional[Tuple]]:
    """Featurize a list of SMILES into graph tuples."""
    results = []
    for smi in smiles_list:
        if not smi:
            results.append(None)
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            results.append(None)
            continue
        results.append(mol_to_graph(mol))
    return results


# ---------------------------------------------------------------------------
# Attentive D-MPNN model
# ---------------------------------------------------------------------------
class _GRUUpdate(nn.Module):
    """One-step GRU edge update: h_e' = GRU([sum_neighbors h_e, h_v_src])."""

    def __init__(self, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_z = nn.Linear(hidden_dim + hidden_dim, hidden_dim, bias=False)
        self.W_r = nn.Linear(hidden_dim + hidden_dim, hidden_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim + hidden_dim, hidden_dim, bias=False)
        self.b_z = nn.Parameter(torch.zeros(hidden_dim))
        self.b_r = nn.Parameter(torch.zeros(hidden_dim))
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))

    def forward(
        self,
        h_e: torch.Tensor,
        h_v_src: torch.Tensor,
        neighbor_h_e_sum: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([neighbor_h_e_sum, h_v_src], dim=-1)
        z = torch.sigmoid(self.W_z(combined) + self.b_z)
        r = torch.sigmoid(self.W_r(combined) + self.b_r)
        h_tilde = torch.tanh(
            self.W_h(torch.cat([r * neighbor_h_e_sum, h_v_src], dim=-1)) + self.b_h
        )
        h_new = (1 - z) * h_e + z * h_tilde
        return h_new


class AttentiveDMPNNModel(nn.Module):
    """Directed Message Passing Neural Network with attentive readout.

    Key difference from vanilla DMPNNModel:
    - Readout uses per-atom attention (softmax over MLP(h_v')) to weight atom contributions.
    - Output = sum_v a_v * h_v'  (attention-weighted sum of atom hidden states).
    """

    def __init__(
        self,
        atom_dim: int = ATOM_FEATURE_DIM,
        bond_dim: int = BOND_FEATURE_DIM,
        hidden: int = DMPNN_HIDDEN,
        depth: int = DMPNN_DEPTH,
        dropout: float = DMPNN_DROPOUT,
    ):
        super().__init__()
        self.hidden = hidden
        self.depth = depth

        self.atom_emb = nn.Linear(atom_dim, hidden)
        self.bond_emb = nn.Linear(bond_dim, hidden)

        self.gru_update = _GRUUpdate(hidden, hidden)

        self.edge_init = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
        )

        # Pre-readout projection: h_v' = ReLU(W * [h_v, outgoing_sum])
        self.readout_pre = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Attention MLP: produces unnormalized attention scores
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

        # Final prediction MLP applied after attention pooling
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

        self.dropout = nn.Dropout(dropout)

    def _forward_one_mol(
        self,
        atom_f: torch.Tensor,
        bond_f: torch.Tensor,
        e_src: torch.Tensor,
        e_dst: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for a single molecule. Returns scalar logit."""
        V = atom_f.size(0)
        E = bond_f.size(0)
        device = atom_f.device

        h_v = self.dropout(torch.relu(self.atom_emb(atom_f)))
        h_e = self.dropout(torch.relu(self.bond_emb(bond_f)))

        for _ in range(self.depth):
            atom_incoming = torch.zeros(V, self.hidden, device=device)
            atom_incoming.index_add_(0, e_dst, h_e)

            neighbor_sum = atom_incoming[e_src]

            h_e = self.gru_update(h_e, h_v[e_src], neighbor_sum)

        # Readout: sum of outgoing edge hidden per atom
        outgoing = torch.zeros(V, self.hidden, device=device)
        outgoing.index_add_(0, e_src, h_e)

        h_comb = torch.cat([h_v, outgoing], dim=-1)

        # Pre-readout projection
        h_v_prime = self.readout_pre(h_comb)  # (V, H)

        # Attention weights: a_v = softmax(MLP(h_v'))
        att_logits = self.attention_mlp(h_v_prime).squeeze(-1)  # (V,)
        att_weights = torch.softmax(att_logits, dim=0)          # (V,)

        # Attention-weighted sum
        pooled = (att_weights.unsqueeze(-1) * h_v_prime).sum(dim=0)  # (H,)

        return self.readout_mlp(pooled).squeeze(-1)

    def forward(
        self,
        atom_feats: torch.Tensor,
        bond_feats: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        batch_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-molecule forward; exclude padding from messages and attention.

        The local featurizer gives every real atom a nonzero element one-hot
        field; collate_graphs uses all-zero node padding and -1 edge padding.
        An explicit (B,V) boolean node mask can override that inference.
        Zero bond features do not mean padding (e.g. an unsupported bond type).
        """
        B = atom_feats.size(0)
        device = atom_feats.device
        if batch_mask is not None and (batch_mask.shape != atom_feats.shape[:2] or batch_mask.dtype != torch.bool):
            raise ValueError("batch_mask must be a (B,V) boolean node mask")
        logits = []
        for b in range(B):
            node_mask = batch_mask[b] if batch_mask is not None else atom_feats[b].abs().sum(-1).ne(0)
            node_indices = node_mask.nonzero(as_tuple=True)[0]
            if node_indices.numel() == 0:
                raise ValueError("A graph with no real atoms cannot receive a prediction")
            src, dst = edge_src[b], edge_dst[b]
            if ((src < 0) != (dst < 0)).any():
                raise ValueError("Both endpoints of padded edges must be negative")
            valid_edges = (src >= 0) & (dst >= 0)
            src, dst = src[valid_edges], dst[valid_edges]
            if (src >= atom_feats.size(1)).any() or (dst >= atom_feats.size(1)).any():
                raise ValueError("Edge endpoint outside the node tensor")
            mapping = torch.full((atom_feats.size(1),), -1, device=device, dtype=torch.long)
            mapping[node_indices] = torch.arange(node_indices.numel(), device=device)
            src, dst = mapping[src], mapping[dst]
            if (src < 0).any() or (dst < 0).any():
                raise ValueError("Real edges cannot refer to padded atoms")
            logit = self._forward_one_mol(
                atom_feats[b, node_indices],
                bond_feats[b, valid_edges],
                src,
                dst,
            )
            logits.append(logit)
        return torch.stack(logits, dim=0)


@dataclass
class AttentiveDMPNNResult:
    metal: str
    model: str
    test_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    per_cell_line: Dict[str, float] = field(default_factory=dict)
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    pos_rate_train: float = 0.0
    pos_rate_test: float = 0.0

    def to_json(self) -> str:
        return json.dumps(
            {
                "metal": self.metal,
                "model": self.model,
                "n_train": self.n_train,
                "n_val": self.n_val,
                "n_test": self.n_test,
                "pos_rate_train": self.pos_rate_train,
                "pos_rate_test": self.pos_rate_test,
                "test_metrics": self.test_metrics,
                "val_metrics": self.val_metrics,
                "per_cell_line_auc": self.per_cell_line,
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def collate_graphs(batch):
    """Collate a batch of graph tuples into padded tensors."""
    valid = [b for b in batch if b is not None]
    if not valid:
        return None

    atom_dim = valid[0][0].shape[1]
    bond_dim = valid[0][1].shape[1]

    max_V = max(g[0].shape[0] for g in valid)
    max_E = max(g[1].shape[0] for g in valid)

    B = len(valid)
    atom_feats = np.zeros((B, max_V, atom_dim), dtype=np.float32)
    bond_feats = np.zeros((B, max_E, bond_dim), dtype=np.float32)
    edge_src = np.full((B, max_E), -1, dtype=np.int64)
    edge_dst = np.full((B, max_E), -1, dtype=np.int64)

    for b, (af, bf, es, ed) in enumerate(valid):
        V = af.shape[0]
        E = bf.shape[0]
        atom_feats[b, :V] = af
        bond_feats[b, :E] = bf
        edge_src[b, :E] = es
        edge_dst[b, :E] = ed

    return (
        torch.from_numpy(atom_feats),
        torch.from_numpy(bond_feats),
        torch.from_numpy(edge_src),
        torch.from_numpy(edge_dst),
    )


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    graphs: List,
    labels: np.ndarray,
    batch_size: int = DMPNN_BATCH_SIZE,
    device: torch.device = torch.device("cpu"),
) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0

    indices = np.arange(len(graphs))
    np.random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch_graphs = [graphs[i] for i in batch_idx]
        batch_labels = labels[batch_idx]

        valid_graphs = [(g, l) for g, l in zip(batch_graphs, batch_labels) if g is not None]
        if not valid_graphs:
            continue

        g_list = [v[0] for v in valid_graphs]
        y_batch = np.array([v[1] for v in valid_graphs], dtype=np.float32)

        batch_data = collate_graphs(g_list)
        if batch_data is None:
            continue

        atom_f, bond_f, e_src, e_dst = batch_data
        atom_f = atom_f.to(device)
        bond_f = bond_f.to(device)
        e_src = e_src.to(device)
        e_dst = e_dst.to(device)
        y_batch = torch.from_numpy(y_batch).to(device)

        optimizer.zero_grad()
        logits = model(atom_f, bond_f, e_src, e_dst)
        loss = nn.BCEWithLogitsLoss()(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        n_samples += len(y_batch)

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def predict(
    model: nn.Module,
    graphs: List,
    labels: np.ndarray,
    batch_size: int = DMPNN_BATCH_SIZE,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_y = []
    all_score = []

    for start in range(0, len(graphs), batch_size):
        batch_graphs = graphs[start:start + batch_size]
        batch_labels = labels[start:start + batch_size]

        valid = [(g, l) for g, l in zip(batch_graphs, batch_labels) if g is not None]
        if not valid:
            for g, l in zip(batch_graphs, batch_labels):
                if g is None:
                    all_y.append(l)
                    all_score.append(0.0)
            continue

        g_list = [v[0] for v in valid]
        y_batch = np.array([v[1] for v in valid], dtype=np.float32)

        batch_data = collate_graphs(g_list)
        if batch_data is None:
            continue

        atom_f, bond_f, e_src, e_dst = batch_data
        atom_f = atom_f.to(device)
        bond_f = bond_f.to(device)
        e_src = e_src.to(device)
        e_dst = e_dst.to(device)

        logits = model(atom_f, bond_f, e_src, e_dst)
        probs = torch.sigmoid(logits).cpu().numpy()

        vi = 0
        for g, l in zip(batch_graphs, batch_labels):
            if g is not None:
                all_y.append(y_batch[vi])
                all_score.append(probs[vi])
                vi += 1
            else:
                all_y.append(l)
                all_score.append(0.0)

    return np.array(all_y, dtype=float), np.array(all_score, dtype=float)


# ---------------------------------------------------------------------------
# Baseline class
# ---------------------------------------------------------------------------
SPLITTER_FACTORIES: Dict[str, callable] = {
    "random": lambda seed: RandomSplitter(seed=seed),
    "ligand_dedup": lambda seed: LigandDeduplicatedSplitter(
        strategy="largest_first", seed=seed
    ),
    "scaffold": lambda seed: ScaffoldSplitter(
        strategy="largest_first", seed=seed
    ),
    "temporal": lambda seed: TemporalSplitter(cutoff_year=2024),
    "chemical": lambda seed: ChemicalSplitter(threshold=0.7, seed=seed),
}


class AttentiveDMPNNBaseline:
    """Attentive D-MPNN classifier for cytotoxicity."""

    def __init__(
        self,
        metal: str = "Ru",
        hidden: int = DMPNN_HIDDEN,
        depth: int = DMPNN_DEPTH,
        dropout: float = DMPNN_DROPOUT,
        lr: float = DMPNN_LR,
        epochs: int = DMPNN_EPOCHS,
        batch_size: int = DMPNN_BATCH_SIZE,
        seed: int = 42,
        splitter: Optional[callable] = None,
        device: Optional[torch.device] = None,
    ):
        self.metal = metal
        self.hidden = hidden
        self.depth = depth
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.splitter = splitter
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model: Optional[AttentiveDMPNNModel] = None
        self._train_idx: Optional[np.ndarray] = None
        self._val_idx: Optional[np.ndarray] = None
        self._test_idx: Optional[np.ndarray] = None

    def _load_dataset(self) -> MetalCytotoxDataset:
        flt = CytotoxFilter(
            time_threshold=24.0,
            ic50_min=0.01,
            metal_whitelist=[self.metal],
            compute_pic50=True,
            compute_active=True,
        )
        ds = MetalCytotoxDataset.from_csv(filters=flt)
        if len(ds) == 0:
            raise RuntimeError(f"No rows for metal={self.metal!r}")
        return ds

    def _split(
        self,
        dataset: MetalCytotoxDataset,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.splitter is not None:
            result = self.splitter(dataset)
            return result.train_idx, result.val_idx, result.test_idx
        from sklearn.model_selection import train_test_split
        y = np.asarray(dataset.active).astype(int)
        idx = np.arange(len(y))
        idx_train, idx_tmp, _, _ = train_test_split(
            idx, y, test_size=0.2, stratify=y, random_state=self.seed
        )
        idx_val, idx_test, _, _ = train_test_split(
            idx_tmp, np.asarray(y[idx_tmp]).astype(int),
            test_size=0.5, stratify=np.asarray(y[idx_tmp]).astype(int),
            random_state=self.seed,
        )
        return idx_train, idx_val, idx_test

    def fit(
        self,
        smiles: np.ndarray,
        y_all: np.ndarray,
        cell_lines: np.ndarray,
        idx_train: np.ndarray,
        idx_val: np.ndarray,
        verbose: bool = True,
    ) -> "AttentiveDMPNNBaseline":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if verbose:
            print(f"[AttentiveDMPNN] Featurizing {len(smiles)} molecules...", flush=True)
        graphs_all = featurize_smiles_list([str(s) for s in smiles])
        graphs_train = [graphs_all[i] for i in idx_train]
        graphs_val = [graphs_all[i] for i in idx_val]
        y_train = y_all[idx_train]
        y_val = y_all[idx_val]

        self.model = AttentiveDMPNNModel(
            atom_dim=ATOM_FEATURE_DIM,
            bond_dim=BOND_FEATURE_DIM,
            hidden=self.hidden,
            depth=self.depth,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_auc = 0.0
        best_state: Optional[Dict] = None

        for epoch in range(self.epochs):
            t0 = time.time()
            train_loss = train_epoch(
                self.model, optimizer, graphs_train, y_train,
                batch_size=self.batch_size, device=self.device,
            )
            y_val_true, y_val_score = predict(
                self.model, graphs_val, y_val,
                batch_size=self.batch_size, device=self.device,
            )
            val_metrics = compute_metrics(y_val_true, y_val_score)
            elapsed = time.time() - t0

            if verbose and (epoch % 5 == 0 or epoch == self.epochs - 1):
                print(
                    f"[AttentiveDMPNN epoch {epoch:3d}] loss={train_loss:.4f} "
                    f"val_auc={val_metrics['roc_auc']:.4f} "
                    f"val_ap={val_metrics['pr_auc']:.4f} "
                    f"time={elapsed:.1f}s",
                    flush=True,
                )

            if val_metrics["roc_auc"] > best_val_auc:
                best_val_auc = val_metrics["roc_auc"]
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }

        if best_state is not None:
            self.model.load_state_dict(best_state)

        if verbose:
            print(f"[AttentiveDMPNN] Best val AUC: {best_val_auc:.4f}")
        return self

    def predict(self, smiles: np.ndarray, idx_test: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        graphs_all = featurize_smiles_list([str(s) for s in smiles])
        graphs_test = [graphs_all[i] for i in idx_test]
        y_dummy = np.zeros(len(idx_test))
        _, scores = predict(
            self.model, graphs_test, y_dummy,
            batch_size=self.batch_size, device=self.device,
        )
        return scores

    def score(
        self,
        smiles: np.ndarray,
        y_all: np.ndarray,
        cell_lines: np.ndarray,
        idx_test: np.ndarray,
    ) -> Dict[str, float]:
        test_score = self.predict(smiles, idx_test)
        y_test = y_all[idx_test]
        return compute_metrics(y_test, test_score)

    def run(self, verbose: bool = True) -> AttentiveDMPNNResult:
        ds = self._load_dataset()
        smiles = ds.smiles
        y_all = ds.active.astype(int)
        cell_lines = ds.df["Cell_line"].astype(str).to_numpy()

        keep = np.array([bool(s) for s in smiles], dtype=bool)
        smiles = smiles[keep]
        y_all = y_all[keep]
        cell_lines = cell_lines[keep]

        idx_train, idx_val, idx_test = self._split(ds)
        self._train_idx = idx_train
        self._val_idx = idx_val
        self._test_idx = idx_test

        self.fit(smiles, y_all, cell_lines, idx_train, idx_val, verbose=verbose)

        test_score = self.predict(smiles, idx_test)
        y_test = y_all[idx_test]
        y_train = y_all[idx_train]
        y_val = y_all[idx_val]
        cl_test = cell_lines[idx_test]

        test_metrics = compute_metrics(y_test, test_score)
        train_score = self.predict(smiles, idx_train)
        val_score = self.predict(smiles, idx_val)
        train_metrics = compute_metrics(y_train, train_score)
        val_metrics = compute_metrics(y_val, val_score)
        per_cl = per_cell_line_auc(cl_test, y_test, test_score)

        return AttentiveDMPNNResult(
            metal=self.metal,
            model="dmpnn_attn",
            test_metrics=test_metrics,
            val_metrics=val_metrics,
            per_cell_line=per_cl,
            n_train=int(len(idx_train)),
            n_val=int(len(idx_val)),
            n_test=int(len(idx_test)),
            pos_rate_train=float(y_train.mean()),
            pos_rate_test=float(y_test.mean()),
        )


def quick_smoke(metal: str = "Ru", max_rows: int = 200) -> AttentiveDMPNNResult:
    """Quick training on small subset for tests."""
    flt = CytotoxFilter(
        time_threshold=24.0,
        ic50_min=0.01,
        metal_whitelist=[metal],
    )
    ds = MetalCytotoxDataset.from_csv(filters=flt)
    df = ds.df.head(max_rows).copy()
    ds_small = MetalCytotoxDataset(df=df)
    baseline = AttentiveDMPNNBaseline(metal=metal, epochs=10)
    baseline._load_dataset = lambda: ds_small  # type: ignore[assignment]
    return baseline.run(verbose=False)


__all__ = [
    "AttentiveDMPNNModel",
    "AttentiveDMPNNBaseline",
    "AttentiveDMPNNResult",
    "SPLITTER_FACTORIES",
    "featurize_smiles_list",
    "mol_to_graph",
    "ATOM_FEATURE_DIM",
    "BOND_FEATURE_DIM",
    "DMPNN_HIDDEN",
    "DMPNN_DEPTH",
    "DMPNN_DROPOUT",
    "DMPNN_LR",
    "DMPNN_EPOCHS",
    "quick_smoke",
]
