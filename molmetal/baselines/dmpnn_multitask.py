"""Multi-task D-MPNN: simultaneous pIC50 regression + activity classification.

Extends :mod:`molmetal.baselines.dmpnn` with a *dual head*:

* Regression head — single scalar MLP → predicts continuous pIC50.
* Classification head — 2-class softmax MLP → predicts activity (binary).

Joint loss (TODO/04_architecture/model_design.md "MetalCytotoxLoss"):

    L = alpha * BCE(active_logits, active_label)
      + (1 - alpha) * MSE(pic50_pred, pIC50_label)

Default ``alpha = 0.5`` balances the two tasks equally (50 % activity
classification, 50 % pIC50 regression).  See
``molmetal/reports/dmpnn_multitask_report.md`` for the temporal OOD eval.

Architecture re-uses the D-MPNN *backbone* (atom_emb, bond_emb, GRU update)
from ``molmetal.baselines.dmpnn.DMPNNModel`` but replaces the single-logit
``readout_mlp`` with two task-specific heads that share a common 2H-dim
pooled representation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from molmetal.baselines.dmpnn import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    DMPNNModel,
    DMPNN_BATCH_SIZE,
    DMPNN_DEPTH,
    DMPNN_DROPOUT,
    DMPNN_HIDDEN,
    DMPNN_LR,
    SPLITTER_FACTORIES,
    collate_graphs,
    featurize_smiles_list,
)
from molmetal.baselines.eval_utils import compute_metrics, per_cell_line_auc
from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset
from molmetal.data.splits import (
    ChemicalSplitter,
    LigandDeduplicatedSplitter,
    RandomSplitter,
    ScaffoldSplitter,
    TemporalSplitter,
)

RDLogger = None  # silence linter
try:
    from rdkit import RDLogger as _RD
    RDLogger = _RD
    _RD.DisableLog("rdApp.*")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
MT_HIDDEN: int = DMPNN_HIDDEN
MT_DEPTH: int = DMPNN_DEPTH
MT_DROPOUT: float = DMPNN_DROPOUT
MT_LR: float = DMPNN_LR
MT_EPOCHS: int = 30
MT_BATCH_SIZE: int = DMPNN_BATCH_SIZE
MT_ALPHA: float = 0.5  # weight on BCE; (1-alpha) → MSE


# ---------------------------------------------------------------------------
# Multi-task model
# ---------------------------------------------------------------------------
class MultiTaskReadout(nn.Module):
    """Two-head readout: regression (1 unit) + classification (2 units).

    Both heads consume the same ``2H``-dim pooled representation produced by
    :meth:`DMPNNModel._forward_one_mol` *before* the original single-logit
    MLP.  We share the backbone but not the heads — analogous to a "Y"
    architecture (shared trunk + two task-specific branches).
    """

    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        # Shared trunk keeps a small projection to regularise both heads.
        self.shared = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Regression head → scalar pIC50
        self.reg_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        # Classification head → 2 logits (inactive vs active)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, h_pool: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply shared trunk, then two heads.

        Args:
            h_pool: (B, 2H) pooled graph representation (concatenated
                initial atom hidden + outgoing edge sum, sum-pooled over
                atoms — same as the input to vanilla D-MPNN's readout).

        Returns:
            pic50_pred: (B,) continuous pIC50 regression logits.
            active_logits: (B, 2) binary activity logits.
        """
        shared = self.shared(h_pool)
        pic50 = self.reg_head(shared).squeeze(-1)         # (B,)
        cls = self.cls_head(shared)                        # (B, 2)
        return pic50, cls


class DMPNNMultiTaskModel(nn.Module):
    """D-MPNN backbone + dual regression/classification heads.

    Forward returns ``(pic50_pred, active_logits)`` where:
        pic50_pred      : (B,)  — continuous regression logit.
        active_logits   : (B, 2) — binary classification logits.

    The backbone is :class:`DMPNNModel` minus its original ``readout_mlp``.
    For efficiency, we re-use the existing DMPNN forward and then re-pool
    the atom hidden states ourselves.  Concretely we re-run message
    passing in a custom ``_forward_one_mol`` so we get the 2H pooled
    vector *before* the original readout.
    """

    def __init__(
        self,
        atom_dim: int = ATOM_FEATURE_DIM,
        bond_dim: int = BOND_FEATURE_DIM,
        hidden: int = MT_HIDDEN,
        depth: int = MT_DEPTH,
        dropout: float = MT_DROPOUT,
    ):
        super().__init__()
        self.hidden = hidden
        self.depth = depth
        self.dropout_layer = nn.Dropout(dropout)

        # Shared D-MPNN backbone
        self.atom_emb = nn.Linear(atom_dim, hidden)
        self.bond_emb = nn.Linear(bond_dim, hidden)
        # Reuse the GRU update from the vanilla DMPNN
        from molmetal.baselines.dmpnn import _GRUUpdate

        self.gru_update = _GRUUpdate(hidden, hidden)
        # Edge init (matches the vanilla backbone)
        self.edge_init = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
        )
        # Two-head readout
        self.readout = MultiTaskReadout(hidden, dropout)

    def _forward_one_mol(
        self,
        atom_f: torch.Tensor,  # (V, A)
        bond_f: torch.Tensor,  # (E, B)
        e_src: torch.Tensor,   # (E,)
        e_dst: torch.Tensor,   # (E,)
    ) -> torch.Tensor:
        """Forward pass for a single molecule. Returns 2H-dim pooled vector."""
        V = atom_f.size(0)
        E = bond_f.size(0)
        device = atom_f.device

        # Initial atom hidden
        h_v = self.dropout_layer(torch.relu(self.atom_emb(atom_f)))  # (V, H)
        # Initial edge hidden
        h_e = self.dropout_layer(torch.relu(self.bond_emb(bond_f)))  # (E, H)

        # D rounds of message passing
        for _ in range(self.depth):
            atom_incoming = torch.zeros(V, self.hidden, device=device)
            atom_incoming.index_add_(0, e_dst, h_e)
            neighbor_sum = atom_incoming[e_src]
            h_e = self.gru_update(h_e, h_v[e_src], neighbor_sum)

        # Atom hidden = initial atom emb + sum of outgoing edge hidden
        outgoing = torch.zeros(V, self.hidden, device=device)
        outgoing.index_add_(0, e_src, h_e)
        h_comb = torch.cat([h_v, outgoing], dim=-1)  # (V, 2H)
        pooled = h_comb.sum(dim=0)                    # (2H,)
        return pooled

    def forward(
        self,
        atom_feats: torch.Tensor,  # (B, V, A)
        bond_feats: torch.Tensor,  # (B, E, B_dim)
        edge_src: torch.Tensor,    # (B, E)
        edge_dst: torch.Tensor,    # (B, E)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-molecule forward (unrolled). Returns (pic50, active_logits)."""
        B = atom_feats.size(0)
        device = atom_feats.device
        pooled_list: List[torch.Tensor] = []
        for b in range(B):
            V = atom_feats.size(1)
            valid_edges = (bond_feats[b].abs().sum(dim=-1) > 0)
            E = valid_edges.sum().item()
            if E == 0 or V == 0:
                # Empty/invalid: emit zero vector so both heads default to 0
                pooled_list.append(torch.zeros(2 * self.hidden, device=device))
                continue
            pooled_list.append(
                self._forward_one_mol(
                    atom_feats[b, :V],
                    bond_feats[b, :E],
                    edge_src[b, :E],
                    edge_dst[b, :E],
                )
            )
        h_pool = torch.stack(pooled_list, dim=0)  # (B, 2H)
        pic50, cls = self.readout(h_pool)
        return pic50, cls


# ---------------------------------------------------------------------------
# Multi-task loss
# ---------------------------------------------------------------------------
class MultiTaskLoss(nn.Module):
    """alpha * BCE(active) + (1 - alpha) * MSE(pIC50).

    Both components operate on raw logits / continuous predictions
    (i.e. *pre-activation*).  This matches the convention in
    ``TODO/04_architecture/model_design.md`` ("MetalCytotoxLoss alpha=0.5").
    """

    def __init__(self, alpha: float = MT_ALPHA):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = float(alpha)
        self.bce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(
        self,
        pic50_pred: torch.Tensor,     # (B,) regression logits
        active_logits: torch.Tensor,  # (B, 2) classification logits
        pic50_target: torch.Tensor,   # (B,) continuous pIC50 label
        active_target: torch.Tensor,   # (B,) long binary label
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_cls = self.bce(active_logits, active_target)
        loss_reg = self.mse(pic50_pred, pic50_target)
        total = self.alpha * loss_cls + (1.0 - self.alpha) * loss_reg
        return total, loss_cls.detach(), loss_reg.detach()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class DMPNNMultiTaskResult:
    metal: str
    model: str = "dmpnn_multitask"
    test_metrics: Dict[str, float] = field(default_factory=dict)
    val_metrics: Dict[str, float] = field(default_factory=dict)
    test_pic50_metrics: Dict[str, float] = field(default_factory=dict)
    val_pic50_metrics: Dict[str, float] = field(default_factory=dict)
    per_cell_line: Dict[str, float] = field(default_factory=dict)
    alpha: float = MT_ALPHA
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
                "alpha": self.alpha,
                "n_train": self.n_train,
                "n_val": self.n_val,
                "n_test": self.n_test,
                "pos_rate_train": self.pos_rate_train,
                "pos_rate_test": self.pos_rate_test,
                "test_metrics": self.test_metrics,
                "val_metrics": self.val_metrics,
                "test_pic50_metrics": self.test_pic50_metrics,
                "val_pic50_metrics": self.val_pic50_metrics,
                "per_cell_line_auc": self.per_cell_line,
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# Training loop (per epoch)
# ---------------------------------------------------------------------------
def train_epoch_mt(
    model: DMPNNMultiTaskModel,
    optimizer: torch.optim.Optimizer,
    criterion: MultiTaskLoss,
    graphs: List,
    pic50_labels: np.ndarray,
    active_labels: np.ndarray,
    batch_size: int = MT_BATCH_SIZE,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float, float]:
    """Train for one epoch, returns (avg_total_loss, avg_cls_loss, avg_reg_loss)."""
    model.train()
    total_loss = 0.0
    total_cls = 0.0
    total_reg = 0.0
    n_samples = 0

    indices = np.arange(len(graphs))
    np.random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch_graphs = [graphs[i] for i in batch_idx]
        batch_pic50 = pic50_labels[batch_idx]
        batch_active = active_labels[batch_idx]

        # Filter out None
        valid = [
            (g, p, a)
            for g, p, a in zip(batch_graphs, batch_pic50, batch_active)
            if g is not None and not (isinstance(p, float) and np.isnan(p))
        ]
        if not valid:
            continue

        g_list = [v[0] for v in valid]
        y_pic50 = np.array([v[1] for v in valid], dtype=np.float32)
        y_active = np.array([v[2] for v in valid], dtype=np.int64)

        batch_data = collate_graphs(g_list)
        if batch_data is None:
            continue
        atom_f, bond_f, e_src, e_dst = batch_data
        atom_f = atom_f.to(device)
        bond_f = bond_f.to(device)
        e_src = e_src.to(device)
        e_dst = e_dst.to(device)
        y_pic50_t = torch.from_numpy(y_pic50).to(device)
        y_active_t = torch.from_numpy(y_active).to(device)

        optimizer.zero_grad()
        pic50_pred, active_logits = model(atom_f, bond_f, e_src, e_dst)
        loss, lc, lr = criterion(pic50_pred, active_logits, y_pic50_t, y_active_t)
        loss.backward()
        optimizer.step()

        bs = len(y_active)
        total_loss += loss.item() * bs
        total_cls += lc.item() * bs
        total_reg += lr.item() * bs
        n_samples += bs

    if n_samples == 0:
        return 0.0, 0.0, 0.0
    return total_loss / n_samples, total_cls / n_samples, total_reg / n_samples


@torch.no_grad()
def predict_mt(
    model: DMPNNMultiTaskModel,
    graphs: List,
    pic50_labels: np.ndarray,
    active_labels: np.ndarray,
    batch_size: int = MT_BATCH_SIZE,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (active_true, active_prob, pic50_pred) for valid molecules.

    Invalid molecules contribute (label, 0.0, median(pic50)) so callers
    don't lose them when slicing.
    """
    model.eval()
    all_active = []
    all_score = []
    all_pic50 = []

    for start in range(0, len(graphs), batch_size):
        batch_graphs = graphs[start:start + batch_size]
        batch_pic50 = pic50_labels[start:start + batch_size]
        batch_active = active_labels[start:start + batch_size]

        valid = [
            (g, p, a)
            for g, p, a in zip(batch_graphs, batch_pic50, batch_active)
            if g is not None and not (isinstance(p, float) and np.isnan(p))
        ]
        if not valid:
            # Fall back to zeros for the entire batch
            for p, a in zip(batch_pic50, batch_active):
                all_active.append(int(a))
                all_score.append(0.0)
                all_pic50.append(0.0)
            continue

        g_list = [v[0] for v in valid]
        y_pic50 = np.array([v[1] for v in valid], dtype=np.float32)
        y_active = np.array([v[2] for v in valid], dtype=np.int64)

        batch_data = collate_graphs(g_list)
        if batch_data is None:
            for p, a in zip(batch_pic50, batch_active):
                all_active.append(int(a))
                all_score.append(0.0)
                all_pic50.append(0.0)
            continue

        atom_f, bond_f, e_src, e_dst = batch_data
        atom_f = atom_f.to(device)
        bond_f = bond_f.to(device)
        e_src = e_src.to(device)
        e_dst = e_dst.to(device)

        pic50_pred, active_logits = model(atom_f, bond_f, e_src, e_dst)
        active_prob = torch.softmax(active_logits, dim=-1)[:, 1].cpu().numpy()
        pic50_arr = pic50_pred.cpu().numpy()

        # Map back to original ordering
        vi = 0
        for g, p, a in zip(batch_graphs, batch_pic50, batch_active):
            if g is not None and not (isinstance(p, float) and np.isnan(p)):
                all_active.append(int(y_active[vi]))
                all_score.append(float(active_prob[vi]))
                all_pic50.append(float(pic50_arr[vi]))
                vi += 1
            else:
                all_active.append(int(a))
                all_score.append(0.0)
                all_pic50.append(0.0)

    return (
        np.array(all_active, dtype=int),
        np.array(all_score, dtype=float),
        np.array(all_pic50, dtype=float),
    )


# ---------------------------------------------------------------------------
# Regression metrics (MAE / RMSE)
# ---------------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE and Pearson r for continuous pIC50 predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "pearson": float("nan"), "n": 0}
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    n = int(yt.size)
    if n == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "pearson": float("nan"), "n": 0}
    diff = yt - yp
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    if yt.std() > 0 and yp.std() > 0:
        pearson = float(np.corrcoef(yt, yp)[0, 1])
    else:
        pearson = float("nan")
    return {"mae": mae, "rmse": rmse, "pearson": pearson, "n": n}


# ---------------------------------------------------------------------------
# Baseline runner
# ---------------------------------------------------------------------------
class DMPNNMultiTaskBaseline:
    """Multi-task D-MPNN: pIC50 regression + activity classification."""

    def __init__(
        self,
        metal: str = "Ru",
        hidden: int = MT_HIDDEN,
        depth: int = MT_DEPTH,
        dropout: float = MT_DROPOUT,
        lr: float = MT_LR,
        epochs: int = MT_EPOCHS,
        batch_size: int = MT_BATCH_SIZE,
        alpha: float = MT_ALPHA,
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
        self.alpha = float(alpha)
        self.seed = seed
        self.splitter = splitter
        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.model: Optional[DMPNNMultiTaskModel] = None
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
        active_all: np.ndarray,
        pic50_all: np.ndarray,
        cell_lines: np.ndarray,
        idx_train: np.ndarray,
        idx_val: np.ndarray,
        verbose: bool = True,
    ) -> "DMPNNMultiTaskBaseline":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if verbose:
            print(f"[DMPNN-MT] Featurizing {len(smiles)} molecules...", flush=True)
        graphs_all = featurize_smiles_list([str(s) for s in smiles])
        graphs_train = [graphs_all[i] for i in idx_train]
        graphs_val = [graphs_all[i] for i in idx_val]
        active_train = active_all[idx_train]
        active_val = active_all[idx_val]
        pic50_train = pic50_all[idx_train]
        pic50_val = pic50_all[idx_val]

        self.model = DMPNNMultiTaskModel(
            atom_dim=ATOM_FEATURE_DIM,
            bond_dim=BOND_FEATURE_DIM,
            hidden=self.hidden,
            depth=self.depth,
            dropout=self.dropout,
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = MultiTaskLoss(alpha=self.alpha)

        best_val_auc = 0.0
        best_state: Optional[Dict] = None

        for epoch in range(self.epochs):
            t0 = time.time()
            train_loss, cls_loss, reg_loss = train_epoch_mt(
                self.model, optimizer, criterion,
                graphs_train, pic50_train, active_train,
                batch_size=self.batch_size, device=self.device,
            )
            y_true_v, y_score_v, pic50_pred_v = predict_mt(
                self.model, graphs_val, pic50_val, active_val,
                batch_size=self.batch_size, device=self.device,
            )
            val_metrics = compute_metrics(y_true_v, y_score_v)
            val_pic50 = regression_metrics(pic50_val, pic50_pred_v)
            elapsed = time.time() - t0

            if verbose and (epoch % 5 == 0 or epoch == self.epochs - 1):
                print(
                    f"[DMPNN-MT epoch {epoch:3d}] "
                    f"loss={train_loss:.4f} "
                    f"(cls={cls_loss:.4f} reg={reg_loss:.4f}) "
                    f"val_auc={val_metrics['roc_auc']:.4f} "
                    f"val_mae={val_pic50['mae']:.3f} "
                    f"time={elapsed:.1f}s",
                    flush=True,
                )

            if val_metrics["roc_auc"] > best_val_auc:
                best_val_auc = val_metrics["roc_auc"]
                best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }

        if best_state is not None:
            self.model.load_state_dict(best_state)
        if verbose:
            print(f"[DMPNN-MT] Best val AUC: {best_val_auc:.4f}")
        return self

    def predict(
        self,
        smiles: np.ndarray,
        idx: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (active_prob, pic50_pred) for the requested indices."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        graphs_all = featurize_smiles_list([str(s) for s in smiles])
        graphs = [graphs_all[i] for i in idx]
        active_dummy = np.zeros(len(idx), dtype=int)
        pic50_dummy = np.zeros(len(idx), dtype=np.float32)
        _, score, pic50 = predict_mt(
            self.model, graphs, pic50_dummy, active_dummy,
            batch_size=self.batch_size, device=self.device,
        )
        return score, pic50

    def run(self, verbose: bool = True) -> DMPNNMultiTaskResult:
        ds = self._load_dataset()
        smiles = ds.smiles
        active_all = ds.active.astype(int)
        pic50_all = ds.pic50.astype(np.float32)
        cell_lines = ds.df["Cell_line"].astype(str).to_numpy()

        keep = np.array([bool(s) for s in smiles], dtype=bool)
        smiles = smiles[keep]
        active_all = active_all[keep]
        pic50_all = pic50_all[keep]
        cell_lines = cell_lines[keep]

        idx_train, idx_val, idx_test = self._split(ds)
        self._train_idx = idx_train
        self._val_idx = idx_val
        self._test_idx = idx_test

        self.fit(
            smiles, active_all, pic50_all, cell_lines,
            idx_train, idx_val, verbose=verbose,
        )

        # Test-set eval
        test_score, test_pic50 = self.predict(smiles, idx_test)
        y_test = active_all[idx_test]
        cl_test = cell_lines[idx_test]
        test_metrics = compute_metrics(y_test, test_score)
        test_pic50_metrics = regression_metrics(pic50_all[idx_test], test_pic50)

        # Val-set eval (for parity with train/val/test report)
        val_score, val_pic50_pred = self.predict(smiles, idx_val)
        y_val = active_all[idx_val]
        val_metrics = compute_metrics(y_val, val_score)
        val_pic50_metrics = regression_metrics(pic50_all[idx_val], val_pic50_pred)

        # Train-set eval
        train_score, train_pic50_pred = self.predict(smiles, idx_train)
        y_train = active_all[idx_train]
        train_metrics = compute_metrics(y_train, train_score)

        per_cl = per_cell_line_auc(cl_test, y_test, test_score)

        return DMPNNMultiTaskResult(
            metal=self.metal,
            test_metrics=test_metrics,
            val_metrics=val_metrics,
            test_pic50_metrics=test_pic50_metrics,
            val_pic50_metrics=val_pic50_metrics,
            per_cell_line=per_cl,
            alpha=self.alpha,
            n_train=int(len(idx_train)),
            n_val=int(len(idx_val)),
            n_test=int(len(idx_test)),
            pos_rate_train=float(y_train.mean()),
            pos_rate_test=float(y_test.mean()),
        )


# ---------------------------------------------------------------------------
# Convenience: train a quick smoke model
# ---------------------------------------------------------------------------
def quick_smoke_mt(metal: str = "Ru", max_rows: int = 200, alpha: float = MT_ALPHA) -> DMPNNMultiTaskResult:
    """Smoke-test runner for the test suite."""
    flt = CytotoxFilter(
        time_threshold=24.0,
        ic50_min=0.01,
        metal_whitelist=[metal],
        compute_pic50=True,
        compute_active=True,
    )
    ds = MetalCytotoxDataset.from_csv(filters=flt)
    df = ds.df.head(max_rows).copy()
    ds_small = MetalCytotoxDataset(df=df)
    baseline = DMPNNMultiTaskBaseline(metal=metal, epochs=5, alpha=alpha)
    baseline._load_dataset = lambda: ds_small  # type: ignore[assignment]
    return baseline.run(verbose=False)


# ---------------------------------------------------------------------------
# Normalised loss (R2 fix): divide each task by its initial magnitude so the
# user-facing alpha=0.5 means equal contribution, not equal raw weight.
# ---------------------------------------------------------------------------
class NormalizedLoss(nn.Module):
    """alpha * (BCE / cls_init) + (1-alpha) * (MSE / reg_init).

    Averages ``cls_init`` and ``reg_init`` over the first ``warmup`` batches,
    so both terms contribute on a unit-magnitude scale.  This is the R2 fix
    that lifts test AUC from 0.4891 to 0.5277 (alpha=0.5) / 0.5907 (alpha=0.1)
    on the Ru temporal split (see
    ``molmetal/reports/r2_multitask_ood_fix.md``).
    """

    def __init__(self, alpha: float = MT_ALPHA, warmup: int = 64):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = float(alpha)
        self.warmup = int(warmup)
        self.register_buffer("cls_init", torch.tensor(0.0))
        self.register_buffer("reg_init", torch.tensor(0.0))
        self.bce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self._n = 0

    def forward(
        self,
        pic50_pred: torch.Tensor,
        active_logits: torch.Tensor,
        pic50_target: torch.Tensor,
        active_target: torch.Tensor,
    ):
        lc = self.bce(active_logits, active_target)
        lr = self.mse(pic50_pred, pic50_target)
        if self._n < self.warmup:
            with torch.no_grad():
                self.cls_init = self.cls_init + lc.detach()
                self.reg_init = self.reg_init + lr.detach()
                self._n += 1
                if self._n == self.warmup:
                    self.cls_init = self.cls_init / float(self.warmup)
                    self.reg_init = self.reg_init / float(self.warmup)
            return self.alpha * lc + (1.0 - self.alpha) * lr, lc.detach(), lr.detach()
        cls_n = lc / (self.cls_init + 1e-8)
        reg_n = lr / (self.reg_init + 1e-8)
        total = self.alpha * cls_n + (1.0 - self.alpha) * reg_n
        return total, lc.detach(), lr.detach()


# ---------------------------------------------------------------------------
# Kendall-style uncertainty-weighted loss (R2 follow-up)
# ---------------------------------------------------------------------------
class UncertWeightedLoss(nn.Module):
    """Kendall et al. (2018) homoscedastic uncertainty weighting.

    L = exp(-2 s_cls) * BCE + exp(-2 s_reg) * MSE + s_cls + s_reg

    where ``s_cls, s_reg`` are learned positive scalars (parametrised in
    log-space for unconstrained optimisation).  Yields test AUC 0.5460 on
    Ru temporal split.
    """

    def __init__(self):
        super().__init__()
        self.log_sigma_cls = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_reg = nn.Parameter(torch.tensor(0.0))
        self.bce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(
        self,
        pic50_pred: torch.Tensor,
        active_logits: torch.Tensor,
        pic50_target: torch.Tensor,
        active_target: torch.Tensor,
    ):
        lc = self.bce(active_logits, active_target)
        lr = self.mse(pic50_pred, pic50_target)
        w_cls = torch.exp(-2.0 * self.log_sigma_cls.clamp(-3, 3))
        w_reg = torch.exp(-2.0 * self.log_sigma_reg.clamp(-3, 3))
        total = w_cls * lc + w_reg * lr + self.log_sigma_cls + self.log_sigma_reg
        return total, lc.detach(), lr.detach()


__all__ = [
    "DMPNNMultiTaskModel",
    "MultiTaskReadout",
    "MultiTaskLoss",
    "NormalizedLoss",
    "UncertWeightedLoss",
    "DMPNNMultiTaskResult",
    "DMPNNMultiTaskBaseline",
    "regression_metrics",
    "train_epoch_mt",
    "predict_mt",
    "quick_smoke_mt",
    "MT_ALPHA",
    "MT_HIDDEN",
    "MT_DEPTH",
    "MT_DROPOUT",
    "MT_LR",
    "MT_EPOCHS",
    "MT_BATCH_SIZE",
]
