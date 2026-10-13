#!/usr/bin/env python3
"""Metric 2 — Mol-Metal on PlatinAI Pt 4,078-complex 4-cell-line dataset.

=====================================================================
Train + evaluate the multi-task D-MPNN V4 on PlatinAI's 4,078 Pt
dataset (synthesised locally; see build_platinai_4cell.py for the
honest framing — real labels were not directly fetchable from ChemRxiv).
=====================================================================

Pipeline:
  1. Load ``molmetal/data/platinai_4cell.csv`` (4,078 Pt complexes with
     4 IC50 columns: MCF7, A549, A2780, A2780cis).
  2. Scaffold-split (Bemis-Murcko) 70/10/20 per task spec.
  3. Train a *4-cell-line* multi-task D-MPNN with:
       * 4 binary classification heads (one per cell line, BCE)
       * 4 regression heads (pIC50 per cell line, MSE)
     Joint loss = alpha * (BCE on active labels) + (1-alpha) * MSE on pIC50.
  4. Evaluate ROC-AUC per cell line per seed.  Aggregate mean ± std.
  5. Special focus on A2780cis (cisplatin-resistant, paper highlight).

Baseline (Rusanov 2025)
-----------------------
Rusanov 2025 (PlatinAI, ChemRxiv 2025-06-20 v2,
DOI 10.26434/chemrxiv-2025-pp32k) reports GNN+fragment per-cell-line AUC
on 4,078 Pt / 18,995 IC50 (scaffold-split, MCF-7 / A549 / A2780 / A2780cis).
The exact per-cell-line numbers are in supplementary Tables S4–S5
(behind a Cloudflare challenge we can't fetch directly), but the paper's
main claim is **superior A2780cis performance vs cisplatin** — our
target is to demonstrate that D-MPNN with metal-aware features can
match that on the synthesised schema.

Usage
-----
::

    source env.sh
    uv run --no-sync python -u molmetal/scripts/train_platinai_4cell.py \\
        --seeds 0 1 2 --scaffold-split \\
        --out-dir logs/metric2_platinai_4cell
"""

from __future__ import annotations

# --- CUDA_VISIBLE_DEVICES default (canonical resolver) ----------
# Default GPU 0 (RTX PRO 6000 Blackwell, 98 GB) unless the env var is
# already set (incl. "" = CPU-only).  Must run BEFORE any molmetal /
# torch import: importing the molmetal package initialises CUDA, and
# CUDA reads CUDA_VISIBLE_DEVICES only once, at init.  Source of
# truth: molmetal/molmetal_lam/sbdd_env/training_utils.py (loaded
# package-free via molmetal/scripts/_cuda_default.py).
import os as _wf130_os
import sys as _wf130_sys

_wf130_dir = _wf130_os.path.dirname(_wf130_os.path.abspath(__file__))
if _wf130_dir not in _wf130_sys.path:
    _wf130_sys.path.append(_wf130_dir)
from _cuda_default import ensure_cuda_visible_devices_default as _wf130_cvd

_wf130_cvd()
del _wf130_cvd, _wf130_dir, _wf130_os, _wf130_sys
# --- end CUDA default --------------------------------------------

# Wf-50 pre-condition: ensure MOLMETAL_DATA_DIR defaults to project-local data dir
# (overridable via env var; previously hard-coded to /mnt/storage which does not exist
#  on this host, causing Wf-60 P1 m1/m2/m19 FileNotFoundError 5/5 seeds).
import os as _wf50_os
from pathlib import Path as _wf50_Path

_wf50_proj_root = _wf50_Path(__file__).resolve().parent.parent.parent
_wf50_default_data = _wf50_proj_root / ".data" / "molmetal"
if not _wf50_os.environ.get("MOLMETAL_DATA_DIR") and _wf50_default_data.exists():
    _wf50_os.environ["MOLMETAL_DATA_DIR"] = str(_wf50_default_data)
del _wf50_os, _wf50_Path, _wf50_proj_root, _wf50_default_data


import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn

from molmetal.baselines.dmpnn import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    DMPNN_HIDDEN,
    DMPNN_DEPTH,
    DMPNN_DROPOUT,
    featurize_smiles_list,
    collate_graphs,
    _GRUUpdate,
)
from molmetal.baselines.eval_utils import compute_metrics
from molmetal.data.splits import ScaffoldSplitter

try:
    from molmetal.scripts.wf386a3_atomic_write import (
        atomic_write_file,
        safe_overwrite_check,
        detect_truncation,
    )

    _WF416_HAS_ATOMIC = True
except ImportError:  # pragma: no cover — fallback when running as standalone
    _WF416_HAS_ATOMIC = False
    atomic_write_file = None  # type: ignore[assignment]
    safe_overwrite_check = None  # type: ignore[assignment]
    detect_truncation = None  # type: ignore[assignment]


RDLogger = None
try:
    from rdkit import RDLogger as _RD

    RDLogger = _RD
    _RD.DisableLog("rdApp.*")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CELL_LINES: Tuple[str, ...] = ("MCF7", "A549", "A2780", "A2780cis")
N_CELL_LINES: int = len(CELL_LINES)
IC50_COLS: Tuple[str, ...] = tuple(f"ic50_{l.lower()}_nm" for l in CELL_LINES)
ACTIVE_THRESHOLD_NM: float = 10_000.0  # 10 µM

MT2_HIDDEN: int = DMPNN_HIDDEN
MT2_DEPTH: int = DMPNN_DEPTH
MT2_DROPOUT: float = DMPNN_DROPOUT
MT2_LR: float = 1e-3
MT2_BATCH_SIZE: int = 32
MT2_EPOCHS: int = 50
MT2_ALPHA: float = 0.5  # BCE weight; (1-alpha) → MSE on pIC50

METRICS_DIR = PROJECT_ROOT / "metrics" / "by_metric"
REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports"


# ---------------------------------------------------------------------------
# Four-cell-line multi-task model
# ---------------------------------------------------------------------------
def _wf416_atomic_write_text(
    path, content: str, min_lines: int = 50, expected_min_length: int = 200
) -> None:
    """Wf-416-A1 atomic-write-protocol wrapper around direct write_text.

    Falls back to direct write_text if atomic_write_file is unavailable
    (e.g. when the script is run standalone without the molmetal package
    on PYTHONPATH).
    """
    if _WF416_HAS_ATOMIC and atomic_write_file is not None:
        if safe_overwrite_check is not None:
            _risk = safe_overwrite_check(path, content, min_lines=min_lines)
            if _risk is not None:
                import logging

                logging.getLogger(__name__).warning(
                    "[Wf-416-A1] safe_overwrite_check diff_ratio=%.3f at %s",
                    _risk,
                    path,
                )
        atomic_write_file(path, content, encoding="utf-8")
        if detect_truncation is not None:
            _trunc = detect_truncation(content, expected_min_length=expected_min_length)
            if _trunc is not None:
                import logging

                logging.getLogger(__name__).warning(
                    "[Wf-416-A1] detect_truncation reason=%r at %s",
                    _trunc,
                    path,
                )
    else:  # pragma: no cover — fallback when atomic primitive unavailable
        Path(path).write_text(content, encoding="utf-8")


class FourCellReadout(nn.Module):
    """Shared trunk + 4 (regression + classification) head pairs.

    Each cell line has its own regression head (1 unit, pIC50) and
    classification head (2 units, inactive vs active).  All 8 heads
    consume the same ``2H``-dim pooled representation.
    """

    def __init__(self, hidden: int, dropout: float, n_lines: int = N_CELL_LINES):
        super().__init__()
        self.n_lines = n_lines
        self.shared = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.reg_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden, hidden // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden // 2, 1),
                )
                for _ in range(n_lines)
            ]
        )
        self.cls_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden, hidden // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden // 2, 2),
                )
                for _ in range(n_lines)
            ]
        )

    def forward(self, h_pool: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (pic50_pred [B, n_lines], cls_logits [B, n_lines, 2])."""
        shared = self.shared(h_pool)
        pic50 = torch.stack(
            [head(shared).squeeze(-1) for head in self.reg_heads], dim=1
        )  # (B, n_lines)
        cls = torch.stack(
            [head(shared) for head in self.cls_heads], dim=1
        )  # (B, n_lines, 2)
        return pic50, cls


class DMPNNFourCellModel(nn.Module):
    """D-MPNN backbone + 4-cell-line dual-head readout."""

    def __init__(
        self,
        atom_dim: int = ATOM_FEATURE_DIM,
        bond_dim: int = BOND_FEATURE_DIM,
        hidden: int = MT2_HIDDEN,
        depth: int = MT2_DEPTH,
        dropout: float = MT2_DROPOUT,
        n_lines: int = N_CELL_LINES,
    ):
        super().__init__()
        self.hidden = hidden
        self.depth = depth
        self.n_lines = n_lines
        self.dropout_layer = nn.Dropout(dropout)
        self.atom_emb = nn.Linear(atom_dim, hidden)
        self.bond_emb = nn.Linear(bond_dim, hidden)
        self.gru_update = _GRUUpdate(hidden, hidden)
        self.edge_init = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
        )
        self.readout = FourCellReadout(hidden, dropout, n_lines=n_lines)

    def _forward_one_mol(
        self,
        atom_f: torch.Tensor,  # (V, A)
        bond_f: torch.Tensor,  # (E, B)
        e_src: torch.Tensor,
        e_dst: torch.Tensor,
    ) -> torch.Tensor:
        V = atom_f.size(0)
        E = bond_f.size(0)
        device = atom_f.device
        h_v = self.dropout_layer(torch.relu(self.atom_emb(atom_f)))
        h_e = self.dropout_layer(torch.relu(self.bond_emb(bond_f)))
        for _ in range(self.depth):
            atom_incoming = torch.zeros(V, self.hidden, device=device)
            atom_incoming.index_add_(0, e_dst, h_e)
            neighbor_sum = atom_incoming[e_src]
            h_e = self.gru_update(h_e, h_v[e_src], neighbor_sum)
        outgoing = torch.zeros(V, self.hidden, device=device)
        outgoing.index_add_(0, e_src, h_e)
        h_comb = torch.cat([h_v, outgoing], dim=-1)
        return h_comb.sum(dim=0)

    def forward(
        self,
        atom_feats: torch.Tensor,
        bond_feats: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = atom_feats.size(0)
        device = atom_feats.device
        pooled_list: List[torch.Tensor] = []
        for b in range(B):
            V = atom_feats.size(1)
            valid_edges = bond_feats[b].abs().sum(dim=-1) > 0
            E = valid_edges.sum().item()
            if E == 0 or V == 0:
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
        h_pool = torch.stack(pooled_list, dim=0)
        return self.readout(h_pool)


class FourCellMultiTaskLoss(nn.Module):
    """alpha * mean(BCE across cell lines) + (1-alpha) * mean(MSE across cell lines).

    Skips cell lines whose labels are NaN/missing for that batch element.
    """

    def __init__(self, alpha: float = MT2_ALPHA, n_lines: int = N_CELL_LINES):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = float(alpha)
        self.n_lines = n_lines
        self.bce = nn.CrossEntropyLoss(reduction="mean")
        self.mse = nn.MSELoss(reduction="mean")

    def forward(
        self,
        pic50_pred: torch.Tensor,  # (B, n_lines)
        cls_logits: torch.Tensor,  # (B, n_lines, 2)
        pic50_targets: torch.Tensor,  # (B, n_lines)
        active_targets: torch.Tensor,  # (B, n_lines) long
        valid_mask: torch.Tensor,  # (B, n_lines) bool — True where label valid
    ) -> Tuple[torch.Tensor, float, float]:
        device = pic50_pred.device
        bce_loss = torch.zeros((), device=device)
        mse_loss = torch.zeros((), device=device)
        n_valid_cls = 0
        n_valid_reg = 0
        for i in range(self.n_lines):
            mask_i = valid_mask[:, i]
            if mask_i.any():
                # Classification (active vs inactive)
                idx = mask_i.nonzero(as_tuple=True)[0]
                logits_i = cls_logits[idx, i]
                tgt_i = active_targets[idx, i].long()
                bce_loss = bce_loss + self.bce(logits_i, tgt_i)
                n_valid_cls += 1
                # Regression (pIC50)
                reg_pred_i = pic50_pred[idx, i]
                reg_tgt_i = pic50_targets[idx, i]
                valid_reg = ~torch.isnan(reg_tgt_i)
                if valid_reg.any():
                    mse_loss = mse_loss + self.mse(
                        reg_pred_i[valid_reg], reg_tgt_i[valid_reg]
                    )
                    n_valid_reg += 1
        bce_avg = bce_loss / max(n_valid_cls, 1)
        mse_avg = mse_loss / max(n_valid_reg, 1)
        total = self.alpha * bce_avg + (1.0 - self.alpha) * mse_avg
        return total, float(bce_avg.detach()), float(mse_avg.detach())


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
@dataclass
class PlatinAI4Cell:
    smiles: np.ndarray  # (N,) canonical SMILES
    ic50_nm: np.ndarray  # (N, 4) IC50 in nM
    pic50: np.ndarray  # (N, 4) pIC50 = -log10(IC50_nm / 1e9)
    active: np.ndarray  # (N, 4) int8 — 1 if IC50 ≤ 10 µM else 0

    @classmethod
    def from_csv(cls, path: Path) -> "PlatinAI4Cell":
        rows: List[Dict[str, str]] = []
        with path.open() as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)
        n = len(rows)
        smiles = np.array([r["smiles"] for r in rows], dtype=object)
        ic50 = np.zeros((n, N_CELL_LINES), dtype=np.float32)
        for j, col in enumerate(IC50_COLS):
            for i, r in enumerate(rows):
                v = r.get(col, "")
                if v == "" or v is None:
                    ic50[i, j] = np.nan
                else:
                    try:
                        ic50[i, j] = float(v)
                    except Exception:
                        ic50[i, j] = np.nan
        # pIC50 = -log10(IC50_nm * 1e-9) = 9 - log10(IC50_nm)
        pic50 = np.zeros_like(ic50)
        valid = ~np.isnan(ic50) & (ic50 > 0)
        pic50[valid] = 9.0 - np.log10(ic50[valid])
        pic50[~valid] = np.nan
        active = (ic50 <= ACTIVE_THRESHOLD_NM).astype(np.int8)
        active[np.isnan(ic50)] = -1  # mark missing
        return cls(smiles=smiles, ic50_nm=ic50, pic50=pic50, active=active)

    def __len__(self) -> int:
        return int(self.smiles.shape[0])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def _train_one_epoch(
    model: DMPNNFourCellModel,
    optimizer: torch.optim.Optimizer,
    criterion: FourCellMultiTaskLoss,
    graphs: List,
    pic50_labels: np.ndarray,
    active_labels: np.ndarray,
    batch_size: int = MT2_BATCH_SIZE,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_cls = 0.0
    total_reg = 0.0
    n_samples = 0
    indices = np.arange(len(graphs))
    np.random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        batch_graphs = [graphs[i] for i in batch_idx]
        batch_pic50 = pic50_labels[batch_idx]
        batch_active = active_labels[batch_idx]
        # Filter rows with at least one valid graph AND any non-NaN pic50
        valid_mask = np.zeros((len(batch_idx), N_CELL_LINES), dtype=bool)
        keep_rows: List[int] = []
        for k, g in enumerate(batch_graphs):
            if g is None:
                continue
            row_pic50 = batch_pic50[k]
            valid_pic50 = ~np.isnan(row_pic50)
            if not valid_pic50.any():
                continue
            valid_mask[k] = True
            keep_rows.append(k)
        if not keep_rows:
            continue
        g_list = [batch_graphs[k] for k in keep_rows]
        y_pic50 = np.stack([batch_pic50[k] for k in keep_rows], axis=0)
        y_active = np.stack([batch_active[k] for k in keep_rows], axis=0)
        y_pic50 = np.nan_to_num(y_pic50, nan=5.0)  # fill for regression head
        y_active = np.where(y_active < 0, 0, y_active)
        vm = valid_mask[keep_rows]
        batch_data = collate_graphs(g_list)
        if batch_data is None:
            continue
        atom_f, bond_f, e_src, e_dst = batch_data
        atom_f = atom_f.to(device)
        bond_f = bond_f.to(device)
        e_src = e_src.to(device)
        e_dst = e_dst.to(device)
        y_pic50_t = torch.from_numpy(y_pic50.astype(np.float32)).to(device)
        y_active_t = torch.from_numpy(y_active.astype(np.int64)).to(device)
        vm_t = torch.from_numpy(vm.astype(bool)).to(device)
        optimizer.zero_grad()
        pic50_pred, cls_logits = model(atom_f, bond_f, e_src, e_dst)
        loss, lc, lr_ = criterion(pic50_pred, cls_logits, y_pic50_t, y_active_t, vm_t)
        loss.backward()
        optimizer.step()
        bs = len(keep_rows)
        total_loss += loss.item() * bs
        total_cls += lc * bs
        total_reg += lr_ * bs
        n_samples += bs
    if n_samples == 0:
        return 0.0, 0.0, 0.0
    return total_loss / n_samples, total_cls / n_samples, total_reg / n_samples


@torch.no_grad()
def _predict(
    model: DMPNNFourCellModel,
    graphs: List,
    active_labels: np.ndarray,
    batch_size: int = MT2_BATCH_SIZE,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (active_true [N, 4], active_prob [N, 4]) for valid rows.

    Invalid molecules contribute (-1, 0.0) per cell line.
    """
    model.eval()
    n = len(graphs)
    out_active = np.full((n, N_CELL_LINES), -1, dtype=np.int32)
    out_prob = np.zeros((n, N_CELL_LINES), dtype=np.float32)
    for start in range(0, n, batch_size):
        batch_graphs = graphs[start : start + batch_size]
        batch_active = active_labels[start : start + batch_size]
        valid_rows: List[int] = []
        g_list: List = []
        for k, g in enumerate(batch_graphs):
            if g is None:
                continue
            valid_rows.append(k)
            g_list.append(g)
        if not g_list:
            continue
        batch_data = collate_graphs(g_list)
        if batch_data is None:
            continue
        atom_f, bond_f, e_src, e_dst = batch_data
        atom_f = atom_f.to(device)
        bond_f = bond_f.to(device)
        e_src = e_src.to(device)
        e_dst = e_dst.to(device)
        _, cls_logits = model(atom_f, bond_f, e_src, e_dst)
        probs = torch.softmax(cls_logits, dim=-1)[..., 1].cpu().numpy()  # (B, 4)
        for vi, k in enumerate(valid_rows):
            # `k` indexes the BATCH; `out_active` is indexed by ROW, so the
            # global offset `start` has to be added. Writing `out_active[k]`
            # made every batch overwrite rows 0..batch_size-1 and left the
            # rest at -1, so the >= 0 mask kept only batch_size rows: every
            # AUC recorded from this function was measured on 32 molecules
            # instead of n_test=815. The exact rationals in the recorded JSON
            # (0.65625 = 21/32, 0.33714285714285713 = 59/175) are the tell.
            gk = start + k
            for line_idx in range(N_CELL_LINES):
                if batch_active[k, line_idx] >= 0:
                    out_active[gk, line_idx] = int(batch_active[k, line_idx])
                    out_prob[gk, line_idx] = float(probs[vi, line_idx])
    return out_active, out_prob


# ---------------------------------------------------------------------------
# Baseline runner
# ---------------------------------------------------------------------------
class FourCellDMPNNRunner:
    """Multi-task D-MPNN runner for 4-cell-line PlatinAI."""

    def __init__(
        self,
        hidden: int = MT2_HIDDEN,
        depth: int = MT2_DEPTH,
        dropout: float = MT2_DROPOUT,
        lr: float = MT2_LR,
        epochs: int = MT2_EPOCHS,
        batch_size: int = MT2_BATCH_SIZE,
        alpha: float = MT2_ALPHA,
        seed: int = 42,
        splitter: Optional[object] = None,
        device: Optional[torch.device] = None,
    ):
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
        self.model: Optional[DMPNNFourCellModel] = None
        self._train_idx: Optional[np.ndarray] = None
        self._val_idx: Optional[np.ndarray] = None
        self._test_idx: Optional[np.ndarray] = None

    def _split(
        self,
        dataset: PlatinAI4Cell,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Wrap dataset in a minimal adapter that ScaffoldSplitter expects.
        if self.splitter is not None:
            res = self.splitter(dataset)
            return res.train_idx, res.val_idx, res.test_idx
        # Fallback: random 70/10/20
        from sklearn.model_selection import train_test_split

        idx = np.arange(len(dataset))
        np.random.seed(self.seed)
        idx_train, idx_tmp = train_test_split(
            idx, test_size=0.3, random_state=self.seed
        )
        idx_val, idx_test = train_test_split(
            idx_tmp, test_size=2 / 3, random_state=self.seed
        )
        return idx_train, idx_val, idx_test

    def fit(
        self,
        ds: PlatinAI4Cell,
        verbose: bool = True,
    ) -> "FourCellDMPNNRunner":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if verbose:
            print(f"[4cell] Featurizing {len(ds)} molecules...", flush=True)
        graphs_all = featurize_smiles_list([str(s) for s in ds.smiles])
        if verbose:
            print(
                f"[4cell] {sum(1 for g in graphs_all if g is not None)} valid "
                f"graphs out of {len(graphs_all)} SMILES",
                flush=True,
            )
        idx_train, idx_val, idx_test = self._split(ds)
        self._train_idx = idx_train
        self._val_idx = idx_val
        self._test_idx = idx_test
        if verbose:
            print(
                f"[4cell] Split: train={len(idx_train)} val={len(idx_val)} "
                f"test={len(idx_test)} (seed={self.seed})",
                flush=True,
            )
        graphs_train = [graphs_all[i] for i in idx_train]
        graphs_val = [graphs_all[i] for i in idx_val]
        active_train = ds.active[idx_train]
        active_val = ds.active[idx_val]
        pic50_train = ds.pic50[idx_train]
        pic50_val = ds.pic50[idx_val]

        self.model = DMPNNFourCellModel(
            atom_dim=ATOM_FEATURE_DIM,
            bond_dim=BOND_FEATURE_DIM,
            hidden=self.hidden,
            depth=self.depth,
            dropout=self.dropout,
            n_lines=N_CELL_LINES,
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = FourCellMultiTaskLoss(alpha=self.alpha, n_lines=N_CELL_LINES)

        best_val_auc_mean = 0.0
        best_state: Optional[Dict] = None

        for epoch in range(self.epochs):
            t0 = time.time()
            train_loss, cls_loss, reg_loss = _train_one_epoch(
                self.model,
                optimizer,
                criterion,
                graphs_train,
                pic50_train,
                active_train,
                batch_size=self.batch_size,
                device=self.device,
            )
            # Val
            y_v, s_v = _predict(
                self.model,
                graphs_val,
                active_val,
                batch_size=self.batch_size,
                device=self.device,
            )
            # Flatten per cell line, compute AUC per line, then mean
            per_line_auc_val = {}
            for j, line in enumerate(CELL_LINES):
                mask = y_v[:, j] >= 0
                if mask.sum() < 30:
                    continue
                m = compute_metrics(y_v[mask, j], s_v[mask, j])
                per_line_auc_val[line] = m["roc_auc"]
            mean_val_auc = (
                float(np.mean(list(per_line_auc_val.values())))
                if per_line_auc_val
                else 0.0
            )
            elapsed = time.time() - t0
            if verbose and (epoch % 5 == 0 or epoch == self.epochs - 1):
                print(
                    f"[4cell ep{epoch:3d}] loss={train_loss:.4f} "
                    f"(cls={cls_loss:.4f} reg={reg_loss:.4f}) "
                    f"val_auc_mean={mean_val_auc:.4f} "
                    f"({', '.join(f'{l}={per_line_auc_val.get(l, 0):.3f}' for l in CELL_LINES)}) "
                    f"time={elapsed:.1f}s",
                    flush=True,
                )
            if mean_val_auc > best_val_auc_mean:
                best_val_auc_mean = mean_val_auc
                best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
        if best_state is not None:
            self.model.load_state_dict(best_state)
        if verbose:
            print(
                f"[4cell] Best val mean AUC = {best_val_auc_mean:.4f}",
                flush=True,
            )
        return self

    def evaluate(
        self,
        ds: PlatinAI4Cell,
        verbose: bool = True,
    ) -> Dict[str, object]:
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        graphs_all = featurize_smiles_list([str(s) for s in ds.smiles])
        idx_test = self._test_idx
        graphs_test = [graphs_all[i] for i in idx_test]
        active_test = ds.active[idx_test]
        pic50_test = ds.pic50[idx_test]
        y_t, s_t = _predict(
            self.model,
            graphs_test,
            active_test,
            batch_size=self.batch_size,
            device=self.device,
        )
        per_line_auc: Dict[str, float] = {}
        per_line_ap: Dict[str, float] = {}
        per_line_n: Dict[str, int] = {}
        per_line_active_rate: Dict[str, float] = {}
        for j, line in enumerate(CELL_LINES):
            mask = y_t[:, j] >= 0
            if mask.sum() < 30:
                continue
            m = compute_metrics(y_t[mask, j], s_t[mask, j])
            per_line_auc[line] = m["roc_auc"]
            per_line_ap[line] = m["pr_auc"]
            per_line_n[line] = int(mask.sum())
            per_line_active_rate[line] = float(y_t[mask, j].mean())
        auc_mean = (
            float(np.mean(list(per_line_auc.values())))
            if per_line_auc
            else float("nan")
        )
        return {
            "per_line_auc": per_line_auc,
            "per_line_ap": per_line_ap,
            "per_line_n": per_line_n,
            "per_line_active_rate": per_line_active_rate,
            "n_test": int(len(idx_test)),
            "auc_mean": auc_mean,
            "auc_a2780cis": per_line_auc.get("A2780cis", float("nan")),
            "y_true": y_t,
            "y_score": s_t,
        }


# ---------------------------------------------------------------------------
# Checkpoint persistence (Wf-P1-1)
# ---------------------------------------------------------------------------
# WHY
# ---
# This trainer used to persist a NUMBER and no WEIGHTS: ``grep -c torch.save
# molmetal/scripts/train_platinai_4cell.py`` returned 0, and the only
# ``state_dict`` call in the file was a ``load_state_dict`` of the best-val
# in-memory copy (line ~690). The run that produced the strong PlatinAI
# number therefore left no artefact that any downstream consumer could load,
# which left the cross-application route in
# ``wf_generated_to_activity.py`` pointed at the *other* model family
# (``MultiTaskWassersteinModel`` under
# ``molmetal/reports/dmpnn_platinai_4cell_50ep/``, whose own recorded test
# AUCs are ~0.51). Naming follows the sibling trainer's convention --
# ``train_dmpnn_multitask_wasserstein.py`` writes
# ``dmpnn_mtw_{metal}_seed{seed}.pt`` -- so one is
# ``platinai4cell_seed{seed}.pt`` here (no metal loop: the cohort is Pt-only).
#
# The schema mirrors the dmpnn one: ``model_state_dict`` plus the
# architecture hyper-parameters needed to REBUILD the model, because
# ``DMPNNFourCellModel.__init__`` takes hidden/depth/dropout/n_lines as
# constructor args and a bare state_dict cannot be loaded without them.
CKPT_FILENAME_TEMPLATE: str = "platinai4cell_seed{seed}.pt"
CKPT_SCHEMA_VERSION: int = 1


def _resolve_ckpt_dir(args: argparse.Namespace) -> Optional[Path]:
    """``None`` means "do not persist weights" (explicit --no-save-checkpoints)."""
    if getattr(args, "no_save_checkpoints", False):
        return None
    raw = getattr(args, "ckpt_dir", None)
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(
    path: Path,
    model: DMPNNFourCellModel,
    *,
    seed: int,
    eval_out: Dict[str, object],
    args: argparse.Namespace,
) -> Path:
    """Persist the evaluated weights + everything needed to rebuild the model.

    Writes the *best-val* weights, i.e. exactly the ones ``evaluate()`` just
    scored -- ``FourCellDMPNNRunner.fit`` restores ``best_state`` into
    ``self.model`` before returning, so ``model.state_dict()`` here is the
    state the recorded AUC belongs to. Saving anything else would attach a
    number to weights that did not produce it.
    """
    torch.save(
        {
            "schema_version": CKPT_SCHEMA_VERSION,
            "model_state_dict": {
                k: v.detach().cpu() for k, v in model.state_dict().items()
            },
            # Architecture -- REQUIRED to rebuild: these are constructor args,
            # not state_dict entries, so a consumer that guesses them loads
            # into a differently-shaped model or crashes on a size mismatch.
            "atom_dim": int(ATOM_FEATURE_DIM),
            "bond_dim": int(BOND_FEATURE_DIM),
            "hidden": int(model.hidden),
            "depth": int(model.depth),
            "dropout": float(model.dropout_layer.p),
            "n_lines": int(model.n_lines),
            "cell_lines": list(CELL_LINES),
            # Provenance / reproduction
            "seed": int(seed),
            "epochs": int(getattr(args, "epochs", MT2_EPOCHS)),
            "batch_size": int(getattr(args, "batch_size", MT2_BATCH_SIZE)),
            "lr": float(getattr(args, "lr", MT2_LR)),
            "alpha": float(getattr(args, "alpha", MT2_ALPHA)),
            "scaffold_split": bool(getattr(args, "scaffold_split", False)),
            "data": str(getattr(args, "data", "")),
            "model_family": "DMPNNFourCellModel",
            # Recorded scores -- the number these weights produced
            "auc_mean": float(eval_out.get("auc_mean", float("nan"))),
            "auc_a2780cis": float(eval_out.get("auc_a2780cis", float("nan"))),
            "per_line_auc": {
                k: float(v) for k, v in (eval_out.get("per_line_auc") or {}).items()
            },
            "n_test": int(eval_out.get("n_test", 0)),
        },
        path,
    )
    return path


def load_checkpoint(path: Path, device: Optional[torch.device] = None) -> Dict[str, object]:
    """Load a ``platinai4cell_seed*.pt`` into a fresh model in eval mode.

    This is the function ``wf_generated_to_activity.py`` should call for the
    PlatinAI 4-cell family (see ``score_platinai_4cell`` there). It rebuilds the
    architecture from the stored constructor args rather than from any
    module-level default, so a checkpoint trained at hidden=256 reloads into a
    hidden=256 model even though ``MT2_HIDDEN`` says 128.

    Strictness is the point: ``load_state_dict(..., strict=True)`` raises on
    any missing or unexpected key, which is the check that catches a
    checkpoint written by a different model family (its keys simply will not
    exist here).
    """
    path = Path(path)
    dev = device or torch.device("cpu")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(
            f"{path} is not a {CKPT_FILENAME_TEMPLATE} checkpoint "
            f"(model_family={ckpt.get('model_family') if isinstance(ckpt, dict) else type(ckpt)!r})"
        )
    model = DMPNNFourCellModel(
        atom_dim=int(ckpt.get("atom_dim", ATOM_FEATURE_DIM)),
        bond_dim=int(ckpt.get("bond_dim", BOND_FEATURE_DIM)),
        hidden=int(ckpt["hidden"]),
        depth=int(ckpt["depth"]),
        dropout=float(ckpt.get("dropout", MT2_DROPOUT)),
        n_lines=int(ckpt.get("n_lines", N_CELL_LINES)),
    ).to(dev)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return {"model": model, "meta": ckpt, "device": dev}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Mol-Metal on PlatinAI 4-cell-line Pt dataset."
    )
    p.add_argument(
        "--data",
        default=str(PROJECT_ROOT / "molmetal" / "data" / "platinai_4cell.csv"),
        help="Path to platinai_4cell.csv",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="RNG seeds (default: 0 1 2)",
    )
    p.add_argument(
        "--epochs", type=int, default=MT2_EPOCHS, help="Epochs per seed (default: 50)"
    )
    p.add_argument(
        "--cuda-visible-devices",
        type=str,
        default="",
        help=(
            "Override CUDA_VISIBLE_DEVICES before device resolution.  "
            "Empty '' (the default) = inherit the shell env; if neither "
            "is set, the canonical default '0' applies (GPU 0 — RTX PRO "
            "6000 Blackwell)."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=MT2_BATCH_SIZE,
        help="Batch size (default: 32)",
    )
    p.add_argument(
        "--lr", type=float, default=MT2_LR, help="Learning rate (default: 1e-3)"
    )
    p.add_argument(
        "--alpha", type=float, default=MT2_ALPHA, help="BCE/MSE weight (default: 0.5)"
    )
    p.add_argument("--hidden", type=int, default=MT2_HIDDEN, help="D-MPNN hidden dim")
    p.add_argument("--depth", type=int, default=MT2_DEPTH, help="D-MPNN depth")
    p.add_argument(
        "--scaffold-split", action="store_true", help="Use Bemis-Murcko scaffold split"
    )
    p.add_argument(
        "--out-dir",
        default="logs/metric2_platinai_4cell",
        help="Directory for per-seed JSON output",
    )
    # Wf-P1-1 — checkpoint persistence. DEFAULT ON, writing into an isolated
    # per-run <out_dir>/checkpoints, exactly as
    # train_dmpnn_multitask_wasserstein.py does (Wf-A10). Off only when the
    # caller asks (--no-save-checkpoints), so a run that forgets the flag
    # still leaves loadable weights behind instead of a number with no model.
    p.add_argument(
        "--ckpt-dir",
        default=None,
        help=(
            "Directory for per-seed .pt checkpoints "
            "(default: <out-dir>/checkpoints). Filename "
            f"{CKPT_FILENAME_TEMPLATE}. Relative paths resolve against the "
            "repo root."
        ),
    )
    p.add_argument(
        "--no-save-checkpoints",
        action="store_true",
        help="Disable checkpoint persistence entirely (default: weights ARE saved).",
    )
    p.add_argument(
        "--out-metric",
        default=None,
        help="Path to aggregate metrics JSON (default: metrics/by_metric/platinai_4cell_auc.json)",
    )
    p.add_argument(
        "--out-report",
        default=None,
        help="Path to markdown report (default: molmetal/reports/wf_tier1_metric2_platinai/final.md)",
    )
    p.add_argument(
        "--config",
        default=None,
        help="YAML config (§5.11). Keys resolve to the dests above; an "
        "explicit CLI flag still wins.",
    )
    args = p.parse_args(argv)
    if getattr(args, "config", None):
        from wfA2_config_loader import apply_yaml_overrides

        args = apply_yaml_overrides(p, args, args.config, argv=list(argv or []))
    return args


def _build_splitter(args: argparse.Namespace, seed: int):
    if args.scaffold_split:
        return ScaffoldSplitter(
            strategy="largest_first",
            fractions=(0.7, 0.1, 0.2),
            seed=seed,
        )
    return None


def _run_one_seed(args: argparse.Namespace, seed: int) -> Dict[str, object]:
    # Wf-130 — GPU-first device defaults (GPU 0 — RTX PRO 6000 Blackwell
    # by default) when allocatable.  Flag '' = inherit (coerce to None so
    # the resolver default applies); env var / non-empty flag still win.
    from molmetal.molmetal_lam.sbdd_env.training_utils import (
        apply_default_cuda_visible_devices,
    )

    apply_default_cuda_visible_devices(
        override=(getattr(args, "cuda_visible_devices", "") or None)
    )
    print(
        f"\n[metric2] === seed {seed} (epochs={args.epochs}, bs={args.batch_size}, "
        f"lr={args.lr}, alpha={args.alpha}, scaffold={args.scaffold_split}) ===",
        flush=True,
    )
    ds = PlatinAI4Cell.from_csv(Path(args.data))
    print(
        f"[metric2]   rows={len(ds)}  "
        f"active_rate={ds.active[(ds.active>=0)].mean():.3f}",
        flush=True,
    )
    splitter = _build_splitter(args, seed)
    runner = FourCellDMPNNRunner(
        hidden=args.hidden,
        depth=args.depth,
        dropout=MT2_DROPOUT,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        alpha=args.alpha,
        seed=seed,
        splitter=splitter,
    )
    t0 = time.time()
    runner.fit(ds, verbose=True)
    elapsed = time.time() - t0
    eval_out = runner.evaluate(ds, verbose=False)
    pl_auc = eval_out["per_line_auc"]
    pl_ap = eval_out["per_line_ap"]
    pl_n = eval_out["per_line_n"]
    pl_act = eval_out["per_line_active_rate"]
    auc_mean = float(np.mean(list(pl_auc.values()))) if pl_auc else float("nan")
    print(
        "[metric2]   test AUC per cell line: "
        + ", ".join(f"{l}={pl_auc.get(l, float('nan')):.4f}" for l in CELL_LINES)
        + f"  mean={auc_mean:.4f}  (elapsed={elapsed:.1f}s)",
        flush=True,
    )
    # Wf-P1-1 — persist the weights that produced the AUC above. Uses the
    # SAME eval_out the AUC was computed from, so the number and the artefact
    # cannot drift apart.
    ckpt_path: Optional[str] = None
    ckpt_dir = _resolve_ckpt_dir(args)
    if ckpt_dir is not None and runner.model is not None:
        path = ckpt_dir / CKPT_FILENAME_TEMPLATE.format(seed=seed)
        save_checkpoint(path, runner.model, seed=seed, eval_out=eval_out, args=args)
        ckpt_path = str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(
            PROJECT_ROOT
        ) else str(path)
        print(f"[metric2] wrote checkpoint {path}", flush=True)
    return {
        "seed": seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "alpha": args.alpha,
        "n_rows": int(len(ds)),
        "n_test": int(eval_out["n_test"]),
        "per_line_auc": pl_auc,
        "per_line_ap": pl_ap,
        "per_line_n": pl_n,
        "per_line_active_rate": pl_act,
        "auc_mean": auc_mean,
        "auc_a2780cis": pl_auc.get("A2780cis", float("nan")),
        "elapsed_seconds": elapsed,
        "checkpoint_path": ckpt_path,
    }


def _aggregate(per_seed: List[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {"n_seeds": len(per_seed)}
    aucs_mean = [d["auc_mean"] for d in per_seed]
    out["auc_mean_avg"] = float(np.mean(aucs_mean))
    out["auc_mean_std"] = float(np.std(aucs_mean))
    out["per_seed_auc_mean"] = aucs_mean
    out["per_seed_auc_a2780cis"] = [d["auc_a2780cis"] for d in per_seed]
    for line in CELL_LINES:
        vals = [d["per_line_auc"].get(line, float("nan")) for d in per_seed]
        out[f"auc_{line}_avg"] = float(np.mean(vals))
        out[f"auc_{line}_std"] = float(np.std(vals))
        out[f"per_seed_auc_{line}"] = vals
    return out


def _write_report(
    out_path: Path,
    args: argparse.Namespace,
    aggregate: Dict[str, object],
    per_seed: List[Dict[str, object]],
) -> None:
    lines: List[str] = [
        "# Metric 2 — PlatinAI Pt 4-cell-line AUC",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d')}",
        "**Model:** 4-cell-line multi-task D-MPNN (4 cls + 4 reg heads)",
        f"**Split:** {'scaffold (Bemis-Murcko) 70/10/20' if args.scaffold_split else 'random 70/10/20'}",
        f"**Seeds:** {args.seeds}  ",
        f"**Epochs:** {args.epochs}, lr={args.lr}, batch={args.batch_size}, alpha={args.alpha}",
        "",
        "## Rusanov 2025 baseline (Tier 1 reference)",
        "",
        "Rusanov et al., *PlatinAI* (ChemRxiv 2025-06-20 v2, DOI 10.26434/chemrxiv-2025-pp32k)",
        "trained a GNN+fragment model on **4,078 Pt complexes / 18,995 IC50** across",
        "**MCF-7, A549, A2780, A2780cis** with scaffold split.  Per-cell-line AUC values",
        "are reported in paper Tables S4–S5 (supplementary, behind a Cloudflare",
        "challenge we can't directly fetch from this host).",
        "",
        "**Paper claim:** superior A2780cis (cisplatin-resistant) AUC vs cisplatin —",
        "our target is to demonstrate that a metal-aware D-MPNN with multi-task",
        "heads can learn per-cell-line signal on the same schema.",
        "",
        "## Our results (this run)",
        "",
        "| Cell line | n eval'd rows for this line (avg) | AUC mean ± std | Active rate |",
        "|---|---|---|---|",
    ]
    for line in CELL_LINES:
        n_avg = float(np.mean([d["per_line_n"].get(line, 0) for d in per_seed]))
        avg = aggregate[f"auc_{line}_avg"]
        std = aggregate[f"auc_{line}_std"]
        ar = float(
            np.mean([d["per_line_active_rate"].get(line, 0.0) for d in per_seed])
        )
        lines.append(f"| {line} | {n_avg:.0f} | {avg:.4f} ± {std:.4f} | {ar:.2%} |")
    overall_mean = aggregate["auc_mean_avg"]
    overall_std = aggregate["auc_mean_std"]
    lines += [
        f"| **Mean** | — | **{overall_mean:.4f} ± {overall_std:.4f}** | — |",
        "",
        "## A2780cis focus (cisplatin-resistant — paper highlight)",
        "",
        f"- AUC on A2780cis: **{aggregate['auc_A2780cis_avg']:.4f} "
        f"± {aggregate['auc_A2780cis_std']:.4f}** (per seed: "
        f"{[round(x, 4) for x in aggregate['per_seed_auc_a2780cis']]})",
        "- The model has a dedicated classification head for A2780cis; this AUC is",
        "  independent of the other 3 cell lines and captures the resistance signal.",
        "",
        "## Honest caveats",
        "",
        "* The ChemRxiv supplementary xlsx tables S4-S5 were not directly fetchable",
        "  from this host (Cloudflare JS challenge).  We therefore built a",
        "  representative 4,078-row Pt dataset with **synthetic IC50 labels** —",
        "  see `molmetal/scripts/build_platinai_4cell.py` for the exact generation",
        "  logic.  The numbers reported here are useful as a **pipeline test** (the",
        "  4-cell-line multi-task D-MPNN learns per-cell-line signal) but are NOT",
        "  directly comparable to Rusanov 2025's reported per-line AUC.",
        "* The D-MPNN backbone here is *not* the full V4 hybrid (no EGNN / Perceiver",
        "  fusion).  This is the metal-aware multi-task D-MPNN baseline; richer",
        "  hybrids should match or beat it.",
        "* Scaffold split is OOD by construction → AUC will be lower than random",
        "  splits.  Any direct comparison to the paper's protocol is therefore",
        "  conservative.",
        "",
        "## Per-seed detail",
        "",
        "| Seed | n_test | AUC mean | "
        + " | ".join(f"AUC {l}" for l in CELL_LINES)
        + " |",
        "|---|---|---|---|" + "---|" * N_CELL_LINES,
    ]
    for d in per_seed:
        row = [f"{d['seed']}", f"{d['n_test']}", f"{d['auc_mean']:.4f}"]
        for line in CELL_LINES:
            row.append(f"{d['per_line_auc'].get(line, float('nan')):.4f}")
        lines.append("| " + " | ".join(row) + " |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _wf416_atomic_write_text(out_path, "\n".join(lines))


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # Wf-P1-1 — default the checkpoint dir to <out_dir>/checkpoints so a run
    # is self-contained: two runs with different --out-dir cannot collide on
    # one checkpoint set, and --ckpt-dir / yaml output.ckpt_dir still wins
    # when set. Same default as train_dmpnn_multitask_wasserstein.py (Wf-A10).
    if not getattr(args, "ckpt_dir", None):
        args.ckpt_dir = str(out_dir / "checkpoints")
        print(
            f"[metric2] --ckpt-dir not set; using {args.ckpt_dir} "
            f"(isolated per --out-dir)",
            flush=True,
        )

    per_seed: List[Dict[str, object]] = []
    for seed in args.seeds:
        try:
            d = _run_one_seed(args, seed)
        except Exception as exc:
            print(f"[metric2] seed {seed} FAILED: {exc!r}", flush=True)
            import traceback

            traceback.print_exc()
            d = {"seed": seed, "error": repr(exc)}
        per_seed.append(d)
        seed_path = out_dir / f"seed{seed}.json"
        _wf416_atomic_write_text(seed_path, json.dumps(d, indent=2, default=str))
        print(f"[metric2] wrote {seed_path}", flush=True)

    aggregate = _aggregate([d for d in per_seed if "error" not in d])
    full_payload = {
        "model": "dmpnn_4cell_multitask",
        "data": args.data,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "alpha": args.alpha,
        "hidden": args.hidden,
        "depth": args.depth,
        "cell_lines": list(CELL_LINES),
        "scaffold_split": args.scaffold_split,
        "ckpt_dir": (
            None if getattr(args, "no_save_checkpoints", False) else args.ckpt_dir
        ),
        "aggregate": aggregate,
        "per_seed": per_seed,
        "rusanov_2025_baseline_note": (
            "Rusanov 2025 reports per-cell-line AUC for 4,078 Pt / 18,995 IC50, "
            "4 cell lines (MCF-7, A549, A2780, A2780cis). Exact numbers in paper "
            "Tables S4-S5 (supplementary xlsx, Cloudflare-blocked from this host). "
            "Our dataset is built with synthetic IC50 labels — see "
            "build_platinai_4cell.py for the honest framing."
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    aggregate_path = out_dir / "aggregate.json"
    _wf416_atomic_write_text(
        aggregate_path, json.dumps(full_payload, indent=2, default=str)
    )
    print(
        "\n[metric2] === AGGREGATE ===",
        f"\n[metric2] AUC mean = {aggregate['auc_mean_avg']:.4f} ± {aggregate['auc_mean_std']:.4f}",
        f"\n[metric2] A2780cis  = {aggregate['auc_A2780cis_avg']:.4f} ± {aggregate['auc_A2780cis_std']:.4f}",
        flush=True,
    )
    print(f"[metric2] wrote {aggregate_path}", flush=True)

    # metrics/by_metric/<key>.json
    metric_path = (
        Path(args.out_metric)
        if args.out_metric
        else METRICS_DIR / "platinai_4cell_auc.json"
    )
    metric_payload = {
        "metric": "platinai_4cell_auc",
        "description": (
            "Test ROC-AUC per cell line on PlatinAI 4,078 Pt 4-cell-line dataset "
            "(synthetic IC50 labels) with D-MPNN 4-cell-line multi-task "
            "(4 cls + 4 reg heads)."
        ),
        "unit": "ROC-AUC (0-1, higher better) per cell line",
        "scope": "tier1_precious_metal_anticancer",
        "source": (
            f"molmetal/scripts/train_platinai_4cell.py (seeds={args.seeds}, "
            f"epochs={args.epochs}, scaffold_split={args.scaffold_split})"
        ),
        "rusanov_2025_baseline_note": full_payload["rusanov_2025_baseline_note"],
        "value_overall_mean_auc": float(aggregate["auc_mean_avg"]),
        "value_overall_mean_std": float(aggregate["auc_mean_std"]),
        "value_a2780cis_auc": float(aggregate["auc_A2780cis_avg"]),
        "value_a2780cis_std": float(aggregate["auc_A2780cis_std"]),
        "per_cell_line": {
            line: {
                "auc_mean": float(aggregate[f"auc_{line}_avg"]),
                "auc_std": float(aggregate[f"auc_{line}_std"]),
                "per_seed": [float(v) for v in aggregate[f"per_seed_auc_{line}"]],
            }
            for line in CELL_LINES
        },
        "n_seeds": int(aggregate["n_seeds"]),
        "n_rows": int(per_seed[0].get("n_rows", 0)) if per_seed else 0,
        "timestamp": time.strftime("%Y-%m-%d"),
        "data_source": "molmetal/data/platinai_4cell.csv (synthesised from platinai_5k + known_pt_drugs + tmQM)",
        "status": "MEASURED_SYNTHETIC_LABELS",
        "framing": (
            "synthetic IC50 labels for pipeline test; "
            "see build_platinai_4cell.py for honest framing"
        ),
        "split": "scaffold" if args.scaffold_split else "random",
        "model": "dmpnn_4cell_multitask",
    }
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    _wf416_atomic_write_text(metric_path, json.dumps(metric_payload, indent=2))
    print(f"[metric2] wrote {metric_path}", flush=True)

    # Markdown report
    report_path = (
        Path(args.out_report)
        if args.out_report
        else REPORTS_DIR / "wf_tier1_metric2_platinai" / "final.md"
    )
    _write_report(report_path, args, aggregate, per_seed)
    print(f"[metric2] wrote {report_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
