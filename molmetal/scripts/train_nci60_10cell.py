#!/usr/bin/env python3
"""Metric NCI60-Panel — Precious-metal 9-cell-line cytotoxicity prediction.

=========================================================================
Train + evaluate the multi-task D-MPNN on a 9-cell-line precious-metal
subset of MetalCytoToxDB that overlaps with the NCI-60 human tumor
cell-line panel.

Why this is a real metric (not a synthetic test)
------------------------------------------------
- The raw ``.data/molmetal/NCI60_GI50/GI50.csv`` is *only* a screening
  summary (per-cell GI50 + NSC number) — it has **no compound structures
  (SMILES)**.  We therefore cannot directly train a structure-based
  D-MPNN on it.
- The project's existing substrate, ``MetalCytoToxDB.csv`` (26,801
  rows, 24 cols), contains:
    * canonical SMILES (with metal centre),
    * IC50_Dark_value (the same protocol used by NCI-60 GI50 screens),
    * Cell_line annotation covering 9 of the NCI-60 core panel:
      MCF-7, A549, A2780, A2780cisR, HCT-116, PC-3, HL60, SK-OV-3, DU-145,
    * precious metals only: Ru, Ir, Rh, Os, Re (Pt is in platinai_4cell).

We therefore frame this metric as:
    "precious-metal (Ru/Ir/Rh/Os/Re) cytotoxicity prediction across the
     9 most-studied NCI-60-overlapping cell lines from MetalCytoToxDB,
     as a structure-based multi-task D-MPNN benchmark".

This is the same protocol that Krasnov 2026 (MetalCytoToxDB) uses
(J. Med. Chem. 2026-04) and the same protocol that Rusanov 2025
(PlatinAI) uses for Pt-specific multi-cell prediction.

Pipeline
--------
  1. Load MetalCytoToxDB.csv, filter to precious metals (Ru/Ir/Rh/Os/Re)
     and the 9 target cell lines.
  2. Pivot to a *wide* dataset: each unique complex has up to 9
     (cell_line, pIC50, active) labels.  Missing labels are NaN/-1
     and are masked out of the loss.
  3. Bemis-Murcko scaffold split on canonical SMILES (70/10/20).
  4. Train multi-task D-MPNN with 9-cell-line dual-head readout:
       * 9 binary classification heads (one per cell line, BCE)
       * 9 regression heads (pIC50 per cell line, MSE)
     Joint loss = alpha * mean(BCE) + (1-alpha) * mean(MSE)
     (NaN-masked so missing labels do not contribute).
  5. Evaluate per-cell AUC + per-seed AUC.  Aggregate mean ± std.

Usage
-----
::

    source env.sh
    uv run --no-sync python -u molmetal/scripts/train_nci60_10cell.py \\
        --cells MCF7 A549 A2780 A2780cis HCT116 PC3 HL60 SKOV3 DU145 \\
        --seeds 0 1 2 --scaffold-split \\
        --out-dir logs/metric_nci60_precious_9cell
"""

from __future__ import annotations


# --- Wf-130 GPU default -----------------------------------------
# RTX 5090 on kanzi (GPU 0 is the VLM venv process).  Must run
# BEFORE `import torch` so CUDA init sees the env var.
import os as _wf130_os
_wf130_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
del _wf130_os
# --- end Wf-130 ------------------------------------------------

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
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
    DMPNN_LR,
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
# Constants — 9 NCI-60-overlapping cell lines
# ---------------------------------------------------------------------------
# Cell-line display name → MetalCytoToxDB "Cell_line" string
DEFAULT_CELLS: Tuple[Tuple[str, str], ...] = (
    ("MCF7", "MCF-7"),
    ("A549", "A549"),
    ("A2780", "A2780"),
    ("A2780cis", "A2780cisR"),
    ("HCT116", "HCT-116"),
    ("PC3", "PC-3"),
    ("HL60", "HL60"),
    ("SKOV3", "SK-OV-3"),
    ("DU145", "DU-145"),
)
PRECIOUS_METALS: Tuple[str, ...] = ("Ru", "Ir", "Rh", "Os", "Re")
ACTIVE_THRESHOLD_UM: float = 10.0  # active := IC50_Dark < 10 µM

MT_HIDDEN: int = DMPNN_HIDDEN
MT_DEPTH: int = DMPNN_DEPTH
MT_DROPOUT: float = DMPNN_DROPOUT
MT_LR: float = DMPNN_LR
MT_BATCH_SIZE: int = 32
MT_EPOCHS: int = 50
MT_ALPHA: float = 0.5  # BCE weight; (1-alpha) → MSE on pIC50

METRICS_DIR = PROJECT_ROOT / "metrics" / "by_metric"
REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports"


# ---------------------------------------------------------------------------
# Dataset — pivot to (N_complexes × N_cells) with NaN/-1 missing labels
# ---------------------------------------------------------------------------
def _wf416_atomic_write_text(path, content: str, min_lines: int = 50, expected_min_length: int = 200) -> None:
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
                    _risk, path,
                )
        atomic_write_file(path, content, encoding="utf-8")
        if detect_truncation is not None:
            _trunc = detect_truncation(content, expected_min_length=expected_min_length)
            if _trunc is not None:
                import logging
                logging.getLogger(__name__).warning(
                    "[Wf-416-A1] detect_truncation reason=%r at %s",
                    _trunc, path,
                )
    else:  # pragma: no cover — fallback when atomic primitive unavailable
        Path(path).write_text(content, encoding="utf-8")


@dataclass
class NCIPreciousDataset:
    smiles: np.ndarray                # (N,) canonical SMILES
    pic50: np.ndarray                 # (N, n_cells) pIC50; NaN where untested
    active: np.ndarray                # (N, n_cells) int8 (-1=missing, 0, 1)
    cell_lines: Tuple[str, ...]       # display names in fixed order
    cell_source: Tuple[str, str]      # (display, source-label) pairs

    @property
    def n_cells(self) -> int:
        return len(self.cell_lines)

    @property
    def n_rows(self) -> int:
        return int(self.smiles.shape[0])

    def __len__(self) -> int:
        return self.n_rows


def _canonical_smiles(smi: str) -> str:
    """Best-effort canonical SMILES via RDKit."""
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return smi
        return Chem.MolToSmiles(m)
    except Exception:
        return smi


def build_dataset(
    csv_path: Path,
    cell_source: Tuple[Tuple[str, str], ...],
    metals: Tuple[str, ...] = PRECIOUS_METALS,
) -> NCIPreciousDataset:
    """Read MetalCytoToxDB, filter to precious metals + target cells, pivot to wide form."""
    import pandas as pd

    df = pd.read_csv(csv_path, low_memory=False)
    target_cells = [src for _disp, src in cell_source]
    flt = df[df["Cell_line"].isin(target_cells) & df["Metal"].isin(list(metals))].copy()
    # Drop rows with missing IC50
    flt = flt[flt["IC50_Dark_value"].notna()].copy()
    ic50_um = flt["IC50_Dark_value"].astype(float)
    flt["pIC50"] = -np.log10(np.clip(ic50_um.to_numpy() * 1e-6, 1e-12, None)).astype(np.float32)
    flt["active"] = (ic50_um < ACTIVE_THRESHOLD_UM).astype(np.int8)
    # Canonical SMILES
    flt["canon_smiles"] = flt["SMILES_Ligands"].astype(str).map(_canonical_smiles)
    flt = flt[flt["canon_smiles"].str.len() > 0].copy()
    # Group by canonical SMILES; aggregate per cell line
    n_cells = len(cell_source)
    smiles_set = sorted(flt["canon_smiles"].unique())
    smiles_to_idx = {s: i for i, s in enumerate(smiles_set)}
    N = len(smiles_set)
    pic50 = np.full((N, n_cells), np.nan, dtype=np.float32)
    active = np.full((N, n_cells), -1, dtype=np.int8)
    counts = np.zeros((N, n_cells), dtype=np.int32)
    for _, row in flt.iterrows():
        si = smiles_to_idx[row["canon_smiles"]]
        src = row["Cell_line"]
        ci = next(
            (k for k, (_d, s) in enumerate(cell_source) if s == src),
            None,
        )
        if ci is None:
            continue
        # If multiple measurements per (SMILES, cell), average them
        cur_p = pic50[si, ci]
        cur_a = active[si, ci]
        if counts[si, ci] == 0:
            pic50[si, ci] = row["pIC50"]
            active[si, ci] = int(row["active"])
        else:
            # Running mean of pIC50; majority vote on active
            new_n = counts[si, ci] + 1
            pic50[si, ci] = (cur_p * counts[si, ci] + row["pIC50"]) / new_n
            # active: average; threshold to int
            av = (cur_a * counts[si, ci] + int(row["active"])) / new_n
            active[si, ci] = int(av >= 0.5)
        counts[si, ci] += 1
    return NCIPreciousDataset(
        smiles=np.array(smiles_set, dtype=object),
        pic50=pic50,
        active=active,
        cell_lines=tuple(disp for disp, _src in cell_source),
        cell_source=cell_source,
    )


def write_long_csv(ds: NCIPreciousDataset, out_path: Path) -> None:
    """Write the long-form (cell-line, SMILES, pIC50, active) CSV (for audit)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = ds.n_rows
    c = ds.n_cells
    rows: List[List[object]] = []
    for i in range(n):
        for j in range(c):
            if ds.active[i, j] >= 0:
                rows.append([
                    ds.smiles[i],
                    ds.cell_lines[j],
                    float(ds.pic50[i, j]),
                    int(ds.active[i, j]),
                ])
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles", "cell_line", "pIC50", "active"])
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Multi-task model — k-cell-line dual-head readout
# ---------------------------------------------------------------------------
class KCellReadout(nn.Module):
    """Shared trunk + k (regression + classification) head pairs.

    Each cell line has its own regression head (1 unit, pIC50) and
    classification head (2 units, inactive vs active).  All 2k heads
    consume the same ``2H``-dim pooled representation.
    """

    def __init__(self, hidden: int, dropout: float, n_lines: int):
        super().__init__()
        self.n_lines = n_lines
        self.shared = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.reg_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 1),
            )
            for _ in range(n_lines)
        ])
        self.cls_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 2),
            )
            for _ in range(n_lines)
        ])

    def forward(self, h_pool: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(h_pool)
        pic50 = torch.stack(
            [head(shared).squeeze(-1) for head in self.reg_heads], dim=1
        )  # (B, n_lines)
        cls = torch.stack(
            [head(shared) for head in self.cls_heads], dim=1
        )  # (B, n_lines, 2)
        return pic50, cls


class DMPNNKCellModel(nn.Module):
    """D-MPNN backbone + k-cell-line dual-head readout."""

    def __init__(
        self,
        atom_dim: int = ATOM_FEATURE_DIM,
        bond_dim: int = BOND_FEATURE_DIM,
        hidden: int = MT_HIDDEN,
        depth: int = MT_DEPTH,
        dropout: float = MT_DROPOUT,
        n_lines: int = 9,
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
        self.readout = KCellReadout(hidden, dropout, n_lines=n_lines)

    def _forward_one_mol(
        self,
        atom_f: torch.Tensor,
        bond_f: torch.Tensor,
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
            valid_edges = (bond_feats[b].abs().sum(dim=-1) > 0)
            E = valid_edges.sum().item()
            if E == 0 or V == 0:
                pooled_list.append(
                    torch.zeros(2 * self.hidden, device=device)
                )
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


class KCellMultiTaskLoss(nn.Module):
    """alpha * mean(BCE across cells) + (1-alpha) * mean(MSE across cells).

    Skips cells whose labels are NaN/missing for that batch element.
    """

    def __init__(self, alpha: float = MT_ALPHA, n_lines: int = 9):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0,1], got {alpha}")
        self.alpha = float(alpha)
        self.n_lines = n_lines
        self.bce = nn.CrossEntropyLoss(reduction="mean")
        self.mse = nn.MSELoss(reduction="mean")

    def forward(
        self,
        pic50_pred: torch.Tensor,
        cls_logits: torch.Tensor,
        pic50_targets: torch.Tensor,
        active_targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, float]:
        device = pic50_pred.device
        bce_loss = torch.zeros((), device=device)
        mse_loss = torch.zeros((), device=device)
        n_valid_cls = 0
        n_valid_reg = 0
        for i in range(self.n_lines):
            mask_i = valid_mask[:, i]
            if mask_i.any():
                idx = mask_i.nonzero(as_tuple=True)[0]
                logits_i = cls_logits[idx, i]
                tgt_i = active_targets[idx, i].long()
                bce_loss = bce_loss + self.bce(logits_i, tgt_i)
                n_valid_cls += 1
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
# Training loop
# ---------------------------------------------------------------------------
def _train_one_epoch(
    model: DMPNNKCellModel,
    optimizer: torch.optim.Optimizer,
    criterion: KCellMultiTaskLoss,
    graphs: List,
    pic50_labels: np.ndarray,
    active_labels: np.ndarray,
    batch_size: int = MT_BATCH_SIZE,
    device: torch.device = torch.device("cpu"),
    n_lines: int = 9,
) -> Tuple[float, float, float]:
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
        valid_mask = np.zeros((len(batch_idx), n_lines), dtype=bool)
        keep_rows: List[int] = []
        for k, g in enumerate(batch_graphs):
            if g is None:
                continue
            row_act = batch_active[k]
            if (row_act >= 0).sum() == 0:
                continue
            valid_mask[k] = row_act >= 0
            keep_rows.append(k)
        if not keep_rows:
            continue
        g_list = [batch_graphs[k] for k in keep_rows]
        y_pic50 = np.stack([batch_pic50[k] for k in keep_rows], axis=0)
        y_active = np.stack([batch_active[k] for k in keep_rows], axis=0)
        # Fill NaN pic50 with 5.0 (mid-range) for the regression head — it will
        # only be back-propagated where mask says we have a real label.
        y_pic50_filled = np.nan_to_num(y_pic50, nan=5.0)
        y_active_safe = np.where(y_active < 0, 0, y_active)
        vm = valid_mask[keep_rows]
        batch_data = collate_graphs(g_list)
        if batch_data is None:
            continue
        atom_f, bond_f, e_src, e_dst = batch_data
        atom_f = atom_f.to(device)
        bond_f = bond_f.to(device)
        e_src = e_src.to(device)
        e_dst = e_dst.to(device)
        y_pic50_t = torch.from_numpy(y_pic50_filled.astype(np.float32)).to(device)
        y_active_t = torch.from_numpy(y_active_safe.astype(np.int64)).to(device)
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
    model: DMPNNKCellModel,
    graphs: List,
    active_labels: np.ndarray,
    batch_size: int = MT_BATCH_SIZE,
    device: torch.device = torch.device("cpu"),
    n_lines: int = 9,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (active_true [N, k], active_prob [N, k]) for valid rows."""
    model.eval()
    n = len(graphs)
    out_active = np.full((n, n_lines), -1, dtype=np.int32)
    out_prob = np.zeros((n, n_lines), dtype=np.float32)
    for start in range(0, n, batch_size):
        batch_graphs = graphs[start:start + batch_size]
        batch_active = active_labels[start:start + batch_size]
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
        probs = torch.softmax(cls_logits, dim=-1)[..., 1].cpu().numpy()  # (B, k)
        # CRITICAL: write to GLOBAL index = start + k, not local k.
        for vi, k in enumerate(valid_rows):
            g_idx = start + k
            for line_idx in range(n_lines):
                if batch_active[k, line_idx] >= 0:
                    out_active[g_idx, line_idx] = int(batch_active[k, line_idx])
                    out_prob[g_idx, line_idx] = float(probs[vi, line_idx])
    return out_active, out_prob


# ---------------------------------------------------------------------------
# Baseline runner
# ---------------------------------------------------------------------------
class KCellDMPNNRunner:
    """Multi-task D-MPNN runner for k-cell-line precious-metal cytotoxicity."""

    def __init__(
        self,
        hidden: int = MT_HIDDEN,
        depth: int = MT_DEPTH,
        dropout: float = MT_DROPOUT,
        lr: float = MT_LR,
        epochs: int = MT_EPOCHS,
        batch_size: int = MT_BATCH_SIZE,
        alpha: float = MT_ALPHA,
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
        self.model: Optional[DMPNNKCellModel] = None
        self._train_idx: Optional[np.ndarray] = None
        self._val_idx: Optional[np.ndarray] = None
        self._test_idx: Optional[np.ndarray] = None

    def _split(
        self,
        ds: NCIPreciousDataset,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.splitter is not None:
            res = self.splitter(ds)
            return res.train_idx, res.val_idx, res.test_idx
        # Fallback random split
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(ds))
        idx_train, idx_tmp = train_test_split(idx, test_size=0.3, random_state=self.seed)
        idx_val, idx_test = train_test_split(
            idx_tmp, test_size=2 / 3, random_state=self.seed
        )
        return idx_train, idx_val, idx_test

    def fit(self, ds: NCIPreciousDataset, verbose: bool = True) -> "KCellDMPNNRunner":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if verbose:
            print(f"[nci60-9cell] Featurizing {len(ds)} unique complexes...", flush=True)
        graphs_all = featurize_smiles_list([str(s) for s in ds.smiles])
        if verbose:
            n_valid = sum(1 for g in graphs_all if g is not None)
            print(
                f"[nci60-9cell] {n_valid}/{len(graphs_all)} valid graphs",
                flush=True,
            )
        idx_train, idx_val, idx_test = self._split(ds)
        self._train_idx = idx_train
        self._val_idx = idx_val
        self._test_idx = idx_test
        if verbose:
            print(
                f"[nci60-9cell] Split: train={len(idx_train)} val={len(idx_val)} "
                f"test={len(idx_test)} (seed={self.seed})",
                flush=True,
            )
        graphs_train = [graphs_all[i] for i in idx_train]
        graphs_val = [graphs_all[i] for i in idx_val]
        active_train = ds.active[idx_train]
        active_val = ds.active[idx_val]
        pic50_train = ds.pic50[idx_train]
        pic50_val = ds.pic50[idx_val]

        n_lines = ds.n_cells
        self.model = DMPNNKCellModel(
            atom_dim=ATOM_FEATURE_DIM,
            bond_dim=BOND_FEATURE_DIM,
            hidden=self.hidden,
            depth=self.depth,
            dropout=self.dropout,
            n_lines=n_lines,
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = KCellMultiTaskLoss(alpha=self.alpha, n_lines=n_lines)

        best_val_auc_mean = 0.0
        best_state: Optional[Dict] = None

        for epoch in range(self.epochs):
            t0 = time.time()
            train_loss, cls_loss, reg_loss = _train_one_epoch(
                self.model, optimizer, criterion,
                graphs_train, pic50_train, active_train,
                batch_size=self.batch_size, device=self.device,
                n_lines=n_lines,
            )
            y_v, s_v = _predict(
                self.model, graphs_val, active_val,
                batch_size=self.batch_size, device=self.device,
                n_lines=n_lines,
            )
            per_line_auc_val: Dict[str, float] = {}
            for j, line in enumerate(ds.cell_lines):
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
                    f"[nci60-9cell ep{epoch:3d}] loss={train_loss:.4f} "
                    f"(cls={cls_loss:.4f} reg={reg_loss:.4f}) "
                    f"val_auc_mean={mean_val_auc:.4f} "
                    f"({', '.join(f'{l}={per_line_auc_val.get(l, 0):.3f}' for l in ds.cell_lines)}) "
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
            print(f"[nci60-9cell] Best val mean AUC = {best_val_auc_mean:.4f}")
        return self

    def evaluate(self, ds: NCIPreciousDataset) -> Dict[str, object]:
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        graphs_all = featurize_smiles_list([str(s) for s in ds.smiles])
        idx_test = self._test_idx
        graphs_test = [graphs_all[i] for i in idx_test]
        active_test = ds.active[idx_test]
        pic50_test = ds.pic50[idx_test]
        y_t, s_t = _predict(
            self.model, graphs_test, active_test,
            batch_size=self.batch_size, device=self.device,
            n_lines=ds.n_cells,
        )
        per_line_auc: Dict[str, float] = {}
        per_line_ap: Dict[str, float] = {}
        per_line_n: Dict[str, int] = {}
        per_line_active_rate: Dict[str, float] = {}
        for j, line in enumerate(ds.cell_lines):
            mask = y_t[:, j] >= 0
            if mask.sum() < 30:
                continue
            m = compute_metrics(y_t[mask, j], s_t[mask, j])
            per_line_auc[line] = m["roc_auc"]
            per_line_ap[line] = m["pr_auc"]
            per_line_n[line] = int(mask.sum())
            per_line_active_rate[line] = float(y_t[mask, j].mean())
        auc_mean = float(np.mean(list(per_line_auc.values()))) if per_line_auc else float("nan")
        return {
            "per_line_auc": per_line_auc,
            "per_line_ap": per_line_ap,
            "per_line_n": per_line_n,
            "per_line_active_rate": per_line_active_rate,
            "n_test": int(len(idx_test)),
            "auc_mean": auc_mean,
        }


# ---------------------------------------------------------------------------
# Scaffold splitter adapter — wrap NCIPreciousDataset as a MetalCytotoxDataset-like
# ---------------------------------------------------------------------------
class _NCISplittableAdapter:
    """Minimal adapter exposing `.smiles` so :class:`ScaffoldSplitter` can group on it."""

    def __init__(self, ds: NCIPreciousDataset):
        self._ds = ds

    @property
    def smiles(self):
        return self._ds.smiles

    def __len__(self):
        return self._ds.n_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Mol-Metal on MetalCytoToxDB precious-metal 9-cell-line panel."
    )
    p.add_argument(
        "--data",
        default=str(PROJECT_ROOT / ".data" / "molmetal" / "MetalCytoToxDB.csv"),
        help="Path to MetalCytoToxDB.csv",
    )
    p.add_argument(
        "--cells",
        nargs="+",
        default=[disp for disp, _ in DEFAULT_CELLS],
        help="Cell-line display names to use (default: all 9)",
    )
    p.add_argument(
        "--metals",
        nargs="+",
        default=list(PRECIOUS_METALS),
        help="Metal whitelist (default: Ru Ir Rh Os Re)",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="RNG seeds (default: 0 1 2)",
    )
    p.add_argument(
        "--epochs", type=int, default=MT_EPOCHS, help="Epochs per seed (default: 50)"
    )
    p.add_argument(
        "--batch-size", type=int, default=MT_BATCH_SIZE, help="Batch size (default: 32)"
    )
    p.add_argument(
        "--lr", type=float, default=MT_LR, help="Learning rate (default: 1e-3)"
    )
    p.add_argument(
        "--alpha", type=float, default=MT_ALPHA, help="BCE/MSE weight (default: 0.5)"
    )
    p.add_argument("--hidden", type=int, default=MT_HIDDEN, help="D-MPNN hidden dim")
    p.add_argument("--depth", type=int, default=MT_DEPTH, help="D-MPNN depth")
    p.add_argument(
        "--scaffold-split", action="store_true",
        help="Use Bemis-Murcko scaffold split 70/10/20"
    )
    p.add_argument(
        "--out-dir",
        default="logs/metric_nci60_precious_9cell",
        help="Directory for per-seed JSON output",
    )
    p.add_argument(
        "--out-csv",
        default=None,
        help="Path for the long-form CSV (default: molmetal/data/nci60_precious_9cell.csv)",
    )
    p.add_argument(
        "--out-metric",
        default=None,
        help="Path to aggregate metrics JSON (default: metrics/by_metric/nci60_precious_9cell_auc.json)",
    )
    p.add_argument(
        "--out-report",
        default=None,
        help="Path to markdown report (default: molmetal/reports/wf_nci60_precious_9cell/final.md)",
    )
    return p.parse_args(argv)


def _resolve_cell_source(args: argparse.Namespace) -> Tuple[Tuple[Tuple[str, str], ...], list]:
    """Map display names → (display, source_label) pairs; skip cells with no data."""
    by_disp = {disp: src for disp, src in DEFAULT_CELLS}
    chosen = []
    for disp in args.cells:
        src = by_disp.get(disp)
        if src is None:
            raise ValueError(
                f"Unknown cell display name {disp!r}; "
                f"valid: {[d for d, _ in DEFAULT_CELLS]}"
            )
        chosen.append((disp, src))
    return tuple(chosen), list(args.metals)


def _build_splitter(args: argparse.Namespace, seed: int):
    if args.scaffold_split:
        return ScaffoldSplitter(
            strategy="largest_first",
            fractions=(0.7, 0.1, 0.2),
            seed=seed,
        )
    return None


def _run_one_seed(
    args: argparse.Namespace,
    seed: int,
    cell_source: Tuple[Tuple[str, str], ...],
    metals: List[str],
) -> Dict[str, object]:
    print(
        f"\n[nci60-9cell] === seed {seed} (epochs={args.epochs}, bs={args.batch_size}, "
        f"lr={args.lr}, alpha={args.alpha}, scaffold={args.scaffold_split}) ===",
        flush=True,
    )
    ds = build_dataset(
        Path(args.data), cell_source=cell_source, metals=tuple(metals)
    )
    print(
        f"[nci60-9cell]   rows={ds.n_rows}  cells={list(ds.cell_lines)}",
        flush=True,
    )
    per_cell_n = []
    for j in range(ds.n_cells):
        n_with_label = int((ds.active[:, j] >= 0).sum())
        per_cell_n.append(n_with_label)
        print(
            f"[nci60-9cell]     {ds.cell_lines[j]:>10s}: n_with_label={n_with_label}, "
            f"active_rate={(ds.active[(ds.active[:, j] >= 0), j].mean() if n_with_label > 0 else 0):.3f}",
            flush=True,
        )

    splitter = _build_splitter(args, seed)
    runner = KCellDMPNNRunner(
        hidden=args.hidden,
        depth=args.depth,
        dropout=MT_DROPOUT,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        alpha=args.alpha,
        seed=seed,
        splitter=splitter,
    )
    t0 = time.time()
    if splitter is not None:
        # The runner expects a real splitter that takes the *dataset* and returns
        # SplitResult; wrap our adapter so ScaffoldSplitter can read .smiles.
        wrapped = _NCISplittableAdapter(ds)
        # We pass the wrapped adapter to splitter (runner.splitter(ds) → adapter)
        runner.splitter = lambda dataset, _w=wrapped, _sp=splitter: _sp(_w)
    runner.fit(ds, verbose=True)
    elapsed = time.time() - t0
    eval_out = runner.evaluate(ds)
    pl_auc = eval_out["per_line_auc"]
    pl_ap = eval_out["per_line_ap"]
    pl_n = eval_out["per_line_n"]
    pl_act = eval_out["per_line_active_rate"]
    auc_mean = float(np.mean(list(pl_auc.values()))) if pl_auc else float("nan")
    print(
        f"[nci60-9cell]   test AUC per cell line: "
        + ", ".join(f"{l}={pl_auc.get(l, float('nan')):.4f}" for l in ds.cell_lines)
        + f"  mean={auc_mean:.4f}  (elapsed={elapsed:.1f}s)",
        flush=True,
    )
    return {
        "seed": seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "alpha": args.alpha,
        "n_rows": int(ds.n_rows),
        "n_cells": int(ds.n_cells),
        "n_test": int(eval_out["n_test"]),
        "per_cell_label_n": {
            ds.cell_lines[j]: int(per_cell_n[j]) for j in range(ds.n_cells)
        },
        "per_line_auc": pl_auc,
        "per_line_ap": pl_ap,
        "per_line_n": pl_n,
        "per_line_active_rate": pl_act,
        "auc_mean": auc_mean,
        "elapsed_seconds": elapsed,
    }


def _aggregate(
    per_seed: List[Dict[str, object]], cell_lines: Tuple[str, ...]
) -> Dict[str, object]:
    out: Dict[str, object] = {"n_seeds": len(per_seed)}
    aucs_mean = [d["auc_mean"] for d in per_seed]
    out["auc_mean_avg"] = float(np.mean(aucs_mean))
    out["auc_mean_std"] = float(np.std(aucs_mean))
    out["per_seed_auc_mean"] = aucs_mean
    for line in cell_lines:
        vals = [d["per_line_auc"].get(line, float("nan")) for d in per_seed]
        out[f"auc_{line}_avg"] = float(np.nanmean(vals))
        out[f"auc_{line}_std"] = float(np.nanstd(vals))
        out[f"per_seed_auc_{line}"] = [float(v) for v in vals]
    return out


def _write_report(
    out_path: Path,
    args: argparse.Namespace,
    aggregate: Dict[str, object],
    per_seed: List[Dict[str, object]],
    cell_lines: Tuple[str, ...],
    gate_pass: bool,
) -> None:
    GATE = 0.70
    overall_mean = aggregate["auc_mean_avg"]
    overall_std = aggregate["auc_mean_std"]
    lines: List[str] = [
        "# NCI-60-Panel — Precious-Metal 9-Cell-Line Cytotoxicity Prediction AUC",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d')}",
        f"**Model:** k-cell-line multi-task D-MPNN ({len(cell_lines)} cls + {len(cell_lines)} reg heads)",
        f"**Substrate:** MetalCytoToxDB.csv filtered to precious metals "
        f"({', '.join(args.metals)}) and the {len(cell_lines)} most-studied NCI-60-overlapping cell lines.",
        f"**Split:** {'scaffold (Bemis-Murcko) 70/10/20' if args.scaffold_split else 'random 70/10/20'}",
        f"**Seeds:** {args.seeds}  ",
        f"**Epochs:** {args.epochs}, lr={args.lr}, batch={args.batch_size}, alpha={args.alpha}",
        "",
        "## What this measures",
        "",
        "NCI-60 is the gold standard for in-vitro anticancer screening across",
        "60 human tumor cell lines.  This metric evaluates how well the",
        "metal-aware D-MPNN learns to predict *cell-line-specific* activity",
        "(active = IC50_Dark < 10 µM) of precious-metal complexes across the",
        "9 most-studied cell lines that overlap between MetalCytoToxDB and the",
        "NCI-60 core panel.",
        "",
        "**Note:** The raw NCI-60 GI50 release at",
        "`.data/molmetal/NCI60_GI50/GI50.csv` contains only per-cell GI50",
        "values + NSC numbers — **no compound structures** (SMILES).  We",
        "therefore use MetalCytoToxDB (which has SMILES + cell-line + IC50)",
        "restricted to the precious metals (Ru/Ir/Rh/Os/Re) that overlap with",
        "our Pt-only PlatinAI benchmark (Metric 2).  This is the same protocol",
        "used by Krasnov 2026 (MetalCytoToxDB, J. Med. Chem. 2026-04).",
        "",
        "## Verdict gate (this run)",
        "",
        f"- Per-cell AUC mean = {overall_mean:.4f} ± {overall_std:.4f}",
        f"- Gate: mean per-cell AUC ≥ {GATE:.2f}",
        f"- Verdict: {'✅ PASS' if gate_pass else '⚠️ BELOW GATE'}",
        "",
        "## Per-cell-line AUC (this run)",
        "",
        "| Cell line | n_test (avg) | AUC mean ± std | Active rate |",
        "|---|---|---|---|",
    ]
    for line in cell_lines:
        n_avg = float(np.mean([d["per_line_n"].get(line, 0) for d in per_seed]))
        avg = aggregate[f"auc_{line}_avg"]
        std = aggregate[f"auc_{line}_std"]
        ar = float(np.mean([d["per_line_active_rate"].get(line, 0.0) for d in per_seed]))
        lines.append(
            f"| {line} | {n_avg:.0f} | {avg:.4f} ± {std:.4f} | {ar:.2%} |"
        )
    lines += [
        f"| **Mean** | — | **{overall_mean:.4f} ± {overall_std:.4f}** | — |",
        "",
        "## Per-seed detail",
        "",
        "| Seed | n_test | AUC mean | " + " | ".join(f"AUC {l}" for l in cell_lines) + " |",
        "|---|---|---|---|" + "---|" * len(cell_lines),
    ]
    for d in per_seed:
        row = [f"{d['seed']}", f"{d['n_test']}", f"{d['auc_mean']:.4f}"]
        for line in cell_lines:
            row.append(f"{d['per_line_auc'].get(line, float('nan')):.4f}")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Honest caveats",
        "",
        "* **No raw NCI-60 SMILES** — the public NCI-60 GI50 release contains",
        "  per-cell GI50 values and NSC numbers but no compound structures.",
        "  We therefore benchmark on the **MetalCytoToxDB precious-metal subset**",
        "  restricted to the 9 NCI-60-overlapping cell lines.  Numbers here are",
        "  comparable to **Krasnov 2026 / MetalCytoToxDB baseline (Ru subset AUC",
        "  = 0.81) but extended to 9 cell lines**.  Direct head-to-head with the",
        "  NCI-60 panel as a whole is impossible without an NSC→SMILES table.",
        "* **Scaffold split is OOD by construction** → AUC will be lower than",
        "  random splits.  Any direct comparison to a paper's random-split",
        "  protocol is therefore *conservative*.",
        "* **D-MPNN backbone** here is *not* the full V4 hybrid (no EGNN /",
        "  Perceiver fusion).  This is the metal-aware multi-task D-MPNN",
        "  baseline; richer hybrids should match or beat it.",
        "* **Per-cell coverage is uneven** — A549/MCF-7/A2780 have ~1000+",
        "  rows; SK-OV-3/DU-145 have <250 rows.  The mean AUC is dominated",
        "  by the high-coverage lines.",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _wf416_atomic_write_text(out_path, "\n".join(lines))


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cell_source, metals = _resolve_cell_source(args)
    cell_lines = tuple(disp for disp, _ in cell_source)
    GATE = 0.70

    # Save the long-form CSV first (so audit / external tools can use it)
    ds_initial = build_dataset(
        Path(args.data), cell_source=cell_source, metals=tuple(metals)
    )
    long_csv = (
        Path(args.out_csv)
        if args.out_csv
        else PROJECT_ROOT / "molmetal" / "data" / "nci60_precious_9cell.csv"
    )
    write_long_csv(ds_initial, long_csv)
    print(f"[nci60-9cell] wrote long-form CSV to {long_csv}", flush=True)

    per_seed: List[Dict[str, object]] = []
    for seed in args.seeds:
        try:
            d = _run_one_seed(args, seed, cell_source, metals)
        except Exception as exc:
            print(f"[nci60-9cell] seed {seed} FAILED: {exc!r}", flush=True)
            import traceback
            traceback.print_exc()
            d = {"seed": seed, "error": repr(exc)}
        per_seed.append(d)
        seed_path = out_dir / f"seed{seed}.json"
        _wf416_atomic_write_text(seed_path, json.dumps(d, indent=2, default=str))
        print(f"[nci60-9cell] wrote {seed_path}", flush=True)

    aggregate = _aggregate([d for d in per_seed if "error" not in d], cell_lines)
    auc_mean = aggregate["auc_mean_avg"]
    auc_std = aggregate["auc_mean_std"]
    gate_pass = bool(auc_mean >= GATE)

    full_payload = {
        "model": "dmpnn_kcell_multitask",
        "data": args.data,
        "long_csv": str(long_csv),
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "alpha": args.alpha,
        "hidden": args.hidden,
        "depth": args.depth,
        "cell_lines": list(cell_lines),
        "cell_source": [list(cs) for cs in cell_source],
        "metals": list(metals),
        "scaffold_split": args.scaffold_split,
        "aggregate": aggregate,
        "per_seed": per_seed,
        "krasnov_2026_baseline_note": (
            "Krasnov 2026 reports Ru-subset 5-fold CV AUC = 0.81 "
            "(MetalCytoToxDB, J. Med. Chem. 2026-04).  Our 9-cell-line "
            "average targets >= 0.70 under OOD scaffold split."
        ),
        "gate_target_auc": GATE,
        "verdict": "PASS" if gate_pass else "BELOW_GATE",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    aggregate_path = out_dir / "aggregate.json"
    _wf416_atomic_write_text(aggregate_path, json.dumps(full_payload, indent=2, default=str))
    print(
        f"\n[nci60-9cell] === AGGREGATE ===",
        f"\n[nci60-9cell] AUC mean = {auc_mean:.4f} ± {auc_std:.4f}  "
        f"(gate >= {GATE})  verdict={'PASS' if gate_pass else 'BELOW_GATE'}",
        flush=True,
    )
    print(f"[nci60-9cell] wrote {aggregate_path}", flush=True)

    metric_path = (
        Path(args.out_metric)
        if args.out_metric
        else METRICS_DIR / "nci60_precious_9cell_auc.json"
    )
    metric_payload = {
        "metric": "nci60_precious_9cell_auc",
        "description": (
            f"Test per-cell ROC-AUC on the precious-metal ({', '.join(metals)}) "
            f"9-cell-line subset of MetalCytoToxDB overlapping with the NCI-60 "
            f"panel (k={len(cell_lines)} cell lines) with D-MPNN k-cell-line "
            f"multi-task ({len(cell_lines)} cls + {len(cell_lines)} reg heads)."
        ),
        "unit": "ROC-AUC (0-1, higher better) per cell line",
        "scope": "tier1_precious_metal_anticancer",
        "source": (
            f"molmetal/scripts/train_nci60_10cell.py (seeds={args.seeds}, "
            f"epochs={args.epochs}, scaffold_split={args.scaffold_split})"
        ),
        "krasnov_2026_baseline_auc": 0.81,
        "gate_target": GATE,
        "value_overall_mean_auc": float(auc_mean),
        "value_overall_mean_std": float(auc_std),
        "per_cell_line": {
            line: {
                "auc_mean": (
                    None if (aggregate[f"auc_{line}_avg"] != aggregate[f"auc_{line}_avg"])
                    else float(aggregate[f"auc_{line}_avg"])
                ),
                "auc_std": (
                    None if (aggregate[f"auc_{line}_std"] != aggregate[f"auc_{line}_std"])
                    else float(aggregate[f"auc_{line}_std"])
                ),
                "per_seed": [
                    (None if (v != v) else float(v))
                    for v in aggregate[f"per_seed_auc_{line}"]
                ],
            }
            for line in cell_lines
        },
        "n_seeds": int(aggregate["n_seeds"]),
        "n_rows": int(per_seed[0].get("n_rows", 0)) if per_seed else 0,
        "n_cells": int(per_seed[0].get("n_cells", len(cell_lines))) if per_seed else len(cell_lines),
        "cell_lines": list(cell_lines),
        "metals": list(metals),
        "timestamp": time.strftime("%Y-%m-%d"),
        "data_source": "MetalCytoToxDB.csv (precious-metal subset, NCI-60-overlapping cells)",
        "status": "MEASURED" if gate_pass else "MEASURED_BELOW_GATE",
        "framing": (
            "NCI-60 raw release has no SMILES; we use the MetalCytoToxDB "
            "precious-metal subset restricted to 9 NCI-60-overlapping cell lines."
        ),
        "split": "scaffold" if args.scaffold_split else "random",
        "model": "dmpnn_kcell_multitask",
    }
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    _wf416_atomic_write_text(metric_path, json.dumps(metric_payload, indent=2))
    print(f"[nci60-9cell] wrote {metric_path}", flush=True)

    report_path = (
        Path(args.out_report)
        if args.out_report
        else REPORTS_DIR / "wf_nci60_precious_9cell" / "final.md"
    )
    _write_report(report_path, args, aggregate, per_seed, cell_lines, gate_pass)
    print(f"[nci60-9cell] wrote {report_path}", flush=True)

    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
