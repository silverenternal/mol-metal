#!/usr/bin/env python3
"""Pretrain a metal-aware D-MPNN encoder on the tmQM 108k corpus.

=====================================================================
tmQM pretraining for Mol-Metal (Metric 22 + R17 close-out)
=====================================================================

Background
----------
Balcells & Skjelstad 2020 (tmQM, J. Chem. Inf. Model. 60, 6135-6146;
the release shipped here is the extended 108,541-row version) provides
~108k mononuclear transition-metal complexes with DFT-computed
properties at TPSSh-D3BJ/def2-SVP.  The fields used as pretraining
targets are:

* HOMO_Energy          — regression (eV)
* LUMO_Energy          — regression (eV)
* HL_Gap               — regression (eV)
* metal_bo_total       — regression (Wiberg bond order of metal centre)
* coord_number (= MND) — regression (coordination number of metal)
* charge               — classification (oxidation state, range -2..+6)

The D-MPNN backbone (atom_emb, bond_emb, edge_init, gru_update) is
trained jointly on all 6 heads for 30 epochs at lr=1e-3, batch=128.
After pretraining, only the backbone weights are saved to
``models/tmqm_pretrained_encoder.pt`` so they can be fine-tuned on
downstream tasks (MetalCytoToxDB, PlatinAI, NCI-60).

Encoder-only export
-------------------
The saved checkpoint contains:

* ``state_dict`` — backbone state dict (atom_emb, bond_emb, edge_init,
  gru_update)
* ``config`` — D-MPNN hyper-params (hidden=128, depth=3, dropout=0.1)
* ``train_meta`` — train RMSE / MAE / classification accuracy per
  head, dataset split sizes, wall time.

This file is consumed by:

* ``molmetal/scripts/finetune_metalcytotox_from_tmqm.py`` (R17+) for
  MetalCytoToxDB Ru subset fine-tune ablation.
* ``TODO/pending/23_weak_to_strong_plan.md`` metric 22 (CFM-vs-Lambda
  via R12 Deflex).

Usage
-----
::

    source env.sh
    uv run --no-sync python -u molmetal/scripts/pretrain_tmqm.py \\
        --data .data/molmetal/tmQM/tmqm_parsed.csv \\
        --out-dir models/tmqm_pretrained \\
        --epochs 30 --batch 128 --lr 1e-3 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
    _GRUUpdate,
    featurize_smiles_list,
)

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
TMQM_REGRESSION_TARGETS: Tuple[str, ...] = (
    "HOMO_Energy",
    "LUMO_Energy",
    "HL_Gap",
    "metal_bo_total",
    "coord_number",
)
TMQM_CLASSIFICATION_TARGETS: Tuple[str, ...] = ("charge",)

# Oxidation-state binning: tmQM charges range from about -2 (electron-rich
# Mn carbonyl anions) to +4 (early TM high-valent). We map -2..+6 onto
# 9 ordinal bins -2,-1,0,1,2,3,4,5,6 (offset = +2).
OX_STATES: Tuple[int, ...] = (-2, -1, 0, 1, 2, 3, 4, 5, 6)
N_OX_BINS: int = len(OX_STATES)


def charge_to_bin(c: int) -> int:
    """Map a tmQM charge value (-2..+6 typical) to a bin index."""
    for i, s in enumerate(OX_STATES):
        if c == s:
            return i
    # Default: nearest neighbour
    return min(range(N_OX_BINS), key=lambda i: abs(OX_STATES[i] - c))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@dataclass
class TmQMDataset:
    """Filtered tmQM rows ready for featurisation."""

    smiles: List[str]
    reg_targets: np.ndarray  # (N, 5)  HOMO/LUMO/gap/bo_total/coord_num
    cls_targets: np.ndarray  # (N,)    oxidation-state bin

    @property
    def n(self) -> int:
        return len(self.smiles)


def load_tmqm(csv_path, require_smiles: bool = True) -> TmQMDataset:
    """Read tmqm_parsed.csv (or compatible) and produce the pretrain tensors.

    Columns expected (per the loader in ``molmetal/data/tmqm.py``):
        0  csd_code
        1  stoichiometry
        2  metal
        3  charge            <- classification target
        4  spin
        5  coord_number      <- regression target
        6  metal_bo_total    <- regression target
        7  metal_bo_sum
        8  bo_n_neighbors
        ...
        13 HL_Gap            <- regression target
        14 HOMO_Energy       <- regression target
        15 LUMO_Energy       <- regression target
        ...
        18 smiles            <- required
    """
    csv_path = Path(csv_path)
    smiles: List[str] = []
    reg: List[List[float]] = []
    cls: List[int] = []
    with csv_path.open("r", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        # Resolve target column indices from header (defensive)
        col = {name: i for i, name in enumerate(header)}
        ix_smiles = col.get("smiles", 18)
        ix_charge = col.get("charge", 3)
        ix_coord = col.get("coord_number", 5)
        ix_bo_total = col.get("metal_bo_total", 6)
        ix_gap = col.get("HL_Gap", 13)
        ix_homo = col.get("HOMO_Energy", 14)
        ix_lumo = col.get("LUMO_Energy", 15)
        for row in reader:
            if not row:
                continue
            smi = row[ix_smiles].strip() if ix_smiles < len(row) else ""
            if require_smiles and not smi:
                continue
            try:
                charge = int(float(row[ix_charge]))
                coord = float(row[ix_coord])
                bo_total = float(row[ix_bo_total])
                gap = float(row[ix_gap])
                homo = float(row[ix_homo])
                lumo = float(row[ix_lumo])
            except (ValueError, IndexError):
                continue
            # Drop rows with NaN/Inf targets
            if any(
                math.isnan(v) or math.isinf(v)
                for v in (coord, bo_total, homo, lumo, gap)
            ):
                continue
            smiles.append(smi)
            reg.append([homo, lumo, gap, bo_total, coord])
            cls.append(charge_to_bin(charge))
    return TmQMDataset(
        smiles=smiles,
        reg_targets=np.asarray(reg, dtype=np.float32),
        cls_targets=np.asarray(cls, dtype=np.int64),
    )


def stratified_split(
    cls_targets: np.ndarray, val_frac: float, test_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split by oxidation-state bin so each split has a similar
    class distribution."""
    rng = np.random.default_rng(seed)
    n = cls_targets.shape[0]
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for c in sorted(set(cls_targets.tolist())):
        cls_idx = np.where(cls_targets == c)[0]
        rng.shuffle(cls_idx)
        n_test = max(1, int(round(cls_idx.shape[0] * test_frac)))
        n_val = max(1, int(round(cls_idx.shape[0] * val_frac)))
        test_idx.extend(cls_idx[:n_test].tolist())
        val_idx.extend(cls_idx[n_test : n_test + n_val].tolist())
        train_idx.extend(cls_idx[n_test + n_val :].tolist())
    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Model — D-MPNN backbone + 5 regression heads + 1 classification head
# ---------------------------------------------------------------------------
class TmqmPretrainModel(nn.Module):
    """D-MPNN backbone shared by tmQM heads."""

    def __init__(
        self,
        atom_dim: int = ATOM_FEATURE_DIM,
        bond_dim: int = BOND_FEATURE_DIM,
        hidden: int = DMPNN_HIDDEN,
        depth: int = DMPNN_DEPTH,
        dropout: float = DMPNN_DROPOUT,
        n_reg: int = len(TMQM_REGRESSION_TARGETS),
        n_cls: int = N_OX_BINS,
    ):
        super().__init__()
        self.hidden = hidden
        self.depth = depth
        self.n_reg = n_reg
        self.n_cls = n_cls

        # ---- backbone (encoder) ---------------------------------------
        self.dropout_layer = nn.Dropout(dropout)
        self.atom_emb = nn.Linear(atom_dim, hidden)
        self.bond_emb = nn.Linear(bond_dim, hidden)
        self.gru_update = _GRUUpdate(hidden, hidden)
        self.edge_init = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
        )

        # ---- heads ----------------------------------------------------
        self.reg_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden * 2, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, 1),
                )
                for _ in range(n_reg)
            ]
        )
        self.cls_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_cls),
        )

    def _encode_one(
        self,
        atom_f: torch.Tensor,
        bond_f: torch.Tensor,
        e_src: torch.Tensor,
        e_dst: torch.Tensor,
    ) -> torch.Tensor:
        """Forward backbone for a single molecule, returns 2H-dim pooled."""
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
        return h_comb.sum(dim=0)  # (2H,)

    def forward(
        self,
        atom_feats: torch.Tensor,  # (B, V, A)
        bond_feats: torch.Tensor,  # (B, E, B_dim)
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (reg_preds (B, n_reg), cls_logits (B, n_cls))."""
        B = atom_feats.size(0)
        device = atom_feats.device
        pooled_list: List[torch.Tensor] = []
        for b in range(B):
            V = atom_feats.size(1)
            valid_edges = bond_feats[b].abs().sum(dim=-1) > 0
            E = int(valid_edges.sum().item())
            if E == 0 or V == 0:
                pooled_list.append(torch.zeros(2 * self.hidden, device=device))
                continue
            pooled_list.append(
                self._encode_one(
                    atom_feats[b, :V],
                    bond_feats[b, :E],
                    edge_src[b, :E],
                    edge_dst[b, :E],
                )
            )
        h_pool = torch.stack(pooled_list, dim=0)  # (B, 2H)

        reg_outs = torch.stack(
            [head(h_pool).squeeze(-1) for head in self.reg_heads],
            dim=1,
        )  # (B, n_reg)
        cls_logits = self.cls_head(h_pool)  # (B, n_cls)
        return reg_outs, cls_logits


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------
def collate_graphs(
    graphs: List[Optional[Tuple]],
    max_atoms: int,
    max_edges: int,
    atom_dim: int,
    bond_dim: int,
):
    """Pad a list of graphs into (atom_feats, bond_feats, e_src, e_dst).

    Returns ``None`` if every molecule is invalid.
    """
    valid = [g for g in graphs if g is not None]
    if not valid:
        return None

    B = len(graphs)
    atom_feats = np.zeros((B, max_atoms, atom_dim), dtype=np.float32)
    bond_feats = np.zeros((B, max_edges, bond_dim), dtype=np.float32)
    e_src = np.zeros((B, max_edges), dtype=np.int64)
    e_dst = np.zeros((B, max_edges), dtype=np.int64)

    for b, g in enumerate(graphs):
        if g is None:
            continue
        af, bf, es, ed = g
        V = af.shape[0]
        E = bf.shape[0]
        if V > max_atoms:
            V = max_atoms
            af = af[:V]
        if E > max_edges:
            E = max_edges
            bf = bf[:E]
            es = es[:E]
            ed = ed[:E]
        atom_feats[b, :V] = af
        bond_feats[b, :E] = bf
        e_src[b, :E] = es
        e_dst[b, :E] = ed

    return (
        torch.from_numpy(atom_feats),
        torch.from_numpy(bond_feats),
        torch.from_numpy(e_src),
        torch.from_numpy(e_dst),
    )


def batch_iter(
    graphs: List[Optional[Tuple]],
    reg_targets: np.ndarray,
    cls_targets: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    max_atoms: int,
    max_edges: int,
    atom_dim: int,
    bond_dim: int,
    shuffle: bool,
    rng: np.random.Generator,
):
    if shuffle:
        order = indices.copy()
        rng.shuffle(order)
    else:
        order = indices
    for start in range(0, order.shape[0], batch_size):
        batch_idx = order[start : start + batch_size]
        batch_graphs = [graphs[int(i)] for i in batch_idx]
        batched = collate_graphs(
            batch_graphs,
            max_atoms,
            max_edges,
            atom_dim,
            bond_dim,
        )
        if batched is None:
            continue
        atom_feats, bond_feats, e_src, e_dst = batched
        reg_t = torch.from_numpy(reg_targets[batch_idx])
        cls_t = torch.from_numpy(cls_targets[batch_idx])
        yield batch_idx, atom_feats, bond_feats, e_src, e_dst, reg_t, cls_t


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
@dataclass
class RegScaler:
    """Per-target mean/std scaler (z-score)."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, reg: np.ndarray) -> "RegScaler":
        mean = reg.mean(axis=0)
        std = reg.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, reg: np.ndarray) -> np.ndarray:
        return (reg - self.mean) / self.std

    def inverse_transform(self, reg: np.ndarray) -> np.ndarray:
        return reg * self.std + self.mean


# ---------------------------------------------------------------------------
# Train + eval
# ---------------------------------------------------------------------------
def _train_one_epoch(
    model: TmqmPretrainModel,
    optimizer: torch.optim.Optimizer,
    reg_criterion: nn.Module,
    cls_criterion: nn.Module,
    graphs: List,
    reg_targets: np.ndarray,
    cls_targets: np.ndarray,
    train_idx: np.ndarray,
    batch_size: int,
    max_atoms: int,
    max_edges: int,
    device: torch.device,
    rng: np.random.Generator,
    reg_weight: float = 1.0,
    cls_weight: float = 1.0,
) -> Dict[str, float]:
    model.train()
    total = 0.0
    total_reg = 0.0
    total_cls = 0.0
    n = 0
    n_batches = 0
    for _, atom_f, bond_f, e_src, e_dst, reg_t, cls_t in batch_iter(
        graphs,
        reg_targets,
        cls_targets,
        train_idx,
        batch_size,
        max_atoms,
        max_edges,
        ATOM_FEATURE_DIM,
        BOND_FEATURE_DIM,
        shuffle=True,
        rng=rng,
    ):
        atom_f = atom_f.to(device)
        bond_f = bond_f.to(device)
        e_src = e_src.to(device)
        e_dst = e_dst.to(device)
        reg_t = reg_t.to(device)
        cls_t = cls_t.to(device)
        reg_pred, cls_logits = model(atom_f, bond_f, e_src, e_dst)
        loss_reg = reg_criterion(reg_pred, reg_t)
        loss_cls = cls_criterion(cls_logits, cls_t)
        loss = reg_weight * loss_reg + cls_weight * loss_cls
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total += float(loss.item())
        total_reg += float(loss_reg.item())
        total_cls += float(loss_cls.item())
        n += int(reg_t.size(0))
        n_batches += 1
    return {
        "loss": total / max(n_batches, 1),
        "loss_reg": total_reg / max(n_batches, 1),
        "loss_cls": total_cls / max(n_batches, 1),
        "n_samples": n,
        "n_batches": n_batches,
    }


@torch.no_grad()
def _evaluate(
    model: TmqmPretrainModel,
    graphs: List,
    reg_targets: np.ndarray,
    cls_targets: np.ndarray,
    idx: np.ndarray,
    batch_size: int,
    max_atoms: int,
    max_edges: int,
    device: torch.device,
    scaler: RegScaler,
) -> Dict[str, float]:
    model.eval()
    reg_pred_all: List[np.ndarray] = []
    cls_pred_all: List[np.ndarray] = []
    reg_true_all: List[np.ndarray] = []
    cls_true_all: List[np.ndarray] = []
    for _, atom_f, bond_f, e_src, e_dst, reg_t, cls_t in batch_iter(
        graphs,
        reg_targets,
        cls_targets,
        idx,
        batch_size,
        max_atoms,
        max_edges,
        ATOM_FEATURE_DIM,
        BOND_FEATURE_DIM,
        shuffle=False,
        rng=np.random.default_rng(0),
    ):
        atom_f = atom_f.to(device)
        bond_f = bond_f.to(device)
        e_src = e_src.to(device)
        e_dst = e_dst.to(device)
        reg_pred, cls_logits = model(atom_f, bond_f, e_src, e_dst)
        reg_pred_all.append(reg_pred.cpu().numpy())
        cls_pred_all.append(cls_logits.argmax(dim=-1).cpu().numpy())
        reg_true_all.append(reg_t.numpy())
        cls_true_all.append(cls_t.numpy())

    if not reg_pred_all:
        return {"rmse": float("nan"), "mae": float("nan"), "cls_acc": float("nan")}

    reg_pred_np = np.concatenate(reg_pred_all, axis=0)
    cls_pred_np = np.concatenate(cls_pred_all, axis=0)
    reg_true_np = np.concatenate(reg_true_all, axis=0)
    cls_true_np = np.concatenate(cls_true_all, axis=0)

    # RMSE / MAE in original units
    reg_pred_orig = scaler.inverse_transform(reg_pred_np)
    err = reg_pred_orig - reg_true_np
    rmse = float(np.sqrt((err**2).mean()))
    mae = float(np.abs(err).mean())
    cls_acc = float((cls_pred_np == cls_true_np).mean())

    per_target = {}
    for j, name in enumerate(TMQM_REGRESSION_TARGETS):
        per_target[f"rmse_{name}"] = float(
            np.sqrt(((reg_pred_orig[:, j] - reg_true_np[:, j]) ** 2).mean())
        )

    return {
        "rmse": rmse,
        "mae": mae,
        "cls_acc": cls_acc,
        "per_target": per_target,
    }


def save_encoder(
    model: TmqmPretrainModel,
    out_dir: Path,
    config: Dict,
    train_meta: Dict,
    scaler: RegScaler,
) -> Path:
    """Save only backbone weights (encoder) + metadata to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = out_dir / "tmqm_pretrained_encoder.pt"
    backbone = {
        "atom_emb.weight": model.atom_emb.weight.detach().cpu(),
        "atom_emb.bias": model.atom_emb.bias.detach().cpu(),
        "bond_emb.weight": model.bond_emb.weight.detach().cpu(),
        "bond_emb.bias": model.bond_emb.bias.detach().cpu(),
        "edge_init.0.weight": model.edge_init[0].weight.detach().cpu(),
        "edge_init.0.bias": model.edge_init[0].bias.detach().cpu(),
        "gru_update.W_z.weight": model.gru_update.W_z.weight.detach().cpu(),
        "gru_update.W_r.weight": model.gru_update.W_r.weight.detach().cpu(),
        "gru_update.W_h.weight": model.gru_update.W_h.weight.detach().cpu(),
        "gru_update.b_z": model.gru_update.b_z.detach().cpu(),
        "gru_update.b_r": model.gru_update.b_r.detach().cpu(),
        "gru_update.b_h": model.gru_update.b_h.detach().cpu(),
    }
    payload = {
        "state_dict": backbone,
        "config": config,
        "train_meta": train_meta,
        "reg_scaler": {
            "mean": scaler.mean.tolist(),
            "std": scaler.std.tolist(),
        },
    }
    torch.save(payload, encoder_path)
    return encoder_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description="Pretrain D-MPNN encoder on tmQM 108k transition metal complexes"
    )
    p.add_argument(
        "--data",
        type=Path,
        default=Path(".data/molmetal/tmQM/tmqm_parsed.csv"),
        help="Path to tmqm_parsed.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("models/tmqm_pretrained"),
        help="Where to write the encoder checkpoint + metrics",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden", type=int, default=DMPNN_HIDDEN)
    p.add_argument("--depth", type=int, default=DMPNN_DEPTH)
    p.add_argument("--dropout", type=float, default=DMPNN_DROPOUT)
    p.add_argument("--max-atoms", type=int, default=120)
    p.add_argument("--max-edges", type=int, default=240)
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu / cuda (default cpu; tmQM is CPU-friendly)",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional row cap for fast smoke (0 = use all rows)",
    )
    args = p.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print(f"[pretrain_tmqm] loading {args.data}")
    ds = load_tmqm(args.data, require_smiles=True)
    if args.max_rows and args.max_rows < ds.n:
        # Deterministic subsample for smoke runs
        idx = np.arange(ds.n)
        rng.shuffle(idx)
        keep = sorted(idx[: args.max_rows].tolist())
        ds = TmQMDataset(
            smiles=[ds.smiles[i] for i in keep],
            reg_targets=ds.reg_targets[keep],
            cls_targets=ds.cls_targets[keep],
        )
    print(
        f"[pretrain_tmqm] rows={ds.n}  reg_targets={len(TMQM_REGRESSION_TARGETS)} "
        f"cls_targets={len(TMQM_CLASSIFICATION_TARGETS)} (N_OX_BINS={N_OX_BINS})"
    )

    # Standardise regression targets on training set only
    train_idx, val_idx, test_idx = stratified_split(
        ds.cls_targets,
        val_frac=0.05,
        test_frac=0.05,
        seed=args.seed,
    )
    print(
        f"[pretrain_tmqm] split: train={len(train_idx)} "
        f"val={len(val_idx)} test={len(test_idx)}"
    )
    scaler = RegScaler.fit(ds.reg_targets[train_idx])

    # Featurise SMILES (CPU side)
    print(f"[pretrain_tmqm] featurising {ds.n} SMILES ...")
    t0 = time.time()
    graphs = featurize_smiles_list(ds.smiles)
    t_featurise = time.time() - t0
    n_valid = sum(g is not None for g in graphs)
    print(
        f"[pretrain_tmqm] featurised in {t_featurise:.1f}s "
        f"(valid {n_valid}/{ds.n} = {n_valid / max(ds.n, 1):.3f})"
    )
    # Save a few smoke stats for downstream consumers
    (out_dir / "featurise_stats.json").write_text(
        json.dumps(
            {
                "n_total": ds.n,
                "n_valid_graphs": n_valid,
                "featurise_seconds": t_featurise,
                "split": {
                    "train": int(len(train_idx)),
                    "val": int(len(val_idx)),
                    "test": int(len(test_idx)),
                },
            },
            indent=2,
        )
    )

    device = torch.device(args.device)

    model = TmqmPretrainModel(
        hidden=args.hidden,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    reg_criterion = nn.MSELoss(reduction="mean")
    cls_criterion = nn.CrossEntropyLoss(reduction="mean")

    # Standardised regression targets for loss (training stability)
    reg_targets_scaled = scaler.transform(ds.reg_targets).astype(np.float32)

    # Encoder config — defined up-front so per-epoch best checkpoints
    # can be written without an UnboundLocalError.
    config = {
        "hidden": args.hidden,
        "depth": args.depth,
        "dropout": args.dropout,
        "atom_dim": ATOM_FEATURE_DIM,
        "bond_dim": BOND_FEATURE_DIM,
        "n_reg_targets": len(TMQM_REGRESSION_TARGETS),
        "n_cls_targets": N_OX_BINS,
        "reg_target_names": list(TMQM_REGRESSION_TARGETS),
        "cls_target_names": list(TMQM_CLASSIFICATION_TARGETS),
        "ox_states": list(OX_STATES),
    }

    # ----------------- training loop -----------------
    curves: List[Dict[str, float]] = []
    best_val_rmse = math.inf
    best_epoch = -1
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        train_metrics = _train_one_epoch(
            model,
            optimizer,
            reg_criterion,
            cls_criterion,
            graphs,
            reg_targets_scaled,
            ds.cls_targets,
            train_idx,
            batch_size=args.batch,
            max_atoms=args.max_atoms,
            max_edges=args.max_edges,
            device=device,
            rng=rng,
        )
        val_metrics = _evaluate(
            model,
            graphs,
            ds.reg_targets,
            ds.cls_targets,
            val_idx,
            batch_size=args.batch,
            max_atoms=args.max_atoms,
            max_edges=args.max_edges,
            device=device,
            scaler=scaler,
        )
        test_metrics = _evaluate(
            model,
            graphs,
            ds.reg_targets,
            ds.cls_targets,
            test_idx,
            batch_size=args.batch,
            max_atoms=args.max_atoms,
            max_edges=args.max_edges,
            device=device,
            scaler=scaler,
        )
        cur = {
            "epoch": epoch,
            "wall_s": round(time.time() - t_epoch, 1),
            **train_metrics,
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_cls_acc": val_metrics["cls_acc"],
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "test_cls_acc": test_metrics["cls_acc"],
        }
        curves.append(cur)
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            # Save best-epoch encoder checkpoint so partial progress
            # is recoverable even if the run is killed mid-training.
            interim_meta = {
                "best_epoch": best_epoch,
                "best_val_rmse": best_val_rmse,
                "last_epoch": epoch,
                "last_test": test_metrics,
                "curves": curves,
                "n_rows": ds.n,
                "n_valid_graphs": n_valid,
                "epochs": args.epochs,
                "batch_size": args.batch,
                "lr": args.lr,
                "seed": args.seed,
            }
            save_encoder(model, out_dir, config, interim_meta, scaler)
            interim_metrics = out_dir / "pretrain_metrics.json"
            interim_metrics.write_text(json.dumps(interim_meta, indent=2))

        print(
            f"[pretrain_tmqm] epoch {epoch:3d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"reg={train_metrics['loss_reg']:.4f} "
            f"cls={train_metrics['loss_cls']:.4f} | "
            f"val RMSE={val_metrics['rmse']:.4f} MAE={val_metrics['mae']:.4f} "
            f"cls_acc={val_metrics['cls_acc']:.3f} | "
            f"test RMSE={test_metrics['rmse']:.4f} MAE={test_metrics['mae']:.4f} "
            f"cls_acc={test_metrics['cls_acc']:.3f} | "
            f"wall {cur['wall_s']}s"
        )

    total_wall = time.time() - t_start

    # ----------------- final test eval + encoder save -----------------
    final_test = _evaluate(
        model,
        graphs,
        ds.reg_targets,
        ds.cls_targets,
        test_idx,
        batch_size=args.batch,
        max_atoms=args.max_atoms,
        max_edges=args.max_edges,
        device=device,
        scaler=scaler,
    )
    train_meta = {
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "final_test": final_test,
        "curves": curves,
        "n_rows": ds.n,
        "n_valid_graphs": n_valid,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "lr": args.lr,
        "seed": args.seed,
        "total_wall_s": round(total_wall, 1),
        "featurise_seconds": round(t_featurise, 1),
    }

    encoder_path = save_encoder(model, out_dir, config, train_meta, scaler)
    metrics_path = out_dir / "pretrain_metrics.json"
    metrics_path.write_text(json.dumps(train_meta, indent=2))
    summary_path = out_dir / "pretrain_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "encoder_path": str(encoder_path),
                "best_epoch": best_epoch,
                "best_val_rmse": best_val_rmse,
                "final_test_rmse": final_test["rmse"],
                "final_test_cls_acc": final_test["cls_acc"],
                "total_wall_s": round(total_wall, 1),
                "n_rows": ds.n,
                "n_valid_graphs": n_valid,
            },
            indent=2,
        )
    )
    print(f"[pretrain_tmqm] wrote encoder to {encoder_path}")
    print(f"[pretrain_tmqm] wrote metrics to {metrics_path}")
    print(
        f"[pretrain_tmqm] final test RMSE={final_test['rmse']:.4f} "
        f"cls_acc={final_test['cls_acc']:.3f} "
        f"wall={total_wall:.1f}s"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
