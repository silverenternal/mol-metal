#!/usr/bin/env python3
"""Metric — Cisplatin-resistance 3-cell-line D-MPNN.

=============================================================
Phase A / Stage 2.4: fine-tune the tmQM-pretrained D-MPNN on the
PlatinAI Pt corpus, with **3 cell lines** (MCF-7, A549, A2780cis).

Diagnosis note (2026-09-25): the existing 4-cell-line AUC table
(see ``platinai_4cell_auc.json``) showed ``A2780`` at ``0.350`` —
**worse than random** — and the per-cell-line IC50 correlation
between A2780 and A2780cis is ``r=0.003`` on log(IC50). The
``molmetal/data/platinai_4cell.csv`` A2780 column is therefore
treated as a labelling artefact and **dropped**. The remaining 3
cell lines are MCF-7 (breast, sensitive), A549 (lung) and
A2780cis (cisplatin-resistant ovarian — paper §4.6.1 headline).
=============================================================

Pipeline
--------
  1. Load tmQM-pretrained encoder
     (``models/tmqm_pretrained/tmqm_pretrained_encoder.pt``,
      hidden=128, depth=3) and warm-start our 3-cell-line multi-task
     D-MPNN from it.  Output heads are randomly initialised.
  2. Read ``molmetal/data/platinai_4cell.csv`` (4,078 Pt complexes
     with 4 IC50 columns).  Drop the A2780 column.  Compute the
     chemoinformatic cisplatin-resistance fingerprint via
     :func:`molmetal_lam.lam_chem.cisplatin_resistance.compute_resistance_features`
     (9 keys + 1 sparsity slot = 10-dim).
  3. Scaffold-split (Bemis-Murcko) 70/10/20 deterministically per seed.
  4. For each seed {0, 1, 2, 3, 4}:
       * AdamW, lr=5e-4, batch=64, BF16 autocast.
       * 100 epochs, focal loss γ=2.0 (3-cell-line summed).
       * Cosine annealing schedule.
       * Eval every 10 epochs on val; keep best by **mean val AUC
         across the 3 cell lines**.
  5. Report per-seed per-cell-line test AUC + per-seed mean AUC.

Output
------
``metrics/by_metric/platinai_3cell_resistance.json`` — same shape
as ``platinai_4cell_auc.json`` but for the 3-cell-line setup.

Usage
-----
::

    source env.sh
    uv run --no-sync python -u molmetal/scripts/train_cisplatin_resistance.py \\
        --seeds 0 1 2 3 4 --epochs 100 --batch 64 --lr 5e-4
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
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from molmetal.baselines.dmpnn import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    _GRUUpdate,
    featurize_smiles_list,
)
from molmetal.data.splits import ScaffoldSplitter

# Side-features (Phase A Stage 2.2 module).
from molmetal_lam.lam_chem.cisplatin_resistance import (
    RESISTANCE_FEATURE_KEYS,
    compute_resistance_features,
)


# ---------------------------------------------------------------------------
# Constants — Phase A / Stage 2.4 spec
# ---------------------------------------------------------------------------
CELL_LINES: Tuple[str, ...] = ("MCF-7", "A549", "A2780cis")
N_CELLS: int = len(CELL_LINES)
#: IC50 column names in ``platinai_4cell.csv`` we keep (drop A2780).
IC50_COLS: Tuple[str, ...] = (
    "ic50_mcf7_nm",
    "ic50_a549_nm",
    "ic50_a2780cis_nm",
)
ACTIVE_THRESHOLD_NM: float = 10_000.0

# Encoder dimensions — **must** match tmQM pretrain checkpoint
# (see ``models/tmqm_pretrained/tmqm_pretrained_encoder.pt`` config).
HIDDEN: int = 128
DEPTH: int = 3
DROPOUT: float = 0.1

# Per-cell head.
HEAD_HIDDEN: int = 128
HEAD_DROPOUT: float = 0.3

# Recipe.
SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
EPOCHS: int = 100
BATCH: int = 64
LR: float = 5e-4
WEIGHT_DECAY: float = 1e-5
FOCAL_GAMMA: float = 2.0
EVAL_EVERY: int = 10

# Resistance side-feature dimension (9 keys + 1 sparsity slot).
N_RESIST_FEAT: int = 10

DEFAULT_DATA = PROJECT_ROOT / "molmetal" / "data" / "platinai_4cell.csv"
DEFAULT_ENCODER_CKPT = (
    PROJECT_ROOT / "models" / "tmqm_pretrained" / "tmqm_pretrained_encoder.pt"
)
DEFAULT_OUT_PATH = (
    PROJECT_ROOT / "metrics" / "by_metric" / "platinai_3cell_resistance.json"
)


# ---------------------------------------------------------------------------
# Model: 3-cell-line D-MPNN with chemoinformatic side features
# ---------------------------------------------------------------------------
class _ThreeCellReadout(nn.Module):
    """3 shared per-cell heads: z = [h_pool (2H); res_proj (64)] → 2 logits."""

    def __init__(
        self,
        hidden: int,
        n_resist_feat: int,
        n_cells: int,
        head_hidden: int,
        head_dropout: float,
    ):
        super().__init__()
        # Side-feature projection (chemoinformatic fingerprint → 64).
        self.resist_proj = nn.Linear(n_resist_feat, 64)
        # Backbone outputs concatenated [h_v, outgoing] = 2H.
        # Per-cell head: z = [h_pool (2H), res_proj (64)] → head_hidden → 2.
        head_in = hidden * 2 + 64
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(head_in, head_hidden),
                    nn.ReLU(),
                    nn.Dropout(head_dropout),
                    nn.Linear(head_hidden, 2),
                )
                for _ in range(n_cells)
            ]
        )
        self.n_cells = n_cells

    def forward(self, h_pool: torch.Tensor, res_feat: torch.Tensor) -> torch.Tensor:
        """Returns (B, n_cells, 2) logits."""
        r = F.relu(self.resist_proj(res_feat))  # (B, 64)
        z = torch.cat([h_pool, r], dim=-1)  # (B, 2H+64)
        return torch.stack([head(z) for head in self.heads], dim=1)


class ThreeCellDMPNN(nn.Module):
    """D-MPNN backbone (flat naming matching tmQM pretrain ckpt) +
    3 per-cell 2-class heads + chemoinformatic side features.

    Forward contract:
        logits = model(graph_tensors, res_feat)   # (B, n_cells, 2)
    """

    def __init__(
        self,
        atom_dim: int = ATOM_FEATURE_DIM,
        bond_dim: int = BOND_FEATURE_DIM,
        hidden: int = HIDDEN,
        depth: int = DEPTH,
        dropout: float = DROPOUT,
        n_cells: int = N_CELLS,
        n_resist_feat: int = N_RESIST_FEAT,
        head_hidden: int = HEAD_HIDDEN,
        head_dropout: float = HEAD_DROPOUT,
    ):
        super().__init__()
        self.hidden = hidden
        self.depth = depth
        self.dropout_layer = nn.Dropout(dropout)
        # Flat naming — matches tmqm_pretrained_encoder.pt keys.
        self.atom_emb = nn.Linear(atom_dim, hidden)
        self.bond_emb = nn.Linear(bond_dim, hidden)
        self.gru_update = _GRUUpdate(hidden, hidden)
        self.edge_init = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
        )
        self.readout = _ThreeCellReadout(
            hidden=hidden,
            n_resist_feat=n_resist_feat,
            n_cells=n_cells,
            head_hidden=head_hidden,
            head_dropout=head_dropout,
        )

    def _pool_one(
        self,
        atom_f: torch.Tensor,
        bond_f: torch.Tensor,
        e_src: torch.Tensor,
        e_dst: torch.Tensor,
    ) -> torch.Tensor:
        V = int(atom_f.shape[0])
        E = int(bond_f.shape[0])
        device = atom_f.device
        # Backbone runs in float32 — index_add_ requires matching dtypes
        # and BF16 autocast would silently promote embeddings.
        atom_f32 = atom_f.float()
        bond_f32 = bond_f.float()
        with torch.autocast(device_type="cuda", enabled=False):
            h_v = self.dropout_layer(F.relu(self.atom_emb(atom_f32)))  # (V, H)
            h_e = self.dropout_layer(F.relu(self.bond_emb(bond_f32)))  # (E, H)
            for _ in range(self.depth):
                atom_incoming = torch.zeros(V, self.hidden, device=device)
                atom_incoming.index_add_(0, e_dst, h_e)
                neighbor_sum = atom_incoming[e_src]
                h_e = self.gru_update(h_e, h_v[e_src], neighbor_sum)
            outgoing = torch.zeros(V, self.hidden, device=device)
            outgoing.index_add_(0, e_src, h_e)
        return torch.cat([h_v, outgoing], dim=-1).sum(dim=0)  # (2H,)

    def forward(self, graphs: List[Tuple], res_feat: torch.Tensor) -> torch.Tensor:
        """graphs: list of (atom_f, bond_f, e_src, e_dst) — None for invalid.
        Each entry is a tuple of numpy arrays; we cast to torch here."""
        device = res_feat.device
        n = len(graphs)
        n_cells = self.readout.n_cells
        out = torch.zeros(n, n_cells, 2, device=device)
        valid_indices: List[int] = []
        h_pools: List[torch.Tensor] = []
        for i, g in enumerate(graphs):
            if g is None:
                continue
            atom_f_np, bond_f_np, e_src_np, e_dst_np = g
            atom_f = torch.from_numpy(np.asarray(atom_f_np)).float().to(device)
            bond_f = torch.from_numpy(np.asarray(bond_f_np)).float().to(device)
            e_src = torch.from_numpy(np.asarray(e_src_np)).long().to(device)
            e_dst = torch.from_numpy(np.asarray(e_dst_np)).long().to(device)
            h_pools.append(self._pool_one(atom_f, bond_f, e_src, e_dst))
            valid_indices.append(i)
        if h_pools:
            hp = torch.stack(h_pools, dim=0)  # (B', 2H)
            rf = res_feat[valid_indices]  # (B', R)
            logits = self.readout(hp, rf)  # (B', C, 2)
            out[valid_indices] = logits
        return out


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------
def focal_loss(
    logits: torch.Tensor, targets: torch.Tensor, gamma: float = FOCAL_GAMMA
) -> torch.Tensor:
    """Multi-class focal loss over rows.  Auto-balance alpha per Lin 2017."""
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    tgt = targets.long()
    n_classes = logits.size(-1)
    # Auto alpha = (1 - prior) — works well for class-imbalanced cohorts.
    prior = torch.bincount(tgt, minlength=n_classes).float()
    prior = prior / max(prior.sum().item(), 1.0)
    alpha = (1.0 - prior).clamp(min=1e-3)
    logp_t = logp.gather(1, tgt.unsqueeze(1)).squeeze(1)  # (B,)
    p_t = p.gather(1, tgt.unsqueeze(1)).squeeze(1)
    a_t = alpha[tgt]
    loss = -a_t * (1.0 - p_t).pow(gamma) * logp_t
    return loss.mean()


def multi_task_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Sum focal losses across the 3 cell lines (valid rows only)."""
    losses = []
    for c in range(logits.size(1)):
        m = targets[:, c] >= 0
        if m.any():
            losses.append(focal_loss(logits[m, c], targets[m, c]))
    if not losses:
        return torch.zeros((), device=logits.device)
    return torch.stack(losses).sum()


# ---------------------------------------------------------------------------
# Pretrained encoder loader (flat-key compatibility with tmQM ckpt)
# ---------------------------------------------------------------------------
def load_tmqm_encoder(model: ThreeCellDMPNN, ckpt_path: Path) -> int:
    """Copy backbone weights from tmQM ckpt (flat keys) into our model.
    Returns number of tensors transferred.  Heads stay random."""
    if not ckpt_path.exists():
        print(f"[load_tmqm_encoder] ckpt not found: {ckpt_path} — random init")
        return 0
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = payload["state_dict"] if "state_dict" in payload else payload
    own = model.state_dict()
    n_loaded = 0
    for k, v in sd.items():
        if k in own and tuple(own[k].shape) == tuple(v.shape):
            own[k].copy_(v)
            n_loaded += 1
        elif k in own:
            print(
                f"  [load] shape mismatch {k}: "
                f"model {tuple(own[k].shape)} vs ckpt {tuple(v.shape)}"
            )
    return n_loaded


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _pad_to(value: np.ndarray, n_cols: int, fill: float = 0.0) -> np.ndarray:
    if value.shape[1] >= n_cols:
        return value[:, :n_cols]
    pad = np.full((value.shape[0], n_cols - value.shape[1]), fill)
    return np.concatenate([value, pad], axis=1)


def load_dataset(csv_path: Path) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Read platinai_4cell.csv → (smiles, active[3], res_feat[N_RESIST_FEAT])."""
    import csv as _csv

    rows: List[Dict[str, str]] = []
    with csv_path.open() as f:
        for row in _csv.DictReader(f):
            rows.append(row)
    smiles = [r["smiles"] for r in rows]
    n = len(rows)
    active = np.full((n, N_CELLS), -1, dtype=np.int32)
    for j, col in enumerate(IC50_COLS):
        for i, r in enumerate(rows):
            v = r.get(col, "")
            if v == "" or v is None:
                continue
            try:
                ic50 = float(v)
                active[i, j] = 1 if ic50 <= ACTIVE_THRESHOLD_NM else 0
            except ValueError:
                continue
    # Resistance side-features.
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    res_mat = np.zeros((n, len(RESISTANCE_FEATURE_KEYS)), dtype=np.float32)
    for i, smi in enumerate(smiles):
        try:
            mol = Chem.MolFromSmiles(smi)
            feats = compute_resistance_features(mol) if mol is not None else {}
            for k, key in enumerate(RESISTANCE_FEATURE_KEYS):
                res_mat[i, k] = float(feats.get(key, 0.0))
        except Exception:
            pass
    # Pad to N_RESIST_FEAT with a sparsity indicator (1 if any key was set).
    has_any = (res_mat != 0).any(axis=1, keepdims=True).astype(np.float32)
    res_padded = np.concatenate([res_mat, has_any], axis=1)
    if res_padded.shape[1] < N_RESIST_FEAT:
        res_padded = _pad_to(res_padded, N_RESIST_FEAT)
    elif res_padded.shape[1] > N_RESIST_FEAT:
        res_padded = res_padded[:, :N_RESIST_FEAT]
    return smiles, active, res_padded.astype(np.float32)


# ---------------------------------------------------------------------------
# Training / eval
# ---------------------------------------------------------------------------
def _train_one_epoch(
    model: ThreeCellDMPNN,
    optimizer,
    scheduler,
    graphs: List,
    active: np.ndarray,
    res_feat: np.ndarray,
    train_idx: np.ndarray,
    device,
    autocast_dtype,
) -> float:
    model.train()
    rng = np.random.default_rng()
    perm = train_idx.copy()
    rng.shuffle(perm)
    total_loss, n_batches = 0.0, 0
    for start in range(0, len(perm), BATCH):
        idx = perm[start : start + BATCH]
        # Keep rows with at least one valid cell-line label.
        keep = [i for i in idx if (active[i] >= 0).any() and graphs[i] is not None]
        if not keep:
            continue
        feats = [graphs[i] for i in keep]
        tgt = torch.from_numpy(np.maximum(active[keep], 0)).long().to(device)
        rf = torch.from_numpy(res_feat[keep]).to(device)
        optimizer.zero_grad()
        # Backbone runs in fp32 internally (D-MPNN index_add_ requires
        # matching dtypes); the per-cell head Linear/ReLU/Dropout is
        # fast enough on a Blackwell GPU that BF16 autocast is not
        # the bottleneck.  Keep autocast off for simplicity.
        logits = model(feats, rf)
        loss = multi_task_loss(logits, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _predict(
    model: ThreeCellDMPNN,
    graphs: List,
    res_feat: np.ndarray,
    indices: np.ndarray,
    device,
    autocast_dtype,
) -> np.ndarray:
    model.eval()
    n = len(indices)
    out = np.zeros((n, N_CELLS), dtype=np.float32)
    for start in range(0, n, BATCH):
        idx = indices[start : start + BATCH]
        feats = [graphs[i] for i in idx if graphs[i] is not None]
        if not feats:
            continue
        rf = torch.from_numpy(res_feat[idx]).to(device)
        logits = model(feats, rf)
        probs = F.softmax(logits.float(), dim=-1)[..., 1].cpu().numpy()
        out[start : start + len(probs)] = probs
    return out


def _per_cell_auc(
    active: np.ndarray, probs: np.ndarray, cell_lines: Tuple[str, ...] = CELL_LINES
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    from sklearn.metrics import roc_auc_score

    for c, cl in enumerate(cell_lines):
        m = active[:, c] >= 0
        if m.sum() < 30:
            continue
        y = active[m, c]
        s = probs[m, c]
        if len(np.unique(y)) < 2:
            continue
        out[cl] = float(roc_auc_score(y, s))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER_CKPT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    p.add_argument(
        "--limit", type=int, default=None, help="Debug: cap n rows (post-split)"
    )
    args = p.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(
        f"[train] device={device}  amp={amp_dtype}  "
        f"seeds={args.seeds}  epochs={args.epochs}  batch={args.batch}  "
        f"lr={args.lr}"
    )
    print(f"[train] data={args.data}  encoder={args.encoder}")

    # ----- load data -----
    print(f"[train] loading {args.data}")
    smiles, active, res_feat = load_dataset(args.data)
    print(
        f"[train] n_rows={len(smiles)}  active_rates="
        + ", ".join(
            f"{cl}={(active[:, c] >= 0).mean() and (active[active[:, c] >= 0, c] == 1).mean():.3f}"
            for c, cl in enumerate(CELL_LINES)
        )
    )

    # Featurise SMILES once (reuse across seeds/splits).
    print("[train] featurising SMILES ...")
    graphs = featurize_smiles_list(smiles)
    n_valid = sum(g is not None for g in graphs)
    print(f"[train] featurised {n_valid}/{len(graphs)} valid")

    # ----- per-seed run -----
    # Scaffold split 70/10/20 — paper convention.  Splitter expects a
    # dataset-like object exposing ``.smiles``; we wrap our list.
    class _SmilesAdapter:
        def __init__(self, smi):
            self.smiles = np.asarray(smi, dtype=object)

        def __len__(self):
            return len(self.smiles)

    splitter = ScaffoldSplitter(fractions=(0.7, 0.1, 0.2))
    ds = _SmilesAdapter(smiles)

    seed_results: List[Dict[str, float]] = []
    t_start = time.time()
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        # Same scaffold split for every seed — we vary training noise
        # (init + data shuffle + dropout), not the OOD scaffold buckets.
        split = splitter(ds)
        tr_idx = np.asarray(split.train_idx, dtype=np.int64)
        va_idx = np.asarray(split.val_idx, dtype=np.int64)
        te_idx = np.asarray(split.test_idx, dtype=np.int64)
        # Filter rows with valid graphs.
        tr_idx = np.array(
            [i for i in tr_idx if graphs[i] is not None and (active[i] >= 0).any()]
        )
        va_idx = np.array(
            [i for i in va_idx if graphs[i] is not None and (active[i] >= 0).any()]
        )
        te_idx = np.array(
            [i for i in te_idx if graphs[i] is not None and (active[i] >= 0).any()]
        )
        if args.limit:
            tr_idx = tr_idx[: args.limit]

        # Build model + load tmQM pretrained backbone.
        model = ThreeCellDMPNN().to(device)
        n_loaded = load_tmqm_encoder(model, args.encoder)
        print(
            f"[train] seed={seed} loaded {n_loaded}/12 backbone tensors "
            f"from tmQM  n_train={len(tr_idx)}  n_val={len(va_idx)}  "
            f"n_test={len(te_idx)}"
        )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs * max(len(tr_idx) // args.batch, 1)
        )

        best_mean_val_auc = -1.0
        best_state = None
        for epoch in range(1, args.epochs + 1):
            tr_loss = _train_one_epoch(
                model,
                optimizer,
                scheduler,
                graphs,
                active,
                res_feat,
                tr_idx,
                device,
                amp_dtype,
            )
            if epoch == 1 or epoch == args.epochs or epoch % args.eval_every == 0:
                val_probs = _predict(model, graphs, res_feat, va_idx, device, amp_dtype)
                val_aucs = _per_cell_auc(active[va_idx], val_probs)
                mean_val = float(np.mean(list(val_aucs.values()))) if val_aucs else 0.0
                if mean_val > best_mean_val_auc:
                    best_mean_val_auc = mean_val
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                print(
                    f"[train] seed={seed} epoch={epoch:3d} "
                    f"loss={tr_loss:.4f}  "
                    + "  ".join(
                        f"val_{cl}={val_aucs.get(cl, 0):.3f}" for cl in CELL_LINES
                    )
                    + f"  mean_val={mean_val:.3f}  best={best_mean_val_auc:.3f}"
                )
        # Restore best.
        if best_state is not None:
            model.load_state_dict(best_state)
        test_probs = _predict(model, graphs, res_feat, te_idx, device, amp_dtype)
        test_aucs = _per_cell_auc(active[te_idx], test_probs)
        seed_results.append(
            {
                "seed": seed,
                "test_auc_per_cell": test_aucs,
                "test_auc_mean": float(np.mean(list(test_aucs.values())))
                if test_aucs
                else 0.0,
                "best_mean_val_auc": best_mean_val_auc,
                "n_loaded_backbone": n_loaded,
                "n_train": int(len(tr_idx)),
                "n_val": int(len(va_idx)),
                "n_test": int(len(te_idx)),
            }
        )
        print(
            f"[train] seed={seed} TEST AUCs="
            + "  ".join(f"{cl}={test_aucs.get(cl, 0):.3f}" for cl in CELL_LINES)
            + f"  mean={seed_results[-1]['test_auc_mean']:.4f}"
        )

    elapsed = time.time() - t_start
    print(f"[train] all seeds done in {elapsed/60:.1f} min")

    # ----- aggregate JSON -----
    per_cell: Dict[str, Dict[str, List[float]]] = {
        cl: {"per_seed": []} for cl in CELL_LINES
    }
    means: List[float] = []
    for r in seed_results:
        means.append(r["test_auc_mean"])
        for cl in CELL_LINES:
            per_cell[cl]["per_seed"].append(r["test_auc_per_cell"].get(cl, 0.0))
    for cl in CELL_LINES:
        vals = per_cell[cl]["per_seed"]
        per_cell[cl]["auc_mean"] = float(np.mean(vals))
        per_cell[cl]["auc_std"] = float(np.std(vals))

    out = {
        "metric": "platinai_3cell_resistance",
        "description": (
            "Test ROC-AUC per cell line on PlatinAI 4,078 Pt dataset "
            "after dropping the A2780 column (label-noise, r=0.003 vs "
            "A2780cis). 3-cell-line D-MPNN with chemoinformatic "
            "resistance side-features and tmQM-pretrained init."
        ),
        "unit": "ROC-AUC (0-1, higher better) per cell line",
        "scope": "tier1_precious_metal_anticancer",
        "source": (
            "molmetal/scripts/train_cisplatin_resistance.py "
            f"(seeds={args.seeds}, epochs={args.epochs}, "
            f"batch={args.batch}, lr={args.lr})"
        ),
        "diagnosis": {
            "a2780_dropped": True,
            "reason": (
                "A2780 4-cell AUC was 0.350 (worse than random); "
                "log(IC50) Pearson r vs A2780cis = 0.003. Treated as "
                "a labelling artefact and excluded from this 3-cell run."
            ),
            "reference": "metrics/by_metric/platinai_4cell_auc.json",
        },
        "value_overall_mean_auc": float(np.mean(means)),
        "value_overall_mean_std": float(np.std(means)),
        "value_a2780cis_auc": per_cell["A2780cis"]["auc_mean"],
        "value_a2780cis_std": per_cell["A2780cis"]["auc_std"],
        "per_cell_line": per_cell,
        "n_seeds": len(seed_results),
        "n_rows": len(smiles),
        "wall_clock_minutes": round(elapsed / 60, 1),
        "encoder_init": f"tmqm_pretrained_encoder.pt ({n_loaded}/12 backbone tensors)",
        "model": "dmpnn_3cell_resistance_with_sidefeat",
        "focal_loss_gamma": FOCAL_GAMMA,
        "split": "scaffold_70_10_20",
        "timestamp": time.strftime("%Y-%m-%d"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[train] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
