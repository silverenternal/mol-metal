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

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
#: pIC50 column names in ``platinai_4cell.csv`` we keep (drop A2780, per the
#: Phase A / Stage 2.4 spec).  Wf-A42 (§6.65c honest-negative close):
#: the on-disk CSV ships ``pIC50_*`` (no ``_nm`` suffix, lowercase ``pic50``
#: prefix on each column); the trainer previously read ``ic50_mcf7_nm`` etc.,
#: which is the *raw* nM schema that was never committed to disk.  Re-tagged
#: from nM to pIC50 to match the live file (verified by `head -2` on the
#: shipped CSV, which shows ``pIC50_mcf7,pIC50_a549,pIC50_a2780,pIC50_a2780cis``).
PIC50_COLS: Tuple[str, ...] = (
    "pIC50_mcf7",
    "pIC50_a549",
    "pIC50_a2780cis",
)
#: pIC50 ≥ 5.0  ⇔  IC50 ≤ 10 µM — preserves the original ACTIVE_THRESHOLD_NM=10_000
#: spec without the unit-conversion trap (a 10_000 nM raw value yields a
#: pIC50 of 5.0, so threshold is mathematically identical; the nM→pIC50
#: cutoff is the only sane way to read this CSV).
ACTIVE_THRESHOLD_PIC50: float = 5.0
#: SMILES column in the live CSV (capitalised SMILES_Ligands, not `smiles`).
SMILES_COL: str = "SMILES_Ligands"
#: Fewest labelled rows a cell line needs before its test AUC is reported.
#: Below this the AUC is not a measurement of anything -- it is the variance of
#: a handful of pairs -- so `_per_cell_auc` returns nothing for that cell rather
#: than a number.  Named rather than inlined because the split-coverage audit
#: quotes it and a test pins the two together.
MIN_EVAL_ROWS: int = 30

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
    smiles = [r[SMILES_COL] for r in rows]
    n = len(rows)
    active = np.full((n, N_CELLS), -1, dtype=np.int32)
    for j, col in enumerate(PIC50_COLS):
        for i, r in enumerate(rows):
            v = r.get(col, "")
            if v == "" or v is None:
                continue
            try:
                pic50 = float(v)
                active[i, j] = 1 if pic50 >= ACTIVE_THRESHOLD_PIC50 else 0
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


def _finite_or_none(x: float) -> float | None:
    """NaN is not JSON.  A cell that was never evaluated is null, not a number."""
    return float(x) if np.isfinite(x) else None


def _json_safe(obj: Any) -> Any:
    """Replace non-finite floats with None, recursively.

    ``json.dumps`` emits bare ``NaN`` for these, which is not valid JSON and
    which a strict reader (or the registry) then has to guess about.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def _per_cell_auc(
    active: np.ndarray, probs: np.ndarray, cell_lines: Tuple[str, ...] = CELL_LINES
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    from sklearn.metrics import roc_auc_score

    for c, cl in enumerate(cell_lines):
        m = active[:, c] >= 0
        if m.sum() < MIN_EVAL_ROWS:
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
def _build_parser() -> argparse.ArgumentParser:
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
        "--split",
        choices=("scaffold", "scaffold_coverage"),
        default="scaffold",
        help=(
            "scaffold: Bemis-Murcko groups, 70/10/20 by scaffold count, "
            "largest-first (the historical protocol). "
            "scaffold_coverage: same groups, but the assignment targets "
            "70/10/20 of each cell line's labelled rows and breaks size ties "
            "on a hash instead of file order. See "
            "metrics/by_metric/split_coverage_4cell.json."
        ),
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Debug: cap n rows (post-split)"
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config (§5.11). Keys resolve to the dests above; an "
        "explicit CLI flag still wins.",
    )
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Module-level entry point for the config-binding gate (§5.11)."""
    p = _build_parser()
    args = p.parse_args(argv)
    if getattr(args, "config", None):
        from molmetal.scripts.wfA2_config_loader import apply_yaml_overrides

        args = apply_yaml_overrides(p, args, args.config, argv=list(argv or []))
    return args


def main() -> int:
    args = parse_args()

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
    if args.split == "scaffold_coverage":
        # Target each cell line's labelled rows, not scaffold counts.  Without
        # this, a cell whose measurements sit on large scaffolds -- and whose
        # small-scaffold ties happen to fall late in the CSV -- can land below
        # MIN_EVAL_ROWS in test and never be scored (A2780cis: 27 of 600).
        splitter = ScaffoldSplitter(
            fractions=(0.7, 0.1, 0.2),
            coverage={
                cl: active[:, c] >= 0 for c, cl in enumerate(CELL_LINES)
            },
        )
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
                        # n/a, not 0.000: _per_cell_auc skips a cell with <30
                        # labelled rows or a single class, and printing the
                        # skipped case as 0.000 reads as "AUC zero" -- a
                        # different and much worse claim than "not evaluated".
                        f"val_{cl}="
                        + (
                            f"{val_aucs[cl]:.3f}"
                            if cl in val_aucs
                            else "n/a"
                        )
                        for cl in CELL_LINES
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
            + "  ".join(
                f"{cl}=" + (f"{test_aucs[cl]:.3f}" if cl in test_aucs else "n/a")
                for cl in CELL_LINES
            )
            + f"  mean={seed_results[-1]['test_auc_mean']:.4f}"
            + f"  (mean over {len(test_aucs)} of {len(CELL_LINES)} cells)"
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
            # NaN, not 0.0, when the cell was not evaluated: appending 0.0
            # would silently turn "too few labelled test rows" into "the model
            # scored random-or-worse", and nanmean below keeps it out instead.
            per_cell[cl]["per_seed"].append(
                r["test_auc_per_cell"].get(cl, float("nan"))
            )
    for cl in CELL_LINES:
        vals = per_cell[cl]["per_seed"]
        finite = [v for v in vals if not np.isnan(v)]
        per_cell[cl]["auc_mean"] = float(np.mean(finite)) if finite else float("nan")
        per_cell[cl]["auc_std"] = float(np.std(finite)) if finite else float("nan")
        per_cell[cl]["n_seeds_evaluated"] = len(finite)

    n_cells_evaluated = len([cl for cl in CELL_LINES if per_cell[cl]["n_seeds_evaluated"]])
    status = {
        3: "MEASURED",
        2: "MEASURED_2_OF_3_CELLS",
    }.get(n_cells_evaluated, "MEASURED_PARTIAL_CELL_COVERAGE")
    # Labelled rows per cell in the test fold.  All seeds share one split, so
    # this is the count that `_per_cell_auc` actually applied its >=30 bar to.
    test_labelled = {
        cl: int((active[te_idx][:, j] >= 0).sum())
        for j, cl in enumerate(CELL_LINES)
    }
    cells_not_evaluated = [
        cl for cl in CELL_LINES if per_cell[cl]["n_seeds_evaluated"] == 0
    ]

    out = {
        # Derived from --out so a run on a different cohort cannot silently
        # claim the same metric id as the shipped one.  The legacy config
        # writes .../platinai_3cell_resistance.json and is unchanged.
        "metric": Path(args.out).stem,
        # Same model, same hyperparameters, same splitter -- only the labels
        # differ.  That is the one case where supersession is the right word:
        # the synthetic row is a valid measurement of a null test, and this row
        # replaces it as the number to quote.
        "supersedes": "platinai_3cell_resistance"
        if Path(args.out).stem != "platinai_3cell_resistance"
        else None,
        "dataset": str(args.data),
        "description": (
            f"Test ROC-AUC per cell line on {Path(args.data).name} after "
            "dropping the A2780 column. 3-cell-line D-MPNN with "
            "chemoinformatic resistance side-features and tmQM-pretrained init."
        ),
        "unit": "ROC-AUC (0-1, higher better) per cell line",
        "scope": "tier1_precious_metal_anticancer",
        # The name says 3-cell; the evaluation often cannot support 3 cells.
        # `_per_cell_auc` needs >=30 labelled rows with both classes, and a
        # Bemis-Murcko split over 1,966 compounds can leave one cell below that
        # in the test fold.  Naming the count in the status keeps the file from
        # reading as a 3-cell result it never measured.
        "status": status,
        "status_basis": (
            "mean test ROC-AUC over the cell lines that had >=30 labelled "
            "test rows with both classes; a cell that failed that bar is null, "
            "not 0"
        ),
        "source": (
            "molmetal/scripts/train_cisplatin_resistance.py "
            f"(seeds={args.seeds}, epochs={args.epochs}, "
            f"batch={args.batch}, lr={args.lr})"
        ),
        "diagnosis": {
            "a2780_dropped": True,
            # Qualified on purpose: the numbers in the old reason were
            # measured on the SYNTHETIC cohort.  Re-deriving them on real
            # labels is a separate experiment, so the text says which cohort
            # the justification belongs to instead of implying it carries over.
            "reason": (
                "A2780 is dropped by this script's cell-line list. The "
                "original justification (4-cell AUC 0.350; log(IC50) Pearson "
                "r vs A2780cis = 0.003) was measured on the SYNTHETIC "
                "platinai_4cell cohort and has NOT been re-derived on real "
                "labels."
            ),
            "reference": "metrics/by_metric/platinai_4cell_auc.json",
        },
        "value_overall_mean_auc": float(np.mean(means)),
        "value_overall_mean_std": float(np.std(means)),
        "value_a2780cis_auc": _finite_or_none(per_cell["A2780cis"]["auc_mean"]),
        "value_a2780cis_std": _finite_or_none(per_cell["A2780cis"]["auc_std"]),
        "per_cell_line": per_cell,
        "verdict": {
            "n_cells_evaluated": n_cells_evaluated,
            "cells_not_evaluated": cells_not_evaluated,
            "test_labelled_rows_per_cell": test_labelled,
            "why": (
                "A cell is evaluated only when its test fold has >=30 labelled "
                "rows spanning both classes. With one Bemis-Murcko split over "
                "1,966 compounds, a cell can fall below that bar -- its "
                "measurements concentrate in scaffolds the splitter put in "
                "train. The value is then a mean over the cells that remain, "
                "and the status says how many."
            ),
        },
        "per_seed": seed_results,
        "n_seeds": len(seed_results),
        "n_rows": len(smiles),
        "split_sizes": {
            "n_train": seed_results[0]["n_train"] if seed_results else None,
            "n_val": seed_results[0]["n_val"] if seed_results else None,
            "n_test": seed_results[0]["n_test"] if seed_results else None,
            # Every seed shares one scaffold split (the loop re-runs the same
            # splitter on the same SMILES), so the spread across seeds is
            # training noise only.  It is NOT a split-level confidence
            # interval: a single 20% test fold is one draw.
            "note": "one split shared by all seeds; seed std measures init/shuffle/dropout, not split variance",
        },
        "wall_clock_minutes": round(elapsed / 60, 1),
        "encoder_init": f"tmqm_pretrained_encoder.pt ({n_loaded}/12 backbone tensors)",
        "model": "dmpnn_3cell_resistance_with_sidefeat",
        "focal_loss_gamma": FOCAL_GAMMA,
        "split": (
            "scaffold_70_10_20_by_cell_coverage"
            if args.split == "scaffold_coverage"
            else "scaffold_70_10_20"
        ),
        "split_protocol": (
            "Bemis-Murcko groups, one scaffold per split, 70/10/20 of each cell "
            "line's labelled rows, size ties broken by a hash of the scaffold"
            if args.split == "scaffold_coverage"
            else "Bemis-Murcko groups, one scaffold per split, 70/10/20 by "
                 "scaffold count, largest-first (size ties follow file order)"
        ),
        "timestamp": time.strftime("%Y-%m-%d"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_json_safe(out), indent=2))
    print(f"[train] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
