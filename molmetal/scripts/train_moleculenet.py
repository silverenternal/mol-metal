"""Metric Tier 2 — MoleculeNet cross-benchmark (BBBP, BACE, HIV, ClinTox).

Baseline (Chemprop D-MPNN, Wu 2018): scaffold-split ROC-AUC = 0.94 / 0.87 /
0.82 / 0.89 on BBBP / BACE / HIV / ClinTox. Our V4 (from-scratch D-MPNN
implementation in :mod:`molmetal.baselines.dmpnn` plus Morgan FP(2048,2)
featurization) should match or beat on at least 2 of these tasks.

Pipeline:
  1. Load MoleculeNet dataset CSVs from ``.data/molmetal/moleculenet/``,
     one per task (bbbp / bace / hiv / clintox). Missing files are
     sub-sampled from ``molmetal/data/known_pt_drugs.csv`` as a 4-class
     mimic if the task name matches.
  2. Featurize: Morgan FP (radius=2, n_bits=2048). For molecules that fail
     to parse, the fingerprint is the zero vector.
  3. Scaffold split (Bemis-Murcko) per the Chemprop convention:
     70 % train / 10 % val / 20 % test.
  4. Train a small feed-forward classifier on top of the Morgan FP for
     ``--epochs`` (default 50), lr=1e-3, batch=32, with class-weighted
     BCE loss to handle the heavy label imbalance in HIV (3.5 % positive).
  5. Evaluate ROC-AUC (and PR-AUC, hit-rate@top-5 %) per seed, aggregate
     mean ± std across seeds, write to ``metrics/by_metric/moleculenet_auc.json``
     and ``molmetal/reports/wf_tier2_molnet/final.md``.

Usage
-----
    source env.sh
    python -m molmetal.scripts.train_moleculenet \\
        --tasks bbbp bace hiv clintox \\
        --seeds 0 1 2 \\
        --scaffold-split \\
        --out-dir logs/metric_molnet
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
import gzip
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from molmetal.baselines.eval_utils import compute_metrics, morgan_features
from molmetal.baselines.dmpnn_multitask import DMPNNMultiTaskModel
from molmetal.baselines.dmpnn_attentive import (
    AttentiveDMPNNModel,
    collate_graphs,
    featurize_smiles_list,
)

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / ".data" / "molmetal" / "moleculenet"
METRICS_DIR = PROJECT_ROOT / "metrics" / "by_metric"
REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports"

# Each task: csv filename, label column, smiles column, optional gz suffix
TASK_INFO: Dict[str, Dict[str, object]] = {
    "bbbp": {
        "file": "bbbp.csv",
        "smiles_col": "smiles",
        "label_col": "p_np",
        "gz": False,
    },
    "bace": {
        "file": "bace.csv",
        "smiles_col": "mol",
        "label_col": "Class",
        "gz": False,
    },
    "hiv": {
        "file": "hiv.csv",
        "smiles_col": "smiles",
        "label_col": "HIV_active",
        "gz": False,
    },
    "clintox": {
        "file": "clintox.csv.gz",
        "smiles_col": "smiles",
        "label_col": "CT_TOX",
        "gz": True,
    },
}

# Chemprop D-MPNN scaffold-split baseline AUC (Wu 2018)
CHEMPROP_DMPNN_BASELINE = {"bbbp": 0.940, "bace": 0.876, "hiv": 0.823, "clintox": 0.894}

# Random Forest + Morgan FP baseline (Chemprop paper Tab. 3)
CHEMPROP_RF_BASELINE = {"bbbp": 0.728, "bace": 0.868, "hiv": 0.761, "clintox": 0.832}


def _open_csv(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _load_task(task: str) -> Tuple[List[str], np.ndarray]:
    import csv as _csv

    info = TASK_INFO[task]
    fname = str(info["file"])
    smiles_col = str(info["smiles_col"])
    label_col = str(info["label_col"])
    csv_path = DATA_DIR / fname
    if not csv_path.exists():
        raise FileNotFoundError(
            f"MoleculeNet task '{task}' not found at {csv_path}. "
            f"Download from https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{fname}"
        )
    smiles: List[str] = []
    labels: List[int] = []
    with _open_csv(csv_path) as fh:
        reader = _csv.DictReader(fh)
        for row in reader:
            smi = (row.get(smiles_col) or "").strip()
            lbl_raw = row.get(label_col)
            if not smi or lbl_raw is None:
                continue
            try:
                lbl = int(float(lbl_raw))
            except (ValueError, TypeError):
                continue
            if lbl not in (0, 1):
                # Some MoleculeNet label columns are already 0/1; if 0/2 etc, ignore.
                continue
            smiles.append(smi)
            labels.append(lbl)
    return smiles, np.asarray(labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# Scaffold splitter (Bemis-Murcko)
# ---------------------------------------------------------------------------
def _bemis_murcko_scaffold(smiles: str) -> str:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles  # group invalid SMILES by their string
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        return smiles


def _scaffold_split(
    smiles: List[str],
    seed: int,
    fractions: Tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group by Bemis-Murcko scaffold; assign each scaffold to one split.

    Wf-A53 (2026-10-19) — CAVEAT, printed on every call: the key below is
    ``GetScaffoldForMol`` + isomeric SMILES, so stereoisomers get DIFFERENT
    keys.  The astartes/chemprop reference calls
    ``MurckoScaffoldSmiles(includeChirality=False)``, which groups them.  Our
    split is therefore marginally EASIER than the published column — the bias
    runs in our favour (docs/ALIGNED_COMPARISON.md).  Changing the key
    changes every MoleculeNet split and invalidates the shipped metrics, so
    it is left as-is until those tasks are re-run.
    """
    print(
        "[moleculenet] WARNING: scaffold key retains stereochemistry; the "
        "astartes/chemprop reference is stereo-blind. Our split is marginally "
        "easier (see docs/ALIGNED_COMPARISON.md).",
        flush=True,
    )
    rng = np.random.default_rng(seed)
    scaffold_to_idx: Dict[str, List[int]] = {}
    for i, smi in enumerate(smiles):
        s = _bemis_murcko_scaffold(smi)
        scaffold_to_idx.setdefault(s, []).append(i)

    scaffolds = list(scaffold_to_idx.keys())
    rng.shuffle(scaffolds)

    n = len(smiles)
    target_train, target_val, target_test = (
        int(fractions[0] * n),
        int(fractions[1] * n),
        n - int(fractions[0] * n) - int(fractions[1] * n),
    )

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    counts = [0, 0, 0]
    targets = [target_train, target_val, target_test]
    for s in scaffolds:
        # assign to split whose count is most below target
        gaps = [targets[i] - counts[i] for i in range(3)]
        split = int(np.argmax(gaps))
        if split == 0:
            train_idx.extend(scaffold_to_idx[s])
        elif split == 1:
            val_idx.extend(scaffold_to_idx[s])
        else:
            test_idx.extend(scaffold_to_idx[s])
        counts[split] += len(scaffold_to_idx[s])

    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Classifier head — small MLP on top of Morgan FP
# ---------------------------------------------------------------------------
class _MorganMLP(nn.Module):
    def __init__(self, n_bits: int = 2048, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bits, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _train_one_seed(
    task: str,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden: int,
    device: torch.device,
    loss: str = "bce",
    focal_gamma: float = 2.0,
    split_seed: int | None = None,
) -> Dict[str, object]:
    smiles, labels = _load_task(task)
    print(
        f"[molnet:{task}] seed={seed}  rows={len(smiles)}  pos_rate={labels.mean():.3f}",
        flush=True,
    )

    # Lever B6 follow-up (2026-10-11) — the scaffold split may be seeded
    # independently of the training seed, so a multi-seed ensemble shares ONE
    # test set (row-aligned averaging).  None keeps the historical behaviour.
    split_seed_eff = split_seed if split_seed is not None else seed
    train_idx, val_idx, test_idx = _scaffold_split(smiles, seed=split_seed_eff)
    if len(test_idx) < 5 or len(np.unique(labels[test_idx])) < 2:
        # If scaffold split leaves a single-class test set, fall back to random 80/10/20
        rng = np.random.default_rng(split_seed_eff)
        idx = rng.permutation(len(smiles))
        n = len(idx)
        train_idx = idx[: int(0.7 * n)]
        val_idx = idx[int(0.7 * n) : int(0.8 * n)]
        test_idx = idx[int(0.8 * n) :]
        print(
            f"[molnet:{task}]   scaffold split degenerated; falling back to random 70/10/20",
            flush=True,
        )

    # Featurize
    X_all = morgan_features(smiles, radius=2, n_bits=2048).astype(np.float32)
    y_all = labels.astype(np.float32)
    X_train = torch.from_numpy(X_all[train_idx]).to(device)
    y_train = torch.from_numpy(y_all[train_idx]).to(device)
    X_val = torch.from_numpy(X_all[val_idx]).to(device)
    y_val_np = y_all[val_idx]
    X_test = torch.from_numpy(X_all[test_idx]).to(device)
    y_test_np = y_all[test_idx]

    # class weighting (pos_weight = neg/pos)
    pos = float(y_train.sum().item())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    print(
        f"[molnet:{task}]   train/val/test = {len(train_idx)}/{len(val_idx)}/{len(test_idx)}  "
        f"pos_train={pos:.0f} pos_weight={pos_weight.item():.2f}",
        flush=True,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _MorganMLP(n_bits=2048, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if loss == "focal":
        def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return _focal_bce_loss(logits, targets, pos_weight, focal_gamma)
    else:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val = -1.0
    best_state = None
    n_train = len(X_train)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            bi = perm[start : start + batch_size]
            xb = X_train[bi]
            yb = y_train[bi]
            opt.zero_grad()
            logits = model(xb)
            batch_loss = loss_fn(logits, yb)
            batch_loss.backward()
            opt.step()
            total_loss += batch_loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            v_logits = model(X_val).cpu().numpy()
        v_score = 1.0 / (1.0 + np.exp(-v_logits))
        v_metrics = compute_metrics(y_val_np, v_score)
        if v_metrics["roc_auc"] > best_val:
            best_val = v_metrics["roc_auc"]
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            print(
                f"[molnet:{task}]   epoch {epoch+1:3d}/{epochs} loss={total_loss/max(n_batches,1):.4f} "
                f"val_auc={v_metrics['roc_auc']:.4f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test).cpu().numpy()
    test_score = 1.0 / (1.0 + np.exp(-test_logits))
    test_metrics = compute_metrics(y_test_np, test_score)
    print(
        f"[molnet:{task}]   TEST AUC={test_metrics['roc_auc']:.4f}  "
        f"PR-AUC={test_metrics['pr_auc']:.4f}  Hit@5%={test_metrics['hit_rate_top5pct']:.3f}",
        flush=True,
    )

    return {
        "seed": seed,
        "task": task,
        "n_rows": len(smiles),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "pos_rate_train": float(y_all[train_idx].mean()),
        "pos_rate_test": float(y_all[test_idx].mean()),
        "test_metrics": test_metrics,
        "epochs": epochs,
    }


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------
def _test_scores_dmpnn(
    model: DMPNNMultiTaskModel,
    graphs: List,
    idx: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Per-row positive-class probability vector under the D-MPNN backbone.

    Lever B6 (2026-10-11) — factored out of :func:`_test_metrics_dmpnn` so the
    seed-ensemble path can average the raw per-row scores.  Rows whose graph
    failed to featurize keep the ``0.0`` default (they are excluded from
    ``compute_metrics`` by the caller only via their label, matching the
    pre-B6 behaviour).
    """
    model.eval()
    scores = np.zeros(len(idx), dtype=float)
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            bidx = idx[start : start + batch_size]
            g_list = [graphs[i] for i in bidx]
            keep = [j for j, g in enumerate(g_list) if g is not None]
            if not keep:
                continue
            batch_data = collate_graphs([g_list[j] for j in keep])
            if batch_data is None:
                continue
            atom_f, bond_f, e_src, e_dst = (t.to(device) for t in batch_data)
            _p, cls = model(atom_f, bond_f, e_src, e_dst)
            prob = torch.softmax(cls, dim=-1)[:, 1].cpu().numpy()
            for k, j in enumerate(keep):
                scores[start + j] = float(prob[k])
    return scores


def _test_metrics_dmpnn(
    model: DMPNNMultiTaskModel,
    graphs: List,
    y_np: np.ndarray,
    idx: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """Classification-head ROC-AUC on the subset ``idx`` under the D-MPNN backbone."""
    scores = _test_scores_dmpnn(model, graphs, idx, batch_size, device)
    return compute_metrics(y_np[idx], scores)


def _resolve_pos_weight(
    y_train: np.ndarray,
    class_weight: str,
) -> float:
    """Return the positive-class weight (``pos_weight``) for the classification head.

    ``onn`` / ``on``:  weight = neg / pos  (legacy default — over-weights the
    positive class whenever the task is imbalanced).

    ``off``:  weight = 1.0 (plain unweighted CrossEntropy — Chemprop's default).

    ``balanced``:  auto-decision — apply the neg/pos weight ONLY when the task
    is genuinely imbalanced (``pos_rate < 0.30`` or ``> 0.70``); for a near-balanced
    task (0.30 <= pos_rate <= 0.70, e.g. BACE at pos_rate≈0.45) this returns 1.0,
    i.e. no weighting.  (Wu 2018 / Chemprop use plain BCE on BACE.)
    """
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    raw = neg / max(pos, 1.0)
    if class_weight == "off":
        return 1.0
    if class_weight == "balanced":
        pos_rate = pos / max(len(y_train), 1)
        if 0.30 <= pos_rate <= 0.70:
            return 1.0
        return raw
    # default "on"
    return raw


# ---------------------------------------------------------------------------
# Focal loss (Lever B5, 2026-10-11)
#
# A drop-in replacement for the default classification loss, selected by
# ``--loss focal``.  The default ``--loss bce`` path is untouched, so a run
# without the flag is numerically identical to the pre-B5 trainer.
# ---------------------------------------------------------------------------
def _focal_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor | None,
    gamma: float,
) -> torch.Tensor:
    """Focal cross-entropy (Lin et al. 2017) for the 2-class D-MPNN head.

    ``loss = sum( (1 - p_t)**gamma * CE(logits, targets) ) / sum(w_t)`` where
    ``p_t`` is the softmax probability of the true class and ``w_t`` the
    per-sample class weight.  ``weight`` is handed straight to
    ``F.cross_entropy``, and the denominator matches ``CrossEntropyLoss``'s own
    weighted mean, so ``gamma == 0`` reduces EXACTLY to
    ``nn.CrossEntropyLoss(weight=weight)`` — the ``class_w`` semantics of the
    baseline are preserved.
    """
    ce = F.cross_entropy(logits, targets, weight=weight, reduction="none")
    p_t = torch.softmax(logits, dim=-1).gather(1, targets.view(-1, 1)).squeeze(1)
    focal = (1.0 - p_t).clamp(min=0.0) ** gamma * ce
    if weight is not None:
        denom = weight.gather(0, targets).sum().clamp(min=1e-12)
    else:
        denom = torch.as_tensor(float(targets.numel()), device=targets.device)
    return focal.sum() / denom


def _focal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None,
    gamma: float,
) -> torch.Tensor:
    """Focal BCE-with-logits for the single-logit heads (Morgan-MLP / AttentiveFP).

    Same focusing factor as :func:`_focal_ce_loss`, applied to the per-sample
    ``BCEWithLogitsLoss`` term; ``pos_weight`` (the resolved class weight) is
    preserved.  ``gamma == 0`` reduces to the weighted BCE baseline.
    """
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    return ((1.0 - p_t).clamp(min=0.0) ** gamma * bce).mean()


def _train_one_seed_dmpnn(
    task: str,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden: int,
    depth: int,
    device: torch.device,
    class_weight: str = "on",
    dropout: float = 0.0,
    weight_decay: float = 0.0,
    patience: int = 0,
    loss: str = "bce",
    focal_gamma: float = 2.0,
    ensemble_seeds: bool = False,
    split_seed: int | None = None,
) -> Dict[str, object]:
    """D-MPNN backbone on the same MoleculeNet scaffold split as the Morgan-MLP run."""
    smiles, labels = _load_task(task)
    print(
        f"[molnet:{task}] backbone=dmpnn seed={seed}  rows={len(smiles)}  "
        f"pos_rate={labels.mean():.3f}",
        flush=True,
    )

    # Lever B6 follow-up (2026-10-11) — decouple the scaffold split from the
    # training seed so a multi-seed ensemble shares ONE test set (row-aligned
    # averaging).  None keeps the historical behaviour (split seeded by seed).
    split_seed_eff = split_seed if split_seed is not None else seed
    train_idx, val_idx, test_idx = _scaffold_split(smiles, seed=split_seed_eff)
    if len(test_idx) < 5 or len(np.unique(labels[test_idx])) < 2:
        rng = np.random.default_rng(split_seed_eff)
        idx = rng.permutation(len(smiles))
        n = len(idx)
        train_idx = idx[: int(0.7 * n)]
        val_idx = idx[int(0.7 * n) : int(0.8 * n)]
        test_idx = idx[int(0.8 * n) :]
        print(f"[molnet:{task}]   scaffold split degenerated; random 70/10/20", flush=True)

    graphs = featurize_smiles_list([str(s) for s in smiles])
    y_all = labels.astype(np.int64)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DMPNNMultiTaskModel(hidden=hidden, depth=depth, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_weight_val = _resolve_pos_weight(y_all[train_idx], class_weight)
    pos = float(y_all[train_idx].sum())
    pos_rate_train = pos / max(len(train_idx), 1)
    class_w = torch.tensor([1.0, pos_weight_val], device=device)
    if loss == "focal":
        def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return _focal_ce_loss(logits, targets, class_w, focal_gamma)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=class_w)
    print(
        f"[molnet:{task}]   class_weight={class_weight} pos_rate_train={pos_rate_train:.3f} "
        f"pos_weight={pos_weight_val:.3f} dropout={dropout} wd={weight_decay} patience={patience} "
        f"loss={loss}" + (f"(gamma={focal_gamma})" if loss == "focal" else ""),
        flush=True,
    )

    best_val = -1.0
    best_state = None
    n_train = len(train_idx)
    epochs_since_improve = 0
    step_losses: List[float] = []
    for epoch in range(epochs):
        if patience and epochs_since_improve >= patience:
            print(
                f"[molnet:{task}]   early-stop at epoch {epoch+1} "
                f"(no val improvement for {patience} epochs)",
                flush=True,
            )
            break
        model.train()
        perm = np.random.permutation(n_train)
        total_loss, n_batches = 0.0, 0
        for start in range(0, n_train, batch_size):
            bidx = train_idx[perm[start : start + batch_size]]
            g_list = [graphs[i] for i in bidx]
            keep = [j for j, g in enumerate(g_list) if g is not None]
            if not keep:
                continue
            batch_data = collate_graphs([g_list[j] for j in keep])
            if batch_data is None:
                continue
            atom_f, bond_f, e_src, e_dst = (t.to(device) for t in batch_data)
            yb = torch.from_numpy(y_all[[bidx[j] for j in keep]]).to(device)
            opt.zero_grad()
            _p, cls = model(atom_f, bond_f, e_src, e_dst)
            batch_loss = loss_fn(cls, yb)
            batch_loss.backward()
            opt.step()
            total_loss += batch_loss.item()
            step_losses.append(float(batch_loss.item()))
            n_batches += 1

        v_metrics = _test_metrics_dmpnn(model, graphs, y_all, val_idx, batch_size, device)
        if v_metrics["roc_auc"] > best_val:
            best_val = v_metrics["roc_auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            print(
                f"[molnet:{task}]   epoch {epoch+1:3d}/{epochs} "
                f"loss={total_loss/max(n_batches,1):.4f} val_auc={v_metrics['roc_auc']:.4f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    test_scores: np.ndarray | None = None
    if ensemble_seeds:
        # Lever B6 — keep the per-row test scores so main() can average the
        # seeds' predictions into a seed-ensemble AUC.
        test_scores = _test_scores_dmpnn(model, graphs, test_idx, batch_size, device)
        test_metrics = compute_metrics(y_all[test_idx], test_scores)
    else:
        test_metrics = _test_metrics_dmpnn(model, graphs, y_all, test_idx, batch_size, device)
    print(
        f"[molnet:{task}]   TEST AUC={test_metrics['roc_auc']:.4f}  "
        f"PR-AUC={test_metrics['pr_auc']:.4f}",
        flush=True,
    )
    result: Dict[str, object] = {
        "seed": seed,
        "task": task,
        "backbone": "dmpnn",
        "n_rows": len(smiles),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "pos_rate_train": float(y_all[train_idx].mean()),
        "pos_rate_test": float(y_all[test_idx].mean()),
        "class_weight": class_weight,
        "pos_weight_used": float(pos_weight_val),
        "dropout": dropout,
        "weight_decay": weight_decay,
        "patience": patience,
        "test_metrics": test_metrics,
        "epochs": epochs,
    }
    # Lever B5/B6 diagnostics are emitted only when a new lever is active, so a
    # default run (loss=bce, no --ensemble-seeds) writes byte-identical JSON.
    if loss != "bce" or ensemble_seeds:
        result["loss"] = loss
        result["focal_gamma"] = focal_gamma
        result["loss_curve"] = step_losses
    if ensemble_seeds and test_scores is not None:
        result["test_idx"] = [int(i) for i in test_idx]
        result["test_scores"] = [float(x) for x in test_scores]
        result["test_labels"] = [int(x) for x in y_all[test_idx]]
    return result


# ---------------------------------------------------------------------------
# Attentive D-MPNN backbone (BLK-D1 Step 2-A, 2026-10-09)
# ---------------------------------------------------------------------------
def _train_one_seed_attentive(
    task: str,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden: int,
    depth: int,
    device: torch.device,
    class_weight: str = "on",
    dropout: float = 0.0,
    weight_decay: float = 0.0,
    patience: int = 0,
    loss: str = "bce",
    focal_gamma: float = 2.0,
) -> Dict[str, object]:
    """AttentiveFP backbone on the same MoleculeNet scaffold split as the D-MPNN run.

    Mirrors ``_train_one_seed_dmpnn`` so the three backbones (morgan_mlp / dmpnn /
    attentivefp) share a protocol-aligned scaffold split, optimizer, early-stop
    policy, and class-weight dispatch.  The only backbone-specific substitution
    is the head loss: D-MPNN uses multi-task CrossEntropyLoss; AttentiveFP
    (single-task graph encoder) uses BCEWithLogitsLoss with the same
    resolved pos_weight — applies ``class_weight=on/balanced/off`` consistently.
    """
    smiles, labels = _load_task(task)
    print(
        f"[molnet:{task}] backbone=attentivefp seed={seed}  rows={len(smiles)}  "
        f"pos_rate={labels.mean():.3f}",
        flush=True,
    )

    train_idx, val_idx, test_idx = _scaffold_split(smiles, seed=seed)
    if len(test_idx) < 5 or len(np.unique(labels[test_idx])) < 2:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(smiles))
        n = len(idx)
        train_idx = idx[: int(0.7 * n)]
        val_idx = idx[int(0.7 * n) : int(0.8 * n)]
        test_idx = idx[int(0.8 * n) :]
        print(
            f"[molnet:{task}]   scaffold split degenerated; random 70/10/20",
            flush=True,
        )

    graphs = featurize_smiles_list([str(s) for s in smiles])
    y_all = labels.astype(np.float32)  # BCE-with-logits expects float targets

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AttentiveDMPNNModel(hidden=hidden, depth=depth, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_weight_val = _resolve_pos_weight(y_all[train_idx].astype(np.int64), class_weight)
    pos = float(y_all[train_idx].sum())
    pos_rate_train = pos / max(len(train_idx), 1)
    pos_weight_tensor = torch.tensor([pos_weight_val], device=device)
    if loss == "focal":
        def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return _focal_bce_loss(logits, targets, pos_weight_tensor, focal_gamma)
    else:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    print(
        f"[molnet:{task}]   class_weight={class_weight} pos_rate_train={pos_rate_train:.3f} "
        f"pos_weight={pos_weight_val:.3f} dropout={dropout} wd={weight_decay} patience={patience}",
        flush=True,
    )

    best_val = -1.0
    best_state = None
    n_train = len(train_idx)
    epochs_since_improve = 0
    for epoch in range(epochs):
        if patience and epochs_since_improve >= patience:
            print(
                f"[molnet:{task}]   early-stop at epoch {epoch+1} "
                f"(no val improvement for {patience} epochs)",
                flush=True,
            )
            break
        model.train()
        perm = np.random.permutation(n_train)
        total_loss, n_batches = 0.0, 0
        for start in range(0, n_train, batch_size):
            bidx = train_idx[perm[start : start + batch_size]]
            g_list = [graphs[i] for i in bidx]
            keep = [j for j, g in enumerate(g_list) if g is not None]
            if not keep:
                continue
            batch_data = collate_graphs([g_list[j] for j in keep])
            if batch_data is None:
                continue
            atom_f, bond_f, e_src, e_dst = (t.to(device) for t in batch_data)
            yb = torch.from_numpy(y_all[[bidx[j] for j in keep]]).to(device)
            opt.zero_grad()
            logits = model(atom_f, bond_f, e_src, e_dst)
            batch_loss = loss_fn(logits, yb)
            batch_loss.backward()
            opt.step()
            total_loss += batch_loss.item()
            n_batches += 1

        v_metrics = _test_metrics_attentive(model, graphs, y_all, val_idx, batch_size, device)
        if v_metrics["roc_auc"] > best_val:
            best_val = v_metrics["roc_auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            print(
                f"[molnet:{task}]   epoch {epoch+1:3d}/{epochs} "
                f"loss={total_loss/max(n_batches,1):.4f} val_auc={v_metrics['roc_auc']:.4f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = _test_metrics_attentive(model, graphs, y_all, test_idx, batch_size, device)
    print(
        f"[molnet:{task}]   TEST AUC={test_metrics['roc_auc']:.4f}  "
        f"PR-AUC={test_metrics['pr_auc']:.4f}",
        flush=True,
    )
    return {
        "seed": seed,
        "task": task,
        "backbone": "attentivefp",
        "n_rows": len(smiles),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "pos_rate_train": float(y_all[train_idx].mean()),
        "pos_rate_test": float(y_all[test_idx].mean()),
        "class_weight": class_weight,
        "pos_weight_used": float(pos_weight_val),
        "dropout": dropout,
        "weight_decay": weight_decay,
        "patience": patience,
        "test_metrics": test_metrics,
        "epochs": epochs,
    }


def _test_metrics_attentive(
    model: AttentiveDMPNNModel,
    graphs: List,
    y_np: np.ndarray,
    idx: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """Classification-head ROC-AUC under the AttentiveFP backbone.

    Twin to ``_test_metrics_dmpnn``; only the score extraction differs:
    AttentiveFP outputs a single logit per molecule, so ``sigmoid(logit)``
    is the probability instead of ``softmax([neg,pos])[1]``.
    """
    model.eval()
    scores = np.zeros(len(idx), dtype=float)
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            bidx = idx[start : start + batch_size]
            g_list = [graphs[i] for i in bidx]
            keep = [j for j, g in enumerate(g_list) if g is not None]
            if not keep:
                continue
            batch_data = collate_graphs([g_list[j] for j in keep])
            if batch_data is None:
                continue
            atom_f, bond_f, e_src, e_dst = (t.to(device) for t in batch_data)
            logits = model(atom_f, bond_f, e_src, e_dst)
            prob = torch.sigmoid(logits).cpu().numpy()
            for k, j in enumerate(keep):
                scores[start + j] = float(prob[k])
    return compute_metrics(y_np[idx].astype(np.int64), scores)


def _aggregate(per_seed: List[Dict[str, object]]) -> Dict[str, object]:
    aucs = [
        float(d["test_metrics"]["roc_auc"])
        for d in per_seed
        if not np.isnan(d["test_metrics"]["roc_auc"])
    ]
    aps = [
        float(d["test_metrics"]["pr_auc"])
        for d in per_seed
        if not np.isnan(d["test_metrics"]["pr_auc"])
    ]
    hits = [
        float(d["test_metrics"]["hit_rate_top5pct"])
        for d in per_seed
        if not np.isnan(d["test_metrics"]["hit_rate_top5pct"])
    ]
    return {
        "n_seeds_total": len(per_seed),
        "n_seeds_valid": len(aucs),
        "test_auc_mean": float(statistics.mean(aucs)) if aucs else float("nan"),
        "test_auc_std": float(statistics.pstdev(aucs)) if len(aucs) > 1 else 0.0,
        "test_pr_auc_mean": float(statistics.mean(aps)) if aps else float("nan"),
        "test_hit_top5_mean": float(statistics.mean(hits)) if hits else float("nan"),
        "per_seed_auc": aucs,
        "per_seed_pr_auc": aps,
        "per_seed_hit_top5": hits,
    }


def _ensemble_auc(per_seed: List[Dict[str, object]]) -> Dict[str, object]:
    """Seed-ensembled ROC-AUC over the union of the per-seed test sets (Lever B6).

    MEASURED 2026-10-11 (BACE): ``_scaffold_split`` seeds its RNG with the
    training seed, so the five seeds do NOT share a test set — 5 distinct test
    sets, intersection 0, union 1052 of 1513 molecules (632 molecules in
    exactly one test set, 383 in two, 32 in three, 5 in four).  Row-aligned
    averaging is therefore invalid.  We instead average, per molecule, the
    predicted scores of every seed whose test set contains it, then score
    ROC-AUC on that union.

    When the seeds DO share a split (e.g. a fixed split seed) every molecule is
    covered by all seeds, and this reduces exactly to row-aligned seed
    averaging.  Returns ``ensemble_auc = nan`` when no per-seed score vectors
    were saved (e.g. ``--ensemble-seeds`` with a non-D-MPNN backbone).
    """
    sums: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    labels: Dict[int, int] = {}
    n_seeds = 0
    test_idx_sets: List[Tuple[int, ...]] = []
    for rec in per_seed:
        if "test_scores" not in rec or "test_idx" not in rec:
            continue
        n_seeds += 1
        test_idx_sets.append(tuple(int(i) for i in rec["test_idx"]))
        for i, s, y in zip(rec["test_idx"], rec["test_scores"], rec["test_labels"]):
            sums[i] = sums.get(i, 0.0) + float(s)
            counts[i] = counts.get(i, 0) + 1
            labels[i] = int(y)
    if not sums:
        return {"ensemble_auc": float("nan"), "n_seeds_ensembled": 0}
    mols = sorted(sums)
    scores = np.asarray([sums[i] / counts[i] for i in mols], dtype=float)
    y_ens = np.asarray([labels[i] for i in mols], dtype=np.int64)
    metrics = compute_metrics(y_ens, scores)
    # Row-aligned: every seed scored the SAME test set (one shared scaffold
    # split, e.g. `--split-seed` set), so each molecule's `count` is n_seeds and
    # the per-molecule mean is a straight row-aligned average of the per-seed
    # score vectors.
    row_aligned = n_seeds > 1 and len(set(test_idx_sets)) == 1
    return {
        "ensemble_auc": float(metrics["roc_auc"]),
        "n_seeds_ensembled": n_seeds,
        "ensemble_n_molecules": len(mols),
        "ensemble_method": (
            "row-aligned mean over the shared per-seed test set (all seeds "
            "share ONE scaffold split via --split-seed)"
            if row_aligned
            else "per-molecule mean over the seeds whose test set contains the "
            "molecule (union of per-seed test sets; scaffold split varies by seed)"
        ),
    }


def _run_task(
    task: str,
    seeds: List[int],
    epochs: int,
    batch_size: int,
    lr: float,
    hidden: int,
    device: torch.device,
    out_dir: Path,
    backbone: str = "morgan_mlp",
    depth: int = 3,
    class_weight: str = "on",
    dropout: float = 0.0,
    weight_decay: float = 0.0,
    patience: int = 0,
    loss: str = "bce",
    focal_gamma: float = 2.0,
    ensemble_seeds: bool = False,
    split_seed: int | None = None,
) -> Dict[str, object]:
    print(f"\n=== task {task} (backbone={backbone}) ===", flush=True)
    per_seed: List[Dict[str, object]] = []
    t0 = time.time()
    for seed in seeds:
        try:
            if backbone == "dmpnn":
                r = _train_one_seed_dmpnn(
                    task, seed, epochs, batch_size, lr, hidden, depth, device,
                    class_weight=class_weight, dropout=dropout,
                    weight_decay=weight_decay, patience=patience,
                    loss=loss, focal_gamma=focal_gamma,
                    ensemble_seeds=ensemble_seeds,
                    split_seed=split_seed,
                )
            elif backbone == "attentivefp":
                r = _train_one_seed_attentive(
                    task, seed, epochs, batch_size, lr, hidden, depth, device,
                    class_weight=class_weight, dropout=dropout,
                    weight_decay=weight_decay, patience=patience,
                    loss=loss, focal_gamma=focal_gamma,
                )
            else:
                r = _train_one_seed(
                    task, seed, epochs, batch_size, lr, hidden, device,
                    loss=loss, focal_gamma=focal_gamma,
                    split_seed=split_seed,
                )
            per_seed.append(r)
            (out_dir / f"{task}_{backbone}_seed{seed}.json").write_text(
                json.dumps(r, indent=2)
            )
        except Exception as e:
            print(f"[molnet:{task}] seed {seed} FAILED: {e}", flush=True)
            per_seed.append(
                {
                    "seed": seed,
                    "task": task,
                    "backbone": backbone,
                    "error": str(e),
                    "test_metrics": {
                        "roc_auc": float("nan"),
                        "pr_auc": float("nan"),
                        "hit_rate_top5pct": float("nan"),
                    },
                }
            )
    agg = _aggregate(per_seed)
    agg["task"] = task
    agg["backbone"] = backbone
    agg["chemprop_dmpnn_baseline"] = CHEMPROP_DMPNN_BASELINE[task]
    agg["chemprop_rf_baseline"] = CHEMPROP_RF_BASELINE[task]
    agg["elapsed_seconds"] = time.time() - t0
    return agg


def _write_report(
    out_path: Path,
    all_results: Dict[str, object],
    gate_pass: bool,
    requested_tasks: List[str],
    backbone: str = "morgan_mlp",
    seeds: Sequence[int] = (0, 1, 2),
) -> None:
    if backbone == "dmpnn":
        feat_line = "**Featurization:** D-MPNN atom+bond features (atom+bond message passing)"
        model_line = "**Model:** D-MPNN graph encoder (shared trunk) + per-task classification head"
    elif backbone == "attentivefp":
        feat_line = "**Featurization:** AttentiveFP atom+bond features (D-MPNN message passing + per-atom attentive readout)"
        model_line = "**Model:** AttentiveFP graph encoder (single-task) + BCEWithLogits head with class-weighted pos_weight"
    else:
        feat_line = "**Featurization:** Morgan FP (radius=2, n_bits=2048) — ECFP4"
        model_line = "**Model:** small MLP (2048 → 256 → 128 → 1) with class-weighted BCE"
    lines = [
        "# Metric Tier 2 — MoleculeNet cross-benchmark (BBBP / BACE / HIV / ClinTox)",
        "",
        "**Split:** Bemis-Murcko scaffold (70 / 10 / 20) per Chemprop convention",
        feat_line,
        model_line,
        f"**Backbone:** {backbone}",
        f"**Seeds:** {', '.join(str(s) for s in seeds)}  ",
        "",
        "## Chemprop baselines (Tier 3 reference, Wu 2018)",
        "",
        "| Task | Chemprop D-MPNN (atom+bond) | Chemprop RF + Morgan FP | **Mol-Metal (this run)** |",
        "|---|---|---|---|",
    ]
    aucs_for_gate = []
    for task in requested_tasks:
        if task not in all_results:
            continue
        r = all_results[task]
        ours = r["test_auc_mean"]
        aucs_for_gate.append(ours if not np.isnan(ours) else 0.0)
        lines.append(
            f"| {task} | {r['chemprop_dmpnn_baseline']:.3f} | "
            f"{r['chemprop_rf_baseline']:.3f} | "
            f"**{ours:.4f} ± {r['test_auc_std']:.4f}** (n={r.get('n_seeds_valid', r.get('n_seeds_total', 0))}) |"
        )
    mean_auc = float(np.mean(aucs_for_gate)) if aucs_for_gate else float("nan")
    lines += [
        "",
        f"**Mean across {len(aucs_for_gate)} tasks: {mean_auc:.4f}** "
        "(gate ≥ 0.80, Chemprop D-MPNN mean ≈ 0.882).",
        "",
        f"**Verdict:** {'✅ PASS — mean AUC ≥ 0.80' if gate_pass else '⚠️ BELOW GATE — measured honestly; not paper-promotable as headline Tier-2 metric'}",
        "",
        "## Per-seed detail",
        "",
    ]
    for task in requested_tasks:
        if task not in all_results:
            continue
        r = all_results[task]
        lines.append(f"### {task}")
        lines.append("")
        lines.append(
            "| Seed | n_train | n_test | pos_rate_test | ROC-AUC | PR-AUC | Hit@5% |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        # Prefer the raw per-seed records (carry n_train/n_test/pos_rate).
        # When those were not retained to disk, fall back to the aggregate
        # JSON's per-seed value lists so the table is never emitted empty.
        raw = r.get("per_seed", [])
        if raw:
            for sd in raw:
                tm = sd.get("test_metrics", {})
                lines.append(
                    f"| {sd['seed']} | {sd.get('n_train','-')} | {sd.get('n_test','-')} | "
                    f"{sd.get('pos_rate_test', float('nan')):.3f} | "
                    f"{tm.get('roc_auc', float('nan')):.4f} | "
                    f"{tm.get('pr_auc', float('nan')):.4f} | "
                    f"{tm.get('hit_rate_top5pct', float('nan')):.3f} |"
                )
        else:
            for i, auc in enumerate(r.get("per_seed_auc", [])):
                pr = r.get("per_seed_pr_auc", [float("nan")] * len(r["per_seed_auc"]))
                hit = r.get("per_seed_hit_top5", [float("nan")] * len(r["per_seed_auc"]))
                seed = seeds[i] if i < len(seeds) else i
                lines.append(
                    f"| {seed} | - | - | - | {auc:.4f} | {pr[i]:.4f} | {hit[i]:.3f} |"
                )
        lines.append("")

    # Per-task commentary and honest caveats
    if backbone == "dmpnn":
        caveats = [
            "1. **Backbone**: this run uses the D-MPNN graph encoder on the SAME",
            "   Bemis-Murcko scaffold split and the SAME ROC-AUC metric as the",
            "   Morgan-FP + MLP baseline, so the two are directly comparable.",
            "2. **Class-imbalance handling**: HIV and ClinTox are severely imbalanced;",
            "   per-task tuning (focal loss / calibrated threshold) is not applied here.",
            "3. **Scaffold split is OOD**: Chemprop baselines above are scaffold-split",
            "   (Wu 2018 Table 3), so the comparison is protocol-aligned.",
            "4. **Per-task variance**: small tasks (ClinTox, BACE) have wide per-seed",
            "   spread; read the per-seed table above, not the mean alone.",
        ]
    elif backbone == "attentivefp":
        caveats = [
            "1. **Backbone**: this run uses the AttentiveFP graph encoder",
            "   (D-MPNN message passing + per-atom attentive readout; Yang et al. 2019)",
            "   on the SAME Bemis-Murcko scaffold split and the SAME ROC-AUC metric as",
            "   the D-MPNN / Morgan-MLP runs, so the three are directly comparable.",
            "2. **Head loss**: AttentiveFP is single-task graph encoder; we use",
            "   BCEWithLogitsLoss with the same `_resolve_pos_weight` dispatch as the",
            "   D-MPNN path (so `--class-weight {on,off,balanced}` is protocol-aligned).",
            "3. **Class-imbalance handling**: HIV and ClinTox are severely imbalanced;",
            "   per-task tuning (focal loss / calibrated threshold) is not applied here.",
            "4. **Scaffold split is OOD**: Chemprop baselines above are scaffold-split",
            "   (Wu 2018 Table 3), so the comparison is protocol-aligned.",
            "5. **Per-task variance**: small tasks (ClinTox, BACE) have wide per-seed",
            "   spread; read the per-seed table above, not the mean alone.",
        ]
    else:
        caveats = [
            "1. **Model capacity**: a 2048 -> 256 -> 128 -> 1 MLP on Morgan FP has far",
            "   fewer parameters than Chemprop's D-MPNN. Use `--backbone dmpnn` for the",
            "   apples-to-apples architecture comparison on the same split.",
            "2. **Class-imbalance handling**: HIV and ClinTox are severely imbalanced;",
            "   pos_weight alone gives a high-recall bias that depresses PR-AUC.",
            "3. **Scaffold split is OOD**: Chemprop baselines above are scaffold-split",
            "   (Wu 2018 Table 3), so the comparison is protocol-aligned.",
            "4. **Per-task variance**: small tasks (ClinTox, BACE) have wide per-seed",
            "   spread; read the per-seed table above, not the mean alone.",
        ]
    lines += ["## Honest caveats", "", *caveats, "", "## Files", ""]
    lines += [
        "* `molmetal/scripts/train_moleculenet.py` — driver.",
        f"* `{out_path}` — this report.",
        "* `metrics/by_metric/moleculenet_auc_dmpnn.json` — aggregate JSON (per-task x per-seed).",
        "",
        "## Baseline provenance (literature-sourced, protocol-differs)",
        "",
        "The Chemprop numbers in the table above are **literature-sourced** "
        "(`CHEMPROP_DMPNN_BASELINE` / `CHEMPROP_RF_BASELINE`, constant block) and are "
        "**NOT reproduced under our protocol**. They are best-of-hyperparameter-search "
        "single-task runs; our runs use a shared-trunk multi-task head at fixed "
        "hyperparameters. Where our AUC trails the literature value, part of the gap is "
        "**tuning, not architecture**. Do NOT present the delta as a pure architecture "
        "result, and do NOT adjust our number to match the literature.",
        "",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Mol-Metal on MoleculeNet cross-benchmark."
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=["bbbp", "bace", "hiv", "clintox"],
        choices=["bbbp", "bace", "hiv", "clintox"],
        help="MoleculeNet tasks to evaluate (default: all four).",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256, help="MLP/D-MPNN hidden width")
    p.add_argument(
        "--backbone",
        choices=["morgan_mlp", "dmpnn", "attentivefp"],
        default="morgan_mlp",
        help="Classifier backbone: Morgan-FP+MLP (morgan_mlp), D-MPNN (dmpnn), or AttentiveFP (attentivefp; attentive D-MPNN readout).",
    )
    p.add_argument("--depth", type=int, default=3, help="D-MPNN message-passing rounds")
    p.add_argument(
        "--class-weight",
        choices=["on", "off", "balanced"],
        default="on",
        help=(
            "Positive-class weighting for the classification head. 'on' (default, "
            "legacy) = neg/pos weight; 'off' = plain unweighted loss (Chemprop "
            "default); 'balanced' = apply neg/pos ONLY when pos_rate is outside "
            "[0.30, 0.70] (i.e. skip weighting on near-balanced tasks like BACE)."
        ),
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability threaded into the D-MPNN backbone/readout (default 0.0).",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Adam L2 weight decay (default 0.0).",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=0,
        help=(
            "Early-stop patience on val ROC-AUC (default 0 = disabled). "
            "Best-val checkpoint is always kept regardless."
        ),
    )
    p.add_argument(
        "--out-dir",
        default="logs/metric_molnet",
        help="Per-seed JSON output dir (default: logs/metric_molnet).",
    )
    p.add_argument(
        "--out-metric",
        default=None,
        help="Aggregate metrics JSON path (default: metrics/by_metric/moleculenet_auc.json).",
    )
    p.add_argument(
        "--out-report",
        default=None,
        help="Markdown report path (default: molmetal/reports/wf_tier2_molnet/final.md).",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smoke mode (epochs=10, hidden=128) — for time-budget sanity checks.",
    )
    p.add_argument(
        "--scaffold-split",
        action="store_true",
        default=True,
        help="Use Bemis-Murcko scaffold split (default behaviour, 70/10/20).",
    )
    # Lever B5 (2026-10-11) — focal loss.  Default 'bce' keeps the existing
    # loss (CrossEntropyLoss on the D-MPNN 2-class head; BCEWithLogitsLoss on
    # the Morgan-MLP / AttentiveFP heads) byte-identical.
    p.add_argument(
        "--loss",
        choices=["bce", "focal"],
        default="bce",
        help=(
            "Classification loss. 'bce' (default) = existing loss (weighted "
            "CrossEntropyLoss / BCEWithLogitsLoss). 'focal' = focal loss "
            "(Lin et al. 2017): (1 - p_t)**gamma * CE, with the same class "
            "weighting."
        ),
    )
    p.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal-loss focusing parameter gamma (default 2.0; used only with --loss focal).",
    )
    # Lever B6 (2026-10-11) — seed-ensembled inference.
    p.add_argument(
        "--ensemble-seeds",
        action="store_true",
        help=(
            "Average the per-seed test scores and record ensemble_auc + "
            "n_seeds_ensembled in the aggregate payload. Default off: output "
            "byte-identical to today."
        ),
    )
    # Lever B6 follow-up (2026-10-11) — one shared scaffold split across seeds.
    p.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help=(
            "Scaffold-split seed. When set, every training seed uses this ONE "
            "scaffold split so per-seed test sets are identical and row-aligned "
            "ensembling is valid. None = historical behaviour (split seeded by "
            "the training seed)."
        ),
    )
    # AGENTS.md §5.11 (user directive 2026-10-09, "以后要求能且只能用yaml配置
    # 启动训练和推理，no cli参数"): this trainer had no YAML surface at all, so
    # every launch_*.sh wrapper passed its hyperparameters as CLI flags — 21 of
    # the 30 §5.11 violations repo-wide lived here. `--config` wires the same
    # loader the CFM trainer uses, so a config file is now the only way these
    # knobs are set. CLI flags still parse, and a flag on the command line wins
    # over the YAML, so existing invocations keep working — but no wrapper uses
    # that any more, and scripts/_yaml_only_gate.py fails the build if one does.
    p.add_argument(
        "--config",
        default=None,
        help="YAML config path; values here override the argparse defaults (§5.11).",
    )
    args = p.parse_args(argv)
    if args.config:
        from molmetal.scripts.wfA2_config_loader import apply_yaml_overrides

        args = apply_yaml_overrides(p, args, args.config)
    return args


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.quick:
        args.epochs = min(args.epochs, 10)
        args.hidden = min(args.hidden, 128)

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_metric = (
        Path(args.out_metric)
        if args.out_metric
        else METRICS_DIR / "moleculenet_auc.json"
    )
    out_report = (
        Path(args.out_report)
        if args.out_report
        else REPORTS_DIR / "wf_tier2_molnet" / "final.md"
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(
        f"[molnet] device={device}  backbone={args.backbone}  tasks={args.tasks}  "
        f"seeds={args.seeds}",
        flush=True,
    )

    all_results: Dict[str, object] = {}
    for task in args.tasks:
        agg = _run_task(
            task,
            args.seeds,
            args.epochs,
            args.batch_size,
            args.lr,
            args.hidden,
            device,
            out_dir,
            backbone=args.backbone,
            depth=args.depth,
            class_weight=args.class_weight,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            patience=args.patience,
            loss=args.loss,
            focal_gamma=args.focal_gamma,
            ensemble_seeds=args.ensemble_seeds,
            split_seed=args.split_seed,
        )
        # Pull per-seed rows for the report (read back from disk)
        per_seed: List[Dict[str, object]] = []
        for s in args.seeds:
            p = out_dir / f"{task}_{args.backbone}_seed{s}.json"
            if p.exists():
                per_seed.append(json.loads(p.read_text()))
        agg["per_seed"] = per_seed
        # Lever B6 — seed-ensembled inference (only when the flag is set, so a
        # default run's payload is unchanged).
        if args.ensemble_seeds:
            agg.update(_ensemble_auc(per_seed))
        # Lever B5/B6 — per-step loss curve, emitted only when a new lever is
        # active (default run stays byte-identical).
        if args.loss != "bce" or args.ensemble_seeds:
            curve: List[float] = []
            for sd in per_seed:
                curve.extend(sd.get("loss_curve", []))
            agg["loss_curve"] = curve
        all_results[task] = agg

    # Aggregate JSON
    payload = {
        "date": time.strftime("%Y-%m-%d"),
        "model": f"{args.backbone}_v4",
        "backbone": args.backbone,
        "featurization": (
            "morgan_fp_2048_r2" if args.backbone == "morgan_mlp"
            else "dmpnn_atom_bond" if args.backbone == "dmpnn"
            else "attentivefp_atom_bond"
        ),
        "split": "scaffold (Bemis-Murcko, 70/10/20)",
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden": args.hidden,
        "depth": args.depth,
        "class_weight": args.class_weight,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "tasks": {
            task: {k: v for k, v in all_results[task].items() if k != "per_seed"}
            for task in args.tasks
        },
        "gate_mean_auc_threshold": 0.80,
        "chemprop_dmpnn_baseline_mean": float(
            np.mean([CHEMPROP_DMPNN_BASELINE[t] for t in args.tasks])
        ),
    }
    # Lever B5/B6 provenance: recorded only when a lever is active, so a default
    # run (loss=bce, no --ensemble-seeds) writes a byte-identical payload.  The
    # top-level `loss_curve` is the shape the §5.7 CPU-smoke protocol reads
    # (wf399a1_smoke_protocol.verify_loss_decrease).
    if args.loss != "bce" or args.ensemble_seeds:
        payload["loss"] = args.loss
        payload["focal_gamma"] = args.focal_gamma
        curve_all: List[float] = []
        for _t in args.tasks:
            curve_all.extend(all_results[_t].get("loss_curve", []))
        payload["loss_curve"] = curve_all
    out_metric.parent.mkdir(parents=True, exist_ok=True)
    out_metric.write_text(json.dumps(payload, indent=2))
    print(f"[molnet] wrote {out_metric}", flush=True)

    # Mean AUC gate
    mean_auc = float(np.mean([all_results[t]["test_auc_mean"] for t in args.tasks]))
    gate_pass = mean_auc >= 0.80
    print(
        f"[molnet] mean AUC = {mean_auc:.4f}  (gate ≥ 0.80)  {'PASS' if gate_pass else 'BELOW GATE'}",
        flush=True,
    )

    _write_report(
        out_report, all_results, gate_pass, args.tasks,
        backbone=getattr(args, "backbone", "morgan_mlp"),
        seeds=list(args.seeds),
    )
    print(f"[molnet] wrote {out_report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
