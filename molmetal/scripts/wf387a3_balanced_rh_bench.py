#!/usr/bin/env python3
"""Wf-387-A3 — Balanced Rh benchmark scaffold (CPU-only).

=====================================================================
Wf-387-A3 / Wf-387 ultracode wave — Per Wf-387-A1 verification, the
500-synthetic-Rh architecture (n=500 with 250 synthetic actives + 250
synthetic inactives from ``molmetal/reports/wf139_p7_rh_active_cohort/
synthetic_rh_pos_neg_normalized.json``) gave:

  * Rh seed 3 — test_auc = 0.8879
  * Rh seed 4 — test_auc = 0.8717

These are above the Krasnov 0.86 gate.  The open question is whether
this fold lift is **REAL** (real-cohort signal propagates to test) or
an **ARTIFACT** of the synthetic actives being trivially separable from
the synthetic inactives (max_sim_to_active_for_actives median 0.1277
vs max_sim_to_active_for_inactives median 0.1829 — a 0.0552 gap that
a similarity-based learner can exploit).

This script implements a **balanced Rh benchmark scaffold** that:

  1. Loads a balanced Rh cohort (250 real actives + 250 real inactives
     per Wf-387-A2 design intent).  Falls back to Wf-139 500-synthetic
     cohort if the Wf-387-A2 balanced cohort is not on disk.
  2. Builds a per-cohort Ridge pipeline using the Wf-380-A2 per-cohort
     config (12-d RDKit for n<300, 24-d RDKit+chemprop v3 surrogate for
     n<1000, with Ridge alpha per cohort-size tier).
  3. Runs 5-fold × 5-seed K-fold CV and computes per-fold + per-seed AUC
     via :func:`balanced_rh_aggregate`.
  4. Emits a verdict comparing the **balanced** AUC to the **Wf-387-A1
     synthetic** AUC of 0.8879 / 0.8717:
       - ``ARTIFACT_LIFT``    if balanced AUC drops below 0.6
       - ``PARTIAL_ARTIFACT`` if 0.6 <= balanced AUC < Wf-387-A1 - 0.05
       - ``REAL_LIFT``        if balanced AUC >= Wf-387-A1 - 0.05
       - ``CEILING_BREAK``    if balanced AUC >= 0.86 (Krasnov gate)

Public API
----------
* :func:`train_eval_balanced`    per-fold AUC computation.
* :func:`balanced_rh_aggregate`  per-seed mean / std AUC aggregation.
* :func:`smoke_run_balanced_rh`  smoke-test entry point on a small
                                  synthetic balanced cohort (default
                                  60 SMILES, 30 actives + 30 inactives).

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* **CPU-only** Ridge pipeline with RDKit 12-d / 24-d features.  No torch
  autograd, no D-MPNN, no chemprop v3 encoder forward pass.  Epochs are
  reduced from Wf-379-A1 50 to **30** for fast CPU smoke (per task
  brief).
* Per-cohort config from Wf-380-A2 (12-d for n<300, 24-d for n<1000)
  is the validated methodology for n_lt_300 cohorts (synthetic R²
  +0.1970 ± 0.0276 per Wf-380-A2 §2.2).
* AUC is the binary-classification metric on the active/inactive split
  (not the regression R² used by Wf-380-A2 m2 R²).  A 0.86 gate
  matches the Krasnov ``metal_lomo_auc.json::gate_target``.
* This scaffold is **not** a replacement for the Wf-379-A1 GPU
  retrain.  Real-cohort validation requires USER_OK-gated GPU launch
  per AGENTS.md §5.1 + kill_then_adapt §0.

Cross-references
----------------
* Wf-387-A1: synthetic Rh AUC 0.8879 / 0.8717 on the 500-synthetic
  cohort (the question is whether this lift is real or artifact).
* Wf-387-A2: balanced Rh cohort design intent (250 real actives +
  250 real inactives).  Falls back to Wf-139 500-synthetic if the
  Wf-387-A2 balanced cohort JSON is not on disk.
* Wf-380-A2: per-cohort config matrix (12-d for n<300, 24-d for n<1000,
  50-d for n>=1000).
* Wf-377-A2 §2: 5-alternative-split diagnostic protocol that this
  scaffold extends from R² to AUC.
* AGENTS.md §5.1 + §5.2: standing GPU grant + honest framing mandate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Repo path resolution (Wf-335-M2 / Wf-367-B1 / Wf-380-A2 pattern)
# ---------------------------------------------------------------------------
def _resolve_repo_path(path: str) -> str:
    if os.path.isabs(path) and os.path.exists(path):
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    for start in (here, os.getcwd()):
        cur = start
        for _ in range(6):
            child = os.path.join(cur, "molmetal")
            if os.path.isdir(child) and (
                os.path.isdir(os.path.join(child, "data"))
                or os.path.isdir(os.path.join(child, "scripts"))
            ):
                candidate = os.path.join(cur, path)
                if os.path.exists(candidate):
                    return candidate
                break
            cur = os.path.dirname(cur)
            if not cur or cur == os.path.dirname(cur):
                break
    return os.path.abspath(path)


# ---------------------------------------------------------------------------
# Per-cohort config (Wf-380-A2 §1)
# ---------------------------------------------------------------------------
WF380A2_PER_COHORT_CONFIG: Dict[str, Dict[str, Any]] = {
    "n_lt_300": {
        "tier": "n_lt_300",
        "n_features_expected": 12,
        "model": "ridge",
        "alpha": 1.0,
        "features": "rdkit_12d",
    },
    "n_lt_1000": {
        "tier": "n_lt_1000",
        "n_features_expected": 24,
        "model": "ridge",
        "alpha": 0.5,
        "features": "rdkit_24d_chemprop_v3",
    },
}


def select_per_cohort_config(cohort_size: int) -> Dict[str, Any]:
    """Select per-cohort config from Wf-380-A2 per-cohort matrix.

    Tier semantics (mirrors Wf-380-A2 §1):
      * n < 300 strictly → n_lt_300 (12-d RDKit + Ridge alpha=1.0)
      * n < 1000 strictly → n_lt_1000 (24-d RDKit+chemprop_v3 + alpha=0.5)
      * n >= 1000 → fallback to n_lt_1000 (CPU smoke is n<1000)

    Returns a config dict that :func:`train_eval_balanced` consumes.
    """
    if cohort_size < 300:
        return WF380A2_PER_COHORT_CONFIG["n_lt_300"]
    return WF380A2_PER_COHORT_CONFIG["n_lt_1000"]


# ---------------------------------------------------------------------------
# Section 1: Feature extraction (CPU-only, RDKit 12-d / 24-d)
# ---------------------------------------------------------------------------
def rdkit_features(smiles_list: List[str], n_dims: int = 12) -> np.ndarray:
    """Extract RDKit 12-d (or 24-d) features per SMILES.

    For n_dims == 12 the 12-d RDKit set is used (MolWt, MolLogP, HeavyAtom,
    AromaticRing, RingCount, TPSA, HBA, HBD, MolMR, BalabanJ, BertzCT,
    Phi) per Wf-380-A2 §1 tier ``n_lt_300``.

    For n_dims == 24 the 12-d base + 12-d chemprop-v3-surrogate axes
    (per Wf-367-B1 chemprop_v3_enhanced_features lite) are concatenated.
    Returns ``np.ndarray`` of shape ``(len(smiles_list), n_dims)``.
    On RDKit ImportError falls back to zero features (deterministic).
    """
    n = len(smiles_list)
    out = np.zeros((n, n_dims), dtype=np.float64)
    try:
        from rdkit import Chem, RDLogger  # type: ignore
        from rdkit.Chem import AllChem, Descriptors  # type: ignore

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        return out
    feats: List[List[float]] = []
    for smi in smiles_list:
        row: List[float] = [0.0] * n_dims
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            feats.append(row)
            continue
        # 12-d RDKit base
        row[0] = float(Descriptors.MolWt(mol))
        row[1] = float(Descriptors.MolLogP(mol))
        row[2] = float(mol.GetNumHeavyAtoms())
        row[3] = float(Descriptors.NumAromaticRings(mol))
        row[4] = float(Descriptors.RingCount(mol))
        row[5] = float(Descriptors.TPSA(mol))
        row[6] = float(Descriptors.NumHAcceptors(mol))
        row[7] = float(Descriptors.NumHDonors(mol))
        row[8] = float(Descriptors.MolMR(mol))
        try:
            row[9] = float(Descriptors.BalabanJ(mol))
        except Exception:
            row[9] = 0.0
        try:
            row[10] = float(Descriptors.BertzCT(mol))
        except Exception:
            row[10] = 0.0
        try:
            row[11] = float(Descriptors.Phi(mol))
        except Exception:
            row[11] = 0.0
        # 12-d chemprop-v3-surrogate for n_dims == 24
        if n_dims >= 24:
            atoms = mol.GetAtoms()
            row[12] = float(sum(1 for a in atoms if str(a.GetHybridization()) == "SP3"))
            row[13] = float(sum(1 for a in atoms if a.GetIsAromatic()))
            row[14] = float(sum(1 for a in atoms if a.GetAtomicNum() not in (1, 6)))
            row[15] = float(sum(1 for a in atoms if a.GetFormalCharge() != 0))
            row[16] = float(sum(1 for a in atoms if a.GetTotalNumHs() > 0))
            row[17] = float(sum(1 for a in atoms if a.IsInRing()))
            row[18] = float(len(set(a.GetAtomicNum() for a in atoms)))
            bonds = mol.GetBonds()
            row[19] = float(sum(1 for b in bonds if str(b.GetBondType()) == "SINGLE"))
            row[20] = float(sum(1 for b in bonds if str(b.GetBondType()) == "DOUBLE"))
            row[21] = float(sum(1 for b in bonds if str(b.GetBondType()) == "TRIPLE"))
            row[22] = float(sum(1 for b in bonds if str(b.GetBondType()) == "AROMATIC"))
            row[23] = float(mol.GetNumRotatableBonds())
        feats.append(row)
    return np.asarray(feats, dtype=np.float64)


# ---------------------------------------------------------------------------
# Section 2: Ridge AUC evaluation (numpy closed-form with INTERCEPT)
# ---------------------------------------------------------------------------
def _ridge_fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Ridge fit + INTERCEPT predictor (numpy closed-form fallback).

    Solves ``min ||Xw - y||² + alpha * ||w||²`` with explicit intercept:
    the closed-form ``w = (XᵀX + αI)^-1 Xᵀy`` is augmented by adding a
    constant 1 column to X.  Falls back to ``np.linalg.lstsq`` if the
    normal equations are singular.
    """
    n_train = X_train.shape[0]
    n_feat = X_train.shape[1]
    X1 = np.hstack([X_train, np.ones((n_train, 1), dtype=np.float64)])
    Xt = np.hstack([X_test, np.ones((X_test.shape[0], 1), dtype=np.float64)])
    A = X1.T @ X1 + alpha * np.eye(n_feat + 1, dtype=np.float64)
    b = X1.T @ y_train
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return Xt @ w


def _auc_ranksum(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC via the rank-sum (Mann-Whitney U) statistic.

    Avoids sklearn dependency (mirrors Wf-367-B1 numpy-fallback
    pattern).  Returns ``0.5`` if either class is constant.
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    # Average ranks for ties
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if i != j:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    sum_ranks_pos = float(ranks[y_true == 1].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(max(0.0, min(1.0, auc)))


def _kfold_indices(n: int, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate K-fold (train_idx, val_idx) tuples with a deterministic seed."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        out.append((train.astype(np.int64), val.astype(np.int64)))
    return out


# ---------------------------------------------------------------------------
# Section 3: train_eval_balanced — per-fold AUC pipeline (PUBLIC API)
# ---------------------------------------------------------------------------
def train_eval_balanced(
    train_csv: str,
    test_csv: Optional[str] = None,
    model: str = "dmpnn_multitask",
    n_seeds: int = 5,
    n_folds: int = 5,
    wass_w: float = 0.05,
    epochs: int = 30,
) -> Dict[str, Any]:
    """Per-fold AUC on a balanced Rh train/test split.

    The model name ``dmpnn_multitask`` is a **CPU-only surrogate** that
    selects the Wf-380-A2 per-cohort Ridge pipeline when ``n < 1000``.
    Per AGENTS.md §5.1 + task brief: no torch autograd, no GPU launch,
    30 epochs (vs Wf-379-A1 50 epochs) for fast CPU smoke.

    Parameters
    ----------
    train_csv : str
        Path to balanced Rh train CSV with columns ``smiles,active``.
        Falls back to the Wf-139 500-synthetic cohort if not on disk.
    test_csv : Optional[str]
        Optional path to balanced Rh test CSV.  If ``None`` a 5-fold
        K-fold within ``train_csv`` is used (the per-fold validation set
        becomes the test set).
    model : str
        Model name (default ``dmpnn_multitask``).  Only the CPU surrogate
        is supported in Wf-387-A3.
    n_seeds : int
        Number of random seeds for K-fold (default 5).
    n_folds : int
        Number of folds (default 5).
    wass_w : float
        Wasserstein loss weight (default 0.05).  CPU surrogate ignores
        this — it is a config axis carried over from Wf-379-A1 / Wf-380-A2.
    epochs : int
        Epoch count for CPU smoke (default 30 per task brief).

    Returns
    -------
    Dict[str, Any] with keys:
      ``per_seed_auc`` (List[float]), ``per_fold_auc`` (List[float]),
      ``auc_mean``, ``auc_std``, ``n_train``, ``n_test``, ``config``,
      ``model``, ``epochs``.
    """
    # Resolve paths
    train_path = _resolve_repo_path(train_csv)
    if not os.path.exists(train_path):
        # Fallback to Wf-139 500-synthetic Rh cohort (the "buggy 500-Rh" cohort)
        fallback = _resolve_repo_path(
            "molmetal/reports/wf139_p7_rh_active_cohort/"
            "synthetic_rh_pos_neg_normalized.json"
        )
        if os.path.exists(fallback):
            train_path = fallback
    # Load cohort
    smiles_list: List[str]
    labels: np.ndarray
    if train_path.endswith(".json"):
        with open(train_path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        # Wf-139 schema: {"per_compound": [{"smiles": ..., "is_active": ...}, ...]}
        # Wf-387-A2 schema (design): {"cohort": {"smiles": [...], "active": [...]}}
        per_compound = blob.get("per_compound") or blob.get("cohort", {}).get(
            "compounds"
        )
        if per_compound is None and "smiles" in blob:
            per_compound = [
                {"smiles": s, "is_active": int(a)}
                for s, a in zip(blob["smiles"], blob["active"])
            ]
        smiles_list = [str(p.get("smiles", "")) for p in per_compound]
        labels = np.asarray(
            [int(p.get("is_active", p.get("active", 0))) for p in per_compound],
            dtype=np.int64,
        )
    else:
        # CSV fallback (smiles,active header)
        import csv

        smiles_list = []
        labels_list: List[int] = []
        with open(train_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                smiles_list.append(str(row.get("smiles", "")))
                labels_list.append(int(row.get("active", 0)))
        labels = np.asarray(labels_list, dtype=np.int64)
    # Filter invalid SMILES
    keep_mask = np.asarray([bool(s) for s in smiles_list])
    smiles_list = [s for s, k in zip(smiles_list, keep_mask) if k]
    labels = labels[keep_mask]
    n_total = len(smiles_list)
    if n_total == 0:
        return {
            "per_seed_auc": [],
            "per_fold_auc": [],
            "auc_mean": float("nan"),
            "auc_std": float("nan"),
            "n_train": 0,
            "n_test": 0,
            "config": {},
            "model": model,
            "epochs": epochs,
            "error": "empty cohort after SMILES filter",
        }
    cfg = select_per_cohort_config(n_total)
    # Build features
    X = rdkit_features(smiles_list, n_dims=cfg["n_features_expected"])
    # Standardize (zero-mean, unit-std)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds < 1e-9] = 1.0
    X = (X - means) / stds
    # K-fold × seeds
    per_seed_auc: List[float] = []
    per_fold_auc: List[float] = []
    for seed in range(n_seeds):
        folds = _kfold_indices(n_total, n_folds, seed=seed + 1)
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_train = X[train_idx]
            y_train = labels[train_idx].astype(np.float64)
            X_val = X[val_idx]
            y_val = labels[val_idx]
            # If both classes absent in train, skip the fold
            if len(np.unique(y_train)) < 2:
                continue
            preds = _ridge_fit_predict(
                X_train,
                y_train,
                X_val,
                alpha=float(cfg["alpha"]),
            )
            fold_auc = _auc_ranksum(y_val, preds)
            per_fold_auc.append(fold_auc)
        if per_fold_auc and len(per_fold_auc) >= n_folds * seed + n_folds:
            seed_aucs = per_fold_auc[seed * n_folds : (seed + 1) * n_folds]
        elif per_fold_auc:
            seed_aucs = per_fold_auc[-n_folds:]
        else:
            seed_aucs = []
        if seed_aucs:
            per_seed_auc.append(float(np.mean(seed_aucs)))
    auc_mean = float(np.mean(per_seed_auc)) if per_seed_auc else float("nan")
    auc_std = float(np.std(per_seed_auc)) if per_seed_auc else float("nan")
    return {
        "per_seed_auc": per_seed_auc,
        "per_fold_auc": per_fold_auc,
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "n_train": int(n_total),
        "n_test": int(n_total) if test_csv is None else 0,
        "config": cfg,
        "model": model,
        "epochs": epochs,
        "wass_w": wass_w,
        "train_source": train_path,
        "verdict": _verdict_from_auc(auc_mean),
    }


def _verdict_from_auc(auc_mean: float) -> str:
    """Map AUC mean to the Wf-387 verdict taxonomy (ARTIFACT_LIFT / REAL_LIFT / CEILING_BREAK)."""
    if not np.isfinite(auc_mean):
        return "INSUFFICIENT_DATA"
    if auc_mean >= 0.86:
        return "CEILING_BREAK"
    if auc_mean >= 0.7:
        return "REAL_LIFT"
    if auc_mean >= 0.6:
        return "PARTIAL_ARTIFACT"
    return "ARTIFACT_LIFT"


# ---------------------------------------------------------------------------
# Section 4: balanced_rh_aggregate — per-seed aggregation (PUBLIC API)
# ---------------------------------------------------------------------------
def balanced_rh_aggregate(per_seed_results: List[float]) -> Dict[str, float]:
    """Aggregate per-seed AUC results into mean + std + verdict.

    Parameters
    ----------
    per_seed_results : List[float]
        Per-seed mean AUC values (typically ``len(per_seed_results) ==
        n_seeds``).

    Returns
    -------
    Dict with keys ``mean``, ``std``, ``n_seeds``, ``verdict``,
    ``gate_margin`` (mean − 0.86 Krasnov gate).
    """
    arr = np.asarray([x for x in per_seed_results if np.isfinite(x)], dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "n_seeds": 0,
            "verdict": "INSUFFICIENT_DATA",
            "gate_margin": float("nan"),
        }
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size >= 2 else 0.0
    return {
        "mean": mean,
        "std": std,
        "n_seeds": int(arr.size),
        "verdict": _verdict_from_auc(mean),
        "gate_margin": mean - 0.86,
    }


# ---------------------------------------------------------------------------
# Section 5: smoke_run_balanced_rh — small synthetic balanced cohort (PUBLIC API)
# ---------------------------------------------------------------------------
def _synthetic_balanced_smiles(
    n_active: int = 30, n_inactive: int = 30, seed: int = 42
) -> Tuple[List[str], np.ndarray]:
    """Build a small synthetic balanced SMILES cohort (CPU-only, no RDKit).

    Returns ``(smiles, labels)``.  Active SMILES use a closed-ring template
    with a long carbon chain + a recognizable polar head; inactives use a
    short inert scaffold.  This is intentionally a SMOKETest (verifies the
    Ridge pipeline + AUC computation produce a valid number, not a real-cohort
    benchmark).
    """
    rng = np.random.default_rng(seed)
    actives: List[str] = []
    inactives: List[str] = []
    for i in range(n_active):
        chain = int(rng.integers(6, 14))
        arom = int(rng.integers(0, 3))
        ring = "c1ccccc1" if arom >= 1 else "C1CC1"
        smi = f"{ring}C{'C' * chain}C(=O)O"
        actives.append(smi)
    for i in range(n_inactive):
        chain = int(rng.integers(2, 5))
        smi = f"C{'C' * chain}"
        inactives.append(smi)
    smiles_list = actives + inactives
    labels = np.asarray(
        [1] * n_active + [0] * n_inactive,
        dtype=np.int64,
    )
    return smiles_list, labels


def _smoke_cohort_to_csv(smiles_list: List[str], labels: np.ndarray, path: str) -> str:
    """Write a tiny balanced cohort CSV (smiles,active) to disk for the smoke."""
    import csv

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["smiles", "active"])
        for s, y in zip(smiles_list, labels):
            writer.writerow([s, int(y)])
    return path


def smoke_run_balanced_rh(
    out_dir: str = "molmetal/reports/wf387a3_balanced_rh_smoke/",
    balanced_cohort: Optional[str] = None,
    n_seeds: int = 2,
    n_folds: int = 3,
    epochs: int = 30,
) -> Dict[str, Any]:
    """Smoke-test the balanced Rh pipeline on a small synthetic cohort.

    If ``balanced_cohort`` is provided AND points to a valid balanced
    Rh cohort JSON/CSV, the smoke uses it.  Otherwise the smoke builds
    an inline synthetic balanced cohort (30 actives + 30 inactives, ~60
    total — well below the Wf-380-A2 n_lt_300 tier) and runs the
    Ridge pipeline with ``n_seeds=2 n_folds=3 epochs=30`` to verify
    that valid AUC output is produced.

    Returns
    -------
    Dict[str, Any] with keys ``result`` (train_eval_balanced output),
    ``aggregate`` (balanced_rh_aggregate output), ``smoke_cohort``,
    ``verdict_compare`` (REAL_LIFT / ARTIFACT_LIFT / CEILING_BREAK vs
    Wf-387-A1 synthetic 0.8879 / 0.8717), ``wall_seconds``.
    """
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    cohort_path = balanced_cohort
    using_inline = False
    if not cohort_path or not os.path.exists(_resolve_repo_path(cohort_path)):
        # Build inline synthetic smoke cohort (60 SMILES)
        smoke_smiles, smoke_labels = _synthetic_balanced_smiles(
            n_active=30,
            n_inactive=30,
            seed=42,
        )
        cohort_path = os.path.join(out_dir, "smoke_balanced_cohort.csv")
        _smoke_cohort_to_csv(smoke_smiles, smoke_labels, cohort_path)
        using_inline = True
    result = train_eval_balanced(
        train_csv=cohort_path,
        test_csv=None,
        n_seeds=n_seeds,
        n_folds=n_folds,
        epochs=epochs,
    )
    agg = balanced_rh_aggregate(result.get("per_seed_auc", []))
    # Wf-387-A1 reference: Rh seed 3 = 0.8879, Rh seed 4 = 0.8717
    wf387a1_synthetic_mean = (0.8879 + 0.8717) / 2.0  # 0.8798
    auc_mean = agg.get("mean", float("nan"))
    delta_vs_synthetic = auc_mean - wf387a1_synthetic_mean
    if not np.isfinite(auc_mean):
        verdict_compare = "INSUFFICIENT_DATA"
    elif auc_mean >= 0.86:
        verdict_compare = "CEILING_BREAK"
    elif delta_vs_synthetic >= -0.05:
        verdict_compare = "REAL_LIFT"
    elif auc_mean >= 0.6:
        verdict_compare = "PARTIAL_ARTIFACT"
    else:
        verdict_compare = "ARTIFACT_LIFT"
    out: Dict[str, Any] = {
        "wf": "Wf-387-A3",
        "task": "balanced Rh benchmark scaffold smoke test",
        "scope": "CPU-only; per-cohort Ridge pipeline per Wf-380-A2; "
        "no torch autograd, no D-MPNN, no GPU launch.",
        "upstream": "Wf-387-A1 (synthetic Rh AUC 0.8879 / 0.8717) + "
        "Wf-380-A2 (per-cohort config) + Wf-377-A2 (verdict protocol).",
        "result": result,
        "aggregate": agg,
        "smoke_cohort": cohort_path,
        "using_inline_synthetic": using_inline,
        "wf387a1_synthetic_mean": wf387a1_synthetic_mean,
        "wf387a1_synthetic_max": 0.8879,
        "wf387a1_synthetic_min": 0.8717,
        "delta_vs_wf387a1_synthetic": delta_vs_synthetic,
        "verdict_compare": verdict_compare,
        "epochs": epochs,
        "n_seeds": n_seeds,
        "n_folds": n_folds,
        "wall_seconds": float(time.time() - t0),
        "honest_framing": {
            "scope": "CPU-only smoke scaffold; AUC is MEASURED on the "
            "synthetic balanced cohort (60 SMILES) — NOT on a "
            "real balanced Rh cohort. The verdict "
            "(REAL_LIFT / ARTIFACT_LIFT / CEILING_BREAK) is a "
            "smoke-test verdict that demonstrates the pipeline "
            "produces valid output, NOT a real-cohort gate verdict.",
            "real_cohort_lift": "PROJECTED pending balanced Rh cohort "
            "(Wf-387-A2 deliverable) + USER_OK-gated "
            "GPU launch per AGENTS.md §5.1 + kill_then_adapt §0.",
            "epoch_reduction": "30 epochs (vs Wf-379-A1 50) per task brief — "
            "faster CPU smoke; per-cohort Ridge surrogate "
            "is not epoch-bound (closed-form).",
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = os.path.join(out_dir, "smoke_result.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    return out


# ---------------------------------------------------------------------------
# Section 6: CLI entry point
# ---------------------------------------------------------------------------
def _main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run the synthetic smoke test (default if no flag).",
    )
    p.add_argument(
        "--balanced-cohort",
        default=None,
        help="Path to balanced Rh cohort JSON or CSV. "
        "Falls back to Wf-139 500-synthetic if missing.",
    )
    p.add_argument("--out-dir", default="molmetal/reports/wf387a3_balanced_rh_smoke/")
    p.add_argument("--n-seeds", type=int, default=2)
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=30)
    args = p.parse_args(argv)
    out = smoke_run_balanced_rh(
        out_dir=args.out_dir,
        balanced_cohort=args.balanced_cohort,
        n_seeds=args.n_seeds,
        n_folds=args.n_folds,
        epochs=args.epochs,
    )
    print(
        f"Wf-387-A3 smoke verdict_compare={out['verdict_compare']} "
        f"auc_mean={out['aggregate']['mean']:+.4f} "
        f"std={out['aggregate']['std']:.4f} "
        f"verdict={out['aggregate']['verdict']} "
        f"wall={out['wall_seconds']:.2f}s"
    )
    print(f"Written {args.out_dir}/smoke_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
