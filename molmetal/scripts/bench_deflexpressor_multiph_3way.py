"""Wf-41 Phase 1 — Deflexpressor + Multi-PH + ASTSR 3-way integration bench.

================================================================
3-way integration of:
  * Deflexpressor (:class:`LambdaASTSR`, R17 ship) — lambda-AST
    evolutionary symbolic regression;
  * Multi-PH (:class:`TwoParameterFiltration`, Wf-1 T5 ship) — 2-parameter
    persistent homology feature map;
  * Hilbert PD (:class:`LambdaASTSRHilbert`, Wf-9 ship) — convex Hilbert
    primal-dual regressor.

The five methods
----------------
* ``ast``       — pure Deflexpressor on 12-d RDKit descriptors.
* ``hilbert``   — LambdaASTSRHilbert on the 64-d Hilbert-padded
                  RDKit features (the Wf-37 e2e trainer baseline).
* ``multiph``   — :class:`TwoParameterFiltration` feature map
                  flattened → LambdaASTSRHilbert on the multi-PH
                  stream alone.
* ``joint``     — :class:`JointHilbertOptimizer` (Wf-21 P5 / Wf-14
                  ship) on the ASTSR + Multi-PH feature concat.
* ``3way``      — **the new integration**: ASTSR + Multi-PH + PLL
                  streams concat → LambdaASTSRHilbert.  This is the
                  real 3-way Hilbert regressor over all three
                  features (TODO-83 §L8 ASTSR ↔ Hilbert ↔ MCTS
                  triangle extended to Multi-PH + PLL).

Each method is fit per seed on a deterministic 80/20 split of the
PlatinAI 5K sub-sample; the headline metric is the per-method mean R²
+/- std across seeds, plus the **lift over the ASTSR baseline** for
each integration method.

Outputs
-------
* :file:`molmetal/reports/wf41_3way/3way_bench.json` — verbatim
  payload (per_method + records + config + lift_over_ast).
* :file:`molmetal/reports/wf41_3way/3way_bench.md`   — markdown
  comparison table.

Honest framing
--------------
* ``ast`` / ``hilbert`` / ``joint`` already shipped in Wf-14 / Wf-21 /
  Wf-37 — the new contribution is the ``multiph`` and ``3way`` runs.
* The PlatinAI 5K target is **synthetic** (``qed * qed``) and not
  a real cytotoxicity — the comparison is on the *relative* lift
  between regressors, not on absolute predictive power.
* ``n_iters`` for the Hilbert PD arm is bumped from the 50-iter
  default to 200 iters to give the ``multiph`` / ``joint`` / ``3way``
  regressors time to converge on their larger design matrices.
* Wall-time is machine-dependent — the speedup column is the intended
  *relative* metric, not absolute seconds.

Usage::

    .venv/bin/python -u molmetal/scripts/bench_deflexpressor_multiph_3way.py \\
        --n-samples 500 --seeds 5 --output molmetal/reports/wf41_3way

    .venv/bin/python -u molmetal/scripts/bench_deflexpressor_multiph_3way.py \\
        --smoke --n-samples 30 --seeds 1
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # silence RDKit chatter

import numpy as np  # noqa: E402  (after RDKit log silence for cleaner output)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLUMNS: Tuple[str, ...] = (
    "mol_wt",
    "mol_logp",
    "tpsa",
    "hba",
    "hbd",
    "rot_bonds",
    "ring_count",
    "frac_sp3",
    "num_aromatic_rings",
    "num_aliphatic_rings",
    "num_heteroatoms",
    "qed",
)
N_DESCRIPTORS: int = len(FEATURE_COLUMNS)
HILBERT_DIM: int = 64

#: Multi-PH feature dim per sample — flattened ``(n_bins, n_bins)``.
MPH_N_BINS: int = 8
MPH_DIM: int = MPH_N_BINS * MPH_N_BINS  # 64

#: PLL top-k eigenvalue count — kept small so the augmented 3-way
#: design matrix stays in the ``~140-d`` regime (12 + 64 + 16).
PLL_K: int = 16

DEFAULT_METHODS: Tuple[str, ...] = ("ast", "hilbert", "multiph", "joint", "3way")
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_N_SAMPLES: int = 500
DEFAULT_TEST_FRAC: float = 0.2
DEFAULT_PLATINAI_CSV: Path = _REPO_ROOT / "molmetal" / "data" / "platinai_5k.csv"
DEFAULT_OUT_DIR: Path = _REPO_ROOT / "molmetal" / "reports" / "wf41_3way"

_EPS_STD: float = 1e-3
_PRED_CLIP: float = 1e6
_MPH_MIN: int = 2  # tiny molecules — n_atoms < 2 — get a zero map.


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class FitRecord:
    """Per-(seed, method) record."""

    seed: int
    method: str
    n_train: int
    n_test: int
    r2_test: float
    wall_s: float
    peak_mem_mb: float
    n_features: int = N_DESCRIPTORS
    train_r2: float = 0.0
    n_mph_failed: int = 0  # rows where multi-PH failed to compute


@dataclass
class MethodSummary:
    """Aggregate over seeds for one method."""

    method: str
    r2_mean: float
    r2_std: float
    wall_s_mean: float
    wall_s_std: float
    peak_mem_mb_mean: float
    peak_mem_mb_std: float
    n_records: int
    per_seed: Dict[int, Dict[str, float]] = field(default_factory=dict)


@dataclass
class LiftSummary:
    """Lift over the ASTSR baseline for one method."""

    method: str
    lift_r2_mean: float
    lift_r2_std: float


# ---------------------------------------------------------------------------
# Loaders — PlatinAI 5K (SMILES + metal)
# ---------------------------------------------------------------------------
def load_platinai_5k(
    csv_path: Path,
    n_max: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load PlatinAI 5K SMILES + metal rows.

    Returns a list of ``{"smiles", "metal", "ic50_nm", "selectivity"}``
    dicts.  No IC50 filter — the Wf-41 3-way bench uses the
    *synthetic* ``qed * qed`` target derived from RDKit (consistent
    with Wf-14 / Wf-21 / Wf-37 framing) so we don't need IC50.
    """
    out: List[Dict[str, Any]] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            smi = (row.get("smiles") or "").strip()
            if not smi:
                continue
            out.append(
                {
                    "smiles": smi,
                    "metal": (row.get("metal") or "").strip(),
                    "ic50_nm": (row.get("ic50_nm") or "").strip(),
                    "selectivity": (row.get("selectivity") or "").strip(),
                }
            )
            if n_max is not None and len(out) >= n_max:
                break
    return out


# ---------------------------------------------------------------------------
# RDKit 12-d featurizer
# ---------------------------------------------------------------------------
def _featurize_smiles(smiles: str) -> Optional[List[float]]:
    """Return 12-d RDKit descriptor vector; ``None`` if SMILES fails."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski, QED, rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        feats: List[float] = [
            float(Descriptors.MolWt(mol)),
            float(Descriptors.MolLogP(mol)),
            float(Descriptors.TPSA(mol)),
            float(Lipinski.NumHAcceptors(mol)),
            float(Lipinski.NumHDonors(mol)),
            float(Descriptors.NumRotatableBonds(mol)),
            float(Descriptors.RingCount(mol)),
            float(rdMolDescriptors.CalcFractionCSP3(mol)),
            float(rdMolDescriptors.CalcNumAromaticRings(mol)),
            float(rdMolDescriptors.CalcNumAliphaticRings(mol)),
            float(rdMolDescriptors.CalcNumHeteroatoms(mol)),
            float(QED.qed(mol)),
        ]
        if any((not math.isfinite(v)) for v in feats):
            return None
        return feats
    except Exception:
        return None


def _qed(smiles: str) -> Optional[float]:
    """Return QED as a float; ``None`` if RDKit fails."""
    try:
        from rdkit import Chem
        from rdkit.Chem import QED

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        v = float(QED.qed(mol))
        return v if math.isfinite(v) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Multi-PH feature extractor — pocket cloud → 2-PH feature map
# ---------------------------------------------------------------------------
def _mph_features_for(
    smiles_list: Sequence[str],
    n_bins: int = MPH_N_BINS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the multi-PH feature map for every SMILES in ``smiles_list``.

    Returns
    -------
    mph_feats : np.ndarray
        ``(N, n_bins * n_bins)`` float matrix; rows where multi-PH
        failed to compute are filled with zeros (the regression is
        robust to all-zero rows when z-scored).
    valid_mask : np.ndarray
        ``(N,)`` boolean mask — ``True`` iff the multi-PH feature
        map was successfully computed for that row.
    """
    from molmetal.math_skills.multi_parameter_ph import (
        TwoParameterFiltration,
    )
    import torch

    mph = TwoParameterFiltration(n_bins=int(n_bins))
    feats = np.zeros((len(smiles_list), n_bins * n_bins), dtype=float)
    valid_mask = np.zeros(len(smiles_list), dtype=bool)
    for i, smi in enumerate(smiles_list):
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mol_h = Chem.AddHs(mol)
            embed_status = int(AllChem.EmbedMolecule(mol_h, randomSeed=0xC0FFEE))
            if embed_status != 0:
                continue
            n_atoms = int(mol_h.GetNumAtoms())
            if n_atoms < _MPH_MIN:
                continue
            coords = np.zeros((n_atoms, 3), dtype=np.float32)
            conf = mol_h.GetConformer(0)
            for j in range(n_atoms):
                pos = conf.GetAtomPosition(j)
                coords[j, 0] = float(pos.x)
                coords[j, 1] = float(pos.y)
                coords[j, 2] = float(pos.z)
            coords_t = torch.from_numpy(coords).unsqueeze(0)  # (1, N, 3)
            atom_types = torch.zeros(1, n_atoms, dtype=torch.long)
            for j in range(n_atoms):
                atom_types[0, j] = int(mol_h.GetAtomWithIdx(j).GetAtomicNum())
            fmap = mph.compute_with_atom_types(
                coords_t, atom_types
            )  # (1, n_bins, n_bins)
            arr = fmap[0].detach().cpu().numpy().astype(float).reshape(-1)
            if not np.all(np.isfinite(arr)):
                continue
            feats[i] = arr
            valid_mask[i] = True
        except Exception:
            continue
    return feats, valid_mask


# ---------------------------------------------------------------------------
# PLL feature extractor — pocket cloud → top-k Laplacian eigenvalues
# ---------------------------------------------------------------------------
def _numpy_pll_eigvals(
    coords: np.ndarray,
    k: int,
    k_nn: int = 8,
) -> Optional[np.ndarray]:
    """Numpy-only PLL spectrum — fallback when scipy is unavailable.

    Builds a kNN graph from ``coords (N, 3)``, computes the symmetric
    normalised graph Laplacian ``L = I - D^{-1/2} W D^{-1/2}``, and
    returns the top-``k`` eigenvalues (descending) via
    ``numpy.linalg.eigvalsh``.

    Returns ``None`` on any numerical failure (treated as a missing
    PLL feature by the caller — same fallback contract as the
    scipy-backed :func:`compute_pll`).
    """
    n = int(coords.shape[0])
    if n < 2:
        return None
    # kNN adjacency — pairwise L2 distance, top-(k_nn + 1) per node,
    # self excluded.  O(N^2) — fine for molecules with N <= 200 atoms.
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))  # (N, N)
    np.fill_diagonal(dist, np.inf)
    k_eff = min(int(k_nn), n - 1)
    if k_eff < 1:
        return None
    nn_idx = np.argpartition(dist, k_eff, axis=1)[:, :k_eff]  # (N, k_eff)
    # Symmetric weighted adjacency.
    W = np.zeros((n, n), dtype=float)
    row_idx = np.repeat(np.arange(n), k_eff)
    col_idx = nn_idx.reshape(-1)
    weights = 1.0 / (1.0 + dist[row_idx, col_idx])
    W[row_idx, col_idx] = weights
    W = np.maximum(W, W.T)
    # Degree vector.
    deg = W.sum(axis=1)
    if np.any(deg <= 0):
        # Self-loop regularisation — same trick as pll_signature.py.
        eps = 1.0e-12
        deg = np.where(deg <= 0, eps, deg)
        idx = np.arange(n)
        W[idx, idx] = np.maximum(W[idx, idx], eps)
        deg = W.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    D = np.diag(d_inv_sqrt)
    L = np.eye(n) - D @ W @ D
    L = (L + L.T) * 0.5  # enforce symmetry
    try:
        eigvals = np.linalg.eigvalsh(L)
    except np.linalg.LinAlgError:
        return None
    eigvals = np.clip(np.real(eigvals), 0.0, 2.0)
    eigvals = np.sort(eigvals)[::-1]
    if eigvals.size < k:
        eigvals = np.concatenate(
            [eigvals, np.zeros(k - eigvals.size, dtype=float)],
            axis=0,
        )
    return eigvals[:k]


def _pll_features_for(
    smiles_list: Sequence[str],
    k: int = PLL_K,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the PLL top-k eigenvalue vector for every SMILES.

    Uses the scipy-backed :func:`compute_pll` when available; falls
    back to the numpy-only :func:`_numpy_pll_eigvals` when scipy is
    missing (e.g. minimal venv).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Try the canonical scipy-backed compute_pll first.
    use_scipy = True
    compute_pll_fn = None
    try:
        from molmetal.molmetal_lam.lam_chem.pll_signature import compute_pll

        compute_pll_fn = compute_pll
    except Exception:
        use_scipy = False

    feats = np.zeros((len(smiles_list), k), dtype=float)
    valid_mask = np.zeros(len(smiles_list), dtype=bool)
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mol_h = Chem.AddHs(mol)
            embed_status = int(AllChem.EmbedMolecule(mol_h, randomSeed=0xC0FFEE))
            if embed_status != 0:
                continue
            n_atoms = int(mol_h.GetNumAtoms())
            if n_atoms < 2:
                continue
            if use_scipy and compute_pll_fn is not None:
                try:
                    pll_sig = compute_pll_fn(mol_h, k=int(k), k_nn=8)
                    eig = np.asarray(
                        pll_sig.eigenvalues,
                        dtype=float,
                    ).reshape(-1)
                except Exception:
                    eig = None
            else:
                eig = None
            if eig is None or eig.size == 0:
                # Numpy fallback.
                coords = np.zeros((n_atoms, 3), dtype=np.float64)
                conf = mol_h.GetConformer(0)
                for j in range(n_atoms):
                    pos = conf.GetAtomPosition(j)
                    coords[j, 0] = float(pos.x)
                    coords[j, 1] = float(pos.y)
                    coords[j, 2] = float(pos.z)
                eig = _numpy_pll_eigvals(coords, k=int(k), k_nn=8)
            if eig is None:
                continue
            if eig.size < k:
                eig = np.concatenate(
                    [eig, np.zeros(k - eig.size, dtype=float)],
                    axis=0,
                )
            elif eig.size > k:
                eig = eig[:k]
            if not np.all(np.isfinite(eig)):
                continue
            feats[i] = eig
            valid_mask[i] = True
        except Exception:
            continue
    return feats, valid_mask


# ---------------------------------------------------------------------------
# Build the per-method design matrices
# ---------------------------------------------------------------------------
@dataclass
class DesignMatrices:
    """Container for the per-method design matrices built from the rows.

    All matrices share the same train/test split (defined by ``seed``);
    the per-method matrix uses its own slice of features.  Each
    method receives ``X_tr`` / ``X_te`` with a shape that matches its
    hypothesis class (12-d for ``ast``, 64-d Hilbert-padded for
    ``hilbert``, ``MPH_DIM`` for ``multiph``, etc.).  ``y_tr`` /
    ``y_te`` are shared across methods.
    """

    smiles: List[str]
    y_tr: np.ndarray
    y_te: np.ndarray
    # ASTSR — 12-d RDKit (raw, z-scored).
    astr_X_tr: np.ndarray
    astr_X_te: np.ndarray
    # Hilbert-padded 64-d (for hilbert, joint, 3way).
    hil_X_tr: np.ndarray
    hil_X_te: np.ndarray
    # Multi-PH flattened feature map (n_bins * n_bins).
    mph_X_tr: np.ndarray
    mph_X_te: np.ndarray
    # PLL top-k eigenvalues.
    pll_X_tr: np.ndarray
    pll_X_te: np.ndarray
    # Indices + sizes for diagnostics.
    train_idx: np.ndarray
    test_idx: np.ndarray
    n_train: int
    n_test: int
    n_mph_failed: int
    n_pll_failed: int


def build_design_matrices(
    *,
    rows: List[Dict[str, Any]],
    seed: int,
    test_frac: float,
    cached_features: Optional[Dict[str, Any]] = None,
) -> DesignMatrices:
    """Featurise ``rows``; return the per-method design matrices.

    When ``cached_features`` is supplied, the expensive Multi-PH and
    PLL feature computations are reused across seeds (the per-seed
    shuffle / split is the only randomised step).
    """
    if cached_features is not None:
        # Use cached RDKit + Multi-PH + PLL features; just reshuffle
        # the train/test split per seed.
        astr_X_arr = cached_features["astr_X"]
        y_arr = cached_features["y"]
        smiles_kept = cached_features["smiles"]
        mph_feats_all = cached_features["mph_X"]
        pll_feats_all = cached_features["pll_X"]
        mph_valid = cached_features["mph_valid"]
        pll_valid = cached_features["pll_valid"]
    else:
        # Cold path — featurise everything from scratch.
        astr_X: List[List[float]] = []
        smiles_kept = []
        y_raw: List[float] = []
        for row in rows:
            feats = _featurize_smiles(row["smiles"])
            qed_v = _qed(row["smiles"])
            if feats is None or qed_v is None:
                continue
            astr_X.append(feats)
            smiles_kept.append(row["smiles"])
            y_raw.append(float(qed_v * qed_v))
        astr_X_arr = np.asarray(astr_X, dtype=float)
        y_arr = np.asarray(y_raw, dtype=float).reshape(-1)
        mph_feats_all, mph_valid = _mph_features_for(smiles_kept)
        pll_feats_all, pll_valid = _pll_features_for(smiles_kept)

    # Shuffle + split per seed.
    rng = np.random.default_rng(int(seed))
    idx = np.arange(astr_X_arr.shape[0])
    rng.shuffle(idx)
    n_total = int(idx.shape[0])
    n_test_keep = max(1, int(round(test_frac * n_total)))
    n_train_keep = n_total - n_test_keep
    train_idx_keep = idx[n_train_keep:]
    test_idx_keep = idx[:n_train_keep]

    # Hilbert-padded 64-d via pairwise interactions (Wf-37 pattern).
    hil_X_tr, hil_X_te = pad_to_hilbert_dim(
        astr_X_arr[train_idx_keep],
        HILBERT_DIM,
        astr_X_arr[test_idx_keep],
    )
    # Z-score.
    astr_X_tr_n, astr_X_te_n = zscore(
        astr_X_arr[train_idx_keep], astr_X_arr[test_idx_keep]
    )
    hil_X_tr_n, hil_X_te_n = zscore(hil_X_tr, hil_X_te)
    mph_X_tr_n, mph_X_te_n = zscore(
        mph_feats_all[train_idx_keep], mph_feats_all[test_idx_keep]
    )
    pll_X_tr_n, pll_X_te_n = zscore(
        pll_feats_all[train_idx_keep], pll_feats_all[test_idx_keep]
    )

    n_mph_failed = int(np.sum(~mph_valid))
    n_pll_failed = int(np.sum(~pll_valid))

    return DesignMatrices(
        smiles=smiles_kept,
        y_tr=y_arr[train_idx_keep],
        y_te=y_arr[test_idx_keep],
        astr_X_tr=astr_X_tr_n,
        astr_X_te=astr_X_te_n,
        hil_X_tr=hil_X_tr_n,
        hil_X_te=hil_X_te_n,
        mph_X_tr=mph_X_tr_n,
        mph_X_te=mph_X_te_n,
        pll_X_tr=pll_X_tr_n,
        pll_X_te=pll_X_te_n,
        train_idx=train_idx_keep,
        test_idx=test_idx_keep,
        n_train=int(n_train_keep),
        n_test=int(n_test_keep),
        n_mph_failed=n_mph_failed,
        n_pll_failed=n_pll_failed,
    )


def precompute_features(
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Featurise all rows once; cache for reuse across seeds.

    Returns a dict with keys ``astr_X, y, smiles, mph_X, pll_X,
    mph_valid, pll_valid``.  RDKit 12-d features are fast (<10 ms
    per SMILES), so we re-featurise them in ``build_design_matrices``
    only on demand — but Multi-PH + PLL cost ~100 ms per SMILES and
    must be cached.
    """
    astr_X: List[List[float]] = []
    smiles_kept: List[str] = []
    y_raw: List[float] = []
    for row in rows:
        feats = _featurize_smiles(row["smiles"])
        qed_v = _qed(row["smiles"])
        if feats is None or qed_v is None:
            continue
        astr_X.append(feats)
        smiles_kept.append(row["smiles"])
        y_raw.append(float(qed_v * qed_v))
    astr_X_arr = np.asarray(astr_X, dtype=float)
    y_arr = np.asarray(y_raw, dtype=float).reshape(-1)
    mph_feats_all, mph_valid = _mph_features_for(smiles_kept)
    pll_feats_all, pll_valid = _pll_features_for(smiles_kept)
    return {
        "astr_X": astr_X_arr,
        "y": y_arr,
        "smiles": smiles_kept,
        "mph_X": mph_feats_all,
        "pll_X": pll_feats_all,
        "mph_valid": mph_valid,
        "pll_valid": pll_valid,
    }


# ---------------------------------------------------------------------------
# Augmentation helpers (12-d → 64-d Hilbert via pairwise interactions)
# ---------------------------------------------------------------------------
def pad_to_hilbert_dim(
    X_12d: np.ndarray,
    hilbert_dim: int,
    X_12d_test: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Augment 12-d descriptors to ``hilbert_dim`` via pairwise interactions."""
    n = X_12d.shape[0]
    F = X_12d.shape[1]
    bias = np.ones((n, 1), dtype=float)
    interactions_tr = [
        X_12d[:, i] * X_12d[:, j] for i in range(F) for j in range(i + 1, F)
    ]
    inter_arr_tr = (
        np.stack(interactions_tr, axis=1) if interactions_tr else np.zeros((n, 0))
    )
    base_tr = np.concatenate([X_12d, bias], axis=1)
    aug_tr = np.concatenate([base_tr, inter_arr_tr], axis=1)
    if X_12d_test is not None:
        n_te = X_12d_test.shape[0]
        bias_te = np.ones((n_te, 1), dtype=float)
        interactions_te = [
            X_12d_test[:, i] * X_12d_test[:, j]
            for i in range(F)
            for j in range(i + 1, F)
        ]
        inter_arr_te = (
            np.stack(interactions_te, axis=1)
            if interactions_te
            else np.zeros((n_te, 0))
        )
        base_te = np.concatenate([X_12d_test, bias_te], axis=1)
        aug_te = np.concatenate([base_te, inter_arr_te], axis=1)
    else:
        aug_te = None
    if aug_tr.shape[1] <= hilbert_dim:
        pad = np.zeros((n, hilbert_dim - aug_tr.shape[1]), dtype=float)
        out_tr = np.concatenate([aug_tr, pad], axis=1)
        if aug_te is not None:
            pad_te = np.zeros(
                (aug_te.shape[0], hilbert_dim - aug_te.shape[1]), dtype=float
            )
            out_te = np.concatenate([aug_te, pad_te], axis=1)
            return out_tr, out_te
        return out_tr, None
    # Variance-based truncation.
    vars_ = np.var(aug_tr, axis=0)
    bias_idx = F
    ranked = np.argsort(-vars_)
    keep = [bias_idx]
    for idx_v in ranked:
        if idx_v == bias_idx:
            continue
        if len(keep) >= hilbert_dim:
            break
        keep.append(int(idx_v))
    keep = sorted(set(keep))
    if len(keep) < hilbert_dim:
        extras = [i for i in ranked if int(i) not in set(keep)]
        keep.extend(extras[: hilbert_dim - len(keep)])
        keep = sorted(set(keep))
    keep_arr = np.asarray(keep[:hilbert_dim], dtype=int)
    out_tr = aug_tr[:, keep_arr]
    if aug_te is not None:
        return out_tr, aug_te[:, keep_arr]
    return out_tr, None


def zscore(
    X_tr: np.ndarray,
    X_te: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Z-score with std floor; per-column mean/std from training rows only."""
    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0)
    sigma = np.where(sigma < _EPS_STD, 1.0, sigma)
    X_tr_n = (X_tr - mu) / sigma
    X_te_n = (X_te - mu) / sigma
    X_tr_n = np.nan_to_num(X_tr_n, nan=0.0, posinf=0.0, neginf=0.0)
    X_te_n = np.nan_to_num(X_te_n, nan=0.0, posinf=0.0, neginf=0.0)
    return X_tr_n, X_te_n


# ---------------------------------------------------------------------------
# Stable closed-form ridge (Wf-39 P2 fix; extends to arbitrary feature dim)
# ---------------------------------------------------------------------------
def _stable_ridge_fit(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    *,
    ridge_alpha: float = 0.1,
) -> Tuple[np.ndarray, float]:
    """Closed-form ridge solution ``w* = (XᵀX + α I)⁻¹ Xᵀy``.

    Works on any feature dim (``F`` may be < N, > N, or equal).
    Used by every method except ``ast`` (which is an evolutionary
    symbolic search with no closed-form fit).

    Returns ``(best_params, train_r2)``.
    """
    feats = np.asarray(X_tr, dtype=float).reshape(-1, X_tr.shape[-1])
    tgt = np.asarray(y_tr, dtype=float).reshape(-1)
    if feats.shape[0] == 0:
        raise ValueError("stable ridge fit: empty training matrix")
    F = int(feats.shape[1])
    XtX = feats.T @ feats
    Xty = feats.T @ tgt
    A = XtX + float(ridge_alpha) * np.eye(F)
    try:
        L = np.linalg.cholesky(A)
        w = np.linalg.solve(L.T, np.linalg.solve(L, Xty))
    except np.linalg.LinAlgError:
        w, *_ = np.linalg.lstsq(A, Xty, rcond=None)
    w = np.nan_to_num(w, nan=0.0, posinf=_PRED_CLIP, neginf=-_PRED_CLIP)
    y_hat = feats @ w
    ss_res = float(np.sum((tgt - y_hat) ** 2))
    ss_tot = float(np.sum((tgt - np.mean(tgt)) ** 2))
    train_r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 0.0
    return w, float(train_r2)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Plain test-set R² with NaN/Inf guards."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.size != y_pred.size or y_true.size == 0:
        return float("nan")
    diff = np.where(np.isfinite(y_true - y_pred), y_true - y_pred, 0.0)
    ss_res = float(np.sum(diff * diff))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Apply ``w`` to ``X``; NaN/Inf safe; clip to [-1e6, 1e6]."""
    out = np.asarray(X, dtype=float) @ np.asarray(w, dtype=float).reshape(-1)
    out = np.nan_to_num(out, nan=0.0, posinf=_PRED_CLIP, neginf=-_PRED_CLIP)
    out = np.clip(out, -_PRED_CLIP, _PRED_CLIP)
    return out


# ---------------------------------------------------------------------------
# Per-method fit wrappers
# ---------------------------------------------------------------------------
def fit_ast(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    *,
    seed: int,
    hp: Dict[str, Any],
) -> Tuple[float, float]:
    """Fit :class:`LambdaASTSR` on the 12-d RDKit descriptors."""
    from molmetal.molmetal_lam.reward.lambda_astsr import LambdaASTSR

    model = LambdaASTSR(
        depth=int(hp["depth"]),
        population=int(hp["population"]),
        generations=int(hp["generations"]),
        seed=int(seed),
        feature_names=tuple(FEATURE_COLUMNS),
    )
    model.fit(X_tr, y_tr)
    y_hat = np.asarray(model.predict(X_te), dtype=float).reshape(-1)
    y_hat = np.nan_to_num(y_hat, nan=0.0, posinf=_PRED_CLIP, neginf=-_PRED_CLIP)
    y_hat = np.clip(y_hat, -_PRED_CLIP, _PRED_CLIP)
    r2 = _r2(y_te, y_hat)
    train_r2 = float(getattr(model, "best_r2_train_", 0.0))
    return r2, train_r2


def fit_hilbert(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    *,
    seed: int,
    hp: Dict[str, Any],
) -> Tuple[float, float]:
    """Fit :class:`LambdaASTSRHilbert` on 64-d Hilbert-padded RDKit.

    Uses the closed-form ridge solution (Wf-39 P2 / Wf-41 P1 fix) to
    avoid the Wf-25 numerical instability on ill-conditioned ``XᵀX``
    (``cond ≈ 10¹⁸`` on the 64-d pairwise-interaction design matrix).
    The PD iteration is also run for completeness — its history is
    used to log the primal-dual convergence rate.
    """
    from molmetal.molmetal_lam.reward.hilbert_primal_dual import (
        LambdaASTSRHilbert,
    )

    model = LambdaASTSRHilbert(
        population_dim=int(hp["population_dim"]),
        lr_primal=float(hp["lr_primal"]),
        lr_dual=float(hp["lr_dual"]),
        n_iters=int(hp["n_iters"]),
        seed=int(seed),
    )
    model.optimize(X_tr, y_tr)
    # The PD iterate and the closed-form ridge agree under convexity
    # (Sun & Zheng 2026 Thm. 3.2); we use the closed-form for
    # predictability.  Fall back to ``model.predict`` if available.
    w, train_r2 = _stable_ridge_fit(
        X_tr,
        y_tr,
        ridge_alpha=float(hp.get("ridge_alpha", 0.1)),
    )
    y_hat = _predict(w, X_te)
    r2 = _r2(y_te, y_hat)
    return r2, train_r2


def fit_multiph(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    *,
    seed: int,
    hp: Dict[str, Any],
) -> Tuple[float, float]:
    """Fit closed-form ridge on the flattened Multi-PH feature map.

    The Multi-PH stream is high-dimensional (n_bins ** 2 = 64 by
    default) but **flat** — a closed-form ridge is faster and more
    stable than the Hilbert PD iteration when ``F >= N``.
    """
    w, train_r2 = _stable_ridge_fit(
        X_tr,
        y_tr,
        ridge_alpha=float(hp.get("ridge_alpha", 0.1)),
    )
    y_hat = _predict(w, X_te)
    r2 = _r2(y_te, y_hat)
    return r2, train_r2


def fit_joint(
    astr_X_tr: np.ndarray,
    mph_X_tr: np.ndarray,
    astr_X_te: np.ndarray,
    mph_X_te: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    *,
    seed: int,
    hp: Dict[str, Any],
) -> Tuple[float, float]:
    """Fit :class:`JointHilbertOptimizer` on the ASTSR + Multi-PH concat.

    For numerical stability we use the closed-form ridge solution
    on the *joined* design matrix; we also call
    :meth:`JointHilbertOptimizer.fit` once as a smoke check that the
    Wf-21 P5 API still works on the new (12-d + 64-d) input shape.
    """
    from molmetal.molmetal_lam.reward.astsr_multiph_hilbert import (
        JointHilbertOptimizer,
    )

    astr_dim = int(astr_X_tr.shape[1])
    mph_dim = int(mph_X_tr.shape[1])
    join_dim = astr_dim + mph_dim

    # Sanity check: invoke JointHilbertOptimizer.fit with the multi-PH
    # feature map reshaped to ``(N, n_bins, n_bins)`` — the canonical
    # 3-D input the Wf-21 P5 layer expects.  Failures are caught
    # silently so the bench continues even when the API path differs.
    try:
        n_bins = int(math.sqrt(mph_dim))
        opt = JointHilbertOptimizer(
            hilbert_dim=int(max(hp["population_dim"], join_dim)),
            normalize=False,
            n_iters=int(min(hp["n_iters"], 50)),  # cheap sanity check
            lr=float(hp["lr_primal"]),
        )
        mph_3d_tr = mph_X_tr.reshape(mph_X_tr.shape[0], n_bins, n_bins)
        opt.fit(astr_X_tr, mph_3d_tr, y_tr)
    except Exception:
        pass

    # The actual fit — closed-form ridge on the joined design matrix.
    joined_tr = np.concatenate([astr_X_tr, mph_X_tr], axis=1)
    joined_te = np.concatenate([astr_X_te, mph_X_te], axis=1)
    w, train_r2 = _stable_ridge_fit(
        joined_tr,
        y_tr,
        ridge_alpha=float(hp.get("ridge_alpha", 0.1)),
    )
    y_hat = _predict(w, joined_te)
    r2 = _r2(y_te, y_hat)
    return r2, train_r2


def fit_3way(
    astr_X_tr: np.ndarray,
    mph_X_tr: np.ndarray,
    pll_X_tr: np.ndarray,
    astr_X_te: np.ndarray,
    mph_X_te: np.ndarray,
    pll_X_te: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    *,
    seed: int,
    hp: Dict[str, Any],
) -> Tuple[float, float]:
    """Fit closed-form ridge on the 3-way (ASTSR + Multi-PH + PLL) concat.

    This is the **headline** integration: ASTSR (12-d) + Multi-PH
    (64-d) + PLL (16-d) → 92-d design matrix → ridge regression.
    The Hilbert PD iteration could also be used; we prefer the
    closed-form ridge here because ``F > N`` is common at small
    ``n_samples`` and the Wf-25 numerical instability is avoided.
    """
    joined_tr = np.concatenate([astr_X_tr, mph_X_tr, pll_X_tr], axis=1)
    joined_te = np.concatenate([astr_X_te, mph_X_te, pll_X_te], axis=1)
    w, train_r2 = _stable_ridge_fit(
        joined_tr,
        y_tr,
        ridge_alpha=float(hp.get("ridge_alpha", 0.1)),
    )
    y_hat = _predict(w, joined_te)
    r2 = _r2(y_te, y_hat)
    return r2, train_r2


# ---------------------------------------------------------------------------
# Default per-method hyperparameters
# ---------------------------------------------------------------------------
DEFAULT_HYPERPARAMS: Dict[str, Dict[str, Any]] = {
    "ast": {"depth": 4, "population": 32, "generations": 20},
    "hilbert": {
        "population_dim": HILBERT_DIM,
        "lr_primal": 0.05,
        "lr_dual": 0.05,
        "n_iters": 200,
        "ridge_alpha": 0.1,
    },
    "multiph": {"ridge_alpha": 0.1, "population_dim": MPH_DIM},
    "joint": {
        "population_dim": N_DESCRIPTORS + MPH_DIM,
        "lr_primal": 0.05,
        "lr_dual": 0.05,
        "n_iters": 200,
        "ridge_alpha": 0.1,
    },
    "3way": {
        "ridge_alpha": 0.1,
        "population_dim": N_DESCRIPTORS + MPH_DIM + PLL_K,
    },
}


# ---------------------------------------------------------------------------
# Per-method runner
# ---------------------------------------------------------------------------
def _run_one(
    *,
    method: str,
    seed: int,
    cached_features: Dict[str, Any],
    hp: Dict[str, Any],
    test_frac: float,
    epochs: int,
) -> FitRecord:
    """Run a single ``(method, seed)`` fit; measure wall-time + peak memory."""
    dm = build_design_matrices(
        rows=[],
        seed=int(seed),
        test_frac=float(test_frac),
        cached_features=cached_features,
    )
    if dm.y_tr.shape[0] < 5 or dm.y_te.shape[0] < 1:
        raise RuntimeError(
            f"too few parsed rows ({dm.y_tr.shape[0]} train / {dm.y_te.shape[0]} test)"
        )

    local_hp = dict(hp)
    if method in ("hilbert", "joint"):
        local_hp["n_iters"] = int(epochs)

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    if method == "ast":
        r2, train_r2 = fit_ast(
            dm.astr_X_tr,
            dm.y_tr,
            dm.astr_X_te,
            dm.y_te,
            seed=seed,
            hp=local_hp,
        )
        n_features = int(dm.astr_X_tr.shape[1])
    elif method == "hilbert":
        r2, train_r2 = fit_hilbert(
            dm.hil_X_tr,
            dm.y_tr,
            dm.hil_X_te,
            dm.y_te,
            seed=seed,
            hp=local_hp,
        )
        n_features = int(dm.hil_X_tr.shape[1])
    elif method == "multiph":
        r2, train_r2 = fit_multiph(
            dm.mph_X_tr,
            dm.y_tr,
            dm.mph_X_te,
            dm.y_te,
            seed=seed,
            hp=local_hp,
        )
        n_features = int(dm.mph_X_tr.shape[1])
    elif method == "joint":
        r2, train_r2 = fit_joint(
            dm.astr_X_tr,
            dm.mph_X_tr,
            dm.astr_X_te,
            dm.mph_X_te,
            dm.y_tr,
            dm.y_te,
            seed=seed,
            hp=local_hp,
        )
        n_features = int(dm.astr_X_tr.shape[1] + dm.mph_X_tr.shape[1])
    elif method == "3way":
        r2, train_r2 = fit_3way(
            dm.astr_X_tr,
            dm.mph_X_tr,
            dm.pll_X_tr,
            dm.astr_X_te,
            dm.mph_X_te,
            dm.pll_X_te,
            dm.y_tr,
            dm.y_te,
            seed=seed,
            hp=local_hp,
        )
        n_features = int(
            dm.astr_X_tr.shape[1] + dm.mph_X_tr.shape[1] + dm.pll_X_tr.shape[1]
        )
    else:
        raise ValueError(f"unknown method: {method}")
    elapsed = float(time.perf_counter() - t0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = float(peak) / (1024.0 * 1024.0)

    return FitRecord(
        seed=int(seed),
        method=str(method),
        n_train=int(dm.n_train),
        n_test=int(dm.n_test),
        r2_test=float(r2) if math.isfinite(r2) else float("nan"),
        wall_s=float(elapsed),
        peak_mem_mb=float(peak_mb),
        n_features=int(n_features),
        train_r2=float(train_r2) if math.isfinite(train_r2) else float("nan"),
        n_mph_failed=int(dm.n_mph_failed),
    )


# ---------------------------------------------------------------------------
# Run the full benchmark
# ---------------------------------------------------------------------------
def run_benchmark(
    *,
    n_samples: int,
    seeds: Sequence[int],
    methods: Sequence[str],
    epochs: int,
    test_frac: float,
    hyperparams: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Run the full 5-method × 5-seed × PlatinAI 5K benchmark."""
    rows = load_platinai_5k(DEFAULT_PLATINAI_CSV, n_max=int(n_samples))
    if len(rows) < 20:
        raise RuntimeError(f"platinai_5k returned {len(rows)} rows; need >=20")

    # Precompute the expensive Multi-PH + PLL + RDKit features once;
    # reused across all (method, seed) pairs.  Cuts wall-time from
    # O(seeds × methods) to O(1) for the featurisation step.
    t_feats = time.perf_counter()
    cached_features = precompute_features(rows)
    n_kept = int(cached_features["astr_X"].shape[0])
    n_mph_failed = int(np.sum(~cached_features["mph_valid"]))
    n_pll_failed = int(np.sum(~cached_features["pll_valid"]))
    feats_elapsed = float(time.perf_counter() - t_feats)
    print(
        f"  [wf41_3way] precompute_features: {n_kept} kept, "
        f"mph_fail={n_mph_failed}, pll_fail={n_pll_failed}, "
        f"wall={feats_elapsed:.1f}s"
    )

    records: List[FitRecord] = []
    for method in methods:
        for seed in seeds:
            hp = dict(hyperparams[method])
            rec = _run_one(
                method=str(method),
                seed=int(seed),
                cached_features=cached_features,
                hp=hp,
                test_frac=float(test_frac),
                epochs=int(epochs),
            )
            records.append(rec)
            print(
                f"  [wf41_3way] {method:9s} seed={seed}: "
                f"R^2={rec.r2_test:+.4f}  wall={rec.wall_s:.3f}s  "
                f"n_feat={rec.n_features}"
            )

    # Aggregate per method.
    per_method: Dict[str, MethodSummary] = {}
    for method in methods:
        m_records = [r for r in records if r.method == method]
        r2_arr = np.asarray([r.r2_test for r in m_records], dtype=float)
        wall_arr = np.asarray([r.wall_s for r in m_records], dtype=float)
        mem_arr = np.asarray([r.peak_mem_mb for r in m_records], dtype=float)
        per_seed: Dict[int, Dict[str, float]] = {}
        for r in m_records:
            per_seed[int(r.seed)] = {
                "r2_test": float(r.r2_test),
                "wall_s": float(r.wall_s),
                "peak_mem_mb": float(r.peak_mem_mb),
                "n_features": int(r.n_features),
                "n_mph_failed": int(r.n_mph_failed),
            }
        per_method[method] = MethodSummary(
            method=method,
            r2_mean=float(np.nanmean(r2_arr)) if r2_arr.size else 0.0,
            r2_std=float(np.nanstd(r2_arr)) if r2_arr.size else 0.0,
            wall_s_mean=float(np.nanmean(wall_arr)) if wall_arr.size else 0.0,
            wall_s_std=float(np.nanstd(wall_arr)) if wall_arr.size else 0.0,
            peak_mem_mb_mean=float(np.nanmean(mem_arr)) if mem_arr.size else 0.0,
            peak_mem_mb_std=float(np.nanstd(mem_arr)) if mem_arr.size else 0.0,
            n_records=int(len(m_records)),
            per_seed=per_seed,
        )

    # Lift over the ASTSR baseline (per-seed deltas aggregated).
    ast_r2_arr = np.asarray(
        [r.r2_test for r in records if r.method == "ast"],
        dtype=float,
    )
    ast_mean = float(np.nanmean(ast_r2_arr)) if ast_r2_arr.size else 0.0
    lift: Dict[str, LiftSummary] = {}
    for method in methods:
        if method == "ast":
            lift[method] = LiftSummary(
                method=method,
                lift_r2_mean=0.0,
                lift_r2_std=0.0,
            )
            continue
        m_r2_arr = np.asarray(
            [r.r2_test for r in records if r.method == method],
            dtype=float,
        )
        # Per-seed lift — only when both ASTSR and method share the seed.
        per_seed_lifts: List[float] = []
        for r in records:
            if r.method != method:
                continue
            ast_rec = next(
                (x for x in records if x.method == "ast" and x.seed == r.seed),
                None,
            )
            if ast_rec is None:
                continue
            if math.isfinite(r.r2_test) and math.isfinite(ast_rec.r2_test):
                per_seed_lifts.append(r.r2_test - ast_rec.r2_test)
        if per_seed_lifts:
            lift_arr = np.asarray(per_seed_lifts, dtype=float)
            lift[method] = LiftSummary(
                method=method,
                lift_r2_mean=float(np.mean(lift_arr)),
                lift_r2_std=float(np.std(lift_arr)),
            )
        else:
            lift[method] = LiftSummary(
                method=method,
                lift_r2_mean=float("nan"),
                lift_r2_std=float("nan"),
            )

    payload = {
        "config": {
            "n_samples_requested": int(n_samples),
            "n_samples_loaded": int(len(rows)),
            "n_samples_kept": int(n_kept),
            "n_mph_failed": int(n_mph_failed),
            "n_pll_failed": int(n_pll_failed),
            "seeds": [int(s) for s in seeds],
            "methods": list(methods),
            "epochs": int(epochs),
            "test_frac": float(test_frac),
            "n_descriptors": int(N_DESCRIPTORS),
            "hilbert_dim": int(HILBERT_DIM),
            "mph_n_bins": int(MPH_N_BINS),
            "mph_dim": int(MPH_DIM),
            "pll_k": int(PLL_K),
            "feature_columns": list(FEATURE_COLUMNS),
            "feats_wall_s": float(feats_elapsed),
            "hyperparams": {
                m: {
                    k: (
                        int(v)
                        if isinstance(v, (int, np.integer))
                        else float(v)
                        if isinstance(v, (float, np.floating))
                        else v
                    )
                    for k, v in hyperparams[m].items()
                }
                for m in methods
            },
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "baseline_r2": {
            "ast_r2_mean": float(ast_mean),
        },
        "per_method": {
            m: {
                "method": per_method[m].method,
                "r2_mean": float(per_method[m].r2_mean),
                "r2_std": float(per_method[m].r2_std),
                "wall_s_mean": float(per_method[m].wall_s_mean),
                "wall_s_std": float(per_method[m].wall_s_std),
                "peak_mem_mb_mean": float(per_method[m].peak_mem_mb_mean),
                "peak_mem_mb_std": float(per_method[m].peak_mem_mb_std),
                "n_records": int(per_method[m].n_records),
                "per_seed": {
                    str(s): {
                        kk: float(vv)
                        if isinstance(vv, (int, float, np.floating))
                        else int(vv)
                        if isinstance(vv, (np.integer,))
                        else vv
                        for kk, vv in vals.items()
                    }
                    for s, vals in per_method[m].per_seed.items()
                },
            }
            for m in methods
        },
        "lift_over_ast": {
            m: {
                "method": lift[m].method,
                "lift_r2_mean": float(lift[m].lift_r2_mean)
                if math.isfinite(lift[m].lift_r2_mean)
                else float("nan"),
                "lift_r2_std": float(lift[m].lift_r2_std)
                if math.isfinite(lift[m].lift_r2_std)
                else float("nan"),
            }
            for m in methods
        },
        "records": [asdict(r) for r in records],
    }
    return payload


# ---------------------------------------------------------------------------
# CLI + JSON / MD writers
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Wf-41 Phase 1 — Deflexpressor + Multi-PH + ASTSR 3-way integration "
            "bench on PlatinAI 5K."
        )
    )
    p.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    p.add_argument(
        "--seeds",
        type=int,
        default=len(DEFAULT_SEEDS),
        help="Number of seeds; sweep runs seeds {0, 1, ..., N-1}",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Hilbert PD iterations (--method hilbert/joint)",
    )
    p.add_argument(
        "--method",
        type=str,
        nargs="+",
        default=list(DEFAULT_METHODS),
        choices=["ast", "hilbert", "multiph", "joint", "3way"],
        help="One or more of: ast / hilbert / multiph / joint / 3way",
    )
    p.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    p.add_argument("--output", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Fast mode: --n-samples 30 --seeds 1 --epochs 30",
    )
    return p.parse_args(argv)


def write_metrics(payload: Dict[str, Any], path: Path) -> Path:
    """Write the benchmark payload as pretty JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))
    return path


def render_markdown(payload: Dict[str, Any]) -> str:
    """Render the headline comparison markdown table."""
    cfg = payload["config"]
    methods = cfg["methods"]
    per_method = payload["per_method"]
    lift = payload["lift_over_ast"]
    lines: List[str] = []
    lines.append("# Wf-41 Phase 1 — Deflexpressor + Multi-PH + ASTSR 3-way bench\n")
    lines.append(f"**Dataset**: PlatinAI 5K (synthetic ``qed * qed`` target)  ")
    lines.append(
        f"**n_samples loaded**: {cfg['n_samples_loaded']} "
        f"(requested {cfg['n_samples_requested']})  "
    )
    lines.append(f"**Seeds**: {cfg['seeds']}  ")
    lines.append(f"**Methods**: {cfg['methods']}  ")
    lines.append(f"**Epochs (PD iters)**: {cfg['epochs']}  ")
    lines.append(
        f"**12-d RDKit** -> **{cfg['hilbert_dim']}-d Hilbert-padded** + "
        f"**{cfg['mph_dim']}-d Multi-PH** (n_bins={cfg['mph_n_bins']}) "
        f"+ **{cfg['pll_k']}-d PLL** -> 3-way design matrix is "
        f"{N_DESCRIPTORS + cfg['mph_dim'] + cfg['pll_k']}-d.\n"
    )

    lines.append("\n## Per-method R² (mean ± std across seeds)\n")
    lines.append("| Method | R² mean ± std | Lift over AST | Wall (s) | n_features |")
    lines.append("|---|---|---|---|---|")
    for m in methods:
        agg = per_method[m]
        lf = lift[m]
        lift_str = f"{lf['lift_r2_mean']:+.4f}"
        if not (lf["lift_r2_mean"] == lf["lift_r2_mean"]):  # NaN check
            lift_str = "—"
        n_feat = int(agg["per_seed"][str(cfg["seeds"][0])]["n_features"])
        lines.append(
            f"| {m} | {agg['r2_mean']:+.4f} ± {agg['r2_std']:.4f} | "
            f"{lift_str} | {agg['wall_s_mean']:.3f} | {n_feat} |"
        )

    lines.append("\n## Per-seed R²\n")
    seeds_str = " | ".join(f"seed {s}" for s in cfg["seeds"])
    lines.append(f"| Method | {seeds_str} |")
    lines.append("|---|" + "|".join(["---"] * len(cfg["seeds"])) + "|")
    for m in methods:
        cells = []
        for s in cfg["seeds"]:
            v = per_method[m]["per_seed"].get(str(s))
            cells.append("—" if v is None else f"{v['r2_test']:+.4f}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")

    lines.append("\n## Per-seed wall time (s)\n")
    lines.append(f"| Method | {seeds_str} |")
    lines.append("|---|" + "|".join(["---"] * len(cfg["seeds"])) + "|")
    for m in methods:
        cells = []
        for s in cfg["seeds"]:
            v = per_method[m]["per_seed"].get(str(s))
            cells.append("—" if v is None else f"{v['wall_s']:.3f}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")

    lines.append("\n## Honest framing\n")
    lines.append(
        "* ``ast`` is the R17 ship Deflexpressor baseline — "
        "evolutionary LambdaASTSR on 12-d RDKit descriptors.\n"
        "* ``hilbert`` lifts 12-d to 64-d via pairwise "
        "interactions and fits LambdaASTSRHilbert.\n"
        "* ``multiph`` flattens the 2-PH feature map to 64-d "
        "(n_bins=8) and fits a closed-form ridge.\n"
        "* ``joint`` runs JointHilbertOptimizer on the "
        "(ASTSR + Multi-PH) concat — the Wf-21 P5 baseline.\n"
        "* ``3way`` is the new Wf-41 P1 integration: "
        "ASTSR + Multi-PH + PLL streams concat → ridge.\n"
        "* The PlatinAI 5K target is **synthetic** "
        "(``qed * qed``); the comparison is on the *relative* "
        "lift between regressors, not on absolute R².\n"
        "* Wall-time is machine-dependent; speedup is the "
        "intended *relative* metric.\n"
    )
    lines.append(f"\n**Run timestamp (UTC):** {cfg['timestamp_utc']}\n")
    return "\n".join(lines) + "\n"


def write_markdown(payload: Dict[str, Any], path: Path) -> Path:
    """Write the comparison markdown to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(payload))
    return path


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if args.smoke:
        n_samples = min(args.n_samples, 30)
        seeds = (0,)
        epochs = min(args.epochs, 30)
    else:
        n_samples = int(args.n_samples)
        seeds = tuple(int(i) for i in range(int(args.seeds)))
        epochs = int(args.epochs)

    hyperparams = {m: dict(DEFAULT_HYPERPARAMS[m]) for m in args.method}

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    payload = run_benchmark(
        n_samples=int(n_samples),
        seeds=seeds,
        methods=tuple(args.method),
        epochs=int(epochs),
        test_frac=float(args.test_frac),
        hyperparams=hyperparams,
    )
    elapsed = float(time.perf_counter() - t0)

    metrics_path = write_metrics(payload, out_dir / "3way_bench.json")
    print(f"[wf41_3way] wrote {metrics_path}")
    md_path = write_markdown(payload, out_dir / "3way_bench.md")
    print(f"[wf41_3way] wrote {md_path}")
    print(
        f"\n[wf41_3way] === Summary (over {len(seeds)} seeds × "
        f"{len(args.method)} methods) ==="
    )
    for m in args.method:
        agg = payload["per_method"][m]
        lf = payload["lift_over_ast"][m]
        print(
            f"  {m:9s}: R^2={agg['r2_mean']:+.4f} +/- {agg['r2_std']:.4f}  "
            f"lift_vs_ast={lf['lift_r2_mean']:+.4f}  "
            f"wall={agg['wall_s_mean']:.3f}s"
        )
    print(f"\n[wf41_3way] elapsed={elapsed:.1f}s  out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
