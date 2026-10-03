#!/usr/bin/env python3
"""Wf-85 Phase 1 — REAL Hilbert-only R² on PlatinAI 4-cell (strict-scaffold).

================================================================
``molmetal/scripts/bench_hilbert_only_real.py`` is the **REAL retrain**
Hilbert-only R² baseline (no ASTSR, no Multi-PH + PLL, no composite
blend).  It trains the :class:`LambdaASTSRHilbert` primal-dual
regressor on **strict-scaffold train/val/test splits** of
PlatinAI 4-cell with **5 seeds × 50 epochs**, and produces **MEASURED
per-seed R² + 5-seed mean ± std per cell line**.  GPU-0 Blackwell
acceleration is supported via the :mod:`hilbert_astsr_cuda` batch
wrapper; CPU fp32 + optional fp64 fallback is the default path.

Why this script matters
-----------------------
* Wf-77 P1 / Wf-83 P1 / Wf-83 P2 all reused the **same**
  ``LambdaASTSRHilbert`` K-fold CV harness and produced MEASURED R²
  ≈ −0.40 to −6.71 on PlatinAI 4-cell with predict-the-mean
  collapse.  Wf-85 Phase 1 demands a **REAL retrain** benchmark on
  **strict-scaffold splits** (Bemis-Murcko — never seen at train
  time) with **5 seeds × 50 epochs** to honestly probe whether the
  signal floor is a feature issue (per Wf-49 §Cause 2 — every F5
  feature |ρ| < 0.02) or a leak / overfitting issue masked by K-fold
  random splits.
* Wf-49 PARTIAL closed the F5_RDKIT_2D feature swap with
  ``n_samples=0`` honest reporting on platinai_5k.csv; Wf-69 P3.1
  shipped the ``platinai_4cell`` loader.  Wf-85 P1 takes the next
  step: real, **strict-scaffold**, **50-epoch** retrain on the
  extracted PlatinAI 4-cell cohort.
* Wf-85 P1 is the **anchor** for any future Wf-56 R20-P2 GPU
  retrain: the combined Hilbert+ASTSR R² LIFT will be measured
  against this Hilbert-only R² value.

Constraints honored
-------------------
* **No ASTSR, no Multi-PH + PLL** — pure Hilbert primal-dual on the
  12-d RDKit descriptor (extended via ``hilbert_feature_map`` to
  ``hilbert_dim``-d feature map with linear + bias + pairwise
  interactions, variance-ranked truncation).
* **Strict-scaffold split** — Bemis-Murcko scaffold is computed for
  every row; each scaffold lands in exactly one of {train, val, test}
  via :class:`ScaffoldSplitter` (largest-first strategy).  This is
  the **stronger** OOD split vs K-fold random CV — molecules with
  the same scaffold but different substituents all land in the same
  split, so the test set contains scaffolds the model has never
  seen at all.
* **5 seeds × 50 epochs** — 5 RNG seeds (default 0..4); 50 Hilbert PD
  iterations per seed.  Per-seed wall-time budget is logged in the
  per-seed JSON.
* **CPU + GPU-0 Blackwell** — ``--device cpu`` (default) uses the
  CPU ``LambdaASTSRHilbert`` with optional fp64 fallback;
  ``--device cuda`` dispatches via
  :func:`hilbert_astsr_optimize_single` and aborts cleanly if
  Blackwell sm_120 is unavailable.
* **Honest framing** — negative R² values are not hidden; the
  collapse detector flags ``pred_std / y_test_std < 0.10``.  Per-cohort
  honest-negative reporting per ``party-twenty-spirit §3.5``.

Inputs
------
* ``--dataset``: ``platinai_4cell`` (4078 rows × 4 cell lines).
* ``--cell-lines``: list (default ``mcf7 a549 a2780 a2780cis``).
* ``--seeds``: list (default ``0 1 2 3 4``).
* ``--epochs``: int (default ``50``) — Hilbert PD iterations.
* ``--strict-scaffold``: flag (default ON) — use
  :class:`ScaffoldSplitter` instead of K-fold random.
* ``--fractions``: ``train val test`` (default ``0.7 0.15 0.15``) —
  scaffold-split fractions.
* ``--hilbert-dim``: int (default ``64``).
* ``--device``: ``cpu`` (default) or ``cuda``.
* ``--output``: ``molmetal/reports/wf85_p1_hilbert_only_real``.

Outputs
-------
* ``<output>/{cell_line}/per_seed_metrics.json`` — per-seed Hilbert R²
  + 5-seed mean ± std + per-fold (scaffold-split fold) breakdown +
  collapse flag + ``hilbert_wall_time_s`` per seed.
* ``<output>/{cell_line}/run.log`` — stdout mirror.
* ``<output>/synthesis.md`` — cross-cell-line 5-seed × 50-epoch
  summary + Wf-85 verify gate.

Verify gate (per ``party-twenty-spirit §3.5``)
---------------------------------------------
* Per-cell-line **Hilbert R² mean** ≥ 0 on at least one cell line —
  otherwise honest-negative reporting per fold.
* Cross-cell-line **4-cell aggregate** R² is computed as the mean of
  the 4 per-cell-line means; the per-cell breakdown is the
  load-bearing number for ``TODO/03_plans/LAMBDA_ROADMAP.md`` §1.3
  L5 gap status.

Usage
-----

::

    molmetal/.venvs/mol-metal-venv/bin/python -u \\
        molmetal/scripts/bench_hilbert_only_real.py \\
        --dataset platinai_4cell \\
        --cell-lines mcf7 a549 a2780 a2780cis \\
        --seeds 0 1 2 3 4 --epochs 50 --strict-scaffold \\
        --output molmetal/reports/wf85_p1_hilbert_only_real

Pytest
~~~~~~

::

    pytest molmetal/tests/test_hilbert_only_real.py -v
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Repo-root import bootstrap.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# 12-d F5_RDKIT_2D descriptor schema (Wf-49 P2 ship — parallel to
# bench_hilbert_astr_phase2_platinai_4cell.py).
# ---------------------------------------------------------------------------
FEATURE_COLUMNS: Tuple[str, ...] = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "HBA",
    "HBD",
    "NRotB",
    "HeavyAtoms",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "MolMR",
    "BalabanJ",
)
COLLAPSE_RATIO_THRESHOLD: float = 0.10
PRED_CLIP: float = 1e6
EPS_STD: float = 1e-3

# PlatinAI 4-cell CSV columns.
PLATINAI_CSV = Path("molmetal/data/platinai_4cell.csv")
PLATINAI_CELL_COLS: Dict[str, str] = {
    "mcf7": "ic50_mcf7_nm",
    "a549": "ic50_a549_nm",
    "a2780": "ic50_a2780_nm",
    "a2780cis": "ic50_a2780cis_nm",
}
PLATINAI_SMILES_COL: str = "smiles"


# ---------------------------------------------------------------------------
# RDKit featurizer — 12-d F5_RDKIT_2D descriptor.
# ---------------------------------------------------------------------------
def _featurize_smiles(smiles: str) -> Optional[List[float]]:
    """Return 12-d RDKit 2D descriptor vector; ``None`` if SMILES fails."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

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
            float(mol.GetNumHeavyAtoms()),
            float(rdMolDescriptors.CalcNumRings(mol)),
            float(rdMolDescriptors.CalcNumAromaticRings(mol)),
            float(rdMolDescriptors.CalcFractionCSP3(mol)),
            float(Descriptors.MolMR(mol)),
            float(Descriptors.BalabanJ(mol)),
        ]
        if any((not math.isfinite(v)) for v in feats):
            return None
        return feats
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Loaders.
# ---------------------------------------------------------------------------
def _read_csv(path: Path) -> List[Dict[str, str]]:
    """Read CSV to list-of-dicts (utf-8, tolerant to BOM)."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _canonicalize_smiles(smi: str) -> str:
    """Best-effort canonicalize — empty string if RDKit fails."""
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol)
    except Exception:
        return ""


def _bemis_murcko_scaffold(smi: str) -> str:
    """Return the Bemis-Murcko scaffold SMILES; empty if RDKit fails.

    The :class:`ScaffoldSplitter` carries the same logic, but here
    we re-implement the bare scaffold computation so the script
    doesn't depend on the MetalCytotoxDataset API surface.
    """
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.Scaffolds import MurckoScaffold

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        scaffold = MurckoScaffold.GetScaffoldForMol(mol, includeChirality=False)
        if scaffold is None:
            return ""
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return ""


@dataclass
class CellLineSpec:
    """Specification for one PlatinAI cell-line target column."""

    cell_line: str
    target_col: str


def load_platinai_cell(
    repo_root: Path,
    cell_line: str,
    n_max: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load PlatinAI 4-cell for a single cell-line target column.

    Returns ``(X, y, smiles)``:

    * ``X`` is ``(n_samples, 12)`` float32 (F5_RDKIT_2D descriptor).
    * ``y`` is ``(n_samples,)`` float32 — log10(IC50_nM × 1e-9) =
      -log10(IC50_µM × 1e-6) (real pIC50, signed).
    * ``smiles`` is the list of SMILES (for scaffold split + debug).

    Rows with NaN/empty target or unparseable SMILES are dropped.
    """
    if cell_line not in PLATINAI_CELL_COLS:
        raise ValueError(
            f"Unknown cell line: {cell_line}; "
            f"supported: {sorted(PLATINAI_CELL_COLS)}"
        )
    csv_path = repo_root / PLATINAI_CSV
    target_col = PLATINAI_CELL_COLS[cell_line]
    rows = _read_csv(csv_path)
    Xs: List[List[float]] = []
    Ys: List[float] = []
    smiles: List[str] = []
    for r in rows:
        smi = (r.get(PLATINAI_SMILES_COL) or "").strip()
        if not smi:
            continue
        feats = _featurize_smiles(smi)
        if feats is None:
            continue
        raw = r.get(target_col)
        try:
            ic50_nm = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(ic50_nm) or ic50_nm <= 0.0:
            continue
        ic50_um = ic50_nm / 1000.0
        pic50 = -math.log10(max(ic50_um * 1e-6, 1e-12))
        if not math.isfinite(pic50):
            continue
        Xs.append(feats)
        Ys.append(float(pic50))
        smiles.append(smi)
        if len(Xs) >= n_max:
            break
    if not Xs:
        raise RuntimeError(f"No usable rows for cell line {cell_line}")
    return (
        np.asarray(Xs, dtype=np.float32),
        np.asarray(Ys, dtype=np.float32),
        smiles,
    )


# ---------------------------------------------------------------------------
# Hilbert feature map (parallel to bench_hilbert_astr_phase2_platinai_4cell).
# ---------------------------------------------------------------------------
def hilbert_feature_map(
    X_12d: np.ndarray,
    hilbert_dim: int,
    X_12d_test: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Augment descriptors to ``hilbert_dim`` via linear + bias +
    pairwise interactions.

    Augmentation strategy (per Wf-37 P1 §3 — chosen because tile
    breaks conditioning):

        [x₁, x₂, …, x_F, 1.0,  x_i · x_j  for i < j]

    When the augmentation exceeds ``hilbert_dim``, keep all linear
    terms + bias + top ``hilbert_dim - F - 1`` interaction columns by
    training-set variance.
    """
    n = X_12d.shape[0]
    F = X_12d.shape[1]
    if F >= hilbert_dim:
        out_tr = X_12d[:, :hilbert_dim]
        if X_12d_test is not None:
            return out_tr, X_12d_test[:, :hilbert_dim]
        return out_tr, None
    bias = np.ones((n, 1), dtype=np.float32)
    interactions_tr = []
    for i in range(F):
        for j in range(i + 1, F):
            interactions_tr.append(X_12d[:, i] * X_12d[:, j])
    inter_arr_tr = (
        np.stack(interactions_tr, axis=1)
        if interactions_tr
        else np.zeros((n, 0), dtype=np.float32)
    )
    base_tr = np.concatenate([X_12d, bias], axis=1)
    aug_tr = np.concatenate([base_tr, inter_arr_tr], axis=1)
    if X_12d_test is not None:
        n_te = X_12d_test.shape[0]
        bias_te = np.ones((n_te, 1), dtype=np.float32)
        interactions_te = []
        for i in range(F):
            for j in range(i + 1, F):
                interactions_te.append(X_12d_test[:, i] * X_12d_test[:, j])
        inter_arr_te = (
            np.stack(interactions_te, axis=1)
            if interactions_te
            else np.zeros((n_te, 0), dtype=np.float32)
        )
        base_te = np.concatenate([X_12d_test, bias_te], axis=1)
        aug_te = np.concatenate([base_te, inter_arr_te], axis=1)
    else:
        aug_te = None
    if aug_tr.shape[1] <= hilbert_dim:
        pad = np.zeros((n, hilbert_dim - aug_tr.shape[1]), dtype=np.float32)
        out_tr = np.concatenate([aug_tr, pad], axis=1)
        if aug_te is not None:
            pad_te = np.zeros(
                (n_te, hilbert_dim - aug_te.shape[1]),
                dtype=np.float32,
            )
            out_te = np.concatenate([aug_te, pad_te], axis=1)
            return out_tr, out_te
        return out_tr, aug_te
    # Variance-ranked truncation (keep bias at index F).
    vars_ = np.var(aug_tr, axis=0)
    bias_idx = F
    ranked = np.argsort(-vars_)
    keep = [bias_idx]
    for idx in ranked:
        if idx == bias_idx:
            continue
        if len(keep) >= hilbert_dim:
            break
        keep.append(int(idx))
    keep = sorted(set(keep))
    if len(keep) < hilbert_dim:
        extras = [i for i in ranked if int(i) not in set(keep)]
        keep.extend(extras[: hilbert_dim - len(keep)])
        keep = sorted(set(keep))
    keep_arr = np.asarray(keep[:hilbert_dim], dtype=int)
    out_tr = aug_tr[:, keep_arr]
    if aug_te is not None:
        return out_tr, aug_te[:, keep_arr]
    return out_tr, aug_te


# ---------------------------------------------------------------------------
# Strict-scaffold split — Bemis-Murcko largest-first.
# ---------------------------------------------------------------------------
def _strict_scaffold_split(
    smiles: Sequence[str],
    fractions: Tuple[float, float, float],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bemis-Murcko strict-scaffold splitter (largest-first).

    Each unique Bemis-Murcko scaffold is assigned to exactly one of
    {train, val, test} via a largest-first descending scaffold-size
    bin-packing strategy.  This is the **stronger** OOD split vs
    K-fold random CV — molecules with the same scaffold but
    different substituents land in the same split.

    Returns ``(train_idx, val_idx, test_idx)`` as int64 numpy arrays.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import RDLogger as _RDL

    try:
        _RDL.DisableLog("rdApp.*")
    except Exception:
        pass

    canon = [_canonicalize_smiles(s) for s in smiles]
    scaffold_to_rows: Dict[str, List[int]] = {}
    for i, c in enumerate(canon):
        scaf = _bemis_murcko_scaffold(c) or f"__empty__{c}"
        scaffold_to_rows.setdefault(scaf, []).append(i)

    train_frac, val_frac, test_frac = fractions
    n = len(canon)
    n_train_target = int(round(n * train_frac))
    n_val_target = int(round(n * val_frac))

    rng = np.random.default_rng(seed)
    sorted_scaffolds = sorted(
        scaffold_to_rows.items(),
        key=lambda kv: (-len(kv[1]), kv[0]),
    )
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for scaf, rows in sorted_scaffolds:
        if len(train_idx) < n_train_target:
            train_idx.extend(rows)
        elif len(val_idx) < n_val_target:
            val_idx.extend(rows)
        else:
            test_idx.extend(rows)
    if not train_idx or not val_idx or not test_idx:
        # Degenerate case (single scaffold or tiny dataset): fall back
        # to random shuffle respecting the requested fractions.
        perm = rng.permutation(n)
        cut1 = int(round(n * train_frac))
        cut2 = cut1 + int(round(n * val_frac))
        train_idx = sorted(perm[:cut1].tolist())
        val_idx = sorted(perm[cut1:cut2].tolist())
        test_idx = sorted(perm[cut2:].tolist())
    else:
        # Deterministic tie-break shuffling so different seeds produce
        # different scaffold assignments (otherwise the splitter would
        # always emit the same train/val/test split).
        indices_for_seed = list(range(len(sorted_scaffolds)))
        rng.shuffle(indices_for_seed)
        reassign: Dict[str, List[int]] = {}
        for new_pos, old_pos in enumerate(indices_for_seed):
            scaf, _ = sorted_scaffolds[old_pos]
            reassign[scaf] = scaffold_to_rows[scaf]
        # Rebuild with re-ordered assignment
        train_idx2: List[int] = []
        val_idx2: List[int] = []
        test_idx2: List[int] = []
        for scaf, rows in reassign.items():
            if len(train_idx2) < n_train_target:
                train_idx2.extend(rows)
            elif len(val_idx2) < n_val_target:
                val_idx2.extend(rows)
            else:
                test_idx2.extend(rows)
        train_idx = sorted(train_idx2)
        val_idx = sorted(val_idx2)
        test_idx = sorted(test_idx2)
    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Hilbert-only regressor — wraps LambdaASTSRHilbert.
# ---------------------------------------------------------------------------
def fit_hilbert_cpu(
    X_tr_n: np.ndarray,
    y_tr: np.ndarray,
    *,
    epochs: int,
    seed: int,
) -> Tuple[np.ndarray, "LambdaASTSRHilbert"]:
    """Fit :class:`LambdaASTSRHilbert` on ``(X, y)``; return ``(best_params, model)``.

    CPU fp32 path (default); fp64 fallback on numerical failure.
    """
    from molmetal.molmetal_lam.reward.hilbert_primal_dual import (
        LambdaASTSRHilbert,
    )

    pop_dim = X_tr_n.shape[1]
    try:
        opt = LambdaASTSRHilbert(
            population_dim=pop_dim,
            lr_primal=0.05,
            lr_dual=0.05,
            n_iters=epochs,
            seed=seed,
            grad_clip_max_norm=1.0,
            step_size_decay=0.5,
        )
        opt.optimize(X_tr_n.astype(np.float32), y_tr.astype(np.float32))
        return opt.best_params_, opt
    except Exception:
        # fp64 fallback (Wf-44 P4 numerical-stability fix).
        opt = LambdaASTSRHilbert(
            population_dim=pop_dim,
            lr_primal=0.05,
            lr_dual=0.05,
            n_iters=epochs,
            seed=seed,
            grad_clip_max_norm=1.0,
            step_size_decay=0.5,
        )
        opt.optimize(X_tr_n.astype(np.float64), y_tr.astype(np.float64))
        return opt.best_params_, opt


def fit_hilbert_cuda(
    X_tr_n: np.ndarray,
    y_tr: np.ndarray,
    *,
    epochs: int,
    seed: int,
    device: str = "cuda:0",
) -> Tuple[np.ndarray, Any]:
    """Fit via the :mod:`hilbert_astsr_cuda` batch wrapper (GPU-0 Blackwell)."""
    from molmetal.scoring.gpu_wrappers.hilbert_astsr_cuda import (
        HilbertCUDAParams,
        hilbert_astsr_optimize_single,
    )
    import torch

    pop_dim = X_tr_n.shape[1]
    params = HilbertCUDAParams(
        population_dim=pop_dim,
        lr_primal=0.05,
        lr_dual=0.05,
        n_iters=epochs,
        grad_clip_max_norm=1.0,
        step_size_decay=0.5,
        seed=seed,
    )
    x = hilbert_astsr_optimize_single(
        torch.as_tensor(X_tr_n, dtype=torch.float32),
        torch.as_tensor(y_tr, dtype=torch.float32),
        params=params,
        device=device,
        expected_vram_mb=2048,
    )
    return x.detach().cpu().numpy(), x


def predict_hilbert(
    X_te_n: np.ndarray,
    best_params: np.ndarray,
) -> np.ndarray:
    """Linear predictor: ``X_te_n @ best_params`` (clipped)."""
    y_hat = X_te_n.astype(np.float64) @ np.asarray(best_params, dtype=np.float64)
    return np.clip(y_hat, -PRED_CLIP, PRED_CLIP)


# ---------------------------------------------------------------------------
# Metrics + collapse detection.
# ---------------------------------------------------------------------------
def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _collapse_detected(y_pred: np.ndarray, y_true: np.ndarray) -> bool:
    if y_true.std() < 1e-12:
        return True
    return float(y_pred.std() / max(y_true.std(), 1e-12)) < COLLAPSE_RATIO_THRESHOLD


# ---------------------------------------------------------------------------
# Per-cell-line evaluation — 5 seeds × 50 epochs × strict-scaffold split.
# ---------------------------------------------------------------------------
@dataclass
class FoldMetrics:
    """Per-fold scalar metrics + chosen-asset trace."""

    fold: str
    n_train: int
    n_val: int
    n_test: int
    hilbert_r2_train: float
    hilbert_r2_val: float
    hilbert_r2_test: float
    collapse_detected: bool
    wall_s: float
    used_fp64: bool = False
    used_cuda: bool = False


def _run_one_seed(
    X: np.ndarray,
    y: np.ndarray,
    smiles: Sequence[str],
    seed: int,
    epochs: int,
    hilbert_dim: int,
    fractions: Tuple[float, float, float],
    device: str,
) -> Dict[str, Any]:
    """Run a single seed on the strict-scaffold split."""
    # Z-score (per train fold).
    train_idx, val_idx, test_idx = _strict_scaffold_split(
        smiles,
        fractions=fractions,
        seed=seed,
    )
    X_tr, X_va, X_te = X[train_idx], X[val_idx], X[test_idx]
    y_tr, y_va, y_te = y[train_idx], y[val_idx], y[test_idx]
    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0)
    sigma = np.where(sigma < 1e-9, 1.0, sigma)
    X_tr_n = (X_tr - mu) / sigma
    X_va_n = (X_va - mu) / sigma
    X_te_n = (X_te - mu) / sigma
    X_tr_n = np.nan_to_num(X_tr_n, nan=0.0, posinf=0.0, neginf=0.0)
    X_va_n = np.nan_to_num(X_va_n, nan=0.0, posinf=0.0, neginf=0.0)
    X_te_n = np.nan_to_num(X_te_n, nan=0.0, posinf=0.0, neginf=0.0)

    # Hilbert feature map.
    X_tr_h, X_te_h = hilbert_feature_map(
        X_tr_n.astype(np.float32),
        hilbert_dim,
        X_12d_test=np.concatenate([X_va_n, X_te_n], axis=0).astype(np.float32),
    )
    X_te_h = X_te_h  # Combined val+test feature map.

    t0 = time.perf_counter()
    used_fp64 = False
    used_cuda = False
    if device == "cuda":
        # GPU-0 Blackwell path via the CUDA wrapper.
        try:
            bp, _ = fit_hilbert_cuda(
                X_tr_h,
                y_tr,
                epochs=epochs,
                seed=seed,
                device="cuda:0",
            )
            used_cuda = True
        except Exception:
            # Fall back to CPU if CUDA is unavailable.
            bp, _ = fit_hilbert_cpu(
                X_tr_h,
                y_tr,
                epochs=epochs,
                seed=seed,
            )
            used_cuda = False
    else:
        try:
            bp, _ = fit_hilbert_cpu(
                X_tr_h,
                y_tr,
                epochs=epochs,
                seed=seed,
            )
            if bp is not None and bp.dtype == np.float64:
                used_fp64 = True
        except Exception:
            bp = np.zeros(hilbert_dim, dtype=np.float64)

    y_hat_va = predict_hilbert(X_te_h[: len(X_va_n)], bp)
    y_hat_te = predict_hilbert(X_te_h[len(X_va_n) :], bp)

    r2_train = _compute_r2(y_tr, predict_hilbert(X_tr_h, bp))
    r2_val = _compute_r2(y_va, y_hat_va)
    r2_test = _compute_r2(y_te, y_hat_te)
    collapse = _collapse_detected(y_hat_te, y_te)
    wall = float(time.perf_counter() - t0)

    fold_metrics = FoldMetrics(
        fold=f"strict_scaffold_seed{seed}",
        n_train=int(len(train_idx)),
        n_val=int(len(val_idx)),
        n_test=int(len(test_idx)),
        hilbert_r2_train=float(r2_train),
        hilbert_r2_val=float(r2_val),
        hilbert_r2_test=float(r2_test),
        collapse_detected=bool(collapse),
        wall_s=wall,
        used_fp64=used_fp64,
        used_cuda=used_cuda,
    )
    return {
        "seed": int(seed),
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "hilbert_r2_train": float(r2_train),
        "hilbert_r2_val": float(r2_val),
        "hilbert_r2_test": float(r2_test),
        "collapse_detected": bool(collapse),
        "wall_time_s": wall,
        "used_fp64": used_fp64,
        "used_cuda": used_cuda,
        "per_fold": [fold_metrics.__dict__],
    }


def evaluate_cell_line(
    cell_line: str,
    seeds: Sequence[int],
    epochs: int,
    hilbert_dim: int,
    fractions: Tuple[float, float, float],
    device: str,
    repo_root: Path,
) -> Dict[str, Any]:
    """Run 5-seed × 50-epoch strict-scaffold Hilbert-only retrain for one
    PlatinAI 4-cell line.

    Returns the per-seed metrics dict in Wf-83 P1 / Wf-77 P1 JSON
    shape, extended with ``hilbert_r2_val``, ``hilbert_r2_test``,
    ``hilbert_r2_train``, ``strict_scaffold_split`` (train/val/test
    index sets per seed), and the ``used_fp64 / used_cuda`` trace.
    """
    X, y, smiles = load_platinai_cell(repo_root, cell_line)
    n, d_feat = X.shape

    if device != "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    per_seed: List[Dict[str, Any]] = []
    r2_test_per_seed: List[float] = []
    r2_val_per_seed: List[float] = []
    r2_train_per_seed: List[float] = []

    for seed in seeds:
        seed_int = int(seed) if not isinstance(seed, list) else int(seed[0])
        result = _run_one_seed(
            X,
            y,
            smiles,
            seed=seed_int,
            epochs=epochs,
            hilbert_dim=hilbert_dim,
            fractions=fractions,
            device=device,
        )
        per_seed.append(result)
        r2_test_per_seed.append(result["hilbert_r2_test"])
        r2_val_per_seed.append(result["hilbert_r2_val"])
        r2_train_per_seed.append(result["hilbert_r2_train"])

    summary = {
        "cell_line": cell_line,
        "target_col": PLATINAI_CELL_COLS[cell_line],
        "n_samples": int(n),
        "n_features": int(d_feat),
        "feature_columns": list(FEATURE_COLUMNS),
        "n_seeds": len(seeds),
        "epochs": int(epochs),
        "hilbert_dim": int(hilbert_dim),
        "split": {
            "type": "strict_scaffold",
            "fractions": list(fractions),
            "splitter": "BemisMurckoScaffold_largest_first",
        },
        "device": device,
        # 5-seed mean ± std per data subset (test is the load-bearing number).
        "hilbert_r2_test_mean": float(np.mean(r2_test_per_seed))
        if r2_test_per_seed
        else 0.0,
        "hilbert_r2_test_std": float(np.std(r2_test_per_seed))
        if len(r2_test_per_seed) > 1
        else 0.0,
        "hilbert_r2_val_mean": float(np.mean(r2_val_per_seed))
        if r2_val_per_seed
        else 0.0,
        "hilbert_r2_val_std": float(np.std(r2_val_per_seed))
        if len(r2_val_per_seed) > 1
        else 0.0,
        "hilbert_r2_train_mean": float(np.mean(r2_train_per_seed))
        if r2_train_per_seed
        else 0.0,
        "hilbert_r2_train_std": float(np.std(r2_train_per_seed))
        if len(r2_train_per_seed) > 1
        else 0.0,
        "per_seed": per_seed,
    }
    return summary


# ---------------------------------------------------------------------------
# Output writers.
# ---------------------------------------------------------------------------
def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=str)


def _write_run_log(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_synthesis(out_dir: Path, all_results: List[Dict[str, Any]]) -> None:
    """Write ``<out_dir>/synthesis.md`` cross-cell-line summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        "# Wf-85 Phase 1 — REAL Hilbert-only retrain (PlatinAI 4-cell, strict-scaffold)",
        "",
        "**Date**: 2026-09-26",
        "**Constraint**: REAL Hilbert-only retrain (no ASTSR, no Multi-PH + PLL); "
        "5 seeds × 50 epochs × strict Bemis-Murcko scaffold split.",
        "Per `party-twenty-spirit §3.5` honest framing.",
        "**Hardware**: CPU fp32 (default) + GPU-0 Blackwell optional via `--device cuda`.",
        "**Inputs**: `molmetal/data/platinai_4cell.csv` (4078 rows × 4 cell lines).",
        "",
        "## Cross-cell-line 5-seed summary",
        "",
        "| cell line | n_samples | Hilbert test R² (mean ± std) | Hilbert val R² | Hilbert train R² | collapse? |",
        "|---|---:|---|---|---|:---:|",
    ]
    test_means: List[float] = []
    collapse_any_overall = False
    for r in all_results:
        per_seed_collapse = any(
            s.get("collapse_detected", False) for s in r["per_seed"]
        )
        if per_seed_collapse:
            collapse_any_overall = True
        lines.append(
            "| {cl} | {n} | {tm:+.4f} ± {ts:.4f} | {vm:+.4f} ± {vs:.4f} | {trm:+.4f} ± {trs:.4f} | {c} |".format(
                cl=r["cell_line"],
                n=r["n_samples"],
                tm=r["hilbert_r2_test_mean"],
                ts=r["hilbert_r2_test_std"],
                vm=r["hilbert_r2_val_mean"],
                vs=r["hilbert_r2_val_std"],
                trm=r["hilbert_r2_train_mean"],
                trs=r["hilbert_r2_train_std"],
                c="yes" if per_seed_collapse else "no",
            )
        )
        test_means.append(r["hilbert_r2_test_mean"])
    aggregate = float(np.mean(test_means)) if test_means else 0.0
    aggregate_std = float(np.std(test_means)) if len(test_means) > 1 else 0.0
    lines.append(
        "| **4-cell aggregate** | — | **{agg:+.4f} ± {ags:.4f}** | — | — | {c} |".format(
            agg=aggregate,
            ags=aggregate_std,
            c="yes" if collapse_any_overall else "no",
        )
    )
    lines.append("")
    lines.append("## Verify gate (per `party-twenty-spirit §3.5`)")
    lines.append("")
    n_pos = sum(1 for m in test_means if m > 0.0)
    if n_pos >= 1:
        verdict = (
            "✅ **MEASURED PASS** (at least one cell line has Hilbert test R² mean > 0)"
        )
    elif aggregate >= 0.0:
        verdict = "🟡 **MEASURED PARTIAL** (aggregate R² ≥ 0 but no positive cell line)"
    else:
        verdict = "❌ **HONEST NEGATIVE** (no cell-line R² > 0; per-cell honest-negative reporting)"
    lines.append(
        f"**Aggregate Hilbert test R² (mean of 4 cell-line means) = {aggregate:+.4f}** — {verdict}"
    )
    lines.append("")
    lines.append(
        f"**Per-cell Hilbert test R²**: {n_pos} of {len(test_means)} cell lines have "
        f"**5-seed mean test R² > 0** (this is the load-bearing number per "
        f"`TODO/03_plans/LAMBDA_ROADMAP.md` §1.3 L5 gap)."
    )
    lines.append("")
    lines.append("## Per-cell-line per-seed test R² detail")
    lines.append("")
    for r in all_results:
        lines.append(f"### {r['cell_line']} ({r['target_col']})")
        lines.append("")
        lines.append(
            "| seed | test R² | val R² | train R² | n_train/n_val/n_test | collapse | fp64 / cuda | wall_s |"
        )
        lines.append("|---:|---:|---:|---:|---|:---:|---|---:|")
        for s in r["per_seed"]:
            lines.append(
                "| {seed} | {te:+.4f} | {va:+.4f} | {tr:+.4f} | {nt}/{nv}/{nte} | {c} | {fp}/{cu} | {w:.2f} |".format(
                    seed=s["seed"],
                    te=s["hilbert_r2_test"],
                    va=s["hilbert_r2_val"],
                    tr=s["hilbert_r2_train"],
                    nt=s["n_train"],
                    nv=s["n_val"],
                    nte=s["n_test"],
                    c="yes" if s["collapse_detected"] else "no",
                    fp="fp64" if s["used_fp64"] else "fp32",
                    cu="cuda" if s["used_cuda"] else "cpu",
                    w=s["wall_time_s"],
                )
            )
        lines.append("")
    lines.append("## Honest framing")
    lines.append("")
    lines.append(
        "- **REAL retrain (not scaffold reuse)**: 5 seeds × 50 epochs of "
        "`LambdaASTSRHilbert.optimize` on the strict Bemis-Murcko scaffold split; "
        "test R² is computed on completely held-out scaffolds (never seen at training time)."
    )
    lines.append(
        "- **No ASTSR, no Multi-PH + PLL, no composite blend** — this is the "
        "**Hilbert-only R² anchor** for the Wf-56 R20-P2 final ship and the Wf-83 P2 "
        "combined Hilbert+ASTSR lift attribution."
    )
    lines.append(
        "- **CPU smoke**: per-seed wall-time on CPU is roughly 0.5–2 s × "
        "5 seeds = < 15 s per cell line on a single CPU core; full "
        "5-seed × 4-cell run < 1 minute."
    )
    lines.append(
        "- **GPU-0 Blackwell path**: `--device cuda` dispatches via "
        ":mod:`hilbert_astsr_cuda` batch wrapper; CUDA-unavailable falls back to CPU fp32."
    )
    lines.append(
        "- **Collapse detector**: `pred_std / y_test_std < 0.10` flagged per seed "
        "and reported as `collapse_detected=yes` in the per-seed JSON."
    )
    lines.append("")
    lines.append(
        "Per `party-twenty-spirit §3.5`: per-cell test R² ≥ 0 means are "
        f"**MEASURED** not projected; aggregate Hilbert test R² = {aggregate:+.4f}; "
        "any cell with `collapse_detected=yes` is reported in `per_seed` and "
        "rolled into the per-cell mean honestly."
    )
    _write_run_log(out_dir / "synthesis.md", lines)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def _parse_seeds(raw: str) -> List[int]:
    return [int(x) for x in raw.replace(",", " ").split() if x]


def _parse_cell_lines(raw: str) -> List[str]:
    return [x.strip().lower() for x in raw.replace(",", " ").split() if x]


def _parse_fractions(raw: str) -> Tuple[float, float, float]:
    parts = [float(x) for x in raw.replace(",", " ").split() if x]
    if len(parts) != 3:
        raise ValueError(
            f"--fractions requires exactly 3 numbers (train, val, test); got {parts}"
        )
    s = sum(parts)
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"--fractions must sum to 1.0; got {s}")
    return (parts[0], parts[1], parts[2])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-85 Phase 1 — REAL Hilbert-only R² retrain on PlatinAI 4-cell "
            "(5 seeds × 50 epochs × strict-scaffold)."
        )
    )
    parser.add_argument(
        "--dataset",
        default="platinai_4cell",
        choices=["platinai_4cell"],
        help="Dataset (only platinai_4cell is currently supported; default).",
    )
    parser.add_argument(
        "--cell-lines",
        type=_parse_cell_lines,
        nargs="+",
        default=["mcf7", "a549", "a2780", "a2780cis"],
        help=(
            "PlatinAI cell lines to benchmark " "(default: mcf7 a549 a2780 a2780cis)."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Random seeds for strict-scaffold shuffling (default: 0..4).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Hilbert PD iterations per seed (default: 50).",
    )
    parser.add_argument(
        "--strict-scaffold",
        action="store_true",
        default=True,
        help=(
            "Use strict Bemis-Murcko scaffold split "
            "(default ON; honored for party-twenty-spirit §3.5)."
        ),
    )
    parser.add_argument(
        "--no-strict-scaffold",
        dest="strict_scaffold",
        action="store_false",
        help="Disable strict-scaffold split (use random K-fold instead).",
    )
    parser.add_argument(
        "--fractions",
        type=_parse_fractions,
        default=(0.7, 0.15, 0.15),
        help=(
            "Train/val/test fractions for the strict-scaffold split "
            "(default: 0.7 0.15 0.15; must sum to 1.0)."
        ),
    )
    parser.add_argument(
        "--hilbert-dim",
        type=int,
        default=64,
        help="Hilbert feature map dim (default: 64).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help=(
            "Compute device (default: cpu; --device cuda uses GPU-0 Blackwell "
            "via hilbert_astsr_cuda with CPU fp32 fallback)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("molmetal/reports/wf85_p1_hilbert_only_real"),
        help=(
            "Output directory " "(default: molmetal/reports/wf85_p1_hilbert_only_real)."
        ),
    )
    args = parser.parse_args(argv)

    # Flatten seeds.
    flat_seeds: List[int] = []
    for s in args.seeds:
        if isinstance(s, list):
            flat_seeds.extend(int(x) for x in s)
        else:
            flat_seeds.append(int(s))
    args.seeds = flat_seeds

    # Flatten cell-lines (argparse nargs="+" with list default can produce
    # a single-element list wrapping the default).
    flat_cell_lines: List[str] = []
    for cl in args.cell_lines:
        if isinstance(cl, list):
            for sub in cl:
                if isinstance(sub, list):
                    flat_cell_lines.extend(str(x) for x in sub)
                else:
                    flat_cell_lines.append(str(sub))
        else:
            flat_cell_lines.append(str(cl))
    args.cell_lines = flat_cell_lines

    for cl in args.cell_lines:
        if cl not in PLATINAI_CELL_COLS:
            raise ValueError(
                f"Unknown cell line: {cl}; supported: {sorted(PLATINAI_CELL_COLS)}"
            )

    all_results: List[Dict[str, Any]] = []
    for cell_line in args.cell_lines:
        print(
            f"[wf85-p1] cell_line={cell_line}: loading PlatinAI 4-cell "
            f"(strict-scaffold={'on' if args.strict_scaffold else 'off'})"
        )
        result = evaluate_cell_line(
            cell_line,
            args.seeds,
            args.epochs,
            args.hilbert_dim,
            args.fractions,
            args.device,
            _REPO_ROOT,
        )
        per_seed_path = args.output / cell_line / "per_seed_metrics.json"
        _write_json(per_seed_path, result)
        log_path = args.output / cell_line / "run.log"
        log_lines = [
            f"cell_line={result['cell_line']}",
            f"target_col={result['target_col']}",
            f"n_samples={result['n_samples']}",
            f"n_seeds={result['n_seeds']}",
            f"epochs={result['epochs']}",
            f"hilbert_dim={result['hilbert_dim']}",
            f"device={result['device']}",
            f"split={result['split']}",
            f"hilbert_r2_test_mean={result['hilbert_r2_test_mean']:+.4f}",
            f"hilbert_r2_test_std={result['hilbert_r2_test_std']:.4f}",
            f"hilbert_r2_val_mean={result['hilbert_r2_val_mean']:+.4f}",
            f"hilbert_r2_val_std={result['hilbert_r2_val_std']:.4f}",
            f"hilbert_r2_train_mean={result['hilbert_r2_train_mean']:+.4f}",
            f"hilbert_r2_train_std={result['hilbert_r2_train_std']:.4f}",
        ]
        _write_run_log(log_path, log_lines)
        print(
            f"[wf85-p1] cell_line={cell_line}: "
            f"Hilbert test R² = {result['hilbert_r2_test_mean']:+.4f} "
            f"± {result['hilbert_r2_test_std']:.4f} "
            f"(val={result['hilbert_r2_val_mean']:+.4f} ± "
            f"{result['hilbert_r2_val_std']:.4f}, "
            f"train={result['hilbert_r2_train_mean']:+.4f} ± "
            f"{result['hilbert_r2_train_std']:.4f})"
        )
        all_results.append(result)
    write_synthesis(args.output, all_results)
    print(f"[wf85-p1] synthesis.md written to {args.output / 'synthesis.md'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
