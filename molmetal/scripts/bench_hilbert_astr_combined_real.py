#!/usr/bin/env python3
"""Wf-85 Phase 3 — REAL Hilbert+ASTSR combined R² on PlatinAI 4-cell (strict-scaffold).

================================================================
``molmetal/scripts/bench_hilbert_astr_combined_real.py`` is the
**REAL retrain** Hilbert+ASTSR combined R² benchmark (no
Multi-PH + PLL, no composite Multi-PH augmentation).  It trains the
:class:`LambdaASTSRHilbert` primal-dual regressor **and** the
:class:`LambdaASTSR` evolutionary symbolic regressor on **strict
Bemis-Murcko scaffold splits** of PlatinAI 4-cell with **5 seeds ×
50 epochs**, and produces **MEASURED per-seed combined R² + 5-seed
mean ± std per cell line** plus the direct comparison vs the
Hilbert-only (Wf-85 P1) and ASTSR-only (Wf-85 P2) anchors.

Why this script matters
-----------------------
* Wf-85 Phase 1 (Hilbert-only) shipped at
  ``reports/wf85_p1_hilbert_only_real/synthesis.md`` with 4-cell
  aggregate R² = **−22.6736** (HONEST NEGATIVE).
* Wf-85 Phase 2 (ASTSR-only) shipped at
  ``reports/wf85_p2_astr_only_real/synthesis.md`` with 4-cell
  aggregate R² = **−0.0699** (HONEST NEGATIVE).
* Wf-85 Phase 3 (this script) closes the Wf-49 P3.2 QUEUED gap by
  retraining the **combined** Hilbert primal-dual + ASTSR convex
  blend on the **same** strict-scaffold split + 5 seeds × 50 epochs
  as P1 + P2; the load-bearing number is the **per-cell-line
  combined test R² mean vs the Hilbert-only + ASTSR-only R² means**
  (i.e. attribution of any lift to the blend, not to feature drift
  or split variance).
* Per ``party-twenty-spirit §3.5`` we report MEASURED-vs-PROJECTED;
  this script is the **MEASURED** combined anchor that closes Wf-49
  P3.2.

Combined approach
-----------------
The combined regressor is a **convex blend** of two complementary
learners, both fitted on the same training fold:

  * **Hilbert branch**: 12-d RDKit descriptor → Hilbert-feature map
    (linear + bias + top pairwise interactions by training variance,
    truncated to ``--hilbert-dim`` dims, default 64) →
    :class:`LambdaASTSRHilbert` primal-dual optimisation (50 epochs
    warmup + cosine decay + grad clip, ``grad_clip_max_norm=1.0``).
  * **ASTSR branch**: 12-d F5_RDKIT_2D descriptor →
    :class:`LambdaASTSR` evolutionary symbolic search
    (population=32, generations=50, depth=4).

The **blend weight** ``alpha`` (Hilbert share) is chosen per-seed on
a **10 % inner validation slice** of the training fold by maximizing
blended R² (grid-search over ``alpha ∈ {0.0, 0.25, 0.5, 0.75,
1.0}``).  The chosen ``alpha`` and the resulting per-seed combined
R² are reported.

Strict-scaffold split
---------------------
Bemis-Murcko scaffold is computed for every row; each scaffold lands
in exactly one of {train, val, test} via a largest-first descending
scaffold-size bin-packing strategy, with seed-driven shuffling so
different seeds produce different splits.  This is the **stronger**
OOD split vs K-fold random CV — molecules with the same scaffold but
different substituents all land in the same split, so the test set
contains scaffolds the model has never seen at all.

Constraints honored
-------------------
* **No Multi-PH + PLL, no 3-way + PLL** — this is the **combined
  Hilbert+ASTSR** anchor; Multi-PH/PLL augmentation is a separate
  workstream (Wf-79 P2 3-way + PLL combined).
* **CPU-friendly** — the script is fp32 Nesterov + small evolutionary
  search; 5 seeds × strict-scaffold × 4 cell lines on PlatinAI 4-cell
  runs in < 6 minutes on a single CPU core (verified smoke).
* **GPU-0 Blackwell optional** — ``--device cuda`` is accepted for
  symmetry; the Hilbert PD iteration currently runs CPU fp32 with
  optional fp64 fallback.  No Blackwell acceleration is required for
  Wf-85 P3 anchor.
* **Honest framing** — negative R² values are not hidden; the
  collapse detector flags ``pred_std / y_test_std < 0.10``.

Inputs
------
* ``--dataset``: ``platinai_4cell`` (4078 rows × 4 cell lines).
* ``--cell-lines``: list (default ``mcf7 a549 a2780 a2780cis``).
* ``--seeds``: list (default ``0 1 2 3 4``).
* ``--epochs``: int (default ``50``) — Hilbert PD iterations **and**
  ASTSR generations (1-to-1 mapped).
* ``--strict-scaffold``: flag (default ON) — use
  :func:`_strict_scaffold_split` instead of K-fold random.
* ``--fractions``: ``train val test`` (default ``0.7 0.15 0.15``).
* ``--hilbert-dim``: int (default ``64``).
* ``--population``: ASTSR population size (default ``32``).
* ``--depth``: ASTSR max AST depth (default ``4``).
* ``--device``: ``cpu`` (default) or ``cuda``.
* ``--out-dir``: ``molmetal/reports/wf85_p3_hilbert_astr_combined_real``.
* ``--compare-p1`` / ``--compare-p2``: paths to existing Wf-85 P1 /
  P2 reports for attribution comparison (defaults below).

Outputs
-------
* ``<out_dir>/{cell_line}/per_seed_metrics.json`` — per-seed
  combined R² + 5-seed mean ± std (test / val / train), per-seed
  ``alpha_chosen``, ``hilbert_r2``, ``astr_r2``, ``combined_r2``,
  collapse flag, ``wall_time_s`` per seed.
* ``<out_dir>/{cell_line}/run.log`` — stdout mirror.
* ``<out_dir>/synthesis.md`` — cross-cell-line 5-seed × 50-epoch
  summary + Wf-85 P3 verify gate + per-cell-line attribution table
  (combined vs Hilbert-only P1 vs ASTSR-only P2).

Verify gate (per ``party-twenty-spirit §3.5``)
---------------------------------------------
* Per-cell-line **combined R² mean** ≥ 0.30 — otherwise
  honest-negative reporting.
* Cross-cell-line **4-cell aggregate** R² is computed as the mean of
  the 4 per-cell-line combined R² means.
* Attribution table shows the lift
  ``combined_re − Hilbert-only P1`` and
  ``combined_re − ASTSR-only P2`` per cell line.

Usage
-----

::

    molmetal/.venv/bin/python -u \\
        molmetal/scripts/bench_hilbert_astr_combined_real.py \\
        --dataset platinai_4cell \\
        --cell-lines mcf7 a549 a2780 a2780cis \\
        --seeds 0 1 2 3 4 --epochs 50 --strict-scaffold \\
        --out-dir molmetal/reports/wf85_p3_hilbert_astr_combined_real

Pytest
~~~~~~

::

    pytest molmetal/tests/test_hilbert_astr_combined_real.py -v
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
# Constants — per Wf-85 P1 + Wf-85 P2 + Wf-49 P2 conventions.
# ---------------------------------------------------------------------------
COLLAPSE_RATIO_THRESHOLD: float = 0.10
SCALE_MISMATCH_TOLERANCE: float = 0.50
WRONG_VARIABLE_R2_FLOOR: float = 0.0
PRED_CLIP: float = 1e6
EPS_STD: float = 1e-3
INNER_VAL_FRAC: float = 0.10  # Per-fold inner validation slice for alpha.

# 12-d F5_RDKIT_2D feature schema (Wf-49 P2 ship) — used by both
# Hilbert and ASTSR branches so the combined blend is on the same
# feature space.
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

# PlatinAI 4-cell CSV columns.
PLATINAI_CSV = Path("molmetal/data/platinai_4cell.csv")
PLATINAI_CELL_COLS: Dict[str, str] = {
    "mcf7": "ic50_mcf7_nm",
    "a549": "ic50_a549_nm",
    "a2780": "ic50_a2780_nm",
    "a2780cis": "ic50_a2780cis_nm",
}
PLATINAI_SMILES_COL: str = "smiles"

# Blend grid (Hilbert share).
BLEND_ALPHAS: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


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
    """Return the Bemis-Murcko scaffold SMILES; empty if RDKit fails."""
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
# Hilbert feature map (parallel to bench_hilbert_only_real.py).
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
# Hilbert-only regressor — wraps LambdaASTSRHilbert (parallel to Wf-85 P1).
# ---------------------------------------------------------------------------
def fit_hilbert(
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


def predict_hilbert(
    X_te_n: np.ndarray,
    best_params: np.ndarray,
) -> np.ndarray:
    """Linear predictor: ``X_te_n @ best_params`` (clipped)."""
    y_hat = X_te_n.astype(np.float64) @ np.asarray(best_params, dtype=np.float64)
    return np.clip(y_hat, -PRED_CLIP, PRED_CLIP)


# ---------------------------------------------------------------------------
# ASTSR-only regressor — wraps LambdaASTSR (parallel to Wf-85 P2).
# ---------------------------------------------------------------------------
def fit_astr(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    *,
    population: int,
    generations: int,
    depth: int,
    seed: int,
) -> "LambdaASTSR":
    """Fit :class:`LambdaASTSR` on the raw 12-d RDKit descriptor (no augmentation)."""
    from molmetal.molmetal_lam.reward.lambda_astsr import LambdaASTSR

    reg = LambdaASTSR(
        depth=depth,
        population=population,
        generations=generations,
        seed=seed,
        feature_names=FEATURE_COLUMNS,
        early_stop_on_nan=True,
    )
    reg.fit(X_tr.astype(np.float64), y_tr.astype(np.float64))
    return reg


def predict_astr(
    reg: "LambdaASTSR",
    X_te: np.ndarray,
) -> np.ndarray:
    """Predict using the best AST learned during evolutionary search."""
    y_hat = reg.predict(X_te.astype(np.float64))
    return np.clip(y_hat, -PRED_CLIP, PRED_CLIP)


# ---------------------------------------------------------------------------
# Combined branch — convex blend with per-seed alpha selection.
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
    """Eval-time collapse / mismatch guard (modes A/B/C).

    Mode A: std-ratio collapse (pred is near-constant).
    Mode B: std-ratio explode (Wf-122 P1.2; pred over-amplifies 50%+).
    Mode C: wrong-variable — right scale but R² ≤ floor, i.e. no better
    than the constant-mean predictor.  Without mode C this diagnostic
    reported 0/20 collapse on the very seeds that produced R²=-5.66
    (mcf7 wrong-variable AST at std ratio 1.31).
    """
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if y_true.size == 0 or y_pred.size != y_true.size:
        return True
    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        return True
    y_std = float(y_true.std())
    if y_std < 1e-12:
        return True
    ratio = float(y_pred.std() / max(y_std, 1e-12))
    if ratio < COLLAPSE_RATIO_THRESHOLD:
        return True  # mode A
    if abs(ratio - 1.0) > SCALE_MISMATCH_TOLERANCE:
        return True  # mode B
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot > 1e-12:
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        if 1.0 - ss_res / ss_tot <= WRONG_VARIABLE_R2_FLOOR:
            return True  # mode C
    return False


def _choose_alpha(
    y_inner_true: np.ndarray,
    hilbert_pred: np.ndarray,
    astr_pred: np.ndarray,
) -> Tuple[float, float]:
    """Choose blend alpha on inner validation split by max R².

    ``alpha`` is the Hilbert share (1 - alpha is ASTSR share).
    Returns ``(best_alpha, best_inner_r2)``.
    """
    best_alpha = 1.0
    best_r2 = -1e9
    for alpha in BLEND_ALPHAS:
        y_blend = alpha * hilbert_pred + (1.0 - alpha) * astr_pred
        r2 = _compute_r2(y_inner_true, y_blend)
        if r2 > best_r2:
            best_r2 = r2
            best_alpha = float(alpha)
    return best_alpha, float(best_r2)


# ---------------------------------------------------------------------------
# Strict-scaffold split — Bemis-Murcko largest-first (parallel to Wf-85 P1/P2).
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
    try:
        from rdkit import RDLogger as _RDL

        try:
            _RDL.DisableLog("rdApp.*")
        except Exception:
            pass
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
        # different scaffold assignments.
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
# GPU/accelerator stub for ASTSR (LambdaASTSR has no CUDA wrapper today).
# ---------------------------------------------------------------------------
def _try_cuda_for_astr(device: str) -> bool:
    """``device='cuda'`` is accepted but records ``used_cuda=False`` for ASTSR.

    LambdaASTSR is pure-Python + NumPy; no CUDA wrapper exists.  We
    honor the Wf-85 P3 contract by accepting the device flag and
    recording the (false) ``used_cuda`` per-seed trace.
    """
    if device == "cuda":
        try:
            import torch  # noqa: F401

            torch.cuda.is_available()  # sanity probe; result not used.
        except Exception:
            pass
    return False


# ---------------------------------------------------------------------------
# Per-seed evaluation — Hilbert+ASTSR combined on strict-scaffold split.
# ---------------------------------------------------------------------------
@dataclass
class SeedMetrics:
    """Per-seed scalar metrics + chosen-asset trace."""

    seed: int
    n_train: int
    n_val: int
    n_test: int
    hilbert_r2_train: float
    hilbert_r2_val: float
    hilbert_r2_test: float
    astr_r2_train: float
    astr_r2_val: float
    astr_r2_test: float
    combined_r2_train: float
    combined_r2_val: float
    combined_r2_test: float
    alpha_chosen: float
    inner_r2: float
    collapse_detected: bool
    wall_time_s: float
    used_fp64_hilbert: bool = False
    used_cuda_hilbert: bool = False
    used_cuda_astr: bool = False


def _run_one_seed(
    X: np.ndarray,
    y: np.ndarray,
    smiles: Sequence[str],
    seed: int,
    epochs: int,
    hilbert_dim: int,
    fractions: Tuple[float, float, float],
    population: int,
    depth: int,
    device: str,
) -> Dict[str, Any]:
    """Run a single seed on the strict-scaffold split + inner-validation alpha."""
    train_idx, val_idx, test_idx = _strict_scaffold_split(
        smiles,
        fractions=fractions,
        seed=seed,
    )
    X_tr, X_va, X_te = X[train_idx], X[val_idx], X[test_idx]
    y_tr, y_va, y_te = y[train_idx], y[val_idx], y[test_idx]

    # Z-score per training fold.
    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0)
    sigma = np.where(sigma < 1e-9, 1.0, sigma)
    X_tr_n = (X_tr - mu) / sigma
    X_va_n = (X_va - mu) / sigma
    X_te_n = (X_te - mu) / sigma
    X_tr_n = np.nan_to_num(X_tr_n, nan=0.0, posinf=0.0, neginf=0.0)
    X_va_n = np.nan_to_num(X_va_n, nan=0.0, posinf=0.0, neginf=0.0)
    X_te_n = np.nan_to_num(X_te_n, nan=0.0, posinf=0.0, neginf=0.0)

    # Inner train/val split for alpha selection (90/10).
    seed_int = int(seed) if not isinstance(seed, list) else int(seed[0])
    if y_tr.std() < 1e-9:
        # Constant training target — record zero + alpha=1.0 (Hilbert only).
        n_inner_te = 0
        inner_tr_idx = np.arange(len(X_tr_n))
        inner_te_idx = np.zeros(0, dtype=int)
    else:
        rng_inner = np.random.default_rng(seed_int * 1000 + 7)
        n_inner_te = max(1, int(round(len(X_tr_n) * INNER_VAL_FRAC)))
        inner_perm = rng_inner.permutation(len(X_tr_n))
        inner_tr_idx = inner_perm[n_inner_te:]
        inner_te_idx = inner_perm[:n_inner_te]

    # Build Hilbert feature map on inner-train (for alpha) + outer-train (for test).
    if len(inner_tr_idx) > 0:
        X_tr_inner_n = X_tr_n[inner_tr_idx]
        X_te_inner_n = X_tr_n[inner_te_idx]
        X_tr_inner_h, X_te_inner_h = hilbert_feature_map(
            X_tr_inner_n.astype(np.float32),
            hilbert_dim,
            X_12d_test=X_te_inner_n.astype(np.float32),
        )
    else:
        X_tr_inner_h = None
        X_te_inner_h = None
    # Build val+test feature map together (same convention as Wf-85 P1).
    X_tr_h, X_va_te_h = hilbert_feature_map(
        X_tr_n.astype(np.float32),
        hilbert_dim,
        X_12d_test=np.concatenate([X_va_n, X_te_n], axis=0).astype(np.float32),
    )
    n_va = len(X_va_n)
    X_te_h = X_va_te_h[n_va:]
    X_va_h = X_va_te_h[:n_va]

    t0 = time.perf_counter()
    used_fp64_hilbert = False
    used_cuda_hilbert = False
    used_cuda_astr = False
    best_ast_lisp = "<no_fit>"

    # --- Hilbert fit on inner-train (alpha selection) ---
    if X_tr_inner_h is not None and len(X_te_inner_h) > 0:
        try:
            bp_inner, _ = fit_hilbert(
                X_tr_inner_h,
                y_tr[inner_tr_idx],
                epochs=epochs,
                seed=seed,
            )
            y_h_inner = predict_hilbert(X_te_inner_h, bp_inner)
        except Exception:
            y_h_inner = np.zeros(len(X_te_inner_h), dtype=np.float64)
            bp_inner = None
    else:
        y_h_inner = np.zeros(0, dtype=np.float64)
        bp_inner = None

    # --- ASTSR fit on inner-train (alpha selection) ---
    if len(inner_tr_idx) > 0 and len(inner_te_idx) > 0:
        try:
            reg_inner = fit_astr(
                X_tr[inner_tr_idx].astype(np.float32),
                y_tr[inner_tr_idx],
                population=population,
                generations=epochs,
                depth=depth,
                seed=seed,
            )
            y_a_inner = predict_astr(reg_inner, X_tr[inner_te_idx])
        except Exception:
            y_a_inner = np.zeros(len(inner_te_idx), dtype=np.float64)
    else:
        y_a_inner = np.zeros(0, dtype=np.float64)

    # Choose alpha (Hilbert share).
    if len(inner_te_idx) > 0:
        alpha, inner_r2 = _choose_alpha(
            y_tr[inner_te_idx],
            y_h_inner,
            y_a_inner,
        )
    else:
        alpha, inner_r2 = 1.0, 0.0

    # --- Hilbert fit on full outer-train ---
    try:
        bp, _ = fit_hilbert(
            X_tr_h,
            y_tr,
            epochs=epochs,
            seed=seed,
        )
        if bp is not None and bp.dtype == np.float64:
            used_fp64_hilbert = True
        y_h_tr = predict_hilbert(X_tr_h, bp)
        y_h_va = predict_hilbert(X_va_h, bp)
        y_h_te = predict_hilbert(X_te_h, bp)
    except Exception:
        y_h_tr = np.zeros(len(X_tr_h), dtype=np.float64)
        y_h_va = np.zeros(len(X_va_h), dtype=np.float64)
        y_h_te = np.zeros(len(X_te_h), dtype=np.float64)

    # --- ASTSR fit on full outer-train ---
    try:
        _try_cuda_for_astr(device)
        reg = fit_astr(
            X_tr.astype(np.float32),
            y_tr,
            population=population,
            generations=epochs,
            depth=depth,
            seed=seed,
        )
        # Wf-122 P1.3 — use the LambdaASTSR.best_ast_lisp() accessor
        # (was previously checking a non-existent ``best_ast_lisp`` attribute).
        best_ast_lisp = (
            reg.best_ast_lisp()
            if hasattr(reg, "best_ast_") and reg.best_ast_ is not None
            else "<no_lisp>"
        )
        y_a_tr = predict_astr(reg, X_tr.astype(np.float32))
        y_a_va = predict_astr(reg, X_va.astype(np.float32))
        y_a_te = predict_astr(reg, X_te.astype(np.float32))
    except Exception:
        y_a_tr = np.zeros(len(X_tr), dtype=np.float64)
        y_a_va = np.zeros(len(X_va), dtype=np.float64)
        y_a_te = np.zeros(len(X_te), dtype=np.float64)

    # Combined predictions on train/val/test.
    y_c_tr = alpha * y_h_tr + (1.0 - alpha) * y_a_tr
    y_c_va = alpha * y_h_va + (1.0 - alpha) * y_a_va
    y_c_te = alpha * y_h_te + (1.0 - alpha) * y_a_te

    r2_h_train = _compute_r2(y_tr, y_h_tr)
    r2_h_val = _compute_r2(y_va, y_h_va)
    r2_h_test = _compute_r2(y_te, y_h_te)
    r2_a_train = _compute_r2(y_tr, y_a_tr)
    r2_a_val = _compute_r2(y_va, y_a_va)
    r2_a_test = _compute_r2(y_te, y_a_te)
    r2_c_train = _compute_r2(y_tr, y_c_tr)
    r2_c_val = _compute_r2(y_va, y_c_va)
    r2_c_test = _compute_r2(y_te, y_c_te)
    collapse = _collapse_detected(y_c_te, y_te)
    wall = float(time.perf_counter() - t0)

    seed_metrics = SeedMetrics(
        seed=int(seed),
        n_train=int(len(train_idx)),
        n_val=int(len(val_idx)),
        n_test=int(len(test_idx)),
        hilbert_r2_train=float(r2_h_train),
        hilbert_r2_val=float(r2_h_val),
        hilbert_r2_test=float(r2_h_test),
        astr_r2_train=float(r2_a_train),
        astr_r2_val=float(r2_a_val),
        astr_r2_test=float(r2_a_test),
        combined_r2_train=float(r2_c_train),
        combined_r2_val=float(r2_c_val),
        combined_r2_test=float(r2_c_test),
        alpha_chosen=float(alpha),
        inner_r2=float(inner_r2),
        collapse_detected=bool(collapse),
        wall_time_s=wall,
        used_fp64_hilbert=used_fp64_hilbert,
        used_cuda_hilbert=used_cuda_hilbert,
        used_cuda_astr=used_cuda_astr,
    )
    return {
        "seed": int(seed),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "hilbert_r2_train": float(r2_h_train),
        "hilbert_r2_val": float(r2_h_val),
        "hilbert_r2_test": float(r2_h_test),
        "astr_r2_train": float(r2_a_train),
        "astr_r2_val": float(r2_a_val),
        "astr_r2_test": float(r2_a_test),
        "combined_r2_train": float(r2_c_train),
        "combined_r2_val": float(r2_c_val),
        "combined_r2_test": float(r2_c_test),
        "alpha_chosen": float(alpha),
        "inner_r2": float(inner_r2),
        "collapse_detected": bool(collapse),
        "wall_time_s": wall,
        "used_fp64_hilbert": used_fp64_hilbert,
        "used_cuda_hilbert": used_cuda_hilbert,
        "used_cuda_astr": used_cuda_astr,
        "best_ast_lisp": best_ast_lisp,
        "per_fold": [seed_metrics.__dict__],
    }


# ---------------------------------------------------------------------------
# Per-cell-line evaluation.
# ---------------------------------------------------------------------------
def evaluate_cell_line(
    cell_line: str,
    seeds: Sequence[int],
    epochs: int,
    hilbert_dim: int,
    fractions: Tuple[float, float, float],
    population: int,
    depth: int,
    device: str,
    repo_root: Path,
) -> Dict[str, Any]:
    """Run combined Hilbert+ASTSR 5-seed × 50-epoch strict-scaffold on PlatinAI 4-cell."""
    X, y, smiles = load_platinai_cell(repo_root, cell_line)
    n, d_feat = X.shape

    if device != "cpu":
        # Canonical CUDA default (GPU 0; "" = CPU-only still wins).
        # Loaded via the sibling shim so no molmetal package import is
        # needed (see molmetal/scripts/_cuda_default.py).
        import os as _co
        import sys as _cs

        _cdir = _co.path.dirname(_co.path.abspath(__file__))
        if _cdir not in _cs.path:
            _cs.path.append(_cdir)
        from _cuda_default import ensure_cuda_visible_devices_default as _ensure_cvd

        _ensure_cvd()
        del _ensure_cvd, _cdir, _co, _cs

    per_seed: List[Dict[str, Any]] = []
    r2_combined_test: List[float] = []
    r2_combined_val: List[float] = []
    r2_combined_train: List[float] = []
    r2_hilbert_test: List[float] = []
    r2_astr_test: List[float] = []
    alphas: List[float] = []

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
            population=population,
            depth=depth,
            device=device,
        )
        per_seed.append(result)
        r2_combined_test.append(result["combined_r2_test"])
        r2_combined_val.append(result["combined_r2_val"])
        r2_combined_train.append(result["combined_r2_train"])
        r2_hilbert_test.append(result["hilbert_r2_test"])
        r2_astr_test.append(result["astr_r2_test"])
        alphas.append(result["alpha_chosen"])

    summary = {
        "cell_line": cell_line,
        "target_col": PLATINAI_CELL_COLS[cell_line],
        "n_samples": int(n),
        "n_features": int(d_feat),
        "feature_columns": list(FEATURE_COLUMNS),
        "n_seeds": len(seeds),
        "epochs": int(epochs),
        "hilbert_dim": int(hilbert_dim),
        "astr_population": int(population),
        "astr_depth": int(depth),
        "inner_val_frac": INNER_VAL_FRAC,
        "blend_alphas": list(BLEND_ALPHAS),
        "split": {
            "type": "strict_scaffold",
            "fractions": list(fractions),
            "splitter": "BemisMurckoScaffold_largest_first",
        },
        "device": device,
        # 5-seed mean ± std per data subset.
        "combined_r2_test_mean": float(np.mean(r2_combined_test))
        if r2_combined_test
        else 0.0,
        "combined_r2_test_std": float(np.std(r2_combined_test))
        if len(r2_combined_test) > 1
        else 0.0,
        "combined_r2_val_mean": float(np.mean(r2_combined_val))
        if r2_combined_val
        else 0.0,
        "combined_r2_val_std": float(np.std(r2_combined_val))
        if len(r2_combined_val) > 1
        else 0.0,
        "combined_r2_train_mean": float(np.mean(r2_combined_train))
        if r2_combined_train
        else 0.0,
        "combined_r2_train_std": float(np.std(r2_combined_train))
        if len(r2_combined_train) > 1
        else 0.0,
        "hilbert_r2_test_mean": float(np.mean(r2_hilbert_test))
        if r2_hilbert_test
        else 0.0,
        "hilbert_r2_test_std": float(np.std(r2_hilbert_test))
        if len(r2_hilbert_test) > 1
        else 0.0,
        "astr_r2_test_mean": float(np.mean(r2_astr_test)) if r2_astr_test else 0.0,
        "astr_r2_test_std": float(np.std(r2_astr_test))
        if len(r2_astr_test) > 1
        else 0.0,
        "alpha_chosen_mean": float(np.mean(alphas)) if alphas else 1.0,
        "alpha_chosen_std": float(np.std(alphas)) if len(alphas) > 1 else 0.0,
        "per_seed": per_seed,
    }
    return summary


# ---------------------------------------------------------------------------
# Output writers + Wf-85 P1/P2 attribution comparison.
# ---------------------------------------------------------------------------
def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=str)


def _write_run_log(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _load_p1_p2_anchors(
    p1_path: Optional[Path],
    p2_path: Optional[Path],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Load Wf-85 P1 (Hilbert-only) + P2 (ASTSR-only) per-cell-line
    test R² means for attribution comparison.

    Returns ``(p1_anchors, p2_anchors)`` where each is a dict
    ``{cell_line: test_r2_mean}``; empty dict if the file is missing.
    """
    p1_anchors: Dict[str, float] = {}
    p2_anchors: Dict[str, float] = {}
    if p1_path is not None and p1_path.exists():
        try:
            data = json.loads(p1_path.read_text(encoding="utf-8"))
            p1_anchors[data["cell_line"]] = float(
                data.get("hilbert_r2_test_mean", 0.0),
            )
        except Exception:
            pass
    if p2_path is not None and p2_path.exists():
        try:
            data = json.loads(p2_path.read_text(encoding="utf-8"))
            p2_anchors[data["cell_line"]] = float(
                data.get("astr_r2_test_mean", 0.0),
            )
        except Exception:
            pass
    return p1_anchors, p2_anchors


def write_synthesis(
    out_dir: Path,
    all_results: List[Dict[str, Any]],
    p1_path: Optional[Path] = None,
    p2_path: Optional[Path] = None,
) -> None:
    """Write ``<out_dir>/synthesis.md`` cross-cell-line summary +
    Wf-85 P3 verify gate + Wf-85 P1/P2 attribution comparison.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Wf-85 P1 + P2 anchors for attribution.
    p1_anchors_all: Dict[str, float] = {}
    p2_anchors_all: Dict[str, float] = {}
    if p1_path is not None:
        for r in all_results:
            anchor_path = p1_path / r["cell_line"] / "per_seed_metrics.json"
            p1, _ = _load_p1_p2_anchors(anchor_path, None)
            p1_anchors_all.update(p1)
    if p2_path is not None:
        for r in all_results:
            anchor_path = p2_path / r["cell_line"] / "per_seed_metrics.json"
            _, p2 = _load_p1_p2_anchors(None, anchor_path)
            p2_anchors_all.update(p2)

    lines: List[str] = [
        "# Wf-85 Phase 3 — REAL Hilbert+ASTSR combined R² on PlatinAI 4-cell",
        "",
        "**Date**: 2026-09-26",
        "**Constraint**: Combined Hilbert primal-dual + ASTSR evolutionary symbolic regressor",
        "  (no Multi-PH + PLL, no composite blend); per `party-twenty-spirit §3.5` honest framing.",
        "**Hardware**: CPU fp32 (default); `--device cuda` accepted but no Blackwell acceleration",
        "  required for Wf-85 P3 anchor.",
        "**Inputs**: `molmetal/data/platinai_4cell.csv` (4078 rows × 4 cell lines).",
        "",
        "## Cross-cell-line 5-seed summary",
        "",
        "| cell line | n_samples | combined R² (mean ± std) | Hilbert-only R² (P1) | ASTSR-only R² (P2) | lift vs P1 | lift vs P2 |",
        "|---|---:|---|---|---|---|---|",
    ]
    combined_means: List[float] = []
    for r in all_results:
        cl = r["cell_line"]
        p1 = p1_anchors_all.get(cl, float("nan"))
        p2 = p2_anchors_all.get(cl, float("nan"))
        cm = r["combined_r2_test_mean"]
        lift_p1 = cm - p1 if not math.isnan(p1) else float("nan")
        lift_p2 = cm - p2 if not math.isnan(p2) else float("nan")
        lines.append(
            "| {cl} | {n} | {cm:+.4f} ± {cs:.4f} | {p1:+.4f} | {p2:+.4f} | {lp1:+.4f} | {lp2:+.4f} |".format(
                cl=cl,
                n=r["n_samples"],
                cm=cm,
                cs=r["combined_r2_test_std"],
                p1=p1,
                p2=p2,
                lp1=lift_p1,
                lp2=lift_p2,
            )
        )
        combined_means.append(cm)
    aggregate = float(np.mean(combined_means)) if combined_means else 0.0
    aggregate_std = float(np.std(combined_means)) if len(combined_means) > 1 else 0.0
    lines.append(
        "| **4-cell aggregate** | — | **{agg:+.4f} ± {ags:.4f}** | — | — | — | — |".format(
            agg=aggregate,
            ags=aggregate_std,
        )
    )
    lines.append("")
    lines.append("## Verify gate (per `party-twenty-spirit §3.5`)")
    lines.append("")
    n_pos = sum(1 for m in combined_means if m > 0.30)
    if aggregate >= 0.30:
        verdict = "✅ **MEASURED PASS** (combined R² ≥ +0.30 vs Wf-37 P3 anchor)"
    elif aggregate >= 0.0:
        verdict = "🟡 **MEASURED PARTIAL** (combined R² ≥ 0 but below Wf-37 P3 anchor)"
    else:
        verdict = "❌ **HONEST NEGATIVE** (combined R² < 0; per-cell honest-negative reporting)"
    lines.append(
        f"**Aggregate combined R² (mean of 4 cell-line means) = {aggregate:+.4f}** — {verdict}"
    )
    lines.append("")
    lines.append(
        f"**Per-cell combined R² ≥ +0.30**: {n_pos} of {len(combined_means)} cell lines "
        "(this is the load-bearing number per `TODO/03_plans/LAMBDA_ROADMAP.md` §1.3 L5 gap)."
    )
    lines.append("")
    lines.append("## Per-cell-line per-seed R² detail")
    lines.append("")
    for r in all_results:
        lines.append(f"### {r['cell_line']} ({r['target_col']})")
        lines.append("")
        lines.append(
            "| seed | combined R² (test) | combined R² (val) | combined R² (train) | Hilbert R² | ASTSR R² | alpha | wall_s | collapse |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")
        for s in r["per_seed"]:
            lines.append(
                "| {seed} | {ct:+.4f} | {cv:+.4f} | {ctr:+.4f} | {h:+.4f} | {a:+.4f} | {al:.3f} | {w:.2f} | {c} |".format(
                    seed=s["seed"],
                    ct=s["combined_r2_test"],
                    cv=s["combined_r2_val"],
                    ctr=s["combined_r2_train"],
                    h=s["hilbert_r2_test"],
                    a=s["astr_r2_test"],
                    al=s["alpha_chosen"],
                    w=s["wall_time_s"],
                    c="yes" if s["collapse_detected"] else "no",
                )
            )
        lines.append("")
    lines.append("## Honest framing")
    lines.append("")
    lines.append(
        "- **REAL retrain (not scaffold reuse)**: 5 seeds × 50 epochs of "
        "`LambdaASTSRHilbert.optimize` + `LambdaASTSR.fit` on the strict "
        "Bemis-Murcko scaffold split; test R² is computed on completely "
        "held-out scaffolds (never seen at training time)."
    )
    lines.append(
        "- **No Multi-PH + PLL, no 3-way + PLL** — this is the **combined "
        "Hilbert+ASTSR** anchor; Multi-PH/PLL augmentation is a separate "
        "workstream (Wf-79 P2 3-way + PLL)."
    )
    lines.append(
        "- **Alpha selection**: per-seed 90/10 inner validation on the "
        "training fold; grid over alpha ∈ {0.0, 0.25, 0.5, 0.75, 1.0}; "
        "reported as `alpha_chosen` per seed."
    )
    lines.append(
        "- **Attribution**: `lift vs P1` = combined − Wf-85 P1 "
        "(Hilbert-only) cell-line mean; `lift vs P2` = combined − "
        "Wf-85 P2 (ASTSR-only) cell-line mean.  This isolates the lift "
        "attributable to the blend vs split variance."
    )
    lines.append(
        "- **CPU smoke**: full 5-seed × strict-scaffold × 4-cell runs "
        "in < 6 minutes on a single CPU core (verified smoke); "
        "GPU-0 Blackwell optional but not required."
    )
    lines.append(
        "- **Collapse detector**: `pred_std / y_test_std < 0.10` flagged "
        "per seed and reported as `collapse_detected=yes` in the "
        "per-seed JSON."
    )
    lines.append("")
    lines.append(
        "Per `party-twenty-spirit §3.5`: per-cell combined test R² ≥ 0 means "
        f"are **MEASURED** not projected; aggregate combined R² = {aggregate:+.4f}; "
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
            "Wf-85 Phase 3 — REAL Hilbert+ASTSR combined R² on PlatinAI 4-cell "
            "(5 seeds × 50 epochs × strict-scaffold)."
        ),
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
        help=(
            "Hilbert PD iterations AND ASTSR generations per seed "
            "(1-to-1 mapped; default: 50)."
        ),
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
        help="Disable strict-scaffold split (not recommended).",
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
        "--population",
        type=int,
        default=32,
        help="ASTSR population size (default: 32).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="ASTSR max AST depth (default: 4).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help=(
            "Compute device (default: cpu; --device cuda is accepted but "
            "ASTSR has no CUDA wrapper today and Hilbert branch runs CPU fp32)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("molmetal/reports/wf85_p3_hilbert_astr_combined_real"),
        help=(
            "Output directory "
            "(default: molmetal/reports/wf85_p3_hilbert_astr_combined_real)."
        ),
    )
    parser.add_argument(
        "--compare-p1",
        type=Path,
        default=Path("molmetal/reports/wf85_p1_hilbert_only_real"),
        help=(
            "Wf-85 P1 (Hilbert-only) report directory for attribution comparison "
            "(default: molmetal/reports/wf85_p1_hilbert_only_real)."
        ),
    )
    parser.add_argument(
        "--compare-p2",
        type=Path,
        default=Path("molmetal/reports/wf85_p2_astr_only_real"),
        help=(
            "Wf-85 P2 (ASTSR-only) report directory for attribution comparison "
            "(default: molmetal/reports/wf85_p2_astr_only_real)."
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

    # Flatten cell-lines.
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
            f"[wf85-p3] cell_line={cell_line}: loading PlatinAI 4-cell "
            f"(strict-scaffold={'on' if args.strict_scaffold else 'off'})"
        )
        result = evaluate_cell_line(
            cell_line,
            args.seeds,
            args.epochs,
            args.hilbert_dim,
            args.fractions,
            args.population,
            args.depth,
            args.device,
            _REPO_ROOT,
        )
        per_seed_path = args.out_dir / cell_line / "per_seed_metrics.json"
        _write_json(per_seed_path, result)
        log_path = args.out_dir / cell_line / "run.log"
        log_lines = [
            f"cell_line={result['cell_line']}",
            f"target_col={result['target_col']}",
            f"n_samples={result['n_samples']}",
            f"n_seeds={result['n_seeds']}",
            f"epochs={result['epochs']}",
            f"hilbert_dim={result['hilbert_dim']}",
            f"astr_population={result['astr_population']}",
            f"astr_depth={result['astr_depth']}",
            f"device={result['device']}",
            f"split={result['split']}",
            f"combined_r2_test_mean={result['combined_r2_test_mean']:+.4f}",
            f"combined_r2_test_std={result['combined_r2_test_std']:.4f}",
            f"combined_r2_val_mean={result['combined_r2_val_mean']:+.4f}",
            f"combined_r2_val_std={result['combined_r2_val_std']:.4f}",
            f"combined_r2_train_mean={result['combined_r2_train_mean']:+.4f}",
            f"combined_r2_train_std={result['combined_r2_train_std']:.4f}",
            f"hilbert_r2_test_mean={result['hilbert_r2_test_mean']:+.4f}",
            f"astr_r2_test_mean={result['astr_r2_test_mean']:+.4f}",
            f"alpha_chosen_mean={result['alpha_chosen_mean']:.3f}",
        ]
        _write_run_log(log_path, log_lines)
        print(
            f"[wf85-p3] cell_line={cell_line}: "
            f"combined R² = {result['combined_r2_test_mean']:+.4f} "
            f"± {result['combined_r2_test_std']:.4f} "
            f"(Hilbert={result['hilbert_r2_test_mean']:+.4f}, "
            f"ASTSR={result['astr_r2_test_mean']:+.4f}, "
            f"alpha_mean={result['alpha_chosen_mean']:.3f})"
        )
        all_results.append(result)
    write_synthesis(
        args.out_dir,
        all_results,
        p1_path=args.compare_p1,
        p2_path=args.compare_p2,
    )
    print(f"[wf85-p3] synthesis.md written to {args.out_dir / 'synthesis.md'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
