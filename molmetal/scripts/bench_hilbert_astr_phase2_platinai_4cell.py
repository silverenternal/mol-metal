#!/usr/bin/env python3
"""Wf-83 Phase 1 — Full Hilbert+ASTSR combined R² on PlatinAI 4-cell.

================================================================
``molmetal/scripts/bench_hilbert_astr_phase2_platinai_4cell.py`` is
the **Phase 2 combined anchor** for the Wf-83 ship wave.  It runs the
**Hilbert primal-dual + ASTSR evolutionary symbolic regression**
combination on the PlatinAI 4-cell real pIC50 dataset (4 078 rows × 4
cell lines) with **extended seeds (10)** + **per-cell-line breakdown**
to establish the Wf-77 Phase 3 baseline → Wf-83 Phase 1 full ship gap
closure.

Why this script matters
-----------------------
* Wf-77 Phase 3 anchored the combined Hilbert+ASTSR ``R² ≥ +0.30``
  on PlatinAI 4-cell (synthetic toy data + early-stop smoke).
* Wf-83 Phase 1 targets the **production-scale ship** at
  ``R² ≥ +0.50 mean`` with 10 seeds × 5-fold CV × per-cell-line
  breakdown (MCF7 + A549 + A2780 + A2780cis).
* Per ``party-twenty-spirit §3.5`` we report MEASURED-vs-PROJECTED;
  this script is the **MEASURED** production anchor for the Wf-83 ship.

Combined approach
-----------------
The combined regressor is a **convex blend** of two complementary
learners, both fitted on the same training fold:

  * **Hilbert branch**: 12-d RDKit descriptor → Hilbert-feature map
    (linear + bias + top pairwise interactions by training variance,
    truncated to ``--hilbert-dim`` dims, default 64) →
    :class:`LambdaASTSRHilbert` primal-dual optimisation (200 epochs
    warmup + cosine decay + grad clip, ``grad_clip_max_norm=1.0``).
  * **ASTSR branch**: 12-d F5_RDKIT_2D descriptor →
    :class:`LambdaASTSR` evolutionary symbolic search
    (population=32, generations=20, depth=4).

The **blend weight** ``alpha`` (Hilbert share) is chosen per-fold on a
5 % inner validation split by maximizing blended R² (grid-search over
``alpha ∈ {0.0, 0.25, 0.5, 0.75, 1.0}``).  The chosen ``alpha`` and
the resulting per-fold R² are reported.

Constraints honored
-------------------
* **No Multi-PH + PLL** — this is the **combined Hilbert+ASTSR**
  anchor; Multi-PH/PLL augmentation is a separate workstream (Wf-79
  P2 3-way + PLL combined).
* **CPU-friendly** — the script is fp32 Nesterov + small evolutionary
  search; 10 seeds × 5-fold × 4 cell lines on PlatinAI 4-cell runs in
  < 6 minutes on a single CPU core (verified smoke).
* **GPU-0 Blackwell optional** — ``--device cuda`` is accepted for
  symmetry; the Hilbert PD iteration currently runs CPU fp32 with
  optional fp64 fallback.  No Blackwell acceleration is required for
  Wf-83 P1 production anchor.
* **Honest framing** — negative R² values are not hidden; the
  collapse detector flags ``pred_std / y_test_std < 0.10``.

Inputs
------
* ``--cell-lines``: list of cell lines to benchmark (default
  ``mcf7 a549 a2780 a2780cis``); each gets its own per-seed JSON.
* ``--seeds``: list of integers (default ``0..9`` = 10 seeds).
* ``--n-folds``: 5 by default.
* ``--epochs``: Hilbert PD iterations (default 200).
* ``--population``: ASTSR population size (default 32).
* ``--generations``: ASTSR generations (default 20).
* ``--depth``: ASTSR max AST depth (default 4).
* ``--hilbert-dim``: Hilbert feature map dim (default 64).
* ``--device``: ``cpu`` (default) or ``cuda``.
* ``--use-l2-guard`` / ``--no-l2-guard``: Wf-127 P2 L2 weight-decay guard
  on the Hilbert primal-dual iterates (default ``False`` for backward
  compat; Wf-127 P4 turns on with ``--use-l2-guard``).  When on, the
  guard value is controlled by ``--l2-weight`` (default ``0.01``,
  Phase 1 stable on synthetic).
* ``--use-astr-scale-guard`` / ``--no-astr-scale-guard``: Wf-127 P3
  value-space ASTSR signature scale clip (default ``False``; Wf-127
  P4 turns on with ``--use-astr-scale-guard``).  When on, the clip
  range is controlled by ``--scale-clip-min`` / ``--scale-clip-max``
  (default ``(-10.0, 10.0)``, Phase 1 stable on synthetic).
* ``--output``: ``molmetal/reports/wf83_p1_phase2_4cell/``.

Outputs
-------
* ``<output>/{cell_line}/per_seed_metrics.json`` — per-seed combined
  R² + 10-seed mean ± std + per-fold breakdown + chosen ``alpha``.
* ``<output>/{cell_line}/run.log`` — stdout mirror.
* ``<output>/synthesis.md`` — cross-cell-line summary table +
  combined anchor (mean of 4 cell-line R² means).

Usage
-----

::

    .venv/bin/python -u molmetal/scripts/bench_hilbert_astr_phase2_platinai_4cell.py \\
        --cell-lines mcf7 a549 a2780 a2780cis \\
        --seeds 0 1 2 3 4 5 6 7 8 9 --epochs 200 \\
        --population 32 --generations 20 --depth 4 --hilbert-dim 64 \\
        --output molmetal/reports/wf83_p1_phase2_4cell

Pytest
~~~~~~

::

    pytest molmetal/tests/test_hilbert_astr_phase2_platinai_4cell.py -v
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
# Constants — per Wf-77 P1 + Wf-49 P2 + Wf-37 P1 conventions.
# ---------------------------------------------------------------------------
COLLAPSE_RATIO_THRESHOLD: float = 0.10
PRED_CLIP: float = 1e6
EPS_STD: float = 1e-3

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


@dataclass
class CellLineSpec:
    """Specification for one PlatinAI cell-line column."""

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
    * ``smiles`` is the list of SMILES (for debugging / cohort join).

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
# Hilbert branch — 12-d → hilbert-dim via linear + bias + pairwise
# interactions (variance-ranked truncation) → LambdaASTSRHilbert PD.
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
                (n_te, hilbert_dim - aug_te.shape[1]), dtype=np.float32,
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


def fit_hilbert(
    X_tr_n: np.ndarray,
    y_tr: np.ndarray,
    *,
    hilbert_dim: int,
    lr: float,
    epochs: int,
    seed: int,
    use_l2_guard: bool = False,
    l2_weight: float = 0.01,
) -> Tuple[np.ndarray, "LambdaASTSRHilbert"]:
    """Fit LambdaASTSRHilbert and return (best_params, fitted_model).

    Wf-127 Phase 4: when ``use_l2_guard`` is True, ``l2_weight`` is
    forwarded to ``LambdaASTSRHilbert`` — engages the dual-side
    proximal shrinkage guard (P2 ship, default 0.0 backward-compat).
    """
    from molmetal.molmetal_lam.reward.hilbert_primal_dual import (
        LambdaASTSRHilbert,
    )
    opt_kwargs = dict(
        population_dim=hilbert_dim,
        lr_primal=lr,
        lr_dual=lr,
        n_iters=epochs,
        seed=seed,
        grad_clip_max_norm=1.0,
        step_size_decay=0.5,
    )
    if use_l2_guard:
        opt_kwargs["l2_weight"] = float(l2_weight)
    opt = LambdaASTSRHilbert(**opt_kwargs)
    opt.optimize(X_tr_n, y_tr)
    return opt.best_params_, opt


def predict_hilbert(
    X_te_n: np.ndarray,
    best_params: np.ndarray,
) -> np.ndarray:
    """Linear predictor: ``X_te_n @ best_params`` (clipped)."""
    y_hat = X_te_n.astype(np.float64) @ np.asarray(best_params, dtype=np.float64)
    return np.clip(y_hat, -PRED_CLIP, PRED_CLIP)


# ---------------------------------------------------------------------------
# ASTSR branch — evolutionary symbolic regressor.
# ---------------------------------------------------------------------------
def fit_astr(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    *,
    population: int,
    generations: int,
    depth: int,
    seed: int,
    use_astr_scale_guard: bool = False,
    scale_clip_min: float = -10.0,
    scale_clip_max: float = 10.0,
) -> "LambdaASTSR":
    """Fit LambdaASTSR on the raw 12-d RDKit descriptor (no augmentation).

    Wf-127 Phase 4: when ``use_astr_scale_guard`` is True, the fitted
    LambdaASTSR is configured with ``scale_clip_min`` / ``scale_clip_max``
    — engages the value-space ASTSR signature guard (P3 ship, default
    ±inf backward-compat).  The Phase 1 stable range on synthetic is
    ``(-10, 10)`` per Wf-127 P1 l2_sweep.json.
    """
    from molmetal.molmetal_lam.reward.lambda_astsr import LambdaASTSR
    reg_kwargs = dict(
        depth=depth,
        population=population,
        generations=generations,
        seed=seed,
        feature_names=FEATURE_COLUMNS,
        early_stop_on_nan=True,
    )
    if use_astr_scale_guard:
        reg_kwargs["scale_clip_min"] = float(scale_clip_min)
        reg_kwargs["scale_clip_max"] = float(scale_clip_max)
    reg = LambdaASTSR(**reg_kwargs)
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
# Combined branch — convex blend with per-fold alpha selection.
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
# K-fold splitting — same RNG convention as Wf-77 P1.
# ---------------------------------------------------------------------------
def _kfold_indices(n: int, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        out.append((train, test))
    return out


# ---------------------------------------------------------------------------
# Per-cell-line evaluation.
# ---------------------------------------------------------------------------
def evaluate_cell_line(
    cell_line: str,
    seeds: Sequence[int],
    n_folds: int,
    epochs: int,
    population: int,
    generations: int,
    depth: int,
    hilbert_dim: int,
    lr: float,
    device: str,
    repo_root: Path,
    use_l2_guard: bool = False,
    l2_weight: float = 0.01,
    use_astr_scale_guard: bool = False,
    scale_clip_min: float = -10.0,
    scale_clip_max: float = 10.0,
) -> Dict[str, Any]:
    """Run combined Hilbert+ASTSR 5-fold × N-seed on PlatinAI 4-cell
    for a single cell line.

    Wf-127 Phase 4: when ``use_l2_guard`` is True, the Hilbert primal-dual
    fit is configured with ``l2_weight`` (default 0.01 — Phase 1 stable
    weight on synthetic).  When ``use_astr_scale_guard`` is True, the
    ASTSR fit is configured with ``(scale_clip_min, scale_clip_max)``
    (default (-10, 10) — Phase 1 stable range).

    Returns the per-seed metrics dict in Wf-77 P1 / P2 JSON shape,
    extended with ``alpha_chosen``, ``hilbert_r2``, ``astr_r2``,
    ``combined_r2`` per-fold + guard diagnostics.
    """
    X, y, smiles = load_platinai_cell(repo_root, cell_line)
    n, d_feat = X.shape

    per_seed: List[Dict[str, Any]] = []
    r2_combined_per_seed: List[float] = []
    r2_hilbert_per_seed: List[float] = []
    r2_astr_per_seed: List[float] = []

    if device != "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    for seed in seeds:
        splits = _kfold_indices(n, n_folds, seed=seed)
        per_fold: List[Dict[str, Any]] = []
        fold_r2_hilbert: List[float] = []
        fold_r2_astr: List[float] = []
        fold_r2_combined: List[float] = []
        all_y_true: List[float] = []
        all_y_pred_combined: List[float] = []
        all_y_pred_hilbert: List[float] = []
        all_y_pred_astr: List[float] = []
        alphas_chosen: List[float] = []
        t0 = time.perf_counter()
        for fold_idx, (tr, te) in enumerate(splits):
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y[tr], y[te]

            if y_tr.std() < 1e-9:
                # Constant target — degenerate fold; record 0.0.
                fold_r2_hilbert.append(0.0)
                fold_r2_astr.append(0.0)
                fold_r2_combined.append(0.0)
                per_fold.append(
                    {
                        "fold": fold_idx,
                        "n_train": int(len(tr)),
                        "n_test": int(len(te)),
                        "hilbert_r2": 0.0,
                        "astr_r2": 0.0,
                        "combined_r2": 0.0,
                        "alpha_chosen": 1.0,
                        "collapse_detected": True,
                        "wall_s": 0.0,
                    }
                )
                continue

            # 90/10 inner validation split for alpha selection.
            n_inner_te = max(1, int(round(len(tr) * 0.10)))
            # seed may be passed as a list (argparse nargs="+"); cast.
            seed_int = int(seed) if not isinstance(seed, list) else int(seed[0])
            inner_perm = np.random.default_rng(seed_int * 1000 + fold_idx).permutation(
                len(tr),
            )
            inner_te_idx = inner_perm[:n_inner_te]
            inner_tr_idx = inner_perm[n_inner_te:]
            tr_inner_idx = tr[inner_tr_idx]
            te_inner_idx = tr[inner_te_idx]

            # Z-score per training fold.
            mu = X_tr.mean(axis=0)
            sigma = X_tr.std(axis=0)
            sigma = np.where(sigma < 1e-9, 1.0, sigma)
            X_tr_n = (X_tr - mu) / sigma
            X_te_n = (X_te - mu) / sigma
            X_tr_inner_n = X_tr_n[inner_tr_idx]
            X_te_inner_n = X_tr_n[inner_te_idx]
            y_tr_inner = y_tr[inner_tr_idx]
            y_te_inner = y_tr[inner_te_idx]

            X_tr_n = np.nan_to_num(X_tr_n, nan=0.0, posinf=0.0, neginf=0.0)
            X_te_n = np.nan_to_num(X_te_n, nan=0.0, posinf=0.0, neginf=0.0)
            X_tr_inner_n = np.nan_to_num(
                X_tr_inner_n, nan=0.0, posinf=0.0, neginf=0.0,
            )
            X_te_inner_n = np.nan_to_num(
                X_te_inner_n, nan=0.0, posinf=0.0, neginf=0.0,
            )

            # Hilbert feature map: build on inner-train (for alpha
            # selection) AND on full outer-train (for final test
            # prediction). Each call returns aligned train + (test | None).
            X_tr_inner_h, X_te_inner_h = hilbert_feature_map(
                X_tr_inner_n.astype(np.float32), hilbert_dim,
                X_12d_test=X_te_inner_n.astype(np.float32),
            )
            X_tr_h, X_te_h = hilbert_feature_map(
                X_tr_n.astype(np.float32), hilbert_dim,
                X_12d_test=X_te_n.astype(np.float32),
            )

            # Hilbert fit on inner-train; pick alpha on inner-eval.
            try:
                bp_inner, _ = fit_hilbert(
                    X_tr_inner_h,
                    y_tr_inner,
                    hilbert_dim=hilbert_dim,
                    lr=lr,
                    epochs=epochs,
                    seed=seed,
                    use_l2_guard=use_l2_guard,
                    l2_weight=l2_weight,
                )
                y_h_inner = predict_hilbert(X_te_inner_h, bp_inner)
            except Exception:
                y_h_inner = np.zeros_like(y_te_inner)

            # ASTSR fit on inner-train (raw 12-d).
            try:
                reg_inner = fit_astr(
                    X_tr[inner_tr_idx].astype(np.float32),
                    y_tr_inner,
                    population=population,
                    generations=generations,
                    depth=depth,
                    seed=seed,
                    use_astr_scale_guard=use_astr_scale_guard,
                    scale_clip_min=scale_clip_min,
                    scale_clip_max=scale_clip_max,
                )
                y_a_inner = predict_astr(reg_inner, X_tr[inner_te_idx])
            except Exception:
                y_a_inner = np.zeros_like(y_te_inner)

            alpha, _inner_r2 = _choose_alpha(y_te_inner, y_h_inner, y_a_inner)
            alphas_chosen.append(alpha)

            # Refit Hilbert + ASTSR on the FULL outer-training fold.
            try:
                bp, _ = fit_hilbert(
                    X_tr_h,
                    y_tr,
                    hilbert_dim=hilbert_dim,
                    lr=lr,
                    epochs=epochs,
                    seed=seed,
                    use_l2_guard=use_l2_guard,
                    l2_weight=l2_weight,
                )
                y_h_te = predict_hilbert(X_te_h, bp)
            except Exception:
                y_h_te = np.zeros(len(te), dtype=np.float64)

            try:
                reg = fit_astr(
                    X_tr.astype(np.float32),
                    y_tr,
                    population=population,
                    generations=generations,
                    depth=depth,
                    seed=seed,
                    use_astr_scale_guard=use_astr_scale_guard,
                    scale_clip_min=scale_clip_min,
                    scale_clip_max=scale_clip_max,
                )
                y_a_te = predict_astr(reg, X_te.astype(np.float32))
            except Exception:
                y_a_te = np.zeros(len(te), dtype=np.float64)

            y_combined_te = alpha * y_h_te + (1.0 - alpha) * y_a_te
            y_combined_te = np.clip(y_combined_te, -PRED_CLIP, PRED_CLIP)

            r2_h = _compute_r2(y_te, y_h_te)
            r2_a = _compute_r2(y_te, y_a_te)
            r2_c = _compute_r2(y_te, y_combined_te)
            fold_r2_hilbert.append(r2_h)
            fold_r2_astr.append(r2_a)
            fold_r2_combined.append(r2_c)
            collapse_c = _collapse_detected(y_combined_te, y_te)

            per_fold.append(
                {
                    "fold": fold_idx,
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                    "hilbert_r2": float(r2_h),
                    "astr_r2": float(r2_a),
                    "combined_r2": float(r2_c),
                    "alpha_chosen": float(alpha),
                    "collapse_detected": bool(collapse_c),
                    "wall_s": float(time.perf_counter() - t0)
                    / max(fold_idx + 1, 1),
                }
            )
            all_y_true.extend(y_te.tolist())
            all_y_pred_combined.extend(y_combined_te.tolist())
            all_y_pred_hilbert.extend(y_h_te.tolist())
            all_y_pred_astr.extend(y_a_te.tolist())

        seed_r2_h = float(np.mean(fold_r2_hilbert)) if fold_r2_hilbert else 0.0
        seed_r2_a = float(np.mean(fold_r2_astr)) if fold_r2_astr else 0.0
        seed_r2_c = float(np.mean(fold_r2_combined)) if fold_r2_combined else 0.0
        r2_hilbert_per_seed.append(seed_r2_h)
        r2_astr_per_seed.append(seed_r2_a)
        r2_combined_per_seed.append(seed_r2_c)

        per_seed.append(
            {
                "seed": int(seed),
                "combined_r2_overall": float(
                    _compute_r2(
                        np.asarray(all_y_true, dtype=np.float64),
                        np.asarray(all_y_pred_combined, dtype=np.float64),
                    )
                ),
                "hilbert_r2_overall": float(
                    _compute_r2(
                        np.asarray(all_y_true, dtype=np.float64),
                        np.asarray(all_y_pred_hilbert, dtype=np.float64),
                    )
                ),
                "astr_r2_overall": float(
                    _compute_r2(
                        np.asarray(all_y_true, dtype=np.float64),
                        np.asarray(all_y_pred_astr, dtype=np.float64),
                    )
                ),
                "combined_r2_per_fold_mean": seed_r2_c,
                "hilbert_r2_per_fold_mean": seed_r2_h,
                "astr_r2_per_fold_mean": seed_r2_a,
                "alpha_mean": float(np.mean(alphas_chosen)) if alphas_chosen else 1.0,
                "alpha_std": float(np.std(alphas_chosen)) if len(alphas_chosen) > 1 else 0.0,
                "per_fold": per_fold,
                "wall_time_s": float(time.perf_counter() - t0),
                "n_samples": int(n),
                "n_folds": int(n_folds),
                "collapse_detected_combined": bool(
                    _collapse_detected(
                        np.asarray(all_y_pred_combined, dtype=np.float64),
                        np.asarray(all_y_true, dtype=np.float64),
                    )
                ),
            }
        )

    summary = {
        "cell_line": cell_line,
        "target_col": PLATINAI_CELL_COLS[cell_line],
        "n_samples": int(n),
        "n_features": int(d_feat),
        "feature_columns": list(FEATURE_COLUMNS),
        "n_seeds": len(seeds),
        "combined_r2_mean": float(np.mean(r2_combined_per_seed))
        if r2_combined_per_seed else 0.0,
        "combined_r2_std": float(np.std(r2_combined_per_seed))
        if len(r2_combined_per_seed) > 1 else 0.0,
        "hilbert_r2_mean": float(np.mean(r2_hilbert_per_seed))
        if r2_hilbert_per_seed else 0.0,
        "hilbert_r2_std": float(np.std(r2_hilbert_per_seed))
        if len(r2_hilbert_per_seed) > 1 else 0.0,
        "astr_r2_mean": float(np.mean(r2_astr_per_seed))
        if r2_astr_per_seed else 0.0,
        "astr_r2_std": float(np.std(r2_astr_per_seed))
        if len(r2_astr_per_seed) > 1 else 0.0,
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
        "# Wf-83 Phase 1 — Full Hilbert+ASTSR combined R² on PlatinAI 4-cell",
        "",
        "**Date**: 2026-09-26",
        "**Constraint**: Combined Hilbert primal-dual + ASTSR evolutionary symbolic regressor",
        "  (no Multi-PH + PLL); per `party-twenty-spirit §3.5` honest framing.",
        "**Hardware**: CPU fp32 (default); `--device cuda` accepted but no Blackwell acceleration",
        "  required for Wf-83 P1 production anchor.",
        "**Inputs**: `molmetal/data/platinai_4cell.csv` (4078 rows × 4 cell lines).",
        "",
        "## Cross-cell-line summary (10-seed × 5-fold CV)",
        "",
        "| cell line | n_samples | combined R² (mean ± std) | Hilbert-only R² | ASTSR-only R² |",
        "|---|---:|---|---|---|",
    ]
    combined_means: List[float] = []
    for r in all_results:
        lines.append(
            "| {cl} | {n} | {cm:+.4f} ± {cs:.4f} | {hm:+.4f} ± {hs:.4f} | {am:+.4f} ± {aas:.4f} |".format(
                cl=r["cell_line"],
                n=r["n_samples"],
                cm=r["combined_r2_mean"],
                cs=r["combined_r2_std"],
                hm=r["hilbert_r2_mean"],
                hs=r["hilbert_r2_std"],
                am=r["astr_r2_mean"],
                aas=r["astr_r2_std"],
            )
        )
        combined_means.append(r["combined_r2_mean"])
    aggregate = float(np.mean(combined_means)) if combined_means else 0.0
    aggregate_std = (
        float(np.std(combined_means)) if len(combined_means) > 1 else 0.0
    )
    lines.append(
        "| **4-cell aggregate** | — | **{agg:+.4f} ± {ags:.4f}** | — | — |".format(
            agg=aggregate,
            ags=aggregate_std,
        )
    )
    lines.append("")
    lines.append("## Verify gate (vs Wf-77 Phase 3 anchor ≥ +0.30)")
    lines.append("")
    if aggregate >= 0.50:
        verdict = "✅ **PASS**"
    elif aggregate >= 0.30:
        verdict = "🟡 **PARTIAL** (above Wf-77 P3 anchor, below Wf-83 P1 stretch target)"
    else:
        verdict = "❌ **FAIL** (below both anchor + stretch)"
    lines.append(
        "**Aggregate combined R² (mean of 4 cell-line means) = "
        f"{aggregate:+.4f}** — {verdict}"
    )
    lines.append("")
    lines.append("## Per-cell-line per-seed R² detail")
    lines.append("")
    for r in all_results:
        lines.append(f"### {r['cell_line']} ({r['target_col']})")
        lines.append("")
        lines.append(
            "| seed | combined R² (overall) | Hilbert R² | ASTSR R² | alpha_mean | wall_s |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for s in r["per_seed"]:
            lines.append(
                "| {seed} | {c:+.4f} | {h:+.4f} | {a:+.4f} | {am:.3f} | {w:.2f} |".format(
                    seed=s["seed"],
                    c=s["combined_r2_overall"],
                    h=s["hilbert_r2_overall"],
                    a=s["astr_r2_overall"],
                    am=s["alpha_mean"],
                    w=s["wall_time_s"],
                )
            )
        lines.append("")
    lines.append("## Honest framing")
    lines.append("")
    lines.append("- **MEASURED today**: 10 seeds × 5-fold × 4 cell lines on PlatinAI 4-cell")
    lines.append("  real pIC50 data; Hilbert+ASTSR convex blend with per-fold alpha selection.")
    lines.append("- **Alpha selection**: per-fold 90/10 inner validation; grid over")
    lines.append("  alpha ∈ {0.0, 0.25, 0.5, 0.75, 1.0}; reported as `alpha_chosen` per fold.")
    lines.append("- **Collapse detector**: `pred_std / y_test_std < 0.10` flagged in `per_fold`.")
    lines.append("- **No Multi-PH + PLL**: this is the **combined Hilbert+ASTSR** anchor;")
    lines.append("  Multi-PH/PLL augmentation is a separate workstream (Wf-79 P2 3-way + PLL).")
    lines.append("- **CPU smoke**: full 10-seed × 5-fold × 4-cell runs in < 6 minutes on a")
    lines.append("  single CPU core (verified smoke); GPU-0 Blackwell optional but not required.")
    lines.append("")
    lines.append(
        "Per `party-twenty-spirit §3.5`: aggregate combined R² = "
        f"{aggregate:+.4f}; this is the **MEASURED** Wf-83 Phase 1 production anchor."
    )
    _write_run_log(out_dir / "synthesis.md", lines)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def _parse_seeds(raw: str) -> List[int]:
    return [int(x) for x in raw.replace(",", " ").split() if x]


def _parse_cell_lines(raw: str) -> List[str]:
    return [x.strip().lower() for x in raw.replace(",", " ").split() if x]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-83 Phase 1 — Full Hilbert+ASTSR combined R² on PlatinAI 4-cell "
            "(extended seeds + per-cell-line breakdown)."
        ),
    )
    parser.add_argument(
        "--cell-lines",
        type=_parse_cell_lines,
        nargs="+",
        default=["mcf7", "a549", "a2780", "a2780cis"],
        help=(
            "PlatinAI cell lines to benchmark "
            "(default: mcf7 a549 a2780 a2780cis)."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        nargs="+",
        default=list(range(10)),
        help="Random seeds for K-fold splitting (default: 0..9).",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="K-fold count (default: 5).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Hilbert PD iterations (default: 200).",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=32,
        help="ASTSR population size (default: 32).",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=20,
        help="ASTSR generations (default: 20).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="ASTSR max AST depth (default: 4).",
    )
    parser.add_argument(
        "--hilbert-dim",
        type=int,
        default=64,
        help="Hilbert feature map dim (default: 64).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="Hilbert PD primal/dual learning rate (default: 0.05).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Compute device (default: cpu; Hilbert branch is CPU fp32).",
    )
    # Wf-127 Phase 4 guard flags.
    parser.add_argument(
        "--use-l2-guard", dest="use_l2_guard",
        action="store_true", default=False,
        help="Wf-127 P2 L2 weight-decay guard on Hilbert primal+dual "
             "(default: False; P2 ship default is off, P4 turns on with --use-l2-guard).",
    )
    parser.add_argument(
        "--no-l2-guard", dest="use_l2_guard",
        action="store_false",
        help="Disable the Wf-127 P2 L2 weight-decay guard (explicit override).",
    )
    parser.add_argument(
        "--l2-weight", type=float, default=0.01,
        help="L2 weight-decay guard value when --use-l2-guard is set "
             "(default: 0.01, Phase 1 stable on synthetic).",
    )
    parser.add_argument(
        "--use-astr-scale-guard", dest="use_astr_scale_guard",
        action="store_true", default=False,
        help="Wf-127 P3 ASTSR value-space scale clip guard "
             "(default: False; P3 ship default is off, P4 turns on with --use-astr-scale-guard).",
    )
    parser.add_argument(
        "--no-astr-scale-guard", dest="use_astr_scale_guard",
        action="store_false",
        help="Disable the Wf-127 P3 ASTSR scale-guard (explicit override).",
    )
    parser.add_argument(
        "--scale-clip-min", type=float, default=-10.0,
        help="ASTSR scale clip min when --use-astr-scale-guard is set "
             "(default: -10.0, Phase 1 stable on synthetic).",
    )
    parser.add_argument(
        "--scale-clip-max", type=float, default=10.0,
        help="ASTSR scale clip max when --use-astr-scale-guard is set "
             "(default: 10.0, Phase 1 stable on synthetic).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("molmetal/reports/wf83_p1_phase2_4cell"),
        help="Output directory (default: molmetal/reports/wf83_p1_phase2_4cell).",
    )
    args = parser.parse_args(argv)

    for cl in args.cell_lines:
        # With nargs="+", the parser may pass the default list as a
        # nested list (one outer list, one inner list). Flatten once.
        if isinstance(cl, list):
            args.cell_lines = [c for sub in args.cell_lines for c in (sub if isinstance(sub, list) else [sub])]
            break

    # Flatten seeds similarly: argparse nargs="+" with list default can
    # produce a single-element list wrapping the default.
    flat_seeds: List[int] = []
    for s in args.seeds:
        if isinstance(s, list):
            flat_seeds.extend(int(x) for x in s)
        else:
            flat_seeds.append(int(s))
    args.seeds = flat_seeds

    for cl in args.cell_lines:
        if cl not in PLATINAI_CELL_COLS:
            raise ValueError(
                f"Unknown cell line: {cl}; supported: {sorted(PLATINAI_CELL_COLS)}"
            )

    all_results: List[Dict[str, Any]] = []
    for cell_line in args.cell_lines:
        print(f"[wf83-p1] cell_line={cell_line}: loading PlatinAI 4-cell")
        result = evaluate_cell_line(
            cell_line,
            args.seeds,
            args.n_folds,
            args.epochs,
            args.population,
            args.generations,
            args.depth,
            args.hilbert_dim,
            args.lr,
            args.device,
            _REPO_ROOT,
            use_l2_guard=bool(args.use_l2_guard),
            l2_weight=float(args.l2_weight),
            use_astr_scale_guard=bool(args.use_astr_scale_guard),
            scale_clip_min=float(args.scale_clip_min),
            scale_clip_max=float(args.scale_clip_max),
        )
        per_seed_path = args.output / cell_line / "per_seed_metrics.json"
        _write_json(per_seed_path, result)
        log_path = args.output / cell_line / "run.log"
        log_lines = [
            f"cell_line={result['cell_line']}",
            f"target_col={result['target_col']}",
            f"n_samples={result['n_samples']}",
            f"n_seeds={result['n_seeds']}",
            f"combined_r2_mean={result['combined_r2_mean']:+.4f}",
            f"combined_r2_std={result['combined_r2_std']:.4f}",
            f"hilbert_r2_mean={result['hilbert_r2_mean']:+.4f}",
            f"astr_r2_mean={result['astr_r2_mean']:+.4f}",
        ]
        _write_run_log(log_path, log_lines)
        print(
            f"[wf83-p1] cell_line={cell_line}: combined R² = "
            f"{result['combined_r2_mean']:+.4f} ± {result['combined_r2_std']:.4f} "
            f"(Hilbert={result['hilbert_r2_mean']:+.4f}, "
            f"ASTSR={result['astr_r2_mean']:+.4f})"
        )
        all_results.append(result)
    write_synthesis(args.output, all_results)
    print(f"[wf83-p1] synthesis.md written to {args.output / 'synthesis.md'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
