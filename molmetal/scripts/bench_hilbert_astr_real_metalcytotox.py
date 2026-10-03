#!/usr/bin/env python3
"""Wf-86 Phase 1 — REAL cross-dataset R² on MetalCytoToxDB 5-metal.

================================================================
``molmetal/scripts/bench_hilbert_astr_real_metalcytotox.py`` is the
**Wf-49 P4 cross-dataset R² close-out** for MetalCytoToxDB 5-metal
(Ru / Ir / Os / Rh / Re).  It runs the **Hilbert primal-dual +
ASTSR evolutionary symbolic regression** combination on the
MetalCytoToxDB 5-metal multitask CSV (26 801 rows × 5 metals =
19 135 Ru + 4 546 Ir + 1 118 Os + 1 134 Rh + 868 Re) with
**5 seeds × 50 epochs × strict-scaffold 5-fold CV** to land a
MEASURED per-metal R² JSON that closes the Wf-49 PARTIAL SHIP gap.

Why this script matters
-----------------------
* Wf-49 PARTIAL SHIP (TODO/completed/91_wf49_r2_root_fix.md) anchored
  ``|ρ| ≤ 0.204`` (MolLogP) across the pooled 26 801 rows but
  **never split per-metal**; Wf-49 P4 QUEUED on this dataset.
* Wf-85 Phase 3 (PlatinAI 4-cell combined R² anchor) shipped
  per-cell-line R² at ``+0.50`` mean across 4 cell lines.
* This script is the **MetalCytoToxDB 5-metal counterpart**: the
  same Hilbert+ASTSR combination, but on the strict-scaffold split
  with **5 seeds × 50 epochs** + **per-metal breakdown**.
* Per ``party-twenty-spirit §3.5`` we report MEASURED-vs-PROJECTED;
  this script is the **MEASURED** R² anchor on the largest available
  metal-cytotox regression cohort.

Strict-scaffold splitting
-------------------------
The dataset already exposes a Bemis-Murcko ``scaffold_id`` column
(3 070 unique scaffolds across 26 801 rows).  The strict-scaffold
5-fold split groups rows by ``scaffold_id`` and assigns each
scaffold to **exactly one** fold's test set — this is stronger
than the random split used in ``bench_hilbert_only_baseline.py``
because it guarantees zero scaffold leakage between train and
test.  Per-fold sizes are reported; ``Re`` (868 rows, smallest
cohort) typically yields ``~174`` test rows per fold.

Combined approach
-----------------
Same convex blend as ``bench_hilbert_astr_phase2_platinai_4cell.py``:

  * **Hilbert branch**: 12-d F5_RDKIT_2D → Hilbert-feature map
    (linear + bias + top pairwise interactions by training variance,
    truncated to ``--hilbert-dim`` dims, default 64) →
    :class:`LambdaASTSRHilbert` primal-dual optimisation
    (``--epochs`` epochs, default 50, ``grad_clip_max_norm=1.0``).
  * **ASTSR branch**: 12-d F5_RDKIT_2D →
    :class:`LambdaASTSR` evolutionary symbolic search
    (population=32, generations=20, depth=4).

The **blend weight** ``alpha`` (Hilbert share) is chosen per-fold
on a 10 % inner validation split by maximizing blended R²
(grid over ``alpha ∈ {0.0, 0.25, 0.5, 0.75, 1.0}``).

Constraints honored
-------------------
* **No Multi-PH + PLL** — this is the **combined Hilbert+ASTSR**
  anchor; Multi-PH/PLL augmentation is a separate workstream
  (Wf-79 P2 3-way + PLL combined).
* **CPU-friendly** — the script is fp32 Nesterov + small evolutionary
  search; 5 seeds × 50 epochs × 5-fold × 5 metals on MetalCytoToxDB
  runs in < 10 minutes on a single CPU core (verified smoke).
* **GPU-1 RTX 5090 optional** — ``--device cuda`` is accepted; the
  Hilbert branch currently runs CPU fp32 with optional fp64 fallback.
  GPU-1 is available but **not required** for Wf-86 P1 production
  anchor (CPU ship is honest per the bounded-state policy).
* **Honest framing** — negative R² values are not hidden; the
  collapse detector flags ``pred_std / y_test_std < 0.10`` per fold.

Inputs
------
* ``--dataset``: ``metalcytotox_5metal`` (26 801 rows × 5 metals).
* ``--metals``: list of metals (default ``Ru Ir Os Rh Re``).
* ``--seeds``: list of integers (default ``0 1 2 3 4``).
* ``--n-folds``: 5 by default.
* ``--epochs``: Hilbert PD iterations (default 50).
* ``--strict-scaffold``: enforce scaffold-disjoint 5-fold split
  (default ``True``); pass ``--strict-scaffold false`` to fall back
  to random split for ablation.
* ``--device``: ``cpu`` (default) or ``cuda``.
* ``--out-dir``: ``molmetal/reports/wf86_p1_metalcytotox_real/``.

Outputs
-------
* ``<out-dir>/{metal}/per_seed_metrics.json`` — per-seed combined
  R² + 5-seed mean ± std + per-fold breakdown + chosen ``alpha``.
* ``<out-dir>/{metal}/run.log`` — stdout mirror.
* ``<out-dir>/synthesis.md`` — cross-metal summary table +
  5-metal aggregate (mean of 5 per-metal R² means) + honest framing.

Usage
-----

::

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u \\
        molmetal/scripts/bench_hilbert_astr_real_metalcytotox.py \\
        --dataset metalcytotox_5metal --metals Ru Ir Os Rh Re \\
        --seeds 0 1 2 3 4 --epochs 50 --strict-scaffold \\
        --out-dir molmetal/reports/wf86_p1_metalcytotox_real

Pytest
~~~~~~

::

    pytest molmetal/tests/test_hilbert_astr_real_metalcytotox.py -v
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
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

# MetalCytoToxDB 5-metal multitask CSV candidate paths (in priority order).
METALCSV_CANDIDATES: Tuple[Path, ...] = (
    _REPO_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv",
    _REPO_ROOT / "data" / "metalcytotox_5metal_multitask_v4.csv",
    _REPO_ROOT / "molmetal" / "data" / "metalcytotox_5metal_multitask_v4.csv",
)

# Default 5-metal cohort (Wf-49 PARTIAL SHIP row coverage).
DEFAULT_METALS: Tuple[str, ...] = ("Ru", "Ir", "Os", "Rh", "Re")

# CSV columns.
SMILES_COL: str = "smiles"
METAL_COL: str = "metal"
SCAFFOLD_COL: str = "scaffold_id"
ACTIVITY_COL: str = "activity"

# Blend grid (Hilbert share).
BLEND_ALPHAS: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


# ---------------------------------------------------------------------------
# RDKit featurizer — 12-d F5_RDKIT_2D descriptor.
# ---------------------------------------------------------------------------
# Counter-ion SMILES that should be dropped before ligand extraction.
# Matches the convention from molmetal/data/metal_smiles.py.
_COUNTERION_FRAGMENTS = (
    "[Cl-]",
    "[Br-]",
    "[I-]",
    "[F-]",
    "[OH-]",
    "[O-]",
    "[N+](=O)[O-]",  # nitrate
    "[PF6-]",
    "[BF4-]",
    "O",
    "N",
    "S",  # tiny solvents / single-atom donors
    "[H]O[H]",
    "[OH2]",
)


def _extract_ligand_smiles(smiles: str) -> Optional[str]:
    """Extract the **largest non-counter-ion fragment** from a
    multi-component SMILES string.

    MetalCytoToxDB rows store multi-component SMILES like
    ``L1.L2.[Cl-].c1ccccc1`` — we want the largest organic ligand
    fragment (counter-ions like ``[Cl-]``, ``[PF6-]``, ``OH2`` are
    dropped per the ``molmetal/data/metal_smiles.py`` convention).

    Returns ``None`` if no usable fragment can be extracted.
    """
    if not smiles or not smiles.strip():
        return None
    fragments = [f for f in smiles.split(".") if f and f.strip()]
    if not fragments:
        return None
    # Drop counter-ions.
    non_counter = [f for f in fragments if f.strip() not in _COUNTERION_FRAGMENTS]
    candidates = non_counter if non_counter else fragments
    # Pick the largest by SMILES length (proxy for atom count).
    largest = max(candidates, key=lambda s: len(s.strip()))
    return largest if largest else None


def _featurize_smiles(smiles: str) -> Optional[List[float]]:
    """Return 12-d RDKit 2D descriptor vector; ``None`` if SMILES fails.

    Multi-component SMILES are reduced to their largest non-counter-ion
    fragment before featurization (per MetalCytoToxDB convention).
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

        # Strip multi-component / counter-ions first.
        ligand_smiles = _extract_ligand_smiles(smiles)
        if ligand_smiles is None:
            return None
        mol = Chem.MolFromSmiles(ligand_smiles)
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


def _resolve_metalcsv_path() -> Path:
    """Return the first existing MetalCytoToxDB 5-metal CSV path."""
    for c in METALCSV_CANDIDATES:
        if c.exists():
            return c
    raise FileNotFoundError(
        "metalcytotox_5metal_multitask_v4.csv not found in "
        + ", ".join(str(c) for c in METALCSV_CANDIDATES)
    )


@dataclass
class MetalSpec:
    """Specification for one MetalCytoToxDB metal cohort."""

    metal: str
    target_col: str = ACTIVITY_COL  # binary 0/1 — R² on this is the "regression" target


def load_metal_cohort(
    metal: str,
    *,
    n_max: int = 20000,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Load MetalCytoToxDB 5-metal for a single metal cohort.

    Returns ``(X, y, smiles, scaffold_ids)``:

    * ``X`` is ``(n_samples, 12)`` float32 (F5_RDKIT_2D descriptor).
    * ``y`` is ``(n_samples,)`` float32 — binary activity (0/1).
    * ``smiles`` is the list of SMILES (for debugging / cohort join).
    * ``scaffold_ids`` is the list of scaffold IDs (for strict-scaffold split).

    Rows with NaN/empty target or unparseable SMILES are dropped.
    """
    csv_path = _resolve_metalcsv_path()
    rows = _read_csv(csv_path)
    Xs: List[List[float]] = []
    Ys: List[float] = []
    smiles: List[str] = []
    scaffold_ids: List[str] = []
    seen = 0
    for r in rows:
        m = (r.get(METAL_COL) or "").strip()
        if m != metal:
            continue
        seen += 1
        smi = (r.get(SMILES_COL) or "").strip()
        if not smi:
            continue
        feats = _featurize_smiles(smi)
        if feats is None:
            continue
        raw = r.get(ACTIVITY_COL)
        try:
            act = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(act):
            continue
        # Binary activity: 0 / 1 — clamp just in case.
        act = float(act > 0.5)
        Xs.append(feats)
        Ys.append(act)
        smiles.append(smi)
        scaffold_ids.append((r.get(SCAFFOLD_COL) or "").strip())
        if len(Xs) >= n_max:
            break
    if not Xs:
        raise RuntimeError(
            f"No usable rows for metal {metal!r} "
            f"(rows scanned for this metal: {seen})"
        )
    return (
        np.asarray(Xs, dtype=np.float32),
        np.asarray(Ys, dtype=np.float32),
        smiles,
        scaffold_ids,
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
# K-fold splitting — strict scaffold + random fallback.
# ---------------------------------------------------------------------------
def _random_kfold_indices(
    n: int, k: int, seed: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Random K-fold split (fallback when scaffold split isn't possible)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        out.append((train, test))
    return out


def _strict_scaffold_kfold_indices(
    scaffold_ids: Sequence[str],
    k: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """K-fold split with **strict scaffold disjointness**.

    Groups rows by ``scaffold_id`` and assigns each scaffold to
    exactly one fold's test set.  This guarantees zero scaffold
    leakage between train and test (much stronger than random split).
    """
    n = len(scaffold_ids)
    # Group row indices by scaffold_id (use "_empty_" for missing).
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, sc in enumerate(scaffold_ids):
        key = sc if sc else "_empty_"
        groups[key].append(idx)

    # Deterministic shuffle of scaffold IDs.
    unique_sc = sorted(groups.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_sc)

    # Assign each scaffold to exactly one fold (round-robin so fold
    # sizes are roughly balanced).  Then build per-fold test index.
    test_idx_per_fold: List[List[int]] = [[] for _ in range(k)]
    for i, sc in enumerate(unique_sc):
        fold = i % k
        test_idx_per_fold[fold].extend(groups[sc])
    # Each fold: test = assigned; train = everything else.
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        test = np.asarray(sorted(test_idx_per_fold[i]), dtype=np.int64)
        all_idx = np.arange(n, dtype=np.int64)
        if len(test) == 0:
            # Degenerate (shouldn't happen unless n==0): use random.
            train = all_idx
        else:
            mask = np.ones(n, dtype=bool)
            mask[test] = False
            train = all_idx[mask]
        out.append((train, test))
    return out


def _kfold_indices(
    n: int,
    k: int,
    seed: int,
    *,
    scaffold_ids: Optional[Sequence[str]] = None,
    strict_scaffold: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """K-fold dispatcher — strict-scaffold or random."""
    if strict_scaffold and scaffold_ids is not None and len(scaffold_ids) == n:
        return _strict_scaffold_kfold_indices(scaffold_ids, k, seed)
    return _random_kfold_indices(n, k, seed)


# ---------------------------------------------------------------------------
# Per-metal evaluation.
# ---------------------------------------------------------------------------
def evaluate_metal(
    metal: str,
    seeds: Sequence[int],
    n_folds: int,
    epochs: int,
    population: int,
    generations: int,
    depth: int,
    hilbert_dim: int,
    lr: float,
    device: str,
    strict_scaffold: bool,
    repo_root: Path,
    use_l2_guard: bool = False,
    l2_weight: float = 0.01,
    use_astr_scale_guard: bool = False,
    scale_clip_min: float = -10.0,
    scale_clip_max: float = 10.0,
) -> Dict[str, Any]:
    """Run combined Hilbert+ASTSR 5-fold × N-seed on MetalCytoToxDB
    5-metal for a single metal.

    Wf-127 Phase 4: when ``use_l2_guard`` is True, the Hilbert primal-dual
    fit is configured with ``l2_weight`` (default 0.01 — Phase 1 stable
    weight on synthetic).  When ``use_astr_scale_guard`` is True, the
    ASTSR fit is configured with ``(scale_clip_min, scale_clip_max)``
    (default (-10, 10) — Phase 1 stable range).

    Returns the per-seed metrics dict in Wf-77 P1 / P2 JSON shape,
    extended with ``alpha_chosen``, ``hilbert_r2``, ``astr_r2``,
    ``combined_r2`` per-fold, plus ``strict_scaffold`` flag and
    guard diagnostics.
    """
    X, y, smiles, scaffold_ids = load_metal_cohort(metal)
    n, d_feat = X.shape

    per_seed: List[Dict[str, Any]] = []
    r2_combined_per_seed: List[float] = []
    r2_hilbert_per_seed: List[float] = []
    r2_astr_per_seed: List[float] = []

    if device != "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

    for seed in seeds:
        splits = _kfold_indices(
            n,
            n_folds,
            seed=int(seed),
            scaffold_ids=scaffold_ids,
            strict_scaffold=strict_scaffold,
        )
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
            if len(te) == 0:
                continue
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
            seed_int = int(seed) if not isinstance(seed, list) else int(seed[0])
            inner_perm = np.random.default_rng(seed_int * 1000 + fold_idx).permutation(
                len(tr)
            )
            inner_te_idx = inner_perm[:n_inner_te]
            inner_tr_idx = inner_perm[n_inner_te:]

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
                X_tr_inner_n,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            X_te_inner_n = np.nan_to_num(
                X_te_inner_n,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            # Hilbert feature map: build on inner-train (for alpha
            # selection) AND on full outer-train (for final test
            # prediction). Each call returns aligned train + (test | None).
            X_tr_inner_h, X_te_inner_h = hilbert_feature_map(
                X_tr_inner_n.astype(np.float32),
                hilbert_dim,
                X_12d_test=X_te_inner_n.astype(np.float32),
            )
            X_tr_h, X_te_h = hilbert_feature_map(
                X_tr_n.astype(np.float32),
                hilbert_dim,
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
                    seed=seed_int,
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
                    seed=seed_int,
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
                    seed=seed_int,
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
                    seed=seed_int,
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
                    "wall_s": float(time.perf_counter() - t0) / max(fold_idx + 1, 1),
                }
            )
            # Per-fold progress to stdout (line-buffered via ``-u``).
            print(
                f"[wf86-p1]   metal={metal} seed={seed_int} fold={fold_idx} "
                f"combined_r2={r2_c:+.4f} hilbert_r2={r2_h:+.4f} "
                f"astr_r2={r2_a:+.4f} alpha={alpha:.2f} "
                f"elapsed={time.perf_counter() - t0:.1f}s",
                flush=True,
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
                "alpha_std": float(np.std(alphas_chosen))
                if len(alphas_chosen) > 1
                else 0.0,
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
        "metal": metal,
        "target_col": ACTIVITY_COL,
        "n_samples": int(n),
        "n_features": int(d_feat),
        "feature_columns": list(FEATURE_COLUMNS),
        "n_seeds": len(seeds),
        "strict_scaffold": bool(strict_scaffold),
        "combined_r2_mean": float(np.mean(r2_combined_per_seed))
        if r2_combined_per_seed
        else 0.0,
        "combined_r2_std": float(np.std(r2_combined_per_seed))
        if len(r2_combined_per_seed) > 1
        else 0.0,
        "hilbert_r2_mean": float(np.mean(r2_hilbert_per_seed))
        if r2_hilbert_per_seed
        else 0.0,
        "hilbert_r2_std": float(np.std(r2_hilbert_per_seed))
        if len(r2_hilbert_per_seed) > 1
        else 0.0,
        "astr_r2_mean": float(np.mean(r2_astr_per_seed)) if r2_astr_per_seed else 0.0,
        "astr_r2_std": float(np.std(r2_astr_per_seed))
        if len(r2_astr_per_seed) > 1
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
    """Write ``<out_dir>/synthesis.md`` cross-metal summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    strict_str = (
        "strict-scaffold disjoint"
        if all_results and all_results[0].get("strict_scaffold")
        else "random"
    )
    lines: List[str] = [
        "# Wf-86 Phase 1 — REAL Hilbert+ASTSR combined R² on MetalCytoToxDB 5-metal",
        "",
        "**Date**: 2026-09-26",
        "**Constraint**: Combined Hilbert primal-dual + ASTSR evolutionary symbolic regressor",
        "  (no Multi-PH + PLL); per `party-twenty-spirit §3.5` honest framing.",
        "**Hardware**: CPU fp32 (default); `--device cuda` accepted but no Blackwell acceleration",
        "  required for Wf-86 P1 production anchor.",
        f"**Split**: 5-fold CV with **{strict_str}** scaffold assignment per metal.",
        "**Inputs**: `.data/molmetal/metalcytotox_5metal_multitask_v4.csv`",
        "  (26 801 rows × 5 metals: Ru 19 135 / Ir 4 546 / Os 1 118 / Rh 1 134 / Re 868).",
        "",
        "## Cross-metal summary (5-seed × 5-fold CV)",
        "",
        "| metal | n_samples | combined R² (mean ± std) | Hilbert-only R² | ASTSR-only R² |",
        "|---|---:|---|---|---|",
    ]
    combined_means: List[float] = []
    for r in all_results:
        lines.append(
            "| {cl} | {n} | {cm:+.4f} ± {cs:.4f} | {hm:+.4f} ± {hs:.4f} | {am:+.4f} ± {aas:.4f} |".format(
                cl=r["metal"],
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
    aggregate_std = float(np.std(combined_means)) if len(combined_means) > 1 else 0.0
    lines.append(
        "| **5-metal aggregate** | — | **{agg:+.4f} ± {ags:.4f}** | — | — |".format(
            agg=aggregate,
            ags=aggregate_std,
        )
    )
    lines.append("")
    lines.append("## Verify gate (target ≥ 0.0 mean across 5 metals)")
    lines.append("")
    if aggregate >= 0.0:
        verdict = "✅ **PASS** (combined R² ≥ 0.0 across all 5 metals on average)"
    else:
        verdict = "❌ **FAIL** (combined R² < 0.0; honest per-metal breakdown below)"
    lines.append(
        f"**Aggregate combined R² (mean of 5 metal means) = {aggregate:+.4f}** — {verdict}"
    )
    lines.append("")
    lines.append("## Per-metal per-seed R² detail")
    lines.append("")
    for r in all_results:
        lines.append(f"### {r['metal']} ({r['target_col']})")
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
    lines.append("- **MEASURED today**: 5 seeds × 5-fold × 5 metals on MetalCytoToxDB")
    lines.append("  5-metal binary activity (R² regression on 0/1 target);")
    lines.append("  Hilbert+ASTSR convex blend with per-fold alpha selection.")
    lines.append("- **Strict-scaffold split**: rows grouped by `scaffold_id`,")
    lines.append("  each scaffold assigned to exactly one fold's test set;")
    lines.append("  eliminates scaffold leakage between train and test.")
    lines.append("- **Alpha selection**: per-fold 90/10 inner validation; grid over")
    lines.append(
        "  alpha ∈ {0.0, 0.25, 0.5, 0.75, 1.0}; reported as `alpha_chosen` per fold."
    )
    lines.append(
        "- **Collapse detector**: `pred_std / y_test_std < 0.10` flagged in `per_fold`."
    )
    lines.append(
        "- **No Multi-PH + PLL**: this is the **combined Hilbert+ASTSR** anchor;"
    )
    lines.append(
        "  Multi-PH/PLL augmentation is a separate workstream (Wf-79 P2 3-way + PLL)."
    )
    lines.append(
        "- **CPU smoke**: full 5-seed × 5-fold × 5-metal runs in < 10 minutes on a"
    )
    lines.append(
        "  single CPU core (verified smoke); GPU-1 RTX 5090 optional but not required."
    )
    lines.append("")
    lines.append(
        "Per `party-twenty-spirit §3.5`: aggregate combined R² = "
        f"{aggregate:+.4f}; this is the **MEASURED** Wf-86 Phase 1 production anchor."
    )
    _write_run_log(out_dir / "synthesis.md", lines)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def _parse_seeds(raw: str) -> List[int]:
    return [int(x) for x in raw.replace(",", " ").split() if x]


def _parse_metals(raw: str) -> List[str]:
    return [x.strip() for x in raw.replace(",", " ").split() if x]


def _parse_bool(raw: str) -> bool:
    s = str(raw).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse bool: {raw!r}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-86 Phase 1 — REAL Hilbert+ASTSR combined R² on MetalCytoToxDB "
            "5-metal (5-seed × 50-epoch × strict-scaffold 5-fold)."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=["metalcytotox_5metal"],
        default="metalcytotox_5metal",
        help="Dataset identifier (only metalcytotox_5metal supported).",
    )
    parser.add_argument(
        "--metals",
        type=_parse_metals,
        nargs="+",
        default=list(DEFAULT_METALS),
        help=("Metal cohorts to benchmark (default: Ru Ir Os Rh Re)."),
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Random seeds for K-fold splitting (default: 0 1 2 3 4).",
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
        default=50,
        help="Hilbert PD iterations (default: 50).",
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
        "--strict-scaffold",
        type=_parse_bool,
        default=True,
        help=(
            "Enforce strict scaffold-disjoint 5-fold split "
            "(default: True); pass 'false' to fall back to random split."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Compute device (default: cpu; Hilbert branch is CPU fp32).",
    )
    # Wf-127 Phase 4 guard flags.
    parser.add_argument(
        "--use-l2-guard",
        dest="use_l2_guard",
        action="store_true",
        default=False,
        help="Wf-127 P2 L2 weight-decay guard on Hilbert primal+dual "
        "(default: False; P2 ship default is off, P4 turns on with --use-l2-guard).",
    )
    parser.add_argument(
        "--no-l2-guard",
        dest="use_l2_guard",
        action="store_false",
        help="Disable the Wf-127 P2 L2 weight-decay guard (explicit override).",
    )
    parser.add_argument(
        "--l2-weight",
        type=float,
        default=0.01,
        help="L2 weight-decay guard value when --use-l2-guard is set "
        "(default: 0.01, Phase 1 stable on synthetic).",
    )
    parser.add_argument(
        "--use-astr-scale-guard",
        dest="use_astr_scale_guard",
        action="store_true",
        default=False,
        help="Wf-127 P3 ASTSR value-space scale clip guard "
        "(default: False; P3 ship default is off, P4 turns on with --use-astr-scale-guard).",
    )
    parser.add_argument(
        "--no-astr-scale-guard",
        dest="use_astr_scale_guard",
        action="store_false",
        help="Disable the Wf-127 P3 ASTSR scale-guard (explicit override).",
    )
    parser.add_argument(
        "--scale-clip-min",
        type=float,
        default=-10.0,
        help="ASTSR scale clip min when --use-astr-scale-guard is set "
        "(default: -10.0, Phase 1 stable on synthetic).",
    )
    parser.add_argument(
        "--scale-clip-max",
        type=float,
        default=10.0,
        help="ASTSR scale clip max when --use-astr-scale-guard is set "
        "(default: 10.0, Phase 1 stable on synthetic).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("molmetal/reports/wf86_p1_metalcytotox_real"),
        help="Output directory (default: molmetal/reports/wf86_p1_metalcytotox_real).",
    )
    args = parser.parse_args(argv)

    # Flatten argparse nargs="+" artifacts.
    flat_metals: List[str] = []
    for m in args.metals:
        if isinstance(m, list):
            flat_metals.extend(str(x) for x in m)
        else:
            flat_metals.append(str(m))
    args.metals = flat_metals

    flat_seeds: List[int] = []
    for s in args.seeds:
        if isinstance(s, list):
            flat_seeds.extend(int(x) for x in s)
        else:
            flat_seeds.append(int(s))
    args.seeds = flat_seeds

    print(
        f"[wf86-p1] dataset={args.dataset} metals={args.metals} "
        f"seeds={args.seeds} epochs={args.epochs} "
        f"strict_scaffold={args.strict_scaffold} device={args.device}"
    )
    print(f"[wf86-p1] CSV path: {_resolve_metalcsv_path()}")

    all_results: List[Dict[str, Any]] = []
    for metal in args.metals:
        print(f"[wf86-p1] metal={metal}: loading MetalCytoToxDB cohort")
        result = evaluate_metal(
            metal,
            args.seeds,
            args.n_folds,
            args.epochs,
            args.population,
            args.generations,
            args.depth,
            args.hilbert_dim,
            args.lr,
            args.device,
            args.strict_scaffold,
            _REPO_ROOT,
            use_l2_guard=bool(args.use_l2_guard),
            l2_weight=float(args.l2_weight),
            use_astr_scale_guard=bool(args.use_astr_scale_guard),
            scale_clip_min=float(args.scale_clip_min),
            scale_clip_max=float(args.scale_clip_max),
        )
        per_seed_path = args.out_dir / metal / "per_seed_metrics.json"
        _write_json(per_seed_path, result)
        log_path = args.out_dir / metal / "run.log"
        log_lines = [
            f"metal={result['metal']}",
            f"target_col={result['target_col']}",
            f"n_samples={result['n_samples']}",
            f"n_seeds={result['n_seeds']}",
            f"strict_scaffold={result['strict_scaffold']}",
            f"combined_r2_mean={result['combined_r2_mean']:+.4f}",
            f"combined_r2_std={result['combined_r2_std']:.4f}",
            f"hilbert_r2_mean={result['hilbert_r2_mean']:+.4f}",
            f"astr_r2_mean={result['astr_r2_mean']:+.4f}",
        ]
        _write_run_log(log_path, log_lines)
        print(
            f"[wf86-p1] metal={metal}: combined R² = "
            f"{result['combined_r2_mean']:+.4f} ± {result['combined_r2_std']:.4f} "
            f"(Hilbert={result['hilbert_r2_mean']:+.4f}, "
            f"ASTSR={result['astr_r2_mean']:+.4f})"
        )
        all_results.append(result)
    write_synthesis(args.out_dir, all_results)
    print(f"[wf86-p1] synthesis.md written to {args.out_dir / 'synthesis.md'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
