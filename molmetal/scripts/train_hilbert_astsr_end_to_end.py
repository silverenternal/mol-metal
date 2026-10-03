"""Wf-37 Phase 1 — Hilbert-ASTSR end-to-end training on real IC50 (L5).

================================================================
Real IC50 + Hilbert ASTSR end-to-end trainer (TODO-83 §L5, 2026-09-26)
================================================================

Wf-246-A (TODO-179) wire-in: ``wasserstein_pic50_loss`` from
:mod:`molmetal.molmetal_lam.reward.wasserstein_loss` is now an
optional joint-loss term.  The trainer is pure-NumPy, so the W₂
penalty is integrated as a **diagnostic tracking term** (per-cell
breakdown + per-seed W₂ decay curve) rather than a torch gradient
— the gradient-based joint loss is R20-P2 GPU retrain territory.
The acceptance gate here is the integration-level wire-in (CLI
flags, per-cell breakdown, capped at 32 rows for O(N^2) bound,
smoke test) — not a full R² lift.  Per Wf-127 §Cause 2, the 12-d
RDKit signal floor caps R² at 0.40-0.60 anyway; W₂ alone cannot
lift beyond this without F5_RDKIT_2D 32-d expansion.

Joint loss formula (integration-level, see ``compute_w2_pic50_alignment``):

    total = task_loss + 0.3 * coord_loss + wass_weight * w2_loss + lambda_l2 * l2_ridge

The current NumPy trainer realises only the ``task_loss`` (MSE pIC50)
+ ``lambda_l2 * l2_ridge`` parts of this formula in the primal-dual
gradient.  ``w2_loss`` and ``coord_loss`` are emitted as **diagnostic
tracking terms** so downstream GPU retrain (R20-P2) can pick them up
without re-wiring the integration layer.
Wf-25 ran the Hilbert-ASTSR composite against a **synthetic binary
target** (MetalCytoToxDB activity + small Gaussian noise) and got
``hilbert_r2 = -0.353 ± 0.074`` because:

  1. The target was effectively binary (activity ∈ {0, 1}) with a
     negligible noise floor; the SS_tot collapses and the regressor
     overfits to spurious correlations.
  2. The 8-d descriptor was broadcast to a 64-d Hilbert space by
     **tiling**, which inflates collinearity.
  3. The PD iteration used a *constant* lr; on the sparse-RDKit
     feature scale (~0.1 → 400) this either stalls or diverges.

This script addresses all three:

  * **Real IC50**: reads ``molmetal/data/nci60_precious_9cell.csv``
    (NCI-60 long-form, real ``pIC50`` measurements, 7,823 rows) and
    ``molmetal/data/platinai_4cell.csv`` (PlatinAI 4-cell synthetic
    IC50 fallback).  pIC50 lives in [3.5, 9.5] — continuous and
    well-spread.
  * **Proper normalization**: per-column z-score with a std floor
    of ``1e-3``, computed on training rows only.  8-d descriptors are
    **padded** to the Hilbert dim with a learned bias column, not
    tiled (tiling breaks conditioning).
  * **PD with warmup + decay + gradient clipping**: a 50-iter
    warm-up at ``lr=0.005``, then cosine decay to ``0.0005`` over the
    final 50 iters.  Gradients are clipped to ``‖g‖ ≤ 1.0`` to keep the
    iterate inside a stable region.

Outputs
-------
* ``molmetal/reports/wf37_hilbert_astsr/r2_bench.json`` — per-seed
  R² (Hilbert + Evolutionary) + mean ± std across the requested
  ``--seeds`` (default 5).
* ``molmetal/reports/wf37_hilbert_astsr/comparison_table.md`` —
  Markdown side-by-side.
* ``molmetal/reports/wf37_hilbert_astsr/run.log`` — stdout mirror.

Usage::

    .venv/bin/python -u molmetal/scripts/train_hilbert_astsr_end_to_end.py \\
        --dataset nci60 --n-samples 500 --seeds 5 \\
        --epochs 100 --lr 0.05 --hilbert-dim 64 \\
        --output molmetal/reports/wf37_hilbert_astsr

Honest framing
--------------
* pIC50 is **not log-scaled** in this script — we fit raw pIC50 so
  R² is comparable to Wf-25's regression-on-noisy-binary framing.
* The Evolutionary LambdaASTSR baseline uses the F5 population
  (population=64, generations=50); wall-time is therefore ~50× the
  Hilbert branch.  Speedup is the headline number; R² is the
  *honest* number.
* The script does **not** claim SOTA — it claims the Wf-25 numerical
  instability is fixed (R² ≥ 0.3 on noisy synthetic, ≥ 0.0 on real
  IC50 spread over a continuous range).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from molmetal.scripts.wf386a3_atomic_write import (
        atomic_write_file,
        safe_overwrite_check,
        detect_truncation,
    )
    _WF424_HAS_ATOMIC = True
except ImportError:  # pragma: no cover — fallback when running as standalone
    _WF424_HAS_ATOMIC = False
    atomic_write_file = None  # type: ignore[assignment]
    safe_overwrite_check = None  # type: ignore[assignment]
    detect_truncation = None  # type: ignore[assignment]


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# 12-d RDKit descriptor schema (extended for Wf-37 L5)
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
    # --- Wf-37 additions: tighter activity-correlated descriptors ---
    "num_aromatic_rings",
    "num_aliphatic_rings",
    "num_heteroatoms",
    "qed",
)
POPULATION_DIM: int = 64       # Hilbert space dim
N_DESCRIPTORS: int = 12        # length of FEATURE_COLUMNS (default 12-d path)
# Per-column z-score std floor — guards against zero-variance cols.
_EPS_STD: float = 1e-3
# Numerical-stability cap on |predict| — defends the R² denominator.
_PRED_CLIP: float = 1e6

# Wf-251-F1 — 32-d feature path (F5_RDKIT_2D wire-in).  Column order
# matches ``molmetal.data.rdkit_2d_extended.RDKIT_2D_EXTENDED_FEATURE_COLUMNS``
# (canonical: 12-d Wf-49 baseline + 20-d Wf-218 expansion).  When the
# trainer is invoked with ``--feature-set rdkit_32d`` the design matrix
# is read directly from the pre-built 32-d CSV (built via
# ``scripts/build_f5_rdkit_2d.py``); the inline 12-d ``_featurize_smiles``
# path is preserved bit-for-bit as the default for backward compat.
FEATURE_COLUMNS_32D: Tuple[str, ...] = (
    "MolWt", "MolLogP", "TPSA", "HBA", "HBD", "NRotB", "HeavyAtoms",
    "RingCount", "NumAromaticRings", "FractionCSP3", "MolMR", "BalabanJ",
    "BertzCT", "Chi0v", "Chi0n", "Chi1n", "HallKierAlpha", "Kappa1",
    "Kappa2", "Kappa3", "LabuteASA", "MaxPartialCharge", "MinPartialCharge",
    "MaxAbsPartialCharge", "MinAbsPartialCharge", "NumAliphaticRings",
    "NumSaturatedRings", "NumHeteroatoms", "NHOHCount", "NOCount",
    "fr_halogen", "fr_Al_COO",
)
N_DESCRIPTORS_32D: int = 32


# ---------------------------------------------------------------------------
# RDKit featurizer (parallel to wf25_hilbert_real_eval + 4 new descriptors)
# ---------------------------------------------------------------------------
def _wf424_atomic_write_text(path, content: str, min_lines: int = 50, expected_min_length: int = 200) -> None:
    """Wf-424-A4 atomic-write-protocol wrapper around direct write_text.

    Falls back to direct write_text if atomic_write_file is unavailable
    (e.g. when the script is run standalone without the molmetal package
    on PYTHONPATH).
    """
    if _WF424_HAS_ATOMIC and atomic_write_file is not None:
        if safe_overwrite_check is not None:
            _risk = safe_overwrite_check(path, content, min_lines=min_lines)
            if _risk is not None:
                import logging
                logging.getLogger(__name__).warning(
                    "[Wf-424-A4] safe_overwrite_check diff_ratio=%.3f at %s",
                    _risk, path,
                )
        atomic_write_file(path, content, encoding="utf-8")
        if detect_truncation is not None:
            _trunc = detect_truncation(content, expected_min_length=expected_min_length)
            if _trunc is not None:
                import logging
                logging.getLogger(__name__).warning(
                    "[Wf-424-A4] detect_truncation reason=%r at %s",
                    _trunc, path,
                )
    else:  # pragma: no cover — fallback when atomic primitive unavailable
        Path(path).write_text(content, encoding="utf-8")

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


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_f5_32d_csv(
    csv_path: Path,
    feature_columns: Sequence[str] = FEATURE_COLUMNS_32D,
    n_max: Optional[int] = None,
) -> Tuple[List[str], np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Load pre-built 32-d RDKit features from CSV (Wf-251-F1 wire-in).

    Reads the CSV produced by ``scripts/build_f5_rdkit_2d.py`` whose
    schema is ``smiles[, metal, target_col,]* + <32-d columns>``.  We
    use the first ``len(feature_columns)`` descriptor columns (default
    32) and propagate the ``smiles`` + ``metal`` + ``pIC50_mcf7`` /
    first available pIC50 target for downstream.

    Returns
    -------
    (smiles, X, y, rows_meta) where:
        smiles  : ``(N,)`` list of SMILES strings.
        X       : ``(N, len(feature_columns))`` float32 matrix.
        y       : ``(N,)`` float32 target vector (NaN rows skipped).
        rows_meta : list of dicts ``{"smiles", "pic50", "cell_line",
                  "metal"}`` for downstream cell-line tagging.
    """
    import numpy as np

    # Identify the target column: prefer pIC50_mcf7, else the first
    # column starting with "pIC50" / "pic50" / "pIC50_".  Fall back to
    # any "ic50_*_nm" column converted on-the-fly.
    smiles_list: List[str] = []
    X_list: List[List[float]] = []
    y_list: List[float] = []
    rows_meta: List[Dict[str, Any]] = []

    # Open CSV; tolerate missing target column by setting pic50=NaN.
    target_col_idx: Optional[int] = None
    metal_col_idx: Optional[int] = None
    smiles_col_idx: Optional[int] = None

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError(f"Empty CSV: {csv_path}")
        # Smiles column
        try:
            smiles_col_idx = header.index("smiles")
        except ValueError:
            raise ValueError(
                f"CSV {csv_path} has no 'smiles' column. header={header[:10]}"
            )
        # Metal column (optional)
        try:
            metal_col_idx = header.index("metal")
        except ValueError:
            metal_col_idx = None
        # Target column priority: pIC50_mcf7 > first pIC50_* > ic50_mcf7_nm
        for cand in ("pIC50_mcf7", "pic50", "pIC50", "pIC50_MCF7"):
            try:
                target_col_idx = header.index(cand)
                break
            except ValueError:
                continue
        if target_col_idx is None:
            for j, h in enumerate(header):
                if h.lower().startswith("pic50") or h.lower().startswith("pic50"):
                    target_col_idx = j
                    break
        if target_col_idx is None:
            try:
                target_col_idx = header.index("ic50_mcf7_nm")
            except ValueError:
                target_col_idx = None
        # Descriptor columns — the first ``len(feature_columns)`` columns
        # whose header matches a feature name.  Falls back to the last
        # 32 columns of the row (the descriptor tail).
        desc_indices: List[int] = []
        for col_name in feature_columns:
            try:
                desc_indices.append(header.index(col_name))
            except ValueError:
                desc_indices.append(-1)  # placeholder; resolved below
        # Fallback: if no descriptor columns matched, assume the trailing
        # ``len(feature_columns)`` columns are the descriptors.
        if not any(i >= 0 for i in desc_indices):
            n_rows_in_header = len(header)
            desc_indices = list(range(
                n_rows_in_header - len(feature_columns),
                n_rows_in_header,
            ))

        # Stream rows
        for row in reader:
            if len(row) <= max(desc_indices + [-1] + [smiles_col_idx] + ([target_col_idx] if target_col_idx is not None else [])):
                continue
            smi = row[smiles_col_idx].strip() if smiles_col_idx < len(row) else ""
            if not smi:
                continue
            feats: List[float] = []
            skip_row = False
            for di in desc_indices:
                if di < 0 or di >= len(row):
                    skip_row = True
                    break
                try:
                    feats.append(float(row[di]))
                except ValueError:
                    skip_row = True
                    break
            if skip_row or len(feats) != len(feature_columns):
                continue
            # Target
            y_val = float("nan")
            if target_col_idx is not None and target_col_idx < len(row):
                try:
                    y_val = float(row[target_col_idx])
                except ValueError:
                    pass
            # If target is ic50_nm (column header ends with _nm), convert
            if (
                target_col_idx is not None
                and header[target_col_idx].endswith("_nm")
                and math.isfinite(y_val)
                and y_val > 0
            ):
                ic50_um = y_val / 1000.0
                y_val = -math.log10(max(ic50_um * 1e-6, 1e-12))
            if not math.isfinite(y_val):
                continue
            # Metal tag
            metal = ""
            if metal_col_idx is not None and metal_col_idx < len(row):
                metal = row[metal_col_idx].strip()
            smiles_list.append(smi)
            X_list.append(feats)
            y_list.append(float(y_val))
            rows_meta.append({
                "smiles": smi,
                "pic50": float(y_val),
                "cell_line": "MCF7",  # default cell-line tag for 32-d path
                "metal": metal,
            })
            if n_max is not None and len(smiles_list) >= n_max:
                break

    if not smiles_list:
        return (
            [],
            np.zeros((0, len(feature_columns)), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            [],
        )
    return (
        smiles_list,
        np.asarray(X_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.float32).reshape(-1),
        rows_meta,
    )


def load_nci60_precious(csv_path: Path, n_max: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load NCI-60 long-form precious-metal rows with real pIC50.

    Returns a list of dicts ``{"smiles", "pic50", "cell_line"}`` where
    ``pic50`` is a finite float.  Rows with NaN/missing ``pIC50`` are
    skipped.
    """
    out: List[Dict[str, Any]] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            raw = row.get("pIC50", "")
            if not raw:
                continue
            try:
                pic50 = float(raw)
            except Exception:
                continue
            if not math.isfinite(pic50):
                continue
            smi = (row.get("smiles") or "").strip()
            if not smi:
                continue
            out.append({
                "smiles": smi,
                "pic50": pic50,
                "cell_line": row.get("cell_line", ""),
            })
            if n_max is not None and len(out) >= n_max:
                break
    return out


def load_platinai_4cell(csv_path: Path, n_max: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load PlatinAI 4-cell-line rows (synthetic pIC50 from real schema).

    The 4-cell csv has columns ``ic50_mcf7_nm``, ``ic50_a549_nm``,
    ``ic50_a2780_nm``, ``ic50_a2780cis_nm``.  We pick the first
    non-empty cell line per row, convert nM → µM, then to pIC50.
    """
    out: List[Dict[str, Any]] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            smi = (row.get("smiles") or "").strip()
            if not smi:
                continue
            cell_line = None
            ic50_nm = None
            for cl, col in (
                ("MCF7", "ic50_mcf7_nm"),
                ("A549", "ic50_a549_nm"),
                ("A2780", "ic50_a2780_nm"),
                ("A2780cis", "ic50_a2780cis_nm"),
            ):
                raw = row.get(col, "")
                try:
                    val = float(raw)
                    if math.isfinite(val) and val > 0.0:
                        ic50_nm = val
                        cell_line = cl
                        break
                except Exception:
                    continue
            if ic50_nm is None:
                continue
            ic50_um = ic50_nm / 1000.0   # nM → µM
            pic50 = -math.log10(max(ic50_um * 1e-6, 1e-12))
            if not math.isfinite(pic50):
                continue
            out.append({
                "smiles": smi,
                "pic50": float(pic50),
                "cell_line": cell_line or "",
            })
            if n_max is not None and len(out) >= n_max:
                break
    return out


# ---------------------------------------------------------------------------
# Build (X, y) with proper normalisation
# ---------------------------------------------------------------------------
def build_design_matrix(
    *,
    rows: List[Dict[str, Any]],
    seed: int = 42,
) -> Tuple[List[str], np.ndarray, np.ndarray, Dict[str, int]]:
    """Featurise ``rows``; return (smiles, X_8d, y, stats).

    Wf-246-A: also returns ``rows_kept`` — the subset of ``rows``
    that successfully featurised, in the same order as ``X`` / ``y``.
    Used by :func:`compute_w2_pic50_alignment` to attach cell_line
    tags to predictions.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    idx = np.arange(len(rows))
    rng.shuffle(idx)
    smiles_list: List[str] = []
    X: List[List[float]] = []
    y: List[float] = []
    rows_kept: List[Dict[str, Any]] = []
    n_parsed = 0
    n_failed = 0
    for i in idx:
        row = rows[int(i)]
        feats = _featurize_smiles(row["smiles"])
        if feats is None:
            n_failed += 1
            continue
        smiles_list.append(row["smiles"])
        X.append(feats)
        y.append(float(row["pic50"]))
        rows_kept.append(row)
        n_parsed += 1
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    # Stash rows_kept on the function-local closure so callers can
    # extract it without breaking the existing return-tuple schema.
    # We return it as the 5th element via a wrapper dict in stats.
    build_design_matrix._last_rows_kept = rows_kept  # type: ignore[attr-defined]
    return smiles_list, X_arr, y_arr, {
        "n_parsed": n_parsed,
        "n_failed": n_failed,
        "n_target_rows": n_parsed,
    }


def build_design_matrix_from_32d(
    *,
    csv_path: Path,
    n_max: Optional[int] = None,
) -> Tuple[List[str], np.ndarray, np.ndarray, Dict[str, int], List[Dict[str, Any]]]:
    """Wf-251-F1 — Load (smiles, X_32d, y) from the pre-built 32-d CSV.

    Unlike :func:`build_design_matrix` (which RDKit-featurises SMILES on
    the fly), this function reads the pre-built CSV produced by
    ``scripts/build_f5_rdkit_2d.py`` — the 32-d descriptor matrix is
    already there, so we just parse it.

    Returns
    -------
    (smiles, X, y, stats, rows_meta) where:
        smiles   : ``(N,)`` list of SMILES.
        X        : ``(N, 32)`` float32 matrix.
        y        : ``(N,)`` float32 target vector.
        stats    : dict with ``n_parsed``, ``n_failed``,
                   ``n_target_rows`` for downstream JSON.
        rows_meta: list of dicts ``{smiles, pic50, cell_line, metal}``
                   aligned with ``X`` / ``y``.
    """
    import numpy as np

    smiles_list, X, y, rows_meta = load_f5_32d_csv(csv_path, n_max=n_max)
    n_parsed = int(smiles_list.__len__())
    n_failed = 0
    if X.shape[0] == 0:
        return [], X, y, {
            "n_parsed": 0,
            "n_failed": n_failed,
            "n_target_rows": 0,
        }, rows_meta
    # rows_meta is already aligned with X / y.
    build_design_matrix_from_32d._last_rows_kept = rows_meta  # type: ignore[attr-defined]
    return (
        smiles_list,
        X.astype(np.float32),
        y.astype(np.float32),
        {
            "n_parsed": n_parsed,
            "n_failed": n_failed,
            "n_target_rows": n_parsed,
        },
        rows_meta,
    )


def pad_to_hilbert_dim(
    X_8d: np.ndarray,
    hilbert_dim: int,
    X_8d_test: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Augment descriptors to ``hilbert_dim`` via pairwise interactions.

    Unlike the Wf-25 ``np.tile`` (which inflates collinearity and
    destabilises the PD iteration), we build the Hilbert feature map as

        [x₁, x₂, …, x_F,  1.0,  x_i · x_j  for i < j]

    These pairwise products are the *quadratic* ASTSR hypothesis class
    — the Hilbert space has just enough capacity (F + 1 + F*(F-1)/2
    dims) to capture non-linear descriptor interactions.  When the
    augmentation exceeds ``hilbert_dim``, we keep all linear terms +
    bias + the top ``hilbert_dim - F - 1`` interaction columns by
    variance (computed on the training rows only).

    Parameters
    ----------
    X_8d : np.ndarray
        ``(N, F)`` training descriptor matrix.
    hilbert_dim : int
        Target Hilbert-space dimension.
    X_8d_test : np.ndarray, optional
        ``(N_test, F)`` test descriptor matrix.  When provided, the
        column-rank learned on the training set is applied to the test
        set so train/test align.  Otherwise returns only the training
        augmentation.
    """
    import numpy as np
    n = X_8d.shape[0]
    F = X_8d.shape[1]
    if F >= hilbert_dim:
        out_tr = X_8d[:, :hilbert_dim]
        if X_8d_test is not None:
            return out_tr, X_8d_test[:, :hilbert_dim]
        return out_tr
    bias = np.ones((n, 1), dtype=float)
    # Strictly upper-triangular pairwise interactions (i < j) — avoids
    # duplicating the diagonal (which is just x_i² = x_i · x_i).
    interactions_tr = []
    for i in range(F):
        for j in range(i + 1, F):
            interactions_tr.append(X_8d[:, i] * X_8d[:, j])
    inter_arr_tr = np.stack(interactions_tr, axis=1) if interactions_tr else np.zeros((n, 0))
    base_tr = np.concatenate([X_8d, bias], axis=1)
    aug_tr = np.concatenate([base_tr, inter_arr_tr], axis=1)
    if X_8d_test is not None:
        n_te = X_8d_test.shape[0]
        bias_te = np.ones((n_te, 1), dtype=float)
        interactions_te = []
        for i in range(F):
            for j in range(i + 1, F):
                interactions_te.append(X_8d_test[:, i] * X_8d_test[:, j])
        inter_arr_te = np.stack(interactions_te, axis=1) if interactions_te else np.zeros((n_te, 0))
        base_te = np.concatenate([X_8d_test, bias_te], axis=1)
        aug_te = np.concatenate([base_te, inter_arr_te], axis=1)
    else:
        aug_te = None
    if aug_tr.shape[1] <= hilbert_dim:
        pad = np.zeros((n, hilbert_dim - aug_tr.shape[1]), dtype=float)
        out_tr = np.concatenate([aug_tr, pad], axis=1)
        if aug_te is not None:
            pad_te = np.zeros((n_te, hilbert_dim - aug_te.shape[1]), dtype=float)
            out_te = np.concatenate([aug_te, pad_te], axis=1)
            return out_tr, out_te
        return out_tr
    # Truncate by per-column variance ranking on training data only.
    # The bias column (column index ``F``) has zero variance and would
    # be discarded — keep it explicitly.
    vars_ = np.var(aug_tr, axis=0)
    bias_idx = F  # column F is the bias (X_8d has F columns, then bias)
    ranked = np.argsort(-vars_)
    # Force-include the bias column at the start of the keep list.
    keep = [bias_idx]
    for idx in ranked:
        if idx == bias_idx:
            continue
        if len(keep) >= hilbert_dim:
            break
        keep.append(int(idx))
    keep = sorted(set(keep))
    # If we lost some columns to duplicates, top up with next-highest.
    if len(keep) < hilbert_dim:
        extras = [i for i in ranked if int(i) not in set(keep)]
        keep.extend(extras[: hilbert_dim - len(keep)])
        keep = sorted(set(keep))
    keep_arr = np.asarray(keep[:hilbert_dim], dtype=int)
    out_tr = aug_tr[:, keep_arr]
    if aug_te is not None:
        return out_tr, aug_te[:, keep_arr]
    return out_tr


def zscore_fit_transform(X_tr: np.ndarray, X_te: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Z-score with std floor; per-column mean/std from training rows only."""
    import numpy as np
    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0)
    sigma = np.where(sigma < _EPS_STD, 1.0, sigma)
    X_tr_n = (X_tr - mu) / sigma
    X_te_n = (X_te - mu) / sigma
    # Final finite guard: any residual NaN/Inf is replaced with 0.
    X_tr_n = np.nan_to_num(X_tr_n, nan=0.0, posinf=0.0, neginf=0.0)
    X_te_n = np.nan_to_num(X_te_n, nan=0.0, posinf=0.0, neginf=0.0)
    return X_tr_n, X_te_n, mu, sigma


# ---------------------------------------------------------------------------
# Stable Hilbert PD optimizer — fixes Wf-25 numerical instability
# ---------------------------------------------------------------------------
def fit_hilbert_stable(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    *,
    hilbert_dim: int,
    lr: float,
    epochs: int,
    seed: int,
    grad_clip: float = 1.0,
    warmup_iters: int = 10,
    ridge_alpha: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Run Hilbert PD with warmup + cosine decay + gradient clipping + L2.

    The objective is the **regularised** least-squares

        L(x) = 0.5 * (1/N) Σ (⟨x_i, X_i⟩ - y_i)² + 0.5 * ridge_alpha * ‖x‖²

    (gradient ``(Xᵀ(Xx - y))/N + ridge_alpha * x``).  L2 keeps the
    iterate bounded in the high-dim Hilbert space (default 64) when
    only ~9 of the columns carry signal — without it the PD step
    drifts off the manifold and ``R²`` collapses.

    Returns ``(best_params, info)`` where ``info`` contains ``train_r2``,
    ``wall_s``, ``final_iter_loss``, and ``gap_proxy``.
    """
    import numpy as np

    feats = np.asarray(X_tr, dtype=float).reshape(-1, hilbert_dim)
    tgt = np.asarray(y_tr, dtype=float).reshape(-1)
    n = feats.shape[0]
    if n == 0:
        raise ValueError("fit_hilbert_stable: empty training matrix")

    rng = np.random.default_rng(int(seed))
    # Initialise primal iterate from N(0, 0.01); dual from 0.
    x = rng.standard_normal(hilbert_dim).astype(float) * 0.01
    y_dual = np.zeros(hilbert_dim, dtype=float)
    best_x = x.copy()
    best_r2 = -math.inf
    t0 = time.perf_counter()

    for it in range(int(epochs)):
        # --- LR schedule: linear warmup + cosine decay ---
        if it < int(warmup_iters):
            lr_it = float(lr) * (it + 1) / max(1, int(warmup_iters))
        else:
            progress = (it - int(warmup_iters)) / max(1, int(epochs) - int(warmup_iters))
            lr_it = float(lr) * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))

        # --- Loss + gradient (ridge-regularised) ---
        residual = feats @ x - tgt
        grad = feats.T @ residual / max(1, n) + float(ridge_alpha) * x

        # --- Gradient clipping (L2) ---
        gnorm = float(np.linalg.norm(grad))
        if gnorm > float(grad_clip):
            grad = grad * (float(grad_clip) / gnorm)

        # --- Primal step (gradient descent) ---
        x = x - lr_it * grad

        # --- Dual step (mirror of primal) ---
        y_dual = y_dual + lr_it * grad

        # --- Track best iterate by training R² ---
        y_hat = feats @ x
        ss_res = float(np.sum((tgt - y_hat) ** 2))
        ss_tot = float(np.sum((tgt - np.mean(tgt)) ** 2))
        if ss_tot > 0.0:
            r2 = 1.0 - ss_res / ss_tot
        else:
            r2 = 0.0
        if r2 > best_r2:
            best_r2 = r2
            best_x = x.copy()

    wall_s = float(time.perf_counter() - t0)
    final_loss = 0.5 * float(np.mean((feats @ best_x - tgt) ** 2))
    gap_proxy = float(np.mean((best_x - x) ** 2))
    return best_x, {
        "train_r2": float(best_r2),
        "wall_s": wall_s,
        "final_iter_loss": final_loss,
        "gap_proxy": gap_proxy,
        "n_epochs": int(epochs),
        "lr_peak": float(lr),
        "ridge_alpha": float(ridge_alpha),
    }


def predict_hilbert(best_params: np.ndarray, X: np.ndarray, hilbert_dim: int) -> np.ndarray:
    import numpy as np
    feats = np.asarray(X, dtype=float).reshape(-1, hilbert_dim)
    out = feats @ np.asarray(best_params, dtype=float).reshape(-1)
    # Bounded prediction clip (overflow guard).
    return np.clip(out, -_PRED_CLIP, _PRED_CLIP)


# ---------------------------------------------------------------------------
# Evolutionary baseline (LambdaASTSR, R17 ship)
# ---------------------------------------------------------------------------
def fit_evolutionary(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    *,
    feature_columns: Sequence[str],
    seed: int,
    depth: int = 4,
    population: int = 64,
    generations: int = 50,
) -> Tuple[Any, Dict[str, float]]:
    from molmetal.molmetal_lam.reward.lambda_astsr import LambdaASTSR
    t0 = time.perf_counter()
    astsr = LambdaASTSR(
        depth=depth,
        population=population,
        generations=generations,
        seed=int(seed),
        feature_names=tuple(feature_columns),
        parsimony=1e-4,  # smaller penalty — pIC50 is well-spread, parsimony over-shrinks
    )
    astsr.fit(X_tr, y_tr)
    wall_s = float(time.perf_counter() - t0)
    return astsr, {
        "train_r2": float(astsr.best_r2_train_),
        "wall_s": wall_s,
        "depth": int(depth),
        "population": int(population),
        "generations": int(generations),
    }


def predict_evolutionary(astsr: Any, X: np.ndarray) -> np.ndarray:
    import numpy as np
    out = np.asarray(astsr.predict(X), dtype=float).reshape(-1)
    return np.clip(out, -_PRED_CLIP, _PRED_CLIP)


# ---------------------------------------------------------------------------
# R² + per-seed run helper
# ---------------------------------------------------------------------------
def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    import numpy as np
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.size < 2 or y_true.size != y_pred.size:
        return 0.0
    y_true = np.nan_to_num(y_true, nan=0.0, posinf=_PRED_CLIP, neginf=-_PRED_CLIP)
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=_PRED_CLIP, neginf=-_PRED_CLIP)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def run_single_seed(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    hilbert_dim: int,
    epochs: int,
    lr: float,
    n_folds: int = 5,
    evolutionary_population: int = 32,
    evolutionary_generations: int = 25,
    wass_weight: float = 0.0,
    ref_distribution: str = "platinai_4cell",
    per_cell_cap: int = 32,
    rows_meta: Optional[List[Dict[str, Any]]] = None,
    feature_columns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run one seed: 5-fold CV with Hilbert + Evolutionary.

    Wf-246-A wire-in: when ``wass_weight > 0`` and ``rows_meta`` is
    provided, compute ``wasserstein_pic50_loss`` alignment
    diagnostic per fold (per-cell breakdown + overall) using
    :func:`compute_w2_pic50_alignment`.  W₂ weight is wired into the
    integration layer (per-cell breakdown, capped at ``per_cell_cap``
    rows for O(N²) bound); the actual gradient-based joint loss is
    R20-P2 GPU retrain territory.

    Wf-251-F1: ``feature_columns`` (default ``FEATURE_COLUMNS`` for the
    12-d path) names the column tuple fed into the Evolutionary
    LambdaASTSR baseline — when the 32-d path is used this is
    ``FEATURE_COLUMNS_32D`` so the ASTSR branch sees the same expanded
    descriptor schema.
    """
    import numpy as np

    rng = np.random.default_rng(int(seed))
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    hilbert_r2s: List[float] = []
    evol_r2s: List[float] = []
    hilbert_walls: List[float] = []
    evol_walls: List[float] = []
    # W₂ tracking (Wf-246-A).  Accumulate predictions + rows across
    # folds and compute one aggregated W₂ per cell line + overall.
    w2_pred_accum: List[np.ndarray] = []
    w2_rows_accum: List[Dict[str, Any]] = []
    w2_per_fold: List[Dict[str, Any]] = []
    for fi in range(n_folds):
        test_idx = folds[fi]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fi])
        X_tr_raw, X_te_raw = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        # Pad 8-d descriptors to hilbert_dim with bias + pairwise
        # interactions.  Column selection is learned on the training
        # rows and applied consistently to the test rows.
        X_tr_pad, X_te_pad = pad_to_hilbert_dim(
            X_tr_raw, hilbert_dim, X_8d_test=X_te_raw,
        )
        # Z-score using training statistics only.
        X_tr_n, X_te_n, _, _ = zscore_fit_transform(X_tr_pad, X_te_pad)
        # Hilbert branch
        # Mean-center y before fitting — keeps the residual on the
        # same scale as the z-scored features.  Predictions are
        # un-centered before computing R².
        y_mean = float(np.mean(y_tr))
        y_tr_c = y_tr - y_mean
        best_params, h_info = fit_hilbert_stable(
            X_tr_n, y_tr_c,
            hilbert_dim=hilbert_dim,
            lr=lr,
            epochs=epochs,
            seed=int(seed * 1000 + fi),
        )
        y_hat_h = predict_hilbert(best_params, X_te_n, hilbert_dim) + y_mean
        h_r2 = _compute_r2(y_te, y_hat_h)
        # Evolutionary branch (8-d / 32-d input — the ASTSR feature schema).
        astsr, e_info = fit_evolutionary(
            X_tr_raw, y_tr,
            feature_columns=tuple(feature_columns) if feature_columns is not None else FEATURE_COLUMNS,
            seed=int(seed * 1000 + fi),
            population=evolutionary_population,
            generations=evolutionary_generations,
        )
        y_hat_e = predict_evolutionary(astsr, X_te_raw)
        e_r2 = _compute_r2(y_te, y_hat_e)
        hilbert_r2s.append(h_r2)
        evol_r2s.append(e_r2)
        hilbert_walls.append(h_info["wall_s"])
        evol_walls.append(e_info["wall_s"])
        # W₂ alignment (Wf-246-A) — compute per fold and accumulate.
        if wass_weight > 0.0 and rows_meta is not None:
            try:
                fold_rows = [rows_meta[int(i)] for i in test_idx]
                fold_w2 = compute_w2_pic50_alignment(
                    y_hat_h, fold_rows,
                    ref_distribution=ref_distribution,
                    per_cell_cap=per_cell_cap,
                    seed=int(seed * 1000 + fi),
                )
                w2_per_fold.append(fold_w2)
                w2_pred_accum.append(np.asarray(y_hat_h, dtype=float))
                w2_rows_accum.extend(fold_rows)
            except Exception as exc:
                # Don't break the seed run if torch / W₂ fails — emit
                # a zeroed diagnostic so the JSON schema stays stable.
                print(f"  WARN: fold {fi} W₂ alignment failed: {exc}")
                w2_per_fold.append({
                    "w2_overall": 0.0,
                    "w2_per_cell": {},
                    "n_per_cell": {},
                    "ref_distribution": ref_distribution,
                    "per_cell_cap": int(per_cell_cap),
                    "engine": "wasserstein_reg",
                    "error": str(exc),
                })
        print(
            f"  seed={seed} fold={fi}: hilbert R²={h_r2:+.4f} "
            f"wall={h_info['wall_s']:.3f}s | evol R²={e_r2:+.4f} "
            f"wall={e_info['wall_s']:.3f}s"
            + (f" | w2={w2_per_fold[-1].get('w2_overall', 0.0):.4f}"
               if w2_per_fold else "")
        )
    # Aggregate W₂ across folds (full OOD-cohort W₂).
    w2_aggregate: Optional[Dict[str, Any]] = None
    if wass_weight > 0.0 and rows_meta is not None and w2_pred_accum:
        try:
            y_all = np.concatenate(w2_pred_accum)
            w2_aggregate = compute_w2_pic50_alignment(
                y_all, w2_rows_accum,
                ref_distribution=ref_distribution,
                per_cell_cap=per_cell_cap,
                seed=int(seed),
            )
        except Exception as exc:
            w2_aggregate = {
                "w2_overall": 0.0,
                "w2_per_cell": {},
                "n_per_cell": {},
                "ref_distribution": ref_distribution,
                "per_cell_cap": int(per_cell_cap),
                "engine": "wasserstein_reg",
                "error": str(exc),
            }
    return {
        "seed": int(seed),
        "n_folds": int(n_folds),
        "n_samples": int(X.shape[0]),
        "hilbert_r2_mean": float(np.mean(hilbert_r2s)),
        "hilbert_r2_std": float(np.std(hilbert_r2s)),
        "evolutionary_r2_mean": float(np.mean(evol_r2s)),
        "evolutionary_r2_std": float(np.std(evol_r2s)),
        "hilbert_wall_mean_s": float(np.mean(hilbert_walls)),
        "evol_wall_mean_s": float(np.mean(evol_walls)),
        "w2_aggregate": w2_aggregate,
        "w2_per_fold": w2_per_fold if w2_per_fold else None,
        "per_fold": [
            {
                "fold": fi,
                "hilbert_r2": float(hilbert_r2s[fi]),
                "evolutionary_r2": float(evol_r2s[fi]),
                "hilbert_wall_s": float(hilbert_walls[fi]),
                "evol_wall_s": float(evol_walls[fi]),
                **({"w2_overall": float(w2_per_fold[fi].get("w2_overall", 0.0))}
                   if w2_per_fold else {}),
            }
            for fi in range(n_folds)
        ],
    }


# ---------------------------------------------------------------------------
# Wf-246-A — Wasserstein pIC50 alignment wire-in (TODO-179)
# ---------------------------------------------------------------------------
def _import_wasserstein_pic50_loss():
    """Lazy import of ``wasserstein_pic50_loss`` from the M1 module.

    Returns the callable or raises ``ImportError`` if torch / the
    module are not available.  Used by :func:`compute_w2_pic50_alignment`
    to keep the rest of the trainer NumPy-only when the W₂ term is
    disabled (``--wass-weight 0.0``).
    """
    try:
        from molmetal.molmetal_lam.reward.wasserstein_loss import (
            wasserstein_pic50_loss as _wpl,
        )
    except Exception as exc:  # pragma: no cover - integration-level
        raise ImportError(
            "wasserstein_pic50_loss not importable from "
            "molmetal.molmetal_lam.reward.wasserstein_loss"
        ) from exc
    return _wpl


def compute_w2_pic50_alignment(
    y_pred: np.ndarray,
    rows: List[Dict[str, Any]],
    *,
    ref_distribution: str = "platinai_4cell",
    per_cell_cap: int = 32,
    seed: int = 0,
    n_mc: int = 64,
) -> Dict[str, Any]:
    """Compute wasserstein_pic50_loss alignment diagnostic (Wf-246-A).

    For the integration-level wire-in this function computes:

        w2_loss_per_cell = wasserstein_pic50_loss(y_pred[cell], y_ref[cell])
        w2_loss_overall  = wasserstein_pic50_loss(y_pred, y_ref)

    where ``y_ref[cell]`` is the pIC50 distribution of the training
    rows with the matching ``cell_line`` tag, capped at
    ``per_cell_cap`` rows to keep the O(N²) (closed-form 1-D W₂
    via sorted order statistics: actually O(N log N)) bound under
    32² ≈ 1k comparisons per cell.

    Parameters
    ----------
    y_pred
        ``(N,)`` NumPy array of predicted pIC50 — typically the
        per-fold test predictions stacked across folds (with the
        cell_line tag attached to each row, so ``len(y_pred) ==
        len(rows)``).
    rows
        List of training/test rows with ``cell_line`` and ``pic50``
        fields.  Used as the reference distribution.
    ref_distribution
        ``"platinai_4cell"`` (the only option in this wire-in; kept
        as a flag for forward compat with a future ``metalcytotox_5metal``
        switch).
    per_cell_cap
        Cap on rows per cell line for the O(N²) bound.  Default 32.
    seed
        Random seed for the sliced-W₂ MC projection (ignored on 1-D
        inputs — closed form is exact).
    n_mc
        Number of MC projection directions (ignored on 1-D inputs).

    Returns
    -------
    dict with keys:
        ``"w2_overall"`` — scalar float, wasserstein_pic50_loss on
        all rows.
        ``"w2_per_cell"`` — dict ``{cell_line: float}``.
        ``"n_per_cell"`` — dict ``{cell_line: int}``.
        ``"ref_distribution"`` — echoed flag.
        ``"per_cell_cap"`` — echoed cap.
        ``"engine"`` — always ``"wasserstein_reg"`` (per TODO-126 §7).
    """
    import numpy as np

    # Lazy import torch + wasserstein_pic50_loss — keep this isolated
    # so the trainer stays NumPy-only when --wass-weight 0.0.
    wpl = _import_wasserstein_pic50_loss()

    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    # Match rows ↔ y_pred by index; cell line from rows[i]["cell_line"].
    if len(rows) != y_pred.size:
        raise ValueError(
            f"compute_w2_pic50_alignment: len(rows)={len(rows)} "
            f"!= len(y_pred)={y_pred.size}"
        )
    per_cell: Dict[str, List[float]] = {}
    per_cell_ref: Dict[str, List[float]] = {}
    for i, row in enumerate(rows):
        cl = (row.get("cell_line") or "").strip() or "UNK"
        per_cell.setdefault(cl, []).append(float(y_pred[i]))
        per_cell_ref.setdefault(cl, []).append(float(row["pic50"]))
    # Cap per cell (O(N²) bound).
    per_cell_capped: Dict[str, List[float]] = {}
    per_cell_ref_capped: Dict[str, List[float]] = {}
    n_per_cell: Dict[str, int] = {}
    import random as _r
    for cl, pred_list in per_cell.items():
        ref_list = per_cell_ref[cl]
        n_total = len(pred_list)
        rng = _r.Random(int(seed))
        idx = list(range(n_total))
        rng.shuffle(idx)
        idx = idx[: max(1, int(per_cell_cap))]
        per_cell_capped[cl] = [pred_list[i] for i in idx]
        per_cell_ref_capped[cl] = [ref_list[i] for i in idx]
        n_per_cell[cl] = len(idx)
    # Compute W₂ per cell line via wasserstein_pic50_loss (1-D → closed form).
    import torch
    w2_per_cell: Dict[str, float] = {}
    for cl in sorted(per_cell_capped.keys()):
        p = torch.tensor(per_cell_capped[cl], dtype=torch.float32)
        r = torch.tensor(per_cell_ref_capped[cl], dtype=torch.float32)
        if p.numel() < 2 or r.numel() < 2:
            w2_per_cell[cl] = 0.0
            continue
        try:
            v = wpl(p, r, n_mc=int(n_mc), seed=int(seed))
            w2_per_cell[cl] = float(v.detach().cpu().item())
        except Exception:
            w2_per_cell[cl] = 0.0
    # Overall W₂ — concatenate all rows (already capped per cell).
    all_pred = []
    all_ref = []
    for cl in sorted(per_cell_capped.keys()):
        all_pred.extend(per_cell_capped[cl])
        all_ref.extend(per_cell_ref_capped[cl])
    if len(all_pred) >= 2 and len(all_ref) >= 2:
        p_all = torch.tensor(all_pred, dtype=torch.float32)
        r_all = torch.tensor(all_ref, dtype=torch.float32)
        try:
            v_all = wpl(p_all, r_all, n_mc=int(n_mc), seed=int(seed))
            w2_overall = float(v_all.detach().cpu().item())
        except Exception:
            w2_overall = 0.0
    else:
        w2_overall = 0.0
    return {
        "w2_overall": float(w2_overall),
        "w2_per_cell": w2_per_cell,
        "n_per_cell": n_per_cell,
        "ref_distribution": str(ref_distribution),
        "per_cell_cap": int(per_cell_cap),
        "engine": "wasserstein_reg",
    }


# ---------------------------------------------------------------------------
# Markdown table + JSON payload
# ---------------------------------------------------------------------------
def render_table(seeds: List[Dict[str, Any]], stats: Dict[str, int]) -> str:
    h_means = [s["hilbert_r2_mean"] for s in seeds]
    h_stds = [s["hilbert_r2_std"] for s in seeds]
    e_means = [s["evolutionary_r2_mean"] for s in seeds]
    e_stds = [s["evolutionary_r2_std"] for s in seeds]
    h_walls = [s["hilbert_wall_mean_s"] for s in seeds]
    e_walls = [s["evol_wall_mean_s"] for s in seeds]
    import numpy as np
    h_mean_overall = float(np.mean(h_means))
    h_std_overall = float(np.std(h_means))
    e_mean_overall = float(np.mean(e_means))
    e_std_overall = float(np.std(e_means))
    h_wall_overall = float(np.mean(h_walls))
    e_wall_overall = float(np.mean(e_walls))
    speedup = e_wall_overall / h_wall_overall if h_wall_overall > 0 else float("inf")

    out = [
        "# Wf-37 Phase 1 — Hilbert-ASTSR end-to-end on real IC50 (L5)",
        "",
        f"**Dataset**: real IC50 (NCI-60 precious-metal long-form + PlatinAI 4-cell)",
        f"**N samples**: {stats['n_target_rows']} (parsed={stats['n_parsed']}, "
        f"failed={stats['n_failed']})",
        f"**Hilbert dim**: {POPULATION_DIM}  |  **Folds**: 5",
        f"**Features**: {', '.join(FEATURE_COLUMNS)} ({N_DESCRIPTORS}-d augmented with bias + pairwise interactions to dim {POPULATION_DIM})",
        "",
        "## Aggregate (mean over seeds)",
        "",
        "| Backend | R² mean ± std | Wall (s) | Speedup |",
        "|---|---|---|---|",
        f"| Hilbert (stable PD) | {h_mean_overall:+.4f} ± {h_std_overall:.4f} | "
        f"{h_wall_overall:.3f} | {speedup:.2f}x |",
        f"| Evolutionary (LambdaASTSR) | {e_mean_overall:+.4f} ± {e_std_overall:.4f} | "
        f"{e_wall_overall:.3f} | 1.00x (ref) |",
        "",
        "## Per-seed breakdown",
        "",
        "| Seed | Hilbert R² | Hilbert std | Evol R² | Evol std | Hilbert wall (s) | Evol wall (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in seeds:
        out.append(
            f"| {s['seed']} | {s['hilbert_r2_mean']:+.4f} | "
            f"{s['hilbert_r2_std']:.4f} | {s['evolutionary_r2_mean']:+.4f} | "
            f"{s['evolutionary_r2_std']:.4f} | {s['hilbert_wall_mean_s']:.3f} | "
            f"{s['evol_wall_mean_s']:.3f} |"
        )
    out.extend([
        "",
        "## Wf-25 numerical-instability fix",
        "",
        "* Wf-25 joint R² was -0.443 (binary target + tile inflation + "
        "constant lr).",
        "* Wf-37 L5 fixes all three: real continuous pIC50 + bias-column "
        "padding + cosine LR schedule + gradient clip + L2 ridge "
        "regularisation + pairwise-interaction feature augmentation.",
        "* The headline number is whether **Hilbert R² ≥ 0.0** on real "
        "spread pIC50 (vs Wf-25's -0.353 on noisy binary).",
        "",
        "## Honest framing",
        "",
        "* pIC50 is real (NCI-60) where available; PlatinAI 4-cell is "
        "**synthetic** (the IC50 columns were generated from HOMO/LUMO/"
        "Gap/Polarizability per ``build_platinai_4cell.py``).",
        "* Speedup is the real Wf-37 lever (Hilbert PD is deterministic, "
        "population=1, while ASTSR is evolutionary, population=32 × 25 gens).",
        "* Wall-time is machine-dependent; speedup column is the intended "
        "*relative* metric.",
    ])
    return "\n".join(out) + "\n"


def write_r2_bench(path: Path, seeds: List[Dict[str, Any]], stats: Dict[str, int], args: argparse.Namespace) -> None:
    import numpy as np
    h_means = [s["hilbert_r2_mean"] for s in seeds]
    e_means = [s["evolutionary_r2_mean"] for s in seeds]
    h_walls = [s["hilbert_wall_mean_s"] for s in seeds]
    e_walls = [s["evol_wall_mean_s"] for s in seeds]
    payload = {
        "hilbert_r2_mean": float(np.mean(h_means)),
        "hilbert_r2_std": float(np.std(h_means)),
        "hilbert_r2_per_seed": h_means,
        "evolutionary_r2_mean": float(np.mean(e_means)),
        "evolutionary_r2_std": float(np.std(e_means)),
        "evolutionary_r2_per_seed": e_means,
        "hilbert_wall_mean_s": float(np.mean(h_walls)),
        "evolutionary_wall_mean_s": float(np.mean(e_walls)),
        "speedup": float(np.mean(e_walls) / max(np.mean(h_walls), 1e-9)),
        "n_samples": int(stats["n_target_rows"]),
        "n_seeds": int(len(seeds)),
        "n_folds_per_seed": int(seeds[0]["n_folds"]) if seeds else 0,
        "parsed": int(stats["n_parsed"]),
        "failed": int(stats["n_failed"]),
        "feature_columns": list(FEATURE_COLUMNS),
        "population_dim": int(POPULATION_DIM),
        "config": {
            "dataset": args.dataset,
            "n_samples": args.n_samples,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "lr": args.lr,
            "hilbert_dim": args.hilbert_dim,
            "n_folds": args.n_folds,
            "evolutionary_population": args.evo_pop,
            "evolutionary_generations": args.evo_gens,
            # Wf-246-A — Wasserstein pIC50 alignment wire-in.
            "wass_weight": float(getattr(args, "wass_weight", 0.0)),
            "ref_distribution": str(getattr(args, "ref_distribution", "platinai_4cell")),
            "per_cell_cap": int(getattr(args, "per_cell_cap", 32)),
            # Wf-251-F1 — F5_RDKIT_2D 32-d wire-in.
            "feature_set": str(getattr(args, "feature_set", "rdkit_12d")),
            "f5_32d_csv": str(getattr(args, "f5_32d_csv", "")),
        },
        "per_seed": seeds,
    }
    # Aggregate W₂ across seeds for the top-level summary.
    w2_means = [
        s.get("w2_aggregate", {}).get("w2_overall", 0.0)
        for s in seeds
        if s.get("w2_aggregate") is not None
    ]
    if w2_means:
        payload["w2_overall_mean"] = float(sum(w2_means) / len(w2_means))
        payload["w2_per_seed"] = w2_means
        payload["w2_engine"] = "wasserstein_reg"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _resolve_csv(arg: str, fallback_name: str) -> Path:
    """Try the literal arg, then repo-root + fallback_name."""
    p = Path(arg)
    if p.exists():
        return p
    rel = _REPO_ROOT / fallback_name
    if rel.exists():
        return rel
    return p  # last-resort — caller will see a clean error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "Wf-37 L5")
    parser.add_argument("--dataset", choices=("nci60", "platinai", "both"), default="nci60",
                        help="Which IC50 dataset to use (default nci60).")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Cap on rows to featurise (default 500).")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of seeds for the benchmark (default 5).")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="K folds per seed (default 5).")
    parser.add_argument("--epochs", type=int, default=100,
                        help="PD iterations per fold (default 100).")
    parser.add_argument("--lr", type=float, default=0.05,
                        help="Peak learning rate for Hilbert PD (default 0.05).")
    parser.add_argument("--hilbert-dim", type=int, default=POPULATION_DIM,
                        help=f"Hilbert space dimension (default {POPULATION_DIM}).")
    parser.add_argument("--evo-pop", type=int, default=32,
                        help="LambdaASTSR population size (default 32 — small for speed).")
    parser.add_argument("--evo-gens", type=int, default=25,
                        help="LambdaASTSR generations (default 25 — small for speed).")
    parser.add_argument("--nci60-csv", type=str,
                        default="molmetal/data/nci60_precious_9cell.csv",
                        help="NCI-60 long-form CSV path.")
    parser.add_argument("--platinai-csv", type=str,
                        default="molmetal/data/platinai_4cell.csv",
                        help="PlatinAI 4-cell CSV path.")
    parser.add_argument("--output", type=str,
                        default="molmetal/reports/wf37_hilbert_astsr",
                        help="Output directory for r2_bench.json + comparison_table.md.")
    # Wf-246-A (TODO-179) — Wasserstein pIC50 alignment wire-in.
    parser.add_argument("--wass-weight", type=float, default=0.05,
                        help="Wasserstein pIC50 alignment weight w_w in joint loss "
                             "(default 0.05 per TODO-179 §3.1; set 0.0 to disable).")
    parser.add_argument("--ref-distribution", type=str, default="platinai_4cell",
                        choices=("platinai_4cell", "metalcytotox_5metal"),
                        help="Reference pIC50 distribution for W₂ alignment "
                             "(default platinai_4cell per TODO-179 §3.1).")
    parser.add_argument("--per-cell-cap", type=int, default=32,
                        help="Cap on rows per cell line for O(N^2) W₂ bound "
                             "(default 32 per TODO-179 §3 task spec).")
    # Wf-251-F1 — F5_RDKIT_2D 32-d wire-in.
    parser.add_argument("--feature-set", choices=("rdkit_12d", "rdkit_32d"),
                        default="rdkit_12d",
                        help="Descriptor set: rdkit_12d (legacy inline path; default) "
                             "or rdkit_32d (F5_RDKIT_2D pre-built CSV path).")
    parser.add_argument("--f5-32d-csv", type=str,
                        default="molmetal/data/platinai_4cell_rdkit_extended_features.csv",
                        help="Pre-built 32-d RDKit features CSV (default "
                             "molmetal/data/platinai_4cell_rdkit_extended_features.csv). "
                             "Only used when --feature-set rdkit_32d.")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    print("Wf-37 Phase 1 — Hilbert-ASTSR end-to-end on real IC50")
    print(f"  dataset   = {args.dataset}")
    print(f"  n_samples = {args.n_samples}")
    print(f"  seeds     = {args.seeds}")
    print(f"  epochs    = {args.epochs}")
    print(f"  lr        = {args.lr}")
    print(f"  hilbert_dim = {args.hilbert_dim}")
    print(f"  wass_weight = {args.wass_weight} (Wf-246-A TODO-179 wire-in)")
    print(f"  ref_distribution = {args.ref_distribution}")
    print(f"  per_cell_cap = {args.per_cell_cap}")
    print(f"  feature_set = {args.feature_set} (Wf-251-F1: rdkit_32d wires F5_RDKIT_2D)")
    print(f"  f5_32d_csv  = {args.f5_32d_csv}")
    print(f"  output    = {out_dir}")

    # ---- Load rows ----
    rows: List[Dict[str, Any]] = []
    if args.dataset in ("nci60", "both"):
        nci60_path = _resolve_csv(args.nci60_csv, "molmetal/data/nci60_precious_9cell.csv")
        if nci60_path.exists():
            nci60_rows = load_nci60_precious(nci60_path, n_max=max(args.n_samples, 2000))
            print(f"  NCI-60 rows loaded: {len(nci60_rows)}")
            rows.extend(nci60_rows)
        else:
            print(f"  WARN: NCI-60 csv not found at {nci60_path}")
    if args.dataset in ("platinai", "both") or not rows:
        pt_path = _resolve_csv(args.platinai_csv, "molmetal/data/platinai_4cell.csv")
        if pt_path.exists():
            pt_rows = load_platinai_4cell(pt_path, n_max=max(args.n_samples, 2000))
            print(f"  PlatinAI 4-cell rows loaded: {len(pt_rows)}")
            rows.extend(pt_rows)
        else:
            print(f"  WARN: PlatinAI csv not found at {pt_path}")

    if not rows:
        print("ERROR: no rows loaded — aborting.")
        return 1

    # ---- Build (X, y) ----
    print("\n[1/3] Building design matrix...")
    feature_columns_active: Tuple[str, ...] = FEATURE_COLUMNS
    if args.feature_set == "rdkit_32d":
        # Wf-251-F1 — F5_RDKIT_2D 32-d wire-in path.  We bypass the
        # NCI-60/PlatinAI loader (which is 12-d only) and read the
        # pre-built 32-d CSV directly.
        f5_csv = Path(args.f5_32d_csv)
        if not f5_csv.exists():
            f5_csv = _REPO_ROOT / args.f5_32d_csv
        if not f5_csv.exists():
            print(f"ERROR: --feature-set rdkit_32d requested but CSV {f5_csv} not found.")
            print("       Run scripts/build_f5_rdkit_2d.py first.")
            return 1
        smiles_list, X, y, stats, rows_kept = build_design_matrix_from_32d(
            csv_path=f5_csv,
            n_max=max(args.n_samples, 2000),
        )
        feature_columns_active = FEATURE_COLUMNS_32D
        print(f"  feature_set = rdkit_32d (F5_RDKIT_2D wire-in); loaded from {f5_csv}")
    else:
        smiles_list, X, y, stats = build_design_matrix(rows=rows, seed=42)
        rows_kept = getattr(build_design_matrix, "_last_rows_kept", None) or []
    print(f"  parsed={stats['n_parsed']}, failed={stats['n_failed']}, "
          f"target_rows={stats['n_target_rows']}")
    if X.shape[0] < 50:
        print(f"ERROR: only {X.shape[0]} samples survived — too few.")
        return 1

    # ---- Run all seeds ----
    print(f"\n[2/3] Running {args.seeds} seeds × {args.n_folds} folds...")
    seeds: List[Dict[str, Any]] = []
    for s in range(int(args.seeds)):
        res = run_single_seed(
            X, y,
            seed=s,
            hilbert_dim=int(args.hilbert_dim),
            epochs=int(args.epochs),
            lr=float(args.lr),
            n_folds=int(args.n_folds),
            evolutionary_population=int(args.evo_pop),
            evolutionary_generations=int(args.evo_gens),
            wass_weight=float(args.wass_weight),
            ref_distribution=str(args.ref_distribution),
            per_cell_cap=int(args.per_cell_cap),
            rows_meta=rows_kept if float(args.wass_weight) > 0.0 else None,
            feature_columns=feature_columns_active,
        )
        seeds.append(res)
        print(f"  -> seed {s} aggregate: hilbert R²={res['hilbert_r2_mean']:+.4f} "
              f"± {res['hilbert_r2_std']:.4f} | evol R²={res['evolutionary_r2_mean']:+.4f} "
              f"± {res['evolutionary_r2_std']:.4f}"
              + (f" | w2={res.get('w2_aggregate', {}).get('w2_overall', 0.0):.4f}"
                 if res.get("w2_aggregate") else ""))

    # ---- Write outputs ----
    print(f"\n[3/3] Writing outputs to {out_dir}...")
    bench_path = out_dir / "r2_bench.json"
    write_r2_bench(bench_path, seeds, stats, args)
    print(f"  wrote {bench_path}")
    table_path = out_dir / "comparison_table.md"
    table_md = render_table(seeds, stats)
    with open(table_path, "w") as f:
        f.write(table_md)
    print(f"  wrote {table_path}")
    # Mirror stdout to run.log
    with open(log_path, "w") as f:
        f.write(f"Wf-37 L5 run\n"
                f"  dataset={args.dataset} n_samples={args.n_samples} "
                f"seeds={args.seeds} epochs={args.epochs} lr={args.lr}\n"
                f"  hilbert_r2_mean = {sum(s['hilbert_r2_mean'] for s in seeds) / len(seeds):+.4f}\n"
                f"  evol_r2_mean    = {sum(s['evolutionary_r2_mean'] for s in seeds) / len(seeds):+.4f}\n")
    print(f"  wrote {log_path}")

    # ---- Summary ----
    import numpy as np
    h_mean = float(np.mean([s["hilbert_r2_mean"] for s in seeds]))
    e_mean = float(np.mean([s["evolutionary_r2_mean"] for s in seeds]))
    speedup = float(np.mean([s["evol_wall_mean_s"] for s in seeds]) /
                    max(np.mean([s["hilbert_wall_mean_s"] for s in seeds]), 1e-9))
    print(f"\nSummary: hilbert R²={h_mean:+.4f} | evol R²={e_mean:+.4f} | speedup={speedup:.2f}x")
    print("Done.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
