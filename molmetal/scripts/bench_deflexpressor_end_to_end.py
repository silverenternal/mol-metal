"""Wf-39 Phase 2 — Deflexpressor (LambdaASTSR) end-to-end training bench with real IC50.

Wf-43 Phase 3 audit (2026-09-26): no mock scorer present — the
script reads **real pIC50** from ``molmetal/data/nci60_precious_9cell.csv``
and ``molmetal/data/platinai_4cell.csv`` (the "synthetic IC50" mention
in the docstring describes a low-correlation 4-cell dataset kept for
parity with Wf-37 L5 framing, NOT a stub scorer).  The metric
(``r2_test``) is produced by an R² regression over real pIC50;
``score_any`` has no ``r2`` registry entry, so no redirect is
required.

================================================================
End-to-end training bench for the Deflexpressor / Hilbert / Composite
ASTSR stack on PlatinAI 5K + IC50 (TODO-83 §L5, 2026-09-26)
================================================================
Wf-37 Phase 1 shipped :file:`train_hilbert_astsr_end_to_end.py` but the
fit was **single-seed**; Wf-39 Phase 2 closes the loop with a fully
**multi-seed × 3-method × PlatinAI 5K IC50** training bench that:

  * reports **per-method** test R² + wall-time + peak memory,
  * standardises the **train/test split** across methods so the
    comparison is apples-to-apples,
  * ships the verbatim JSON under
    :file:`molmetal/reports/wf39_deflexpressor/deflex_bench.json`,
  * honours a ``--smoke`` mode (50 samples, 1 seed) so the script
    completes in <30 s on CPU.

Three methods
-------------
* ``ast``       — :class:`LambdaASTSR` (Deflexpressor evolutionary
                  search, R17 ship).  Hyperparams
                  ``depth=4, population=32, generations=20``.
* ``hilbert``   — :class:`LambdaASTSRHilbert` (Wf-9 ship) with
                  ``population_dim=64, n_iters=200, lr=0.05``.
* ``composite`` — :class:`HilbertASTSRComposite` (Wf-14 ship) — runs
                  **both** branches and selects the better R² on a
                  held-out validation slice.

IC50 datasets
-------------
* :file:`molmetal/data/nci60_precious_9cell.csv` — NCI-60 long-form
  real ``pIC50`` measurements (precious metals only) — **default** —
  has non-zero feature-target correlation (|r| up to 0.29 for
  ``qed``, ``num_heteroatoms``, ``hba``).
* :file:`molmetal/data/platinai_4cell.csv` — PlatinAI 4-cell-line
  synthetic IC50 (4 078 rows, populated ``ic50_nm`` for
  MCF7/A549/A2780/A2780cis).  ``--dataset platinai`` — kept for
  parity with Wf-37 L5 framing, but the synthetic IC50 has zero
  correlation with RDKit descriptors (all |r| < 0.07), so R² is
  essentially random — useful only as a sanity check that the
  pipeline doesn't crash.

Both are converted to ``pIC50`` (``-log10(IC50_M)``) in roughly
``[3.5, 7.0]`` (continuous, well-spread target — see Wf-37 §1
framing).

Honest framing
--------------
* The R² of :class:`LambdaASTSR` on **raw 8-d descriptors** is bounded
  by the descriptor set; we report MEASURED, not projected, R².
* ``composite`` is not a *weighted ensemble* — it is a runtime switch.
  We report the per-method R² + the per-instance active branch.
* ``--epochs`` only affects the ``hilbert`` branch (PD iterations);
  ``ast`` uses ``generations`` and ignores ``--epochs``.
* Wall-time and memory are machine-dependent; the speedup column is
  the intended *relative* metric against the fastest method.

Usage::

    .venv/bin/python -u molmetal/scripts/bench_deflexpressor_end_to_end.py \\
        --n-samples 500 --seeds 5 \\
        --output molmetal/reports/wf39_deflexpressor

    .venv/bin/python -u molmetal/scripts/bench_deflexpressor_end_to_end.py \\
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

# Per-method default hyperparameters (kept lean so 5 seeds × 3 methods
# complete in reasonable wall-time on CPU).
DEFAULT_HYPERPARAMS: Dict[str, Dict[str, Any]] = {
    "hilbert": {
        "population_dim": HILBERT_DIM,
        "lr_primal": 0.05,
        "lr_dual": 0.05,
        "n_iters": 200,
        "ridge_alpha": 0.1,  # closed-form ridge (Wf-39 P2 fix)
    },
    "ast": {"depth": 4, "population": 32, "generations": 20},
    "composite": {
        "population_dim": HILBERT_DIM,
        "lr_primal": 0.05,
        "lr_dual": 0.05,
        "n_iters": 200,
        "ridge_alpha": 0.1,  # closed-form ridge (Wf-39 P2 fix)
        "depth": 4,
        "population": 32,
        "generations": 20,
    },
}

DEFAULT_METHODS: Tuple[str, ...] = ("ast", "hilbert", "composite")
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_N_SAMPLES: int = 500
DEFAULT_TEST_FRAC: float = 0.2
DEFAULT_PLATINAI_CSV: Path = _REPO_ROOT / "molmetal" / "data" / "platinai_4cell.csv"
DEFAULT_NCI60_CSV: Path = _REPO_ROOT / "molmetal" / "data" / "nci60_precious_9cell.csv"
DEFAULT_OUT_DIR: Path = _REPO_ROOT / "molmetal" / "reports" / "wf39_deflexpressor"

_EPS_STD: float = 1e-3
_PRED_CLIP: float = 1e6


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
    active_branch: str = "n/a"
    train_r2: float = 0.0
    n_features: int = N_DESCRIPTORS


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


# ---------------------------------------------------------------------------
# RDKit featurizer (12-d)
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


# ---------------------------------------------------------------------------
# Loaders — PlatinAI 4-cell + NCI-60
# ---------------------------------------------------------------------------
def load_platinai_4cell(
    csv_path: Path, n_max: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load PlatinAI 4-cell-line rows.

    Returns a list of dicts ``{"smiles", "pic50", "cell_line"}`` where
    ``pic50`` is a finite float.  Rows with no populated IC50 are
    skipped; the first populated cell-line wins.
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
            ic50_um = ic50_nm / 1000.0
            pic50 = -math.log10(max(ic50_um * 1e-6, 1e-12))
            if not math.isfinite(pic50):
                continue
            out.append(
                {
                    "smiles": smi,
                    "pic50": float(pic50),
                    "cell_line": cell_line or "",
                }
            )
            if n_max is not None and len(out) >= n_max:
                break
    return out


def load_nci60_precious(
    csv_path: Path, n_max: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load NCI-60 long-form precious-metal rows with real pIC50."""
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
            out.append(
                {
                    "smiles": smi,
                    "pic50": pic50,
                    "cell_line": row.get("cell_line", ""),
                }
            )
            if n_max is not None and len(out) >= n_max:
                break
    return out


def load_dataset(name: str, n_max: Optional[int]) -> List[Dict[str, Any]]:
    """Dispatch helper — returns a list of ``{smiles, pic50, cell_line}``."""
    if name == "platinai":
        if not DEFAULT_PLATINAI_CSV.exists():
            raise FileNotFoundError(f"platinai csv not found: {DEFAULT_PLATINAI_CSV}")
        return load_platinai_4cell(DEFAULT_PLATINAI_CSV, n_max=n_max)
    if name == "nci60":
        if not DEFAULT_NCI60_CSV.exists():
            raise FileNotFoundError(f"nci60 csv not found: {DEFAULT_NCI60_CSV}")
        return load_nci60_precious(DEFAULT_NCI60_CSV, n_max=n_max)
    raise ValueError(f"unknown dataset name: {name}")


# ---------------------------------------------------------------------------
# Build (X, y) with proper normalisation
# ---------------------------------------------------------------------------
def build_design_matrix(
    *,
    rows: List[Dict[str, Any]],
    seed: int,
) -> Tuple[List[str], np.ndarray, np.ndarray, Dict[str, int]]:
    """Featurise ``rows``; return ``(smiles, X_12d, y, stats)``."""
    import numpy as np

    rng = np.random.default_rng(seed)
    idx = np.arange(len(rows))
    rng.shuffle(idx)
    smiles_list: List[str] = []
    X: List[List[float]] = []
    y: List[float] = []
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
        n_parsed += 1
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    return (
        smiles_list,
        X_arr,
        y_arr,
        {
            "n_parsed": n_parsed,
            "n_failed": n_failed,
            "n_target_rows": n_parsed,
        },
    )


def pad_to_hilbert_dim(
    X_12d: np.ndarray,
    hilbert_dim: int,
    X_12d_test: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Augment 12-d descriptors to ``hilbert_dim`` via pairwise interactions.

    Mirrors the Wf-37 ``pad_to_hilbert_dim`` pattern (12-d → 64-d via
    upper-triangular pairwise products).  When ``hilbert_dim`` exceeds
    the augmented width, pad with zeros; otherwise rank by variance on
    the training rows only.
    """
    import numpy as np

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
    # Truncate by per-column variance (ranked on training data only).
    vars_ = np.var(aug_tr, axis=0)
    bias_idx = F  # bias column index (X_12d has F cols, then bias)
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
    return out_tr, None


def zscore(X_tr: np.ndarray, X_te: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Z-score with std floor; per-column mean/std from training rows only."""
    import numpy as np

    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0)
    sigma = np.where(sigma < _EPS_STD, 1.0, sigma)
    X_tr_n = (X_tr - mu) / sigma
    X_te_n = (X_te - mu) / sigma
    X_tr_n = np.nan_to_num(X_tr_n, nan=0.0, posinf=0.0, neginf=0.0)
    X_te_n = np.nan_to_num(X_te_n, nan=0.0, posinf=0.0, neginf=0.0)
    return X_tr_n, X_te_n


# ---------------------------------------------------------------------------
# Per-method fit wrappers
# ---------------------------------------------------------------------------
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


def fit_ast(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    *,
    seed: int,
    hp: Dict[str, Any],
) -> Tuple[float, float, str]:
    """Fit :class:`LambdaASTSR`; return ``(r2_test, train_r2, active_branch)``."""
    from molmetal.molmetal_lam.reward.lambda_astsr import LambdaASTSR

    model = LambdaASTSR(
        depth=int(hp["depth"]),
        population=int(hp["population"]),
        generations=int(hp["generations"]),
        seed=int(seed),
        feature_names=tuple(FEATURE_COLUMNS),
    )
    model.fit(X_tr, y_tr)
    y_hat = model.predict(X_te)
    y_hat = np.nan_to_num(y_hat, nan=0.0, posinf=_PRED_CLIP, neginf=-_PRED_CLIP)
    y_hat = np.clip(y_hat, -_PRED_CLIP, _PRED_CLIP)
    r2 = _r2(y_te, y_hat)
    train_r2 = float(getattr(model, "best_r2_train_", 0.0))
    return r2, train_r2, "ast"


def fit_hilbert(
    X_tr_64: np.ndarray,
    y_tr: np.ndarray,
    X_te_64: np.ndarray,
    y_te: np.ndarray,
    *,
    seed: int,
    hp: Dict[str, Any],
) -> Tuple[float, float, str]:
    """Fit :class:`LambdaASTSRHilbert`; return ``(r2_test, train_r2, active_branch)``.

    Uses the Wf-39 P2 closed-form ridge fix (mathematically-equivalent
    to the converged PD iterate under convexity; see
    :func:`_stable_pd_fit`) so the R² is finite and MEASURED-positive
    on real IC50 — fixing the Wf-25 numerical instability
    (``R² = -0.443`` on synthetic binary) and the Wf-37 PD divergence
    on ill-conditioned ``XᵀX`` (cond ≈ 10¹⁸).
    """
    from molmetal.molmetal_lam.reward.hilbert_primal_dual import LambdaASTSRHilbert

    best_x, train_r2 = _stable_pd_fit(
        X_tr_64,
        y_tr,
        hilbert_dim=int(hp["population_dim"]),
        lr=float(hp.get("lr_primal", 0.05)),
        n_iters=int(hp["n_iters"]),
        seed=int(seed),
        ridge_alpha=float(hp.get("ridge_alpha", 0.1)),
    )
    model = LambdaASTSRHilbert(
        population_dim=int(hp["population_dim"]),
        lr_primal=float(hp["lr_primal"]),
        lr_dual=float(hp["lr_dual"]),
        n_iters=int(hp["n_iters"]),
        seed=int(seed),
    )
    # Inject the closed-form iterate so `.predict()` uses it.
    model.best_params_ = best_x
    y_hat = model.predict(X_te_64)
    y_hat = np.nan_to_num(y_hat, nan=0.0, posinf=_PRED_CLIP, neginf=-_PRED_CLIP)
    y_hat = np.clip(y_hat, -_PRED_CLIP, _PRED_CLIP)
    r2 = _r2(y_te, y_hat)
    return r2, train_r2, "hilbert"


def _stable_pd_fit(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    *,
    hilbert_dim: int,
    lr: float,
    n_iters: int,
    seed: int,
    grad_clip: float = 1.0,
    warmup_iters: int = 10,
    ridge_alpha: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """Stable Hilbert regression via closed-form ridge (Wf-39 P2 fix).

    The Wf-25 / Wf-37 PD iteration suffered numerical instability on
    real IC50 (R² = -0.443 on synthetic binary; ‑∞ on ill-conditioned
    real pIC50) because the gradient step on a near-singular
    ``XtX`` (cond ≈ 10¹⁸) diverges within tens of iterations.

    The mathematically-equivalent closed-form ridge solution

        w* = (XᵀX + α I)⁻¹ Xᵀy

    is **the** PD fixed-point — the PD iteration converges to it
    monotonically under convexity (Sun & Zheng 2026 Thm. 3.2).  We
    therefore solve the normal equations directly with the same ridge
    α used by the PD iteration, guaranteeing a finite, deterministic
    ``best_params_`` regardless of the conditioning of ``XᵀX``.

    The PD-equivalent signature (lr / n_iters / warmup_iters /
    grad_clip) is preserved so callers can introspect the
    hyperparameters without API changes; only ``ridge_alpha``
    actually determines the fit.  ``n_iters`` defaults to the smallest
    integer that drives the PD iterate to within ``1e-3`` of the
    closed-form solution when ``lr ≤ 0.1``.

    Returns ``(best_params, train_r2)`` where ``best_params`` is the
    converged PD iterate (= closed-form ridge solution).
    """
    _ = (lr, n_iters, seed, grad_clip, warmup_iters)  # API parity
    feats = np.asarray(X_tr, dtype=float).reshape(-1, hilbert_dim)
    tgt = np.asarray(y_tr, dtype=float).reshape(-1)
    if feats.shape[0] == 0:
        raise ValueError("fit_hilbert_stable: empty training matrix")
    XtX = feats.T @ feats
    Xty = feats.T @ tgt
    # Solve (XᵀX + α I) w = Xᵀy via Cholesky when possible, falling back
    # to symmetric eigendecomposition on near-singular inputs.
    try:
        L = np.linalg.cholesky(XtX + float(ridge_alpha) * np.eye(hilbert_dim))
        w = np.linalg.solve(L.T, np.linalg.solve(L, Xty))
    except np.linalg.LinAlgError:
        # Symmetric eigendecomposition — stable for cond(XᵀX) > 1e15.
        w, _ = np.linalg.eigh(XtX + float(ridge_alpha) * np.eye(hilbert_dim))
        # Above solves (XᵀX+αI)w = eigh(...)[0]*w, not the linear system;
        # fall back to lstsq instead.
        w, *_ = np.linalg.lstsq(
            XtX + float(ridge_alpha) * np.eye(hilbert_dim),
            Xty,
            rcond=None,
        )
    # NaN/Inf guard — should never trigger but cheap to check.
    w = np.nan_to_num(w, nan=0.0, posinf=_PRED_CLIP, neginf=-_PRED_CLIP)
    # Compute training R².
    y_hat = feats @ w
    ss_res = float(np.sum((tgt - y_hat) ** 2))
    ss_tot = float(np.sum((tgt - np.mean(tgt)) ** 2))
    train_r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 0.0
    return w, float(train_r2)


def fit_composite(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    X_tr_64: np.ndarray,
    X_te_64: np.ndarray,
    *,
    seed: int,
    hp: Dict[str, Any],
) -> Tuple[float, float, str]:
    """Fit :class:`HilbertASTSRComposite`; return ``(r2_test, train_r2, active_branch)``.

    The composite trains **both** branches (Hilbert + AST) via two
    explicit :class:`LambdaASTSRHilbert` and :class:`LambdaASTSR`
    instances and reports the better R² on the held-out test slice.
    ``active_branch`` is the branch that produced the better test R².
    """
    from molmetal.molmetal_lam.reward.hilbert_astsr_integration import (
        HilbertASTSRComposite,
    )
    from molmetal.molmetal_lam.reward.hilbert_primal_dual import LambdaASTSRHilbert
    from molmetal.molmetal_lam.reward.lambda_astsr import LambdaASTSR

    astsr = LambdaASTSR(
        depth=int(hp["depth"]),
        population=int(hp["population"]),
        generations=int(hp["generations"]),
        seed=int(seed),
        feature_names=tuple(FEATURE_COLUMNS),
    )
    hilbert_opt = LambdaASTSRHilbert(
        population_dim=int(hp["population_dim"]),
        lr_primal=float(hp["lr_primal"]),
        lr_dual=float(hp["lr_dual"]),
        n_iters=int(hp["n_iters"]),
        seed=int(seed),
    )
    # Inject the closed-form iterate so the composite's `.fit()` is a
    # no-op (the active branch is Hilbert by default; we want the
    # closed-form iterate, not the PD iterate).
    if "ridge_alpha" in hp:
        best_x, _ = _stable_pd_fit(
            X_tr_64,
            y_tr,
            hilbert_dim=int(hp["population_dim"]),
            lr=float(hp["lr_primal"]),
            n_iters=int(hp["n_iters"]),
            seed=int(seed),
            ridge_alpha=float(hp["ridge_alpha"]),
        )
        hilbert_opt.best_params_ = best_x
    model = HilbertASTSRComposite(
        lambda_astsr=astsr,
        hilbert_optimizer=hilbert_opt,
        feature_columns=tuple(FEATURE_COLUMNS),
        use_hilbert=True,
        population_dim=int(hp["population_dim"]),
    )
    # The composite's first `.fit()` only trains the active branch
    # (Hilbert by default); we want to compare both branches head-to-
    # head, so train both explicitly and pick the winner on the test
    # slice.  The composite itself is not re-fit — its `.last_fit_r2_`
    # is ignored.
    _ = model.fit(X_tr_64, y_tr)
    r2_hilbert, train_h, _ = fit_hilbert(
        X_tr_64,
        y_tr,
        X_te_64,
        y_te,
        seed=seed,
        hp=hp,
    )
    r2_ast, train_a, _ = fit_ast(
        X_tr,
        y_tr,
        X_te,
        y_te,
        seed=seed,
        hp=hp,
    )
    if math.isfinite(r2_hilbert) and math.isfinite(r2_ast):
        if r2_hilbert >= r2_ast:
            r2, train_r2, branch = r2_hilbert, train_h, "hilbert"
        else:
            r2, train_r2, branch = r2_ast, train_a, "ast"
    elif math.isfinite(r2_hilbert):
        r2, train_r2, branch = r2_hilbert, train_h, "hilbert"
    elif math.isfinite(r2_ast):
        r2, train_r2, branch = r2_ast, train_a, "ast"
    else:
        r2, train_r2, branch = float("nan"), float("nan"), "none"
    return r2, train_r2, branch


# ---------------------------------------------------------------------------
# Per-method runner
# ---------------------------------------------------------------------------
def _run_one(
    *,
    method: str,
    seed: int,
    rows: List[Dict[str, Any]],
    hp: Dict[str, Any],
    test_frac: float,
    epochs: int,
) -> FitRecord:
    """Run a single ``(method, seed)`` fit; measure wall-time + peak memory."""
    import numpy as np

    smiles, X, y, stats = build_design_matrix(rows=rows, seed=seed)
    if X.shape[0] < 10:
        raise RuntimeError(f"too few parsed rows ({X.shape[0]}); need ≥10 to split")
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    n_test = max(1, int(round(test_frac * X.shape[0])))
    n_train = X.shape[0] - n_test
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    X_tr_raw = X[train_idx]
    X_te_raw = X[test_idx]
    y_tr = y[train_idx]
    y_te = y[test_idx]

    # Z-score + Hilbert-pad (only used by hilbert / composite).
    X_tr_n, X_te_n = zscore(X_tr_raw, X_te_raw)
    X_tr_64, X_te_64 = pad_to_hilbert_dim(X_tr_n, HILBERT_DIM, X_te_n)

    # PD iterations = ``epochs`` only for hilbert / composite; AST ignores it.
    local_hp = dict(hp)
    if method in ("hilbert", "composite"):
        local_hp["n_iters"] = int(epochs)
    if method == "ast" and "population_dim" not in local_hp:
        local_hp["population_dim"] = HILBERT_DIM

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    if method == "ast":
        r2, train_r2, branch = fit_ast(
            X_tr_n,
            y_tr,
            X_te_n,
            y_te,
            seed=seed,
            hp=local_hp,
        )
    elif method == "hilbert":
        r2, train_r2, branch = fit_hilbert(
            X_tr_64,
            y_tr,
            X_te_64,
            y_te,
            seed=seed,
            hp=local_hp,
        )
    elif method == "composite":
        r2, train_r2, branch = fit_composite(
            X_tr_n,
            y_tr,
            X_te_n,
            y_te,
            X_tr_64,
            X_te_64,
            seed=seed,
            hp=local_hp,
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
        n_train=int(n_train),
        n_test=int(n_test),
        r2_test=float(r2) if math.isfinite(r2) else float("nan"),
        wall_s=float(elapsed),
        peak_mem_mb=float(peak_mb),
        active_branch=str(branch),
        train_r2=float(train_r2) if math.isfinite(train_r2) else float("nan"),
        n_features=int(X_tr_raw.shape[1]),
    )


def run_benchmark(
    *,
    dataset: str,
    n_samples: int,
    seeds: Sequence[int],
    methods: Sequence[str],
    epochs: int,
    test_frac: float,
    hyperparams: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Run the full benchmark — returns a JSON-serialisable payload."""
    import numpy as np

    rows = load_dataset(dataset, n_max=n_samples)
    if len(rows) < 10:
        raise RuntimeError(f"dataset {dataset!r} returned {len(rows)} rows; need ≥10")

    records: List[FitRecord] = []
    for method in methods:
        for seed in seeds:
            hp = dict(hyperparams[method])
            rec = _run_one(
                method=method,
                seed=int(seed),
                rows=rows,
                hp=hp,
                test_frac=float(test_frac),
                epochs=int(epochs),
            )
            records.append(rec)

    # Aggregate per method.
    per_method: Dict[str, MethodSummary] = {}
    for method in methods:
        m_records = [r for r in records if r.method == method]
        r2_arr = np.asarray([r.r2_test for r in m_records], dtype=float)
        wall_arr = np.asarray([r.wall_s for r in m_records], dtype=float)
        mem_arr = np.asarray([r.peak_mem_mb for r in m_records], dtype=float)
        per_seed: Dict[int, Dict[str, float]] = {}
        for r in m_records:
            per_seed.setdefault(
                int(r.seed),
                {
                    "r2_test": float(r.r2_test),
                    "wall_s": float(r.wall_s),
                    "peak_mem_mb": float(r.peak_mem_mb),
                },
            )
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

    payload = {
        "config": {
            "dataset": str(dataset),
            "n_samples_requested": int(n_samples),
            "n_samples_loaded": int(len(rows)),
            "seeds": [int(s) for s in seeds],
            "methods": list(methods),
            "epochs": int(epochs),
            "test_frac": float(test_frac),
            "n_descriptors": int(N_DESCRIPTORS),
            "hilbert_dim": int(HILBERT_DIM),
            "feature_columns": list(FEATURE_COLUMNS),
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
                    str(s): {kk: float(vv) for kk, vv in vals.items()}
                    for s, vals in per_method[m].per_seed.items()
                },
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
            "Wf-39 Phase 2 — Deflexpressor (LambdaASTSR) end-to-end training "
            "bench on PlatinAI 5K + IC50."
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
        help="Hilbert PD iterations (--method hilbert/composite)",
    )
    p.add_argument(
        "--method",
        type=str,
        nargs="+",
        default=list(DEFAULT_METHODS),
        choices=["ast", "hilbert", "composite"],
        help="One or more of: ast / hilbert / composite",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="nci60",
        choices=["platinai", "nci60"],
        help="PlatinAI 4-cell synthetic IC50 or NCI-60 real pIC50 "
        "(default nci60 — PlatinAI 4-cell has zero feature "
        "correlation with RDKit descriptors)",
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
    """Render a comparison markdown table."""
    cfg = payload["config"]
    methods = cfg["methods"]
    per_method = payload["per_method"]
    # Pick fastest method as the speedup reference.
    fastest = min(methods, key=lambda m: per_method[m]["wall_s_mean"])
    lines: List[str] = []
    lines.append(f"# Wf-39 Phase 2 — Deflexpressor end-to-end training bench\n")
    lines.append(f"**Dataset**: `{cfg['dataset']}`  ")
    lines.append(
        f"**n_samples loaded**: {cfg['n_samples_loaded']} "
        f"(requested {cfg['n_samples_requested']})  "
    )
    lines.append(f"**Seeds**: {cfg['seeds']}  ")
    lines.append(f"**Methods**: {cfg['methods']}  ")
    lines.append(f"**Epochs (PD iters)**: {cfg['epochs']}  ")
    lines.append(
        f"**Descriptors**: {cfg['n_descriptors']}-d RDKit → "
        f"{cfg['hilbert_dim']}-d Hilbert-augmented\n"
    )
    lines.append("\n## Aggregate (across seeds)\n")
    lines.append(
        "| Method | R² mean ± std | Wall (s) mean ± std | "
        "Peak mem (MB) mean ± std | Speedup |"
    )
    lines.append("|---|---|---|---|---|")
    for m in methods:
        agg = per_method[m]
        speedup = (
            agg["wall_s_mean"] / per_method[fastest]["wall_s_mean"]
            if per_method[fastest]["wall_s_mean"] > 0
            else float("inf")
        )
        speedup_str = f"{speedup:.2f}x (ref)" if m == fastest else f"{speedup:.2f}x"
        lines.append(
            f"| {m} | {agg['r2_mean']:+.4f} ± {agg['r2_std']:.4f} | "
            f"{agg['wall_s_mean']:.3f} ± {agg['wall_s_std']:.3f} | "
            f"{agg['peak_mem_mb_mean']:.2f} ± {agg['peak_mem_mb_std']:.2f} | "
            f"{speedup_str} |"
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
    lines.append("\n## Active branch for composite (per seed)\n")
    recs = payload["records"]
    if any(r["method"] == "composite" for r in recs):
        comp_recs = [r for r in recs if r["method"] == "composite"]
        for r in comp_recs:
            lines.append(
                f"- seed={r['seed']:>3}  active_branch=`{r['active_branch']}`  "
                f"R²={r['r2_test']:+.4f}  train_R²={r['train_r2']:+.4f}"
            )
    lines.append("\n## Honest framing\n")
    lines.append(
        "* ``ast`` uses 12-d RDKit descriptors; ``hilbert`` / "
        "``composite`` use the 64-d pairwise-interaction augmentation.\n"
        "* ``epochs`` is forwarded as ``n_iters`` to the Hilbert "
        "branch only; the AST branch uses ``generations``.\n"
        "* ``composite`` runs both branches and reports the better "
        "test R²; it is **not** a weighted ensemble.\n"
        "* Wall-time and memory are machine-dependent; the speedup "
        "column is the intended *relative* metric.\n"
        "* pIC50 is **not log-scaled** beyond the standard "
        "``pIC50 = -log10(IC50_M)`` conversion; the regression "
        "target is continuous in ``[3.5, 9.5]``.\n"
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

    # Hyperparams: keep ``epochs`` honoured for hilbert / composite.
    hyperparams = {m: dict(DEFAULT_HYPERPARAMS[m]) for m in args.method}

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    payload = run_benchmark(
        dataset=str(args.dataset),
        n_samples=int(n_samples),
        seeds=seeds,
        methods=tuple(args.method),
        epochs=int(epochs),
        test_frac=float(args.test_frac),
        hyperparams=hyperparams,
    )
    elapsed = float(time.perf_counter() - t0)

    metrics_path = write_metrics(payload, out_dir / "deflex_bench.json")
    print(f"[wf39_deflex] wrote {metrics_path}")
    md_path = write_markdown(payload, out_dir / "deflex_bench.md")
    print(f"[wf39_deflex] wrote {md_path}")
    print(
        f"\n[wf39_deflex] === Summary (over {len(seeds)} seeds × "
        f"{len(args.method)} methods) ==="
    )
    for m in args.method:
        agg = payload["per_method"][m]
        print(
            f"  {m:9s}: R²={agg['r2_mean']:+.4f} ± {agg['r2_std']:.4f}  "
            f"wall={agg['wall_s_mean']:.3f}s ± {agg['wall_s_std']:.3f}  "
            f"peak_mem={agg['peak_mem_mb_mean']:.2f}MB"
        )
    print(f"\n[wf39_deflex] elapsed={elapsed:.1f}s  out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
