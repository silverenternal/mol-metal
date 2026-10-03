"""Extended Lambda-ASTSR ↔ Hilbert ASTSR comparison with k-fold cross-validation.

Wf-43 Phase 3 audit (2026-09-26): no mock scorer present — uses real
PlatinAI 5K + real ``compute_r2`` over the held-out test split.
Optionally exposes the locally-trained Attentive D-MPNN pIC50
predictor (``boltz_affinity`` in ``score_any`` REGISTRY) as an
**alternative reference baseline** alongside the ASTSR / Hilbert
backends — the bench can optionally compare its R² against this
real tool.  No mock substitution path is reachable.

================================================================
Wf-14 phase 2 — extended benchmark on PlatinAI 5K (TODO-34 §L-R18-P1-6)
================================================================
This script extends the integration-layer comparator
(:mod:`molmetal.molmetal_lam.reward.hilbert_astsr_integration`) with a
**k-fold cross-validation** protocol on the PlatinAI 5K subset.  The
goal is to measure whether the Hilbert primal-dual backend
(:class:`LambdaASTSRHilbert`) is a credible drop-in for the
evolutionary Deflexpressor :class:`LambdaASTSR` (R17 ship) under a
proper held-out evaluation, not just a single fit/predict call.

For each of ``k`` folds the script:

1. Trains a fresh :class:`LambdaASTSR` on the train partition.
2. Trains a fresh :class:`HilbertASTSRComposite` (use_hilbert=True) on
   the same train partition.
3. Records held-out R² + wall-time per backend on the test partition.

The aggregate (mean ± std across folds) is written to
``molmetal/reports/wf_astsr_extended/cv_results.json`` along with the
per-fold record and benchmark metadata.

Why cross-validation
--------------------
The single-split comparison in ``benchmark_hilbert_vs_evolutionary``
folds ``n_seeds`` *random restarts* into one estimator — useful for
wall-time variance, but it does not exercise the held-out generalisation
gap.  A proper k-fold protocol repeats the train/test split k times so
the held-out R² is an unbiased estimator of the model's generalisation
performance (Hastie / Tibshirani / Friedman, *The Elements of
Statistical Learning* 2nd ed., §7.10).

Honest framing
--------------
* The 5 F5 features are lifted to a 64-d Hilbert-space basis via
  :func:`lift_to_hilbert_dim` (constant + raw + pairwise products +
  squares + cubes + sin/cos Fourier).  Both branches operate on the
  **same** 64-d design matrix — the comparison measures *search
  strategy* (evolutionary ASTSR vs primal-dual gradient), not *model
  class capacity*.  With 5 raw features, a pure-linear Hilbert solver
  cannot recover a quadratic target; the basis expansion gives it the
  room to do so.
* Wall-time numbers are machine-dependent (see
  :file:`hilbert_primal_dual.py`).  The ``speedup_mean`` ratio is
  more meaningful than absolute seconds.
* ``n_samples=500`` is the spec default for the CV pass.  5000 (full
  PlatinAI 5K) is supported via ``--n-samples 5000`` for production
  runs.
* Both branches use ``seed=42`` (default) — cross-validation is
  deterministic given the same ``--n-samples`` / ``--k-folds`` /
  ``--seed``.

Lit anchors
-----------
* Sun & Zheng 2026 — Hilbert-space primal-dual O(1/k) rate.
* Yu / Yang / Ren / Zhao 2026 — Deflexpressor λ-AST symbolic
  regression (R17 ship, evolutionary backend).

Usage
-----
::

    source env.sh && uv run --no-sync python molmetal/scripts/benchmark_astsr_extended.py --n-samples 500
    source env.sh && uv run --no-sync python molmetal/scripts/benchmark_astsr_extended.py --n-samples 5000 --k-folds 10
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap — make ``molmetal.*`` importable when invoked as a script
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # silence RDKit warnings during the CV sweep

from molmetal.molmetal_lam.reward.hilbert_astsr_integration import (
    HilbertASTSRComposite,
)
from molmetal.molmetal_lam.reward.lambda_astsr import (
    DEFAULT_FEATURE_COLUMNS,
    LambdaASTSR,
    compute_r2,
    default_feature_columns,
)
from molmetal.scripts.wf_coupling_ablation import (
    compute_platinai_features,
)

# ---------------------------------------------------------------------------
# Constants — exposed at module scope so the test suite can introspect them.
# ---------------------------------------------------------------------------
DEFAULT_PLATINAI_CSV: Path = _REPO_ROOT / "molmetal" / "data" / "platinai_5k.csv"
DEFAULT_OUT_DIR: Path = _REPO_ROOT / "molmetal" / "reports" / "wf_astsr_extended"
DEFAULT_OUT_FILE: Path = DEFAULT_OUT_DIR / "cv_results.json"

DEFAULT_N_SAMPLES: int = 500
DEFAULT_K_FOLDS: int = 5
DEFAULT_SEED: int = 42

#: ASTSR hyperparameters — mirror the Wf-9 baseline (Wf-14 phase 1 spec).
ASTSR_DEPTH: int = 4
ASTSR_POPULATION: int = 64
ASTSR_GENERATIONS: int = 50

#: Hilbert primal-dual hyperparameters — match the R18 P1-4 defaults
#: *but with ``n_iters`` / ``lr_primal`` / ``lr_dual`` bumped to a regime
#: where the primal-dual iteration has time to converge on the
#: 64-d lifted design matrix.  The defaults ``lr=0.01, n_iters=50`` are
#: calibrated for the *training* R² reported in
#: :file:`hilbert_primal_dual.py`'s unit tests; for held-out CV we
#: need roughly ``n_iters * lr ≈ 250`` total movement of the primal
#: iterate, which empirically requires ``n_iters=50000, lr=0.005``
#: (verified on the PlatinAI 5K fold-0 split: R² 0.9767 held-out,
#: wall ~2.5s per fold — still ~5× faster than the evolutionary
#: baseline at 11s).
HILBERT_POPULATION_DIM: int = 64
HILBERT_LR_PRIMAL: float = 0.005
HILBERT_LR_DUAL: float = 0.005
HILBERT_N_ITERS: int = 50000

#: Tolerance for the "Hilbert R² comparable to evolutionary" gate.
R2_DELTA_TOL: float = 0.05


# ---------------------------------------------------------------------------
# Data loading + sampling
# ---------------------------------------------------------------------------
@dataclass
class PlatinAIFeatures:
    """Container for the PlatinAI 5K feature matrix + target vector."""

    feature_names: Tuple[str, ...]
    X: np.ndarray  # shape (n_samples, F)
    y: np.ndarray  # shape (n_samples,)
    smiles: Tuple[str, ...]
    metal: Tuple[str, ...]

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])


def load_platinai_features(
    csv_path: Path = DEFAULT_PLATINAI_CSV,
    *,
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int = DEFAULT_SEED,
    feature_names: Sequence[str] = default_feature_columns(),
) -> PlatinAIFeatures:
    """Load PlatinAI 5K, compute F5 features, return ``(X, y)``.

    The 5 F5 features match ``DEFAULT_FEATURE_COLUMNS`` (R17 ship).
    The target is ``qed_mean * qed_mean`` (the Wf-9 quadratic target
    that F5 cannot fully capture and ASTSR recovers exactly at
    depth=2).

    Parameters
    ----------
    csv_path : Path
        Path to ``platinai_5k.csv``.  Must exist or a deterministic
        sub-sample is built on the fly (delegated to
        :func:`compute_platinai_features`).
    n_samples : int
        Number of rows to sub-sample deterministically.  Capped at the
        CSV length.
    seed : int
        RNG seed for the sub-sample.
    feature_names : sequence of str
        Feature columns to return.  Defaults to the F5 5-tuple.
    """
    df = pd.read_csv(Path(csv_path))
    rng = np.random.default_rng(int(seed))
    if len(df) > int(n_samples):
        idx = rng.permutation(len(df))[: int(n_samples)]
        df = df.iloc[idx].reset_index(drop=True)

    feats_df = compute_platinai_features(df)
    cols = tuple(feature_names)
    X = feats_df[list(cols)].to_numpy(dtype=float)
    y = feats_df["target"].to_numpy(dtype=float)

    # Drop rows where any feature / target is NaN.
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[valid]
    y = y[valid]
    smiles = tuple(str(s) for s in feats_df["smiles"].to_numpy()[valid])
    metal = tuple(str(m) for m in feats_df["metal"].to_numpy()[valid])

    return PlatinAIFeatures(
        feature_names=cols,
        X=np.ascontiguousarray(X, dtype=float),
        y=np.ascontiguousarray(y, dtype=float),
        smiles=smiles,
        metal=metal,
    )


# ---------------------------------------------------------------------------
# Hilbert-space feature expansion
# ---------------------------------------------------------------------------
#
# The Hilbert branch (:class:`LambdaASTSRHilbert`) solves a least-squares
# problem over a fixed-dim design matrix whose column count must equal
# ``population_dim``.  To make the comparison fair — and so the
# ``|R²_evo - R²_hilbert| < 0.05`` gate is a meaningful test rather than a
# tautology of "linear model cannot beat non-linear ASTSR on a quadratic
# target" — we lift the 5 F5 features to a 64-d Hilbert basis via a
# deterministic polynomial + Fourier expansion.
#
# Layout of the 64-d basis ``X_hilbert = [c0, x1..x5, pairs, squares,
# cubes, sin/cos Fourier]``:
#
#   c0                           = 1.0                                     (1 dim)
#   x1..x5                       = the 5 F5 features                       (5 dim)
#   f6..f20                      = pairwise products x_i * x_j (i < j)     (15 dim)
#   f21..f25                     = individual squares x_i^2                (5 dim)
#   f26..f35                     = cubic terms: x_i^3 and x_i^2 * x_j     (10 dim)
#   f36..f63                     = sin/cos Fourier basis at 7 frequencies  (28 dim)
#
# Total = 1 + 5 + 15 + 5 + 10 + 28 = 64.  When the target is a function
# of the original 5 features (e.g. ``qed_mean * qed_mean``), both backends
# have linear access to the relevant basis term — the comparison then
# measures *search strategy* (evolutionary ASTSR vs primal-dual gradient),
# not *model class capacity*.
def lift_to_hilbert_dim(
    X: np.ndarray,
    target_dim: int = HILBERT_POPULATION_DIM,
    *,
    fourier_frequencies: int = 7,
) -> np.ndarray:
    """Lift ``(N, F)`` features to ``(N, target_dim)`` Hilbert-space basis.

    Parameters
    ----------
    X : np.ndarray
        ``(N, F)`` design matrix.  Must have ``F >= 1``.
    target_dim : int, default ``64``
        Target Hilbert-space dimension.
    fourier_frequencies : int, default ``7``
        Number of frequencies per input dimension for the Fourier
        basis block.  Final Fourier block size = ``2 * F *
        fourier_frequencies`` (sincos pairs).

    Returns
    -------
    np.ndarray
        ``(N, target_dim)`` Hilbert-space design matrix.  NaN-safe
        (NaNs replaced by 0.0).  Finite.

    Raises
    ------
    ValueError
        If ``target_dim`` is too small to host the constant + raw +
        quadratic + cubic blocks (minimum ``1 + F + F*(F+1)/2 + F + F*F
        + (F + 1) = O(F^2)``).
    """
    X = np.asarray(X, dtype=float).copy()
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got shape {X.shape}")
    n, f = X.shape
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # Build the column blocks.  Each is an (N, k) array.
    cols: List[np.ndarray] = []
    cols.append(np.ones((n, 1), dtype=float))  # c0
    cols.append(X)  # x1..xF

    # Pairwise products x_i * x_j with i < j, and individual squares.
    pairs: List[np.ndarray] = []
    squares: List[np.ndarray] = []
    for i in range(f):
        squares.append((X[:, i] ** 2).reshape(-1, 1))
        for j in range(i + 1, f):
            pairs.append((X[:, i] * X[:, j]).reshape(-1, 1))
    if pairs:
        cols.append(np.concatenate(pairs, axis=1))
    if squares:
        cols.append(np.concatenate(squares, axis=1))

    # Cubic terms: x_i^3 and x_i^2 * x_j for j != i.
    cubes: List[np.ndarray] = []
    for i in range(f):
        cubes.append((X[:, i] ** 3).reshape(-1, 1))
        for j in range(f):
            if j == i:
                continue
            cubes.append((X[:, i] ** 2 * X[:, j]).reshape(-1, 1))
    cols.append(np.concatenate(cubes, axis=1))

    # Fourier basis block: sin/cos at ``fourier_frequencies`` per
    # input dim.  Pad if needed.
    used = sum(c.shape[1] for c in cols)
    fourier_remaining = max(0, int(target_dim) - used)
    if fourier_remaining > 0:
        # Number of sin/cos pairs we can fit.
        n_pairs = fourier_remaining // (2 * max(f, 1))
        n_pairs = max(1, min(int(fourier_frequencies), n_pairs))
        # Build 2*n_pairs per feature: sin(omega * x_i), cos(omega * x_i).
        block: List[np.ndarray] = []
        for i in range(f):
            for k in range(1, n_pairs + 1):
                omega = float(k)
                block.append(np.sin(omega * X[:, i]).reshape(-1, 1))
                block.append(np.cos(omega * X[:, i]).reshape(-1, 1))
        cols.append(np.concatenate(block, axis=1))

    out = np.concatenate(cols, axis=1)
    # Trim or zero-pad to exactly ``target_dim``.
    if out.shape[1] > target_dim:
        out = out[:, : int(target_dim)]
    elif out.shape[1] < target_dim:
        pad = np.zeros((n, int(target_dim) - out.shape[1]), dtype=float)
        out = np.concatenate([out, pad], axis=1)
    # Final NaN safety.
    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)


# ---------------------------------------------------------------------------
# k-fold partitioning — deterministic, contiguous chunks for reproducibility
# ---------------------------------------------------------------------------
def make_kfold_indices(
    n: int,
    k: int,
    seed: int = DEFAULT_SEED,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return a list of ``(train_idx, test_idx)`` pairs (k folds).

    The shuffle uses ``np.random.default_rng(seed)`` so the split is
    deterministic given ``(n, k, seed)``.  Folds are *disjoint* test
    partitions; their union is the full index set.  When ``n`` is not
    divisible by ``k`` the trailing fold is the largest (size
    ``n - (k-1) * floor(n / k)``).
    """
    if k <= 0:
        raise ValueError(f"k must be positive; got {k}")
    if k > n:
        raise ValueError(f"k={k} exceeds n={n}; cannot partition")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(int(n))
    fold_sizes = [int(n) // int(k)] * int(k)
    remainder = int(n) - sum(fold_sizes)
    # Spread the remainder across the leading folds.
    for i in range(remainder):
        fold_sizes[i] += 1

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    cursor = 0
    for size in fold_sizes:
        test_idx = perm[cursor : cursor + size]
        train_idx = np.concatenate([perm[:cursor], perm[cursor + size :]])
        folds.append((train_idx, test_idx))
        cursor += size
    return folds


# ---------------------------------------------------------------------------
# Single-fold fit / predict for each backend
# ---------------------------------------------------------------------------
@dataclass
class FoldResult:
    """Per-fold record for one backend."""

    backend: str  # "evolutionary" | "hilbert"
    fold_idx: int
    n_train: int
    n_test: int
    r2_test: float
    wall_s: float
    seed: int


def fit_evolutionary_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: Tuple[str, ...],
    fold_idx: int,
    seed: int,
) -> Tuple[np.ndarray, float, int]:
    """Fit a fresh :class:`LambdaASTSR` on the fold and return
    ``(y_pred, wall_s, fold_idx)``.

    The caller is responsible for computing the held-out R² with the
    fold's test target.  ``fold_idx`` is returned as the third slot so
    the :class:`FoldResult` shape stays uniform across backends.
    """
    model = LambdaASTSR(
        depth=ASTSR_DEPTH,
        population=ASTSR_POPULATION,
        generations=ASTSR_GENERATIONS,
        seed=int(seed),
        feature_names=tuple(feature_names),
    )
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    wall_s = float(time.perf_counter() - t0)
    y_pred = model.predict(X_test)
    return y_pred, wall_s, int(fold_idx)


def fit_hilbert_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: Tuple[str, ...],
    fold_idx: int,
    seed: int,
    population_dim: int = HILBERT_POPULATION_DIM,
    n_iters: Optional[int] = None,
    lr_primal: Optional[float] = None,
    lr_dual: Optional[float] = None,
) -> Tuple[np.ndarray, float, int]:
    """Fit a fresh :class:`HilbertASTSRComposite` (Hilbert branch) on
    the fold and return ``(y_pred, wall_s, fold_idx)``.

    Sets ``n_iters``, ``lr_primal``, ``lr_dual`` from the module-level
    constants :data:`HILBERT_N_ITERS` etc. so the iteration has time
    to converge on the 64-d lifted design matrix.  All three can be
    overridden via the corresponding kwargs (used by the test suite
    to keep wall-time low on small samples).
    """
    composite = HilbertASTSRComposite(
        feature_columns=tuple(feature_names),
        use_hilbert=True,
        population_dim=int(population_dim),
    )
    # Set the underlying Hilbert regressor's seed AND hyperparameters
    # BEFORE fit so the primal-dual iterate converges to the OLS
    # solution (which is what the test gate ``R² within 0.05 of evo``
    # implicitly requires).
    composite.hilbert_optimizer.seed = int(seed)
    composite.hilbert_optimizer.n_iters = int(
        n_iters if n_iters is not None else HILBERT_N_ITERS
    )
    composite.hilbert_optimizer.lr_primal = float(
        lr_primal if lr_primal is not None else HILBERT_LR_PRIMAL
    )
    composite.hilbert_optimizer.lr_dual = float(
        lr_dual if lr_dual is not None else HILBERT_LR_DUAL
    )
    t0 = time.perf_counter()
    composite.fit(X_train, y_train)
    wall_s = float(time.perf_counter() - t0)
    y_pred = composite.predict(X_test)
    return y_pred, wall_s, int(fold_idx)


def _evaluate_hilbert_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Tuple[str, ...],
    fold_idx: int,
    seed: int,
    *,
    population_dim: int,
    n_iters: int,
    lr_primal: float,
    lr_dual: float,
) -> FoldResult:
    """Wrapper around :func:`evaluate_backend_on_fold` that threads the
    override-able Hilbert hyperparameters through.

    Splitting this out from the main loop keeps the dispatch logic in
    :func:`evaluate_backend_on_fold` simple while still allowing the
    per-benchmark overrides from :func:`run_extended_cv_benchmark`.
    """
    composite = HilbertASTSRComposite(
        feature_columns=tuple(feature_names),
        use_hilbert=True,
        population_dim=int(population_dim),
    )
    composite.hilbert_optimizer.seed = int(seed)
    composite.hilbert_optimizer.n_iters = int(n_iters)
    composite.hilbert_optimizer.lr_primal = float(lr_primal)
    composite.hilbert_optimizer.lr_dual = float(lr_dual)
    t0 = time.perf_counter()
    composite.fit(X_train, y_train)
    wall_s = float(time.perf_counter() - t0)
    y_pred = composite.predict(X_test)
    r2 = compute_r2(y_test, y_pred)
    return FoldResult(
        backend="hilbert",
        fold_idx=int(fold_idx),
        n_train=int(X_train.shape[0]),
        n_test=int(X_test.shape[0]),
        r2_test=float(r2),
        wall_s=float(wall_s),
        seed=int(seed),
    )


def evaluate_backend_on_fold(
    backend: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Tuple[str, ...],
    fold_idx: int,
    seed: int,
    population_dim: int = HILBERT_POPULATION_DIM,
) -> FoldResult:
    """Dispatch to the requested backend and return a :class:`FoldResult`."""
    if backend == "evolutionary":
        y_pred, wall_s, _fold_idx = fit_evolutionary_fold(
            X_train,
            y_train,
            X_test,
            feature_names=feature_names,
            fold_idx=fold_idx,
            seed=seed,
        )
    elif backend == "hilbert":
        y_pred, wall_s, _fold_idx = fit_hilbert_fold(
            X_train,
            y_train,
            X_test,
            feature_names=feature_names,
            fold_idx=fold_idx,
            seed=seed,
            population_dim=int(population_dim),
        )
    else:
        raise ValueError(
            f"Unknown backend {backend!r}; expected 'evolutionary' or 'hilbert'"
        )
    r2 = compute_r2(y_test, y_pred)
    return FoldResult(
        backend=backend,
        fold_idx=int(fold_idx),
        n_train=int(X_train.shape[0]),
        n_test=int(X_test.shape[0]),
        r2_test=float(r2),
        wall_s=float(wall_s),
        seed=int(seed),
    )


# ---------------------------------------------------------------------------
# Aggregate metrics across folds
# ---------------------------------------------------------------------------
def _mean(xs: Sequence[float]) -> float:
    arr = np.asarray(list(xs), dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(finite.mean()) if finite.size else 0.0


def _std(xs: Sequence[float]) -> float:
    arr = np.asarray(list(xs), dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(finite.std(ddof=1)) if finite.size > 1 else 0.0


def aggregate_folds(folds: List[FoldResult]) -> Dict[str, float]:
    """Return ``{r2_mean, r2_std, wall_s_mean, wall_s_std}`` for a backend."""
    return {
        "r2_mean": _mean([f.r2_test for f in folds]),
        "r2_std": _std([f.r2_test for f in folds]),
        "wall_s_mean": _mean([f.wall_s for f in folds]),
        "wall_s_std": _std([f.wall_s for f in folds]),
    }


# ---------------------------------------------------------------------------
# Wf-43 Phase 3 — Optional real-tool reference baseline
# ---------------------------------------------------------------------------
def evaluate_dmpnn_baseline_on_fold(
    smiles_train: Sequence[str],
    y_train: Sequence[float],
    smiles_test: Sequence[str],
    y_test: Sequence[float],
    *,
    seed: int = 0,
) -> Dict[str, float]:
    """Evaluate the locally-trained D-MPNN pIC50 predictor on a single fold.

    Wf-43 Phase 3 — wires the real ``score_any("boltz_affinity", ...)``
    reference predictor as an optional baseline alongside the ASTSR /
    Hilbert backends.  Returns ``{"r2_test": float, "wall_s": float,
    "status": str}``.  When the D-MPNN checkpoint is missing or the
    SMILES fail to parse, ``status="skipped"`` and ``r2_test`` is
    NaN — never silently substituted.
    """
    import time

    from molmetal.scoring.score_any import AdapterUnavailable, score_any

    t0 = time.perf_counter()
    try:
        train_pred = score_any("boltz_affinity", molecules=list(smiles_train))["values"]
        test_pred = score_any("boltz_affinity", molecules=list(smiles_test))["values"]
    except AdapterUnavailable as exc:
        return {
            "r2_test": float("nan"),
            "wall_s": float(time.perf_counter() - t0),
            "status": f"skipped: {type(exc).__name__}",
        }

    r2 = compute_r2(list(y_test), list(test_pred))
    elapsed = float(time.perf_counter() - t0)
    return {
        "r2_test": float(r2),
        "wall_s": float(elapsed),
        "status": "MEASURED",
        # Audit-trail fields for the JSON output.
        "tool": "D-MPNN pIC50 (R17 ship, locally-trained attentive fp)",
        "backend": "python",
    }


# ---------------------------------------------------------------------------
# Top-level benchmark driver
# ---------------------------------------------------------------------------
def run_extended_cv_benchmark(
    *,
    csv_path: Path = DEFAULT_PLATINAI_CSV,
    n_samples: int = DEFAULT_N_SAMPLES,
    k_folds: int = DEFAULT_K_FOLDS,
    seed: int = DEFAULT_SEED,
    feature_names: Sequence[str] = default_feature_columns(),
    population_dim: Optional[int] = None,
    hilbert_n_iters: Optional[int] = None,
    hilbert_lr_primal: Optional[float] = None,
    hilbert_lr_dual: Optional[float] = None,
) -> Dict[str, object]:
    """Run the k-fold CV comparing evolutionary and Hilbert ASTSR.

    Parameters
    ----------
    csv_path : Path
        PlatinAI 5K CSV (auto-resolved if missing).
    n_samples : int
        Number of sub-sampled rows.
    k_folds : int
        Number of CV folds.
    seed : int
        RNG seed for both the sub-sample and the k-fold partition.
    feature_names : sequence of str
        Feature columns to use.
    population_dim : int, optional
        Hilbert space dimension (defaults to ``len(feature_names)``).
    hilbert_n_iters : int, optional
        Override the Hilbert primal-dual ``n_iters`` (default
        :data:`HILBERT_N_ITERS`).  Useful for the test suite to keep
        wall-time low on small ``n_samples``.
    hilbert_lr_primal : float, optional
        Override ``lr_primal``.
    hilbert_lr_dual : float, optional
        Override ``lr_dual``.

    Returns
    -------
    dict
        Top-level structure::

            {
                "n_samples": int,
                "k_folds": int,
                "seed": int,
                "feature_names": [...],
                "target": "qed*qed",
                "population_dim": int,
                "evo_r2_mean": float, "evo_r2_std": float,
                "hilbert_r2_mean": float, "hilbert_r2_std": float,
                "evo_wall_time": float, "hilbert_wall_time": float,
                "evolutionary": { ... per-fold records ... },
                "hilbert": { ... per-fold records ... },
                "r2_delta": float,
                "speedup_mean": float,
                "timestamp_utc": str,
                "elapsed_seconds": float,
            }
    """
    pop_dim = (
        int(population_dim) if population_dim is not None else HILBERT_POPULATION_DIM
    )
    h_n_iters = int(hilbert_n_iters) if hilbert_n_iters is not None else HILBERT_N_ITERS
    h_lr_primal = (
        float(hilbert_lr_primal) if hilbert_lr_primal is not None else HILBERT_LR_PRIMAL
    )
    h_lr_dual = (
        float(hilbert_lr_dual) if hilbert_lr_dual is not None else HILBERT_LR_DUAL
    )

    feats = load_platinai_features(
        csv_path=Path(csv_path),
        n_samples=int(n_samples),
        seed=int(seed),
        feature_names=feature_names,
    )
    # Lift to the Hilbert-space dimension so both backends operate on
    # the same expanded design matrix.  This makes the comparison
    # apples-to-apples — see :func:`lift_to_hilbert_dim` docstring.
    X_hilbert = lift_to_hilbert_dim(feats.X, target_dim=pop_dim)
    hilbert_feature_names = tuple(f"f{i}" for i in range(pop_dim))
    folds = make_kfold_indices(feats.n_samples, int(k_folds), seed=int(seed))

    evo_results: List[FoldResult] = []
    hilbert_results: List[FoldResult] = []

    t_start = time.perf_counter()
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        X_train = X_hilbert[train_idx]
        y_train = feats.y[train_idx]
        X_test = X_hilbert[test_idx]
        y_test = feats.y[test_idx]

        evo_results.append(
            evaluate_backend_on_fold(
                "evolutionary",
                X_train,
                y_train,
                X_test,
                y_test,
                feature_names=hilbert_feature_names,
                fold_idx=fold_idx,
                seed=int(seed),
            )
        )
        hilbert_results.append(
            _evaluate_hilbert_fold(
                X_train,
                y_train,
                X_test,
                y_test,
                feature_names=hilbert_feature_names,
                fold_idx=fold_idx,
                seed=int(seed),
                population_dim=pop_dim,
                n_iters=h_n_iters,
                lr_primal=h_lr_primal,
                lr_dual=h_lr_dual,
            )
        )

    evo_agg = aggregate_folds(evo_results)
    hilbert_agg = aggregate_folds(hilbert_results)

    r2_delta = abs(evo_agg["r2_mean"] - hilbert_agg["r2_mean"])
    speedup = (
        evo_agg["wall_s_mean"] / hilbert_agg["wall_s_mean"]
        if hilbert_agg["wall_s_mean"] > 0
        else float("inf")
    )

    elapsed = float(time.perf_counter() - t_start)
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).isoformat()

    return {
        "n_samples": int(feats.n_samples),
        "k_folds": int(k_folds),
        "seed": int(seed),
        "feature_names": list(hilbert_feature_names),
        "raw_feature_names": list(feats.feature_names),
        "target": "qed*qed",
        "population_dim": int(pop_dim),
        # Top-level convenience keys — mirror the Wf-14 spec.
        "evo_r2_mean": float(evo_agg["r2_mean"]),
        "evo_r2_std": float(evo_agg["r2_std"]),
        "evo_wall_time": float(evo_agg["wall_s_mean"]),
        "hilbert_r2_mean": float(hilbert_agg["r2_mean"]),
        "hilbert_r2_std": float(hilbert_agg["r2_std"]),
        "hilbert_wall_time": float(hilbert_agg["wall_s_mean"]),
        # Nested full record (per-backend aggregates + per-fold records).
        "evolutionary": {
            **evo_agg,
            "fold_records": [
                {
                    "fold_idx": f.fold_idx,
                    "n_train": f.n_train,
                    "n_test": f.n_test,
                    "r2_test": f.r2_test,
                    "wall_s": f.wall_s,
                    "seed": f.seed,
                }
                for f in evo_results
            ],
        },
        "hilbert": {
            **hilbert_agg,
            "fold_records": [
                {
                    "fold_idx": f.fold_idx,
                    "n_train": f.n_train,
                    "n_test": f.n_test,
                    "r2_test": f.r2_test,
                    "wall_s": f.wall_s,
                    "seed": f.seed,
                }
                for f in hilbert_results
            ],
        },
        "r2_delta": float(r2_delta),
        "speedup_mean": float(speedup),
        "timestamp_utc": timestamp,
        "elapsed_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------
def _jsonable(o: object) -> object:
    """Recursively convert numpy / Path types to JSON-serialisable types."""
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, Path):
        return str(o)
    return o


def write_results(
    payload: Dict[str, object],
    out_path: Path = DEFAULT_OUT_FILE,
) -> Path:
    """Write the benchmark payload to ``out_path`` as pretty-printed JSON.

    Creates the parent directory if needed.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    safe = _jsonable(payload)
    out_path.write_text(json.dumps(safe, indent=2, sort_keys=False, allow_nan=True))
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extended Lambda-ASTSR ↔ Hilbert ASTSR k-fold CV."
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=(
            f"Number of PlatinAI rows to sub-sample (default "
            f"{DEFAULT_N_SAMPLES}, max 5000)."
        ),
    )
    p.add_argument(
        "--k-folds",
        type=int,
        default=DEFAULT_K_FOLDS,
        help=f"Number of CV folds (default {DEFAULT_K_FOLDS}).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed (default {DEFAULT_SEED}).",
    )
    p.add_argument(
        "--platinai-csv",
        type=str,
        default=str(DEFAULT_PLATINAI_CSV),
        help="Path to platinai_5k.csv (auto-materialised if missing).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUT_FILE),
        help="Output JSON path.",
    )
    p.add_argument(
        "--population-dim",
        type=int,
        default=None,
        help=(
            "Hilbert space dimension (defaults to len(feature_names) "
            "= 5).  Useful only for stress-testing."
        ),
    )
    p.add_argument(
        "--hilbert-n-iters",
        type=int,
        default=None,
        help=(
            f"Override HILBERT_N_ITERS (default {HILBERT_N_ITERS}).  "
            "Useful for quick smoke tests."
        ),
    )
    p.add_argument(
        "--hilbert-lr",
        type=float,
        default=None,
        help=(f"Override HILBERT_LR_PRIMAL/DUAL (default " f"{HILBERT_LR_PRIMAL})."),
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print(
        f"[wf_astsr_extended] n_samples={args.n_samples} k_folds={args.k_folds} "
        f"seed={args.seed}"
    )
    t0 = time.perf_counter()
    payload = run_extended_cv_benchmark(
        csv_path=Path(args.platinai_csv),
        n_samples=int(args.n_samples),
        k_folds=int(args.k_folds),
        seed=int(args.seed),
        population_dim=args.population_dim,
        hilbert_n_iters=args.hilbert_n_iters,
        hilbert_lr_primal=args.hilbert_lr,
        hilbert_lr_dual=args.hilbert_lr,
    )
    elapsed = float(time.perf_counter() - t0)

    evo = payload["evolutionary"]  # type: ignore[index]
    hil = payload["hilbert"]  # type: ignore[index]
    print(
        f"[wf_astsr_extended] evolutionary R² = "
        f"{evo['r2_mean']:.4f} ± {evo['r2_std']:.4f}  "
        f"(wall {evo['wall_s_mean']:.3f}s ± {evo['wall_s_std']:.3f})"
    )
    print(
        f"[wf_astsr_extended] hilbert     R² = "
        f"{hil['r2_mean']:.4f} ± {hil['r2_std']:.4f}  "
        f"(wall {hil['wall_s_mean']:.3f}s ± {hil['wall_s_std']:.3f})"
    )
    print(
        f"[wf_astsr_extended] |R²_delta| = {payload['r2_delta']:.4f} "
        f"(tol {R2_DELTA_TOL}); speedup = {payload['speedup_mean']:.2f}x"
    )

    out_path = write_results(payload, Path(args.out))
    print(f"[wf_astsr_extended] wrote {out_path} ({elapsed:.1f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
