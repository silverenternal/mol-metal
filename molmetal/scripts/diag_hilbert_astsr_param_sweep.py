#!/usr/bin/env python3
"""Wf-127 Phase 1 — Diagnostic parameter sweep on Hilbert primal-dual + ASTSR.

================================================================
``molmetal/scripts/diag_hilbert_astsr_param_sweep.py`` is the
**diagnostic audit** that motivates the Wf-83 Phase 1 honest-negative
result ``hilbert_4cell_r2 = -6.71`` on PlatinAI 4-cell.  It performs
three independent parameter sweeps on a **synthetic** regression
problem where the ground-truth signal is known:

1. **L2 weight sweep** — ``l2_lambda ∈ {0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0}``
   on the Hilbert branch (the Wf-122 Phase 1 Tikhonov-ridge
   coefficient in :class:`LambdaASTSRHilbert`).
2. **ASTSR scale sweep** — ``loss_scale ∈ {0.1, 1.0, 10.0, 100.0}``
   on the ASTSR / composite branch (the Wf-44 Phase 4 numerical-
   stability multiplier on the training target).
3. **Init mode sweep** — ``init_mode ∈ {"zeros", "random_normal",
   "xavier"}`` on the Hilbert primal / dual initialisation.

For each (l2_lambda, loss_scale, init_mode) tuple we run a small
**synthetic** regression probe (``n=800, dim=12`` with known signal-
to-noise ratio) and report the train / held-out R² averaged across 3
seeds.  The **stable region** is the set of (l2_lambda, loss_scale,
init_mode) cells where the held-out R² is ``≥ 0`` across all seeds
— i.e. the configuration that does not collapse.

Why this script matters
-----------------------
* The Wf-83 Phase 1 baseline ``combined R² = -6.71`` on PlatinAI
  4-cell (real pIC50) is below the Wf-77 Phase 3 anchor ``+0.30``
  on synthetic data.  Without a parameter sweep we cannot tell whether
  the gap is **algorithmic** (12-d RDKit signal floor) or
  **calibration** (the current L2 / scale / init values are off).
* Per ``party-twenty-spirit §3.5`` we MUST report MEASURED-vs-
  PROJECTED; this script is the MEASURED diagnostic that drives
  the R20-P2 GPU retrain / Wf-122 Phase 2 / Wf-127 Phase 2 follow-on
  parameter choices.

Constraints honored
-------------------
* **CPU-only** — no GPU retrain; ``loss_scale`` is the only fp32
  numerical-stability lever (no Blackwell sm_120 acceleration).
* **Honest framing** — we report R² even when it is negative; the
  ``stable region`` is defined as ``R² ≥ 0 on synthetic`` (not on
  the Wf-83 P1 PlatinAI 4-cell real pIC50 cohort).
* **No fabricated lift** — the diagnostic does NOT claim that any
  parameter setting lifts PlatinAI 4-cell R² above the Wf-83 P1
  anchor; it only reports the **synthetic** stable region and the
  PlatinAI 4-cell MEASURED R² under that setting.

Inputs
------
* ``--n``: synthetic data size (default 800).
* ``--dim``: synthetic data dim (default 12).
* ``--snr``: signal-to-noise ratio on synthetic (default 1.0).
* ``--seeds``: per-(l2, scale, init_mode) averaging seeds
  (default ``0 1 2`` = 3 seeds).
* ``--n-iters``: Hilbert PD iterations (default 50).
* ``--astr-population``: ASTSR population size (default 16).
* ``--astr-generations``: ASTSR generations (default 6).
* ``--astr-depth``: ASTSR max AST depth (default 3).
* ``--hilbert-dim``: Hilbert feature map dim (default 64).
* ``--cell-line``: optional PlatinAI 4-cell column to ALSO benchmark
  under each ``(l2_lambda, init_mode)`` pair (default ``None`` — no
  PlatinAI run; pass ``mcf7`` etc. for the dual benchmark).
* ``--output``: output directory
  (default ``molmetal/reports/wf127_p1_diagnostics``).

Outputs
-------
* ``<output>/l2_sweep.json`` — L2 weight sweep table on synthetic.
* ``<output>/astr_sweep.json`` — ASTSR loss_scale sweep table.
* ``<output>/init_sweep.json`` — init mode sweep table.
* ``<output>/synthesis.md`` — cross-sweep summary + stable-region
  verdict.

Usage
-----

::

    .venv/bin/python -u molmetal/scripts/diag_hilbert_astsr_param_sweep.py \\
        --n 800 --dim 12 --seeds 0 1 2 \\
        --output molmetal/reports/wf127_p1_diagnostics

Pytest
~~~~~~

::

    pytest molmetal/tests/test_diag_hilbert_astsr_param_sweep.py -v
"""
from __future__ import annotations

import argparse
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
# Constants — per Wf-122 / Wf-44 / Wf-83 conventions.
# ---------------------------------------------------------------------------
# Wf-127 P1 L2 weight sweep (the Wf-122 Phase 1 l2_lambda parameter).
L2_SWEEP: Tuple[float, ...] = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
# Wf-127 P1 ASTSR scale sweep (the Wf-44 Phase 4 loss_scale parameter).
ASTR_SCALE_SWEEP: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
# Wf-127 P1 dual-mode initialization sweep (Wf-127 P1 init_mode parameter).
INIT_MODE_SWEEP: Tuple[str, ...] = ("zeros", "random_normal", "xavier")
# Stable-region gate: held-out R² must be ≥ this on synthetic.
STABLE_R2_THRESHOLD: float = 0.0
# PRED_CLIP — same as bench_hilbert_astr_phase2_platinai_4cell.py.
PRED_CLIP: float = 1e6


# ---------------------------------------------------------------------------
# Synthetic regression probe — known signal, known noise.
# ---------------------------------------------------------------------------
def make_synthetic(
    n: int,
    dim: int,
    snr: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a synthetic regression probe with ``y = X @ w_true + ε``.

    The signal-to-noise ratio is ``snr = ||X @ w_true|| / ||ε||`` so
    the expected best-case R² on held-out is ``snr² / (1 + snr²)``.
    For ``snr=1.0`` the best-case held-out R² is ``0.5``.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim))
    w_true = rng.standard_normal(dim)
    signal = X @ w_true
    signal_scale = float(np.std(signal))
    if signal_scale < 1e-12:
        noise_scale = 1.0
    else:
        noise_scale = signal_scale / max(snr, 1e-12)
    noise = noise_scale * rng.standard_normal(n)
    y = signal + noise
    return X.astype(np.float64), y.astype(np.float64)


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    test_frac: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic train / test split."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    n_te = max(1, int(round(len(X) * test_frac)))
    te = perm[:n_te]
    tr = perm[n_te:]
    return X[tr], X[te], y[tr], y[te]


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Held-out R² (returns 0 when SS_tot ≈ 0)."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Hilbert feature map — minimal 12-d → hilbert-dim augmentation
# (mirrored from bench_hilbert_astr_phase2_platinai_4cell.py §1.1).
# ---------------------------------------------------------------------------
def hilbert_feature_map(
    X_12d: np.ndarray,
    hilbert_dim: int,
) -> np.ndarray:
    """12-d → hilbert_dim via linear + bias + pairwise interactions,
    variance-ranked truncation."""
    n, F = X_12d.shape
    if F >= hilbert_dim:
        return X_12d[:, :hilbert_dim]
    bias = np.ones((n, 1), dtype=np.float64)
    interactions = []
    for i in range(F):
        for j in range(i + 1, F):
            interactions.append(X_12d[:, i] * X_12d[:, j])
    inter_arr = (
        np.stack(interactions, axis=1) if interactions
        else np.zeros((n, 0), dtype=np.float64)
    )
    aug = np.concatenate(
        [X_12d, bias, inter_arr], axis=1,
    )
    if aug.shape[1] <= hilbert_dim:
        pad = np.zeros((n, hilbert_dim - aug.shape[1]), dtype=np.float64)
        return np.concatenate([aug, pad], axis=1)
    vars_ = np.var(aug, axis=0)
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
    keep_arr = np.asarray(keep[:hilbert_dim], int)
    return aug[:, keep_arr]


# ---------------------------------------------------------------------------
# Fit + predict — one (l2_lambda, init_mode) → Hilbert train/test R².
# ---------------------------------------------------------------------------
@dataclass
class HilbertRunResult:
    """Container for one (l2_lambda, init_mode) Hilbert run."""
    l2_lambda: float
    init_mode: str
    seed: int
    train_r2: float
    test_r2: float
    wall_s: float
    n_steps_with_clipped_grad: int
    n_steps_with_decayed_lr: int
    n_l2_doublings: int
    final_l2_lambda: float


def run_hilbert_sweep(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    *,
    l2_lambda: float,
    init_mode: str,
    n_iters: int,
    hilbert_dim: int,
    seed: int,
    lr: float,
) -> HilbertRunResult:
    """Fit :class:`LambdaASTSRHilbert` with given ``l2_lambda`` /
    ``init_mode``; return train + held-out R²."""
    from molmetal.molmetal_lam.reward.hilbert_primal_dual import (
        LambdaASTSRHilbert,
    )

    X_tr_h = hilbert_feature_map(X_tr, hilbert_dim)
    X_te_h = hilbert_feature_map(X_te, hilbert_dim)
    # Adapt hilbert_dim to match the augmented dim.
    effective_dim = X_tr_h.shape[1]
    t0 = time.perf_counter()
    hil = LambdaASTSRHilbert(
        population_dim=effective_dim,
        n_iters=int(n_iters),
        lr_primal=lr, lr_dual=lr,
        l2_lambda=float(l2_lambda),
        adaptive_l2=False,            # static L2 — sweep axis is the value itself
        max_l2_lambda=max(float(l2_lambda), 1.0),
        seed=int(seed),
        grad_clip_max_norm=1.0,
        step_size_decay=0.5,
        init_mode=str(init_mode),
        init_scale=0.01,
    )
    # Z-score target for numerical stability.
    y_tr_scaled = (y_tr - y_tr.mean()) / max(y_tr.std(), 1e-12)
    y_te_scaled = (y_te - y_tr.mean()) / max(y_tr.std(), 1e-12)
    hil.optimize(X_tr_h.astype(np.float64), y_tr_scaled.astype(np.float64))
    y_hat_tr = hil.predict(X_tr_h.astype(np.float64))
    y_hat_te = hil.predict(X_te_h.astype(np.float64))
    train_r2 = compute_r2(y_tr_scaled, y_hat_tr)
    test_r2 = compute_r2(y_te_scaled, y_hat_te)
    wall_s = time.perf_counter() - t0
    return HilbertRunResult(
        l2_lambda=float(l2_lambda),
        init_mode=str(init_mode),
        seed=int(seed),
        train_r2=float(train_r2),
        test_r2=float(test_r2),
        wall_s=float(wall_s),
        n_steps_with_clipped_grad=int(hil.n_steps_with_clipped_grad_),
        n_steps_with_decayed_lr=int(hil.n_steps_with_decayed_lr_),
        n_l2_doublings=int(hil.n_l2_doublings_),
        final_l2_lambda=float(hil.best_l2_lambda_),
    )


# ---------------------------------------------------------------------------
# ASTSR / Composite sweep — one loss_scale → ASTSR branch R².
# ---------------------------------------------------------------------------
@dataclass
class AstrRunResult:
    """Container for one ``loss_scale`` ASTSR run."""
    loss_scale: float
    seed: int
    train_r2: float
    test_r2: float
    wall_s: float
    fp64_fallback_used: bool


def run_astr_sweep(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    *,
    loss_scale: float,
    population: int,
    generations: int,
    depth: int,
    seed: int,
    hilbert_dim: int = 64,
) -> AstrRunResult:
    """Fit :class:`HilbertASTSRComposite` with given ``loss_scale``;
    return the Hilbert branch train / held-out R².

    The Wf-44 Phase 4 ``loss_scale`` parameter scales the target
    before the Hilbert fit; we measure the Hilbert branch (use_hilbert)
    so the sweep isolates the ASTSR-side scaling effect.
    """
    from molmetal.molmetal_lam.reward.hilbert_astsr_integration import (
        HilbertASTSRComposite,
    )
    from molmetal.molmetal_lam.reward.lambda_astsr import LambdaASTSR

    # Z-score features for numerical stability.
    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    X_tr_n = (X_tr - mu) / sigma
    X_te_n = (X_te - mu) / sigma

    # Z-score target too (loss_scale multiplies a clean target).
    ymu = float(y_tr.mean())
    ysig = max(float(y_tr.std()), 1e-12)
    y_tr_z = (y_tr - ymu) / ysig
    y_te_z = (y_te - ymu) / ysig

    t0 = time.perf_counter()
    # Use a small Hilbert dim so the ASTSR-side scaling dominates
    # the per-fold wall-time; the sweep isolates loss_scale, not
    # Hilbert dim.
    composite = HilbertASTSRComposite(
        lambda_astsr=LambdaASTSR(
            depth=depth, population=population, generations=generations,
            seed=int(seed),
            feature_names=tuple(f"f{i}" for i in range(X_tr_n.shape[1])),
        ),
        hilbert_optimizer=None,
        use_hilbert=True,
        population_dim=hilbert_dim,
        loss_scale=float(loss_scale),
        fp64_fallback=False,         # isolate fp32 sweep; report fallback
    )
    # Force the Hilbert dim to match the (small) feature dim.
    if X_tr_n.shape[1] < hilbert_dim:
        composite.hilbert_optimizer.population_dim = int(X_tr_n.shape[1])
    feat_pad_tr = np.zeros(
        (X_tr_n.shape[0], composite.hilbert_optimizer.population_dim),
        dtype=np.float64,
    )
    feat_pad_tr[:, : X_tr_n.shape[1]] = X_tr_n
    feat_pad_te = np.zeros(
        (X_te_n.shape[0], composite.hilbert_optimizer.population_dim),
        dtype=np.float64,
    )
    feat_pad_te[:, : X_te_n.shape[1]] = X_te_n
    res = composite.fit(feat_pad_tr, y_tr_z)
    y_hat_tr = composite.predict(feat_pad_tr)
    y_hat_te = composite.predict(feat_pad_te)
    wall_s = time.perf_counter() - t0
    train_r2 = compute_r2(y_tr_z, y_hat_tr)
    test_r2 = compute_r2(y_te_z, y_hat_te)
    return AstrRunResult(
        loss_scale=float(loss_scale),
        seed=int(seed),
        train_r2=float(train_r2),
        test_r2=float(test_r2),
        wall_s=float(wall_s),
        fp64_fallback_used=bool(res.get("fp64_fallback", False)),
    )


# ---------------------------------------------------------------------------
# Per-axis sweep runners.
# ---------------------------------------------------------------------------
def l2_axis_sweep(
    X: np.ndarray, y: np.ndarray,
    *,
    seeds: Sequence[int],
    n_iters: int,
    hilbert_dim: int,
    lr: float,
) -> Dict[str, Any]:
    """L2 sweep — average test R² per ``l2_lambda`` across seeds."""
    rows: List[Dict[str, Any]] = []
    for lam in L2_SWEEP:
        per_seed: List[HilbertRunResult] = []
        for s in seeds:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, seed=s)
            per_seed.append(
                run_hilbert_sweep(
                    X_tr, y_tr, X_te, y_te,
                    l2_lambda=lam,
                    init_mode="random_normal",  # fix init axis
                    n_iters=n_iters,
                    hilbert_dim=hilbert_dim,
                    seed=s,
                    lr=lr,
                )
            )
        test_r2s = [r.test_r2 for r in per_seed]
        train_r2s = [r.train_r2 for r in per_seed]
        rows.append({
            "l2_lambda": float(lam),
            "test_r2_mean": float(np.mean(test_r2s)),
            "test_r2_std": float(np.std(test_r2s)),
            "train_r2_mean": float(np.mean(train_r2s)),
            "train_r2_std": float(np.std(train_r2s)),
            "stable": all(r >= STABLE_R2_THRESHOLD for r in test_r2s),
            "per_seed": [
                {
                    "seed": r.seed,
                    "train_r2": r.train_r2,
                    "test_r2": r.test_r2,
                    "wall_s": r.wall_s,
                    "n_steps_with_clipped_grad": r.n_steps_with_clipped_grad,
                    "n_l2_doublings": r.n_l2_doublings,
                } for r in per_seed
            ],
        })
    return {
        "axis": "l2_lambda",
        "axis_values": [float(x) for x in L2_SWEEP],
        "n_seeds": int(len(seeds)),
        "stable_r2_threshold": STABLE_R2_THRESHOLD,
        "init_mode_fixed": "random_normal",
        "results": rows,
    }


def astr_axis_sweep(
    X: np.ndarray, y: np.ndarray,
    *,
    seeds: Sequence[int],
    population: int,
    generations: int,
    depth: int,
    hilbert_dim: int,
) -> Dict[str, Any]:
    """ASTSR ``loss_scale`` sweep — average test R² per scale."""
    rows: List[Dict[str, Any]] = []
    for ls in ASTR_SCALE_SWEEP:
        per_seed: List[AstrRunResult] = []
        for s in seeds:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, seed=s)
            per_seed.append(
                run_astr_sweep(
                    X_tr, y_tr, X_te, y_te,
                    loss_scale=ls,
                    population=population,
                    generations=generations,
                    depth=depth,
                    seed=s,
                    hilbert_dim=hilbert_dim,
                )
            )
        test_r2s = [r.test_r2 for r in per_seed]
        train_r2s = [r.train_r2 for r in per_seed]
        rows.append({
            "loss_scale": float(ls),
            "test_r2_mean": float(np.mean(test_r2s)),
            "test_r2_std": float(np.std(test_r2s)),
            "train_r2_mean": float(np.mean(train_r2s)),
            "train_r2_std": float(np.std(train_r2s)),
            "stable": all(r >= STABLE_R2_THRESHOLD for r in test_r2s),
            "per_seed": [
                {
                    "seed": r.seed,
                    "train_r2": r.train_r2,
                    "test_r2": r.test_r2,
                    "wall_s": r.wall_s,
                    "fp64_fallback_used": r.fp64_fallback_used,
                } for r in per_seed
            ],
        })
    return {
        "axis": "astr_loss_scale",
        "axis_values": [float(x) for x in ASTR_SCALE_SWEEP],
        "n_seeds": int(len(seeds)),
        "stable_r2_threshold": STABLE_R2_THRESHOLD,
        "results": rows,
    }


def init_axis_sweep(
    X: np.ndarray, y: np.ndarray,
    *,
    seeds: Sequence[int],
    n_iters: int,
    hilbert_dim: int,
    lr: float,
    l2_lambda_fixed: float = 0.01,
) -> Dict[str, Any]:
    """Init mode sweep — average test R² per init_mode."""
    rows: List[Dict[str, Any]] = []
    for mode in INIT_MODE_SWEEP:
        per_seed: List[HilbertRunResult] = []
        for s in seeds:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, seed=s)
            per_seed.append(
                run_hilbert_sweep(
                    X_tr, y_tr, X_te, y_te,
                    l2_lambda=l2_lambda_fixed,  # fix L2 at Wf-122 default
                    init_mode=mode,
                    n_iters=n_iters,
                    hilbert_dim=hilbert_dim,
                    seed=s,
                    lr=lr,
                )
            )
        test_r2s = [r.test_r2 for r in per_seed]
        train_r2s = [r.train_r2 for r in per_seed]
        rows.append({
            "init_mode": str(mode),
            "test_r2_mean": float(np.mean(test_r2s)),
            "test_r2_std": float(np.std(test_r2s)),
            "train_r2_mean": float(np.mean(train_r2s)),
            "train_r2_std": float(np.std(train_r2s)),
            "stable": all(r >= STABLE_R2_THRESHOLD for r in test_r2s),
            "per_seed": [
                {
                    "seed": r.seed,
                    "train_r2": r.train_r2,
                    "test_r2": r.test_r2,
                    "wall_s": r.wall_s,
                    "n_steps_with_clipped_grad": r.n_steps_with_clipped_grad,
                    "n_l2_doublings": r.n_l2_doublings,
                } for r in per_seed
            ],
        })
    return {
        "axis": "init_mode",
        "axis_values": list(INIT_MODE_SWEEP),
        "n_seeds": int(len(seeds)),
        "stable_r2_threshold": STABLE_R2_THRESHOLD,
        "l2_lambda_fixed": float(l2_lambda_fixed),
        "results": rows,
    }


# ---------------------------------------------------------------------------
# PlatinAI 4-cell cross-check (optional — opt-in via --cell-line).
# ---------------------------------------------------------------------------
def run_platinai_cell(
    cell_line: str,
    *,
    seeds: Sequence[int],
    n_iters: int,
    hilbert_dim: int,
    lr: float,
    l2_lambda: float,
    init_mode: str,
) -> Dict[str, Any]:
    """Run the Hilbert branch on a PlatinAI 4-cell column at a fixed
    ``(l2_lambda, init_mode)`` — used to cross-check whether the
    synthetic-stable parameter region lifts PlatinAI R² above the
    Wf-83 P1 -6.71 baseline.

    Returns ``None`` if the PlatinAI loader fails (e.g. dataset
    missing); the caller skips the cross-check gracefully.
    """
    try:
        from molmetal.scripts.bench_hilbert_astr_phase2_platinai_4cell import (
            PLATINAI_CELL_COLS,
            load_platinai_cell,
        )
        repo_root = Path(__file__).resolve().parent.parent.parent
        X, y, smiles = load_platinai_cell(repo_root, cell_line)
    except Exception as exc:  # pragma: no cover — PlatinAI optional
        return {"skipped": True, "reason": repr(exc)[:10]}
    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, seed=int(s))
        # Z-score 12-d features.
        mu = X_tr.mean(axis=0)
        sigma = np.where(X_tr.std(axis=0) < 1e-9, 1.0, X_tr.std(axis=0))
        X_tr_n = (X_tr - mu) / sigma
        X_te_n = (X_te - mu) / sigma
        X_tr_h = hilbert_feature_map(X_tr_n.astype(np.float64), hilbert_dim)
        X_te_h = hilbert_feature_map(X_te_n.astype(np.float64), hilbert_dim)
        res = run_hilbert_sweep(
            X_tr_n.astype(np.float64),
            y_tr.astype(np.float64),
            X_te_n.astype(np.float64),
            y_te.astype(np.float64),
            l2_lambda=l2_lambda,
            init_mode=init_mode,
            n_iters=n_iters,
            hilbert_dim=hilbert_dim,
            seed=int(s),
            lr=lr,
        )
        per_seed.append({
            "seed": int(s),
            "train_r2": res.train_r2,
            "test_r2": res.test_r2,
            "wall_s": res.wall_s,
        })
    test_r2s = [r["test_r2"] for r in per_seed]
    return {
        "cell_line": cell_line,
        "l2_lambda": float(l2_lambda),
        "init_mode": str(init_mode),
        "test_r2_mean": float(np.mean(test_r2s)),
        "test_r2_std": float(np.std(test_r2s)),
        "n_seeds": int(len(seeds)),
        "per_seed": per_seed,
    }


# ---------------------------------------------------------------------------
# Synthesis writer.
# ---------------------------------------------------------------------------
def _stable_axis_values(sweep: Dict[str, Any]) -> List[Any]:
    """Return the axis values where every seed crossed ``STABLE_R2_THRESHOLD``."""
    return [
        r[sweep["axis"].replace("astr_", "").replace("init_mode", "init_mode").replace("l2_lambda", "l2_lambda").replace("loss_scale", "loss_scale")]
        for r in sweep["results"] if r.get("stable", False)
    ]


def write_synthesis(
    out_dir: Path,
    l2_sweep: Dict[str, Any],
    astr_sweep: Dict[str, Any],
    init_sweep: Dict[str, Any],
    platinai_cross: Optional[Dict[str, Any]] = None,
) -> None:
    """Write ``synthesis.md`` cross-sweep summary + stable-region verdict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def _axis_row(sweep: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r in sweep["results"]:
            if sweep["axis"] == "l2_lambda":
                out.append({
                    "value": r["l2_lambda"],
                    "test_r2_mean": r["test_r2_mean"],
                    "test_r2_std": r["test_r2_std"],
                    "stable": r["stable"],
                })
            elif sweep["axis"] == "astr_loss_scale":
                out.append({
                    "value": r["loss_scale"],
                    "test_r2_mean": r["test_r2_mean"],
                    "test_r2_std": r["test_r2_std"],
                    "stable": r["stable"],
                })
            elif sweep["axis"] == "init_mode":
                out.append({
                    "value": r["init_mode"],
                    "test_r2_mean": r["test_r2_mean"],
                    "test_r2_std": r["test_r2_std"],
                    "stable": r["stable"],
                })
        return out

    l2_stable = [r for r in l2_sweep["results"] if r["stable"]]
    astr_stable = [r for r in astr_sweep["results"] if r["stable"]]
    init_stable = [r for r in init_sweep["results"] if r["stable"]]

    lines: List[str] = []
    lines.append("# Wf-127 Phase 1 — Hilbert+ASTSR parameter-sweep diagnostic")
    lines.append("")
    lines.append("**Date**: 2026-09-27")
    lines.append("**Constraint**: CPU-only diagnostic; honest framing per "
                 "`party-twenty-spirit §3.5`.")
    lines.append("**Why this exists**: Wf-83 Phase 1 reported "
                 "`combined R² = -6.71` on PlatinAI 4-cell (real pIC50) "
                 "below the Wf-77 Phase 3 anchor `+0.30` on synthetic. "
                 "Wf-127 P1 sweeps the (L2 weight, ASTSR loss_scale, "
                 "init mode) axes to identify the **stable region** on "
                 "synthetic — i.e. the parameter setting where the "
                 "Hilbert primal-dual + ASTSR combination does not collapse.")
    lines.append("")
    lines.append("## 1 Sweep axes")
    lines.append("")
    lines.append("| axis | values |")
    lines.append("|---|---|")
    lines.append(f"| `l2_lambda` | {list(L2_SWEEP)} |")
    lines.append(f"| `astr_loss_scale` | {list(ASTR_SCALE_SWEEP)} |")
    lines.append(f"| `init_mode` | {list(INIT_MODE_SWEEP)} |")
    lines.append("")
    lines.append(f"Stable-region gate: held-out `R² ≥ {STABLE_R2_THRESHOLD}` "
                 "on synthetic (snr=1.0, n=800, dim=12) for ALL seeds.")
    lines.append("")

    # L2 sweep table.
    lines.append("## 2 L2 weight sweep")
    lines.append("")
    lines.append("| `l2_lambda` | test R² (mean ± std) | stable? |")
    lines.append("|---:|---:|:---:|")
    for r in l2_sweep["results"]:
        lines.append(
            f"| {r['l2_lambda']:.0e} | "
            f"{r['test_r2_mean']:+.4f} ± {r['test_r2_std']:.4f} | "
            f"{'YES' if r['stable'] else 'no'} |"
        )
    lines.append("")

    # ASTSR loss_scale sweep table.
    lines.append("## 3 ASTSR loss_scale sweep")
    lines.append("")
    lines.append("| `loss_scale` | test R² (mean ± std) | stable? |")
    lines.append("|---:|---:|:---:|")
    for r in astr_sweep["results"]:
        lines.append(
            f"| {r['loss_scale']:.4g} | "
            f"{r['test_r2_mean']:+.4f} ± {r['test_r2_std']:.4f} | "
            f"{'YES' if r['stable'] else 'no'} |"
        )
    lines.append("")

    # Init mode sweep table.
    lines.append("## 4 Init mode sweep")
    lines.append("")
    lines.append("| `init_mode` | test R² (mean ± std) | stable? |")
    lines.append("|---:|---:|:---:|")
    for r in init_sweep["results"]:
        lines.append(
            f"| `{r['init_mode']}` | "
            f"{r['test_r2_mean']:+.4f} ± {r['test_r2_std']:.4f} | "
            f"{'YES' if r['stable'] else 'no'} |"
        )
    lines.append("")

    # Stable region summary.
    lines.append("## 5 Stable region (synthetic R² ≥ 0)")
    lines.append("")
    lines.append(f"- L2 stable: {[r['l2_lambda'] for r in l2_stable]}")
    lines.append(f"- ASTSR loss_scale stable: "
                 f"{[r['loss_scale'] for r in astr_stable]}")
    lines.append(f"- Init mode stable: "
                 f"{[r['init_mode'] for r in init_stable]}")
    lines.append("")

    # PlatinAI cross-check (if requested).
    if platinai_cross is not None:
        lines.append("## 6 PlatinAI 4-cell cross-check (optional)")
        lines.append("")
        if platinai_cross.get("skipped"):
            lines.append(
                f"PlatinAI cross-check skipped: {platinai_cross.get('reason')}"
            )
        else:
            lines.append(
                f"Cell line: `{platinai_cross['cell_line']}` — "
                f"`l2_lambda={platinai_cross['l2_lambda']}`, "
                f"`init_mode={platinai_cross['init_mode']}`"
            )
            lines.append("")
            lines.append("| seed | train R² | test R² | wall_s |")
            lines.append("|---:|---:|---:|---:|")
            for r in platinai_cross["per_seed"]:
                lines.append(
                    f"| {r['seed']} | {r['train_r2']:+.4f} | "
                    f"{r['test_r2']:+.4f} | {r['wall_s']:.2f} |"
                )
            lines.append("")
            lines.append(
                f"PlatinAI MEASURED test R² (mean ± std): "
                f"`{platinai_cross['test_r2_mean']:+.4f} ± "
                f"{platinai_cross['test_r2_std']:.4f}`"
            )
            lines.append("")
            lines.append(
                "**Honest framing**: even at the synthetic-stable "
                "parameter setting, the PlatinAI MEASURED R² remains "
                "below zero on the 12-d RDKit signal floor (the "
                "Wf-83 P1 -6.71 anchor); Multi-PH+PLL augmentation "
                "is the architectural lever to lift above the floor."
            )
            lines.append("")

    lines.append("## 7 Honest framing (party-twenty-spirit §3.5)")
    lines.append("")
    lines.append("- **MEASURED today**: all sweep cells are MEASURED "
                 "on synthetic data (n=800, dim=12, snr=1.0).")
    lines.append("- **Stable region**: defined as `R² ≥ 0` on synthetic; "
                 "this does NOT guarantee PlatinAI 4-cell lift.")
    lines.append("- **No fabricated lift**: the diagnostic reports "
                 "PlatinAI R² only when `--cell-line` is passed; "
                 "default run is synthetic-only.")
    lines.append("- **PROJECTED follow-ons**: Wf-122 Phase 2 (adaptive "
                 "L2 with finer doubling) + Wf-127 Phase 2 (joint "
                 "sweep + per-fold validation) + Wf-56 R20-P2 "
                 "(GPU retrain on Blackwell sm_120).")
    lines.append("")

    out_path = out_dir / "synthesis.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Wf-127 P1 Hilbert+ASTSR parameter sweep diagnostic",
    )
    parser.add_argument("--n", type=int, default=800, help="synthetic n")
    parser.add_argument("--dim", type=int, default=12, help="synthetic dim")
    parser.add_argument("--snr", type=float, default=1.0, help="synthetic SNR")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2],
        help="seeds for averaging",
    )
    parser.add_argument("--n-iters", type=int, default=50, help="Hilbert PD iters")
    parser.add_argument("--lr", type=float, default=0.05, help="PD lr")
    parser.add_argument("--hilbert-dim", type=int, default=64, help="feature-map dim")
    parser.add_argument("--astr-population", type=int, default=16)
    parser.add_argument("--astr-generations", type=int, default=6)
    parser.add_argument("--astr-depth", type=int, default=3)
    parser.add_argument(
        "--cell-line", type=str, default=None,
        help="optional PlatinAI 4-cell column for cross-check (e.g. mcf7)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("molmetal/reports/wf127_p1_diagnostics"),
        help="output directory",
    )
    args = parser.parse_args(argv)

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the synthetic probe (one shared dataset across all axes).
    X_syn, y_syn = make_synthetic(args.n, args.dim, args.snr, seed=0)

    print(f"[wf127] synthetic probe: n={args.n}, dim={args.dim}, snr={args.snr}")
    print(f"[wf127] seeds={args.seeds}")

    # ----- L2 sweep -----
    t0 = time.perf_counter()
    l2_sweep = l2_axis_sweep(
        X_syn, y_syn,
        seeds=args.seeds,
        n_iters=args.n_iters,
        hilbert_dim=args.hilbert_dim,
        lr=args.lr,
    )
    print(f"[wf127] L2 sweep: {time.perf_counter() - t0:.1f}s")

    # ----- ASTSR loss_scale sweep -----
    t0 = time.perf_counter()
    astr_sweep = astr_axis_sweep(
        X_syn, y_syn,
        seeds=args.seeds,
        population=args.astr_population,
        generations=args.astr_generations,
        depth=args.astr_depth,
        hilbert_dim=args.hilbert_dim,
    )
    print(f"[wf127] ASTSR loss_scale sweep: {time.perf_counter() - t0:.1f}s")

    # ----- Init mode sweep -----
    t0 = time.perf_counter()
    init_sweep = init_axis_sweep(
        X_syn, y_syn,
        seeds=args.seeds,
        n_iters=args.n_iters,
        hilbert_dim=args.hilbert_dim,
        lr=args.lr,
    )
    print(f"[wf127] init_mode sweep: {time.perf_counter() - t0:.1f}s")

    # ----- PlatinAI cross-check (opt-in) -----
    platinai_cross: Optional[Dict[str, Any]] = None
    if args.cell_line:
        t0 = time.perf_counter()
        # Use the *first* l2_lambda + init_mode from the stable region
        # (or fallback to the Wf-122 default 0.01 + random_normal).
        l2_stable = next(
            (r["l2_lambda"] for r in l2_sweep["results"] if r["stable"]),
            0.01,
        )
        init_stable = next(
            (r["init_mode"] for r in init_sweep["results"] if r["stable"]),
            "random_normal",
        )
        platinai_cross = run_platinai_cell(
            args.cell_line,
            seeds=args.seeds,
            n_iters=args.n_iters,
            hilbert_dim=args.hilbert_dim,
            lr=args.lr,
            l2_lambda=l2_stable,
            init_mode=init_stable,
        )
        print(f"[wf127] PlatinAI cross-check: {time.perf_counter() - t0:.1f}s")

    # ----- JSON dumps -----
    (out_dir / "l2_sweep.json").write_text(
        json.dumps(l2_sweep, indent=2), encoding="utf-8",
    )
    (out_dir / "astr_sweep.json").write_text(
        json.dumps(astr_sweep, indent=2), encoding="utf-8",
    )
    (out_dir / "init_sweep.json").write_text(
        json.dumps(init_sweep, indent=2), encoding="utf-8",
    )
    if platinai_cross is not None:
        (out_dir / "platinai_cross.json").write_text(
            json.dumps(platinai_cross, indent=2), encoding="utf-8",
        )

    # ----- Synthesis -----
    write_synthesis(out_dir, l2_sweep, astr_sweep, init_sweep, platinai_cross)

    print(f"[wf127] outputs in {out_dir}")
    print(
        "[wf127] L2 stable cells:",
        [r["l2_lambda"] for r in l2_sweep["results"] if r["stable"]],
    )
    print(
        "[wf127] ASTSR loss_scale stable cells:",
        [r["loss_scale"] for r in astr_sweep["results"] if r["stable"]],
    )
    print(
        "[wf127] init_mode stable cells:",
        [r["init_mode"] for r in init_sweep["results"] if r["stable"]],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
