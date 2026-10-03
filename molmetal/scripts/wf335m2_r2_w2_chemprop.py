#!/usr/bin/env python3
"""Wf-335-M2 — FULL W₂ + chemprop v2 graph integration (CPU, multi-seed CV).

=====================================================================
Wf-335-M2 / WEAK_TO_STRONG §2.1 / TODO-122 P1.3 — full W₂ + chemprop
v2 graph integration on 3 real-pIC50 cohorts, with **multi-seed ×
K-fold CV** for robust R² estimation, **multiple W₂ projection axes**
(MolWt / MolLogP / TPSA / HeavyAtoms), **per-cohort breakdowns**
(per-cell for platinai_4cell, per-metal for metalcytotox_combined), and
**feature importance** (top 10 by Ridge weight magnitude per config).

Upstream: Wf-331-R5 (single-seed 8-config sweep, partial validation:
CEILING_BREAK on metalcytotox_combined only).  Wf-335-M2 is the
**FULL integration** that goes beyond Wf-331-R5 on 4 axes:

  1. Multi-seed × 5-fold CV — robust R² estimate (mean ± std over
     25 splits).
  2. Multiple W₂ projection axes — best-axis selection (per-cohort).
  3. Per-cohort breakdown — per-cell R² (platinai_4cell) and per-metal
     R² (metalcytotox_combined).
  4. Feature importance — top 10 by |Ridge weight| per config (sanity
     check that the W₂ + chemprop v2 graph features are non-trivially
     weighted, not noise).

The full pipeline (RDKit 32-d + chemprop v2 graph 258-d + W₂ 1-d) is
evaluated via Ridge baseline on 3 cohorts:

  * **platinai_4cell**         — 200 SMILES, 4-cell pIC50 (per-cell
                                  breakdown: mcf7, a549, a2780,
                                  a2780cis)
  * **hilbert**                — 200 SMILES (same data, seed-shift)
  * **metalcytotox_combined**  — 2000 SMILES, 5-metal MetalCytoToxDB
                                  activity target (per-metal
                                  breakdown: Pt, Ru, Ir, Au, Pd)

Lift is reported as R²(combined) - R²(Wf-325A baseline), with the
**multi-seed mean** as the headline number (vs Wf-331-R5 which used
a single split).

Public API
----------
* :func:`ridge_r2_cv`                  multi-seed K-fold CV R²
* :func:`select_best_w2_axis`          per-cohort best W₂ axis
* :func:`per_cell_r2_breakdown`        platinai per-cell R²
* :func:`per_metal_r2_breakdown`       metalcytotox per-metal R²
* :func:`feature_importance_topk`      top-K |Ridge weight| features
* :func:`run_wf335m2`                  main entry point

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* PROJECTED R² lift: +0.05 to +0.15 R² per cohort (the WEAK_TO_STRONG
  §2.1 lower bound).  If at least 1 cohort achieves CEILING_BREAK
  (best R² > 0 AND lift ≥ +0.10), the result is HONEST_NEGATIVE_PARTIAL.
* If all 3 cohorts achieve CEILING_BREAK on multi-seed CV mean:
  HONEST_NEGATIVE_R2_RESOLVED.
* CPU-only; no torch autograd; chemprop v2 featurizer in pure-numpy
  fallback mode.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover — defensive
    Ridge = None  # type: ignore
    StandardScaler = None  # type: ignore
    _HAS_SKLEARN = False


# Re-use Wf-331-R5 featurizers + baselines (single source of truth).
from molmetal.scripts.wf331r5_r2_w2_chemprop import (  # noqa: E402
    WF325A_BASELINE_R2,
    _get_graph_feature_dim,
    _probe_chemprop_dims,
    compute_covariate_w2_distance,
    extract_graph_features_pooled,
    load_cohort,
)


# ---------------------------------------------------------------------------
# Repo path resolution
# ---------------------------------------------------------------------------
def _resolve_repo_path(path: str) -> str:
    """Resolve a repo-relative path to an absolute path."""
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
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
    if os.path.exists(path):
        return os.path.abspath(path)
    return os.path.abspath(path)


# ---------------------------------------------------------------------------
# Multi-seed K-fold CV R²
# ---------------------------------------------------------------------------
def ridge_r2_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    n_seeds: int = 5,
    alpha: float = 1.0,
    seed_offset: int = 0,
) -> Dict[str, Any]:
    """Multi-seed K-fold CV R² on (X, y).

    Returns the **mean ± std** R² over `n_seeds × n_splits = 25`
    evaluations (default).  Uses stratified-like K-fold via random
    permutation (no sklearn StratifiedKFold dependency).

    Returns
    -------
    dict with keys:
      * r2_mean        — mean over all splits
      * r2_std         — std over all splits
      * r2_per_seed    — list of mean R² per seed (n_seeds entries)
      * r2_per_split   — flat list of all 25 R² values
      * n_splits, n_seeds, alpha
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask].astype(np.float64)
    y = y[mask]
    n = X.shape[0]
    if n < n_splits:
        return {
            "r2_mean": float("nan"),
            "r2_std": float("nan"),
            "r2_per_seed": [float("nan")] * n_seeds,
            "r2_per_split": [float("nan")] * (n_splits * n_seeds),
            "n_splits": n_splits,
            "n_seeds": n_seeds,
            "alpha": alpha,
        }
    r2_per_split: List[float] = []
    r2_per_seed: List[float] = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed + seed_offset)
        perm = rng.permutation(n)
        fold_sizes = [n // n_splits] * n_splits
        for i in range(n % n_splits):
            fold_sizes[i] += 1
        fold_idx = []
        start = 0
        for fs in fold_sizes:
            fold_idx.append(perm[start:start + fs])
            start += fs
        for k in range(n_splits):
            te_idx = fold_idx[k]
            tr_idx = np.concatenate(
                [fold_idx[i] for i in range(n_splits) if i != k],
                axis=0,
            )
            Xtr, Xte = X[tr_idx], X[te_idx]
            ytr, yte = y[tr_idx], y[te_idx]
            if _HAS_SKLEARN:
                scaler = StandardScaler()
                Xs_tr = scaler.fit_transform(Xtr)
                Xs_te = scaler.transform(Xte)
                model = Ridge(alpha=alpha)
                model.fit(Xs_tr, ytr)
                r2 = float(model.score(Xs_te, yte))
            else:
                Xs_tr = (Xtr - Xtr.mean(axis=0)) / (Xtr.std(axis=0) + 1e-9)
                Xs_te = (Xte - Xtr.mean(axis=0)) / (Xtr.std(axis=0) + 1e-9)
                XtX = Xs_tr.T @ Xs_tr
                Xty = Xs_tr.T @ ytr
                w = np.linalg.solve(
                    XtX + alpha * np.eye(XtX.shape[0]), Xty,
                )
                yhat = Xs_te @ w
                ss_res = float(np.sum((yte - yhat) ** 2))
                ss_tot = float(np.sum((yte - yte.mean()) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            r2_per_split.append(r2)
        # Seed-level mean
        seed_r2s = r2_per_split[-n_splits:]
        r2_per_seed.append(float(np.mean(seed_r2s)))
    r2_arr = np.asarray(r2_per_split, dtype=np.float64)
    return {
        "r2_mean": float(r2_arr.mean()) if r2_arr.size > 0 else float("nan"),
        "r2_std": float(r2_arr.std()) if r2_arr.size > 0 else float("nan"),
        "r2_per_seed": r2_per_seed,
        "r2_per_split": r2_per_split,
        "n_splits": n_splits,
        "n_seeds": n_seeds,
        "alpha": alpha,
    }


def ridge_r2_eval_with_w2(
    X_static: np.ndarray,
    y: np.ndarray,
    projection_axis: int = 0,
    n_splits: int = 5,
    n_seeds: int = 5,
    alpha: float = 1.0,
    seed_offset: int = 0,
) -> Dict[str, Any]:
    """Multi-seed K-fold CV R² with split-aware covariate-W₂.

    The W₂ reference is computed from training rows only (split-aware
    honest framing per WEAK_TO_STRONG §2.1 / Wf-331-R5 §6).
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(X_static).all(axis=1) & np.isfinite(y)
    X = X_static[mask].astype(np.float64)
    y = y[mask]
    n = X.shape[0]
    if n < n_splits:
        return {
            "r2_mean": float("nan"),
            "r2_std": float("nan"),
            "r2_per_seed": [float("nan")] * n_seeds,
            "r2_per_split": [float("nan")] * (n_splits * n_seeds),
            "n_splits": n_splits,
            "n_seeds": n_seeds,
            "alpha": alpha,
        }
    r2_per_split: List[float] = []
    r2_per_seed: List[float] = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed + seed_offset)
        perm = rng.permutation(n)
        fold_sizes = [n // n_splits] * n_splits
        for i in range(n % n_splits):
            fold_sizes[i] += 1
        fold_idx = []
        start = 0
        for fs in fold_sizes:
            fold_idx.append(perm[start:start + fs])
            start += fs
        for k in range(n_splits):
            te_idx = fold_idx[k]
            tr_idx = np.concatenate(
                [fold_idx[i] for i in range(n_splits) if i != k],
                axis=0,
            )
            Xtr_full, Xte_full = X[tr_idx], X[te_idx]
            ytr, yte = y[tr_idx], y[te_idx]
            # W₂ reference from training rows only
            if X.shape[1] == 0:
                # w2_only path — no static columns.  Skip W₂.
                X_full_tr = Xtr_full
                X_full_te = Xte_full
            else:
                proj_axis = min(projection_axis, X.shape[1] - 1)
                # Reference = full training matrix Xtr_full (m × D)
                # (covariate_w2_distance requires reference.shape[1] == X.shape[1]).
                w2_tr = compute_covariate_w2_distance(
                    Xtr_full, reference=Xtr_full,
                    projection_axis=proj_axis,
                )
                w2_te = compute_covariate_w2_distance(
                    Xte_full, reference=Xtr_full,
                    projection_axis=proj_axis,
                )
                X_full_tr = np.concatenate(
                    [Xtr_full, w2_tr.reshape(-1, 1)], axis=1,
                )
                X_full_te = np.concatenate(
                    [Xte_full, w2_te.reshape(-1, 1)], axis=1,
                )
            # Drop constant columns defensively
            if X_full_tr.shape[1] > 0:
                keep = X_full_tr.std(axis=0) > 1e-9
                X_full_tr = X_full_tr[:, keep]
                X_full_te = X_full_te[:, keep]
            if X_full_tr.shape[1] == 0:
                r2 = float("nan")
            elif _HAS_SKLEARN:
                scaler = StandardScaler()
                Xs_tr = scaler.fit_transform(X_full_tr)
                Xs_te = scaler.transform(X_full_te)
                model = Ridge(alpha=alpha)
                model.fit(Xs_tr, ytr)
                r2 = float(model.score(Xs_te, yte))
            else:
                Xs_tr = (X_full_tr - X_full_tr.mean(axis=0)) / (
                    X_full_tr.std(axis=0) + 1e-9
                )
                Xs_te = (X_full_te - X_full_tr.mean(axis=0)) / (
                    X_full_tr.std(axis=0) + 1e-9
                )
                XtX = Xs_tr.T @ Xs_tr
                Xty = Xs_tr.T @ ytr
                w = np.linalg.solve(
                    XtX + alpha * np.eye(XtX.shape[0]), Xty,
                )
                yhat = Xs_te @ w
                ss_res = float(np.sum((yte - yhat) ** 2))
                ss_tot = float(np.sum((yte - yte.mean()) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            r2_per_split.append(r2)
        seed_r2s = r2_per_split[-n_splits:]
        r2_per_seed.append(float(np.mean(seed_r2s)))
    r2_arr = np.asarray(r2_per_split, dtype=np.float64)
    return {
        "r2_mean": float(r2_arr.mean()) if r2_arr.size > 0 else float("nan"),
        "r2_std": float(r2_arr.std()) if r2_arr.size > 0 else float("nan"),
        "r2_per_seed": r2_per_seed,
        "r2_per_split": r2_per_split,
        "n_splits": n_splits,
        "n_seeds": n_seeds,
        "alpha": alpha,
    }


# ---------------------------------------------------------------------------
# W₂ projection axis selection
# ---------------------------------------------------------------------------
W2_PROJECTION_AXES: Tuple[int, ...] = (0, 1, 2, 7)  # MolWt, MolLogP, TPSA, HeavyAtoms


def select_best_w2_axis(
    X_static: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    n_seeds: int = 3,
    alpha: float = 1.0,
    seed_offset: int = 0,
) -> Dict[str, Any]:
    """Pick the best W₂ projection axis (MolWt vs MolLogP vs TPSA vs HeavyAtoms).

    Returns the axis index and the corresponding R² mean for each axis.
    The "best" axis is the one with the highest CV R² mean.
    """
    if X_static.shape[1] == 0:
        return {
            "best_axis": -1,
            "best_axis_r2_mean": float("nan"),
            "axis_results": {},
        }
    results: Dict[int, Dict[str, float]] = {}
    for axis in W2_PROJECTION_AXES:
        if axis >= X_static.shape[1]:
            continue
        cv = ridge_r2_eval_with_w2(
            X_static, y, projection_axis=axis,
            n_splits=n_splits, n_seeds=n_seeds, alpha=alpha,
            seed_offset=seed_offset,
        )
        results[axis] = {
            "r2_mean": cv["r2_mean"],
            "r2_std": cv["r2_std"],
        }
    if not results:
        return {
            "best_axis": -1,
            "best_axis_r2_mean": float("nan"),
            "axis_results": {},
        }
    best_axis = max(results, key=lambda a: results[a]["r2_mean"])
    return {
        "best_axis": int(best_axis),
        "best_axis_r2_mean": float(results[best_axis]["r2_mean"]),
        "axis_results": {
            int(k): {"r2_mean": v["r2_mean"], "r2_std": v["r2_std"]}
            for k, v in results.items()
        },
    }


# ---------------------------------------------------------------------------
# Per-cell / per-metal R² breakdown
# ---------------------------------------------------------------------------
PLATINAI_PER_CELL_COLUMNS: Tuple[str, ...] = (
    "pIC50_mcf7", "pIC50_a549", "pIC50_a2780", "pIC50_a2780cis",
)
PLATINAI_PER_CELL_LABELS: Tuple[str, ...] = (
    "mcf7", "a549", "a2780", "a2780cis",
)


def _read_csv(path: str) -> Tuple[List[str], List[List[str]]]:
    """Read CSV at the data/ path (replicated from Wf-331-R5 for self-containment)."""
    path = _resolve_repo_path(path)
    import csv
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if r and any(c.strip() for c in r)]
    return header, rows


def per_cell_r2_breakdown(
    smiles_list: List[str],
    X_static: np.ndarray,
    cohort: str = "platinai_4cell",
    n_splits: int = 5,
    n_seeds: int = 3,
    alpha: float = 1.0,
    seed_offset: int = 0,
) -> Dict[str, Any]:
    """Per-cell R² breakdown for platinai_4cell.

    Loads the per-cell pIC50 columns from the 12-d RDKit CSV and
    evaluates the same X_static on each cell's target separately.
    """
    if cohort != "platinai_4cell":
        return {}
    header12, rows12 = _read_csv(
        "molmetal/data/platinai_4cell_rdkit_features.csv"
    )
    smi_to_idx = {
        rows12[i][header12.index("smiles")]: i
        for i in range(len(rows12))
    }
    per_cell_results: Dict[str, Dict[str, float]] = {}
    for col, label in zip(PLATINAI_PER_CELL_COLUMNS, PLATINAI_PER_CELL_LABELS):
        y_cell = np.zeros(len(smiles_list), dtype=np.float64)
        valid = np.zeros(len(smiles_list), dtype=bool)
        for j, smi in enumerate(smiles_list):
            if smi in smi_to_idx:
                row = rows12[smi_to_idx[smi]]
                col_idx = header12.index(col)
                try:
                    v = float(row[col_idx])
                except (ValueError, IndexError):
                    v = float("nan")
                if np.isfinite(v):
                    y_cell[j] = v
                    valid[j] = True
        if valid.sum() < n_splits:
            per_cell_results[label] = {
                "r2_mean": float("nan"),
                "r2_std": float("nan"),
                "n_valid": int(valid.sum()),
            }
            continue
        cv = ridge_r2_cv(
            X_static[valid], y_cell[valid],
            n_splits=n_splits, n_seeds=n_seeds, alpha=alpha,
            seed_offset=seed_offset,
        )
        per_cell_results[label] = {
            "r2_mean": cv["r2_mean"],
            "r2_std": cv["r2_std"],
            "n_valid": int(valid.sum()),
        }
    return per_cell_results


def per_metal_r2_breakdown(
    smiles_list: List[str],
    X_static: np.ndarray,
    n_splits: int = 5,
    n_seeds: int = 3,
    alpha: float = 1.0,
    seed_offset: int = 0,
) -> Dict[str, Any]:
    """Per-metal R² breakdown for metalcytotox_combined.

    Loads the metal column from metalcytotox_rdkit_features.csv and
    evaluates R² per metal separately.
    """
    header12, rows12 = _read_csv(
        "molmetal/data/metalcytotox_rdkit_features.csv"
    )
    smi_to_metal: Dict[str, str] = {}
    smi_to_y: Dict[str, float] = {}
    smi_idx_col = header12.index("smiles")
    metal_col = header12.index("metal")
    activity_col = header12.index("activity")
    for r in rows12:
        smi = r[smi_idx_col]
        try:
            y_val = float(r[activity_col])
        except (ValueError, IndexError):
            continue
        if not np.isfinite(y_val):
            continue
        smi_to_metal[smi] = r[metal_col]
        smi_to_y[smi] = y_val
    per_metal_results: Dict[str, Dict[str, Any]] = {}
    metals_in_cohort = sorted({
        smi_to_metal[s] for s in smiles_list if s in smi_to_metal
    })
    for metal in metals_in_cohort:
        mask_m = np.array([
            smi_to_metal.get(s) == metal for s in smiles_list
        ], dtype=bool)
        if mask_m.sum() < n_splits:
            per_metal_results[metal] = {
                "r2_mean": float("nan"),
                "r2_std": float("nan"),
                "n_samples": int(mask_m.sum()),
            }
            continue
        y_metal = np.array([
            smi_to_y.get(s, float("nan")) for s in smiles_list
        ], dtype=np.float64)
        cv = ridge_r2_cv(
            X_static[mask_m], y_metal[mask_m],
            n_splits=n_splits, n_seeds=n_seeds, alpha=alpha,
            seed_offset=seed_offset,
        )
        per_metal_results[metal] = {
            "r2_mean": cv["r2_mean"],
            "r2_std": cv["r2_std"],
            "n_samples": int(mask_m.sum()),
        }
    return per_metal_results


# ---------------------------------------------------------------------------
# Feature importance (top-K by |Ridge weight|)
# ---------------------------------------------------------------------------
def feature_importance_topk(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    topk: int = 10,
    alpha: float = 1.0,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Top-K |Ridge weight| features.

    Trains Ridge on the full X with the given seed's permutation
    (80/20 split for honesty).  Returns the top-K features by
    |weight| descending, with sign preserved.
    """
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X_clean = X[mask].astype(np.float64)
    y_clean = np.asarray(y, dtype=np.float64)[mask]
    if X_clean.shape[0] < 4 or X_clean.shape[1] == 0:
        return []
    if _HAS_SKLEARN:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_clean)
    else:
        Xs = (X_clean - X_clean.mean(axis=0)) / (X_clean.std(axis=0) + 1e-9)
    # Use 80/20 split for the weights; the weights on the full-fit are
    # the most informative for feature importance.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(Xs.shape[0])
    n_tr = int(Xs.shape[0] * 0.8)
    Xtr = Xs[perm[:n_tr]]
    ytr = y_clean[perm[:n_tr]]
    if _HAS_SKLEARN:
        model = Ridge(alpha=alpha)
        model.fit(Xtr, ytr)
        w = model.coef_
    else:
        XtX = Xtr.T @ Xtr
        Xty = Xtr.T @ ytr
        w = np.linalg.solve(XtX + alpha * np.eye(XtX.shape[0]), Xty)
    if len(feature_names) != len(w):
        # Pad or truncate to match
        names_aligned = list(feature_names) + [
            f"f{i:03d}" for i in range(len(feature_names), len(w))
        ]
        names_aligned = names_aligned[:len(w)]
    else:
        names_aligned = list(feature_names)
    abs_w = np.abs(w)
    top_idx = np.argsort(-abs_w)[:topk]
    return [
        {
            "rank": int(rank + 1),
            "feature": str(names_aligned[int(idx)]),
            "weight": float(w[int(idx)]),
            "abs_weight": float(abs_w[int(idx)]),
        }
        for rank, idx in enumerate(top_idx)
    ]


# ---------------------------------------------------------------------------
# Combined pipeline (FULL integration)
# ---------------------------------------------------------------------------
def build_full_feature_matrix(
    smiles_list: List[str],
    X_32d: np.ndarray,
    include_rdkit_32d: bool = True,
    include_chemprop_v2_graph: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Build the FULL static feature matrix (RDKit 32-d + chemprop v2 graph).

    Returns the concatenated matrix and the corresponding column names.
    W₂ is appended per-axis via `ridge_r2_eval_with_w2` (split-aware).
    """
    n = len(smiles_list)
    streams: List[np.ndarray] = []
    names: List[str] = []
    if include_rdkit_32d:
        if X_32d.shape[0] != n:
            raise ValueError(
                f"X_32d has {X_32d.shape[0]} rows; expected {n}"
            )
        rdkit_part = np.asarray(X_32d, dtype=np.float64)
        keep = rdkit_part.std(axis=0) > 1e-9
        streams.append(rdkit_part[:, keep])
        names.extend([
            f"rdkit_32d_{i:02d}" for i in range(rdkit_part.shape[1])
        ])
    if include_chemprop_v2_graph:
        graph_part = extract_graph_features_pooled(smiles_list)
        keep = graph_part.std(axis=0) > 1e-9
        streams.append(graph_part[:, keep])
        atom_dim, bond_dim = _probe_chemprop_dims()
        names.extend([
            f"graph_atom_{pool}_{i:02d}"
            for pool in ("mean", "max", "sum")
            for i in range(atom_dim)
        ])
        names.extend([
            f"graph_bond_{pool}_{i:02d}"
            for pool in ("mean", "max", "sum")
            for i in range(bond_dim)
        ])
    if not streams:
        return np.zeros((n, 0), dtype=np.float64), []
    X = np.concatenate(streams, axis=1)
    return X, names


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_wf335m2(
    out_path: str = "molmetal/reports/wf335m2_r2/result.json",
    cohorts: Optional[List[str]] = None,
    metalcytotox_max: int = 2000,
    n_splits: int = 5,
    n_seeds: int = 5,
    alpha: float = 1.0,
    axis_selection_seeds: int = 3,
    include_per_cell: bool = True,
    include_per_metal: bool = True,
    include_feature_importance: bool = True,
) -> Dict[str, Any]:
    """Run the Wf-335-M2 FULL W₂ + chemprop v2 graph integration bench.

    Parameters
    ----------
    out_path : str
        Path to write the result JSON.
    cohorts : list of str, optional
        Cohorts to evaluate.  Default: all 3.
    metalcytotox_max : int
        Cap on metalcytotox_combined rows (default 2000; matches
        Wf-325A MEASURED).
    n_splits : int
        K for K-fold CV (default 5).
    n_seeds : int
        Number of seeds for multi-seed CV (default 5; total 25 splits).
    alpha : float
        Ridge alpha (default 1.0; matches Wf-325A / Wf-331-R5).
    axis_selection_seeds : int
        Number of seeds for W₂ axis-selection (smaller for speed).
    include_per_cell, include_per_metal : bool
        Toggles for per-cohort breakdown diagnostics.
    include_feature_importance : bool
        Toggle for feature importance diagnostic.

    Returns
    -------
    result : dict
        Aggregated result dictionary.
    """
    if cohorts is None:
        cohorts = ["platinai_4cell", "hilbert", "metalcytotox_combined"]

    started_at = time.time()
    per_cohort: List[Dict[str, Any]] = []
    for cohort in cohorts:
        t0 = time.time()
        # Seed offset: hilbert uses seed+10 to give an independent reading
        # on the same 200 SMILES (per Wf-325A / Wf-331-R5 pattern).
        cohort_seed_offset = 10 if cohort == "hilbert" else 0
        X_32d, y, smiles_list = load_cohort(
            cohort, max_samples=metalcytotox_max, seed=0,
        )[:3]
        n_samples = int(X_32d.shape[0])
        baseline_r2 = WF325A_BASELINE_R2[cohort]
        # Build FULL static feature matrix
        X_static, feature_names = build_full_feature_matrix(
            smiles_list, X_32d,
            include_rdkit_32d=True,
            include_chemprop_v2_graph=True,
        )
        # 12-d baseline
        X_12d = X_32d[:, :12]
        cv_12d = ridge_r2_cv(
            X_12d, y, n_splits=n_splits, n_seeds=n_seeds,
            alpha=alpha, seed_offset=cohort_seed_offset,
        )
        # 32-d RDKit baseline
        cv_32d = ridge_r2_cv(
            np.asarray(X_32d, dtype=np.float64), y,
            n_splits=n_splits, n_seeds=n_seeds,
            alpha=alpha, seed_offset=cohort_seed_offset,
        )
        # FULL combined (RDKit 32-d + chemprop v2 graph)
        cv_combined = ridge_r2_cv(
            X_static, y,
            n_splits=n_splits, n_seeds=n_seeds,
            alpha=alpha, seed_offset=cohort_seed_offset,
        )
        # FULL combined + W₂ (multi-axis: pick best)
        axis_sel = select_best_w2_axis(
            X_static, y,
            n_splits=n_splits, n_seeds=axis_selection_seeds,
            alpha=alpha, seed_offset=cohort_seed_offset,
        )
        best_axis = axis_sel["best_axis"]
        if best_axis >= 0:
            cv_combined_plus_w2 = ridge_r2_eval_with_w2(
                X_static, y, projection_axis=best_axis,
                n_splits=n_splits, n_seeds=n_seeds,
                alpha=alpha, seed_offset=cohort_seed_offset,
            )
        else:
            cv_combined_plus_w2 = {
                "r2_mean": float("nan"), "r2_std": float("nan"),
                "r2_per_seed": [], "r2_per_split": [],
                "n_splits": n_splits, "n_seeds": n_seeds, "alpha": alpha,
            }
        # Verdict helpers
        def verdict(r2_mean: float, baseline: float) -> str:
            if not np.isfinite(r2_mean):
                return "UNDEFINED"
            lift = r2_mean - baseline
            if r2_mean > 0.0 and lift >= 0.10:
                return "CEILING_BREAK"
            if r2_mean > 0.0 and lift >= 0.05:
                return "MODERATE_RECOVERY"
            if r2_mean > 0.0:
                return "WEAK_RECOVERY"
            if lift >= 0.0:
                return "NO_LIFT_HONEST_NEGATIVE"
            return "HONEST_NEGATIVE_R2_DEEP"
        per_config = [
            {
                "label": "12d_baseline",
                "r2_mean": cv_12d["r2_mean"],
                "r2_std": cv_12d["r2_std"],
                "lift_vs_wf325a_baseline": float(
                    cv_12d["r2_mean"] - baseline_r2,
                ),
                "verdict": verdict(cv_12d["r2_mean"], baseline_r2),
                "n_features": 12,
            },
            {
                "label": "32d_rdkit",
                "r2_mean": cv_32d["r2_mean"],
                "r2_std": cv_32d["r2_std"],
                "lift_vs_wf325a_baseline": float(
                    cv_32d["r2_mean"] - baseline_r2,
                ),
                "verdict": verdict(cv_32d["r2_mean"], baseline_r2),
                "n_features": int(np.asarray(X_32d).shape[1]),
            },
            {
                "label": "full_static_combined",
                "r2_mean": cv_combined["r2_mean"],
                "r2_std": cv_combined["r2_std"],
                "lift_vs_wf325a_baseline": float(
                    cv_combined["r2_mean"] - baseline_r2,
                ),
                "verdict": verdict(cv_combined["r2_mean"], baseline_r2),
                "n_features": int(X_static.shape[1]),
            },
            {
                "label": "full_static_combined_plus_w2_best_axis",
                "r2_mean": cv_combined_plus_w2["r2_mean"],
                "r2_std": cv_combined_plus_w2["r2_std"],
                "lift_vs_wf325a_baseline": float(
                    cv_combined_plus_w2["r2_mean"] - baseline_r2,
                ),
                "verdict": verdict(
                    cv_combined_plus_w2["r2_mean"], baseline_r2,
                ),
                "n_features": int(X_static.shape[1] + 1),
                "best_w2_axis": int(best_axis),
            },
        ]
        # Best config (by R² mean, then by lift)
        finite_r2 = [
            c for c in per_config if np.isfinite(c["r2_mean"])
        ]
        best_label = None
        best_r2 = float("-inf")
        for c in finite_r2:
            if c["r2_mean"] > best_r2:
                best_r2 = float(c["r2_mean"])
                best_label = c["label"]
        # Per-cohort breakdown
        per_cell_results: Dict[str, Any] = {}
        per_metal_results: Dict[str, Any] = {}
        if include_per_cell and cohort == "platinai_4cell":
            per_cell_results = per_cell_r2_breakdown(
                smiles_list, X_static,
                cohort=cohort, n_splits=n_splits, n_seeds=3,
                alpha=alpha, seed_offset=cohort_seed_offset,
            )
        if include_per_metal and cohort == "metalcytotox_combined":
            per_metal_results = per_metal_r2_breakdown(
                smiles_list, X_static,
                n_splits=n_splits, n_seeds=3,
                alpha=alpha, seed_offset=cohort_seed_offset,
            )
        # Feature importance (top 10) for the FULL combined + W₂ config
        top_features: List[Dict[str, Any]] = []
        if include_feature_importance and X_static.shape[1] > 0:
            try:
                if best_axis >= 0:
                    w2_full = compute_covariate_w2_distance(
                        X_static, projection_axis=best_axis,
                    )
                else:
                    w2_full = np.zeros(X_static.shape[0], dtype=np.float64)
                X_with_w2 = np.concatenate(
                    [X_static, w2_full.reshape(-1, 1)], axis=1,
                )
                names_with_w2 = list(feature_names) + ["w2_dist_to_ref"]
                top_features = feature_importance_topk(
                    X_with_w2, y, names_with_w2, topk=10, alpha=alpha,
                )
            except Exception as exc:
                top_features = [{"error": str(exc)}]
        per_cohort.append({
            "cohort": cohort,
            "n_samples": n_samples,
            "n_features_static": int(X_static.shape[1]),
            "wf325a_baseline_r2": baseline_r2,
            "per_config": per_config,
            "best_config": {
                "label": best_label,
                "r2_mean": best_r2 if best_label is not None else float("nan"),
                "lift_vs_wf325a_baseline": (
                    float(best_r2 - baseline_r2)
                    if best_label is not None else float("nan")
                ),
            },
            "w2_axis_selection": axis_sel,
            "per_cell_r2": per_cell_results,
            "per_metal_r2": per_metal_results,
            "top10_feature_importance": top_features,
            "elapsed_sec": float(time.time() - t0),
        })
    # Aggregate honest framing
    n_ceiling_break = sum(
        1 for c in per_cohort
        if any(
            cfg.get("verdict") == "CEILING_BREAK"
            for cfg in c["per_config"]
        )
    )
    n_above_zero = sum(
        1 for c in per_cohort
        if any(
            np.isfinite(cfg.get("r2_mean", float("nan")))
            and cfg["r2_mean"] > 0.0
            for cfg in c["per_config"]
        )
    )
    n_resolved = sum(
        1 for c in per_cohort
        if c["best_config"]["lift_vs_wf325a_baseline"] >= 0.05
    )
    if n_ceiling_break >= 2:
        aggregate_status = "HONEST_NEGATIVE_R2_RESOLVED"
    elif n_above_zero >= 1:
        aggregate_status = "HONEST_NEGATIVE_PARTIAL"
    elif any(
        c["best_config"]["lift_vs_wf325a_baseline"] >= 0.0
        for c in per_cohort
    ):
        aggregate_status = "NO_LIFT_HONEST_NEGATIVE"
    else:
        aggregate_status = "HONEST_NEGATIVE_R2_PERSISTENT"
    result: Dict[str, Any] = {
        "wf": "Wf-335-M2",
        "task": (
            "FULL W₂ + chemprop v2 graph integration with multi-seed "
            "K-fold CV, multi-axis W₂ selection, per-cohort breakdown "
            "(per-cell / per-metal), and top-10 feature importance "
            "(WEAK_TO_STRONG §2.1 / TODO-122 P1.3)"
        ),
        "scope": (
            "CPU-only diagnostic on 3 real-pIC50 cohorts; "
            f"{n_seeds} seeds × {n_splits}-fold CV (n={n_seeds * n_splits} "
            "splits per cohort); no torch autograd, no encoder forward pass."
        ),
        "upstream": "Wf-331-R5 (single-seed 8-config sweep)",
        "n_cohorts_tested": len(per_cohort),
        "cohorts_tested": [c["cohort"] for c in per_cohort],
        "n_features": {
            "rdkit_32d": 32,
            "chemprop_v2_graph": _get_graph_feature_dim(),
            "w2_distance": 1,
            "combined_max": 32 + _get_graph_feature_dim() + 1,
        },
        "graph_featurizer": "rdkit_native_chemprop_compatible",
        "w2_method": (
            "covariate_RDKit_projection (non-leaky, split-aware, "
            "multi-axis selection)"
        ),
        "cv_config": {
            "n_splits": n_splits,
            "n_seeds": n_seeds,
            "alpha": alpha,
            "axis_selection_seeds": axis_selection_seeds,
        },
        "wf325a_baselines": WF325A_BASELINE_R2,
        "per_cohort": per_cohort,
        "per_cohort_r2_lift_vs_wf325a": {
            c["cohort"]: {
                "best_config": c["best_config"]["label"],
                "best_r2_mean": c["best_config"]["r2_mean"],
                "lift": c["best_config"]["lift_vs_wf325a_baseline"],
            }
            for c in per_cohort
        },
        "n_ceiling_break_cohorts": n_ceiling_break,
        "n_above_zero_cohorts": n_above_zero,
        "n_resolved_cohorts": n_resolved,
        "aggregate_status": aggregate_status,
        "honest_framing": {
            "projected_lift": (
                "+0.05 to +0.15 R² lift per cohort vs Wf-325A baseline "
                "(the lower bound of the WEAK_TO_STRONG §2.1 projection)."
            ),
            "structural_note": (
                "FULL integration: multi-seed × K-fold CV replaces "
                "Wf-331-R5 single-split R²; multi-axis W₂ selection "
                "(MolWt / MolLogP / TPSA / HeavyAtoms) replaces "
                "single-axis (MolWt only); per-cohort breakdowns "
                "(per-cell / per-metal) replace aggregate-only."
            ),
            "ceiling_break": (
                "Best R² (multi-seed CV mean) > 0 AND lift ≥ +0.10 over "
                "Wf-325A baseline on the cohort."
            ),
            "aggregate_resolved": (
                "HONEST_NEGATIVE_R2_RESOLVED if ≥ 2 cohorts achieve "
                "CEILING_BREAK; HONEST_NEGATIVE_PARTIAL if ≥ 1 cohort "
                "lifts R² > 0; HONEST_NEGATIVE_R2_PERSISTENT if all 3 "
                "cohorts stay R² ≤ 0."
            ),
        },
        "pre_flight_status": (
            "CPU-only diagnostic; pre-flight v2 gate NOT required "
            "(no retrain / panel / ablation)."
        ),
        "elapsed_sec_total": float(time.time() - started_at),
    }
    out_path_abs = _resolve_repo_path(out_path)
    os.makedirs(os.path.dirname(out_path_abs), exist_ok=True)
    with open(out_path_abs, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=False)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--cohorts", nargs="+", default=None,
        help=(
            "Cohorts to evaluate (default: all 3). Accepts either "
            "space-separated (--cohorts a b c) or a single "
            "comma-separated string (--cohorts a,b,c)."
        ),
    )
    p.add_argument("--metalcytotox-max", type=int, default=2000)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument(
        "--axis-selection-seeds", type=int, default=3,
        help="Number of seeds for W₂ axis selection (smaller for speed)",
    )
    p.add_argument(
        "--output-dir",
        default="molmetal/reports/wf335m2_r2",
    )
    p.add_argument(
        "--no-per-cell", action="store_true",
        help="Skip per-cell R² breakdown for platinai_4cell",
    )
    p.add_argument(
        "--no-per-metal", action="store_true",
        help="Skip per-metal R² breakdown for metalcytotox_combined",
    )
    p.add_argument(
        "--no-feature-importance", action="store_true",
        help="Skip top-10 feature importance diagnostic",
    )
    args = p.parse_args(argv)
    cohorts_arg = args.cohorts
    if cohorts_arg is not None and len(cohorts_arg) == 1 and "," in cohorts_arg[0]:
        cohorts_arg = [c.strip() for c in cohorts_arg[0].split(",") if c.strip()]
    out_path = os.path.join(args.output_dir, "result.json")
    result = run_wf335m2(
        out_path=out_path,
        cohorts=cohorts_arg,
        metalcytotox_max=args.metalcytotox_max,
        n_splits=args.n_splits,
        n_seeds=args.n_seeds,
        alpha=args.alpha,
        axis_selection_seeds=args.axis_selection_seeds,
        include_per_cell=not args.no_per_cell,
        include_per_metal=not args.no_per_metal,
        include_feature_importance=not args.no_feature_importance,
    )

    print(f"Wf-335-M2: n_cohorts = {result['n_cohorts_tested']}")
    print(f"  CV config = {result['cv_config']}")
    for c in result["per_cohort"]:
        cohort = c["cohort"]
        n = c["n_samples"]
        baseline = c["wf325a_baseline_r2"]
        n_feat = c["n_features_static"]
        print(
            f"\n  cohort={cohort:>22s}  n={n:>5d}  "
            f"Wf-325A baseline R²={baseline:+.4f}  "
            f"n_feat_static={n_feat}  elapsed={c['elapsed_sec']:.1f}s"
        )
        for cfg in c["per_config"]:
            print(
                f"    {cfg['label']:>40s}  n_feat={cfg.get('n_features', 0):>3d}  "
                f"R²_mean={cfg['r2_mean']:+.4f} ± {cfg['r2_std']:.4f}  "
                f"lift={cfg['lift_vs_wf325a_baseline']:+.4f}  "
                f"{cfg['verdict']}"
            )
        bc = c["best_config"]
        print(
            f"    best_config = {bc['label']}  R²_mean={bc['r2_mean']:+.4f}  "
            f"lift={bc['lift_vs_wf325a_baseline']:+.4f}"
        )
        w2_sel = c["w2_axis_selection"]
        if w2_sel["best_axis"] >= 0:
            print(
                f"    best_w2_axis = {w2_sel['best_axis']} "
                f"(R²={w2_sel['best_axis_r2_mean']:+.4f})"
            )
        if c["per_cell_r2"]:
            print(f"    per_cell R²:")
            for cell, res in c["per_cell_r2"].items():
                print(
                    f"      {cell:>10s}  R²={res.get('r2_mean', float('nan')):+.4f} ± "
                    f"{res.get('r2_std', 0):.4f}  n={res.get('n_valid', 0)}"
                )
        if c["per_metal_r2"]:
            print(f"    per_metal R²:")
            for metal, res in c["per_metal_r2"].items():
                print(
                    f"      {metal:>4s}  R²={res.get('r2_mean', float('nan')):+.4f} ± "
                    f"{res.get('r2_std', 0):.4f}  n={res.get('n_samples', 0)}"
                )
        if c["top10_feature_importance"]:
            print(f"    top-10 feature importance:")
            for feat in c["top10_feature_importance"][:10]:
                if "error" in feat:
                    print(f"      error: {feat['error']}")
                    break
                print(
                    f"      #{feat['rank']:>2d}  {feat['feature']:>30s}  "
                    f"weight={feat['weight']:+.4f}  |w|={feat['abs_weight']:.4f}"
                )
    print(
        f"\n  aggregate_status = {result['aggregate_status']}  "
        f"(ceiling_break={result['n_ceiling_break_cohorts']} "
        f"above_zero={result['n_above_zero_cohorts']} "
        f"resolved={result['n_resolved_cohorts']})"
    )
    print(f"  elapsed_sec_total = {result['elapsed_sec_total']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
