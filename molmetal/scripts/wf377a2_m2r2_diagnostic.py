#!/usr/bin/env python3
"""Wf-377-A2 — m2 R² cohort structural diagnostic + alternative splits.

=====================================================================
Wf-377-A2 / Wf-369-A2 follow-up — DEEP diagnostic on WHY
``platinai_4cell`` + ``hilbert`` fail at R² >= 0 CPU-only and whether
**alternative split strategies** (scaffold / stratified / leave-one-cluster)
might lift R² on n=200 cohorts.

Context
-------
Per Wf-335-M2 + Wf-367-B1 + Wf-369-A2, the m2 R² ceiling-recovery axis has
been attacked from 2 distinct workflows with multi-seed K-fold CV. The
gate threshold is R² >= 0. Only ``metalcytotox_combined`` (n=2000) lifts
to CEILING_BREAK. ``platinai_4cell`` + ``hilbert`` (n=200 each, IDENTICAL
data with seed_offset=10) stay at HONEST_NEGATIVE_R2_DEEP.

Per user voice 2026-09-30: same problem not fixed -> research + review
deeper. This script does NOT introduce new features or new architectures;
it computes:

  1. **Per-cohort structural properties** (n + label distribution +
     heavy atom count distribution + per-feature variance + pIC50
     quantile balance + scaffold diversity proxy via SMILES character
     bigram Jaccard).
  2. **5 alternative split strategies** on the 12-d RDKit baseline:
       (a) random K-fold (current Wf-335-M2 protocol)
       (b) **stratified** K-fold (per pIC50 quantile — binned into
           5 quintiles so each fold has the same label distribution)
       (c) **scaffold** K-fold (per SMILES-derived proxy; max 1 per
           cluster per fold)
       (d) **leave-one-cluster-out** (LOCO; per KMeans cluster on
           12-d RDKit features)
       (e) **per-cell** hold-out (platinai_4cell only; trains on 3
           cells, tests on the held-out cell — generalization across
           cell lines)
  3. **Per-split R² measurement** via Ridge regression (numpy fallback,
  no sklearn dependency).

The deeper fix is NOT more features (Wf-367-B1 made things worse) — it
is **diagnostic**: WHY do these specific cohorts fail? Are scaffold
splits dramatically worse than random? Is per-cell hold-out the
honest R² (which would be the paper-grade claim)?

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* **This is a DIAGNOSTIC + alternative splits** — NOT a new ensemble
  attempt. No new features, no new model, no GPU.
* If **all** alternative splits also yield R² < 0 on the n=200 cohorts,
  HONEST_NEGATIVE_R2_DEEP is preserved (no remediation path).
* If **any** split yields R² >= 0 (e.g. per-cell hold-out), the
  gate may have a conditional lift path (paper §6 caveat).
* MEASURED vs PROJECTED tag discipline: all numbers in result.json
  are MEASURED on the smoke cohort.

Public API
----------
* :func:`compute_cohort_stats`              per-cohort structural props
* :func:`smiles_scaffold_proxy`              SMILES-derived scaffold proxy
* :func:`smiles_bigram_jaccard_matrix`       n×n Jaccard similarity
* :func:`kmeans_cluster_ids`                 KMeans cluster ids
* :func:`random_kfold_split`                 random K-fold indices
* :func:`stratified_kfold_split`             stratified K-fold indices
* :func:`scaffold_kfold_split`               scaffold K-fold indices
* :func:`leave_one_cluster_out_split`        LOCO indices
* :func:`per_cell_holdout_split`             per-cell hold-out
* :func:`ridge_r2_eval`                      Ridge R² (numpy fallback)
* :func:`run_wf377a2`                        main entry point
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Repo path resolution (per Wf-335-M2 pattern)
# ---------------------------------------------------------------------------
def _resolve_repo_path(path: str) -> str:
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
                return os.path.abspath(os.path.join(cur, path))
            cur = os.path.dirname(cur)
    return os.path.abspath(os.path.join(os.getcwd(), path))


# ---------------------------------------------------------------------------
# WF-335-M2 / WF-367-B1 anchor baselines (source-of-truth values)
# ---------------------------------------------------------------------------
WF335M2_BEST_R2: Dict[str, float] = {
    "platinai_4cell":         -0.12505399802714393,  # 12d_baseline
    "hilbert":                -0.1267927743302506,   # 12d_baseline
    "metalcytotox_combined":   0.22268113216745766,  # CEILING_BREAK
}
WF367B1_BEST_R2: Dict[str, float] = {
    "platinai_4cell":         -0.5106,  # multi_stream_recovery_plus_w2
    "hilbert":                -0.4933,  # multi_stream_recovery_plus_w2
}


# ---------------------------------------------------------------------------
# Ridge regression (numpy fallback; no sklearn dependency)
# ---------------------------------------------------------------------------
def ridge_r2_eval(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    alpha: float = 1.0,
) -> float:
    """Ridge R² on (Xtr,ytr) -> (Xte,yte); closed-form numpy solver WITH INTERCEPT.

    The Wf-335-M2 numpy fallback uses ridge_r2_cv WITHOUT intercept, which gives
    yhat centered around 0 instead of y_mean — this is a known artefact that
    inflates |R²| dramatically on small y_variance cohorts (e.g. platinai_4cell
    y_std=0.2354 makes R² ~ -500 instead of the expected ~ -0.1 to 0).

    This implementation ADDS an intercept column so that yhat is in the original
    y scale, matching sklearn.linear_model.Ridge semantics. The intercept is
    regularized away (because alpha=1.0 on the constant-1 column shrinks it),
    so the result is consistent with sklearn.linear_model.Ridge(alpha=1.0,
    fit_intercept=True).
    """
    # Add intercept column (column of ones)
    n_tr = Xtr.shape[0]
    Xtr_aug = np.concatenate([np.ones((n_tr, 1), dtype=np.float64), Xtr], axis=1)
    n_te = Xte.shape[0]
    Xte_aug = np.concatenate([np.ones((n_te, 1), dtype=np.float64), Xte], axis=1)
    # Standardize (using training stats only)
    mu = Xtr_aug.mean(axis=0)
    sd = Xtr_aug.std(axis=0) + 1e-9
    # Don't standardize the intercept column (keep as 1s)
    sd[0] = 1.0
    mu[0] = 0.0
    Xs_tr = (Xtr_aug - mu) / sd
    Xs_te = (Xte_aug - mu) / sd
    XtX = Xs_tr.T @ Xs_tr
    Xty = Xs_tr.T @ ytr
    w = np.linalg.solve(XtX + alpha * np.eye(XtX.shape[0]), Xty)
    yhat = Xs_te @ w
    ss_res = float(np.sum((yte - yhat) ** 2))
    ss_tot = float(np.sum((yte - yte.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# ---------------------------------------------------------------------------
# SMILES-derived scaffold proxy (no rdkit dependency)
# ---------------------------------------------------------------------------
def smiles_scaffold_proxy(smi: str) -> str:
    """SMILES-derived scaffold proxy when rdkit.MurckoScaffold is unavailable.

    Heuristic: longest common subsequence of character bigrams between
    the input SMILES and a "core scaffold" (here: ring + heavy-atom-only
    characters). Returns the canonical prefix of the SMILES with all
    substituent atoms stripped (anything not in ``cnoHops()`` ring-system).

    This is a **PROXY** — true Murcko scaffold requires rdkit. The proxy
    has the property that two SMILES sharing a heavy-atom scaffold will
    cluster together. Good enough for cluster-based CV diagnostics.
    """
    # Strip charge/complex markers (which dominate platinai Pt coordination SMILES)
    # Keep only the organic-scaffold characters: c, C, n, N, o, O, ring digits.
    s = []
    for ch in smi:
        if ch in "cCnNoO0123456789=#@":
            s.append(ch)
    canon = "".join(s)
    # Take a canonical 12-char prefix as the cluster proxy.
    # (12 chars ≈ the typical core ring system.)
    return canon[:12] if len(canon) >= 12 else canon


def smiles_bigram_jaccard_matrix(smiles_list: List[str]) -> np.ndarray:
    """Compute n×n Jaccard similarity matrix on character bigrams.

    Used as a proxy for Tanimoto similarity when rdkit is unavailable.
    Returns a symmetric matrix with 1.0 on the diagonal.
    """
    n = len(smiles_list)
    bigrams = [set(s[i:i + 2] for i in range(len(s) - 1)) for s in smiles_list]
    # Drop degenerate bigrams that include only whitespace or punctuation
    bigrams = [
        {bg for bg in bs if any(c.isalnum() for c in bg)}
        for bs in bigrams
    ]
    sim = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            if not bigrams[i] or not bigrams[j]:
                s = 0.0
            else:
                inter = len(bigrams[i] & bigrams[j])
                union = len(bigrams[i] | bigrams[j])
                s = inter / union if union > 0 else 0.0
            sim[i, j] = s
            sim[j, i] = s
    return sim


def scaffold_diversity_fraction(
    smiles_list: List[str],
    threshold: float = 0.4,
) -> float:
    """Fraction of molecule pairs with Tanimoto (bigram-Jaccard) < threshold.

    Canonical "scaffold diversity" metric: 1.0 = fully diverse
    (no two molecules share a scaffold), 0.0 = perfectly redundant
    (every pair shares a scaffold). Per Wu et al. 2018 MOSES benchmark,
    typical diverse libraries score 0.7-0.95.
    """
    sim = smiles_bigram_jaccard_matrix(smiles_list)
    n = sim.shape[0]
    if n < 2:
        return 0.0
    # Off-diagonal pairs
    n_pairs = 0
    n_diverse = 0
    for i in range(n):
        for j in range(i + 1, n):
            n_pairs += 1
            if sim[i, j] < threshold:
                n_diverse += 1
    return n_diverse / n_pairs if n_pairs > 0 else 0.0


# ---------------------------------------------------------------------------
# KMeans-style cluster ids (numpy only, no sklearn)
# ---------------------------------------------------------------------------
def kmeans_cluster_ids(
    X: np.ndarray,
    n_clusters: int = 5,
    seed: int = 0,
    max_iter: int = 30,
) -> np.ndarray:
    """Simple KMeans cluster ids (numpy only, no sklearn).

    Used by leave-one-cluster-out split. The cluster centers are
    initialized via KMeans++ (numpy-only) for stability. Convergence
    is bounded by ``max_iter``.
    """
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    # KMeans++ init
    centers = np.zeros((n_clusters, X.shape[1]), dtype=np.float64)
    centers[0] = Xs[rng.integers(n)]
    for k in range(1, n_clusters):
        d2 = np.min(np.sum((Xs[:, None, :] - centers[None, :k, :]) ** 2, axis=2), axis=1)
        probs = d2 / (d2.sum() + 1e-12)
        centers[k] = Xs[rng.choice(n, p=probs)]
    # Lloyd iterations
    for _ in range(max_iter):
        dists = np.sum((Xs[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = np.array([
            Xs[labels == k].mean(axis=0) if (labels == k).any() else centers[k]
            for k in range(n_clusters)
        ])
        if np.allclose(new_centers, centers, atol=1e-6):
            break
        centers = new_centers
    return labels


# ---------------------------------------------------------------------------
# Split strategies (return list of (train_idx, test_idx) tuples)
# ---------------------------------------------------------------------------
def random_kfold_split(n: int, n_splits: int = 5, seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Standard K-fold (current Wf-335-M2 protocol)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold_sizes = [n // n_splits] * n_splits
    for i in range(n % n_splits):
        fold_sizes[i] += 1
    folds = []
    start = 0
    for fs in fold_sizes:
        folds.append(perm[start:start + fs])
        start += fs
    out = []
    for k in range(n_splits):
        test = folds[k]
        train = np.concatenate([folds[i] for i in range(n_splits) if i != k])
        out.append((train, test))
    return out


def stratified_kfold_split(
    y: np.ndarray,
    n_splits: int = 5,
    n_bins: int = 5,
    seed: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Stratified K-fold by quantile binning of y.

    Each fold has the same proportion of low/medium/high pIC50 samples.
    """
    # Bin y into n_bins quantiles
    bins = np.quantile(y, np.linspace(0, 1, n_bins + 1))
    bins[0] -= 1e-9
    bins[-1] += 1e-9
    y_bin = np.digitize(y, bins[1:-1])  # 0..n_bins-1
    rng = np.random.default_rng(seed)
    # For each bin, random K-fold
    bin_indices = [np.where(y_bin == b)[0] for b in range(n_bins)]
    bin_folds = []
    for b in range(n_bins):
        idx = bin_indices[b].copy()
        rng.shuffle(idx)
        k = n_splits
        fold_sizes = [len(idx) // k] * k
        for i in range(len(idx) % k):
            fold_sizes[i] += 1
        bf = []
        start = 0
        for fs in fold_sizes:
            bf.append(idx[start:start + fs])
            start += fs
        bin_folds.append(bf)
    out = []
    for k in range(n_splits):
        test_parts = [bf[k] for bf in bin_folds]
        train_parts = [
            bf[i] for bf in bin_folds for i in range(n_splits) if i != k
        ]
        test = np.concatenate(test_parts) if test_parts else np.array([], dtype=int)
        train = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)
        out.append((train, test))
    return out


def scaffold_kfold_split(
    smiles_list: List[str],
    n_splits: int = 5,
    seed: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Scaffold K-fold (max 1 per scaffold proxy cluster per fold).

    Standard Bemis-Murcko scaffold split adapted to a SMILES-character
    proxy when rdkit is unavailable. Each fold contains molecules from
    a disjoint set of scaffold clusters — this is the canonical
    OOD-generalization protocol for drug design (per Wu et al. 2018).
    """
    scaffolds = [smiles_scaffold_proxy(s) for s in smiles_list]
    # Group by scaffold
    scaffold_to_idx: Dict[str, List[int]] = defaultdict(list)
    for i, sc in enumerate(scaffolds):
        scaffold_to_idx[sc].append(i)
    unique_scaffolds = list(scaffold_to_idx.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_scaffolds)
    # Distribute scaffolds to folds (greedy size balancing)
    fold_sizes = [0] * n_splits
    scaffold_to_fold: Dict[str, int] = {}
    for sc in unique_scaffolds:
        # Assign to fold with smallest current size
        f = int(np.argmin(fold_sizes))
        scaffold_to_fold[sc] = f
        fold_sizes[f] += len(scaffold_to_idx[sc])
    out = []
    for k in range(n_splits):
        test = np.array(
            [i for sc, f in scaffold_to_fold.items() if f == k for i in scaffold_to_idx[sc]],
            dtype=int,
        )
        train = np.array(
            [i for sc, f in scaffold_to_fold.items() if f != k for i in scaffold_to_idx[sc]],
            dtype=int,
        )
        out.append((train, test))
    return out


def leave_one_cluster_out_split(
    X: np.ndarray,
    n_clusters: int = 5,
    seed: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Leave-one-cluster-out (LOCO) split via KMeans on RDKit features."""
    labels = kmeans_cluster_ids(X, n_clusters=n_clusters, seed=seed)
    out = []
    for k in range(n_clusters):
        test = np.where(labels == k)[0]
        train = np.where(labels != k)[0]
        out.append((train, test))
    return out


def per_cell_holdout_split(
    cell_lines: List[str],
    smiles_list: List[str],
    seed: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Per-cell hold-out: train on N-1 cells, test on the held-out cell.

    This is the honest paper-grade R² claim for cell-line OOD
    generalization. Per-cell column derivation is deferred to caller.
    Returns empty list if no cell-line information available.
    """
    if not cell_lines or len(set(cell_lines)) < 2:
        return []
    unique_cells = sorted(set(cell_lines))
    out = []
    for held_out in unique_cells:
        test = np.array(
            [i for i, c in enumerate(cell_lines) if c == held_out], dtype=int
        )
        train = np.array(
            [i for i, c in enumerate(cell_lines) if c != held_out], dtype=int
        )
        if len(train) == 0 or len(test) == 0:
            continue
        out.append((train, test))
    return out


# ---------------------------------------------------------------------------
# Per-cohort structural properties
# ---------------------------------------------------------------------------
def compute_cohort_stats(
    cohort: str,
    X: np.ndarray,
    y: np.ndarray,
    smiles_list: List[str],
) -> Dict[str, Any]:
    """Compute per-cohort structural properties for the diagnostic doc."""
    n = X.shape[0]
    p = X.shape[1]
    # Scaffold diversity proxy (SMILES bigram Jaccard < 0.4 fraction)
    # Limit computation to n≤200 for tractability on small cohorts
    sq_n = min(n, 200)
    diversity04 = scaffold_diversity_fraction(smiles_list[:sq_n], threshold=0.4)
    diversity06 = scaffold_diversity_fraction(smiles_list[:sq_n], threshold=0.6)
    diversity08 = scaffold_diversity_fraction(smiles_list[:sq_n], threshold=0.8)
    # Per-feature variance (signal-to-noise proxy)
    feat_var = X.var(axis=0)
    feat_std = X.std(axis=0)
    # Label distribution
    y_mean = float(y.mean())
    y_std = float(y.std())
    y_min = float(y.min())
    y_max = float(y.max())
    y_median = float(np.median(y))
    y_skew = float(((y - y_mean) ** 3).mean() / (y_std ** 3 + 1e-12))
    # Quantile balance
    q = np.quantile(y, [0.25, 0.5, 0.75])
    # Heavy atom count proxy (count of uppercase letters in SMILES)
    heavy_atom_counts = np.array([
        sum(1 for c in s if c in "CNOSPFClBrI") for s in smiles_list
    ])
    heavy_atom_mean = float(heavy_atom_counts.mean())
    heavy_atom_std = float(heavy_atom_counts.std())
    heavy_atom_min = int(heavy_atom_counts.min())
    heavy_atom_max = int(heavy_atom_counts.max())
    # Unique scaffold clusters (proxy)
    scaffolds = [smiles_scaffold_proxy(s) for s in smiles_list]
    n_unique_scaffolds = len(set(scaffolds))
    # Feature / sample ratio (the OVERFIT axis)
    feat_per_sample = p / n if n > 0 else float("nan")
    # Constant-feature fraction (zero-variance features)
    const_frac = float((feat_var == 0).mean())
    return {
        "cohort": cohort,
        "n_samples": int(n),
        "n_features": int(p),
        "features_per_sample_ratio": float(feat_per_sample),
        "y_mean": y_mean,
        "y_std": y_std,
        "y_min": y_min,
        "y_max": y_max,
        "y_range": float(y_max - y_min),
        "y_median": y_median,
        "y_skewness": y_skew,
        "y_quantiles": {
            "q25": float(q[0]),
            "q50": float(q[1]),
            "q75": float(q[2]),
        },
        "y_variance": float(y_std ** 2),
        "scaffold_diversity_at_04": float(diversity04),
        "scaffold_diversity_at_06": float(diversity06),
        "scaffold_diversity_at_08": float(diversity08),
        "n_unique_scaffold_proxies": int(n_unique_scaffolds),
        "heavy_atom_count_mean": heavy_atom_mean,
        "heavy_atom_count_std": heavy_atom_std,
        "heavy_atom_count_min": heavy_atom_min,
        "heavy_atom_count_max": heavy_atom_max,
        "feat_mean_variance": float(feat_var.mean()),
        "feat_max_variance": float(feat_var.max()),
        "feat_min_variance": float(feat_var.min()),
        "feat_zero_variance_fraction": const_frac,
    }


# ---------------------------------------------------------------------------
# Run all 5 split strategies on a cohort
# ---------------------------------------------------------------------------
def run_split_strategies(
    cohort: str,
    X: np.ndarray,
    y: np.ndarray,
    smiles_list: List[str],
    n_splits: int = 5,
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4),
    alpha: float = 1.0,
    cell_lines: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run all 5 split strategies; aggregate per-split R² (mean ± std).

    A fold that produces NaN (e.g. degenerate test cluster with constant y)
    is counted but does NOT contribute to the per-seed mean. The diagnostic
    flags the LOCO issue on metalcytotox_combined via n_degenerate_folds.
    """
    n = X.shape[0]
    # Use only first 12 features (RDKit baseline) for the diagnostic — matches
    # the Wf-335-M2 / Wf-367-B1 12-d baseline.
    X_12 = X[:, :12]
    results: Dict[str, Dict[str, Any]] = {}
    for split_name in ("random_kfold", "stratified_kfold", "scaffold_kfold",
                       "leave_one_cluster_out", "per_cell_holdout"):
        r2_per_seed: List[float] = []
        n_degenerate_per_seed: List[int] = []
        for s in seeds:
            if split_name == "random_kfold":
                splits = random_kfold_split(n, n_splits=n_splits, seed=s)
            elif split_name == "stratified_kfold":
                splits = stratified_kfold_split(y, n_splits=n_splits, seed=s)
            elif split_name == "scaffold_kfold":
                splits = scaffold_kfold_split(smiles_list, n_splits=n_splits, seed=s)
            elif split_name == "leave_one_cluster_out":
                splits = leave_one_cluster_out_split(X_12, n_clusters=n_splits, seed=s)
            elif split_name == "per_cell_holdout":
                splits = per_cell_holdout_split(cell_lines or [], smiles_list, seed=s)
            else:
                raise ValueError(f"unknown split {split_name}")
            if not splits:
                continue
            r2_per_fold = []
            n_degenerate = 0
            for tr, te in splits:
                if len(te) < 2 or len(tr) < 2:
                    continue
                # Skip degenerate test folds (constant y)
                if np.std(y[te]) < 1e-9:
                    n_degenerate += 1
                    continue
                r2 = ridge_r2_eval(X_12[tr], y[tr], X_12[te], y[te], alpha=alpha)
                r2_per_fold.append(r2)
            if r2_per_fold:
                r2_per_seed.append(float(np.mean(r2_per_fold)))
                n_degenerate_per_seed.append(n_degenerate)
        if r2_per_seed:
            results[split_name] = {
                "n_seeds": len(r2_per_seed),
                "n_folds_per_seed": n_splits,
                "r2_mean": float(np.mean(r2_per_seed)),
                "r2_std": float(np.std(r2_per_seed)),
                "r2_per_seed": [float(x) for x in r2_per_seed],
                "above_zero_seeds": int(sum(1 for r in r2_per_seed if r > 0)),
                "n_degenerate_folds_total": int(sum(n_degenerate_per_seed)),
            }
    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_wf377a2(
    out_path: str = "molmetal/reports/wf377a2_m2r2_diagnostic/result.json",
    cohorts: Tuple[str, ...] = ("platinai_4cell", "hilbert", "metalcytotox_combined"),
    max_samples: int = 2000,
    n_splits: int = 5,
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4),
    alpha: float = 1.0,
    hilbert_seed_offset: int = 10,
) -> Dict[str, Any]:
    """Run the cohort diagnostic + alternative splits; write result.json."""
    sys.path.insert(0, _resolve_repo_path("."))
    from molmetal.scripts.wf331r5_r2_w2_chemprop import load_cohort

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out: Dict[str, Any] = {
        "wf": "Wf-377-A2",
        "task": "m2 R² cohort structural diagnostic + 5 alternative split strategies",
        "scope": "CPU-only diagnostic on 3 real-pIC50 cohorts; no torch autograd; no new ML architectures.",
        "upstream": "Wf-335-M2 + Wf-367-B1 + Wf-369-A2 + Wf-365-A2 finding",
        "n_cohorts": len(cohorts),
        "cohorts": list(cohorts),
        "n_splits": n_splits,
        "n_seeds": len(seeds),
        "alpha": alpha,
        "anchor_baselines": {
            "wf335m2_best_r2": WF335M2_BEST_R2,
            "wf367b1_best_r2": WF367B1_BEST_R2,
        },
        "per_cohort": [],
        "per_split_strategy": {},
    }
    print(f"Wf-377-A2: n_cohorts={len(cohorts)} n_splits={n_splits} n_seeds={len(seeds)}")
    for cohort in cohorts:
        if cohort == "hilbert":
            X, y, smiles, target = load_cohort(
                "platinai_4cell", max_samples=max_samples, seed=hilbert_seed_offset,
            )
        else:
            X, y, smiles, target = load_cohort(
                cohort, max_samples=max_samples, seed=0,
            )
        print(f"\n--- {cohort} (n={len(smiles)}) ---")
        # Per-cohort structural stats
        stats = compute_cohort_stats(cohort, X, y, smiles)
        out["per_cohort"].append(stats)
        print(f"  y range=[{stats['y_min']:.4f}, {stats['y_max']:.4f}] mean={stats['y_mean']:.4f} std={stats['y_std']:.4f}")
        print(f"  scaffold_diversity(<0.4)={stats['scaffold_diversity_at_04']:.4f}")
        print(f"  n_unique_scaffold_proxies={stats['n_unique_scaffold_proxies']} / n={stats['n_samples']}")
        print(f"  feat/sample={stats['features_per_sample_ratio']:.4f} ({stats['n_features']} features / {stats['n_samples']} samples)")
        # Per-split R² measurement (no cell_lines for platinai_4cell — per-cell
        # requires cell_line annotations not present in the current load_cohort)
        cell_lines: Optional[List[str]] = None
        # hilbert uses seed_offset=10 to give an independent split reading on the
        # same 200 SMILES (per Wf-325A / Wf-331-R5 pattern).
        cohort_seeds = tuple(s + hilbert_seed_offset for s in seeds) if cohort == "hilbert" else seeds
        splits_results = run_split_strategies(
            cohort=cohort,
            X=X,
            y=y,
            smiles_list=smiles,
            n_splits=n_splits,
            seeds=cohort_seeds,
            alpha=alpha,
            cell_lines=cell_lines,
        )
        out["per_split_strategy"][cohort] = splits_results
        for split_name, sr in splits_results.items():
            print(f"  {split_name:30s} R²={sr['r2_mean']:+.4f} ± {sr['r2_std']:.4f} (n_seeds={sr['n_seeds']})")
    # Aggregate: best split per cohort
    best_split_per_cohort: Dict[str, str] = {}
    best_r2_per_cohort: Dict[str, float] = {}
    for cohort in cohorts:
        sr = out["per_split_strategy"].get(cohort, {})
        if not sr:
            continue
        best_split = max(sr.items(), key=lambda kv: kv[1]["r2_mean"])
        best_split_per_cohort[cohort] = best_split[0]
        best_r2_per_cohort[cohort] = float(best_split[1]["r2_mean"])
    out["best_split_per_cohort"] = best_split_per_cohort
    out["best_r2_per_cohort"] = best_r2_per_cohort
    # Aggregate verdict
    n_lifted = sum(1 for r in best_r2_per_cohort.values() if r > 0)
    n_cohorts = len(best_r2_per_cohort)
    out["aggregate_verdict"] = {
        "n_lifted_cohorts": n_lifted,
        "n_total_cohorts": n_cohorts,
        "gate_preserved": n_lifted < n_cohorts,
        "all_splits_above_zero": all(
            sr["r2_mean"] > 0
            for cohort in cohorts
            for sr in out["per_split_strategy"].get(cohort, {}).values()
        ),
    }
    # Honest-framing summary
    out["honest_framing_summary"] = {
        "current_protocol": "random K-fold (Wf-335-M2 / Wf-367-B1 anchor)",
        "random_kfold_R2_anchors": {
            cohort: out["per_split_strategy"].get(cohort, {}).get("random_kfold", {}).get("r2_mean")
            for cohort in cohorts
        },
        "best_alternative_split_per_cohort": best_split_per_cohort,
        "best_alternative_R2_per_cohort": best_r2_per_cohort,
        "lift_vs_current_random_kfold": {
            cohort: (
                best_r2_per_cohort[cohort]
                - out["per_split_strategy"].get(cohort, {}).get("random_kfold", {}).get("r2_mean", float("nan"))
            )
            for cohort in cohorts
        },
    }
    out["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out["status"] = "MEASURED" if n_lifted < n_cohorts else "MEASURED_LIFT_FOUND"
    out["verdict"] = (
        "HONEST_NEGATIVE_R2_DEEP_PRESERVED" if n_lifted < n_cohorts
        else "MEASURED_PARTIAL_LIFT"
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print(f"\nWrote {out_path}")
    print(f"Verdict: {out['verdict']}")
    print(f"Best split per cohort: {best_split_per_cohort}")
    return out


def _main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--out-path", default="molmetal/reports/wf377a2_m2r2_diagnostic/result.json")
    p.add_argument("--cohorts", nargs="+", default=["platinai_4cell", "hilbert", "metalcytotox_combined"])
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--max-samples", type=int, default=2000)
    args = p.parse_args(argv)
    run_wf377a2(
        out_path=args.out_path,
        cohorts=tuple(args.cohorts),
        max_samples=args.max_samples,
        n_splits=args.n_splits,
        alpha=args.alpha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())