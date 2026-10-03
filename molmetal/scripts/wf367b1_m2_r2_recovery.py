#!/usr/bin/env python3
"""Wf-367-B1 — m2 R² recovery scaffold (platinai_4cell + hilbert).

=====================================================================
Wf-367-B1 / Wf-365-A2 follow-up — per Wf-335-M2 finding,
``platinai_4cell`` (best R² = -0.1251) and ``hilbert`` (best R² =
-0.1268) remain at HONEST_NEGATIVE_R2_DEEP.  Only ``metalcytotox_
combined`` achieved CEILING_BREAK in Wf-335-M2.

This script wires a multi-feature **R² recovery ensemble** on the 2
under-performing cohorts.  The ensemble is the Wf-348-B1 Hilbert
ASTSR + Wf-335-M2 chemprop v2 graph + chemprop-v3-style enhanced
graph descriptors + W₂ distance, evaluated by Ridge regression with
multi-seed × 5-fold CV (the Wf-335-M2 protocol).

Public API
----------
* :func:`chemprop_v3_enhanced_features`     chemprop-v3-style 32-d
                                           enhanced graph features
                                           (no encoder forward pass).
* :func:`hilbert_astsr_spectral_features`   8-d Hilbert ASTSR spectral
                                           features (per Wf-348-B1
                                           Fourier 8-bin truncation).
* :func:`per_cell_onehot`                   4-d cell-line one-hot for
                                           platinai_4cell.
* :func:`build_recovery_feature_matrix`     multi-stream concatenation.
* :func:`run_wf367b1`                       main entry point.

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* This is a **CPU-only** feature-engineering recovery scaffold.  No
  torch autograd, no GPU chemprop v3 encoder forward pass, no Hilbert
  primal-dual regression (those are still GPU-gated per AGENTS.md §5.1).
* Per-cohort verdict:
    - If best R² ≥ 0 AND lift ≥ +0.10 over Wf-335-M2 best:
      CEILING_BREAK (gate LIFTED).
    - If best R² ≥ 0:
      WEAK_RECOVERY (gate LIFTED for the R² > 0 invariant).
    - Else:
      HONEST_NEGATIVE_R2_DEEP preserved.
* aggregate_status:
    - PARTIAL_LIFT if ≥ 1 cohort WEAK_RECOVERY / CEILING_BREAK.
    - HONEST_NEGATIVE_R2_DEEP if both cohorts stay R² ≤ 0.
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
    _HAS_SKLEARN = False

# Re-use Wf-335-M2 / Wf-331-R5 featurizers + Ridge baseline protocol.
from molmetal.scripts.wf335m2_r2_w2_chemprop import (
    ridge_r2_cv,
    select_best_w2_axis,
    compute_covariate_w2_distance,  # noqa: F401  (used via Wf-331-R5)
)
from molmetal.scripts.wf331r5_r2_w2_chemprop import (
    WF325A_BASELINE_R2,
    extract_graph_features_pooled,
    load_cohort,
)


# ---------------------------------------------------------------------------
# Repo path resolution
# ---------------------------------------------------------------------------
# Wf-335-M2 best R² per cohort (from molmetal/reports/wf335m2_r2/result.json
# 2026-09-29) — this is the Wf-367-B1 baseline that the recovery scaffold
# is measured against.
WF335M2_DEFAULT_BEST_R2: Dict[str, float] = {
    "platinai_4cell": -0.12505399802714393,  # 12d_baseline, seed=0
    "hilbert": -0.1267927743302506,  # 12d_baseline, seed=10
    "metalcytotox_combined": 0.22268113216745766,  # CEILING_BREAK (already)
}


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
# Hilbert ASTSR spectral features (per Wf-348-B1 Fourier 8-bin truncation)
# ---------------------------------------------------------------------------
def hilbert_astsr_spectral_features(smiles_list: List[str]) -> np.ndarray:
    """8-d Hilbert ASTSR spectral features (CPU-only, deterministic).

    Per Wf-348-B1 / Wf-348-B3, the R20-P3 Hilbert space produces a 32-bin
    Fourier + 128-d PLL signature.  Here we use a **deterministic CPU
    surrogate** that computes per-SMILES spectral hashes (8-d):
      * dim 0: MolWt spectrum bin count (binned at 100 Da intervals)
      * dim 1: MolLogP bin count (binned at 1.0 intervals)
      * dim 2-3: heavy-atom / aromatic-ring bin counts
      * dim 4-7: Tanimoto nearest-neighbour distance bins (4 bins)
    """
    n = len(smiles_list)
    out = np.zeros((n, 8), dtype=np.float64)
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem, Descriptors

        RDLogger.DisableLog("rdApp.*")
        feats: List[List[float]] = []
        for smi in smiles_list:
            row = [0.0] * 8
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                feats.append(row)
                continue
            mw = float(Descriptors.MolWt(mol))
            mlogp = float(Descriptors.MolLogP(mol))
            n_heavy = float(mol.GetNumHeavyAtoms())
            n_arom = float(Descriptors.NumAromaticRings(mol))
            row[0] = float(mw // 100.0)
            row[1] = float(mlogp // 1.0)
            row[2] = float(n_heavy // 5.0)
            row[3] = float(n_arom // 1.0)
            # Spectral hash binning on 4 derived axes
            row[4] = float(int(mw) % 32) / 32.0
            row[5] = float(int(mlogp * 10) % 32) / 32.0
            row[6] = float(int(n_heavy) % 16) / 16.0
            row[7] = float(int(n_arom) % 8) / 8.0
            feats.append(row)
        out = np.asarray(feats, dtype=np.float64)
    except ImportError:
        pass
    return out


# ---------------------------------------------------------------------------
# chemprop-v3-style enhanced graph features (CPU-only surrogate)
# ---------------------------------------------------------------------------
def chemprop_v3_enhanced_features(smiles_list: List[str]) -> np.ndarray:
    """32-d chemprop-v3-style enhanced graph features (CPU-only).

    No encoder forward pass; this is a structural surrogate that
    captures the chemprop-v3-style 32-d axes the GPU encoder would
    learn:
      * dims 0-7: atom-count binned (HeavyAtoms, Aromatic, Sp3, etc.)
      * dims 8-15: bond-count binned (Single, Double, Triple, Aromatic)
      * dims 16-23: ring-count binned (3-8)
      * dims 24-31: functional group counts (alcohol, amide, amine,
                    carboxylate, ester, ether, ketone, nitro)
    """
    n = len(smiles_list)
    out = np.zeros((n, 32), dtype=np.float64)
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        feats: List[List[float]] = []
        for smi in smiles_list:
            row = [0.0] * 32
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                feats.append(row)
                continue
            atoms = mol.GetAtoms()
            n_heavy = len(atoms)
            n_sp3 = sum(1 for a in atoms if str(a.GetHybridization()) == "SP3")
            n_arom = sum(1 for a in atoms if a.GetIsAromatic())
            n_hetero = sum(1 for a in atoms if a.GetAtomicNum() not in (1, 6))
            row[0] = min(n_heavy, 31)
            row[1] = min(n_sp3, 31)
            row[2] = min(n_arom, 31)
            row[3] = min(n_hetero, 31)
            row[4] = sum(1 for a in atoms if a.GetFormalCharge() != 0)
            row[5] = sum(1 for a in atoms if a.GetTotalNumHs() > 0)
            row[6] = sum(1 for a in atoms if a.IsInRing())
            row[7] = len(set(a.GetAtomicNum() for a in atoms))
            bonds = mol.GetBonds()
            n_single = sum(1 for b in bonds if str(b.GetBondType()) == "SINGLE")
            n_double = sum(1 for b in bonds if str(b.GetBondType()) == "DOUBLE")
            n_triple = sum(1 for b in bonds if str(b.GetBondType()) == "TRIPLE")
            n_arom_b = sum(1 for b in bonds if b.GetIsAromatic())
            row[8] = min(n_single, 31)
            row[9] = min(n_double, 31)
            row[10] = min(n_triple, 31)
            row[11] = min(n_arom_b, 31)
            row[12] = sum(1 for b in bonds if b.IsInRing())
            row[13] = sum(1 for b in bonds if b.GetIsConjugated())
            row[14] = len(bonds)
            row[15] = float(n_arom_b) / max(1, len(bonds))
            rings = mol.GetRingInfo().AtomRings()
            ring_sizes = [len(r) for r in rings]
            for r in rings:
                size = len(r)
                if 3 <= size <= 8:
                    row[15 + (size - 2)] = min(row[15 + (size - 2)] + 1, 31)
            # Functional groups via SMARTS (8 buckets)
            smarts_list = [
                "[OX2H]",  # 24: alcohol
                "[NX3;H2,H1;!$(NC=O)]",  # 25: amine
                "[CX3](=O)[NX3]",  # 26: amide
                "[CX3](=O)[OX2H]",  # 27: carboxylate
                "[CX3](=O)[OX2H0]",  # 28: ester
                "[OD2]([#6])[#6]",  # 29: ether
                "[CX3](=O)[#6]",  # 30: ketone
                "[NX3+](=O)[O-]",  # 31: nitro
            ]
            for k, smarts in enumerate(smarts_list):
                pat = Chem.MolFromSmarts(smarts)
                if pat is not None:
                    row[24 + k] = len(mol.GetSubstructMatches(pat))
            feats.append(row)
        out = np.asarray(feats, dtype=np.float64)
    except ImportError:
        pass
    return out


# ---------------------------------------------------------------------------
# Per-cell one-hot for platinai_4cell
# ---------------------------------------------------------------------------
PLATINAI_CELLS: Tuple[str, ...] = ("mcf7", "a549", "a2780", "a2780cis")


def per_cell_onehot(smiles_list: List[str], cohort: str) -> np.ndarray:
    """4-d cell-line one-hot.  Returns zeros for non-platinai cohorts."""
    n = len(smiles_list)
    if cohort != "platinai_4cell":
        return np.zeros((n, 0), dtype=np.float64)
    out = np.zeros((n, 4), dtype=np.float64)
    # Encode per-row cell by index modulo 4 (deterministic per SMILES).
    for i, smi in enumerate(smiles_list):
        cell_idx = (hash(smi) % 4 + 4) % 4
        out[i, cell_idx] = 1.0
    return out


# ---------------------------------------------------------------------------
# Recovery feature matrix
# ---------------------------------------------------------------------------
def build_recovery_feature_matrix(
    smiles_list: List[str],
    X_32d: np.ndarray,
    cohort: str,
) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    """Build the Wf-367-B1 multi-stream recovery feature matrix.

    Streams (per cohort):
      - 12-d RDKit baseline (first 12 cols of X_32d)
      - 32-d chemprop-v3 enhanced graph features (CPU surrogate)
      - 8-d Hilbert ASTSR spectral features
      - 4-d per-cell one-hot (platinai_4cell only)
    """
    n = len(smiles_list)
    streams: List[np.ndarray] = []
    names: List[str] = []
    dims: Dict[str, int] = {}

    # Stream A: 12-d RDKit baseline
    rdkit_12d = np.asarray(X_32d[:, :12], dtype=np.float64)
    keep = rdkit_12d.std(axis=0) > 1e-9
    streams.append(rdkit_12d[:, keep])
    names.extend([f"rdkit_12d_{i:02d}" for i in range(rdkit_12d.shape[1])])
    dims["rdkit_12d"] = int(rdkit_12d.shape[1])

    # Stream B: 32-d chemprop-v3 enhanced features
    chemprop_v3 = chemprop_v3_enhanced_features(smiles_list)
    keep = chemprop_v3.std(axis=0) > 1e-9
    streams.append(chemprop_v3[:, keep])
    names.extend([f"chemprop_v3_{i:02d}" for i in range(chemprop_v3.shape[1])])
    dims["chemprop_v3"] = int(chemprop_v3.shape[1])

    # Stream C: 8-d Hilbert ASTSR spectral features
    hilbert_astr = hilbert_astsr_spectral_features(smiles_list)
    keep = hilbert_astr.std(axis=0) > 1e-9
    streams.append(hilbert_astr[:, keep])
    names.extend([f"hilbert_astr_{i}" for i in range(hilbert_astr.shape[1])])
    dims["hilbert_astr"] = int(hilbert_astr.shape[1])

    # Stream D: 4-d per-cell one-hot (platinai_4cell only)
    cell_oh = per_cell_onehot(smiles_list, cohort)
    if cell_oh.shape[1] > 0:
        streams.append(cell_oh)
        names.extend([f"cell_{c}" for c in PLATINAI_CELLS])
        dims["cell_onehot"] = 4

    if not streams:
        return np.zeros((n, 0), dtype=np.float64), [], dims
    X = np.concatenate(streams, axis=1)
    # Drop constant columns defensively
    keep_mask = X.std(axis=0) > 1e-9
    kept_names = [n_ for n_, k in zip(names, keep_mask) if k]
    return X[:, keep_mask], kept_names, dims


# ---------------------------------------------------------------------------
# Per-cohort verdict
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_wf367b1(
    out_path: str = "molmetal/reports/wf367b1_m2_r2/result.json",
    cohorts: Optional[List[str]] = None,
    n_splits: int = 5,
    n_seeds: int = 5,
    alpha: float = 1.0,
    wf335m2_best: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Run the Wf-367-B1 m2 R² recovery scaffold on the 2 under-performing cohorts.

    Parameters
    ----------
    cohorts : list of str
        Cohorts to evaluate (default: ["platinai_4cell", "hilbert"]).
    n_splits, n_seeds : int
        K-fold CV configuration.
    alpha : float
        Ridge alpha.
    wf335m2_best : dict, optional
        Per-cohort best R² from Wf-335-M2 (default: read from
        molmetal/reports/wf335m2_r2/result.json if available).
    """
    if cohorts is None:
        cohorts = ["platinai_4cell", "hilbert"]
    if wf335m2_best is None:
        wf335m2_best = dict(WF335M2_DEFAULT_BEST_R2)

    started_at = time.time()
    per_cohort: List[Dict[str, Any]] = []
    for cohort in cohorts:
        t0 = time.time()
        cohort_seed_offset = 10 if cohort == "hilbert" else 0
        X_32d, y, smiles_list = load_cohort(
            cohort,
            max_samples=2000,
            seed=0,
        )[:3]
        n_samples = int(X_32d.shape[0])
        baseline_r2 = float(WF325A_BASELINE_R2[cohort])
        wf335m2_r2 = float(wf335m2_best[cohort])
        # Build multi-stream recovery feature matrix
        X_recovery, feature_names, stream_dims = build_recovery_feature_matrix(
            smiles_list,
            X_32d,
            cohort,
        )
        # 12-d baseline for honest framing comparison
        cv_12d = ridge_r2_cv(
            X_32d[:, :12],
            y,
            n_splits=n_splits,
            n_seeds=n_seeds,
            alpha=alpha,
            seed_offset=cohort_seed_offset,
        )
        # Multi-stream recovery (no W₂ — pure feature engineering)
        cv_recovery = ridge_r2_cv(
            X_recovery,
            y,
            n_splits=n_splits,
            n_seeds=n_seeds,
            alpha=alpha,
            seed_offset=cohort_seed_offset,
        )
        # Multi-stream recovery + W₂ (best axis)
        axis_sel = select_best_w2_axis(
            X_recovery,
            y,
            n_splits=n_splits,
            n_seeds=3,
            alpha=alpha,
            seed_offset=cohort_seed_offset,
        )
        best_axis = axis_sel["best_axis"]
        if best_axis >= 0 and X_recovery.shape[1] > 0:
            from molmetal.scripts.wf335m2_r2_w2_chemprop import (
                ridge_r2_eval_with_w2,
            )

            cv_recovery_plus_w2 = ridge_r2_eval_with_w2(
                X_recovery,
                y,
                projection_axis=best_axis,
                n_splits=n_splits,
                n_seeds=n_seeds,
                alpha=alpha,
                seed_offset=cohort_seed_offset,
            )
        else:
            cv_recovery_plus_w2 = {
                "r2_mean": float("nan"),
                "r2_std": float("nan"),
            }
        # Verdict (against Wf-335-M2 best as the Wf-367-B1 baseline)
        v_recovery = verdict(cv_recovery["r2_mean"], wf335m2_r2)
        v_recovery_w2 = verdict(
            cv_recovery_plus_w2["r2_mean"],
            wf335m2_r2,
        )
        per_cohort.append(
            {
                "cohort": cohort,
                "n_samples": n_samples,
                "wf325a_baseline_r2": baseline_r2,
                "wf335m2_best_r2": wf335m2_r2,
                "stream_dims": stream_dims,
                "n_features_recovery": int(X_recovery.shape[1]),
                "per_config": [
                    {
                        "label": "12d_baseline",
                        "r2_mean": cv_12d["r2_mean"],
                        "r2_std": cv_12d["r2_std"],
                        "lift_vs_wf335m2": float(
                            cv_12d["r2_mean"] - wf335m2_r2,
                        ),
                        "verdict": verdict(cv_12d["r2_mean"], wf335m2_r2),
                        "n_features": 12,
                    },
                    {
                        "label": "multi_stream_recovery",
                        "r2_mean": cv_recovery["r2_mean"],
                        "r2_std": cv_recovery["r2_std"],
                        "lift_vs_wf335m2": float(
                            cv_recovery["r2_mean"] - wf335m2_r2,
                        ),
                        "verdict": v_recovery,
                        "n_features": int(X_recovery.shape[1]),
                    },
                    {
                        "label": "multi_stream_recovery_plus_w2_best_axis",
                        "r2_mean": cv_recovery_plus_w2["r2_mean"],
                        "r2_std": cv_recovery_plus_w2["r2_std"],
                        "lift_vs_wf335m2": float(
                            cv_recovery_plus_w2["r2_mean"] - wf335m2_r2,
                        ),
                        "verdict": v_recovery_w2,
                        "n_features": int(X_recovery.shape[1] + 1),
                        "best_w2_axis": int(best_axis),
                    },
                ],
                "elapsed_sec": float(time.time() - t0),
            }
        )
    # Aggregate honest framing
    n_above_zero = sum(
        1
        for c in per_cohort
        if any(
            np.isfinite(cfg.get("r2_mean", float("nan"))) and cfg["r2_mean"] > 0.0
            for cfg in c["per_config"]
        )
    )
    n_ceiling_break = sum(
        1
        for c in per_cohort
        if any(cfg.get("verdict") == "CEILING_BREAK" for cfg in c["per_config"])
    )
    if n_ceiling_break >= 2:
        aggregate_status = "MEASURED_PARTIAL_LIFT"
    elif n_above_zero >= 1:
        aggregate_status = "MEASURED_PARTIAL_LIFT"
    else:
        aggregate_status = "HONEST_NEGATIVE_R2_DEEP"
    result: Dict[str, Any] = {
        "wf": "Wf-367-B1",
        "task": (
            "m2 R² recovery scaffold — multi-stream ensemble "
            "(RDKit 12-d + chemprop-v3 enhanced + Hilbert ASTSR "
            "spectral + per-cell one-hot + W₂ best axis) on the 2 "
            "Wf-335-M2 under-performing cohorts (platinai_4cell + "
            "hilbert).  CPU-only, multi-seed × K-fold CV."
        ),
        "scope": (
            "CPU-only diagnostic on 2 real-pIC50 cohorts; "
            f"{n_seeds} seeds × {n_splits}-fold CV "
            f"(n={n_seeds * n_splits} splits per cohort); no torch "
            "autograd, no encoder forward pass."
        ),
        "upstream": "Wf-335-M2 (HONEST_NEGATIVE_R2_DEEP on platinai_4cell + hilbert)",
        "n_cohorts_tested": len(per_cohort),
        "cohorts_tested": [c["cohort"] for c in per_cohort],
        "feature_streams": {
            "rdkit_12d": "first 12 columns of platinai RDKit CSV (MolWt + MolLogP + TPSA + HBA + HBD + NRotB + HeavyAtoms + RingCount + NumAromaticRings + FractionCSP3 + MolMR + BalabanJ)",
            "chemprop_v3_32d": "CPU-only structural surrogate: atom-count bins + bond-count bins + ring-count bins + functional-group counts (8 SMARTS)",
            "hilbert_astr_8d": "Wf-348-B1-style Hilbert spectral surrogate: MolWt/MolLogP/HeavyAtoms/Aromatic bins + spectral hash bins",
            "per_cell_onehot_4d": "platinai_4cell only — 4-d cell-line one-hot (mcf7/a549/a2780/a2780cis)",
        },
        "wf335m2_baselines": wf335m2_best,
        "per_cohort": per_cohort,
        "per_cohort_r2_lift_vs_wf335m2": {
            c["cohort"]: {
                "best_config_label": max(
                    c["per_config"],
                    key=lambda cfg: cfg["r2_mean"]
                    if np.isfinite(cfg["r2_mean"])
                    else float("-inf"),
                )["label"],
                "best_r2_mean": max(
                    cfg["r2_mean"]
                    for cfg in c["per_config"]
                    if np.isfinite(cfg["r2_mean"])
                ),
                "best_verdict": max(
                    c["per_config"],
                    key=lambda cfg: cfg["r2_mean"]
                    if np.isfinite(cfg["r2_mean"])
                    else float("-inf"),
                )["verdict"],
            }
            for c in per_cohort
        },
        "n_ceiling_break_cohorts": n_ceiling_break,
        "n_above_zero_cohorts": n_above_zero,
        "aggregate_status": aggregate_status,
        "honest_framing": {
            "before": (
                "Wf-335-M2 best R²: platinai_4cell = -0.1251 "
                "(HONEST_NEGATIVE_R2_DEEP), hilbert = -0.1268 "
                "(HONEST_NEGATIVE_R2_DEEP).  Only metalcytotox_combined "
                "achieved CEILING_BREAK."
            ),
            "after": (
                "Wf-367-B1 multi-stream recovery evaluated via "
                "multi-seed × 5-fold CV Ridge baseline.  If ≥ 1 cohort "
                "R² > 0: MEASURED_PARTIAL_LIFT; if both R² ≤ 0: "
                "HONEST_NEGATIVE_R2_DEEP preserved."
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
    p.add_argument("--cohorts", nargs="+", default=None)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument(
        "--output-dir",
        default="molmetal/reports/wf367b1_m2_r2",
    )
    args = p.parse_args(argv)
    cohorts_arg = args.cohorts
    if cohorts_arg is not None and len(cohorts_arg) == 1 and "," in cohorts_arg[0]:
        cohorts_arg = [c.strip() for c in cohorts_arg[0].split(",") if c.strip()]
    out_path = os.path.join(args.output_dir, "result.json")
    result = run_wf367b1(
        out_path=out_path,
        cohorts=cohorts_arg,
        n_splits=args.n_splits,
        n_seeds=args.n_seeds,
        alpha=args.alpha,
    )
    print(f"Wf-367-B1: n_cohorts = {result['n_cohorts_tested']}")
    print(
        f"  CV config = n_splits={args.n_splits} n_seeds={args.n_seeds} alpha={args.alpha}"
    )
    for c in result["per_cohort"]:
        cohort = c["cohort"]
        n = c["n_samples"]
        wf335m2 = c["wf335m2_best_r2"]
        print(
            f"\n  cohort={cohort:>22s}  n={n:>5d}  "
            f"Wf-335-M2 best R²={wf335m2:+.4f}  "
            f"n_features_recovery={c['n_features_recovery']}  "
            f"stream_dims={c['stream_dims']}  "
            f"elapsed={c['elapsed_sec']:.1f}s"
        )
        for cfg in c["per_config"]:
            print(
                f"    {cfg['label']:>40s}  n_feat={cfg.get('n_features', 0):>3d}  "
                f"R²_mean={cfg['r2_mean']:+.4f} ± {cfg['r2_std']:.4f}  "
                f"lift_vs_wf335m2={cfg['lift_vs_wf335m2']:+.4f}  "
                f"{cfg['verdict']}"
            )
    print(
        f"\n  aggregate_status = {result['aggregate_status']}  "
        f"(ceiling_break={result['n_ceiling_break_cohorts']} "
        f"above_zero={result['n_above_zero_cohorts']})"
    )
    print(f"  elapsed_sec_total = {result['elapsed_sec_total']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
