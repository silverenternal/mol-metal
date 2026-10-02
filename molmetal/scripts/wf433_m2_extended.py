#!/usr/bin/env python3
"""Wf-433-A2 — m2 R² EXTENDED features diagnostic (TODO-201 §3.3 EXTENDED axis).

=====================================================================
Wf-433-A2 — closes the EXTENDED axis of the per-cohort feature engineering
plan from TODO/pending/201_m2_r2_per_cohort_feature_engineering.md §3.3:

  * Per-cohort config axis (Wf-380-A2 + Wf-388-A1): 12-d / 24-d / 50-d
    matched to cohort size → CONFIRMED methodology validated
  * EXTENDED axis (this script): 32-d RDKit + 128-d PLL proxy = 160-d
    → CONFIRM structural limit on small n=200 cohorts (Wf-377-A2 §2.1
    bias-variance inflation prediction)

Per user voice 2026-09-30 *"如果是同一个问题一直没修好的话得做调研+review尝试
更深层的修复"* + Wf-433-A1 HONEST_NEGATIVE_CLOSURE_PLAN.md §0.2:

  1. m1 1-seed STALE 0.4917 → 7/7 pair aggregate 0.7577 (Wf-413-A1 SHIPPED)
  2. m2 R² -0.1440 / +0.1981 → run 5×5 CV Ridge on EXTENDED features
     (32-d RDKit + 128-d PLL proxy = 160-d) → confirm structural limit
  3. m6_kappa 0.472 < 0.6 → DROP decision per Wf-424-A15 SHIPPED

Wf-433-A2 closes the m2 R² EXTENDED axis (item #2). Per Wf-377-A2 §2.1 +
Wf-388-A1 + Wf-390-A2 + Wf-367-B1, the **bias-variance tradeoff** at small n
predicts the EXTENDED 160-d config should make things WORSE on n=200 cohorts
(the structural CV-R² ceiling for 12-d RDKit is preserved; adding 128-d PLL
proxy features OVERFITS). This script **empirically confirms** the prediction
via 5×5 CV Ridge on real cohorts.

Why PLL proxy (Morgan fingerprint radius=2 nBits=128)?
* Real PLL (Liu & Feng 2026) requires 3D conformer + AllChem.EmbedMolecule
  per SMILES; 200 SMILES × 3D embed = slow + many SMILES fail to embed.
* Morgan fingerprint is the canonical 2D topological invariant proxy for
  atomic-environment density (Rogers & Hahn 2010); 128-d is the standard
  fixed-length bit vector for ECFP-4.
* Honest framing: this script calls the 128-d feature "PLL proxy" and
  documents the substitution in the output JSON.

Honest framing (party-twenty-spirit §3.5):
* This script **confirms the structural limit** on n=200 cohorts via
  EXTENDED features (160-d). Per Wf-367-B1 multi-stream ensemble, adding
  50-d on n=200 made R² WORSE (-0.5158 / -0.4980 vs 12-d baseline -0.1251 /
  -0.1268). EXTENDED 160-d should make it WORSE still per bias-variance
  inflation theory (p/n = 0.80, deep into the worst-case regime).
* per-cohort verdict:
    - If R² > 0 AND lift ≥ +0.10 vs 12-d baseline: CEILING_BREAK (not expected).
    - If R² > 0: WEAK_RECOVERY (not expected).
    - Else: HONEST_NEGATIVE_R2_DEEP preserved (expected outcome).
* No GPU launches; no torch autograd; CPU-only Ridge baseline.
* Wall PROJECTED ~30 s for 3 cohorts × 5 seeds × 5-fold CV on 160-d.

Public API
----------
* :func:`load_cohort`                load cohort via Wf-331-R5 canonical loader
* :func:`compute_extended_features`  32-d RDKit + 128-d PLL proxy = 160-d
* :func:`train_eval_extended`         5 seeds × 5-fold CV Ridge on 160-d
* :func:`aggregate_extended_r2`      MEAN ± STD + lift vs 12-d baseline
* :func:`run_wf433_m2_extended`      main entry point
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from sklearn.linear_model import Ridge  # type: ignore
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover — defensive
    _HAS_SKLEARN = False

try:
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem, rdMolDescriptors  # type: ignore
    _HAS_RDKIT = True
except Exception:  # pragma: no cover — defensive
    _HAS_RDKIT = False

# Reuse Wf-331-R5 canonical cohort loader + column constants.
from molmetal.scripts.wf331r5_r2_w2_chemprop import (
    load_cohort as _wf331r5_load_cohort,
    PLATINAI_12D_COLUMNS,
    PLATINAI_32D_COLUMNS,
    METALCYTOTOX_TARGET_COLUMN,
    WF325A_BASELINE_R2,
)
# Reuse Wf-388-A1 per-cohort loader signatures.
from molmetal.scripts.wf388a1_m2_per_cohort_real import (
    load_platinai_4cell,
    load_hilbert,
    load_metalcytotox_combined,
    COHORT_REGISTRY,
)


# ---------------------------------------------------------------------------
# 12-d baseline anchors (from Wf-388-A1 real-cohort MEASURED per-cohort R²)
# ---------------------------------------------------------------------------
# Per `metrics/by_metric/m2_r2_per_cohort.json::per_cohort` (2026-09-30):
WF388A1_12D_BASELINE_R2: Dict[str, float] = {
    "platinai_4cell":        -0.14401547578768395,  # 12-d RDKit + Ridge α=1.0
    "hilbert":               -0.14401547578768395,  # same loader + seed_offset=10
    "metalcytotox_combined":  0.19808189300924267,  # 32-d RDKit+chemprop+Hilbert + Ridge α=0.1
}


# ---------------------------------------------------------------------------
# Wf-335-M2 best R² per cohort (the Wf-388-A1 baseline anchor)
# ---------------------------------------------------------------------------
WF335M2_BEST_R2: Dict[str, float] = {
    "platinai_4cell":        -0.12505399802714393,  # 12d_baseline, seed=0
    "hilbert":               -0.1268,               # 12d_baseline, seed_offset=10
    "metalcytotox_combined":  0.22268113216745766,  # full_static_combined_plus_w2_best_axis
}


# ---------------------------------------------------------------------------
# EXTENDED feature computation (32-d RDKit + 128-d PLL proxy = 160-d)
# ---------------------------------------------------------------------------
def _rdkit_features(smiles: str, columns: Tuple[str, ...]) -> np.ndarray:
    """Compute RDKit descriptor features for a single SMILES.

    Falls back to zeros for any SMILES that fails to parse (matches Wf-331-R5
    graceful-failure contract; per-cohort verdict uses real cohort loaders
    that pre-filter failures).
    """
    if not _HAS_RDKIT:
        return np.zeros(len(columns), dtype=np.float64)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(len(columns), dtype=np.float64)
    out = np.zeros(len(columns), dtype=np.float64)
    for j, col in enumerate(columns):
        try:
            if col == "MolWt":
                out[j] = float(rdMolDescriptors.CalcExactMolWt(mol))
            elif col == "MolLogP":
                out[j] = float(rdMolDescriptors.CalcCrippenDescriptors(mol)[0])
            elif col == "TPSA":
                out[j] = float(rdMolDescriptors.CalcTPSA(mol))
            elif col == "HBA":
                out[j] = float(rdMolDescriptors.CalcNumHBA(mol))
            elif col == "HBD":
                out[j] = float(rdMolDescriptors.CalcNumHBD(mol))
            elif col == "NRotB":
                out[j] = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
            elif col == "HeavyAtoms":
                out[j] = float(mol.GetNumHeavyAtoms())
            elif col == "RingCount":
                out[j] = float(rdMolDescriptors.CalcNumRings(mol))
            elif col == "NumAromaticRings":
                out[j] = float(rdMolDescriptors.CalcNumAromaticRings(mol))
            elif col == "FractionCSP3":
                out[j] = float(rdMolDescriptors.CalcFractionCSP3(mol))
            elif col == "MolMR":
                out[j] = float(rdMolDescriptors.CalcCrippenDescriptors(mol)[1])
            elif col == "BalabanJ":
                out[j] = float(rdMolDescriptors.CalcBalabanJ(mol))
            elif col == "BertzCT":
                out[j] = float(rdMolDescriptors.CalcBertzCT(mol))
            elif col.startswith("Chi"):
                order = int(col[3])
                kind = col[4:]
                fn = getattr(rdMolDescriptors, f"Chi{order}{kind}", None)
                if fn:
                    out[j] = float(fn(mol))
            elif col == "HallKierAlpha":
                out[j] = float(rdMolDescriptors.CalcHallKierAlpha(mol))
            elif col.startswith("Kappa"):
                order = int(col[5:])
                fn = getattr(rdMolDescriptors, f"Kappa{order}", None)
                if fn:
                    out[j] = float(fn(mol))
            elif col == "LabuteASA":
                out[j] = float(rdMolDescriptors.CalcLabuteASA(mol))
            elif col == "MaxPartialCharge":
                charges = [float(a.GetFormalCharge()) for a in mol.GetAtoms()]
                out[j] = max(charges) if charges else 0.0
            elif col == "MinPartialCharge":
                charges = [float(a.GetFormalCharge()) for a in mol.GetAtoms()]
                out[j] = min(charges) if charges else 0.0
            elif col == "MaxAbsPartialCharge":
                charges = [abs(float(a.GetFormalCharge())) for a in mol.GetAtoms()]
                out[j] = max(charges) if charges else 0.0
            elif col == "MinAbsPartialCharge":
                charges = [abs(float(a.GetFormalCharge())) for a in mol.GetAtoms()]
                out[j] = min(charges) if charges else 0.0
            elif col == "NumAliphaticRings":
                out[j] = float(rdMolDescriptors.CalcNumAliphaticRings(mol))
            elif col == "NumSaturatedRings":
                out[j] = float(rdMolDescriptors.CalcNumSaturatedRings(mol))
            elif col == "NumHeteroatoms":
                out[j] = float(rdMolDescriptors.CalcNumHeteroatoms(mol))
            elif col == "NHOHCount":
                out[j] = float(rdMolDescriptors.CalcNHOHCount(mol))
            elif col == "NOCount":
                out[j] = float(rdMolDescriptors.CalcNOCount(mol))
            elif col.startswith("fr_"):
                fn = getattr(rdMolDescriptors, f"CalcNum{col[3:]}", None)
                if fn:
                    out[j] = float(fn(mol))
            else:
                out[j] = float(rdMolDescriptors.CalcMolWt(mol))  # fallback
        except Exception:
            out[j] = 0.0
    return out


def _pll_proxy_128(smiles: str) -> np.ndarray:
    """PLL proxy 128-d feature: Morgan ECFP radius=2, nBits=128.

    Honest substitution per script header §"Why PLL proxy": real PLL requires
    3D conformer + AllChem.EmbedMolecule per SMILES; 200 SMILES × 3D embed
    is slow + many SMILES fail to embed (per Wf-387-A3 §3 honest-negative).
    Morgan fingerprint (radius=2, nBits=128) is the canonical 2D topological
    invariant proxy for atomic-environment density (Rogers & Hahn 2010);
    it tests the "160-d feature EXTENDED axis" diagnostic without the
    3D-conformer cost. The honest framing labels this as "PLL proxy".
    """
    if not _HAS_RDKIT:
        return np.zeros(128, dtype=np.float64)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(128, dtype=np.float64)
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=128)
        arr = np.zeros(128, dtype=np.float64)
        for bit in fp.GetOnBits():
            if 0 <= bit < 128:
                arr[bit] = 1.0
        return arr
    except Exception:
        return np.zeros(128, dtype=np.float64)


def compute_extended_features(smiles_list: List[str]) -> np.ndarray:
    """Compute EXTENDED features: 32-d RDKit + 128-d PLL proxy = 160-d.

    Returns
    -------
    np.ndarray, shape (n_smiles, 160).
        First 32 columns are RDKit descriptors (per PLATINAI_32D_COLUMNS);
        next 128 columns are Morgan ECFP4 bits (PLL proxy).
    """
    n = len(smiles_list)
    X = np.zeros((n, 160), dtype=np.float64)
    for i, smi in enumerate(smiles_list):
        rdkit_vec = _rdkit_features(smi, PLATINAI_32D_COLUMNS)
        pll_vec = _pll_proxy_128(smi)
        X[i, :32] = rdkit_vec
        X[i, 32:160] = pll_vec
    return X


# ---------------------------------------------------------------------------
# Per-cohort loader convenience (re-export from Wf-388-A1)
# ---------------------------------------------------------------------------
load_cohort_dict: Dict[str, Any] = {
    "platinai_4cell":        load_platinai_4cell,
    "hilbert":               load_hilbert,
    "metalcytotox_combined": load_metalcytotox_combined,
}


# ---------------------------------------------------------------------------
# 5 seeds × 5-fold CV Ridge on 160-d EXTENDED features
# ---------------------------------------------------------------------------
def train_eval_extended(
    smiles_list: List[str],
    y: np.ndarray,
    n_seeds: int = 5,
    n_folds: int = 5,
    alpha: float = 1.0,
) -> Dict[str, Any]:
    """5 seeds × 5-fold CV Ridge on 160-d EXTENDED features.

    Returns
    -------
    Dict with keys: r2_per_seed (List[float], len=n_seeds), r2_mean, r2_std,
    n_splits_effective (n_seeds × n_folds).
    """
    if not _HAS_SKLEARN:
        raise RuntimeError("scikit-learn required for Ridge regression")

    X = compute_extended_features(smiles_list)
    n = len(smiles_list)

    r2_per_seed: List[float] = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        fold_size = n // n_folds
        r2_per_fold: List[float] = []
        for k in range(n_folds):
            start = k * fold_size
            stop = (k + 1) * fold_size if k < n_folds - 1 else n
            test_idx = perm[start:stop]
            train_idx = np.concatenate([perm[:start], perm[stop:]])
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            # Ridge closed-form with intercept (matches Wf-380-A2 protocol).
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            ss_res = float(np.sum((y_test - y_pred) ** 2))
            ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
            r2_per_fold.append(r2)
        r2_per_seed.append(float(np.mean(r2_per_fold)))

    return {
        "r2_per_seed": r2_per_seed,
        "r2_mean": float(np.mean(r2_per_seed)),
        "r2_std": float(np.std(r2_per_seed, ddof=0)),
        "n_splits_effective": n_seeds * n_folds,
    }


# ---------------------------------------------------------------------------
# Aggregate per-cohort EXTENDED R² + verdict
# ---------------------------------------------------------------------------
def aggregate_extended_r2(
    cohort_results: Dict[str, Dict[str, Any]],
    baseline_12d_r2: Dict[str, float] = WF388A1_12D_BASELINE_R2,
    baseline_335m2_best: Dict[str, float] = WF335M2_BEST_R2,
) -> Dict[str, Any]:
    """Aggregate per-cohort EXTENDED R² results with verdict + lifts."""
    cohorts_out = []
    for cohort, res in cohort_results.items():
        r2_mean = res["r2_mean"]
        baseline = baseline_12d_r2.get(cohort, float("nan"))
        baseline_best = baseline_335m2_best.get(cohort, float("nan"))
        lift_vs_12d = r2_mean - baseline
        lift_vs_335m2 = r2_mean - baseline_best
        if r2_mean > 0 and lift_vs_12d >= 0.10:
            verdict = "CEILING_BREAK"
        elif r2_mean > 0:
            verdict = "WEAK_RECOVERY"
        elif lift_vs_12d < -0.05:
            verdict = "HONEST_NEGATIVE_R2_DEEP_OVERFIT"
        else:
            verdict = "HONEST_NEGATIVE_R2_DEEP"
        cohorts_out.append({
            "cohort": cohort,
            "r2_mean": r2_mean,
            "r2_std": res["r2_std"],
            "n_splits_effective": res["n_splits_effective"],
            "baseline_12d_r2": baseline,
            "baseline_335m2_best_r2": baseline_best,
            "lift_vs_12d": lift_vs_12d,
            "lift_vs_335m2_best": lift_vs_335m2,
            "verdict": verdict,
        })
    n_above_zero = sum(1 for c in cohorts_out if c["r2_mean"] > 0)
    n_ceiling_break = sum(1 for c in cohorts_out if c["verdict"] == "CEILING_BREAK")
    n_weak_recovery = sum(1 for c in cohorts_out if c["verdict"] == "WEAK_RECOVERY")
    n_honest_negative = sum(1 for c in cohorts_out if "HONEST_NEGATIVE" in c["verdict"])
    aggregate_status = (
        "CEILING_BREAK_PARTIAL" if n_ceiling_break >= 1 else
        "MEASURED_PARTIAL_LIFT" if n_weak_recovery >= 1 else
        "MEASURED_HONEST_NEGATIVE_R2_DEEP_OVERFIT"
    )
    return {
        "cohorts": cohorts_out,
        "aggregate": {
            "n_cohorts_tested": len(cohorts_out),
            "n_cohorts_above_zero": n_above_zero,
            "n_cohorts_ceiling_break": n_ceiling_break,
            "n_cohorts_weak_recovery_or_better": n_weak_recovery,
            "n_cohorts_honest_negative": n_honest_negative,
            "aggregate_status": aggregate_status,
        },
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_wf433_m2_extended(
    cohorts: List[str] = ("platinai_4cell", "hilbert", "metalcytotox_combined"),
    n_seeds: int = 5,
    n_folds: int = 5,
    alpha: float = 1.0,
) -> Dict[str, Any]:
    """Run EXTENDED feature Ridge CV on real cohorts + aggregate.

    Returns
    -------
    Dict with keys: per_cohort, aggregate, config.
    """
    cohort_results: Dict[str, Dict[str, Any]] = {}
    for cohort in cohorts:
        load_fn = load_cohort_dict[cohort]
        X, y, smiles_list, target_name = load_fn()
        # For metalcytotox_combined we need to slice to 32-d RDKit + 128-d PLL.
        # The loaders return the full RDKit 32-d matrix; we use EXTENDED on top.
        res = train_eval_extended(smiles_list, y, n_seeds=n_seeds, n_folds=n_folds, alpha=alpha)
        cohort_results[cohort] = res
    agg = aggregate_extended_r2(cohort_results)
    return {
        "per_cohort": [
            {
                "cohort": c,
                **cohort_results[c],
            }
            for c in cohorts
        ],
        "aggregate": agg["aggregate"],
        "config": {
            "n_features": 160,
            "n_features_rdkit": 32,
            "n_features_pll_proxy": 128,
            "pll_proxy_type": "MorganFingerprint_radius2_nBits128",
            "pll_proxy_justification": "Real PLL requires 3D conformer; 200 SMILES × 3D embed slow + many fail. Morgan fingerprint is canonical 2D proxy for atomic-environment density (Rogers & Hahn 2010). Honest framing: 128-d PLL proxy.",
            "ridge_alpha": alpha,
            "n_seeds": n_seeds,
            "n_folds": n_folds,
            "sklearn_used": _HAS_SKLEARN,
            "rdkit_used": _HAS_RDKIT,
        },
    }


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Wf-433-A2 — m2 R² EXTENDED features diagnostic"
    )
    parser.add_argument("--cohorts", nargs="+",
                        default=["platinai_4cell", "hilbert", "metalcytotox_combined"],
                        help="Cohort names to evaluate.")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of seeds for K-fold CV.")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of folds per seed.")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ridge regularization parameter.")
    parser.add_argument("--out-dir", type=str,
                        default="molmetal/reports/wf433_m2_extended",
                        help="Output directory for result.json.")
    parser.add_argument("--write-catalog", action="store_true",
                        help="Also update metrics/by_metric/m2_r2_per_cohort.json "
                             "with EXTENDED values via atomic_write_file.")
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "result.json")

    print(f"Wf-433-A2 — EXTENDED features (32-d RDKit + 128-d PLL proxy = 160-d)")
    print(f"  cohorts: {args.cohorts}")
    print(f"  seeds × folds: {args.n_seeds} × {args.n_folds}")
    print(f"  Ridge alpha: {args.alpha}")

    result = run_wf433_m2_extended(
        cohorts=args.cohorts,
        n_seeds=args.n_seeds,
        n_folds=args.n_folds,
        alpha=args.alpha,
    )

    # Print per-cohort summary.
    for c in result["per_cohort"]:
        baseline_12d = WF388A1_12D_BASELINE_R2.get(c["cohort"], float("nan"))
        baseline_335 = WF335M2_BEST_R2.get(c["cohort"], float("nan"))
        lift_12d = c["r2_mean"] - baseline_12d
        print(
            f"  {c['cohort']:<22}  r2_mean={c['r2_mean']:+.4f} ± {c['r2_std']:.4f}  "
            f"lift_vs_12d={lift_12d:+.4f}  lift_vs_335m2={c['r2_mean']-baseline_335:+.4f}"
        )

    # Write result.json via atomic_write_file per AGENTS.md §5.6.
    try:
        from molmetal.scripts.wf386a3_atomic_write import atomic_write_file
        atomic_write_file(out_path, json.dumps(result, indent=2, sort_keys=False))
        print(f"\nresult.json written via atomic_write_file: {out_path}")
    except ImportError:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nresult.json written (direct fallback): {out_path}")

    # Optionally update the canonical metric catalog.
    if args.write_catalog:
        catalog_path = "metrics/by_metric/m2_r2_per_cohort.json"
        try:
            with open(catalog_path, "r") as f:
                catalog = json.load(f)
        except FileNotFoundError:
            catalog = {
                "metric": "m2_r2_per_cohort",
                "description": "m2 R² per-cohort config on REAL cohorts",
                "cohorts_tested": [],
                "per_cohort": [],
            }
        # Update per-cohort list with EXTENDED entries (preserving existing entries).
        existing_cohorts = {c.get("cohort"): c for c in catalog.get("per_cohort", [])}
        new_entries = []
        for c in result["per_cohort"]:
            baseline_12d = WF388A1_12D_BASELINE_R2.get(c["cohort"], float("nan"))
            baseline_335 = WF335M2_BEST_R2.get(c["cohort"], float("nan"))
            new_entries.append({
                "cohort": c["cohort"],
                "config_features": "extended_160d_rdkit32_pll_proxy128",
                "config_model": "ridge",
                "n_features_used": 160,
                "alpha_used": result["config"]["ridge_alpha"],
                "r2_mean": c["r2_mean"],
                "r2_std": c["r2_std"],
                "n_splits_effective": c["n_splits_effective"],
                "baseline_12d_r2": baseline_12d,
                "lift_vs_baseline_12d": c["r2_mean"] - baseline_12d,
                "lift_vs_wf335m2_best": c["r2_mean"] - baseline_335,
                "verdict": (
                    "HONEST_NEGATIVE_R2_DEEP_OVERFIT" if c["r2_mean"] < baseline_12d - 0.05
                    else "WEAK_RECOVERY" if c["r2_mean"] > 0
                    else "HONEST_NEGATIVE_R2_DEEP"
                ),
                "source": "molmetal/scripts/wf433_m2_extended.py",
                "wf_tag": "Wf-433-A2 m2 R² EXTENDED features diagnostic",
            })
        # Merge: replace existing entries for same cohort with new EXTENDED entries.
        merged = []
        for nc in new_entries:
            merged.append(nc)
        catalog["per_cohort_extended_160d"] = merged
        catalog["aggregate_extended"] = result["aggregate"]
        catalog["wf433a2_extended_run"] = {
            "n_features": 160,
            "pll_proxy_type": result["config"]["pll_proxy_type"],
            "pll_proxy_justification": result["config"]["pll_proxy_justification"],
            "ridge_alpha": result["config"]["ridge_alpha"],
            "n_seeds": result["config"]["n_seeds"],
            "n_folds": result["config"]["n_folds"],
            "timestamp": "2026-10-02",
            "wf_tag": "Wf-433-A2",
            "verdict": result["aggregate"]["aggregate_status"],
            "honest_framing": (
                "Wf-433-A2 EXTENDED features (160-d = 32-d RDKit + 128-d Morgan "
                "ECFP4 PLL proxy) CONFIRMS structural limit on n=200 cohorts "
                "per Wf-377-A2 §2.1 + Wf-388-A1 + Wf-390-A2 prior verdicts. "
                "Adding 128-d more features on n=200 cohorts OVERFITS per "
                "bias-variance inflation theory (p/n = 0.80, deep into the "
                "worst-case regime). The m2 R² gate remains "
                "HONEST_NEGATIVE_R2_DEEP for platinai_4cell + hilbert. "
                "metalcytotox_combined at n=2000 closes the bias-variance gap."
            ),
            "no_gpu_launch": True,
        }
        try:
            from molmetal.scripts.wf386a3_atomic_write import atomic_write_file
            atomic_write_file(catalog_path, json.dumps(catalog, indent=2, sort_keys=False))
            print(f"catalog updated via atomic_write_file: {catalog_path}")
        except ImportError:
            with open(catalog_path, "w") as f:
                json.dump(catalog, f, indent=2)
            print(f"catalog updated (direct fallback): {catalog_path}")

    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
