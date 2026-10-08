#!/usr/bin/env python3
"""L5 Hilbert-ASTSR feature-ceiling probe (CPU-only).

Measures the ACHIEVABLE LINEAR CEILING (ridge / OLS R^2) of candidate
feature sets vs PlatinAI 4-cell pIC50, to decide whether a richer feature
set can carry real signal past the documented 12-d RDKit signal floor.

Honest framing: the number printed is an in-sample/held-out R^2 on REAL
labels. If the ceiling stays ~0, L5 is a GENUINE honest-negative and no
search (or GPU retrain) will lift it.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV = REPO_ROOT / "molmetal/data/platinai_4cell.csv"
CELL_TARGET = {
    "mcf7": "ic50_mcf7_nm",
    "a549": "ic50_a549_nm",
    "a2780": "ic50_a2780_nm",
    "a2780cis": "ic50_a2780cis_nm",
}


def _read_rows() -> List[Dict[str, str]]:
    import csv as _csv

    with open(CSV, "r", encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in _csv.DictReader(f)]


def _pic50(ic50_nm: float) -> float:
    return -math.log10(max(ic50_nm / 1000.0 * 1e-6, 1e-12))


# ---------------------------------------------------------------------------
# Feature builders. Each returns (names, matrix) computed from RDKit mol.
# ---------------------------------------------------------------------------
_RDKIT12_ATTRS = [
    ("MolWt", "MolWt"),
    ("MolLogP", "MolLogP"),
    ("TPSA", "TPSA"),
    ("HBA", "NumHAcceptors"),
    ("HBD", "NumHDonors"),
    ("NRotB", "NumRotatableBonds"),
    ("HeavyAtoms", "HeavyAtomCount"),
    ("RingCount", "RingCount"),
    ("NumAromaticRings", "NumAromaticRings"),
    ("FractionCSP3", "FractionCSP3"),
    ("MolMR", "MolMR"),
    ("BalabanJ", "BalabanJ"),
]

# A richer 2D set: canonical RDKit descriptors + metal-relevant extras.
_RDKIT_RICH_ATTRS = _RDKIT12_ATTRS + [
    ("BertzCT", "BertzCT"),
    ("Chi0v", "Chi0v"),
    ("Chi0n", "Chi0n"),
    ("Chi1n", "Chi1n"),
    ("Chi1v", "Chi1v"),
    ("Chi2n", "Chi2n"),
    ("Chi2v", "Chi2v"),
    ("Chi3n", "Chi3n"),
    ("Chi4n", "Chi4n"),
    ("HallKierAlpha", "HallKierAlpha"),
    ("Kappa1", "Kappa1"),
    ("Kappa2", "Kappa2"),
    ("Kappa3", "Kappa3"),
    ("LabuteASA", "LabuteASA"),
    ("PEOE_VSA1", "PEOE_VSA1"),
    ("PEOE_VSA2", "PEOE_VSA2"),
    ("PEOE_VSA3", "PEOE_VSA3"),
    ("PEOE_VSA6", "PEOE_VSA6"),
    ("PEOE_VSA10", "PEOE_VSA10"),
    ("SMR_VSA1", "SMR_VSA1"),
    ("SMR_VSA5", "SMR_VSA5"),
    ("SlogP_VSA1", "SlogP_VSA1"),
    ("SlogP_VSA2", "SlogP_VSA2"),
    ("MaxPartialCharge", "MaxPartialCharge"),
    ("MinPartialCharge", "MinPartialCharge"),
    ("MaxAbsPartialCharge", "MaxAbsPartialCharge"),
    ("MinAbsPartialCharge", "MinAbsPartialCharge"),
    ("NumAliphaticRings", "NumAliphaticRings"),
    ("NumSaturatedRings", "NumSaturatedRings"),
    ("NumHeteroatoms", "NumHeteroatoms"),
    ("NHOHCount", "NHOHCount"),
    ("NOCount", "NOCount"),
    ("NumValenceElectrons", "NumValenceElectrons"),
    ("NumRadicalElectrons", "NumRadicalElectrons"),
    ("fr_halogen", "fr_halogen"),
    ("fr_Al_COO", "fr_Al_COO"),
    ("fr_Ar_N", "fr_Ar_N"),
    ("fr_NH0", "fr_NH0"),
    ("fr_NH1", "fr_NH1"),
    ("fr_C_O", "fr_C_O"),
    ("fr_ester", "fr_ester"),
    ("fr_amide", "fr_amide"),
]


def _mol(smiles: str):
    from rdkit import Chem

    return Chem.MolFromSmiles(smiles)


def build_12d(smiles: str) -> Optional[List[float]]:
    from rdkit.Chem import Descriptors

    m = _mol(smiles)
    if m is None:
        return None
    out = []
    for _n, attr in _RDKIT12_ATTRS:
        try:
            out.append(float(getattr(Descriptors, attr)(m)))
        except Exception:
            out.append(float("nan"))
    return out


def build_rich2d(smiles: str) -> Optional[List[float]]:
    from rdkit.Chem import Descriptors, rdMolDescriptors

    m = _mol(smiles)
    if m is None:
        return None
    out = []
    for _n, attr in _RDKIT_RICH_ATTRS:
        try:
            fn = getattr(Descriptors, attr, None)
            if fn is None:
                fn = getattr(rdMolDescriptors, attr, None)
            out.append(float(fn(m)) if fn is not None else float("nan"))
        except Exception:
            out.append(float("nan"))
    return out


def build_3d(smiles: str) -> Optional[List[float]]:
    """3D / conformer descriptors from RDKit ETKDG embedding (heavy only)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors3D, rdMolDescriptors

    m = _mol(smiles)
    if m is None:
        return None
    try:
        mh = Chem.AddHs(m)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xC0FFEE
        cid = AllChem.EmbedMolecule(mh, params)
        if cid != 0:
            # retry without chirality constraints, random coords
            params.useRandomCoords = True
            cid = AllChem.EmbedMolecule(mh, params)
        if cid != 0:
            return None
        AllChem.MMFFOptimizeMolecule(mh, maxIters=200)
        names = [
            ("Asphericity", Descriptors3D.Asphericity),
            ("Eccentricity", Descriptors3D.Eccentricity),
            ("InertialShapeFactor", Descriptors3D.InertialShapeFactor),
            ("NPR1", Descriptors3D.NPR1),
            ("NPR2", Descriptors3D.NPR2),
            ("PMI1", Descriptors3D.PMI1),
            ("PMI2", Descriptors3D.PMI2),
            ("PMI3", Descriptors3D.PMI3),
            ("RadiusOfGyration", Descriptors3D.RadiusOfGyration),
            ("SpherocityIndex", Descriptors3D.SpherocityIndex),
        ]
        rr = AllChem.GetBestRMS if False else None  # placeholder to avoid lint
        out = []
        for _n, fn in names:
            try:
                out.append(float(fn(mh)))
            except Exception:
                out.append(float("nan"))
        out.append(float(rdMolDescriptors.CalcNumRotatableBonds(mh)))
        return out
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Ceiling estimator: OLS in-sample R^2 + 5-fold CV R^2 (ridge).
# ---------------------------------------------------------------------------
def _ols_r2(X: np.ndarray, y: np.ndarray) -> float:
    A = np.column_stack([np.ones(len(X)), X])
    coef, _res, _rank, _sv = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _cv_r2(X: np.ndarray, y: np.ndarray, folds: int, seed: int, alpha: float) -> float:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    preds = np.zeros_like(y)
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        mdl = Ridge(alpha=alpha).fit(sc.transform(X[tr]), y[tr])
        preds[te] = mdl.predict(sc.transform(X[te]))
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _feature_target_corr(X: np.ndarray, names: List[str], y: np.ndarray, k: int = 5):
    from scipy.stats import pearsonr

    rows = []
    for j, nm in enumerate(names):
        col = X[:, j]
        if np.std(col) == 0:
            rows.append((nm, 0.0))
            continue
        try:
            r = abs(float(pearsonr(col, y)[0]))
        except Exception:
            r = 0.0
        rows.append((nm, r))
    rows.sort(key=lambda t: -t[1])
    return rows[:k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=list(CELL_TARGET))
    ap.add_argument("--n-max", type=int, default=5000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--sets", nargs="+", default=["12d", "rich2d", "3d"])
    args = ap.parse_args()

    rows = _read_rows()
    builders = {"12d": (build_12d, [n for n, _ in _RDKIT12_ATTRS]),
                "rich2d": (build_rich2d, [n for n, _ in _RDKIT_RICH_ATTRS]),
                "3d": (build_3d, ["Asphericity", "Eccentricity", "InertialShapeFactor",
                                  "NPR1", "NPR2", "PMI1", "PMI2", "PMI3",
                                  "RadiusOfGyration", "SpherocityIndex", "NRotB_3d"])}

    # Precompute per-SMILES feature dicts once.
    feats: Dict[str, Dict[str, Optional[List[float]]]] = {}
    targets: Dict[str, Dict[str, float]] = {}
    smiles_list: List[str] = []
    for r in rows:
        smi = (r.get("smiles") or "").strip()
        if not smi:
            continue
        tgts = {}
        ok = True
        for cell, col in CELL_TARGET.items():
            try:
                v = float(r.get(col))
            except (TypeError, ValueError):
                ok = False
                break
            if not math.isfinite(v) or v <= 0:
                ok = False
                break
            tgts[cell] = _pic50(v)
        if not ok:
            continue
        fdict = {s: builders[s][0](smi) for s in args.sets}
        if all(fdict[s] is None for s in args.sets):
            continue
        feats[smi] = fdict
        targets[smi] = tgts
        smiles_list.append(smi)
        if len(smiles_list) >= args.n_max:
            break

    n = len(smiles_list)
    print(f"[probe] n_rows={n} cells={args.cells} sets={args.sets} folds={args.folds} seeds={args.seeds}")

    results = {}
    for cell in args.cells:
        y = np.array([targets[s][cell] for s in smiles_list], dtype=float)
        results[cell] = {}
        for s in args.sets:
            valid = [i for i, smi in enumerate(smiles_list) if feats[smi][s] is not None]
            if len(valid) < 50:
                print(f"  {cell:9s} {s:7s} SKIP (only {len(valid)} mols featurized)")
                continue
            names = builders[s][1]
            X = np.array([feats[smiles_list[i]][s] for i in valid], dtype=float)
            yv = y[valid]
            # impute nan by column median
            col_med = np.nanmedian(X, axis=0)
            inds = np.where(np.isnan(X))
            if len(inds[0]):
                X[inds] = np.take(col_med, inds[1])
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            ols = _ols_r2(X, yv)
            cvs = [_cv_r2(X, yv, args.folds, sd, args.alpha) for sd in args.seeds]
            top = _feature_target_corr(X, names, yv, k=5)
            results[cell][s] = {"n": len(valid), "d": X.shape[1], "ols_R2": ols,
                                "cv_R2_mean": float(np.mean(cvs)),
                                "cv_R2_std": float(np.std(cvs)),
                                "top_rho": top}
            print(f"  {cell:9s} {s:7s} n={len(valid):4d} d={X.shape[1]:3d} "
                  f"OLS_R2={ols:+.4f} CV_R2={np.mean(cvs):+.4f}+-{np.std(cvs):.4f} "
                  f"max|rho|={top[0][1]:.4f} ({top[0][0]})")

    # Aggregate mean CV_R2 per set (across cells) for the decision.
    print("\n[aggregate] mean CV_R2 over cells (per set):")
    for s in args.sets:
        vals = [results[c][s]["cv_R2_mean"] for c in args.cells if s in results[c]]
        olss = [results[c][s]["ols_R2"] for c in args.cells if s in results[c]]
        if vals:
            print(f"  {s:7s} CV_R2_mean={np.mean(vals):+.4f}  OLS_R2_mean={np.mean(olss):+.4f}")

    out = REPO_ROOT / "molmetal/reports/l5_feature_ceiling_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    import json

    out.write_text(json.dumps({
        "metric": "hilbert_4cell_r2",
        "purpose": "L5 feature-ceiling probe — does a richer feature set carry real signal?",
        "csv": str(CSV.relative_to(REPO_ROOT)),
        "n_rows": n,
        "folds": args.folds,
        "seeds": args.seeds,
        "ridge_alpha": args.alpha,
        "results": results,
        "tag": "MEASURED",
    }, indent=2))
    print(f"\n[saved] {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
