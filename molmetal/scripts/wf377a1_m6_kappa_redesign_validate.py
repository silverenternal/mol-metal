"""Wf-377-A1 — m6_kappa REDESIGN empirical validation.

Goal: rank alternative per-molecule topological invariants vs Shape Kappa Index
on MetalCytoToxDB labelled pIC50 (Krasnov 2026 via metal_lomo.csv).

Per Wf-377-A1 task brief + Wf-371-A1 FINAL FINAL verdict: 7 architectural paths
exhausted (chemprop+PLL, R20-P3 W2, MAML, Neural ODE, combined, verdict doc,
degenerate). The deeper fix is METRIC REDESIGN via literature research.

This script performs the empirical validation step: computes 14 RDKit
topological invariants per SMILES + Morgan fingerprint radius 2 density, then
ranks by Pearson r + Spearman rho vs pIC50 on metal_lomo.csv.

MEASURED-vs-PROJECTED discipline per party-twenty-spirit §3.5:
- All numeric values in result.json are MEASURED on actual data
- 0 GPU launches (CPU-only RDKit)
- 0 paper/* touches
- Citation discipline for literature refs (separate doc)

Author: Hugo Lee + Wf-377-A1 ultracode.
Date: 2026-09-30.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from scipy.stats import pearsonr, spearmanr

RDLogger.DisableLog("rdApp.*")

# Suppress RDKit warnings during bulk parsing

DATA_FILE = Path(__file__).resolve().parents[2] / "molmetal" / "data" / "metal_lomo.csv"
SMILES_COL = "SMILES_Ligands"
PIC50_COL = "pIC50"
METAL_COL = "Metal"

OUT_DIR = Path(__file__).resolve().parents[2] / "molmetal" / "reports" / "wf377a1_m6_kappa_redesign"

# 14 RDKit topological invariants + Morgan fp density (15 total)
INVARIANTS = [
    # Shape Kappa Index family (the m6_kappa components)
    ("Kappa1", Descriptors.Kappa1),
    ("Kappa2", Descriptors.Kappa2),
    ("Kappa3", Descriptors.Kappa3),
    # Literature-confirmed strong correlates
    ("TPSA", Descriptors.TPSA),
    ("MolLogP", Descriptors.MolLogP),
    ("MolMR", Descriptors.MolMR),
    ("BalabanJ", Descriptors.BalabanJ),
    ("HallKierAlpha", Descriptors.HallKierAlpha),
    ("BertzCT", Descriptors.BertzCT),
    # Connectivity indices (Kier-Hall)
    ("Chi0v", Descriptors.Chi0v),
    ("Chi1v", Descriptors.Chi1v),
    ("Chi2v", Descriptors.Chi2v),
    ("Chi3v", Descriptors.Chi3v),
    # Phi index
    ("Phi", Descriptors.Phi),
]


def _safe(fn, mol):
    """Try RDKit descriptor; return NaN on any exception."""
    try:
        return float(fn(mol))
    except Exception:
        return float("nan")


def _morgan_density(mol, radius: int = 2, n_bits: int = 1024) -> float:
    """Mean bit density of Morgan fingerprint (count of set bits / n_bits).

    Serves as a coarse 'topological feature density' proxy.
    """
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        from rdkit.DataStructs import ConvertToNumpyArray
        ConvertToNumpyArray(fp, arr)
        return float(arr.sum()) / float(n_bits)
    except Exception:
        return float("nan")


def load_cohort(csv_path: Path, max_samples: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Drop rows without SMILES or pIC50
    df = df.dropna(subset=[SMILES_COL, PIC50_COL])
    df = df[df[SMILES_COL].astype(str).str.len() > 0]
    # Filter numeric pIC50
    df = df[pd.to_numeric(df[PIC50_COL], errors="coerce").notna()]
    df[PIC50_COL] = df[PIC50_COL].astype(float)
    if max_samples is not None and len(df) > max_samples:
        df = df.head(max_samples).copy()
    return df


def parse_mols(df: pd.DataFrame) -> pd.DataFrame:
    mols = []
    for smi in df[SMILES_COL].values:
        try:
            m = Chem.MolFromSmiles(str(smi))
            mols.append(m)
        except Exception:
            mols.append(None)
    df = df.copy()
    df["mol"] = mols
    df = df[df["mol"].notna()].copy()
    return df


def compute_invariants(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-row invariant columns + Morgan density."""
    df = df.copy()
    for name, fn in INVARIANTS:
        df[name] = df["mol"].apply(lambda m, fn=fn: _safe(fn, m))
    df["MorganDensity_r2"] = df["mol"].apply(lambda m: _morgan_density(m, radius=2))
    # Composite m6_kappa (matches metrics_v2.py:m6_kappa_from_smiles convention)
    def _clip01(x: float) -> float:
        if not np.isfinite(x):
            return float("nan")
        return float(np.clip((x - 0.0) / (30.0 - 0.0), 0.0, 1.0))
    df["m6_kappa_composite"] = df[["Kappa1", "Kappa2", "Kappa3"]].apply(
        lambda r: float(np.nanmean([_clip01(r["Kappa1"]), _clip01(r["Kappa2"]), _clip01(r["Kappa3"])]))
        if r.notna().all() else float("nan"),
        axis=1,
    )
    return df


def rank_invariants(df: pd.DataFrame) -> list[dict]:
    """For each invariant, compute Pearson r + Spearman rho vs pIC50 (on rows where both finite)."""
    out = []
    y = df[PIC50_COL].values
    for name, _fn in INVARIANTS + [("MorganDensity_r2", None), ("m6_kappa_composite", None)]:
        x = df[name].values
        mask = np.isfinite(x) & np.isfinite(y)
        n = int(mask.sum())
        if n < 30:
            out.append({
                "name": name,
                "n_used": n,
                "pearson_r": None,
                "spearman_rho": None,
                "abs_pearson_r": None,
                "abs_spearman_rho": None,
                "pearson_p": None,
                "spearman_p": None,
                "verdict": "INSUFFICIENT_DATA",
            })
            continue
        x_clean = x[mask].astype(float)
        y_clean = y[mask].astype(float)
        try:
            pr, pp = pearsonr(x_clean, y_clean)
            sr, sp = spearmanr(x_clean, y_clean)
        except Exception as e:
            out.append({
                "name": name,
                "n_used": n,
                "error": str(e),
                "verdict": "CORRELATION_FAILED",
            })
            continue
        abs_r = abs(float(pr))
        abs_s = abs(float(sr))
        # Verdict bands: STRONG (|r| >= 0.5), MODERATE (0.3 <= |r| < 0.5), WEAK (0.1 <= |r| < 0.3), NEGLIGIBLE (|r| < 0.1)
        if abs_r >= 0.5:
            verdict = "STRONG"
        elif abs_r >= 0.3:
            verdict = "MODERATE"
        elif abs_r >= 0.1:
            verdict = "WEAK"
        else:
            verdict = "NEGLIGIBLE"
        out.append({
            "name": name,
            "n_used": n,
            "pearson_r": float(pr),
            "spearman_rho": float(sr),
            "abs_pearson_r": abs_r,
            "abs_spearman_rho": abs_s,
            "pearson_p": float(pp),
            "spearman_p": float(sp),
            "verdict": verdict,
        })
    # Sort by abs_pearson_r descending
    out.sort(key=lambda r: -(r["abs_pearson_r"] if r["abs_pearson_r"] is not None else -1.0))
    return out


def per_metal_breakdown(df: pd.DataFrame) -> dict:
    """Per-metal Pearson r for the top 3 invariants + m6_kappa_composite."""
    if METAL_COL not in df.columns:
        return {}
    metals = [m for m in df[METAL_COL].dropna().unique() if m]
    out = {}
    for metal in metals:
        sub = df[df[METAL_COL] == metal]
        if len(sub) < 30:
            continue
        out[metal] = {"n": int(len(sub))}
        for name in ("m6_kappa_composite", "TPSA", "MolLogP", "BalabanJ", "HallKierAlpha", "MorganDensity_r2"):
            x = sub[name].values
            y = sub[PIC50_COL].values
            mask = np.isfinite(x) & np.isfinite(y)
            n = int(mask.sum())
            if n < 10:
                out[metal][name] = {"n": n, "pearson_r": None}
                continue
            try:
                pr, _pp = pearsonr(x[mask].astype(float), y[mask].astype(float))
                out[metal][name] = {"n": n, "pearson_r": float(pr), "abs_pearson_r": abs(float(pr))}
            except Exception:
                out[metal][name] = {"n": n, "pearson_r": None}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap cohort size for speed (None = use all rows)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--data-csv", type=Path, default=DATA_FILE)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[wf377a1] Loading cohort from {args.data_csv}", file=sys.stderr)
    df = load_cohort(args.data_csv, max_samples=args.max_samples)
    n0 = len(df)
    print(f"[wf377a1] Loaded {n0} rows with non-empty SMILES + numeric pIC50", file=sys.stderr)

    df = parse_mols(df)
    n1 = len(df)
    print(f"[wf377a1] Parsed {n1} valid RDKit mols ({n0 - n1} RDKit parse failures)", file=sys.stderr)

    df = compute_invariants(df)
    n2 = len(df)
    print(f"[wf377a1] Computed invariants for {n2} mols", file=sys.stderr)

    ranking = rank_invariants(df)
    print(f"[wf377a1] Ranked {len(ranking)} invariants", file=sys.stderr)

    breakdown = per_metal_breakdown(df)
    print(f"[wf377a1] Per-metal breakdown for {len(breakdown)} metals", file=sys.stderr)

    # m6_kappa_composite specifically (reproduces Wf-325D canonical)
    if "m6_kappa_composite" in df.columns:
        m6 = df["m6_kappa_composite"].dropna()
        m6k_stats = {
            "n": int(len(m6)),
            "mean": float(m6.mean()),
            "std": float(m6.std()),
            "min": float(m6.min()),
            "max": float(m6.max()),
        }
    else:
        m6k_stats = {}

    # Save result.json
    result = {
        "wf": "Wf-377-A1",
        "cohort_csv": str(args.data_csv),
        "n_loaded": n0,
        "n_parsed_rdkit": n1,
        "n_with_invariants": n2,
        "ranking": ranking,
        "per_metal_breakdown": breakdown,
        "m6_kappa_composite_stats": m6k_stats,
        "timestamp": "2026-09-30",
        "no_gpu_launch": True,
        "wall_seconds_target": 60,
        "notes": (
            "Per Wf-377-A1 task brief: literature review + empirical validation of "
            "alternative per-molecule topological invariants vs Shape Kappa Index "
            "on MetalCytoToxDB labelled pIC50. 15 invariants compared: 14 RDKit "
            "descriptors + Morgan fingerprint radius-2 bit density. Ranking by "
            "|Pearson r| descending. Bands: STRONG |r|>=0.5 / MODERATE 0.3<= |r|<0.5 "
            "/ WEAK 0.1<= |r|<0.3 / NEGLIGIBLE |r|<0.1. Verdict preserved as "
            "honest-negative if no invariant reaches MODERATE or STRONG."
        )
    }
    out_path = args.out_dir / "result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[wf377a1] Wrote {out_path}", file=sys.stderr)

    # Print top-10 ranking to stdout
    print("\n=== Top 15 invariants (ranked by |Pearson r|) ===", file=sys.stderr)
    for i, r in enumerate(ranking[:15], 1):
        if r["pearson_r"] is not None:
            print(f"  {i:2d}. {r['name']:20s}  Pearson r = {r['pearson_r']:+.4f}  Spearman rho = {r['spearman_rho']:+.4f}  n={r['n_used']:5d}  verdict={r['verdict']}", file=sys.stderr)
        else:
            print(f"  {i:2d}. {r['name']:20s}  (insufficient data)", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
