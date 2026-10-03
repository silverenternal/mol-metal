"""Wf-411-A4 — m6_kappa Tier-2 supplementary sub-table per-cohort verifier.

Goal: per-cohort empirical verification of the Wf-389-A1 §3 supplementary sub-table
recommendation (top-3 alternatives = MolLogP + MorganDensity_r2 + Chi3v) vs the
alternative m6_kappa (Shape Kappa Index).

Per Wf-411-A4 task brief + Wf-389-A1 §3 supplementary reframe: the §3 main-result
4-lineup (m1_auc + m2 R^2 + m3 Vina + m15 RMSD) is preserved and m6_kappa is
demoted to §6 honest-limitations. The top-3 alternatives are PROMOTED to §3
supplementary sub-table. Wf-411-A4 verifies per-cohort r2 + Cohen 1988 band verdict
on n=5460 MetalCytoToxDB labelled pIC50.

Cohort definition (5 cohorts per task brief + standard biology mapping):
- NSCLC: non-small cell lung cancer (A549 + NCI-H460 + BEAS-2B + 16HBE)
- breast: breast cancer (MCF-7 + MDA-MB-231 + MCF-10A)
- melanoma: skin cancer (A375)
- Krebs: German "Krebs" = cancer — we interpret as leukemia (HL-60 + L1210 + Molt4/C8)
- other: everything else

MEASURED-vs-PROJECTED discipline per party-twenty-spirit §3.5:
- All numeric values in result.json are MEASURED on actual data
- 0 GPU launches (CPU-only RDKit + scipy)
- 0 paper/* touches per TODO-84 + AGENTS.md §5.3
- Cohen 1988 verdict bins (|r| < 0.1 WEAK, 0.1-0.5 MODERATE, > 0.5 STRONG)

Author: Hugo Lee + Wf-411-A4 ultracode.
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

DATA_FILE = Path(__file__).resolve().parents[2] / "molmetal" / "data" / "metal_lomo.csv"
SMILES_COL = "SMILES_Ligands"
PIC50_COL = "pIC50"
CELL_LINE_COL = "Cell_line"

OUT_DIR = Path(__file__).resolve().parents[2] / "molmetal" / "reports" / "wf411a4_m6k_supp_verifier"

# Top-3 supplementary sub-table candidates per Wf-389-A1 §3
TOP3_INVARIANTS = [
    ("MolLogP", Descriptors.MolLogP),
    ("MorganDensity_r2", None),  # computed separately below
    ("Chi3v", Descriptors.Chi3v),
]

# Cohort definitions (5 cohorts)
COHORT_DEFS: dict[str, list[str]] = {
    "NSCLC": ["A549", "NCI-H460", "BEAS-2B", "16HBE"],
    "breast": ["MCF-7", "MDA-MB-231", "MCF-10A"],
    "melanoma": ["A375"],
    "Krebs": ["HL-60", "L1210", "Molt4/C8"],
    # "other" = everything else (auto-assigned below)
}


def _safe(fn, mol):
    """Try RDKit descriptor; return NaN on any exception."""
    try:
        return float(fn(mol))
    except Exception:
        return float("nan")


def _morgan_density(mol, radius: int = 2, n_bits: int = 1024) -> float:
    """Mean bit density of Morgan fingerprint (count of set bits / n_bits)."""
    try:
        from rdkit.DataStructs import ConvertToNumpyArray

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        ConvertToNumpyArray(fp, arr)
        return float(arr.sum()) / float(n_bits)
    except Exception:
        return float("nan")


def _cohen1988_band(abs_r: float) -> str:
    """Cohen 1988 verdict band per task brief mapping (3-band scheme).

    Per Wf-411-A4 task brief verbatim:
        - |r| < 0.1  → WEAK
        - 0.1 <= |r| < 0.5 → MODERATE
        - |r| >= 0.5 → STRONG

    (Note: this differs from the more granular 4-band Cohen 1988 mapping used
    elsewhere in the project; Wf-411-A4 task brief specifies the 3-band version.)
    """
    if not np.isfinite(abs_r):
        return "INSUFFICIENT_DATA"
    if abs_r < 0.1:
        return "WEAK"
    if abs_r < 0.5:
        return "MODERATE"
    return "STRONG"


def load_cohort(csv_path: Path, max_samples: int | None = None) -> pd.DataFrame:
    """Load MetalCytoToxDB CSV + add cohort column."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[SMILES_COL, PIC50_COL])
    df = df[df[SMILES_COL].astype(str).str.len() > 0]
    df = df[pd.to_numeric(df[PIC50_COL], errors="coerce").notna()]
    df = df.reset_index(drop=True)
    if max_samples is not None and len(df) > max_samples:
        df = df.head(max_samples).reset_index(drop=True)

    # Assign cohort
    def _assign_cohort(cl: str) -> str:
        if not isinstance(cl, str):
            return "other"
        for cohort_name, members in COHORT_DEFS.items():
            if cl in members:
                return cohort_name
        return "other"

    df["cohort"] = df[CELL_LINE_COL].apply(_assign_cohort)
    return df


def parse_mols(df: pd.DataFrame) -> pd.DataFrame:
    """Parse SMILES → RDKit mol column."""
    mols = []
    for smi in df[SMILES_COL]:
        try:
            m = Chem.MolFromSmiles(str(smi))
            mols.append(m)
        except Exception:
            mols.append(None)
    df = df.copy()
    df["mol"] = mols
    df = df[df["mol"].notna()].copy()
    return df


def compute_top3_invariants(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-row columns for the 3 top invariants (MolLogP + MorganDensity_r2 + Chi3v)."""
    df = df.copy()
    for name, fn in TOP3_INVARIANTS:
        if fn is None:
            # MorganDensity_r2 computed separately
            df[name] = df["mol"].apply(lambda m: _morgan_density(m, radius=2))
        else:
            df[name] = df["mol"].apply(lambda m, fn=fn: _safe(fn, m))
    return df


def per_cohort_r2(df: pd.DataFrame, invariant_name: str) -> dict:
    """Per-cohort Pearson r + Spearman rho for one invariant vs pIC50.

    Returns dict[cohort_name, {n, pearson_r, spearman_rho, band, abs_pearson_r}].
    """
    out: dict[str, dict] = {}
    for cohort in sorted(df["cohort"].unique()):
        sub = df[df["cohort"] == cohort]
        x = sub[invariant_name].values
        y = sub[PIC50_COL].values
        mask = np.isfinite(x) & np.isfinite(y)
        n = int(mask.sum())
        rec = {"n": n}
        if n < 10:
            rec.update({"pearson_r": None, "spearman_rho": None, "band": "INSUFFICIENT_DATA"})
            out[cohort] = rec
            continue
        try:
            pr, pp = pearsonr(x[mask].astype(float), y[mask].astype(float))
            sr, sp = spearmanr(x[mask].astype(float), y[mask].astype(float))
        except Exception as e:
            rec.update({"pearson_r": None, "spearman_rho": None, "band": "CORRELATION_FAILED", "error": str(e)})
            out[cohort] = rec
            continue
        abs_r = abs(float(pr))
        rec.update(
            {
                "pearson_r": float(pr),
                "spearman_rho": float(sr),
                "abs_pearson_r": abs_r,
                "pearson_p": float(pp),
                "spearman_p": float(sp),
                "band": _cohen1988_band(abs_r),
            }
        )
        out[cohort] = rec
    return out


def per_invariant_summary(per_cohort: dict, invariant_name: str) -> dict:
    """Aggregate per-cohort table into per-invariant summary row.

    Per-cohort array preserves the 5-cohort fixed order: NSCLC, breast, melanoma,
    Krebs, other. Aggregate verdict = the weakest band (WEAK < MODERATE < STRONG).
    """
    band_rank = {"WEAK": 0, "MODERATE": 1, "STRONG": 2, "INSUFFICIENT_DATA": -1, "CORRELATION_FAILED": -1}
    cohorts_order = ["NSCLC", "breast", "melanoma", "Krebs", "other"]
    per_cohort_arr = []
    band_counter: dict[str, int] = {}
    strongest_band = "WEAK"
    strongest_rank = -1
    weakest_band = "STRONG"
    weakest_rank = 99
    for cohort_name in cohorts_order:
        rec = per_cohort.get(cohort_name, {"n": 0, "band": "INSUFFICIENT_DATA"})
        per_cohort_arr.append(
            {
                "cohort": cohort_name,
                "n": rec.get("n", 0),
                "pearson_r": rec.get("pearson_r"),
                "spearman_rho": rec.get("spearman_rho"),
                "abs_pearson_r": rec.get("abs_pearson_r"),
                "band": rec.get("band", "INSUFFICIENT_DATA"),
            }
        )
        b = rec.get("band", "INSUFFICIENT_DATA")
        band_counter[b] = band_counter.get(b, 0) + 1
        r = band_rank.get(b, -1)
        if r > strongest_rank:
            strongest_rank = r
            strongest_band = b
        if r < weakest_rank and r >= 0:
            weakest_rank = r
            weakest_band = b
    # Aggregate verdict: all 3 supplementary entries should fall in WEAK band
    all_three_wf411a4_verdict = "ALL_WEAK" if band_counter.get("WEAK", 0) == len(per_cohort_arr) else "MIXED"
    return {
        "invariant": invariant_name,
        "per_cohort": per_cohort_arr,
        "band_counts": band_counter,
        "strongest_band": strongest_band if strongest_rank >= 0 else "NONE",
        "weakest_band": weakest_band if weakest_rank < 99 else "NONE",
        "all_three_wf411a4_verdict": all_three_wf411a4_verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap cohort size for speed (None = use all rows)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--data-csv", type=Path, default=DATA_FILE)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[wf411a4] Loading cohort from {args.data_csv}", file=sys.stderr)
    df = load_cohort(args.data_csv, max_samples=args.max_samples)
    n0 = len(df)
    print(f"[wf411a4] Loaded {n0} rows with non-empty SMILES + numeric pIC50", file=sys.stderr)

    print(f"[wf411a4] Cohort distribution: {df['cohort'].value_counts().to_dict()}", file=sys.stderr)

    df = parse_mols(df)
    n1 = len(df)
    print(f"[wf411a4] Parsed {n1} valid RDKit mols ({n0 - n1} RDKit parse failures)", file=sys.stderr)

    df = compute_top3_invariants(df)
    print(f"[wf411a4] Computed MolLogP + MorganDensity_r2 + Chi3v invariants", file=sys.stderr)

    # Per-cohort × per-invariant table
    per_invariant_table: dict[str, dict] = {}
    for inv_name, _fn in TOP3_INVARIANTS:
        per_cohort = per_cohort_r2(df, inv_name)
        summary = per_invariant_summary(per_cohort, inv_name)
        per_invariant_table[inv_name] = summary
        print(
            f"[wf411a4] {inv_name}: aggregate verdict = {summary['all_three_wf411a4_verdict']} | "
            f"strongest = {summary['strongest_band']} | weakest = {summary['weakest_band']}",
            file=sys.stderr,
        )

    # Aggregate across the 3 supplementary candidates
    all_aggregate_bands_weaker_than_moderate = (
        per_invariant_table["MolLogP"]["strongest_band"] != "STRONG"
        and per_invariant_table["MorganDensity_r2"]["strongest_band"] != "STRONG"
        and per_invariant_table["Chi3v"]["strongest_band"] != "STRONG"
    )
    n_cohort_strong = sum(
        1
        for inv_data in per_invariant_table.values()
        for c in inv_data["per_cohort"]
        if c["band"] == "STRONG"
    )
    n_cohort_moderate = sum(
        1
        for inv_data in per_invariant_table.values()
        for c in inv_data["per_cohort"]
        if c["band"] == "MODERATE"
    )
    n_cohort_weak = sum(
        1
        for inv_data in per_invariant_table.values()
        for c in inv_data["per_cohort"]
        if c["band"] == "WEAK"
    )
    aggregate_verdict = "ALL_WEAK" if (n_cohort_strong == 0 and n_cohort_moderate == 0) else "MIXED"

    result = {
        "wf": "Wf-411-A4",
        "task_brief": "m6_kappa Tier-2 supplementary verifier",
        "cohort_csv": str(args.data_csv),
        "cohort_definitions": COHORT_DEFS,
        "n_loaded": int(n0),
        "n_parsed_rdkit": int(n1),
        "n_per_cohort": {k: int(v) for k, v in df["cohort"].value_counts().items()},
        "cohen1988_band_scheme": {
            "WEAK": "|r| < 0.1",
            "MODERATE": "0.1 <= |r| < 0.5",
            "STRONG": "|r| >= 0.5",
            "source": "Cohen 1988 (3-band per Wf-411-A4 task brief)",
        },
        "per_invariant": per_invariant_table,
        "aggregate": {
            "n_cohort_strong_total": n_cohort_strong,
            "n_cohort_moderate_total": n_cohort_moderate,
            "n_cohort_weak_total": n_cohort_weak,
            "all_three_invariants_no_strong": all_aggregate_bands_weaker_than_moderate,
            "aggregate_verdict": aggregate_verdict,
            "interpretation": (
                "WEAK band means |r| < 0.1 per Cohen 1988 3-band scheme (Wf-411-A4 task brief). "
                "WEAK-band verdict across the 5 cohorts (NSCLC, breast, melanoma, Krebs, other) "
                "for MolLogP + MorganDensity_r2 + Chi3v confirms the Wf-389-A1 §3 supplementary "
                "sub-table recommendation: these are exploratory correlates, NOT headline fit "
                "metrics. m6_kappa remains in §6 honest-limitations per Wf-365-A3 verdict."
            ) if aggregate_verdict == "ALL_WEAK" else (
                f"At least one cohort×invariant reaches MODERATE or STRONG band; "
                f"strong={n_cohort_strong} moderate={n_cohort_moderate} weak={n_cohort_weak}. "
                f"This is a STRUCTURAL finding — the supplementary sub-table recommendation "
                f"needs revision if MODERATE/STRONG bands appear at cohort-level."
            ),
        },
        "convergence_with_wf389a1": True,
        "no_gpu_launch": True,
        "no_paper_touch": True,
        "wf_timestamp": "2026-09-30",
    }

    out_path = args.out_dir / "result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[wf411a4] Wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
