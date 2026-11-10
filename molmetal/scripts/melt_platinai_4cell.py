#!/usr/bin/env python3
"""Melt the wide PlatinAI 4-cell-line CSV into the canonical long format
required by molmetal/data/cytotox.py.

=====================================================================
Wf-131 P3 — Pt × 4-cell-line LOMO data preparation
=====================================================================

Source:  /home/hugo/codes/mol-metal/.data/molmetal/platinai_4cell.csv
Schema:  SMILES_Ligands, pIC50_mcf7, pIC50_a549, pIC50_a2780, pIC50_a2780cis, metal
Rows:    4078 unique Pt complexes × 4 cell-lines (Rusanov 2025 §1.2 anchor)

Output:  /home/hugo/codes/mol-metal/.data/molmetal/platinai_4cell_long.csv
Schema:  smiles, metal, scaffold_id, split, activity, SMILES_Ligands, Metal,
         IC50_Dark_value, pic50, cell_line, source
Rows:    4078 * 4 = ~16312 long-format rows

HONEST FRAMING (party-twenty-spirit §3.5)
-----------------------------------------
The wide-format source platinai_4cell.csv is SYNTHETIC per the
.data/molmetal/platinai_4cell.meta.json honest-negative note:
"labels generated via regression on DFT features (HOMO/LUMO/Gap/Polarizability)
not from Rusanov 2025 supplementary S4-S5 tables (those are behind a
JavaScript Cloudflare challenge and not directly fetchable from this host)".

The melted long-format CSV preserves this honest framing:
- Source CSV is Rusanov-2025-anchored Pt complex corpus (4078 unique SMILES)
- Per-cell pIC50 values are SYNTHETIC regression-derived, NOT real
- Useful as pipeline test + scaffold-split sanity, NOT as paper-grade m2.

Usage:
    python molmetal/scripts/melt_platinai_4cell.py --out .data/molmetal/platinai_4cell_long.csv
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd


PLATINAI_4CELL_CSV = (
    Path(__file__).resolve().parents[2] / ".data" / "molmetal" / "platinai_4cell.csv"
)
DEFAULT_OUT_CSV = (
    Path(__file__).resolve().parents[2]
    / ".data"
    / "molmetal"
    / "platinai_4cell_long.csv"
)
# Optional: 5-metal CSV (Ir/Os/Re/Rh/Ru, ~26k) — concatenated with the melted
# PlatinAI 4-cell Pt rows so LOMO Pt has a non-empty TRAIN fold
# (Ir/Os/Re/Rh/Ru) and a Pt TEST fold (4078 × 4 cell-lines).
DEFAULT_FIVEMETAL_CSV = (
    Path(__file__).resolve().parents[2]
    / ".data"
    / "molmetal"
    / "metalcytotox_5metal_multitask_v4.csv"
)
DEFAULT_MERGED_CSV = (
    Path(__file__).resolve().parents[2]
    / ".data"
    / "molmetal"
    / "platinai_4cells_with_5metal_train.csv"
)

CELL_COLUMNS = [
    ("mcf7",  "pIC50_mcf7"),
    ("a549",  "pIC50_a549"),
    ("a2780", "pIC50_a2780"),
    ("a2780cis", "pIC50_a2780cis"),
]


def _scaffold_id(smiles: str) -> str:
    """Stable per-SMILES scaffold_id (Bemis-Murcko if RDKit available; else sha1 prefix)."""
    try:
        from rdkit import Chem  # noqa: F401  (RDKit is in project venv)
        from rdkit.Chem.Scaffolds import MurckoScaffold
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        pass
    return "sc_" + hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:12]


def melt(src: Path, dst: Path) -> pd.DataFrame:
    df = pd.read_csv(src, low_memory=False)
    required = {"SMILES_Ligands", "metal"} | {c for _, c in CELL_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"platinai_4cell.csv missing columns: {sorted(missing)}")

    # Group by SMILES so each unique complex → 4 rows (one per cell line)
    out_rows = []
    n_no_data = 0
    for smiles, sub in df.groupby("SMILES_Ligands", sort=False):
        smi = str(smiles)
        scaf = _scaffold_id(smi)
        metal_token = str(sub["metal"].iloc[0])
        for cell_short, col in CELL_COLUMNS:
            pic50_val = sub[col].iloc[0]
            if pd.isna(pic50_val):
                n_no_data += 1
                continue
            pic50 = float(pic50_val)
            # IC50_Dark_value in micromolar: 10^(9 - pIC50)
            ic50_um = 10.0 ** (9.0 - pic50)
            activity = bool(ic50_um < 10.0)
            out_rows.append({
                "smiles": smi,
                "metal": metal_token,
                "scaffold_id": scaf,
                "split": "train",          # placeholder; trainer handles actual split
                "activity": int(activity),
                "SMILES_Ligands": smi,
                "Metal": metal_token,
                "IC50_Dark_value": ic50_um,
                "pic50": pic50,
                "cell_line": cell_short,
                "source": "PlatinAI_4cell_synthetic_Rusanov2025_§1.2",
            })

    long_df = pd.DataFrame(out_rows)
    if n_no_data:
        print(f"[melt_platinai_4cell] WARN: {n_no_data} rows had NaN pIC50 (dropped)", file=sys.stderr)

    dst.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(dst, index=False)
    return long_df


def merge_with_5metal(platinai_long: Path, fivemetal: Path, merged: Path) -> pd.DataFrame:
    """Concatenate the melted Pt long-format with the 5-metal CSV.

    Required schema columns (canonical MetalCytoToxDB loader expects):
        smiles, metal, scaffold_id, split, activity, SMILES_Ligands, Metal,
        IC50_Dark_value, pic50, cell_line, source

    Returns the merged dataframe. The 5-metal CSV is used as the train fold
    (Ir/Os/Re/Rh/Ru); PlatinAI melted rows form the Pt test fold.
    """
    plat = pd.read_csv(platinai_long, low_memory=False)
    fm = pd.read_csv(fivemetal, low_memory=False)
    # Schema alignment: the 5metal CSV already has all canonical columns;
    # platinai_long also has all canonical columns. Direct stack:
    missing_cols = set(plat.columns) - set(fm.columns)
    extra_cols = set(fm.columns) - set(plat.columns)
    if missing_cols:
        for c in missing_cols:
            fm[c] = pd.NA
    if extra_cols:
        for c in extra_cols:
            plat[c] = pd.NA
    # Re-order to canonical schema
    canonical = ["smiles", "metal", "scaffold_id", "split", "activity",
                 "SMILES_Ligands", "Metal", "IC50_Dark_value", "pic50",
                 "cell_line", "source"]
    plat = plat.reindex(columns=canonical)
    fm = fm.reindex(columns=canonical)
    merged_df = pd.concat([fm, plat], ignore_index=True, sort=False)
    merged_df.to_csv(merged, index=False)
    return merged_df


def main() -> int:
    p = argparse.ArgumentParser(description="Melt platinai_4cell wide CSV → long format")
    p.add_argument("--src", type=Path, default=PLATINAI_4CELL_CSV)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_CSV)
    p.add_argument("--fivemetal-csv", type=Path, default=DEFAULT_FIVEMETAL_CSV,
                   help="5-metal CSV (Ir/Os/Re/Rh/Ru) for train fold")
    p.add_argument("--merged-out", type=Path, default=DEFAULT_MERGED_CSV,
                   help="Merged (5metal + melted Pt) CSV for LOMO Pt")
    args = p.parse_args()

    if not args.src.exists():
        print(f"FAIL: source missing: {args.src}", file=sys.stderr)
        return 1

    long_df = melt(args.src, args.out)
    # Degenerate-run gate: refuse to report success when the melt produced
    # no rows or no metal coverage.
    if len(long_df) == 0 or long_df["metal"].nunique() < 1:
        print(
            "[FAIL] melt produced no rows / no metal coverage",
            file=sys.stderr,
        )
        return 1
    print(f"[melt_platinai_4cell] OK: {len(long_df)} long-format rows → {args.out}")
    print(f"[melt_platinai_4cell] unique SMILES: {long_df['smiles'].nunique()}")
    print(f"[melt_platinai_4cell] cell lines: {sorted(long_df['cell_line'].unique())}")
    print(f"[melt_platinai_4cell] active rate: {long_df['activity'].mean():.3f}")

    # Optional: build merged CSV if 5metal exists
    if args.fivemetal_csv.exists():
        merged_df = merge_with_5metal(args.out, args.fivemetal_csv, args.merged_out)
        print(f"[melt_platinai_4cell] merged {len(merged_df)} rows → {args.merged_out}")
        print(f"[melt_platinai_4cell] metals: {sorted(merged_df['metal'].dropna().unique())}")
    else:
        print(f"[melt_platinai_4cell] SKIP merged CSV (5metal not found: {args.fivemetal_csv})",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())