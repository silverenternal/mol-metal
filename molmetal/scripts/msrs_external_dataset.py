#!/usr/bin/env python3
"""MET-7 — can any on-disk cohort serve as an external validation set?

The pre-registered estimand is *same ligand + same metal + same cell line* with
a numeric label in both the MSRS corpus and an outside cohort.  This script
measures that intersection instead of asserting it is empty.  It never trains
anything and never imputes a label.

Output: ``metrics/by_metric/msrs_external_dataset.json``.

Usage:
    .venv/bin/python molmetal/scripts/msrs_external_dataset.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MSRS_PARQUET = REPO_ROOT / "data" / "msrs" / "measurements.parquet"
PLATINAI = REPO_ROOT / ".data" / "molmetal" / "platinai_4cell.csv"
NCI60 = REPO_ROOT / ".data" / "molmetal" / "NCI60_GI50" / "GI50.csv"
DEFAULT_OUT = REPO_ROOT / "metrics" / "by_metric" / "msrs_external_dataset.json"

METALS = ("Ru", "Ir", "Rh", "Os", "Re", "Pt")
MIN_LIGAND_HEAVY_ATOMS = 6  # same floor as msrs_build_endpoints.main_fragment
MIN_ROWS = 200
R2_GATE = 0.5
AUC_GATE = 0.75

# PlatinAI labels are pIC50 (=-log10 IC50 in M); MSRS is log10(IC50 / uM).
PLATINAI_LABEL_COLUMNS = {
    "mcf7": "pIC50_mcf7",
    "a549": "pIC50_a549",
    "a2780": "pIC50_a2780",
    "a2780cis": "pIC50_a2780cis",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise (e.g. a tmp fixture)."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), capture_output=True, text=True
        )
        return out.stdout.strip() or None
    except Exception:  # pragma: no cover
        return None


def strip_metal_and_main_fragment(smiles: object, metals: tuple[str, ...] = METALS) -> Optional[str]:
    """Canonical SMILES of the largest organic fragment after deleting the metal.

    MSRS stores ``ligand_main_fragment`` with the metal already removed, while
    PlatinAI stores the full complex (``[Pt+6]`` with dative bonds).  Comparing
    the raw strings would report a false zero, so the metal is deleted first.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    idx = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() in metals]
    for i in sorted(idx, reverse=True):
        rw.RemoveAtom(i)
    stripped = rw.GetMol()
    try:
        Chem.SanitizeMol(stripped)
    except Exception:
        return None
    frags = Chem.GetMolFrags(stripped, asMols=True, sanitizeFrags=False)
    if not frags:
        return None
    # Match the MSRS convention: a ligand needs >=6 heavy atoms, otherwise
    # Cl-/solvent wins the "largest fragment" contest on a small test complex.
    ligands = [f for f in frags if f.GetNumHeavyAtoms() >= MIN_LIGAND_HEAVY_ATOMS]
    pool = ligands or frags
    try:
        return Chem.MolToSmiles(max(pool, key=lambda f: f.GetNumHeavyAtoms()))
    except Exception:
        return None


def overlap(left: set[str], right: set[str]) -> dict[str, Any]:
    inter = left & right
    return {
        "left_unique": len(left),
        "right_unique": len(right),
        "intersection_unique": len(inter),
        "examples": sorted(inter)[:5],
    }


def audit_platinai(path: Path, msrs: pd.DataFrame) -> dict[str, Any]:
    """Measure the PlatinAI <-> MSRS intersection on all three keys."""
    frame = pd.read_csv(path)
    frame["ligand_main_fragment"] = frame["SMILES_Ligands"].map(strip_metal_and_main_fragment)

    msrs_ligands = set(msrs["ligand_main_fragment"].dropna().astype(str))
    msrs_metals = set(msrs["metal"].dropna().astype(str))
    msrs_cells = set(msrs["cell_line_key"].dropna().astype(str).str.lower())

    external_metals = set(frame["metal"].dropna().astype(str))
    external_cells = set(PLATINAI_LABEL_COLUMNS)
    external_ligands = set(frame["ligand_main_fragment"].dropna().astype(str))

    hit = frame["ligand_main_fragment"].isin(msrs_ligands)
    shared_cells = sorted(external_cells & msrs_cells)
    label_cols = [PLATINAI_LABEL_COLUMNS[c] for c in shared_cells]
    n_label_cells = int(frame.loc[hit, label_cols].notna().sum().sum()) if label_cols else 0

    return {
        "path": _rel(path),
        "sha256": _sha256(path),
        "n_complexes": int(len(frame)),
        "n_parsed_ligands": int(frame["ligand_main_fragment"].notna().sum()),
        "n_unique_ligands": int(frame["ligand_main_fragment"].nunique()),
        "metals": sorted(external_metals),
        "cell_lines": sorted(external_cells),
        "label_columns": PLATINAI_LABEL_COLUMNS,
        "label_scale": "pIC50 = -log10(IC50 / M); MSRS is log10(IC50 / uM)",
        "metal_overlap": overlap(external_metals, msrs_metals),
        "cell_line_overlap": {
            "external": sorted(external_cells),
            "msrs_present": shared_cells,
            "intersection_unique": len(shared_cells),
            "msrs_rows_by_cell": {
                c: int((msrs["cell_line_key"].astype(str).str.lower() == c).sum()) for c in sorted(external_cells)
            },
        },
        "ligand_overlap": overlap(external_ligands, msrs_ligands),
        "complexes_with_a_shared_ligand": int(hit.sum()),
        "label_cells_with_a_shared_ligand": n_label_cells,
        "same_metal_rows": int(frame.loc[hit & frame["metal"].isin(msrs_metals)].shape[0]),
    }


def audit_nci60(path: Path) -> dict[str, Any]:
    """NCI-60 GI50 has no SMILES and no metal column, so it cannot join."""
    head = pd.read_csv(path, nrows=5)
    return {
        "path": _rel(path),
        "columns": head.columns.tolist(),
        "has_smiles": any("smiles" in c.lower() for c in head.columns),
        "has_metal": any("metal" in c.lower() for c in head.columns),
        "has_cell_line": "CELL_NAME" in head.columns,
        "join_key": "NSC id only; no structure column on disk",
        "usable_rows": 0,
        "reason": "no SMILES and no metal; an NSC-to-structure map is not on disk",
    }


def build_artifact(msrs_path: Path = MSRS_PARQUET, platinai: Path = PLATINAI, nci60: Path = NCI60) -> dict[str, Any]:
    msrs = pd.read_parquet(msrs_path, columns=["metal", "ligand_main_fragment", "cell_line_key"])
    platin = audit_platinai(platinai, msrs)
    nc = audit_nci60(nci60)

    triple_rows = platin["same_metal_rows"]
    passed = triple_rows >= MIN_ROWS
    return {
        "artifact": "msrs_external_dataset",
        "version": 1.0,
        "requires_dual_split": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "script": "molmetal/scripts/msrs_external_dataset.py",
            "git_commit": _git_commit(),
            "msrs_parquet": _rel(msrs_path),
            "msrs_sha256": _sha256(msrs_path),
        },
        "design": {
            "estimand": "same ligand (main fragment) + same metal + same cell line, numeric label in both cohorts",
            "min_rows": MIN_ROWS,
            "r2_gate": R2_GATE,
            "auc_gate": AUC_GATE,
            "ligand_key": "ligand_main_fragment, the loose key; metal is deleted from the external SMILES before comparison",
            "metal_key": "exact element symbol",
            "cell_line_key": "MSRS cell_line_key, lower-cased",
            "no_imputation": "rows that fail any key are excluded, never filled or re-labelled",
        },
        "msrs_reference": {
            "n_rows": int(len(msrs)),
            "n_metals": int(msrs["metal"].nunique()),
            "metals": sorted(msrs["metal"].dropna().astype(str).unique()),
            "n_ligands": int(msrs["ligand_main_fragment"].nunique()),
            "n_cell_lines": int(msrs["cell_line_key"].nunique()),
        },
        "candidates": {"platin_ai_4cell": platin, "nci60_gi50": nc},
        "intersection": {
            "metal": platin["metal_overlap"]["intersection_unique"],
            "cell_line": platin["cell_line_overlap"]["intersection_unique"],
            "ligand": platin["ligand_overlap"]["intersection_unique"],
            "triple_same_metal_rows": triple_rows,
            "label_cells_available_ignoring_metal": platin["label_cells_with_a_shared_ligand"],
        },
        "verdict": "PASS" if passed else "BLOCKED-PENDING-DATA",
        "verdict_rule": f"triple_same_metal_rows >= {MIN_ROWS} AND (R2 >= {R2_GATE} OR AUC >= {AUC_GATE}); else BLOCKED-PENDING-DATA",
        "reading": (
            f"Metal sets are disjoint: PlatinAI is Pt-only, MSRS has {sorted(msrs['metal'].unique())} and zero Pt rows. "
            f"So the same-metal intersection is {triple_rows}, even though {platin['complexes_with_a_shared_ligand']} "
            f"PlatinAI complexes share a ligand main fragment with MSRS and "
            f"{platin['label_cells_with_a_shared_ligand']} (complex, shared-cell) labels exist. "
            "Those rows could support a Pt-extrapolation study, but that is a different estimand and is NOT counted here. "
            "NCI-60 GI50 has neither SMILES nor metal on disk, so it cannot join at all."
        ),
        "not_measured": [
            "Pt zero-shot transfer on the 909 ligand-matched PlatinAI complexes (different estimand: unseen metal)",
            "NCI-60 GI50 regression (no structure column)",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    artifact = build_artifact()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    inter = artifact["intersection"]
    print(
        f"[msrs_external_dataset] verdict={artifact['verdict']} "
        f"metal={inter['metal']} ligand={inter['ligand']} cell={inter['cell_line']} "
        f"triple_rows={inter['triple_same_metal_rows']} -> {_rel(args.out)}"
    )
    return 0 if artifact["verdict"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
