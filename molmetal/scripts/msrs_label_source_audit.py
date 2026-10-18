#!/usr/bin/env python3
"""Can any unused label source add ligands to the MSRS surface?

The learning curve (AUD-M4) said the binding constraint is **ligand count**, not
descriptor richness: R2 keeps climbing at 100% of the data (0.0081 -> 0.1014 ->
0.2136 -> 0.3081 at 10/25/50/100%).  So the next question is not "which feature
block" but "which rows are sitting on disk unused".

This script answers it without training anything:

1. Every endpoint in the long table vs ``potency``: how many ligands, distinct
   keys, and ``cell x doi`` cells does it add?
2. Is ``cisplatin_ratio`` an independent measurement, or an affine transform of
   ``potency``?  If ``potency - RI`` is constant within ``(cell_line, doi)``,
   then RI is ``potency`` minus the cisplatin IC50 of that assay, and training
   on it adds exactly one number per assay -- not one ligand.
3. Which external files under ``molmetal/data/`` hold labelled complexes whose
   ligands never appear in MSRS?

The output is a decision table, not a model.  A source that adds zero ligands
cannot move a learning curve whose x-axis is ligand count, no matter how many
rows it contributes.

Usage::

    .venv/bin/python molmetal/scripts/msrs_label_source_audit.py
    .venv/bin/python molmetal/scripts/msrs_label_source_audit.py --out /tmp/a.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

BASE_ENDPOINT = "potency"
#: The loose ligand key.  Grouping CV uses it, so it is the unit the learning
#: curve counts; a source that adds rows but no new values here adds no groups.
LIGAND_KEY = "ligand_main_fragment"
#: A matched cohort in the decision layer.  A source that only re-measures
#: existing cohorts cannot teach the surface a new comparison.
COHORT_KEYS = ["ligand_main_fragment", "cell_line_key", "metal", "doi", "time_h"]
#: ``potency - RI`` is constant within an assay if RI is a normalised potency.
OFFSET_TOL = 1e-6

EXTERNAL_SOURCES: Dict[str, Dict[str, str]] = {
    "platinai_5k": {
        "path": "molmetal/data/platinai_5k.csv",
        "smiles": "smiles",
        "metal": "metal",
        "labels": "ic50_nm,selectivity",
    },
    "platinai_4cell": {
        "path": "molmetal/data/platinai_4cell.csv",
        "smiles": "smiles",
        "metal": "",
        "labels": "ic50_mcf7_nm,ic50_a549_nm,ic50_a2780_nm,ic50_a2780cis_nm",
    },
    "nci60_precious_9cell": {
        "path": "molmetal/data/nci60_precious_9cell.csv",
        "smiles": "smiles",
        "metal": "",
        "labels": "pIC50",
    },
    "metallo_drugs_500_train": {
        "path": "molmetal/data/metallo_drugs_500_train.csv",
        "smiles": "smiles",
        "metal": "",
        "labels": "",
    },
}


def canonical(smiles: object) -> Optional[str]:
    """RDKit canonical SMILES, or ``None`` when the string does not parse."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(mol) if mol is not None else None


def exact_rows(long: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    """Rows with a numeric target and an uncensored observation."""
    return long[
        (long["endpoint"] == endpoint)
        & long["value_log10"].notna()
        & (long["censoring"].astype(str) == "none")
        & long[LIGAND_KEY].notna()
    ]


def endpoint_inventory(long: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    """Rows / ligands / distinct cohorts / censoring share per endpoint."""
    out: Dict[str, Dict[str, object]] = {}
    for endpoint, block in long.groupby("endpoint"):
        clean = block[block[LIGAND_KEY].notna()]
        out[str(endpoint)] = {
            "n_rows": int(len(block)),
            "n_rows_with_target": int(block["value_log10"].notna().sum()),
            "n_exact_rows": int(len(exact_rows(long, endpoint))),
            "n_ligands": int(clean[LIGAND_KEY].nunique()),
            "n_distinct_cohorts": int(clean[COHORT_KEYS].drop_duplicates().shape[0]),
            "censored_share": round(float((clean["censoring"].astype(str) != "none").mean()), 4),
        }
    return out


def key_overlap(base: pd.DataFrame, other: pd.DataFrame) -> Dict[str, object]:
    """How much of ``other`` is not already in ``base``.

    Counted at the ligand key, at the full cohort key, and at ``cell x doi``.
    The last one matters because a source can only teach a new assay context
    through a new cell/paper pair.
    """
    base_lig = set(base[LIGAND_KEY].dropna())
    other_lig = set(other[LIGAND_KEY].dropna())
    base_keys = set(map(tuple, base[COHORT_KEYS].drop_duplicates().to_numpy()))
    other_keys = set(map(tuple, other[COHORT_KEYS].drop_duplicates().to_numpy()))
    base_cd = set(map(tuple, base[["cell_line_key", "doi"]].drop_duplicates().to_numpy()))
    other_cd = set(map(tuple, other[["cell_line_key", "doi"]].drop_duplicates().to_numpy()))
    return {
        "n_ligands": len(other_lig),
        "n_new_ligands": len(other_lig - base_lig),
        "n_cohorts": len(other_keys),
        "n_new_cohorts": len(other_keys - base_keys),
        "n_cell_doi": len(other_cd),
        "n_new_cell_doi": len(other_cd - base_cd),
    }


def cisplatin_identity(potency: pd.DataFrame, ratio: pd.DataFrame) -> Dict[str, object]:
    """Test ``potency - RI == log10(IC50 cisplatin | cell, doi)``.

    If the difference is constant within ``(cell_line, doi)`` then RI carries no
    ligand information the potency rows do not already carry: it is the same
    measurement divided by the assay's cisplatin reference.  The share of
    exactly-constant groups is the evidence; the exceptions are reported rather
    than smoothed away.
    """
    cols = COHORT_KEYS + ["value_log10"]
    p = potency[cols].drop_duplicates(COHORT_KEYS)
    r = ratio[cols].drop_duplicates(COHORT_KEYS)
    merged = p.merge(r, on=COHORT_KEYS, suffixes=("_pot", "_ri"))
    if merged.empty:
        return {"n_matched_cohorts": 0, "share_exact_constant": None}

    merged = merged.assign(offset=merged["value_log10_pot"] - merged["value_log10_ri"])
    per_group = merged.groupby(["cell_line_key", "doi"])["offset"].agg(["mean", "std", "size"])
    # A single-row group has no within-group variation to test; it is counted
    # separately so the share is not inflated by unidentifiable groups.
    multi = per_group[per_group["size"] > 1]
    exact = multi["std"].fillna(0.0) < OFFSET_TOL
    return {
        "n_matched_cohorts": int(len(merged)),
        "n_groups": int(len(per_group)),
        "n_groups_multi_row": int(len(multi)),
        "share_exact_constant": round(float(exact.mean()), 4) if len(multi) else None,
        "median_within_group_sd": round(float(multi["std"].median()), 6) if len(multi) else None,
        "p90_within_group_sd": round(float(multi["std"].quantile(0.9)), 6) if len(multi) else None,
        "offset_log10_range": [
            round(float(per_group["mean"].min()), 3),
            round(float(per_group["mean"].max()), 3),
        ],
        "offset_log10_sd_across_groups": round(float(per_group["mean"].std()), 3),
    }


def external_source(
    path: Path, spec: Dict[str, str], msrs_canonical: set, msrs_fragments: set
) -> Dict[str, object]:
    """Ligand novelty and label availability of one external CSV.

    Novelty is measured at **both** representation levels, because comparing an
    external file's full-complex SMILES against MSRS's main-fragment set makes
    every row look new.  The strict comparison decides; the loose one only says
    whether a new complex reuses a known ligand.
    """
    if not path.exists():
        return {"present": False}
    frame = pd.read_csv(path, low_memory=False)
    col = spec["smiles"]
    if col not in frame.columns:
        return {"present": True, "error": f"missing column {col}"}

    from molmetal.scripts.msrs_build_endpoints import main_fragment

    smiles = frame[col].dropna().map(canonical).dropna()
    ligands = set(smiles)
    fragments = set(smiles.map(main_fragment).dropna())
    label_cols = [c for c in spec["labels"].split(",") if c and c in frame.columns]
    labels = {
        c: {
            "non_null": int(frame[c].notna().sum()),
            "median": round(float(frame[c].median()), 3) if frame[c].notna().any() else None,
        }
        for c in label_cols
    }
    metals: Dict[str, int] = {}
    if spec["metal"] and spec["metal"] in frame.columns:
        metals = {str(k): int(v) for k, v in frame[spec["metal"]].value_counts().items()}
    return {
        "present": True,
        "n_rows": int(len(frame)),
        "n_unique_ligands": len(ligands),
        "n_new_ligands_vs_msrs_canonical": len(ligands - msrs_canonical),
        "n_new_ligands_vs_msrs_fragment": len(fragments - msrs_fragments),
        "n_reused_ligands_fragment": len(fragments & msrs_fragments),
        "labels": labels,
        "metals": metals,
        "has_any_label": bool(any(v["non_null"] for v in labels.values())),
    }


def build_audit(long: pd.DataFrame, external: Dict[str, Dict[str, str]], root: Path) -> Dict[str, object]:
    """Assemble the decision table."""
    inventory = endpoint_inventory(long)
    base = exact_rows(long, BASE_ENDPOINT)
    msrs_canonical = set(long["ligand_canonical"].dropna())
    msrs_fragments = set(long[LIGAND_KEY].dropna())

    overlaps: Dict[str, object] = {}
    for endpoint in inventory:
        if endpoint == BASE_ENDPOINT:
            continue
        overlaps[endpoint] = key_overlap(base, exact_rows(long, endpoint))

    ratio = exact_rows(long, "cisplatin_ratio")
    identity = cisplatin_identity(base, ratio) if len(ratio) else {"n_matched_cohorts": 0}

    sources = {name: external_source(root / spec["path"], spec, msrs_canonical, msrs_fragments)
               for name, spec in external.items()}

    # A source is only usable if it carries labels *and* brings ligands the
    # surface has never seen at the strict level.  Rows for known complexes are
    # a re-measurement, not new information about ligand chemistry.
    usable = {
        name: info["n_new_ligands_vs_msrs_canonical"]
        for name, info in sources.items()
        if info.get("present") and info.get("has_any_label")
    }
    return {
        "base_endpoint": BASE_ENDPOINT,
        "n_base_exact_rows": int(len(base)),
        "n_base_ligands": int(base[LIGAND_KEY].nunique()),
        "endpoints": inventory,
        "endpoint_overlap_vs_base": overlaps,
        "cisplatin_ratio_identity": identity,
        "external_sources": sources,
        "verdict": {
            "aux_endpoints_add_ligands": any(v["n_new_ligands"] > 0 for v in overlaps.values()),
            "aux_endpoints_add_cohorts": any(v["n_new_cohorts"] > 0 for v in overlaps.values()),
            "cisplatin_ratio_is_a_transform": bool(
                identity.get("share_exact_constant") is not None
                and identity["share_exact_constant"] >= 0.5
            ),
            "external_sources_with_labels_and_new_ligands": usable,
            "max_new_ligands_available": int(max(usable.values())) if usable else 0,
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--long", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_label_source_audit.json",
    )
    args = parser.parse_args(argv)

    if not args.long.exists():
        print(f"FAIL: missing {args.long}")
        return 1

    long = pd.read_parquet(args.long)
    out = {
        "artifact": "msrs_label_source_audit",
        "version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "long_parquet": str(args.long),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        },
        **build_audit(long, EXTERNAL_SOURCES, PROJECT_ROOT),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(args.out, json.dumps(out, indent=2, ensure_ascii=False) + "\n")

    v = out["verdict"]
    print(
        f"[label-audit] base rows={out['n_base_exact_rows']} ligands={out['n_base_ligands']}\n"
        f"  aux endpoints add ligands={v['aux_endpoints_add_ligands']} "
        f"cohorts={v['aux_endpoints_add_cohorts']}\n"
        f"  RI is a potency transform={v['cisplatin_ratio_is_a_transform']} "
        f"(share exact constant={out['cisplatin_ratio_identity'].get('share_exact_constant')})\n"
        f"  labelled external sources with new ligands: {v['external_sources_with_labels_and_new_ligands']}\n"
        f"wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
