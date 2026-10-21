#!/usr/bin/env python
"""Build the REAL-label PlatinAI 4-cell cohort from the MB-Finder harvest.

Why this script exists
----------------------
`.data/molmetal/platinai_4cell.csv` — the cohort every `m2` / `platinai_3cell`
number rests on — carries **simulated** pIC50 labels
(`molmetal/data/platinai_4cell.meta.json` records the generator parameters;
`docs/DATA_INVENTORY.json` marks it DO-NOT-TRAIN).  Meanwhile the 2026-10-19
MB-Finder harvest sits on disk with real, DOI-backed IC50s and SMILES on every
row.  Nothing joined the two facts, so the paper line kept quoting a simulator.

Unit provenance — the one thing that has to be right
----------------------------------------------------
The MB-Finder API exposes **no unit column** (`get-ceil-line-bio-data` returns
`numeric_value_raw` such as `"1.42 ± 0.81"`).  Guessing would be worse than
refusing: a nM/µM mix-up shifts every pIC50 by 3.

The unit is therefore taken from the paper that MB-Finder curates.  Rusanov
2025 supplementary information (`.data/molmetal/rusanov_2025/chemrxiv-2025-
pp32k/supplementary_information_v2.pdf`) states it on every per-cell-line
figure caption:

    "Boxplots represent log-scaled IC50 values (µM) for individual cell
     lines, with replicates shown as white dots."   — Figures S17-S28

So: IC50 is in µM, and pIC50 = 6 - log10(value_µM).

Filters (all counted in the meta file, none silent)
---------------------------------------------------
1. cell line in {A549, A2780, A2780cis, MCF-7}   — the four PlatinAI cells
2. numeric ``value_mean`` > 0
3. ``smiles`` present and not the ``__no_smiles__`` sentinel
4. SMILES parses under RDKit
5. 3 <= pIC50 <= 10 (IC50 between 0.1 nM and 1 mM) — outside this window the
   value is a unit or transcription error, not a measurement
6. one row per (compound_code, cell_line): median of the replicates

Output
------
``.data/molmetal/platinai_4cell_real_cohort.csv``  wide, one row per compound
``.data/molmetal/platinai_4cell_real_cohort.meta.json``  provenance + counts

Read-only with respect to the source harvest.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

SOURCE = (
    ROOT
    / ".data"
    / "molmetal"
    / "wf_a37_platinai_biodata"
    / "pt"
    / "platinai_pt_all_cells_with_smiles.jsonl"
)
OUT_CSV = ROOT / ".data" / "molmetal" / "platinai_4cell_real_cohort.csv"
OUT_META = ROOT / ".data" / "molmetal" / "platinai_4cell_real_cohort.meta.json"
#: Same rows, in the schema `load_platinai_4cell_split_table` accepts
#: (SMILES_Ligands + pIC50_{cell} + metal).  Missing measurements stay empty:
#: that loader drops NaN per cell, so a sparse cohort is expected input rather
#: than something to impute.
OUT_TRAINER = ROOT / ".data" / "molmetal" / "platinai_4cell_real_trainer.csv"

#: Harvest spelling -> column suffix.  MCF-7 keeps its hyphen in the harvest.
CELLS = {"A549": "a549", "A2780": "a2780", "A2780cis": "a2780cis", "MCF-7": "mcf7"}

#: IC50 in µM.  Outside this window the value cannot be a cell-viability IC50:
#: 1e-4 µM = 0.1 nM (more potent than any Pt complex in this corpus) and
#: 1e3 µM = 1 mM (no cytotoxicity titration reaches it).
PIC50_MIN, PIC50_MAX = 3.0, 10.0
NO_SMILES = "__no_smiles__"

UNIT_EVIDENCE = (
    "Rusanov 2025 supplementary information "
    "(.data/molmetal/rusanov_2025/chemrxiv-2025-pp32k/"
    "supplementary_information_v2.pdf), figure captions S17-S28: 'Boxplots "
    "represent log-scaled IC50 values (uM) for individual cell lines, with "
    "replicates shown as white dots.' The MB-Finder API itself exposes no "
    "unit column, so this is the authoritative statement of the unit."
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build() -> Dict[str, Any]:
    if not SOURCE.is_file():
        raise SystemExit(f"source harvest not found: {SOURCE}")

    counts: Dict[str, int] = defaultdict(int)
    per_pair: Dict[tuple, List[float]] = defaultdict(list)
    smiles_of: Dict[str, str] = {}
    dois_of: Dict[str, set] = defaultdict(set)

    with SOURCE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            counts["rows_total"] += 1
            rec = json.loads(line)
            cell = rec.get("cell_line")
            if cell not in CELLS:
                counts["dropped_other_cell_line"] += 1
                continue
            code = rec.get("compound_code")
            smi = (rec.get("smiles") or "").strip()
            if not smi or smi == NO_SMILES:
                counts["dropped_no_smiles"] += 1
                continue
            try:
                value = float(rec.get("value_mean"))
            except (TypeError, ValueError):
                counts["dropped_non_numeric"] += 1
                continue
            if value <= 0:
                counts["dropped_non_positive"] += 1
                continue
            pic50 = 6.0 - math.log10(value)
            if not (PIC50_MIN <= pic50 <= PIC50_MAX):
                counts["dropped_implausible_pic50"] += 1
                continue
            per_pair[(code, cell)].append(pic50)
            smiles_of[code] = smi
            if rec.get("doi"):
                dois_of[code].add(str(rec["doi"]))

    # Collapse replicates: median per (compound, cell).  Median rather than mean
    # because a single 10x outlier inside a triplicate is a transcription error.
    by_compound: Dict[str, Dict[str, float]] = defaultdict(dict)
    n_meas: Dict[str, int] = defaultdict(int)
    for (code, cell), vals in per_pair.items():
        by_compound[code][CELLS[cell]] = round(statistics.median(vals), 4)
        n_meas[code] += len(vals)

    # Structure gate last: it is the expensive one, and doing it after the
    # numeric filters keeps the "how many failed RDKit" number meaningful.
    rows: List[Dict[str, Any]] = []
    for code, cells in by_compound.items():
        smi = smiles_of[code]
        if Chem.MolFromSmiles(smi) is None:
            counts["dropped_rdkit_invalid"] += 1
            continue
        row = {"compound_code": code, "smiles": smi}
        for suffix in CELLS.values():
            row[f"pIC50_{suffix}"] = cells.get(suffix, "")
        row["n_cells"] = len(cells)
        row["n_measurements"] = n_meas[code]
        row["n_dois"] = len(dois_of[code])
        rows.append(row)

    rows.sort(key=lambda r: (-r["n_cells"], r["compound_code"]))
    return {
        "rows": rows,
        "counts": dict(counts),
        "source": SOURCE,
    }


def write(result: Dict[str, Any]) -> Dict[str, Any]:
    rows = result["rows"]
    if not rows:
        raise SystemExit("no rows survived the filters; refusing to write an empty cohort")

    suffixes = list(CELLS.values())
    header = ["compound_code", "smiles"] + [f"pIC50_{s}" for s in suffixes] + [
        "n_cells",
        "n_measurements",
        "n_dois",
    ]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_CSV.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r[h]) for h in header) + "\n")
    tmp.replace(OUT_CSV)

    per_cell = {
        s: sum(1 for r in rows if r[f"pIC50_{s}"] != "") for s in suffixes
    }
    meta = {
        "dataset": "platinai_4cell_real",
        "status": "REAL_LABELS",
        "label_provenance": {
            "labels": "REAL",
            "label_type": "experimental IC50",
            "source": "MB-Finder (api.mb-finder.org), underlying paper Rusanov 2025 "
            "PlatinAI (DOI 10.26434/chemrxiv-2025-pp32k)",
            "unit": "uM",
            "unit_evidence": UNIT_EVIDENCE,
            "conversion": "pIC50 = 6 - log10(IC50_uM)",
            "source_file": str(result["source"].relative_to(ROOT)),
            "source_sha256": _sha256(result["source"]),
        },
        "filters": {
            "cell_lines": list(CELLS),
            "pic50_window": [PIC50_MIN, PIC50_MAX],
            "pic50_window_rationale": (
                "0.1 nM .. 1 mM. Outside this window a cell-viability IC50 is a "
                "unit or transcription error, not a measurement."
            ),
            "replicate_collapse": "median of all biodata rows for (compound_code, cell_line)",
            "structure_gate": "RDKit MolFromSmiles must parse",
        },
        "counts": result["counts"],
        "n_compounds": len(rows),
        "n_measurements_total": sum(r["n_measurements"] for r in rows),
        "per_cell_compounds": per_cell,
        "n_compounds_with_2plus_cells": sum(1 for r in rows if r["n_cells"] >= 2),
        "n_compounds_with_4_cells": sum(1 for r in rows if r["n_cells"] == 4),
        "framing": (
            "REAL structures + REAL IC50 labels for the four PlatinAI cell lines, "
            "built from the MB-Finder harvest. This is the cohort that can replace "
            "the synthetic .data/molmetal/platinai_4cell.csv for m2 R2 and "
            "platinai_3cell_resistance. It is NOT a reproduction of Rusanov 2025 "
            "Table S4-S5: their 4,078 complexes come from 592 cell lines and are "
            "not fully reachable through the public API."
        ),
        "generated_by": "molmetal/scripts/build_platinai_4cell_real_cohort.py",
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta


def write_trainer_view(rows: List[Dict[str, Any]]) -> int:
    """Project the cohort into the D-MPNN loader's wide schema.

    ``load_platinai_4cell_split_table`` wants one row per SMILES with a
    ``pIC50_<cell>`` column per cell line and calls ``dropna`` per column, so
    writing an empty string where a compound has no measurement for that cell
    is the intended input — it becomes NaN and the row is skipped for that
    head.  Nothing is imputed: filling a missing IC50 would put a number on the
    paper line that no experiment produced.
    """
    suffixes = list(CELLS.values())
    header = ["SMILES_Ligands"] + [f"pIC50_{s}" for s in suffixes] + ["metal"]
    tmp = OUT_TRAINER.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            cells = [str(r[f"pIC50_{s}"]) for s in suffixes]
            f.write(",".join([str(r["smiles"])] + cells + ["Pt"]) + "\n")
    tmp.replace(OUT_TRAINER)
    return len(rows)


def main() -> int:
    result = build()
    meta = write(result)
    n_trainer = write_trainer_view(result["rows"])
    print(f"wrote {OUT_CSV.relative_to(ROOT)}  ({meta['n_compounds']} compounds)")
    print(f"wrote {OUT_META.relative_to(ROOT)}")
    print(f"wrote {OUT_TRAINER.relative_to(ROOT)}  ({n_trainer} rows, trainer schema)")
    print(f"  per cell: {meta['per_cell_compounds']}")
    print(f"  >=2 cells: {meta['n_compounds_with_2plus_cells']}  4 cells: {meta['n_compounds_with_4_cells']}")
    print(f"  dropped: { {k: v for k, v in meta['counts'].items() if k.startswith('dropped')} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
