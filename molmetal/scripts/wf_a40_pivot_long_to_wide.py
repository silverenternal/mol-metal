"""Wf-A40 pivot long→wide for PlatinAI 4-cell real-label pool (catalog §6.65c).

=====================================================================
Sub-blocker #3 of catalog §6.65c — wide-format CSV for `load_dataset()`
in ``molmetal/scripts/train_cisplatin_resistance.py:108-122``.

The on-disk real-label pool
(``.data/molmetal/platinai_4cell_real_label_pool_with_smiles.csv``)
ships in long format:

    compound_code, canonical_smiles, cell_line_long, cell_line_short,
    value_mean, value_sd, doi, method, incubationTime, label_source,
    smiles

where ``value_mean`` is per-cell-line IC50 in nM.  The trainer
expects the wide schema:

    SMILES_Ligands, pIC50_mcf7, pIC50_a549, pIC50_a2780cis

per ``PIC50_COLS`` (``train_cisplatin_resistance.py:117-121``).  This
script pivots long → wide and emits the unit conversion:

    pIC50 = -log10(value_mean * 1e-6)

For compounds with multiple rows per cell line (assay replicates)
we average ``value_mean`` first then convert.  Compounds whose cell
line is dropped (``A2780``) are filtered out, matching the trainer's
"drop A2780" policy (``train_cisplatin_resistance.py:13``).

Cell-line → output column mapping
---------------------------------
    MCF-7  -> pIC50_mcf7
    A549   -> pIC50_a549
    A2780  -> (dropped, per trainer spec)
    A2780cis -> pIC50_a2780cis

SMILES selection
----------------
Prefer non-empty ``canonical_smiles``; fall back to ``smiles`` (col
11).  Sentinels ``__no_smiles__`` / empty string yield an empty
``SMILES_Ligands`` (the trainer's RDKit pipeline tolerates empty
SMILES via the ``mol is None`` branch in :func:`load_dataset`).

Verification
------------
::

    python molmetal/scripts/wf_a40_pivot_long_to_wide.py \\
        --in .data/molmetal/platinai_4cell_real_label_pool_with_smiles.csv \\
        --out /tmp/platinai_3cell.csv

Exit 0; output row count = unique compound_code count in source
(<= 2053).
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

LOG = logging.getLogger("wf_a40_pivot")

#: Cell line (long) -> pIC50 output column.
CELL_TO_COL: Dict[str, str] = {
    "MCF-7": "pIC50_mcf7",
    "A549": "pIC50_a549",
    "A2780cis": "pIC50_a2780cis",
}

#: Output schema (matches PIC50_COLS in train_cisplatin_resistance.py).
OUTPUT_HEADER: Tuple[str, ...] = (
    "SMILES_Ligands",
    "pIC50_mcf7",
    "pIC50_a549",
    "pIC50_a2780cis",
)


def _select_smiles(row: Dict[str, str]) -> str:
    """Pick best-available SMILES (skip ``__no_smiles__`` sentinel)."""
    for key in ("canonical_smiles", "smiles"):
        val = (row.get(key) or "").strip()
        if val and val != "__no_smiles__":
            return val
    return ""


def _to_pic50(value_mean_nm: float) -> float:
    """Convert nM IC50 to unitless pIC50.

    ``pIC50 = -log10(IC50_M) = -log10(value_mean * 1e-6)``.
    """
    return -math.log10(value_mean_nm * 1e-6)


def pivot_long_to_wide(in_path: Path) -> List[List[str]]:
    """Read long-format CSV and return wide-format rows."""
    # compound_code -> { smiles, cell_line -> [value_mean_nm, ...] }
    agg: Dict[str, Dict[str, object]] = {}
    with in_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = (row.get("compound_code") or "").strip()
            if not code:
                continue
            cell = (row.get("cell_line_long") or "").strip()
            if cell not in CELL_TO_COL:
                # Drop A2780 (per trainer spec) and unknown cell lines.
                continue
            try:
                v_mean = float(row.get("value_mean", "") or "nan")
            except ValueError:
                continue
            if not math.isfinite(v_mean) or v_mean <= 0:
                continue
            bucket = agg.setdefault(
                code,
                {"smiles": "", "cells": defaultdict(list)},
            )
            if not bucket["smiles"]:
                bucket["smiles"] = _select_smiles(row)
            bucket["cells"][cell].append(v_mean)  # type: ignore[index]

    out_rows: List[List[str]] = []
    for code, bucket in sorted(agg.items()):
        cells_map = bucket["cells"]
        row_out: List[str] = [bucket["smiles"]]  # type: ignore[arg-type]
        for cell_long, col in CELL_TO_COL.items():
            vals = cells_map.get(cell_long, [])
            if not vals:
                row_out.append("")  # missing -> empty (NaN-like)
            else:
                mean_nm = sum(vals) / len(vals)
                row_out.append(f"{_to_pic50(mean_nm):.4f}")
        out_rows.append(row_out)
    return out_rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Pivot PlatinAI 4-cell real-label long CSV to wide format "
            "matching train_cisplatin_resistance.py PIC50_COLS schema."
        )
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        required=True,
        type=Path,
        help="Path to long-format source CSV (platinai_4cell_real_label_pool_with_smiles.csv).",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        required=True,
        type=Path,
        help="Path to wide-format output CSV (default schema: SMILES_Ligands + 3 pIC50 columns).",
    )
    args = parser.parse_args(argv)

    if not args.in_path.exists():
        LOG.error("input CSV not found: %s", args.in_path)
        return 2

    rows = pivot_long_to_wide(args.in_path)
    n_unique = len(rows)

    # Serialise to CSV string (atomic write below).
    out_lines: List[str] = [",".join(OUTPUT_HEADER)]
    for r in rows:
        out_lines.append(",".join(r))
    content = "\n".join(out_lines) + "\n"

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(args.out_path, content)

    LOG.info(
        "pivot: %d unique compounds -> %s (cells: %s)",
        n_unique,
        args.out_path,
        ", ".join(CELL_TO_COL.values()),
    )
    print(f"OK pivoted {n_unique} unique compounds -> {args.out_path}")
    print(f"OUTPUT_HEADER={list(OUTPUT_HEADER)}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    sys.exit(main())
