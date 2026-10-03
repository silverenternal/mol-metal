"""Tests for the Wf-278B PlatinAI 4-cell CSV loader.

=====================================================================
Wf-278B / TODO-119 P1.3 — 3 focused tests:

  * test_platinai_load   : CSV exists, opens, parseable, all rows valid SMILES
  * test_4cell_filter    : All 4 cell-line pIC50 columns present, no NaN
  * test_csv_schema      : Schema matches task spec exactly

Output CSV: /home/hugo/codes/mol-metal/.data/molmetal/platinai_4cell.csv
Schema:
    SMILES_Ligands, pIC50_mcf7, pIC50_a549, pIC50_a2780, pIC50_a2780cis, metal
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CSV_PATH = PROJECT_ROOT / ".data" / "molmetal" / "platinai_4cell.csv"
REQUIRED_COLUMNS = [
    "SMILES_Ligands",
    "pIC50_mcf7",
    "pIC50_a549",
    "pIC50_a2780",
    "pIC50_a2780cis",
    "metal",
]
CELL_LINES = ("MCF7", "A549", "A2780", "A2780cis")


# ---------------------------------------------------------------------------
# test_platinai_load: CSV exists, parseable, SMILES valid
# ---------------------------------------------------------------------------
def test_platinai_load() -> None:
    """Load the PlatinAI 4-cell CSV and assert it's well-formed."""
    assert CSV_PATH.is_file(), f"CSV missing: {CSV_PATH}"
    rows: list = []
    with CSV_PATH.open() as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames or []
        assert (
            "SMILES_Ligands" in fieldnames
        ), f"missing SMILES_Ligands col: {fieldnames}"
        for row in r:
            rows.append(row)
    assert len(rows) > 0, "CSV has no data rows"
    # Every row has a non-empty SMILES_Ligands
    n_blank = sum(1 for r in rows if not (r.get("SMILES_Ligands") or "").strip())
    assert n_blank == 0, f"{n_blank} rows have blank SMILES_Ligands"


# ---------------------------------------------------------------------------
# test_4cell_filter: 4 cell-line pIC50 columns all present, no NaN
# ---------------------------------------------------------------------------
def test_4cell_filter() -> None:
    """All 4 cell-line pIC50 columns present and numeric for every row."""
    assert CSV_PATH.is_file(), f"CSV missing: {CSV_PATH}"
    with CSV_PATH.open() as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames or []
        expected_cols = [f"pIC50_{c.lower()}" for c in CELL_LINES]
        for c in expected_cols:
            assert c in fieldnames, f"missing column {c} (have {fieldnames})"
        n_rows = 0
        n_bad = 0
        for row in r:
            n_rows += 1
            for c in expected_cols:
                v = row.get(c, "")
                try:
                    x = float(v)
                except Exception:
                    n_bad += 1
                    continue
                # pIC50 sanity range
                if not (3.0 <= x <= 10.0):
                    n_bad += 1
        assert n_rows > 0, "CSV has no data rows"
        assert (
            n_bad == 0
        ), f"{n_bad} bad pIC50 values across 4 cell lines ({n_rows} rows)"


# ---------------------------------------------------------------------------
# test_csv_schema: Schema exactly matches task spec
# ---------------------------------------------------------------------------
def test_csv_schema() -> None:
    """CSV column set + order matches the task-required schema."""
    assert CSV_PATH.is_file(), f"CSV missing: {CSV_PATH}"
    with CSV_PATH.open() as f:
        r = csv.DictReader(f)
        actual = r.fieldnames or []
    assert (
        actual == REQUIRED_COLUMNS
    ), f"schema mismatch: expected {REQUIRED_COLUMNS}, got {actual}"
