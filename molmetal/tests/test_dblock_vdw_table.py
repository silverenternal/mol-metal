"""Tests for the aggregated d-block VDW literature-mining table (Wf-143 Stage 2).

================================================================
Wf-143 Stage 2: aggregation of 4 parallel literature-mining agents
(Group A: Pt/Ru; Group B: Ir/Os; Group C: Rh/Pd; Group D: Au/Re).

This file ships the **8 pytest contracts** that pin the behaviour
of the aggregated d-block VDW table at
``molmetal/references/soft_params/dblock_vdw_table.json``:

The 8 tests
------------
Per-metal presence + sanity (one test per metal)
 1. test_Pt_present_and_in_range
 2. test_Ru_present_and_in_range
 3. test_Ir_present_and_in_range
 4. test_Os_present_and_in_range
 5. test_Rh_present_and_in_range
 6. test_Pd_present_and_in_range
 7. test_Au_present_and_in_range
 8. test_Re_present_and_in_range

Each test verifies:
- metal key exists in the table
- r_angstrom is in [1.3, 2.5]
- epsilon_kcal_mol is in [0.01, 0.5]
- source_primary is a non-empty string
- uncertainty_range is a non-empty string

All tests are CPU-only, deterministic, and complete in < 1 s combined.

Run with::

    uv run --no-sync pytest -q molmetal/tests/test_dblock_vdw_table.py --tb=short
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict

import pytest

# --- shared loader --------------------------------------------------------

_TABLE_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "references"
    / "soft_params"
    / "dblock_vdw_table.json"
)

# Sanity ranges — documented in the aggregation meta block.
_R_MIN, _R_MAX = 1.3, 2.5
_EPS_MIN, _EPS_MAX = 0.01, 0.5


def _load_table() -> Dict[str, Any]:
    """Load the aggregated d-block VDW table from disk."""
    assert _TABLE_PATH.is_file(), f"missing aggregated table: {_TABLE_PATH}"
    with _TABLE_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _assert_metal_in_range(table: Dict[str, Any], element: str) -> None:
    """Run the full per-metal contract: presence + ranges + non-empty source."""
    assert element in table, f"{element} missing from dblock_vdw_table.json"
    record = table[element]
    assert isinstance(record, dict), f"{element} record is not a dict"

    # r_angstrom sanity
    assert "r_angstrom" in record, f"{element} missing r_angstrom"
    r = record["r_angstrom"]
    assert isinstance(r, (int, float)), f"{element} r_angstrom not numeric"
    assert _R_MIN <= r <= _R_MAX, (
        f"{element} r_angstrom={r} outside [{_R_MIN}, {_R_MAX}]"
    )

    # epsilon_kcal_mol sanity
    assert "epsilon_kcal_mol" in record, f"{element} missing epsilon_kcal_mol"
    eps = record["epsilon_kcal_mol"]
    assert isinstance(eps, (int, float)), f"{element} epsilon_kcal_mol not numeric"
    assert _EPS_MIN <= eps <= _EPS_MAX, (
        f"{element} epsilon_kcal_mol={eps} outside [{_EPS_MIN}, {_EPS_MAX}]"
    )

    # source_primary non-empty
    assert "source_primary" in record, f"{element} missing source_primary"
    src = record["source_primary"]
    assert isinstance(src, str), f"{element} source_primary not a string"
    assert src.strip(), f"{element} source_primary is empty"

    # uncertainty_range non-empty
    assert "uncertainty_range" in record, f"{element} missing uncertainty_range"
    unc = record["uncertainty_range"]
    assert isinstance(unc, str), f"{element} uncertainty_range not a string"
    assert unc.strip(), f"{element} uncertainty_range is empty"


# --- 8 per-metal tests -----------------------------------------------------


def test_Pt_present_and_in_range() -> None:
    """Pt (Group A — cisplatin-class square planar Pt(II))."""
    table = _load_table()
    _assert_metal_in_range(table, "Pt")
    pt = table["Pt"]
    # Pt-specific documented value (Bondi + UFF Pt4+2)
    assert pt["r_angstrom"] == pytest.approx(1.75)
    assert pt["epsilon_kcal_mol"] == pytest.approx(0.080)


def test_Ru_present_and_in_range() -> None:
    """Ru (Group A — octahedral Ru(II) arene + polypyridyl)."""
    table = _load_table()
    _assert_metal_in_range(table, "Ru")
    ru = table["Ru"]
    assert ru["r_angstrom"] == pytest.approx(2.05)
    assert ru["epsilon_kcal_mol"] == pytest.approx(0.056)


def test_Ir_present_and_in_range() -> None:
    """Ir (Group B — 5d half-sandwich Cp* anticancer)."""
    table = _load_table()
    _assert_metal_in_range(table, "Ir")
    ir = table["Ir"]
    assert ir["r_angstrom"] == pytest.approx(1.866)
    assert ir["epsilon_kcal_mol"] == pytest.approx(0.073)


def test_Os_present_and_in_range() -> None:
    """Os (Group B — 5d octahedral polypyridyl anticancer)."""
    table = _load_table()
    _assert_metal_in_range(table, "Os")
    os_ = table["Os"]
    assert os_["r_angstrom"] == pytest.approx(1.700)
    assert os_["epsilon_kcal_mol"] == pytest.approx(0.037)


def test_Rh_present_and_in_range() -> None:
    """Rh (Group C — 4d Wilkinson-class Rh(III) octahedral)."""
    table = _load_table()
    _assert_metal_in_range(table, "Rh")
    rh = table["Rh"]
    assert rh["r_angstrom"] == pytest.approx(1.73)
    assert rh["epsilon_kcal_mol"] == pytest.approx(0.053)


def test_Pd_present_and_in_range() -> None:
    """Pd (Group C — 4d square planar Pd(II) phosphine)."""
    table = _load_table()
    _assert_metal_in_range(table, "Pd")
    pd = table["Pd"]
    assert pd["r_angstrom"] == pytest.approx(1.63)
    assert pd["epsilon_kcal_mol"] == pytest.approx(0.048)


def test_Au_present_and_in_range() -> None:
    """Au (Group D — Au(I) auranofin + Au(III) satraplatin-analog)."""
    table = _load_table()
    _assert_metal_in_range(table, "Au")
    au = table["Au"]
    assert au["r_angstrom"] == pytest.approx(1.66)
    assert au["epsilon_kcal_mol"] == pytest.approx(0.039)


def test_Re_present_and_in_range() -> None:
    """Re (Group D — 186Re/188Re radiopharmaceutical)."""
    table = _load_table()
    _assert_metal_in_range(table, "Re")
    re_ = table["Re"]
    assert re_["r_angstrom"] == pytest.approx(1.55)
    assert re_["epsilon_kcal_mol"] == pytest.approx(0.066)
