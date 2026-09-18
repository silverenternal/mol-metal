"""Tests for molmetal_lam.training.pic50_censored_loader.

WF-Extra-1: censor-aware D-MPNN training data loader.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

_PKG_PARENT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from molmetal_lam.training.pic50_censored_loader import (  # noqa: E402
    CensoredPIC50Dataset,
    RAW_IC50_COL,
    SMILES_COL,
    VALUE_IC50_COL,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_frame() -> pd.DataFrame:
    """Minimal DataFrame mirroring the audit columns."""
    return pd.DataFrame(
        [
            # Censored right-bound: IC50 > 10 uM
            {
                SMILES_COL: "CCO",
                RAW_IC50_COL: ">10",
                VALUE_IC50_COL: 10.0,
                "Metal": "Ru",
                "Cell_line": "HeLa",
                "Time(h)": 48.0,
                "DOI": "10.1234/test1",
                "Counterion": "Cl",
                "Oxidation_state": 2,
                "Charge_complex": 2,
            },
            # Censored left-bound: IC50 < 0.1 uM
            {
                SMILES_COL: "c1ccccc1",
                RAW_IC50_COL: "<0.1",
                VALUE_IC50_COL: 0.1,
                "Metal": "Ru",
                "Cell_line": "HeLa",
                "Time(h)": 48.0,
                "DOI": "10.1234/test2",
                "Counterion": "Cl",
                "Oxidation_state": 2,
                "Charge_complex": 2,
            },
            # Exact measurement (no censor symbol)
            {
                SMILES_COL: "CC(=O)O",
                RAW_IC50_COL: "5.3",
                VALUE_IC50_COL: 5.3,
                "Metal": "Pt",
                "Cell_line": "HeLa",
                "Time(h)": 48.0,
                "DOI": "10.1234/test3",
                "Counterion": "",
                "Oxidation_state": 2,
                "Charge_complex": 2,
            },
            # Invalid SMILES — must be dropped with a warning
            {
                SMILES_COL: "this-is-not-a-smiles@@",
                RAW_IC50_COL: ">5",
                VALUE_IC50_COL: 5.0,
                "Metal": "Ru",
                "Cell_line": "HeLa",
                "Time(h)": 48.0,
                "DOI": "10.1234/test4",
                "Counterion": "Cl",
                "Oxidation_state": 2,
                "Charge_complex": 2,
            },
        ]
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_censored_loader_returns_bounds(sample_frame):
    """Row with IC50 > 10 uM must yield (10.0, True, 10.0)."""
    ds = CensoredPIC50Dataset(sample_frame)
    # Find the row whose SMILES is ethanol (the >10 uM case)
    target_idx = None
    for i in range(len(ds)):
        smi, _t, _c, _b = ds[i]
        if smi == "CCO":
            target_idx = i
            break
    assert target_idx is not None, "ethanol row missing from dataset"

    smi, target, is_censored, bound_value = ds[target_idx]
    assert smi == "CCO"
    assert target == pytest.approx(10.0)
    assert is_censored is True
    assert bound_value == pytest.approx(10.0)


def test_loader_skips_unparseable_smiles(sample_frame):
    """Invalid SMILES rows are dropped with a warning, not raised."""
    with pytest.warns(UserWarning, match="unparseable SMILES"):
        ds = CensoredPIC50Dataset(sample_frame)
    # We seeded 4 rows: one invalid → 3 remaining
    assert len(ds) == 3
    # And the invalid SMILES must not appear in any row
    for i in range(len(ds)):
        smi, _t, _c, _b = ds[i]
        assert "this-is-not" not in smi


def test_loader_preserves_formulation_meta(sample_frame):
    """Formulation meta must round-trip into the dataset row."""
    ds = CensoredPIC50Dataset(sample_frame)
    # Pick the exact-value Pt row (acetic acid) and inspect its meta
    target_idx = None
    for i in range(len(ds)):
        smi, _t, _c, _b = ds[i]
        if smi == "CC(=O)O":
            target_idx = i
            break
    assert target_idx is not None
    meta = ds.formulation_meta(target_idx)
    assert meta["Metal"] == "Pt"
    assert meta["Cell_line"] == "HeLa"
    assert meta["Time(h)"] == 48.0
    assert meta["DOI"] == "10.1234/test3"
    assert meta["Counterion"] == ""
    assert meta["Oxidation_state"] == 2
    assert meta["Charge_complex"] == 2
    # And the formulation_id must be present + stable
    row = ds.get_row(target_idx)
    assert row.formulation_id
    assert "Pt" in row.formulation_id
    assert "HeLa" in row.formulation_id


# ---------------------------------------------------------------------------
# Lightweight extra coverage (so the loader is not just a 3-test stub)
# ---------------------------------------------------------------------------
def test_exact_row_is_not_censored(sample_frame):
    ds = CensoredPIC50Dataset(sample_frame)
    for i in range(len(ds)):
        smi, _t, is_censored, bound = ds[i]
        if smi == "CC(=O)O":
            assert is_censored is False
            assert bound == pytest.approx(5.3)
            return
    pytest.fail("acetic-acid row missing")


def test_left_censored_row_keeps_bound(sample_frame):
    ds = CensoredPIC50Dataset(sample_frame)
    for i in range(len(ds)):
        smi, _t, is_censored, bound = ds[i]
        if smi == "c1ccccc1":
            assert is_censored is True
            assert bound == pytest.approx(0.1)
            return
    pytest.fail("benzene row missing")


def test_n_censored_total_excludes_dropped(sample_frame):
    ds = CensoredPIC50Dataset(sample_frame)
    # Two of the three surviving rows are censored; one invalid SMILES was dropped.
    assert ds.n_censored_total == 2
