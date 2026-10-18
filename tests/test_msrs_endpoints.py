"""Tests for the MSRS-0 endpoint builder.

Three properties are load-bearing for every downstream MSRS number:

  I1. Censoring is parsed, not swallowed. ``>100`` must not become an exact
      measurement; the direction must be recoverable.
  I2. Cell-line keys merge spelling variants but never merge distinct lines --
      the p53 +/- pair is the concrete case that a naive strip-all-punctuation
      normaliser gets wrong.
  I3. No source row disappears silently: potency rows + the recorded drop count
      must equal the source row count.

CPU-only, no network, no GPU.

Run: pytest tests/test_msrs_endpoints.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts.msrs_build_endpoints import (  # noqa: E402
    CIS,
    CIS_VALUE,
    DARK,
    DARK_VALUE,
    LIGHT,
    LIGHT_VALUE,
    build_dedup,
    build_long_table,
    combine_censoring,
    normalize_cell_line,
    parse_ic50,
)


# ---------------------------------------------------------------------------
# I1 — censoring
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected_value,expected_censor",
    [
        ("36", 36.0, "none"),
        ("36.0", 36.0, "none"),
        (">100", 100.0, "right"),
        ("> 100", 100.0, "right"),
        (">50.0", 50.0, "right"),
        ("<1", 1.0, "left"),
        (" < 0.5 ", 0.5, "left"),
    ],
)
def test_parse_ic50_censoring(raw, expected_value, expected_censor):
    value, censor = parse_ic50(raw, np.nan)
    assert value == pytest.approx(expected_value)
    assert censor == expected_censor


def test_parse_ic50_blank_is_missing_not_zero():
    value, censor = parse_ic50("", np.nan)
    assert np.isnan(value)
    assert censor == "missing"


def test_parse_ic50_falls_back_to_value_column_only_when_primary_blank():
    assert parse_ic50("", 12.0) == (12.0, "none")
    # A censored primary column must NOT be replaced by the numeric bound.
    assert parse_ic50(">100", 100.0) == (100.0, "right")


@pytest.mark.parametrize(
    "num,den,expected",
    [
        ("none", "none", "none"),
        ("right", "none", "right"),
        ("left", "none", "left"),
        ("none", "right", "left"),
        ("none", "left", "right"),
        ("right", "right", "indeterminate"),
        ("right", "left", "indeterminate"),
    ],
)
def test_combine_censoring_direction(num, den, expected):
    assert combine_censoring(num, den) == expected


# ---------------------------------------------------------------------------
# I2 — cell-line keys
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "a,b",
    [
        ("SK-HEP-1", "SK-Hep1"),
        ("HuT-78", "HuT78"),
        ("PLC-PRF-5", "PLC/PRF/5"),
        ("RT-112", "RT112"),
        ("LS 174T", "Ls174T"),
        ("A549", " a549 "),
    ],
)
def test_cell_line_variants_merge(a, b):
    assert normalize_cell_line(a) == normalize_cell_line(b)


def test_cell_line_p53_status_is_preserved():
    plus = normalize_cell_line("HCT-116 p53+/+")
    minus = normalize_cell_line("HCT-116 p53-/-")
    assert plus != minus
    assert plus.startswith("HCT116P53")
    assert plus.endswith("_P53++")
    assert minus.endswith("_P53--")


# ---------------------------------------------------------------------------
# I3 — table construction on a hand-built frame
# ---------------------------------------------------------------------------
def _tiny_frame() -> pd.DataFrame:
    rows = [
        # ligand A, Ru, MCF7, exact 10 uM dark
        dict(
            SMILES_Ligands="CCO",
            Counterion="Cl",
            Metal="Ru",
            Oxidation_state=2,
            Charge_complex=1,
            Atomic_number=44,
            Valence_e=8,
            Cell_line="MCF-7",
            **{"Time(h)": 24.0, "DOI": "10.1/a", "Year": 2020,
               "Excitation_Wavelength(nm)": 450.0, "Irradiation_Time(minutes)": 30.0,
               "Irradiation_Power(W*m^-2)": 10.0,
               DARK: "10", DARK_VALUE: 10.0,
               LIGHT: "2", LIGHT_VALUE: 2.0,
               CIS: "5", CIS_VALUE: 5.0},
        ),
        # ligand A, Ru, MCF7, replicate at 20 uM -> dedup median log10(10,20)
        dict(
            SMILES_Ligands="CCO",
            Counterion="Cl",
            Metal="Ru",
            Oxidation_state=2,
            Charge_complex=1,
            Atomic_number=44,
            Valence_e=8,
            Cell_line="MCF7",
            **{"Time(h)": 24.0, "DOI": "10.1/a", "Year": 2020,
               "Excitation_Wavelength(nm)": 450.0, "Irradiation_Time(minutes)": 30.0,
               "Irradiation_Power(W*m^-2)": 10.0,
               DARK: "20", DARK_VALUE: 20.0,
               LIGHT: "", LIGHT_VALUE: np.nan,
               CIS: "", CIS_VALUE: np.nan},
        ),
        # ligand B, Os, censored dark, exact light -> PI right-censored
        dict(
            SMILES_Ligands="c1ccccc1",
            Counterion="",
            Metal="Os",
            Oxidation_state=2,
            Charge_complex=0,
            Atomic_number=76,
            Valence_e=6,
            Cell_line="A549",
            **{"Time(h)": 48.0, "DOI": "10.1/b", "Year": 2021,
               "Excitation_Wavelength(nm)": 450.0, "Irradiation_Time(minutes)": 30.0,
               "Irradiation_Power(W*m^-2)": 10.0,
               DARK: ">100", DARK_VALUE: 100.0,
               LIGHT: "1", LIGHT_VALUE: 1.0,
               CIS: ">50", CIS_VALUE: 50.0},
        ),
    ]
    return pd.DataFrame(rows)


def test_long_table_keeps_every_source_row_for_potency():
    df = _tiny_frame()
    long, ledger = build_long_table(df)
    pot = long[long["endpoint"] == "potency"]
    assert len(pot) + ledger["potency"]["dropped_no_dark_ic50"] == len(df)
    assert len(pot) == 3
    assert pot["ligand_canonical"].notna().all()


def test_long_table_marks_censored_dark_and_derives_pi_direction():
    long, _ = build_long_table(_tiny_frame())
    b = long[(long["endpoint"] == "potency") & (long["metal"] == "Os")].iloc[0]
    assert b["censoring"] == "right"
    assert b["value_log10"] == pytest.approx(2.0)

    pi = long[(long["endpoint"] == "phototoxicity") & (long["metal"] == "Os")].iloc[0]
    assert pi["censoring"] == "right"
    assert pi["value_log10"] == pytest.approx(2.0)

    ri = long[(long["endpoint"] == "cisplatin_ratio") & (long["metal"] == "Os")].iloc[0]
    assert ri["censoring"] == "indeterminate"


def test_missing_light_drops_only_the_phototoxicity_endpoint():
    long, _ = build_long_table(_tiny_frame())
    # Row 2 has no light IC50; it must still carry a potency row.
    assert (long["endpoint"] == "potency").sum() == 3
    assert (long["endpoint"] == "phototoxicity").sum() == 2


def test_dedup_takes_median_and_counts_censored_separately():
    long, _ = build_long_table(_tiny_frame())
    dedup = build_dedup(long)
    ru = dedup[(dedup["metal"] == "Ru") & (dedup["endpoint"] == "potency")]
    # MCF-7 and MCF7 normalise to one key, so the two replicates collapse.
    assert len(ru) == 1
    assert ru.iloc[0]["n_measurements"] == 2
    assert ru.iloc[0]["n_exact"] == 2
    assert ru.iloc[0]["n_censored"] == 0
    assert ru.iloc[0]["n_measurements"] == ru.iloc[0]["n_exact"] + ru.iloc[0]["n_censored"]
    assert ru.iloc[0]["value_log10"] == pytest.approx(np.log10(np.sqrt(10 * 20)))
    assert not bool(ru.iloc[0]["all_censored"])


def test_dedup_flags_all_censored_groups():
    long, _ = build_long_table(_tiny_frame())
    dedup = build_dedup(long)
    os_row = dedup[(dedup["metal"] == "Os") & (dedup["endpoint"] == "potency")].iloc[0]
    assert bool(os_row["all_censored"])
    assert np.isnan(os_row["value_log10"])
    assert os_row["n_censored"] == 1
