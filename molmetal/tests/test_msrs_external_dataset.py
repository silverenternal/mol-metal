"""MET-7 tests — the external-cohort intersection must be measured, not asserted."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from molmetal.scripts.msrs_external_dataset import (
    audit_nci60,
    build_artifact,
    overlap,
    strip_metal_and_main_fragment,
)


# ---------------------------------------------------------------------------
# ligand key
# ---------------------------------------------------------------------------


def test_strip_metal_keeps_the_organic_fragment():
    # A phenyl ligand plus two chlorides: only the phenyl ring survives.
    got = strip_metal_and_main_fragment("[Pt+2](<-[Cl-])(<-[Cl-])(<-N)c1ccccc1")
    assert got == "c1ccccc1"


def test_strip_metal_on_a_metal_free_smiles_is_the_canonical_form():
    assert strip_metal_and_main_fragment("c1ccccc1O") == "Oc1ccccc1"


def test_strip_metal_returns_none_on_an_unparseable_string():
    assert strip_metal_and_main_fragment("not-a-smiles") is None


def test_strip_metal_removes_every_metal_in_the_set():
    got = strip_metal_and_main_fragment("[Ru+2].[Ir+3].c1ccccc1")
    assert got == "c1ccccc1"


def test_the_raw_string_comparison_would_have_missed_the_match():
    """A metal-bearing SMILES never equals a metal-stripped one."""
    complex_smiles = "[Pt+2](<-[Cl-])(<-[Cl-])(<-N)c1ccccc1"
    assert complex_smiles != strip_metal_and_main_fragment(complex_smiles)


# ---------------------------------------------------------------------------
# overlap helper
# ---------------------------------------------------------------------------


def test_overlap_counts_and_reports_examples():
    out = overlap({"a", "b", "c"}, {"b", "c", "d"})
    assert out["intersection_unique"] == 2
    assert out["examples"] == ["b", "c"]


def test_overlap_of_disjoint_sets_is_zero():
    assert overlap({"Pt"}, {"Ru", "Ir"})["intersection_unique"] == 0


# ---------------------------------------------------------------------------
# NCI-60 audit
# ---------------------------------------------------------------------------


def test_nci60_without_a_structure_column_is_unusable(tmp_path: Path):
    path = tmp_path / "GI50.csv"
    pd.DataFrame(
        {
            "NSC": [1, 2],
            "CELL_NAME": ["A549", "MCF7"],
            "AVERAGE": [-7.1, -6.2],
        }
    ).to_csv(path, index=False)
    out = audit_nci60(path)
    assert out["has_smiles"] is False
    assert out["has_metal"] is False
    assert out["usable_rows"] == 0


def test_nci60_with_a_smiles_column_is_flagged_as_structured(tmp_path: Path):
    path = tmp_path / "GI50.csv"
    pd.DataFrame({"SMILES": ["CCO"], "CELL_NAME": ["A549"], "AVERAGE": [-6.0]}).to_csv(
        path, index=False
    )
    assert audit_nci60(path)["has_smiles"] is True


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------


def _fixture(tmp_path: Path, n: int, metal: str) -> tuple[Path, Path, Path]:
    msrs = pd.DataFrame(
        {
            "metal": [metal] * n,
            "ligand_main_fragment": [f"lig{i}" for i in range(n)],
            "cell_line_key": ["a549"] * n,
        }
    )
    msrs_path = tmp_path / "msrs.parquet"
    msrs.to_parquet(msrs_path)

    plat = pd.DataFrame(
        {
            "SMILES_Ligands": ["[Pt+2](<-[Cl-])(<-[Cl-])(<-N)c1ccccc1"] * n,
            "metal": ["Pt"] * n,
            "pIC50_mcf7": [5.0] * n,
            "pIC50_a549": [5.0] * n,
            "pIC50_a2780": [5.0] * n,
            "pIC50_a2780cis": [5.0] * n,
        }
    )
    plat_path = tmp_path / "platinai.csv"
    plat.to_csv(plat_path, index=False)

    nci = tmp_path / "GI50.csv"
    pd.DataFrame({"NSC": [1], "CELL_NAME": ["A549"], "AVERAGE": [-7.0]}).to_csv(nci, index=False)
    return msrs_path, plat_path, nci


def test_a_disjoint_metal_set_blocks_regardless_of_ligand_matches(tmp_path: Path):
    msrs, plat, nci = _fixture(tmp_path, n=250, metal="Ru")
    art = build_artifact(msrs, plat, nci)
    assert art["intersection"]["metal"] == 0
    assert art["intersection"]["triple_same_metal_rows"] == 0
    assert art["verdict"] == "BLOCKED-PENDING-DATA"


def test_a_matching_metal_with_enough_rows_passes_the_row_gate(tmp_path: Path):
    # The ligand key still has to match; here both sides are "lig{i}" only if
    # the external SMILES strips to that string, so use a metal-only fixture.
    msrs = pd.DataFrame(
        {"metal": ["Ru"] * 250, "ligand_main_fragment": ["c1ccccc1"] * 250, "cell_line_key": ["a549"] * 250}
    )
    msrs_path = tmp_path / "msrs.parquet"
    msrs.to_parquet(msrs_path)
    plat = pd.DataFrame(
        {
            "SMILES_Ligands": ["[Ru+2](<-[Cl-])(<-[Cl-])(<-N)c1ccccc1"] * 250,
            "metal": ["Ru"] * 250,
            "pIC50_mcf7": [5.0] * 250,
            "pIC50_a549": [5.0] * 250,
            "pIC50_a2780": [5.0] * 250,
            "pIC50_a2780cis": [5.0] * 250,
        }
    )
    plat_path = tmp_path / "platinai.csv"
    plat.to_csv(plat_path, index=False)
    nci = tmp_path / "GI50.csv"
    pd.DataFrame({"NSC": [1], "CELL_NAME": ["A549"], "AVERAGE": [-7.0]}).to_csv(nci, index=False)

    art = build_artifact(msrs_path, plat_path, nci)
    assert art["intersection"]["metal"] == 1
    assert art["intersection"]["triple_same_metal_rows"] == 250
    assert art["verdict"] == "PASS"


def test_the_shipped_artifact_is_blocked_and_traceable():
    path = Path("metrics/by_metric/msrs_external_dataset.json")
    art = json.loads(path.read_text())
    assert art["verdict"] == "BLOCKED-PENDING-DATA"
    assert art["intersection"]["triple_same_metal_rows"] == 0
    assert art["intersection"]["ligand"] > 0  # the metal is what blocks it
    assert art["provenance"]["msrs_sha256"]
