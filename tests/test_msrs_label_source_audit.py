"""Tests for the label-source audit (AUD-M3 precondition)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from molmetal.scripts.msrs_label_source_audit import (  # noqa: E402
    COHORT_KEYS,
    build_audit,
    cisplatin_identity,
    endpoint_inventory,
    exact_rows,
    external_source,
    key_overlap,
    main,
)


def _long(rows):
    """Minimal long table: one row per (endpoint, ligand, cell, metal, doi, time)."""
    return pd.DataFrame(
        [
            {
                "endpoint": ep,
                "value_log10": y,
                "censoring": censor,
                "ligand_main_fragment": lig,
                "ligand_canonical": f"{lig}-full",
                "cell_line_key": cell,
                "metal": metal,
                "doi": doi,
                "time_h": 48,
            }
            for ep, lig, cell, metal, doi, y, censor in rows
        ]
    )


def test_exact_rows_drops_censored_and_missing_target():
    long = _long(
        [
            ("potency", "L1", "A", "Ru", "d1", 0.5, "none"),
            ("potency", "L2", "A", "Ru", "d1", 2.0, "right"),  # censored
            ("potency", "L3", "A", "Ru", "d1", float("nan"), "none"),
        ]
    )
    out = exact_rows(long, "potency")
    assert out["ligand_main_fragment"].tolist() == ["L1"]


def test_endpoint_inventory_counts_rows_ligands_and_censoring():
    long = _long(
        [
            ("potency", "L1", "A", "Ru", "d1", 0.5, "none"),
            ("potency", "L1", "B", "Ru", "d1", 1.0, "right"),
            ("potency", "L2", "A", "Ru", "d1", 1.5, "none"),
        ]
    )
    inv = endpoint_inventory(long)["potency"]
    assert inv["n_rows"] == 3
    assert inv["n_exact_rows"] == 2
    assert inv["n_ligands"] == 2
    assert inv["n_distinct_cohorts"] == 3
    assert inv["censored_share"] == pytest.approx(1 / 3, abs=1e-4)


def test_key_overlap_separates_new_ligands_from_new_cohorts():
    base = _long([("potency", "L1", "A", "Ru", "d1", 0.5, "none")])
    # Same ligand, new cell -> new cohort, no new ligand.
    other = _long([("potency", "L1", "B", "Ru", "d1", 0.7, "none")])
    out = key_overlap(base, other)
    assert out["n_ligands"] == 1
    assert out["n_new_ligands"] == 0
    assert out["n_new_cohorts"] == 1
    # A new cell line inside the same paper is a new cell x doi context.
    assert out["n_new_cell_doi"] == 1

    # Same cell and paper, new ligand -> new cohort, but no new assay context.
    other_same_cell = _long([("potency", "L7", "A", "Ru", "d1", 0.8, "none")])
    out_same = key_overlap(base, other_same_cell)
    assert out_same["n_new_ligands"] == 1
    assert out_same["n_new_cohorts"] == 1
    assert out_same["n_new_cell_doi"] == 0

    # New ligand -> new at every level.
    other2 = _long([("potency", "L9", "C", "Ru", "d2", 0.9, "none")])
    out2 = key_overlap(base, other2)
    assert out2["n_new_ligands"] == 1
    assert out2["n_new_cohorts"] == 1
    assert out2["n_new_cell_doi"] == 1


def test_cisplatin_identity_detects_exact_constant_offset():
    """RI = potency - c per (cell, doi): the difference has zero variance."""
    rows = []
    for lig, y in [("L1", 0.5), ("L2", 1.2), ("L3", -0.3)]:
        rows.append(("potency", lig, "A", "Ru", "d1", y, "none"))
        rows.append(("cisplatin_ratio", lig, "A", "Ru", "d1", y - 1.0, "none"))
    long = _long(rows)
    out = cisplatin_identity(exact_rows(long, "potency"), exact_rows(long, "cisplatin_ratio"))
    assert out["n_matched_cohorts"] == 3
    assert out["share_exact_constant"] == 1.0
    assert out["median_within_group_sd"] == 0.0
    # potency - RI = y - (y - 1) = +1: the cisplatin reference sits 1 log10 below.
    assert out["offset_log10_range"] == [1.0, 1.0]


def test_cisplatin_identity_flags_a_non_constant_offset():
    """If the offset varies inside an assay, RI is not a pure transform."""
    rows = [
        ("potency", "L1", "A", "Ru", "d1", 0.5, "none"),
        ("potency", "L2", "A", "Ru", "d1", 1.2, "none"),
        ("cisplatin_ratio", "L1", "A", "Ru", "d1", 0.5 - 1.0, "none"),
        ("cisplatin_ratio", "L2", "A", "Ru", "d1", 1.2 - 2.0, "none"),
    ]
    long = _long(rows)
    out = cisplatin_identity(exact_rows(long, "potency"), exact_rows(long, "cisplatin_ratio"))
    assert out["share_exact_constant"] == 0.0
    assert out["median_within_group_sd"] == pytest.approx(0.7071, abs=1e-3)


def test_cisplatin_identity_handles_no_overlap():
    long = _long([("potency", "L1", "A", "Ru", "d1", 0.5, "none")])
    out = cisplatin_identity(exact_rows(long, "potency"), exact_rows(long, "cisplatin_ratio"))
    assert out == {"n_matched_cohorts": 0, "share_exact_constant": None}


def test_external_source_compares_at_the_canonical_level(tmp_path):
    """A file whose complexes are already in MSRS must report zero new ligands."""
    smiles = "c1ccccc1"
    from rdkit import Chem

    canon = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    csv = tmp_path / "ext.csv"
    pd.DataFrame({"smiles": [smiles, smiles], "ic50": [1.0, 2.0]}).to_csv(csv, index=False)
    spec = {"path": str(csv), "smiles": "smiles", "metal": "", "labels": "ic50"}

    known = external_source(csv, spec, {canon}, {canon})
    assert known["n_unique_ligands"] == 1
    assert known["n_new_ligands_vs_msrs_canonical"] == 0
    assert known["has_any_label"] is True
    assert known["labels"]["ic50"]["non_null"] == 2

    novel = external_source(csv, spec, set(), set())
    assert novel["n_new_ligands_vs_msrs_canonical"] == 1


def test_external_source_missing_file_is_not_an_error(tmp_path):
    spec = {"path": "nope.csv", "smiles": "smiles", "metal": "", "labels": ""}
    out = external_source(tmp_path / "nope.csv", spec, set(), set())
    assert out == {"present": False}


def test_build_audit_verdict_flags_transform_and_novel_source(tmp_path):
    rows = [
        ("potency", "L1", "A", "Ru", "d1", 0.5, "none"),
        ("potency", "L2", "A", "Ru", "d1", 1.2, "none"),
        ("cisplatin_ratio", "L1", "A", "Ru", "d1", -0.5, "none"),
        ("cisplatin_ratio", "L2", "A", "Ru", "d1", 0.2, "none"),
    ]
    long = _long(rows)
    csv = tmp_path / "ext.csv"
    pd.DataFrame({"smiles": ["CCN"], "ic50": [3.0]}).to_csv(csv, index=False)
    spec = {"platinai": {"path": str(csv), "smiles": "smiles", "metal": "", "labels": "ic50"}}

    out = build_audit(long, spec, tmp_path)
    assert out["verdict"]["aux_endpoints_add_ligands"] is False
    assert out["verdict"]["cisplatin_ratio_is_a_transform"] is True
    assert out["verdict"]["max_new_ligands_available"] == 1
    assert out["n_base_ligands"] == 2


def test_main_writes_json(tmp_path):
    long = _long(
        [
            ("potency", "L1", "A", "Ru", "d1", 0.5, "none"),
            ("potency", "L2", "A", "Ru", "d1", 1.2, "none"),
        ]
    )
    parquet = tmp_path / "m.parquet"
    long.to_parquet(parquet)
    out = tmp_path / "audit.json"
    assert main(["--long", str(parquet), "--out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["artifact"] == "msrs_label_source_audit"
    assert payload["n_base_ligands"] == 2


def test_cohort_keys_are_the_ones_the_decision_layer_uses():
    assert COHORT_KEYS == ["ligand_main_fragment", "cell_line_key", "metal", "doi", "time_h"]
