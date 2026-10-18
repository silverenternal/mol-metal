"""Tests for the coordination block (FIX-L3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from molmetal.features.coordination import (
    COUNT_COLUMNS,
    CoordinationProfile,
    coordination_block,
    coordination_profile,
    estimated_coordination_number,
    geometry_vector,
)

BIPY = "c1ccncc1"
TERPY = "c1ccncc1.c1ccncc1.c1ccncc1"


def test_pyridine_has_one_donor():
    p = coordination_profile(BIPY)
    assert p.n_N == 1
    assert p.n_donor == 1
    assert p.has_donor


def test_three_pyridines_are_three_components_one_donor_each():
    p = coordination_profile(TERPY)
    assert p.n_donor == 3
    assert p.max_component_donors == 1
    assert p.n_donor_components == 3


def test_bidentate_ligand_reports_denticity_proxy():
    p = coordination_profile("c1ccc(nc1)-c1ccccn1")  # 2,2'-bipyridine
    assert p.n_N == 2
    assert p.max_component_donors == 2


def test_anionic_oxygen_is_counted_and_flagged():
    p = coordination_profile("CC(=O)[O-]")
    assert p.n_O == 2
    assert p.n_anionic == 1


def test_counterions_alone_have_no_donors():
    p = coordination_profile("[Cl-].[Cl-]")
    assert p.n_donor == 0
    assert not p.has_donor


def test_unparseable_smiles_yields_zero_profile():
    p = coordination_profile("not_a_smiles")
    assert p == CoordinationProfile()


def test_none_is_handled():
    assert coordination_profile(None) == CoordinationProfile()


def test_estimated_cn_is_capped_by_metal_capacity():
    p = CoordinationProfile(n_donor=10)
    assert estimated_coordination_number(p, "Ir", 1) == 4  # capacity 4
    assert estimated_coordination_number(p, "Ru", 2) == 6


def test_no_donors_gives_zero_cn():
    assert estimated_coordination_number(CoordinationProfile(), "Ru", 2) == 0


def test_geometry_vector_is_one_hot_for_known_case():
    vec, cls = geometry_vector(6, "Ru", 2)
    assert cls == "octahedral"
    assert vec.sum() == pytest.approx(1.0)


def test_geometry_vector_is_zero_for_unknown_cn():
    vec, cls = geometry_vector(0, "Ru", 2)
    assert cls == "none"
    assert vec.sum() == 0.0


def test_block_shapes_and_coverage_keys():
    frame = pd.DataFrame(
        {
            "ligand_canonical": [TERPY, BIPY, "[Cl-]"],
            "metal": ["Ru", "Ru", "Ru"],
            "oxidation_state": [2, 3, 2],
        }
    )
    X, names, coverage = coordination_block(frame, include_geometry=True)
    assert X.shape[0] == 3
    assert names[: len(COUNT_COLUMNS)] == list(COUNT_COLUMNS)
    assert coverage["has_donor"] == pytest.approx(2 / 3, abs=1e-4)
    assert 0.0 <= coverage["has_geometry"] <= 1.0
    assert np.asarray(X[:, COUNT_COLUMNS.index("coord_n_N")].todense()).ravel().tolist() == [
        3.0,
        1.0,
        0.0,
    ]
