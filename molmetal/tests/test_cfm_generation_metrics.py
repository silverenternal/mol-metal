"""MET-9 tests — novelty and diversity arithmetic on real fingerprints."""

from __future__ import annotations

import pytest

from molmetal.scripts.cfm_generation_metrics import (
    canonical,
    intdiv1,
    max_tanimoto_to_reference,
    metal_profile,
    morgan,
)


def test_canonical_rejects_unparseable_smiles():
    assert canonical("not-a-smiles") is None
    assert canonical("CCO") == "CCO"


def test_max_tanimoto_is_one_against_itself():
    f = morgan("CCO")
    assert max_tanimoto_to_reference([f], [f])[0] == pytest.approx(1.0)


def test_max_tanimoto_is_low_for_unrelated_molecules():
    f = morgan("CCO")
    ref = morgan("c1ccc2ccccc2c1")  # naphthalene
    assert max_tanimoto_to_reference([f], [ref])[0] < 0.2


def test_intdiv1_is_zero_for_a_single_molecule_with_itself():
    f = morgan("CCO")
    assert intdiv1([f]) == pytest.approx(0.0)


def test_intdiv1_is_positive_for_distinct_molecules():
    fps = [morgan("CCO"), morgan("c1ccccc1"), morgan("CC(=O)O")]
    assert 0.0 < intdiv1(fps) < 1.0


def test_metal_profile_counts_exactly_one_metal():
    out = metal_profile(["CCO", "[Ru](Cl)(Cl)(Cl)(Cl)(Cl)Cl", "c1ccccc1"])
    assert out["n_parsed"] == 3
    assert out["n_exactly_one_metal"] == 1
    assert out["compliance"] == pytest.approx(1 / 3, abs=1e-4)


def test_metal_profile_reports_the_metal_identity():
    out = metal_profile(["[Ir](Cl)(Cl)(Cl)(Cl)(Cl)Cl"])
    assert out["metal_distribution"] == {"Ir": 1}
