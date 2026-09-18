"""Behavioral tests for retrosynthesis cost estimation."""

import math

import pytest

from molmetal_lam.synthesis.cost_estimator import SynthesisCostEstimator, synthesis_cost


MOLECULES = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "cisplatin": "N.N.Cl[Pt]Cl",
    "glucose": "C(C1C(C(C(C(O1)O)O)O)O)O",
    "cholesterol": "CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CC(C)C12C",
}


@pytest.mark.parametrize("name", MOLECULES)
def test_known_molecules_score_in_range(name):
    value = synthesis_cost(MOLECULES[name])
    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0
    assert math.isfinite(value)


def test_building_block_exact_match_is_cheap():
    smiles = MOLECULES["aspirin"]
    assert synthesis_cost(smiles, [smiles]) < synthesis_cost(smiles, [])


def test_empty_inventory_has_higher_cost_than_known_inventory():
    smiles = MOLECULES["glucose"]
    empty = synthesis_cost(smiles, [])
    listed = synthesis_cost(smiles, [smiles])
    assert empty > listed


def test_custom_weights_are_normalized_and_used():
    smiles = MOLECULES["aspirin"]
    estimator = SynthesisCostEstimator([smiles], sa_weight=2, bb_weight=1, complexity_weight=1)
    value = estimator.cost(smiles)
    assert 0.0 <= value <= 1.0
    assert estimator.sa_weight == pytest.approx(0.5)
    assert estimator.bb_weight == pytest.approx(0.25)


def test_invalid_smiles_returns_impossible_cost():
    assert synthesis_cost("this is not smiles") == 1.0
    assert SynthesisCostEstimator().cost("[invalid") == 1.0

