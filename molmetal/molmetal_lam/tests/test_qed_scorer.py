"""Tests for the RDKit QED scorer public API."""

import math

import pytest

from molmetal_lam.sbdd_env.qed_scorer import QEDScorer, qed_from_smiles


MOLECULES = [
    ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("cisplatin", "Cl[Pt](Cl)(N)N"),
    ("caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"),
    ("cholesterol", "CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CC=C12C"),
    ("glycine", "NCC(=O)O"),
]


def test_score_known_molecules_are_finite_and_bounded():
    scorer = QEDScorer()
    for _, smiles in MOLECULES:
        value = scorer.score(smiles)
        assert value is not None
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0


def test_score_accepts_rdkit_mol():
    rdkit = pytest.importorskip("rdkit")
    mol = rdkit.Chem.MolFromSmiles(MOLECULES[0][1])
    assert QEDScorer().score(mol) == pytest.approx(QEDScorer().score(MOLECULES[0][1]))


def test_batch_matches_scalar_scores():
    smiles = [s for _, s in MOLECULES]
    scorer = QEDScorer()
    batch = scorer.score_batch(smiles)
    scalar = [scorer.score(s) for s in smiles]
    assert batch == pytest.approx(scalar)


def test_invalid_smiles_returns_zero():
    scorer = QEDScorer()
    assert scorer.score("this-is-not-smiles") == 0.0
    assert qed_from_smiles("this-is-not-smiles") == 0.0


def test_empty_batch_returns_empty_list():
    assert QEDScorer().score_batch([]) == []


def test_score_with_sa_shape_and_keys():
    smiles = [MOLECULES[0][1], MOLECULES[-1][1]]
    records = QEDScorer().score_with_sa(smiles)
    assert len(records) == len(smiles)
    for record, smi in zip(records, smiles):
        assert set(record) == {"smiles", "qed", "sa"}
        assert record["smiles"] == smi
        assert isinstance(record["qed"], float)
        assert record["sa"] is None or isinstance(record["sa"], float)

