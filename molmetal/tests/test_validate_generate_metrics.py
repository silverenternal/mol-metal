"""Behavioral checks for the round-13 generation descriptor summaries."""

import math

import pytest
import torch

from molmetal.domain import Molecule
from molmetal.scripts.validate_generate import evaluate_set, mol_to_rdkit


def candidate(smiles="", *, atoms=(6, 6)):
    return Molecule(
        coords=torch.zeros(len(atoms), 3),
        atom_types=torch.tensor(atoms, dtype=torch.long),
        bonds=torch.zeros(2, 0, dtype=torch.long),
        bond_types=torch.zeros(0, dtype=torch.long),
        formal_charges=torch.zeros(len(atoms), dtype=torch.long),
        smiles=smiles,
    )


def test_empty_has_nan_statistics_and_no_passes():
    metrics = evaluate_set([])
    assert metrics["n_total"] == metrics["n_valid"] == metrics["n_invalid"] == 0
    assert metrics["mw_range_300_700_flags"] == []
    for name in ("qed", "mw", "logp", "tpsa", "rotb"):
        assert math.isnan(metrics[f"{name}_mean"])
        assert math.isnan(metrics[f"{name}_std"])
    assert all(value == 0.0 for key, value in metrics.items() if "frac" in key and key != "fraction_denominator")


@pytest.mark.parametrize("molecule", [
    candidate("invalid smiles"), candidate("C(C)(C)(C)(C)C"),
    candidate("*"), candidate(" "), candidate(), candidate(atoms=()),
])
def test_invalid_or_undecoded_candidates_do_not_get_fabricated_descriptors(molecule):
    assert mol_to_rdkit(molecule) is None
    metrics = evaluate_set([molecule])
    assert metrics["n_valid"] == 0
    assert metrics["n_invalid"] == 1
    assert metrics["mw_range_300_700_flags"] == [None]
    assert metrics["rotb_frac_lt_10"] == 0.0
    assert metrics["logp_frac_2_5"] == 0.0
    assert metrics["tpsa_frac_60_150"] == 0.0


def test_known_molecules_have_population_std_and_all_candidate_denominators():
    metrics = evaluate_set([
        candidate("CC(=O)Oc1ccccc1C(=O)O"),  # aspirin
        candidate("CCCCCCCCCC"),  # decane
        candidate("invalid smiles"),
    ])
    assert metrics["n_valid"] == 2
    assert metrics["n_total"] == 3
    assert metrics["std_ddof"] == 0
    assert metrics["logp_mean"] == pytest.approx((1.3101 + 4.1470) / 2)
    assert metrics["logp_std"] == pytest.approx((4.1470 - 1.3101) / 2)
    assert metrics["tpsa_mean"] == pytest.approx(31.8)
    assert metrics["tpsa_std"] == pytest.approx(31.8)
    assert metrics["rotb_mean"] == pytest.approx(4.5)
    assert metrics["rotb_std"] == pytest.approx(2.5)
    assert metrics["logp_frac_2_5"] == pytest.approx(1 / 3)
    assert metrics["tpsa_frac_60_150"] == pytest.approx(1 / 3)
    assert metrics["rotb_frac_lt_10"] == pytest.approx(2 / 3)


def test_rotatable_bond_cutoff_is_strictly_less_than_ten():
    metrics = evaluate_set([candidate("C" * 12), candidate("C" * 13)])
    # Dodecane has 9 rotatable bonds, tridecane has 10.
    assert metrics["rotb_mean"] == pytest.approx(9.5)
    assert metrics["rotb_std"] == pytest.approx(0.5)
    assert metrics["rotb_frac_lt_10"] == 0.5
    assert metrics["logp_frac_2_5"] == 0.5
    assert metrics["logp_frac_gt_5"] == 0.5


def test_molecular_weight_flags_never_filter_out_high_weight_molecules():
    metrics = evaluate_set([
        candidate("C" * 22), candidate("C" * 51), candidate("invalid smiles"),
    ])
    assert metrics["n_valid"] == 2
    assert metrics["mw_range_300_700_flags"] == [True, False, None]
    assert metrics["mw_frac_300_700"] == pytest.approx(1 / 3)
    assert metrics["mw_mean"] == pytest.approx((310.610 + 717.393) / 2)
    assert metrics["logp_mean"] > 10.0  # Both candidates are retained.


def test_decoded_bonds_are_used_when_smiles_is_missing():
    # Methanol graph, with the symmetric edge convention used by the model.
    molecule = Molecule(
        coords=torch.zeros(2, 3), atom_types=torch.tensor([6, 8]),
        bonds=torch.tensor([[0, 1], [1, 0]]), bond_types=torch.tensor([1, 1]),
        formal_charges=torch.zeros(2, dtype=torch.long),
    )
    metrics = evaluate_set([molecule])
    assert metrics["n_valid"] == 1
    assert metrics["mw_mean"] == pytest.approx(32.042)
    assert metrics["tpsa_mean"] == pytest.approx(20.23)
    assert metrics["tpsa_std"] == 0.0


@pytest.mark.parametrize("failure", ["exception", "nan"])
def test_late_descriptor_failure_discards_the_entire_row(monkeypatch, failure):
    from rdkit.Chem import Crippen

    def fail(_mol):
        if failure == "exception":
            raise ValueError("descriptor unavailable")
        return float("nan")

    monkeypatch.setattr(Crippen, "MolLogP", fail)
    metrics = evaluate_set([candidate("CCO")])
    assert metrics["n_valid"] == 0
    assert metrics["n_invalid"] == 1
    assert math.isnan(metrics["qed_mean"])
    assert metrics["rotb_frac_lt_10"] == 0.0
