"""A reagent seed must come from a witnessed reaction, never a test ligand."""
from types import SimpleNamespace

import pytest
from rdkit import rdBase

from molmetal.scripts import lambda_100pocket_sweep as runner


def call(tmp_path, **kwargs):
    with rdBase.BlockLogs():
        return runner.run_one_pocket(str(tmp_path), 4, 1, 0, False,
            tile_library="standard_12", click_rules="CuAAC", seed=42,
            early_stop=False, **kwargs)


def test_real_small_mcts_returns_new_molecules_without_relaxing_gates(tmp_path):
    result = call(tmp_path, seed_strategy="click_tile")
    assert result["status"] == "ok"
    assert result["init_strategy"] == "click_tile"
    assert result["n_generated_candidates"] >= 1
    assert result["search_diagnostics"]["simulations_completed"] == 4
    chosen = result["chosen_tile"]
    assert chosen["validated_rule"] == "CuAAC"
    assert chosen["canonical_seed"] == result["canonical_seed"]
    assert chosen["validation_products"]
    assert chosen["validation_reduction_calls"] >= 1
    seed = runner.MoleculeClosedTerm.from_smiles(chosen["smiles"], embed_3d=False)
    partner = runner.MoleculeClosedTerm.from_smiles(chosen["partner_smiles"], embed_3d=False)
    products = runner.REACTION_RULES["CuAAC"].reduce((seed, partner))
    actual = {runner._structure_key(runner._smi_of(p)) for p in products}
    assert set(chosen["validation_products"]).issubset(actual)
    assert result["canonical_seed"] not in chosen["validation_products"]
    for candidate in result["candidates"]:
        if candidate["is_generated"]:
            assert runner._structure_key(candidate["smiles"]) != result["canonical_seed"]


def test_initialization_is_repeatable_and_never_reads_test_ligands(tmp_path, monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail("click_tile must not read paired/reference ligand chemistry")
    monkeypatch.setattr(runner, "build_seed_smiles", forbidden)
    monkeypatch.setattr(runner.Chem, "SDMolSupplier", forbidden)
    one = call(tmp_path, seed_strategy="click_tile", ligand_path="missing_test_ligand.sdf")
    two = call(tmp_path / "other_pocket", seed_strategy="click_tile", ligand_path="other_test_ligand.sdf")
    assert one["chosen_tile"] == two["chosen_tile"]
    assert one["canonical_seed"] == two["canonical_seed"]
    assert one["search_config"]["seed_strategy"] == "click_tile"


def test_no_compatible_pair_fails_explicitly(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "build_tile_library", lambda *a, **kw: [SimpleNamespace(smiles="CCO")])
    result = call(tmp_path, seed_strategy="click_tile")
    assert result["status"] == "initialization_fail"
    assert result["seed_smiles"] is None
    assert result["chosen_tile"] is None
    assert result["has_generated_candidates"] is False
    assert "No reactive ordered" in result["error"]


def test_partner_must_survive_actual_branching_cap(tmp_path, monkeypatch):
    # Only the azide survives the one-tile expansion cap. CuAAC requires the
    # partner slot to contain an alkyne, even though an alkyne exists in the
    # full seed library; reversing reactant order is not allowed.
    monkeypatch.setattr(runner, "build_tile_library", lambda *a, **kw: [
        SimpleNamespace(smiles="CCN=[N+]=[N-]"), SimpleNamespace(smiles="C#CC")])
    result = call(tmp_path, seed_strategy="click_tile", branching_target=1)
    assert result["status"] == "initialization_fail"


def test_reference_default_remains_explicit_ligand_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "build_seed_smiles", lambda _: "CCO")
    result = call(tmp_path)
    assert result["init_strategy"] == "reference"
    assert result["canonical_seed"] == "CCO"
    assert result["chosen_tile"] is None
    assert result["n_generated_candidates"] == 0


def test_unknown_strategy_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="seed_strategy"):
        call(tmp_path, seed_strategy="random_smiles")
