"""Bounded config/chemistry checks: no docking or pocket sweep required."""
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from rdkit import Chem

from molmetal.scripts import lambda_100pocket_sweep as runner


@pytest.fixture
def tiny_pool(monkeypatch):
    tiles = [SimpleNamespace(smiles=s) for s in ["C#CC", "CCO", "CCN", "CC(=O)O"]]
    monkeypatch.setattr(runner, "build_tile_library", lambda *a, **kw: tiles)
    monkeypatch.setattr(runner, "FRAGMENT_LIBRARY_200_TILES", lambda: tiles)
    monkeypatch.setattr(runner, "build_seed_smiles", lambda _: "CCN=[N+]=[N-]")
    return tiles


def run(tmp_path, **kwargs):
    return runner.run_one_pocket(str(tmp_path), 2, 1, 0, False, **kwargs)


def test_real_expansion_uses_selected_tiles_and_rules(tmp_path, tiny_pool, monkeypatch):
    snapshots = []
    real_search = runner.MCTSProofSearch.search

    def inspect(self, seed, max_depth):
        self._expand(seed)
        snapshots.append((list(self.rules), len(self._resolve_expand_tile_pool()), self.nfe_reductions))
        return real_search(self, seed, max_depth=max_depth)

    monkeypatch.setattr(runner.MCTSProofSearch, "search", inspect)
    result = run(tmp_path, tile_library="standard_12", click_rules="CuAAC",
                 branching_target=2, top_k=1, early_stop=False, patience=4)
    assert snapshots[0] == (["CuAAC"], 2, 2)
    assert result["status"] == "ok"
    assert result["search_config"]["branching_attempts"] == 2
    assert result["search_config"]["use_fragment_pool"] is False
    assert result["search_config"]["patience"] == 4
    assert result["n_candidates"] <= 1


def test_all_five_and_nondivisible_cap_are_real(tmp_path, tiny_pool, monkeypatch):
    monkeypatch.setattr(runner.MCTSProofSearch, "search", lambda self, seed, max_depth: [])
    result = run(tmp_path, tile_library="extended_204", click_rules="all_5", branching_target=12)
    cfg = result["search_config"]
    assert cfg["rule_names"] == ["CuAAC", "SPAAC", "ThiolEne", "Suzuki", "AmideCoupling"]
    assert cfg["tile_count"] == 2
    assert cfg["branching_attempts"] == 10  # floor(12 / 5) actual tiles
    assert result["empty_candidates_reason"]


def test_top_k_controls_output_and_real_constructor(tmp_path, tiny_pool, monkeypatch):
    seen = {}
    def search(self, seed, max_depth):
        seen.update(top_k=self.top_k, early_stop=self.early_stop, patience=self.patience)
        return [seed] * 25
    monkeypatch.setattr(runner.MCTSProofSearch, "search", search)
    result = run(tmp_path, top_k=23, early_stop=False, patience=9)
    assert seen == dict(top_k=23, early_stop=False, patience=9)
    assert result["n_candidates"] == 23  # no hidden candidates[:20]


@pytest.mark.parametrize("kwargs", [dict(branching_target=0), dict(branching_target=True),
    dict(branching_target=2, click_rules="all_5"), dict(click_rules="not_a_rule"),
    dict(tile_library="not_a_library"), dict(top_k=0)])
def test_invalid_configuration_fails_loudly(tmp_path, kwargs):
    with pytest.raises(ValueError):
        run(tmp_path, **kwargs)


def test_explicit_ligand_and_seed_have_no_hidden_fallback(tmp_path, tiny_pool, monkeypatch):
    ligand = tmp_path / "paired.sdf"
    with Chem.SDWriter(str(ligand)) as writer:
        writer.write(Chem.MolFromSmiles("CCO"))
    draws = []
    def search(self, seed, max_depth):
        draws.append((self.rng.random(), random.random(), np.random.random(), torch.rand(1).item()))
        return []
    monkeypatch.setattr(runner.MCTSProofSearch, "search", search)
    for _ in range(2):
        result = run(tmp_path, ligand_path=str(ligand), seed=42)
        assert result["seed_smiles"] == "CCO"
        assert result["search_config"]["seed"] == 42
    assert draws[0] == draws[1]
    failed = run(tmp_path, ligand_path=str(tmp_path / "missing.sdf"))
    assert failed["status"] == "seed_parse_fail"
    assert failed["seed_smiles"] is None


def test_ertl_score_matches_reference():
    from rdkit.Contrib.SA_Score import sascorer
    assert runner._sa_score("CCO") == pytest.approx(sascorer.calculateScore(Chem.MolFromSmiles("CCO")))
    assert np.isnan(runner._sa_score("invalid"))


def test_optional_requests_are_not_silently_applied(tmp_path, tiny_pool, monkeypatch):
    monkeypatch.setattr(runner.MCTSProofSearch, "search", lambda self, seed, max_depth: [])
    result = run(tmp_path, symbolic_prior="ridge", synthesis_oracle="external",
                 symbolic_prior_refit_every=7)
    assert set(result["search_config"]["not_applied"]) == {
        "symbolic_prior", "synthesis_oracle", "symbolic_prior_refit_every"}


def test_early_stop_changes_real_simulation_budget(tmp_path, tiny_pool):
    outputs = [runner.run_one_pocket(str(tmp_path), 8, 1, 0, False,
               tile_library="standard_12", click_rules="CuAAC", branching_target=1,
               early_stop=enabled, patience=1, seed=7) for enabled in (True, False)]
    short, full = [r["search_diagnostics"] for r in outputs]
    assert short["early_stopped"]
    assert short["simulations_completed"] < full["simulations_completed"] == 8
