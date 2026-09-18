"""Seed retention must never count as generation; no docking needed."""
from types import SimpleNamespace

import pytest

from molmetal.scripts import lambda_100pocket_sweep as runner


@pytest.fixture
def azide_and_alkyne(monkeypatch):
    monkeypatch.setattr(runner, "build_seed_smiles", lambda _: "CCN=[N+]=[N-]")
    monkeypatch.setattr(runner, "build_tile_library", lambda *a, **kw: [SimpleNamespace(smiles="C#CC")])


def run(tmp_path):
    return runner.run_one_pocket(str(tmp_path), 2, 1, 0, False,
        tile_library="standard_12", click_rules="CuAAC", seed=0)


def test_root_only_is_not_generated(tmp_path, monkeypatch, azide_and_alkyne):
    monkeypatch.setattr(runner.MCTSProofSearch, "search", lambda self, seed, max_depth: [seed])
    result = run(tmp_path)
    assert result["status"] == "ok"  # execution success, not generation success
    assert result["generation_status"] == "seed_only"
    assert result["n_seed_candidates"] == 1
    assert result["n_generated_candidates"] == result["n_unique_generated"] == 0
    assert result["has_generated_candidates"] is False
    assert result["candidates"][0]["is_generated"] is False
    assert result["generation_diagnostics"]["no_generated_reasons"]


def test_actual_click_product_is_generated(tmp_path, monkeypatch, azide_and_alkyne):
    def click(self, seed, max_depth):
        products = self._safe_reduce(self.rules["CuAAC"], seed, self.tile_library[0])
        assert products  # a real CuAAC product, not a made-up molecule fixture
        return [seed, products[0], products[0]]
    monkeypatch.setattr(runner.MCTSProofSearch, "search", click)
    result = run(tmp_path)
    assert result["n_seed_candidates"] == 1
    assert result["n_generated_candidates"] == 2
    assert result["n_unique_generated"] == 1
    assert result["generation_status"] == "generated"
    assert result["generation_diagnostics"]["unique_new_products_observed"] >= 1
    assert not result["generation_diagnostics"]["no_generated_reasons"]


def test_atom_maps_hydrogens_and_smiles_order_do_not_invent_novelty():
    assert runner._structure_key("[CH3:7]CO") == runner._structure_key("OCC")
    assert runner._structure_key("[H]OC([H])([H])C([H])([H])[H]") == runner._structure_key("CCO")
    for bad in ("", "*", "C*", "bad-smiles"):
        assert runner._structure_key(bad) is None


def test_nonmatching_reaction_explains_seed_only(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "build_seed_smiles", lambda _: "CCO")
    monkeypatch.setattr(runner, "build_tile_library", lambda *a, **kw: [SimpleNamespace(smiles="CCO")])
    result = run(tmp_path)  # real MCTS, CuAAC cannot consume ethanol
    diag = result["generation_diagnostics"]
    assert result["n_generated_candidates"] == 0
    assert diag["reduction_calls"] > 0
    assert diag["products_returned"] == 0
    assert diag["reductions_without_products"] == diag["reduction_calls"]
    assert any("non-matching" in reason for reason in diag["no_generated_reasons"])


def test_observer_preserves_gate_results_and_labels_overlapping_scope():
    product = runner.MoleculeClosedTerm.from_smiles("CCN", embed_3d=False)
    fake = SimpleNamespace(_safe_reduce=lambda *a: [product],
        _satisfies_predicates=lambda *a: False, _binds_target=lambda *a: False)
    counts, states = runner._observe_search(fake, runner._structure_key("CCO"))
    assert fake._safe_reduce(None, None, None) == [product]
    assert fake._satisfies_predicates(product, []) is False
    assert fake._binds_target(product) is False
    assert counts["products_returned"] == 1
    assert states["typed_rejected"] == states["binding_rejected"] == {"CCN"}


def test_synthesis_rejections_remain_in_generation_denominator(tmp_path, monkeypatch, azide_and_alkyne):
    from molmetal_lam.search_alg import sweep_guidance
    from molmetal_lam.sbdd_env.aizynth_adapter import RetrosynthesisReport

    def click(self, seed, max_depth):
        products = self._safe_reduce(self.rules["CuAAC"], seed, self.tile_library[0])
        return [seed, products[0], products[0]]

    monkeypatch.setattr(runner.MCTSProofSearch, "search", click)
    monkeypatch.setattr(sweep_guidance, "build_synthesis_gate", lambda *args: (
        lambda smi: RetrosynthesisReport(smi, False, 0, engine="aizynthfinder"),
        {"status": "ready", "backend": "aizynthfinder", "learned": True, "applied": True}))
    result = run(tmp_path)
    assert result["generation_status"] == "generated"
    assert result["has_generated_candidates"]
    assert result["n_generated_candidates"] == 2
    assert result["n_unique_generated"] == 1
    assert result["n_generated_synthesis_rejected_candidates"] == 2
    assert result["n_synthesis_rejected_candidates"] == 3
    assert result["n_retained_candidates"] == result["n_selected_candidates"] == 0
    assert result["n_evaluated_candidates"] == 0
    assert result["candidates"] == []
    assert len(result["all_candidates"]) == 3
    assert all(c["synthesis_passed"] is False for c in result["all_candidates"])
    assert not result["generation_diagnostics"]["no_generated_reasons"]


def test_top_k_selection_does_not_erase_other_generated_products(tmp_path, monkeypatch, azide_and_alkyne):
    def click(self, seed, max_depth):
        products = self._safe_reduce(self.rules["CuAAC"], seed, self.tile_library[0])
        return [products[0], products[0], seed]

    monkeypatch.setattr(runner.MCTSProofSearch, "search", click)
    result = runner.run_one_pocket(str(tmp_path), 2, 1, 0, False,
        tile_library="standard_12", click_rules="CuAAC", seed=0, top_k=1)
    assert result["n_pre_synthesis_candidates"] == result["n_retained_candidates"] == 3
    assert result["n_generated_candidates"] == 2
    assert result["n_generated_selected_candidates"] == 1
    assert len(result["candidates"]) == 1
    assert [c["selected_for_evaluation"] for c in result["all_candidates"]] == [True, False, False]
    assert all(c["synthesis_passed"] is None for c in result["all_candidates"])
