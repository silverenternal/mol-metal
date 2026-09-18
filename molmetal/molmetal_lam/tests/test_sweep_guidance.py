"""No synthetic training labels, no test-set learning by default, no fake oracle."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from rdkit.Chem import QED

from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
from molmetal_lam.search_alg import sweep_guidance as guidance
from molmetal.scripts import lambda_100pocket_sweep as runner


@pytest.fixture
def observations():
    # Actual descriptor reward measurements on development molecules.
    states = [MoleculeClosedTerm.from_smiles(s, embed_3d=False) for s in
              ["CCO", "CCCO", "CCCCO", "CCCCCO", "CCCCCCO", "c1ccccc1", "CCN", "CCCCN"]]
    score = SimpleNamespace(score_final=lambda s: float(QED.qed(s.to_rdkit())))
    return guidance.candidate_observations(states, score, "development_fixture")


def train(observations):
    state, _ = guidance.prepare_prior(None, seed=42, mode="train", data_split="development")
    return guidance.update_prior(state, observations, mode="train", data_split="development", refit_every=1)


def test_real_observations_fit_serializable_linear_prior(observations):
    state, report = train(observations)
    assert report["fit_performed"] is True
    assert state["backend"] == "linear_descriptor_prior"
    assert state["source_splits"] == ["development"]
    restored, prior = guidance.prepare_prior(json.loads(json.dumps(state, allow_nan=False)), seed=7)
    assert restored == state
    assert prior.fitted
    small = MoleculeClosedTerm.from_smiles("CCO", embed_3d=False)
    large = MoleculeClosedTerm.from_smiles("CCCCCCO", embed_3d=False)
    assert prior.predict_proba(small) != prior.predict_proba(large)


def test_warmup_requires_enough_real_observations_and_variation(observations):
    state, report = train(observations[:2])
    assert state["model"] is None
    assert report["update_status"] == "insufficient_real_observations"
    constant = [{**row, "score": 0.5} for row in observations]
    state, report = train(constant)
    assert state["model"] is None
    assert report["update_status"] == "insufficient_observed_variation"


def test_frozen_state_never_collects_test_rewards_or_refits(observations):
    state, _ = train(observations)
    original = deepcopy(state)
    output, report = guidance.update_prior(state, [{"invalid": "heldout data must not be read"}],
        mode="frozen", data_split="test", refit_every=1)
    assert output == state == original
    assert report == {"update_status": "frozen", "fit_performed": False}


def test_refit_interval_counts_pockets_and_repeated_data_is_not_new(observations):
    state, _ = guidance.prepare_prior(None, seed=42, mode="train", data_split="development")
    state, report = guidance.update_prior(state, observations, mode="train", data_split="development", refit_every=2)
    assert not report["fit_performed"] and state["pockets_seen"] == 1
    state, report = guidance.update_prior(state, observations, mode="train", data_split="development", refit_every=2)
    assert report["fit_performed"] and state["pockets_seen"] == 2
    state, report = guidance.update_prior(state, observations, mode="train", data_split="development", refit_every=1)
    assert report["update_status"] == "no_new_observations"
    assert state["fit_count"] == 1


def test_no_silent_test_training_or_shared_seed_state(observations):
    with pytest.raises(ValueError, match="held-out"):
        guidance.prepare_prior(None, seed=42, mode="train", data_split="test")
    state, _ = train(observations)
    with pytest.raises(ValueError, match="independent"):
        guidance.prepare_prior(state, seed=0, mode="train", data_split="development")
    state, _ = guidance.prepare_prior(None, seed=42, mode="transductive", data_split="test")
    state, _ = guidance.update_prior(state, observations, mode="transductive", data_split="test", refit_every=1)
    assert state["transductive"] is True
    with pytest.raises(ValueError, match="transductive"):
        guidance.prepare_prior(state, seed=42, mode="frozen")


def test_fitted_prior_actually_precedes_reward_heuristic(observations):
    state, _ = train(observations)
    _, prior = guidance.prepare_prior(state, seed=42)
    mol = MoleculeClosedTerm.from_smiles("CCO", embed_3d=False)
    # Exercise the actual overridden PUCT dispatcher with a distinct fallback.
    search = guidance.GuidedMCTS(tile_library=[mol], rules={}, target_predicates=[],
        binding_site=runner.PROTEASE_GENERIC, prior=prior, use_fragment_pool=False)
    search.heuristic = lambda state: 0.123
    assert search._prior(mol) == pytest.approx(prior.predict_proba(mol))
    assert search._prior(mol) != 0.123
    assert search.learned_prior_calls == 2
    search.prior = None
    assert search._prior(mol) == 0.123


def test_runner_frozen_prior_does_not_mutate_state(tmp_path, monkeypatch, observations):
    state, _ = train(observations)
    original = deepcopy(state)
    monkeypatch.setattr(runner, "build_seed_smiles", lambda _: "CCN=[N+]=[N-]")
    monkeypatch.setattr(runner, "build_tile_library", lambda *a, **kw: [SimpleNamespace(smiles="C#CC")])
    result = runner.run_one_pocket(str(tmp_path), 2, 1, 0, False, click_rules="CuAAC",
        tile_library="standard_12", seed=42, symbolic_prior=True, prior_state=state)
    assert result["status"] == "ok"
    assert result["prior_report"]["puct_calls"] > 0
    assert result["prior_report"]["mode"] == "frozen"
    assert result["prior_state"] == state == original
    assert result["prior_report"]["pysr_symbolic_regression"] is False


def test_requested_learned_oracle_requires_config_not_sa(tmp_path):
    checker, metadata = guidance.build_synthesis_gate(True, str(tmp_path / 'missing.yml'))
    assert checker is None
    assert metadata["status"] == "missing_aizynth_config"
    assert metadata["backend"] == "aizynthfinder"
    assert metadata["applied"] is False


def test_explicit_smarts_gate_really_filters_and_is_not_learned():
    checker, metadata = guidance.build_synthesis_gate('smarts')
    good = MoleculeClosedTerm.from_smiles("CCn1cc(C)nn1", embed_3d=False)
    bad = MoleculeClosedTerm.from_smiles("C", embed_3d=False)
    passed, report = guidance.gate_candidates([good, bad], checker, metadata)
    assert metadata["backend"] == "smarts_heuristic" and metadata["learned"] is False
    assert report["n_checked"] == 2
    assert report["n_passed"] == 1
    assert passed == [good]


def test_aizynth_runtime_fallback_cannot_pass_as_learned_route():
    molecule = MoleculeClosedTerm.from_smiles("CCO", embed_3d=False)
    checker = lambda s: SimpleNamespace(engine="smarts_fallback", synthesizable=True, depth=1)
    passed, report = guidance.gate_candidates([molecule], checker,
        {"backend": "aizynthfinder", "learned": True, "applied": True})
    assert passed == []
    assert report["reports"][0]["status"] == "backend_fallback_rejected"


def test_disabled_settings_are_not_unsupported_requests(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "build_seed_smiles", lambda _: "CCO")
    monkeypatch.setattr(runner, "build_tile_library", lambda *a, **kw: [SimpleNamespace(smiles="CCO")])
    result = runner.run_one_pocket(str(tmp_path), 1, 1, 0, False, symbolic_prior=False,
                                  synthesis_oracle=0, symbolic_prior_refit_every=0)
    assert result['search_config']['not_applied'] == []
    assert set(result['search_config']['disabled']) == {
        'symbolic_prior', 'synthesis_oracle', 'symbolic_prior_refit_every'}


@pytest.mark.parametrize('failure', ['ligand', 'initialization', 'search'])
def test_failed_jobs_preserve_replay_state(tmp_path, monkeypatch, observations, failure):
    state, _ = train(observations)
    monkeypatch.setattr(runner, 'build_seed_smiles', lambda _: 'CCO')
    monkeypatch.setattr(runner, 'build_tile_library', lambda *a, **kw: [SimpleNamespace(smiles='CCO')])
    options = {}
    if failure == 'ligand':
        options['ligand_path'] = str(tmp_path / 'missing.sdf')
    elif failure == 'initialization':
        options['seed_strategy'] = 'click_tile'
    else:
        def broken(*a, **kw):
            raise RuntimeError('deliberate test failure')
        monkeypatch.setattr(runner.MCTSProofSearch, 'search', broken)
    result = runner.run_one_pocket(str(tmp_path), 1, 1, 0, False, symbolic_prior=True,
                                  prior_state=state, seed=42, **options)
    assert result['status'] != 'ok'
    assert result['prior_state'] == state
