"""Measured reward must run during real search and failures stay observable."""
import sys
from types import SimpleNamespace
import pytest

from molmetal.scripts import lambda_100pocket_sweep as runner
from molmetal.scripts import r4_c_full_sweep as sweep


@pytest.mark.parametrize('weight', [0.0, 0.4])
def test_measured_provider_is_called_inside_actual_mcts(tmp_path, monkeypatch, weight):
    instances = []
    class Provider:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.calls = []
            instances.append(self)
        def __call__(self, state):
            self.calls.append(state.canonical_smiles())
            return -float(state.to_rdkit().GetNumHeavyAtoms())
        def report(self):
            return {'n_docked': len(self.calls), 'smiles': self.calls}
    monkeypatch.setitem(sys.modules, 'molmetal_lam.sbdd_env.pocket_docking_reward',
                        SimpleNamespace(PocketDockingReward=Provider))
    result = runner.run_one_pocket(str(tmp_path), 4, 1, 0, False, seed=42,
        click_rules='CuAAC', tile_library='standard_12', seed_strategy='click_tile',
        docking_reward_config={'receptor_path': 'exact.pdb', 'ligand_path': 'exact.sdf',
                               'output_dir': str(tmp_path), 'weight': weight})
    assert result['n_generated_candidates'] > 0
    assert instances[0].calls
    assert instances[0].options['seed'] == 42
    assert instances[0].options['receptor_path'] == 'exact.pdb'
    assert 'weight' not in instances[0].options
    report = result['docking_reward_report']
    assert report['applied'] is (weight > 0)
    assert report['energy_feedback_enabled'] is (weight > 0)
    assert report['status'] == 'executed'
    assert report['execution']['n_docked'] == len(instances[0].calls)
    assert result['search_config']['guidance']['docking_reward'] == report


def test_unavailable_reward_keeps_independent_generation_and_cannot_pass_as_conditioned(tmp_path, monkeypatch):
    class Unavailable:
        def __init__(self, **kwargs):
            raise RuntimeError('receptor preparation failed')
    monkeypatch.setitem(sys.modules, 'molmetal_lam.sbdd_env.pocket_docking_reward',
                        SimpleNamespace(PocketDockingReward=Unavailable))
    result = sweep.run_one_pocket(str(tmp_path), 4, 1, seed=42,
        receptor_path='exact.pdb', ligand_path='exact.sdf',
        click_rules='CuAAC', tile_library='standard_12', seed_strategy='click_tile',
        docking_reward_config={'output_dir': str(tmp_path)})
    assert result.n_generated_candidates > 0
    assert result.status == 'docking_reward_unavailable'
    assert result.diagnostics['docking_reward']['applied'] is False
    assert 'receptor preparation failed' in result.diagnostics['docking_reward']['error']


def test_energy_weight_changes_actual_search_order_when_oracle_disagrees(tmp_path, monkeypatch):
    class SmallPreferred:
        def __init__(self, **kwargs):
            self.calls = 0
        def __call__(self, state):
            self.calls += 1
            return float(state.to_rdkit().GetNumHeavyAtoms()) * 10.0
        def report(self):
            return {'n_docked': self.calls}
    monkeypatch.setitem(sys.modules, 'molmetal_lam.sbdd_env.pocket_docking_reward',
                        SimpleNamespace(PocketDockingReward=SmallPreferred))
    treatments = []
    for weight in (0.0, .4):
        result = runner.run_one_pocket(str(tmp_path), 4, 1, 0, False, seed=42,
            click_rules='CuAAC', tile_library='standard_12', seed_strategy='click_tile',
            docking_reward_config={'weight': weight})
        assert result['docking_reward_report']['execution']['n_docked'] > 0
        treatments.append(result)
    off, on = treatments
    assert {c['smiles'] for c in off['candidates']} == {c['smiles'] for c in on['candidates']}
    assert off['candidates'][0]['smiles'] != on['candidates'][0]['smiles']
