"""Docked graph/charge/stereo/pose alignment must survive Meeko conversion."""
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from molmetal.domain import Molecule, Pocket
from molmetal.ports import DockingConfig
from molmetal.molmetal_lam.sbdd_env import vina_adapter as module


def test_docking_preserves_chemistry_and_all_conformers(monkeypatch, tmp_path):
    import meeko
    source = Chem.AddHs(Chem.MolFromSmiles('C[C@H](O)C[NH3+]'))
    AllChem.EmbedMultipleConfs(source, numConfs=2, randomSeed=42)
    # Distinct coordinates prove each energy is mapped to its own pose.
    conf = source.GetConformer(1)
    for i in range(source.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x + 10, p.y, p.z))
    monkeypatch.setattr(meeko, 'PDBQTMolecule', lambda *a, **k: object())
    monkeypatch.setattr(meeko.RDKitMolCreate, 'from_pdbqt_mol', lambda *a, **k: [source])
    monkeypatch.setattr(module, '_ligand_to_pdbqt', lambda *a, **k: tmp_path / 'lig.pdbqt')
    adapter = module.VinaDockingAdapter(engine='vina', cpu_count=1)
    monkeypatch.setattr(adapter, '_prepare_receptor', lambda p: tmp_path / 'rec.pdbqt')
    monkeypatch.setattr(adapter, '_dock_python_binding', lambda **k: (np.array([[-7,0,0,0,0],[-6,0,0,0,0]]), 'poses'))
    pocket = Pocket('test', torch.zeros(1,3), torch.tensor([6]), torch.tensor([1]), ['A'], torch.ones(1,dtype=torch.bool), torch.zeros(3))
    rows = adapter.dock(Molecule.from_rdkit_mol(source), pocket, DockingConfig(n_poses=2))
    assert len(rows) == len(adapter.last_pose_mols) == 2
    assert [r.vina_score for r in rows] == [-7, -6]
    expected = Chem.RemoveHs(source)
    for i, row in enumerate(rows):
        assert row.molecule.n_bonds == expected.GetNumBonds() * 2
        assert row.molecule.formal_charges.sum().item() == 1
        np.testing.assert_allclose(row.molecule.coords, expected.GetConformer(i).GetPositions(), atol=1e-6)
        assert Chem.MolToSmiles(adapter.last_pose_mols[i]) == Chem.MolToSmiles(expected)
        assert adapter.last_pose_mols[i].GetNumConformers() == 1


def test_physical_denominator_keeps_failed_and_unselected_products():
    from molmetal.scripts.evaluate_generated_poses import summarize
    result = summarize([{'score_kcal_mol': -5, 'posebusters': {'pb_valid': True}},
                        {'status': 'failed'}, {'score_kcal_mol': float('nan')}], 5)
    assert result['n_docked'] == 1
    assert result['pb_pass_rate_all_generated'] == .2
    assert result['pb_pass_rate_selected'] == 1/3
    assert result['vina_mean_docked_kcal_mol'] == -5


def test_cli_invalid_score_preserves_corresponding_model_slot():
    energies=module._parse_vina_result_energies('MODEL 1\nREMARK VINA RESULT: nan 0 0\nENDMDL\nMODEL 2\nREMARK VINA RESULT: -7.5 0 0\nENDMDL\n')
    assert energies.shape==(2,5)
    assert np.isnan(energies[0,0]) and energies[1,0]==-7.5
    missing=module._parse_vina_result_energies('MODEL 1\nENDMDL\nMODEL 2\nREMARK VINA RESULT: -3 0 0\nENDMDL\n')
    assert np.isnan(missing[0,0]) and missing[1,0]==-3
