"""Pose checking must preserve docking coordinates and retain failures in rates."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from molmetal.validation import posebusters_runner as runner


CONFIG = {"modules": [{"name": "Loading", "chosen_binary_test_output": ["loaded"],
                       "rename_outputs": {"loaded": "MOL_COND loaded"}},
                      {"name": "Geometry", "chosen_binary_test_output": ["geometry_ok"]}]}


@pytest.fixture
def pose():
    mol = Chem.MolFromSmiles("CCO")
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.Set3D(True)
    for i, xyz in enumerate(((12.4, -3, 7), (13.8, -2.5, 6.8), (14.9, -2, 7.1))):
        conf.SetAtomPosition(i, xyz)
    mol.AddConformer(conf)
    return mol


@pytest.fixture
def receptor(tmp_path):
    path = tmp_path / "receptor.pdb"
    path.write_text("END\n")
    return path


def backend(monkeypatch, frame, callback=None):
    class FakePB:
        def __init__(self, config):
            assert config == "dock"
            self.config = CONFIG
        def bust(self, *, mol_pred, mol_cond, full_report):
            assert full_report is True
            if callback:
                callback(mol_pred, mol_cond)
            return frame
    monkeypatch.setitem(sys.modules, "posebusters", SimpleNamespace(PoseBusters=FakePB))
    monkeypatch.setattr(runner, "posebusters_available", lambda: True)


def test_coordinates_and_graph_are_preserved_without_embedding(monkeypatch, pose, receptor):
    before = pose.GetConformer().GetPositions().copy()
    graph = Chem.MolToSmiles(pose)
    def forbidden(*a, **kw):
        pytest.fail("Input docked pose must not be embedded or optimized")
    for fn in ("EmbedMolecule", "EmbedMultipleConfs", "MMFFOptimizeMolecule", "UFFOptimizeMolecule"):
        monkeypatch.setattr(AllChem, fn, forbidden)
    def capture(mol_pred, mol_cond):
        assert mol_pred is not pose
        assert mol_cond == str(receptor)
        np.testing.assert_array_equal(mol_pred.GetConformer().GetPositions(), before)
        assert Chem.MolToSmiles(mol_pred) == graph
        mol_pred.GetConformer().SetAtomPosition(0, (0, 0, 0))  # backend cannot mutate caller
    backend(monkeypatch, pd.DataFrame({"mol_cond_loaded": [True], "geometry_ok": [True]}), capture)
    result = runner.check_docked_pose(pose, receptor)
    assert result["status"] == "passed"
    assert result["pb_valid"] is True
    assert result["backend"] == "posebusters" and result["config"] == "dock"
    assert result["checks"] == {"mol_cond_loaded": True, "geometry_ok": True}
    np.testing.assert_array_equal(pose.GetConformer().GetPositions(), before)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("value", [False, None, np.nan, pd.NA, 1.0, "True"])
def test_missing_or_nonboolean_checks_never_pass(monkeypatch, pose, receptor, value):
    backend(monkeypatch, pd.DataFrame({"mol_cond_loaded": [True], "geometry_ok": [value]}))
    result = runner.check_docked_pose(pose, receptor)
    assert result["pb_valid"] is False
    assert "geometry_ok" in result["failures"]
    json.dumps(result, allow_nan=False)


def test_absent_expected_column_is_recorded(monkeypatch, pose, receptor):
    # Irrelevant numeric diagnostics and unselected mol_true loading cannot
    # substitute for a missing required geometry check.
    backend(monkeypatch, pd.DataFrame({"mol_cond_loaded": [True], "energy": [1.2], "mol_true_loaded": [False]}))
    result = runner.check_docked_pose(pose, receptor)
    assert result["pb_valid"] is False
    assert result["missing_checks"] == ["geometry_ok"]
    assert result["checks"]["geometry_ok"] is None


def test_nullable_booleans_and_all_rows_are_checked(monkeypatch, pose, receptor):
    backend(monkeypatch, pd.DataFrame({"mol_cond_loaded": pd.Series([True, True], dtype="boolean"),
                                      "geometry_ok": pd.Series([True, False], dtype="boolean")}))
    result = runner.check_docked_pose(pose, receptor)
    assert result["pb_valid"] is False
    assert len(result["row_checks"]) == 2


def test_empty_backend_report_fails(monkeypatch, pose, receptor):
    backend(monkeypatch, pd.DataFrame())
    assert runner.check_docked_pose(pose, receptor)["pb_valid"] is False


def test_backend_failure_and_unavailability_are_explicit(monkeypatch, pose, receptor):
    def broken(*a):
        raise RuntimeError("backend failed")
    backend(monkeypatch, None, broken)
    failed = runner.check_docked_pose(pose, receptor)
    assert failed["status"] == "error"
    assert failed["pb_valid"] is False
    monkeypatch.setattr(runner, "posebusters_available", lambda: False)
    unavailable = runner.check_docked_pose(pose, receptor)
    assert unavailable["status"] == "unavailable"
    assert unavailable["pb_valid"] is None
    assert runner.pb_pass_rate([{"pb_valid": True}, failed, unavailable]) == pytest.approx(1 / 3)


def test_invalid_coordinate_inputs_do_not_reembed(pose, receptor):
    missing = runner.check_docked_pose(Chem.MolFromSmiles("CCO"), receptor)
    assert missing["status"] == "invalid_input"
    pose.GetConformer().SetAtomPosition(0, (np.nan, 0, 0))
    assert runner.check_docked_pose(pose, receptor)["failures"] == ["nonfinite_input_coordinates"]


def test_all_inputs_stay_in_pass_rate_denominator():
    results = [{"pb_valid": True}, {"pb_valid": False}, {"pb_valid": None}, {}, None,
               {"pb_valid": float("nan")}, {"pb_valid": "yes"}, {"pb_valid": 1},
               {"pb_valid": True, "skipped": True}]
    assert runner.pb_pass_rate(results) == pytest.approx(1 / 9)
    assert runner.pb_pass_rate([]) == 0.0


def test_embedding_returns_real_conformer_ids(monkeypatch):
    optimized = []
    def embed(mol, numConfs, params):
        for cid in (4, 9):
            conf = Chem.Conformer(mol.GetNumAtoms())
            conf.SetId(cid)
            mol.AddConformer(conf, assignId=False)
        return [4, 9]  # mimics the iterable ID-vector contract
    monkeypatch.setattr(AllChem, "EmbedMultipleConfs", embed)
    monkeypatch.setattr(AllChem, "MMFFOptimizeMolecule", lambda mol, confId, maxIters: optimized.append(confId) or 0)
    result = runner._embed_conformers("CCO", 2, "ETKDGv3", "MMFF94")
    assert [cid for _, cid in result] == [4, 9]
    assert optimized == [4, 9]


def test_real_rdkit_embedding_is_not_an_empty_failure():
    result = runner._embed_conformers("CCO", 2, "ETKDGv3", "MMFF94")
    assert len(result) == 2
    for mol, cid in result:
        assert np.isfinite(mol.GetConformer(cid).GetPositions()).all()


def test_missing_receptor_and_multiple_conformers_fail_closed(pose, receptor):
    missing = runner.check_docked_pose(pose, str(receptor) + '.missing')
    assert missing['failures'] == ['receptor_file_missing']
    pose.AddConformer(Chem.Conformer(pose.GetConformer()), assignId=True)
    multiple = runner.check_docked_pose(pose, receptor)
    assert multiple['failures'] == ['exactly_one_input_conformer_required']
    assert multiple['pb_valid'] is False


def test_numpy_boolean_pass_rate_verdicts():
    assert runner.pb_pass_rate([{'pb_valid': np.bool_(True)},
        {'pb_valid': np.bool_(True), 'skipped': np.bool_(True)}]) == 0.5
