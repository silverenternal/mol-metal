"""Budget/cache/failure behavior against a mock docking port, not fake benchmarks."""
from pathlib import Path
from types import SimpleNamespace
import json

import pytest
from rdkit import Chem

from molmetal.molmetal_lam.sbdd_env.pocket_docking_reward import PocketDockingReward, PocketDockingRewardError


class FakeDocking:
    def __init__(self, *, failure=False, trace=True, wrong_pose=False):
        self.calls = []
        self.last_pose_mols = []
        self.last_docking_seed = {"requested": 42, "effective": 42}
        self.last_run_metadata = {}
        self.failure, self.trace, self.wrong_pose = failure, trace, wrong_pose

    def dock(self, molecule, pocket, config):
        self.calls.append(molecule.smiles)
        self.last_run_metadata = {"gpu_verified": True,
                                  "gpu_evidence": "kernel_trace" if self.trace else "engine_device_log"}
        if self.failure:
            self.last_run_metadata.update(status="failed", error="synthetic port failure")
            raise RuntimeError("synthetic port failure")
        mol = Chem.MolFromSmiles("CC" if self.wrong_pose else molecule.smiles)
        conformer = Chem.Conformer(mol.GetNumAtoms())
        conformer.Set3D(True)
        for i in range(mol.GetNumAtoms()):
            conformer.SetAtomPosition(i, (float(i), 0, 0))
        mol.AddConformer(conformer)
        self.last_pose_mols = [mol]
        return [SimpleNamespace(vina_score=-3.25)]


def provider(tmp_path, **fake_options):
    receptor, ligand = tmp_path / "source.pdb", tmp_path / "geometry.sdf"
    receptor.write_text("test source")
    ligand.write_text("test geometry")
    reward = PocketDockingReward(receptor, ligand, tmp_path / "output", seed=42,
                                  gpu_config={"mock": True}, max_evaluations=1)
    engine = FakeDocking(**fake_options)
    # Isolate the cache/budget/pose contract from optional preparation tools.
    reward._ensure_ready = lambda: None
    reward._adapter, reward._pocket, reward._docking_config = engine, object(), object()
    return reward, engine


def test_canonical_smiles_cache_and_state_callback_keep_raw_energy(tmp_path):
    reward, engine = provider(tmp_path)
    assert reward("CCO") == -3.25
    state = SimpleNamespace(canonical_smiles=lambda: "OCC")
    assert reward(state) == -3.25
    assert len(engine.calls) == 1
    assert Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(engine.calls[0]))) == "CCO"
    report = reward.report()
    assert (report["n_attempted"], report["n_docked"], report["n_cache_hits"]) == (1, 1, 1)
    candidate = report["candidates"][0]
    assert Path(candidate["pose_sdf"]).is_file()
    assert candidate["poses"][0]["score_kcal_mol"] == -3.25
    saved = json.loads((tmp_path / "output/search_docking_reward.json").read_text())
    assert len(saved["calls"]) == 2


def test_exhausted_budget_raises_and_cached_energy_remains_available(tmp_path):
    reward, engine = provider(tmp_path)
    reward("CCO")
    with pytest.raises(PocketDockingRewardError, match="budget_exhausted"):
        reward("CCC")
    assert reward("OCC") == -3.25
    report = reward.report()
    assert report["n_attempted"] == 1 and report["n_budget_exhausted"] == 1
    assert report["calls"][1]["score_kcal_mol"] is None
    assert len(engine.calls) == 1


def test_failed_docking_consumes_budget_and_caches_failure_without_energy(tmp_path):
    reward, engine = provider(tmp_path, failure=True)
    with pytest.raises(PocketDockingRewardError, match="synthetic port failure"):
        reward("CCO")
    with pytest.raises(PocketDockingRewardError, match="cached_failure"):
        reward("OCC")
    report = reward.report()
    assert (report["n_attempted"], report["n_docked"], report["n_cache_hits"]) == (1, 0, 1)
    assert report["candidates"][0]["score_kcal_mol"] is None
    assert report["candidates"][0]["gpu_execution"]["status"] == "failed"
    assert len(engine.calls) == 1


def test_invalid_smiles_retained_without_consuming_docking_budget(tmp_path):
    reward, engine = provider(tmp_path)
    for value in ("", "invalid", None):
        with pytest.raises(PocketDockingRewardError):
            reward(value)
    report = reward.report()
    assert report["n_attempted"] == 0 and len(report["calls"]) == 3
    assert report["calls"][1]["input_smiles"] == "invalid"
    assert not engine.calls


@pytest.mark.parametrize("option", ["trace", "wrong_pose"])
def test_finite_score_without_trace_or_matching_pose_is_rejected(tmp_path, option):
    reward, _ = provider(tmp_path, **({"trace": False} if option == "trace" else {"wrong_pose": True}))
    with pytest.raises(PocketDockingRewardError):
        reward("CCO")
    report = reward.report()
    assert report["n_attempted"] == 1 and report["n_docked"] == 0
    assert report["candidates"][0]["score_kcal_mol"] is None


def test_source_mutation_prevents_cached_score_reuse(tmp_path):
    reward, engine = provider(tmp_path)
    reward("CCO")
    reward.receptor_path.write_text("changed source")
    with pytest.raises(PocketDockingRewardError, match="source_files_changed"):
        reward("OCC")
    assert len(engine.calls) == 1
    assert reward.report()["calls"][-1]["score_kcal_mol"] is None
