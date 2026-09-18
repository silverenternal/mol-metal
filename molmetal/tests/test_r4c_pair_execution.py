"""Input integrity, seeded jobs, honest outputs and restart behavior."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from molmetal.scripts import r4_c_full_sweep as sweep
from molmetal.scripts.stage_crossdocked100 import cached_test_pairs
from molmetal.molmetal_lam.sbdd_env.vina_adapter import _parse_vina_result_energies


def manifest_fixture(tmp_path, ids=("heldout_b", "heldout_a")):
    rows = []
    for pid in ids:
        rec, lig = tmp_path / f"{pid}.pdb", tmp_path / f"{pid}.sdf"
        rec.write_text("ATOM\n")
        lig.write_text("fixture\n")
        rows.append(dict(pocket_id=pid, receptor_path=str(rec), ligand_path=str(lig)))
    manifest = tmp_path / "pairs.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest, rows


def test_staging_resolves_extraction_root_and_checks_every_pair(tmp_path):
    root = tmp_path / "crossdocked"
    extracted = root / "extracted" / "crossdocked_pocket10"
    extracted.mkdir(parents=True)
    pairs = []
    for i in range(100):
        rec, lig = f"rec_{i}.pdb", f"lig_{i}.sdf"
        (extracted / rec).touch()
        (extracted / lig).touch()
        pairs.append((rec, lig))
    torch.save({"test": pairs, "train": []}, root / "split_by_name.pt")
    staged = cached_test_pairs(tmp_path)
    assert len(staged) == 100
    assert all(Path(row[k]).is_file() for row in staged for k in ("receptor_path", "ligand_path"))
    (extracted / pairs[-1][1]).unlink()
    assert cached_test_pairs(tmp_path) is None


def test_pair_selection_preserves_test_split_order_and_exact_ligand(tmp_path):
    manifest, rows = manifest_fixture(tmp_path)
    (tmp_path / "000_training_directory").mkdir()
    assert sweep.load_manifest(str(manifest), 0, 1) == rows[:1]
    assert sweep.load_manifest(str(manifest), 1, 1) == rows[1:]
    Path(rows[1]["ligand_path"]).unlink()
    with pytest.raises(FileNotFoundError):
        sweep.load_manifest(str(manifest), 0, 2)


def test_wrapper_passes_effective_settings_and_preserves_backend(monkeypatch):
    received = {}
    def run(pocket_dir, **kwargs):
        received.update(kwargs)
        return {"status": "ok", "candidates": [{"smiles": "CCO", "sa": 1.98, "qed": .4, "lipinski": True}],
                "search_config": {"top_k": kwargs["top_k"], "seed": kwargs["seed"]},
                "metric_backend": {"sa": "ertl"}, "seed_smiles": "CCO",
                "prior_state": kwargs["prior_state"], "synthesis_report": {"n_checked": 1}}
    monkeypatch.setattr(sweep, "load_runner", lambda: SimpleNamespace(run_one_pocket=run))
    result = sweep.run_one_pocket("root", 17, 4, 60, seed=1234, top_k=27,
                                  early_stop=False, patience=2, ligand_path="exact.sdf",
                                  symbolic_prior=True, synthesis_oracle="aizynthfinder_isolated",
                                  prior_state={"frozen": True}, synthesis_config_path="real.yaml")
    assert received["seed"] == 1234 and received["seed_offset"] == 0
    assert received["top_k"] == 27 and received["branching_target"] == 60
    assert received["early_stop"] is False and received["patience"] == 2
    assert received["ligand_path"] == "exact.sdf"
    assert received["symbolic_prior"] and received["synthesis_oracle"]
    assert received["prior_state"] == {"frozen": True}
    assert received["prior_mode"] == "frozen" and received["prior_data_split"] == "test"
    assert received["synthesis_config_path"] == "real.yaml"
    assert result.diagnostics["prior_state"] == {"frozen": True}
    assert result.diagnostics["synthesis"] == {"n_checked": 1}
    assert result.metric_backend == {"sa": "ertl"}
    assert result.mean_sa == 1.98 and result.diagnostics["seed_smiles"] == "CCO"


def test_multiseed_checkpoint_resumes_without_rerunning_or_losing_metrics(tmp_path, monkeypatch):
    manifest, rows = manifest_fixture(tmp_path)
    prefix = tmp_path / "run"
    argv = ["r4", "--manifest", str(manifest), "--n-pockets", "2", "--seeds", "42", "0", "1234",
            "--output-prefix", str(prefix), "--n-simulations", "2", "--branching-target", "60"]
    monkeypatch.setattr("sys.argv", argv)
    calls = []
    def execute(job, search, timeout, log_path):
        calls.append((job["pocket_id"], job["seed"]))
        if len(calls) == 2:
            raise KeyboardInterrupt()
        return sweep.PocketResult(pocket_id=job["pocket_id"], seed=job["seed"], n_candidates=1,
                                  top1_smiles="CCO", mean_sa=2., mean_qed=.4, lipinski_pass_count=1,
                                  candidates=[{"smiles": "CCO", "sa": 2.}], search_config=search)
    monkeypatch.setattr(sweep, "execute_job", execute)
    with pytest.raises(KeyboardInterrupt):
        sweep.main()
    saved = json.loads(prefix.with_suffix(".json").read_text())
    assert len(saved["per_pocket"]) == 1
    first = saved["per_pocket"][0]
    monkeypatch.setattr("sys.argv", argv + ["--append"])
    assert sweep.main() == 0
    final = json.loads(prefix.with_suffix(".json").read_text())
    assert final["per_pocket"][0] == first
    assert len(final["per_pocket"]) == 6
    assert calls.count(("heldout_b", 42)) == 1
    assert final["summary"]["sa_mean"] == 2.
    ncalls = len(calls)
    assert sweep.main() == 0 and len(calls) == ncalls
    Path(rows[0]["ligand_path"]).write_text("changed input")
    assert sweep.main() == 2 and len(calls) == ncalls
    assert prefix.with_suffix(".json").read_text() == json.dumps(final, indent=2) + "\n"


def test_empty_job_is_failure_and_report_has_no_fake_energy_or_success(tmp_path, monkeypatch):
    manifest, _ = manifest_fixture(tmp_path, ("empty",))
    prefix = tmp_path / "empty_run"
    monkeypatch.setattr("sys.argv", ["r4", "--manifest", str(manifest), "--n-pockets", "1",
                                   "--seeds", "42", "--output-prefix", str(prefix)])
    monkeypatch.setattr(sweep, "execute_job", lambda job, *args: sweep.PocketResult(
        pocket_id=job["pocket_id"], seed=42, status="no_candidates"))
    assert sweep.main() == 1
    payload = json.loads(prefix.with_suffix(".json").read_text())
    assert payload["summary"]["lipinski_pass_rate"] is None
    assert payload["summary"]["n_pockets_fail"] == 1
    report = prefix.with_suffix(".md").read_text()
    assert "no kcal/mol interpretation" in report and "same protocol" not in report
    assert "MEASURED" not in report


def test_real_quickvina_remark_scores_are_parsed_without_rmsd_as_energy():
    text = "MODEL 1\nREMARK VINA RESULT: -2.6 0.0 0.0\nENDMDL\nMODEL 2\n  REMARK VINA RESULT: -1.5 2.1 2.8\nENDMDL"
    energies = _parse_vina_result_energies(text)
    np.testing.assert_allclose(energies[:, 0], [-2.6, -1.5])
    assert energies.shape == (2, 5) and np.isnan(energies[:, 1:]).all()
    assert _parse_vina_result_energies("REMARK VINA RESULT: nan 0 0\nVINA RESULT:\n").shape == (0, 5)
    assert _parse_vina_result_energies("VINA RESULT: -4 0 0")[0, 0] == -4


def test_staging_failure_and_dryrun_preserve_existing_manifest(tmp_path, monkeypatch):
    from molmetal.scripts import stage_crossdocked100 as stage
    output = tmp_path / 'existing.csv'
    output.write_text('previous valid data')
    monkeypatch.setattr('sys.argv', ['stage', '--data-root', str(tmp_path), '--out', str(output)])
    assert stage.main() == 1
    assert output.read_text() == 'previous valid data'
    monkeypatch.setattr(stage, 'cached_test_pairs', lambda root: [{'pocket_id': 'valid'}] * 100)
    monkeypatch.setattr('sys.argv', ['stage', '--out', str(output), '--dry-run'])
    assert stage.main() == 0
    assert output.read_text() == 'previous valid data'


def test_timeout_terminates_worker_and_records_failure(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    manifest, rows = manifest_fixture(tmp_path, ('timed',))
    real_popen = subprocess.Popen
    processes = []
    def slow_worker(command, **kwargs):
        proc = real_popen([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs)
        processes.append(proc)
        return proc
    monkeypatch.setattr(sweep.subprocess, 'Popen', slow_worker)
    result = sweep.execute_job({**rows[0], 'seed': 42}, {'n_simulations': 1},
                               .1, tmp_path / 'worker.log')
    assert result.status.startswith('timeout:') and result.n_candidates == 0
    assert processes[0].poll() is not None
    with pytest.raises(ProcessLookupError):
        os.kill(processes[0].pid, 0)


def test_runtime_change_stops_before_mixing_jobs(tmp_path, monkeypatch):
    manifest, _ = manifest_fixture(tmp_path)
    prefix = tmp_path / 'runtime_change'
    monkeypatch.setattr('sys.argv', ['r4', '--manifest', str(manifest), '--n-pockets', '2',
                                   '--seeds', '42', '--output-prefix', str(prefix)])
    runtime = {'algorithm.py': 'original'}
    monkeypatch.setattr(sweep, 'runtime_fingerprints', lambda: dict(runtime))
    calls = []
    def execute(job, search, timeout, log_path):
        calls.append(job['pocket_id'])
        runtime['algorithm.py'] = 'modified'
        return sweep.PocketResult(pocket_id=job['pocket_id'], seed=42)
    monkeypatch.setattr(sweep, 'execute_job', execute)
    assert sweep.main() == 2
    saved = json.loads(prefix.with_suffix('.json').read_text())
    assert len(calls) == len(saved['per_pocket']) == 1
    assert saved['metadata']['runtime_sha256'] == {'algorithm.py': 'original'}


def test_physical_timeout_retains_search_and_finished_pose_records(tmp_path, monkeypatch):
    worker = tmp_path / 'partial_worker.py'
    result = sweep.PocketResult(pocket_id='exact', seed=42, n_candidates=1,
                               n_generated_candidates=1, candidates=[{'smiles':'CCO','is_generated':True}],
                               physical={'candidates':[{'smiles':'CCO','score_kcal_mol':-3.}]})
    worker.write_text('import sys,json,time\nfrom pathlib import Path\n'
                      f'payload={sweep.asdict(result)!r}\n'
                      'Path(sys.argv[sys.argv.index("--worker-output")+1]).write_text(json.dumps(payload))\n'
                      'time.sleep(10)\n')
    monkeypatch.setattr(sweep, '__file__', str(worker))
    actual = sweep.execute_job({'pocket_id':'exact','seed':42,'receptor_path':'r.pdb','ligand_path':'l.sdf'},
                               {'n_simulations':1,'max_depth':1}, .4, str(tmp_path/'log'))
    assert actual.n_generated_candidates == 1
    assert actual.physical['status'] == 'interrupted'
    assert actual.physical['candidates'][0]['score_kcal_mol'] == -3.
    assert 'timeout' in actual.physical['error']


def test_cli_frozen_prior_and_synthesis_config_survive_jobs_and_checked_resume(tmp_path, monkeypatch):
    manifest, _ = manifest_fixture(tmp_path, ("heldout",))
    state = {"schema_version": 1, "backend": "linear_descriptor_prior",
             "features": ["n_atoms", "n_bonds", "n_free_sites", "n_heavy"],
             "source_splits": ["development"], "observations": [],
             "model": {"weights": [0.1, 0.2, 0.3, 0.4], "intercept": 0.5}}
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps(state))
    synthesis = tmp_path / "synthesis.yaml"
    synthesis.write_text("expansion: {}\n")
    prefix = tmp_path / "guided"
    argv = ["r4", "--manifest", str(manifest), "--n-pockets", "1", "--seeds", "42",
            "--output-prefix", str(prefix), "--prior-state", str(prior),
            "--synthesis-oracle", "aizynthfinder_isolated", "--synthesis-config", str(synthesis)]
    monkeypatch.setattr("sys.argv", argv)
    calls = []
    def execute(job, search, timeout, log_path):
        calls.append(search)
        return sweep.PocketResult(pocket_id=job["pocket_id"], seed=42)
    monkeypatch.setattr(sweep, "execute_job", execute)
    assert sweep.main() == 0
    assert calls[0]["prior_state"] == state
    assert calls[0]["prior_mode"] == "frozen" and calls[0]["prior_data_split"] == "test"
    assert calls[0]["synthesis_config_path"] == str(synthesis)
    payload = json.loads(prefix.with_suffix(".json").read_text())
    assert payload["metadata"]["guidance_inputs"]["prior_state"]["sha256"] == sweep.file_digest(str(prior))
    monkeypatch.setattr("sys.argv", argv + ["--append"])
    assert sweep.main() == 0 and len(calls) == 1
    synthesis.write_text("expansion: {changed: true}\n")
    assert sweep.main() == 2 and len(calls) == 1
    assert json.loads(prefix.with_suffix(".json").read_text()) == payload
    state["source_splits"] = ["test"]
    prior.write_text(json.dumps(state))
    monkeypatch.setattr("sys.argv", argv)
    assert sweep.main() == 2 and len(calls) == 1


def test_guidance_config_change_stops_remaining_jobs(tmp_path, monkeypatch):
    manifest, _ = manifest_fixture(tmp_path)
    synthesis = tmp_path / "synthesis.yaml"
    synthesis.write_text("expansion: {}\n")
    prefix = tmp_path / "changed_guidance"
    monkeypatch.setattr("sys.argv", ["r4", "--manifest", str(manifest), "--n-pockets", "2",
                                   "--seeds", "42", "--output-prefix", str(prefix),
                                   "--synthesis-config", str(synthesis)])
    calls = []
    def execute(job, search, timeout, log_path):
        calls.append(job)
        synthesis.write_text("expansion: {new_model: true}\n")
        return sweep.PocketResult(pocket_id=job["pocket_id"], seed=42)
    monkeypatch.setattr(sweep, "execute_job", execute)
    assert sweep.main() == 2 and len(calls) == 1
    assert len(json.loads(prefix.with_suffix(".json").read_text())["per_pocket"]) == 1
