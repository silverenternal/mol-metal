"""The isolated bridge batches real reports and rejects unavailable workers."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from molmetal.molmetal_lam.sbdd_env import aizynth_isolated as isolated
from molmetal.molmetal_lam.sbdd_env.aizynth_isolated import IsolatedAiZynthChecker
from molmetal.molmetal_lam.sbdd_env.synthesis_gate import gate_candidates


def test_batch_uses_one_worker_and_rejects_smarts(monkeypatch, tmp_path):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        req = json.loads(kwargs["input"])
        rows = [dict(smiles=s, synthesizable=True, depth=1, route_smiles=[],
                     engine="aizynthfinder" if i == 0 else "smarts_fallback")
                for i, s in enumerate(req["smiles"])]
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps(
            {"schema_version": 1, "ready": True, "metadata": {}, "reports": rows}))

    monkeypatch.setattr(isolated, "_run_process_group", run)
    checker = IsolatedAiZynthChecker(tmp_path / "a config.yml")
    candidates = [SimpleNamespace(canonical_smiles=lambda s=s: s) for s in ["CCO", "CCN"]]
    kept, report = gate_candidates(candidates, checker, {"learned": True})
    assert kept == candidates[:1]
    assert report["reports"][1]["status"] == "backend_fallback_rejected"
    assert len(calls) == 1
    assert calls[0][0][0] == "bash"
    assert calls[0][1]["timeout"] > 2 * 30
    assert "shell" not in calls[0][1]


def test_unready_worker_rejects_probe(monkeypatch, tmp_path):
    response = {"schema_version": 1, "ready": False,
                "metadata": {"load_error": "missing actual model"}, "reports": []}
    monkeypatch.setattr(isolated, "_run_process_group", lambda *a, **k: SimpleNamespace(
        returncode=2, stdout=json.dumps(response), stderr=""))
    with pytest.raises(RuntimeError, match="missing actual model"):
        IsolatedAiZynthChecker(tmp_path / "config.yml").probe()


def test_malformed_worker_output_rejects_batch(monkeypatch, tmp_path):
    monkeypatch.setattr(isolated, "_run_process_group", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout="not json", stderr="broken"))
    with pytest.raises(RuntimeError, match="no valid JSON"):
        IsolatedAiZynthChecker(tmp_path / "config.yml").check_many(["CCO"])


def test_timeout_terminates_real_descendant_process(tmp_path):
    marker = tmp_path / "child_pid"
    source = (
        "import subprocess,sys,time; from pathlib import Path; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(60)"
    )
    with pytest.raises(TimeoutError, match="process group was terminated"):
        isolated._run_process_group([sys.executable, "-c", source, str(marker)], input="{}", timeout=0.5)
    assert marker.exists(), "Test launcher did not start its child before deadline"
    pid = int(marker.read_text())
    status = Path(f"/proc/{pid}/status")
    # A terminated orphan can briefly remain a zombie until PID 1 reaps it.
    assert not status.exists() or "State:\tZ" in status.read_text()
