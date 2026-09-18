"""Behavioral transport checks; fake workers never claim real model validation."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

import pytest

from molmetal.molmetal_lam.sbdd_env.reinvent_prior_adapter import REINVENT4PriorAdapter

SHA = "b" * 64


def adapter(tmp_path, behavior="", timeout=2.0, startup=""):
    script = tmp_path / "fake_jsonl_worker.py"
    script.write_text(f'''
import json,sys,time,subprocess,os
metadata={{"protocol":"reinvent-prior-jsonl-v1","backend":"reinvent-prior","model_sha256":{SHA!r},"device":"cpu"}}
{startup}
for line in sys.stdin:
 request=json.loads(line)
 if request["op"]=="capabilities":
  response={{"metadata":metadata}}
 else:
  {behavior or 'pass'}
  rows=[]
  for i,s in enumerate(request["smiles"]):
   rows.append({{"index":i,"smiles":s,"status":"ok" if s else "error","prior_nll":17.75 if s else None,"predicted_token_count":4 if s else None,"error":None if s else "empty_input"}})
  meta=dict(metadata,n_requested=len(rows),n_scored=sum(r["status"]=="ok" for r in rows),neural_forwards=[{{"parameter_device":"cpu","input_device":"cpu","count":1}}])
  response={{"metadata":meta,"results":rows}}
 response["id"]=request["id"]
 print(json.dumps(response),flush=True)
''')
    return REINVENT4PriorAdapter(worker_python=sys.executable, worker_script=str(script),
        prior_path="unused-by-fake", prior_sha256=SHA, device="cpu", timeout=timeout)


def test_raw_nll_order_duplicates_and_invalid_slots_preserved(tmp_path):
    with adapter(tmp_path) as client:
        rows = client.likelihood(["CCO", "", "CCO", None])
        assert [r.index for r in rows] == [0, 1, 2, 3]
        assert [r.smiles for r in rows] == ["CCO", "", "CCO", None]
        assert rows[0].prior_nll == rows[2].prior_nll == 17.75  # no [0,1] clipping
        assert rows[1].status == rows[3].status == "error"
        assert all(r.model_sha256 == SHA and r.device == "cpu" for r in rows)
        pid = client._proc.pid
        assert client.likelihood(["CCO"])[0].status == "ok"
        assert client._proc.pid == pid  # model process reused
    assert client._proc is None


def test_empty_batch_does_not_start_model(tmp_path):
    with adapter(tmp_path) as client:
        assert client.likelihood([]) == []
        assert client._proc is None


def test_wrong_backend_is_rejected_before_scoring(tmp_path):
    with adapter(tmp_path, startup='metadata["backend"]="rdkit-proxy"') as client:
        rows = client.likelihood(["CCO", "CCC"])
        assert len(rows) == 2 and all(r.status == "error" for r in rows)
        assert "backend_model_or_device_mismatch" in client.last_error
        assert client._proc is None


def test_worker_exit_retains_denominator(tmp_path):
    with adapter(tmp_path, behavior="sys.exit(7)") as client:
        rows = client.likelihood(["CCO", "CCC", "CO"])
        assert len(rows) == 3 and all(r.status == "error" for r in rows)
        assert "closed_stdout" in client.last_error


def test_malformed_response_reaps_worker(tmp_path):
    with adapter(tmp_path, behavior='print("not json",flush=True); time.sleep(30)') as client:
        rows = client.likelihood(["CCO"])
        assert rows[0].status == "error"
        assert "JSONDecodeError" in client.last_error
        assert client._proc is None


def test_one_deadline_includes_startup_and_inference(tmp_path):
    with adapter(tmp_path, startup="time.sleep(0.18)", behavior="time.sleep(0.18)", timeout=0.28) as client:
        started = time.monotonic()
        rows = client.likelihood(["CCO", "CCN"])
        elapsed = time.monotonic() - started
        assert len(rows) == 2 and all(r.status == "error" for r in rows)
        assert "total_request_timeout" in client.last_error
        assert elapsed < 0.9
        assert client._proc is None


def test_partial_line_timeout_kills_process_group_descendant(tmp_path):
    pidfile = tmp_path / "child.pid"
    behavior = (f'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); '
                f'open({str(pidfile)!r},"w").write(str(child.pid)); '
                'sys.stdout.write("{partial"); sys.stdout.flush(); time.sleep(30)')
    with adapter(tmp_path, behavior=behavior, timeout=0.5) as client:
        rows = client.likelihood(["CCO"])
        assert rows[0].status == "error" and "total_request_timeout" in client.last_error
        assert pidfile.exists()
        child_pid = int(pidfile.read_text())
        stat = Path(f"/proc/{child_pid}/stat")
        deadline = time.monotonic() + 0.5
        while stat.exists() and stat.read_text().split()[2] != "Z" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not stat.exists() or stat.read_text().split()[2] == "Z"


def test_explicit_mode_required_in_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"mode": "multiproperty"}))
    with pytest.raises(ValueError, match="prior_nll"):
        REINVENT4PriorAdapter.from_config(path)
