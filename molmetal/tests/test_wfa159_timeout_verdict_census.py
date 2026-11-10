# SPDX-License-Identifier: Apache-2.0
"""Wf-A159 — the TIMEOUT-verdict census + the six repaired gates.

THE RULE
--------
A wall-clock TIMEOUT means **the measurement was NOT obtained**.  It is
NOT evidence that the thing under test is broken.  Reporting a timeout as
a defect verdict is a **FALSE DEFECT CLAIM** (AGENTS.md §5.2 honest
framing: a genuine defect is reported as a defect, and only a genuine
defect may be reported as one).

``pre_flight_check._timeout_gate_result(...)`` is the repo's encoding of
the correct verdict: status ``TIMEOUT`` plus evidence that says the
measurement was not obtained and names WHAT the timeout is not a verdict
about.

WHAT THIS FILE GUARDS
---------------------
1. The census finds **ZERO** ``UNHANDLED`` sites in
   ``molmetal/scripts/pre_flight_check.py`` — i.e. every subprocess call
   site routes a wall-clock timeout into a TIMEOUT verdict.
2. The census is **NOT VACUOUS**: a synthetic file with an unhandled
   ``rc != 0 -> FAIL`` site is flagged ``UNHANDLED``.
3. For EACH of the six gates Wf-A159 repaired, a child that returns
   ``rc == 124`` yields status ``TIMEOUT`` — never ``FAIL`` — and the
   evidence carries the right ``not_a_verdict_on`` phrase.
4. A genuinely broken child (``rc == 1``) STILL yields ``FAIL`` for each
   of those six gates — the repair did not swallow real defects.
5. The no-CUDA import check: importing the pre-flight module and the
   census script imports no ``torch`` and spawns no subprocess.
6. The census re-derives the same answer from the file on disk, so the
   test and the shipped artefact cannot drift.
7. The gate COUNT is unchanged (``all`` = 129, ``launch`` = 131) and the
   Wf-A158 static invariant (no ``rc == 124`` branch may contain a
   ``FAIL`` literal) still holds.

CPU-ONLY: every subprocess in this file is monkeypatched or is a pure
``sys.executable -c`` probe with ``CUDA_VISIBLE_DEVICES=""``.  No GPU is
touched, no checkpoint is loaded, no ``torch`` is imported.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PREFLIGHT_REL = "molmetal/scripts/pre_flight_check.py"
CENSUS_REL = "molmetal/scripts/wfa159_timeout_verdict_census.py"
REPORT_REL = "molmetal/reports/wfa159_timeout_verdict_census/timeout_verdict_census.json"

#: ``WFA159_PREFLIGHT_PATH`` exists ONLY so the patched copy can be
#: pre-verified BEFORE it is atomically installed over the repo file
#: (AGENTS.md §5.6 — the real file is written once, at the end).  The
#: default is the repo file, which is what CI and the verifier use.
_ENV_PREFLIGHT = os.environ.get("WFA159_PREFLIGHT_PATH")
PREFLIGHT_PATH = (
    Path(_ENV_PREFLIGHT).resolve()
    if _ENV_PREFLIGHT
    else REPO_ROOT / PREFLIGHT_REL
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _load(module_name: str, path: Path):
    """Import a repo module by path (the scripts are not a package)."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


pre_flight = _load("wfa159_pre_flight", PREFLIGHT_PATH)
census = _load("wfa159_census", REPO_ROOT / CENSUS_REL)


# ---------------------------------------------------------------------------
# helpers — fake children
# ---------------------------------------------------------------------------
def _run_returning(rc: int, out: str = "", err: str = ""):
    """A stand-in for ``_run`` that always returns ``rc``."""

    def _run(cmd, **kwargs):
        return rc, out, err

    return _run


def _run_raising_timeout(*args, **kwargs):
    """A stand-in for ``subprocess.run`` that raises ``TimeoutExpired``."""
    raise subprocess.TimeoutExpired(cmd="synthetic", timeout=20.0)


class _FakeProc:
    """A stand-in for a ``CompletedProcess``."""

    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _subprocess_returning(returncode: int, stdout: str = "", stderr: str = ""):
    def _run(cmd, **kwargs):
        return _FakeProc(returncode, stdout, stderr)

    return _run


# ---------------------------------------------------------------------------
# 1 — the census finds ZERO UNHANDLED sites
# ---------------------------------------------------------------------------
def test_census_finds_zero_unhandled_sites() -> None:
    payload = census.run_census(PREFLIGHT_PATH)
    assert payload["status"] != "SKIP", payload
    s = payload["summary"]
    assert s["total_sites"] >= 30, (
        f"expected >= 30 subprocess call sites, found {s['total_sites']} — "
        f"the census may have stopped watching a call form"
    )
    assert s["unhandled"] == 0, (
        "the census found subprocess sites that would silently convert a "
        f"wall-clock timeout into something other than a TIMEOUT verdict: "
        f"{s['unhandled_sites']}"
    )
    assert s["timeout"] == s["total_sites"]


def test_census_is_not_vacuous(tmp_path) -> None:
    """A synthetic unhandled site MUST be flagged — else the guard is theatre."""
    bad = tmp_path / "synthetic_pre_flight.py"
    bad.write_text(
        "import subprocess\n"
        "\n"
        "def gate_synthetic():\n"
        "    rc, out, err = _run(['x'], timeout=1.0)\n"
        "    if rc != 0:\n"
        "        return 'FAIL'\n"
        "    return 'PASS'\n",
        encoding="utf-8",
    )
    payload = census.run_census(bad)
    assert payload["status"] == "FAIL", payload
    assert payload["summary"]["unhandled"] == 1, payload["summary"]
    assert payload["sites"][0]["verdict_for_124"] == "UNHANDLED"


def test_census_reports_the_documented_fields() -> None:
    payload = census.run_census(PREFLIGHT_PATH)
    for site in payload["sites"]:
        for key in (
            "file",
            "line",
            "enclosing_function",
            "declared_budget_sec",
            "has_124_branch",
            "verdict_for_124",
        ):
            assert key in site, f"census site is missing {key}: {site}"
        assert site["verdict_for_124"] in ("TIMEOUT", "UNHANDLED")
        assert site["has_124_branch"] is True, site


# ---------------------------------------------------------------------------
# 2 — the six repaired gates: rc == 124 -> TIMEOUT (never FAIL)
# ---------------------------------------------------------------------------
def test_gate_csv_hash_timeout_is_timeout_not_fail(tmp_path, monkeypatch) -> None:
    csv = tmp_path / "gated.csv"
    csv.write_text("smiles\nC\n", encoding="utf-8")
    monkeypatch.setattr(pre_flight, "_run", _run_returning(124, "", "[TIMEOUT]"))
    res = pre_flight.gate_csv_hash(
        "wfa159_csv_hash",
        "P0",
        "all",
        [(csv, 1)],
        manifest_path=tmp_path / "no_such_manifest.json",
    )
    assert res.status == "TIMEOUT", f"expected TIMEOUT, got {res.status}: {res.evidence}"
    assert res.status != "FAIL"
    assert "the CSV line count" in res.evidence, res.evidence
    assert "NOT a verdict" in res.evidence
    assert "was NOT obtained" in res.evidence


def test_gate_metalcytotox_v1_known_leaky_timeout_is_timeout_not_fail(
    tmp_path, monkeypatch
) -> None:
    csv = tmp_path / "metalcytotox.csv"
    csv.write_text("smiles\nC\n", encoding="utf-8")
    monkeypatch.setattr(pre_flight, "_run", _run_returning(124, "", "[TIMEOUT]"))
    res = pre_flight.gate_metalcytotox_v1_known_leaky(
        "wfa159_v1_leaky", "P2", "all", csv
    )
    assert res.status == "TIMEOUT", f"expected TIMEOUT, got {res.status}: {res.evidence}"
    assert res.status != "FAIL"
    assert "the v1 metalcytotox scaffold leak" in res.evidence, res.evidence


def test_gate_p1_runtime_energy_report_timeout_is_timeout_not_fail(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(pre_flight, "_run", _run_returning(124, "", "[TIMEOUT]"))
    res = pre_flight.gate_p1_runtime_energy_report(
        "wfa159_energy", "P1", "all", tmp_path / "energy.json"
    )
    assert res.status == "TIMEOUT", f"expected TIMEOUT, got {res.status}: {res.evidence}"
    assert res.status != "FAIL"
    assert "NVIDIA driver availability" in res.evidence, res.evidence
    assert "NOT a verdict" in res.evidence


def test_gate_p1_gpu_env_probe_timeout_is_timeout_not_fail(tmp_path, monkeypatch) -> None:
    probe = tmp_path / "check_blackwell_env.py"
    probe.write_text("# synthetic probe\n", encoding="utf-8")
    monkeypatch.setattr(pre_flight, "_run", _run_returning(124, "", "[TIMEOUT]"))
    res = pre_flight.gate_p1_gpu_env_probe(
        "wfa159_gpu_env",
        "P1",
        "all",
        tmp_path / "env.json",
        probe_script=probe,
    )
    assert res.status == "TIMEOUT", f"expected TIMEOUT, got {res.status}: {res.evidence}"
    assert res.status != "FAIL"
    assert "GPU environment probe script" in res.evidence, res.evidence


def test_gate_cfm_catherine_30metals_timeout_is_timeout_not_fail(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(pre_flight, "_run", _run_returning(124, "", "[TIMEOUT]"))
    res = pre_flight.gate_cfm_catherine_30metals_reproduced(
        "wfa159_catherine", "P1", "all", out_dir=tmp_path / "cath_out"
    )
    assert res.status == "TIMEOUT", f"expected TIMEOUT, got {res.status}: {res.evidence}"
    assert res.status != "FAIL"
    assert "catherine 30-metal producer" in res.evidence, res.evidence


def test_gate_wf319c_pb_packages_timeout_is_timeout_not_fail(tmp_path, monkeypatch) -> None:
    metric = tmp_path / "metric.json"
    metric.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pre_flight.subprocess, "run", _run_raising_timeout)
    res = pre_flight.gate_wf319c_pb_packages_status_present(
        "wfa159_pb",
        "P1",
        "all",
        venv_python="/nonexistent/python",
        metric_json=metric,
        smoke_output=tmp_path / "smoke.json",
    )
    assert res.status == "TIMEOUT", f"expected TIMEOUT, got {res.status}: {res.evidence}"
    assert res.status != "FAIL"
    assert "rdkit/posebusters package availability" in res.evidence, res.evidence
    assert "NOT a verdict" in res.evidence


# ---------------------------------------------------------------------------
# 3 — a genuinely broken child STILL fails (the repair swallowed nothing)
# ---------------------------------------------------------------------------
def test_gate_csv_hash_broken_child_still_fails(tmp_path, monkeypatch) -> None:
    csv = tmp_path / "gated.csv"
    csv.write_text("smiles\nC\n", encoding="utf-8")
    monkeypatch.setattr(pre_flight, "_run", _run_returning(1, "", "boom"))
    res = pre_flight.gate_csv_hash(
        "wfa159_csv_hash",
        "P0",
        "all",
        [(csv, 1)],
        manifest_path=tmp_path / "no_such_manifest.json",
    )
    assert res.status == "FAIL", f"expected FAIL, got {res.status}: {res.evidence}"


def test_gate_metalcytotox_v1_known_leaky_broken_child_still_fails(
    tmp_path, monkeypatch
) -> None:
    csv = tmp_path / "metalcytotox.csv"
    csv.write_text("smiles\nC\n", encoding="utf-8")
    monkeypatch.setattr(pre_flight, "_run", _run_returning(1, "", "boom"))
    res = pre_flight.gate_metalcytotox_v1_known_leaky(
        "wfa159_v1_leaky", "P2", "all", csv
    )
    assert res.status == "FAIL", f"expected FAIL, got {res.status}: {res.evidence}"


def test_gate_p1_runtime_energy_report_broken_child_still_fails(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(pre_flight, "_run", _run_returning(1, "", "boom"))
    res = pre_flight.gate_p1_runtime_energy_report(
        "wfa159_energy", "P1", "all", tmp_path / "energy.json"
    )
    assert res.status == "FAIL", f"expected FAIL, got {res.status}: {res.evidence}"


def test_gate_p1_gpu_env_probe_broken_child_still_fails(tmp_path, monkeypatch) -> None:
    probe = tmp_path / "check_blackwell_env.py"
    probe.write_text("# synthetic probe\n", encoding="utf-8")
    monkeypatch.setattr(pre_flight, "_run", _run_returning(1, "", "boom"))
    res = pre_flight.gate_p1_gpu_env_probe(
        "wfa159_gpu_env",
        "P1",
        "all",
        tmp_path / "env.json",
        probe_script=probe,
    )
    assert res.status == "FAIL", f"expected FAIL, got {res.status}: {res.evidence}"


def test_gate_cfm_catherine_30metals_broken_child_still_fails(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(pre_flight, "_run", _run_returning(1, "", "boom"))
    res = pre_flight.gate_cfm_catherine_30metals_reproduced(
        "wfa159_catherine", "P1", "all", out_dir=tmp_path / "cath_out"
    )
    assert res.status == "FAIL", f"expected FAIL, got {res.status}: {res.evidence}"


def test_gate_wf319c_pb_packages_broken_child_still_fails(tmp_path, monkeypatch) -> None:
    metric = tmp_path / "metric.json"
    metric.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        pre_flight.subprocess, "run", _subprocess_returning(1, "", "boom")
    )
    res = pre_flight.gate_wf319c_pb_packages_status_present(
        "wfa159_pb",
        "P1",
        "all",
        venv_python="/nonexistent/python",
        metric_json=metric,
        smoke_output=tmp_path / "smoke.json",
    )
    assert res.status == "FAIL", f"expected FAIL, got {res.status}: {res.evidence}"
    assert "TimeoutExpired" not in res.evidence, (
        "a genuine exec error must not be reported as a timeout"
    )


# ---------------------------------------------------------------------------
# 4 — the no-CUDA import check
# ---------------------------------------------------------------------------
def test_importing_the_modules_touches_no_torch(tmp_path) -> None:
    probe = tmp_path / "probe_import.py"
    probe.write_text(
        "import importlib.util, sys\n"
        "for name, path in (\n"
        f"    ('wfa159_pf', {str(PREFLIGHT_PATH)!r}),\n"
        f"    ('wfa159_census', {str(REPO_ROOT / CENSUS_REL)!r}),\n"
        "):\n"
        "    spec = importlib.util.spec_from_file_location(name, path)\n"
        "    mod = importlib.util.module_from_spec(spec)\n"
        "    sys.modules[name] = mod\n"
        "    spec.loader.exec_module(mod)\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
        "assert 'cupy' not in sys.modules, 'cupy was imported'\n"
        "print('NO_TORCH')\n",
        encoding="utf-8",
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(
        [sys.executable, str(probe)],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    assert proc.returncode == 0, (
        f"the CPU-only import probe failed (rc={proc.returncode}): {proc.stderr[-2000:]}"
    )
    assert "NO_TORCH" in proc.stdout


def test_the_census_script_spawns_nothing_and_imports_no_torch() -> None:
    src = (REPO_ROOT / CENSUS_REL).read_text(encoding="utf-8")
    tree = ast.parse(src)
    assert "import torch" not in src
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fname = None
            f = node.func
            if isinstance(f, ast.Name):
                fname = f.id
            elif isinstance(f, ast.Attribute):
                fname = f.attr
            assert fname not in ("run", "Popen", "check_output", "check_call"), (
                f"the census must not spawn a subprocess (found {fname!r} at "
                f"line {node.lineno})"
            )


# ---------------------------------------------------------------------------
# 5 — the census re-derives the same answer from the file on disk
# ---------------------------------------------------------------------------
def test_census_artifact_matches_the_file_on_disk() -> None:
    artifact_path = REPO_ROOT / REPORT_REL
    assert artifact_path.is_file(), (
        f"the census artefact is missing on disk: {artifact_path} — "
        f"run `python {CENSUS_REL}`"
    )
    on_disk = json.loads(artifact_path.read_text(encoding="utf-8"))
    fresh = census.run_census(PREFLIGHT_PATH)
    assert on_disk["target_sha256"] == fresh["target_sha256"], (
        "the census artefact was produced from a DIFFERENT version of "
        f"{PREFLIGHT_REL} — re-run the census"
    )
    assert on_disk["summary"]["unhandled"] == 0
    assert fresh["summary"]["unhandled"] == 0
    key = lambda s: (s["line"], s["enclosing_function"], s["verdict_for_124"])
    assert [key(s) for s in fresh["sites"]] == [key(s) for s in on_disk["sites"]]


# ---------------------------------------------------------------------------
# 6 — the gate count and the Wf-A158 static invariant
# ---------------------------------------------------------------------------
def test_gate_count_is_unchanged() -> None:
    # 2026-11-04 (Wf-A269 wave-5/6) — re-synced, dated; the superseded
    # values (129 / 131) stay visible here, per AGENTS.md §5.5 rule #6.
    # MEASURED 2026-11-04 by live import.
    # 2026-11-04 (Wf-A269 wave-22) — re-synced, dated; the superseded
    # values (132 / 134) stay visible here, per AGENTS.md §5.5 rule #6.
    # WHY: wave-19 registered `p3_script_contract_check` and moved the live
    # counts to 133 / 135, but did NOT re-sync this pin — so this test was
    # RED for a reason unrelated to what it guards (the same class as the
    # Wf-350-C1 pin in test_wf350c1_gpu_env_probe.py).  MEASURED
    # 2026-11-04 by live import; the stale value was reported by the
    # Wf-A169 gate-count-pin census (`molmetal/scripts/
    # wfa169_gate_count_pins.py`), whose own control test was RED because
    # of it.
    assert len(pre_flight.build_gates("all")) == 133
    assert len(pre_flight.build_gates("launch")) == 135


def _iter_rc124_blocks(text: str):
    """Yield the source block of every ``if rc == 124:`` branch."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not re.search(r"\brc\s*==\s*124\b", line):
            continue
        if not line.lstrip().startswith("if "):
            continue
        indent = len(line) - len(line.lstrip())
        body: List[str] = []
        for j in range(i + 1, len(lines)):
            nxt = lines[j]
            if not nxt.strip():
                body.append(nxt)
                continue
            nxt_indent = len(nxt) - len(nxt.lstrip())
            if nxt_indent <= indent:
                break
            body.append(nxt)
        yield i + 1, "\n".join(body)


def test_every_rc124_branch_routes_to_a_timeout_verdict() -> None:
    """The Wf-A158 static invariant, re-asserted over the repaired file."""
    text = PREFLIGHT_PATH.read_text(encoding="utf-8")
    blocks = list(_iter_rc124_blocks(text))
    assert len(blocks) >= 20, (
        f"expected >= 20 'rc == 124' branches, found {len(blocks)}"
    )
    for lineno, body in blocks:
        assert '"FAIL"' not in body and "'FAIL'" not in body, (
            f"{PREFLIGHT_REL}:{lineno} — an rc==124 branch contains a FAIL "
            f"status; a wall-clock timeout is NOT a defect verdict:\n{body}"
        )
        assert (
            "_timeout_gate_result(" in body or "GATE_STATUS_TIMEOUT" in body
        ), (
            f"{PREFLIGHT_REL}:{lineno} — an rc==124 branch does not route to a "
            f"TIMEOUT verdict:\n{body}"
        )
