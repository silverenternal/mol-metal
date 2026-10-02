"""Wf-415-A2 — pre_flight_check.py sys.path injection regression tests.

Closes the ModuleNotFoundError: No module named 'molmetal' regression
that surfaces when pre_flight_check.py is invoked via:

    python molmetal/scripts/pre_flight_check.py --pipeline all

WITHOUT `PYTHONPATH=.` prepended to the environment. This was the
canonical wrapper invocation pattern across Wf-243..Wf-252 + Wf-355 +
Wf-355-B1 + Wf-413 + Wf-414 — all of which use

    pre_flight_check_path="${REPO_ROOT}/molmetal/scripts/pre_flight_check.py"
    python "${pre_flight_check_path}" --pipeline all --json ...

without prepending PYTHONPATH.

Three tests:

* test_preflight_import_works_without_PYTHONPATH — direct invocation via
  `python <script>` exits rc=0 with --pipeline cfm --strict
* test_preflight_import_works_with_PYTHONPATH — explicit PYTHONPATH=.
  invocation also works (no regression of the existing happy path)
* test_preflight_pipeline_cfm_strict — full --pipeline cfm --strict
  pre-flight gate exit code = 0 (or 2 if --strict surfaces a SKIP), with
  valid JSON on stdout
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


# Repo root is the parent of molmetal/ (the dir containing pyproject.toml).
# From this test file at molmetal/tests/test_wf415a2_preflight_path.py:
#   .parent           = molmetal/tests/
#   .parent.parent    = molmetal/
#   .parent.parent.parent = <repo_root>/
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PRE_FLIGHT_PATH = REPO_ROOT / "molmetal" / "scripts" / "pre_flight_check.py"


def _run_pre_flight(*args: str, env_extra: dict[str, str] | None = None,
                    timeout_sec: int = 600) -> subprocess.CompletedProcess:
    """Invoke pre_flight_check.py as a subprocess.

    Reproduces the canonical wrapper invocation pattern that fails
    without PYTHONPATH=. prefix.
    """
    env = os.environ.copy()
    # Strip any inherited PYTHONPATH so we test the truly-no-prefix case.
    env.pop("PYTHONPATH", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(PRE_FLIGHT_PATH), *args],
        env=env,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )


def test_preflight_import_works_without_PYTHONPATH() -> None:
    """The Wf-415-A2 regression test: invocation without PYTHONPATH=. must
    succeed because pre_flight_check.py now injects the repo root into
    sys.path at lines 54-65 (after standard imports, before the
    read_anchor_json import).
    """
    result = _run_pre_flight("--pipeline", "cfm", "--strict", "--json", "/tmp/wf415a2_test_cfm.json")
    # Must NOT raise ModuleNotFoundError; exit code 0 or 2 are both PASS
    # (0 = all gates PASS, 2 = --strict surfaces a SKIP). Exit code 1
    # would mean a gate FAIL but not the path-injection regression.
    assert result.returncode in (0, 2), (
        f"pre_flight_check.py exited with rc={result.returncode}; "
        f"expected 0 (all PASS) or 2 (SKIP under --strict). "
        f"stderr tail: {result.stderr[-500:]}"
    )
    # Belt-and-braces: assert no ModuleNotFoundError in stderr.
    assert "ModuleNotFoundError" not in result.stderr, (
        f"ModuleNotFoundError regressed: {result.stderr[-500:]}"
    )
    # The JSON output must be parseable.
    json_path = Path("/tmp/wf415a2_test_cfm.json")
    if json_path.exists():
        data = json.loads(json_path.read_text())
        assert "pipeline" in data
        assert data["pipeline"] == "cfm"


def test_preflight_import_works_with_PYTHONPATH() -> None:
    """Existing happy path must remain intact: invocation WITH
    PYTHONPATH=. prefix also works (no regression of the existing
    wrapper pattern that prepends PYTHONPATH).
    """
    result = _run_pre_flight(
        "--pipeline", "cfm", "--strict", "--json", "/tmp/wf415a2_test_pyp.json",
        env_extra={"PYTHONPATH": str(REPO_ROOT)},
    )
    assert result.returncode in (0, 2), (
        f"pre_flight_check.py with PYTHONPATH={REPO_ROOT} exited "
        f"rc={result.returncode}; expected 0 or 2. "
        f"stderr tail: {result.stderr[-500:]}"
    )
    assert "ModuleNotFoundError" not in result.stderr


def test_preflight_pipeline_cfm_strict() -> None:
    """Full --pipeline cfm --strict pre-flight gate suite produces valid
    JSON output and exits cleanly. This is the canonical Wf-413 +
    Wf-355-B1 wrapper invocation pattern.

    Asserts:
    * rc in {0, 2} (no gate FAIL, no ModuleNotFoundError)
    * stdout contains valid JSON with required top-level keys
    * JSON has at least one gate definition
    """
    result = _run_pre_flight(
        "--pipeline", "cfm", "--strict",
        "--json", "/tmp/wf415a2_test_cfm_strict.json",
        timeout_sec=900,  # cfm --strict runs all 6 cfm gates
    )
    assert result.returncode in (0, 2), (
        f"pre_flight_check.py --pipeline cfm --strict exited "
        f"rc={result.returncode}; expected 0 or 2. "
        f"stderr tail: {result.stderr[-500:]}"
    )
    assert "ModuleNotFoundError" not in result.stderr
    json_path = Path("/tmp/wf415a2_test_cfm_strict.json")
    assert json_path.exists(), (
        f"--json output file {json_path} was not written"
    )
    data = json.loads(json_path.read_text())
    # Required top-level keys per Wf-259 + Wf-353-B1 contract.
    assert "pipeline" in data, f"missing 'pipeline' key in {data.keys()}"
    assert data["pipeline"] == "cfm"
    assert "timestamp" in data, f"missing 'timestamp' key"
    # At least one gate suite should be present.
    assert "gates" in data, (
        f"expected 'gates' key in pre_flight JSON, got {data.keys()}"
    )
    assert isinstance(data["gates"], list), (
        f"'gates' should be a list, got {type(data['gates'])}"
    )
    assert len(data["gates"]) >= 1, (
        f"expected ≥1 gate in 'gates' list, got {len(data['gates'])}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
