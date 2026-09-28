"""Pytest: WF-IMPROVEMENT-BACKLOG I2 — every subsystem smoke runs PASS end-to-end.

These tests are the canonical CI gate for the IMPROVEMENT_BACKLOG §I2
deliverable: each of the 4 smoke scripts in
``molmetal/scripts/smoke/`` (CFM, Lambda, evaluators, pitfalls) must
exit 0 AND emit a ``<NAME>_SMOKE PASS`` line within the 180 s per-smoke
budget.

We invoke each script as a subprocess (no mocks) so the contract we
test is exactly the one the dispatch harness consumes.  Skips
automatically when ``molmetal/scripts/smoke/`` is not present (e.g. an
empty branch checkout) so the rest of the suite still runs.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SMOKE_DIR = PROJECT_ROOT / "molmetal" / "scripts" / "smoke"


def _smoke_dir_exists() -> bool:
    return SMOKE_DIR.is_dir()


pytestmark = pytest.mark.skipif(
    not _smoke_dir_exists(),
    reason=f"smoke scripts directory not present at {SMOKE_DIR}",
)


# Locate a python that has rdkit + torch installed.  We honour the
# UV_PROJECT_ENVIRONMENT convention used by env.sh (CadButEaas shared
# venv with rdkit + torch+cu130 pre-installed).
_SHARED_VENV = Path("/home/hugo/codes/CadButEaas/.venv-vlm/bin/python")
_DEFAULT_PY = _SHARED_VENV if _SHARED_VENV.exists() else Path(sys.executable)


def _run_smoke(script: Path, timeout: float = 180.0) -> subprocess.CompletedProcess:
    """Invoke ``bash <script>`` and capture stdout/stderr/rc."""
    if not script.is_file():
        pytest.skip(f"smoke script not present at {script}")
    return subprocess.run(
        ["bash", str(script)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Smoke directory / script surface
# ---------------------------------------------------------------------------
def test_smoke_directory_present():
    """The smoke scripts directory exists and contains the 4 scripts + all.sh."""
    assert SMOKE_DIR.is_dir(), f"missing smoke directory: {SMOKE_DIR}"
    expected = [
        "smoke_cfm.sh",
        "smoke_lambda.sh",
        "smoke_evaluators.sh",
        "smoke_pitfalls.sh",
        "all.sh",
    ]
    for name in expected:
        path = SMOKE_DIR / name
        assert path.is_file(), f"missing smoke script: {path}"


@pytest.mark.parametrize(
    "script_name, expect_pass_token",
    [
        ("smoke_cfm.sh", "CFM_SMOKE PASS"),
        ("smoke_lambda.sh", "LAMBDA_SMOKE PASS"),
        ("smoke_evaluators.sh", "EVALUATORS_SMOKE PASS"),
        ("smoke_pitfalls.sh", "PITFALLS_SMOKE PASS"),
    ],
)
def test_smoke_script_exits_zero_with_pass_token(
    script_name: str, expect_pass_token: str
):
    """Each smoke script exits 0 and emits a PASS token on the good state."""
    script = SMOKE_DIR / script_name
    proc = _run_smoke(script, timeout=180.0)
    assert proc.returncode == 0, (
        f"{script_name} returned rc={proc.returncode}; expected 0.\n"
        f"=== STDOUT (tail) ===\n{proc.stdout[-1500:]}\n"
        f"=== STDERR (tail) ===\n{proc.stderr[-1500:]}\n"
    )
    # The PASS token must appear on stdout (the canonical contract).
    assert expect_pass_token in proc.stdout, (
        f"{script_name} did not emit {expect_pass_token!r} on stdout.\n"
        f"=== STDOUT (tail) ===\n{proc.stdout[-1500:]}\n"
    )


# ---------------------------------------------------------------------------
# Aggregate all.sh — runs all 4 smokes + reports a TOTAL_PASS=4/4 line.
# ---------------------------------------------------------------------------
def test_all_sh_runs_all_4_smokes_and_reports_4_of_4():
    """all.sh runs the 4 subsystem smokes sequentially and reports 4/4 PASS.

    The aggregate all.sh wall-budget is ~3 min on CPU.  We give it a
    600 s timeout (generous) to absorb CI contention.  The expected
    contract: exit 0 AND stdout contains ``ALL_SMOKE TOTAL_PASS=4/4``.
    """
    all_sh = SMOKE_DIR / "all.sh"
    if not all_sh.is_file():
        pytest.skip(f"all.sh not present at {all_sh}")
    proc = subprocess.run(
        ["bash", str(all_sh)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=600.0,
    )
    assert proc.returncode == 0, (
        f"all.sh returned rc={proc.returncode}; expected 0.\n"
        f"=== STDOUT (tail) ===\n{proc.stdout[-2000:]}\n"
        f"=== STDERR (tail) ===\n{proc.stderr[-1500:]}\n"
    )
    assert "ALL_SMOKE TOTAL_PASS=4/4" in proc.stdout, (
        f"all.sh did not emit the ALL_SMOKE TOTAL_PASS=4/4 line.\n"
        f"=== STDOUT (tail) ===\n{proc.stdout[-2000:]}\n"
    )


# ---------------------------------------------------------------------------
# Smoke-script wall-time sanity — every script must finish in < 120 s on CPU.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "script_name",
    [
        "smoke_cfm.sh",
        "smoke_lambda.sh",
        "smoke_evaluators.sh",
        "smoke_pitfalls.sh",
    ],
)
def test_smoke_script_completes_under_120s(script_name: str):
    """Each smoke script completes in < 120 s on CPU (IMPROVEMENT_BACKLOG §I2 budget)."""
    script = SMOKE_DIR / script_name
    if not script.is_file():
        pytest.skip(f"smoke script not present at {script}")
    t0 = time.time()
    proc = subprocess.run(
        ["bash", str(script)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=180.0,
    )
    wall = time.time() - t0
    assert proc.returncode == 0, (
        f"{script_name} returned rc={proc.returncode} in {wall:.1f}s; expected 0."
    )
    assert wall < 120.0, (
        f"{script_name} took {wall:.1f}s; expected < 120 s (IMPROVEMENT_BACKLOG §I2 budget)."
    )
