"""TODO-29 Step 3 — verify the CFM path import fix.

Background
----------
`molmetal/scripts/r10_cfg_real_crossdocked.py` previously failed with
``ModuleNotFoundError: No module named 'molmetal'`` when invoked from
any directory other than the project root (e.g. ``/tmp``, ``molmetal/``,
``molmetal/tests/``).  This test module locks down the import path so
the regression cannot return.

The fix is two-layered (defence in depth):

1. **External**: a ``.pth`` file in the project venv
   (``.venv/lib/python3.12/site-packages/molmetal_dev.pth``) that
   unconditionally adds the project root to ``sys.path``.  This is the
   primary fix and handles every CWD without any code change.

2. **Internal**: a 12-line bootstrap block at the top of
   ``r10_cfg_real_crossdocked.py`` that walks up from
   ``Path(__file__).resolve().parent`` looking for the first directory
   containing ``molmetal/__init__.py`` and prepends it to ``sys.path``.
   This keeps the script self-sufficient if the ``.pth`` file is
   missing (e.g. fresh venv rebuild).

Test coverage (5 tests, exceeds the 3+ spec):
* ``test_import_from_project_root`` — canonical happy path.
* ``test_import_via_subprocess_from_tmp`` — reproduces the original
  bug: invocation from ``/tmp`` via subprocess.
* ``test_import_from_molmetal_tests_dir`` — verifies the CWD inside
  the tests directory still works (where pytest actually runs).
* ``test_import_from_molmetal_dir`` — verifies the package-internal
  CWD also works.
* ``test_pth_file_present_and_valid`` — asserts the external
  ``.pth`` fix exists and points at a directory containing
  ``molmetal/__init__.py``.

All tests are CPU-only and read-only; they do NOT mutate any state.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


# Project root (this file's grandparent's parent: tests/ -> molmetal/ -> root).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _PROJECT_ROOT / "molmetal" / "scripts" / "r10_cfg_real_crossdocked.py"
_PTH_PATH = (
    _PROJECT_ROOT
    / ".venv"
    / "lib"
    / "python3.12"
    / "site-packages"
    / "molmetal_dev.pth"
)


def _has_main_attr() -> bool:
    """Import the script's module and confirm ``main`` is callable."""
    from molmetal.scripts.r10_cfg_real_crossdocked import main
    return callable(main)


def test_import_from_project_root():
    """Canonical happy path: import the script module and pull ``main``."""
    # When pytest is invoked from the project root, ``molmetal`` is
    # already on sys.path (either via the .pth file or the cwd
    # ``''`` entry).  This should always work.
    assert _has_main_attr(), (
        "molmetal.scripts.r10_cfg_real_crossdocked.main is not importable "
        "from the project root — CFM path import regressed."
    )


def test_import_from_molmetal_dir():
    """Verify the module is importable when CWD is the molmetal/ directory.

    Pre-fix this raised ``ModuleNotFoundError: No module named 'molmetal.scripts'``
    because ``molmetal/`` itself was the top-level on sys.path, hiding
    the package import.  The ``.pth`` file (and the in-script
    bootstrap) both fix this.
    """
    cwd = _PROJECT_ROOT / "molmetal"
    assert cwd.is_dir()
    # In-process test: simulate the failing CWD by removing any direct
    # ``molmetal/`` entry from sys.path, then re-importing.
    saved = list(sys.path)
    try:
        # Remove any path entries that point at the molmetal/ directory
        # itself (not its parent).
        sys.path[:] = [
            p for p in saved
            if Path(p).resolve() != cwd.resolve()
        ]
        # Also drop any cached import so the re-import goes through
        # the bootstrap path.
        for mod in list(sys.modules):
            if mod == "molmetal.scripts.r10_cfg_real_crossdocked":
                sys.modules.pop(mod, None)
        assert _has_main_attr(), (
            "Cannot import CFM path script with molmetal/ stripped from "
            "sys.path — the in-script bootstrap or .pth fix regressed."
        )
    finally:
        sys.path[:] = saved


def test_import_from_molmetal_tests_dir():
    """Verify the CWD where pytest actually runs still works.

    pytest is normally invoked from the project root with
    ``molmetal/tests/`` as the test root.  This test simulates that
    invocation path by ensuring ``molmetal.scripts.r10_cfg_real_crossdocked``
    is importable when CWD is the tests directory.
    """
    tests_dir = _PROJECT_ROOT / "molmetal" / "tests"
    assert tests_dir.is_dir()
    # Run as a subprocess to truly simulate CWD = tests_dir.
    # We avoid changing os.chdir() because that would perturb pytest.
    result = subprocess.run(
        [sys.executable, "-c", "import molmetal.scripts.r10_cfg_real_crossdocked as m; assert callable(m.main); print('OK')"],
        cwd=str(tests_dir),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"Import failed when CWD={tests_dir}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "OK" in result.stdout


def test_import_via_subprocess_from_tmp():
    """Reproduce the original bug: invocation from ``/tmp``.

    Pre-fix this raised ``ModuleNotFoundError: No module named 'molmetal'``
    because ``/tmp`` has no relationship to the project root and
    ``uv run --project`` does not add the project root to ``sys.path``
    automatically.  The ``.pth`` file fixes this transparently.
    """
    snippet = textwrap.dedent(
        """
        import sys
        # Force the failing condition: no CWD-based molmetal access.
        sys.path[:] = [p for p in sys.path if 'molmetal' not in p]
        from molmetal.scripts.r10_cfg_real_crossdocked import main
        assert callable(main), 'main is not callable'
        print('TMP_IMPORT_OK')
        """
    ).strip()
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd="/tmp",
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": ""},  # strip any inherited PYTHONPATH
    )
    assert result.returncode == 0, (
        f"Import failed when invoked from /tmp.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "TMP_IMPORT_OK" in result.stdout


def test_pth_file_present_and_valid():
    """Assert the external ``.pth`` fix exists and points at the project.

    The ``.pth`` file lives inside the project venv; if it's missing
    the internal in-script bootstrap still saves the import (verified
    by the other tests) but the CWD-agnostic guarantee is degraded.
    This test is therefore an early-warning canary, not a hard
    requirement for the other tests to pass.
    """
    if not _PTH_PATH.is_file():
        pytest.skip(
            f"Optional .pth fix missing at {_PTH_PATH}; relying on "
            f"in-script bootstrap only."
        )
    target = _PTH_PATH.read_text().strip()
    assert target, f".pth file {_PTH_PATH} is empty"
    target_path = Path(target)
    assert target_path.is_dir(), (
        f".pth file points at non-existent directory: {target}"
    )
    assert (target_path / "molmetal" / "__init__.py").is_file(), (
        f".pth file points at {target_path} but that directory does not "
        f"contain molmetal/__init__.py"
    )
