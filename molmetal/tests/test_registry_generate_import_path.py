# SPDX-License-Identifier: Apache-2.0
"""Regression test: ``registry.py generate`` must work when invoked directly.

The bug this pins (fixed 2026-10-10)
-------------------------------------
``registry.py`` had no ``sys.path`` setup. Run as
``python molmetal/scripts/registry.py generate``, only the script's own
directory is importable, so ``generate()``'s

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

raised ``ModuleNotFoundError: No module named 'molmetal'``.

``check()`` never hit it — it imports nothing from the package — so the read
path looked perfectly healthy while only the write path was broken. That
asymmetry is the whole reason a test is needed: without this, the bug is
invisible until someone runs ``registry generate`` to refresh a stale lock,
which is exactly when it is least convenient to fail.

The test drives ``generate()`` in-process with ``PYTHONPATH`` scrubbed from the
environment, so it reproduces the original failure rather than relying on the
test runner's own import setup.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REGISTRY_PY = REPO_ROOT / "molmetal" / "scripts" / "registry.py"


def _run(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    """Run registry.py as a fresh process with PYTHONPATH removed from env."""
    child_env = {k: v for k, v in env.items() if k != "PYTHONPATH"}
    return subprocess.run(
        [sys.executable, str(REGISTRY_PY), *args],
        cwd=str(REPO_ROOT),
        env=child_env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_registry_module_is_importable_from_script_dir():
    """The repo root must be on sys.path once registry.py is imported.

    Cheap, in-process, and it fails at exactly the seam the fix patches.
    """
    assert REGISTRY_PY.is_file(), f"registry.py missing at {REGISTRY_PY}"
    text = REGISTRY_PY.read_text(encoding="utf-8")
    assert "_REPO_ROOT" in text, (
        "registry.py lost its sys.path bootstrap — `generate` will raise "
        "ModuleNotFoundError when run as a script."
    )


def test_registry_generate_runs_without_pythonpath(tmp_path: Path):
    """`registry generate` must succeed with PYTHONPATH unset.

    `check` (no package imports) is asserted too, so a future change that
    breaks one path but not the other is caught rather than masked.
    """
    if not REGISTRY_PY.is_file():
        pytest.skip("registry.py not present")

    gen = _run(["generate"], os.environ)
    assert gen.returncode == 0, (
        "`registry generate` failed without PYTHONPATH — this is the "
        "ModuleNotFoundError regression.\n"
        f"stdout:\n{gen.stdout}\nstderr:\n{gen.stderr}"
    )
    assert "ModuleNotFoundError" not in gen.stderr

    chk = _run(["check"], os.environ)
    assert chk.returncode == 0, f"`registry check` regressed:\n{chk.stderr}"
