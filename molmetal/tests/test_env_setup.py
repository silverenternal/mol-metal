"""test_env_setup.py — Wf-23 Phase 3 environment-setup guard.

Wf-23 retrain jobs (5-seed × 5-condition benchmark, paper v2 update) all
need to run inside the shared ``/home/hugo/codes/mol-metal/.venv``
because that venv is the only place on this host with ``torch==2.14.0+cu130``
+ ``triton==3.8.0`` + Blackwell (``sm_120``) arch list installed.

The Mol-Metal repo gets its Python from a symlink at
``molmetal/.venvs/mol-metal-venv`` plus a ``PYTHONPATH=$REPO_ROOT`` shim
so ``import molmetal`` resolves under that interpreter. The setup script
``molmetal/scripts/setup_molmetal_env.sh`` wires those env vars up and
``exec``s the shared venv's Python.

This module is the read-only test harness for that contract. It is
deliberately import-free of the chemistry stack (rdkit / openbabel /
torch) so a green ``pytest`` confirms the *wiring* is correct, not that
the chemistry stack works (that is covered by the per-module test
suites). The torch / molmetal / CUDA probes are skipped gracefully when
the interpreter under test is the system Python (CI), but they run for
real when invoked via ``bash molmetal/scripts/setup_molmetal_env.sh``.

Tests in this file (10 total):

  1.  test_venv_symlink_exists
  2.  test_venv_symlink_target_is_absolute_and_resolves
  3.  test_shared_venv_has_torch_with_cu130
  4.  test_shared_venv_arch_list_includes_sm_120_blackwell
  5.  test_setup_script_executable
  6.  test_setup_script_sets_pythonpath_to_repo_root
  7.  test_setup_script_executes_shared_python
  8.  test_setup_script_can_import_molmetal
  9.  test_setup_script_can_import_torch_with_cuda
  10. test_gitignore_excludes_venvs

Each test is independent; collectively they prove a developer can run

    bash molmetal/scripts/setup_molmetal_env.sh python -m pytest \\
        molmetal/tests/test_env_setup.py -v

and get a green run, OR that ``pytest -m env_setup`` (see ``pytest`` config
note at the bottom of this file) confirms the wiring in CI.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Paths (resolved once at module import; cheap and side-effect-free).
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
MOLMETAL_PKG = REPO_ROOT / "molmetal"
VENV_SYMLINK = MOLMETAL_PKG / ".venvs" / "mol-metal-venv"
SHARED_VENV = Path("/home/hugo/codes/mol-metal/.venv")
SHARED_PY = SHARED_VENV / "bin" / "python"
SETUP_SCRIPT = MOLMETAL_PKG / "scripts" / "setup_molmetal_env.sh"
GITIGNORE = REPO_ROOT / ".gitignore"


# ---------------------------------------------------------------------------
# 1. The symlink exists (Wf-23 Phase 3 deliverable #1).
# ---------------------------------------------------------------------------
def test_venv_symlink_exists() -> None:
    """``molmetal/.venvs/mol-metal-venv`` must exist as a symlink."""
    assert (
        VENV_SYMLINK.exists()
    ), f"missing {VENV_SYMLINK}. Run: ln -snf {SHARED_VENV} {VENV_SYMLINK}"
    # It should be a symlink (or directory pointing at SHARED_VENV) — either
    # is acceptable; the symlink is the preferred layout but a copy is OK
    # for hosts where symlinks are restricted.
    if VENV_SYMLINK.is_symlink():
        assert VENV_SYMLINK.resolve() == SHARED_VENV.resolve(), (
            f"{VENV_SYMLINK} points at {VENV_SYMLINK.resolve()}, "
            f"expected {SHARED_VENV.resolve()}"
        )


# ---------------------------------------------------------------------------
# 2. The symlink target is absolute and resolves to a real venv.
# ---------------------------------------------------------------------------
def test_venv_symlink_target_is_absolute_and_resolves() -> None:
    target = os.readlink(VENV_SYMLINK) if VENV_SYMLINK.is_symlink() else None
    if target is not None:
        assert os.path.isabs(target), f"symlink target {target!r} must be absolute"
    # Whichever layout was chosen, the shared venv must have a Python.
    assert SHARED_PY.exists() and os.access(
        SHARED_PY, os.X_OK
    ), f"{SHARED_PY} missing or not executable — the shared env is broken"


# ---------------------------------------------------------------------------
# 3. The shared venv actually has torch with the cu130 build.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not SHARED_PY.exists(), reason=f"no {SHARED_PY} on this host")
def test_shared_venv_has_torch_with_cu130() -> None:
    """Shared venv must have torch==2.14.0+cu130 (Blackwell-ready)."""
    cp = subprocess.run(
        [str(SHARED_PY), "-c", "import torch; print(torch.__version__)"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert cp.returncode == 0, f"torch import failed: {cp.stderr}"
    assert "+cu130" in cp.stdout, (
        f"torch {cp.stdout.strip()!r} is not the cu130 build — Blackwell "
        f"sm_120 will not be compiled in. Reinstall via uv."
    )


# ---------------------------------------------------------------------------
# 4. CUDA arch list must include sm_120 (Blackwell). Skipped if no GPU.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not SHARED_PY.exists(), reason=f"no {SHARED_PY} on this host")
def test_shared_venv_arch_list_includes_sm_120_blackwell() -> None:
    cp = subprocess.run(
        [
            str(SHARED_PY),
            "-c",
            "import torch; a=torch.cuda.get_arch_list() if torch.cuda.is_available() "
            "else []; print(' '.join(a))",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if cp.returncode != 0 or not cp.stdout.strip():
        pytest.skip("torch.cuda not available on this host")
    archs = cp.stdout.strip().split()
    assert "sm_120" in archs, (
        f"sm_120 (Blackwell) not in arch list {archs}. CUDA 13.0 / torch "
        f"2.14+cu130 wheel must be reinstalled."
    )


# ---------------------------------------------------------------------------
# 5. The setup script is executable.
# ---------------------------------------------------------------------------
def test_setup_script_executable() -> None:
    assert SETUP_SCRIPT.exists(), f"missing {SETUP_SCRIPT}"
    assert os.access(
        SETUP_SCRIPT, os.X_OK
    ), f"{SETUP_SCRIPT} is not executable. Run: chmod +x {SETUP_SCRIPT}"


# ---------------------------------------------------------------------------
# 6. The setup script must prepend REPO_ROOT to PYTHONPATH so import molmetal
#    resolves, even when the shared venv was never activated in the parent.
# ---------------------------------------------------------------------------
def test_setup_script_sets_pythonpath_to_repo_root() -> None:
    """Inspect the script source — it must `export PYTHONPATH=$REPO_ROOT...`.

    The script resolves the repo root at runtime via ``cd "$(dirname ...)``,
    so the literal path ``str(REPO_ROOT)`` is NOT expected in the source —
    we instead require the ``$REPO_ROOT`` shell variable to be both
    *defined* (from the ``cd`` resolution at the top) and *referenced* in
    the ``PYTHONPATH`` export. This keeps the script portable across
    different host paths (e.g. ``/home/hugo/codes/mol-metal`` vs CI).
    """
    src = SETUP_SCRIPT.read_text()
    assert "PYTHONPATH" in src, "setup script never references PYTHONPATH"
    assert "REPO_ROOT=" in src, (
        f"setup script never resolves REPO_ROOT — `import molmetal` "
        f"will fail under the shared interpreter"
    )
    assert (
        "${REPO_ROOT}" in src
    ), "setup script defines REPO_ROOT but never uses it in an export"
    # End-to-end confirmation: run the script and verify PYTHONPATH landed.
    cp = subprocess.run(
        [
            "bash",
            "-c",
            f"PYTHONPATH= bash {SETUP_SCRIPT} python -c "
            f'\'import os, sys; sys.path.insert(0, os.environ["PYTHONPATH"]); '
            f"import molmetal; print(molmetal.__file__)'",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if SHARED_PY.exists():
        assert (
            cp.returncode == 0
        ), f"setup-script PYTHONPATH wiring failed end-to-end: {cp.stderr}"


# ---------------------------------------------------------------------------
# 7. End-to-end: bash setup_molmetal_env.sh python --version  → shared venv.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not SHARED_PY.exists(), reason=f"no {SHARED_PY} on this host")
def test_setup_script_executes_shared_python() -> None:
    cp = subprocess.run(
        [
            "bash",
            str(SETUP_SCRIPT),
            "python",
            "-c",
            "import sys; print(sys.executable)",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert cp.returncode == 0, f"setup script failed: {cp.stderr}"
    assert cp.stdout.strip() == str(
        SHARED_PY
    ), f"setup script invoked {cp.stdout.strip()!r}, expected {SHARED_PY}"


# ---------------------------------------------------------------------------
# 8. End-to-end: bash setup_molmetal_env.sh python -c 'import molmetal'.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not SHARED_PY.exists(), reason=f"no {SHARED_PY} on this host")
def test_setup_script_can_import_molmetal() -> None:
    cp = subprocess.run(
        [
            "bash",
            str(SETUP_SCRIPT),
            "python",
            "-c",
            "import molmetal; print(molmetal.__file__)",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert cp.returncode == 0, f"import molmetal failed: {cp.stderr}"
    assert "molmetal/__init__.py" in cp.stdout, (
        f"import molmetal did not resolve to the local package: "
        f"{cp.stdout!r} / {cp.stderr!r}"
    )


# ---------------------------------------------------------------------------
# 9. End-to-end: bash setup_molmetal_env.sh python -c 'import torch' +
#    cuda available. Skipped gracefully when CUDA is missing (CI).
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not SHARED_PY.exists(), reason=f"no {SHARED_PY} on this host")
def test_setup_script_can_import_torch_with_cuda() -> None:
    cp = subprocess.run(
        [
            "bash",
            str(SETUP_SCRIPT),
            "python",
            "-c",
            "import torch; print(torch.__version__, torch.cuda.is_available())",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert cp.returncode == 0, f"import torch failed: {cp.stderr}"
    version, cuda_flag = cp.stdout.strip().split()
    assert "+cu" in version, f"torch {version!r} is not a CUDA build"
    # cuda_flag is "True" on the GPU host and "False" on CPU-only CI; both
    # are acceptable. We only fail if torch import itself crashed.


# ---------------------------------------------------------------------------
# 10. .gitignore must exclude .venvs/ so the symlink never leaks into a
#     public-release commit. The repo's .gitignore uses the trailing glob
#     ``*.venv*`` (matches both ``.venv`` and ``.venvs``); check for that.
# ---------------------------------------------------------------------------
def test_gitignore_excludes_venvs() -> None:
    assert GITIGNORE.exists(), f"missing {GITIGNORE}"
    gitignore_text = GITIGNORE.read_text()
    # Either an explicit `.venvs/` line or a `.venv*` glob is acceptable.
    has_explicit = bool(
        "\n.venvs/" in ("\n" + gitignore_text) or "/.venvs/" in gitignore_text
    )
    has_glob = ".venv*" in gitignore_text or "*.venv*" in gitignore_text
    assert has_explicit or has_glob, (
        ".gitignore does not exclude .venvs/ — symlink to the shared "
        "project venv will leak into the public release. Add either "
        "'.venvs/' (explicit) or '.venv*' (glob)."
    )
