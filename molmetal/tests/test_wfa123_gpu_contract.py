"""Wf-A123 FIX 5 -- the standing NO-GPU contract for the test suite.

The standing user directive for this repository is 禁止使用gpu: a test run
must not silently take VRAM.  MEASURED 2026-10-31 (before this fix): a full
``pytest molmetal/tests`` run held 1444 MiB (the pytest process itself) plus
2218 MiB (a smoke subprocess) of GPU 0 VRAM.

THE DURABLE FORM is in ``molmetal/tests/conftest.py``: at MODULE IMPORT TIME
it blanks ``CUDA_VISIBLE_DEVICES`` unless the caller opted in with
``MOLMETAL_TEST_ALLOW_GPU=1``.  Because ``torch.cuda.is_available()``
consults that variable at CALL time, every test that self-selects
``"cuda" if torch.cuda.is_available() else "cpu"`` resolves to ``cpu``.

These tests check the CONTRACT, not the physics: nothing here allocates GPU
memory, and no test in this file needs a GPU.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TESTS_DIR = PROJECT_ROOT / "molmetal" / "tests"
CONFTEST = TESTS_DIR / "conftest.py"

#: The modules the Wf-A123 scout classified GPU-ACCIDENTAL: each one
#: self-selects the device and therefore takes VRAM on a GPU host unless it
#: is told not to.  Each carries the module-level guard.
GPU_ACCIDENTAL = [
    "test_lipman_import_isolation.py",
    "test_lipman_sampling_contract.py",
    "test_pocket_conditioned_lipman.py",
    "test_end_to_end_cached_path.py",
    "test_dmpnn_scatter_wiring.py",
    "test_egnn_coord_scatter_wired.py",
    "test_metal_hybrid_v3.py",
    "test_wf328w9_triton_edge.py",
    "test_wf328w7_dmpnn_pyg.py",
    "test_amp_context.py",
]

#: The modules the scout classified GPU-REQUIRED: a GPU test that silently
#: runs on CPU is a FALSE GREEN, so these must SKIP cleanly.
GPU_REQUIRED = [
    "test_rocm_egnn.py",
    "test_wf25_gpu_smoke.py",
]


def _run_python(code: str, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )


_LOAD_CONFTEST = (
    "import importlib.util, os, sys;"
    "spec=importlib.util.spec_from_file_location('wfa123_conftest', {conftest!r});"
    "mod=importlib.util.module_from_spec(spec);"
    "spec.loader.exec_module(mod);"
    "print('CVD=' + repr(os.environ.get('CUDA_VISIBLE_DEVICES')))"
)


def test_the_conftest_carries_the_no_gpu_lock() -> None:
    """The lock is in the conftest SOURCE (checked even under the opt-in)."""
    text = CONFTEST.read_text(encoding="utf-8")
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = ""' in text, (
        "molmetal/tests/conftest.py no longer blanks CUDA_VISIBLE_DEVICES; "
        "the no-GPU contract (Wf-A123 FIX 5) has been removed"
    )
    assert 'MOLMETAL_TEST_ALLOW_GPU' in text, (
        "the conftest lost its explicit GPU opt-in (MOLMETAL_TEST_ALLOW_GPU)"
    )
    # It must be MODULE-level code, not a fixture: the variable has to be set
    # before anything can initialise a CUDA context.
    head = text.split("def _preload_init")[0]
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = ""' in head, (
        "the CUDA lock is not at module level in the conftest"
    )


def test_importing_the_conftest_blanks_the_device_list() -> None:
    """END TO END: importing the conftest blanks a caller's device list."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.pop("MOLMETAL_TEST_ALLOW_GPU", None)
    proc = _run_python(_LOAD_CONFTEST.format(conftest=str(CONFTEST)), env)
    assert proc.returncode == 0, proc.stderr
    assert "CVD=''" in proc.stdout, (
        f"the conftest did not blank CUDA_VISIBLE_DEVICES: {proc.stdout!r}"
    )


def test_the_opt_out_leaves_the_device_list_alone() -> None:
    """END TO END: MOLMETAL_TEST_ALLOW_GPU=1 keeps the caller's device list."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["MOLMETAL_TEST_ALLOW_GPU"] = "1"
    proc = _run_python(_LOAD_CONFTEST.format(conftest=str(CONFTEST)), env)
    assert proc.returncode == 0, proc.stderr
    assert "CVD='0'" in proc.stdout, (
        f"the GPU opt-in did not preserve CUDA_VISIBLE_DEVICES: {proc.stdout!r}"
    )


def test_this_session_itself_is_cpu_only() -> None:
    """In a default (no opt-in) session the device list is blank."""
    if os.environ.get("MOLMETAL_TEST_ALLOW_GPU") == "1":
        pytest.skip("GPU opt-in session: the conftest deliberately does not blank")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", (
        "this test session is NOT CPU-only; a test took the GPU"
    )


@pytest.mark.parametrize("module", GPU_ACCIDENTAL)
def test_the_gpu_accidental_module_forces_cpu(module: str) -> None:
    """Every self-selecting module carries the module-level no-GPU guard."""
    path = TESTS_DIR / module
    assert path.is_file(), f"missing test module {module}"
    text = path.read_text(encoding="utf-8")
    assert "MOLMETAL_TEST_ALLOW_GPU" in text, (
        f"{module} self-selects the device but carries no no-GPU guard"
    )
    assert 'os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")' in text, (
        f"{module} does not blank the device list"
    )
    # The guard must precede the FIRST ``import torch`` STATEMENT (module
    # level or lazy): a CUDA context can only be initialised after torch is
    # imported.  Prose mentions of "import torch" in a docstring do not
    # count, so the search is line-anchored.
    guard_at = text.index('os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")')
    torch_at = min(
        (
            i
            for i, line in enumerate(text.splitlines())
            if line.strip() == "import torch"
        ),
        default=len(text),
    )
    guard_line = text[:guard_at].count("\n")
    assert guard_line < torch_at, (
        f"{module} blanks the device list AFTER importing torch "
        f"(guard line {guard_line}, import torch line {torch_at})"
    )


@pytest.mark.parametrize("module", GPU_REQUIRED)
def test_the_gpu_required_module_skips_cleanly(module: str) -> None:
    """Every GPU-REQUIRED module skips when no CUDA device is visible."""
    path = TESTS_DIR / module
    assert path.is_file(), f"missing test module {module}"
    text = path.read_text(encoding="utf-8")
    assert "pytest.mark.skipif(" in text, (
        f"{module} is GPU-REQUIRED but has no skipif: a GPU test that runs on "
        f"CPU is a false green"
    )
    assert "not torch.cuda.is_available()" in text, (
        f"{module} does not gate on torch.cuda.is_available()"
    )


def test_the_gpu_required_modules_really_skip_in_a_cpu_session() -> None:
    """END TO END: the GPU-REQUIRED modules SKIP (not fail, not run on CPU).

    Run in a subprocess with the device list blanked, exactly as a normal
    test session sees it.
    """
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.pop("MOLMETAL_TEST_ALLOW_GPU", None)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--no-header",
        "-p",
        "no:cacheprovider",
        *[str(TESTS_DIR / m) for m in GPU_REQUIRED],
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=env, cwd=str(PROJECT_ROOT), timeout=900
    )
    tail = proc.stdout[-4000:]
    assert "skipped" in tail, (
        "the GPU-REQUIRED modules did not report any skip:\n" + tail
    )
    assert "failed" not in tail and "error" not in tail.lower(), (
        "the GPU-REQUIRED modules did not skip cleanly:\n" + tail
    )
    assert proc.returncode == 0, "pytest exited non-zero:\n" + tail
