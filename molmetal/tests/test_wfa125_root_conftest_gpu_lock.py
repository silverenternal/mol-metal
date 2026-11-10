"""Wf-A125 G1 — the NO-GPU contract must reach the top-level ``tests/`` tree.

THE GAP (measured 2026-10-31, before this wave).  The CPU lock lived only in
``molmetal/tests/conftest.py:53-55``.  Pytest loads a ``conftest.py`` from the
ANCESTORS of the collected file, and ``molmetal/tests/`` is a *sibling* of the
repo-root ``tests/`` tree, not an ancestor of it — so the lock never applied to
``tests/``, whose modules self-select the device:

    tests/test_fused_ops.py:32-39
    tests/test_new_kernels.py:38-41
    tests/test_triton_fused_residual_wired.py:57-60

THE FIX.  A repo-root ``conftest.py`` — the root IS an ancestor of both
``tests/`` and ``molmetal/tests/`` — blanks ``CUDA_VISIBLE_DEVICES`` at import
time unless ``MOLMETAL_TEST_ALLOW_GPU=1``.

THE PROOF (this file).  (a) the root conftest exists; (b) a FRESH subprocess
that executes it leaves ``CUDA_VISIBLE_DEVICES == ""``; (c) the same
subprocess with ``MOLMETAL_TEST_ALLOW_GPU=1`` pre-set leaves the caller's
value UNCHANGED; (d) a pytest run over a file under ``tests/`` prints the
root conftest's status line — which is only possible if the root conftest was
loaded as an ancestor of that file; (e) the ``tests/`` tree's GPU-dependent
assertions SKIP instead of reaching for a CUDA device.

No file is written into ``tests/``: the ancestry is proven by running pytest
over an existing file there.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_CONFTEST = REPO_ROOT / "conftest.py"
TESTS_DIR = REPO_ROOT / "tests"

#: The status line the root conftest prints (in the header, or — because
#: pytest suppresses the whole header block under ``-q`` — in the terminal
#: summary).  Its presence in a pytest run over ``tests/`` is the ancestry
#: proof: only the root conftest emits it.
CPU_LOCK_LINE_RE = re.compile(r"mol-metal: CPU lock ON \(CUDA_VISIBLE_DEVICES=''\)")

#: The GPU-contract summary line (Wf-A125 G9).
GPU_CONTRACT_LINE_RE = re.compile(
    r"mol-metal GPU contract: (\d+) GPU-required test\(s\) did not run"
)

_RUN_ROOT_CONFTEST = (
    "import os, runpy;"
    "runpy.run_path({conftest!r});"
    "print('CVD=' + repr(os.environ.get('CUDA_VISIBLE_DEVICES')))"
)


def _cpu_env() -> dict:
    """The environment a CPU-only test session sees."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.pop("MOLMETAL_TEST_ALLOW_GPU", None)
    return env


def _run(cmd: list, env: dict | None = None, timeout: int = 300):
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env if env is not None else _cpu_env(),
        timeout=timeout,
    )


def _pytest(*args: str, env: dict | None = None):
    return _run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args],
        env=env,
    )


# ---------------------------------------------------------------------------
# (a) the file exists
# ---------------------------------------------------------------------------
def test_the_root_conftest_exists() -> None:
    """G1: the fix is a repo-root ``conftest.py``."""
    assert ROOT_CONFTEST.is_file(), (
        f"missing {ROOT_CONFTEST} — the CPU lock cannot reach tests/ without it"
    )


# ---------------------------------------------------------------------------
# the lock is in the SOURCE, at MODULE level
# ---------------------------------------------------------------------------
def test_the_root_conftest_carries_the_no_gpu_lock() -> None:
    """The lock must be module-level code, not a fixture."""
    text = ROOT_CONFTEST.read_text(encoding="utf-8")
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = ""' in text, (
        "the root conftest no longer blanks CUDA_VISIBLE_DEVICES; the no-GPU "
        "contract (Wf-A123 FIX 5 / Wf-A125 G1) has been removed"
    )
    assert "MOLMETAL_TEST_ALLOW_GPU" in text, (
        "the root conftest lost its explicit GPU opt-in"
    )
    # It must be at MODULE level: the variable has to be set before anything
    # can initialise a CUDA context.  Everything before the first ``def`` is
    # module level.
    head = text.split("def ")[0]
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = ""' in head, (
        "the CUDA lock is not at module level in the root conftest"
    )


# ---------------------------------------------------------------------------
# (b) a fresh subprocess that executes the conftest blanks the device list
# ---------------------------------------------------------------------------
def test_executing_the_root_conftest_blanks_the_device_list() -> None:
    """END TO END: executing the root conftest blanks a caller's device list."""
    env = _cpu_env()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.pop("MOLMETAL_TEST_ALLOW_GPU", None)
    proc = _run(
        [
            sys.executable,
            "-c",
            _RUN_ROOT_CONFTEST.format(conftest=str(ROOT_CONFTEST)),
        ],
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "CVD=''" in proc.stdout, (
        f"the root conftest did not blank CUDA_VISIBLE_DEVICES: {proc.stdout!r}"
    )


# ---------------------------------------------------------------------------
# (c) the opt-out leaves the caller's value alone
# ---------------------------------------------------------------------------
def test_the_gpu_opt_out_leaves_the_device_list_alone() -> None:
    """END TO END: MOLMETAL_TEST_ALLOW_GPU=1 keeps the caller's device list."""
    env = _cpu_env()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["MOLMETAL_TEST_ALLOW_GPU"] = "1"
    proc = _run(
        [
            sys.executable,
            "-c",
            _RUN_ROOT_CONFTEST.format(conftest=str(ROOT_CONFTEST)),
        ],
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "CVD='0'" in proc.stdout, (
        f"the GPU opt-in did not preserve CUDA_VISIBLE_DEVICES: {proc.stdout!r}"
    )


# ---------------------------------------------------------------------------
# (d) ANCESTRY PROOF — the root conftest is loaded for a file under tests/
# ---------------------------------------------------------------------------
def test_the_root_conftest_is_loaded_for_a_file_under_tests() -> None:
    """The root conftest must be an ANCESTOR of the top-level ``tests/`` tree.

    ``pytest --collect-only -q tests/test_fused_ops.py`` prints the root
    conftest's status line.  Only the root conftest emits it, so its presence
    proves pytest loaded the root conftest while collecting a file under
    ``tests/``.  NO file is written into ``tests/``.
    """
    proc = _pytest("--collect-only", "tests/test_fused_ops.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert CPU_LOCK_LINE_RE.search(proc.stdout), (
        "the root conftest was NOT loaded for tests/test_fused_ops.py — the "
        "CPU lock does not reach the top-level tests/ tree:\n"
        + proc.stdout[-3000:]
    )


# ---------------------------------------------------------------------------
# (e) BEHAVIOURAL PROOF — the tests/ tree no longer reaches for a device
# ---------------------------------------------------------------------------
def test_the_tests_tree_skips_its_gpu_dependent_assertions() -> None:
    """``tests/test_cpu_tensor_rejection.py`` must SKIP, not take a device.

    Four of its assertions are gated on ``torch.cuda.is_available()``.  With
    the root conftest loaded they SKIP (4 skipped); without it, on a GPU host
    they would RUN and take VRAM.  MEASURED 2026-10-31 (CPU session):
    ``2 passed, 4 skipped``.
    """
    proc = _pytest("tests/test_cpu_tensor_rejection.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert re.search(r"\b4 skipped\b", proc.stdout), (
        "the tests/ tree did not skip its GPU-dependent assertions:\n"
        + proc.stdout[-3000:]
    )
    match = GPU_CONTRACT_LINE_RE.search(proc.stdout)
    assert match is not None, (
        "the GPU-contract summary line is missing:\n" + proc.stdout[-3000:]
    )
    assert int(match.group(1)) == 4, (
        f"expected 4 GPU-required skips, got {match.group(1)}:\n"
        + proc.stdout[-3000:]
    )


# ---------------------------------------------------------------------------
# the lock must never change a session's exit code
# ---------------------------------------------------------------------------
def test_the_lock_does_not_change_the_exit_code() -> None:
    """A CPU-only session that skips the GPU tests is a PASSING session."""
    proc = _pytest("tests/test_cpu_tensor_rejection.py")
    assert proc.returncode == 0, (
        f"the CPU lock changed the exit code to {proc.returncode}:\n"
        + proc.stdout[-3000:]
    )
