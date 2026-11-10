"""Repo-root pytest conftest — the NO-GPU contract for BOTH test trees.

Wf-A125 / G1 + G9.  The standing user directive for this repository is
禁止使用gpu: a test run must not silently take VRAM.

WHY THIS FILE EXISTS AT THE REPO ROOT
------------------------------------
``molmetal/tests/conftest.py`` (Wf-A123 FIX 5) already blanks
``CUDA_VISIBLE_DEVICES`` at import time — but pytest only loads a
``conftest.py`` from the ANCESTORS of the collected file, and
``molmetal/tests/`` is a *sibling* of the top-level ``tests/`` tree, not an
ancestor of it.  So the lock never reached ``tests/``, whose modules
self-select the device:

    tests/test_fused_ops.py:32-39
    tests/test_new_kernels.py:38-41
    tests/test_triton_fused_residual_wired.py:57-60

The repo root is an ancestor of BOTH ``tests/`` and ``molmetal/tests/``, so
ONE file here covers both trees.  ``pyproject.toml:142`` calls ``tests/``
"the CPU-only regression-guard surface" — this file makes that true.

THE LOCK IS AT MODULE LEVEL, NOT IN A FIXTURE
---------------------------------------------
``torch`` may already be imported by the time any fixture runs, and the
environment variable has to be in place BEFORE anything can initialise a CUDA
context.  ``torch.cuda.is_available()`` consults ``CUDA_VISIBLE_DEVICES`` at
CALL time, so every test that self-selects ``"cuda" if
torch.cuda.is_available() else "cpu"`` now resolves to ``cpu``.

GPU TESTING IS STILL AVAILABLE, EXPLICITLY: run with
``MOLMETAL_TEST_ALLOW_GPU=1 pytest ...`` and ``CUDA_VISIBLE_DEVICES`` is left
exactly as the caller set it.  GPU-REQUIRED tests skip cleanly when CUDA is
unavailable (a GPU test that silently runs on CPU is a false green).

THE GPU-SKIP IS NO LONGER SILENT (Wf-A125 / G9)
-----------------------------------------------
Before this file, a CPU session that skipped nine GPU-REQUIRED tests exited
0 and printed nothing about them: ``grep -rn pytest_terminal_summary``
returned 0 hits repo-wide.  This file counts the GPU-caused skips and prints
EXACTLY ONE machine-readable line per session:

    mol-metal GPU contract: N GPU-required test(s) did not run (...)

``N`` is the first integer on the line.  The exit code is never touched.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Set

import pytest


REPO_ROOT = Path(__file__).resolve().parent

#: The environment variable that opts a session back into GPU work.
MOLMETAL_TEST_ALLOW_GPU_ENV = "MOLMETAL_TEST_ALLOW_GPU"

# --- THE CPU LOCK (module level — MUST be before any CUDA context) ----------
# Byte-identical in behaviour to ``molmetal/tests/conftest.py:53-55``.  It is
# duplicated here rather than imported because ``molmetal/tests/conftest.py``
# is NOT an ancestor of the top-level ``tests/`` tree.  The literal
# ``os.environ["CUDA_VISIBLE_DEVICES"] = ""`` is spelled out (not hidden
# behind a constant) so that a grep for the lock finds BOTH conftests.
if os.environ.get(MOLMETAL_TEST_ALLOW_GPU_ENV) != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _cpu_lock_on() -> bool:
    """True when this session has been forced onto the CPU."""
    return os.environ.get(MOLMETAL_TEST_ALLOW_GPU_ENV) != "1"


def _cpu_lock_line() -> str:
    """The one-line, machine-readable statement of the CPU-lock state."""
    if _cpu_lock_on():
        return (
            "mol-metal: CPU lock ON (CUDA_VISIBLE_DEVICES='')"
        )
    return (
        "mol-metal: CPU lock OFF (MOLMETAL_TEST_ALLOW_GPU=1, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r})"
    )


#: The modules that are GPU-REQUIRED: a GPU test that silently runs on CPU
#: is a FALSE GREEN, so these must SKIP cleanly.  This list is the ONE place
#: the contract lives; ``molmetal/tests/test_wfa123_gpu_contract.py`` carries
#: the same list as ``GPU_REQUIRED`` (basenames) and the Wf-A125 contract
#: test asserts the two agree, so a future drift is caught, not hidden.
GPU_REQUIRED_MODULES = (
    "molmetal/tests/test_rocm_egnn.py",
    "molmetal/tests/test_wf25_gpu_smoke.py",
)

#: A skip whose reason names the device is a GPU skip: the test was written
#: to need a GPU and did not get one.  Every GPU-REQUIRED skip reason in this
#: repo names it ("GPU-REQUIRED: no CUDA device visible ...", "CUDA not
#: available", "needs a CUDA device", "GPU unavailable", ...).
_GPU_REASON_RE = re.compile(r"\b(cuda|gpu)\b", re.IGNORECASE)

#: nodeids of the items this conftest marked with the ``gpu`` marker.
_GPU_MARKED_NODEIDS: Set[str] = set()

#: The number of GPU-caused skips observed so far in this session.
_GPU_SKIP_COUNT = 0


def _repo_relative(path: Path) -> str:
    """``path`` relative to the repo root, as a POSIX-ish string."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except (ValueError, OSError):
        return str(path)


def pytest_report_header(config: pytest.Config) -> str:
    """One line stating the CPU-lock state (Wf-A125 / G1)."""
    return _cpu_lock_line()


def pytest_collection_modifyitems(
    config: pytest.Config, items: List[pytest.Item]
) -> None:
    """Mark every item of the GPU-REQUIRED modules with the ``gpu`` marker.

    ``gpu`` is ALREADY declared (``pyproject.toml:151``) and ``--strict-markers``
    is on, so this is a label, not a new marker.  The marker is what lets the
    terminal summary count GPU skips without re-deriving them from strings
    alone.
    """
    for item in items:
        try:
            rel = _repo_relative(item.path)
        except AttributeError:  # pragma: no cover - very old pytest
            rel = str(getattr(item, "fspath", ""))
        if rel in GPU_REQUIRED_MODULES:
            item.add_marker(pytest.mark.gpu)
            _GPU_MARKED_NODEIDS.add(item.nodeid)


def _skip_reason(report: pytest.TestReport) -> str:
    """Extract the human-readable reason of a skipped report.

    ``report.longrepr`` is a tuple ``(path, lineno, reason)`` for a
    ``skipif`` marker and an exception repr for an in-body ``pytest.skip()``;
    both carry the reason text, so ``str()`` of it is enough.
    """
    longrepr = getattr(report, "longrepr", None)
    if longrepr is None:
        return ""
    if isinstance(longrepr, tuple) and len(longrepr) >= 3:
        return str(longrepr[2])
    return str(longrepr)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Count the GPU-caused skips of this session (Wf-A125 / G9).

    A skip counts when EITHER

    * the item carries the ``gpu`` marker this conftest applied, OR
    * the skip reason names the device (``cuda`` / ``gpu``) — that is how a
      module outside ``GPU_REQUIRED_MODULES`` (e.g.
      ``tests/test_cpu_tensor_rejection.py``) announces "I needed a GPU".
    """
    global _GPU_SKIP_COUNT
    if not getattr(report, "skipped", False):
        return
    if report.nodeid in _GPU_MARKED_NODEIDS:
        _GPU_SKIP_COUNT += 1
        return
    if _GPU_REASON_RE.search(_skip_reason(report)):
        _GPU_SKIP_COUNT += 1


def pytest_terminal_summary(
    terminalreporter, exitstatus: int, config: pytest.Config
) -> None:
    """Print EXACTLY ONE machine-readable GPU-contract line (Wf-A125 / G9).

    The exit code is never touched: a CPU-only session that skipped the
    GPU-required tests is a PASSING session, and the line says so.
    """
    header_printed = bool(getattr(terminalreporter, "showheader", True)) and not bool(
        getattr(terminalreporter, "no_header", False)
    )
    if not header_printed:
        # pytest suppresses the whole header block under ``-q`` / ``--no-header``
        # — including ``pytest_report_header``.  Re-state the lock here so the
        # status line is present in EVERY session.
        terminalreporter.write_line(_cpu_lock_line())

    if _cpu_lock_on():
        tail = "CPU-only session; set MOLMETAL_TEST_ALLOW_GPU=1 to enable them"
    else:
        tail = "GPU opt-in session; CUDA may still be unavailable"
    terminalreporter.write_line(
        f"mol-metal GPU contract: {_GPU_SKIP_COUNT} GPU-required test(s) "
        f"did not run ({tail})"
    )
