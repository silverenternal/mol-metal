# SPDX-License-Identifier: Apache-2.0
"""Wf-A269 wave-18 (2026-11-04) — the pre-flight must pin the CPU thread
count of the gate children it runs in parallel.

THE DEFECT THIS PINS
--------------------
``pre_flight_check.py`` runs its gates in a
``ThreadPoolExecutor(max_workers=jobs)`` with ``jobs = min(8,
os.cpu_count())`` — 8 on this 32-core host.  Each gate that shells out to a
torch trainer spawns a CHILD process, and the child inherits the parent's
environment — including the thread-count knobs.  Nothing pinned them, so
every child took torch's default:

    MEASURED 2026-11-04 on this host:
        os.cpu_count()                  = 32
        torch.get_num_threads()         = 24
        torch.get_num_interop_threads() = 24

=> 8 concurrent gates x 24 threads = 192 runnable threads on 32 cores.
MEASURED with 8 concurrent copies of the exact command the
``p1_chemprop_smoke`` gate runs:

    arm                        epoch-1 elapsed (8/8 jobs)
    default threads (24 each)  260.7 .. 267.1 s
    OMP_NUM_THREADS=2          0.6  .. 1.0    s

That is where the ``p1_chemprop_smoke`` TIMEOUT at 137.75 s came from: its
own log line reported ``epoch=0/2 ... elapsed=106.5 s`` while the SAME smoke
measured 5.1 s end-to-end standalone.  The gate was not slow — it was
starved by its seven sibling gates.

THE REPAIR
----------
``pre_flight_check._pin_gate_child_thread_caps(jobs)`` exports
``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` / ``OPENBLAS_NUM_THREADS`` into
the parent's environment before the executor starts, at
``max(1, cpu_count // jobs)``.  ``MOLMETAL_PREFLIGHT_OMP_THREADS``
overrides the computed value; a non-integer override is IGNORED (a typo
must not silently change the machine's parallelism).

CPU-only: this file never imports torch in-process.  The one test that
needs to observe torch's thread count spawns a CHILD with
``CUDA_VISIBLE_DEVICES=""`` and reads the child's own report.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

#: The repository root — the parent of the ``molmetal`` package directory.
#: Repo-relative by construction (never an absolute path literal): a test
#: that hardcodes a machine path is green only on the machine that wrote it.
REPO_ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = REPO_ROOT / "molmetal" / "scripts" / "pre_flight_check.py"


@pytest.fixture(scope="module")
def preflight():
    """Import the pre-flight module WITHOUT importing torch.

    The module is imported by path (not via ``import molmetal...``) so the
    test does not depend on the package's ``__init__`` side effects.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "wfa269_pf_thread_caps", PREFLIGHT
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_cap_is_installed_before_the_executor_starts(preflight) -> None:
    """An export placed AFTER the executor starts protects nothing.

    Source ORDER is the contract here: the caps must be pinned before the
    ``ThreadPoolExecutor`` is constructed, because every child inherits the
    environment as it is at SPAWN time.

    The comparison is done on the AST, not on the raw text: the module's own
    comment block mentions ``ThreadPoolExecutor(max_workers=jobs)``, so a
    text search would find the COMMENT rather than the construction.  (A
    guard that reads its own documentation as its code is a guard that
    passes for the wrong reason.)
    """
    import ast

    tree = ast.parse(PREFLIGHT.read_text(encoding="utf-8"))
    cap_lines: list = []
    pool_lines: list = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.attr
            if isinstance(func, ast.Attribute)
            else (func.id if isinstance(func, ast.Name) else None)
        )
        if name == "_pin_gate_child_thread_caps":
            cap_lines.append(node.lineno)
        elif name == "ThreadPoolExecutor":
            pool_lines.append(node.lineno)

    assert cap_lines, "the pre-flight never calls _pin_gate_child_thread_caps"
    assert pool_lines, "the pre-flight no longer constructs a ThreadPoolExecutor"
    assert min(cap_lines) < min(pool_lines), (
        f"the thread caps are pinned at line {min(cap_lines)}, AFTER the "
        f"executor is constructed at line {min(pool_lines)} — the children "
        f"would inherit the uncapped thread count"
    )


def test_the_cap_is_a_positive_integer(preflight) -> None:
    """The cap must never be 0 (that would serialise the child) and never
    negative."""
    for jobs in (1, 2, 3, 8, 64):
        caps = preflight._pin_gate_child_thread_caps(jobs)
        for name, value in caps.items():
            assert value.isdigit(), f"{name}={value!r} is not an integer"
            assert int(value) >= 1, f"{name}={value!r} would serialise the child"


def test_the_cap_is_the_fair_share_of_the_machine(preflight) -> None:
    """``cpu_count // jobs``, floored at 1."""
    n_cores = os.cpu_count() or 1
    for jobs in (1, 2, 4, 8, 64):
        expected = max(1, n_cores // jobs)
        caps = preflight._pin_gate_child_thread_caps(jobs)
        assert caps["OMP_NUM_THREADS"] == str(expected), (
            f"jobs={jobs}: expected a cap of {expected}, got {caps}"
        )


def test_one_job_keeps_the_whole_machine(preflight) -> None:
    """A serial run must not be slowed down by the repair."""
    n_cores = os.cpu_count() or 1
    caps = preflight._pin_gate_child_thread_caps(1)
    assert caps["OMP_NUM_THREADS"] == str(n_cores)


def test_the_cap_covers_every_standard_knob(preflight) -> None:
    """OMP alone is not enough: MKL backs Intel GEMM and OPENBLAS backs the
    BLAS numpy ships in manylinux wheels."""
    caps = preflight._pin_gate_child_thread_caps(8)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert name in caps, f"{name} is not pinned"


def test_the_cap_reaches_the_child_environment(preflight, monkeypatch) -> None:
    """BEHAVIOURAL: the values must be in ``os.environ`` — that is what a
    spawned child inherits."""
    for name in preflight.THREAD_CAP_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    caps = preflight._pin_gate_child_thread_caps(8)
    for name, value in caps.items():
        assert os.environ.get(name) == value


def test_an_explicit_override_is_honoured(preflight, monkeypatch) -> None:
    monkeypatch.setenv(preflight.ENV_THREAD_CAP_OVERRIDE, "7")
    caps = preflight._pin_gate_child_thread_caps(8)
    assert caps["OMP_NUM_THREADS"] == "7"


def test_a_non_integer_override_is_ignored(preflight, monkeypatch, capsys) -> None:
    """A typo must not silently change the machine's parallelism."""
    monkeypatch.setenv(preflight.ENV_THREAD_CAP_OVERRIDE, "banana")
    expected = max(1, (os.cpu_count() or 1) // 8)
    caps = preflight._pin_gate_child_thread_caps(8)
    assert caps["OMP_NUM_THREADS"] == str(expected)
    assert "not an integer" in capsys.readouterr().err


def test_the_child_reports_the_pinned_thread_count(preflight) -> None:
    """END-TO-END: a spawned child must see torch honour the cap.

    This is the difference between "the env var is set" and "the thread
    count is actually capped" — the defect was never the missing variable,
    it was the missing *effect*.
    """
    cap = preflight._pin_gate_child_thread_caps(8)["OMP_NUM_THREADS"]
    probe = (
        "import json, os, torch\n"
        "print(json.dumps({\n"
        "  'omp': os.environ.get('OMP_NUM_THREADS'),\n"
        "  'torch_threads': torch.get_num_threads(),\n"
        "}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        timeout=300,
    )
    assert proc.returncode == 0, f"the child probe failed: {proc.stderr[-2000:]}"
    payload = proc.stdout.strip().splitlines()[-1]
    import json as _json

    data = _json.loads(payload)
    assert data["omp"] == cap, f"the child did not inherit the cap: {data}"
    assert data["torch_threads"] == int(cap), (
        f"torch ignored the cap: {data} — the env var is set but the thread "
        f"count is not pinned"
    )
