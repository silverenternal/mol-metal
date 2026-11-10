# SPDX-License-Identifier: Apache-2.0
"""Wf-A269 wave-18 (2026-11-04) — the pitfall-guard runner must run its
pytest under the interpreter that runs the runner.

THE DEFECT THIS PINS
--------------------
``molmetal/scripts/wf_pitfall_guard/check_all.py::_run_pytest`` preferred the
BARE ``pytest`` on PATH, with the docstring claiming that this "lands in the
project's managed virtualenv where the test deps are installed".  It does
the opposite: a bare ``pytest`` lands in WHATEVER environment provides it.

MEASURED 2026-11-04 on this host::

    $ which pytest
    /home/linuxbrew/.linuxbrew/opt/python@3.14/bin/python3.14
    $ molmetal/scripts/smoke/smoke_pitfalls.sh
    [smoke_pitfalls] FAIL runner_crashed rc=2 after 1s
    FATAL: pytest crashed (rc=4) in 0.4s:
    ImportError while loading conftest 'molmetal/tests/conftest.py'.
    molmetal/tests/conftest.py:269: in _force_torch_scatter_backend_for_cpu_tests
        import torch
    E   ModuleNotFoundError: No module named 'torch'

⇒ the smoke reported a FAIL that had nothing to do with the seven pitfall
guards it exists to check: the repo's own ``conftest.py`` could not import
torch under a Homebrew python 3.14.  AGENTS.md §5.9 says the repo runs
under a SINGLE venv, so the interpreter that runs the tests must be the
interpreter that runs the runner — ``sys.executable -m pytest``.

The bare form is kept as a LAST resort, and the loop returns as soon as a
run produced output; a crashed pytest always produces output, so that
output is a VERDICT and must be returned rather than retried under a
different interpreter.

WHY A TEST AND NOT JUST A PATCH
-------------------------------
The ordering is two list entries; nothing about it is hard to undo by
accident.  This file makes the CLASS hard to reintroduce, and it is
CPU-only and hermetic: it never runs pytest.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

#: The repository root — the parent of the ``molmetal`` package directory.
#: Repo-relative by construction (never an absolute path literal): a test
#: that hardcodes a machine path is green only on the machine that wrote it.
REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "molmetal" / "scripts" / "wf_pitfall_guard" / "check_all.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("wfa269_check_all", RUNNER)
    assert spec is not None and spec.loader is not None, "the runner is unreadable"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_runner_file_exists():
    """If this path were wrong, every test below would pass vacuously."""
    assert RUNNER.is_file(), f"the pitfall-guard runner is missing at {RUNNER}"


def test_the_pinned_interpreter_is_the_primary_command():
    """``sys.executable -m pytest`` must come FIRST in the command list."""
    mod = _load_runner()
    recorded: list[list[str]] = []

    class _Proc:
        returncode = 0
        stdout = "ok\n"
        stderr = ""

    def _fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
        recorded.append(list(cmd))
        return _Proc()

    original = mod.subprocess.run
    mod.subprocess.run = _fake_run
    try:
        mod._run_pytest()
    finally:
        mod.subprocess.run = original

    assert recorded, "the runner never invoked pytest"
    first = recorded[0]
    assert first[0] == sys.executable and first[1:2] == ["-m"], (
        f"the FIRST pytest invocation is {first!r} — it must be "
        f"[sys.executable, '-m', 'pytest', ...] so the tests run under the "
        f"interpreter that runs the runner (AGENTS.md §5.9)"
    )
    assert first[2] == "pytest", f"the first invocation is not a pytest call: {first!r}"


def test_the_bare_pytest_is_only_a_last_resort():
    """The bare form may still exist, but never ahead of the pinned one."""
    src = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(src)
    pinned_at = src.index('sys.executable, "-m", "pytest"')
    bare_at = src.find('["pytest", rel')
    assert bare_at != -1, "the bare fallback is gone entirely — acceptable, but note it"
    assert bare_at > pinned_at, (
        "the bare `pytest` precedes the pinned interpreter — that is the "
        "defect this file pins"
    )
    assert tree is not None


def test_the_loop_returns_on_any_output_so_a_crash_is_a_verdict():
    """A CRASHED pytest always produces output; that output must be
    RETURNED, not retried under another interpreter.

    The old code tried the bare ``pytest`` first and returned on ANY
    output — so the crash it produced (rc=4, stderr non-empty) was
    returned as the verdict and the pinned fallback was never reached.
    """
    mod = _load_runner()
    attempts: list[list[str]] = []

    class _Proc:
        returncode = 4
        stdout = ""
        stderr = "FATAL: pytest crashed (rc=4)\n"

    def _fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
        attempts.append(list(cmd))
        return _Proc()

    original = mod.subprocess.run
    mod.subprocess.run = _fake_run
    try:
        rc, out, err = mod._run_pytest()
    finally:
        mod.subprocess.run = original

    assert len(attempts) == 1, (
        f"the runner retried a CRASHED pytest under a second interpreter "
        f"({len(attempts)} attempts) — a crash is a verdict, not an "
        f"invitation to retry"
    )
    assert rc == 4
    assert "FATAL" in err


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
