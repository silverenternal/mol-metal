# SPDX-License-Identifier: Apache-2.0
"""Wf-A269 wave-17 (2026-11-04) — every ``molmetal/scripts/smoke/*.sh``
must carry the repo's standing CPU lock.

THE DEFECT THIS PINS
--------------------
``molmetal/scripts/smoke/smoke_cfm.sh`` (and its four siblings) run
CPU-BY-DESIGN smokes — each script's own header budgets the run as
"<2 min on CPU" — but NONE of them locked the device.  A caller that ran
them without ``CUDA_VISIBLE_DEVICES=""`` therefore got a GPU allocation
even though the smoke never used a GPU:

    MEASURED 2026-11-04 on this host: ONE invocation of a smoke script
    allocated 694 MiB of GPU memory (nvidia-smi
    ``--query-compute-apps`` → ``pid=<smoke child>``, 694 MiB); a
    concurrent pair held 1388 MiB.

The cause is the IMPORT CHAIN, not the smoke.  The chain reaches
``torch.cuda.get_device_capability()``
(``molmetal/adapters/flow_matching_lipman/amp.py:187``), and
``torch.cuda.is_available()`` initialises the CUDA driver (``cuInit``),
which creates the context.  This is the SAME class Wf-A162 repaired
inside the pre-flight (``molmetal/scripts/pre_flight_check.py::_smoke_device``
short-circuits to ``"cpu"`` under the lock, without spawning anything) —
but the pre-flight does NOT reference ``molmetal/scripts/smoke/`` at all
(measured: 0 hits), so the smoke layer had no lock of its own.

THE REPAIR
----------
Each smoke script now exports, immediately after it resolves
``REPO_ROOT``:

    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"
    export MOLMETAL_SMOKE_DEVICE="${MOLMETAL_SMOKE_DEVICE:-cpu}"

``CUDA_VISIBLE_DEVICES`` is exported with the *unset* default, so an
unset value becomes "" (no device visible) while an EXPLICIT value is
preserved — a deliberate GPU smoke is still possible, and the smoke
scripts are not the place to make that decision silently.

WHY A TEST AND NOT JUST A PATCH
-------------------------------
The fix is five lines in five files; nothing about it is hard to undo by
accident.  This file makes the class hard to REINTRODUCE: the assertion
is on the FILE, not on a running GPU, so it is CPU-only, hermetic, and
costs milliseconds.

CPU-only: this file never imports torch and never touches a device.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

#: The repository root — the parent of the ``molmetal`` package directory.
#: Repo-relative by construction (never an absolute path literal): a test
#: that hardcodes a machine path is green only on the machine that wrote
#: it, which is the "author-only red" defect class.
REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_DIR = REPO_ROOT / "molmetal" / "scripts" / "smoke"

#: Every script in the smoke directory must carry the lock.
SMOKE_SCRIPTS = sorted(p for p in SMOKE_DIR.glob("*.sh") if p.is_file())

#: The export lines the repair installs.  Both are required: the first
#: hides the device, the second is the repo's parity kill-switch (honoured
#: by ``metallo_drug_smoke_retrain.py::_resolve_device``).
CUDA_EXPORT_RE = re.compile(
    r'^export\s+CUDA_VISIBLE_DEVICES="\$\{CUDA_VISIBLE_DEVICES[-:]?[^}]*\}"\s*$',
    re.MULTILINE,
)
SMOKE_DEVICE_EXPORT_RE = re.compile(
    r'^export\s+MOLMETAL_SMOKE_DEVICE="\$\{MOLMETAL_SMOKE_DEVICE[-:]?[^}]*\}"\s*$',
    re.MULTILINE,
)

#: A smoke script that asks for a GPU device defeats the lock it just
#: installed, so the literal is banned outright.
CUDA_DEVICE_LITERALS = ("--device cuda", "--device=cuda", 'device="cuda', "device='cuda")


def test_the_smoke_directory_is_not_empty():
    """If this file's glob silently matched nothing, every test below would
    pass vacuously — the same "no silent caps" failure the gate census
    keeps rediscovering.  Assert the population explicitly."""
    assert SMOKE_SCRIPTS, f"no smoke scripts found under {SMOKE_DIR}"


@pytest.mark.parametrize("script", SMOKE_SCRIPTS, ids=lambda p: p.name)
def test_smoke_script_exports_the_cpu_lock(script: Path):
    """The lock must be EXPORTED (a bare assignment would not reach the
    child process), and both halves of it must be present."""
    src = script.read_text(encoding="utf-8")
    assert CUDA_EXPORT_RE.search(src), (
        f"{script.name} does not export CUDA_VISIBLE_DEVICES — a caller that "
        f"runs it without the lock pays a real GPU allocation (MEASURED "
        f"694 MiB per invocation, 2026-11-04)"
    )
    assert SMOKE_DEVICE_EXPORT_RE.search(src), (
        f"{script.name} does not export MOLMETAL_SMOKE_DEVICE — the repo's "
        f"parity kill-switch is missing"
    )


@pytest.mark.parametrize("script", SMOKE_SCRIPTS, ids=lambda p: p.name)
def test_the_lock_is_installed_before_the_first_python_child(script: Path):
    """An export placed AFTER the child is launched protects nothing.

    The smoke scripts launch their Python child through a venv variable
    (``${PY}``), so the child is located by the first occurrence of that
    token rather than by a literal ``python``.
    """
    src = script.read_text(encoding="utf-8")
    export_at = src.find("export CUDA_VISIBLE_DEVICES=")
    assert export_at != -1, f"{script.name}: no CUDA export (see the other test)"

    # The first launch of the Python child: the venv invocation.
    child_at = src.find('"${PY}"')
    if child_at == -1:
        # Fall back to any python invocation for scripts that do not use
        # the shared-venv variable.
        child_at = src.find("python ")
    if child_at == -1:
        pytest.skip(f"{script.name}: no Python child invocation detected")

    assert export_at < child_at, (
        f"{script.name} installs the CPU lock at offset {export_at}, AFTER "
        f"the Python child is launched at offset {child_at} — the child "
        f"would not inherit it"
    )


@pytest.mark.parametrize("script", SMOKE_SCRIPTS, ids=lambda p: p.name)
def test_no_smoke_script_asks_for_a_cuda_device(script: Path):
    """The lock and a hard-coded CUDA device cannot both be honoured."""
    src = script.read_text(encoding="utf-8")
    for literal in CUDA_DEVICE_LITERALS:
        assert literal not in src, (
            f"{script.name} contains {literal!r} — a CPU-locked smoke that "
            f"asks for a CUDA device is self-contradictory"
        )


@pytest.mark.parametrize("script", SMOKE_SCRIPTS, ids=lambda p: p.name)
def test_the_cuda_export_defaults_to_no_visible_device(script: Path):
    """BEHAVIOURAL check, not a text check: evaluate the script's own
    export line with the variable UNSET and assert the result is empty.

    This is what makes the lock load-bearing — it is the difference
    between "the line exists" and "the line hides the device".
    """
    src = script.read_text(encoding="utf-8")
    match = CUDA_EXPORT_RE.search(src)
    assert match, f"{script.name}: no CUDA export line to evaluate"
    line = match.group(0)

    proc = subprocess.run(
        ["bash", "-c", f"unset CUDA_VISIBLE_DEVICES; {line}; printf '%s' \"$CUDA_VISIBLE_DEVICES\""],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"{script.name}: export line failed: {proc.stderr}"
    assert proc.stdout == "", (
        f"{script.name}: with CUDA_VISIBLE_DEVICES unset the lock evaluates "
        f"to {proc.stdout!r}, not '' — the device stays visible"
    )


#: The bare-``python`` interpreter fallback that AGENTS.md §5.9 forbids.
#:
#: §5.9: "``scripts/launch_*.sh`` MUST use
#: ``/home/hugo/codes/mol-metal/.venv/bin/python``.  NO bare ``python``, NO
#: ``CadButEaas``/``kanzi_venv``."  The smoke scripts ran
#: ``PY="${PYTHON:-python}"`` as the last resort — i.e. when the repo venv
#: was missing, they silently fell back to the AMBIENT interpreter, which
#: on this host may be another project's environment.  A smoke that measures
#: under an unpinned interpreter is a measurement nobody may quote
#: (AGENTS.md §5.2), so the fallback is now fail-CLOSED: an explicit
#: ``PYTHON=`` override is honoured, a bare ``python`` is not.
BARE_PYTHON_FALLBACK = "PYTHON:-python"


@pytest.mark.parametrize("script", SMOKE_SCRIPTS, ids=lambda p: p.name)
def test_no_smoke_script_falls_back_to_a_bare_python(script: Path):
    """AGENTS.md §5.9: the single venv, or nothing — never a bare `python`."""
    src = script.read_text(encoding="utf-8")
    assert BARE_PYTHON_FALLBACK not in src, (
        f"{script.name} still falls back to a bare `python` "
        f"({BARE_PYTHON_FALLBACK!r}) — AGENTS.md §5.9 forbids it: a smoke that "
        f"silently runs under the ambient interpreter measures an unpinned "
        f"environment, and the ambient interpreter may belong to another "
        f"project (kanzi_venv / CadButEaas)"
    )


@pytest.mark.parametrize("script", SMOKE_SCRIPTS, ids=lambda p: p.name)
def test_the_smoke_interpreter_is_fail_closed(script: Path):
    """Where the script resolves its interpreter, a missing venv must be a
    hard stop with a diagnostic — not a silent downgrade."""
    src = script.read_text(encoding="utf-8")
    if "SHARED_VENV=" not in src:
        pytest.skip(f"{script.name}: no interpreter block of its own")
    assert ".venv/bin/python" in src, (
        f"{script.name} does not name the repo venv (AGENTS.md §5.9)"
    )
    assert re.search(r"^\s*exit 1\s*$", src, re.MULTILINE), (
        f"{script.name}: the interpreter block has no fail-closed `exit 1` — "
        f"a missing venv must stop the smoke, not downgrade it"
    )
