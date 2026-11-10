# SPDX-License-Identifier: Apache-2.0
"""Wf-A162 — the pre-flight's GPU-visibility probe must not allocate GPU
memory, and the smoke gates must not silently train on a forbidden device.

WHAT THIS FILE PINS
-------------------
``molmetal/scripts/pre_flight_check.py`` used to answer "does this host
have a GPU?" by spawning ``import torch; print(torch.cuda.is_available())``
with **no ``env=``**, so the child inherited the ambient environment.  In
the installed torch that call takes the ``else`` branch of ``is_available()``
whose own comment says it "uses the CUDA Runtime API ``cudaGetDeviceCount``
which in turn initializes the CUDA Driver API via ``cuInit``" -- i.e. the
question itself allocated GPU memory.

MEASURED (2026-10-31): the GPU watchdog recorded exactly ONE mol-metal
allocation in its life -- 550 MiB -- and no ``.pt`` file was written that
day, so no training ran; what allocated was the visibility query.

The probe also IGNORED the repo's standing CPU lock, which the same file
already computed for the GPU-REQUIRED gate
(``_session_cuda_locked = os.environ.get("CUDA_VISIBLE_DEVICES", None) == ""``).
And because the two Wf-446 smoke gates feed the probe's answer straight to
a trainer's ``--device``, the pre-flight could **silently start a 2-epoch
GPU training** in a session that forbade GPU.

THE REPAIR (Wf-A162)
--------------------
1. ``_smoke_device()`` short-circuits to ``"cpu"`` when the session blanks
   ``CUDA_VISIBLE_DEVICES`` -- **without spawning torch at all**.  The
   probe would have answered ``False`` anyway, so this is
   behaviour-identical and allocation-free.
2. When the probe does run, it carries ``PYTORCH_NVML_BASED_CUDA_CHECK=1``
   so the answer comes from NVML rather than ``cuInit``.  NVML does not
   honour the device mask, which is why (1) is the load-bearing guard and
   why a mask that hides every device is short-circuited as well.
3. Both consuming gates assert the device against
   ``_device_authorisation_error(...)`` and FAIL loudly -- the smoke is
   NOT run -- when the session forbids the device.

CPU-only.  No GPU is touched, queried or trained on by this file: the
probe is faked, and the only real subprocess is none at all.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from molmetal.scripts import pre_flight_check as pf

#: The repository root -- the parent of the ``molmetal`` package directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
PF_PATH = REPO_ROOT / "molmetal" / "scripts" / "pre_flight_check.py"


@pytest.fixture(autouse=True)
def _reset_smoke_device_cache(monkeypatch):
    """The probe is cached process-wide; every test starts from a cold
    cache so one test's answer cannot leak into the next.

    Wf-A269 wave-12 (2026-11-02): the ambient ``MOLMETAL_SMOKE_DEVICE`` is
    cleared too, for the same reason.  ``_smoke_device()`` honours that
    variable (the parity kill-switch with
    ``metallo_drug_smoke_retrain.py::_resolve_device``), so an operator
    who exports ``MOLMETAL_SMOKE_DEVICE=cpu`` -- the natural way to honour
    this repo's 0-GPU contract -- would otherwise make the two tests below
    fail on an AMBIENT variable rather than on a defect: the NVML probe is
    then never reached (the override short-circuits it) and the
    "the host has a GPU" case is answered ``cpu``.  A unit under test must
    be isolated from the operator's environment, or it measures the
    environment and calls it a defect.
    """
    monkeypatch.delenv("MOLMETAL_SMOKE_DEVICE", raising=False)
    pf._SMOKE_DEVICE_CACHE = None
    yield
    pf._SMOKE_DEVICE_CACHE = None


# ---------------------------------------------------------------------------
# 1. THE LOAD-BEARING GUARD -- under the lock, no subprocess at all
# ---------------------------------------------------------------------------


def test_under_the_cpu_lock_the_probe_spawns_no_subprocess(monkeypatch):
    """The single most important assertion in this file.

    Under the repo's standing CPU lock the probe must answer ``"cpu"``
    WITHOUT spawning anything -- that is what removes the allocation
    entirely, rather than merely making the allocation smaller.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    def _boom(*_args, **_kwargs):
        raise AssertionError(
            "the GPU-visibility probe spawned a subprocess while the "
            "session blanks CUDA_VISIBLE_DEVICES"
        )

    monkeypatch.setattr(pf.subprocess, "run", _boom)
    assert pf._smoke_device() == "cpu"


def test_under_the_cpu_lock_the_probe_does_not_even_ask_torch(monkeypatch):
    """The short-circuit happens BEFORE the torch probe, not after it."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    def _boom(*_args, **_kwargs):
        raise AssertionError("torch was imported by the probe under the lock")

    monkeypatch.setattr(pf.subprocess, "run", _boom)
    assert pf._smoke_device() == "cpu"
    assert pf._SMOKE_DEVICE_CACHE == "cpu"


# ---------------------------------------------------------------------------
# 2. The belt -- NVML instead of cuInit when the probe does run
# ---------------------------------------------------------------------------


def test_the_probe_asks_nvml_rather_than_cuinit(monkeypatch):
    """When the probe runs at all it must carry
    ``PYTORCH_NVML_BASED_CUDA_CHECK=1``, so ``is_available()`` is answered
    by NVML (no ``cuInit``) instead of by the CUDA Runtime API."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    seen: dict = {}

    class _Proc:
        returncode = 0
        stdout = "False\n"
        stderr = ""

    def _fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["env"] = kwargs.get("env")
        return _Proc()

    monkeypatch.setattr(pf.subprocess, "run", _fake_run)
    assert pf._smoke_device() == "cpu"
    assert seen["env"] is not None, "the probe ran without an explicit env="
    assert seen["env"]["PYTORCH_NVML_BASED_CUDA_CHECK"] == "1"


def test_the_probe_still_reports_a_gpu_when_the_host_has_one(monkeypatch):
    """The repair must not break the normal (unlocked) behaviour."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    class _Proc:
        returncode = 0
        stdout = "True\n"
        stderr = ""

    monkeypatch.setattr(pf.subprocess, "run", lambda cmd, **kw: _Proc())
    assert pf._smoke_device() == "cuda:0"


# ---------------------------------------------------------------------------
# 2b. THE PARITY KILL-SWITCH -- MOLMETAL_SMOKE_DEVICE (Wf-A269 wave-12)
# ---------------------------------------------------------------------------


def test_the_cpu_override_forces_the_cpu_without_probing(monkeypatch):
    """``MOLMETAL_SMOKE_DEVICE=cpu`` must answer ``"cpu"`` WITHOUT spawning
    the probe at all -- the whole point of the switch is that a pre-flight
    can be told to stay off the GPU without touching the GPU."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("MOLMETAL_SMOKE_DEVICE", "cpu")

    def _boom(*_args, **_kwargs):
        raise AssertionError(
            "the GPU-visibility probe spawned a subprocess even though "
            "MOLMETAL_SMOKE_DEVICE=cpu"
        )

    monkeypatch.setattr(pf.subprocess, "run", _boom)
    assert pf._smoke_device() == "cpu"


def test_the_cuda_override_asks_for_the_gpu(monkeypatch):
    """The other half of the switch: ``cuda`` must be honoured when the
    session does NOT forbid the GPU."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("MOLMETAL_SMOKE_DEVICE", "cuda")

    def _boom(*_args, **_kwargs):
        raise AssertionError(
            "the GPU-visibility probe spawned a subprocess even though "
            "MOLMETAL_SMOKE_DEVICE=cuda"
        )

    monkeypatch.setattr(pf.subprocess, "run", _boom)
    assert pf._smoke_device() == "cuda:0"


def test_the_lock_overrides_a_cuda_request(monkeypatch, capsys):
    """TWO DIRECTIVES THAT CONTRADICT EACH OTHER.

    ``MOLMETAL_SMOKE_DEVICE=cuda`` asks for the GPU; the standing CPU lock
    (``CUDA_VISIBLE_DEVICES=""``) forbids it.  **THE LOCK WINS** -- and the
    conflict is ANNOUNCED, because an override that is silently ignored is
    indistinguishable from one that was never read.

    Why the lock and not the override: the two consuming gates already
    call ``_device_authorisation_error`` and FAIL outright, so letting the
    override through would turn a contradictory instruction into a
    spurious FAIL of an unrelated trainer-smoke gate.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("MOLMETAL_SMOKE_DEVICE", "cuda")
    assert pf._smoke_device() == "cpu"
    err = capsys.readouterr().err
    assert "MOLMETAL_SMOKE_DEVICE=cuda" in err
    assert "overridden to cpu" in err


def test_the_override_is_normalised(monkeypatch):
    """Whitespace and case must not silently disable the switch -- an
    unrecognised value is IGNORED (the probe decides), never treated as
    ``cpu``."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("MOLMETAL_SMOKE_DEVICE", "  CPU  ")
    assert pf._smoke_device() == "cpu"


def test_an_unrecognised_override_is_ignored(monkeypatch):
    """``MOLMETAL_SMOKE_DEVICE=nonsense`` must not be read as ``cpu``.

    A typo in a kill-switch must not silently disable a GPU allocation:
    the value is not one of the two accepted literals, so the switch does
    not fire and the probe decides (here: a GPU is visible).
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("MOLMETAL_SMOKE_DEVICE", "nonsense")

    class _Proc:
        returncode = 0
        stdout = "True\n"
        stderr = ""

    monkeypatch.setattr(pf.subprocess, "run", lambda cmd, **kw: _Proc())
    assert pf._smoke_device() == "cuda:0"


def test_the_override_announces_itself_on_stderr_not_stdout(monkeypatch, capsys):
    """The announcement must not pollute stdout.

    The gate consumers parse this pre-flight's stdout; a device line
    injected into stdout could be parsed as a gate's own output.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("MOLMETAL_SMOKE_DEVICE", "cpu")
    assert pf._smoke_device() == "cpu"
    captured = capsys.readouterr()
    assert "MOLMETAL_SMOKE_DEVICE" in captured.err
    assert "MOLMETAL_SMOKE_DEVICE" not in captured.out


# ---------------------------------------------------------------------------
# 3. The authorisation check -- the gates must FAIL loudly
# ---------------------------------------------------------------------------


def test_a_locked_session_forbids_a_gpu_device(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert pf._device_authorisation_error("cpu") is None
    err = pf._device_authorisation_error("cuda:0")
    assert err is not None
    assert "CUDA_VISIBLE_DEVICES" in err


def test_a_mask_that_hides_every_device_forbids_a_gpu_device(monkeypatch):
    """``CUDA_VISIBLE_DEVICES=-1`` hides every device without blanking the
    variable -- the authorisation check must catch it too."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert pf._visible_device_count_hint() == 0
    assert pf._device_authorisation_error("cpu") is None
    assert pf._device_authorisation_error("cuda:0") is not None


def test_an_unlocked_session_authorises_a_gpu_device(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert pf._visible_device_count_hint() == 1
    assert pf._device_authorisation_error("cuda:0") is None


def test_the_visible_device_hint_reads_the_mask(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    assert pf._visible_device_count_hint() == 3
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,-1")
    assert pf._visible_device_count_hint() == 1
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert pf._visible_device_count_hint() is None


# ---------------------------------------------------------------------------
# 4. THE STRUCTURAL GUARD -- no un-guarded call site may reappear
# ---------------------------------------------------------------------------


def test_every_smoke_device_call_site_is_guarded():
    """A source-level guard on the CLASS of defect, not just this instance.

    Every real call of ``_smoke_device()`` must live in a function that
    also calls ``_device_authorisation_error(...)``.  A new smoke gate
    that forgot the check fails this test.

    AST-based on purpose: the module's PROSE mentions ``_smoke_device()``
    and ``torch.cuda.is_available()`` in docstrings and comments, and a
    text scan would flag the documentation instead of the code.
    """
    src = PF_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)

    call_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_smoke_device"
    ]
    assert call_lines, (
        "no _smoke_device() call site found -- this guard would be vacuous"
    )

    lines = src.splitlines()
    unguarded: list = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = "\n".join(lines[func.lineno - 1 : func.end_lineno])
        sites = [ln for ln in call_lines if func.lineno <= ln <= func.end_lineno]
        if not sites:
            continue
        if "_device_authorisation_error(" not in body:
            unguarded.append(f"{func.name} (line {func.lineno})")
    assert not unguarded, (
        f"_smoke_device() used without the authorisation check in: "
        f"{unguarded}"
    )


def test_the_probe_is_never_called_without_an_explicit_env():
    """No probe may run without an explicit ``env=``, so it cannot inherit
    the ambient environment.

    AST-based: the probe is identified by the ``subprocess.run`` call whose
    argument list carries the ``import torch;print(torch.cuda.is_available())``
    string literal -- not by a text scan over the whole file.
    """
    src = PF_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)

    probe_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        args = node.args
        if not args:
            continue
        literal = None
        if isinstance(args[0], ast.Constant) and isinstance(args[0].value, str):
            literal = args[0].value
        elif (
            isinstance(args[0], (ast.List, ast.Tuple))
            and args[0].elts
            and isinstance(args[0].elts[-1], ast.Constant)
            and isinstance(args[0].elts[-1].value, str)
        ):
            literal = args[0].elts[-1].value
        if literal and "torch.cuda.is_available()" in literal:
            probe_calls.append(node)

    assert probe_calls, (
        "the GPU-visibility probe was not found -- this guard would be "
        "vacuous; re-check it before trusting it"
    )
    for call in probe_calls:
        has_env = any(kw.arg == "env" for kw in call.keywords)
        assert has_env, (
            f"the GPU probe at line {call.lineno} runs without an explicit "
            f"env=, so it inherits the ambient environment"
        )
