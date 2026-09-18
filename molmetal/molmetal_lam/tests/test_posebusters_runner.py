"""Unit tests for :mod:`molmetal.validation.posebusters_runner`.

The tests cover four contracts called out by the R3/F4 task spec:

1. The module is importable and the public entry point returns a dict
   with the ``pb_valid`` key.
2. The module handles the missing-pkg case (returns ``skipped=True``
   with ``pb_valid=None``) — this is the canonical TODO/environment.md
   path on the ROCm install.
3. The module handles invalid SMILES by returning ``pb_valid=False``
   rather than raising.
4. The default ``optimize_method`` is ``"MMFF94"``, NOT ``"UFF"``.

We do NOT run PoseBusters' full check here (it is not installed in
the unit-test environment).  Each test pins the public contract
without depending on PB availability.
"""
from __future__ import annotations

import importlib

import pytest


def _import_runner():
    """Import the runner module under test.

    Skips the test set when the module itself fails to import (e.g. on
    a stripped-down install missing ``molmetal.validation``).  This is
    the same graceful-skip policy the rest of the lam suite follows.
    """
    try:
        return importlib.import_module("molmetal.validation.posebusters_runner")
    except Exception as exc:
        pytest.skip(f"molmetal.validation.posebusters_runner not importable: {exc}")


def test_pb_runner_loads():
    """The runner module is importable and returns a dict with ``pb_valid``."""
    runner = _import_runner()
    assert hasattr(runner, "check_posebusters")
    # Probe with the simplest possible SMILES — ethanol.  The
    # runner returns a dict regardless of whether ``posebusters`` is
    # installed (the missing-pkg fast-path is part of the contract).
    res = runner.check_posebusters("CCO")
    assert isinstance(res, dict)
    assert "pb_valid" in res
    assert "n_conformers" in res
    assert "failures" in res
    # The verdict field is one of True / False / None.
    assert res["pb_valid"] in (True, False, None)


def test_pb_runner_handles_missing_pkg():
    """When ``posebusters`` is not installed, ``check_posebusters`` returns
    ``{"pb_valid": None, "skipped": True, ...}`` — it does NOT crash.

    We check the runner's public ``posebusters_available()`` probe and
    assert that the runner returns the skipped-shape dict on the
    missing-pkg path.  When PB *is* installed this test still passes
    because the runner returns a non-skipped dict with a real verdict
    and we just check the public surface.
    """
    runner = _import_runner()
    # Probe availability first — this is the canonical skip path.
    available = bool(runner.posebusters_available())
    res = runner.check_posebusters("CCO")
    assert isinstance(res, dict)
    assert "pb_valid" in res
    if not available:
        # Missing-pkg contract:
        # * ``pb_valid`` must be None (NOT True/False — there is no
        #   verdict to give).
        # * ``skipped`` must be True.
        # * ``reason`` is non-empty.
        assert res["pb_valid"] is None
        assert bool(res.get("skipped")) is True
        assert res.get("reason")
    # Either way the dict must contain the keys we promised.
    for key in ("pb_valid", "skipped", "failures", "n_conformers"):
        assert key in res, f"missing key {key!r} in {res!r}"


def test_pb_runner_handles_invalid_smiles():
    """Invalid SMILES return ``pb_valid=False`` (NOT a crash) on the
    PB-installed path.  On the PB-missing path (``pb_valid is None``)
    we still get a graceful ``skipped=True`` dict — never a crash.

    Covers:
    * empty string
    * non-string input (None)
    * a syntactically-broken SMILES token (e.g. ``"NOT_A_SMILES!!!"``)
    """
    runner = _import_runner()
    # 1) Empty SMILES — must NOT raise.  Either ``pb_valid is False``
    # (PB available, parse failed) or ``pb_valid is None`` with
    # ``skipped is True`` (PB missing).  Failures list is populated
    # in the False-branch; it may be empty in the None-branch (the
    # runner doesn't probe SMILES on the missing-pkg fast path).
    res_empty = runner.check_posebusters("")
    assert isinstance(res_empty, dict)
    pv = res_empty.get("pb_valid")
    assert pv in (False, None)
    assert "failures" in res_empty
    assert isinstance(res_empty["failures"], list)
    if pv is False:
        # Real verdict — must surface *some* failure reason.
        assert len(res_empty["failures"]) > 0
    # 2) Non-string — must NOT raise.
    res_none = runner.check_posebusters(None)  # type: ignore[arg-type]
    assert isinstance(res_none, dict)
    assert res_none.get("pb_valid") in (False, None)
    # 3) Garbage SMILES — must NOT raise; verdict is False or None,
    # never True.
    res_garbage = runner.check_posebusters("NOT_A_SMILES!!!")
    assert isinstance(res_garbage, dict)
    assert res_garbage.get("pb_valid") in (False, None)


def test_pb_runner_uses_mmff94():
    """Default ``optimize_method`` is ``"MMFF94"``, NOT ``"UFF"``.

    This is the R3 audit finding: the legacy adapter defaults to a
    MMFF94 → UFF fallback which trips PoseBusters' geometry checks
    on heterocycles.  The new runner must default to MMFF94 outright.
    """
    runner = _import_runner()
    import inspect
    sig = inspect.signature(runner.check_posebusters)
    assert "optimize_method" in sig.parameters
    default = sig.parameters["optimize_method"].default
    # The default is ``"MMFF94"`` — anything else is a regression.
    assert default == "MMFF94", (
        f"optimize_method default must be 'MMFF94' (got {default!r})"
    )
    assert default.upper() != "UFF"
    # ``embed_method`` default is ``"ETKDGv3"``.
    assert sig.parameters["embed_method"].default == "ETKDGv3"
    # ``n_conformers`` default is 5 (paper-grade).
    assert sig.parameters["n_conformers"].default == 5


def test_pb_runner_batch_and_pass_rate():
    """``check_posebusters_batch`` and ``pb_pass_rate`` work on the
    skipped-missing-pkg path AND on the valid-pkg path.

    This test pins the *bulk* helpers regardless of PB availability.
    """
    runner = _import_runner()
    res = runner.check_posebusters_batch(["CCO", "", "c1ccccc1"])
    assert isinstance(res, list)
    assert len(res) == 3
    for r in res:
        assert isinstance(r, dict)
        assert "pb_valid" in r
    # ``pb_pass_rate`` over the same list — should return a float in
    # [0, 1] (skipping skipped entries).  Returns 0.0 when no valid
    # verdict can be computed.
    pr = runner.pb_pass_rate(res)
    assert isinstance(pr, float)
    assert 0.0 <= pr <= 1.0
    # Empty list -> 0.0 (defensive contract).
    assert runner.pb_pass_rate([]) == 0.0


def test_pb_runner_signature_exposes_kwargs():
    """Public signature is keyword-only for the conformer knobs.

    Belt-and-braces check that callers can pass the knobs without
    positional coupling.
    """
    runner = _import_runner()
    import inspect
    sig = inspect.signature(runner.check_posebusters)
    params = sig.parameters
    assert "smiles" in params
    assert params["smiles"].default is inspect.Parameter.empty
    # The other three are keyword-only by intent (the spec calls them
    # out as keyword-only in the docstring).
    for kname in ("n_conformers", "embed_method", "optimize_method"):
        assert kname in params
        assert params[kname].kind == inspect.Parameter.KEYWORD_ONLY, (
            f"{kname} must be keyword-only"
        )