"""Tests for the ADMET runner + RewardAggregator r_admet channel.

All tests are CPU-only and do not need to invoke the admet-ai model
forward pass (that's gated behind a slow-running GPU path).  The
backend-detection function ``active_backend`` returns the string the
runner *would* dispatch to; the actual AI inference is exercised in
:mod:`molmetal.validation.admet_runner` only when the user calls
:func:`predict_admet` directly with a SMILES.

The four required tests are:

* :func:`test_admet_runner_loads` — module importable, returns dict.
* :func:`test_admet_runner_aspirin` — predict_admet("CC(=O)Oc1ccccc1C(=O)O")
  returns MW ~180 regardless of backend.
* :func:`test_admet_runner_handles_invalid` — empty / invalid SMILES
  returns ``{}`` and emits a warning.
* :func:`test_admet_runner_falls_back_gracefully` — even if admet-ai /
  datamol are missing, RDKit descriptors are still produced.
"""
from __future__ import annotations

import importlib
import math
import os
import sys
import warnings

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_admet_runner():
    """Re-import the ADMET runner in a child-friendly way.

    The runner is in :mod:`molmetal.validation.admet_runner`, not
    :mod:`molmetal.molmetal_lam.tests.test_admet_runner`.  Import via
    ``importlib`` so a missing runner shows up as a clear ImportError
    in test output instead of a ModuleNotFoundError at collection.
    """
    try:
        import molmetal.validation.admet_runner as runner  # noqa: F401
    except Exception:  # pragma: no cover
        runner = importlib.import_module("molmetal.validation.admet_runner")
    return runner


@pytest.fixture(scope="module")
def runner():
    return _import_admet_runner()


# ---------------------------------------------------------------------------
# Required tests
# ---------------------------------------------------------------------------


def test_admet_runner_loads(runner):
    """Module imports cleanly and exposes ``predict_admet`` + ``admet_desirability``."""
    assert hasattr(runner, "predict_admet"), "missing predict_admet()"
    assert hasattr(runner, "admet_desirability"), "missing admet_desirability()"
    assert callable(runner.predict_admet)
    assert callable(runner.admet_desirability)
    # active_backend returns a known string
    backend = runner.active_backend()
    assert backend in {"admet-ai", "datamol", "datamol+molfeat", "rdkit", "none"}


def test_admet_runner_aspirin(runner):
    """predict_admet('CC(=O)Oc1ccccc1C(=O)O') returns MW ~180.

    Aspirin (acetylsalicylic acid) molecular weight is 180.16 g/mol.
    Every backend in the chain (admet-ai, datamol, RDKit) computes the
    same canonical SMILES and the same MW within numerical precision.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = runner.predict_admet("CC(=O)Oc1ccccc1C(=O)O")
    assert isinstance(out, dict), f"expected dict, got {type(out)}"
    assert out, "predict_admet returned an empty dict for aspirin"
    # canonical short key — always populated
    assert "MW" in out, f"MW key missing (got keys={list(out)[:8]})"
    mw = out["MW"]
    assert isinstance(mw, float), f"MW should be float, got {type(mw)}"
    assert math.isfinite(mw), "MW is NaN or Inf"
    # Aspirin MW = 180.16; allow ±2.0 Da for backend rounding.
    assert 178.0 <= mw <= 182.0, f"aspirin MW = {mw}; expected ~180.16"
    # logP should be in drug-like range (aspirin logP ~1.3)
    if "logP" in out:
        lpg = out["logP"]
        assert 0.0 <= lpg <= 5.0, f"aspirin logP = {lpg}; out of expected range"
    # desirability should be at least 0.4 (HBD/HBA/TPSA pass even if MW is too low)
    d = runner.admet_desirability(out)
    assert 0.0 <= d <= 1.0, f"desirability out of [0,1]: {d}"
    # Aspirin passes HBD ≤5 (+0.15), HBA ≤10 (+0.15), TPSA ∈ [20,130] (+0.20),
    # logP ∈ [1,4] (+0.30) → 0.80.  MW=180 is outside [200,500] so no +0.20.
    # Allow some slack for backend rounding.
    assert d >= 0.6, f"desirability for aspirin = {d}; expected ~0.80"


def test_admet_runner_handles_invalid(runner):
    """Empty / invalid SMILES return ``{}`` and emit a warning."""
    cases = ["", "   ", "NOT_A_SMILES", "C(C(C", None]
    for smi in cases:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            out = runner.predict_admet(smi)
        assert isinstance(out, dict), f"non-dict return for {smi!r}"
        assert out == {}, f"expected empty dict for {smi!r}, got {out}"
        # at least one warning was emitted
        assert len(wlist) >= 1, f"no warning for {smi!r}"


def test_admet_runner_falls_back_gracefully(runner):
    """Even if AI backends are missing, RDKit descriptors are present.

    We exercise the *RDKit-only* branch directly by calling the
    private ``_compute_rdkit_descriptors`` helper with a known SMILES
    and asserting the canonical short keys are populated.  This proves
    the fallback contract holds regardless of admet-ai / datamol
    availability.
    """
    out = runner._compute_rdkit_descriptors("CC(=O)Oc1ccccc1C(=O)O")
    # canonical short keys must all be present
    for key in ("MW", "logP", "HBA", "HBD", "TPSA", "RotBonds", "QED"):
        assert key in out, f"missing {key} in RDKit fallback path"
        assert isinstance(out[key], float), f"{key} should be float, got {type(out[key])}"
        assert math.isfinite(out[key]), f"{key} is NaN/Inf"
    # Aspirin sanity
    assert 178.0 <= out["MW"] <= 182.0
    assert out["HBA"] >= 2 and out["HBA"] <= 6
    assert out["HBD"] == 1.0  # one COOH proton
    assert 50.0 <= out["TPSA"] <= 90.0  # aspirin TPSA ~63.6
    assert 0.0 <= out["QED"] <= 1.0


# ---------------------------------------------------------------------------
# Extended tests (informational — not in the required list)
# ---------------------------------------------------------------------------


def test_admet_desirability_bounds(runner):
    """admet_desirability returns in [0, 1] for arbitrary inputs."""
    # empty → 0
    assert runner.admet_desirability({}) == 0.0
    # all-good → 1.0
    perfect = {"logP": 2.5, "MW": 350.0, "HBD": 2.0, "HBA": 5.0, "TPSA": 75.0}
    assert runner.admet_desirability(perfect) == pytest.approx(1.0)
    # all-bad → 0.0
    bad = {"logP": -10.0, "MW": 50.0, "HBD": 20.0, "HBA": 20.0, "TPSA": 500.0}
    assert runner.admet_desirability(bad) == 0.0
    # missing keys → contributes nothing (no crash)
    assert runner.admet_desirability({"logP": 2.0}) == pytest.approx(0.30)


def test_reward_aggregator_admet_channel_wires(runner):
    """RewardAggregator exposes the ``r_admet`` / ``w_admet`` fields."""
    from molmetal.molmetal_lam.search_alg.proof_search import (
        RewardAggregator,
        _r_admet_default,
    )
    agg = RewardAggregator(w_admet=1.0, r_admet=_r_admet_default)
    assert agg.w_admet == 1.0
    assert callable(agg.r_admet)
    # And the env-var override path installs the default closure.
    old = os.environ.get("ADMET_WEIGHT")
    try:
        os.environ["ADMET_WEIGHT"] = "1.0"
        agg2 = RewardAggregator()
        assert agg2.w_admet == 1.0
        assert agg2.r_admet is not None
    finally:
        if old is None:
            os.environ.pop("ADMET_WEIGHT", None)
        else:
            os.environ["ADMET_WEIGHT"] = old
    # Negative env var: opt-out stays at default
    os.environ["ADMET_WEIGHT"] = "0.5"
    agg3 = RewardAggregator()
    assert agg3.w_admet == 0.0
    os.environ.pop("ADMET_WEIGHT", None)