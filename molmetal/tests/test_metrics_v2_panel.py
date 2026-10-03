"""Tests for the Wf-119 Phase 3 cleanup of the 4 MOCK_DELETED external scorers.

This module verifies the silent-None anti-pattern that lived in
:func:`molmetal_lam.sbdd_env.metrics_v2._safe_external_score` has been
eliminated:

* The four Phase-4A external scorers (``ptiv_reduction_potential``,
  ``coord_geometry_proxy``, ``phototherapy_activity``,
  ``metalloprotein_affinity`` — note ``metalloprotein_affinity`` is not
  exposed as a wrapper function but is exercised via the adapter
  contract) now raise :class:`AdapterUnavailable` on every failure path
  (missing module, missing adapter attribute, ``score()`` returning
  ``None``, ``score()`` raising, empty / non-string SMILES).
* The :func:`all_metrics_one` and :func:`all_metrics_mean` panel
  functions still return ``None`` for the three external-scorer keys
  when ``external_scorers_enabled=False`` (no exception is raised).
* The same panel functions raise :class:`AdapterUnavailable` when
  ``external_scorers_enabled=True` (because the adapter module is
  absent — Wf-43 MOCK_DELETED state).

Honest framing (Wf-43 invariant): the panel functions never silently
fabricate a reading by returning ``None`` when the adapter is missing.
The caller must opt in (or out) explicitly.

CPU-only; no paper work touched (TODO-84 gate honored).
"""

from __future__ import annotations

import os
import sys

# Ensure repo root is on sys.path so ``molmetal_lam`` resolves when pytest
# runs from a different working directory.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import pytest

from molmetal_lam.sbdd_env.metrics_v2 import (
    AdapterUnavailable,
    _safe_external_score,
    all_metrics_mean,
    all_metrics_one,
    coord_geometry_proxy,
    phototherapy_activity,
    ptiv_reduction_potential,
)


# No RDKit dependency: these tests exercise the AdapterUnavailable
# contract only (the MOCK_DELETED state — adapter module absent — does
# not touch RDKit at all). Keeping the skipif off makes the Wf-119
# contract testable on CI boxes without RDKit.


# ---------------------------------------------------------------------------
# AdapterUnavailable contract
# ---------------------------------------------------------------------------
def test_adapter_unavailable_is_runtime_error_subclass():
    """AdapterUnavailable must be a RuntimeError subclass (matches the
    score_any.AdapterUnavailable contract so callers can catch both)."""
    assert issubclass(AdapterUnavailable, RuntimeError)


def test_adapter_unavailable_can_be_instantiated():
    """AdapterUnavailable must accept an arbitrary message."""
    msg = "test failure path"
    err = AdapterUnavailable(msg)
    assert str(err) == msg
    # Default RuntimeError accepts *args too — keep the message round-trip
    assert isinstance(err, RuntimeError)


# ---------------------------------------------------------------------------
# _safe_external_score: each failure path raises (no silent None)
# ---------------------------------------------------------------------------
class _FakeAdapter:
    """Fake adapter module exposing the four Phase-4A attributes."""

    def __init__(self, behavior: str = "ok", score_return=None):
        self._behavior = behavior
        self._score_return = score_return

    def score(self, smiles):  # noqa: D401 - protocol method
        if self._behavior == "raise":
            raise RuntimeError("simulated adapter failure")
        if self._behavior == "none":
            return None
        return self._score_return


def test_safe_external_score_raises_when_module_missing(monkeypatch):
    """If ``external_scorers`` import fails (Wf-43 MOCK_DELETED state),
    _safe_external_score must raise AdapterUnavailable, NOT return None."""
    # Simulate the MOCK_DELETED state by making the import fail
    import molmetal_lam.sbdd_env as sbdd_env

    monkeypatch.delattr(sbdd_env, "external_scorers", raising=False)
    with pytest.raises(AdapterUnavailable) as exc_info:
        _safe_external_score("coord_geometry_proxy", "coord_geometry_proxy", "CCO")
    msg = str(exc_info.value)
    assert "external_scorers" in msg
    assert "MOCK_DELETED" in msg
    assert "external_scorers_enabled=False" in msg


def test_safe_external_score_raises_when_adapter_missing(monkeypatch):
    """If the named adapter is missing from the external_scorers module,
    _safe_external_score must raise AdapterUnavailable."""
    import molmetal_lam.sbdd_env as sbdd_env

    # Build a fake module exposing only one of the four adapters
    class _Partial:
        coord_geometry_proxy = _FakeAdapter(score_return={"validity_score": 1.0})

    monkeypatch.setattr(sbdd_env, "external_scorers", _Partial(), raising=False)
    with pytest.raises(AdapterUnavailable) as exc_info:
        _safe_external_score(
            "metalloprotein_affinity", "metalloprotein_affinity", "CCO"
        )
    msg = str(exc_info.value)
    assert "metalloprotein_affinity" in msg
    assert "not found" in msg


def test_safe_external_score_raises_when_score_returns_none(monkeypatch):
    """If adapter.score() returns None, _safe_external_score must raise
    AdapterUnavailable (no silent propagation)."""
    import molmetal_lam.sbdd_env as sbdd_env

    class _Mod:
        coord_geometry_proxy = _FakeAdapter(behavior="none")

    monkeypatch.setattr(sbdd_env, "external_scorers", _Mod(), raising=False)
    with pytest.raises(AdapterUnavailable) as exc_info:
        _safe_external_score("coord_geometry_proxy", "coord_geometry_proxy", "CCO")
    assert "returned None" in str(exc_info.value)


def test_safe_external_score_raises_when_score_raises(monkeypatch):
    """If adapter.score() raises, _safe_external_score must wrap the
    exception in AdapterUnavailable so the panel never sees raw adapter
    errors."""
    import molmetal_lam.sbdd_env as sbdd_env

    class _Mod:
        coord_geometry_proxy = _FakeAdapter(behavior="raise")

    monkeypatch.setattr(sbdd_env, "external_scorers", _Mod(), raising=False)
    with pytest.raises(AdapterUnavailable) as exc_info:
        _safe_external_score("coord_geometry_proxy", "coord_geometry_proxy", "CCO")
    msg = str(exc_info.value)
    assert "adapter.score() raised" in msg
    assert "RuntimeError" in msg


def test_safe_external_score_raises_on_empty_smiles():
    """Empty / non-string SMILES must raise AdapterUnavailable (the
    silent-None path used to return ``None`` for this — eliminated in
    Wf-119 Phase 3)."""
    with pytest.raises(AdapterUnavailable):
        _safe_external_score("coord_geometry_proxy", "coord_geometry_proxy", "")
    with pytest.raises(AdapterUnavailable):
        _safe_external_score("coord_geometry_proxy", "coord_geometry_proxy", "   ")
    with pytest.raises(AdapterUnavailable):
        _safe_external_score("coord_geometry_proxy", "coord_geometry_proxy", None)  # type: ignore[arg-type]


def test_safe_external_score_happy_path_for_coord_geometry_proxy(monkeypatch):
    """When the adapter is live and returns a record with validity_score,
    _safe_external_score returns that value as a float."""
    import molmetal_lam.sbdd_env as sbdd_env

    class _Mod:
        coord_geometry_proxy = _FakeAdapter(
            score_return={"validity_score": 0.85, "metal": "Pt"}
        )

    monkeypatch.setattr(sbdd_env, "external_scorers", _Mod(), raising=False)
    val = _safe_external_score("coord_geometry_proxy", "coord_geometry_proxy", "CCO")
    assert val == 0.85
    assert isinstance(val, float)


def test_safe_external_score_happy_path_for_scalar_adapter(monkeypatch):
    """For non-coord scorers (ptiv_reduction_potential,
    phototherapy_activity, metalloprotein_affinity), _safe_external_score
    returns the adapter's scalar score as a float."""
    import molmetal_lam.sbdd_env as sbdd_env

    class _Mod:
        ptiv_reduction_potential = _FakeAdapter(score_return=0.92)

    monkeypatch.setattr(sbdd_env, "external_scorers", _Mod(), raising=False)
    val = _safe_external_score(
        "ptiv_reduction_potential", "ptiv_reduction_potential", "CCO"
    )
    assert val == 0.92
    assert isinstance(val, float)


# ---------------------------------------------------------------------------
# Wrapper functions: raise AdapterUnavailable in MOCK_DELETED state
# ---------------------------------------------------------------------------
def test_ptiv_reduction_potential_raises_in_mock_deleted_state():
    """ptiv_reduction_potential wrapper raises AdapterUnavailable when
    the external_scorers module is absent (current Wf-43 state)."""
    with pytest.raises(AdapterUnavailable):
        ptiv_reduction_potential("CCO")


def test_coord_geometry_proxy_raises_in_mock_deleted_state():
    """coord_geometry_proxy wrapper raises AdapterUnavailable when
    the external_scorers module is absent."""
    with pytest.raises(AdapterUnavailable):
        coord_geometry_proxy("CCO")


def test_phototherapy_activity_raises_in_mock_deleted_state():
    """phototherapy_activity wrapper raises AdapterUnavailable when
    the external_scorers module is absent."""
    with pytest.raises(AdapterUnavailable):
        phototherapy_activity("CCO")


# ---------------------------------------------------------------------------
# Panel functions: external_scorers_enabled controls exception propagation
# ---------------------------------------------------------------------------
def test_all_metrics_one_disabled_returns_none_for_external_scorers():
    """When external_scorers_enabled=False (default), all three
    external-scorer keys are None without raising — the panel is
    lightweight."""
    panel = all_metrics_one("CCO", external_scorers_enabled=False)
    assert panel["ptiv_reduction_potential"] is None
    assert panel["coord_geometry_proxy"] is None
    assert panel["phototherapy_activity"] is None
    # The other 9 metric keys must still be populated
    for k in (
        "logp7_4",
        "gi50_proxy",
        "cell_permeability_logPapp",
        "herg_cardio_risk",
        "ames_mutagen",
        "hepatotox_index",
        "aqueous_solubility_logS",
        "plasma_protein_binding",
        "ring_size_distribution",
        "md_relax_energy",
    ):
        assert k in panel


def test_all_metrics_one_enabled_raises_adapter_unavailable():
    """When external_scorers_enabled=True, all_metrics_one must raise
    AdapterUnavailable (no silent None on the MOCK_DELETED path)."""
    with pytest.raises(AdapterUnavailable):
        all_metrics_one("CCO", external_scorers_enabled=True)


def test_all_metrics_mean_disabled_returns_none_for_external_scorers():
    """When external_scorers_enabled=False (default), the mean panel's
    three external-scorer keys are None without raising."""
    panel = all_metrics_mean(["CCO", "c1ccccc1"], external_scorers_enabled=False)
    assert panel["ptiv_reduction_potential"] is None
    assert panel["coord_geometry_proxy"] is None
    assert panel["phototherapy_activity"] is None
    # Sanity: the 9 organic-only keys are still populated
    for k in (
        "logp7_4",
        "gi50_proxy",
        "cell_permeability_logPapp",
        "herg_cardio_risk",
        "ames_mutagen",
        "hepatotox_index",
        "aqueous_solubility_logS",
        "plasma_protein_binding",
        "ring_size_distribution",
    ):
        assert k in panel


def test_all_metrics_mean_enabled_raises_adapter_unavailable():
    """When external_scorers_enabled=True, all_metrics_mean must raise
    AdapterUnavailable."""
    with pytest.raises(AdapterUnavailable):
        all_metrics_mean(["CCO", "c1ccccc1"], external_scorers_enabled=True)


# ---------------------------------------------------------------------------
# Audit: silent None anti-pattern is gone from _safe_external_score
# ---------------------------------------------------------------------------
def test_safe_external_score_no_silent_none_on_missing_module(monkeypatch):
    """Regression: the pre-Wf-119 _safe_external_score returned ``None``
    on every failure path. Verify NONE of those paths returns None now
    — the only allowed outcome is AdapterUnavailable (or a successful
    float return)."""
    import molmetal_lam.sbdd_env as sbdd_env

    # Path 1: missing module
    monkeypatch.delattr(sbdd_env, "external_scorers", raising=False)
    with pytest.raises(AdapterUnavailable):
        result = _safe_external_score(
            "coord_geometry_proxy", "coord_geometry_proxy", "CCO"
        )
        # The assertion above guarantees we never reach the next line,
        # but defensively also check the docstring return type:
        assert result is not None, "silent None path eliminated"


def test_safe_external_score_no_silent_none_on_empty_smiles():
    """Regression: the pre-Wf-119 _safe_external_score returned ``None``
    for empty SMILES. Verify that path now raises AdapterUnavailable."""
    with pytest.raises(AdapterUnavailable):
        _safe_external_score("coord_geometry_proxy", "coord_geometry_proxy", "")
