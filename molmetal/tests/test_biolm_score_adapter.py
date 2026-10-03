"""test_biolm_score_adapter.py — pytest tests for the BioLM-Score adapter.

This module covers the **adapter contract** documented in
``TODO/01_research/lambda_integration_plan.md`` for the file
``molmetal/molmetal_lam/sbdd_env/biolm_score_adapter.py``.

The adapter wraps the vendored BioLM-Score repo
(``molmetal/references/BioLM-Score``) for protein-ligand scoring
and exposes a thin ``(smiles, pocket) -> float`` callable.  These
tests are CPU-only: the upstream BioLM-Score package is **not**
required (the adapter instantiates lazily and degrades gracefully
when the vendored repo or torch_scatter extensions are missing).

Contracts covered (each is its own test function):

1. Adapter instantiates without the upstream BioLM-Score package —
   i.e. the constructor never imports the heavy upstream stack.
2. ``adapter.score(smiles, pocket=None)`` returns ``0.0`` when
   ``pocket is None`` — graceful-degrade contract for callers that
   have not yet loaded a Pocket.
3. ``adapter.score(smiles, pocket=mock_pocket)`` returns a finite
   float in ``[0, 1]`` when given a usable Pocket stub.
4. :class:`RewardAggregator` accepts ``r_biolm_score`` as a new
   channel and the aggregator's ``__call__`` invokes it without
   crashing when the channel is wired with ``w_biolm_score=0.0``.
5. ``adapter.score_channel(name="r_biolm_score", smiles=...)``
   routes the call back to the BioLM-Score channel and returns
   a finite float — closes the (channel-name, smiles) routing loop.

All tests are CPU-only and self-skip when the adapter module is not
importable yet (the module ships in a follow-up wave alongside the
``r_biolm_score`` reward channel wire-up).

Run from the project root::

    uv run pytest molmetal/tests/test_biolm_score_adapter.py -q --tb=short
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Lazy imports — keep pytest collection working even when the adapter
# module is not on disk yet.  Each test that needs the adapter calls
# these helpers and self-skips when the import fails.
# ---------------------------------------------------------------------------
def _import_adapter():
    """Return ``(BioLMScoreAdapter, score_channel_name)`` or raise
    ``pytest.skip.Exception`` when the module is missing.
    """
    try:
        from molmetal_lam.sbdd_env.biolm_score_adapter import (
            BioLMScoreAdapter,
            BIOLM_SCORE_CHANNEL_NAME,
        )
    except Exception as exc:
        pytest.skip(f"biolm_score_adapter module not importable: {exc}")
    return BioLMScoreAdapter, BIOLM_SCORE_CHANNEL_NAME


def _import_reward_aggregator():
    """Lazy-import :class:`RewardAggregator` so tests self-skip when the
    heavier search_alg / torch stack is broken.
    """
    try:
        from molmetal_lam.search_alg.proof_search import RewardAggregator

        return RewardAggregator
    except Exception as exc:  # pragma: no cover - defensive
        pytest.skip(f"RewardAggregator not importable: {exc}")


@dataclass
class _MockPocket:
    """Minimal pocket stub — just enough surface area for adapter.score().

    The BioLM-Score adapter accepts either a :class:`molmetal.domain.Pocket`
    dataclass, a string path, or any object exposing ``.pdb_id`` /
    ``.coords``.  The mock below uses a string pdb_id so the adapter's
    score() path can read it without instantiating the heavy
    domain.dataclass chain (which requires torch tensors).
    """

    pdb_id: str = "mock_pdb_0001"
    coords = None


@dataclass
class _DummyState:
    """Minimal state-object stub mimicking MoleculeClosedTerm.smiles."""

    smiles: str


# ---------------------------------------------------------------------------
# Test 1 — Adapter instantiates lazily without BioLM-Score upstream
# ---------------------------------------------------------------------------
def test_adapter_instantiates_without_upstream(tmp_path: Path) -> None:
    """Constructing :class:`BioLMScoreAdapter` must NOT import the heavy
    BioLM-Score upstream stack (``torch_scatter``, ESM, Chemformer, …) —
    import is deferred until the first ``score()`` call.

    Passing ``repo_root`` at a non-existent path makes the "no upstream"
    contract explicit without depending on the test host's filesystem.
    """
    BioLMScoreAdapter, _channel_name = _import_adapter()
    adapter = BioLMScoreAdapter(
        repo_root=str(tmp_path / "no_such_biolm_repo"),
    )
    assert adapter is not None
    assert adapter.is_available() is False


# ---------------------------------------------------------------------------
# Test 2 — score() with pocket=None returns 0.0 (graceful)
# ---------------------------------------------------------------------------
def test_score_with_none_pocket_returns_zero(tmp_path: Path) -> None:
    """``adapter.score(smiles, pocket=None)`` returns 0.0 — the graceful
    contract for callers that have not yet loaded a Pocket.  This is
    the same behaviour as the ``r_vina_proxy`` channel when the docking
    backend is unavailable.
    """
    BioLMScoreAdapter, _channel_name = _import_adapter()
    adapter = BioLMScoreAdapter(repo_root=str(tmp_path / "no_such_biolm_repo"))
    v = adapter.score("CCO", pocket=None)
    assert v == 0.0, f"Expected 0.0 for None pocket, got {v!r}"


# ---------------------------------------------------------------------------
# Test 3 — score() with mock pocket returns finite float in [0, 1]
# ---------------------------------------------------------------------------
def test_score_with_mock_pocket_returns_finite_float_in_unit_interval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``adapter.score(smiles, pocket=mock)`` returns a finite float in
    ``[0, 1]``.  We monkeypatch the upstream score function so the
    test does NOT depend on the heavy BioLM-Score torch stack.
    """
    BioLMScoreAdapter, _channel_name = _import_adapter()
    adapter = BioLMScoreAdapter(repo_root=str(tmp_path / "synthetic_biolm"))

    # Stub the upstream loader: return a deterministic score from [0, 1].
    monkeypatch.setattr(adapter, "_score_via_upstream", lambda smiles, pocket: 0.42)
    v = adapter.score("c1ccccc1NC(=O)C", pocket=_MockPocket(pdb_id="1abc"))
    assert isinstance(v, float)
    assert v == v, "score returned NaN"
    assert 0.0 <= v <= 1.0, f"Score out of [0,1]: {v}"


# ---------------------------------------------------------------------------
# Test 4 — RewardAggregator accepts r_biolm_score as a new channel
# ---------------------------------------------------------------------------
def test_reward_aggregator_accepts_r_biolm_score_channel(tmp_path: Path) -> None:
    """``RewardAggregator`` exposes ``r_biolm_score`` as a callable
    attribute and tolerates the channel being wired with a closure
    that returns 0.0 (regression-proof contract).  When
    ``w_biolm_score = 0.0`` (the default), the weighted reward is
    bit-for-bit identical to the unwired baseline.
    """
    BioLMScoreAdapter, _channel_name = _import_adapter()
    RewardAggregator = _import_reward_aggregator()

    adapter = BioLMScoreAdapter(repo_root=str(tmp_path / "no_such_biolm_repo"))

    # Baseline: only QED channel.
    baseline = RewardAggregator(r_qed=lambda s: 0.7, w_qed=1.0)
    state = _DummyState(smiles="CCO")
    v_base = baseline(state)

    # Wired: same baseline PLUS the (weight=0.0) BioLM-Score channel.
    agg_wired = RewardAggregator(r_qed=lambda s: 0.7, w_qed=1.0)
    assert hasattr(agg_wired, "r_biolm_score")
    assert hasattr(agg_wired, "w_biolm_score")
    assert agg_wired.w_biolm_score == 0.0
    # Wire the adapter's score_channel as the r_biolm_score closure.
    agg_wired.r_biolm_score = adapter.score_channel
    v_wired = agg_wired(state)
    assert v_base == pytest.approx(v_wired, abs=1e-12), (
        f"weight=0.0 must be bit-for-bit identical to baseline; "
        f"got baseline={v_base} wired={v_wired}"
    )


# ---------------------------------------------------------------------------
# Test 5 — score_channel(name="r_biolm_score") routes to the adapter
# ---------------------------------------------------------------------------
def test_score_channel_with_name_routes_to_adapter(tmp_path: Path) -> None:
    """``adapter.score_channel(name="r_biolm_score", smiles=...)``
    routes the call back to the BioLM-Score channel and returns a
    finite float.  This is the routing contract the aggregator's
    channel-name dispatch uses.
    """
    BioLMScoreAdapter, channel_name = _import_adapter()
    adapter = BioLMScoreAdapter(repo_root=str(tmp_path / "no_such_biolm_repo"))
    # Routing: the only accepted channel name is the canonical one.
    assert channel_name == "r_biolm_score"
    v = adapter.score_channel(name="r_biolm_score", smiles="CCO")
    assert v == 0.0, f"Expected 0.0 (no upstream available, no pocket) — got {v!r}"
    # Routing to an unknown channel name must be a no-op (returns 0.0).
    v_unknown = adapter.score_channel(name="r_unknown_channel", smiles="CCO")
    assert v_unknown == 0.0
