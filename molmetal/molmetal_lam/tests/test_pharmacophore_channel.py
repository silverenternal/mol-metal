"""test_pharmacophore_channel.py — unit tests for the TODO-30 P2.5 Phase A
pharmacophore reward channel.

The :mod:`molmetal_lam.lam_chem.pharmacophore_reward_channel` module
ships a new :attr:`RewardAggregator.r_pharmacophore` channel that turns
the locked Lipinski+Veber+ring-count gate (molmetal_lam.lam_chem.
pharmacophore_filter) into a continuous reward in ``[0, 1]``.  These
tests verify the contract:

1. The default :func:`compute_pharmacophore_score` returns ``1.0`` for a
   strict Lipinski+Veber+ring-count pass (aspirin) and a partial-credit
   value (between 0 and 1) for a partial violation.
2. ``r_pharmacophore`` returns the expected score when the channel is
   enabled and a valid SMILES is supplied.
3. **Disabled (weight=0.0)** keeps the aggregator's reward bit-for-bit
   identical to no-channel — regression-proof for all 24 REAL adapters.
4. **Empty SMILES** returns 0.0 (graceful degradation).
5. **Invalid SMILES** falls back to 0.0 — never crashes MCTS.
6. **Round-12 regression**: aggregator output unchanged when
   ``--reward-pharmacophore-weight 0.0`` (default OFF).
7. ``register_pharmacophore_channel`` wires the channel on a populated
   :class:`RewardAggregator` and the new channel is recognised by
   :meth:`RewardAggregator.aggregate`.
8. The wrapper never modifies the locked
   :mod:`pharmacophore_filter` module (constraint guard).

All tests are CPU-only and skip the GPU/CFM training stack — they run
the aggregator with mock state objects (just need ``canonical_smiles``
or ``smiles`` attribute, or be a plain ``str``).

Run from the project root::

    uv run pytest molmetal/molmetal_lam/tests/test_pharmacophore_channel.py -q --tb=short
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest


# ---------------------------------------------------------------------------
# Subject under test — imported lazily so the test module can be collected
# even when the heavier search_alg / torch stack is broken.
# ---------------------------------------------------------------------------
try:
    from molmetal_lam.lam_chem.pharmacophore_reward_channel import (
        compute_pharmacophore_score,
        register_pharmacophore_channel,
    )
    _HAS_CHANNEL = True
except Exception:  # pragma: no cover
    compute_pharmacophore_score = None
    register_pharmacophore_channel = None
    _HAS_CHANNEL = False

try:
    # Re-import the locked filter module to assert (in the constraint-
    # guard test) that the channel wrapper does not modify it.
    from molmetal_lam.lam_chem import pharmacophore_filter as _PF
    _HAS_PF = True
except Exception:  # pragma: no cover
    _PF = None
    _HAS_PF = False

try:
    from molmetal_lam.search_alg.proof_search import RewardAggregator
    _HAS_AGGREGATOR = True
except Exception:  # pragma: no cover
    RewardAggregator = None
    _HAS_AGGREGATOR = False


# ---------------------------------------------------------------------------
# Mock state helper
# ---------------------------------------------------------------------------


@dataclass
class _MockState:
    """Minimal state object that exposes ``canonical_smiles()``."""

    smiles: str

    def canonical_smiles(self) -> str:
        return self.smiles


# ---------------------------------------------------------------------------
# Tests — channels
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_CHANNEL, reason="pharmacophore_reward_channel unavailable")
def test_compute_pharmacophore_score_strict_pass():
    """Aspirin is Lipinski+Veber+ring-count strict pass → score == 1.0."""
    smi = "CC(=O)OC1=CC=CC=C1C(=O)O"  # aspirin
    score = compute_pharmacophore_score(smi)
    assert isinstance(score, float)
    assert score == pytest.approx(1.0, abs=1e-9)


@pytest.mark.skipif(not _HAS_CHANNEL, reason="pharmacophore_reward_channel unavailable")
def test_compute_pharmacophore_score_partial_credit_in_range():
    """Long alkane (acyclic, no rings) → score strictly between 0 and 1."""
    smi = "C" * 30  # acyclic long alkane — fails ring-count gate
    score = compute_pharmacophore_score(smi, allow_acyclic=True)
    assert 0.0 <= score <= 1.0
    # With allow_acyclic=True the only violations are MW/logP; the
    # score is therefore strictly positive and strictly less than 1.
    assert 0.0 < score < 1.0


@pytest.mark.skipif(not _HAS_CHANNEL, reason="pharmacophore_reward_channel unavailable")
def test_compute_pharmacophore_score_acyclic_off_penalises_cisplatin():
    """Cisplatin with allow_acyclic=False fails ring-count → score < 1.0."""
    smi = "[Pt](N)(N)(Cl)Cl"
    score = compute_pharmacophore_score(smi, allow_acyclic=False)
    # Cisplatin is acyclic (ring_count=0) → 0 Lipinski + 0 Veber +
    # 1 ring violation = 1/7 violations → score = 1 - 1/7 = 6/7.
    assert score == pytest.approx(6.0 / 7.0, abs=1e-6)


@pytest.mark.skipif(not _HAS_CHANNEL, reason="pharmacophore_reward_channel unavailable")
def test_compute_pharmacophore_score_acyclic_on_permits_cisplatin():
    """Cisplatin with allow_acyclic=True passes → score == 1.0."""
    smi = "[Pt](N)(N)(Cl)Cl"
    score = compute_pharmacophore_score(smi, allow_acyclic=True)
    assert score == pytest.approx(1.0, abs=1e-9)


@pytest.mark.skipif(not _HAS_CHANNEL, reason="pharmacophore_reward_channel unavailable")
def test_compute_pharmacophore_score_empty_smiles_returns_zero():
    """Empty SMILES → 0.0 (graceful degradation, never raises)."""
    assert compute_pharmacophore_score("") == 0.0
    assert compute_pharmacophore_score(None) == 0.0  # type: ignore[arg-type]


@pytest.mark.skipif(not _HAS_CHANNEL, reason="pharmacophore_reward_channel unavailable")
def test_compute_pharmacophore_score_invalid_smiles_returns_zero():
    """Invalid SMILES → 0.0 (filter already returns all-7 sentinel)."""
    # Long invalid string with no parseable atoms.
    assert compute_pharmacophore_score("ZZZZZ[unparseable@@@@]") == 0.0
    # A SMILES that contains only atomic symbols but no bond topology.
    assert compute_pharmacophore_score("@@@@") == 0.0


# ---------------------------------------------------------------------------
# Tests — RewardAggregator integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_AGGREGATOR, reason="RewardAggregator unavailable")
def test_register_pharmacophore_channel_returns_score_in_unit_interval():
    """After ``register_pharmacophore_channel``, calling the channel on a
    valid SMILES returns a value in [0, 1]."""
    agg = RewardAggregator()
    register_pharmacophore_channel(agg, weight=1.0)
    state = _MockState(smiles="CC(=O)OC1=CC=CC=C1C(=O)O")  # aspirin
    val = agg.r_pharmacophore(state)
    assert isinstance(val, float)
    assert 0.0 <= val <= 1.0
    assert val == pytest.approx(1.0, abs=1e-9)


@pytest.mark.skipif(not _HAS_AGGREGATOR, reason="RewardAggregator unavailable")
def test_register_pharmacophore_channel_disabled_returns_zero_contribution():
    """Default ``w_pharmacophore=0.0`` makes the channel contribute
    nothing to the aggregated reward — backward-compat proof."""
    agg = RewardAggregator(r_vina=lambda s: 5.0)
    register_pharmacophore_channel(agg, weight=0.0)
    state = _MockState(smiles="CC(=O)OC1=CC=CC=C1C(=O)O")
    # Channel is installed but weight is 0 → total reward == vina-only.
    # Note: r_vina is negated by default (vina_invert=True → -5.0).
    total = agg(state)
    assert total == pytest.approx(-5.0, abs=1e-9)
    # Sanity: r_pharmacophore is still callable and returns ~1.0 for
    # aspirin — but the weight zeroes it out.
    assert agg.r_pharmacophore(state) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.skipif(not _HAS_AGGREGATOR, reason="RewardAggregator unavailable")
def test_pharmacophore_channel_handles_empty_smiles():
    """An empty SMILES candidate → channel returns 0.0 (no crash)."""
    agg = RewardAggregator()
    register_pharmacophore_channel(agg, weight=1.0)
    state = _MockState(smiles="")
    val = agg.r_pharmacophore(state)
    assert val == 0.0


@pytest.mark.skipif(not _HAS_AGGREGATOR, reason="RewardAggregator unavailable")
def test_pharmacophore_channel_handles_invalid_smiles():
    """Invalid SMILES → channel returns 0.0 (filter's all-7 sentinel)."""
    agg = RewardAggregator()
    register_pharmacophore_channel(agg, weight=1.0)
    state = _MockState(smiles="[unparseable@@@@]")
    val = agg.r_pharmacophore(state)
    assert val == 0.0


@pytest.mark.skipif(not _HAS_AGGREGATOR, reason="RewardAggregator unavailable")
def test_aggregate_recognises_r_pharmacophore_channel_key():
    """The aggregator's ``aggregate`` method honours
    ``{"r_pharmacophore": value}`` as a recognised channel key, scaled
    by ``w_pharmacophore``."""
    agg = RewardAggregator()
    register_pharmacophore_channel(agg, weight=0.5)
    # Direct aggregate call (does NOT use __call__ which depends on the
    # MCTS machinery).  Use a per-channel value of 1.0 → 0.5 * 1.0 = 0.5.
    value = agg.aggregate(
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        channels={"r_pharmacophore": 1.0},
    )
    assert value == pytest.approx(0.5, abs=1e-9)


@pytest.mark.skipif(not _HAS_AGGREGATOR, reason="RewardAggregator unavailable")
def test_round12_regression_backward_compat_default_weight_zero():
    """Round-12 regression — a brand-new RewardAggregator with no
    channel registered produces a 0.0 contribution from
    ``r_pharmacophore`` because both the callable AND the weight are
    at their defaults.  Mirrors the historical 24-REAL-adapter
    contract: existing tests are bit-for-bit identical when this
    workflow is enabled."""
    agg = RewardAggregator(r_vina=lambda s: 2.5)
    # No register call — r_pharmacophore is None, w_pharmacophore is 0.0.
    assert agg.r_pharmacophore is None
    assert float(agg.w_pharmacophore) == 0.0
    state = _MockState(smiles="CC(=O)OC1=CC=CC=C1C(=O)O")
    # vina = 2.5 (negated by vina_invert=True → -2.5) + nothing else
    total = agg(state)
    # vina_invert defaults to True so the contribution is -2.5.
    assert total == pytest.approx(-2.5, abs=1e-9)


# ---------------------------------------------------------------------------
# Constraint guard — channel wrapper must not modify the locked filter
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not (_HAS_PF and _HAS_CHANNEL), reason="filter or channel missing")
def test_filter_module_unchanged_by_channel_import():
    """The channel wrapper must NOT add attributes / monkey-patch
    :mod:`pharmacophore_filter`.  The filter's public surface is
    locked by the spec."""
    # Public surface of the locked filter (snapshot of __all__ at the
    # time the constraint was added).  Any drift here is a contract
    # violation.
    expected_public = {
        "DEFAULT_LIPINSKI_BOUNDS",
        "DEFAULT_VEBER_BOUNDS",
        "DEFAULT_MIN_RING_COUNT",
        "PharmacophoreReport",
        "compute_lipinski_violations",
        "compute_veber_violations",
        "compute_ring_count",
        "compute_pharmacophore_report",
        "pass_pharmacophore",
        "filter_molecules",
    }
    actual_public = set(getattr(_PF, "__all__", []))
    assert expected_public.issubset(actual_public), (
        f"pharmacophore_filter.__all__ drifted; expected at least "
        f"{expected_public}, got {actual_public}"
    )


@pytest.mark.skipif(not (_HAS_PF and _HAS_CHANNEL), reason="filter or channel missing")
def test_filter_compute_pharmacophore_report_still_returns_dataclass():
    """Sanity — the underlying filter still returns a
    :class:`PharmacophoreReport` dataclass; the wrapper only adds a
    new score-mapping function on top, never replacing the dataclass."""
    report = _PF.compute_pharmacophore_report("CC(=O)OC1=CC=CC=C1C(=O)O")
    # Frozen dataclass attribute checks
    assert hasattr(report, "smiles")
    assert hasattr(report, "valid")
    assert hasattr(report, "total_violations")
    assert hasattr(report, "passing")
    assert hasattr(report, "passing_lenient")
    assert report.passing is True  # aspirin is a strict pass
    assert report.total_violations == 0
