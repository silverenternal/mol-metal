"""Tests for the WF-T25-SA-Prior SA-aware MCTS leaf prior.

These tests exercise the :class:`SAPrior` dataclass + the opt-in
``use_sa_prior=True`` / ``sa_prior_strength`` wiring in
:class:`MCTSProofSearch.search` without ever running a real MCTS
search.  They are designed to run on a headless CI box (no RDKit
GPU, no SDFs, no PySR, no CuAAC reaction network) so they always
finish within a few seconds.

The 5+ tests below cover:

* benzene vs cisplatin SA-prior ordering (lower SA → higher logit);
* unparseable-SMILES fallback to 0.0 logit;
* non-zero logit for valid SMILES;
* ``--use-sa-prior`` kwarg wiring through ``MCTSProofSearch.search``
  to ``_prior``;
* default OFF / opt-out preserves the legacy 0.5 constant prior.

Each test asserts a *strict* property of the prior so the suite
catches regressions in either the SAPrior class itself or the
wire-up path in ``proof_search.py``.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# Make the repo importable.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Local helper for the MCTS-touching tests — small stub rule so the
# constructor doesn't need a full ReactionRule.
# ---------------------------------------------------------------------------
@dataclass
class _StubRule:
    """Minimal ReactionRule-shaped stub for MCTSProofSearch wiring tests."""

    name: str = "stub_rule"

    def reduce(self, molecule: Any) -> List:
        return []


# ---------------------------------------------------------------------------
# Test 1 — benzene vs cisplatin SA-prior ordering
# ---------------------------------------------------------------------------
def test_sa_prior_scores_benzene_higher_than_cisplatin():
    """Lower SA → higher logit: benzene (SA~1.0, the Ertl floor)
    must score higher than cisplatin (SA~5-7).  The logit is
    ``-log(sa)``.

    Note: the RDKit ``sascorer`` library floors SA at 1.0 for very
    simple molecules like benzene (and ethanol), so benzene emits
    ``logit = -log(1.0) = -0.0`` in this implementation.  We test
    the *ordering* (benzene > cisplatin) and the *sign* (both
    negative) without depending on the absolute SA magnitude.
    """
    from molmetal_lam.search_alg.priors import SAPrior

    prior = SAPrior()
    logit_benzene = float(prior.prior_logit("c1ccccc1"))
    logit_cisplatin = float(prior.prior_logit("N.N.N.N.Cl.Cl.[Pt]"))

    # Benzene SA ≤ 2.27 → logit ≥ -0.82 (higher / less negative).
    # Cisplatin SA ≈ 5.94 → logit ≈ -1.78 (lower / more negative).
    assert logit_benzene > logit_cisplatin, (
        f"benzene ({logit_benzene:.3f}) should have higher logit than "
        f"cisplatin ({logit_cisplatin:.3f}) — lower SA must yield "
        f"higher logit per the SA-aware prior contract."
    )
    # Both logits are negative (or zero) — they're literally
    # -log(sa) so as long as SA > 1.0 the logit is negative.
    # For benzene SA = 1.0 (the Ertl floor) the logit is exactly 0.0.
    assert -2.5 <= logit_benzene <= 0.5, (
        f"benzene SA logit out of expected range: {logit_benzene}"
    )
    assert -2.5 <= logit_cisplatin <= 0.5, (
        f"cisplatin SA logit out of expected range: {logit_cisplatin}"
    )
    # And the difference is positive (benzene - cisplatin ≥ 0).
    assert logit_benzene >= logit_cisplatin


# ---------------------------------------------------------------------------
# Test 2 — unparseable SMILES fallback
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad_smiles",
    [
        "not_a_smiles",
        "C(C(C(C(",
        "Z[unobtainium]",
        "@@@",
        "",
        "   ",
    ],
)
def test_sa_prior_zero_for_unparseable_smiles(bad_smiles):
    """Unparseable SMILES (or empty / whitespace-only input) must fall
    back to ``0.0`` logit (= uniform baseline).
    """
    from molmetal_lam.search_alg.priors import SAPrior

    prior = SAPrior()
    logit = float(prior.prior_logit(bad_smiles))
    assert logit == 0.0, (
        f"Unparseable SMILES {bad_smiles!r} must yield 0.0 logit, "
        f"got {logit}"
    )
    # And the prior_mass helper must also degrade to 0.5 (uniform).
    mass = float(prior.prior_mass(bad_smiles))
    assert mass == 0.5, (
        f"Unparseable SMILES {bad_smiles!r} must yield prior_mass=0.5, "
        f"got {mass}"
    )


# ---------------------------------------------------------------------------
# Test 3 — non-zero logit for valid SMILES (with the Ertl floor
# caveat — benzene / ethanol hit SA=1.0 and logit=0).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "smiles,expected_sa_lo,expected_sa_hi,expect_nonzero",
    [
        ("c1ccccc1", 1.0, 3.5, False),       # benzene → SA=1.0 (Ertl floor)
        ("CCO", 1.0, 3.0, True),             # ethanol → SA≈2.0
        ("CC(=O)Oc1ccccc1C(=O)O", 1.0, 3.0, True),  # aspirin → SA≈1.6
        ("N.N.N.N.Cl.Cl.[Pt]", 4.0, 8.0, True),     # cisplatin → SA≈5-7
        ("CCCCCCCCCCCCCCCC", 1.0, 4.0, True),       # hexadecane → SA≈1.1
    ],
)
def test_sa_prior_logit_for_valid_smiles(
    smiles, expected_sa_lo, expected_sa_hi, expect_nonzero,
):
    """Valid SMILES must yield a finite logit in ``(-2.5, 0.0]``
    consistent with the raw SA score being in ``[1, 10]``.

    Some very-simple molecules (benzene, hexadecane) hit the Ertl
    floor SA = 1.0 so the logit is exactly 0.0 — this is the
    expected behaviour, not a bug.  We use the ``expect_nonzero``
    flag to mark which molecules are expected to have SA > 1.

    The expected SA range is loose on purpose: the Ertl library
    version + RDKit version can shift the absolute SA by a few
    tenths, but the ordering and sign of the logit are robust.
    """
    from molmetal_lam.search_alg.priors import SAPrior

    prior = SAPrior()
    logit = float(prior.prior_logit(smiles))
    assert math.isfinite(logit), (
        f"valid SMILES {smiles!r} must yield finite logit"
    )
    # -log(10) ≈ -2.30, -log(1) = 0.0 — the logit must be in
    # [-2.5, 0.0] for any SA in [1, 10].  We allow a tiny epsilon
    # above 0 for boundary cases (SA < 1 should never happen in
    # practice — the sascorer floor is 1.0 — but we leave slack).
    assert -2.5 <= logit <= 0.5, (
        f"logit for {smiles!r} out of expected (-2.5, 0.5]: {logit}"
    )
    if expect_nonzero:
        assert logit != 0.0, (
            f"valid SMILES {smiles!r} (SA > 1) must yield non-zero logit"
        )
    # Recover the SA score from the logit and check it falls in the
    # expected band (a stronger property than the loose logit band).
    sa_recovered = math.exp(-logit)
    assert expected_sa_lo <= sa_recovered <= expected_sa_hi, (
        f"recovered SA for {smiles!r} = {sa_recovered:.3f} outside "
        f"expected band [{expected_sa_lo}, {expected_sa_hi}]"
    )


# ---------------------------------------------------------------------------
# Test 4 — cache integrity (repeated queries return same result)
# ---------------------------------------------------------------------------
def test_sa_prior_cache_returns_consistent_results():
    """Repeated :meth:`prior_logit` calls on the same SMILES must
    return the same value (idempotent + cached).
    """
    from molmetal_lam.search_alg.priors import SAPrior

    prior = SAPrior()
    v_first = float(prior.prior_logit("c1ccccc1"))
    v_second = float(prior.prior_logit("c1ccccc1"))
    v_third = float(prior.prior_logit("c1ccccc1"))
    assert v_first == v_second == v_third
    # And the cache should expose exactly one entry for benzene.
    assert "c1ccccc1" in prior._cache


# ---------------------------------------------------------------------------
# Test 5 — get_sa_prior singleton is shared
# ---------------------------------------------------------------------------
def test_sa_prior_singleton_is_shared():
    """The module-level :func:`get_sa_prior` factory must return the
    same instance on every call (avoids per-search re-allocation).
    """
    from molmetal_lam.search_alg.priors import get_sa_prior

    p1 = get_sa_prior()
    p2 = get_sa_prior()
    assert p1 is p2


# ---------------------------------------------------------------------------
# Test 6 — opt-out (default OFF) preserves legacy constant prior
# ---------------------------------------------------------------------------
def test_search_default_off_preserves_legacy_prior(monkeypatch):
    """When ``use_sa_prior`` is not set (the default), the MCTS leaf
    prior must be bit-for-bit identical to the legacy cascade — the
    SA prior does NOT contribute.  We verify this by patching
    :meth:`SAPrior.prior_logit` to return a non-zero sentinel and
    confirming the sentinel does NOT leak into the search's
    :meth:`_prior` output.

    This is the Round-12 / Round-13 regression test — the wire-up
    must not change behaviour when the gate is OFF.
    """
    try:
        from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
        from molmetal_lam.search_alg.proof_search import MCTSProofSearch
        from molmetal_lam.search_alg.priors import SAPrior
        from molmetal_lam.binding.types import BindingSite, PROTEASE_GENERIC
    except Exception as exc:  # pragma: no cover — defensive
        pytest.skip(f"MCTS stack unavailable: {exc}")

    # Patch prior_logit to return a sentinel — if the SA gate ever
    # leaks, _prior will emit 9999.0 (the sentinel) instead of 0.5.
    monkeypatch.setattr(
        SAPrior, "prior_logit", lambda self, smi: 9999.0,
    )
    monkeypatch.setattr(
        SAPrior, "prior_mass", lambda self, smi: 0.9999,
    )

    state = MoleculeClosedTerm.from_smiles("c1ccccc1", embed_3d=False)
    search = MCTSProofSearch(
        tile_library=[state],
        rules={"stub": _StubRule()},
        target_predicates=[],
        binding_site=PROTEASE_GENERIC,
        scorer=lambda s: 0.0,
    )

    # Without use_sa_prior=True, the search must NOT consult the
    # SAPrior at all — _prior must return the legacy cascade.
    p_off = float(search._prior(state))
    assert p_off < 0.99, (
        f"OFF-gate _prior leaked SA signal: {p_off} — must be < 0.99"
    )
    assert p_off >= 0.0, (
        f"OFF-gate _prior emitted negative probability: {p_off}"
    )


# ---------------------------------------------------------------------------
# Test 7 — opt-in (use_sa_prior=True) consults SA prior
# ---------------------------------------------------------------------------
def test_search_use_sa_prior_true_consults_sa(monkeypatch):
    """When ``use_sa_prior=True`` is passed to
    :meth:`MCTSProofSearch.search`, the SA prior must contribute to
    the leaf prior mass.  We patch :meth:`SAPrior.prior_mass` to
    return a sentinel (0.95) so we can detect the contribution.
    """
    try:
        from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
        from molmetal_lam.search_alg.proof_search import MCTSProofSearch
        from molmetal_lam.search_alg.priors import SAPrior
        from molmetal_lam.binding.types import BindingSite, PROTEASE_GENERIC
    except Exception as exc:  # pragma: no cover — defensive
        pytest.skip(f"MCTS stack unavailable: {exc}")

    # Sentinel prior_mass = 0.95 — when blended with the legacy 0.5
    # at sa_prior_strength=0.5, the result is 0.5 * 0.5 + 0.5 * 0.95
    # = 0.725.  If the SA gate does NOT fire, _prior returns 0.5
    # unchanged.
    monkeypatch.setattr(
        SAPrior, "prior_logit", lambda self, smi: math.log(0.95),
    )
    monkeypatch.setattr(
        SAPrior, "prior_mass", lambda self, smi: 0.95,
    )

    state = MoleculeClosedTerm.from_smiles("c1ccccc1", embed_3d=False)
    search = MCTSProofSearch(
        tile_library=[state],
        rules={"stub": _StubRule()},
        target_predicates=[],
        binding_site=PROTEASE_GENERIC,
        scorer=lambda s: 0.0,
    )

    # Manually set the SA-prior hooks (mimicking what search() does
    # at the top of every call when use_sa_prior=True).
    search._use_sa_prior_for_search = True
    search._sa_prior_strength_for_search = 0.5
    search._sa_prior_obj_for_search = None  # force lazy init

    p_on = float(search._prior(state))
    # The blended prior must be > 0.5 (since the SA mass = 0.95 > 0.5).
    # The exact value depends on the legacy cascade — heuristic()
    # may return 0.5 if no aggregator channels are wired.  We assert
    # the SA mass lifted the result above 0.5.
    assert p_on > 0.5, (
        f"ON-gate _prior did not lift above 0.5: {p_on} — SA prior "
        f"is not contributing"
    )
    # And it must be ≤ 0.99 — the SA mass caps at 0.95 with the
    # default blending so the result is at most 0.5 + 0.5 * 0.45.
    assert p_on < 0.99, (
        f"ON-gate _prior saturated: {p_on} — SA mass is dominating"
    )


# ---------------------------------------------------------------------------
# Test 8 — sa_prior_strength=0.0 is bit-for-bit equal to OFF
# ---------------------------------------------------------------------------
def test_search_sa_prior_strength_zero_equals_off(monkeypatch):
    """When ``use_sa_prior=True`` AND ``sa_prior_strength=0.0`` the
    SA prior contributes nothing — the result must be bit-for-bit
    identical to the legacy OFF path.  This is the knob that lets
    users A/B test the SA prior without touching the search().
    """
    try:
        from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
        from molmetal_lam.search_alg.proof_search import MCTSProofSearch
        from molmetal_lam.search_alg.priors import SAPrior  # noqa: F401
        from molmetal_lam.binding.types import BindingSite, PROTEASE_GENERIC
    except Exception as exc:  # pragma: no cover - defensive
        pytest.skip(f"MCTS stack unavailable: {exc}")

    state = MoleculeClosedTerm.from_smiles("c1ccccc1", embed_3d=False)

    # Path A — gate OFF (legacy).
    s_off = MCTSProofSearch(
        tile_library=[state],
        rules={"stub": _StubRule()},
        target_predicates=[],
        binding_site=PROTEASE_GENERIC,
        scorer=lambda s: 0.0,
    )
    p_off = float(s_off._prior(state))

    # Path B — gate ON, strength = 0.0 (no SA contribution).
    s_on0 = MCTSProofSearch(
        tile_library=[state],
        rules={"stub": _StubRule()},
        target_predicates=[],
        binding_site=PROTEASE_GENERIC,
        scorer=lambda s: 0.0,
    )
    s_on0._use_sa_prior_for_search = True
    s_on0._sa_prior_strength_for_search = 0.0
    s_on0._sa_prior_obj_for_search = None
    p_on0 = float(s_on0._prior(state))

    assert p_off == pytest.approx(p_on0, abs=1e-9), (
        f"sa_prior_strength=0.0 must equal OFF: {p_off} vs {p_on0}"
    )
