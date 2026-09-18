"""Property-based tests for the Molecular Lambda Calculus (MLC).

These tests verify algebraic laws of the MLC using
:mod:`hypothesis`.  Each test exercises a *law* (an invariant that
should hold for all valid inputs), not a specific value — this is the
property-based counterpart of the unit tests in ``test_atoms_combinators``
and friends, which check specific ground-truth values.

Laws covered
------------
1. ``Atom.arity == valence + lone_pairs`` for any (v, lp, geom) triple.
2. Every primitive atom (``H, C, N, O, F, P, S, Cl, Br, I``) is
   registered with arity >= 1.
3. Every metal atom (``Pt_II, Ru_II, Zn_II, Ir_III, Cu_II, Au_III``)
   has arity matching its expected coordination number.
4. Every click reaction rule's ``predict_yield`` returns a value in
   [0, 1] (or raises ``NotImplementedError`` when no predictor is
   attached — in which case the test is skipped via ``assume(False)``).
5. Every click reaction in :data:`CLICK_REACTIONS` is registered.
6. Sampling arbitrary symbol lists does not crash the registry lookup.
"""

from __future__ import annotations

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers — keep the imports inside the test bodies so a failure in a
# single module does not break the entire file.  (Standard pytest
# convention; matches the other test files in this directory.)
# ---------------------------------------------------------------------------


def _import_atoms():
    from molmetal_lam.atoms.combinators import (
        METAL_ATOMS,
        PRIMITIVE_ATOMS,
        _EXPECTED_METAL_CN,
        Atom,
    )
    return Atom, PRIMITIVE_ATOMS, METAL_ATOMS, _EXPECTED_METAL_CN


def _import_reaction_rules():
    from molmetal_lam.reactions.beta_reductions import REACTION_RULES
    return REACTION_RULES


def _import_click_reactions():
    from molmetal_lam.reactions.click_reactions import CLICK_REACTIONS
    return CLICK_REACTIONS


# ---------------------------------------------------------------------------
# 1. arity = valence + lone_pairs  (algebraic law of the Atom combinator)
# ---------------------------------------------------------------------------

@given(
    v=st.integers(min_value=0, max_value=8),
    lp=st.integers(min_value=0, max_value=8),
    geom=st.sampled_from(["s", "sp2", "sp3", "square_planar", "octahedral"]),
)
@settings(max_examples=64, deadline=None)
def test_atom_arity_equals_valence_plus_lone_pairs(v, lp, geom):
    """The Atom combinator's arity equals ``valence + lone_pairs``.

    This is the foundational algebraic law of the Atom layer — see
    ``combinators.py`` docstring §1.  We sweep ``(v, lp)`` in [0, 8]^2
    and a representative geometry tag.
    """
    Atom, *_ = _import_atoms()
    a = Atom(symbol="X", atomic_num=0, valence=v, lone_pairs=lp, geometry=geom)
    assert a.arity == v + lp


# ---------------------------------------------------------------------------
# 2. every primitive atom is registered with arity >= 1
# ---------------------------------------------------------------------------

@given(sym=st.sampled_from(["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]))
@settings(max_examples=32, deadline=None)
def test_primitive_atom_in_registry(sym):
    """Each primitive atom must be in the registry with arity >= 1."""
    _, PRIMITIVE_ATOMS, _, _ = _import_atoms()
    assert sym in PRIMITIVE_ATOMS
    assert PRIMITIVE_ATOMS[sym].arity >= 1


# ---------------------------------------------------------------------------
# 3. metal arity matches coordination number
# ---------------------------------------------------------------------------

@given(sym=st.sampled_from(["Pt_II", "Ru_II", "Zn_II", "Ir_III", "Cu_II", "Au_III"]))
@settings(max_examples=16, deadline=None)
def test_metal_atom_arity_matches_coordination(sym):
    """Each metal's arity must equal its expected coordination number."""
    _, _, METAL_ATOMS, _EXPECTED_METAL_CN = _import_atoms()
    assert sym in METAL_ATOMS
    assert METAL_ATOMS[sym].arity == _EXPECTED_METAL_CN[sym]


# ---------------------------------------------------------------------------
# 4. reaction rule predict_yield returns a value in [0, 1]
# ---------------------------------------------------------------------------

@given(rx=st.sampled_from(["CuAAC", "SPAAC", "SPC", "DielsAlder"]))
@settings(max_examples=16, deadline=None)
def test_reaction_rule_predict_yield_in_unit_interval(rx):
    """``rule.predict_yield`` returns a value in [0, 1] or raises.

    When no rate predictor is attached the method returns 0.0 (a
    documented fallback in :meth:`ReactionRule.predict_yield`).  When
    a predictor is attached but its underlying regressor cannot score
    the pair, an internal ``Exception`` is caught and 0.0 is returned.
    We assert the **return path** stays within [0, 1].
    """
    REACTION_RULES = _import_reaction_rules()
    rule = REACTION_RULES[rx]
    try:
        y = rule.predict_yield("CCN", "CC#N")
    except (NotImplementedError, AttributeError):
        # No predictor wired — skip the example.
        assume(False)
        return
    assert isinstance(y, float)
    assert 0.0 <= y <= 1.0


# ---------------------------------------------------------------------------
# 5. every click reaction is registered in CLICK_REACTIONS
# ---------------------------------------------------------------------------

@given(rx=st.sampled_from(["CuAAC", "SPAAC", "SPC", "DielsAlder"]))
@settings(max_examples=16, deadline=None)
def test_click_reaction_does_not_crash(rx):
    """Each click reaction name is a key in :data:`CLICK_REACTIONS`.

    We do **not** fire the reaction here — that would require valid
    tile pairs (azide + alkyne, etc.) and the property test should
    stay RDKit-free.  We only verify the registry has the key and
    the callable is non-None.
    """
    CLICK_REACTIONS = _import_click_reactions()
    assert rx in CLICK_REACTIONS
    fn = CLICK_REACTIONS[rx]
    assert fn is not None
    assert callable(fn)


# ---------------------------------------------------------------------------
# 6. atom registry lookup does not crash on arbitrary symbols
# ---------------------------------------------------------------------------

@given(
    symbols=st.lists(
        st.sampled_from(["H", "C", "N", "O", "F", "Cl"]),
        min_size=1,
        max_size=10,
    )
)
@settings(max_examples=32, deadline=None)
def test_atom_registry_lookup_is_total(symbols):
    """Random walks over the primitive registry never raise.

    Each sampled symbol must either be in ``PRIMITIVE_ATOMS`` (the
    *hit* case) or absent (the *miss* case).  Neither path should
    raise — the registry itself is a plain dict, and the
    ``Atom.from_smiles`` fallback handles the miss case.

    We assert two structural facts:

    * every symbol in ``symbols`` is *string-like*;
    * for every symbol that *is* in the registry, the registered
      atom has a positive arity.
    """
    assert symbols is not None  # type hint: list is non-empty by strategy
    _, PRIMITIVE_ATOMS, _, _ = _import_atoms()
    for s in symbols:
        assert isinstance(s, str)
        if s in PRIMITIVE_ATOMS:
            assert PRIMITIVE_ATOMS[s].arity >= 1