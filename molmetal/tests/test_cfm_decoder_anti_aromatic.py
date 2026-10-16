"""Wf-A55 W7-P5 — anti-aromatic (4n) ring detection tests.

Tests the Hückel anti-aromaticity demotion added to
``cfm_decoder_yuelbond.py``.  Operates directly on a crafted
``bond_orders`` list, calling ``_detect_and_demote_anti_aromatic``
in isolation — no full decode() pipeline, no GPU.

Coverage
--------
1. benzene (6-ring all aromatic) → no demotion, all aromatic kept.
2. cyclobutadiene (4-ring all aromatic) → demoted to alternating SINGLE,
   HONEST-NEGATIVE print captured.
3. naphthalene (10-atom fused 6-rings all aromatic) → no demotion.
4. cyclooctatetraene (8-ring all aromatic) → demoted (8 % 4 == 0).
5. 4-ring with mixed aromatic + single bonds → no demotion (the gate
   requires ALL bonds in the ring to be aromatic).

All tests are CPU-only — bond_orders is a pure Python list of tuples.
"""

from __future__ import annotations

import contextlib
import io

from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (  # noqa: E402
    _detect_and_demote_anti_aromatic,
    _is_anti_aromatic,
)
from molmetal.models.bond_head import (  # noqa: E402
    BOND_AROMATIC,
    BOND_DOUBLE,
    BOND_SINGLE,
)


# ===========================================================================
# Test 1 — _is_anti_aromatic unit table
# ===========================================================================
def test_is_anti_aromatic_unit_table():
    """Pin the Hückel 4n+2 / 4n table so a future edit cannot silently
    change which ring sizes trigger demotion."""
    assert _is_anti_aromatic(4) is True, "4n=4 is anti-aromatic"
    assert _is_anti_aromatic(8) is True, "4n=8 is anti-aromatic (conservatively)"
    assert _is_anti_aromatic(12) is True, "4n=12 is anti-aromatic"
    # 4n+2 rings are aromatic — must NOT trigger the demotion gate.
    assert _is_anti_aromatic(6) is False, "6-ring is aromatic (4n+2=6)"
    assert _is_anti_aromatic(10) is False, "10-ring is aromatic (4n+2=10)"
    assert _is_anti_aromatic(14) is False, "14-ring is aromatic (4n+2=14)"
    # 5 and 7 are neither rule.
    assert _is_anti_aromatic(5) is False, "5-ring is neither (4n+1)"
    assert _is_anti_aromatic(7) is False, "7-ring is neither (4n+3)"
    # Below 4 we don't probe.
    assert _is_anti_aromatic(3) is False
    assert _is_anti_aromatic(0) is False


# ===========================================================================
# Test 2 — benzene 6-ring all aromatic → no demotion
# ===========================================================================
def test_benzene_6ring_all_aromatic_no_demotion():
    """benzene C6 ring with all BOND_AROMATIC bonds — stays aromatic.

    6 % 4 == 2, not 4n, so the anti-aromatic gate does NOT fire.
    """
    # 6 atoms in a ring: 0-1-2-3-4-5-0 (6 bonds)
    bond_orders = [
        (0, 1, BOND_AROMATIC),
        (1, 2, BOND_AROMATIC),
        (2, 3, BOND_AROMATIC),
        (3, 4, BOND_AROMATIC),
        (4, 5, BOND_AROMATIC),
        (5, 0, BOND_AROMATIC),
    ]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        n_demoted = _detect_and_demote_anti_aromatic(bond_orders)
    assert n_demoted == 0, (
        f"benzene 6-ring must NOT trigger demotion (6 % 4 = 2); "
        f"got n_demoted={n_demoted}"
    )
    # All bonds remain aromatic.
    assert all(b[2] == BOND_AROMATIC for b in bond_orders), (
        "all 6 benzene bonds must remain aromatic"
    )
    assert "[wfA55-anti-aromatic]" not in buf.getvalue(), (
        "no HONEST-NEGATIVE print expected for benzene"
    )


# ===========================================================================
# Test 3 — cyclobutadiene 4-ring all aromatic → demoted + HONEST-NEGATIVE
# ===========================================================================
def test_cyclobutadiene_4ring_demoted_to_alternating_single():
    """cyclobutadiene 4-ring with all BOND_AROMATIC bonds — demoted.

    4 % 4 == 0 (4n), so the anti-aromatic gate fires.  The DFS
    traversal order determines WHICH bonds are demoted; the invariant
    we verify is the ALTERNATING pattern around the cycle (not the
    specific indices, since those depend on the DFS start atom).
    """
    bond_orders = [
        (0, 1, BOND_AROMATIC),
        (1, 2, BOND_AROMATIC),
        (2, 3, BOND_AROMATIC),
        (3, 0, BOND_AROMATIC),
    ]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        n_demoted = _detect_and_demote_anti_aromatic(bond_orders)
    assert n_demoted == 1, (
        f"exactly 1 cyclobutadiene ring should be demoted; "
        f"got n_demoted={n_demoted}"
    )
    stdout = buf.getvalue()
    assert "[wfA55-anti-aromatic]" in stdout, (
        f"HONEST-NEGATIVE print expected for cyclobutadiene; got: {stdout!r}"
    )
    assert "ring_size=4" in stdout, (
        f"print should carry ring_size=4; got: {stdout!r}"
    )
    # Verify the alternating pattern around the cycle.  We don't know
    # which side RDKit will see as the "start" of the cycle, but the
    # pattern must be SINGLE / AROMATIC / SINGLE / AROMATIC (or the
    # reverse — both are equivalent for RDKit).  Build an adjacency
    # walk and check that adjacent bonds in the cycle have DIFFERENT
    # classes.
    cycle_edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
    ]
    classes_by_edge = {}
    for k, (a, b, c) in enumerate(bond_orders):
        classes_by_edge[(min(a, b), max(a, b))] = c
    cycle_classes = [classes_by_edge[(min(a, b), max(a, b))] for (a, b) in cycle_edges]
    # Alternating means adjacent edges in the cycle differ.
    for k in range(len(cycle_classes)):
        assert cycle_classes[k] != cycle_classes[(k + 1) % len(cycle_classes)], (
            f"cycle edges at idx {k} and {(k+1)%len(cycle_classes)} must "
            f"alternate but both are {cycle_classes[k]}"
        )
    # Exactly half SINGLE, half AROMATIC.
    n_single = sum(1 for c in cycle_classes if c == BOND_SINGLE)
    n_aromatic = sum(1 for c in cycle_classes if c == BOND_AROMATIC)
    assert n_single == 2 and n_aromatic == 2, (
        f"cyclobutadiene should have 2 SINGLE + 2 AROMATIC after demotion, "
        f"got SINGLE={n_single} AROMATIC={n_aromatic}"
    )


# ===========================================================================
# Test 4 — naphthalene fused 6-rings all aromatic → no demotion
# ===========================================================================
def test_naphthalene_fused_6rings_no_demotion():
    """Naphthalene C10 (two fused 6-rings sharing one edge) — no demotion.

    The cycle finder detects the two 6-cycles.  Neither is 4n, so neither
    triggers the gate.  (The 10-perimeter cycle is skipped by the
    ``len(path) > 8`` bound — but 10 % 4 = 2 anyway, so it would not
    trigger even if found.)
    """
    # Naphthalene atoms 0..9; ring 1: 0-1-2-3-9-8-0; ring 2: 3-4-5-6-7-8-3
    # Fusion edge: 3-8.
    bond_orders = [
        # Ring 1
        (0, 1, BOND_AROMATIC),
        (1, 2, BOND_AROMATIC),
        (2, 3, BOND_AROMATIC),
        (3, 9, BOND_AROMATIC),
        (9, 8, BOND_AROMATIC),
        (8, 0, BOND_AROMATIC),
        # Fusion edge
        (3, 8, BOND_AROMATIC),
        # Ring 2
        (3, 4, BOND_AROMATIC),
        (4, 5, BOND_AROMATIC),
        (5, 6, BOND_AROMATIC),
        (6, 7, BOND_AROMATIC),
        (7, 8, BOND_AROMATIC),
    ]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        n_demoted = _detect_and_demote_anti_aromatic(bond_orders)
    assert n_demoted == 0, (
        f"naphthalene's two 6-rings must NOT trigger demotion (6 % 4 = 2); "
        f"got n_demoted={n_demoted}"
    )
    assert all(b[2] == BOND_AROMATIC for b in bond_orders), (
        "all naphthalene bonds must remain aromatic"
    )


# ===========================================================================
# Test 5 — cyclooctatetraene 8-ring all aromatic → demoted (8 is 4n)
# ===========================================================================
def test_cyclooctatetraene_8ring_demoted():
    """Cyclooctatetraene 8-ring with all BOND_AROMATIC — demoted.

    Footnote from the design: real COT is non-planar and behaves as a
    polyene, NOT anti-aromatic.  The decoder has no planarity probe, so
    it conservatively demotes all 4n aromatic rings.  This test pins
    that conservative behaviour so a future edit to add a planarity
    probe will fail it loudly.

    The DFS traversal order determines WHICH bonds are demoted; we
    verify the alternating pattern around the cycle, not specific
    indices.
    """
    bond_orders = [
        (0, 1, BOND_AROMATIC),
        (1, 2, BOND_AROMATIC),
        (2, 3, BOND_AROMATIC),
        (3, 4, BOND_AROMATIC),
        (4, 5, BOND_AROMATIC),
        (5, 6, BOND_AROMATIC),
        (6, 7, BOND_AROMATIC),
        (7, 0, BOND_AROMATIC),
    ]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        n_demoted = _detect_and_demote_anti_aromatic(bond_orders)
    assert n_demoted == 1, (
        f"COT 8-ring must trigger demotion (8 % 4 = 0); "
        f"got n_demoted={n_demoted}"
    )
    stdout = buf.getvalue()
    assert "[wfA55-anti-aromatic]" in stdout
    assert "ring_size=8" in stdout
    # Alternating pattern around the cycle.
    cycle_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0),
    ]
    classes_by_edge = {}
    for k, (a, b, c) in enumerate(bond_orders):
        classes_by_edge[(min(a, b), max(a, b))] = c
    cycle_classes = [classes_by_edge[(min(a, b), max(a, b))] for (a, b) in cycle_edges]
    n_single = sum(1 for c in cycle_classes if c == BOND_SINGLE)
    n_aromatic = sum(1 for c in cycle_classes if c == BOND_AROMATIC)
    assert n_single == 4 and n_aromatic == 4, (
        f"COT should have 4 SINGLE + 4 AROMATIC after demotion, "
        f"got SINGLE={n_single} AROMATIC={n_aromatic}"
    )
    for k in range(len(cycle_classes)):
        assert cycle_classes[k] != cycle_classes[(k + 1) % len(cycle_classes)], (
            f"cycle edges at idx {k} and {(k+1)%len(cycle_classes)} must alternate"
        )


# ===========================================================================
# Test 6 — 4-ring with MIXED bond orders → no demotion (all-aromatic only)
# ===========================================================================
def test_4ring_mixed_bonds_not_demoted():
    """A 4-ring with mixed SINGLE + DOUBLE + AROMATIC bonds must NOT
    trigger demotion — the gate requires ALL bonds in the ring to be
    BOND_AROMATIC (a real mixed ring is a non-aromatic 4-ring, not an
    anti-aromatic one)."""
    bond_orders = [
        (0, 1, BOND_AROMATIC),
        (1, 2, BOND_SINGLE),
        (2, 3, BOND_AROMATIC),
        (3, 0, BOND_DOUBLE),
    ]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        n_demoted = _detect_and_demote_anti_aromatic(bond_orders)
    assert n_demoted == 0, (
        f"4-ring with non-aromatic edges must NOT be demoted; "
        f"got n_demoted={n_demoted}"
    )
    # Bonds unchanged.
    assert bond_orders[0][2] == BOND_AROMATIC
    assert bond_orders[1][2] == BOND_SINGLE
    assert bond_orders[2][2] == BOND_AROMATIC
    assert bond_orders[3][2] == BOND_DOUBLE


# ===========================================================================
# Test 7 — empty bond_orders → no demotion, no crash
# ===========================================================================
def test_empty_bond_orders_no_crash():
    """Empty bond_orders list returns 0 (no crash)."""
    n_demoted = _detect_and_demote_anti_aromatic([])
    assert n_demoted == 0
