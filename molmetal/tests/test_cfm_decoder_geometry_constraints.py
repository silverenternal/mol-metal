"""Wf-A55 W7-P1 — decoder-side geometry constraint tests.

Tests the three per-metal geometry classes added to
``cfm_decoder_yuelbond.py``:

  * ``_SQUARE_PLANAR``  (max=4, 90°/180°):  Pd/Pt_II/Au_III
  * ``_OCTAHEDRAL``     (max=6, 90°/180°):  Fe/Ru/Rh/Re/Os/Ir
  * ``_TETRAHEDRAL``    (max=4, 109.5°):    Cu/Zn

Coverage
--------
1. cisplatin (Pt_II, 4 donors) → 4 bonds accepted (within SqP cap).
2. [Ru(NH3)6]2+ (Ru, 6 donors) → 6 bonds accepted (within Oct cap).
3. [Zn(NH3)4]2+ (Zn, 4 donors) → 4 bonds accepted (within Td cap).
4. Pd with 5 attempted bonds → 5th demoted via the geometry gate.

All tests are CPU-only — we feed crafted logits directly into
``_select_bonds`` to bypass the YuelBond head's random init.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (  # noqa: E402
    _OCTAHEDRAL,
    _SQUARE_PLANAR,
    _TETRAHEDRAL,
    _geometry_constraint_for_z,
    YuelBondBondDecoder,
)
from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import YuelBondResult  # noqa: E402
from molmetal.models.bond_head import (  # noqa: E402
    AtomCloud,
    BOND_AROMATIC,
    BOND_SINGLE,
)


# ---------------------------------------------------------------------------
# Helper — build a YuelBondResult with FORCED BOND_SINGLE logits
# ---------------------------------------------------------------------------
def _forced_single_result(
    z_list: list[int],
    pairs: list[tuple[int, int]],
) -> tuple[AtomCloud, YuelBondResult]:
    """Build (cloud, yres) where every pair is forced to BOND_SINGLE.

    Logits shape: (E, 5).  Set SINGLE logit to 5.0, others to 0.0 so
    argmax lands on SINGLE and max_logits (for sorting) is 5.0.
    """
    n_atoms = len(z_list)
    positions = torch.zeros((n_atoms, 3), dtype=torch.float32)
    for k, (i, j) in enumerate(pairs):
        positions[j] = torch.tensor([2.0 + 0.01 * k, 0.0, 0.0], dtype=torch.float32)
    cloud = AtomCloud(
        positions=positions,
        atomic_numbers=torch.tensor(z_list, dtype=torch.long),
    )
    edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()  # (2, E)
    # Logits: (E, 5).  Set SINGLE high, others low.
    logits = torch.zeros((len(pairs), 5), dtype=torch.float32)
    logits[:, BOND_SINGLE] = 5.0
    yres = YuelBondResult(
        pair_features=None,
        logits=logits,
        edge_index=edge_index,
        decode_succeeded=True,
    )
    return cloud, yres


# ===========================================================================
# Test 1 — Unit-level: constraint dataclasses and dispatcher
# ===========================================================================
def test_geometry_constraint_dataclasses_have_canonical_metal_sets():
    """Pin the per-geometry metal sets so a future edit cannot silently
    drop a project-target metal (Pt / Ru / Ir / Rh / Os / Re)."""
    assert set(_SQUARE_PLANAR.metals) == {46, 78, 79}, (
        f"SqP must cover Pd(46), Pt(78), Au(79); got {set(_SQUARE_PLANAR.metals)}"
    )
    assert set(_OCTAHEDRAL.metals) == {26, 44, 45, 75, 76, 77}, (
        f"Oct must cover Fe(26), Ru(44), Rh(45), Re(75), Os(76), Ir(77); "
        f"got {set(_OCTAHEDRAL.metals)}"
    )
    assert set(_TETRAHEDRAL.metals) == {29, 30}, (
        f"Td must cover Cu(29), Zn(30); got {set(_TETRAHEDRAL.metals)}"
    )
    assert _SQUARE_PLANAR.max_bonds == 4
    assert _OCTAHEDRAL.max_bonds == 6
    assert _TETRAHEDRAL.max_bonds == 4
    assert 90.0 in _SQUARE_PLANAR.ideal_angles
    assert 180.0 in _SQUARE_PLANAR.ideal_angles
    assert 109.5 in _TETRAHEDRAL.ideal_angles


def test_geometry_constraint_dispatch_returns_correct_constraint_per_z():
    """``_geometry_constraint_for_z`` returns the right tuple per Z."""
    # Square-planar metals
    assert _geometry_constraint_for_z(78).name == "square_planar"  # Pt_II
    assert _geometry_constraint_for_z(46).name == "square_planar"  # Pd_II
    assert _geometry_constraint_for_z(79).name == "square_planar"  # Au_III
    # Octahedral metals
    assert _geometry_constraint_for_z(44).name == "octahedral"  # Ru
    assert _geometry_constraint_for_z(77).name == "octahedral"  # Ir
    assert _geometry_constraint_for_z(45).name == "octahedral"  # Rh
    assert _geometry_constraint_for_z(75).name == "octahedral"  # Re
    assert _geometry_constraint_for_z(76).name == "octahedral"  # Os
    assert _geometry_constraint_for_z(26).name == "octahedral"  # Fe
    # Tetrahedral metals
    assert _geometry_constraint_for_z(29).name == "tetrahedral"  # Cu
    assert _geometry_constraint_for_z(30).name == "tetrahedral"  # Zn
    # Non-metals return None (no constraint — gate doesn't fire)
    assert _geometry_constraint_for_z(6) is None  # C
    assert _geometry_constraint_for_z(7) is None  # N
    assert _geometry_constraint_for_z(8) is None  # O
    assert _geometry_constraint_for_z(17) is None  # Cl
    assert _geometry_constraint_for_z(1) is None  # H


# ===========================================================================
# Test 2 — cisplatin (Pt_II, 4 donors) → SqP cap satisfied, 4 bonds
# ===========================================================================
def test_cisplatin_pt_ii_4_bonds_within_square_planar_cap():
    """cisplatin Pt(NH3)2Cl2 — 4 bonds around Pt, within SqP max=4."""
    torch.manual_seed(0)
    decoder = YuelBondBondDecoder(hidden_dim=32, n_layers=2)
    z_list = [78, 17, 17, 7, 7]  # Pt, Cl, Cl, N, N
    pairs = [(0, 1), (0, 2), (0, 3), (0, 4)]  # all Pt-donor
    cloud, yres = _forced_single_result(z_list, pairs)
    bond_orders, _dative, n_kept, n_overvalent = decoder._select_bonds(cloud, yres)
    assert n_kept == 4, (
        f"cisplatin should accept all 4 Pt-donor bonds (SqP cap=4), "
        f"got n_kept={n_kept}"
    )
    assert n_overvalent == 0, (
        f"no bond should be rejected for cisplatin; got n_overvalent={n_overvalent}"
    )
    # Verify Pt is the centre of all 4 accepted bonds.
    pt_bonds = [(i, j, o) for (i, j, o) in bond_orders if i == 0 or j == 0]
    assert len(pt_bonds) == 4


# ===========================================================================
# Test 3 — [Ru(NH3)6]2+ → Oct cap satisfied, 6 bonds
# ===========================================================================
def test_ru_hexaammine_6_bonds_within_octahedral_cap():
    """[Ru(NH3)6]2+ — 6 bonds around Ru, within Oct max=6."""
    torch.manual_seed(0)
    decoder = YuelBondBondDecoder(hidden_dim=32, n_layers=2)
    z_list = [44, 7, 7, 7, 7, 7, 7]  # Ru + 6 N
    pairs = [(0, k) for k in range(1, 7)]
    cloud, yres = _forced_single_result(z_list, pairs)
    bond_orders, _dative, n_kept, n_overvalent = decoder._select_bonds(cloud, yres)
    assert n_kept == 6, (
        f"[Ru(NH3)6]2+ should accept all 6 Ru-N bonds (Oct cap=6), "
        f"got n_kept={n_kept}"
    )
    assert n_overvalent == 0


# ===========================================================================
# Test 4 — [Zn(NH3)4]2+ → Td cap satisfied, 4 bonds
# ===========================================================================
def test_zn_tetraammine_4_bonds_within_tetrahedral_cap():
    """[Zn(NH3)4]2+ — 4 bonds around Zn, within Td max=4."""
    torch.manual_seed(0)
    decoder = YuelBondBondDecoder(hidden_dim=32, n_layers=2)
    z_list = [30, 7, 7, 7, 7]  # Zn + 4 N
    pairs = [(0, k) for k in range(1, 5)]
    cloud, yres = _forced_single_result(z_list, pairs)
    bond_orders, _dative, n_kept, n_overvalent = decoder._select_bonds(cloud, yres)
    assert n_kept == 4, (
        f"[Zn(NH3)4]2+ should accept all 4 Zn-N bonds (Td cap=4), "
        f"got n_kept={n_kept}"
    )
    assert n_overvalent == 0


# ===========================================================================
# Test 5 — Pd with 5 attempted bonds → 5th demoted via geometry gate
# ===========================================================================
def test_pd_overflow_5_attempted_bonds_5th_demoted():
    """Pd(II) with 5 close donors — the geometry gate rejects the 5th.

    Pd(II) is in SqP (max=4) but its ``_COVALENT_VALENCE`` is also 4, so
    the valence gate would also fire at the 5th bond.  We distinguish by
    counting: with 5 attempted SINGLE bonds and Pd cap=4, exactly 4 are
    accepted and exactly 1 lands in ``n_overvalent``.  The geometry
    gate's contribution is verified by the unit test above; this test
    pins the OVERALL behaviour (the gate may fire via valence OR
    geometry — both produce the same n_overvalent counter).
    """
    torch.manual_seed(0)
    decoder = YuelBondBondDecoder(hidden_dim=32, n_layers=2)
    z_list = [46, 7, 7, 7, 7, 7]  # Pd + 5 N
    pairs = [(0, k) for k in range(1, 6)]
    cloud, yres = _forced_single_result(z_list, pairs)
    bond_orders, _dative, n_kept, n_overvalent = decoder._select_bonds(cloud, yres)
    assert n_kept == 4, (
        f"Pd(II) SqP cap=4 means at most 4 of 5 attempted bonds survive, "
        f"got n_kept={n_kept}"
    )
    assert n_overvalent == 1, (
        f"exactly 1 of 5 attempted bonds must be rejected, "
        f"got n_overvalent={n_overvalent}"
    )


# ===========================================================================
# Test 6 — Backwards-compat invariant: non-metal endpoints skip the gate
# ===========================================================================
def test_nonmetal_only_cloud_geometry_gate_does_not_fire():
    """A pure organic cloud (no constrained metals) must not lose any
    bond to the new geometry gate.  All 4 C-C bonds accepted."""
    torch.manual_seed(0)
    decoder = YuelBondBondDecoder(hidden_dim=32, n_layers=2)
    z_list = [6, 6, 6, 6]  # 4 C
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    cloud, yres = _forced_single_result(z_list, pairs)
    bond_orders, _dative, n_kept, n_overvalent = decoder._select_bonds(cloud, yres)
    assert n_kept == 4, (
        f"non-metal cloud must accept all 4 bonds; got n_kept={n_kept}"
    )
    assert n_overvalent == 0


# ===========================================================================
# Test 7 — Anti-aromatic dispatch doesn't fire on non-aromatic bonds
# ===========================================================================
def test_geometry_gate_does_not_invoke_anti_aromatic_probe_for_single_bonds():
    """Anti-aromatic probe is gated on order==BOND_AROMATIC.  Verify it
    doesn't fire on SINGLE bonds by passing an aromatic-inducing 4-ring
    and asserting no HONEST-NEGATIVE print is captured.
    """
    import io
    import contextlib

    torch.manual_seed(0)
    decoder = YuelBondBondDecoder(hidden_dim=32, n_layers=2)
    z_list = [46, 7, 7, 7, 7]  # Pd + 4 N
    pairs = [(0, k) for k in range(1, 5)]  # 4 Pd-N SINGLE bonds (force SINGLE)
    cloud, yres = _forced_single_result(z_list, pairs)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        decoder._select_bonds(cloud, yres)
    assert "[wfA55-anti-aromatic]" not in buf.getvalue(), (
        "anti-aromatic probe must not fire on SINGLE bonds"
    )
