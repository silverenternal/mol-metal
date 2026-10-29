# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 T77 — the metal budget is a COUNT; it must be charged in counts.

WHY THIS FILE EXISTS
--------------------
``_select_bonds`` starts each atom at ``_cap_for_z(Z)`` and subtracts
``_CLASS_VALENCE_COST[class]`` per accepted bond (aromatic 1.5, double 2.0).

For a carbon that is right: ``_COVALENT_VALENCE[6] = 4`` is a bond-ORDER sum.

For a metal it is not.  T72 set the metal caps from tmQM's ``coord_number``
(MND p99: Ru 10, Rh 10, Os 10, Ir 9, Re 9, Pt 4) — a COUNT of metal-ligand
bonds.  Charging 1.5 for an aromatic-class metal bond spends 1.5 of a
budget denominated in bonds, so the selector stops before the coordination
number the cap was derived from.

MEASURED over all 86k complexes
(``metrics/by_metric/wfa113_t77_metal_bond_cost_units.json``): the cost sum
exceeded the bond count in 92.3% of Re complexes (median cost 7.5 against
median count 6.0), 27.8% of Ru, 12.1% of Rh.  Re carries 6,613 aromatic and
1,498 double metal-incident bonds out of 26,688.

The fix charges a metal bond 1.0 regardless of class.  The ligand endpoint
still pays its own order cost, because a carbon's ceiling really is an
order sum.

What this file pins
-------------------
1. A Re centre with nine bonds, six of them aromatic-class, keeps all nine.
2. The ligand endpoint is unchanged: a carbon still refuses a third
   aromatic bond (3 x 1.5 > 4).
3. ``_bridge_components`` uses the same unit rule, so it cannot add a bond
   the selector's accounting would have refused.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (  # noqa: E402
    YuelBondBondDecoder,
)
from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import YuelBondResult  # noqa: E402
from molmetal.models.bond_head import (  # noqa: E402
    AtomCloud,
    BOND_AROMATIC,
    BOND_SINGLE,
    BOND_TRIPLE,
)

RE = 75


def _result(z_list, pairs, classes):
    """A cloud whose edges are forced to the given class per pair."""
    n_atoms = len(z_list)
    positions = torch.zeros((n_atoms, 3), dtype=torch.float32)
    for k in range(1, n_atoms):
        positions[k] = torch.tensor(
            [1.7 + 0.01 * k, 0.05 * k, 0.0], dtype=torch.float32
        )
    cloud = AtomCloud(
        positions=positions,
        atomic_numbers=torch.tensor(z_list, dtype=torch.long),
    )
    edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    logits = torch.zeros((len(pairs), 5), dtype=torch.float32)
    for k, cls in enumerate(classes):
        logits[k, cls] = 5.0
    yres = YuelBondResult(
        pair_features=None,
        logits=logits,
        edge_index=edge_index,
        decode_succeeded=True,
    )
    return cloud, yres


def _decoder() -> YuelBondBondDecoder:
    torch.manual_seed(0)
    return YuelBondBondDecoder(hidden_dim=32, n_layers=2)


def test_re_keeps_nine_bonds_when_six_are_aromatic_class():
    """Re + 9 donors, six bonds classed aromatic.

    Under the old accounting the metal was charged 6 x 1.5 + 3 x 1.0 = 12
    against a cap of 9, so three bonds were dropped and the coordination
    sphere came out at six.
    """
    z_list = [RE] + [7] * 9
    pairs = [(0, k) for k in range(1, 10)]
    classes = [BOND_AROMATIC] * 6 + [BOND_SINGLE] * 3
    cloud, yres = _result(z_list, pairs, classes)
    bond_orders, _dative, n_kept, n_overvalent = _decoder()._select_bonds(cloud, yres)
    assert n_kept == 9, (
        f"Re must keep all 9 metal bonds; got {n_kept} "
        f"(n_overvalent={n_overvalent}).  A metal ceiling derived from a "
        "coordination number cannot be spent in bond-order units."
    )
    assert n_overvalent == 0


def test_ligand_endpoint_still_pays_the_order_cost():
    """A carbon still refuses bonds past its order sum of 4.

    A triple bond costs 3, so a triple plus an aromatic (1.5) is 4.5 and the
    second bond must be refused.  This is the property that must survive the
    aromatic rules: the ligand endpoint is NOT free in general, only for a
    bond to a metal (T79) and for the Kekule-single cases (T78, T81).

    The earlier form of this test asserted that THREE aromatic bonds at a
    carbon are refused, i.e. 3 x 1.5 = 4.5 > 4.  That is the fusion-carbon
    defect T81 fixes: a ring-fusion carbon really has three aromatic bonds,
    and no Kekule structure gives it two doubles, so the true sum is 4.
    """
    z_list = [6, 6, 7]
    pairs = [(0, 1), (0, 2)]
    cloud, yres = _result(z_list, pairs, [BOND_TRIPLE, BOND_AROMATIC])
    bond_orders, _dative, n_kept, _n_over = _decoder()._select_bonds(cloud, yres)
    assert n_kept == 1, (
        "a carbon with a triple bond has 1 of its 4 valence slots left; an "
        f"aromatic bond at 1.5 does not fit (kept {n_kept})"
    )
    assert bond_orders[0][2] == BOND_TRIPLE


def test_fusion_carbon_keeps_all_three_aromatic_bonds():
    """A ring-fusion carbon has three aromatic bonds and must keep all three.

    Kekule gives it one double and two single, so the sum is 4, not 4.5.
    Before T81 the selector charged 1.5 for each, hit 4.5 against a cap of 4,
    and dropped the third bond -- which breaks the fusion and splits the ring
    system.
    """
    z_list = [6, 6, 6, 6]
    pairs = [(0, 1), (0, 2), (0, 3)]
    cloud, yres = _result(z_list, pairs, [BOND_AROMATIC] * 3)
    _bo, _dative, n_kept, n_over = _decoder()._select_bonds(cloud, yres)
    assert (n_kept, n_over) == (3, 0)


def test_bridge_component_uses_the_same_unit_rule():
    """The bridge must not add a bond the selector's budget refuses.

    Six aromatic metal bonds charge 6 of Re's 9 under the fix (9 under the
    old rule).  A seventh candidate must be accepted under the fix and
    refused under the old rule -- this pins the direction.
    """
    z_list = [RE] + [7] * 6
    pairs = [(0, k) for k in range(1, 7)]
    cloud, yres = _result(z_list, pairs, [BOND_AROMATIC] * 6)
    dec = _decoder()
    bond_orders, dative, n_kept, _n_over = dec._select_bonds(cloud, yres)
    assert n_kept == 6

    # Now feed a bond list that has already consumed the metal budget under
    # the OLD rule (9 of 9) and ask the bridge for one more.
    consumed_old = [(0, k, BOND_AROMATIC) for k in range(1, 7)]
    cloud2, yres2 = _result(
        [RE] + [7] * 7,
        [(0, k) for k in range(1, 8)],
        [BOND_AROMATIC] * 7,
    )
    added_new, _d, n_added_new = dec._bridge_components(
        cloud2, yres2, list(consumed_old), []
    )
    # Under the count rule the metal has spent 6 of 9, so the 7th is allowed.
    assert n_added_new == 1, (
        "the bridge must charge a metal bond 1.0 like the selector does; "
        f"it added {n_added_new} bond(s)"
    )
