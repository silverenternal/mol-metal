# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 T79 — the ligand endpoint must not pay for its bond to the metal.

WHY THIS FILE EXISTS
--------------------
``_select_bonds`` charges every accepted bond against both endpoints' valence
budget.  For a metal-ligand bond that is right at the METAL (its budget is a
count of coordination bonds, T77) and wrong at the LIGAND: the electron pair
is the ligand's lone pair, so the donor's valence is unchanged.

RDKit implements exactly this.  ``SanitizeMol`` runs
``cleanupOrganometallics()``, which rewrites a single bond between a metal and
a hypervalent atom into ``BondType.DATIVE``, and a dative bond does not count
towards the donor's explicit valence.  Upstream calls it "the organometallic
cleanup code (this converts single bonds between hypervalent atoms and metals
to dative bonds)" (rdkit/rdkit#8312).

Charging the ligand therefore made the selector STRICTER than the validator it
feeds.  It dropped a bond RDKit would have kept, which opens a ring and turns a
one-fragment decode into a multi-fragment one.

MEASURED on the six-metal cohort (33,429 complexes, heavy atoms only, the graph
the decoder actually sees) — ``metrics/by_metric/wfa113_t79_dative_ligand_budget.json``:

    complexes with at least one ligand atom over its cap
        before 13,591 / 33,429 = 40.66%
        after   3,836 / 33,429 = 11.48%

    ligand atoms over their cap, by element (before -> after)
        O   9.18% -> 0.07%      (9.11 pp of it caused by the metal bond alone)
        N   3.64% -> 0.13%      (3.51 pp)
        Se 26.29% -> 1.10%     (25.18 pp)
        S  34.72% -> 20.76%    (13.95 pp)
        C   1.23% -> 0.24%     (0.99 pp)

No ligand atom in the cohort has two metal neighbours (0 of 193,703), so
RDKit's one-dative-bond-per-atom limit is never reached here.

What this file pins
-------------------
1. ``_endpoint_valence_cost`` charges the ligand 0, the metal 1.0.
2. An ether O with two C and one Ru keeps all three bonds.
3. A carbonyl O with a double bond and one Ru keeps both.
4. A thioether S with two C and one Ru keeps all three.
5. The metal still pays: a metal over its cap still loses the bond.
6. ``_bridge_components`` uses the same rule (it used to carry a copy).
7. RDKit actually accepts the graph the selector now keeps — the relaxation is
   the validator's own behaviour, not an assumption about it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (  # noqa: E402
    _COVALENT_VALENCE,
    _endpoint_valence_cost,
    YuelBondBondDecoder,
)
from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import YuelBondResult  # noqa: E402
from molmetal.models.bond_head import (  # noqa: E402
    AtomCloud,
    BOND_AROMATIC,
    BOND_DOUBLE,
    BOND_SINGLE,
)

_METAL_ZS = frozenset({44, 45, 75, 76, 77, 78})


def _is_metal_z(z: int) -> bool:
    return int(z) in _METAL_ZS


def _cloud_and_result(z_list, pairs, classes):
    """Build an AtomCloud + YuelBondResult with one forced class per pair."""
    n_atoms = len(z_list)
    positions = torch.zeros((n_atoms, 3), dtype=torch.float32)
    for k, (_i, j) in enumerate(pairs):
        # Distinct, well-inside-cutoff distances so the confidence sort is
        # deterministic and geometry never decides the outcome.
        positions[j] = torch.tensor(
            [1.6 + 0.01 * k, 0.05 * k, 0.0], dtype=torch.float32
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


def _decoder(**kw) -> YuelBondBondDecoder:
    torch.manual_seed(0)
    return YuelBondBondDecoder(hidden_dim=32, n_layers=2, **kw)


# ---------------------------------------------------------------------------
# 1. The pure function
# ---------------------------------------------------------------------------
def test_endpoint_cost_is_zero_for_the_ligand_and_one_for_the_metal():
    # ligand -> metal
    assert _endpoint_valence_cost(BOND_SINGLE, 8, 44, _is_metal_z) == 0.0
    assert _endpoint_valence_cost(BOND_DOUBLE, 8, 44, _is_metal_z) == 0.0
    # metal -> ligand
    assert _endpoint_valence_cost(BOND_SINGLE, 44, 8, _is_metal_z) == 1.0
    assert _endpoint_valence_cost(BOND_DOUBLE, 44, 8, _is_metal_z) == 1.0
    # ligand -> ligand keeps the class cost
    assert _endpoint_valence_cost(BOND_SINGLE, 6, 6, _is_metal_z) == 1.0
    assert _endpoint_valence_cost(BOND_DOUBLE, 6, 6, _is_metal_z) == 2.0
    assert _endpoint_valence_cost(BOND_AROMATIC, 6, 6, _is_metal_z) == 1.5
    # metal -> metal: both sides are the metal rule, never 0
    assert _endpoint_valence_cost(BOND_SINGLE, 44, 45, _is_metal_z) == 1.0


# ---------------------------------------------------------------------------
# 2-4. The three donor types the corpus is full of
# ---------------------------------------------------------------------------
def test_ether_oxygen_keeps_its_metal_bond():
    """O with 2 C and 1 Ru: 3 bonds, cap 2, all three must survive."""
    z = [8, 6, 6, 44]
    pairs = [(0, 1), (0, 2), (0, 3)]
    cloud, yres = _cloud_and_result(z, pairs, [BOND_SINGLE] * 3)
    d = _decoder()
    bond_orders, _dative, n_kept, n_over = d._select_bonds(cloud, yres)
    assert n_kept == 3, (
        "the ether O lost a bond to the metal: RDKit accepts this graph "
        "(cleanupOrganometallics makes the M-O bond dative), so the selector "
        "must not refuse it"
    )
    assert n_over == 0
    assert {frozenset((i, j)) for i, j, _o in bond_orders} == {
        frozenset(p) for p in pairs
    }


def test_carbonyl_oxygen_keeps_its_metal_bond():
    """O with one double bond to C and one single to Ru: cap 2, keep both."""
    z = [8, 6, 44]
    pairs = [(0, 1), (0, 2)]
    cloud, yres = _cloud_and_result(z, pairs, [BOND_DOUBLE, BOND_SINGLE])
    d = _decoder()
    _bo, _dative, n_kept, n_over = d._select_bonds(cloud, yres)
    assert (n_kept, n_over) == (2, 0)


def test_thioether_sulfur_keeps_its_metal_bond():
    z = [16, 6, 6, 44]
    pairs = [(0, 1), (0, 2), (0, 3)]
    cloud, yres = _cloud_and_result(z, pairs, [BOND_SINGLE] * 3)
    d = _decoder()
    _bo, _dative, n_kept, n_over = d._select_bonds(cloud, yres)
    assert (n_kept, n_over) == (3, 0)


# ---------------------------------------------------------------------------
# 5. The metal side is unchanged
# ---------------------------------------------------------------------------
def test_the_metal_still_pays_one_per_bond():
    """Pd's cap is 4; a fifth metal bond is still refused."""
    z = [46] + [7] * 5
    pairs = [(0, k) for k in range(1, 6)]
    cloud, yres = _cloud_and_result(z, pairs, [BOND_SINGLE] * 5)
    d = _decoder()
    _bo, _dative, n_kept, n_over = d._select_bonds(cloud, yres)
    assert int(_COVALENT_VALENCE[46]) == 4
    assert n_kept == 4 and n_over == 1


# ---------------------------------------------------------------------------
# 6. The bridge uses the same rule
# ---------------------------------------------------------------------------
def test_bridge_component_uses_the_same_rule():
    """A saturated ether O must still accept a metal bridge.

    ``_bridge_components`` carried its own copy of the accounting.  With the
    ligand paying for the metal bond its remaining budget is 0, so the bridge
    was refused and the Ru stayed a separate fragment.
    """
    z = [44, 8, 6, 6]
    pairs = [(1, 2), (1, 3), (0, 1)]
    cloud, yres = _cloud_and_result(z, pairs, [BOND_SINGLE] * 3)
    d = _decoder(connectivity_prior=True, connectivity_prior_min_logit=-1e9)
    # The selector keeps the two C-O bonds and the Ru-O bond; drop the Ru-O
    # one to simulate a fragment the prior has to repair.
    bond_orders = [(1, 2, BOND_SINGLE), (1, 3, BOND_SINGLE)]
    out, _dative, n_added = d._bridge_components(cloud, yres, bond_orders, [])
    assert n_added == 1, (
        "the bridge refused a metal bond into a saturated ether O; the ligand "
        "endpoint does not pay for a dative bond"
    )
    assert any({i, j} == {0, 1} for i, j, _o in out)


# ---------------------------------------------------------------------------
# 7. RDKit agrees — the relaxation is the validator's own behaviour
# ---------------------------------------------------------------------------
def test_rdkit_accepts_the_graph_the_selector_keeps():
    """Assemble and sanitise the ether-O + Ru decode; it must be a valid mol."""
    from rdkit import Chem

    z = [44, 8, 6, 6]
    pairs = [(0, 1), (1, 2), (1, 3)]
    cloud, yres = _cloud_and_result(z, pairs, [BOND_SINGLE] * 3)
    d = _decoder()
    bond_orders, dative, _n_kept, _n_over = d._select_bonds(cloud, yres)
    decoded = d._assemble_mol(cloud, bond_orders, dative)
    assert decoded.sanitized, decoded.error
    mol = Chem.MolFromSmiles(decoded.smiles)
    assert mol is not None
    # RDKit's own conversion is what makes the third bond legal.
    assert any(b.GetBondType() == Chem.BondType.DATIVE for b in decoded.mol.GetBonds())
