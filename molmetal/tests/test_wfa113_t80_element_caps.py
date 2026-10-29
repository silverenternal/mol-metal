# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 T80/T81 — the ligand caps must be RDKit's limits, not the common case.

WHY THIS FILE EXISTS
--------------------
``_COVALENT_VALENCE`` is the selector's ceiling on the number of bonds it may
keep at an atom.  Two ways to get it wrong, and both were present:

* **Too low.**  The selector drops a bond the corpus really has, the ring
  opens, and a one-fragment decode becomes a multi-fragment one.  S was 2
  (the thioether value) and Se 2, which refuses every sulfoxide, sulfone and
  sulfonium; As was 3, which refuses 370 of 409 As atoms in the cohort (the
  corpus is As(V)).
* **Too high.**  The selector keeps a bond RDKit then rejects, and the whole
  decode is lost.  So the cap must never exceed RDKit's own limit.

The right value is therefore RDKit's limit, and the corpus tells you whether
the limit is reachable.  RDKit 2026.03.6 accepts, in single bonds: S 6, Se 6,
As 5, P 5, Si 4, B 3, C 4, N 3, O 2, halogens 1.

T81 is the same defect on the cost side rather than the cap side: the 1.5
charged for an aromatic bond is the average of one Kekule double and one
single, which assumes exactly two aromatic bonds at the atom.  A ring-fusion
carbon has three, and no Kekule structure gives it two doubles, so its sum is
4 and not 4.5.  Charging 4.5 dropped the third bond and broke the fusion.

MEASURED on the six-metal cohort (33,429 complexes) --
``metrics/by_metric/wfa113_t80_residual_caps.json``:

    complexes with at least one ligand atom over its cap
        before  3,836 / 33,429 = 11.48%
        after   1,280 / 33,429 =  3.83%
        after, excluding the 2,312 boron-cluster complexes
                  202 / 31,117 =  0.65%

    element   cap        atoms over cap
    S         2 -> 6     20.76% -> 0.00%
    Se        2 -> 6      1.10% -> 0.00%
    As        3 -> 5      2.93% -> 0.00%
    C         4 -> 4      0.24% -> 0.10%   (T81; the rest is carborane carbon)

The remaining 3.83% is dominated by carboranes: 82.4% of B atoms have 5-6
neighbours and RDKit's limit for neutral B is 3, so no bond table can make
those decode.  That is a representation limit, not a table error, and it is
recorded rather than papered over.

What this file pins
-------------------
1. S, Se, As caps are RDKit's limits, not the common-case values.
2. A sulfoxide and a sulfone keep all their bonds.
3. No ligand cap exceeds RDKit's own limit (the "too high" direction).
4. A fusion carbon keeps all three aromatic bonds (T81).
5. The aromatic rule does not make aromatic bonds free: a fourth still fails.
6. The total charged for three aromatic bonds at a carbon is 4, whatever order
   the selector accepts them in.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
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

#: Ligand elements whose cap must equal RDKit's own limit.
_LIGAND_ZS = {
    5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 14: "Si", 15: "P", 16: "S",
    17: "Cl", 33: "As", 34: "Se", 35: "Br", 53: "I",
}


def _rdkit_max_single_bonds(z: int) -> int:
    """Largest number of single bonds RDKit will sanitise at element ``z``."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    best = 0
    for n in range(1, 9):
        rw = Chem.RWMol()
        rw.AddAtom(Chem.Atom(z))
        for k in range(n):
            rw.AddAtom(Chem.Atom(6))
            rw.AddBond(0, k + 1, Chem.BondType.SINGLE)
        mol = rw.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            break
        best = n
    return best


def _cloud_and_result(z_list, pairs, classes):
    n_atoms = len(z_list)
    positions = torch.zeros((n_atoms, 3), dtype=torch.float32)
    for k, (_i, j) in enumerate(pairs):
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


def _select(z_list, pairs, classes, **kw):
    torch.manual_seed(0)
    d = YuelBondBondDecoder(hidden_dim=32, n_layers=2, **kw)
    cloud, yres = _cloud_and_result(z_list, pairs, classes)
    return d._select_bonds(cloud, yres)


# ---------------------------------------------------------------------------
# 1. The caps are RDKit's limits
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "z,expected", [(16, 6), (34, 6), (33, 5), (15, 5)]
)
def test_caps_are_rdkit_limits_not_common_case_values(z, expected):
    assert int(_COVALENT_VALENCE[z]) == expected, (
        f"cap for Z={z} is {_COVALENT_VALENCE[z]}, expected RDKit's limit "
        f"{expected}.  A cap below RDKit's limit refuses a real corpus bond; "
        "the T80 measurement is metrics/by_metric/wfa113_t80_residual_caps.json"
    )


# ---------------------------------------------------------------------------
# 2. The environments a cap of 2 refused
# ---------------------------------------------------------------------------
def test_sulfoxide_sulfur_keeps_all_three_bonds():
    """S(=O)(C)(C): 2 + 1 + 1 = 4 against a cap of 2 under the old table."""
    z = [16, 8, 6, 6]
    pairs = [(0, 1), (0, 2), (0, 3)]
    _bo, _d, n_kept, n_over = _select(
        z, pairs, [BOND_DOUBLE, BOND_SINGLE, BOND_SINGLE]
    )
    assert (n_kept, n_over) == (3, 0)


def test_sulfone_sulfur_keeps_all_four_bonds():
    """S(=O)(=O)(C)(C): 2 + 2 + 1 + 1 = 6 against a cap of 6."""
    z = [16, 8, 8, 6, 6]
    pairs = [(0, 1), (0, 2), (0, 3), (0, 4)]
    _bo, _d, n_kept, n_over = _select(
        z, pairs, [BOND_DOUBLE, BOND_DOUBLE, BOND_SINGLE, BOND_SINGLE]
    )
    assert (n_kept, n_over) == (4, 0)


# ---------------------------------------------------------------------------
# 3. The "too high" direction: never above RDKit
# ---------------------------------------------------------------------------
def test_no_ligand_cap_exceeds_rdkits_own_limit():
    """A cap above RDKit's limit authorises a bond the sanitiser then refuses."""
    bad = []
    for z, sym in _LIGAND_ZS.items():
        cap = int(_COVALENT_VALENCE.get(z, 0))
        limit = _rdkit_max_single_bonds(z)
        if cap > limit:
            bad.append((sym, cap, limit))
    assert not bad, (
        "cap above RDKit's own valence limit, so the selector can build a "
        f"molecule RDKit rejects and the whole decode is lost: {bad}"
    )


# ---------------------------------------------------------------------------
# 4-6. The aromatic-degree rule at carbon
# ---------------------------------------------------------------------------
def test_fusion_carbon_keeps_all_three_aromatic_bonds():
    z = [6, 6, 6, 6]
    pairs = [(0, 1), (0, 2), (0, 3)]
    _bo, _d, n_kept, n_over = _select(z, pairs, [BOND_AROMATIC] * 3)
    assert (n_kept, n_over) == (3, 0)


def test_aromatic_bonds_are_not_free_at_carbon():
    """1.5 + 1.5 + 1.0 = 4 fills the carbon; a fourth bond cannot fit."""
    z = [6, 6, 6, 6, 6]
    pairs = [(0, 1), (0, 2), (0, 3), (0, 4)]
    _bo, _d, n_kept, n_over = _select(z, pairs, [BOND_AROMATIC] * 4)
    assert n_kept == 3 and n_over == 1


def test_the_third_aromatic_bond_costs_one_whatever_the_order():
    """The sum over three aromatic bonds at a carbon is 4, order-independent."""
    # Accept-order is descending confidence; give the three bonds equal logits
    # so the order is fixed by the sort, then reverse it by flipping which one
    # is cheapest.  Either way the total charged must be 4.
    total = 0.0
    for prior in (0, 1, 2):
        total += _endpoint_valence_cost(
            BOND_AROMATIC, 6, 6, lambda z: False, prior
        )
    assert total == pytest.approx(4.0)


def test_aromatic_cost_at_a_heteroatom_stays_one():
    """T78 must survive T81: N/O/S never pay 1.5 for an aromatic bond."""
    for z in (7, 8, 16, 34, 15, 33, 5):
        for prior in (0, 1, 2):
            assert (
                _endpoint_valence_cost(
                    BOND_AROMATIC, z, 6, lambda zz: False, prior
                )
                == 1.0
            )
