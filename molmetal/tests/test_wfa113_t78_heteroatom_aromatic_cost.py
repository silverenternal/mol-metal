# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 T78 — an aromatic bond at a heteroatom is Kekulé-single for valence.

WHY THIS FILE EXISTS
--------------------
``_CLASS_VALENCE_COST[BOND_AROMATIC] = 1.5`` is the average of the Kekulé
single/double pair.  That is right for carbon — benzene C is 2 x 1.5 + 1 H = 4,
exactly its cap — and wrong for a heteroatom, which carries the aromatic bond
as a lone pair (pyrrole N, furan O, thiophene S) or as the Kekulé partner with
no second substituent (pyridine N).  Billing 1.5 there over-charges every ring
by 0.5 per bond, so the selector drops a real bond and the molecule splits:

    pyrrole N    2 x 1.5 + 1 H = 4.0  >  cap 3
    furan O      2 x 1.5       = 3.0  >  cap 2
    aryl amide N 1.5 + 1 + 1   = 3.5  >  cap 3

MEASURED on the six-metal cohort (33,429 complexes, every bond >= 0.3,
hydrogens dropped as the trainer drops them): 61.7% of N atoms and 30.7% of O
atoms exceeded their cap on the CORPUS's own graph, i.e. the decoder could not
rebuild the truth it was trained on.  With the rule below: N 3.6%, O 9.2%.
Carbon is unchanged at 1.2% — the control.

The same probe found the P cap of 3 refusing the 4th bond on 98.8% of P atoms
(the corpus is phosphate, P with four single bonds, and P appears in 7.7% of
the cohort's complexes).  Cap is now 5; after: 0.03%.

What this file pins
-------------------
1. Pyrrole N keeps both aromatic ring bonds plus its H (3 bonds).
2. Furan O keeps both aromatic ring bonds (2 bonds).
3. Carbon is UNCHANGED: a carbon still cannot take three aromatic bonds plus
   a single one (3 x 1.5 + 1 = 5.5 > 4).
4. Phosphate P keeps all four single bonds.
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
)


def _result(z_list, pairs, classes):
    n_atoms = len(z_list)
    positions = torch.zeros((n_atoms, 3), dtype=torch.float32)
    for k in range(1, n_atoms):
        positions[k] = torch.tensor(
            [1.4 + 0.01 * k, 0.05 * k, 0.0], dtype=torch.float32
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


def test_pyrrole_nitrogen_keeps_three_bonds():
    """N in a 5-ring: two aromatic C-N bonds plus N-H."""
    z_list = [7, 6, 6, 1]
    pairs = [(0, 1), (0, 2), (0, 3)]
    cloud, yres = _result(z_list, pairs, [BOND_AROMATIC, BOND_AROMATIC, BOND_SINGLE])
    _bo, _d, n_kept, n_over = _decoder()._select_bonds(cloud, yres)
    assert n_kept == 3, (
        f"pyrrole N must keep 3 bonds, got {n_kept} (n_overvalent={n_over}). "
        "Charging 1.5 per aromatic bond at N makes the cap unreachable."
    )


def test_furan_oxygen_keeps_both_ring_bonds():
    z_list = [8, 6, 6]
    pairs = [(0, 1), (0, 2)]
    cloud, yres = _result(z_list, pairs, [BOND_AROMATIC, BOND_AROMATIC])
    _bo, _d, n_kept, _n_over = _decoder()._select_bonds(cloud, yres)
    assert n_kept == 2, f"furan O must keep both aromatic bonds, got {n_kept}"


def test_carbon_still_pays_the_kekule_average():
    """The control: carbon's aromatic bond stays at 1.5.

    Three aromatic bonds plus one single = 5.5 against a cap of 4, so at
    least one is refused.  If this test ever passes with 4 kept, the rule
    has been applied to carbon and the carbon cap has stopped meaning
    anything.
    """
    z_list = [6, 6, 6, 6, 7]
    pairs = [(0, 1), (0, 2), (0, 3), (0, 4)]
    cloud, yres = _result(
        z_list, pairs, [BOND_AROMATIC, BOND_AROMATIC, BOND_AROMATIC, BOND_SINGLE]
    )
    _bo, _d, n_kept, n_over = _decoder()._select_bonds(cloud, yres)
    assert n_kept < 4, (
        "carbon must not be exempted from the Kekule average: 3 aromatic + 1 "
        f"single costs 5.5 against a cap of 4 (kept {n_kept})"
    )
    assert n_over >= 1


def test_phosphate_phosphorus_keeps_four_bonds():
    z_list = [15, 8, 8, 8, 8]
    pairs = [(0, k) for k in range(1, 5)]
    cloud, yres = _result(z_list, pairs, [BOND_SINGLE] * 4)
    _bo, _d, n_kept, _n_over = _decoder()._select_bonds(cloud, yres)
    assert n_kept == 4, (
        f"phosphate P must keep four P-O bonds, got {n_kept}.  The corpus is "
        "P(4) and the cap used to be 3."
    )
