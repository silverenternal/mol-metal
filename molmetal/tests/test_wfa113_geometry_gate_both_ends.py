# SPDX-License-Identifier: Apache-2.0
"""T51 — the coordination-geometry gate must count both ends of a bond.

THE DEFECT
----------
``_select_bonds`` gates a metal's bond count against the coordination
geometry's ``max_bonds``:

    bonds_at_metal = sum(1 for (a, _, _) in bond_orders if a == metal_idx)

Candidate pairs come from ``torch.triu_indices(n, n, offset=1)``, so every
stored pair has ``a < b``.  A metal that sits at a HIGHER index than its
partner is therefore never counted, and the gate never fires for it.

MEASURED over 9,111 tmQM metal complexes: 7,700 (84.5%) put the metal at
heavy index 0, and 11.84% (5,729 of 48,371) of metal bonds have the metal at
the higher index.

WHY THE DEFECT WAS INVISIBLE
----------------------------
The gate is redundant with ``_cap_for_z`` for every metal in the table —
``max_bonds`` equals the covalent cap there, so the valence check just above
already refuses the same bonds.  It becomes the binding constraint only when
a caller raises the cap past the geometry, which is exactly the Pt_IV case
(``valence_cap={78: 6}`` against square planar's 4).  The tests below use
that configuration, because it is the only one where the gate decides
anything.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

torch = pytest.importorskip("torch")
pytest.importorskip("rdkit")

from molmetal.models.bond_head import AtomCloud  # noqa: E402
from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (  # noqa: E402
    YuelBondBondDecoder,
)
from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (  # noqa: E402
    YuelBondResult,
)

N_CLASSES = 5
NO_BOND = 0
BOND_SINGLE = 1
PT = 78


def _five_coordinate_cloud(metal_index: int):
    """One Pt plus five carbons on a pentagon of radius 1.9 A.

    Every Pt-C pair is 1.9 A and every C-C pair is 2.23 A, so all fifteen
    pairs are candidate edges; the logits below make the five Pt-C pairs the
    only ones the selector wants.
    """
    r = 1.9
    carbons = [
        [r * math.cos(2 * math.pi * k / 5), r * math.sin(2 * math.pi * k / 5), 0.0]
        for k in range(5)
    ]
    pts, zs, ci = [], [], 0
    for k in range(6):
        if k == metal_index:
            pts.append([0.0, 0.0, 0.0])
            zs.append(PT)
        else:
            pts.append(carbons[ci])
            zs.append(6)
            ci += 1
    return AtomCloud(
        positions=torch.tensor(pts, dtype=torch.float32),
        atomic_numbers=torch.tensor(zs, dtype=torch.long),
    )


def _metal_only_logits(cloud: AtomCloud) -> YuelBondResult:
    """BOND_SINGLE on every metal-involving pair, NO_BOND everywhere else."""
    n = cloud.atomic_numbers.shape[0]
    z = [int(v) for v in cloud.atomic_numbers.tolist()]
    i, j = torch.triu_indices(n, n, offset=1)
    edge_index = torch.stack([i, j], dim=0)
    logits = torch.full((edge_index.shape[1], N_CLASSES), -10.0)
    for k in range(edge_index.shape[1]):
        if z[int(i[k])] == PT or z[int(j[k])] == PT:
            logits[k, NO_BOND] = 0.0
            logits[k, BOND_SINGLE] = 10.0
        else:
            logits[k, NO_BOND] = 10.0
    return YuelBondResult(
        pair_features=None,
        logits=logits,
        edge_index=edge_index,
        decode_succeeded=True,
    )


def _decode(cloud: AtomCloud, *, valence_cap):
    dec = YuelBondBondDecoder(
        hidden_dim=16,
        n_layers=1,
        valence_cap=valence_cap,
        connectivity_prior=False,  # isolate the selector from the bridge prior
    )
    yres = _metal_only_logits(cloud)
    dec._decoder.featurise = lambda _cloud: yres  # type: ignore[assignment]
    return dec.decode(cloud), yres


def _bonds_at_metal(bond_orders, metal_idx: int) -> tuple[int, int]:
    """(count counting one end, count counting both ends)."""
    one = sum(1 for (a, _, _) in bond_orders if a == metal_idx)
    both = sum(1 for (a, b, _) in bond_orders if a == metal_idx or b == metal_idx)
    return one, both


@pytest.mark.parametrize("metal_index", [0, 5])
def test_the_gate_counts_both_ends_of_every_metal_bond(metal_index):
    """The metal's bonds are counted the same whether it is index 0 or index 5.

    With the old single-endpoint count, `metal_index=5` reported 0 bonds for
    the whole decode, so the square-planar cap of 4 never fired.
    """
    cloud = _five_coordinate_cloud(metal_index)
    out, _ = _decode(cloud, valence_cap={PT: 6})
    one, both = _bonds_at_metal(out.decoded.bond_orders, metal_index)
    assert both == 4, (
        f"expected the geometry gate to stop the metal at 4 bonds, got {both} "
        f"(bond_orders={out.decoded.bond_orders})"
    )
    if metal_index == 5:
        assert one == 0, (
            "this cloud no longer reproduces the defect: the metal is index 5 "
            "and the one-end count should be blind to it"
        )
        assert out.bonds_rejected_overvalent >= 1, (
            "the 5th Pt bond was not refused — the gate did not count it"
        )


def test_the_fifth_pt_bond_is_refused_when_the_cap_is_raised():
    """Pt_IV: valence_cap={78: 6} lets the valence check pass 5 bonds.

    Square planar allows 4, so the geometry gate is the only thing that can
    refuse the fifth.  If it counts one end only, all five are kept.
    """
    cloud = _five_coordinate_cloud(metal_index=5)
    out, _ = _decode(cloud, valence_cap={PT: 6})
    pt_bonds = [
        b for b in out.decoded.bond_orders if 5 in (b[0], b[1])
    ]
    assert len(pt_bonds) == 4, (
        f"expected 4 Pt bonds under square planar, got {len(pt_bonds)}: "
        f"{pt_bonds}"
    )
    assert out.bonds_rejected_overvalent >= 1


def test_metal_at_index_zero_is_unchanged():
    """The default construction is bit-identical: for metal-at-0 clouds the
    one-end and two-end counts are the same number, so the gate cannot behave
    differently there.
    """
    cloud = _five_coordinate_cloud(metal_index=0)
    out, _ = _decode(cloud, valence_cap=None)
    one, both = _bonds_at_metal(out.decoded.bond_orders, 0)
    assert one == both, (
        "on a metal-at-index-0 cloud the two counts must agree; if they do "
        "not, the fix changed behaviour on the 84.5% majority case"
    )
    assert len(out.decoded.bond_orders) == 4, (
        "the default cap (square planar, 4) must still stop the metal at 4"
    )
