# SPDX-License-Identifier: Apache-2.0
"""T51 → T72 — the metal bond ceiling must not depend on the metal's index.

HISTORY
-------
``_select_bonds`` used to carry a SECOND ceiling, from the coordination
geometry's ``max_bonds``, computed as

    bonds_at_metal = sum(1 for (a, _, _) in bond_orders if a == metal_idx)

Candidate pairs come from ``torch.triu_indices(n, n, offset=1)``, so every
stored pair has ``a < b``; a metal at a HIGHER index was never counted and the
gate never fired for it.  MEASURED over 9,111 tmQM metal complexes: 7,700
(84.5%) put the metal at heavy index 0, and 11.84% (5,729 of 48,371) of metal
bonds have the metal at the higher index.

That gate is now GONE (Wf-A113 T72): it was redundant with ``_cap_for_z`` while
the two tables agreed, and it became actively wrong once the valence caps were
derived from the corpus, because ``_OCTAHEDRAL.max_bonds`` is 6 and would
delete the sixth through ninth bonds of every eta6-arene complex.

What remains is one ceiling, ``_COVALENT_VALENCE``, applied through
``remaining[metal_idx]``.  This file pins the property the old gate broke:
the number of bonds a metal ends up with is the same whether the metal sits at
index 0 or index 5, and a ``valence_cap`` override is honoured rather than
silently overridden by a second table.
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


def _bonds_at_metal(bond_orders, metal_idx: int) -> int:
    return sum(1 for (a, b, _) in bond_orders if a == metal_idx or b == metal_idx)


@pytest.mark.parametrize("metal_index", [0, 5])
def test_default_pt_cap_stops_at_four_regardless_of_index(metal_index):
    """Pt keeps 4 bonds whether it is atom 0 or atom 5."""
    cloud = _five_coordinate_cloud(metal_index)
    out, _ = _decode(cloud, valence_cap=None)
    assert _bonds_at_metal(out.decoded.bond_orders, metal_index) == 4
    assert out.bonds_rejected_overvalent >= 1


@pytest.mark.parametrize("metal_index", [0, 5])
def test_valence_cap_override_is_honoured(metal_index):
    """``valence_cap={78: 6}`` must yield 6, not 4.

    This is the Pt_IV case the override exists for.  Before T72 a second
    (geometry) ceiling capped Pt at 4 anyway, so the documented override did
    nothing — the same "knob with no effect" shape as T70.
    """
    cloud = _five_coordinate_cloud(metal_index)
    out, _ = _decode(cloud, valence_cap={PT: 6})
    assert _bonds_at_metal(out.decoded.bond_orders, metal_index) == 5, (
        "a raised cap must admit all five Pt-C bonds; got "
        f"{out.decoded.bond_orders}"
    )
    assert out.bonds_rejected_overvalent == 0


def test_metal_at_index_zero_is_unchanged():
    """The default construction still stops at 4 (no regression for the
    84.5% majority case where the metal is heavy index 0)."""
    cloud = _five_coordinate_cloud(metal_index=0)
    out, _ = _decode(cloud, valence_cap=None)
    assert len(out.decoded.bond_orders) == 4
