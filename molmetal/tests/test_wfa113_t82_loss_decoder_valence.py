# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 T82 — the soft-valence LOSS must use the decoder's valence model.

WHY THIS FILE EXISTS
--------------------
``gen_validity_weight`` adds ``valence_log_barrier`` to the training objective:
a squared hinge on per-atom valence, meant to stop the head predicting bonds
the decoder will refuse.  It was fed a different table and a different
accounting from the decoder:

    loss (before)   ``DEFAULT_VALENCES`` + ``TMQM_MEASURED_METAL_CAPS``
                    ``DEFAULT_VALENCES`` is ``PRIMITIVE_ATOMS.arity``, i.e.
                    the COORDINATION NUMBER (valence + lone pairs), not the
                    valence: O 4, F/Cl/Br/I 4, N 4, P 4, S 4, Se 2.
                    aromatic bond charged 1.5 at every atom
                    metal charged its class order

    decoder         ``_COVALENT_VALENCE`` (RDKit's valence model) with the
                    measured coordination caps for the metals
                    aromatic 1.0 at a heteroatom (T78) and 1.0 for a carbon's
                    third and later (T81)
                    metal charged 1.0 per bond (T77), ligand charged 0 (T79)

Two consequences, in both directions:

* the loss PENALISED atoms the decoder accepts — every sulfone (6.0 against a
  cap of 4), every fusion carbon (4.5 against 4), every thiophene S — so the
  gradient pointed away from the corpus truth;
* the loss PERMITTED atoms the decoder refuses — it gave B a cap of 4 when
  RDKit's limit is 3, and it could never fire on O or the halogens at all.

MEASURED on the six-metal cohort (33,429 complexes) —
``metrics/by_metric/wfa113_t82_loss_decoder_valence_split.json``:

    complexes where the two disagree          9,002 / 33,429 = 26.93%
        C  1.13% of atoms   (fusion carbons, 4.5 vs 4)
        P  9.38%            (4.5 vs 4)
        S 18.73%            (sulfone 6.0 vs 4)
        Se 36.76%
        B 10.13% in the other direction (loss allows the 4th bond, decoder
                 refuses it)

Fixing the CAPS alone is not enough: it makes the disagreement worse (77.11%
of complexes) because the flat 1.5 aromatic charge then exceeds the tighter
ligand caps — N 61.59%, O 30.66%.  Caps and accounting have to move together.
With both, the two agree on every ligand atom: 0 mismatched of 1.24M.

What this file pins
-------------------
1. The barrier is silent on the motifs the decoder accepts (dative, fusion
   carbon, sulfone).
2. It fires on the motifs the decoder refuses (4 aromatic bonds at carbon,
   quaternary N, Pt's 5th bond).
3. It uses the valence table, not the coordination-number table: an O with 3
   single bonds is over-valent, which the old table could not see.
4. A metal is charged one per bond against its coordination cap, so ten
   aromatic-class bonds at Re (cap 9) are penalised and nine are not.
5. On random graphs the barrier is positive exactly when the decoder's own
   per-endpoint accounting exceeds a cap.
"""

from __future__ import annotations

import random
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
)
from molmetal.molmetal_lam.lam_chem.decoder_rework import (  # noqa: E402
    TMQM_MEASURED_METAL_COORDINATION_CAPS as METAL_CAPS,
    valence_log_barrier,
)
from molmetal.models.bond_head import (  # noqa: E402
    BOND_AROMATIC,
    BOND_DOUBLE,
    BOND_SINGLE,
)

_ORDER = torch.tensor([0.0, 1.0, 2.0, 3.0, 1.5])


def _is_metal(z: int) -> bool:
    return int(z) in METAL_CAPS


def _barrier(zs, pairs, classes):
    z = torch.tensor(zs, dtype=torch.long)
    src = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    dst = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    onehot = torch.zeros((len(pairs), 5))
    for k, c in enumerate(classes):
        onehot[k, c] = 1.0
    orders = (onehot * _ORDER).sum(dim=-1)
    return float(
        valence_log_barrier(
            orders, z, src, dst, len(zs), aromatic_mass=onehot[:, 4]
        )
    )


def _decoder_excess(zs, pairs, classes):
    """The decoder's per-atom over-cap excess, as a list (one per atom)."""
    caps = {**_COVALENT_VALENCE, **METAL_CAPS}
    cost = [0.0] * len(zs)
    arom = [0] * len(zs)
    for (i, j), c in zip(pairs, classes):
        cost[i] += _endpoint_valence_cost(c, zs[i], zs[j], _is_metal, arom[i])
        cost[j] += _endpoint_valence_cost(c, zs[j], zs[i], _is_metal, arom[j])
        if c == BOND_AROMATIC:
            arom[i] += 1
            arom[j] += 1
    return [
        max(0.0, cost[a] - float(caps.get(zs[a], 4))) for a in range(len(zs))
    ]


#: The hinge is ``max(0, used - cap + eps)^2`` with eps = 1e-3, so an atom at
#: exactly its cap contributes ``eps**2 = 1e-6``.  Anything at or below this
#: floor means "the decoder accepts it too".
_FLOOR = 1e-5


# ---------------------------------------------------------------------------
# 1. Silent on what the decoder accepts
# ---------------------------------------------------------------------------
def test_barrier_is_silent_on_a_dative_bond():
    """Ether O with 2 C and 1 Ru: the decoder keeps all three, so must the loss."""
    assert _barrier([44, 8, 6, 6], [(0, 1), (1, 2), (1, 3)], [1, 1, 1]) < _FLOOR


def test_barrier_is_silent_on_a_fusion_carbon():
    assert _barrier([6, 6, 6, 6], [(0, 1), (0, 2), (0, 3)], [4, 4, 4]) < _FLOOR


def test_barrier_is_silent_on_a_sulfone():
    assert (
        _barrier(
            [16, 8, 8, 6, 6],
            [(0, 1), (0, 2), (0, 3), (0, 4)],
            [BOND_DOUBLE, BOND_DOUBLE, BOND_SINGLE, BOND_SINGLE],
        )
        < _FLOOR
    )


# ---------------------------------------------------------------------------
# 2. Fires where the decoder refuses
# ---------------------------------------------------------------------------
def test_barrier_fires_on_a_fourth_aromatic_bond_at_carbon():
    assert _barrier([6, 6, 6, 6, 6], [(0, k) for k in range(1, 5)], [4] * 4) > 0


def test_barrier_fires_on_a_quaternary_neutral_nitrogen():
    assert _barrier([7, 6, 6, 6, 6], [(0, k) for k in range(1, 5)], [1] * 4) > 0


def test_barrier_fires_on_platinum_fifth_bond():
    assert _barrier([78] + [7] * 5, [(0, k) for k in range(1, 6)], [1] * 5) > 0


# ---------------------------------------------------------------------------
# 3. The table is the VALENCE, not the coordination number
# ---------------------------------------------------------------------------
def test_oxygen_with_three_bonds_is_over_valent():
    """The old table gave O a cap of 4 (2 valence + 2 lone pairs), so this
    atom — which RDKit rejects — cost the loss nothing."""
    assert _barrier([8, 6, 6, 6], [(0, 1), (0, 2), (0, 3)], [1, 1, 1]) > 0


def test_halogen_with_two_bonds_is_over_valent():
    assert _barrier([17, 6, 6], [(0, 1), (0, 2)], [1, 1]) > 0


# ---------------------------------------------------------------------------
# 4. A metal is charged one per bond against its coordination cap
# ---------------------------------------------------------------------------
def test_metal_charged_per_bond_against_the_coordination_cap():
    # Re's coordination cap is 9.  Nine aromatic-class bonds fit; ten do not.
    nine = _barrier([75] + [7] * 9, [(0, k) for k in range(1, 10)], [4] * 9)
    ten = _barrier([75] + [7] * 10, [(0, k) for k in range(1, 11)], [4] * 10)
    assert nine < _FLOOR
    assert ten > 0.0


# ---------------------------------------------------------------------------
# 5. Agreement with the decoder's own accounting, on random graphs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_barrier_agrees_with_the_decoder_on_random_graphs(seed):
    rng = random.Random(seed)
    zpool = [6, 6, 6, 6, 7, 8, 16, 5, 9, 44, 78]
    for _ in range(40):
        n = rng.randint(2, 8)
        zs = [rng.choice(zpool) for _ in range(n)]
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.6:
                    pairs.append((i, j))
        if not pairs:
            continue
        classes = [rng.choice([1, 1, 1, 2, 3, 4]) for _ in pairs]
        pen = _barrier(zs, pairs, classes)
        per_atom = _decoder_excess(zs, pairs, classes)
        excess = sum(per_atom)
        assert (pen > _FLOOR) == (excess > 0.0), (
            f"barrier and decoder disagree on z={zs} pairs={pairs} "
            f"classes={classes}: penalty={pen}, decoder excess={excess}"
        )
        if excess == 0.0:
            assert pen < _FLOOR
        else:
            # The hinge squares each atom's (excess + eps) and then sums.
            assert pen == pytest.approx(
                sum((e + 1e-3) ** 2 for e in per_atom), abs=1e-3
            )
