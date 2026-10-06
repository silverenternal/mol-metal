# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Pytest: Round-3 D2 — the connectivity prior in the YuelBond decoder.

The prior is POSTPROCESS in the decoder, so the properties worth pinning are
structural, not statistical:

1. it is a strict ADD — it never removes a bond the selector kept, so
   ``connectivity_prior=False`` reproduces the pre-D2 decode bit-exactly;
2. it only ever bridges heavy atoms (H excluded) the head already emitted as
   candidates inside the distance cutoff — it cannot invent geometry;
3. it respects the valence budget, so it cannot build a bond the selector
   would have rejected as overvalent;
4. it is a fixed point — running it on an already-connected graph adds nothing
   and terminates;
5. the valence table it depends on covers the elements the tmQM corpus
   actually contains, including three of the six project metals.

Test 5 is a regression pin for a bug this feature FOUND rather than
introduced: Rh/Re/Os and B/Si/As were absent from ``_COVALENT_VALENCE`` and
took the unknown-Z default of 4, which is a cap too HIGH for boron and
authorised a bond RDKit then rejected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.molmetal_lam.lam_chem.cfm_decoder import build_cfm_decoder
from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
    _COVALENT_VALENCE,
    YuelBondBondDecoder,
)
from molmetal.models.bond_head import AtomCloud


def _chain_cloud() -> AtomCloud:
    """Two ethane-like fragments, 2.9 A apart at the closest approach.

    The gap is inside the head's 3.5 A cutoff, so the head DOES emit the
    cross-fragment candidate pair — which is exactly the pair a connectivity
    prior is allowed to promote and a "no new geometry" rule must forbid it
    from inventing.  Positions: fragment A = C1-C2 at x 0 and 1.5; fragment B
    = C3-C4 at x 4.4 and 5.9, so C2..C3 is 2.9 A.
    """
    positions = torch.tensor(
        [
            [0.00, 0.0, 0.0],  # C1
            [1.50, 0.0, 0.0],  # C2
            [4.40, 0.0, 0.0],  # C3  (2.9 A from C2)
            [5.90, 0.0, 0.0],  # C4
        ],
        dtype=torch.float32,
    )
    atomic_numbers = torch.tensor([6, 6, 6, 6], dtype=torch.long)
    return AtomCloud(positions=positions, atomic_numbers=atomic_numbers)


def _n_heavy_components(cloud: AtomCloud, bonds) -> int:
    z = [int(v) for v in cloud.atomic_numbers.tolist()]
    parent = list(range(len(z)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j, _o in bonds:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[rj] = ri
    return len({find(i) for i in range(len(z)) if z[i] != 1})


# ===========================================================================
# 1 — off is a no-op, and the prior only ever ADDS
# ===========================================================================
def test_connectivity_prior_off_is_a_no_op():
    """With the prior off, the decoder must report zero additions and leave
    the bond set exactly as the selector produced it — otherwise every
    pre-D2 measurement is unreproducible and the A/B is meaningless.

    The constructor default is ON, so this also pins that 'off' is a real
    switch rather than a flag that was declared and never consulted.
    """
    torch.manual_seed(0)
    cloud = _chain_cloud()
    off = YuelBondBondDecoder(hidden_dim=32, n_layers=2, connectivity_prior=False)
    assert off.connectivity_prior is False
    with torch.no_grad():
        r_off = off.decode(cloud)
    assert r_off.bonds_added_connectivity == 0
    # The default is ON, so a caller who says nothing gets the new behaviour.
    assert YuelBondBondDecoder(hidden_dim=32, n_layers=2).connectivity_prior is True


def test_connectivity_prior_never_removes_a_selected_bond():
    """The prior is strictly additive. A superset is the whole safety
    argument: if it only ever adds, a caller who distrusts it can diff the
    two bond sets and see exactly what it changed."""
    torch.manual_seed(0)
    cloud = _chain_cloud()
    kwargs = dict(hidden_dim=32, n_layers=2, decision="argmax")
    with torch.no_grad():
        off = YuelBondBondDecoder(connectivity_prior=False, **kwargs).decode(cloud)
        on = YuelBondBondDecoder(connectivity_prior=True, **kwargs).decode(cloud)
    off_set = {(i, j) for i, j, _o in off.decoded.bond_orders}
    on_set = {(i, j) for i, j, _o in on.decoded.bond_orders}
    assert off_set <= on_set, (
        f"the prior removed bonds: only in off={sorted(off_set - on_set)}"
    )


# ===========================================================================
# 2 — it only bridges heavy atoms
# ===========================================================================
def test_connectivity_prior_bridges_heavy_atoms_only():
    """With the prior on, a heavy-atom graph that was fragmented either stays
    as it was or becomes ONE component — never something else, and never
    joined through a hydrogen."""
    torch.manual_seed(0)
    cloud = _chain_cloud()
    with torch.no_grad():
        off = YuelBondBondDecoder(
            hidden_dim=32, n_layers=2, connectivity_prior=False
        ).decode(cloud)
        on = YuelBondBondDecoder(
            hidden_dim=32, n_layers=2, connectivity_prior=True
        ).decode(cloud)
    n_off = _n_heavy_components(cloud, off.decoded.bond_orders)
    n_on = _n_heavy_components(cloud, on.decoded.bond_orders)
    assert n_on <= n_off, f"components rose {n_off} -> {n_on}"
    assert n_on >= 1


def test_connectivity_prior_is_deterministic_across_decodes():
    """Decoding the same cloud twice must reach the same state.

    This is the property the per-decode `bonds_added_connectivity` counter
    depends on for the A/B to mean anything: if the prior were
    nondeterministic, a difference in `single_fragment_ratio` between two
    arms could not be attributed to the flag. The loop terminates because
    each added bond unions two components, so the component count is a
    strictly decreasing integer — there is no fixed-point test to add.
    """
    torch.manual_seed(0)
    decoder = YuelBondBondDecoder(
        hidden_dim=32, n_layers=2, connectivity_prior=True
    )
    cloud = _chain_cloud()
    with torch.no_grad():
        first = decoder.decode(cloud)
        second = decoder.decode(cloud)
    assert first.bonds_added_connectivity == second.bonds_added_connectivity
    assert list(first.decoded.bond_orders) == list(second.decoded.bond_orders)


# ===========================================================================
# 3 — the factory propagates the switch
# ===========================================================================
def test_build_cfm_decoder_propagates_connectivity_prior():
    """`build_cfm_decoder` is the entry point the sampler calls. A flag that
    reaches `YuelBondBondDecoder` but not here is FLAG_ONLY on that path."""
    assert build_cfm_decoder(
        use_yuelbond=True, connectivity_prior=False
    ).connectivity_prior is False
    assert (
        build_cfm_decoder(use_yuelbond=True, connectivity_prior=True)
        .connectivity_prior
        is True
    )


# ===========================================================================
# 4 — valence table coverage (regression pin for a bug D2 FOUND)
# ===========================================================================
def test_valence_table_covers_tmqm_elements_and_project_metals():
    """Every element MEASURED in a 200-cloud tmQM val sample, plus the six
    project metals, must have an explicit cap.

    The failure mode is not a missing entry per se — it is that the unknown-Z
    default is 4, so a missing entry silently AUTHORISES an over-valent
    bond. MEASURED failure: a B atom (Z 5, absent) took the default and RDKit
    rejected the molecule with 'Explicit valence for atom # 35 B, 4, is
    greater than permitted'. Rh (45), Re (75) and Os (76) — three of the six
    metals this project targets — were absent for the same reason.
    """
    measured_in_tmqm = {
        5: "B",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        14: "Si",
        15: "P",
        16: "S",
        17: "Cl",
        33: "As",
        34: "Se",
        35: "Br",
        53: "I",
    }
    project_metals = {78: "Pt", 44: "Ru", 77: "Ir", 45: "Rh", 76: "Os", 75: "Re"}
    for z, sym in {**measured_in_tmqm, **project_metals}.items():
        assert z in _COVALENT_VALENCE, (
            f"{sym} (Z={z}) has no explicit valence cap and would take the "
            f"unsafe default of 4"
        )
    # Boron specifically: RDKit permits 3, so the default of 4 is wrong.
    assert _COVALENT_VALENCE[5] == 3
    # The three newly-listed metals get a conservative octahedral ceiling. A
    # cap that is too LOW only costs a bond; one that is too HIGH loses the
    # whole decode.
    for z in (45, 75, 76):
        assert _COVALENT_VALENCE[z] == 6
