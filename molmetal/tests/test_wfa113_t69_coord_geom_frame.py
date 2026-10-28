# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 T69 — ``coord_geom_class`` was classified from another atom's degree.

``per_atom_nbr_zsets`` is keyed by the RAW tmQM index, but the builder read it
with ``metal_idx_in_kept`` — an index in the kept (hydrogen-dropped) frame.
Whenever a hydrogen precedes the metal in tmQM's atom order the two differ, and
the class was derived from a hydrogen's or a carbon's degree.

MEASURED on a 48-cloud cohort: the two frames disagree on 7 of 48 clouds
(14.6%), including six where the "metal" was given CN=1 (a hydrogen's only
neighbour).  The field has no consumer anywhere in the repo today, so no
trained number changes; these tests exist because the field is documented as a
MOD5 conditioning slot.
"""

from __future__ import annotations

import torch

from molmetal.data.tmqm_clouds import _BoInfo, _build_cloud_from_kept


def _cloud(symbols, keep_idx, zt, nbrs, metal):
    return _build_cloud_from_kept(
        code="TEST",
        keep_idx=keep_idx,
        pos=torch.zeros(len(keep_idx), 3),
        zt=torch.tensor(zt, dtype=torch.long),
        symbols=symbols,
        row={"metal": metal},
        bo_info=_BoInfo(
            pairs_with_class=[],
            pairs_continuous=[],
            per_atom_nbr_zsets=nbrs,
        ),
        natural_charges_map=None,
    )


def test_metal_after_a_hydrogen_uses_the_metals_own_neighbour_list():
    """H at raw 0, Pt at raw 1 -> kept index 0, raw index 1."""
    cloud = _cloud(
        symbols=["H", "Pt", "N", "N", "N", "N"],
        keep_idx=[1, 2, 3, 4, 5],
        zt=[78, 7, 7, 7, 7],
        # Raw-indexed neighbour sets: the hydrogen has one neighbour (the Pt),
        # the Pt has four (the four N).
        nbrs={0: [78], 1: [7, 7, 7, 7], 2: [78], 3: [78], 4: [78], 5: [78]},
        metal="Pt",
    )
    assert cloud is not None
    assert cloud.coord_geom_class == "square_planar", (
        "coord_geom_class was classified from the raw index equal to the "
        f"kept index (the hydrogen, CN=1) instead of the metal: got "
        f"{cloud.coord_geom_class!r}"
    )


def test_metal_at_index_zero_is_unaffected():
    """The common case: no hydrogen precedes the metal, so both frames agree."""
    cloud = _cloud(
        symbols=["Pt", "H", "N", "N", "N", "N"],
        keep_idx=[0, 2, 3, 4, 5],
        zt=[78, 7, 7, 7, 7],
        nbrs={0: [7, 7, 7, 7], 1: [78], 2: [78], 3: [78], 4: [78], 5: [78]},
        metal="Pt",
    )
    assert cloud is not None
    assert cloud.coord_geom_class == "square_planar"


def test_unmapped_coordination_number_stays_none():
    """CN=7 on Pt has no mapped geometry; None is the documented result."""
    cloud = _cloud(
        symbols=["H", "Pt"] + ["N"] * 7,
        keep_idx=list(range(1, 9)),
        zt=[78] + [7] * 7,
        nbrs={0: [78], 1: [7] * 7, **{k: [78] for k in range(2, 9)}},
        metal="Pt",
    )
    assert cloud is not None
    assert cloud.coord_geom_class is None
