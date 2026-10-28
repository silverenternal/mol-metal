# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 T68 — the Wiberg BO target was looked up in the wrong atom frame.

THE DEFECT
----------
``_build_cloud_from_kept`` drops hydrogens (``include_hydrogens=False`` is the
default) and remaps every bond pair through ``old_to_new``.  The
``bond_orders_continuous`` lookup, however, was keyed by the RAW tmQM indices
and then queried with the REMAPPED ones.  tmQM's atom order interleaves
hydrogens — MEASURED, the first H sits at index 2 in one of the first
complexes — so nearly every heavy atom shifts down and the lookup missed.
The miss was silent (``.get(..., 0.0)``):

  * ``bond_orders_continuous`` was 0.0 on 55.2% of real bonds and carried
    ANOTHER bond's order on the rest;
  * ``is_dative_pair`` is derived from ``v < 1.0``, so it fired on every metal
    bond regardless of the bond's real order.

Both are supervision: the BO head regresses ``log(bo + 1)`` (or the linear
aux MSE at ``flow.bond_aux_mse_weight > 0``, which several configs enable at
0.1), and the decoder uses ``is_dative_pair`` to mark coordination bonds.

These tests pin the frame alignment on a synthetic 3-atom block, so they do not
need the 33k-complex corpus.
"""

from __future__ import annotations

import torch

from molmetal.data.tmqm_clouds import _BoInfo, _build_cloud_from_kept


def _cloud():
    """C-H-N with the H at index 1; heavy atoms keep indices 0 and 2."""
    symbols = ["C", "H", "N"]
    keep_idx = [0, 2]
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.30, 0.0, 0.0]])
    zt = torch.tensor([6, 7], dtype=torch.long)
    bo = _BoInfo(
        pairs_with_class=[(0, 1, 1), (1, 2, 1), (0, 2, 1)],
        pairs_continuous=[(0, 1, 0.95), (1, 2, 0.85), (0, 2, 1.30)],
        per_atom_nbr_zsets={0: [1, 7], 1: [6, 7], 2: [1, 6]},
    )
    cloud = _build_cloud_from_kept(
        code="TEST",
        keep_idx=keep_idx,
        pos=pos,
        zt=zt,
        symbols=symbols,
        row={"metal": ""},
        bo_info=bo,
        natural_charges_map=None,
    )
    assert cloud is not None
    return cloud


def test_only_the_heavy_pair_survives_the_hydrogen_drop():
    c = _cloud()
    assert c.bonds.shape == (2, 1)
    assert sorted(c.bonds.tolist()) == [[0], [1]]


def test_continuous_bo_comes_from_the_same_pair_not_from_the_raw_index():
    """C-N must read 1.30, not the C-H 0.95 that occupies raw index (0, 1)."""
    c = _cloud()
    assert c.bond_orders_continuous is not None
    assert abs(float(c.bond_orders_continuous[0]) - 1.30) < 1e-5, (
        "the lookup used the raw tmQM indices instead of the kept-atom frame: "
        f"got {float(c.bond_orders_continuous[0])}, expected 1.30 (C-N), "
        "which is what the pre-T68 code returned as 0.95 (C-H)"
    )


def test_no_real_bond_carries_a_zero_bo():
    c = _cloud()
    assert float(c.bond_orders_continuous.min()) > 0.0


def test_dative_uses_the_real_order():
    """A metal bond with BO 1.6 is covalent, not dative."""
    symbols = ["Pt", "H", "N"]
    cloud = _build_cloud_from_kept(
        code="TEST2",
        keep_idx=[0, 2],
        pos=torch.tensor([[0.0, 0.0, 0.0], [2.05, 0.0, 0.0]]),
        zt=torch.tensor([78, 7], dtype=torch.long),
        symbols=symbols,
        row={"metal": "Pt"},
        bo_info=_BoInfo(
            pairs_with_class=[(0, 1, 1), (0, 2, 1)],
            pairs_continuous=[(0, 1, 0.55), (0, 2, 1.60)],
            per_atom_nbr_zsets={0: [7, 1], 2: [78]},
        ),
        natural_charges_map=None,
    )
    assert cloud is not None
    assert cloud.is_dative_pair is not None
    assert bool(cloud.is_dative_pair[0]) is False, (
        "Pt-N with a Wiberg BO of 1.60 was flagged dative; the pre-T68 code "
        "read the raw-index entry for the dropped Pt-H pair (0.55) and "
        "returned True"
    )
