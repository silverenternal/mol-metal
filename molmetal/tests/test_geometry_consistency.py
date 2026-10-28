# SPDX-License-Identifier: Apache-2.0
"""The geometry term must be zero when the geometry is fine, and trainable.

The failure this guards against is not a wrong constant — it is a term that
looks wired and carries no gradient.  ``clash_penalty_weight`` in the trainer
computes on ground-truth positions and detaches the result, so enabling it
changes the reported loss and nothing else.  These tests assert the two
properties that separate a loss from a monitor: it fires on bad geometry, and
``backward()`` reaches the coordinates.
"""

from __future__ import annotations

import pytest
import torch

from molmetal.loss.geometry_consistency import (
    DEFAULT_ML_BAND_A,
    METAL_ML_BAND_A,
    geometry_consistency,
)

PT = 78
C = 6
N = 7


def _one_cloud(x, z, bonds=None, radius=None):
    n = int(x.shape[0])
    z = torch.tensor(z, dtype=torch.long) if not torch.is_tensor(z) else z
    radius = (
        torch.full((n,), 0.76)
        if radius is None
        else torch.as_tensor(radius, dtype=torch.float32)
    )
    if bonds is None:
        bonds = torch.zeros((2, 0), dtype=torch.long)
    return geometry_consistency(
        x,
        z,
        torch.zeros(n, dtype=torch.long),
        bonds,
        radius,
        1,
    )


def test_a_clash_fires_and_a_sane_gap_does_not():
    far = torch.tensor([[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]])
    near = torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
    z = [C, C]

    assert float(_one_cloud(far, z)["clash"]) == 0.0
    assert float(_one_cloud(near, z)["clash"]) > 0.0


def test_a_stretched_bond_fires_and_a_normal_one_does_not():
    bonds = torch.tensor([[0], [1]], dtype=torch.long)
    good = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    long = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    z = [C, C]

    assert float(_one_cloud(good, z, bonds)["bond"]) == 0.0
    assert float(_one_cloud(long, z, bonds)["bond"]) > 0.0


def test_the_metal_band_is_the_measured_one():
    """A Pt-L distance inside the DFT p10/p90 band must score zero.

    Wf-A113 (T56) — the metal term scores BONDED M-L pairs only.  Before that
    change it scored every metal/non-metal pair in the cloud, which on real
    geometry meant 2,893 pairs of which 5.98% could satisfy the band; the
    median M-X distance was 3.25 A.  A non-bonded ligand atom on the far side
    of the complex is now nobody's target, so this test carries the bond.
    """
    lo, hi = METAL_ML_BAND_A[PT]
    z = [PT, N]
    bond = torch.tensor([[0], [1]], dtype=torch.long)
    inside = torch.tensor([[0.0, 0.0, 0.0], [(lo + hi) / 2.0, 0.0, 0.0]])
    too_short = torch.tensor([[0.0, 0.0, 0.0], [1.234, 0.0, 0.0]])

    assert float(_one_cloud(inside, z, bond)["metal"]) == 0.0
    assert float(_one_cloud(too_short, z, bond)["metal"]) > 0.0


def test_a_non_bonded_ligand_atom_is_not_scored_by_the_metal_term():
    """The defect T56 removed: 3.25 A median M-X is a ligand, not a bond."""
    z = [PT, N]
    far = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    out = _one_cloud(far, z, torch.zeros((2, 0), dtype=torch.long))
    assert float(out["n_metal_pairs"]) == 0.0
    assert float(out["metal"]) == 0.0


def test_an_unknown_metal_falls_back_to_the_measured_union():
    lo, hi = DEFAULT_ML_BAND_A
    assert lo == min(b[0] for b in METAL_ML_BAND_A.values())
    assert hi == max(b[1] for b in METAL_ML_BAND_A.values())


def test_the_term_is_differentiable():
    """The property `clash_penalty_weight` does not have."""
    x = torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]], requires_grad=True)
    out = _one_cloud(x, [C, C])
    out["clash"].backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0.0


def test_clouds_are_scored_independently():
    """Two clouds 0.8 A apart must not clash with EACH OTHER."""
    x = torch.tensor([[0.0, 0.0, 0.0], [3.5, 0.0, 0.0], [0.0, 0.8, 0.0], [3.5, 0.8, 0.0]])
    z = torch.tensor([C, C, C, C], dtype=torch.long)
    batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    out = geometry_consistency(
        x, z, batch_idx, torch.zeros((2, 0), dtype=torch.long), torch.full((4,), 0.76), 2
    )
    assert float(out["clash"]) == 0.0
    assert float(out["n_clash_pairs"]) == 0.0


def test_a_bond_beyond_the_first_cloud_does_not_index_out_of_range():
    """A bond in a LATER cloud must be scored in that cloud.

    MEASURED on the first smoke (2026-10-25): the local index map was sized by
    the cloud's own largest index, so a bond belonging to a LATER cloud raised
    ``IndexError`` and the whole term silently fell back to zero.

    Wf-A113 (T56) — ``Batch.bonds`` carries per-cloud LOCAL indices, so the
    bond of cloud 1 is ``(0, 1)`` in ITS OWN frame, and ``pair_cloud_idx``
    says which cloud owns it.  The old version of this test passed global
    indices, which is exactly the frame confusion that made the real term
    score 10.8% of the bonds.
    """
    x = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0],
            [10.0, 0.0, 0.0],
            [11.5, 0.0, 0.0],
            [10.0, 1.5, 0.0],
        ]
    )
    z = torch.tensor([C, C, C, C, C, C], dtype=torch.long)
    batch_idx = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    bonds = torch.tensor([[0, 0], [1, 1]], dtype=torch.long)
    pair_cloud_idx = torch.tensor([0, 1], dtype=torch.long)
    out = geometry_consistency(
        x,
        z,
        batch_idx,
        bonds,
        torch.full((6,), 0.76),
        2,
        pair_cloud_idx=pair_cloud_idx,
    )
    assert float(out["n_bond_pairs"]) == 2.0
    assert float(out["bond"]) == 0.0


def test_a_multi_cloud_batch_without_pair_cloud_idx_raises():
    """The frame is not inferable; a silent wrong answer is worse than a stop."""
    x = torch.zeros((4, 3))
    with pytest.raises(ValueError, match="pair_cloud_idx"):
        geometry_consistency(
            x,
            torch.tensor([C, C, C, C], dtype=torch.long),
            torch.tensor([0, 0, 1, 1], dtype=torch.long),
            torch.tensor([[0], [1]], dtype=torch.long),
            torch.full((4,), 0.76),
            2,
        )


def test_an_empty_batch_returns_exact_zeros():
    x = torch.zeros((0, 3))
    out = geometry_consistency(
        x,
        torch.zeros((0,), dtype=torch.long),
        torch.zeros((0,), dtype=torch.long),
        None,
        torch.zeros((0,)),
        0,
    )
    for k in ("clash", "bond", "metal"):
        assert float(out[k]) == 0.0
