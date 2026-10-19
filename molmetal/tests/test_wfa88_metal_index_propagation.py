# SPDX-License-Identifier: Apache-2.0
"""Generated clouds must carry the template's metal index.

WHY THIS EXISTS
---------------
``generate_clouds_from_noise`` copies ``atomic_numbers`` from the template
and re-draws only the coordinates, so the metal sits at the same index in the
sample as in the template.  It used to drop ``metal_atom_index`` anyway, and
``BondAwareDecoder._featurise`` reads that field to fill ``angle_to_metal``
and ``d_to_metal``.  Every sampled decode therefore ran with 2 of the 11
pair-feature columns pinned at 0.0, while the same decoder saw their real
values on every training cloud: a silent train/inference feature mismatch that
made the sampled decode a strictly easier problem than the one the head was
fitted on.  Nothing in the log said so.

RED CHECK: the end-to-end case drives the real sampler with a zero-velocity
model and asserts the returned cloud carries the index.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _cloud(z_list, metal_index=None):
    from molmetal.models.bond_head import AtomCloud

    return AtomCloud(
        positions=torch.zeros(len(z_list), 3),
        atomic_numbers=torch.tensor(z_list, dtype=torch.long),
        metal_atom_index=metal_index,
    )


def test_carries_explicit_template_index():
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _carry_metal_index,
    )

    tpl = _cloud([6, 6, 7, 78, 1], metal_index=torch.tensor(3))
    assert int(_carry_metal_index(tpl).item()) == 3


def test_infers_index_from_atomic_numbers_when_field_absent():
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _carry_metal_index,
    )

    # No metal_atom_index field at all: the loader never set it.
    tpl = _cloud([6, 6, 44, 7, 1])
    assert int(_carry_metal_index(tpl).item()) == 2


def test_respects_the_no_metal_sentinel():
    from molmetal.models.bond_head import AtomCloud
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _carry_metal_index,
    )

    tpl = _cloud(
        [6, 6, 78], metal_index=torch.tensor(AtomCloud.NO_METAL_INDEX)
    )
    assert int(_carry_metal_index(tpl).item()) == AtomCloud.NO_METAL_INDEX


def test_out_of_range_index_is_not_trusted():
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _carry_metal_index,
    )

    # 9 is not a valid index into a 4-atom cloud; fall back to the Z scan.
    tpl = _cloud([6, 77, 6, 1], metal_index=torch.tensor(9))
    assert int(_carry_metal_index(tpl).item()) == 1


class _ZeroVelocityModel(torch.nn.Module):
    """Minimal stand-in for CFMJointModel: the sampler only needs a field."""

    def __init__(self, cutoff: float = 3.5):
        super().__init__()
        self.cutoff = float(cutoff)

    def vel_field(self, x, z, edge_index, t, edge_mask=None):
        return {"vel": torch.zeros_like(x)}


@pytest.mark.parametrize("pad_groups", [False, True])
def test_sampler_propagates_metal_index(pad_groups):
    from molmetal.models.bond_head import AtomCloud
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        generate_clouds_from_noise,
    )

    tpl = _cloud([6, 6, 7, 78, 1], metal_index=torch.tensor(3))
    out = generate_clouds_from_noise(
        _ZeroVelocityModel(),
        [tpl],
        torch.device("cpu"),
        method="euler",
        n_steps=1,
        seed=0,
        pad_groups=pad_groups,
        pad_group_bucket=8,
    )
    assert len(out) == 1
    assert out[0].metal_atom_index is not None
    assert int(out[0].metal_atom_index.item()) == 3
    assert int(out[0].atomic_numbers.shape[0]) == 5
