# SPDX-License-Identifier: Apache-2.0
"""Wf-A112 (MOD5): the coordination-number conditioner and its CN label.

What is pinned here
-------------------
1. ``coordination_number_from_bonds`` counts truth bonds incident to the metal
   in the LOCAL index frame ``Batch`` actually uses, returns 0 for a cloud with
   no metal, and does not count a pair twice when the metal is its own
   neighbour (a self-loop is not a coordination bond, and this test says so
   rather than leaving it to chance).
2. ``CNConditioner``'s null request emits an exact ZERO tensor -- this is the
   bit-exactness device: a caller that never heard of MOD5 must get the same
   forward, and the CFG unconditional branch has to be the null, not a learned
   vector.
3. The non-null rows are NOT zero, so the channel can carry a request from the
   first step instead of waiting for the optimiser to dig it out of zero.
4. ``ConditionRequest.dropped`` replaces exactly the sampled rows, so training
   with dropout teaches ``p(v | cn = empty)``.
5. An out-of-range CN is clamped and COUNTED, never a crash and never silent.
"""

from __future__ import annotations

import torch

from molmetal.adapters.flow_matching_lipman.conditioning import (
    CNConditioner,
    ConditionRequest,
    coordination_number_from_bonds,
)


# ---------------------------------------------------------------- CN labels
def test_cn_counts_the_bonds_incident_to_the_metal_in_the_local_frame():
    # cloud 0: atoms 0..3, metal at 0, bonded to 1 and 2
    # cloud 1: atoms 0..3, metal at 2, bonded to 0, 1, 3
    bonds = torch.tensor([[0, 0, 2, 2, 2], [1, 2, 0, 1, 3]], dtype=torch.long)
    pair_cloud_idx = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    metal = torch.tensor([0, 2], dtype=torch.long)
    cn = coordination_number_from_bonds(
        bonds, pair_cloud_idx, metal, n_clouds=2
    )
    assert cn.tolist() == [2, 3]


def test_a_cloud_with_no_metal_gets_zero_rather_than_a_fabricated_value():
    # pair 0 -> cloud 0, which has no metal; pair 1 -> cloud 1, whose metal is
    # atom 0, so cloud 1's CN is 1 and cloud 0's is 0 even though its own pair
    # touches atom 0 of ITS OWN frame.
    bonds = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    pair_cloud_idx = torch.tensor([0, 1], dtype=torch.long)
    metal = torch.tensor([-1, 0], dtype=torch.long)
    cn = coordination_number_from_bonds(bonds, pair_cloud_idx, metal, n_clouds=2)
    assert cn.tolist() == [0, 1]


def test_the_local_frame_is_the_one_batch_uses():
    """The same pair indices must mean different atoms in different clouds.

    ``Batch`` concatenates each cloud's bond block WITHOUT adding the offset,
    so pair ``(0, 1)`` of cloud 1 is that cloud's atoms 0 and 1.  A
    flat-frame implementation would read cloud 1's pair against cloud 0's
    metal and report a wrong count; this is the test that fails if the frame
    is ever "fixed" to the other convention.
    """
    bonds = torch.tensor([[0], [1]], dtype=torch.long)
    pair_cloud_idx = torch.tensor([1], dtype=torch.long)
    metal = torch.tensor([2, 0], dtype=torch.long)  # cloud 1's metal is atom 0
    cn = coordination_number_from_bonds(bonds, pair_cloud_idx, metal, n_clouds=2)
    assert cn.tolist() == [0, 1]


def test_a_self_loop_on_the_metal_counts_once_and_that_is_the_documented_choice():
    bonds = torch.tensor([[0], [0]], dtype=torch.long)
    pair_cloud_idx = torch.tensor([0], dtype=torch.long)
    metal = torch.tensor([0], dtype=torch.long)
    cn = coordination_number_from_bonds(bonds, pair_cloud_idx, metal, n_clouds=1)
    # One pair, one increment: the pair touches the metal on both ends, and
    # counting it twice would report a coordination number that no geometry
    # can produce.
    assert cn.tolist() == [1]


def test_an_empty_bond_list_is_zero_not_an_error():
    cn = coordination_number_from_bonds(
        torch.zeros((2, 0), dtype=torch.long),
        torch.zeros((0,), dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
        n_clouds=2,
    )
    assert cn.tolist() == [0, 0]


# ------------------------------------------------------------ conditioner
def test_a_null_request_emits_an_exact_zero_bias():
    cond = CNConditioner(hidden_dim=16, cn_max=8)
    out = cond(None, b=5)
    assert out.shape == (5, 1, 16)
    assert torch.equal(out, torch.zeros_like(out))
    out0 = cond(torch.zeros(5, dtype=torch.long), b=5)
    assert torch.equal(out0, torch.zeros_like(out0))


def test_the_null_row_is_zero_but_the_real_rows_are_not():
    cond = CNConditioner(hidden_dim=16, cn_max=8)
    assert torch.equal(cond.embed.weight[0], torch.zeros(16))
    for k in range(1, 9):
        assert float(cond.embed.weight[k].abs().sum()) > 0.0


def test_the_request_changes_the_bias_and_the_same_request_is_deterministic():
    cond = CNConditioner(hidden_dim=16, cn_max=8)
    a = cond(torch.tensor([4]), b=1)
    b = cond(torch.tensor([6]), b=1)
    assert not torch.allclose(a, b)
    assert torch.equal(a, cond(torch.tensor([4]), b=1))


def test_an_out_of_range_cn_is_clamped_and_counted():
    cond = CNConditioner(hidden_dim=8, cn_max=4)
    out = cond(torch.tensor([9]), b=1)
    assert torch.equal(out, cond.embed.weight[4].view(1, 1, 8))
    assert cond.n_clamped == 1


def test_a_negative_cn_falls_back_to_the_null_row():
    cond = CNConditioner(hidden_dim=8, cn_max=4)
    out = cond(torch.tensor([-3]), b=1)
    assert torch.equal(out, torch.zeros_like(out))


def test_a_mismatched_request_length_is_an_error_not_a_broadcast():
    cond = CNConditioner(hidden_dim=8, cn_max=4)
    try:
        cond(torch.tensor([1, 2]), b=3)
    except ValueError:
        return
    raise AssertionError("a 2-entry request for a batch of 3 must raise")


# --------------------------------------------------------------- dropout
def test_dropout_replaces_exactly_the_sampled_rows_with_the_null_token():
    req = ConditionRequest(cn=torch.tensor([3, 3, 3, 3, 3, 3, 3, 3]))
    g = torch.Generator().manual_seed(0)
    dropped = req.dropped(0.5, generator=g)
    kept = (dropped.cn == 3).sum().item()
    zeroed = (dropped.cn == 0).sum().item()
    assert kept + zeroed == 8
    assert 0 < zeroed < 8
    # The original is untouched: the request object is not mutated in place.
    assert (req.cn == 3).all()


def test_dropout_zero_is_a_no_op():
    req = ConditionRequest(cn=torch.tensor([5, 2]))
    out = req.dropped(0.0)
    assert torch.equal(out.cn, req.cn)


def test_dropout_one_drops_everything():
    req = ConditionRequest(cn=torch.tensor([5, 2]))
    out = req.dropped(1.0)
    assert torch.equal(out.cn, torch.zeros_like(req.cn))


# ------------------------------------------------------- velocity-field wiring
def _batch(b: int = 2, n: int = 6, seed: int = 0):
    from molmetal.adapters.flow_matching_lipman import radius_edge_index

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, n, 3, generator=g) * 1.5
    z = torch.randint(1, 20, (b, n), generator=g)
    node_mask = torch.ones(b, n, dtype=torch.bool)
    edge_index, edge_mask = radius_edge_index(x, node_mask, cutoff=10.0)
    return x, z, edge_index, edge_mask


def _field(cn: bool, seed: int = 0):
    from molmetal.adapters.flow_matching_lipman import EGNNVelocityField

    torch.manual_seed(seed)
    return EGNNVelocityField(
        hidden_dim=16,
        n_layers=2,
        max_atomic_number=32,
        context_dropout=0.0,
        cn_conditioning=cn,
        cn_max=8,
    ).eval()


def _wake_head(field, seed: int = 5) -> None:
    """The velocity head is zero-initialised; a fresh field emits v = 0."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        field.vel_head.weight.copy_(
            torch.randn(field.vel_head.weight.shape, generator=g) * 0.2
        )
        if field.vel_head.bias is not None:
            field.vel_head.bias.copy_(
                torch.randn(field.vel_head.bias.shape, generator=g) * 0.05
            )


def _vel(field, batch, cond_cn=None):
    x, z, edge_index, edge_mask = batch
    t = torch.full((x.shape[0],), 0.5)
    return field(x, z, edge_index, t, edge_mask=edge_mask, cond_cn=cond_cn)["vel"]


def test_the_default_build_constructs_no_cn_module():
    field = _field(cn=False)
    assert field.cn_cond is None
    assert not any("cn_cond" in k for k in field.state_dict())


def test_a_null_request_is_the_unconditioned_forward_exactly():
    batch = _batch()
    field = _field(cn=True)
    _wake_head(field)
    v_none = _vel(field, batch, cond_cn=None)
    v_null = _vel(field, batch, cond_cn=torch.zeros(2, dtype=torch.long))
    assert torch.equal(v_none, v_null)


def test_a_real_request_moves_the_velocity():
    batch = _batch()
    field = _field(cn=True)
    _wake_head(field)
    v_null = _vel(field, batch, cond_cn=torch.zeros(2, dtype=torch.long))
    v_six = _vel(field, batch, cond_cn=torch.tensor([6, 6]))
    assert not torch.allclose(v_null, v_six)
    # The conditioning is additive on h, so it must also be deterministic.
    assert torch.equal(v_six, _vel(field, batch, cond_cn=torch.tensor([6, 6])))


def test_two_different_requests_give_different_velocities():
    batch = _batch()
    field = _field(cn=True)
    _wake_head(field)
    v4 = _vel(field, batch, cond_cn=torch.tensor([4, 4]))
    v6 = _vel(field, batch, cond_cn=torch.tensor([6, 6]))
    assert not torch.allclose(v4, v6)


def test_gradient_reaches_the_requested_row_and_not_the_null_row():
    batch = _batch()
    field = _field(cn=True)
    _wake_head(field)
    v = _vel(field, batch, cond_cn=torch.tensor([6, 6]))
    v.sum().backward()
    g = field.cn_cond.embed.weight.grad
    assert g is not None
    assert float(g[6].abs().sum()) > 0.0
    assert float(g[0].abs().sum()) == 0.0


def test_building_the_conditioner_does_not_advance_the_global_rng():
    """A conditioning channel must not perturb the rest of the model.

    MEASURED hazard this pins: the trainer seeds the global RNG once, before
    building the model.  An ``nn.init.normal_`` inside this module would then
    shift every later draw -- the remaining initialisation, the first dropout
    mask, the first noise sample -- so an on/off A/B of the channel would
    compare two different models and read the difference as the channel's
    effect.  The null-request-is-bit-exact claim is unprovable without this.
    """
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    CNConditioner(hidden_dim=16, cn_max=8)
    after = torch.get_rng_state()
    assert torch.equal(before, after), "constructing CNConditioner consumed RNG"


def test_the_conditioner_is_deterministic_across_instances():
    a = CNConditioner(hidden_dim=16, cn_max=8).embed.weight
    b = CNConditioner(hidden_dim=16, cn_max=8).embed.weight
    assert torch.equal(a, b)


def test_the_label_survives_a_real_batch_collate():
    """The frame is a property of ``Batch.collate_fn``, so pin it against it.

    ``coordination_number_from_bonds`` is only correct because the collate
    concatenates each cloud's ``(2, E)`` block verbatim.  If that ever starts
    adding a per-cloud atom offset, the hand-built fixture above would keep
    passing while every cloud after the first got a wrong label -- the
    two-definitions defect this module's docstring warns about.  Building the
    batch through the real collate makes that change fail HERE.
    """
    from molmetal.data.batch import Batch
    from molmetal.models.bond_head import AtomCloud

    # cloud 0: metal at 0, bonds (0-1), (0-2)  -> CN 2
    # cloud 1: metal at 2, bonds (2-0), (2-1), (2-3)  -> CN 3
    c0 = AtomCloud(
        positions=torch.zeros(4, 3),
        atomic_numbers=torch.tensor([78, 6, 7, 8]),
        bonds=torch.tensor([[0, 0], [1, 2]]),
        bond_types=torch.tensor([1, 1]),
        metal_atom_index=torch.tensor(0),
    )
    c1 = AtomCloud(
        positions=torch.zeros(4, 3),
        atomic_numbers=torch.tensor([6, 7, 78, 8]),
        bonds=torch.tensor([[2, 2, 2], [0, 1, 3]]),
        bond_types=torch.tensor([1, 1, 1]),
        metal_atom_index=torch.tensor(2),
    )
    batch = Batch.collate_fn([c0, c1])
    cn = coordination_number_from_bonds(
        batch.bonds,
        batch.pair_cloud_idx,
        batch.metal_atom_index,
        n_clouds=2,
    )
    assert cn.tolist() == [2, 3]


def test_the_metal_index_survives_the_device_round_trip():
    """``_to_device_cloud`` must carry the metal index, not just the bonds.

    MEASURED defect this pins (2026-10-26): the helper rebuilt each cloud from
    a one-cloud ``Batch`` and copied positions / atomic numbers / bonds /
    bond types only, so ``metal_atom_index`` became ``NO_METAL_INDEX`` for
    every cloud the training loop drew.  ``coordination_number_from_bonds``
    then returned 0 for every cloud, the MOD5 conditioner's row 0 was the only
    row ever used, and the channel was inert -- an on/off A/B would have
    reported "no effect" on a channel that never received a label.
    """
    from molmetal.data.batch import Batch
    from molmetal.models.bond_head import AtomCloud
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import _to_device_cloud

    cloud = AtomCloud(
        positions=torch.zeros(4, 3),
        atomic_numbers=torch.tensor([6, 7, 78, 8]),
        bonds=torch.tensor([[2, 2], [0, 3]]),
        bond_types=torch.tensor([1, 1]),
        metal_atom_index=torch.tensor(2),
    )
    moved = _to_device_cloud(cloud, torch.device("cpu"))
    assert moved.metal_atom_index is not None, "metal index was dropped"
    assert int(moved.metal_atom_index.item()) == 2

    # And the label computed from the moved cloud is the same as from the
    # original: the round trip is not allowed to change the supervision.
    batch = Batch.collate_fn([moved])
    cn = coordination_number_from_bonds(
        batch.bonds,
        batch.pair_cloud_idx,
        batch.metal_atom_index,
        n_clouds=1,
    )
    assert cn.tolist() == [2]
