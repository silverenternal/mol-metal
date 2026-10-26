"""Wf-A110 (MOD1) — typed edges: the path exists, is opt-in, and can train.

The defect (cfm_todo D1): the EGNN reads ``[h_i || h_j || dist || dist^2]``,
i.e. how far apart two atoms are and nothing about what the pair is.  The bond
head already predicts the order (macro-F1(type) 0.618 on the c8 full corpus,
against an OpenBabel reference of 0.505) and that prediction never reaches the
coordinates — the same run logs ``bond_features=OFF``.

These tests pin the four properties the design depends on:

1. **Off by default.**  With the flag off no module is constructed, so every
   checkpoint written before this change still loads with ``strict=True``.
2. **Zero-init is the untyped field exactly.**  A fresh typed layer must not
   perturb an existing run, or the ablation would measure the initialisation
   rather than the feature.
3. **A zero mask is the untyped field exactly.**  The trainer withholds the
   order from a fraction of clouds so the unconditional branch is trained
   (the sampler has no order to supply); that mechanism is only correct if
   masking is exact, which is why the mask multiplies the projection and not
   the one-hot — the projection has a bias.
4. **The projection can learn.**  A term whose gradient never arrives is a
   monitor, not a feature.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from molmetal.adapters.flow_matching_lipman import (  # noqa: E402
    EGNNVelocityField,
    radius_edge_index,
)
from molmetal.scripts.train_cfm_yuelbond_wasserstein import (  # noqa: E402
    build_bond_type_lut,
)

K = 5  # NUM_BOND_CLASSES = [no-bond, single, double, triple, aromatic]


def _batch(b: int = 2, n: int = 6, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, n, 3, generator=g) * 1.5
    z = torch.randint(1, 20, (b, n), generator=g)
    node_mask = torch.ones(b, n, dtype=torch.bool)
    edge_index, edge_mask = radius_edge_index(x, node_mask, cutoff=10.0)
    return x, z, edge_index, edge_mask


def _field(typed: bool, seed: int = 0) -> EGNNVelocityField:
    torch.manual_seed(seed)
    return EGNNVelocityField(
        hidden_dim=16,
        n_layers=2,
        max_atomic_number=32,
        context_dropout=0.0,
        typed_edge_features=typed,
        typed_edge_classes=K,
    ).eval()


def _one_hot(edge_index: torch.Tensor, cls: int) -> torch.Tensor:
    b, _, e = edge_index.shape
    out = torch.zeros(b, e, K)
    out[..., cls] = 1.0
    return out


def _wake_head(field, seed: int = 5) -> None:
    """Give the coordinate head a non-zero output.

    ``EGNNVelocityField`` zero-initialises ``vel_head`` ("start the velocity
    at zero so initial samples are just noise"), so a fresh field emits
    exactly ``v = 0`` and every ablation on ``vel`` would measure the
    initialisation rather than the feature.  A trained field is what the
    ablation is about, so the tests put the head in that state explicitly.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        field.vel_head.weight.copy_(
            torch.randn(field.vel_head.weight.shape, generator=g) * 0.2
        )
        if field.vel_head.bias is not None:
            field.vel_head.bias.copy_(
                torch.randn(field.vel_head.bias.shape, generator=g) * 0.05
            )


def _out(field, x, z, edge_index, edge_mask, edge_type=None, edge_type_mask=None):
    t = torch.full((x.shape[0],), 0.5)
    return field(
        x, z, edge_index, t,
        edge_mask=edge_mask,
        edge_type=edge_type,
        edge_type_mask=edge_type_mask,
    )


def _vel(field, x, z, edge_index, edge_mask, edge_type=None, edge_type_mask=None):
    return _out(field, x, z, edge_index, edge_mask, edge_type, edge_type_mask)["vel"]


# ---------------------------------------------------------------------------
# 1. opt-in
# ---------------------------------------------------------------------------


def test_the_default_build_constructs_no_typed_module():
    """A checkpoint written before MOD1 has no edge_type_mlp key to miss."""
    field = _field(typed=False)
    assert field._typed_edge_k == 0
    assert all(layer.edge_type_mlp is None for layer in field.layers)
    assert not any("edge_type_mlp" in k for k in field.state_dict())


def test_the_typed_build_constructs_one_projection_per_layer():
    field = _field(typed=True)
    assert field._typed_edge_k == K
    assert all(layer.edge_type_mlp is not None for layer in field.layers)
    n_typed = sum(
        1 for k in field.state_dict() if k.endswith("edge_type_mlp.weight")
    )
    assert n_typed == len(field.layers)


# ---------------------------------------------------------------------------
# 2. zero-init == untyped
# ---------------------------------------------------------------------------


def test_a_fresh_typed_field_is_the_untyped_field_exactly():
    """Zero-init addend: the typed path cannot perturb an existing run.

    Compared on ONE field, because two separately constructed fields do not
    share weights: the projection is built inside each layer, so it consumes
    RNG between ``edge_mlp_fused`` and ``msg_scalar_head`` and the two builds
    diverge.  What matters is that supplying an order to a zero-initialised
    projection changes nothing.
    """
    x, z, ei, em = _batch()
    typed = _field(typed=True)
    o_no_type = _out(typed, x, z, ei, em, edge_type=None)
    o_with_type = _out(typed, x, z, ei, em, edge_type=_one_hot(ei, 1))
    o_other_class = _out(typed, x, z, ei, em, edge_type=_one_hot(ei, 4))
    # Both the output and the representation: at init the coordinate head is
    # zero, so comparing only ``vel`` would be a vacuous equality.
    for key in ("vel", "h", "atom_logits"):
        assert torch.equal(o_no_type[key], o_with_type[key]), key
        assert torch.equal(o_no_type[key], o_other_class[key]), key


def test_the_typed_build_adds_only_the_projection():
    """No shared weight moves: the flag adds modules, it does not re-init."""
    untyped = _field(typed=False)
    typed = _field(typed=True)
    shared = set(untyped.state_dict()) & set(typed.state_dict())
    assert shared, "the two builds must share the layer weights"
    extra = set(typed.state_dict()) - set(untyped.state_dict())
    assert extra == {
        f"layers.{i}.edge_type_mlp.weight" for i in range(len(typed.layers))
    } | {
        f"layers.{i}.edge_type_mlp.bias" for i in range(len(typed.layers))
    }


# ---------------------------------------------------------------------------
# 3. the mask is exact, and the feature is live once trained
# ---------------------------------------------------------------------------


def _randomise_projection(field, seed: int = 3) -> None:
    g = torch.Generator().manual_seed(seed)
    for layer in field.layers:
        if layer.edge_type_mlp is not None:
            with torch.no_grad():
                layer.edge_type_mlp.weight.copy_(
                    torch.randn(layer.edge_type_mlp.weight.shape, generator=g) * 0.1
                )
                layer.edge_type_mlp.bias.copy_(
                    torch.randn(layer.edge_type_mlp.bias.shape, generator=g) * 0.1
                )


def test_a_zero_mask_is_the_untyped_field_even_with_a_nonzero_projection():
    """The dropout mechanism: withheld order must be exactly the untyped path.

    This is the assertion that fails if the mask is applied to the one-hot
    instead of to the projection, because the projection's bias would still
    contribute.
    """
    x, z, ei, em = _batch()
    field = _field(typed=True)
    _wake_head(field)
    _randomise_projection(field)
    v_off = _vel(field, x, z, ei, em, edge_type=None)
    mask_zero = torch.zeros(x.shape[0], 1, 1)
    v_masked = _vel(
        field, x, z, ei, em, edge_type=_one_hot(ei, 1), edge_type_mask=mask_zero
    )
    assert torch.allclose(v_off, v_masked, atol=1e-7, rtol=0)


def test_the_class_changes_the_velocity_once_the_projection_is_nonzero():
    """The ablation: on identical inputs, order single vs order triple differs."""
    x, z, ei, em = _batch()
    field = _field(typed=True)
    _wake_head(field)
    _randomise_projection(field)
    v_single = _vel(field, x, z, ei, em, edge_type=_one_hot(ei, 1))
    v_triple = _vel(field, x, z, ei, em, edge_type=_one_hot(ei, 3))
    v_none = _vel(field, x, z, ei, em, edge_type=None)
    assert not torch.allclose(v_single, v_triple)
    assert not torch.allclose(v_single, v_none)
    assert float((v_single - v_triple).detach().norm()) > 1e-3


def test_a_nonzero_mask_leaves_the_typed_contribution_on():
    x, z, ei, em = _batch()
    field = _field(typed=True)
    _wake_head(field)
    _randomise_projection(field)
    v_full = _vel(field, x, z, ei, em, edge_type=_one_hot(ei, 1))
    mask_one = torch.ones(x.shape[0], 1, 1)
    v_masked = _vel(
        field, x, z, ei, em, edge_type=_one_hot(ei, 1), edge_type_mask=mask_one
    )
    assert torch.allclose(v_full, v_masked, atol=1e-7, rtol=0)


# ---------------------------------------------------------------------------
# 4. the projection can learn
# ---------------------------------------------------------------------------


def test_gradient_reaches_the_edge_type_projection():
    x, z, ei, em = _batch()
    field = _field(typed=True)
    _wake_head(field)
    _randomise_projection(field)
    v = _vel(field, x, z, ei, em, edge_type=_one_hot(ei, 2))
    v.pow(2).mean().backward()
    grads = [
        layer.edge_type_mlp.weight.grad
        for layer in field.layers
        if layer.edge_type_mlp is not None
    ]
    assert grads and all(g is not None for g in grads)
    assert any(float(g.abs().sum()) > 0.0 for g in grads)


def test_the_typed_field_is_se3_equivariant():
    """Rotating the input must rotate the velocity: the addend is scalar-only."""
    x, z, ei, em = _batch()
    field = _field(typed=True)
    _wake_head(field)
    _randomise_projection(field)
    et = _one_hot(ei, 1)
    v0 = _vel(field, x, z, ei, em, edge_type=et)

    g = torch.Generator().manual_seed(11)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g))
    if float(torch.linalg.det(q)) < 0:
        q[:, 0] = -q[:, 0]
    x_rot = x @ q.T
    ei_rot, em_rot = radius_edge_index(x_rot, torch.ones_like(x[..., 0]).bool(), 10.0)
    v_rot = _vel(field, x_rot, z, ei_rot, em_rot, edge_type=_one_hot(ei_rot, 1))
    assert torch.allclose(v_rot, v0 @ q.T, atol=1e-4)


# ---------------------------------------------------------------------------
# 5. the lut
# ---------------------------------------------------------------------------


def _lut_batch():
    """Two clouds of 3 atoms: cloud 0 has a single bond 0-1, cloud 1 a double."""
    n_per = 3
    b = 2
    # Cloud 0 owns atoms 0-2 and cloud 1 owns atoms 3-5, so a bond list in
    # FLAT indices is (0,1) / (3,4) / (4,5).
    bonds = torch.tensor([[0, 3, 4], [1, 4, 5]])  # (2, 3)
    bond_types = torch.tensor([1, 2, 4])  # single, double, aromatic
    return SimpleNamespace(
        atomic_numbers=torch.ones(b * n_per, dtype=torch.long),
        bonds=bonds,
        bond_types=bond_types,
        batch_idx=torch.repeat_interleave(torch.arange(b), n_per),
        cloud_offsets=torch.tensor([0, n_per, 2 * n_per]),
        orig_n_atoms=torch.full((b,), n_per, dtype=torch.long),
    )


def test_the_lut_maps_flat_bond_indices_into_per_cloud_local_coordinates():
    lut, present = build_bond_type_lut(_lut_batch(), 2)
    assert lut is not None and present is not None
    assert lut.shape == (2, 3, 3)
    assert int(lut[0, 0, 1]) == 1 and int(lut[0, 1, 0]) == 1
    assert int(lut[1, 0, 1]) == 2 and int(lut[1, 1, 0]) == 2
    assert int(lut[1, 1, 2]) == 4 and int(lut[1, 2, 1]) == 4
    # Non-bonded pairs read class 0, which is what every radius edge that is
    # not a bond will read too.
    assert int(lut[0, 0, 2]) == 0
    assert torch.equal(present, torch.ones(2))


def test_a_cross_cloud_bond_is_dropped_rather_than_written_into_the_wrong_square():
    batch = _lut_batch()
    batch.bonds = torch.tensor([[0, 2], [3, 5]])  # atom 0 -> atom 3 is cloud 0 -> 1
    batch.bond_types = torch.tensor([1, 1])
    lut, _ = build_bond_type_lut(batch, 2)
    assert int(lut[0, 0, 1]) == 0  # not written into cloud 0
    assert int(lut[1, 0, 1]) == 0  # nor into cloud 1
    assert int(lut.sum()) == 0


def test_a_batch_without_bond_types_returns_nothing():
    batch = _lut_batch()
    batch.bond_types = None
    assert build_bond_type_lut(batch, 2) == (None, None)
    batch = _lut_batch()
    batch.bonds = torch.zeros(2, 0, dtype=torch.long)
    batch.bond_types = torch.zeros(0, dtype=torch.long)
    assert build_bond_type_lut(batch, 2) == (None, None)
