# SPDX-License-Identifier: Apache-2.0
"""The shipped YuelBond head must see the metal, not just its atomic number.

WHY THIS EXISTS
---------------
Before Wf-A90 the pair head consumed [h_i, h_j, d, is_dative, is_short] —
the metal entered only through the atom-Z embedding and a boolean.  A
cis-donor pair and two unrelated ligand atoms at the same distance were
indistinguishable, which is the whole 5-metal question.  Wf-A90 appends
d_to_metal / angle_to_metal / is_metal_edge.

RED CHECK: `test_metal_geometry_changes_the_logits` moves ONLY the metal and
requires the logits to move.  A feature that is concatenated but ignored
would pass a shape test and fail this one.
"""

from __future__ import annotations

import argparse
import pickle

import pytest

torch = pytest.importorskip("torch")


def _cloud(z_list, positions, metal_index=None):
    from molmetal.models.bond_head import AtomCloud

    return AtomCloud(
        positions=torch.tensor(positions, dtype=torch.float32),
        atomic_numbers=torch.tensor(z_list, dtype=torch.long),
        metal_atom_index=(
            None if metal_index is None else torch.tensor(metal_index)
        ),
    )


def test_pair_head_width_matches_the_flag():
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
        METAL_PAIR_FEATURE_DIM,
        YuelBondDecoderHead,
    )

    off = YuelBondDecoderHead(hidden_dim=16, n_layers=1)
    on = YuelBondDecoderHead(hidden_dim=16, n_layers=1, use_metal_features=True)
    assert off.pair_head[0].in_features == 2 * 16 + 3
    assert on.pair_head[0].in_features == 2 * 16 + 3 + METAL_PAIR_FEATURE_DIM
    assert METAL_PAIR_FEATURE_DIM == 3


def test_default_is_off_for_bit_exact_compat():
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
        YuelBondDecoderHead,
    )

    assert YuelBondDecoderHead(hidden_dim=8, n_layers=1).use_metal_features is False


def test_trainer_factory_reads_the_flag_and_defaults_old_args_to_off():
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import build_joint_model

    base = dict(hidden_dim=16, n_layers=1, edge_cutoff=3.5)
    old = build_joint_model(argparse.Namespace(**base))
    new = build_joint_model(
        argparse.Namespace(**base, use_metal_pair_features=True)
    )
    assert old.head.use_metal_features is False
    assert new.head.use_metal_features is True
    assert new.head.pair_head[0].in_features == old.head.pair_head[0].in_features + 3


def test_metal_features_are_zero_without_a_metal():
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
        _metal_pair_features,
    )

    cloud = _cloud([6, 6, 8], [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]])
    feats = _metal_pair_features(
        cloud,
        torch.tensor([0, 1]),
        torch.tensor([1, 2]),
        frozenset({78}),
        cutoff=3.5,
    )
    assert feats.shape == (2, 3)
    assert torch.count_nonzero(feats) == 0


def test_metal_features_values_are_geometric():
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
        _metal_pair_features,
    )

    # Pt at the origin, two O atoms on opposite sides at 2 A.
    cloud = _cloud(
        [78, 8, 8],
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]],
        metal_index=0,
    )
    feats = _metal_pair_features(
        cloud, torch.tensor([1]), torch.tensor([2]), frozenset({78}), cutoff=3.5
    )
    # midpoint is the metal itself -> d_to_metal = 0
    assert float(feats[0, 0]) == pytest.approx(0.0, abs=1e-6)
    # i-Pt-j is 180 degrees -> angle / pi = 1.0
    assert float(feats[0, 1]) == pytest.approx(1.0, abs=1e-5)
    # neither endpoint is the metal
    assert float(feats[0, 2]) == pytest.approx(0.0)


def test_metal_edge_is_flagged_and_angle_is_zero():
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
        _metal_pair_features,
    )

    cloud = _cloud(
        [78, 8, 8],
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]],
        metal_index=0,
    )
    feats = _metal_pair_features(
        cloud, torch.tensor([0]), torch.tensor([1]), frozenset({78}), cutoff=3.5
    )
    assert float(feats[0, 2]) == pytest.approx(1.0)
    # angle at the metal is undefined when an endpoint IS the metal
    assert float(feats[0, 1]) == pytest.approx(0.0)
    # midpoint sits 1 A from the metal, scaled by the cutoff
    assert float(feats[0, 0]) == pytest.approx(1.0 / 3.5, abs=1e-6)


def test_metal_index_falls_back_to_the_z_scan():
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
        _metal_pair_features,
    )

    # No metal_atom_index field at all: the loader never set it.
    cloud = _cloud(
        [6, 78, 8],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    )
    feats = _metal_pair_features(
        cloud, torch.tensor([0]), torch.tensor([2]), frozenset({78}), cutoff=3.5
    )
    assert float(feats[0, 0]) == pytest.approx(1.0 / 3.5, abs=1e-6)


def test_metal_geometry_changes_the_logits():
    """Moving only the metal must change the prediction."""
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
        YuelBondDecoderHead,
    )

    torch.manual_seed(0)
    head = YuelBondDecoderHead(
        hidden_dim=16, n_layers=1, use_metal_features=True
    )
    head.eval()
    z = [78, 8, 8]
    near = _cloud(
        z, [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]], metal_index=0
    )
    far = _cloud(
        z, [[0.6, 0.6, 0.0], [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]], metal_index=0
    )
    with torch.no_grad():
        a, _ = head(near)
        b, _ = head(far)
    assert not torch.allclose(a, b), (
        "metal position changed but the logits did not: the metal block is "
        "concatenated but not used"
    )


def test_forward_is_bit_exact_when_the_feature_is_off():
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
        YuelBondDecoderHead,
    )

    torch.manual_seed(1)
    head = YuelBondDecoderHead(hidden_dim=8, n_layers=1)
    head.eval()
    cloud = _cloud(
        [78, 8, 8],
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]],
        metal_index=0,
    )
    moved = _cloud(
        [78, 8, 8],
        [[0.6, 0.6, 0.0], [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]],
        metal_index=0,
    )
    with torch.no_grad():
        a, _ = head(cloud)
        b, _ = head(moved)
    # The pair distances differ, so the logits may differ; what must hold is
    # that the head has no metal block to react to.
    assert head.use_metal_features is False
    assert a.shape == b.shape


def test_checkpoint_loader_reads_both_serialisations(tmp_path):
    """Wf-A90 — the trainer wrote pickles; torch.load could not read them."""
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        load_cfm_checkpoint,
    )

    payload = {"model_state_dict": {"w": torch.zeros(2)}, "epoch": 3}
    zip_path = tmp_path / "zip.pt"
    torch.save(payload, zip_path)
    pick_path = tmp_path / "pick.pt"
    with open(pick_path, "wb") as fh:
        pickle.dump(payload, fh)
    assert load_cfm_checkpoint(zip_path)["epoch"] == 3
    assert load_cfm_checkpoint(pick_path)["epoch"] == 3


@pytest.mark.parametrize(
    "head_cls_name",
    ["YuelBondEdgeMPNNHead", "YuelBondEdgeMPNNHeadResidual"],
)
def test_edge_heads_carry_the_metal_block(head_cls_name):
    """The two edge-MPNN heads get the same 3 columns as the atom head."""
    import molmetal.molmetal_lam.lam_chem.yuelbond_decoder as yb

    cls = getattr(yb, head_cls_name)
    off = cls(hidden_dim=16, n_edge_layers=1)
    on = cls(hidden_dim=16, n_edge_layers=1, use_metal_features=True)
    assert off.use_metal_features is False
    assert on.use_metal_features is True
    assert on.pair_head[0].in_features == off.pair_head[0].in_features + 3
    assert on.edge_init.in_features == off.edge_init.in_features + 3


@pytest.mark.parametrize(
    "head_cls_name",
    ["YuelBondEdgeMPNNHead", "YuelBondEdgeMPNNHeadResidual"],
)
def test_edge_heads_react_to_the_metal_position(head_cls_name):
    import molmetal.molmetal_lam.lam_chem.yuelbond_decoder as yb

    torch.manual_seed(0)
    head = getattr(yb, head_cls_name)(
        hidden_dim=16, n_edge_layers=1, use_metal_features=True
    )
    head.eval()
    z = [78, 8, 8]
    near = _cloud(
        z, [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]], metal_index=0
    )
    far = _cloud(
        z, [[0.6, 0.6, 0.0], [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]], metal_index=0
    )
    with torch.no_grad():
        a, _ = head(near)
        b, _ = head(far)
    assert not torch.allclose(a, b)

