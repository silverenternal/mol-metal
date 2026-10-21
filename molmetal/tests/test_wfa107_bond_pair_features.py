"""Wf-A107 — RBF distance + pair-chemistry features on the bond head.

Wf-A72 Operators 2 and 3 were planned on 2026-10-18 and never shipped
(only Operator 1, per-class logit adjustment, landed as Wf-A73).  These
tests pin the contract the trainer relies on:

  1. default off -> the head's widths are exactly the pre-A107 widths,
  2. the RBF block peaks at its centre and decays,
  3. the 5 chemistry columns encode the pair classes they claim,
  4. with the flags on the forward runs, is finite, and the new input
     weights receive a gradient,
  5. the residual subclass forwards the flags (it had its own signature).
"""
from __future__ import annotations

import pytest
import torch

from molmetal.models.bond_head import AtomCloud
from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
    PAIR_CHEMISTRY_DIM,
    RBF_FEATURE_DIM,
    YuelBondDecoderHead,
    YuelBondEdgeMPNNHead,
    YuelBondEdgeMPNNHeadResidual,
    _pair_chemistry_features,
    rbf_expand,
)


def _cloud() -> AtomCloud:
    # C, C, N, Pt, H, C at hand-chosen distances (some < 3.5, some > 3.5).
    z = torch.tensor([6, 6, 7, 78, 1, 6], dtype=torch.long)
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 1.4, 0.0],
            [2.0, 1.4, 0.0],
            [1.0, 1.0, 1.0],
            [4.0, 4.0, 4.0],
        ],
        dtype=torch.float32,
    )
    return AtomCloud(positions=pos, atomic_numbers=z)


def test_default_widths_are_the_pre_a107_widths():
    h = YuelBondDecoderHead(hidden_dim=32, n_layers=2)
    assert h.use_rbf_distance is False and h.use_pair_chemistry is False
    assert h.msg_mlps[0][0].in_features == 32 * 2 + 4
    assert h.pair_head[0].in_features == 32 * 2 + 3


def test_rbf_block_shape_and_peak():
    d = torch.tensor([0.9, 1.5, 2.2, 3.5])
    out = rbf_expand(d, n_rbf=8)
    assert out.shape == (4, 8)
    # Every row is a Gaussian bump: max at the nearest centre, and the
    # row sums are finite and positive.
    assert torch.isfinite(out).all() and (out > 0).all()
    assert out.sum(dim=1).min() > 0.0
    # A distance equal to a centre activates that centre most.
    centres = torch.exp(torch.linspace(torch.log(torch.tensor(0.9)),
                                       torch.log(torch.tensor(3.5)), 8))
    k = int(torch.argmin((centres - 2.2).abs()))
    row = rbf_expand(torch.tensor([float(centres[k])]), n_rbf=8)[0]
    assert int(torch.argmax(row)) == k


def test_rbf_rejects_bad_geometry():
    with pytest.raises(ValueError):
        rbf_expand(torch.tensor([1.0]), n_rbf=0)
    with pytest.raises(ValueError):
        rbf_expand(torch.tensor([1.0]), d_min=3.0, d_max=1.0)


def test_pair_chemistry_columns():
    # (C,C) (C,N) (N,O) (Pt,N) (H,C)
    z_i = torch.tensor([6, 6, 7, 78, 1])
    z_j = torch.tensor([6, 7, 8, 7, 6])
    out = _pair_chemistry_features(z_i, z_j, frozenset({78}))
    assert out.shape == (5, PAIR_CHEMISTRY_DIM)
    assert out.tolist() == [
        [1.0, 0.0, 0.0, 0.0, 0.0],  # C-C
        [0.0, 1.0, 0.0, 0.0, 0.0],  # C-N
        [0.0, 0.0, 1.0, 0.0, 0.0],  # N-O
        [0.0, 0.0, 0.0, 1.0, 0.0],  # Pt-N
        [0.0, 0.0, 0.0, 0.0, 1.0],  # H-C
    ]


@pytest.mark.parametrize(
    "cls", [YuelBondDecoderHead, YuelBondEdgeMPNNHead, YuelBondEdgeMPNNHeadResidual]
)
def test_features_on_forward_is_finite_and_trainable(cls):
    torch.manual_seed(0)
    head = cls(
        hidden_dim=32,
        use_rbf_distance=True,
        use_pair_chemistry=True,
        n_rbf=RBF_FEATURE_DIM,
    )
    logits, (i_idx, j_idx) = head(_cloud())
    assert logits.shape[1] == 5
    assert i_idx.shape == j_idx.shape == (logits.shape[0],)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    # The first Linear of each block must have received a gradient.
    first = (
        head.msg_mlps[0][0]
        if hasattr(head, "msg_mlps")
        else head.edge_init
    )
    assert first.weight.grad is not None
    assert first.weight.grad.abs().sum() > 0.0


def test_features_on_change_the_width_by_the_declared_amount():
    h = YuelBondDecoderHead(
        hidden_dim=32, n_layers=2, use_rbf_distance=True, use_pair_chemistry=True
    )
    extra = RBF_FEATURE_DIM + PAIR_CHEMISTRY_DIM
    assert h.msg_mlps[0][0].in_features == 32 * 2 + 4 + extra
    assert h.pair_head[0].in_features == 32 * 2 + 3 + extra


def test_residual_subclass_forwards_donor_set():
    head = YuelBondEdgeMPNNHeadResidual(hidden_dim=16, donor_zs=(7, 8))
    assert head.donor_zs == frozenset({7, 8})
