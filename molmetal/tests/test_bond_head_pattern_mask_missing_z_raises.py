"""Wf-A61 (2026-10-17) L9 — ``BondOrderHead`` fails fast when the trainer
opted into ``bond_pattern_mask_in_train=True`` but forgot to wire
``z_i`` / ``z_j``.

The previous silent-skip made the vocab-discipline mask a no-op without
telling anyone, which was a regression vs. the A5 intent (M-A8).
"""

from __future__ import annotations

import pytest
import torch

from molmetal.models.bond_head import BondOrderHead


def _make_head() -> BondOrderHead:
    """Construct a minimal BondOrderHead with the mask flag ON.

    The ``bond_pattern_mask`` is auto-built from ``atom_vocab`` via
    :func:`build_bond_pattern_mask` and registered as a buffer; we
    don't pass it explicitly.  A small atom_vocab (4 elements) keeps
    the test fast.
    """
    return BondOrderHead(
        in_dim=11,
        hidden_dim=16,
        num_classes=5,
        atom_vocab=(1, 6, 7, 8),
        bond_pattern_mask_in_train=True,
    )


def test_bond_head_pattern_mask_missing_z_raises() -> None:
    """L9 — passing no ``z_i``/``z_j`` with the mask flag ON raises."""
    head = _make_head()
    pair_features = torch.zeros((4, 11))
    with pytest.raises(ValueError, match="bond_pattern_mask_in_train=True requires z_i and z_j"):
        head(pair_features, z_i=None, z_j=None)


def test_bond_head_pattern_mask_with_z_runs() -> None:
    """L9 — providing ``z_i`` and ``z_j`` lets the masked branch run."""
    head = _make_head()
    pair_features = torch.zeros((4, 11))
    z_i = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    z_j = torch.tensor([1, 0, 1, 2], dtype=torch.long)
    out = head(pair_features, z_i=z_i, z_j=z_j)
    assert out.shape == (4, 5)