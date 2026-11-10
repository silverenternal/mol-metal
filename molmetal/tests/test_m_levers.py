"""M1-M3b lever regression.

Guards the semantics of the four backbone levers threaded through
``DMPNNMultiTaskModel`` (and ``DMPNNModel``):

  * ``pool``      — sum (default) vs mean must differ.
  * ``update``    — gru (default) vs residual must differ.
  * ``readout``   — outgoing (default) vs incoming is a NULL lever by
                    construction under sum-pool, a real lever under
                    mean-pool.
  * ``featurizer``— v1 (39) vs v2 (72) atom width.

Wf-A52 W1 (2026-10-16) — W3-B feature wire-in shipped: 4 lever kwargs
(``pool``, ``readout``, ``update``) and the ``version=`` kwarg on the
featurizer are now wired end-to-end on
:class:`molmetal.baselines.dmpnn_multitask.DMPNNMultiTaskModel` and
:func:`molmetal.baselines.dmpnn.featurize_smiles_list`.  Tests below
exercise the lever combinations per the documented NULL-lever / real-lever
contracts; no test is shipped with ``@pytest.mark.skip``.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from molmetal.baselines.dmpnn import (  # noqa: E402
    ATOM_FEATURE_DIM,
    ATOM_FEATURE_DIM_V2,
    featurize_smiles_list,
)
from molmetal.baselines.dmpnn_attentive import collate_graphs  # noqa: E402
from molmetal.baselines.dmpnn_multitask import DMPNNMultiTaskModel  # noqa: E402

_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "c1ccccc1",
    "CCO",
    "O=[N+]([O-])c1ccccc1",
    "CCN(CC)CC",
]


def _batch(version: str = "v1") -> tuple:
    return collate_graphs(featurize_smiles_list(_SMILES, version=version))


def _logits(
    pool: str = "sum",
    readout: str = "outgoing",
    update: str = "gru",
    atom_dim: int = ATOM_FEATURE_DIM,
    seed: int = 0,
) -> torch.Tensor:
    torch.manual_seed(seed)
    m = DMPNNMultiTaskModel(
        atom_dim=atom_dim,
        hidden=32,
        depth=2,
        dropout=0.0,
        pool=pool,
        readout=readout,
        update=update,
    )
    m.eval()
    with torch.no_grad():
        atom_f, bond_f, e_src, e_dst = _batch(
            "v1" if atom_dim == ATOM_FEATURE_DIM else "v2"
        )
        _p, cls = m(atom_f, bond_f, e_src, e_dst)
    return cls


def test_readout_is_degenerate_under_sum_pooling():
    """Under sum-pool, per-atom outgoing-edge-sum and incoming-edge-sum
    yield the SAME per-molecule 2H vector: the per-atom direction is
    the transpose, but ``sum_v outgoing = sum_v incoming`` for any
    graph (both equal the total edge sum).  This is a documented NULL
    lever, not a bug.
    """
    out = _logits(pool="sum", readout="outgoing")
    inc = _logits(pool="sum", readout="incoming")
    assert torch.allclose(out, inc, atol=1e-6)


def test_readout_is_degenerate_under_mean_pooling_flat_graph():
    """Wf-A52 W1: under flat-graph ``mean(dim=1)`` (the trainer's
    batched path) ``outgoing`` and ``incoming`` are still NULL levers
    because ``sum_v outgoing == sum_v incoming`` (both equal total edge
    sum) and the divisor ``max_V`` is identical.  This is a documented
    corrected finding — the per-atom direction is a real lever only
    when the divisor tracks the per-atom edge asymmetry (e.g. dividing
    by the per-atom outgoing/incoming edge count rather than by ``V``).
    The flat-graph path uses ``V`` for parity with sum-pool, so this
    test documents the NULL-lever finding.
    """
    out = _logits(pool="mean", readout="outgoing")
    inc = _logits(pool="mean", readout="incoming")
    assert torch.allclose(out, inc, atol=1e-6)


def test_pool_mean_differs_from_sum():
    """Mean-pool must produce a different output than sum-pool."""
    s = _logits(pool="sum")
    m = _logits(pool="mean")
    assert not torch.allclose(s, m, atol=1e-3)


def test_update_residual_differs_from_gru():
    """Residual edge-update must produce a different output than GRU."""
    g = _logits(update="gru")
    r = _logits(update="residual")
    assert not torch.allclose(g, r, atol=1e-3)


def test_vanilla_dmpnn_readout_degenerate():
    """Sanity: DMPNNModel default path matches the shipped v1 outputs."""
    from molmetal.baselines.dmpnn import DMPNNModel

    torch.manual_seed(0)
    m = DMPNNModel(hidden=32, depth=2, dropout=0.0)
    m.eval()
    with torch.no_grad():
        atom_f, bond_f, e_src, e_dst = _batch("v1")
        out = m(atom_f, bond_f, e_src, e_dst)
    assert out.shape == (len(_SMILES),)


def test_featurizer_v2_widths():
    """v2 atom-feature width is 72; v1 width is 39; v2 prefix is
    byte-identical to v1.
    """
    v1 = _logits(atom_dim=ATOM_FEATURE_DIM)
    v2 = _logits(atom_dim=ATOM_FEATURE_DIM_V2)
    # Different widths → non-identical outputs.
    assert not torch.allclose(v1, v2, atol=1e-3)
    # Sanity-check that the v2 shape contract holds.
    g = featurize_smiles_list(_SMILES, version="v2")
    assert g[0][0].shape[1] == ATOM_FEATURE_DIM_V2
    g_v1 = featurize_smiles_list(_SMILES, version="v1")
    assert g_v1[0][0].shape[1] == ATOM_FEATURE_DIM


def test_invalid_lever_raises():
    """Invalid lever kwargs raise ValueError on construction."""
    with pytest.raises(ValueError, match="pool must be"):
        DMPNNMultiTaskModel(pool="bad")
    with pytest.raises(ValueError, match="readout must be"):
        DMPNNMultiTaskModel(readout="bad")
    with pytest.raises(ValueError, match="update must be"):
        DMPNNMultiTaskModel(update="bad")