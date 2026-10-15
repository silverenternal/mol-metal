"""M1-M3b lever regression (re-implemented 2026-10-15).

Guards the semantics of the four backbone levers threaded through
``DMPNNMultiTaskModel`` (and ``DMPNNModel``):

  * ``pool``      — sum (default) vs mean must differ.
  * ``update``    — gru (default) vs residual must differ.
  * ``readout``   — outgoing (default) vs incoming is a NULL lever by
                    construction: the per-atom edge aggregate summed over
                    atoms equals the total edge sum regardless of direction,
                    so the pooled vector is invariant under any sum/mean
                    pooling.  This is a documented corrected finding, not a bug.
  * ``featurizer``— v1 (39) vs v2 (72) atom width.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/hugo/codes/mol-metal")

from molmetal.baselines.dmpnn import (  # noqa: E402
    ATOM_FEATURE_DIM,
    ATOM_FEATURE_DIM_V2,
    DMPNNModel,
)
from molmetal.baselines.dmpnn_attentive import collate_graphs, featurize_smiles_list  # noqa: E402
from molmetal.baselines.dmpnn_multitask import DMPNNMultiTaskModel  # noqa: E402

_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "CCO", "O=[N+]([O-])c1ccccc1", "CCN(CC)CC",
]


def _batch(version: int = 1):
    return collate_graphs(featurize_smiles_list(_SMILES, version=version))


def _logits(pool: str, readout: str, update: str, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    m = DMPNNMultiTaskModel(
        atom_dim=ATOM_FEATURE_DIM, hidden=32, depth=2, dropout=0.0,
        pool=pool, readout=readout, update=update,
    )
    m.eval()
    with torch.no_grad():
        _p, cls = m(*_batch())
    return cls


def test_readout_is_degenerate_under_sum_pooling():
    """Swapping the edge-readout direction cannot change the model."""
    a = _logits("sum", "outgoing", "gru")
    b = _logits("sum", "incoming", "gru")
    assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()


def test_readout_is_degenerate_under_mean_pooling():
    a = _logits("mean", "outgoing", "gru")
    b = _logits("mean", "incoming", "gru")
    assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()


def test_pool_mean_differs_from_sum():
    a = _logits("sum", "outgoing", "gru")
    b = _logits("mean", "outgoing", "gru")
    assert not torch.allclose(a, b, atol=1e-4)


def test_update_residual_differs_from_gru():
    a = _logits("sum", "outgoing", "gru")
    b = _logits("sum", "outgoing", "residual")
    assert not torch.allclose(a, b, atol=1e-4)


def test_vanilla_dmpnn_readout_degenerate():
    """Same invariant on the vanilla DMPNNModel (forward returns a scalar)."""
    b = _batch()
    torch.manual_seed(0)
    m_out = DMPNNModel(atom_dim=ATOM_FEATURE_DIM, hidden=32, depth=2, dropout=0.0,
                       pool="sum", readout="outgoing")
    m_in = DMPNNModel(atom_dim=ATOM_FEATURE_DIM, hidden=32, depth=2, dropout=0.0,
                      pool="sum", readout="incoming")
    m_in.load_state_dict(m_out.state_dict())
    m_out.eval(); m_in.eval()
    with torch.no_grad():
        a = m_out(*b)
        c = m_in(*b)
    assert torch.allclose(a, c, atol=1e-6)


def test_featurizer_v2_widths():
    g1 = featurize_smiles_list(_SMILES, version=1)
    g2 = featurize_smiles_list(_SMILES, version=2)
    assert g1[0][0].shape[1] == ATOM_FEATURE_DIM
    assert g2[0][0].shape[1] == ATOM_FEATURE_DIM_V2
    # prefix 0..38 byte-identical
    assert np.array_equal(g1[0][0][:, :ATOM_FEATURE_DIM], g2[0][0][:, :ATOM_FEATURE_DIM])
