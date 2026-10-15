"""M2/M3 regression: the flat forward must match the per-mol oracle for every
(update, pool, readout) combination.

The oracle convention matches ``test_dmpnn_flat_forward.py``: ``_forward_flat``
sums over padded atoms too (their bias-only ``h_v`` enters the pool), so the
oracle is fed the *padded* width, not the true per-molecule V.

Also asserts the defaults (gru/sum/outgoing) leave the model byte-identical to
the pre-M2/M3 code path — the ablation arms must not perturb the baseline.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/hugo/codes/mol-metal")

from molmetal.baselines.dmpnn import collate_graphs, featurize_smiles_list  # noqa: E402
from molmetal.baselines.dmpnn_multitask import DMPNNMultiTaskModel  # noqa: E402

_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "O=C(O)c1ccccc1O",
    "CCO",
    "c1ccccc1",
    "CCN(CC)CC",
    "OCC(O)CO",
    "CC(C)O",
]


def _batch():
    af, bf, es, ed = collate_graphs(featurize_smiles_list(_SMILES))
    return af, bf, es, ed


def _oracle(model, af, bf, es, ed, true_v: bool):
    """Per-mol oracle.

    ``true_v=False`` feeds the padded width (the flat *sum* path includes
    padded atoms — legacy quirk).  ``true_v=True`` feeds the true per-molecule
    V (the flat *mean* path averages over real atoms only).
    """
    B = af.size(0)
    max_V = af.size(1)
    out = []
    for b in range(B):
        valid_edges = bf[b].abs().sum(dim=-1) > 0
        E = int(valid_edges.sum().item())
        V = int((af[b].abs().sum(dim=-1) > 0).sum().item()) if true_v else max_V
        V = max(V, 1)
        out.append(model._forward_one_mol(af[b, :V], bf[b, :E], es[b, :E], ed[b, :E]))
    return torch.stack(out, dim=0)


def test_m2_m3_flat_matches_oracle_all_combos():
    torch.manual_seed(0)
    np.random.seed(0)
    af, bf, es, ed = _batch()
    for update in ("gru", "residual"):
        for pool in ("sum", "mean"):
            for readout in ("outgoing", "incoming"):
                m = DMPNNMultiTaskModel(
                    hidden=64, depth=3, dropout=0.0,
                    update=update, pool=pool, readout=readout,
                ).eval()
                flat = m._forward_flat(af, bf, es, ed)
                orc = _oracle(m, af, bf, es, ed, true_v=(pool == "mean"))
                assert flat.shape == orc.shape == (len(_SMILES), 128)
                assert torch.isfinite(flat).all()
                assert torch.allclose(flat, orc, rtol=1e-4, atol=1e-5), (
                    f"{update}/{pool}/{readout}: max|diff|="
                    f"{(flat - orc).abs().max().item():.2e}"
                )


def test_m2_m3_defaults_preserve_baseline():
    """The default (gru/sum/outgoing) model must equal a model built with the
    legacy constructor (no M2/M3 kwargs) — i.e. the ablation arms default off."""
    torch.manual_seed(0)
    np.random.seed(0)
    af, bf, es, ed = _batch()
    torch.manual_seed(1)
    m_legacy = DMPNNMultiTaskModel(hidden=64, depth=3, dropout=0.0).eval()
    torch.manual_seed(1)
    m_default = DMPNNMultiTaskModel(
        hidden=64, depth=3, dropout=0.0,
        update="gru", pool="sum", readout="outgoing",
    ).eval()
    with torch.no_grad():
        a = m_legacy._forward_flat(af, bf, es, ed)
        b = m_default._forward_flat(af, bf, es, ed)
    assert torch.allclose(a, b, rtol=0.0, atol=0.0)


def test_residual_update_rejects_bad_kind():
    import pytest

    with pytest.raises(ValueError):
        DMPNNMultiTaskModel(hidden=16, depth=1, update="nope")
    with pytest.raises(ValueError):
        DMPNNMultiTaskModel(hidden=16, depth=1, pool="nope")
    with pytest.raises(ValueError):
        DMPNNMultiTaskModel(hidden=16, depth=1, readout="nope")
