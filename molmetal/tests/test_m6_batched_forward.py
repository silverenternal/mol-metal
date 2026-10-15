"""M6 — equivalence tests for the batched D-MPNN forward.

The batched path (:func:`molmetal.baselines.dmpnn_p0wins.forward_batched_dmpnn`)
runs message passing once over a flat disjoint-union batch instead of the
per-molecule Python loop in :meth:`molmetal.baselines.dmpnn.DMPNNModel.forward`.

Two claims are pinned here:

1. **Same-size equivalence** — when every molecule has the same atom/edge
   count (no padding), the batched forward is bit-identical to the padded
   per-molecule forward.
2. **Padding correction** — for MIXED-size batches the two differ, and the
   batched path is the CORRECT one: it equals a direct call to
   ``_forward_one_mol`` on each molecule's true (unpadded) graph, whereas the
   padded path additionally pools over all-zero padding atoms whose
   ``atom_emb`` bias pollutes the sum.  This is a latent defect in the main
   model: its output depends on batch composition.
"""

from __future__ import annotations

import numpy as np
import torch

from molmetal.baselines.dmpnn import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    DMPNNModel,
    collate_graphs,
    predict,
    train_epoch,
)
from molmetal.baselines.dmpnn_p0wins import (
    batch_mol_graph_from_tuples,
    forward_batched_dmpnn,
)


def _mk(nv: int, ne: int, seed: int):
    rng = np.random.default_rng(seed)
    af = rng.random((nv, ATOM_FEATURE_DIM)).astype(np.float32)
    bf = rng.random((ne, BOND_FEATURE_DIM)).astype(np.float32)
    es = rng.integers(0, nv, size=ne).astype(np.int64)
    ed = rng.integers(0, nv, size=ne).astype(np.int64)
    return (af, bf, es, ed)


def test_batched_equals_padded_when_same_size():
    torch.manual_seed(0)
    m = DMPNNModel()
    m.eval()
    graphs = [_mk(10, 12, s) for s in range(4)]

    af, bf, es, ed = collate_graphs(graphs)
    with torch.no_grad():
        padded = m(af, bf, es, ed)
        batched = forward_batched_dmpnn(m, batch_mol_graph_from_tuples(graphs))

    assert torch.allclose(padded, batched, atol=1e-6), (
        padded.tolist(),
        batched.tolist(),
    )


def test_batched_matches_unpadded_reference_on_mixed_sizes():
    """The batched path must equal a direct per-molecule forward on the TRUE
    (unpadded) graphs — i.e. it is padding-free."""
    torch.manual_seed(0)
    m = DMPNNModel()
    m.eval()
    graphs = [_mk(5, 6, 1), _mk(12, 20, 2), _mk(8, 9, 3)]

    with torch.no_grad():
        ref = torch.stack(
            [
                m._forward_one_mol(
                    torch.from_numpy(af),
                    torch.from_numpy(bf),
                    torch.from_numpy(es),
                    torch.from_numpy(ed),
                )
                for (af, bf, es, ed) in graphs
            ]
        )
        batched = forward_batched_dmpnn(m, batch_mol_graph_from_tuples(graphs))

    assert torch.allclose(ref, batched, atol=1e-6), (
        ref.tolist(),
        batched.tolist(),
    )


def test_padded_path_is_padding_sensitive():
    """Document the latent defect: the padded path changes with padding,
    the batched path does not."""
    torch.manual_seed(0)
    m = DMPNNModel()
    m.eval()
    mol = _mk(8, 9, 7)
    # Same molecule batched with a small partner vs a large partner.
    small = [mol, _mk(8, 9, 8)]
    large = [mol, _mk(40, 60, 9)]

    af_s, bf_s, es_s, ed_s = collate_graphs(small)
    af_l, bf_l, es_l, ed_l = collate_graphs(large)
    with torch.no_grad():
        padded_small = m(af_s, bf_s, es_s, ed_s)[0]
        padded_large = m(af_l, bf_l, es_l, ed_l)[0]
        batched_small = forward_batched_dmpnn(m, batch_mol_graph_from_tuples(small))[0]
        batched_large = forward_batched_dmpnn(m, batch_mol_graph_from_tuples(large))[0]

    # Padded path: same molecule, different answer depending on batch padding.
    assert not torch.allclose(padded_small, padded_large, atol=1e-6)
    # Batched path: invariant to the partner molecule.
    assert torch.allclose(batched_small, batched_large, atol=1e-6)


def test_train_epoch_and_predict_batched_run():
    torch.manual_seed(0)
    m = DMPNNModel()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    graphs = [_mk(6 + (i % 3), 7 + (i % 4), i) for i in range(16)]
    labels = np.array([float(i % 2) for i in range(16)], dtype=np.float32)

    loss = train_epoch(m, opt, graphs, labels, batch_size=8, batched=True)
    assert np.isfinite(loss)

    y, s = predict(m, graphs, labels, batch_size=8, batched=True)
    assert y.shape == s.shape == (16,)
    assert np.isfinite(s).all()
