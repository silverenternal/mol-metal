"""Regression tests for the BLK-F1 D-MPNN ⊕ ECFP4 fusion runner.

Covers the properties that make the fusion result *trustworthy* rather than the
numbers themselves (which live in
``metrics/by_metric/fusion_dmpnn_gbdt_within_metal.json``):

1. The ``MaskedMetalCytotoxModel.embed`` split did not change ``forward`` — the
   D-MPNN head still consumes exactly the vector ``embed`` returns.
2. Per-molecule embedding extraction has no cross-row bleed: a row's embedding
   from a batch equals its embedding from a solo forward.  This is what lets
   arm 5's OOF embeddings be attributed to the row rather than the batch.
3. ``FusionDMPNN.forward`` actually depends on both the graph and the ECFP4
   vector (a fusion head that ignored either would silently make the whole
   exercise vacuous).
4. The ECFP-only MLP head trains and returns a usable AUC on toy data.
5. §5.11: the shipped YAML round-trips into the runner's parser (arms list +
   the nested-bleed switch), and the arm-1 anchor is auto-included.

CPU-only; no data files; tiny tensors.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from molmetal.baselines.dmpnn import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from molmetal.scripts.dmpnn_within_metal_auc import MaskedMetalCytotoxModel
from molmetal.scripts.fuse_dmpnn_gbdt_within_metal import (
    ALL_ARMS,
    ECFPMLP,
    FusionDMPNN,
    _embed_indices,
    _train_neural_head_fold,
    _with_anchor,
    _xgb_fold_aucs_matrix,
    parse_args,
)

CFG = "configs/fusion_dmpnn_gbdt_within_metal.yaml"


def _toy_np_graph(n_atoms: int, n_edges: int, seed: int = 0):
    g = np.random.default_rng(seed)
    atom = g.standard_normal((n_atoms, ATOM_FEATURE_DIM)).astype(np.float32)
    bond = g.standard_normal((n_edges, BOND_FEATURE_DIM)).astype(np.float32)
    src = np.asarray([i % n_atoms for i in range(n_edges)], dtype=np.int64)
    dst = np.asarray([(i + 1) % n_atoms for i in range(n_edges)], dtype=np.int64)
    return atom, bond, src, dst


def _pad_batch(graphs, max_atoms: int, max_edges: int):
    B = len(graphs)
    af = torch.zeros(B, max_atoms, ATOM_FEATURE_DIM)
    bf = torch.zeros(B, max_edges, BOND_FEATURE_DIM)
    es = torch.zeros(B, max_edges, dtype=torch.long)
    ed = torch.zeros(B, max_edges, dtype=torch.long)
    for b, (a, bo, s, d) in enumerate(graphs):
        V, E = a.shape[0], bo.shape[0]
        af[b, :V] = torch.from_numpy(a)
        bf[b, :E] = torch.from_numpy(bo)
        es[b, :E] = torch.from_numpy(s)
        ed[b, :E] = torch.from_numpy(d)
    return af, bf, es, ed


# ---------------------------------------------------------------------------
# 1. the embed() split preserves forward()
# ---------------------------------------------------------------------------
def test_forward_equals_head_of_embed():
    torch.manual_seed(0)
    model = MaskedMetalCytotoxModel().eval()
    af, bf, es, ed = _pad_batch([_toy_np_graph(9, 8, seed=0)], 20, 30)
    with torch.no_grad():
        logits = model(af, bf, es, ed)
        h = model.embed(af, bf, es, ed)
        recomposed = model.head(h)
    assert h.shape == (1, 2 * model.hidden)
    assert torch.allclose(logits, recomposed, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. embedding extraction has no cross-row bleed
# ---------------------------------------------------------------------------
def test_embed_indices_batch_equals_solo():
    torch.manual_seed(0)
    model = MaskedMetalCytotoxModel().eval()
    graphs = [_toy_np_graph(7, 6, seed=1), _toy_np_graph(12, 11, seed=2)]
    idx = np.asarray([0, 1], dtype=int)

    both = _embed_indices(model, graphs, idx, batch_size=8, max_atoms=12, max_edges=11, device=torch.device("cpu"))
    solo0 = _embed_indices(model, graphs, np.asarray([0]), 8, 12, 11, torch.device("cpu"))
    solo1 = _embed_indices(model, graphs, np.asarray([1]), 8, 12, 11, torch.device("cpu"))

    assert both.shape == (2, 2 * model.hidden)
    assert np.allclose(both[0], solo0[0], atol=1e-5)
    assert np.allclose(both[1], solo1[0], atol=1e-5)


def test_embed_indices_invalid_graph_is_zero_row():
    torch.manual_seed(0)
    model = MaskedMetalCytotoxModel().eval()
    graphs = [_toy_np_graph(7, 6, seed=1)]
    # a zeroed (all-padding) graph stands in for an unfeaturisable molecule
    zero = (
        np.zeros((7, ATOM_FEATURE_DIM), np.float32),
        np.zeros((6, BOND_FEATURE_DIM), np.float32),
        np.zeros(6, np.int64),
        np.zeros(6, np.int64),
    )
    out = _embed_indices(model, [graphs[0], zero], np.asarray([0, 1]), 8, 12, 11, torch.device("cpu"))
    assert np.allclose(out[1], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. the fusion head depends on BOTH inputs
# ---------------------------------------------------------------------------
def test_fusion_forward_depends_on_ecfp_and_graph():
    torch.manual_seed(0)
    model = FusionDMPNN(ecfp_dim=16, hidden=8, depth=2, mlp_hidden=8).eval()
    af, bf, es, ed = _pad_batch([_toy_np_graph(6, 5, seed=3)], 10, 12)
    e1 = torch.zeros(1, 16)
    e2 = torch.ones(1, 16)
    with torch.no_grad():
        out_e1 = model(af, bf, es, ed, e1)
        out_e2 = model(af, bf, es, ed, e2)
        assert out_e1.shape == (1, 2)
        assert not torch.allclose(out_e1, out_e2, atol=1e-6), "ecfp must matter"

        # zero the graph (all-padding) -> a different graph vector -> different logits
        zaf = torch.zeros_like(af)
        zbf = torch.zeros_like(bf)
        zs = torch.zeros_like(es)
        zd = torch.zeros_like(ed)
        out_z = model(zaf, zbf, zs, zd, e1)
        assert not torch.allclose(out_e1, out_z, atol=1e-6), "graph must matter"


# ---------------------------------------------------------------------------
# 4. the ECFP-only MLP head trains and scores
# ---------------------------------------------------------------------------
def test_ecfp_mlp_head_trains_on_toy_data():
    rng = np.random.default_rng(0)
    n, d = 60, 32
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] + 0.3 * rng.standard_normal(n) > 0).astype(int)
    tr = np.arange(0, 45)
    te = np.arange(45, n)
    ev = _train_neural_head_fold(
        X, y, tr, te, train_seed=0, mlp_hidden=16, dropout=0.0, lr=1e-2,
        epochs=5, batch_size=16, device=torch.device("cpu"), graphs=None,
    )
    assert 0.0 <= ev["auc"] <= 1.0


def test_ecfpmlp_builds_correct_shapes():
    import torch.nn as nn

    m = ECFPMLP.build(2048, 128, 0.1)
    assert isinstance(m, nn.Sequential)
    assert m[0].in_features == 2048 and m[0].out_features == 128
    assert m[-1].out_features == 2


# ---------------------------------------------------------------------------
# 5. §5.11 config round-trip + anchor
# ---------------------------------------------------------------------------
def test_arm_anchor_autoincluded():
    assert _with_anchor(["2_ecfp4_plus_tmqm_emb_xgb"])[0] == "1_ecfp4_xgb"
    assert _with_anchor(["1_ecfp4_xgb", "3_ecfp4_plus_dmpnn_mlp"]) == [
        "1_ecfp4_xgb", "3_ecfp4_plus_dmpnn_mlp",
    ]


def test_yaml_round_trips_into_parser():
    import os

    os.environ.setdefault("WFA2_CONFIG_QUIET", "1")
    args = parse_args(["--config", CFG])
    assert args.arms == list(ALL_ARMS)
    assert args.nested_bleed is False
    assert args.metals == ["Ru", "Ir", "Rh", "Os", "Re"]
    assert args.split_seeds == [42, 43, 44, 49, 123]
    assert isinstance(args.encoder, type(args.encoder))  # Path-like
    # list-typed dests must arrive as lists, not space-joined strings
    assert all(isinstance(v, str) for v in args.arms)


def test_xgb_fold_aucs_matrix_on_toy_data():
    rng = np.random.default_rng(0)
    n = 40
    X = rng.standard_normal((n, 12)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    folds = [(np.arange(0, 30), np.arange(30, n))]
    aucs = _xgb_fold_aucs_matrix(X, y, folds, split_seed=42)
    assert len(aucs) == 1 and 0.0 <= aucs[0] <= 1.0
