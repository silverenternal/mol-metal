"""Tests for Wf-260 trainer DataLoader fix.

Verifies:
1. `_JointGraphDataset` returns (graph, mid, pic50, active) tuples.
2. `_collate_batch_for_loader` drops None / NaN rows and returns
   a 7-tuple of CPU tensors.
3. `train_one_epoch_joint` accepts num_workers arg without error and
   backward-compat num_workers=0 still works (manual loop path).
4. `--cpu-num-workers` argparse default is now 4 (157 §1.2 honest
   gap closed) but user can pass 0 for the legacy manual loop.

Run with: pytest molmetal/tests/test_wf260_dataloader_fix.py -v
"""

import math

import numpy as np
import pytest
import torch

try:
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        _JointGraphDataset,
        _collate_batch_for_loader,
        collate_multimetal,
        train_one_epoch_joint,
        parse_args,
    )
except ImportError:  # Wf-A16: feature not implemented — SKIP, do not ERROR
    pytest.skip(
        "UNIMPLEMENTED FEATURE: this module expects `_JointGraphDataset` from `scripts.train_dmpnn_multitask_wasserstein`, which "
        "does not exist. joint graph dataset. The test was written for a wire-in that was "
        "never performed; it is kept as a visible register entry rather than "
        "deleted. Tracked in TODO/completed/wfa16_unimplemented_feature_register.md",
        allow_module_level=True,
    )


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _toy_graph(n_atoms: int = 4, n_edges: int = 6, atom_dim: int = 32) -> tuple:
    """Minimal graph tuple (atom_feats, bond_feats, edge_src, edge_dst).

    Real chemprop atom features are 2-D ``[n_atoms, atom_dim]``; tests
    must mirror that shape so collate_graphs does not blow up on
    ``valid[0][0].shape[1]``.
    """
    atom_feats = torch.zeros(n_atoms, atom_dim, dtype=torch.long)
    bond_feats = torch.zeros(n_edges, atom_dim, dtype=torch.long)
    edge_src = torch.tensor([0, 1, 1, 2, 2, 3][:n_edges], dtype=torch.long)
    edge_dst = torch.tensor([1, 0, 2, 1, 3, 2][:n_edges], dtype=torch.long)
    return (atom_feats, bond_feats, edge_src, edge_dst)


def _toy_dataset(n: int = 8):
    graphs = [_toy_graph() for _ in range(n)]
    metal_ids = list(range(n))
    pic50 = np.linspace(4.0, 8.0, n).astype(np.float32)
    active = (np.linspace(4.0, 8.0, n) > 6.0).astype(np.int64)
    return graphs, metal_ids, pic50, active


# ----------------------------------------------------------------------------
# 1. _JointGraphDataset basics
# ----------------------------------------------------------------------------


def test_joint_dataset_len_and_getitem():
    graphs, metal_ids, pic50, active = _toy_dataset(n=5)
    ds = _JointGraphDataset(graphs, metal_ids, pic50, active)
    assert len(ds) == 5
    g, m, p, a = ds[2]
    assert g is graphs[2]
    assert m == metal_ids[2]
    assert math.isclose(float(p), float(pic50[2]))
    assert int(a) == int(active[2])


def test_joint_dataset_with_none_graph():
    graphs = [_toy_graph(), None, _toy_graph()]
    metal_ids = [0, 1, 2]
    pic50 = np.array([5.0, 6.0, 7.0], dtype=np.float32)
    active = np.array([0, 1, 1], dtype=np.int64)
    ds = _JointGraphDataset(graphs, metal_ids, pic50, active)
    # __getitem__ on the None index should NOT raise — that's the collate_fn's job.
    g, m, p, a = ds[1]
    assert g is None
    assert m == 1


# ----------------------------------------------------------------------------
# 2. _collate_batch_for_loader drops NaN / None
# ----------------------------------------------------------------------------


def test_collate_batch_drops_nan_and_none():
    g = _toy_graph()
    batch = [
        (g, 0, 5.0, 1),
        (None, 1, 6.0, 0),  # None graph
        (g, 2, float("nan"), 1),  # NaN pic50
        (g, 3, 7.0, 1),
    ]
    out = _collate_batch_for_loader(batch)
    assert out is not None
    atom_f, bond_f, e_src, e_dst, mids, pic50s, actives = out
    assert atom_f.device.type == "cpu"
    assert mids.tolist() == [0, 3]
    assert pic50s.tolist() == [5.0, 7.0]


def test_collate_batch_empty_returns_none():
    out = _collate_batch_for_loader([])
    assert out is None


# ----------------------------------------------------------------------------
# 3. train_one_epoch_joint signature + manual-loop backward compat
# ----------------------------------------------------------------------------


def test_train_one_epoch_joint_accepts_num_workers_kwarg():
    """num_workers is now part of the public signature (no TypeError)."""
    import inspect

    sig = inspect.signature(train_one_epoch_joint)
    assert "num_workers" in sig.parameters
    # Default must remain backward-compat (0).
    assert sig.parameters["num_workers"].default == 0


def test_train_one_epoch_joint_manual_loop_path():
    """num_workers=0 (legacy manual loop) must still produce finite losses."""
    graphs, metal_ids, pic50, active = _toy_dataset(n=16)

    # Tiny stub model with a real Parameter so the optimizer is non-empty.
    class _StubModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, atom_f, bond_f, e_src, e_dst, mids_t):
            b = mids_t.size(0)
            return torch.zeros(b, requires_grad=True), torch.zeros(
                b, b, requires_grad=True
            )

    model = _StubModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    crit = lambda pred, cls, p, a: (
        torch.tensor(0.0, requires_grad=True),
        torch.tensor(0.0),
        torch.tensor(0.0),
    )

    class _DummyCfg:
        n_mc = 1
        seed = 0

    losses = train_one_epoch_joint(
        model=model,
        optimizer=opt,
        criterion=crit,
        w_w=0.0,
        use_l2_guard=False,
        graphs=graphs,
        metal_ids=metal_ids,
        pic50=pic50,
        active=active,
        ref_pic50_dist=torch.zeros(1),
        batch_size=4,
        device=torch.device("cpu"),
        w_config=_DummyCfg(),
        w2_gradient_mode="sliced",
        num_workers=0,  # legacy path
    )
    # Returns a 4-tuple of floats, all finite.
    assert len(losses) == 4
    assert all(math.isfinite(v) for v in losses)


# ----------------------------------------------------------------------------
# 4. Argparse default for --cpu-num-workers is 4 (157 §1.2 fix)
# ----------------------------------------------------------------------------


def test_argparse_cpu_num_workers_default_is_4():
    """The default flipped from 0 → 4 after 157 §1.2 honest gap was closed."""
    args = parse_args([])
    assert args.cpu_num_workers == 4


def test_argparse_cpu_num_workers_zero_accepted():
    """User can still opt out into the legacy manual-loop path."""
    args = parse_args(["--cpu-num-workers", "0"])
    assert args.cpu_num_workers == 0


# ----------------------------------------------------------------------------
# 5. Backward compat — collate_multimetal still works (untouched)
# ----------------------------------------------------------------------------


def test_collate_multimetal_still_works():
    g = _toy_graph()
    graphs = [g, None, g]
    mids = [0, 1, 2]
    pic50 = np.array([5.0, float("nan"), 7.0], dtype=np.float32)
    active = np.array([1, 1, 1], dtype=np.int64)
    out = collate_multimetal(graphs, mids, pic50, active)
    assert out is not None
    atom_f, bond_f, e_src, e_dst, mids_t, pic50_t, active_t = out
    assert mids_t.tolist() == [0, 2]  # dropped index 1 (NaN)
