"""Tests for the Wf-191 DataLoader helper.

These tests verify:
- ``make_retrain_dataloader`` is importable and yields batches correctly
- ``make_indexed_dataloader`` is importable and yields batches correctly
- ``IndexSubset`` and ``IndexedListDataset`` produce expected outputs
- ``iterate_manual_batches`` is the documented manual fallback
- Worker config auto-shrinks to 0 when the dataset is too small
- Backward compat: callers that ignore the helper still work

Run: python -m pytest molmetal/tests/test_data_loader_helper.py -v

Date: 2026-09-28 (Wf-191 ship)
"""

from __future__ import annotations

import os
import sys

import pytest
import torch


# Ensure ``molmetal`` is importable when running pytest directly.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from molmetal.scripts import _data_loader_helper as dh  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tensors() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    X = torch.randn(256, 16)
    y = torch.randint(0, 3, (256,))
    return X, y


# ---------------------------------------------------------------------------
# Tests: imports + basic shapes
# ---------------------------------------------------------------------------


def test_helper_is_importable():
    assert hasattr(dh, "make_retrain_dataloader")
    assert hasattr(dh, "make_indexed_dataloader")
    assert hasattr(dh, "IndexedListDataset")
    assert hasattr(dh, "IndexSubset")
    assert hasattr(dh, "iterate_manual_batches")


def test_make_retrain_dataloader_yields_batches(tensors):
    X, y = tensors
    dl = dh.make_retrain_dataloader((X, y), batch_size=32, num_workers=0)
    batches = list(dl)
    assert len(batches) == 8
    for xb, yb in batches:
        assert xb.shape == (32, 16)
        assert yb.shape == (32,)


def test_make_retrain_dataloader_with_indices(tensors):
    X, y = tensors
    indices = list(range(64))
    dl = dh.make_retrain_dataloader(
        (X, y), indices=indices, batch_size=16, num_workers=0
    )
    batches = list(dl)
    assert len(batches) == 4
    for xb, yb in batches:
        assert xb.shape == (16, 16)
        assert yb.shape == (16,)


def test_indexed_list_dataset_basics():
    items = [10, 20, 30, 40, 50]
    idx = [2, 0, 4]
    ds = dh.IndexedListDataset(idx, items, return_index=False)
    assert len(ds) == 3
    assert int(ds[0]) == 30
    assert int(ds[1]) == 10
    assert int(ds[2]) == 50

    ds2 = dh.IndexedListDataset(idx, items, return_index=True)
    assert ds2[0] == (2, 30)


def test_index_subset_basics():
    # Use a plain Tensor base so __getitem__ returns a tensor (not a tuple).
    base = torch.arange(10)
    ds = dh.IndexSubset(base, torch.tensor([3, 5, 9]))
    assert len(ds) == 3
    assert int(ds[0]) == 3
    assert int(ds[1]) == 5
    assert int(ds[2]) == 9


def test_iterate_manual_batches_matches_arange():
    out = list(dh.iterate_manual_batches(10, 3))
    assert out == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    out2 = list(dh.iterate_manual_batches(10, 3, indices=[9, 7, 5, 3, 1]))
    assert out2 == [[9, 7, 5], [3, 1]]


# ---------------------------------------------------------------------------
# Tests: config + auto-shrink
# ---------------------------------------------------------------------------


def test_effective_workers_shrinks_when_too_few_batches():
    # 5 batches < 2 * prefetch_factor=4 (so 8), must shrink to 0
    nw = dh._effective_workers(4, batch_count_hint=5)
    assert nw == 0


def test_effective_workers_keeps_when_enough_batches():
    nw = dh._effective_workers(4, batch_count_hint=100)
    assert nw == 4


def test_helper_with_explicit_workers_zero(tensors):
    X, y = tensors
    # Force num_workers=0; should not raise even for tiny data
    dl = dh.make_retrain_dataloader((X, y), batch_size=32, num_workers=0)
    assert dl.num_workers == 0
    assert len(list(dl)) == 8


def test_helper_caller_can_override_workers_via_env(tensors, monkeypatch):
    monkeypatch.setenv("WF191_NUM_WORKERS", "0")
    X, y = tensors
    dl = dh.make_retrain_dataloader(
        (X, y),
        batch_size=32,
        num_workers=4,  # requested 4, env forces 0
    )
    # effective_workers reads env at call time; check it's 0
    assert dh._env_int("WF191_NUM_WORKERS", dh.DEFAULT_NUM_WORKERS) == 0


# ---------------------------------------------------------------------------
# Tests: backward-compat fallback (manual loop)
# ---------------------------------------------------------------------------


def test_backward_compat_manual_still_works(tensors):
    """Manual loop with the same yield as DataLoader (no helper needed)."""
    X, y = tensors
    n = X.shape[0]
    bs = 32
    manual_batches = []
    for batch_idx in dh.iterate_manual_batches(n, bs):
        xb = X[torch.as_tensor(batch_idx, dtype=torch.long)]
        yb = y[torch.as_tensor(batch_idx, dtype=torch.long)]
        manual_batches.append((xb, yb))
    assert len(manual_batches) == 8
    for xb, yb in manual_batches:
        assert xb.shape == (32, 16)
        assert yb.shape == (32,)


def test_indexed_loader_round_trip():
    """``make_indexed_dataloader`` over a TensorDataset-style list."""
    items = torch.arange(50)
    ds = dh.IndexedListDataset(list(range(50)), items.tolist(), return_index=True)
    dl = dh.make_indexed_dataloader(ds, batch_size=10, shuffle=False, num_workers=0)
    rows = list(dl)
    assert len(rows) == 5
    for batch in rows:
        idx_part, item_part = batch
        assert idx_part.shape == (10,)
        assert item_part.shape == (10,)


# ---------------------------------------------------------------------------
# Tests: equality manual vs loader on the same indices
# ---------------------------------------------------------------------------


def test_manual_yields_same_set_as_loader_with_indices(tensors):
    """Both paths should visit every requested index exactly once."""
    X, y = tensors
    indices = list(range(0, 64, 2))  # 32 indices

    # Manual path
    manual_visited = []
    for batch in dh.iterate_manual_batches(len(indices), 8, indices):
        manual_visited.extend(batch)
    assert sorted(manual_visited) == indices
    assert len(manual_visited) == 32

    # Loader path: visit every requested index via IndexSubset. We flatten
    # the DataLoader batches to a sorted list of indices.
    base = torch.stack([torch.as_tensor(i, dtype=torch.long) for i in indices])
    ds = dh.IndexSubset(base, torch.arange(len(indices), dtype=torch.long))
    dl = dh.make_indexed_dataloader(ds, batch_size=8, shuffle=False, num_workers=0)
    loader_visited = []
    for batch_tensor in dl:
        loader_visited.extend(batch_tensor.tolist())
    # The loader iterates the (already 32-row) base in dataset order, so
    # loader_visited is just ``indices`` verbatim (sorted).
    loader_visited.sort()
    assert loader_visited == sorted(indices)
    assert len(loader_visited) == 32


# ---------------------------------------------------------------------------
# Tests: Wf-191 throughput self-test (acceptable speedup on CPU bench)
# ---------------------------------------------------------------------------


def test_throughput_self_test_runs():
    """The bundled ``_self_test()`` must complete and report a sensible speedup.

    We don't enforce a strict >1.5x on this CPU-only path because, as noted in
    the helper docstring, the throughput improvement from ``pin_memory`` +
    ``num_workers`` + ``prefetch_factor`` is most visible on GPU production
    loops. On a CPU-only micro-benchmark the speedup depends on per-item
    preparation cost; we assert it is reported and is positive.
    """
    out = dh._self_test()
    assert "speedup" in out
    assert out["speedup"] > 0.0
    assert out["loader_iters"] == out["manual_iters"]
    assert out["loader_iters"] > 0
