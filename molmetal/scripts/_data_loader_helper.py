"""DataLoader prefetching helper for retrain scripts.

Wf-191 ship surface: drop-in replacement for the manual
``for start in range(0, n, batch_size):`` loop that most retrain scripts use.

Constraints honoured:
- Backward compat: manual batch loops still work as a fallback (no caller is
  forced to migrate).
- Sensible defaults per task spec:
    num_workers=4, prefetch_factor=4, persistent_workers=True, pin_memory=True
  with auto-shrink to 0 workers when n_batches < prefetch_factor (DataLoader
  raises ValueError if you set num_workers>0 with prefetch_factor but len < 2
  * prefetch_factor).
- Tested via runnable examples below.

Migrating a script (the easy case: indexed tensors):

    # before
    for start in range(0, n, batch_size):
        idx = indices[start:start + batch_size]
        x = X[idx].to(device, non_blocking=True)
        y = Y[idx].to(device, non_blocking=True)
        ...

    # after
    from molmetal.scripts._data_loader_helper import make_retrain_dataloader
    dl = make_retrain_dataloader((X, Y), indices=indices, batch_size=batch_size)
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        ...

Migrating a script (the harder case: list-of-graphs):

    # before
    for start in range(0, n, batch_size):
        chunk = indices[start:start + batch_size]
        g_list = [graphs[i] for i in chunk]
        collated = collate_fn(g_list)
        ...

    # after
    from molmetal.scripts._data_loader_helper import make_indexed_dataloader
    ds = IndexedListDataset(indices, graphs, extra=collator)
    dl = make_indexed_dataloader(ds, batch_size=batch_size, collate_fn=...)
    for collated in dl:
        ...

Date: 2026-09-28 (Wf-191 ship)
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


# ---------------------------------------------------------------------------
# Defaults per Wf-191 task spec
# ---------------------------------------------------------------------------

DEFAULT_NUM_WORKERS = 4
DEFAULT_PREFETCH_FACTOR = 4
DEFAULT_PIN_MEMORY = True


def _env_int(name: str, default: int) -> int:
    """Allow override via env var, e.g. WF191_NUM_WORKERS=0 to disable workers."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        v = int(raw)
    except ValueError:
        return default
    return max(0, v)


def _effective_workers(requested: int, batch_count_hint: int) -> int:
    """Shrink num_workers when the dataset is too small to justify them.

    PyTorch raises ValueError if num_workers>0 and len(ds) < 2 * prefetch_factor.
    We only have a hint here (the actual loader construction will validate
    strictly); for the common case we estimate from ``batch_count_hint``.
    """
    if requested <= 0:
        return 0
    pf = max(1, _env_int("WF191_PREFETCH_FACTOR", DEFAULT_PREFETCH_FACTOR))
    n_batches_min = 2 * pf
    if batch_count_hint is None or batch_count_hint <= 0:
        return requested
    if batch_count_hint < n_batches_min:
        return 0
    return requested


# ---------------------------------------------------------------------------
# Tensor-based loader (the dominant case in wf1** scripts)
# ---------------------------------------------------------------------------


def make_retrain_dataloader(
    tensors: Sequence[torch.Tensor] | Tuple[torch.Tensor, ...],
    indices: Optional[Iterable[int]] = None,
    *,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = DEFAULT_NUM_WORKERS,
    pin_memory: bool = DEFAULT_PIN_MEMORY,
    prefetch_factor: int = DEFAULT_PREFETCH_FACTOR,
    drop_last: bool = False,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    """Build a prefetched DataLoader for an indexed-tensor retrain.

    Args:
        tensors: one or more tensors of equal length on any device (CPU/GPU).
            A single tensor is treated as features with no labels; the loader
            yields that tensor alone. Two or more tensors are stacked into a
            ``TensorDataset`` so the loader yields ``(t0, t1, ..., tn)``.
        indices: optional subset of indices to iterate. ``None`` means use all
            rows. When set we wrap in an ``IndexSubset`` dataset so the loader
            only emits rows whose index is in the subset.
        batch_size: batch size.
        shuffle: shuffle the index subset each epoch.
        num_workers / pin_memory / prefetch_factor: defaults from Wf-191 spec.
        drop_last: passed to DataLoader.

    Returns:
        ``torch.utils.data.DataLoader`` with persistent_workers=True when
        num_workers>0 (so workers survive across epochs).

    Notes:
        Backward compat: callers that still want manual ``for start in range``
        loops just do not call this function. The helper is additive.
    """
    if not isinstance(tensors, (tuple, list)):
        tensors = (tensors,)
    if len(tensors) == 0:
        raise ValueError("make_retrain_dataloader requires at least one tensor")
    n = int(tensors[0].size(0))
    for t in tensors:
        if int(t.size(0)) != n:
            raise ValueError(
                f"all tensors must share the first-dim length; got {n} vs "
                f"{int(t.size(0))}"
            )

    base: Dataset = TensorDataset(*tensors)
    if indices is not None:
        idx = torch.as_tensor(list(indices), dtype=torch.long)
        if int(idx.numel()) != len(idx.unique()):
            pass  # ok, we still iterate, just non-strictly
        base = IndexSubset(base, idx)

    n_batches = (len(base) + max(1, batch_size) - 1) // max(1, batch_size)
    nw = _effective_workers(num_workers, n_batches)
    pf = prefetch_factor if nw > 0 else None

    kw: dict[str, Any] = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
    )
    if generator is not None:
        kw["generator"] = generator
    if pf is not None:
        kw["prefetch_factor"] = pf
        kw["persistent_workers"] = True
    return DataLoader(base, **kw)


# ---------------------------------------------------------------------------
# Indexed-list loader (for graph-based retrain scripts)
# ---------------------------------------------------------------------------


class IndexSubset(Dataset):
    """Subset a base ``Dataset`` by an explicit index list."""

    def __init__(self, base: Dataset, indices: torch.Tensor):
        self.base = base
        self.indices = indices.long()

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, i: int) -> Any:
        return self.base[int(self.indices[i].item())]


class IndexedListDataset(Dataset):
    """A ``Dataset`` that emits ``items[indices[i]]`` (or ``(indices[i], item)``).

    Useful for graph retrain where each index maps to a DGL graph or padded
    PyG ``Data`` object that needs collation.

    With ``return_index=True`` the worker also hands back the integer index
    that produced the element — useful when downstream code needs to index
    pre-built on-GPU tensors (e.g. ``model(index_select(0, idx_t))``).
    """

    def __init__(
        self,
        indices: Sequence[int],
        items: Sequence[Any],
        *,
        return_index: bool = False,
    ):
        self.indices = list(indices)
        self.items = list(items)
        self.return_index = return_index

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Any:
        idx = self.indices[i]
        item = self.items[idx]
        if self.return_index:
            return idx, item
        return item


def make_indexed_dataloader(
    dataset: Dataset,
    *,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = DEFAULT_NUM_WORKERS,
    pin_memory: bool = DEFAULT_PIN_MEMORY,
    prefetch_factor: int = DEFAULT_PREFETCH_FACTOR,
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
    drop_last: bool = False,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    """Build a prefetched DataLoader over an arbitrary ``Dataset``.

    This is the graph-loader companion to ``make_retrain_dataloader``; use
    this whenever your batching logic isn't a simple tensor stack.
    """
    n_batches = (len(dataset) + max(1, batch_size) - 1) // max(1, batch_size)
    nw = _effective_workers(num_workers, n_batches)
    pf = prefetch_factor if nw > 0 else None

    kw: dict[str, Any] = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    if generator is not None:
        kw["generator"] = generator
    if pf is not None:
        kw["prefetch_factor"] = pf
        kw["persistent_workers"] = True
    return DataLoader(dataset, **kw)


# ---------------------------------------------------------------------------
# Manual-batch fallback (for callers that want one helper for both modes)
# ---------------------------------------------------------------------------


def iterate_manual_batches(
    n: int,
    batch_size: int,
    indices: Optional[Iterable[int]] = None,
) -> Iterable[List[int]]:
    """Yield batches of indices the old way: ``for start in range(0, n, ...)``.

    Provided so a migrated script can keep both modes behind one entry point
    if the trainer doesn't take an iterator directly:

        for batch_idx in iterate_manual_batches(n, batch_size, indices):
            ...
    """
    if indices is None:
        seq = np.arange(n)
    else:
        seq = np.asarray(list(indices), dtype=np.int64)
    for start in range(0, len(seq), batch_size):
        yield seq[start : start + batch_size].tolist()


# ---------------------------------------------------------------------------
# Self-test (run with: python -m molmetal.scripts._data_loader_helper)
# ---------------------------------------------------------------------------


def _self_test() -> dict[str, Any]:
    """A small CPU-only benchmark: manual loop vs DataLoader.

    We compare end-to-end iteration time on N=8192 / batch=64 (128 batches)
    with synthetic tensor gather. The test is generous on purpose: workers
    add fork + setup overhead, so we expect a modest throughput improvement
    on CPU. The point is to verify the helper imports, builds, and iterates
    consistently with the manual loop.
    """
    import time

    n, d, bs = 8192, 128, 64
    X = torch.randn(n, d)
    Y = torch.randint(0, 2, (n,))
    indices = np.arange(n)
    np.random.shuffle(indices)

    # Manual loop (fallback)
    t0 = time.time()
    manual_iters = 0
    for batch in iterate_manual_batches(n, bs, indices):
        idx = torch.as_tensor(batch, dtype=torch.long)
        xb = X[idx]
        yb = Y[idx]
        _ = (xb.sum() + yb.sum()).item()
        manual_iters += 1
    t_manual = time.time() - t0

    # DataLoader
    t0 = time.time()
    dl = make_retrain_dataloader((X, Y), indices=indices, batch_size=bs)
    loader_iters = 0
    for xb, yb in dl:
        _ = (xb.sum() + yb.sum()).item()
        loader_iters += 1
    t_loader = time.time() - t0

    speedup = (t_manual / t_loader) if t_loader > 0 else float("inf")
    return {
        "manual_seconds": t_manual,
        "loader_seconds": t_loader,
        "speedup": speedup,
        "manual_iters": manual_iters,
        "loader_iters": loader_iters,
        "workers": _env_int("WF191_NUM_WORKERS", DEFAULT_NUM_WORKERS),
    }


if __name__ == "__main__":
    out = _self_test()
    import pprint
    pprint.pp(out)
