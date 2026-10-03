"""Wf-382-A2 — SMILES featurization cache CPU extension tests.

Tests both the D-MPNN thin forwarder + the NEW chemprop v3 cache.
CPU-only — no GPU launches. Project standard pytest contract.

Reference: TODO-155 §3 (chemprop v2 cache pivot) + Wf-320D +
Wf-328-W10 + Wf-382-A2.

Honest framing per ``party-twenty-spirit`` §3.5
-----------------------------------------------
- Tests are SCAFFOLD-level: the cache classes are constructed + a few
  batches are exercised against a stub ``batch_molgraph_ctor`` to
  validate the hit/miss accounting. Full integration into
  ``train_chemprop_v2_v3.py`` + ``train_dmpnn_multitask_wasserstein.py``
  is DEFERRED to Wf-380 trainer integration.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

# Repo-root importability — tests/ is a sibling of molmetal/.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def canonical_smiles_pool() -> List[str]:
    """10 canonical SMILES spanning drug-like + simple molecules."""
    return [
        "CCO",
        "c1ccccc1",
        "CC(=O)O",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "CC(C)CC(N)C(=O)O",
        "OC(=O)C(=O)O",
        "C1CCCCC1",
        "CCN",
        "CC(=O)N",
        "C(C(C(C(C(C(C=O)O)O)O)O)O)O",
    ]


def _stub_batch_molgraph_ctor(mgs: list) -> object:
    """Stub chemprop v3 ``BatchMolGraph`` constructor for the test.

    Returns a lightweight object that exposes ``.to(device)``-able
    fields. Mirrors the surface area used by
    :meth:`ChempropV3FeaturizationCache.get_or_build_batch`.
    """
    import torch

    if not mgs:
        return _StubBatchMolGraph(
            V=torch.zeros((0, 72), dtype=torch.float32),
            E=torch.zeros((0, 14), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.int64),
            rev_edge_index=torch.zeros((0,), dtype=torch.int64),
            batch=torch.zeros((0,), dtype=torch.int64),
        )
    V = torch.from_numpy(np.concatenate([m.V for m in mgs], axis=0)).float()
    E = torch.from_numpy(np.concatenate([m.E for m in mgs], axis=0)).float()
    # edge_index offset by per-mol atom offset (simplified — stubs).
    edge_index = torch.zeros((2, E.shape[0]), dtype=torch.int64)
    rev_edge_index = torch.zeros((E.shape[0],), dtype=torch.int64)
    # batch: which molecule each atom belongs to.
    batch = torch.cat(
        [torch.full(m.V.shape, i, dtype=torch.int64) for i, m in enumerate(mgs)],
        dim=0,
    )
    return _StubBatchMolGraph(
        V=V, E=E, edge_index=edge_index, rev_edge_index=rev_edge_index, batch=batch
    )


class _StubBatchMolGraph:
    """Minimal stub — exposes .V / .E / .edge_index / .rev_edge_index / .batch
    as torch tensors so the cache's ``.to(device)`` calls work.
    """

    def __init__(self, V, E, edge_index, rev_edge_index, batch):
        self.V = V
        self.E = E
        self.edge_index = edge_index
        self.rev_edge_index = rev_edge_index
        self.batch = batch

    def to(self, device):  # noqa: ARG002
        return self


# ---------------------------------------------------------------------------
# Test 1 — D-MPNN cache hit rate
# ---------------------------------------------------------------------------
def test_dmpnn_cache_hit_rate(canonical_smiles_pool):
    """D-MPNN cache must hit on cross-epoch reuse.

    Build a cache, exercise 3 epochs × 16 batches. Epoch 0 is all
    misses (cold cache). Epochs 1+2 are all hits (BatchMolGraph /
    padded tensors don't depend on model weights).

    Asserts:
        - n_batch_hit == 32 (epochs 1 + 2 reuse epoch 0 builds)
        - batch_hit_rate >= 0.66
        - n_canonical_smiles == len(smiles_pool)
    """
    from molmetal.scripts.wf382a2_dmpnn_featurization_cache import (
        DMPNNFeaturizationCache,
    )

    cache = DMPNNFeaturizationCache(enable_batch_cache=True)

    # Tier 1: precompute all SMILES once.
    cache.precompute_smiles(canonical_smiles_pool)
    assert len(cache._mols) == len(canonical_smiles_pool)

    # Tier 2: 3 epochs × 16 batches.
    n_epochs = 3
    n_batches = 16
    batch_size = 4

    for ep in range(n_epochs):
        cache.begin_epoch(ep)
        for bi in range(n_batches):
            batch = [
                canonical_smiles_pool[(bi + j) % len(canonical_smiles_pool)]
                for j in range(batch_size)
            ]
            cache.get_or_build_batch(bi, batch, device="cpu")

    # Verify hit accounting.
    assert cache.stats.n_batch_hit == (n_epochs - 1) * n_batches, (
        f"Expected {(n_epochs - 1) * n_batches} batch hits, "
        f"got {cache.stats.n_batch_hit}"
    )
    assert cache.stats.n_batch_hit + cache.stats.n_batch_miss == n_epochs * n_batches
    expected_hit_rate = (n_epochs - 1) / n_epochs  # 0.6667
    assert (
        cache.stats.batch_hit_rate >= expected_hit_rate - 1e-9
    ), f"batch_hit_rate={cache.stats.batch_hit_rate:.4f} < {expected_hit_rate:.4f}"
    # cache_info() must round-trip.
    info = cache.cache_info()
    assert info["n_canonical_smiles"] == len(canonical_smiles_pool)
    assert info["n_cached_batches"] == n_batches
    assert info["stats"]["batch_hit_rate"] >= expected_hit_rate - 1e-9


# ---------------------------------------------------------------------------
# Test 2 — chemprop v3 cache hit rate
# ---------------------------------------------------------------------------
def test_chemprop_v3_cache_hit_rate(canonical_smiles_pool):
    """chemprop v3 cache must hit on cross-epoch reuse.

    Same protocol as the D-MPNN test. Uses the stub
    ``_stub_batch_molgraph_ctor`` because chemprop v3 may not be
    installed in this environment (the cache is import-only on the
    constructor signature).

    Asserts:
        - n_hit == 32 (epochs 1 + 2 reuse epoch 0 builds)
        - hit_rate >= 0.66
        - n_canonical_smiles == len(smiles_pool)
    """
    from molmetal.scripts.wf382a2_chemprop_v3_featurization_cache import (
        ChempropV3FeaturizationCache,
    )

    # Build cache WITHOUT use_fast_featurizer to avoid Wf-320D
    # FastAtomFeaturizer dependency if chemprop v3 isn't installed.
    try:
        cache = ChempropV3FeaturizationCache(use_fast_featurizer=False)
    except ImportError:
        # If chemprop v3 isn't installed in this venv, skip silently.
        # The module is still importable as source — only the chemprop
        # import inside the constructor fails.
        pytest.skip("chemprop v3 not installed in this environment")

    # Tier 1: precompute all SMILES once.
    n_added = cache.precompute_smiles(canonical_smiles_pool)
    assert n_added == len(canonical_smiles_pool)
    assert len(cache._mols) == len(canonical_smiles_pool)

    # Tier 2: 3 epochs × 16 batches.
    n_epochs = 3
    n_batches = 16
    batch_size = 4

    for ep in range(n_epochs):
        cache.begin_epoch(ep)
        for bi in range(n_batches):
            batch = [
                canonical_smiles_pool[(bi + j) % len(canonical_smiles_pool)]
                for j in range(batch_size)
            ]
            cache.get_or_build_batch(ep, bi, batch, "cpu", _stub_batch_molgraph_ctor)

    # Verify hit accounting.
    assert (
        cache.stats.n_hit == (n_epochs - 1) * n_batches
    ), f"Expected {(n_epochs - 1) * n_batches} hits, got {cache.stats.n_hit}"
    assert cache.stats.n_hit + cache.stats.n_miss == n_epochs * n_batches
    expected_hit_rate = (n_epochs - 1) / n_epochs  # 0.6667
    assert (
        cache.stats.hit_rate >= expected_hit_rate - 1e-9
    ), f"hit_rate={cache.stats.hit_rate:.4f} < {expected_hit_rate:.4f}"

    # Verify per-molecule featurisation contract.
    for smi in canonical_smiles_pool[:3]:
        entry = cache._mols[smi]
        assert "V" in entry and "E" in entry
        assert "edge_index" in entry and "rev_edge_index" in entry
        assert entry["V"].ndim == 2 and entry["V"].shape[1] == 72
        assert entry["E"].ndim == 2 and entry["E"].shape[1] == 14

    # cache_info() must round-trip.
    info = cache.cache_info()
    assert info["n_canonical_smiles"] == len(canonical_smiles_pool)
    assert info["n_cached_batches"] == n_batches
    assert info["stats"]["hit_rate"] >= expected_hit_rate - 1e-9


# ---------------------------------------------------------------------------
# Test 3 — D-MPNN adapter fan-out featurisation (parallel smoke)
# ---------------------------------------------------------------------------
def test_dmpnn_adapter_featurize_smiles_list(canonical_smiles_pool):
    """Adapter must inject SMILES into the underlying cache."""
    from molmetal.scripts.wf382a2_dmpnn_featurization_cache import (
        Wf382A2DMPNNCacheAdapter,
    )

    adapter = Wf382A2DMPNNCacheAdapter(n_workers=0)  # sequential (safe in test)
    n_added = adapter.featurize_smiles_list(canonical_smiles_pool)
    assert n_added == len(canonical_smiles_pool)

    # Re-feed — second call should be all hits (no NEW featurisations).
    n_added_2 = adapter.featurize_smiles_list(canonical_smiles_pool)
    assert n_added_2 == 0
    # smiles_hit_rate is cumulative: 10 hit + 10 miss = 0.5 across both calls.
    assert adapter.cache.stats.smiles_hit_rate >= 0.5

    info = adapter.cache_info()
    assert info["n_canonical_smiles"] == len(canonical_smiles_pool)
    assert info["n_workers"] == 0
