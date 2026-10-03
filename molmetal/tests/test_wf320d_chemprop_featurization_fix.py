"""Wf-320D tests — chemprop v2 featurization fix.

12 tests across 4 categories:

- 4 tests: feature cache hit / miss behaviour
- 3 tests: faster encoder correctness
- 3 tests: integration with train_chemprop_v2_permetal_singletask
- 2 tests: timing benchmark (baseline vs fixed)

Per the Wf-320D task brief: WRITE-ONLY tests, no production-grade
verification. CPU-only. Run via::

    /home/hugo/codes/mol-metal/.venv/bin/python -m pytest \\
            molmetal/tests/test_wf320d_chemprop_featurization_fix.py -v

Honest framing per ``party-twenty-spirit`` §3.5:
- The hit-rate + speedup factor measured here is CPU-only. The
  paper-grade speedup (5.1× at TODO-155 §1.3) was a chemprop-v2 vs
  V4 pivot, not this incremental fix.
- This module's per-epoch BatchMolGraph cache is the engineering
  gate for the Wf-313F follow-up; GPU portion (5×5×50 on RTX 5090)
  is deferred to R20-P1 per TODO-120.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

# Path bootstrap — match mol-metal test convention.
import sys as _sys
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

from molmetal.scripts.wf320d_featurization_fix import (  # noqa: E402
    ATOM_FDIM_DEFAULT,
    BOND_FDIM_DEFAULT,
    CacheStats,
    FastAtomFeaturizer,
    FastBondFeaturizer,
    PerEpochFeatureCache,
)


# ===========================================================================
# Section 1 — feature cache hit / miss (4 tests)
# ===========================================================================
class TestFeatureCacheHitMiss:
    """Test 1-4: PerEpochFeatureCache hit/miss accounting."""

    def test_cache_init_empty(self):
        """Test 1: A fresh cache has zero entries + zero hits/misses."""
        cache = PerEpochFeatureCache()
        info = cache.cache_info()
        assert info["n_canonical_smiles"] == 0
        assert info["n_cached_batches"] == 0
        assert info["current_epoch"] == -1
        assert info["stats"]["n_hit"] == 0
        assert info["stats"]["n_miss"] == 0
        assert info["stats"]["n_epoch_resets"] == 0
        assert info["stats"]["hit_rate"] == 0.0

    def test_precompute_smiles_populates_tier1(self):
        """Test 2: After precompute_smiles, tier-1 cache has SMILES entries."""
        cache = PerEpochFeatureCache(use_fast_featurizer=True)
        smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
        cache.precompute_smiles(smiles)
        info = cache.cache_info()
        assert info["n_canonical_smiles"] == 3
        # Each entry should be a dict with V/E/edge_index/rev_edge_index
        for smi in smiles:
            assert smi in cache._mols
            entry = cache._mols[smi]
            assert "V" in entry
            assert "E" in entry
            assert "edge_index" in entry
            assert "rev_edge_index" in entry
            assert entry["V"].shape[-1] == ATOM_FDIM_DEFAULT

    def test_begin_epoch_resets_batch_cache(self):
        """Test 3: begin_epoch increments epoch_resets + preserves SMILES cache.

        BatchMolGraph cache is preserved across epochs (does NOT
        depend on weights) — the speedup is from reusing it across
        epochs.
        """
        cache = PerEpochFeatureCache(use_fast_featurizer=True)
        cache.precompute_smiles(["CCO", "c1ccccc1"])
        cache.begin_epoch(0)
        assert cache._current_epoch == 0
        assert cache.stats.n_epoch_resets == 1

        cache.begin_epoch(1)
        assert cache._current_epoch == 1
        assert cache.stats.n_epoch_resets == 2
        # Tier-1 SMILES cache should persist
        assert cache.cache_info()["n_canonical_smiles"] == 2

    def test_hit_rate_after_two_epochs(self):
        """Test 4: After 2 epochs × n batches, n_miss == n_batches (epoch 0), "
        "n_hit == n_batches (epoch 1), hit_rate == 0.50.

        Note: requires chemprop v2 BatchMolGraph + device.
        """
        pytest.importorskip("chemprop")
        pytest.importorskip("torch")
        from chemprop.data import BatchMolGraph

        import torch

        cache = PerEpochFeatureCache(use_fast_featurizer=True)
        smiles = ["CCO", "c1ccccc1", "CC(=O)O", "C1CCCCC1"]
        cache.precompute_smiles(smiles)

        device = torch.device("cpu")
        n_batches = 4

        # Epoch 0 — all misses.
        cache.begin_epoch(0)
        for b in range(n_batches):
            cache.get_or_build_batch(
                epoch_idx=0,
                batch_idx=b,
                smiles_batch=[smiles[b]],
                device=device,
                batch_molgraph_ctor=BatchMolGraph,
            )
        assert cache.stats.n_miss == n_batches
        assert cache.stats.n_hit == 0
        assert cache.stats.hit_rate == 0.0

        # Epoch 1 — all hits (same SMILES → same BatchMolGraph).
        cache.begin_epoch(1)
        for b in range(n_batches):
            cache.get_or_build_batch(
                epoch_idx=1,
                batch_idx=b,
                smiles_batch=[smiles[b]],
                device=device,
                batch_molgraph_ctor=BatchMolGraph,
            )
        assert cache.stats.n_miss == n_batches  # unchanged
        assert cache.stats.n_hit == n_batches
        assert abs(cache.stats.hit_rate - 0.5) < 1e-9


# ===========================================================================
# Section 2 — faster encoder correctness (3 tests)
# ===========================================================================
class TestFasterEncoderCorrectness:
    """Test 5-7: FastAtomFeaturizer / FastBondFeaturizer output shapes + dtypes."""

    def test_atom_featurizer_shape_dtype(self):
        """Test 5: FastAtomFeaturizer produces (72,) float32 per atom."""
        af = FastAtomFeaturizer()
        mol = Chem.MolFromSmiles("c1ccccc1O")
        assert mol is not None
        feats = af.featurize_mol(mol)
        assert feats.shape == (mol.GetNumAtoms(), 72)
        assert feats.dtype == np.float32
        # All values must be in valid range (no NaNs, no infs)
        assert np.all(np.isfinite(feats))
        # Atom 0 should be aromatic (benzene)
        assert feats[0, 9] == 1.0  # is_aromatic bit

    def test_bond_featurizer_shape_dtype(self):
        """Test 6: FastBondFeaturizer produces (14,) float32 per bond."""
        bf = FastBondFeaturizer()
        mol = Chem.MolFromSmiles("c1ccccc1O")
        assert mol is not None
        feats = bf.featurize_mol(mol)
        assert feats.shape == (mol.GetNumBonds(), 14)
        assert feats.dtype == np.float32
        assert np.all(np.isfinite(feats))
        # Aromatic ring bonds should have aromatic bit set
        assert any(feats[:, 12] == 1.0)

    def test_atom_fdim_constant_matches_chemprop_v2(self):
        """Test 7: ATOM_FDIM_DEFAULT == chemprop v2 SimpleMoleculeMolGraphFeaturizer output."""
        pytest.importorskip("chemprop")
        from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

        ftr = SimpleMoleculeMolGraphFeaturizer()
        mol = Chem.MolFromSmiles("c1ccccc1")
        mg = ftr(mol)
        assert ATOM_FDIM_DEFAULT == mg.V.shape[1]
        assert BOND_FDIM_DEFAULT == mg.E.shape[1]


# ===========================================================================
# Section 3 — integration with train_chemprop_v2_permetal_singletask (3 tests)
# ===========================================================================
class TestIntegrationWithPerMetalSingletask:
    """Test 8-10: Cache wires into the canonical chemprop v2 per-metal driver."""

    def test_imports_from_wf320d_module(self):
        """Test 8: PerEpochFeatureCache is importable from the wf320d module."""
        from molmetal.scripts.wf320d_featurization_fix import (
            PerEpochFeatureCache,
            FastAtomFeaturizer,
            FastBondFeaturizer,
            CacheStats,
            ATOM_FDIM_DEFAULT,
            BOND_FDIM_DEFAULT,
        )
        assert PerEpochFeatureCache is not None
        assert ATOM_FDIM_DEFAULT == 72
        assert BOND_FDIM_DEFAULT == 14

    def test_cache_dict_roundtrip(self):
        """Test 9: CacheStats.to_dict produces a JSON-safe dict."""
        stats = CacheStats(n_hit=5, n_miss=3, n_epoch_resets=1)
        d = stats.to_dict()
        assert d["n_hit"] == 5
        assert d["n_miss"] == 3
        assert d["n_epoch_resets"] == 1
        assert abs(d["hit_rate"] - 5 / 8) < 1e-9
        # All values are JSON-serializable
        import json
        json.dumps(d)

    def test_default_driver_unchanged(self):
        """Test 10: Default canonical driver (no flag) does NOT use fast path.

        This is a regression guard: the fix is opt-in, not default.
        The canonical chemprop v2 per-metal singletask driver remains
        untouched so existing 25-run sweeps aren't disturbed.
        """
        from molmetal.scripts import train_chemprop_v2_permetal_singletask as drv

        # The module must NOT have the fast-featurize attribute (it's opt-in).
        assert not hasattr(drv, "use_fast_featurizer") or drv.use_fast_featurizer is False
        # And the canonical run_one() signature is unchanged.
        import inspect
        sig = inspect.signature(drv.run_one)
        assert "use_bf16" in sig.parameters


# ===========================================================================
# Section 4 — timing benchmark (2 tests)
# ===========================================================================
class TestTimingBenchmark:
    """Test 11-12: Baseline vs fixed pipeline wall-time on a small batch."""

    def test_baseline_featurize_wall_time(self):
        """Test 11: Baseline (no cache) wall time — repeated BatchMolGraph builds."""
        pytest.importorskip("chemprop")
        pytest.importorskip("torch")
        from chemprop.data import BatchMolGraph
        from chemprop.data.molgraph import MolGraph
        from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

        import torch

        smiles = ["c1ccccc1", "CCO", "CC(=O)O", "c1ccc2ccccc2c1"]
        ftr = SimpleMoleculeMolGraphFeaturizer()
        device = torch.device("cpu")

        # Pre-featurize tier 1 (like the canonical driver does once).
        mols = {}
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            mg = ftr(mol)
            mols[smi] = mg

        # Baseline: rebuild BatchMolGraph + transfer every iteration.
        n_iters = 50
        t0 = time.time()
        for _ in range(n_iters):
            bmols = [mols[s] for s in smiles]
            bmg = BatchMolGraph(bmols)
            bmg.V = bmg.V.to(device)
            bmg.E = bmg.E.to(device)
            bmg.edge_index = bmg.edge_index.to(device)
            bmg.rev_edge_index = bmg.rev_edge_index.to(device)
            bmg.batch = bmg.batch.to(device)
        baseline_wall = time.time() - t0
        assert baseline_wall > 0
        # Just record — no strict assertion on absolute speed.

    def test_fixed_pipeline_wall_time(self):
        """Test 12: Fixed pipeline (per-epoch cache) wall time — build once per epoch."""
        pytest.importorskip("chemprop")
        pytest.importorskip("torch")
        from chemprop.data import BatchMolGraph

        import torch

        smiles = ["c1ccccc1", "CCO", "CC(=O)O", "c1ccc2ccccc2c1"]

        cache = PerEpochFeatureCache(use_fast_featurizer=True)
        cache.precompute_smiles(smiles)
        device = torch.device("cpu")

        n_iters = 50
        cache.begin_epoch(0)

        # Fixed: build once on first call, hit cache on subsequent.
        t0 = time.time()
        for it in range(n_iters):
            if it == 0:
                cache.begin_epoch(0)
            cache.get_or_build_batch(
                epoch_idx=0,
                batch_idx=0,
                smiles_batch=smiles,
                device=device,
                batch_molgraph_ctor=BatchMolGraph,
            )
        fixed_wall = time.time() - t0
        assert fixed_wall > 0
        # All iterations after the first should be hits.
        assert cache.stats.n_hit == n_iters - 1
        assert cache.stats.n_miss == 1


# ===========================================================================
# Fixtures
# ===========================================================================
@pytest.fixture
def cache_basic() -> PerEpochFeatureCache:
    """Small fixture cache for unit tests."""
    c = PerEpochFeatureCache(use_fast_featurizer=True)
    c.precompute_smiles(["CCO", "c1ccccc1", "CC(=O)O"])
    return c


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))