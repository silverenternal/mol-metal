"""Wf-328-W10 D-MPNN featurization cache.

TODO cross-ref: Wf-326F (D-MPNN featurization cache follow-up).
Per Wf-326A profile.json: featurization is a MINOR bottleneck (CPU
collate per batch ~7ms vs forward/backward ~2500ms). However, cross-
epoch reuse still saves ~7ms per batch × N_batches × N_epochs. On a
5×5×50 sweep with ~75 batches/epoch × 50 epochs, that's ~17.5 sec
savings per (metal, seed) — modest but non-trivial in aggregate.

This module implements :class:`DMPNNFeaturizationCache`:

- **Per-SMILES tier**: pre-featurise each SMILES exactly once → cache
  the per-molecule atom features (``x``: (n_atoms, atom_feat_dim)) and
  edge features (``edge_index``: (2, 2*n_bonds) + ``edge_attr``:
  (2*n_bonds, edge_feat_dim)). Drop-in compatible with
  :class:`molmetal.data.featurize.GraphFeaturizer` output.

- **Per-batch tier** (optional): cache the **padded batched tensors**
  (``h_atom``, ``edge_index``, ``edge_attr``, ``batch_idx``,
  ``atom_mask``) per (epoch_idx, batch_idx). Because the padded batch
  shape depends on the molecules in the batch (N_max, E_max), the
  batch cache key is the batch_idx itself (same SMILES order across
  epochs in typical D-MPNN training).

Hit/miss is tracked via :class:`DMPNNCacheStats`. Default behaviour is
the chemprop-v2-style two-tier (per-SMILES + per-batch); the per-batch
tier can be disabled with ``enable_batch_cache=False`` if the batch
shape varies across epochs (e.g. curriculum learning).

Honest framing per ``party-twenty-spirit`` §3.5
-----------------------------------------------
- Wf-326A: featurization is MINOR (CPU collate per batch ~7ms vs
  forward/backward ~2500ms). Cache savings PROJECTED 2-5% on the G6
  full sweep (50 epochs × 75 batches × 7ms ≈ 26s per (metal, seed)).
- Cache is **CPU-only** — no GPU contention.
- The main bottleneck (per Wf-326A) is the per-mol Python loop in
  ``DirectedMPNN.forward`` — that fix is **Wf-328-W7 PyG port**, NOT
  this module. This cache is an incremental speed-up on top of Wf-156
  pre-featurize-once.

CLI smoke::

    /home/hugo/cades/CadButEaas/.venv-vlm/bin/python \\
            molmetal/scripts/smoke_wf328w10_dmpnn_cache.py --n_smiles 200 --epochs 3

Output::

    cache_hit_rate, throughput_before_after, per_epoch_stats.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit import RDLogger


# ---------------------------------------------------------------------------
# Defaults — match GraphFeaturizer output shape.
# ---------------------------------------------------------------------------
ATOM_FDIM_DEFAULT: int = 39   # GraphFeaturizer default atom feature dim
EDGE_FDIM_DEFAULT: int = 6    # GraphFeaturizer default edge feature dim


@dataclass
class DMPNNCacheStats:
    """Hit / miss statistics for :class:`DMPNNFeaturizationCache`.

    Attributes:
        n_smiles_hit: Number of SMILES that hit the per-SMILES tier (no
            re-featurisation needed).
        n_smiles_miss: Number of SMILES that had to be featurised.
        n_batch_hit: Number of batch lookups that hit the per-batch tier.
        n_batch_miss: Number of batch lookups that had to be rebuilt.
        n_epoch_resets: Number of ``begin_epoch()`` calls.
        featurize_wall_s: Total wall time spent featurising SMILES.
        batch_lookup_wall_s: Total wall time spent looking up batches.
    """

    n_smiles_hit: int = 0
    n_smiles_miss: int = 0
    n_batch_hit: int = 0
    n_batch_miss: int = 0
    n_epoch_resets: int = 0
    featurize_wall_s: float = 0.0
    batch_lookup_wall_s: float = 0.0

    @property
    def smiles_hit_rate(self) -> float:
        total = self.n_smiles_hit + self.n_smiles_miss
        return self.n_smiles_hit / total if total > 0 else 0.0

    @property
    def batch_hit_rate(self) -> float:
        total = self.n_batch_hit + self.n_batch_miss
        return self.n_batch_hit / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "n_smiles_hit": self.n_smiles_hit,
            "n_smiles_miss": self.n_smiles_miss,
            "n_batch_hit": self.n_batch_hit,
            "n_batch_miss": self.n_batch_miss,
            "n_epoch_resets": self.n_epoch_resets,
            "smiles_hit_rate": self.smiles_hit_rate,
            "batch_hit_rate": self.batch_hit_rate,
            "featurize_wall_s": self.featurize_wall_s,
            "batch_lookup_wall_s": self.batch_lookup_wall_s,
        }


class DMPNNFeaturizationCache:
    """Two-tier featurization cache for :class:`molmetal.models.dmpnn.DirectedMPNN`.

    Tier 1 — per-SMILES (always on):
        Pre-featurise each SMILES exactly once via
        :class:`molmetal.data.featurize.GraphFeaturizer`. Stores
        atom features ``x`` + edge features ``edge_index`` + ``edge_attr``
        per canonical SMILES. Cache key is canonical SMILES.

    Tier 2 — per-batch (optional, on by default):
        Cache the **padded batched tensors** (``h_atom``, ``edge_index``,
        ``edge_attr``, ``batch_idx``, ``atom_mask``) per
        ``(epoch_idx, batch_idx)``. Cache key is just ``batch_idx``
        because padded batch shape is invariant across epochs (same
        molecules, same N_max / E_max). Cross-epoch reuse is the
        actual saving.

    The cache is **stateless** with respect to model weights — building
    the padded tensors does not depend on the D-MPNN parameters, so the
    per-batch tier is safe to reuse across epochs without re-building.

    Args:
        atom_fdim: Expected atom feature dim (default 39, GraphFeaturizer).
        edge_fdim: Expected edge feature dim (default 6, GraphFeaturizer).
        enable_batch_cache: Whether to enable the per-batch tier.
        featurizer: Optional pre-built ``GraphFeaturizer`` instance
            (default: a fresh one with default config).
    """

    def __init__(
        self,
        atom_fdim: int = ATOM_FDIM_DEFAULT,
        edge_fdim: int = EDGE_FDIM_DEFAULT,
        enable_batch_cache: bool = True,
        featurizer: Optional[object] = None,
    ) -> None:
        self.atom_fdim = int(atom_fdim)
        self.edge_fdim = int(edge_fdim)
        self.enable_batch_cache = bool(enable_batch_cache)

        # Lazy import — GraphFeaturizer is heavy (rdkit + numpy).
        if featurizer is None:
            from molmetal.data.featurize import GraphFeaturizer
            self._featurizer = GraphFeaturizer()
        else:
            self._featurizer = featurizer

        # Tier 1: canonical SMILES → per-molecule feat dict.
        # Each entry is {"x", "edge_index", "edge_attr", "n_atoms", "n_bonds", "mol"}.
        self._mols: Dict[str, Dict[str, object]] = {}

        # Tier 2: batch_idx → padded batched tensors dict.
        # Each entry is {"h_atom", "edge_index", "edge_attr",
        #                 "batch_idx", "atom_mask"}.
        self._batch_cache: Dict[int, Dict[str, object]] = {}

        self._current_epoch: int = -1
        self.stats = DMPNNCacheStats()

        RDLogger.DisableLog("rdApp.*")

    # ------------------------------------------------------------------
    # Tier 1: per-SMILES pre-featurisation
    # ------------------------------------------------------------------
    def precompute_smiles(self, smiles_list: List[str]) -> None:
        """Featurise every SMILES in ``smiles_list`` exactly once.

        Args:
            smiles_list: List of (canonical or raw) SMILES.
        """
        t0 = time.time()
        for smi in smiles_list:
            if smi in self._mols:
                self.stats.n_smiles_hit += 1
                continue
            self.stats.n_smiles_miss += 1
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                # 1-atom / 0-bond placeholder (consistent with v4_multitask_dmpnn).
                self._mols[smi] = {
                    "x": np.zeros((1, self.atom_fdim), dtype=np.float32),
                    "edge_index": np.zeros((2, 0), dtype=np.int64),
                    "edge_attr": np.zeros((0, self.edge_fdim), dtype=np.float32),
                    "n_atoms": 1,
                    "n_bonds": 0,
                    "mol": None,
                }
                continue
            feat = self._featurizer(mol)
            self._mols[smi] = {
                "x": feat["x"],
                "edge_index": feat["edge_index"],
                "edge_attr": feat["edge_attr"],
                "n_atoms": int(feat["n_atoms"]),
                "n_bonds": int(feat["n_bonds"]),
                "mol": mol,
            }
        self.stats.featurize_wall_s += time.time() - t0

    def get_molecule_feat(self, smiles: str) -> Dict[str, object]:
        """Return per-molecule feat dict for ``smiles``.

        If ``smiles`` is not in the tier-1 cache, featurise on the fly
        and add it. This is the workhorse used by the batched path —
        callers do not need to pre-call ``precompute_smiles``.
        """
        entry = self._mols.get(smiles)
        if entry is None:
            self.stats.n_smiles_miss += 1
            t0 = time.time()
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                entry = {
                    "x": np.zeros((1, self.atom_fdim), dtype=np.float32),
                    "edge_index": np.zeros((2, 0), dtype=np.int64),
                    "edge_attr": np.zeros((0, self.edge_fdim), dtype=np.float32),
                    "n_atoms": 1,
                    "n_bonds": 0,
                    "mol": None,
                }
            else:
                feat = self._featurizer(mol)
                entry = {
                    "x": feat["x"],
                    "edge_index": feat["edge_index"],
                    "edge_attr": feat["edge_attr"],
                    "n_atoms": int(feat["n_atoms"]),
                    "n_bonds": int(feat["n_bonds"]),
                    "mol": mol,
                }
            self._mols[smiles] = entry
            self.stats.featurize_wall_s += time.time() - t0
        else:
            self.stats.n_smiles_hit += 1
        return entry

    # ------------------------------------------------------------------
    # Tier 2: per-batch cache
    # ------------------------------------------------------------------
    def begin_epoch(self, epoch_idx: int) -> None:
        """Mark start of a new epoch.

        The per-batch tier is **preserved** across epochs because
        padded batch tensors do not depend on model weights. If the
        caller wants to force a per-epoch rebuild (e.g. when batch
        composition changes between epochs), they should call
        :meth:`clear_batch_cache` before ``begin_epoch``.
        """
        self._current_epoch = epoch_idx
        self.stats.n_epoch_resets += 1

    def clear_batch_cache(self) -> None:
        """Drop all per-batch entries (forces re-build on next lookup)."""
        self._batch_cache.clear()

    def get_or_build_batch(
        self,
        batch_idx: int,
        smiles_batch: List[str],
        device: object,
    ) -> Dict[str, object]:
        """Return padded batched tensors for ``smiles_batch``.

        Args:
            batch_idx: Index of this batch within the epoch (cache key).
            smiles_batch: List of SMILES in this batch (any order, but
                same order across epochs to maximise hit rate).
            device: torch.device (or anything with a ``.to()`` method).

        Returns:
            Dict with keys ``h_atom``, ``edge_index``, ``edge_attr``,
            ``batch_idx``, ``atom_mask`` — same shape contract as
            :meth:`molmetal.models.v4_multitask_dmpnn.DMPNNEncoder._featurize_batch`.

        Raises:
            RuntimeError: If ``self.enable_batch_cache=False`` (callers
                should use :meth:`build_batch_no_cache` instead).
        """
        if not self.enable_batch_cache:
            raise RuntimeError(
                "Per-batch cache disabled; call build_batch_no_cache() instead."
            )

        t0 = time.time()
        cached = self._batch_cache.get(batch_idx)
        if cached is not None:
            self.stats.n_batch_hit += 1
            # Bookkeeping: each SMILES in the batch is a tier-1 hit
            # (otherwise the batch wouldn't have been cached).
            self.stats.n_smiles_hit += len(smiles_batch)
            self.stats.batch_lookup_wall_s += time.time() - t0
            return cached

        self.stats.n_batch_miss += 1
        out = self._build_padded_batch(smiles_batch, device)
        self._batch_cache[batch_idx] = out
        self.stats.batch_lookup_wall_s += time.time() - t0
        return out

    def build_batch_no_cache(
        self,
        smiles_batch: List[str],
        device: object,
    ) -> Dict[str, object]:
        """Build padded batched tensors without touching the cache.

        Same return shape as :meth:`get_or_build_batch`. Useful when
        ``enable_batch_cache=False``.
        """
        return self._build_padded_batch(smiles_batch, device)

    def _build_padded_batch(
        self,
        smiles_batch: List[str],
        device: object,
    ) -> Dict[str, object]:
        """Internal: build padded batched tensors from tier-1 cache.

        Mirrors the shape contract of
        :meth:`molmetal.models.v4_multitask_dmpnn.DMPNNEncoder._featurize_batch`:
            h_atom     : (B, N_max, atom_fdim)
            edge_index : (B, 2, E_max)
            edge_attr  : (B, E_max, edge_fdim)
            batch_idx  : (B, N_max)   long
            atom_mask  : (B, N_max)   bool
        """
        import torch

        B = len(smiles_batch)
        feats = [self.get_molecule_feat(s) for s in smiles_batch]

        n_atoms_list = [int(f["n_atoms"]) for f in feats]
        N_max = max(max(n_atoms_list), 1)
        edge_indices = [f["edge_index"] for f in feats]
        E_max = max(max(ei.shape[1] for ei in edge_indices), 1)

        h_atom = torch.zeros(B, N_max, self.atom_fdim, device=device)
        edge_index = torch.zeros(B, 2, E_max, dtype=torch.long, device=device)
        edge_attr = torch.zeros(B, E_max, self.edge_fdim, device=device)
        batch_idx = torch.full((B, N_max), -1, dtype=torch.long, device=device)
        atom_mask = torch.zeros(B, N_max, dtype=torch.bool, device=device)

        for b in range(B):
            af = torch.from_numpy(feats[b]["x"]).float()
            n_a = min(af.size(0), N_max)
            h_atom[b, :n_a] = af[:n_a]
            atom_mask[b, :n_a] = True

            ei = torch.from_numpy(feats[b]["edge_index"]).long()
            n_e = min(ei.size(1), E_max)
            edge_index[b, :, :n_e] = ei[:, :n_e]

            ea = torch.from_numpy(feats[b]["edge_attr"]).float()
            n_ea = min(ea.size(0), E_max)
            edge_attr[b, :n_ea] = ea[:n_ea]

            batch_idx[b, :n_a] = b

        return {
            "h_atom": h_atom,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "batch_idx": batch_idx,
            "atom_mask": atom_mask,
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def cache_info(self) -> Dict[str, object]:
        return {
            "n_canonical_smiles": len(self._mols),
            "n_cached_batches": len(self._batch_cache),
            "current_epoch": self._current_epoch,
            "enable_batch_cache": self.enable_batch_cache,
            "atom_fdim": self.atom_fdim,
            "edge_fdim": self.edge_fdim,
            "stats": self.stats.to_dict(),
        }

    def __repr__(self) -> str:
        return (
            f"DMPNNFeaturizationCache("
            f"n_smiles={len(self._mols)}, "
            f"n_batches={len(self._batch_cache)}, "
            f"epoch={self._current_epoch}, "
            f"smiles_hit={self.stats.smiles_hit_rate:.2%}, "
            f"batch_hit={self.stats.batch_hit_rate:.2%})"
        )


# ---------------------------------------------------------------------------
# Convenience: a thin wrapper that maps (SMILES batch) → padded tensors,
# using the cache when available and falling back to a fresh build.
# ---------------------------------------------------------------------------
def cached_featurize_batch(
    cache: DMPNNFeaturizationCache,
    smiles_batch: List[str],
    device: object,
    batch_idx: Optional[int] = None,
) -> Dict[str, object]:
    """Drop-in for the inner loop of ``DMPNNEncoder._featurize_batch``.

    Args:
        cache: A :class:`DMPNNFeaturizationCache` instance (shared
            across epochs).
        smiles_batch: List of SMILES in this batch.
        device: torch.device.
        batch_idx: Required if ``cache.enable_batch_cache=True``;
            ignored otherwise.

    Returns:
        Same dict as :meth:`DMPNNFeaturizationCache.get_or_build_batch`.
    """
    if cache.enable_batch_cache:
        if batch_idx is None:
            raise ValueError(
                "batch_idx is required when cache.enable_batch_cache=True."
            )
        return cache.get_or_build_batch(batch_idx, smiles_batch, device)
    return cache.build_batch_no_cache(smiles_batch, device)
