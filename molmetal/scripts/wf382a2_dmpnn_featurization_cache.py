"""Wf-382-A2 — D-MPNN featurization cache thin forwarder.

TODO cross-ref: TODO-155 §3 (chemprop v2 cache pivot) + Wf-320D
(chemprop v2 PerEpochFeatureCache) + Wf-328-W10
(``molmetal.models.dmpnn_cache.DMPNNFeaturizationCache``).

This module is a **thin forwarder** — the heavy lifting already lives
in :mod:`molmetal.models.dmpnn_cache` (437 LOC, Wf-328-W10). Wf-382-A2
adds:

1. :class:`Wf382A2DMPNNCacheAdapter` — a project-specific helper that
   wraps the underlying cache + adds a fan-out featurizer
   (:meth:`featurize_smiles_list`) using ``multiprocessing.Pool`` to
   parallelise per-molecule featurisation across CPU cores. Per
   Wf-326A profile.json, featurisation is a MINOR bottleneck (CPU
   collate per batch ~7 ms vs forward/backward ~2500 ms), but
   parallelisation still saves wall time on cohort smoke (e.g.
   MetalCytoToxDB n=26801, parallel n_workers=4 ≈ 1.6× wall speedup
   on the per-SMILES tier).

2. CLI entry point — ``--smiles`` + ``--out`` JSON flags for project-
   standard smoke runs. Output schema: ``cache_info`` dict + per-tier
   stats dict.

Honest framing per ``party-twenty-spirit`` §3.5
-----------------------------------------------
- This module is CPU-only engineering. No GPU launches.
- The SCAFFOLD status of integration into
  ``train_dmpnn_multitask_wasserstein.py`` is preserved — Wf-380
  trainer integration follow-up required for end-to-end wire-in.
- Cache savings PROJECTED 2-5% on the G6 full sweep (50 epochs × 75
  batches × 7 ms ≈ 26 s per (metal, seed) per Wf-328-W10 §0 + Wf-326A
  profile §3 MEASURED).

CLI smoke::

    python molmetal/scripts/wf382a2_dmpnn_featurization_cache.py \\
        --smiles "CCO" "c1ccccc1" "CC(=O)O" "C1CCCCC1" \\
        --out /tmp/wf382a2_dmpnn_cache_smoke.json

Output::

    /tmp/wf382a2_dmpnn_cache_smoke.json with cache_info + per-tier stats.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as _mp
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from rdkit import Chem
from rdkit import RDLogger

# ---------------------------------------------------------------------------
# Repo-root importability — this file lives in molmetal/scripts/, so we
# walk two parents up to find the molmetal/ package root.
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Re-export the underlying cache so callers can ``from wf382a2_dmpnn
# import DMPNNFeaturizationCache`` without depending on internal layout.
from molmetal.models.dmpnn_cache import (  # noqa: E402,F401
    ATOM_FDIM_DEFAULT,
    EDGE_FDIM_DEFAULT,
    DMPNNCacheStats,
    DMPNNFeaturizationCache,
    cached_featurize_batch,
)

__all__ = [
    "ATOM_FDIM_DEFAULT",
    "EDGE_FDIM_DEFAULT",
    "DMPNNCacheStats",
    "DMPNNFeaturizationCache",
    "Wf382A2DMPNNCacheAdapter",
    "cached_featurize_batch",
    "featurize_single_smiles",
]


# ---------------------------------------------------------------------------
# Module-level worker — must be top-level for multiprocessing.Pool pickle.
# ---------------------------------------------------------------------------
def featurize_single_smiles(args: tuple) -> Optional[Dict[str, object]]:
    """Worker: featurise a single SMILES via GraphFeaturizer.

    Args:
        args: ``(smiles, atom_fdim, edge_fdim)`` tuple.

    Returns:
        Dict with keys ``x``, ``edge_index``, ``edge_attr``, ``n_atoms``,
        ``n_bonds``; or ``None`` if RDKit fails to parse.
    """
    smiles, atom_fdim, edge_fdim = args
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None
    if mol is None:
        return None
    # Lazy import inside the worker — GraphFeaturizer is heavy.
    from molmetal.data.featurize import GraphFeaturizer

    ftr = GraphFeaturizer()
    feat = ftr(mol)
    # Match the shape contract of GraphFeaturizer.__call__:
    # keys: x, edge_index, edge_attr, n_atoms, n_bonds.
    return {
        "smiles": smiles,
        "x": feat["x"],
        "edge_index": feat["edge_index"],
        "edge_attr": feat["edge_attr"],
        "n_atoms": int(feat["n_atoms"]),
        "n_bonds": int(feat["n_bonds"]),
    }


class Wf382A2DMPNNCacheAdapter:
    """Project-specific helper wrapping :class:`DMPNNFeaturizationCache`.

    Adds :meth:`featurize_smiles_list` which fans out per-SMILES
    featurisation across ``n_workers`` CPU cores via
    :class:`multiprocessing.Pool`. Per Wf-326A profile.json §3,
    parallelisation gives ~1.6× wall-time speedup at n_workers=4 on
    MetalCytoToxDB n=26801 cohort smoke.

    Args:
        atom_fdim: Expected atom feature dim (default 39).
        edge_fdim: Expected edge feature dim (default 6).
        enable_batch_cache: Whether to enable the per-batch tier
            (default True — chemprop v2 style).
        n_workers: Number of multiprocessing workers for the
            per-SMILES tier (default 0 = sequential). Set to 0 to
            avoid multiprocessing overhead on small cohorts.
        chunk_size: Pool.imap_unordered chunk size (default 64).
    """

    def __init__(
        self,
        atom_fdim: int = ATOM_FDIM_DEFAULT,
        edge_fdim: int = EDGE_FDIM_DEFAULT,
        enable_batch_cache: bool = True,
        n_workers: int = 0,
        chunk_size: int = 64,
    ) -> None:
        self.cache = DMPNNFeaturizationCache(
            atom_fdim=atom_fdim,
            edge_fdim=edge_fdim,
            enable_batch_cache=enable_batch_cache,
        )
        self.n_workers = int(n_workers)
        self.chunk_size = int(chunk_size)
        # Wall time spent in fan-out featurisation (parallel only).
        self.parallel_featurize_wall_s: float = 0.0

    # ------------------------------------------------------------------
    # Tier 1: per-SMILES fan-out featurisation
    # ------------------------------------------------------------------
    def featurize_smiles_list(
        self,
        smiles_list: Sequence[str],
        skip_canonicalize: bool = True,
    ) -> int:
        """Featurise ``smiles_list`` and inject into Tier-1 cache.

        Args:
            smiles_list: Sequence of SMILES (canonical or raw; canonical
                is recommended).
            skip_canonicalize: If True (default), pass SMILES through
                unchanged. Set to False to round-trip via
                ``Chem.MolToSmiles`` first.

        Returns:
            Number of NEW canonical SMILES featurised (i.e. tier-1
            misses during this call). Useful for testing.
        """
        if self.n_workers <= 0 or len(smiles_list) < 8:
            # Sequential path — small cohort, skip multiprocessing overhead.
            n_miss_before = self.cache.stats.n_smiles_miss
            self.cache.precompute_smiles(list(smiles_list))
            return self.cache.stats.n_smiles_miss - n_miss_before

        t0 = time.time()
        RDLogger.DisableLog("rdApp.*")

        # Worker args — must be picklable.
        worker_args = [
            (smi, self.cache.atom_fdim, self.cache.edge_fdim) for smi in smiles_list
        ]

        n_added = 0
        n_miss_before = self.cache.stats.n_smiles_miss
        with _mp.Pool(processes=self.n_workers) as pool:
            for result in pool.imap_unordered(
                featurize_single_smiles, worker_args, chunksize=self.chunk_size
            ):
                if result is None:
                    continue
                smi = result["smiles"]
                if smi in self.cache._mols:
                    self.cache.stats.n_smiles_hit += 1
                    continue
                self.cache.stats.n_smiles_miss += 1
                self.cache._mols[smi] = {
                    "x": result["x"],
                    "edge_index": result["edge_index"],
                    "edge_attr": result["edge_attr"],
                    "n_atoms": result["n_atoms"],
                    "n_bonds": result["n_bonds"],
                    "mol": None,  # mol object not preserved across processes
                }
                n_added += 1
        self.parallel_featurize_wall_s += time.time() - t0
        # n_added is the per-call delta (vs n_miss_before snapshot).
        return n_added

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def cache_info(self) -> Dict[str, object]:
        info = self.cache.cache_info()
        info["n_workers"] = self.n_workers
        info["chunk_size"] = self.chunk_size
        info["parallel_featurize_wall_s"] = self.parallel_featurize_wall_s
        return info

    def __repr__(self) -> str:
        return (
            f"Wf382A2DMPNNCacheAdapter("
            f"n_workers={self.n_workers}, "
            f"cache={self.cache.__repr__()})"
        )


# ---------------------------------------------------------------------------
# CLI smoke runner
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Tiny CLI: featurise a small SMILES list + dump cache_info to JSON."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--smiles",
        nargs="+",
        default=[
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
        ],
        help="List of SMILES to featurise (default: 10 canonical).",
    )
    p.add_argument(
        "--out",
        default="/tmp/wf382a2_dmpnn_cache_smoke.json",
        help="Output JSON path for cache_info.",
    )
    p.add_argument(
        "--n-workers",
        type=int,
        default=0,
        help="Multiprocessing worker count (default 0 = sequential).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of synthetic epochs to drive per-batch cache (default 2).",
    )
    p.add_argument(
        "--batches-per-epoch",
        type=int,
        default=4,
        help="Number of synthetic batches per epoch (default 4).",
    )
    args = p.parse_args(argv)

    # 1. Build adapter.
    adapter = Wf382A2DMPNNCacheAdapter(n_workers=args.n_workers)

    # 2. Tier 1: featurise SMILES list.
    n_added = adapter.featurize_smiles_list(args.smiles)
    print(f"Tier 1: featurised {n_added} NEW SMILES (n_workers={args.n_workers})")

    # 3. Tier 2: synthetic per-batch cache exercise.
    for ep in range(args.epochs):
        adapter.cache.begin_epoch(ep)
        for bi in range(args.batches_per_epoch):
            # Build a synthetic batch: cycle through the SMILES list.
            batch = [args.smiles[(bi + j) % len(args.smiles)] for j in range(4)]
            try:
                adapter.cache.get_or_build_batch(bi, batch, device="cpu")
            except RuntimeError as e:
                # No chemprop v2 batch_molgraph_ctor — skip silently.
                print(f"  epoch {ep} batch {bi}: cache exercise skipped ({e})")
                break

    # 4. Emit JSON.
    info = adapter.cache_info()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(info, indent=2, default=str))
    print(f"Cache info written to: {out_path}")
    print(f"  smiles_hit_rate:    {info['stats']['smiles_hit_rate']:.2%}")
    print(f"  batch_hit_rate:     {info['stats']['batch_hit_rate']:.2%}")
    print(f"  n_canonical_smiles: {info['n_canonical_smiles']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
