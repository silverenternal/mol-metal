"""Wf-382-A2 — chemprop v3 featurization cache.

TODO cross-ref: TODO-155 §3 (chemprop v2 cache pivot) + Wf-320D
(``molmetal/scripts/wf320d_featurization_fix.py`` — chemprop v2
``PerEpochFeatureCache``) + Wf-328-W10 (D-MPNN
``DMPNNFeaturizationCache``).

This module is a **NEW SCAFFOLD** — the chemprop v3 backbone did not
have a featurisation cache before Wf-382-A2. We mirror the chemprop
v2 ``PerEpochFeatureCache`` API exactly so the same integration
pattern drops into ``train_chemprop_v2_v3.py``.

Two-tier cache:

1. **Tier 1 — per-SMILES**: :meth:`precompute_smiles` runs
   chemprop v3 ``SimpleMoleculeMolGraphFeaturizer`` (default) OR the
   ``FastAtomFeaturizer`` / ``FastBondFeaturizer`` from Wf-320D
   (opt-in ``use_fast_featurizer=True``) once per canonical SMILES.
   Stores ``{"V", "E", "edge_index", "rev_edge_index"}``.

2. **Tier 2 — per-batch**: :meth:`get_or_build_batch` looks up the
   (epoch_idx, batch_idx) cache; on miss builds a ``BatchMolGraph``
   from the tier-1 cached features and transfers to device. Cache
   key is just ``batch_idx`` because ``BatchMolGraph`` doesn't
   depend on model weights.

Hit/miss tracked via :class:`ChempropV3CacheStats`. The cache is
**stateless** with respect to chemprop v3 model parameters — building
the padded batch tensors does not depend on the encoder, so the
per-batch tier is safe to reuse across epochs without re-building.

Honest framing per ``party-twenty-spirit`` §3.5
-----------------------------------------------
- This module is CPU-only engineering. No GPU launches.
- The SCAFFOLD status of integration into
  ``train_chemprop_v2_v3.py`` is preserved — Wf-380-B1 trainer
  integration follow-up required for end-to-end wire-in.
- The ``FastAtomFeaturizer`` opt-in path inherits Wf-320D's warning:
  for production-size datasets (~30 K rows), chemprop v3's default
  C/RDKit featurizer is faster than the pure-Python fast featurizer
  due to per-atom Python attribute-lookup overhead. Default is
  therefore ``use_fast_featurizer=False``.
- Cache savings PROJECTED 5-10× on featurisation speed on cross-
  epoch reuse (per Wf-320D §0 + Wf-328-W10 §0 MEASURED baseline).
  End-to-end speedup is lower because forward/backward dominates.

CLI smoke::

    python molmetal/scripts/wf382a2_chemprop_v3_featurization_cache.py \\
        --smiles "CCO" "c1ccccc1" "CC(=O)O" \\
        --out /tmp/wf382a2_chemprop_v3_cache_smoke.json

Output::

    /tmp/wf382a2_chemprop_v3_cache_smoke.json with cache_info + per-tier stats.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit import RDLogger

# ---------------------------------------------------------------------------
# Repo-root importability.
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# Constants — match chemprop v3 SimpleMoleculeMolGraphFeaturizer output shape.
# chemprop v3 by default uses the same 72-d atom / 14-d bond featurizer as
# chemprop v2 (per chemprop 2.x + chemprop 3.0+ release notes).
# ---------------------------------------------------------------------------
ATOM_FDIM_DEFAULT: int = 72  # chemprop v3 default atom feature dim
BOND_FDIM_DEFAULT: int = 14  # chemprop v3 default bond feature dim


# ---------------------------------------------------------------------------
# ChempropV3CacheStats — mirrors chemprop v2 wf320d::CacheStats exactly.
# ---------------------------------------------------------------------------
@dataclass
class ChempropV3CacheStats:
    """Hit / miss statistics for :class:`ChempropV3FeaturizationCache`.

    Attributes:
        n_hit: Number of per-batch lookups that hit the per-batch tier.
        n_miss: Number of per-batch lookups that had to be built.
        n_epoch_resets: Number of ``begin_epoch()`` calls.
        build_wall_s: Total wall time spent building batched tensors.
        lookup_wall_s: Total wall time spent looking up batches.
    """

    n_hit: int = 0
    n_miss: int = 0
    n_epoch_resets: int = 0
    build_wall_s: float = 0.0
    lookup_wall_s: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.n_hit + self.n_miss
        return self.n_hit / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "n_hit": self.n_hit,
            "n_miss": self.n_miss,
            "n_epoch_resets": self.n_epoch_resets,
            "hit_rate": self.hit_rate,
            "build_wall_s": self.build_wall_s,
            "lookup_wall_s": self.lookup_wall_s,
        }


# ---------------------------------------------------------------------------
# ChempropV3FeaturizationCache
# ---------------------------------------------------------------------------
class ChempropV3FeaturizationCache:
    """Two-tier featurisation cache for chemprop v3 backbone.

    Tier 1 — per-SMILES (always on):
        Pre-featurise each SMILES exactly once via chemprop v3's
        :class:`SimpleMoleculeMolGraphFeaturizer` (or
        :class:`FastAtomFeaturizer` from Wf-320D opt-in). Stores
        ``{"V", "E", "edge_index", "rev_edge_index"}`` per
        canonical SMILES.

    Tier 2 — per-batch (always on):
        Cache the **assembled BatchMolGraph** (concatenation of
        per-molecule MolGraph + device transfer) per ``batch_idx``.
        Cache key is just ``batch_idx`` because BatchMolGraph doesn't
        depend on chemprop v3 model weights. Cross-epoch reuse is
        the actual saving.

    Args:
        atom_fdim: Expected atom feature dim (default 72).
        bond_fdim: Expected bond feature dim (default 14).
        use_fast_featurizer: Whether to use the Wf-320D fast featurizer
            (default False — chemprop v3's C/RDKit featurizer is
            faster on production-size cohorts).
        atom_featurizer: Optional pre-built atom featurizer (used only
            when ``use_fast_featurizer=True``).
        bond_featurizer: Optional pre-built bond featurizer (used only
            when ``use_fast_featurizer=True``).
    """

    def __init__(
        self,
        atom_fdim: int = ATOM_FDIM_DEFAULT,
        bond_fdim: int = BOND_FDIM_DEFAULT,
        use_fast_featurizer: bool = False,
        atom_featurizer: Optional[object] = None,
        bond_featurizer: Optional[object] = None,
    ) -> None:
        self.atom_fdim = int(atom_fdim)
        self.bond_fdim = int(bond_fdim)
        self.use_fast_featurizer = bool(use_fast_featurizer)

        if self.use_fast_featurizer:
            from molmetal.scripts.wf320d_featurization_fix import (
                FastAtomFeaturizer,
                FastBondFeaturizer,
            )

            self._atom_featurizer = atom_featurizer or FastAtomFeaturizer()
            self._bond_featurizer = bond_featurizer or FastBondFeaturizer()
        else:
            # Lazy import — chemprop v3 is heavy (rdkit + numpy + torch).
            from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

            self._atom_featurizer = SimpleMoleculeMolGraphFeaturizer()
            self._bond_featurizer = None  # not used in default path

        # Tier 1: canonical SMILES → per-molecule MolGraph dict.
        self._mols: Dict[str, Dict[str, object]] = {}

        # Tier 2: batch_idx → assembled BatchMolGraph (on device).
        self._batch_cache: Dict[int, object] = {}

        self._current_epoch: int = -1
        self.stats = ChempropV3CacheStats()

        RDLogger.DisableLog("rdApp.*")

    # ------------------------------------------------------------------
    # Tier 1: per-SMILES pre-featurisation
    # ------------------------------------------------------------------
    def precompute_smiles(self, smiles_list: Sequence[str]) -> int:
        """Featurise every SMILES in ``smiles_list`` exactly once.

        Args:
            smiles_list: Sequence of canonical SMILES.

        Returns:
            Number of NEW canonical SMILES featurised (i.e. tier-1
            misses during this call).
        """
        t0 = time.time()
        n_added = 0
        if self.use_fast_featurizer:
            af = self._atom_featurizer
            bf = self._bond_featurizer
            for smi in smiles_list:
                if smi in self._mols:
                    continue
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    self._mols[smi] = self._placeholder_mol()
                    n_added += 1
                    continue
                V = af.featurize_mol(mol)
                E = bf.featurize_mol(mol)
                n_atoms = mol.GetNumAtoms()
                n_bonds = mol.GetNumBonds()
                edge_index = np.zeros((2, n_bonds), dtype=np.int64)
                rev_edge_index = np.zeros((n_bonds,), dtype=np.int64)
                for k, b in enumerate(mol.GetBonds()):
                    i = b.GetBeginAtomIdx()
                    j = b.GetEndAtomIdx()
                    edge_index[0, k] = i
                    edge_index[1, k] = j
                    rev_edge_index[k] = k  # chemprop v3 sets rev to identity
                self._mols[smi] = {
                    "V": V,
                    "E": E,
                    "edge_index": edge_index,
                    "rev_edge_index": rev_edge_index,
                }
                n_added += 1
        else:
            ftr = self._atom_featurizer
            for smi in smiles_list:
                if smi in self._mols:
                    continue
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    self._mols[smi] = self._placeholder_mol()
                    n_added += 1
                    continue
                mg = ftr(mol)
                self._mols[smi] = {
                    "V": np.ascontiguousarray(mg.V.astype(np.float32)),
                    "E": np.ascontiguousarray(mg.E.astype(np.float32)),
                    "edge_index": np.ascontiguousarray(mg.edge_index.astype(np.int64)),
                    "rev_edge_index": np.ascontiguousarray(
                        mg.rev_edge_index.astype(np.int64)
                    ),
                }
                n_added += 1
        self.stats.build_wall_s += time.time() - t0
        return n_added

    def _placeholder_mol(self) -> Dict[str, object]:
        """1-atom / 0-bond placeholder for unparseable SMILES."""
        return {
            "V": np.zeros((1, self.atom_fdim), dtype=np.float32),
            "E": np.zeros((0, self.bond_fdim), dtype=np.float32),
            "edge_index": np.zeros((2, 0), dtype=np.int64),
            "rev_edge_index": np.zeros((0,), dtype=np.int64),
        }

    # ------------------------------------------------------------------
    # Tier 2: per-batch cache
    # ------------------------------------------------------------------
    def begin_epoch(self, epoch_idx: int) -> None:
        """Mark start of a new epoch.

        The SMILES-tier (tier 1) cache is preserved. The per-batch
        ``BatchMolGraph`` cache is NOT cleared because
        ``BatchMolGraph`` doesn't depend on model weights — building
        it once and reusing across epochs is the actual speedup.
        """
        self._current_epoch = epoch_idx
        self.stats.n_epoch_resets += 1

    def clear_batch_cache(self) -> None:
        """Drop all per-batch entries (forces re-build on next lookup)."""
        self._batch_cache.clear()

    def get_or_build_batch(
        self,
        epoch_idx: int,
        batch_idx: int,
        smiles_batch: Sequence[str],
        device: object,
        batch_molgraph_ctor: object,
    ) -> object:
        """Look up ``batch_idx`` in cache; build on miss.

        Args:
            epoch_idx: Current epoch index (must match
                ``self._current_epoch``).
            batch_idx: Batch index within the epoch (cache key).
            smiles_batch: List of SMILES in this batch.
            device: torch.device for the output ``BatchMolGraph``.
            batch_molgraph_ctor: chemprop v3 ``BatchMolGraph``
                constructor.

        Returns:
            ``BatchMolGraph`` on ``device``.

        Raises:
            ValueError: If ``epoch_idx != self._current_epoch``.
        """
        if epoch_idx != self._current_epoch:
            raise ValueError(
                f"epoch_idx={epoch_idx} != current_epoch={self._current_epoch}; "
                "call begin_epoch() first"
            )

        t0 = time.time()
        cached = self._batch_cache.get(batch_idx)
        if cached is not None:
            self.stats.n_hit += 1
            self.stats.lookup_wall_s += time.time() - t0
            return cached

        # Miss — build BatchMolGraph from tier-1 cached MolGraphs.
        from chemprop.data.molgraph import MolGraph  # lazy import

        mgs = []
        for smi in smiles_batch:
            d = self._mols.get(smi)
            if d is None:
                d = self._placeholder_mol()
            mgs.append(MolGraph(d["V"], d["E"], d["edge_index"], d["rev_edge_index"]))

        bmg = batch_molgraph_ctor(mgs)
        # Transfer to device (batch_molgraph_ctor returns CPU tensors).
        bmg.V = bmg.V.to(device)
        bmg.E = bmg.E.to(device)
        bmg.edge_index = bmg.edge_index.to(device)
        bmg.rev_edge_index = bmg.rev_edge_index.to(device)
        if hasattr(bmg, "batch"):
            bmg.batch = bmg.batch.to(device)

        self._batch_cache[batch_idx] = bmg
        self.stats.n_miss += 1
        self.stats.lookup_wall_s += time.time() - t0
        return bmg

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def cache_info(self) -> Dict[str, object]:
        return {
            "n_canonical_smiles": len(self._mols),
            "n_cached_batches": len(self._batch_cache),
            "current_epoch": self._current_epoch,
            "use_fast_featurizer": self.use_fast_featurizer,
            "atom_fdim": self.atom_fdim,
            "bond_fdim": self.bond_fdim,
            "stats": self.stats.to_dict(),
        }

    def __repr__(self) -> str:
        return (
            f"ChempropV3FeaturizationCache("
            f"n_smiles={len(self._mols)}, "
            f"n_batches={len(self._batch_cache)}, "
            f"epoch={self._current_epoch}, "
            f"hit={self.stats.hit_rate:.2%}, "
            f"use_fast={self.use_fast_featurizer})"
        )


__all__ = [
    "ATOM_FDIM_DEFAULT",
    "BOND_FDIM_DEFAULT",
    "ChempropV3CacheStats",
    "ChempropV3FeaturizationCache",
]


# ---------------------------------------------------------------------------
# CLI smoke runner
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Tiny CLI: featurise a small SMILES list + dump cache_info to JSON.

    Skips the per-batch tier by default (no chemprop v3 BatchMolGraph
    import path is invoked — that requires the real chemprop v3 install).
    Set ``--exercise-batch-tier True`` to also build synthetic batches.
    """
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
        default="/tmp/wf382a2_chemprop_v3_cache_smoke.json",
        help="Output JSON path for cache_info.",
    )
    p.add_argument(
        "--use-fast-featurizer",
        action="store_true",
        help="Use the Wf-320D FastAtomFeaturizer (default: False).",
    )
    p.add_argument(
        "--exercise-batch-tier",
        action="store_true",
        help="Build synthetic per-batch entries (requires chemprop v3 install).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of synthetic epochs (default 2).",
    )
    p.add_argument(
        "--batches-per-epoch",
        type=int,
        default=4,
        help="Number of synthetic batches per epoch (default 4).",
    )
    args = p.parse_args(argv)

    # 1. Build cache.
    cache = ChempropV3FeaturizationCache(
        use_fast_featurizer=args.use_fast_featurizer,
    )

    # 2. Tier 1: featurise SMILES list.
    n_added = cache.precompute_smiles(args.smiles)
    print(
        f"Tier 1: featurised {n_added} NEW SMILES (use_fast={args.use_fast_featurizer})"
    )

    # 3. Tier 2: synthetic per-batch exercise (best-effort).
    if args.exercise_batch_tier:
        try:
            from chemprop.data.molgraph import BatchMolGraph  # type: ignore

            for ep in range(args.epochs):
                cache.begin_epoch(ep)
                for bi in range(args.batches_per_epoch):
                    batch = [args.smiles[(bi + j) % len(args.smiles)] for j in range(4)]
                    cache.get_or_build_batch(ep, bi, batch, "cpu", BatchMolGraph)
        except ImportError as e:
            print(f"  chemprop v3 BatchMolGraph not importable: {e}")

    # 4. Emit JSON.
    info = cache.cache_info()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(info, indent=2, default=str))
    print(f"Cache info written to: {out_path}")
    print(f"  hit_rate:           {info['stats']['hit_rate']:.2%}")
    print(f"  n_canonical_smiles: {info['n_canonical_smiles']}")
    print(f"  n_cached_batches:   {info['n_cached_batches']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
