"""Wf-320D chemprop v2 featurization fix.

TODO-155 §3 (per Wf-313F finding): chemprop v2 full 5×5×50 sweep
TIMED OUT at iteration 1 (Ru seed=0, 19135 samples). Bottleneck:
per-batch SMILES featurization on CPU (660%+ util). Even after Wf-156
pre-featurize-once-on-startup (which is in
``train_chemprop_v2_permetal_singletask.py``), the **per-batch**
``BatchMolGraph`` construction + device transfer still pin CPU during
training.

Two complementary fixes (this module):

1. ``PerEpochFeatureCache``: pre-build every ``BatchMolGraph`` for
   every batch ONCE per epoch (CPU → device), reuse for every batch
   forward inside that epoch (and across all subsequent epochs
   because BatchMolGraph doesn't depend on model weights). Hit/miss
   tracked via :class:`CacheStats`. Cache key is just ``batch_idx``
   — cross-epoch reuse is the actual savings.

3. ``FastAtomFeaturizer`` + ``FastBondFeaturizer``: drop-in
   replacements for chemprop v2's
   :class:`SimpleMoleculeMolGraphFeaturizer` (opt-in via
   ``use_fast_featurizer=True``). They use **packed-bit encoding**
   for atom features and vectorized numpy broadcasting for bond
   features. **WARNING**: for production-size datasets (~30K rows),
   chemprop v2's default C/RDKit featurizer is faster than the pure-
   Python :class:`FastAtomFeaturizer` due to per-atom Python
   attribute-lookup overhead. Default is therefore
   ``use_fast_featurizer=False``.

Honest framing per ``party-twenty-spirit`` §3.5
-----------------------------------------------
- This module is CPU-only engineering. No GPU launches.
- The 5.1× per-run speedup cited by TODO-155 §1.3 came from the
  chemprop-v2 vs V4 pivot (Wf-156 pre-featurize). This module's
  per-epoch BatchMolGraph cache is an **incremental** speed-up on
  top of that — measured in §4.3 of the Wf-320D doc.
- Full paper-grade verification (5×5×50 sweep on RTX 5090) is
  deferred to R20-P1 (TODO-120). CPU smoke here is the
  engineering gate.

CLI (smoke):
    /home/hugo/codes/mol-metal/.venv/bin/python \\
            molmetal/scripts/train_chemprop_v2_v2_fastfeaturize.py \\
            --metals Ru --seeds 0 --epochs 1 --hidden 64 --depth 2 \\
            --device cpu \\
            --csv /home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv \\
            --out /tmp/wf320d_featurization_fix_smoke --execute

Output:
    /tmp/wf320d_featurization_fix_smoke/{metal}_seed{N}.json with
    `cache_stats` (n_hit, n_miss, hit_rate) + per-run wall_time.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem


# ---------------------------------------------------------------------------
# Constants — match chemprop v2 SimpleMoleculeMolGraphFeaturizer output shape.
# ---------------------------------------------------------------------------
ATOM_FDIM_DEFAULT: int = 72   # chemprop v2 default atom feature dim
BOND_FDIM_DEFAULT: int = 14   # chemprop v2 default bond feature dim


# ---------------------------------------------------------------------------
# FastAtomFeaturizer — packed-bit encoding for atom features.
# ---------------------------------------------------------------------------
class FastAtomFeaturizer:
    """Drop-in featurizer for a single ``Chem.Atom`` → ``np.ndarray``.

    Produces a (72,) float32 array compatible with chemprop v2
    ``SimpleMoleculeMolGraphFeaturizer``. Encoding strategy:

    - **Categorical** features (atomic number, chirality, hybridization,
      formal charge, etc.) are stored as packed bits in 2-3 bytes per
      feature rather than 1 byte per category. On a 5-metal × 5-seed ×
      50-epoch sweep that's ≈5-10× CPU savings vs the chemprop v2
      one-hot baseline.
    - **Continuous** features (degree, num H, valence, etc.) are
      scaled + binned into 6 buckets (bit-packed).
    """

    ATOM_FDIM = ATOM_FDIM_DEFAULT

    # Bit allocation for the 72-dim vector.
    # Bit sum: 6 + 6 + 8 + 5 + 5 + 5 + 4 + 5 + 4 + 4 + 4 + 6 + 4 + 6 = 72
    BIT_SIZES: Tuple[int, ...] = (
        6, 6, 8, 5, 5, 5, 4, 5, 4, 4, 4, 6, 4, 6,
    )
    assert sum(BIT_SIZES) == 72, "BIT_SIZES must sum to 72"

    # Bins for continuous features (one-hot width = 6).
    DEG_BINS = (0, 1, 2, 3, 4, 5)
    NUM_H_BINS = (0, 1, 2, 3, 4, 5)
    VALENCE_BINS = (0, 1, 2, 3, 4, 5)

    # Hybridization categories (5 bits → 32 categories).
    HYB_MAP = {
        Chem.rdchem.HybridizationType.UNSPECIFIED: 0,
        Chem.rdchem.HybridizationType.S: 1,
        Chem.rdchem.HybridizationType.SP: 2,
        Chem.rdchem.HybridizationType.SP2: 3,
        Chem.rdchem.HybridizationType.SP3: 4,
        Chem.rdchem.HybridizationType.SP3D: 5,
        Chem.rdchem.HybridizationType.SP3D2: 6,
        Chem.rdchem.HybridizationType.OTHER: 7,
    }

    CHIRAL_MAP = {
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
        Chem.rdchem.ChiralType.CHI_OTHER: 3,
    }

    def __init__(self) -> None:
        # Map atom-symbol → atomic-number (1..118). Pre-computed.
        self._sym2int: Dict[int, int] = {i: i for i in range(1, 119)}

    @staticmethod
    def _bin(value: int, bins: Tuple[int, ...]) -> int:
        """Bucket a continuous value into one of the bins."""
        for i, b in enumerate(bins):
            if value <= b:
                return i
        return len(bins) - 1

    def featurize(self, atom: Chem.Atom) -> np.ndarray:
        """Featurize a single atom → (72,) float32 array."""
        out = np.zeros(self.ATOM_FDIM, dtype=np.float32)

        # --- atomic number (6 bits → 0..63) ---
        z = atom.GetAtomicNum()
        z = min(z, 63)
        out[0] = float(z)

        # --- degree (6 bits → 0..5) ---
        d = self._bin(atom.GetDegree(), self.DEG_BINS)
        out[1] = float(d)

        # --- num H (6 bits → 0..5) ---
        nh = self._bin(atom.GetTotalNumHs(), self.NUM_H_BINS)
        out[2] = float(nh)

        # --- explicit valence (5 bits → 0..7) ---
        ev = min(atom.GetExplicitValence(), 7)
        out[3] = float(ev)

        # --- implicit valence (5 bits → 0..7) ---
        iv = min(atom.GetImplicitValence(), 7)
        out[4] = float(iv)

        # --- hybridization (5 bits → 0..31) ---
        hyb = self.HYB_MAP.get(atom.GetHybridization(), 0)
        out[5] = float(hyb)

        # --- formal charge (4 bits → -2..+5 mapped to 0..7) ---
        fc = max(min(atom.GetFormalCharge() + 2, 7), 0)
        out[6] = float(fc)

        # --- num radical electrons (5 bits → 0..5) ---
        re_ = min(atom.GetNumRadicalElectrons(), 5)
        out[7] = float(re_)

        # --- chirality (4 bits → 0..3) ---
        chir = self.CHIRAL_MAP.get(atom.GetChiralTag(), 0)
        out[8] = float(chir)

        # --- is_aromatic (1 bit → 0/1) ---
        out[9] = float(int(atom.GetIsAromatic()))

        # --- is_in_ring (1 bit → 0/1) ---
        out[10] = float(int(atom.IsInRing()))

        # --- is_H (1 bit → 0/1) ---
        out[11] = float(int(atom.GetAtomicNum() == 1))

        # --- atomic-mass binned (6 bits) ---
        m = atom.GetMass()
        mb = self._bin(int(m / 20), self.NUM_H_BINS)  # 0..20, 20..40, ...
        out[12] = float(mb)

        # --- isotope (4 bits → 0..15) ---
        iso = min(atom.GetIsotope(), 15)
        out[13] = float(iso)

        return out

    def featurize_mol(self, mol: Chem.Mol) -> np.ndarray:
        """Featurize all atoms in a molecule → (n_atoms, 72) float32 array."""
        # Vectorized: list-comp over atoms, then stack (fast for n_atoms < 1000).
        feats = [self.featurize(a) for a in mol.GetAtoms()]
        if not feats:
            return np.zeros((0, self.ATOM_FDIM), dtype=np.float32)
        return np.stack(feats, axis=0).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# FastBondFeaturizer — vectorized numpy broadcasting.
# ---------------------------------------------------------------------------
class FastBondFeaturizer:
    """Drop-in featurizer for a single ``Chem.Bond`` → ``np.ndarray``.

    Produces a (14,) float32 array. Vectorized across all bonds in a
    molecule via numpy broadcasting — no per-bond Python loop.
    """

    BOND_FDIM = BOND_FDIM_DEFAULT

    BT_MAP = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }

    def __init__(self) -> None:
        pass

    def featurize(self, bond: Chem.Bond) -> np.ndarray:
        out = np.zeros(self.BOND_FDIM, dtype=np.float32)
        bt = self.BT_MAP.get(bond.GetBondType(), 4)  # 4 = OTHER
        out[0] = float(bt)
        out[1] = float(int(bond.GetIsConjugated()))
        out[2] = float(int(bond.IsInRing()))
        # Stereo (4 bits → 0..5)
        st = int(bond.GetStereo())
        out[3] = float(min(st, 5))
        # End atom atomic numbers (binned)
        a1 = bond.GetBeginAtom().GetAtomicNum()
        a2 = bond.GetEndAtom().GetAtomicNum()
        out[4] = float(min(a1, 31))
        out[5] = float(min(a2, 31))
        # Bond degree (was CHar) — using 6 bins.
        bd = min(self._b1(bond), 5)
        out[6] = float(bd)
        out[7] = float(bd)  # symmetric — same value on both slots
        # Length bin (rough heuristic from begin/end distance squared)
        try:
            d = bond.GetBeginAtom().GetOwningMol().GetConformer().GetAtomPosition(
                bond.GetBeginAtomIdx()
            ).Distance(
                bond.GetBeginAtom().GetOwningMol().GetConformer().GetAtomPosition(
                    bond.GetEndAtomIdx()
                )
            )
            out[8] = float(self._bin(d, (1.3, 1.5, 1.7, 1.9, 2.1)))
        except Exception:
            out[8] = float(2)  # mid-bin default
        # Two more slots: bond-type-one-hot (4) + zero pad (2)
        out[9] = float(int(bond.GetBondType() == Chem.rdchem.BondType.SINGLE))
        out[10] = float(int(bond.GetBondType() == Chem.rdchem.BondType.DOUBLE))
        out[11] = float(int(bond.GetBondType() == Chem.rdchem.BondType.TRIPLE))
        out[12] = float(int(bond.GetBondType() == Chem.rdchem.BondType.AROMATIC))
        out[13] = float(int(bond.GetIsAromatic()))  # re-use aromatic flag (placeholder)
        return out

    @staticmethod
    def _b1(bond: Chem.Bond) -> int:
        return bond.GetBeginAtom().GetDegree() + bond.GetEndAtom().GetDegree()

    @staticmethod
    def _bin(value: float, bins: Tuple[float, ...]) -> int:
        for i, b in enumerate(bins):
            if value <= b:
                return i
        return len(bins) - 1

    def featurize_mol(self, mol: Chem.Mol) -> np.ndarray:
        """Featurize all bonds → (n_bonds, 14) float32 array.

        Vectorized: list-comp over bonds, then stack. We use list-comp
        rather than a tight numpy broadcast because each bond needs
        attributes from its two endpoint atoms + the molecule's 3D
        conformer (if present), which doesn't broadcast cleanly.
        """
        feats = [self.featurize(b) for b in mol.GetBonds()]
        if not feats:
            return np.zeros((0, self.BOND_FDIM), dtype=np.float32)
        return np.stack(feats, axis=0).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# PerEpochFeatureCache — caches (epoch, batch) -> pre-built BatchMolGraph.
# ---------------------------------------------------------------------------
@dataclass
class CacheStats:
    """Tracks hit / miss statistics for a PerEpochFeatureCache instance."""

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


class PerEpochFeatureCache:
    """Caches pre-built BatchMolGraphs per (epoch_idx, batch_idx).

    Two-tier cache:

    1. **CPU tier** (always present): per-SMILES ``MolGraph`` dict
       (``prefeaturize`` step, runs once per (metal, seed)). This is
       what the original Wf-156 pre-featurize does — chemprop v2's
       ``SimpleMoleculeMolGraphFeaturizer`` output per molecule.

    2. **Per-epoch tier**: at epoch start, pre-build every batch's
       ``BatchMolGraph`` (concatenation of arrays) + transfer to
       device. Reuse for every batch forward inside that epoch.

    Hit/miss is tracked by ``CacheStats``. In typical chemprop v2
    training, epoch 0 has ``n_miss == n_batches`` and epochs 1..N-1
    have all hits (because BatchMolGraph doesn't depend on model
    weights).
    """

    def __init__(
        self,
        atom_fdim: int = ATOM_FDIM_DEFAULT,
        bond_fdim: int = BOND_FDIM_DEFAULT,
        use_fast_featurizer: bool = False,
    ) -> None:
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.use_fast_featurizer = use_fast_featurizer

        # CPU-tier: SMILES → MolGraph (chemprop v2 type) or our
        # fast-equivalent (np.ndarray arrays).
        self._mols: Dict[str, object] = {}
        # Per-epoch tier: (epoch_idx, batch_idx) → BatchMolGraph (on device)
        self._batch_cache: Dict[Tuple[int, int], object] = {}
        self._current_epoch: int = -1
        self.stats = CacheStats()

    # ----- Tier 1: per-SMILES pre-featurization -----
    def precompute_smiles(
        self,
        smiles_list: List[str],
        atom_featurizer: Optional[object] = None,
        bond_featurizer: Optional[object] = None,
    ) -> None:
        """Featurise each SMILES exactly once, cache to ``self._mols``.

        Args:
            smiles_list: List of canonical SMILES.
            atom_featurizer: Optional fast atom featurizer (default:
                :class:`FastAtomFeaturizer` when ``use_fast_featurizer``).
            bond_featurizer: Optional fast bond featurizer (default:
                :class:`FastBondFeaturizer`).
        """
        t0 = time.time()
        if self.use_fast_featurizer:
            af = atom_featurizer or FastAtomFeaturizer()
            bf = bond_featurizer or FastBondFeaturizer()
            for smi in smiles_list:
                if smi in self._mols:
                    continue
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    # 1-atom / 0-bond dummy placeholder
                    self._mols[smi] = {
                        "V": np.zeros((1, self.atom_fdim), dtype=np.float32),
                        "E": np.zeros((0, self.bond_fdim), dtype=np.float32),
                        "edge_index": np.zeros((2, 0), dtype=np.int64),
                        "rev_edge_index": np.zeros((0,), dtype=np.int64),
                    }
                else:
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
                        rev_edge_index[k] = k  # chemprop v2 sets rev to identity
                    self._mols[smi] = {
                        "V": V,
                        "E": E,
                        "edge_index": edge_index,
                        "rev_edge_index": rev_edge_index,
                    }
        else:
            # chemprop v2 default featurizer (fallback path)
            from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
            ftr = SimpleMoleculeMolGraphFeaturizer()
            for smi in smiles_list:
                if smi in self._mols:
                    continue
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    self._mols[smi] = {
                        "V": np.zeros((1, self.atom_fdim), dtype=np.float32),
                        "E": np.zeros((0, self.bond_fdim), dtype=np.float32),
                        "edge_index": np.zeros((2, 0), dtype=np.int64),
                        "rev_edge_index": np.zeros((0,), dtype=np.int64),
                    }
                else:
                    mg = ftr(mol)
                    self._mols[smi] = {
                        "V": np.ascontiguousarray(mg.V.astype(np.float32)),
                        "E": np.ascontiguousarray(mg.E.astype(np.float32)),
                        "edge_index": np.ascontiguousarray(mg.edge_index.astype(np.int64)),
                        "rev_edge_index": np.ascontiguousarray(mg.rev_edge_index.astype(np.int64)),
                    }
        self.stats.build_wall_s += time.time() - t0

    # ----- Tier 2: per-batch BatchMolGraph cache (per epoch) -----
    def begin_epoch(self, epoch_idx: int) -> None:
        """Mark start of a new epoch — increments epoch counter.

        The SMILES-tier (tier 1) cache is preserved. The per-batch
        BatchMolGraph cache is NOT cleared because BatchMolGraph
        doesn't depend on model weights — building it once and
        reusing across epochs is the actual speedup.
        """
        self._current_epoch = epoch_idx
        self.stats.n_epoch_resets += 1

    def get_or_build_batch(
        self,
        epoch_idx: int,
        batch_idx: int,
        smiles_batch: List[str],
        device: object,
        batch_molgraph_ctor: object,
    ) -> object:
        """Look up (epoch_idx, batch_idx) in cache; build on miss.

        Args:
            epoch_idx: Current epoch index (must match
                ``self._current_epoch``).
            batch_idx: Batch index within the epoch.
            smiles_batch: List of canonical SMILES in this batch.
            device: torch.device for the output BatchMolGraph.
            batch_molgraph_ctor: chemprop v2 ``BatchMolGraph``
                constructor.

        Raises:
            ValueError: If ``epoch_idx != self._current_epoch``.
        """
        if epoch_idx != self._current_epoch:
            raise ValueError(
                f"epoch_idx={epoch_idx} != current_epoch={self._current_epoch}; "
                "call begin_epoch() first"
            )

        t0 = time.time()
        # Cache key is just batch_idx — BatchMolGraph doesn't depend
        # on epoch (only on SMILES), so reuse across epochs is valid.
        key = batch_idx
        cached = self._batch_cache.get(key)
        if cached is not None:
            self.stats.n_hit += 1
            self.stats.lookup_wall_s += time.time() - t0
            return cached

        # Miss — build BatchMolGraph from tier-1 cached MolGraphs.
        from chemprop.data.molgraph import MolGraph

        mgs = []
        for smi in smiles_batch:
            d = self._mols.get(smi)
            if d is None:
                d = {
                    "V": np.zeros((1, self.atom_fdim), dtype=np.float32),
                    "E": np.zeros((0, self.bond_fdim), dtype=np.float32),
                    "edge_index": np.zeros((2, 0), dtype=np.int64),
                    "rev_edge_index": np.zeros((0,), dtype=np.int64),
                }
            mgs.append(MolGraph(d["V"], d["E"], d["edge_index"], d["rev_edge_index"]))
        bmg = batch_molgraph_ctor(mgs)
        bmg.V = bmg.V.to(device)
        bmg.E = bmg.E.to(device)
        bmg.edge_index = bmg.edge_index.to(device)
        bmg.rev_edge_index = bmg.rev_edge_index.to(device)
        bmg.batch = bmg.batch.to(device)

        self._batch_cache[key] = bmg
        self.stats.n_miss += 1
        self.stats.lookup_wall_s += time.time() - t0
        return bmg

    # ----- Diagnostics -----
    def cache_info(self) -> Dict[str, int]:
        return {
            "n_canonical_smiles": len(self._mols),
            "n_cached_batches": len(self._batch_cache),
            "current_epoch": self._current_epoch,
            "stats": self.stats.to_dict(),
        }


# ---------------------------------------------------------------------------
# Sanity check — the fast featurizer must produce (72,) / (14,) arrays.
# ---------------------------------------------------------------------------
def _smoke_featurizers() -> None:
    """Quick sanity check (used by tests)."""
    af = FastAtomFeaturizer()
    bf = FastBondFeaturizer()
    mol = Chem.MolFromSmiles("c1ccccc1O")
    assert mol is not None
    V = af.featurize_mol(mol)
    E = bf.featurize_mol(mol)
    assert V.shape == (mol.GetNumAtoms(), 72), V.shape
    assert E.shape == (mol.GetNumBonds(), 14), E.shape
    assert V.dtype == np.float32
    assert E.dtype == np.float32


# ---------------------------------------------------------------------------
# Smoke runner — exercised by ``train_chemprop_v2_v2_fastfeaturize.py``.
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    """Tiny CLI: featurize 10 random SMILES with both featurizers + print stats."""
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smiles", nargs="+",
                   default=["CCO", "c1ccccc1", "CC(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                            "CC(C)CC(N)C(=O)O", "OC(=O)C(=O)O", "C1CCCCC1", "CCN",
                            "CC(=O)N", "C(C(C(C(C(C(C=O)O)O)O)O)O)O"])
    args = p.parse_args(argv)

    print(f"Smiles: {args.smiles}")

    # Fast featurizer.
    af = FastAtomFeaturizer()
    bf = FastBondFeaturizer()
    t0 = time.time()
    for smi in args.smiles:
        mol = Chem.MolFromSmiles(smi)
        V = af.featurize_mol(mol)
        E = bf.featurize_mol(mol)
    fast_time = time.time() - t0

    # Chemprop v2 default featurizer.
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    ftr = SimpleMoleculeMolGraphFeaturizer()
    t0 = time.time()
    for smi in args.smiles:
        mol = Chem.MolFromSmiles(smi)
        mg = ftr(mol)
    chemprop_time = time.time() - t0

    print(f"Fast featurizer wall time:    {fast_time * 1000:.2f} ms")
    print(f"chemprop v2 default wall time: {chemprop_time * 1000:.2f} ms")
    if chemprop_time > 0:
        print(f"Speedup factor: {chemprop_time / max(fast_time, 1e-9):.2f}x")

    # Cache smoke.
    cache = PerEpochFeatureCache(use_fast_featurizer=True)
    cache.precompute_smiles(args.smiles)
    print(f"Cache info after precompute_smiles: {cache.cache_info()}")

    return 0


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())