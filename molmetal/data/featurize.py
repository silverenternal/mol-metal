"""Featurisers — Morgan fingerprints + graph (atom/edge) features.

The framework targets two families of generative models:

* **Sequence / non-graph baselines** (MLP, Transformer, …) — Morgan
  fingerprints via :class:`MorganFingerprinter`.
* **Graph-conditioned models** (D-MPNN, EGNN, SchNet) — atom and edge
  features via :class:`GraphFeaturizer`.

The feature conventions follow the
[D-MPNN paper](https://pubs.rsc.org/en/content/articlelanding/2020/sc/c9sc03936j)
and the [PyG `torch_geometric.utils.from_rdmol` recipe](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/from_rdmol.html),
adapted so the output is pure ``numpy`` + Python dict (no torch-geometric
hard dependency).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 1. Morgan fingerprinter
# ---------------------------------------------------------------------------
class MorganFingerprinter:
    """Morgan / ECFP4 bit-vector fingerprint generator.

    Parameters
    ----------
    radius : int
        ECFP radius.  ``radius=2`` ≈ ECFP4 (default).
    n_bits : int
        Bit-vector length.  ``2048`` is the default (matches most published
        chemo-informatics baselines).
    use_features : bool
        If True, use RDKit feature invariants (atom symbol, degree, …) — i.e.
        ECFP *with* feature definitions; otherwise ECFP uses raw atom
        invariants only.
    """

    def __init__(
        self,
        radius: int = 2,
        n_bits: int = 2048,
        nBits: Optional[int] = None,  # alias for callers using camelCase
        use_features: bool = False,
    ) -> None:
        if radius < 0:
            raise ValueError(f"radius must be >= 0, got {radius}")
        # nBits alias for ergonomic interop with RDKit-style kwargs.
        if nBits is not None:
            n_bits = int(nBits)
        if n_bits < 1:
            raise ValueError(f"n_bits must be >= 1, got {n_bits}")
        self.radius = int(radius)
        self.n_bits = int(n_bits)
        self.use_features = bool(use_features)

    @property
    def dim(self) -> int:
        return self.n_bits

    # ------------------------------------------------------------------
    def __call__(self, smiles: str | List[str]) -> np.ndarray:
        """Featurise one or many SMILES.

        Parameters
        ----------
        smiles : str | list[str]

        Returns
        -------
        np.ndarray
            If input is a single SMILES: shape ``(n_bits,)`` uint8.
            If input is a list: shape ``(N, n_bits)`` uint8.
            For invalid SMILES, the row is all-zeros.
        """
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import AllChem

        RDLogger.DisableLog("rdApp.*")

        single = isinstance(smiles, str)
        s_list = [smiles] if single else list(smiles)
        out = np.zeros((len(s_list), self.n_bits), dtype=np.uint8)
        for i, s in enumerate(s_list):
            try:
                mol = Chem.MolFromSmiles(s)
                if mol is None:
                    continue
                if self.use_features:
                    fp = AllChem.GetMorganFingerprintAsBitVect(
                        mol, self.radius, nBits=self.n_bits, useFeatures=True
                    )
                else:
                    fp = AllChem.GetMorganFingerprintAsBitVect(
                        mol, self.radius, nBits=self.n_bits
                    )
                DataStructs.ConvertToNumpyArray(fp, out[i])
            except Exception:
                # leave row as zeros
                continue
        return out[0] if single else out

    def fingerprint_mol(self, mol) -> np.ndarray:
        """Featurise an RDKit ``Chem.Mol`` directly (skips SMILES parsing)."""
        from rdkit import DataStructs
        from rdkit.Chem import AllChem

        if self.use_features:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, self.radius, nBits=self.n_bits, useFeatures=True
            )
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, self.radius, nBits=self.n_bits
            )
        arr = np.zeros((self.n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr


# ---------------------------------------------------------------------------
# 2. Graph featurizer (atom + edge features)
# ---------------------------------------------------------------------------
# Atom-feature one-hot tables -------------------------------------------------
_ATOM_SYMBOLS = [
    "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B", "Si", "Se",
    "H", "Other",
]
_SYMBOL_TO_IDX = {s: i for i, s in enumerate(_ATOM_SYMBOLS)}

_HYBRIDIZATIONS = [
    "SP", "SP2", "SP3", "SP3D", "SP3D2", "UNSPECIFIED", "OTHER",
]
_HYB_TO_IDX = {h: i for i, h in enumerate(_HYBRIDIZATIONS)}

# Mapping RDKit HybridizationType → string table above
def _hybrid_to_str(h) -> str:
    name = h.name if hasattr(h, "name") else str(h)
    if name.startswith("SP"):
        return name
    if name == "UNSPECIFIED":
        return "UNSPECIFIED"
    return "OTHER"

# Bond-type one-hot table -----------------------------------------------------
_BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]


@dataclass
class GraphFeaturizerConfig:
    """Hyper-parameters for :class:`GraphFeaturizer`."""

    atom_feature_dim: int = 0  # computed automatically
    edge_feature_dim: int = 0  # computed automatically
    include_stereo: bool = False  # not needed for ESM / docking baselines


class GraphFeaturizer:
    """Convert RDKit ``Chem.Mol`` (or SMILES) into atom/edge tensors.

    Atom features (per atom, concatenated)::

        one_hot(symbol, len(_ATOM_SYMBOLS))   # Z-bin
        one_hot(degree, 6)                    # 0..5+
        one_hot(hybridization, len(_HYBRIDIZATIONS))
        one_hot(charge, 5)                    # -2,-1,0,1,2+
        in_ring                              # 1
        aromatic                              # 1
        num_hs                               # 0..4+
        chirality                            # 1 only if include_stereo

    Edge features (per undirected bond, both directions)::

        one_hot(bond_type, len(_BOND_TYPES))
        conjugated                           # 1
        in_ring                              # 1

    Output::

        ``feat(smiles)`` → dict with
        ``x``      (N_atoms, atom_feature_dim)  float32
        ``edge_index`` (2, 2*N_bonds) long
        ``edge_attr``   (2*N_bonds, edge_feature_dim) float32
    """

    def __init__(self, config: Optional[GraphFeaturizerConfig] = None) -> None:
        self.cfg = config or GraphFeaturizerConfig()
        # Compute static feature dims
        self._atom_dim = (
            len(_ATOM_SYMBOLS)
            + 6                   # degree
            + len(_HYBRIDIZATIONS)
            + 5                   # charge bucket
            + 1                   # in_ring
            + 1                   # aromatic
            + 5                   # num_hs bucket 0..4+
            + (1 if self.cfg.include_stereo else 0)
        )
        self._edge_dim = (
            len(_BOND_TYPES)
            + 1                   # conjugated
            + 1                   # in_ring
        )

    @property
    def atom_feature_dim(self) -> int:
        return self._atom_dim

    @property
    def edge_feature_dim(self) -> int:
        return self._edge_dim

    # ------------------------------------------------------------------
    def __call__(self, smiles_or_mol) -> Dict[str, np.ndarray]:
        from rdkit import Chem
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
        if isinstance(smiles_or_mol, str):
            mol = Chem.MolFromSmiles(smiles_or_mol)
            if mol is None:
                raise ValueError(f"RDKit could not parse SMILES: {smiles_or_mol!r}")
        else:
            mol = smiles_or_mol
        x = self._atom_features(mol)
        edge_index, edge_attr = self._edge_features(mol)
        return {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "smiles": Chem.MolToSmiles(mol) if mol.GetNumAtoms() > 0 else "",
            "n_atoms": int(mol.GetNumAtoms()),
            "n_bonds": int(edge_index.shape[1] // 2),
        }

    # ------------------------------------------------------------------
    # Per-atom encoding
    # ------------------------------------------------------------------
    def _atom_features(self, mol) -> np.ndarray:
        n = mol.GetNumAtoms()
        feats: List[np.ndarray] = []
        for i in range(n):
            a = mol.GetAtomWithIdx(i)
            f: List[np.ndarray] = []
            # Z-bin
            z = a.GetSymbol()
            sym = _SYMBOL_TO_IDX.get(z, len(_ATOM_SYMBOLS) - 1)
            oh = np.zeros(len(_ATOM_SYMBOLS), dtype=np.float32)
            oh[sym] = 1.0
            f.append(oh)
            # degree (0..5)
            deg = min(int(a.GetDegree()), 5)
            oh = np.zeros(6, dtype=np.float32)
            oh[deg] = 1.0
            f.append(oh)
            # hybridisation
            hyb_str = _hybrid_to_str(a.GetHybridization())
            hyb_idx = _HYB_TO_IDX.get(hyb_str, len(_HYBRIDIZATIONS) - 1)
            oh = np.zeros(len(_HYBRIDIZATIONS), dtype=np.float32)
            oh[hyb_idx] = 1.0
            f.append(oh)
            # formal charge bucket (-2,-1,0,1,2+)
            ch = int(a.GetFormalCharge())
            ch_idx = max(0, min(ch + 2, 4))
            oh = np.zeros(5, dtype=np.float32)
            oh[ch_idx] = 1.0
            f.append(oh)
            # in_ring, aromatic
            f.append(np.array([1.0 if a.IsInRing() else 0.0], dtype=np.float32))
            f.append(np.array([1.0 if a.GetIsAromatic() else 0.0], dtype=np.float32))
            # num Hs (0..4+)
            nhs = min(int(a.GetTotalNumHs()), 4)
            oh = np.zeros(5, dtype=np.float32)
            oh[nhs] = 1.0
            f.append(oh)
            # chirality (optional)
            if self.cfg.include_stereo:
                chir = 1.0 if str(a.GetChiralTag()) != "CHI_UNSPECIFIED" else 0.0
                f.append(np.array([chir], dtype=np.float32))
            feats.append(np.concatenate(f, axis=0))
        if not feats:
            return np.zeros((0, self._atom_dim), dtype=np.float32)
        return np.stack(feats, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Per-bond encoding (bidirectional)
    # ------------------------------------------------------------------
    def _edge_features(self, mol):
        src, dst, attrs = [], [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bt = bond.GetBondType().name if hasattr(bond.GetBondType(), "name") else str(bond.GetBondType())
            bt_idx = _BOND_TYPES.index(bt) if bt in _BOND_TYPES else 0
            conjugated = 1.0 if bond.GetIsConjugated() else 0.0
            in_ring = 1.0 if bond.IsInRing() else 0.0
            base = np.zeros(self._edge_dim, dtype=np.float32)
            base[bt_idx] = 1.0
            base[len(_BOND_TYPES)] = conjugated
            base[len(_BOND_TYPES) + 1] = in_ring
            for (u, v) in ((i, j), (j, i)):
                src.append(u)
                dst.append(v)
                attrs.append(base)
        if not src:
            return (
                np.zeros((2, 0), dtype=np.int64),
                np.zeros((0, self._edge_dim), dtype=np.float32),
            )
        return (
            np.array([src, dst], dtype=np.int64),
            np.stack(attrs, axis=0).astype(np.float32),
        )


__all__ = ["MorganFingerprinter", "GraphFeaturizer", "GraphFeaturizerConfig"]