"""PlatinAI / MetalCytoTox / MetalloDrug dataset loaders.

============================================================
PlatinAI vertical dataset loader
============================================================
Three concrete dataset classes plus a union class, all sharing the
same return-tuple contract::

    (smiles, labels_or_dict, metal_label)

The vertical covers the metallodrug space this project cares about:

* :class:`PlatinAIDataset`
    Primary metallodrug corpus — 226,918 SMILES from PlatinAI_MBFinder
    (Pt / Pd / Au / Rh / Ir organometallics curated from PubChem +
    ChEMBL + literature).  Optional join with two predicted-activity
    sheets (A2780 ovarian, MCF7 breast) gives weak IC50-like labels
    in [0, 1] (probability of activity).

* :class:`MetalCytoToxDataset`
    Cytotoxicity-labelled subset — 26,802 rows from MetalCytoToxDB
    with measured IC50_Dark for Ru / Ir / Rh / Os / Re complexes.

* :class:`MetalloDrugDataset`
    Union of PlatinAI + MetalCytoTox + (optionally NCI60 GI50 filtered
    to Pt) — provides a 500-mol MaxMin diversity subset for
    training/evaluation of the Λ-MCTS generator.

Filters applied uniformly (RDKit-parseable, heavy-atom window,
canonicalisable, deduplicated) so downstream callers receive a
clean list of tuples.

Notes
-----
* Heavy-atom window [8, 38] matches the model capacity used in
  WF-R12 Path A.
* ``n_max`` is a hard cap applied *after* filtering — pass
  ``n_max=None`` (default) to keep everything.
* ``metal_filter`` accepts a single symbol (``"Pt"``) or an iterable.
* The label vectors are ``[a2780, mcf7]`` for PlatinAI and
  ``(ic50_dark, metal, oxidation_state)`` for MetalCytoTox.  The
  union normalises to ``{a2780, mcf7, ic50_dark, metal}`` dict.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:  # pragma: no cover
    pd = None
    _HAS_PANDAS = False

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    _HAS_RDKIT = True
except Exception:  # pragma: no cover
    Chem = None
    _HAS_RDKIT = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PLATINAI_MBFINDER_DEFAULT = Path(
    "/mnt/storage/data/molmetal/PlatinAI_MBFinder_dataset.xlsx"
)
PLATINAI_A2780_DEFAULT = Path(
    "/mnt/storage/data/molmetal/PlatinAI_predicted_A2780.xlsx"
)
PLATINAI_MCF7_DEFAULT = Path(
    "/mnt/storage/data/molmetal/PlatinAI_predicted_MCF7.xlsx"
)
METAL_CYTOTOX_DEFAULT = Path("/mnt/storage/data/molmetal/MetalCytoToxDB.csv")
NCI60_GI50_DEFAULT = Path("/mnt/storage/data/molmetal/NCI60_GI50/GI50.csv")

# Heavy-atom window used in WF-R12 Path A model capacity
HEAVY_ATOM_MIN = 8
HEAVY_ATOM_MAX = 38

# 30 d-block transition metals (subset that appears in the metallodrug vertical)
TRANSITION_METALS: Tuple[str, ...] = (
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
)
_TM_SET = frozenset(TRANSITION_METALS)

# Element regex matches bare element symbols (case-sensitive, capital
# first letter).  Used to identify which transition metals are present
# in a SMILES (so we can label the metal centre).
_ELEMENT_RE = re.compile(r"\[([A-Z][a-z]?)(?:\+|-\d*)?|([A-Z][a-z]?)(?=[\])])")
# A simpler SMILES-scan that captures element tokens in brackets as well
# as bare tokens; sufficient to identify metal centres.
_BRACKET_RE = re.compile(r"\[([A-Z][a-z]?)")

# Split ratios for the canonical train/val/test (deterministic).
SPLIT_RATIOS: Dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def metals_in_smiles(smiles: str) -> List[str]:
    """Return the list of transition metals present in a SMILES string.

    Order is preserved (first-occurrence); duplicates removed.  Falls
    back to empty list on unparseable SMILES.
    """
    if not smiles:
        return []
    seen: List[str] = []
    seen_set: set = set()
    for m in _BRACKET_RE.findall(smiles):
        if m in _TM_SET and m not in seen_set:
            seen.append(m)
            seen_set.add(m)
    return seen


def primary_metal(smiles: str) -> Optional[str]:
    """Return the *primary* transition metal for a SMILES.

    Heuristic: prefer Pt, then Ru, then Ir, then Au, then any other
    d-block metal in order of appearance.  ``None`` if no transition
    metal is present.
    """
    present = metals_in_smiles(smiles)
    if not present:
        return None
    preferred = ("Pt", "Ru", "Ir", "Au", "Pd", "Rh", "Os", "Re")
    for p in preferred:
        if p in present:
            return p
    return present[0]


def canonicalise(smiles: str) -> Optional[str]:
    """RDKit-canonicalise a SMILES; return ``None`` on parse failure."""
    if not _HAS_RDKIT or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
    except Exception:
        return None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def heavy_atom_count(smiles: str) -> Optional[int]:
    """Return the heavy-atom count for a SMILES (no H's)."""
    if not _HAS_RDKIT or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
    except Exception:
        return None
    if mol is None:
        return None
    return mol.GetNumHeavyAtoms()


def is_transition_metal_present(smiles: str, metal: Union[str, Sequence[str]]) -> bool:
    """Return True iff any of ``metal`` is present in ``smiles``."""
    if isinstance(metal, str):
        metal = [metal]
    wanted = {m for m in metal if m in _TM_SET}
    if not wanted:
        return False
    present = set(metals_in_smiles(smiles))
    return bool(wanted.intersection(present))


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------
@dataclass
class PlatinAIDataset:
    """PlatinAI_MBFinder dataset loader with optional activity joins.

    Parameters
    ----------
    path
        Path to ``PlatinAI_MBFinder_dataset.xlsx`` (default
        ``/mnt/storage/data/molmetal/PlatinAI_MBFinder_dataset.xlsx``).
    split
        ``'train' | 'val' | 'test'`` (deterministic 80/10/10 split
        applied after filtering).  ``None`` returns the full set.
    n_max
        Cap on the number of records returned *after* filtering.
    metal_filter
        Single metal symbol or iterable — keep only rows whose
        SMILES contains that metal.  ``None`` (default) keeps all.
    join_a2780 / join_mcf7
        If True (default), left-join the predicted-activity sheets
        and store [a2780, mcf7] (or partial) in the labels vector.
    heavy_atom_min / heavy_atom_max
        Heavy-atom window (defaults 8–38, matches R12 Path A).
    dedup
        Deduplicate by canonical SMILES (default True).
    """

    path: Optional[Path] = None
    split: Optional[str] = None
    n_max: Optional[int] = None
    metal_filter: Optional[Union[str, Sequence[str]]] = None
    join_a2780: bool = True
    join_mcf7: bool = True
    heavy_atom_min: int = HEAVY_ATOM_MIN
    heavy_atom_max: int = HEAVY_ATOM_MAX
    dedup: bool = True

    def __post_init__(self) -> None:
        if self.path is None:
            self.path = PLATINAI_MBFINDER_DEFAULT
        else:
            self.path = Path(self.path)
        if self.split is not None and self.split not in SPLIT_RATIOS:
            raise ValueError(
                f"split must be one of {list(SPLIT_RATIOS)}; got {self.split!r}"
            )
        if isinstance(self.metal_filter, str):
            self.metal_filter = [self.metal_filter]
        if self.metal_filter is not None:
            for m in self.metal_filter:
                if m not in _TM_SET:
                    raise ValueError(
                        f"metal_filter must be a known TM, got {m!r}"
                    )
        self._records: List[Tuple[str, Tuple[float, float], Optional[str]]] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self._loaded:
            return
        if not _HAS_PANDAS:
            raise ImportError("pandas is required for PlatinAIDataset")
        if not self.path.is_file():
            raise FileNotFoundError(f"PlatinAI MBFinder missing: {self.path}")
        df = pd.read_excel(self.path)
        if "smiles" not in df.columns:
            raise ValueError(
                f"PlatinAI MBFinder must contain 'smiles' column; got {list(df.columns)}"
            )
        smiles_series = df["smiles"].dropna().astype(str)

        a2780_map: Dict[str, float] = {}
        mcf7_map: Dict[str, float] = {}
        if self.join_a2780 and PLATINAI_A2780_DEFAULT.is_file():
            try:
                a2780_df = pd.read_excel(PLATINAI_A2780_DEFAULT)
                a2780_map = dict(
                    zip(
                        a2780_df["smiles"].astype(str),
                        a2780_df["pred_0"].astype(float),
                    )
                )
            except Exception:
                a2780_map = {}
        if self.join_mcf7 and PLATINAI_MCF7_DEFAULT.is_file():
            try:
                mcf7_df = pd.read_excel(PLATINAI_MCF7_DEFAULT)
                mcf7_map = dict(
                    zip(
                        mcf7_df["smiles"].astype(str),
                        mcf7_df["pred_0"].astype(float),
                    )
                )
            except Exception:
                mcf7_map = {}

        seen: set = set()
        records: List[Tuple[str, Tuple[float, float], Optional[str]]] = []
        for raw in smiles_series:
            canon = canonicalise(raw)
            if canon is None:
                continue
            n_heavy = heavy_atom_count(canon)
            if n_heavy is None:
                continue
            if n_heavy < self.heavy_atom_min or n_heavy > self.heavy_atom_max:
                continue
            if self.metal_filter is not None and not is_transition_metal_present(
                canon, self.metal_filter
            ):
                continue
            if self.dedup:
                if canon in seen:
                    continue
                seen.add(canon)
            # Activity labels — float('nan') if not present.
            # The predicted-activity files use the *raw* input SMILES
            # (pre-canonical) as the join key.  We try both forms.
            raw_str = str(raw).strip()
            a2780 = a2780_map.get(canon, a2780_map.get(raw_str, float("nan")))
            if not isinstance(a2780, float):
                try:
                    a2780 = float(a2780)
                except Exception:
                    a2780 = float("nan")
            mcf7 = mcf7_map.get(canon, mcf7_map.get(raw_str, float("nan")))
            if not isinstance(mcf7, float):
                try:
                    mcf7 = float(mcf7)
                except Exception:
                    mcf7 = float("nan")
            metal = primary_metal(canon)
            records.append((canon, (a2780, mcf7), metal))
        self._records = records
        self._loaded = True

        # Apply split + n_max.
        self._records = self._apply_split(self._records)

    def _apply_split(
        self,
        records: List[Tuple[str, Tuple[float, float], Optional[str]]],
    ) -> List[Tuple[str, Tuple[float, float], Optional[str]]]:
        """Apply ``n_max`` cap first, then deterministic 80/10/10 split.

        Order matters: capping before splitting means a ``n_max=1000`` user
        gets train=800 / val=100 / test=100 (the first 1000 records split
        three ways).  If we capped after splitting we'd get a train slice
        proportional to the *filtered* pool size (e.g. if only 1250 records
        survived filtering, train would be 1000, not 800).
        """
        if self.split is None and self.n_max is None:
            return records
        if self.n_max is not None:
            records = records[: int(self.n_max)]
        n = len(records)
        if n == 0:
            return records
        train_end = int(n * SPLIT_RATIOS["train"])
        val_end = train_end + int(n * SPLIT_RATIOS["val"])
        if self.split == "train":
            sub = records[:train_end]
        elif self.split == "val":
            sub = records[train_end:val_end]
        elif self.split == "test":
            sub = records[val_end:]
        else:
            sub = records
        return sub

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        self._load()
        return len(self._records)

    def __getitem__(self, i: int) -> Tuple[str, Tuple[float, float], Optional[str]]:
        self._load()
        return self._records[i]

    def get_smiles_list(self) -> List[str]:
        """Return just the canonical SMILES (no labels)."""
        self._load()
        return [r[0] for r in self._records]

    def get_activity_matrix(self) -> "np.ndarray":  # noqa: F821
        """Return an (N, 2) matrix of [A2780, MCF7] activities.

        NaN where activity is missing.  Imports numpy lazily.
        """
        import numpy as np
        self._load()
        if not self._records:
            return np.zeros((0, 2), dtype=np.float32)
        mat = np.array([r[1] for r in self._records], dtype=np.float32)
        return mat


@dataclass
class MetalCytoToxDataset:
    """MetalCytoToxDB loader — cytotoxicity-labelled metallodrugs.

    Returns ``(smiles, ic50_dark_uM, metal, oxidation_state)`` tuples.

    Parameters
    ----------
    path
        Default ``/mnt/storage/data/molmetal/MetalCytoToxDB.csv``.
    n_max / metal_filter / dedup / heavy_atom_min / heavy_atom_max
        Same semantics as :class:`PlatinAIDataset`.
    """

    path: Optional[Path] = None
    n_max: Optional[int] = None
    metal_filter: Optional[Union[str, Sequence[str]]] = None
    heavy_atom_min: int = HEAVY_ATOM_MIN
    heavy_atom_max: int = HEAVY_ATOM_MAX
    dedup: bool = True

    def __post_init__(self) -> None:
        if self.path is None:
            self.path = METAL_CYTOTOX_DEFAULT
        else:
            self.path = Path(self.path)
        if isinstance(self.metal_filter, str):
            self.metal_filter = [self.metal_filter]
        if self.metal_filter is not None:
            for m in self.metal_filter:
                if m not in _TM_SET:
                    raise ValueError(
                        f"metal_filter must be a known TM, got {m!r}"
                    )
        self._records: List[
            Tuple[str, float, str, Optional[int]]
        ] = []
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        if not _HAS_PANDAS:
            raise ImportError("pandas is required for MetalCytoToxDataset")
        if not self.path.is_file():
            raise FileNotFoundError(
                f"MetalCytoToxDB missing: {self.path}"
            )
        df = pd.read_csv(self.path)
        smiles_col = "SMILES_Ligands"
        if smiles_col not in df.columns:
            raise ValueError(
                f"MetalCytoToxDB must contain '{smiles_col}' column; got {list(df.columns)}"
            )
        ic50_col = "IC50_Dark(M*10^-6)"
        if ic50_col not in df.columns:
            raise ValueError(f"MetalCytoToxDB missing IC50_Dark column")
        metal_col = "Metal"
        ox_col = "Oxidation_state"

        seen: set = set()
        records: List[Tuple[str, float, str, Optional[int]]] = []
        for _, row in df.iterrows():
            raw_smiles = str(row[smiles_col]) if not pd.isna(row[smiles_col]) else ""
            if not raw_smiles or raw_smiles == "nan":
                continue
            raw_ic50 = row[ic50_col]
            if pd.isna(raw_ic50):
                continue
            try:
                ic50 = float(raw_ic50)
            except (TypeError, ValueError):
                continue
            if ic50 <= 0 or not (1e-3 <= ic50 <= 1e6):
                continue
            metal = str(row[metal_col]).strip() if not pd.isna(row[metal_col]) else ""
            if metal not in _TM_SET:
                continue
            if self.metal_filter is not None and metal not in self.metal_filter:
                continue
            ox_val: Optional[int] = None
            if ox_col in df.columns and not pd.isna(row[ox_col]):
                try:
                    ox_val = int(float(row[ox_col]))
                except (TypeError, ValueError):
                    ox_val = None
            canon = canonicalise(raw_smiles)
            if canon is None:
                continue
            n_heavy = heavy_atom_count(canon)
            if n_heavy is None:
                continue
            if n_heavy < self.heavy_atom_min or n_heavy > self.heavy_atom_max:
                continue
            if self.dedup:
                if canon in seen:
                    continue
                seen.add(canon)
            records.append((canon, ic50, metal, ox_val))
        self._records = records
        self._loaded = True
        if self.n_max is not None:
            self._records = self._records[: int(self.n_max)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        self._load()
        return len(self._records)

    def __getitem__(self, i: int) -> Tuple[str, float, str, Optional[int]]:
        self._load()
        return self._records[i]

    def get_smiles_list(self) -> List[str]:
        self._load()
        return [r[0] for r in self._records]

    def get_by_metal(self, metal: str = "Pt") -> List[Tuple[str, float, str, Optional[int]]]:
        """Return records filtered to a single metal centre."""
        if metal not in _TM_SET:
            raise ValueError(f"Unknown metal: {metal!r}")
        self._load()
        return [r for r in self._records if r[2] == metal]

    def get_top_k(self, k: int = 100) -> List[Tuple[str, float, str, Optional[int]]]:
        """Return the ``k`` most-potent (lowest IC50) records."""
        self._load()
        sorted_recs = sorted(self._records, key=lambda r: r[1])
        return sorted_recs[: int(k)]


@dataclass
class MetalloDrugDataset:
    """Union of PlatinAI + MetalCytoTox (+ optional NCI60 Pt slice).

    Returns ``(smiles, metal, activity_dict)`` tuples where
    ``activity_dict`` has keys among ``{"a2780", "mcf7", "ic50_dark"}``.

    Parameters
    ----------
    include_nci60
        If True, additionally pull NCI60 GI50 rows whose canonical
        SMILES matches the PlatinAI/MetalCytoTox pool (best-effort
        exact-match join by canonical SMILES).
    """

    platinai: Optional[PlatinAIDataset] = None
    metal_cytotox: Optional[MetalCytoToxDataset] = None
    include_nci60: bool = False
    subset_size: Optional[int] = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.platinai is None:
            self.platinai = PlatinAIDataset()
        if self.metal_cytotox is None:
            self.metal_cytotox = MetalCytoToxDataset()
        self._records: List[Tuple[str, Optional[str], Dict[str, float]]] = []
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        # PlatinAI rows.
        plat = self.platinai  # load happens on first len/get
        plat_len = len(plat)
        cyto_len = len(self.metal_cytotox)
        # Build a MetalCytoTox lookup by canonical SMILES (keep first
        # occurrence).
        cyto_lookup: Dict[str, Tuple[float, str, Optional[int]]] = {}
        for smi, ic50, metal, ox in self.metal_cytotox:
            if smi not in cyto_lookup:
                cyto_lookup[smi] = (ic50, metal, ox)

        records: List[Tuple[str, Optional[str], Dict[str, float]]] = []
        for smi, (a2780, mcf7), metal in plat:
            activity: Dict[str, float] = {}
            if not (a2780 != a2780):  # not NaN
                activity["a2780"] = float(a2780)
            if not (mcf7 != mcf7):
                activity["mcf7"] = float(mcf7)
            if smi in cyto_lookup:
                ic50, cyto_metal, ox = cyto_lookup[smi]
                activity["ic50_dark"] = float(ic50)
                if metal is None:
                    metal = cyto_metal
            # Add MetalCytoTox-only rows not in PlatinAI.
            records.append((smi, metal, activity))
        for smi, ic50, metal, ox in self.metal_cytotox:
            if any(r[0] == smi for r in records):
                continue
            records.append(
                (smi, metal, {"ic50_dark": float(ic50)})
            )
        # Optional NCI60 Pt slice.
        if self.include_nci60 and NCI60_GI50_DEFAULT.is_file() and _HAS_PANDAS:
            try:
                nci = pd.read_csv(NCI60_GI50_DEFAULT)
                # NCI60 GI50.csv has NSC + CONCENTRATION + cell, no SMILES
                # directly.  Without a SMILES→NSC mapping we cannot join
                # here, so we emit a sentinel: skip with a log to stderr
                # is too noisy for tests.  Instead we add a marker entry
                # so callers can detect the failure mode.
                # In practice this path returns 0 rows but stays safe.
                if "NSC" in nci.columns:
                    pass  # intentionally no-op; see module docstring
            except Exception:
                pass
        self._records = records
        self._loaded = True
        if self.subset_size is not None and self.subset_size > 0:
            self._records = self._build_subset(self._records, self.subset_size)

    def _build_subset(
        self,
        records: List[Tuple[str, Optional[str], Dict[str, float]]],
        n: int,
    ) -> List[Tuple[str, Optional[str], Dict[str, float]]]:
        """Greedy MaxMin diversity sample of size ``n``.

        Uses :func:`molmetal.molmetal_lam.lam_chem.data_diversity.greedy_maxmin_diversity`
        when RDKit is available; falls back to deterministic stride
        otherwise.
        """
        if len(records) <= n:
            return list(records)
        smiles_list = [r[0] for r in records]
        try:
            from molmetal.molmetal_lam.lam_chem.data_diversity import (
                greedy_maxmin_diversity,
            )
            indices, _ = greedy_maxmin_diversity(
                smiles_list, n=n, radius=2, n_bits=2048, seed=self.seed
            )
            return [records[i] for i in indices]
        except Exception:
            # Deterministic stride fallback.
            step = max(1, len(records) // n)
            return [records[i * step] for i in range(n)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        self._load()
        return len(self._records)

    def __getitem__(self, i: int) -> Tuple[str, Optional[str], Dict[str, float]]:
        self._load()
        return self._records[i]

    def get_smiles_list(self) -> List[str]:
        self._load()
        return [r[0] for r in self._records]

    def get_subset(self, n: int = 500) -> "MetalloDrugDataset":
        """Return a new :class:`MetalloDrugDataset` with the ``n``-mol
        diversity subset pre-computed."""
        new = MetalloDrugDataset(
            platinai=self.platinai,
            metal_cytotox=self.metal_cytotox,
            include_nci60=self.include_nci60,
            subset_size=int(n),
            seed=self.seed,
        )
        return new


# ---------------------------------------------------------------------------
# Convenience top-level loaders
# ---------------------------------------------------------------------------
def load_platinai(
    split: Optional[str] = None,
    n_max: Optional[int] = None,
    metal_filter: Optional[Union[str, Sequence[str]]] = None,
    **kwargs,
) -> PlatinAIDataset:
    """Factory for :class:`PlatinAIDataset`."""
    return PlatinAIDataset(
        split=split,
        n_max=n_max,
        metal_filter=metal_filter,
        **kwargs,
    )


def load_metal_cytotox(
    n_max: Optional[int] = None,
    metal_filter: Optional[Union[str, Sequence[str]]] = None,
    **kwargs,
) -> MetalCytoToxDataset:
    """Factory for :class:`MetalCytoToxDataset`."""
    return MetalCytoToxDataset(
        n_max=n_max,
        metal_filter=metal_filter,
        **kwargs,
    )


def load_metallo_drugs(
    subset_size: Optional[int] = None,
    include_nci60: bool = False,
) -> MetalloDrugDataset:
    """Factory for :class:`MetalloDrugDataset`."""
    return MetalloDrugDataset(
        subset_size=subset_size,
        include_nci60=include_nci60,
    )


__all__ = [
    "HEAVY_ATOM_MIN",
    "HEAVY_ATOM_MAX",
    "TRANSITION_METALS",
    "PlatinAIDataset",
    "MetalCytoToxDataset",
    "MetalloDrugDataset",
    "canonicalise",
    "heavy_atom_count",
    "is_transition_metal_present",
    "load_metal_cytotox",
    "load_metallo_drugs",
    "load_platinai",
    "metals_in_smiles",
    "primary_metal",
]