"""MetalCytoToxDB loader — precious-metal anticancer cytotoxicity data.

Dataset
-------
`MetalCytoToxDB.csv` (26,801 rows, 24 columns) collects photo-activated
cytotoxicity measurements on Ru/Ir/Rh/Os/Re organometallic complexes across
many cell lines.  Each row contains:

    SMILES_Ligands   : ligand SMILES (may include metal centre + counterion)
    Counterion       : counter-ion text (often '.' separated)
    IC50_Dark_value  : dark-toxicity IC50 in micromolar (float)
    IC50_Light_value : light-activated IC50 in micromolar (float)
    Time(h)          : incubation time in hours
    Year             : publication year
    Cell_line        : cell-line identifier
    Metal            : 'Ru' | 'Ir' | 'Rh' | 'Os' | 'Re'
    Charge_complex   : net charge of the complex (int)
    Oxidation_state  : oxidation state of the metal centre (int)
    ...

We standardise labels into:

    pIC50   = -log10(IC50_Dark_value * 1e-6)       (M → pIC50)
    active  = (IC50_Dark_value < 10 uM)            (binary)

A configurable :class:`CytotoxFilter` selects rows by:

    * time_threshold    (hours, default 24)       → keep rows with Time(h) <= t
    * ic50_min          (uM, default 0.01)         → drop rows with IC50 < t
                                                     (very low values are noisy)
    * metal_whitelist   (list[str], default None)  → restrict to given metals
    * year_min/year_max (int, default None)        → temporal subset

3D conformer generation is on-demand and cached to an LRU dict (max 1000
entries).  Cache files are stored under
``/mnt/storage/data/molmetal/3d_cache/{hash}.pkl`` so cache survives across
process restarts.

Usage
-----
>>> ds = MetalCytotoxDataset.from_csv(metal_whitelist=["Ru"])
>>> print(len(ds))                    # ~19,000 rows
>>> item = ds[0]
>>> item["smiles"], item["metal"], item["pIC50"]
"""

from __future__ import annotations

import hashlib
import os
import pickle
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from molmetal.domain import Molecule

# Default data root — overridable via env MOLMETAL_DATA_DIR
DEFAULT_DATA_DIR = Path("/mnt/storage/data/molmetal")
DEFAULT_CSV = DEFAULT_DATA_DIR / "MetalCytoToxDB.csv"
DEFAULT_CACHE_DIR = DEFAULT_DATA_DIR / "3d_cache"

# LRU size for the in-process conformer cache.  ~1000 entries × ~10 KB ≈ 10 MB.
CONFORMER_CACHE_LRU = 1000


# ---------------------------------------------------------------------------
# Filter dataclass
# ---------------------------------------------------------------------------
@dataclass
class CytotoxFilter:
    """Configurable row-level filters for the MetalCytoToxDB loader.

    All numeric fields are optional; ``None`` means "do not filter on this
    criterion".  Sensible *recommended* defaults are documented in the
    field help-strings (e.g. ``time_threshold=24.0`` keeps only short-term
    measurements), but the constructor defaults to ``None`` so an
    unfiltered ``metal_whitelist=['Ru']`` returns the full ~19 k rows for
    that metal.
    """

    time_threshold: Optional[float] = None        # hours, drop Time(h) > t
    ic50_min: Optional[float] = None             # uM,    drop IC50 < t
    ic50_max: Optional[float] = None             # uM,    drop IC50 > t
    metal_whitelist: Optional[Sequence[str]] = None  # e.g. ["Ru"]
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    require_active_field: bool = True  # require IC50_Dark_value non-null
    # Extra derived columns to add per row (defaults below)
    compute_pic50: bool = True
    compute_active: bool = True


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MetalCytotoxDataset:
    """Memory-resident loader for the MetalCytoToxDB CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-filtered dataframe with at least the columns documented in the
        module docstring.  Use :meth:`from_csv` for the typical path.
    cache_dir : Path
        Directory for on-disk 3D conformer pickles.  Created if missing.
    conformer_cache_lru : int
        Maximum number of in-memory conformers.  Past this, least-recently
        used conformers are evicted from the LRU dict.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        conformer_cache_lru: int = CONFORMER_CACHE_LRU,
    ) -> None:
        if df.empty:
            raise ValueError("MetalCytotoxDataset received empty DataFrame")
        # Reset index so positional access matches row-order
        self.df = df.reset_index(drop=True).copy()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lru_max = int(conformer_cache_lru)
        self._conformer_cache: "OrderedDict[str, Molecule]" = OrderedDict()
        # Pre-extract year / metal as plain numpy arrays for fast splits
        self._years = self.df["Year"].to_numpy() if "Year" in self.df.columns else np.zeros(len(self.df))
        self._metals = self.df["Metal"].astype(str).to_numpy() if "Metal" in self.df.columns else np.array(["?"] * len(self.df))

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_csv(
        cls,
        path: str | Path = DEFAULT_CSV,
        filters: Optional[CytotoxFilter] = None,
        **filter_kwargs,
    ) -> "MetalCytotoxDataset":
        """Load + filter MetalCytoToxDB.csv.

        ``filters`` may be a pre-built :class:`CytotoxFilter`; otherwise
        kwargs are forwarded into :class:`CytotoxFilter`.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"MetalCytoToxDB not found at {path}")
        df = pd.read_csv(path, low_memory=False)
        f = filters or CytotoxFilter(**filter_kwargs)
        df = _apply_filters(df, f)
        if f.compute_pic50:
            df = _add_pic50(df)
        if f.compute_active:
            df = _add_active(df)
        return cls(df=df)

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[int(idx)]
        out: Dict[str, object] = {
            "smiles": str(row.get("SMILES_Ligands", "")),
            "metal": str(row.get("Metal", "")),
            "cell_line": str(row.get("Cell_line", "")),
            "pIC50": float(row.get("pIC50", float("nan"))),
            "active": bool(row.get("active", False)),
            "year": int(row.get("Year", 0)),
            "charge_complex": int(row.get("Charge_complex", 0)),
            "oxidation_state": int(row.get("Oxidation_state", 0)),
        }
        return out

    # ------------------------------------------------------------------
    # Column accessors used by splitters / featurisers
    # ------------------------------------------------------------------
    @property
    def smiles(self) -> np.ndarray:
        return self.df["SMILES_Ligands"].astype(str).to_numpy()

    @property
    def metals(self) -> np.ndarray:
        return self._metals

    @property
    def years(self) -> np.ndarray:
        return self._years

    @property
    def pic50(self) -> np.ndarray:
        return self.df["pIC50"].to_numpy(dtype=np.float32)

    @property
    def active(self) -> np.ndarray:
        return self.df["active"].astype(bool).to_numpy()

    def metal_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for m in self._metals:
            out[m] = out.get(m, 0) + 1
        return out

    # ------------------------------------------------------------------
    # 3D conformer cache
    # ------------------------------------------------------------------
    def get_conformer(
        self,
        smiles: str,
        *,
        embed_3d: bool = True,
    ) -> Optional[Molecule]:
        """Return a 3D conformer for ``smiles`` (None if embedding fails).

        Cache key is a SHA1 of the canonical SMILES.  On cache hit the
        pickled Molecule dataclass is restored.  On miss, RDKit ETKDGv3 +
        MMFF94 is used, then the result is pickled to
        ``{cache_dir}/{hash}.pkl``.
        """
        from rdkit import Chem
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")  # suppress noisy parse warnings
        canon_smiles = _canonical_smiles(smiles)
        if not canon_smiles:
            return None
        key = hashlib.sha1(canon_smiles.encode("utf-8")).hexdigest()
        # In-memory LRU
        if key in self._conformer_cache:
            self._conformer_cache.move_to_end(key)
            return self._conformer_cache[key]
        # On-disk
        cache_file = self.cache_dir / f"{key}.pkl"
        mol_obj: Optional[Molecule] = None
        if cache_file.exists():
            try:
                with cache_file.open("rb") as fh:
                    mol_obj = pickle.load(fh)
            except Exception:
                mol_obj = None
        if mol_obj is None:
            try:
                mol_obj = Molecule.from_smiles(canon_smiles, embed_3d=embed_3d)
            except Exception:
                return None
            # Persist
            try:
                with cache_file.open("wb") as fh:
                    pickle.dump(mol_obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                # Cache failures are non-fatal.
                pass
        # Update LRU
        self._conformer_cache[key] = mol_obj
        self._conformer_cache.move_to_end(key)
        while len(self._conformer_cache) > self._lru_max:
            self._conformer_cache.popitem(last=False)
        return mol_obj

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, object]:
        return {
            "n_rows": int(len(self)),
            "metals": self.metal_counts(),
            "year_min": int(np.min(self._years)) if len(self._years) else None,
            "year_max": int(np.max(self._years)) if len(self._years) else None,
            "pic50_min": float(np.nanmin(self.pic50)) if len(self.pic50) else None,
            "pic50_max": float(np.nanmax(self.pic50)) if len(self.pic50) else None,
            "active_fraction": float(np.mean(self.active)) if len(self) else 0.0,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _apply_filters(df: pd.DataFrame, f: CytotoxFilter) -> pd.DataFrame:
    """Apply :class:`CytotoxFilter` row-wise.  Returns a *copy*."""
    out = df
    if f.require_active_field and "IC50_Dark_value" in out.columns:
        out = out[out["IC50_Dark_value"].notna()]
    if f.time_threshold is not None and "Time(h)" in out.columns:
        out = out[out["Time(h)"] <= float(f.time_threshold)]
    if f.ic50_min is not None and "IC50_Dark_value" in out.columns:
        out = out[out["IC50_Dark_value"] >= float(f.ic50_min)]
    if f.ic50_max is not None and "IC50_Dark_value" in out.columns:
        out = out[out["IC50_Dark_value"] <= float(f.ic50_max)]
    if f.metal_whitelist is not None and "Metal" in out.columns:
        wl = set(f.metal_whitelist)
        out = out[out["Metal"].astype(str).isin(wl)]
    if f.year_min is not None and "Year" in out.columns:
        out = out[out["Year"] >= int(f.year_min)]
    if f.year_max is not None and "Year" in out.columns:
        out = out[out["Year"] <= int(f.year_max)]
    return out.copy()


def _add_pic50(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pIC50 = -log10(IC50_uM * 1e-6) safely."""
    ic50 = df["IC50_Dark_value"].astype(float)
    # Avoid log(0) by clipping
    ic50_m = np.clip(ic50.to_numpy() * 1e-6, 1e-12, None)
    df = df.copy()
    df["pIC50"] = -np.log10(ic50_m).astype(np.float32)
    return df


def _add_active(df: pd.DataFrame) -> pd.DataFrame:
    """active := IC50_Dark_value < 10 uM."""
    df = df.copy()
    df["active"] = (df["IC50_Dark_value"].astype(float) < 10.0).astype(bool)
    return df


def _canonical_smiles(smiles: str) -> str:
    """Best-effort canonical SMILES.  Returns original on failure."""
    try:
        from rdkit import Chem
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol)
    except Exception:
        return smiles


__all__ = ["MetalCytotoxDataset", "CytotoxFilter"]