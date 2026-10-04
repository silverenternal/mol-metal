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

# Default data root — overridable via env MOLMETAL_DATA_DIR.
#
# Wf-A9 FIX (2026-10-03): this env var was documented but never actually
# read — ``DEFAULT_DATA_DIR`` was a hard-coded constant pointing at the
# ``/mnt/storage`` mount, which does not exist on this host.  The D-MPNN
# retrain path therefore died with
# ``FileNotFoundError: MetalCytoToxDB not found at
# /mnt/storage/data/molmetal/MetalCytoToxDB.csv`` before the first epoch
# (raised from :meth:`MetalCytotoxDataset.from_csv`).  The real CSV lives
# in the project-local ``.data/molmetal/`` tree (see :file:`env.sh` §1b,
# which exports ``MOLMETAL_DATA_DIR=/…/.data/molmetal``).
#
# This now honours the env var the same way
# :func:`molmetal.data.four_domain_combined._resolve_metalcytotox_csv`
# already did, and keeps ``/mnt/storage`` as the last-resort fallback so
# hosts that do have the mount are unaffected.
DEFAULT_DATA_DIR = Path(
    os.environ.get("MOLMETAL_DATA_DIR") or "/mnt/storage/data/molmetal"
)
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

    time_threshold: Optional[float] = None  # hours, drop Time(h) > t
    ic50_min: Optional[float] = None  # uM,    drop IC50 < t
    ic50_max: Optional[float] = None  # uM,    drop IC50 > t
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
        self._years = (
            self.df["Year"].to_numpy()
            if "Year" in self.df.columns
            else np.zeros(len(self.df))
        )
        self._metals = (
            self.df["Metal"].astype(str).to_numpy()
            if "Metal" in self.df.columns
            else np.array(["?"] * len(self.df))
        )

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


# ---------------------------------------------------------------------------
# Paper-funnel cohort builder + DOI-split 10-fold CV (OPT-IN, additive)
# ---------------------------------------------------------------------------
# Krasnov 2026 (MetalCytoToxDB v2, Zenodo 10.5281/zenodo.17106822) funnel:
#
#     Ru rows -> measurement-level dedup -> non-cancer cell-line exclusion
#
# Everything below is ADDITIVE.  Nothing here changes the behaviour of
# :class:`CytotoxFilter` / :meth:`MetalCytotoxDataset.from_csv`, and nothing in
# the trainer calls these helpers unless a caller explicitly opts in.  The
# funnel targets v2 (md5 f270dc907dbdfede953063a14f0825a7); the frozen
# calibration anchor on that release is 15,799 rows / 4,793 complexes /
# 591 cell lines (measured 2026-10-04; asserted in
# molmetal/tests/test_paper_cohort_doi_cv.py).
#
# "Complex" follows the release's own definition: a unique
# (SMILES_Ligands, Metal) pair.  Dedup is measurement-level
# (SMILES_Ligands, Cell_line, Time(h)) with keep="first"; adding DOI to the
# key is supported and moves 15,799 -> 16,166 rows on v2.  The paper applies
# NO exposure-time or IC50 noise-floor filter — incubation time stays a
# feature — so this builder only drops rows without an IC50 value.
#
# The exclusion list below is the external non-cancer curation the paper uses
# (the CSV carries no cancer/non-cancer flag).  It is calibrated against the
# frozen counts above; cell-line names are matched exactly after
# strip().upper() normalisation (never by substring — e.g. tumour-derived
# "HLF" must not collide with normal "HFL-1").  Ambiguous or tumour-derived
# lines (ECV-304, Chang Liver, T4-2, RAW 264.7, PC-12, HLF, HPL1D, MS1, OSE,
# WRL68, Hs88) are deliberately NOT excluded.
NON_CANCER_CELL_LINES: tuple = (
    # kidney / embryonic kidney
    "HEK293", "HEK293T",
    # diploid fibroblasts
    "MRC-5", "MRC5pd30", "IMR-90", "WI-38", "W138", "HFL-1", "HELF", "L132",
    "NFF", "HFF", "HFF-1", "HDF", "HDFa", "HdFa", "NHDF", "BJ", "Hs68",
    "KMST-6", "1BR.3.G",
    # endothelium
    "HUVEC", "HBMEC", "MLuMEC",
    # non-tumorigenic breast
    "MCF-10A", "MCF-12A",
    # keratinocyte
    "HaCaT",
    # airway epithelium
    "16HBE", "BEAS-2B", "HBE", "HSAEC",
    # hepatocytes
    "LO2", "HL-7702", "THLE-2",
    # normal kidney / ovary / hamster lung (incl. non-human)
    "HK-2", "NRK", "Vero", "BHK", "BHK-21", "CHO", "CHO p-40", "MDBK",
    "BGM", "V79",
    # GI / prostate / retina / cornea epithelium
    "GES-1", "NCM460", "PNT2", "PNT1A", "ARPE-19", "RPE-1", "HEEC", "HCEC",
    # primary PBMC
    "PBMC", "PBMs", "PBMCs", "PBM", "PBMC-PHA", "PBMC+PHA",
    # normal mouse fibroblast / muscle
    "NIH-3T3", "BALB/3T3", "BALB/3T3 clone A31", "Balb/c 3T3", "3T3-L1",
    "MC3T3-e1", "L-929", "L6", "H9c2",
    # normal cervix (the data carries the literal "Ect1/E6E7" string)
    "Ect1/E6E7",
    # CCD normal fibroblast series
    "CCD-18Co", "CCD-19Co", "CCD-841", "CCD-841-CON", "CCD-1029Sk",
    "CCD-1059Sk", "CCD-19Lu", "CCD1064Sk",
)

# Default dedup key: measurement-level (paper-style), keep="first".
PAPER_DEDUP_KEYS: tuple = ("SMILES_Ligands", "Cell_line", "Time(h)")


def _norm_cell_line(value: object) -> str:
    """Case/whitespace-normalised cell-line name (exact match only)."""
    return str(value).strip().upper()


def paper_cohort_mask(
    df: pd.DataFrame,
    *,
    metal: Optional[str] = "Ru",
    dedup_keys: Optional[Sequence[str]] = PAPER_DEDUP_KEYS,
    exclude_non_cancer: bool = True,
    require_ic50: bool = True,
    non_cancer_cell_lines: Sequence[str] = NON_CANCER_CELL_LINES,
    cell_lines: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Boolean row mask (aligned to ``df``) selecting the paper-style cohort.

    Order of operations matters for reproducibility: IC50 presence + metal
    filter first, then dedup *within that subset* (keep="first" in original
    row order), then the non-cancer exclusion, then the optional cell-line
    restriction.  ``metal=None`` disables the metal filter; ``dedup_keys=None``
    disables dedup; ``cell_lines=None`` keeps every surviving line.  Never
    mutates ``df``.

    ``cell_lines`` exists for the Krasnov-comparable subset.  The paper's Ru ML
    set is 23 cell lines; it does not publish WHICH 23, so the subset has to be
    chosen here and DECLARED.  MEASURED: the top 23 Ru lines by row volume
    (``top_cell_lines``) hold 11,517 rows of the 15,799-row paper-style cohort
    — within 2% of the paper's 11,294, which is why this is the best available
    reconstruction rather than merely a convenient one.
    """
    if df.empty and len(df) == 0:
        return np.zeros(0, dtype=bool)
    mask = np.ones(len(df), dtype=bool)
    if require_ic50:
        if "IC50_Dark_value" not in df.columns:
            raise KeyError("paper cohort requires column 'IC50_Dark_value'")
        mask &= df["IC50_Dark_value"].notna().to_numpy()
    if metal is not None:
        if "Metal" not in df.columns:
            raise KeyError("paper cohort requires column 'Metal'")
        mask &= (df["Metal"].astype(str) == str(metal)).to_numpy()
    if dedup_keys:
        missing = [c for c in dedup_keys if c not in df.columns]
        if missing:
            raise KeyError(f"dedup_keys missing from frame: {missing}")
        pos = np.flatnonzero(mask)
        if pos.size:
            sub = df.iloc[pos]
            dup = sub.duplicated(subset=list(dedup_keys), keep="first").to_numpy()
            mask[pos[dup]] = False
    if exclude_non_cancer:
        if "Cell_line" not in df.columns:
            raise KeyError("paper cohort requires column 'Cell_line'")
        excluded = {_norm_cell_line(x) for x in non_cancer_cell_lines}
        keep = ~df["Cell_line"].map(_norm_cell_line).isin(excluded)
        mask &= keep.to_numpy()
    if cell_lines is not None:
        if "Cell_line" not in df.columns:
            raise KeyError("cell_lines filter requires column 'Cell_line'")
        mask &= df["Cell_line"].astype(str).isin(
            {str(c) for c in cell_lines}
        ).to_numpy()

    return mask


def top_cell_lines(df: pd.DataFrame, n: int) -> List[str]:
    """The ``n`` cell lines with the most rows in ``df``, ties broken by name.

    Used to reconstruct the paper's 23-cell-line scope, which it does not
    publish.  The tie-break is on the NAME, not on frame order, so the result
    does not depend on how the CSV happened to be sorted.
    """
    if "Cell_line" not in df.columns:
        raise KeyError("top_cell_lines requires column 'Cell_line'")
    counts = df["Cell_line"].astype(str).value_counts()
    ordered = sorted(counts.index.tolist(), key=lambda c: (-int(counts[c]), c))
    return ordered[: int(n)]


def build_paper_cohort(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Filtered *copy* of ``df`` holding the paper-style cohort.

    Returns ``df.loc[paper_cohort_mask(df, **kwargs)]`` — the input frame is
    untouched and its index is preserved, so
    :func:`build_paper_cohort_indices` positions and ``.loc`` both line up.
    """
    return df.loc[paper_cohort_mask(df, **kwargs)]


def build_paper_cohort_indices(df: pd.DataFrame, **kwargs) -> np.ndarray:
    """Positional row indices of the paper-style cohort inside ``df``."""
    return np.flatnonzero(paper_cohort_mask(df, **kwargs))


def assign_doi_folds(
    dois,
    *,
    n_folds: int = 10,
    seed: int = 0,
    smiles=None,
) -> np.ndarray:
    """Deterministic fold id per row; all rows sharing a DOI get one fold.

    Groups are formed by exact DOI string; a missing/NaN/empty DOI is treated
    as unique per row (never silently merged into one giant group).  Group
    order is shuffled with ``np.random.default_rng(seed)`` and each group is
    assigned greedily to the currently-smallest fold (largest groups first),
    so folds balance on row count.  Deterministic across calls for a fixed
    ``(input, n_folds, seed)``.

    ``smiles`` (2026-10-04) makes the split usable by the D-MPNN trainer.
    That trainer derives each row's split from a **SMILES-keyed** lookup
    (``build_metal_split_indices``: ``smi_to_split[smi] = row["split"]``, last
    write wins, and a SMILES it has not seen defaults to ``train``).  So a
    row-level DOI split is silently incoherent there whenever one complex
    appears in two articles — MEASURED, 6.1% of cohort complexes span more than
    one DOI, up to 13 — and the trainer would collapse those groups, leaking
    across folds.

    Passing ``smiles`` merges any DOIs linked by a shared complex into a single
    group (union-find over the DOI<->SMILES bipartite graph) before folds are
    assigned.  Every complex then sits wholly in one fold, so the DOI guarantee
    and the trainer's SMILES keying agree.
    """
    if int(n_folds) < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    codes = doi_group_ids(dois, smiles=smiles)
    sizes = np.bincount(codes, minlength=int(codes.max()) + 1)
    rng = np.random.default_rng(int(seed))
    n_groups = int(codes.max()) + 1
    order = rng.permutation(n_groups)
    order_by_size = order[np.argsort(-sizes[order], kind="stable")]
    fold_of_group = np.empty(n_groups, dtype=np.int64)
    loads = np.zeros(int(n_folds), dtype=np.int64)
    for gi in order_by_size:
        f = int(np.argmin(loads))
        fold_of_group[gi] = f
        loads[f] += int(sizes[gi])
    return fold_of_group[codes]


def doi_group_ids(dois, *, smiles=None) -> np.ndarray:
    """Dense group id per row; DOIs joined by a shared complex share an id.

    Extracted from :func:`assign_doi_folds` (2026-10-04) so that "which rows
    must travel together" has ONE definition.  A caller that needs to carve a
    sub-split (e.g. an inner validation slice of a cross-validation fold)
    needs these ids, not fold numbers — the earlier inline copy in
    ``run_m1_paper_cv.py`` used RAW DOIs, which is not the same grouping: a
    complex appearing in two articles links those articles into one group, and
    MEASURED 6.1% of cohort complexes do exactly that.  Splitting on raw DOIs
    therefore leaked 36 DOI groups (MEASURED, the driver's own guard caught
    it).
    """
    values = dois.to_numpy() if isinstance(dois, pd.Series) else np.asarray(dois, dtype=object)
    if smiles is not None:
        smiles = (
            smiles.to_numpy()
            if isinstance(smiles, pd.Series)
            else np.asarray(smiles, dtype=object)
        )
        if len(smiles) != len(values):
            raise ValueError(
                f"smiles has {len(smiles)} entries but dois has {len(values)}"
            )
    keys = []
    for i, v in enumerate(values):
        if v is None or (isinstance(v, float) and v != v) or pd.isna(v):
            keys.append(f"__row_{i}")
        else:
            s = str(v).strip()
            keys.append(s if s else f"__row_{i}")
    keys = np.asarray(keys, dtype=object)
    codes, uniques = pd.factorize(keys, sort=False)

    if smiles is not None:
        # Union DOI groups joined by a shared complex.  Union-find over the
        # DOI<->SMILES bipartite graph; the representative becomes the group
        # key, so the loop below balances the MERGED groups.
        parent = list(range(len(uniques)))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: int, b: int) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        smi_of_group: Dict[str, int] = {}
        for row, smi in enumerate(smiles):
            s_key = str(smi)
            gi = int(codes[row])
            prev = smi_of_group.get(s_key)
            if prev is None:
                smi_of_group[s_key] = gi
            elif _find(prev) != _find(gi):
                _union(prev, gi)

        merged = np.array([_find(int(c)) for c in codes], dtype=np.int64)
        codes = pd.factorize(merged, sort=False)[0]
    return np.asarray(codes, dtype=np.int64)


def build_doi_split_folds(
    df: pd.DataFrame,
    *,
    doi_col: str = "DOI",
    smiles_col: Optional[str] = "SMILES_Ligands",
    n_folds: int = 10,
    seed: int = 0,
) -> List[Dict[str, np.ndarray]]:
    """DOI-grouped 10-fold CV: fold k is validation, everything else trains.

    Returns a list of ``n_folds`` dicts ``{"fold": k, "train_idx", "val_idx"}``
    whose indices are positional into ``df``.  Every row lands in exactly one
    validation fold, and no DOI appears on both sides of any fold.

    ``smiles_col`` additionally guarantees no COMPLEX appears on both sides —
    see :func:`assign_doi_folds`.  Without it the D-MPNN trainer, which keys
    its split on SMILES, would collapse the 279 complexes (MEASURED, 1.8% of
    the paper cohort) that span two articles and leak them across folds.
    Pass ``None`` for a DOI-only split.
    """
    if doi_col not in df.columns:
        raise KeyError(f"DOI split requires column {doi_col!r}")
    _smi = df[smiles_col].to_numpy() if smiles_col and smiles_col in df.columns else None
    fold_ids = assign_doi_folds(
        df[doi_col].to_numpy(), n_folds=n_folds, seed=seed, smiles=_smi
    )
    folds: List[Dict[str, np.ndarray]] = []
    for k in range(int(n_folds)):
        val_idx = np.flatnonzero(fold_ids == k)
        train_idx = np.flatnonzero(fold_ids != k)
        folds.append({"fold": k, "train_idx": train_idx, "val_idx": val_idx})
    return folds


__all__ = [
    "MetalCytotoxDataset",
    "CytotoxFilter",
    "NON_CANCER_CELL_LINES",
    "PAPER_DEDUP_KEYS",
    "paper_cohort_mask",
    "build_paper_cohort",
    "build_paper_cohort_indices",
    "assign_doi_folds",
    "build_doi_split_folds",
]
