"""Loader for the MetalCytoToxDB anticancer cytotoxicity dataset.

Returns a tidy :class:`pandas.DataFrame` with the canonical columns
``smiles, metal, pIC50, active`` from the project-shipped MetalCytoToxDB
CSV.  Source CSV uses the column names from the original release
(``SMILES_Ligands, Metal, IC50_Dark_value``); the LOMO subset
(``molmetal/data/metal_lomo.csv``) carries pre-computed ``pIC50`` and
``active`` columns.  This loader normalises the SMILES column to
``smiles`` and either copies the pre-computed ``pIC50``/``active``
columns or derives them from ``IC50_Dark_value``.

Path resolution
---------------
The loader tries, in order:

1. ``$MOLMETAL_DATA_DIR/MetalCytoToxDB.csv`` (env override used by
   :mod:`molmetal.data.cytotox`).
2. ``<PROJECT_ROOT>/.data/molmetal/MetalCytoToxDB.csv`` (the canonical
   dev-box location; 26,801 rows).
3. ``/mnt/storage/data/molmetal/MetalCytoToxDB.csv`` (production
   default from :mod:`molmetal.data.cytotox`).
4. ``<PROJECT_ROOT>/molmetal/data/metal_lomo.csv`` (local 5,460-row
   LOMO subset, last-resort fallback when the full release is absent).

Usage
-----
>>> from molmetal.data.metal_cytotox import load_metal_cytotox
>>> df = load_metal_cytotox()
>>> df.columns.tolist()
['smiles', 'metal', 'pIC50', 'active']
>>> len(df)
26801
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_CANDIDATES: tuple[Path, ...] = (
    # 2. project-local canonical store (full release — preferred)
    PROJECT_ROOT / ".data" / "molmetal" / "MetalCytoToxDB.csv",
    # 3. production /mnt default (matches molmetal.data.cytotox)
    Path("/mnt/storage/data/molmetal/MetalCytoToxDB.csv"),
    # 4. local LOMO subset — fallback only when full file absent
    PROJECT_ROOT / "molmetal" / "data" / "metal_lomo.csv",
)

_REQUIRED_COLUMNS = ("smiles", "metal", "pIC50", "active")
_ACTIVE_THRESHOLD_UM = 10.0  # active := IC50_Dark_value < 10 µM

# ---------------------------------------------------------------------------
# Wf-323B / Wf-327F2 (Wf-A16 wire-in) — v3/v4 column-synonym map.
#
# The original release (v3) uses uppercase ``SMILES_Ligands`` / ``Metal``;
# the persisted multi-task CSVs (v4) carry BOTH the lowercase canonical AND
# the uppercase provenance columns.  Renaming the uppercase → lowercase on a
# v4 frame would collide with the already-present lowercase column and raise
# ``ValueError: Buffer has wrong number of dimensions``.  The helpers below
# make the precedence explicit: the lowercase canonical form is always first
# in each alias tuple, so :func:`_has_column` prefers it and the rename skips
# when the target already exists.
#
# Keys are the canonical output columns; each value is an ordered tuple of
# source aliases with the preferred (lowercase) form first.
_COLUMN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "smiles": ("smiles", "SMILES_Ligands", "smiles_ligands"),
    "metal": ("metal", "Metal"),
    "pIC50": ("pic50", "pIC50", "IC50_Dark_value"),
    "active": ("active", "activity"),
}


def _has_column(df: pd.DataFrame, canonical: str) -> Optional[str]:
    """Return the first present alias for *canonical* in *df*, else ``None``.

    Iterates ``_COLUMN_SYNONYMS[canonical]`` in order (preferred lowercase
    form first) so a v4 frame with both ``smiles`` and ``SMILES_Ligands``
    resolves to ``smiles`` — avoiding the duplicate-column collision.
    """
    aliases = _COLUMN_SYNONYMS.get(canonical, (canonical,))
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


def _rename_v3_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Rename v3-style uppercase provenance columns to canonical names.

    Behaviour (Wf-323B contract):

    * Only the two v3 provenance columns that must be normalised for the
      canonical contract are renamed: ``SMILES_Ligands`` → ``smiles`` and
      ``Metal`` → ``metal``.  The pIC50/active handling stays in the loader
      (it *derives* rather than renames), so ``pic50`` / ``activity`` are
      NEVER touched here — that is what makes the v4-like frame a no-op.
    * When the lowercase canonical column is ALREADY present (v4 path), that
      column's rename is skipped — no duplicate is introduced.
    """
    rename: dict[str, str] = {}
    for canonical, v3_alias in (("smiles", "SMILES_Ligands"), ("metal", "Metal")):
        if canonical in df.columns:
            # Lowercase canonical already present — never rename over it.
            continue
        if v3_alias in df.columns:
            rename[v3_alias] = canonical
    if not rename:
        return df
    return df.rename(columns=rename)


def _resolve_csv() -> Path:
    """Return the first existing CSV path.

    On total miss the loader raises FileNotFoundError with all tried
    paths listed.  ``$MOLMETAL_DATA_DIR`` is honoured before the
    project-local candidates when set.
    """
    candidates: list[Path] = list(_CANDIDATES)
    env_root = os.environ.get("MOLMETAL_DATA_DIR")
    if env_root:
        candidates.insert(0, Path(env_root) / "MetalCytoToxDB.csv")

    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "MetalCytoToxDB.csv not found.  Tried:\n  - "
        + "\n  - ".join(str(c) for c in candidates)
    )


def _derive_pic50(df: pd.DataFrame) -> pd.Series:
    """Compute pIC50 = -log10(IC50_Dark_value * 1e-6) safely.

    Drops NaN inputs; ``pd.to_numeric`` already coerces non-numeric
    entries to NaN.
    """
    ic50_um = pd.to_numeric(df["IC50_Dark_value"], errors="coerce")
    ic50_m = np.clip(ic50_um.to_numpy(dtype=float) * 1e-6, 1e-12, None)
    return pd.Series(-np.log10(ic50_m), index=df.index, dtype=np.float32)


def _derive_active(df: pd.DataFrame) -> pd.Series:
    """Compute active := IC50_Dark_value < 10 µM."""
    ic50_um = pd.to_numeric(df["IC50_Dark_value"], errors="coerce")
    return (ic50_um < _ACTIVE_THRESHOLD_UM).astype(bool)


def load_metal_cytotox(path: Optional[Path] = None) -> pd.DataFrame:
    """Load MetalCytoToxDB and return the canonical 4-column DataFrame.

    Parameters
    ----------
    path
        Optional explicit CSV path.  When ``None`` (default) the loader
        walks :data:`_CANDIDATES` in priority order and returns the
        first existing file.

    Returns
    -------
    pandas.DataFrame
        Columns ``smiles, metal, pIC50, active`` (always exactly these,
        in this order).  ``pIC50`` is ``float64``; ``active`` is
        ``bool``.  Rows with NaN ``pIC50`` are dropped to match the
        convention used elsewhere in the project.
    """
    csv_path = Path(path) if path is not None else _resolve_csv()
    df = pd.read_csv(csv_path, low_memory=False)

    # Column normalisation via the v3/v4 synonym map (Wf-323B).  The
    # original release (v3) uses ``SMILES_Ligands`` / ``Metal``; the v4 CSVs
    # carry BOTH those uppercase columns AND the lowercase canonical ones —
    # ``_rename_v3_to_canonical`` skips any canonical column already present,
    # so the v4 collision (duplicate ``smiles``) cannot occur.
    df = _rename_v3_to_canonical(df)

    if "smiles" not in df.columns or "metal" not in df.columns:
        raise ValueError(
            f"{csv_path} has no SMILES/metal columns; "
            f"present columns: {df.columns.tolist()}"
        )

    # Prefer pre-computed pIC50/active when present (LOMO subset);
    # otherwise derive them from IC50_Dark_value.
    if "pIC50" in df.columns:
        out_pic50 = pd.to_numeric(df["pIC50"], errors="coerce")
    elif "IC50_Dark_value" in df.columns:
        out_pic50 = _derive_pic50(df)
    else:
        raise ValueError(
            f"{csv_path} is missing both pIC50 and IC50_Dark_value; "
            f"present columns: {df.columns.tolist()}"
        )

    if "active" in df.columns:
        out_active = df["active"].astype(bool)
    elif "IC50_Dark_value" in df.columns:
        out_active = _derive_active(df)
    else:
        raise ValueError(
            f"{csv_path} is missing both active and IC50_Dark_value; "
            f"present columns: {df.columns.tolist()}"
        )

    out = pd.DataFrame(
        {
            "smiles": df["smiles"].astype(str),
            "metal": df["metal"].astype(str),
            "pIC50": out_pic50,
            "active": out_active,
        }
    )
    out = out.dropna(subset=["pIC50"]).reset_index(drop=True)
    return out


__all__ = [
    "load_metal_cytotox",
    "_COLUMN_SYNONYMS",
    "_has_column",
    "_rename_v3_to_canonical",
]
