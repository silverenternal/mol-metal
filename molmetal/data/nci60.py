"""Loader for the NCI-60 precious-metal cytotoxicity subset.

The on-disk CSV is ``molmetal/data/nci60_platinai_link.csv``: a 7,823-row
join of MetalCytoToxDB Ru/Ir/Rh/Os/Re compounds against the NCI-60
panel (one row per compound × cell-line for the 9 precious-metal-active
cell lines).  Source columns use the ``_precious`` suffix to disambiguate
from any future ``platinai_*`` columns; this loader normalises them to
the **canonical 6-column schema** shared with ``load_binding_db`` and
``load_drugood``:

    smiles, metal, cell_line, protein, pIC50, active

The two extra columns (``metal`` and ``protein``) are filled with the
literal string ``"NA"`` because the NCI-60 precious-metal CSV carries no
protein-target identifier (it's a cell-line panel, not a target panel)
and no per-row metal symbol (the metal lives in the parent MetalCytoToxDB
row that was used to build the link CSV, not in the link itself).  We
keep the 6-col schema for cross-loader concatenation in
``molmetal/scripts/pre_featurize_dmpnn.py`` and similar pipelines.

Usage
-----
>>> from molmetal.data.nci60 import load_nci60
>>> df = load_nci60()
>>> df.columns.tolist()
['smiles', 'metal', 'cell_line', 'protein', 'pIC50', 'active']
>>> len(df)
7823
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_CSV = PROJECT_ROOT / "molmetal" / "data" / "nci60_platinai_link.csv"

_REQUIRED_COLUMNS = ("smiles", "metal", "cell_line", "protein", "pIC50", "active")

# Source columns (with the ``_precious`` disambiguation suffix) get
# renamed to the canonical 6-col schema; ``metal`` and ``protein`` are
# not present in the source CSV and are filled with this NA literal.
_NA_FILL = "NA"


def load_nci60(path: Optional[Path] = None) -> pd.DataFrame:
    """Load NCI-60 precious-metal rows with canonical 6-column schema.

    Parameters
    ----------
    path
        Optional explicit CSV path.  Defaults to
        ``molmetal/data/nci60_platinai_link.csv`` (the project-shipped
        precious-metal subset).

    Returns
    -------
    pandas.DataFrame
        Columns ``smiles, metal, cell_line, protein, pIC50, active``
        (always exactly these, in this order).  ``cell_line`` holds the
        NCI-60 cell-line identifier (e.g. ``MCF7``); ``metal`` and
        ``protein`` are filled with the literal string ``"NA"`` because
        the source CSV carries no per-row metal or protein identifier.
        Rows with NaN ``pIC50`` are dropped.
    """
    csv_path = Path(path) if path is not None else DEFAULT_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"NCI-60 CSV not found at {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    # Normalise the source ``_precious``-suffixed columns to canonical.
    rename = {
        "cell_line_precious": "cell_line",
        "pIC50_precious": "pIC50",
        "active_precious": "active",
    }
    df = df.rename(columns=rename)

    # NCI-60 link CSV carries no per-row metal or protein identifier;
    # fill with ``"NA"`` so the 6-col schema matches ``load_binding_db``.
    if "metal" not in df.columns:
        df["metal"] = _NA_FILL
    if "protein" not in df.columns:
        df["protein"] = _NA_FILL

    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns {missing}; "
            f"present columns: {df.columns.tolist()}"
        )

    out = df[list(_REQUIRED_COLUMNS)].copy()
    out["pIC50"] = pd.to_numeric(out["pIC50"], errors="coerce")
    out = out.dropna(subset=["pIC50"]).reset_index(drop=True)
    out["active"] = out["active"].astype(bool)
    for c in ("smiles", "metal", "cell_line", "protein"):
        out[c] = out[c].astype(str)
    return out


__all__ = ["load_nci60"]
