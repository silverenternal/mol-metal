"""Loader for the protein-metal BindingDB subset.

The on-disk CSV is ``molmetal/data/protein_metal_binding.csv``: a
5,257-row metal-filtered BindingDB subset (5 d-block metals — Ru/Ir/Re/
Os/Rh — over a small number of protein targets; *not* the full ~1.07 M
BindingDB).  The source file already carries the canonical 6-column
schema (``smiles, metal, cell_line, protein, pIC50, active``), so the
loader is largely a thin pandas wrapper that coerces types and drops
rows with NaN ``pIC50``.

Usage
-----
>>> from molmetal.data.binding_db import load_binding_db
>>> df = load_binding_db()
>>> df.columns.tolist()
['smiles', 'metal', 'cell_line', 'protein', 'pIC50', 'active']
>>> len(df)
5257
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_CSV = PROJECT_ROOT / "molmetal" / "data" / "protein_metal_binding.csv"

_REQUIRED_COLUMNS = (
    "smiles",
    "metal",
    "cell_line",
    "protein",
    "pIC50",
    "active",
)


def load_binding_db(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the metal-filtered BindingDB subset with canonical schema.

    Parameters
    ----------
    path
        Optional explicit CSV path.  Defaults to
        ``molmetal/data/protein_metal_binding.csv``.

    Returns
    -------
    pandas.DataFrame
        Columns ``smiles, metal, cell_line, protein, pIC50, active``
        (always exactly these, in this order).  Rows with NaN ``pIC50``
        are dropped.
    """
    csv_path = Path(path) if path is not None else DEFAULT_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"BindingDB CSV not found at {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

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


__all__ = ["load_binding_db"]
