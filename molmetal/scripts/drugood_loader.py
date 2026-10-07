# Copyright 2026 The Mol-Metal Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DrugOOD IC50 split loader (CSV flavour, iMoLD-comparable axis splits).

This module is the *CSV* counterpart of
:mod:`molmetal.data.drugood_loader`, which reads the mmcv/DrugOOD JSON
payloads under ``.data/molmetal/drugood/ic50/official/``.  The CSV
curator output under ``.data/molmetal/drugood/ic50/csv/`` is the
artefact actually present on this host, so this module is the entry
point used by trainers and comparison harnesses.

Each ``<subset>_<axis>.csv`` file carries the same 11-column schema::

    config, split, smiles, y, assay_id, cls_label, domain_id,
    scaffold_split_id, size_split_id, protein_split_id,
    protein_family_split_id

``split`` holds the DrugOOD split label
(``train`` / ``ood_val`` / ``ood_test`` / ``iid_val`` / ``iid_test``).
Exactly one of the four ``*_split_id`` columns is populated per file —
the one matching the file's axis; the rest are ``NaN``.  The returned
``scaffold_id`` column is that populated ``*_split_id`` column, so it
is a genuine *scaffold* id only for ``axis="scaffold"``; for
``axis="size"`` it carries size-bin ids and for ``axis="assay"`` it is
all-``NA`` (assay files populate ``assay_id``, which is returned
separately).  See :data:`AXIS_ID_COLUMN` for the full mapping.

Honest framing
--------------
* ``load_drugood_split`` is **read-only**; it never mutates the CSVs.
* The ``scaffold_id`` column is axis-dependent by construction, not by
  omission.  Callers that require a scaffold id on every row must pass
  ``axis="scaffold"``; callers on the assay axis must use ``assay_id``.
* The ``protein`` axis is SBAP-only (``sbap_core_ic50_protein``); the
  LBAP subsets do not ship a ``protein`` axis file.  The ``ec50`` /
  ``ki`` / ``potency`` axes are LBAP-core-only alternate label types and
  are outside the 3x3 IC50 official grid.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

__all__ = [
    "DrugOODSplit",
    "AXIS_ID_COLUMN",
    "DEFAULT_CSV_DIR",
    "OFFICIAL_SUBSETS",
    "OFFICIAL_AXES",
    "list_official_splits",
    "is_official_split",
    "drugood_csv_path",
    "load_drugood_split",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

#: Root of the DrugOOD IC50 CSV curator output on this host.
DEFAULT_CSV_DIR = Path(
    os.environ.get(
        "DRUGOOD_IC50_CSV_DIR",
        str(PROJECT_ROOT / ".data" / "molmetal" / "drugood" / "ic50" / "csv"),
    )
)

#: Official DrugOOD curator subsets (subset stem, i.e. without axis suffix).
OFFICIAL_SUBSETS: Tuple[str, ...] = (
    "lbap_refined_ic50",
    "lbap_core_ic50",
    "sbap_core_ic50",
)

#: Split axes that have a file on disk for *all three* subsets.  These
#: are the three iMoLD-comparable axis splits (per wf-R33).
OFFICIAL_AXES: Tuple[str, ...] = ("assay", "scaffold", "size")

#: Extra axis files that exist but are not part of the 3x3 official grid.
#: ``protein`` is SBAP-only; ``ec50`` / ``ki`` / ``potency`` are LBAP-core-only
#: alternate label types (out of scope for the IC50 grid).
EXTRA_AXIS_FILES: Tuple[Tuple[str, str], ...] = (
    ("lbap_core_ic50", "ec50"),
    ("lbap_core_ic50", "ki"),
    ("lbap_core_ic50", "potency"),
    ("sbap_core_ic50", "protein"),
)

#: Which populated ``*_split_id`` column belongs to which axis.
AXIS_ID_COLUMN: Dict[str, str] = {
    "assay": "assay_id",
    "scaffold": "scaffold_split_id",
    "size": "size_split_id",
    "protein": "protein_split_id",
    "ec50": "assay_id",
    "ki": "assay_id",
    "potency": "assay_id",
}

#: Columns of the returned DataFrame, in canonical order.
OUTPUT_COLUMNS: Tuple[str, ...] = (
    "split",
    "smiles",
    "label",
    "assay_id",
    "scaffold_id",
)

#: DrugOOD split labels.
VALID_SPLITS: Tuple[str, ...] = (
    "train",
    "ood_val",
    "ood_test",
    "iid_val",
    "iid_test",
)


@dataclass(frozen=True)
class DrugOODSplit:
    """An official DrugOOD ``(subset, axis)`` split.

    Attributes:
        subset: One of :data:`OFFICIAL_SUBSETS`.
        axis: One of :data:`OFFICIAL_AXES` (or an extra axis in
            :data:`EXTRA_AXIS_FILES` when ``strict=False``).
        strict: When True (default) the constructor rejects any
            ``(subset, axis)`` pair without a CSV file on disk.  Set
            False to construct a spec without touching the filesystem.
    """

    subset: str
    axis: str
    strict: bool = True

    def __post_init__(self) -> None:
        if self.strict and not is_official_split(self.subset, self.axis):
            raise ValueError(
                f"no official DrugOOD split for subset={self.subset!r} "
                f"axis={self.axis!r}; see list_official_splits()"
            )

    @property
    def stem(self) -> str:
        """``<subset>_<axis>`` — the CSV filename stem."""
        return f"{self.subset}_{self.axis}"

    @property
    def id_column(self) -> str:
        """Source column in the CSV backing :attr:`scaffold_id`."""
        return AXIS_ID_COLUMN.get(self.axis, "scaffold_split_id")

    def path(self, csv_dir: Path | str = DEFAULT_CSV_DIR) -> Path:
        return Path(csv_dir) / f"{self.stem}.csv"

    def exists(self, csv_dir: Path | str = DEFAULT_CSV_DIR) -> bool:
        return self.path(csv_dir).is_file()


def _all_pairs() -> List[Tuple[str, str]]:
    pairs = [(s, a) for s in OFFICIAL_SUBSETS for a in OFFICIAL_AXES]
    pairs.extend(EXTRA_AXIS_FILES)
    return pairs


def list_official_splits(csv_dir: Path | str = DEFAULT_CSV_DIR) -> List[str]:
    """Return the on-disk ``<subset>_<axis>`` stems, official grid first.

    The 3x3 official grid (3 subsets x 3 axes) comes first, followed by
    any extra axis files that exist on disk.  Stems whose CSV is absent
    are omitted, so this is a filesystem-truth listing.
    """
    csv_dir = Path(csv_dir)
    found: List[str] = []
    for subset, axis in _all_pairs():
        stem = f"{subset}_{axis}"
        if (csv_dir / f"{stem}.csv").is_file():
            found.append(stem)
    return found


def is_official_split(
    subset: str, axis: str, csv_dir: Path | str = DEFAULT_CSV_DIR
) -> bool:
    """True iff ``(subset, axis)`` is a known split with a CSV on disk."""
    return (Path(csv_dir) / f"{subset}_{axis}.csv").is_file()


def drugood_csv_path(
    subset: str, axis: str, csv_dir: Path | str = DEFAULT_CSV_DIR
) -> Path:
    """Resolve the CSV path for ``(subset, axis)``, raising if absent."""
    path = Path(csv_dir) / f"{subset}_{axis}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"DrugOOD CSV not found: {path}; "
            f"available stems: {list_official_splits(csv_dir)}"
        )
    return path


@functools.lru_cache(maxsize=16)
def _load_cached(path_str: str) -> pd.DataFrame:
    """Read one CSV and project it to :data:`OUTPUT_COLUMNS` (cached)."""
    df = pd.read_csv(path_str, low_memory=False)
    missing = {"split", "smiles", "y", "assay_id"} - set(df.columns)
    if missing:
        raise ValueError(f"{path_str}: missing required columns {sorted(missing)}")

    out = pd.DataFrame(
        {
            "split": df["split"].astype("string"),
            "smiles": df["smiles"].astype("string"),
            "label": pd.to_numeric(df["y"], errors="coerce"),
            "assay_id": df["assay_id"].astype("string"),
        }
    )

    # scaffold_id comes from whichever *_split_id column this axis
    # populates; the rest are NaN by construction (see module docstring).
    id_col = None
    for cand in ("scaffold_split_id", "size_split_id",
                 "protein_split_id", "protein_family_split_id"):
        if cand in df.columns and df[cand].notna().any():
            id_col = cand
            break
    out["scaffold_id"] = (
        df[id_col].astype("string") if id_col is not None else pd.Series(pd.NA, index=df.index, dtype="string")
    )
    return out[list(OUTPUT_COLUMNS)]


def load_drugood_split(
    subset: str,
    axis: str,
    csv_dir: Path | str = DEFAULT_CSV_DIR,
) -> pd.DataFrame:
    """Load one official DrugOOD IC50 split as a canonical DataFrame.

    Args:
        subset: One of :data:`OFFICIAL_SUBSETS` (e.g. ``"lbap_refined_ic50"``).
        axis: One of :data:`OFFICIAL_AXES` (e.g. ``"assay"``), or an extra
            axis in :data:`EXTRA_AXIS_FILES`.
        csv_dir: Override the CSV root (defaults to :data:`DEFAULT_CSV_DIR`,
            overridable via the ``DRUGOOD_IC50_CSV_DIR`` env var).

    Returns:
        ``pandas.DataFrame`` with columns ``split, smiles, label,
        assay_id, scaffold_id`` — one row per CSV record, all splits
        concatenated (filter on ``split`` to isolate train / val / test).
        ``scaffold_id`` is the axis's own ``*_split_id`` column: a
        scaffold id for ``axis="scaffold"``, size-bin ids for
        ``axis="size"``, all-``NA`` for ``axis="assay"``.

    Raises:
        FileNotFoundError: if the CSV for ``(subset, axis)`` is absent.
    """
    path = drugood_csv_path(subset, axis, csv_dir)
    return _load_cached(str(path)).copy()


if __name__ == "__main__":  # pragma: no cover - manual smoke surface
    import argparse

    ap = argparse.ArgumentParser(description="List / load DrugOOD IC50 splits")
    ap.add_argument("--csv-dir", default=str(DEFAULT_CSV_DIR))
    ap.add_argument("--subset", default=None)
    ap.add_argument("--axis", default="assay")
    args = ap.parse_args()

    if args.subset is None:
        for stem in list_official_splits(args.csv_dir):
            print(stem)
    else:
        d = load_drugood_split(args.subset, args.axis, args.csv_dir)
        print(f"{args.subset}/{args.axis} shape={d.shape}")
        print(d["split"].value_counts().to_string())
