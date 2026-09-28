"""Loader for the DrugOOD (Ji et al. NeurIPS 2022) ChEMBL-29 IC50 subset.

The DrugOOD library (https://github.com/tencent-ailab/DrugOOD) ships a
``DrugOODDataset`` mmcv-style dataset class.  Its on-disk artefact is a
JSON file at
``${DRUGOOD_IC50_TARGET_ROOT}/<subset>.json`` containing a
``{"split": {"train": [...], "ood_val": [...], ...}}`` payload, where
each per-row dict has keys ``smiles, reg_label, assay_id, cls_label,
domain_id``.  This module:

  1. Builds a ``DrugOODDataset`` from the configured subset.
  2. Flattens the split we want into the canonical 6-column schema
     (``smiles, metal, cell_line, protein, pIC50, active``) shared with
     ``molmetal.data.binding_db`` and ``molmetal.data.nci60``.
  3. Casts ``reg_label`` to ``pIC50`` (float), ``cls_label`` to a Python
     ``bool``, and exposes ``assay_id`` as the ``protein`` proxy column
     (DrugOOD carries no per-row metal / cell-line identifier — those
     columns are filled with the literal string ``"NA"``).

This loader is the canonical entry-point for *external* DrugOOD data
(ChEMBL-29 IC50 assay panel).  For the project's curated
``molmetal/data/drugood/train.csv`` (a 296,676-row upstream-format
copy used by the in-repo multi-task featuriser), see
``molmetal/data/four_domain_combined.py``.

Environment / Python
--------------------
The DrugOOD library requires ``mmcv==1.5.0`` and a Torch stack that
may not match the project's primary Python env.  This module therefore
imports ``drugood`` lazily (only inside :func:`load_drugood`) so that
the rest of ``molmetal.data`` stays importable from environments
without DrugOOD installed.  Callers are expected to invoke
:func:`load_drugood` from the ``drugood`` conda env
(``/home/hugo/.conda_envs/drugood/bin/python``) when running end-to-end.

Usage
-----
>>> from molmetal.data.drugood_loader import load_drugood
>>> df = load_drugood(split="train", subset="lbap_refined_ic50_assay")
>>> df.columns.tolist()
['smiles', 'metal', 'cell_line', 'protein', 'pIC50', 'active']
>>> len(df) > 0
True
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Default location of DrugOOD's official IC50 curator output.  This is
# the path used by the upstream DrugOOD curator configs (see
# ``${DRUGOOD_ROOT}/configs/_base_/curators/lbap_defaults.py``).
DEFAULT_DRUGOOD_TARGET_ROOT = (
    PROJECT_ROOT / ".data" / "molmetal" / "drugood" / "ic50" / "official"
)

# The three "refined" curator outputs.  These are Ji et al.'s
# strictest-curation splits and the recommended entry-point for any
# regression / classifier comparison (see TODO-pending/99_drugood_loader.md).
DEFAULT_SUBSETS = (
    "lbap_refined_ic50_assay",
    "lbap_refined_ic50_scaffold",
    "lbap_refined_ic50_size",
)

# Splits available in the official DrugOOD JSON payloads.
VALID_SPLITS = ("train", "ood_val", "ood_test", "iid_val", "iid_test")

# Canonical 6-col schema — must match ``molmetal.data.binding_db`` and
# ``molmetal.data.nci60`` exactly (GATE-6 contract).
_CANONICAL_COLUMNS = (
    "smiles",
    "metal",
    "cell_line",
    "protein",
    "pIC50",
    "active",
)

# DrugOOD carries no per-row metal / cell-line identifier (it is a
# ligand-centric assay panel).  We fill these with the literal string
# ``"NA"`` so the schema matches ``load_binding_db`` / ``load_nci60``.
_NA_FILL = "NA"


def _resolve_subset_path(
    subset: str,
    target_root: Optional[Path] = None,
) -> Path:
    """Return the JSON path for ``subset`` under ``target_root``.

    Falls back to the ``DRUGOOD_IC50_TARGET_ROOT`` env var if
    ``target_root`` is not given explicitly.
    """
    if target_root is None:
        env = os.environ.get("DRUGOOD_IC50_TARGET_ROOT")
        target_root = Path(env) if env else DEFAULT_DRUGOOD_TARGET_ROOT
    target_root = Path(target_root)
    return target_root / f"{subset}.json"


def _load_via_drugood_lib(
    subset: str,
    split: str,
    target_root: Optional[Path],
) -> pd.DataFrame:
    """Build a DrugOODDataset via the upstream library and flatten it.

    The DrugOOD library requires mmcv + the project's installed DrugOOD
    tree; this helper raises an informative ImportError if either is
    missing.
    """
    try:
        from drugood.datasets import DrugOODDataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "drugood is not importable in this Python environment. "
            "Re-run from the drugood conda env "
            "('/home/hugo/.conda_envs/drugood/bin/python') or "
            "`pip install -e /home/hugo/codes/mol-metal/.data/molmetal/DrugOOD`. "
            f"Underlying ImportError: {exc}"
        ) from exc

    json_path = _resolve_subset_path(subset, target_root)
    if not json_path.exists():
        raise FileNotFoundError(
            f"DrugOOD subset JSON not found at {json_path}. "
            f"Run the upstream DrugOOD curator first "
            f"(see {PROJECT_ROOT}/.data/molmetal/DrugOOD/configs/curators/"
            f"lbap_refined_ic50_assay.py)."
        )

    # DrugOODDataset.__init__ accepts an empty pipeline list (no
    # transforms), an ``ann_file`` pointing at the JSON, a ``split``
    # selector, and a ``test_mode`` flag.
    ds = DrugOODDataset(
        ann_file=str(json_path),
        pipeline=[],
        split=split,
        test_mode=(split != "train"),
    )

    # Flatten ``data_infos`` (list[dict]) into the canonical schema.
    rows: list[dict] = []
    for info in ds.data_infos:
        # ``reg_label`` is the raw pIC50 (DrugOOD's IC50 measurement,
        # negative-log10 of the molar IC50 — same units as our other
        # loaders' ``pIC50``).
        # ``cls_label`` is the binary activity label (1 = active, 0 =
        # inactive) at the curator's classification threshold (default
        # pIC50 >= 5).
        rows.append(
            {
                "smiles": str(info["smiles"]),
                "metal": _NA_FILL,
                "cell_line": _NA_FILL,
                "protein": str(info["assay_id"]),
                "pIC50": float(info["reg_label"]),
                "active": bool(int(info["cls_label"])),
            }
        )

    out = pd.DataFrame(rows, columns=list(_CANONICAL_COLUMNS))
    out["pIC50"] = pd.to_numeric(out["pIC50"], errors="coerce")
    out = out.dropna(subset=["pIC50"]).reset_index(drop=True)
    return out


def _load_via_raw_json(
    subset: str,
    split: str,
    target_root: Optional[Path],
) -> pd.DataFrame:
    """Fallback path that reads the DrugOOD JSON directly.

    Used when the ``drugood`` library is not importable in the current
    Python environment (e.g. when running under ``.venv-vlm``).  The
    JSON layout is identical to what ``DrugOODDataset.load_annotations``
    consumes — ``{"split": {"train": [...], "ood_val": [...], ...}}``
    where each list element is the same
    ``{smiles, reg_label, assay_id, cls_label, domain_id}`` dict.
    """
    json_path = _resolve_subset_path(subset, target_root)
    if not json_path.exists():
        raise FileNotFoundError(
            f"DrugOOD subset JSON not found at {json_path}. "
            f"Set DRUGOOD_IC50_TARGET_ROOT or pass target_root explicitly."
        )

    with open(json_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if "split" not in payload or split not in payload["split"]:
        raise ValueError(
            f"{json_path} has no '{split}' split; "
            f"available splits: {list(payload.get('split', {}).keys())}"
        )

    rows: list[dict] = []
    for info in payload["split"][split]:
        rows.append(
            {
                "smiles": str(info["smiles"]),
                "metal": _NA_FILL,
                "cell_line": _NA_FILL,
                "protein": str(info["assay_id"]),
                "pIC50": float(info["reg_label"]),
                "active": bool(int(info["cls_label"])),
            }
        )

    out = pd.DataFrame(rows, columns=list(_CANONICAL_COLUMNS))
    out["pIC50"] = pd.to_numeric(out["pIC50"], errors="coerce")
    out = out.dropna(subset=["pIC50"]).reset_index(drop=True)
    return out


def load_drugood(
    subset: str = "lbap_refined_ic50_assay",
    split: str = "train",
    target_root: Optional[Path] = None,
    prefer_library: bool = True,
) -> pd.DataFrame:
    """Load a DrugOOD IC50 split into the canonical 6-column schema.

    Parameters
    ----------
    subset
        One of the curator subset names: ``lbap_refined_ic50_assay``,
        ``lbap_refined_ic50_scaffold``, ``lbap_refined_ic50_size``.
        Defaults to ``lbap_refined_ic50_assay`` (the assay-domain OOD
        split that the project's multi-task featuriser benchmarks on).
    split
        One of ``train``, ``ood_val``, ``ood_test``, ``iid_val``,
        ``iid_test``.  Defaults to ``train``.
    target_root
        Directory containing the curator's ``<subset>.json`` outputs.
        Defaults to ``${DRUGOOD_IC50_TARGET_ROOT}`` env var if set,
        else
        ``molmetal/.data/molmetal/drugood/ic50/official/``.
    prefer_library
        When ``True`` (default), build the upstream
        ``drugood.datasets.DrugOODDataset`` instance.  When ``False``,
        skip the library import and read the JSON payload directly —
        useful when running under a Python env that lacks mmcv.

    Returns
    -------
    pandas.DataFrame
        Columns ``smiles, metal, cell_line, protein, pIC50, active``
        (always exactly these, in this order).  ``metal`` and
        ``cell_line`` are filled with the literal string ``"NA"``
        because DrugOOD carries no per-row metal / cell-line
        identifier; ``protein`` holds the assay id (stringified) as a
        proxy for the bioassay target.  Rows with NaN ``pIC50`` are
        dropped.
    """
    if subset not in DEFAULT_SUBSETS:
        warnings.warn(
            f"subset={subset!r} is not one of the three 'refined' "
            f"curator outputs {DEFAULT_SUBSETS}; loader will still try "
            f"to read ${subset}.json from target_root."
        )
    if split not in VALID_SPLITS:
        raise ValueError(
            f"split={split!r} is not valid; choose one of {VALID_SPLITS}"
        )

    if prefer_library:
        try:
            return _load_via_drugood_lib(subset, split, target_root)
        except ImportError:
            # Library not installed in this env — fall back to raw JSON
            # so smoke tests can still run from a non-DrugOOD Python.
            return _load_via_raw_json(subset, split, target_root)
    return _load_via_raw_json(subset, split, target_root)


__all__ = [
    "load_drugood",
    "DEFAULT_DRUGOOD_TARGET_ROOT",
    "DEFAULT_SUBSETS",
    "VALID_SPLITS",
]
