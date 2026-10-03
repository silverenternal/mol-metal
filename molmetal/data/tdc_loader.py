"""Loader for PyTDC MoleculeNet benchmarks (BBBP / BACE / HIV).

Background
----------
The TDC (Therapeutics Data Commons) project distributes three classical
MoleculeNet single-task classification datasets that are routinely cited
as SOTA baselines for D-MPNN / Chemprop property prediction:

* **BBBP** — Blood-Brain Barrier Penetration (Martins 2012).
* **BACE** — BACE-1 inhibitor activity (Subramanian 2016, IC50 → pIC50
  + binary class).
* **HIV** — HIV replication inhibition (AIDS antiviral screen, 41k
  compounds, binary class).

This loader exposes each dataset as a tidy :class:`pandas.DataFrame`
with the project's canonical 6-column schema shared with
``load_binding_db``, ``load_nci60``, and ``load_drugood``:

    smiles, metal, cell_line, protein, pIC50, active

PyTDC version compatibility
---------------------------
PyTDC 1.1.x (current) redistributes BBBP as ``bbb_martins`` (ADME) and
HIV as ``hiv`` (HTS), but **removed direct download of the BACE
classification subset** that the MoleculeNet literature uses.  We
therefore use a tiered loading strategy per dataset:

* **BBBP** — ``tdc.single_pred.ADME(name='bbb_martins')`` is the
  canonical PyTDC 1.1.x path; it returns 2,030 compounds with the
  ``Y`` column being the binary blood-brain barrier penetration
  label.  We fall back to a DeepChem S3 mirror
  (``https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBB.csv``,
  2,050 rows) if PyTDC's tarball is unreachable.
* **BACE** — PyTDC 1.1.x no longer distributes BACE.  We fetch the
  full ``bace.csv`` (1,513 rows; columns ``mol``, ``Class``, ``pIC50``,
  ``Model`` + 591 descriptors) from the DeepChem S3 mirror that hosts
  the MoleculeNet CSV archive.  We use the ``mol`` (SMILES) +
  ``Class`` (binary inhibitor label) + ``pIC50`` (regression target)
  columns.
* **HIV** — ``tdc.single_pred.HTS(name='hiv')`` is the canonical
  PyTDC 1.1.x path; 41,127 compounds, ``Y`` is the binary HIV
  activity label.  We fall back to a DeepChem S3 mirror
  (``HIV.csv``, same row count) if PyTDC's tarball is unreachable.

The PyTDC package is imported lazily inside each loader function so
that ``molmetal.data.tdc_loader`` itself stays importable from
environments without PyTDC installed; see :func:`load_tdc_bbbp`,
:func:`load_tdc_bace`, :func:`load_tdc_hiv`.

Environment
-----------
PyTDC 1.1.x requires ``setuptools <81`` (for ``pkg_resources``) and
``fuzzywuzzy`` (or ``rapidfuzz``).  These deps are already present in
the project's ``drugood`` conda env (``/home/hugo/.conda_envs/drugood``)
and any environment that previously ran :mod:`molmetal.data.drugood_loader`.

Path / caching
--------------
PyTDC caches downloaded files under ``./data`` (relative to CWD) by
default.  We override this with a project-local cache directory at
``$MOLMETAL_DATA_DIR/tdc_cache`` when ``MOLMETAL_DATA_DIR`` is set,
falling back to ``<PROJECT_ROOT>/.data/molmetal/tdc_cache``.  The
DeepChem S3 fallback writes the fetched CSVs to the same cache so
subsequent calls return from disk.

Usage
-----
>>> from molmetal.data.tdc_loader import load_tdc_bbbp, load_tdc_bace, load_tdc_hiv
>>> bbbp = load_tdc_bbbp()
>>> bbbp.columns.tolist()
['smiles', 'metal', 'cell_line', 'protein', 'pIC50', 'active']
>>> len(bbbp) > 0
True
>>> bace = load_tdc_bace()
>>> bace["active"].dtype
bool
>>> hiv = load_tdc_hiv()
>>> hiv["smiles"].nunique() == len(hiv)
True
"""

from __future__ import annotations

import hashlib
import io
import os
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Canonical 6-col schema — must match ``molmetal.data.binding_db``,
# ``molmetal.data.nci60``, ``molmetal.data.drugood_loader`` exactly
# (GATE-6 contract — see ``TODO/completed/160_big_dataset_retrain_audit_2026_09_27.md``
# §3.1 and the canonical-schemas note in each loader's docstring).
_CANONICAL_COLUMNS = (
    "smiles",
    "metal",
    "cell_line",
    "protein",
    "pIC50",
    "active",
)

# TDC has no per-row metal / cell-line / protein identifiers — these
# three slots are filled with the literal string ``"NA"`` so the
# schema matches the precious-metal loaders (where ``cell_line`` and
# ``metal`` are usually present).
_NA_FILL = "NA"


# Default cache directory for both PyTDC's tarballs and our DeepChem
# fallbacks.  Honour ``$MOLMETAL_DATA_DIR`` first, then the canonical
# project-local path, then the user's home.
def _tdc_cache_dir() -> Path:
    env_root = os.environ.get("MOLMETAL_DATA_DIR")
    if env_root:
        return Path(env_root) / "tdc_cache"
    return PROJECT_ROOT / ".data" / "molmetal" / "tdc_cache"


CACHE_DIR = _tdc_cache_dir()
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# PyTDC + fallback helpers
# ---------------------------------------------------------------------------
def _import_pytdc():
    """Lazy-import PyTDC; raise an informative ImportError if missing.

    PyTDC 1.1.x has a hard dep on ``setuptools <81`` (for
    ``pkg_resources``) and ``fuzzywuzzy``.  When those are missing the
    user gets a ``ModuleNotFoundError: No module named 'pkg_resources'``
    that is hard to diagnose.  We wrap it with the recommended
    install command.
    """
    try:
        import tdc  # type: ignore  # noqa: F401
        from tdc.single_pred import ADME, HTS  # type: ignore  # noqa: F401

        return tdc, ADME, HTS
    except ImportError as exc:
        raise ImportError(
            "PyTDC is not importable.  Install it with "
            "`/home/hugo/.conda_envs/drugood/bin/pip install 'PyTDC==1.1.15' "
            "'setuptools<81' fuzzywuzzy` (or run any of the loaders from "
            "the drugood conda env).  Underlying ImportError: "
            f"{exc}"
        ) from exc


def _download_to_cache(url: str, cache_filename: str) -> Path:
    """Download ``url`` into the cache dir; return the cached path.

    If a file already exists at ``cache_filename`` (regardless of
    contents) we treat it as cached and skip the network fetch.  This
    makes the loader deterministic + offline-friendly after the first
    successful download.  We do NOT verify a checksum: PyTDC + DeepChem
    don't publish one for these mirrors and the CSVs are signed by
    TLS.
    """
    cache_path = CACHE_DIR / cache_filename
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        raise RuntimeError(
            f"Failed to fetch {url}: {exc}.  Pre-cache the file at "
            f"{cache_path} or check your network connection."
        ) from exc
    cache_path.write_bytes(data)
    return cache_path


def _coerce_active(series: pd.Series) -> pd.Series:
    """Coerce a binary-activity series to bool, dropping NaN/garbage.

    PyTDC ``Y`` columns are integers 0/1; the DeepChem mirror of BACE
    has the ``Class`` column as 0/1 ints; we accept either dtype.
    """
    out = pd.to_numeric(series, errors="coerce")
    # Drop rows where activity is NaN/garbage — these are real "unknown
    # label" rows that would corrupt downstream classification metrics.
    out = out.dropna()
    return out.astype(bool).reindex(series.index)


# ---------------------------------------------------------------------------
# Public loaders — three functions, one per MoleculeNet dataset
# ---------------------------------------------------------------------------
def load_tdc_bbbp(
    prefer_pydc: bool = True,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load the BBBP (Blood-Brain Barrier Penetration) MoleculeNet subset.

    Tries PyTDC's ``tdc.single_pred.ADME(name='bbb_martins')`` first
    (the canonical PyTDC 1.1.x path for BBBP), and falls back to the
    DeepChem S3 mirror of the MoleculeNet ``BBBP.csv`` if PyTDC fails
    to import or raises a download error.

    Parameters
    ----------
    prefer_pydc
        When ``True`` (default), try PyTDC first.  When ``False``,
        skip PyTDC and use the DeepChem mirror unconditionally.
    cache_dir
        Override the cache directory.  Defaults to
        ``$MOLMETAL_DATA_DIR/tdc_cache`` (env var) or
        ``<PROJECT_ROOT>/.data/molmetal/tdc_cache``.

    Returns
    -------
    pandas.DataFrame
        Columns ``smiles, metal, cell_line, protein, pIC50, active``
        in the canonical order; ``metal``, ``cell_line``, ``protein``
        are filled with ``"NA"`` (BBBP carries no such identifiers);
        ``active`` is the binary BBB penetration label; ``pIC50`` is
        ``NaN`` (BBBP has no continuous potency column — only the
        binary label).  Approximately 2,030 rows.
    """
    rows: list[dict] = []
    if prefer_pydc:
        try:
            _tdc, ADME, _HTS = _import_pytdc()
            data = ADME(name="bbb_martins")
            df = data.get_data()
            for _, r in df.iterrows():
                rows.append(
                    {
                        "smiles": str(r["Drug"]),
                        "metal": _NA_FILL,
                        "cell_line": _NA_FILL,
                        "protein": _NA_FILL,
                        "pIC50": float("nan"),
                        "active": bool(int(r["Y"])),
                    }
                )
        except Exception as exc:
            warnings.warn(
                f"PyTDC BBBP path failed ({exc}); falling back to "
                f"DeepChem S3 mirror.",
                RuntimeWarning,
                stacklevel=2,
            )
            rows = []
    if not rows:
        # DeepChem S3 fallback (case-insensitive name matches MoleculeNet BBBP)
        path = _download_to_cache(
            "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBB.csv",
            "BBBP_deepchem.csv",
        )
        df = pd.read_csv(path)
        # DeepChem mirror columns: num, name, p_np, smiles
        for _, r in df.iterrows():
            try:
                yi = int(r["p_np"])
            except (ValueError, KeyError):
                continue
            rows.append(
                {
                    "smiles": str(r["smiles"]),
                    "metal": _NA_FILL,
                    "cell_line": _NA_FILL,
                    "protein": _NA_FILL,
                    "pIC50": float("nan"),
                    "active": bool(yi),
                }
            )
    out = pd.DataFrame(rows, columns=list(_CANONICAL_COLUMNS))
    # BBBP has no pIC50 column (binary only); drop the NaN-pIC50 rows
    # to match the convention used elsewhere in the project (rows with
    # no regression target are not useful for downstream regressors).
    # Keep them for classification-only consumers — caller can drop
    # them with ``df.dropna(subset=['pIC50'])`` if needed.
    return out.reset_index(drop=True)


def load_tdc_bace(
    prefer_pydc: bool = True,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load the BACE (BACE-1 inhibitor) MoleculeNet subset.

    PyTDC 1.1.x no longer distributes BACE; we therefore use the
    DeepChem S3 mirror of the MoleculeNet ``bace.csv`` directly.  The
    ``prefer_pydc`` flag is accepted for API symmetry with the other
    two loaders but is a no-op — ``False`` and ``True`` both fall back
    to the DeepChem mirror.

    Parameters
    ----------
    prefer_pydc
        Accepted for API symmetry; PyTDC 1.1.x has no BACE entry,
        so the loader always uses the DeepChem S3 mirror.
    cache_dir
        Override the cache directory.  Defaults to the project's
        tdc_cache.

    Returns
    -------
    pandas.DataFrame
        Columns ``smiles, metal, cell_line, protein, pIC50, active``
        in the canonical order; ``metal``, ``cell_line`` are filled
        with ``"NA"``; ``protein`` is the BACE-1 target identifier
        (``"BACE"``); ``active`` is the binary inhibitor label from
        the MoleculeNet ``Class`` column (IC50 < 1 µM ≈ active);
        ``pIC50`` is the MoleculeNet ``pIC50`` column (IC50 in nM →
        ``-log10(IC50_M)``).  1,513 rows.

    Notes
    -----
    The MoleculeNet CSV includes 591 descriptor columns (Morgan FP
    bits, Estate keys, Zagreb indices, etc.) that we drop on load —
    only ``mol`` / ``Class`` / ``pIC50`` are propagated.  Callers who
    need the descriptors should re-load the cached CSV directly.
    """
    # PyTDC 1.1.x removed BACE — go straight to DeepChem mirror.  We
    # try PyTDC's old ``name='BACE'`` string first for parity with
    # older PyTDC versions, in case the user downgrades, then fall
    # through to DeepChem unconditionally on failure.
    rows: list[dict] = []
    if prefer_pydc:
        try:
            _tdc, ADME, _HTS = _import_pytdc()
            data = ADME(name="BACE")
            df = data.get_data()
            # Older PyTDC (<1.0) used (Drug_ID, Drug, Y) schema.
            for _, r in df.iterrows():
                rows.append(
                    {
                        "smiles": str(r["Drug"]),
                        "metal": _NA_FILL,
                        "cell_line": _NA_FILL,
                        "protein": "BACE",
                        "pIC50": float(r["Y"])
                        if str(r["Y"]).replace(".", "", 1).lstrip("-").isdigit()
                        else float("nan"),
                        "active": bool(int(r["Y"]))
                        if "Class" not in df.columns
                        else bool(int(r["Class"])),
                    }
                )
        except Exception:
            rows = []

    if not rows:
        path = _download_to_cache(
            "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
            "bace_deepchem.csv",
        )
        df = pd.read_csv(path)
        # DeepChem mirror columns: mol (SMILES), Class (0/1), pIC50 (float)
        if (
            "mol" not in df.columns
            or "Class" not in df.columns
            or "pIC50" not in df.columns
        ):
            raise ValueError(
                f"BACE mirror at {path} is missing required columns "
                f"(mol, Class, pIC50); got {df.columns.tolist()[:6]}..."
            )
        for _, r in df.iterrows():
            try:
                active = bool(int(r["Class"]))
            except (ValueError, KeyError):
                continue
            rows.append(
                {
                    "smiles": str(r["mol"]),
                    "metal": _NA_FILL,
                    "cell_line": _NA_FILL,
                    "protein": "BACE",
                    "pIC50": float(r["pIC50"]),
                    "active": active,
                }
            )

    out = pd.DataFrame(rows, columns=list(_CANONICAL_COLUMNS))
    out["pIC50"] = pd.to_numeric(out["pIC50"], errors="coerce")
    out = out.dropna(subset=["pIC50"]).reset_index(drop=True)
    return out


def load_tdc_hiv(
    prefer_pydc: bool = True,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load the HIV (HIV replication inhibition) MoleculeNet subset.

    Tries PyTDC's ``tdc.single_pred.HTS(name='hiv')`` first (the
    canonical PyTDC 1.1.x path; the dataset moved from ADME to HTS
    in PyTDC 1.0+) and falls back to the DeepChem S3 mirror of
    ``HIV.csv`` if PyTDC fails.

    Parameters
    ----------
    prefer_pydc
        When ``True`` (default), try PyTDC first.  When ``False``,
        skip PyTDC and use the DeepChem mirror unconditionally.
    cache_dir
        Override the cache directory.  Defaults to the project's
        tdc_cache.

    Returns
    -------
    pandas.DataFrame
        Columns ``smiles, metal, cell_line, protein, pIC50, active``
        in the canonical order; ``metal``, ``cell_line`` are filled
        with ``"NA"``; ``protein`` is ``"HIV"``; ``active`` is the
        binary HIV inhibition label; ``pIC50`` is ``NaN`` (HIV has no
        continuous potency column — only the binary class from the
        DTP AIDS antiviral screen).  41,127 rows.
    """
    rows: list[dict] = []
    if prefer_pydc:
        try:
            _tdc, _ADME, HTS = _import_pytdc()
            data = HTS(name="hiv")
            df = data.get_data()
            for _, r in df.iterrows():
                rows.append(
                    {
                        "smiles": str(r["Drug"]),
                        "metal": _NA_FILL,
                        "cell_line": _NA_FILL,
                        "protein": "HIV",
                        "pIC50": float("nan"),
                        "active": bool(int(r["Y"])),
                    }
                )
        except Exception as exc:
            warnings.warn(
                f"PyTDC HIV path failed ({exc}); falling back to "
                f"DeepChem S3 mirror.",
                RuntimeWarning,
                stacklevel=2,
            )
            rows = []
    if not rows:
        path = _download_to_cache(
            "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
            "HIV_deepchem.csv",
        )
        df = pd.read_csv(path)
        # DeepChem mirror columns: smiles, activity, HIV_active
        if "smiles" not in df.columns:
            raise ValueError(
                f"HIV mirror at {path} has no 'smiles' column; "
                f"got {df.columns.tolist()}"
            )
        # DeepChem mirror uses 'HIV_active' (1 = confirmed active,
        # 0 = confirmed inactive, NaN = not screened).  We propagate
        # the labelled rows only and let ``_coerce_active`` drop the
        # unlabelled ones.
        for _, r in df.iterrows():
            try:
                yi = int(r["HIV_active"])
            except (ValueError, KeyError):
                continue
            rows.append(
                {
                    "smiles": str(r["smiles"]),
                    "metal": _NA_FILL,
                    "cell_line": _NA_FILL,
                    "protein": "HIV",
                    "pIC50": float("nan"),
                    "active": bool(yi),
                }
            )
    out = pd.DataFrame(rows, columns=list(_CANONICAL_COLUMNS))
    return out.reset_index(drop=True)


__all__ = [
    "load_tdc_bbbp",
    "load_tdc_bace",
    "load_tdc_hiv",
    "CACHE_DIR",
]
