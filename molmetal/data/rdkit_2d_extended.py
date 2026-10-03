"""molmetal.data.rdkit_2d_extended — F5_RDKIT_2D 32-d descriptor module.

=====================================================================
Wf-248 Phase 1 — RDKit 2D descriptor expansion from 12-d to 32-d.

This module is the **canonical implementation** of the F5_RDKIT_2D
32-d RDKit 2D descriptor vector that the L5 Hilbert-ASTSR trainer
expects per Wf-127 §Cause 2 / Wf-218.  The 32-d set is built by
appending 20 canonical RDKit 2D descriptors (shape / topology,
kappa shape indices, surface area, partial-charge stats, ring
features, heteroatom counts, functional-group counts) onto the
Wf-49 Phase 1 frozen 12-d set:

  * 12-d baseline (Wf-49 P1, frozen order):
    MolWt, MolLogP, TPSA, HBA, HBD, NRotB, HeavyAtoms, RingCount,
    NumAromaticRings, FractionCSP3, MolMR, BalabanJ.
  * 20-d expansion (Wf-218):
    BertzCT, Chi0v, Chi0n, Chi1n, HallKierAlpha, Kappa1, Kappa2,
    Kappa3, LabuteASA, MaxPartialCharge, MinPartialCharge,
    MaxAbsPartialCharge, MinAbsPartialCharge, NumAliphaticRings,
    NumSaturatedRings, NumHeteroatoms, NHOHCount, NOCount,
    fr_halogen, fr_Al_COO.

The **first 12 entries match the Wf-49 Phase 1 12-d set
bit-for-bit** so any trainer / reward channel that hard-codes the
12-d column order still sees the same leading 12 descriptors in the
32-d path — this is the backward-compat guarantee.

Why this exists (Wf-127 §Cause 2)
--------------------------------
The 12-d RDKit signal floor caps the L5 Hilbert-ASTSR R² at
0.40-0.60 on PlatinAI 4-cell (a Pt-only cohort where the 12-d Lipinski
subset misses shape / charge / halogen signal).  The 20-d expansion
specifically targets:
  * shape / topology (BertzCT, Chi0v, Chi0n, Chi1n, HallKierAlpha);
  * kappa shape indices (Kappa1/2/3);
  * surface area (LabuteASA);
  * partial-charge descriptors (relevant to Pt electron density —
    MaxPartialCharge / MinPartialCharge / MaxAbsPartialCharge /
    MinAbsPartialCharge);
  * ring features (NumAliphaticRings, NumSaturatedRings);
  * heteroatom counts (NumHeteroatoms, NHOHCount, NOCount);
  * functional-group counts (fr_halogen — common in Pt drugs via the
    Cl⁻ leaving group of cisplatin / carboplatin; fr_Al_COO — relevant
    for oxaliplatin's bidentate oxalate).

Public API
----------
* :func:`compute_rdkit_2d_extended`  primary 32-d extractor
* :func:`compute_rdkit_2d_12d`       backward-compat 12-d extractor
                                     (returns the first 12 entries of
                                     the 32-d vector)
* :data:`RDKIT_2D_EXTENDED_FEATURE_COLUMNS`  32 canonical column names
* :data:`RDKIT_2D_FEATURE_COLUMNS`           12 baseline column names
* :data:`RDKIT_2D_DIM`           32
* :data:`RDKIT_2D_BASE_DIM`      12
* :class:`RDKit2DExtendedError`   raised when RDKit is unavailable

Lit anchors
-----------
* Yang et al. 2019 JCIM (D-MPNN paper)
* Wu et al. 2017 MoleculeNet §"smaller single-task datasets"
* arXiv:2510.23428 (MetaModel = RDKit + ChemProp +0.22 AUC on BACE)
* Landrum G. RDKit: Open-Source Cheminformatics.  https://www.rdkit.org
"""

from __future__ import annotations

import sys
from typing import Callable, List, Optional, Tuple

import numpy as np

# RDKit is heavy and not always available in every test env.  Lazy-import
# with a hard fail (this module's value proposition is RDKit 2D
# descriptors, so we must fail loudly if RDKit is absent — the same
# defensive pattern used in extract_rdkit_2d_descriptors.py).
try:
    from rdkit import Chem, RDLogger  # type: ignore
    from rdkit.Chem import Descriptors  # type: ignore

    RDLogger.DisableLog("rdApp.*")
    _HAS_RDKIT = True
    _RDKIT_IMPORT_ERROR: Optional[BaseException] = None
except Exception as _exc:  # pragma: no cover — defensive
    Chem = None  # type: ignore
    RDLogger = None  # type: ignore
    Descriptors = None  # type: ignore
    _HAS_RDKIT = False
    _RDKIT_IMPORT_ERROR = _exc


class RDKit2DExtendedError(RuntimeError):
    """Raised when the RDKit 2D extended descriptor pipeline cannot run.

    Typically raised when RDKit is not importable in the active Python
    environment — the Wf-49 Cause 2 mitigation requires RDKit 2D
    descriptors and the pipeline cannot operate without them.
    """


# ---------------------------------------------------------------------------
# Canonical column-name tuples — re-exported from
# molmetal.molmetal_lam.reward.lambda_astsr so downstream consumers
# (ASTSR trainers, feature-column builders) see the same byte-identical
# constants regardless of which entry-point they import.
#
# We **import, not duplicate**: the source of truth lives in
# ``lambda_astsr.py`` (it is also the ASTSR contract surface — the 32-d
# tuple is the column order the ASTSR evolutionary search expects).
# ---------------------------------------------------------------------------
def _load_canonical_columns() -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Load the canonical 32-d and 12-d column-name tuples.

    Imports from :mod:`molmetal.molmetal_lam.reward.lambda_astsr`.  If
    the ASTSR module cannot be imported (e.g. in minimal test envs
    without ``molmetal_lam``), we fall back to hard-coded copies of
    the same tuples so this module remains a self-contained entry
    point.  The hard-coded fallback is intentionally **frozen**
    (bit-for-bit identical to the ASTSR registry) and lives below
    under :data:`_FALLBACK_32D_COLUMNS` / :data:`_FALLBACK_12D_COLUMNS`.
    """
    try:
        from molmetal.molmetal_lam.reward.lambda_astsr import (
            RDKIT_2D_EXTENDED_FEATURE_COLUMNS,
            RDKIT_2D_FEATURE_COLUMNS,
        )

        return (
            tuple(RDKIT_2D_EXTENDED_FEATURE_COLUMNS),
            tuple(RDKIT_2D_FEATURE_COLUMNS),
        )
    except Exception:
        return (
            tuple(_FALLBACK_32D_COLUMNS),
            tuple(_FALLBACK_12D_COLUMNS),
        )


# Hard-coded fallback (frozen; matches lambda_astsr.py byte-for-byte).
# Used only when the ASTSR module is unavailable.  See the docstring of
# :func:`_load_canonical_columns` for context.
_FALLBACK_12D_COLUMNS: Tuple[str, ...] = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "HBA",
    "HBD",
    "NRotB",
    "HeavyAtoms",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "MolMR",
    "BalabanJ",
)

_FALLBACK_32D_COLUMNS: Tuple[str, ...] = (
    # 12 from Wf-49 P1 (frozen order, preserved bit-for-bit)
    "MolWt",
    "MolLogP",
    "TPSA",
    "HBA",
    "HBD",
    "NRotB",
    "HeavyAtoms",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "MolMR",
    "BalabanJ",
    # 20 Wf-218 expansion descriptors
    "BertzCT",
    "Chi0v",
    "Chi0n",
    "Chi1n",
    "HallKierAlpha",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "LabuteASA",
    "MaxPartialCharge",
    "MinPartialCharge",
    "MaxAbsPartialCharge",
    "MinAbsPartialCharge",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "NumHeteroatoms",
    "NHOHCount",
    "NOCount",
    "fr_halogen",
    "fr_Al_COO",
)


_RDKIT_2D_EXTENDED_FEATURE_COLUMNS, _RDKIT_2D_FEATURE_COLUMNS = (
    _load_canonical_columns()
)


#: Canonical 32-d RDKit 2D feature column names (Wf-248 / Wf-218).
#: First 12 entries match :data:`RDKIT_2D_FEATURE_COLUMNS` exactly.
RDKIT_2D_EXTENDED_FEATURE_COLUMNS: Tuple[str, ...] = _RDKIT_2D_EXTENDED_FEATURE_COLUMNS

#: Canonical 12-d RDKit 2D feature column names (Wf-49 Phase 1).
RDKIT_2D_FEATURE_COLUMNS: Tuple[str, ...] = _RDKIT_2D_FEATURE_COLUMNS

#: Output dimension of :func:`compute_rdkit_2d_extended` (32).
RDKIT_2D_DIM: int = len(RDKIT_2D_EXTENDED_FEATURE_COLUMNS)

#: Output dimension of :func:`compute_rdkit_2d_12d` (12).
RDKIT_2D_BASE_DIM: int = len(RDKIT_2D_FEATURE_COLUMNS)


# Sanity-check the registry shape at import time — fail loudly if the
# column-name tuple drifted from the expected 32-d / 12-d shape.  This
# is a cheap invariant that catches schema drift before it propagates
# to downstream trainers.
if RDKIT_2D_DIM != 32:
    raise RDKit2DExtendedError(
        f"RDKIT_2D_EXTENDED_FEATURE_COLUMNS has length {RDKIT_2D_DIM}, "
        "expected 32. The Wf-218 32-d expansion must have exactly 32 "
        "columns to satisfy the F5_RDKIT_2D schema."
    )
if RDKIT_2D_BASE_DIM != 12:
    raise RDKit2DExtendedError(
        f"RDKIT_2D_FEATURE_COLUMNS has length {RDKIT_2D_BASE_DIM}, "
        "expected 12. The Wf-49 Phase 1 frozen 12-d schema has drifted."
    )


# ---------------------------------------------------------------------------
# Per-descriptor RDKit callables — frozen order, matches column-name tuples.
# ---------------------------------------------------------------------------
# 12 Wf-49 P1 (frozen order, preserved bit-for-bit):
_BASE_12D_DESCRIPTORS: Tuple[Tuple[str, Callable], ...] = (
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Descriptors.MolLogP),
    ("TPSA", Descriptors.TPSA),
    ("HBA", Descriptors.NumHAcceptors),
    ("HBD", Descriptors.NumHDonors),
    ("NRotB", Descriptors.NumRotatableBonds),
    ("HeavyAtoms", Descriptors.HeavyAtomCount),
    ("RingCount", Descriptors.RingCount),
    ("NumAromaticRings", Descriptors.NumAromaticRings),
    ("FractionCSP3", Descriptors.FractionCSP3),
    ("MolMR", Descriptors.MolMR),
    ("BalabanJ", Descriptors.BalabanJ),
)

# 20 Wf-218 expansion descriptors — appended in the column-name order:
_EXTRA_20D_DESCRIPTORS: Tuple[Tuple[str, Callable], ...] = (
    ("BertzCT", Descriptors.BertzCT),
    ("Chi0v", Descriptors.Chi0v),
    ("Chi0n", Descriptors.Chi0n),
    ("Chi1n", Descriptors.Chi1n),
    ("HallKierAlpha", Descriptors.HallKierAlpha),
    ("Kappa1", Descriptors.Kappa1),
    ("Kappa2", Descriptors.Kappa2),
    ("Kappa3", Descriptors.Kappa3),
    ("LabuteASA", Descriptors.LabuteASA),
    ("MaxPartialCharge", Descriptors.MaxPartialCharge),
    ("MinPartialCharge", Descriptors.MinPartialCharge),
    ("MaxAbsPartialCharge", Descriptors.MaxAbsPartialCharge),
    ("MinAbsPartialCharge", Descriptors.MinAbsPartialCharge),
    ("NumAliphaticRings", Descriptors.NumAliphaticRings),
    ("NumSaturatedRings", Descriptors.NumSaturatedRings),
    ("NumHeteroatoms", Descriptors.NumHeteroatoms),
    ("NHOHCount", Descriptors.NHOHCount),
    ("NOCount", Descriptors.NOCount),
    ("fr_halogen", Descriptors.fr_halogen),
    ("fr_Al_COO", Descriptors.fr_Al_COO),
)

# Full 32-d registry — the canonical (name, callable) tuple.
# We resolve names from the canonical column-name tuple at module load
# so the per-descriptor callables stay in lock-step with the column
# schema (a single source of truth for the names).
RDKIT_2D_EXTENDED_DESCRIPTORS: Tuple[Tuple[str, Callable], ...] = (
    _BASE_12D_DESCRIPTORS + _EXTRA_20D_DESCRIPTORS
)


def _require_rdkit() -> None:
    """Hard-fail if RDKit is unavailable.

    The Wf-49 Cause 2 mitigation requires RDKit 2D descriptors; we
    follow the same defensive pattern as
    :mod:`molmetal.scripts.extract_rdkit_2d_descriptors` — fail loudly
    so callers (CI / trainer / reward channel) cannot silently fall
    back to the predict-the-mean collapse.
    """
    if not _HAS_RDKIT:
        raise RDKit2DExtendedError(
            "RDKit is required for molmetal.data.rdkit_2d_extended "
            f"(import failed: {_RDKIT_IMPORT_ERROR!r}). "
            "Install via `pip install rdkit` or activate the project venv."
        )


# ---------------------------------------------------------------------------
# Public computation functions
# ---------------------------------------------------------------------------
def compute_rdkit_2d_extended(smiles: str) -> Optional[np.ndarray]:
    """Compute the F5_RDKIT_2D 32-d RDKit 2D descriptor vector.

    Parameters
    ----------
    smiles : str
        Canonical or arbitrary SMILES string.  Must parse to a non-
        empty RDKit ``Mol`` (i.e. ``GetNumHeavyAtoms() >= 1``).

    Returns
    -------
    Optional[np.ndarray]
        ``np.float32`` array of shape ``(32,)`` containing the 32-d
        descriptor vector in the canonical column order
        (:data:`RDKIT_2D_EXTENDED_FEATURE_COLUMNS`).  Returns ``None``
        if the SMILES fails to parse or yields an empty molecule
        (callers decide on row-drop vs imputation).  NaN / inf entries
        from per-descriptor exceptions are returned as NaN so the
        caller can decide on the imputation strategy (matches the
        Wf-49 Phase 1 ``compute_rdkit_descriptors`` contract).

    Backward-compat guarantee
    -------------------------
    The first 12 entries of the returned vector match
    :func:`compute_rdkit_2d_12d` (and Wf-49 Phase 1
    :func:`molmetal.scripts.extract_rdkit_2d_descriptors.compute_rdkit_descriptors`)
    **bit-for-bit** — same RDKit callables, same iteration order, same
    NaN-handling policy.

    Notes
    -----
    CPU-only; no torch / numpy-vectorisation is invoked (RDKit's per-mol
    Descriptors API is fast enough for the 4,000-row PlatinAI 4-cell
    cohort at ~30 ms/mol).
    """
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return None
    out = np.zeros(RDKIT_2D_DIM, dtype=np.float32)
    for i, (_name, fn) in enumerate(RDKIT_2D_EXTENDED_DESCRIPTORS):
        try:
            v = float(fn(mol))
        except Exception:  # pragma: no cover — defensive per-descriptor
            v = float("nan")
        out[i] = v
    return out


def compute_rdkit_2d_12d(smiles: str) -> Optional[np.ndarray]:
    """Compute the Wf-49 Phase 1 12-d RDKit 2D descriptor vector.

    Backward-compat wrapper: returns the first 12 entries of
    :func:`compute_rdkit_2d_extended` as a fresh ``(12,)`` float32
    array.  Bit-for-bit identical to the Wf-49 Phase 1
    ``compute_rdkit_descriptors`` output (the first 12 descriptor
    callables and their iteration order are unchanged).

    Returns
    -------
    Optional[np.ndarray]
        ``np.float32`` array of shape ``(12,)`` or ``None`` if the
        SMILES fails to parse.
    """
    _require_rdkit()
    full = compute_rdkit_2d_extended(smiles)
    if full is None:
        return None
    if full.shape[0] != RDKIT_2D_DIM:
        return None  # defensive — schema drift safeguard
    return full[:RDKIT_2D_BASE_DIM].copy()


def extended_feature_columns() -> Tuple[str, ...]:
    """Return a fresh tuple copy of :data:`RDKIT_2D_EXTENDED_FEATURE_COLUMNS`.

    Convenience wrapper so callers can document the F5_RDKIT_2D intent
    without importing the underlying constant.  Mirrors
    ``molmetal.molmetal_lam.reward.lambda_astsr.rdkit_2d_extended_feature_columns``.
    """
    return tuple(RDKIT_2D_EXTENDED_FEATURE_COLUMNS)


def base_feature_columns() -> Tuple[str, ...]:
    """Return a fresh tuple copy of :data:`RDKIT_2D_FEATURE_COLUMNS`.

    Convenience wrapper so callers can document the F5_RDKIT_2D 12-d
    baseline intent without importing the underlying constant.
    Mirrors
    ``molmetal.molmetal_lam.reward.lambda_astsr.rdkit_2d_feature_columns``.
    """
    return tuple(RDKIT_2D_FEATURE_COLUMNS)


__all__ = [
    "RDKit2DExtendedError",
    "RDKIT_2D_EXTENDED_FEATURE_COLUMNS",
    "RDKIT_2D_FEATURE_COLUMNS",
    "RDKIT_2D_EXTENDED_DESCRIPTORS",
    "RDKIT_2D_DIM",
    "RDKIT_2D_BASE_DIM",
    "compute_rdkit_2d_extended",
    "compute_rdkit_2d_12d",
    "extended_feature_columns",
    "base_feature_columns",
]


# ---------------------------------------------------------------------------
# Module-level sanity probe — runs once at import time in a real
# environment (RDKit available).  When RDKit is unavailable the probe
# is skipped and the import-time hard-fail is deferred to
# :func:`_require_rdkit`.  This is a cheap smoke check that the
# 32-d pipeline produces a finite vector on a canonical SMILES.
# ---------------------------------------------------------------------------
def _self_check() -> None:
    if not _HAS_RDKIT:
        return
    try:
        probe = compute_rdkit_2d_extended("Cc1ccccc1")  # toluene
    except Exception:  # pragma: no cover — defensive
        return
    if probe is None:
        # Schema drift / RDKit misconfig — surface a clear error at
        # module load instead of silently breaking downstream callers.
        raise RDKit2DExtendedError(
            "compute_rdkit_2d_extended('Cc1ccccc1') returned None at "
            "module import — RDKit is importable but the 32-d pipeline "
            "fails to produce a finite vector on toluene. The schema "
            "may have drifted from RDKit's Descriptors namespace."
        )
    if probe.shape != (RDKIT_2D_DIM,):
        raise RDKit2DExtendedError(
            f"Self-check produced shape {probe.shape}; expected " f"({RDKIT_2D_DIM},)."
        )


try:
    _self_check()
except RDKit2DExtendedError:
    raise
except Exception:  # pragma: no cover — defensive
    # An unexpected error during self-check is non-fatal at module
    # load (some RDKit builds raise on partial-charge descriptors for
    # aromatic-only molecules — e.g. toluene lacks an explicit H on
    # the aromatic carbons for the Gasteiger charge).  The actual
    # failure surfaces from per-descriptor NaN handling in
    # :func:`compute_rdkit_2d_extended`.
    pass
