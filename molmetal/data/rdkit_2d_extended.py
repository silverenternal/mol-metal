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
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# RDKit is heavy and not always available in every test env.  Lazy-import
# with a hard fail (this module's value proposition is RDKit 2D
# descriptors, so we must fail loudly if RDKit is absent — the same
# defensive pattern used in extract_rdkit_2d_descriptors.py).
try:
    from rdkit import Chem, RDLogger  # type: ignore
    from rdkit.Chem import AllChem, Descriptors  # type: ignore

    RDLogger.DisableLog("rdApp.*")
    _HAS_RDKIT = True
    _RDKIT_IMPORT_ERROR: Optional[BaseException] = None
except Exception as _exc:  # pragma: no cover — defensive
    Chem = None  # type: ignore
    RDLogger = None  # type: ignore
    Descriptors = None  # type: ignore
    AllChem = None  # type: ignore
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
    ("MaxPartialCharge", None),  # resolved by _partial_charges() — see below
    ("MinPartialCharge", None),
    ("MaxAbsPartialCharge", None),
    ("MinAbsPartialCharge", None),
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


#: Names of the 4 partial-charge columns, in canonical order.
PARTIAL_CHARGE_COLUMNS: Tuple[str, ...] = (
    "MaxPartialCharge",
    "MinPartialCharge",
    "MaxAbsPartialCharge",
    "MinAbsPartialCharge",
)

#: Metal elements Gasteiger cannot parameterize.  When a charge model
#: fails on one of these we retry on the metal-free organic subgraph
#: rather than emitting a fabricated or constant-zero charge.
_METAL_SYMBOLS: frozenset = frozenset(
    {
        "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr",
        "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb",
        "Te", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb",
        "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os",
        "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Th", "U",
    }
)

#: How each molecule's partial charges were obtained.  Reset per-process;
#: read via :func:`partial_charge_stats` so a caller can REPORT the
#: fallback rate instead of silently ingesting zeros.
PARTIAL_CHARGE_STATS: Dict[str, int] = {
    "n_molecules": 0,
    "n_rdkit_native": 0,  # Descriptors.*PartialCharge returned finite values
    "n_gasteiger_full": 0,  # fell back to Gasteiger on the whole molecule
    "n_gasteiger_organic": 0,  # Gasteiger on the metal-free organic subgraph
    "n_failed": 0,  # no finite charge at all -> 0.0 emitted (counted)
}


def reset_partial_charge_stats() -> None:
    """Zero the per-process partial-charge provenance counters."""
    for key in PARTIAL_CHARGE_STATS:
        PARTIAL_CHARGE_STATS[key] = 0


def partial_charge_stats() -> Dict[str, int]:
    """Return a copy of the partial-charge provenance counters.

    ``n_gasteiger_organic`` + ``n_failed`` is the fallback count — how
    many molecules RDKit's native partial-charge descriptors could not
    parameterize.  Reported rather than assumed, because a silent 0.0
    is indistinguishable from a real neutral molecule downstream.
    """
    return dict(PARTIAL_CHARGE_STATS)


def _gasteiger_charges(mol) -> List[float]:
    """Return the finite Gasteiger charges on ``mol`` ([] when undefined)."""
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        return []
    out: List[float] = []
    for atom in mol.GetAtoms():
        if not atom.HasProp("_GasteigerCharge"):
            continue
        val = float(atom.GetDoubleProp("_GasteigerCharge"))
        if np.isfinite(val):
            out.append(val)
    return out


def _partial_charges(mol) -> Tuple[float, float, float, float]:
    """Compute the 4 partial-charge columns for ``mol``.

    Resolution order, first that yields finite values wins:

      1. ``rdkit.Chem.Descriptors.{Max,Min,MaxAbs,MinAbs}PartialCharge``
         — bit-identical to the historical path on ordinary organics.
      2. ``ComputeGasteigerCharges`` on the whole molecule.
      3. ``ComputeGasteigerCharges`` on the metal-free organic subgraph —
         Gasteiger has no parameters for Pt/Ru/Ir centers, so on these
         metallodrugs step 1 returns NaN and step 2 returns all-NaN.
      4. ``0.0`` for all four, with ``n_failed`` incremented.

    Every outcome is counted in :data:`PARTIAL_CHARGE_STATS`.
    """
    PARTIAL_CHARGE_STATS["n_molecules"] += 1

    # 1. RDKit's own descriptor — preserves historical values where valid.
    try:
        native = (
            float(Descriptors.MaxPartialCharge(mol)),
            float(Descriptors.MinPartialCharge(mol)),
            float(Descriptors.MaxAbsPartialCharge(mol)),
            float(Descriptors.MinAbsPartialCharge(mol)),
        )
        if all(np.isfinite(v) for v in native):
            PARTIAL_CHARGE_STATS["n_rdkit_native"] += 1
            return native
    except Exception:
        pass

    # 2. Explicit Gasteiger on the full molecule.
    charges = _gasteiger_charges(mol)
    if charges:
        PARTIAL_CHARGE_STATS["n_gasteiger_full"] += 1
        return (
            max(charges),
            min(charges),
            max(abs(c) for c in charges),
            min(abs(c) for c in charges),
        )

    # 3. Gasteiger on the metal-free organic subgraph.
    metal_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() in _METAL_SYMBOLS]
    if metal_idx:
        editable = Chem.RWMol(mol)
        for idx in sorted(metal_idx, reverse=True):
            editable.RemoveAtom(idx)
        stripped = editable.GetMol()
        try:
            Chem.SanitizeMol(stripped)
        except Exception:
            stripped = None
        if stripped is not None and stripped.GetNumAtoms() > 0:
            charges = _gasteiger_charges(stripped)
            if charges:
                PARTIAL_CHARGE_STATS["n_gasteiger_organic"] += 1
                return (
                    max(charges),
                    min(charges),
                    max(abs(c) for c in charges),
                    min(abs(c) for c in charges),
                )

    # 4. Nothing worked — emit a counted 0.0 rather than a silent NaN.
    PARTIAL_CHARGE_STATS["n_failed"] += 1
    return (0.0, 0.0, 0.0, 0.0)


#: Per-name partial-charge accessor, so the generic descriptor loop can
#: stay a plain ``(name, callable)`` dispatch.
_PARTIAL_CHARGE_ACCESSORS: Dict[str, Callable] = {
    "MaxPartialCharge": lambda mol: _partial_charges(mol)[0],
    "MinPartialCharge": lambda mol: _partial_charges(mol)[1],
    "MaxAbsPartialCharge": lambda mol: _partial_charges(mol)[2],
    "MinAbsPartialCharge": lambda mol: _partial_charges(mol)[3],
}
# Each accessor recomputes the tuple; memoize per-molecule so the 4 columns
# cost one charge model evaluation, not four.
_ORG_CACHE_ATTR = "_molmetal_gasteiger_cache"


def _cached_partial_charges(mol) -> Tuple[float, float, float, float]:
    """Memoize :func:`_partial_charges` on the molecule object itself."""
    try:
        cached = mol.GetProp(_ORG_CACHE_ATTR)
        return tuple(float(x) for x in cached.split(","))  # type: ignore[return-value]
    except KeyError:
        pass
    vals = _partial_charges(mol)
    try:
        mol.SetProp(_ORG_CACHE_ATTR, ",".join(repr(float(v)) for v in vals))
    except Exception:
        pass
    return vals


for _name in PARTIAL_CHARGE_COLUMNS:
    _PARTIAL_CHARGE_ACCESSORS[_name] = (
        lambda mol, _i=PARTIAL_CHARGE_COLUMNS.index(_name): _cached_partial_charges(mol)[
            _i
        ]
    )
del _name

# Replace the four None placeholders with the memoized accessors.
_EXTRA_20D_DESCRIPTORS = tuple(
    (_name, _PARTIAL_CHARGE_ACCESSORS[_name])
    if _name in PARTIAL_CHARGE_COLUMNS
    else (_name, _fn)
    for _name, _fn in _EXTRA_20D_DESCRIPTORS
)
RDKIT_2D_EXTENDED_DESCRIPTORS = _BASE_12D_DESCRIPTORS + _EXTRA_20D_DESCRIPTORS

#: Column names that no longer resolve to a live descriptor callable.
UNRESOLVED_DESCRIPTOR_COLUMNS: Tuple[str, ...] = tuple(
    _name for _name, _fn in RDKIT_2D_EXTENDED_DESCRIPTORS if _fn is None
)
if UNRESOLVED_DESCRIPTOR_COLUMNS:
    raise RDKit2DExtendedError(
        "32-d registry has columns with no descriptor callable: "
        f"{list(UNRESOLVED_DESCRIPTOR_COLUMNS)} — every column name must "
        "resolve to a real descriptor or the vector silently ships zeros."
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
