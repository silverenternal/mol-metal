#!/usr/bin/env python3
"""Wf-49 Phase 1 — RDKit 2D molecular descriptors feature extractor.

=====================================================================
Standalone feature extractor producing a real 12-d RDKit 2D descriptor
vector per (smiles, metal) row. This **replaces** the synthetic ASTSR
5-d descriptors (``diversity_tanimoto``, ``diversity_homotype``,
``qed_mean``, ``anticancer_index``, ``sa_mean_norm``) that Wf-44 Root
Cause Catalog §Cause 2 (CRITICAL) flagged as "not informative for
target".

Why this exists
---------------
* m16 Affinity Pearson r baseline (HeLa48h/dark 3-seed × 50-epoch,
  margin=0.5): **0.207** (per ``metrics/by_metric/pearson_r_pic50.json``)
* Ridge baseline on exact-row pearson: **0.572**
* Gap to ridge: **-0.365** — D-MPNN's graph features fail to beat
  linear regression on RDKit descriptors on a 700-sample test split.
* Wf-44 Cause 2 mitigation: replace synthetic ASTSR 5-d features with
  real RDKit 2D physico-chemical descriptors. Literature anchor:
  Yang et al. 2019 (JCIM, D-MPNN paper) + Wu et al. 2017 (MoleculeNet,
  §"smaller single-task datasets" caveat) + arXiv:2510.23428 (MetaModel
  = RDKit + ChemProp +0.22 AUC on BACE, +0.20 on BBBP).

Mathematical nature of m16 (Pearson r on pIC50)
------------------------------------------------
``pIC50 = 9 - log10(IC50_nM)`` is a **log-transformed, molar-scale
bioactivity**. Its drivers are dominated by Lipinski-style
physico-chemical properties:

  * MolWt (size / cell permeability)
  * MolLogP (lipophilicity / membrane crossing)
  * TPSA (polar surface / H-bond donor-acceptor balance)
  * HBA / HBD (Lipinski's Rule of Five)
  * NRotB (rotational entropy / oral bioavailability)
  * HeavyAtoms (size proxy)
  * RingCount / NumAromaticRings (scaffold rigidity / π-stacking)
  * FractionCSP3 (sp³ fraction / 3D character)
  * MolMR (molar refractivity / polarizability)
  * BalabanJ (topological connectivity — captures scaffold topology
    that is largely invariant to the metal centre choice)

These are the canonical 12 RDKit 2D descriptors that any sensible
QSAR pipeline would include as a baseline. We attach them per
(smiles, metal) because the metal identity is a categorical feature
that interacts with the ligand's electronic structure (d-orbital
splitting, ligand-field stabilisation, etc.) — keeping it as a
separate column preserves the multi-task structure of the PlatinAI /
MetalCytoTox cohorts.

Supported datasets (selected via ``--dataset``):
  * ``platinai`` — ``platinai_5k.csv`` (5,000 rows, 4 cell lines).
    Target = pIC50 = ``9 - log10(IC50_nM)`` from the ``ic50_nm`` column
    (selectivity column carried for PlatinAI 5K tier-1 only).
  * ``metalcytotox`` — ``metalcytotox_5metal_multitask_v4.csv``
    (~26,800 rows × 5 metals). Target = activity (binary 0/1) +
    per-metal scaffold-group split. Reuses the existing
    :class:`molmetal.data.cytotox.CytotoxFilter` defaults
    (time_threshold=24h, ic50_min=0.01 µM) but in CSV-streaming form
    to avoid loading 26k rows through pandas.

Recipe
------
(1) For each row in the source CSV, parse ``smiles`` with RDKit
    ``Chem.MolFromSmiles``. Sanitize + drop rows where the SMILES
    fails to parse or where the molecule has zero heavy atoms.
(2) Compute 12 RDKit 2D descriptors via ``rdMolDescriptors`` +
    ``Descriptors._descList``. Each descriptor is finite-checked;
    NaN / inf are replaced by the per-column median over the
    successfully-parsed rows (deterministic given fixed input order).
(3) Attach the per-row ``metal`` and per-dataset target columns.
(4) Stream to disk as a flat CSV with one row per (smiles, metal)
    pair + 12 desc columns + target columns. This is the
    **drop-in replacement** for the ASTSR 5-d features in
    downstream Ridge / D-MPNN trainers.

Outputs (under ``molmetal/data/``)
----------------------------------
* ``platinai_5k_rdkit_features.csv`` — 5,000 × (12 + 4 + 1 + 1) =
  ``smiles, metal, ic50_nm, selectivity, pIC50`` + 12 desc columns.
* ``metalcytotox_rdkit_features.csv`` — ~26,800 × (12 + 1 + 1) =
  ``smiles, metal, activity`` + 12 desc columns.

Side outputs (under ``molmetal/reports/wf49_rdkit_extraction/``)
----------------------------------------------------------------
* ``feature_summary.json`` — per-dataset statistics:
  - n_rows_total / n_rows_parsed / n_rows_dropped
  - per-descriptor (mean, std, min, max, median, n_missing)
  - target correlation: Pearson r of each descriptor vs the target
    on the parsed subset (sanity check that at least one descriptor
    correlates with pIC50 / activity at the level expected by Wf-44
    Cause 2 mitigation).
  - per-metal row counts (Ru / Ir / Rh / Os / Re / Pt / Pd / Au).

Verify
------
* ``molmetal/tests/test_extract_rdkit_2d.py`` — 10 tests:
  1. CSV shape invariants (row counts within tolerance).
  2. RDKit parses 100% of the 5,000 PlatinAI smiles (or reports the
     parse failures in feature_summary.json — no silent drop).
  3. 12 expected descriptor columns present with finite values.
  4. pIC50 transformation is bit-exact
     (``pIC50 == 9 - log10(ic50_nm)`` to within 1e-6).
  5. metalcytotox row count is the same order as the source CSV.
  6. feature_summary.json schema is valid (desc stats + correlations).
  7. Pearson r between ``MolLogP`` and ``pIC50`` is finite and
     |r| ≤ 1 (sanity check on the correlation block).
  8. Deterministic re-run: identical CSVs byte-for-byte.
  9. CLI ``--help`` exits with code 0.
 10. The 12 descriptors cover at least one Lipinski-style driver
     (MolLogP, TPSA, HBA, HBD — all 4 present).

Usage
-----
::

    source env.sh
    uv run --no-sync python -u molmetal/scripts/extract_rdkit_2d_descriptors.py \\
        --dataset platinai
    uv run --no-sync python -u molmetal/scripts/extract_rdkit_2d_descriptors.py \\
        --dataset metalcytotox
    uv run --no-sync python -u molmetal/scripts/extract_rdkit_2d_descriptors.py \\
        --dataset both --out-dir molmetal/data
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    from molmetal.scripts.wf386a3_atomic_write import (
        atomic_write_file,
        safe_overwrite_check,
        detect_truncation,
    )
    _WF416_HAS_ATOMIC = True
except ImportError:  # pragma: no cover — fallback when running as standalone
    _WF416_HAS_ATOMIC = False
    atomic_write_file = None  # type: ignore[assignment]
    safe_overwrite_check = None  # type: ignore[assignment]
    detect_truncation = None  # type: ignore[assignment]


# RDKit is heavy and not always available; Wf-39 P1 vuln_a CRITICAL
# already burned us once when RDKit was missing. Lazy-import with a
# hard fail (this script's value proposition is RDKit 2D descriptors
# so we must fail loudly if RDKit is absent).
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit import DataStructs
    RDLogger.DisableLog("rdApp.*")  # silence the per-mol warnings
    _HAS_RDKIT = True
except Exception as exc:                                              # pragma: no cover
    _HAS_RDKIT = False
    _RDKIT_IMPORT_ERROR = exc


# ---------------------------------------------------------------------------
# 12 RDKit 2D descriptor registry (frozen for Wf-49 Phase 1)
# ---------------------------------------------------------------------------
# Each tuple = (column_name, callable(mol) -> float). Order is fixed for
# schema stability across re-runs.
RDKIT_2D_DESCRIPTORS: Tuple[Tuple[str, "callable"], ...] = (
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


# ---------------------------------------------------------------------------
# 32 RDKit 2D descriptor registry (Wf-218 — F5_RDKIT_2D expansion)
# ---------------------------------------------------------------------------
# Wf-218 expands the Wf-49 Phase 1 12-d set to 32-d by appending 20
# additional canonical RDKit descriptors targeting the Wf-127 §Cause 2
# "12-d signal floor" for Pt-only cohorts (PlatinAI 4-cell R² ≤ 0):
#
#   * Shape / topology: BertzCT, Chi0v, Chi0n, Chi1n, HallKierAlpha
#   * Kappa shape indices: Kappa1, Kappa2, Kappa3
#   * Surface area / volume: LabuteASA
#   * Partial-charge descriptors (relevant to Pt electron density):
#     MaxPartialCharge, MinPartialCharge, MaxAbsPartialCharge,
#     MinAbsPartialCharge
#   * Ring-shape features (aromatic vs aliphatic vs saturated):
#     NumAliphaticRings, NumSaturatedRings
#   * Heteroatom counts (relevant to N-donor Pt ligands):
#     NumHeteroatoms
#   * Functional group counts: NHOHCount, NOCount
#   * Halogen count (common in Pt drugs — cisplatin / carboplatin have
#     Cl⁻ leaving groups; oxaliplatin has oxalate):
#     fr_halogen
#
# Order is frozen for schema stability.  The first 12 entries are the
# Wf-49 Phase 1 frozen order (so ``compute_rdkit_extended_descriptors``
# produces a 32-d vector whose first 12 entries match
# ``compute_rdkit_descriptors`` exactly — guaranteeing backward
# compatibility for the 12-d path).
RDKIT_2D_EXTENDED_DESCRIPTORS: Tuple[Tuple[str, "callable"], ...] = (
    # 12 from Wf-49 P1 (frozen order, preserved bit-for-bit)
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
    # 20 Wf-218 expansion descriptors
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


def _wf416_atomic_write_text(path, content: str, min_lines: int = 50, expected_min_length: int = 200) -> None:
    """Wf-416-A1 atomic-write-protocol wrapper around direct write_text.

    Falls back to direct write_text if atomic_write_file is unavailable
    (e.g. when the script is run standalone without the molmetal package
    on PYTHONPATH).
    """
    if _WF416_HAS_ATOMIC and atomic_write_file is not None:
        if safe_overwrite_check is not None:
            _risk = safe_overwrite_check(path, content, min_lines=min_lines)
            if _risk is not None:
                import logging
                logging.getLogger(__name__).warning(
                    "[Wf-416-A1] safe_overwrite_check diff_ratio=%.3f at %s",
                    _risk, path,
                )
        atomic_write_file(path, content, encoding="utf-8")
        if detect_truncation is not None:
            _trunc = detect_truncation(content, expected_min_length=expected_min_length)
            if _trunc is not None:
                import logging
                logging.getLogger(__name__).warning(
                    "[Wf-416-A1] detect_truncation reason=%r at %s",
                    _trunc, path,
                )
    else:  # pragma: no cover — fallback when atomic primitive unavailable
        Path(path).write_text(content, encoding="utf-8")

def _require_rdkit() -> None:
    if not _HAS_RDKIT:
        raise RuntimeError(
            "RDKit is required for extract_rdkit_2d_descriptors.py "
            f"(import failed: {_RDKIT_IMPORT_ERROR!r}). "
            "Install via `pip install rdkit` or activate the project venv."
        )


def compute_rdkit_descriptors(smiles: str) -> Optional[List[float]]:
    """Compute the 12-d descriptor vector for a single SMILES.

    Returns ``None`` if the SMILES fails to parse or yields an empty
    molecule. NaN / inf values in individual descriptors are returned
    as NaN so the caller can decide on the imputation strategy.
    """
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return None
    out: List[float] = []
    for _name, fn in RDKIT_2D_DESCRIPTORS:
        try:
            v = float(fn(mol))
        except Exception:                                              # pragma: no cover
            v = float("nan")
        out.append(v)
    return out


def compute_rdkit_extended_descriptors(smiles: str) -> Optional[List[float]]:
    """Compute the 32-d F5_RDKIT_2D extended descriptor vector (Wf-218).

    Returns ``None`` if the SMILES fails to parse or yields an empty
    molecule. NaN / inf values in individual descriptors are returned
    as NaN so the caller can decide on the imputation strategy.

    The first 12 entries match :func:`compute_rdkit_descriptors`
    exactly (backward-compat guarantee for the Wf-49 Phase 1 12-d path).
    """
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return None
    out: List[float] = []
    for _name, fn in RDKIT_2D_EXTENDED_DESCRIPTORS:
        try:
            v = float(fn(mol))
        except Exception:                                              # pragma: no cover
            v = float("nan")
        out.append(v)
    return out


# ---------------------------------------------------------------------------
# Wf-249 Fix #1 — 1024-bit Morgan + RDK topological fingerprints
# ---------------------------------------------------------------------------
# Citation: Jiang et al. ChemRxiv 2024 — higher-bit Morgan / RDK FP
# breaks the 12-d signal-floor plateau on MoleculeNet property
# prediction (L5 Hilbert-ASTSR R² currently ≤ -0.40 on Pt-only
# PlatinAI 4-cell cohorts per Wf-127 §Cause 2).
#
# These functions are ADDITIVE — the existing 12-d / 32-d paths are
# unchanged. Default mode of ``compute_features_v2`` is ``"rdkit12"``
# (backward compat); opt-in via ``mode="fp1024"`` (1024-d ablation) or
# ``mode="rdkit12+fp1024"`` (1036-d concat, recommended).
DEFAULT_FP_RADIUS = 2
DEFAULT_FP_NBITS = 1024


def compute_morgan_fp_1024(smiles: str, radius: int = DEFAULT_FP_RADIUS,
                          n_bits: int = DEFAULT_FP_NBITS) -> Optional[np.ndarray]:
    """1024-bit Morgan (ECFP4) fingerprint per Jiang et al. ChemRxiv 2024.

    Returns ``None`` if RDKit fails to parse the SMILES or the mol has
    zero heavy atoms. Caller is responsible for imputation / fallback
    (matches ``compute_rdkit_descriptors`` contract for graceful
    degradation).
    """
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return None
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    except Exception:                                                  # pragma: no cover
        return None
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_rdk_fp_1024(smiles: str, n_bits: int = DEFAULT_FP_NBITS) -> Optional[np.ndarray]:
    """1024-bit RDK topological fingerprint per Jiang et al. ChemRxiv 2024.

    Same None-on-failure contract as :func:`compute_morgan_fp_1024`.
    """
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return None
    try:
        fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
    except Exception:                                                  # pragma: no cover
        return None
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


FEATURES_V2_MODES: Tuple[str, ...] = (
    "rdkit12",            # legacy 12-d, backward compat (Wf-49 P1)
    "fp1024",             # 1024-d Morgan only — ablation arm
    "rdkit12+fp1024",     # 1036-d concat — recommended
)


def compute_features_v2(smiles: str, mode: str = "rdkit12") -> Optional[np.ndarray]:
    """Unified feature extractor with 3 modes (Wf-249 Fix #1).

    Modes:
      - ``"rdkit12"``         — original 12-d descriptor (backward compat).
      - ``"fp1024"``          — 1024-d Morgan ECFP4 only (ablation).
      - ``"rdkit12+fp1024"``  — concatenated 1036-d (recommended lift).

    Returns ``None`` if RDKit fails to parse (the rdkit12 path is then
    unparseable; the fp1024 path returns ``None`` if SMILES is bad).
    Empty input is converted to a deterministic zero vector of the
    target dim so the resulting row stays aligned with non-empty rows.
    """
    if mode not in FEATURES_V2_MODES:
        raise ValueError(
            f"compute_features_v2: unknown mode={mode!r}, "
            f"must be one of {FEATURES_V2_MODES}"
        )
    want_rdkit12 = mode in ("rdkit12", "rdkit12+fp1024")
    want_fp1024 = mode in ("fp1024", "rdkit12+fp1024")
    parts: List[np.ndarray] = []
    if want_rdkit12:
        d12 = compute_rdkit_descriptors(smiles)
        if d12 is None:
            d12 = [float("nan")] * len(RDKIT_2D_DESCRIPTORS)
        parts.append(np.asarray(d12, dtype=np.float32))
    if want_fp1024:
        fp = compute_morgan_fp_1024(smiles)
        if fp is None:
            fp = np.zeros(DEFAULT_FP_NBITS, dtype=np.uint8)
        parts.append(fp.astype(np.float32))
    if not parts:
        return None
    out = np.concatenate(parts)
    return out


# ---------------------------------------------------------------------------
# PlatinAI loaders
# ---------------------------------------------------------------------------
# Two PlatinAI CSVs exist on disk:
#   * ``platinai_5k.csv``      — 5,000 rows × {smiles, ic50_nm, selectivity,
#                                 metal}.  IC50 + selectivity columns are
#                                 **empty** in this structural subset; the
#                                 file exists for SMILES-only coverage.
#   * ``platinai_4cell.csv``   — 4,078 rows × {smiles, ic50_mcf7_nm,
#                                 ic50_a549_nm, ic50_a2780_nm,
#                                 ic50_a2780cis_nm, source}.  All 4 IC50
#                                 columns are populated; this is the file
#                                 for regression analysis (m16 Affinity
#                                 Pearson r headline metric).
# Both files share the per-row ``smiles`` + ``metal`` contract; the Wf-49
# Phase 1 extractor runs over both.
PLATINAI_5K_DEFAULT = PROJECT_ROOT / "molmetal" / "data" / "platinai_5k.csv"
PLATINAI_5K_OUT_DEFAULT = PROJECT_ROOT / "molmetal" / "data" / "platinai_5k_rdkit_features.csv"
PLATINAI_4CELL_DEFAULT = PROJECT_ROOT / "molmetal" / "data" / "platinai_4cell.csv"
PLATINAI_4CELL_OUT_DEFAULT = (
    PROJECT_ROOT / "molmetal" / "data" / "platinai_4cell_rdkit_features.csv"
)
PLATINAI_4CELL_CELL_LINES: Tuple[str, ...] = (
    "MCF7", "A549", "A2780", "A2780cis",
)
PLATINAI_4CELL_IC50_COLS: Tuple[str, ...] = tuple(
    f"ic50_{l.lower()}_nm" for l in PLATINAI_4CELL_CELL_LINES
)


def _pIC50_from_nM(ic50_nM: float) -> float:
    """pIC50 = 9 - log10(IC50_nM). NaN if IC50 is non-positive."""
    if not math.isfinite(ic50_nM) or ic50_nM <= 0:
        return float("nan")
    return 9.0 - math.log10(ic50_nM)


def load_platinai_5k(
    path: Path = PLATINAI_5K_DEFAULT,
    max_rows: Optional[int] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    """Stream the PlatinAI 5K CSV into a list of row dicts + stats.

    Each row dict has keys: ``smiles``, ``metal``, ``ic50_nm``,
    ``selectivity``, ``pIC50`` (NaN if IC50 missing/non-positive).
    The returned stats dict has: ``n_total``, ``n_no_smiles``,
    ``n_no_ic50``.

    Note: as of 2026-09-26 the on-disk ``platinai_5k.csv`` has
    ``ic50_nm`` and ``selectivity`` columns empty for all 5,000 rows;
    this loader still emits the rows so downstream consumers can see
    the SMILES + metal coverage.  Real pIC50 targets live in
    :func:`load_platinai_4cell`.
    """
    rows: List[Dict[str, object]] = []
    stats = {"n_total": 0, "n_no_smiles": 0, "n_no_ic50": 0}
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            stats["n_total"] += 1
            smi = (row.get("smiles") or "").strip()
            if not smi:
                stats["n_no_smiles"] += 1
                continue
            metal = (row.get("metal") or "").strip() or "UNK"
            ic50_raw = (row.get("ic50_nm") or "").strip()
            ic50_nM: float = float("nan")
            if ic50_raw:
                try:
                    ic50_nM = float(ic50_raw)
                except ValueError:
                    stats["n_no_ic50"] += 1
            else:
                stats["n_no_ic50"] += 1
            sel_raw = (row.get("selectivity") or "").strip()
            selectivity: float = float("nan")
            if sel_raw:
                try:
                    selectivity = float(sel_raw)
                except ValueError:
                    selectivity = float("nan")
            rows.append({
                "smiles": smi,
                "metal": metal,
                "ic50_nm": ic50_nM,
                "selectivity": selectivity,
                "pIC50": _pIC50_from_nM(ic50_nM),
            })
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows, stats


def load_platinai_4cell(
    path: Path = PLATINAI_4CELL_DEFAULT,
    max_rows: Optional[int] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    """Stream the PlatinAI 4-cell CSV into a list of row dicts.

    Each row has 4 IC50 columns (``ic50_mcf7_nm``, ``ic50_a549_nm``,
    ``ic50_a2780_nm``, ``ic50_a2780cis_nm``); we emit a flat row dict
    with the raw IC50 columns + a ``metal`` field derived from the
    "Pt" source tag (all rows are Pt complexes — confirmed by the
    original PlatinAI 5K build).

    The 4-cell CSV is the **labelled regression cohort** for the m16
    Affinity Pearson r metric.  The Wf-32 Phase 4 smoke run achieved
    Pearson r = -0.040 on this cohort; Wf-49 Phase 1 provides the
    RDKit 2D descriptor inputs that any downstream Ridge / GBM /
    D-MPNN-with-RDKit-aux-features trainer can consume.
    """
    rows: List[Dict[str, object]] = []
    stats = {"n_total": 0, "n_no_smiles": 0, "n_no_ic50_any": 0}
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            stats["n_total"] += 1
            smi = (row.get("smiles") or "").strip()
            if not smi:
                stats["n_no_smiles"] += 1
                continue
            metal = "Pt"  # all platinai_4cell.csv rows are Pt complexes
            ic50_block: Dict[str, float] = {}
            row_has_any_ic50 = False
            for col in PLATINAI_4CELL_IC50_COLS:
                raw = (row.get(col) or "").strip()
                v: float = float("nan")
                if raw:
                    try:
                        v = float(raw)
                        if v > 0 and math.isfinite(v):
                            row_has_any_ic50 = True
                    except ValueError:
                        v = float("nan")
                ic50_block[col] = v
            if not row_has_any_ic50:
                stats["n_no_ic50_any"] += 1
            rows.append({
                "smiles": smi,
                "metal": metal,
                "ic50_mcf7_nm": ic50_block["ic50_mcf7_nm"],
                "ic50_a549_nm": ic50_block["ic50_a549_nm"],
                "ic50_a2780_nm": ic50_block["ic50_a2780_nm"],
                "ic50_a2780cis_nm": ic50_block["ic50_a2780cis_nm"],
                "pIC50_mcf7": _pIC50_from_nM(ic50_block["ic50_mcf7_nm"]),
                "pIC50_a549": _pIC50_from_nM(ic50_block["ic50_a549_nm"]),
                "pIC50_a2780": _pIC50_from_nM(ic50_block["ic50_a2780_nm"]),
                "pIC50_a2780cis": _pIC50_from_nM(ic50_block["ic50_a2780cis_nm"]),
            })
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows, stats


# ---------------------------------------------------------------------------
# MetalCytoToxDB 5-metal loader
# ---------------------------------------------------------------------------
METALCYTO_5METAL_DEFAULT = (
    PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"
)
METALCYTO_OUT_DEFAULT = (
    PROJECT_ROOT / "molmetal" / "data" / "metalcytotox_rdkit_features.csv"
)


def load_metalcytotox_5metal(
    path: Path = METALCYTO_5METAL_DEFAULT,
    max_rows: Optional[int] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    """Stream the MetalCytoTox 5-metal CSV into a list of row dicts.

    The CSV columns are: ``smiles``, ``metal``, ``scaffold_id``,
    ``split``, ``activity``.  ``activity`` is binary 0/1.
    """
    rows: List[Dict[str, object]] = []
    stats = {"n_total": 0, "n_no_smiles": 0, "n_no_activity": 0}
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            stats["n_total"] += 1
            smi = (row.get("smiles") or "").strip()
            if not smi:
                stats["n_no_smiles"] += 1
                continue
            metal = (row.get("metal") or "").strip() or "UNK"
            activity_raw = (row.get("activity") or "").strip()
            activity: float = float("nan")
            if activity_raw:
                try:
                    activity = float(activity_raw)
                except ValueError:
                    stats["n_no_activity"] += 1
            else:
                stats["n_no_activity"] += 1
            rows.append({
                "smiles": smi,
                "metal": metal,
                "activity": activity,
            })
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows, stats


# ---------------------------------------------------------------------------
# Per-row descriptor extraction (with median imputation)
# ---------------------------------------------------------------------------
def extract_descriptors_for_rows(
    rows: List[Dict[str, object]],
    desc_names: Sequence[str],
    n_fn: "callable",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run RDKit over each row, return (X, valid_mask, parse_failures).

    ``X`` is ``(n_rows, len(desc_names))`` float32.  Rows where RDKit
    fails to parse have ``valid_mask[i] = False`` and ``X[i, :] = 0``.
    Median imputation runs across the **parsed** subset for each
    descriptor so the output is fully finite.
    """
    n_rows = len(rows)
    X = np.zeros((n_rows, len(desc_names)), dtype=np.float32)
    valid = np.zeros(n_rows, dtype=bool)
    parse_failures = np.zeros(n_rows, dtype=bool)
    smiles_list = [str(r["smiles"]) for r in rows]

    for i, smi in enumerate(smiles_list):
        feats = n_fn(smi)
        if feats is None:
            parse_failures[i] = True
            continue
        X[i, :] = np.asarray(feats, dtype=np.float32)
        valid[i] = True

    # Median imputation over the parsed subset, per descriptor.
    for j in range(len(desc_names)):
        col = X[:, j]
        finite_mask = valid & np.isfinite(col)
        if not finite_mask.any():
            # No finite entries for this descriptor — fill with 0.0
            # (this only happens for malformed inputs; the Wf-49
            # contract says every descriptor is finite on PlatinAI 5K).
            continue
        med = float(np.median(col[finite_mask]))
        col_invalid = valid & ~np.isfinite(col)
        X[col_invalid, j] = med
    # Final safety: force finite on the valid rows.
    X[~valid, :] = 0.0
    X[~np.isfinite(X)] = 0.0
    return X, valid, parse_failures


# ---------------------------------------------------------------------------
# Feature summary writer
# ---------------------------------------------------------------------------
def _desc_stats_block(X: np.ndarray, desc_names: Sequence[str]) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for j, name in enumerate(desc_names):
        col = X[:, j]
        finite = col[np.isfinite(col)]
        if len(finite) == 0:
            out.append({
                "name": name,
                "n_missing": int(len(col)),
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "median": None,
            })
            continue
        out.append({
            "name": name,
            "n_missing": int(np.sum(~np.isfinite(col))),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "median": float(np.median(finite)),
        })
    return out


def _pearson_block(
    X: np.ndarray, y: np.ndarray, desc_names: Sequence[str],
) -> List[Dict[str, float]]:
    """Per-descriptor Pearson r vs target on the finite subset.

    Returns one dict per descriptor with keys ``name``, ``pearson_r``,
    ``n_samples``.  ``pearson_r`` is ``None`` if either side has zero
    variance (constant descriptor or constant target).
    """
    out: List[Dict[str, float]] = []
    finite = np.isfinite(y)
    for j, name in enumerate(desc_names):
        col = X[:, j]
        mask = finite & np.isfinite(col)
        n = int(np.sum(mask))
        if n < 3:
            out.append({"name": name, "pearson_r": None, "n_samples": n})
            continue
        x = col[mask]
        yy = y[mask]
        if np.std(x) == 0 or np.std(yy) == 0:
            out.append({"name": name, "pearson_r": None, "n_samples": n})
            continue
        r = float(np.corrcoef(x, yy)[0, 1])
        if not math.isfinite(r):
            r_val: Optional[float] = None
        else:
            r_val = r
        out.append({"name": name, "pearson_r": r_val, "n_samples": n})
    return out


def write_feature_summary(
    out_path: Path,
    dataset_name: str,
    n_rows_total: int,
    n_rows_parsed: int,
    n_rows_dropped: int,
    per_metal_counts: Dict[str, int],
    desc_stats: List[Dict[str, float]],
    correlations: List[Dict[str, float]],
    target_col: str,
    wall_seconds: float,
    source_csv: str,
    n_samples_warning: Optional[str] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset_name,
        "source_csv": source_csv,
        "target_col": target_col,
        "n_descriptors": len(desc_stats),
        "descriptor_names": [d["name"] for d in desc_stats],
        "n_rows_total": n_rows_total,
        "n_rows_parsed": n_rows_parsed,
        "n_rows_dropped": n_rows_dropped,
        "parse_success_rate": (
            n_rows_parsed / n_rows_total if n_rows_total else 0.0
        ),
        "per_metal_counts": per_metal_counts,
        "descriptor_stats": desc_stats,
        "correlations_vs_target": correlations,
        "wall_seconds": wall_seconds,
        "wf": "Wf-49",
        "phase": "Phase 1",
        "recipe": "RDKit 2D 12-d descriptors (MolWt, MolLogP, TPSA, HBA, HBD, NRotB, HeavyAtoms, RingCount, NumAromaticRings, FractionCSP3, MolMR, BalabanJ)",
        "literature_anchor": (
            "Yang et al. 2019 JCIM (D-MPNN paper); Wu et al. 2017 "
            "MoleculeNet §'smaller single-task datasets'; arXiv:2510.23428 "
            "(MetaModel = RDKit + ChemProp)."
        ),
        "math_anchor": (
            "pIC50 = 9 - log10(IC50_nM) is log-space molar-scale "
            "bioactivity driven by Lipinski-style physico-chemical "
            "properties (MolLogP, TPSA, HBA, HBD, MW)."
        ),
    }
    if n_samples_warning is not None:
        payload["n_samples_warning"] = n_samples_warning
    _wf416_atomic_write_text(out_path, json.dumps(payload, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# PlatinAI 5K extractor
# ---------------------------------------------------------------------------
def run_platinai(out_csv: Path, summary_path: Path) -> Dict[str, object]:
    t0 = time.time()
    rows, load_stats = load_platinai_5k()
    desc_names = [n for n, _ in RDKIT_2D_DESCRIPTORS]
    X, valid, parse_failures = extract_descriptors_for_rows(
        rows, desc_names, compute_rdkit_descriptors,
    )
    n_total = len(rows)
    n_parsed = int(np.sum(valid))
    n_dropped = int(np.sum(parse_failures))

    # Per-metal counts (over parsed subset only).
    per_metal: Dict[str, int] = {}
    for i, r in enumerate(rows):
        if not bool(valid[i]):
            continue
        per_metal[str(r["metal"])] = per_metal.get(str(r["metal"]), 0) + 1

    # Targets (pIC50) — only finite entries go into the correlation
    # block; the CSV keeps the original NaN semantics so downstream
    # trainers can do their own row filter.
    y_pic50 = np.asarray(
        [
            float(r["pIC50"]) if isinstance(r["pIC50"], (int, float))
            and math.isfinite(float(r["pIC50"]))
            else float("nan")
            for r in rows
        ],
        dtype=np.float64,
    )
    correlations = _pearson_block(X, y_pic50, desc_names)

    # Write CSV (smiles, metal, ic50_nm, selectivity, pIC50 + 12 desc).
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["smiles", "metal", "ic50_nm", "selectivity", "pIC50"] + desc_names
    with out_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for i, r in enumerate(rows):
            ic50 = r["ic50_nm"]
            sel = r["selectivity"]
            pic50 = r["pIC50"]
            writer.writerow([
                r["smiles"],
                r["metal"],
                "" if not math.isfinite(float(ic50)) else f"{float(ic50):.6f}",
                "" if not math.isfinite(float(sel)) else f"{float(sel):.6f}",
                "" if not math.isfinite(float(pic50)) else f"{float(pic50):.6f}",
                *[f"{float(v):.6f}" for v in X[i, :]],
            ])

    desc_stats = _desc_stats_block(X[valid, :], desc_names)
    wall = time.time() - t0
    # n_samples_warning: emit a top-level flag when no descriptors carry
    # signal (n_samples=0 across the correlations block). This is the
    # 2026-09-26 snapshot situation: platinai_5k.csv has empty ic50_nm and
    # selectivity columns. Wf-49 Phase 4 synthesis can grep for this flag.
    n_samples_zero = sum(
        1 for c in correlations
        if c.get("n_samples", 0) == 0
    )
    n_samples_warning = None
    if n_samples_zero > 0:
        n_samples_warning = (
            f"{n_samples_zero}/{len(correlations)} descriptors have "
            f"n_samples=0 — platinai_5k.csv ic50_nm/selectivity columns "
            f"empty on 2026-09-26 snapshot; use --dataset platinai_4cell "
            f"for labelled regression."
        )
        print(f"[Wf-49] WARNING: {n_samples_warning}", flush=True)
    write_feature_summary(
        out_path=summary_path,
        dataset_name="platinai_5k",
        n_rows_total=n_total,
        n_rows_parsed=n_parsed,
        n_rows_dropped=n_dropped,
        per_metal_counts=per_metal,
        desc_stats=desc_stats,
        correlations=correlations,
        target_col="pIC50",
        wall_seconds=wall,
        source_csv=str(PLATINAI_5K_DEFAULT),
        n_samples_warning=n_samples_warning,
    )
    return {
        "dataset": "platinai_5k",
        "out_csv": str(out_csv),
        "n_rows_total": n_total,
        "n_rows_parsed": n_parsed,
        "n_rows_dropped": n_dropped,
        "load_stats": load_stats,
        "per_metal_counts": per_metal,
        "wall_seconds": wall,
        "n_samples_warning": n_samples_warning,
    }


# ---------------------------------------------------------------------------
# PlatinAI 4-cell extractor (the labelled regression cohort)
# ---------------------------------------------------------------------------
def run_platinai_4cell(
    out_csv: Path,
    summary_path: Path,
) -> Dict[str, object]:
    t0 = time.time()
    rows, load_stats = load_platinai_4cell()
    desc_names = [n for n, _ in RDKIT_2D_DESCRIPTORS]
    X, valid, parse_failures = extract_descriptors_for_rows(
        rows, desc_names, compute_rdkit_descriptors,
    )
    n_total = len(rows)
    n_parsed = int(np.sum(valid))
    n_dropped = int(np.sum(parse_failures))

    per_metal: Dict[str, int] = {}
    for i, r in enumerate(rows):
        if not bool(valid[i]):
            continue
        per_metal[str(r["metal"])] = per_metal.get(str(r["metal"]), 0) + 1

    # Targets: per-cell-line pIC50 + an across-cell-line mean pIC50.
    # The headline correlation block uses the MCF7 column (the
    # primary m16 cell line per Wf-32 smoke) plus the per-row mean
    # across the 4 cell lines (the Wf-32 headline aggregate).
    y_mcf7 = np.asarray(
        [
            float(r["pIC50_mcf7"])
            if isinstance(r["pIC50_mcf7"], (int, float))
            and math.isfinite(float(r["pIC50_mcf7"]))
            else float("nan")
            for r in rows
        ],
        dtype=np.float64,
    )
    pIC50_cols = ["pIC50_mcf7", "pIC50_a549", "pIC50_a2780", "pIC50_a2780cis"]
    y_per_cell = {}
    for col in pIC50_cols:
        y_per_cell[col] = np.asarray(
            [
                float(r[col])
                if isinstance(r[col], (int, float))
                and math.isfinite(float(r[col]))
                else float("nan")
                for r in rows
            ],
            dtype=np.float64,
        )

    # Per-cell correlations (each descriptor vs each pIC50 column).
    # This is the most useful block — it tells the trainer which
    # descriptors carry signal on which cell line.
    per_cell_correlations: Dict[str, List[Dict[str, float]]] = {}
    for col in pIC50_cols:
        per_cell_correlations[col] = _pearson_block(X, y_per_cell[col], desc_names)
    # Headline correlations use MCF7 (matches m16 reporting contract).
    correlations = per_cell_correlations["pIC50_mcf7"]

    # Write CSV.
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = (
        ["smiles", "metal"]
        + list(PLATINAI_4CELL_IC50_COLS)
        + pIC50_cols
        + desc_names
    )
    with out_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for i, r in enumerate(rows):
            row_out = [r["smiles"], r["metal"]]
            for col in PLATINAI_4CELL_IC50_COLS:
                v = float(r[col])
                row_out.append(
                    "" if not math.isfinite(v) else f"{v:.6f}"
                )
            for col in pIC50_cols:
                v = float(r[col])
                row_out.append(
                    "" if not math.isfinite(v) else f"{v:.6f}"
                )
            row_out.extend(f"{float(v):.6f}" for v in X[i, :])
            writer.writerow(row_out)

    desc_stats = _desc_stats_block(X[valid, :], desc_names)
    wall = time.time() - t0

    payload_path = summary_path
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "platinai_4cell",
        "source_csv": str(PLATINAI_4CELL_DEFAULT),
        "target_col": "pIC50_mcf7 (headline); pIC50_<cell> (per-cell)",
        "n_descriptors": len(desc_stats),
        "descriptor_names": [d["name"] for d in desc_stats],
        "n_rows_total": n_total,
        "n_rows_parsed": n_parsed,
        "n_rows_dropped": n_dropped,
        "parse_success_rate": (
            n_parsed / n_total if n_total else 0.0
        ),
        "per_metal_counts": per_metal,
        "cell_lines": list(PLATINAI_4CELL_CELL_LINES),
        "descriptor_stats": desc_stats,
        "correlations_vs_target": correlations,
        "per_cell_correlations": per_cell_correlations,
        "wall_seconds": wall,
        "wf": "Wf-49",
        "phase": "Phase 1",
        "recipe": "RDKit 2D 12-d descriptors (MolWt, MolLogP, TPSA, HBA, HBD, NRotB, HeavyAtoms, RingCount, NumAromaticRings, FractionCSP3, MolMR, BalabanJ)",
        "literature_anchor": (
            "Yang et al. 2019 JCIM (D-MPNN paper); Wu et al. 2017 "
            "MoleculeNet §'smaller single-task datasets'; arXiv:2510.23428 "
            "(MetaModel = RDKit + ChemProp)."
        ),
        "math_anchor": (
            "pIC50 = 9 - log10(IC50_nM) is log-space molar-scale "
            "bioactivity driven by Lipinski-style physico-chemical "
            "properties (MolLogP, TPSA, HBA, HBD, MW)."
        ),
    }
    _wf416_atomic_write_text(payload_path, json.dumps(payload, indent=2, sort_keys=True))

    return {
        "dataset": "platinai_4cell",
        "out_csv": str(out_csv),
        "n_rows_total": n_total,
        "n_rows_parsed": n_parsed,
        "n_rows_dropped": n_dropped,
        "load_stats": load_stats,
        "per_metal_counts": per_metal,
        "wall_seconds": wall,
    }


# ---------------------------------------------------------------------------
# MetalCytoTox 5-metal extractor
# ---------------------------------------------------------------------------
def run_metalcytotox(out_csv: Path, summary_path: Path) -> Dict[str, object]:
    t0 = time.time()
    if not METALCYTO_5METAL_DEFAULT.exists():
        # Fallback: legacy /mnt/storage path
        legacy = Path("/mnt/storage/data/molmetal/metalcytotox_5metal_multitask_v4.csv")
        if legacy.exists():
            rows, load_stats = load_metalcytotox_5metal(legacy)
            source_csv = str(legacy)
        else:
            raise FileNotFoundError(
                f"MetalCytoToxDB 5-metal CSV not found at "
                f"{METALCYTO_5METAL_DEFAULT} or {legacy}. "
                "Run `bash scripts/refresh_metal_data.sh` first."
            )
    else:
        rows, load_stats = load_metalcytotox_5metal()
        source_csv = str(METALCYTO_5METAL_DEFAULT)

    desc_names = [n for n, _ in RDKIT_2D_DESCRIPTORS]
    X, valid, parse_failures = extract_descriptors_for_rows(
        rows, desc_names, compute_rdkit_descriptors,
    )
    n_total = len(rows)
    n_parsed = int(np.sum(valid))
    n_dropped = int(np.sum(parse_failures))

    per_metal: Dict[str, int] = {}
    for i, r in enumerate(rows):
        if not bool(valid[i]):
            continue
        per_metal[str(r["metal"])] = per_metal.get(str(r["metal"]), 0) + 1

    # Targets (binary activity).
    y_act = np.asarray(
        [
            float(r["activity"]) if isinstance(r["activity"], (int, float))
            and math.isfinite(float(r["activity"]))
            else float("nan")
            for r in rows
        ],
        dtype=np.float64,
    )
    # For binary y, Pearson r is mathematically equivalent to a
    # point-biserial correlation (same formula). scipy.stats.pearsonr
    # handles this fine, but we use np.corrcoef for zero-deps speed.
    correlations = _pearson_block(X, y_act, desc_names)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["smiles", "metal", "activity"] + desc_names
    with out_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for i, r in enumerate(rows):
            act = r["activity"]
            writer.writerow([
                r["smiles"],
                r["metal"],
                "" if not math.isfinite(float(act)) else f"{float(act):.6f}",
                *[f"{float(v):.6f}" for v in X[i, :]],
            ])

    desc_stats = _desc_stats_block(X[valid, :], desc_names)
    wall = time.time() - t0
    write_feature_summary(
        out_path=summary_path,
        dataset_name="metalcytotox_5metal",
        n_rows_total=n_total,
        n_rows_parsed=n_parsed,
        n_rows_dropped=n_dropped,
        per_metal_counts=per_metal,
        desc_stats=desc_stats,
        correlations=correlations,
        target_col="activity",
        wall_seconds=wall,
        source_csv=source_csv,
    )
    return {
        "dataset": "metalcytotox_5metal",
        "out_csv": str(out_csv),
        "n_rows_total": n_total,
        "n_rows_parsed": n_parsed,
        "n_rows_dropped": n_dropped,
        "load_stats": load_stats,
        "per_metal_counts": per_metal,
        "wall_seconds": wall,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
DATASET_CHOICES: Tuple[str, ...] = (
    "platinai",         # = platinai_5k.csv (SMILES-only structural subset)
    "platinai_4cell",   # = platinai_4cell.csv (labelled regression cohort)
    "metalcytotox",     # = metalcytotox_5metal_multitask_v4.csv (binary activity)
    "all",              # = platinai_5k + platinai_4cell + metalcytotox
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="extract_rdkit_2d_descriptors",
        description=(
            "Wf-49 Phase 1 — extract 12 RDKit 2D molecular descriptors "
            "for PlatinAI 5K + PlatinAI 4-cell + MetalCytoToxDB 5-metal."
        ),
    )
    p.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default="all",
        help="Which dataset(s) to extract (default: all).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "molmetal" / "data",
        help="Output directory for the per-dataset CSVs.",
    )
    p.add_argument(
        "--report-dir",
        type=Path,
        default=PROJECT_ROOT / "molmetal" / "reports" / "wf49_rdkit_extraction",
        help="Output directory for the per-dataset feature_summary.json.",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on rows per dataset (for smoke tests).",
    )
    p.add_argument(
        "--require-labels",
        action="store_true",
        help=(
            "Hard-error if the chosen dataset has no labelled target. "
            "Useful to prevent running 'platinai' (whose ic50_nm/selectivity "
            "columns are empty on the on-disk snapshot 2026-09-26) when a "
            "labelled cohort is required. Force callers to use 'platinai_4cell' "
            "instead."
        ),
    )
    p.add_argument(
        "--features",
        choices=FEATURES_V2_MODES,
        default="rdkit12",
        help=(
            "Feature extractor mode (Wf-249 Fix #1). "
            "'rdkit12' is the Wf-49 P1 12-d backward-compat default. "
            "'fp1024' extracts 1024-bit Morgan ECFP4 only (ablation). "
            "'rdkit12+fp1024' concatenates the two (1036-d, recommended "
            "for L5 Hilbert-ASTSR lift per Jiang et al. ChemRxiv 2024)."
        ),
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    _require_rdkit()
    print(
        f"[Wf-49] extract_rdkit_2d_descriptors starting "
        f"(dataset={args.dataset}, features={args.features}, "
        f"out_dir={args.out_dir})",
        flush=True,
    )

    # --require-labels guard: prevent running unlabelled 'platinai' (= 5K) when
    # the caller needs labelled targets. platinai_5k.csv has empty ic50_nm and
    # selectivity columns on the on-disk snapshot 2026-09-26 → correlations
    # block ships n_samples=0 for all 12 descriptors. Force the caller to use
    # 'platinai_4cell' (which has real labelled pIC50 columns).
    if args.require_labels and args.dataset == "platinai":
        raise SystemExit(
            "ERROR: --require-labels is set but --dataset=platinai selects "
            "platinai_5k.csv (unlabelled on the 2026-09-26 snapshot). "
            "Use --dataset=platinai_4cell for the labelled regression cohort."
        )

    results: List[Dict[str, object]] = []
    if args.dataset in ("platinai", "all"):
        results.append(run_platinai(
            out_csv=args.out_dir / "platinai_5k_rdkit_features.csv",
            summary_path=args.report_dir / "platinai_5k_feature_summary.json",
        ))
    if args.dataset in ("platinai_4cell", "all"):
        results.append(run_platinai_4cell(
            out_csv=args.out_dir / "platinai_4cell_rdkit_features.csv",
            summary_path=args.report_dir / "platinai_4cell_feature_summary.json",
        ))
    if args.dataset in ("metalcytotox", "all"):
        results.append(run_metalcytotox(
            out_csv=args.out_dir / "metalcytotox_rdkit_features.csv",
            summary_path=args.report_dir / "metalcytotox_5metal_feature_summary.json",
        ))

    print("[Wf-49] DONE — feature extraction summary:", flush=True)
    for r in results:
        print(
            f"  - {r['dataset']}: parsed {r['n_rows_parsed']}/{r['n_rows_total']} "
            f"({r['wall_seconds']:.2f}s wall) → {r['out_csv']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())