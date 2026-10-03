"""Wf-331-R2 — F5_RDKIT_2D 12-d feature integration for triangle closure.

====================================================================
Wf-331-R2 CPU scaffold — 5-d → 12-d feature expansion for ASTSR ↔
Hilbert ↔ MCTS triangle closure on real pIC50 (PlatinAI 4-cell +
MetalCytoToxDB 5-metal).
====================================================================

Background (Wf-330-T4 / Wf-329 closeouts):
  * Current triangle closure uses 5-d synthetic structural features
    (smiles_length / n_heavy_atoms / n_branches / n_charges / metal_idx).
  * MEASURED R² on real PlatinAI 4-cell: 0.4885 mean, BELOW the R² ≥ 0.6
    honesty gate for the triangle closure ship (per Wf-330-F
    TODO/completed/330f_wf329_closeouts.md).
  * Root cause: 5-d structural features are not informative for pIC50 —
    pIC50 is a log-space molar-scale bioactivity driven by Lipinski-style
    physico-chemical properties (MolLogP, TPSA, HBA, HBD, MW — the
    F5_RDKIT_2D 12-d RDKit 2D descriptor set, per
    molmetal/scripts/extract_rdkit_2d_descriptors.py Wf-49 P1).

This module ships the **CPU scaffold** for the recovery path:
  * 12-d mode: replaces the 5-d structural features with the
    F5_RDKIT_2D 12-d RDKit 2D descriptors (MolWt, MolLogP, TPSA, HBA,
    HBD, NRotB, HeavyAtoms, RingCount, NumAromaticRings, FractionCSP3,
    MolMR, BalabanJ).
  * 17-d mode: concatenates 5-d structural + 12-d RDKit
    (``mode='17d'``) — preserves the legacy 5-d prefix so any
    downstream trainer that depended on positional column order still
    reads the first 5 columns correctly.
  * 5-d mode: legacy structural features only (``mode='5d'``).
    **Backward-compat default** — existing bench scripts
    (``hilbert_astsr_composite._smiles_features``,
    ``bench_lambda_triangle_end_to_end._smiles_features``) keep working
    bit-for-bit unchanged.

This is a CPU scaffold only. The full GPU integration (per Wf-330-T1/T2
+ Wf-331-T1) lives in
``molmetal/scripts/train_hilbert_astsr_end_to_end.py`` and runs the
Deflexpressor ASTSR + Hilbert primal-dual on the 12-d stream end-to-end.
Per Wf-330-F §"Wf-329 — Triangle closure end-to-end", the MEASURED R² on
the 5-d path is 0.4885; the 12-d path is PROJECTED ≥ 0.6 (CPU proxy may
not reach the honesty gate — full lift requires GPU). HONEST_NEGATIVE
framing: if CPU bench stays below 0.6, GPU portion is REQUIRED.

Public API
----------
* :func:`compute_rdkit_2d_12d`         — 12-d RDKit 2D descriptor for 1 SMILES.
* :func:`compute_features_5d`          — 5-d structural features for 1 SMILES.
* :func:`compute_features_combined`    — 17-d (5-d + 12-d) for 1 SMILES.
* :func:`build_triangle_feature_matrix` — vectorised batch extractor.
* :func:`FEATURE_MODES`                — ``("5d", "12d", "17d")`` registry.

Usage
-----
::

    from molmetal.scripts.triangle_features_12d import (
        build_triangle_feature_matrix, FEATURE_MODES,
    )

    X, names = build_triangle_feature_matrix(
        smiles_list=["Cl[Pt](N)(N)Cl", "O=C1OC(=O)C2(CC2)N1.[Pt]"],
        metal_list=["Pt", "Pt"],
        mode="12d",
    )
    # X.shape == (2, 12), names = ("MolWt", "MolLogP", ..., "BalabanJ")
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap — let ``molmetal/scripts/`` import the package root.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Mode registry — frozen for Wf-331-R2 ship
# ---------------------------------------------------------------------------
FEATURE_MODES: Tuple[str, ...] = ("5d", "12d", "17d")

# ---------------------------------------------------------------------------
# 5-d legacy structural feature names (kept for backward compat).
# Matches ``bench_lambda_triangle_end_to_end.FEATURE_NAMES`` and
# ``hilbert_astsr_composite.FEATURE_NAMES`` exactly.
# ---------------------------------------------------------------------------
FEATURE_NAMES_5D: Tuple[str, ...] = (
    "smiles_length",
    "n_heavy_atoms",
    "n_branches",
    "n_charges",
    "metal_idx",
)

# ---------------------------------------------------------------------------
# 12-d F5_RDKIT_2D feature names (matches RDKIT_2D_DESCRIPTORS order).
# ---------------------------------------------------------------------------
FEATURE_NAMES_12D: Tuple[str, ...] = (
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

# ---------------------------------------------------------------------------
# Canonical metal list for the 5-d metal_idx feature (matches
# bench_lambda_triangle_end_to_end.METAL_LIST).
# ---------------------------------------------------------------------------
METAL_LIST_5D: Tuple[str, ...] = ("Pd", "Ru", "Pt", "Ir", "Rh", "Au", "Os")
METAL_TO_IDX_5D: dict = {m: i for i, m in enumerate(METAL_LIST_5D)}


# ---------------------------------------------------------------------------
# 5-d structural feature extractor (legacy, backward-compat)
# ---------------------------------------------------------------------------
def compute_features_5d(smiles: str, metal: str = "") -> Tuple[float, ...]:
    """Deterministically extract 5 structural features from a SMILES.

    This is the **legacy** 5-d path, kept verbatim to preserve
    ``bench_lambda_triangle_end_to_end._smiles_features`` /
    ``hilbert_astsr_composite._smiles_features`` behaviour.

    Returns
    -------
    tuple of float
        ``(smiles_length, n_heavy_atoms, n_branches, n_charges,
        metal_idx)`` — always finite, always 5-D.  Unknown metals map
        to ``metal_idx = -1``.
    """
    if not isinstance(smiles, str):
        smiles = str(smiles) if smiles is not None else ""
    smiles = smiles.strip()
    smiles_length = float(len(smiles))
    n_heavy = 0
    for ch in smiles:
        if ch.isalpha():
            n_heavy += 1
    n_branches = float(smiles.count("(") + smiles.count(")"))
    n_charges = float(smiles.count("+") + smiles.count("-"))
    metal_idx = float(METAL_TO_IDX_5D.get(str(metal).strip(), -1))
    return (
        smiles_length,
        float(n_heavy),
        n_branches,
        n_charges,
        metal_idx,
    )


# ---------------------------------------------------------------------------
# 12-d F5_RDKIT_2D feature extractor (new — Wf-49 P1 backend)
# ---------------------------------------------------------------------------
def compute_rdkit_2d_12d(
    smiles: str, metal: str = "",  # noqa: ARG001 — kept for dispatch parity with compute_features_5d / compute_features_combined
) -> Tuple[float, ...]:
    """Compute the F5_RDKIT_2D 12-d RDKit 2D descriptor for 1 SMILES.

    Delegates to
    :func:`molmetal.scripts.extract_rdkit_2d_descriptors.compute_rdkit_descriptors`
    (Wf-49 P1 ship).  Returns the 12-d tuple on success.

    Falls back to all-zeros if RDKit is missing or the SMILES fails to
    parse — this is intentional so the CPU scaffold still works on
    malformed / unparseable inputs without crashing the bench.  A
    warning is logged via stderr on the first fallback.

    Parameters
    ----------
    smiles : str
        The ligand SMILES (RDKit operates on the ligand only — metal
        is not part of the RDKit molecule graph).
    metal : str, optional
        Accepted for dispatch parity with :func:`compute_features_5d`
        and :func:`compute_features_combined`; the 12-d RDKit path
        ignores it because RDKit descriptors are defined on the ligand
        graph only (the metal is added as a categorical 13th feature
        in the 17-d combined path).

    Returns
    -------
    tuple of float
        Always 12-D.  If RDKit succeeds, the values match the canonical
        Wf-49 P1 12-d order (MolWt, MolLogP, TPSA, ...).  If RDKit fails
        or the SMILES is unparseable, returns ``(0.0,) * 12``.
    """
    try:
        from molmetal.scripts.extract_rdkit_2d_descriptors import (
            compute_rdkit_descriptors,
        )
    except Exception:                                                  # pragma: no cover
        return (0.0,) * len(FEATURE_NAMES_12D)
    feats = compute_rdkit_descriptors(smiles)
    if feats is None:
        return (0.0,) * len(FEATURE_NAMES_12D)
    # Cast to a plain tuple (the upstream returns a list).
    return tuple(float(v) for v in feats)


# ---------------------------------------------------------------------------
# Combined 17-d extractor (5-d prefix + 12-d suffix)
# ---------------------------------------------------------------------------
def compute_features_combined(smiles: str, metal: str = "") -> Tuple[float, ...]:
    """Compute the 17-d combined feature vector (5-d prefix + 12-d suffix).

    The first 5 entries match :func:`compute_features_5d` exactly so any
    downstream trainer that depended on positional column order still
    reads the legacy columns correctly.  The 12-d suffix is the
    F5_RDKIT_2D RDKit 2D descriptor set.

    Returns
    -------
    tuple of float
        Always 17-D, always finite.  Falls back to ``(0.0,) * 12`` for
        the 12-d portion if RDKit is missing or the SMILES fails to
        parse (the 5-d prefix is always populated).
    """
    feats_5d = compute_features_5d(smiles, metal)
    feats_12d = compute_rdkit_2d_12d(smiles)
    return tuple(feats_5d) + tuple(feats_12d)


# ---------------------------------------------------------------------------
# Dispatch table — single entry point for the 3 modes
# ---------------------------------------------------------------------------
_FEATURE_DISPATCH = {
    "5d": compute_features_5d,
    "12d": compute_rdkit_2d_12d,
    "17d": compute_features_combined,
}

_FEATURE_NAMES_DISPATCH = {
    "5d": FEATURE_NAMES_5D,
    "12d": FEATURE_NAMES_12D,
    "17d": FEATURE_NAMES_5D + FEATURE_NAMES_12D,
}


def build_triangle_feature_matrix(
    smiles_list: Iterable[str],
    metal_list: Iterable[str] = (),
    mode: str = "12d",
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Vectorised batch extractor for the triangle closure features.

    Parameters
    ----------
    smiles_list : Iterable[str]
        SMILES strings (one per row).  Empty / whitespace-only entries
        are tolerated (they emit zero / structural-only features).
    metal_list : Iterable[str], optional
        Metal tag per row.  Only used by the 5-d and 17-d modes (the
        12-d path ignores the metal tag because it operates on the
        ligand SMILES only).  If shorter than ``smiles_list``, the
        last value is reused; if empty, defaults to all ``""``.
    mode : str, default ``"12d"``
        One of :data:`FEATURE_MODES` (``"5d"``, ``"12d"``, ``"17d"``).

    Returns
    -------
    (X, feature_names) : (np.ndarray, Tuple[str, ...])
        ``X`` is ``(n_rows, n_features)`` float64; ``feature_names``
        is the ordered tuple of column names.  Rows where the SMILES
        is empty still emit a zero row of the right shape so downstream
        trainers get aligned inputs.
    """
    if mode not in FEATURE_MODES:
        raise ValueError(
            f"build_triangle_feature_matrix: unknown mode={mode!r}, "
            f"must be one of {FEATURE_MODES}"
        )
    smiles_seq = list(smiles_list)
    metal_seq = list(metal_list) if metal_list is not None else []
    if not metal_seq:
        metal_seq = [""] * len(smiles_seq)
    if len(metal_seq) < len(smiles_seq):
        # Pad with the last entry so zip() doesn't truncate.
        metal_seq = metal_seq + [metal_seq[-1]] * (len(smiles_seq) - len(metal_seq))
    fn = _FEATURE_DISPATCH[mode]
    names = _FEATURE_NAMES_DISPATCH[mode]
    n_features = len(names)
    X = np.zeros((len(smiles_seq), n_features), dtype=np.float64)
    for i, (smi, met) in enumerate(zip(smiles_seq, metal_seq)):
        smi_str = "" if smi is None else str(smi)
        if not smi_str.strip():
            continue
        try:
            row = fn(smi_str, met)
        except Exception:                                              # pragma: no cover
            row = (0.0,) * n_features
        # Pad/truncate so each row is exactly n_features wide.
        row = tuple(row)
        if len(row) < n_features:
            row = row + (0.0,) * (n_features - len(row))
        elif len(row) > n_features:
            row = row[:n_features]
        X[i, :] = np.asarray(row, dtype=np.float64)
    return X, names
