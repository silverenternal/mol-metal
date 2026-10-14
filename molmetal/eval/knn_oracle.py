"""knn_oracle.py — KNNEvaluator Tanimoto kNN oracle over a metallo-drug pool.

Wf-A37-4 (2026-10-15) NEW-1. Implements the oracle cited by
:mod:`mol-metal-anticancer` SKILL.md §3 as
``oracle = KNNEvaluator(ds, k=5, fp_radius=2, fp_bits=2048)``.

The oracle accepts an arbitrary (pool_smiles, pool_labels) pair and
scores a query SMILES by averaging the labels of its k nearest
neighbours (Tanimoto over RDKit Morgan-ECFP4), weighted by
``1 / max(eps, 1 - tanimoto)``. ``pool_labels`` is keyed by canonical
SMILES and may be partial — neighbours without labels are silently
dropped from the weighted sum for that cell line.

Public API
==========

- :class:`KNNEvaluator` — generic Tanimoto kNN over a labelled pool.
- :func:`build_default_knn_oracle` — factory that reads BOTH
  ``molmetal/data/metallo_drugs_500_train.csv`` and
  ``.data/molmetal/platinai_4cell.csv``, builds the COMPLETE oracle
  (PlatinAI 4-cell provides the labelled index; the 500-mol pool is
  exposed as a metallo-drug filter diagnostic).

Honest framing
==============

- The default oracle's pool is the COMPLETE PlatinAI 4-cell corpus
  (4078 mols, 4 cell lines: MCF7 / A549 / A2780 / A2780cis).  Every
  pool mol carries a per-cell-line pIC50 label.
- The 500-mol MaxMin pool (``metallo_drugs_500_train.csv``, 450 rows
  → 329 canonical SMILES) is NOT used as the kNN index — it is
  exposed as :attr:`KNNEvaluator.metallo_pool_size` and
  :attr:`KNNEvaluator.metallo_in_corpus` so callers can check whether
  a query sits inside the metallo-drug chemotype.
- Labels per ``.data/molmetal/platinai_4cell.meta.json`` are SYNTHETIC
  pIC50 (Rusanov 2025 supplementary S4-S5 is behind a Cloudflare JS
  challenge): useful as pipeline-test signal, NOT directly comparable
  to the Rusanov 2025 Table S4-S5 numbers.
- Scores are returned per cell line; ``NaN`` is propagated when no
  surviving neighbour has a label for that line.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, DataStructs

    _HAS_RDKIT = True
    RDLogger.DisableLog("rdApp.*")  # keep stdout clean
except Exception:  # pragma: no cover
    Chem = None
    AllChem = None
    DataStructs = None
    _HAS_RDKIT = False


# -----------------------------------------------------------------------------
# Canonical paths
# -----------------------------------------------------------------------------
DEFAULT_POOL_PATH: str = "molmetal/data/metallo_drugs_500_train.csv"
# Wf-A37-fixup-1 COMPLETE fix (REAL MB-Finder harvest): prefer the
# real-label pool when present, fall back to the synthetic placeholder
# only if neither real source is on disk.
DEFAULT_LABEL_PATH: str = (
    ".data/molmetal/platinai_4cell_real_label_pool.csv"
    if os.path.isfile(".data/molmetal/platinai_4cell_real_label_pool.csv")
    else ".data/molmetal/platinai_4cell.csv"
)
DEFAULT_REAL_POOL_CSV: str = ".data/molmetal/platinai_4cell_real_label_pool.csv"
DEFAULT_FALLBACK_POOL_CSV: str = ".data/molmetal/platinai_4cell.csv"

DEFAULT_K: int = 5
DEFAULT_FP_RADIUS: int = 2
DEFAULT_FP_BITS: int = 2048
DEFAULT_LABEL_CELLS: Tuple[str, ...] = (
    "pIC50_mcf7",
    "pIC50_a549",
    "pIC50_a2780",
    "pIC50_a2780cis",
)
DEFAULT_LABEL_CELL_SHORT: Tuple[str, ...] = ("mcf7", "a549", "a2780", "a2780cis")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _safe_canon(smiles: str) -> Optional[str]:
    """Return RDKit-canonical SMILES, or None on parse failure."""
    if not _HAS_RDKIT or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def _morgan_fp(smiles: str, radius: int, n_bits: int):
    """Return Morgan fingerprint or None."""
    if not _HAS_RDKIT or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Result row
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class KNNNeighbour:
    """One Tanimoto kNN hit."""

    smiles: str  # canonical SMILES of the neighbour
    tanimoto: float  # in [0, 1]


@dataclass(frozen=True)
class KNNScore:
    """Per-query kNN output: per-cell-line weighted-mean score."""

    smiles: str  # canonical SMILES of the query
    scores: Dict[str, float]  # cell-line short name -> weighted mean (NaN = no signal)
    neighbours: Tuple[KNNNeighbour, ...]  # the surviving top-k neighbours


# -----------------------------------------------------------------------------
# Core class
# -----------------------------------------------------------------------------
class KNNEvaluator:
    """Tanimoto-kNN oracle over a (pool_smiles, pool_labels) pair.

    Parameters
    ----------
    pool_smiles : Sequence[str]
        Canonical SMILES of the pool.  Each is fingerprinted once at
        construction time.  Duplicates are deduplicated by canonical form
        (last write wins).
    pool_labels : Dict[str, Dict[str, float]]
        ``{canonical_smiles: {cell_short_name: pIC50_value}}``.
        Cell short names default to ``("mcf7", "a549", "a2780",
        "a2780cis")``.  Mols in ``pool_smiles`` that are absent from
        ``pool_labels`` still appear in the index but contribute no
        label weight (they degrade to ``NaN`` for that cell line).
    k : int, default 5
        Number of nearest neighbours to keep.  If fewer than ``k``
        pool mols parse, the actual returned neighbour count is
        ``min(k, parseable_pool_size)``.
    fp_radius : int, default 2
        Morgan fingerprint radius (ECFP4 = radius 2).
    fp_bits : int, default 2048
        Morgan fingerprint bit width.
    cell_names : Sequence[str], optional
        Override the default cell-line short names.  Must match the
        keys used in ``pool_labels``.
    """

    def __init__(
        self,
        pool_smiles: Sequence[str],
        pool_labels: Dict[str, Dict[str, float]],
        *,
        k: int = DEFAULT_K,
        fp_radius: int = DEFAULT_FP_RADIUS,
        fp_bits: int = DEFAULT_FP_BITS,
        cell_names: Sequence[str] = DEFAULT_LABEL_CELL_SHORT,
    ) -> None:
        self.k = int(k)
        self.fp_radius = int(fp_radius)
        self.fp_bits = int(fp_bits)
        self.cell_names: Tuple[str, ...] = tuple(cell_names)
        self.pool_labels: Dict[str, Dict[str, float]] = dict(pool_labels)
        # Deduplicate pool by canonical SMILES; fingerprinted once.
        self.pool_smiles: List[str] = []
        self.pool_fingerprints = []
        seen = set()
        for raw in pool_smiles:
            c = _safe_canon(raw)
            if c is None or c in seen:
                continue
            fp = _morgan_fp(c, self.fp_radius, self.fp_bits)
            if fp is None:
                continue
            seen.add(c)
            self.pool_smiles.append(c)
            self.pool_fingerprints.append(fp)
        self.labelled_in_pool = sum(
            1 for c in self.pool_smiles if c in self.pool_labels
        )
        # Metallo-drug filter (populated by build_default_knn_oracle
        # to expose how many pool mols sit inside the 500-mol
        # metallo-drug chemotype; remains None for hand-built oracles).
        self._metallo_pool: Optional[List[str]] = None
        self.metallo_pool_size: int = 0
        self.metallo_in_corpus: int = 0

    # ---- diagnostics -----------------------------------------------------
    @property
    def pool_size(self) -> int:
        """Number of canonical, parseable pool mols (== fingerprints)."""
        return len(self.pool_smiles)

    def available(self) -> bool:
        """True iff the pool has at least one parseable mol."""
        return self.pool_size > 0 and _HAS_RDKIT

    # ---- scoring ---------------------------------------------------------
    def find_neighbours(self, smiles: str) -> List[KNNNeighbour]:
        """Return the top-k Tanimoto neighbours sorted by descending sim.

        Returns an empty list when the query SMILES does not parse,
        when the pool is empty, or when RDKit is unavailable.
        """
        if not self.available():
            return []
        canon = _safe_canon(smiles)
        if canon is None:
            return []
        fp = _morgan_fp(canon, self.fp_radius, self.fp_bits)
        if fp is None:
            return []
        try:
            sims = DataStructs.BulkTanimotoSimilarity(fp, self.pool_fingerprints)
        except Exception:
            return []
        n = min(self.k, len(sims))
        indexed = sorted(enumerate(sims), key=lambda kv: kv[1], reverse=True)[:n]
        return [
            KNNNeighbour(smiles=self.pool_smiles[i], tanimoto=float(t))
            for i, t in indexed
        ]

    def predict(self, smiles: str) -> KNNScore:
        """Score ``smiles`` against the pool.

        Per cell line, the score is::

            score_c = sum_i (w_i * label_i_c) / sum_i w_i
            w_i     = 1 / max(eps, 1 - tanimoto_i)

        where the sum runs over the top-k neighbours that carry a
        non-NaN label for cell line ``c``.  If no neighbour carries
        a label for ``c``, the score for ``c`` is ``NaN`` (graceful
        degrade — not 0.0, since 0.0 would be a MEASURED claim).
        """
        neighbours = self.find_neighbours(smiles)
        eps = 1e-6
        canon = _safe_canon(smiles) or smiles
        if not neighbours:
            return KNNScore(
                smiles=canon,
                scores={c: float("nan") for c in self.cell_names},
                neighbours=(),
            )
        # Per-cell weighted sum.
        sums: Dict[str, float] = {c: 0.0 for c in self.cell_names}
        dens: Dict[str, float] = {c: 0.0 for c in self.cell_names}
        for nb in neighbours:
            labels = self.pool_labels.get(nb.smiles)
            if not labels:
                continue
            w = 1.0 / max(eps, 1.0 - nb.tanimoto)
            for c in self.cell_names:
                v = labels.get(c)
                if v is None or v != v:  # NaN-safe
                    continue
                sums[c] += w * float(v)
                dens[c] += w
        scores: Dict[str, float] = {}
        for c in self.cell_names:
            if dens[c] <= 0.0:
                scores[c] = float("nan")
            else:
                scores[c] = sums[c] / dens[c]
        return KNNScore(
            smiles=canon,
            scores=scores,
            neighbours=tuple(neighbours),
        )

    def predict_batch(self, smiles_list: Iterable[str]) -> List[KNNScore]:
        """Vector-friendly: predict a sequence of SMILES."""
        return [self.predict(s) for s in smiles_list]


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------
def _load_pool_smiles(path: str) -> List[str]:
    """Read ``smiles`` column from the 500-mol pool CSV."""
    out: List[str] = []
    if not os.path.isfile(path):
        return out
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            smi = row.get("smiles") or row.get("SMILES_Ligands") or ""
            if smi:
                out.append(smi)
    return out


def _load_platinai_4cell(path: str) -> Dict[str, Dict[str, float]]:
    """Read PlatinAI 4-cell CSV → ``{canonical_smi: {cell_short: pIC50}}``.

    Auto-detects REAL vs SYNTHETIC schema.  The Wf-A37-fixup-1 REAL
    pool (``platinai_4cell_real_label_pool.csv``) schema is
    ``compound_code, canonical_smiles, cell_line_long, cell_line_short,
    value_mean, value_sd, doi, method, incubationTime, label_source`` —
    in REAL MB-Finder per-cell-line IC50, not synthetic pIC50.  The
    older synthetic schema
    (``SMILES_Ligands, pIC50_mcf7, pIC50_a549, pIC50_a2780,
    pIC50_a2780cis, metal``) is read when ``SMILES_Ligands`` is the
    header, preserving Wf-293B backward compatibility.

    Returns ``{canonical_smiles: {cell_short: numeric_label}}``.  Rows
    with empty canonical_smiles (REAL pool, SMILES not yet joined) are
    SKIPPED from the Tanimoto pool — their compound_code + value_mean
    still appear in the JSON provenance via the merged CSV.
    """
    out: Dict[str, Dict[str, float]] = {}
    if not os.path.isfile(path):
        return out
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        headers = r.fieldnames or []
        # Wf-A37-fixup-1: REAL pool schema (canonical_smiles + cell_line_short)
        is_real_pool = (
            "canonical_smiles" in headers and "cell_line_short" in headers
        )
        if is_real_pool:
            for row in r:
                smi = (row.get("canonical_smiles") or "").strip()
                if not smi:
                    continue  # SMILES join still in progress; skip from Tanimoto pool
                canon = _safe_canon(smi) or smi
                cl = (row.get("cell_line_short") or "").strip()
                try:
                    val = float(row.get("value_mean"))
                except (TypeError, ValueError):
                    continue
                if not cl or val <= 0:
                    continue
                # Wf-A37-fixup-1: convert MB-Finder raw IC50 (μM) to
                # pIC50 so the cell-line axis shares the same units as
                # the synthetic platinai_4cell.csv schema.
                # pIC50 = -log10(IC50_M) = 6 - log10(IC50_uM).
                pic50 = 6.0 - math.log10(val)
                if canon in out:
                    prev = out[canon]
                    if cl in prev:
                        prev[cl] = 0.5 * (prev[cl] + pic50)
                    else:
                        prev[cl] = pic50
                else:
                    out[canon] = {cl: pic50}
        else:
            # Wf-293B / pre-A37 synthetic schema: SMILES_Ligands + pIC50_<cell>
            for row in r:
                raw = row.get("SMILES_Ligands") or row.get("smiles") or ""
                canon = _safe_canon(raw)
                if canon is None:
                    continue
                cell_dict: Dict[str, float] = {}
                for short, long in zip(DEFAULT_LABEL_CELL_SHORT, DEFAULT_LABEL_CELLS):
                    try:
                        cell_dict[short] = float(row[long])
                    except (KeyError, ValueError, TypeError):
                        continue
                if canon in out:
                    prev = out[canon]
                    for c, v in cell_dict.items():
                        if c in prev:
                            prev[c] = 0.5 * (prev[c] + v)
                        else:
                            prev[c] = v
                else:
                    out[canon] = cell_dict
    return out


# -----------------------------------------------------------------------------
# Default factory — wires PlatinAI 4-cell as the COMPLETE labelled pool,
# the 500-mol pool as a metallo-drug filter diagnostic.
# -----------------------------------------------------------------------------
def build_default_knn_oracle(
    pool_path: str = DEFAULT_POOL_PATH,
    label_path: str = DEFAULT_LABEL_PATH,
    *,
    k: int = DEFAULT_K,
    fp_radius: int = DEFAULT_FP_RADIUS,
    fp_bits: int = DEFAULT_FP_BITS,
) -> KNNEvaluator:
    """Construct a :class:`KNNEvaluator` that scores against the COMPLETE
    PlatinAI 4-cell corpus (4078 mols, every mol carries a 4-cell
    pIC50 label).  The 500-mol MaxMin pool is read separately and
    exposed as a metallo-drug filter:

    - :attr:`KNNEvaluator.metallo_pool_size` — number of canonical
      SMILES in the 500-mol pool (metallo drug chemotype).
    - :attr:`KNNEvaluator.metallo_in_corpus` — how many of those
      metallo-drug SMILES are also in the kNN corpus (intersection
      count).

    This two-source wiring gives callers a fully-functional oracle
    (every query gets real per-cell-line scores) AND a way to check
    whether their query sits inside the metallo-drug chemotype.
    """
    # The labelled pool = PlatinAI 4-cell (4078 mols, all labelled).
    labels = _load_platinai_4cell(label_path)
    pool = list(labels.keys())
    oracle = KNNEvaluator(
        pool_smiles=pool,
        pool_labels=labels,
        k=k,
        fp_radius=fp_radius,
        fp_bits=fp_bits,
    )
    # Metallo-drug filter diagnostic (500-mol pool).
    metallo_raw = _load_pool_smiles(pool_path)
    metallo_canon: List[str] = []
    seen = set()
    for raw in metallo_raw:
        c = _safe_canon(raw)
        if c is None or c in seen:
            continue
        seen.add(c)
        metallo_canon.append(c)
    oracle._metallo_pool = metallo_canon
    oracle.metallo_pool_size = len(metallo_canon)
    oracle.metallo_in_corpus = sum(1 for c in metallo_canon if c in labels)
    return oracle


__all__ = [
    "DEFAULT_POOL_PATH",
    "DEFAULT_LABEL_PATH",
    "DEFAULT_K",
    "DEFAULT_FP_RADIUS",
    "DEFAULT_FP_BITS",
    "DEFAULT_LABEL_CELLS",
    "DEFAULT_LABEL_CELL_SHORT",
    "KNNNeighbour",
    "KNNScore",
    "KNNEvaluator",
    "build_default_knn_oracle",
]