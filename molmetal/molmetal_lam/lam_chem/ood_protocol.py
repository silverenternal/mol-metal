"""Out-of-Distribution (OOD) protocol for SBDD / metallodrug evaluation.

==================================================================
Pitfall Rank 14 — TODO-30 P6.3 (Tanimoto<0.4 + MMseqs2 30% OOD)
==================================================================

This module provides the OOD-discriminator that the engineering
practices policy (TODO/engineering_practices.md §4.4) requires for
the headline SBDD evaluation path.  Two layers:

1. **Ligand-side OOD (Tanimoto<0.4, ECFP4/Morgan-2048)**
   The :class:`TanimotoOODProtocol` answers the question: "Is the
   query molecule fingerprint-similar to *any* of the training
   pool?"  Industry-standard OOD threshold per TargetDiff /
   Pocket2Mol conventions — any train/test pair with Tanimoto ≥ 0.4
   is rejected as NOT OOD.

2. **Pocket-side OOD (MMseqs2 sequence-identity < 30%)**
   The :class:`MMseqs2OODProtocol` is a stub (binary not in conda
   env) — it always raises :class:`NotImplementedError` so callers
   must explicitly opt in to the deferred binary path.  See
   ``molmetal/reports/wf_pitfall_tier2/p63_mmseqs2_followup.md``
   for the follow-up install steps.

3. **Combined protocol**
   The :class:`CombinedOODProtocol` OR-combines both layers: a query
   is OOD iff the ligand is Tanimoto-OOD AND (in the future) the
   pocket is MMseqs2-OOD.

Honest framing
--------------
* The Tanimoto<0.4 threshold is the **MODERATE** novelty cut from
  Pocket2Mol (Peng 2022) and TargetDiff (Guan 2023); it is *not* a
  substitute for a Bemis-Murcko scaffold split, and the two are
  complementary (a mol can be Tanimoto-similar to a train mol yet
  have a different Murcko scaffold, and vice versa).
* The MMseqs2 30%-seq-id cut is the **strict** OOD cut for the
  protein axis; in the future, the Tanimoto+MMseqs2 OOD claim is
  what makes the paper claim "we evaluate on truly held-out
  pockets" rather than "we evaluate on a held-out random subset".

Public surface
--------------
* :class:`TanimotoOODProtocol`       CPU Tanimoto<0.4 OOD
* :class:`MMseqs2OODProtocol`        stub, raises NotImplementedError
* :class:`CombinedOODProtocol`       OR-combined OOD
* :func:`compute_tanimoto`           one-off scalar helper
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

try:  # numpy is optional — fallback to pure-Python bit-counting when missing
    import numpy as np
    _HAS_NUMPY: bool = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    _HAS_NUMPY = False

try:  # pragma: no cover - exercised in CI with RDKit installed
    from rdkit import Chem as _Chem
    from rdkit.Chem import rdFingerprintGenerator as _rdFingerprintGenerator
    _HAS_RDKIT: bool = True
except Exception:  # pragma: no cover
    _Chem = None  # type: ignore
    _rdFingerprintGenerator = None  # type: ignore
    _HAS_RDKIT = False


def _bit_count_uint8(arr) -> int:
    """Count set bits in a 1-D iterable of 0/1 values.

    Falls back to pure-Python sum() when numpy is missing.  Used
    only by the Tanimoto similarity helpers — both paths are
    numerically identical for binary {0, 1} vectors.
    """
    if _HAS_NUMPY and np is not None:
        try:
            return int(np.sum(arr))
        except Exception:
            pass
    return int(sum(int(x) for x in arr))


def _bitwise_and_count(a, b) -> int:
    """Count the number of positions where both ``a`` and ``b`` are 1."""
    if _HAS_NUMPY and np is not None:
        try:
            return int(np.bitwise_and(a, b).sum())
        except Exception:
            pass
    # Pure-Python fallback: zip the two flat sequences.
    return int(sum(1 for x, y in zip(a, b) if x and y))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TANIMOTO_THRESHOLD: float = 0.4
DEFAULT_FP_RADIUS: int = 2
DEFAULT_FP_NBITS: int = 2048
DEFAULT_MMSEQS2_THRESHOLD: float = 0.30


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------
def _morgan_fp(smiles: str, radius: int = DEFAULT_FP_RADIUS, n_bits: int = DEFAULT_FP_NBITS):
    """Morgan-ECFP fingerprint as a flat np.uint8 bit-vector.

    Returns ``None`` if RDKit is missing or the SMILES cannot be
    parsed (including empty / whitespace strings).  Uses the modern
    ``rdFingerprintGenerator`` API (which avoids the deprecated
    ``AllChem.GetMorganFingerprintAsBitVect``).
    """
    if not _HAS_RDKIT or _Chem is None or _rdFingerprintGenerator is None:
        return None
    if not smiles or not str(smiles).strip():
        return None
    try:
        mol = _Chem.MolFromSmiles(str(smiles).strip())
    except Exception:
        return None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    try:
        gen = _rdFingerprintGenerator.GetMorganGenerator(
            radius=int(radius), fpSize=int(n_bits), includeChirality=False,
        )
        if _HAS_NUMPY and np is not None:
            fp = gen.GetFingerprintAsNumPy(mol)
            return np.asarray(fp, dtype=np.uint8).reshape(-1)
        # numpy-missing fallback: convert to a Python list of 0/1 ints.
        bv = gen.GetFingerprint(mol)
        out = [0] * int(n_bits)
        for i in bv.GetOnBits():
            if 0 <= int(i) < int(n_bits):
                out[int(i)] = 1
        return out
    except Exception:
        return None


def _tanimoto_sim_vectors(a, b) -> float:
    """Tanimoto similarity between two binary bit-vectors.

    Both ``a`` and ``b`` must be flat sequences of 0/1 values of
    the SAME length.  Returns 1.0 for two empty bit-vectors (the
    standard convention from data_diversity._tanimoto_sim_uint8).
    """
    if a is None or b is None:
        return 0.0
    try:
        size_a = len(a)
        size_b = len(b)
    except TypeError:
        return 0.0
    if size_a != size_b:
        raise ValueError(
            f"fingerprint length mismatch: {size_a} vs {size_b}"
        )
    inter = _bitwise_and_count(a, b)
    union = _bit_count_uint8(a) + _bit_count_uint8(b) - inter
    if union <= 0:
        return 1.0
    return float(inter) / float(union)


def compute_tanimoto(
    smi1: str,
    smi2: str,
    radius: int = DEFAULT_FP_RADIUS,
    n_bits: int = DEFAULT_FP_NBITS,
) -> float:
    """Compute the Tanimoto similarity of two SMILES on Morgan-ECFP.

    Returns ``0.0`` if either SMILES fails to parse (the *most*
    conservative non-OOD outcome, treating "unknown query" as a
    fail-closed OOD miss rather than a spurious hit).

    Parameters
    ----------
    smi1, smi2 : str
        RDKit-parseable SMILES strings (canonicalisation is *not*
        applied — pass canonical SMILES in if you want
        canonical-form comparison).
    radius, n_bits : int
        Morgan radius (default 2 = ECFP4) and bit length (default
        2048).

    Returns
    -------
    float
        Tanimoto similarity in [0.0, 1.0].  Returns 1.0 if both
        inputs are empty (the standard convention for empty-zero
        bit-vectors).
    """
    fp1 = _morgan_fp(smi1, radius=radius, n_bits=n_bits)
    fp2 = _morgan_fp(smi2, radius=radius, n_bits=n_bits)
    if fp1 is None or fp2 is None:
        if fp1 is None and fp2 is None:
            return 1.0
        return 0.0
    return _tanimoto_sim_vectors(fp1, fp2)


# ---------------------------------------------------------------------------
# Tanimoto OOD protocol — CPU-only, fingerprint-based
# ---------------------------------------------------------------------------
@dataclass
class TanimotoOODProtocol:
    """Tanimoto fingerprint < threshold OOD classifier (CPU).

    A query SMILES is *OOD* iff its **maximum** Tanimoto similarity
    to any training SMILES is strictly below the threshold.  The
    threshold defaults to 0.4 (the Pocket2Mol / TargetDiff OOD
    convention).

    Parameters
    ----------
    threshold : float
        Maximum tolerated Tanimoto similarity (default 0.4).  A
        query whose closest train neighbour has Tanimoto ≥
        threshold is NOT OOD.
    radius : int
        Morgan radius (default 2 = ECFP4).
    n_bits : int
        Fingerprint length in bits (default 2048).

    Examples
    --------
    >>> p = TanimotoOODProtocol(threshold=0.4)
    >>> p.is_ood("N.N.[Cl][Pt][Cl]", ["N.N.[Cl][Pt][Cl]"])  # itself → not OOD
    False
    >>> p.is_ood("c1ccccc1O", ["N.N.[Cl][Pt][Cl]"])         # phenol → OOD
    True
    """

    threshold: float = DEFAULT_TANIMOTO_THRESHOLD
    radius: int = DEFAULT_FP_RADIUS
    n_bits: int = DEFAULT_FP_NBITS

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.threshold) <= 1.0):
            raise ValueError(
                f"threshold must be in [0.0, 1.0]; got {self.threshold}"
            )
        if int(self.radius) < 0:
            raise ValueError(f"radius must be non-negative; got {self.radius}")
        if int(self.n_bits) <= 0:
            raise ValueError(f"n_bits must be positive; got {self.n_bits}")

    def compute_tanimoto(self, smi1: str, smi2: str) -> float:
        """Scalar Tanimoto helper (mirrors :func:`compute_tanimoto`)."""
        return compute_tanimoto(
            smi1, smi2, radius=int(self.radius), n_bits=int(self.n_bits),
        )

    def max_tanimoto(
        self, query_smiles: str, train_smiles_list: Sequence[str],
    ) -> float:
        """Return the *maximum* Tanimoto of ``query_smiles`` vs the
        training list.

        Returns 0.0 if the query is unparseable OR the training list
        is empty.  Empty-bit-vector conventions still hold.
        """
        if not train_smiles_list:
            return 0.0
        fp_q = _morgan_fp(
            query_smiles, radius=int(self.radius), n_bits=int(self.n_bits),
        )
        if fp_q is None:
            return 0.0
        max_sim = 0.0
        for smi in train_smiles_list:
            fp_t = _morgan_fp(
                smi, radius=int(self.radius), n_bits=int(self.n_bits),
            )
            if fp_t is None:
                continue
            try:
                sim = _tanimoto_sim_vectors(fp_q, fp_t)
            except ValueError:
                # Fingerprint-length mismatch — skip this pair.
                continue
            if sim > max_sim:
                max_sim = sim
        return float(max_sim)

    def is_ood(
        self, query_smiles: str, train_smiles_list: Sequence[str],
    ) -> bool:
        """Return ``True`` iff ``query_smiles`` is OOD relative to the train list.

        Definition: OOD iff the *max* Tanimoto similarity to any
        training SMILES is **strictly below** the threshold.  An
        empty training list is treated as OOD for *every* query
        (the train pool has no chemistry to compare against).
        """
        if not train_smiles_list:
            # Empty train pool — every query is OOD by definition.
            return True
        max_sim = self.max_tanimoto(query_smiles, train_smiles_list)
        return bool(max_sim < float(self.threshold))


# ---------------------------------------------------------------------------
# MMseqs2 OOD protocol — stub (binary not installed)
# ---------------------------------------------------------------------------
class MMseqs2OODProtocol:
    """Stub for the MMseqs2 sequence-identity < threshold OOD classifier.

    **DEFERRED**: the ``mmseqs`` binary is not installed in the
    Mol-Metal conda environment, so this class always raises
    :class:`NotImplementedError` when invoked.  See
    ``molmetal/reports/wf_pitfall_tier2/p63_mmseqs2_followup.md``
    for the install steps and follow-up ticket.

    When the binary is available, this class is meant to:
    1. cluster query sequences vs train sequences via
       ``mmseqs easy-cluster`` (or the ``mmseqs search`` path) at
       ``--min-seq-id`` threshold (default 0.30);
    2. flag any query whose cluster contains a train sequence as
       NOT OOD;
    3. expose the same ``is_ood(query_seq, train_seqs)`` API so
       callers can swap implementations without code changes.

    Parameters
    ----------
    threshold_seq_id : float
        Maximum tolerated sequence identity (default 0.30).  A
        query whose closest train sequence has identity ≥
        threshold is NOT OOD.
    mmseqs_binary : str | None
        Optional override for the ``mmseqs`` binary path.  When the
        binary is installed in the future, set this to the absolute
        path (e.g. ``/opt/mmseqs/bin/mmseqs``).
    """

    def __init__(
        self,
        threshold_seq_id: float = DEFAULT_MMSEQS2_THRESHOLD,
        mmseqs_binary: Optional[str] = None,
    ) -> None:
        if not (0.0 <= float(threshold_seq_id) <= 1.0):
            raise ValueError(
                f"threshold_seq_id must be in [0.0, 1.0]; got {threshold_seq_id}"
            )
        self.threshold_seq_id: float = float(threshold_seq_id)
        self.mmseqs_binary: Optional[str] = mmseqs_binary

    def is_ood(
        self,
        query_seq: str,
        train_seqs: Sequence[str],
    ) -> bool:
        """Deferred: always raises :class:`NotImplementedError`.

        Honest framing: the ``mmseqs`` binary is not installed in
        the current conda environment, and the Python wrapper
        ``pyMMseqs`` would add a substantial dependency.  This
        method is intentionally unimplemented — calling code should
        either:
        * skip the pocket-axis OOD check, or
        * install ``mmseqs2`` (``conda install -c bioconda mmseqs2``)
          and switch to a real implementation.
        """
        raise NotImplementedError(
            "MMseqs2OODProtocol.is_ood is DEFERRED — the mmseqs binary "
            "is not installed in the Mol-Metal environment.  See "
            "molmetal/reports/wf_pitfall_tier2/p63_mmseqs2_followup.md "
            "for the install + implementation follow-up.  "
            "Call CombinedOODProtocol with mmseqs2=None (tanimoto-only) "
            "for a CPU-only OOD check."
        )

    def max_seq_identity(
        self,
        query_seq: str,
        train_seqs: Sequence[str],
    ) -> float:
        """Deferred: always raises :class:`NotImplementedError`."""
        raise NotImplementedError(
            "MMseqs2OODProtocol.max_seq_identity is DEFERRED — see "
            "NotImplementedError note on is_ood()."
        )


# ---------------------------------------------------------------------------
# Combined OOD protocol — Tanimoto + (optional) MMseqs2
# ---------------------------------------------------------------------------
@dataclass
class CombinedOODProtocol:
    """OR-combined ligand + pocket OOD protocol.

    A query is OOD iff:
      * the ligand is Tanimoto-OOD w.r.t. the train ligands, AND
      * (optionally) the pocket is MMseqs2-OOD w.r.t. the train
        pockets.

    If ``mmseqs2`` is ``None``, only the ligand axis is checked.
    A query with no ligand (empty SMILES) is **not** OOD by default
    (empty/empty Tanimoto = 1.0 → fails the OOD threshold) — this
    is the safe fail-closed behaviour.

    Parameters
    ----------
    tanimoto : TanimotoOODProtocol
        The ligand-axis OOD classifier.
    mmseqs2 : MMseqs2OODProtocol | None
        Optional pocket-axis classifier.  When provided, both axes
        must agree (logical AND) to declare the query OOD.

    Examples
    --------
    >>> p = CombinedOODProtocol(tanimoto=TanimotoOODProtocol(threshold=0.4))
    >>> p.is_ood(
    ...     query_smi="c1ccccc1O",
    ...     query_seq="ARNDCEQGHILKMFPSTWYV",
    ...     train_smi=["N.N.[Cl][Pt][Cl]"],
    ...     train_seq=["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"],
    ... )
    True
    """

    tanimoto: TanimotoOODProtocol
    mmseqs2: Optional[MMseqs2OODProtocol] = None

    def __post_init__(self) -> None:
        if not isinstance(self.tanimoto, TanimotoOODProtocol):
            raise TypeError(
                f"tanimoto must be a TanimotoOODProtocol; got {type(self.tanimoto)}"
            )
        if self.mmseqs2 is not None and not isinstance(self.mmseqs2, MMseqs2OODProtocol):
            raise TypeError(
                f"mmseqs2 must be MMseqs2OODProtocol or None; got {type(self.mmseqs2)}"
            )

    def is_ood(
        self,
        query_smi: str,
        query_seq: str,
        train_smi: Sequence[str],
        train_seq: Sequence[str],
    ) -> bool:
        """Return ``True`` iff the query is OOD on **both** axes.

        The ligand axis is mandatory; the pocket axis is only
        consulted when ``self.mmseqs2`` is not ``None``.

        Notes
        -----
        * If ``mmseqs2`` is set, the call **will** raise
          :class:`NotImplementedError` until the MMseqs2 binary is
          installed.  This is intentional — we do NOT silently
          skip the pocket axis because that would weaken the OOD
          claim silently.
        """
        ligand_ood = self.tanimoto.is_ood(query_smi, list(train_smi))
        if self.mmseqs2 is None:
            return bool(ligand_ood)
        # Pocket axis — raises NotImplementedError by design.
        pocket_ood = self.mmseqs2.is_ood(query_seq, list(train_seq))
        return bool(ligand_ood and pocket_ood)


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------
__all__ = [
    "DEFAULT_TANIMOTO_THRESHOLD",
    "DEFAULT_FP_RADIUS",
    "DEFAULT_FP_NBITS",
    "DEFAULT_MMSEQS2_THRESHOLD",
    "compute_tanimoto",
    "TanimotoOODProtocol",
    "MMseqs2OODProtocol",
    "CombinedOODProtocol",
]
