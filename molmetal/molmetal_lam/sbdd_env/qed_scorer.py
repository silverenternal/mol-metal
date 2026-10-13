"""Batched RDKit quantitative estimate of drug-likeness (QED) scoring.

The :class:`QEDScorer` class provides a small, dependency-tolerant API for
scoring individual molecules or lists of SMILES strings.  RDKit is imported
at module load time when available; installations without RDKit can still
import this module, in which case scoring methods return ``None`` (or a list
of ``None`` values).  A SMILES that RDKit cannot parse is likewise ``None``,
never ``0.0`` — see :func:`qed_from_smiles` for why that distinction is load-bearing.

``score_with_sa`` combines QED with the Ertl synthetic accessibility score
from :mod:`sa_score` for callers that need both common SBDD objectives.
"""

from __future__ import annotations

from typing import Any, Optional, Union

try:  # Keep importing the package possible in lightweight environments.
    from rdkit import Chem, RDLogger  # type: ignore
    from rdkit.Chem import QED  # type: ignore

    RDLogger.DisableLog("rdApp.*")
    _RDKIT_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the runtime environment
    Chem = None  # type: ignore
    QED = None  # type: ignore
    _RDKIT_AVAILABLE = False


#: Number of inputs that RDKit could not turn into a molecule.  A caller that
#: averages qed_from_smiles() output MUST check this, because ``None`` values
#: are dropped from means and would otherwise be invisible in the total.
_QED_FALLBACK_COUNT = 0


def qed_fallback_count() -> int:
    """How many inputs have been unparseable since import.

    Exists because the failure this guards against is invisible in the output:
    a batch where 5 of 818 molecules are unparseable returns a mean that looks
    exactly like a clean batch, just very slightly lower.
    """
    return _QED_FALLBACK_COUNT


def qed_reset_fallback_count() -> None:
    """Zero the counter.  For callers that report per-cohort fallback rates."""
    global _QED_FALLBACK_COUNT
    _QED_FALLBACK_COUNT = 0


def qed_from_smiles(smi: str) -> float | None:
    """Return RDKit QED for *smi*, or ``None`` when it cannot be computed.

    ``None`` covers three distinct cases: RDKit unavailable, input not a
    non-empty string, and RDKit refusing to parse the molecule.  Each counts
    toward :func:`qed_fallback_count`.

    This used to return ``0.0`` for invalid input.  That silently conflated
    "RDKit could not parse this" with "a real molecule whose QED is exactly
    zero", and because the callers here average the returned list, every
    unparseable molecule quietly dragged a cohort mean toward zero.  The
    arithmetic callers (e.g. ``r4_lib/metrics.py``) already skip ``None`` and
    ``NaN``, so returning ``None`` excludes unparseable rows from the mean
    instead of corrupting it.  Do not change this back to ``0.0``.
    """
    global _QED_FALLBACK_COUNT
    if not _RDKIT_AVAILABLE:
        _QED_FALLBACK_COUNT += 1
        return None
    if not isinstance(smi, str) or not smi.strip():
        _QED_FALLBACK_COUNT += 1
        return None
    try:
        mol = Chem.MolFromSmiles(smi.strip())
        if mol is None:
            _QED_FALLBACK_COUNT += 1
            return None
        return float(QED.qed(mol))
    except Exception:
        _QED_FALLBACK_COUNT += 1
        return None


class QEDScorer:
    """RDKit-backed scorer supporting scalar and batched molecule inputs.

    Parameters
    ----------
    show_progress:
        If true, ``score_batch`` displays a tqdm progress bar when tqdm is
        installed.  Progress is disabled by default for library use.
    """

    def __init__(self, show_progress: bool = False) -> None:
        self.show_progress = bool(show_progress)
        self.available = _RDKIT_AVAILABLE

    def score(self, smiles_or_mol: Union[str, Any]) -> float:
        """Score one SMILES string or an RDKit ``Mol`` instance.

        Returns ``None`` for anything RDKit cannot score — an unparseable
        string, a ``None`` molecule, an exception, or RDKit being unavailable.
        It used to return ``0.0`` for the first three, which was
        indistinguishable from a genuine QED of zero inside any mean; see
        :func:`qed_from_smiles`.  Each ``None`` counts toward
        :func:`qed_fallback_count`.
        """
        global _QED_FALLBACK_COUNT
        if not _RDKIT_AVAILABLE:
            _QED_FALLBACK_COUNT += 1
            return None  # type: ignore[return-value]
        # Chem.MolFromSmiles("") returns an EMPTY Mol, not None, so an empty
        # string would otherwise sail past the `mol is None` branch below and be
        # scored as a real (0-atom) molecule. qed_from_smiles already guards
        # this; score() must too, or the two entry points disagree.
        if isinstance(smiles_or_mol, str) and not smiles_or_mol.strip():
            _QED_FALLBACK_COUNT += 1
            return None  # type: ignore[return-value]
        try:
            if isinstance(smiles_or_mol, str):
                mol = Chem.MolFromSmiles(smiles_or_mol.strip())
            else:
                mol = smiles_or_mol
            if mol is None:
                _QED_FALLBACK_COUNT += 1
                return None  # type: ignore[return-value]
            return float(QED.qed(mol))
        except Exception:
            _QED_FALLBACK_COUNT += 1
            return None  # type: ignore[return-value]

    def score_batch(self, smiles_list: list[str]) -> list[float]:
        """Return one QED score per SMILES, preserving input order."""
        if not smiles_list:
            return []
        values = smiles_list
        if self.show_progress:
            try:
                from tqdm.auto import tqdm  # type: ignore

                values = tqdm(smiles_list, desc="QED", unit="mol")
            except Exception:
                pass
        return [self.score(smi) for smi in values]  # type: ignore[list-item]

    def score_with_sa(self, smiles_list: list[str]) -> list[dict]:
        """Return ``{smiles, qed, sa}`` records for each input SMILES."""
        try:
            from .sa_score import sa_score_ertl
        except Exception:  # pragma: no cover - package/import edge case
            sa_score_ertl = None
        records = []
        for smi, qed in zip(smiles_list, self.score_batch(smiles_list)):
            sa: Optional[float]
            if sa_score_ertl is None:
                sa = None
            else:
                try:
                    raw = float(sa_score_ertl(smi))
                    sa = raw if raw == raw else None
                except Exception:
                    sa = None
            records.append({"smiles": smi, "qed": qed, "sa": sa})
        return records


__all__ = [
    "QEDScorer",
    "qed_from_smiles",
    "qed_fallback_count",
    "qed_reset_fallback_count",
    "_RDKIT_AVAILABLE",
]
