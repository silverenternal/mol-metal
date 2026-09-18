"""Batched RDKit quantitative estimate of drug-likeness (QED) scoring.

The :class:`QEDScorer` class provides a small, dependency-tolerant API for
scoring individual molecules or lists of SMILES strings.  RDKit is imported
at module load time when available; installations without RDKit can still
import this module, in which case scoring methods return ``None`` (or a list
of ``None`` values).  Invalid SMILES are represented by a score of ``0.0``.

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


def qed_from_smiles(smi: str) -> float | None:
    """Return RDKit QED for *smi*, ``0.0`` for invalid input.

    ``None`` is returned when RDKit is unavailable.  Empty and malformed
    strings are ordinary invalid molecules and therefore return ``0.0``.
    """
    if not _RDKIT_AVAILABLE:
        return None
    if not isinstance(smi, str) or not smi.strip():
        return 0.0
    try:
        mol = Chem.MolFromSmiles(smi.strip())
        if mol is None:
            return 0.0
        return float(QED.qed(mol))
    except Exception:
        return 0.0


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

        Invalid molecules yield ``0.0``; ``None`` is returned if RDKit is
        unavailable (despite the nominal float return annotation).
        """
        if not _RDKIT_AVAILABLE:
            return None  # type: ignore[return-value]
        try:
            if isinstance(smiles_or_mol, str):
                mol = Chem.MolFromSmiles(smiles_or_mol.strip())
            else:
                mol = smiles_or_mol
            if mol is None:
                return 0.0
            return float(QED.qed(mol))
        except Exception:
            return 0.0

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


__all__ = ["QEDScorer", "qed_from_smiles", "_RDKIT_AVAILABLE"]
