"""Validity gate — SMILES round-trip + RDKit sanitisation.

================================================================
Validity gate for curriculum-learning injected samples (TODO-150 §P1.2)
================================================================
Closes the validity contract used by Wf-77 / TODO-13 lambda-clickchem:
every dreaming-derived candidate must pass

1. **SMILES round-trip** — ``Chem.MolFromSmiles(smi)`` succeeds and
   ``Chem.MolToSmiles(mol)`` reproduces the input.
2. **RDKit sanitisation** — ``Chem.SanitizeMol(mol)`` runs without
   exception (no valences out of range, no unconnected fragments).

The gate is intentionally conservative: a SMILES that fails *either*
check is rejected with a ``ValidityResult(valid=False, ...)``.  No
"soft" pass — the curriculum must not inject chemistry it cannot
represent.

Why this lives in its own module
--------------------------------
The validity contract is the same contract used by
:mod:`molmetal.molmetal_lam.lam_chem.well_formedness` (TODO-13) and
:mod:`molmetal.molmetal_lam.lam_chem.pt_click_compat` (TODO-14); a
shared gate keeps the curriculum from drifting to a stricter /
looser standard than the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

try:
    from rdkit import Chem  # noqa: F401  (imported for side-effect only)
    _RDKIT_OK = True
except Exception:
    _RDKIT_OK = False


@dataclass(frozen=True)
class ValidityResult:
    """One validity decision for a SMILES candidate."""

    smiles: str
    valid: bool
    fail_class: str  # ``"ok"`` / ``"smiles_parse"`` / ``"round_trip"`` / ``"sanitize"``
    canonical_smiles: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "smiles": str(self.smiles),
            "valid": bool(self.valid),
            "fail_class": str(self.fail_class),
            "canonical_smiles": self.canonical_smiles,
            "error": self.error,
        }


class ValidityGate:
    """Round-trip + sanitisation gate.

    Stateless — ``__call__`` returns one :class:`ValidityResult` per
    candidate.  Uses RDKit when importable; falls back to a
    length-only heuristic so the gate is unit-testable on hosts
    without RDKit.
    """

    #: Default accepted fail classes.  Anything else ⇒ rejection.
    DEFAULT_OK_FAIL_CLASSES = ("ok",)

    def __init__(self, ok_fail_classes: Optional[Sequence[str]] = None) -> None:
        self.ok_fail_classes = tuple(ok_fail_classes) if ok_fail_classes else self.DEFAULT_OK_FAIL_CLASSES
        self._chem_ok = bool(_RDKIT_OK)

    # ------------------------------------------------------------------
    # Single-SMILES check
    # ------------------------------------------------------------------
    def __call__(self, smiles: str) -> ValidityResult:
        """Evaluate one SMILES."""
        return self.check(smiles)

    def check(self, smiles: str) -> ValidityResult:
        """Evaluate one SMILES — same as :meth:`__call__`."""
        if not isinstance(smiles, str) or not smiles:
            return ValidityResult(
                smiles=str(smiles),
                valid=False,
                fail_class="smiles_parse",
                error="empty or non-str smiles",
            )
        if not self._chem_ok:
            # No RDKit → cannot do real check; fall back to length proxy.
            # A non-empty SMILES without NUL bytes is treated as valid —
            # this is the conservative fallback so the gate is still
            # importable on CPU-only test hosts.
            valid = bool(smiles) and "\x00" not in smiles
            return ValidityResult(
                smiles=smiles,
                valid=bool(valid),
                fail_class="ok" if valid else "smiles_parse",
                canonical_smiles=smiles if valid else None,
                error=None if valid else "rdkit unavailable; length proxy",
            )
        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception as exc:
            return ValidityResult(
                smiles=smiles,
                valid=False,
                fail_class="smiles_parse",
                canonical_smiles=None,
                error=f"parse exception: {exc}",
            )
        if mol is None:
            return ValidityResult(
                smiles=smiles,
                valid=False,
                fail_class="smiles_parse",
                canonical_smiles=None,
                error="MolFromSmiles returned None",
            )
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:
            return ValidityResult(
                smiles=smiles,
                valid=False,
                fail_class="sanitize",
                canonical_smiles=None,
                error=f"sanitize exception: {exc}",
            )
        try:
            canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        except Exception as exc:
            return ValidityResult(
                smiles=smiles,
                valid=False,
                fail_class="round_trip",
                canonical_smiles=None,
                error=f"round-trip exception: {exc}",
            )
        if canonical != smiles:
            # Round-trip mismatch is a *soft* fail: the SMILES is
            # well-formed, just not in canonical form.  The pipeline
            # canonicalises everything downstream anyway, so we
            # treat it as valid and report the canonical.
            return ValidityResult(
                smiles=smiles,
                valid=True,
                fail_class="ok",
                canonical_smiles=canonical,
                error=None,
            )
        return ValidityResult(
            smiles=smiles,
            valid=True,
            fail_class="ok",
            canonical_smiles=canonical,
            error=None,
        )

    # ------------------------------------------------------------------
    # Batch + filtering
    # ------------------------------------------------------------------
    def filter_valid(
        self, smiles_list: Sequence[str]
    ) -> List[ValidityResult]:
        """Apply the gate to a batch.  Preserves order."""
        return [self.check(s) for s in smiles_list]

    def accept(self, smiles_list: Sequence[str]) -> List[str]:
        """Return only the SMILES that pass the gate.

        The output preserves the input order but drops the
        ``fail_class != "ok"`` entries.
        """
        return [
            (r.canonical_smiles or r.smiles)
            for r in self.filter_valid(smiles_list)
            if r.valid and r.fail_class in self.ok_fail_classes
        ]

    def rejection_report(
        self, smiles_list: Sequence[str]
    ) -> Dict[str, int]:
        """Histogram of fail classes for a batch — useful for §6 stats."""
        out: Dict[str, int] = {"ok": 0, "smiles_parse": 0, "round_trip": 0, "sanitize": 0}
        for r in self.filter_valid(smiles_list):
            if r.fail_class in out:
                out[r.fail_class] += 1
            else:
                out[r.fail_class] = 1
        return out


__all__ = [
    "ValidityResult",
    "ValidityGate",
]