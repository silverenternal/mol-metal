"""Canonical SBDD scoring facade (round-7 cherry-pick, clone-reuse target).

The three wrappers under this package each expose a thin ``score(smiles_list, ...)``
API on top of the metric functions copied/imported from cloned SBDD repos:

- :mod:`molmetal.eval.scoring.targetdiff_score`
    QED + SA + logP + Lipinski + ring-size histogram (from targetdiff).
- :mod:`molmetal.eval.scoring.pocket2mol_score`
    QED + SA + logP + hacc/hdon + Lipinski + RDKit RMSD (from Pocket2Mol).
- :mod:`molmetal.eval.scoring.softmol_score`
    Hit-rate gate + top-k% Vina affinity roll-up (from SoftMol). TDC oracles
    wrapped behind a try/except so the wrapper degrades to RDKit when TDC is
    not installed.

Heavy dependencies (QVina, TDC, ESM, BioLM-Score) are wrapped in try/except
blocks; each wrapper sets a module-level ``AVAILABLE`` flag and returns
``None`` for any metric that depends on a missing dep.

Public API
----------

- :func:`score_with` dispatches a list of SMILES through one of the three
  wrappers by name. ``method`` must be one of ``"targetdiff"``,
  ``"pocket2mol"``, ``"softmol"``.
- :func:`available_methods` returns the list of methods whose metric deps are
  installed on the current Python env.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from molmetal.eval.scoring import pocket2mol_score, softmol_score, targetdiff_score


__all__ = ["score_with", "available_methods", "list_methods"]


_REGISTERED = {
    "targetdiff": targetdiff_score,
    "pocket2mol": pocket2mol_score,
    "softmol": softmol_score,
}


def list_methods() -> List[str]:
    """Return the list of registered scoring method names."""
    return list(_REGISTERED.keys())


def available_methods() -> List[str]:
    """Return only the methods whose metric deps are importable on this env.

    A method is "available" if its wrapper module sets ``AVAILABLE = True``
    (i.e. the RDKit + numpy backbone is present; heavy extras like TDC or
    QVina are NOT required for the wrapper to count as available — they only
    gate individual metrics within the wrapper).
    """
    return [name for name, mod in _REGISTERED.items() if getattr(mod, "AVAILABLE", False)]


def score_with(
    method: str,
    smiles: List[str],
    pocket: Optional[Any] = None,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """Dispatch a list of SMILES through one of the registered scoring wrappers.

    Parameters
    ----------
    method : str
        One of ``"targetdiff"``, ``"pocket2mol"``, ``"softmol"``.
    smiles : list[str]
        SMILES strings to score. ``None`` / empty strings are tolerated and
        produce ``None`` entries in the per-mol output.
    pocket : optional
        Pocket object (Pocket2Mol targetdiff docking path only); the scoring
        wrappers do not require it.
    **kwargs
        Forwarded to the wrapper's ``score(...)`` call.

    Returns
    -------
    dict or None
        Per-method metric dictionary (see wrapper docstrings for schema).
        Returns ``None`` if the method is unknown or the wrapper is missing
        a required dep.
    """
    if method not in _REGISTERED:
        return None
    return _REGISTERED[method].score(smiles, pocket=pocket, **kwargs)
