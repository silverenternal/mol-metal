"""Ertl-Schuffenhauer synthetic accessibility (SA) score.

This module wraps RDKit's canonical ``Contrib/SA_Score/sascorer.py``
implementation (Ertl & Schuffenhauer, *Mol. Inf.* 2009) so the rest of
the Lambda pipeline can score any SMILES with the same metric used by:

* Ertl et al., *J. Chem. Inf. Model.* 2008 (original publication)
* Pocket2Mol (Peng et al., ICML 2022) -- SA in [0.6, 1.0] good
* TargetDiff (Guan et al., ICLR 2023) -- uses SA-score in the same range
* Most SBDD / molecule-generation papers since 2020

The raw ``calculateScore`` function returns a value in **[1, 10]** where
1 = trivially synthesizable (e.g. glycine) and 10 = practically
un-synthesizable (e.g. taxol, polycyclic natural products). For metrics
that want "higher = better" (consistent with our Vina / pIC50 columns),
we expose :func:`sa_score_to_unit` which linearly maps [1, 10] ->
[1.0, 0.0].

Source
------
The canonical sascorer ships with RDKit as a Contrib script::

    from rdkit.Chem import RDConfig
    import os
    p = os.path.join(RDConfig.RDContribDir, "SA_Score", "sascorer.py")

This module dynamically prepends that directory to ``sys.path`` so the
shipped reference implementation is used unmodified -- no copy, no
fork, no risk of drift from upstream Ertl algorithm.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Optional

# ---------------------------------------------------------------------------
# Locate sascorer.py inside the RDKit Contrib tree and import it.
# ---------------------------------------------------------------------------
try:
    from rdkit.Chem import RDConfig  # type: ignore

    _SA_DIR = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if os.path.isdir(_SA_DIR) and _SA_DIR not in sys.path:
        sys.path.insert(0, _SA_DIR)
    import sascorer  # type: ignore  # noqa: E402  (the canonical Ertl impl)

    _SASCORER_AVAILABLE = True
    _SASCORER_ERROR: Optional[str] = None
except Exception as _e:  # pragma: no cover - environment guard
    sascorer = None  # type: ignore
    _SASCORER_AVAILABLE = False
    _SASCORER_ERROR = repr(_e)


# ---------------------------------------------------------------------------
# Lazy / cached mol -> SA lookup.  Caching is safe because sascorer is
# purely functional -- the score depends only on the molecule graph.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4096)
def _sa_cached(mol) -> float:
    """Uncached helper wrapped by :func:`sa_score_ertl`."""
    return float(sascorer.calculateScore(mol))


def sa_score_ertl(smiles: str) -> float:
    """Compute the Ertl-Schuffenhauer SA score for a SMILES string.

    Parameters
    ----------
    smiles : str
        A canonical or non-canonical SMILES. Whitespace-only / empty
        strings and unparseable molecules yield ``NaN``.

    Returns
    -------
    float
        SA in **[1, 10]** (lower = more synthetically accessible).
        Returns ``float('nan')`` on parse failure or when sascorer
        cannot be imported.
    """
    if not _SASCORER_AVAILABLE:
        return float("nan")
    s = (smiles or "").strip()
    if not s:
        return float("nan")
    try:
        from rdkit import Chem  # type: ignore
        from rdkit import RDLogger  # type: ignore

        RDLogger.DisableLog("rdApp.*")  # silence parser warnings
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return float("nan")
        return _sa_cached(mol)
    except Exception:
        return float("nan")


def sa_score_to_unit(sa: float) -> float:
    """Linearly map raw SA ``[1, 10]`` -> unit-score ``[1.0, 0.0]``.

    Higher = better (consistent with pIC50, Vina). A perfect
    SA of 1.0 (trivially synthesizable) maps to 1.0; the hardest
    practical SA of 10.0 maps to 0.0.

    Values outside [1, 10] are clamped before mapping so a malformed
    input never produces a negative or >1 score.
    """
    try:
        x = float(sa)
    except Exception:
        return 0.0
    if x != x:  # NaN
        return 0.0
    if x < 1.0:
        x = 1.0
    elif x > 10.0:
        x = 10.0
    return (10.0 - x) / 9.0


def batch_sa_score(smiles_list) -> dict:
    """Score a list of SMILES. Returns mean/min/max + n_valid."""
    vals = [sa_score_ertl(s) for s in smiles_list]
    valid = [v for v in vals if v == v]  # filter NaN
    if not valid:
        return {"mean": float("nan"), "min": float("nan"),
                "max": float("nan"), "n_valid": 0, "n_total": len(vals)}
    return {
        "mean": sum(valid) / len(valid),
        "min": min(valid),
        "max": max(valid),
        "n_valid": len(valid),
        "n_total": len(vals),
    }


__all__ = [
    "sa_score_ertl",
    "sa_score_to_unit",
    "batch_sa_score",
    "_SASCORER_AVAILABLE",
    "_SASCORER_ERROR",
]


if __name__ == "__main__":  # quick smoke
    for smi, label in [
        ("CC(=O)Oc1ccccc1C(=O)O", "aspirin"),
        ("OCC1OC(O)C(O)C(O)C1O", "glucose"),
        ("NCCO", "ethanolamine"),
        ("C1=CC=CC=C1", "benzene"),
    ]:
        sa = sa_score_ertl(smi)
        print(f"{label:<14}  SA={sa:.3f}   unit={sa_score_to_unit(sa):.3f}")
