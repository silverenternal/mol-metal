"""TargetDiff scoring wrapper — clone-reuse from targetdiff repo.

Source
------
``molmetal/references/targetdiff/utils/evaluation/scoring_func.py`` exposes:

- :func:`is_pains` — PAINS-A filter.
- :func:`obey_lipinski` — Lipinski rule-of-five count (0-5).
- :func:`get_basic` — n_atoms, n_bonds, n_rings, mol weight.
- :func:`get_logp` — Crippen MolLogP.
- :func:`get_chem` — returns dict ``{qed, sa, logp, lipinski, ring_size}``.

This wrapper copies those pure-RDKit metric functions into the molmetal
namespace (no model inference; no docking call) and exposes them via a
``score(smiles_list, pocket_pdb=None) -> Dict[str, Any]`` API.

The Vina-docking paths are NOT lifted because they require a QVina/AutoDock
Vina binary which is not installed on this machine (per TODO/environment
constraints). Heavy imports are guarded; missing deps return ``None`` for
that metric only, not for the whole wrapper.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski
    from rdkit.Chem.FilterCatalog import (
        FilterCatalog,
        FilterCatalogParams,
    )
    from rdkit.Chem.QED import qed
    _RDKIT_OK = True
except ImportError:  # pragma: no cover - guard
    _RDKIT_OK = False

# Heavy deps are all opt-in. SA score uses the vendored Pocket2Mol-derived
# sascorer next to this module (works without TDC / QVina / BioLM).
try:
    from molmetal.eval.scoring.sascorer import compute_sa_score
    _SA_OK = True
except Exception:  # pragma: no cover - guard
    _SA_OK = False


AVAILABLE = _RDKIT_OK  # SA is recommended but optional


# ---------------------------------------------------------------------------
# Lifted metric functions (verbatim from targetdiff scoring_func.py)
# ---------------------------------------------------------------------------
def is_pains(mol):
    params_pain = FilterCatalogParams()
    params_pain.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    catalog_pain = FilterCatalog(params_pain)
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    return catalog_pain.GetFirstMatch(mol) is not None


def obey_lipinski(mol):
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    logp = get_logp(mol)
    rule_4 = (logp >= -2) & (logp <= 5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return int(np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]]))


def get_basic(mol):
    n_atoms = len(mol.GetAtoms())
    n_bonds = len(mol.GetBonds())
    n_rings = len(Chem.GetSymmSSSR(mol))
    weight = Descriptors.ExactMolWt(mol)
    return n_atoms, n_bonds, n_rings, weight


def get_logp(mol):
    return Crippen.MolLogP(mol)


def get_chem(mol):
    qed_score = qed(mol)
    sa_score = compute_sa_score(mol) if _SA_OK else None
    logp_score = get_logp(mol)
    lipinski_score = obey_lipinski(mol)
    ring_info = mol.GetRingInfo()
    ring_size = Counter([len(r) for r in ring_info.AtomRings()])
    return {
        "qed": qed_score,
        "sa": sa_score,
        "logp": logp_score,
        "lipinski": lipinski_score,
        "ring_size": dict(ring_size),
    }


# ---------------------------------------------------------------------------
# Public wrapper API
# ---------------------------------------------------------------------------
def _mol_from_smiles(smi: str) -> Optional[Chem.Mol]:
    if not smi:
        return None
    return Chem.MolFromSmiles(str(smi))


def score(smiles_list: List[str], pocket: Optional[Any] = None, **_: Any) -> Optional[Dict[str, Any]]:
    """Score a list of SMILES with the TargetDiff metric suite.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES strings. ``None`` / empty / unparseable entries produce
        ``None`` rows in the output.
    pocket : optional, unused
        The targetdiff Vina-docking path is intentionally not lifted here
        (requires a QVina binary that is not installed). The docking
        score is therefore always returned as ``None``.

    Returns
    -------
    dict with keys:
        - ``per_mol``: list[dict | None] — per-molecule metric dict
          (qed, sa, logp, lipinski, ring_size, pains, vina).
        - ``summary``: dict of mean values over successful mols.
        - ``vina_scores``: always ``None`` (docking path not lifted).
    """
    if not _RDKIT_OK:
        return None

    per_mol: List[Optional[Dict[str, Any]]] = []
    n_ok = 0
    agg_qed: List[float] = []
    agg_sa: List[float] = []
    agg_logp: List[float] = []
    agg_lip: List[int] = []
    agg_pains: List[bool] = []

    for smi in smiles_list:
        mol = _mol_from_smiles(smi)
        if mol is None:
            per_mol.append(None)
            continue
        try:
            chem = get_chem(mol)
            pains = bool(is_pains(mol))
            row = {
                "qed": chem["qed"],
                "sa": chem["sa"],
                "logp": chem["logp"],
                "lipinski": chem["lipinski"],
                "ring_size": chem["ring_size"],
                "pains": pains,
                "vina": None,  # docking path intentionally not lifted
            }
            per_mol.append(row)
            n_ok += 1
            agg_qed.append(chem["qed"])
            if chem["sa"] is not None:
                agg_sa.append(chem["sa"])
            agg_logp.append(chem["logp"])
            agg_lip.append(chem["lipinski"])
            agg_pains.append(pains)
        except Exception:
            per_mol.append(None)

    summary: Dict[str, Any] = {
        "n_input": len(smiles_list),
        "n_ok": n_ok,
        "qed_mean": float(np.mean(agg_qed)) if agg_qed else None,
        "sa_mean": float(np.mean(agg_sa)) if agg_sa else None,
        "logp_mean": float(np.mean(agg_logp)) if agg_logp else None,
        "lipinski_mean": float(np.mean(agg_lip)) if agg_lip else None,
        "pains_rate": float(np.mean(agg_pains)) if agg_pains else None,
    }
    return {"per_mol": per_mol, "summary": summary, "vina_scores": None}


__all__ = ["score", "get_chem", "get_logp", "obey_lipinski", "is_pains", "get_basic"]
