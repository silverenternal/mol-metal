"""Pocket2Mol scoring wrapper — clone-reuse from Pocket2Mol repo.

Source
------
``molmetal/references/Pocket2Mol/evaluation/scoring_func.py`` exposes:

- :func:`obey_lipinski` — Lipinski rule-of-five count (0-5).
- :func:`get_basic` — n_atoms, n_bonds, n_rings, mol weight.
- :func:`get_rdkit_rmsd` — RDKit conformer RMSD profile (max, min, median).
- :func:`get_logp` — Crippen MolLogP.
- :func:`get_chem` — tuple ``(qed, sa, logp, hacc, hdon)``.
- :class:`SimilarityWithTrain` — training-set Tanimoto similarity.

This wrapper lifts the pure-RDKit helpers (no QVina binary required).
The ``SimilarityWithTrain`` class is also lifted as a stub whose
``.get_top_sims(...)`` call returns ``None`` when no train fingerprint
cache is on disk (the original requires Pocket2Mol-specific paths).

Heavy imports are guarded; missing deps return ``None`` for that metric.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski
    from rdkit.Chem.QED import qed
    _RDKIT_OK = True
except ImportError:  # pragma: no cover
    _RDKIT_OK = False

try:
    from molmetal.eval.scoring.sascorer import compute_sa_score
    _SA_OK = True
except Exception:  # pragma: no cover
    _SA_OK = False


AVAILABLE = _RDKIT_OK


# ---------------------------------------------------------------------------
# Lifted metric functions (verbatim from Pocket2Mol scoring_func.py)
# ---------------------------------------------------------------------------
def obey_lipinski(mol):
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    rule_4 = (logp := Crippen.MolLogP(mol) >= -2) & (logp <= 5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return int(np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]]))


def get_basic(mol):
    n_atoms = len(mol.GetAtoms())
    n_bonds = len(mol.GetBonds())
    n_rings = len(Chem.GetSymmSSSR(mol))
    weight = Descriptors.ExactMolWt(mol)
    return n_atoms, n_bonds, n_rings, weight


def get_rdkit_rmsd(mol, n_conf: int = 20, random_seed: int = 42) -> List[float]:
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    mol3d = Chem.AddHs(mol)
    try:
        confIds = AllChem.EmbedMultipleConfs(mol3d, n_conf, randomSeed=random_seed)
    except Exception:
        return [float("nan"), float("nan"), float("nan")]
    rmsd_list: List[float] = []
    for confId in confIds:
        try:
            AllChem.UFFOptimizeMolecule(mol3d, confId=confId)
            rmsd_list.append(float(Chem.rdMolAlign.GetBestRMS(mol, mol3d, refId=confId)))
        except Exception:
            rmsd_list.append(float("nan"))
    if not rmsd_list:
        return [float("nan"), float("nan"), float("nan")]
    arr = np.array(rmsd_list)
    return [float(np.max(arr)), float(np.min(arr)), float(np.median(arr))]


def get_logp(mol):
    return float(Crippen.MolLogP(mol))


def get_chem(mol):
    qed_score = float(qed(mol))
    sa_score = float(compute_sa_score(mol)) if _SA_OK else None
    logp_score = float(Crippen.MolLogP(mol))
    hacc_score = int(Lipinski.NumHAcceptors(mol))
    hdon_score = int(Lipinski.NumHDonors(mol))
    return qed_score, sa_score, logp_score, hacc_score, hdon_score


class SimilarityWithTrain:
    """Stub of Pocket2Mol's training-set Tanimoto similarity.

    The original implementation reads from a Pocket2Mol-specific data
    directory (``./data/crossdocked_pocket10``). Without that cache on
    disk, ``get_top_sims(...)`` returns ``(None, None)`` so callers can
    detect the missing-train-cache case without raising.
    """

    def __init__(self) -> None:
        self.train_smiles = None
        self.train_fingers = None
        self.train_uni_smiles = None
        self.train_uni_fingers = None

    def _get_train_mols(self) -> bool:
        # Lazily disabled — no Pocket2Mol data directory in this env.
        return False

    def get_similarity(self, mol: Chem.Mol) -> Optional[np.ndarray]:
        if self.train_fingers is None and not self._get_train_mols():
            return None
        fp = Chem.RDKFingerprint(mol)
        sims = [
            DataStructs.TanimotoSimilarity(fp, ref) for ref in self.train_uni_fingers
        ]
        return np.asarray(sims, dtype=float)

    def get_top_sims(self, mol: Chem.Mol, top: int = 3):
        sims = self.get_similarity(mol)
        if sims is None:
            return None, None
        idx = np.argsort(sims)[::-1][:top]
        return sims[idx], self.train_uni_smiles[idx]


# ---------------------------------------------------------------------------
# Public wrapper API
# ---------------------------------------------------------------------------
def _mol_from_smiles(smi: str) -> Optional[Chem.Mol]:
    if not smi:
        return None
    return Chem.MolFromSmiles(str(smi))


def score(smiles_list: List[str], pocket: Optional[Any] = None, **_: Any) -> Optional[Dict[str, Any]]:
    """Score a list of SMILES with the Pocket2Mol metric suite.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES strings. ``None`` / empty / unparseable entries produce
        ``None`` rows in the output.
    pocket : optional, unused
        QVina docking path is intentionally not lifted (no binary).

    Returns
    -------
    dict with keys:
        - ``per_mol``: list[dict | None] — per-molecule metric dict
          (qed, sa, logp, hacc, hdon, lipinski, rdkit_rmsd).
        - ``summary``: dict of mean values over successful mols.
        - ``vina_scores``: always ``None`` (QVina docking not lifted).
    """
    if not _RDKIT_OK:
        return None

    per_mol: List[Optional[Dict[str, Any]]] = []
    n_ok = 0
    agg_qed: List[float] = []
    agg_sa: List[float] = []
    agg_logp: List[float] = []
    agg_hacc: List[int] = []
    agg_hdon: List[int] = []
    agg_lip: List[int] = []

    for smi in smiles_list:
        mol = _mol_from_smiles(smi)
        if mol is None:
            per_mol.append(None)
            continue
        try:
            q, sa, lp, hacc, hdon = get_chem(mol)
            lip = obey_lipinski(mol)
            rmsd = get_rdkit_rmsd(mol, n_conf=5)  # smaller for spot-check speed
            row = {
                "qed": q,
                "sa": sa,
                "logp": lp,
                "hacc": hacc,
                "hdon": hdon,
                "lipinski": lip,
                "rdkit_rmsd": rmsd,
                "vina": None,  # QVina path intentionally not lifted
            }
            per_mol.append(row)
            n_ok += 1
            agg_qed.append(q)
            if sa is not None:
                agg_sa.append(sa)
            agg_logp.append(lp)
            agg_hacc.append(hacc)
            agg_hdon.append(hdon)
            agg_lip.append(lip)
        except Exception:
            per_mol.append(None)

    summary: Dict[str, Any] = {
        "n_input": len(smiles_list),
        "n_ok": n_ok,
        "qed_mean": float(np.mean(agg_qed)) if agg_qed else None,
        "sa_mean": float(np.mean(agg_sa)) if agg_sa else None,
        "logp_mean": float(np.mean(agg_logp)) if agg_logp else None,
        "hacc_mean": float(np.mean(agg_hacc)) if agg_hacc else None,
        "hdon_mean": float(np.mean(agg_hdon)) if agg_hdon else None,
        "lipinski_mean": float(np.mean(agg_lip)) if agg_lip else None,
    }
    return {"per_mol": per_mol, "summary": summary, "vina_scores": None}


__all__ = [
    "score",
    "get_chem",
    "get_logp",
    "obey_lipinski",
    "get_basic",
    "get_rdkit_rmsd",
    "SimilarityWithTrain",
]
