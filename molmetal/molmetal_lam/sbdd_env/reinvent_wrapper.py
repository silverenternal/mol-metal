"""REINVENT4-style multi-property scorer.

This is a *fallback* scorer — we wrap the REINVENT4 repo (if cloned) so we
can discover/configure its components, but the actual scoring falls back to
RDKit-based proxies so the code runs without the heavy REINVENT4 plugins
(QSAR models, docking binaries, GPU RL loop).

Defaults follow the REINVENT4 multi-property optimization recipe:
``{"qed": 0.6, "binding": 1.0, "sas": 0.4, "novelty": 0.3}``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

try:
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors  # type: ignore
    _HAVE_RDKIT = True
except Exception:  # pragma: no cover
    _HAVE_RDKIT = False


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_SCORER_WEIGHTS: Dict[str, float] = {
    "qed": 0.6,
    "binding": 1.0,
    "sas": 0.4,
    "novelty": 0.3,
}


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------
@dataclass
class REINVENT4Scorer:
    """Multi-property scorer with REINVENT4-style weights.

    The scorer is intentionally *fallback-friendly*: if the REINVENT4 repo
    is missing or its plugins can't be imported, all component scores
    degrade to RDKit-based proxies so downstream code keeps running.
    """

    reinvent_repo_path: str = "molmetal/references/REINVENT4"
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SCORER_WEIGHTS))

    # Optional: a reference SMILES set used by ``_compute_novelty``.
    train_smiles_set: Optional[set] = None

    # Lazy fingerprint cache for novelty (only used if set).
    _train_fps: Optional[List] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.weights, dict):
            raise TypeError("weights must be a dict[str, float]")
        # Make sure all expected keys are present.
        for key in DEFAULT_SCORER_WEIGHTS:
            self.weights.setdefault(key, DEFAULT_SCORER_WEIGHTS[key])
        self._verify_repo()

    # ----------------------------------------------------------------- utils
    def _verify_repo(self) -> bool:
        """Log + return whether the REINVENT4 checkout exists.

        We don't actually *use* the repo at runtime — its heavy scorer
        plugins (QSAR, docking, RL) are not installed in this env.
        """
        repo = Path(self.reinvent_repo_path)
        exists = repo.is_dir() and (repo / "reinvent").is_dir()
        if exists:
            logger.info("REINVENT4 repo found at %s — fallback scorer only.", repo)
        else:
            logger.info(
                "REINVENT4 repo NOT found at %s — falling back to RDKit-only scoring.",
                repo,
            )
        return exists

    # ---------------------------------------------------------- per-component
    @staticmethod
    def _compute_qed(smiles: str) -> float:
        """RDKit QED in [0, 1]. Returns 0.0 on parse failure."""
        if not _HAVE_RDKIT:
            return 0.0
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
            return float(QED.qed(mol))
        except Exception:
            return 0.0

    @staticmethod
    def _compute_sas(smiles: str) -> float:
        """Proxy for Ertl-Schuffenhauer SAS in [0, 1].

        Real REINVENT4 uses the original SA score (1..10). We map it to
        ``1 / (1 + NumAromaticRings)`` so simpler, less-aromatic
        molecules score higher (closer to 1).
        """
        if not _HAVE_RDKIT:
            return 0.0
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
            n_ar = rdMolDescriptors.CalcNumAromaticRings(mol)
            return float(1.0 / (1.0 + n_ar))
        except Exception:
            return 0.0

    def _compute_novelty(self, smiles: str, train_smiles_set: Optional[Iterable[str]] = None) -> float:
        """1 - max(Tanimoto similarity) to training set. Higher = more novel.

        Falls back to 0.5 (neutral) when no train set is available.
        """
        ref_set = train_smiles_set if train_smiles_set is not None else self.train_smiles_set
        if ref_set is None or not _HAVE_RDKIT:
            return 0.5
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.5
            fp_query = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        except Exception:
            return 0.5

        if self._train_fps is None:
            self._train_fps = []
            for s in ref_set:
                m = Chem.MolFromSmiles(s)
                if m is not None:
                    self._train_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048))
            if not self._train_fps:
                return 0.5

        from rdkit import DataStructs  # type: ignore
        sims = DataStructs.BulkTanimotoSimilarity(fp_query, self._train_fps)
        max_sim = max(sims) if sims else 0.0
        return float(1.0 - max_sim)

    @staticmethod
    def _compute_binding(smiles: str) -> float:
        """STUB Phase 0 binding score — proxy by rotatable bonds.

        Real REINVENT4 uses docking binaries (Glide, GOLD, Vina) or a
        trained QSAR model. Until we wire one in we expose a transparent
        proxy: ``0.5 + 0.1 * NumRotatableBonds`` clamped to [0, 1].
        """
        if not _HAVE_RDKIT:
            return 0.0
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
            n_rot = Descriptors.NumRotatableBonds(mol)
            return float(min(1.0, max(0.0, 0.5 + 0.1 * n_rot)))
        except Exception:
            return 0.0

    # ----------------------------------------------------------------- core
    def score(self, smiles: str) -> float:
        """Weighted multi-property score for a single SMILES."""
        if not smiles or not isinstance(smiles, str):
            return 0.0
        components = {
            "qed": self._compute_qed(smiles),
            "binding": self._compute_binding(smiles),
            "sas": self._compute_sas(smiles),
            "novelty": self._compute_novelty(smiles),
        }
        return float(sum(self.weights[k] * components[k] for k in self.weights))

    def component_breakdown(self, smiles: str) -> Dict[str, float]:
        """Return the raw (unweighted) component scores — useful for debugging."""
        return {
            "qed": self._compute_qed(smiles),
            "binding": self._compute_binding(smiles),
            "sas": self._compute_sas(smiles),
            "novelty": self._compute_novelty(smiles),
        }

    def batch_score(self, smiles_list: Sequence[str]) -> np.ndarray:
        """Vectorised batch scoring — same numbers as repeated :meth:`score`."""
        return np.asarray([self.score(s) for s in smiles_list], dtype=np.float32)


# ---------------------------------------------------------------------------
# Functional helpers (convenience wrappers)
# ---------------------------------------------------------------------------
def batch_score(smiles_list: Sequence[str], weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    """One-shot batch scoring with a fresh :class:`REINVENT4Scorer`."""
    scorer = REINVENT4Scorer(weights=weights or DEFAULT_SCORER_WEIGHTS)
    return scorer.batch_score(smiles_list)


__all__ = ["DEFAULT_SCORER_WEIGHTS", "REINVENT4Scorer", "batch_score"]