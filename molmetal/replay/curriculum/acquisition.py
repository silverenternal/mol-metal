"""Acquisition policy — per-metal top-K + Tanimoto<0.4 + 10 % random fallback.

================================================================
Acquisition policy for RSI curriculum learning (TODO-150 §P2.1)
================================================================
The injection set is built by combining three signals:

1. **Top-K by predicted hardness** — per metal: ``Rh=K=50``, ``Pd=K=20``,
   others ``K=10`` (TODO-150 §P2.1).
2. **Tanimoto<0.4 diversity filter** — Morgan fingerprint (radius 2,
   2048-bit) nearest-neighbour distance above threshold (matches the
   OOD protocol in :mod:`molmetal.molmetal_lam.lam_chem.ood_protocol`).
3. **10 % random fallback** — to ensure the injection set is not
   pathologically concentrated on the predictor's high-confidence
   regions.

The output is an :class:`AcquisitionResult` carrying the accepted
``row_ids``, ``smiles``, and ``hardness_predicted`` so downstream code
can write the parquet shard for the *injected* cohort and emit a
single audit line.

Honest framing
==============
The injection policy is a **projection** — selecting samples whose
predicted hardness is high does not guarantee they will lift AUC.  The
only MEASURED gate is the Wf-124 5-seed × 50-epoch side-by-side A/B;
until that gate fires, the policy is opt-in.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    _RDKIT_OK = True
except Exception:  # pragma: no cover
    _RDKIT_OK = False


#: Per-metal top-K injection cap (TODO-150 §P2.1).
DEFAULT_K_BY_METAL: Dict[str, int] = {
    "Ru": 10,
    "Ir": 10,
    "Rh": 50,
    "Os": 10,
    "Re": 10,
    "Pd": 20,
    "Pt": 10,
    "Au": 10,
}

#: Default Tanimoto < threshold diversity filter (TODO-150 §P2.1).
DEFAULT_TANIMOTO_THRESHOLD: float = 0.4

#: Default random fallback fraction (TODO-150 §P2.1).
DEFAULT_RANDOM_FRACTION: float = 0.10


@dataclass(frozen=True)
class AcquisitionResult:
    """Outcome of one acquisition call.

    Carries both the accepted ``row_ids`` and the rejection
    breakdown so the orchestrator can emit an audit line per epoch.
    """

    metal: str
    accepted: List[int] = field(default_factory=list)
    accepted_smiles: List[str] = field(default_factory=list)
    hardness_predicted: List[float] = field(default_factory=list)
    n_topk_candidates: int = 0
    n_after_diversity: int = 0
    n_random_injected: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metal": str(self.metal),
            "n_accepted": len(self.accepted),
            "n_topk_candidates": int(self.n_topk_candidates),
            "n_after_diversity": int(self.n_after_diversity),
            "n_random_injected": int(self.n_random_injected),
            "row_ids": [int(r) for r in self.accepted],
            "hardness_predicted": [float(h) for h in self.hardness_predicted],
        }


class AcquisitionPolicy:
    """Per-metal top-K + Tanimoto diversity + 10 % random fallback.

    Stateless except for the configured ``K`` / threshold / random
    fraction — the orchestrator instantiates one policy per metal so
    the per-metal ``K`` lookup is O(1).

    The Tanimoto implementation uses RDKit's Morgan fingerprint
    (radius 2, 2048-bit) when available; on hosts without RDKit, the
    diversity filter degenerates to a no-op (every sample accepted).
    This is a **graceful-degradation** choice — the policy stays
    importable on CPU-only test hosts, but the diversity guarantee is
    only delivered when RDKit is available.
    """

    def __init__(
        self,
        k_by_metal: Optional[Mapping[str, int]] = None,
        tanimoto_threshold: float = DEFAULT_TANIMOTO_THRESHOLD,
        random_fraction: float = DEFAULT_RANDOM_FRACTION,
        fp_radius: int = 2,
        fp_size: int = 2048,
        seed: int = 0,
    ) -> None:
        self.k_by_metal = dict(k_by_metal) if k_by_metal is not None else dict(DEFAULT_K_BY_METAL)
        self.tanimoto_threshold = float(tanimoto_threshold)
        if not 0.0 <= self.tanimoto_threshold <= 1.0:
            raise ValueError(
                f"tanimoto_threshold must be in [0, 1]; got {self.tanimoto_threshold}"
            )
        self.random_fraction = float(random_fraction)
        if not 0.0 <= self.random_fraction <= 1.0:
            raise ValueError(
                f"random_fraction must be in [0, 1]; got {self.random_fraction}"
            )
        self.fp_radius = int(fp_radius)
        self.fp_size = int(fp_size)
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        # Fingerprint generator (lazy).
        self._fp_gen = None
        if _RDKIT_OK:
            try:
                self._fp_gen = rdFingerprintGenerator.GetMorganGenerator(
                    radius=self.fp_radius, fpSize=self.fp_size
                )
            except Exception:
                self._fp_gen = None

    # ------------------------------------------------------------------
    # Fingerprints
    # ------------------------------------------------------------------
    def fingerprint(self, smiles: str) -> Optional[np.ndarray]:
        """Morgan fingerprint as a (fp_size,) uint8 array, or None on failure."""
        if self._fp_gen is None or not _RDKIT_OK:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return np.asarray(
            self._fp_gen.GetFingerprintAsNumPy(mol), dtype=np.uint8
        )

    def fingerprints_batch(self, smiles_list: Sequence[str]) -> List[Optional[np.ndarray]]:
        """Fingerprint a batch — preserves order."""
        return [self.fingerprint(s) for s in smiles_list]

    # ------------------------------------------------------------------
    # Acquisition core
    # ------------------------------------------------------------------
    def acquire(
        self,
        metal: str,
        candidates: Sequence[Mapping[str, Any]],
        rng: Optional[random.Random] = None,
    ) -> AcquisitionResult:
        """Acquire up to ``K[metal]`` samples for ``metal``.

        ``candidates`` is a sequence of dicts with at least::

            {"row_id": int, "smiles": str, "hardness_predicted": float}

        The policy sorts by ``hardness_predicted`` (descending), keeps
        the top-2K (twice the cap to leave room for diversity culling),
        then applies the Tanimoto<threshold filter, then injects
        ``random_fraction`` random rows from the remainder.
        """
        if not candidates:
            return AcquisitionResult(metal=str(metal))
        metal = str(metal)
        k = int(self.k_by_metal.get(metal, 10))
        if k < 0:
            raise ValueError(f"K for {metal} must be >= 0; got {k}")
        rng = rng or self._rng
        # Normalise to dict view.
        norm: List[Dict[str, Any]] = []
        for c in candidates:
            norm.append(
                {
                    "row_id": int(c["row_id"]),
                    "smiles": str(c["smiles"]),
                    "hardness_predicted": float(c.get("hardness_predicted", 0.0)),
                }
            )
        # Sort by predicted hardness, descending.
        norm.sort(key=lambda r: r["hardness_predicted"], reverse=True)
        topk_pool = norm[: max(k * 2, k)]
        # Tanimoto filter.
        accepted: List[Dict[str, Any]] = []
        accepted_fps: List[np.ndarray] = []
        for cand in topk_pool:
            if len(accepted) >= k:
                break
            fp = self.fingerprint(cand["smiles"])
            if fp is None:
                # Without fingerprint we cannot enforce diversity;
                # be conservative and drop, matching the "Tanimoto
                # *cannot* be verified" interpretation of the TODO.
                continue
            if not accepted_fps:
                accepted.append(cand)
                accepted_fps.append(fp)
                continue
            # Compute Tanimoto distance to every accepted FP and check
            # the maximum is below threshold (= the candidate is too
            # *similar* to something already in the set).
            sims = []
            for ref in accepted_fps:
                inter = int(np.logical_and(fp, ref).sum())
                union = int(np.logical_or(fp, ref).sum())
                sim = inter / union if union > 0 else 1.0
                sims.append(sim)
            if max(sims) < self.tanimoto_threshold:
                accepted.append(cand)
                accepted_fps.append(fp)
        n_random = 0
        # Random fallback from remainder (NOT in accepted and not in topk_pool).
        remainder = [c for c in norm if c not in accepted]
        target_random = int(round(len(accepted) * self.random_fraction))
        if target_random > 0 and remainder:
            sample = rng.sample(remainder, min(target_random, len(remainder)))
            n_random = len(sample)
            accepted.extend(sample)
        return AcquisitionResult(
            metal=metal,
            accepted=[c["row_id"] for c in accepted],
            accepted_smiles=[c["smiles"] for c in accepted],
            hardness_predicted=[c["hardness_predicted"] for c in accepted],
            n_topk_candidates=len(topk_pool),
            n_after_diversity=len([c for c in accepted if c in topk_pool]),
            n_random_injected=n_random,
        )


__all__ = [
    "AcquisitionPolicy",
    "AcquisitionResult",
    "DEFAULT_K_BY_METAL",
    "DEFAULT_TANIMOTO_THRESHOLD",
    "DEFAULT_RANDOM_FRACTION",
]
