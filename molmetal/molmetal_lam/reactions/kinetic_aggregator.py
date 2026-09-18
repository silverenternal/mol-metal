"""Combine reaction-rate predictions into a synthesis feasibility score.

The aggregator uses a weighted harmonic mean: a slow (or unavailable)
reaction therefore limits the overall route score.  An optional, lightweight
Ridge model interpolates rates from known SMILES pairs when PySR/Julia is not
available in the execution environment.
"""
from __future__ import annotations

import logging
from typing import Iterable, Mapping, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "CuAAC": 1.0,
    "SPAAC": 0.8,
    "thiol-ene": 0.7,
    "Suzuki": 0.5,
    "amide-coupling": 0.6,
}


def aggregate_kinetic_score(reaction_rates: dict[str, float]) -> float:
    """Return the harmonic mean of supplied rates, constrained to ``[0, 1]``.

    Empty input has score zero.  Rates are clipped before aggregation so
    malformed predictors cannot produce an invalid feasibility score.
    """
    if not reaction_rates:
        return 0.0
    values = np.clip(np.asarray(list(reaction_rates.values()), dtype=float), 0.0, 1.0)
    if np.any(values <= 0):
        return 0.0
    score = float(len(values) / np.sum(1.0 / values))
    return float(np.clip(score, 0.0, 1.0))


def _pair_similarity(pair: tuple[str, str]) -> float:
    """Compute molecular fingerprint similarity for a SMILES pair."""
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem

        molecules = [Chem.MolFromSmiles(s) for s in pair]
        if any(m is None for m in molecules):
            return 0.0
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in molecules]
        return float(DataStructs.TanimotoSimilarity(fps[0], fps[1]))
    except Exception as exc:  # pragma: no cover - optional dependency path
        LOGGER.warning("Fingerprint similarity unavailable: %s", exc)
        return 0.0


class KineticAggregator:
    """Weighted kinetic score facade with optional sklearn interpolation."""

    def __init__(self, weights: dict | None = None, known_rate_pairs=None):
        # Supplying weights defines the reaction set explicitly; this makes
        # small route-specific aggregators possible without implicit zeros.
        merged = (dict(DEFAULT_WEIGHTS) if weights is None else
                  {str(k): float(v) for k, v in weights.items()})
        self._weights = merged
        self._known_rate_pairs = known_rate_pairs
        self._model = None

    @property
    def weights(self) -> dict[str, float]:
        """Return a copy of reaction weights."""
        return dict(self._weights)

    def __call__(self, reaction_rates: Mapping[str, float]) -> float:
        weighted = {}
        for name, weight in self._weights.items():
            if name not in reaction_rates:
                LOGGER.warning("Missing reaction rate for %s; defaulting to 0.0", name)
                rate = 0.0
            else:
                rate = float(reaction_rates[name])
            weighted[name] = float(np.clip(rate, 0.0, 1.0))
        # Weighting is applied as a weighted harmonic mean. Zero weights are
        # omitted; if all weights are zero, no kinetic evidence is available.
        positive = [(k, self._weights[k]) for k in weighted if self._weights[k] > 0]
        if not positive or any(weighted[k] <= 0 for k, _ in positive):
            return 0.0
        denominator = sum(w / weighted[k] for k, w in positive)
        return float(np.clip(sum(w for _, w in positive) / denominator, 0.0, 1.0))

    def fit_rate_interpolator(self, known_rate_pairs=None) -> "KineticAggregator":
        """Fit Ridge on pairwise fingerprint similarities and known rates."""
        pairs = known_rate_pairs if known_rate_pairs is not None else self._known_rate_pairs
        rows = self._normalise_pairs(pairs)
        if len(rows) < 2:
            raise ValueError("at least two known SMILES/rate pairs are required")
        from sklearn.linear_model import Ridge

        x = np.asarray([[_pair_similarity(pair)] for pair, _ in rows])
        y = np.asarray([rate for _, rate in rows])
        self._model = Ridge(alpha=1.0).fit(x, y)
        return self

    def interpolate_rate(self, smiles_a: str, smiles_b: str, known_rate_pairs=None) -> float:
        """Predict a rate for a SMILES pair using the sklearn fallback."""
        if known_rate_pairs is not None or self._model is None:
            self.fit_rate_interpolator(known_rate_pairs)
        value = float(self._model.predict([[_pair_similarity((smiles_a, smiles_b))]])[0])
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _normalise_pairs(pairs) -> list[tuple[tuple[str, str], float]]:
        if pairs is None:
            return []
        if isinstance(pairs, Mapping):
            pairs = [(*key, value) for key, value in pairs.items()]
        rows = []
        for item in pairs:
            if len(item) == 2 and isinstance(item[0], (tuple, list)):
                pair, rate = item
            else:
                a, b, rate = item
                pair = (a, b)
            rows.append(((str(pair[0]), str(pair[1])), float(np.clip(rate, 0.0, 1.0))))
        return rows


__all__ = ["DEFAULT_WEIGHTS", "KineticAggregator", "aggregate_kinetic_score"]
