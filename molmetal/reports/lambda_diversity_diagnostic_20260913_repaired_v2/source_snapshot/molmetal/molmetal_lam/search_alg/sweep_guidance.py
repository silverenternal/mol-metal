"""Explicit, serializable guidance for controlled sweep ablations.

The fitted backend is a linear descriptor prior, NOT PySR symbolic discovery.
Frozen inference is the default; training requires development/training data,
while opt-in transductive updates are labelled and cannot become frozen-test
pretraining. No implicit process-global replay, synthetic labels or SA oracle.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import numpy as np

from .proof_search import MCTSProofSearch, SymbolicPrior, _default_feature_extractor

FEATURES = ["n_atoms", "n_bonds", "n_free_sites", "n_heavy"]
MIN_OBSERVATIONS = 8


class AffineModel:
    def __init__(self, weights, intercept):
        self.weights = np.asarray(weights, dtype=float)
        self.intercept = float(intercept)
    def predict(self, x):
        return np.asarray(x) @ self.weights + self.intercept


class GuidedMCTS(MCTSProofSearch):
    """Use a fitted prior in PUCT before the baseline reward heuristic."""
    def _prior(self, state):
        if self.prior is not None and self.prior.fitted:
            self.learned_prior_calls = getattr(self, "learned_prior_calls", 0) + 1
            return float(self.prior.predict_proba(state))
        return super()._prior(state)


def prepare_prior(state, *, seed, mode="frozen", data_split="test"):
    if mode not in {"frozen", "train", "transductive"}:
        raise ValueError("prior_mode must be frozen, train, or transductive")
    if mode == "train" and data_split not in {"train", "development"}:
        raise ValueError("train prior_mode requires train/development data, never held-out test")
    if state is None:
        state = {"schema_version": 1, "backend": "linear_descriptor_prior",
                 "features": FEATURES, "experiment_seed": seed,
                 "observations": [], "model": None, "pockets_seen": 0,
                 "fit_count": 0, "last_fit_observations": 0,
                 "source_splits": [], "transductive": False}
    result = deepcopy(state)
    if result.get("schema_version") != 1 or result.get("features") != FEATURES:
        raise ValueError("Incompatible prior_state schema/features")
    if result.get("backend") != "linear_descriptor_prior":
        raise ValueError("Only explicitly labelled linear_descriptor_prior state is supported")
    if mode != "frozen" and result.get("experiment_seed") != seed:
        raise ValueError("Mutable prior replay must remain independent for each experiment seed")
    for row in result.get("observations", []):
        if len(row["features"]) != len(FEATURES) or not all(math.isfinite(float(x)) for x in [*row["features"], row["score"]]):
            raise ValueError("Nonfinite/incompatible prior replay observation")
    splits = set(result.get("source_splits", [])) | {r.get("split") for r in result.get("observations", [])}
    if mode != "transductive" and (result.get("transductive") or splits - {"train", "development"}):
        raise ValueError("Held-out/transductive observations cannot be presented as frozen pretraining")
    prior = None
    model = result.get("model")
    if model is not None:
        if len(model["weights"]) != len(FEATURES) or not all(math.isfinite(float(v)) for v in [*model["weights"], model["intercept"]]):
            raise ValueError("Invalid linear prior coefficients")
        if not splits:
            raise ValueError("Fitted prior requires recorded training provenance")
        prior = SymbolicPrior()
        prior._model = AffineModel(model["weights"], model["intercept"])
        prior.fitted = True
        prior.n_features = len(FEATURES)
    return result, prior


def update_prior(state, observations, *, mode, data_split, refit_every):
    """Fit only real supplied observations; state is copied, never mutated."""
    result = deepcopy(state)
    if mode == "frozen":
        return result, {"update_status": "frozen", "fit_performed": False}
    if refit_every < 1:
        raise ValueError("symbolic_prior_refit_every must be positive (units: pockets)")
    if mode == "train" and data_split not in {"train", "development"}:
        raise ValueError("Cannot train on held-out observations")
    records = {r["smiles"]: r for r in result["observations"]}
    for observation in observations:
        row = deepcopy(observation)
        if len(row["features"]) != len(FEATURES) or not all(math.isfinite(float(v)) for v in [*row["features"], row["score"]]):
            continue
        row["split"] = data_split
        records[row["smiles"]] = row
    result["observations"] = list(records.values())
    result["source_splits"] = sorted(set(result["source_splits"]) | ({data_split} if observations else set()))
    result["transductive"] = bool(result["transductive"] or mode == "transductive")
    result["pockets_seen"] += 1
    details = {"fit_performed": False, "n_observations": len(records),
               "min_observations": MIN_OBSERVATIONS}
    if result["pockets_seen"] % refit_every:
        return result, {**details, "update_status": "waiting_for_pocket_interval"}
    if len(records) < MIN_OBSERVATIONS:
        return result, {**details, "update_status": "insufficient_real_observations"}
    def values(rows):
        return sorted((r["smiles"], tuple(r["features"]), r["score"]) for r in rows)
    if result["model"] is not None and values(result["observations"]) == values(state["observations"]):
        return result, {**details, "update_status": "no_new_observations"}
    x = np.asarray([r["features"] for r in records.values()], dtype=float)
    y = np.asarray([r["score"] for r in records.values()], dtype=float)
    if np.unique(x, axis=0).shape[0] < 2 or np.ptp(y) < 1e-12:
        return result, {**details, "update_status": "insufficient_observed_variation"}
    from sklearn.linear_model import Ridge
    center, scale = x.mean(0), x.std(0)
    scale[scale < 1e-12] = 1.0
    fitted = Ridge(alpha=1.0).fit((x - center) / scale, y)
    weights = fitted.coef_ / scale
    result["model"] = {"weights": weights.tolist(),
                       "intercept": float(fitted.intercept_ - weights @ center)}
    result["fit_count"] += 1
    result["last_fit_observations"] = len(records)
    return result, {**details, "fit_performed": True, "update_status": "fitted",
                    "target": "observed_search_reward_not_experimental_affinity"}


def candidate_observations(candidates, search, source_id):
    """Labels are evaluated search rewards on actual returned candidates."""
    rows = []
    for state in candidates:
        try:
            features = _default_feature_extractor(state)
            score = float(search.score_final(state))
            if all(math.isfinite(float(v)) for v in [*features, score]):
                rows.append({"smiles": state.canonical_smiles(), "features": features,
                             "score": score, "source_id": source_id})
        except Exception:
            continue
    return rows



# Preserve the sweep API while allowing isolated chemistry workers to import
# the synthesis gate without pulling in proof search or ROCm torch.
from ..sbdd_env.synthesis_gate import build_synthesis_gate, gate_candidates  # noqa: E402,F401
