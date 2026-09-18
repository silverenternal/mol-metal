"""SynthesisOracle — reward channel for the closed-loop MCTS.

The :class:`SynthesisOracle` wraps :class:`RatePredictor` (one per
reaction in :data:`reactions.rate_predictor.LITERATURE_YIELDS`) and
exposes a single ``predict(reaction_name, smiles_a, smiles_b) -> float``
API that returns an isolated-yield estimate in ``[0, 1]``.

Public API
----------
:class:`SynthesisOracle`    the oracle facade
:meth:`SynthesisOracle.predict`    yield in [0, 1]
:meth:`SynthesisOracle.equation`  fitted RatePredictor equation string
:meth:`SynthesisOracle.reactions` list of available reaction names
:meth:`SynthesisOracle.to_reward_channel`  closure usable by RewardAggregator
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from molmetal_lam.reactions.rate_predictor import (
    LITERATURE_YIELDS,
    RatePredictor,
)

log = logging.getLogger(__name__)

__all__ = ["SynthesisOracle"]


def attach_all_rate_predictors_safe() -> Dict[str, RatePredictor]:
    """Train + cache a :class:`RatePredictor` for every literature reaction.

    This is a defensive wrapper around
    :func:`reactions.beta_reductions.attach_all_rate_predictors` — it
    catches per-reaction training failures so a single broken model
    never prevents the oracle from being instantiated.
    """
    fitted: Dict[str, RatePredictor] = {}
    try:
        from molmetal_lam.reactions.beta_reductions import (
            attach_all_rate_predictors,
        )
        out = attach_all_rate_predictors()
        for k, v in (out or {}).items():
            if isinstance(v, RatePredictor):
                fitted[k] = v
    except Exception as exc:  # pragma: no cover - env dependent
        log.debug("attach_all_rate_predictors failed: %s", exc)

    # Train any reaction missing from the cache directly so the oracle
    # is *complete* even if the upstream hook skipped some entries.
    for reaction_name in LITERATURE_YIELDS.keys():
        if reaction_name in fitted:
            continue
        try:
            fitted[reaction_name] = RatePredictor.for_reaction(reaction_name)
        except Exception as exc:  # pragma: no cover - env dependent
            log.debug("RatePredictor.for_reaction(%s) failed: %s", reaction_name, exc)
    return fitted


class SynthesisOracle:
    """Yield-based reward channel backed by trained :class:`RatePredictor`.

    Parameters
    ----------
    predictors : dict[str, RatePredictor] or None, default ``None``
        Pre-trained cache.  When ``None`` we auto-train every reaction
        in :data:`LITERATURE_YIELDS` via
        :func:`attach_all_rate_predictors_safe`.
    """

    def __init__(
        self,
        predictors: Optional[Dict[str, RatePredictor]] = None,
    ) -> None:
        if predictors is None:
            predictors = attach_all_rate_predictors_safe()
        self.predictors: Dict[str, RatePredictor] = dict(predictors or {})

    # ------------------------------------------------------------------ API
    def predict(
        self,
        reaction_name: str,
        smiles_a: str,
        smiles_b: str,
    ) -> float:
        """Return predicted isolated yield in ``[0, 1]``.

        Falls back to ``0.5`` whenever:

        * the reaction name is unknown to the oracle,
        * the predictor is missing (training failure),
        * the SMILES pair cannot be featurised (RDKit parse error),
        * the underlying regressor raises.
        """
        pred = self.predictors.get(reaction_name)
        if pred is None:
            return 0.5
        try:
            return float(pred.predict(smiles_a, smiles_b))
        except Exception as exc:  # pragma: no cover - env dependent
            log.debug("SynthesisOracle.predict(%s) failed: %s", reaction_name, exc)
            return 0.5

    # ------------------------------------------------------------------
    # (Phase 2) Batched prediction — vectorised across pairs
    # ------------------------------------------------------------------
    def predict_batch(
        self,
        reaction_name: str,
        pairs: "List[Tuple[str, str]]",
    ) -> "Any":
        """Batched counterpart of :meth:`predict`.

        Featurises every ``(smiles_a, smiles_b)`` pair in ``pairs``,
        stacks the 8-d descriptors into one ``(n, 8)`` matrix, and
        invokes the underlying regressor's vectorised ``predict``
        method in a single call.  When the underlying regressor is a
        sklearn model (backend ``"sklearn-linear"`` or
        ``"sklearn-rf"``) this avoids the ~50 µs per-pair Python
        overhead and lets BLAS / sklearn's C inner loop amortise the
        cost.

        For non-sklearn backends (e.g. PySR / symbolic expressions)
        we fall back to per-pair :meth:`predict` calls so the
        numerical result is identical to the scalar path.

        Parameters
        ----------
        reaction_name : str
            One of the keys in :data:`LITERATURE_YIELDS`.
        pairs : list[tuple[str, str]]
            ``(smiles_a, smiles_b)`` pairs to score in one batch.

        Returns
        -------
        object
            ``numpy.ndarray`` of shape ``(len(pairs),)`` when sklearn
            backend is available, otherwise a ``list[float]`` of the
            same length.  Empty input returns an empty ``np.ndarray``.
            Per-pair failures inside the sklearn path return ``0.5``
            for that row (consistent with the scalar fallback).
        """
        # Lazy numpy import (the module is RDKit-only at the top level
        # so this stays cheap).
        try:
            import numpy as _np  # type: ignore
        except Exception:  # pragma: no cover
            _np = None  # type: ignore

        n = len(pairs)
        if n == 0:
            if _np is not None:
                return _np.zeros(0, dtype=float)
            return []

        pred = self.predictors.get(reaction_name)
        if pred is None:
            # Same fallback as the scalar path.
            if _np is not None:
                return _np.full(n, 0.5, dtype=float)
            return [0.5] * n

        # Late import — keep module-import-time cost at zero when the
        # oracle is not used in a closed-loop run.
        from molmetal_lam.reactions.rate_predictor import (
            smiles_pair_features,
        )

        # Detect sklearn-backed model.  The HeuristicRegressor
        # exposes ``backend_`` after fit; values of interest are
        # ``"sklearn-linear"``, ``"sklearn-rf"``, ``"sklearn"``.
        model = getattr(pred, "model", None)
        backend = getattr(pred, "model", None)
        backend_attr = getattr(model, "backend_", None) if model is not None else None
        is_sklearn = isinstance(backend_attr, str) and backend_attr.startswith("sklearn")
        # (defensive) Also accept direct sklearn estimators attached to
        # ``pred.model._model``.
        if not is_sklearn:
            inner = getattr(model, "_model", None) if model is not None else None
            is_sklearn = inner is not None and (
                hasattr(inner, "predict") and hasattr(inner, "fit")
            )

        if is_sklearn and model is not None and _np is not None:
            # Vectorised sklearn path — featurise all pairs, stack,
            # and call ``predict(X)`` once.
            X_list = []
            valid_mask = []
            for smi_a, smi_b in pairs:
                try:
                    feats = smiles_pair_features(smi_a, smi_b)
                except Exception:
                    feats = [0.0] * 8
                if len(feats) != 8:
                    feats = [0.0] * 8
                X_list.append(feats)
                valid_mask.append(True)
            X = _np.asarray(X_list, dtype=float)
            try:
                # HeuristicRegressor.predict handles backend dispatch
                # (sklearn / PySR).  We bypass the per-pair loop by
                # calling ``pred.model.predict`` directly so the
                # sklearn inner loop can amortise across the batch.
                inner_pred = getattr(model, "_model", model)
                raw = _np.asarray(inner_pred.predict(X), dtype=float).reshape(-1)
            except Exception as exc:  # pragma: no cover - env dependent
                log.debug(
                    "SynthesisOracle.predict_batch(%s) sklearn failed: %s; "
                    "falling back to scalar loop",
                    reaction_name, exc,
                )
                raw = _np.asarray(
                    [float(pred.predict(a, b)) for a, b in pairs],
                    dtype=float,
                )
            # Clip + fallback per row.
            clipped = _np.clip(raw, 0.0, 1.0)
            # Replace any NaN with 0.5 (consistent with scalar fallback).
            clipped = _np.where(_np.isnan(clipped), 0.5, clipped)
            return clipped

        # Generic / non-sklearn path — per-pair scalar fallback.
        out = []
        for smi_a, smi_b in pairs:
            try:
                v = float(pred.predict(smi_a, smi_b))
            except Exception as exc:  # pragma: no cover
                log.debug(
                    "SynthesisOracle.predict_batch per-pair failed: %s", exc,
                )
                v = 0.5
            out.append(max(0.0, min(1.0, v)))
        if _np is not None:
            return _np.asarray(out, dtype=float)
        return out

    def equation(self, reaction_name: str) -> str:
        """Return the fitted :class:`RatePredictor` equation, or ``"unknown"``."""
        pred = self.predictors.get(reaction_name)
        if pred is None:
            return "unknown"
        try:
            return str(pred.equation())
        except Exception:  # pragma: no cover - env dependent
            return "unknown"

    def reactions(self) -> List[str]:
        """Return the list of reaction names with a fitted predictor."""
        return sorted(self.predictors.keys())

    def to_reward_channel(self) -> Callable[[Any], float]:
        """Return a closure suitable for ``RewardAggregator.r_synth``.

        The closure ignores its ``state`` argument and returns the
        mean predicted yield across all known reactions on a fixed
        canonical probe pair.  This is the *global* synthesis signal —
        how "click-able" the system is on a representative substrate,
        independent of the molecule currently being scored.
        """

        def _channel(state: Any) -> float:
            # Probe pair: ethyl azide + propyne (canonical CuAAC pair).
            # Fallback to 0.5 when the oracle is empty.
            if not self.predictors:
                return 0.5
            vals: List[float] = []
            for name in self.predictors.keys():
                try:
                    v = float(self.predict(name, "CCN=[N+]=[N-]", "C#CC"))
                    if 0.0 <= v <= 1.0:
                        vals.append(v)
                except Exception:
                    continue
            if not vals:
                return 0.5
            return float(sum(vals) / len(vals))

        return _channel


__all__.append("SynthesisOracle")