"""wetlab_reward_channel — additive MCTS reward channel for wet-lab feedback.

============================================================
TODO-30 P5.2 Tier 1 (2026-09-17)
============================================================
Tier-1 wet-lab plumbing (no collaborators, no IRB, just API + data):

* :func:`compute_wetlab_signed_error`  — for a candidate ``smiles`` and
  a measurement dict ``{smiles: pIC50}``, return
  ``-|predicted - measured|`` when measured, else ``0.0``.
* :func:`make_wetlab_channel`  — wraps :func:`compute_wetlab_signed_error`
  into a closure ``(state) -> float`` suitable for
  :class:`RewardAggregator.r_wetlab`.
* :func:`register_wetlab_channel`  — wires the channel + weight on a
  :class:`RewardAggregator` instance.

Honest framing
--------------
* The channel returns a *signed error* (always <= 0) so adding it to the
  aggregator reward PENALISES predictions that disagree with in-vitro
  evidence.  A perfect prediction contributes ``0.0``; a +1.0 log-unit
  error contributes ``-1.0``.  When ``w_wetlab=1.0`` the channel can
  subtract at most ``1.0`` from the leaf reward per candidate.
* For MCTS shaping this means **candidates whose dry-lab prediction
  matches measured pIC50 are favoured**.  This is the *opposite* of
  a pIC50-prediction oracle — we already have ``r_pic50`` for that;
  the wet-lab channel adds the **distance-to-evidence** term.
* Calibration re-fit is NOT in this module.  The companion script
  :mod:`molmetal.scripts.recalibrate_from_assay` reads the same TSV
  via :func:`molmetal_lam.lam_chem.wetlab_protocol.load_assays` and
  emits a JSON calibration report (Pearson r + residual analysis).
* The channel is **opt-in** (``w_wetlab=0.0`` default) — adding the
  channel does not change the aggregator's existing reward surface.

Lit anchors
-----------
* Settles 2012 (J. Med. Chem.) — "pIC50 ought to be within ±0.5 of
  measured for a useful predictor"; we adopt ``|err| < 0.5`` as the
  "good" regime for residual analysis (used by the recalibration
  script).
* Sheridan 2013 (J. Chem. Inf. Model.) — residual-aware QSAR:
  *model accuracy on the in-vitro cohort is the metric that matters*,
  not in-sample accuracy on the training set.  The wet-lab channel
  implements this directly: the candidate's dry-lab prediction is
  REWARDED only when it agrees with the in-vitro measurement.

Constraints
-----------
* CPU-only — no GPU dependencies, no external API calls.  The channel
  is a simple dictionary lookup + scalar arithmetic.
* Additive — does not modify the aggregator's other channels.
* Graceful degradation — never raises; missing measurement returns
  0.0 so the MCTS leaf value remains finite.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional


__all__ = [
    "compute_wetlab_signed_error",
    "make_wetlab_channel",
    "register_wetlab_channel",
]
__version__ = "0.0.1-wetlab-channel"


# ---------------------------------------------------------------------------
# Core scalar mapping
# ---------------------------------------------------------------------------


def compute_wetlab_signed_error(
    smiles: str,
    measurement_dict: Optional[Mapping[str, float]],
    *,
    predicted_pic50: Optional[float] = None,
    predicted_lookup: Optional[Callable[[str], Optional[float]]] = None,
    zero_when_missing: bool = True,
) -> float:
    """Return ``-|predicted - measured|`` for ``smiles`` when measured.

    Parameters
    ----------
    smiles : str
        Candidate SMILES.  Must be the **same canonical form** that was
        used to populate ``measurement_dict`` (we do not re-canonicalise
        here — the loader canonicalised on disk, callers should match).
    measurement_dict : Mapping[str, float] | None
        ``{smiles: pIC50}`` map (typically :func:`wetlab_protocol.assays_to_dict`
        output).  When ``None`` or empty, the function returns ``0.0``.
    predicted_pic50 : float, optional
        Pre-computed pIC50 prediction.  When supplied, this is the
        "predicted" side of the signed-error formula.  When ``None``
        the function falls back to ``predicted_lookup`` (if provided)
        and finally to ``0.0`` (i.e. the contribution collapses to
        ``-|measured|`` — still bounded, never NaN).
    predicted_lookup : Callable[[str], Optional[float]], optional
        A callable that returns a predicted pIC50 for a SMILES.
        Used only when ``predicted_pic50`` is ``None``.  Returning
        ``None`` from the callable is treated as "no prediction" and
        falls back to the no-prediction case below.
    zero_when_missing : bool, default True
        When ``True`` (default), a candidate with **no** measurement
        returns ``0.0`` (no penalty, no bonus).  When ``False`` a
        missing measurement returns ``float("nan")`` so callers can
        distinguish "no evidence" from "perfect prediction" if they
        prefer.  The reward channel expects ``0.0``; the script
        :mod:`molmetal.scripts.recalibrate_from_assay` is the only
        caller that might want ``False``.

    Returns
    -------
    float
        ``-|predicted - measured|`` when both predicted + measured are
        available; ``0.0`` (default) or ``nan`` (when
        ``zero_when_missing=False``) when either side is missing.
        The output is always <= 0.0 so the channel penalises
        disagreement without rewarding agreement explicitly (a perfect
        prediction contributes 0.0; a worse prediction contributes a
        more negative value).
    """
    if not measurement_dict:
        return 0.0 if zero_when_missing else float("nan")
    if smiles is None or smiles == "":
        return 0.0 if zero_when_missing else float("nan")
    if smiles not in measurement_dict:
        return 0.0 if zero_when_missing else float("nan")
    measured = measurement_dict[smiles]
    try:
        measured_f = float(measured)
    except (TypeError, ValueError):
        return 0.0 if zero_when_missing else float("nan")
    # NaN check (NaN != NaN).
    try:
        if measured_f != measured_f:
            return 0.0 if zero_when_missing else float("nan")
    except Exception:
        return 0.0 if zero_when_missing else float("nan")

    # Resolve the predicted side.
    if predicted_pic50 is not None:
        try:
            predicted_f = float(predicted_pic50)
        except (TypeError, ValueError):
            predicted_f = float("nan")
    elif predicted_lookup is not None:
        try:
            predicted_f = predicted_lookup(smiles)
        except Exception:
            predicted_f = float("nan")
        if predicted_f is None:
            predicted_f = float("nan")
        else:
            try:
                predicted_f = float(predicted_f)
            except (TypeError, ValueError):
                predicted_f = float("nan")
    else:
        # No prediction supplied.  Honour ``zero_when_missing`` so the
        # caller can decide whether "no prediction" should silently be
        # treated as 0.0 contribution or as NaN.
        return 0.0 if zero_when_missing else float("nan")

    try:
        if predicted_f != predicted_f:  # NaN check
            return 0.0 if zero_when_missing else float("nan")
    except Exception:
        return 0.0 if zero_when_missing else float("nan")

    err = predicted_f - measured_f
    # Bound the error contribution so a single candidate cannot blow
    # up the leaf value.  Realistic pIC50 prediction errors are in
    # [0, 3] log-units; clamp to that range defensively.
    if err > 3.0:
        err = 3.0
    elif err < -3.0:
        err = -3.0
    # Signed error is always <= 0 (we negate the magnitude).
    return -abs(err)


# ---------------------------------------------------------------------------
# RewardAggregator closures
# ---------------------------------------------------------------------------


def make_wetlab_channel(
    measurement_dict: Optional[Mapping[str, float]],
    *,
    predicted_lookup: Optional[Callable[[str], Optional[float]]] = None,
    zero_when_missing: bool = True,
) -> Callable[[Any], float]:
    """Return a closure ``(state) -> float`` for :attr:`r_wetlab`.

    Parameters
    ----------
    measurement_dict : Mapping[str, float] | None
        ``{smiles: pIC50}`` map.  When ``None`` the returned closure
        always emits 0.0 (so the channel can be safely wired before
        measurements are loaded — opt-in contract).
    predicted_lookup : Callable[[str], Optional[float]], optional
        Forwarded to :func:`compute_wetlab_signed_error`.  When
        ``None`` the channel silently degrades to ``-|measured|``
        which is still well-bounded but slightly penalises any
        measured candidate.  Production callers should wire a
        real pIC50 predictor here.
    zero_when_missing : bool, default True
        Forwarded.  Always leave as ``True`` when the closure is fed
        to :attr:`RewardAggregator.r_wetlab` so the aggregator never
        sees NaN.

    Returns
    -------
    Callable[[Any], float]
        A closure that accepts an MCTS state (anything with a
        ``canonical_smiles()`` method, a ``smiles`` attribute, or a
        plain string) and returns the signed-error contribution for
        that candidate.  Returns ``0.0`` when:

        * the channel was constructed with ``measurement_dict=None``,
        * the candidate SMILES is not in the measurement dict,
        * the candidate SMILES is empty,
        * the predicted or measured value is NaN,
        * RDKit / state parsing raises.

        The closure never raises (MCTS safety contract).
    """

    measurements = dict(measurement_dict) if measurement_dict else {}

    def _channel(state: Any) -> float:
        try:
            smi_fn = getattr(state, "canonical_smiles", None)
            if callable(smi_fn):
                smi = str(smi_fn() or "")
            elif isinstance(state, str):
                smi = state
            else:
                smi = str(getattr(state, "smiles", "") or "")
        except Exception:
            return 0.0
        return compute_wetlab_signed_error(
            smi,
            measurements,
            predicted_lookup=predicted_lookup,
            zero_when_missing=zero_when_missing,
        )

    return _channel


def register_wetlab_channel(
    aggregator: Any,
    measurement_dict: Optional[Mapping[str, float]] = None,
    *,
    weight: float = 0.0,
    predicted_lookup: Optional[Callable[[str], Optional[float]]] = None,
    zero_when_missing: bool = True,
) -> Any:
    """Wire ``r_wetlab`` + ``w_wetlab`` on a :class:`RewardAggregator`.

    Convenience wrapper mirroring
    :func:`register_pharmacophore_channel` — useful for callers that
    want a single import for both the helper and the channel
    registration.

    Parameters
    ----------
    aggregator : RewardAggregator
        The aggregator instance to mutate.  ``agg.r_wetlab`` and
        ``agg.w_wetlab`` are set on return; ``agg._wetlab_n_loaded``
        is also set to ``len(measurement_dict)`` for diagnostics.
    measurement_dict : Mapping[str, float] | None
        ``{smiles: pIC50}`` map.  ``None`` produces a no-op closure
        (always returns 0.0) — still safe to register; the weight
        default ``0.0`` keeps the historical reward bit-for-bit
        identical.  Production callers should load the dict via
        :mod:`molmetal_lam.lam_chem.wetlab_protocol` and pass the
        output of :func:`wetlab_protocol.assays_to_dict`.
    weight : float, default 0.0
        Forwarded to :attr:`w_wetlab`.  Default 0.0 keeps existing
        reward bit-for-bit identical when unused (opt-in contract).
        Set 1.0 for a unit-weight contribution (most useful starting
        point); the channel is bounded so unit weight adds at most
        ``-1.0`` per candidate.
    predicted_lookup : Callable[[str], Optional[float]], optional
        Forwarded to :func:`make_wetlab_channel`.  When ``None`` the
        channel silently degrades to ``-|measured|``.
    zero_when_missing : bool, default True
        Forwarded.  Always leave as ``True`` for the production
        :class:`RewardAggregator` channel so the aggregator never
        sees NaN.

    Returns
    -------
    RewardAggregator
        The same aggregator instance, returned for fluent chaining.
    """
    channel = make_wetlab_channel(
        measurement_dict,
        predicted_lookup=predicted_lookup,
        zero_when_missing=zero_when_missing,
    )
    try:
        aggregator.r_wetlab = channel
        aggregator.w_wetlab = float(weight)
        aggregator._wetlab_n_loaded = (
            int(len(measurement_dict)) if measurement_dict else 0
        )
    except Exception:
        # Aggregator schema missing the new field — fail loudly so
        # callers know they need a proof_search.py with the wetlab
        # fields.  Mirrors the pharmacophore contract: if the field
        # isn't there the aggregator doesn't know how to honour it.
        raise AttributeError(
            "aggregator does not expose r_wetlab / w_wetlab; "
            "ensure molmetal_lam.search_alg.proof_search is up-to-date."
        )
    return aggregator
