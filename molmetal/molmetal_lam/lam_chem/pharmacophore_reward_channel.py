"""Pharmacophore reward channel — additive MCTS shaping signal.

============================================================
TODO-30 P2.5 Phase A (2026-09-17)
============================================================
This module is a thin wrapper over the locked
:mod:`molmetal_lam.lam_chem.pharmacophore_filter` module.  Its only
purpose is to expose a *continuous* pharmacophore score in ``[0, 1]``
suitable for additive MCTS leaf shaping:

* :func:`compute_pharmacophore_score` returns ``max(0.0, 1 - total_violations / 7)``
  (a partial-credit reward that returns 1.0 for a strict pass and
  0.0 for the all-violations sentinel).  It never raises — RDKit
  parse failures degrade to 0.0.

Why a wrapper (and not modify ``pharmacophore_filter.py`` directly)
------------------------------------------------------------------
The filter is a **boolean gate** used by the MLC pipeline as a hard
pre-filter before the expensive docking evaluators (Vina, PoseBusters,
AiZynth).  Modifying its public surface would silently change every
downstream caller (Round-12 / Round-13 sweeps, the closed-loop
regression tests, the 7/10 pre-existing tests in
``test_pharmacophore_filter.py``).  Keeping the filter locked and
adding this wrapper preserves bit-for-bit behaviour for all 24 REAL
adapters.

Reward semantics
----------------
For a single SMILES ``s`` the wrapper computes

    report = compute_pharmacophore_report(s, ...)
    score  = max(0.0, 1.0 - report.total_violations / 7.0)

``report.total_violations`` ranges from 0 (strict pass) to 7 (all
Lipinski+Veber+ring violations, including RDKit parse failure).
The mapping is monotonic non-increasing — fewer violations → higher
score — so it can be used as a soft reward shaping term without
distorting the dominant Vina / SA / QED reward channels.

Public surface
--------------
* :func:`compute_pharmacophore_score`  continuous [0, 1] score
* :func:`register_pharmacophore_channel`  RewardAggregator hook
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from molmetal_lam.lam_chem.pharmacophore_filter import (
    compute_pharmacophore_report,
    pass_pharmacophore,
)


__all__ = [
    "compute_pharmacophore_score",
    "register_pharmacophore_channel",
]
__version__ = "0.0.1-pharmacophore-channel"


# ---------------------------------------------------------------------------
# Continuous reward mapping
# ---------------------------------------------------------------------------


def compute_pharmacophore_score(
    smiles: str,
    *,
    allow_acyclic: bool = False,
) -> float:
    """Return a continuous pharmacophore reward in ``[0, 1]``.

    Maps ``compute_pharmacophore_report(s).total_violations`` to
    ``max(0.0, 1.0 - total_violations / 7)``.  ``total_violations``
    is the sum of Lipinski Ro5 (0..4) + Veber (0..2) + ring
    violations (0 or 1) and is bounded by the report dataclass so
    the maximum is exactly 7.  The function is intentionally simple
    so the resulting signal is well-bounded, monotonic, and easy to
    reason about as an additive MCTS leaf prior.

    Parameters
    ----------
    smiles : str
        Input SMILES.  Empty / unparseable SMILES returns 0.0 (the
        same behaviour as the underlying
        :func:`compute_pharmacophore_report`).
    allow_acyclic : bool, default False
        Forwarded to :func:`compute_pharmacophore_report`; bypasses
        the ring-count rule (useful for cisplatin / nedaplatin).

    Returns
    -------
    float
        Continuous score in ``[0.0, 1.0]``.  Higher = more
        drug-like.  ``1.0`` iff ``total_violations == 0`` (strict
        pass).  ``0.0`` iff ``total_violations >= 7`` (all-fail
        sentinel, includes RDKit parse failure).
    """
    if not smiles or not isinstance(smiles, str):
        return 0.0
    try:
        report = compute_pharmacophore_report(smiles, allow_acyclic=allow_acyclic)
    except Exception:
        # Graceful degradation — never raise from a reward channel.
        return 0.0
    viol_attr = getattr(report, "total_violations", 7)
    try:
        viol = int(viol_attr) if viol_attr is not None else 7
    except (TypeError, ValueError):
        viol = 7
    score = 1.0 - (float(viol) / 7.0)
    # Clamp to [0, 1] so future callers passing non-default bounds
    # cannot blow up the MCTS leaf value.
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return float(score)


# ---------------------------------------------------------------------------
# RewardAggregator hook
# ---------------------------------------------------------------------------


def register_pharmacophore_channel(
    aggregator: Any,
    *,
    weight: float = 0.0,
    strict: bool = False,
    allow_acyclic: bool = False,
) -> Any:
    """Wire ``r_pharmacophore`` on a :class:`RewardAggregator` instance.

    Convenience wrapper around
    :meth:`RewardAggregator.register_pharmacophore_channel` — useful
    for callers that want a single import for both the helper and
    the channel registration.

    Parameters
    ----------
    aggregator : RewardAggregator
        The aggregator instance to wire the channel into.  Mutated
        in place — ``agg.r_pharmacophore`` and ``agg.w_pharmacophore``
        are set on return.
    weight : float, default 0.0
        Per-channel weight forwarded to :attr:`w_pharmacophore`.
        Default ``0.0`` keeps the historical reward bit-for-bit
        identical when unused (opt-in).
    strict : bool, default False
        When True, use the boolean strict mode of
        :func:`pass_pharmacophore` (1.0 if zero violations, else
        0.0).  When False (the default), use the continuous
        partial-credit mode via
        :func:`compute_pharmacophore_score`.
    allow_acyclic : bool, default False
        Forwarded to :func:`compute_pharmacophore_score`; bypasses
        the ring-count rule for metal-acylclic derivatives.

    Returns
    -------
    RewardAggregator
        The same aggregator instance, returned for fluent chaining.
    """
    register = getattr(aggregator, "register_pharmacophore_channel", None)
    if not callable(register):
        raise AttributeError(
            "aggregator does not expose register_pharmacophore_channel; "
            "ensure molmetal_lam.search_alg.proof_search is importable."
        )
    register(
        weight=float(weight),
        strict=bool(strict),
        allow_acyclic=bool(allow_acyclic),
    )
    return aggregator
