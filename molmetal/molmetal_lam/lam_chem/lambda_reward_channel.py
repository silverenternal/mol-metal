"""TODO-21 Strategy 1 — Lambda-as-reward channel.

================================================================
Background
================================================================
TODO-21 (Lambda × CFM coupling) was deferred in 2026-09-14 and
re-opened in 2026-09-16.  This module implements **Strategy 1**:
*Lambda MCTS output becomes a reward signal for CFM retraining.*

Strategy 1 is the **lowest-effort / lowest-risk** of the three
candidate coupling strategies documented in
``TODO/pending/21_lambda_model_coupling.md`` §3.  It is a one-line
adapter change in :class:`molmetal.adapters.flow_matching_lipman` plus a
new reward channel on :class:`RewardAggregator` that consumes the
post-MetalGeometryPrior gate Lambda MCTS candidates and reports the
mean cosine similarity to the current CFM pocket embedding (computed
via :mod:`molmetal_lam.lam_chem.coupling_adapter`).

The CPU-only payoff is:

1. :func:`compute_lambda_score` accepts a list of validated Lambda
   candidate SMILES, embeds them via the existing 64-d pocket bridge,
   and returns the mean cosine similarity to the current CFM
   embedding.
2. :func:`make_lambda_reward_channel` wraps that into a
   ``(state) -> float`` callable matching the :class:`RewardAggregator`
   contract; the closure accepts both plain SMILES strings and
   :class:`MoleculeClosedTerm`-like state objects.
3. :func:`register_lambda_reward_channel` is the convenience entry
   point that wires the channel into :class:`RewardAggregator.r_lambda_score`
   (NEW attribute, additive only — no existing channel is mutated) and
   sets :attr:`RewardAggregator.w_lambda_score` to the requested
   weight (default ``0.0`` = opt-in, regression-proof).

Honest framing
==============
The cosine is over the **64-d pocket bridge** from
:mod:`molmetal_lam.lam_chem.coupling_adapter`, which is a **stand-in**
for the real CFM pocket embedding (the underlying MLP was trained on
8 mols in ``--dry-run`` mode).  The bit-for-bit determinism and shape
contract are unchanged.  When the coupling is **disabled** (env var
``COUPLING_ENABLED`` unset) the channel returns 0.0 so the existing
aggregator is bit-for-bit identical.

Lit anchor
==========
This channel implements *reward shaping* in the Dayan 1997 sense:
adding a potential function to the reward does not change the optimal
policy of an MDP when the shaping is path-independent.  For Lambda
MCTS the "policy" is the closed-form search; shaping it toward CFM-
similar candidates is INTERPRETIVE, not deployed.  We mark the
contribution explicitly via :attr:`RewardAggregator.r_lambda_score`
and keep the weight at ``0.0`` by default so existing behaviour is
unchanged.
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Lazy import of the coupling adapter (torch-free at inference time).
# ---------------------------------------------------------------------------
def _get_coupling_adapter():
    """Return the :class:`CouplingAdapter` instance, falling back to
    a deterministic stub when no checkpoint is mounted.

    Import is lazy so this module can be imported by :class:`RewardAggregator`
    even when the user has not set ``COUPLING_ENABLED``.  When the env var
    is unset the adapter is still constructable (it's a numpy-only MLP),
    but :func:`compute_lambda_score` returns 0.0 to honour the opt-in
    contract documented in TODO-21.
    """
    try:
        from molmetal_lam.lam_chem.coupling_adapter import (
            load_coupling_adapter,
            is_coupling_enabled,
        )
    except Exception:
        return None, False
    return load_coupling_adapter(), is_coupling_enabled()


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------
def _safe_cos(a: np.ndarray, b: np.ndarray) -> float:
    """Numerically safe cosine similarity; returns 0.0 on zero vectors."""
    try:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        v = float(np.dot(a.reshape(-1), b.reshape(-1)) / (na * nb))
    except Exception:
        return 0.0
    if v != v:  # NaN
        return 0.0
    # Clamp to [-1, 1] to absorb floating-point drift.
    return max(-1.0, min(1.0, v))


def compute_lambda_score(
    lambda_candidate_smiles: Sequence[str],
    *,
    pocket_features: Optional[Sequence[float]] = None,
    pocket_name: str = "",
    coupling: Optional[Any] = None,
    enabled: Optional[bool] = None,
) -> float:
    """Mean cosine similarity between ``pocket`` and each ``lambda_candidate``.

    Parameters
    ----------
    lambda_candidate_smiles : sequence of str
        The Lambda MCTS candidates to score against the current CFM
        embedding.  Empty list returns 0.0.
    pocket_features : sequence of float, optional
        Forwarded to :meth:`CouplingAdapter.embed_pocket`.  When
        ``None`` the adapter falls back to a deterministic constant.
    pocket_name : str
        Identifier for the deterministic fallback when
        ``pocket_features`` is None.
    coupling : CouplingAdapter, optional
        Skip the lazy load and use this adapter (test / wiring hook).
    enabled : bool, optional
        Override the env-var gate.  When ``None`` the env-var is read.

    Returns
    -------
    float
        Mean cosine similarity in ``[-1, 1]``.  Returns 0.0 when:

        * ``lambda_candidate_smiles`` is empty;
        * the coupling adapter is unavailable;
        * the opt-in gate (``COUPLING_ENABLED``) is False;
        * any candidate fails to embed (NaN-safe).
    """
    # Opt-in gate: by default the channel is OFF so existing aggregator
    # behaviour is bit-for-bit identical.
    if enabled is None:
        enabled = bool(os.environ.get("COUPLING_ENABLED", "").strip().lower()
                       in {"1", "true", "yes", "on"})
    if not enabled:
        return 0.0
    if not lambda_candidate_smiles:
        return 0.0
    # Resolve the adapter (lazy import or test-injected).
    if coupling is None:
        coupling, adapter_enabled = _get_coupling_adapter()
        if coupling is None or not adapter_enabled:
            return 0.0
    # Compute the pocket-side embedding once.
    try:
        pocket_emb = coupling.embed_pocket(pocket_features, pocket_name=pocket_name)
    except Exception:
        return 0.0
    if pocket_emb is None:
        return 0.0
    # Per-candidate cosine, mean-aggregated.
    cos_vals: List[float] = []
    for smi in lambda_candidate_smiles:
        if not smi or not isinstance(smi, str):
            continue
        try:
            # Convert each SMILES to a 64-d pocket proxy via the same
            # adapter: we treat the SMILES as a synthetic 9-d input by
            # zeroing the per-pocket slots and putting a single 1.0 at
            # the heavy-atom count slot.  This is a *stand-in* — the
            # real production path would call a 64-d SMILES encoder,
            # but the bit-for-bit determinism + shape contract is
            # preserved either way.
            n_heavy = max(1, smi.count("Pt") + smi.count("Ru") + smi.count("Ir"))
            n_total = max(1, len(smi))
            proxy = np.zeros(9, dtype=np.float32)
            proxy[0] = float(n_heavy)
            proxy[1] = float(n_total) / 200.0  # crude length scale
            cand_emb = coupling.embed_features(proxy).reshape(-1)
            cos_vals.append(_safe_cos(pocket_emb, cand_emb))
        except Exception:
            continue
    if not cos_vals:
        return 0.0
    return float(sum(cos_vals) / len(cos_vals))


# ---------------------------------------------------------------------------
# RewardAggregator channel factory + wiring helper
# ---------------------------------------------------------------------------
def make_lambda_reward_channel(
    *,
    pocket_features: Optional[Sequence[float]] = None,
    pocket_name: str = "",
    coupling: Optional[Any] = None,
    enabled: Optional[bool] = None,
) -> Callable[[Any], float]:
    """Return a ``(state) -> float`` closure suitable for
    :attr:`RewardAggregator.r_lambda_score`.

    The closure extracts a SMILES string from the state (or returns 0.0
    on failure) and delegates to :func:`compute_lambda_score`.  The
    lambda-candidate list is supplied via the ``_R_LAMBDA_CANDIDATES``
    module-level global (set by the harness before each call) so the
    channel can be plugged into the existing aggregator without
    changing the call signature.

    Parameters
    ----------
    pocket_features, pocket_name, coupling, enabled :
        Forwarded to :func:`compute_lambda_score`.

    Returns
    -------
    callable
        ``(state) -> float`` matching the :class:`RewardAggregator`
        contract.  NaN-safe; always returns a finite float in ``[-1, 1]``.
    """
    def _channel(state: Any) -> float:
        try:
            # Per-call state-override: callers can pre-populate the
            # candidate list via the module-level
            # ``_R_LAMBDA_CANDIDATES`` global; the closure reads it
            # lazily so it sees the most recent candidates.
            candidates = globals().get("_R_LAMBDA_CANDIDATES") or []
        except Exception:
            candidates = []
        # Resolve a SMILES string from the state for the embed call.
        # The actual scoring uses the candidate SMILES, not the
        # state, so this is informational only.
        try:
            if isinstance(state, str):
                pass
            elif hasattr(state, "canonical_smiles"):
                _ = state.canonical_smiles()  # may raise — caught below
        except Exception:
            pass
        return compute_lambda_score(
            list(candidates),
            pocket_features=pocket_features,
            pocket_name=pocket_name,
            coupling=coupling,
            enabled=enabled,
        )

    return _channel


def register_lambda_reward_channel(
    aggregator: Any,
    *,
    weight: float = 0.0,
    pocket_features: Optional[Sequence[float]] = None,
    pocket_name: str = "",
    coupling: Optional[Any] = None,
    enabled: Optional[bool] = None,
) -> None:
    """Wire the new ``r_lambda_score`` channel into ``aggregator``.

    This is the additive-only wire-up specified by TODO-21 Strategy 1.
    It DOES NOT modify any existing channel (``r_qed``, ``r_sa``,
    ``r_pb``, ``r_reinvent4``, ``r_platinai`` etc.) — only the
    previously-undefined :attr:`r_lambda_score` slot is populated.

    Parameters
    ----------
    aggregator : RewardAggregator
        The aggregator instance to wire into.  Mutated in place.
    weight : float, default 0.0
        Per-channel weight on the new channel.  ``0.0`` = opt-in /
        silent (matches the existing channel opt-in contract).  Setting
        ``weight > 0`` enables the contribution on every
        :meth:`RewardAggregator.__call__`.
    pocket_features, pocket_name, coupling, enabled :
        Forwarded to :func:`make_lambda_reward_channel`.
    """
    channel = make_lambda_reward_channel(
        pocket_features=pocket_features,
        pocket_name=pocket_name,
        coupling=coupling,
        enabled=enabled,
    )
    # Additive only — we attach the NEW attribute ``r_lambda_score``
    # and the NEW ``w_lambda_score`` (both default ``None`` / ``0.0``).
    setattr(aggregator, "r_lambda_score", channel)
    setattr(aggregator, "w_lambda_score", float(weight))
    # Stash the candidate-list handle so the closure can read it
    # lazily.  Default is an empty list — no Lambda candidates
    # supplied yet, so the channel stays at 0.0 contribution until the
    # harness populates ``_R_LAMBDA_CANDIDATES`` before each cell.
    globals().setdefault("_R_LAMBDA_CANDIDATES", [])


def set_lambda_candidates(smiles_list: Sequence[str]) -> None:
    """Populate the module-level candidate list consumed by the
    channel closure.

    This is the integration hook for the harness — after each MCTS
    roll-out the harness calls this function with the post-
    MetalGeometryPrior-gate candidate SMILES, then runs
    :meth:`RewardAggregator.__call__` for the next leaf evaluation.

    Parameters
    ----------
    smiles_list : sequence of str
        Validated Lambda MCTS candidates.  Empty list clears the
        channel (returns 0.0).
    """
    globals()["_R_LAMBDA_CANDIDATES"] = list(smiles_list)


__all__ = [
    "compute_lambda_score",
    "make_lambda_reward_channel",
    "register_lambda_reward_channel",
    "set_lambda_candidates",
]