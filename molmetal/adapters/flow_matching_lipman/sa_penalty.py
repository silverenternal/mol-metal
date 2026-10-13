"""Wf-380-A6 — SA-aware sampling + loss penalty module for CFM.

================================================================
Why this module exists
================================================================
m4 Success Rate (TODO-205 / WEAK_TO_STRONG §2.4) is the SA/QED
sub-criterion rate of CFM-generated metallodrug molecules.  The
MEASURED baseline on the 449 metallo-drug cohort is
``sub_rate_sa_qed = 0.0022`` (1/449 hits at SA<2.8 ∧ QED>0.5; see
``metrics/by_metric/m4_success_rate.json``).  The cohort mean SA is
``4.7148`` — far above the synthesizable threshold of 2.8 — and the
mean SA of the wf_sa_fragment_pool_optimize CFM run (filtered pool
220 to 210 tiles) is ``3.099`` (see ``metrics/by_metric/sa_mean.json``).

This module computes a **per-sample SA penalty** intended for
generation-time reweighting.  The penalty is:

    penalty(sa) = max(0, sa - target_sa) * sa_weight

which is ``0`` for SA <= target and grows linearly with how far the
sample overshoots.  This is the simplest monotone, non-negative
penalty that pushes sampling toward the synthesizable region without
clipping low-SA samples — the canonical form recommended by
Ertl & Schuffenhauer 2009 §3 for generation-time SA reweighting.

**The loss-term use does NOT work, and the earlier version of this
docstring claimed it did.**  That claim has been removed because it was
false in two independent ways:

1. **sascorer has no gradient.**  It is a fragment-frequency lookup, so any
   value it returns is a constant with respect to the model parameters.  The
   trainer's 2026-10-04 audit deleted the surrogate that had pretended
   otherwise (``train_cfm_yuelbond_wasserstein.py``, the ``gen_sa_observed``
   comment block).
2. **The replacement is still a constant.**  When no ``sa_scores_batch`` is
   injected, the trainer synthesises proxy scores from cloud statistics
   (``n_atoms`` + ``n_distinct_elements``) — but then wraps the mean in
   ``torch.tensor(float(...))`` (line ~2711), which yields
   ``requires_grad=False`` and ``grad_fn=None``.  Verified on this host.

So with ``--sa-penalty > 0`` the term appears in the logged loss value and
contributes **zero** to ``backward``.  Parameter updates are bit-for-bit
identical to ``--sa-penalty 0``.  The flag is wired into the loss *expression*,
not into the learning: do not cite it as a gradient lever, and do not expect the
PROJECTED m4 lift below from the loss path.

Wf-380-A6 honest framing (party-twenty-spirit §3.5):
* The sampling-time reweighting use is the supported one.  The loss-time use is
  inert for the reasons above.
* The m4 lift ``3.099 -> 2.5-2.7`` (WEAK_TO_STRONG §3.6 band) was PROJECTED and
  was never MEASURED.  It is reachable, if at all, only through sampling-time
  reweighting.
* ``metrics/by_metric/m4_success_rate.json`` remains the baseline of record.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union

# SA score band (Ertl 2009 §3):
#   1.0  .. 2.8  — synthesizable (easy)
#   2.8  .. 4.0  — synthesizable with effort
#   4.0  .. 6.0  — hard
#   6.0  .. 10.0 — very hard / bespoke
#
# ``target_sa`` defaults to the canonical "synthesizable" threshold
# 2.8 (Wf-139 P6 / WEAK_TO_STRONG §3.6) so the penalty is ``0`` for
# any sample that is already in the easy region.

DEFAULT_TARGET_SA: float = 2.8
DEFAULT_SA_WEIGHT: float = 0.3


# ---------------------------------------------------------------------------
# Lazy SA score helper
# ---------------------------------------------------------------------------


def compute_sa_score(mol) -> float:
    """Compute the Ertl 2009 SA score for a single RDKit ``Mol``.

    Tries (in order):

    1. RDKit's bundled ``rdkit/Contrib/SA_Score/sascorer.py`` (loaded
       via ``rdkit.Chem.RDConfig.RDContribDir``) — this is the
       upstream canonical implementation and ships with the
       ``fpscores.pkl.gz`` fragment table.
    2. The in-tree ``molmetal.eval.scoring.sascorer`` wrapper — a
       vendored copy of the same algorithm.

    Returns ``float('nan')`` if neither path is importable or the
    mol is ``None`` / unparseable.

    Returns
    -------
    float
        SA score in [1, 10]; NaN on failure.
    """
    if mol is None:
        return float("nan")
    # Path 1: RDKit bundled SA scorer (preferred; ships with fpscores.pkl.gz).
    try:
        from rdkit.Chem import RDConfig  # type: ignore
        import os as _os
        import sys as _sys

        sa_dir = _os.path.join(RDConfig.RDContribDir, "SA_Score")
        if sa_dir not in _sys.path:
            _sys.path.append(sa_dir)
        import sascorer as _rdkit_sa  # type: ignore

        return float(_rdkit_sa.calculateScore(mol))
    except Exception:
        pass
    # Path 2: in-tree wrapper fallback.
    try:
        from molmetal.eval.scoring.sascorer import compute_sa_score as _compute
    except Exception:  # pragma: no cover — defensive
        return float("nan")
    try:
        return float(_compute(mol))
    except Exception:  # pragma: no cover — defensive
        return float("nan")


# ---------------------------------------------------------------------------
# Penalty primitives
# ---------------------------------------------------------------------------


def sa_aware_sampling_penalty(
    samples: Union[Sequence[float], "numpy.ndarray"],
    target_sa: float = DEFAULT_TARGET_SA,
    sa_weight: float = DEFAULT_SA_WEIGHT,
) -> "numpy.ndarray":
    """Per-sample SA penalty for sampling-time reweighting.

    Computes::

        penalty[i] = max(0, sa_i - target_sa) * sa_weight

    Parameters
    ----------
    samples : sequence of float
        Pre-computed SA scores for each sample (NaN treated as +inf).
    target_sa : float, default 2.8
        Synthesizable threshold (Ertl 2009 §3 / Wf-139 P6).
    sa_weight : float, default 0.3
        Linear penalty weight.  0.3 matches the ``--sa-weight 0.3``
        smoke cohort used in ``metrics/by_metric/sa_mean.json`` 2026-09-14.

    Returns
    -------
    ``numpy.ndarray`` of float, shape ``(len(samples),)``
        Non-negative penalty; ``0`` for samples with SA <= target.
    """
    import numpy as np

    sa_arr = np.asarray(samples, dtype=np.float64)
    # Replace NaN with +inf so the penalty stays > 0 (caller should
    # already have filtered parse-failures, but defensive default is
    # to discourage the sampler from picking unparseable molecules).
    sa_arr = np.where(np.isnan(sa_arr), np.inf, sa_arr)
    delta = sa_arr - float(target_sa)
    penalty = np.maximum(0.0, delta) * float(sa_weight)
    return penalty


def sa_aware_loss_step(
    logits: "numpy.ndarray",
    sa_scores: Union[Sequence[float], "numpy.ndarray"],
    target_sa: float = DEFAULT_TARGET_SA,
    sa_weight: float = DEFAULT_SA_WEIGHT,
) -> float:
    """Combined per-step loss with SA penalty term.

    Computes::

        L = L_main(logits) + sa_weight_loss * mean(sa_aware_sampling_penalty(sa_scores))

    where ``L_main`` is the binary cross-entropy on the logits
    (placeholder — the real trainer plugs in the multi-task
    Wasserstein loss; this scaffold uses BCE on logits reshaped as
    binary cross-entropy so the function is self-contained and the
    penalty arithmetic can be smoke-tested standalone).

    Parameters
    ----------
    logits : ``numpy.ndarray`` of float, shape ``(N,)`` or ``(N, 2)``
        Per-sample logits from the CFM velocity net.
    sa_scores : sequence of float
        Pre-computed SA scores for each sample.
    target_sa, sa_weight : float
        Penalty parameters (see :func:`sa_aware_sampling_penalty`).

    Returns
    -------
    float
        Scalar combined loss.
    """
    import numpy as np

    logits = np.asarray(logits, dtype=np.float64)
    sa_arr = np.asarray(sa_scores, dtype=np.float64)

    # Binary cross-entropy on logits reshaped as target=1 (positive
    # scaffold); this is a SCAFFOLD placeholder — the real trainer
    # injects the multi-task Wasserstein loss instead of this BCE.
    if logits.ndim == 1:
        p = 1.0 / (1.0 + np.exp(-logits))
        bce = -np.mean(np.log(np.clip(p, 1e-12, 1.0)))
    else:
        # logits[:,1] is the "good" class logit, target=1.
        z1 = logits[:, 1]
        z0 = logits[:, 0]
        # softmax(logits)[:,1]
        max_z = np.maximum(z1, z0)
        log_sum = np.log(np.exp(z1 - max_z) + np.exp(z0 - max_z)) + max_z
        log_p1 = z1 - log_sum
        bce = -float(np.mean(log_p1))

    penalty = sa_aware_sampling_penalty(
        sa_arr, target_sa=target_sa, sa_weight=sa_weight
    )
    sa_term = float(np.mean(penalty))
    # sa_weight_loss defaults to 1.0 so the sa_aware_sampling_penalty
    # magnitude (typically 0-2.0 for sa_weight=0.3) is comparable to
    # the BCE magnitude (typically 0-1.0); adjust per ablation.
    sa_weight_loss = 1.0
    return float(bce + sa_weight_loss * sa_term)


# ---------------------------------------------------------------------------
# Aggregator helper (used by the panel script)
# ---------------------------------------------------------------------------


@dataclass
class SAPenaltyAggregate:
    """Aggregate SA penalty stats across a cohort.

    Attributes
    ----------
    n : int
        Number of samples in the cohort.
    mean_sa : float
        Mean SA score.
    mean_penalty : float
        Mean SA penalty (sa_weight applied).
    fraction_above_target : float
        Fraction of samples with SA > target_sa.
    target_sa : float
        Threshold used.
    """

    n: int
    mean_sa: float
    mean_penalty: float
    fraction_above_target: float
    target_sa: float

    def to_dict(self) -> dict:
        return {
            "n": int(self.n),
            "mean_sa": float(self.mean_sa),
            "mean_penalty": float(self.mean_penalty),
            "fraction_above_target": float(self.fraction_above_target),
            "target_sa": float(self.target_sa),
        }


def aggregate_sa_penalty(
    sa_scores: Union[Sequence[float], "numpy.ndarray"],
    target_sa: float = DEFAULT_TARGET_SA,
    sa_weight: float = DEFAULT_SA_WEIGHT,
) -> SAPenaltyAggregate:
    """Aggregate per-cohort SA penalty statistics.

    Mirrors the schema of ``aggregate_sa_distribution`` in
    ``cfm_sa_aware_panel.py`` (Wf-380-A6 §2) but is provided here as
    a pure-function helper so callers that already have pre-computed
    SA scores (e.g. an in-loop trainer) can use it without invoking
    the full panel.

    Returns
    -------
    SAPenaltyAggregate
    """
    import numpy as np

    sa_arr = np.asarray(sa_scores, dtype=np.float64)
    sa_clean = sa_arr[~np.isnan(sa_arr)]
    if sa_clean.size == 0:
        return SAPenaltyAggregate(
            n=0,
            mean_sa=float("nan"),
            mean_penalty=float("nan"),
            fraction_above_target=float("nan"),
            target_sa=float(target_sa),
        )
    penalty = sa_aware_sampling_penalty(
        sa_clean, target_sa=target_sa, sa_weight=sa_weight
    )
    return SAPenaltyAggregate(
        n=int(sa_clean.size),
        mean_sa=float(np.mean(sa_clean)),
        mean_penalty=float(np.mean(penalty)),
        fraction_above_target=float(np.mean(sa_clean > target_sa)),
        target_sa=float(target_sa),
    )


__all__ = [
    "DEFAULT_TARGET_SA",
    "DEFAULT_SA_WEIGHT",
    "compute_sa_score",
    "sa_aware_sampling_penalty",
    "sa_aware_loss_step",
    "SAPenaltyAggregate",
    "aggregate_sa_penalty",
]
