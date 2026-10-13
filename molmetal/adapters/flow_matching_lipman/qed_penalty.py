"""Wf-C4 — QED-aware sampling penalty for CFM (BLK-C4).

============================================================
What this module is
============================================================
The SA-penalty analogue of ``sa_penalty.py``, for the QED half of the
m4 Success-Rate sub-criterion ``SA < 2.8 AND QED > 0.5``
(``metrics/by_metric/m4_success_rate.json``; MEASURED sub-rate
0.0022 = 1/449 on the metallo-drug cohort).

Because QED is a "higher is better" score in ``[0, 1]`` while SA is a
"lower is better" score in ``[1, 10]``, the penalty is the mirrored
hinge::

    penalty(qed) = max(0, target_qed - qed) * qed_weight

Non-negative, monotone decreasing in QED, and exactly ``0`` for any
molecule already at or above ``target_qed`` — so good molecules are
not biased downward, which is the whole point of the SA convention
(Ertl & Schuffenhauer 2009 §3).

============================================================
Honest framing — READ BEFORE USING (party-twenty-spirit §3.5)
============================================================
This is a REPLACEMENT for a false premise: ``todo.json`` BLK-C4
claimed a ``qed_penalty.py`` SCAFFOLD existed and only needed wiring.
It does not exist anywhere in the repo. What exists is
``molmetal/molmetal_lam/sbdd_env/qed_scorer.py`` (a QED *scorer*,
which silently returns ``0.0`` for an unparseable SMILES — see
"Failure policy" below for why that is not reusable as-is) and
``molmetal/scripts/wf323f_m4_sa_qed_subcriterion.py``.

This module is NOT wired into any trainer. See the trainer's own
2026-10-04 audit note (``train_cfm_yuelbond_wasserstein.py``, in the
``loss = (...)`` comment block): the SA penalty term was **deleted**
from the loss sum because ``sascorer`` is a fragment-frequency lookup
and therefore *structurally incapable of carrying a gradient* — it was
a constant. **QED has exactly the same property**: it is an RDKit
descriptor computed from a parsed molecule, non-differentiable by
construction. So this module is a valid SAMPLING-TIME REWEIGHTING and
REPORTING quantity, and a valid ablation knob for score-space
search — it is NOT a differentiable loss term, and putting
``weight * mean(penalty)`` into a velocity loss would re-create the
constant the audit deleted.

Failure policy (never a silent zero)
------------------------------------
An unparseable SMILES is assigned the MAXIMUM attainable penalty
(``target_qed * qed_weight``, i.e. as if QED were exactly 0.0) — the
same conservative direction as ``sa_penalty``, which maps NaN to
``+inf``. It is additionally **counted**: every entry point returns a
``n_fallback`` / ``fallback_indices`` channel, so a caller can never
average a fallback into a molecule mean without seeing it. This is
deliberately different from ``qed_scorer.qed_from_smiles``, which
returns ``0.0`` for an unparseable SMILES with no counter and
therefore cannot distinguish "RDKit refused to parse this" from "this
is a real molecule with a genuinely terrible QED of 0.0".

MEASURED fallback rates on real data (2026-10-09, this wave):
* ``molmetal/data/metallo_drugs_500_train.csv`` — 0/449 unparseable
  (0.0000). These are curated/docked complexes, not raw generations.
* Real CFM decode, ``metrics/by_metric/cfm_decode_real.json`` — 5/818
  samples failed (0.0061) at the RDKit-strict decode step.
So the "generated SMILES are frequently unparseable" premise is
OVERSTATED for these two sources; the fallback path is still built and
tested because a 0.6% rate is exactly the magnitude that hides inside
a 818-sample mean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union

# QED band (Bickerton & Schreiber 2016; RDKit ``Chem.QED.qed``):
#   0.0 .. 0.3 — poorly drug-like
#   0.3 .. 0.7 — drug-like
#   0.7 .. 1.0 — unusually drug-like
#
# ``target_qed`` defaults to the canonical 0.5 from the m4 sub-criterion
# (SA<2.8 AND QED>0.5, Wf-139 P6 / Wf-315A), mirroring how
# ``sa_penalty.DEFAULT_TARGET_SA`` defaults to that criterion's 2.8.
DEFAULT_TARGET_QED: float = 0.5
DEFAULT_QED_WEIGHT: float = 0.3


# ---------------------------------------------------------------------------
# RDKit availability (resolved once, lazily imported)
# ---------------------------------------------------------------------------


def _rdkit_qed():
    """Return RDKit's ``QED`` module, or ``None`` when unavailable.

    Import is deferred so that importing this module never requires
    RDKit (same defensive posture as ``sa_penalty.compute_sa_score``).
    """
    try:
        from rdkit import Chem, RDLogger  # type: ignore
        from rdkit.Chem import QED  # type: ignore

        RDLogger.DisableLog("rdApp.*")
        return Chem, QED
    except Exception:  # pragma: no cover - depends on runtime env
        return None


_RDKIT_PAIR = _rdkit_qed()


def rdkit_available() -> bool:
    """True if RDKit could be imported for QED scoring."""
    return _RDKIT_PAIR is not None


# ---------------------------------------------------------------------------
# Per-molecule QED score
# ---------------------------------------------------------------------------


def compute_qed_score(mol) -> float:
    """Compute the Bickerton QED for one RDKit ``Mol``.

    Returns ``float('nan')`` when the molecule is ``None`` or QED
    cannot be computed — the same failure signal
    :func:`sa_penalty.compute_sa_score` uses, so both penalties can be
    fed through the same NaN-aware aggregation path.

    Returns
    -------
    float
        QED in ``[0, 1]``; NaN on failure.
    """
    if mol is None or _RDKIT_PAIR is None:
        return float("nan")
    _, QED = _RDKIT_PAIR
    try:
        return float(QED.qed(mol))
    except Exception:
        return float("nan")


def parse_and_score(smiles: str) -> float:
    """Parse *smiles* and return its QED, or NaN if it does not parse.

    NaN here means "no molecule" — never "a molecule with QED 0.0".
    Keeping those two states distinct is the entire point of this
    function; :func:`qed_scorer.qed_from_smiles` collapses them.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return float("nan")
    if _RDKIT_PAIR is None:
        return float("nan")
    Chem, _ = _RDKIT_PAIR
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
    except Exception:
        return float("nan")
    return compute_qed_score(mol)


# ---------------------------------------------------------------------------
# Penalty primitives (scalar QED values)
# ---------------------------------------------------------------------------


def qed_aware_sampling_penalty(
    samples: Union[Sequence[float], "Any"],
    target_qed: float = DEFAULT_TARGET_QED,
    qed_weight: float = DEFAULT_QED_WEIGHT,
) -> "Any":
    """Per-sample QED penalty for sampling-time reweighting.

    Computes::

        penalty[i] = max(0, target_qed - qed_i) * qed_weight

    This is the exact mirror of
    :func:`sa_penalty.sa_aware_sampling_penalty`, which computes
    ``max(0, sa_i - target_sa) * sa_weight``. The direction flips
    because QED is better-when-higher; the hinge shape, the clamping
    at zero, and the linear weight do not.

    Parameters
    ----------
    samples : sequence of float
        Pre-computed QED values. ``NaN`` means "no molecule" and is
        mapped to the MAXIMUM penalty (``target_qed * qed_weight``),
        the conservative direction — same intent as SA's NaN -> +inf,
        but finite so that it does not poison a downstream mean with
        an infinity. Callers should still report the NaN count; use
        :func:`qed_penalty_batch` to get it for free.
    target_qed : float, default 0.5
        Drug-likeness threshold (m4 sub-criterion / Wf-139 P6).
    qed_weight : float, default 0.3
        Linear penalty weight; matches the SA module's default so the
        two terms are on a comparable scale.

    Returns
    -------
    ``numpy.ndarray`` of float, shape ``(len(samples),)``
        Non-negative penalty; exactly ``0`` for ``qed >= target_qed``.
    """
    import numpy as np

    qed_arr = np.asarray(samples, dtype=np.float64)
    target = float(target_qed)
    weight = float(qed_weight)
    # NaN -> 0.0 QED -> maximum attainable penalty.
    qed_arr = np.where(np.isnan(qed_arr), 0.0, qed_arr)
    penalty = np.maximum(0.0, target - qed_arr) * weight
    return penalty


# ---------------------------------------------------------------------------
# SMILES-in / penalty-out entry point, with the fallback counter
# ---------------------------------------------------------------------------


@dataclass
class QEDPenaltyBatch:
    """Result of scoring a batch of SMILES, including fallback counts.

    Attributes
    ----------
    smiles : list of str
        The input, echoed back so the caller can join on it.
    qed : numpy.ndarray
        Per-molecule QED; ``NaN`` where the SMILES did not parse.
    penalties : numpy.ndarray
        Per-molecule penalty, same order and length as ``smiles``.
    n_fallback : int
        **How many molecules fell back** because RDKit could not parse
        them (or RDKit is absent). Never hidden — this is the channel
        the caller reports.
    fallback_indices : list of int
        Positions in the input of the fallback molecules.
    target_qed, qed_weight : float
        Parameters actually used.
    """

    smiles: list
    qed: Any
    penalties: Any
    n_fallback: int
    fallback_indices: list
    target_qed: float
    qed_weight: float

    @property
    def n(self) -> int:
        return len(self.smiles)

    @property
    def mean_penalty(self) -> float:
        """Mean penalty over ALL inputs (fallbacks included)."""
        import numpy as np

        if self.penalties.size == 0:
            return float("nan")
        return float(np.mean(self.penalties))

    @property
    def mean_qed_parsed_only(self) -> float:
        """Mean QED over successfully parsed molecules ONLY.

        Falls back to NaN when every molecule failed to parse. This
        is the number to log: averaging QED over a batch where half
        the entries are fallbacks would understate the cohort.
        """
        import numpy as np

        parsed = self.qed[~np.isnan(self.qed)]
        if parsed.size == 0:
            return float("nan")
        return float(np.mean(parsed))

    @property
    def fallback_rate(self) -> float:
        """Fraction of inputs that fell back; NaN for an empty batch."""
        if self.n == 0:
            return float("nan")
        return float(self.n_fallback) / float(self.n)

    def to_dict(self) -> dict:
        return {
            "n": int(self.n),
            "n_fallback": int(self.n_fallback),
            "fallback_rate": float(self.fallback_rate),
            "fallback_indices": [int(i) for i in self.fallback_indices],
            "mean_penalty": float(self.mean_penalty),
            "mean_qed_parsed_only": float(self.mean_qed_parsed_only),
            "target_qed": float(self.target_qed),
            "qed_weight": float(self.qed_weight),
        }


def qed_penalty_batch(
    smiles: Sequence[str],
    target_qed: float = DEFAULT_TARGET_QED,
    qed_weight: float = DEFAULT_QED_WEIGHT,
) -> QEDPenaltyBatch:
    """Score a batch of SMILES, returning one penalty per input in order.

    This is the SMILES-in / penalty-out entry point. Unlike
    :func:`qed_aware_sampling_penalty` (which takes pre-computed QED
    values), this one owns the parse, so it can report how many inputs
    failed to parse.

    An unparseable SMILES receives the maximum attainable penalty
    ``target_qed * qed_weight`` and its index is recorded in
    ``fallback_indices`` / counted in ``n_fallback``. It is never given
    a silent ``0.0``.
    """
    import numpy as np

    smis = list(smiles)
    qeds = np.array(
        [parse_and_score(s) for s in smis], dtype=np.float64
    )
    fallback_idx = [int(i) for i, q in enumerate(qeds) if math.isnan(q)]
    penalties = qed_aware_sampling_penalty(
        qeds, target_qed=target_qed, qed_weight=qed_weight
    )
    return QEDPenaltyBatch(
        smiles=smis,
        qed=qeds,
        penalties=penalties,
        n_fallback=len(fallback_idx),
        fallback_indices=fallback_idx,
        target_qed=float(target_qed),
        qed_weight=float(qed_weight),
    )


def qed_penalty_single(
    smiles: str,
    target_qed: float = DEFAULT_TARGET_QED,
    qed_weight: float = DEFAULT_QED_WEIGHT,
) -> float:
    """Penalty for one SMILES. Returns ``0.0`` if RDKit is unavailable.

    NOTE: this scalar helper deliberately does NOT raise on a bad
    SMILES — it returns the fallback penalty, exactly as the batch
    function does. The scalar form cannot report the fallback count,
    which is why :func:`qed_penalty_batch` is the preferred entry
    point and this one exists only for symmetry with
    ``sa_aware_sampling_penalty``.
    """
    batch = qed_penalty_batch([smiles], target_qed=target_qed, qed_weight=qed_weight)
    return float(batch.penalties[0]) if batch.penalties.size else 0.0


# ---------------------------------------------------------------------------
# Aggregator helper (mirrors sa_penalty.SAPenaltyAggregate)
# ---------------------------------------------------------------------------


@dataclass
class QEDPenaltyAggregate:
    """Aggregate QED penalty stats across a cohort, fallbacks surfaced.

    Mirrors :class:`sa_penalty.SAPenaltyAggregate` but additionally
    carries the fallback accounting, which is the whole reason this
    module is not just a call into ``qed_scorer``.

    Attributes
    ----------
    n : int
        Number of molecules submitted.
    n_fallback : int
        Number that RDKit could not parse.
    mean_qed : float
        Mean QED over parsed molecules only; NaN if none parsed.
    mean_penalty : float
        Mean penalty over ALL molecules (fallbacks at max penalty).
    fraction_below_target : float
        Fraction of PARSED molecules with QED < target_qed.
    target_qed : float
        Threshold used.
    """

    n: int
    n_fallback: int
    mean_qed: float
    mean_penalty: float
    fraction_below_target: float
    target_qed: float

    def to_dict(self) -> dict:
        return {
            "n": int(self.n),
            "n_fallback": int(self.n_fallback),
            "fallback_rate": (
                float(self.n_fallback) / float(self.n) if self.n else float("nan")
            ),
            "mean_qed": float(self.mean_qed),
            "mean_penalty": float(self.mean_penalty),
            "fraction_below_target": float(self.fraction_below_target),
            "target_qed": float(self.target_qed),
        }


def aggregate_qed_penalty(
    smiles: Sequence[str],
    target_qed: float = DEFAULT_TARGET_QED,
    qed_weight: float = DEFAULT_QED_WEIGHT,
) -> QEDPenaltyAggregate:
    """Aggregate per-cohort QED penalty statistics from raw SMILES.

    Pure-function sibling of :func:`sa_penalty.aggregate_sa_penalty`,
    but taking SMILES rather than pre-computed scores so it can count
    parse failures.
    """
    import numpy as np

    batch = qed_penalty_batch(smiles, target_qed=target_qed, qed_weight=qed_weight)
    parsed = batch.qed[~np.isnan(batch.qed)]
    if parsed.size == 0:
        return QEDPenaltyAggregate(
            n=batch.n,
            n_fallback=batch.n_fallback,
            mean_qed=float("nan"),
            mean_penalty=batch.mean_penalty,
            fraction_below_target=float("nan"),
            target_qed=float(target_qed),
        )
    return QEDPenaltyAggregate(
        n=batch.n,
        n_fallback=batch.n_fallback,
        mean_qed=float(np.mean(parsed)),
        mean_penalty=batch.mean_penalty,
        fraction_below_target=float(np.mean(parsed < float(target_qed))),
        target_qed=float(target_qed),
    )


__all__ = [
    "DEFAULT_TARGET_QED",
    "DEFAULT_QED_WEIGHT",
    "rdkit_available",
    "compute_qed_score",
    "parse_and_score",
    "qed_aware_sampling_penalty",
    "qed_penalty_batch",
    "qed_penalty_single",
    "QEDPenaltyBatch",
    "QEDPenaltyAggregate",
    "aggregate_qed_penalty",
]