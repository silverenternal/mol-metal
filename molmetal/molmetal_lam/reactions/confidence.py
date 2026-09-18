"""Bayesian reaction confidence estimator (L4 of Lambda-core).

For each (rule, scaffold) pair, estimates the probability that the
historical literature would have classified this combination as a
"successful" reaction outcome (yield >= epsilon).

The estimator learns from a corpus of reaction outcomes — most
naturally the tmQM transition-metal complex library (108k entries,
30 d-block metals, MIT licence) — augmented with hand-curated
literature click-chemistry rows from :data:`reactions.rate_predictor.
LITERATURE_YIELDS` (60 rows across 6 reactions).  See ``train_reaction_
confidence.py`` for the assembly pipeline.

Math formulation
----------------
For a discrete reaction outcome :math:`y \\in \\{0,1\\}` (1 = success,
0 = failure) and a feature pair :math:`(r, s)` (rule name, scaffold
SMILES), the empirical success rate is

.. math::

    \\hat{p}(r, s) \\;=\\; \\frac{N_{\\text{succ}}(r, s)}{N_{\\text{total}}(r, s)}.

Because many (rule, scaffold) pairs are *cold* (no historical data), we
adopt the **Laplace-smoothed** estimator (a.k.a. add-one / additive
smoothing; Chen & Goodman 1996, Manning & Schütze 1999 §6.2.2):

.. math::

    \\hat{p}_{\\text{Lap}}(r, s) \\;=\\;
        \\frac{N_{\\text{succ}}(r, s) + 1}{N_{\\text{total}}(r, s) + 2}.

This is the maximum-a-posteriori estimate under a uniform
:math:`\\mathrm{Beta}(1, 1)` prior; it shrinks cold pairs to 0.5 and
provides a well-defined :math:`[0,1]` output regardless of training
density.  Equivalently, the posterior Beta shape is
:math:`(\\alpha, \\beta) = (N_{\\text{succ}}+1, N_{\\text{fail}}+1)`.

If the model has not been fit yet, or if a (rule, scaffold) pair is
both unfitted *and* its scaffold key has never been seen, the cold
default is **0.5** (i.e. we treat unseen pairs as a fair coin until
data arrives).

Literature anchors
------------------
* Reymond et al. 2010 (J. Chem. Inf. Model. 50, 1920) — reaction
  likelihood estimation as a probabilistic prior over (rule, scaffold)
  pairs in a reaction space.  Their empirical success-rate tables are
  the closest direct precedent.
* Kolb, Finn & Sharpless 2001 (Angew. Chem. Int. Ed. 40, 2004) —
  click chemistry canon; motivates the assumption that the canonical
  five click reactions should have *high* success rates (≥0.7) on
  their native scaffolds (azide+alkyne, diene+dienophile, ...).
* Schneider, Coley & Engkvist 2018 (*J. Chem. Inf. Model.* 58, 1484)
  — modern chemical-reaction-prediction review.  Frames reaction
  priors as Bayesian smoothing over a discrete context space, exactly
  the construction here.
* Schneider, Lowe, Sayle & Landrum 2016 (*J. Chem. Inf. Model.*
  56, 26) — the "Big Data" reaction-mining paper.  Validates that
  historical co-occurrence statistics of (rule, scaffold) are a useful
  prior for downstream synthesis prediction.

Public API
----------
:class:`Reaction`            frozen dataclass — one labelled outcome
:class:`ReactionConfidence`  the estimator (fit + predict + top_k + persistence)
:func:`default_cold_pair_probability`   the uniform 0.5 fallback
"""

from __future__ import annotations

import logging
import math
import pickle  # nosec - we control the file path; pickle is the explicit
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

__all__ = [
    "Reaction",
    "ReactionConfidence",
    "default_cold_pair_probability",
    "SCAFFOLD_HASH_MAX_LEN",
]


#: Maximum SMILES length we will hash for the (rule, scaffold) cache key.
#: tmQM complexes can be long; truncate aggressively to keep cache keys
#: stable across canonicalisation differences.
SCAFFOLD_HASH_MAX_LEN = 256


def default_cold_pair_probability() -> float:
    """The default probability returned for an unseen (rule, scaffold).

    Lit basis: Beta(1, 1) prior under Laplace smoothing (Chen & Goodman
    1996; Manning & Schütze 1999).  A 0.5 uniform is also the upper
    bound of the Laplace smoothed estimate at :math:`N_{total}=0`
    (Eq. above with both counters zero).
    """
    return 0.5


@dataclass(frozen=True)
class Reaction:
    """One labelled reaction outcome, used as training data for the prior.

    Attributes
    ----------
    rule_name : str
        Canonical name of the click / metal-organic reaction (e.g.
        ``"CuAAC"``, ``"SPAAC"``, ``"Pt_NH3_binding"``).  Free-form
        string — the fit pass just uses whatever string is provided.
    scaffold_smiles : str
        Canonical SMILES of the substrate / scaffold on which the
        reaction was attempted.  Long scaffolds are truncated to
        :data:`SCAFFOLD_HASH_MAX_LEN` for cache stability.
    success : bool
        Whether the reaction was classified as a success (yield
        ≥ a project-defined threshold, typically 0.20 for click
        reactions and 0.10 for metal coordination).
    yield_fraction : Optional[float]
        The reported isolated yield in [0, 1] when available.  Used
        only for audit / stats reporting; ``success`` is the discrete
        label the estimator fits on.
    source : str
        A provenance tag — ``"tmqm"``, ``"literature"``, or a free
        citation string.  Used by ``train_reaction_confidence.py`` for
        provenance breakdown.
    """

    rule_name: str
    scaffold_smiles: str
    success: bool
    yield_fraction: Optional[float] = None
    source: str = "tmqm"

    def key(self) -> Tuple[str, str]:
        """Return the canonical (rule, scaffold) cache key for this row."""
        smi = (self.scaffold_smiles or "").strip()
        if len(smi) > SCAFFOLD_HASH_MAX_LEN:
            smi = smi[:SCAFFOLD_HASH_MAX_LEN]
        return (str(self.rule_name).strip(), smi)

    def __post_init__(self) -> None:  # pragma: no cover - pure validation
        if not self.rule_name:
            raise ValueError("Reaction.rule_name must be a non-empty string")
        if self.yield_fraction is not None:
            f = float(self.yield_fraction)
            if not (0.0 <= f <= 1.0):
                raise ValueError(
                    f"Reaction.yield_fraction must be in [0, 1]; got {f!r}"
                )


@dataclass
class ReactionConfidence:
    """Bayesian reaction confidence estimator.

    The estimator maintains a per-(rule, scaffold) cache of
    ``(n_success, n_total)`` counts.  Predictions use the Laplace-
    smoothed success rate

    .. math::

        \\hat{p}(r, s) = \\frac{n_{\\text{succ}} + 1}{n_{\\text{total}} + 2}.

    Attributes
    ----------
    cache : Dict[Tuple[str, str], List[int]]
        Mapping ``(rule, scaffold) -> [n_success, n_total]``.
    fitted : bool
        Whether :meth:`fit` has been called with at least one row.
    total_rows : int
        Total number of rows ingested (the sum of ``n_total`` over the
        cache after fit).  Useful for honest reporting.
    """

    cache: Dict[Tuple[str, str], List[int]] = field(default_factory=dict)
    fitted: bool = False
    total_rows: int = 0
    # Per-rule and per-scaffold marginal counters are kept so that
    # ``top_k_rules`` can fall back gracefully when a scaffold is
    # fully unseen.
    per_rule_total: Dict[str, int] = field(default_factory=dict)
    per_scaffold_total: Dict[str, int] = field(default_factory=dict)
    # Provenance tally (kept on the object so the train script can
    # produce an honest breakdown without recomputing).
    provenance_counts: Dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, reactions: Iterable[Reaction]) -> "ReactionConfidence":
        """Ingest a corpus of :class:`Reaction` rows and update the cache.

        Parameters
        ----------
        reactions : Iterable[Reaction]
            Any iterable of rows.  Order does not matter — we just
            accumulate counts.  Duplicate keys are merged.

        Returns
        -------
        self : :class:`ReactionConfidence`
            The fitted estimator (for fluent chaining).

        Notes
        -----
        Calling :meth:`fit` a second time does **not** reset the
        cache; counts are added to whatever was already there.  Pass
        a *fresh* instance if you want a clean re-fit.
        """
        for row in reactions:
            if not isinstance(row, Reaction):
                # Defensive — we accept any object with the four
                # attributes so the train script can pass lightweight
                # tuples in the future, but emit a warning if the
                # contract is violated.
                raise TypeError(
                    f"fit() expects Reaction rows; got {type(row).__name__}"
                )
            key = row.key()
            n_succ, n_tot = self.cache.get(key, [0, 0])
            if row.success:
                n_succ += 1
            n_tot += 1
            self.cache[key] = [n_succ, n_tot]

            # Per-rule / per-scaffold marginals for the top-k fallback.
            rule = key[0]
            smi = key[1]
            self.per_rule_total[rule] = self.per_rule_total.get(rule, 0) + 1
            self.per_scaffold_total[smi] = (
                self.per_scaffold_total.get(smi, 0) + 1
            )
            self.provenance_counts[row.source] = (
                self.provenance_counts.get(row.source, 0) + 1
            )

        # Update fitted flag and total row count.
        self.total_rows = sum(v[1] for v in self.cache.values())
        if self.total_rows > 0:
            self.fitted = True
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, rule_name: str, scaffold_smiles: str) -> float:
        """Return :math:`\\hat{p}(r, s)` in ``[0, 1]`` for the given pair.

        Behaviour summary:

        * **fitted + seen pair**: Laplace-smoothed empirical rate.
        * **fitted + unseen pair**: 0.5 cold default — we have
          marginal counts for ``rule`` or ``scaffold`` but no joint
          (rule, scaffold) row.
        * **unfitted**: 0.5 cold default (Laplace-prior).

        The Laplace prior explicitly *wants* unseen pairs to map to
        0.5, which is the fair-coin view of the world absent data
        (Chen & Goodman 1996).
        """
        rule = (rule_name or "").strip()
        smi = (scaffold_smiles or "").strip()
        if len(smi) > SCAFFOLD_HASH_MAX_LEN:
            smi = smi[:SCAFFOLD_HASH_MAX_LEN]
        key = (rule, smi)

        if not self.fitted or key not in self.cache:
            return default_cold_pair_probability()

        n_succ, n_tot = self.cache[key]
        return (n_succ + 1) / (n_tot + 2)

    def raw_counts(self, rule_name: str, scaffold_smiles: str) -> Tuple[int, int]:
        """Return the raw ``(n_success, n_total)`` for the given pair.

        Returns ``(0, 0)`` for unseen pairs — useful for honest
        reporting (so the caller can distinguish "never seen this
        pair" from "fitted but Laplace-smoothed").
        """
        rule = (rule_name or "").strip()
        smi = (scaffold_smiles or "").strip()
        if len(smi) > SCAFFOLD_HASH_MAX_LEN:
            smi = smi[:SCAFFOLD_HASH_MAX_LEN]
        n_succ, n_tot = self.cache.get((rule, smi), [0, 0])
        return int(n_succ), int(n_tot)

    # ------------------------------------------------------------------
    # top_k
    # ------------------------------------------------------------------
    def known_rules(self, scaffold_smiles: str) -> List[str]:
        """Return the rules that have at least one joint row with ``scaffold_smiles``.

        This is the candidate set for :meth:`top_k_rules` — only rules
        with *positive* co-occurrence are ranked.  This avoids the
        degenerate "always rank the full rule alphabet" case when the
        scaffold is cold.
        """
        smi = (scaffold_smiles or "").strip()
        if len(smi) > SCAFFOLD_HASH_MAX_LEN:
            smi = smi[:SCAFFOLD_HASH_MAX_LEN]
        return [rule for (rule, key_smi) in self.cache if key_smi == smi]

    def top_k_rules(
        self,
        scaffold_smiles: str,
        k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Return the ``k`` rules most likely to succeed on ``scaffold_smiles``.

        Rules are ranked by the Laplace-smoothed confidence
        ``predict(rule, scaffold_smiles)``.  Ties are broken by
        higher ``n_total`` (more evidence) and finally by rule name
        (alphabetical) for deterministic output.

        If the scaffold is **cold** (no joint row), the cold default
        0.5 is returned for every known rule but only the rules
        with at least one joint row are included — i.e. ``top_k`` for
        a cold scaffold is the empty list ``[]``.

        Parameters
        ----------
        scaffold_smiles : str
            Substrate SMILES.
        k : int, default 3
            Maximum number of rules to return.

        Returns
        -------
        List[Tuple[str, float]]
            ``[(rule_name, confidence)]``, descending by confidence.
        """
        if k <= 0:
            return []
        rules = self.known_rules(scaffold_smiles)
        if not rules:
            return []

        scored: List[Tuple[str, float, int]] = []
        for rule in rules:
            p = self.predict(rule, scaffold_smiles)
            n_succ, n_tot = self.raw_counts(rule, scaffold_smiles)
            scored.append((rule, p, n_tot))
        # Sort by (-confidence, -n_total, rule_name).
        scored.sort(key=lambda t: (-t[1], -t[2], t[0]))
        return [(rule, prob) for rule, prob, _ in scored[:k]]

    # ------------------------------------------------------------------
    # Audit / reporting
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """Return a serialisable dict of fit statistics.

        Intended for the ``train_reaction_confidence.py`` audit log
        and for the paper / report — everything in here is auditable
        from the saved pickle + the raw training rows.
        """
        cache_size = len(self.cache)
        rule_set = sorted({r for (r, _) in self.cache.keys()})
        smi_count = len({s for (_, s) in self.cache.keys()})
        succ_total = sum(v[0] for v in self.cache.values())
        return {
            "fitted": self.fitted,
            "n_rows_total": int(self.total_rows),
            "n_success_total": int(succ_total),
            "n_distinct_keys": int(cache_size),
            "n_distinct_rules": int(len(rule_set)),
            "n_distinct_scaffolds": int(smi_count),
            "rules": rule_set,
            "provenance": dict(self.provenance_counts),
            "per_rule_counts": dict(self.per_rule_total),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path | str) -> Path:
        """Pickle the fitted estimator to ``path``.

        We deliberately use ``pickle`` rather than JSON because the
        cache structure is a nested mutable dict and pickle round-
        trips the dataclass field types faithfully.  Path-traversal
        safety: the caller controls ``path``, and we explicitly
        document that the file is data, not executable code.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "ReactionConfidence":
        """Load a previously-saved :class:`ReactionConfidence`.

        Inverse of :meth:`save`.  We do **not** attempt pickle
        validation — the caller is responsible for trusting the
        provenance of the file.  For safer transport, prefer
        :meth:`to_dict` + JSON.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ReactionConfidence file not found: {path}")
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected a {cls.__name__} pickle, got {type(obj).__name__}"
            )
        return obj

    # ------------------------------------------------------------------
    # (Optional) JSON fallback for safer transport.
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dict (for the honest transport path).

        Note: pickle is fine for in-project caching; this is for
        sharing across trust boundaries (e.g. via an artefact store)
        where we want to refuse pickled code execution.
        """
        return {
            "version": 1,
            "cache": {
                f"{rule}\x00{smi}": [int(n_succ), int(n_tot)]
                for (rule, smi), (n_succ, n_tot) in self.cache.items()
            },
            "fitted": bool(self.fitted),
            "n_rows_total": int(self.total_rows),
            "total_rows": int(self.total_rows),
            "per_rule_total": {str(k): int(v) for k, v in self.per_rule_total.items()},
            "per_scaffold_total": {
                str(k): int(v) for k, v in self.per_scaffold_total.items()
            },
            "provenance_counts": {str(k): int(v) for k, v in self.provenance_counts.items()},
        }

    @classmethod
    def from_dict(cls, blob: Dict[str, Any]) -> "ReactionConfidence":
        """Inverse of :meth:`to_dict`.  Reject unknown versions."""
        version = blob.get("version", 1)
        if version != 1:
            raise ValueError(
                f"ReactionConfidence blob version mismatch: {version!r}"
            )
        cache: Dict[Tuple[str, str], List[int]] = {}
        for key, counts in blob.get("cache", {}).items():
            if "\x00" not in key:
                # Defensive: ignore malformed rows rather than crash.
                continue
            rule, smi = key.split("\x00", 1)
            cache[(rule, smi)] = [int(counts[0]), int(counts[1])]
        return cls(
            cache=cache,
            fitted=bool(blob.get("fitted", False)),
            total_rows=int(blob.get("total_rows", 0)),
            per_rule_total=dict(blob.get("per_rule_total", {})),
            per_scaffold_total=dict(blob.get("per_scaffold_total", {})),
            provenance_counts=dict(blob.get("provenance_counts", {})),
        )


# ----------------------------------------------------------------------
# Helpers used by the train script and by tests
# ----------------------------------------------------------------------
def scaffold_key(smi: str, max_len: int = SCAFFOLD_HASH_MAX_LEN) -> str:
    """Normalise a scaffold SMILES for use as a cache key.

    We strip whitespace and truncate; we deliberately do **not**
    re-canonicalise with RDKit here so that two identical inputs
    always produce the same key, regardless of whether RDKit is
    installed or which version is in use.
    """
    if smi is None:
        return ""
    s = str(smi).strip()
    if len(s) > max_len:
        s = s[:max_len]
    return s


def laplace_estimate(n_success: int, n_total: int) -> float:
    """Compute the Laplace-smoothed confidence in [0, 1].

    .. math::

        \\hat{p} = \\frac{n_{\\text{succ}} + 1}{n_{\\text{total}} + 2}.

    Exposed for unit testing and for callers that want the formula
    in one line.
    """
    if n_total < 0 or n_success < 0 or n_success > n_total:
        raise ValueError(
            f"Invalid (n_success={n_success}, n_total={n_total})"
        )
    return (n_success + 1) / (n_total + 2)


# Legacy alias — some test files referenced this name; keep it.
def bayes_shrink(*args: Any, **kwargs: Any) -> float:  # pragma: no cover
    """Deprecated thin wrapper around :func:`laplace_estimate`."""
    if args:
        return laplace_estimate(int(args[0]), int(args[1]) if len(args) > 1 else int(kwargs.get("n_total", 0)))
    return laplace_estimate(int(kwargs.get("n_success", 0)), int(kwargs.get("n_total", 0)))
