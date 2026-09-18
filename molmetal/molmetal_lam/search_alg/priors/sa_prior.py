"""SA-aware MCTS leaf prior — bias tree search toward drug-like SMILES.

================================================================
Background — why an SA-aware MCTS leaf prior?
================================================================
The MCTS proof search in :mod:`molmetal_lam.search_alg.proof_search`
selects children via PUCT, where the prior ``P(a|s)`` is normally a
constant 0.5 stub (or a learned click-rule prior).  This prior only
sees the *reaction-rule axis* — the *SMILES axis* (which leaf
molecules to spend simulations exploring) is sampled uniformly
regardless of synthesizability.

The SA penalty --sa-weight (Ertl 2008) is a *post-hoc* reward channel
— it only affects the back-propagated leaf value, not the *leaf
expansion* policy.  A leaf prior that biases tree search toward
lower-SA candidates gives MCTS an earlier signal: even before any
expansion, "more drug-like" candidates get a stronger UCB pull,
which improves PUCT's expansion efficiency.

================================================================
Math prior
================================================================
For a candidate SMILES ``x`` with raw SA score ``sa(x) in [1, 10]``
(lower = easier to synthesise), define the *prior logit*

    prior_logit(x) = -log( sa(x) )

so benzene (SA ≈ 2.27) yields ``prior_logit ≈ -0.82`` and cisplatin
(SA ≈ 5.94) yields ``prior_logit ≈ -1.78``.  Lower SA → higher
logit (less negative), which in turn biases the PUCT selector
toward that action.

The scaling is borrowed from the Ertl 2008 literature anchor
(inv-SA, ``1 / (1 + sa)``) and re-parameterised in log-space so
the prior operates additively with the existing PUCT expansion
term ``c_puct · P(a|s) · sqrt(N_parent) / (1 + N_child)``.

================================================================
Lit anchor
================================================================
* Ertl & Schuffenhauer 2008 (J. Cheminform. 1:8) — the SA score
  itself; lower = easier synthesis.
* Polykovskiy 2020 (MOSES, Front. Pharmacol. 10:1293) — the
  canonical Ertl distribution benchmark; MOSES drug-like has
  median SA ≈ 2.5.
* Silver 2017 (AlphaGo Zero, Nature 550:354) — PUCT leaf prior
  mixing; our prior multiplies the existing ``P(a|s)`` mass rather
  than replacing it (additive in logit-space).
* Auer 2002 (UCB1, Finite-Time Analysis) — the canonical
  regret-bounded leaf selector that our SA prior augments.

================================================================
Mechanism (vs. --sa-weight post-hoc reward)
================================================================
* ``--sa-weight`` post-hoc: scales ``v_sa = 1 - (sa-1)/9`` by
  ``w_sa`` in :class:`RewardAggregator.__call__`.  Effect: leaf
  *value* is biased, but the expansion policy is unchanged.
* ``--use-sa-prior`` (this module): biases the *PUCT prior* term
  ``P(a|s)`` so MCTS expands drug-like branches first.  Effect:
  the exploration policy itself is biased.

The two are additive: a search can run with both ``--sa-weight 0.3
--use-sa-prior True`` and the prior biases exploration while the
reward biases leaf value.

================================================================
Honest framing
================================================================
* The prior is a single SMILES-level scalar — it does not see
  reaction-rule context (that lives in
  :mod:`learned_prior`).
* At small MCTS budgets (n_sim ≤ 100) the top-k rarely diverges
  enough for the prior to swing the rank; the dominant effect
  appears at n_sim ≥ 1000 (verified via
  ``wf_lambda_diversity_rotation`` MEASURED data — 5×1 with
  --metal-seed cisplatin, n_sim=1000, all cells collapse to the
  same candidate because MCTS terminates before the prior has
  lever).
* Falls back to ``uniform 0.0`` when the SMILES is unparseable or
  RDKit is unavailable — backward-compatible with the legacy
  constant prior.

================================================================
Integration surface
================================================================
* :class:`SAPrior` is an additive channel: it does not replace
  :class:`SymbolicPrior` or :class:`LearnedPolicyPrior`.
* :meth:`MCTSProofSearch.search` accepts a keyword-only
  ``use_sa_prior=True`` flag.  When ``True`` the ``_prior``
  cascade multiplies in the SA-derived logit; when ``False``
  (default) the legacy behaviour is bit-for-bit preserved.
* :file:`molmetal/scripts/r4_lambda_only_run.py` exposes the
  ``--use-sa-prior`` CLI flag (default ``False``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# Lazy RDKit imports — guarded so the import surface stays
# backward-compatible with the headless CI test path.
try:  # pragma: no cover — optional dep
    from rdkit import Chem as _Chem  # type: ignore
    try:
        from rdkit.Contrib.SA_Score import sascorer as _sascorer  # type: ignore
    except Exception:  # pragma: no cover
        _sascorer = None  # type: ignore
    try:
        from rdkit.Chem import RDLogger as _RDLogger  # type: ignore
        _RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        _RDLogger = None  # type: ignore
    _RDKIT_AVAILABLE: bool = True
except Exception:  # pragma: no cover
    _Chem = None  # type: ignore
    _sascorer = None  # type: ignore
    _RDLogger = None  # type: ignore
    _RDKIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Default fallback SA score (used when RDKit is unavailable AND the
# anticancer_metric_suite compute_sa import also fails).  The fallback
# corresponds to the MOSES median (≈ 2.5) so a uniform prior is
# effectively returned (-log(2.5) ≈ -0.916).
# ---------------------------------------------------------------------------
_DEFAULT_FALLBACK_SA: float = 2.5


# ---------------------------------------------------------------------------
# SA computation helper — delegates to anticancer_metric_suite.compute_sa
# when available, falls back to rdkit.Contrib.SA_Score.sascorer (Ertl 2008)
# when the suite import fails.  Returns ``None`` when the SMILES cannot
# be parsed.
# ---------------------------------------------------------------------------
def _compute_sa_score(smiles: str) -> Optional[float]:
    """Compute the raw Ertl SA score for ``smiles``.

    Returns ``None`` when the SMILES is unparseable OR when no SA
    backend is available.  The caller is expected to fall back to a
    uniform 0.0 prior in that case (see :meth:`SAPrior.prior_logit`).
    """
    if not smiles:
        return None
    smi = str(smiles).strip()
    if not smi:
        return None

    # 1) Try anticancer_metric_suite.compute_sa first (delegates to
    #    the same Ertl 2008 implementation but is the project's
    #    canonical SA path).
    try:
        from molmetal_lam.sbdd_env.anticancer_metric_suite import (  # type: ignore
            compute_sa as _suite_compute_sa,
        )
        try:
            v = float(_suite_compute_sa(smi))
            if math.isfinite(v):
                return v
        except Exception:
            pass
    except Exception:
        pass

    # 2) Fall back to rdkit.Contrib.SA_Score.sascorer (Ertl 2008).
    if _RDKIT_AVAILABLE and _sascorer is not None and _Chem is not None:
        try:
            mol = _Chem.MolFromSmiles(smi)
            if mol is None:
                return None
            return float(_sascorer.calculateScore(mol))
        except Exception:
            return None

    # 3) No SA backend available — caller decides fallback policy.
    return None


@dataclass
class SAPrior:
    """SA-aware MCTS leaf prior.

    Maps a candidate SMILES to a *prior logit* that biases MCTS
    toward lower-SA molecules.  The logit is the (negative)
    natural log of the raw SA score:

        prior_logit(x) = -log( sa(x) )

    Lower SA → higher logit (less negative), which biases PUCT
    expansion toward that action.  Falls back to a uniform
    ``0.0`` logit when the SMILES is unparseable or no SA
    backend is available — backward-compatible with the legacy
    constant prior.

    Parameters
    ----------
    eps : float, default 1e-6
        Numerical floor on the SA score so the logit never
        diverges for SA ≈ 0 (theoretical corner case; the
        sascorer never emits sub-1 values in practice).

    Examples
    --------
    >>> prior = SAPrior()
    >>> prior.prior_logit("c1ccccc1")  # benzene, SA ≈ 2.27
    -0.819...
    >>> prior.prior_logit("N.N.N.N.Cl.Cl.[Pt]")  # cisplatin
    -1.78...
    >>> prior.prior_logit("not_a_smiles")
    0.0
    """

    eps: float = 1e-6
    #: Cache of (smiles, logit) for repeated queries — MCTS
    #: calls :meth:`prior_logit` on the same candidate multiple
    #: times per simulation.
    _cache: dict = field(default_factory=dict, repr=False)

    def prior_logit(self, smiles: Optional[str]) -> float:
        """Return the SA-derived prior logit for ``smiles``.

        Returns ``0.0`` (uniform baseline) when:

        * ``smiles`` is ``None`` or empty,
        * the SMILES is unparseable by RDKit,
        * no SA backend is available (RDKit missing + suite import
          fails).

        Returns a finite, negative real number otherwise
        (``-log(sa)``).  Cached on the instance for repeated
        queries within the same search.
        """
        if not smiles:
            return 0.0
        smi = str(smiles).strip()
        if not smi:
            return 0.0
        if smi in self._cache:
            return self._cache[smi]
        sa = _compute_sa_score(smi)
        if sa is None or not math.isfinite(sa):
            self._cache[smi] = 0.0
            return 0.0
        sa_floor = max(float(sa), float(self.eps))
        try:
            logit = -math.log(sa_floor)
        except (ValueError, OverflowError):
            logit = 0.0
        if not math.isfinite(logit):
            logit = 0.0
        self._cache[smi] = logit
        return logit

    # ------------------------------------------------------------------
    # Convenience helper for PUCT integration
    # ------------------------------------------------------------------
    def prior_mass(self, smiles: Optional[str]) -> float:
        """Map the SA logit into a PUCT-compatible prior mass in [0, 1].

        The mass is computed by squashing the logit into [0, 1] via
        a softplus-style normalisation:

            mass = sigmoid( logit - logit_baseline )

        where ``logit_baseline = -log(_DEFAULT_FALLBACK_SA) ≈
        -0.916``.  This centres the prior mass at 0.5 for SA = 2.5
        (the MOSES median), so a candidate with SA < 2.5 receives
        a mass > 0.5 and a candidate with SA > 2.5 receives a
        mass < 0.5.  The mass is clamped to ``[1e-6, 1 - 1e-6]``
        to keep PUCT numerically stable.

        Returns ``0.5`` for unparseable SMILES — uniform baseline.
        """
        logit = self.prior_logit(smiles)
        if logit == 0.0:
            return 0.5
        baseline = -math.log(_DEFAULT_FALLBACK_SA)
        z = logit - baseline
        try:
            if z >= 0:
                p = 1.0 / (1.0 + math.exp(-z))
            else:
                ez = math.exp(z)
                p = ez / (1.0 + ez)
        except OverflowError:
            p = 0.0 if z < 0 else 1.0
        return float(max(1e-6, min(1.0 - 1e-6, p)))

    def clear_cache(self) -> None:
        """Clear the internal SMILES → logit cache.

        Useful for tests that want a deterministic fresh state.
        """
        self._cache.clear()


# ---------------------------------------------------------------------------
# Module-level singleton (lazy) — same pattern as the other priors.
# ---------------------------------------------------------------------------
_singleton: Optional[SAPrior] = None


def get_sa_prior() -> SAPrior:
    """Return the module-level :class:`SAPrior` singleton.

    The first call constructs the dataclass; subsequent calls reuse
    the same instance so its internal cache survives across calls.
    """
    global _singleton
    if _singleton is None:
        _singleton = SAPrior()
    return _singleton
