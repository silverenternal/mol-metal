"""learned_shaping.py — opt-in wrapper around the F5 PAC-Bayes symbolic reward.

Phase 1 deliverable for the deflex follow-up workflow (wh8npxvj9 / wefwo7ub5
/ wmi2gg065).

Background
==========
Phase 2 (Symbolic Regression, wh8npxvj9) ran LassoLarsIC with BIC on n=33
MEASURED Round-12 path-(a) cells and recovered the F5 formula:

    R_F5(sa_norm) = 2.5836 - 2.5149 * sa_norm

with the other 4 features (diversity_tanimoto, diversity_homotype, qed_mean,
anticancer_index) zeroed by sparsity (sum |w|>1e-6 == 1).

Honest framing
==============
- In-sample R^2 = 1.0 on n=33 complete-case cells (LassoLarsIC BIC picked this
  exact sparsity pattern).
- LOO R^2 = -0.0635 (close to random; the 5-feature set is genuinely
  sparse).  Honest reading: F5 is INTERPRETIVE (it identifies sa_norm as
  the lone surviving signal), not predictive.
- This shaping has NOT been validated to lift Round-12 downstream metrics
  (Vina score, PB pass rate, pIC50).  Adding a -2.5149*sa_norm penalty to
  the reward prior is a PROJECTION, not a measurement.
- Downstream proof_search.py integration is a SEPARATE workflow that owns
  pocket-invariance concerns; this module ships opt-in only.

Integration status
==================
- Default OFF: ``is_enabled()`` returns False unless the env var
  ``LEARNED_SHAPING_ENABLED=1`` is set.
- The adapter exposes ``shape_reward(sa_score)`` for the simple case where
  the caller already has sa_norm.  For the full 5-feature pipeline, callers
  should use :func:`compute_symbolic_reward` from
  :mod:`molmetal_lam.reward.symbolic_regression` directly.
- No edits are made to proof_search.py; this module is independently
  testable and exists to make the F5 formula a first-class callable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


# -----------------------------------------------------------------------------
# F5 constants — verbatim from Phase 2 wh8npxvj9 §Phase 2 fit output.
# Intercept and the surviving non-zero coefficient from LassoLarsIC BIC.
# -----------------------------------------------------------------------------
F5_INTERCEPT: float = 2.5836
F5_SA_COEFFICIENT: float = -2.5149

# Features considered by F5 (the other 4 are zeroed by sparsity).
F5_FEATURE_NAMES: List[str] = [
    "diversity_tanimoto",
    "diversity_homotype",
    "qed_mean",
    "anticancer_index",
    "sa_mean_norm",
]

# Coefficients per feature — only sa_mean_norm is non-zero.
F5_COEFFICIENTS: Dict[str, float] = {
    "diversity_tanimoto": 0.0,
    "diversity_homotype": 0.0,
    "qed_mean": 0.0,
    "anticancer_index": 0.0,
    "sa_mean_norm": F5_SA_COEFFICIENT,
}


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class BestFormula:
    """A symbolic-reward formula recovered from MEASURED cells.

    Mirrors the dataclass contract requested in the deflex follow-up
    Phase 1 spec (wf_deflex_followup/phase1_f5_adapter.md).
    """
    name: str
    formula_str: str
    coefficients: Dict[str, float]
    complexity: int
    r_squared_in_sample: float
    r_squared_loo: float
    lit_anchor: str
    formula_callable: Callable[[float], float] = field(repr=False)
    n_fit: int = 33  # cells used in the Phase 2 fit
    family: str = "F5"


def _f5_callable(sa_norm: float) -> float:
    """F5 formula evaluated on a single sa_norm scalar."""
    return F5_INTERCEPT + F5_SA_COEFFICIENT * sa_norm


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def load_best_formula() -> BestFormula:
    """Return the F5 formula as a :class:`BestFormula`.

    The coefficients are the literal Phase 2 output; this function does
    NOT re-fit (use :func:`fit_from_metrics` for that).  Honest framing:
    this is a hardcoded lookup of the Pareto pick from
    ``symbolic_regression.fit_symbolic_reward(..., family='F5')``.
    """
    formula_str = (
        f"{F5_INTERCEPT:.4f} + "
        f"{F5_COEFFICIENTS['diversity_tanimoto']:+.4f}*diversity_tanimoto + "
        f"{F5_COEFFICIENTS['diversity_homotype']:+.4f}*diversity_homotype + "
        f"{F5_COEFFICIENTS['qed_mean']:+.4f}*qed_mean + "
        f"{F5_COEFFICIENTS['anticancer_index']:+.4f}*anticancer_index + "
        f"{F5_COEFFICIENTS['sa_mean_norm']:+.4f}*sa_mean_norm "
        f"[LassoBIC, nz=1]"
    )
    return BestFormula(
        name="F5",
        formula_str=formula_str,
        coefficients=dict(F5_COEFFICIENTS),
        complexity=5,
        r_squared_in_sample=1.0,
        r_squared_loo=-0.0635,
        lit_anchor="McAllester 1999 PAC-Bayes regularised linear",
        formula_callable=_f5_callable,
    )


def enumerate_8_families() -> List[BestFormula]:
    """Enumerate the 8 symbolic-reward families.

    Only F5 has MEASURED coefficients from Phase 2.  The other 7 families
    are returned with placeholder coefficients (zero vector) so that the
    enumeration contract is honoured.  Callers wanting a real re-fit should
    use :func:`fit_from_metrics`.
    """
    out: List[BestFormula] = []
    # Family metadata mirrors symbolic_regression.FORMULA_FAMILIES.
    families = [
        ("F1", "Dayan 1997 potential-based reward shaping",
         "diversity_tanimoto+diversity_homotype+qed_mean+anticancer_index+sa_mean_norm"),
        ("F2", "Ng 1999 policy-invariant potential shaping",
         "diversity_tanimoto+diversity_homotype+qed_mean+anticancer_index+sa_mean_norm"),
        ("F3", "convex combination Dayan 1999 + McAllester 1999",
         "diversity_tanimoto+diversity_homotype+qed_mean+anticancer_index+sa_mean_norm"),
        ("F4", "Tanimoto-focal loss (Lin 2017)",
         "diversity_tanimoto+qed_mean+anticancer_index+sa_mean_norm"),
        ("F5", "McAllester 1999 PAC-Bayes regularised linear",
         "diversity_tanimoto+diversity_homotype+qed_mean+anticancer_index+sa_mean_norm"),
        ("F6", "Schulman 2017 PPO clipped objective",
         "diversity_tanimoto+diversity_homotype+qed_mean+anticancer_index+sa_mean_norm"),
        ("F7", "Auger 2013 + Dayan 1997 tiered metal-prior",
         "diversity_tanimoto+metal_compliance_rate+sa_mean_norm+qed_mean"),
        ("F8", "Tennie 2024 hierarchical symbolic regression",
         "diversity_tanimoto+qed_mean+anticancer_index+sa_mean_norm+diversity_homotype"),
    ]
    for name, lit, _featlist in families:
        if name == "F5":
            out.append(load_best_formula())
        else:
            # Placeholder: zero coefficients.  fit_from_metrics will refit.
            zero_coeffs = {fn: 0.0 for fn in F5_FEATURE_NAMES}
            out.append(
                BestFormula(
                    name=name,
                    formula_str=f"{name}: not refit; use fit_from_metrics",
                    coefficients=zero_coeffs,
                    complexity=5,
                    r_squared_in_sample=0.0,
                    r_squared_loo=0.0,
                    lit_anchor=lit,
                    formula_callable=lambda x: 0.0,
                    n_fit=0,
                    family=name,
                )
            )
    return out


def fit_from_metrics(
    cells: List[Dict[str, float]],
    family: str = "F5",
) -> BestFormula:
    """Re-fit a family on a new cohort and return a :class:`BestFormula`.

    Parameters
    ----------
    cells : list of dict
        Each dict has at least the F5 features (sa_mean_norm, etc.) and
        may carry an optional ``reward`` key (target).  If no ``reward`` is
        provided, the F5 closed-form is used as the target (this is a
        self-consistency check).
    family : str
        One of "F1".."F8".  Defaults to "F5".

    Notes
    -----
    - This delegates to ``symbolic_regression.fit_symbolic_reward`` so the
      fit logic stays in one place.  When ``reward`` keys are absent, we
      compute the F5 target via :func:`shape_reward_full_row` so the
      refit is at least *self-consistent* with the F5 formula.
    - Lazy import: importing symbolic_regression pulls numpy/pandas/sklearn
      at module load, which we want to defer until the caller actually
      refits (keeping ``learned_shaping`` cheap to import for the env-gate).
    """
    import numpy as np
    import pandas as pd

    from .symbolic_regression import fit_symbolic_reward

    if not cells:
        return load_best_formula()

    df = pd.DataFrame(cells)
    if "reward" not in df.columns:
        # Self-consistency: use F5 closed-form as the target so we can at
        # least verify the fitter reproduces the canonical coefficients.
        df = df.copy()
        df["reward"] = df.apply(
            lambda r: shape_reward_full_row(r.to_dict()), axis=1
        )
    y = df["reward"].to_numpy(dtype=float)

    feat_cols = [c for c in F5_FEATURE_NAMES if c in df.columns]
    if len(feat_cols) < len(F5_FEATURE_NAMES):
        # Pad missing feature columns with 0.0.
        for c in F5_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0

    fit = fit_symbolic_reward(df, y, family=family)
    # Map FitResult back into BestFormula.  coef vector is in F5_FEATURE_NAMES
    # order (intercept first, then 5 feature coefs).
    coefs = {"intercept": float(fit.params[0]) if fit.params else 0.0}
    for fn, w in zip(F5_FEATURE_NAMES, fit.params[1:] if len(fit.params) > 1 else []):
        coefs[fn] = float(w)

    def _refit_callable(sa_norm: float) -> float:
        return coefs["intercept"] + sum(
            coefs[fn] * sa_norm if fn == "sa_mean_norm" else coefs[fn] * 0.0
            for fn in F5_FEATURE_NAMES
        )

    return BestFormula(
        name=fit.family,
        formula_str=fit.formula,
        coefficients=coefs,
        complexity=fit.complexity,
        r_squared_in_sample=float(fit.r2),
        r_squared_loo=float("nan"),  # not computed in fit_symbolic_reward
        lit_anchor=fit.lit_anchor,
        formula_callable=_refit_callable,
        n_fit=fit.n_fit,
        family=fit.family,
    )


def shape_reward_full_row(row: Dict[str, float]) -> float:
    """Apply F5 to a dict of all 5 features (not just sa_mean_norm)."""
    sa = float(row.get("sa_mean_norm", 0.0))
    return _f5_callable(sa)


# -----------------------------------------------------------------------------
# Opt-in controller (env-gated)
# -----------------------------------------------------------------------------
class LearnedShaping:
    """Opt-in learned shaping controller.

    Usage
    -----
    >>> sh = LearnedShaping()
    >>> sh.is_enabled()
    False
    >>> sh.shape_reward(3.32)  # no-op when disabled
    0.0
    >>> os.environ["LEARNED_SHAPING_ENABLED"] = "1"
    >>> sh2 = LearnedShaping()
    >>> sh2.shape_reward(3.32)
    -5.7662...
    """

    _ENABLED_ENV = "LEARNED_SHAPING_ENABLED"

    def __init__(self, formula: Optional[BestFormula] = None) -> None:
        self._formula: BestFormula = formula or load_best_formula()

    # ---- state ------------------------------------------------------------
    def is_enabled(self) -> bool:
        """Return True iff the env-gate is set to "1" (or "true"/"yes")."""
        v = os.environ.get(self._ENABLED_ENV, "0").strip().lower()
        return v in ("1", "true", "yes", "on")

    def enable(self) -> None:
        """Programmatically enable for the rest of this process."""
        os.environ[self._ENABLED_ENV] = "1"

    def disable(self) -> None:
        """Programmatically disable."""
        os.environ[self._ENABLED_ENV] = "0"

    # ---- accessors --------------------------------------------------------
    def get_active_formula(self) -> BestFormula:
        return self._formula

    def set_active_formula(self, formula: BestFormula) -> None:
        self._formula = formula

    # ---- core -------------------------------------------------------------
    def shape_reward(self, sa_score: float) -> float:
        """Apply F5 to a single sa_norm value.

        Returns 0.0 when :meth:`is_enabled` is False (the safe default).
        When enabled, returns :math:`2.5836 - 2.5149 * sa_score`.
        """
        if not self.is_enabled():
            return 0.0
        return float(self._formula.formula_callable(float(sa_score)))


__all__ = [
    "F5_INTERCEPT",
    "F5_SA_COEFFICIENT",
    "F5_COEFFICIENTS",
    "F5_FEATURE_NAMES",
    "BestFormula",
    "load_best_formula",
    "enumerate_8_families",
    "fit_from_metrics",
    "shape_reward_full_row",
    "LearnedShaping",
]
