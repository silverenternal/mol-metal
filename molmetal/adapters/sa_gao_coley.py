"""Gao-Coley PARAMETERS, asymmetric-exp FORMULA, normalised SA score.

================================================================
Source
================================================================
Gao, W.; Coley, C. W. "The Synthesizability of Molecules Proposed by
Generative Models." *J. Chem. Inf. Model.* **2020**, 60 (12), 5714-5723.
DOI 10.1021/acs.jcim.0c00174 (preprint arXiv:2002.07007).

================================================================
Formula and parameters (verbatim from the brief + corroborating
sources)
================================================================
Two normalisations of the Ertl & Schuffenhauer (2009) raw score are in
common use and they are NOT comparable:

* Linear hack (used repo-wide today):
      SA_norm_linear = 1 - SA_raw / 10
  Maps the Ertl range [1, 10] to [0, 0.9]. 0 = hard, 1 = easy.

* This module (Gao-Coley PARAMETERS, asymmetric-exp FORMULA):
      SA_norm = exp( -(SA_raw - mu) / sigma )
  ASYMMETRIC EXPONENTIAL DECAY -- not a Gaussian, not a half-Gaussian.
  Monotonically decreasing in SA_raw; values below mu cap at exp(0)=1.0
  because the exponent is bounded above by 0 (delta is negative so
  -delta/sigma is positive). The function name and prior docstring
  called this a "half-Gaussian" -- that was WRONG. Corrected 2026-10-12
  by adversarial review (cfm_sa_gao_coley_reference.json::corrections_2026_10_12_evening).
  Parameters (corroborated from MOMO-master/losses.py::normalize_sa and
  Brown 2019 GuacaMol implementation references; provenance to the
  Gao-Coley paper itself for the specific values is UNVERIFIED):
      mu    = 2.230044
      sigma = 0.6526308
  These are the population mean / standard deviation of Ertl SA across
  the Enamine REAL / ZINC15-clean-3 lead-like library used by Gao &
  Coley as the "synthesizable" reference. They are NOT derived from
  tmQM or any organometallic corpus.

  AXIS CONVENTION: 0 = hard to synthesise, 1 = easy. Same as the linear
  hack. Higher is better. Different from the raw Ertl scale where LOWER
  is better.

  CANONICAL MOMO FORM DIFFERENCE: MOMO-master/losses.py::normalize_sa
  uses ``exp(-0.5 * ((max(sa, mu) - mu) / sigma)^2)`` -- a SYMMETRIC
  Gaussian clamped at mu. At sa_raw=6.44 (our generated mean) the
  canonical form evaluates to 9.205e-10, this module evaluates to
  1.58e-03 -- a factor of 1.7e6. The two are NOT numerically
  interchangeable; any side-by-side with literature using MOMO-form
  values must say so explicitly. This module is kept as-is for the
  repo's published numbers (cfm_sa_gao_coley_reference.json) and the
  misnomer is recorded; rerunning through the canonical form is a
  downstream decision and would change every reported value.

================================================================
Why this matters for the CFM record
================================================================
* This repo's m4 SA has always been quoted on 1 - SA_raw / 10.
* The LigandDiff 0.68-0.80 SA_norm band (Jin & Merz, JCTC 2024,
  DOI 10.1021/acs.jctc.4c00232) is reported 0=easy / 1=easy convention
  BUT THE NORMALISATION USED IN THAT PAPER IS NOT STATED. They do not
  cite Gao & Coley, do not cite GuacaMol, and the SI does not quote a
  formula. Their Table 2 raw numbers (0.68 for 11-40 atom PPR_100
  ligands drawn from 100 Pt/Pd/Ru CSD complexes) are at the high end
  of achievable ligand SA, well above any raw value our 39-heavy-atom
  cohort produces; on either normalisation scale that gap shrinks
  dramatically and may even flip.

* Two normalisations of the same raw SA are NOT comparable unless
  computed on the same scale. Any side-by-side of "LigandDiff 0.68"
  with "CFM 0.36" must say which scale, and until LigandDiff's
  implementation is read, NOT VERIFIED.

================================================================
API
================================================================
``asymmetric_exponential_sa(sa_raw)`` -- the corrected, accurate name.
  The formula this module ACTUALLY computes: exp(-(sa - mu) / sigma),
  asymmetric exponential decay, no clamp.
``gao_coley_sa_norm(sa_raw)`` -- kept as a backward-compat alias for
  ``asymmetric_exponential_sa``; its prior docstring's "half-Gaussian"
  wording was wrong and is corrected above.
``gao_coley_sa_norm_array(sa_raws)`` -- vectorised wrapper.
``GAO_COLEY_MU``, ``GAO_COLEY_SIGMA`` -- module constants; provenance
  to Gao-Coley 2020 is UNVERIFIED for the specific values; the
  constants match MOMO-master/losses.py, NOT the indexed Gao-Coley paper.

Verification value: at sa_raw=mu=2.230044 the function returns 1.0
exactly.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Union

# Population mean / sd of Ertl SA across the Enamine REAL / ZINC15
# clean lead-like library used by Gao & Coley as the "synthesizable"
# reference. NOT derived from tmQM or any organometallic corpus.
# Quoted to 7 dp to match the values in published implementations.
# Provenance to the indexed Gao-Coley 2020 paper is UNVERIFIED for the
# specific values; corroborated against MOMO-master/losses.py.
GAO_COLEY_MU: float = 2.230044
GAO_COLEY_SIGMA: float = 0.6526308


def asymmetric_exponential_sa(sa_raw: float) -> float:
    """Normalised SA, asymmetric exponential decay of the Ertl raw score.

    Formula: ``sa_norm = exp(-(sa_raw - mu) / sigma)``
    with mu=2.230044, sigma=0.6526308.

    NOTE: the function NAME is the only correct part of the prior
    docstring. The shape is a pure asymmetric exponential decay in
    SA_raw, NOT a Gaussian (which would be exp(-0.5*((x-mu)/sigma)^2))
    and NOT a half-Gaussian. Values below mu cap at 1.0 because the
    exponent is bounded above by 0 (negative delta times negative
    1/sigma).

    Parameters
    ----------
    sa_raw : float
        Ertl & Schuffenhauer SA score in [1, 10]. Values above mu
        decay roughly exponentially toward 0; values below mu cap at
        1.0.

    Returns
    -------
    float
        Normalised SA in (0, 1]. Higher is more synthetically
        accessible.
    """
    import math

    delta = float(sa_raw) - GAO_COLEY_MU
    return math.exp(-delta / GAO_COLEY_SIGMA)


# Backward-compat alias. The prior name "gao_coley_sa_norm" implied
# the function implements the canonical Gao-Coley form, which it does
# NOT (the canonical form is a clamped symmetric Gaussian -- see
# module docstring). Kept as an alias so existing callers and the
# published JSON do not break.
gao_coley_sa_norm = asymmetric_exponential_sa


def asymmetric_exponential_sa_array(sa_raws: Iterable[float]) -> List[float]:
    """Vectorised wrapper: map an iterable of raw SA scores."""
    return [asymmetric_exponential_sa(x) for x in sa_raws]


# Backward-compat alias.
gao_coley_sa_norm_array = asymmetric_exponential_sa_array


__all__ = [
    "GAO_COLEY_MU",
    "GAO_COLEY_SIGMA",
    "asymmetric_exponential_sa",
    "asymmetric_exponential_sa_array",
    "gao_coley_sa_norm",  # backward-compat alias
    "gao_coley_sa_norm_array",  # backward-compat alias
]
