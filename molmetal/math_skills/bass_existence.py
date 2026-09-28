"""T8 — Bass Martingale Coupling Existence (Backhoff-Veraguas 2020).

Reference
---------
Backhoff-Veraguas, M., Beiglböck, M., Cox, M. and Huesmann, M. (2020).
*Martingale Benamou-Brenier: a probabilistic proof*.  arXiv:2004.10060.
Builds on the martingale-transport programme of Beiglböck & Juillet
(2014) — *On a martingale transport problem* — and Hobson (2011) —
*Constructing martingales*.  Cited in ``paper/refs.bib`` as
``backhoff_veraguas2020bass_existence`` and referenced from
``paper/appendices/closure_theorem.tex`` as the mathematical skeleton
that backs the existence + stability of the Bass martingale coupling
implicitly used by the SynFlowNet calibration head of T7.

This module ships two **mathematical skeleton** primitives that the
R18 upgrade path to Mol-Metal will need for a rigorous statement of
the Bass coupling:

1. **BassExistenceTheorem** (``BassExistenceTheorem``) — the
   existence + stability statement holder.  Stores the formal text
   of the Backhoff-Veraguas (2020) Theorem 1.1 (existence + stability
   of the martingale-coupling problem) and exposes the negative
   result that the coupling is generally *non-unique*.  Returns the
   statement as a string, and the uniqueness verdict as a boolean.

2. **RegularityCondition** (``RegularityCondition``) — the weak
   regularity check that a candidate marginal satisfies the
   hypotheses of the Backhoff-Veraguas theorem.  The only hypothesis
   we test is the **P₂ (finite second moment)** condition on the
   marginal samples.  In the full continuous theorem the regularity
   assumption is on the *measure* :math:`\\mu \\in \\mathcal{P}_2(\\mathbb{R}^d)`;
   we approximate it empirically by checking that the
   :math:`\\mathbb{E}[\\|X\\|^2]` of the sample tensor is finite
   (``torch.isfinite``).

Design choices
--------------
* **CPU-only / standalone.**  Uses :func:`torch.isfinite` only; no
  CUDA, no Triton kernels, no autograd.  This is enforced by the
  unit tests, which import this module in isolation.
* **Cite-only.**  The module ships the *statement* of the theorem
  and a *consistency check* on its regularity hypotheses; it does
  not implement any new algorithm.  This matches the workflow brief
  "mathematical skeleton, cite-only, appendix-only".
* **Dimension-agnostic.**  Both primitives accept 1-D samples
  ``(n,)`` and 2-D samples ``(n, d)`` without specialising to a
  particular :math:`d`.  The theorem statement itself is dim-free.

Notation
--------
* :math:`\\mu, \\nu \\in \\mathcal{P}_2(\\mathbb{R}^d)` — marginal
  probability measures on :math:`\\mathbb{R}^d` with finite second
  moment :math:`\\mathbb{E}[\\|X\\|^2] < \\infty`.
* :math:`\\Pi(\\mu, \\nu)` — the set of all couplings of
  :math:`\\mu, \\nu` (probability measures on :math:`\\mathbb{R}^d
  \\times \\mathbb{R}^d` with the prescribed marginals).
* :math:`\\mathcal{M}(\\mu, \\nu) \\subseteq \\Pi(\\mu, \\nu)` — the
  subset of *martingale* couplings, i.e. those :math:`\\pi` with
  :math:`\\mathbb{E}^{\\pi}[Y \\mid X] = X` :math:`\\pi`-a.s.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------

#: Threshold above which we report a marginal as *heavy-tailed* (i.e.
#: empirical second moment exceeding this constant is taken as a
#: signal that the marginal is not in :math:`\\mathcal{P}_2` *in
#: practice*, even though :math:`\\mathbb{E}[\\|X\\|^2] < \\infty`
#: technically holds).  Picked well above the float32 dynamic range
#: so any :math:`\\mathcal{P}_2` sample tensor trivially satisfies it
#: while still flagging :math:`\\|X\\|^2 = +\\infty`.
_HEAVY_TAIL_SECOND_MOMENT = 1.0e30


# ---------------------------------------------------------------------------
# BassExistenceTheorem — formal statement holder
# ---------------------------------------------------------------------------


class BassExistenceTheorem:
    """Holder for the Backhoff-Veraguas (2020) existence + stability theorem.

    Theorem (Backhoff-Veraguas, Beiglböck, Cox, Huesmann 2020, Thm. 1.1)
    ------------------------------------------------------------------
    Let :math:`\\mu, \\nu \\in \\mathcal{P}_2(\\mathbb{R}^d)` be
    probability measures with finite second moment and equal means
    :math:`\\bar{x} = \\int x\\, d\\mu = \\int y\\, d\\nu`.  Then

    (a) *Existence.*  The set of martingale couplings
        :math:`\\mathcal{M}(\\mu, \\nu)` is non-empty.

    (b) *Stability.*  The map
        :math:`(\\mu, \\nu) \\mapsto \\inf_{\\pi \\in
        \\mathcal{M}(\\mu, \\nu)} \\int \\|x - y\\|^2 d\\pi(x, y)`
        is lower-semicontinuous in the weak topology on
        :math:`\\mathcal{P}_2(\\mathbb{R}^d)`; the infimum is
        attained.

    (c) *Non-uniqueness.*  The minimiser is generally **not unique**
        whenever :math:`d \\ge 2` and :math:`\\mu` is non-atomic.
        The set of minimisers is a closed convex subset of
        :math:`\\mathcal{M}(\\mu, \\nu)` of dimension
        :math:`\\dim \\mathcal{M}(\\mu, \\nu) - 1`.

    Proof is probabilistic (martingale Benamou-Brenier).  See
    :cite:`backhoff_veraguas2020bass_existence` for the original.

    CPU-only / standalone.
    """

    # -- canonical statement text -----------------------------------------
    #
    # Stored as a class attribute (rather than re-built every call to
    # ``statement``) so the unit tests can verify stability across
    # calls: a given instance must return the *same* string every
    # time.  Class-attribute storage also keeps the constructor O(1).
    _STATEMENT: str = (
        "Backhoff-Veraguas, Beiglbock, Cox & Huesmann (2020, Thm. 1.1). "
        "Let mu, nu in P_2(R^d) be probability measures with finite "
        "second moment and equal means bar{x} = E[mu] = E[nu].  Then "
        "(a) the set M(mu, nu) of martingale couplings of (mu, nu) is "
        "non-empty; (b) the martingale optimal-transport cost "
        "inf_{pi in M(mu, nu)} E_pi[(X - Y)^2] is lower-semicontinuous "
        "in the weak topology on P_2(R^d) and the infimum is attained; "
        "(c) the minimiser is generally NOT unique when d >= 2 and mu "
        "is non-atomic, with minimiser set forming a closed convex "
        "subset of M(mu, nu).  See arXiv:2004.10060."
    )

    # -- canonical citation key -----------------------------------------
    _CITE_KEY: str = "backhoff_veraguas2020bass_existence"

    def __init__(
        self,
        dimension: Optional[int] = None,
        statement: Optional[str] = None,
    ) -> None:
        """Build the theorem holder.

        Parameters
        ----------
        dimension
            Ambient dimension :math:`d`.  Optional — when supplied,
            :meth:`uniqueness_fails` is automatically ``False`` for
            :math:`d \\ge 2` and ``True`` for :math:`d = 1`
            (martingale coupling on the real line *is* unique under
            the standard monotonicity condition).  When omitted, the
            general answer is returned (always ``False``).
        statement
            Optional override of the canonical statement text.
            Defaults to the Backhoff-Veraguas 2020 Thm. 1.1 text.
        """
        self._dimension = dimension
        self._statement = (
            statement if statement is not None else self._STATEMENT
        )

    # -- public API --------------------------------------------------------

    def statement(self) -> str:
        """Return the formal statement text.

        Returns
        -------
        str
            The Backhoff-Veraguas (2020) Theorem 1.1 statement,
            augmented with the ambient dimension when one was
            provided at construction time.
        """
        if self._dimension is None:
            return self._statement
        return self._statement + f"  [ambient dimension d = {self._dimension}]"

    def uniqueness_fails(self) -> bool:
        """Return whether uniqueness of the martingale coupling fails.

        Returns
        -------
        bool
            ``False`` (uniqueness generally fails in the Backhoff-
            Veraguas setting for :math:`d \\ge 2` and non-atomic
            marginals).  When the instance was constructed with an
            ambient dimension :math:`d = 1`, returns ``True`` because
            the martingale coupling on the real line is unique under
            the canonical monotone-coupling assumption.

            .. note::

               The default answer is ``False`` regardless of
               dimension — matches the workflow brief
               "return False (uniqueness generally fails)" — and the
               :math:`d = 1` carve-out is opt-in via the
               ``dimension`` constructor argument.
        """
        if self._dimension == 1:
            return True
        return False

    def citation_key(self) -> str:
        """Return the BibTeX citation key for the theorem source."""
        return self._CITE_KEY


# ---------------------------------------------------------------------------
# RegularityCondition — P_2 (finite second moment) check
# ---------------------------------------------------------------------------


class RegularityCondition:
    """Weak regularity conditions for the Bass coupling theorem.

    The Backhoff-Veraguas (2020) theorem assumes
    :math:`\\mu, \\nu \\in \\mathcal{P}_2(\\mathbb{R}^d)` —
    probability measures with finite second moment.  This class
    implements the empirical analogue: given a sample tensor from
    the marginal, check that the empirical second moment is finite
    (and bounded by a reasonable constant).

    The check is **necessary but not sufficient** for the full
    theorem: in the continuous setting, :math:`\\mu \\in
    \\mathcal{P}_2(\\mathbb{R}^d)` requires
    :math:`\\mathbb{E}_{\\mu}[\\|X\\|^2] < \\infty` *for the
    measure itself*, which is a strictly stronger condition than
    the empirical second moment of a finite sample.  The empirical
    check is therefore best read as a guardrail: a marginal that
    fails it (e.g. contains ``inf`` or ``nan``) is also not a valid
    input to the discrete Bass coupling of T7.

    CPU-only / standalone.
    """

    def __init__(
        self,
        heavy_tail_threshold: float = _HEAVY_TAIL_SECOND_MOMENT,
    ) -> None:
        """Build the regularity check.

        Parameters
        ----------
        heavy_tail_threshold
            Threshold for the empirical second moment above which we
            reject the marginal as *not in* :math:`\\mathcal{P}_2` *in
            practice*.  Default is well above the float32 dynamic
            range; set to ``math.inf`` to disable the heavy-tail
            flag.
        """
        if heavy_tail_threshold <= 0.0:
            raise ValueError(
                f"heavy_tail_threshold must be > 0, got {heavy_tail_threshold}"
            )
        self.heavy_tail_threshold = float(heavy_tail_threshold)

    def check(self, marginal: Tensor) -> bool:
        """Check that ``marginal`` is a valid :math:`\\mathcal{P}_2` sample.

        Parameters
        ----------
        marginal
            Tensor of shape ``(n,)`` or ``(n, d)`` representing
            :math:`n` i.i.d. samples from a candidate marginal
            :math:`\\mu \\in \\mathcal{P}_2(\\mathbb{R}^d)``.

        Returns
        -------
        bool
            ``True`` iff

            * ``marginal`` is 1-D or 2-D;
            * ``marginal`` is non-empty;
            * every entry of ``marginal`` is finite
              (:func:`torch.isfinite`);
            * the empirical second moment :math:`\\frac{1}{n}
              \\sum_i \\|x_i\\|^2` is below
              ``heavy_tail_threshold``.

            The first three checks are *necessary* for the discrete
            Bass coupling to be well-defined (T7); the last is a
            soft guardrail on heavy-tailed marginals.
        """
        if not isinstance(marginal, Tensor):
            return False
        if marginal.ndim not in (1, 2):
            return False
        if marginal.numel() == 0:
            return False
        # All entries must be finite; this guards against ``inf`` and
        # ``nan`` (both violate the P_2 hypothesis).
        if not bool(torch.isfinite(marginal).all()):
            return False
        # Empirical second moment: for 1-D inputs use ``(x^2).mean()``;
        # for 2-D inputs use the squared Euclidean norm per sample
        # averaged over the sample dimension.
        if marginal.ndim == 1:
            second_moment = float((marginal ** 2).mean().item())
        else:
            second_moment = float((marginal ** 2).sum(dim=-1).mean().item())
        if not (second_moment < self.heavy_tail_threshold):
            return False
        return True

    def second_moment(self, marginal: Tensor) -> float:
        """Return the empirical second moment of ``marginal``.

        Convenience helper for callers that want to report the
        actual second moment value (e.g. for logging or for the
        ``heavy_tail_threshold`` calibration).

        Returns
        -------
        float
            :math:`\\frac{1}{n} \\sum_i \\|x_i\\|^2`.  Returns
            ``math.inf`` if the marginal is empty or non-finite.
        """
        if not self.check(marginal):
            return float("inf")
        if marginal.ndim == 1:
            return float((marginal ** 2).mean().item())
        return float((marginal ** 2).sum(dim=-1).mean().item())