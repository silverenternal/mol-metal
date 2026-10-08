"""Wasserstein Gradient Flow — mathematical skeleton (T4).

Reference
---------
Xue, X.-P. (薛小平) (2026).  *Wasserstein Gradient Flows and their
Applications in Generative Modeling*.  (Preprint — see
``paper/refs.bib`` for the canonical citation once the final version
lands.)

This module implements three primitives that the R18 upgrade path
to Mol-Metal will need for *Wasserstein (W₂) gradient flow*
formulations of reaction chemistry:

1. **W2GradientFlow** (``W2GradientFlow``) — the gradient flow of
   the KL functional

   .. math::

       F(\\rho) = \\mathrm{KL}(\\rho \\,\\|\\, \\rho_{\\rm ref})
       = \\int \\rho(x) \\log \\frac{\\rho(x)}{\\rho_{\\rm ref}(x)} \\, \\mathrm{d} x,

   on the Wasserstein-2 space of probability measures
   :math:`(\\mathcal{P}_2(\\mathbb{R}^d), W_2)`.  The Wasserstein
   gradient flow is

   .. math::

       \\partial_t \\rho = \\nabla \\cdot \\bigl( \\rho \\,
           \\nabla \\tfrac{\\delta F}{\\delta \\rho} \\bigr),

   with first variation

   .. math::

       \\tfrac{\\delta F}{\\delta \\rho}(x)
           = \\log \\rho(x) - \\log \\rho_{\\rm ref}(x) + 1,

   so the associated *particle* drift (the negative Wasserstein
   gradient of :math:`F`) is

   .. math::

       v_t(x) = -\\nabla_x \\tfrac{\\delta F}{\\delta \\rho}(x_t)
              = -\\bigl( s(x) - s_{\\rm ref}(x) \\bigr),

   where :math:`s(x) = \\nabla \\log \\rho(x)` is the score and
   :math:`s_{\\rm ref}` is the reference score.  The drift drives
   :math:`\\rho_t` toward :math:`\\rho_{\\rm ref}` along the JKO
   scheme (Jordan, Kinderlehrer & Otto, 1998).

2. **ChemicalGraphMetric** (``ChemicalGraphMetric``) — a discrete
   metric on chemical graphs, induced by a bond-type-weighted graph
   edit distance.  This is the *discrete analogue* of :math:`W_2`
   on chemical-graph space and underwrites the
   :class:`ReactionStepW2` step below.

3. **ReactionStepW2** (``ReactionStepW2``) — a *single* reaction
   step, viewed as one W₂ gradient step in the chemical-graph
   metric space.  Given a reactant and product fingerprint
   (probability vectors over bond types per atom), returns the W₂
   displacement that carries reactant to product in one explicit
   Euler step of the JKO scheme.

Design choices
--------------
* **CPU-only.**  Uses ``torch.linalg``, ``torch.logsumexp`` and
  basic tensor ops only; no CUDA / Triton kernels are needed for
  the *mathematical skeleton*.
* **Standalone.**  No imports from ``molmetal.adapters.*``,
  ``molmetal.molmetal_lam.*`` or any other *internal* subpackage —
  only :mod:`torch`.  This is enforced by the unit tests
  (``test_wasserstein_gradient_flow.py``) which import this module
  in isolation.
* **Batched.**  All public methods accept a leading batch
  dimension ``(B, ...)`` and broadcast over it.

Notation
--------
* :math:`\\rho_t` — time-dependent probability density on
  :math:`\\mathbb{R}^d`.
* :math:`s(x)` — score function :math:`\\nabla \\log \\rho(x)`.
* :math:`v_t` — particle drift (negative W₂ gradient of :math:`F`).
* :math:`\\mathrm{KL}` — Kullback-Leibler divergence.
* :math:`W_2` — Wasserstein-2 distance.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------

#: Floor used when taking ``log`` of density values.  Below this we
#: treat the value as zero to avoid ``log(0) = -inf`` (the score
#: function is otherwise undefined on the support of :math:`\\rho`).
_EPS_LOG = 1.0e-30

#: Floor for the kernel bandwidth squared.  Below this we clamp
#: ``sigma^2`` to avoid division-by-zero in the Gaussian-kernel
#: smoothing of the score function.
_EPS_SIGMA2 = 1.0e-12

#: Default drift-tolerance for the W₂ step assertion in
#: :class:`ReactionStepW2`.  Picked to balance the truncation error
#: of an explicit Euler step against the round-off of a float32
#: tensor at ``dt = 1``.
_EPS_DRIFT = 1.0e-5


# ---------------------------------------------------------------------------
# Mathematical preconditions (CHECK_MATH_PRECONDITIONS=1)
# ---------------------------------------------------------------------------
# The W₂ gradient flow on (P_2(R^d), W_2) is only defined when the
# particle positions ``x`` and the score field ``s(x)`` live in the
# *same* space (matched shape, finite) — otherwise the drift
# ``v = -(s - s_ref)`` is not a tangent vector and the JKO step is
# meaningless.  The :class:`ReactionStepW2` step further assumes the
# fingerprint rows are probability vectors over bond types (the simplex
# metric ``v = p - r`` is only the OT direction when ``r`` and ``p``
# are both in the simplex).  These are checked cheaply without
# touching the numerics.  Default OFF.


def math_preconditions_enabled() -> bool:
    """Return True when ``CHECK_MATH_PRECONDITIONS=1`` is set in the env."""
    import os

    return os.environ.get("CHECK_MATH_PRECONDITIONS", "0") == "1"


def check_preconditions(
    inputs,
    *,
    entry: str = "w2_gradient_flow",
    score=None,
    reference_score=None,
    dt: Optional[float] = None,
    product=None,
    n_bond_types: Optional[int] = None,
    **kwargs,
) -> list:
    """Report violated mathematical preconditions of the W₂ module.

    Parameters
    ----------
    inputs : Tensor
        ``(B, d)`` particle positions ``x`` (W2GradientFlow) or
        ``(..., K)`` reactant fingerprint (ReactionStepW2).
    entry : str
        ``"w2_gradient_flow"`` (W2GradientFlow.drift / .step) or
        ``"reaction_step"`` (ReactionStepW2.forward).
    score : Tensor, optional
        Score field ``s(x)``; must match ``inputs`` shape.
    reference_score : Tensor, optional
        Reference score; must match ``inputs`` shape.
    dt : float, optional
        Step size; must be non-negative.
    product : Tensor, optional
        Product fingerprint (ReactionStepW2); must match ``inputs`` shape.
    n_bond_types : int, optional
        Expected trailing dimension ``K``.

    Returns
    -------
    list[str]
        Empty when all preconditions hold; else one string per
        violation, naming it.  Never raises.
    """
    violations = []
    if inputs is None or not isinstance(inputs, Tensor):
        return ["x: input is None or not a Tensor"]
    if not bool(torch.isfinite(inputs).all()):
        violations.append("x: contains non-finite (NaN/inf) positions")

    if entry == "reaction_step":
        if product is not None:
            if not isinstance(product, Tensor) or product.shape != inputs.shape:
                violations.append(
                    f"product: shape {None if not isinstance(product, Tensor) else tuple(product.shape)} "
                    f"does not match reactant {tuple(inputs.shape)}"
                )
            elif not bool(torch.isfinite(product).all()):
                violations.append("product: contains non-finite entries")
        if n_bond_types is not None and inputs.shape[-1] != n_bond_types:
            violations.append(
                f"reactant: last dim must be n_bond_types={n_bond_types}, "
                f"got {inputs.shape[-1]}"
            )
        # Simplex precondition: each row is a probability vector over K
        # bond types (non-negative, summing to 1).  The W_2 drift
        # v = p - r is only the OT direction in the simplex metric.
        if bool((inputs < 0.0).any()):
            violations.append(
                "reactant: has negative entries — not a probability vector over "
                "bond types"
            )
        row_sums = inputs.reshape(-1, inputs.shape[-1]).sum(dim=-1)
        if bool((row_sums - 1.0).abs().gt(1e-4).any()):
            bad = float((row_sums - 1.0).abs().max())
            violations.append(
                f"reactant: rows are not probability vectors (max |sum-1|={bad:.4g}) — "
                f"the W2 drift requires the bond-type simplex"
            )
        if dt is not None and dt < 0.0:
            violations.append(f"dt: must be >= 0 for an explicit-Euler step, got {dt}")
        return violations

    # entry == "w2_gradient_flow"
    if score is None:
        violations.append("score: missing — the W2 drift requires the score field s(x)")
    else:
        if not isinstance(score, Tensor) or score.shape != inputs.shape:
            violations.append(
                f"score: shape {None if not isinstance(score, Tensor) else tuple(score.shape)} "
                f"does not match x {tuple(inputs.shape)} (drift needs s(x) and x in the same space)"
            )
        elif not bool(torch.isfinite(score).all()):
            violations.append("score: contains non-finite (NaN/inf) entries")
    if reference_score is not None:
        if not isinstance(reference_score, Tensor) or reference_score.shape != inputs.shape:
            violations.append(
                "reference_score: shape must match x (it is subtracted from the score "
                "element-wise)"
            )
    if dt is not None and dt < 0.0:
        violations.append(f"dt: must be >= 0 for an explicit-Euler step, got {dt}")
    return violations


# ---------------------------------------------------------------------------
# W2GradientFlow — KL functional gradient flow on (P_2(R^d), W_2)
# ---------------------------------------------------------------------------


class W2GradientFlow:
    """W₂ gradient flow of the KL functional :math:`F(\\rho) = \\mathrm{KL}(\\rho \\,\\|\\, \\rho_{\\rm ref})`.

    The Wasserstein gradient flow of the KL divergence is the
    canonical *Fokker-Planck* flow:

    .. math::

        \\partial_t \\rho = \\nabla \\cdot \\bigl( \\rho \\,
            \\nabla (s - s_{\\rm ref}) \\bigr),

    where :math:`s(x) = \\nabla \\log \\rho(x)` is the data score
    and :math:`s_{\\rm ref}(x) = \\nabla \\log \\rho_{\\rm ref}(x)`
    is the reference score.  The particle-level drift (the
    characteristic velocity of the associated ODE) is

    .. math::

        v_t(x) = -\\bigl( s(x) - s_{\\rm ref}(x) \\bigr).

    We expose the **score-aware** form, where the user supplies
    :math:`s(x)` directly (the canonical plug-and-play interface for
    score-based learning).

    Parameters
    ----------
    kernel
        Smoothing kernel applied to the score before the drift is
        returned.  One of ``"gaussian"``, ``"imq"``
        (inverse-multi-quadratic) or ``"none"`` (no smoothing).
        The smoothing is necessary when :math:`\\rho` is a particle
        approximation rather than a smooth density (see e.g.  the
        *kernelized Wasserstein gradient flow* of Liu et al., 2022).
    sigma
        Bandwidth of the smoothing kernel; must be strictly positive.

    CPU-only / standalone.
    """

    _KERNELS = ("gaussian", "imq", "none")

    def __init__(self, kernel: str = "gaussian", sigma: float = 1.0) -> None:
        if kernel not in self._KERNELS:
            raise ValueError(f"kernel must be one of {self._KERNELS}, got {kernel!r}")
        if not (sigma > 0.0):
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.kernel = str(kernel)
        self.sigma = float(sigma)

    # -- smoothing kernels -------------------------------------------------

    def _smooth_score(self, x: Tensor, score: Tensor) -> Tensor:
        """Apply the chosen kernel smoothing to the score field.

        For :math:`K(x, y) = K(\\|x - y\\| / \\sigma)`, the kernel-
        smoothed score at :math:`x_i` is

        .. math::

            \\tilde s(x_i) = \\sum_j K(x_i, x_j) \\, s(x_j) \\,
                / \\sum_j K(x_i, x_j).

        We use a *self-attention* form (each query is also a key)
        so the smoothing is parameter-free and depends only on the
        sample set ``x`` and the score field ``score``.

        Parameters
        ----------
        x
            Sample positions of shape ``(B, d)``.
        score
            Score field of shape ``(B, d)``.

        Returns
        -------
        Tensor
            Smoothed score of shape ``(B, d)``.
        """
        if x.shape != score.shape:
            raise ValueError(
                f"x and score must have the same shape, got "
                f"{tuple(x.shape)} vs {tuple(score.shape)}"
            )
        if self.kernel == "none":
            return score
        # Pairwise squared distances ‖x_i − x_j‖² of shape (B, B).
        diff = x.unsqueeze(1) - x.unsqueeze(0)
        sq = (diff * diff).sum(dim=-1)
        if self.kernel == "gaussian":
            w = torch.exp(-0.5 * sq / (self.sigma * self.sigma + _EPS_SIGMA2))
        else:  # "imq"
            w = 1.0 / torch.sqrt(1.0 + sq / (self.sigma * self.sigma + _EPS_SIGMA2))
        # Self-attention form: weighted score, normalised by weight sum.
        denom = w.sum(dim=-1, keepdim=True).clamp_min(_EPS_LOG)
        return (w @ score) / denom

    # -- drift & step ------------------------------------------------------

    def drift(
        self,
        x: Tensor,
        score: Tensor,
        reference_score: Optional[Tensor] = None,
    ) -> Tensor:
        """Return the W₂ drift :math:`v_t(x) = -\\nabla \\tfrac{\\delta F}{\\delta \\rho}(x)`.

        For the KL functional :math:`F = \\mathrm{KL}(\\rho \\,\\|\\,
        \\rho_{\\rm ref})` the first variation is

        .. math::

            \\tfrac{\\delta F}{\\delta \\rho}(x)
                = \\log \\rho(x) - \\log \\rho_{\\rm ref}(x) + 1,

        whose spatial gradient is :math:`s(x) - s_{\\rm ref}(x)`.
        The Wasserstein gradient is its negative.

        Parameters
        ----------
        x
            Sample positions of shape ``(B, d)``.
        score
            Score field :math:`s(x)` of shape ``(B, d)``.
        reference_score
            Optional reference score :math:`s_{\\rm ref}(x)` of
            shape ``(B, d)``.  ``None`` means the reference is the
            uniform density (score zero) — the standard
            Schrödinger-bridge endpoint convention.

        Returns
        -------
        Tensor
            Drift :math:`v_t(x)` of shape ``(B, d)``.
        """
        if x.shape != score.shape:
            raise ValueError(
                f"x and score must have the same shape, got "
                f"{tuple(x.shape)} vs {tuple(score.shape)}"
            )
        if reference_score is not None and reference_score.shape != x.shape:
            raise ValueError(
                f"reference_score must have the same shape as x, "
                f"got {tuple(reference_score.shape)} vs {tuple(x.shape)}"
            )
        smoothed = self._smooth_score(x, score)
        ref = (
            reference_score if reference_score is not None else torch.zeros_like(score)
        )
        if math_preconditions_enabled():
            import warnings as _warnings

            for _v in check_preconditions(
                x,
                entry="w2_gradient_flow",
                score=score,
                reference_score=reference_score,
            ):
                _warnings.warn(f"[math-precondition] {_v}", stacklevel=2)
        return -(smoothed - ref)

    def step(
        self,
        x: Tensor,
        score: Tensor,
        dt: float,
        reference_score: Optional[Tensor] = None,
    ) -> Tensor:
        """One explicit-Euler step of the W₂ gradient flow.

        Updates :math:`x \\mapsto x + \\mathrm{dt} \\cdot v_t(x)`.
        This is the *Jordan-Kinderlehrer-Otto* (JKO) explicit
        scheme at step size ``dt`` (Jordan, Kinderlehrer & Otto,
        1998).  The JKO scheme is *unconditionally stable* for
        :math:`\\mathrm{dt} \\le 1`; we ship the explicit form
        because it is the canonical rate-matched ML baseline.

        Parameters
        ----------
        x
            Current positions of shape ``(B, d)``.
        score
            Score field :math:`s(x)` of shape ``(B, d)``.
        dt
            Step size (non-negative scalar).
        reference_score
            Optional reference score :math:`s_{\\rm ref}(x)`; see
            :meth:`drift` for details.

        Returns
        -------
        Tensor
            Updated positions of shape ``(B, d)``.
        """
        if dt < 0.0:
            raise ValueError(f"dt must be >= 0, got {dt}")
        return x + dt * self.drift(x, score, reference_score=reference_score)

    def kl_decreases(
        self,
        score_norm: float,
        reference_norm: float = 0.0,
    ) -> float:
        """Check whether the KL is non-increasing at one step.

        For the explicit Euler scheme the KL satisfies
        :math:`F(\\rho_{t+\\mathrm{dt}}) \\le F(\\rho_t)` whenever
        :math:`\\mathrm{dt} \\cdot \\| s - s_{\\rm ref} \\|^2 \\le
        2 F(\\rho_t)`, i.e. the *CFL-like* condition on the relative
        score norm.  We expose this as a static check (no
        tensor-side effects) so the trainer can gate updates.

        Parameters
        ----------
        score_norm
            ``\\| s(x) \\|_2`` (any norm) evaluated at the current
            sample.
        reference_norm
            ``\\| s_{\\rm ref}(x) \\|_2`` (default zero for the
            uniform reference).

        Returns
        -------
        float
            The maximum :math:`\\mathrm{dt}` for which the KL is
            guaranteed not to increase.
        """
        if score_norm < 0.0 or reference_norm < 0.0:
            raise ValueError(
                "score norms must be non-negative, got "
                f"{score_norm} and {reference_norm}"
            )
        denom = (score_norm - reference_norm) ** 2
        if denom < _EPS_LOG:
            return float("inf")
        return 2.0 * score_norm / denom


# ---------------------------------------------------------------------------
# ChemicalGraphMetric — discrete metric on chemical graphs
# ---------------------------------------------------------------------------

#: A simple in-memory graph representation: a ``dict`` with
#: ``"nodes"`` (an iterable of node identifiers) and ``"edges"``
#: (an iterable of ``(i, j, bond_type)`` tuples, undirected).
GraphLike = Union[Dict[str, Sequence], Tuple[Sequence, Sequence]]


def _as_graph(g: GraphLike) -> Tuple[List, List[Tuple]]:
    """Coerce a graph-like object into a ``(nodes, edges)`` pair.

    Accepts a dict with ``"nodes"`` / ``"edges"`` keys, or a
    2-tuple of ``(nodes, edges)``.  Returns lists so the caller can
    safely index / sort them.
    """
    if isinstance(g, dict):
        nodes = list(g.get("nodes", []))
        edges = list(g.get("edges", []))
    elif isinstance(g, tuple) and len(g) == 2:
        nodes, edges = g
        nodes, edges = list(nodes), list(edges)
    else:
        raise TypeError(
            "graph must be a dict with 'nodes'/'edges' or a "
            f"(nodes, edges) tuple, got {type(g).__name__}"
        )
    return nodes, edges


class ChemicalGraphMetric:
    """Bond-type-weighted graph edit distance.

    The discrete analogue of the W₂ metric on chemical-graph space:
    the *cost* of editing one graph into another is the minimum-cost
    sequence of node / edge additions, deletions and substitutions,
    with the edge cost weighted by the bond-type difference.

    Specifically, we use the simpler *edge-overlap* proxy: the
    distance is the total bond-type-weighted absolute difference of
    the edge sets, normalised by the symmetric difference.

    .. math::

        d(g_1, g_2) = \\sum_{(i, j) \\in \\Delta}
            \\, | b_1(i, j) - b_2(i, j) |,

    where :math:`\\Delta` is the symmetric difference of the edge
    sets (edge present in one graph but not the other) and
    :math:`b_k(i, j)` is the bond type of edge :math:`(i, j)` in
    graph :math:`g_k`.  Edges present in *both* graphs but with
    different bond types are charged the absolute difference of the
    bond-type codes; edges present in only one graph are charged the
    *maximum* bond-type code (the cost of a full insertion /
    deletion).

    This is metric (positive-definite, symmetric, triangle-inequality
    holding on the discrete domain) and CPU-cheap.

    Parameters
    ----------
    bond_types
        List of bond-type identifiers.  The *cost* of an
        insert / delete is the maximum of this list (the cost of a
        "full" bond); the cost of a bond-type substitution is the
        absolute difference of the two codes.
    """

    def __init__(self, bond_types: Iterable[int]) -> None:
        bt = sorted({int(b) for b in bond_types})
        if not bt:
            raise ValueError("bond_types must be non-empty")
        if any(b < 0 for b in bt):
            raise ValueError(f"bond_types must be non-negative, got {bt}")
        self.bond_types: List[int] = bt
        self.insert_delete_cost = max(bt)

    def _edge_map(self, edges: Sequence[Tuple]) -> Dict[Tuple[int, int], int]:
        """Coerce an edge list into a ``{(i, j): bond_type}`` dict.

        We canonicalise ``(i, j)`` to ``(min(i, j), max(i, j))`` so
        the graph is undirected.
        """
        out: Dict[Tuple[int, int], int] = {}
        for e in edges:
            if len(e) != 3:
                raise ValueError(f"edge must be (i, j, bond_type), got {tuple(e)}")
            i, j, b = e
            key = (min(int(i), int(j)), max(int(i), int(j)))
            out[key] = int(b)
        return out

    def distance(self, g1: GraphLike, g2: GraphLike) -> float:
        """Bond-type-weighted graph edit distance.

        Parameters
        ----------
        g1, g2
            Two graphs, each represented as a dict with
            ``"nodes"`` / ``"edges"`` keys or a ``(nodes, edges)``
            2-tuple.  Edges are ``(i, j, bond_type)`` tuples.

        Returns
        -------
        float
            Non-negative metric distance, equal to ``0`` iff the
            two edge sets are identical.
        """
        _, e1 = _as_graph(g1)
        _, e2 = _as_graph(g2)
        m1 = self._edge_map(e1)
        m2 = self._edge_map(e2)
        keys = set(m1.keys()) | set(m2.keys())
        d = 0.0
        for k in keys:
            b1 = m1.get(k)
            b2 = m2.get(k)
            if b1 is None or b2 is None:
                d += float(self.insert_delete_cost)
            else:
                d += float(abs(b1 - b2))
        return d


# ---------------------------------------------------------------------------
# ReactionStepW2 — single reaction step as W₂ gradient step
# ---------------------------------------------------------------------------


class ReactionStepW2:
    """Single reaction step as a W₂ gradient step in the chemical-graph metric.

    Given a reactant fingerprint :math:`r \\in \\mathbb{R}^{N \\times K}`
    and a product fingerprint :math:`p \\in \\mathbb{R}^{N \\times K}`
    (each row is a probability vector over :math:`K` bond types
    for one atom), the canonical *reaction step* is the W₂
    displacement that takes :math:`r` to :math:`p` in one explicit
    Euler step of the gradient flow

    .. math::

        x_{n+1} = x_n + \\mathrm{dt} \\, v_t(x_n),

    with :math:`v_t(x) = p - r` (the optimal transport direction in
    the simplex metric).  We return the *optimal displacement
    vector* (drift) and the corresponding W₂ distance under the
    :class:`ChemicalGraphMetric`.

    Parameters
    ----------
    n_bond_types
        Number of bond-type bins (default 4: single / double /
        triple / aromatic).
    dt
        Step size for the explicit Euler scheme; default 1.0.  At
        ``dt = 1.0`` one ``forward`` call carries reactant fully to
        product; at ``dt = 0.5`` the call returns the geodesic
        midpoint, etc.  The constructor ``dt`` is the *default*
        used by :meth:`forward`; callers can also pass an explicit
        ``dt_override`` to that method.

    CPU-only / standalone.
    """

    def __init__(self, n_bond_types: int = 4, dt: float = 1.0) -> None:
        if n_bond_types < 1:
            raise ValueError(f"n_bond_types must be >= 1, got {n_bond_types}")
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")
        self.n_bond_types = int(n_bond_types)
        self.dt = float(dt)
        self.metric = ChemicalGraphMetric(bond_types=list(range(n_bond_types)))

    def drift(self, reactant: Tensor, product: Tensor) -> Tensor:
        """Compute the optimal W₂ drift from reactant to product.

        The drift is the *velocity field* :math:`(p - r) / T` where
        :math:`T` is the canonical reaction time.  We absorb
        :math:`T = 1` (unit reaction time) and use ``dt`` (the
        constructor parameter) to scale the Euler step, not the
        drift itself.

        Parameters
        ----------
        reactant, product
            Tensors of shape ``(..., K)`` where ``K`` equals
            ``n_bond_types``.  The two leading shapes must match.

        Returns
        -------
        Tensor
            Drift of the same shape as the inputs, equal to
            ``product - reactant`` (the optimal transport direction
            in the simplex metric).
        """
        if reactant.shape != product.shape:
            raise ValueError(
                f"reactant and product must have the same shape, "
                f"got {tuple(reactant.shape)} vs {tuple(product.shape)}"
            )
        if reactant.shape[-1] != self.n_bond_types:
            raise ValueError(
                f"last dim must be n_bond_types={self.n_bond_types}, "
                f"got {reactant.shape[-1]}"
            )
        return product - reactant

    def forward(
        self,
        reactant: Tensor,
        product: Tensor,
        dt: Optional[float] = None,
    ) -> Tensor:
        """Apply one reaction step: return ``reactant + dt * drift``.

        Parameters
        ----------
        reactant, product
            Tensors of shape ``(..., K)`` where ``K`` equals
            ``n_bond_types``.  The two leading shapes must match.
        dt
            Optional override for the step size; ``None`` uses the
            constructor ``dt``.  At ``dt = 1.0`` the call returns
            ``product``; at ``dt = 0.5`` the geodesic midpoint; etc.

        Returns
        -------
        Tensor
            Updated fingerprint of the same shape as the inputs.
            The output is returned detached so it can be used
            directly as the next-step reactant.
        """
        step = self.dt if dt is None else float(dt)
        if step < 0.0:
            raise ValueError(f"dt must be >= 0, got {step}")
        if math_preconditions_enabled():
            import warnings as _warnings

            for _v in check_preconditions(
                reactant,
                entry="reaction_step",
                product=product,
                dt=step,
                n_bond_types=self.n_bond_types,
            ):
                _warnings.warn(f"[math-precondition] {_v}", stacklevel=2)
        return (reactant + step * self.drift(reactant, product)).detach()

    def w2_distance(self, reactant: Tensor, product: Tensor) -> Tensor:
        """Return the W₂ distance under the chemical-graph metric.

        Computed as the row-wise L1 distance between the
        fingerprints, summed over the leading axes.  This is the
        *flat* W₂ on the bond-type simplex and matches the JKO
        step in the limit :math:`\\mathrm{dt} \\to 0`.

        Parameters
        ----------
        reactant, product
            Tensors of shape ``(..., K)`` where ``K`` equals
            ``n_bond_types``.

        Returns
        -------
        Tensor
            Scalar W₂ distance.
        """
        if reactant.shape != product.shape:
            raise ValueError(
                f"reactant and product must have the same shape, "
                f"got {tuple(reactant.shape)} vs {tuple(product.shape)}"
            )
        if reactant.shape[-1] != self.n_bond_types:
            raise ValueError(
                f"last dim must be n_bond_types={self.n_bond_types}, "
                f"got {reactant.shape[-1]}"
            )
        return (reactant - product).abs().sum()

    def is_reversible(
        self, reactant: Tensor, product: Tensor, tol: float = _EPS_DRIFT
    ) -> bool:
        """Check that ``forward(reactant, product) == product`` (and vice versa).

        For the canonical explicit-Euler W₂ step, the step
        :math:`r \\mapsto p` and the reverse step :math:`p \\mapsto r`
        are inverses; we expose this as a property so callers can
        guard against drift arithmetic errors.

        Parameters
        ----------
        reactant, product
            Tensors of shape ``(..., K)``.
        tol
            Absolute tolerance for the equality check.

        Returns
        -------
        bool
            ``True`` iff the forward and reverse steps are mutual
            inverses.
        """
        forward = self.forward(reactant, product)
        reverse = self.forward(product, reactant)
        return bool(
            torch.allclose(forward, product, atol=tol)
            and torch.allclose(reverse, reactant, atol=tol)
        )


__all__ = [
    "ChemicalGraphMetric",
    "ReactionStepW2",
    "W2GradientFlow",
]
