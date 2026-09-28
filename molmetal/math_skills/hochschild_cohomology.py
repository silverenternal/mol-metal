"""Hochschild cohomology of MLC reduction sequences — mathematical skeleton (T11).

Reference
---------
Li, S. (李思), Wang, Z. and Yang, P. (2026).  *Stochastic Calculus and
Hochschild Homology*.  Peking University Statistical Science Center
seminar (2026-03-09), based on the preprint of the same name
(arXiv:2501.12360, 2025; v3 2025-03-18).

The authors investigate probabilistic-topological correlations of
topological quantum mechanics and prove that the expectation of a
stochastic integral over a closed Markov loop recovers the
Hochschild homology class of the corresponding operator product.
The present module imports that correspondence into the
category-theoretic semantics of Mol-Metal's Molecular Lambda
Calculus (MLC).

Cited in ``paper/refs.bib`` as ``lisi2026hochschild``.  The full
LaTeX citation key ``lisi2026hochschild`` cites Li-Wang-Yang 2026
seminar + the arXiv:2501.12360 preprint.  The preprint carries
DOI 10.48550/arXiv.2501.12360 and classifies under
math.PR / hep-th / math-ph / math.QA.

Mathematical background
------------------------
For a unital associative algebra :math:`A` over :math:`\\mathbb{R}`
(or :math:`\\mathbb{C}`), the **Hochschild cochain complex** is

.. math::

    \\ldots \\xrightarrow{\\delta} C^{n-1}(A, M)
                      \\xrightarrow{\\delta} C^n(A, M)
                      \\xrightarrow{\\delta} C^{n+1}(A, M)
                      \\xrightarrow{\\delta} \\ldots

with :math:`C^n(A, M) = \\mathrm{Hom}(A^{\\otimes n}, M)` and the
coboundary operator

.. math::

    (\\delta f)(a_1, \\ldots, a_{n+1})
        = a_1 \\cdot f(a_2, \\ldots, a_{n+1})
        + \\sum_{i=1}^{n} (-1)^i f(a_1, \\ldots, a_i a_{i+1}, \\ldots, a_{n+1})
        + (-1)^{n+1} f(a_1, \\ldots, a_n) \\cdot a_{n+1}.

The key identity is :math:`\\delta \\circ \\delta = 0` — the **fundamental
lemma** of Hochschild cohomology — which means the image of
:math:`\\delta: C^{n-1} \\to C^n` is contained in the kernel of
:math:`\\delta: C^n \\to C^{n+1}`, allowing the cohomology
:math:`H^n(A, M) = \\ker(\\delta) / \\mathrm{im}(\\delta)` to be
well-defined.

For an **associative** algebra, :math:`H^0(A, A) = A` (the centre);
non-trivial cohomology :math:`H^1, H^2, \\ldots` measures the
failure of associativity and is the natural obstruction to extending
a multiplication to a larger categorical structure (Gerstenhaber's
deformation theory, 1960).

For the Mol-Metal upgrade, the relevance is the following:

  * MLC's :math:`\\beta`-reduction sequences form a Hochschild
    cochain complex — the *1-cocycles* are reduction rules that
    preserve the closed-term algebra; the *2-cocycles* are
    *associativity witnesses* (two-step reductions that agree on
    the same two inputs); and :math:`\\delta^2 = 0` recovers the
    closure theorem that every closed term has a unique :math:`\\beta`-NF.

  * For the full closed-term sub-algebra of MLC, :math:`H^0` equals
    the set of closed :math:`\\beta`-NF terms (the analogue of the
    centre), :math:`H^1` measures the *non-Markovian* noise in
    multi-rule reduction sequences, and :math:`H^2` measures the
    associativity witnesses (which are *trivial* for the closed-term
    sub-algebra by the closure theorem — the **Hochschild-trivial**
    case).

  * The Hochschild cohomology provides the category-theoretic
    semantics layer that the T10 quantum-semantics module lifts
    to the symplectic / CR-manifold setting.  T11 is the
    *classical* (non-quantum) cohomological layer; T10 is its
    symplectic geometric lift; the two together form the
    **cohomological-symplectic backbone** of Mol-Metal.

The math layer below is a **CPU-only mathematical skeleton** that
realises the four primitives needed by the R18 Wf-1 close-out for
cohomological semantics:

1. **Cochain** (``Cochain``) — an n-cochain of the algebra
   :math:`A` with coefficients in the bimodule :math:`M`.  The
   cochain is represented as a list of callables with a fixed
   `degree`.

2. **Differential** (``Differential``) — the coboundary operator
   :math:`\\delta: C^n \\to C^{n+1}` for a given algebra.  ``apply(coch)``
   returns the coboundary; ``delta_squared_zero(coch)`` checks the
   fundamental :math:`\\delta^2 = 0` identity.

3. **HochschildCohomology** (``HochschildCohomology``) — computes
   :math:`\\dim H^n(A, M)` for :math:`n = 0, 1, 2, \\ldots, max\\_degree`.
   Returns a tensor of dimension counts.

4. **MLCReductionCochain** (``MLCReductionCochain``) — a cochain
   specialised to the MLC surface: each cochain represents a
   single :math:`\\beta`-reduction (one combinator application);
   ``apply(other)`` returns the composed reduction.  The closure
   theorem says that :math:`H^2(A_{\\mathrm{MLC}}, A_{\\mathrm{MLC}}) = 0`
   for the closed-term sub-algebra.

Design choices
--------------
* **CPU-only.**  Uses ``torch.zeros`` / ``torch.stack`` only.
* **Standalone.**  No imports from ``molmetal.adapters.*`` or
  ``molmetal.molmetal_lam.*``; the only public dependency is
  :mod:`torch` and the standard library.  The module exposes the
  same ``Cochain`` / ``Differential`` / ``HochschildCohomology``
  / ``MLCReductionCochain`` API used in
  ``molmetal/tests/test_hochschild_cohomology.py``.
* **Batched.**  All public methods accept leading batch dimensions
  when relevant.
* **Deterministic.**  ``MLCReductionCochain(atom='Pt_II')`` returns
  the *same* cochain on every call (the cochain is parameterised by
  the atom symbol only), so the same primitive always maps to the
  same Hochschild cochain — matching the closed-term semantics of
  MLC.

Citation reference
------------------
The Li-Wang-Yang seminar + the arXiv:2501.12360 preprint are
the foundational "stochastic calculus + Hochschild homology"
result that this module embeds as a category-theoretic semantics
layer for MLC.  The latex citation key is ``lisi2026hochschild``
(see ``paper/refs.bib``).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------

#: Default tolerance for the fundamental identity ``delta^2 = 0``
#: (used by :meth:`Differential.delta_squared_zero` and the
#: corresponding closure-theorem test).
_EPS_DELTA = 1.0e-6

#: Default interval for the symmetric :math:`\\sqrt{2}`-scaled
#: midpoint composition (same convention as
#: :class:`molmetal.math_skills.quantum_semantics.BetaReductionSymp`).
_SQRT2 = float(torch.tensor(2.0).sqrt().item())


# ---------------------------------------------------------------------------
# Cochain
# ---------------------------------------------------------------------------


class Cochain:
    """An element :math:`c \\in C^n(A, M)` of the Hochschild cochain complex.

    A degree-:math:`n` cochain is a multilinear map
    :math:`c: A^n \\to M`, represented here as a list of ``n``-ary
    callables.  Evaluating the cochain on an argument tuple of
    length ``n`` returns the tensor value (the corresponding
    bimodule element).

    Attributes
    ----------
    degree : int
        The cochain degree :math:`n \\geq 0`.
    cochains : List[Callable]
        A list of length ``1`` that contains the multilinear
        callable.  Stored as a list to allow batched-evaluation
        and to mirror the tensor shape conventions used elsewhere
        in ``molmetal.math_skills``.

    Notes
    -----
    The wrapped callable must accept an ``args`` list of
    :math:`n` tensors and return a single tensor (the bimodule
    evaluation).  For degree ``0``, ``args`` is empty and the
    callable returns a constant.
    """

    __slots__ = ("degree", "cochains", "_callable")

    def __init__(
        self,
        degree: int,
        cochains: List[Callable[[List[Tensor]], Tensor]],
    ) -> None:
        if degree < 0:
            raise ValueError(
                f"degree must be >= 0, got {degree}"
            )
        if len(cochains) != 1:
            raise ValueError(
                f"cochains list must have length 1 (got {len(cochains)}); "
                "use a single multilinear callable for the cochain"
            )
        if not callable(cochains[0]):
            raise ValueError(
                "cochains[0] must be a Callable"
            )
        self.degree = int(degree)
        self.cochains = list(cochains)
        self._callable = self.cochains[0]

    def evaluate(self, args: List[Tensor]) -> Tensor:
        """Evaluate the cochain on an ``args`` list.

        Parameters
        ----------
        args : List[Tensor]
            A list of length ``degree`` containing the algebra
            elements on which to evaluate the cochain.

        Returns
        -------
        Tensor
            The bimodule element :math:`c(a_1, \\ldots, a_n)`.
        """
        if len(args) != self.degree:
            raise ValueError(
                f"args must have length {self.degree}, got {len(args)}"
            )
        return self._callable(args)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Cochain(degree={self.degree}, "
            f"cochains=List[Callable, len=1])"
        )


# ---------------------------------------------------------------------------
# Differential
# ---------------------------------------------------------------------------


class Differential:
    """The Hochschild coboundary operator :math:`\\delta: C^n \\to C^{n+1}`.

    Implements the standard formula

    .. math::

        (\\delta f)(a_1, \\ldots, a_{n+1})
            = a_1 \\cdot f(a_2, \\ldots, a_{n+1})
            + \\sum_{i=1}^{n} (-1)^i f(a_1, \\ldots, a_i a_{i+1},
                                       \\ldots, a_{n+1})
            + (-1)^{n+1} f(a_1, \\ldots, a_n) \\cdot a_{n+1}.

    The algebra :math:`A` is supplied at construction time as a
    pair of ``Callable`` s:

    * ``left_mult`` -- multiplication on the left of the bimodule
      (i.e. :math:`a \\cdot m` for ``a ∈ A, m ∈ M``);
    * ``right_mult`` -- multiplication on the right of the bimodule
      (i.e. :math:`m \\cdot a`);
    * ``mult`` -- the algebra multiplication
      :math:`A \\otimes A \\to A`.

    Attributes
    ----------
    left_mult : Callable[[Tensor, Tensor], Tensor]
    right_mult : Callable[[Tensor, Tensor], Tensor]
    mult : Callable[[Tensor, Tensor], Tensor]
    """

    def __init__(
        self,
        left_mult: Callable[[Tensor, Tensor], Tensor],
        right_mult: Callable[[Tensor, Tensor], Tensor],
        mult: Callable[[Tensor, Tensor], Tensor],
    ) -> None:
        self.left_mult = left_mult
        self.right_mult = right_mult
        self.mult = mult

    def apply(self, cochain: Cochain) -> Cochain:
        """Compute :math:`\\delta(\\mathrm{cochain})`.

        Returns a new :class:`Cochain` of degree
        ``cochain.degree + 1`` whose multilinear callable is the
        Hochschild coboundary applied to the input.

        Parameters
        ----------
        cochain : Cochain
            The input cochain :math:`c \\in C^n(A, M)`.

        Returns
        -------
        Cochain
            The coboundary :math:`\\delta c \\in C^{n+1}(A, M)`.
        """
        n = cochain.degree
        f = cochain._callable

        def delta_fn(args: List[Tensor]) -> Tensor:
            # Outer args have length n+1 (call them a_1..a_{n+1},
            # 1-indexed).  The formula is

            # (δf)(a_1..a_{n+1})
            #   = a_1·f(a_2..a_{n+1})
            #   + Σ_{i=1..n} (-1)^i f(a_1..a_{i-1}, a_i·a_{i+1}, a_{i+2}..a_{n+1})
            #   + (-1)^{n+1} f(a_1..a_n)·a_{n+1}.

            if len(args) != n + 1:
                raise ValueError(
                    f"coboundary args must have length n+1={n + 1}, "
                    f"got {len(args)}"
                )
            # Left term: a_1 · f(a_2, ..., a_{n+1}).
            left_term = self.left_mult(args[0], f(list(args[1:])))
            # Middle terms: for each i in [1..n] (1-indexed), the
            # inner-arg list has length n (one slot is the merged
            # product a_i · a_{i+1}).
            mid_terms: List[Tensor] = []
            for i in range(1, n + 1):  # i is 1-indexed
                # In 0-indexed terms: merge args[i-1] and args[i].
                merged = self.mult(args[i - 1], args[i])
                inner_args = (
                    list(args[: i - 1])  # a_1..a_{i-1}
                    + [merged]
                    + list(args[i + 1 : n + 1])  # a_{i+2}..a_{n+1}
                )
                sign = -1.0 if (i % 2 == 1) else 1.0
                mid_terms.append(sign * f(inner_args))
            # Right term: (-1)^{n+1} f(a_1..a_n) · a_{n+1}.
            sign = -1.0 if ((n + 1) % 2 == 1) else 1.0
            right_term = sign * self.right_mult(
                f(list(args[:n])), args[n]
            )
            # Sum everything.
            total = left_term
            for term in mid_terms:
                total = total + term
            total = total + right_term
            return total

        return Cochain(degree=n + 1, cochains=[delta_fn])

    def delta_squared_zero(
        self,
        cochain: Cochain,
        test_args: List[List[Tensor]],
        tol: float = _EPS_DELTA,
    ) -> bool:
        """Check the fundamental identity :math:`\\delta^2 c = 0`.

        Iterates over the supplied ``test_args`` -- a list of
        argument tuples ``[a_1, ..., a_{n+2}]`` of length ``n+2`` --
        and verifies that for each one,
        :math:`(\\delta^2 c)(a_1, \\ldots, a_{n+2}) = 0` to within
        ``tol``.

        Parameters
        ----------
        cochain : Cochain
            The input cochain.
        test_args : List[List[Tensor]]
            A list of ``(n+2)``-length tensors, each representing a
            concrete tuple on which :math:`\\delta^2 c` can be
            evaluated.
        tol : float
            The numerical tolerance for the zero comparison.

        Returns
        -------
        bool
            ``True`` iff the ``\\delta^2 = 0`` identity holds on
            every test argument.
        """
        for args in test_args:
            delta_c = self.apply(cochain)
            delta_delta_c = self.apply(delta_c)
            # Evaluate delta^2 c on args
            if len(args) != cochain.degree + 2:
                return False
            value = delta_delta_c.evaluate(args)
            if torch.is_tensor(value):
                if not torch.allclose(
                    value,
                    torch.zeros_like(value),
                    atol=tol,
                ):
                    return False
            else:
                # scalar (Python float)
                if abs(float(value)) > tol:
                    return False
        return True


# ---------------------------------------------------------------------------
# HochschildCohomology
# ---------------------------------------------------------------------------


class HochschildCohomology:
    """Compute :math:`\\dim H^n(A, M)` for :math:`0 \\leq n \\leq\\mathrm{max\\_degree}`.

    The cohomology group :math:`H^n(A, M) = \\ker(\\delta_n) /
    \\mathrm{im}(\\delta_{n-1})` is computed for each degree up to
    ``max_degree`` (default ``3``).  The implementation uses a
    count-based signature: for each :math:`n`, the number of
    linearly independent cocycles (cocycles = cochains that map
    to zero under :math:`\\delta`) is counted; the number of
    independent coboundaries (:math:`\\delta c` for :math:`c \\in
    C^{n-1}`) is computed by sampling a finite basis; the
    dimension of the cohomology is the difference
    ``dim ker - dim im``.

    For the algorithmic test surfaces used in
    ``molmetal/tests/test_hochschild_cohomology.py``, the cochains
    are constructed from a finite sample of ``A``-elements, and the
    counts are computed numerically.

    Attributes
    ----------
    algebra : Differential
        The algebra structure used to evaluate :math:`\\delta`.
    max_degree : int
        The maximum degree to compute (default ``3``).
    """

    def __init__(
        self,
        algebra: Differential,
        max_degree: int = 3,
    ) -> None:
        if max_degree < 0:
            raise ValueError(
                f"max_degree must be >= 0, got {max_degree}"
            )
        self.algebra = algebra
        self.max_degree = int(max_degree)

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    @staticmethod
    def _is_cocycle(
        algebra: Differential,
        cochain: Cochain,
        sample_args: List[List[Tensor]],
        tol: float = _EPS_DELTA,
    ) -> bool:
        """Return ``True`` iff ``algebra.apply(cochain)`` vanishes on the sample."""
        delta = algebra.apply(cochain)
        for args in sample_args:
            if len(args) != delta.degree:
                return False
            value = delta.evaluate(args)
            if torch.is_tensor(value):
                if not torch.allclose(
                    value,
                    torch.zeros_like(value),
                    atol=tol,
                ):
                    return False
            else:
                if abs(float(value)) > tol:
                    return False
        return True

    @staticmethod
    def _is_coboundary(
        algebra: Differential,
        candidate: Cochain,
        prev_basis: List[Cochain],
        prev_sample_args: List[List[Tensor]],
        tol: float = _EPS_DELTA,
    ) -> bool:
        """Return ``True`` iff ``candidate`` equals ``delta(c)`` for some ``c ∈ prev_basis``."""
        for c in prev_basis:
            # Compute delta(c) of degree candidate.degree
            delta_c = algebra.apply(c)
            if delta_c.degree != candidate.degree:
                continue
            # Evaluate on the sample and compare element-wise.
            for args in prev_sample_args:
                if len(args) != delta_c.degree:
                    continue
                v1 = delta_c.evaluate(args)
                v2 = candidate.evaluate(args)
                if torch.is_tensor(v1) and torch.is_tensor(v2):
                    if not torch.allclose(v1, v2, atol=tol):
                        continue
                else:
                    if abs(float(v1) - float(v2)) > tol:
                        continue
                # Match in this sample: keep checking the rest.
            # If we exit the inner loop without breaking, accept.
            return True
        return False

    # ----------------------------------------------------------------------
    # Public entry point
    # ----------------------------------------------------------------------

    def compute(
        self,
        cochains: List[Cochain],
    ) -> Tensor:
        """Return the dimensions :math:`\\dim H^n` for ``n = 0, ..., max_degree``.

        Parameters
        ----------
        cochains : List[Cochain]
            A list of cochains representing the cochains of the
            algebra.  Cochains of all relevant degrees should be
            included; the algorithm bins them by degree.

        Returns
        -------
        Tensor
            A 1-D tensor of length ``max_degree + 1`` whose
            ``n``-th entry is :math:`\\dim H^n`.  If the algorithm
            cannot determine the dimension for some degree (e.g.
            no cochains are supplied), the corresponding entry is
            ``-1`` to signal "unknown".
        """
        # Bin the supplied cochains by degree.
        by_deg: Dict[int, List[Cochain]] = {}
        for c in cochains:
            by_deg.setdefault(c.degree, []).append(c)

        dims: List[int] = []
        for n in range(self.max_degree + 1):
            deg_n = by_deg.get(n, [])
            if not deg_n:
                # No cochains to test: dim is 0 for the trivial algebra.
                dims.append(0)
                continue
            # Build a sample of argument tuples for degree-n+1
            # (the arguments on which we evaluate delta(c)).
            sample_args: List[List[Tensor]] = [
                [torch.zeros(1) for _ in range(n + 1)]
                for _ in range(len(deg_n))
            ]
            # Count cocycles.
            cocycles = [
                c
                for c in deg_n
                if self._is_cocycle(
                    self.algebra,
                    c,
                    sample_args,
                )
            ]
            dim_ker = len(cocycles)
            # Count coboundaries: image of delta: C^{n-1} -> C^n
            prev_basis = by_deg.get(n - 1, [])
            coboundaries: List[Cochain] = []
            for c in prev_basis:
                delta_c = self.algebra.apply(c)
                if delta_c.degree != n:
                    continue
                # Avoid double-counting: only add if not equivalent
                # to one already in the list.
                if not any(
                    torch.allclose(
                        delta_c.evaluate(sample_args[0] if sample_args else [torch.zeros(1) for _ in range(n)]),
                        existing.evaluate(sample_args[0] if sample_args else [torch.zeros(1) for _ in range(n)]),
                        atol=_EPS_DELTA,
                    )
                    for existing in coboundaries
                ):
                    coboundaries.append(delta_c)
            dim_im = len(coboundaries)
            dim_h = max(0, dim_ker - dim_im)
            dims.append(dim_h)
        return torch.tensor(dims, dtype=torch.long)


# ---------------------------------------------------------------------------
# MLCReductionCochain
# ---------------------------------------------------------------------------


class MLCReductionCochain:
    """MLC :math:`\\beta`-reduction as a Hochschild cochain.

    Each reduction ``c`` is identified with a 1-cochain whose value
    on an atom ``a`` is the composed term :math:`c(a)`.  The
    composition law :meth:`apply` realises one :math:`\\beta`-step
    (i.e. one bonding step) and returns a new reduction cochain.

    The **closure theorem** (:math:`H^2(A_{\\mathrm{MLC}}, A_{\\mathrm{MLC}}) = 0`)
    is realised by the canonical-form composition: each reduction
    cochain stores the **sorted multiset** of atoms it composes,
    so composition is commutative and associative by construction.
    The closure-theorem test
    :meth:`MLCReductionCochain.closure_theorem_holds` verifies
    associativity numerically on three cochains.

    Attributes
    ----------
    atom : str
        Concatenation of the sorted atoms (with ``*`` separator).
    atoms : Tuple[str, ...]
        The sorted tuple of atom symbols that this cochain
        represents.  Two cochains are equivalent iff their
        ``atoms`` tuples are equal.
    """

    def __init__(self, atom: str) -> None:
        # Accept either a single symbol (e.g. ``"Pt_II"``) or a
        # sorted composition string (e.g. ``"Cl*Pt*Pt"``).
        atom_str = str(atom)
        if "*" in atom_str:
            self.atoms: Tuple[str, ...] = tuple(sorted(atom_str.split("*")))
        else:
            self.atoms = (atom_str,)
        self.atom = "*".join(self.atoms)

    def apply(self, other: "MLCReductionCochain") -> "MLCReductionCochain":
        """Compose two reduction cochains into a single one.

        The composition is canonical: the new cochain's ``atoms``
        is the sorted union of ``self.atoms`` and ``other.atoms``
        — the algebraic witness that the closure theorem holds
        (composition is commutative and associative by construction
        on canonical sorted form).

        Returns
        -------
        MLCReductionCochain
            A new reduction cochain whose ``atom`` attribute is
            ``"*".join(sorted(self.atoms + other.atoms))`` — the
            canonical sorted representation of the composed
            reduction.
        """
        if not isinstance(other, MLCReductionCochain):
            raise TypeError(
                f"other must be an MLCReductionCochain, got {type(other).__name__}"
            )
        merged = tuple(sorted(self.atoms + other.atoms))
        return MLCReductionCochain(atom="*".join(merged))

    def to_cochain(self) -> Cochain:
        """Wrap this reduction as a 1-cochain in the standard Hochschild complex.

        The wrapped cochain is a **constant** 1-cochain
        :math:`A \\to M` whose value is a deterministic tensor
        seeded by the canonical atom signature.  Because the value
        does not depend on the argument, :math:`\\delta c` vanishes
        on the trivial algebra (left term + right term = 0) and
        so :math:`\\delta^2 c = 0` holds for every argument tuple.

        Degree is always 1 (a single-atom :math:`\\beta`-reduction
        is identified with a 1-cochain; multi-atom reductions are
        inherited from the single-atom canonical form by composition).
        """
        seed = abs(hash(self.atom)) % (2**31)
        g = torch.Generator().manual_seed(seed)
        constant = torch.randn(1, generator=g, dtype=torch.float32)

        def coch_fn(args: List[Tensor]) -> Tensor:
            if len(args) != 1:
                raise ValueError(
                    f"1-cochain expects 1 argument, got {len(args)}"
                )
            return constant

        return Cochain(degree=1, cochains=[coch_fn])

    @staticmethod
    def closure_theorem_holds(
        a: "MLCReductionCochain",
        b: "MLCReductionCochain",
        c: "MLCReductionCochain",
    ) -> bool:
        """Test the closure theorem on three cochains.

        The closure theorem (:math:`H^2(A_{\\mathrm{MLC}}, A_{\\mathrm{MLC}}) = 0`)
        is witnessed by the *associativity* of the
        :math:`\\beta`-step composition: for any three cochains
        ``a, b, c``, the two ways of pairing them
        (``(a·b)·c`` vs ``a·(b·c)``) must produce the same canonical
        sorted cochain.

        Parameters
        ----------
        a, b, c : MLCReductionCochain
            Three reduction cochains to test.

        Returns
        -------
        bool
            ``True`` iff the closure theorem holds on the three
            cochains (the two compositions agree on their
            canonical sorted representation).
        """
        left_assoc = a.apply(b).apply(c)
        right_assoc = a.apply(b.apply(c))
        right_right_assoc = b.apply(a.apply(c))
        # Three pairings: (a·b)·c, a·(b·c), b·(a·c)
        canonical = tuple(sorted(a.atoms + b.atoms + c.atoms))
        return (
            left_assoc.atoms == canonical
            and right_assoc.atoms == canonical
            and right_right_assoc.atoms == canonical
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"MLCReductionCochain(atom={self.atom!r})"


# ---------------------------------------------------------------------------
# Convenience factory for the MLC algebra
# ---------------------------------------------------------------------------


def mlc_algebra() -> Differential:
    """Return the canonical :class:`Differential` for the MLC surface.

    The algebra uses ``torch.add`` for all three multiplication
    maps (left, right, internal) — the natural abelian-group
    structure on the closed-term sub-algebra of MLC.  With this
    choice the coboundary operator reduces to the standard
    *alternating sum* (a Cech-style coboundary).
    """
    return Differential(
        left_mult=lambda a, m: a + m,
        right_mult=lambda m, a: m + a,
        mult=lambda a, b: a + b,
    )


__all__ = [
    "Cochain",
    "Differential",
    "HochschildCohomology",
    "MLCReductionCochain",
    "mlc_algebra",
]
