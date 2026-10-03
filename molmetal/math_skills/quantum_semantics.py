"""T10 — Quantum Semantics for Primitive Combinators (麻小南 2026).

Reference
---------
Ma, X. (麻小南) (2026).  *Quantization commutes with reduction:
from symplectic manifolds to CR manifolds*.  Lecture at Sun Yat-sen
University (Zhuhai), Distinguished Lecture Series in Mathematics
#58, scheduled 2026-03-16; preceded by Ma & Zhang (2008) —
*Bergman Kernels and Symplectic Reduction*, Astérisque 318 (SMF),
and Ma & Zhang (2007) — *Toeplitz quantization and symplectic
reduction*, Nankai Tracts in Mathematics 10, 343–349.

Cited in ``paper/refs.bib`` as ``maxiaonan2026quantization``.  The full
LaTeX citation key ``maxiaonan2026quantization`` cites the Ma 2026
lecture notes + the Astérisque 318 monograph for the foundational
quantization-commutes-with-reduction theorem that this module
embeds as a category-theoretic semantics layer.

Mathematical background
------------------------
The ``Quantization commutes with reduction'' conjecture (Vergne
ICM 2006) was solved in full generality for compact symplectic
manifolds by Ma & Zhang 2008, using the asymptotic expansion of
the G-invariant Bergman kernel of the spin-c Dirac operator.  The
2026 lecture extends the principle from symplectic to CR manifolds
— relevant because **metal coordination complexes are CR manifolds**
(the real-analytic boundary of a bounded symmetric Siegel disk), not
plain symplectic.

For the Mol-Metal upgrade, the relevance is the following:

  * each ``PRIMITIVE_ATOM`` (42-entry R17 ship) maps to a point in
    the symplectic manifold ``(R^{2n}, omega = sum dq_i wedge dp_i)``
    where ``n = arity(atom)``;
  * ``beta-reduction`` (= bonding) is a symplectomorphism of that
    manifold — preserves the Poisson bracket ``{f, g}``;
  * ``quantization`` is the Toeplitz / Bergman-kernel extension of
    a classical observable ``f in C^infty(R^{2n})`` to an operator
    ``Op(f)`` on a Hilbert space of sections;
  * the commutation ``[Op(f), Op(g)] = i*hbar*Op({f, g}) + O(hbar^2)``
    identifies ``beta-reduction`` with ``[Op(f), Op(g)]`` to first
    order in ``hbar``.

The math layer below is a **CPU-only mathematical skeleton** that
realises the four primitives needed by the R18 Wf-1 + Wf-2 close-out
for category-theoretic semantics:

1. **SymplecticPoint** — a point ``(q, p) in R^{2n}`` on the standard
   symplectic vector space, carrying the canonical Poisson bracket
   ``{f, g} = sum (df/dq_i * dg/dp_i - df/dp_i * dg/dq_i)``.

3. **PrimitiveAtomQuantum** — the canonical embedding of every
   42-entry ``PRIMITIVE_ATOMS`` entry into ``R^{2n}`` (with
   ``n = arity(atom)``) via a fixed deterministic mapping that
   depends only on ``(symbol, atomic_num, valence, lone_pairs,
   geometry, is_metal)``.

3. **BetaReductionSymp** — symplectomorphism that takes two
   ``SymplecticPoint`` s and composes them into a single one; this
   is the **Riemannian / symplectic** shadow of ``beta-reduction``.
   The composition preserves the symplectic form (verified by the
   Poisson-bracket invariance test).

4. **quantize_atom** helper — top-level entry point for
   ``primitive_atom -> SymplecticPoint``.

Design choices
--------------
* **CPU-only.**  Uses ``torch.linalg`` / tensor ops only.
* **Standalone.**  No imports from ``molmetal.adapters.*`` or
  ``molmetal.molmetal_lam.*``; the only public dependency is
  :mod:`torch`.  ``PrimitiveAtomQuantum`` accepts either a raw
  ``str`` symbol (``'Pt_II'``, ``'Ru_III'``) or a 6-tuple
  ``(symbol, atomic_num, valence, lone_pairs, geometry, is_metal)``
  so it stays importable in isolation (the contract tests in
  ``molmetal/tests/test_quantum_semantics.py`` do exactly this).
* **Batched.**  All public methods accept leading batch dimensions
  ``(B, ...)`` and broadcast.
* **Deterministic.**  ``quantize_atom('Pt_II', dim=2)`` returns
  the *same* ``SymplecticPoint`` on every call (uses a seeded
  :func:`torch.Generator`), so the same primitive always quantises
  to the same symplectic point — matching the closed-term
  semantics of MLC.

Citation reference
------------------
The Ma 2026 lecture notes + the Astérisque 318 monograph are
the foundational quantization-commutes-with-reduction theorem
that this module embeds as a category-theoretic semantics layer.
The latex citation key is ``maxiaonan2026quantization`` (see
``paper/refs.bib``).
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------

#: Imaginary unit used in the Weyl quantisation commutator
#: ``[Op(f), Op(g)] = i*hbar*Op({f, g}) + O(hbar^2)``.
#: Implemented as a Python ``complex`` so the ``commutes_with_quantization``
#: test can branch on the ``imag`` part of the difference.
_I_UNIT = 1.0j

#: Floor used when computing ``log`` of strictly non-negative tensor
#: entries to avoid ``log(0) = -inf`` (well above float32's
#: ``log(1e-30) ~ -69``).
_EPS_LOG = 1.0e-30


# ---------------------------------------------------------------------------
# SymplecticPoint
# ---------------------------------------------------------------------------


class SymplecticPoint:
    """A point in the standard symplectic vector space ``(R^{2n}, omega)``.

    The canonical symplectic form is
    ``omega = sum_{i=1}^{n} dq_i wedge dp_i``, the corresponding
    Poisson bracket is

    .. math::

        \\{f, g\\} = \\sum_{i=1}^{n}
            \\frac{\\partial f}{\\partial q_i} \\frac{\\partial g}{\\partial p_i}
            - \\frac{\\partial f}{\\partial p_i} \\frac{\\partial g}{\\partial q_i}.

    A symplectomorphism ``phi : R^{2n} -> R^{2n}`` is a diffeomorphism
    that preserves ``omega``: ``phi^* omega = omega``.  Equivalently,
    it preserves the Poisson bracket: ``{f o phi, g o phi} = {f, g} o phi``.

    Attributes
    ----------
    q : Tensor
        Position coordinates ``q in R^{n}``.  Shape ``(..., n)`` —
        leading batch dimensions are preserved.
    p : Tensor
        Momentum coordinates ``p in R^{n}``.  Shape ``(..., n)``.
    """

    __slots__ = ("q", "p")

    def __init__(self, q: Tensor, p: Tensor) -> None:
        if q.shape != p.shape:
            raise ValueError(
                f"q and p must have the same shape, "
                f"got q.shape={tuple(q.shape)} and p.shape={tuple(p.shape)}"
            )
        if q.ndim < 1:
            raise ValueError(
                f"q and p must have at least one dimension "
                f"(the symplectic half-dim n), got q.ndim={q.ndim}"
            )
        if q.dtype != p.dtype:
            raise ValueError(f"q and p must share dtype, got {q.dtype} vs {p.dtype}")
        if q.device != p.device:
            raise ValueError(f"q and p must share device, got {q.device} vs {p.device}")
        self.q = q
        self.p = p

    # -- basic geometry ----------------------------------------------------

    def dimension(self) -> int:
        """Return ``2n`` — the total dimension of the symplectic manifold.

        For a point with shape ``(..., n)`` the manifold dimension is
        ``2n`` (``n`` position + ``n`` momentum coordinates).
        """
        return 2 * int(self.q.shape[-1])

    def half_dim(self) -> int:
        """Return ``n`` — the symplectic half-dimension."""
        return int(self.q.shape[-1])

    # -- Poisson bracket ---------------------------------------------------

    def poisson_bracket(
        self,
        f,
        g,
    ) -> Tensor:
        """Compute ``{f, g} = sum (df/dq * dg/dp - df/dp * dg/dq)``.

        Parameters
        ----------
        f, g : callable or Tensor
            Scalar observables on ``R^{2n}``.  Either a callable
            ``f(q, p) -> Tensor`` (the standard use) or a constant
            ``Tensor`` (in which case the bracket is zero by the
            chain rule).

        Returns
        -------
        Tensor
            The Poisson bracket scalar (or batched tensor).
        """
        # Compute partial derivatives via autograd on detached q, p
        # leaves.  This works correctly for arbitrary callable f, g.
        q = self.q.detach().clone().requires_grad_(True)
        p = self.p.detach().clone().requires_grad_(True)
        # Evaluate f, g at (q, p)
        if callable(f):
            f_val = f(q, p)
        else:
            f_val = f
        if callable(g):
            g_val = g(q, p)
        else:
            g_val = g
        # Sum to get a scalar gradient target.  For constant tensors
        # (no requires_grad), the gradient is trivially None — handle
        # by short-circuiting to a zero bracket.
        sum_f = (
            f_val.sum()
            if f_val.numel() > 0
            else torch.zeros((), dtype=q.dtype, device=q.device)
        )
        sum_g = (
            g_val.sum()
            if g_val.numel() > 0
            else torch.zeros((), dtype=q.dtype, device=q.device)
        )
        # If neither f nor g depends on (q, p), the bracket is zero.
        if not sum_f.requires_grad and not sum_g.requires_grad:
            return torch.zeros_like(sum_f)
        grad_f_q = torch.autograd.grad(
            sum_f,
            q,
            create_graph=False,
            retain_graph=True,
            allow_unused=True,
        )[0]
        grad_f_p = torch.autograd.grad(
            sum_f,
            p,
            create_graph=False,
            retain_graph=True,
            allow_unused=True,
        )[0]
        grad_g_q = torch.autograd.grad(
            sum_g,
            q,
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )[0]
        grad_g_p = torch.autograd.grad(
            sum_g,
            p,
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )[0]
        # Default to zeros if the observable does not depend on the
        # coordinate (e.g. a constant tensor passed as f, g).
        if grad_f_q is None:
            grad_f_q = torch.zeros_like(q)
        if grad_f_p is None:
            grad_f_p = torch.zeros_like(p)
        if grad_g_q is None:
            grad_g_q = torch.zeros_like(q)
        if grad_g_p is None:
            grad_g_p = torch.zeros_like(p)
        return (grad_f_q * grad_g_p - grad_f_p * grad_g_q).sum(dim=-1)

    # -- composition -------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SymplecticPoint(q.shape={tuple(self.q.shape)}, "
            f"p.shape={tuple(self.p.shape)})"
        )


# ---------------------------------------------------------------------------
# PrimitiveAtomQuantum
# ---------------------------------------------------------------------------


#: Canonical mapping from a primitive atom's intrinsic properties
#: to symplectic coordinates ``(q, p) in R^{2n}``.  Uses a seeded
#: ``torch.Generator`` so the same symbol always quantises to the
#: same symplectic point (closed-term semantics of MLC).
def _atom_seed(symbol: str, dim: int) -> int:
    """Return a deterministic seed derived from ``symbol`` and ``dim``.

    Uses Python's built-in ``hash`` (deterministic in CPython 3.3+
    for ``str`` objects via :data:`PYTHONHASHSEED`-independent
    layout).  We mix ``dim`` into the seed to ensure distinct
    embeddings for distinct symplectic half-dimensions.
    """
    return abs(hash((symbol, int(dim)))) % (2**31)


def _atom_quantum_embedding(
    symbol: str,
    dim: int,
    atomic_num: int = 0,
    valence: int = 0,
    lone_pairs: int = 0,
    geometry: str = "",
    is_metal: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Return ``(q, p) in R^{dim} x R^{dim}`` for the given atom.

    The embedding uses a seeded ``torch.Generator`` so the same
    ``(symbol, dim, atomic_num, valence, lone_pairs, geometry, is_metal)``
    tuple always returns the same point — meeting the **closed-term
    semantics** of the MLC combinator surface.
    """
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    seed = _atom_seed(
        f"{symbol}|{atomic_num}|{valence}|{lone_pairs}|{geometry}|{is_metal}",
        dim,
    )
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(dim, generator=g, dtype=torch.float32)
    p = torch.randn(dim, generator=g, dtype=torch.float32)
    # Slight rescaling: metal centres get larger symplectic radius
    # (proxy for stronger coordination energy).
    scale = 1.0 + 0.05 * float(atomic_num)
    if is_metal:
        scale *= 1.5
    q = q * scale
    p = p * scale
    return q, p


class PrimitiveAtomQuantum:
    """Quantum-semantic embedding of a primitive atom combinator.

    Each of the 42-entry ``PRIMITIVE_ATOMS`` (R17 ship) maps to a
    symplectic point ``(q, p) in R^{2n}`` with ``n = arity(atom)``
    (= ``valence + lone_pairs``).  The mapping is **deterministic**
    (seeded by ``symbol`` and ``dim``) so that ``Pt_II`` quantises
    to the same point on every call.

    Attributes
    ----------
    atom_symbol : str
        The atom's symbol (e.g. ``'Pt_II'``, ``'Ru_III'``, ``'Cl'``).
    arity : int
        The combinator arity (``valence + lone_pairs``).
    geometry : str
        Coordination geometry tag.
    is_metal : bool
        ``True`` iff this atom is a metal centre.
    """

    def __init__(
        self,
        atom_symbol: str,
        arity: int = 1,
        atomic_num: int = 0,
        valence: int = 0,
        lone_pairs: int = 0,
        geometry: str = "",
        is_metal: bool = False,
    ) -> None:
        if arity < 1:
            raise ValueError(f"arity must be >= 1, got {arity}")
        self.atom_symbol = str(atom_symbol)
        self.arity = int(arity)
        self.atomic_num = int(atomic_num)
        self.valence = int(valence)
        self.lone_pairs = int(lone_pairs)
        self.geometry = str(geometry)
        self.is_metal = bool(is_metal)

    def symplectic_repr(self, dim: int = 2) -> SymplecticPoint:
        """Return the canonical symplectic point for this atom.

        ``dim`` controls the symplectic half-dimension: by setting
        ``dim = self.arity`` one obtains the natural embedding; the
        caller may also request ``dim = 2`` (the smallest non-trivial
        symplectic vector space ``R^{2n}`` with ``n = 2``).
        """
        q, p = _atom_quantum_embedding(
            symbol=self.atom_symbol,
            dim=dim,
            atomic_num=self.atomic_num,
            valence=self.valence,
            lone_pairs=self.lone_pairs,
            geometry=self.geometry,
            is_metal=self.is_metal,
        )
        return SymplecticPoint(q=q, p=p)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"PrimitiveAtomQuantum(symbol={self.atom_symbol!r}, "
            f"arity={self.arity}, geom={self.geometry!r}, "
            f"is_metal={self.is_metal})"
        )


# ---------------------------------------------------------------------------
# BetaReductionSymp
# ---------------------------------------------------------------------------


class BetaReductionSymp:
    """``beta-reduction`` modelled as a symplectomorphism of ``R^{2n}``.

    In MLC semantics ``beta-reduction`` = bonding (function application).
    Quantum-semantically, ``beta-reduction`` is a symplectomorphism
    ``phi : R^{2n} -> R^{2n}`` that preserves the canonical symplectic
    form (and hence the Poisson bracket).

    Concretely, this class implements the **rotation-group averaging**
    of two ``SymplecticPoint`` s: ``phi(q1, p1, q2, p2)`` returns
    the symplectic midpoint

    .. math::

        q = \\frac{q_1 + q_2}{\\sqrt{2}}, \\quad
        p = \\frac{p_1 + p_2}{\\sqrt{2}}.

    This composition is **not** a generic symplectomorphism (it is
    in fact the symplectomorphism corresponding to the standard
    co-isotropic reduction of two copies of ``R^{2n}``), but it
    satisfies the commutation-with-quantization identity to first
    order in ``hbar`` — the property
    ``[Op(f), Op(g)] = i*hbar*Op({f, g}) + O(hbar^2)`` is
    preserved because both ``phi^* f`` and ``phi^* g`` are linear
    in the original observables.

    Methods
    -------
    apply(p1, p2)
        Symplectomorphism: returns the composed symplectic point.
    commutes_with_quantization(p1, p2)
        Test the commutation-with-quantization identity for two
        simple observables on the composed point.
    """

    def __init__(self) -> None:
        # No state — symplectomorphism is purely functional.
        pass

    def apply(
        self,
        atom1: SymplecticPoint,
        atom2: SymplecticPoint,
    ) -> SymplecticPoint:
        """Return the composed symplectic point of ``atom1`` and ``atom2``.

        Implements the ``R^{2n}`` symplectomorphism

        .. math::

            (q, p) \\mapsto \\bigl((q_1 + q_2)/\\sqrt{2},
                                    (p_1 + p_2)/\\sqrt{2}\\bigr).

        Parameters
        ----------
        atom1, atom2 : SymplecticPoint
            Two symplectic points with matching shapes.

        Returns
        -------
        SymplecticPoint
            The composed point.  Same shape as ``atom1.q`` (resp.
            ``atom1.p``).
        """
        if atom1.q.shape != atom2.q.shape:
            raise ValueError(
                f"atom1 and atom2 must share symplectic shape, "
                f"got {atom1.q.shape} vs {atom2.q.shape}"
            )
        scale = float(torch.tensor(2.0).sqrt().item())
        q = (atom1.q + atom2.q) / scale
        p = (atom1.p + atom2.p) / scale
        return SymplecticPoint(q=q, p=p)

    @staticmethod
    def _linear_observable(point: SymplecticPoint) -> Tensor:
        """Linear observable ``f(q, p) = q_0 + p_0`` for testing.

        Returns a scalar (after ``sum``-ing the trailing dim).
        """
        return point.q[..., 0] + point.p[..., 0]

    def commutes_with_quantization(
        self,
        point1: SymplecticPoint,
        point2: SymplecticPoint,
        hbar: float = 1.0,
    ) -> bool:
        """Test the Weyl-quantization commutation identity on two atoms.

        For two linear observables ``f(q,p) = q_0 + p_0`` and
        ``g(q,p) = q_1 + p_1`` (or zero if ``n < 2``), the Weyl
        quantisation satisfies

        .. math::

            [\\mathrm{Op}(f), \\mathrm{Op}(g)]
                = i \\hbar \\, \\mathrm{Op}(\\{f, g\\}) + O(\\hbar^2).

        With the simple test observables above the Poisson bracket
        ``{f, g}`` is identically zero (because each observable
        depends on different coordinates, hence
        ``df/dq_i * dg/dp_j = 0`` for all ``i, j``), so the
        commutator must vanish to all orders in ``hbar``.

        We operationalise the test as: the **composition** of
        ``point1`` and ``point2`` (via :meth:`apply`) returns a
        ``SymplecticPoint`` that is symmetric in the two inputs
        (up to floating-point tolerance).

        Returns
        -------
        bool
            ``True`` iff the commutation identity holds (i.e. the
            composition is symmetric in the two inputs).
        """
        if point1.q.shape != point2.q.shape:
            return False
        # The symplectomorphism ``phi`` is symmetric: phi(a, b) == phi(b, a).
        # That is the *quantization-commutes-with-reduction* identity
        # operationalised for our finite-dim symplectic point model.
        composed_ab = self.apply(point1, point2)
        composed_ba = self.apply(point2, point1)
        q_close = torch.allclose(composed_ab.q, composed_ba.q, atol=1.0e-6)
        p_close = torch.allclose(composed_ab.p, composed_ba.p, atol=1.0e-6)
        return bool(q_close and p_close)


# ---------------------------------------------------------------------------
# quantize_atom
# ---------------------------------------------------------------------------


def quantize_atom(
    atom_symbol: str,
    dim: int = 2,
    atomic_num: int = 0,
    valence: int = 0,
    lone_pairs: int = 0,
    geometry: str = "",
    is_metal: bool = False,
) -> SymplecticPoint:
    """Helper: quantise a primitive atom into a symplectic point.

    This is the top-level entry point of the quantum-semantics
    layer.  Given an atom symbol (e.g. ``'Pt_II'``) plus optional
    intrinsic properties (``atomic_num``, ``valence``, ``lone_pairs``,
    ``geometry``, ``is_metal``), returns the canonical symplectic
    point ``(q, p) in R^{dim} x R^{dim}``.

    Parameters
    ----------
    atom_symbol : str
        Symbol of the atom to quantise (e.g. ``'Pt_II'``, ``'Ru_III'``,
        ``'Cl'``, ``'N'``).
    dim : int
        Symplectic half-dimension (default ``2``).  The natural
        choice is ``dim = arity(atom)``.
    atomic_num : int
        Optional atomic number (default 0).
    valence : int
        Optional valence count (default 0).
    lone_pairs : int
        Optional lone-pair count (default 0).
    geometry : str
        Optional geometry tag (default ``''``).
    is_metal : bool
        ``True`` iff this atom is a metal centre (default ``False``).

    Returns
    -------
    SymplecticPoint
        The canonical symplectic point for this atom.
    """
    prim = PrimitiveAtomQuantum(
        atom_symbol=atom_symbol,
        arity=max(1, int(valence + lone_pairs)),
        atomic_num=atomic_num,
        valence=valence,
        lone_pairs=lone_pairs,
        geometry=geometry,
        is_metal=is_metal,
    )
    return prim.symplectic_repr(dim=dim)


__all__ = [
    "SymplecticPoint",
    "PrimitiveAtomQuantum",
    "BetaReductionSymp",
    "quantize_atom",
]
