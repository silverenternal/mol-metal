"""Lambda-calculus AST nodes for the Molecular Lambda Calculus (MLC).

This file implements the *syntactic* layer of MLC (λ-terms as immutable
trees). Operational semantics (β-reduction → chemistry, α-conversion →
equivariance) are encoded as methods on the dataclasses.

Design choices
--------------
1. ``frozen=True`` so terms are hashable → safe for MCTS transposition
   tables and proof-search memoisation.
2. Methods return *new* trees rather than mutating in place — keeps the
   algebra referentially transparent.
3. ``to_normal_form`` is bounded by ``max_depth`` to guarantee
   termination on pathological terms (e.g. ``Ω = (λx.xx)(λx.xx)``).
4. ``subst`` implements the usual capture-avoiding substitution: when
   substituting ``n`` for ``x`` in ``λy.M`` and ``y`` is free in ``n``,
   we rename ``y`` to a fresh variable first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Optional, Set, Union

# Global counter used to mint fresh variables during capture-avoiding
# substitution. Wrapping in a function lets us reset / scope it per-call
# without leaking state across tests.
_FRESH_COUNTER = 0


def _fresh_var(hint: str = "v") -> str:
    """Mint a globally-fresh variable name based on ``hint``."""
    global _FRESH_COUNTER
    _FRESH_COUNTER += 1
    return f"__{hint}{_FRESH_COUNTER}"


def _alpha_rename(node: "LamNode", avoid: Set[str]) -> "LamNode":
    """Alpha-conversion: walk ``node`` and rename binders so that no
    binder shadows any name in ``avoid`` or in any enclosing binder.

    Returns a structurally equivalent (alpha-equivalent) term whose
    binders are guaranteed not to clash with the ``avoid`` set.
    """
    if isinstance(node, LamVar):
        return node
    if isinstance(node, LamAbs):
        if node.var.name in avoid:
            fresh = LamVar(name=_fresh_var(node.var.name))
            # Replace the binder, then recurse so the body uses the
            # new name in place of the old one.
            renamed_body = node.body.subst(node.var.name, fresh)
            return LamAbs(var=fresh, body=_alpha_rename(renamed_body, avoid | {fresh.name}))
        return LamAbs(
            var=node.var,
            body=_alpha_rename(node.body, avoid | {node.var.name}),
        )
    # LamApp
    assert isinstance(node, LamApp)
    return LamApp(
        func=_alpha_rename(node.func, avoid),
        arg=_alpha_rename(node.arg, avoid),
    )


# ---------------------------------------------------------------------------
# Forward-reference type alias
# ---------------------------------------------------------------------------
# A ``LamNode`` is one of ``LamVar | LamAbs | LamApp``. Because the
# dataclasses reference each other, we define the union as a string
# alias for forward references inside ``__future__`` annotations.
LamNode = Union["LamVar", "LamAbs", "LamApp"]


# ---------------------------------------------------------------------------
# LamVar
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LamVar:
    """A variable, e.g. ``T`` (temperature), ``cat`` (catalyst conc).

    Parameters
    ----------
    name : str
        Identifier — must start with a letter or underscore.
    domain : str, default 'real'
        Optional *type-tag* (e.g. ``'real'``, ``'molar'``, ``'kelvin'``).
        Not enforced structurally; left to the type-layer.
    """

    name: str
    domain: str = "real"

    def to_string(self) -> str:  # noqa: D401
        """Pretty-print — variables just print as their ``name``."""
        return self.name

    def free_vars(self) -> FrozenSet[str]:
        """A bare variable is itself a free variable."""
        return frozenset({self.name})

    def subst(self, x: str, n: "LamNode") -> "LamNode":
        """Substitute ``n`` for free occurrences of ``x``."""
        if self.name == x:
            return n
        return self

    def beta_reduce(self) -> Optional["LamNode"]:
        """A bare variable is already a value — no β-reduction possible."""
        return None

    def to_normal_form(self, max_depth: int = 256) -> "LamNode":
        """A variable is in normal form by definition."""
        return self


# ---------------------------------------------------------------------------
# LamAbs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LamAbs:
    """A lambda-abstraction, e.g. ``λx.M``.

    The ``var`` is the binder; occurrences of ``var.name`` inside
    ``body`` are *bound*, not free.
    """

    var: LamVar
    body: LamNode

    def to_string(self) -> str:
        """LISP-style: ``λx.M``."""
        return f"λ{self.var.name}.{self.body.to_string()}"

    def free_vars(self) -> FrozenSet[str]:
        """Free variables = (free in body) minus the binder."""
        return self.body.free_vars() - {self.var.name}

    def subst(self, x: str, n: "LamNode") -> "LamNode":
        """Capture-avoiding substitution into the body.

        Two capture scenarios must be avoided:

        1. ``self.var.name`` is *free* in ``n``: the body of ``n``
           could contain an occurrence that would be captured by the
           binder ``self.var``. We rename the body of ``n`` away from
           ``self.var.name``.
        2. ``self.var.name`` is *bound* in ``n`` (via some nested
           ``λself.var.name.…``): after substitution that inner
           binder would shadow the outer one. We alpha-rename the
           binders in ``n`` first.
        """
        if self.var.name == x:
            # ``x`` is the binder here, so it shadows ``n`` entirely
            # inside the body. Just return ``self``.
            return self

        # Step 1: alpha-rename any binders inside ``n`` that would
        # shadow ``self.var.name``.
        n = _alpha_rename(n, avoid={self.var.name})

        free_in_n = n.free_vars()
        if self.var.name in free_in_n:
            # Capture would still occur: rename the binder to a fresh var.
            fresh = LamVar(name=_fresh_var(self.var.name))
            renamed_body = self.body.subst(self.var.name, fresh)
            return LamAbs(var=fresh, body=renamed_body.subst(x, n))
        return LamAbs(var=self.var, body=self.body.subst(x, n))

    def beta_reduce(self) -> Optional["LamNode"]:
        """An abstraction is a value — no β-reduction at the top level."""
        return None

    def to_normal_form(self, max_depth: int = 256) -> "LamNode":
        """Recursively normalise the body; the abstraction itself stays."""
        if max_depth <= 0:
            return self
        return LamAbs(var=self.var, body=self.body.to_normal_form(max_depth - 1))


# ---------------------------------------------------------------------------
# LamApp
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LamApp:
    """Function application, e.g. ``(M N)``.

    Reduction rule:    (λx.M) N  →  M[x := N]
    """

    func: LamNode
    arg: LamNode

    def to_string(self) -> str:
        """LISP-style: ``(M N)``."""
        return f"({self.func.to_string()} {self.arg.to_string()})"

    def free_vars(self) -> FrozenSet[str]:
        """Union of free variables in ``func`` and ``arg``."""
        return self.func.free_vars() | self.arg.free_vars()

    def subst(self, x: str, n: "LamNode") -> "LamNode":
        """Substitute through both sides of the application."""
        return LamApp(
            func=self.func.subst(x, n),
            arg=self.arg.subst(x, n),
        )

    def beta_reduce(self) -> Optional["LamNode"]:
        """One-step leftmost β-reduction.

        * If the left side is an abstraction, perform β-reduction.
        * Else, try reducing the left side first (normal-order).
        """
        if isinstance(self.func, LamAbs):
            # (λx.M) N  →  M[x := N]
            return self.func.body.subst(self.func.var.name, self.arg)

        reduced_func = self.func.beta_reduce()
        if reduced_func is not None:
            return LamApp(func=reduced_func, arg=self.arg)

        reduced_arg = self.arg.beta_reduce()
        if reduced_arg is not None:
            return LamApp(func=self.func, arg=reduced_arg)

        return None  # already in β-normal form

    def to_normal_form(self, max_depth: int = 256) -> "LamNode":
        """Iteratively apply :meth:`beta_reduce` until a fixed point or
        the depth budget is exhausted.

        We deliberately loop instead of recursing so that ``Ω``
        (``(λx.xx)(λx.xx)``) doesn't blow the Python stack.
        """
        current: LamNode = self
        for _ in range(max_depth):
            nxt = current.beta_reduce()
            if nxt is None:
                return current
            current = nxt
        return current


# ---------------------------------------------------------------------------
# Module-level convenience helpers
# ---------------------------------------------------------------------------
# Note: avoid Python builtins ``abs``/``app`` in the names of these
# shorthand constructors to keep IDE auto-completion and ``from … import
# abs`` happy. They are exposed as ``lam`` (abstraction) and ``apply``
# (application) at the package level.
def var(name: str, domain: str = "real") -> LamVar:
    """Shorthand constructor for :class:`LamVar`."""
    return LamVar(name=name, domain=domain)


def lam(x: LamVar, body: LamNode) -> LamAbs:
    """Shorthand constructor for :class:`LamAbs` (avoids clobbering ``abs``)."""
    return LamAbs(var=x, body=body)


def apply(f: LamNode, a: LamNode) -> LamApp:
    """Shorthand constructor for :class:`LamApp` (avoids clobbering ``app``)."""
    return LamApp(func=f, arg=a)


__all__ = [
    "LamVar",
    "LamAbs",
    "LamApp",
    "LamNode",
    "var",
    "lam",
    "apply",
]