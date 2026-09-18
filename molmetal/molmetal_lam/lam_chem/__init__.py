"""lam_chem — Lambda-calculus AST + PySR heuristic wrapper.

============================================================
Lambda-calculus AST for chemistry
============================================================
This module provides a minimal, purely-functional representation of the
(untyped) lambda-calculus. Each :class:`LamNode` is a *frozen* dataclass,
so the AST can be safely hashed, deep-copied, and used as dictionary
keys in MCTS / proof-search contexts.

AST nodes
---------
    LamVar(name, domain='real')
        A variable such as ``T`` (temperature), ``cat`` (catalyst conc),
        ``t`` (time), or ``azide_conc``. The optional ``domain`` string
        is a *type-tag* placeholder that the type layer can use later
        (e.g. ``'real'``, ``'molar'``, ``'kelvin'``, ``'seconds'``).

    LamAbs(var, body)
        A lambda-abstraction, e.g. ``λx.M``.

    LamApp(func, arg)
        Function application, e.g. ``(M N)``.

Operations
----------
* :meth:`LamNode.to_string`           LISP-style pretty-print
* :meth:`LamNode.free_vars`           set of free variables
* :meth:`LamNode.subst(x, n)`         capture-avoiding substitution
* :meth:`LamNode.beta_reduce`         one-step β-reduction (leftmost)
* :meth:`LamNode.to_normal_form`      full β-NF, with depth limit

PySR / heuristic wrapper
------------------------
:class:`HeuristicRegressor` is a thin wrapper around PySR symbolic
regression that gracefully degrades to scikit-learn when the Julia
backend is unavailable. The fallback uses a small
:class:`~sklearn.ensemble.RandomForestRegressor` (or
:class:`~sklearn.linear_model.Ridge` for tiny problems).
"""

from .ast import LamAbs, LamApp, LamNode, LamVar, var, lam, apply
from .pysr_wrapper import HeuristicRegressor
from .well_formedness import (
    WellFormednessError,
    assert_well_formed,
    check_arity_conservation,
    check_beta_normal_form,
    check_closed_term,
)
from .cisplatin_builder import build_cisplatin

__all__ = [
    "LamVar",
    "LamAbs",
    "LamApp",
    "LamNode",
    "HeuristicRegressor",
    "var",
    "lam",
    "apply",
    "WellFormednessError",
    "check_closed_term",
    "check_arity_conservation",
    "check_beta_normal_form",
    "assert_well_formed",
    "build_cisplatin",
    "__version__",
]

__version__ = "0.0.1-lamchem"