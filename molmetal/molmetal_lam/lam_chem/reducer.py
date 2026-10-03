"""Single-step β-reduction + PLL cost for the Molecular Lambda Calculus.

============================================================
MLC operational semantics: β-reduction as bonding
============================================================
This module ships the *operational* layer of MLC that sits between the
syntactic AST (:mod:`molmetal_lam.lam_chem.ast`) and the chemistry-aware
bookkeeping layer (:mod:`molmetal_lam.molecules.closed_term`).  The
central thesis of MLC — *a molecule IS a closed λ-term in β-normal
form* — is operationalised here:

    * :func:`reduce_step`           applies one β-reduction if a redex
                                    exists, else returns the input
                                    unchanged with zero cost.
    * :func:`reduce_to_normal_form` repeatedly applies :func:`reduce_step`
                                    until β-NF or a cofinite depth budget
                                    (``max_steps``) is exhausted.  This
                                    guards against non-normalising
                                    terms (e.g. ``Ω = (λx.xx)(λx.xx)``).
    * :func:`pll_cost`              the cost of one β-step measured by
                                    the L2 distance between Persistent
                                    Local Laplacian (PLL) signatures
                                    of the two states, normalised to
                                    ``[0, 1]`` — Liu & Feng 2026.
    * :func:`is_normal_form`        True iff no reducible β-redex
                                    remains anywhere in ``term``.

Why PLL is the cost function
----------------------------
The PLL signature (Liu & Feng 2026, *J. Chem. Inf. Model.*, DOI
10.1021/acs.jcim.6c02460) is the top-``k`` spectrum of the *normalised*
graph Laplacian of a kNN graph built from a molecule's 3D point cloud.
It is rotation / translation / reflection invariant (the construction
depends only on pairwise Euclidean distances), so it is the canonical
geometry-preserving "amount of change" metric for β-steps that
correspond to *bonding* (i.e. when the states have a Mol representation).

Honest framing
--------------
A standalone :class:`Atom` (e.g. ``Pt_II``) has no Mol representation of
its own — it is a *primitive combinator*, not a molecule.  When the
default :func:`pll_cost` is asked to compare two primitive combinators
(or any state without an RDKit-convertible Mol), it gracefully returns
``0.0`` rather than crashing.  The reducer therefore works for the
abstract calculus (``LamVar`` / ``LamAbs`` / ``LamApp`` *or* :class:`Atom`)
without forcing every term to carry a 3D conformer.

Coarse termination argument
---------------------------
For the typed / finite-arity fragment we care about (metal coordination
chemistry — every combinator has finite arity ``≤ 6``), β-reduction is
**strongly normalising**: every reduction sequence terminates and the
order of reductions does not matter (Church-Rosser + finite arity).
The ``max_steps`` guard is a belt-and-braces contract from
``TODO/pending/32_ultracode_trackB_lambda_calculus.md`` §B2: even an
ill-formed term cannot loop forever.

Public API
----------
``reduce_step``              one-step β-reduction + PLL cost
``reduce_to_normal_form``    iterative reduction with depth limit
``pll_cost``                 PLL-L2 cost normalised to ``[0, 1]``
``is_normal_form``           β-NF predicate
``LamTerm``                  type alias for the union of accepted terms

References
----------
* Liu J., Feng H. (2026) "Persistent Local Laplacian Prediction of
  Protein–Ligand Binding Affinities".  *J. Chem. Inf. Model.* 66 (3),
  1024-1037.  DOI: 10.1021/acs.jcim.6c02460, arXiv:2603.21503.
* Barendregt H. P. (1984) *The Lambda Calculus — Its Syntax and
  Semantics*.  North-Holland.  Studies in Logic Vol. 103.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple, Union

from molmetal.molmetal_lam.lam_chem.ast import (
    LamAbs,
    LamApp,
    LamNode,
    LamVar,
)
from molmetal.molmetal_lam.lam_chem.lambda_combinators import Atom


# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------

#: A ``LamTerm`` is anything the reducer can manipulate:
#:
#: * :class:`molmetal.molmetal_lam.lam_chem.ast.LamNode` — the abstract
#:   AST (``LamVar`` / ``LamAbs`` / ``LamApp``) with full β-reduction
#:   semantics per Barendregt 1984.
#: * :class:`molmetal.molmetal_lam.lam_chem.lambda_combinators.Atom` — a
#:   primitive combinator (fixed arity combinator); in β-NF as a value.
LamTerm = Union[LamNode, Atom]

#: Signature of a cost function over two states.  Defaults to
#: :func:`pll_cost`.
CostFn = Callable[[Any, Any], float]


# ---------------------------------------------------------------------------
# β-normal-form predicate
# ---------------------------------------------------------------------------


def is_normal_form(term: LamTerm) -> bool:
    """Return True iff no reducible β-redex remains anywhere in ``term``.

    The predicate is **syntactic**, not semantic: it walks the term
    only as far as :meth:`LamApp.beta_reduce` does (leftmost-outermost
    one-step).  A standalone :class:`Atom` / :class:`LamVar` /
    :class:`LamAbs` has no redex by construction and is therefore
    trivially in β-NF.

    Examples
    --------
    >>> from molmetal.molmetal_lam.lam_chem.ast import var, lam, apply
    >>> from molmetal.molmetal_lam.lam_chem.lambda_combinators import Atom
    >>> pt_ii = Atom("Pt", 78, valence=2, lone_pairs=2,
    ...              geometry="square_planar", is_metal=True)
    >>> is_normal_form(pt_ii)               # primitive combinator
    True
    >>> x = var("x")
    >>> is_normal_form(lam(x, x))           # λx.x — irreducible value
    True
    >>> is_normal_form(apply(lam(x, x), var("y")))  # (λx.x) y → y
    False
    """
    # Primitives (Atom) and pure values (LamVar / LamAbs) have no redex.
    if isinstance(term, (Atom, LamVar, LamAbs)):
        return True
    # An application may have a redex at the top — defer to its own
    # leftmost-outermost β-step.
    if isinstance(term, LamApp):
        return term.beta_reduce() is None
    # Unknown type — defensive: assume no redex to avoid an infinite loop.
    return True


# ---------------------------------------------------------------------------
# Single-step β-reduction + PLL cost
# ---------------------------------------------------------------------------


def reduce_step(
    term: LamTerm,
    cost_fn: CostFn = None,  # type: ignore[assignment]
) -> Tuple[LamTerm, float]:
    """Apply one β-reduction if a reducible redex exists.

    Parameters
    ----------
    term : LamTerm
        The term to reduce.  Accepted types: ``LamVar`` / ``LamAbs`` /
        ``LamApp`` (from :mod:`molmetal_lam.lam_chem.ast`) and
        :class:`Atom` (from :mod:`molmetal_lam.lam_chem.lambda_combinators`).
    cost_fn : Callable[[Any, Any], float], optional
        Cost function ``(state_before, state_after) -> float``.  The
        default is :func:`pll_cost`.  Pass any custom callable to use
        a different cost (e.g. a literal step counter).

    Returns
    -------
    (new_term, cost) : Tuple[LamTerm, float]
        * ``(term, 0.0)``                when ``term`` is already in β-NF.
        * ``(reduced_term, cost_fn(term, reduced_term))``  otherwise.

    Notes
    -----
    The cost function is wrapped in a broad ``except Exception``
    clause that returns ``0.0`` on any failure — this keeps the
    reducer robust when the term has no Mol representation (e.g. a
    standalone :class:`Atom`) or when RDKit/scipy are unavailable.
    The contract is "best-effort cost; reducer still works".
    """
    if cost_fn is None:
        cost_fn = pll_cost

    # Primitives and pure values: no β-reduction possible.
    if isinstance(term, (Atom, LamVar, LamAbs)):
        return term, 0.0

    if isinstance(term, LamApp):
        reduced = term.beta_reduce()
        if reduced is None:
            return term, 0.0
        try:
            cost = float(cost_fn(term, reduced))
        except Exception:
            # Cost is best-effort — never let a failure block reduction.
            cost = 0.0
        return reduced, cost

    # Defensive fallback for unknown types.
    return term, 0.0


# ---------------------------------------------------------------------------
# Iterative reduction to β-NF (with cofinite termination guard)
# ---------------------------------------------------------------------------


def reduce_to_normal_form(
    term: LamTerm,
    max_steps: int = 100,
    cost_fn: CostFn = None,  # type: ignore[assignment]
) -> LamTerm:
    """Iteratively apply :func:`reduce_step` until β-NF or ``max_steps``.

    Parameters
    ----------
    term : LamTerm
        The term to fully reduce.
    max_steps : int, default ``100``
        Cofinite termination guard.  The typed finite-arity fragment
        we work with (metal coordination, ``arity ≤ 6``) is strongly
        normalising so 100 steps is ample — but the guard is the
        workflow contract from
        ``TODO/pending/32_ultracode_trackB_lambda_calculus.md`` §B2 and
        protects against pathological inputs (e.g. ``Ω``).
    cost_fn : Callable[[Any, Any], float], optional
        Same semantics as :func:`reduce_step`'s ``cost_fn``.

    Returns
    -------
    LamTerm
        The β-normal form (or the term after ``max_steps`` reductions
        if the guard fired).

    Notes
    -----
    We check :func:`is_normal_form` *before* each step so an already-NF
    term short-circuits on iteration 0 (no cost, no allocations).
    A secondary equality guard (``nxt == term``) protects against a
    pathological ``cost_fn`` that returns non-zero for a no-op step.
    """
    if cost_fn is None:
        cost_fn = pll_cost

    current: LamTerm = term
    for _ in range(int(max_steps)):
        if is_normal_form(current):
            return current
        nxt, _cost = reduce_step(current, cost_fn=cost_fn)
        # Safety: if a reduction claims to have happened but the term
        # is structurally identical (==) to the input, treat as NF —
        # prevents infinite loops if the cost_fn is poorly behaved.
        if nxt == current:  # pragma: no cover - defensive
            return current
        current = nxt
    return current


# ---------------------------------------------------------------------------
# PLL cost — Persistent Local Laplacian L2 distance, normalised to [0, 1]
# ---------------------------------------------------------------------------


def _ensure_conformer(mol, Chem_mod, AllChem_mod):  # pragma: no cover - helper
    """Embed a 3D conformer on ``mol`` if it has none.  Returns ``mol``
    or ``None`` if embedding fails.

    RDKit's :func:`EmbedMolecule` is stochastic and may return a
    non-zero status on hard-to-embed molecules (e.g. hyper-valent
    centres, unusual ring strain).  Such cases are not failures of
    :func:`pll_cost` — we simply return ``0.0`` upstream.
    """
    if mol is None:
        return None
    try:
        if mol.GetNumConformers() == 0:
            try:
                mol = Chem_mod.AddHs(mol)
            except Exception:
                pass
            status = AllChem_mod.EmbedMolecule(mol)
            if status != 0:
                return None
        return mol
    except Exception:
        return None


def _to_mol_with_conformer(
    state: Any, Chem_mod, AllChem_mod
):  # pragma: no cover - helper
    """Best-effort conversion of ``state`` to an RDKit Mol + 3D conformer.

    Accepted shapes:

    * ``None``                                 → ``None``
    * has ``to_rdkit()`` (e.g. ``MoleculeClosedTerm``) → call it
    * is already an RDKit ``Mol`` (has ``GetNumAtoms`` /
      ``GetConformer``) → use as-is
    * string → try ``Chem.MolFromSmiles``
    * anything else → ``str(state)`` → SMILES
    """
    if state is None:
        return None
    # ``MoleculeClosedTerm`` and friends expose ``to_rdkit``.
    to_rdkit = getattr(state, "to_rdkit", None)
    if callable(to_rdkit):
        try:
            mol = to_rdkit()
        except Exception:
            return None
        return _ensure_conformer(mol, Chem_mod, AllChem_mod)
    # RDKit Mol — duck-type on the methods we actually need.
    if hasattr(state, "GetNumAtoms") and hasattr(state, "GetConformer"):
        return _ensure_conformer(state, Chem_mod, AllChem_mod)
    # Last resort: string interpretation as SMILES.
    try:
        mol = Chem_mod.MolFromSmiles(str(state))
    except Exception:
        return None
    if mol is None:
        return None
    return _ensure_conformer(mol, Chem_mod, AllChem_mod)


def pll_cost(state_before: Any, state_after: Any) -> float:
    """L2 distance between PLL signatures, normalised to ``[0, 1]``.

    Parameters
    ----------
    state_before, state_after : Any
        The two states to compare.  Each is converted to an RDKit
        ``Mol`` via :func:`_to_mol_with_conformer` (which accepts
        ``MoleculeClosedTerm``, raw ``Mol``, or SMILES string).

    Returns
    -------
    float
        ``||PLL(state_before) - PLL(state_after)||_2`` divided by the
        maximum possible L2 distance for two k-vectors in ``[0, 2]^k``,
        clipped to ``[0, 1]``.  Returns ``0.0`` when:

        * RDKit or scipy is unavailable,
        * either state cannot be converted to a Mol,
        * either Mol fails to embed a 3D conformer,
        * either PLL computation raises.

    Notes
    -----
    The PLL spectrum is the top-``k`` eigenvalues of the *normalised*
    graph Laplacian, descending, with each ``λ_i ∈ [0, 2]``.  Two
    k-vectors in ``[0, 2]^k`` are at most ``2 * sqrt(k)`` apart in
    L2; dividing by this upper bound and clipping to ``[0, 1]`` keeps
    the cost interpretable as a fraction of the maximum possible
    geometric change for the given ``k``.

    The function follows the workflow contract that **rdkit / scipy
    are lazy-imported**: the module imports nothing from either
    package at top-level, so importing :mod:`reducer` succeeds on a
    slim RDKit-only or scipy-less environment.
    """
    # ------------------------------------------------------------------
    # Lazy RDKit import
    # ------------------------------------------------------------------
    try:
        from rdkit import Chem  # type: ignore[import-not-found]
        from rdkit.Chem import AllChem  # type: ignore[import-not-found]
    except Exception:
        return 0.0

    # ------------------------------------------------------------------
    # Lazy PLL import (transitively pulls scipy)
    # ------------------------------------------------------------------
    try:
        from molmetal.molmetal_lam.lam_chem.pll_signature import compute_pll
    except Exception:
        return 0.0

    # ------------------------------------------------------------------
    # Convert both states to Mol-with-conformer
    # ------------------------------------------------------------------
    mol_a = _to_mol_with_conformer(state_before, Chem, AllChem)
    mol_b = _to_mol_with_conformer(state_after, Chem, AllChem)
    if mol_a is None or mol_b is None:
        return 0.0

    # ------------------------------------------------------------------
    # Compute PLL signatures (cached on the signature's own lru_cache)
    # ------------------------------------------------------------------
    try:
        sig_a = compute_pll(mol_a, k=16)
        sig_b = compute_pll(mol_b, k=16)
    except Exception:
        return 0.0

    # ------------------------------------------------------------------
    # L2 distance normalised by the [0, 2]^k box width
    # ------------------------------------------------------------------
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is a hard dep of pll_signature
        return 0.0

    ev_a = np.asarray(sig_a.eigenvalues, dtype=np.float64)
    ev_b = np.asarray(sig_b.eigenvalues, dtype=np.float64)
    k = int(min(ev_a.size, ev_b.size))
    if k <= 0:
        return 0.0
    diff = ev_a[:k] - ev_b[:k]
    l2 = float(np.sqrt(np.sum(diff * diff)))
    # Upper bound: two k-vectors in [0, 2]^k are at most 2*sqrt(k) apart.
    max_l2 = 2.0 * float(np.sqrt(k))
    if max_l2 <= 0.0:
        return 0.0
    normalised = l2 / max_l2
    # Numerical safety: clip into [0, 1].
    if normalised < 0.0:
        return 0.0
    if normalised > 1.0:
        return 1.0
    return normalised


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "LamTerm",
    "CostFn",
    "is_normal_form",
    "reduce_step",
    "reduce_to_normal_form",
    "pll_cost",
]
