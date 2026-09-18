"""Synthesis = β-reduction sequence (forward) and β-expansion (retro).

This module implements the **Synthesis layer** of the Molecular Lambda
Calculus (MLC), formalized in
``TODO/13_lambda_clickchem/molecular_lambda_calculus.md`` §5.

Core thesis
-----------
* **Forward synthesis** = β-reduction sequence: starting from a pool of
  *starting materials* (closed λ-terms), repeatedly fire a reaction rule
  (= one β-reduction) to produce intermediates and eventually the target.
* **Retrosynthesis** = β-expansion: from the target, identify each
  *redex*-pattern that could have produced it and enumerate the precursor
  term that would have been the redex's reductant.

We expose:

* :class:`ReactionPattern` — a ``(rule, precursor)`` tuple returned by
  :func:`retrosynthesize`.
* :class:`SynthesisPath`   — the witness of a successful forward search:
  the target, the sequence of ``(rule, intermediate)`` steps, and the
  starting materials.  Carries :meth:`to_lambda_expr` (the "killer
  figure" string) and :meth:`is_mass_balanced` (end-to-end invariant).
* :func:`synthesize`       — BFS over β-reduction applications.
* :func:`retrosynthesize`  — β-expansion enumeration.

The implementation uses RDKit only at the bond-formation step (via
:class:`~molmetal_lam.reactions.beta_reductions.ReactionRule`) and is
importable without RDKit for the higher-level control flow.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import combinations
from typing import (
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

from molmetal_lam.atoms.combinators import Atom
from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
from molmetal_lam.reactions.beta_reductions import ReactionRule


# ---------------------------------------------------------------------------
# Pattern type — a (rule, precursor) pair returned by retrosynthesis
# ---------------------------------------------------------------------------


#: A ``ReactionPattern`` is a candidate β-expansion: applying ``rule`` to
#: ``precursor`` (or its fragments) yields a successor that includes
#: ``target``.  This is the standard ML/retrosynthesis output type.
ReactionPattern = Tuple[ReactionRule, MoleculeClosedTerm]


# ---------------------------------------------------------------------------
# SynthesisPath — the witness of a successful forward search
# ---------------------------------------------------------------------------


@dataclass
class SynthesisPath:
    """A synthesis path = a β-reduction sequence.

    Attributes
    ----------
    target : MoleculeClosedTerm
        The closed term we want to construct (β-normal form).
    steps : list[(ReactionRule, MoleculeClosedTerm)]
        The sequence of reactions applied, paired with the *product*
        intermediate that the rule produced.  ``steps[0]`` is the first
        reaction and ``steps[-1][1]`` is the target (alpha-equivalent).
    starting_materials : list[MoleculeClosedTerm]
        The closed terms available as reactants (the BFS frontier's
        initial nodes).  The order is the user-supplied order — it is
        preserved for reproducibility of the printed λ-expression.

    The class is intentionally a plain dataclass so it can be pickled,
    logged, and rendered by the pipeline / paper-figure code.
    """

    target: MoleculeClosedTerm
    steps: List[Tuple[ReactionRule, MoleculeClosedTerm]] = field(
        default_factory=list
    )
    starting_materials: List[MoleculeClosedTerm] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Pretty-printing — the "killer figure" of the paper
    # ------------------------------------------------------------------

    def to_lambda_expr(self) -> str:
        """Render this synthesis as a human-readable λ-expression.

        Format
        ------
        ``((<rule> <SMILES_1>) <SMILES_2>) -> <target_smiles>``

        For multi-step syntheses, the steps are separated by ``>>`` (a
        standard chemistry reaction arrow).  The expression reproduces
        the example from the formalization memo::

            ((CuAAC azide_NH3) alkyne_CC) -> triazole_CC_NH3

        where ``azide_NH3`` / ``alkyne_CC`` / ``triazole_CC_NH3`` are
        short SMILES fragments abbreviating the reactant / product
        identities.

        Example
        -------
        >>> path.to_lambda_expr()        # doctest-style usage
        'CuAAC(N=[N+]=[N-], C#C) -> c1cn(...)1'
        """
        if not self.steps:
            return f"(no_synthesis) -> {self._smi(self.target)}"

        # The "outer" reactant-pair is the final step; the "inner" pair
        # is the first step.  We render in chemistry-rewind order: the
        # outermost application corresponds to the first reaction fired.
        parts: List[str] = []
        for rule, _intermediate in self.steps:
            parts.append(rule.name)
        # Compress consecutive identical rule names ("CuAAC > CuAAC").
        # Not strictly necessary, but it makes the expression readable
        # when the same rule fires multiple times.
        compressed: List[str] = []
        for name in parts:
            if compressed and compressed[-1] == name:
                continue
            compressed.append(name)

        # Render the left-associated application with all starting
        # materials as the inner arguments.  We use the first
        # len(starting_materials) SMILES as the reactant slot names,
        # abbreviating them.
        smi_names = [self._short_smi(m) for m in self.starting_materials]
        if not smi_names:
            inner = "()"
        elif len(smi_names) == 1:
            inner = f"({compressed[0]} {smi_names[0]})"
        else:
            head = f"({compressed[0]} {smi_names[0]})"
            tail = " ".join(smi_names[1:])
            inner = f"({head} {tail})"
        return f"{inner} -> {self._smi(self.target)}"

    # ------------------------------------------------------------------
    # Mass balance — the end-to-end invariant
    # ------------------------------------------------------------------

    def is_mass_balanced(self, tol: int = 0) -> bool:
        """Return ``True`` iff the synthesis is heavy-atom balanced.

        "Mass balance" here means: the multiset of heavy atoms in the
        starting materials equals (modulo a tolerance) the multiset of
        heavy atoms in the final target.  We use the rules'
        :attr:`ReactionRule.stoichiometry` to fold in the net atom
        changes produced by each step, so a click chemistry step with
        stoichiometry ``{}`` contributes nothing and a condensation
        with ``{"H2O": -1}`` would subtract one water.

        Parameters
        ----------
        tol : int, default 0
            Allowed absolute deviation in any element's count.  Useful
            when the stoichiometry dict is approximate (e.g. a coarse
            yield model).

        Returns
        -------
        bool
            ``True`` iff every element's |predicted - observed| ≤ ``tol``.
        """
        try:
            predicted = _predicted_atom_counts(self.starting_materials)
        except Exception:
            return False
        # Apply each step's stoichiometry (net atom changes).
        for rule, _ in self.steps:
            for sym, delta in rule.stoichiometry.items():
                predicted[sym] = predicted.get(sym, 0) + delta
        try:
            observed = _observed_atom_counts(self.target)
        except Exception:
            return False
        return _balanced(predicted, observed, tol=tol)

    def mass_balance_report(self) -> Dict[str, int]:
        """Return the per-element atom-count difference ``predicted - observed``.

        Empty dict ⇒ the synthesis is mass-balanced to within rounding.
        Non-empty dict ⇒ the keys / values describe the imbalance.
        """
        predicted = _predicted_atom_counts(self.starting_materials)
        for rule, _ in self.steps:
            for sym, delta in rule.stoichiometry.items():
                predicted[sym] = predicted.get(sym, 0) + delta
        observed = _observed_atom_counts(self.target)
        diff: Dict[str, int] = {}
        for sym in set(predicted) | set(observed):
            d = predicted.get(sym, 0) - observed.get(sym, 0)
            if d != 0:
                diff[sym] = d
        return diff

    # ------------------------------------------------------------------
    # Cosmetics
    # ------------------------------------------------------------------

    def _smi(self, m: MoleculeClosedTerm) -> str:
        """Canonical SMILES with lazy RDKit fallback."""
        try:
            return m.canonical_smiles()
        except Exception:
            return self._short_smi(m)

    def _short_smi(self, m: MoleculeClosedTerm) -> str:
        """Abbreviation for nested rendering (last-resort SMILES)."""
        if m.source_smiles:
            return m.source_smiles
        try:
            return m.canonical_smiles()
        except Exception:
            # Pure-Python fallback — concatenate element symbols.
            return "".join(a.symbol for a in m.atoms)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        n_steps = len(self.steps)
        return (
            f"SynthesisPath(target={self._short_smi(self.target)!r}, "
            f"n_steps={n_steps}, "
            f"starting_materials={len(self.starting_materials)})"
        )


# ---------------------------------------------------------------------------
# Forward synthesis — BFS over β-reductions
# ---------------------------------------------------------------------------


def synthesize(
    target: MoleculeClosedTerm,
    starting_materials: Sequence[MoleculeClosedTerm],
    rules: Sequence[ReactionRule],
    max_depth: int = 10,
    *,
    is_goal: Optional[Callable[[MoleculeClosedTerm], bool]] = None,
) -> Optional[List[ReactionRule]]:
    """Forward synthesis = BFS over β-reduction applications.

    Algorithm
    ---------
    We perform a *breadth-first search* in the space of β-reduction
    applications.  The initial frontier is the multiset of starting
    materials; at each step we pop the smallest-cost frontier and try
    every (rule, pair-of-reactants) combination.  When a rule fires on a
    pair and produces a product alpha-equivalent to the target (or
    satisfying :func:`is_goal`), we return the sequence of rules that
    produced it.

    Parameters
    ----------
    target : MoleculeClosedTerm
        The closed term we want to derive (β-normal form).
    starting_materials : sequence of MoleculeClosedTerm
        The educts available at depth 0.  All must be closed λ-terms.
    rules : sequence of ReactionRule
        The β-reduction rules available to the search.  Each rule's
        :meth:`reduce` interface is called once per (unordered) pair of
        frontier nodes.
    max_depth : int, default 10
        Hard cap on the number of β-reduction steps.  ``max_depth <= 0``
        returns ``None`` immediately (no derivations considered).
    is_goal : callable, optional
        A user-supplied goal predicate.  Defaults to
        ``target.alpha_equivalent(candidate)``.

    Returns
    -------
    list[ReactionRule] or None
        The shortest β-reduction sequence producing ``target``, or
        ``None`` if no derivation was found within ``max_depth`` steps.

    Notes
    -----
    * The function is a *pure* forward search: it does not call
      :func:`retrosynthesize`.  This mirrors the distinction between
      proof search (forward) and proof extraction (retro) in
      constructive type theory.
    * For the CuAAC / SPAAC / SPC / DielsAlder / ThiolEne registry of
      click rules, a derivation length of ``max_depth=5`` is typically
      sufficient for sub-200-Da targets.  Larger macrocycles may need
      ``max_depth >= 8``.
    """
    if max_depth <= 0:
        return None

    goal = is_goal or (lambda m: target.alpha_equivalent(m))

    # Trivial case: target is one of the starting materials.
    for sm in starting_materials:
        if goal(sm):
            return []

    # BFS frontier: each entry is (depth, accumulated_rule_sequence,
    # current_frontier_set).  We keep the frontier as a tuple of
    # canonical SMILES so the visited-set can deduplicate at the SMILES
    # level (alpha-equivalent molecules are conflated, which is the
    # intended semantics).
    FrontierEntry = Tuple[int, Tuple[ReactionRule, ...], Tuple[str, ...]]

    initial_smis = tuple(_smi(m) for m in starting_materials)
    queue: deque[FrontierEntry] = deque(
        [(0, (), initial_smis)]
    )
    visited: set[Tuple[str, ...]] = {initial_smis}

    while queue:
        depth, rule_seq, frontier_smis = queue.popleft()
        if depth >= max_depth:
            continue

        # Reconstruct the frontier MoleculeClosedTerms.  We rebuild them
        # from SMILES on demand (cheap; we don't mutate).
        frontier_terms = [_from_smi_lazy(s) for s in frontier_smis]

        # Try every (rule, unordered-pair-of-reactants) combination.
        for rule in rules:
            for a, b in combinations(frontier_terms, 2):
                products = _safe_reduce(rule, a, b)
                if not products:
                    continue
                for prod in products:
                    prod_smi = _smi(prod)
                    if goal(prod):
                        return list(rule_seq + (rule,))
                    # Build the next frontier: original frontier plus
                    # the new product, deduplicated by SMILES.
                    new_frontier = _dedup_frontier(
                        frontier_smis + (prod_smi,)
                    )
                    if new_frontier in visited:
                        continue
                    visited.add(new_frontier)
                    queue.append((
                        depth + 1,
                        rule_seq + (rule,),
                        new_frontier,
                    ))

    return None


# ---------------------------------------------------------------------------
# Retrosynthesis — β-expansion enumeration
# ---------------------------------------------------------------------------


def retrosynthesize(
    target: MoleculeClosedTerm,
    rules: Optional[Sequence[ReactionRule]] = None,
) -> List[ReactionPattern]:
    """Retrosynthesis = β-expansion.

    For each rule whose redex-pattern could have produced ``target`` (or a
    fragment of it), return the precursor term that would have been the
    reductant.  A precursor is one of the two educts that, combined with
    a partner under the same rule, fires the inverse β-expansion to
    yield the target.

    Algorithm
    ---------
    For each (rule, sub-fragment of target) pair we *try to expand*:

    1. Split ``target`` into two disjoint fragments (a coarse bisection;
       see :func:`_candidate_bisections`).
    2. For each rule, attempt to apply its forward reduction in a
       *reverse direction*: compute the rule's expected product SMILES
       from the candidate fragments, and check whether it matches the
       target.
    3. When it does, emit a :class:`ReactionPattern` ``(rule, precursor)``.

    The current implementation is conservative — it returns at most one
    pattern per (rule, sub-fragment) pair and only when the fragment is
    independently a closed λ-term.  A future revision can add a fragment-
    *aware* matcher that splits ``target`` along its Murcko scaffold.

    Parameters
    ----------
    target : MoleculeClosedTerm
        The molecule we want to de-construct.
    rules : sequence of ReactionRule, optional
        Rules to consider.  ``None`` ⇒ enumerate all rules in the
        :data:`molmetal_lam.reactions.beta_reductions.REACTION_RULES`
        registry (lazy import to avoid hard dependency).

    Returns
    -------
    list[ReactionPattern]
        A list of ``(rule, precursor)`` tuples, possibly empty if no
        rule can produce the target.
    """
    if rules is None:
        rules = _default_rules()

    patterns: List[ReactionPattern] = []
    seen: set[Tuple[str, str]] = set()

    for rule in rules:
        # 1) Try the rule's inverse on the target as a single educt.
        #    Most click rules are bi-molecular so this usually fails,
        #    but the hook is here for future unimolecular rules.
        for precursor in _safe_reduce_inverse_single(rule, target):
            key = (rule.name, _smi(precursor))
            if key in seen:
                continue
            seen.add(key)
            patterns.append((rule, precursor))

        # 2) For every bisection of the target into two fragments, try
        #    the rule as a bi-molecular forward reducer on the two
        #    fragments.  When the products equal the target (by α-
        #    equivalence) the *smaller* fragment is recorded as the
        #    precursor and the rule is the candidate transform.
        for frag_a, frag_b in _candidate_bisections(target):
            for prod in _safe_reduce(rule, frag_a, frag_b):
                if not prod.alpha_equivalent(target):
                    continue
                # Pick the smaller fragment as the "precursor" (the
                # other is assumed to be the partner / co-reactant).
                precursor = frag_a if frag_a.n_atoms <= frag_b.n_atoms else frag_b
                key = (rule.name, _smi(precursor))
                if key in seen:
                    continue
                seen.add(key)
                patterns.append((rule, precursor))

    return patterns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _smi(m: MoleculeClosedTerm) -> str:
    """Canonical SMILES with a value-signature fallback (no RDKit)."""
    try:
        return m.canonical_smiles()
    except Exception:
        try:
            sig = m._value_signature()
            return "|".join(str(x) for x in sig)
        except Exception:
            return repr(m)


def _from_smi_lazy(smi: str) -> MoleculeClosedTerm:
    """Parse ``smi`` into a MoleculeClosedTerm; passthrough if not parseable."""
    # If the SMILES contains the value-signature fallback delimiter, we
    # cannot reconstruct the term — return an empty term so the BFS
    # skips this candidate harmlessly.
    if "|" in smi and not smi.count("|") >= 1:
        return MoleculeClosedTerm()
    try:
        return MoleculeClosedTerm.from_smiles(smi, embed_3d=False)
    except Exception:
        return MoleculeClosedTerm()


def _safe_reduce(
    rule: ReactionRule,
    a: MoleculeClosedTerm,
    b: MoleculeClosedTerm,
) -> List[MoleculeClosedTerm]:
    """Apply ``rule.reduce((a, b))`` and return the products (or [])."""
    try:
        return rule.reduce((a, b))
    except Exception:
        return []


def _safe_reduce_inverse_single(
    rule: ReactionRule, target: MoleculeClosedTerm
) -> List[MoleculeClosedTerm]:
    """Placeholder inverse reducer for unimolecular rules.

    No unimolecular rules are defined in the current registry, so this
    always returns ``[]``.  The hook is provided so that future rules
    (e.g. sigmatropic rearrangements) can plug in here without changing
    the retrosynthesis dispatcher.
    """
    return []


def _dedup_frontier(frontier: Iterable[str]) -> Tuple[str, ...]:
    """Remove duplicates while preserving first-occurrence order."""
    seen: set[str] = set()
    out: List[str] = []
    for s in frontier:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return tuple(out)


def _candidate_bisections(
    target: MoleculeClosedTerm,
) -> List[Tuple[MoleculeClosedTerm, MoleculeClosedTerm]]:
    """Enumerate all (frag_a, frag_b) bisections of ``target``.

    For each pair of disjoint atom-subsets whose union equals
    ``target.atoms``, we materialise the corresponding fragment
    MoleculeClosedTerms.  The number of bisections is exponential in
    the atom count; for targets with more than 14 atoms we downsample
    by skipping bisections where both fragments are larger than 1 atom
    (a conservative heuristic that still finds single-fragment + whole
    pairings).

    Parameters
    ----------
    target : MoleculeClosedTerm
        The molecule to bisect.

    Returns
    -------
    list[(MoleculeClosedTerm, MoleculeClosedTerm)]
        A list of (fragment_a, fragment_b) pairs.  Each fragment is a
        closed term built from the subset of ``target.atoms`` plus the
        subset of bonds that lie inside it.
    """
    atoms = target.atoms
    n = len(atoms)
    if n < 2:
        return []

    out: List[Tuple[MoleculeClosedTerm, MoleculeClosedTerm]] = []
    # For each non-trivial subset of atom indices (excluding empty and
    # full) we construct a pair (subset, complement).
    n_full = 1 << n
    # Cap at 2^14 to keep this tractable; for larger molecules the BFS
    # will rely on starting_materials + already-known intermediates
    # rather than fragmenting the target.
    if n_full > (1 << 14):
        # Coarse: only enumerate 1-atom splits (used by single-fragment
        # retrosynthesis: "where could this one atom have come from?").
        bitmasks: Iterable[int] = (1 << i for i in range(n))
    else:
        bitmasks = range(1, n_full - 1)

    for mask in bitmasks:
        # Skip half the subsets (we'll get the complement when we
        # iterate the other half); keep only those whose lowest bit is
        # index 0 to enforce an ordering.
        if mask & 1 == 0:
            continue
        idx_a = [i for i in range(n) if (mask >> i) & 1]
        idx_b = [i for i in range(n) if not ((mask >> i) & 1)]
        if not idx_a or not idx_b:
            continue
        # Skip too-large pairs (cap on per-fragment size).
        if max(len(idx_a), len(idx_b)) > 8:
            continue
        frag_a = _subterm(target, idx_a)
        frag_b = _subterm(target, idx_b)
        if frag_a.n_atoms == 0 or frag_b.n_atoms == 0:
            continue
        out.append((frag_a, frag_b))
    return out


def _subterm(
    target: MoleculeClosedTerm, indices: Sequence[int]
) -> MoleculeClosedTerm:
    """Build the sub-term of ``target`` restricted to atoms at ``indices``.

    Bonds are kept iff both endpoints are in ``indices``.  The atom list
    is remapped: position ``i`` in ``indices`` becomes position ``i`` of
    the returned term.  Atom *identity* is preserved (we don't copy) so
    ledger operations downstream keep working.
    """
    index_set = set(indices)
    new_atoms = [target.atoms[i] for i in indices]
    new_bonds = [
        b for b in target.bonds
        if b.atom_a in index_set and b.atom_b in index_set
    ]
    return MoleculeClosedTerm(
        atoms=new_atoms,
        bonds=new_bonds,
        ledger=target.ledger,
        valence_used={i: target.valence_used.get(orig, 0)
                      for i, orig in enumerate(indices)},
        source_smiles=target.source_smiles,
        term=target.term,
    )


def _observed_atom_counts(m: MoleculeClosedTerm) -> Dict[str, int]:
    """Heavy-atom counts by element symbol for ``m``."""
    counts: Dict[str, int] = {}
    for a in m.atoms:
        sym = a.symbol.split("_", 1)[0]  # 'Pt_II' -> 'Pt'
        if sym == "NH3":
            sym = "N"
        counts[sym] = counts.get(sym, 0) + 1
    return counts


def _predicted_atom_counts(
    starting_materials: Sequence[MoleculeClosedTerm],
) -> Dict[str, int]:
    """Sum of heavy-atom counts across ``starting_materials``."""
    counts: Dict[str, int] = {}
    for m in starting_materials:
        for sym, n in _observed_atom_counts(m).items():
            counts[sym] = counts.get(sym, 0) + n
    return counts


def _balanced(
    predicted: Dict[str, int],
    observed: Dict[str, int],
    tol: int = 0,
) -> bool:
    for sym in set(predicted) | set(observed):
        if abs(predicted.get(sym, 0) - observed.get(sym, 0)) > tol:
            return False
    return True


def _default_rules() -> Sequence[ReactionRule]:
    """Lazy import of the click-rule registry."""
    try:
        from molmetal_lam.reactions.beta_reductions import REACTION_RULES
        return list(REACTION_RULES.values())
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


__all__ = [
    "ReactionPattern",
    "SynthesisPath",
    "synthesize",
    "retrosynthesize",
]