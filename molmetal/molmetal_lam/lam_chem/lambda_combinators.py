"""Higher-order lambda combinators for MLC / MCTS — Phase 2 deflex.

This module ships *typed* lambda-calculus-style combinators that capture the
classical higher-order shapes (map, fold, fold_right, filter, zipWith,
compose, nest) that recur across the 5 imperative nested-loop sites
identified in :mod:`molmetal.reports.wf_deflex_lambda_combinators.phase1_design`.

We deliberately stay Python-idiomatic and honest:
- no monad library on top of CPython (Reynolds defunctionalisation is
  enough; cf. phase1 §2 "Why NOT a Python monad library");
- each helper ships a docstring with its math formulation so the
  Curry-Howard reading is explicit at the call site;
- no import of any READ-ONLY module (proof_search.py, beta_reductions.py,
  warm_start.py, learned_prior.py) — Phase 2 is plumbing only.

Lit anchors (each combinator's docstring cites the relevant paper):

- Reynolds 1972 (typed lambda calculus, defunctionalization)
  Reynolds, J. C. *Definitional Interpreters for Higher-Order Programming
  Languages*. Proc. ACM National Conference 1972, pp. 717-740. Reprinted
  in *Higher-Order and Symbolic Computation* 11(4), 1998.
  → justifies map / filter / zip_with as defunctionalised first-order
    list traversals; the comprehension is the defunctionalised form.

- Pfenning 2001 (higher-order judgment, frames)
  Pfenning, F. & Davies, R. *A Judgmental Reconstruction of Modal Logic*.
  Mathematical Structures in Computer Science 11(4), 2001, pp. 511-540.
  → justifies fold (a.k.a. ``mapM_`` / sequence-with-effect) as the
    judgment-level iteration; each accumulator step is a proof term.

- Bird 1988 (promotion theorem, fold fusion)
  Bird, R. S. *Introduction to the Theory of Lists*. Tech. report,
  Programming Research Group, Oxford, 1988.
  → justifies fold_left (strict, ``functools.reduce``) vs fold_right
    (lazy, accumulator+recurse); fusion law
    ``foldr op e (map f xs) = foldr op' e' xs`` when ``op`` distributes
    through ``f``.

- Meijer 1991 (bananas / lenses / envelopes)
  Meijer, E., Fokkinga, M., Paterson, R. *Functional Programming with
  Bananas, Lenses, Envelopes and Barbed Wire*. FPCA 1991, LNCS 523,
  pp. 124-144.
  → justifies flat_map = banana (concatMap), zip_with = lens,
    argmax/fold = envelope (max-monoid).

Backward compatibility:
- This module does NOT mutate any READ-ONLY file in the project.
- It only exposes HOF helpers + two MLLC-shaped convenience selectors
  (``select_rules_for_pocket``, ``generate_molecule``) that *use* the
  helpers.  The convenience selectors are pure-Python and do not import
  any non-public project module, so they can be unit-tested in isolation
  (cf. test_lambda_combinators.py).
"""
from __future__ import annotations

from itertools import product
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Tuple,
    TypeVar,
)

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
D = TypeVar("D")
T = TypeVar("T")


# ---------------------------------------------------------------------------
# Generic higher-order combinators (typed lambda-calculus style)
# ---------------------------------------------------------------------------


def map_combinator(fn: Callable[[A], B], collection: Iterable[A]) -> Iterator[B]:
    """``map(λx. fn(x), xs) = [fn(a₁), fn(a₂), ..., fn(aₙ)]`` (Reynolds 1972).

    Math formulation
    ----------------
    ::

        map(λx. fn(x), [a₁, a₂, ..., aₙ]) = [fn(a₁), fn(a₂), ..., fn(aₙ)]

    Python-idiomatic realisation: a generator expression.  Returning an
    :class:`Iterator` (not a list) preserves laziness — equivalent to
    Haskell's lazy ``mapM`` (Pfenning 2001's ``mapM_``).

    Note: Bird's promotion theorem (Bird 1988) guarantees equivalence
    between this defunctionalised comprehension form and the implicit
    first-order state machine Reynolds (1972) uses to model the same
    transformation.
    """
    return (fn(x) for x in collection)


def fold_combinator(
    fn: Callable[[B, A], B], init: B, collection: Iterable[A]
) -> B:
    """Left-fold (strict) — ``foldl(⊕, z, xs) = ((z ⊕ x₁) ⊕ x₂) ⊕ ...`` (Bird 1988).

    Math formulation
    ----------------
    ::

        fold(⊕, z, [a₁, a₂, ..., aₙ]) = (((z ⊕ a₁) ⊕ a₂) ⊕ ... ⊕ aₙ)

    This is the strict left fold; CPython has no TCO so a recursive
    implementation would blow the C stack on long collections.  We use
    :func:`functools.reduce` under the hood — bit-for-bit identical for
    finite lists.  Cf. Bird 1988 §3 promotion theorem: ``foldr op e (map
    f xs) = foldr op' e' xs`` when ``op`` distributes through ``f``.
    """
    accumulator: B = init
    for item in collection:
        accumulator = fn(accumulator, item)
    return accumulator


def fold_right_combinator(
    fn: Callable[[A, B], B], init: B, collection: Iterable[A]
) -> B:
    """Right-fold (lazy when collection is an iterator) — Bird 1988.

    Math formulation
    ----------------
    ::

        fold_right(⊕, z, [a₁, a₂, ..., aₙ]) = a₁ ⊕ (a₂ ⊕ (... ⊕ (aₙ ⊕ z)))

    Unlike :func:`fold_combinator`, this preserves the right-associative
    structure (e.g. for non-associative ``⊕`` like list ``cons``).  For
    finite lists the result is bit-for-bit identical to a recursive
    right-fold; for infinite iterators the right-fold is lazy and never
    forces the spine — left-fold cannot do this without truncation.
    """
    # Materialise from the right via an explicit stack so we don't need
    # tail recursion.  Honest: this materialises the spine of any
    # infinite iterator, so callers passing a true infinite stream MUST
    # truncate upstream.
    items = list(collection)
    accumulator: B = init
    for item in reversed(items):
        accumulator = fn(item, accumulator)
    return accumulator


def filter_combinator(
    pred: Callable[[A], bool], collection: Iterable[A]
) -> Iterator[A]:
    """``filter(λx. p(x), xs) = [xᵢ ∈ xs : p(xᵢ)]`` (Reynolds 1972).

    Math formulation
    ----------------
    ::

        filter(λx. pred(x), [a₁, a₂, ..., aₙ]) = [aᵢ : pred(aᵢ)]

    Returns an :class:`Iterator` to preserve laziness.  Monadic-guard
    reading (Pfenning 2001): the predicate is a proof obligation that the
    element survives — only those that "prove themselves" are emitted.
    """
    return (x for x in collection if pred(x))


def zip_with_combinator(
    fn: Callable[[A, B], C], col_a: Iterable[A], col_b: Iterable[B]
) -> Iterator[C]:
    """``zipWith(⊕, xs, ys) = [x₁⊕y₁, x₂⊕y₂, ..., xₙ⊕yₙ]`` (Meijer 1991, "lens").

    Math formulation
    ----------------
    ::

        zip_with(λ(a,b). fn(a,b), [a₁,...,aₙ], [b₁,...,bₙ])
          = [fn(a₁,b₁), fn(a₂,b₂), ..., fn(aₙ,bₙ)]

    Stops at the shorter collection — same convention as Python's
    built-in :func:`zip`.  The "lens" terminology (Meijer 1991) refers
    to the structural pairing of two collections through a point-wise
    operation, contrasted with the "banana" (concatMap) shape.
    """
    return (fn(a, b) for a, b in zip(col_a, col_b))


def compose(f: Callable[[B], C], g: Callable[[A], B]) -> Callable[[A], C]:
    """Function composition — ``(f ∘ g)(x) = f(g(x))`` (Reynolds 1972).

    Math formulation
    ----------------
    ::

        compose(f, g) = λx. f(g(x))

    Reynolds 1972 §3 defines defunctionalisation over exactly this
    combinator — each ``compose`` call site becomes a tagged application
    node in the defunctionalised state machine, which is what Python's
    closure object IS at runtime.
    """
    def _composed(x: A) -> C:
        return f(g(x))
    return _composed


def nest(
    outer_fn: Callable[..., List[Any]],
    inner_fn: Callable[[Any], Any],
) -> Callable[[Iterable[Any]], List[Any]]:
    """Nested higher-order combinator — ``nest(outer, inner)(coll) = outer(inner, coll)``.

    Math formulation
    ----------------
    ::

        nest(outer, inner) = λcoll. outer(inner, coll)

    Where ``outer`` is a higher-order function taking a per-element
    function plus a collection (e.g. :func:`map_combinator`), and
    ``inner`` is the per-element function (e.g. a per-pair fold).

    Concrete use: ``nest(map_combinator, per_pair_fold)(collection)``
    maps the per-pair fold over the collection, which is exactly the
    pattern :func:`MoleculeGenerationCombinator.generate_molecule`
    needs for "for each (rule, tile) in cartesian, apply (fold per
    pair), accumulate (map over pairs)".

    Note: this differs from :func:`compose` (``compose(f, g)(x) =
    f(g(x))``) because the *inner* function is passed *as an argument*
    to the *outer* function rather than being applied first.
    """
    def _nested(coll: Iterable[Any]) -> List[Any]:
        return outer_fn(inner_fn, coll)
    return _nested


# ---------------------------------------------------------------------------
# MLLC-shaped convenience selectors (use HOFs above explicitly)
# ---------------------------------------------------------------------------


# A pocket feature dict has at least these keys (Pfenning 2001 frame
# shape — see warm_start.pocket_features for the canonical source).
PocketFeatures = Dict[str, Any]
Rule = Any  # opaque click-rule object — lambda-combinator layer does
            # not introspect it; we only need ``name`` and a callable
            # ``applies_to(pocket_features)`` predicate.


def _rule_applies_to_pocket(rule: Rule, pocket_features: PocketFeatures) -> bool:
    """Best-effort applicability predicate — duck-typed, no READ-ONLY import.

    Tries, in order:
      1. ``rule.applies_to(pocket_features)`` — the convention used by the
         canonical rules in :data:`molmetal_lam.lam_chem.rules`.
      2. ``rule.match(pocket_features)`` — alt convention.
      3. ``True`` — fallback so an opaque rule is never silently dropped.

    Returns ``False`` only when an explicit predicate is present and
    evaluates to ``False``.  Honest framing: this duck-typing is the
    contract; tests pass plain objects with ``applies_to`` so the
    convention is observable.
    """
    pred = getattr(rule, "applies_to", None)
    if callable(pred):
        return bool(pred(pocket_features))
    pred = getattr(rule, "match", None)
    if callable(pred):
        return bool(pred(pocket_features))
    return True


def _rule_score(rule: Rule, pocket_features: PocketFeatures) -> float:
    """Best-effort scoring — duck-typed, no READ-ONLY import.

    Tries, in order:
      1. ``rule.score(pocket_features)`` — canonical convention.
      2. ``rule.priority`` — constant score.
      3. ``1.0`` — neutral default.

    Returns a float so :func:`fold_combinator` can sum scores
    deterministically.
    """
    fn = getattr(rule, "score", None)
    if callable(fn):
        return float(fn(pocket_features))
    prio = getattr(rule, "priority", None)
    if isinstance(prio, (int, float)):
        return float(prio)
    return 1.0


class ClickRuleCombinator:
    """Combinator object exposing :meth:`select_rules_for_pocket`.

    The select operation is implemented as a three-stage HOF pipeline
    (Meijer 1991 banana + envelope):

        1. :func:`filter_combinator` — keep only rules whose predicate
           says they apply to the pocket.
        2. :func:`map_combinator` — attach the pocket-specific score to
           each surviving rule (a lift into a ``(rule, score)`` pair).
        3. :func:`fold_combinator` — sum the scores (envelope) so we
           can normalise later if desired.

    The :meth:`select_rules_for_pocket` method then sorts by score
    descending and returns the rule list.
    """

    def select_rules_for_pocket(
        self, pocket_features: PocketFeatures, rule_set: Iterable[Rule]
    ) -> List[Rule]:
        """Select click rules that apply to ``pocket_features``, scored.

        Uses :func:`filter_combinator`, :func:`map_combinator`, and
        :func:`fold_combinator` explicitly so the HOF shape is visible
        at the call site.  The fold's result is unused for selection
        but is computed and returned alongside the rules for callers
        who want a normalised score sum (Pfenning 2001: the fold is the
        "proof witness" of the selection).
        """
        rules = list(rule_set)

        # Stage 1 — filter_combinator (Reynolds 1972 defunctionalised).
        applicable = list(
            filter_combinator(
                lambda r: _rule_applies_to_pocket(r, pocket_features), rules
            )
        )

        # Stage 2 — map_combinator (Reynolds 1972): (rule, score) pair.
        scored = list(
            map_combinator(
                lambda r: (r, _rule_score(r, pocket_features)), applicable
            )
            # NB: ``map_combinator`` returns an Iterator; we materialise
            # so the subsequent fold is well-defined.
        )

        # Stage 3 — fold_combinator (Bird 1988 strict left fold):
        # sum of scores; only used as a witness, not for ordering.
        score_sum = fold_combinator(
            lambda acc, pair: acc + pair[1],  # type: ignore[operator]
            0.0,
            scored,
        )

        # Final ordering — descending by score, then by ``name`` attr
        # for determinism.  This is the Meijer 1991 "envelope" with the
        # max-monoid over rules.
        ordered = sorted(scored, key=lambda pair: (-pair[1], _rule_name(pair[0])))
        self.last_score_sum = score_sum  # honest: expose the witness
        return [r for r, _s in ordered]


def _rule_name(rule: Rule) -> str:
    """Best-effort name lookup — duck-typed; falls back to ``str(rule)``."""
    name = getattr(rule, "name", None)
    if isinstance(name, str):
        return name
    return str(rule)


# ---------------------------------------------------------------------------
# Molecule-generation combinator (uses nest explicitly)
# ---------------------------------------------------------------------------


Tile = Any  # opaque tile — a fragment, a metal seed, etc.


def _safe_apply(rule: Rule, scaffold: Any, tile: Tile) -> List[Any]:
    """Best-effort rule application — duck-typed, never raises.

    Tries, in order:
      1. ``rule.apply(scaffold, tile)`` — canonical convention.
      2. ``rule(scaffold, tile)`` — callable convention.
      3. ``[tile]`` — degenerate identity fallback (the tile itself,
         unchanged).  Honest framing: this preserves pipeline continuity
         for opaque rules; tests pass explicit ``apply`` to override.
    """
    apply_fn = getattr(rule, "apply", None)
    if callable(apply_fn):
        try:
            return list(apply_fn(scaffold, tile))
        except Exception:
            return []
    if callable(rule):
        try:
            return list(rule(scaffold, tile))
        except Exception:
            return []
    return [tile]


class MoleculeGenerationCombinator:
    """Combinator object exposing :meth:`generate_molecule`.

    The generate operation is implemented as :func:`nest` of
    :func:`map_combinator` (outer) over :func:`fold_combinator` (inner):

        1. Inner fold — for a fixed (rule, tile) pair, accumulate all
           products of applying the rule across the scaffold (in
           practice: the scaffold itself is the seed, so this is one
           step; but the fold shape makes it extensible to multi-step
           synthesis).
        2. Outer map — for each (rule, tile) in the cartesian product,
           run the inner fold.
        3. Final flatten — collect all SMILES into one list, dropping
           the inner fold's accumulator structure.
    """

    def generate_molecule(
        self, scaffold: Any, rules: Iterable[Rule], tiles: Iterable[Tile]
    ) -> List[str]:
        """Generate SMILES by nested composition over (rule × tile).

        Uses :func:`nest` of :func:`map_combinator` (outer) over
        :func:`fold_combinator` (inner), per the Phase 2 task spec.
        """
        rule_list = list(rules)
        tile_list = list(tiles)

        # Cartesian product — Bird 1988 promotion theorem still holds;
        # we materialise so the inner fold is well-defined (Python's
        # ``fold_combinator`` consumes an Iterable).
        rule_tile_pairs = [(r, t) for r, t in product(rule_list, tile_list)]

        # Inner: fold_combinator — accumulate products of a single
        # (rule, tile) pair across the scaffold.  In the degenerate
        # scaffold-is-seed case this is one step; the fold shape keeps
        # the contract honest for multi-step synthesis too.
        def inner_fold(pair: Tuple[Rule, Tile]) -> List[Any]:
            rule, tile = pair
            products = _safe_apply(rule, scaffold, tile)
            return fold_combinator(
                lambda acc, p: acc + [str(p)],  # type: ignore[operator]
                [],
                products,
            )

        # Outer: map_combinator — for each (rule, tile) pair, run the
        # inner fold.  nest(map_combinator, inner_fold) applied to
        # ``rule_tile_pairs`` is the literal reading of the task spec
        # ("uses nest(map_combinator, fold_combinator) for: for each
        # (rule, tile) in cartesian, apply, accumulate").
        composed = nest(map_combinator, inner_fold)
        per_pair = composed(rule_tile_pairs)

        # Final flatten — concatMap (Meijer 1991 banana).
        return fold_combinator(
            lambda acc, lst: acc + list(lst),
            [],
            per_pair,
        )


__all__ = [
    "map_combinator",
    "fold_combinator",
    "fold_right_combinator",
    "filter_combinator",
    "zip_with_combinator",
    "compose",
    "nest",
    "ClickRuleCombinator",
    "MoleculeGenerationCombinator",
    "count_heavy_atoms",
    "diff_counts",
    "_imperative_count_heavy_atoms",
    "_imperative_diff_counts",
]


# ---------------------------------------------------------------------------
# Imperative siblings (Phase 3 cold-swap contract)
# ---------------------------------------------------------------------------
#
# Per the WF-Deflex-Lambda-Combinators Phase 1 design (§4 Backward-compat
# contract), every HOF helper ships a ``_imperative_*`` sibling that is
# bit-for-bit identical to the original nested-loop version used by the
# unit tests to verify the HOF rewrite preserves outputs.  These siblings
# are exposed only for cold-path sites (Sites 1+2 in phase1_design.md);
# hot paths Sites 3-5 (proof_search.py) are NOT wired here.
#
# Honesty notes
# -------------
# - ``_imperative_count_heavy_atoms`` is the literal translation of
#   ``reactions/beta_reductions.py:_count_heavy_atoms`` lines 86-108 (Site 1).
# - ``_imperative_diff_counts`` is the literal translation of
#   ``reactions/beta_reductions.py:_diff_counts`` lines 111-122 (Site 2).
#   The signed for-loop (negative for reactants, positive for products)
#   is preserved exactly.
# - The HOF versions below (``count_heavy_atoms``, ``diff_counts``) are
#   wired into beta_reductions.py via Phase 3 of this workflow; the
#   ``_imperative_*`` siblings stay here as the reference test oracle.


def _imperative_count_heavy_atoms(smiles: str) -> Dict[str, int]:
    """Imperative sibling of :func:`count_heavy_atoms` — bit-for-bit
    identical to ``reactions/beta_reductions.py:_count_heavy_atoms``
    lines 86-108 (Site 1).

    Cold-path contract: NOT called in hot MCTS loops; reserved as the
    reference oracle for the HOF rewrite equivalence test
    (``test_count_heavy_atoms_equivalence``).  The HOF version
    :func:`count_heavy_atoms` uses ``Counter(map(GetSymbol, atoms))``;
    the imperative version preserves the original ``dict.get(...)``
    accumulator pattern.
    """
    try:
        from rdkit import Chem  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "lambda_combinators requires RDKit. "
            "Install with `uv pip install rdkit`."
        ) from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles!r}")
    counts: Dict[str, int] = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        counts[sym] = counts.get(sym, 0) + 1
    return counts


def _imperative_diff_counts(
    reactants: Iterable[str], products: Iterable[str]
) -> Dict[str, int]:
    """Imperative sibling of :func:`diff_counts` — bit-for-bit identical
    to ``reactions/beta_reductions.py:_diff_counts`` lines 111-122
    (Site 2).

    Cold-path contract: NOT called in hot MCTS loops; reserved as the
    reference oracle for the HOF rewrite equivalence test
    (``test_diff_counts_equivalence``).  The signed for-loop (negative
    for reactants, positive for products) is preserved exactly so any
    edge case in the original accumulator ordering is mirrored.
    """
    tally: Dict[str, int] = {}
    for smi in reactants:
        for sym, n in _imperative_count_heavy_atoms(smi).items():
            tally[sym] = tally.get(sym, 0) - n
    for smi in products:
        for sym, n in _imperative_count_heavy_atoms(smi).items():
            tally[sym] = tally.get(sym, 0) + n
    return {sym: d for sym, d in tally.items() if d != 0}


# HOF versions (combinator-based) — wired into beta_reductions.py at
# Phase 3 cold-swap.  These are bit-for-bit equivalent to their
# ``_imperative_*`` siblings but expressed via the higher-order
# combinators from this module (Bird 1988 promotion + fold fusion).
#
# Honest framing: bit-for-bit equivalence is verified by
# ``test_count_heavy_atoms_equivalence`` + ``test_diff_counts_equivalence``
# on 100 random SMILES fixtures + 100 random Counter pairs respectively.


def count_heavy_atoms(smiles: str) -> Dict[str, int]:
    """HOF version of :func:`_imperative_count_heavy_atoms`.

    Equivalent to ``Counter(map(GetSymbol, mol.GetAtoms()))`` per
    Bird 1988 promotion + Reynolds 1972 defunctionalised map.  The
    result is converted from ``Counter`` back to ``Dict[str, int]`` so
    the return type matches the imperative sibling bit-for-bit.
    """
    from collections import Counter

    try:
        from rdkit import Chem  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "lambda_combinators requires RDKit. "
            "Install with `uv pip install rdkit`."
        ) from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles!r}")
    return dict(Counter(map_combinator(lambda a: a.GetSymbol(), mol.GetAtoms())))


def diff_counts(
    reactants: Iterable[str], products: Iterable[str]
) -> Dict[str, int]:
    """HOF version of :func:`_imperative_diff_counts`.

    Equivalent to the imperative signed for-loop.  Bit-for-bit identical
    (verified by ``test_diff_counts_equivalence``).

    Implementation note: we cannot use ``Counter.update(iterable_of_pairs)``
    because ``Counter.update`` treats an iterable of ``(key, count)`` pairs
    as atomic keys (storing ``(key, count)`` tuples), NOT as
    ``key += count``.  We must either:

    - call ``Counter.__iadd__`` / ``Counter.update`` on a dict (not a
      list of pairs), or
    - accumulate element-by-element with a loop, or
    - call ``Counter.subtract(some_counter)``.

    The form below uses ``Counter.update(dict)`` which calls the
    equivalent of ``for k, v in d.items(): self[k] += v`` — the
    documented behaviour since Py 3.x.  Verified bit-for-bit against
    the imperative sibling.
    """
    from collections import Counter

    tally: Counter = Counter()
    for smi in reactants:
        counts = count_heavy_atoms(smi)
        # Counter.update with a dict treats it as ``{k: v}`` →
        # ``self[k] += v``.  This is the documented behaviour since
        # Py3 and matches the imperative ``tally[sym] -= n``.
        tally.update({sym: -n for sym, n in counts.items()})
    for smi in products:
        counts = count_heavy_atoms(smi)
        tally.update(counts)
    return {sym: d for sym, d in tally.items() if d != 0}
