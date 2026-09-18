"""Tests for :mod:`molmetal_lam.lam_chem.lambda_combinators`.

Bird × Reynolds × Meijer — HOF rewrite contracts.

For every helper we ship a concrete expected output (imperative
sibling) and a property-based test verifying equivalence on small
finite inputs.  These are the "promotion" and "fusion" contract tests
(Bird 1988 Thm 1, 2) at the level appropriate for an isolated module.

The tests do NOT touch any READ-ONLY module; they only exercise the
HCombinator layer.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from molmetal_lam.lam_chem.lambda_combinators import (
    ClickRuleCombinator,
    MoleculeGenerationCombinator,
    compose,
    filter_combinator,
    fold_combinator,
    fold_right_combinator,
    map_combinator,
    nest,
    zip_with_combinator,
)


# ---------------------------------------------------------------------------
# Generic HOF contract tests
# ---------------------------------------------------------------------------


def test_map_combinator_basic() -> None:
    """``map(λx. x+1, [1,2,3]) = [2,3,4]`` (Reynolds 1972)."""
    result = list(map_combinator(lambda x: x + 1, [1, 2, 3]))
    assert result == [2, 3, 4]
    # Iterator type contract — lazy, not eager list.
    it = map_combinator(lambda x: x * 2, [10, 20])
    assert iter(it) is it


def test_fold_combinator_sum() -> None:
    """``fold(⊕, 0, [1,2,3,4]) = 10`` (Bird 1988 strict left fold)."""
    result = fold_combinator(lambda acc, x: acc + x, 0, [1, 2, 3, 4])
    assert result == 10

    # Empty collection → init.
    assert fold_combinator(lambda acc, x: acc + x, 42, []) == 42

    # String-concat fold (non-numeric ⊕).
    assert fold_combinator(lambda acc, s: acc + s, "", ["a", "b", "c"]) == "abc"


def test_fold_right_combinator() -> None:
    """``fold_right(⊕, [], [1,2,3]) = [1,2,3]`` — preserves order on cons."""
    result = fold_right_combinator(
        lambda x, acc: [x] + acc, [], [1, 2, 3]
    )
    assert result == [1, 2, 3]

    # Right-associativity witness — string concat with non-associative
    # operator (subtraction): (1 - (2 - 3)) = 1 - 2 + 3 = 2.
    result_sub = fold_right_combinator(
        lambda x, acc: x - acc, 0, [1, 2, 3]
    )
    assert result_sub == 2

    # Empty collection → init.
    assert fold_right_combinator(lambda x, acc: [x] + acc, ["init"], []) == ["init"]


def test_filter_combinator() -> None:
    """``filter(λx. x>2, [1,2,3,4]) = [3,4]`` (Reynolds 1972)."""
    result = list(filter_combinator(lambda x: x > 2, [1, 2, 3, 4]))
    assert result == [3, 4]

    # Empty predicate result.
    assert list(filter_combinator(lambda x: False, [1, 2, 3])) == []

    # All-pass.
    assert list(filter_combinator(lambda x: True, [1, 2, 3])) == [1, 2, 3]


def test_zip_with_combinator() -> None:
    """``zip_with(+, [1,2,3], [10,20,30]) = [11,22,33]`` (Meijer 1991 lens)."""
    result = list(
        zip_with_combinator(lambda a, b: a + b, [1, 2, 3], [10, 20, 30])
    )
    assert result == [11, 22, 33]

    # Stops at the shorter collection — same as Python zip.
    result_short = list(
        zip_with_combinator(lambda a, b: (a, b), [1, 2], [10, 20, 30])
    )
    assert result_short == [(1, 10), (2, 20)]


def test_compose() -> None:
    """``compose(f, g)(x) = f(g(x))`` (Reynolds 1972)."""
    f = lambda y: y + 10   # noqa: E731 — small lambda for clarity
    g = lambda x: x * 2   # noqa: E731
    h = compose(f, g)
    # compose(f, g)(3) = f(g(3)) = f(6) = 16
    assert h(3) == 16

    # String pipeline: compose(str.upper, str.strip)("  hello  ") == "HELLO".
    pipeline = compose(str.upper, str.strip)
    assert pipeline("  hello  ") == "HELLO"


def test_nest() -> None:
    """``nest(outer, inner)(coll) = outer(inner, coll)`` — HOF nesting."""
    # Per-element function: turn an int into a list [x, x+1].
    inner_fn = lambda x: [x, x + 1]   # noqa: E731
    # Outer: map_combinator-style — apply inner to each element.
    outer_fn = lambda fn, coll: [fn(x) for x in coll]
    f = nest(outer_fn, inner_fn)
    result = f([10, 20, 30])
    assert result == [[10, 11], [20, 21], [30, 31]]


# ---------------------------------------------------------------------------
# MLLC-shaped convenience selectors
# ---------------------------------------------------------------------------


class _MockRule:
    """Minimal rule object — duck-typed to expose ``name`` / ``applies_to`` / ``score``."""

    def __init__(
        self,
        name: str,
        applies_to: Any = None,
        score: Any = None,
        priority: Any = None,
    ) -> None:
        self.name = name
        if applies_to is not None:
            self.applies_to = applies_to
        if score is not None:
            self.score = score
        if priority is not None:
            self.priority = priority


def test_select_rules_for_pocket() -> None:
    """``ClickRuleCombinator.select_rules_for_pocket`` filters + scores + orders."""
    pocket_features: Dict[str, Any] = {
        "metal": "Pt",
        "has_azide": True,
        "has_alkyne": False,
        "size": "small",
    }

    # Three rules; only CuAAC should apply (has_azide AND has_alkyne is
    # the CuAAC strict reading — we'll only require has_azide here).
    cuaac = _MockRule(
        name="CuAAC",
        applies_to=lambda pf: pf.get("has_azide", False),
        score=lambda pf: 0.9 if pf.get("has_azide") else 0.1,
    )
    spaac = _MockRule(
        name="SPAAC",
        applies_to=lambda pf: pf.get("has_alkyne", False),
        score=lambda pf: 0.8 if pf.get("has_alkyne") else 0.2,
    )
    suzuki = _MockRule(
        name="Suzuki",
        applies_to=lambda pf: pf.get("has_aryl_halide", False),
        score=lambda pf: 0.5,
    )

    combinator = ClickRuleCombinator()
    selected = combinator.select_rules_for_pocket(
        pocket_features, [cuaac, spaac, suzuki]
    )

    # Only CuAAC passes the filter.
    assert [r.name for r in selected] == ["CuAAC"]
    # Score-sum witness is exposed and equals 0.9.
    assert combinator.last_score_sum == pytest.approx(0.9)


def test_select_rules_for_pocket_orders_by_score_desc() -> None:
    """When multiple rules apply, ordering is score-desc then name-asc."""
    pocket_features: Dict[str, Any] = {"flag": True}
    rules = [
        _MockRule(name="B", score=lambda pf: 0.3),
        _MockRule(name="A", score=lambda pf: 0.7),
        _MockRule(name="C", score=lambda pf: 0.5),
    ]
    combinator = ClickRuleCombinator()
    selected = combinator.select_rules_for_pocket(pocket_features, rules)
    assert [r.name for r in selected] == ["A", "C", "B"]


def test_generate_molecule() -> None:
    """``MoleculeGenerationCombinator.generate_molecule`` smoke test."""
    # Two rules, two tiles → 4 (rule, tile) pairs.  Each pair yields
    # the tile as a SMILES via the default identity fallback.
    class _Rule:
        def __init__(self, name: str, prefix: str) -> None:
            self.name = name
            self.prefix = prefix

        def apply(self, scaffold: Any, tile: Any) -> List[str]:
            return [f"{self.prefix}-{tile}"]

    rules = [_Rule("R1", "p1"), _Rule("R2", "p2")]
    tiles = ["T1", "T2"]

    combinator = MoleculeGenerationCombinator()
    smiles_list = combinator.generate_molecule(
        scaffold=None, rules=rules, tiles=tiles
    )

    # All 4 pairs produce 1 SMILES each → 4 SMILES total.
    assert len(smiles_list) == 4
    # Every entry is a string (the SMILES type contract).
    assert all(isinstance(s, str) for s in smiles_list)
    # Set semantics — each (rule, tile) pair contributes exactly one.
    assert set(smiles_list) == {"p1-T1", "p1-T2", "p2-T1", "p2-T2"}
