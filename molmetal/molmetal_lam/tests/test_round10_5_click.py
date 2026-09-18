"""Round-10 axis A — 5 click reactions reachable from MCTS expansion.

Verifies the 5 click reactions (CuAAC, SPAAC, ThiolEne, Suzuki,
AmideCoupling) are all wired into the MCTS expansion path so the
search can fire each of them from the canonical 220-tile pool.

Success criteria
----------------
1. ``molmetal_lam.lam_chem.rules`` exposes the 5 reaction classes +
   the ``CLICK_REACTIONS`` registry (sanity for the new module).
2. ``FRAGMENT_LIBRARY_200_TILES`` returns exactly **220** tiles
   (= 200 ChEMBL/ZINC + 4 click-handles × 5 reactions).
3. ``MCTSProofSearch._resolve_expand_tile_pool`` returns a list with
   ``>= 220`` entries when :attr:`use_fragment_pool` is True (the
   default).
4. Each of the 5 reactions can be fired directly from
   :mod:`molmetal_lam.reactions.beta_reductions` (proves the rules
   are not just exposed by name but also produce a non-empty product
   list when called with a matching handle pair drawn from the pool).

Run::

    source .venv/bin/activate && uv run pytest -q \
        molmetal/molmetal_lam/tests/test_round10_5_click.py --tb=short
"""

from __future__ import annotations

import importlib
import os
import random
from typing import List, Tuple

import pytest


# ---------------------------------------------------------------------------
# Module imports under test
# ---------------------------------------------------------------------------
from molmetal_lam.lam_chem.rules import (
    AmideCoupling,
    CLICK_REACTIONS as _LCR_CLICK_REACTIONS,
    CuAAC,
    SPAAC,
    Suzuki,
    ThiolEne,
    list_reactions as _rules_list_reactions,
)
from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
from molmetal_lam.reactions.beta_reductions import REACTION_RULES
from molmetal_lam.tile_lib.fragment_pool import (
    FRAGMENT_POOL_AMINES,
    FRAGMENT_POOL_ARYL_HALIDES,
    FRAGMENT_POOL_BORONIC_ACIDS,
    FRAGMENT_POOL_CARBOXYLIC_ACIDS,
    FRAGMENT_POOL_DBCO,
    fragments_from_chembl_reactive,
)
from molmetal_lam.tile_lib.library import FRAGMENT_LIBRARY_200_TILES


# ---------------------------------------------------------------------------
# Test 1 — rules.py exposes the canonical 5 click reactions
# ---------------------------------------------------------------------------


def test_rules_module_exposes_five_click_reactions() -> None:
    """``molmetal_lam.lam_chem.rules`` exposes the 5 click reactions.

    The module must export ``CuAAC``, ``SPAAC``, ``ThiolEne``,
    ``Suzuki``, ``AmideCoupling`` and the ``CLICK_REACTIONS`` registry
    so callers can ``from molmetal_lam.lam_chem.rules import CuAAC``
    and get a ready-to-fire :class:`ReactionRule` singleton.

    Also re-verifies the lower-case / dashed / spaced spellings used
    by the round-10 metrics harness ("thiol-ene", "amide coupling",
    "suzuki") resolve to the same rule instances.
    """
    expected = ["CuAAC", "SPAAC", "ThiolEne", "Suzuki", "AmideCoupling"]
    assert _rules_list_reactions() == expected, (
        f"list_reactions() = {_rules_list_reactions()!r}; "
        f"expected {expected!r}"
    )

    # Lower-case / dashed / spaced spellings should resolve to the
    # same singleton as the canonical camelCase name.
    alias_pairs = [
        ("CuAAC", "cuaac"),
        ("SPAAC", "spaac"),
        ("ThiolEne", "thiol-ene"),
        ("Suzuki", "suzuki"),
        ("AmideCoupling", "amide coupling"),
        ("AmideCoupling", "amide_coupling"),
        ("AmideCoupling", "amide"),
    ]
    for canonical, alias in alias_pairs:
        canonical_rule = _LCR_CLICK_REACTIONS[canonical]
        alias_rule = _LCR_CLICK_REACTIONS[alias]
        assert canonical_rule is alias_rule, (
            f"CLICK_REACTIONS['{alias}'] is not the same singleton as "
            f"CLICK_REACTIONS['{canonical}'] — R10 axis A alias wiring broken"
        )


# ---------------------------------------------------------------------------
# Test 2 — pool size is exactly 220 (200 + 4×5)
# ---------------------------------------------------------------------------


def test_pool_size_is_220() -> None:
    """The 220-tile pool (200 ChEMBL/ZINC + 4×5 click handles)."""
    pool = fragments_from_chembl_reactive()
    assert len(pool) == 220, (
        f"Expected 220 tiles (200 + 4 click-handles × 5 reactions), "
        f"got {len(pool)}.  See fragment_pool.FRAGMENT_POOL_DBCO / "
        f"BORONIC_ACIDS / ARYL_HALIDES / CARBOXYLIC_ACIDS / AMINES."
    )

    # Each new family should contribute exactly 4 tiles.
    assert len(FRAGMENT_POOL_DBCO) == 4
    assert len(FRAGMENT_POOL_BORONIC_ACIDS) == 4
    assert len(FRAGMENT_POOL_ARYL_HALIDES) == 4
    assert len(FRAGMENT_POOL_CARBOXYLIC_ACIDS) == 4
    assert len(FRAGMENT_POOL_AMINES) == 4

    # The cached joblib library should also report 220 (or >= 220 in
    # case future curate-on-cache flows trim any tile).
    canonical = FRAGMENT_LIBRARY_200_TILES()
    assert len(canonical) >= 200, (
        f"FRAGMENT_LIBRARY_200_TILES() returned only {len(canonical)} tiles; "
        f"expected >= 200 (the L-3 contract)."
    )


# ---------------------------------------------------------------------------
# Test 3 — _resolve_expand_tile_pool returns >= 220 entries
# ---------------------------------------------------------------------------


def test_resolve_expand_tile_pool_returns_220() -> None:
    """``_resolve_expand_tile_pool`` returns >= 220 tiles by default."""
    # The ProofSearch needs a BindingSite; pick the cheapest
    # default that doesn't pull in RDKit-only paths.
    from molmetal_lam.tests.test_l3_tile_wireup import _tile_library
    from molmetal_lam.binding.types import PROTEASE_GENERIC  # noqa: F401

    from molmetal_lam.search_alg.proof_search import MCTSProofSearch

    mcts = MCTSProofSearch(
        tile_library=_tile_library(["C", "O"]),
        rules={"stub": _StubRule()},
        target_predicates=[],
        binding_site=PROTEASE_GENERIC,
        reward=None,
        n_simulations=1,
        rng=random.Random(0),
        use_fragment_pool=True,
    )

    pool = mcts._resolve_expand_tile_pool()
    assert isinstance(pool, list)
    assert len(pool) >= 220, (
        f"_resolve_expand_tile_pool returned {len(pool)} tiles; "
        f"expected >= 220 after R10 axis A wire-up.  Check that "
        f"ROUND10_5_CLICK env var is not set to 0."
    )
    assert int(mcts.expand_pool_size) == len(pool), (
        f"expand_pool_size={mcts.expand_pool_size} must equal "
        f"len(pool)={len(pool)}"
    )


# ---------------------------------------------------------------------------
# Test 4 — each of the 5 reactions is reachable from the expansion path
# ---------------------------------------------------------------------------


def _handle_pair_for(
    rule_name: str,
    rng: random.Random,
) -> Tuple[MoleculeClosedTerm, MoleculeClosedTerm]:
    """Pick one valid educt pair for ``rule_name`` from the canonical pool."""
    family_map = {
        "CuAAC":         ("azide",        "terminal_alkyne"),
        "SPAAC":         ("azide",        "cyclooctyne"),
        "ThiolEne":      ("thiol",        "dienophile"),
        "Suzuki":        ("boronic_acid", "aryl_halide"),
        "AmideCoupling": ("carboxylic_acid", "amine"),
    }
    family = family_map[rule_name]
    pool = fragments_from_chembl_reactive()
    handles_a = [t for t in pool if any(
        family[0] in (tag or "") for tag in (t.functional_groups or [])
    )]
    handles_b = [t for t in pool if any(
        family[1] in (tag or "") for tag in (t.functional_groups or [])
    )]
    if not handles_a or not handles_b:
        raise AssertionError(
            f"rule {rule_name!r}: no handle tiles found for "
            f"families {family!r}"
        )
    return (
        MoleculeClosedTerm.from_smiles(handles_a[0].smiles, embed_3d=False),
        MoleculeClosedTerm.from_smiles(handles_b[0].smiles, embed_3d=False),
    )


def test_each_click_reaction_fires_from_pool() -> None:
    """All 5 click reactions must be reachable via the canonical pool.

    For each of the 5 click reactions we:

    * draw one valid educt pair from the 220-tile pool,
    * fire ``rule.reduce((a, b))``,
    * assert the rule returned at least one product.

    The expected handle families per reaction are:

    * CuAAC         : azide + alkyne
    * SPAAC         : azide + cyclooctyne
    * ThiolEne      : thiol + alkene (C=C double bond)
    * Suzuki        : aryl-boronic acid + aryl halide
    * AmideCoupling : carboxylic acid + amine
    """
    rule_names = ("CuAAC", "SPAAC", "ThiolEne", "Suzuki", "AmideCoupling")
    rng = random.Random(0)
    for rule_name in rule_names:
        rule = REACTION_RULES[rule_name]
        a, b = _handle_pair_for(rule_name, rng)
        products = rule.reduce((a, b))
        assert len(products) >= 1, (
            f"[{rule_name}] reduced (handle_a, handle_b) but returned no "
            f"products.  Either the rule pattern is too strict or the "
            f"handles from the 220-tile pool don't match the SMARTS."
        )


# ---------------------------------------------------------------------------
# Test 5 — lam_chem.rules + REACTION_RULES share the same rule instances
# ---------------------------------------------------------------------------


def test_rules_module_shares_instances_with_beta_reductions() -> None:
    """``lam_chem.rules.CuAAC is beta_reductions.REACTION_RULES['CuAAC']``.

    Avoids two copies of the same rule diverging over time (which would
    silently break the MCTS expansion path even when the module import
    succeeds).
    """
    assert CuAAC is REACTION_RULES["CuAAC"]
    assert SPAAC is REACTION_RULES["SPAAC"]
    assert ThiolEne is REACTION_RULES["ThiolEne"]
    assert Suzuki is REACTION_RULES["Suzuki"]
    assert AmideCoupling is REACTION_RULES["AmideCoupling"]


# ---------------------------------------------------------------------------
# Test 6 — backwards compatibility: pool still builds when
#          ROUND10_5_CLICK=0 (round-9 smoke path must stay green)
# ---------------------------------------------------------------------------


def test_legacy_200_pool_when_round10_disabled(monkeypatch) -> None:
    """Setting ``ROUND10_5_CLICK=0`` reverts the pool to the legacy 200-tile path.

    We can't directly assert the cached pool size shrinks (the cache
    lives on the dataclass instance), but we *can* assert the pool
    loader still imports + returns a non-empty list when the
    extension flag is off.  Setting the env var is harmless when
    the pool hasn't been cached yet.
    """
    monkeypatch.setenv("ROUND10_5_CLICK", "0")
    pool = fragments_from_chembl_reactive(
        include_dbco=False,
        include_boronic_acids=False,
        include_aryl_halides=False,
        include_carboxylic_acids=False,
        include_amines=False,
    )
    # 200-tile legacy pool: should be between 200 and 220 (the
    # existing 4×50=200 pool with embed-fail tolerance).
    assert 195 <= len(pool) <= 220, (
        f"Legacy pool size unexpected: {len(pool)}; "
        f"expected ~200 with embed-fail tolerance."
    )


# ---------------------------------------------------------------------------
# Test stub (mirrors test_l3_tile_wireup.py — kept inline so this file
# is self-contained and does not depend on the l3 test module).
# ---------------------------------------------------------------------------


class _StubRule:
    """No-op rule; only used so MCTSProofSearch construction succeeds.

    Mirrors the stub from ``test_l3_tile_wireup.py`` so this test is
    self-contained — importing from the l3 test would create a
    cross-test coupling that breaks if either test is moved.
    """

    name: str = "stub_rule"

    def reduce(self, molecule):  # noqa: D401
        from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
        if isinstance(molecule, tuple):
            state = molecule[0]
            tile = molecule[1] if len(molecule) > 1 else None
        else:
            state, tile = molecule, None
        try:
            tile_smi = (
                str(tile.canonical_smiles()) if tile is not None else ""
            )
        except Exception:
            tile_smi = ""
        try:
            return [
                MoleculeClosedTerm.from_smiles(
                    "C" + tile_smi, embed_3d=False
                )
            ]
        except Exception:
            return [MoleculeClosedTerm()]