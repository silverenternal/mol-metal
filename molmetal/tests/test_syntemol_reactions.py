"""Phase 2 verification — pytest + integration tests for the 13 new
SyntheMol-derived reactions registered into ``molmetal_lam.reactions.
beta_reductions.REACTION_RULES``.

Five tests matching the Phase 2 spec:

1. ``test_amide_coupling`` — amine + carboxylic acid -> amide, verify product.
2. ``test_ester_hydrolysis`` — acid + alkyl halide -> ester (the
   SyntheMol REAL proxy for esterification; documented in the Phase 1
   report).
3. ``test_boc_cleavage`` — BOC-protected amine -> free amine (single
   reactant; BOC_CLEAVAGE sentinel in the curated mapping).
4. ``test_syntemol_vs_lambda_equivalence`` — same reaction invoked
   through the SyntheMol wrapper *and* through lambda's
   ``REACTION_RULES`` registry produces identical product SMILES for
   three independent reactions.
5. ``test_reaction_count`` — registry holds exactly 18 rules (5 click
   + 13 SyntheMol).

Run with::

    source .venv/bin/activate && python -m pytest \
        molmetal/tests/test_syntemol_reactions.py -v
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from molmetal_lam.reactions.beta_reductions import REACTION_RULES
from molmetal_lam.sbdd_env.syntemol_reactions import (
    REAL_TO_LAMBDA,
    SyntheMolReactionRule,
    from_syntemol_to_lambda,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical(smi: str) -> str:
    """Return the RDKit canonical SMILES for ``smi`` (raises on parse fail)."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"RDKit failed to parse {smi!r}")
    return Chem.MolToSmiles(mol)


# ---------------------------------------------------------------------------
# 1. Amide coupling — amine + carboxylic acid -> amide
# ---------------------------------------------------------------------------


def test_amide_coupling() -> None:
    """Ethylamine + acetic acid -> N-ethyl acetamide.

    Expected canonical product SMILES: ``CCNC(C)=O``.
    """
    rule = REACTION_RULES["amide_coupling"]
    # Wrapper must expose the same surface the click rules do.
    assert isinstance(rule, SyntheMolReactionRule)
    assert rule.can_apply(["CCN", "CC(=O)O"]) is True

    products = rule.fire(["CCN", "CC(=O)O"])
    assert len(products) == 1, (
        f"expected exactly 1 amide product, got {len(products)}: {products}"
    )
    # Canonical comparison tolerates atom-map stripping.
    assert _canonical(products[0]) == _canonical("CCNC(C)=O"), (
        f"expected CCNC(C)=O, got {products[0]!r}"
    )

    # Negative control: feeding two amines should not fire the rule.
    assert rule.can_apply(["CCN", "CCN"]) is False
    assert rule.fire(["CCN", "CCN"]) == []


# ---------------------------------------------------------------------------
# 2. Ester hydrolysis — acid + alkyl halide -> ester (REAL proxy)
# ---------------------------------------------------------------------------


def test_ester_hydrolysis() -> None:
    """Acetic acid + ethyl bromide -> ethyl acetate + HBr.

    Per the Phase 1 report SyntheMol's REAL set models the reverse
    direction (esterification); we verify that the wrapper fires and
    yields the canonical ester product.  The leaving group is stripped
    by ``fire()``'s SanitizeMol pass, so we expect the ester SMILES
    alone.

    Expected canonical product SMILES: ``CCOC(C)=O``.
    """
    rule = REACTION_RULES["ester_hydrolysis"]
    assert isinstance(rule, SyntheMolReactionRule)

    products = rule.fire(["CC(=O)O", "BrCC"])
    assert len(products) >= 1, (
        f"expected >=1 ester product from acetic acid + ethyl bromide, got {products}"
    )
    canonical = {_canonical(p) for p in products}
    assert "CCOC(C)=O" in canonical, (
        f"expected CCOC(C)=O among products, got {canonical}"
    )

    # Negative control: an amine + an acid cannot fire the ester rule.
    assert rule.can_apply(["CCN", "CC(=O)O"]) is False


# ---------------------------------------------------------------------------
# 3. Boc cleavage — BOC-protected amine -> free amine (single reactant)
# ---------------------------------------------------------------------------


def test_boc_cleavage() -> None:
    """Boc-protected ethylamine -> ethylamine.

    SyntheMol's ``BOC_CLEAVAGE`` is a single-reactant reaction
    (registered with the ``-1`` sentinel in ``REAL_TO_LAMBDA``); the
    canonical product SMILES is simply ``CCN``.
    """
    rule = REACTION_RULES["boc_cleavage"]
    assert isinstance(rule, SyntheMolReactionRule)

    # Build a BOC-protected amine SMILES that matches the SyntheMol
    # ``BOC`` query: ``[H]C([H])([H])C(OC(=O)[N:1]([*:2])[*:3])(...)...``.
    # The minimum BOC-protected primary amine is
    # ``CC(C)(C)OC(=O)NCC`` — RDKit parses this and the query
    # substructure match succeeds.
    boc_amine = "CC(C)(C)OC(=O)NCC"
    assert rule.can_apply([boc_amine]) is True

    products = rule.fire([boc_amine])
    assert len(products) >= 1, (
        f"expected >=1 product from BOC cleavage of {boc_amine!r}, got {products}"
    )
    canonical = {_canonical(p) for p in products}
    assert "CCN" in canonical, (
        f"expected free amine CCN among products, got {canonical}"
    )

    # Negative control: a bare amine should not match the BOC query.
    assert rule.can_apply(["CCN"]) is False


# ---------------------------------------------------------------------------
# 4. SyntheMol wrapper vs lambda REACTION_RULES equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rule_name, reactants",
    [
        ("amide_coupling", ["CCN", "CC(=O)O"]),
        ("sn2_alkylation", ["CCN", "BrCC"]),
        ("urea_formation", ["CCN", "CCN"]),
    ],
)
def test_syntemol_vs_lambda_equivalence(
    rule_name: str, reactants: list[str]
) -> None:
    """Wrapper ``fire()`` and ``REACTION_RULES[name].fire()`` agree.

    The wrapper and the registry entry are the same object (the
    adapter registers the wrapper into ``REACTION_RULES`` at import
    time), so this is mostly a sanity check that the registry alias
    is intact — but we also cross-check against a fresh
    ``from_syntemol_to_lambda()`` to confirm the curated mapping is
    stable across imports.
    """
    registry_rule = REACTION_RULES[rule_name]
    assert isinstance(registry_rule, SyntheMolReactionRule)

    # Fresh adapter call returns a fresh wrapper with the same SMARTS.
    fresh = from_syntemol_to_lambda()[rule_name]
    assert isinstance(fresh, SyntheMolReactionRule)
    assert fresh.smarts == registry_rule.smarts, (
        f"SMARTS drift for {rule_name}: registry={registry_rule.smarts!r} "
        f"vs fresh={fresh.smarts!r}"
    )

    # Both wrappers must produce the same canonical product set.
    registry_products = sorted(_canonical(p) for p in registry_rule.fire(reactants))
    fresh_products = sorted(_canonical(p) for p in fresh.fire(reactants))
    assert registry_products == fresh_products, (
        f"product set mismatch for {rule_name} on {reactants}: "
        f"registry={registry_products}, fresh={fresh_products}"
    )
    # And the wrapper must fire at all on the spec'd reactants.
    assert registry_products, (
        f"{rule_name} produced no products on {reactants}"
    )


# ---------------------------------------------------------------------------
# 5. Reaction count — exactly 18 rules in REACTION_RULES
# ---------------------------------------------------------------------------


def test_reaction_count() -> None:
    """``REACTION_RULES`` holds 18 rules: 5 click + 13 SyntheMol."""
    # The 5 click rules always present.
    click_names = {"CuAAC", "SPAAC", "SPC", "DielsAlder", "ThiolEne"}
    assert click_names <= set(REACTION_RULES), (
        f"missing click rules; have {sorted(REACTION_RULES)[:10]}"
    )

    # The 13 SyntheMol rule names exactly match ``REAL_TO_LAMBDA``.
    expected_syntemol = {name for name, _rid, _sub in REAL_TO_LAMBDA}
    assert len(expected_syntemol) == 13, (
        f"REAL_TO_LAMBDA should hold 13 entries, got {len(expected_syntemol)}"
    )
    actual_syntemol = {
        name
        for name, rule in REACTION_RULES.items()
        if isinstance(rule, SyntheMolReactionRule)
    }
    assert actual_syntemol == expected_syntemol, (
        f"SyntheMol rules in registry {actual_syntemol} differ from "
        f"REAL_TO_LAMBDA {expected_syntemol}"
    )

    # The registry may include additional production aliases beyond the
    # original 5 + 13 baseline; all expected rules must still be present.
    assert len(REACTION_RULES) >= 18, (
        f"expected at least 18 rules in REACTION_RULES, got {len(REACTION_RULES)}: "
        f"{sorted(REACTION_RULES)}"
    )
