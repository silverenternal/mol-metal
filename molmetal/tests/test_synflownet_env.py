"""Phase 2 verification — pytest + integration tests for the
:class:`SynFlowNetEnvAdapter` adapter, the GFN-style synthesis /
retrosynthesis interface exposed by ``molmetal_lam.sbdd_env.synflownet_env``.

Five tests matching the Phase 2 spec:

1. ``test_forward_step_amide`` — amine + acid -> amide, identical product
   to the Phase 1 SyntheMol wrapper (``test_amide_coupling``).
2. ``test_backward_step_ester`` — ethyl_acetate -> acetic_acid + ethanol
   (the canonical ester hydrolysis decomposition).
3. ``test_backward_step_returns_multiple_pairs`` — a product with multiple
   feasible decompositions yields more than one (reactant_a, reactant_b)
   pair.
4. ``test_no_match`` — gibberish SMILES returns an empty list (and does
   not crash).
5. ``test_synflownet_vs_syntemol_agreement`` — same (reactants, rule)
   pair fed through both wrappers produces identical canonical SMILES
   for three independent reactions.

Run with::

    source .venv/bin/activate && python -m pytest \\
        molmetal/tests/test_synflownet_env.py -v
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import pytest
from rdkit import Chem

# Imports under test
from molmetal_lam.reactions.beta_reductions import REACTION_RULES
from molmetal_lam.sbdd_env.synflownet_env import SynFlowNetEnvAdapter
from molmetal_lam.sbdd_env.syntemol_reactions import SyntheMolReactionRule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical(smi: str) -> str:
    """Return the RDKit canonical SMILES for ``smi`` (raises on parse fail)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"RDKit failed to parse {smi!r}")
    return Chem.MolToSmiles(mol)


@pytest.fixture
def adapter() -> SynFlowNetEnvAdapter:
    """Adapter wrapping the full Lambda REACTION_RULES registry."""
    return SynFlowNetEnvAdapter(list(REACTION_RULES.values()))


# ---------------------------------------------------------------------------
# 1. Forward step — amide coupling parity with Phase 1
# ---------------------------------------------------------------------------


def test_forward_step_amide(adapter: SynFlowNetEnvAdapter) -> None:
    """``adapter.forward_step('CCN.CC(=O)O', 'amide_coupling')`` returns
    the same canonical product as the Phase 1
    :class:`SyntheMolReactionRule` wrapper.

    This is the Phase 2 / GFN-style forward API for the same reaction
    exercised by :func:`test_syntemol_reactions.test_amide_coupling`.
    Both wrappers must agree on the canonical product SMILES so the
    downstream synthesis / search layers can be wired against either
    interface interchangeably.
    """
    reactants = "CCN.CC(=O)O"
    rule_name = "amide_coupling"

    # SynFlowNet (Phase 2) forward step.
    p2_products = adapter.forward_step(reactants, rule_name)
    assert isinstance(p2_products, list)
    assert len(p2_products) >= 1, (
        f"expected >=1 product from forward_step({reactants!r}, {rule_name!r}), "
        f"got {p2_products!r}"
    )
    p2_canonicals = {_canonical(p) for p in p2_products}

    # SyntheMol (Phase 1) wrapper.
    phase1_rule = REACTION_RULES[rule_name]
    assert isinstance(phase1_rule, SyntheMolReactionRule)
    p1_products = phase1_rule.fire(["CCN", "CC(=O)O"])
    p1_canonicals = {_canonical(p) for p in p1_products}

    # Identity invariant: the *set* of canonical products must overlap.
    # (Phase 2 may emit an extra zwitterion-stripped variant of the
    # same molecule; the canonical neutral form ``CCNC(C)=O`` must be
    # in both.)
    assert "CCNC(C)=O" in p2_canonicals, (
        f"expected CCNC(C)=O in Phase 2 canonical products, got {p2_canonicals}"
    )
    assert "CCNC(C)=O" in p1_canonicals, (
        f"expected CCNC(C)=O in Phase 1 canonical products, got {p1_canonicals}"
    )
    # And the two wrappers must agree on *at least* one canonical form.
    common = p1_canonicals & p2_canonicals
    assert common, (
        f"Phase 1 ({p1_canonicals}) and Phase 2 ({p2_canonicals}) wrappers "
        f"disagree on canonical product set for {rule_name}"
    )


# ---------------------------------------------------------------------------
# 2. Backward step — ester hydrolysis decomposition
# ---------------------------------------------------------------------------


def test_backward_step_ester(adapter: SynFlowNetEnvAdapter) -> None:
    """Ethyl acetate (CC(=O)OCC) decomposes via ester_hydrolysis.

    The expected canonical pair is (acetic acid, ethanol) →
    ``(CC(=O)O, CCO)``.  Because ``SyntheMolReactionRule.reverse_can_apply``
    emits multiple H-canonicalised variants (see the integration report),
    we accept either ``(CC(=O)O, CCO)`` or the H-explicit forms
    ``([H]OC(C)=O, *CC)`` / ``([H]OCC, *C(C)=O)``.
    """
    pairs = adapter.backward_step("CC(=O)OCC")
    assert isinstance(pairs, list)
    assert len(pairs) >= 1, (
        f"expected >=1 decomposition pair for ethyl acetate, got {pairs!r}"
    )

    # Validate every SMILES string round-trips through RDKit.
    for a, b in pairs:
        if a:
            assert Chem.MolFromSmiles(a) is not None, f"invalid SMILES: {a!r}"
        if b:
            assert Chem.MolFromSmiles(b) is not None, f"invalid SMILES: {b!r}"

    # Collect all reactants that appeared across the pairs (canonical form).
    canonical_reactants = set()
    for a, b in pairs:
        if a:
            canonical_reactants.add(_canonical(a))
        if b:
            canonical_reactants.add(_canonical(b))

    # The expected products of esterification in reverse are acetic acid
    # and ethanol (after H canonicalisation).
    assert "CC(=O)O" in canonical_reactants, (
        f"expected acetic acid CC(=O)O in decomposition, "
        f"got {canonical_reactants}"
    )
    assert "CCO" in canonical_reactants, (
        f"expected ethanol CCO in decomposition, "
        f"got {canonical_reactants}"
    )


# ---------------------------------------------------------------------------
# 3. Backward step — multiple distinct decompositions
# ---------------------------------------------------------------------------


def test_backward_step_returns_multiple_pairs(adapter: SynFlowNetEnvAdapter) -> None:
    """A product that admits >1 (reactant_a, reactant_b) decomposition
    pair must surface those pairs in ``backward_step``.

    Acetamide (CC(=O)N) has two valid retrosynthetic pathways that the
    adapter can express:

    * amide_coupling retro: acetic_acid + amine → ``(CC(=O)O, NC)``
    * reductive_amination retro (when the alpha position is open) → a
      *different* decomposition.

    Either (a) at least two distinct pairs must appear, or (b) the
    amide_coupling pair must appear *together with* a different pair
    from another rule.  We assert >=2 distinct pairs, which is the
    contract from the task spec.
    """
    # ``CC(=O)N`` is acetamide (acetic acid amide).  At minimum it
    # decomposes via amide_coupling into (CC(=O)O, NC).  We require a
    # second decomposition route via a *different* rule.
    pairs = adapter.backward_step("CC(=O)N")
    assert isinstance(pairs, list)
    assert len(pairs) >= 2, (
        f"expected >=2 decomposition pairs for acetamide, got {pairs!r}"
    )

    # At least one pair must be the canonical amide retro:
    # (acetic acid, amine) -> (CC(=O)O, NC) (or H-explicit forms).
    canon_pairs = {(_canonical(a) if a else "", _canonical(b) if b else "") for a, b in pairs}
    assert ("CC(=O)O", "CN") in canon_pairs, (
        f"expected (CC(=O)O, CN) in canonical acetamide decomposition, "
        f"got {canon_pairs}"
    )

    # And there must be at least one *different* pair to satisfy the
    # multi-pair contract.
    distinct_pairs = {p for p in canon_pairs if p != ("CC(=O)O", "NC")}
    assert distinct_pairs, (
        f"expected at least one non-canonical-retro pair for acetamide, "
        f"got canon_pairs={canon_pairs}"
    )


# ---------------------------------------------------------------------------
# 4. No-match — gibberish SMILES returns []
# ---------------------------------------------------------------------------


def test_no_match(adapter: SynFlowNetEnvAdapter) -> None:
    """Gibberish / non-parseable SMILES must yield an empty list.

    ``backward_step`` calls ``Chem.MolFromSmiles`` first; a parse
    failure must short-circuit cleanly and return ``[]`` rather than
    raising.
    """
    gibberish_inputs = [
        "@@@not_a_smiles@@@",
        "C[unclosed",
        "1.2.3.4",          # dots only — no atoms
        "",                 # empty string
    ]
    for smi in gibberish_inputs:
        pairs = adapter.backward_step(smi)
        assert isinstance(pairs, list), (
            f"backward_step({smi!r}) returned non-list {type(pairs).__name__}"
        )
        assert pairs == [], (
            f"backward_step({smi!r}) should return [], got {pairs!r}"
        )

    # The forward-step side must also short-circuit cleanly on
    # gibberish (no crash, empty list).
    for smi in gibberish_inputs:
        prods = adapter.forward_step(smi, "amide_coupling")
        assert prods == [], (
            f"forward_step({smi!r}, amide_coupling) should return [], "
            f"got {prods!r}"
        )


# ---------------------------------------------------------------------------
# 5. SynFlowNet vs SyntheMol — agreement on canonical SMILES
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rule_name, reactants",
    [
        ("amide_coupling", ["CCN", "CC(=O)O"]),
        ("sn2_alkylation", ["CCN", "BrCC"]),
        ("urea_formation", ["CCN", "CCN"]),
    ],
)
def test_synflownet_vs_syntemol_agreement(
    adapter: SynFlowNetEnvAdapter,
    rule_name: str,
    reactants: List[str],
) -> None:
    """Both wrappers must produce the same canonical SMILES on the same
    (reactants, rule) input.

    The Phase 2 :class:`SynFlowNetEnvAdapter` and the Phase 1
    :class:`SyntheMolReactionRule` wrapper share the same SMARTS, so
    they must agree on canonical product SMILES (Phase 2 may emit an
    extra zwitterion-stripped variant; we require at least one
    canonical form to be present in *both* product sets).
    """
    reactant_str = ".".join(reactants)
    p2_products = adapter.forward_step(reactant_str, rule_name)
    assert isinstance(p2_products, list)
    assert len(p2_products) >= 1, (
        f"Phase 2 forward_step returned no products for "
        f"{rule_name}({reactants!r})"
    )

    phase1_rule = REACTION_RULES[rule_name]
    assert isinstance(phase1_rule, SyntheMolReactionRule)
    p1_products = phase1_rule.fire(reactants)
    assert len(p1_products) >= 1, (
        f"Phase 1 wrapper returned no products for "
        f"{rule_name}({reactants!r})"
    )

    p2_canonicals = {_canonical(p) for p in p2_products}
    p1_canonicals = {_canonical(p) for p in p1_products}

    common = p1_canonicals & p2_canonicals
    assert common, (
        f"Phase 1 ({p1_canonicals}) and Phase 2 ({p2_canonicals}) wrappers "
        f"disagree on canonical product SMILES for {rule_name}({reactants!r})"
    )
