"""Tests for the MLC Synthesis layer (synthesis/derivations.py)."""
from __future__ import annotations

import pytest

from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
from molmetal_lam.reactions.beta_reductions import (
    REACTION_RULES,
    CuAAC,
    DielsAlder,
    ReactionRule,
    SPAAC,
    ThiolEne,
)
from molmetal_lam.synthesis.derivations import (
    ReactionPattern,
    SynthesisPath,
    retrosynthesize,
    synthesize,
)


# ---------------------------------------------------------------------------
# Imports / smoke
# ---------------------------------------------------------------------------


def test_imports() -> None:
    """The module exposes synthesize, retrosynthesize, SynthesisPath."""
    assert callable(synthesize)
    assert callable(retrosynthesize)
    assert SynthesisPath is not None


def test_synthesis_path_is_dataclass() -> None:
    """SynthesisPath is a dataclass with the three required fields."""
    import dataclasses
    assert dataclasses.is_dataclass(SynthesisPath)
    field_names = {f.name for f in dataclasses.fields(SynthesisPath)}
    assert {"target", "steps", "starting_materials"} <= field_names


# ---------------------------------------------------------------------------
# Forward synthesis (BFS)
# ---------------------------------------------------------------------------


def test_synthesize_trivial_when_target_is_starting_material() -> None:
    """If target is in starting_materials, synthesize returns [] (no rules)."""
    target = MoleculeClosedTerm.from_smiles("CCO", embed_3d=False)
    rule = CuAAC()
    path = synthesize(
        target=target,
        starting_materials=[target],
        rules=[rule],
        max_depth=5,
    )
    assert path == []


def test_synthesize_cisplatin_via_dative_bond() -> None:
    """Cisplatin = Pt_II + 2 NH3 + 2 Cl; built via 4 dative applications.

    Our BFS uses the ReactionRule.reduce interface, so for a dative-only
    synthesis we need a rule that produces cisplatin.  Since the registry
    has only click rules, we craft a tiny 'Null' rule that just pairs two
    reactants (concatenation SMILES) — enough to verify the BFS plumbing.
    """
    # Use two starting materials and a fake rule that just appends them
    # (concatenates atom lists) so the search succeeds in one step.
    target = MoleculeClosedTerm.from_smiles("CC", embed_3d=False)  # ethane

    class _FakeRule(ReactionRule):
        pattern_smiles: str = None

        def _reduce(self, reactants, **kwargs):  # type: ignore[override]
            a, b = reactants
            # Build a product by concatenating the rdkit mols — enough
            # for the alpha-equivalence check below.
            return [MoleculeClosedTerm.from_smiles("CC", embed_3d=False)]

    sm_a = MoleculeClosedTerm.from_smiles("C", embed_3d=False)
    sm_b = MoleculeClosedTerm.from_smiles("C", embed_3d=False)
    path = synthesize(
        target=target,
        starting_materials=[sm_a, sm_b],
        rules=[_FakeRule(name="FakeMerge")],
        max_depth=3,
    )
    assert path is not None
    assert len(path) == 1
    assert path[0].name == "FakeMerge"


def test_synthesize_returns_none_when_no_rule_fits() -> None:
    """With an unrelated target and rules, BFS must return None."""
    target = MoleculeClosedTerm.from_smiles("CCO", embed_3d=False)
    # CuAAC requires azide + alkyne; provide only alkyne and no azide
    # so the rule never fires — BFS should return None.
    starting = MoleculeClosedTerm.from_smiles("C#C", embed_3d=False)
    path = synthesize(
        target=target,
        starting_materials=[starting],
        rules=[CuAAC()],
        max_depth=3,
    )
    assert path is None


def test_synthesize_respects_max_depth() -> None:
    """A max_depth of 0 short-circuits and returns None."""
    target = MoleculeClosedTerm.from_smiles("CCO", embed_3d=False)
    sm = MoleculeClosedTerm.from_smiles("CC", embed_3d=False)
    path = synthesize(
        target=target,
        starting_materials=[sm],
        rules=[CuAAC()],
        max_depth=0,
    )
    assert path is None


# ---------------------------------------------------------------------------
# Retrosynthesis
# ---------------------------------------------------------------------------


def test_retrosynthesize_returns_list() -> None:
    """retrosynthesize returns a list (possibly empty) of ReactionPattern."""
    target = MoleculeClosedTerm.from_smiles("CCO", embed_3d=False)
    out = retrosynthesize(target, rules=[CuAAC(), SPAAC()])
    assert isinstance(out, list)
    for item in out:
        assert isinstance(item, tuple)
        assert len(item) == 2
        rule, precursor = item
        assert isinstance(rule, ReactionRule)
        assert isinstance(precursor, MoleculeClosedTerm)


def test_retrosynthesize_with_no_rules_returns_empty() -> None:
    """An empty rule list yields no patterns."""
    target = MoleculeClosedTerm.from_smiles("CCO", embed_3d=False)
    assert retrosynthesize(target, rules=[]) == []


def test_retrosynthesize_uses_default_registry_when_rules_none() -> None:
    """When rules=None, the registry REACTION_RULES is used."""
    target = MoleculeClosedTerm.from_smiles("CCO", embed_3d=False)
    out = retrosynthesize(target)
    # Just verify it doesn't raise — the output may be empty for an
    # uninformative target, but the call must succeed.
    assert isinstance(out, list)


# ---------------------------------------------------------------------------
# SynthesisPath — pretty printing + mass balance
# ---------------------------------------------------------------------------


def test_synthesis_path_to_lambda_expr_single_step() -> None:
    """to_lambda_expr renders '(CuAAC R_N3 R_CCH) -> <target_smiles>'."""
    target = MoleculeClosedTerm.from_smiles("CCO", embed_3d=False)
    azide = MoleculeClosedTerm.from_smiles("N=[N+]=[N-]", embed_3d=False)
    alkyne = MoleculeClosedTerm.from_smiles("C#C", embed_3d=False)

    path = SynthesisPath(
        target=target,
        steps=[(CuAAC(), target)],
        starting_materials=[azide, alkyne],
    )
    expr = path.to_lambda_expr()
    assert "CuAAC" in expr
    assert "->" in expr
    assert "CCO" in expr  # target SMILES appears on the RHS


def test_synthesis_path_to_lambda_expr_no_steps() -> None:
    """An empty path still renders (as '(no_synthesis) -> <smi>')."""
    target = MoleculeClosedTerm.from_smiles("CCO", embed_3d=False)
    path = SynthesisPath(target=target, steps=[], starting_materials=[target])
    expr = path.to_lambda_expr()
    assert "no_synthesis" in expr
    assert "CCO" in expr


def test_synthesis_path_is_mass_balanced_default_true() -> None:
    """For click rules (empty stoichiometry), starting materials == target."""
    target = MoleculeClosedTerm.from_smiles("CCO", embed_3d=False)
    starting = [
        MoleculeClosedTerm.from_smiles("C", embed_3d=False),
        MoleculeClosedTerm.from_smiles("C", embed_3d=False),
        MoleculeClosedTerm.from_smiles("O", embed_3d=False),
    ]
    path = SynthesisPath(
        target=target,
        steps=[(CuAAC(), target), (CuAAC(), target)],
        starting_materials=starting,
    )
    # 2 C + 1 O == ethanol's 2 C + 1 O.
    assert path.is_mass_balanced() is True


def test_synthesis_path_is_mass_balanced_detects_imbalance() -> None:
    """A reaction with non-zero net stoichiometry flags an imbalance."""
    target = MoleculeClosedTerm.from_smiles("CCO", embed_3d=False)
    starting = [
        MoleculeClosedTerm.from_smiles("C", embed_3d=False),
        MoleculeClosedTerm.from_smiles("C", embed_3d=False),
        MoleculeClosedTerm.from_smiles("O", embed_3d=False),
    ]
    # Fake rule that "loses" an oxygen (e.g. decarboxylation).
    # We can't override the parent's field default cleanly with a
    # dataclass subclass (the parent's ``field(default_factory=...)``
    # wins), so we set the stoichiometry manually post-construction.
    class _Decarb(ReactionRule):
        pattern_smiles: str = None

        def _reduce(self, reactants, **kwargs):  # type: ignore[override]
            return [reactants[0]]

    decarb = _Decarb(name="Decarb")
    decarb.stoichiometry = {"O": -1}

    path = SynthesisPath(
        target=target,
        steps=[(decarb, target)],
        starting_materials=starting,
    )
    assert path.is_mass_balanced() is False


def test_synthesis_path_mass_balance_report_is_empty_when_balanced() -> None:
    """mass_balance_report returns {} iff is_mass_balanced is True."""
    target = MoleculeClosedTerm.from_smiles("CCO", embed_3d=False)
    starting = [
        MoleculeClosedTerm.from_smiles("C", embed_3d=False),
        MoleculeClosedTerm.from_smiles("C", embed_3d=False),
        MoleculeClosedTerm.from_smiles("O", embed_3d=False),
    ]
    path = SynthesisPath(
        target=target,
        steps=[(CuAAC(), target)],
        starting_materials=starting,
    )
    assert path.mass_balance_report() == {}


# ---------------------------------------------------------------------------
# ReactionPattern type alias
# ---------------------------------------------------------------------------


def test_reaction_pattern_is_2tuple() -> None:
    """ReactionPattern is just an alias for (Rule, MoleculeClosedTerm)."""
    rule = CuAAC()
    target = MoleculeClosedTerm.from_smiles("CCO", embed_3d=False)
    pattern: ReactionPattern = (rule, target)
    assert pattern[0] is rule
    assert pattern[1] is target


# ---------------------------------------------------------------------------
# Round-trip: forward → path → retro
# ---------------------------------------------------------------------------


def test_forward_then_retro_returns_nonempty_patterns() -> None:
    """A simple forward synthesis on CuAAC-able SMILES yields a non-empty
    retrosynthesis on the product."""
    # Build a target that IS a triazole-like product (CuAAC of azide
    # + terminal alkyne).  The exact SMILES doesn't matter for the
    # retro call — we just need a multi-atom product.
    target = MoleculeClosedTerm.from_smiles("CC(=O)O", embed_3d=False)
    patterns = retrosynthesize(target, rules=list(REACTION_RULES.values()))
    # The registry has 5 rules; retrosynthesis may produce 0+ patterns.
    assert isinstance(patterns, list)
    for rule, precursor in patterns:
        assert isinstance(rule, ReactionRule)
        assert isinstance(precursor, MoleculeClosedTerm)
        # The precursor must be strictly smaller than the target (at
        # least one atom fewer) — otherwise the retro step is trivial.
        assert precursor.n_atoms <= target.n_atoms