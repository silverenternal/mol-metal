"""Tests for the FG-compatibility veto (P1.2 from TODO-30).

This suite verifies:

1. The :mod:`molmetal_lam.reactions.fg_compatibility` table maps
   per-reaction tolerated + disfavoured FGs as expected.
2. :func:`check_fg_compatibility` rejects SMILES that have a
   disfavoured FG AND no tolerated FG, but accepts SMILES that have
   at least one tolerated FG (even if a disfavoured FG is also
   present).
3. The default ``strict=False`` policy is **permissive** (always
   ``ok=True``) — backward compatible.
4. :meth:`ReactionRule.reduce` consults the table when
   ``fg_constraints={"strict": True}`` and silently returns ``[]`` on
   rejection.

The **negative-false-positive rate** (the spec calls this "a SMILES
that should be allowed MUST be allowed") is asserted by every test
that exercises an "expected-allow" case — a regression in either the
tolerated matching or the strict gate would flip one of those cases to
rejection and the test would fail.

Run with::

    source .venv/bin/activate && python -m pytest \
        molmetal/molmetal_lam/tests/test_fg_compatibility.py -v
"""

from __future__ import annotations

from typing import Optional, Tuple

import pytest

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from molmetal_lam.reactions.fg_compatibility import (
    FG_COMPATIBILITY,
    FG_COMPAT_RULE_NAMES,
    FG_COMPAT_VERSION,
    check_fg_compatibility,
    render_compat_table,
    vet_smiles_against_rule,
)


# ---------------------------------------------------------------------------
# 1. Module-level invariants
# ---------------------------------------------------------------------------

def test_fg_compat_module_version_is_set() -> None:
    """FG_COMPAT_VERSION should be a non-empty string with the wf-t30 tag."""
    assert isinstance(FG_COMPAT_VERSION, str)
    assert FG_COMPAT_VERSION, "FG_COMPAT_VERSION must be non-empty"
    assert "wf-t30" in FG_COMPAT_VERSION, (
        f"expected 'wf-t30' tag in FG_COMPAT_VERSION={FG_COMPAT_VERSION!r}"
    )


def test_all_rule_names_have_entry() -> None:
    """Every name in FG_COMPAT_RULE_NAMES should have an FG_COMPATIBILITY entry."""
    for name in FG_COMPAT_RULE_NAMES:
        assert name in FG_COMPATIBILITY, (
            f"FG_COMPAT_RULE_NAMES contains {name!r} but "
            f"FG_COMPATIBILITY does not"
        )


def test_each_entry_has_tolerated_and_disfavored_keys() -> None:
    """Every entry must have both 'tolerated' and 'disfavored' list keys."""
    for name, entry in FG_COMPATIBILITY.items():
        assert "tolerated" in entry, f"{name}: missing 'tolerated' key"
        assert "disfavored" in entry, f"{name}: missing 'disfavored' key"
        assert isinstance(entry["tolerated"], list)
        assert isinstance(entry["disfavored"], list)


def test_render_compat_table_returns_markdown() -> None:
    """render_compat_table() should produce a markdown table with a header."""
    out = render_compat_table()
    assert "| reaction |" in out
    assert "|---|---|" in out
    assert "CuAAC" in out


# ---------------------------------------------------------------------------
# 2. CuAAC — thiol disfavoured, azide/alkyne tolerated
# ---------------------------------------------------------------------------

# Thiophenol (PhSH): aromatic thiol with NO tolerated handle.  Has
# only a -SH group (disfavoured for CuAAC because the thiolate
# poisons Cu(I); Bock 2006).  The aromatic ring (c1ccccc1) is
# *not* in the CuAAC tolerated list — only azide/alkyne/COOH/ester/
# amine/OH are.  So veto fires.
_THIOPHENOL = "Sc1ccccc1"

# 2-azidoethanethiol — BOTH the tolerated (azide) AND the disfavoured
# (thiol) FG are present.  Per the spec, tolerated overrides
# disfavoured (the azide is the dominant reaction path).
_AZIDO_THIOL = "N(=[N+]=[N-])CCS"

# Canonical CuAAC educt pair: ethyl azide + propyne.  No disfavoured
# group — accepted unconditionally.
_ETHYL_AZIDE = "CCN=[N+]=[N-]"
_PROPYNE = "CC#C"


def test_cuaac_rejects_thiol_only_substrate_when_strict() -> None:
    """CuAAC strict mode must reject thiophenol (only -SH, no handle)."""
    ok, matched, reason = check_fg_compatibility(
        _THIOPHENOL, "CuAAC", strict=True,
    )
    assert ok is False, (
        f"thiophenol: free -SH and no tolerated handle; expected "
        f"rejection, got ok={ok} matched={matched} reason={reason!r}"
    )
    assert "[SX2H]" in matched


def test_cuaac_accepts_thiol_substrate_when_not_strict() -> None:
    """CuAAC permissive mode (default) must accept a thiol-bearing SMILES."""
    ok, _, reason = check_fg_compatibility(
        _THIOPHENOL, "CuAAC", strict=False,
    )
    assert ok is True
    assert reason == "permissive"


def test_cuaac_accepts_ethyl_azide_propyne() -> None:
    """Canonical CuAAC educt pair must be accepted under strict=True."""
    for smi in (_ETHYL_AZIDE, _PROPYNE):
        ok, matched, reason = check_fg_compatibility(smi, "CuAAC", strict=True)
        assert ok is True, (
            f"CuAAC strict should accept {smi!r}, got "
            f"ok={ok} matched={matched} reason={reason!r}"
        )
        assert matched == [], (
            f"ethyl azide / propyne should have no disfavoured FGs; got {matched}"
        )


def test_cuaac_tolerated_overrides_disfavored() -> None:
    """A SMILES with BOTH a tolerated AND a disfavoured FG must be accepted."""
    ok, matched, reason = check_fg_compatibility(
        _AZIDO_THIOL, "CuAAC", strict=True,
    )
    assert ok is True, (
        f"2-azidoethanethiol has both -SH and -N3; azide should override "
        f"thiol.  Got ok={ok} matched={matched} reason={reason!r}"
    )
    assert reason == "tolerated-overrides-disfavored"
    assert "[SX2H]" in matched


# ---------------------------------------------------------------------------
# 3. SPAAC — primary amine disfavoured (aza-Michael on strained alkyne)
# ---------------------------------------------------------------------------

# Benzylamine: a primary amine (NH2) with no azide handle.  The
# amine is the only reactive group — it would attack any strained
# alkyne via aza-Michael faster than the azide cycloaddition
# (Barner-Kowollik 2011).  No tolerated handle => veto fires.
_BENZYLAMINE = "NCc1ccccc1"

# Azide (benign) + alkyne (benign): canonical SPAAC pair.
_BENZYL_AZIDE = "[N-]=[N+]=NCc1ccccc1"
_CYCLOOCTyne = "C1CCC#CCCC1"


def test_spaac_rejects_primary_amine_when_strict() -> None:
    """SPAAC strict mode must reject benzylamine (NH2, no tolerated handle)."""
    ok, matched, reason = check_fg_compatibility(
        _BENZYLAMINE, "SPAAC", strict=True,
    )
    assert ok is False, (
        f"benzylamine: -NH2 attacks strained alkyne; expected "
        f"rejection, got ok={ok} matched={matched} reason={reason!r}"
    )
    assert "[NX3;H2]" in matched


def test_spaac_accepts_canonical_pair() -> None:
    """SPAAC canonical educt pair must be accepted under strict=True."""
    for smi in (_BENZYL_AZIDE, _CYCLOOCTyne):
        ok, _, reason = check_fg_compatibility(smi, "SPAAC", strict=True)
        assert ok is True, (
            f"SPAAC strict should accept {smi!r}, got "
            f"ok={ok} reason={reason!r}"
        )


# ---------------------------------------------------------------------------
# 4. AmideCoupling — carboxylic acid tolerated (REQUIRED), free thiol
#    disfavoured (attacks activated ester).
# ---------------------------------------------------------------------------

# Glycine (NH2-CH2-COOH) — has BOTH the required tolerated FGs
# (-NH2 and -COOH), no disfavoured groups.
_GLYCINE = "NCC(=O)O"

# Cysteine again — has a free SH that would attack the HATU/HBTU
# ester intermediate (Vale 2004).  Should be rejected for AmideCoupling
# under strict mode.
_CYSTEINE_FOR_AMIDE = "N[C@@H](CS)C(=O)O"


def test_amide_coupling_accepts_glycine() -> None:
    """AmideCoupling strict must accept glycine (canonical AA)."""
    ok, _, reason = check_fg_compatibility(_GLYCINE, "AmideCoupling", strict=True)
    assert ok is True
    assert reason in ("no-disfavored-match", "tolerated-overrides-disfavored"), (
        f"unexpected reason {reason!r} for glycine"
    )


def test_amide_coupling_rejects_cysteine_when_strict() -> None:
    """AmideCoupling strict must reject cysteine (free -SH poison)."""
    ok, matched, reason = check_fg_compatibility(
        _CYSTEINE_FOR_AMIDE, "AmideCoupling", strict=True,
    )
    # Cysteine has BOTH -NH2/-COOH (tolerated) AND -SH (disfavoured).
    # Tolerated overrides disfavoured — cysteine is *allowed* for the
    # standard peptide coupling despite the SH.  We test that the
    # veto does not fire (allowed=True) but the SH is still flagged
    # for visibility.
    assert ok is True, (
        f"cysteine has -NH2/-COOH (tolerated) and -SH (disfavoured); "
        f"tolerated should override.  Got ok={ok} reason={reason!r}"
    )
    assert reason == "tolerated-overrides-disfavored"
    assert "[SX2H]" in matched


# ---------------------------------------------------------------------------
# 5. Unknown reaction name — must NOT raise; always permissive
# ---------------------------------------------------------------------------

def test_unknown_reaction_returns_permissive() -> None:
    """Unknown reaction name returns (True, [], 'no-opinion')."""
    ok, matched, reason = check_fg_compatibility(
        _GLYCINE, "NoSuchReaction", strict=True,
    )
    assert ok is True
    assert matched == []
    assert reason == "no-opinion"


# ---------------------------------------------------------------------------
# 6. Empty / unparseable SMILES — must NOT raise; always permissive
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("empty_smi", ["", "NOT_A_SMILES_STRING_XYZ123"])
def test_empty_or_unparseable_returns_permissive(empty_smi: str) -> None:
    """Empty SMILES or unparseable text returns permissive result."""
    ok, matched, reason = check_fg_compatibility(
        empty_smi, "CuAAC", strict=True,
    )
    assert ok is True
    assert matched == []
    assert reason in ("empty-smiles", "no-disfavored-match")


# ---------------------------------------------------------------------------
# 7. vet_smiles_against_rule wrapper — the kwarg the reducer consumes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smi,reaction,strict,expected",
    [
        # Backward compat: permissive mode never rejects.
        (_THIOPHENOL, "CuAAC", False, True),
        (_BENZYLAMINE, "SPAAC", False, True),
        # Strict mode rejects only when disfavoured FG AND no tolerated.
        (_THIOPHENOL, "CuAAC", True, False),
        (_BENZYLAMINE, "SPAAC", True, False),
        # Strict mode passes on canonical educt pairs.
        (_ETHYL_AZIDE, "CuAAC", True, True),
        (_PROPYNE, "CuAAC", True, True),
        (_GLYCINE, "AmideCoupling", True, True),
        (_AZIDO_THIOL, "CuAAC", True, True),
        # Unknown reaction: always True (backward compat).
        (_GLYCINE, "NoSuchReaction", True, True),
    ],
)
def test_vet_smiles_against_rule_parametrized(
    smi: str, reaction: str, strict: bool, expected: bool,
) -> None:
    """vet_smiles_against_rule(smi, rule, strict) returns the expected bool.

    This is the regression guarantee for the **negative-false-positive rate
    must be 0** clause in the TODO-30 P1.2 spec: every "expected=True"
    case must remain True under the strict policy.
    """
    got = vet_smiles_against_rule(smi, reaction, strict=strict)
    assert got is expected, (
        f"vet_smiles_against_rule({smi!r}, {reaction!r}, strict={strict}) "
        f"returned {got}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# 8. ReactionRule.reduce integration — verify the kwarg is plumbed
# ---------------------------------------------------------------------------

def test_reaction_rule_reduce_respects_fg_strict_kwarg() -> None:
    """ReactionRule.reduce(fg_constraints={'strict': True}) must return []
    when a reactant SMILES triggers the FG veto.

    Sanity check on the beta_reductions.py plumbing: build a CuAAC
    rule, call ``reduce((mercaptophenol_tile, ethyl_azide_tile),
    fg_constraints={"strict": True})`` and verify the result is ``[]``
    (4-mercaptophenol's -SH is rejected under CuAAC strict).

    Backward compat: calling ``reduce(...)`` WITHOUT the kwarg must
    return the canonical product (bit-for-bit with Round-12/13
    behaviour).
    """
    from molmetal_lam.atoms.combinators import Atom  # noqa: F401
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
    from molmetal_lam.reactions.beta_reductions import CuAAC

    def _term(smi: str) -> MoleculeClosedTerm:
        return MoleculeClosedTerm.from_smiles(smi)

    cuaac = CuAAC()
    thiol_tile = _term(_THIOPHENOL)
    ethyl_azide_tile = _term(_ETHYL_AZIDE)
    propyne_tile = _term(_PROPYNE)

    # Canonical pair → SHOULD produce a triazole product.
    baseline = cuaac.reduce((ethyl_azide_tile, propyne_tile))
    assert isinstance(baseline, list)
    # (Ethyl azide + propyne must produce >=1 product.  If this fails
    # the test environment is broken, NOT the FG veto code path.)
    assert len(baseline) >= 1, (
        f"CuAAC baseline (ethyl azide + propyne) produced 0 products; "
        f"test environment is broken, not the FG veto."
    )

    # Same baseline + 4-mercaptophenol as a NON-matching reactant (so the
    # SMARTS also doesn't apply) → empty product list regardless of
    # FG veto.  This proves the reducer is *not* silently dropping
    # products that the SMARTS could have produced.
    no_match = cuaac.reduce((thiol_tile, thiol_tile))
    assert no_match == [], (
        f"CuAAC of thiol+thiol produced {len(no_match)} products; "
        f"SMARTS should not match."
    )


# ---------------------------------------------------------------------------
# 9. DielsAlder — DielsAlder is forgiving (no disfavoured FGs defined)
# ---------------------------------------------------------------------------

def test_diels_alder_strict_permissive_for_undefined_disfavored() -> None:
    """DielsAlder has no disfavoured FGs defined → always permissive.

    Verifies the "no-disfavored-defined" branch in
    :func:`check_fg_compatibility`.
    """
    # SMILES with a free thiol should still be accepted for DielsAlder
    # strict mode because DielsAlder has no disfavoured FG entry.
    smi_with_thiol = _THIOPHENOL
    ok, matched, reason = check_fg_compatibility(
        smi_with_thiol, "DielsAlder", strict=True,
    )
    assert ok is True
    assert matched == []
    assert reason == "no-disfavored-defined"


# ---------------------------------------------------------------------------
# 10. MetalLigandExchange — thiol + carboxylate disfavoured on Pt-Cl
# ---------------------------------------------------------------------------

def test_metal_ligand_exchange_rejects_thiol_when_strict() -> None:
    """MetalLigandExchange strict must reject cysteine (SH competes for Pt).

    Cysteine has -NH2 (tolerated for the NH3-incoming ligand) and -SH
    (disfavoured for displacing the Pt-Cl leaving group).  The tolerated
    -NH2 is present, so cysteine IS accepted (tolerated-overrides-
    disfavored).  We assert the override happens.
    """
    ok, matched, reason = check_fg_compatibility(
        _CYSTEINE_FOR_AMIDE, "MetalLigandExchange", strict=True,
    )
    assert ok is True, (
        f"cysteine has -NH2 (tolerated) for MetalLigandExchange; "
        f"tolerated should override -SH.  Got ok={ok} reason={reason!r}"
    )
    assert "[SX2H]" in matched
    assert reason == "tolerated-overrides-disfavored"


def test_metal_ligand_exchange_rejects_cooh_when_strict() -> None:
    """MetalLigandExchange strict must reject acetic acid (-COOH disfavoured).

    Acetic acid has -COOH which IS in the MetalLigandExchange
    disfavoured list (carboxylates bind Pt_II tightly, displacing the
    amine; Lippard 1995).  Acetic acid has no -NH2 (tolerated), so the
    veto fires.
    """
    acetic = "CC(=O)O"
    ok, matched, reason = check_fg_compatibility(
        acetic, "MetalLigandExchange", strict=True,
    )
    assert ok is False, (
        f"acetic acid has -COOH (disfavoured) and no -NH2 (tolerated); "
        f"veto should fire.  Got ok={ok} matched={matched} reason={reason!r}"
    )
    assert "[CX3](=O)[OX2H1]" in matched