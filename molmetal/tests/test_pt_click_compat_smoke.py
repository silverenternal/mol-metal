"""Scaffold-aware click-rule compatibility (pt_click_compat) regression tests.

Background
----------
TODO-30 Rank-3 (P1.4 — Known-Pt regression tests).  This file is pure
INSURANCE — every test should pass on the existing compatibility
matrix as shipped in
``molmetal/molmetal_lam/lam_chem/pt_click_compat.py``.  The matrix
encodes the chemistry-knowledge verdict (per WF-MCTS-Chemistry-Research
2026-09-14) on which click reactions are safe to fire on each metal-
scaffold category without violating the metal-scaffold's reactivity:

  5 scaffolds (rows):
    strict_Pt_II   — d8 square-planar (cisplatin, nedaplatin)
    Pt_II_chelating — d8 with chelating carboxylate (carboplatin)
    Pt_IV          — d6 octahedral prodrug (satraplatin)
    labile_metal   — Cu / Zn / Fe / Mn (fast ligand exchange)
    unknown        — fallback

  7 rules (columns; F2(a) extension adds 2 metal-coord rules):
    CuAAC, SPAAC, ThiolEne, Suzuki, AmideCoupling,
    metal_ligand_exchange, aqua_exchange

Verdicts (one of):
    compatible     — always safe to fire
    marginal       — fires with known side-reactions (opt-in)
    incompatible   — breaks the metal-scaffold (opt-in only + WARNING)

Honest framing
--------------
* The matrix is HAND-CURATED from WF-MCTS-Chemistry-Research + Lippard
  1995 + Reedijk 1987 + Comba-Hambley 2009 — NOT learned.
* ``strict_pt_ii_allows_metal_coordination()`` is the F2(a) invariant
  pinned by this test suite so an accidental future tightening would
  fail loudly.
"""

from __future__ import annotations

import pytest

# All tests in this file are CPU-only (no GPU, no RDKit embedding).
from molmetal.molmetal_lam.lam_chem.pt_click_compat import (
    COMPAT_MATRIX,
    CLICK_RULE_NAMES,
    SCAFFOLD_NAMES,
    default_compatible_rules,
    detect_scaffold,
    incompatible_rules,
    marginal_rules,
    render_compat_table,
    strict_pt_ii_allows_metal_coordination,
    _name_hint_to_scaffold,
)


# ---------------------------------------------------------------------------
# Tests — full 5 scaffolds × 5 click rules = 25 cells (the F2(a)-core subset)
# ---------------------------------------------------------------------------
def test_compat_matrix_full_grid():
    """Pin every 5x5 cell (5 scaffolds × 5 click rules = 25 verdicts).

    The matrix is the source of truth for which click rule can fire on
    each metal-scaffold without violating reactivity.  We assert each
    cell is one of the three valid verdict strings, AND we pin the
    *content* of each cell so an accidental edit would fail loudly.
    """
    click_rules_5 = ("CuAAC", "SPAAC", "ThiolEne", "Suzuki", "AmideCoupling")
    # ------------------------------------------------------------------
    # strict_Pt_II row (d8 square-planar; cisplatin-like):
    #   CuAAC / SPAAC      = compatible  (cycloaddition on remote handle)
    #   ThiolEne           = incompatible (thiolate attacks Pt-Cl)
    #   Suzuki             = marginal     (transmetalation side-reaction)
    #   AmideCoupling      = incompatible (Pt-Cl2 has no carboxylate to couple)
    # ------------------------------------------------------------------
    s = COMPAT_MATRIX["strict_Pt_II"]
    assert s["CuAAC"] == "compatible"
    assert s["SPAAC"] == "compatible"
    assert s["ThiolEne"] == "incompatible"
    assert s["Suzuki"] == "marginal"
    assert s["AmideCoupling"] == "incompatible"

    # ------------------------------------------------------------------
    # Pt_II_chelating row (carboplatin / oxaliplatin; CBDCA chelate):
    #   ThiolEne           = compatible   (carboxylate-thiol exchange)
    #   AmideCoupling      = compatible   (CBDCA carboxylate can couple)
    #   Suzuki             = marginal     (transmetalation possible)
    #   CuAAC / SPAAC      = compatible
    # ------------------------------------------------------------------
    s = COMPAT_MATRIX["Pt_II_chelating"]
    assert s["CuAAC"] == "compatible"
    assert s["SPAAC"] == "compatible"
    assert s["ThiolEne"] == "compatible"
    assert s["Suzuki"] == "marginal"
    assert s["AmideCoupling"] == "compatible"

    # ------------------------------------------------------------------
    # Pt_IV row (satraplatin / tetraplatin; d6 octahedral prodrug):
    #   all 5 = compatible  (kinetically inert until intracellular
    #                        reduction to Pt(II))
    # ------------------------------------------------------------------
    s = COMPAT_MATRIX["Pt_IV"]
    for rule in click_rules_5:
        assert s[rule] == "compatible", (
            f"Pt_IV must allow all 5 click rules; got {s[rule]} for {rule}"
        )

    # ------------------------------------------------------------------
    # labile_metal row (Cu / Zn / Fe / Mn; fast ligand exchange):
    #   all 5 = compatible
    # ------------------------------------------------------------------
    s = COMPAT_MATRIX["labile_metal"]
    for rule in click_rules_5:
        assert s[rule] == "compatible", (
            f"labile_metal must allow all 5 click rules; got {s[rule]} for {rule}"
        )

    # ------------------------------------------------------------------
    # unknown row (fallback):
    #   all 5 = compatible  (conservative default)
    # ------------------------------------------------------------------
    s = COMPAT_MATRIX["unknown"]
    for rule in click_rules_5:
        assert s[rule] == "compatible", (
            f"unknown scaffold must default to all 5 compatible; "
            f"got {s[rule]} for {rule}"
        )

    # Structural invariant: the matrix must contain ALL 5 scaffold rows.
    assert set(COMPAT_MATRIX.keys()) == set(SCAFFOLD_NAMES)
    # The CLICK_RULE_NAMES tuple must contain all 5 click rules.
    for rule in click_rules_5:
        assert rule in CLICK_RULE_NAMES
    # Plus the 2 F2(a) metal-coordination columns.
    assert "metal_ligand_exchange" in CLICK_RULE_NAMES
    assert "aqua_exchange" in CLICK_RULE_NAMES
    # Total column count = 7 (5 click + 2 metal-coord).
    assert len(CLICK_RULE_NAMES) == 7
    # Total scaffold count = 5.
    assert len(SCAFFOLD_NAMES) == 5


# ---------------------------------------------------------------------------
# Per-cell compatibility — strict_Pt_II verdicts via default_compatible_rules
# ---------------------------------------------------------------------------
def test_strict_pt_ii_cuacc_allowed():
    """CuAAC on strict_Pt_II must be in the default-compatible set."""
    rules = default_compatible_rules("strict_Pt_II")
    assert "CuAAC" in rules, (
        f"CuAAC must be compatible on strict_Pt_II; got {rules}"
    )
    # SPAAC is the other safe cycloaddition on strict Pt_II.
    assert "SPAAC" in rules
    # And the F2(a) metal-coordination rules are explicitly compatible
    # on strict_Pt_II (this is the invariant pinned by
    # strict_pt_ii_allows_metal_coordination).
    assert "metal_ligand_exchange" in rules
    assert "aqua_exchange" in rules


def test_strict_pt_ii_thiol_ene_disallowed():
    """ThiolEne on strict_Pt_II must NOT be in the default-compatible set.

    The thiolate attacks the soft Lewis-acidic Pt-Cl bond, protonates
    + ejects Cl-, breaking the Pt-Cl bond.  Opt-in via
    ``allow_incompatible=True`` only.
    """
    # Default-compatible: ThiolEne is INCOMPATIBLE -> excluded.
    rules_default = default_compatible_rules("strict_Pt_II")
    assert "ThiolEne" not in rules_default, (
        f"ThiolEne must be excluded from default-compatible on strict_Pt_II; "
        f"got {rules_default}"
    )
    # Opt-in via allow_incompatible: now ThiolEne is included.
    rules_opt = default_compatible_rules(
        "strict_Pt_II", allow_incompatible=True
    )
    assert "ThiolEne" in rules_opt, (
        "ThiolEne must be included when allow_incompatible=True"
    )
    # AmideCoupling: same story — incompatible on strict_Pt_II by
    # default, opt-in only.
    assert "AmideCoupling" not in rules_default
    assert "AmideCoupling" in rules_opt


# ---------------------------------------------------------------------------
# Labile metal — promiscuous: all 5 click rules allowed
# ---------------------------------------------------------------------------
def test_labile_pt_promiscuous():
    """labile_metal allows all 5 click rules + the 2 metal-coord rules."""
    rules = default_compatible_rules("labile_metal")
    expected = {
        "CuAAC", "SPAAC", "ThiolEne", "Suzuki", "AmideCoupling",
        "metal_ligand_exchange", "aqua_exchange",
    }
    assert set(rules) == expected, (
        f"labile_metal must allow all 7 rules; got {set(rules)}"
    )


def test_unknown_scaffold_promiscuous():
    """Unknown scaffold -> conservative default = all 7 rules allowed."""
    rules = default_compatible_rules("unknown")
    expected = {
        "CuAAC", "SPAAC", "ThiolEne", "Suzuki", "AmideCoupling",
        "metal_ligand_exchange", "aqua_exchange",
    }
    assert set(rules) == expected, (
        f"unknown scaffold must default to all 7 rules; got {set(rules)}"
    )


# ---------------------------------------------------------------------------
# Scaffold auto-detection from canonical SMILES
# ---------------------------------------------------------------------------
def test_auto_pt_strict_picks_cuacc_only():
    """``detect_scaffold(cisplatin_smiles)`` -> ``strict_Pt_II`` and
    the default-compatible subset on strict_Pt_II includes CuAAC but
    NOT ThiolEne.

    End-to-end check: the friendly SMILES ``N.N.[Cl][Pt][Cl]`` must be
    classified as ``strict_Pt_II`` (no carboxylate, no octahedral
    coordination), and the auto-compat subset derived from that
    classification must follow the per-cell matrix assertions.
    """
    cisplatin_smiles = "N.N.[Cl][Pt][Cl]"
    # Auto-detect scaffold.
    scaffold = detect_scaffold(cisplatin_smiles)
    assert scaffold == "strict_Pt_II", (
        f"cisplatin SMILES must detect as strict_Pt_II; got {scaffold}"
    )
    # Derive default-compatible subset.
    rules = default_compatible_rules(scaffold)
    # CuAAC + SPAAC + the 2 metal-coord rules must be present.
    assert "CuAAC" in rules
    assert "SPAAC" in rules
    assert "metal_ligand_exchange" in rules
    assert "aqua_exchange" in rules
    # ThiolEne + AmideCoupling must be excluded.
    assert "ThiolEne" not in rules
    assert "AmideCoupling" not in rules
    # Suzuki is MARGINAL -> included by default
    # (the helper always includes marginal rules).
    assert "Suzuki" in rules, (
        "Suzuki is MARGINAL on strict_Pt_II -> always included"
    )


# ---------------------------------------------------------------------------
# Per-scaffold detection — friendly-name overrides
# ---------------------------------------------------------------------------
def test_friendly_name_carboplatin_chelating():
    """``detect_scaffold(smiles, name_hint='carboplatin')`` -> Pt_II_chelating."""
    scaffold = detect_scaffold(
        "N.N.[Pt]12(OC(=O)CCC1)OC(=O)CCC2",
        name_hint="carboplatin",
    )
    assert scaffold == "Pt_II_chelating"


def test_friendly_name_satraplatin_pt_iv():
    """``detect_scaffold(smiles, name_hint='satraplatin')`` -> Pt_IV."""
    scaffold = detect_scaffold(
        "CC1=C(C(=O)O[Pt]([NH3])([NH3])(OC(=O)c2ccccc2)"
        "C(C)(C)C)C(C)(C)C=C1",
        name_hint="satraplatin",
    )
    assert scaffold == "Pt_IV"


def test_friendly_name_labile_metal_copper():
    """``detect_scaffold(smiles, name_hint='copper')`` -> labile_metal."""
    scaffold = detect_scaffold("[Cu+2]", name_hint="copper")
    assert scaffold == "labile_metal"


def test_friendly_name_unknown_defaults_unknown():
    """Unknown name hint + unparseable SMILES -> unknown scaffold.

    We use a SMILES that RDKit cannot parse (a bare metal with no
    bonding) so the RDKit-based heuristic path is bypassed; only the
    friendly-name fallback runs.  Without a recognised friendly hint
    the function must return "unknown" (NOT silently guess a scaffold).
    """
    # An empty / unparseable SMILES -> RDKit returns None for the
    # MolFromSmiles parse -> the helper falls back to the name hint.
    # With an unknown hint, the result must be "unknown".
    scaffold = detect_scaffold("not_a_valid_smiles_!!", name_hint="nonexistent_hint")
    assert scaffold == "unknown", (
        f"unknown SMILES + unknown hint must return 'unknown'; got {scaffold}"
    )

    # And: empty SMILES + None hint -> unknown.
    scaffold_empty = detect_scaffold("", name_hint=None)
    assert scaffold_empty == "unknown"

    # And: None SMILES + None hint -> unknown.
    scaffold_none = detect_scaffold(None, name_hint=None)
    assert scaffold_none == "unknown"


# ---------------------------------------------------------------------------
# Per-rule verdict helpers — incompatible_rules + marginal_rules
# ---------------------------------------------------------------------------
def test_incompatible_rules_per_scaffold():
    """Pin the ``incompatible_rules`` helper per scaffold."""
    # strict_Pt_II: ThiolEne + AmideCoupling are INCOMPATIBLE.
    inc = incompatible_rules("strict_Pt_II")
    assert "ThiolEne" in inc
    assert "AmideCoupling" in inc
    assert "CuAAC" not in inc
    assert "SPAAC" not in inc
    # Pt_IV: nothing is incompatible.
    assert incompatible_rules("Pt_IV") == []
    # labile_metal: nothing is incompatible.
    assert incompatible_rules("labile_metal") == []
    # Pt_II_chelating: nothing is incompatible (carboxylate chelate
    # makes all 5 click rules safe).
    assert incompatible_rules("Pt_II_chelating") == []


def test_marginal_rules_per_scaffold():
    """Pin the ``marginal_rules`` helper per scaffold."""
    # strict_Pt_II: Suzuki is the only MARGINAL rule.
    marg = marginal_rules("strict_Pt_II")
    assert marg == ["Suzuki"], (
        f"strict_Pt_II should have exactly Suzuki as marginal; got {marg}"
    )
    # Pt_II_chelating: Suzuki is MARGINAL.
    assert "Suzuki" in marginal_rules("Pt_II_chelating")
    # Pt_IV / labile_metal / unknown: no marginal rules.
    assert marginal_rules("Pt_IV") == []
    assert marginal_rules("labile_metal") == []
    assert marginal_rules("unknown") == []


# ---------------------------------------------------------------------------
# F2(a) invariant — strict_Pt_II allows metal-coordination rules
# ---------------------------------------------------------------------------
def test_strict_pt_ii_allows_metal_coordination_invariant():
    """The F2(a) invariant: strict_Pt_II must be COMPATIBLE on both
    ``metal_ligand_exchange`` AND ``aqua_exchange``.

    This is what unlocks diversity in the Lambda search while keeping
    ``metal_compliance == 1.0``: Pt_II square-planar can fire Pt-Cl
    substitution by NH3 (cisplatin activation step) without violating
    the metal-scaffold prior.
    """
    assert strict_pt_ii_allows_metal_coordination() is True
    # Pin the per-cell verdicts directly.
    row = COMPAT_MATRIX["strict_Pt_II"]
    assert row["metal_ligand_exchange"] == "compatible"
    assert row["aqua_exchange"] == "compatible"


# ---------------------------------------------------------------------------
# Helper pretty-printer does not raise
# ---------------------------------------------------------------------------
def test_render_compat_table_smoke():
    """The pretty-printer must produce a non-empty markdown table."""
    table = render_compat_table()
    assert isinstance(table, str)
    assert "| scaffold" in table  # markdown header
    # Every scaffold row must appear in the rendered table.
    for s in SCAFFOLD_NAMES:
        assert s in table
    # Every click-rule column header must appear.
    for rule in CLICK_RULE_NAMES:
        assert rule in table


# ---------------------------------------------------------------------------
# Friendly-name hint helper coverage
# ---------------------------------------------------------------------------
def test_name_hint_to_scaffold_coverage():
    """Pin every named hint used in production."""
    assert _name_hint_to_scaffold("cisplatin") == "strict_Pt_II"
    assert _name_hint_to_scaffold("nedaplatin") == "strict_Pt_II"
    assert _name_hint_to_scaffold("ru_arene") == "strict_Pt_II"
    assert _name_hint_to_scaffold("ir_cp_star") == "strict_Pt_II"
    assert _name_hint_to_scaffold("carboplatin") == "Pt_II_chelating"
    assert _name_hint_to_scaffold("oxaliplatin") == "Pt_II_chelating"
    assert _name_hint_to_scaffold("satraplatin") == "Pt_IV"
    assert _name_hint_to_scaffold("tetraplatin") == "Pt_IV"
    assert _name_hint_to_scaffold("iproplatin") == "Pt_IV"
    assert _name_hint_to_scaffold("copper") == "labile_metal"
    assert _name_hint_to_scaffold("zinc") == "labile_metal"
    assert _name_hint_to_scaffold("iron") == "labile_metal"
    assert _name_hint_to_scaffold("manganese") == "labile_metal"
    # Unknown name -> unknown.
    assert _name_hint_to_scaffold("nonexistent") == "unknown"
    # None / empty -> unknown.
    assert _name_hint_to_scaffold(None) == "unknown"
    assert _name_hint_to_scaffold("") == "unknown"
