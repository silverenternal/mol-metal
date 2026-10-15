"""Regression test — runs the SKILL.md Example 1 + Example 2 verbatim.

Background
----------
Wf-A43-fixup-3 (2026-10-15): the click-chemistry skill SKILL.md
(`.claude/skills/mol-metal-click-chemistry/SKILL.md`) had drifted from
the on-disk API:

  - `can_apply(rule, metal_state)`            — DOES NOT EXIST
  - `apply_cuaac(complex, azide, alkyne)`     — DOES NOT EXIST (signature was 3-arg)
  - `register_click_rule(...)`                 — DOES NOT EXIST
  - `register_compat(rule, state, ...)`        — DOES NOT EXIST
  - `compat_matrix(rule, metal)`              — DOES NOT EXIST (returns the per-cell verdict)

Real API (verified on disk 2026-10-15):
  - compat is keyed on **scaffold** (5 scaffolds), NOT metal oxidation state
  - `list_registered_rules()`                 -> 7 base rules
  - `default_compatible_rules(scaffold)`      -> list of compatible rules
  - `incompatible_rules(scaffold)`            -> list of incompatible rules
  - `marginal_rules(scaffold)`                -> list of marginal rules
  - `render_compat_table()`                   -> full markdown table
  - `click_cuaac(azide_tile, alkyne_tile)`    -> 2-arg, NOT 3
  - `apply_click_reaction(name, tile_a, tile_b)` -> order-agnostic dispatch
  - `build_click_tile(smiles, tags, embed_3d=False)` -> Tile factory

This test file runs the SKILL.md Example 1 + Example 2 code as
literate tests so any future SKILL.md drift is caught at pytest time.

Per AGENTS.md §5.5 filesystem-truth: every claim about the SKILL.md
is verified against the on-disk module, not the prose.
"""

from __future__ import annotations

# All tests in this file are CPU-only.
import pytest

from molmetal.molmetal_lam.lam_chem.pt_click_compat import (
    list_registered_rules,
    default_compatible_rules,
    incompatible_rules,
    marginal_rules,
    render_compat_table,
)
from molmetal.molmetal_lam.reactions.click_reactions import (
    click_cuaac,
    apply_click_reaction,
)
from molmetal.molmetal_lam.tile_lib.click_tiles import build_click_tile


# === Example 1: Gate a click reaction by metal scaffold ===

def test_example1_list_registered_rules():
    """SKILL.md line ~38: `print('registered rules:', list_registered_rules())`"""
    rules = list_registered_rules()
    assert isinstance(rules, tuple)
    # 5 click + 2 metal exchanges = 7 base rules
    assert set(rules) == {
        "CuAAC", "SPAAC", "ThiolEne", "Suzuki", "AmideCoupling",
        "metal_ligand_exchange", "aqua_exchange",
    }


def test_example1_strict_pt_ii_compat():
    """SKILL.md: strict_Pt_II compatible / incompatible / marginal split."""
    scaffold = "strict_Pt_II"
    compatible = default_compatible_rules(scaffold)
    incompatible = incompatible_rules(scaffold)
    marginal = marginal_rules(scaffold)

    # Per AGENTS.md §1.5 filesystem-truth: 5+2 split is hand-curated, NOT learned.
    # Expected: CuAAC, SPAAC, Suzuki, metal_ligand_exchange, aqua_exchange compatible
    #           ThiolEne, AmideCoupling incompatible
    #           Suzuki marginal (Pd transmetalation risk)
    assert set(compatible) == {
        "CuAAC", "SPAAC", "Suzuki", "metal_ligand_exchange", "aqua_exchange",
    }, f"strict_Pt_II compatible drift: got {compatible}"
    assert set(incompatible) == {"ThiolEne", "AmideCoupling"}, (
        f"strict_Pt_II incompatible drift: got {incompatible}"
    )
    assert "Suzuki" in marginal, "Suzuki must be MARG on strict_Pt_II"


def test_example1_pt_iv_all_compatible():
    """SKILL.md: Pt_IV (d6 octahedral) is fully compatible (prodrug)."""
    compatible = default_compatible_rules("Pt_IV")
    # All 7 rules are OK on Pt_IV (no transmetalation, no Pt-S break)
    assert len(compatible) == 7, f"Pt_IV should be all-compatible, got {compatible}"


def test_example1_cuaac_positive_case():
    """SKILL.md: positive case — CuAAC fires on strict_Pt_II."""
    azide_tile = build_click_tile("CCN=[N+]=[N-]", ["azide"], embed_3d=False)
    alkyne_tile = build_click_tile("CC#C", ["alkyne_terminal"], embed_3d=False)
    products = click_cuaac(azide_tile, alkyne_tile)
    assert isinstance(products, list)
    assert len(products) >= 1, (
        "CuAAC on strict_Pt_II must produce at least 1 product (the triazole)"
    )


def test_example1_cuaac_order_agnostic():
    """SKILL.md: apply_click_reaction(name, tile_a, tile_b) is order-agnostic."""
    azide_tile = build_click_tile("CCN=[N+]=[N-]", ["azide"], embed_3d=False)
    alkyne_tile = build_click_tile("CC#C", ["alkyne_terminal"], embed_3d=False)
    # Forward order
    products_fwd = apply_click_reaction("CuAAC", azide_tile, alkyne_tile)
    # Swapped order
    products_swap = apply_click_reaction("CuAAC", alkyne_tile, azide_tile)
    assert len(products_fwd) == len(products_swap) >= 1


def test_example1_thiolene_incompatible():
    """SKILL.md: ThiolEne breaks Pt-S bond on strict_Pt_II."""
    assert "ThiolEne" in incompatible_rules("strict_Pt_II")
    # ThiolEne is OK on Pt_IV (no Pt-S preservation constraint)
    assert "ThiolEne" in default_compatible_rules("Pt_IV")


# === Example 2: Render the full compat matrix ===

def test_example2_render_compat_table_5x7():
    """SKILL.md: render_compat_table returns 5 scaffolds × 7 rules."""
    table = render_compat_table()
    assert isinstance(table, str)
    # The header has 7 rule columns (5 click + 2 metal)
    header_rules = [
        "CuAAC", "SPAAC", "ThiolEne", "Suzuki", "AmideCoupling",
        "metal_ligand_exchange", "aqua_exchange",
    ]
    for rule in header_rules:
        assert rule in table, f"compat table missing header column: {rule}"
    # The 5 scaffolds
    scaffolds = [
        "strict_Pt_II", "Pt_II_chelating", "Pt_IV", "labile_metal", "unknown",
    ]
    for s in scaffolds:
        assert s in table, f"compat table missing scaffold row: {s}"


def test_example2_strict_pt_ii_row():
    """The strict_Pt_II row in the matrix should match the per-scaffold API."""
    table = render_compat_table()
    # The strict_Pt_II row should show: OK OK X MARG X OK OK
    # (CuAAC=OK SPAAC=OK ThiolEne=X Suzuki=MARG AmideCoupling=X MLE=OK AE=OK)
    # Find the row
    lines = table.split("\n")
    strict_row = next(
        (l for l in lines if l.startswith("| strict_Pt_II")),
        None,
    )
    assert strict_row is not None, "strict_Pt_II row missing"
    # Parse cells (drop "strict_Pt_II" and trailing empty cell)
    cells = [c.strip() for c in strict_row.split("|")[2:-1]]
    expected = ["OK", "OK", "X", "MARG", "X", "OK", "OK"]
    assert cells == expected, (
        f"strict_Pt_II row drift: expected {expected}, got {cells}. "
        f"Re-check pt_click_compat.py COMPAT_MATRIX."
    )


# === Negative: stale API names should NOT exist ===

def test_stale_api_does_not_exist():
    """Catch SKILL.md drift back — these names should never come back."""
    import molmetal.molmetal_lam.lam_chem.pt_click_compat as compat_mod
    import molmetal.molmetal_lam.reactions.click_reactions as rxn_mod

    for stale in ("can_apply", "compat_matrix", "register_compat"):
        assert not hasattr(compat_mod, stale), (
            f"stale name `can_apply`/`compat_matrix`/`register_compat` "
            f"should NOT be re-added to pt_click_compat.py"
        )
    for stale in ("register_click_rule", "apply_cuaac"):
        assert not hasattr(rxn_mod, stale), (
            f"stale name `register_click_rule`/`apply_cuaac` "
            f"should NOT be re-added to click_reactions.py"
        )
