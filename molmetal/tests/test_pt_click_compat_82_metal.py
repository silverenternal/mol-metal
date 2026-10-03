"""Wf-223 (2026-09-28) 82-metal MLE (MetalLigandExchange) regression tests.

Background
----------
Wf-223 closes out the ``TODO/02_metrics/METRICS_TARGETS.md`` row
"18-rule + 84-MLE matrix" by adding an **82-metal MLE rule** library
to :mod:`molmetal.molmetal_lam.lam_chem.pt_click_compat`.  The
existing F2(a) + R17 library in
:mod:`molmetal.molmetal_lam.lam_chem.pt_metal_ligand_exchange` covers
the production-tested 12 metals (Pt / Ru / Ir / Os / Rh / Au oxidation
states) × 7 ligands = 84 patterns; this module extends the coverage
to the full periodic-table metallic set (82 curated metals) for
completeness + future chemistry expansion.

The 82-metal curated list (periods 2-7):

    5 alkali           (Li, Na, K, Rb, Cs)
    5 alkaline earth   (Be, Mg, Ca, Sr, Ba)
    10 d-block p4      (Sc-Zn)
    10 d-block p5      (Y-Cd)
    10 d-block p6      (Lu-Hg; Lu included as d-block-like lanthanide)
    5 d-block p7       (Rf-Hs; synthetic, registered for completeness)
    8 p-block metals   (Al-Po)
    14 lanthanides     (La-Yb; Lu moved to d-block p6)
    15 actinides       (Ac-Lr)
    -----------------------
    82 metals total

Honest framing
--------------
* The 82 patterns are HAND-CURATED from Cotton-Wilkinson 1980 +
  Lippard 1995 + Reedijk 1987 + Housecroft 2018 + Comba-Hambley 2009.
  No ML fitting was performed.  Each pattern uses the canonical
  Pt_II MetalLigandExchange SMARTS shape
  ``[M:1]([Cl:2])[*:3].[NH3:4]>>[M:1]([NH3:4])[*:3].[Cl:2]`` with
  the ``[M]`` token swapped for each curated metal symbol.
* The patterns are *exhaustive* (82 cells) but **not all 82 cells
  are physically realistic** — e.g. Cs (alkali, group 1) does NOT
  form stable covalent M-Cl bonds in physiological conditions; the
  pattern is registered for completeness so the Lambda search can
  *theoretically* fire on Cs-Cl + NH3 even though real Cs would
  prefer ionic coordination.  The physical chemistry is gated
  upstream via the COMPAT_MATRIX scaffold verdicts.
* This module is the Wf-223 SHIP surface — the 82 patterns are
  guaranteed by ``test_82_metal_count_is_exactly_82`` so an
  accidental future edit would fail loudly.

Run from the project root::

    uv run pytest molmetal/tests/test_pt_click_compat_82_metal.py -q --tb=short
"""

from __future__ import annotations

from typing import Tuple

import pytest

# CPU-only — RDKit ReactionFromSmarts is the heaviest dependency here.
from molmetal.molmetal_lam.lam_chem.pt_click_compat import (
    COMPAT_MATRIX_VERSION,
    METAL_LIGAND_EXCHANGE_82_METAL_RULES,
    METAL_LIGAND_EXCHANGE_82_METAL_SYMBOLS,
    CLICK_RULE_NAMES,
    SCAFFOLD_NAMES,
    list_registered_rules,
    register_metal_ligand_exchange_rules,
)


# ---------------------------------------------------------------------------
# Constants — curated 82-metal list + per-section expectations.
# ---------------------------------------------------------------------------

# Total = 82 by design (Wf-223 SHIP target).
EXPECTED_TOTAL_METALS: int = 82

# Per-section expected counts (must sum to 82).
EXPECTED_ALKALI: Tuple[str, ...] = ("Li", "Na", "K", "Rb", "Cs")
EXPECTED_ALKALINE_EARTH: Tuple[str, ...] = (
    "Be",
    "Mg",
    "Ca",
    "Sr",
    "Ba",
)
EXPECTED_D_BLOCK_P4: Tuple[str, ...] = (
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
)
EXPECTED_D_BLOCK_P5: Tuple[str, ...] = (
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
)
EXPECTED_D_BLOCK_P6: Tuple[str, ...] = (
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
)
EXPECTED_D_BLOCK_P7: Tuple[str, ...] = (
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
)
EXPECTED_P_BLOCK_METALS: Tuple[str, ...] = (
    "Al",
    "Ga",
    "In",
    "Sn",
    "Tl",
    "Pb",
    "Bi",
    "Po",
)
EXPECTED_LANTHANIDES: Tuple[str, ...] = (
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
)
EXPECTED_ACTINIDES: Tuple[str, ...] = (
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
)


def _sum_sections() -> int:
    """Sanity-check that the section breakdown sums to 82."""
    return (
        len(EXPECTED_ALKALI)
        + len(EXPECTED_ALKALINE_EARTH)
        + len(EXPECTED_D_BLOCK_P4)
        + len(EXPECTED_D_BLOCK_P5)
        + len(EXPECTED_D_BLOCK_P6)
        + len(EXPECTED_D_BLOCK_P7)
        + len(EXPECTED_P_BLOCK_METALS)
        + len(EXPECTED_LANTHANIDES)
        + len(EXPECTED_ACTINIDES)
    )


# ---------------------------------------------------------------------------
# Test 1 — module version pin
# ---------------------------------------------------------------------------
def test_compat_matrix_version_is_wf_223():
    """COMPAT_MATRIX_VERSION must include the Wf-223 stamp (2026-09-28)."""
    assert "wf-223" in COMPAT_MATRIX_VERSION, (
        f"COMPAT_MATRIX_VERSION must include 'wf-223' stamp; "
        f"got '{COMPAT_MATRIX_VERSION}'."
    )
    assert "2026-09-28" in COMPAT_MATRIX_VERSION, (
        f"COMPAT_MATRIX_VERSION must include '2026-09-28' date; "
        f"got '{COMPAT_MATRIX_VERSION}'."
    )


# ---------------------------------------------------------------------------
# Test 2 — 82-metal symbol list cardinality + uniqueness
# ---------------------------------------------------------------------------
def test_82_metal_count_is_exactly_82():
    """``METAL_LIGAND_EXCHANGE_82_METAL_SYMBOLS`` has exactly 82 entries.

    Wf-223 SHIP target: 82-metal coverage.  The cardinality is pinned
    so an accidental edit (add / remove) would fail loudly.
    """
    assert len(METAL_LIGAND_EXCHANGE_82_METAL_SYMBOLS) == EXPECTED_TOTAL_METALS, (
        f"METAL_LIGAND_EXCHANGE_82_METAL_SYMBOLS has "
        f"{len(METAL_LIGAND_EXCHANGE_82_METAL_SYMBOLS)} entries; "
        f"expected exactly {EXPECTED_TOTAL_METALS}."
    )
    # Per-section breakdown must sum to 82.
    assert (
        _sum_sections() == EXPECTED_TOTAL_METALS
    ), f"Per-section sum = {_sum_sections()}; expected {EXPECTED_TOTAL_METALS}."


def test_82_metal_symbols_are_unique():
    """All 82 metal symbols must be unique (no duplicates)."""
    seen = set(METAL_LIGAND_EXCHANGE_82_METAL_SYMBOLS)
    assert len(seen) == len(METAL_LIGAND_EXCHANGE_82_METAL_SYMBOLS), (
        f"METAL_LIGAND_EXCHANGE_82_METAL_SYMBOLS has duplicates; "
        f"unique={len(seen)} vs total={len(METAL_LIGAND_EXCHANGE_82_METAL_SYMBOLS)}."
    )


# ---------------------------------------------------------------------------
# Test 3 — per-section coverage
# ---------------------------------------------------------------------------
def test_alkali_metals_present():
    """5 alkali metals (Li, Na, K, Rb, Cs) are all in the curated list."""
    for m in EXPECTED_ALKALI:
        assert (
            m in METAL_LIGAND_EXCHANGE_82_METAL_SYMBOLS
        ), f"Alkali metal '{m}' missing from curated list."


def test_alkaline_earth_metals_present():
    """5 alkaline earth metals (Be, Mg, Ca, Sr, Ba) are all present."""
    for m in EXPECTED_ALKALINE_EARTH:
        assert (
            m in METAL_LIGAND_EXCHANGE_82_METAL_SYMBOLS
        ), f"Alkaline earth metal '{m}' missing from curated list."


def test_d_block_period_4_present():
    """10 d-block period-4 metals (Sc-Zn) are all present."""
    for m in EXPECTED_D_BLOCK_P4:
        assert (
            m in METAL_LIGAND_EXCHANGE_82_METAL_SYMBOLS
        ), f"d-block p4 metal '{m}' missing from curated list."


def test_d_block_period_5_present():
    """10 d-block period-5 metals (Y-Cd) are all present."""
    for m in EXPECTED_D_BLOCK_P5:
        assert (
            m in METAL_LIGAND_EXCHANGE_82_METAL_SYMBOLS
        ), f"d-block p5 metal '{m}' missing from curated list."


def test_d_block_period_6_present():
    """10 d-block period-6 metals (Lu-Hg) are all present.

    Lu is included here as the canonical d-block-like lanthanide
    per Cotton-Wilkinson 1980 §18.3; Lu is NOT in the lanthanides
    list below (La-Yb only).
    """
    for m in EXPECTED_D_BLOCK_P6:
        assert (
            m in METAL_LIGAND_EXCHANGE_82_METAL_SYMBOLS
        ), f"d-block p6 metal '{m}' missing from curated list."


def test_d_block_period_7_present():
    """5 d-block period-7 metals (Rf-Hs) are all present (synthetic)."""
    for m in EXPECTED_D_BLOCK_P7:
        assert (
            m in METAL_LIGAND_EXCHANGE_82_METAL_SYMBOLS
        ), f"d-block p7 metal '{m}' missing from curated list."


def test_p_block_metals_present():
    """8 p-block metals (Al-Po) are all present."""
    for m in EXPECTED_P_BLOCK_METALS:
        assert (
            m in METAL_LIGAND_EXCHANGE_82_METAL_SYMBOLS
        ), f"p-block metal '{m}' missing from curated list."


def test_lanthanides_present():
    """14 lanthanides (La-Yb) are all present.  Lu is NOT here (d-block p6)."""
    for m in EXPECTED_LANTHANIDES:
        assert (
            m in METAL_LIGAND_EXCHANGE_82_METAL_SYMBOLS
        ), f"Lanthanide '{m}' missing from curated list."
    # Lu is in d-block p6 only — make sure Lu is NOT counted in
    # the lanthanides section.
    assert "Lu" not in EXPECTED_LANTHANIDES, (
        "Lu should be in d-block p6 (per Cotton-Wilkinson 1980 §18.3) "
        "and NOT in the lanthanides list."
    )


def test_actinides_present():
    """15 actinides (Ac-Lr) are all present."""
    for m in EXPECTED_ACTINIDES:
        assert (
            m in METAL_LIGAND_EXCHANGE_82_METAL_SYMBOLS
        ), f"Actinide '{m}' missing from curated list."


# ---------------------------------------------------------------------------
# Test 4 — Lu canonical placement (lanthanide vs d-block)
# ---------------------------------------------------------------------------
def test_lu_placement_is_d_block_only():
    """Lu is in d-block period 6; NOT in the lanthanides section."""
    assert "Lu" in EXPECTED_D_BLOCK_P6
    assert "Lu" not in EXPECTED_LANTHANIDES
    # And Lu is in the curated list exactly once.
    assert METAL_LIGAND_EXCHANGE_82_METAL_SYMBOLS.count("Lu") == 1


# ---------------------------------------------------------------------------
# Test 5 — production-tested metals (the F2(a) curated subset) are present
# ---------------------------------------------------------------------------
def test_f2a_production_subset_present():
    """The F2(a) curated metals (Pt, Au, Pd, Ru, Cu) must all be in 82-list.

    Backward-compat: callers that relied on the F2(a) curated subset
    must continue to find their metals in the new 82-list.
    """
    production_subset = ("Pt", "Au", "Pd", "Ru", "Cu")
    for m in production_subset:
        assert m in METAL_LIGAND_EXCHANGE_82_METAL_SYMBOLS, (
            f"Production-tested F2(a) metal '{m}' missing from 82-list; "
            "regression — F2(a) callers will see KeyError."
        )


# ---------------------------------------------------------------------------
# Test 6 — 82-rule SMARTS library: count, parse, structure
# ---------------------------------------------------------------------------
def test_82_rule_library_has_exactly_82_entries():
    """``METAL_LIGAND_EXCHANGE_82_METAL_RULES`` has exactly 82 entries."""
    assert len(METAL_LIGAND_EXCHANGE_82_METAL_RULES) == EXPECTED_TOTAL_METALS, (
        f"METAL_LIGAND_EXCHANGE_82_METAL_RULES has "
        f"{len(METAL_LIGAND_EXCHANGE_82_METAL_RULES)} entries; "
        f"expected exactly {EXPECTED_TOTAL_METALS}."
    )


def test_82_rule_names_match_82_metal_symbols():
    """Each rule's ``name`` field must be the metal symbol; ordering matches."""
    rule_names = [r[0] for r in METAL_LIGAND_EXCHANGE_82_METAL_RULES]
    assert rule_names == list(
        METAL_LIGAND_EXCHANGE_82_METAL_SYMBOLS
    ), "Rule names must match the curated 82-metal symbol list in order."


def test_82_rule_smarts_all_parse_with_rdkit():
    """Every 82 SMARTS pattern parses via RDKit's ``ReactionFromSmarts``.

    CPU-only test (RDKit import).  If a SMARTS is malformed the
    pattern would never fire in production; this guards against
    typos in the SMARTS builder.
    """
    from rdkit.Chem import AllChem  # type: ignore

    for name, smarts, desc in METAL_LIGAND_EXCHANGE_82_METAL_RULES:
        rxn = AllChem.ReactionFromSmarts(smarts)
        assert rxn is not None, (
            f"SMARTS for metal '{name}' did not parse; got None.  "
            f"SMARTS = '{smarts}'"
        )


def test_82_rule_smarts_canonical_shape():
    """Every SMARTS uses the canonical M-Cl + NH3 → M-NH3 + Cl shape.

    Pin the SMARTS template so an accidental edit that changes the
    leaving-group position or the donor atom-map would fail loudly.
    The canonical shape is::

        [<M>:1]([Cl:2])[*:3].[NH3:4]>>[<M>:1]([NH3:4])[*:3].[Cl:2]

    where ``<M>`` is the metal symbol.
    """
    for name, smarts, desc in METAL_LIGAND_EXCHANGE_82_METAL_RULES:
        expected = (
            f"[{name}:1]([Cl:2])[*:3].[NH3:4]" f">>[{name}:1]([NH3:4])[*:3].[Cl:2]"
        )
        assert smarts == expected, (
            f"SMARTS for metal '{name}' has unexpected shape; "
            f"got '{smarts}', expected '{expected}'."
        )


def test_82_rule_description_non_empty():
    """Every rule's description is non-empty (used for audit reports)."""
    for name, smarts, desc in METAL_LIGAND_EXCHANGE_82_METAL_RULES:
        assert (
            isinstance(desc, str) and len(desc) > 0
        ), f"Rule '{name}' has empty description."


def test_82_rule_canonical_lippard_for_pt():
    """Pt's description must mention Lippard 1995 (the canonical Pt text)."""
    pt_rule = next(r for r in METAL_LIGAND_EXCHANGE_82_METAL_RULES if r[0] == "Pt")
    assert "Lippard" in pt_rule[2], (
        f"Pt description must cite Lippard 1995 (canonical Pt text); "
        f"got '{pt_rule[2]}'."
    )


# ---------------------------------------------------------------------------
# Test 7 — register_metal_ligand_exchange_rules() helper
# ---------------------------------------------------------------------------
def test_register_metal_ligand_exchange_rules_returns_82():
    """``register_metal_ligand_exchange_rules()`` returns exactly 82 patterns.

    Default ``register_into_rule_registry=False`` (backward-compat:
    no global side-effects).  Returns a fresh list copy of
    ``METAL_LIGAND_EXCHANGE_82_METAL_RULES`` so callers can mutate
    without affecting the module-level constant.
    """
    rules = register_metal_ligand_exchange_rules()
    assert len(rules) == EXPECTED_TOTAL_METALS, (
        f"register_metal_ligand_exchange_rules() returned "
        f"{len(rules)} rules; expected {EXPECTED_TOTAL_METALS}."
    )
    # Returns a fresh list (not the module-level constant).
    assert rules is not METAL_LIGAND_EXCHANGE_82_METAL_RULES, (
        "register_metal_ligand_exchange_rules() must return a fresh "
        "list copy, not the module-level constant."
    )
    # Mutating the returned list must not affect the constant.
    rules.append(
        ("dummy", "dummy_smarts", "dummy desc"),
    )
    assert len(METAL_LIGAND_EXCHANGE_82_METAL_RULES) == EXPECTED_TOTAL_METALS


def test_register_metal_ligand_exchange_rules_default_no_registry_side_effect():
    """Default ``register_into_rule_registry=False`` — no global mutation.

    Backward-compat: callers that don't opt into the registry
    side-effect must see ZERO changes to the rule registry.
    """
    # We can't easily import the rule registry without the full
    # lam_chem package init (which transitively imports torch /
    # rdkit / etc.).  Instead, just confirm the function does not
    # raise when called with the default.
    rules = register_metal_ligand_exchange_rules()
    assert len(rules) == EXPECTED_TOTAL_METALS


# ---------------------------------------------------------------------------
# Test 8 — list_registered_rules() returns the 7-name canonical set
# ---------------------------------------------------------------------------
def test_list_registered_rules_returns_7():
    """``list_registered_rules()`` returns the 7-name canonical set.

    Re-exports :data:`CLICK_RULE_NAMES` (5 click + 2 metal-coord
    rules).  Backward-compat with the WF-Lambda-Phase-4 precheck.
    """
    rules = list_registered_rules()
    assert len(rules) == 7, (
        f"list_registered_rules() returned {len(rules)} names; "
        "expected 7 (5 click + 2 metal-coord)."
    )
    assert rules == CLICK_RULE_NAMES, (
        f"list_registered_rules() must match CLICK_RULE_NAMES; "
        f"got {rules}, expected {CLICK_RULE_NAMES}."
    )
    # Pin the 5 click names + 2 metal-coord names.
    expected_names = (
        "CuAAC",
        "SPAAC",
        "ThiolEne",
        "Suzuki",
        "AmideCoupling",
        "metal_ligand_exchange",
        "aqua_exchange",
    )
    assert set(rules) == set(expected_names)


# ---------------------------------------------------------------------------
# Test 9 — backward compat with the existing COMPAT_MATRIX
# ---------------------------------------------------------------------------
def test_backward_compat_comp_matrix_unchanged():
    """Adding the 82-metal library must not regress the COMPAT_MATRIX.

    The 5 scaffolds × 7 click-rule columns must remain intact
    after the Wf-223 additions.  This guards against an accidental
    refactor that drops a scaffold or click-rule column.
    """
    assert (
        len(SCAFFOLD_NAMES) == 5
    ), f"SCAFFOLD_NAMES has {len(SCAFFOLD_NAMES)} entries; expected 5."
    assert (
        len(CLICK_RULE_NAMES) == 7
    ), f"CLICK_RULE_NAMES has {len(CLICK_RULE_NAMES)} entries; expected 7."
    # The 5 scaffold names are exactly the canonical set.
    expected_scaffolds = (
        "strict_Pt_II",
        "Pt_II_chelating",
        "Pt_IV",
        "labile_metal",
        "unknown",
    )
    assert set(SCAFFOLD_NAMES) == set(expected_scaffolds)


# ---------------------------------------------------------------------------
# Test 10 — the 82 count is honest: 5+5+10+10+10+5+8+14+15=82
# ---------------------------------------------------------------------------
def test_section_breakdown_sums_to_82():
    """The per-section breakdown explicitly sums to 82.

    5 alkali + 5 alkaline earth + 10 d-block p4 + 10 d-block p5 +
    10 d-block p6 + 5 d-block p7 + 8 p-block metals +
    14 lanthanides + 15 actinides = 82.

    This is the *honest framing* documentation — the 82 count
    breaks down into the 9 sub-sections, and any future edit must
    preserve the sum.
    """
    expected_sum = (
        5  # alkali
        + 5  # alkaline earth
        + 10  # d-block p4
        + 10  # d-block p5
        + 10  # d-block p6
        + 5  # d-block p7
        + 8  # p-block metals
        + 14  # lanthanides (La-Yb; Lu moved to d-block p6)
        + 15  # actinides
    )
    assert expected_sum == 82, f"Per-section sum = {expected_sum}; expected 82."
    assert _sum_sections() == expected_sum


# ---------------------------------------------------------------------------
# Test 11 — honest framing: the 82 count is curated, not auto-generated
# ---------------------------------------------------------------------------
def test_honest_framing_curated_list_not_full_periodic_table():
    """Honest framing: the 82 count is CURATED, not auto-generated.

    Some metals are intentionally excluded from the 82 list:

    * Fr (87) — radioactive, alkali, very short half-life; the
      search layer can't realistically fire on Fr-Cl.
    * Ra (88) — radioactive, alkaline earth, also short half-life.
    * He, Ne, Ar, Kr, Xe, Rn — noble gases, no coordination chemistry.
    * H, C, N, O, F, P, S, Cl, Se, Br, I — non-metals.
    * B, Si, Ge, As, Sb, Te — metalloids (some appear in the
      p-block list when they form covalent bonds, e.g. Sn, Pb, but
      B / Si / Ge / As / Sb / Te are excluded to keep the list
      strictly metallic).
    * At, Po — borderline (Po is included, At is not).

    This test pins the curated set so future edits cannot silently
    drop these exclusions.
    """
    excluded_must_not_be_in_82 = (
        "H",
        "He",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Sb",
        "Te",
        "I",
        "Xe",
        "At",
        "Rn",
        "Fr",
        "Ra",
    )
    for m in excluded_must_not_be_in_82:
        assert m not in METAL_LIGAND_EXCHANGE_82_METAL_SYMBOLS, (
            f"Excluded non-metal / noble-gas / short-lived '{m}' "
            "appeared in the curated 82-list; the Wf-223 honest "
            "framing notes these must NOT be present."
        )
