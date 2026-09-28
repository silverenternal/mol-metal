"""Tests for ``metal_pdbqt_overrides`` (Wf-121 Phase 1).

================================================================
Why these tests exist
================================================================
TODO-110 / Wf-121 P1.1 — meeko 0.8.0 ships no SMARTS rules for the 8
d-block Pt-group metals (Pt/Ru/Ir/Os/Rh/Pd/Au/Re). Without our overrides
``PDBQTWriterLegacy.write_string`` rejects any metallodrug molecule
because ``setup.atom_type is None``.

These tests pin the SMARTS coverage:

1. Each of the 8 Pt-group elements has a SMARTS rule in
   :data:`METAL_ATOM_TYPE_RULES`.
2. Each rule maps to a non-empty atype (``"M"`` or ``"M1"``).
3. Each rule matches the canonical probe SMILES exactly once.
4. ``get_metal_atom_type_rules`` returns well-formed rows that meeko
   can consume (``smarts`` + ``atype`` + ``IDX`` keys).
5. ``get_metal_atom_type_rules`` rejects unknown elements.
6. ``verify_metal_smarts`` returns ok=True for every supported metal.
7. ``add_metal_atom_types`` mutates a real ``MoleculePreparation`` such
   that ``atom_params["add_atom_types"]`` (or a numbered slot) holds
   the 8 rules.

Honest framing
--------------
* Tests are CPU-only and depend on rdkit + meeko being installed in the
  venv (``mol-metal-venv``).
* Test #7 uses ``MoleculePreparation()`` directly so it exercises the
  real meeko API surface rather than a mock.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from molmetal.molmetal_lam.sbdd_env.metal_pdbqt_overrides import (
    COORDINATION_ATYPE,
    GENERIC_ATYPE,
    METAL_ATOM_TYPE_RULES,
    PROBE_SMILES,
    PT_GROUP_ELEMENTS,
    add_metal_atom_types,
    get_metal_atom_type_rules,
    verify_metal_smarts,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_ELEMENTS = ("Pt", "Ru", "Ir", "Os", "Rh", "Pd", "Au", "Re")


# ---------------------------------------------------------------------------
# Static-rule tests
# ---------------------------------------------------------------------------
class TestMetalAtomTypeRulesStatic:
    """Pin the rule table structure."""

    def test_all_eight_elements_have_rules(self):
        assert set(METAL_ATOM_TYPE_RULES.keys()) == set(EXPECTED_ELEMENTS)

    def test_pt_group_elements_constant_matches(self):
        assert set(PT_GROUP_ELEMENTS) == set(EXPECTED_ELEMENTS)

    def test_each_rule_has_required_keys(self):
        for elem, rule in METAL_ATOM_TYPE_RULES.items():
            assert "smarts" in rule, f"{elem} missing 'smarts'"
            assert "atype" in rule, f"{elem} missing 'atype'"
            assert "IDX" in rule, f"{elem} missing 'IDX'"
            assert rule["atype"], f"{elem} has empty atype"
            assert rule["smarts"], f"{elem} has empty smarts"

    def test_atype_is_generic_M_or_M1(self):
        for elem, rule in METAL_ATOM_TYPE_RULES.items():
            assert rule["atype"] in (GENERIC_ATYPE, COORDINATION_ATYPE), (
                f"{elem} atype={rule['atype']!r} not in "
                f"({GENERIC_ATYPE!r}, {COORDINATION_ATYPE!r})"
            )

    def test_smarts_uses_atomic_number_form(self):
        """Each SMARTS must be an atomic-number query like ``[Pt]``.

        Atomic-number form is the same convention meeko uses in
        ``ad4_types.json`` (``[#12]`` for Mg, ``[#26]`` for Fe, …).
        """
        for elem, rule in METAL_ATOM_TYPE_RULES.items():
            assert rule["smarts"] == f"[{elem}]", (
                f"{elem} smarts={rule['smarts']!r} (expected '[{elem}]')"
            )


# ---------------------------------------------------------------------------
# Probe-SMILES verification
# ---------------------------------------------------------------------------
class TestMetalSmartsAgainstProbes:
    """Each rule must match its canonical probe SMILES exactly."""

    @pytest.mark.parametrize("elem", EXPECTED_ELEMENTS)
    def test_rule_matches_probe_smiles(self, elem):
        from rdkit import Chem  # type: ignore

        probe = PROBE_SMILES[elem]
        mol = Chem.MolFromSmiles(probe)
        assert mol is not None, f"failed to parse probe SMILES for {elem}: {probe!r}"

        n_metal_atoms = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == elem)
        assert n_metal_atoms == 1, (
            f"probe {probe!r} should contain exactly 1 {elem} atom "
            f"(found {n_metal_atoms})"
        )

        pattern = Chem.MolFromSmarts(METAL_ATOM_TYPE_RULES[elem]["smarts"])
        assert pattern is not None
        n_matches = len(mol.GetSubstructMatches(pattern))
        assert n_matches == 1, (
            f"{elem} rule did not match probe {probe!r} exactly once "
            f"(got {n_matches} matches)"
        )

    def test_verify_metal_smarts_returns_ok_for_every_supported_metal(self):
        results = verify_metal_smarts()
        assert set(results.keys()) == set(EXPECTED_ELEMENTS)
        for elem, row in results.items():
            assert row["ok"], (
                f"{elem} verify_metal_smarts returned ok=False: {row}"
            )
            assert row["matched"] == row["expected"]
            assert row["expected"] >= 1


# ---------------------------------------------------------------------------
# get_metal_atom_type_rules — public helper
# ---------------------------------------------------------------------------
class TestGetMetalAtomTypeRules:
    def test_default_returns_eight_rules_in_order(self):
        rules = get_metal_atom_type_rules()
        assert len(rules) == 8
        assert [r["smarts"] for r in rules] == [
            METAL_ATOM_TYPE_RULES[e]["smarts"] for e in EXPECTED_ELEMENTS
        ]

    def test_subset_returns_only_requested(self):
        rules = get_metal_atom_type_rules(["Pt", "Au"])
        assert len(rules) == 2
        smarts = {r["smarts"] for r in rules}
        assert smarts == {"[Pt]", "[Au]"}

    def test_unknown_element_raises(self):
        with pytest.raises(ValueError, match="unsupported metals"):
            get_metal_atom_type_rules(["Pt", "Unobtainium"])

    def test_empty_subset_returns_empty(self):
        assert get_metal_atom_type_rules([]) == []

    def test_returned_rows_are_copies(self):
        """Mutating the returned list must not poison the module dict."""
        rules = get_metal_atom_type_rules(["Pt"])
        rules[0]["atype"] = "TAMPERED"
        assert METAL_ATOM_TYPE_RULES["Pt"]["atype"] == GENERIC_ATYPE


# ---------------------------------------------------------------------------
# add_metal_atom_types — meeko integration
# ---------------------------------------------------------------------------
class TestAddMetalAtomTypesIntegration:
    """Exercises the real meeko ``MoleculePreparation`` surface."""

    def test_add_metal_atom_types_injects_rules_into_preparator(self):
        from meeko import MoleculePreparation  # type: ignore

        preparator = MoleculePreparation()
        # Sanity — preparator must expose the mutable atom_params dict
        # that meeko's ``get_atom_params`` populates.
        assert hasattr(preparator, "atom_params")
        assert isinstance(preparator.atom_params, dict)

        # Wf-143 P3: default ``chemotype_vdw=True`` so atype is the
        # element symbol; explicit ``chemotype_vdw=False`` preserves the
        # Wf-121 P1 generic ``M`` behaviour (covered separately).
        add_metal_atom_types(preparator, chemotype_vdw=False)

        # Either the canonical ``add_atom_types`` key or a numbered
        # fallback slot must hold our 8 rules.
        injected_key = None
        for candidate in ("add_atom_types", "add_atom_types_0"):
            if candidate in preparator.atom_params:
                injected_key = candidate
                break
        assert injected_key is not None, (
            "add_metal_atom_types did not inject rules into atom_params; "
            f"keys present: {list(preparator.atom_params)}"
        )
        injected = preparator.atom_params[injected_key]
        assert len(injected) == 8

        smarts = sorted(r["smarts"] for r in injected)
        assert smarts == sorted(f"[{e}]" for e in EXPECTED_ELEMENTS)

        # Every injected rule must carry a usable atype so meeko does
        # not re-emit ``None`` types downstream.
        for r in injected:
            assert r["atype"]
            assert r["atype"] in (GENERIC_ATYPE, COORDINATION_ATYPE)

    def test_add_metal_atom_types_uses_unique_key_when_already_populated(self):
        from meeko import MoleculePreparation  # type: ignore

        preparator = MoleculePreparation()
        # Pre-populate the canonical key with a stub rule so the helper
        # is forced to find a non-colliding slot.
        preparator.atom_params["add_atom_types"] = [
            {"smarts": "[#1]", "atype": "H", "IDX": [0]},
        ]

        add_metal_atom_types(preparator)

        # The original stub must be preserved (we do not overwrite user
        # rules) and our 8 rules must live in a different group.
        assert preparator.atom_params["add_atom_types"] == [
            {"smarts": "[#1]", "atype": "H", "IDX": [0]},
        ]
        assert "add_atom_types_0" in preparator.atom_params
        assert len(preparator.atom_params["add_atom_types_0"]) == 8
