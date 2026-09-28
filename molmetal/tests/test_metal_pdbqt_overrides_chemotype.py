"""Tests for Wf-143 P3 chemotype-tuned VDW in ``metal_pdbqt_overrides``.

================================================================
What this file covers
================================================================
Wf-143 P3 extends :mod:`molmetal.molmetal_lam.sbdd_env.metal_pdbqt_overrides`
with chemotype-specific atype strings (e.g. ``"Pt"``, ``"Ru"``) for
meeko / Vina, replacing the Wf-121 P1 generic ``"M"`` placeholder.

The 10 tests in this file pin the contract:

1. ``get_metal_vdw_table`` returns 8 metals.
2. ``get_metal_vdw_table`` r/ε match the Wf-143 Stage 2 aggregated
   ``dblock_vdw_table.json`` values.
3. ``get_metal_vdw_table`` rejects unknown elements.
4. ``get_metal_vdw_table`` raises ``FileNotFoundError`` when the
   table is missing.
5. ``get_metal_atom_type_rules(chemotype_vdw=True)`` returns atype
   strings equal to the element symbol (e.g. ``"Pt"``, ``"Ru"``).
6. ``get_metal_atom_type_rules(chemotype_vdw=False)`` preserves the
   Wf-121 P1 generic ``"M"`` atype (backward compat).
7. ``add_metal_atom_types`` with ``chemotype_vdw=True`` injects
   chemotype-specific atype strings into a real
   ``MoleculePreparation`` (meeko integration).
8. ``add_metal_atom_types`` with ``chemotype_vdw=False`` injects the
   generic ``"M"`` atype strings (backward compat).
9. ``_rewrite_metal_atype_for_vina`` in
   :mod:`molmetal.scripts.bench_cisplatin_real_vina` rewrites ``M``
   → chemotype atype (e.g. ``Pt``) on a representative cisplatin
   PDBQT — i.e. **not** ``Mg``.
10. ``_rewrite_metal_atype_for_vina`` with ``chemotype_vdw=False``
    falls back to ``Mg`` directly (backward compat).
11. Non-metal ligands pass through unchanged.
12. ``_probe_vina_accepts_chemotype_atype`` correctly detects the
    chemotype atype when Vina's table is mocked.
13. ``_probe_vina_accepts_chemotype_atype`` rewrites to ``Mg`` when
    Vina's table does not contain the chemotype atype.

Honest framing
--------------
* Tests #9–#13 call into the bench script's rewrite helper — they
  verify the **text-level** rewrite only; real Vina acceptance is
  downstream of the rewrite and is NOT tested here (the bench
  script's full sweep is the real verification path).
* Test #12 mocks the Vina atom-type table so the probe's decision
  path is exercised deterministically.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any, Dict, List
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from molmetal.molmetal_lam.sbdd_env.metal_pdbqt_overrides import (
    DBLOCK_VDW_TABLE_PATH,
    GENERIC_ATYPE,
    PT_GROUP_ELEMENTS,
    VINA_FALLBACK_ATYPE,
    add_metal_atom_types,
    get_metal_atom_type_rules,
    get_metal_vdw_table,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_ELEMENTS = ("Pt", "Ru", "Ir", "Os", "Rh", "Pd", "Au", "Re")

#: Documented r/ε values from the Wf-143 Stage 2 aggregated table.
#: Sourced from ``dblock_vdw_table.json`` — see
#: ``molmetal/tests/test_dblock_vdw_table.py`` for the per-metal
#: pytest contracts.
EXPECTED_VDW_VALUES: Dict[str, tuple] = {
    "Pt": (1.75, 0.080),
    "Ru": (2.05, 0.056),
    "Ir": (1.866, 0.073),
    "Os": (1.700, 0.037),
    "Rh": (1.73, 0.053),
    "Pd": (1.63, 0.048),
    "Au": (1.66, 0.039),
    "Re": (1.55, 0.066),
}


# ---------------------------------------------------------------------------
# get_metal_vdw_table — 4 tests
# ---------------------------------------------------------------------------
class TestGetMetalVdwTable:
    """Pin the per-metal (r, ε) loader from Wf-143 Stage 2."""

    def test_returns_eight_metals_by_default(self):
        """Default ``elements=None`` must return all 8 Pt-group metals."""
        table = get_metal_vdw_table()
        assert set(table.keys()) == set(EXPECTED_ELEMENTS)
        assert len(table) == 8

    def test_r_and_epsilon_match_aggregated_table(self):
        """r/ε must match the Wf-143 Stage 2 aggregated values."""
        table = get_metal_vdw_table()
        for elem, (r_expected, eps_expected) in EXPECTED_VDW_VALUES.items():
            r, eps = table[elem]
            assert r == pytest.approx(r_expected), (
                f"{elem} r={r} expected {r_expected}"
            )
            assert eps == pytest.approx(eps_expected), (
                f"{elem} ε={eps} expected {eps_expected}"
            )

    def test_unknown_element_raises(self):
        with pytest.raises(ValueError, match="unsupported metals"):
            get_metal_vdw_table(["Pt", "Unobtainium"])

    def test_missing_table_raises_file_not_found(self, tmp_path):
        bad_path = tmp_path / "missing.json"
        with pytest.raises(FileNotFoundError):
            get_metal_vdw_table(table_path=bad_path)


# ---------------------------------------------------------------------------
# get_metal_atom_type_rules — chemotype vs generic — 4 tests
# ---------------------------------------------------------------------------
class TestChemotypeAtypeRules:
    """Pin the chemotype_vdw switch in :func:`get_metal_atom_type_rules`."""

    def test_chemotype_atype_equals_element_symbol(self):
        """``chemotype_vdw=True`` → atype field is the element symbol."""
        rules = get_metal_atom_type_rules(["Pt", "Ru", "Au"], chemotype_vdw=True)
        assert len(rules) == 3
        atypes = {r["smarts"]: r["atype"] for r in rules}
        assert atypes == {"[Pt]": "Pt", "[Ru]": "Ru", "[Au]": "Au"}

    def test_generic_atype_preserved_when_chemotype_disabled(self):
        """``chemotype_vdw=False`` → atype is the generic ``"M"`` (Wf-121 P1)."""
        rules = get_metal_atom_type_rules(chemotype_vdw=False)
        assert len(rules) == 8
        for rule in rules:
            assert rule["atype"] == GENERIC_ATYPE
            assert rule["atype"] == "M"

    def test_default_is_chemotype_atype(self):
        """The default (``chemotype_vdw`` argument omitted) must be
        chemotype-specific — Wf-143 P3 contract."""
        rules = get_metal_atom_type_rules()
        assert len(rules) == 8
        smarts_to_atype = {r["smarts"]: r["atype"] for r in rules}
        for elem in EXPECTED_ELEMENTS:
            assert smarts_to_atype[f"[{elem}]"] == elem, (
                f"default chemotype_vdw must set atype='{elem}', got "
                f"'{smarts_to_atype[f'[{elem}]']}'"
            )

    def test_each_rule_has_required_keys_in_both_modes(self):
        for chemotype_vdw in (True, False):
            rules = get_metal_atom_type_rules(chemotype_vdw=chemotype_vdw)
            for rule in rules:
                assert "smarts" in rule
                assert "atype" in rule
                assert "IDX" in rule
                assert rule["IDX"] == [0]
                assert rule["atype"]  # non-empty


# ---------------------------------------------------------------------------
# add_metal_atom_types — meeko integration in both modes — 2 tests
# ---------------------------------------------------------------------------
class TestAddMetalAtomTypesChemotypeIntegration:
    """Real ``MoleculePreparation`` integration with chemotype_vdw."""

    def test_chemotype_true_injects_element_atype_into_preparator(self):
        from meeko import MoleculePreparation  # type: ignore

        preparator = MoleculePreparation()
        add_metal_atom_types(preparator, chemotype_vdw=True)

        injected_key = None
        for candidate in ("add_atom_types", "add_atom_types_0"):
            if candidate in preparator.atom_params:
                injected_key = candidate
                break
        assert injected_key is not None, (
            "add_metal_atom_types did not inject rules; keys: "
            f"{list(preparator.atom_params)}"
        )
        injected = preparator.atom_params[injected_key]
        assert len(injected) == 8
        # Every rule's atype must equal the element symbol (chemotype).
        for rule in injected:
            elem = rule["smarts"].strip("[]")
            assert rule["atype"] == elem, (
                f"{rule['smarts']} atype={rule['atype']!r} (expected "
                f"chemotype symbol {elem!r})"
            )

    def test_chemotype_false_injects_generic_M_into_preparator(self):
        """Backward compat: ``chemotype_vdw=False`` preserves Wf-121 P1
        generic ``M`` atype in the injected rules."""
        from meeko import MoleculePreparation  # type: ignore

        preparator = MoleculePreparation()
        add_metal_atom_types(preparator, chemotype_vdw=False)

        injected_key = None
        for candidate in ("add_atom_types", "add_atom_types_0"):
            if candidate in preparator.atom_params:
                injected_key = candidate
                break
        assert injected_key is not None
        injected = preparator.atom_params[injected_key]
        assert len(injected) == 8
        for rule in injected:
            assert rule["atype"] == GENERIC_ATYPE


# ---------------------------------------------------------------------------
# _rewrite_metal_atype_for_vina — bench script integration — 4 tests
# ---------------------------------------------------------------------------
class TestRewriteMetalAtypeForVina:
    """Pin the chemotype vs fallback behaviour of the rewrite helper."""

    @pytest.fixture
    def rewrite_helper(self):
        # Lazy import so missing ``vina`` / ``meeko`` envs don't crash
        # the test collection.
        from molmetal.scripts import bench_cisplatin_real_vina
        return bench_cisplatin_real_vina._rewrite_metal_atype_for_vina

    @pytest.fixture
    def probe_helper(self):
        from molmetal.scripts import bench_cisplatin_real_vina
        return bench_cisplatin_real_vina._probe_vina_accepts_chemotype_atype

    @pytest.fixture
    def cisplatin_pdbqt_text(self) -> str:
        """A meeko-style cisplatin PDBQT with one ``M`` atype row."""
        return (
            "REMARK SMILES [NH2][Pt]([NH2])([Cl])[Cl]\n"
            "ROOT\n"
            "ATOM      1 Pt   UNL     1      -0.016   0.710   0.008  "
            "1.00  0.00    +0.000 M \n"
            "ATOM      2 Cl   UNL     1       0.076   2.060   2.229  "
            "1.00  0.00    +0.000 Cl\n"
            "ATOM      3 N    UNL     1      -1.795  -0.615   0.226  "
            "1.00  0.00    +0.000 NA\n"
            "ENDROOT\n"
            "TORSDOF 2\n"
        )

    def test_chemotype_vdw_true_writes_Pt_not_Mg(
        self, rewrite_helper, cisplatin_pdbqt_text, tmp_path,
    ):
        lig = tmp_path / "cisplatin.pdbqt"
        lig.write_text(cisplatin_pdbqt_text)
        out = rewrite_helper(lig, chemotype_vdw=True)
        assert out == lig
        text = out.read_text()
        pt_line = [l for l in text.splitlines() if l.startswith("ATOM") and "Pt" in l[12:16]][0]
        atype = pt_line[77:79].strip()
        assert atype == "Pt", (
            f"chemotype_vdw=True must rewrite 'M' → 'Pt' for cisplatin; "
            f"got atype={atype!r}"
        )
        # The Cl and N atom-types must NOT be touched.
        cl_line = [l for l in text.splitlines() if l.startswith("ATOM") and "Cl" in l[12:16]][0]
        assert cl_line[77:79].strip() == "Cl"
        n_line = [l for l in text.splitlines() if l.startswith("ATOM") and "N " in l[12:16]][0]
        assert n_line[77:79].strip() == "NA"

    def test_chemotype_vdw_false_writes_Mg_fallback(
        self, rewrite_helper, cisplatin_pdbqt_text, tmp_path,
    ):
        """Backward compat: chemotype_vdw=False keeps Wf-121 P1 Mg behaviour."""
        lig = tmp_path / "cisplatin.pdbqt"
        lig.write_text(cisplatin_pdbqt_text)
        out = rewrite_helper(lig, chemotype_vdw=False)
        text = out.read_text()
        pt_line = [l for l in text.splitlines() if l.startswith("ATOM") and "Pt" in l[12:16]][0]
        assert pt_line[77:79].strip() == "Mg"

    def test_non_metal_atype_pass_through_unchanged(
        self, rewrite_helper, tmp_path,
    ):
        """Atoms NOT typed 'M' must pass through untouched in both modes."""
        # No 'M' atype anywhere in the file.
        pdbqt = (
            "ATOM      1  N   UNL     1      -1.795  -0.615   0.226  "
            "1.00  0.00    +0.000 NA\n"
            "ATOM      2  C   UNL     1      -1.000   0.500   0.000  "
            "1.00  0.00    +0.000 C \n"
        )
        lig = tmp_path / "organic.pdbqt"
        lig.write_text(pdbqt)
        original = lig.read_text()
        out = rewrite_helper(lig, chemotype_vdw=True)
        assert out.read_text() == original
        out = rewrite_helper(lig, chemotype_vdw=False)
        assert out.read_text() == original

    def test_probe_returns_chemotype_when_vina_accepts_pdbqt(
        self, probe_helper, rewrite_helper, cisplatin_pdbqt_text, tmp_path,
    ):
        """When Vina accepts the chemotype PDBQT (no parse error),
        the probe must return the chemotype atype and NOT rewrite."""
        lig = tmp_path / "cisplatin.pdbqt"
        lig.write_text(cisplatin_pdbqt_text)
        # First, rewrite M → Pt via the chemotype helper (the realistic
        # call sequence in run_pocket_sweep).
        rewrite_helper(lig, chemotype_vdw=True)

        # Mock vina.Vina so the probe succeeds (no parse error).
        fake_vina = mock.MagicMock()
        fake_instance = mock.MagicMock()
        fake_instance.set_ligand_from_file.return_value = None  # no error
        fake_vina.return_value = fake_instance
        with mock.patch.dict(sys.modules, {"vina": mock.MagicMock(Vina=fake_vina)}):
            result = probe_helper(lig)
        assert result == "Pt"
        # The file should NOT have been rewritten to Mg.
        text = lig.read_text()
        pt_line = [l for l in text.splitlines() if "Pt" in l[12:16]][0]
        assert pt_line[77:79].strip() == "Pt"

    def test_probe_falls_back_to_Mg_when_vina_rejects_pdbqt(
        self, probe_helper, rewrite_helper, cisplatin_pdbqt_text, tmp_path,
    ):
        """When Vina rejects the chemotype PDBQT (raises parse error),
        the probe must rewrite to ``Mg`` and return ``"Mg-fallback"``."""
        lig = tmp_path / "cisplatin.pdbqt"
        lig.write_text(cisplatin_pdbqt_text)
        # First, rewrite M → Pt via the chemotype helper (the realistic
        # call sequence in run_pocket_sweep).
        rewrite_helper(lig, chemotype_vdw=True)

        # Mock vina.Vina so the probe raises a parse error.
        fake_vina = mock.MagicMock()
        fake_instance = mock.MagicMock()
        fake_instance.set_ligand_from_file.side_effect = RuntimeError(
            "PDBQT parsing error: Atom type Pt is not a valid AutoDock type"
        )
        fake_vina.return_value = fake_instance
        with mock.patch.dict(sys.modules, {"vina": mock.MagicMock(Vina=fake_vina)}):
            result = probe_helper(lig)
        assert result == f"{VINA_FALLBACK_ATYPE}-fallback"
        text = lig.read_text()
        pt_line = [l for l in text.splitlines() if "Pt" in l[12:16]][0]
        assert pt_line[77:79].strip() == "Mg"

    def test_fallback_atype_constant_is_Mg(self):
        """Pin :data:`VINA_FALLBACK_ATYPE` so downstream callers can rely on it."""
        assert VINA_FALLBACK_ATYPE == "Mg"

    def test_chemotype_vdw_cli_flag_default_is_true(self):
        """The ``--chemotype-vdw`` CLI flag must default to ``True``."""
        from molmetal.scripts import bench_cisplatin_real_vina
        parser = bench_cisplatin_real_vina._build_parser()
        args = parser.parse_args([])  # no flags
        assert args.chemotype_vdw is True
        # And ``--no-chemotype-vdw`` must flip it off.
        args_off = parser.parse_args(["--no-chemotype-vdw"])
        assert args_off.chemotype_vdw is False

    def test_chemotype_vdw_cli_flag_can_be_disabled(self):
        from molmetal.scripts import bench_cisplatin_real_vina
        parser = bench_cisplatin_real_vina._build_parser()
        args = parser.parse_args(["--no-chemotype-vdw"])
        assert args.chemotype_vdw is False