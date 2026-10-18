"""Tests for the granularity gate (FIX-P2 / DEF-6)."""

from __future__ import annotations

import pytest

from molmetal.features.granularity import (
    GRANULARITY_COMPLEX,
    GRANULARITY_FRAGMENT,
    assert_granularity,
    check_artifact,
    granularity_of_source,
    granularity_of_spec,
    requires_check,
)
from molmetal.features.spec import COMPLEX, LEGACY_MAIN


class _Bundle:
    def __init__(self, spec):
        self.spec = spec


def test_sources_map_to_units():
    assert granularity_of_source("ligand_canonical") == GRANULARITY_COMPLEX
    assert granularity_of_source("ligand_main_fragment") == GRANULARITY_FRAGMENT
    with pytest.raises(ValueError):
        granularity_of_source("ligand_smiles")


def test_presets_declare_their_unit():
    assert granularity_of_spec(LEGACY_MAIN) == GRANULARITY_FRAGMENT
    assert granularity_of_spec(COMPLEX) == GRANULARITY_COMPLEX


def test_assert_granularity_accepts_matching_bundle():
    assert_granularity(_Bundle(COMPLEX), GRANULARITY_COMPLEX)


def test_assert_granularity_rejects_fragment_matrix_for_complex_cohort():
    with pytest.raises(ValueError, match="DEF-6"):
        assert_granularity(_Bundle(LEGACY_MAIN), GRANULARITY_COMPLEX)


def _payload(**design):
    return {
        "artifact": "x",
        "best_arm": "complex",
        "splits": {"canonical_grouped": {"complex": {"r2": 0.5}}},
        "design": design,
    }


def test_artifact_with_complex_headline_passes():
    payload = _payload(arm_ligand_source={"complex": "ligand_canonical"})
    assert check_artifact(payload) == []


def test_artifact_with_fragment_headline_fails():
    payload = _payload(arm_ligand_source={"complex": "ligand_main_fragment"})
    problems = check_artifact(payload)
    assert len(problems) == 1
    assert "DEF-6" in problems[0]


def test_best_arm_canonical_split_key_is_used():
    """msrs_complex_features names its headline `best_arm_canonical_split`."""
    payload = {
        "artifact": "msrs_complex_features",
        "best_arm_canonical_split": "complex",
        "splits": {"canonical_grouped": {"complex": {"r2": 0.5}}},
        "design": {
            "arm_ligand_source": {
                "main": "ligand_main_fragment",
                "complex": "ligand_canonical",
            }
        },
    }
    assert check_artifact(payload) == []
    payload["best_arm_canonical_split"] = "main"
    assert check_artifact(payload)


def test_artifact_without_source_declaration_fails():
    problems = check_artifact(_payload())
    assert len(problems) == 1
    assert "does not declare" in problems[0]


def test_ligand_block_string_is_used_as_fallback():
    ok = _payload(ligand_block="ECFP4(ligand_canonical), 2048 bits")
    assert check_artifact(ok) == []
    bad = _payload(ligand_block="ECFP4(ligand_main_fragment)")
    assert check_artifact(bad)


def test_artifact_without_complex_split_is_out_of_scope():
    payload = {"artifact": "x", "splits": {"fragment_grouped": {}}}
    assert requires_check(payload) is False
    assert check_artifact(payload) == []
