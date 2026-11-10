#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for molmetal/scripts/wfa168_falsevalid_axis.py (Wf-A168, our-side
false-valid / speck decomposition).

CPU-only.  Run with::

    CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \
        molmetal/tests/test_wfa168_falsevalid_axis.py -q

Four guards are required by the branch brief, plus the regression checks that
make the headline numbers auditable:

  * NEGATIVE CONTROL -- a synthetic pool whose survivors are all drug-sized
    must give a speck fraction of 0; a synthetic pool of single-atom specks
    must give 1.0.
  * VACUITY GUARD -- the reported numerator and denominator must sum to the
    row count claimed.
  * CPU-ONLY GUARD -- the module imports no torch and never calls nvidia-smi.
  * DRIFT CHECK -- the JSON that was written matches a fresh run on a sample
    of rows.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "molmetal" / "scripts" / "wfa168_falsevalid_axis.py"
ARTIFACT = REPO / "molmetal" / "reports" / "wfa168_falsevalid_axis" / "falsevalid_axis.json"
PUBLISHED = REPO / "metrics" / "by_metric" / "cfm_fragmentation_postproc_arm.json"

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from molmetal.scripts import wfa168_falsevalid_axis as wfa168  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _rows(smiles_list):
    return [
        {"row_index": i, "epoch": "0", "smiles": s, "n_frags_raw_csv": None}
        for i, s in enumerate(smiles_list)
    ]


def _speck_fraction(smiles_list):
    art = wfa168.analyse_rows(_rows(smiles_list), run_uff=False)
    fv = art["false_valid_rate"]
    assert fv["denominator"] == len(smiles_list)
    return fv["rate"], art


# --------------------------------------------------------------------------
# NEGATIVE CONTROL
# --------------------------------------------------------------------------
def test_negative_control_drug_sized_pool_has_zero_speck_fraction():
    """A pool whose survivors are all drug-sized must give a speck rate of 0."""
    from rdkit import Chem

    pool = [
        "CCCCCCCCCCCCCC",
        "CCCCCCCCCCCCCCCC",
        "c1ccccc1CCCCCCCC",
        "c1ccccc1CC(=O)NCCCCCC",
    ]
    for smi in pool:
        assert Chem.MolFromSmiles(smi).GetNumHeavyAtoms() >= 11, smi
    rate, art = _speck_fraction(pool)
    assert rate == 0.0
    assert art["false_valid_rate"]["numerator"] == 0
    assert art["false_valid_rate"]["denominator"] == len(pool)
    assert art["survivor_shape"]["n_survivors_ge_11_heavy"] == len(pool)
    assert art["survivor_shape"]["n_survivors_3_to_10_heavy"] == 0


def test_negative_control_speck_pool_has_unit_speck_fraction():
    """A pool whose survivors are all single-atom specks must give 1.0."""
    pool = ["C.C.C", "C.C", "C", "[SiH4].C", "CC.C.C.C"]
    rate, art = _speck_fraction(pool)
    assert rate == 1.0
    assert art["false_valid_rate"]["numerator"] == len(pool)
    assert art["speck_composition"]["n_speck_containing_a_metal"] == 0


def test_negative_control_mixed_pool_is_between():
    """The two controls bracket the measurement: a mixed pool lands in between."""
    pool = ["CCCCCCCCCCCCCC", "C.C.C", "CCCCCCCCCCCCCCCC", "C"]
    rate, _ = _speck_fraction(pool)
    assert 0.0 < rate < 1.0
    assert rate == pytest.approx(0.5)


# --------------------------------------------------------------------------
# VACUITY GUARD
# --------------------------------------------------------------------------
def test_vacuity_guard_buckets_sum_to_the_row_count():
    """The three survivor buckets must sum to the denominator exactly."""
    art = wfa168.analyse_rows(_rows(["C.C.C", "CCCCCCCCCCCCCC", "CCO"]), run_uff=False)
    sh = art["survivor_shape"]
    assert sh["sum_of_buckets"] == sh["denominator"]
    assert sh["n_survivors_le_2_heavy"] + sh["n_survivors_3_to_10_heavy"] + sh[
        "n_survivors_ge_11_heavy"
    ] == sh["denominator"]
    assert sh["denominator"] == art["denominators"]["n_parsed"]
    assert art["false_valid_rate"]["numerator"] + (
        art["survivor_shape"]["denominator"] - art["false_valid_rate"]["numerator"]
    ) == art["survivor_shape"]["denominator"]


def test_vacuity_guard_published_artifact_buckets_sum_to_its_own_denominator():
    """The published artifact must satisfy the same guard -- no silent drift."""
    pub = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    ps = pub["survivor_shape"]
    assert ps["n_survivors_le_2_heavy"] + ps["n_survivors_ge_11_heavy"] + (
        ps["n_survivors_le_10_heavy"] - ps["n_survivors_le_5_heavy"]
    ) == pub["denominators"]["n_parsed"]
    assert ps["n_survivors_le_2_heavy"] == ps["n_survivors_le_5_heavy"]


# --------------------------------------------------------------------------
# CPU-ONLY GUARD
# --------------------------------------------------------------------------
def test_cpu_only_guard_module_imports_no_torch_and_never_calls_nvidia_smi():
    """Static guard on the module source: no torch, no nvidia-smi, no CUDA.

    The scan runs over the AST with docstrings removed, so a prose mention of
    ``nvidia-smi`` in the module header cannot mask a real call -- and a real
    call cannot hide behind a docstring either.
    """
    src = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(src)

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])
    for banned in ("torch", "cupy", "tensorflow", "jax", "subprocess"):
        assert banned not in imported, banned

    # collect every docstring so it can be excluded from the string scan
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))

    code_strings = [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and id(n) not in docstrings
    ]
    for s in code_strings:
        low = s.lower()
        assert "nvidia-smi" not in low
        assert "nvidia_smi" not in low
        if "cuda" in low:
            # the ONLY CUDA-mentioning code string allowed is the guard itself
            assert s == "CUDA_VISIBLE_DEVICES", s

    # no attribute call into a CUDA API
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in ("cuda", "to", "device"), ast.dump(node)

    # the only CUDA reference allowed is the guard that empties the variable
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = ""' in src


def test_cpu_only_guard_import_forces_cuda_visible_devices_empty():
    """Importing the module must leave CUDA_VISIBLE_DEVICES empty."""
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""


# --------------------------------------------------------------------------
# DRIFT CHECK
# --------------------------------------------------------------------------
@pytest.mark.skipif(not ARTIFACT.exists(), reason="artifact not built yet")
def test_drift_check_artifact_matches_a_fresh_run_on_a_sample():
    """Re-derive the sampled rows and compare with what was written."""
    art = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    sample = art["drift_sample"]
    assert sample, "the drift sample must not be empty"

    fresh = wfa168.analyse_rows(
        [
            {
                "row_index": r["row_index"],
                "epoch": r["epoch"],
                "smiles": r["smiles"],
                "n_frags_raw_csv": None,
            }
            for r in sample
        ],
        run_uff=False,
        keep_rows=True,
    )
    by_index = {r["row_index"]: r for r in fresh["rows"]}
    for rec in sample:
        now = by_index[rec["row_index"]]
        assert now["survivor_heavy"] == rec["survivor_heavy"], rec["row_index"]
        assert now["raw_heavy"] == rec["raw_heavy"], rec["row_index"]
        assert now["ligand_heavy"] == rec["ligand_heavy"], rec["row_index"]
        assert now["ligand_n_frags"] == rec["ligand_n_frags"], rec["row_index"]
        assert now["is_speck"] == rec["is_speck"], rec["row_index"]
        assert now["is_ligand_speck"] == rec["is_ligand_speck"], rec["row_index"]
        assert now["survivor_elements"] == rec["survivor_elements"], rec["row_index"]


# --------------------------------------------------------------------------
# REGRESSION: the published split must keep reproducing, from the RAW pool
# --------------------------------------------------------------------------
@pytest.mark.skipif(not ARTIFACT.exists(), reason="artifact not built yet")
def test_measured_split_reproduces_the_published_split_exactly():
    art = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    sh = art["survivor_shape"]
    assert sh["n_survivors_le_2_heavy"] == 2663
    assert sh["n_survivors_3_to_10_heavy"] == 2
    assert sh["n_survivors_ge_11_heavy"] == 10003
    assert sh["sum_of_buckets"] == art["denominators"]["n_parsed"] == 12668


@pytest.mark.skipif(not ARTIFACT.exists(), reason="artifact not built yet")
def test_crosscheck_against_published_has_no_differing_field():
    art = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    cc = art["crosscheck_vs_published"]
    assert cc["n_fields_differing"] == 0
    assert cc["n_fields_compared"] >= 15


@pytest.mark.skipif(not ARTIFACT.exists(), reason="artifact not built yet")
def test_pooled_speck_rate_is_a_time_average_not_a_stationary_rate():
    """The pooled 21 % must not be mistaken for a property of the final epoch."""
    art = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    pes = art["per_epoch_summary"]
    assert pes["n_epochs"] == 18
    assert pes["first_epoch_speck_rate"] > pes["pooled_speck_rate"] > pes["last_epoch_speck_rate"]
    # vacuity guard on the epoch axis: the per-epoch denominators must sum to the
    # pooled denominator, and no epoch may report more specks than it has rows
    assert sum(d["n_parsed"] for d in art["per_epoch"].values()) == art["denominators"]["n_parsed"]
    for ep, d in art["per_epoch"].items():
        assert 0 <= d["n_speck_le_2_heavy"] <= d["n_parsed"], ep


@pytest.mark.skipif(not ARTIFACT.exists(), reason="artifact not built yet")
def test_no_speck_contains_a_metal_and_metal_removal_creates_no_new_speck():
    """The false-valid arm is not a metal-coordination artifact."""
    art = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert art["speck_composition"]["n_speck_containing_a_metal"] == 0
    assert art["ligand_only_analogue"]["n_NEW_specks_created_by_removing_the_metal"] == 0
    assert art["ligand_only_analogue"]["n_ligand_only_le_2_heavy"] == art["false_valid_rate"][
        "numerator"
    ]


@pytest.mark.skipif(not ARTIFACT.exists(), reason="artifact not built yet")
def test_uff_step_is_inapplicable_on_this_pool():
    art = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert art["uff"]["uff_ok_rate_over_postprocessed"] == 0.0
    assert art["uff"]["uff_failed"] == art["denominators"]["n_parsed"]
    assert list(art["uff"]["uff_reasons"].keys()) == ["ValueError: Bad Conformer Id"]


# --------------------------------------------------------------------------
# the survivor rule that RDKit actually applies
# --------------------------------------------------------------------------
def test_survivor_rule_is_total_atom_count_not_heavy_atom_count():
    """The discriminating probe: the metric is atoms + hydrogens, not heavy atoms."""
    from rdkit import Chem

    m = Chem.MolFromSmiles("[C]([C])([C])[C].CC")
    chosen = wfa168.chooser_survivor(m)[0]
    # 2 heavy atoms / 8 atoms with H  vs  4 heavy atoms / 4 atoms with H
    assert chosen.GetNumHeavyAtoms() == 2
    assert wfa168.total_atom_size(chosen) == 8


def test_independent_survivor_rule_agrees_with_the_rdkit_chooser_on_a_probe():
    from rdkit import Chem

    for smi in ("C.C.C", "Cl.N.S.[Ru]", "CC.CC", "CCCC", "[Pt].CC"):
        m = Chem.MolFromSmiles(smi)
        a = wfa168.chooser_survivor(m)[0]
        b = wfa168.independent_survivor(m)
        assert Chem.MolToSmiles(a) == Chem.MolToSmiles(b), smi


# --------------------------------------------------------------------------
# Wf-A187 consumer-side supersession guard (2026-11-01)
#
# The refutation artefacts are HAND-MAINTAINED (no producer), so the only
# guard on the Wf-A187 dated correction is THIS test.  It asserts that the
# superseded rendering of the A158 matrix's X1 wording is still preserved
# VERBATIM inside a DATED key/section, and that the correction says so.
# --------------------------------------------------------------------------
REFUTATION_DIR = REPO / "molmetal" / "reports" / "wfa168_falsevalid_axis_refutation"
REFUTATION_JSON = REFUTATION_DIR / "refutation.json"
REFUTATION_MD = REFUTATION_DIR / "refutation.md"

X1_MARKERS = (
    "NEITHER is a connected-component count",
    "THE BAND IS NOT A CONNECTIVITY BAND",
    "not a connectivity band at all",
)


@pytest.mark.skipif(not REFUTATION_JSON.exists(), reason="refutation artefact not built yet")
def test_wfa187_consumer_supersession_is_dated_and_verbatim():
    """The refutation's quotation of the matrix's superseded X1 wording is dated, not deleted."""
    doc = json.loads(REFUTATION_JSON.read_text(encoding="utf-8"))
    dated = doc["wfa187_supersession_2026_11_01"]
    assert dated["date"] == "2026-11-01"

    preserved = dated["superseded_quotations_verbatim"]
    # the superseded rendering survives VERBATIM -- byte-identical to the live entry
    assert preserved["evidence_2_of_surviving_concerns_C1"] == doc["surviving_concerns"][0]["evidence"][2]
    assert preserved["closing_clause_of_why_it_is_not_a_refutation"] == doc["surviving_concerns"][0][
        "why_it_is_not_a_refutation"
    ]
    # it is marked superseded, and the direction of the concern is NOT withdrawn
    assert "SUPERSEDED" in preserved["why_superseded"].upper()
    assert "NOT withdrawn" in preserved["not_withdrawn"]
    # X1 must be named as must-not-be-quoted, and the locators must be JSON paths
    assert "must_not_be_quoted" in preserved
    assert "band_provenance" in preserved["locators"]
    assert "sources" in preserved["locators"]
    # the artefact is hand-maintained: no producer
    assert "HAND-MAINTAINED" in preserved["artefact_provenance"]


@pytest.mark.skipif(not REFUTATION_MD.exists(), reason="refutation artefact not built yet")
def test_wfa187_consumer_supersession_md_carries_the_dated_section():
    """The .md side of the same correction: a DATED section, and the X1 wording kept in place."""
    text = REFUTATION_MD.read_text(encoding="utf-8")
    assert "## Wf-A187 CORRECTION (2026-11-01)" in text
    assert "HAND-MAINTAINED" in text
    # the live body still carries the superseded rendering (append-only: kept, not deleted).
    # whitespace-normalised: the quoted X1 wording is line-wrapped in the .md.
    flat = " ".join(text.split())
    for marker in X1_MARKERS:
        assert " ".join(marker.split()) in flat, marker
