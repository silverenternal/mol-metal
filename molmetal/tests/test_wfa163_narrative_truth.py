#!/usr/bin/env python
"""Wf-A163 -- narrative-truth guard tests.

Pins the defect class found in Wf-A155 (and generalised by Wf-A163): a
``tag: MEASURED`` artifact whose PROSE states numbers that the artifact's own
machine fields do not (and cannot) back.

The canonical fixture is the PRE-REPAIR shape of
``wf_a155_ligand_connected_rate.py``: the prose carries the wrong row count,
the wrong sub-population count, and the WRONG POOL PATH, while the machine
fields carry the right ones.

CPU-ONLY.  No torch, no CUDA, no decode, no trainer.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "molmetal" / "scripts"))

import wfa163_narrative_truth_guard as guard  # noqa: E402


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def _repaired_c7() -> dict:
    """The REPAIRED c7 shape: prose computed from the machine fields."""
    return {
        "source_pool": "molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv",
        "denominators": {
            "n_rows_in_csv": 12669,
            "n_unparseable": 1,
            "n_analyzed": 12668,
            "n_metal_alone_in_own_fragment": 3056,
            "n_with_free_halide_counterion": 1903,
        },
        "metal_inclusive": {"single_fragment_rate": 0.08509630565203663},
        "purpose": (
            "This artifact is ONE pool of 12668 rows from ONE post-hoc decode "
            "(molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv)."
        ),
        "caveats": [
            "The 1903 rows with a free halide counterion are counted separately.",
        ],
    }


def _pre_repair_c8() -> dict:
    """The PRE-REPAIR shape: prose literals carried over from another pool."""
    return {
        "source_pool": "molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv",
        "denominators": {
            "n_rows_in_csv": 2112,
            "n_unparseable": 0,
            "n_analyzed": 2112,
            "n_metal_alone_in_own_fragment": 60,
        },
        "purpose": (
            "This artifact is never a restatement of the 8.51 % headline; it "
            "is ONE pool of 2112 rows from ONE post-hoc decode "
            "(cfm_c8_fullgen_2026_10_24/generated_smiles.csv)."
        ),
        "caveats": [
            "The 60 rows with a free halide counterion are counted separately.",
        ],
    }


# --------------------------------------------------------------------------
# the defect class
# --------------------------------------------------------------------------

def test_repaired_artifact_passes_declared_mode():
    art = _repaired_c7()
    report = guard.narrative_truth_report(
        art, narrative_paths=[("purpose",), ("caveats", "*")]
    )
    assert report["pass"], report
    assert report["n_errors"] == 0


def test_pre_repair_artifact_fails_declared_mode():
    """The prose names a POOL the artifact's own source_pool does not name."""
    art = _pre_repair_c8()
    # Force the c8 pool to be the artifact's own pool for this fixture so the
    # failure is about the ROW COUNT, not the pool path.
    art["source_pool"] = "molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv"
    art["purpose"] = (
        "This artifact is never a restatement of the 8.51 % headline; it is "
        "ONE pool of 9999 rows from ONE post-hoc decode."
    )
    report = guard.narrative_truth_report(
        art, narrative_paths=[("purpose",), ("caveats", "*")]
    )
    assert not report["pass"], report
    assert report["n_unbacked"] >= 1, report


def test_pool_misattribution_is_detected():
    """Prose naming a pool the artifact's own source_pool does not name."""
    art = _repaired_c7()
    art["purpose"] = (
        "Sourced from cfm_c8_fullgen_2026_10_24/generated_smiles.csv, "
        "ONE pool of 12668 rows."
    )
    found = guard.find_pool_misattribution(art)
    assert found, "expected a pool-misattribution finding"
    assert found[0]["severity"] == "contradiction"


def test_pool_misattribution_allows_cross_referenced_pool():
    """A pool named by ANOTHER machine field is a legitimate cross-reference."""
    art = _repaired_c7()
    art["related_artifacts"] = ["cfm_c8_fullgen_2026_10_24/generated_smiles.csv"]
    art["purpose"] = (
        "Cross-checked against cfm_c8_fullgen_2026_10_24/generated_smiles.csv; "
        "ONE pool of 12668 rows here."
    )
    found = guard.find_pool_misattribution(art)
    assert not found, found


def test_count_contradiction_detected():
    art = _repaired_c7()
    art["caveats"] = ["The 7777 rows with a free halide counterion are counted separately."]
    found = guard.find_count_contradictions(art)
    assert found, "expected a count contradiction"
    assert found[0]["value"] == 7777


def test_assert_raises_on_declared_unbacked_number():
    art = _repaired_c7()
    art["purpose"] = "ONE pool of 424242 rows from ONE post-hoc decode."
    with pytest.raises(guard.NarrativeTruthError):
        guard.assert_narrative_truth(
            art, narrative_paths=[("purpose",)], artifact_name="fixture"
        )


def test_assert_passes_on_repaired_artifact():
    art = _repaired_c7()
    report = guard.assert_narrative_truth(
        art, narrative_paths=[("purpose",), ("caveats", "*")], artifact_name="fixture"
    )
    assert report["pass"]


# --------------------------------------------------------------------------
# the guard must not fire on ordinary, non-narrative content
# --------------------------------------------------------------------------

def test_auto_mode_does_not_fire_on_non_claim_numbers():
    """A version string / TODO id / date is not a count claim."""
    art = {
        "source_pool": "molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv",
        "date": "2026-11-01",
        "notes": "See TODO-140 and TODO-141; generated by script v2.",
        "denominators": {"n_analyzed": 12668},
        "purpose": "ONE pool of 12668 rows.",
    }
    report = guard.narrative_truth_report(art)
    assert report["n_unbacked"] == 0, report


def test_auto_mode_is_advisory_not_a_gate():
    """Auto mode's unbacked findings must NOT raise by default."""
    art = {
        "source_pool": "molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv",
        "denominators": {"n_analyzed": 12668},
        "purpose": "The 4242 rows were kept.",
    }
    report = guard.assert_narrative_truth(art, artifact_name="fixture")
    assert report["n_unbacked"] >= 1
    assert report["pass"] is False
    # default strict=False in auto mode -> no raise
    with pytest.raises(guard.NarrativeTruthError):
        guard.assert_narrative_truth(art, artifact_name="fixture", strict=True)


# --------------------------------------------------------------------------
# cross-attribute citations: the one rule that can VERIFY a quoted number
# --------------------------------------------------------------------------

_CITED = "metrics/by_metric/cfm_fragmentation_audit.json"
_CITED_PATH = "/summary/generation_side/single_molecule_rate_end_to_end"


def _cited_value() -> float:
    p = REPO_ROOT / _CITED
    if not p.exists():
        pytest.skip("cited artifact not on disk")
    obj = json.loads(p.read_text(encoding="utf-8"))
    cur = obj
    for seg in _CITED_PATH.strip("/").split("/"):
        cur = cur[seg]
    if isinstance(cur, dict):
        return float(cur["median"])
    return float(cur)


def test_consistent_cross_reference_passes():
    v = _cited_value()
    art = {
        "note": (
            f"NOT a restatement of {_CITED} -> {_CITED_PATH} "
            f"(median {v})."
        )
    }
    found = [f for f in guard.find_stale_cross_references(art) if f["severity"] == "contradiction"]
    assert not found, found


def test_stale_cross_reference_is_detected():
    """A quoted value that no longer matches the cited artifact."""
    v = _cited_value()
    stale = round(v * 100.0 + 3.0, 2)  # a wrong percent rendering
    art = {
        "note": (
            f"NOT a restatement of {_CITED} -> {_CITED_PATH} "
            f"(median {stale} %)."
        )
    }
    found = [f for f in guard.find_stale_cross_references(art) if f["severity"] == "contradiction"]
    assert found, "expected a stale cross-reference finding"
    assert found[0]["rule"] == "stale_cross_reference"


# --------------------------------------------------------------------------
# the Wf-A163 counterfactual: the class, demonstrated
# --------------------------------------------------------------------------

def test_counterfactual_pre_repair_c7_is_flagged_if_present():
    p = (
        REPO_ROOT
        / "molmetal"
        / "reports"
        / "wfa163_narrative_truth_scan"
        / "counterfactual_pre_repair_c7.json"
    )
    if not p.exists():
        pytest.skip("counterfactual not on disk")
    art = json.loads(p.read_text(encoding="utf-8"))
    report = guard.narrative_truth_report(art)
    assert report["n_pool_misattribution"] >= 1, report
    assert report["n_count_contradiction"] >= 1, report


# --------------------------------------------------------------------------
# the guard must not fire on the REPAIRED a155 tool's own output
# --------------------------------------------------------------------------

def test_repaired_a155_tool_artifact_passes_if_present():
    """If the repaired a155 artifact is on disk, it must pass."""
    p = REPO_ROOT / "metrics" / "by_metric" / "wfa155_ligand_connected_rate.json"
    if not p.exists():
        pytest.skip("published a155 artifact not on disk")
    art = json.loads(p.read_text(encoding="utf-8"))
    report = guard.narrative_truth_report(art)
    assert report["n_pool_misattribution"] == 0, report["pool_misattribution"]
    assert report["n_count_contradiction"] == 0, report["count_contradiction"]


# --------------------------------------------------------------------------
# the guard module itself must stay importable and CPU-only
# --------------------------------------------------------------------------

def test_guard_module_has_no_gpu_dependency():
    src = (REPO_ROOT / "molmetal" / "scripts" / "wfa163_narrative_truth_guard.py").read_text(
        encoding="utf-8"
    )
    for banned in ("import torch", "torch.cuda", "cuda"):
        assert banned not in src, banned


def test_guard_import_is_side_effect_free():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    code = (
        "import sys; sys.path.insert(0, %r);"
        "import wfa163_narrative_truth_guard as g;"
        "print(g.narrative_truth_report({'a': 1})['pass'])"
        % str(REPO_ROOT / "molmetal" / "scripts")
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(REPO_ROOT)
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "True", out.stdout
