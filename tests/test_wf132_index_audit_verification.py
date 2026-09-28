"""pytest wrapper for Wf-132 Phase 3 cross-link verification.

Run: pytest tests/test_wf132_index_audit_verification.py -v
CPU-only static markdown structure check.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFICATION_PY = REPO_ROOT / "molmetal" / "reports" / "wf132_index_audit" / "verification.py"


def _load_verification_module():
    spec = importlib.util.spec_from_file_location("wf132_verification", VERIFICATION_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def verification_module():
    return _load_verification_module()


@pytest.fixture(scope="module")
def report(verification_module):
    return verification_module.main()


def test_all_5_critical_markers_present(report):
    """5 critical markers in INDEX.md §17 must all be present."""
    assert report["summary"]["critical_markers_failed"] == 0
    assert report["summary"]["critical_markers_passed"] == 5
    markers = report["checks"]["critical_markers"]["checks"]
    for m in markers:
        assert m["status"] == "PASS", f"Missing marker: {m['marker']}"


def test_lambda_roadmap_section_4_2_present(report):
    """LAMBDA_ROADMAP.md §4.2 must exist with all 12 L-gaps."""
    sec = report["checks"]["lambda_roadmap_section"]
    assert sec["section_present"] is True
    assert sec["l_gaps_complete"] is True
    assert sec["status"] == "PASS"


def test_orphan_close_out_links_resolve(report):
    """All 7 orphan close-out cross-links in §17.10 must resolve on disk.

    Per Wf-133 brief: 7 close-out docs at TODO/completed/{121,122,123,
    125,126,127,129}_*.md — every one should be present.
    """
    s = report["summary"]
    assert s["orphan_close_out_failed"] == 0, (
        f"orphan_close_out has {s['orphan_close_out_failed']} failures"
    )
    assert s["orphan_expected_resolved"] == 7
    assert s["orphan_still_missing"] == 0
    rows = report["checks"]["orphan_close_out"]["rows"]
    assert len(rows) == 7
    resolved = sum(1 for r in rows if r["verdict"] == "RESOLVED")
    missing = sum(1 for r in rows if r["verdict"] == "MISSING")
    assert resolved == 7
    assert missing == 0


def test_cross_link_paths_resolve(report):
    """All 8 §17.12 cross-link file paths resolve on disk."""
    s = report["summary"]
    assert s["cross_link_paths_failed"] == 0
    assert s["cross_link_paths_passed"] == 8


def test_data_assembly_present_with_lead_claim_keys(report):
    """data_assembly.md must exist with all 6 lead-claim keys."""
    da = report["checks"]["data_assembly"]
    assert da["exists"] is True
    assert da["lead_claim_keys_present"] == da["lead_claim_keys_total"]


def test_3_audit_report_refs_present(report):
    """3 audit report refs (A1/A2/A3) in both INDEX.md §17 and data_assembly.md §1."""
    assert report["summary"]["audit_refs_status"] == "PASS"
    refs = report["checks"]["audit_report_refs"]
    assert all(refs["refs_in_index"].values())
    assert all(refs["refs_in_data_assembly"].values())
    assert refs["all_present"] is True


def test_overall_status_pass(report):
    """Overall verification PASS — every check category green."""
    assert report["summary"]["overall_status"] == "PASS"


def test_honest_framing_notes_present(report):
    """Honest framing notes (party-twenty-spirit §3.5) — must include
    at least 5 entries to satisfy the brief."""
    notes = report["honest_framing_notes"]
    assert len(notes) >= 5
    text = " ".join(notes).lower()
    assert "wf-121" in text
    assert "wf-122" in text
    assert "wf-123" in text
    assert "wf-127" in text
    assert "wf-129" in text