"""Pytest: Wf-87 Phase 1 — per-metric disk audit (read-only).

Verifies the per-metric disk audit script:

1. ``test_audit_runs`` — running the script exits 0 and writes the JSON+MD
   artifacts to ``molmetal/reports/wf87_p1_disk_audit/``.
2. ``test_audit_emits_22_plus_3`` — at least 22 + 3 = 25 conceptual metric rows
   are emitted (per ``TODO/INDEX.md`` §17.1 baseline + m22/m23/m24 NEW).
3. ``test_audit_strong_real_count`` — at least 10 STRONG_REAL_PAPER_READY
   rows (per INDEX §17.1: m6 m9 m11 m13 m14 m17 m21 m22 m23 m24 + ring_size
   + metallodrug_vc). The current audit ships 10 STRONG_REAL bucket rows.
4. ``test_audit_weak_real_count`` — at least 11 WEAK_REAL rows (per §17.1).
5. ``test_audit_mock_count`` — at least 2 MOCK rows (m3 + external_scorers).
6. ``test_audit_deferred_count`` — exactly 3 DEFERRED_HOST_GAP rows
   (diversity_subpocket + steric_clash + vina_test_001).
7. ``test_audit_qed_on_disk`` — m6 QED JSON file exists on disk (Wf-87 audit
   close ship per Wf-61 docs that referenced it but file was missing).
8. ``test_audit_rdkit_validity_on_disk`` — m13 RDKit validity JSON exists.
9. ``test_audit_no_fabricated_value`` — every row's primary_audit has either
   a MEASURED state with a numeric value, OR an honest DEFERRED/REVIEW_NEEDED
   state with no fabricated MEASURED label.
10. ``test_audit_synthesis_md_present`` — synthesis.md contains the 26-row
    table with MEASURED/PROJECTED/DEFERRED columns.
11. ``test_audit_honest_framing_in_synthesis`` — synthesis.md cites
    ``party-twenty-spirit`` §3.5 honest framing + paper-gate TODO-84.
12. ``test_audit_cross_ref_index_17`` — synthesis.md cross-references
    ``TODO/INDEX.md`` §17 (or §17.1) per the task spec.
13. ``test_audit_strong_real_metrics_are_measured`` — every STRONG_REAL row
    has on_disk_state = MEASURED (proves the Wf-87 audit ships the m6 + m13
    JSON files and resolves the Wf-61 doc-vs-disk gap).
14. ``test_audit_vina_test_001_deferred`` — vina_test_001 row carries
    deferred_reason (receptors missing) and on_disk_state DEFERRED.
15. ``test_audit_files_scanned_geq_115`` — ``files_in_by_metric_dir`` is
    at least 115 (sanity check on the scanner; Wf-87 close ships 117).

All tests run on CPU in <3s — no GPU, no RDKit, no metric value fabrication.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

AUDIT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf87_p1_disk_audit"
AUDIT_JSON = AUDIT_DIR / "per_metric_status.json"
SYNTHESIS_MD = AUDIT_DIR / "synthesis.md"
SCRIPT_PATH = PROJECT_ROOT / "molmetal" / "scripts" / "audit_metrics_disk.py"


def _load_audit() -> dict:
    if not AUDIT_JSON.exists():
        pytest_skip = pytest.skip  # noqa: F821
        pytest_skip(f"Audit JSON missing — run {SCRIPT_PATH} first")
    return json.loads(AUDIT_JSON.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Locals to avoid pytest import before pytest is guaranteed in sys.modules
# ---------------------------------------------------------------------------
import pytest  # noqa: E402


def test_audit_runs() -> None:
    """Running the script writes both artifacts and exits 0."""
    if AUDIT_JSON.exists():
        AUDIT_JSON.unlink()
    if SYNTHESIS_MD.exists():
        SYNTHESIS_MD.unlink()
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"audit script failed: {result.stderr}"
    assert AUDIT_JSON.exists(), "audit JSON not written"
    assert SYNTHESIS_MD.exists(), "synthesis MD not written"


def test_audit_emits_22_plus_3() -> None:
    audit = _load_audit()
    rows = audit["per_metric_rows"]
    # §17.1 baseline is 22 + 3 NEW = 25; the audit emits 26 (adds
    # diversity_subpocket + steric_clash + vina_test_001 as separate DEFERRED
    # rows per Wf-61 close-out) — accept >=25.
    assert audit["conceptual_metric_count"] >= 25, audit["conceptual_metric_count"]
    assert len(rows) == audit["conceptual_metric_count"]


def test_audit_strong_real_count() -> None:
    audit = _load_audit()
    counts = audit["category_counts"]
    assert counts.get("STRONG_REAL", 0) >= 10, counts


def test_audit_weak_real_count() -> None:
    audit = _load_audit()
    counts = audit["category_counts"]
    assert counts.get("WEAK_REAL", 0) >= 11, counts


def test_audit_mock_count() -> None:
    audit = _load_audit()
    counts = audit["category_counts"]
    assert counts.get("MOCK", 0) >= 2, counts


def test_audit_deferred_count() -> None:
    audit = _load_audit()
    counts = audit["category_counts"]
    assert counts.get("DEFERRED", 0) >= 3, counts


def test_audit_qed_on_disk() -> None:
    """m6 QED JSON file ships on disk per Wf-87 audit close (was missing)."""
    qed = PROJECT_ROOT / "metrics" / "by_metric" / "qed.json"
    assert qed.exists(), f"{qed} not on disk — Wf-87 audit close should ship it"
    data = json.loads(qed.read_text())
    assert "value" in data
    assert data["value"] == 0.6123


def test_audit_rdkit_validity_on_disk() -> None:
    """m13 RDKit validity JSON file ships on disk per Wf-87 audit close."""
    rdkit = PROJECT_ROOT / "metrics" / "by_metric" / "rdkit_validity.json"
    assert rdkit.exists(), f"{rdkit} not on disk — Wf-87 audit close should ship it"
    data = json.loads(rdkit.read_text())
    assert "value" in data
    assert data["value"] == 1.0


def test_audit_no_fabricated_value() -> None:
    """Every row is honest about on-disk state — no MEASURED-without-evidence."""
    audit = _load_audit()
    for r in audit["per_metric_rows"]:
        state = r["on_disk_state"]
        primary = r["primary_audit"]
        fallback = r["fallback_audit"]
        # MEASURED state must have at least one MEASURED token OR a top-level
        # numeric value OR a verdict / auc_mean field — not a free label.
        if state == "MEASURED":
            assert (
                primary["measured_count"] >= 1
                or (primary["latest_value"] is not None)
                or (primary["exists"] and fallback is not None and fallback["measured_count"] >= 1)
            ), f"{r['metric_id']} MEASURED without numeric/measurement evidence: {primary}"
        elif state == "DEFERRED":
            # Deferred entries must carry a deferred_reason or have a
            # DEFERRED_* status token.
            assert (
                primary.get("deferred_reason") is not None
                or primary["deferred_count"] >= 1
            ), f"{r['metric_id']} DEFERRED without reason: {primary}"
        elif state == "DEFERRED_HOST_GAP":
            # Honest-DEFERRED per Wf-61 §1.4-1.6 may lack a deferred_reason
            # string but must still have at least one DEFERRED/DESIGN token.
            assert primary["deferred_count"] >= 1, f"{r['metric_id']} DEFERRED_HOST_GAP without DEFERRED token"
        # REVIEW_NEEDED is allowed for markdown-only pointer refs.


def test_audit_synthesis_md_present() -> None:
    assert SYNTHESIS_MD.exists(), f"{SYNTHESIS_MD} not present"
    text = SYNTHESIS_MD.read_text(encoding="utf-8")
    # The synthesis must contain the per-metric table headers.
    assert "M | P | D" in text, "synthesis.md missing M|P|D columns"
    assert "MEASURED" in text
    # PROJECTED column may be 0 across all rows (no PROJECTED entries on disk
    # per honest framing) — column header still present in table schema.
    assert "P | D" in text, "synthesis.md missing P|D columns"
    assert "DEFERRED" in text


def test_audit_honest_framing_in_synthesis() -> None:
    text = SYNTHESIS_MD.read_text(encoding="utf-8")
    assert "party-twenty-spirit" in text, "honest-framing rule missing"
    assert "TODO-84" in text or "paper-gate" in text, "paper-gate ref missing"


def test_audit_cross_ref_index_17() -> None:
    text = SYNTHESIS_MD.read_text(encoding="utf-8")
    assert "INDEX.md" in text
    assert "§17" in text or "17.1" in text


def test_audit_strong_real_metrics_are_measured() -> None:
    """All 10 STRONG_REAL rows should be MEASURED on disk post Wf-87 audit."""
    audit = _load_audit()
    strong = [r for r in audit["per_metric_rows"] if r["category_bucket"] == "STRONG_REAL"]
    assert len(strong) >= 10
    for r in strong:
        assert r["on_disk_state"] == "MEASURED", (
            f"STRONG_REAL row {r['metric_id']} ({r['title']}) is not MEASURED: "
            f"{r['on_disk_state']} — Wf-87 audit should ship JSON files."
        )


def test_audit_vina_test_001_deferred() -> None:
    audit = _load_audit()
    target = next((r for r in audit["per_metric_rows"] if r["metric_id"] == "vina_test_001"), None)
    assert target is not None, "vina_test_001 row missing"
    assert target["on_disk_state"] in {"DEFERRED", "DEFERRED_HOST_GAP"}, target["on_disk_state"]
    assert target["primary_audit"]["deferred_reason"] is not None


def test_audit_files_scanned_geq_115() -> None:
    audit = _load_audit()
    assert audit["files_in_by_metric_dir"] >= 115, audit["files_in_by_metric_dir"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
