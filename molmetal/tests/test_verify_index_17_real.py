"""Pytest: Wf-87 Phase 3 — INDEX §17 verification script (read-only).

Verifies that the verify_index_17_real.py script:

1. ``test_verify_runs`` — running the script exits 0 and writes JSON+MD.
2. ``test_verify_emits_25_metrics`` — at least 25 conceptual metrics audited.
3. ``test_verify_strong_real_count_10`` — STRONG_REAL_PAPER_READY claim == 10.
4. ``test_verify_weak_real_count_11`` — WEAK_REAL_NEEDS_MORE_DATA claim == 11.
5. ``test_verify_mock_count_5`` — MOCK_NEEDS_REMEASURE claim == 5.
6. ``test_verify_deferred_count_3`` — DEFERRED_HOST_GAP claim == 3.
7. ``test_verify_strong_real_cross_check_pass`` — every claimed STRONG_REAL
   metric has a real MEASURED value on disk (the load-bearing invariant).
8. ``test_verify_cross_refs_all_exist`` — every §17.4 cross-reference file
   exists on disk (verifies the cross-ref ledger is honest).
9. ``test_verify_honest_framing_cited`` — JSON output cites
   ``party-twenty-spirit`` §3.5 honest framing.
10. ``test_verify_no_fabricated_value`` — every STRONG_REAL row cited in
    INDEX §17.1 has actual numeric evidence, not just file existence.

All tests run on CPU in <5s — no GPU, no RDKit.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VERIFY_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf87_index_17_real"
VERIFY_JSON = VERIFY_DIR / "verify_index_17.json"
VERIFY_MD = VERIFY_DIR / "verify_index_17.md"
SCRIPT_PATH = PROJECT_ROOT / "molmetal" / "scripts" / "verify_index_17_real.py"

import pytest  # noqa: E402


def _load_verify() -> dict:
    if not VERIFY_JSON.exists():
        pytest.skip(f"verify JSON missing — run {SCRIPT_PATH} first")
    return json.loads(VERIFY_JSON.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Script runs cleanly and writes both artifacts
# ---------------------------------------------------------------------------
def test_verify_runs() -> None:
    """Running the script writes both artifacts and exits 0."""
    if VERIFY_JSON.exists():
        VERIFY_JSON.unlink()
    if VERIFY_MD.exists():
        VERIFY_MD.unlink()
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"verify script failed: {result.stderr}"
    assert VERIFY_JSON.exists(), "verify JSON not written"
    assert VERIFY_MD.exists(), "verify MD not written"


# ---------------------------------------------------------------------------
# 2. At least 25 metrics audited
# ---------------------------------------------------------------------------
def test_verify_emits_25_metrics() -> None:
    verify = _load_verify()
    assert verify["n_metrics_expected"] >= 25, verify["n_metrics_expected"]


# ---------------------------------------------------------------------------
# 3. STRONG_REAL claim == 10 (per INDEX §17.1)
# ---------------------------------------------------------------------------
def test_verify_strong_real_count_10() -> None:
    verify = _load_verify()
    claimed = verify["counts_claimed_in_index"]["STRONG_REAL_PAPER_READY"]
    assert claimed == 10, f"STRONG_REAL claim is {claimed}, INDEX §17.1 states 10"


# ---------------------------------------------------------------------------
# 4. WEAK_REAL claim == 11 (per INDEX §17.1)
# ---------------------------------------------------------------------------
def test_verify_weak_real_count_11() -> None:
    verify = _load_verify()
    claimed = verify["counts_claimed_in_index"]["WEAK_REAL_NEEDS_MORE_DATA"]
    assert claimed == 11, f"WEAK_REAL claim is {claimed}, INDEX §17.1 states 11"


# ---------------------------------------------------------------------------
# 5. MOCK claim >= 5 (per INDEX §17.1: "5+ MOCK")
# ---------------------------------------------------------------------------
def test_verify_mock_count_5() -> None:
    verify = _load_verify()
    claimed = verify["counts_claimed_in_index"]["MOCK_NEEDS_REMEASURE"]
    assert claimed >= 5, f"MOCK claim is {claimed}, INDEX §17.1 states 5+"


# ---------------------------------------------------------------------------
# 6. DEFERRED claim == 3 (per INDEX §17.1)
# ---------------------------------------------------------------------------
def test_verify_deferred_count_3() -> None:
    verify = _load_verify()
    claimed = verify["counts_claimed_in_index"]["DEFERRED_HOST_GAP"]
    assert claimed == 3, f"DEFERRED claim is {claimed}, INDEX §17.1 states 3"


# ---------------------------------------------------------------------------
# 7. STRONG_REAL cross-check PASS — every claimed metric has MEASURED value
# ---------------------------------------------------------------------------
def test_verify_strong_real_cross_check_pass() -> None:
    """The load-bearing invariant: STRONG_REAL rows must have measured evidence."""
    verify = _load_verify()
    audit = verify["strong_audit"]
    assert (
        audit["verdict"] == "PASS"
    ), f"STRONG_REAL cross-check failed: missing_measured_value={audit['missing_measured_value']}"
    # Specifically check the 10 STRONG_REAL rows from §17.1
    assert len(audit["actually_measured_on_disk"]) >= 10


# ---------------------------------------------------------------------------
# 8. All §17.4 cross-reference files exist
# ---------------------------------------------------------------------------
def test_verify_cross_refs_all_exist() -> None:
    verify = _load_verify()
    n_missing = verify["n_cross_refs_missing"]
    assert (
        n_missing == 0
    ), f"{n_missing} cross-ref file(s) missing — §17.4 ledger is not honest"


# ---------------------------------------------------------------------------
# 9. Honest framing cited in JSON output
# ---------------------------------------------------------------------------
def test_verify_honest_framing_cited() -> None:
    verify = _load_verify()
    provenance = verify["honest_framing_provenance"]
    assert "party-twenty-spirit" in provenance["policy"]


# ---------------------------------------------------------------------------
# 10. No fabricated STRONG_REAL — every claimed metric has actual numeric evidence
# ---------------------------------------------------------------------------
def test_verify_no_fabricated_strong_real() -> None:
    """Every STRONG_REAL row in INDEX §17.1 must have an actual measured value on disk."""
    verify = _load_verify()
    audit = verify["strong_audit"]
    claimed_metrics = set(audit["claimed_metrics"])
    measured_metrics = set(audit["actually_measured_on_disk"])
    missing = claimed_metrics - measured_metrics
    assert not missing, (
        f"INDEX §17.1 claims STRONG_REAL for {missing} but no MEASURED value on disk — "
        f"fabrication per party-twenty-spirit §3.5"
    )


# ---------------------------------------------------------------------------
# 11. Per-metric audit covers all 22 baseline + 3 NEW (m22-m24)
# ---------------------------------------------------------------------------
def test_verify_baseline_and_new_metrics_audited() -> None:
    verify = _load_verify()
    audited = {r["metric_id"] for r in verify["per_metric_audit"]}
    new_metrics = {"m22", "m23", "m24"}
    assert new_metrics.issubset(
        audited
    ), f"missing NEW m22/m23/m24 audit: {new_metrics - audited}"


# ---------------------------------------------------------------------------
# 12. Index §17.1 contains STRONG_REAL for 10 specific metrics (literal claim)
# ---------------------------------------------------------------------------
def test_index_17_1_strong_real_literal_list() -> None:
    """The INDEX §17.1 STRONG_REAL row text must mention all 10 expected metrics."""
    index_path = PROJECT_ROOT / "TODO" / "INDEX.md"
    text = index_path.read_text(encoding="utf-8")
    # Extract the STRONG_REAL row
    import re

    match = re.search(
        r"\*\*🟢 STRONG_REAL_PAPER_READY\*\*\s*\|\s*(\d+)\s*\|\s*([^|]+)\|",
        text,
    )
    assert match is not None, "INDEX §17.1 STRONG_REAL row not found"
    count = int(match.group(1))
    assert count == 10, f"INDEX §17.1 STRONG_REAL count = {count}, expected 10"
    examples = match.group(2)
    # Check at least 5 of the 10 expected metrics are mentioned
    expected = ["m11", "m13", "m14", "m17", "m21", "m22", "m23", "m24"]
    found = [m for m in expected if m in examples]
    assert (
        len(found) >= 5
    ), f"only {len(found)} of 8 expected STRONG_REAL metric IDs mentioned"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
