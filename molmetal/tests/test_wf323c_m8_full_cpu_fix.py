"""Wf-323C unit tests — UFF pre-relax + Vina re-dock CPU smoke for m8_full lift.

Per Wf-319C + WEAK_TO_STRONG §2.6: 8 NEW tests verify the Wf-323C
pipeline that takes Wf-319B UFF-min Pt-pool conformers, applies UFF
pre-relax + Vina re-dock on the 1h36 receptor, and runs PoseBusters
dock mode on the docked output.

Honest-framing note (party-twenty-spirit §3.5):
Per WEAK_TO_STRONG §2.6 R18 §2.7 the projected delta was +0.10 to +0.15
above the Wf-315D 0/15 baseline. The MEASURED Wf-323C result is
m8_full_pass_rate = 0.0 (0/14); the test asserts the gate only
requires the rate be MEASURED (and within [0, 1]) — not that it is
above any specific threshold. This matches the HONEST_NEGATIVE_UFF_
INSUFFICIENT contract documented in the Wf-323C ship surface.

Test breakdown:
  3 tests — script existence + JSON-loadable + MEASURED rate in [0, 1]
  2 tests — pre-relaxed + Pt→Zn substitution pipeline records
  2 tests — pre-flight gate (gate_wf323c_m8_full_pass_rate_documented)
  1 test — gate count consistency (EXPECTED_GATE_COUNTS['all'] bumped to 57)

All CPU-only. No GPU launch per AGENTS.md §5.1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "molmetal" / "reports" / "wf323c_m8_full"
WF319B_POOL_JSON = ROOT / "molmetal" / "reports" / "wf319b_uff_min_pt_pool" / "pool.json"
WF246D_DIR = ROOT / "molmetal" / "reports" / "wf246d_m8_pb_dock"
METRICS_DIR = ROOT / "metrics" / "by_metric"
RESULT_JSON = REPORTS_DIR / "result.json"
METRIC_1H36 = METRICS_DIR / "steric_clash_pb_dock_1h36.json"
WF323C_SCRIPT = ROOT / "molmetal" / "scripts" / "wf323c_m8_full_cpu_fix.py"


def _load_result() -> dict:
    if not RESULT_JSON.exists():
        pytest.skip(
            f"result JSON not found at {RESULT_JSON}; run "
            f"wf323c_m8_full_cpu_fix.py --execute first"
        )
    with RESULT_JSON.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_metric() -> dict:
    if not METRIC_1H36.exists():
        pytest.skip(f"metric JSON not found: {METRIC_1H36}")
    with METRIC_1H36.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Test 1: Wf-323C script exists on disk
# ---------------------------------------------------------------------------
def test_wf323c_script_present():
    """The Wf-323C entry-point script must be on disk."""
    assert WF323C_SCRIPT.exists(), f"script missing: {WF323C_SCRIPT}"
    src = WF323C_SCRIPT.read_text(encoding="utf-8")
    # Sanity: contains the canonical docstring anchor + key fn name
    assert "Wf-323C" in src, "Wf-323C anchor missing from docstring"
    assert "_uff_prerelax_one" in src, "_uff_prerelax_one fn missing"
    assert "_vina_redock" in src, "_vina_redock fn missing"
    assert "Pt" in src, "Pt handling missing"


# ---------------------------------------------------------------------------
# Test 2: result.json is JSON-loadable + has the required top-level keys
# ---------------------------------------------------------------------------
def test_wf323c_result_required_keys():
    """The result JSON must carry the canonical top-level keys."""
    result = _load_result()
    required = [
        "workflow",
        "engine",
        "input_pool",
        "pocket_id",
        "vina_config",
        "honest_framing",
        "n_poses_pre_relaxed",
        "n_pt_zn_substituted",
        "n_poses_total",
        "m8_protein_aware_pass_rate",
        "m8_full_pass_rate",
        "wall_seconds_total",
        "results",
    ]
    for k in required:
        assert k in result, f"missing required top-level key: {k}"


# ---------------------------------------------------------------------------
# Test 3: m8_full_pass_rate + m8_protein_aware_pass_rate are valid floats
# ---------------------------------------------------------------------------
def test_wf323c_pass_rates_in_unit_interval():
    """The two m8 rates must be floats in [0.0, 1.0]."""
    result = _load_result()
    full = result.get("m8_full_pass_rate")
    pa = result.get("m8_protein_aware_pass_rate")
    assert isinstance(full, float), f"m8_full_pass_rate type={type(full).__name__}"
    assert isinstance(pa, float), f"m8_protein_aware_pass_rate type={type(pa).__name__}"
    assert 0.0 <= full <= 1.0, f"m8_full_pass_rate={full} not in [0, 1]"
    assert 0.0 <= pa <= 1.0, f"m8_protein_aware_pass_rate={pa} not in [0, 1]"
    # Per Wf-319C: rates are MEASURED — both must be set, never None.
    assert full is not None, "m8_full_pass_rate should not be None"
    assert pa is not None, "m8_protein_aware_pass_rate should not be None"


# ---------------------------------------------------------------------------
# Test 4: Per-conformer records — at least one Vina OK with Pt→Zn substitution
# ---------------------------------------------------------------------------
def test_wf323c_pt_zn_substitution_recorded():
    """At least one per-conformer record must carry pt_zn_substituted=True
    with a valid Vina kcal/mol (the Wf-323C pipeline ran end-to-end)."""
    result = _load_result()
    records = result.get("results", [])
    assert len(records) > 0, "no per-conformer records"
    ok_records = [
        r for r in records
        if isinstance(r.get("vina_status"), str)
        and r["vina_status"].startswith("OK")
    ]
    assert len(ok_records) > 0, (
        f"no OK Vina records (statuses: "
        f"{[r.get('vina_status') for r in records]})"
    )
    # At least one record must have pt_zn_substituted=True + non-None kcal_mol
    zn_records = [
        r for r in ok_records
        if r.get("pt_zn_substituted") is True
        and r.get("kcal_mol") is not None
    ]
    assert len(zn_records) > 0, (
        "no Pt→Zn substituted records with kcal_mol — pipeline did not run"
    )


# ---------------------------------------------------------------------------
# Test 5: n_poses_pre_relaxed is non-zero + matches HONEST_NEGATIVE framing
# ---------------------------------------------------------------------------
def test_wf323c_n_poses_pre_relaxed_matches_records():
    """n_poses_pre_relaxed (top-level) must equal the count of OK Vina
    records in the results list."""
    result = _load_result()
    top_n = result.get("n_poses_pre_relaxed", 0)
    records = result.get("results", [])
    ok_n = sum(
        1 for r in records
        if isinstance(r.get("vina_status"), str)
        and r["vina_status"].startswith("OK")
    )
    assert top_n == ok_n, (
        f"n_poses_pre_relaxed={top_n} != OK records in results={ok_n}"
    )
    # Must be > 0 (else the pipeline didn't run).
    assert top_n > 0, "n_poses_pre_relaxed == 0; pipeline did not execute"


# ---------------------------------------------------------------------------
# Test 6: HONEST_NEGATIVE tags present when m8_full stays below 5%
# ---------------------------------------------------------------------------
def test_wf323c_honest_negative_tags_consistent():
    """If m8_full_pass_rate < 0.05, HONEST_NEGATIVE_UFF_INSUFFICIENT
    must be set with a non-empty reason. This pins the honest-framing
    mandate from party-twenty-spirit §3.5."""
    result = _load_result()
    full = result.get("m8_full_pass_rate")
    assert full is not None
    if full < 0.05:
        flag = result.get("HONEST_NEGATIVE_UFF_INSUFFICIENT")
        reason = result.get("honest_negative_reason", "")
        assert flag is True, (
            f"m8_full={full} < 0.05 but "
            f"HONEST_NEGATIVE_UFF_INSUFFICIENT={flag} (expected True)"
        )
        assert isinstance(reason, str) and len(reason) >= 80, (
            f"honest_negative_reason too short or wrong type: {reason[:80]!r}"
        )
        # Pt→Zn substitution tag should also be present (per Wf-323C).
        if result.get("n_pt_zn_substituted", 0) > 0:
            pt_flag = result.get("HONEST_NEGATIVE_VINA_PT_REJECTED")
            pt_reason = result.get("honest_negative_vina_pt_reason", "")
            assert pt_flag is True, (
                f"Pt substitution ran but HONEST_NEGATIVE_VINA_PT_REJECTED={pt_flag}"
            )
            assert isinstance(pt_reason, str) and len(pt_reason) >= 80, (
                f"vina_pt_reason too short: {pt_reason[:80]!r}"
            )


# ---------------------------------------------------------------------------
# Test 7: Pre-flight gate (gate_wf323c_m8_full_pass_rate_documented) PASS
# ---------------------------------------------------------------------------
def test_wf323c_preflight_gate_pass():
    """The ``p3_m8_full_pass_rate_documented`` pre-flight gate must PASS.

    PASS conditions:
      (1) result.json exists + is JSON-loadable + has the canonical
          m8_full_pass_rate + m8_protein_aware_pass_rate keys.
      (2) The steric_clash_pb_dock_1h36.json metric is MEASURED_SMOKE
          and references the Wf-323C pipeline + UFF pre-relax +
          Pt→Zn substitution history.
      (3) Wf-319B pool JSON exists with ≥80 conformers.
    """
    from molmetal.scripts.pre_flight_check import (
        gate_wf323c_m8_full_pass_rate_documented,
        GateResult,
    )

    result = gate_wf323c_m8_full_pass_rate_documented(
        name="p3_m8_full_pass_rate_documented",
        priority="P3",
        pipeline="wf323c",
        result_json=RESULT_JSON,
        metric_json=METRIC_1H36,
        pool_json=WF319B_POOL_JSON,
    )
    assert isinstance(result, GateResult), (
        f"gate returned wrong type: {type(result).__name__}"
    )
    assert result.status in ("PASS", "FAIL"), (
        f"unexpected status {result.status!r}"
    )
    if RESULT_JSON.exists() and METRIC_1H36.exists() and WF319B_POOL_JSON.exists():
        # The gate is intentionally strict — if the 3 artefacts ship
        # together (which they do post-Wf-323C run), it MUST pass.
        assert result.status == "PASS", (
            f"3 artefacts ship but gate FAILED: {result.evidence}"
        )


# ---------------------------------------------------------------------------
# Test 8: EXPECTED_GATE_COUNTS['all'] bumped to 57 (was 56 pre-Wf-323C)
# ---------------------------------------------------------------------------
def test_wf323c_expected_gate_count_bumped():
    """Wf-323C adds 1 gate (``p3_m8_full_pass_rate_documented``) so the
    'all'-pipeline expected gate count MUST be ≥ 57 (was 56 pre-Wf-323C).

    This test pins the audit gate's expected count so a future
    inadvertent edit to EXPECTED_GATE_COUNTS doesn't silently break
    the Wf-321D gate-count consistency audit.
    """
    from molmetal.scripts.pre_flight_check import EXPECTED_GATE_COUNTS
    expected = EXPECTED_GATE_COUNTS.get("all")
    assert expected is not None and expected >= 57, (
        f"EXPECTED_GATE_COUNTS['all']={expected}; "
        f"Wf-323C expects ≥57 (was 56 pre-Wf-323C). If you're intentionally "
        f"adding/removing a gate, update this test + EXPECTED_GATE_COUNTS in "
        f"sync."
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))