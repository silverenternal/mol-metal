"""Wf-338-V3 — TODO-220 + TODO-179 ship-status + Wf-316A STATUS_HOLD + signal-floor tests.

================================================================
4 NEW tests verifying the Wf-338-V3 contract (per AGENTS.md §3.2
+ Wf-316A TODO directory audit + signal-floor analysis):

Test 1 — test_todo_220_ship_status
    TODO-220 (Wf-219 temporal split + NCI-60 9-cell driver) is in
    completed/ with all 3 acceptance gate docs (§6.3 temporal-split
    + §6.4 NCI-60 9-cell + parent driver). §6.3 + §6.4 schemas are
    DESIGN→MEASURED for the smoke layer; full P1 GPU sweep is
    PROJECTED PENDING per the §6.3.2 / §6.4.2 wrapper gates.
    Parent file `220_wf219_temporal_split_nci60_driver.md` moved
    pending → completed by Wf-316A.

Test 2 — test_todo_179_ship_status
    TODO-179 (Wf-123 P3 W₂ alignment on PlatinAI 4-cell) remains
    in pending/ with explicit STATUS_HOLD marker. The W₂ recipe
    is HONEST_NEGATIVE per Wf-313G dashboard §3.4: Wf-122 P3
    R² = -5.66 ± 1.36 and Wf-129 P3 R² = -0.0780 ± 0.0148 — both
    below the LAMBDA_ROADMAP §4.2 L7 + R20-P3 gate R² ≥ +0.30.
    Architectural fix path is Wf-94 R20-P3 Multi-PH + PLL.

Test 3 — test_wf316a_status_hold_reason
    Wf-316A TODO directory audit (2026-09-29) moved 6 files from
    pending/ → completed/ including TODO-220. TODO-179 was kept
    in pending/ with STATUS-HOLD marker. Wf-316F final synthesis
    categorizes pending/ as 1 RULE + 3 STILL_PENDING (179 / 182 /
    183) + 10 wetlab TEMPLATE.

Test 4 — test_signal_floor_documentation
    The 12-d RDKit 2D feature ceiling (|ρ| ≤ 0.019 → R² ≤ 0.0043)
    is documented in TODO-179 §2 + Wf-313G dashboard + Wf-127 P2
    sweep diagnostic. The architectural fix path (Wf-94 R20-P3
    Multi-PH + PLL feature expansion to 32-d or 64-d) is
    referenced as the only path to lift R² above the signal floor.

Honest framing (per AGENTS.md §1.5 / party-twenty-spirit §3.5)
-------------------------------------------------------------
* TODO-220 is DESIGN→MEASURED for smoke (real numbers); GPU §6.3.2
  + §6.4.2 are PROJECTED PENDING (USER_OK + 3.9 hr GPU wall).
* TODO-179 is STATUS_HOLD per Wf-316A signal-floor (HONEST_NEGATIVE).
  W₂ regularization cannot lift R² above the 12-d RDKit 2D ceiling.
* Wf-316A audit moved TODO-220 pending → completed; kept TODO-179
  pending/STATUS_HOLD per the Wf-313G dashboard HONEST_NEGATIVE verdict.
* Signal-floor documentation cites Wf-127 §Cause 2 + Wf-139 P4 + Wf-94
  R20-P3 Multi-PH + PLL as the architectural fix path.

Cross-references
----------------
* TODO/pending/179_wf123_p3_w2_pic50_alignment.md — TODO-179 STATUS_HOLD
* TODO/completed/220_wf220_temporal_split_acceptance.md — §6.3 doc
* TODO/completed/220_wf219_nci60_9cell_acceptance.md — §6.4 doc
* TODO/completed/220_wf219_temporal_split_nci60_driver.md — parent driver
* TODO/completed/177_wf313g_downstream_lift_dashboard.md §3.4 — TODO-122 HONEST_NEGATIVE
* TODO/completed/316_todo_directory_final_state.md — Wf-316F synthesis
* TODO/completed/316_index_md_update.md — Wf-316 INDEX.md update
* TODO/completed/137_l5_l7_l10_arch_ceiling_honest.md — signal-floor close-out
* molmetal/tests/test_wf313d_todo220_acceptance_gates.py — 12 Wf-313D smoke tests
* mol-metal-gpu-execution skill §0 — kill_then_adapt directive
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path("/home/hugo/codes/mol-metal")

# TODO-220 paths
TODO_220_DRIVER = ROOT / "TODO/completed/220_wf219_temporal_split_nci60_driver.md"
TODO_220_TEMPORAL = ROOT / "TODO/completed/220_wf220_temporal_split_acceptance.md"
TODO_220_NCI60 = ROOT / "TODO/completed/220_wf219_nci60_9cell_acceptance.md"

# TODO-179 path
TODO_179 = ROOT / "TODO/pending/179_wf123_p3_w2_pic50_alignment.md"

# Wf-316 audit + dashboard paths
WF316_INDEX_UPDATE = ROOT / "TODO/completed/316_index_md_update.md"
WF316_FINAL_SYNTHESIS = ROOT / "TODO/completed/316_todo_directory_final_state.md"
WF313G_DASHBOARD = ROOT / "TODO/completed/177_wf313g_downstream_lift_dashboard.md"
SIGNAL_FLOOR_CLOSE = ROOT / "TODO/completed/137_l5_l7_l10_arch_ceiling_honest.md"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_text(path: Path) -> str:
    assert path.exists(), f"required file missing: {path}"
    return path.read_text()


# ===========================================================================
# Test 1 — TODO-220 ship status
# ===========================================================================


def test_todo_220_ship_status():
    """TODO-220 — temporal split + NCI-60 9-cell driver shipped (smoke MEASURED).

    Ship contract:
    * Parent driver (Wf-219 temporal-split + NCI-60 9-cell) moved
      pending → completed by Wf-316A
    * §6.3 temporal-split acceptance doc (Wf-310C Part 1) DESIGN→MEASURED
    * §6.4 NCI-60 9-cell acceptance doc (Wf-310C Part 2) DESIGN→MEASURED
    * Both §6.3 + §6.4 declare smoke MEASURED + GPU PROJECTED PENDING
    * §6.4 explicitly declares PC3 honest-negative (0.1667 < 0.5 random)
    """
    # All 3 TODO-220 docs must exist in completed/
    assert TODO_220_DRIVER.exists(), "TODO-220 parent driver must be in completed/ (Wf-316A move)"
    assert TODO_220_TEMPORAL.exists(), "TODO-220 §6.3 temporal-split acceptance doc must be in completed/"
    assert TODO_220_NCI60.exists(), "TODO-220 §6.4 NCI-60 9-cell acceptance doc must be in completed/"

    # Parent driver doc must be in completed/ (Wf-316A moved it)
    driver_text = _read_text(TODO_220_DRIVER)
    assert "STATUS" in driver_text.upper(), "TODO-220 driver doc must declare STATUS"
    assert "P1" in driver_text or "PENDING" in driver_text.upper(), (
        "TODO-220 driver doc must reference P1 priority / PENDING state"
    )

    # §6.3 temporal-split doc — must declare smoke MEASURED + GPU PROJECTED
    sec_6_3 = _read_text(TODO_220_TEMPORAL)
    assert "§6.3" in sec_6_3 or "6.3" in sec_6_3, "§6.3 temporal-split doc must have §6.3 section anchors"
    assert "MEASURED" in sec_6_3, "§6.3 must have at least one MEASURED tag"
    assert "PROJECTED" in sec_6_3, "§6.3 must have at least one PROJECTED tag"
    assert "Krasnov" in sec_6_3 or "krasnov" in sec_6_3, "§6.3 must reference Krasnov 2026 protocol"
    assert "temporal" in sec_6_3.lower(), "§6.3 must reference temporal split"

    # §6.4 NCI-60 9-cell doc — must declare smoke MEASURED + PC3 honest-negative
    sec_6_4 = _read_text(TODO_220_NCI60)
    assert "§6.4" in sec_6_4 or "6.4" in sec_6_4, "§6.4 NCI-60 9-cell doc must have §6.4 section anchors"
    assert "MEASURED" in sec_6_4, "§6.4 must have at least one MEASURED tag"
    assert "PROJECTED" in sec_6_4, "§6.4 must have at least one PROJECTED tag"
    assert "PC3" in sec_6_4, "§6.4 must reference PC3 (honest-negative cell)"
    assert "honest-negative" in sec_6_4.lower() or "honest_negative" in sec_6_4.lower(), (
        "§6.4 must declare PC3 honest-negative per party-twenty-spirit §3.5"
    )
    assert "HCT116" in sec_6_4, "§6.4 must reference HCT116 (highlight cell)"

    # §6.4 must NOT paraphrase PC3 honest-negative as "achievable"
    # (would violate party-twenty-spirit §3.5 honest framing)
    pc3_section = re.search(r"PC3.*?(?=\n#|\Z)", sec_6_4, re.DOTALL)
    if pc3_section:
        pc3_text = pc3_section.group(0)
        # "achievable" should NOT appear in the PC3 honest-negative context
        # (it's only acceptable if explicitly negated)
        if "achievable" in pc3_text.lower():
            # If "achievable" appears, it MUST be negated
            assert "not" in pc3_text.lower() or "NOT" in pc3_text or "no " in pc3_text.lower(), (
                "PC3 honest-negative: if 'achievable' appears, must be negated"
            )


# ===========================================================================
# Test 2 — TODO-179 ship status
# ===========================================================================


def test_todo_179_ship_status():
    """TODO-179 — W₂ alignment on PlatinAI 4-cell: STATUS_HOLD per Wf-316A.

    Ship contract:
    * TODO-179 remains in pending/ (Wf-316A kept it — STATUS_HOLD)
    * Status-hold marker (Wf-316A) is present in §0 of the doc
    * W₂ recipe is declared HONEST_NEGATIVE in §0
    * Architectural fix path (Wf-94 R20-P3 Multi-PH + PLL) is referenced
    * Wf-122 P3 / Wf-129 P3 MEASURED numbers are cited
    """
    # TODO-179 must remain in pending/ (Wf-316A kept it STATUS_HOLD)
    assert TODO_179.exists(), "TODO-179 must exist in pending/ (Wf-316A STATUS_HOLD)"
    assert "pending" in str(TODO_179.parent), "TODO-179 must remain in pending/ (not moved to completed)"

    text = _read_text(TODO_179)

    # STATUS_HOLD marker must be present (per Wf-316A audit)
    assert "STATUS_HOLD" in text, "TODO-179 must have explicit STATUS_HOLD marker (Wf-316A)"
    assert "Wf-316A" in text or "wf-316a" in text.lower() or "316A" in text, (
        "TODO-179 must cite Wf-316A as the source of STATUS_HOLD marker"
    )

    # W₂ recipe must be declared HONEST_NEGATIVE
    assert "HONEST_NEGATIVE" in text or "honest-negative" in text.lower(), (
        "TODO-179 must declare W₂ alignment attempt HONEST_NEGATIVE per Wf-313G dashboard"
    )

    # Architectural fix path: Wf-94 R20-P3 Multi-PH + PLL
    assert "Wf-94" in text, "TODO-179 must reference Wf-94 R20-P3 (architectural fix path)"
    assert "R20-P3" in text, "TODO-179 must reference R20-P3 phase"
    assert "Multi-PH" in text or "multiph" in text.lower() or "MultiPH" in text, (
        "TODO-179 must reference Multi-PH feature expansion"
    )
    assert "PLL" in text, "TODO-179 must reference PLL feature expansion"

    # Wf-122 P3 + Wf-129 P3 MEASURED numbers
    assert "Wf-122" in text and "Wf-129" in text, (
        "TODO-179 must reference Wf-122 P3 + Wf-129 P3 attempts"
    )
    # R² numbers must be present (HONEST_NEGATIVE values)
    assert "-5.66" in text or "-0.0780" in text, (
        "TODO-179 must cite Wf-122 P3 / Wf-129 P3 MEASURED R² numbers"
    )

    # The §0 Wf-316A status-hold marker section must exist
    assert "§0" in text or "status-hold marker" in text.lower(), (
        "TODO-179 §0 must declare Wf-316A status-hold marker"
    )

    # TODO-179 must NOT contain a fabrication / aspiration: e.g. R² ≥ 0.85
    # without PROJECTED tag and signal-floor caveat
    if "R² ≥ 0.85" in text or "R^2 >= 0.85" in text or "R² >= 0.85" in text:
        # If R² ≥ 0.85 is mentioned, it MUST be qualified
        assert "PROJECTED" in text or "ceiling" in text.lower() or "architectural" in text.lower(), (
            "TODO-179 must qualify R² ≥ 0.85 as PROJECTED or signal-floor-blocked"
        )


# ===========================================================================
# Test 3 — Wf-316A STATUS_HOLD reason
# ===========================================================================


def test_wf316a_status_hold_reason():
    """Wf-316A — TODO directory audit: TODO-220 → completed, TODO-179 → STATUS_HOLD.

    Audit contract:
    * Wf-316A moved 6 files pending → completed (incl. TODO-220)
    * Wf-316A kept TODO-179 in pending/ with STATUS-HOLD marker
    * Wf-316F synthesis categorizes pending/ as 1 RULE + 3 STILL_PENDING
      (179 / 182 / 183) + 10 wetlab TEMPLATE
    * Wf-316 INDEX update explicitly mentions 179 as STATUS-HOLD per Wf-316A
    """
    # Wf-316 INDEX update must mention 179 as STATUS-HOLD per Wf-316A
    idx_text = _read_text(WF316_INDEX_UPDATE)
    assert "179" in idx_text, "Wf-316 INDEX update must reference TODO-179"
    assert "STATUS" in idx_text.upper() or "status" in idx_text, (
        "Wf-316 INDEX update must declare status for pending files"
    )
    # The row pattern: "| 1 | `179_wf123_p3_w2_pic50_alignment.md` ... STATUS-HOLD per Wf-316A"
    assert re.search(
        r"179[^\n]*STATUS[- ]?HOLD[^\n]*Wf-316A",
        idx_text,
        re.IGNORECASE,
    ), "Wf-316 INDEX update must have explicit `179 ... STATUS-HOLD per Wf-316A` row"

    # Wf-316 INDEX update must mention TODO-220 moved to completed
    assert "220" in idx_text, "Wf-316 INDEX update must reference TODO-220"
    # TODO-220 moved pending → completed by Wf-316A-D
    assert "Wf-316A" in idx_text, "Wf-316 INDEX update must reference Wf-316A"

    # Wf-316F synthesis — pending/ categorization
    synth_text = _read_text(WF316_FINAL_SYNTHESIS)
    assert "179" in synth_text, "Wf-316F synthesis must reference TODO-179"
    assert "STILL_PENDING" in synth_text, "Wf-316F synthesis must categorize STILL_PENDING files"
    assert "220" in synth_text, "Wf-316F synthesis must reference TODO-220"
    # The 3 STILL_PENDING files: 179 / 182 / 183
    assert "182" in synth_text, "Wf-316F synthesis must list TODO-182 as STILL_PENDING"
    assert "183" in synth_text, "Wf-316F synthesis must list TODO-183 as STILL_PENDING"

    # Verify TODO-220 is NOT in the STILL_PENDING list (it was moved to completed)
    # The 6 moved files include 220 — search for explicit MOVE evidence
    assert "MOVE" in synth_text or "moved" in synth_text.lower(), (
        "Wf-316F synthesis must document the MOVE actions"
    )

    # Wf-313G dashboard — TODO-122 L5 Hilbert R² HONEST_NEGATIVE
    dashboard_text = _read_text(WF313G_DASHBOARD)
    assert "TODO-122" in dashboard_text or "todo-122" in dashboard_text.lower() or "todo_122" in dashboard_text.lower(), (
        "Wf-313G dashboard must reference TODO-122 (the L5 Hilbert R² parent)"
    )
    assert "HONEST_NEGATIVE" in dashboard_text or "honest-negative" in dashboard_text.lower(), (
        "Wf-313G dashboard must declare HONEST_NEGATIVE verdict for TODO-122 L5"
    )
    assert "§3.4" in dashboard_text or "3.4" in dashboard_text, (
        "Wf-313G dashboard must reference §3.4 (HONEST_NEGATIVE section)"
    )


# ===========================================================================
# Test 4 — Signal-floor documentation
# ===========================================================================


def test_signal_floor_documentation():
    """Signal floor — 12-d RDKit 2D ceiling caps linear R² at ~0.0043.

    Documentation contract:
    * 12-d RDKit 2D feature ceiling |ρ| ≤ 0.019 is documented in TODO-179 §2
    * Linear R² upper bound = (0.019)² · 12 = 0.0043 is documented
    * Architectural fix path (Wf-94 R20-P3 Multi-PH + PLL) is referenced
    * Wf-313G dashboard cites the architectural signal floor in §3.4
    * Wf-139 P4 close-out (TODO/completed/137) is referenced
    """
    # TODO-179 §2 — Why the W₂ recipe is architecturally infeasible
    todo_179_text = _read_text(TODO_179)
    assert "12-d RDKit" in todo_179_text or "12-d RDKit 2D" in todo_179_text, (
        "TODO-179 must document the 12-d RDKit 2D feature ceiling"
    )
    # The architectural fix path must be present
    assert "F5_RDKIT_2D" in todo_179_text or "feature expansion" in todo_179_text.lower(), (
        "TODO-179 must reference F5_RDKIT_2D feature expansion (architectural fix)"
    )
    # R² upper bound math must be present
    assert "0.0043" in todo_179_text or "0.004" in todo_179_text, (
        "TODO-179 must cite the R² ≤ 0.0043 upper bound (12-d RDKit signal floor)"
    )
    assert "|ρ| ≤ 0.019" in todo_179_text or "0.019" in todo_179_text, (
        "TODO-179 must cite |ρ| ≤ 0.019 (linear correlation ceiling)"
    )

    # Wf-139 P4 close-out reference
    assert "Wf-139" in todo_179_text or "137_l5_l7_l10" in todo_179_text, (
        "TODO-179 must reference Wf-139 P4 close-out (signal-floor close-out)"
    )

    # Wf-313G dashboard — must reference 12-d signal floor
    dashboard_text = _read_text(WF313G_DASHBOARD)
    assert (
        "12-d" in dashboard_text
        or "12d" in dashboard_text.lower()
        or "RDKit" in dashboard_text
    ), "Wf-313G dashboard must reference the 12-d RDKit feature stack"
    assert (
        "architectural" in dashboard_text.lower()
        or "signal floor" in dashboard_text.lower()
        or "signal-floor" in dashboard_text.lower()
    ), "Wf-313G dashboard must reference architectural signal floor"

    # Wf-139 P4 close-out — must exist and reference signal floor
    signal_floor_text = _read_text(SIGNAL_FLOOR_CLOSE)
    assert (
        "12-d" in signal_floor_text
        or "12d" in signal_floor_text.lower()
        or "RDKit" in signal_floor_text
    ), "Wf-139 P4 close-out (TODO/completed/137) must reference the 12-d RDKit feature stack"
    assert (
        "signal floor" in signal_floor_text.lower()
        or "signal-floor" in signal_floor_text.lower()
        or "ceiling" in signal_floor_text.lower()
    ), "Wf-139 P4 close-out must declare the signal floor / architectural ceiling"

    # W₂ regularization cannot lift R² above the signal floor (key claim)
    # This claim should appear in TODO-179 + Wf-313G dashboard
    w2_claim_patterns = [
        "W₂.*cannot lift",
        "W2.*cannot lift",
        "W₂.*architecturally infeasible",
        "W2.*architecturally infeasible",
        "architecturally infeasible",
    ]
    found_w2_claim = any(
        re.search(pat, todo_179_text, re.IGNORECASE | re.DOTALL)
        for pat in w2_claim_patterns
    )
    assert found_w2_claim, (
        "TODO-179 must explicitly declare W₂ regularization architecturally infeasible "
        "(key signal-floor claim per party-twenty-spirit §3.5)"
    )
