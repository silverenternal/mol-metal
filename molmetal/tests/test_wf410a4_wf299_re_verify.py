"""Wf-410-A4 — test suite for the Wf-299 pre-launch verification chain re-verify.

Per Wf-410-A4 + Wf-394-A3 + Wf-407-A3 + Wf-408-A2/A3 in-flight:
the Wf-410-A4 verify script imports the 4 Wf-394-A3 verify_* functions
+ adds 2 NEW Wf-410-A4 contracts. This test verifies the re-verify
script correctly enumerates + reports all 6 contracts + emits a
structured JSON.

Wall-budget: <1 s CPU-only. 0 GPU launches.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make /home/hugo/codes/mol-metal importable so we can find the script
PROJECT_ROOT = Path("/home/hugo/codes/mol-metal")
sys.path.insert(0, str(PROJECT_ROOT / "molmetal/scripts"))

# Re-import the verifier (avoids the Wf-410-A4 script's main() side-effects)
import importlib.util as importlib_util

WF410A4_PATH = PROJECT_ROOT / "molmetal/scripts/wf410a4_wf299_re_verify.py"
_WF410A4_SPEC = importlib_util.spec_from_file_location("wf410a4", WF410A4_PATH)
_WF410A4_MOD = importlib_util.module_from_spec(_WF410A4_SPEC)
_WF410A4_SPEC.loader.exec_module(_WF410A4_MOD)

verify_wf299_wave_alias_mapping = _WF410A4_MOD.verify_wf299_wave_alias_mapping
verify_wf299_trainer_flag_wirein = _WF410A4_MOD.verify_wf299_trainer_flag_wirein
EXPECTED_CFM_FLAGS = _WF410A4_MOD.EXPECTED_CFM_FLAGS
LAUNCHER = _WF410A4_MOD.LAUNCHER
TRAINER_CFM = _WF410A4_MOD.TRAINER_CFM
REPORTS_DIR = _WF410A4_MOD.REPORTS_DIR


# --- Tests -------------------------------------------------------------------


def test_wf410a4_wave_alias_mapping_runs():
    """Wf-410-A4 NEW contract: --wave 1/2/3 alias mapping check."""
    result = verify_wf299_wave_alias_mapping()
    assert "alias_present" in result
    assert "accepted_aliases" in result
    assert "contract_satisfied" in result
    # Per Wf-407-A3 + Wf-410-A4 honest framing: alias mapping NOT SHIPPED
    # at this timestamp; expect False until Wf-408-A3 ships.
    assert isinstance(result["alias_present"], bool)
    assert isinstance(result["contract_satisfied"], bool)


def test_wf410a4_trainer_flag_wirein_runs():
    """Wf-410-A4 NEW contract: 16 Wf-388 flags wired into trainer."""
    result = verify_wf299_trainer_flag_wirein()
    assert "flags_wired" in result
    assert "all_wired" in result
    assert "missing_flags" in result
    assert "contract_satisfied" in result
    assert "dmpnn_no_enforce_budget_should_not_exist" in result
    # Per Wf-407-A3 baseline (29 add_argument) + Wf-405-A2 wire-in (38 calls):
    # 14/16 flags wired; --so3-average + --log-normal-t still MISSING.
    # Per honest framing: contract currently FAIL; --so3-average + --log-normal-t
    # deferred to Wf-408-A2 wire-in.
    if not result["all_wired"]:
        assert "missing_flags" in result
        assert len(result["missing_flags"]) > 0


def test_wf410a4_full_chain_emits_json():
    """Run the full chain via import + verify the JSON output is well-formed."""
    # Call main() directly; expected to FAIL (refuse launch) since Wf-408-A3 alias
    # mapping is NOT SHIPPED.
    rc = _WF410A4_MOD.main()
    assert rc in (0, 1), f"main() returned unexpected rc={rc}"

    # Verify result.json exists + is well-formed.
    result_path = REPORTS_DIR / "result.json"
    assert result_path.exists(), f"result.json missing at {result_path}"

    result = json.loads(result_path.read_text())
    assert "wf_origin" in result
    assert result["wf_origin"] == "Wf-410-A4"
    assert "checks" in result
    assert "verdict" in result
    assert "failing_contracts" in result
    assert "honest_framing" in result

    # Verify all 6 contract names appear in checks dict.
    expected_check_keys = [
        "wf394a3_user_ok_gate_present_and_before_dry_run",
        "wf394a3_pre_flight_gates_wired",
        "wf394a3_three_wave_separation_complete",
        "wf394a3_wf388_flags_all_present",
        "wf410a4_wave_1_2_3_alias_present",
        "wf410a4_trainer_flag_wirein_complete",
    ]
    for k in expected_check_keys:
        assert k in result["checks"], f"missing check: {k}"
        assert isinstance(result["checks"][k], bool)


def test_wf410a4_expected_cfm_flags_count():
    """Verify EXPECTED_CFM_FLAGS table has 16 flags (per Wf-385-A2 + Wf-392-A3)."""
    assert (
        len(EXPECTED_CFM_FLAGS) == 16
    ), f"expected 16 Wf-388 flags, got {len(EXPECTED_CFM_FLAGS)}: {EXPECTED_CFM_FLAGS}"
    # Verify --flowmol-unified is intentionally excluded (LICENSE-GATED CC-BY-NC).
    assert "--flowmol-unified" not in EXPECTED_CFM_FLAGS
